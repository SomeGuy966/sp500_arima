from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500_arima.strategy import (
    StrategyConfig,
    compare_strategies,
    cost_sensitivity,
    evaluate_model_signal,
    max_drawdown,
    performance,
    performance_by_era,
    positions_from_forecast,
    strategy_returns,
)


def test_strategy_config_validation() -> None:
    with pytest.raises(ValueError, match="cost_bps"):
        StrategyConfig(cost_bps=-1)
    with pytest.raises(ValueError, match="threshold"):
        StrategyConfig(threshold=-0.1)


def test_positions_modes_and_threshold() -> None:
    idx = pd.bdate_range("2020-01-01", periods=4)
    pred = pd.Series([0.002, -0.001, 0.0, 0.0004], index=idx)
    ls = positions_from_forecast(pred, StrategyConfig(mode="long_short"))
    assert ls.tolist() == [1.0, -1.0, 0.0, 1.0]
    lf = positions_from_forecast(pred, StrategyConfig(mode="long_flat"))
    assert lf.tolist() == [1.0, 0.0, 0.0, 1.0]
    thr = positions_from_forecast(pred, StrategyConfig(threshold=0.0005))
    assert thr.tolist() == [1.0, -1.0, 0.0, 0.0]


def test_strategy_returns_charges_turnover() -> None:
    idx = pd.bdate_range("2020-01-01", periods=4)
    realised = pd.Series([0.01, -0.02, 0.03, 0.01], index=idx)
    positions = pd.Series([1.0, 1.0, -1.0, 0.0], index=idx)
    net = strategy_returns(realised, positions, cost_bps=10.0)
    # turnover: 1 (enter), 0, 2 (flip), 1 (exit) at 10 bps each
    expected = [0.01 - 0.001, -0.02, -0.03 - 0.002, 0.0 - 0.001]
    np.testing.assert_allclose(net.to_numpy(), expected)


def test_max_drawdown() -> None:
    assert max_drawdown(np.array([1.0, 1.2, 0.9, 1.1, 0.6])) == pytest.approx(0.6 / 1.2 - 1)
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0


def test_performance_constant_return() -> None:
    idx = pd.bdate_range("2020-01-01", periods=252)
    stats = performance(pd.Series(0.001, index=idx), "flat")
    assert stats.n_days == 252
    assert stats.total_return == pytest.approx(1.001**252 - 1)
    assert stats.annual_return == pytest.approx(1.001**252 - 1)
    assert stats.max_drawdown == 0.0
    assert stats.hit_rate == 1.0
    assert stats.time_in_market == 1.0
    assert np.isnan(stats.sharpe)  # zero volatility


def test_evaluate_model_signal_perfect_foresight_is_profitable() -> None:
    """A signal that always knows tomorrow's direction must beat buy-and-hold before costs."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = 100 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
    previous = np.concatenate([[prices[0]], prices[:-1]])
    frame = pd.DataFrame(
        {
            "model": "oracle",
            "date": idx,
            "actual": prices,
            "previous": previous,
            "one_step": prices,  # forecast equals realised
        }
    )
    stats, net = evaluate_model_signal(frame, StrategyConfig(cost_bps=0.0), "oracle")
    assert stats.hit_rate > 0.99
    assert stats.total_return > (prices[-1] / previous[0] - 1)
    assert len(net) == 300


def test_compare_strategies_shapes() -> None:
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=120)
    prices = 100 * np.exp(np.cumsum(rng.standard_normal(120) * 0.01))
    previous = np.concatenate([[prices[0]], prices[:-1]])
    rows = []
    for model, noise in (("arima", 0.01), ("drift", 0.0)):
        rows.append(
            pd.DataFrame(
                {
                    "model": model,
                    "date": idx,
                    "actual": prices,
                    "previous": previous,
                    "one_step": previous * np.exp(0.0003 + noise * rng.standard_normal(120)),
                }
            )
        )
    stats, equity = compare_strategies(pd.concat(rows), StrategyConfig(cost_bps=5.0))
    assert list(stats.index) == [
        "Buy & hold",
        "arima signal (long/short)",
        "drift signal (long/short)",
    ]
    assert equity.shape == (120, 3)
    assert stats.loc["Buy & hold", "turnover"] == 0.0
    assert stats.loc["drift signal (long/short)", "time_in_market"] == 1.0

    table = cost_sensitivity(pd.concat(rows), costs_bps=(0.0, 10.0))
    assert list(table.index) == list(stats.index)
    assert list(table["sharpe"].columns) == [0.0, 10.0]
    # costs can only hurt a signal that trades, and cannot touch buy-and-hold
    arima = "arima signal (long/short)"
    assert table.loc[arima, ("sharpe", 10.0)] < table.loc[arima, ("sharpe", 0.0)]
    assert table.loc["Buy & hold", ("sharpe", 10.0)] == table.loc["Buy & hold", ("sharpe", 0.0)]


def test_performance_by_era_splits_and_reports_autocorrelation() -> None:
    rng = np.random.default_rng(5)
    n = 900
    idx = pd.bdate_range("2018-01-01", periods=n)
    r = rng.standard_normal(n) * 0.01
    prices = 100 * np.exp(np.cumsum(r))
    previous = np.concatenate([[100.0], prices[:-1]])
    frame = pd.DataFrame(
        {
            "model": "arima",
            "date": idx,
            "actual": prices,
            "previous": previous,
            "one_step": previous * np.exp(rng.standard_normal(n) * 0.005),
        }
    )
    eras = performance_by_era(frame, breakpoints=("2019-07-01",), cost_bps=5.0)
    assert len(eras) == 2
    assert eras["days"].sum() == n
    assert eras["lag1_autocorr"].abs().max() < 0.2
    assert (eras["sharpe_net"] <= eras["sharpe_gross"]).all()
    # eras shorter than a year are dropped rather than reported on noise
    short = performance_by_era(frame, breakpoints=("2018-02-01",))
    assert len(short) == 1
    with pytest.raises(ValueError, match="no predictions"):
        performance_by_era(frame, model="nope")
