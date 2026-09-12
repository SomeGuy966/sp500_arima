"""Turn one-step-ahead forecasts into a daily trading signal and evaluate it.

The signal for day *t* is the sign of the forecast return from the close of *t-1* to the
close of *t*. That forecast is produced from data up to *t-1* only (see
:meth:`~sp500_arima.model.ArimaForecaster.one_step_ahead`), so the position is decided at
the *t-1* close and earns the realised return on day *t* - no look-ahead.

Transaction costs are charged on turnover (change in position) in basis points, which is
the term that usually erases whatever small edge a linear model finds in daily returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd

from sp500_arima.data import TRADING_DAYS_PER_YEAR
from sp500_arima.metrics import directional_accuracy

Mode = Literal["long_short", "long_flat"]


@dataclass(frozen=True)
class StrategyConfig:
    """Signal construction and cost assumptions."""

    mode: Mode = "long_short"
    cost_bps: float = 5.0
    threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")


@dataclass(frozen=True)
class StrategyStats:
    """Performance summary of a daily return series."""

    name: str
    n_days: int
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    time_in_market: float

    def to_row(self) -> dict[str, object]:
        """Flatten into a dictionary for tabular reporting."""
        return {
            "strategy": self.name,
            "days": self.n_days,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "annual_vol": self.annual_vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "hit_rate": self.hit_rate,
            "turnover": self.turnover,
            "time_in_market": self.time_in_market,
        }


def positions_from_forecast(predicted_log_return: pd.Series, config: StrategyConfig) -> pd.Series:
    """Map forecast returns to positions in ``{-1, 0, +1}``."""
    signal = np.sign(predicted_log_return.to_numpy(float))
    signal[np.abs(predicted_log_return.to_numpy(float)) < config.threshold] = 0.0
    if config.mode == "long_flat":
        signal = np.clip(signal, 0.0, 1.0)
    return pd.Series(signal, index=predicted_log_return.index, name="position")


def strategy_returns(
    realised_simple_return: pd.Series,
    positions: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Daily net returns of holding ``positions`` through ``realised_simple_return``.

    Turnover on day *t* is ``|pos_t - pos_{t-1}|`` (starting flat), charged at
    ``cost_bps`` per unit.
    """
    pos = positions.to_numpy(float)
    prev_pos = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev_pos)
    net = pos * realised_simple_return.to_numpy(float) - turnover * cost_bps / 1e4
    return pd.Series(net, index=realised_simple_return.index, name="strategy_return")


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline of an equity curve, as a negative fraction."""
    running_max = np.maximum.accumulate(equity)
    return float(np.min(equity / running_max - 1.0))


def performance(returns: pd.Series, name: str, positions: pd.Series | None = None) -> StrategyStats:
    """Annualised performance statistics of a daily simple-return series (``rf = 0``)."""
    r = returns.to_numpy(float)
    n = r.size
    equity = np.cumprod(1.0 + r)
    years = n / TRADING_DAYS_PER_YEAR
    total = float(equity[-1] - 1.0)
    annual = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if n > 1 else float("nan")
    daily_std = float(np.std(r, ddof=1)) if n > 1 else float("nan")
    sharpe = (
        float(np.mean(r) / daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
        if daily_std > 1e-12  # a constant return series has no meaningful Sharpe ratio
        else float("nan")
    )
    if positions is None:
        active = np.ones(n, dtype=bool)
        turnover = 0.0
    else:
        pos = positions.to_numpy(float)
        active = pos != 0
        turnover = float(np.mean(np.abs(np.diff(np.concatenate([[0.0], pos])))))
    hit = float(np.mean(r[active] > 0)) if active.any() else float("nan")
    return StrategyStats(
        name=name,
        n_days=n,
        total_return=total,
        annual_return=annual,
        annual_vol=vol,
        sharpe=sharpe,
        max_drawdown=max_drawdown(equity),
        hit_rate=hit,
        turnover=turnover,
        time_in_market=float(np.mean(active)),
    )


def evaluate_model_signal(
    predictions: pd.DataFrame,
    config: StrategyConfig,
    name: str,
) -> tuple[StrategyStats, pd.Series]:
    """Evaluate the signal implied by one model's ``one_step`` forecasts.

    ``predictions`` must be a single-model slice of
    :attr:`~sp500_arima.backtest.BacktestResult.predictions`, sorted by date.
    """
    block = predictions.sort_values("date").set_index("date")
    predicted = pd.Series(
        np.log(block["one_step"].to_numpy(float) / block["previous"].to_numpy(float)),
        index=block.index,
        name="predicted",
    )
    realised = (block["actual"] / block["previous"] - 1.0).rename("realised")
    positions = positions_from_forecast(predicted, config)
    net = strategy_returns(realised, positions, config.cost_bps)
    return performance(net, name, positions), net


def compare_strategies(
    predictions: pd.DataFrame,
    config: StrategyConfig | None = None,
    models: tuple[str, ...] = ("arima", "drift"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate each model's signal alongside buy-and-hold on the same days.

    Returns
    -------
    stats, equity
        A table of :class:`StrategyStats` rows and a DataFrame of equity curves (starting at
        1.0) with one column per strategy.
    """
    config = config or StrategyConfig()
    rows: list[dict[str, object]] = []
    curves: dict[str, pd.Series] = {}
    first = predictions[predictions["model"] == models[0]].sort_values("date").set_index("date")
    buy_hold = (first["actual"] / first["previous"] - 1.0).rename("buy_hold")
    rows.append(performance(buy_hold, "Buy & hold").to_row())
    curves["Buy & hold"] = (1.0 + buy_hold).cumprod()
    for model in models:
        slice_ = predictions[predictions["model"] == model]
        if slice_.empty:
            continue
        label = f"{model} signal ({config.mode.replace('_', '/')})"
        stats, net = evaluate_model_signal(slice_, config, label)
        rows.append(stats.to_row())
        curves[label] = (1.0 + net).cumprod()
    equity = pd.DataFrame(curves)
    equity.index.name = "date"
    return pd.DataFrame(rows).set_index("strategy"), equity


def cost_sensitivity(
    predictions: pd.DataFrame,
    costs_bps: tuple[float, ...] = (0.0, 1.0, 2.5, 5.0, 10.0),
    mode: Mode = "long_short",
    models: tuple[str, ...] = ("arima", "drift"),
) -> pd.DataFrame:
    """Sharpe ratio and CAGR of each model's signal across transaction-cost assumptions.

    The first question to ask of any daily signal is whether it survives realistic costs;
    this table answers it directly. Buy-and-hold is included as the zero-turnover reference.
    """
    rows: list[dict[str, object]] = []
    for cost in costs_bps:
        stats, _ = compare_strategies(predictions, StrategyConfig(mode=mode, cost_bps=cost), models)
        for name, row in stats.iterrows():
            rows.append(
                {
                    "cost_bps": cost,
                    "strategy": name,
                    "sharpe": row["sharpe"],
                    "annual_return": row["annual_return"],
                }
            )
    table = pd.DataFrame(rows).pivot(index="strategy", columns="cost_bps")
    return table.reindex(stats.index)


def performance_by_era(
    predictions: pd.DataFrame,
    breakpoints: tuple[str, ...] = ("1990-01-01", "2005-01-01"),
    cost_bps: float = 5.0,
    mode: Mode = "long_short",
    model: str = "arima",
) -> pd.DataFrame:
    """Compare the signal with buy-and-hold within sub-periods split at ``breakpoints``.

    Also reports the lag-1 autocorrelation of daily log returns in each era - the quantity
    a low-order ARIMA is effectively betting on - and the signal's directional accuracy.
    """
    block = predictions[predictions["model"] == model].sort_values("date")
    if block.empty:
        raise ValueError(f"no predictions for model {model!r}")
    edges = [block["date"].min(), *[pd.Timestamp(b) for b in breakpoints], block["date"].max()]
    rows: list[dict[str, object]] = []
    for start, end in pairwise(edges):
        last = end == edges[-1]
        era = block[
            (block["date"] >= start) & ((block["date"] <= end) if last else (block["date"] < end))
        ]
        if len(era) < TRADING_DAYS_PER_YEAR:
            continue
        realised = np.log(era["actual"].to_numpy(float) / era["previous"].to_numpy(float))
        predicted = np.log(era["one_step"].to_numpy(float) / era["previous"].to_numpy(float))
        gross, _ = evaluate_model_signal(era, StrategyConfig(mode=mode, cost_bps=0.0), model)
        net, _ = evaluate_model_signal(era, StrategyConfig(mode=mode, cost_bps=cost_bps), model)
        buy_hold = performance(pd.Series(np.exp(realised) - 1.0, index=era["date"]), "bh")
        label_end = era["date"].max()
        rows.append(
            {
                "era": f"{era['date'].min():%Y}–{label_end:%Y}",
                "days": len(era),
                "lag1_autocorr": float(np.corrcoef(realised[1:], realised[:-1])[0, 1]),
                "dir_acc": directional_accuracy(realised, predicted),
                "sharpe_gross": gross.sharpe,
                "sharpe_net": net.sharpe,
                "sharpe_buy_hold": buy_hold.sharpe,
            }
        )
    columns = [
        "era",
        "days",
        "lag1_autocorr",
        "dir_acc",
        "sharpe_gross",
        "sharpe_net",
        "sharpe_buy_hold",
    ]
    return pd.DataFrame(rows, columns=columns).set_index("era")
