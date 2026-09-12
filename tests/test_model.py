from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sp500_arima.baselines import DriftForecaster, NaiveForecaster
from sp500_arima.model import ArimaForecaster, ModelConfig, select_order


def test_model_config_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ModelConfig(max_p=-1)
    with pytest.raises(ValueError, match="trend"):
        ModelConfig(trends=())
    with pytest.raises(ValueError, match="criterion"):
        ModelConfig(criterion="mse")  # type: ignore[arg-type]


def test_candidates_enumerate_grid_and_fixed_order() -> None:
    grid = ModelConfig(max_p=1, max_q=1, trends=("n",)).candidates
    assert grid == [((0, 1, 0), "n"), ((0, 1, 1), "n"), ((1, 1, 0), "n"), ((1, 1, 1), "n")]
    fixed = ModelConfig(fixed_order=(2, 1, 0), fixed_trend="t").candidates
    assert fixed == [((2, 1, 0), "t")]


def test_select_order_recovers_ar1(ar1_prices: pd.Series) -> None:
    """With phi=0.6 in the returns, the AIC search should prefer an AR term."""
    selection = select_order(np.log(ar1_prices), ModelConfig(max_p=2, max_q=2, trends=("n",)))
    assert selection.order[1] == 1
    assert selection.order[0] >= 1
    assert selection.table["aic"].is_monotonic_increasing
    assert selection.label.startswith("ARIMA(")


def test_select_order_prefers_random_walk_on_white_noise(random_walk_prices: pd.Series) -> None:
    """BIC is consistent, so on a pure random walk it should not add AR/MA terms."""
    config = ModelConfig(max_p=2, max_q=2, criterion="bic")
    selection = select_order(np.log(random_walk_prices), config)
    assert selection.order == (0, 1, 0)
    assert selection.table["bic"].is_monotonic_increasing


def test_arima_forecaster_end_to_end(ar1_prices: pd.Series) -> None:
    log_prices = np.log(ar1_prices)
    train, test = log_prices.iloc[:-20], log_prices.iloc[-20:]
    model = ArimaForecaster(ModelConfig(max_p=2, max_q=1, trends=("n",))).fit(train)

    fc = model.forecast(test.index, alpha=0.05)
    assert list(fc.index) == list(test.index)
    assert (fc.lower < fc.mean).all() and (fc.mean < fc.upper).all()
    # interval widens with horizon
    widths = (fc.upper - fc.lower).to_numpy()
    assert widths[-1] > widths[0]
    # price-space interval is exp of log-space interval
    price = fc.to_price()
    np.testing.assert_allclose(price.mean.to_numpy(), np.exp(fc.mean.to_numpy()))

    one_step = model.one_step_ahead(test)
    assert len(one_step) == len(test)
    # first one-step prediction equals the first path step (same information set)
    assert one_step.iloc[0] == pytest.approx(fc.mean.iloc[0])
    # later one-step predictions use realised data and should differ from the path
    assert not np.allclose(one_step.to_numpy()[1:], fc.mean.to_numpy()[1:])

    resid = model.residuals()
    assert len(resid) == len(train) - model.order[1]
    assert np.isfinite(model.aic) and np.isfinite(model.bic)
    assert "ARIMA" in model.summary()
    assert model.fit_seconds > 0


def test_arima_forecaster_guards(random_walk_prices: pd.Series) -> None:
    model = ArimaForecaster()
    with pytest.raises(RuntimeError, match="fit"):
        model.forecast(random_walk_prices.index[:5])
    with pytest.raises(ValueError, match="at least 10"):
        model.fit(np.log(random_walk_prices.iloc[:5]))
    fitted = model.fit(np.log(random_walk_prices.iloc[:100]))
    with pytest.raises(ValueError, match="empty"):
        fitted.forecast(pd.DatetimeIndex([]))
    assert fitted.one_step_ahead(pd.Series(dtype=float)).empty


def test_random_walk_order_matches_naive_baseline(random_walk_prices: pd.Series) -> None:
    """ARIMA(0,1,0) with no trend *is* the naive forecaster; both must agree exactly."""
    log_prices = np.log(random_walk_prices)
    train, test = log_prices.iloc[:200], log_prices.iloc[200:230]
    arima = ArimaForecaster(ModelConfig(fixed_order=(0, 1, 0), fixed_trend="n")).fit(train)
    naive = NaiveForecaster().fit(train)
    np.testing.assert_allclose(arima.forecast(test.index).mean, naive.forecast(test.index).mean)
    np.testing.assert_allclose(arima.one_step_ahead(test), naive.one_step_ahead(test))


def test_naive_and_drift_baselines(random_walk_prices: pd.Series) -> None:
    log_prices = np.log(random_walk_prices)
    train, test = log_prices.iloc[:200], log_prices.iloc[200:210]
    naive = NaiveForecaster().fit(train)
    drift = DriftForecaster().fit(train)

    fc = naive.forecast(test.index)
    assert (fc.mean == train.iloc[-1]).all()
    widths = (fc.upper - fc.lower).to_numpy()
    # sqrt-of-time: width at step 4 is twice width at step 1
    assert widths[3] == pytest.approx(2 * widths[0])

    mu = np.diff(train.to_numpy()).mean()
    expected = train.iloc[-1] + mu * np.arange(1, 11)
    np.testing.assert_allclose(drift.forecast(test.index).mean.to_numpy(), expected)

    one = naive.one_step_ahead(test)
    assert one.iloc[0] == train.iloc[-1]
    np.testing.assert_allclose(one.to_numpy()[1:], test.to_numpy()[:-1])
    np.testing.assert_allclose(drift.one_step_ahead(test).to_numpy(), one.to_numpy() + mu)

    with pytest.raises(RuntimeError, match="fit"):
        NaiveForecaster().forecast(test.index)
    with pytest.raises(ValueError, match="at least 3"):
        NaiveForecaster().fit(train.iloc[:2])
