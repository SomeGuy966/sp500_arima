from __future__ import annotations

import numpy as np
import pytest

from sp500_arima.metrics import (
    diebold_mariano,
    directional_accuracy,
    interval_coverage,
    mae,
    mape,
    rmse,
)


def test_point_metrics_hand_computed() -> None:
    actual = [100.0, 200.0, 400.0]
    predicted = [110.0, 180.0, 400.0]
    assert mape(actual, predicted) == pytest.approx((10 + 10 + 0) / 3)
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((100 + 400 + 0) / 3))
    assert mae(actual, predicted) == pytest.approx(30 / 3)


def test_point_metrics_validate_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        mape([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        rmse([], [])
    with pytest.raises(ValueError, match="zeros"):
        mape([0.0, 1.0], [1.0, 1.0])


def test_directional_accuracy_ignores_zero_forecasts() -> None:
    actual = np.array([0.01, -0.02, 0.03, -0.01])
    assert directional_accuracy(actual, np.array([0.02, -0.01, -0.01, -0.03])) == 0.75
    assert directional_accuracy(actual, np.array([0.0, 0.0, 0.01, 0.0])) == 1.0
    assert np.isnan(directional_accuracy(actual, np.zeros(4)))
    # exp(log(x)) round-off must not count as a directional call
    assert np.isnan(directional_accuracy(actual, np.full(4, 1e-15)))


def test_interval_coverage() -> None:
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.5, 2.5, 2.0, 3.0])
    upper = np.array([1.5, 3.5, 3.5, 3.5])
    assert interval_coverage(actual, lower, upper) == 0.5


def test_diebold_mariano_identical_forecasts_is_null() -> None:
    e = np.random.default_rng(0).standard_normal(200)
    result = diebold_mariano(e, e.copy())
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_diebold_mariano_detects_clearly_better_model() -> None:
    rng = np.random.default_rng(1)
    good = rng.standard_normal(500) * 0.5
    bad = rng.standard_normal(500) * 1.5
    result = diebold_mariano(good, bad)
    assert result.statistic < 0  # negative favours the first argument
    assert result.p_value < 1e-6
    flipped = diebold_mariano(bad, good)
    assert flipped.statistic == pytest.approx(-result.statistic)


def test_diebold_mariano_absolute_loss_and_horizon() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(300), rng.standard_normal(300)
    res = diebold_mariano(a, b, horizon=5, loss="absolute")
    assert res.horizon == 5
    assert res.loss == "absolute"
    assert 0.0 <= res.p_value <= 1.0


def test_diebold_mariano_guards() -> None:
    with pytest.raises(ValueError, match="horizon"):
        diebold_mariano([1.0] * 10, [1.0] * 10, horizon=0)
    with pytest.raises(ValueError, match="at least"):
        diebold_mariano([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], horizon=1)
