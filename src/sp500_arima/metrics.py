"""Forecast accuracy metrics and the Diebold-Mariano test.

Point metrics are computed in price space so they are interpretable in index points or
percent. Directional accuracy is computed on returns because the sign of a price forecast
is trivially "up" for any positive series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

ArrayLike = np.ndarray | pd.Series | list[float]
Loss = Literal["squared", "absolute"]


def _pair(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if a.size == 0:
        raise ValueError("empty inputs")
    return a, p


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute percentage error, in percent."""
    a, p = _pair(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when actual contains zeros")
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error."""
    a, p = _pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute error."""
    a, p = _pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def directional_accuracy(
    actual_returns: ArrayLike, predicted_returns: ArrayLike, atol: float = 1e-10
) -> float:
    """Share of periods where the forecast return has the same sign as the realised return.

    Periods where either return is zero (within ``atol``, to absorb ``exp(log(x))``
    round-off) are excluded, so a forecast of "no change" is neither rewarded nor penalised.
    """
    a, p = _pair(actual_returns, predicted_returns)
    mask = (np.abs(a) > atol) & (np.abs(p) > atol)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(a[mask]) == np.sign(p[mask])))


def interval_coverage(actual: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Empirical coverage of a prediction interval (target: ``1 - alpha``)."""
    a = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (a.shape == lo.shape == hi.shape):
        raise ValueError("actual, lower and upper must have the same shape")
    return float(np.mean((a >= lo) & (a <= hi)))


@dataclass(frozen=True)
class DieboldMariano:
    """Result of a Diebold-Mariano equal-predictive-accuracy test.

    ``statistic`` is negative when the *first* forecast has lower expected loss.
    """

    statistic: float
    p_value: float
    horizon: int
    n_obs: int
    loss: Loss
    mean_loss_diff: float

    def to_row(self) -> dict[str, object]:
        """Flatten into a dictionary for tabular reporting."""
        return {
            "dm_stat": round(self.statistic, 3),
            "dm_p_value": round(self.p_value, 4),
            "dm_horizon": self.horizon,
            "dm_n": self.n_obs,
        }


def diebold_mariano(
    errors_a: ArrayLike,
    errors_b: ArrayLike,
    horizon: int = 1,
    loss: Loss = "squared",
) -> DieboldMariano:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    Tests H0: ``E[L(e_a)] == E[L(e_b)]`` against a two-sided alternative. The long-run
    variance of the loss differential uses a rectangular kernel with ``horizon - 1`` lags,
    which is exact for optimal ``h``-step forecasts whose errors are MA(h-1).

    Parameters
    ----------
    errors_a, errors_b
        Forecast errors (actual minus forecast) of the two competing models, aligned.
    horizon
        Forecast horizon ``h`` the errors were produced at.
    loss
        ``"squared"`` or ``"absolute"`` loss.
    """
    ea, eb = _pair(errors_a, errors_b)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    d = (ea**2 - eb**2) if loss == "squared" else (np.abs(ea) - np.abs(eb))
    n = d.size
    if n < 2 * horizon + 2:
        raise ValueError(f"need at least {2 * horizon + 2} observations for horizon {horizon}")
    d_bar = float(d.mean())
    centred = d - d_bar
    gamma = [float(np.dot(centred[k:], centred[: n - k]) / n) for k in range(horizon)]
    long_run_var = gamma[0] + 2.0 * sum(gamma[1:])
    if long_run_var <= 0:  # rectangular kernel is not guaranteed PSD; fall back to gamma_0
        long_run_var = gamma[0]
    if long_run_var == 0:
        return DieboldMariano(0.0, 1.0, horizon, n, loss, d_bar)
    dm = d_bar / np.sqrt(long_run_var / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_star = float(dm * correction)
    p_value = float(2 * stats.t.sf(abs(dm_star), df=n - 1))
    return DieboldMariano(dm_star, p_value, horizon, n, loss, d_bar)


def summarize_point_forecasts(
    actual: pd.Series,
    predicted: pd.Series,
    previous: pd.Series,
) -> dict[str, float]:
    """Bundle the standard point-forecast metrics for a block of aligned predictions.

    ``previous`` is the price each forecast was made relative to (for direction).
    """
    a, p, prev = actual.to_numpy(float), predicted.to_numpy(float), previous.to_numpy(float)
    return {
        "mape": mape(a, p),
        "rmse": rmse(a, p),
        "mae": mae(a, p),
        "directional_accuracy": directional_accuracy(np.log(a / prev), np.log(p / prev)),
    }
