"""Random-walk baselines.

Under the weak-form efficient market hypothesis the best forecast of tomorrow's price is
today's price (possibly plus a small drift). Any model that claims to forecast the index
must therefore be measured against these two forecasters rather than in isolation.

Both baselines expose the same ``fit`` / ``forecast`` / ``one_step_ahead`` interface as
:class:`~sp500_arima.model.ArimaForecaster` so the backtest treats them identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from sp500_arima.model import Forecast


class NaiveForecaster:
    """Driftless random walk: ``E[y_{t+h}] = y_t``, ``Var = h * sigma^2``."""

    name = "naive"
    label = "Random walk"

    def __init__(self) -> None:
        self.last: float = float("nan")
        self.sigma: float = float("nan")
        self.fit_seconds = 0.0

    def _drift(self) -> float:
        return 0.0

    def fit(self, log_prices: pd.Series) -> NaiveForecaster:
        """Record the final observation and the return volatility of the window."""
        y = np.asarray(log_prices, dtype=float)
        if len(y) < 3:
            raise ValueError("need at least 3 observations")
        self.last = float(y[-1])
        self.sigma = float(np.std(np.diff(y), ddof=1))
        return self

    def forecast(self, index: pd.Index, alpha: float = 0.05) -> Forecast:
        """Flat (or linearly drifting) path with a square-root-of-time interval."""
        if np.isnan(self.last):
            raise RuntimeError("call fit() before forecasting")
        h = np.arange(1, len(index) + 1, dtype=float)
        mean = self.last + self._drift() * h
        z = stats.norm.ppf(1 - alpha / 2)
        half_width = z * self.sigma * np.sqrt(h)
        return Forecast(
            pd.Series(mean, index=index, name="mean"),
            pd.Series(mean - half_width, index=index, name="lower"),
            pd.Series(mean + half_width, index=index, name="upper"),
            alpha,
        )

    def one_step_ahead(self, new_log_prices: pd.Series) -> pd.Series:
        """Predict each new observation from the one immediately before it."""
        if np.isnan(self.last):
            raise RuntimeError("call fit() before forecasting")
        previous = np.concatenate([[self.last], np.asarray(new_log_prices, dtype=float)[:-1]])
        return pd.Series(previous + self._drift(), index=new_log_prices.index, name="one_step")


class DriftForecaster(NaiveForecaster):
    """Random walk with drift: ``E[y_{t+h}] = y_t + h * mean(return)``."""

    name = "drift"
    label = "Random walk + drift"

    def __init__(self) -> None:
        super().__init__()
        self.mu: float = float("nan")

    def _drift(self) -> float:
        return self.mu

    def fit(self, log_prices: pd.Series) -> DriftForecaster:
        """Estimate the mean log return of the window in addition to the naive state."""
        super().fit(log_prices)
        self.mu = float(np.mean(np.diff(np.asarray(log_prices, dtype=float))))
        return self
