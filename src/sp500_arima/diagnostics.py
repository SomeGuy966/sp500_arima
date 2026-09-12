"""Stationarity and residual diagnostics.

ARIMA assumes the differenced series is (weakly) stationary. Two complementary tests are
used because they have opposite null hypotheses:

* **ADF** (Augmented Dickey-Fuller) - H0: the series has a unit root (non-stationary).
* **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin) - H0: the series is stationary.

A series that *rejects* ADF and *fails to reject* KPSS is unambiguously stationary; the
opposite pattern is an unambiguous unit root. Mixed outcomes indicate borderline behaviour
and are reported as such rather than silently resolved.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass(frozen=True)
class StationarityResult:
    """Outcome of a single unit-root / stationarity test."""

    test: str
    series: str
    statistic: float
    p_value: float
    lags: int
    null_hypothesis: str
    stationary: bool

    def to_row(self) -> dict[str, object]:
        """Flatten into a dictionary for tabular reporting."""
        return {
            "series": self.series,
            "test": self.test,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "lags": self.lags,
            "null": self.null_hypothesis,
            "conclusion": "stationary" if self.stationary else "non-stationary",
        }


def adf_test(series: pd.Series, alpha: float = 0.05, regression: str = "c") -> StationarityResult:
    """Run the Augmented Dickey-Fuller test with AIC-selected lag length."""
    values = np.asarray(series.dropna(), dtype=float)
    res = adfuller(values, regression=regression, autolag="AIC", result_object=True)
    return StationarityResult(
        test="ADF",
        series=str(series.name or "series"),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        lags=int(res.lags),
        null_hypothesis="unit root",
        stationary=bool(res.pvalue < alpha),
    )


def kpss_test(series: pd.Series, alpha: float = 0.05, regression: str = "c") -> StationarityResult:
    """Run the KPSS test with automatic bandwidth selection.

    statsmodels interpolates the p-value from a small table and warns when the statistic
    falls outside it; the p-value is then clipped to ``[0.01, 0.10]``, which is sufficient
    for a decision at conventional significance levels.
    """
    values = np.asarray(series.dropna(), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        res = kpss(values, regression=regression, nlags="auto", result_object=True)
    return StationarityResult(
        test="KPSS",
        series=str(series.name or "series"),
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        lags=int(res.lags),
        null_hypothesis="stationary",
        stationary=bool(res.pvalue >= alpha),
    )


def stationarity_report(log_prices: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Test log prices and log returns with both ADF and KPSS."""
    log_returns = log_prices.diff().dropna().rename("log_return")
    log_prices = log_prices.rename("log_price")
    rows = [
        adf_test(log_prices, alpha).to_row(),
        kpss_test(log_prices, alpha).to_row(),
        adf_test(log_returns, alpha).to_row(),
        kpss_test(log_returns, alpha).to_row(),
    ]
    return pd.DataFrame(rows)


def choose_differencing(log_prices: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    """Return the smallest ``d`` for which the differenced series is stationary.

    A level of differencing is accepted when ADF rejects a unit root *and* KPSS does not
    reject stationarity. If no level up to ``max_d`` satisfies both, ``max_d`` is returned.
    """
    series = log_prices.astype(float)
    for d in range(max_d + 1):
        candidate = series.diff(d).dropna() if d else series.dropna()
        if adf_test(candidate, alpha).stationary and kpss_test(candidate, alpha).stationary:
            return d
    return max_d


def ljung_box(residuals: pd.Series, lags: tuple[int, ...] = (5, 10, 20)) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation in a residual series at several lag lengths.

    Small p-values indicate remaining serial correlation - i.e. the model has left
    predictable structure in the residuals.
    """
    values = residuals.dropna()
    table = acorr_ljungbox(values, lags=list(lags), return_df=True)
    table.index.name = "lag"
    renamed: pd.DataFrame = table.rename(columns={"lb_stat": "statistic", "lb_pvalue": "p_value"})
    return renamed
