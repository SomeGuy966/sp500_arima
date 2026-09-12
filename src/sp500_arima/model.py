"""ARIMA forecaster with information-criterion order selection.

All models in this package operate on **log prices**. An ARIMA(p, 1, q) on log prices is
an ARMA(p, q) on log returns, which is the natural stationary object for equity indices.
Forecasting in log space also guarantees positive price forecasts and yields correctly
asymmetric prediction intervals once exponentiated.

Order selection is a plain grid search over ``(p, q)`` and the trend specification,
scored by AIC or BIC. This replaces ``pmdarima.auto_arima`` with ~40 lines of statsmodels
and no extra dependency, at the cost of not implementing stepwise search - unnecessary for
the small grids that are appropriate for daily returns.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

Trend = Literal["n", "t"]
Criterion = Literal["aic", "bic"]

# Random walk: the null model every ARIMA on prices must beat.
RANDOM_WALK_ORDER = (0, 1, 0)


@dataclass(frozen=True)
class Forecast:
    """Point forecast and prediction interval for a block of future dates, in log space."""

    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    alpha: float

    @property
    def index(self) -> pd.Index:
        """Dates being forecast."""
        return self.mean.index

    def to_price(self) -> Forecast:
        """Exponentiate the log-space forecast into price space."""
        return Forecast(np.exp(self.mean), np.exp(self.lower), np.exp(self.upper), self.alpha)


class Forecaster(Protocol):
    """Common interface of the ARIMA model and the random-walk baselines."""

    name: str

    def fit(self, log_prices: pd.Series) -> Forecaster:
        """Estimate the model on a window of log prices."""
        ...

    def forecast(self, index: pd.Index, alpha: float = 0.05) -> Forecast:
        """Multi-step forecast for each date in ``index``."""
        ...

    def one_step_ahead(self, new_log_prices: pd.Series) -> pd.Series:
        """One-step-ahead predictions for new observations with fixed parameters."""
        ...


@dataclass(frozen=True)
class ModelConfig:
    """Search space and scoring rule for ARIMA order selection."""

    max_p: int = 3
    max_q: int = 3
    d: int = 1
    trends: tuple[Trend, ...] = ("n", "t")
    criterion: Criterion = "aic"
    fixed_order: tuple[int, int, int] | None = None
    fixed_trend: Trend | None = None

    def __post_init__(self) -> None:
        """Validate the search space."""
        if self.max_p < 0 or self.max_q < 0 or self.d < 0:
            raise ValueError("max_p, max_q and d must be non-negative")
        if not self.trends:
            raise ValueError("at least one trend specification is required")
        if self.criterion not in ("aic", "bic"):
            raise ValueError(f"criterion must be 'aic' or 'bic', got {self.criterion!r}")

    @property
    def candidates(self) -> list[tuple[tuple[int, int, int], Trend]]:
        """Enumerate ``(order, trend)`` pairs to evaluate."""
        if self.fixed_order is not None:
            trends = (self.fixed_trend,) if self.fixed_trend else self.trends
            return [(self.fixed_order, t) for t in trends]
        return [
            ((p, self.d, q), t)
            for p, q, t in product(range(self.max_p + 1), range(self.max_q + 1), self.trends)
        ]


@dataclass(frozen=True)
class OrderSelection:
    """Result of an information-criterion grid search."""

    order: tuple[int, int, int]
    trend: Trend
    criterion: Criterion
    table: pd.DataFrame = field(repr=False)

    @property
    def label(self) -> str:
        """Human-readable model label, e.g. ``ARIMA(1,1,0)+drift``."""
        p, d, q = self.order
        return f"ARIMA({p},{d},{q})" + ("+drift" if self.trend == "t" else "")


def _fit_arima(y: np.ndarray, order: tuple[int, int, int], trend: Trend) -> ARIMAResults:
    """Fit a single ARIMA specification, silencing optimizer chatter."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(y, order=order, trend=trend, enforce_stationarity=True)
        return model.fit(method_kwargs={"warn_convergence": False})


def select_order(y: np.ndarray | pd.Series, config: ModelConfig) -> OrderSelection:
    """Grid-search ARIMA orders and return the specification minimising the criterion.

    Candidates that fail to converge or produce a non-finite criterion are dropped; if
    every candidate fails, the random walk is returned so that downstream evaluation
    can proceed with an explicit, conservative fallback.
    """
    values = np.asarray(y, dtype=float)
    rows: list[dict[str, object]] = []
    for order, trend in config.candidates:
        try:
            res = _fit_arima(values, order, trend)
            score = float(res.aic if config.criterion == "aic" else res.bic)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.isfinite(score):
            continue
        rows.append(
            {
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "trend": trend,
                "aic": float(res.aic),
                "bic": float(res.bic),
                "llf": float(res.llf),
            }
        )
    table = pd.DataFrame(rows, columns=["p", "d", "q", "trend", "aic", "bic", "llf"])
    if table.empty:
        return OrderSelection(RANDOM_WALK_ORDER, "n", config.criterion, table)
    table = table.sort_values(config.criterion, kind="stable").reset_index(drop=True)
    best = table.iloc[0]
    return OrderSelection(
        order=(int(best["p"]), int(best["d"]), int(best["q"])),
        trend=str(best["trend"]),  # type: ignore[arg-type]
        criterion=config.criterion,
        table=table,
    )


class ArimaForecaster:
    """ARIMA on log prices with automatic order selection.

    Parameters
    ----------
    config
        Search space for order selection. Use ``fixed_order`` to skip the search.
    """

    name = "arima"

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.selection: OrderSelection | None = None
        self.results: ARIMAResults | None = None
        self.train_index: pd.Index | None = None
        self.fit_seconds: float = float("nan")

    # ------------------------------------------------------------------ fitting
    def fit(self, log_prices: pd.Series) -> ArimaForecaster:
        """Select an order on ``log_prices`` and fit the chosen model."""
        y = np.asarray(log_prices, dtype=float)
        if y.ndim != 1 or len(y) < 10:
            raise ValueError("need a 1-D series with at least 10 observations")
        started = time.perf_counter()
        self.selection = select_order(y, self.config)
        self.results = _fit_arima(y, self.selection.order, self.selection.trend)
        self.fit_seconds = time.perf_counter() - started
        self.train_index = log_prices.index
        return self

    def _require_fit(self) -> tuple[ARIMAResults, OrderSelection]:
        if self.results is None or self.selection is None:
            raise RuntimeError("call fit() before forecasting")
        return self.results, self.selection

    # ------------------------------------------------------------- forecasting
    def forecast(self, index: pd.Index, alpha: float = 0.05) -> Forecast:
        """Multi-step forecast for each date in ``index``, from the end of training."""
        results, _ = self._require_fit()
        steps = len(index)
        if steps == 0:
            raise ValueError("forecast index is empty")
        pred = results.get_forecast(steps)
        ci = np.asarray(pred.conf_int(alpha=alpha))
        mean = pd.Series(np.asarray(pred.predicted_mean), index=index, name="mean")
        lower = pd.Series(ci[:, 0], index=index, name="lower")
        upper = pd.Series(ci[:, 1], index=index, name="upper")
        return Forecast(mean, lower, upper, alpha)

    def one_step_ahead(self, new_log_prices: pd.Series) -> pd.Series:
        """One-step-ahead predictions for ``new_log_prices`` with parameters held fixed.

        The Kalman filter is run forward through the new observations, so the prediction
        for date *t* uses every observation up to *t-1* - exactly what a live system would
        have - without re-estimating coefficients. This is how "daily" forecasts are
        produced between monthly re-fits.
        """
        results, _ = self._require_fit()
        if new_log_prices.empty:
            return pd.Series(dtype=float, index=new_log_prices.index, name="one_step")
        n_train = int(results.nobs)
        extended = results.append(np.asarray(new_log_prices, dtype=float), refit=False)
        pred = extended.get_prediction(start=n_train, end=n_train + len(new_log_prices) - 1)
        return pd.Series(
            np.asarray(pred.predicted_mean), index=new_log_prices.index, name="one_step"
        )

    # -------------------------------------------------------------- inspection
    @property
    def order(self) -> tuple[int, int, int]:
        """Selected ``(p, d, q)``."""
        return self._require_fit()[1].order

    @property
    def trend(self) -> Trend:
        """Selected trend specification."""
        return self._require_fit()[1].trend

    @property
    def label(self) -> str:
        """Human-readable label of the fitted specification."""
        return self._require_fit()[1].label

    @property
    def aic(self) -> float:
        """AIC of the fitted model."""
        return float(self._require_fit()[0].aic)

    @property
    def bic(self) -> float:
        """BIC of the fitted model."""
        return float(self._require_fit()[0].bic)

    def residuals(self) -> pd.Series:
        """In-sample one-step-ahead residuals (the first ``d`` entries are dropped)."""
        results, selection = self._require_fit()
        resid = np.asarray(results.resid)[selection.order[1] :]
        index = self.train_index[selection.order[1] :] if self.train_index is not None else None
        return pd.Series(resid, index=index, name="residual")

    def summary(self) -> str:
        """Statsmodels summary table for the fitted model."""
        return str(self._require_fit()[0].summary())
