"""Walk-forward (rolling-origin) evaluation.

For every calendar month in the evaluation range the models are fitted on the trailing
``train_window`` trading days and asked for two kinds of forecast:

* a **path** forecast - the whole month in one shot from the last training day; and
* **one-step-ahead** forecasts - each day predicted from all data up to the day before,
  with parameters frozen at the monthly fit.

Every forecast is therefore strictly out-of-sample. The month-by-month structure gives a
long, non-overlapping series of forecast experiments (600 months for 1975-2024) on which
accuracy can be compared across models with proper hypothesis tests.
"""

from __future__ import annotations

import multiprocessing
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sp500_arima.baselines import DriftForecaster, NaiveForecaster
from sp500_arima.metrics import (
    DieboldMariano,
    diebold_mariano,
    directional_accuracy,
    interval_coverage,
    mape,
)
from sp500_arima.model import ArimaForecaster, Forecaster, ModelConfig

RefitFreq = Literal["W", "M", "Q"]
MODEL_LABELS = {"arima": "ARIMA (auto)", "naive": "Random walk", "drift": "Random walk + drift"}


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters of a walk-forward run."""

    start: str = "1975-01-01"
    end: str = "2025-01-01"
    train_window: int = 252
    refit: RefitFreq = "M"
    alpha: float = 0.05
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.train_window < 30:
            raise ValueError("train_window must be at least 30 observations")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("start must be before end")


@dataclass(frozen=True)
class Fold:
    """Positional boundaries of one train/test split (``end`` bounds are exclusive)."""

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def n_train(self) -> int:
        """Number of training observations."""
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        """Number of test observations (the forecast horizon)."""
        return self.test_end - self.test_start


def make_folds(index: pd.DatetimeIndex, config: BacktestConfig) -> list[Fold]:
    """Partition the evaluation range into one test block per refit period.

    Raises
    ------
    ValueError
        If the first test block does not have ``train_window`` observations before it.
    """
    start, end = pd.Timestamp(config.start), pd.Timestamp(config.end)
    in_range = np.flatnonzero((index >= start) & (index < end))
    if in_range.size == 0:
        raise ValueError(f"no observations between {start.date()} and {end.date()}")
    periods = index[in_range].to_period(config.refit)
    folds: list[Fold] = []
    for fold_id, (_, positions) in enumerate(pd.Series(in_range).groupby(periods.astype(str))):
        test_start, test_end = int(positions.iloc[0]), int(positions.iloc[-1]) + 1
        train_start = test_start - config.train_window
        if train_start < 0:
            earliest = index[config.train_window]
            raise ValueError(
                f"not enough history before {index[test_start].date()} for a "
                f"{config.train_window}-day window; use --start >= {earliest.date()}"
            )
        folds.append(Fold(fold_id, train_start, test_start, test_start, test_end))
    return folds


# --------------------------------------------------------------------------- workers
def _forecasters(config: ModelConfig) -> list[Forecaster]:
    return [ArimaForecaster(config), NaiveForecaster(), DriftForecaster()]


def run_fold(
    fold: Fold,
    log_prices: np.ndarray,
    index: pd.DatetimeIndex,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit every model on one fold and return per-day predictions plus fold metadata."""
    y_train = pd.Series(
        log_prices[fold.train_start : fold.train_end],
        index=index[fold.train_start : fold.train_end],
    )
    y_test = pd.Series(
        log_prices[fold.test_start : fold.test_end], index=index[fold.test_start : fold.test_end]
    )
    origin = float(np.exp(y_train.iloc[-1]))
    previous = np.exp(np.concatenate([[y_train.iloc[-1]], y_test.to_numpy()[:-1]]))
    actual = np.exp(y_test.to_numpy())

    frames: list[pd.DataFrame] = []
    meta: dict[str, object] = {
        "fold": fold.fold_id,
        "train_start": y_train.index[0],
        "train_end": y_train.index[-1],
        "test_start": y_test.index[0],
        "test_end": y_test.index[-1],
        "n_train": fold.n_train,
        "n_test": fold.n_test,
    }
    for forecaster in _forecasters(config.model):
        forecaster.fit(y_train)
        path = forecaster.forecast(y_test.index, alpha=config.alpha).to_price()
        one_step = np.exp(forecaster.one_step_ahead(y_test).to_numpy())
        frames.append(
            pd.DataFrame(
                {
                    "fold": fold.fold_id,
                    "date": y_test.index,
                    "step": np.arange(1, fold.n_test + 1),
                    "model": forecaster.name,
                    "actual": actual,
                    "previous": previous,
                    "origin": origin,
                    "path": path.mean.to_numpy(),
                    "path_lower": path.lower.to_numpy(),
                    "path_upper": path.upper.to_numpy(),
                    "one_step": one_step,
                }
            )
        )
        if isinstance(forecaster, ArimaForecaster):
            p, d, q = forecaster.order
            meta.update(
                {
                    "p": p,
                    "d": d,
                    "q": q,
                    "trend": forecaster.trend,
                    "label": forecaster.label,
                    "aic": forecaster.aic,
                    "bic": forecaster.bic,
                    "fit_seconds": forecaster.fit_seconds,
                }
            )
    return pd.concat(frames, ignore_index=True), meta


_WORKER_STATE: dict[str, object] = {}


def _init_worker(cache_path: str, config: BacktestConfig) -> None:
    """Load the shared price array once per worker.

    The data is handed over as a small ``.npz`` path rather than pickled into ``initargs``:
    a large ``initargs`` payload (this one is ~200 KB) exceeds the 64 KB pipe buffer used by
    the ``spawn`` start method, so the parent blocks inside ``Process.start()`` and cannot
    notice a worker that dies while bootstrapping.
    """
    with np.load(cache_path) as data:
        _WORKER_STATE.update(
            log_prices=data["log_prices"],
            index=pd.DatetimeIndex(data["index"]),
            config=config,
        )


def _run_fold_in_worker(fold: Fold) -> tuple[pd.DataFrame, dict[str, object]]:
    return run_fold(
        fold,
        _WORKER_STATE["log_prices"],  # type: ignore[arg-type]
        _WORKER_STATE["index"],  # type: ignore[arg-type]
        _WORKER_STATE["config"],  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- results
@dataclass
class BacktestResult:
    """Predictions and fold metadata from a walk-forward run."""

    predictions: pd.DataFrame
    folds: pd.DataFrame
    config: BacktestConfig
    elapsed_seconds: float = float("nan")

    @property
    def models(self) -> list[str]:
        """Model identifiers present in the result, ARIMA first."""
        present = list(dict.fromkeys(self.predictions["model"]))
        return sorted(
            present, key=lambda m: list(MODEL_LABELS).index(m) if m in MODEL_LABELS else 99
        )

    def fold_metrics(self) -> pd.DataFrame:
        """Per-fold, per-model accuracy: MAPE / RMSE for both horizons and interval coverage."""
        rows: list[dict[str, object]] = []
        for (fold, model), block in self.predictions.groupby(["fold", "model"], sort=True):
            a = block["actual"].to_numpy(float)
            rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "test_start": block["date"].iloc[0],
                    "n": len(block),
                    "path_mape": mape(a, block["path"]),
                    "path_rmse_pct": _rmse_pct(a, block["path"]),
                    "path_mse_log": float(np.mean(np.log(a / block["path"]) ** 2)),
                    "path_coverage": interval_coverage(a, block["path_lower"], block["path_upper"]),
                    "one_step_mape": mape(a, block["one_step"]),
                    "one_step_rmse_pct": _rmse_pct(a, block["one_step"]),
                }
            )
        return pd.DataFrame(rows)

    def summary(self, baseline: str = "naive") -> pd.DataFrame:
        """Aggregate accuracy per model with Diebold-Mariano tests against ``baseline``.

        Path-horizon DM tests use the monthly series of mean squared log errors (one
        observation per fold, non-overlapping, ``h=1``); one-step DM tests use the pooled
        daily log errors.
        """
        per_fold = self.fold_metrics()
        preds = self.predictions.sort_values(["model", "fold", "step"])
        base_fold = per_fold[per_fold["model"] == baseline].set_index("fold")
        base_daily = preds[preds["model"] == baseline]
        rows: list[dict[str, object]] = []
        for model in self.models:
            block = preds[preds["model"] == model]
            mine = per_fold[per_fold["model"] == model].set_index("fold")
            a = block["actual"].to_numpy(float)
            origin = block["origin"].to_numpy(float)
            prev = block["previous"].to_numpy(float)
            row: dict[str, object] = {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "folds": int(mine.shape[0]),
                "days": len(block),
                "path_mape": mape(a, block["path"]),
                "path_mape_monthly_mean": float(mine["path_mape"].mean()),
                "path_rmse_pct": _rmse_pct(a, block["path"]),
                "path_dir_acc": directional_accuracy(
                    np.log(a / origin), np.log(block["path"].to_numpy(float) / origin)
                ),
                "path_coverage": interval_coverage(a, block["path_lower"], block["path_upper"]),
                "one_step_mape": mape(a, block["one_step"]),
                "one_step_rmse_pct": _rmse_pct(a, block["one_step"]),
                "one_step_dir_acc": directional_accuracy(
                    np.log(a / prev), np.log(block["one_step"].to_numpy(float) / prev)
                ),
            }
            if model != baseline and not base_fold.empty:
                # Path horizon: DM on the monthly MSE series (proxy errors = sqrt of MSE).
                dm_path = _dm_or_nan(
                    np.sqrt(mine["path_mse_log"].reindex(base_fold.index)),
                    np.sqrt(base_fold["path_mse_log"]),
                )
                dm_step = _dm_or_nan(
                    np.log(a / block["one_step"].to_numpy(float)),
                    np.log(
                        base_daily["actual"].to_numpy(float)
                        / base_daily["one_step"].to_numpy(float)
                    ),
                )
                row.update(
                    {
                        "path_dm_stat": dm_path.statistic,
                        "path_dm_p": dm_path.p_value,
                        "one_step_dm_stat": dm_step.statistic,
                        "one_step_dm_p": dm_step.p_value,
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    def order_counts(self) -> pd.DataFrame:
        """How often each ARIMA specification was selected across folds."""
        counts = self.folds.groupby("label").size().sort_values(ascending=False)
        return (
            counts.rename("folds").to_frame().assign(share=lambda f: f["folds"] / f["folds"].sum())
        )

    # ------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        """Write predictions and fold metadata as CSV files."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.predictions.to_csv(directory / "backtest_predictions.csv", index=False)
        self.folds.to_csv(directory / "backtest_folds.csv", index=False)
        return directory

    @classmethod
    def load(cls, directory: str | Path, config: BacktestConfig | None = None) -> BacktestResult:
        """Reload a result written by :meth:`save`."""
        directory = Path(directory)
        predictions = pd.read_csv(directory / "backtest_predictions.csv", parse_dates=["date"])
        folds = pd.read_csv(
            directory / "backtest_folds.csv",
            parse_dates=["train_start", "train_end", "test_start", "test_end"],
        )
        return cls(predictions, folds, config or BacktestConfig())


def _dm_or_nan(
    errors_a: pd.Series | np.ndarray, errors_b: pd.Series | np.ndarray
) -> DieboldMariano:
    """Diebold-Mariano at ``h=1``, or a NaN result when the sample is too short to test."""
    try:
        return diebold_mariano(errors_a, errors_b, horizon=1)
    except ValueError:
        n = int(np.asarray(errors_a).size)
        return DieboldMariano(float("nan"), float("nan"), 1, n, "squared", float("nan"))


def _rmse_pct(actual: np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    """RMSE of log errors expressed in percent (scale-free, comparable across decades)."""
    err = np.log(actual / np.asarray(predicted, dtype=float))
    return float(np.sqrt(np.mean(err**2)) * 100.0)


# --------------------------------------------------------------------------- driver
def _can_spawn_workers() -> bool:
    """Whether ``spawn``-based workers can bootstrap from the current ``__main__``.

    With the ``spawn`` start method (the default on macOS and Windows) each worker re-runs
    the main script by path. A script piped through stdin has no re-runnable path, so the
    pool would hang; every other entry point (file, ``-m``, ``-c``, notebook) is fine.
    """
    main = sys.modules.get("__main__")
    return getattr(main, "__file__", None) != "<stdin>"


def _check_not_bootstrapping() -> None:
    """Fail fast if called while a spawned worker is still importing ``__main__``.

    That happens when a script calls :func:`run_backtest` at module level without an
    ``if __name__ == "__main__":`` guard: every worker re-runs the script, tries to start
    its own pool, and the parent waits forever. A clear error beats a silent hang.
    """
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        raise RuntimeError(
            "run_backtest was called while a worker process was bootstrapping. Wrap the "
            'call in `if __name__ == "__main__":` (required for multiprocessing on '
            "macOS/Windows) or pass n_jobs=1."
        )


def run_backtest(
    prices: pd.Series,
    config: BacktestConfig | None = None,
    n_jobs: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    """Run the walk-forward evaluation over every fold, optionally in parallel.

    Parameters
    ----------
    prices
        Daily close prices indexed by date (validated by :func:`~sp500_arima.data.load_prices`).
    config
        Backtest parameters; defaults to :class:`BacktestConfig`.
    n_jobs
        Worker processes. ``None`` or ``0`` uses all cores, ``1`` runs sequentially.
        Parallel runs use ``multiprocessing`` with the platform's default start method, so
        a script must guard the call with ``if __name__ == "__main__":`` on macOS/Windows.
    progress
        Optional callback ``progress(done, total)`` invoked after each fold completes.
    """
    config = config or BacktestConfig()
    if n_jobs != 1:
        _check_not_bootstrapping()
    index = pd.DatetimeIndex(prices.index)
    log_prices = np.log(prices.to_numpy(dtype=float))
    folds = make_folds(index, config)
    started = time.perf_counter()

    if n_jobs != 1 and not _can_spawn_workers():
        warnings.warn(
            "parallel backtest is unavailable when the main script is read from stdin "
            "(multiprocessing cannot re-import it); running sequentially",
            RuntimeWarning,
            stacklevel=2,
        )
        n_jobs = 1

    frames: list[pd.DataFrame] = []
    metas: list[dict[str, object]] = []

    def collect(outputs: Iterable[tuple[pd.DataFrame, dict[str, object]]]) -> None:
        for done, (frame, meta) in enumerate(outputs, start=1):
            frames.append(frame)
            metas.append(meta)
            if progress is not None:
                progress(done, len(folds))

    if n_jobs == 1:
        collect(run_fold(f, log_prices, index, config) for f in folds)
    else:
        with tempfile.TemporaryDirectory(prefix="sp500_arima_") as tmp:
            cache_path = Path(tmp) / "prices.npz"
            np.savez(cache_path, log_prices=log_prices, index=index.to_numpy())
            with ProcessPoolExecutor(
                max_workers=n_jobs or None,
                initializer=_init_worker,
                initargs=(str(cache_path), config),
            ) as executor:
                collect(executor.map(_run_fold_in_worker, folds, chunksize=4))

    predictions = pd.concat(frames, ignore_index=True)
    fold_frame = pd.DataFrame(metas)
    return BacktestResult(predictions, fold_frame, config, time.perf_counter() - started)


def describe_config(config: BacktestConfig) -> dict[str, object]:
    """Flatten a configuration into a JSON-friendly dictionary for reports."""
    flat = asdict(config)
    flat["model"] = asdict(config.model)
    return flat


def fold_schedule(index: Sequence[pd.Timestamp], folds: Sequence[Fold]) -> pd.DataFrame:
    """Human-readable table of fold boundaries (useful for sanity-checking a config)."""
    return pd.DataFrame(
        {
            "fold": [f.fold_id for f in folds],
            "train_start": [index[f.train_start] for f in folds],
            "train_end": [index[f.train_end - 1] for f in folds],
            "test_start": [index[f.test_start] for f in folds],
            "test_end": [index[f.test_end - 1] for f in folds],
            "n_train": [f.n_train for f in folds],
            "n_test": [f.n_test for f in folds],
        }
    )
