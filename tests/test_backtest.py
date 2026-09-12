from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500_arima.backtest import (
    BacktestConfig,
    BacktestResult,
    Fold,
    describe_config,
    fold_schedule,
    make_folds,
    run_backtest,
    run_fold,
)
from sp500_arima.model import ModelConfig

SMALL_MODEL = ModelConfig(max_p=1, max_q=1, trends=("n",))


def test_backtest_config_validation() -> None:
    with pytest.raises(ValueError, match="train_window"):
        BacktestConfig(train_window=10)
    with pytest.raises(ValueError, match="alpha"):
        BacktestConfig(alpha=1.5)
    with pytest.raises(ValueError, match="before end"):
        BacktestConfig(start="2020-01-01", end="2019-01-01")


def test_make_folds_monthly_blocks_are_contiguous(random_walk_prices: pd.Series) -> None:
    index = pd.DatetimeIndex(random_walk_prices.index)
    config = BacktestConfig(start="2016-01-01", end="2016-07-01", train_window=120)
    folds = make_folds(index, config)
    assert len(folds) == 6
    for prev, fold in pairwise(folds):
        assert fold.test_start == prev.test_end  # no gaps, no overlap
    for fold in folds:
        assert fold.train_end == fold.test_start  # test starts right after training
        assert fold.n_train == 120
        block = index[fold.test_start : fold.test_end]
        assert block.month.nunique() == 1  # exactly one calendar month
    schedule = fold_schedule(list(index), folds)
    assert list(schedule.columns) == [
        "fold",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "n_train",
        "n_test",
    ]
    assert (schedule["train_end"] < schedule["test_start"]).all()


def test_make_folds_rejects_insufficient_history(random_walk_prices: pd.Series) -> None:
    index = pd.DatetimeIndex(random_walk_prices.index)
    with pytest.raises(ValueError, match="not enough history"):
        make_folds(index, BacktestConfig(start="2015-02-01", end="2015-04-01", train_window=252))
    with pytest.raises(ValueError, match="no observations"):
        make_folds(index, BacktestConfig(start="2030-01-01", end="2031-01-01"))


def test_make_folds_quarterly(random_walk_prices: pd.Series) -> None:
    index = pd.DatetimeIndex(random_walk_prices.index)
    folds = make_folds(
        index, BacktestConfig(start="2016-01-01", end="2017-01-01", train_window=60, refit="Q")
    )
    assert len(folds) == 4
    assert all(f.n_test >= 60 for f in folds)


def test_run_fold_has_no_lookahead(random_walk_prices: pd.Series) -> None:
    index = pd.DatetimeIndex(random_walk_prices.index)
    log_prices = np.log(random_walk_prices.to_numpy())
    fold = Fold(0, 0, 100, 100, 121)
    frame, meta = run_fold(
        fold, log_prices, index, BacktestConfig(train_window=100, model=SMALL_MODEL)
    )

    assert set(frame["model"]) == {"arima", "naive", "drift"}
    assert (frame["fold"] == 0).all()
    assert meta["n_train"] == 100 and meta["n_test"] == 21
    assert meta["label"].startswith("ARIMA(")

    naive = frame[frame["model"] == "naive"].sort_values("step")
    origin = random_walk_prices.iloc[99]
    # naive path is flat at the forecast origin; one-step lags actual by one day
    np.testing.assert_allclose(naive["path"], origin)
    np.testing.assert_allclose(naive["origin"], origin)
    np.testing.assert_allclose(naive["one_step"].to_numpy()[1:], naive["actual"].to_numpy()[:-1])
    np.testing.assert_allclose(naive["previous"].to_numpy()[1:], naive["actual"].to_numpy()[:-1])
    assert naive["previous"].iloc[0] == pytest.approx(origin)
    np.testing.assert_allclose(naive["actual"], random_walk_prices.iloc[100:121])


def test_run_backtest_sequential_and_parallel_agree(random_walk_prices: pd.Series) -> None:
    config = BacktestConfig(
        start="2016-01-01", end="2016-04-01", train_window=100, model=SMALL_MODEL
    )
    seen: list[tuple[int, int]] = []
    sequential = run_backtest(
        random_walk_prices, config, n_jobs=1, progress=lambda d, t: seen.append((d, t))
    )
    parallel = run_backtest(random_walk_prices, config, n_jobs=2)
    assert seen[-1] == (3, 3)
    assert len(sequential.folds) == 3
    pd.testing.assert_frame_equal(sequential.predictions, parallel.predictions)
    assert sequential.models == ["arima", "naive", "drift"]
    assert sequential.elapsed_seconds > 0


def test_result_metrics_summary_and_roundtrip(
    random_walk_prices: pd.Series, tmp_path: Path
) -> None:
    config = BacktestConfig(
        start="2016-01-01", end="2016-07-01", train_window=100, model=SMALL_MODEL
    )
    result = run_backtest(random_walk_prices, config, n_jobs=1)

    per_fold = result.fold_metrics()
    assert len(per_fold) == 6 * 3
    assert (per_fold["path_coverage"].between(0, 1)).all()

    summary = result.summary()
    assert list(summary.index) == ["arima", "naive", "drift"]
    assert summary.loc["arima", "folds"] == 6
    assert np.isnan(summary.loc["naive", "path_dir_acc"])  # random walk has no direction
    assert np.isnan(summary.loc["naive", "one_step_dir_acc"])
    assert 0 <= summary.loc["arima", "one_step_dm_p"] <= 1
    assert 0 <= summary.loc["arima", "path_dm_p"] <= 1
    assert np.isnan(summary.loc["naive", "path_dm_p"])

    counts = result.order_counts()
    assert counts["folds"].sum() == 6
    assert counts["share"].sum() == pytest.approx(1.0)

    result.save(tmp_path)
    loaded = BacktestResult.load(tmp_path, config)
    pd.testing.assert_frame_equal(loaded.predictions, result.predictions, check_dtype=False)
    assert len(loaded.folds) == 6
    assert loaded.config == config


def test_describe_config_is_flat() -> None:
    flat = describe_config(BacktestConfig(model=SMALL_MODEL))
    assert flat["train_window"] == 252
    assert flat["model"]["max_p"] == 1


def test_stdin_main_falls_back_to_sequential(
    random_walk_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = "<stdin>"
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    config = BacktestConfig(
        start="2016-01-01", end="2016-03-01", train_window=100, model=SMALL_MODEL
    )
    with pytest.warns(RuntimeWarning, match="stdin"):
        result = run_backtest(random_walk_prices, config, n_jobs=4)
    assert len(result.folds) == 2


def test_bootstrapping_worker_raises_instead_of_hanging(
    random_walk_prices: pd.Series, monkeypatch: pytest.MonkeyPatch
) -> None:
    import multiprocessing

    monkeypatch.setattr(multiprocessing.current_process(), "_inheriting", True, raising=False)
    config = BacktestConfig(
        start="2016-01-01", end="2016-03-01", train_window=100, model=SMALL_MODEL
    )
    with pytest.raises(RuntimeError, match="__main__"):
        run_backtest(random_walk_prices, config, n_jobs=2)
    # sequential runs are always safe
    assert len(run_backtest(random_walk_prices, config, n_jobs=1).folds) == 2
