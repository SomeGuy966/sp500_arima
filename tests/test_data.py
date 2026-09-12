from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sp500_arima.data import (
    load_ohlcv,
    load_prices,
    normalize_index,
    save_ohlcv,
    slice_prices,
    to_log_returns,
    validate_prices,
)


def test_bundled_snapshot_is_clean(real_prices: pd.Series) -> None:
    assert real_prices.index.is_monotonic_increasing
    assert not real_prices.index.has_duplicates
    assert real_prices.index.tz is None
    assert (real_prices > 0).all()
    assert real_prices.index[0] == pd.Timestamp("1974-01-02")
    assert real_prices.index[-1] == pd.Timestamp("2024-12-31")
    assert len(real_prices) > 12_000


def test_normalize_index_strips_exchange_offsets() -> None:
    raw = pd.Index(["2024-01-02 00:00:00-05:00", "2024-07-01 00:00:00-04:00"])
    idx = normalize_index(raw)
    assert list(idx) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-07-01")]
    assert idx.tz is None
    assert idx.name == "Date"


@pytest.mark.parametrize(
    "values, message",
    [
        ([], "empty"),
        ([1.0, np.nan], "missing"),
        ([1.0, -1.0], "non-positive"),
        ([1.0, 0.0], "non-positive"),
    ],
)
def test_validate_prices_rejects_bad_series(values: list[float], message: str) -> None:
    index = pd.bdate_range("2020-01-01", periods=len(values))
    with pytest.raises(ValueError, match=message):
        validate_prices(pd.Series(values, index=index, dtype=float))


def test_validate_prices_rejects_duplicates() -> None:
    index = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_prices(pd.Series([1.0, 2.0], index=index))


def test_validate_prices_sorts() -> None:
    index = pd.DatetimeIndex(["2020-01-03", "2020-01-02"])
    out = validate_prices(pd.Series([2.0, 1.0], index=index))
    assert list(out) == [1.0, 2.0]


def test_slice_prices_bounds(random_walk_prices: pd.Series) -> None:
    window = slice_prices(random_walk_prices, "2015-06-01", "2015-07-01")
    assert window.index[0] >= pd.Timestamp("2015-06-01")
    assert window.index[-1] < pd.Timestamp("2015-07-01")
    with pytest.raises(ValueError, match="before end"):
        slice_prices(random_walk_prices, "2016-01-01", "2015-01-01")
    with pytest.raises(ValueError, match="first observation"):
        slice_prices(random_walk_prices, "1990-01-01")
    with pytest.raises(ValueError, match="no observations"):
        slice_prices(random_walk_prices, "2015-01-03", "2015-01-04")  # a weekend


def test_log_returns_roundtrip(random_walk_prices: pd.Series) -> None:
    r = to_log_returns(random_walk_prices)
    assert len(r) == len(random_walk_prices) - 1
    rebuilt = np.exp(np.log(random_walk_prices.iloc[0]) + r.cumsum())
    np.testing.assert_allclose(rebuilt.to_numpy(), random_walk_prices.iloc[1:].to_numpy())


def test_save_and_load_ohlcv_roundtrip(tmp_path: Path) -> None:
    index = pd.bdate_range("2020-01-01", periods=5, name="Date")
    frame = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "High": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "Close": [1.2, 2.2, 3.2, 4.2, 5.2],
            "Volume": [10, 20, 30, 40, 50],
        },
        index=index,
    )
    path = save_ohlcv(frame, tmp_path / "prices.csv")
    loaded = load_ohlcv(path)
    pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
    assert load_prices(path).tolist() == frame["Close"].tolist()


def test_load_prices_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch"):
        load_prices(tmp_path / "nope.csv")


def test_load_prices_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("Date,Price\n2020-01-01,1\n")
    with pytest.raises(ValueError, match="Close"):
        load_prices(path)
