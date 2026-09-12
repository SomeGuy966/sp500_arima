"""Loading, validating and transforming daily S&P 500 price data.

The repository ships a snapshot of daily ``^GSPC`` OHLCV bars (1974-2024) in
``data/sp500_daily.csv`` so that every command works offline and every result is
reproducible. ``fetch_prices`` can refresh that snapshot from Yahoo Finance when the
optional ``yfinance`` extra is installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TRADING_DAYS_PER_YEAR = 252


def default_data_path() -> Path:
    """Return the path of the bundled price snapshot.

    Resolves relative to the repository root when running from a source checkout, and
    falls back to ``data/sp500_daily.csv`` under the current working directory.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "data" / "sp500_daily.csv"
    if candidate.exists():
        return candidate
    return Path("data") / "sp500_daily.csv"


def normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    """Coerce an index of dates into a sorted, timezone-naive ``DatetimeIndex``.

    Yahoo exports timestamps with an exchange-local UTC offset (``-05:00`` / ``-04:00``);
    the offset carries no information for daily bars and complicates comparisons, so it is
    stripped after conversion to UTC.
    """
    dt = pd.to_datetime(index, utc=True).tz_convert(None).normalize()
    return pd.DatetimeIndex(dt, name="Date")


def validate_prices(prices: pd.Series) -> pd.Series:
    """Validate a daily close series and return a clean, sorted copy.

    Raises
    ------
    ValueError
        If the series is empty, contains non-positive or missing values, or has duplicate
        dates. A price series that fails these checks cannot be log-transformed safely.
    """
    if prices.empty:
        raise ValueError("price series is empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("price series must be indexed by DatetimeIndex")
    if prices.index.has_duplicates:
        dupes = prices.index[prices.index.duplicated()][:5].tolist()
        raise ValueError(f"duplicate dates in price series: {dupes}")
    if prices.isna().any():
        raise ValueError(f"{int(prices.isna().sum())} missing values in price series")
    if (prices <= 0).any():
        raise ValueError("price series contains non-positive values")
    return prices.sort_index().astype(float)


def load_ohlcv(path: str | Path | None = None) -> pd.DataFrame:
    """Load the daily OHLCV snapshot as a DataFrame indexed by date."""
    path = Path(path) if path is not None else default_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `sp500-arima fetch` (requires the `fetch` extra) "
            "or point --data at a CSV with a Date column and a Close column."
        )
    frame = pd.read_csv(path)
    if "Date" not in frame.columns or "Close" not in frame.columns:
        raise ValueError(f"{path} must contain 'Date' and 'Close' columns")
    frame.index = normalize_index(pd.Index(frame.pop("Date")))
    frame = frame.sort_index()
    keep = [c for c in OHLCV_COLUMNS if c in frame.columns]
    return frame[keep]


def load_prices(path: str | Path | None = None) -> pd.Series:
    """Load the daily close series from the snapshot CSV."""
    close = load_ohlcv(path)["Close"].rename("Close")
    return validate_prices(close)


def slice_prices(prices: pd.Series, start: str | None = None, end: str | None = None) -> pd.Series:
    """Return the sub-series within ``[start, end)`` and verify it is non-empty."""
    start_ts = pd.Timestamp(start) if start else prices.index[0]
    end_ts = pd.Timestamp(end) if end else prices.index[-1] + pd.Timedelta(days=1)
    if start_ts >= end_ts:
        raise ValueError(f"start ({start_ts.date()}) must be before end ({end_ts.date()})")
    if start_ts < prices.index[0]:
        raise ValueError(
            f"start ({start_ts.date()}) is before the first observation ({prices.index[0].date()})"
        )
    window = prices.loc[(prices.index >= start_ts) & (prices.index < end_ts)]
    if window.empty:
        raise ValueError(f"no observations between {start_ts.date()} and {end_ts.date()}")
    return window


def to_log_returns(prices: pd.Series) -> pd.Series:
    """Convert a price series into daily log returns ``r_t = ln(P_t / P_{t-1})``."""
    returns = pd.Series(np.log(prices.to_numpy(dtype=float)), index=prices.index).diff()
    return returns.dropna().rename("log_return")


def fetch_prices(
    ticker: str = "^GSPC",
    start: str = "1974-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV bars from Yahoo Finance (requires the ``fetch`` extra)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "yfinance is not installed; install with `pip install sp500-arima[fetch]`"
        ) from exc

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel("Ticker")
    raw.columns.name = None
    raw.index = normalize_index(raw.index)
    frame = raw[OHLCV_COLUMNS].copy()
    frame[["Open", "High", "Low", "Close"]] = frame[["Open", "High", "Low", "Close"]].round(2)
    frame["Volume"] = frame["Volume"].astype("int64")
    return pd.DataFrame(frame)


def save_ohlcv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist an OHLCV frame in the snapshot format used by :func:`load_ohlcv`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="Date", date_format="%Y-%m-%d")
    return path
