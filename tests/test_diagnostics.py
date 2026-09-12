from __future__ import annotations

import numpy as np
import pandas as pd

from sp500_arima.diagnostics import (
    adf_test,
    choose_differencing,
    kpss_test,
    ljung_box,
    stationarity_report,
)


def test_random_walk_is_i1(random_walk_prices: pd.Series) -> None:
    log_prices = np.log(random_walk_prices).rename("log_price")
    assert not adf_test(log_prices).stationary
    assert not kpss_test(log_prices).stationary
    returns = log_prices.diff().dropna().rename("log_return")
    assert adf_test(returns).stationary
    assert kpss_test(returns).stationary
    assert choose_differencing(log_prices) == 1


def test_stationarity_report_shape(random_walk_prices: pd.Series) -> None:
    table = stationarity_report(np.log(random_walk_prices))
    assert list(table["series"]) == ["log_price", "log_price", "log_return", "log_return"]
    assert list(table["test"]) == ["ADF", "KPSS", "ADF", "KPSS"]
    assert set(table["conclusion"]) <= {"stationary", "non-stationary"}
    row = adf_test(np.log(random_walk_prices)).to_row()
    assert row["null"] == "unit root"


def test_ljung_box_flags_autocorrelation(
    ar1_prices: pd.Series, random_walk_prices: pd.Series
) -> None:
    ar_returns = np.log(ar1_prices).diff().dropna()
    table = ljung_box(ar_returns, lags=(5, 10))
    assert list(table.index) == [5, 10]
    assert (table["p_value"] < 0.01).all()
    white = np.log(random_walk_prices).diff().dropna()
    # seeded fixture: no autocorrelation at the 1% level (0.0495 at lag 10 by draw)
    assert ljung_box(white, lags=(10,))["p_value"].iloc[0] > 0.01


def test_choose_differencing_on_stationary_series() -> None:
    rng = np.random.default_rng(0)
    series = pd.Series(rng.standard_normal(400), index=pd.bdate_range("2020-01-01", periods=400))
    assert choose_differencing(series) == 0
