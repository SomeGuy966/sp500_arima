from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def simulate_prices(
    n: int = 600,
    phi: float = 0.0,
    mu: float = 0.0003,
    sigma: float = 0.01,
    seed: int = 0,
    start: str = "2015-01-01",
) -> pd.Series:
    """Geometric random walk with AR(1) log returns, on a business-day calendar."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * sigma
    r = np.empty(n)
    r[0] = mu + eps[0]
    for t in range(1, n):
        r[t] = mu + phi * (r[t - 1] - mu) + eps[t]
    log_prices = np.log(100.0) + np.cumsum(r)
    index = pd.bdate_range(start, periods=n, name="Date")
    return pd.Series(np.exp(log_prices), index=index, name="Close")


@pytest.fixture(scope="session")
def random_walk_prices() -> pd.Series:
    return simulate_prices(phi=0.0, seed=1)


@pytest.fixture(scope="session")
def ar1_prices() -> pd.Series:
    """Strong, unrealistic AR(1) structure so order selection has something to find."""
    return simulate_prices(n=1500, phi=0.6, mu=0.0, seed=2)


@pytest.fixture(scope="session")
def real_prices() -> pd.Series:
    from sp500_arima.data import load_prices

    return load_prices()
