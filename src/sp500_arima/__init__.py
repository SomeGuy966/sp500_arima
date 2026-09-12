"""ARIMA forecasting of the S&P 500, evaluated honestly against random-walk baselines."""

from sp500_arima.backtest import BacktestConfig, BacktestResult, run_backtest
from sp500_arima.baselines import DriftForecaster, NaiveForecaster
from sp500_arima.data import load_prices, to_log_returns
from sp500_arima.model import ArimaForecaster, Forecast, ModelConfig, select_order

__version__ = "1.0.0"

__all__ = [
    "ArimaForecaster",
    "BacktestConfig",
    "BacktestResult",
    "DriftForecaster",
    "Forecast",
    "ModelConfig",
    "NaiveForecaster",
    "__version__",
    "load_prices",
    "run_backtest",
    "select_order",
    "to_log_returns",
]
