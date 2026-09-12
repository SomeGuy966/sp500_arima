"""Command-line interface: ``sp500-arima <command> [options]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from sp500_arima import __version__
from sp500_arima.backtest import BacktestConfig, BacktestResult, run_backtest
from sp500_arima.baselines import DriftForecaster, NaiveForecaster
from sp500_arima.data import (
    default_data_path,
    fetch_prices,
    load_prices,
    save_ohlcv,
    slice_prices,
    to_log_returns,
)
from sp500_arima.diagnostics import ljung_box, stationarity_report
from sp500_arima.metrics import interval_coverage, summarize_point_forecasts
from sp500_arima.model import ArimaForecaster, Forecast, Forecaster, ModelConfig, Trend
from sp500_arima.report import (
    build_report,
    cost_sensitivity_table,
    era_table,
    strategy_table,
    summary_tables,
    to_markdown,
)
from sp500_arima.strategy import (
    StrategyConfig,
    compare_strategies,
    cost_sensitivity,
    performance_by_era,
)

matplotlib.use("Agg")

from sp500_arima import plotting

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


# --------------------------------------------------------------------------- helpers
def _model_config(args: argparse.Namespace) -> ModelConfig:
    trends: tuple[Trend, ...] = ("n", "t") if args.trend == "auto" else (args.trend,)
    fixed: tuple[int, int, int] | None = None
    if args.order:
        parts = [int(x) for x in args.order.split(",")]
        if len(parts) != 3:
            raise SystemExit("--order must be three comma-separated integers, e.g. 1,1,0")
        fixed = (parts[0], parts[1], parts[2])
    return ModelConfig(
        max_p=args.max_p,
        max_q=args.max_q,
        d=args.d,
        trends=trends,
        criterion=args.criterion,
        fixed_order=fixed,
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("model")
    group.add_argument(
        "--max-p", type=int, default=3, help="max AR order in the search (default 3)"
    )
    group.add_argument(
        "--max-q", type=int, default=3, help="max MA order in the search (default 3)"
    )
    group.add_argument("--d", type=int, default=1, help="differencing order (default 1)")
    group.add_argument(
        "--trend",
        choices=["auto", "n", "t"],
        default="auto",
        help="'n' none, 't' drift, or 'auto' to let the criterion choose",
    )
    group.add_argument("--criterion", choices=["aic", "bic"], default="aic")
    group.add_argument(
        "--order", default=None, help="skip the search and fit a fixed p,d,q (e.g. 0,1,0)"
    )


def _add_range_args(parser: argparse.ArgumentParser, start: str, end: str) -> None:
    parser.add_argument("--start", default=start, help=f"first date, inclusive (default {start})")
    parser.add_argument("--end", default=end, help=f"last date, exclusive (default {end})")


def _progress(done: int, total: int) -> None:
    width = 30
    filled = int(width * done / total)
    sys.stderr.write(f"\r  folds [{'#' * filled}{'.' * (width - filled)}] {done}/{total}")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- commands
def cmd_fetch(args: argparse.Namespace) -> int:
    """Download a fresh OHLCV snapshot from Yahoo Finance."""
    frame = fetch_prices(args.ticker, start=args.start, end=args.end)
    out = save_ohlcv(frame, args.out)
    print(
        f"Saved {len(frame):,} rows ({frame.index[0].date()} → {frame.index[-1].date()}) to {out}"
    )
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Stationarity tests on log prices and log returns."""
    prices = slice_prices(load_prices(args.data), args.start, args.end)
    log_prices = np.log(prices)
    print(f"\n{len(prices):,} observations, {prices.index[0].date()} → {prices.index[-1].date()}\n")
    print("Stationarity (alpha = 0.05):")
    print(stationarity_report(log_prices).to_string(index=False))
    print("\nLjung-Box on log returns (H0: no autocorrelation):")
    print(ljung_box(to_log_returns(prices)).to_string())
    print(
        "\nInterpretation: log prices are I(1) - difference once (d=1). If Ljung-Box does not\n"
        "reject at conventional levels, there is little linear structure for ARMA terms to\n"
        "exploit, and the random walk is a hard baseline."
    )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """Single train/test split with order selection, diagnostics and a forecast plot."""
    prices = slice_prices(load_prices(args.data), args.start, args.end)
    if args.split:
        split = pd.Timestamp(args.split)
        train, test = prices[prices.index < split], prices[prices.index >= split]
    else:
        cut = int(len(prices) * args.train_frac)
        train, test = prices.iloc[:cut], prices.iloc[cut:]
    if len(train) < 30 or test.empty:
        raise SystemExit("split leaves too little data on one side")

    log_train, log_test = np.log(train), np.log(test)
    arima = ArimaForecaster(_model_config(args)).fit(log_train)
    print(
        f"\nTraining {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days); "
        f"testing {test.index[0].date()} → {test.index[-1].date()} ({len(test)} days)\n"
    )
    print(
        f"Selected {arima.label}  "
        f"(AIC {arima.aic:.1f}, BIC {arima.bic:.1f}, fit {arima.fit_seconds:.2f}s)"
    )
    if arima.selection is not None:
        print("\nTop candidates by", args.criterion.upper())
        print(arima.selection.table.head(8).to_string(index=False))
    if args.verbose:
        print("\n" + arima.summary())
    print("\nLjung-Box on residuals:")
    print(ljung_box(arima.residuals()).to_string())

    forecasters: dict[str, Forecaster] = {
        "arima": arima,
        "naive": NaiveForecaster().fit(log_train),
        "drift": DriftForecaster().fit(log_train),
    }
    rows: list[dict[str, object]] = []
    forecasts: dict[str, Forecast] = {}
    for name, model in forecasters.items():
        fc = model.forecast(test.index, alpha=args.alpha).to_price()
        one = np.exp(model.one_step_ahead(log_test))
        forecasts[name] = fc
        origin = pd.Series(train.iloc[-1], index=test.index)
        previous = test.shift(1).fillna(train.iloc[-1])
        path_metrics = summarize_point_forecasts(test, fc.mean, origin)
        step_metrics = summarize_point_forecasts(test, one, previous)
        rows.append(
            {
                "model": plotting.LABELS[name],
                "path_mape": path_metrics["mape"],
                "path_rmse": path_metrics["rmse"],
                "path_coverage": interval_coverage(test, fc.lower, fc.upper),
                "one_step_mape": step_metrics["mape"],
                "one_step_dir_acc": step_metrics["directional_accuracy"],
            }
        )
    print("\nOut-of-sample accuracy on the test block:")
    print(pd.DataFrame(rows).set_index("model").to_string())

    figures = Path(args.figures)
    title = f"{arima.label} forecast of the S&P 500, {test.index[0]:%b %Y} – {test.index[-1]:%b %Y}"
    plotting.plot_forecast(train, test, forecasts, title, path=figures / "fit_forecast.png")
    plotting.plot_residual_diagnostics(
        arima.residuals(), arima.label, path=figures / "fit_residuals.png"
    )
    print(f"\nFigures written to {figures}/fit_forecast.png and {figures}/fit_residuals.png")
    return 0


def _write_backtest_outputs(result: BacktestResult, args: argparse.Namespace) -> None:
    out = Path(args.out)
    figures_dir = out / "figures"
    strategy_cfg = StrategyConfig(mode=args.mode, cost_bps=args.cost_bps)
    stats, equity = compare_strategies(result.predictions, strategy_cfg)
    summary = result.summary()
    path_table, step_table = summary_tables(summary)

    print("\nPath forecasts (monthly horizon):\n")
    print(path_table)
    print("\nOne-step-ahead forecasts:\n")
    print(step_table)
    print("\nSelected specifications:\n")
    print(to_markdown(result.order_counts().head(8), {"share": ".1%"}))
    print(
        f"\nTrading the one-step signal "
        f"({strategy_cfg.mode}, {strategy_cfg.cost_bps:g} bps/side):\n"
    )
    print(strategy_table(stats))
    sensitivity = cost_sensitivity(result.predictions, mode=strategy_cfg.mode)
    print("\nSharpe ratio by transaction cost:\n")
    print(cost_sensitivity_table(sensitivity))
    eras = performance_by_era(
        result.predictions, cost_bps=strategy_cfg.cost_bps, mode=strategy_cfg.mode
    )
    if not eras.empty:
        print("\nARIMA signal by era:\n")
        print(era_table(eras))

    figures: dict[str, Path] = {}
    if not args.no_figures:
        fold_metrics = result.fold_metrics()
        rel = Path("figures")
        plotting.plot_rolling_error(fold_metrics, path=figures_dir / "rolling_mape.png")
        figures["Rolling monthly MAPE"] = rel / "rolling_mape.png"
        plotting.plot_order_counts(result.order_counts(), path=figures_dir / "order_counts.png")
        figures["Selected specifications"] = rel / "order_counts.png"
        plotting.plot_coverage_by_horizon(
            result.predictions,
            1 - result.config.alpha,
            path=figures_dir / "coverage_by_horizon.png",
        )
        figures["Interval coverage by horizon"] = rel / "coverage_by_horizon.png"
        plotting.plot_equity_curves(equity, path=figures_dir / "equity_curves.png")
        figures["Equity curves"] = rel / "equity_curves.png"
        _plot_latest_fold(result, args, figures_dir)
        figures["Latest fold forecast"] = rel / "latest_fold.png"

    note = (
        f"Position = sign of the one-step forecast return ({strategy_cfg.mode.replace('_', '/')}), "
        f"decided at the prior close, charged {strategy_cfg.cost_bps:g} bps per unit turnover. "
        "Buy-and-hold is evaluated on the same days. Sharpe uses a zero risk-free rate."
    )
    report = build_report(result, stats, note, figures, sensitivity, eras)
    (out / "backtest_summary.md").write_text(report)
    summary.to_csv(out / "backtest_summary.csv")
    stats.to_csv(out / "strategy_stats.csv")
    sensitivity.to_csv(out / "cost_sensitivity.csv")
    eras.to_csv(out / "performance_by_era.csv")
    equity.to_csv(out / "equity_curves.csv")
    print(f"\nReport written to {out / 'backtest_summary.md'}")


def _plot_latest_fold(result: BacktestResult, args: argparse.Namespace, figures_dir: Path) -> None:
    """Re-fit the final fold so the README can show a concrete forecast fan chart."""
    prices = load_prices(args.data)
    last = result.folds.iloc[-1]
    train = prices[(prices.index >= last["train_start"]) & (prices.index <= last["train_end"])]
    test = prices[(prices.index >= last["test_start"]) & (prices.index <= last["test_end"])]
    log_train = np.log(train)
    arima = ArimaForecaster(result.config.model).fit(log_train)
    models: dict[str, Forecaster] = {
        "arima": arima,
        "naive": NaiveForecaster().fit(log_train),
        "drift": DriftForecaster().fit(log_train),
    }
    forecasts = {
        k: m.forecast(test.index, alpha=result.config.alpha).to_price() for k, m in models.items()
    }
    title = f"Final fold: {arima.label} path forecast for {test.index[0]:%B %Y}"
    plotting.plot_forecast(train, test, forecasts, title, path=figures_dir / "latest_fold.png")


def cmd_backtest(args: argparse.Namespace) -> int:
    """Walk-forward evaluation with monthly re-fits."""
    prices = load_prices(args.data)
    config = BacktestConfig(
        start=args.start,
        end=args.end,
        train_window=args.window,
        refit=args.refit,
        alpha=args.alpha,
        model=_model_config(args),
    )
    print(
        f"\nWalk-forward backtest {config.start} → {config.end}, "
        f"window {config.train_window} days, "
        f"refit '{config.refit}', grid p≤{config.model.max_p} q≤{config.model.max_q} "
        f"trend∈{config.model.trends}, jobs={args.jobs or 'all'}"
    )
    result = run_backtest(prices, config, n_jobs=args.jobs, progress=_progress)
    print(f"  {len(result.folds)} folds in {result.elapsed_seconds:.1f}s")
    result.save(args.out)
    _write_backtest_outputs(result, args)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Rebuild tables and figures from a saved backtest without re-running it."""
    result = BacktestResult.load(args.out)
    _write_backtest_outputs(result, args)
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sp500-arima",
        description="ARIMA forecasts of the S&P 500 evaluated against random-walk baselines.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--data", default=None, help=f"price CSV (default {default_data_path()})")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch", help="download a fresh price snapshot (needs the 'fetch' extra)"
    )
    fetch.add_argument("--ticker", default="^GSPC")
    _add_range_args(fetch, "1974-01-01", "")
    fetch.add_argument("--out", default=str(default_data_path()))
    fetch.set_defaults(func=cmd_fetch)

    diagnose = sub.add_parser("diagnose", help="stationarity and autocorrelation tests")
    _add_range_args(diagnose, "2020-01-01", "2025-01-01")
    diagnose.set_defaults(func=cmd_diagnose)

    fit = sub.add_parser("fit", help="fit one model on a train/test split and plot the forecast")
    _add_range_args(fit, "2024-01-01", "2025-01-01")
    fit.add_argument("--split", default=None, help="test-set start date (overrides --train-frac)")
    fit.add_argument("--train-frac", type=float, default=0.8)
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.add_argument("--figures", default="reports/figures")
    fit.add_argument("-v", "--verbose", action="store_true", help="print the statsmodels summary")
    _add_model_args(fit)
    fit.set_defaults(func=cmd_fit)

    backtest = sub.add_parser("backtest", help="walk-forward evaluation with monthly re-fits")
    _add_range_args(backtest, "1975-01-01", "2025-01-01")
    backtest.add_argument("--window", type=int, default=252, help="training window in trading days")
    backtest.add_argument("--refit", choices=["W", "M", "Q"], default="M")
    backtest.add_argument("--alpha", type=float, default=0.05)
    backtest.add_argument("--jobs", type=int, default=0, help="worker processes (0 = all cores)")
    backtest.add_argument("--out", default="reports")
    backtest.add_argument("--no-figures", action="store_true")
    backtest.add_argument(
        "--cost-bps", type=float, default=5.0, help="one-way cost per unit turnover"
    )
    backtest.add_argument("--mode", choices=["long_short", "long_flat"], default="long_short")
    _add_model_args(backtest)
    backtest.set_defaults(func=cmd_backtest)

    report = sub.add_parser("report", help="rebuild the report from saved backtest CSVs")
    report.add_argument("--out", default="reports")
    report.add_argument("--no-figures", action="store_true")
    report.add_argument("--cost-bps", type=float, default=5.0)
    report.add_argument("--mode", choices=["long_short", "long_flat"], default="long_short")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
