from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sp500_arima.backtest import BacktestConfig, run_backtest
from sp500_arima.cli import build_parser, main
from sp500_arima.data import save_ohlcv
from sp500_arima.model import ModelConfig
from sp500_arima.report import build_report, to_markdown
from sp500_arima.strategy import compare_strategies


@pytest.fixture
def csv_path(tmp_path: Path, random_walk_prices: pd.Series) -> Path:
    frame = pd.DataFrame({"Close": random_walk_prices})
    frame["Open"] = frame["High"] = frame["Low"] = frame["Close"]
    frame["Volume"] = 1
    return save_ohlcv(frame, tmp_path / "prices.csv")


def test_parser_has_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest", "--max-p", "1", "--order", "0,1,0", "--jobs", "1"])
    assert args.command == "backtest"
    assert args.order == "0,1,0"


def test_cli_diagnose(csv_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["--data", str(csv_path), "diagnose", "--start", "2015-01-01", "--end", "2016-01-01"])
        == 0
    )
    out = capsys.readouterr().out
    assert "ADF" in out and "KPSS" in out and "Ljung-Box" in out


def test_cli_fit_writes_figures(
    csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    figures = tmp_path / "figs"
    code = main(
        [
            "--data",
            str(csv_path),
            "fit",
            "--start",
            "2015-01-01",
            "--end",
            "2016-01-01",
            "--max-p",
            "1",
            "--max-q",
            "1",
            "--trend",
            "n",
            "--figures",
            str(figures),
            "-v",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Selected ARIMA(" in out
    assert "Out-of-sample accuracy" in out
    assert (figures / "fit_forecast.png").exists()
    assert (figures / "fit_residuals.png").exists()


def test_cli_fit_with_fixed_order_and_split(csv_path: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--data",
            str(csv_path),
            "fit",
            "--start",
            "2015-01-01",
            "--end",
            "2016-01-01",
            "--split",
            "2015-11-01",
            "--order",
            "0,1,0",
            "--trend",
            "t",
            "--figures",
            str(tmp_path),
        ]
    )
    assert code == 0


def test_cli_bad_order_exits(csv_path: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--data", str(csv_path), "fit", "--order", "1,1", "--figures", str(tmp_path)])


def test_cli_errors_are_reported_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--data", str(tmp_path / "missing.csv"), "diagnose"])
    assert excinfo.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_backtest_and_report(
    csv_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "reports"
    code = main(
        [
            "--data",
            str(csv_path),
            "backtest",
            "--start",
            "2016-01-01",
            "--end",
            "2016-04-01",
            "--window",
            "100",
            "--max-p",
            "1",
            "--max-q",
            "0",
            "--trend",
            "n",
            "--jobs",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Path forecasts" in printed and "Random walk" in printed
    for name in (
        "backtest_predictions.csv",
        "backtest_folds.csv",
        "backtest_summary.md",
        "backtest_summary.csv",
        "strategy_stats.csv",
        "equity_curves.csv",
        "cost_sensitivity.csv",
        "performance_by_era.csv",
    ):
        assert (out / name).exists(), name
    for fig in (
        "rolling_mape",
        "order_counts",
        "coverage_by_horizon",
        "equity_curves",
        "latest_fold",
    ):
        assert (out / "figures" / f"{fig}.png").exists(), fig
    report = (out / "backtest_summary.md").read_text()
    assert "## Path forecasts" in report and "![" in report

    # rebuild from the saved CSVs without re-running
    assert (
        main(
            [
                "--data",
                str(csv_path),
                "report",
                "--out",
                str(out),
                "--no-figures",
                "--mode",
                "long_flat",
            ]
        )
        == 0
    )
    assert "long/flat" in capsys.readouterr().out


def test_report_markdown_helpers(random_walk_prices: pd.Series) -> None:
    config = BacktestConfig(
        start="2016-01-01",
        end="2016-03-01",
        train_window=100,
        model=ModelConfig(max_p=1, max_q=0, trends=("n",)),
    )
    result = run_backtest(random_walk_prices, config, n_jobs=1)
    stats, _ = compare_strategies(result.predictions)
    text = build_report(result, stats, "note", {"Figure": Path("figures/x.png")})
    assert text.startswith("# Walk-forward backtest summary")
    assert "| Model |" in text and "![Figure](figures/x.png)" in text

    frame = pd.DataFrame({"a": [1.23456, float("nan")], "b": ["x", pd.Timestamp("2020-01-02")]})
    md = to_markdown(frame, {"a": ".2f"}, index=False)
    assert md.splitlines()[0] == "| a | b |"
    assert "1.23" in md and "—" in md and "2020-01-02" in md
