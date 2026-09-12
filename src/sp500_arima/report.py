"""Markdown rendering of backtest results (no ``tabulate`` dependency)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sp500_arima.backtest import BacktestResult, describe_config


def to_markdown(
    frame: pd.DataFrame, formats: Mapping[str, str] | None = None, index: bool = True
) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    ``formats`` maps column name to a format spec (e.g. ``".2f"`` or ``".1%"``).
    """
    formats = formats or {}
    table = frame.reset_index() if index else frame.copy()
    columns = [str(c) for c in table.columns]

    def cell(column: str, value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        spec = formats.get(column)
        if spec and isinstance(value, int | float):
            return format(value, spec)
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(columns))) + "|"
    body = [
        "| " + " | ".join(cell(col, table.iloc[i][col]) for col in columns) + " |"
        for i in range(len(table))
    ]
    return "\n".join([header, rule, *body])


SUMMARY_COLUMNS = {
    "label": "Model",
    "path_mape": "MAPE %",
    "path_mape_monthly_mean": "Mean monthly MAPE %",
    "path_rmse_pct": "RMSE %",
    "path_dir_acc": "Direction",
    "path_coverage": "95% PI coverage",
    "path_dm_p": "DM p vs RW",
}
ONE_STEP_COLUMNS = {
    "label": "Model",
    "one_step_mape": "MAPE %",
    "one_step_rmse_pct": "RMSE %",
    "one_step_dir_acc": "Direction",
    "one_step_dm_p": "DM p vs RW",
}
STRATEGY_COLUMNS = {
    "annual_return": "CAGR",
    "annual_vol": "Vol",
    "sharpe": "Sharpe",
    "max_drawdown": "Max DD",
    "hit_rate": "Hit rate",
    "turnover": "Turnover/day",
    "time_in_market": "Time in mkt",
}
FORMATS = {
    "MAPE %": ".2f",
    "Mean monthly MAPE %": ".2f",
    "RMSE %": ".2f",
    "Direction": ".1%",
    "95% PI coverage": ".1%",
    "DM p vs RW": ".3f",
    "CAGR": ".1%",
    "Vol": ".1%",
    "Sharpe": ".2f",
    "Max DD": ".1%",
    "Hit rate": ".1%",
    "Turnover/day": ".2f",
    "Time in mkt": ".0%",
    "share": ".1%",
}


def summary_tables(summary: pd.DataFrame) -> tuple[str, str]:
    """Markdown tables for the path-forecast and one-step-ahead horizons."""
    path = summary[[c for c in SUMMARY_COLUMNS if c in summary]].rename(columns=SUMMARY_COLUMNS)
    step = summary[[c for c in ONE_STEP_COLUMNS if c in summary]].rename(columns=ONE_STEP_COLUMNS)
    return (
        to_markdown(path, FORMATS, index=False),
        to_markdown(step, FORMATS, index=False),
    )


def strategy_table(stats: pd.DataFrame) -> str:
    """Markdown table of strategy performance."""
    view = stats[list(STRATEGY_COLUMNS)].rename(columns=STRATEGY_COLUMNS)
    view.index.name = "Strategy"
    return to_markdown(view, FORMATS)


def cost_sensitivity_table(sensitivity: pd.DataFrame) -> str:
    """Markdown table of Sharpe ratios by strategy and transaction-cost level."""
    sharpe = pd.DataFrame(sensitivity["sharpe"])
    sharpe.columns = [f"{c:g} bps" for c in sharpe.columns]
    sharpe.index.name = "Sharpe by cost"
    return to_markdown(sharpe, dict.fromkeys(sharpe.columns, ".2f"))


ERA_COLUMNS = {
    "days": "Days",
    "lag1_autocorr": "Lag-1 autocorr",
    "dir_acc": "Direction",
    "sharpe_gross": "Sharpe (gross)",
    "sharpe_net": "Sharpe (net)",
    "sharpe_buy_hold": "Sharpe (buy & hold)",
}


def era_table(eras: pd.DataFrame) -> str:
    """Markdown table of signal performance by era."""
    view = eras[list(ERA_COLUMNS)].rename(columns=ERA_COLUMNS)
    view.index.name = "Era"
    formats = {
        "Lag-1 autocorr": "+.3f",
        "Direction": ".1%",
        "Sharpe (gross)": ".2f",
        "Sharpe (net)": ".2f",
        "Sharpe (buy & hold)": ".2f",
    }
    return to_markdown(view, formats)


def build_report(
    result: BacktestResult,
    strategy_stats: pd.DataFrame,
    strategy_note: str,
    figures: Mapping[str, Path] | None = None,
    sensitivity: pd.DataFrame | None = None,
    eras: pd.DataFrame | None = None,
) -> str:
    """Assemble the full Markdown report for a backtest run."""
    summary = result.summary()
    path_table, step_table = summary_tables(summary)
    orders = result.order_counts().head(10)
    orders.index.name = "Specification"
    cfg = describe_config(result.config)
    model_cfg = cfg["model"]
    assert isinstance(model_cfg, dict)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    n_folds = int(result.folds.shape[0])
    first, last = result.folds["test_start"].iloc[0], result.folds["test_end"].iloc[-1]
    lines = [
        "# Walk-forward backtest summary",
        "",
        f"Generated {generated} · {n_folds} monthly folds from {first:%Y-%m} to {last:%Y-%m} · "
        f"{int(result.predictions['date'].nunique())} trading days · "
        f"{result.elapsed_seconds:.0f}s wall time"
        if pd.notna(result.elapsed_seconds)
        else f"Generated {generated} · {n_folds} monthly folds from {first:%Y-%m} to {last:%Y-%m}",
        "",
        "## Configuration",
        "",
        f"- Training window: **{cfg['train_window']} trading days**, "
        f"re-fit every **{cfg['refit']}**",
        f"- Order search: p ≤ {model_cfg['max_p']}, q ≤ {model_cfg['max_q']}, "
        f"d = {model_cfg['d']}, trend ∈ {{{', '.join(model_cfg['trends'])}}}, "
        f"scored by **{str(model_cfg['criterion']).upper()}**",
        f"- Prediction intervals at **{(1 - result.config.alpha):.0%}**",
        "",
        "## Path forecasts (whole month from the last training day)",
        "",
        path_table,
        "",
        "*Direction* is the sign agreement between forecast and realised month-over-origin "
        "return. *DM p* is the two-sided Diebold-Mariano p-value against the driftless random "
        "walk on monthly squared log errors; small values mean the accuracy difference is "
        "unlikely to be noise.",
        "",
        "## One-step-ahead forecasts (daily, parameters frozen between re-fits)",
        "",
        step_table,
        "",
        "The random walk forecasts a zero return, so its direction metric is undefined.",
        "",
        "## Selected specifications",
        "",
        to_markdown(orders, FORMATS),
        "",
        "## Trading the one-step signal",
        "",
        strategy_note,
        "",
        strategy_table(strategy_stats),
        "",
    ]
    if sensitivity is not None:
        lines += [
            "### Sensitivity to transaction costs",
            "",
            cost_sensitivity_table(sensitivity),
            "",
        ]
    if eras is not None and not eras.empty:
        lines += [
            "### By era",
            "",
            "Lag-1 autocorrelation of daily log returns is the structure a low-order ARIMA "
            "exploits; gross Sharpe is before costs, net is after.",
            "",
            era_table(eras),
            "",
        ]
    if figures:
        lines += ["## Figures", ""]
        for caption, fig_path in figures.items():
            lines.append(f"![{caption}]({fig_path.as_posix()})")
            lines.append("")
    return "\n".join(lines)
