"""Figures for reports and the README.

Colours are assigned to *entities* and never cycled: the same model has the same colour
in every figure. Marks are thin and gridlines recessive so the data carries the chart.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats
from statsmodels.tsa.stattools import acf

from sp500_arima.model import Forecast

INK = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
COLORS: Mapping[str, str] = {
    "actual": INK,
    "train": MUTED,
    "arima": "#2a78d6",
    "naive": "#eb6834",
    "drift": "#1baf7a",
    "buy_hold": "#8a8985",
}
LABELS: Mapping[str, str] = {
    "arima": "ARIMA (auto)",
    "naive": "Random walk",
    "drift": "Random walk + drift",
}

matplotlib.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "legend.frameon": False,
        "font.size": 10,
        "lines.linewidth": 1.6,
    }
)


def _finish(fig: Figure, path: str | Path | None) -> Figure:
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return fig


def color_for(model: str) -> str:
    """Colour assigned to a model or series name."""
    return COLORS.get(model, INK)


def plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecasts: Mapping[str, Forecast],
    title: str,
    context_days: int = 60,
    path: str | Path | None = None,
) -> Figure:
    """Fan chart: recent history, the held-out block, and each model's path forecast.

    ``forecasts`` maps model name to a price-space :class:`Forecast`; the ARIMA interval is
    shaded, the baselines are drawn as lines only.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))
    history = train.iloc[-context_days:]
    ax.plot(history.index, history, color=COLORS["train"], label="Training window (tail)")
    ax.plot(test.index, test, color=COLORS["actual"], label="Actual (held out)")
    # Baselines dashed and drawn first; ARIMA solid and on top, so that a fold where the
    # selected model *is* the random walk still shows both lines.
    ordered = sorted(forecasts.items(), key=lambda kv: kv[0] == "arima")
    for model, fc in ordered:
        colour = color_for(model)
        style = "-" if model == "arima" else "--"
        width = 2.0 if model == "arima" else 1.4
        ax.plot(
            fc.mean.index,
            fc.mean,
            color=colour,
            label=LABELS.get(model, model),
            linestyle=style,
            linewidth=width,
        )
        if model == "arima":
            ax.fill_between(
                fc.lower.index,
                fc.lower,
                fc.upper,
                color=colour,
                alpha=0.14,
                linewidth=0,
                label=f"ARIMA {int((1 - fc.alpha) * 100)}% interval",
            )
    ax.axvline(test.index[0], color=GRID, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("S&P 500 close")
    ax.legend(loc="upper left", ncols=2)
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_rolling_error(
    fold_metrics: pd.DataFrame,
    window: int = 12,
    metric: str = "path_mape",
    path: str | Path | None = None,
) -> Figure:
    """Plot the rolling monthly error per model and the ARIMA-minus-random-walk gap."""
    wide = fold_metrics.pivot(index="test_start", columns="model", values=metric).sort_index()
    rolling = wide.rolling(window, min_periods=window).mean()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    for model in [m for m in ("arima", "naive", "drift") if m in rolling]:
        top.plot(rolling.index, rolling[model], color=color_for(model), label=LABELS[model])
    years = (wide.index[-1] - wide.index[0]).days / 365.25
    top.set_ylabel(f"{window}-month rolling MAPE (%)")
    top.set_title(
        f"Monthly path-forecast error, {wide.index[0]:%Y}–{wide.index[-1]:%Y} ({years:.0f} years)"
    )
    top.legend(loc="upper right", ncols=3)

    if {"arima", "naive"} <= set(wide.columns):
        gap = (wide["arima"] - wide["naive"]).rolling(window, min_periods=window).mean()
        bottom.axhline(0, color=INK, linewidth=0.8)
        bottom.fill_between(
            gap.index, gap, 0, where=gap >= 0, color=COLORS["naive"], alpha=0.5, linewidth=0
        )
        bottom.fill_between(
            gap.index, gap, 0, where=gap < 0, color=COLORS["arima"], alpha=0.5, linewidth=0
        )
        bottom.set_ylabel("ARIMA − random walk\n(MAPE pts)")
        bottom.text(
            0.01,
            0.92,
            "below zero = ARIMA more accurate",
            transform=bottom.transAxes,
            fontsize=9,
            color="#52514e",
            va="top",
        )
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_order_counts(
    order_counts: pd.DataFrame, top_n: int = 12, path: str | Path | None = None
) -> Figure:
    """Horizontal bars of how often each specification won the AIC search."""
    counts = order_counts.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(counts) + 1.4))
    bars = ax.barh(counts.index, counts["folds"], color=COLORS["arima"], height=0.62)
    for bar, share in zip(bars, counts["share"], strict=True):
        ax.text(
            bar.get_width() + max(counts["folds"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{share:.0%}",
            va="center",
            fontsize=9,
            color="#52514e",
        )
    ax.set_xlabel("Months selected")
    ax.set_title("Which specification does AIC pick on a rolling 1-year window?")
    ax.grid(axis="y", visible=False)
    return _finish(fig, path)


def plot_coverage_by_horizon(
    predictions: pd.DataFrame,
    nominal: float = 0.95,
    max_step: int = 22,
    path: str | Path | None = None,
) -> Figure:
    """Empirical prediction-interval coverage at each forecast step, per model."""
    block = predictions[predictions["step"] <= max_step]
    inside = (block["actual"] >= block["path_lower"]) & (block["actual"] <= block["path_upper"])
    coverage = inside.groupby([block["model"], block["step"]]).mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(nominal, color=INK, linewidth=0.8, linestyle="--", label=f"Nominal {nominal:.0%}")
    for model in [m for m in ("arima", "naive", "drift") if m in coverage]:
        ax.plot(
            coverage.index,
            coverage[model],
            color=color_for(model),
            marker="o",
            markersize=3.5,
            label=LABELS[model],
        )
    ax.set_xlabel("Forecast horizon (trading days ahead)")
    ax.set_ylabel("Share of actuals inside interval")
    low, high = float(coverage.min().min()), float(coverage.max().max())
    ax.set_ylim(low - 0.02, max(nominal, high) + 0.015)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Are the 95% intervals honest? Coverage by horizon")
    ax.legend(loc="lower left", ncols=2)
    return _finish(fig, path)


def plot_equity_curves(equity: pd.DataFrame, path: str | Path | None = None) -> Figure:
    """Cumulative growth of 1.0 for each strategy on a log scale."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    finals: list[tuple[float, str, str]] = []
    for column in equity.columns:
        key = "buy_hold" if column.lower().startswith("buy") else column.split()[0]
        ax.plot(equity.index, equity[column], color=color_for(key), label=column)
        finals.append((float(equity[column].iloc[-1]), column, color_for(key)))
    # End-of-line labels, nudged apart in log space so identical finishes stay legible.
    finals.sort()
    spread = np.log10(max(f[0] for f in finals) / max(min(f[0] for f in finals), 1e-9)) or 1.0
    min_gap = max(0.05 * spread, 0.03)
    log_positions = [np.log10(v) for v, _, _ in finals]
    for i in range(1, len(log_positions)):
        log_positions[i] = max(log_positions[i], log_positions[i - 1] + min_gap)
    for (value, _, colour), log_y in zip(finals, log_positions, strict=True):
        ax.annotate(
            f"{value:.1f}×",
            (equity.index[-1], 10**log_y),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=9,
            color=colour,
            va="center",
        )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title("Daily sign-of-forecast strategies vs. buy-and-hold (net of costs)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    return _finish(fig, path)


def plot_residual_diagnostics(
    residuals: pd.Series, label: str, lags: int = 30, path: str | Path | None = None
) -> Figure:
    """ACF, histogram against a normal density, and QQ plot of model residuals."""
    r = residuals.dropna().to_numpy(float)
    fig, (ax_acf, ax_hist, ax_qq) = plt.subplots(1, 3, figsize=(12, 3.8))

    rho = acf(r, nlags=lags, fft=True)
    bound = 1.96 / np.sqrt(len(r))
    ax_acf.bar(range(1, lags + 1), rho[1:], color=COLORS["arima"], width=0.6)
    ax_acf.axhspan(-bound, bound, color=GRID, alpha=0.8, linewidth=0)
    ax_acf.axhline(0, color=INK, linewidth=0.8)
    ax_acf.set_title("Residual ACF")
    ax_acf.set_xlabel("Lag")

    z = (r - r.mean()) / r.std(ddof=1)
    ax_hist.hist(z, bins=50, density=True, color=COLORS["arima"], alpha=0.85)
    grid = np.linspace(-5, 5, 200)
    ax_hist.plot(grid, stats.norm.pdf(grid), color=INK, linewidth=1.2, label="N(0,1)")
    ax_hist.set_xlim(-5, 5)
    ax_hist.set_title("Standardised residuals")
    ax_hist.legend(loc="upper right")

    (osm, osr), (slope, intercept, _) = stats.probplot(z, dist="norm")
    ax_qq.scatter(osm, osr, s=8, color=COLORS["arima"], alpha=0.7, linewidths=0)
    ax_qq.plot(osm, slope * np.asarray(osm) + intercept, color=INK, linewidth=1.0)
    ax_qq.set_title("Normal QQ plot")
    ax_qq.set_xlabel("Theoretical quantiles")
    ax_qq.set_ylabel("Sample quantiles")

    fig.suptitle(f"{label} residual diagnostics", fontsize=12, fontweight="bold")
    return _finish(fig, path)
