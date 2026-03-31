"""Visualization helpers for the GDELT × Brent project."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).parents[2] / "notebooks" / "figures"

# Notable geopolitical events for annotation
NOTABLE_EVENTS = {
    "2019-09-14": "Aramco attack",
    "2020-01-03": "Soleimani killing",
    "2020-03-09": "COVID price collapse",
    "2022-02-24": "Ukraine invasion",
    "2023-10-07": "Israel-Gaza conflict",
}


def plot_brent_with_gdelt(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot Brent close price alongside the GDELT Goldstein score.

    Args:
        df: Final dataset with 'brent_close' and 'goldstein_mean' columns.
        save: If True, save the figure to the figures directory.

    Returns:
        Matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(df.index, df["brent_close"], color="steelblue", linewidth=1)
    ax1.set_ylabel("Brent Close (USD)")
    ax1.set_title("Brent Crude Oil Price vs GDELT Geopolitical Tension")

    ax2.plot(df.index, df["goldstein_mean"], color="firebrick", linewidth=0.8, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Goldstein Score (mean)")
    ax2.set_xlabel("Date")

    for date_str, label in NOTABLE_EVENTS.items():
        dt = pd.Timestamp(date_str)
        if dt in df.index or (df.index.min() <= dt <= df.index.max()):
            for ax in (ax1, ax2):
                ax.axvline(dt, color="orange", linewidth=0.8, linestyle=":")
            ax2.annotate(
                label,
                xy=(dt, df["goldstein_mean"].get(dt, 0)),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=7,
                color="darkorange",
            )

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "brent_vs_gdelt.png", dpi=150)
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    model_name: str = "Random Forest",
    save: bool = True,
) -> plt.Figure:
    """Bar chart of feature importances.

    Args:
        feature_names: List of feature column names.
        importances: Array of importance scores from the fitted model.
        model_name: Label for the plot title.
        save: If True, save the figure.

    Returns:
        Matplotlib Figure object.
    """
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(importances)), importances[idx], color="steelblue")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            FIGURES_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png",
            dpi=150,
        )
    return fig


def plot_cumulative_returns(
    df: pd.DataFrame,
    predictions: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """Compare cumulative returns of the model signal vs buy-and-hold.

    Args:
        df: DataFrame with 'brent_close' column, indexed by Date.
        predictions: Binary predictions aligned with df's index (1=long, 0=short/flat).
        save: If True, save the figure.

    Returns:
        Matplotlib Figure object.
    """
    daily_ret = df["brent_close"].pct_change().shift(-1)  # next-day return
    strategy_ret = daily_ret * (predictions * 2 - 1)      # long or short

    cumulative_bh = (1 + daily_ret).cumprod()
    cumulative_strat = (1 + strategy_ret).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative_bh.index, cumulative_bh, label="Buy & Hold", color="steelblue")
    ax.plot(cumulative_strat.index, cumulative_strat, label="Model Signal", color="firebrick")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Backtested Strategy vs Buy & Hold")
    ax.legend()
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "cumulative_returns.png", dpi=150)
    return fig


def plot_cv_accuracy(results_by_model: dict, save: bool = True) -> plt.Figure:
    """Box plot comparing CV accuracy across models.

    Args:
        results_by_model: Dict mapping model name to list of per-fold result dicts.
        save: If True, save the figure.

    Returns:
        Matplotlib Figure object.
    """
    data = {
        name: [r["accuracy"] for r in folds]
        for name, folds in results_by_model.items()
    }
    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot.boxplot(ax=ax)
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Validation Accuracy by Model")
    ax.axhline(0.5, color="gray", linestyle="--", label="50% baseline")
    ax.legend()
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "cv_accuracy.png", dpi=150)
    return fig


def plot_feature_correlation(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Correlation heatmap between GDELT features and the target variable.

    Args:
        df: Final dataset with GDELT features and 'target' column.
            'brent_close' and 'brent_volume' are excluded from the matrix.
        save: If True, save the figure.

    Returns:
        Matplotlib Figure object.
    """
    exclude = {"brent_close", "brent_volume"}
    feature_cols = [c for c in df.columns if c not in exclude]
    corr = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Feature Correlation Matrix (incl. target)")
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "feature_correlation.png", dpi=150)
    return fig
