"""Aggregate raw GDELT events into daily features for the Middle East."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_GDELT_DIR = Path(__file__).parents[2] / "data" / "raw" / "gdelt"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

# CAMEO event codes associated with conflict / violence
CONFLICT_PREFIXES = ("18", "19", "20")


def _is_conflict(event_code: pd.Series) -> pd.Series:
    """Return boolean mask for conflict-type CAMEO codes (18x, 19x, 20x)."""
    return event_code.astype(str).str[:2].isin(CONFLICT_PREFIXES)


def aggregate_day(df: pd.DataFrame) -> dict:
    """Compute daily feature aggregates from a single day's GDELT events.

    Args:
        df: Filtered GDELT DataFrame for one day (Middle East events only).

    Returns:
        Dictionary of feature values for that day.
    """
    df = df.copy()
    df["GoldsteinScale"] = pd.to_numeric(df["GoldsteinScale"], errors="coerce")
    df["NumMentions"] = pd.to_numeric(df["NumMentions"], errors="coerce")
    df["NumArticles"] = pd.to_numeric(df["NumArticles"], errors="coerce")
    df["AvgTone"] = pd.to_numeric(df["AvgTone"], errors="coerce")

    conflict_mask = _is_conflict(df["EventCode"])
    n_events = len(df)
    n_conflict = conflict_mask.sum()

    return {
        "n_events": n_events,
        "n_conflict_events": int(n_conflict),
        "conflict_ratio": n_conflict / n_events if n_events > 0 else np.nan,
        "goldstein_mean": df["GoldsteinScale"].mean(),
        "goldstein_min": df["GoldsteinScale"].min(),
        "avg_tone": df["AvgTone"].mean(),
        "n_mentions": df["NumMentions"].sum(),
        "n_articles": df["NumArticles"].sum(),
    }


def build_gdelt_features(raw_dir: Path = RAW_GDELT_DIR) -> pd.DataFrame:
    """Load all per-day GDELT CSV files and produce a daily feature DataFrame.

    Args:
        raw_dir: Directory containing per-day filtered GDELT CSV files.

    Returns:
        DataFrame indexed by Date with one row per calendar day.
    """
    records = {}
    csv_files = sorted(raw_dir.glob("*.csv"))

    for path in csv_files:
        date_str = path.stem  # YYYYMMDD
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", path.name, exc)
            continue

        if df.empty:
            continue

        records[date_str] = aggregate_day(df)

    features = pd.DataFrame.from_dict(records, orient="index")
    features.index = pd.to_datetime(features.index, format="%Y%m%d")
    features.index.name = "Date"
    features = features.sort_index()

    features = _add_derived_features(features)
    return features


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling and spike features to the daily GDELT feature DataFrame.

    Args:
        df: DataFrame with base daily features.

    Returns:
        DataFrame with additional derived columns.
    """
    df = df.copy()

    # Note: 7-day calendar rolling — may cover fewer than 7 observations
    # if GDELT has gaps (e.g., weekends with no events)
    df["goldstein_7d_ma"] = df["goldstein_mean"].rolling(7, min_periods=3).mean()

    # Binary spike flag: current Goldstein drops > 1.5 std below 7-day average
    rolling_std = df["goldstein_mean"].rolling(7, min_periods=3).std()
    df["tension_spike"] = (
        df["goldstein_mean"] < df["goldstein_7d_ma"] - 1.5 * rolling_std
    ).astype(int)

    # 7-day rolling mean of mentions
    df["mentions_7d_ma"] = df["n_mentions"].rolling(7, min_periods=3).mean()

    # Z-score of mentions relative to 7-day window
    rolling_mentions_std = df["n_mentions"].rolling(7, min_periods=3).std()
    df["mentions_zscore"] = (
        (df["n_mentions"] - df["mentions_7d_ma"]) / rolling_mentions_std
    )

    return df


if __name__ == "__main__":
    features = build_gdelt_features()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "gdelt_features.parquet"
    features.to_parquet(out_path)
    print(f"Saved GDELT features: {len(features)} rows → {out_path}")
