"""Merge GDELT features with Brent prices and construct the target variable."""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
HORIZON = 3  # trading days ahead for the target


def build_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Add a binary classification target: will Brent close higher in `horizon` days?

    Args:
        df: DataFrame with a 'brent_close' column, indexed by Date.
        horizon: Number of trading days ahead for the prediction target.

    Returns:
        DataFrame with an added 'target' column (1 = price up, 0 = price down/flat).
        Rows where the target cannot be computed are dropped.
    """
    df = df.copy()
    df["target"] = (df["brent_close"].shift(-horizon) > df["brent_close"]).astype(int)
    # Drop the last `horizon` rows where the target is undefined
    df = df.iloc[:-horizon]
    return df


def build_dataset() -> pd.DataFrame:
    """Merge GDELT features with Brent prices and construct the target.

    Reads:
        - data/processed/gdelt_features.parquet
        - data/raw/brent/brent_raw.parquet

    Returns:
        Final merged DataFrame (also saved to data/processed/final_dataset.parquet).
    """
    gdelt_path = PROCESSED_DIR / "gdelt_features.parquet"
    brent_path = Path(__file__).parents[2] / "data" / "raw" / "brent" / "brent_raw.parquet"

    if not gdelt_path.exists():
        raise FileNotFoundError(
            f"GDELT features not found at {gdelt_path}. Run gdelt_features.py first."
        )
    if not brent_path.exists():
        raise FileNotFoundError(
            f"Brent data not found at {brent_path}. Run download_brent.py first."
        )

    gdelt = pd.read_parquet(gdelt_path)
    brent = pd.read_parquet(brent_path)

    # Align on trading days (inner join keeps only days with both Brent data and GDELT features)
    df = brent.join(gdelt, how="inner")

    df = build_target(df, horizon=HORIZON)
    df = df.dropna()

    out_path = PROCESSED_DIR / "final_dataset.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"Final dataset: {len(df)} rows, {df.shape[1]} columns → {out_path}")
    return df


if __name__ == "__main__":
    build_dataset()
