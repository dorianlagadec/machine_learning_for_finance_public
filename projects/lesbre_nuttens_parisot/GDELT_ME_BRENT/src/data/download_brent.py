"""Download Brent crude oil price data via yfinance."""

from pathlib import Path

import pandas as pd
import yfinance as yf

BRENT_TICKER = "BZ=F"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"


def download_brent(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Fetch Brent crude daily OHLCV data from Yahoo Finance.

    Args:
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with columns [brent_close, brent_volume], indexed by Date.
    """
    raw = yf.download(BRENT_TICKER, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close", "Volume"]].copy()
    df.index.name = "Date"
    df.columns = ["brent_close", "brent_volume"]
    df = df.dropna(subset=["brent_close"])
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parents[2] / "data" / "raw" / "brent"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = download_brent()
    out_path = out_dir / "brent_raw.parquet"
    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows to {out_path}")
