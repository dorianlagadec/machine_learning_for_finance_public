"""Download and filter GDELT 1.0 event data for Middle East countries."""

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# CAMEO country codes for the Middle East region
ME_COUNTRY_CODES = {"IRN", "IRQ", "SAU", "ISR", "PSE", "YEM", "SYR", "ARE", "KWT"}

# GDELT 1.0 column names (header not included in files)
GDELT_COLUMNS = [
    "GlobalEventID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
    "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat",
    "Actor1Geo_Long", "Actor1Geo_FeatureID", "Actor2Geo_Type",
    "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL",
]

GDELT_INDEX_URL = "http://data.gdeltproject.org/events/index.html"
GDELT_BASE_URL = "http://data.gdeltproject.org/events/"

USEFUL_COLS = [
    "SQLDATE", "Actor1CountryCode", "Actor2CountryCode",
    "EventCode", "EventBaseCode", "GoldsteinScale",
    "NumMentions", "NumArticles", "AvgTone", "ActionGeo_CountryCode",
]


def download_day(date: str) -> pd.DataFrame:
    """Download and filter GDELT events for a single day.

    Args:
        date: Date string in YYYYMMDD format.

    Returns:
        Filtered DataFrame with only Middle East events and useful columns.
        Empty DataFrame if the file is not found.
    """
    url = f"{GDELT_BASE_URL}{date}.export.CSV.zip"
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        logger.warning("No GDELT file found for %s (HTTP %s)", date, response.status_code)
        return pd.DataFrame(columns=USEFUL_COLS)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=GDELT_COLUMNS,
                dtype=str,
                low_memory=False,
            )

    df = df[USEFUL_COLS].copy()

    # Keep only rows involving the Middle East
    me_mask = (
        df["Actor1CountryCode"].isin(ME_COUNTRY_CODES)
        | df["Actor2CountryCode"].isin(ME_COUNTRY_CODES)
        | df["ActionGeo_CountryCode"].isin(ME_COUNTRY_CODES)
    )
    return df[me_mask].reset_index(drop=True)


def download_range(start: str, end: str, output_dir: Path) -> None:
    """Download GDELT files for a date range and save filtered CSVs.

    Args:
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.
        output_dir: Directory where per-day CSV files are saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start=start, end=end, freq="D")

    for dt in dates:
        date_str = dt.strftime("%Y%m%d")
        out_path = output_dir / f"{date_str}.csv"

        if out_path.exists():
            logger.info("Skipping %s (already downloaded)", date_str)
            continue

        logger.info("Downloading %s", date_str)
        try:
            df = download_day(date_str)
        except requests.RequestException as exc:
            logger.warning("Failed to download %s: %s", date_str, exc)
            continue
        df.to_csv(out_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_dir = Path(__file__).parents[2] / "data" / "raw" / "gdelt"
    download_range("2015-01-01", "2024-12-31", raw_dir)
