"""
Fetches Walmart stock prices and Consumer Confidence Index from public sources.
"""

import yfinance as yf
import pandas as pd

from pytrends.request import TrendReq

def fetch_walmart_stock(start="2012-09-01", end="2013-04-30"):
    """Fetch daily Walmart (WMT) closing prices from Yahoo Finance."""
    wmt = yf.Ticker("WMT")
    daily_prices = wmt.history(start=start, end=end, interval="1d")[["Close"]]
    daily_prices.index = pd.to_datetime(daily_prices.index).tz_localize(None)
    daily_prices.index.name = "date"
    return daily_prices


def fetch_consumer_confidence(start="2012-09-01", end="2013-04-30"):
    """Load Consumer Confidence Index CSV and upsample to daily.

    The CSV is expected at `data/consumer_confidence.csv` and to contain the
    columns `DATE` and `UMCSENT`.
    """
    cci = pd.read_csv("data/consumer_confidence.csv")

    # The expected column name is `DATE`, but some exports may use
    # `observation_date`. Support both.
    date_col = "DATE" if "DATE" in cci.columns else "observation_date"
    if date_col not in cci.columns:
        raise ValueError(
            "consumer_confidence.csv must contain a date column named `DATE` "
            "or `observation_date`."
        )

    cci[date_col] = pd.to_datetime(cci[date_col])
    cci = cci.set_index(date_col)
    cci = cci.rename(columns={"UMCSENT": "consumer_confidence"})
    cci["consumer_confidence"] = pd.to_numeric(
        cci["consumer_confidence"], errors="coerce"
    )

    # Weekly observations: take the first value in each week, then interpolate
    # linearly on the weekly series before carrying forward to daily.
    cci_weekly = cci.resample("W").first()
    cci_weekly = cci_weekly.interpolate(method="linear")
    cci = cci_weekly.resample("D").ffill()

    # Keep the requested date range.
    cci = cci.loc[start:end]
    cci.index.name = "date"
    return cci


KEYWORDS = ["Walmart", "grocery store", "discount", "inflation"]


def fetch_google_trends(
    keywords=None,
    start_date="2012-09-01",
    end_date="2013-04-30",
):
    """
    Fetch weekly Google Trends interest scores and upsample to daily for the provided keywords.

    Parameters
    ----------
    keywords : list[str] | None
        List of Google Trends search terms to fetch. If None, default keywords
        are used.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pandas.DataFrame
        Daily DataFrame indexed by date with one column per keyword and a
        composite `consumer_search_index` column equal to the simple average
        of all keyword scores.
    """
    kw_list = keywords or KEYWORDS
    timeframe = f"{start_date} {end_date}"

    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(kw_list=kw_list, timeframe=timeframe, geo="")
    trends = pytrends.interest_over_time()

    if trends.empty:
        columns = list(kw_list) + ["consumer_search_index"]
        daily_idx = pd.date_range(start=start_date, end=end_date, freq="D")
        empty_df = pd.DataFrame(index=daily_idx, columns=columns)
        empty_df.index.name = "date"
        return empty_df

    trends = trends.drop(columns=["isPartial"], errors="ignore")
    trends.index = pd.to_datetime(trends.index).tz_localize(None)
    trends = trends.resample("W").mean()
    # Forward-fill weekly Google Trends values to daily.
    trends = trends.resample("D").ffill()
    trends["consumer_search_index"] = trends[kw_list].mean(axis=1)
    trends.index.name = "date"
    return trends
