"""
Merges occupancy, confidence, trend, and Walmart stock signals.
"""

import pandas as pd
from data_collection import fetch_walmart_stock, fetch_consumer_confidence, fetch_google_trends
from features import compute_occupancy

def build_dataset():
    """Merge occupancy, stock, confidence, and search-trend signals."""
    prices = fetch_walmart_stock()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    occupancy = compute_occupancy(f"data/annotations.coco.json")
    occupancy.index = pd.to_datetime(occupancy.index)
    occupancy = occupancy.sort_index()

    cci = fetch_consumer_confidence()
    cci.index = pd.to_datetime(cci.index)
    cci = cci.sort_index()

    trends = fetch_google_trends()
    trends.index = pd.to_datetime(trends.index)
    trends = trends.sort_index()

    df = occupancy.join(cci, how="inner")
    # df = df.join(trends, how="inner")
    # df = df.join(prices, how="inner")
    print(f"After occupancy+CCI join: {df.shape}")
    df = df.join(trends, how="inner")
    print(f"After +trends join: {df.shape}")
    df = df.join(prices, how="inner")
    print(f"After +prices join: {df.shape}")
    df["price_direction"] = (df["Close"].diff() > 0).astype(int)

    df["weekly_return"] = (df["Close"].shift(-5) / df["Close"]) - 1 # 5-day forward return for supervised regression later.
    df = df.dropna()
    return df