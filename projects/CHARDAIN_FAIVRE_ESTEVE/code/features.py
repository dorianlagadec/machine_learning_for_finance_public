"""
Extracts parking occupancy rate from labeled images using COCO annotations.
"""

import json
import os
from datetime import datetime
import pandas as pd


def parse_date_from_filename(filename):
    """Extract date from PKLot filename: YYYY-MM-DD_HH_MM_SS_jpg..."""
    try:
        return datetime.strptime(filename[:10], "%Y-%m-%d")
    except ValueError:
        return None


def compute_occupancy(annotations_path):
    """
    Compute daily parking occupancy rate from COCO annotations.
    Category 1 = occupied, Category 0 = empty 
    Returns a DataFrame indexed by date with occupancy_rate and engineered features.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images = {img["id"]: img["file_name"] for img in coco["images"]}

    records = []
    for ann in coco["annotations"]:
        filename = images[ann["image_id"]]
        date = parse_date_from_filename(filename)
        label = categories[ann["category_id"]]
        if date:
            records.append({"date": date, "occupied": 1 if label == "space-occupied" else 0})

    df = pd.DataFrame(records)
    daily = df.groupby("date")["occupied"].mean().reset_index()
    daily.columns = ["date", "occupancy_rate"]
    daily = daily.sort_values("date")

    # Lagged occupancy features
    for lag in [1, 2, 3, 7]:
        daily[f"occupancy_rate_lag_{lag}"] = daily["occupancy_rate"].shift(lag)

    # Calendar feature: Monday=0, Sunday=6
    daily["day_of_week"] = daily["date"].dt.dayofweek

    # Short-term trend feature using a 7-day rolling mean
    daily["occupancy_rate_roll7_mean"] = daily["occupancy_rate"].rolling(window=7).mean()

    return daily.set_index("date")
