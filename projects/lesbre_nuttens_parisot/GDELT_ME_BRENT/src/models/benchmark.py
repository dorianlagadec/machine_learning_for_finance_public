"""Naive benchmark models for the Brent direction prediction task."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report


def majority_class_benchmark(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Always predict the majority class from the training set.

    Args:
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        Dictionary with accuracy and classification report.
    """
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = clf.predict(np.zeros((len(y_test), 1)))

    return {
        "model": "majority_class",
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def rolling_volatility_benchmark(
    df: pd.DataFrame,
    window: int = 20,
    vol_threshold: float = 0.015,
) -> dict:
    """Go long when recent realized volatility is below a threshold.

    Low volatility → predict upward move (1). High volatility → predict down (0).

    Args:
        df: DataFrame with columns 'brent_close' and 'target'.
        window: Rolling window size for volatility computation (trading days).
        vol_threshold: Annualized daily volatility threshold.

    Returns:
        Dictionary with accuracy and classification report.
    """
    df = df.copy()
    daily_ret = df["brent_close"].pct_change()
    df["rolling_vol"] = daily_ret.rolling(window).std()
    df = df.dropna(subset=["rolling_vol", "target"])

    y_pred = (df["rolling_vol"] < vol_threshold).astype(int)
    y_test = df["target"]

    return {
        "model": "rolling_volatility",
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
