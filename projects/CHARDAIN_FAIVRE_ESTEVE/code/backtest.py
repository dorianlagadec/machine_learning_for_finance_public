"""Backtesting utilities for the Walmart direction model."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBClassifier



DATA_DIR = "data"
BACKTEST_PNG = "backtest.png"


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe is indexed by a sorted DatetimeIndex."""
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    return df.sort_index()


def _get_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return (X, y, feature_columns) for backtesting.

    Input features exclude:
    - `Close` (used only for computing returns)
    - `price_direction` (target)
    - `weekly_return` (leakage-prone derived label feature)
    """
    drop_cols = {"Close", "price_direction", "weekly_return"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["price_direction"].astype(int)
    return X, y, feature_cols


def _sharpe_ratio(daily_returns: pd.Series) -> float:
    """Compute the (non-annualized) Sharpe ratio with 0 risk-free rate."""
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return float("nan")

    std = daily_returns.std(ddof=0)
    if std == 0:
        return float("nan")
    return float(daily_returns.mean() / std)


def run_backtest(df: pd.DataFrame) -> None:
    """Run a simple long/cash backtest based on predicted `price_direction`.

    Strategy
    --------
    - If predicted `price_direction` is 1: invest in Walmart (daily Close returns).
    - If predicted `price_direction` is 0: hold cash (0% daily return).

    The model is trained using the last fold of a 5-fold `TimeSeriesSplit`
    (training segment = last fold's training indices; test segment =
    last fold's test indices).
    """
    if XGBClassifier is None:  # pragma: no cover
        raise ImportError(
            "xgboost is required for the backtest. Install it with: pip install xgboost"
        )

    df = _ensure_datetime_index(df)

    required_cols = {"Close", "price_direction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    X, y, _ = _get_feature_target(df)

    # Drop rows needed for modeling/returns to keep indices aligned.
    model_cols = list(X.columns) + ["Close", "price_direction"]
    df_model = df[model_cols].dropna()
    X, y, _ = _get_feature_target(df_model)
    close = df_model["Close"].astype(float)

    # Split chronologically; use the last fold for out-of-sample testing.
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    close_test = close.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test).astype(int)
    print(f"Predictions: {preds}")
    print(f"Fraction predicted up: {preds.mean():.2f}")

    # Daily returns (buy-and-hold) based on Close.pct_change().
    daily_returns = close.pct_change().fillna(0.0)
    daily_returns_test = daily_returns.iloc[test_idx]

    strategy_returns = daily_returns_test * preds
    buy_hold_returns = daily_returns_test

    strategy_cum = (1.0 + strategy_returns).cumprod()
    buy_hold_cum = (1.0 + buy_hold_returns).cumprod()

    strategy_sharpe = _sharpe_ratio(strategy_returns)
    buy_hold_sharpe = _sharpe_ratio(buy_hold_returns)

    strategy_total_return = float(strategy_cum.iloc[-1] - 1.0)
    buy_hold_total_return = float(buy_hold_cum.iloc[-1] - 1.0)

    # Plot cumulative curves.
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for backtest plotting. Install it with: pip install matplotlib"
        ) from exc

    out_dir = os.path.join(os.path.dirname(__file__), DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, BACKTEST_PNG)

    plt.figure(figsize=(10, 5))
    plt.plot(strategy_cum.index, strategy_cum.values, label="Strategy (long/cash)")
    plt.plot(buy_hold_cum.index, buy_hold_cum.values, label="Buy & hold Walmart")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return (growth of $1)")
    plt.title("Backtest: Strategy vs Buy-and-hold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("\n=== Backtest summary (last fold) ===")
    print(
        f"Strategy total return: {strategy_total_return:.4f}, Sharpe: {strategy_sharpe:.4f}"
    )
    print(
        f"Buy & hold total return: {buy_hold_total_return:.4f}, Sharpe: {buy_hold_sharpe:.4f}"
    )
    print(f"Saved cumulative return plot to: {out_path}")

