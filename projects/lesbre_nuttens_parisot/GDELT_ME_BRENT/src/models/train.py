"""Train, evaluate and compare Logistic Regression and Random Forest models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from .benchmark import majority_class_benchmark, rolling_volatility_benchmark
except ImportError:
    from benchmark import majority_class_benchmark, rolling_volatility_benchmark

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

FEATURE_COLS = [
    "n_events",
    "n_conflict_events",
    "conflict_ratio",
    "goldstein_mean",
    "goldstein_min",
    "avg_tone",
    "n_mentions",
    "n_articles",
    "goldstein_7d_ma",
    "tension_spike",
    "mentions_7d_ma",
    "mentions_zscore",
]
TARGET_COL = "target"
N_SPLITS = 5


def load_dataset() -> pd.DataFrame:
    """Load the final merged dataset from disk."""
    return pd.read_parquet(PROCESSED_DIR / "final_dataset.parquet")


def evaluate_fold(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit a model on one fold and return evaluation metrics.

    Args:
        model: Sklearn-compatible classifier.
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.

    Returns:
        Dictionary with accuracy, classification report, and confusion matrix.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def cross_validate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    scale: bool = True,
) -> list[dict]:
    """Run time-series cross-validation on a classifier.

    Uses TimeSeriesSplit to avoid look-ahead bias.

    Args:
        model: Sklearn-compatible classifier.
        X: Feature matrix (rows = trading days, ordered chronologically).
        y: Binary target series.
        n_splits: Number of CV folds.
        scale: Whether to apply StandardScaler within each fold.

    Returns:
        List of per-fold evaluation dictionaries.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        fold_result = evaluate_fold(model, X_train, y_train, X_test, y_test)
        fold_result["fold"] = fold_idx
        results.append(fold_result)

    return results


def mean_accuracy(results: list[dict]) -> float:
    """Compute mean accuracy across CV folds."""
    return float(np.mean([r["accuracy"] for r in results]))


def run_training() -> dict:
    """Train and compare all models using time-series CV.

    Returns:
        Dictionary mapping model name to CV results and benchmark comparisons.
    """
    df = load_dataset()
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    all_results = {}
    for name, model in models.items():
        results = cross_validate(model, X, y)
        acc = mean_accuracy(results)
        all_results[name] = {"cv_results": results, "mean_accuracy": acc}
        print(f"{name}: mean CV accuracy = {acc:.4f}")

    # Evaluate benchmarks on the last CV fold (most recent test period)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    last_train_idx, last_test_idx = list(tscv.split(X))[-1]

    y_train_last = y.iloc[last_train_idx]
    y_test_last = y.iloc[last_test_idx]
    df_test_last = df.iloc[last_test_idx]

    print("\n=== BENCHMARK: Majority Class ===")
    bench_majority = majority_class_benchmark(y_train_last, y_test_last)
    print(f"Accuracy: {bench_majority['accuracy']:.4f}")

    print("\n=== BENCHMARK: Rolling Volatility ===")
    bench_vol = rolling_volatility_benchmark(df_test_last)
    print(f"Accuracy: {bench_vol['accuracy']:.4f}")

    all_results["benchmarks"] = {
        "majority_class": bench_majority,
        "rolling_volatility": bench_vol,
    }

    return all_results


if __name__ == "__main__":
    run_training()
