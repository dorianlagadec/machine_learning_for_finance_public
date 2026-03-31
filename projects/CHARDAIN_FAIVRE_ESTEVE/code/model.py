"""Model training and evaluation for Walmart direction prediction."""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier



DATASET_CSV_PATH = "data/dataset.csv"
FEATURE_IMPORTANCE_PNG = "data/feature_importance.png"


def load_dataset(csv_path: str = DATASET_CSV_PATH) -> pd.DataFrame:
    """Load the full dataset from CSV and return it sorted by date.

    Parameters
    ----------
    csv_path:
        Path to the `dataset.csv` file produced by the pipeline.

    Returns
    -------
    pandas.DataFrame
        Dataset indexed by `date`, sorted in chronological order.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def get_feature_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Split the dataset into (X, y) and return feature column names.

    The model input features X include all columns except:
    - `Close`
    - `price_direction` (target)
    - `weekly_return`
    """
    drop_cols = {"Close", "price_direction", "weekly_return"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["price_direction"].astype(int)
    return X, y, feature_cols


def evaluate(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> None:
    """Print accuracy and classification report for a given model."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} — Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds, zero_division=0))


def cross_validate_naive_majority(
    y: pd.Series,
    tscv: TimeSeriesSplit,
) -> np.ndarray:
    """Cross-validate the naive majority-class baseline."""
    accuracies: List[float] = []

    for train_idx, test_idx in tscv.split(y):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        majority_class = y_train.mode().iloc[0]
        y_pred = np.full(shape=len(y_test), fill_value=majority_class, dtype=int)
        accuracies.append(accuracy_score(y_test, y_pred))

    return np.asarray(accuracies, dtype=float)


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    tscv: TimeSeriesSplit,
    model_factory: Callable[[], BaseEstimator],
) -> np.ndarray:
    """Cross-validate a model and return per-fold accuracies.

    Notes
    -----
    To avoid data leakage, the model_factory is called inside each fold so
    any fitting (including scalers in a Pipeline) happens only on the
    training portion of the fold.
    """
    accuracies: List[float] = []

    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))

    return np.asarray(accuracies, dtype=float)


def plot_feature_importances(
    model: BaseEstimator,
    feature_names: List[str],
    output_png_path: str = FEATURE_IMPORTANCE_PNG,
) -> None:
    """Plot XGBoost feature importances and save to disk."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    ordered_names = [feature_names[i] for i in order]
    ordered_importances = importances[order]

    plt.figure(figsize=(10, max(4, 0.35 * len(feature_names))))
    plt.barh(ordered_names, ordered_importances)
    plt.gca().invert_yaxis()
    plt.title("XGBoost Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()


def main() -> None:
    """Run 5-fold time-series CV for multiple models and plot importances."""
    if XGBClassifier is None:  # pragma: no cover
        raise ImportError(
            "xgboost is required for the XGBoost model. "
            "Install it with: pip install xgboost"
        )

    df = load_dataset()
    df = df.dropna()

    X, y, feature_cols = get_feature_target(df)
    tscv = TimeSeriesSplit(n_splits=5)

    # Naive baseline (majority class) per fold.
    naive_scores = cross_validate_naive_majority(y, tscv)
    print(
        f"Naive baseline — accuracy: mean={naive_scores.mean():.4f}, "
        f"std={naive_scores.std(ddof=1):.4f}"
    )

    model_factories: Dict[str, Callable[[], BaseEstimator]] = {
        "Logistic Regression": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results: Dict[str, np.ndarray] = {}
    for name, factory in model_factories.items():
        scores = cross_validate_model(X, y, tscv, factory)
        results[name] = scores
        print(
            f"{name} — accuracy: mean={scores.mean():.4f}, "
            f"std={scores.std(ddof=1):.4f}"
        )

    # Train final XGBoost on the full dataset and plot feature importances.
    final_xgb = model_factories["XGBoost"]()
    final_xgb.fit(X, y)
    plot_feature_importances(final_xgb, feature_cols, FEATURE_IMPORTANCE_PNG)


if __name__ == "__main__":
    main()