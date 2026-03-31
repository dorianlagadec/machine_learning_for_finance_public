"""Orchestrate the full pipeline end-to-end."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from analysis import run_granger_test
from backtest import run_backtest
from merge import build_dataset
import model as model_module


def main() -> None:
    """Run data preparation, modeling, backtesting, and Granger analysis."""

    # Build dataset and save it.
    print("\n=== Building dataset ===")
    df = build_dataset()
    df.to_csv("data/dataset.csv")

    # Print dataset info.
    print(f"Dataset shape: {df.shape}")
    print("Dataset head:")
    print(df.head())

    # Train and evaluate models using the full feature set.
    print("\n=== Model training & evaluation ===")
    model_module.main()

    # Run backtest using the merged dataset.
    print("\n=== Backtest ===")
    run_backtest(df)

    # Run Granger causality test.
    print("\n=== Granger causality test ===")
    run_granger_test(df, max_lag=3)

    # Final summary message.
    print("\n=== Pipeline completed ===")


if __name__ == "__main__":
    main()