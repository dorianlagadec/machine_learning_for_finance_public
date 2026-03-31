"""Statistical analysis helpers for the Walmart direction dataset."""

from __future__ import annotations

from typing import Dict

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def run_granger_test(df: pd.DataFrame, max_lag: int = 3) -> None:
    """Run Granger causality tests: `occupancy_rate` -> `price_direction`.

    Parameters
    ----------
    df:
        Merged daily dataset. Must contain `occupancy_rate` and
        `price_direction` columns.
    max_lag:
        Maximum lag (in days/rows) to test.

    Prints
    ------
    p-values for each lag.

    Notes
    -----
    A p-value < 0.05 suggests that `occupancy_rate` has predictive power over
    `price_direction` beyond past values of `price_direction` alone.
    Results should be interpreted cautiously given the small sample size of
    approximately 200 daily observations.
    """

    required_cols = {"occupancy_rate", "price_direction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        # If date exists as a column, sort by it; otherwise sort by index.
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.sort_values("date")
        else:
            data = data.sort_index()
    else:
        data = data.sort_index()

    data = data[["price_direction", "occupancy_rate"]].dropna()

    # The test expects columns in the order (dependent, causing).
    # Here: price_direction is the dependent variable, occupancy_rate is the cause.
    test_result: Dict[int, dict] = grangercausalitytests(
        data[["price_direction", "occupancy_rate"]].to_numpy(),
        maxlag=max_lag,
        verbose=False,
    )

    print(f"\nGranger causality test: occupancy_rate -> price_direction (max_lag={max_lag})")
    for lag in range(1, max_lag + 1):
        lag_result = test_result[lag]
        # Use the chi-square test p-value from the SSR test.
        p_value = lag_result[0]["ssr_chi2test"][1]
        print(f"Lag {lag}: p-value = {p_value:.6f}")

    # p-value < 0.05 suggests occupancy_rate has predictive power over
    # price_direction beyond past values of price_direction alone.
    # Interpret results cautiously given the small sample size
    # of approximately 200 daily observations.

