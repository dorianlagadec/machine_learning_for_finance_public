"""Exploratory Data Analysis functions for financial data."""

import pandas as pd


def basic_summary(df):
    """
    Print basic summary statistics for the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"\nBasic statistics:")
    print(df.describe())
    print("=" * 80)

