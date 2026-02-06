"""Feature engineering functions for financial data."""

import pandas as pd
import numpy as np


def prepare_features(df, feature_cols=None):
    """
    Prepare features for modeling.
    
    For now, this is a pass-through function. In future sessions,
    this will include feature engineering steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with features
    feature_cols : list, optional
        List of feature column names. If None, auto-detect (X1, X2, ...)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with prepared features
    """
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.startswith("X")]
    
    features_df = df[feature_cols].copy()
    
    # TODO Handle missing values (simple forward fill for now)
    
    return features_df


def prepare_target(df, target_col="returns"):
    """
    Prepare target variable for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str, default="returns"
        Name of target column
        
    Returns:
    --------
    pd.Series
        Target variable series
    """
    # TODO might change target value depending on the use case (classification vs regression)
    return df[target_col].copy()

