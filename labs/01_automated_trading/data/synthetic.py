"""Synthetic financial data generation module."""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta


def generate_single_feature_with_noise(
    n, distribution, dist_params, noise_std, seed=None
):
    """
    Generate a vector of N observations from a statistical distribution with added noise.

    Parameters:
    -----------
    n : int
        Number of observations
    distribution : scipy.stats distribution
        Statistical distribution to sample from
    dist_params : dict
        Parameters for the distribution (e.g., {'loc': 0, 'scale': 1})
    noise_std : float
        Standard deviation of Gaussian noise to add
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Vector of N observations
    """
    if seed is not None:
        np.random.seed(seed)

    samples = distribution.rvs(size=n, **dist_params)
    noise = np.random.normal(0, noise_std, n)
    return samples + noise


def relationship(feature_vectors, weights=None, noise_std=0, seed=None):
    """
    Combine multiple feature vectors into a single relationship vector.

    Parameters:
    -----------
    feature_vectors : list of np.ndarray
        List of m feature vectors (each of length N)
    weights : array-like, optional
        Weights for each feature. If None, uses equal weights (1/m)
    noise_std : float, default=0
        Standard deviation of additional Gaussian noise
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Combined relationship vector of length N
    """
    if seed is not None:
        np.random.seed(seed)

    m = len(feature_vectors)
    n = len(feature_vectors[0])

    if weights is None:
        weights = np.ones(m) / m

    result = np.zeros(n)
    for i, vec in enumerate(feature_vectors):
        result += weights[i] * vec

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, n)
        result += noise

    return result


def normalize_to_returns(values, target_mean=0, target_std=0.01, autocorr=0, seed=None):
    """
    Normalize values to realistic financial returns with optional autocorrelation.

    Parameters:
    -----------
    values : np.ndarray
        Input values to normalize
    target_mean : float, default=0
        Target mean for returns
    target_std : float, default=0.01
        Target standard deviation for returns
    autocorr : float, default=0
        Autocorrelation coefficient (AR(1) process)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Returns vector with specified properties
    """
    if seed is not None:
        np.random.seed(seed)

    # Normalize to target mean and std
    current_mean = np.mean(values)
    current_std = np.std(values)

    if current_std > 0:
        normalized = (values - current_mean) / current_std * target_std + target_mean
    else:
        normalized = np.full_like(values, target_mean)

    # Apply autocorrelation (AR(1) process)
    if autocorr != 0:
        returns = np.zeros_like(normalized)
        returns[0] = normalized[0]
        for t in range(1, len(normalized)):
            returns[t] = autocorr * returns[t - 1] + (1 - abs(autocorr)) * normalized[t]
    else:
        returns = normalized

    return returns


def generate_synthetic_data(
    n_features,
    n_observations,
    feature_distributions,
    feature_dist_params,
    feature_noise_stds,
    relationship=relationship,
    relationship_kwargs=None,
    returns_mean=0,
    returns_std=0.01,
    returns_autocorr=0,
    spread=0.001,
    missing_data_pct=0.05,
    start_date="2020-01-01",
    seed=None,
):
    """
    Generate synthetic financial data with features, prices, and bid/ask.

    Parameters:
    -----------
    n_features : int
        Number of features to generate
    n_observations : int
        Number of time observations
    feature_distributions : list of scipy.stats distributions
        Distribution for each feature
    feature_dist_params : list of dict
        Parameters for each feature distribution
    feature_noise_stds : list of float
        Noise std for each feature
    relationship : callable, default=relationship
        Function to combine features into target relationship
    relationship_kwargs : dict, optional
        Keyword arguments passed to relationship() function
        (e.g., {'weights': [0.5, 0.3, 0.2], 'noise_std': 0.05})
    returns_mean : float, default=0
        Target mean for returns
    returns_std : float, default=0.01
        Target std for returns
    returns_autocorr : float, default=0
        Autocorrelation in returns (AR(1))
    spread : float, default=0.001
        Bid/ask spread (e.g., 0.001 = 0.1%)
    missing_data_pct : float, default=0.05
        Percentage of missing data in X (features only)
    start_date : str, default='2020-01-01'
        Start date for timestamps (business days)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: timestamp, feature columns (X1, X2, ...),
        price, bid, ask, returns
    """
    if seed is not None:
        np.random.seed(seed)

    if relationship_kwargs is None:
        relationship_kwargs = {}

    # Generate features
    features = []
    for i in range(n_features):
        feat = generate_single_feature_with_noise(
            n_observations,
            feature_distributions[i],
            feature_dist_params[i],
            feature_noise_stds[i],
            seed=None,  # Use global seed, then increment
        )
        features.append(feat)

    # Create relationship (target returns)
    # Returns represent forward returns: features[T] predict returns[T] (from T to T+1)
    returns = relationship(features, seed=None, **relationship_kwargs)
    returns = normalize_to_returns(
        returns, returns_mean, returns_std, returns_autocorr, seed=None
    )

    # Convert to prices: price[T+1] = price[T] * (1 + returns[T])
    # Start with initial price of 100
    prices = 100 * np.cumprod(np.concatenate([[1], 1 + returns]))
    prices = np.round(prices, 2)

    # Use prices[0:n_observations] for alignment with features and returns
    prices_aligned = prices[:n_observations]

    # Generate bid/ask
    bid = prices_aligned * (1 - spread / 2)
    ask = prices_aligned * (1 + spread / 2)
    bid = np.round(bid, 2)
    ask = np.round(ask, 2)

    # Create timestamps (business days)
    start = pd.Timestamp(start_date)
    timestamps = pd.bdate_range(start=start, periods=n_observations, freq="B")

    # Build DataFrame
    data = {"timestamp": timestamps}
    for i, feat in enumerate(features):
        data[f"X{i+1}"] = feat
    data["price"] = prices_aligned
    data["bid"] = bid
    data["ask"] = ask
    data["returns"] = returns

    df = pd.DataFrame(data)

    # Inject missing data in features only
    if missing_data_pct > 0:
        n_missing = int(n_observations * n_features * missing_data_pct)
        feature_cols = [f"X{i+1}" for i in range(n_features)]
        for _ in range(n_missing):
            row_idx = np.random.randint(0, n_observations)
            col_idx = np.random.randint(0, n_features)
            df.loc[row_idx, feature_cols[col_idx]] = np.nan

    return df
