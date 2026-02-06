"""Custom relationship functions for synthetic data generation."""

import numpy as np


def session2_relationship(feature_vectors, noise_std=0, seed=None):
    """
    Complex non-linear relationship for Session 2 dataset.

    Relationship: (X1 + 10*X1² + 100*X1³ + X2 - if X2 > 1 + exp(X3)) + time-dependent component

    Parameters:
    -----------
    feature_vectors : list of np.ndarray
        List of 4 feature vectors: [X1, X2, X3, X4]
        X1: normal, X2: poisson, X3: binomial, X4: normal
    noise_std : float, default=0
        Standard deviation of additional Gaussian noise
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray
        Combined relationship vector
    """
    if seed is not None:
        np.random.seed(seed)

    X1, X2, X3, X4 = feature_vectors
    n = len(X1)

    # Base relationship: X1 + 10*X1² + 100*X1³
    result = X1 + 10 * (X1**2) + 100 * (X1**3)

    # Add X2, but subtract if X2 > 1
    result += X2
    result -= np.where(X2 > 1, 1, 0)

    # Add exp(X3)
    result += np.exp(X3)

    # Time-dependent component: add 5*X4 only for second half
    second_half_start = n // 2
    result[second_half_start:] += 5 * X4[second_half_start:]

    # Add noise if specified
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, n)
        result += noise

    return result

