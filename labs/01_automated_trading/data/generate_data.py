"""Generate and save synthetic financial data."""

from pathlib import Path
from scipy import stats
from .synthetic import generate_synthetic_data
from .relationships import session2_relationship


if __name__ == "__main__":

    # Create saved directory if it doesn't exist
    saved_dir = Path(__file__).parent / "saved"
    saved_dir.mkdir(exist_ok=True)

    # Generate synthetic data
    df = generate_synthetic_data(
        n_features=3,
        n_observations=10000,
        feature_distributions=[stats.norm, stats.norm, stats.norm],
        feature_dist_params=[{"loc": 0, "scale": 1}] * 3,
        feature_noise_stds=[10, 5, 3],
        relationship_kwargs={"weights": [0.5, 0.3, 0.2], "noise_std": 2.0},
        returns_mean=0,
        returns_std=0.01,
        returns_autocorr=0.0,
        spread=0.001,
        missing_data_pct=0.05,
        seed=42,
    )
    # Save to CSV
    output_path = saved_dir / "stock_a.csv"
    df.to_csv(output_path, index=False)

    df = generate_synthetic_data(
        n_features=3,
        n_observations=10000,
        feature_distributions=[stats.norm, stats.norm, stats.norm],
        feature_dist_params=[{"loc": 0, "scale": 1}] * 3,
        feature_noise_stds=[10, 5, 3],
        relationship_kwargs={"weights": [0.5, 0.3, 0.2], "noise_std": 2.0},
        returns_mean=0,
        returns_std=0.01,
        returns_autocorr=0.008,
        spread=0.001,
        missing_data_pct=0.05,
        seed=42,
    )
    # Save to CSV
    output_path = saved_dir / "stock_b.csv"
    df.to_csv(output_path, index=False)

    # Generate Session 2 dataset with complex non-linear relationship
    df_session2 = generate_synthetic_data(
        n_features=4,
        n_observations=10000,
        feature_distributions=[
            stats.norm,  # X1: normal
            stats.poisson,  # X2: poisson
            stats.binom,  # X3: binomial
            stats.norm,  # X4: normal
        ],
        feature_dist_params=[
            {"loc": 0, "scale": 1},  # X1: normal(0, 1)
            {"mu": 2},  # X2: poisson(2)
            {"n": 10, "p": 0.5},  # X3: binomial(10, 0.5)
            {"loc": 2, "scale": 2},  # X4: normal(0, 1)
        ],
        feature_noise_stds=[0.1, 0.1, 0.1, 0.1],
        relationship=session2_relationship,
        relationship_kwargs={"noise_std": 0.5},
        returns_mean=0,
        returns_std=0.01,
        returns_autocorr=0.0,
        spread=0.001,
        missing_data_pct=0.05,
        seed=42,
    )
    # Save to CSV
    output_path = saved_dir / "stock_session2.csv"
    df_session2.to_csv(output_path, index=False)
    print(f"Generated Session 2 dataset: {df_session2.shape}")
    print(f"Saved to: {output_path}")
