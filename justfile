# Justfile for Machine Learning for Finance labs

# Generate synthetic data
generate:
    uv run python -m labs.01_automated_trading.data.generate_data

# Start Jupyter Lab
lab:
    uv run jupyter lab

