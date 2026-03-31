# Deep Learning from Scratch

A ground-up implementation of deep learning in pure NumPy, applied to financial time series forecasting on SP500 data. The goal is to understand every component of a neural network — forward pass, backpropagation, optimizers, regularization — by building it without relying on any deep learning framework.


**The results of all studies conducted in the notebooks are analyzed and discussed in the report [`report/main.pdf`](report/main.pdf).**

---

## Project Structure

```
deep_learning_implementation/
├── notebooks/
│   ├── 1_financial_data_and_baselines.ipynb
│   ├── 2_activation_functions.ipynb
│   ├── 3_weight_initialization.ipynb
│   ├── 4_fnn_volatility_forecasting.ipynb
│   └── 5_rnn_volatility_forecasting.ipynb
├── src/
│   ├── dl_utils/          # Core deep learning library (layers, loss, optimizer, trainer)
│   ├── data_utils/        # Financial data download and dataset construction
│   ├── model_analysis/    # Visualization utilities (weight/activation distributions)
│   ├── maths_utils.py     # Low-level math: chain rule, manual gradient, simple linear model
│   └── neural_network_utils.py  # Early-stage neural network (pre dl_utils)
├── data/                  # Cached SP500 log return series (CSV)
└── report/                # LaTeX report with figures
```

---

## Notebooks

### `notebooks/1_financial_data_and_baselines.ipynb` — Financial Data & Baselines
SP500 log return data loading and exploration, followed by baseline models (linear regression) that establish the performance floor for subsequent deep learning experiments.

### `notebooks/2_activation_functions.ipynb` — Activation Functions
Visual reference for all implemented activation functions.
- Each function plotted alongside its derivative (computed via the Operation's own `backward()`)
- Markdown pros/cons for each: Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish
- Side-by-side overview grid of all six

### `notebooks/3_weight_initialization.ipynb` — Weight Initialization
Experiments on how weight initialization affects training dynamics.
- For each strategy: plots weight distributions, pre-activation signals, and post-activation signals before and after training
- Shows concretely how bad initialization (standard normal + Sigmoid) causes saturation while Glorot keeps signals stable
- Final MAE comparison table across all strategies: Standard, Glorot Normal, Glorot Uniform, He Normal, He Uniform, LeCun Normal

### `notebooks/4_fnn_volatility_forecasting.ipynb` — FNN Volatility Forecasting
Full feedforward network pipeline using the `dl_utils` framework applied to SP500 volatility (absolute return) forecasting.
- Benchmarks FNN vs linear regression; exploits the ARCH effect (autocorrelated volatility)
- Explores depth, activation functions, dropout regularization, learning rate decay, and momentum
- PyTorch equivalent included for reference validation

### `notebooks/5_rnn_volatility_forecasting.ipynb` — RNN Volatility Forecasting
RNN experiments on SP500 volatility forecasting using the PyTorch wrappers (`Rnn.py`).
- Sequential dataset construction with sliding window for recurrent models
- Comparison of RNN architectures against FNN baseline

---

## Modules

| Module | Description |
|--------|-------------|
| `dl_utils` | Core framework: operations, layers, loss functions, optimizers, trainer; also includes PyTorch-based RNN wrappers (`Rnn.py`) for reference validation |
| `data_utils` | SP500 data download via yfinance, lag-feature dataset construction (2D and sequential 3D) |
| `model_analysis` | Weight/activation distribution plotting utilities used in notebooks |
| `maths_utils` | Chain rule, manual gradients, simple linear regression from scratch |
| `neural_network_utils` | Pre-framework neural network, kept as a reference implementation |

See the `README.md` inside each module folder for detailed documentation.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

**1. Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repo and navigate into it:**
```bash
git clone <repo-url>
cd dl_from_scratch
```

**3. Create the virtual environment and install dependencies:**
```bash
uv sync
```

This reads `pyproject.toml` and installs all dependencies (numpy, matplotlib, scikit-learn, torch, yfinance) into a local `.venv`.

**4. Register the kernel for Jupyter:**
```bash
uv run python -m ipykernel install --user --name dl-from-scratch
```

**5. Launch Jupyter:**
```bash
uv run jupyter notebook
```

Then select the `dl-from-scratch` kernel when opening any notebook.

**Dependencies** (from `pyproject.toml`):

| Package | Use |
|---------|-----|
| `numpy` | Core numerical computing |
| `matplotlib` | Plotting |
| `scikit-learn` | Benchmarking, preprocessing, train/test split |
| `torch` | PyTorch reference implementations for validation |
| `yfinance` | SP500 data download |

> Python 3.13+ required.

---

## Quick Start

```python
import sys
sys.path.append('./src')
import numpy as np
from dl_utils import *
from data_utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
df = pd.read_csv('data/df_sp_500_log_ret.csv', index_col='Date')
X, y = build_dataset_abs_returns(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test  = scaler_x.transform(X_test)

# Build model
model = NeuralNetwork(
    layers=[
        Dense(neurons=32, activation=ReLU(), weight_init='he_normal', dropout=0.8),
        Dense(neurons=16, activation=ReLU(), weight_init='he_normal', dropout=0.8),
        Dense(neurons=1,  activation=Linear())
    ],
    loss=MeanSquaredError()
)

# Train
trainer = Trainer(model, SGD_MOMENTUM(lr=0.001, final_lr=0.0001))
trainer.fit(X_train, y_train, X_test, y_test,
            epochs=300, eval_every=10, patience=5)

# Evaluate
preds = model.forward(X_test, inference=True)
print(f"MAE: {np.mean(np.abs(preds - y_test)):.4f}")
```
