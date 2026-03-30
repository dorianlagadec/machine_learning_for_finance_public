# data_utils

Utilities for downloading financial time series data and building supervised learning datasets from them.

---

## Modules

### `Download_yf.py`

Downloads price and return data from Yahoo Finance via `yfinance`.

| Function | Description |
|----------|-------------|
| `dowload_price_series(ticker, start_date, end_date)` | Downloads OHLCV price data for any ticker. If no dates provided, fetches the full history. |
| `download_sp500_price_series(start_date, end_date)` | Shortcut for SPY (S&P 500 ETF) price data |
| `download_return_series(ticker, ...)` | Simple percentage returns: `Close.pct_change()` |
| `download_log_return_series(ticker, ...)` | Log returns: `log(Close / Close.shift(1))` |
| `download_sp500_log_return_series(...)` | Shortcut for SP500 log returns |

Log returns are preferred over simple returns for financial modelling — they are additive over time and more symmetric.

---

### `Dataset.py`

Builds supervised datasets from return series using a sliding window of lagged values.

| Function | Description |
|----------|-------------|
| `build_dataset_returns(df, n_lags)` | Features = `n_lags` past returns, target = next return |
| `build_dataset_abs_returns(df, n_lags)` | Features = `n_lags` past returns, target = next **absolute** return (volatility proxy) |
| `scale_data(X, y)` | Applies `StandardScaler` to both X and y, returns scaled arrays and both scalers |

**Why `build_dataset_abs_returns`?**
Predicting raw returns is very hard — daily log returns are close to a random walk with near-zero autocorrelation. Absolute returns (volatility) have much stronger autocorrelation (the ARCH effect), making them a more tractable prediction target where neural networks can outperform linear regression.

---

## Usage

```python
from data_utils import *

# Download and cache
df = download_sp500_log_return_series()
df.to_csv('data/sp500.csv')

# Build dataset (volatility prediction)
X, y = build_dataset_abs_returns(df, n_lags=10)

# Scale
X, y, scaler_x, scaler_y = scale_data(X, y)

# Inverse transform predictions back to original scale
preds_original = scaler_y.inverse_transform(model.forward(X_test, inference=True))
```
