"""Backtesting engine for trading strategies."""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio.
    
    Parameters:
    -----------
    returns : array-like
        Portfolio returns
    risk_free_rate : float, default=0.0
        Risk-free rate
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate

    # TODO implement Sharpe ratio calculation
    sharpe = 0.0
    return sharpe


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown.
    
    Parameters:
    -----------
    equity_curve : array-like
        Cumulative equity curve
        
    Returns:
    --------
    float
        Maximum drawdown (as positive percentage)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # TODO implemnent max drawdown calculation


def backtest_strategy(
    df, predictions, initial_capital=100000, transaction_cost=0.001, trade_at="close"
):
    """
    Backtest a trading strategy based on predictions.
    
    Simple strategy: buy if prediction > 0, sell if prediction < 0.
    Assumes we can trade at close price (or next open if trade_at="open").
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with prices and timestamps
    predictions : array-like
        Model predictions (returns)
    initial_capital : float, default=100000
        Starting capital
    transaction_cost : float, default=0.001
        Transaction cost as fraction (0.001 = 0.1%)
    trade_at : str, default="close"
        When to execute trades: "close" or "open"
        
    Returns:
    --------
    dict
        Dictionary with backtest results and metrics
    """
    df = df.copy()
    df["prediction"] = predictions
    
    # Determine trade signals
    # Signal at time t predicts return from t to t+1
    df["signal"] = np.where(df["prediction"] > 0, 1, -1)  # 1 = buy, -1 = sell
    
    # TODO might change the below logic depending on the trade_at parameter (close vs open)
    df["position"] = df["signal"].astype(int)
    df["strategy_returns"] = df["position"] * df["returns"]
    
    # Apply transaction costs when position changes
    position_changes = df["position"].diff().abs() > 0
    df.loc[position_changes, "strategy_returns"] -= transaction_cost
    
    # Calculate equity curve
    df["equity"] = initial_capital * (1 + df["strategy_returns"]).cumprod()
    
    # Calculate metrics
    strategy_returns = df["strategy_returns"].dropna()
    
    total_return = (df["equity"].iloc[-1] / initial_capital - 1) * 100
    sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
    max_drawdown = calculate_max_drawdown(df["equity"])
    win_rate = 0.0 # TODO implement win rate calculation (percentage of profitable trades)
    
    # ML metrics (on predictions vs actual returns)
    actual_returns = df["returns"].dropna()
    pred_returns = df["prediction"].dropna()
    min_len = min(len(actual_returns), len(pred_returns))
    
    rmse = np.sqrt(mean_squared_error(actual_returns[:min_len], pred_returns[:min_len]))
    mae = mean_absolute_error(actual_returns[:min_len], pred_returns[:min_len])
    r2 = r2_score(actual_returns[:min_len], pred_returns[:min_len])
    
    results = {
        "equity_curve": df["equity"],
        "strategy_returns": strategy_returns,
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown * 100,
        "win_rate_pct": win_rate,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "df": df,
    }
    
    return results


def print_backtest_metrics(results):
    """
    Print backtest metrics in a formatted way.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from backtest_strategy()
    """
    print("=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"\nFinancial Metrics:")
    print(f"  Total Return: {results['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate: {results['win_rate_pct']:.2f}%")
    print(f"\nML Metrics:")
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  MAE: {results['mae']:.6f}")
    print(f"  R²: {results['r2']:.4f}")
    print("=" * 80)

