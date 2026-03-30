import yfinance as yf
import pandas as pd
import numpy as np

def dowload_price_series(ticker, start_date =  None, end_date = None):
    
    df = None

    try:
        if start_date is None or end_date is None:
            df = yf.download(ticker, period='max')
        else:
            df = yf.download(ticker, start = start_date, end_date = end_date)
    except Exception as e:
        print(f'Not able to dowload the data because : {e}')
    return df

def download_sp500_price_series(start_date = None, end_date = None):
    return dowload_price_series("SPY", start_date, end_date)

def download_return_series(ticker, start_date = None, end_date = None):
    df_prices = dowload_price_series(ticker, start_date = start_date, end_date = end_date)
    return df_prices['Close'].pct_change().dropna()

def download_log_return_series(ticker, start_date = None, end_date = None):
    df_prices = dowload_price_series(ticker, start_date = start_date, end_date = end_date)
    return np.log(df_prices['Close']/df_prices['Close'].shift(1)).dropna()

def download_sp500_log_return_series( start_date = None, end_date = None):
    df_prices = download_sp500_price_series( start_date = start_date, end_date = end_date)
    return np.log(df_prices['Close']/df_prices['Close'].shift(1)).dropna()