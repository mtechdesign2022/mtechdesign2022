import yfinance as yf
import pandas as pd

def fetch_one(ticker: str, period="6mo", interval="1d"):
    return yf.download(ticker, period=period, interval=interval)

def fetch_many(tickers, period="6mo", interval="1d"):
    out = {}
    for t in tickers:
        try:
            out[t] = fetch_one(t, period, interval)
        except Exception as e:
            print(f"Error fetching {t}: {e}")
    return out
