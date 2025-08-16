# utils/data.py
from __future__ import annotations
import time
from typing import Dict, List
import pandas as pd
import yfinance as yf

# Use Streamlit's cache so downloads persist across reruns / button clicks
try:
    import streamlit as st
except ImportError:
    # fallback shim so local scripts can still run without streamlit
    class _Shim:
        def cache_data(self, **_): 
            def deco(fn): return fn
            return deco
    st = _Shim()  # type: ignore


def _years_to_period(years: int) -> str:
    if years >= 10: return "10y"
    if years >= 5:  return "5y"
    if years >= 3:  return "3y"
    if years >= 2:  return "2y"
    return "1y"

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.title() for c in df.columns})


def _fetch_ohlcv_raw(ticker: str, years: int = 3, retries: int = 4, sleep_s: float = 1.2) -> pd.DataFrame:
    """
    Single-ticker robust fetch with retries. Uses period+interval and single-threaded mode
    to reduce JSONDecode errors on Streamlit Cloud cold starts / rate limits.
    """
    period = _years_to_period(years)
    last_err = None
    for _ in range(retries):
        try:
            # Approach 1: history() usually behaves a bit better
            df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _clean_cols(df)

            # Fallback: download()
            df = yf.download(
                tickers=ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,      # important on Streamlit Cloud
                group_by="ticker"
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        df = df.xs(ticker, axis=1, level=0)
                    except Exception:
                        pass
                if not df.empty:
                    return _clean_cols(df)
        except Exception as e:
            last_err = e
        time.sleep(sleep_s)

    print(f"[WARN] Fetch failed for {ticker}: {last_err}")
    return pd.DataFrame()


# ---- CACHED WRAPPER (TTL = 1 day) --------------------------------------------
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_ohlcv(ticker: str, years: int = 3) -> pd.DataFrame:
    """
    Cached per-ticker fetch. The cache key is (ticker, years).
    """
    return _fetch_ohlcv_raw(ticker, years=years)


def fetch_many(tickers: List[str], years: int = 3, max_batch: int | None = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple tickers sequentially. Optionally limit how many you pull this run
    to avoid Yahoo rate limits. Uses cached per-ticker fetch under the hood.
    """
    out: Dict[str, pd.DataFrame] = {}
    if max_batch is not None:
        tickers = tickers[:max_batch]
    for t in tickers:
        d = fetch_ohlcv(t, years=years)  # cached call
        if not d.empty:
            out[t] = d
    return out
