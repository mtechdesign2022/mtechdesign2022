# utils/data.py
from __future__ import annotations
import time
from typing import Dict, List
import pandas as pd
import yfinance as yf

# ---- Streamlit cache (safe fallback if streamlit absent) --------------------
try:
    import streamlit as st
except ImportError:
    class _Shim:
        def cache_data(self, **_):
            def deco(fn): return fn
            return deco
    st = _Shim()  # type: ignore

# ---- HTTP session with on-disk caching + retries + UA -----------------------
import requests_cache
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Cache HTTP responses for 24h to avoid re-hitting Yahoo
_http_cache = requests_cache.CachedSession(
    cache_name="yf_http_cache",
    backend="sqlite",
    expire_after=24 * 60 * 60,   # 24 hours
)

# Be polite and resilient
_retry = Retry(
    total=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry)
_http_cache.mount("https://", _adapter)
_http_cache.headers.update({
    # Pretend to be a normal browser to avoid some bot defenses
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
})

# Pass the session to yfinance
_yf_session = _http_cache


def _years_to_period(years: int) -> str:
    if years >= 10: return "10y"
    if years >= 5:  return "5y"
    if years >= 3:  return "3y"
    if years >= 2:  return "2y"
    return "1y"

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.title() for c in df.columns})


def _fetch_ohlcv_raw(ticker: str, years: int = 3, retries: int = 3, sleep_s: float = 1.0) -> pd.DataFrame:
    """
    Single-ticker robust fetch with retries. Uses a cached HTTP session with UA+retries
    to reduce JSONDecode/429 errors on Streamlit Cloud.
    """
    period = _years_to_period(years)
    last_err = None
    for _ in range(retries):
        try:
            # 1) Try history() (often more stable)
            tk = yf.Ticker(ticker, session=_yf_session)
            df = tk.history(period=period, interval="1d", auto_adjust=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return _clean_cols(df)

            # 2) Fallback to download()
            df = yf.download(
                tickers=ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,           # important: avoid parallel calls on Cloud
                group_by="ticker",
                session=_yf_session,     # critical: use our cached/retry session
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


# Cached wrapper: cache the processed DataFrame for 24h
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_ohlcv(ticker: str, years: int = 3) -> pd.DataFrame:
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
        d = fetch_ohlcv(t, years=years)  # cached call (HTTP + Streamlit)
        if not d.empty:
            out[t] = d
    return out
