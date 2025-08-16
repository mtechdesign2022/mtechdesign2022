import time
from typing import Dict, List
import pandas as pd
import yfinance as yf

# Map integer years to yfinance period string
def _years_to_period(years: int) -> str:
    if years >= 10: return "10y"
    if years >= 5:  return "5y"
    if years >= 3:  return "3y"
    if years >= 2:  return "2y"
    return "1y"

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.title() for c in df.columns})

def fetch_ohlcv(ticker: str, years: int = 3, retries: int = 3, sleep_s: float = 1.0) -> pd.DataFrame:
    """
    Robust EoD fetch with retries. Uses period+interval to avoid JSONDecode errors on cold starts.
    """
    period = _years_to_period(years)
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,       # reduce rate-limit issues on Streamlit Cloud
                group_by="ticker"
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                # if yfinance returns multi-index when group_by="ticker"
                if isinstance(df.columns, pd.MultiIndex):
                    # pick the first level (the ticker) if present
                    try:
                        df = df.xs(ticker, axis=1, level=0)
                    except Exception:
                        pass
                return _clean_cols(df)
        except Exception as e:
            last_err = e
        time.sleep(sleep_s)
    # On persistent failure, return empty DF (caller will skip)
    print(f"[WARN] Fetch failed for {ticker}: {last_err}")
    return pd.DataFrame()

def fetch_many(tickers: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        d = fetch_ohlcv(t, years=years)
        if not d.empty:
            out[t] = d
    return out
