# .github/scripts/fetch_eod.py
import os, sys, time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
UNIVERSE_CSV = os.path.join(ROOT, "data", "universe.csv")
OUT_DIR = os.path.join(ROOT, "data", "cache")

HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "420"))  # ~1.5 yrs
BATCH        = int(os.getenv("BATCH", "12"))
RETRIES      = int(os.getenv("RETRIES", "3"))
SLEEP_BETWEEN= float(os.getenv("SLEEP_BETWEEN", "2.0"))

def load_universe():
    df = pd.read_csv(UNIVERSE_CSV)
    col = "Ticker" if "Ticker" in df.columns else df.columns[0]
    ticks = df[col].dropna().astype(str).str.strip().unique().tolist()
    fixed = []
    for t in ticks:
        t = t.upper()
        if "." not in t:   # add .NS if no suffix
            t = f"{t}.NS"
        fixed.append(t)
    return fixed

def fetch_range(tickers):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=HISTORY_DAYS)
    ok = {}
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i+BATCH]
        for _try in range(RETRIES):
            try:
                data = yf.download(
                    tickers=chunk,
                    start=start,
                    end=end + timedelta(days=1),
                    group_by="ticker",
                    progress=False,
                    threads=False,
                    interval="1d",
                    timeout=60,
                    auto_adjust=False,
                )
                if isinstance(data.columns, pd.MultiIndex):
                    for t in chunk:
                        if t in data.columns.get_level_values(0):
                            df = data[t].copy()
                            if not df.empty:
                                ok[t] = df
                else:
                    t = chunk[0]
                    if not data.empty:
                        ok[t] = data.copy()
                break
            except Exception as e:
                if _try == RETRIES - 1:
                    print(f"[WARN] Failed {chunk}: {e}")
                time.sleep(SLEEP_BETWEEN)
        time.sleep(SLEEP_BETWEEN)
    return ok

def normalize_save(data_map):
    os.makedirs(OUT_DIR, exist_ok=True)
    for tkr, df in data_map.items():
        df = df.reset_index().rename(columns={
            "Date":"Date", "Open":"Open", "High":"High",
            "Low":"Low", "Close":"Close", "Volume":"Volume",
        })
        keep = ["Date","Open","High","Low","Close","Volume"]
        df = df[keep].dropna(subset=["Date","Open","High","Low","Close"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
        out_path = os.path.join(OUT_DIR, f"{tkr}.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] {tkr} -> {out_path} rows={len(df)}")

def main():
    if not os.path.exists(UNIVERSE_CSV):
        print(f"[ERROR] Missing {UNIVERSE_CSV}")
        sys.exit(1)
    ticks = load_universe()
    if not ticks:
        print("[ERROR] Empty universe.csv")
        sys.exit(1)
    data = fetch_range(ticks)
    if not data:
        print("[ERROR] No data fetched")
        sys.exit(2)
    normalize_save(data)
    print("[DONE] Updated per-ticker CSVs.")

if __name__ == "__main__":
    main()
