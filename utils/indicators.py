import pandas as pd
import numpy as np

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        (df['High'] - df['Low']).abs(),
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={c: c.title() for c in out.columns})
    out['Ema5'] = ema(out['Close'], 5)
    out['Ema20'] = ema(out['Close'], 20)
    out['Ema50'] = ema(out['Close'], 50)
    out['Ema200'] = ema(out['Close'], 200)
    out['Rsi14'] = rsi(out['Close'], 14)
    out['Atr14'] = atr(out, 14)
    out['AvgVol20'] = out['Volume'].rolling(20).mean()
    out['TurnoverCr'] = (out['Close'] * out['Volume']) / 1e7
    out['High20'] = out['High'].rolling(20).max()
    out['High55'] = out['High'].rolling(55).max()
    out['Hi52'] = out['High'].rolling(252).max()
    out['Lo52'] = out['Low'].rolling(252).min()
    out['DistLo52'] = (out['Close'] - out['Lo52']) / out['Lo52']
    return out

