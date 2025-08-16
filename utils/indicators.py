import pandas as pd

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['EMA20'] = out['Close'].ewm(span=20, adjust=False).mean()
    out['EMA50'] = out['Close'].ewm(span=50, adjust=False).mean()
    delta = out['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    out['RSI'] = 100 - (100 / (1 + rs))
    out['ATR'] = (out['High'] - out['Low']).rolling(14).mean()
    return out
