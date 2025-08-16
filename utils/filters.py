def apply_liquidity_filter(df):
    return df[df['Volume'] > 100000]

def apply_trend_filter(df):
    df['BUY'] = (df['EMA20'] > df['EMA50']) & (df['RSI'] > 30) & (df['RSI'] < 70)
    return df
