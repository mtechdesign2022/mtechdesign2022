import pandas as pd
import numpy as np

# ---------- Common filters ----------
def apply_liquidity_filter(df: pd.DataFrame, min_turnover_cr: float = 5.0) -> pd.Series:
    return (df['TurnoverCr'].rolling(5).mean() >= min_turnover_cr).fillna(False)

def apply_trend_filter(df: pd.DataFrame) -> pd.Series:
    return ((df['Close'] > df['Ema50']) & (df['Close'] > df['Ema200'])).fillna(False)

# ---------- Strategy 1: Breakout + Volume ----------
def breakout_volume_signal(df: pd.DataFrame, breakout_window: int = 20, vol_mult: float = 2.0) -> pd.Series:
    hi_col = f'High{breakout_window}' if f'High{breakout_window}' in df.columns else 'High20'
    cond_breakout = df['Close'] > df[hi_col]
    cond_volume   = df['Volume'] >= (vol_mult * df['AvgVol20'])
    cond_rsi_band = (df['Rsi14'] >= 40) & (df['Rsi14'] <= 65)
    return (cond_breakout & cond_volume & cond_rsi_band).fillna(False)

# ---------- Strategy 2: 52w Low Reversal ----------
def low52_reversal_signal(df: pd.DataFrame, min_above=0.03, max_above=0.20) -> pd.Series:
    in_band = (df['DistLo52'] >= min_above) & (df['DistLo52'] <= max_above)
    rsi_turn = df['Rsi14'] > df['Rsi14'].shift(1)
    ema5_reclaim = df['Close'] > df['Ema5']
    return (in_band & rsi_turn & ema5_reclaim).fillna(False)

# ---------- Scoring ----------
def compute_score(df: pd.DataFrame) -> pd.Series:
    atrpct = (df['Atr14'] / df['Close']).replace([np.inf,-np.inf], np.nan).clip(lower=0)
    dist50 = (df['Close'] / df['Ema50'] - 1.0).abs().replace([np.inf,-np.inf], np.nan)
    volx   = (df['Volume'] / df['AvgVol20']).replace([np.inf,-np.inf], np.nan).clip(upper=5)
    score = (1 - atrpct.rank(pct=True))*0.4 + (1 - dist50.rank(pct=True))*0.4 + (volx.rank(pct=True))*0.2
    return score.fillna(0)

# ---------- Fundamentals merge & filter ----------
def merge_with_fundamentals(picks: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals is None or fundamentals.empty:
        return picks
    return picks.merge(fundamentals, on='Ticker', how='left')

def apply_fundamental_filters(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    out = df.copy()
    if 'DebtToEquity' in out.columns:
        out = out[out['DebtToEquity'].fillna(999) <= rules.get('DebtToEquity', 1.0)]
    if 'OperatingCashFlow' in out.columns:
        out = out[out['OperatingCashFlow'].fillna(0) > 0]
    if 'PromoterPledgePct' in out.columns:
        out = out[out['PromoterPledgePct'].fillna(100) <= rules.get('PromoterPledgePct', 5.0)]
    if 'SalesGrowthYoY' in out.columns:
        out = out[out['SalesGrowthYoY'].fillna(-999) >= rules.get('SalesGrowthYoY', 10.0)]
    if 'ProfitGrowthYoY' in out.columns:
        out = out[out['ProfitGrowthYoY'].fillna(-999) >= rules.get('ProfitGrowthYoY', 10.0)]
    return out
