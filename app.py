import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data import fetch_many
from utils.indicators import add_indicators
from utils.filters import (
    apply_liquidity_filter, apply_trend_filter,
    breakout_volume_signal, low52_reversal_signal,
    compute_score, merge_with_fundamentals, apply_fundamental_filters
)

st.set_page_config(page_title='AI Stock Selection — EoD (Enhanced)', layout='wide')
st.title('🧠 AI Stock Selection — EoD (Enhanced)')
st.caption('Breakout+Volume and 52w Low Reversal strategies, with universe overrides and optional fundamentals.')

# Sidebar controls
with st.sidebar:
    st.header('Settings')
    years = st.slider('Years of history', 1, 5, 3)
    min_turnover = st.number_input('Min avg turnover (₹ Cr, 5-day)', min_value=0.0, value=5.0, step=0.5)

    st.subheader('Strategies')
    use_breakout = st.checkbox('Breakout + Volume', value=True)
    breakout_window = st.selectbox('Breakout window (days)', [20, 55], index=0)
    vol_mult = st.slider('Volume multiple (×AvgVol20)', 1.0, 5.0, 2.0, 0.1)

    use_low52 = st.checkbox('52-Week Low Reversal', value=True)
    low52_min = st.slider('Min % above 52w low', 0.0, 0.20, 0.03, 0.01)
    low52_max = st.slider('Max % above 52w low', 0.05, 0.50, 0.20, 0.01)

    st.markdown('---')
    st.subheader('Universe (override)')
    pasted = st.text_area('Paste tickers (one per line, e.g., RELIANCE.NS)', value='')
    uni_file = st.file_uploader('Or upload a CSV with a "Ticker" column', type=['csv'])

    st.markdown('---')
    st.subheader('Fundamental Filters (optional)')
    fund_file = st.file_uploader('Upload Screener CSV', type=['csv'])
    rules = {
        'DebtToEquity': st.number_input('Max D/E', 0.0, 10.0, 1.0, 0.1),
        'PromoterPledgePct': st.number_input('Max Promoter Pledge %', 0.0, 100.0, 5.0, 0.5),
        'SalesGrowthYoY': st.number_input('Min Sales Growth YoY %', -100.0, 200.0, 10.0, 1.0),
        'ProfitGrowthYoY': st.number_input('Min Profit Growth YoY %', -100.0, 200.0, 10.0, 1.0),
    }
    st.markdown('---')
    run_btn = st.button('Run Scan')

# Load default universe
default_uni = pd.read_csv('data/universe.csv')

# Override via paste
tickers = None
if pasted.strip():
    tickers = [t.strip() for t in pasted.strip().splitlines() if t.strip()]
# Override via upload
elif uni_file is not None:
    try:
        udf = pd.read_csv(uni_file)
        col = 'Ticker' if 'Ticker' in udf.columns else udf.columns[0]
        tickers = udf[col].dropna().astype(str).str.strip().tolist()
    except Exception as e:
        st.error(f'Failed to parse uploaded universe CSV: {e}')
# Fallback to default
if tickers is None:
    tickers = default_uni[default_uni.columns[0]].dropna().astype(str).str.strip().tolist()

st.write('**Universe size:**', len(tickers), 'tickers')

# Optionally load fundamentals
fund_df = None
if fund_file is not None:
    try:
        fund_df = pd.read_csv(fund_file)
        st.success(f'Loaded fundamentals with {fund_df.shape[0]} rows.')
    except Exception as e:
        st.error(f'Failed to parse fundamentals CSV: {e}')

if run_btn:
    with st.spinner('Fetching EoD data and computing filters...'):
        data_map = fetch_many(tickers, years=years)

        rows = []
        charts_store = {}

        for tkr, df in data_map.items():
            feat = add_indicators(df)
            if len(feat) < 210:
                continue

            liquidity_ok = apply_liquidity_filter(feat, min_turnover_cr=min_turnover).iloc[-1]
            trend_ok     = apply_trend_filter(feat).iloc[-1]
            sig_breakout = breakout_volume_signal(feat, breakout_window=breakout_window, vol_mult=vol_mult).iloc[-1] if use_breakout else False
            sig_low52    = low52_reversal_signal(feat, min_above=low52_min, max_above=low52_max).iloc[-1] if use_low52 else False

            passed = bool(liquidity_ok and trend_ok and (sig_breakout or sig_low52))
            score = float(compute_score(feat).iloc[-1])

            last = feat.iloc[-1]
            rows.append({
                'Ticker': tkr,
                'Passed': passed,
                'Score': round(score, 4),
                'Close': round(float(last['Close']), 2),
                'EMA50': round(float(last['Ema50']), 2),
                'EMA200': round(float(last['Ema200']), 2),
                'ATR14': round(float(last['Atr14']), 2) if not np.isnan(last['Atr14']) else np.nan,
                'AvgVol20': round(float(last['AvgVol20']), 0) if not np.isnan(last['AvgVol20']) else np.nan,
                'TurnoverCr(5d avg)': round(float(feat['TurnoverCr'].tail(5).mean()), 2),
                'Breakout+Vol?': bool(sig_breakout),
                '52w Low Reversal?': bool(sig_low52)
            })

            sub = feat.tail(200)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=sub.index, open=sub['Open'], high=sub['High'], low=sub['Low'], close=sub['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=sub.index, y=sub['Ema50'], name='EMA50'))
            fig.add_trace(go.Scatter(x=sub.index, y=sub['Ema200'], name='EMA200'))
            charts_store[tkr] = fig

        result = pd.DataFrame(rows)
        result = merge_with_fundamentals(result, fund_df)
        if fund_df is not None and not fund_df.empty:
            result = apply_fundamental_filters(result, rules)

        st.subheader('Scan Results')
        result_sorted = result.sort_values(by=['Passed','Score'], ascending=[False, False])
        st.dataframe(result_sorted, use_container_width=True, hide_index=True)
        st.download_button('Download CSV', result_sorted.to_csv(index=False).encode('utf-8'), file_name='scan_results.csv', mime='text/csv')

        st.subheader('Charts (Passed Picks)')
        passed_tickers = result_sorted[result_sorted['Passed']]['Ticker'].tolist()
        if not passed_tickers:
            st.info('No tickers passed today. Try relaxing thresholds or expand your universe.')
        for t in passed_tickers[:10]:
            st.markdown(f'**{t}**')
            st.plotly_chart(charts_store[t], use_container_width=True)
