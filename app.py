import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from utils.data import fetch_many
from utils.indicators import add_indicators
from utils.filters import apply_liquidity_filter, apply_trend_filter

st.title("📊 AI Stock Agent MVP (Cloud Ready)")

tickers_df = pd.read_csv("data/universe.csv")
tickers = tickers_df['TICKER'].tolist()

if st.button("Run Scan"):
    data = fetch_many(tickers)
    results = []
    for tkr, df in data.items():
        df = add_indicators(df)
        df = apply_liquidity_filter(df)
        df = apply_trend_filter(df)
        if not df.empty and df['BUY'].iloc[-1]:
            results.append(tkr)
    st.success(f"Selected Stocks: {results}")
