"""Auto-refreshing dashboard: live prices + next-close prediction for
Indian metal stocks (NSE) and global metal commodity futures.

Run with:
    streamlit run app.py
"""
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

from stock_model import predict_ticker

TICKERS = {
    "Tata Steel": ("TATASTEEL.NS", "Stock"),
    "Hindalco": ("HINDALCO.NS", "Stock"),
    "JSW Steel": ("JSWSTEEL.NS", "Stock"),
    "Vedanta": ("VEDL.NS", "Stock"),
    "SAIL": ("SAIL.NS", "Stock"),
    "Jindal Steel": ("JINDALSTEL.NS", "Stock"),
    "National Aluminium": ("NATIONALUM.NS", "Stock"),
    "Gold (Global, USD/oz)": ("GC=F", "Commodity"),
    "Silver (Global, USD/oz)": ("SI=F", "Commodity"),
    "Copper (Global, USD/lb)": ("HG=F", "Commodity"),
}


@st.cache_data(ttl=55, show_spinner=False)
def get_prediction(ticker: str, period: str) -> dict:
    return predict_ticker(ticker, period=period)


@st.cache_data(ttl=15, show_spinner=False)
def get_live_price(ticker: str):
    return yf.Ticker(ticker).fast_info["last_price"]


st.set_page_config(page_title="Indian Stocks & Metals Predictor", layout="wide")

st.sidebar.header("Settings")
selected_names = st.sidebar.multiselect("Track", list(TICKERS), default=list(TICKERS))
custom_ticker = st.sidebar.text_input("Add custom ticker (e.g. RELIANCE.NS)").strip().upper()
period = st.sidebar.selectbox("History used for training", ["1y", "2y", "5y"], index=1)
refresh_secs = st.sidebar.slider("Auto-refresh interval (seconds)", 30, 300, 60, step=30)

st_autorefresh(interval=refresh_secs * 1000, key="datarefresh")

st.title("Indian Stocks & Metals — Live Price Prediction")
st.caption(
    "NSE data via Yahoo Finance is typically ~15 min delayed; global metal futures (Gold/Silver/Copper) "
    "are near-live in USD. Predictions are a simple ML baseline for illustration only — not financial advice."
)

symbols = [(name, sym, cat) for name, (sym, cat) in TICKERS.items() if name in selected_names]
if custom_ticker:
    symbols.append((custom_ticker, custom_ticker, "Custom"))

for display_name, symbol, category in symbols:
    with st.expander(f"{display_name}  ·  {symbol}  ·  {category}", expanded=True):
        try:
            result = get_prediction(symbol, period)
        except Exception as exc:
            st.error(f"Could not fetch/predict {symbol}: {exc}")
            continue

        try:
            live_price = get_live_price(symbol)
        except Exception:
            live_price = None

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Live Price", f"{live_price:.2f}" if live_price is not None else "N/A")
        col2.metric("Last Close", f"{result['last_close']:.2f}")
        delta = result["next_close_pred"] - result["last_close"]
        col3.metric("Predicted Next Close", f"{result['next_close_pred']:.2f}", f"{delta:+.2f}")
        col4.metric("Model RMSE", f"{result['rmse']:.2f}")

        chart_df = pd.DataFrame(
            {"Actual": result["actual"], "Predicted": result["predicted"]},
            index=result["test_dates"],
        )
        st.line_chart(chart_df)

st.caption(f"Last updated: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%Y-%m-%d %H:%M:%S IST')}")
