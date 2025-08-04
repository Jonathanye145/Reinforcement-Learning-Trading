
# trading_bot/app.py
# The main entry point of the application, containing the Streamlit UI.

import streamlit as st
import logging
import pandas as pd
from config import TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, TICKER
from data_preparation import prepare_data
from model_handler import train_model, test_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_resource
def run_training():
    train_data = prepare_data(TRAIN_START_DATE, TRAIN_END_DATE)
    if train_data.empty:
        st.error("Could not prepare training data. Check logs.")
        return None
    model = train_model(train_data)
    return model

def main():
    st.set_page_config(layout="wide")
    st.title(f"AI-Powered Stock Trading Bot ({TICKER})")

    st.header("1. Model Training")
    with st.spinner(f"Training RL Agent on {TRAIN_START_DATE} to {TRAIN_END_DATE} data... (Cached)"):
        model = run_training()
    if model is None: return
    st.success("Model training complete!")

    st.header("2. Model Backtesting")
    with st.spinner(f"Backtesting model on {TEST_START_DATE} to {TEST_END_DATE} data..."):
        test_data = prepare_data(TEST_START_DATE, TEST_END_DATE)
        if test_data.empty:
            st.error("Could not prepare testing data. Check logs.")
            return
        test_data_viz, actions, portfolio_values = test_model(model, test_data)
    st.success("Backtesting complete!")

    if not test_data_viz.empty:
        st.header("3. Performance Analysis")
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        returns = (final_value - initial_value) / initial_value * 100
        col1, col2 = st.columns(2)
        col1.metric("Final Portfolio Value", f"${final_value:,.2f}")
        col2.metric("Total Return", f"{returns:.2f}%")
        st.subheader("Sentiment Source Analysis")
        sentiment_df = test_data_viz[['news_sentiment', 'x_sentiment', 'combined_sentiment']]
        st.line_chart(sentiment_df)
        st.subheader("Portfolio Value vs. Stock Price")
        chart_df = pd.DataFrame({'Portfolio Value': portfolio_values[1:], f'{TICKER} Close Price': test_data_viz['Adj Close']}, index=test_data_viz.index)
        st.line_chart(chart_df)
        st.subheader("Learned Trading Strategy: Stock Allocation")
        allocation_df = pd.DataFrame({'Allocation (%)': [a * 100 for a in actions]}, index=test_data_viz.index)
        st.area_chart(allocation_df)
    else:
        st.error("Could not run backtest. Check logs for data processing issues.")

if __name__ == "__main__":
    main()
