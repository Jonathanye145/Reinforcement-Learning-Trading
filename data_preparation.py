
# trading_bot/data_preparation.py
# Responsible for fetching all market data and preparing the final DataFrame.

import logging
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from config import TICKER, NEWS_WEIGHT, X_WEIGHT
from sentiment_analysis import get_news_api_sentiment, get_x_sentiment

logger = logging.getLogger(__name__)

def get_historical_pe(start_date: str, end_date: str) -> pd.Series:
    try:
        ticker = yf.Ticker(TICKER)
        earnings = ticker.quarterly_earnings
        if earnings.empty: return pd.Series(dtype=float)
        earnings['TTM_EPS'] = earnings['Earnings'].rolling(window=4).sum()
        earnings.dropna(inplace=True)
        earnings.index = earnings.index.to_timestamp()
        prices = yf.download(TICKER, start=start_date, end=end_date, progress=False)['Adj Close']
        pe_df = pd.DataFrame(index=prices.index)
        pe_df['Price'] = prices
        pe_df['TTM_EPS'] = earnings['TTM_EPS'].reindex(pe_df.index, method='ffill')
        pe_df['P_E'] = pe_df.apply(lambda row: row['Price'] / row['TTM_EPS'] if row['TTM_EPS'] and row['TTM_EPS'] > 0 else 0, axis=1)
        logger.info("Successfully calculated historical P/E ratio.")
        return pe_df['P_E']
    except Exception as e:
        logger.error(f"Could not calculate historical P/E: {e}")
        return pd.Series(dtype=float)

def prepare_data(start_date: str, end_date: str) -> pd.DataFrame:
    logger.info(f"Preparing data from {start_date} to {end_date} for {TICKER}...")
    stock_df = yf.download(TICKER, start=start_date, end=end_date, progress=False)
    if stock_df.empty:
        logger.error("Failed to download stock data.")
        return pd.DataFrame()
    stock_df.ta.rsi(length=14, append=True)
    stock_df.ta.macd(fast=12, slow=26, signal=9, append=True)
    stock_df.ta.sma(length=20, append=True)
    bbands = stock_df.ta.bbands(length=20)
    stock_df['Bollinger_Width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands['BBM_20_2.0']
    tech_indicators = stock_df.rename(columns={'RSI_14': 'RSI', 'MACD_12_26_9': 'MACD', 'SMA_20': 'SMA_20'})
    pe_ratio = get_historical_pe(start_date, end_date)
    news_sentiment = get_news_api_sentiment(start_date, end_date)
    x_sentiment = get_x_sentiment(start_date, end_date)
    data = tech_indicators.join(pe_ratio.rename("P_E"))
    data = data.join(news_sentiment.rename("news_sentiment"))
    data = data.join(x_sentiment.rename("x_sentiment"))
    data['combined_sentiment'] = (data['news_sentiment'].fillna(0) * NEWS_WEIGHT) + (data['x_sentiment'].fillna(0) * X_WEIGHT)
    data["sentiment_momentum"] = data["combined_sentiment"].diff(3)
    data = data.fillna(method="ffill").fillna(method="bfill").dropna()
    logger.info(f"Data prep complete. Shape: {data.shape}. Columns: {data.columns.tolist()}")
    return data

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_start, test_end = "2024-01-01", "2024-03-31"
    print("\n--- Testing Data Preparation Module ---")
    final_data = prepare_data(test_start, test_end)
    if not final_data.empty:
        print(f"\nSuccessfully prepared data with shape: {final_data.shape}")
        print(f"Columns: {final_data.columns.tolist()}")
        print(f"\nNaN values check:\n{final_data.isnull().sum()}")
    else:
        print("\nFailed to prepare data.")
