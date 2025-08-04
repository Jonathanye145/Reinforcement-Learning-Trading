
# trading_bot/config.py
# This file centralizes all configurations for the project.

import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- Stock & Data Settings ---
TICKER = "TSLA"
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = datetime.today().strftime('%Y-%m-%d')

# --- Sentiment Analysis Settings ---
NEWS_WEIGHT = 0.6
X_WEIGHT = 0.4

# --- RL Model & Environment Settings ---
WINDOW_SIZE = 10  # Number of past days' data to observe
MAX_ALLOCATION = 0.25  # Max % of portfolio to allocate to the stock
COMMISSION_RATE = 0.001  # Transaction cost (0.1%)
INITIAL_BALANCE = 1_000_000
STOP_LOSS_THRESHOLD = 0.90  # Sell if portfolio value drops to 90% of initial
TRAINING_TIMESTEPS = 120000

# Features to be used by the RL model
# This list MUST match the columns created in data_preparation.py
RL_FEATURES = [
    "combined_sentiment", "sentiment_momentum", "RSI", "MACD",
    "SMA_20", "Bollinger_Width", "P_E"
]
