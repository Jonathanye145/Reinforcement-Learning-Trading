
# trading_bot/sentiment_analysis.py
# Handles all logic related to fetching and processing sentiment data.

import logging
import pandas as pd
import numpy as np
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
from config import NEWS_API_KEY

logger = logging.getLogger(__name__)

class FinBERT:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FinBERT, cls).__new__(cls)
            cls._instance.model, cls._instance.tokenizer, cls._instance.device = cls._instance._load_model()
        return cls._instance

    def _load_model(self) -> Tuple:
        try:
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logger.info(f"FinBERT model loaded successfully on {device}.")
            return model, tokenizer, device
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            return None, None, None

    def get_batch_sentiment(self, texts: List[str]) -> List[float]:
        if not all([self.model, self.tokenizer, self.device]) or not texts: return [0.0] * len(texts)
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            return (scores[:, 2] - scores[:, 0]).tolist()
        except Exception as e:
            logger.error(f"Batch sentiment analysis error: {e}")
            return [0.0] * len(texts)

finbert_analyzer = FinBERT()

def get_news_api_sentiment(start_date: str, end_date: str) -> pd.Series:
    logger.info("Fetching sentiment from News API...")
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_ACTUAL_NEWSAPI_KEY":
        logger.warning("News API key not found. Returning empty sentiment series.")
        return pd.Series(dtype=float)
    url = "https://newsapi.org/v2/everything"
    params = {"q": "Tesla OR TSLA", "from": start_date, "to": end_date, "apiKey": NEWS_API_KEY, "language": "en", "pageSize": 100, "sortBy": "publishedAt"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles: return pd.Series(dtype=float)
        df = pd.DataFrame(articles)
        df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.date
        df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')
        sentiments = finbert_analyzer.get_batch_sentiment(df['text'].tolist())
        df['sentiment'] = sentiments
        daily_sentiment = df.groupby('publishedAt')['sentiment'].mean()
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return daily_sentiment.reindex(all_dates, fill_value=0.0)
    except requests.exceptions.RequestException as e:
        logger.error(f"NewsAPI request error: {e}")
        return pd.Series(dtype=float)

def get_x_sentiment(start_date: str, end_date: str) -> pd.Series:
    logger.info("Using mock implementation for X sentiment.")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    mock_sentiments = np.random.normal(loc=0.05, scale=0.2, size=len(dates))
    return pd.Series(np.clip(mock_sentiments, -1, 1), index=dates)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_start, test_end = "2025-07-01", "2025-07-05"
    print("\n--- Testing Sentiment Analysis Module ---")
    news_sent = get_news_api_sentiment(test_start, test_end)
    print("\nNews API Sentiment:")
    if not news_sent.empty: print(news_sent.head())
    else: print("Could not fetch news sentiment (check API key).")
    x_sent = get_x_sentiment(test_start, test_end)
    print("\nX (Mock) Sentiment:")
    print(x_sent.head())
    sample_texts = ["Record profits lead to stock surge!", "Market drops on bad news."]
    sentiments = finbert_analyzer.get_batch_sentiment(sample_texts)
    print(f"\nFinBERT Direct Test Results: {sentiments}")
