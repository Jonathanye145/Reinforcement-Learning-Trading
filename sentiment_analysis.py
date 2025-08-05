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
import matplotlib.pyplot as plt
import urllib.parse
from typing import List, Optional
import os
from datetime import datetime, timedelta
import time
import json

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

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

def alpha_vantage_api(function, tickers, time_from, time_to, sort, limit):
    """
    Fetch Alpha Vantage data for a given time range, convert to DataFrame,
    and print the last retrieved dataset date.
    Returns a DataFrame.
    """
    url = "https://www.alphavantage.co/query"
    apikey = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not apikey:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set in environment variables")
    params = {
        "function": function,
        "tickers": tickers,
        "time_from": time_from,
        "time_to": time_to,
        "sort": sort,
        "limit": limit,
        "apikey": apikey
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    if "Error Message" in data:
        raise ValueError(f"API Error: {data['Error Message']}")
    
    if function == "NEWS_SENTIMENT" and "feed" in data:
        df = pd.DataFrame(data["feed"])
        if "time_published" in df.columns:
            df["time_published"] = pd.to_datetime(df["time_published"])
            last_date = df["time_published"].max()
            print(f"Last retrieved dataset date for {time_from} to {time_to}: {last_date}")
        return df
    else:
        raise ValueError("Unsupported function or response structure")
    
def alpha_vantage_api_paginated(function, tickers, start_date, end_date, sort="asc", limit=1000, resume_file="last_date.json", max_calls_per_day=3):
    """
    Paginate over date range, handle >1000 records, limit to 25 calls/day, handle overlaps, and support resuming.
    Returns a combined DataFrame.
    """
    dfs = []
    call_count = 0
    seen_urls = set()
    start_date_dt = datetime.strptime(start_date, "%Y%m%dT%H%M")
    end_date_dt = datetime.strptime(end_date, "%Y%m%dT%H%M")
    
    # Load seen URLs and resume point
    if os.path.exists(resume_file):
        with open(resume_file, "r") as f:
            resume_data = json.load(f)
            last_date = datetime.strptime(resume_data["last_date"], "%Y%m%dT%H%M")
            last_time_to = datetime.strptime(resume_data.get("last_time_to", start_date), "%Y%m%dT%H%M")
            seen_urls = set(resume_data.get("seen_urls", []))
            current_date = max(start_date_dt, last_date + timedelta(seconds=1))
        print(f"Resuming from {current_date} (last time_to: {last_time_to}, {len(seen_urls)} URLs seen)")
    else:
        current_date = start_date_dt
        last_time_to = start_date_dt
    
    while current_date < end_date_dt and call_count < max_calls_per_day:
        time_from = current_date.strftime("%Y%m%dT%H%M")
        time_to = min(current_date + timedelta(days=7), end_date_dt).strftime("%Y%m%dT%H%M")
        print(f"Fetching data from {time_from} to {time_to} (Call {call_count + 1}/{max_calls_per_day})")
        
        try:
            df = alpha_vantage_api(function, tickers, time_from, time_to, sort, limit)
            call_count += 1
            if not df.empty:
                # Filter out duplicates based on URL
                if "url" in df.columns:
                    initial_len = len(df)
                    df = df[~df["url"].isin(seen_urls)]
                    seen_urls.update(df["url"].tolist())
                    print(f"Filtered {initial_len - len(df)} duplicates; kept {len(df)} records")
                dfs.append(df)
                # Save last date and time_to for resuming
                if "time_published" in df.columns:
                    last_date = df["time_published"].max()
                    with open(resume_file, "w") as f:
                        json.dump({
                            "last_date": last_date.strftime("%Y%m%dT%H%M"),
                            "last_time_to": time_to,
                            "seen_urls": list(seen_urls)
                        }, f)
                
                # Check if limit was hit
                if len(df) == limit:
                    print(f"Limit of {limit} reached; splitting time range")
                    sub_dfs = []
                    sub_date = current_date
                    while sub_date < min(current_date + timedelta(days=7), end_date_dt) and call_count < max_calls_per_day:
                        sub_time_from = sub_date.strftime("%Y%m%dT%H%M")
                        sub_time_to = min(sub_date + timedelta(days=1), end_date_dt).strftime("%Y%m%dT%H%M")
                        print(f"Fetching sub-range from {sub_time_from} to {sub_time_to} (Call {call_count + 1}/{max_calls_per_day})")
                        sub_df = alpha_vantage_api(function, tickers, sub_time_from, sub_time_to, sort, limit)
                        call_count += 1
                        if not sub_df.empty:
                            # Filter duplicates
                            if "url" in sub_df.columns:
                                initial_len = len(sub_df)
                                sub_df = sub_df[~sub_df["url"].isin(seen_urls)]
                                seen_urls.update(sub_df["url"].tolist())
                                print(f"Filtered {initial_len - len(sub_df)} duplicates; kept {len(sub_df)} records")
                            sub_dfs.append(sub_df)
                            # Save last date and time_to
                            if "time_published" in sub_df.columns:
                                last_date = sub_df["time_published"].max()
                                with open(resume_file, "w") as f:
                                    json.dump({
                                        "last_date": last_date.strftime("%Y%m%dT%H%M"),
                                        "last_time_to": sub_time_to,
                                        "seen_urls": list(seen_urls)
                                    }, f)
                        sub_date += timedelta(days=1)
                        time.sleep(12)  # Respect free tier: 5 requests/min
                    if sub_dfs:
                        dfs.extend(sub_dfs)
            time.sleep(12)  # Respect free tier: 5 requests/min
        except Exception as e:
            print(f"Error fetching {time_from} to {time_to}: {e}")
        
        current_date = max(current_date + timedelta(days=7), last_date + timedelta(seconds=1) if "last_date" in locals() else current_date + timedelta(days=7))
        
        # Stop if max calls reached
        if call_count >= max_calls_per_day:
            print(f"Reached daily API limit of {max_calls_per_day}. Saved progress to {resume_file}. Resume tomorrow.")
            break

    # Combine DataFrames
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        if "url" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["url"])
        if "time_published" in combined_df.columns:
            last_date = combined_df["time_published"].max()
            print(f"Final last retrieved dataset date: {last_date}")
        return combined_df
    return pd.DataFrame()


def get_x_sentiment(start_date: str, end_date: str) -> pd.Series:
    logger.info("Using mock implementation for X sentiment.")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    mock_sentiments = np.random.normal(loc=0.05, scale=0.2, size=len(dates))
    return pd.Series(np.clip(mock_sentiments, -1, 1), index=dates)
# %%
if __name__ == '__main__':  
    logging.basicConfig(level=logging.INFO)
    test_start, test_end = "2025-07-10", "2025-07-25"
    print("\n--- Testing Sentiment Analysis Module ---")
    df = alpha_vantage_api_paginated(
        function="NEWS_SENTIMENT",
        tickers="AAPL",
        start_date="20240101T0000",
        end_date="20250101T0000",
        sort="asc",
        limit=1000
    )
    print(df.head())
    df.to_csv("news_sentiment_2024_2025.csv", index=False)

    # print("\nNews API Sentiment:")
    # if not news_sent.empty: print(news_sent.head())
    # else: print("Could not fetch news sentiment (check API key).")
    x_sent = get_x_sentiment(test_start, test_end)

    print("\nX (Mock) Sentiment:")
    print(x_sent.head())
    sample_texts = ["Record profits lead to stock surge!", "Market drops on bad news."]
    sentiments = finbert_analyzer.get_batch_sentiment(sample_texts)
    print(f"\nFinBERT Direct Test Results: {sentiments}")
    print("new file has been s")
    plt.plot([0,1,2],[0,5,4])

# %%
