import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import pickle

# Define cache file path
CACHE_FILE = "data/cache.pkl"

# Load cache if it exists
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

# Save cache to file
def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

# Fetch market data for a list of tickers
def get_portfolio_data(tickers):
    """
    Given a list of tickers, fetch 60-day price history, current price, sector, beta, and market cap.
    Returns a clean pandas DataFrame.
    """
    cache = load_cache()
    result = []
    now = datetime.now()

    for ticker in tickers:
        # Check cache
        if ticker in cache and (now - cache[ticker]['timestamp']).days < 1:
            result.append(cache[ticker]['data'])
            continue

        try:
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            history = stock.history(period="60d")
            info = stock.info

            # Extract relevant data
            data = {
                "ticker": ticker,
                "current_price": info.get("currentPrice"),
                "sector": info.get("sector"),
                "beta": info.get("beta"),
                "market_cap": info.get("marketCap"),
                "price_history": history["Close"].tolist(),
            }

            # Add to result and cache
            result.append(data)
            cache[ticker] = {"data": data, "timestamp": now}
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    # Save updated cache
    save_cache(cache)

    # Convert result to DataFrame
    return pd.DataFrame(result)