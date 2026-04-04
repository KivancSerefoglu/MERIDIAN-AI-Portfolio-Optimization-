import yfinance as yf
import pandas as pd
import os
import pickle
from datetime import datetime

CACHE_FILE = "data/cache.pkl"


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict) -> None:
    # Fix 4: ensure the directory exists before writing
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def get_portfolio_data(tickers: list[str]) -> dict[str, dict]:

    cache = load_cache()
    result: dict[str, dict] = {}
    now = datetime.now()

    for ticker in tickers:
        # Return cached entry if it's less than 1 day old
        if ticker in cache and (now - cache[ticker]["timestamp"]).days < 1:
            result[ticker] = cache[ticker]["data"]
            continue

        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1y")   # 1y to match risk_agent needs
            info = stock.info

            data = {
                "current_price": (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("previousClose")
                ),
                "sector": info.get("sector", "Unknown"),
                "beta": info.get("beta") or 1.0,
                "market_cap": info.get("marketCap"),
                "price_history": history["Close"],
            }

            result[ticker] = data
            cache[ticker] = {"data": data, "timestamp": now}

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    save_cache(cache)
    return result