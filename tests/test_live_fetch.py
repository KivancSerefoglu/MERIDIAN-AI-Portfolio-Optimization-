"""Quick live test: fetch one article and print its full content."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from news import get_news

articles = get_news(ticker="AAPL", company_name="Apple Inc.", lookback_days=3, max_articles=1)

if not articles:
    print("No articles found.")
    sys.exit(1)

a = articles[0]
print(f"Title:   {a.title}")
print(f"Source:  {a.source}")
print(f"URL:     {a.url}")
print(f"Date:    {a.published_at}")
print(f"Content chars: {len(a.content) if a.content else 0}")
print()
print("--- CONTENT PREVIEW (first 1000 chars) ---")
print((a.content or "")[:30000])
