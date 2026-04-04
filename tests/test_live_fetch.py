"""Quick live test: fetch one article and print its full content."""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from news import build_summarization_user_text, get_news, summarize_articles

TICKER = "AAPL"
COMPANY_NAME = "Apple Inc."
LOOKBACK_DAYS = 3

if not os.getenv("GROQ_API_KEY"):
    print("GROQ_API_KEY is not set. Please export it and run again.")
    sys.exit(1)

articles = get_news(
    ticker=TICKER,
    company_name=COMPANY_NAME,
    lookback_days=LOOKBACK_DAYS,
    max_articles=1,
)

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

# Build and show the exact user payload that is sent to Groq.
original_text = build_summarization_user_text(
    articles=articles,
    company_name=COMPANY_NAME,
    ticker=TICKER,
)

print("--- ORIGINAL TEXT ---")
print(original_text)
print()

# Run real summarization against Groq and print the model output.
summary = summarize_articles(articles=articles, company_name=COMPANY_NAME, ticker=TICKER)

print("--- SUMMARIZED TEXT ---")
print(summary)
