"""Quick live test: fetch real news, summarize it, and run market_intel_agent."""

import sys
import os
import json
from dataclasses import asdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from agents.market_intel_agent import market_intel_agent
from utilities.news import build_summarization_user_text, get_news, summarize_articles

PORTFOLIO = [
    {"ticker": "AAPL", "name": "Apple Inc."},
    {"ticker": "NVDA", "name": "NVIDIA Corporation"},
]
LOOKBACK_DAYS = 3
MAX_ARTICLES_PER_TICKER = 2

if not os.getenv("GROQ_API_KEY"):
    print("GROQ_API_KEY is not set. Please export it and run again.")
    sys.exit(1)

news_summaries: dict[str, str] = {}

for holding in PORTFOLIO:
    ticker = holding["ticker"]
    company_name = holding.get("name") or ticker

    articles = get_news(
        ticker=ticker,
        company_name=company_name,
        lookback_days=LOOKBACK_DAYS,
        max_articles=MAX_ARTICLES_PER_TICKER,
    )

    print(f"\n=== {ticker} | {company_name} ===")
    print(f"Fetched articles: {len(articles)}")
    if articles:
        top = articles[0]
        print(f"Top title: {top.title}")
        print(f"Top source: {top.source}")
        print(f"Top date: {top.published_at}")

        # Build and show the exact user payload sent to summarization model.
        original_text = build_summarization_user_text(
            articles=articles,
            company_name=company_name,
            ticker=ticker,
        )
        print("\n--- ORIGINAL TEXT ---")
        print(original_text)
        print()

    summary = summarize_articles(
        articles=articles,
        company_name=company_name,
        ticker=ticker,
    )
    news_summaries[ticker] = summary

    print("--- SUMMARIZED TEXT ---")
    print(summary)

print("\n=== NEWS MAP PASSED TO MARKET INTEL AGENT ===")
print(json.dumps(news_summaries, indent=2, ensure_ascii=True))

market_intel = market_intel_agent(PORTFOLIO, news_summaries)
print("\n=== MARKET INTEL OUTPUT ===")
print(json.dumps(asdict(market_intel), indent=2, ensure_ascii=True))
