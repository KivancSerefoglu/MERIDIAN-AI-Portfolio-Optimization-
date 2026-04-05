"""Market intelligence agent — consumes pre-summarized news per ticker."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from google import genai
from google.genai import types

from schemas import HoldingSentiment, MarketIntelOutput, PortfolioInput


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

_NO_NEWS_SENTINELS = {
    "- no article content available to summarize.",
    "- no summary was generated.",
}

_SYSTEM_PROMPT = (
    "You are a disciplined financial market intelligence analyst. "
    "You are given a pre-computed company news summary already aggregated from multiple articles. "
    "Your task is to estimate the likely effect of this information on the company’s value. "

    "CRITICAL INSTRUCTION: "
    "Before making any judgment, mentally rewrite the news in neutral, plain language. "
    "Strip away all hype, promotional wording, emotional tone, praise, fear, or dramatic framing. "
    "Then evaluate only the underlying facts. "

    "RULES: "
    "Ignore fancy, exaggerated, or manipulative language. "
    "Do NOT be influenced by words like 'great opportunity', 'incredible', 'huge', 'perfect', "
    "'breakthrough', 'amazing', 'terrible', or 'crisis'. "
    "Separate facts from opinions, speculation, and commentary. "
    "Never use tone, excitement, or confidence of the source as evidence. "
    "Focus only on concrete business, financial, legal, regulatory, macro, or operational impact. "
    "Be conservative and do NOT over-infer from weak, vague, or incomplete information. "

    "EVALUATION CRITERIA: "
    "1. Factual business relevance such as earnings, revenue, guidance, regulation, lawsuits, macro exposure, or operations. "
    "2. Evidence strength where specific facts are stronger than opinions or speculation. "
    "3. Materiality, meaning whether this information meaningfully affects company value, risk, or expectations. "

    "SCORING RULES: "
    "sentiment_score must reflect real business impact, not writing style. "
    "Use strong positive or negative scores only when clearly supported by facts. "
    "If the summary is vague, speculative, or lacks financial or operational detail, keep sentiment_score near 0. "
    "If no meaningful value-relevant catalyst exists, use sentiment_score near 0, event_type='none', and impact='low'. "

    "ALLOWED VALUES: "
    "event_type must be exactly one of ['earnings','regulatory','lawsuit','macro','none']. "
    "impact must be exactly one of ['low','medium','high']. "

    "RETURN STRICT JSON ONLY with fields: sentiment_score, event_type, impact, summary, catalysts. "
    "summary must be one sentence. catalysts must be a list of short strings. "
)



def _portfolio_holdings(portfolio: Any) -> list[dict[str, Any]]:
    """Normalize accepted portfolio shapes into list[holding_dict]."""
    if isinstance(portfolio, list):
        return [dict(h) for h in portfolio]

    if isinstance(portfolio, dict):
        holdings = portfolio.get("holdings")
        if isinstance(holdings, list):
            return [dict(h) for h in holdings]

    if isinstance(portfolio, PortfolioInput):
        return [asdict(h) for h in portfolio.holdings]

    if is_dataclass(portfolio) and hasattr(portfolio, "holdings"):
        raw_holdings = getattr(portfolio, "holdings")
        if isinstance(raw_holdings, list):
            out: list[dict[str, Any]] = []
            for h in raw_holdings:
                if is_dataclass(h):
                    out.append(asdict(h))
                elif isinstance(h, dict):
                    out.append(dict(h))
            return out

    raise ValueError(
        "portfolio must be a holdings list, a dict with 'holdings', or PortfolioInput"
    )


def _clamp_sentiment(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, score))


def _clean_event_type(value: Any) -> str:
    allowed = {"earnings", "regulatory", "lawsuit", "macro", "none"}
    text = str(value or "").strip().lower()
    return text if text in allowed else "none"


def _clean_impact(value: Any) -> str:
    allowed = {"low", "medium", "high"}
    text = str(value or "").strip().lower()
    return text if text in allowed else "low"


def _is_no_news(summary_text: str) -> bool:
    normalized = summary_text.strip().lower()
    return not normalized or normalized in _NO_NEWS_SENTINELS


def _no_news_holding(ticker: str) -> HoldingSentiment:
    return HoldingSentiment(
        ticker=ticker,
        sentiment_score=0.0,
        event_type="none",
        impact="low",
        summary="No clear value-relevant catalyst from available summarized news.",
        catalysts=[],
    )


def _parse_llm_json(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _analyze_summary_with_llm(client: genai.Client, ticker: str, summary_text: str) -> HoldingSentiment:
    user_prompt = (
        f"Ticker: {ticker.upper()}\\n"
        "Company summarized news (already aggregated from multiple articles):\\n"
        f"{summary_text}\\n\\n"
        "Return JSON only."
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.1,
        ),
    )
    parsed = _parse_llm_json(response.text or "")
    catalysts_raw = parsed.get("catalysts", [])
    catalysts = [str(c).strip() for c in catalysts_raw if str(c).strip()] if isinstance(catalysts_raw, list) else []

    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        summary = "No clear value-relevant catalyst from available summarized news."

    return HoldingSentiment(
        ticker=ticker,
        sentiment_score=_clamp_sentiment(parsed.get("sentiment_score")),
        event_type=_clean_event_type(parsed.get("event_type")),
        impact=_clean_impact(parsed.get("impact")),
        summary=summary,
        catalysts=catalysts,
    )


def _aggregate_sentiment(holdings: list[dict[str, Any]], scored: list[HoldingSentiment]) -> float:
    if not scored:
        return 0.0

    weights: list[float] = []
    has_full_weights = len(holdings) == len(scored)
    if has_full_weights:
        for h in holdings:
            weight = h.get("weight")
            if weight is None:
                has_full_weights = False
                break
            try:
                weights.append(float(weight))
            except (TypeError, ValueError):
                has_full_weights = False
                break

    if has_full_weights:
        total_weight = sum(max(0.0, w) for w in weights)
        if total_weight > 0:
            return sum(s.sentiment_score * max(0.0, w) for s, w in zip(scored, weights)) / total_weight

    return sum(s.sentiment_score for s in scored) / len(scored)


def market_intel_agent(
    portfolio: Any,
    news: dict[str, str],
) -> MarketIntelOutput:
    """Analyze summarized news by ticker and produce structured market intelligence."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    if not isinstance(news, dict):
        raise ValueError("news must be a dict mapping ticker -> summarized news string")

    holdings = _portfolio_holdings(portfolio)
    client = genai.Client(api_key=GEMINI_API_KEY)

    holdings_sentiment: list[HoldingSentiment] = []
    active_catalysts: list[str] = []

    for holding in holdings:
        ticker = str(holding.get("ticker") or "").upper().strip()
        if not ticker:
            continue

        summary_text = str(news.get(ticker) or "").strip()
        if _is_no_news(summary_text):
            scored = _no_news_holding(ticker)
        else:
            try:
                scored = _analyze_summary_with_llm(client=client, ticker=ticker, summary_text=summary_text)
            except Exception:
                scored = _no_news_holding(ticker)
                scored.summary = "Unable to parse a clear value-relevant catalyst from summarized news."

        holdings_sentiment.append(scored)
        for catalyst in scored.catalysts:
            active_catalysts.append(f"{ticker}: {catalyst}")

    aggregate_score = _aggregate_sentiment(holdings=holdings, scored=holdings_sentiment)

    return MarketIntelOutput(
        sentiment_score=round(aggregate_score, 4),
        holdings_sentiment=holdings_sentiment,
        catalysts=active_catalysts,
        articles=[],
    )
