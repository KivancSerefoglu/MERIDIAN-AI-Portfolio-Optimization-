"""Market intelligence agent — consumes pre-summarized news per ticker."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from google import genai
from google.genai import types

import traceback

from schemas import Catalyst, HoldingSentiment, MarketIntelOutput, PortfolioInput


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-pro", etc.

_NO_NEWS_SENTINELS = {
    "- no article content available to summarize.",
    "- no summary was generated.",
}

SYSTEM_PROMPT = (
    "You are a disciplined financial market intelligence analyst. "
    "You are given a pre-computed company news summary already aggregated from multiple articles. "
    "Your task is to estimate the likely effect of this information on the company's value. "

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

    "ALLOWED VALUES: "
    "event_type must be exactly one of ['earnings','regulatory','lawsuit','macro','none']. "
    "impact must be exactly one of ['low','medium','high']. "

    "CATALYST GRADING RULES: "
    "Each catalyst must be returned as a JSON object with two fields: "
    "  \"text\": a short description of the specific value-relevant observation, "
    "  \"grade\": an integer from -3 to +3 representing the magnitude and direction of impact. "
    "Grading scale: "
    "  +3 = Strong positive (hard facts: earnings beat, major contract, strong revenue growth). "
    "  +2 = Moderate positive (credible growth signal, favorable analyst upgrade backed by reasoning). "
    "  +1 = Mild positive (speculative upside, minor operational improvement). "
    "   0 = Neutral, mixed, or no clear directional effect. "
    "  -1 = Mild negative (minor headwind, speculative risk). "
    "  -2 = Moderate negative (confirmed regulatory probe, notable revenue miss, credible competitive threat). "
    "  -3 = Strong negative (lawsuit verdict, major earnings miss, executive fraud, confirmed material loss). "
    "Be conservative: analyst price targets and opinions alone do NOT justify grades above +1 or below -1. "
    "Institutional position changes without disclosed size or strategic rationale are grade 0. "

    "RETURN STRICT JSON ONLY with these fields: "
    "{ \"event_type\": <string>, \"impact\": <string>, "
    "\"summary\": <string>, \"catalysts\": [{\"text\": <string>, \"grade\": <int>}, ...] } "
    "summary must explain the relation between the news and the company's value in fewer than 3 sentences. "
    "Do not wrap the JSON in markdown code fences. Return raw JSON only."
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
        catalysts=[],  # list[Catalyst]
    )


def _parse_llm_json(content: str) -> dict[str, Any]:
    import re
    text = (content or "").strip()
    # Strip markdown code fences
    if text.startswith("```"):
        inner = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        text = inner.group(1).strip() if inner else text.split("```", 2)[1].lstrip("json").strip()
    # Extract first {...} block in case the model appended extra text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)
    return json.loads(text)


def _analyze_summary_with_llm(client: genai.Client, ticker: str, summary_text: str) -> HoldingSentiment:
    user_prompt = (
        f"Ticker: {ticker.upper()}\n"
        "Company summarized news (already aggregated from multiple articles):\n"
        f"{summary_text}\n\n"
        "Return JSON only."
    )

    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    try:
        parsed = _parse_llm_json(response.text or "")
    except (json.JSONDecodeError, ValueError):
        return _no_news_holding(ticker)
    catalysts_raw = parsed.get("catalysts", [])
    catalysts: list[Catalyst] = []
    if isinstance(catalysts_raw, list):
        for c in catalysts_raw:
            if isinstance(c, dict) and c.get("text"):
                grade = max(-3, min(3, int(c.get("grade", 0))))
                catalysts.append(Catalyst(text=str(c["text"]).strip(), grade=grade))

    if catalysts:
        sentiment_score = max(-1.0, min(1.0, sum(c.grade for c in catalysts) / (3.0 * len(catalysts))))
    else:
        sentiment_score = 0.0

    summary = str(parsed.get("summary") or "").strip()
    if not summary:
        summary = "No clear value-relevant catalyst from available summarized news."

    return HoldingSentiment(
        ticker=ticker,
        sentiment_score=sentiment_score,
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
    articles: list | None = None,
) -> MarketIntelOutput:
    """Analyze summarized news by ticker and produce structured market intelligence."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    if not isinstance(news, dict):
        raise ValueError("news must be a dict mapping ticker -> summarized news string")

    holdings = _portfolio_holdings(portfolio)
    client = genai.Client(api_key=GEMINI_API_KEY)

    holdings_sentiment: list[HoldingSentiment] = []
    active_catalysts: list[Catalyst] = []

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
                traceback.print_exc()
                scored = _no_news_holding(ticker)
                scored.summary = "Unable to parse a clear value-relevant catalyst from summarized news."

        holdings_sentiment.append(scored)
        for c in scored.catalysts:
            active_catalysts.append(Catalyst(text=f"{ticker}: {c.text}", grade=c.grade))

    aggregate_score = _aggregate_sentiment(holdings=holdings, scored=holdings_sentiment)

    return MarketIntelOutput(
        sentiment_score=round(aggregate_score, 4),
        holdings_sentiment=holdings_sentiment,
        catalysts=active_catalysts,
        articles=list(articles) if articles else [],
    )
