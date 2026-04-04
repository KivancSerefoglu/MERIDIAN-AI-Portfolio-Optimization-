"""
Market Intelligence Agent.

Phase 0: stub with static/mock sentiment signals.
Phase 1+: will integrate real news APIs, yfinance, and LLM summarisation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import BaseAgent
from portfolio_types import PortfolioInput, MarketIntelOutput


# Static mock sentiment map — will be replaced with live news/LLM in Phase 1
_MOCK_SENTIMENTS: dict[str, float] = {
    "NVDA":  0.80,
    "TSLA":  0.10,
    "AAPL":  0.55,
    "MSFT":  0.65,
    "AMZN":  0.50,
    "GOOGL": 0.60,
    "META":  0.45,
    "SPY":   0.40,
    "QQQ":   0.42,
    "BND":   0.20,
    "GLD":   0.35,
    "BTC":  -0.10,
    "ETH":  -0.15,
}
_DEFAULT_SENTIMENT = 0.30

_MOCK_SIGNALS: dict[str, list[str]] = {
    "NVDA": ["Strong data center demand", "Blackwell GPU ramp", "AI capex tailwind"],
    "TSLA": ["Margin pressure ongoing", "Price cuts impacting profitability", "FSD progress mixed"],
    "AAPL": ["Services revenue growing", "iPhone cycle maturing", "India expansion"],
    "MSFT": ["Copilot monetisation", "Azure cloud growth", "OpenAI partnership"],
    "AMZN": ["AWS reacceleration", "Advertising revenue strong", "Cost cuts complete"],
    "GOOGL": ["Search AI integration", "YouTube ad recovery", "Cloud gaining share"],
    "META": ["Ad market recovery", "Llama open-source strategy", "Reality Labs drag"],
    "BTC":  ["ETF inflows slowing", "Macro headwinds", "Halving priced in"],
    "ETH":  ["L2 cannibalisation concern", "Staking yield attractive", "ETF approval uncertain"],
}
_DEFAULT_SIGNALS = ["Limited recent coverage — treat sentiment as low-confidence"]

_MACRO_SUMMARY = (
    "Macro environment (mock): Fed holding rates steady, inflation cooling gradually. "
    "AI infrastructure spend remains a dominant theme. Credit markets stable. "
    "NOTE: This is stub data — Phase 1 will pull live macro context."
)


class MarketIntelAgent(BaseAgent):
    name = "market_intel"

    def run(self, portfolio: PortfolioInput) -> MarketIntelOutput:
        ticker_sentiments: dict[str, float] = {}
        ticker_signals: dict[str, list[str]] = {}

        for h in portfolio.holdings:
            t = h.ticker.upper()
            ticker_sentiments[h.ticker] = _MOCK_SENTIMENTS.get(t, _DEFAULT_SENTIMENT)
            ticker_signals[h.ticker] = _MOCK_SIGNALS.get(t, _DEFAULT_SIGNALS)

        # Weighted average sentiment by cost basis
        total = portfolio.total_cost_basis
        sentiment_score = (
            sum(
                ticker_sentiments[h.ticker] * (h.cost_basis / total)
                for h in portfolio.holdings
            )
            if total > 0 else 0.0
        )

        level = (
            "BEARISH" if sentiment_score < -0.2
            else "SLIGHTLY BEARISH" if sentiment_score < 0.1
            else "NEUTRAL" if sentiment_score < 0.35
            else "SLIGHTLY BULLISH" if sentiment_score < 0.6
            else "BULLISH"
        )

        explanation = (
            f"Overall portfolio sentiment is {level} (score: {sentiment_score:+.2f}). "
            f"Analysed {len(portfolio.holdings)} positions. "
            "Phase 0 uses static mock signals — live news integration coming in Phase 1."
        )

        return MarketIntelOutput(
            sentiment_score=round(sentiment_score, 4),
            ticker_sentiments=ticker_sentiments,
            ticker_signals=ticker_signals,
            macro_summary=_MACRO_SUMMARY,
            explanation=explanation,
            confidence=0.5,  # Phase 0 is low-confidence mock data
        )
