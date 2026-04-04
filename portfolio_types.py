"""
Shared type definitions for the Portfolio Analysis Multi-Agent System.
All agents consume PortfolioInput and return typed output dicts.
"""

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

@dataclass
class Holding:
    ticker: str
    shares: float
    cost: float  # cost basis per share

    @property
    def cost_basis(self) -> float:
        return self.shares * self.cost

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "shares": self.shares,
            "cost": self.cost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Holding":
        return cls(
            ticker=d["ticker"],
            shares=float(d["shares"]),
            cost=float(d["cost"]),
        )


@dataclass
class PortfolioInput:
    holdings: list[Holding]

    @property
    def tickers(self) -> list[str]:
        return [h.ticker for h in self.holdings]

    @property
    def total_cost_basis(self) -> float:
        return sum(h.cost_basis for h in self.holdings)

    def to_dict(self) -> dict:
        return {"holdings": [h.to_dict() for h in self.holdings]}

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioInput":
        return cls(holdings=[Holding.from_dict(h) for h in d["holdings"]])


# ─────────────────────────────────────────────
# AGENT OUTPUTS
# ─────────────────────────────────────────────

@dataclass
class RiskOutput:
    """Output from the Risk Analysis Agent."""

    # Overall portfolio risk score (0–100, higher = riskier)
    portfolio_risk_score: float

    # Per-ticker risk scores
    ticker_risk_scores: dict[str, float]

    # Concentration risk: % of portfolio in top holding
    concentration_pct: float

    # Sector / correlation flags
    flags: list[str]

    # Human-readable explanation
    explanation: str

    # Optional breakdown per holding
    holding_details: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "portfolio_risk_score": self.portfolio_risk_score,
            "ticker_risk_scores": self.ticker_risk_scores,
            "concentration_pct": self.concentration_pct,
            "flags": self.flags,
            "explanation": self.explanation,
            "holding_details": self.holding_details,
        }


@dataclass
class MarketIntelOutput:
    """Output from the Market Intelligence Agent."""

    # Overall sentiment score (-1.0 bearish → +1.0 bullish)
    sentiment_score: float

    # Per-ticker sentiment
    ticker_sentiments: dict[str, float]

    # Key news / signals per ticker
    ticker_signals: dict[str, list[str]]

    # Macro context summary
    macro_summary: str

    # Human-readable explanation
    explanation: str

    # Confidence in the intel (0–1)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "sentiment_score": self.sentiment_score,
            "ticker_sentiments": self.ticker_sentiments,
            "ticker_signals": self.ticker_signals,
            "macro_summary": self.macro_summary,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


@dataclass
class AgentRunResult:
    """Top-level result wrapping both agent outputs."""

    portfolio: PortfolioInput
    risk: Optional[RiskOutput] = None
    market_intel: Optional[MarketIntelOutput] = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "portfolio": self.portfolio.to_dict(),
            "risk": self.risk.to_dict() if self.risk else None,
            "market_intel": self.market_intel.to_dict() if self.market_intel else None,
            "errors": self.errors,
        }
