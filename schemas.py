"""Shared types for the portfolio advisor system."""

from dataclasses import dataclass, field
from typing import Optional


# ── Input ──

@dataclass
class Holding:
    ticker: str
    shares: float
    cost: float  # cost basis per share


@dataclass
class PortfolioInput:
    holdings: list[Holding]

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioInput":
        return cls(
            holdings=[Holding(**h) for h in d["holdings"]]
        )

    @property
    def tickers(self) -> list[str]:
        return [h.ticker for h in self.holdings]


# ── Agent Outputs ──

@dataclass
class RiskFlag:
    category: str          # e.g. "sector_concentration", "high_correlation", "high_beta", "drawdown"
    severity: str          # "low" | "medium" | "high"
    message: str


@dataclass
class RiskOutput:
    risk_score: float                          # 0-100
    sector_concentration: dict[str, float]     # sector -> % weight
    correlation_matrix: dict[str, dict[str, float]]  # ticker -> ticker -> corr
    portfolio_beta: float
    max_drawdowns: dict[str, float]            # ticker -> max drawdown %
    flags: list[RiskFlag] = field(default_factory=list)


@dataclass
class Article:
    ticker: str
    company_name: str
    title: str
    source: str
    published_at: str
    url: str
    description: Optional[str] = None
    content: Optional[str] = None


@dataclass
class HoldingSentiment:
    ticker: str
    sentiment_score: float       # -1 to +1
    event_type: str              # "earnings" | "regulatory" | "lawsuit" | "macro" | "none"
    impact: str                  # "low" | "medium" | "high"
    summary: str
    catalysts: list[str] = field(default_factory=list)


@dataclass
class MarketIntelOutput:
    sentiment_score: float                     # aggregate -1 to +1
    holdings_sentiment: list[HoldingSentiment]
    catalysts: list[str]                       # list of active catalyst descriptions
    articles: list[Article]


# ── Final Output ──

@dataclass
class Advisory:
    summary: str                  # plain English overview
    recommendations: list[str]    # specific action items
    risk: RiskOutput
    market_intel: MarketIntelOutput
