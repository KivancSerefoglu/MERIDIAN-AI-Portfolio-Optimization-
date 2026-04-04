"""
Phase 0 tests — validate types, data contract, and agent interfaces.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from portfolio_types import Holding, PortfolioInput, RiskOutput, MarketIntelOutput, AgentRunResult
from agents.risk_agent import RiskAgent
from agents.market_intel_agent import MarketIntelAgent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_portfolio() -> PortfolioInput:
    return PortfolioInput.from_dict({
        "holdings": [
            {"ticker": "NVDA", "shares": 50, "cost": 120},
            {"ticker": "AAPL", "shares": 30, "cost": 175},
            {"ticker": "BND",  "shares": 100, "cost": 74},
        ]
    })


@pytest.fixture
def single_holding_portfolio() -> PortfolioInput:
    return PortfolioInput.from_dict({
        "holdings": [{"ticker": "TSLA", "shares": 10, "cost": 200}]
    })


# ── Types tests ───────────────────────────────────────────────────────────────

class TestHolding:
    def test_cost_basis(self):
        h = Holding(ticker="NVDA", shares=50, cost=120)
        assert h.cost_basis == 6000.0

    def test_roundtrip(self):
        h = Holding(ticker="AAPL", shares=10, cost=150.5)
        assert Holding.from_dict(h.to_dict()) == h


class TestPortfolioInput:
    def test_tickers(self, sample_portfolio):
        assert set(sample_portfolio.tickers) == {"NVDA", "AAPL", "BND"}

    def test_total_cost_basis(self, sample_portfolio):
        expected = (50 * 120) + (30 * 175) + (100 * 74)
        assert sample_portfolio.total_cost_basis == expected

    def test_roundtrip(self, sample_portfolio):
        restored = PortfolioInput.from_dict(sample_portfolio.to_dict())
        assert restored.tickers == sample_portfolio.tickers
        assert restored.total_cost_basis == sample_portfolio.total_cost_basis

    def test_from_dict_data_contract(self):
        """Validates the agreed JSON data contract format."""
        raw = {"holdings": [{"ticker": "SPY", "shares": 5, "cost": 450}]}
        p = PortfolioInput.from_dict(raw)
        assert p.holdings[0].ticker == "SPY"
        assert p.holdings[0].shares == 5.0
        assert p.holdings[0].cost == 450.0


# ── Risk Agent tests ──────────────────────────────────────────────────────────

class TestRiskAgent:
    def test_returns_risk_output(self, sample_portfolio):
        output = RiskAgent().run(sample_portfolio)
        assert isinstance(output, RiskOutput)

    def test_score_in_range(self, sample_portfolio):
        output = RiskAgent().run(sample_portfolio)
        assert 0 <= output.portfolio_risk_score <= 100

    def test_all_tickers_scored(self, sample_portfolio):
        output = RiskAgent().run(sample_portfolio)
        assert set(output.ticker_risk_scores.keys()) == set(sample_portfolio.tickers)

    def test_concentration_single_holding(self, single_holding_portfolio):
        output = RiskAgent().run(single_holding_portfolio)
        assert output.concentration_pct == 100.0
        assert any("CONCENTRATION" in f for f in output.flags)

    def test_to_dict_keys(self, sample_portfolio):
        d = RiskAgent().run(sample_portfolio).to_dict()
        assert "portfolio_risk_score" in d
        assert "ticker_risk_scores" in d
        assert "flags" in d
        assert "explanation" in d


# ── Market Intel Agent tests ──────────────────────────────────────────────────

class TestMarketIntelAgent:
    def test_returns_market_intel_output(self, sample_portfolio):
        output = MarketIntelAgent().run(sample_portfolio)
        assert isinstance(output, MarketIntelOutput)

    def test_sentiment_score_in_range(self, sample_portfolio):
        output = MarketIntelAgent().run(sample_portfolio)
        assert -1.0 <= output.sentiment_score <= 1.0

    def test_all_tickers_have_sentiment(self, sample_portfolio):
        output = MarketIntelAgent().run(sample_portfolio)
        assert set(output.ticker_sentiments.keys()) == set(sample_portfolio.tickers)

    def test_all_tickers_have_signals(self, sample_portfolio):
        output = MarketIntelAgent().run(sample_portfolio)
        for ticker in sample_portfolio.tickers:
            assert ticker in output.ticker_signals
            assert len(output.ticker_signals[ticker]) > 0

    def test_to_dict_keys(self, sample_portfolio):
        d = MarketIntelAgent().run(sample_portfolio).to_dict()
        assert "sentiment_score" in d
        assert "ticker_sentiments" in d
        assert "ticker_signals" in d
        assert "explanation" in d


# ── AgentRunResult tests ──────────────────────────────────────────────────────

class TestAgentRunResult:
    def test_empty_result(self, sample_portfolio):
        r = AgentRunResult(portfolio=sample_portfolio)
        assert r.risk is None
        assert r.market_intel is None
        assert r.errors == []

    def test_to_dict_with_outputs(self, sample_portfolio):
        risk_out = RiskAgent().run(sample_portfolio)
        intel_out = MarketIntelAgent().run(sample_portfolio)
        r = AgentRunResult(portfolio=sample_portfolio, risk=risk_out, market_intel=intel_out)
        d = r.to_dict()
        assert d["risk"] is not None
        assert d["market_intel"] is not None
        assert d["errors"] == []
