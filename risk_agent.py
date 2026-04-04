"""
Risk Analysis Agent.

Phase 0: stub implementation with heuristic-based risk scoring.
Phase 1+: will integrate real volatility data (yfinance/beta).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import BaseAgent
from portfolio_types import PortfolioInput, RiskOutput


# Rough heuristic beta/volatility proxies — will be replaced with live data in Phase 1
_TICKER_RISK_PROXY: dict[str, float] = {
    "NVDA": 72,
    "TSLA": 85,
    "AAPL": 45,
    "MSFT": 40,
    "AMZN": 55,
    "GOOGL": 48,
    "META": 60,
    "SPY": 30,
    "QQQ": 38,
    "BND": 10,
    "GLD": 25,
    "BTC": 95,
    "ETH": 92,
}
_DEFAULT_RISK = 55  # unknown tickers get a moderate-high score


class RiskAgent(BaseAgent):
    name = "risk"

    def run(self, portfolio: PortfolioInput) -> RiskOutput:
        total_value = portfolio.total_cost_basis
        ticker_risk_scores: dict[str, float] = {}
        holding_details: dict[str, dict] = {}
        flags: list[str] = []

        # ── Per-holding risk ────────────────────────────────────────────
        for h in portfolio.holdings:
            risk = _TICKER_RISK_PROXY.get(h.ticker.upper(), _DEFAULT_RISK)
            ticker_risk_scores[h.ticker] = risk

            weight = h.cost_basis / total_value if total_value > 0 else 0
            holding_details[h.ticker] = {
                "shares": h.shares,
                "cost_basis": round(h.cost_basis, 2),
                "weight_pct": round(weight * 100, 2),
                "risk_score": risk,
                "weighted_risk": round(risk * weight, 2),
            }

        # ── Weighted portfolio risk ─────────────────────────────────────
        portfolio_risk_score = sum(
            d["weighted_risk"] for d in holding_details.values()
        )

        # ── Concentration check ─────────────────────────────────────────
        max_weight = max(
            d["weight_pct"] for d in holding_details.values()
        ) if holding_details else 0
        concentration_pct = round(max_weight, 2)

        if concentration_pct > 40:
            flags.append(f"HIGH CONCENTRATION: top holding is {concentration_pct:.1f}% of portfolio")
        if concentration_pct > 25:
            flags.append("Consider diversifying — single position >25%")

        # ── High-risk ticker flag ───────────────────────────────────────
        high_risk_tickers = [t for t, s in ticker_risk_scores.items() if s >= 80]
        if high_risk_tickers:
            flags.append(f"HIGH VOLATILITY tickers: {', '.join(high_risk_tickers)}")

        # ── Explanation ─────────────────────────────────────────────────
        level = (
            "LOW" if portfolio_risk_score < 30
            else "MODERATE" if portfolio_risk_score < 55
            else "HIGH" if portfolio_risk_score < 75
            else "VERY HIGH"
        )
        explanation = (
            f"Portfolio risk is {level} (score: {portfolio_risk_score:.1f}/100). "
            f"Largest position is {concentration_pct:.1f}% of total cost basis. "
            + (f"Flags: {'; '.join(flags)}." if flags else "No major concentration or volatility flags.")
        )

        return RiskOutput(
            portfolio_risk_score=round(portfolio_risk_score, 2),
            ticker_risk_scores=ticker_risk_scores,
            concentration_pct=concentration_pct,
            flags=flags,
            explanation=explanation,
            holding_details=holding_details,
        )
