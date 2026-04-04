"""
Portfolio Analysis Multi-Agent System — Entry Point
Usage:
    python main.py                         # runs with sample portfolio
    python main.py portfolio.json          # runs with a JSON file
"""

import json
import sys
import os

# Make sure local imports resolve regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from portfolio_types import PortfolioInput, AgentRunResult
from agents.risk_agent import RiskAgent
from agents.market_intel_agent import MarketIntelAgent


# ── Sample portfolio (used when no file is passed) ────────────────────────────
SAMPLE_PORTFOLIO = {
    "holdings": [
        {"ticker": "NVDA",  "shares": 50,  "cost": 120.00},
        {"ticker": "AAPL",  "shares": 30,  "cost": 175.00},
        {"ticker": "MSFT",  "shares": 20,  "cost": 310.00},
        {"ticker": "TSLA",  "shares": 15,  "cost": 200.00},
        {"ticker": "BND",   "shares": 100, "cost": 74.00},
    ]
}


def load_portfolio(path: str | None) -> PortfolioInput:
    if path:
        with open(path) as f:
            raw = json.load(f)
    else:
        raw = SAMPLE_PORTFOLIO
    return PortfolioInput.from_dict(raw)


def print_section(title: str, content: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    print(content)


def run(portfolio: PortfolioInput) -> AgentRunResult:
    result = AgentRunResult(portfolio=portfolio)
    agents = [RiskAgent(), MarketIntelAgent()]

    for agent in agents:
        print(f"  → Running {agent.name} agent...")
        try:
            output = agent.run(portfolio)
            if agent.name == "risk":
                result.risk = output
            elif agent.name == "market_intel":
                result.market_intel = output
        except Exception as e:
            msg = f"{agent.name} agent failed: {e}"
            result.errors.append(msg)
            print(f"  ✗ {msg}")

    return result


def display(result: AgentRunResult) -> None:
    p = result.portfolio

    print_section("PORTFOLIO SUMMARY", "")
    print(f"  {'Ticker':<8} {'Shares':>8} {'Cost/sh':>10} {'Basis':>12}")
    print(f"  {'──────':<8} {'──────':>8} {'───────':>10} {'─────':>12}")
    for h in p.holdings:
        print(f"  {h.ticker:<8} {h.shares:>8.1f} {h.cost:>10.2f} {h.cost_basis:>12,.2f}")
    print(f"\n  Total cost basis: ${p.total_cost_basis:>12,.2f}")

    if result.risk:
        r = result.risk
        print_section("RISK ANALYSIS", "")
        print(f"  Portfolio Risk Score : {r.portfolio_risk_score:.1f} / 100")
        print(f"  Concentration (top)  : {r.concentration_pct:.1f}%")
        print(f"\n  Per-ticker risk:")
        for t, s in r.ticker_risk_scores.items():
            bar = "█" * int(s / 5)
            print(f"    {t:<6} {s:>5.1f}  {bar}")
        if r.flags:
            print(f"\n  ⚠  Flags:")
            for flag in r.flags:
                print(f"     • {flag}")
        print(f"\n  {r.explanation}")

    if result.market_intel:
        m = result.market_intel
        print_section("MARKET INTELLIGENCE", "")
        score_bar = "▓" * int((m.sentiment_score + 1) * 15)
        print(f"  Sentiment Score : {m.sentiment_score:+.2f}   [{score_bar}]")
        print(f"  Confidence      : {m.confidence * 100:.0f}%")
        print(f"\n  Per-ticker sentiment:")
        for t, s in m.ticker_sentiments.items():
            arrow = "▲" if s > 0.4 else ("▼" if s < 0.1 else "►")
            print(f"    {t:<6} {s:+.2f}  {arrow}  {', '.join(m.ticker_signals[t][:2])}")
        print(f"\n  Macro: {m.macro_summary}")
        print(f"\n  {m.explanation}")

    if result.errors:
        print_section("ERRORS", "\n".join(f"  ✗ {e}" for e in result.errors))

    print(f"\n{'═' * 60}\n")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║      Portfolio Analysis Multi-Agent System  v0.1        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    portfolio = load_portfolio(path)
    print(f"\n  Loaded {len(portfolio.holdings)} holdings from {'file' if path else 'sample data'}")
    print("  Starting agent run...\n")

    result = run(portfolio)
    display(result)

    # Write JSON output to data/
    os.makedirs("data", exist_ok=True)
    out_path = "data/last_run.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"  Full output saved to {out_path}")


if __name__ == "__main__":
    main()
