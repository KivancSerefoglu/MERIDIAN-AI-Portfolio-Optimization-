"""
orchestrator.py — LangGraph portfolio analysis pipeline.

Flow: START → [risk_node, market_intel_node] (parallel) → synthesizer_node → END
"""

from __future__ import annotations

import os
import textwrap
from typing import Optional, TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from langgraph.graph import END, START, StateGraph

from agents.market_intel_agent import market_intel_agent
from agents.risk_agent import RiskOutput as AgentRiskOutput
from agents.risk_agent import risk_agent
from schemas import Advisory, MarketIntelOutput, RiskFlag, RiskOutput, PortfolioInput

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# ── Graph state ──────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    portfolio: dict
    risk_output: Optional[AgentRiskOutput]
    market_intel_output: Optional[MarketIntelOutput]
    advisory: Optional[Advisory]


# ── News helper ──────────────────────────────────────────────────────────────

def _fetch_news(portfolio: dict) -> dict[str, str]:
    """
    Fetch recent news headlines per ticker via yfinance.
    Returns dict[ticker -> concatenated news text].
    Falls back to empty string on any failure.
    """
    import yfinance as yf

    news_dict: dict[str, str] = {}
    for holding in portfolio.get("holdings", []):
        ticker = str(holding.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        try:
            items = yf.Ticker(ticker).news or []
            parts: list[str] = []
            for item in items[:10]:
                title = item.get("content", {}).get("title", "") or item.get("title", "")
                desc = (
                    item.get("content", {}).get("summary", "")
                    or item.get("summary", "")
                    or item.get("description", "")
                )
                if title:
                    parts.append(f"- {title}" + (f": {desc}" if desc else ""))
            news_dict[ticker] = "\n".join(parts)
        except Exception:
            news_dict[ticker] = ""

    return news_dict


# ── Conversion: agent RiskOutput → schema RiskOutput ────────────────────────

def _to_schema_risk(agent_risk: AgentRiskOutput) -> RiskOutput:
    """Map the agent's detailed RiskOutput to the shared schemas.RiskOutput."""
    flags: list[RiskFlag] = [
        RiskFlag(category="critical_risk", severity="high", message=r)
        for r in agent_risk.critical_risks
    ] + [
        RiskFlag(category="warning", severity="medium", message=w)
        for w in agent_risk.warnings
    ]

    sector_concentration = {
        e.sector: e.portfolio_weight
        for e in agent_risk.computed.sector_exposures
    }
    max_drawdowns = {
        d.ticker: d.max_drawdown
        for d in agent_risk.computed.drawdowns
    }

    return RiskOutput(
        risk_score=float(agent_risk.risk_score),
        sector_concentration=sector_concentration,
        correlation_matrix=agent_risk.computed.correlation_matrix,
        portfolio_beta=agent_risk.computed.portfolio_beta,
        max_drawdowns=max_drawdowns,
        flags=flags,
    )


# ── Graph nodes ──────────────────────────────────────────────────────────────

def risk_node(state: GraphState) -> dict:
    result = risk_agent(state["portfolio"])
    return {"risk_output": result}


def market_intel_node(state: GraphState) -> dict:
    news = _fetch_news(state["portfolio"])
    result = market_intel_agent(state["portfolio"], news)
    return {"market_intel_output": result}


def synthesizer_node(state: GraphState) -> dict:
    agent_risk: AgentRiskOutput = state["risk_output"]
    intel: MarketIntelOutput = state["market_intel_output"]

    advisory = _synthesize_with_claude(agent_risk, intel)
    return {"advisory": advisory}


# ── Claude synthesizer ───────────────────────────────────────────────────────

def _synthesize_with_claude(
    agent_risk: AgentRiskOutput,
    intel: MarketIntelOutput,
) -> Advisory:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it: export GEMINI_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build factor compression section if available
    fc = agent_risk.computed.factor_compression
    if fc:
        cluster_lines = "\n".join(
            f"    Cluster {c.cluster_id + 1} ({c.dominant_sector}): "
            f"{', '.join(c.tickers)} "
            f"({'standalone' if c.avg_intra_correlation is None else f'avg intra-correlation: {c.avg_intra_correlation:.2f}'})"
            for c in fc.clusters
        )
        factor_section = textwrap.dedent(f"""\
        FACTOR COMPRESSION
        Holdings count: {fc.num_holdings}
        Effective N: {fc.effective_n} (compression ratio: {fc.compression_ratio:.1%})
        Variance explained by top 3 factors: {fc.variance_explained_top3:.1%}
        Correlated clusters:
{cluster_lines}
        """)
    else:
        factor_section = "\nFACTOR COMPRESSION\nNot computed (fewer than 2 holdings with price data).\n"

    risk_section = textwrap.dedent(f"""\
        RISK ANALYSIS
        -------------
        Risk Score      : {agent_risk.risk_score}/100
        Portfolio Beta  : {agent_risk.computed.portfolio_beta}
        Dominant Sector : {agent_risk.computed.max_sector_name} ({agent_risk.computed.max_sector_weight:.1%} of portfolio)
        Avg Correlation : {agent_risk.computed.avg_pairwise_correlation:.2f}
        Worst Drawdown  : {agent_risk.computed.worst_drawdown_ticker} ({agent_risk.computed.worst_drawdown_pct:.1%})
        High-Corr Pairs : {", ".join(agent_risk.computed.high_corr_pairs) or "None above 0.85"}

        {factor_section}
        Critical Risks (ranked):
        {chr(10).join(f"  {i + 1}. {r}" for i, r in enumerate(agent_risk.critical_risks))}

        Warnings:
        {chr(10).join(f"  * {w}" for w in agent_risk.warnings)}

        Analyst Explanation:
        {agent_risk.explanation}
    """)

    sentiment_lines = "\n".join(
        f"  {hs.ticker:6s} score={hs.sentiment_score:+.2f}  {hs.event_type:12s} ({hs.impact:6s}) — {hs.summary}"
        for hs in intel.holdings_sentiment
    )
    intel_section = textwrap.dedent(f"""\
        MARKET INTELLIGENCE
        -------------------
        Aggregate Sentiment : {intel.sentiment_score:+.4f}  (range -1 to +1)
        Active Catalysts    : {", ".join(intel.catalysts) or "None identified"}

        Per-holding sentiment:
{sentiment_lines}
    """)

    prompt = textwrap.dedent(f"""\
        You are a senior portfolio advisor. Below is a combined risk analysis and market
        intelligence report for an investor's portfolio. Synthesize both into a clear,
        actionable advisory.

        {risk_section}

        {intel_section}

        Produce your response in EXACTLY this format — no extra text before or after:

        SUMMARY:
        <A 3-5 sentence plain-English overview that explains the portfolio's overall risk
        posture and current market environment. Reference specific tickers and numbers.>

        RECOMMENDATIONS:
        1. <Specific, actionable recommendation referencing actual tickers or metrics>
        2. <Specific, actionable recommendation>
        3. <Specific, actionable recommendation>
        4. <Specific, actionable recommendation>
        5. <Specific, actionable recommendation>
    """)

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.3),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    response_text = response.candidates[0].content.parts[0].text.strip()

    # Parse SUMMARY and RECOMMENDATIONS sections
    summary = ""
    recommendations: list[str] = []

    if "SUMMARY:" in response_text and "RECOMMENDATIONS:" in response_text:
        rec_split = response_text.split("RECOMMENDATIONS:")
        summary = rec_split[0].replace("SUMMARY:", "").strip()
        for line in rec_split[1].strip().splitlines():
            line = line.strip()
            if line and line[0].isdigit() and len(line) > 2 and line[1] in ".)":
                recommendations.append(line[2:].strip())
    else:
        summary = response_text
        recommendations = ["Review portfolio manually — structured advisory could not be parsed."]

    schema_risk = _to_schema_risk(agent_risk)

    return Advisory(
        summary=summary,
        recommendations=recommendations,
        risk=schema_risk,
        market_intel=intel,
    )


# ── Graph assembly ───────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    builder.add_node("risk_node", risk_node)
    builder.add_node("market_intel_node", market_intel_node)
    builder.add_node("synthesizer_node", synthesizer_node)

    # Parallel fan-out from START
    builder.add_edge(START, "risk_node")
    builder.add_edge(START, "market_intel_node")

    # Fan-in: both branches must complete before synthesizer runs
    builder.add_edge("risk_node", "synthesizer_node")
    builder.add_edge("market_intel_node", "synthesizer_node")

    builder.add_edge("synthesizer_node", END)

    return builder.compile()


_graph = _build_graph()


# ── Public API ───────────────────────────────────────────────────────────────

def run_analysis(portfolio: dict) -> Advisory:
    """
    Run the full portfolio analysis pipeline and return an Advisory.

    Args:
        portfolio: {"holdings": [{"ticker": str, "shares": int|float, "cost": float}]}

    Returns:
        Advisory with plain-English summary, action recommendations, risk data,
        and market intelligence.

    Raises:
        ValueError: if portfolio is empty or a ticker cannot be resolved.
        RuntimeError: if an API call fails.
    """
    holdings = portfolio.get("holdings", [])
    if not holdings:
        raise ValueError("Portfolio must contain at least one holding.")

    # Validate ticker format early so errors surface before any API calls
    for h in holdings:
        ticker = str(h.get("ticker", "")).strip()
        if not ticker:
            raise ValueError(f"Invalid holding — missing ticker: {h}")

    initial_state: GraphState = {
        "portfolio": portfolio,
        "risk_output": None,
        "market_intel_output": None,
        "advisory": None,
    }

    final_state = _graph.invoke(initial_state)

    advisory: Advisory = final_state["advisory"]
    if advisory is None:
        raise RuntimeError("Graph completed but advisory was not produced.")

    return advisory


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_portfolio = {
        "holdings": [
            {"ticker": "NVDA", "shares": 50, "cost": 120},
            {"ticker": "AAPL", "shares": 30, "cost": 150},
        ]
    }

    print("=" * 60)
    print("Portfolio Analysis — running full pipeline...")
    print("=" * 60)
    print(f"Holdings: {[h['ticker'] for h in sample_portfolio['holdings']]}\n")

    advisory = run_analysis(sample_portfolio)

    print("ADVISORY SUMMARY")
    print("-" * 60)
    print(advisory.summary)

    print("\nACTION RECOMMENDATIONS")
    print("-" * 60)
    for i, rec in enumerate(advisory.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\nRISK SNAPSHOT")
    print("-" * 60)
    print(f"  Risk Score : {advisory.risk.risk_score}/100")
    print(f"  Beta       : {advisory.risk.portfolio_beta}")
    top_sector = max(advisory.risk.sector_concentration, key=advisory.risk.sector_concentration.get, default="N/A")
    print(f"  Top Sector : {top_sector} ({advisory.risk.sector_concentration.get(top_sector, 0):.1%})")

    print("\nMARKET INTEL SNAPSHOT")
    print("-" * 60)
    print(f"  Sentiment  : {advisory.market_intel.sentiment_score:+.4f}")
    if advisory.market_intel.catalysts:
        print("  Catalysts  :")
        for c in advisory.market_intel.catalysts[:]:
            print(f"    - {c}")
