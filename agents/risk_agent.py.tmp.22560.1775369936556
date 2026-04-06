"""
agents/risk_agent.py

Phase 2 Person A: Risk Analysis Agent (two-stage)
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import asdict, dataclass
from typing import Optional
import debug as debug
from utilities.market import get_portfolio_data
from agents.factor_compression import (
    FactorCluster,
    FactorCompression,
    compute_factor_compression,
)

from utilities.market import get_portfolio_data
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# S&P 500 approximate sector weights (GICS, 2024)
SP500_SECTOR_WEIGHTS: dict[str, float] = {
    "Technology": 0.29,
    "Health Care": 0.13,
    "Financials": 0.13,
    "Consumer Discretionary": 0.10,
    "Industrials": 0.09,
    "Communication Services": 0.09,
    "Consumer Staples": 0.06,
    "Energy": 0.04,
    "Utilities": 0.03,
    "Real Estate": 0.02,
    "Materials": 0.02,
}


@dataclass
class SectorExposure:
    sector: str
    portfolio_weight: float    # 0-1
    benchmark_weight: float    # 0-1
    deviation: float           # portfolio - benchmark


@dataclass
class DrawdownResult:
    ticker: str
    max_drawdown: float        # negative, e.g. -0.42
    drawdown_start: Optional[str]
    drawdown_end: Optional[str]


@dataclass
class ComputedMetrics:
    """Raw numbers from Stage 1 — no interpretation."""
    total_portfolio_value: float
    sector_exposures: list[SectorExposure]
    correlation_matrix: dict[str, dict[str, float]]
    portfolio_beta: float
    drawdowns: list[DrawdownResult]
    # Hidden Factor Compression Layer
    factor_compression: Optional[FactorCompression]
    # Derived summaries fed to LLM
    max_sector_weight: float
    max_sector_name: str
    avg_pairwise_correlation: float
    worst_drawdown_ticker: str
    worst_drawdown_pct: float
    high_corr_pairs: list[str]   # e.g. ["NVDA/MSFT (0.92)"]


@dataclass
class LLMInterpretation:
    """Output from Stage 2 — Gemini's reasoning layer."""
    risk_score: int            # 0-100 assigned by LLM
    critical_risks: list[str]  # ranked from most to least critical
    warnings: list[str]        # 2-3 plain-English investor warnings
    explanation: str           # full narrative paragraph


@dataclass
class RiskOutput:
    """Final output combining both stages — matches orchestrator contract."""
    # Stage 1
    computed: ComputedMetrics
    # Stage 2
    risk_score: int
    critical_risks: list[str]
    warnings: list[str]
    explanation: str


#  STAGE 1: COMPUTATION 

_SECTOR_NAME_MAP = {
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
    "Healthcare":         "Health Care",
    "Communication Services": "Communication Services",
}

ETF_SECTOR_OVERRIDES = {
    "GLD": "ETF",
    "TLT": "ETF",
    "SPY": "ETF",
}

def _normalize_sector(name: str) -> str:
    return _SECTOR_NAME_MAP.get(name, name)

def _sector_concentration(
    holdings: list[dict],
    total_value: float,
    market_data: dict[str, dict],
) -> list[SectorExposure]:
    sector_values: dict[str, float] = {}

    for h in holdings:
        ticker = h["ticker"]
        data = market_data.get(ticker, {})
        price = data.get("current_price") or 0
        sector = _normalize_sector(data.get("sector", "Unknown"))
        sector = ETF_SECTOR_OVERRIDES.get(ticker, sector)
        sector_values[sector] = sector_values.get(sector, 0) + h["shares"] * price

    return [
        SectorExposure(
            sector=s,
            portfolio_weight=round(v / total_value, 4) if total_value else 0,
            benchmark_weight=round(SP500_SECTOR_WEIGHTS.get(s, 0.0), 4),
            deviation=round(
                v / total_value - SP500_SECTOR_WEIGHTS.get(s, 0.0), 4
            ) if total_value else 0,
        )
        for s, v in sector_values.items()
    ]


def _correlation_matrix(
    prices: pd.DataFrame,
    lookback_days: int = 252,
) -> tuple[dict[str, dict[str, float]], list[str], float]:
    """Returns (corr_dict, high_corr_pairs, avg_pairwise_corr)."""
    returns = prices.pct_change().dropna()
    returns = returns.tail(lookback_days)

    if returns.shape[1] < 2:
        return {}, [], 0.0

    corr = returns.corr()
    corr_dict: dict[str, dict[str, float]] = {
        t1: {t2: round(float(corr.loc[t1, t2]), 4) for t2 in corr.columns}
        for t1 in corr.columns
    }

    tickers = list(corr.columns)
    high_corr_pairs: list[str] = []
    off_diag: list[float] = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            c = float(corr.iloc[i, j])
            off_diag.append(c)
            if c > 0.85:
                high_corr_pairs.append(f"{tickers[i]}/{tickers[j]} ({c:.2f})")

    avg_corr = round(float(np.mean(off_diag)), 4) if off_diag else 0.0
    return corr_dict, high_corr_pairs, avg_corr


def _portfolio_beta(
    holdings: list[dict],
    total_value: float,
    market_data: dict[str, dict],
) -> float:
    weighted = 0.0
    for h in holdings:
        ticker = h["ticker"]
        data = market_data.get(ticker, {})
        price = data.get("current_price") or 0
        beta = data.get("beta") or 1.0
        weight = (h["shares"] * price) / total_value if total_value else 0
        weighted += weight * beta
    return round(weighted, 3)


def _max_drawdowns(prices: pd.DataFrame) -> list[DrawdownResult]:
    results: list[DrawdownResult] = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            continue
        roll_max = series.cummax()
        dd_series = (series - roll_max) / roll_max
        max_dd = float(dd_series.min())
        trough_idx = dd_series.idxmin()
        peak_slice = series.loc[:trough_idx]
        peak_idx = peak_slice.idxmax() if not peak_slice.empty else None
        def _fmt_idx(idx):
            return str(idx.date()) if hasattr(idx, "date") else str(idx)

        results.append(DrawdownResult(
            ticker=ticker,
            max_drawdown=round(max_dd, 4),
            drawdown_start=_fmt_idx(peak_idx) if peak_idx is not None else None,
            drawdown_end=_fmt_idx(trough_idx),
        ))
    return results


def compute_metrics(portfolio: dict) -> ComputedMetrics:
    """
    Stage 1 entry point.
    Fetches market data via market.py (single fetch, cached) and computes
    all four risk metrics + factor compression. Returns ComputedMetrics —
    pure numbers, zero interpretation.
    """
    holdings: list[dict] = portfolio.get("holdings", [])
    if not holdings:
        raise ValueError("Portfolio must contain at least one holding.")

    for h in holdings:
        h["ticker"] = h["ticker"].upper()
    tickers = [h["ticker"] for h in holdings]

    import time
    market_data: dict = {}
    for attempt in range(3):
        market_data = get_portfolio_data(tickers)
        missing = [t for t in tickers if t not in market_data
                   or market_data[t].get("current_price") is None]
        if not missing:
            break
        print(f"[WARNING] Attempt {attempt+1}: missing data for {missing}. "
              f"Retrying in {2**attempt}s…")
        time.sleep(2 ** attempt)

    missing = [t for t in tickers if t not in market_data
               or market_data[t].get("current_price") is None]
    if missing:
        print(
            f"\n[WARNING] Could not fetch data for: {missing}\n"
            f"  These tickers are EXCLUDED from sector weights, beta, correlation,\n"
            f"  and drawdowns. This may suppress high-correlation pairs and\n"
            f"  understate sector concentration. Re-run to retry.\n"
        )

    # Build price DataFrame
    price_series = {}
    for t in tickers:
        if t in market_data:
            series = market_data[t]["price_history"]
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            idx = series.index
            if hasattr(idx, "tz") and idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            series = series.copy()
            series.index = pd.to_datetime(idx).normalize()
            price_series[t] = series

    prices = pd.DataFrame(price_series).ffill().dropna(how="any")

    if prices.empty:
        raise ValueError(f"No price data returned for tickers: {tickers}")

    total_value = sum(
        h["shares"] * (market_data[h["ticker"]].get("current_price") or 0)
        for h in holdings
        if h["ticker"] in market_data
    )
    if total_value == 0:
        raise ValueError("Portfolio market value is zero — check tickers.")

    sector_exposures = _sector_concentration(holdings, total_value, market_data)
    corr_dict, high_corr_pairs, avg_corr = _correlation_matrix(prices)
    beta = _portfolio_beta(holdings, total_value, market_data)
    drawdowns = _max_drawdowns(prices)

    # Hidden Factor Compression Layer (separate module)
    factor_compression = compute_factor_compression(prices, market_data)

    top_sector = max(sector_exposures, key=lambda e: e.portfolio_weight) if sector_exposures else None
    worst_dd = min(drawdowns, key=lambda d: d.max_drawdown) if drawdowns else None

    return ComputedMetrics(
        total_portfolio_value=round(total_value, 2),
        sector_exposures=sector_exposures,
        correlation_matrix=corr_dict,
        portfolio_beta=beta,
        drawdowns=drawdowns,
        factor_compression=factor_compression,
        max_sector_weight=top_sector.portfolio_weight if top_sector else 0.0,
        max_sector_name=top_sector.sector if top_sector else "N/A",
        avg_pairwise_correlation=avg_corr,
        worst_drawdown_ticker=worst_dd.ticker if worst_dd else "N/A",
        worst_drawdown_pct=worst_dd.max_drawdown if worst_dd else 0.0,
        high_corr_pairs=high_corr_pairs,
    )

#  STAGE 2: LLM INTERPRETATION (Gemini) 

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior portfolio risk analyst. Provide a concise, investor-grade risk assessment.

    Rules:
    - REASON about risks, don't describe numbers — explain WHY each risk matters.
    - Flag compounding risks explicitly (e.g. concentration + correlation = single point of failure).
    - Be ruthlessly concise: no redundancy, no restatement of raw numbers.
    - critical_risks: only important items, one punchy sentence each (≤25 words).
    - warnings: one actionable "what to do / watch out for" based on critical risks (≤20 words). Do not restate the risk — prescribe a response to it.
    - explanation: 3-5 sentences max. State dominant risk, how factors compound, net verdict.
    - risk_score: integer 0 (very safe) to 100 (extremely dangerous).
    - Pairwise correlation measures short-term co-movement; factor compression measures latent structural exposure.
    - If SPY appears in a high-correlation cluster, explicitly state “market beta clustering detected".                                 

    - Effective N interpretation guide:
        - >10 = well diversified
        - 6-10 = moderately concentrated
        - 3-5 = highly factor concentrated
        - <3 = extreme single-factor risk
                                 
    Only flag as a risk if:
    - A single sector exceeds 35%" of portfolio weight
    - Sector deviation from benchmark exceeds +20%
    - Beta > 1.3
    - Avg correlation > 0.65 or high-corr pairs exist
    - A single stock drawdown exceeds 40% AND its portfolio weight exceeds 10%
    - Always flag any single-stock drawdown exceeding 50%, regardless of position size.
    - Effective N / num_holdings ratio (compression_ratio) below 0.5 — meaning more than half the apparent diversification is illusory.
    - A portfolio with beta <1.0, no drawdown >40%, and avg correlation <0.3, scores above 60 require multiple multiple uncorrelated risk drivers; otherwise, risk is moderated.

    Do NOT flag underweights. Do NOT invent risks from missing sectors.

    Respond ONLY with valid JSON:
    {
      "risk_score": <integer 0-100>,
      "critical_risks": [<string>, ...],
      "warnings": [<string>, <string>, <string>],
      "explanation": <string>
    }
    No markdown, no preamble, no text outside the JSON object.
""")


def _build_metrics_prompt(metrics: ComputedMetrics) -> str:
    sector_lines = "\n".join(
        f"  - {e.sector}: {e.portfolio_weight:.1%} portfolio "
        f"vs {e.benchmark_weight:.1%} S&P500 "
        f"(deviation: {e.deviation:+.1%})"
        for e in sorted(metrics.sector_exposures, key=lambda e: -e.portfolio_weight)
    )
    drawdown_lines = "\n".join(
        f"  - {d.ticker}: {d.max_drawdown:.1%} "
        f"({d.drawdown_start} to {d.drawdown_end})"
        for d in sorted(metrics.drawdowns, key=lambda d: d.max_drawdown)
    )
    high_corr_text = (
        ", ".join(metrics.high_corr_pairs)
        if metrics.high_corr_pairs
        else "None above 0.85 threshold"
    )

    # Factor compression section
    fc = metrics.factor_compression
    if fc:
        cluster_lines = "\n".join(
            f"    Cluster {c.cluster_id + 1} ({c.dominant_sector}): "
            f"{', '.join(c.tickers)} "
            f"({'standalone' if c.avg_intra_correlation is None else f'avg intra-correlation: {c.avg_intra_correlation:.2f}'})"
            for c in fc.clusters
        )
        factor_section = textwrap.dedent(f"""\
        HIDDEN FACTOR COMPRESSION
        Holdings count: {fc.num_holdings}
        Effective independent bets: {fc.effective_n}
        Compression ratio: {fc.compression_ratio:.1%} (1.0 = perfectly diversified, lower = illusory diversification)
        Variance explained by top 3 factors: {fc.variance_explained_top3:.1%}
        Correlated clusters:
{cluster_lines}
        """)
    else:
        factor_section = "HIDDEN FACTOR COMPRESSION\n        Not computed (fewer than 2 holdings with price data).\n"

    return textwrap.dedent(f"""\
        Analyse the following portfolio risk metrics and return your assessment as JSON.

        === PORTFOLIO METRICS ===

        Total Market Value: ${metrics.total_portfolio_value:,.0f}

        SECTOR CONCENTRATION
        Dominant sector: {metrics.max_sector_name} at {metrics.max_sector_weight:.1%} of portfolio
        All sector exposures:
{sector_lines}

        CORRELATION (60-day pairwise return correlations)
        Average pairwise correlation: {metrics.avg_pairwise_correlation:.2f}
        Highly correlated pairs (>0.85): {high_corr_text}

        {factor_section}
        PORTFOLIO BETA
        Weighted-average beta: {metrics.portfolio_beta:.2f}
        (Market beta = 1.0. Values >1.2 are aggressive, >1.5 are high-risk.)

        MAX DRAWDOWN (peak-to-trough, past 1 year)
        Worst: {metrics.worst_drawdown_ticker} at {metrics.worst_drawdown_pct:.1%}
        All holdings:
{drawdown_lines}

        === INSTRUCTIONS ===
        Identify the 1-2 dominant risk factors, note any compounding interactions
        (especially if effective N reveals illusory diversification), and return JSON only.
    """)


def interpret_with_gemini(metrics: ComputedMetrics) -> LLMInterpretation:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Export it: export GEMINI_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_metrics_prompt(metrics)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
        ),
    )
    raw_text = response.candidates[0].content.parts[0].text.strip()

    if raw_text.startswith("```"):
        parts = raw_text.split("```")
        raw_text = parts[1] if len(parts) > 1 else raw_text
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Gemini returned non-JSON output.\nRaw:\n{response.text}"
        ) from exc

    return LLMInterpretation(
        risk_score=int(parsed.get("risk_score", 50)),
        critical_risks=parsed.get("critical_risks", []),
        warnings=parsed.get("warnings", []),
        explanation=parsed.get("explanation", ""),
    )


#  PUBLIC INTERFACE 

def risk_agent(portfolio: dict) -> RiskOutput:
    """
    Two-stage risk analysis pipeline.

    Args:
        portfolio: {"holdings": [{"ticker": str, "shares": int|float, "cost": float}]}

    Returns:
        RiskOutput — computed metrics (Stage 1) + LLM interpretation (Stage 2).
    """
    metrics = compute_metrics(portfolio)
    interpretation = interpret_with_gemini(metrics)

    return RiskOutput(
        computed=metrics,
        risk_score=interpretation.risk_score,
        critical_risks=interpretation.critical_risks,
        warnings=interpretation.warnings,
        explanation=interpretation.explanation,
    )


# SMOKE TEST (python agents/risk_agent.py) 

if __name__ == "__main__":
    sample_portfolio = {
        "holdings": [
            # --- Mega-cap tech cluster (high correlation stress test) ---
            {"ticker": "AAPL", "shares": 25, "cost": 180},
            {"ticker": "MSFT", "shares": 20, "cost": 310},
            {"ticker": "NVDA", "shares": 15, "cost": 450},
            {"ticker": "GOOGL", "shares": 30, "cost": 140},
            {"ticker": "AMZN", "shares": 22, "cost": 160},
            {"ticker": "META", "shares": 18, "cost": 300},

            # --- High beta “risk amplifiers” ---
            {"ticker": "TSLA", "shares": 12, "cost": 250},
            {"ticker": "PLTR", "shares": 40, "cost": 25},

            # --- Financial sector (cycle exposure) ---
            {"ticker": "JPM", "shares": 25, "cost": 150},
            {"ticker": "BAC", "shares": 40, "cost": 40},

            # --- Defensive sector (low correlation anchors) ---
            {"ticker": "PG", "shares": 35, "cost": 155},
            {"ticker": "KO", "shares": 60, "cost": 60},
            {"ticker": "PEP", "shares": 30, "cost": 170},

            # --- Energy (macro-sensitive divergence) ---
            {"ticker": "XOM", "shares": 30, "cost": 110},

            # --- Hedge / low correlation asset ---
            {"ticker": "GLD", "shares": 20, "cost": 190},

            # --- Market baseline proxy (optional compression anchor) ---
            {"ticker": "SPY", "shares": 10, "cost": 450},
        ]
    }

    print("=== STAGE 1: Computing metrics ===")
    m = compute_metrics(sample_portfolio)
    print(f"Portfolio value     : ${m.total_portfolio_value:,.0f}")
    print(f"Beta                : {m.portfolio_beta}")
    print(f"Top sector          : {m.max_sector_name} ({m.max_sector_weight:.0%})")
    print(f"Avg correlation     : {m.avg_pairwise_correlation:.2f}")
    print(f"Worst drawdown      : {m.worst_drawdown_ticker} {m.worst_drawdown_pct:.1%}")
    print(f"High-corr pairs     : {m.high_corr_pairs or 'none'}")

    if m.factor_compression:
        fc = m.factor_compression
        print(f"\n--- Factor Compression ---")
        print(f"Effective N         : {fc.effective_n} / {fc.num_holdings} holdings")
        print(f"Compression ratio   : {fc.compression_ratio:.1%}")
        print(f"Top 3 factors       : {fc.variance_explained_top3:.1%} of variance")
        for c in fc.clusters:
            rho = f"ρ={c.avg_intra_correlation:.2f}" if c.avg_intra_correlation is not None else "standalone"
            print(f"  Cluster {c.cluster_id+1} ({c.dominant_sector}): {', '.join(c.tickers)} ({rho})")

    print("\n=== STAGE 2: Gemini interpretation ===")
    result = risk_agent(sample_portfolio)
    print(f"Risk Score  : {result.risk_score} / 100")
    print(f"\nExplanation :\n{result.explanation}")
    print(f"\nCritical Risks (ranked):")
    for i, r in enumerate(result.critical_risks, 1):
        print(f"  {i}. {r}")
    print(f"\nWarnings:")
    for w in result.warnings:
        print(f"  * {w}")