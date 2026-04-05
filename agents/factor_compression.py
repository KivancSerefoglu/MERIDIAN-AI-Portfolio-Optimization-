"""
agents/factor_compression.py

Hidden Factor Compression Layer
Detects correlated clusters and computes effective number of independent bets
using eigenvalue decomposition of the return correlation matrix.

No external dependencies beyond numpy and pandas (already in the project).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── Dataclasses ──────────────────────────────────────────────────────

@dataclass
class FactorCluster:
    """A group of holdings that move together (same hidden factor)."""
    cluster_id: int
    tickers: list[str]
    avg_intra_correlation: float | None
    dominant_sector: str


@dataclass
class FactorCompression:
    """Output of the Hidden Factor Compression Layer."""
    num_holdings: int
    effective_n: float            # effective number of independent bets
    compression_ratio: float      # effective_n / num_holdings
    clusters: list[FactorCluster]
    eigenvalues: list[float]      # sorted descending
    variance_explained_top3: float


# ── Sector normalization (shared with risk_agent) ────────────────────

_SECTOR_NAME_MAP = {
    "Financial Services": "Financials",
    "Consumer Defensive": "Consumer Staples",
    "Healthcare":         "Health Care",
    "Communication Services": "Communication Services",
}


def _normalize_sector(name: str) -> str:
    return _SECTOR_NAME_MAP.get(name, name)


# ── Core math ────────────────────────────────────────────────────────

def _effective_n_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    """
    Participation ratio: Effective N = (Σ λ)² / Σ (λ²)
    Equals N for perfectly uncorrelated assets, approaches 1 for lockstep.
    """
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0
    total = float(np.sum(eigenvalues))
    sum_sq = float(np.sum(eigenvalues ** 2))
    if sum_sq == 0:
        return 0.0
    return round(total ** 2 / sum_sq, 2)


def _cluster_by_correlation(
    corr_matrix: pd.DataFrame,
    market_data: dict[str, dict],
    threshold: float = 0.45,
) -> list[FactorCluster]:
    """
    Greedy agglomerative clustering — no scipy dependency.
    Merges the two clusters with the highest avg inter-cluster correlation
    until no pair exceeds `threshold`.
    """
    tickers = list(corr_matrix.columns)
    n = len(tickers)
    if n == 0:
        return []

    clusters: list[set[int]] = [{i} for i in range(n)]

    changed = True
    while changed:
        changed = False
        best_score = -1.0
        best_pair = (0, 0)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pairs = [
                    float(corr_matrix.iloc[a, b])
                    for a in clusters[i]
                    for b in clusters[j]
                ]
                avg = float(np.mean(pairs))
                if avg > best_score:
                    best_score = avg
                    best_pair = (i, j)

        if best_score >= threshold:
            i, j = best_pair
            clusters[i] = clusters[i] | clusters[j]
            clusters.pop(j)
            changed = True

    # Build FactorCluster objects
    result: list[FactorCluster] = []
    for cid, members in enumerate(clusters):
        member_tickers = [tickers[idx] for idx in sorted(members)]

        # Avg intra-cluster correlation
        if len(members) > 1:
            member_list = sorted(members)
            intra_corrs = [
                float(corr_matrix.iloc[member_list[i], member_list[j]])
                for i in range(len(member_list))
                for j in range(i + 1, len(member_list))
            ]
            avg_intra = round(float(np.mean(intra_corrs)), 4)
        else:
            avg_intra = None

        # Dominant sector
        sectors = [_normalize_sector(market_data.get(t, {}).get("sector", "Unknown")) 
                for t in member_tickers]
        unique_sectors = list(dict.fromkeys(sectors))  # preserves order, dedupes
        dominant_sector = " + ".join(unique_sectors)  # "Technology + Financials"

        result.append(FactorCluster(
            cluster_id=cid,
            tickers=member_tickers,
            avg_intra_correlation=avg_intra,
            dominant_sector=dominant_sector,
        ))

    return result


# ── Public entry point ───────────────────────────────────────────────

def compute_factor_compression(
    prices: pd.DataFrame,
    market_data: dict[str, dict],
    lookback_days: int = 252,
) -> Optional[FactorCompression]:
    """
    Main entry point — call from risk_agent.compute_metrics().

    Args:
        prices:        DataFrame of adjusted close prices (columns = tickers).
        market_data:   Dict returned by market.get_portfolio_data().
        lookback_days: Rolling window for return correlations.

    Returns:
        FactorCompression or None if < 2 tickers have data.
    """
    returns = prices.pct_change().dropna().tail(lookback_days)

    if returns.shape[1] < 2:
        return None

    corr = returns.corr()
    num_holdings = corr.shape[0]

    # Eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(corr.values)
    eigenvalues = np.sort(eigenvalues)[::-1]

    effective_n = _effective_n_from_eigenvalues(eigenvalues)
    compression_ratio = round(effective_n / num_holdings, 3) if num_holdings > 0 else 0.0

    # Variance explained by top 3 eigenvalues
    total_var = float(np.sum(eigenvalues[eigenvalues > 0]))
    top3_var = float(np.sum(eigenvalues[:3])) if len(eigenvalues) >= 3 else total_var
    variance_explained_top3 = round(top3_var / total_var, 4) if total_var > 0 else 0.0

    clusters = _cluster_by_correlation(corr, market_data)

    return FactorCompression(
        num_holdings=num_holdings,
        effective_n=effective_n,
        compression_ratio=compression_ratio,
        clusters=clusters,
        eigenvalues=[round(float(ev), 4) for ev in eigenvalues],
        variance_explained_top3=variance_explained_top3,
    )