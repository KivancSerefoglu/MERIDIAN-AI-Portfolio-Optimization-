from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.risk_agent import ComputedMetrics

def debug_print_metrics(metrics: ComputedMetrics) -> None:
    """Pretty-print Stage 1 metrics for debugging."""
    import json
    from dataclasses import asdict
    
    metrics_dict = asdict(metrics)
    
    print("\n" + "="*70)
    print("STAGE 1 METRICS (Raw Computation)")
    print("="*70)
    
    print("\n[QUICK SUMMARY]")
    print(f"  Portfolio Value:           ${metrics.total_portfolio_value:,.2f}")
    print(f"  Portfolio Beta:            {metrics.portfolio_beta}")
    print(f"  Top Sector:                {metrics.max_sector_name} ({metrics.max_sector_weight:.1%})")
    print(f"  Avg Pairwise Correlation:  {metrics.avg_pairwise_correlation:.4f}")
    print(f"  Worst Drawdown:            {metrics.worst_drawdown_ticker} ({metrics.worst_drawdown_pct:.1%})")
    
    fc = metrics.factor_compression
    if fc:
        print(f"\n[HIDDEN FACTOR COMPRESSION]")
        print(f"  Holdings:                  {fc.num_holdings}")
        print(f"  Effective Independent Bets:{fc.effective_n}")
        print(f"  Compression Ratio:         {fc.compression_ratio:.1%}")
        print(f"  Top 3 Factors Explain:     {fc.variance_explained_top3:.1%} of variance")
        print(f"  Clusters:")
        for c in fc.clusters:
            rho_str = f"avg ρ = {c.avg_intra_correlation:.2f}" if c.avg_intra_correlation is not None else "standalone"
            print(f"    [{c.cluster_id + 1}] {c.dominant_sector:20s}  "
                f"{', '.join(c.tickers):40s}  "
                f"({rho_str})")
    else:
        print(f"\n[HIDDEN FACTOR COMPRESSION] Not computed")
    
    if metrics.high_corr_pairs:
        print(f"\n[HIGH CORRELATION PAIRS (>0.85)]")
        for pair in metrics.high_corr_pairs:
            print(f"  • {pair}")
    else:
        print(f"\n[HIGH CORRELATION PAIRS] None")
    
    print(f"\n[SECTOR EXPOSURES]")
    for exp in metrics.sector_exposures:
        print(f"  {exp.sector:25s}  "
              f"Portfolio: {exp.portfolio_weight:6.1%}  "
              f"Benchmark: {exp.benchmark_weight:6.1%}  "
              f"Deviation: {exp.deviation:+6.1%}")
    
    print(f"\n[MAX DRAWDOWNS]")
    for dd in metrics.drawdowns:
        print(f"  {dd.ticker:6s}  {dd.max_drawdown:>7.1%}  "
              f"({dd.drawdown_start} to {dd.drawdown_end})")
    
    print(f"\n[FULL METRICS JSON]")
    print(json.dumps(metrics_dict, indent=2))
    print("="*70 + "\n")