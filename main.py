"""Portfolio Advisor — main entry point."""

from schemas import PortfolioInput

# ── Sample portfolios for testing ──

SAMPLE_PORTFOLIOS = {
    "tech_heavy": {
        "holdings": [
            {"ticker": "NVDA", "shares": 50, "cost": 120.0},
            {"ticker": "AAPL", "shares": 30, "cost": 150.0},
            {"ticker": "MSFT", "shares": 20, "cost": 310.0},
            {"ticker": "GOOGL", "shares": 15, "cost": 140.0},
            {"ticker": "META", "shares": 25, "cost": 300.0},
        ]
    },
    "conservative": {
        "holdings": [
            {"ticker": "JNJ", "shares": 40, "cost": 160.0},
            {"ticker": "PG", "shares": 35, "cost": 150.0},
            {"ticker": "KO", "shares": 50, "cost": 58.0},
            {"ticker": "VZ", "shares": 60, "cost": 38.0},
            {"ticker": "PFE", "shares": 80, "cost": 30.0},
        ]
    },
    "meme_stocks": {
        "holdings": [
            {"ticker": "GME", "shares": 100, "cost": 25.0},
            {"ticker": "AMC", "shares": 200, "cost": 8.0},
            {"ticker": "BBBY", "shares": 150, "cost": 5.0},
            {"ticker": "PLTR", "shares": 120, "cost": 18.0},
            {"ticker": "SOFI", "shares": 180, "cost": 7.0},
        ]
    },
}


def main():
    # Parse the sample portfolio
    portfolio = PortfolioInput.from_dict(SAMPLE_PORTFOLIOS["tech_heavy"])

    print("=" * 50)
    print("Portfolio Advisor")
    print("=" * 50)
    print(f"\nLoaded {len(portfolio.holdings)} holdings:")
    for h in portfolio.holdings:
        print(f"  {h.ticker:6s}  {h.shares:>6.0f} shares @ ${h.cost:.2f}")
    print(f"\nTickers: {portfolio.tickers}")
    print("\n✅ Phase 0 complete — scaffold is working.")
    print("   Next: implement data pipelines (Phase 1)")


if __name__ == "__main__":
    main()
