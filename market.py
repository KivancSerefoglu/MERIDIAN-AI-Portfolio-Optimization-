"""Market data pipeline — Phase 1 (Person A)."""

from schemas import PortfolioInput


def get_portfolio_data(tickers: list[str]):
    """Pull 60-day price history, current price, sector, beta, market cap.

    Returns a clean pandas DataFrame. Results are cached locally.
    """
    raise NotImplementedError("Phase 1 — Person A")
