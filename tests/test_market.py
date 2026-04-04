import unittest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.market import get_portfolio_data

class TestMarketData(unittest.TestCase):

    def test_get_portfolio_data(self):
        """Test the get_portfolio_data function with mock tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        
        # Call the function
        result = get_portfolio_data(tickers)

        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Check if the DataFrame has the expected columns
        expected_columns = ["ticker", "current_price", "sector", "beta", "market_cap", "price_history"]
        self.assertTrue(all(column in result.columns for column in expected_columns))

        # Check if the DataFrame has data for all tickers
        self.assertEqual(len(result), len(tickers))

        # Check if price history is a list
        for price_history in result["price_history"]:
            self.assertIsInstance(price_history, list)

if __name__ == "__main__":
    unittest.main()