import unittest
from unittest.mock import patch

import pandas as pd

from features.feature04_dynamic_portfolio import page


class DynamicAdjustedOpenTests(unittest.TestCase):
    def test_open_uses_same_adjustment_factor_as_close(self):
        index = pd.DatetimeIndex(["2020-01-02"])
        columns = pd.MultiIndex.from_product(
            [["Open", "Close", "Adj Close"], ["SPY"]]
        )
        raw = pd.DataFrame([[100.0, 110.0, 55.0]], index=index, columns=columns)
        page._download_prices.clear()

        with patch.object(page.yf, "download", return_value=raw):
            adjusted_close, adjusted_open = page._download_prices(
                ("SPY",), "2020-01-02", "2020-01-02"
            )

        self.assertEqual(float(adjusted_close.at[index[0], "SPY"]), 55.0)
        self.assertEqual(float(adjusted_open.at[index[0], "SPY"]), 50.0)


if __name__ == "__main__":
    unittest.main()
