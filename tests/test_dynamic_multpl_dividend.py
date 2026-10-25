import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pandas as pd

from features.feature04_dynamic_portfolio.market_data import (
    load_sp500_dividend_yield,
    parse_multpl_dividend_yield,
)


SAMPLE_HTML = """
<table id="datatable">
  <tr><th>Date</th><th>Value</th></tr>
  <tr><td>Oct 23, 2026</td><td><abbr title="Estimate">x</abbr> 1.06%</td></tr>
  <tr><td>Oct 31, 2026</td><td><abbr title="Estimate">x</abbr> 1.05%</td></tr>
  <tr><td>Sep 30, 2026</td><td>1.07%</td></tr>
</table>
"""


class DynamicMultplDividendTests(unittest.TestCase):
    def test_parser_keeps_latest_row_and_estimate_flag(self):
        frame = parse_multpl_dividend_yield(SAMPLE_HTML).set_index("Date")
        self.assertEqual(frame.loc[pd.Timestamp("2026-10-31"), "SP500_DIV_YIELD"], 1.05)
        self.assertTrue(bool(frame.loc[pd.Timestamp("2026-10-31"), "Estimate"]))
        self.assertFalse(bool(frame.loc[pd.Timestamp("2026-09-30"), "Estimate"]))

    def test_packaged_csv_is_used_without_network(self):
        with TemporaryDirectory(dir=".") as temp_dir:
            packaged = Path(temp_dir) / "dividend.csv"
            cache = Path(temp_dir) / "cache.csv"
            pd.DataFrame({
                "Date": ["2026-09-30", "2026-10-31"],
                "SP500_DIV_YIELD": [1.07, 1.05],
                "Estimate": [False, True],
                "ObservationDate": ["2026-09-30", "2026-10-31"],
            }).to_csv(packaged, index=False)
            request_get = Mock(side_effect=AssertionError("network must not be called"))
            result = load_sp500_dividend_yield(
                "2026-09-01", "2026-10-25",
                packaged_path=packaged,
                cache_path=cache,
                request_get=request_get,
            )
        self.assertEqual(result.tolist(), [1.07, 1.05])
        self.assertEqual(result.attrs["estimate_months"], ["2026-10"])
        request_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
