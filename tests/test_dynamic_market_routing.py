import unittest
from unittest.mock import patch

import pandas as pd

from features.feature04_dynamic_portfolio import page


class _Response:
    content = b"fake-shiller-workbook"

    def raise_for_status(self):
        return None


class DynamicMarketRoutingTests(unittest.TestCase):
    def test_spreads_cannot_fall_through_to_fred(self):
        expected = pd.Series(
            [-0.5], index=pd.DatetimeIndex(["2024-01-31"]), name="T10Y3M"
        )
        page._download_fred_series.clear()
        with patch.object(page, "load_treasury_spreads", return_value={"T10Y3M": expected}) as loader, patch.object(
            page.requests, "get", side_effect=AssertionError("FRED must not be called")
        ):
            actual = page._download_fred_series("T10Y3M", "2024-01-01", "2024-01-31")
        loader.assert_called_once_with({"T10Y3M"}, "2024-01-01", "2024-01-31")
        pd.testing.assert_series_equal(actual, expected)

    def test_shiller_uses_data_sheet_and_decimal_year_header(self):
        rows = [[None] * 5 for _ in range(8)]
        rows[7] = ["Date", "P", "D", "E", "CPI"]
        rows.extend(
            [
                [2020.01, 100.0, 2.0, None, None],
                [2020.02, 110.0, 2.2, None, None],
            ]
        )
        raw = pd.DataFrame(rows)
        page._download_shiller_dividend_yield.clear()

        with patch.object(page.requests, "get", return_value=_Response()), patch.object(
            page.pd, "read_excel", return_value=raw
        ) as read_excel:
            result = page._download_shiller_dividend_yield("2020-01-01", "2020-02-29")

        self.assertEqual(read_excel.call_args.kwargs["sheet_name"], "Data")
        self.assertEqual(read_excel.call_args.kwargs["engine"], "xlrd")
        self.assertEqual(result.index.tolist(), [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")])
        self.assertEqual(result.tolist(), [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
