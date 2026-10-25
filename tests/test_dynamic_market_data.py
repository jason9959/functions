import tempfile
import unittest
from pathlib import Path

import pandas as pd

from features.feature04_dynamic_portfolio.market_data import load_treasury_spreads


class _Response:
    def __init__(self, rows):
        self._rows = rows

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": self._rows}


class DynamicMarketDataTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_complete_packaged_months_never_use_network(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            packaged = root / "packaged.csv"
            cache = root / "cache.csv"
            self._write_csv(packaged, [
                {"Date": "2024-01-31", "T10Y2Y": -0.2, "T10Y3M": -1.1},
                {"Date": "2024-02-29", "T10Y2Y": -0.1, "T10Y3M": -1.0},
            ])

            def fail_if_called(*args, **kwargs):
                raise AssertionError("network must not be used for complete CSV data")

            result = load_treasury_spreads(
                {"T10Y3M"}, "2024-01-01", "2024-02-29",
                packaged_path=packaged, cache_path=cache, request_get=fail_if_called,
            )
            self.assertEqual(result["T10Y3M"].tolist(), [-1.1, -1.0])
            self.assertFalse(cache.exists())

    def test_only_missing_months_are_updated_and_cached(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            packaged = root / "packaged.csv"
            cache = root / "runtime" / "cache.csv"
            self._write_csv(packaged, [
                {"Date": "2024-01-31", "T10Y2Y": -0.2, "T10Y3M": -1.1},
            ])
            calls = []

            def request_get(url, **kwargs):
                calls.append(kwargs["params"])
                return _Response([
                    {"date": "2024-02-01", "value": -0.9},
                    {"date": "2024-02-29", "value": -0.8},
                ])

            result = load_treasury_spreads(
                {"T10Y3M"}, "2024-01-01", "2024-02-29",
                packaged_path=packaged, cache_path=cache, request_get=request_get,
            )
            self.assertEqual(len(calls), 1)
            self.assertIn("2024-02-01", calls[0]["filter"])
            self.assertEqual(result["T10Y3M"].tolist(), [-1.1, -0.8])
            self.assertTrue(cache.exists())

    def test_unresolved_month_has_actionable_error(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            packaged = root / "packaged.csv"
            self._write_csv(packaged, [
                {"Date": "2024-01-31", "T10Y2Y": -0.2, "T10Y3M": -1.1},
            ])

            def request_get(url, **kwargs):
                return _Response([{"date": "2024-01-31", "value": -1.1}])

            with self.assertRaisesRegex(ValueError, "T10Y3M: 2024-02"):
                load_treasury_spreads(
                    {"T10Y3M"}, "2024-01-01", "2024-02-29",
                    packaged_path=packaged, cache_path=root / "cache.csv", request_get=request_get,
                )

    def test_broken_runtime_cache_does_not_override_packaged_data(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            packaged = root / "packaged.csv"
            cache = root / "cache.csv"
            self._write_csv(packaged, [
                {"Date": "2024-01-31", "T10Y2Y": -0.2, "T10Y3M": -1.1},
            ])
            cache.write_text("not,a,valid,spread,cache\n", encoding="utf-8")

            result = load_treasury_spreads(
                {"T10Y2Y"}, "2024-01-01", "2024-01-31",
                packaged_path=packaged, cache_path=cache,
                request_get=lambda *args, **kwargs: self.fail("network must not be used"),
            )
            self.assertEqual(result["T10Y2Y"].tolist(), [-0.2])


if __name__ == "__main__":
    unittest.main()
