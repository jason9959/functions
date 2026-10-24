import datetime as dt
import unittest

from features.feature04_dynamic_portfolio.engine import resolve_recent_period


class DynamicPeriodTests(unittest.TestCase):
    def test_recent_period_uses_calendar_years(self):
        self.assertEqual(
            resolve_recent_period(dt.date(2026, 10, 24), 3),
            (dt.date(2023, 10, 24), dt.date(2026, 10, 24)),
        )

    def test_recent_period_handles_february_29(self):
        self.assertEqual(
            resolve_recent_period(dt.date(2024, 2, 29), 1),
            (dt.date(2023, 2, 28), dt.date(2024, 2, 29)),
        )

if __name__ == "__main__":
    unittest.main()
