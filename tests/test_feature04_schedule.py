import unittest

import pandas as pd

from features.feature04_dynamic_portfolio.engine import (
    CASH,
    _apply_market_overlay,
    _market_signal_details,
    MARKET_SIGNAL_SUPPORT,
    MARKET_SIGNAL_REQUIRED_BY_PRESETS,
    MARKET_SIGNAL_TESTED,
    normalize_market_signals,
    validate_preset_market_signals,
    _ranked,
    scheduled_trade_dates,
)


class ScheduledTradeDatesTests(unittest.TestCase):
    def test_monthly_schedule_preserves_start_day_and_uses_next_session(self):
        sessions = pd.bdate_range("2020-01-01", "2020-06-30")

        result = scheduled_trade_dates(
            sessions,
            pd.Timestamp("2020-02-03"),
            pd.Timestamp("2020-06-30"),
            "월별",
        )

        self.assertEqual(
            [(planned.date(), actual.date()) for planned, actual in result],
            [
                (pd.Timestamp("2020-02-03").date(), pd.Timestamp("2020-02-03").date()),
                (pd.Timestamp("2020-03-03").date(), pd.Timestamp("2020-03-03").date()),
                (pd.Timestamp("2020-04-03").date(), pd.Timestamp("2020-04-03").date()),
                (pd.Timestamp("2020-05-03").date(), pd.Timestamp("2020-05-04").date()),
                (pd.Timestamp("2020-06-03").date(), pd.Timestamp("2020-06-03").date()),
            ],
        )

    def test_quarterly_schedule_does_not_carry_start_day_forward(self):
        sessions = pd.bdate_range("2020-01-01", "2020-12-31")

        result = scheduled_trade_dates(
            sessions,
            pd.Timestamp("2020-02-03"),
            pd.Timestamp("2020-12-31"),
            "분기",
        )

        self.assertEqual(
            [(planned.date(), actual.date()) for planned, actual in result],
            [
                (pd.Timestamp("2020-02-03").date(), pd.Timestamp("2020-02-03").date()),
                (pd.Timestamp("2020-05-03").date(), pd.Timestamp("2020-05-04").date()),
                (pd.Timestamp("2020-08-03").date(), pd.Timestamp("2020-08-03").date()),
                (pd.Timestamp("2020-11-03").date(), pd.Timestamp("2020-11-03").date()),
            ],
        )

    def test_mid_month_start_uses_previous_session_only_for_first_rebalance(self):
        sessions = pd.bdate_range("2020-02-03", "2020-04-30")

        result = scheduled_trade_dates(
            sessions,
            pd.Timestamp("2020-02-18"),
            pd.Timestamp("2020-04-30"),
            "월별",
        )

        self.assertEqual(result[0][0], pd.Timestamp("2020-02-18"))
        self.assertEqual(result[0][1], pd.Timestamp("2020-02-18"))
        self.assertEqual(result[1][0], pd.Timestamp("2020-03-18"))
        self.assertEqual(result[1][1], pd.Timestamp("2020-03-18"))

    def test_raw_rank_values_and_exact_ties_are_stable(self):
        self.assertEqual(
            _ranked({"A": 1e-13, "B": 0.0, "C": 1e-13}, set(), ["A", "B", "C"]),
            ["A", "C", "B"],
        )

    def test_unverified_market_signals_are_rejected_at_engine_boundary(self):
        with self.assertRaises(ValueError):
            normalize_market_signals(["실업률"])
        self.assertEqual(normalize_market_signals(["실업률"], allow_unverified=True), ["실업률"])
        self.assertTrue(MARKET_SIGNAL_SUPPORT["S&P500 배당수익률"])

    def test_preset_signal_validation_rejects_unverified_values(self):
        validate_preset_market_signals(
            {"방어형": {"market_risk_signals": ["S&P500 배당수익률"]}},
            required_by_presets=MARKET_SIGNAL_REQUIRED_BY_PRESETS,
        )
        with self.assertRaises(ValueError):
            validate_preset_market_signals(
                {"잘못된 프리셋": {"market_risk_signals": ["실업률"]}},
                required_by_presets=MARKET_SIGNAL_REQUIRED_BY_PRESETS,
            )

    def test_required_by_preset_options_are_not_disabled(self):
        self.assertTrue(MARKET_SIGNAL_REQUIRED_BY_PRESETS <= set(MARKET_SIGNAL_SUPPORT))
        self.assertTrue(all(MARKET_SIGNAL_SUPPORT[name] for name in MARKET_SIGNAL_REQUIRED_BY_PRESETS))
        disabled = {name for name, enabled in MARKET_SIGNAL_SUPPORT.items() if not enabled}
        self.assertTrue(disabled.isdisjoint(MARKET_SIGNAL_REQUIRED_BY_PRESETS))
        self.assertFalse(MARKET_SIGNAL_TESTED["실업률"])

    def test_ten_year_two_year_signal_uses_persistent_negative_window(self):
        index = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        monthly = pd.date_range("2020-01-31", "2021-12-31", freq="M")
        prices = pd.DataFrame({"SPY": 100.0}, index=index)
        dgs10 = pd.Series(4.0, index=monthly)
        dgs2 = pd.Series(2.0, index=monthly)
        dgs10.loc["2021-07-31":] = 1.0
        dgs2.loc["2021-07-31":] = 2.0
        data = {"DGS10": dgs10, "DGS2": dgs2}
        config = {
            "market_risk": {
                "signals": ["장단기 금리차 역전 (10년-2년)"],
                "spread_lag_months": 0,
                "spread_active_months": 3,
            },
            "_market_data": data,
        }

        result = _market_signal_details(prices, pd.Timestamp("2021-10-15"), config)

        self.assertTrue(result[0]["위험 판정"])
        self.assertIn("2년", result[0]["판정 근거"])

    def test_ten_year_three_month_delay_threshold_and_duration(self):
        index = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        monthly = pd.date_range("2020-01-31", "2023-12-31", freq="M")
        prices = pd.DataFrame({"SPY": 100.0}, index=index)
        spread = pd.Series(0.0, index=monthly)
        # Event at 2022-01 is active at T=2022-08 through 2023-03
        spread.loc["2022-01-31"] = -0.500001
        data = {
            "DGS10": pd.Series(4.0, index=monthly),
            "DGS3MO": pd.Series(2.0, index=monthly),
            "T10Y3MM": spread,
        }
        config = {
            "market_risk": {
                "signals": ["장단기 금리차 역전 (10년-3개월)"],
                "spread_lag_months": 7,
                "spread_active_months": 8,
                "spread_threshold": -0.5,
            },
            "_market_data": data,
        }
        self.assertTrue(_market_signal_details(prices, pd.Timestamp("2022-08-15"), config)[0]["위험 판정"])
        self.assertFalse(_market_signal_details(prices, pd.Timestamp("2022-07-15"), config)[0]["위험 판정"])
        self.assertFalse(_market_signal_details(prices, pd.Timestamp("2023-04-15"), config)[0]["위험 판정"])
        spread.loc["2022-01-31"] = -0.5
        self.assertFalse(_market_signal_details(prices, pd.Timestamp("2022-08-15"), config)[0]["위험 판정"])

    def test_market_or_and_combination_uses_each_signal(self):
        index = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        monthly = pd.date_range("2020-01-31", "2021-12-31", freq="M")
        prices = pd.DataFrame({"SPY": 100.0}, index=index)
        dgs10 = pd.Series(4.0, index=monthly)
        dgs2 = pd.Series(2.0, index=monthly)
        dgs10.loc["2021-07-31":] = 1.0
        dividend = pd.Series(2.0, index=monthly)
        data = {"DGS10": dgs10, "DGS2": dgs2, "SP500_DIV_YIELD": dividend}
        base = {
            "market_risk": {
                "enabled": True,
                "signals": ["장단기 금리차 역전 (10년-2년)", "S&P500 배당수익률"],
                "spread_lag_months": 0,
                "spread_active_months": 3,
                "dividend_threshold": 1.32,
            },
            "_market_data": data,
        }
        base["market_risk"]["combine"] = "OR"
        _, _, or_summary = _apply_market_overlay(
            prices, pd.Timestamp("2021-10-15"), {"SPY": 1.0, CASH: 0.0}, base
        )
        self.assertTrue(or_summary["triggered"])

        base["market_risk"]["combine"] = "AND"
        _, _, and_summary = _apply_market_overlay(
            prices, pd.Timestamp("2021-10-15"), {"SPY": 1.0, CASH: 0.0}, base
        )
        self.assertFalse(and_summary["triggered"])


if __name__ == "__main__":
    unittest.main()
