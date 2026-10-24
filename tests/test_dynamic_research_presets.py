import ast
import unittest
from pathlib import Path


def _load_presets():
    source_path = Path(__file__).parents[1] / "features" / "feature04_dynamic_portfolio" / "page.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RESEARCH_PRESETS"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("RESEARCH_PRESETS definition was not found")


class DynamicResearchPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presets = _load_presets()

    def test_defensive_preset_matches_research_definition(self):
        preset = self.presets["1. 분산투자 방어형"]
        self.assertEqual(preset["attack_tickers"], ["QQQ", "SOXX", "XLE", "XLK", "XLV", "EDV", "GLD", "PDBC"])
        self.assertEqual(preset["attack_mode"], "상대 모멘텀")
        self.assertEqual(preset["attack_select_count"], 4)
        self.assertEqual(preset["attack_relative"]["score_type"], "1-3-6-12개월 단순")
        self.assertEqual(preset["defense_tickers"], ["BIL", "IEF", "HYG"])
        self.assertEqual(preset["defense_mode"], "상대 모멘텀")
        self.assertTrue(preset["negative_same_momentum"])
        self.assertEqual(preset["negative_threshold"], 5)
        self.assertEqual(preset["risk_custom_momentum"]["months"], 11)
        self.assertEqual(preset["market_risk_signals"], ["S&P500 배당수익률"])
        self.assertAlmostEqual(preset["dividend_threshold"], 1.32)

    def test_aggressive_preset_matches_research_definition(self):
        preset = self.presets["2. 집중투자 공격형"]
        self.assertEqual(preset["attack_tickers"], ["QQQ", "SCHD", "VWO"])
        self.assertEqual(preset["attack_select_count"], 1)
        self.assertEqual(preset["attack_relative"]["score_type"], "3-6-12개월 단순")
        self.assertEqual(preset["defense_tickers"], ["GLD", "HYG", "PDBC"])
        self.assertEqual(preset["defense_mode"], "상대 & 절대 모멘텀 + 현금 스위치")
        self.assertEqual(preset["defense_relative"]["months"], 6)
        self.assertEqual(preset["defense_absolute"]["months"], 6)
        self.assertFalse(preset["defense_same_momentum"])
        self.assertFalse(preset["cash_replacement"])
        self.assertFalse(preset["defense_use_bil"])
        self.assertEqual(preset["risk_custom_momentum"]["months"], 12)
        self.assertEqual(
            preset["market_risk_signals"],
            ["장단기 금리차 역전 (10년-3개월)", "S&P500 배당수익률"],
        )
        self.assertEqual((preset["spread_lag"], preset["spread_active"]), (7, 8))
        self.assertAlmostEqual(preset["spread_threshold"], -0.5)
        self.assertAlmostEqual(preset["dividend_threshold"], 1.6)


if __name__ == "__main__":
    unittest.main()
