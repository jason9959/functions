import pandas as pd

from features.feature15_reb_statistics.api import parse_response
from features.feature15_reb_statistics.analysis import analyze_statistics, build_region_comparison


def _payload(values=(100.0, 102.0)):
    return {
        "SttsApiTblData": [
            {"head": [{"list_total_count": len(values)}, {"RESULT": {"CODE": "INFO-000", "MESSAGE": "정상"}}]},
            {
                "row": [
                    {
                        "WRTTIME_IDTFR_ID": f"2024{month:02d}",
                        "WRTTIME_DESC": f"2024년 {month}월",
                        "DTA_VAL": value,
                        "CLS_NM": "전국",
                        "ITM_NM": "지수",
                        "UI_NM": "지수",
                    }
                    for month, value in enumerate(values, 1)
                ]
            },
        ]
    }


def test_parse_monthly_rone_response():
    frame, total = parse_response(_payload(), "MM")
    assert total == 2
    assert frame["날짜"].tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
    assert frame["값"].tolist() == [100.0, 102.0]


def test_index_analysis_calculates_period_and_total_change():
    frame, _ = parse_response(_payload(), "MM")
    result = analyze_statistics(frame, "index")
    assert pd.isna(result.loc[0, "기간 변동률 (%)"])
    assert round(result.loc[1, "기간 변동률 (%)"], 6) == 2.0
    assert round(result.loc[1, "누적 변화 (%)"], 6) == 2.0


def test_rate_analysis_compounds_period_returns():
    frame, _ = parse_response(_payload((1.0, 2.0)), "MM")
    result = analyze_statistics(frame, "rate")
    assert result["기간 변동률 (%)"].tolist() == [1.0, 2.0]
    assert round(result.loc[1, "누적 변화 (%)"], 6) == 3.02


def test_region_comparison_uses_common_dates_and_ranks_growth():
    first, _ = parse_response(_payload((100.0, 110.0)), "MM")
    first["지역명"] = "지역A"
    second, _ = parse_response(_payload((100.0, 105.0)), "MM")
    second["지역명"] = "지역B"
    frame = pd.concat([first, second], ignore_index=True)
    normalized, ranking = build_region_comparison(frame, "index")
    assert round(normalized.loc[pd.Timestamp("2024-02-01"), "지역A"], 6) == 110.0
    assert ranking.iloc[0]["지역"] == "지역A"
    assert round(ranking.iloc[0]["누적 상승률 (%)"], 6) == 10.0
