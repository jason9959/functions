from __future__ import annotations

import pandas as pd


def analyze_statistics(frame: pd.DataFrame, value_kind: str) -> pd.DataFrame:
    """Add comparable period and cumulative changes to an R-ONE series."""
    result = frame.copy()
    if value_kind == "index":
        result["기간 변동률 (%)"] = result["값"].pct_change() * 100
        result["누적 변화 (%)"] = (result["값"] / result["값"].iloc[0] - 1) * 100
    else:
        result["기간 변동률 (%)"] = result["값"]
        result["누적 변화 (%)"] = ((1 + result["값"] / 100).cumprod() - 1) * 100
    return result

