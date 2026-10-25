"""Stable, CSV-first loaders for dynamic-portfolio market signals."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import requests


SPREAD_COLUMNS = {
    "T10Y2Y": "2e2963b9-dc37-455b-8e66-b43a3ed4ff9e",
    "T10Y3M": "6ee33d8e-45c0-4a5a-b1a4-96574058cb46",
}
PACKAGED_SPREAD_CSV = Path(__file__).with_name("data") / "treasury_spreads_monthly.csv"
RUNTIME_SPREAD_CSV = Path(".cache") / "treasury_spreads_monthly.csv"


def _read_spread_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["T10Y2Y", "T10Y3M"], dtype=float)
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"금리 CSV에 Date 열이 없습니다: {path}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    for column in SPREAD_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.index = frame.index.to_period("M").to_timestamp("M")
    return frame[list(SPREAD_COLUMNS)].groupby(level=0).last()


def _download_series(
    series_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    request_get: Callable = requests.get,
) -> pd.Series:
    dataset_id = SPREAD_COLUMNS[series_id]
    url = f"https://autario.com/api/v1/public/datasets/{dataset_id}/data"
    response = request_get(
        url,
        params={"limit": 20_000, "filter": f"date:gte:{start.date().isoformat()}", "sort": "date:asc"},
        timeout=(8, 25),
        headers={"User-Agent": "stock-simulator/1.0"},
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows:
        raise ValueError(f"{series_id} 갱신 응답이 비어 있습니다.")
    dates = pd.to_datetime([row.get("date") for row in rows], errors="coerce")
    values = pd.to_numeric(pd.Series([row.get("value") for row in rows]), errors="coerce")
    daily = pd.Series(values.to_numpy(), index=dates, name=series_id).dropna().sort_index()
    daily = daily.loc[start:end]
    if daily.empty:
        raise ValueError(f"{series_id} 갱신 응답에 요청 기간 데이터가 없습니다.")
    periods = daily.index.to_period("M")
    monthly = daily.groupby(periods).last().dropna()
    monthly.index = monthly.index.to_timestamp("M")
    return monthly


def load_treasury_spreads(
    series_ids: set[str],
    start_iso: str,
    end_iso: str,
    *,
    packaged_path: Path = PACKAGED_SPREAD_CSV,
    cache_path: Path = RUNTIME_SPREAD_CSV,
    request_get: Callable = requests.get,
) -> dict[str, pd.Series]:
    """Use bundled monthly CSV data and update only missing months on demand.

    Runtime cache writes are best-effort because Streamlit Cloud storage is
    ephemeral.  The bundled CSV remains the deterministic baseline.
    """
    requested = set(series_ids) & set(SPREAD_COLUMNS)
    if not requested:
        return {}
    start = pd.Timestamp(start_iso).to_period("M").to_timestamp("M")
    end = pd.Timestamp(end_iso).to_period("M").to_timestamp("M")
    required_months = pd.period_range(start, end, freq="M")

    packaged = _read_spread_csv(packaged_path)
    try:
        cached = _read_spread_csv(cache_path)
    except (ValueError, OSError, pd.errors.ParserError):
        # A Streamlit instance can stop while replacing its ephemeral cache.
        # A damaged runtime cache must never invalidate the bundled baseline.
        cached = pd.DataFrame(columns=list(SPREAD_COLUMNS), dtype=float)
    combined = pd.concat([packaged, cached]).groupby(level=0).last().sort_index()

    def missing_for(series_id: str) -> pd.PeriodIndex:
        available = combined[series_id].dropna().index.to_period("M")
        return required_months.difference(available)

    missing_before = {series_id: missing_for(series_id) for series_id in requested}
    update_errors: dict[str, Exception] = {}
    if any(len(months) for months in missing_before.values()):
        updated = combined.copy()
        for series_id, months in missing_before.items():
            if not len(months):
                continue
            try:
                fetched = _download_series(
                    series_id,
                    months.min().to_timestamp(how="start"),
                    end,
                    request_get=request_get,
                )
            except (requests.RequestException, ValueError, OSError) as error:
                update_errors[series_id] = error
                continue
            updated = updated.combine_first(fetched.to_frame())
            updated.loc[fetched.index, series_id] = fetched
        combined = updated.sort_index()
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(".tmp")
            combined.rename_axis("Date").reset_index().to_csv(temp_path, index=False, date_format="%Y-%m-%d")
            temp_path.replace(cache_path)
        except OSError:
            pass

    missing_after = {series_id: missing_for(series_id) for series_id in requested}
    unresolved = {
        series_id: [str(month) for month in months[:6]]
        for series_id, months in missing_after.items()
        if len(months)
    }
    if unresolved:
        detail = "; ".join(f"{series_id}: {', '.join(months)}" for series_id, months in unresolved.items())
        update_detail = "; ".join(f"{series_id}: {error}" for series_id, error in update_errors.items())
        suffix = f" 자동 업데이트 실패: {update_detail}" if update_detail else ""
        raise ValueError(f"금리 CSV에서 필요한 월을 확보하지 못했습니다 ({detail}).{suffix}")

    return {
        series_id: combined.loc[start:end, series_id].dropna().rename(series_id)
        for series_id in requested
    }
