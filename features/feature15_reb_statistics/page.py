from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common.metric_ui import metric as render_metric, metric_dataframe
from .api import fetch_statistics, resolve_api_key
from .analysis import analyze_statistics


PRESETS = {
    "전국 아파트 매매가격지수": {
        "stat_table_id": "A_2024_00045",
        "cycle": "MM",
        "region_id": "500001",
        "item_id": "100001",
        "value_kind": "index",
        "description": "전국 아파트 매매가격지수의 월별 흐름입니다.",
    },
    "전국 월별 지가변동률": {
        "stat_table_id": "A_2024_00903",
        "cycle": "MM",
        "region_id": "500001",
        "item_id": "100001",
        "value_kind": "rate",
        "description": "전국 토지가격의 월별 변동률입니다.",
    },
}


def _reset() -> None:
    for key in list(st.session_state):
        if key.startswith("reb_"):
            st.session_state.pop(key, None)


def _excel_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    export = frame.copy()
    export["날짜"] = export["날짜"].dt.strftime("%Y-%m-%d")
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="R-ONE 통계", index=False)
    return buffer.getvalue()


def _conditions() -> None:
    st.title("🏘️ 한국부동산원 부동산 통계")
    st.markdown(
        "<p class='step-caption'>R-ONE 공식 통계를 조회해 지표 추이와 기간별 변화를 분석합니다.</p>",
        unsafe_allow_html=True,
    )
    if st.button("↺ 조건 초기화", key="reb_reset"):
        _reset(); st.rerun()

    st.subheader("통계 선택")
    preset_name = st.selectbox(
        "분석할 통계",
        [*PRESETS, "직접 코드 입력"],
        key="reb_preset",
    )
    custom = preset_name == "직접 코드 입력"
    preset = PRESETS.get(preset_name, {})

    if custom:
        st.caption("R-ONE 통계코드 검색에서 확인한 코드를 입력하세요.")
        first, second = st.columns(2)
        with first:
            stat_table_id = st.text_input("통계표 코드", value="A_2024_00045", key="reb_stat_table_id").strip()
            region_id = st.text_input("지역·분류 코드", value="500001", key="reb_region_id").strip()
        with second:
            item_id = st.text_input("항목 코드", value="100001", key="reb_item_id").strip()
            cycle_label = st.selectbox("수록 주기", ["월", "분기", "연", "주"], key="reb_cycle_label")
        cycle = {"월": "MM", "분기": "QY", "연": "YY", "주": "WK"}[cycle_label]
        value_kind = st.selectbox(
            "값의 종류",
            ["지수", "기간 변동률"],
            key="reb_value_kind_label",
        )
        value_kind = "index" if value_kind == "지수" else "rate"
    else:
        stat_table_id = preset["stat_table_id"]
        region_id = preset["region_id"]
        item_id = preset["item_id"]
        cycle = preset["cycle"]
        value_kind = preset["value_kind"]
        st.info(preset["description"])

    st.divider(); st.subheader("조회 기간")
    current_year = dt.date.today().year
    start_col, end_col = st.columns(2)
    with start_col:
        start_year = st.number_input("시작 연도", min_value=1900, max_value=current_year, value=max(2006, current_year - 10), step=1, key="reb_start_year")
    with end_col:
        end_year = st.number_input("종료 연도", min_value=1900, max_value=current_year, value=current_year, step=1, key="reb_end_year")

    api_key = resolve_api_key(st.secrets)
    if api_key:
        st.success("R-ONE 인증키가 연결되어 전체 기간을 조회합니다.")
    else:
        st.warning("RONE_API_KEY가 없어 공식 샘플 모드로 조회합니다. 샘플 모드는 첫 5건만 제공합니다.")

    back_col, run_col = st.columns(2)
    if back_col.button("뒤로", key="reb_back", use_container_width=True):
        st.session_state["current_page"] = "feature"; st.rerun()
    run = run_col.button("결과 보기", type="primary", key="reb_run", use_container_width=True)
    if not run:
        return
    if int(start_year) > int(end_year):
        st.error("시작 연도는 종료 연도보다 늦을 수 없습니다."); return
    if not all((stat_table_id, region_id, item_id)):
        st.error("통계표·지역·항목 코드를 모두 입력해주세요."); return

    try:
        with st.spinner("한국부동산원 R-ONE 자료를 불러오고 있습니다..."):
            raw, metadata = fetch_statistics(
                stat_table_id=stat_table_id,
                cycle=cycle,
                region_id=region_id,
                item_id=item_id,
                start_year=int(start_year),
                end_year=int(end_year),
                api_key=api_key,
            )
        st.session_state["reb_result"] = {
            "data": analyze_statistics(raw, value_kind),
            "metadata": metadata,
            "preset_name": preset_name,
            "value_kind": value_kind,
            "codes": (stat_table_id, region_id, item_id, cycle),
        }
        st.session_state["current_page"] = "reb_statistics_results"
        st.rerun()
    except Exception as exc:
        st.error(f"R-ONE 통계를 불러오지 못했습니다: {exc}")


def _results() -> None:
    result = st.session_state.get("reb_result")
    if not result:
        st.session_state["current_page"] = "reb_statistics_conditions"; st.rerun()
    data: pd.DataFrame = result["data"]
    metadata = result["metadata"]
    unit = str(data.get("UI_NM", pd.Series([""])).iloc[0] or "")
    region = str(data.get("CLS_FULLNM", data.get("CLS_NM", pd.Series(["선택 지역"]))).iloc[0])
    item = str(data.get("ITM_NM", pd.Series(["통계값"])).iloc[0])

    st.title("🏘️ 한국부동산원 통계 분석 결과")
    st.caption(f"{result['preset_name']} · {region} · {item}")
    if metadata["sample_mode"]:
        st.warning(f"인증키가 없는 샘플 결과입니다. 전체 {metadata['total_count']:,}건 중 {metadata['returned_count']}건만 표시합니다.")

    first_value, last_value = float(data["값"].iloc[0]), float(data["값"].iloc[-1])
    total_change = float(data["누적 변화 (%)"].iloc[-1])
    change = data["기간 변동률 (%)"].dropna()
    cols = st.columns(4)
    with cols[0]: render_metric(st, "시작 값", f"{first_value:,.2f} {unit}", help="조회된 첫 시점의 통계값입니다.")
    with cols[1]: render_metric(st, "최근 값", f"{last_value:,.2f} {unit}", help="조회된 마지막 시점의 통계값입니다.")
    with cols[2]: render_metric(st, "누적 변화", f"{total_change:+.2f}%", help="첫 시점 대비 마지막 시점의 누적 변화입니다.")
    with cols[3]: render_metric(st, "상승 기간 비율", f"{(change.gt(0).mean() * 100 if len(change) else np.nan):.1f}%", help="기간 변동률이 0%보다 큰 관측치의 비율입니다.")

    st.subheader("통계값 추이")
    line = go.Figure(go.Scatter(x=data["날짜"], y=data["값"], mode="lines", name=item, line={"color": "#3182F6", "width": 2.5}))
    line.update_layout(xaxis_title="시점", yaxis_title=f"{item} ({unit})" if unit else item, hovermode="x unified", margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=420)
    st.plotly_chart(line, use_container_width=True)

    st.subheader("기간별 변동률")
    colors = np.where(data["기간 변동률 (%)"].fillna(0) >= 0, "#20C997", "#F04452")
    bar = go.Figure(go.Bar(x=data["날짜"], y=data["기간 변동률 (%)"], marker_color=colors, name="변동률"))
    bar.add_hline(y=0, line_color="#8B95A1", line_width=1)
    bar.update_layout(xaxis_title="시점", yaxis_title="변동률 (%)", margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=380)
    st.plotly_chart(bar, use_container_width=True)

    st.subheader("원자료")
    display = pd.DataFrame({
        "시점": data["WRTTIME_DESC"],
        "지역": data.get("CLS_FULLNM", data.get("CLS_NM")),
        "항목": data["ITM_NM"],
        f"값 ({unit})" if unit else "값": data["값"].round(4),
        "기간 변동률 (%)": data["기간 변동률 (%)"].round(4),
        "누적 변화 (%)": data["누적 변화 (%)"].round(4),
    })
    metric_dataframe(st, display, hide_index=True, width="stretch")

    back_col, save_col = st.columns(2)
    if back_col.button("뒤로", key="reb_result_back", use_container_width=True):
        st.session_state["current_page"] = "reb_statistics_conditions"; st.rerun()
    save_col.download_button(
        "결과 저장",
        data=_excel_bytes(display.assign(날짜=data["날짜"])),
        file_name="r-one-real-estate-statistics.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="reb_result_save",
        type="primary",
        use_container_width=True,
    )


def render(page: str) -> None:
    if page == "reb_statistics_conditions":
        _conditions()
    elif page == "reb_statistics_results":
        _results()
    else:
        st.session_state["current_page"] = "reb_statistics_conditions"
        st.rerun()

