from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common.metric_ui import metric_dataframe, render_metric_table
from common.metrics import performance_summary
from .backtest import run_backtest
from .data import INDICATORS, fetch_current_fear_greed, fetch_fear_greed, fetch_prices


def _rating(score: float) -> str:
    return "극단적 공포" if score < 25 else "공포" if score < 45 else "중립" if score <= 55 else "탐욕" if score < 75 else "극단적 탐욕"


def _reset() -> None:
    for key in list(st.session_state):
        if key.startswith("fg_"):
            st.session_state.pop(key, None)
    st.session_state["fear_greed_result"] = None


def _conditions() -> None:
    st.title("😨 공포·탐욕 지수 리밸런싱")
    st.markdown("<p class='step-caption'>시장 심리 지표가 기준값에 도달했을 때 포트폴리오를 목표 비율로 되돌립니다.</p>", unsafe_allow_html=True)
    if st.button("↺ 조건 초기화", key="fg_reset"):
        _reset(); st.rerun()

    try:
        current = fetch_current_fear_greed()
        score = float(current["values"].get("fear_and_greed_historical", np.nan))
        if np.isfinite(score):
            st.info(f"최근 종합 지수: **{score:.1f} · {_rating(score)}** ({current['date']:%Y-%m-%d})")
    except Exception as exc:
        st.warning(f"현재 지수를 불러오지 못했습니다. 백테스트 실행 시 다시 시도합니다. ({exc})")

    today = dt.date.today()
    start_col, end_col, initial_col = st.columns(3)
    with start_col:
        start = st.date_input("시작 날짜", value=st.session_state.setdefault("fg_start", today - dt.timedelta(days=365 * 5)), min_value=dt.date(2010, 1, 1), max_value=today, key="fg_start")
    with end_col:
        end = st.date_input("종료 날짜", value=st.session_state.setdefault("fg_end", today), min_value=dt.date(2010, 1, 1), max_value=today, key="fg_end")
    with initial_col:
        initial = st.number_input("초기 투자금", min_value=1.0, step=1000.0, value=float(st.session_state.setdefault("fg_initial", 10000.0)), key="fg_initial")

    left, right = st.columns(2)
    with left:
        investment_type = st.selectbox("투자 방식", ["거치식", "적립식"], key="fg_investment_type")
        contribution = st.number_input("정기 적립금", min_value=1.0, step=100.0, value=float(st.session_state.setdefault("fg_contribution", 1000.0)), key="fg_contribution", disabled=investment_type != "적립식")
    with right:
        rebalance_frequency = st.selectbox("리밸런싱 확인 주기", ["매일", "매월", "매분기", "매년"], key="fg_rebalance")
        contribution_frequency = st.selectbox("적립 주기", ["매월", "매분기", "매년"], key="fg_contribution_frequency", disabled=investment_type != "적립식")

    st.divider(); st.subheader("지표 조건")
    metric_col, direction_col, threshold_col = st.columns([2, 1, 1])
    with metric_col:
        indicator_key = st.selectbox("리밸런싱에 사용할 지표", list(INDICATORS), format_func=lambda key: INDICATORS[key], key="fg_indicator")
    with direction_col:
        direction = st.selectbox("조건", ["이하일 때", "이상일 때"], key="fg_direction")
    with threshold_col:
        threshold = st.number_input("기준값", min_value=0.0, max_value=100.0, value=float(st.session_state.setdefault("fg_threshold", 25.0)), step=1.0, key="fg_threshold")

    st.divider(); st.subheader("포트폴리오 구성")
    st.caption("티커와 목표 비중의 합계를 100%로 입력하세요.")
    items = []
    for index in range(5):
        ticker_col, weight_col = st.columns([3, 1])
        with ticker_col:
            ticker = st.text_input(f"자산 {index + 1}", placeholder="예: SPY 또는 QQQ", key=f"fg_ticker_{index}").strip().upper()
        with weight_col:
            weight = st.number_input(f"비중 {index + 1} (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"fg_weight_{index}")
        if ticker and weight > 0:
            items.append({"ticker": ticker, "weight": float(weight)})
    st.caption(f"현재 입력 비중: {sum(item['weight'] for item in items):.0f}%")

    back_col, result_col = st.columns(2)
    with back_col:
        back = st.button("뒤로", key="fg_back", use_container_width=True)
    with result_col:
        run = st.button("결과 보기", type="primary", key="fg_run", use_container_width=True)
    if back:
        st.session_state["current_page"] = "feature"; st.rerun()
    if not run:
        return
    if start >= end:
        st.error("시작 날짜는 종료 날짜보다 앞서야 합니다."); return
    if not items or len({item["ticker"] for item in items}) != len(items):
        st.error("서로 다른 티커와 비중을 입력해주세요."); return
    total = sum(item["weight"] for item in items)
    if not np.isclose(total, 100.0):
        st.error("포트폴리오 비중의 합계가 100%여야 합니다."); return
    try:
        tickers = tuple(item["ticker"] for item in items)
        prices = fetch_prices(tickers, start.isoformat(), end.isoformat())
        fg = fetch_fear_greed(start.isoformat(), end.isoformat())
        if indicator_key not in fg.columns:
            raise ValueError("선택한 지표의 데이터가 없습니다.")
        calc = run_backtest(prices, fg[indicator_key], np.array([item["weight"] / 100 for item in items]), float(initial), rebalance_frequency, float(threshold), direction, float(contribution if investment_type == "적립식" else 0), contribution_frequency)
        st.session_state["fear_greed_result"] = {**calc, "items": items, "indicator_key": indicator_key, "start": start, "end": end, "initial": float(initial), "investment_type": investment_type, "rebalance_frequency": rebalance_frequency, "direction": direction, "threshold": float(threshold), "contribution": float(contribution if investment_type == "적립식" else 0), "contribution_frequency": contribution_frequency}
        st.session_state["current_page"] = "fear_greed_results"; st.rerun()
    except Exception as exc:
        st.error(f"백테스트를 계산하지 못했습니다: {exc}")


def _results() -> None:
    result = st.session_state.get("fear_greed_result")
    if not result:
        st.session_state["current_page"] = "fear_greed_conditions"; st.rerun(); return
    values = result["values"]; invested = result["invested"]; indicator = result["indicator"]
    st.title("😨 공포·탐욕 지수 리밸런싱 결과")
    st.caption(f"{result['start']} ~ {result['end']} · {INDICATORS[result['indicator_key']]} · {result['direction']} {result['threshold']:.0f} · {result['rebalance_frequency']}")
    figure, axis = plt.subplots(figsize=(12, 5.4)); axis2 = axis.twinx()
    axis.plot(values.index, values, color="#3182F6", linewidth=2.2, label="포트폴리오 자산")
    axis.plot(invested.index, invested, color="#8B95A1", linestyle="--", label="총 투입금")
    axis2.plot(indicator.index, indicator, color="#F04452", alpha=.65, label="선택 지표")
    axis2.axhline(result["threshold"], color="#F04452", linestyle=":", alpha=.8)
    axis.set_ylabel("자산 가치"); axis2.set_ylabel("지표 점수"); axis.grid(alpha=.2)
    lines, labels = axis.get_legend_handles_labels(); lines2, labels2 = axis2.get_legend_handles_labels(); axis.legend(lines + lines2, labels + labels2, loc="upper left")
    st.pyplot(figure); plt.close(figure)
    final_value = float(values.iloc[-1]); total_invested = float(invested.iloc[-1])
    metric_cols = st.columns(4)
    metric_cols[0].metric("최종 자산", f"{final_value:,.0f}")
    metric_cols[1].metric("총 투입금", f"{total_invested:,.0f}")
    metric_cols[2].metric("총 수익률", f"{(final_value / total_invested - 1) * 100:+.2f}%")
    metric_cols[3].metric("리밸런싱 횟수", f"{len(result['events'][result['events']['type'].eq('조건 충족 리밸런싱')]) if not result['events'].empty else 0}회")
    contributions = invested.diff().fillna(0)
    perf = performance_summary(values, contributions, float(invested.iloc[0]))
    render_metric_table({"공포·탐욕 조건 전략": perf})
    st.subheader("리밸런싱 이벤트")
    if result["events"].empty: st.info("조건을 충족한 리밸런싱이 없었습니다.")
    else: metric_dataframe(st, result["events"], hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        if st.button("뒤로", key="fg_result_back", use_container_width=True): st.session_state["current_page"] = "fear_greed_conditions"; st.rerun()
    with right:
        st.download_button("결과 저장", result["events"].to_csv(index=False).encode("utf-8-sig"), "fear-greed-rebalance-events.csv", "text/csv", key="fg_save", use_container_width=True)


def render(page: str) -> None:
    if page == "fear_greed_conditions": _conditions()
    elif page == "fear_greed_results": _results()

