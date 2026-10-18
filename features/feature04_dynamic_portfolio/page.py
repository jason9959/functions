"""Streamlit screens for the dynamic portfolio backtest."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import streamlit as st
import yfinance as yf

from .catalog import display_name, normalize_ticker, suggestions
from .engine import MomentumConfig, run_backtest
from .results import download_summary_csv, render as render_results


EARLIEST_DATE = dt.date(1900, 1, 1)
BENCHMARK_LABELS = ["S&P500", "NASDAQ100", "KOSPI"]
ATTACK_MODES = [
    "상대 모멘텀", "절대 모멘텀", "절대 모멘텀 + 방어 스위치",
    "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치",
]
DEFENSE_MODES = [
    "전체 동일 비중", "상대 모멘텀", "절대 모멘텀",
    "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 현금 스위치",
]
SCORE_TYPES = [
    "1-3-6개월 단순", "3-6-12개월 단순", "1-3-6-12개월 단순",
    "1-3-6-9-12개월 단순", "1-3-6-12개월 가중",
]


def _default_form() -> dict[str, Any]:
    today = dt.date.today()
    return {
        "dynamic_start": today - dt.timedelta(days=365 * 10),
        "dynamic_end": today,
        "dynamic_initial": 10_000.0,
        "dynamic_investment_type": "거치식",
        "dynamic_rebalance": "월별",
        "dynamic_benchmark": "S&P500",
        "dynamic_contribution": 1_000.0,
        "dynamic_contribution_frequency": "매월",
        "dynamic_market": "해외 ETF",
        "dynamic_attack_mode": "상대 모멘텀",
        "dynamic_attack_select_count": 1,
        "dynamic_defense_enabled": False,
        "dynamic_defense_mode": "전체 동일 비중",
        "dynamic_defense_select_count": 1,
        "dynamic_include_cash": True,
        "dynamic_same_momentum": True,
        "dynamic_fixed_defense": False,
        "dynamic_fixed_defense_weight": 20.0,
        "dynamic_use_transaction_cost": False,
        "dynamic_transaction_cost": 0.1,
    }


def _initialize_state() -> None:
    for key, value in _default_form().items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("dynamic_attack_tickers", ["SPY", "QQQ"])
    st.session_state.setdefault("dynamic_defense_tickers", ["IEF", "IAU"])
    st.session_state.setdefault("dynamic_show_results", False)
    st.session_state.setdefault("dynamic_result", None)
    st.session_state.setdefault("dynamic_result_config", None)
    st.session_state.setdefault("dynamic_error", None)


def _reset_form() -> None:
    for key in list(st.session_state):
        if key.startswith("dynamic_") and key not in {"dynamic_show_results", "dynamic_result", "dynamic_result_config"}:
            del st.session_state[key]
    for key, value in _default_form().items():
        st.session_state[key] = value
    st.session_state["dynamic_attack_tickers"] = ["SPY", "QQQ"]
    st.session_state["dynamic_defense_tickers"] = ["IEF", "IAU"]
    st.session_state["dynamic_result"] = None
    st.session_state["dynamic_result_config"] = None
    st.session_state.pop("_dynamic_market_seen", None)
    st.session_state.pop("dynamic_saved_widget_values", None)


def _render_error_dialog() -> None:
    message = st.session_state.get("dynamic_error")
    if not message:
        return

    @st.dialog("요청을 완료하지 못했어요", dismissible=False)
    def dialog() -> None:
        st.error(message)
        st.caption("입력 조건이나 데이터 상태를 확인한 뒤 다시 실행해주세요.")
        if st.button("닫기", type="primary", key="close_dynamic_error", use_container_width=True):
            st.session_state["dynamic_error"] = None
            st.rerun()

    dialog()


def _asset_selector(role: str, market: str) -> None:
    state_key = f"dynamic_{role}_tickers"
    label = "공격자산" if role == "attack" else "방어자산"
    st.markdown(f"#### {label}")
    st.caption("티커 또는 ETF 이름을 입력해 후보를 추가하세요. 등록된 항목은 제거할 수 있습니다.")
    query = st.text_input(
        f"{label} 검색", key=f"dynamic_{role}_query", placeholder="예: QQQ, KODEX, ACE",
        help="검색어와 일치하는 ETF를 제안합니다. 목록에 없는 Yahoo Finance 티커도 직접 추가할 수 있습니다.",
    )
    matches = suggestions(query, market)
    selected_ticker = None
    if matches:
        labels = [f"{ticker} · {name}" for ticker, name in matches]
        selected_label = st.selectbox(f"{label} 검색 결과", labels, key=f"dynamic_{role}_suggestion")
        selected_ticker = matches[labels.index(selected_label)][0]
    elif query.strip():
        selected_ticker = normalize_ticker(query, market)

    if st.button(f"{label}에 추가", key=f"dynamic_add_{role}", use_container_width=True, disabled=not selected_ticker):
        if selected_ticker and selected_ticker not in st.session_state[state_key]:
            st.session_state[state_key].append(selected_ticker)
        st.rerun()

    selected = list(st.session_state[state_key])
    if not selected:
        st.info(f"등록된 {label}이 없습니다.")
        return
    for index, ticker in enumerate(selected):
        text_col, button_col = st.columns([5, 1])
        text_col.markdown(f"**{display_name(ticker, market)}**")
        if button_col.button("제거", key=f"dynamic_remove_{role}_{index}", use_container_width=True):
            st.session_state[state_key] = [value for value in selected if value != ticker]
            st.rerun()


def _momentum_input(prefix: str, title: str) -> MomentumConfig:
    st.markdown(f"##### {title}")
    kind = st.radio(
        f"{title} 산출 기준", ["수익률", "이동평균", "모멘텀 스코어"], horizontal=True,
        key=f"{prefix}_kind",
        help="수익률은 과거 가격 변화, 이동평균은 현재가와 일별 종가 평균의 비율, 모멘텀 스코어는 여러 기간 수익률의 조합입니다.",
    )
    if kind == "모멘텀 스코어":
        score_type = st.selectbox("스코어 구성", SCORE_TYPES, key=f"{prefix}_score_type")
        return MomentumConfig(kind=kind, score_type=score_type)
    months = st.selectbox("기준 기간", list(range(1, 13)), format_func=lambda value: f"{value}개월", key=f"{prefix}_months")
    return MomentumConfig(kind=kind, months=int(months))


def _uses_relative(mode: str) -> bool:
    return mode in {"상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치"}


def _uses_absolute(mode: str) -> bool:
    return mode not in {"상대 모멘텀", "전체 동일 비중"}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _download_prices(symbols: tuple[str, ...], start_iso: str, end_iso: str) -> pd.DataFrame:
    raw = yf.download(
        list(symbols), start=pd.Timestamp(start_iso), end=pd.Timestamp(end_iso) + pd.Timedelta(days=1),
        auto_adjust=False, progress=False, threads=False, group_by="column",
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance에서 가격 데이터를 가져오지 못했습니다.")
    if isinstance(raw.columns, pd.MultiIndex):
        level = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[level].copy()
    else:
        column = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[column]].copy()
        prices.columns = [symbols[0]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    return prices.apply(pd.to_numeric, errors="coerce").sort_index()


def _prepare_prices(config: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    market = config["market"]
    requested = list(dict.fromkeys(config["attack_tickers"] + config["defense_tickers"]))
    if market == "해외 ETF":
        benchmark_source = {"S&P500": "SPY", "NASDAQ100": "QQQ", "KOSPI": "069500.KS"}[config["benchmark"]]
        extra = [benchmark_source, "BIL"] + (["KRW=X"] if config["benchmark"] == "KOSPI" else [])
    else:
        benchmark_source = {"S&P500": "360750.KS", "NASDAQ100": "133690.KS", "KOSPI": "069500.KS"}[config["benchmark"]]
        extra = [benchmark_source, "BIL", "KRW=X"]
    symbols = tuple(dict.fromkeys(requested + extra))
    lookback = pd.Timestamp(config["start_date"]) - pd.DateOffset(months=13, days=10)
    raw = _download_prices(symbols, lookback.date().isoformat(), config["end_date"].isoformat())
    missing = [ticker for ticker in symbols if ticker not in raw.columns or raw[ticker].dropna().empty]
    if missing:
        raise ValueError("가격 데이터가 없는 종목: " + ", ".join(missing))
    prices = raw.ffill()
    if market == "국내 ETF":
        prices["BIL_KRW"] = prices["BIL"] * prices["KRW=X"]
        return prices, benchmark_source, "BIL_KRW"
    if config["benchmark"] == "KOSPI":
        prices["KOSPI_USD"] = prices[benchmark_source] / prices["KRW=X"]
        return prices, "KOSPI_USD", "BIL"
    return prices, benchmark_source, "BIL"


def _normalized(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = series.reindex(index).ffill().bfill()
    if aligned.dropna().empty:
        raise ValueError("비교 대상의 공통 가격 데이터가 없습니다.")
    return (aligned / aligned.iloc[0]).rename(series.name)


def _run(config: dict[str, Any]) -> dict[str, Any]:
    prices, benchmark_ticker, bil_ticker = _prepare_prices(config)
    config["bil_ticker"] = bil_ticker
    result = run_backtest(prices, config)
    index = result["nav"].index
    benchmark_nav = _normalized(prices[benchmark_ticker].rename("벤치마크"), index)
    attack_navs = pd.concat([_normalized(prices[ticker].rename(ticker), index) for ticker in config["attack_tickers"]], axis=1)
    equal_weight_nav = attack_navs.mean(axis=1).rename("동일 비중")
    all_assets = list(dict.fromkeys(config["attack_tickers"] + config["defense_tickers"]))
    asset_nav = pd.concat([_normalized(prices[ticker].rename(ticker), index) for ticker in all_assets], axis=1)
    result.update({"benchmark_nav": benchmark_nav, "equal_weight_nav": equal_weight_nav, "asset_nav": asset_nav})
    return result


def _render_conditions() -> None:
    st.title("🔄 동적 포트폴리오 백테스트")
    st.markdown("<p class='step-caption'>시장 조건에 따라 투자자산과 방어자산의 비중을 바꾸는 전략을 검증합니다.</p>", unsafe_allow_html=True)
    if st.button("↺ 조건 초기화", key="reset_dynamic_conditions"):
        _reset_form()
        st.rerun()

    tabs = st.tabs(["기본설정", "종목선택", "투자자산 선택", "방어자산 선택", "포트폴리오 비중"])
    with tabs[0]:
        date_col, end_col, initial_col = st.columns(3)
        start_date = date_col.date_input("시작 날짜", min_value=EARLIEST_DATE, max_value=dt.date.today(), key="dynamic_start")
        end_date = end_col.date_input("종료 날짜", min_value=EARLIEST_DATE, max_value=dt.date.today(), key="dynamic_end")
        initial = initial_col.number_input("초기 투자금", min_value=1.0, step=1_000.0, key="dynamic_initial")
        type_col, rebalance_col, benchmark_col = st.columns(3)
        investment_type = type_col.selectbox("투자 방식", ["거치식", "적립식"], key="dynamic_investment_type")
        rebalance = rebalance_col.selectbox("리밸런싱 주기", ["월별", "분기", "반기", "연별"], key="dynamic_rebalance")
        benchmark = benchmark_col.selectbox("벤치마크", BENCHMARK_LABELS, key="dynamic_benchmark")
        contribution, contribution_frequency = 0.0, "매월"
        if investment_type == "적립식":
            amount_col, frequency_col = st.columns(2)
            contribution = amount_col.number_input("정기 적립금", min_value=1.0, step=100.0, key="dynamic_contribution")
            contribution_frequency = frequency_col.selectbox("적립 주기", ["매월", "매분기", "매년"], key="dynamic_contribution_frequency")
        use_cost = st.checkbox("거래비용 반영", key="dynamic_use_transaction_cost")
        cost_rate = st.number_input(
            "편도 거래비용률 (%)", min_value=0.0, max_value=10.0, step=0.01,
            key="dynamic_transaction_cost", disabled=not use_cost,
            help="매수와 매도 각각의 실제 거래금액에 적용합니다.",
        )

    with tabs[1]:
        market = st.selectbox("ETF 종류", ["해외 ETF", "국내 ETF"], key="dynamic_market")
        previous_market = st.session_state.get("_dynamic_market_seen")
        if previous_market is None:
            st.session_state["_dynamic_market_seen"] = market
        elif previous_market != market:
            st.session_state["dynamic_attack_tickers"] = []
            st.session_state["dynamic_defense_tickers"] = []
            st.session_state["_dynamic_market_seen"] = market
            st.info("ETF 종류가 변경되어 기존 후보 목록을 초기화했습니다.")
        attack_col, defense_col = st.columns(2)
        with attack_col:
            _asset_selector("attack", market)
        with defense_col:
            _asset_selector("defense", market)

    attack_tickers = list(st.session_state["dynamic_attack_tickers"])
    defense_tickers = list(st.session_state["dynamic_defense_tickers"])
    with tabs[2]:
        attack_mode = st.radio("투자자산 선택 방식", ATTACK_MODES, key="dynamic_attack_mode", help="후보군에서 투자자산을 고르고 공격자산 몫을 배분하는 규칙입니다.")
        attack_select_count = 1
        if _uses_relative(attack_mode):
            attack_count_options = list(range(1, max(1, len(attack_tickers)) + 1))
            st.session_state["dynamic_attack_select_count"] = min(
                int(st.session_state.get("dynamic_attack_select_count", 1)),
                len(attack_count_options),
            )
            attack_select_count = st.selectbox(
                "선택자산 수", attack_count_options,
                key="dynamic_attack_select_count", disabled=not attack_tickers,
            )
        attack_relative = _momentum_input("dynamic_attack_relative", "상대모멘텀") if _uses_relative(attack_mode) else MomentumConfig()
        attack_absolute = _momentum_input("dynamic_attack_absolute", "절대모멘텀") if _uses_absolute(attack_mode) else MomentumConfig()
        attack_use_bil = st.checkbox("절대모멘텀 기준을 BIL로 변경", key="dynamic_attack_use_bil", disabled=not _uses_absolute(attack_mode))

    with tabs[3]:
        defense_enabled = st.checkbox("방어자산 선택 사용", key="dynamic_defense_enabled")
        defense_mode = st.radio("방어자산 선택 방식", DEFENSE_MODES, key="dynamic_defense_mode", disabled=not defense_enabled)
        include_cash = st.checkbox("현금을 방어 후보에 포함", key="dynamic_include_cash", disabled=not defense_enabled)
        defense_select_count = 1
        if defense_mode in {"상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 현금 스위치"}:
            count = len(defense_tickers) + (1 if include_cash else 0)
            defense_count_options = list(range(1, max(1, count) + 1))
            st.session_state["dynamic_defense_select_count"] = min(
                int(st.session_state.get("dynamic_defense_select_count", 1)),
                len(defense_count_options),
            )
            defense_select_count = st.selectbox(
                "선택자산 수", defense_count_options,
                key="dynamic_defense_select_count", disabled=not defense_enabled,
            )
        same_momentum = st.checkbox("투자자산 선택기준과 동일 모멘텀 적용", key="dynamic_same_momentum", disabled=not defense_enabled or defense_mode == "전체 동일 비중")
        if same_momentum:
            defense_relative, defense_absolute, defense_use_bil = attack_relative, attack_absolute, attack_use_bil
            st.caption("투자자산 탭의 산출 기준·기간·BIL 설정을 사용합니다.")
        else:
            defense_relative = _momentum_input("dynamic_defense_relative", "방어 상대모멘텀") if _uses_relative(defense_mode) else MomentumConfig()
            defense_absolute = _momentum_input("dynamic_defense_absolute", "방어 절대모멘텀") if _uses_absolute(defense_mode) else MomentumConfig()
            defense_use_bil = st.checkbox("방어 절대모멘텀 기준을 BIL로 변경", key="dynamic_defense_use_bil", disabled=not defense_enabled or not _uses_absolute(defense_mode))

    with tabs[4]:
        use_fixed_defense = st.checkbox("방어자산 고정비중 사용", key="dynamic_fixed_defense")
        fixed_defense = st.number_input("방어자산 고정비중 (%)", min_value=0.0, max_value=100.0, step=1.0, key="dynamic_fixed_defense_weight", disabled=not use_fixed_defense)
        effective_defense = fixed_defense if use_fixed_defense else 0.0
        split = pd.DataFrame({"구분": ["공격자산 기본 몫", "방어자산 고정 몫"], "비중": [100.0 - effective_defense, effective_defense]})
        st.dataframe(split.style.format({"비중": "{:.1f}%"}), hide_index=True, use_container_width=True)
        if use_fixed_defense and not defense_enabled:
            st.info("방어자산 선택을 사용하지 않아 방어자산 고정 몫은 현금으로 보유합니다.")

    back_col, result_col = st.columns(2)
    with back_col:
        go_back = st.button("뒤로", key="back_dynamic_portfolio", use_container_width=True)
    with result_col:
        run = st.button("결과 보기", type="primary", key="run_dynamic_portfolio", use_container_width=True)
    if go_back:
        st.session_state["current_page"] = "feature"
        st.rerun()
    if not run:
        return
    if start_date >= end_date:
        st.session_state["dynamic_error"] = "시작 날짜는 종료 날짜보다 앞서야 합니다."
        st.rerun()
    if not attack_tickers:
        st.session_state["dynamic_error"] = "공격자산을 한 개 이상 등록해주세요."
        st.rerun()
    if defense_enabled and not defense_tickers and not include_cash:
        st.session_state["dynamic_error"] = "방어자산을 등록하거나 현금을 방어 후보에 포함해주세요."
        st.rerun()

    config = {
        "start_date": start_date, "end_date": end_date, "initial_investment": float(initial),
        "investment_type": investment_type, "rebalance_frequency": rebalance, "benchmark": benchmark,
        "contribution_amount": float(contribution), "contribution_frequency": contribution_frequency,
        "market": market, "currency": "KRW" if market == "국내 ETF" else "USD",
        "attack_tickers": attack_tickers, "defense_tickers": defense_tickers,
        "attack_mode": attack_mode, "attack_select_count": int(attack_select_count),
        "attack_relative": attack_relative, "attack_absolute": attack_absolute, "attack_use_bil": bool(attack_use_bil),
        "defense_enabled": bool(defense_enabled), "defense_mode": defense_mode,
        "defense_select_count": int(defense_select_count), "defense_relative": defense_relative,
        "defense_absolute": defense_absolute, "defense_use_bil": bool(defense_use_bil),
        "include_cash": bool(include_cash), "fixed_defense_weight": float(effective_defense / 100.0),
        "use_transaction_cost": bool(use_cost), "transaction_cost_rate": float(cost_rate / 100.0),
    }
    try:
        st.session_state["dynamic_saved_widget_values"] = {
            key: value
            for key, value in st.session_state.items()
            if key.startswith("dynamic_")
            and key not in {"dynamic_result", "dynamic_result_config", "dynamic_show_results", "dynamic_error"}
        }
        with st.spinner("가격 데이터와 리밸런싱 경로를 계산하고 있어요..."):
            result = _run(config)
        st.session_state["dynamic_result"] = result
        st.session_state["dynamic_result_config"] = config
        st.session_state["dynamic_show_results"] = True
        st.rerun()
    except Exception as error:
        st.session_state["dynamic_error"] = f"동적 백테스트를 계산하지 못했습니다: {error}"
        st.rerun()


def _render_result_page() -> None:
    result, config = st.session_state.get("dynamic_result"), st.session_state.get("dynamic_result_config")
    if not result or not config:
        st.session_state["dynamic_show_results"] = False
        st.rerun()
    render_results(result, config)
    back_col, save_col = st.columns(2)
    with back_col:
        if st.button("뒤로", key="back_dynamic_conditions", use_container_width=True):
            st.session_state["dynamic_show_results"] = False
            st.rerun()
    with save_col:
        st.download_button(
            "결과 저장", data=download_summary_csv(result),
            file_name="dynamic-portfolio-rebalance-history.csv", mime="text/csv",
            key="download_dynamic_result", use_container_width=True,
        )


def render(page: str) -> None:
    if not st.session_state.get("dynamic_show_results"):
        for key, value in st.session_state.get("dynamic_saved_widget_values", {}).items():
            st.session_state.setdefault(key, value)
    _initialize_state()
    _render_error_dialog()
    if page != "dynamic_portfolio_conditions":
        st.session_state["current_page"] = "dynamic_portfolio_conditions"
        st.rerun()
        return
    if st.session_state.get("dynamic_show_results"):
        _render_result_page()
    else:
        _render_conditions()
