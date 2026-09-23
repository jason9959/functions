import datetime
import html
import io
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from features.feature01_comparison import page as feature01
from features.feature02_periodic_returns import page as feature02
from features.feature03_portfolio import page as feature03
from features.feature04_monte_carlo_normal import page as feature04
from features.feature05_monte_carlo_bootstrap import page as feature05
from features.feature07_inflation import page as feature07
from features.feature08_dollar import page as feature08
from features.feature09_financial import page as feature09
from features.feature10_moving_average import page as feature10
from features.feature11_laoer import page as feature11


st.set_page_config(
    page_title="통합 포트폴리오 대시보드",
    page_icon="📌",
    layout="wide",
)


FEATURES = [
    {"id":"return_comparison","icon":"📈","title":"1. 상대 수익률 비교","description":"여러 종목을 같은 기준값 100으로 비교합니다."},
    {"id":"periodic_returns","icon":"🗓️","title":"2. 주기별 수익률 분석","description":"과거 수익률의 분포와 주기별 흐름을 확인합니다."},
    {"id":"portfolio_backtest","icon":"💼","title":"3. 포트폴리오 백테스트","description":"자산 비중과 리밸런싱 전략을 과거 데이터로 확인합니다."},
    {"id":"allocation_sweep","icon":"⚖️","title":"4. 비율별 리밸런싱 분석","description":"종목 비율을 바꿔가며 포트폴리오 성과를 비교합니다."},
    {"id":"monte_carlo_normal","icon":"🎲","title":"5. 정규분포 몬테카를로","description":"과거 수익률의 평균과 변동성으로 미래 주가를 만듭니다."},
    {"id":"monte_carlo_bootstrap","icon":"🧩","title":"6. Bootstrap 몬테카를로","description":"실제 과거 수익률을 다시 뽑아 미래 주가를 만듭니다."},
    {"id":"inflation_compass","icon":"🧭","title":"7. 인플레이션 나침반","description":"성장과 기대인플레이션 국면에 따른 자산 전략을 검증합니다."},
    {"id":"dollar_compass","icon":"💵","title":"8. 달러 환율 나침반","description":"환율·달러지수 조건과 원화/달러 스위칭 전략을 분석합니다."},
    {"id":"financial_dashboard","icon":"🏢","title":"9. 기업 재무 대시보드","description":"DART·SEC 공식 공시로 기업 재무상태·손익·현금흐름을 확인합니다."},
    {"id":"moving_average","icon":"📊","title":"10. 이동평균 투자전략 백테스트","description":"BUY/SELL 확인 횟수와 LIMIT을 포함한 이동평균 전략을 검증합니다."},
    {"id":"laoer_infinite","icon":"♾️","title":"11. 라오어 무한매수법 백테스트","description":"V2.2 분할매수·LOC·부분매도 규칙을 과거 데이터로 검증합니다."},
]


STOCK_DICT = {
    "AAPL": "Apple Inc. (애플)",
    "MSFT": "Microsoft Corporation (마이크로소프트)",
    "NVDA": "NVIDIA Corporation (엔비디아)",
    "TSLA": "Tesla Inc. (테슬라)",
    "AMZN": "Amazon.com Inc. (아마존)",
    "GOOGL": "Alphabet Inc. (구글)",
    "META": "Meta Platforms (메타)",
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "SCHD": "Schwab U.S. Dividend Equity ETF",
    "005930.KS": "삼성전자 (Samsung Electronics)",
    "000660.KS": "SK하이닉스 (SK Hynix)",
    "379800.KS": "KODEX 미국S&P500TR",
    "005380.KS": "현대차 (Hyundai Motor)",
    "035420.KS": "NAVER (네이버)",
    "035720.KS": "카카오 (카카오)",
}


for key, default in {
    "current_page": "feature",
    "comparison_result": None,
    "periodic_result": None,
    "portfolio_result": None,
    "allocation_result": None,
    "monte_normal_result": None,
    "monte_bootstrap_result": None,
    "modal_error": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


EARLIEST_ANALYSIS_DATE = datetime.date(1900, 1, 1)

# Streamlit은 다른 페이지로 이동하며 더 이상 렌더링되지 않은 위젯의 상태를
# 정리한다. 조건 입력값을 별도로 보존해 결과 화면에서 돌아왔을 때도 유지한다.
CONDITION_STATE_KEYS = {
    "comparison": [
        "comparison_start_date",
        "comparison_end_date",
        *[f"comparison_ticker_{index}" for index in range(1, 11)],
    ],
    "periodic": [
        "periodic_start",
        "periodic_end",
        "periodic_ticker",
        "periodic_frequency",
    ],
    "portfolio": [
        "portfolio_start",
        "portfolio_end",
        "portfolio_initial",
        "portfolio_invest_type",
        "portfolio_rebalance",
        "portfolio_contribution",
        "portfolio_contribution_frequency",
        *[f"portfolio_ticker_{index}" for index in range(5)],
        *[f"portfolio_weight_{index}" for index in range(5)],
    ],
    "allocation": [
        "allocation_start", "allocation_end", "allocation_count", "allocation_step",
        "allocation_ticker_1", "allocation_ticker_2", "allocation_ticker_3",
        "allocation_use_fixed", "allocation_fixed_index", "allocation_fixed_ratio",
    ],
    "normal": [
        "mc_normal_ticker",
        "mc_normal_start",
        "mc_normal_end",
        "mc_normal_horizon",
        "mc_normal_simulations",
        "mc_normal_initial",
        "mc_normal_include_contribution",
        "mc_normal_contribution",
        "mc_normal_contribution_frequency",
    ],
    "bootstrap": [
        "mc_bootstrap_ticker",
        "mc_bootstrap_start",
        "mc_bootstrap_end",
        "mc_bootstrap_horizon",
        "mc_bootstrap_simulations",
        "mc_bootstrap_initial",
        "mc_bootstrap_include_contribution",
        "mc_bootstrap_contribution",
        "mc_bootstrap_contribution_frequency",
    ],
    "f06": [
        "f06_start", "f06_end", "f06_preset", "f06_initial", "f06_threshold",
        "f06_growth", "f06_be", "f06_asset", "f06_lag", "f06_execution", "f06_cost",
    ],
    "f07": [
        "f07_calc_start", "f07_calc_end", "f07_initial_krw", "f07_fee_pct", "f07_recurring",
        "f07_monthly_krw", "f07_switch_pct", "f07_cmp_start", "f07_cmp_end", "f07_cmp_first", "f07_cmp_second",
    ],
    "f08": [
        "f08_query", "f08_frequency", "f08_start_year", "f08_end_year", "f08_start_quarter", "f08_end_quarter", "f08_company",
    ],
    "f09": [
        "f09_ticker", "f09_start_date", "f09_end_date", "f09_ma_months", "f09_confirmation_count",
        "f09_signal_limit_days", "f09_initial_amount", "f09_use_contribution", "f09_contribution_amount", "f09_contribution_frequency",
    ],
    "f10": [
        "f10_ticker", "f10_start", "f10_end", "f10_capital", "f10_splits", "f10_fee", "f10_take_profit", "f10_quarter_stop",
    ],
}

RESULT_STATE_KEYS = {
    "comparison": "comparison_result",
    "periodic": "periodic_result",
    "portfolio": "portfolio_result",
    "allocation": "allocation_result",
    "normal": "monte_normal_result",
    "bootstrap": "monte_bootstrap_result",
}


st.markdown(
    """
    <style>
    /* 작은 화면에서는 자연스럽게 줄고, 큰 화면에서는 콘텐츠가 과도하게 늘어나지 않는다. */
    .block-container,
    [data-testid="stMainBlockContainer"] {
        width: 100%;
        max-width: 1180px !important;
        margin-left: auto;
        margin-right: auto;
    }
    .feature-intro {
        margin: 0 0 18px;
        color: #191F28;
        font-size: 22px;
        font-weight: 600;
        line-height: 1.45;
    }
    div[data-testid="stButton"] > button {
        min-height: 0;
        padding: 16px 28px;
        border: 1px solid #E5E8EB;
        border-radius: 16px;
        justify-content: flex-start !important;
        text-align: left !important;
        white-space: normal;
    }
    div[data-testid="stButton"] > button > div {
        width: 100%;
        justify-content: flex-start !important;
        text-align: left !important;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #3182F6;
        background: #F5F8FF;
    }
    div[data-testid="stButton"] > button p {
        width: 100%;
        margin: 0;
        font-size: 16px;
        font-weight: 400;
        line-height: 1.55;
        text-align: left !important;
        white-space: normal;
    }
    div[data-testid="stButton"] > button p strong {
        font-size: 22px;
        font-weight: 700;
    }
    .st-key-back_to_feature button {
        background: #FFFFFF !important;
        border-color: #F58220 !important;
        color: #F58220 !important;
    }
    .st-key-back_to_feature button p {
        color: #F58220 !important;
        font-weight: 600;
        text-align: center !important;
    }
    .st-key-back_to_feature button > div,
    .st-key-show_comparison_result button > div,
    div[class*="st-key-run_"] button > div {
        justify-content: center !important;
    }
    .st-key-show_comparison_result button p,
    div[class*="st-key-run_"] button p {
        text-align: center !important;
    }
    .st-key-back_to_feature button:hover {
        background: #FFF7ED !important;
        border-color: #E66F00 !important;
    }
    .st-key-back_to_conditions button {
        background: #FFFFFF !important;
        border-color: #F58220 !important;
        color: #F58220 !important;
    }
    .st-key-back_to_conditions button p {
        color: #F58220 !important;
        font-weight: 600;
        text-align: center !important;
    }
    .st-key-back_to_conditions button > div,
    .st-key-download_comparison_report button > div {
        justify-content: center !important;
    }
    .st-key-download_comparison_report button {
        min-height: 0;
        padding: 16px 28px;
        background: #FF4B4B !important;
        border: 1px solid #FF4B4B !important;
        border-radius: 16px;
        color: #FFFFFF !important;
    }
    .st-key-download_comparison_report button p {
        color: #FFFFFF !important;
        font-size: 16px;
        font-weight: 600;
        text-align: center !important;
    }
    .st-key-download_comparison_report button:hover {
        background: #E83E3E !important;
        border-color: #E83E3E !important;
    }
    div[class*="st-key-back_"] button {
        background: #FFFFFF !important;
        border-color: #F58220 !important;
        color: #F58220 !important;
    }
    div[class*="st-key-back_"] button > div,
    div[class*="st-key-download_"] button > div {
        justify-content: center !important;
    }
    div[class*="st-key-back_"] button p {
        color: #F58220 !important;
        font-weight: 600;
        text-align: center !important;
    }
    div[class*="st-key-download_"] button {
        min-height: 0;
        padding: 16px 28px;
        background: #FF4B4B !important;
        border: 1px solid #FF4B4B !important;
        border-radius: 16px;
        color: #FFFFFF !important;
    }
    div[class*="st-key-download_"] button p {
        color: #FFFFFF !important;
        font-size: 16px;
        font-weight: 600;
        text-align: center !important;
    }
    .st-key-show_comparison_result button,
    div[class*="st-key-run_"] button {
        min-height: 0;
        padding: 16px 28px;
        background: #FF4B4B !important;
        border: 1px solid #FF4B4B !important;
        border-radius: 16px;
        color: #FFFFFF !important;
    }
    .st-key-show_comparison_result button p,
    div[class*="st-key-run_"] button p {
        color: #FFFFFF !important;
        font-size: 16px;
        font-weight: 600;
        text-align: center !important;
    }
    .st-key-show_comparison_result button:hover,
    div[class*="st-key-run_"] button:hover {
        background: #E83E3E !important;
        border-color: #E83E3E !important;
    }
    /* 메뉴 카드의 왼쪽 정렬은 유지하고, 조건 화면 CTA만 버튼 자체부터 중앙 정렬한다. */
    div[class*="st-key-show_comparison_result"] div[data-testid="stButton"] > button,
    div[class*="st-key-run_"] div[data-testid="stButton"] > button,
    div[data-testid="stButton"][class*="st-key-show_comparison_result"] > button,
    div[data-testid="stButton"][class*="st-key-run_"] > button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        background: #FF4B4B !important;
        border: 1px solid #FF4B4B !important;
        border-radius: 16px;
    }
    div[class*="st-key-show_comparison_result"] div[data-testid="stButton"] > button > div,
    div[class*="st-key-run_"] div[data-testid="stButton"] > button > div,
    div[data-testid="stButton"][class*="st-key-show_comparison_result"] > button > div,
    div[data-testid="stButton"][class*="st-key-run_"] > button > div {
        display: block !important;
        flex: 1 1 auto !important;
        width: 100% !important;
        text-align: center !important;
    }
    div[class*="st-key-show_comparison_result"] div[data-testid="stButton"] > button p,
    div[class*="st-key-run_"] div[data-testid="stButton"] > button p,
    div[data-testid="stButton"][class*="st-key-show_comparison_result"] > button p,
    div[data-testid="stButton"][class*="st-key-run_"] > button p {
        width: 100% !important;
        margin: 0 !important;
        color: #FFFFFF !important;
        font-size: 16px;
        font-weight: 600;
        text-align: center !important;
    }
    div[class*="st-key-show_comparison_result"] div[data-testid="stButton"] > button:hover,
    div[class*="st-key-run_"] div[data-testid="stButton"] > button:hover,
    div[data-testid="stButton"][class*="st-key-show_comparison_result"] > button:hover,
    div[data-testid="stButton"][class*="st-key-run_"] > button:hover {
        background: #E83E3E !important;
        border-color: #E83E3E !important;
    }
    div[data-testid="stButton"][class*="st-key-reset_"] {
        width: fit-content;
    }
    div[data-testid="stButton"][class*="st-key-reset_"] > button {
        min-height: 0;
        padding: 6px 12px;
        border-radius: 10px;
        border-color: #DDE3EA;
        background: #FFFFFF;
    }
    div[data-testid="stButton"][class*="st-key-reset_"] > button > div {
        justify-content: center !important;
    }
    div[data-testid="stButton"][class*="st-key-reset_"] > button p {
        color: #6B7684;
        font-size: 13px;
        font-weight: 600;
        text-align: center !important;
    }
    div[data-testid="stButton"][class*="st-key-reset_"] > button:hover {
        border-color: #3182F6;
        background: #F5F8FF;
    }
    .step-caption {
        color: #6B7684;
        font-size: 15px;
        margin-bottom: 24px;
    }
    .analysis-loading-overlay {
        position: fixed;
        inset: 0;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(25, 31, 40, 0.38);
        pointer-events: all;
    }
    .analysis-loading-card {
        display: flex;
        align-items: center;
        gap: 14px;
        max-width: 360px;
        padding: 22px 24px;
        border-radius: 18px;
        background: #FFFFFF;
        color: #191F28;
        box-shadow: 0 16px 44px rgba(0, 0, 0, 0.20);
        font-size: 16px;
        font-weight: 600;
    }
    .analysis-loading-spinner {
        width: 22px;
        height: 22px;
        border: 3px solid #E8F3FF;
        border-top-color: #3182F6;
        border-radius: 50%;
        animation: analysis-spin 0.8s linear infinite;
    }
    @keyframes analysis-spin { to { transform: rotate(360deg); } }
    </style>
    """,
    unsafe_allow_html=True,
)


def preserve_condition_state() -> None:
    """페이지 전환 시에도 조건 위젯의 값을 Streamlit 세션에 남긴다."""
    for keys in CONDITION_STATE_KEYS.values():
        for key in keys:
            if key in st.session_state:
                st.session_state[key] = st.session_state[key]


def reset_condition_state(feature: str) -> None:
    """선택한 기능의 조건과 이전 결과만 초기값으로 되돌린다."""
    for key in CONDITION_STATE_KEYS[feature]:
        st.session_state.pop(key, None)
    st.session_state[RESULT_STATE_KEYS[feature]] = None


def condition_default_values() -> dict[str, object]:
    """각 조건 입력의 기본값을 한 곳에서 정의한다."""
    today = datetime.date.today()
    defaults: dict[str, object] = {
        "comparison_start_date": today - datetime.timedelta(days=365),
        "comparison_end_date": today,
        "periodic_start": today - datetime.timedelta(days=365 * 10),
        "periodic_end": today,
        "periodic_ticker": "SPY",
        "periodic_frequency": "월",
        "portfolio_start": today - datetime.timedelta(days=365 * 3),
        "portfolio_end": today,
        "portfolio_initial": 10000.0,
        "portfolio_invest_type": "거치식",
        "portfolio_rebalance": "매일",
        "portfolio_contribution": 1000.0,
        "portfolio_contribution_frequency": "매월",
        "allocation_start": today - datetime.timedelta(days=365 * 3),
        "allocation_end": today,
        "allocation_count": "2개",
        "allocation_step": "10%",
        "allocation_ticker_1": "QQQ",
        "allocation_ticker_2": "IAU",
        "allocation_ticker_3": "SPY",
        "allocation_use_fixed": False,
        "allocation_fixed_index": "1번 종목",
        "allocation_fixed_ratio": 25.0,
    }
    defaults.update({f"comparison_ticker_{index}": "" for index in range(1, 11)})
    defaults.update({f"portfolio_ticker_{index}": "" for index in range(5)})
    defaults.update({f"portfolio_weight_{index}": 0.0 for index in range(5)})
    for method in ("normal", "bootstrap"):
        prefix = f"mc_{method}"
        defaults.update(
            {
                f"{prefix}_ticker": "QQQ",
                f"{prefix}_start": today - datetime.timedelta(days=365 * 5),
                f"{prefix}_end": today,
                f"{prefix}_horizon": 10.0,
                f"{prefix}_simulations": 5000,
                f"{prefix}_initial": 10000.0,
                f"{prefix}_include_contribution": False,
                f"{prefix}_contribution": 1000.0,
                f"{prefix}_contribution_frequency": "매월",
            }
        )
    return defaults


def initialize_condition_state() -> None:
    """처음 진입하거나 초기화 후 누락된 조건 값만 기본값으로 채운다."""
    for key, value in condition_default_values().items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_condition_reset_button(feature: str) -> None:
    """각 조건 입력 화면 상단에 작은 초기화 버튼을 표시한다."""
    if st.button("↺ 조건 초기화", key=f"reset_{feature}_conditions"):
        reset_condition_state(feature)
        st.rerun()


initialize_condition_state()
preserve_condition_state()


def show_error_modal(message: str) -> None:
    """다음 렌더링에서 사용자가 닫을 때까지 유지되는 오류 팝업을 예약한다."""
    st.session_state["modal_error"] = message


def render_error_modal() -> None:
    """오류를 화면 중앙의 모달로 보여주고 닫기 전에는 화면을 조작하지 못하게 한다."""
    message = st.session_state.get("modal_error")
    if not message:
        return

    @st.dialog("요청을 완료하지 못했어요", dismissible=False)
    def error_dialog() -> None:
        st.error(message)
        st.caption("내용을 확인한 뒤 닫기를 누르면 입력 화면으로 돌아갑니다.")
        if st.button("닫기", type="primary", key="close_error_dialog", use_container_width=True):
            st.session_state["modal_error"] = None
            st.rerun()

    error_dialog()


def render_loading_overlay(message: str):
    """긴 계산 동안 입력을 막는 중앙 로딩 레이어를 만들고 제거용 placeholder를 반환한다."""
    placeholder = st.empty()
    placeholder.markdown(
        "<div class='analysis-loading-overlay'>"
        "<div class='analysis-loading-card'>"
        "<span class='analysis-loading-spinner'></span>"
        f"<span>{html.escape(message)}</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    return placeholder


def resolve_ticker(user_input: str) -> str | None:
    """입력한 티커 또는 등록 종목명을 Yahoo Finance 티커로 정리한다."""
    if not user_input:
        return None

    cleaned = user_input.strip()
    if not cleaned or re.fullmatch(r"[\u3131-\u318E]+", cleaned):
        return None

    cleaned_upper = cleaned.upper()
    if cleaned_upper in STOCK_DICT:
        return cleaned_upper

    for ticker, name in STOCK_DICT.items():
        if cleaned.lower() in name.lower() or cleaned_upper in ticker:
            return ticker

    return cleaned_upper


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_adjusted_close(
    tickers: tuple[str, ...], start_iso: str, end_iso: str
) -> pd.DataFrame:
    """Yahoo Finance에서 종료일을 포함한 수정 종가를 받아온다."""
    request_end = pd.Timestamp(end_iso) + pd.Timedelta(days=1)
    raw = yf.download(
        list(tickers),
        start=pd.Timestamp(start_iso),
        end=request_end,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance에서 데이터를 가져오지 못했습니다.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise ValueError("Yahoo Finance 응답에 수정 종가 데이터가 없습니다.")
        prices = raw["Adj Close"].copy()
    else:
        if "Adj Close" not in raw.columns:
            raise ValueError("Yahoo Finance 응답에 수정 종가 데이터가 없습니다.")
        prices = raw[["Adj Close"]].copy()
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    available = [ticker for ticker in tickers if ticker in prices.columns]
    if not available:
        raise ValueError("입력한 종목의 가격 데이터가 없습니다.")

    prices = prices[available].apply(pd.to_numeric, errors="coerce")
    prices.index = pd.to_datetime(prices.index)
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    return prices.sort_index()


def prepare_common_price_data(
    tickers: list[str], start_date: datetime.date, end_date: datetime.date
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """종목별 가격이 모두 존재하는 공통 기간만 남긴다."""
    prices = download_adjusted_close(
        tuple(tickers), start_date.isoformat(), end_date.isoformat()
    )

    unavailable = [ticker for ticker in tickers if ticker not in prices.columns]
    if unavailable:
        raise ValueError("가격 데이터가 없는 종목: " + ", ".join(unavailable))

    first_dates = {}
    last_dates = {}
    for ticker in tickers:
        series = prices[ticker].dropna()
        if series.empty:
            raise ValueError(f"{ticker}의 유효한 가격 데이터가 없습니다.")
        first_dates[ticker] = series.index.min()
        last_dates[ticker] = series.index.max()

    common_start = max(first_dates.values())
    common_end = min(last_dates.values())
    if common_start >= common_end:
        raise ValueError("모든 종목의 공통 데이터 기간이 없습니다.")

    common = prices.loc[common_start:common_end, tickers].ffill()
    if common.isna().any().any():
        raise ValueError("공통 기간의 가격 데이터가 충분하지 않습니다.")
    if len(common) < 2:
        raise ValueError("비교할 수 있는 거래일이 2일 미만입니다.")
    return common, common_start, common_end


PERIOD_LABELS = {
    "월": [f"{month}월" for month in range(1, 13)],
    "분기": [f"{quarter}분기" for quarter in range(1, 5)],
    "반기": ["상반기", "하반기"],
    "년": ["연간"],
}


def calculate_periodic_returns(prices: pd.Series, frequency: str) -> pd.DataFrame:
    """조회 범위 안에서 월·분기·반기·연도별 첫/마지막 거래일 수익률을 계산한다."""
    series = prices.dropna().sort_index()
    if len(series) < 2:
        raise ValueError("수익률을 계산할 가격 데이터가 충분하지 않습니다.")

    rows = []
    grouped: dict[tuple[int, str], list[tuple[pd.Timestamp, float]]] = {}
    for date, value in series.items():
        timestamp = pd.Timestamp(date)
        if frequency == "월":
            label = f"{timestamp.month}월"
        elif frequency == "분기":
            label = f"{((timestamp.month - 1) // 3) + 1}분기"
        elif frequency == "반기":
            label = "상반기" if timestamp.month <= 6 else "하반기"
        else:
            label = "연간"
        grouped.setdefault((timestamp.year, label), []).append((timestamp, float(value)))

    for (year, label), observations in grouped.items():
        first_date, first_price = observations[0]
        last_date, last_price = observations[-1]
        period_return = (last_price / first_price - 1) * 100
        rows.append(
            {
                "연도": year,
                "주기": label,
                "시작일": first_date,
                "종료일": last_date,
                "수익률": period_return,
            }
        )
    return pd.DataFrame(rows).sort_values(["연도", "시작일"]).reset_index(drop=True)


def return_color(value: float, scale: float) -> str:
    """0에서 멀어질수록 음수는 빨강, 양수는 초록이 진해지는 셀 색을 반환한다."""
    if pd.isna(value):
        return ""
    intensity = 0.12 + 0.70 * min(abs(float(value)) / max(scale, 1.0), 1.0)
    color = "244, 67, 54" if value < 0 else "22, 163, 74"
    return f"background-color: rgba({color}, {intensity:.2f}); color: #191F28;"


def make_return_distribution(period_returns: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """0%를 정확한 경계로 포함하는 1% 간격 히스토그램 구간과 빈도를 만든다."""
    values = period_returns.dropna().to_numpy(dtype=float)
    lower = min(int(np.floor(values.min())), -1)
    upper = max(int(np.ceil(values.max())), 1)
    edges = np.arange(lower, upper + 1 + 1e-9, 1.0)
    counts, edges = np.histogram(values, bins=edges)
    return counts, edges


def calculate_periodic_summary(result: dict) -> dict[str, float]:
    """가격 추이와 주기별 수익률에서 화면·보고서용 핵심 통계를 계산한다."""
    indexed = result["indexed_prices"].dropna()
    period_values = result["period_returns"]["수익률"].dropna()
    total_return = (indexed.iloc[-1] / indexed.iloc[0] - 1) * 100
    elapsed_years = max((indexed.index[-1] - indexed.index[0]).days / 365.2425, 1 / 365.2425)
    annualized_return = ((indexed.iloc[-1] / indexed.iloc[0]) ** (1 / elapsed_years) - 1) * 100
    daily_returns = indexed.pct_change().dropna()
    annualized_volatility = daily_returns.std(ddof=1) * np.sqrt(252) * 100 if len(daily_returns) > 1 else 0.0
    drawdown = indexed / indexed.cummax() - 1
    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "max_drawdown": float(drawdown.min() * 100),
        "positive_probability": float((period_values > 0).mean() * 100),
        "nonpositive_probability": float((period_values <= 0).mean() * 100),
        "average_period_return": float(period_values.mean()),
        "median_period_return": float(period_values.median()),
        "best_period_return": float(period_values.max()),
        "worst_period_return": float(period_values.min()),
    }


def build_comparison_report_image(result: dict) -> bytes:
    """상대 수익률 비교 결과를 휴대폰에서도 보기 좋은 PNG 레포트로 만든다."""
    prices = result["prices"]
    common_start = result["common_start"]
    common_end = result["common_end"]

    figure = plt.figure(figsize=(12, 13), dpi=180, facecolor="white")
    try:
        figure.text(
            0.08,
            0.955,
            "RELATIVE RETURN REPORT",
            fontsize=22,
            fontweight="bold",
            color="#191F28",
        )
        figure.text(
            0.08,
            0.928,
            f"Analysis period: {common_start:%Y-%m-%d} to {common_end:%Y-%m-%d}",
            fontsize=10,
            color="#6B7684",
        )

        axis = figure.add_axes([0.08, 0.47, 0.84, 0.36])
        axis.axhline(100, color="#8B95A1", linestyle="--", linewidth=1, label="Base 100")
        for ticker in result["tickers"]:
            axis.plot(prices.index, prices[ticker], linewidth=2, label=ticker)
        axis.set_ylabel("Indexed value")
        axis.grid(alpha=0.2)
        axis.legend()

        summary_lines = ["FINAL PERFORMANCE"]
        for ticker in result["tickers"]:
            final_value = float(prices[ticker].iloc[-1])
            summary_lines.append(
                f"{ticker:<8} {final_value - 100:+.2f}%  (final index {final_value:.2f})"
            )
        figure.text(
            0.08,
            0.35,
            "\n".join(summary_lines),
            fontsize=12,
            color="#333D4B",
            linespacing=1.7,
        )
        figure.text(
            0.08,
            0.08,
            "For informational purposes only. This report is not investment advice.\n"
            "Data is provided by Yahoo Finance and may be delayed or inaccurate.",
            fontsize=9,
            color="#8B95A1",
        )

        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", dpi=180, facecolor="white")
        image_buffer.seek(0)
        return image_buffer.getvalue()
    finally:
        plt.close(figure)


def build_periodic_report_image(result: dict) -> bytes:
    """주기별 수익률의 가격 추이·분포·연도별 색상표를 PNG 보고서로 만든다."""
    indexed = result["indexed_prices"]
    period_returns = result["period_returns"]
    pivot = result["pivot"]
    summary = calculate_periodic_summary(result)
    report_frequency = {"월": "Monthly", "분기": "Quarterly", "반기": "Half-yearly", "년": "Yearly"}.get(
        result["frequency"], result["frequency"]
    )
    report_columns = {
        "월": [f"M{month}" for month in range(1, 13)],
        "분기": [f"Q{quarter}" for quarter in range(1, 5)],
        "반기": ["H1", "H2"],
        "년": ["Year"],
    }.get(result["frequency"], list(pivot.columns))
    counts, edges = make_return_distribution(period_returns["수익률"])
    height = max(15.0, min(30.0, 10.0 + len(pivot) * 0.34))
    figure = plt.figure(figsize=(14, height), dpi=160, facecolor="white")
    grid = figure.add_gridspec(4, 1, height_ratios=[3.2, 2.4, 0.8, max(2.2, len(pivot) * 0.22)])

    price_axis = figure.add_subplot(grid[0])
    price_axis.plot(indexed.index, indexed, color="#3182F6", linewidth=2)
    price_axis.axhline(100, color="#8B95A1", linestyle="--", linewidth=1)
    price_axis.set_title(f"{result['ticker']} indexed performance (base 100)", loc="left")
    price_axis.grid(alpha=0.2)

    histogram_axis = figure.add_subplot(grid[1])
    centers = (edges[:-1] + edges[1:]) / 2
    colors = ["#F04452" if center < 0 else "#16A34A" for center in centers]
    histogram_axis.bar(centers, counts, width=0.9, color=colors, alpha=0.78)
    histogram_axis.axvline(0, color="#191F28", linewidth=1.5)
    histogram_axis.set_title(f"{report_frequency} return distribution", loc="left")
    histogram_axis.set_xlabel("Return (%)")
    histogram_axis.set_ylabel("Frequency")
    histogram_axis.grid(axis="y", alpha=0.2)

    summary_axis = figure.add_subplot(grid[2])
    summary_axis.axis("off")
    summary_axis.text(
        0,
        0.72,
        f"Analysis period: {result['actual_start']:%Y-%m-%d} to {result['actual_end']:%Y-%m-%d}   "
        f"Total: {summary['total_return']:+.2f}%   CAGR: {summary['annualized_return']:+.2f}%   "
        f"Volatility: {summary['annualized_volatility']:.2f}%   MDD: {summary['max_drawdown']:.2f}%",
        fontsize=11,
        color="#333D4B",
    )
    summary_axis.text(
        0,
        0.22,
        f"Periods: {len(period_returns):,}   Positive probability: {summary['positive_probability']:.1f}%   "
        f"Average: {summary['average_period_return']:+.2f}%   Median: {summary['median_period_return']:+.2f}%",
        fontsize=11,
        color="#333D4B",
    )

    table_axis = figure.add_subplot(grid[3])
    values = pivot.to_numpy(dtype=float)
    scale = max(float(np.nanmax(np.abs(values))), 1.0)
    color_map = plt.get_cmap("RdYlGn").copy()
    color_map.set_bad("#F2F4F6")
    table_axis.imshow(np.ma.masked_invalid(values), cmap=color_map, vmin=-scale, vmax=scale, aspect="auto")
    table_axis.set_xticks(np.arange(len(pivot.columns)), labels=report_columns)
    table_axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    table_axis.set_title(f"Annual × {report_frequency} return table (%)", loc="left")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isnan(value):
                table_axis.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=8)
    figure.tight_layout(pad=2.2)
    return figure_to_png(figure)


def map_schedule_to_data_dates(
    data_index: pd.DatetimeIndex, first_date: pd.Timestamp, frequency: str
) -> set[pd.Timestamp]:
    """월·분기·연 단위 예정일을 실제 거래일에 맞춘다."""
    months_by_frequency = {"매월": 1, "매분기": 3, "매년": 12}
    months = months_by_frequency.get(frequency)
    if months is None:
        return set()

    mapped_dates = set()
    step = 1
    while True:
        scheduled = pd.Timestamp(first_date) + pd.DateOffset(months=months * step)
        if scheduled > data_index[-1]:
            break
        position = data_index.searchsorted(scheduled, side="left")
        if position < len(data_index):
            mapped_dates.add(data_index[position])
        step += 1
    return mapped_dates


def calculate_rebalanced_portfolio(
    prices: pd.DataFrame,
    weights: np.ndarray,
    initial_investment: float,
    rebalance_frequency: str,
    contribution_amount: float = 0.0,
    contribution_frequency: str = "매월",
) -> dict:
    """목표 비중을 주기적으로 다시 맞추는 간결한 포트폴리오 백테스트다."""
    if len(prices) < 2:
        raise ValueError("백테스트에 필요한 거래일이 충분하지 않습니다.")

    dates = pd.DatetimeIndex(prices.index)
    price_values = prices.to_numpy(dtype=float)
    shares = initial_investment * weights / price_values[0]
    portfolio_values = np.empty(len(dates), dtype=float)
    invested_capital = np.empty(len(dates), dtype=float)
    portfolio_values[0] = initial_investment
    invested_capital[0] = initial_investment

    contribution_dates = (
        map_schedule_to_data_dates(dates, dates[0], contribution_frequency)
        if contribution_amount > 0
        else set()
    )
    rebalance_dates = (
        set(dates[1:])
        if rebalance_frequency == "매일"
        else map_schedule_to_data_dates(dates, dates[0], rebalance_frequency)
    )
    events = []
    total_invested = float(initial_investment)

    for index in range(1, len(dates)):
        date = dates[index]
        current_prices = price_values[index]
        current_value = float(np.sum(shares * current_prices))
        contribution = contribution_amount if date in contribution_dates else 0.0
        total_invested += contribution

        if date in rebalance_dates:
            shares = (current_value + contribution) * weights / current_prices
            if contribution or rebalance_frequency != "매일":
                events.append({"date": date, "type": "리밸런싱", "amount": contribution})
        elif contribution:
            current_weights = shares * current_prices / current_value
            shares += contribution * current_weights / current_prices
            events.append({"date": date, "type": "적립", "amount": contribution})

        portfolio_values[index] = float(np.sum(shares * current_prices))
        invested_capital[index] = total_invested

    value_series = pd.Series(portfolio_values, index=dates, name="Portfolio value")
    invested_series = pd.Series(invested_capital, index=dates, name="Invested capital")
    drawdown = value_series / value_series.cummax() - 1
    return {
        "values": value_series,
        "invested": invested_series,
        "drawdown": drawdown,
        "events": pd.DataFrame(events),
    }


def run_monte_carlo_simulation(
    ticker: str,
    historical_prices: pd.Series,
    method: str,
    horizon_years: float,
    simulations: int,
    initial_investment: float,
    contribution_amount: float = 0.0,
    contribution_frequency: str = "매월",
) -> dict:
    """정규분포 또는 과거 수익률 Bootstrap 방식으로 미래 자산 경로를 계산한다."""
    log_returns = np.log(historical_prices / historical_prices.shift(1)).dropna()
    if len(log_returns) < 30:
        raise ValueError("최소 30개 거래일 이상의 과거 데이터가 필요합니다.")

    n_days = max(1, int(round(horizon_years * 252)))
    rng = np.random.default_rng()
    if method == "normal":
        simulated_returns = rng.normal(
            loc=float(log_returns.mean()),
            scale=float(log_returns.std(ddof=1)),
            size=(n_days, simulations),
        )
        method_name = "정규분포"
    else:
        historical_return_values = log_returns.to_numpy(dtype=float)
        sampled_indices = rng.integers(
            0, len(historical_return_values), size=(n_days, simulations)
        )
        simulated_returns = historical_return_values[sampled_indices]
        method_name = "Historical Bootstrap"

    initial_price = float(historical_prices.iloc[-1])
    price_paths = initial_price * np.exp(np.cumsum(simulated_returns, axis=0))
    price_paths = np.vstack([np.full((1, simulations), initial_price), price_paths])

    initial_shares = initial_investment / initial_price
    shares = np.full(simulations, initial_shares, dtype=float)
    contribution_steps = set()
    if contribution_amount > 0:
        days_by_frequency = {"매월": 21, "매분기": 63, "매년": 252}
        interval = days_by_frequency[contribution_frequency]
        contribution_steps = set(range(interval, n_days + 1, interval))

    value_paths = np.empty_like(price_paths)
    value_paths[0] = initial_investment
    for step in range(1, n_days + 1):
        if step in contribution_steps:
            shares += contribution_amount / price_paths[step]
        value_paths[step] = shares * price_paths[step]

    percentiles = np.percentile(value_paths, [5, 50, 95], axis=1)
    final_values = value_paths[-1]
    total_invested = initial_investment + contribution_amount * len(contribution_steps)
    return {
        "ticker": ticker,
        "method": method_name,
        "historical_start": historical_prices.index[0],
        "historical_end": historical_prices.index[-1],
        "horizon_years": horizon_years,
        "simulations": simulations,
        "initial_investment": initial_investment,
        "contribution_amount": contribution_amount,
        "contribution_frequency": contribution_frequency if contribution_amount else None,
        "contribution_count": len(contribution_steps),
        "total_invested": total_invested,
        "percentile_paths": percentiles,
        "final_values": final_values,
        "annualized_volatility": float(log_returns.std(ddof=1) * np.sqrt(252)),
    }


def figure_to_png(figure: plt.Figure) -> bytes:
    """Matplotlib 그림을 다운로드에 바로 쓸 PNG 바이트로 바꾼다."""
    try:
        image_buffer = io.BytesIO()
        figure.savefig(image_buffer, format="png", dpi=180, facecolor="white")
        image_buffer.seek(0)
        return image_buffer.getvalue()
    finally:
        plt.close(figure)


def build_portfolio_report_image(result: dict) -> bytes:
    figure = plt.figure(figsize=(12, 12), dpi=180, facecolor="white")
    values = result["values"]
    invested = result["invested"]
    axis = figure.add_axes([0.08, 0.47, 0.84, 0.34])
    axis.plot(values.index, values, label="Portfolio value", linewidth=2.5)
    axis.plot(invested.index, invested, label="Invested capital", linestyle="--")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.2)
    axis.legend()

    final_value = float(values.iloc[-1])
    final_invested = float(invested.iloc[-1])
    total_return = (final_value / final_invested - 1) * 100 if final_invested else 0
    figure.text(0.08, 0.94, "PORTFOLIO BACKTEST REPORT", fontsize=22, fontweight="bold", color="#191F28")
    figure.text(
        0.08,
        0.91,
        f"Analysis period: {values.index[0]:%Y-%m-%d} to {values.index[-1]:%Y-%m-%d}",
        fontsize=10,
        color="#6B7684",
    )
    figure.text(
        0.08,
        0.31,
        "FINAL SUMMARY\n"
        f"Final portfolio value: {final_value:,.0f}\n"
        f"Total invested capital: {final_invested:,.0f}\n"
        f"Total return: {total_return:+.2f}%",
        fontsize=12,
        color="#333D4B",
        linespacing=1.7,
    )
    figure.text(0.08, 0.08, "For informational purposes only. This is not investment advice.", fontsize=9, color="#8B95A1")
    return figure_to_png(figure)


def build_monte_carlo_report_image(result: dict) -> bytes:
    figure = plt.figure(figsize=(12, 12), dpi=180, facecolor="white")
    percentile_paths = result["percentile_paths"]
    steps = np.arange(percentile_paths.shape[1]) / 252
    axis = figure.add_axes([0.08, 0.47, 0.84, 0.34])
    axis.fill_between(steps, percentile_paths[0], percentile_paths[2], alpha=0.2, label="P5-P95")
    axis.plot(steps, percentile_paths[1], linewidth=2.5, label="Median")
    axis.set_xlabel("Years")
    axis.set_ylabel("Portfolio value")
    axis.grid(alpha=0.2)
    axis.legend()

    final_p5, final_median, final_p95 = np.percentile(result["final_values"], [5, 50, 95])
    figure.text(0.08, 0.94, "MONTE CARLO REPORT", fontsize=22, fontweight="bold", color="#191F28")
    figure.text(0.08, 0.91, f"{result['ticker']} · {result['method']} · {result['simulations']:,} simulations", fontsize=10, color="#6B7684")
    figure.text(
        0.08,
        0.31,
        "FINAL VALUE ESTIMATE\n"
        f"Total planned investment: {result['total_invested']:,.0f}\n"
        f"P5: {final_p5:,.0f}\n"
        f"Median: {final_median:,.0f}\n"
        f"P95: {final_p95:,.0f}",
        fontsize=12,
        color="#333D4B",
        linespacing=1.7,
    )
    figure.text(0.08, 0.08, "For informational purposes only. This is not investment advice.", fontsize=9, color="#8B95A1")
    return figure_to_png(figure)


def render_feature_page() -> None:
    st.title("📌 통합 포트폴리오 대시보드")
    st.markdown(
        "<p class='feature-intro'>어떤 분석을 시작할까요? 원하는 기능을 선택하세요.</p>",
        unsafe_allow_html=True,
    )

    for feature in FEATURES:
        label = (
            f"{feature['icon']}  **{feature['title']}**\u2003\u2003"
            f"{feature['description']}"
        )
        if st.button(label, key=f"choose_{feature['id']}", use_container_width=True):
            page_by_feature = {
                "return_comparison": "return_comparison_conditions",
                "periodic_returns": "periodic_conditions",
                "portfolio_backtest": "portfolio_conditions",
                "allocation_sweep": "allocation_conditions",
                "monte_carlo_normal": "monte_normal_conditions",
                "monte_carlo_bootstrap": "monte_bootstrap_conditions",
                "inflation_compass": "f06_home",
                "dollar_compass": "f07_home",
                "financial_dashboard": "f08_conditions",
                "moving_average": "f09_conditions",
                "laoer_infinite": "f10_conditions",
            }
            st.session_state["current_page"] = page_by_feature[feature["id"]]
            st.rerun()


def render_return_comparison_conditions() -> None:
    st.title("📈 상대 수익률 비교")

    st.markdown(
        "<p class='step-caption'>조건을 입력한 뒤 결과 보기를 누르면, 모든 종목을 같은 기준값 100으로 비교합니다.</p>",
        unsafe_allow_html=True,
    )
    render_condition_reset_button("comparison")
    st.subheader("조회 기간")

    today = datetime.date.today()
    start_col, end_col = st.columns(2)
    with start_col:
        start_date = st.date_input(
            "시작 날짜",
            min_value=EARLIEST_ANALYSIS_DATE,
            max_value=today,
            key="comparison_start_date",
        )
    with end_col:
        end_date = st.date_input(
            "종료 날짜",
            min_value=EARLIEST_ANALYSIS_DATE,
            max_value=today,
            key="comparison_end_date",
        )

    st.divider()
    st.subheader("비교 종목 입력")
    st.caption("최대 10개까지 입력할 수 있어요. 미국 주식 티커(예: AAPL, NVDA)를 입력하세요.")

    ticker_columns = st.columns(2)
    raw_inputs = []
    for index in range(10):
        with ticker_columns[index % 2]:
            raw_inputs.append(
                st.text_input(
                    f"종목 {index + 1}",
                    placeholder="예: AAPL",
                    key=f"comparison_ticker_{index + 1}",
                )
            )

    back_col, result_col = st.columns(2)
    with back_col:
        go_back = st.button(
            "뒤로",
            key="back_to_feature",
            use_container_width=True,
        )
    with result_col:
        show_result = st.button(
            "결과 보기",
            type="primary",
            key="show_comparison_result",
            use_container_width=True,
        )

    if go_back:
        st.session_state["current_page"] = "feature"
        st.rerun()

    if show_result:
        if start_date >= end_date:
            show_error_modal("시작 날짜는 종료 날짜보다 앞서야 합니다.")
            return

        tickers = []
        for raw_input in raw_inputs:
            ticker = resolve_ticker(raw_input)
            if ticker and ticker not in tickers:
                tickers.append(ticker)

        if not tickers:
            show_error_modal("최소 한 개의 종목 티커를 입력해주세요.")
            return

        loading_overlay = render_loading_overlay("주가 데이터를 불러와 비교 그래프를 만드는 중이에요...")
        try:
            prices, common_start, common_end = prepare_common_price_data(
                tickers, start_date, end_date
            )
            indexed_prices = prices / prices.iloc[0] * 100
            st.session_state["comparison_result"] = {
                "prices": indexed_prices,
                "tickers": tickers,
                "common_start": common_start,
                "common_end": common_end,
            }
            st.session_state["current_page"] = "return_comparison_results"
            st.rerun()
        except Exception as error:
            show_error_modal(f"데이터를 불러오지 못했습니다: {error}")
        finally:
            loading_overlay.empty()


def render_return_comparison_results() -> None:
    result = st.session_state.get("comparison_result")
    if not result:
        st.session_state["current_page"] = "return_comparison_conditions"
        st.rerun()

    st.title("📈 상대 수익률 비교 결과")

    prices = result["prices"]
    common_start = result["common_start"]
    common_end = result["common_end"]
    st.caption(
        f"공통 분석 기간: {common_start:%Y-%m-%d} ~ {common_end:%Y-%m-%d} · 시작값 100"
    )

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.axhline(100, color="#8B95A1", linestyle="--", linewidth=1, label="Base 100")
    for ticker in result["tickers"]:
        axis.plot(prices.index, prices[ticker], linewidth=2, label=ticker)
    axis.set_ylabel("Indexed value")
    axis.grid(alpha=0.2)
    axis.legend()
    st.pyplot(figure)
    plt.close(figure)

    st.subheader("최종 수익률")
    metric_columns = st.columns(min(len(result["tickers"]), 5))
    for index, ticker in enumerate(result["tickers"]):
        final_value = float(prices[ticker].iloc[-1])
        metric_columns[index % 5].metric(
            ticker,
            f"{final_value:.2f}",
            f"{final_value - 100:+.2f}%",
        )

    with st.expander("지수화된 원본 데이터 보기"):
        st.dataframe(prices, use_container_width=True)

    report_image = build_comparison_report_image(result)
    input_col, save_col = st.columns(2)
    with input_col:
        if st.button(
            "종목 입력",
            key="back_to_conditions",
            use_container_width=True,
        ):
            st.session_state["current_page"] = "return_comparison_conditions"
            st.rerun()
    with save_col:
        st.download_button(
            "결과 저장",
            data=report_image,
            file_name=(
                f"relative-return-report-{common_start:%Y%m%d}-{common_end:%Y%m%d}.png"
            ),
            mime="image/png",
            key="download_comparison_report",
            use_container_width=True,
        )


def render_periodic_return_conditions() -> None:
    st.title("🗓️ 주기별 수익률 분석")
    st.markdown(
        "<p class='step-caption'>한 종목의 과거 수익률을 월·분기·반기·연 단위로 나눠 분포와 표로 확인합니다.</p>",
        unsafe_allow_html=True,
    )
    render_condition_reset_button("periodic")

    today = datetime.date.today()
    start_col, end_col = st.columns(2)
    with start_col:
        start_date = st.date_input(
            "시작 날짜",
            min_value=EARLIEST_ANALYSIS_DATE,
            max_value=today,
            key="periodic_start",
        )
    with end_col:
        end_date = st.date_input(
            "종료 날짜",
            min_value=EARLIEST_ANALYSIS_DATE,
            max_value=today,
            key="periodic_end",
        )

    ticker_col, frequency_col = st.columns(2)
    with ticker_col:
        ticker_input = st.text_input(
            "분석 종목",
            placeholder="예: SPY 또는 AAPL",
            key="periodic_ticker",
        )
    with frequency_col:
        frequency = st.selectbox(
            "집계 주기",
            ["월", "분기", "반기", "년"],
            key="periodic_frequency",
        )

    st.caption("조회 시작·종료 구간이 주기 중간에 걸치면 해당 부분 기간도 포함해 계산합니다.")
    back_col, result_col = st.columns(2)
    with back_col:
        go_back = st.button("뒤로", key="back_to_feature_periodic", use_container_width=True)
    with result_col:
        run_analysis = st.button(
            "결과 보기",
            type="primary",
            key="run_periodic_analysis",
            use_container_width=True,
        )

    if go_back:
        st.session_state["current_page"] = "feature"
        st.rerun()
    if not run_analysis:
        return
    ticker = resolve_ticker(ticker_input)
    if not ticker:
        show_error_modal("분석할 종목 티커를 입력해주세요.")
        return
    if start_date >= end_date:
        show_error_modal("시작 날짜는 종료 날짜보다 앞서야 합니다.")
        return

    loading_overlay = render_loading_overlay("주가를 불러와 주기별 수익률을 집계하는 중이에요...")
    try:
        downloaded = download_adjusted_close((ticker,), start_date.isoformat(), end_date.isoformat())
        prices = downloaded[ticker].dropna()
        if len(prices) < 2:
            raise ValueError("선택한 기간의 가격 데이터가 충분하지 않습니다.")
        period_returns = calculate_periodic_returns(prices, frequency)
        pivot = period_returns.pivot(index="연도", columns="주기", values="수익률")
        pivot = pivot.reindex(columns=PERIOD_LABELS[frequency])
        st.session_state["periodic_result"] = {
            "ticker": ticker,
            "frequency": frequency,
            "prices": prices,
            "indexed_prices": prices / prices.iloc[0] * 100,
            "period_returns": period_returns,
            "pivot": pivot,
            "actual_start": prices.index.min(),
            "actual_end": prices.index.max(),
        }
        st.session_state["current_page"] = "periodic_results"
        st.rerun()
    except Exception as error:
        show_error_modal(f"주기별 수익률을 계산하지 못했습니다: {error}")
    finally:
        loading_overlay.empty()


def render_periodic_return_results() -> None:
    result = st.session_state.get("periodic_result")
    if not result:
        st.session_state["current_page"] = "periodic_conditions"
        st.rerun()

    period_returns = result["period_returns"]
    pivot = result["pivot"]
    counts, edges = make_return_distribution(period_returns["수익률"])
    centers = (edges[:-1] + edges[1:]) / 2
    scale = max(float(np.nanmax(np.abs(period_returns["수익률"]))), 1.0)
    summary = calculate_periodic_summary(result)

    st.title("🗓️ 주기별 수익률 분석 결과")
    st.caption(
        f"{result['ticker']} · {result['frequency']} 단위 · "
        f"실제 분석 기간: {result['actual_start']:%Y-%m-%d} ~ {result['actual_end']:%Y-%m-%d}"
    )

    st.subheader("수익률 추이")
    price_figure, price_axis = plt.subplots(figsize=(12, 5.5))
    price_axis.plot(result["indexed_prices"].index, result["indexed_prices"], color="#3182F6", linewidth=2)
    price_axis.axhline(100, color="#8B95A1", linestyle="--", linewidth=1)
    price_axis.set_ylabel("Indexed value (base 100)")
    price_axis.grid(alpha=0.2)
    st.pyplot(price_figure)
    plt.close(price_figure)

    return_col, annual_col, volatility_col, drawdown_col = st.columns(4)
    return_col.metric("전체 수익률", f"{summary['total_return']:+.2f}%")
    annual_col.metric("연환산 수익률", f"{summary['annualized_return']:+.2f}%")
    volatility_col.metric("연환산 변동성", f"{summary['annualized_volatility']:.2f}%")
    drawdown_col.metric("최대 낙폭", f"{summary['max_drawdown']:.2f}%")
    st.caption("연환산 수익률과 변동성은 실제 조회 기간 및 일별 가격을 기준으로 계산합니다.")

    st.subheader(f"{result['frequency']} 수익률 분포")
    histogram, histogram_axis = plt.subplots(figsize=(12, 4.5))
    bar_colors = ["#F04452" if center < 0 else "#16A34A" for center in centers]
    histogram_axis.bar(centers, counts, width=0.9, color=bar_colors, alpha=0.78)
    histogram_axis.axvline(0, color="#191F28", linewidth=1.5)
    histogram_axis.set_xlabel("Return (%)")
    histogram_axis.set_ylabel("Frequency")
    histogram_axis.grid(axis="y", alpha=0.2)
    st.pyplot(histogram)
    plt.close(histogram)

    positive_col, negative_col, average_col, median_col = st.columns(4)
    positive_col.metric("상승 확률", f"{summary['positive_probability']:.1f}%")
    negative_col.metric("하락·보합 확률", f"{summary['nonpositive_probability']:.1f}%")
    average_col.metric("평균 수익률", f"{summary['average_period_return']:+.2f}%")
    median_col.metric("중앙값 수익률", f"{summary['median_period_return']:+.2f}%")
    st.caption(
        f"총 {len(period_returns):,}개 구간 · 최고 {summary['best_period_return']:+.2f}% · "
        f"최저 {summary['worst_period_return']:+.2f}%"
    )

    st.subheader("수익률 구간별 빈도")
    interval_labels = [f"{edges[index]:.0f}%~{edges[index + 1]:.0f}%" for index in range(len(counts))]
    frequency_table = pd.DataFrame({"수익률 구간": interval_labels, "빈도": counts.astype(int)})
    interval_styles = pd.DataFrame("", index=frequency_table.index, columns=frequency_table.columns)
    interval_styles["수익률 구간"] = [return_color(center, scale) for center in centers]
    frequency_style = frequency_table.style.apply(lambda _: interval_styles, axis=None)
    frequency_left, frequency_center, frequency_right = st.columns([1, 1.35, 1])
    with frequency_center:
        st.dataframe(
            frequency_style,
            width="stretch",
            hide_index=True,
            height=min(900, 38 + 35 * len(frequency_table)),
        )

    st.subheader(f"연도별 {result['frequency']} 수익률")
    def style_period_table(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [[return_color(value, scale) for value in row] for row in data.to_numpy()],
            index=data.index,
            columns=data.columns,
        )

    pivot_style = pivot.style.format("{:.2f}%", na_rep="-").apply(style_period_table, axis=None)
    target_width = {12: 1050, 4: 650, 2: 500, 1: 420}.get(len(pivot.columns), 850)
    side_width = max((1180 - target_width) / 2, 1)
    pivot_left, pivot_center, pivot_right = st.columns([side_width, target_width, side_width])
    with pivot_center:
        st.dataframe(pivot_style, width="stretch")

    report_image = build_periodic_report_image(result)
    input_col, save_col = st.columns(2)
    with input_col:
        if st.button("조건 입력", key="back_to_periodic_conditions", use_container_width=True):
            st.session_state["current_page"] = "periodic_conditions"
            st.rerun()
    with save_col:
        st.download_button(
            "결과 저장",
            data=report_image,
            file_name=f"periodic-return-{result['ticker']}-{result['frequency']}.png",
            mime="image/png",
            key="download_periodic_report",
            use_container_width=True,
        )


def _allocation_combinations(tickers: list[str], step: int, fixed_index: int | None = None, fixed_ratio: int | None = None) -> list[tuple[int, ...]]:
    """합계 100%가 되는 비율 조합을 생성한다."""
    units = 100 // step
    combos = []
    if fixed_index is None:
        if len(tickers) == 2:
            combos = [(first * step, 100 - first * step) for first in range(units + 1)]
        else:
            for first in range(units + 1):
                for second in range(units - first + 1):
                    combos.append((first * step, second * step, 100 - (first + second) * step))
    else:
        remaining = 100 - fixed_ratio
        for value in range(0, remaining + 1, step):
            weights = [None] * len(tickers)
            weights[fixed_index] = fixed_ratio
            other = [index for index in range(len(tickers)) if index != fixed_index]
            weights[other[0]] = value
            weights[other[1]] = remaining - value
            combos.append(tuple(weights))
    return combos


def _allocation_metrics(calculation: dict) -> dict[str, float]:
    """리밸런싱 경로에서 조합 비교용 성과·위험 지표를 계산한다."""
    values = calculation["values"].dropna()
    daily = values.pct_change().dropna()
    total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
    years = max((values.index[-1] - values.index[0]).days / 365.2425, 1 / 365.2425)
    cagr = ((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) * 100
    volatility = daily.std(ddof=1) * np.sqrt(252) * 100 if len(daily) > 1 else 0.0
    sharpe = (daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 1 and daily.std(ddof=1) else 0.0
    drawdown = values / values.cummax() - 1
    mdd = float(drawdown.min() * 100)
    trough = drawdown.idxmin()
    peak_before = values.loc[:trough].cummax().iloc[-1]
    recovery = values.loc[trough:][values.loc[trough:] >= peak_before]
    recovery_days = float((recovery.index[0] - trough).days) if not recovery.empty else np.nan
    return {
        "총수익률": float(total_return),
        "CAGR": float(cagr),
        "변동성": float(volatility),
        "최대낙폭": mdd,
        "샤프지수": float(sharpe),
        "회복기간(일)": recovery_days,
    }


def _allocation_best_rows(rows: pd.DataFrame) -> list[tuple[str, pd.Series, str]]:
    """최종 금액을 제외하고 지표별 최고 조합을 반환한다. 낮을수록 좋은 지표는 별도 처리한다."""
    rules = [("총수익률", "max", "+.2f"), ("CAGR", "max", "+.2f"), ("변동성", "min", ".2f"), ("최대낙폭", "max", ".2f"), ("샤프지수", "max", ".3f"), ("회복기간(일)", "min", ".0f")]
    best = []
    for metric, rule, fmt in rules:
        candidates = rows[metric].dropna()
        if candidates.empty:
            continue
        index = candidates.idxmax() if rule == "max" else candidates.idxmin()
        best.append((metric, rows.loc[index], fmt))
    return best


def render_allocation_conditions() -> None:
    st.title("⚖️ 비율별 리밸런싱 분석")
    st.markdown("<p class='step-caption'>종목 비율을 바꿔가며 같은 기간의 리밸런싱 성과를 비교합니다.</p>", unsafe_allow_html=True)
    render_condition_reset_button("allocation")
    today = datetime.date.today()
    start_col, end_col = st.columns(2)
    with start_col:
        start_date = st.date_input("시작일", min_value=EARLIEST_ANALYSIS_DATE, max_value=today, key="allocation_start")
    with end_col:
        end_date = st.date_input("종료일", min_value=EARLIEST_ANALYSIS_DATE, max_value=today, key="allocation_end")

    count_col, step_col = st.columns(2)
    with count_col:
        count = st.selectbox("종목 수", ["2개", "3개"], key="allocation_count")
    with step_col:
        use_fixed = count == "3개" and st.session_state.get("allocation_use_fixed", False)
        step_label = st.selectbox("변화 비율", ["5%", "10%", "20%"], key="allocation_step", disabled=use_fixed)
    use_fixed = False
    if count == "3개":
        use_fixed = st.checkbox("고정 비율 사용", key="allocation_use_fixed")
        fixed_col, ratio_col = st.columns(2)
        with fixed_col:
            fixed_index_label = st.selectbox("고정 종목 번호", ["1번 종목", "2번 종목", "3번 종목"], key="allocation_fixed_index", disabled=not use_fixed)
        with ratio_col:
            fixed_ratio = st.number_input("고정 비율 (%)", min_value=0.0, max_value=100.0, step=5.0, key="allocation_fixed_ratio", disabled=not use_fixed)
    else:
        fixed_index_label, fixed_ratio = "1번 종목", 0.0

    tickers = []
    for index in range(1, 4 if count == "3개" else 3):
        if index == 3 and count != "3개":
            break
        tickers.append(resolve_ticker(st.text_input(f"종목 {index}", key=f"allocation_ticker_{index}") or ""))

    back_col, result_col = st.columns(2)
    with back_col:
        go_back = st.button("뒤로", key="back_to_feature_allocation", use_container_width=True)
    with result_col:
        run = st.button("결과 보기", type="primary", key="run_allocation", use_container_width=True)
    if go_back:
        st.session_state["current_page"] = "feature"
        st.rerun()
    if not run:
        return
    if start_date >= end_date:
        show_error_modal("시작일은 종료일보다 앞서야 합니다.")
        return
    if any(ticker is None for ticker in tickers) or len(set(tickers)) != len(tickers):
        show_error_modal("서로 다른 종목을 입력해주세요.")
        return
    step = 5 if use_fixed else int(step_label.rstrip("%"))
    fixed_index = ["1번 종목", "2번 종목", "3번 종목"].index(fixed_index_label) if use_fixed else None
    if use_fixed and int(fixed_ratio) % 5 != 0:
        show_error_modal("고정 비율은 5% 단위로 입력해주세요.")
        return
    combos = _allocation_combinations(tickers, step, fixed_index, int(fixed_ratio) if use_fixed else None)
    loading = render_loading_overlay(f"{len(combos)}개 비율 조합을 계산하는 중이에요...")
    try:
        prices, common_start, common_end = prepare_common_price_data(tickers, start_date, end_date)
        rows = []
        for weights in combos:
            calculation = calculate_rebalanced_portfolio(prices, np.array(weights, dtype=float) / 100, 10000.0, "매일")
            final_value = float(calculation["values"].iloc[-1])
            rows.append({**{f"{ticker} 비율": weight for ticker, weight in zip(tickers, weights)}, "최종 금액": final_value, **_allocation_metrics(calculation)})
        st.session_state["allocation_result"] = {"tickers": tickers, "rows": pd.DataFrame(rows), "common_start": common_start, "common_end": common_end, "step": step}
        st.session_state["current_page"] = "allocation_results"
        st.rerun()
    except Exception as error:
        show_error_modal(f"비율별 백테스트를 계산하지 못했습니다: {error}")
    finally:
        loading.empty()


def render_allocation_results() -> None:
    result = st.session_state.get("allocation_result")
    if not result:
        st.session_state["current_page"] = "allocation_conditions"
        st.rerun()
    st.title("⚖️ 비율별 리밸런싱 분석 결과")
    st.caption(f"공통 분석 기간: {result['common_start']:%Y-%m-%d} ~ {result['common_end']:%Y-%m-%d} · {len(result['rows'])}개 조합")
    rows = result["rows"]
    if len(result["tickers"]) == 2:
        figure, axis = plt.subplots(figsize=(12, 5.5))
        axis.plot(rows[f"{result['tickers'][0]} 비율"], rows["총수익률"], marker="o", color="#3182F6")
        axis.set_xlabel(f"{result['tickers'][0]} 비율 (%)")
        axis.set_ylabel("총수익률 (%)")
        axis.grid(alpha=0.2)
        st.pyplot(figure)
        plt.close(figure)
    else:
        st.info("3종목 결과는 비율 조합별 성과표로 확인할 수 있습니다.")
    st.subheader("지표별 최고 성과 조합")
    best_rows = _allocation_best_rows(rows)
    metric_cols = st.columns(min(3, max(1, len(best_rows))))
    for index, (metric, best, fmt) in enumerate(best_rows):
        ratio_text = " / ".join(f"{column.replace(' 비율', '')} {int(best[column])}%" for column in rows.columns if column.endswith(" 비율"))
        value = best[metric]
        suffix = "일" if metric == "회복기간(일)" else ("" if metric == "샤프지수" else "%")
        display_value = "회복 불가" if pd.isna(value) else f"{format(value, fmt)}{suffix}"
        metric_cols[index % len(metric_cols)].metric(metric, display_value, ratio_text)
    table_format = {"최종 금액": "{:,.0f}", "총수익률": "{:+.2f}%", "CAGR": "{:+.2f}%", "변동성": "{:.2f}%", "최대낙폭": "{:.2f}%", "샤프지수": "{:.3f}", "회복기간(일)": "{:.0f}"}
    st.dataframe(rows.style.format(table_format, na_rep="-"), width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        if st.button("조건 입력", key="back_to_allocation_conditions", use_container_width=True):
            st.session_state["current_page"] = "allocation_conditions"
            st.rerun()
    with right:
        st.download_button("결과 저장", rows.to_csv(index=False).encode("utf-8-sig"), "allocation-sweep.csv", "text/csv", key="download_allocation", use_container_width=True)


def render_portfolio_conditions() -> None:
    st.title("💼 포트폴리오 백테스트")
    st.markdown(
        "<p class='step-caption'>자산 비중과 투자 방식을 입력하면, 과거 기간의 포트폴리오 자산 변화를 계산합니다.</p>",
        unsafe_allow_html=True,
    )
    render_condition_reset_button("portfolio")

    today = datetime.date.today()
    date_col, end_col, investment_col = st.columns(3)
    with date_col:
        start_date = st.date_input(
            "시작 날짜",
            min_value=EARLIEST_ANALYSIS_DATE,
            max_value=today,
            key="portfolio_start",
        )
    with end_col:
        end_date = st.date_input(
            "종료 날짜",
            min_value=EARLIEST_ANALYSIS_DATE,
            max_value=today,
            key="portfolio_end",
        )
    with investment_col:
        initial_investment = st.number_input(
            "초기 투자금", min_value=1.0, step=1000.0, key="portfolio_initial"
        )

    invest_type_col, rebalance_col = st.columns(2)
    with invest_type_col:
        investment_type = st.selectbox("투자 방식", ["거치식", "적립식"], key="portfolio_invest_type")
    with rebalance_col:
        rebalance_frequency = st.selectbox(
            "리밸런싱 주기", ["매일", "매월", "매분기", "매년"], key="portfolio_rebalance"
        )

    contribution_amount = 0.0
    contribution_frequency = "매월"
    if investment_type == "적립식":
        contribution_col, frequency_col = st.columns(2)
        with contribution_col:
            contribution_amount = st.number_input(
                "정기 적립금", min_value=1.0, step=100.0, key="portfolio_contribution"
            )
        with frequency_col:
            contribution_frequency = st.selectbox(
                "적립 주기", ["매월", "매분기", "매년"], key="portfolio_contribution_frequency"
            )

    st.divider()
    st.subheader("포트폴리오 구성")
    st.caption("티커와 목표 비중을 입력하세요. 입력한 비중의 합계는 100%여야 합니다.")
    portfolio_items = []
    for index in range(5):
        ticker_col, weight_col = st.columns([3, 1])
        with ticker_col:
            raw_ticker = st.text_input(
                f"자산 {index + 1}", placeholder="예: VTI 또는 AAPL", key=f"portfolio_ticker_{index}"
            )
        with weight_col:
            weight = st.number_input(
                f"비중 {index + 1} (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                key=f"portfolio_weight_{index}",
            )
        ticker = resolve_ticker(raw_ticker)
        if ticker and weight > 0:
            portfolio_items.append({"ticker": ticker, "weight": float(weight)})

    total_weight = sum(item["weight"] for item in portfolio_items)
    if portfolio_items:
        st.caption(f"현재 입력 비중: {total_weight:.0f}%")

    back_col, result_col = st.columns(2)
    with back_col:
        go_back = st.button("뒤로", key="back_to_feature_portfolio", use_container_width=True)
    with result_col:
        run_backtest = st.button(
            "결과 보기", type="primary", key="run_portfolio_backtest", use_container_width=True
        )

    if go_back:
        st.session_state["current_page"] = "feature"
        st.rerun()
    if not run_backtest:
        return
    if start_date >= end_date:
        show_error_modal("시작 날짜는 종료 날짜보다 앞서야 합니다.")
        return
    if not portfolio_items:
        show_error_modal("티커와 비중을 입력한 자산이 최소 한 개 필요합니다.")
        return
    if len({item["ticker"] for item in portfolio_items}) != len(portfolio_items):
        show_error_modal("같은 티커는 한 번만 입력해주세요.")
        return
    if not np.isclose(total_weight, 100.0):
        show_error_modal("포트폴리오 비중의 합계가 100%여야 합니다.")
        return

    loading_overlay = render_loading_overlay("과거 주가로 포트폴리오를 계산하는 중이에요...")
    try:
        tickers = [item["ticker"] for item in portfolio_items]
        prices, common_start, common_end = prepare_common_price_data(tickers, start_date, end_date)
        calculation = calculate_rebalanced_portfolio(
            prices,
            np.array([item["weight"] / 100 for item in portfolio_items]),
            float(initial_investment),
            rebalance_frequency,
            float(contribution_amount),
            contribution_frequency,
        )
        st.session_state["portfolio_result"] = {
            **calculation,
            "portfolio": portfolio_items,
            "common_start": common_start,
            "common_end": common_end,
            "rebalance_frequency": rebalance_frequency,
            "investment_type": investment_type,
        }
        st.session_state["current_page"] = "portfolio_results"
        st.rerun()
    except Exception as error:
        show_error_modal(f"백테스트를 계산하지 못했습니다: {error}")
    finally:
        loading_overlay.empty()


def render_portfolio_results() -> None:
    result = st.session_state.get("portfolio_result")
    if not result:
        st.session_state["current_page"] = "portfolio_conditions"
        st.rerun()

    values = result["values"]
    invested = result["invested"]
    st.title("💼 포트폴리오 백테스트 결과")
    st.caption(
        f"공통 분석 기간: {result['common_start']:%Y-%m-%d} ~ {result['common_end']:%Y-%m-%d} · "
        f"리밸런싱: {result['rebalance_frequency']}"
    )

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(values.index, values, label="Portfolio value", linewidth=2.5)
    axis.plot(invested.index, invested, label="Invested capital", linestyle="--")
    axis.set_ylabel("Value")
    axis.grid(alpha=0.2)
    axis.legend()
    st.pyplot(figure)
    plt.close(figure)

    final_value = float(values.iloc[-1])
    final_invested = float(invested.iloc[-1])
    total_return = (final_value / final_invested - 1) * 100 if final_invested else 0
    metric_cols = st.columns(4)
    metric_cols[0].metric("최종 자산", f"{final_value:,.0f}")
    metric_cols[1].metric("총 투입금", f"{final_invested:,.0f}")
    metric_cols[2].metric("총 수익률", f"{total_return:+.2f}%")
    metric_cols[3].metric("최대 낙폭", f"{float(result['drawdown'].min() * 100):.2f}%")

    st.subheader("낙폭 추이")
    drawdown_figure, drawdown_axis = plt.subplots(figsize=(12, 3.8))
    drawdown_percent = result["drawdown"] * 100
    drawdown_axis.fill_between(
        drawdown_percent.index,
        drawdown_percent.to_numpy(),
        0,
        color="#F04452",
        alpha=0.22,
    )
    drawdown_axis.plot(drawdown_percent.index, drawdown_percent, color="#F04452", linewidth=1.8)
    drawdown_axis.axhline(0, color="#8B95A1", linewidth=1)
    drawdown_axis.set_ylabel("Drawdown (%)")
    drawdown_axis.grid(alpha=0.2)
    st.pyplot(drawdown_figure)
    plt.close(drawdown_figure)

    st.subheader("설정한 자산 비중")
    st.dataframe(pd.DataFrame(result["portfolio"]), use_container_width=True, hide_index=True)
    if not result["events"].empty:
        with st.expander("적립 · 리밸런싱 이벤트 보기"):
            st.dataframe(result["events"], use_container_width=True, hide_index=True)

    report_image = build_portfolio_report_image(result)
    input_col, save_col = st.columns(2)
    with input_col:
        if st.button("조건 입력", key="back_to_portfolio_conditions", use_container_width=True):
            st.session_state["current_page"] = "portfolio_conditions"
            st.rerun()
    with save_col:
        st.download_button(
            "결과 저장", data=report_image, file_name="portfolio-backtest-report.png", mime="image/png",
            key="download_portfolio_report", use_container_width=True,
        )


def render_monte_carlo_conditions(method: str) -> None:
    is_normal = method == "normal"
    title = "🎲 정규분포 몬테카를로" if is_normal else "🧩 Bootstrap 몬테카를로"
    explanation = (
        "과거 로그수익률의 평균과 변동성을 이용해 미래 경로를 만듭니다."
        if is_normal
        else "과거에 실제로 관측된 로그수익률을 다시 뽑아 미래 경로를 만듭니다."
    )
    prefix = f"mc_{method}"
    st.title(title)
    st.markdown(f"<p class='step-caption'>{explanation}</p>", unsafe_allow_html=True)
    render_condition_reset_button(method)

    ticker_input = st.text_input("시뮬레이션 종목", key=f"{prefix}_ticker")
    today = datetime.date.today()
    start_col, end_col = st.columns(2)
    with start_col:
        start_date = st.date_input(
            "과거 데이터 시작 날짜",
            min_value=EARLIEST_ANALYSIS_DATE,
            max_value=today,
            key=f"{prefix}_start",
        )
    with end_col:
        end_date = st.date_input(
            "과거 데이터 종료 날짜",
            min_value=EARLIEST_ANALYSIS_DATE,
            max_value=today,
            key=f"{prefix}_end",
        )

    horizon_col, simulations_col, investment_col = st.columns(3)
    with horizon_col:
        horizon_years = st.number_input(
            "미래 기간 (년)", min_value=0.5, max_value=30.0, step=1.0, key=f"{prefix}_horizon"
        )
    with simulations_col:
        simulations = st.number_input(
            "시뮬레이션 횟수", min_value=500, max_value=20000, step=500, key=f"{prefix}_simulations"
        )
    with investment_col:
        initial_investment = st.number_input(
            "초기 투자금", min_value=1.0, step=1000.0, key=f"{prefix}_initial"
        )

    include_contribution = st.checkbox("적립 시뮬레이션 포함", key=f"{prefix}_include_contribution")
    contribution_amount = 0.0
    contribution_frequency = "매월"
    if include_contribution:
        amount_col, frequency_col = st.columns(2)
        with amount_col:
            contribution_amount = st.number_input(
                "정기 적립금", min_value=1.0, step=100.0, key=f"{prefix}_contribution"
            )
        with frequency_col:
            contribution_frequency = st.selectbox(
                "적립 주기", ["매월", "매분기", "매년"], key=f"{prefix}_contribution_frequency"
            )

    st.caption("시뮬레이션 횟수가 많거나 기간이 길수록 계산에 시간이 더 걸립니다.")
    back_col, result_col = st.columns(2)
    with back_col:
        go_back = st.button("뒤로", key=f"back_to_feature_{method}", use_container_width=True)
    with result_col:
        run_simulation = st.button(
            "결과 보기", type="primary", key=f"run_{method}_simulation", use_container_width=True
        )

    if go_back:
        st.session_state["current_page"] = "feature"
        st.rerun()
    if not run_simulation:
        return

    ticker = resolve_ticker(ticker_input)
    if not ticker:
        show_error_modal("올바른 티커를 입력해주세요.")
        return
    if start_date >= end_date:
        show_error_modal("과거 데이터 시작 날짜는 종료 날짜보다 앞서야 합니다.")
        return

    loading_overlay = render_loading_overlay("과거 주가를 불러와 미래 경로를 계산하는 중이에요...")
    try:
        prices = download_adjusted_close((ticker,), start_date.isoformat(), end_date.isoformat())
        historical_prices = prices[ticker].dropna()
        result = run_monte_carlo_simulation(
            ticker,
            historical_prices,
            method,
            float(horizon_years),
            int(simulations),
            float(initial_investment),
            float(contribution_amount),
            contribution_frequency,
        )
        st.session_state[f"monte_{method}_result"] = result
        st.session_state["current_page"] = f"monte_{method}_results"
        st.rerun()
    except Exception as error:
        show_error_modal(f"시뮬레이션을 계산하지 못했습니다: {error}")
    finally:
        loading_overlay.empty()


def render_monte_carlo_results(method: str) -> None:
    result = st.session_state.get(f"monte_{method}_result")
    if not result:
        st.session_state["current_page"] = f"monte_{method}_conditions"
        st.rerun()

    title = "🎲 정규분포 몬테카를로 결과" if method == "normal" else "🧩 Bootstrap 몬테카를로 결과"
    st.title(title)
    st.caption(
        f"{result['ticker']} · {result['method']} · 과거 데이터: "
        f"{result['historical_start']:%Y-%m-%d} ~ {result['historical_end']:%Y-%m-%d}"
    )
    if result["contribution_amount"]:
        st.caption(
            f"적립 조건: {result['contribution_frequency']} {result['contribution_amount']:,.0f} · "
            f"예상 적립 {result['contribution_count']}회"
        )

    percentile_paths = result["percentile_paths"]
    steps = np.arange(percentile_paths.shape[1]) / 252
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.fill_between(steps, percentile_paths[0], percentile_paths[2], alpha=0.2, label="P5-P95")
    axis.plot(steps, percentile_paths[1], linewidth=2.5, label="Median")
    axis.set_xlabel("Years")
    axis.set_ylabel("Portfolio value")
    axis.grid(alpha=0.2)
    axis.legend()
    st.pyplot(figure)
    plt.close(figure)

    final_p5, final_median, final_p95 = np.percentile(result["final_values"], [5, 50, 95])
    metric_cols = st.columns(4)
    metric_cols[0].metric("총 예상 투입금", f"{result['total_invested']:,.0f}")
    metric_cols[1].metric("하위 5%", f"{final_p5:,.0f}")
    metric_cols[2].metric("중위값", f"{final_median:,.0f}")
    metric_cols[3].metric("상위 5%", f"{final_p95:,.0f}")

    histogram, histogram_axis = plt.subplots(figsize=(12, 4))
    histogram_axis.hist(result["final_values"], bins=60, alpha=0.82)
    histogram_axis.axvline(final_median, color="#191F28", linestyle="--", linewidth=2, label="Median")
    histogram_axis.set_xlabel("Final portfolio value")
    histogram_axis.set_ylabel("Simulation count")
    histogram_axis.legend()
    st.pyplot(histogram)
    plt.close(histogram)

    st.subheader("최종 예상 금액 분위")
    percentile_levels = np.arange(5, 100, 5)
    percentile_values = np.percentile(result["final_values"], percentile_levels)
    percentile_table = pd.DataFrame(
        {
            "분위": [f"{level}%" for level in percentile_levels],
            "최종 예상 금액": [f"{value:,.0f}" for value in percentile_values],
            "총 예상 투입금 대비": [
                f"{(value / result['total_invested'] - 1) * 100:+.2f}%"
                for value in percentile_values
            ],
        }
    )
    st.dataframe(percentile_table, use_container_width=True, hide_index=True)

    report_image = build_monte_carlo_report_image(result)
    input_col, save_col = st.columns(2)
    with input_col:
        if st.button("조건 입력", key=f"back_to_{method}_conditions", use_container_width=True):
            st.session_state["current_page"] = f"monte_{method}_conditions"
            st.rerun()
    with save_col:
        st.download_button(
            "결과 저장", data=report_image, file_name=f"monte-carlo-{method}-report.png", mime="image/png",
            key=f"download_{method}_report", use_container_width=True,
        )


page_slot = st.empty()
page_slot.empty()
page = st.session_state["current_page"]

# 모든 페이지를 하나의 컨테이너에서만 렌더링한다. 페이지가 바뀌면 이전 화면의
# 위젯과 버튼이 함께 제거되어, 여러 기능의 입력 화면이 섞이지 않는다.
with page_slot.container():
    if page == "feature":
        render_feature_page()
    elif page.startswith("return_comparison_"):
        feature01.render(page, render_return_comparison_conditions, render_return_comparison_results)
    elif page.startswith("periodic_"):
        feature02.render(page, render_periodic_return_conditions, render_periodic_return_results)
    elif page.startswith("portfolio_"):
        feature03.render(page, render_portfolio_conditions, render_portfolio_results)
    elif page.startswith("allocation_"):
        if page == "allocation_conditions":
            render_allocation_conditions()
        else:
            render_allocation_results()
    elif page.startswith("monte_normal_"):
        feature04.render(
            page,
            lambda: render_monte_carlo_conditions("normal"),
            lambda: render_monte_carlo_results("normal"),
        )
    elif page.startswith("monte_bootstrap_"):
        feature05.render(
            page,
            lambda: render_monte_carlo_conditions("bootstrap"),
            lambda: render_monte_carlo_results("bootstrap"),
        )
    elif page.startswith("f06_"):
        feature07.render(page)
    elif page.startswith("f07_"):
        feature08.render(page)
    elif page.startswith("f08_"):
        feature09.render(page)
    elif page.startswith("f09_"):
        feature10.render(page)
    elif page.startswith("f10_"):
        feature11.render(page)
    else:
        st.session_state["current_page"] = "feature"
        st.rerun()

render_error_modal()

st.stop()
