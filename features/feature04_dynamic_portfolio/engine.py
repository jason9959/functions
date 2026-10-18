"""Calculation engine for the dynamic portfolio backtest."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CASH = "현금"
SCORE_PERIODS = {
    "1-3-6개월 단순": ((1, 3, 6), None),
    "3-6-12개월 단순": ((3, 6, 12), None),
    "1-3-6-12개월 단순": ((1, 3, 6, 12), None),
    "1-3-6-9-12개월 단순": ((1, 3, 6, 9, 12), None),
    "1-3-6-12개월 가중": ((1, 3, 6, 12), (12.0, 4.0, 2.0, 1.0)),
}
FREQUENCY_MONTHS = {"월별": 1, "분기": 3, "반기": 6, "연별": 12}
CONTRIBUTION_MONTHS = {"매월": 1, "매분기": 3, "매년": 12}


@dataclass(frozen=True)
class MomentumConfig:
    kind: str = "수익률"
    months: int = 12
    score_type: str = "1-3-6-12개월 단순"


def _asof_value(series: pd.Series, when: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    valid = series.loc[:when].dropna()
    if valid.empty:
        raise ValueError("비교 시점 이전의 가격 데이터가 없습니다.")
    return pd.Timestamp(valid.index[-1]), float(valid.iloc[-1])


def momentum_detail(
    series: pd.Series, signal_date: pd.Timestamp, config: MomentumConfig
) -> dict[str, Any]:
    """Return the score and intermediate values used for one asset."""
    series = series.dropna().sort_index()
    current_date, current = _asof_value(series, signal_date)
    detail: dict[str, Any] = {
        "판단일": current_date,
        "수정종가": current,
        "산출 기준": config.kind,
    }

    def period_return(months: int) -> tuple[float, pd.Timestamp, float]:
        base_date, base = _asof_value(series, signal_date - pd.DateOffset(months=months))
        return current / base - 1.0, base_date, base

    if config.kind == "수익률":
        value, base_date, base = period_return(config.months)
        detail.update({"기준 기간": f"{config.months}개월", "기준일": base_date, "기준가": base})
    elif config.kind == "이동평균":
        window_start = signal_date - pd.DateOffset(months=config.months)
        window = series.loc[window_start:signal_date].dropna()
        if window.empty:
            raise ValueError("이동평균을 계산할 가격 데이터가 없습니다.")
        average = float(window.mean())
        value = current / average - 1.0
        detail.update(
            {
                "기준 기간": f"{config.months}개월",
                "관측 수": int(len(window)),
                "이동평균": average,
                "P/MA": current / average,
            }
        )
    else:
        periods, weights = SCORE_PERIODS[config.score_type]
        returns = []
        for months in periods:
            period_value, _, _ = period_return(months)
            returns.append(period_value)
            detail[f"R{months}"] = period_value
        if weights is None:
            value = float(np.mean(returns))
        else:
            value = float(np.dot(np.asarray(returns), np.asarray(weights)))
            for months, weight, result in zip(periods, weights, returns):
                detail[f"{int(weight)}×R{months}"] = result * weight
        detail["기준 기간"] = config.score_type

    detail["계산값"] = float(value)
    return detail


def scheduled_trade_dates(
    index: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency: str,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Map start-anchored calendar dates to the nearest previous market date."""
    months = FREQUENCY_MONTHS[frequency]
    dates: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    step = 0
    while True:
        planned = start_date + pd.DateOffset(months=months * step)
        if planned > end_date:
            break
        candidates = index[index <= planned]
        if len(candidates):
            actual = pd.Timestamp(candidates[-1])
            if not dates or actual != dates[-1][1]:
                dates.append((planned, actual))
        step += 1
    return dates


def _ranked(
    values: dict[str, float], previous: set[str], registration_order: list[str]
) -> list[str]:
    order = {ticker: i for i, ticker in enumerate(registration_order)}
    return sorted(
        values,
        key=lambda ticker: (
            -round(values[ticker], 12),
            0 if ticker in previous else 1,
            order.get(ticker, 10_000),
        ),
    )


def _absolute_pass(
    value: float, benchmark_value: float | None
) -> bool:
    threshold = benchmark_value if benchmark_value is not None else 0.0
    return value >= threshold


_ATTACK_ABSOLUTE_MODES = {
    "절대 모멘텀",
    "절대 모멘텀 + 방어 스위치",
    "상대 & 절대 모멘텀",
    "상대 & 절대 모멘텀 + 방어 스위치",
}


def _weight_momentum_config(config: dict[str, Any], mode: str) -> tuple[MomentumConfig, str]:
    """Choose the momentum calculation used by the portfolio-weight option."""
    same_key = "negative_same_momentum" if mode == "마이너스 절대 모멘텀 수" else "positive_same_momentum"
    if not config.get(same_key, True):
        custom_key = "negative_momentum" if mode == "마이너스 절대 모멘텀 수" else "positive_momentum"
        return config[custom_key], "독립 기준"
    if config.get("attack_mode") in _ATTACK_ABSOLUTE_MODES:
        return config["attack_absolute"], "투자자산 절대 기준"
    return config["attack_relative"], "투자자산 상대 기준"


def _evaluate_candidates(
    prices: pd.DataFrame,
    candidates: list[str],
    signal_date: pd.Timestamp,
    relative_config: MomentumConfig,
    absolute_config: MomentumConfig,
    use_bil: bool,
    bil_ticker: str,
    role: str,
) -> tuple[dict[str, dict[str, Any]], float | None]:
    rows: dict[str, dict[str, Any]] = {}
    bil_score = None
    if use_bil:
        bil_score = momentum_detail(prices[bil_ticker], signal_date, absolute_config)["계산값"]
    for ticker in candidates:
        relative = momentum_detail(prices[ticker], signal_date, relative_config)
        absolute = momentum_detail(prices[ticker], signal_date, absolute_config)
        rows[ticker] = {
            "역할": role,
            "티커": ticker,
            "판단 기준 종가": relative["수정종가"],
            "상대 계산값": relative["계산값"],
            "절대 계산값": absolute["계산값"],
            "BIL 기준값": bil_score,
            "데이터 유효": True,
            "상대 상세": relative,
            "절대 상세": absolute,
        }
    return rows, bil_score


def _attack_allocation(
    mode: str,
    amount: float,
    rows: dict[str, dict[str, Any]],
    select_count: int,
    previous: set[str],
    registration_order: list[str],
    bil_score: float | None,
) -> tuple[dict[str, float], float, list[dict[str, Any]]]:
    scores = {ticker: row["상대 계산값"] for ticker, row in rows.items()}
    ranked = _ranked(scores, previous, registration_order)
    for rank, ticker in enumerate(ranked, 1):
        rows[ticker]["순위"] = rank
    uses_relative = mode in {"상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 방어 스위치"}
    uses_absolute = mode != "상대 모멘텀"
    selected = ranked[:select_count] if uses_relative else list(registration_order)
    passed = [
        ticker for ticker in selected
        if not uses_absolute or _absolute_pass(rows[ticker]["절대 계산값"], bil_score)
    ]
    switched = mode in {"절대 모멘텀 + 방어 스위치", "상대 & 절대 모멘텀 + 방어 스위치"}
    weights: dict[str, float] = {}
    defense_amount = 0.0
    audit: list[dict[str, Any]] = []
    if switched:
        base = amount / len(selected) if selected else 0.0
        for ticker in selected:
            if ticker in passed:
                weights[ticker] = base
            else:
                defense_amount += base
    elif passed:
        weights = {ticker: amount / len(passed) for ticker in passed}
    else:
        defense_amount = 0.0

    for ticker, row in rows.items():
        relative_selected = ticker in selected if uses_relative else True
        absolute_passed = _absolute_pass(row["절대 계산값"], bil_score) if uses_absolute else None
        if ticker in passed:
            reason = "선택 및 기준 통과" if uses_absolute else "상대 상위 자산 선택"
        elif not relative_selected:
            reason = "상대 순위 미달"
        elif uses_absolute and not absolute_passed:
            reason = "절대 기준 탈락 · 방어 전환" if switched else "절대 기준 탈락 · 현금 전환"
        else:
            reason = "미선택"
        audit.append(
            {
                **{key: value for key, value in row.items() if not key.endswith("상세")},
                "상대 선택": relative_selected,
                "절대 통과": absolute_passed,
                "선택 여부": ticker in passed,
                "최종 목표 비중": weights.get(ticker, 0.0),
                "결과 사유": reason,
            }
        )
    cash_amount = 0.0 if switched else max(0.0, amount - sum(weights.values()))
    return weights, defense_amount + 0.0, audit + ([{"역할": "공격", "티커": CASH, "최종 목표 비중": cash_amount, "결과 사유": "공격자산 기준 미달"}] if cash_amount else [])


def _defense_allocation(
    mode: str,
    amount: float,
    rows: dict[str, dict[str, Any]],
    select_count: int,
    include_cash: bool,
    previous: set[str],
    registration_order: list[str],
    bil_score: float | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if amount <= 1e-12:
        return {}, []
    candidates = list(registration_order)
    score_map = {ticker: row["상대 계산값"] for ticker, row in rows.items()}
    if include_cash:
        score_map[CASH] = 0.0
        candidates.append(CASH)
    if mode == "전체 동일 비중":
        chosen = candidates
        passed = chosen
        uses_absolute = False
        cash_switch = False
    else:
        ranked = _ranked(score_map, previous, candidates)
        for rank, ticker in enumerate(ranked, 1):
            if ticker in rows:
                rows[ticker]["순위"] = rank
        uses_relative = mode in {"상대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 현금 스위치"}
        uses_absolute = mode in {"절대 모멘텀", "상대 & 절대 모멘텀", "상대 & 절대 모멘텀 + 현금 스위치"}
        chosen = ranked[:select_count] if uses_relative else ranked
        passed = []
        for ticker in chosen:
            value = 0.0 if ticker == CASH else rows[ticker]["절대 계산값"]
            if not uses_absolute or _absolute_pass(value, bil_score):
                passed.append(ticker)
        cash_switch = mode == "상대 & 절대 모멘텀 + 현금 스위치"

    weights: dict[str, float] = {}
    if cash_switch:
        base = amount / len(chosen) if chosen else 0.0
        for ticker in chosen:
            target = ticker if ticker in passed else CASH
            weights[target] = weights.get(target, 0.0) + base
    elif passed:
        weights = {ticker: amount / len(passed) for ticker in passed}
    else:
        weights = {CASH: amount}

    audit = []
    for ticker in candidates:
        selected = ticker in passed
        row = rows.get(ticker, {"역할": "방어", "티커": CASH, "상대 계산값": 0.0, "절대 계산값": 0.0})
        audit.append(
            {
                **{key: value for key, value in row.items() if not key.endswith("상세")},
                "선택 여부": selected,
                "최종 목표 비중": weights.get(ticker, 0.0),
                "결과 사유": "방어자산 선택" if selected else "방어자산 기준 미달",
            }
        )
    return weights, audit


def _previous_market_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    candidates = index[index < date]
    if not len(candidates):
        raise ValueError("최초 투자 전에 모멘텀을 계산할 과거 데이터가 부족합니다.")
    return pd.Timestamp(candidates[-1])


def run_backtest(prices: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run the strategy and preserve each rebalance decision snapshot."""
    prices = prices.sort_index().astype(float).ffill()
    requested_start = pd.Timestamp(config["start_date"])
    requested_end = pd.Timestamp(config["end_date"])
    required_tickers = list(dict.fromkeys(config["attack_tickers"] + config["defense_tickers"]))
    valid_starts = [prices[ticker].dropna().index.min() for ticker in required_tickers]
    valid_ends = [prices[ticker].dropna().index.max() for ticker in required_tickers]
    start = max([requested_start, *valid_starts])
    end = min([requested_end, *valid_ends])
    evaluation_index = prices.index[(prices.index >= start) & (prices.index <= end)]
    if len(evaluation_index) < 2:
        raise ValueError("평가 기간의 공통 거래일이 부족합니다.")
    start_actual, end_actual = pd.Timestamp(evaluation_index[0]), pd.Timestamp(evaluation_index[-1])
    schedules = scheduled_trade_dates(evaluation_index, start_actual, end_actual, config["rebalance_frequency"])
    schedule_map = {actual: planned for planned, actual in schedules}
    contribution_dates = {
        actual
        for _, actual in scheduled_trade_dates(
            evaluation_index,
            start_actual,
            end_actual,
            {"매월": "월별", "매분기": "분기", "매년": "연별"}[config["contribution_frequency"]],
        )[1:]
    } if config["contribution_amount"] > 0 else set()

    tickers = list(dict.fromkeys(config["attack_tickers"] + config["defense_tickers"]))
    shares = pd.Series(0.0, index=tickers)
    cash = float(config["initial_investment"])
    total_invested = cash
    fee_rate = float(config["transaction_cost_rate"]) if config["use_transaction_cost"] else 0.0
    values, invested, flows, nav_returns = [], [], [], []
    history: list[dict[str, Any]] = []
    candidate_snapshots: dict[str, pd.DataFrame] = {}
    target_weights: dict[str, float] = {CASH: 1.0}
    previous_attack: set[str] = set()
    previous_defense: set[str] = set()
    turnover_sum = 0.0

    for date in evaluation_index:
        contribution = float(config["contribution_amount"]) if date in contribution_dates else 0.0
        if contribution:
            cash += contribution
            total_invested += contribution
        before_value = float((shares * prices.loc[date, tickers]).sum() + cash)
        if date in schedule_map:
            signal_date = _previous_market_date(prices.index, date)
            attack_rows, attack_bil = _evaluate_candidates(
                prices, config["attack_tickers"], signal_date,
                config["attack_relative"], config["attack_absolute"],
                config["attack_use_bil"], config["bil_ticker"], "공격",
            )
            weight_mode = config.get("weight_mode", "고정 범위")
            weight_summary = "고정 범위"
            if weight_mode == "마이너스 절대 모멘텀 수":
                weight_config, weight_basis = _weight_momentum_config(config, weight_mode)
                weight_values = {}
                for ticker in config["attack_tickers"]:
                    if config.get("negative_same_momentum", True):
                        field = "절대 계산값" if config.get("attack_mode") in _ATTACK_ABSOLUTE_MODES else "상대 계산값"
                        value = float(attack_rows[ticker][field])
                    else:
                        value = float(momentum_detail(prices[ticker], signal_date, weight_config)["계산값"])
                    weight_values[ticker] = value
                    attack_rows[ticker]["비중 기준값"] = value
                    attack_rows[ticker]["비중 통과"] = value >= 0.0
                negative_count = sum(value < 0.0 for value in weight_values.values())
                attack_base = 0.0 if negative_count >= int(config["negative_threshold"]) else 1.0
                weight_summary = f"{weight_basis} · 마이너스 {negative_count}개 / 기준 {int(config['negative_threshold'])}개"
            elif weight_mode == "플러스 절대 모멘텀 자산 개수 비중 한도":
                weight_config, weight_basis = _weight_momentum_config(config, weight_mode)
                weight_values = {}
                for ticker in config["attack_tickers"]:
                    if config.get("positive_same_momentum", True):
                        field = "절대 계산값" if config.get("attack_mode") in _ATTACK_ABSOLUTE_MODES else "상대 계산값"
                        value = float(attack_rows[ticker][field])
                    else:
                        value = float(momentum_detail(prices[ticker], signal_date, weight_config)["계산값"])
                    weight_values[ticker] = value
                    attack_rows[ticker]["비중 기준값"] = value
                    attack_rows[ticker]["비중 통과"] = value >= 0.0
                valid_count = len(weight_values)
                positive_count = sum(value >= 0.0 for value in weight_values.values())
                attack_base = positive_count / valid_count if valid_count else 0.0
                weight_summary = f"{weight_basis} · 플러스 {positive_count}/{valid_count}개"
            else:
                attack_base = 1.0 - float(config.get("fixed_defense_weight", 0.0))
            attack_weights, switched_amount, attack_audit = _attack_allocation(
                config["attack_mode"], attack_base, attack_rows,
                config["attack_select_count"], previous_attack, config["attack_tickers"], attack_bil,
            )
            base_defense = 1.0 - attack_base
            fixed_and_switched = base_defense + switched_amount
            defense_weights: dict[str, float] = {}
            defense_audit: list[dict[str, Any]] = []
            if fixed_and_switched > 0:
                if config["defense_enabled"]:
                    defense_rows, defense_bil = _evaluate_candidates(
                        prices, config["defense_tickers"], signal_date,
                        config["defense_relative"], config["defense_absolute"],
                        config["defense_use_bil"], config["bil_ticker"], "방어",
                    ) if config["defense_tickers"] else ({}, None)
                    defense_weights, defense_audit = _defense_allocation(
                        config["defense_mode"], fixed_and_switched, defense_rows,
                        config["defense_select_count"], config["include_cash"],
                        previous_defense, config["defense_tickers"], defense_bil,
                    )
                else:
                    defense_weights = {CASH: fixed_and_switched}
                    defense_audit = [{"역할": "방어", "티커": CASH, "선택 여부": True, "최종 목표 비중": fixed_and_switched, "결과 사유": "방어자산 기능 비활성화"}]

            new_weights: dict[str, float] = {}
            for source in (attack_weights, defense_weights):
                for ticker, weight in source.items():
                    new_weights[ticker] = new_weights.get(ticker, 0.0) + weight
            attack_cash = sum(row.get("최종 목표 비중", 0.0) for row in attack_audit if row.get("티커") == CASH)
            if attack_cash:
                new_weights[CASH] = new_weights.get(CASH, 0.0) + attack_cash
            remainder = 1.0 - sum(new_weights.values())
            if remainder > 1e-10:
                new_weights[CASH] = new_weights.get(CASH, 0.0) + remainder
            if not math.isclose(sum(new_weights.values()), 1.0, abs_tol=1e-8):
                raise ValueError("최종 목표 비중 합계가 100%가 아닙니다.")

            current_values = shares * prices.loc[date, tickers]
            current_weights = {ticker: float(value / before_value) for ticker, value in current_values.items()}
            current_weights[CASH] = cash / before_value if before_value else 0.0
            all_assets = set(current_weights) | set(new_weights)
            turnover = 0.5 * sum(abs(new_weights.get(asset, 0.0) - current_weights.get(asset, 0.0)) for asset in all_assets)
            if date != start_actual:
                turnover_sum += turnover

            gross_trade = sum(abs(before_value * new_weights.get(ticker, 0.0) - current_values.get(ticker, 0.0)) for ticker in tickers)
            fee = gross_trade * fee_rate
            investable = max(0.0, before_value - fee)
            for ticker in tickers:
                shares[ticker] = investable * new_weights.get(ticker, 0.0) / float(prices.at[date, ticker])
            cash = investable * new_weights.get(CASH, 0.0)
            target_weights = new_weights
            previous_attack = {ticker for ticker in attack_weights if ticker != CASH}
            previous_defense = {ticker for ticker in defense_weights if ticker != CASH}
            after_value = float((shares * prices.loc[date, tickers]).sum() + cash)
            rows = attack_audit + defense_audit
            snapshot_key = date.strftime("%Y-%m-%d")
            candidate_snapshots[snapshot_key] = pd.DataFrame(rows)
            history.append(
                {
                    "예정일": schedule_map[date].date(),
                    "적용일": date.date(),
                    "판단 기준일": signal_date.date(),
                    "선택된 종목": ", ".join(f"{ticker} {weight:.1%}" for ticker, weight in new_weights.items() if weight > 0),
                    "공격자산 비중": sum(attack_weights.values()),
                    "방어자산 비중": sum(value for ticker, value in defense_weights.items() if ticker != CASH),
                    "현금 비중": new_weights.get(CASH, 0.0),
                    "포트폴리오 비중 방식": weight_mode,
                    "비중 판정": weight_summary,
                    "턴오버": 0.0 if date == start_actual else turnover,
                    "거래비용": fee,
                    "평가금액": after_value,
                }
            )
            before_value = after_value
        elif contribution:
            # Use new cash to move toward the current target without selling.
            for ticker, weight in target_weights.items():
                if ticker == CASH or weight <= 0:
                    continue
                amount = contribution * weight
                fee = amount * fee_rate
                shares[ticker] += max(0.0, amount - fee) / float(prices.at[date, ticker])
                cash -= amount
            before_value = float((shares * prices.loc[date, tickers]).sum() + cash)

        period_return = (
            before_value / float(config["initial_investment"]) - 1.0
            if not values
            else (before_value - contribution) / values[-1] - 1.0
        )
        values.append(before_value)
        invested.append(total_invested)
        flows.append(contribution)
        nav_returns.append(period_return)

    value_series = pd.Series(values, index=evaluation_index, name="포트폴리오 평가액")
    invested_series = pd.Series(invested, index=evaluation_index, name="누적 투입금")
    flow_series = pd.Series(flows, index=evaluation_index, name="외부 현금흐름")
    nav = (1.0 + pd.Series(nav_returns, index=evaluation_index).fillna(0.0)).cumprod()
    nav.iloc[0] = 1.0
    drawdown = nav / nav.cummax() - 1.0
    years = max((evaluation_index[-1] - evaluation_index[0]).days / 365.25, 1 / 365.25)
    return {
        "values": value_series,
        "invested": invested_series,
        "flows": flow_series,
        "nav": nav.rename("포트폴리오"),
        "drawdown": drawdown.rename("포트폴리오"),
        "history": pd.DataFrame(history),
        "candidate_snapshots": candidate_snapshots,
        "final_weights": target_weights,
        "annual_turnover": turnover_sum / years,
        "common_start": start_actual,
        "common_end": end_actual,
    }
