from __future__ import annotations


def mortgage_payment(principal: float, annual_rate: float, years: int, method: str) -> float:
    """월별 대출 상환액. 만기일시상환은 월 이자만 반환한다."""
    months = max(int(years * 12), 1)
    rate = annual_rate / 100 / 12
    if method == "만기일시상환":
        return principal * rate
    if method == "원금균등상환":
        return principal / months + principal * rate
    if rate == 0:
        return principal / months
    return principal * rate * (1 + rate) ** months / ((1 + rate) ** months - 1)
