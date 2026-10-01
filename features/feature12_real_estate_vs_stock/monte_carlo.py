from __future__ import annotations

import numpy as np
import pandas as pd


def historical_price_path(prices: pd.Series, horizon_years: int) -> pd.Series:
    """Return the observed adjusted-price path for the requested horizon."""
    clean = prices.dropna().sort_index()
    if len(clean) < 2:
        raise ValueError("기존 주가 경로를 만들 데이터가 부족합니다.")
    observed_days = (clean.index[-1] - clean.index[0]).days
    required_days = max(int(horizon_years * 365.25) - 10, 1)
    if observed_days < required_days:
        raise ValueError(
            f"요청한 {horizon_years}년보다 실제 주가 이력이 짧습니다. "
            f"사용 가능한 기간은 약 {observed_days / 365.25:.1f}년입니다. 기간을 줄여주세요."
        )
    return clean.copy()


def median_price_path(
    prices: pd.Series,
    horizon_years: int,
    method: str = "Bootstrap",
    simulations: int = 500,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Return the simulated path nearest the median terminal value."""
    paths = simulated_price_paths(prices, horizon_years, method, simulations, seed)
    terminal = paths.iloc[-1].to_numpy(dtype=float)
    selected = int(np.argmin(np.abs(terminal - np.median(terminal))))
    return paths.iloc[:, selected].copy(), pd.Series(terminal)


def simulated_price_paths(
    prices: pd.Series,
    horizon_years: int,
    method: str = "Bootstrap",
    simulations: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Return every simulated adjusted-price path."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < 2:
        raise ValueError("몬테카를로 경로를 만들 과거 수익률 데이터가 부족합니다.")
    steps = max(int(horizon_years * 252), 1)
    rng = np.random.default_rng(seed)
    if method == "정규분포":
        draws = rng.normal(float(log_returns.mean()), float(log_returns.std(ddof=1)), (steps, simulations))
    else:
        draws = rng.choice(log_returns.to_numpy(), size=(steps, simulations), replace=True)
    paths = np.empty((steps + 1, simulations), dtype=float)
    paths[0] = float(prices.iloc[-1])
    paths[1:] = paths[0] * np.exp(np.cumsum(draws, axis=0))
    dates = pd.bdate_range(pd.Timestamp.today().normalize(), periods=steps + 1)
    return pd.DataFrame(paths, index=dates)
