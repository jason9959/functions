from __future__ import annotations

import numpy as np
import pandas as pd


def median_price_path(
    prices: pd.Series,
    horizon_years: int,
    method: str = "Bootstrap",
    simulations: int = 500,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Return the simulated path nearest the median terminal value."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    steps = max(int(horizon_years * 252), 1)
    rng = np.random.default_rng(seed)
    if method == "정규분포":
        draws = rng.normal(float(log_returns.mean()), float(log_returns.std(ddof=1)), (steps, simulations))
    else:
        draws = rng.choice(log_returns.to_numpy(), size=(steps, simulations), replace=True)
    paths = np.empty((steps + 1, simulations), dtype=float)
    paths[0] = float(prices.iloc[-1])
    paths[1:] = paths[0] * np.exp(np.cumsum(draws, axis=0))
    terminal = paths[-1]
    selected = int(np.argmin(np.abs(terminal - np.median(terminal))))
    dates = pd.bdate_range(pd.Timestamp.today().normalize(), periods=steps + 1)
    return pd.Series(paths[:, selected], index=dates), pd.Series(terminal)
