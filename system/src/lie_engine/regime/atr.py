from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=max(2, period // 2)).mean()


def compute_atr_zscore(df: pd.DataFrame, atr_period: int = 14, z_window: int = 60) -> float:
    if df.empty:
        return 0.0
    atr = compute_atr(df, period=atr_period)
    mu = atr.rolling(z_window, min_periods=max(5, z_window // 3)).mean()
    sigma = atr.rolling(z_window, min_periods=max(5, z_window // 3)).std(ddof=0)
    latest_atr = float(atr.iloc[-1])
    latest_mu = float(mu.iloc[-1]) if np.isfinite(mu.iloc[-1]) else latest_atr
    latest_sigma = float(sigma.iloc[-1]) if np.isfinite(sigma.iloc[-1]) else 1e-6
    if latest_sigma <= 1e-9:
        return 0.0
    return float((latest_atr - latest_mu) / latest_sigma)
