from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def compute_hurst_rs(returns: np.ndarray, scales: Iterable[int] = (10, 20, 30, 40, 60)) -> float:
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return 0.5

    log_n: list[float] = []
    log_rs: list[float] = []

    for n in scales:
        if n < 5 or x.size < n:
            continue
        m = x.size // n
        if m < 1:
            continue
        rs_values = []
        for i in range(m):
            seg = x[i * n : (i + 1) * n]
            mu = seg.mean()
            y = seg - mu
            z = np.cumsum(y)
            r = z.max() - z.min()
            s = seg.std(ddof=0)
            if s > 1e-10 and r > 0:
                rs_values.append(r / s)
        if not rs_values:
            continue
        log_n.append(math.log(float(n)))
        log_rs.append(math.log(float(np.mean(rs_values))))

    if len(log_n) < 2:
        return 0.5

    slope, _ = np.polyfit(np.array(log_n), np.array(log_rs), deg=1)
    return float(np.clip(slope, 0.0, 1.0))


def latest_multi_scale_hurst(prices: np.ndarray, windows: Iterable[int] = (60, 90, 120)) -> float:
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < 30:
        return 0.5

    vals = []
    log_returns = np.diff(np.log(np.maximum(p, 1e-8)))
    for w in windows:
        if log_returns.size < w:
            continue
        vals.append(compute_hurst_rs(log_returns[-w:]))
    if not vals:
        return compute_hurst_rs(log_returns)
    return float(np.mean(vals))
