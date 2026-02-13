from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kelly_fraction(win_rate: float, payoff: float) -> float:
    p = float(np.clip(win_rate, 0.0, 1.0))
    b = max(float(payoff), 1e-9)
    q = 1.0 - p
    f = (p * b - q) / b
    return float(np.clip(f, 0.0, 1.0))


def infer_edge_from_trades(trades: pd.DataFrame) -> tuple[float, float]:
    if trades.empty or "pnl" not in trades.columns:
        return 0.45, 2.0

    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float((pnl > 0).mean()) if len(pnl) else 0.45
    if losses.empty:
        payoff = 2.0
    else:
        payoff = float(wins.mean() / abs(losses.mean())) if not wins.empty else 0.8
    payoff = float(np.clip(payoff, 0.2, 10.0))
    return win_rate, payoff
