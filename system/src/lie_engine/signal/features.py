from __future__ import annotations

import numpy as np
import pandas as pd

from lie_engine.regime.atr import compute_atr


def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("ts").copy()
    out["ma20"] = out["close"].rolling(20, min_periods=5).mean()
    out["ma60"] = out["close"].rolling(60, min_periods=10).mean()
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=5).mean()
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["atr14"] = compute_atr(out, period=14).bfill().ffill()
    out["rsi14"] = _compute_rsi(out["close"], period=14)
    out["roll_high20"] = out["high"].rolling(20, min_periods=5).max()
    out["roll_low20"] = out["low"].rolling(20, min_periods=5).min()
    out["roll_high10_prev"] = out["high"].shift(1).rolling(10, min_periods=5).max()
    out["roll_low10_prev"] = out["low"].shift(1).rolling(10, min_periods=5).min()
    return out


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period, min_periods=max(3, period // 2)).mean()
    down = -delta.clip(upper=0).rolling(period, min_periods=max(3, period // 2)).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def candle_pattern_score(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    prev = df.iloc[-2]
    cur = df.iloc[-1]
    score = 0.0

    engulf_bull = cur["close"] > cur["open"] and prev["close"] < prev["open"] and cur["close"] > prev["open"] and cur["open"] < prev["close"]
    engulf_bear = cur["close"] < cur["open"] and prev["close"] > prev["open"] and cur["open"] > prev["close"] and cur["close"] < prev["open"]
    if engulf_bull or engulf_bear:
        score += 1.5

    body = abs(cur["close"] - cur["open"])
    upper_wick = cur["high"] - max(cur["close"], cur["open"])
    lower_wick = min(cur["close"], cur["open"]) - cur["low"]
    if body > 0 and (upper_wick / body > 2.0 or lower_wick / body > 2.0):
        score += 0.5
    return min(2.0, score)
