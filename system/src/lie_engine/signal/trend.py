from __future__ import annotations

import numpy as np
import pandas as pd

from lie_engine.models import Side
from lie_engine.signal.features import candle_pattern_score


def score_trend(df: pd.DataFrame) -> dict[str, float | Side]:
    if df.empty:
        return {
            "position": 0.0,
            "structure": 0.0,
            "momentum": 0.0,
            "side": Side.FLAT,
        }

    cur = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else cur

    side = Side.LONG if cur["close"] >= cur["ma20"] else Side.SHORT

    position = 0.0
    ma20_slope_up = cur["ma20"] > prev["ma20"]
    if cur["close"] > cur["ma20"] and ma20_slope_up:
        position += 2.0
    if cur["close"] > cur["ma60"]:
        position += 1.0

    span = max(1e-9, cur["roll_high20"] - cur["roll_low20"])
    retrace = (cur["close"] - cur["roll_low20"]) / span
    if side == Side.LONG and 0.618 <= retrace <= 0.786:
        position += 2.0
    if side == Side.SHORT and 0.214 <= retrace <= 0.382:
        position += 2.0

    structure = 0.0
    if side == Side.LONG and cur["close"] > cur["roll_high10_prev"]:
        structure += 3.0
    if side == Side.SHORT and cur["close"] < cur["roll_low10_prev"]:
        structure += 3.0

    recent = df.tail(5)
    highs = recent["high"].to_numpy()
    lows = recent["low"].to_numpy()
    if side == Side.LONG and np.all(np.diff(highs) > -1e-9) and np.all(np.diff(lows) > -1e-9):
        structure += 2.0
    if side == Side.SHORT and np.all(np.diff(highs) < 1e-9) and np.all(np.diff(lows) < 1e-9):
        structure += 2.0

    breakout = (cur["close"] > cur["roll_high20"] * 0.995) if side == Side.LONG else (cur["close"] < cur["roll_low20"] * 1.005)
    if breakout:
        structure += 2.0

    momentum = 0.0
    momentum += candle_pattern_score(df)

    gap_up = cur["low"] > prev["high"]
    gap_down = cur["high"] < prev["low"]
    if gap_up or gap_down:
        momentum += 1.0

    if cur["volume"] >= 1.5 * max(cur["vol_ma20"], 1e-9):
        momentum += 2.0

    return {
        "position": float(min(5.0, position)),
        "structure": float(min(7.0, structure)),
        "momentum": float(min(5.0, momentum)),
        "side": side,
    }
