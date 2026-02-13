from __future__ import annotations

import pandas as pd

from lie_engine.models import Side


def score_range(df: pd.DataFrame) -> dict[str, float | Side]:
    if len(df) < 2:
        return {"ls": 0.0, "mr": 0.0, "vc": 0.0, "side": Side.FLAT}

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    ls = 0.0
    broke_high_revert = cur["high"] > prev["roll_high10_prev"] and cur["close"] < prev["roll_high10_prev"]
    broke_low_revert = cur["low"] < prev["roll_low10_prev"] and cur["close"] > prev["roll_low10_prev"]
    if broke_high_revert or broke_low_revert:
        ls += 3.0

    body = abs(cur["close"] - cur["open"])
    wick = (cur["high"] - max(cur["close"], cur["open"])) + (min(cur["close"], cur["open"]) - cur["low"])
    if body > 0 and wick / body > 2:
        ls += 2.0

    mr = 0.0
    deviation = abs(cur["close"] - cur["ma20"])
    if deviation > 2.0 * max(cur["atr14"], 1e-9):
        mr += 2.0
    if cur["rsi14"] > 70 or cur["rsi14"] < 30:
        mr += 1.0
    near_upper = cur["close"] >= cur["roll_high20"] * 0.99
    near_lower = cur["close"] <= cur["roll_low20"] * 1.01
    if near_upper or near_lower:
        mr += 2.0

    vc = 0.0
    if (broke_high_revert or broke_low_revert) and cur["volume"] <= cur["vol_ma20"]:
        vc += 2.0
    if cur["volume"] >= 1.2 * cur["vol_ma20"]:
        vc += 2.0

    side = Side.FLAT
    if cur["rsi14"] <= 35 or near_lower:
        side = Side.LONG
    elif cur["rsi14"] >= 65 or near_upper:
        side = Side.SHORT

    return {
        "ls": float(min(5.0, ls)),
        "mr": float(min(5.0, mr)),
        "vc": float(min(4.0, vc)),
        "side": side,
    }
