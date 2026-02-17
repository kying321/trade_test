from __future__ import annotations

from lie_engine.models import Side


def time_stop_days(signal_horizon: str) -> tuple[int, int]:
    if signal_horizon == "short":
        return (3, 5)
    return (8, 10)


def trailing_stop_price(side: Side, entry: float, current: float, atr: float, realized_multiple: float) -> float:
    if side == Side.LONG:
        if realized_multiple > 5:
            return max(entry + 0.6 * (current - entry), current - 5 * atr)
        if realized_multiple > 3:
            return entry + 0.5 * (current - entry)
        if realized_multiple > 2:
            return entry + atr
        if realized_multiple > 1:
            return entry
        return entry - 1.5 * atr

    if realized_multiple > 5:
        return min(entry - 0.6 * (entry - current), current + 5 * atr)
    if realized_multiple > 3:
        return entry - 0.5 * (entry - current)
    if realized_multiple > 2:
        return entry - atr
    if realized_multiple > 1:
        return entry
    return entry + 1.5 * atr
