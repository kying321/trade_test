from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModeSpace:
    name: str
    hold_days_min: int
    hold_days_max: int
    trades_min: int
    trades_max: int
    confidence_min: float
    confidence_max: float
    convexity_min: float
    convexity_max: float
    min_total_trades: int


MODE_SPACES: dict[str, ModeSpace] = {
    "ultra_short": ModeSpace(
        name="ultra_short",
        hold_days_min=1,
        hold_days_max=3,
        trades_min=2,
        trades_max=5,
        confidence_min=35.0,
        confidence_max=70.0,
        convexity_min=1.2,
        convexity_max=3.2,
        min_total_trades=80,
    ),
    "swing": ModeSpace(
        name="swing",
        hold_days_min=4,
        hold_days_max=12,
        trades_min=1,
        trades_max=3,
        confidence_min=32.0,
        confidence_max=62.0,
        convexity_min=1.2,
        convexity_max=3.0,
        min_total_trades=20,
    ),
    "long": ModeSpace(
        name="long",
        hold_days_min=13,
        hold_days_max=35,
        trades_min=1,
        trades_max=1,
        confidence_min=30.0,
        confidence_max=58.0,
        convexity_min=1.0,
        convexity_max=2.4,
        min_total_trades=8,
    ),
}
