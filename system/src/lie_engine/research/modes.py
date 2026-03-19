from __future__ import annotations

from dataclasses import dataclass, replace


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


DIRECT_MARKET_ASSET_CLASSES = {"crypto", "future", "spot", "perp", "perpetual"}


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


def adapt_mode_space(
    space: ModeSpace,
    *,
    asset_classes: set[str] | None = None,
    info_available: bool = True,
) -> ModeSpace:
    classes = {str(x).strip().lower() for x in (asset_classes or set()) if str(x).strip()}
    direct_market_mode = bool(classes) and classes.issubset(DIRECT_MARKET_ASSET_CLASSES)
    if not direct_market_mode and bool(info_available):
        return space

    confidence_floor = float(space.confidence_min)
    convexity_floor = float(space.convexity_min)
    min_total_trades = int(space.min_total_trades)
    trades_max = int(space.trades_max)

    if direct_market_mode:
        confidence_floor -= 8.0
        convexity_floor -= 0.15
        min_total_trades = max(4, int(round(min_total_trades * 0.80)))
        trades_max = min(6, max(trades_max, space.trades_max + 1))

    if not bool(info_available):
        confidence_floor -= 10.0
        convexity_floor -= 0.20
        min_total_trades = max(3, int(round(min_total_trades * 0.75)))

    if direct_market_mode and not bool(info_available) and space.name == "ultra_short":
        confidence_floor = min(confidence_floor, 14.0)
    elif direct_market_mode and not bool(info_available):
        confidence_floor = min(confidence_floor, 18.0)

    return replace(
        space,
        confidence_min=max(8.0, confidence_floor),
        convexity_min=max(0.8, convexity_floor),
        trades_max=trades_max,
        min_total_trades=min_total_trades,
    )
