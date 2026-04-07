from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_fuel_oil_2607_strategy_matrix(
    *,
    input_packet: Mapping[str, Any],
    scenario_tree: Mapping[str, Any],
    validation_ring: Mapping[str, Any],
    trade_space: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    technical = dict(input_packet.get("technical_snapshot") or {})
    fundamental = dict(input_packet.get("fundamental_snapshot") or {})
    price = dict(input_packet.get("price_snapshot") or {})
    coverage = dict(input_packet.get("coverage") or {})
    primary = str(scenario_tree.get("primary_scenario") or "")
    weighted = dict(trade_space.get("weighted_range") or {})
    current = _safe_float(price.get("last_price"))
    atr = max(10.0, _safe_float(technical.get("atr14"), 60.0))
    support = list(technical.get("support_levels") or [current - atr])
    resistance = list(technical.get("resistance_levels") or [current + atr])
    coverage_ratio = _safe_float(coverage.get("coverage_ratio"), 0.0)
    risk_budget = 0.30 if coverage_ratio >= 0.80 else 0.20 if coverage_ratio >= 0.65 else 0.12

    if primary == "macro_bear_slump" or trade_space.get("directional_bias") == "down":
        preferred_bias = "short"
        strategies = [
            {
                "strategy_name": "FU2607 单边趋势空",
                "direction": "short",
                "entry_zone": [support[0], current],
                "take_profit": weighted.get("lower"),
                "stop_loss": resistance[0] + atr * 0.35,
                "position_budget": risk_budget,
            },
            {
                "strategy_name": "燃油-原油比值回落空头",
                "direction": "relative_short",
                "entry_zone": [float(fundamental.get("relative_strength_score") or 0.0), 1.0],
                "take_profit": weighted.get("mid"),
                "stop_loss": resistance[0] + atr * 0.50,
                "position_budget": risk_budget * 0.7,
            },
            {
                "strategy_name": "FU2607/FU2609 反套观察",
                "direction": "relative_short",
                "entry_zone": [float(technical.get("calendar_spread") or 0.0), float(technical.get("calendar_spread") or 0.0) + 40.0],
                "take_profit": 0.0,
                "stop_loss": float(technical.get("calendar_spread") or 0.0) + 60.0,
                "position_budget": risk_budget * 0.5,
            },
        ]
    else:
        preferred_bias = "long"
        strategies = [
            {
                "strategy_name": "燃油-原油裂解价差多头",
                "direction": "relative_long",
                "entry_zone": [current, resistance[0]],
                "take_profit": weighted.get("upper"),
                "stop_loss": support[0] - atr * 0.35,
                "position_budget": risk_budget,
            },
            {
                "strategy_name": "FU2607 顺势回踩做多",
                "direction": "long",
                "entry_zone": [support[0], current],
                "take_profit": weighted.get("upper"),
                "stop_loss": support[-1] - atr * 0.50,
                "position_budget": risk_budget * 0.7,
            },
            {
                "strategy_name": "FU2607/FU2609 正套观察",
                "direction": "relative_long",
                "entry_zone": [0.0, float(technical.get("calendar_spread") or 0.0)],
                "take_profit": float(technical.get("calendar_spread") or 0.0) + 40.0,
                "stop_loss": -20.0,
                "position_budget": risk_budget * 0.5,
            },
        ]

    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "primary_scenario": primary,
        "boundary_pressure": str(validation_ring.get("boundary_pressure") or ""),
        "preferred_bias": preferred_bias,
        "risk_budget": risk_budget,
        "priority_strategies": strategies,
    }
