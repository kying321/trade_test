from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def build_fuel_oil_2607_trade_space(
    *,
    input_packet: Mapping[str, Any],
    scenario_tree: Mapping[str, Any],
    validation_ring: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    technical = dict(input_packet.get("technical_snapshot") or {})
    price = dict(input_packet.get("price_snapshot") or {})
    current = _safe_float(price.get("last_price"), 0.0)
    atr = max(10.0, _safe_float(technical.get("atr14"), 60.0))
    support_levels = list(technical.get("support_levels") or [])
    resistance_levels = list(technical.get("resistance_levels") or [])
    support1 = _safe_float(support_levels[0] if len(support_levels) > 0 else current - atr * 0.8, current - atr * 0.8)
    support2 = _safe_float(support_levels[1] if len(support_levels) > 1 else current - atr * 1.4, current - atr * 1.4)
    resistance1 = _safe_float(resistance_levels[0] if len(resistance_levels) > 0 else current + atr * 0.8, current + atr * 0.8)
    resistance2 = _safe_float(resistance_levels[1] if len(resistance_levels) > 1 else current + atr * 1.5, current + atr * 1.5)

    scenario_ranges = {
        "base_repricing": {
            "lower": min(support1, current - atr * 0.35),
            "upper": max(resistance1, current + atr * 0.45),
        },
        "geopolitical_bull_shock": {
            "lower": max(current, resistance1 * 0.995),
            "upper": max(resistance2, resistance1) + atr * 1.2,
        },
        "macro_bear_slump": {
            "lower": min(support2, support1) - atr,
            "upper": min(current, support1 + atr * 0.25),
        },
    }
    probs = {str(row.get("scenario_id") or ""): float(row.get("path_probability") or 0.0) for row in list(scenario_tree.get("scenario_nodes") or [])}
    weighted_lower = sum(probs.get(key, 0.0) * value["lower"] for key, value in scenario_ranges.items())
    weighted_upper = sum(probs.get(key, 0.0) * value["upper"] for key, value in scenario_ranges.items())
    weighted_mid = (weighted_lower + weighted_upper) / 2.0
    primary = str(scenario_tree.get("primary_scenario") or "")
    if primary == "macro_bear_slump" or weighted_mid < current * 0.99:
        directional_bias = "down"
    elif weighted_mid > current * 1.01:
        directional_bias = "up"
    else:
        directional_bias = "range"
    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "primary_scenario": primary,
        "validation_boundary_pressure": str(validation_ring.get("boundary_pressure") or ""),
        "scenario_ranges": scenario_ranges,
        "weighted_range": {"lower": weighted_lower, "upper": weighted_upper, "mid": weighted_mid},
        "directional_bias": directional_bias,
        "upside_pct": weighted_upper / max(current, 1e-9) - 1.0 if current else 0.0,
        "downside_pct": weighted_lower / max(current, 1e-9) - 1.0 if current else 0.0,
    }
