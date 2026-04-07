from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np


SCENARIO_LABELS = {
    "base_repricing": "基准重定价",
    "geopolitical_bull_shock": "地缘冲击多头",
    "macro_bear_slump": "宏观衰退空头",
}


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _signal_score(signal: str, *, positive: str, negative: str) -> float:
    if signal == positive:
        return 1.0
    if signal == negative:
        return 0.0
    return 0.5


def build_fuel_oil_2607_scenario_tree(
    *,
    input_packet: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    fundamental = dict(input_packet.get("fundamental_snapshot") or {})
    technical = dict(input_packet.get("technical_snapshot") or {})
    participants = dict(input_packet.get("participant_snapshot") or {})
    price = dict(input_packet.get("price_snapshot") or {})

    supportive = np.mean(
        [
            _signal_score(str(fundamental.get("inventory_signal") or ""), positive="tightening", negative="loosening"),
            _signal_score(str(fundamental.get("freight_signal") or ""), positive="firm", negative="soft"),
            _signal_score(str(fundamental.get("demand_signal") or ""), positive="firm", negative="soft"),
            1.0 if technical.get("daily_trend") == "up" else 0.0 if technical.get("daily_trend") == "down" else 0.5,
            1.0 if technical.get("weekly_trend") == "up" else 0.0 if technical.get("weekly_trend") == "down" else 0.5,
            1.0 if participants.get("long_short_bias") == "net_long" else 0.0 if participants.get("long_short_bias") == "net_short" else 0.5,
            _clamp01(float(fundamental.get("relative_strength_score") or 0.5)),
        ]
    )
    bearish = np.mean(
        [
            _signal_score(str(fundamental.get("inventory_signal") or ""), positive="loosening", negative="tightening"),
            _signal_score(str(fundamental.get("freight_signal") or ""), positive="soft", negative="firm"),
            _signal_score(str(fundamental.get("demand_signal") or ""), positive="soft", negative="firm"),
            1.0 if technical.get("daily_trend") == "down" else 0.0 if technical.get("daily_trend") == "up" else 0.5,
            1.0 if technical.get("weekly_trend") == "down" else 0.0 if technical.get("weekly_trend") == "up" else 0.5,
            1.0 if participants.get("long_short_bias") == "net_short" else 0.0 if participants.get("long_short_bias") == "net_long" else 0.5,
            _clamp01((0.08 - float(fundamental.get("relative_strength_score") or 0.5) * 0.16 + 0.08) / 0.16),
        ]
    )
    contract_pct = float(price.get("contract_pct_change_1d") or 0.0)
    benchmark_pct = float(price.get("benchmark_pct_change_1d") or 0.0)
    logistics_shock = np.mean(
        [
            1.0 if fundamental.get("freight_signal") == "firm" else 0.0 if fundamental.get("freight_signal") == "soft" else 0.5,
            _clamp01(max(contract_pct, benchmark_pct, 0.0) / 0.05),
            1.0 if float(fundamental.get("cargo_volume_yoy") or 0.0) > 8.0 else 0.25,
        ]
    )

    raw_scores = {
        "base_repricing": 0.30 + 0.42 * supportive + 0.12 * (1.0 - bearish) + 0.16 * (1.0 - abs(supportive - bearish)),
        "geopolitical_bull_shock": 0.18 + 0.28 * supportive + 0.36 * logistics_shock + 0.18 * (1.0 - bearish),
        "macro_bear_slump": 0.18 + 0.54 * bearish + 0.16 * (1.0 - supportive) + 0.12 * _clamp01(max(-contract_pct, -benchmark_pct, 0.0) / 0.04),
    }
    total = float(sum(raw_scores.values()) or 1.0)
    scenario_nodes = []
    for scenario_id in ("base_repricing", "geopolitical_bull_shock", "macro_bear_slump"):
        prob = float(raw_scores[scenario_id] / total)
        scenario_nodes.append(
            {
                "scenario_id": scenario_id,
                "label": SCENARIO_LABELS[scenario_id],
                "path_probability": prob,
                "path_score": float(raw_scores[scenario_id]),
                "trigger_conditions": [
                    "inventory_signal",
                    "freight_signal",
                    "demand_signal",
                    "trend_state",
                    "participant_bias",
                ],
                "invalidators": ["coverage_gap_expands", "benchmark_reversal"],
            }
        )
    primary = max(scenario_nodes, key=lambda row: row["path_probability"])["scenario_id"]
    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "contract_focus": str(input_packet.get("contract_focus") or ""),
        "primary_scenario": primary,
        "scenario_nodes": scenario_nodes,
    }
