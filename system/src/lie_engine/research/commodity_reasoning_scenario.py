from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


SCENARIO_IDS = (
    "supply_chain_tightening",
    "cost_push_support",
    "demand_drag",
    "cross_market_fragmentation",
    "policy_relief_watch",
)


CONTRACT_FOCUS_MAP = {
    "BU": {"sector": "energy_chemicals", "commodity": "asphalt"},
}


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _contract_focus_metadata(contract_focus: str) -> dict[str, str]:
    text = str(contract_focus or "").strip().upper()
    prefix = "".join(ch for ch in text if ch.isalpha())[:2]
    mapped = CONTRACT_FOCUS_MAP.get(prefix, {"sector": "domestic_commodities", "commodity": prefix.lower() or "unknown"})
    return {
        "sector_focus": mapped["sector"],
        "commodity_focus": mapped["commodity"],
        "contract_focus": text,
    }


def _joined_text(rows: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for row in rows:
        for key in ("headline", "summary", "summary_text", "takeaway"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip().lower())
    return " ".join(parts)


def _primary_scenario(text: str) -> str:
    if any(token in text for token in ("扰动", "supply", "shipping", "航运", "原油", "炼化", "能源")):
        return "supply_chain_tightening"
    if any(token in text for token in ("需求", "inventory", "消费", "demand")):
        return "demand_drag"
    if any(token in text for token in ("信用", "cross", "fragment", "碎片化")):
        return "cross_market_fragmentation"
    return "policy_relief_watch"


def _scenario_nodes(primary: str, sector_focus: str, commodity_focus: str, contract_focus: str) -> list[dict[str, Any]]:
    base_nodes = {
        "supply_chain_tightening": {
            "label": "供应链收紧",
            "trigger_conditions": [f"{commodity_focus}_cost_push", f"{sector_focus}_upstream_disruption"],
            "invalidators": ["freight_relief", "feedstock_supply_normalization"],
            "confidence_score": 0.68,
        },
        "cost_push_support": {
            "label": "成本抬升支撑",
            "trigger_conditions": [f"{commodity_focus}_margin_support"],
            "invalidators": ["inventory_rebuild", "crack_spread_compression"],
            "confidence_score": 0.58,
        },
        "demand_drag": {
            "label": "需求拖累",
            "trigger_conditions": [f"{sector_focus}_demand_softening"],
            "invalidators": ["terminal_restocking", "policy_stimulus"],
            "confidence_score": 0.42,
        },
        "cross_market_fragmentation": {
            "label": "跨市场分化",
            "trigger_conditions": [f"{contract_focus}_basis_dispersion"],
            "invalidators": ["cross_region_spread_recompression"],
            "confidence_score": 0.47,
        },
        "policy_relief_watch": {
            "label": "政策缓和观察",
            "trigger_conditions": ["policy_relief_signals"],
            "invalidators": ["unexpected_supply_shock"],
            "confidence_score": 0.35,
        },
    }
    secondary = [scenario_id for scenario_id in SCENARIO_IDS if scenario_id != primary][:2]
    rows = [
        {
            "scenario_id": primary,
            "label": base_nodes[primary]["label"],
            "level": "primary",
            "trigger_conditions": list(base_nodes[primary]["trigger_conditions"]),
            "invalidators": list(base_nodes[primary]["invalidators"]),
            "confidence_score": _clamp01(float(base_nodes[primary]["confidence_score"])),
        }
    ]
    for scenario_id in secondary:
        node = base_nodes[scenario_id]
        rows.append(
            {
                "scenario_id": scenario_id,
                "label": node["label"],
                "level": "secondary",
                "trigger_conditions": list(node["trigger_conditions"]),
                "invalidators": list(node["invalidators"]),
                "confidence_score": _clamp01(float(node["confidence_score"])),
            }
        )
    return rows


def build_commodity_reasoning_scenario_tree(
    *,
    event_artifacts: Sequence[Mapping[str, Any]],
    research_artifacts: Sequence[Mapping[str, Any]],
    contract_focus: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    meta = _contract_focus_metadata(contract_focus)
    joined = _joined_text([*event_artifacts, *research_artifacts])
    primary = _primary_scenario(joined)
    nodes = _scenario_nodes(
        primary,
        meta["sector_focus"],
        meta["commodity_focus"],
        meta["contract_focus"],
    )
    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "root_theme": "domestic_commodity_reasoning",
        "sector_focus": meta["sector_focus"],
        "commodity_focus": meta["commodity_focus"],
        "contract_focus": meta["contract_focus"],
        "primary_scenario": primary,
        "secondary_scenarios": [row["scenario_id"] for row in nodes[1:]],
        "scenario_nodes": nodes,
    }
