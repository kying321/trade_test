from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


CHAIN_BY_SCENARIO = {
    "base_repricing": "inventory_tightening_chain",
    "geopolitical_bull_shock": "geopolitical_logistics_shock_chain",
    "macro_bear_slump": "macro_demand_drag_chain",
}


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _chain_nodes(scenario_id: str, input_packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    price = dict(input_packet.get("price_snapshot") or {})
    fundamental = dict(input_packet.get("fundamental_snapshot") or {})
    contract = str(input_packet.get("contract_focus") or "")
    if scenario_id == "geopolitical_bull_shock":
        return [
            {"stage": "event", "node": "geopolitical_risk", "direction": "up"},
            {"stage": "cost", "node": "crude_cost", "direction": "up", "evidence": price.get("benchmark_pct_change_1d")},
            {"stage": "logistics", "node": "tanker_freight", "direction": "up", "evidence": fundamental.get("bdti_index")},
            {"stage": "availability", "node": "asia_supply_tightness", "direction": "up", "evidence": fundamental.get("inventory_signal")},
            {"stage": "price", "node": contract, "direction": "up"},
        ]
    if scenario_id == "macro_bear_slump":
        return [
            {"stage": "macro", "node": "global_growth", "direction": "down"},
            {"stage": "demand", "node": "cargo_and_port_throughput", "direction": "down", "evidence": fundamental.get("cargo_volume_yoy")},
            {"stage": "inventory", "node": "fuel_oil_inventory", "direction": "up", "evidence": fundamental.get("fuel_oil_inventory_delta")},
            {"stage": "price", "node": contract, "direction": "down"},
        ]
    return [
        {"stage": "cost", "node": "crude_cost_anchor", "direction": "up" if float(price.get("benchmark_pct_change_1d") or 0.0) >= 0 else "flat"},
        {"stage": "logistics", "node": "freight_and_bunkering", "direction": "up" if fundamental.get("freight_signal") == "firm" else "flat"},
        {"stage": "inventory", "node": "fuel_oil_inventory", "direction": "up" if fundamental.get("inventory_signal") == "tightening" else "down" if fundamental.get("inventory_signal") == "loosening" else "flat"},
        {"stage": "price", "node": contract, "direction": "up" if fundamental.get("relative_strength_score", 0.5) >= 0.5 else "flat"},
    ]


def build_fuel_oil_2607_transmission_map(
    *,
    input_packet: Mapping[str, Any],
    scenario_tree: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    chains = []
    for row in list(scenario_tree.get("scenario_nodes") or []):
        scenario_id = str(row.get("scenario_id") or "")
        chains.append(
            {
                "scenario_id": scenario_id,
                "chain_id": CHAIN_BY_SCENARIO.get(scenario_id, "inventory_tightening_chain"),
                "path_probability": float(row.get("path_probability") or 0.0),
                "path_nodes": _chain_nodes(scenario_id, input_packet),
            }
        )
    primary = str(scenario_tree.get("primary_scenario") or "")
    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "contract_focus": str(input_packet.get("contract_focus") or ""),
        "primary_chain": CHAIN_BY_SCENARIO.get(primary, "inventory_tightening_chain"),
        "chains": chains,
    }
