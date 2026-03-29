from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


CHAIN_IDS = (
    "feedstock_cost_push_chain",
    "inventory_restock_chain",
    "policy_relief_chain",
    "cross_market_spread_chain",
)


CHAIN_LABELS = {
    "supply_chain_tightening": "feedstock_cost_push_chain",
    "cost_push_support": "feedstock_cost_push_chain",
    "demand_drag": "inventory_restock_chain",
    "cross_market_fragmentation": "cross_market_spread_chain",
    "policy_relief_watch": "policy_relief_chain",
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


def build_commodity_reasoning_transmission_map(
    *,
    scenario_tree: Mapping[str, Any],
    contract_focus: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    primary_scenario = str(scenario_tree.get("primary_scenario") or "policy_relief_watch")
    sector_focus = str(scenario_tree.get("sector_focus") or "domestic_commodities")
    commodity_focus = str(scenario_tree.get("commodity_focus") or "unknown")
    contract_focus = str(contract_focus or scenario_tree.get("contract_focus") or "").strip().upper()

    primary_chain = CHAIN_LABELS.get(primary_scenario, "policy_relief_chain")
    nodes_by_chain = {
        "feedstock_cost_push_chain": ["event_shock", "feedstock_cost", "refining_margin", contract_focus],
        "inventory_restock_chain": ["event_shock", "inventory", "spot_demand", contract_focus],
        "policy_relief_chain": ["event_shock", "policy_signal", "risk_repricing", contract_focus],
        "cross_market_spread_chain": ["event_shock", "cross_market_spread", "basis_shift", contract_focus],
    }
    confidence_by_chain = {
        "feedstock_cost_push_chain": 0.67,
        "inventory_restock_chain": 0.44,
        "policy_relief_chain": 0.38,
        "cross_market_spread_chain": 0.51,
    }

    chains = []
    for chain_id in CHAIN_IDS:
        chains.append(
            {
                "chain_id": chain_id,
                "from_event": primary_scenario,
                "sector": sector_focus,
                "commodity": commodity_focus,
                "contract": contract_focus,
                "path_nodes": list(nodes_by_chain[chain_id]),
                "range_scope": "contract_focused" if chain_id == primary_chain else "adjacent_watch",
                "boundary_strength": "medium" if chain_id == primary_chain else "watch",
                "confidence_score": _clamp01(confidence_by_chain[chain_id]),
            }
        )

    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "primary_chain": primary_chain,
        "chains": chains,
    }
