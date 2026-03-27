"""Minimal builder for the event game state snapshot artifact."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Sequence

GAME_STATES = [
    "stable_competition",
    "financial_pressure",
    "commodity_weaponization",
    "bloc_fragmentation",
    "systemic_repricing",
]

PRIMARY_THEATER = "usd_liquidity_and_sanctions"

DOMINANT_CONFLICT_AXES = [
    "usd_liquidity_pressure",
    "sanctions_financial_fragmentation",
    "energy_supply_pressure",
]

DOMINANT_TRANSMISSION_AXES = [
    "usd_liquidity_chain",
    "financial_sanctions_chain",
    "risk_off_deleveraging_chain",
]

ACTOR_SPECIFICATIONS = [
    {
        "actor": "united_states",
        "theater": "usd_liquidity_and_sanctions",
        "primary_objectives": ["protect_dollar", "maintain_liquidity"],
        "hard_constraints": ["sanctions_policy", "domestic_debt_limits"],
        "available_actions": ["sanctions", "reserve_policy"],
        "observed_actions": ["financial_fragmentation", "dollar_swap_directives"],
        "dominant_strategy": "liquidity_rollover",
        "escalation_level": 0.45,
        "deescalation_probability": 0.35,
        "market_impact_channels": ["usd_liquidity_chain", "risk_off_deleveraging_chain"],
    },
    {
        "actor": "china",
        "theater": "technology_export_controls",
        "primary_objectives": ["preserve_supply_chains", "support_exporters"],
        "hard_constraints": ["credit_risks", "capital_controls"],
        "available_actions": ["dual_currency_settlements", "export_controls"],
        "observed_actions": ["currency_swap_expansions", "supply_chain_protocols"],
        "dominant_strategy": "strategic_resilience",
        "escalation_level": 0.4,
        "deescalation_probability": 0.38,
        "market_impact_channels": ["credit_intermediary_chain", "energy_supply_chain"],
    },
    {
        "actor": "european_union",
        "theater": "usd_liquidity_and_sanctions",
        "primary_objectives": ["energy_rotation", "credit_stability"],
        "hard_constraints": ["energy_dependency", "policy_divergence"],
        "available_actions": ["diversified_imports", "macro_stability_support"],
        "observed_actions": ["joint_liquidity_swaps", "renewables_investments"],
        "dominant_strategy": "policy_coordination",
        "escalation_level": 0.3,
        "deescalation_probability": 0.5,
        "market_impact_channels": ["energy_supply_chain", "credit_intermediary_chain"],
    },
    {
        "actor": "russia",
        "theater": "energy_supply_and_shipping",
        "primary_objectives": ["protect_energy_exports", "maintain_chokepoints"],
        "hard_constraints": ["sanctions", "logistics"],
        "available_actions": ["currency_swaps", "strategic_withdrawals"],
        "observed_actions": ["shipping_channel_reroutes", "energy_weaponization"],
        "dominant_strategy": "commodity_weaponization",
        "escalation_level": 0.5,
        "deescalation_probability": 0.2,
        "market_impact_channels": ["energy_supply_chain", "shipping_supply_chain"],
    },
    {
        "actor": "opec_plus_gulf",
        "theater": "energy_supply_and_shipping",
        "primary_objectives": ["price_stability", "market_share"],
        "hard_constraints": ["demand_shocks", "production_limits"],
        "available_actions": ["production_cuts", "diplomatic_cohesion"],
        "observed_actions": ["quota_adjustments", "joint_coordination"],
        "dominant_strategy": "supply_management",
        "escalation_level": 0.25,
        "deescalation_probability": 0.4,
        "market_impact_channels": ["energy_supply_chain", "usd_liquidity_chain"],
    },
    {
        "actor": "regional_conflict_nodes",
        "theater": "regional_conflict_escalation",
        "primary_objectives": ["local_stability", "strategic_buffer"],
        "hard_constraints": ["military_readiness", "humanitarian_pressure"],
        "available_actions": ["targeted_strikes", "mediation"],
        "observed_actions": ["proxy_clashes", "conflict_escalation"],
        "dominant_strategy": "containment",
        "escalation_level": 0.55,
        "deescalation_probability": 0.22,
        "market_impact_channels": ["shipping_supply_chain", "energy_supply_chain"],
        "regional_nodes": [
            "iran",
            "israel",
            "ukraine",
            "red_sea_shipping",
            "hormuz_shipping",
        ],
    },
]


# Public API

def build_event_game_state_snapshot(
    *,
    event_rows: Sequence[Mapping[str, Any]],
    market_inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a minimal snapshot describing the headline game state."""
    now = datetime.now(timezone.utc)
    generated_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    snapshot = {
        "generated_at_utc": generated_at,
        "primary_theater": PRIMARY_THEATER,
        "game_state": "financial_pressure",
        "confidence_score": 0.68,
        "dominant_conflict_axes": list(DOMINANT_CONFLICT_AXES),
        "dominant_transmission_axes": list(DOMINANT_TRANSMISSION_AXES),
        "systemic_escalation_probability": 0.32,
        "policy_relief_probability": 0.41,
        "actors": [dict(actor_spec) for actor_spec in ACTOR_SPECIFICATIONS],
    }

    _normalize_actor_axes(snapshot["actors"])
    _inject_context(event_rows, market_inputs)

    return snapshot


def _normalize_actor_axes(actors: List[Dict[str, Any]]) -> None:
    for actor in actors:
        actor.setdefault("primary_objectives", [])
        actor.setdefault("hard_constraints", [])
        actor.setdefault("available_actions", [])
        actor.setdefault("observed_actions", [])
        actor.setdefault("market_impact_channels", [])


def _inject_context(
    event_rows: Sequence[Mapping[str, Any]], market_inputs: Mapping[str, Any]
) -> None:
    # Minimal stub: keep the deterministic defaults, but record counts for traceability
    summary = {
        "event_row_count": len(event_rows),
        "market_input_keys": sorted(market_inputs.keys()),
    }
    # We purposely avoid network I/O or deep inference for this research-only skeleton.
    # The summary can be extended later when more data becomes available.
    return summary
