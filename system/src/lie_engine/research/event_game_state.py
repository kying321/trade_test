"""Minimal builder for the event game state snapshot artifact."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence
from copy import deepcopy

GAME_STATES = (
    "stable_competition",
    "financial_pressure",
    "commodity_weaponization",
    "bloc_fragmentation",
    "systemic_repricing",
)

DEFAULT_GAME_STATE = GAME_STATES[1]

PRIMARY_THEATER = "usd_liquidity_and_sanctions"

DOMINANT_CONFLICT_AXES = (
    "usd_liquidity_pressure",
    "sanctions_financial_fragmentation",
    "energy_supply_pressure",
)

DOMINANT_TRANSMISSION_AXES = (
    "usd_liquidity_chain",
    "financial_sanctions_chain",
    "risk_off_deleveraging_chain",
)

REGIONAL_CONFLICT_NODES = (
    "iran",
    "israel",
    "ukraine",
    "red_sea_shipping",
    "hormuz_shipping",
)

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
        "regional_nodes": list(REGIONAL_CONFLICT_NODES),
    },
]

GAME_STATE_PROFILES = {
    "stable_competition": {
        "confidence_score": 0.62,
        "systemic_escalation_probability": 0.12,
        "policy_relief_probability": 0.62,
    },
    "financial_pressure": {
        "confidence_score": 0.68,
        "systemic_escalation_probability": 0.32,
        "policy_relief_probability": 0.41,
    },
    "commodity_weaponization": {
        "confidence_score": 0.7,
        "systemic_escalation_probability": 0.48,
        "policy_relief_probability": 0.28,
    },
    "bloc_fragmentation": {
        "confidence_score": 0.72,
        "systemic_escalation_probability": 0.56,
        "policy_relief_probability": 0.22,
    },
    "systemic_repricing": {
        "confidence_score": 0.75,
        "systemic_escalation_probability": 0.74,
        "policy_relief_probability": 0.16,
    },
}


# Public API

def build_event_game_state_snapshot(
    *,
    event_rows: Sequence[Mapping[str, Any]],
    market_inputs: Mapping[str, Any],
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a minimal snapshot describing the headline game state."""
    if generated_at is None:
        now = datetime.now(timezone.utc)
        generated_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    game_state = _infer_game_state(event_rows=event_rows, market_inputs=market_inputs)
    profile = GAME_STATE_PROFILES[game_state]

    snapshot = {
        "generated_at_utc": generated_at,
        "primary_theater": PRIMARY_THEATER,
        "game_state": game_state,
        "confidence_score": profile["confidence_score"],
        "dominant_conflict_axes": list(DOMINANT_CONFLICT_AXES),
        "dominant_transmission_axes": list(DOMINANT_TRANSMISSION_AXES),
        "systemic_escalation_probability": profile["systemic_escalation_probability"],
        "policy_relief_probability": profile["policy_relief_probability"],
        "actors": [deepcopy(actor_spec) for actor_spec in ACTOR_SPECIFICATIONS],
        "context": {},
    }

    _normalize_actor_axes(snapshot["actors"])
    snapshot["context"] = _inject_context(event_rows, market_inputs)

    return snapshot


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _market_score(market_inputs: Mapping[str, Any], key: str) -> float:
    raw = market_inputs.get(key, 0.0)
    try:
        return _clamp01(float(raw))
    except (TypeError, ValueError):
        return 0.0


def _infer_game_state(
    *, event_rows: Sequence[Mapping[str, Any]], market_inputs: Mapping[str, Any]
) -> str:
    credit = _market_score(market_inputs, "credit_liquidity_stress_score")
    energy = _market_score(market_inputs, "energy_geopolitical_stress_score")
    cross_asset = _market_score(market_inputs, "cross_asset_deleveraging_score")
    breadth = _market_score(market_inputs, "breadth_score")

    if len(event_rows) == 0 and max(credit, energy, cross_asset, breadth) < 0.35:
        return "stable_competition"
    if max(credit, cross_asset) >= 0.82:
        return "systemic_repricing"
    if energy >= 0.72:
        return "commodity_weaponization"
    if len(event_rows) >= 3 and max(credit, energy, breadth) >= 0.55:
        return "bloc_fragmentation"
    return DEFAULT_GAME_STATE


def _normalize_actor_axes(actors: List[Dict[str, Any]]) -> None:
    for actor in actors:
        actor.setdefault("primary_objectives", [])
        actor.setdefault("hard_constraints", [])
        actor.setdefault("available_actions", [])
        actor.setdefault("observed_actions", [])
        actor.setdefault("market_impact_channels", [])


def _inject_context(
    event_rows: Sequence[Mapping[str, Any]], market_inputs: Mapping[str, Any]
) -> Dict[str, Any]:
    summary = {
        "event_row_count": len(event_rows),
        "market_input_keys": sorted(market_inputs.keys()),
    }
    return summary
