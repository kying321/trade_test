from __future__ import annotations

import datetime
from typing import Any, Mapping

STATUS_CHOICES = {"watch", "active", "dominant"}

DOMINANT_CONFLICT_AXES = (
    "usd_liquidity_pressure",
    "sanctions_financial_fragmentation",
    "energy_supply_pressure",
    "shipping_chokepoint_pressure",
    "technology_supply_chain_pressure",
    "regional_conflict_escalation",
)

DOMINANT_TRANSMISSION_CHAIN_IDS = (
    "usd_liquidity_chain",
    "financial_sanctions_chain",
    "energy_supply_chain",
    "shipping_supply_chain",
    "credit_intermediary_chain",
    "risk_off_deleveraging_chain",
)

PRIMARY_THEATER_DEFAULT = "usd_liquidity_and_sanctions"

GAME_STATE_CHAIN_MAP = {
    "stable_competition": "usd_liquidity_chain",
    "financial_pressure": "credit_intermediary_chain",
    "commodity_weaponization": "energy_supply_chain",
    "bloc_fragmentation": "shipping_supply_chain",
    "systemic_repricing": "risk_off_deleveraging_chain",
}

CHAIN_DEFINITIONS = [
    {
        "chain_id": "usd_liquidity_chain",
        "origin": "usd_liquidity_pressure",
        "intermediate_nodes": [
            "usd_swap_line_tightness",
            "repo_market_spreads",
            "short_term_funding_dislocations",
        ],
        "terminal_assets": [
            "usd_swap_usage",
            "usd_libor_oo",
            "critical_payment_flows",
        ],
        "scores": {"intensity": 0.62, "velocity": 0.55, "confidence": 0.72},
    },
    {
        "chain_id": "financial_sanctions_chain",
        "origin": "sanctions_financial_fragmentation",
        "intermediate_nodes": [
            "cross_border_payment_cliffs",
            "banking_channel_deprivation",
        ],
        "terminal_assets": ["euro_swift_rolloff", "emerging_market_cds"],
        "scores": {"intensity": 0.68, "velocity": 0.61, "confidence": 0.7},
    },
    {
        "chain_id": "energy_supply_chain",
        "origin": "energy_supply_pressure",
        "intermediate_nodes": [
            "gas_pipeline_backups",
            "oil_production_rationing",
        ],
        "terminal_assets": ["brent", "natural_gas"],
        "scores": {"intensity": 0.75, "velocity": 0.65, "confidence": 0.66},
    },
    {
        "chain_id": "shipping_supply_chain",
        "origin": "shipping_chokepoint_pressure",
        "intermediate_nodes": ["red_sea_disruptions", "suez_capacity"],
        "terminal_assets": ["container_rates", "global_inventory"],
        "scores": {"intensity": 0.55, "velocity": 0.48, "confidence": 0.63},
    },
    {
        "chain_id": "credit_intermediary_chain",
        "origin": "regional_conflict_escalation",
        "intermediate_nodes": [
            "bank_credit_spreads",
            "money_market_premia",
        ],
        "terminal_assets": ["usd_credit_spread", "investment_grade_cds"],
        "scores": {"intensity": 0.7, "velocity": 0.58, "confidence": 0.69},
    },
    {
        "chain_id": "risk_off_deleveraging_chain",
        "origin": "sanctions_financial_fragmentation",
        "intermediate_nodes": ["flight_to_quality", "crypto_deleveraging"],
        "terminal_assets": ["gold", "treasury_bonds"],
        "scores": {"intensity": 0.6, "velocity": 0.5, "confidence": 0.65},
    },
]

STANDARD_PRIMARY_THEATER_BY_STATE = {
    "stable_competition": "usd_liquidity_and_sanctions",
    "financial_pressure": "usd_liquidity_and_sanctions",
    "commodity_weaponization": "energy_supply_and_shipping",
    "bloc_fragmentation": "tech_and_regional_splits",
    "systemic_repricing": "cross_asset_contagion",
}

GAME_STATE_CHAIN_INTENSITY_SCALE = {
    "stable_competition": 0.45,
    "financial_pressure": 0.65,
    "commodity_weaponization": 0.8,
    "bloc_fragmentation": 0.9,
    "systemic_repricing": 1.0,
}

GAME_STATE_ACTIVE_SUPPORT_CHAINS = {
    "stable_competition": frozenset(),
    "financial_pressure": frozenset(),
    "commodity_weaponization": frozenset({"shipping_supply_chain"}),
    "bloc_fragmentation": frozenset({"financial_sanctions_chain", "energy_supply_chain"}),
    "systemic_repricing": frozenset(
        {"usd_liquidity_chain", "financial_sanctions_chain", "credit_intermediary_chain"}
    ),
}


def _build_iso_utc_timestamp() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _ensure_status(chain_id: str, dominant_chain: str, active_support_chains: frozenset[str]) -> str:
    if chain_id == dominant_chain:
        return "dominant"
    if chain_id in active_support_chains:
        return "active"
    return "watch"


def _chain_payload(chain_def: Mapping[str, Any], status: str, intensity_scale: float) -> dict[str, Any]:
    scores = chain_def["scores"]
    return {
        "chain_id": chain_def["chain_id"],
        "origin": chain_def["origin"],
        "intermediate_nodes": list(chain_def["intermediate_nodes"]),
        "terminal_assets": list(chain_def["terminal_assets"]),
        "intensity_score": float(scores["intensity"]) * intensity_scale,
        "velocity_score": float(scores["velocity"]) * intensity_scale,
        "confidence_score": float(scores["confidence"]),
        "status": status,
    }


def build_event_transmission_chain_map(
    *,
    game_state_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = game_state_snapshot or {}
    game_state = str(snapshot.get("game_state", "financial_pressure"))
    dominant_chain = GAME_STATE_CHAIN_MAP.get(game_state, "credit_intermediary_chain")
    intensity_scale = GAME_STATE_CHAIN_INTENSITY_SCALE.get(game_state, 0.65)
    active_support_chains = GAME_STATE_ACTIVE_SUPPORT_CHAINS.get(
        game_state, frozenset()
    )

    primary_theater = STANDARD_PRIMARY_THEATER_BY_STATE.get(
        game_state, PRIMARY_THEATER_DEFAULT
    )

    chains: list[dict[str, Any]] = []
    for chain_def in CHAIN_DEFINITIONS:
        status = _ensure_status(
            chain_def["chain_id"], dominant_chain, active_support_chains
        )
        chains.append(_chain_payload(chain_def, status, intensity_scale))

    return {
        "generated_at_utc": _build_iso_utc_timestamp(),
        "primary_theater": primary_theater,
        "dominant_chain": dominant_chain,
        "chains": chains,
    }
