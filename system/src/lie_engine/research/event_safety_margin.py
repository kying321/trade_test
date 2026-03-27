"""Minimal builder for the event safety margin snapshot artifact."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence

CHAIN_GROUPS: Dict[str, Sequence[str]] = {
    "liquidity": [
        "usd_liquidity_chain",
        "financial_sanctions_chain",
        "risk_off_deleveraging_chain",
    ],
    "credit": ["credit_intermediary_chain", "usd_liquidity_chain"],
    "energy": ["energy_supply_chain", "shipping_supply_chain"],
}
ACTIVE_STATUSES = {"active", "dominant"}


# Helpers

def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


def _chain_threat(chain: Mapping[str, Any]) -> float:
    intensity = _clamp01(float(chain.get("intensity_score", 0.0)))
    velocity = _clamp01(float(chain.get("velocity_score", 0.0)))
    status = chain.get("status")
    base = _clamp01(0.6 * intensity + 0.4 * velocity)
    if status == "dominant":
        base = _clamp01(base + 0.15)
    elif status == "active":
        base = _clamp01(base + 0.08)
    return base


def _build_chain_map(chains: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(chain.get("chain_id")): chain
        for chain in chains
        if chain.get("chain_id") is not None
    }


def _average_threat(chain_map: Dict[str, Mapping[str, Any]], chain_ids: Sequence[str]) -> float:
    threats: List[float] = []
    for chain_id in chain_ids:
        chain = chain_map.get(chain_id)
        if not chain:
            continue
        threats.append(_chain_threat(chain))
    if not threats:
        return 0.25
    return sum(threats) / len(threats)


def _margin_from_threat(threat: float) -> float:
    return _clamp01(1.0 - threat)


def _active_chain_threats(chains: Iterable[Mapping[str, Any]]) -> List[float]:
    output: List[float] = []
    for chain in chains:
        if chain.get("status") in ACTIVE_STATUSES:
            output.append(_chain_threat(chain))
    return output


# Public API

def build_event_safety_margin_snapshot(
    *,
    game_state_snapshot: Mapping[str, Any] | None = None,
    transmission_chain_map: Mapping[str, Any] | None = None,
    regime_snapshot: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    """Build a compact safety margin snapshot reflecting chains, policy, and boundaries."""
    game_state_snapshot = game_state_snapshot or {}
    transmission_chain_map = transmission_chain_map or {}
    regime_snapshot = regime_snapshot or {}

    chains: List[Mapping[str, Any]] = transmission_chain_map.get("chains", [])
    chain_map = _build_chain_map(chains)

    dominant_chain = transmission_chain_map.get("dominant_chain")
    dominant_chain_data = chain_map.get(str(dominant_chain)) if dominant_chain else None

    liquidity_threat = _average_threat(chain_map, CHAIN_GROUPS["liquidity"])
    credit_threat = _average_threat(chain_map, CHAIN_GROUPS["credit"])
    energy_threat = _average_threat(chain_map, CHAIN_GROUPS["energy"])

    liquidity_margin = _margin_from_threat(liquidity_threat)
    credit_margin = _margin_from_threat(credit_threat)
    energy_margin = _margin_from_threat(energy_threat)

    policy_relief = _clamp01(float(game_state_snapshot.get("policy_relief_probability", 0.25)))
    dominant_threat = (_chain_threat(dominant_chain_data) if dominant_chain_data else 0.0)
    policy_margin = _clamp01(policy_relief + 0.15 * (1.0 - dominant_threat))

    margins = [liquidity_margin, credit_margin, energy_margin, policy_margin]
    base_average = sum(margins) / len(margins)
    worst_margin = min(margins)

    active_threats = _active_chain_threats(chains)
    max_active_threat = max(active_threats) if active_threats else 0.0
    dominant_margin_equiv = _margin_from_threat(dominant_threat)
    danger_influence = _clamp01(0.6 * dominant_margin_equiv + 0.4 * (1.0 - max_active_threat))

    system_base = (
        0.3 * base_average
        + 0.3 * danger_influence
        + 0.2 * worst_margin
        + 0.2 * (1.0 - max_active_threat)
    )

    regime_state = str(regime_snapshot.get("regime_state", ""))
    game_state = str(game_state_snapshot.get("game_state", ""))
    canary_trigger = any(
        (
            game_state == "systemic_repricing",
            regime_state == "systemic_risk",
            dominant_chain == "risk_off_deleveraging_chain" and dominant_chain_data and dominant_chain_data.get("status") == "dominant",
            credit_margin < 0.25,
        )
    )

    multi_active = len(active_threats) >= 2
    low_system_margin = system_base < 0.35
    new_risk_trigger = multi_active or low_system_margin

    shadow_only = not (canary_trigger or new_risk_trigger) and min(margins) < 0.6

    boundary_penalty = 0.12 if canary_trigger else 0.08 if new_risk_trigger else 0.0
    system_margin_score = _clamp01(system_base - boundary_penalty)

    hard_boundaries = {
        "canary_hard_block": canary_trigger,
        "new_risk_hard_block": new_risk_trigger,
        "shadow_only_boundary": shadow_only,
    }

    reasons: List[str] = []
    if game_state == "systemic_repricing":
        reasons.append("game_state=systemic_repricing")
    if regime_state == "systemic_risk":
        reasons.append("regime_state=systemic_risk")
    if dominant_chain == "risk_off_deleveraging_chain" and dominant_chain_data and dominant_chain_data.get("status") == "dominant":
        reasons.append("dominant_chain=risk_off_deleveraging_chain")
    if credit_margin < 0.25:
        reasons.append("credit_margin_critical")
    if multi_active:
        reasons.append("multiple_active_chains")
    if low_system_margin:
        reasons.append("system_margin_low")
    if shadow_only and not reasons:
        reasons.append("shadow_only_early_warning")

    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "liquidity_margin": liquidity_margin,
        "credit_margin": credit_margin,
        "energy_margin": energy_margin,
        "policy_margin": policy_margin,
        "system_margin_score": system_margin_score,
        "hard_boundaries": hard_boundaries,
        "boundary_reasons": reasons,
    }
