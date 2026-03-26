from __future__ import annotations

from typing import Dict, List


DEFAULT_ARCHETYPE_LIBRARY: List[Dict[str, object]] = [
    {
        "archetype_id": "gfc_2008",
        "description": "Global financial crisis with cascading credit and liquidity stress.",
        "match_axes": [
            "credit_deterioration",
            "liquidity_redemption_stress",
            "bank_intermediary_chain_risk",
        ],
        "mismatch_axes": ["energy_supply_shock"],
    },
    {
        "archetype_id": "energy_credit_2014_2016",
        "description": "Energy-led credit turbulence after price collapse.",
        "match_axes": ["energy_supply_shock", "credit_liquidity_stress"],
        "mismatch_axes": ["crypto_market_infrastructure_shock"],
    },
    {
        "archetype_id": "private_credit_redemption_2026",
        "description": "Private credit liquidity and redemption pressures spilling into banks.",
        "match_axes": [
            "liquidity_redemption_stress",
            "bank_intermediary_chain_risk",
            "credit_deterioration",
        ],
        "mismatch_axes": ["sovereign_fx_rate_shock"],
    },
    {
        "archetype_id": "oil_shock_1973_1979",
        "description": "Major energy supply shock with inflationary and stagflation themes.",
        "match_axes": ["energy_supply_shock", "policy_geopolitical_shock"],
        "mismatch_axes": ["bank_intermediary_chain_risk"],
    },
]


def build_default_archetypes() -> List[Dict[str, object]]:
    return [dict(entry) for entry in DEFAULT_ARCHETYPE_LIBRARY]


def build_top_analogues(event_axes: List[str], max_results: int = 3) -> List[Dict[str, object]]:
    if not event_axes:
        event_axes = []

    analogues: List[Dict[str, object]] = []
    for archetype in DEFAULT_ARCHETYPE_LIBRARY:
        match_axes = [axis for axis in archetype.get("match_axes", []) if axis in event_axes]
        mismatch_axes = [axis for axis in archetype.get("mismatch_axes", []) if axis not in event_axes]
        raw_score = 0.5
        if archetype.get("match_axes"):
            raw_score += 0.5 * (len(match_axes) / max(len(archetype["match_axes"]), 1))
        similarity_score = min(1.0, round(raw_score, 3))
        analogues.append(
            {
                "archetype_id": archetype["archetype_id"],
                "similarity_score": similarity_score,
                "match_axes": match_axes,
                "mismatch_axes": mismatch_axes,
            }
        )
    analogues.sort(key=lambda payload: payload["similarity_score"], reverse=True)
    return analogues[:max_results]
