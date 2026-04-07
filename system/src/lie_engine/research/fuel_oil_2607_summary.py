from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_fuel_oil_2607_summary(
    *,
    input_packet: Mapping[str, Any],
    scenario_tree: Mapping[str, Any],
    validation_ring: Mapping[str, Any],
    trade_space: Mapping[str, Any],
    strategy_matrix: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    weighted = dict(trade_space.get("weighted_range") or {})
    strategies = list(strategy_matrix.get("priority_strategies") or [])
    first_strategy = dict(strategies[0] if strategies else {})
    confirmations = list(validation_ring.get("confirmations") or [])
    counter = list(validation_ring.get("counter_evidence") or [])
    coverage = dict(input_packet.get("coverage") or {})
    primary = str(scenario_tree.get("primary_scenario") or "")
    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "headline": f"{input_packet.get('contract_focus') or 'FU2607'} {primary} | bias={strategy_matrix.get('preferred_bias') or 'neutral'}",
        "primary_scenario_brief": primary,
        "validation_brief": str(validation_ring.get("boundary_pressure") or ""),
        "key_evidence_brief": ",".join(confirmations[:3]),
        "key_risk_brief": ",".join(counter[:2] + list(coverage.get("missing_fields") or [])[:1]),
        "weighted_range_brief": f"{float(weighted.get('lower') or 0.0):.1f}-{float(weighted.get('upper') or 0.0):.1f}",
        "priority_strategy_brief": str(first_strategy.get("strategy_name") or ""),
        "preferred_bias": str(strategy_matrix.get("preferred_bias") or ""),
        "coverage_brief": f"{float(coverage.get('coverage_ratio') or 0.0):.0%}",
        "contracts_in_focus": [
            str(input_packet.get("contract_focus") or ""),
            str(input_packet.get("benchmark_contract") or ""),
        ],
    }
