from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_commodity_reasoning_summary(
    *,
    scenario_tree: Mapping[str, Any],
    transmission_map: Mapping[str, Any],
    boundary_strength: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    contract_focus = str(scenario_tree.get("contract_focus") or "")
    row = next(iter(boundary_strength.get("boundary_rows") or []), {})
    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "headline": f"{contract_focus or 'commodity'} reasoning",
        "primary_scenario_brief": str(scenario_tree.get("primary_scenario") or ""),
        "primary_chain_brief": str(transmission_map.get("primary_chain") or ""),
        "range_scope_brief": str(boundary_strength.get("range_summary") or ""),
        "boundary_strength_brief": str(row.get("boundary_strength") or ""),
        "invalidator_brief": ",".join(list(row.get("fragility_flags") or [])) if row else "",
        "contracts_in_focus": [contract_focus] if contract_focus else [],
    }
