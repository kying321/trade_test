from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_commodity_reasoning_boundary_strength(
    *,
    transmission_map: Mapping[str, Any],
    validation_ring: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    rows = []
    for row in list(transmission_map.get("chains") or []):
        if not isinstance(row, Mapping):
            continue
        fragility_flags = list(validation_ring.get("scope_adjustments") or [])
        counter_evidence = list(validation_ring.get("counter_evidence") or [])
        rows.append(
            {
                "target_level": "contract",
                "target_id": str(row.get("contract") or ""),
                "range_scope": str(row.get("range_scope") or ""),
                "boundary_strength": str(validation_ring.get("boundary_pressure") or row.get("boundary_strength") or "watch"),
                "persistence_strength": "watch",
                "fragility_flags": fragility_flags,
                "counter_evidence": counter_evidence,
            }
        )
    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "range_summary": rows[0]["range_scope"] if rows else "unknown",
        "boundary_rows": rows,
    }
