from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_commodity_reasoning_validation_ring(
    *,
    transmission_map: Mapping[str, Any],
    cross_section_news: Sequence[Mapping[str, Any]],
    cross_section_data: Sequence[Mapping[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    counter_evidence: list[str] = []
    scope_adjustments: list[str] = []

    for row in cross_section_news:
        headline = str(row.get("headline") or "").strip()
        stance = str(row.get("stance") or "").strip().lower()
        if headline and stance in {"counter", "invalidating", "weakening"}:
            counter_evidence.append(headline)

    for row in cross_section_data:
        contract = str(row.get("contract") or "").strip().upper()
        basis_state = str(row.get("basis_state") or "").strip().lower()
        scope_signal = str(row.get("scope_signal") or "").strip().lower()
        if contract and basis_state in {"weak", "divergent"}:
            counter_evidence.append(f"{contract}:basis_{basis_state}")
        if contract and scope_signal in {"narrow", "local"}:
            scope_adjustments.append(f"{contract}:scope_{scope_signal}")

    boundary_pressure = "watch"
    if len(counter_evidence) >= 2:
        boundary_pressure = "tightening"
    review_required = bool(counter_evidence or scope_adjustments)

    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "counter_evidence": counter_evidence,
        "scope_adjustments": scope_adjustments,
        "boundary_pressure": boundary_pressure,
        "review_required": review_required,
        "promotion_allowed": False,
        "primary_chain": str(transmission_map.get("primary_chain") or ""),
    }
