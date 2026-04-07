from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _build_iso_utc_timestamp(provided: str | None = None) -> str:
    if provided:
        return provided if provided.endswith("Z") else f"{provided}Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_fuel_oil_2607_validation_ring(
    *,
    input_packet: Mapping[str, Any],
    scenario_tree: Mapping[str, Any],
    transmission_map: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    _ = transmission_map
    fundamental = dict(input_packet.get("fundamental_snapshot") or {})
    technical = dict(input_packet.get("technical_snapshot") or {})
    participants = dict(input_packet.get("participant_snapshot") or {})
    refinery_proxy = dict(input_packet.get("refinery_proxy_snapshot") or {})
    coverage = dict(input_packet.get("coverage") or {})
    primary = str(scenario_tree.get("primary_scenario") or "")
    confirmations: list[str] = []
    counter_evidence: list[str] = []

    if primary == "macro_bear_slump":
        if fundamental.get("inventory_signal") == "loosening":
            confirmations.append("inventory_build_confirms_bear")
        else:
            counter_evidence.append("inventory_not_building")
        if fundamental.get("demand_signal") == "soft":
            confirmations.append("cargo_port_soft_confirms_bear")
        else:
            counter_evidence.append("demand_not_soft")
        if technical.get("daily_trend") == "down":
            confirmations.append("daily_trend_down")
        else:
            counter_evidence.append("daily_trend_not_down")
        if participants.get("long_short_bias") == "net_short":
            confirmations.append("member_net_short")
        else:
            counter_evidence.append("member_not_net_short")
        if refinery_proxy.get("margin_signal") == "compressed":
            confirmations.append("refinery_margin_proxy_compressed")
        else:
            counter_evidence.append("refinery_margin_proxy_not_compressed")
        if refinery_proxy.get("run_rate_signal") == "slowing":
            confirmations.append("refinery_run_proxy_slowing")
        else:
            counter_evidence.append("refinery_run_proxy_not_slowing")
    else:
        if fundamental.get("inventory_signal") == "tightening":
            confirmations.append("inventory_draw_supports_price")
        else:
            counter_evidence.append("inventory_signal_not_supportive")
        if fundamental.get("freight_signal") == "firm":
            confirmations.append("freight_signal_firm")
        else:
            counter_evidence.append("freight_signal_not_firm")
        if fundamental.get("demand_signal") == "firm":
            confirmations.append("cargo_and_port_support")
        else:
            counter_evidence.append("demand_signal_not_firm")
        if technical.get("daily_trend") == "up":
            confirmations.append("daily_trend_up")
        else:
            counter_evidence.append("daily_trend_not_up")
        if participants.get("long_short_bias") == "net_long":
            confirmations.append("member_net_long")
        else:
            counter_evidence.append("member_not_net_long")
        if refinery_proxy.get("margin_signal") == "supportive":
            confirmations.append("refinery_margin_proxy_support")
        else:
            counter_evidence.append("refinery_margin_proxy_not_supportive")
        if refinery_proxy.get("run_rate_signal") == "active":
            confirmations.append("refinery_run_proxy_active")
        else:
            counter_evidence.append("refinery_run_proxy_not_active")

    coverage_gaps = list(coverage.get("missing_fields") or [])
    coverage_ratio = float(coverage.get("coverage_ratio") or 0.0)
    if primary == "macro_bear_slump":
        boundary_pressure = "fragile"
    elif coverage_ratio < 0.65 or len(confirmations) <= len(counter_evidence):
        boundary_pressure = "balanced"
    else:
        boundary_pressure = "supportive"
    return {
        "generated_at_utc": _build_iso_utc_timestamp(generated_at),
        "primary_scenario": primary,
        "confirmations": confirmations,
        "counter_evidence": counter_evidence,
        "coverage_gaps": coverage_gaps,
        "coverage_ratio": coverage_ratio,
        "boundary_pressure": boundary_pressure,
    }
