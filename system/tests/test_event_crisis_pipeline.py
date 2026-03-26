from typing import List

import datetime as dt

from lie_engine.research.event_crisis_pipeline import (
    build_event_asset_shock_map,
    build_event_live_guard_overlay,
    build_event_regime_snapshot,
)


def _sample_event_rows() -> List[dict]:
    return [
        {
            "event_id": "private-credit-redemption-2026",
            "event_classes": [
                "credit_deterioration",
                "liquidity_redemption_stress",
                "bank_intermediary_chain_risk",
            ],
            "headline": "Private credit redemption wave deepens",
        }
    ]


def _sample_market_inputs() -> dict:
    return {
        "credit_liquidity_stress_score": 0.6,
        "energy_geopolitical_stress_score": 0.35,
        "cross_asset_deleveraging_score": 0.32,
        "breadth_score": 0.45,
        "contagion_score": 0.4,
        "persistence_score": 0.5,
        "policy_offset_score": 0.2,
        "confidence_score": 0.65,
    }


def test_build_event_regime_snapshot_outputs_scores_and_regime_state() -> None:
    snapshot = build_event_regime_snapshot(
        event_rows=_sample_event_rows(), market_inputs=_sample_market_inputs()
    )

    assert snapshot["regime_state"] in {
        "watch",
        "sector_stress",
        "cross_asset_contagion",
        "systemic_risk",
    }
    assert snapshot["regime_state"] == "sector_stress"
    assert 0.0 <= float(snapshot["event_severity_score"]) <= 1.0
    assert 0.0 <= float(snapshot["systemic_risk_score"]) <= 1.0


def test_build_event_regime_snapshot_defaults_to_sector_stress_for_recent_event() -> None:
    snapshot = build_event_regime_snapshot(event_rows=_sample_event_rows(), market_inputs={})
    assert snapshot["regime_state"] == "sector_stress"


def test_build_event_asset_shock_map_covers_priority_assets() -> None:
    payload = build_event_asset_shock_map(
        event_rows=_sample_event_rows(), market_inputs=_sample_market_inputs()
    )
    asset_names = {entry["asset"] for entry in payload["assets"]}
    assert {"BTC", "ETH", "SOL", "BNB", "GOLD", "UST_LONG", "OIL", "BANKS", "HIGH_YIELD"} <= asset_names


def test_build_event_asset_shock_map_senses_raw_stress_inputs() -> None:
    low_inputs = {
        "credit_liquidity_stress_score": 0.1,
        "energy_geopolitical_stress_score": 0.1,
        "cross_asset_deleveraging_score": 0.1,
        "contagion_score": 0.1,
    }
    high_inputs = {
        "credit_liquidity_stress_score": 0.9,
        "energy_geopolitical_stress_score": 0.85,
        "cross_asset_deleveraging_score": 0.8,
        "contagion_score": 0.9,
    }

    low_payload = build_event_asset_shock_map(
        event_rows=_sample_event_rows(), market_inputs=low_inputs
    )
    high_payload = build_event_asset_shock_map(
        event_rows=_sample_event_rows(), market_inputs=high_inputs
    )

    low_btc = next(entry for entry in low_payload["assets"] if entry["asset"] == "BTC")
    high_btc = next(entry for entry in high_payload["assets"] if entry["asset"] == "BTC")
    assert high_btc["risk_1d"] > low_btc["risk_1d"]


def test_build_event_live_guard_overlay_provides_valid_degraded_state() -> None:
    snapshot = build_event_regime_snapshot(
        event_rows=_sample_event_rows(), market_inputs=_sample_market_inputs()
    )
    overlay = build_event_live_guard_overlay(
        regime_snapshot=snapshot,
        generated_at=dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc),
    )
    assert overlay["risk_multiplier_override"] <= 1.0
    assert isinstance(overlay["canary_freeze"], bool)
    assert overlay["override_reason_codes"] == ["event_state:sector_stress"]
    assert overlay["valid_until_utc"].endswith("Z")
