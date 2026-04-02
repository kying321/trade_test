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


def test_live_guard_overlay_uses_margin_and_hard_boundaries() -> None:
    overlay = build_event_live_guard_overlay(
        regime_snapshot={"regime_state": "sector_stress"},
        safety_margin_snapshot={
            "system_margin_score": 0.32,
            "hard_boundaries": {
                "canary_hard_block": False,
                "new_risk_hard_block": True,
                "shadow_only_boundary": False,
            },
        },
        transmission_chain_map={"dominant_chain": "risk_off_deleveraging_chain"},
        generated_at=dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc),
    )
    assert overlay["canary_freeze"] is True
    assert overlay["risk_multiplier_override"] == 0.32
    assert "event_state:sector_stress" in overlay["override_reason_codes"]
    assert "event_boundary:new_risk_hard_block" in overlay["override_reason_codes"]
    assert "event_chain:risk_off_deleveraging_chain" in overlay["override_reason_codes"]


def test_regime_snapshot_absorbs_game_state_and_transmission_inputs() -> None:
    base_snapshot = build_event_regime_snapshot(
        event_rows=_sample_event_rows(), market_inputs=_sample_market_inputs()
    )
    game_state_snapshot = {"game_state": "financial_pressure"}
    transmission_chain_map = {"dominant_chain": "credit_intermediary_chain"}
    updated_snapshot = build_event_regime_snapshot(
        event_rows=_sample_event_rows(),
        market_inputs=_sample_market_inputs(),
        game_state_snapshot=game_state_snapshot,
        transmission_chain_map=transmission_chain_map,
    )
    assert updated_snapshot["game_state"] == "financial_pressure"
    assert updated_snapshot["dominant_chain"] == "credit_intermediary_chain"
    assert (
        updated_snapshot["systemic_risk_score"] >= base_snapshot["systemic_risk_score"]
    )
    assert updated_snapshot["regime_state"] in {
        "watch",
        "sector_stress",
        "cross_asset_contagion",
        "systemic_risk",
    }


def test_regime_snapshot_without_geostrategy_inputs_stays_backwards_compatible() -> None:
    snapshot = build_event_regime_snapshot(
        event_rows=_sample_event_rows(), market_inputs=_sample_market_inputs()
    )
    assert snapshot["game_state"] is None
    assert snapshot["dominant_chain"] is None


def test_asset_shock_map_absorbs_dominant_chain_at_artifact_level() -> None:
    transmission_chain_map = {"dominant_chain": "risk_off_deleveraging_chain"}
    baseline = build_event_asset_shock_map(
        event_rows=_sample_event_rows(), market_inputs=_sample_market_inputs()
    )
    updated = build_event_asset_shock_map(
        event_rows=_sample_event_rows(),
        market_inputs=_sample_market_inputs(),
        transmission_chain_map=transmission_chain_map,
    )
    assert updated["dominant_chain"] == "risk_off_deleveraging_chain"
    assert all("dominant_chain" not in asset for asset in updated["assets"])
    baseline_btc = next(entry for entry in baseline["assets"] if entry["asset"] == "BTC")
    updated_btc = next(entry for entry in updated["assets"] if entry["asset"] == "BTC")
    assert updated_btc["risk_1d"] >= baseline_btc["risk_1d"]
