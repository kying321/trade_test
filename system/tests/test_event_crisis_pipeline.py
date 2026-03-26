from typing import List

from lie_engine.research.event_crisis_pipeline import (
    build_event_asset_shock_map,
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


def test_build_event_asset_shock_map_covers_priority_assets() -> None:
    payload = build_event_asset_shock_map(
        event_rows=_sample_event_rows(), market_inputs=_sample_market_inputs()
    )
    asset_names = {entry["asset"] for entry in payload["assets"]}
    assert {"BTC", "ETH", "SOL", "BNB", "GOLD", "UST_LONG", "OIL", "BANKS", "HIGH_YIELD"} <= asset_names
