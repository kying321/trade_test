from lie_engine.research.event_safety_margin import build_event_safety_margin_snapshot


def _dummy_chain(chain_id: str, status: str = "dominant") -> dict[str, object]:
    return {
        "chain_id": chain_id,
        "intensity_score": 0.6,
        "velocity_score": 0.55,
        "confidence_score": 0.7,
        "status": status,
    }


def _base_payload() -> dict[str, object]:
    return {
        "game_state_snapshot": {"game_state": "systemic_repricing", "policy_relief_probability": 0.4},
        "transmission_chain_map": {
            "dominant_chain": "risk_off_deleveraging_chain",
            "chains": [
                _dummy_chain("usd_liquidity_chain", status="active"),
                _dummy_chain("risk_off_deleveraging_chain", status="dominant"),
                _dummy_chain("credit_intermediary_chain", status="active"),
            ],
        },
        "regime_snapshot": {"regime_state": "systemic_risk"},
    }


def test_build_event_safety_margin_snapshot_outputs_margins_and_boundaries() -> None:
    payload = build_event_safety_margin_snapshot(**_base_payload())
    assert 0.0 <= float(payload["system_margin_score"]) <= 1.0
    assert set(payload["hard_boundaries"].keys()) == {
        "canary_hard_block",
        "new_risk_hard_block",
        "shadow_only_boundary",
    }


def test_event_safety_margin_snapshot_contract_fields_are_stable() -> None:
    payload = build_event_safety_margin_snapshot(**_base_payload())
    assert payload["generated_at_utc"].endswith("Z")
    assert 0.0 <= float(payload["liquidity_margin"]) <= 1.0
    assert 0.0 <= float(payload["credit_margin"]) <= 1.0
    assert 0.0 <= float(payload["energy_margin"]) <= 1.0
    assert 0.0 <= float(payload["policy_margin"]) <= 1.0
    assert 0.0 <= float(payload["system_margin_score"]) <= 1.0
    assert isinstance(payload["boundary_reasons"], list)
    assert set(payload["hard_boundaries"].keys()) == {
        "canary_hard_block",
        "new_risk_hard_block",
        "shadow_only_boundary",
    }
