from lie_engine.research.commodity_reasoning_validation import (
    build_commodity_reasoning_validation_ring,
)


def test_build_commodity_reasoning_validation_ring_collects_counter_evidence() -> None:
    payload = build_commodity_reasoning_validation_ring(
        transmission_map={
            "primary_chain": "feedstock_cost_push_chain",
            "chains": [
                {
                    "chain_id": "feedstock_cost_push_chain",
                    "contract": "BU2606",
                    "range_scope": "contract_focused",
                    "boundary_strength": "medium",
                }
            ],
        },
        cross_section_news=[
            {"headline": "炼厂利润收缩，沥青跟涨不及原油", "stance": "counter"},
        ],
        cross_section_data=[
            {"contract": "BU2606", "basis_state": "weak", "scope_signal": "narrow"},
        ],
        generated_at="2026-03-27T14:00:00Z",
    )
    assert payload["generated_at_utc"].endswith("Z")
    assert isinstance(payload["counter_evidence"], list)
    assert isinstance(payload["scope_adjustments"], list)
    assert payload["counter_evidence"]


def test_validation_ring_never_promotes_authority() -> None:
    payload = build_commodity_reasoning_validation_ring(
        transmission_map={"primary_chain": "policy_relief_chain", "chains": []},
        cross_section_news=[],
        cross_section_data=[],
        generated_at="2026-03-27T14:00:00Z",
    )
    assert payload["promotion_allowed"] is False
    assert isinstance(payload["boundary_pressure"], str)
    assert isinstance(payload["review_required"], bool)
