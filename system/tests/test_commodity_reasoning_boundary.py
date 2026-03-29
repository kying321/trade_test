from lie_engine.research.commodity_reasoning_boundary import (
    build_commodity_reasoning_boundary_strength,
)
from lie_engine.research.commodity_reasoning_summary import (
    build_commodity_reasoning_summary,
)


def test_build_commodity_reasoning_boundary_strength_outputs_range_and_fragility() -> None:
    payload = build_commodity_reasoning_boundary_strength(
        transmission_map={
            "chains": [
                {
                    "contract": "BU2606",
                    "sector": "energy_chemicals",
                    "commodity": "asphalt",
                    "range_scope": "contract_focused",
                    "boundary_strength": "medium",
                }
            ]
        },
        validation_ring={
            "counter_evidence": ["BU2606:basis_weak"],
            "scope_adjustments": ["BU2606:scope_narrow"],
            "boundary_pressure": "tightening",
        },
        generated_at="2026-03-27T14:20:00Z",
    )
    assert payload["generated_at_utc"].endswith("Z")
    assert payload["boundary_rows"]
    assert all(isinstance(row["fragility_flags"], list) for row in payload["boundary_rows"])
    assert all(isinstance(row["counter_evidence"], list) for row in payload["boundary_rows"])


def test_build_commodity_reasoning_summary_outputs_operator_briefs() -> None:
    payload = build_commodity_reasoning_summary(
        scenario_tree={
            "primary_scenario": "supply_chain_tightening",
            "contract_focus": "BU2606",
        },
        transmission_map={
            "primary_chain": "feedstock_cost_push_chain",
            "chains": [{"contract": "BU2606"}],
        },
        boundary_strength={
            "range_summary": "contract_focused",
            "boundary_rows": [{"boundary_strength": "tight", "fragility_flags": ["basis_weak"]}],
        },
        generated_at="2026-03-27T14:20:00Z",
    )
    assert payload["generated_at_utc"].endswith("Z")
    assert payload["primary_scenario_brief"]
    assert payload["primary_chain_brief"]
    assert payload["range_scope_brief"]
    assert payload["boundary_strength_brief"]
    assert payload["invalidator_brief"]
    assert payload["contracts_in_focus"] == ["BU2606"]
