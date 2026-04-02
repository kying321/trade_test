from lie_engine.research.commodity_reasoning_transmission import (
    CHAIN_IDS,
    build_commodity_reasoning_transmission_map,
)


def test_build_commodity_reasoning_transmission_map_outputs_sector_to_contract_chain() -> None:
    payload = build_commodity_reasoning_transmission_map(
        scenario_tree={
            "primary_scenario": "supply_chain_tightening",
            "sector_focus": "energy_chemicals",
            "commodity_focus": "asphalt",
            "contract_focus": "BU2606",
        },
        contract_focus="BU2606",
        generated_at="2026-03-27T13:40:00Z",
    )
    assert payload["primary_chain"] in CHAIN_IDS
    assert payload["chains"]
    assert any(row["contract"] == "BU2606" for row in payload["chains"])
    assert payload["chains"][0]["sector"] == "energy_chemicals"
    assert payload["chains"][0]["commodity"] == "asphalt"


def test_commodity_reasoning_transmission_map_contract_is_stable() -> None:
    payload = build_commodity_reasoning_transmission_map(
        scenario_tree={
            "primary_scenario": "policy_relief_watch",
            "sector_focus": "energy_chemicals",
            "commodity_focus": "asphalt",
            "contract_focus": "BU2606",
        },
        contract_focus="BU2606",
        generated_at="2026-03-27T13:40:00Z",
    )
    assert payload["generated_at_utc"].endswith("Z")
    assert payload["primary_chain"] in CHAIN_IDS
    assert isinstance(payload["chains"], list)
    assert all(0.0 <= float(row["confidence_score"]) <= 1.0 for row in payload["chains"])
    assert all(isinstance(row["path_nodes"], list) for row in payload["chains"])
