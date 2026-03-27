from lie_engine.research.commodity_reasoning_scenario import (
    SCENARIO_IDS,
    build_commodity_reasoning_scenario_tree,
)


def test_build_commodity_reasoning_scenario_tree_emits_primary_and_secondary_scenarios() -> None:
    payload = build_commodity_reasoning_scenario_tree(
        event_artifacts=[
            {
                "artifact_id": "event_crisis_operator_summary",
                "headline": "中东扰动推升原油与炼化链成本",
                "summary": "能源链扰动仍在扩散",
            }
        ],
        research_artifacts=[
            {
                "artifact_id": "hot_universe_research",
                "summary_text": "metals_all 暂无边际，energy_liquids 维持 shadow watch",
            }
        ],
        contract_focus="BU2606",
        generated_at="2026-03-27T13:30:00Z",
    )

    assert payload["root_theme"] == "domestic_commodity_reasoning"
    assert payload["sector_focus"] == "energy_chemicals"
    assert payload["commodity_focus"] == "asphalt"
    assert payload["contract_focus"] == "BU2606"
    assert payload["primary_scenario"] in SCENARIO_IDS
    assert isinstance(payload["secondary_scenarios"], list)
    assert payload["secondary_scenarios"]
    assert payload["scenario_nodes"]
    assert payload["scenario_nodes"][0]["scenario_id"] == payload["primary_scenario"]


def test_commodity_reasoning_scenario_tree_contract_fields_are_stable() -> None:
    payload = build_commodity_reasoning_scenario_tree(
        event_artifacts=[],
        research_artifacts=[],
        contract_focus="BU2606",
        generated_at="2026-03-27T13:30:00Z",
    )

    assert payload["generated_at_utc"].endswith("Z")
    assert payload["primary_scenario"] in SCENARIO_IDS
    assert isinstance(payload["secondary_scenarios"], list)
    assert isinstance(payload["scenario_nodes"], list)
    assert all(0.0 <= float(row["confidence_score"]) <= 1.0 for row in payload["scenario_nodes"])
    assert all(isinstance(row["trigger_conditions"], list) for row in payload["scenario_nodes"])
    assert all(isinstance(row["invalidators"], list) for row in payload["scenario_nodes"])
