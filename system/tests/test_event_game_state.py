from lie_engine.research.event_game_state import (
    DEFAULT_GAME_STATE,
    GAME_STATES,
    REGIONAL_CONFLICT_NODES,
    build_event_game_state_snapshot,
)


def test_build_event_game_state_snapshot_includes_required_actors_and_state() -> None:
    timestamp = "2026-01-01T00:00:00Z"
    snapshot = build_event_game_state_snapshot(
        event_rows=[], market_inputs={}, generated_at=timestamp
    )
    actors = {actor["actor"] for actor in snapshot["actors"]}
    assert {"united_states", "china", "european_union", "russia", "opec_plus_gulf"}.issubset(actors)
    assert "regional_conflict_nodes" in actors
    regional_actor = next(
        actor for actor in snapshot["actors"] if actor["actor"] == "regional_conflict_nodes"
    )
    assert set(regional_actor.get("regional_nodes", [])) == set(REGIONAL_CONFLICT_NODES)
    assert snapshot["game_state"] == DEFAULT_GAME_STATE
    assert snapshot["game_state"] in GAME_STATES
    assert snapshot["generated_at_utc"] == timestamp
    snapshot["actors"][0]["primary_objectives"].append("leaked")

    snapshot2 = build_event_game_state_snapshot(event_rows=[], market_inputs={}, generated_at=timestamp)
    assert snapshot2["actors"][0]["primary_objectives"] == ["protect_dollar", "maintain_liquidity"]
    assert snapshot2["context"]["event_row_count"] == 0
    assert snapshot2["context"]["market_input_keys"] == []


def test_event_game_state_snapshot_contract_fields_are_stable() -> None:
    snapshot = build_event_game_state_snapshot(event_rows=[], market_inputs={}, generated_at="2026-01-01T00:00:00Z")
    assert snapshot["generated_at_utc"].endswith("Z")
    assert 0.0 <= float(snapshot["confidence_score"]) <= 1.0
    assert 0.0 <= float(snapshot["systemic_escalation_probability"]) <= 1.0
    assert 0.0 <= float(snapshot["policy_relief_probability"]) <= 1.0
    assert isinstance(snapshot["dominant_conflict_axes"], list)
    assert isinstance(snapshot["dominant_transmission_axes"], list)
    assert snapshot["game_state"] in GAME_STATES
    custom_snapshot = build_event_game_state_snapshot(
        event_rows=[{"id": "event"}], market_inputs={"foo": 1}, generated_at="2026-01-01T00:00:00Z"
    )
    assert custom_snapshot["context"]["event_row_count"] == 1
    assert custom_snapshot["context"]["market_input_keys"] == ["foo"]
