from lie_engine.research.event_game_state import build_event_game_state_snapshot


def test_build_event_game_state_snapshot_includes_required_actors_and_state() -> None:
    snapshot = build_event_game_state_snapshot(event_rows=[], market_inputs={})
    actors = {actor["actor"] for actor in snapshot["actors"]}
    assert {"united_states", "china", "european_union", "russia", "opec_plus_gulf"}.issubset(actors)
    assert snapshot["game_state"] in {
        "stable_competition",
        "financial_pressure",
        "commodity_weaponization",
        "bloc_fragmentation",
        "systemic_repricing",
    }


def test_event_game_state_snapshot_contract_fields_are_stable() -> None:
    snapshot = build_event_game_state_snapshot(event_rows=[], market_inputs={})
    assert snapshot["generated_at_utc"].endswith("Z")
    assert 0.0 <= float(snapshot["confidence_score"]) <= 1.0
    assert 0.0 <= float(snapshot["systemic_escalation_probability"]) <= 1.0
    assert 0.0 <= float(snapshot["policy_relief_probability"]) <= 1.0
    assert isinstance(snapshot["dominant_conflict_axes"], list)
    assert isinstance(snapshot["dominant_transmission_axes"], list)
    assert snapshot["game_state"] in {
        "stable_competition",
        "financial_pressure",
        "commodity_weaponization",
        "bloc_fragmentation",
        "systemic_repricing",
    }
