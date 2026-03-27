from lie_engine.research.event_transmission import (
    DOMINANT_TRANSMISSION_CHAIN_IDS,
    STATUS_CHOICES,
    build_event_transmission_chain_map,
)


def _dummy_game_state_snapshot() -> dict[str, str]:
    return {"game_state": "financial_pressure"}


def test_build_event_transmission_chain_map_emits_dominant_chain() -> None:
    payload = build_event_transmission_chain_map(
        game_state_snapshot=_dummy_game_state_snapshot()
    )

    assert payload["dominant_chain"] in DOMINANT_TRANSMISSION_CHAIN_IDS
    assert payload["chains"], "chains list must not be empty"


def test_event_transmission_chain_map_contract_fields_are_stable() -> None:
    payload = build_event_transmission_chain_map(
        game_state_snapshot=_dummy_game_state_snapshot()
    )

    assert payload["generated_at_utc"].endswith("Z")
    assert payload["dominant_chain"] in DOMINANT_TRANSMISSION_CHAIN_IDS
    assert isinstance(payload["chains"], list)
    assert all(0.0 <= float(row["intensity_score"]) <= 1.0 for row in payload["chains"])
    assert all(0.0 <= float(row["velocity_score"]) <= 1.0 for row in payload["chains"])
    assert all(0.0 <= float(row["confidence_score"]) <= 1.0 for row in payload["chains"])
    assert all(row["status"] in STATUS_CHOICES for row in payload["chains"])
