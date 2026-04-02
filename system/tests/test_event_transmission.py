from lie_engine.research.event_transmission import build_event_transmission_chain_map

DOMINANT_TRANSMISSION_CHAIN_IDS = {
    "usd_liquidity_chain",
    "financial_sanctions_chain",
    "energy_supply_chain",
    "shipping_supply_chain",
    "credit_intermediary_chain",
    "risk_off_deleveraging_chain",
}

STATUS_ENUM = {"watch", "active", "dominant"}

DOMINANT_CONFLICT_AXES = {
    "usd_liquidity_pressure",
    "sanctions_financial_fragmentation",
    "energy_supply_pressure",
    "shipping_chokepoint_pressure",
    "technology_supply_chain_pressure",
    "regional_conflict_escalation",
}


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

    chain_ids = [row["chain_id"] for row in payload["chains"]]
    assert len(chain_ids) == len(set(chain_ids)), "chain_id must be unique"

    assert sum(1 for row in payload["chains"] if row["status"] == "dominant") == 1
    assert any(
        row["chain_id"] == payload["dominant_chain"] and row["status"] == "dominant"
        for row in payload["chains"]
    )

    assert all(0.0 <= float(row["intensity_score"]) <= 1.0 for row in payload["chains"])
    assert all(0.0 <= float(row["velocity_score"]) <= 1.0 for row in payload["chains"])
    assert all(0.0 <= float(row["confidence_score"]) <= 1.0 for row in payload["chains"])

    assert all(row["status"] in STATUS_ENUM for row in payload["chains"])
    assert all(row["origin"] in DOMINANT_CONFLICT_AXES for row in payload["chains"])


def test_financial_pressure_transmission_keeps_single_hot_chain() -> None:
    payload = build_event_transmission_chain_map(
        game_state_snapshot={"game_state": "financial_pressure"}
    )

    hot_chain_ids = [
        row["chain_id"] for row in payload["chains"] if row["status"] in {"active", "dominant"}
    ]
    assert hot_chain_ids == ["credit_intermediary_chain"]
