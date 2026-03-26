from lie_engine.research.event_crisis_analogies import (
    build_default_archetypes,
    build_top_analogues,
)


def test_analogy_library_contains_required_archetypes() -> None:
    archetypes = build_default_archetypes()
    archetype_ids = {entry["archetype_id"] for entry in archetypes}
    required = {
        "gfc_2008",
        "energy_credit_2014_2016",
        "private_credit_redemption_2026",
        "oil_shock_1973_1979",
        "ltcm_russia_1998",
        "eurozone_sovereign_2011",
        "covid_liquidity_2020",
        "europe_energy_2022",
        "regional_banks_2023",
    }
    assert required <= archetype_ids


def test_archetype_library_returns_independent_objects() -> None:
    first_batch = build_default_archetypes()
    first_batch[0]["archetype_id"] = "mutated"
    second_batch = build_default_archetypes()
    assert second_batch[0]["archetype_id"] == "gfc_2008"


def test_top_analogues_reports_conflict_and_penalizes_mismatch_axes() -> None:
    with_conflict = build_top_analogues(
        ["energy_supply_shock", "bank_intermediary_chain_risk"], max_results=20
    )
    without_conflict = build_top_analogues(["energy_supply_shock"], max_results=20)

    oil_with_conflict = next(
        (entry for entry in with_conflict if entry["archetype_id"] == "oil_shock_1973_1979"), None
    )
    oil_without_conflict = next(
        (entry for entry in without_conflict if entry["archetype_id"] == "oil_shock_1973_1979"), None
    )

    assert oil_with_conflict is not None
    assert oil_without_conflict is not None
    assert "bank_intermediary_chain_risk" in oil_with_conflict["mismatch_axes"]
    assert oil_with_conflict["similarity_score"] <= oil_without_conflict["similarity_score"]
