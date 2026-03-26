from lie_engine.research.event_crisis_analogies import build_default_archetypes


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
