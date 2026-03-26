from lie_engine.research.event_crisis_analogies import build_default_archetypes


def test_analogy_library_contains_required_archetypes() -> None:
    archetypes = build_default_archetypes()
    archetype_ids = {entry["archetype_id"] for entry in archetypes}
    assert "gfc_2008" in archetype_ids
    assert "energy_credit_2014_2016" in archetype_ids
    assert "private_credit_redemption_2026" in archetype_ids
