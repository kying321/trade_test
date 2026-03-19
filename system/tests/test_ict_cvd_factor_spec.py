from __future__ import annotations

from pathlib import Path

import yaml


def test_ict_cvd_factor_spec_contains_expected_factors() -> None:
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "ict_cvd_factor_spec.yaml"
    )
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    assert payload["registry"] == "fenlie_ict_cvd_factor_spec"
    factors = payload["factors"]
    factor_ids = {item["id"] for item in factors}
    assert {
        "CVD_SWEEP_DIVERGENCE",
        "CVD_DISPLACEMENT_CONFIRMATION",
        "CVD_ABSORPTION_REVERSAL",
        "CVD_FAILED_AUCTION_PROXY",
        "CVD_EFFORT_RESULT_DIVERGENCE",
    }.issubset(factor_ids)
    assert payload["integration"]["authority"] == "confirm_and_veto_only"
    assert payload["integration"]["strict_cvd_supported"] is False
    semantic_layers = payload["semantic_layers"]
    assert semantic_layers["cvd_context_mode"]["values"][0]["id"] == "continuation"
    trust_ids = {item["id"] for item in semantic_layers["cvd_trust_tier"]["values"]}
    assert "cross_exchange_confirmed" in trust_ids
    veto_ids = {item["id"] for item in semantic_layers["cvd_veto_reasons"]["values"]}
    assert "effort_result_divergence" in veto_ids
    sweep = next(item for item in factors if item["id"] == "CVD_SWEEP_DIVERGENCE")
    assert sweep["default_context_mode"] == "reversal"
    assert sweep["trust_tier_required_min"] == "single_exchange_ok"
