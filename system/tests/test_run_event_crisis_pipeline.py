from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_event_crisis_pipeline.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("event_crisis_pipeline_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_row() -> dict[str, object]:
    return {
        "event_id": "evt-1",
        "event_ts": "2026-03-25T12:00:00Z",
        "event_classes": ["credit_deterioration"],
    }


def test_run_event_crisis_pipeline_writes_latest_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    artifacts = module.run_pipeline(
        output_root=tmp_path,
        mode="snapshot",
        event_rows=[_fake_row()],
        market_inputs={"credit_liquidity_stress_score": 0.6},
        generated_at=dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc),
    )

    expected = [
        "latest_event_intake.json",
        "latest_event_regime_snapshot.json",
        "latest_event_crisis_analogy.json",
        "latest_event_asset_shock_map.json",
        "latest_event_crisis_operator_summary.json",
    ]

    for suffix in expected:
        path = tmp_path / "review" / suffix
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert path in artifacts.values()

    overlay_path = tmp_path / "state" / "event_live_guard_overlay.json"
    assert overlay_path.exists()
    overlay_payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    assert "event_state:sector_stress" in overlay_payload["override_reason_codes"]
    assert overlay_payload["risk_multiplier_override"] <= 0.9
    assert overlay_payload["dominant_chain"] == "credit_intermediary_chain"
    assert 0.0 <= float(overlay_payload["system_margin_score"]) <= 1.0
    assert overlay_path == artifacts["overlay"]


def test_run_event_crisis_pipeline_writes_geostrategy_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    artifacts = module.run_pipeline(
        output_root=tmp_path,
        mode="snapshot",
        event_rows=[_fake_row()],
        market_inputs={"credit_liquidity_stress_score": 0.6},
        generated_at=dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc),
    )

    assert module.ARTIFACT_ORDER == [
        "latest_event_intake.json",
        "latest_event_game_state_snapshot.json",
        "latest_event_transmission_chain_map.json",
        "latest_event_regime_snapshot.json",
        "latest_event_crisis_analogy.json",
        "latest_event_asset_shock_map.json",
        "latest_event_safety_margin_snapshot.json",
        "event_live_guard_overlay.json",
        "latest_event_crisis_operator_summary.json",
    ]

    expected = {
        "game_state": "latest_event_game_state_snapshot.json",
        "transmission": "latest_event_transmission_chain_map.json",
        "safety_margin": "latest_event_safety_margin_snapshot.json",
    }

    for key, filename in expected.items():
        path = tmp_path / "review" / filename
        assert path.exists()
        assert path == artifacts[key]

    operator_summary = json.loads(artifacts["operator_summary"].read_text(encoding="utf-8"))
    assert operator_summary["event_crisis_primary_theater_brief"] == "usd_liquidity_and_sanctions"
    assert operator_summary["event_crisis_dominant_chain_brief"] == "credit_intermediary_chain"
    assert "system_margin=" in operator_summary["event_crisis_safety_margin_brief"]
    assert operator_summary["event_crisis_hard_boundary_brief"] in {
        "shadow_only_boundary",
        "new_risk_hard_block",
        "canary_hard_block",
        "none",
    }
    for removed_key in (
        "event_crisis_regime_brief",
        "event_crisis_top_analogue_brief",
        "event_crisis_watch_assets_brief",
        "event_crisis_guard_brief",
    ):
        assert removed_key not in operator_summary


def test_run_event_crisis_pipeline_overlay_reflects_safety_margin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    monkeypatch.setattr(
        module,
        "build_event_safety_margin_snapshot",
        lambda **_: {
            "system_margin_score": 0.24,
            "hard_boundaries": {
                "canary_hard_block": True,
                "new_risk_hard_block": True,
                "shadow_only_boundary": False,
            },
        },
    )
    monkeypatch.setattr(
        module,
        "build_event_transmission_chain_map",
        lambda **_: {"dominant_chain": "risk_off_deleveraging_chain", "chains": []},
    )

    artifacts = module.run_pipeline(
        output_root=tmp_path,
        mode="snapshot",
        event_rows=[_fake_row()],
        generated_at=dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc),
    )

    overlay_payload = json.loads(artifacts["overlay"].read_text(encoding="utf-8"))
    assert overlay_payload["canary_freeze"] is True
    assert overlay_payload["risk_multiplier_override"] == 0.24
    assert "event_boundary:canary_hard_block" in overlay_payload["override_reason_codes"]
    assert "event_chain:risk_off_deleveraging_chain" in overlay_payload["override_reason_codes"]


def test_run_event_crisis_pipeline_empty_input_does_not_freeze_overlay(tmp_path: Path) -> None:
    module = _load_module()
    artifacts = module.run_pipeline(
        output_root=tmp_path,
        mode="snapshot",
        event_rows=[],
        market_inputs={},
        generated_at=dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone.utc),
    )

    game_state_payload = json.loads(artifacts["game_state"].read_text(encoding="utf-8"))
    safety_margin_payload = json.loads(artifacts["safety_margin"].read_text(encoding="utf-8"))
    overlay_payload = json.loads(artifacts["overlay"].read_text(encoding="utf-8"))
    assert game_state_payload["game_state"] == "stable_competition"
    assert overlay_payload["canary_freeze"] is False
    assert overlay_payload["risk_multiplier_override"] >= 0.6
    assert overlay_payload["hard_boundaries"]["new_risk_hard_block"] is False
    assert overlay_payload["hard_boundaries"]["shadow_only_boundary"] is False
    assert safety_margin_payload["boundary_reasons"] == []
    assert overlay_payload["override_reason_codes"][0] == "event_state:watch"


def test_run_pipeline_normalizes_naive_datetime(tmp_path: Path) -> None:
    module = _load_module()
    naive = dt.datetime(2026, 3, 25, 12, 0)
    artifacts = module.run_pipeline(
        output_root=tmp_path,
        mode="snapshot",
        event_rows=[_fake_row()],
        generated_at=naive,
    )
    intake = json.loads(artifacts["intake"].read_text(encoding="utf-8"))
    assert intake["generated_at_utc"].endswith("Z")


def test_run_pipeline_handles_aware_datetime(tmp_path: Path) -> None:
    module = _load_module()
    aware = dt.datetime(2026, 3, 25, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    artifacts = module.run_pipeline(
        output_root=tmp_path,
        mode="snapshot",
        event_rows=[_fake_row()],
        generated_at=aware,
    )
    intake = json.loads(artifacts["intake"].read_text(encoding="utf-8"))
    assert intake["generated_at_utc"].endswith("Z")


def test_load_event_rows_from_file_validates(tmp_path: Path) -> None:
    module = _load_module()
    payload = {"events": "not-a-list"}
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="events must be a list"):
        module.load_event_rows_from_file(path)
