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


def test_load_event_rows_from_file_validates(tmp_path: Path) -> None:
    module = _load_module()
    payload = {"events": "not-a-list"}
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="events must be a list"):
        module.load_event_rows_from_file(path)
