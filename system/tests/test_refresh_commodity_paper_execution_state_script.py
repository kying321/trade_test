from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refresh_commodity_paper_execution_state.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("commodity_refresh_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step_now_is_monotonic() -> None:
    mod = _load_module()
    base = mod.parse_now("2026-03-11T08:40:00Z")
    assert mod.step_now(base, 0).isoformat() == "2026-03-11T08:40:00+00:00"
    assert mod.step_now(base, 3).isoformat() == "2026-03-11T08:40:03+00:00"
    assert mod.step_now(base, 5) > mod.step_now(base, 4)


def test_write_hot_brief_snapshot_persists_refresh_owned_copy(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    source_path = review_dir / "20260316T090525Z_hot_universe_operator_brief.json"
    source_text = json.dumps(
        {"status": "ok", "artifact": str(source_path), "operator_status": "ok"},
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    source_path.write_text(source_text, encoding="utf-8")

    snapshot_path = mod.write_hot_brief_snapshot(
        review_dir,
        stamp="20260316T090525Z",
        brief_payload={"artifact": str(source_path), "operator_status": "ok"},
    )

    assert snapshot_path.name == "20260316T090525Z_commodity_paper_execution_refresh_hot_brief_snapshot.json"
    assert snapshot_path.read_text(encoding="utf-8") == source_text
