from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_domestic_commodity_reasoning.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("domestic_commodity_reasoning_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_domestic_commodity_reasoning_writes_all_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    artifacts = module.run_pipeline(
        output_root=tmp_path,
        contract_focus="BU2606",
        event_artifacts=[],
        research_artifacts=[],
        cross_section_news=[],
        cross_section_data=[],
        generated_at="2026-03-27T14:40:00Z",
    )
    expected = {
        "scenario_tree": "latest_commodity_reasoning_scenario_tree.json",
        "transmission_map": "latest_commodity_reasoning_transmission_map.json",
        "boundary_strength": "latest_commodity_reasoning_boundary_strength.json",
        "summary": "latest_commodity_reasoning_summary.json",
    }
    for key, filename in expected.items():
        path = tmp_path / "review" / filename
        assert path.exists()
        assert artifacts[key] == path
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
