from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_hot_universe_research.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("run_hot_universe_research_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latest_review_artifact_ignores_future_stamped_file(tmp_path: Path) -> None:
    mod = _load_script_module()
    mod.now_utc = lambda: dt.datetime(2026, 3, 10, 15, 58, tzinfo=dt.timezone.utc)
    current = tmp_path / "20260310T155630Z_crypto_route_operator_brief.json"
    future = tmp_path / "20260310T234630Z_crypto_route_operator_brief.json"
    current.write_text("{}", encoding="utf-8")
    future.write_text("{}", encoding="utf-8")
    path = mod.latest_review_artifact(tmp_path, "crypto_route_operator_brief")
    assert path == current
