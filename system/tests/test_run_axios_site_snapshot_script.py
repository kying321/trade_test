from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_axios_site_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_axios_site_snapshot_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_snapshot_writes_public_research_artifact(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fetch_news_entries(self, limit: int = 20):
            return [
                {
                    "title": "Oil market braces for inventory shock",
                    "url": "https://www.axios.com/2026/04/06/oil-shock",
                    "published_at": "2026-04-06T13:15:05+00:00",
                    "keywords": ["Oil", "Markets"],
                    "is_local": False,
                },
                {
                    "title": "Phoenix growth slows",
                    "url": "https://www.axios.com/local/phoenix/2026/04/06/growth-slows",
                    "published_at": "2026-04-06T12:15:05+00:00",
                    "keywords": ["Phoenix", "Economy"],
                    "is_local": True,
                },
            ]

    monkeypatch.setattr(mod, "AxiosSiteClient", FakeClient)
    monkeypatch.setattr(mod, "now_utc", lambda: mod.dt.datetime(2026, 4, 6, 13, 20, tzinfo=mod.dt.timezone.utc))

    result = mod.run_snapshot(workspace=workspace, limit=20)

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["change_class"] == "RESEARCH_ONLY"
    assert result["mode"] == "axios_site_snapshot"
    assert result["summary"]["news_total"] == 2
    assert result["summary"]["local_total"] == 1
    assert result["takeaway"] == "Oil market braces for inventory shock"
    assert Path(result["artifact_json"]).exists()
    assert Path(result["artifact_md"]).exists()
    payload = json.loads(Path(result["artifact_json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["top_titles"][0] == "Oil market braces for inventory shock"
