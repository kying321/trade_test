from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_external_intelligence_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_external_intelligence_snapshot_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_snapshot_merges_jin10_and_axios_sidecars(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    (review_dir / "latest_jin10_mcp_snapshot.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "takeaway": "美国至4月3日当周EIA原油库存(万桶)",
                "recommended_brief": "calendar=244 | flash=20 | quotes=2",
                "summary": {
                    "calendar_total": 244,
                    "high_importance_count": 5,
                    "flash_total": 20,
                    "quote_watch": [
                        {"code": "XAUUSD", "name": "现货黄金"},
                        {"code": "USOIL", "name": "WTI原油"},
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_axios_site_snapshot.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "takeaway": "Anthropic cuts third party usage",
                "recommended_brief": "axios news=10 | local=8 | national=2",
                "summary": {
                    "news_total": 10,
                    "local_total": 8,
                    "national_total": 2,
                    "top_titles": ["Anthropic cuts third party usage"],
                    "top_keywords": ["OpenAI", "Anthropic"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    payload = mod.run_snapshot(workspace=workspace)

    assert payload["ok"] is True
    assert payload["status"] == "ok"
    assert payload["mode"] == "external_intelligence_snapshot"
    assert payload["summary"]["sources_total"] == 2
    assert payload["summary"]["calendar_total"] == 244
    assert payload["summary"]["axios_news_total"] == 10
    assert payload["summary"]["quote_watch"] == ["现货黄金", "WTI原油"]
    assert payload["takeaway"] == "美国至4月3日当周EIA原油库存(万桶) ｜ Anthropic cuts third party usage"
    assert Path(payload["artifact_json"]).exists()
    assert Path(payload["artifact_md"]).exists()


def test_run_snapshot_marks_blocked_jin10_as_inactive_source(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    (review_dir / "latest_jin10_mcp_snapshot.json").write_text(
        json.dumps(
            {
                "status": "blocked_auth_missing",
                "ok": False,
                "takeaway": "",
                "recommended_brief": "",
                "summary": {
                    "calendar_total": 0,
                    "high_importance_count": 0,
                    "flash_total": 0,
                    "quote_watch": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (review_dir / "latest_axios_site_snapshot.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "ok": True,
                "takeaway": "Gilbert's Heritage Park to open first phase in September",
                "recommended_brief": "axios news=10 | local=8 | national=2",
                "summary": {
                    "news_total": 10,
                    "local_total": 8,
                    "national_total": 2,
                    "top_titles": ["Gilbert's Heritage Park to open first phase in September"],
                    "top_keywords": ["Gilbert", "heritage park"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    payload = mod.run_snapshot(workspace=workspace)

    assert payload["ok"] is True
    assert payload["status"] == "partial"
    assert payload["summary"]["sources_total"] == 1
    assert payload["summary"]["active_sources"] == ["axios"]
    assert payload["recommended_brief"] == "sources=1 | calendar=0 | flash=0 | quotes=0 | news=10"
    assert payload["takeaway"] == "Gilbert's Heritage Park to open first phase in September"
