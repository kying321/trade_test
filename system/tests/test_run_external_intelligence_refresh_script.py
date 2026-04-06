from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_external_intelligence_refresh.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_external_intelligence_refresh_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_runs_external_intelligence_refresh_chain(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    public_dir = system_root / "dashboard" / "web" / "public"
    review_dir.mkdir(parents=True, exist_ok=True)

    seen_cmds: dict[str, list[str]] = {}

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_cmds[name] = list(cmd)
        (public_dir / "data").mkdir(parents=True, exist_ok=True)
        if name == "run_jin10_mcp_snapshot":
            return {
                "status": "ok",
                "artifact_json": str(review_dir / "latest_jin10_mcp_snapshot.json"),
                "recommended_brief": "calendar=244 | flash=20 | quotes=2",
            }
        if name == "run_axios_site_snapshot":
            return {
                "status": "ok",
                "artifact_json": str(review_dir / "latest_axios_site_snapshot.json"),
                "recommended_brief": "axios news=10 | local=8 | national=2",
            }
        if name == "run_external_intelligence_snapshot":
            return {
                "status": "ok",
                "artifact_json": str(review_dir / "latest_external_intelligence_snapshot.json"),
                "recommended_brief": "sources=2 | calendar=244 | flash=20 | quotes=2 | news=10",
                "takeaway": "美国至4月3日当周EIA原油库存(万桶) ｜ Anthropic cuts third party usage",
            }
        if name == "build_dashboard_frontend_snapshot":
            public_snapshot = public_dir / "data" / "fenlie_dashboard_snapshot.json"
            internal_snapshot = public_dir / "data" / "fenlie_dashboard_internal_snapshot.json"
            public_snapshot.write_text(json.dumps({"surface": "public"}) + "\n", encoding="utf-8")
            internal_snapshot.write_text(json.dumps({"surface": "internal"}) + "\n", encoding="utf-8")
            return {
                "outputs": [
                    {
                        "surface": "public",
                        "path": str(public_snapshot),
                        "artifact_payload_count": 27,
                        "catalog_count": 80,
                        "backtest_artifact_count": 10,
                    },
                    {
                        "surface": "internal",
                        "path": str(internal_snapshot),
                        "artifact_payload_count": 27,
                        "catalog_count": 80,
                        "backtest_artifact_count": 10,
                    },
                ]
            }
        raise AssertionError(name)

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_external_intelligence_refresh.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-04-06T14:40:00Z",
            "--axios-limit",
            "12",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["mode"] == "external_intelligence_refresh"
    assert payload["change_class"] == "RESEARCH_ONLY"
    assert payload["status"] == "ok"
    assert payload["stamp"] == "20260406T144000Z"
    assert payload["jin10_status"] == "ok"
    assert payload["axios_status"] == "ok"
    assert payload["external_status"] == "ok"
    assert payload["recommended_brief"] == "sources=2 | calendar=244 | flash=20 | quotes=2 | news=10"
    assert payload["takeaway"] == "美国至4月3日当周EIA原油库存(万桶) ｜ Anthropic cuts third party usage"
    assert payload["external_intelligence_path"] == str(review_dir / "latest_external_intelligence_snapshot.json")
    assert payload["dashboard_outputs"] == [
        {
            "surface": "public",
            "path": str(public_dir / "data" / "fenlie_dashboard_snapshot.json"),
            "artifact_payload_count": 27,
            "catalog_count": 80,
            "backtest_artifact_count": 10,
        },
        {
            "surface": "internal",
            "path": str(public_dir / "data" / "fenlie_dashboard_internal_snapshot.json"),
            "artifact_payload_count": 27,
            "catalog_count": 80,
            "backtest_artifact_count": 10,
        },
    ]
    assert Path(payload["artifact_json"]).exists()
    assert Path(payload["artifact_md"]).exists()

    assert list(seen_cmds) == [
        "run_jin10_mcp_snapshot",
        "run_axios_site_snapshot",
        "run_external_intelligence_snapshot",
        "build_dashboard_frontend_snapshot",
    ]
    assert seen_cmds["run_jin10_mcp_snapshot"] == [
        sys.executable,
        str(system_root / "scripts" / "run_jin10_mcp_snapshot.py"),
        "--workspace",
        str(workspace),
        "--token-env",
        "JIN10_MCP_BEARER_TOKEN",
    ]
    assert seen_cmds["run_axios_site_snapshot"] == [
        sys.executable,
        str(system_root / "scripts" / "run_axios_site_snapshot.py"),
        "--workspace",
        str(workspace),
        "--limit",
        "12",
    ]
    assert seen_cmds["run_external_intelligence_snapshot"] == [
        sys.executable,
        str(system_root / "scripts" / "run_external_intelligence_snapshot.py"),
        "--workspace",
        str(workspace),
    ]
    assert seen_cmds["build_dashboard_frontend_snapshot"] == [
        sys.executable,
        str(system_root / "scripts" / "build_dashboard_frontend_snapshot.py"),
        "--workspace",
        str(workspace),
        "--public-dir",
        str(public_dir),
    ]
