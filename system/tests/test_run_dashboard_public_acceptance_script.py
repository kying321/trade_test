from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_acceptance.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("dashboard_public_acceptance_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_acceptance_aggregates_topology_and_workspace_results(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True)

    monkeypatch.setattr(mod, "resolve_system_root", lambda value: system_root)
    monkeypatch.setattr(mod, "now_utc", lambda: mod.dt.datetime(2026, 3, 20, 7, 15, tzinfo=mod.dt.timezone.utc))
    seen_cmds: dict[str, list[str]] = {}

    def fake_run_script(*, name: str, cmd: list[str], cwd: Path):
        seen_cmds[name] = list(cmd)
        if name == "topology_smoke":
            payload = {
                "ok": True,
                "status": "ok",
                "change_class": "RESEARCH_ONLY",
                "report_path": "/tmp/topology.json",
            }
        elif name == "workspace_routes_smoke":
            payload = {
                "ok": True,
                "status": "ok",
                "change_class": "RESEARCH_ONLY",
                "report_path": "/tmp/workspace-routes.json",
                "network_observation": {
                    "public_snapshot_fetch_count": 4,
                    "internal_snapshot_fetch_count": 0,
                },
                "artifacts_filter_assertion": {
                    "route": "#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "active_artifact": "intraday_orderflow_blueprint",
                    "visible_artifacts": [
                        "intraday_orderflow_blueprint",
                        "intraday_orderflow_research_gate_blocker",
                    ],
                },
            }
        else:
            raise AssertionError(name)
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": 0,
            "stdout": json.dumps(payload, ensure_ascii=False),
            "stderr": "",
            "payload": payload,
        }

    monkeypatch.setattr(mod, "run_json_script", fake_run_script)

    result = mod.run_acceptance(
        workspace=workspace,
        skip_workspace_build=True,
        workspace_timeout_seconds=45.0,
    )

    assert result["ok"] is True
    assert result["change_class"] == "RESEARCH_ONLY"
    assert result["checks"]["topology_smoke"]["ok"] is True
    assert result["checks"]["workspace_routes_smoke"]["network_observation"]["public_snapshot_fetch_count"] == 4
    assert result["checks"]["workspace_routes_smoke"]["network_observation"]["internal_snapshot_fetch_count"] == 0
    assert result["checks"]["workspace_routes_smoke"]["artifacts_filter_assertion"] == {
        "route": "#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
        "group": "research_cross_section",
        "search_scope": "title",
        "search": "orderflow",
        "active_artifact": "intraday_orderflow_blueprint",
        "visible_artifacts": [
            "intraday_orderflow_blueprint",
            "intraday_orderflow_research_gate_blocker",
        ],
    }
    assert "--allow-insecure-tls-fallback" in seen_cmds["topology_smoke"]


def test_main_writes_public_acceptance_report(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True)

    monkeypatch.setattr(mod, "resolve_system_root", lambda value: system_root)
    monkeypatch.setattr(mod, "now_utc", lambda: mod.dt.datetime(2026, 3, 20, 7, 16, tzinfo=mod.dt.timezone.utc))
    monkeypatch.setattr(
        mod,
        "run_acceptance",
        lambda **_: {
            "action": "dashboard_public_acceptance",
            "ok": True,
            "status": "ok",
            "change_class": "RESEARCH_ONLY",
            "generated_at_utc": "2026-03-20T07:16:00Z",
            "checks": {
                "topology_smoke": {"ok": True},
                "workspace_routes_smoke": {"ok": True},
            },
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dashboard_public_acceptance.py",
            "--workspace",
            str(workspace),
            "--skip-workspace-build",
        ],
    )

    mod.main()

    report_path = review_dir / "20260320T071600Z_dashboard_public_acceptance.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["checks"]["topology_smoke"]["ok"] is True
    assert payload["checks"]["workspace_routes_smoke"]["ok"] is True


def test_run_acceptance_preserves_failed_subcommand_audit_fields(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True)

    monkeypatch.setattr(mod, "resolve_system_root", lambda value: system_root)
    monkeypatch.setattr(mod, "now_utc", lambda: mod.dt.datetime(2026, 3, 20, 7, 17, tzinfo=mod.dt.timezone.utc))

    def fake_run_script(*, name: str, cmd: list[str], cwd: Path):
        if name == "topology_smoke":
            return {
                "name": name,
                "cmd": cmd,
                "cwd": str(cwd),
                "returncode": 1,
                "stdout": '{"action":"dashboard_public_topology_smoke","ok":false,"status":"failed"}\n',
                "stderr": "Traceback: timeout during root_public probe\n",
                "payload": {
                    "action": "dashboard_public_topology_smoke",
                    "ok": False,
                    "status": "failed",
                },
            }
        if name == "workspace_routes_smoke":
            payload = {
                "ok": True,
                "status": "ok",
                "change_class": "RESEARCH_ONLY",
            }
            return {
                "name": name,
                "cmd": cmd,
                "cwd": str(cwd),
                "returncode": 0,
                "stdout": json.dumps(payload, ensure_ascii=False),
                "stderr": "",
                "payload": payload,
            }
        raise AssertionError(name)

    monkeypatch.setattr(mod, "run_json_script", fake_run_script)

    result = mod.run_acceptance(
        workspace=workspace,
        skip_workspace_build=True,
        workspace_timeout_seconds=45.0,
    )

    assert result["ok"] is False
    topology_check = result["checks"]["topology_smoke"]
    assert topology_check["ok"] is False
    assert topology_check["returncode"] == 1
    assert topology_check["stdout"].startswith('{"action":"dashboard_public_topology_smoke"')
    assert "timeout during root_public probe" in topology_check["stderr"]


def test_run_acceptance_marks_missing_json_payload_as_failed(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True)

    monkeypatch.setattr(mod, "resolve_system_root", lambda value: system_root)

    def fake_run_script(*, name: str, cmd: list[str], cwd: Path):
        if name == "topology_smoke":
            return {
                "name": name,
                "cmd": cmd,
                "cwd": str(cwd),
                "returncode": 0,
                "stdout": "non-json health probe output\n",
                "stderr": "",
                "payload": None,
            }
        payload = {
            "ok": True,
            "status": "ok",
            "change_class": "RESEARCH_ONLY",
        }
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": 0,
            "stdout": json.dumps(payload, ensure_ascii=False),
            "stderr": "",
            "payload": payload,
        }

    monkeypatch.setattr(mod, "run_json_script", fake_run_script)

    result = mod.run_acceptance(
        workspace=workspace,
        skip_workspace_build=True,
        workspace_timeout_seconds=45.0,
    )

    assert result["ok"] is False
    topology_check = result["checks"]["topology_smoke"]
    assert topology_check["status"] == "failed"
    assert topology_check["failure_reason"] == "missing_json_payload"
    assert topology_check["stdout"] == "non-json health probe output\n"


def test_run_acceptance_requires_orderflow_artifacts_filter_assertion(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True)

    monkeypatch.setattr(mod, "resolve_system_root", lambda value: system_root)

    def fake_run_script(*, name: str, cmd: list[str], cwd: Path):
        payload = {
            "ok": True,
            "status": "ok",
            "change_class": "RESEARCH_ONLY",
        }
        if name == "workspace_routes_smoke":
            payload = {
                **payload,
                "network_observation": {
                    "public_snapshot_fetch_count": 5,
                    "internal_snapshot_fetch_count": 0,
                },
            }
        return {
            "name": name,
            "cmd": cmd,
            "cwd": str(cwd),
            "returncode": 0,
            "stdout": json.dumps(payload, ensure_ascii=False),
            "stderr": "",
            "payload": payload,
        }

    monkeypatch.setattr(mod, "run_json_script", fake_run_script)

    result = mod.run_acceptance(
        workspace=workspace,
        skip_workspace_build=True,
        workspace_timeout_seconds=45.0,
    )

    assert result["ok"] is False
    workspace_check = result["checks"]["workspace_routes_smoke"]
    assert workspace_check["ok"] is False
    assert workspace_check["status"] == "failed"
    assert workspace_check["failure_reason"] == "missing_orderflow_artifacts_filter_assertion"
