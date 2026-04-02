from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_dashboard_public_acceptance.py"


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
    monkeypatch.setattr(mod, "current_python_executable", lambda: "/opt/miniconda3/bin/python3")
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
                    "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "source_available": True,
                    "active_artifact": "intraday_orderflow_blueprint",
                    "visible_artifacts": [
                        "intraday_orderflow_blueprint",
                        "intraday_orderflow_research_gate_blocker",
                    ],
                },
                "research_audit_search_assertion": {
                    "route": "/search",
                    "cases_available": True,
                    "cases": [
                        {
                            "case_id": "optimizer_trial_trade_journal",
                            "scope": "artifact",
                            "query": "trial_001_ultra_short_trade_journal",
                            "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_path": "system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv",
                        }
                    ],
                },
                "contracts_acceptance_inspector_assertion": {
                    "checks_by_id": {
                        "topology_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-check-topology_smoke",
                            "page_section": "contracts-check-topology_smoke",
                            "search_link_href": "",
                            "artifact_link_href": "",
                            "raw_link_href": "",
                        },
                        "workspace_routes_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                            "page_section": "contracts-check-workspace_routes_smoke",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
                        "graph_home_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                            "page_section": "contracts-check-graph_home_smoke",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        }
                    },
                    "subcommands_by_id": {
                        "topology_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-subcommand-topology_smoke",
                            "page_section": "contracts-subcommand-topology_smoke",
                            "check_route": "/workspace/contracts?page_section=contracts-check-topology_smoke",
                            "search_link_href": "",
                            "artifact_link_href": "",
                            "raw_link_href": "",
                        },
                        "workspace_routes_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke",
                            "page_section": "contracts-subcommand-workspace_routes_smoke",
                            "check_route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
                        "graph_home_smoke": {
                            "route": "/workspace/contracts?page_section=contracts-subcommand-graph_home_smoke",
                            "page_section": "contracts-subcommand-graph_home_smoke",
                            "check_route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        }
                    },
                },
            }
        elif name == "graph_home_smoke":
            payload = {
                "ok": True,
                "status": "ok",
                "change_class": "RESEARCH_ONLY",
                "report_path": "/tmp/graph-home.json",
                "graph_home_assertion": {
                    "default_route": "/",
                    "resolved_route": "/graph-home",
                    "research_audit_link_assertions": [
                        {
                            "selected_heading": "交易中枢",
                            "case_id": "optimizer_trial_trade_journal",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
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
        "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
        "group": "research_cross_section",
        "search_scope": "title",
        "search": "orderflow",
        "source_available": True,
        "active_artifact": "intraday_orderflow_blueprint",
        "visible_artifacts": [
            "intraday_orderflow_blueprint",
            "intraday_orderflow_research_gate_blocker",
        ],
    }
    assert result["checks"]["workspace_routes_smoke"]["research_audit_search_assertion"] == {
        "route": "/search",
        "cases_available": True,
        "cases": [
            {
                "case_id": "optimizer_trial_trade_journal",
                "scope": "artifact",
                "query": "trial_001_ultra_short_trade_journal",
                "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_path": "system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv",
            }
        ],
    }
    assert result["checks"]["workspace_routes_smoke"]["contracts_acceptance_inspector_assertion"] == {
        "checks_by_id": {
            "topology_smoke": {
                "route": "/workspace/contracts?page_section=contracts-check-topology_smoke",
                "page_section": "contracts-check-topology_smoke",
                "search_link_href": "",
                "artifact_link_href": "",
                "raw_link_href": "",
            },
            "workspace_routes_smoke": {
                "route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                "page_section": "contracts-check-workspace_routes_smoke",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            },
            "graph_home_smoke": {
                "route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                "page_section": "contracts-check-graph_home_smoke",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            }
        },
        "subcommands_by_id": {
            "topology_smoke": {
                "route": "/workspace/contracts?page_section=contracts-subcommand-topology_smoke",
                "page_section": "contracts-subcommand-topology_smoke",
                "check_route": "/workspace/contracts?page_section=contracts-check-topology_smoke",
                "search_link_href": "",
                "artifact_link_href": "",
                "raw_link_href": "",
            },
            "workspace_routes_smoke": {
                "route": "/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke",
                "page_section": "contracts-subcommand-workspace_routes_smoke",
                "check_route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            },
            "graph_home_smoke": {
                "route": "/workspace/contracts?page_section=contracts-subcommand-graph_home_smoke",
                "page_section": "contracts-subcommand-graph_home_smoke",
                "check_route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
            }
        },
    }
    assert result["checks"]["graph_home_smoke"]["graph_home_assertion"]["research_audit_link_assertions"] == [
        {
            "selected_heading": "交易中枢",
            "case_id": "optimizer_trial_trade_journal",
            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
        }
    ]
    assert seen_cmds["topology_smoke"][0] == "/opt/miniconda3/bin/python3"
    assert seen_cmds["workspace_routes_smoke"][0] == "/opt/miniconda3/bin/python3"
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
                "artifacts_filter_assertion": {
                    "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "source_available": False,
                    "active_artifact": "",
                    "visible_artifacts": [],
                },
                "research_audit_search_assertion": {
                    "route": "/search",
                    "cases_available": True,
                    "cases": [
                        {
                            "case_id": "optimizer_trial_trade_journal",
                            "query": "trial_001_ultra_short_trade_journal",
                            "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_path": "system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv",
                        },
                    ],
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
        if name == "graph_home_smoke":
            payload = {
                "ok": True,
                "status": "ok",
                "change_class": "RESEARCH_ONLY",
                "graph_home_assertion": {
                    "resolved_route": "/graph-home",
                    "research_audit_link_assertions": [
                        {
                            "selected_heading": "交易中枢",
                            "case_id": "optimizer_trial_trade_journal",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
                    ],
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


def test_run_acceptance_allows_orderflow_source_unavailable_degrade(monkeypatch, tmp_path: Path) -> None:
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
                "artifacts_filter_assertion": {
                    "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "source_available": False,
                    "active_artifact": "",
                    "visible_artifacts": [],
                },
                "research_audit_search_assertion": {
                    "route": "/search",
                    "cases_available": True,
                    "cases": [
                        {
                            "case_id": "optimizer_trial_trade_journal",
                            "query": "trial_001_ultra_short_trade_journal",
                            "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_path": "system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv",
                        },
                    ],
                },
            }
        elif name == "graph_home_smoke":
            payload = {
                **payload,
                "graph_home_assertion": {
                    "resolved_route": "/graph-home",
                    "research_audit_link_assertions": [
                        {
                            "selected_heading": "交易中枢",
                            "case_id": "optimizer_trial_trade_journal",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
                    ],
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

    assert result["ok"] is True
    workspace_check = result["checks"]["workspace_routes_smoke"]
    assert workspace_check["artifacts_filter_assertion"] == {
        "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
        "group": "research_cross_section",
        "search_scope": "title",
        "search": "orderflow",
        "source_available": False,
        "active_artifact": "",
        "visible_artifacts": [],
    }


def test_run_acceptance_requires_research_audit_search_assertion(monkeypatch, tmp_path: Path) -> None:
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
                "artifacts_filter_assertion": {
                    "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "source_available": True,
                    "active_artifact": "intraday_orderflow_blueprint",
                    "visible_artifacts": [
                        "intraday_orderflow_blueprint",
                        "intraday_orderflow_research_gate_blocker",
                    ],
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
    assert workspace_check["failure_reason"] == "missing_research_audit_search_assertion"


def test_run_acceptance_requires_graph_home_research_audit_link_assertions(monkeypatch, tmp_path: Path) -> None:
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
                "artifacts_filter_assertion": {
                    "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "source_available": True,
                    "active_artifact": "intraday_orderflow_blueprint",
                    "visible_artifacts": [
                        "intraday_orderflow_blueprint",
                        "intraday_orderflow_research_gate_blocker",
                    ],
                },
                "research_audit_search_assertion": {
                    "route": "/search",
                    "cases_available": True,
                    "cases": [
                        {
                            "case_id": "optimizer_trial_trade_journal",
                            "query": "trial_001_ultra_short_trade_journal",
                            "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_path": "system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv",
                        }
                    ],
                },
            }
        elif name == "graph_home_smoke":
            payload = {
                **payload,
                "graph_home_assertion": {
                    "default_route": "/",
                    "resolved_route": "/graph-home",
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
    graph_home_check = result["checks"]["graph_home_smoke"]
    assert graph_home_check["ok"] is False
    assert graph_home_check["status"] == "failed"
    assert graph_home_check["failure_reason"] == "missing_graph_home_research_audit_link_assertions"


def test_run_acceptance_allows_explicit_empty_research_audit_degrade(monkeypatch, tmp_path: Path) -> None:
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
                "artifacts_filter_assertion": {
                    "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "source_available": True,
                    "active_artifact": "intraday_orderflow_blueprint",
                    "visible_artifacts": [
                        "intraday_orderflow_blueprint",
                        "intraday_orderflow_research_gate_blocker",
                    ],
                },
                "research_audit_search_assertion": {
                    "route": "/search",
                    "cases_available": False,
                    "cases": [],
                },
            }
        elif name == "graph_home_smoke":
            payload = {
                **payload,
                "graph_home_assertion": {
                    "default_route": "/",
                    "resolved_route": "/graph-home",
                    "research_audit_link_assertions": [],
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

    assert result["ok"] is True
    assert result["checks"]["workspace_routes_smoke"]["research_audit_search_assertion"] == {
        "route": "/search",
        "cases_available": False,
        "cases": [],
    }
    assert result["checks"]["graph_home_smoke"]["graph_home_assertion"] == {
        "default_route": "/",
        "resolved_route": "/graph-home",
        "research_audit_link_assertions": [],
    }


def test_run_acceptance_requires_contracts_acceptance_inspector_assertion(monkeypatch, tmp_path: Path) -> None:
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
                "artifacts_filter_assertion": {
                    "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                    "group": "research_cross_section",
                    "search_scope": "title",
                    "search": "orderflow",
                    "source_available": True,
                    "active_artifact": "intraday_orderflow_blueprint",
                    "visible_artifacts": [
                        "intraday_orderflow_blueprint",
                        "intraday_orderflow_research_gate_blocker",
                    ],
                },
                "research_audit_search_assertion": {
                    "route": "/search",
                    "cases_available": True,
                    "cases": [
                        {
                            "case_id": "optimizer_trial_trade_journal",
                            "query": "trial_001_ultra_short_trade_journal",
                            "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_path": "system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv",
                        }
                    ],
                },
            }
        elif name == "graph_home_smoke":
            payload = {
                **payload,
                "graph_home_assertion": {
                    "default_route": "/",
                    "resolved_route": "/graph-home",
                    "research_audit_link_assertions": [
                        {
                            "selected_heading": "交易中枢",
                            "case_id": "optimizer_trial_trade_journal",
                            "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                            "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                            "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                        },
                    ],
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
    assert workspace_check["failure_reason"] == "missing_contracts_acceptance_inspector_assertion"


def test_validate_workspace_routes_payload_requires_topology_inspector_rows() -> None:
    mod = load_module()

    payload = {
        "artifacts_filter_assertion": {
            "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
            "group": "research_cross_section",
            "search_scope": "title",
            "search": "orderflow",
            "source_available": True,
            "active_artifact": "intraday_orderflow_blueprint",
            "visible_artifacts": [
                "intraday_orderflow_blueprint",
                "intraday_orderflow_research_gate_blocker",
            ],
        },
        "research_audit_search_assertion": {
            "route": "/search",
            "cases_available": True,
            "cases": [
                {
                    "case_id": "optimizer_trial_trade_journal",
                    "query": "trial_001_ultra_short_trade_journal",
                    "search_route": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                    "result_artifact": "audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                    "workspace_route": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                    "raw_path": "system/output/research/20260330_133956/ultra_short/trial_001_ultra_short_trade_journal.csv",
                }
            ],
        },
        "contracts_acceptance_inspector_assertion": {
            "checks_by_id": {
                "workspace_routes_smoke": {
                    "route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                    "page_section": "contracts-check-workspace_routes_smoke",
                    "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                    "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                    "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                },
                "graph_home_smoke": {
                    "route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                    "page_section": "contracts-check-graph_home_smoke",
                    "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                    "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                    "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                },
            },
            "subcommands_by_id": {
                "workspace_routes_smoke": {
                    "route": "/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke",
                    "page_section": "contracts-subcommand-workspace_routes_smoke",
                    "check_route": "/workspace/contracts?page_section=contracts-check-workspace_routes_smoke",
                    "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                    "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                    "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                },
                "graph_home_smoke": {
                    "route": "/workspace/contracts?page_section=contracts-subcommand-graph_home_smoke",
                    "page_section": "contracts-subcommand-graph_home_smoke",
                    "check_route": "/workspace/contracts?page_section=contracts-check-graph_home_smoke",
                    "search_link_href": "/search?q=trial_001_ultra_short_trade_journal&scope=artifact",
                    "artifact_link_href": "/workspace/artifacts?artifact=audit:recent_strategy_backtests:ultra_short:trial_001:trade_journal",
                    "raw_link_href": "/workspace/raw?artifact=system%2Foutput%2Fresearch%2F20260330_133956%2Fultra_short%2Ftrial_001_ultra_short_trade_journal.csv",
                },
            },
        },
    }

    assert mod.validate_workspace_routes_payload(payload) == "invalid_contracts_acceptance_inspector_assertion"
