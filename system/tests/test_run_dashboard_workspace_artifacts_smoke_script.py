from __future__ import annotations

import datetime as dt
import contextlib
import importlib.util
import json
import sys
from pathlib import Path
import pytest


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_workspace_artifacts_smoke.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("dashboard_workspace_artifacts_smoke_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_workspace_routes_smoke_spec_covers_all_workspace_sections(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "workspace-routes-smoke.png"
    result_path = tmp_path / "workspace-routes-smoke.json"

    spec = mod.build_workspace_routes_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
    )

    assert "#/overview" in spec
    assert "研究主线摘要" in spec
    assert "#/workspace/artifacts" in spec
    assert "工件池" in spec
    assert "#/workspace/alignment" in spec
    assert "对齐页" in spec
    assert "仅内部可见" in spec
    assert "回测池" in spec
    assert "回测主池" in spec
    assert "原始层" in spec
    assert "原始快照" in spec
    assert "契约层" in spec
    assert "公开入口拓扑" in spec
    assert "公开面验收" in spec
    assert "root overview 截图" in spec
    assert "pages overview 截图" in spec
    assert "root contracts 截图" in spec
    assert "pages contracts 截图" in spec
    assert "公开快照拉取次数" in spec
    assert "内部快照拉取次数" in spec
    assert "研究地图" in spec
    assert "#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow" in spec
    assert "intraday_orderflow_blueprint" in spec
    assert "intraday_orderflow_research_gate_blocker" in spec
    assert "expectStableMarker" in spec
    assert "toLowerCase().includes(text.toLowerCase())" in spec
    assert "new RegExp(escapeRegExp(marker), 'i')" in spec
    assert "context-nav" in spec
    assert "getByText(route.nav_label, { exact: true }).click()" in spec
    assert "internalSnapshotRequests.length" in spec
    assert "document.documentElement.dataset.theme" in spec
    assert "contracts-subcommand-workspace_routes_smoke" in spec
    assert "data-accordion-id=\"contracts-subcommand-workspace_routes_smoke\"" in spec
    assert str(result_path) in spec


def test_build_artifact_payload_reports_workspace_route_matrix(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "workspace-routes-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.42,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "public",
                "effective_surface": "public",
                "visited_routes": [
                    {"route": "#/overview", "headline": "总览"},
                    {"route": "#/workspace/artifacts", "headline": "工件目标池"},
                    {"route": "#/workspace/alignment", "headline": "方向对齐投射"},
                    {"route": "#/workspace/backtests", "headline": "回测主池"},
                    {"route": "#/workspace/raw", "headline": "原始快照"},
                    {"route": "#/workspace/contracts", "headline": "公开入口拓扑"},
                ],
                "snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=1",
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=2",
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=3",
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=4",
                    "http://127.0.0.1:4173/data/fenlie_dashboard_snapshot.json?ts=5",
                ],
                "internal_snapshot_requests": [],
                "theme_assertion": {
                    "route": "#/workspace/contracts?theme=light",
                    "requested_theme": "light",
                    "resolved_theme": "light",
                },
                "page_section_assertion": {
                    "route": "#/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke",
                    "page_section": "contracts-subcommand-workspace_routes_smoke",
                    "active_label": "工作区路由子命令",
                    "accordion_state": "open",
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
            },
        },
        base_url="http://127.0.0.1:4173/",
    )

    assert payload["action"] == "dashboard_workspace_routes_browser_smoke"
    assert payload["ok"] is True
    assert payload["change_class"] == "RESEARCH_ONLY"
    assert [row["route"] for row in payload["routes"]] == [
        "#/overview",
        "#/workspace/artifacts",
        "#/workspace/alignment",
        "#/workspace/backtests",
        "#/workspace/raw",
        "#/workspace/contracts",
    ]
    assert payload["network_observation"]["public_snapshot_fetch_count"] == 5
    assert payload["network_observation"]["internal_snapshot_fetch_count"] == 0
    assert payload["surface_assertion"]["requested_surface"] == "public"
    assert payload["surface_assertion"]["effective_surface"] == "public"
    assert payload["theme_assertion"] == {
        "route": "#/workspace/contracts?theme=light",
        "requested_theme": "light",
        "resolved_theme": "light",
    }
    assert payload["page_section_assertion"] == {
        "route": "#/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke",
        "page_section": "contracts-subcommand-workspace_routes_smoke",
        "active_label": "工作区路由子命令",
        "accordion_state": "open",
    }
    assert payload["artifacts_filter_assertion"] == {
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


def test_build_internal_alignment_smoke_spec_uses_internal_snapshot_projection_markers(tmp_path: Path) -> None:
    mod = load_module()
    screenshot_path = tmp_path / "internal-alignment-smoke.png"
    result_path = tmp_path / "internal-alignment-smoke.json"

    spec = mod.build_internal_alignment_smoke_spec(
        base_url="http://127.0.0.1:4173/",
        screenshot_path=screenshot_path,
        result_path=result_path,
        projection_summary_headline="将高价值对话反馈投射到内部对齐页",
        top_event_headline="拆分研究工作区左栏四控件",
        top_action="先刷新 internal snapshot 并验证对齐页，再收紧 internal fallback 语义。",
    )

    assert "#/workspace/alignment?view=internal&page_section=alignment-summary" in spec
    assert "方向对齐投射" in spec
    assert "将高价值对话反馈投射到内部对齐页" in spec
    assert "拆分研究工作区左栏四控件" in spec
    assert "先刷新 internal snapshot 并验证对齐页，再收紧 internal fallback 语义。" in spec
    assert "internalSnapshotRequests.length" in spec
    assert "publicSnapshotRequests.length" in spec
    assert "expect(publicSnapshotRequests.length).toBe(0)" in spec
    assert "requested_surface: 'internal'" in spec
    assert "effective_surface: 'internal'" in spec
    assert str(result_path) in spec


def test_build_artifact_payload_reports_internal_alignment_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "internal-alignment-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.24,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "internal",
                "effective_surface": "internal",
                "visited_routes": [
                    {
                        "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
                        "headline": "方向对齐投射",
                    }
                ],
                "snapshot_requests": [],
                "internal_snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_internal_snapshot.json?ts=1",
                ],
                "projection_assertion": {
                    "headline": "将高价值对话反馈投射到内部对齐页",
                    "top_event_headline": "拆分研究工作区左栏四控件",
                    "top_action": "先刷新 internal snapshot 并验证对齐页，再收紧 internal fallback 语义。",
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="internal_alignment",
        expected_route_markers=[
            {
                "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": [
                    "将高价值对话反馈投射到内部对齐页",
                    "拆分研究工作区左栏四控件",
                ],
            }
        ],
    )

    assert payload["action"] == "dashboard_internal_alignment_browser_smoke"
    assert payload["ok"] is True
    assert payload["surface_assertion"]["requested_surface"] == "internal"
    assert payload["surface_assertion"]["effective_surface"] == "internal"
    assert payload["surface_assertion"]["snapshot_endpoint_observed"] == "/data/fenlie_dashboard_internal_snapshot.json"
    assert payload["network_observation"]["public_snapshot_fetch_count"] == 0
    assert payload["network_observation"]["internal_snapshot_fetch_count"] == 1
    assert payload["projection_assertion"] == {
        "headline": "将高价值对话反馈投射到内部对齐页",
        "top_event_headline": "拆分研究工作区左栏四控件",
        "top_action": "先刷新 internal snapshot 并验证对齐页，再收紧 internal fallback 语义。",
    }
    assert payload["expected_route_markers"] == [
        {
            "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
            "nav_label": "对齐页",
            "headline": "方向对齐投射",
            "markers": [
                "将高价值对话反馈投射到内部对齐页",
                "拆分研究工作区左栏四控件",
            ],
        }
    ]


def test_build_artifact_payload_reports_internal_alignment_manual_probe_surface(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "internal-alignment-manual-probe-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.21,
        smoke_result={
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "internal",
                "effective_surface": "internal",
                "visited_routes": [
                    {
                        "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
                        "headline": "方向对齐投射",
                    }
                ],
                "snapshot_requests": [],
                "internal_snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_internal_snapshot.json?ts=1",
                ],
                "projection_assertion": {
                    "headline": "Manual probe headline",
                    "top_event_headline": "Manual probe headline",
                    "top_action": "Manual probe action",
                },
            },
        },
        base_url="http://127.0.0.1:4173/",
        mode="internal_alignment_manual_probe",
        expected_route_markers=[
            {
                "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": [
                    "Manual probe headline",
                    "Manual probe action",
                ],
            }
        ],
    )

    assert payload["action"] == "dashboard_internal_alignment_manual_probe_browser_smoke"
    assert payload["surface_assertion"]["requested_surface"] == "internal"
    assert payload["surface_assertion"]["effective_surface"] == "internal"
    assert payload["network_observation"]["public_snapshot_fetch_count"] == 0
    assert payload["network_observation"]["internal_snapshot_fetch_count"] == 1
    assert payload["projection_assertion"] == {
        "headline": "Manual probe headline",
        "top_event_headline": "Manual probe headline",
        "top_action": "Manual probe action",
    }


def test_build_artifact_payload_keeps_failure_assertion_for_manual_probe_failures(tmp_path: Path) -> None:
    mod = load_module()
    report_path = tmp_path / "report.json"
    screenshot_path = tmp_path / "internal-alignment-manual-probe-smoke.png"

    payload = mod.build_artifact_payload(
        workspace=tmp_path,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result={"returncode": 0},
        server_ready_seconds=0.0,
        smoke_result={
            "returncode": 1,
            "stdout": "",
            "stderr": "refresh_after_manual_probe_restore_failed",
            "playwright_result": {},
        },
        base_url="http://127.0.0.1:4173/",
        mode="internal_alignment_manual_probe",
        expected_route_markers=[
            {
                "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": [],
            }
        ],
        failure_assertion={
            "failure_stage": "refresh_after_manual_probe_restore_failed",
            "failure_detail": "refresh_after_manual_probe_restore_failed",
            "probe_feedback_id": mod.MANUAL_PROBE_FEEDBACK_ID,
            "manual_probe_state": {
                "manual_file_exists": False,
                "manual_row_count": 0,
                "manual_probe_present": False,
            },
        },
        force_failed=True,
    )

    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["failure_assertion"] == {
        "failure_stage": "refresh_after_manual_probe_restore_failed",
        "failure_detail": "refresh_after_manual_probe_restore_failed",
        "probe_feedback_id": mod.MANUAL_PROBE_FEEDBACK_ID,
        "manual_probe_state": {
            "manual_file_exists": False,
            "manual_row_count": 0,
            "manual_probe_present": False,
        },
    }


def test_temporary_manual_probe_restores_manual_jsonl_after_context(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    original_rows = [
        {
            "feedback_id": "existing_manual",
            "headline": "existing manual",
            "summary": "existing summary",
            "recommended_action": "existing action",
        }
    ]
    manual_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in original_rows) + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, list[str], str]] = []

    def fake_ensure_success(*, name: str, cmd: list[str], cwd: Path):
        calls.append((name, cmd, str(cwd)))
        return {"name": name, "returncode": 0}

    monkeypatch.setattr(mod, "ensure_success", fake_ensure_success)

    runtime_now = dt.datetime(2026, 3, 22, 7, 0, tzinfo=dt.timezone.utc)
    with mod.temporary_manual_probe(
        workspace=workspace,
        system_root=system_root,
        review_dir=review_dir,
        runtime_now=runtime_now,
    ) as probe_event:
        rows = [json.loads(line) for line in manual_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert [row["feedback_id"] for row in rows] == ["existing_manual", probe_event["feedback_id"]]
        assert probe_event["source"] == "manual"
        assert probe_event["created_at_utc"] == "2026-03-22T07:00:00Z"

    restored_rows = [json.loads(line) for line in manual_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["feedback_id"] for row in restored_rows] == ["existing_manual"]
    assert [name for name, _, _ in calls] == [
        "refresh_after_manual_probe_seed",
        "refresh_after_manual_probe_restore",
    ]


def test_temporary_manual_probe_restores_manual_jsonl_when_seed_refresh_fails(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    original_text = json.dumps(
        {
            "feedback_id": "existing_manual",
            "headline": "existing manual",
            "summary": "existing summary",
            "recommended_action": "existing action",
        },
        ensure_ascii=False,
    ) + "\n"
    manual_path.write_text(original_text, encoding="utf-8")

    def fake_ensure_success(*, name: str, cmd: list[str], cwd: Path):
        raise RuntimeError(f"{name}_failed")

    monkeypatch.setattr(mod, "ensure_success", fake_ensure_success)

    runtime_now = dt.datetime(2026, 3, 22, 7, 5, tzinfo=dt.timezone.utc)
    try:
        with mod.temporary_manual_probe(
            workspace=workspace,
            system_root=system_root,
            review_dir=review_dir,
            runtime_now=runtime_now,
        ):
            raise AssertionError("should_not_enter_context")
    except RuntimeError as exc:
        assert str(exc) == "refresh_after_manual_probe_seed_failed"

    assert manual_path.read_text(encoding="utf-8") == original_text


def test_main_writes_failure_artifact_when_manual_probe_restore_fails(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    web_root = system_root / "dashboard" / "web"
    dist_dir = web_root / "dist"
    review_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<!doctype html><title>ok</title>", encoding="utf-8")

    fixed_now = dt.datetime(2026, 3, 22, 7, 10, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(mod, "now_utc", lambda: fixed_now)
    monkeypatch.setattr(mod, "choose_port", lambda host: 4173)
    monkeypatch.setattr(mod, "wait_http_ready", lambda url, timeout_seconds: 0.12)
    monkeypatch.setattr(mod, "http_server", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        mod,
        "load_internal_alignment_expectations",
        lambda dist_dir: {
            "headline": "Manual probe headline",
            "top_event_headline": "Manual probe headline",
            "top_action": "Manual probe action",
            "route_assertions": [
                {
                    "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
                    "nav_label": "对齐页",
                    "headline": "方向对齐投射",
                    "markers": ["Manual probe headline", "Manual probe action"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        mod,
        "run_playwright_smoke",
        lambda **kwargs: {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "playwright_result": {
                "requested_surface": "internal",
                "effective_surface": "internal",
                "visited_routes": [
                    {
                        "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
                        "headline": "方向对齐投射",
                    }
                ],
                "snapshot_requests": [],
                "internal_snapshot_requests": [
                    "http://127.0.0.1:4173/data/fenlie_dashboard_internal_snapshot.json?ts=1",
                ],
                "projection_assertion": {
                    "headline": "Manual probe headline",
                    "top_event_headline": "Manual probe headline",
                    "top_action": "Manual probe action",
                },
            },
        },
    )

    @contextlib.contextmanager
    def broken_probe(**kwargs):
        yield {
            "feedback_id": mod.MANUAL_PROBE_FEEDBACK_ID,
            "created_at_utc": "2026-03-22T07:10:00Z",
            "source": "manual",
        }
        raise RuntimeError("refresh_after_manual_probe_restore_failed")

    monkeypatch.setattr(mod, "temporary_manual_probe", broken_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dashboard_workspace_artifacts_smoke.py",
            "--workspace",
            str(workspace),
            "--skip-build",
            "--mode",
            "internal_alignment_manual_probe",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1

    report_path = review_dir / "20260322T071000Z_dashboard_internal_alignment_manual_probe_browser_smoke.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["action"] == "dashboard_internal_alignment_manual_probe_browser_smoke"
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["failure_assertion"]["failure_stage"] == "refresh_after_manual_probe_restore_failed"
    assert payload["failure_assertion"]["manual_probe_state"] == {
        "manual_file_exists": False,
        "manual_row_count": 0,
        "manual_probe_present": False,
    }


def test_main_writes_failure_artifact_when_build_fails_before_smoke(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    web_root = system_root / "dashboard" / "web"
    review_dir.mkdir(parents=True, exist_ok=True)
    web_root.mkdir(parents=True, exist_ok=True)

    fixed_now = dt.datetime(2026, 3, 22, 7, 20, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(mod, "now_utc", lambda: fixed_now)

    def fake_ensure_success(*, name: str, cmd: list[str], cwd: Path):
        raise RuntimeError("dashboard_build_failed: mocked build failure")

    monkeypatch.setattr(mod, "ensure_success", fake_ensure_success)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dashboard_workspace_artifacts_smoke.py",
            "--workspace",
            str(workspace),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1

    report_path = review_dir / "20260322T072000Z_dashboard_workspace_routes_browser_smoke.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["action"] == "dashboard_workspace_routes_browser_smoke"
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["failure_assertion"] == {
        "failure_stage": "dashboard_build_failed",
        "failure_detail": "dashboard_build_failed: mocked build failure",
    }


def test_main_writes_failure_artifact_when_choose_port_fails(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    web_root = system_root / "dashboard" / "web"
    dist_dir = web_root / "dist"
    review_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "index.html").write_text("<!doctype html><title>ok</title>", encoding="utf-8")

    fixed_now = dt.datetime(2026, 3, 22, 7, 25, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(mod, "now_utc", lambda: fixed_now)
    monkeypatch.setattr(mod, "choose_port", lambda host: (_ for _ in ()).throw(RuntimeError("choose_port_failed: mocked port allocation failure")))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dashboard_workspace_artifacts_smoke.py",
            "--workspace",
            str(workspace),
            "--skip-build",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1

    report_path = review_dir / "20260322T072500Z_dashboard_workspace_routes_browser_smoke.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["action"] == "dashboard_workspace_routes_browser_smoke"
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["failure_assertion"] == {
        "failure_stage": "choose_port_failed",
        "failure_detail": "choose_port_failed: mocked port allocation failure",
    }
