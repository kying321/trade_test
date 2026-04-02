#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

PUBLIC_WORKSPACE_ROUTE_ASSERTIONS = [
    {
        "route": "/ops/overview",
        "nav_label": "总览",
        "headline": "总览",
        "markers": ["关键摘要", "系统运行", "调度心跳", "研究主线", "退出风控", "下一步去哪"],
    },
    {
        "route": "/workspace/artifacts",
        "nav_label": "工件池",
        "headline": "工件目标池",
        "markers": ["研究地图"],
    },
    {
        "route": "/workspace/alignment",
        "nav_label": "对齐页",
        "headline": "方向对齐投射",
        "markers": ["仅内部可见"],
    },
    {
        "route": "/workspace/backtests",
        "nav_label": "回测池",
        "headline": "回测主池",
        "markers": ["穿透层 1 / 回测池", "穿透层 2 / 近期比较行"],
    },
    {
        "route": "/workspace/raw",
        "nav_label": "原始层",
        "headline": "原始快照",
        "markers": ["告警定向原始层", "操作面板"],
    },
    {
        "route": "/workspace/contracts",
        "nav_label": "契约层",
        "headline": "公开入口拓扑",
        "markers": [
            "公开面验收",
            "穿透层 1 / 验收总览",
            "接口目录",
            "源头主线",
            "回退链",
        ],
    },
]
CHANGE_CLASS = "RESEARCH_ONLY"
MANUAL_PROBE_FEEDBACK_ID = "manual_alignment_browser_smoke_probe"
CONTRACTS_SOURCE_HEAD_ASSERTION = {
    "route": "/workspace/contracts?page_section=contracts-source-head-operator_panel",
    "page_section": "contracts-source-head-operator_panel",
    "source_head_id": "operator_panel",
    "visible_markers": [
        "状态",
        "研究结论",
        "生成时间",
        "路径",
    ],
}
CONTRACTS_SOURCE_GAP_ASSERTION = {
    "route": "/workspace/contracts?page_section=contracts-fallback",
    "page_section": "contracts-fallback",
    "visible_markers": [
        "/workspace/raw",
        "/data/fenlie_dashboard_snapshot.json",
        "/operator_task_visual_panel.html",
    ],
}
CONTRACTS_ACCEPTANCE_INSPECTOR_CHECK_ROUTE = "/workspace/contracts?page_section=contracts-check-graph_home_smoke"
CONTRACTS_ACCEPTANCE_INSPECTOR_SUBCOMMAND_ROUTE = "/workspace/contracts?page_section=contracts-subcommand-graph_home_smoke"
ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION = {
    "route": "/workspace/artifacts?artifact=price_action_exit_risk_break_even_review_conclusion",
    "group": "research_exit_risk",
    "search_scope": "title",
    "search": "",
    "section_label": "支撑证据",
    "active_artifact": "price_action_exit_risk_break_even_review_conclusion",
    "visible_artifacts": [
        "price_action_exit_risk_break_even_guarded_review",
        "price_action_exit_risk_break_even_review_packet",
        "price_action_exit_risk_break_even_review_conclusion",
        "price_action_exit_risk_break_even_primary_anchor_review",
        "price_action_exit_risk_hold_selection_aligned_break_even_review_lane",
    ],
    "visible_markers": [
        "price_action_exit_risk_break_even_guarded_review",
        "price_action_exit_risk_break_even_review_packet",
        "price_action_exit_risk_break_even_review_conclusion",
        "price_action_exit_risk_break_even_primary_anchor_review",
        "price_action_exit_risk_hold_selection_aligned_break_even_review_lane",
    ],
}
TERMINAL_SIGNAL_RISK_ROUTE = "/terminal/internal?panel=signal-risk&section=focus-slots"
TERMINAL_SIGNAL_RISK_TITLE = "信号发生器与风险节流阀"
TERMINAL_FOCUS_SLOTS_TITLE = "穿透层 3 / 焦点槽位"
TERMINAL_FOCUS_SLOT_LABELS = {
    "primary": "主槽位",
    "followup": "跟进槽位",
    "secondary": "次级槽位",
}
COMMODITY_VISIBILITY_OVERVIEW_ROUTE = "/ops/overview"
COMMODITY_VISIBILITY_TERMINAL_ROUTE = "/ops/risk"
COMMODITY_REASONING_LINE_TITLE = "国内商品推理线"
GRAPH_HOME_ROUTE_ASSERTIONS = [
    {
        "route": "/graph-home",
        "nav_label": "图谱主页",
        "headline": "图谱化主页",
        "markers": [
            "默认管道",
            "推荐下一跳",
            "回到交易中枢",
            "去操作终端",
            "去研究工作区",
            "打开全局搜索",
        ],
    }
]


def load_graph_home_route_assertions(*, dist_dir: Path) -> list[dict[str, Any]]:
    route_assertions = [
        {
            **route,
            "markers": list(route.get("markers") or []),
        }
        for route in GRAPH_HOME_ROUTE_ASSERTIONS
    ]
    snapshot_path = dist_dir / "data" / "fenlie_dashboard_snapshot.json"
    if not snapshot_path.exists():
        return route_assertions
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return route_assertions
    route_assertions[0]["research_audit_search_cases"] = derive_research_audit_search_cases(snapshot)
    return route_assertions


def commodity_visibility_markers_from_snapshot(snapshot: dict[str, Any]) -> list[str]:
    commodity_summary_payload = (
        ((snapshot.get("artifact_payloads") or {}).get("commodity_reasoning_summary") or {}).get("payload") or {}
    )
    markers = [
        COMMODITY_REASONING_LINE_TITLE,
        str(commodity_summary_payload.get("primary_scenario_brief") or "").strip(),
        str(commodity_summary_payload.get("primary_chain_brief") or "").strip(),
        str(((commodity_summary_payload.get("contracts_in_focus") or [None])[0]) or "").strip(),
    ]
    return [marker for marker in markers if marker]


def _path_stem(path_value: Any) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    return Path(raw).stem


def derive_research_audit_search_cases(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    payloads = snapshot.get("artifact_payloads") or {}
    cases: list[dict[str, str]] = []

    recent_backtests = ((payloads.get("recent_strategy_backtests") or {}).get("payload") or {})
    for mode_summary in list(recent_backtests.get("mode_summaries") or []):
        mode = str((mode_summary or {}).get("mode") or "").strip()
        for artifact in list((mode_summary or {}).get("trial_artifacts") or []):
            trade_path = str((artifact or {}).get("trade_journal_path") or "").strip()
            if not trade_path:
                continue
            mode_name = str((artifact or {}).get("mode") or mode or "").strip()
            trial_raw = str((artifact or {}).get("trial") or "").strip()
            trial_label = f"trial_{trial_raw.zfill(3)}" if trial_raw else "trial"
            cases.append(
                {
                    "case_id": "optimizer_trial_trade_journal",
                    "scope": "artifact",
                    "query": _path_stem(trade_path),
                    "search_route": f"/search?q={_path_stem(trade_path)}&scope=artifact",
                    "result_artifact": f"audit:recent_strategy_backtests:{mode_name}:{trial_label}:trade_journal",
                    "result_label": f"{mode_name} / {trial_label} / trade_journal",
                    "workspace_route": f"/workspace/artifacts?artifact=audit:recent_strategy_backtests:{mode_name}:{trial_label}:trade_journal",
                    "raw_path": trade_path,
                }
            )
            break
        if cases:
            break

    strategy_lab = ((payloads.get("strategy_lab_summary") or {}).get("payload") or {})
    strategy_candidates = list(strategy_lab.get("candidates") or [])
    strategy_best = strategy_lab.get("best_candidate") or {}
    best_trade_path = str((strategy_best or {}).get("trade_journal_path") or "").strip()
    best_name = str((strategy_best or {}).get("name") or "best_candidate").strip()
    strategy_case: dict[str, str] | None = None
    fallback_candidate_case: dict[str, str] | None = None
    for index, candidate in enumerate(strategy_candidates, start=1):
        trade_path = str((candidate or {}).get("trade_journal_path") or "").strip()
        if not trade_path:
            continue
        name = str((candidate or {}).get("name") or f"candidate_{index}").strip()
        candidate_case = {
            "case_id": "strategy_lab_candidate_trade_journal",
            "scope": "artifact",
            "query": _path_stem(trade_path),
            "search_route": f"/search?q={_path_stem(trade_path)}&scope=artifact",
            "result_artifact": f"audit:strategy_lab_summary:{name}:trade_journal",
            "result_label": f"{name} / trade_journal",
            "workspace_route": f"/workspace/artifacts?artifact=audit:strategy_lab_summary:{name}:trade_journal",
            "raw_path": trade_path,
        }
        if fallback_candidate_case is None:
            fallback_candidate_case = candidate_case
        if trade_path != best_trade_path or name != best_name:
            strategy_case = candidate_case
            break
    if strategy_case is None and fallback_candidate_case is not None:
        strategy_case = fallback_candidate_case
    if strategy_case is None:
        trade_path = str((strategy_best or {}).get("trade_journal_path") or "").strip()
        if trade_path:
            strategy_case = {
                "case_id": "strategy_lab_best_trade_journal",
                "scope": "artifact",
                "query": _path_stem(trade_path),
                "search_route": f"/search?q={_path_stem(trade_path)}&scope=artifact",
                "result_artifact": "audit:strategy_lab_summary:best_candidate:trade_journal",
                "result_label": f"{best_name} / trade_journal",
                "workspace_route": "/workspace/artifacts?artifact=audit:strategy_lab_summary:best_candidate:trade_journal",
                "raw_path": trade_path,
            }
    if strategy_case:
        cases.append(strategy_case)

    return [case for case in cases if case.get("query") and case.get("result_artifact") and case.get("raw_path")]


def load_public_workspace_route_assertions(*, dist_dir: Path) -> list[dict[str, Any]]:
    route_assertions = [
        {
            **route,
            "markers": list(route.get("markers") or []),
        }
        for route in PUBLIC_WORKSPACE_ROUTE_ASSERTIONS
    ]
    snapshot_path = dist_dir / "data" / "fenlie_dashboard_snapshot.json"
    if not snapshot_path.exists():
        return route_assertions
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return route_assertions

    def resolve_artifact_label(artifact_id: str) -> str:
        normalized = str(artifact_id or "").strip().lower()
        if not normalized:
            return ""
        for row in list(snapshot.get("catalog") or []):
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("id") or "").strip().lower()
            row_payload_key = str(row.get("payload_key") or "").strip().lower()
            if normalized in {row_id, row_payload_key}:
                label = str(row.get("label") or "").strip()
                if label:
                    return label
        return artifact_id

    default_focus = dict(snapshot.get("workspace_default_focus") or {})
    hold_selection_payload = (
        ((snapshot.get("artifact_payloads") or {}).get("hold_selection_handoff") or {}).get("payload") or {}
    )
    active_baseline = str(hold_selection_payload.get("active_baseline") or "").strip()
    commodity_markers = commodity_visibility_markers_from_snapshot(snapshot)
    if active_baseline:
        route_assertions[0]["markers"] = [
            "系统状态 / 入口 / 路由总览",
            "研究主线摘要",
            "查看源头主线",
            "契约验收",
            active_baseline,
            *commodity_markers,
        ]
    elif commodity_markers:
        route_assertions[0]["markers"] = [
            "系统状态 / 入口 / 路由总览",
            "研究主线摘要",
            "查看源头主线",
            "契约验收",
            *commodity_markers,
        ]
    if len(route_assertions) > 1:
        catalog_rows = list(snapshot.get("catalog") or [])
        catalog_refs = {
            str(value).strip().lower()
            for row in catalog_rows
            for value in [row.get("id"), row.get("payload_key"), row.get("label"), row.get("path")]
            if str(value).strip()
        }
        orderflow_visible_artifacts = [
            artifact
            for artifact in ["intraday_orderflow_blueprint", "intraday_orderflow_research_gate_blocker"]
            if artifact.lower() in catalog_refs
        ]
        exit_risk_review_visible_artifacts = [
            artifact
            for artifact in ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["visible_artifacts"]
            if artifact.lower() in catalog_refs
        ]
        route_assertions[1]["expected_default_artifact"] = str(default_focus.get("artifact") or "price_action_breakout_pullback")
        route_assertions[1]["expected_focus_panel"] = str(default_focus.get("panel") or "lab-review")
        route_assertions[1]["expected_focus_section"] = str(default_focus.get("section") or "research-heads")
        route_assertions[1]["orderflow_filter_route"] = "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow"
        route_assertions[1]["orderflow_filter_visible_artifacts"] = orderflow_visible_artifacts
        route_assertions[1]["orderflow_filter_active_artifact"] = orderflow_visible_artifacts[0] if orderflow_visible_artifacts else ""
        route_assertions[1]["exit_risk_review_route"] = ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["route"]
        route_assertions[1]["exit_risk_review_visible_artifacts"] = exit_risk_review_visible_artifacts
        route_assertions[1]["exit_risk_review_active_artifact"] = exit_risk_review_visible_artifacts[0] if exit_risk_review_visible_artifacts else ""
        route_assertions[1]["research_audit_search_cases"] = derive_research_audit_search_cases(snapshot)
    if len(route_assertions) > 4:
        raw_focus_marker = resolve_artifact_label(str(default_focus.get("artifact") or "")) or "操作面板"
        route_assertions[4]["markers"] = ["告警定向原始层", raw_focus_marker]
    return route_assertions


def load_commodity_visibility_route_assertions(*, dist_dir: Path) -> list[dict[str, Any]]:
    snapshot_path = dist_dir / "data" / "fenlie_dashboard_snapshot.json"
    if not snapshot_path.exists():
        markers = [COMMODITY_REASONING_LINE_TITLE]
    else:
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            snapshot = {}
        markers = commodity_visibility_markers_from_snapshot(snapshot) or [COMMODITY_REASONING_LINE_TITLE]
    return [
        {
            "route": COMMODITY_VISIBILITY_OVERVIEW_ROUTE,
            "nav_label": "总览",
            "headline": "总览",
            "markers": markers,
        },
        {
            "route": COMMODITY_VISIBILITY_TERMINAL_ROUTE,
            "nav_label": "操作终端",
            "headline": "Observe / Diagnose / Act",
            "markers": markers,
        },
    ]


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def current_python_executable() -> str:
    return sys.executable or "python3"


def build_manual_probe_event(*, runtime_now: dt.datetime) -> dict[str, Any]:
    return {
        "feedback_id": MANUAL_PROBE_FEEDBACK_ID,
        "created_at_utc": fmt_utc(runtime_now),
        "source": "manual",
        "domain": "research",
        "headline": "Manual probe headline",
        "summary": "验证 manual JSONL 事件能经 projection 进入 internal alignment 页面。",
        "recommended_action": "Manual probe action",
        "alignment_delta": 16,
        "blocker_delta": 2,
        "execution_delta": 12,
        "readability_delta": 1,
        "impact_score": 99,
        "confidence": 0.99,
        "status": "active",
        "anchors": [
            {
                "route": "/workspace/alignment?view=internal",
                "artifact": "latest_conversation_feedback_projection_internal",
                "component": "AlignmentWorkspace",
            }
        ],
    }


def resolve_system_root(workspace: Path) -> Path:
    if (workspace / "system").exists():
        return workspace / "system"
    if workspace.name == "system":
        return workspace
    raise FileNotFoundError(f"cannot resolve system root from {workspace}")


def ensure_success(*, name: str, cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    payload = {
        "name": name,
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"returncode={proc.returncode}"
        raise RuntimeError(f"{name}_failed: {detail}")
    return payload


def wait_http_ready(url: str, *, timeout_seconds: float) -> float:
    started = time.monotonic()
    last_error = ""
    while time.monotonic() - started <= timeout_seconds:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if 200 <= int(response.status) < 500:
                    return round(time.monotonic() - started, 3)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.2)
    raise TimeoutError(f"http_ready_timeout: {url} last_error={last_error}")


def choose_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def temporary_manual_probe(
    *,
    workspace: Path,
    system_root: Path,
    review_dir: Path,
    runtime_now: dt.datetime,
) -> Iterator[dict[str, Any]]:
    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    manual_path.parent.mkdir(parents=True, exist_ok=True)
    original_text = manual_path.read_text(encoding="utf-8") if manual_path.exists() else None
    probe_event = build_manual_probe_event(runtime_now=runtime_now)
    with manual_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(probe_event, ensure_ascii=False) + "\n")
    primary_error: Exception | None = None
    try:
        ensure_success(
            name="refresh_after_manual_probe_seed",
            cmd=[
                current_python_executable(),
                str(system_root / "scripts" / "run_operator_panel_refresh.py"),
                "--workspace",
                str(workspace),
                "--now",
                fmt_utc(runtime_now),
            ],
            cwd=system_root,
        )
        yield probe_event
    except Exception as exc:  # noqa: BLE001
        primary_error = exc
        raise
    finally:
        if original_text is None:
            manual_path.unlink(missing_ok=True)
        else:
            manual_path.write_text(original_text, encoding="utf-8")
        try:
            ensure_success(
                name="refresh_after_manual_probe_restore",
                cmd=[
                    current_python_executable(),
                    str(system_root / "scripts" / "run_operator_panel_refresh.py"),
                    "--workspace",
                    str(workspace),
                    "--now",
                    fmt_utc(runtime_now),
                ],
                cwd=system_root,
            )
        except Exception:  # noqa: BLE001
            if original_text is not None:
                manual_path.write_text(original_text, encoding="utf-8")
            else:
                manual_path.unlink(missing_ok=True)
            if primary_error is None:
                raise


def inspect_manual_probe_state(*, review_dir: Path) -> dict[str, Any]:
    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    if not manual_path.exists():
        return {
            "manual_file_exists": False,
            "manual_row_count": 0,
            "manual_probe_present": False,
        }
    rows = [
        json.loads(line)
        for line in manual_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "manual_file_exists": True,
        "manual_row_count": len(rows),
        "manual_probe_present": any(str(row.get("feedback_id") or "") == MANUAL_PROBE_FEEDBACK_ID for row in rows),
    }


@contextlib.contextmanager
def http_server(*, dist_dir: Path, host: str, port: int) -> Iterator[subprocess.Popen[str]]:
    server_script = Path(__file__).with_name("serve_spa_fallback.py")
    cmd = [
        current_python_executable(),
        str(server_script),
        str(port),
        host,
        str(dist_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=dist_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def build_workspace_routes_smoke_spec(
    *,
    base_url: str,
    screenshot_path: Path,
    result_path: Path,
    route_assertions: list[dict[str, Any]] | None = None,
) -> str:
    route_matrix_json = json.dumps(route_assertions or PUBLIC_WORKSPACE_ROUTE_ASSERTIONS, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTES = {route_matrix_json};
            const BASE_URL = {base_url!r}.replace(/\\/$/, '');

            function escapeRegExp(value) {{
              return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
            }}

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
            }}

            function currentRoute() {{
                  return `${{window.location.pathname}}${{window.location.search}}`;
            }}

            async function waitForRoute(page, expectedPathname, expectedParams = {{}}) {{
              await page.waitForFunction(({{ expectedPathname, expectedParams }}) => {{
                const pathname = window.location.pathname;
                if (pathname !== expectedPathname) return false;
                const params = new URLSearchParams(window.location.search);
                return Object.entries(expectedParams).every(([key, value]) => params.get(key) === value);
              }}, {{ expectedPathname, expectedParams }});
            }}

            async function ensureSidebarPageSectionsVisible(page) {{
              const visibleNavSelector = 'nav[aria-label="page-sections-nav"]:visible';
              const pageSectionsNav = page.locator(visibleNavSelector).first();
              const navVisible = (await pageSectionsNav.count()) > 0;
              if (!navVisible) {{
                const expandToggle = page.getByRole('button', {{ name: '展开侧边导航' }});
                if (await expandToggle.isVisible().catch(() => false)) {{
                  await expandToggle.click();
                  await page.getByRole('button', {{ name: '收起侧边导航' }}).waitFor();
                }}
              }}
              await expect(page.locator(visibleNavSelector).first()).toBeVisible();
            }}

            async function ensureInspectorOpen(page) {{
              const inspectorPanel = page.locator('.inspector-rail-inner').first();
              if (await inspectorPanel.isVisible().catch(() => false)) {{
                return inspectorPanel;
              }}
              const expandToggle = page.getByRole('button', {{ name: '展开对象检查器' }}).first();
              if (await expandToggle.isVisible().catch(() => false)) {{
                await expandToggle.click();
              }}
              await expect(inspectorPanel).toBeVisible();
              return inspectorPanel;
            }}

            function artifactFromHref(href) {{
              const raw = String(href || '');
              const [, query = ''] = raw.split('?');
              return new URLSearchParams(query).get('artifact');
            }}

            async function clickContextNav(page, label) {{
              const nav = page.getByRole('navigation', {{ name: 'context-nav' }});
              const link = nav.getByRole('link', {{ name: new RegExp(`^${{escapeRegExp(label)}}(?:\\\\s|$)`) }}).first();
              const linkVisible = await link.isVisible().catch(() => false);
              if (!linkVisible) {{
                const expandToggle = page.getByRole('button', {{ name: '展开侧边导航' }});
                if (await expandToggle.isVisible().catch(() => false)) {{
                  await expandToggle.click();
                  await page.getByRole('button', {{ name: '收起侧边导航' }}).waitFor();
                }}
              }}
              await expect(link).toBeVisible();
              await link.click();
            }}

            test.use({{ browserName: 'chromium' }});

            test('workspace routes smoke', async ({{ page }}) => {{
              const snapshotRequests = [];
              const internalSnapshotRequests = [];
              const visitedRoutes = [];
              const overviewRoute = ROUTES[0];
              const workspaceStartRoute = ROUTES[1];
              const defaultWorkspaceArtifact = String(workspaceStartRoute.expected_default_artifact || 'price_action_breakout_pullback');
              const defaultWorkspacePanel = String(workspaceStartRoute.expected_focus_panel || 'lab-review');
              const defaultWorkspaceSection = String(workspaceStartRoute.expected_focus_section || 'research-heads');
              const artifactsFilterRoute = String(workspaceStartRoute.orderflow_filter_route || '/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow');
              const orderflowArtifacts = Array.isArray(workspaceStartRoute.orderflow_filter_visible_artifacts)
                ? workspaceStartRoute.orderflow_filter_visible_artifacts
                : ['intraday_orderflow_blueprint', 'intraday_orderflow_research_gate_blocker'];
              const orderflowActiveArtifact = Object.prototype.hasOwnProperty.call(workspaceStartRoute, 'orderflow_filter_active_artifact')
                ? String(workspaceStartRoute.orderflow_filter_active_artifact || '')
                : 'intraday_orderflow_blueprint';
              const exitRiskReviewRoute = String(workspaceStartRoute.exit_risk_review_route || {ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["route"]!r});
              const exitRiskReviewArtifacts = Array.isArray(workspaceStartRoute.exit_risk_review_visible_artifacts)
                ? workspaceStartRoute.exit_risk_review_visible_artifacts
                : {json.dumps(ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["visible_artifacts"], ensure_ascii=False, indent=2)};
              const exitRiskReviewSectionHint = {ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["section_label"]!r};
              const exitRiskReviewActiveArtifact = Object.prototype.hasOwnProperty.call(workspaceStartRoute, 'exit_risk_review_active_artifact')
                ? String(workspaceStartRoute.exit_risk_review_active_artifact || '')
                : {ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["active_artifact"]!r};
              const researchAuditCases = Array.isArray(workspaceStartRoute.research_audit_search_cases)
                ? workspaceStartRoute.research_audit_search_cases
                : [];
              let exitRiskReviewSectionLabel = exitRiskReviewSectionHint;
              let exitRiskReviewVisibleArtifacts = [];
              const researchAuditSearchAssertions = [];
              const themeRoute = '/workspace/contracts?theme=light';
              const pageSectionRoute = '/workspace/contracts?page_section=contracts-acceptance-subcommands';
              const contractsSourceHeadRoute = {CONTRACTS_SOURCE_HEAD_ASSERTION["route"]!r};
              const contractsSourceHeadMarkers = {json.dumps(CONTRACTS_SOURCE_HEAD_ASSERTION["visible_markers"], ensure_ascii=False, indent=2)};
              const contractsSourceGapRoute = {CONTRACTS_SOURCE_GAP_ASSERTION["route"]!r};
              const contractsSourceGapMarkers = {json.dumps(CONTRACTS_SOURCE_GAP_ASSERTION["visible_markers"], ensure_ascii=False, indent=2)};
              const contractsInspectorCheckRoute = {CONTRACTS_ACCEPTANCE_INSPECTOR_CHECK_ROUTE!r};
              const contractsInspectorSubcommandRoute = {CONTRACTS_ACCEPTANCE_INSPECTOR_SUBCOMMAND_ROUTE!r};
              const contractsSourceGapPayloadKey = 'finding_count';

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) snapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto(BASE_URL + overviewRoute.route, {{ waitUntil: 'networkidle' }});
              for (const marker of overviewRoute.markers) {{
                await expectStableMarker(page, marker);
              }}
              visitedRoutes.push({{
                route: overviewRoute.route,
                headline: overviewRoute.headline,
                url: page.url(),
              }});

              await page.goto(BASE_URL + workspaceStartRoute.route, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction((artifactId) => `${{window.location.pathname}}${{window.location.search}}`.includes(`artifact=${{artifactId}}`), defaultWorkspaceArtifact);
              await page.waitForFunction((panelId) => `${{window.location.pathname}}${{window.location.search}}`.includes(`panel=${{panelId}}`), defaultWorkspacePanel);
              await page.waitForFunction((sectionId) => `${{window.location.pathname}}${{window.location.search}}`.includes(`section=${{sectionId}}`), defaultWorkspaceSection);
              await expect(page).toHaveURL(new RegExp(`artifact=${{escapeRegExp(defaultWorkspaceArtifact)}}`));
              await expect(page).toHaveURL(new RegExp(`panel=${{escapeRegExp(defaultWorkspacePanel)}}`));
              await expect(page).toHaveURL(new RegExp(`section=${{escapeRegExp(defaultWorkspaceSection)}}`));

              const activeArtifactValue = page.locator('.artifact-button.active .value-text').first();
              await expect(activeArtifactValue).toHaveAttribute('title', defaultWorkspaceArtifact);
              await expect(page.getByRole('heading', {{ name: workspaceStartRoute.headline, exact: true }})).toBeVisible();
              for (const marker of workspaceStartRoute.markers) {{
                await expectStableMarker(page, marker);
              }}
              visitedRoutes.push({{
                route: workspaceStartRoute.route,
                headline: workspaceStartRoute.headline,
                url: page.url(),
              }});

              if (orderflowArtifacts.length) {{
                await page.goto(BASE_URL + artifactsFilterRoute, {{ waitUntil: 'networkidle' }});
                await page.waitForFunction((fragment) => `${{window.location.pathname}}${{window.location.search}}`.includes(fragment), artifactsFilterRoute);
                await page.waitForFunction((artifactId) => `${{window.location.pathname}}${{window.location.search}}`.includes(`artifact=${{artifactId}}`), orderflowActiveArtifact);
                for (const artifactId of orderflowArtifacts) {{
                  await expectStableMarker(page, artifactId);
                }}
                const activeOrderflowArtifact = page.locator('.artifact-layer-cross .artifact-button.active .value-text').first();
                await expect(activeOrderflowArtifact).toHaveAttribute('title', orderflowActiveArtifact);
              }}

              if (exitRiskReviewArtifacts.length && exitRiskReviewActiveArtifact) {{
                await page.goto(BASE_URL + exitRiskReviewRoute, {{ waitUntil: 'networkidle' }});
                await page.waitForFunction((fragment) => `${{window.location.pathname}}${{window.location.search}}`.includes(fragment), exitRiskReviewRoute);
                await page.waitForFunction((artifactId) => `${{window.location.pathname}}${{window.location.search}}`.includes(`artifact=${{artifactId}}`), exitRiskReviewActiveArtifact);
                await expectStableMarker(page, exitRiskReviewActiveArtifact);
                const exitRiskReviewSectionState = await page.evaluate((sectionHint) => {{
                  const normalizedSectionHint = String(sectionHint || '').replace(/\\s+/g, '').toLowerCase();
                  const sections = Array.from(document.querySelectorAll('.artifact-layer-section'));
                  const target = sections.find((node) => {{
                    const summaryText = String(node.querySelector('summary')?.textContent || '').replace(/\\s+/g, '').toLowerCase();
                    return summaryText.includes(normalizedSectionHint);
                  }});
                  if (!target) return {{ opened: false, label: '' }};
                  if (target instanceof HTMLDetailsElement) {{
                    target.open = true;
                  }}
                  return {{
                    opened: true,
                    label: String(target.querySelector('summary')?.textContent || '').trim(),
                  }};
                }}, exitRiskReviewSectionHint);
                expect(exitRiskReviewSectionState.opened).toBeTruthy();
                await expectStableMarker(page, exitRiskReviewSectionHint);
                exitRiskReviewSectionLabel = exitRiskReviewSectionState.label || exitRiskReviewSectionHint;
                exitRiskReviewVisibleArtifacts = await page.evaluate(({{ sectionHint, artifactIds }}) => {{
                  const normalizedSectionHint = String(sectionHint || '').replace(/\\s+/g, '').toLowerCase();
                  const target = Array.from(document.querySelectorAll('.artifact-layer-section')).find((node) => {{
                    const summaryText = String(node.querySelector('summary')?.textContent || '').replace(/\\s+/g, '').toLowerCase();
                    return summaryText.includes(normalizedSectionHint);
                  }});
                  if (!target) return [];
                  const rawTitles = Array.from(target.querySelectorAll('.value-text[title]'))
                    .map((node) => String(node.getAttribute('title') || '').trim().toLowerCase())
                    .filter(Boolean);
                  return artifactIds.filter((artifactId) => rawTitles.includes(String(artifactId || '').trim().toLowerCase()));
                }}, {{ sectionHint: exitRiskReviewSectionHint, artifactIds: exitRiskReviewArtifacts }});
                expect(exitRiskReviewVisibleArtifacts).toEqual(exitRiskReviewArtifacts);
                const activeExitRiskReviewArtifact = page.locator('.artifact-layer-exit .artifact-button.active .value-text').first();
                await expect(activeExitRiskReviewArtifact).toHaveAttribute('title', exitRiskReviewActiveArtifact);
              }}

              for (const auditCase of researchAuditCases) {{
                const searchRoute = String(auditCase.search_route || '/search');
                const auditScope = String(auditCase.scope || 'artifact');
                const auditQuery = String(auditCase.query || '');
                const auditResultArtifact = String(auditCase.result_artifact || '');
                const auditResultLabel = String(auditCase.result_label || auditQuery || auditResultArtifact);
                const auditRawPath = String(auditCase.raw_path || '');
                await page.goto(BASE_URL + searchRoute, {{ waitUntil: 'networkidle' }});
                    await waitForRoute(page, '/search', {{ q: auditQuery, scope: auditScope }});
                await expect(page.locator('.global-search-input')).toHaveValue(auditQuery);
                await expectStableMarker(page, auditRawPath);
                const searchResultButton = page.getByRole('button', {{ name: new RegExp(escapeRegExp(auditRawPath)) }}).first();
                await expect(searchResultButton).toBeVisible();
                await searchResultButton.click();
                    await waitForRoute(page, '/workspace/artifacts', {{ artifact: auditResultArtifact }});
                await expectStableMarker(page, auditResultArtifact);
                await expectStableMarker(page, auditRawPath);
                const rawLink = page.locator(`a[href*="${{encodeURIComponent(auditRawPath)}}"]`).first();
                await expect(rawLink).toHaveAttribute('href', new RegExp(`/workspace/raw\\\\?artifact=${{escapeRegExp(encodeURIComponent(auditRawPath))}}`));
                await rawLink.click();
                    await waitForRoute(page, '/workspace/raw', {{ artifact: auditRawPath }});
                await expectStableMarker(page, auditRawPath);
                researchAuditSearchAssertions.push({{
                  case_id: String(auditCase.case_id || ''),
                  scope: auditScope,
                  query: auditQuery,
                  search_route: auditCase.search_route,
                  result_artifact: auditResultArtifact,
                  result_label: auditResultLabel,
                  workspace_route: auditCase.workspace_route,
                  raw_path: auditCase.raw_path,
                }});
              }}

              for (const route of ROUTES.slice(2)) {{
                await page.goto(BASE_URL + route.route, {{ waitUntil: 'networkidle' }});
                    await page.waitForFunction((fragment) => `${{window.location.pathname}}${{window.location.search}}`.includes(fragment), route.route);
                await expect(page.getByRole('heading', {{ name: route.headline, exact: true }})).toBeVisible();
                for (const marker of route.markers) {{
                  await expectStableMarker(page, marker);
                }}
                visitedRoutes.push({{
                  route: route.route,
                  headline: route.headline,
                  url: page.url(),
                }});
              }}

              await page.goto(BASE_URL + themeRoute, {{ waitUntil: 'networkidle' }});
                  await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('theme=light'));
              await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
              await expect(page.getByRole('button', {{ name: '白天主题' }})).toHaveAttribute('aria-pressed', 'true');
              const contractsPageSectionHrefs = await page.evaluate(() => Array.from(document.querySelectorAll('nav[aria-label="page-sections-nav"] .page-section-link')).map((node) => String(node.getAttribute('href') || '')));
              let pageSectionActiveLabel = '';
              let pageSectionAccordionState = '';
              if (contractsPageSectionHrefs.some((href) => href.includes('page_section=contracts-acceptance-subcommands'))) {{
                await page.goto(BASE_URL + pageSectionRoute, {{ waitUntil: 'networkidle' }});
                    await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-acceptance-subcommands'));
                const activePageSection = page.locator('nav[aria-label="page-sections-nav"]').first().locator('.page-section-link.active').first();
                await expect(activePageSection).toContainText('子命令证据');
                pageSectionActiveLabel = ((await activePageSection.textContent()) || '').trim();
              }}

              let contractsSourceHeadAccordionState = '';
              let contractsSourceHeadObservedMarkers = [];
              if (contractsPageSectionHrefs.some((href) => href.includes('page_section=contracts-source-head-operator_panel'))) {{
                await page.goto(BASE_URL + contractsSourceHeadRoute, {{ waitUntil: 'networkidle' }});
                    await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-source-head-operator_panel'));
                const contractsSourceHeadAccordion = page.locator('[data-accordion-id="contracts-source-head-operator_panel"]').first();
                await expect(contractsSourceHeadAccordion).toHaveAttribute('data-state', 'open');
                for (const marker of contractsSourceHeadMarkers) {{
                  await expect(contractsSourceHeadAccordion).toContainText(marker);
                }}
                contractsSourceHeadAccordionState = await contractsSourceHeadAccordion.getAttribute('data-state') || '';
                contractsSourceHeadObservedMarkers = contractsSourceHeadMarkers;
              }}

              let contractsSourceGapObservedMarkers = [];
              if (contractsPageSectionHrefs.some((href) => href.includes('page_section=contracts-fallback'))) {{
                await page.goto(BASE_URL + contractsSourceGapRoute, {{ waitUntil: 'networkidle' }});
                    await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-fallback'));
                const contractsSourceGapSection = page.locator('[data-workspace-section="contracts-fallback"]').first();
                await expect(contractsSourceGapSection).toBeVisible();
                for (const marker of contractsSourceGapMarkers) {{
                  await expect(contractsSourceGapSection).toContainText(marker);
                }}
                contractsSourceGapObservedMarkers = contractsSourceGapMarkers;
              }}

              let contractsAcceptanceInspectorAssertion = {{
                checks_by_id: {{
                  topology_smoke: {{
                    route: '/workspace/contracts?page_section=contracts-check-topology_smoke',
                    page_section: 'contracts-check-topology_smoke',
                    search_link_href: '',
                    artifact_link_href: '',
                    raw_link_href: '',
                  }},
                  workspace_routes_smoke: {{
                    route: '/workspace/contracts?page_section=contracts-check-workspace_routes_smoke',
                    page_section: 'contracts-check-workspace_routes_smoke',
                    search_link_href: '',
                    artifact_link_href: '',
                    raw_link_href: '',
                  }},
                  graph_home_smoke: {{
                    route: contractsInspectorCheckRoute,
                    page_section: 'contracts-check-graph_home_smoke',
                    search_link_href: '',
                    artifact_link_href: '',
                    raw_link_href: '',
                  }},
                }},
                subcommands_by_id: {{
                  topology_smoke: {{
                    route: '/workspace/contracts?page_section=contracts-subcommand-topology_smoke',
                    page_section: 'contracts-subcommand-topology_smoke',
                    check_route: '/workspace/contracts?page_section=contracts-check-topology_smoke',
                    search_link_href: '',
                    artifact_link_href: '',
                    raw_link_href: '',
                  }},
                  workspace_routes_smoke: {{
                    route: '/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke',
                    page_section: 'contracts-subcommand-workspace_routes_smoke',
                    check_route: '/workspace/contracts?page_section=contracts-check-workspace_routes_smoke',
                    search_link_href: '',
                    artifact_link_href: '',
                    raw_link_href: '',
                  }},
                  graph_home_smoke: {{
                    route: contractsInspectorSubcommandRoute,
                    page_section: 'contracts-subcommand-graph_home_smoke',
                    check_route: contractsInspectorCheckRoute,
                    search_link_href: '',
                    artifact_link_href: '',
                    raw_link_href: '',
                  }},
                }},
              }};
              if (researchAuditCases.length && contractsPageSectionHrefs.some((href) => href.includes('page_section=contracts-check-graph_home_smoke'))) {{
                const firstResearchAuditCase = researchAuditCases[0];
                await page.goto(BASE_URL + '/workspace/contracts', {{ waitUntil: 'networkidle' }});
                const statusStrip = page.locator('[data-testid="contracts-acceptance-status-strip"]').first();
                await expect(statusStrip).toBeVisible();
                const topologyStatusButton = statusStrip.getByRole('button', {{ name: /入口拓扑烟测/ }}).first();
                await expect(topologyStatusButton).toBeVisible();
                await topologyStatusButton.click();
                await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-check-topology_smoke'));
                let inspectorPanel = await ensureInspectorOpen(page);
                await expect(inspectorPanel).toContainText('当前验收项');

                const topologySubcommandLink = statusStrip.getByRole('link', {{ name: '入口拓扑烟测 / 查看子命令' }}).first();
                await expect(topologySubcommandLink).toBeVisible();
                await topologySubcommandLink.click();
                await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-subcommand-topology_smoke'));
                inspectorPanel = await ensureInspectorOpen(page);
                await expect(inspectorPanel).toContainText('当前子命令');

                await page.goto(BASE_URL + '/workspace/contracts', {{ waitUntil: 'networkidle' }});
                await expect(statusStrip).toBeVisible();
                const workspaceRoutesStatusButton = statusStrip.getByRole('button', {{ name: /工作区五页面烟测/ }}).first();
                await expect(workspaceRoutesStatusButton).toBeVisible();
                await workspaceRoutesStatusButton.click();
                await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-check-workspace_routes_smoke'));
                inspectorPanel = await ensureInspectorOpen(page);
                await expect(inspectorPanel).toContainText('当前验收项');
                const workspaceCheckSearchLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计检索' }}).first();
                const workspaceCheckArtifactLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计工件' }}).first();
                const workspaceCheckRawLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计原始层' }}).first();
                await expect(workspaceCheckSearchLink).toHaveAttribute('href', firstResearchAuditCase.search_route);
                expect(artifactFromHref(await workspaceCheckArtifactLink.getAttribute('href'))).toBe(firstResearchAuditCase.result_artifact);
                await expect(workspaceCheckRawLink).toHaveAttribute('href', new RegExp(`/workspace/raw\\\\?artifact=${{escapeRegExp(encodeURIComponent(firstResearchAuditCase.raw_path))}}`));
                contractsAcceptanceInspectorAssertion.checks_by_id.workspace_routes_smoke.search_link_href = String(await workspaceCheckSearchLink.getAttribute('href') || '');
                contractsAcceptanceInspectorAssertion.checks_by_id.workspace_routes_smoke.artifact_link_href = String(await workspaceCheckArtifactLink.getAttribute('href') || '');
                contractsAcceptanceInspectorAssertion.checks_by_id.workspace_routes_smoke.raw_link_href = String(await workspaceCheckRawLink.getAttribute('href') || '');

                const workspaceRoutesSubcommandLink = statusStrip.getByRole('link', {{ name: '工作区五页面烟测 / 查看子命令' }}).first();
                await expect(workspaceRoutesSubcommandLink).toBeVisible();
                await workspaceRoutesSubcommandLink.click();
                await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-subcommand-workspace_routes_smoke'));
                inspectorPanel = await ensureInspectorOpen(page);
                await expect(inspectorPanel).toContainText('当前子命令');
                const workspaceSubcommandSearchLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计检索' }}).first();
                const workspaceSubcommandArtifactLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计工件' }}).first();
                const workspaceSubcommandRawLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计原始层' }}).first();
                await expect(workspaceSubcommandSearchLink).toHaveAttribute('href', firstResearchAuditCase.search_route);
                expect(artifactFromHref(await workspaceSubcommandArtifactLink.getAttribute('href'))).toBe(firstResearchAuditCase.result_artifact);
                await expect(workspaceSubcommandRawLink).toHaveAttribute('href', new RegExp(`/workspace/raw\\\\?artifact=${{escapeRegExp(encodeURIComponent(firstResearchAuditCase.raw_path))}}`));
                contractsAcceptanceInspectorAssertion.subcommands_by_id.workspace_routes_smoke.search_link_href = String(await workspaceSubcommandSearchLink.getAttribute('href') || '');
                contractsAcceptanceInspectorAssertion.subcommands_by_id.workspace_routes_smoke.artifact_link_href = String(await workspaceSubcommandArtifactLink.getAttribute('href') || '');
                contractsAcceptanceInspectorAssertion.subcommands_by_id.workspace_routes_smoke.raw_link_href = String(await workspaceSubcommandRawLink.getAttribute('href') || '');

                const graphHomeStatusButton = statusStrip.getByRole('button', {{ name: /图谱主页烟测/ }}).first();
                await expect(graphHomeStatusButton).toBeVisible();
                await graphHomeStatusButton.click();
                await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-check-graph_home_smoke'));
                inspectorPanel = await ensureInspectorOpen(page);
                await expect(inspectorPanel).toContainText('当前验收项');
                const inspectorSearchLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计检索' }}).first();
                const inspectorArtifactLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计工件' }}).first();
                const inspectorRawLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计原始层' }}).first();
                await expect(inspectorSearchLink).toHaveAttribute('href', firstResearchAuditCase.search_route);
                expect(artifactFromHref(await inspectorArtifactLink.getAttribute('href'))).toBe(firstResearchAuditCase.result_artifact);
                await expect(inspectorRawLink).toHaveAttribute('href', new RegExp(`/workspace/raw\\\\?artifact=${{escapeRegExp(encodeURIComponent(firstResearchAuditCase.raw_path))}}`));
                contractsAcceptanceInspectorAssertion.checks_by_id.graph_home_smoke.search_link_href = String(await inspectorSearchLink.getAttribute('href') || '');
                contractsAcceptanceInspectorAssertion.checks_by_id.graph_home_smoke.artifact_link_href = String(await inspectorArtifactLink.getAttribute('href') || '');
                contractsAcceptanceInspectorAssertion.checks_by_id.graph_home_smoke.raw_link_href = String(await inspectorRawLink.getAttribute('href') || '');

                const graphHomeSubcommandLink = statusStrip.getByRole('link', {{ name: '图谱主页烟测 / 查看子命令' }}).first();
                await expect(graphHomeSubcommandLink).toBeVisible();
                await graphHomeSubcommandLink.click();
                await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('page_section=contracts-subcommand-graph_home_smoke'));
                inspectorPanel = await ensureInspectorOpen(page);
                await expect(inspectorPanel).toContainText('当前子命令');
                const subcommandSearchLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计检索' }}).first();
                const subcommandArtifactLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计工件' }}).first();
                const subcommandRawLink = inspectorPanel.getByRole('link', {{ name: '查看研究审计原始层' }}).first();
                await expect(subcommandSearchLink).toHaveAttribute('href', firstResearchAuditCase.search_route);
                expect(artifactFromHref(await subcommandArtifactLink.getAttribute('href'))).toBe(firstResearchAuditCase.result_artifact);
                await expect(subcommandRawLink).toHaveAttribute('href', new RegExp(`/workspace/raw\\\\?artifact=${{escapeRegExp(encodeURIComponent(firstResearchAuditCase.raw_path))}}`));
                contractsAcceptanceInspectorAssertion.subcommands_by_id.graph_home_smoke.search_link_href = String(await subcommandSearchLink.getAttribute('href') || '');
                contractsAcceptanceInspectorAssertion.subcommands_by_id.graph_home_smoke.artifact_link_href = String(await subcommandArtifactLink.getAttribute('href') || '');
                contractsAcceptanceInspectorAssertion.subcommands_by_id.graph_home_smoke.raw_link_href = String(await subcommandRawLink.getAttribute('href') || '');
              }}

              expect(snapshotRequests.length).toBeGreaterThanOrEqual(ROUTES.length);
              expect(internalSnapshotRequests.length).toBe(0);

              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    requested_surface: 'public',
                    effective_surface: 'public',
                    visited_routes: visitedRoutes,
                    snapshot_requests: snapshotRequests,
                    internal_snapshot_requests: internalSnapshotRequests,
                    theme_assertion: {{
                      route: themeRoute,
                      requested_theme: 'light',
                      resolved_theme: await page.evaluate(() => document.documentElement.dataset.theme || ''),
                    }},
                    page_section_assertion: {{
                      route: pageSectionRoute,
                      page_section: 'contracts-acceptance-subcommands',
                      active_label: pageSectionActiveLabel,
                      accordion_state: pageSectionAccordionState,
                    }},
                    contracts_source_head_assertion: {{
                      route: contractsSourceHeadRoute,
                      page_section: 'contracts-source-head-operator_panel',
                      source_head_id: 'operator_panel',
                      accordion_state: contractsSourceHeadAccordionState,
                      visible_markers: contractsSourceHeadObservedMarkers,
                    }},
                    contracts_source_gap_assertion: {{
                      route: contractsSourceGapRoute,
                      page_section: 'contracts-fallback',
                      visible_markers: contractsSourceGapObservedMarkers,
                    }},
                    contracts_acceptance_inspector_assertion: contractsAcceptanceInspectorAssertion,
                    artifacts_filter_assertion: {{
                      route: artifactsFilterRoute,
                      group: 'research_cross_section',
                      search_scope: 'title',
                      search: 'orderflow',
                      source_available: orderflowArtifacts.length > 0,
                      active_artifact: orderflowActiveArtifact,
                      visible_artifacts: orderflowArtifacts,
                    }},
                    artifacts_exit_risk_review_assertion: {{
                      route: exitRiskReviewRoute,
                      group: 'research_exit_risk',
                      search_scope: 'title',
                      search: '',
                      source_available: exitRiskReviewArtifacts.length > 0,
                      section_label: exitRiskReviewSectionLabel,
                      active_artifact: exitRiskReviewActiveArtifact,
                      visible_artifacts: exitRiskReviewVisibleArtifacts,
                      visible_markers: exitRiskReviewVisibleArtifacts,
                    }},
                    research_audit_search_assertion: {{
                          route: '/search',
                      cases_available: researchAuditCases.length > 0,
                      cases: researchAuditSearchAssertions,
                    }},
                  }},
                  null,
                  2,
                ),
                'utf8',
              );

              await page.screenshot({{ path: {str(screenshot_path)!r}, fullPage: true }});
            }});
            """
        ).strip()
        + "\n"
    )


def load_internal_alignment_expectations(*, dist_dir: Path) -> dict[str, Any]:
    snapshot_path = dist_dir / "data" / "fenlie_dashboard_internal_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    projection = snapshot.get("conversation_feedback_projection") or {}
    summary = projection.get("summary") or {}
    events = list(projection.get("events") or [])
    actions = list(projection.get("actions") or [])
    headline = str(summary.get("headline") or "").strip() or "暂无高价值反馈"
    top_event_headline = str((events[0] or {}).get("headline") or "").strip() if events else ""
    top_action = str((actions[0] or {}).get("recommended_action") or "").strip() if actions else ""
    markers = [headline]
    if top_event_headline:
        markers.append(top_event_headline)
    if top_action:
        markers.append(top_action)
    return {
        "headline": headline,
        "top_event_headline": top_event_headline,
        "top_action": top_action,
        "route_assertions": [
            {
                "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": markers,
            }
        ],
    }


def visible_marker(value: str, *, max_chars: int = 28) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def focus_slot_label(value: str) -> str:
    raw = str(value or "").strip()
    return TERMINAL_FOCUS_SLOT_LABELS.get(raw, raw or "主槽位")


def load_internal_terminal_focus_expectations(*, dist_dir: Path) -> dict[str, Any]:
    snapshot_path = dist_dir / "data" / "fenlie_dashboard_internal_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    operator_panel = (((snapshot.get("artifact_payloads") or {}).get("operator_panel") or {}).get("payload") or {})
    commodity_summary = (((snapshot.get("artifact_payloads") or {}).get("commodity_reasoning_summary") or {}).get("payload") or {})
    focus_slots = list(operator_panel.get("focus_slots") or [])
    focus_row = (focus_slots[0] if focus_slots else {}) or {}
    focus_row_id = str(focus_row.get("slot") or "primary").strip() or "primary"
    focus_row_label = focus_slot_label(focus_row_id)
    visible_markers = [
        TERMINAL_SIGNAL_RISK_TITLE,
        TERMINAL_FOCUS_SLOTS_TITLE,
        focus_row_label,
        str(commodity_summary.get("primary_scenario_brief") or "").strip(),
        str(commodity_summary.get("primary_chain_brief") or "").strip(),
        str(((commodity_summary.get("contracts_in_focus") or [None])[0]) or "").strip(),
    ]
    visible_markers = [marker for marker in visible_markers if marker]
    return {
        "focus_row_id": focus_row_id,
        "focus_row_label": focus_row_label,
        "route_assertions": [
            {
                "route": TERMINAL_SIGNAL_RISK_ROUTE,
                "nav_label": TERMINAL_SIGNAL_RISK_TITLE,
                "headline": TERMINAL_SIGNAL_RISK_TITLE,
                "markers": visible_markers,
            }
        ],
    }


def build_internal_alignment_smoke_spec(
    *,
    base_url: str,
    screenshot_path: Path,
    result_path: Path,
    projection_summary_headline: str,
    top_event_headline: str,
    top_action: str,
) -> str:
    route = "/workspace/alignment?view=internal&page_section=alignment-summary"
    markers = [
        value
        for value in [
            visible_marker(projection_summary_headline),
            visible_marker(top_event_headline),
            visible_marker(top_action),
        ]
        if value
    ]
    markers_json = json.dumps(markers, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTE = {route!r};
            const BASE_URL = {base_url!r}.replace(/\\/$/, '');
            const MARKERS = {markers_json};
            const PROJECTION_ASSERTION = {{
              headline: {projection_summary_headline!r},
              top_event_headline: {top_event_headline!r},
              top_action: {top_action!r},
            }};

            function escapeRegExp(value) {{
              return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
            }}

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
            }}

            async function selectGraphStageHeading(page, graphBox, detailHeadingLocator, targetHeading) {{
              const candidateOffsets = [
                [220, 0],
                [68, 209],
                [-178, 129],
                [-178, -129],
                [68, -209],
              ];
              for (const [dx, dy] of candidateOffsets) {{
                await page.mouse.click(graphBox.x + graphBox.width / 2 + dx, graphBox.y + graphBox.height / 2 + dy);
                await page.waitForTimeout(450);
                const nextHeading = String(await detailHeadingLocator.textContent() || '').trim();
                if (nextHeading === targetHeading) {{
                  return nextHeading;
                }}
              }}
              return '';
            }}

            async function waitForRoute(page, expectedPathname, expectedParams = {{}}) {{
              await page.waitForFunction(({{ expectedPathname, expectedParams }}) => {{
                const pathname = window.location.pathname;
                if (pathname !== expectedPathname) return false;
                const params = new URLSearchParams(window.location.search);
                return Object.entries(expectedParams).every(([key, value]) => params.get(key) === value);
              }}, {{ expectedPathname, expectedParams }});
            }}

            test.use({{ browserName: 'chromium' }});

            test('internal alignment smoke', async ({{ page }}) => {{
              const publicSnapshotRequests = [];
              const internalSnapshotRequests = [];

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) publicSnapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto(BASE_URL + ROUTE, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('view=internal'));
              await expect(page.getByRole('heading', {{ name: '方向对齐投射', exact: true }})).toBeVisible();
              for (const marker of MARKERS) {{
                await expectStableMarker(page, marker);
              }}
              expect(internalSnapshotRequests.length).toBeGreaterThanOrEqual(1);
              expect(publicSnapshotRequests.length).toBe(0);

              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    requested_surface: 'internal',
                    effective_surface: 'internal',
                    visited_routes: [
                      {{
                        route: ROUTE,
                        headline: '方向对齐投射',
                        url: page.url(),
                      }},
                    ],
                    snapshot_requests: publicSnapshotRequests,
                    internal_snapshot_requests: internalSnapshotRequests,
                    projection_assertion: PROJECTION_ASSERTION,
                  }},
                  null,
                  2,
                ),
                'utf8',
              );

              await page.screenshot({{ path: {str(screenshot_path)!r}, fullPage: true }});
            }});
            """
        ).strip()
        + "\n"
    )


def build_internal_terminal_focus_smoke_spec(
    *,
    base_url: str,
    screenshot_path: Path,
    result_path: Path,
    focus_row_id: str,
    focus_row_label: str,
    visible_markers: list[str],
) -> str:
    route = TERMINAL_SIGNAL_RISK_ROUTE
    visible_markers_json = json.dumps(visible_markers, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTE = {route!r};
            const BASE_URL = {base_url!r}.replace(/\\/$/, '');
            const PANEL_TITLE = {TERMINAL_SIGNAL_RISK_TITLE!r};
            const SECTION_TITLE = {TERMINAL_FOCUS_SLOTS_TITLE!r};
            const FOCUS_ROW_ID = {focus_row_id!r};
            const FOCUS_ROW_LABEL = {focus_row_label!r};
            const EXPECTED_ROW_FRAGMENT = {'row=' + focus_row_id!r};
            const VISIBLE_MARKERS = {visible_markers_json};

            function escapeRegExp(value) {{
              return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
            }}

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
            }}

            async function selectGraphStageHeading(page, graphBox, detailHeadingLocator, targetHeading) {{
              const candidateOffsets = [
                [220, 0],
                [68, 209],
                [-178, 129],
                [-178, -129],
                [68, -209],
              ];
              for (const [dx, dy] of candidateOffsets) {{
                await page.mouse.click(graphBox.x + graphBox.width / 2 + dx, graphBox.y + graphBox.height / 2 + dy);
                await page.waitForTimeout(450);
                const nextHeading = String(await detailHeadingLocator.textContent() || '').trim();
                if (nextHeading === targetHeading) {{
                  return nextHeading;
                }}
              }}
              return '';
            }}

            async function waitForRoute(page, expectedPathname, expectedParams = {{}}) {{
              await page.waitForFunction(({{ expectedPathname, expectedParams }}) => {{
                const pathname = window.location.pathname;
                if (pathname !== expectedPathname) return false;
                const params = new URLSearchParams(window.location.search);
                return Object.entries(expectedParams).every(([key, value]) => params.get(key) === value);
              }}, {{ expectedPathname, expectedParams }});
            }}

            test.use({{ browserName: 'chromium' }});

            test('internal terminal focus smoke', async ({{ page }}) => {{
              const publicSnapshotRequests = [];
              const internalSnapshotRequests = [];

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) publicSnapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto(BASE_URL + ROUTE, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => `${{window.location.pathname}}${{window.location.search}}`.includes('panel=signal-risk') && `${{window.location.pathname}}${{window.location.search}}`.includes('section=focus-slots'));
              await expect(page.getByRole('heading', {{ name: PANEL_TITLE, exact: true }})).toBeVisible();
              for (const marker of VISIBLE_MARKERS) {{
                await expectStableMarker(page, marker);
              }}

              const drilldownState = await page.evaluate((sectionHint) => {{
                const normalizedSectionHint = String(sectionHint || '').replace(/\\s+/g, '').toLowerCase();
                const section = Array.from(document.querySelectorAll('.drill-section')).find((node) => {{
                  const summaryText = String(node.querySelector('summary.drill-summary')?.textContent || '').replace(/\\s+/g, '').toLowerCase();
                  return summaryText.includes(normalizedSectionHint);
                }});
                if (!section) {{
                  return {{
                    section_found: false,
                    section_open: false,
                    summary_link_count: -1,
                    action_link_count: 0,
                    action_link_href: '',
                    action_label_before_click: '',
                  }};
                }}
                if (section instanceof HTMLDetailsElement) {{
                  section.open = true;
                }}
                const firstCard = section.querySelector('.drill-list .drill-card');
                if (firstCard instanceof HTMLDetailsElement) {{
                  firstCard.open = true;
                }}
                const summaryLinkCount = firstCard ? firstCard.querySelectorAll('summary.drill-card-summary .drill-card-link').length : 0;
                const summaryLinks = firstCard ? Array.from(firstCard.querySelectorAll('summary.drill-card-summary .drill-card-link')) : [];
                const firstSummaryLink = summaryLinks[0];
                return {{
                  section_found: true,
                  section_open: section instanceof HTMLDetailsElement ? section.open : false,
                  summary_link_count: summaryLinkCount,
                  summary_link_href: String(firstSummaryLink?.getAttribute('href') || '').trim(),
                  summary_label_before_click: String(firstSummaryLink?.textContent || '').trim(),
                }};
              }}, SECTION_TITLE);

              expect(drilldownState.section_found).toBeTruthy();
              expect(drilldownState.section_open).toBeTruthy();
              const summaryLinkCount = drilldownState.summary_link_count;
              expect(summaryLinkCount).toBeGreaterThanOrEqual(1);
              expect(drilldownState.summary_label_before_click).toBe('定位此项');
              expect(drilldownState.summary_link_href).toContain(EXPECTED_ROW_FRAGMENT);

              const focusLink = page.locator('.drill-section-focused summary.drill-card-summary .drill-card-link').first();
              await expect(focusLink).toBeVisible();
              await expect(focusLink).toContainText('定位此项');
              await focusLink.click();

              await page.waitForFunction((rowFragment) => `${{window.location.pathname}}${{window.location.search}}`.includes(rowFragment), EXPECTED_ROW_FRAGMENT);
              const focusedSummaryLinkCount = await page.evaluate(() => {{
                const focusSection = document.querySelector('.drill-section-focused');
                return focusSection ? focusSection.querySelectorAll('summary.drill-card-summary .drill-card-link').length : -1;
              }});
              expect(focusedSummaryLinkCount).toBeGreaterThanOrEqual(1);

              const activeFocusLink = page.locator('.drill-section-focused summary.drill-card-summary .drill-card-link.active').first();
              await expect(activeFocusLink).toBeVisible();
              await expect(activeFocusLink).toContainText('当前焦点');
              const summaryLabelAfterClick = ((await activeFocusLink.textContent()) || '').trim();
              const summaryLinkHref = (await activeFocusLink.getAttribute('href')) || '';

              expect(internalSnapshotRequests.length).toBeGreaterThanOrEqual(1);
              expect(publicSnapshotRequests.length).toBe(0);

              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    requested_surface: 'internal',
                    effective_surface: 'internal',
                    visited_routes: [
                      {{
                        route: ROUTE,
                        headline: PANEL_TITLE,
                        url: page.url(),
                      }},
                    ],
                    snapshot_requests: publicSnapshotRequests,
                    internal_snapshot_requests: internalSnapshotRequests,
                    terminal_drilldown_assertion: {{
                      route: ROUTE,
                      panel: 'signal-risk',
                      section: 'focus-slots',
                      focus_row_id: FOCUS_ROW_ID,
                      focus_row_label: FOCUS_ROW_LABEL,
                      summary_link_count: summaryLinkCount,
                      summary_label_before_click: drilldownState.summary_label_before_click,
                      summary_label_after_click: summaryLabelAfterClick,
                      summary_link_href: summaryLinkHref,
                    }},
                  }},
                  null,
                  2,
                ),
                'utf8',
              );

              await page.screenshot({{ path: {str(screenshot_path)!r}, fullPage: true }});
            }});
            """
        ).strip()
        + "\n"
    )


def build_commodity_visibility_smoke_spec(
    *,
    base_url: str,
    screenshot_path: Path,
    result_path: Path,
    route_assertions: list[dict[str, Any]],
) -> str:
    routes_json = json.dumps(route_assertions, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTES = {routes_json};
            const BASE_URL = {base_url!r}.replace(/\\/$/, '');

            function escapeRegExp(value) {{
              return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
            }}

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
            }}

            async function selectGraphStageHeading(page, graphBox, detailHeadingLocator, targetHeading) {{
              const candidateOffsets = [
                [220, 0],
                [68, 209],
                [-178, 129],
                [-178, -129],
                [68, -209],
              ];
              for (const [dx, dy] of candidateOffsets) {{
                await page.mouse.click(graphBox.x + graphBox.width / 2 + dx, graphBox.y + graphBox.height / 2 + dy);
                await page.waitForTimeout(450);
                const nextHeading = String(await detailHeadingLocator.textContent() || '').trim();
                if (nextHeading === targetHeading) return nextHeading;
              }}
              return '';
            }}

            async function waitForRoute(page, expectedPathname, expectedParams = {{}}) {{
              await page.waitForFunction(({{ expectedPathname, expectedParams }}) => {{
                if (window.location.pathname !== expectedPathname) return false;
                const params = new URLSearchParams(window.location.search || '');
                return Object.entries(expectedParams).every(([key, value]) => params.get(key) === value);
              }}, {{ expectedPathname, expectedParams }});
            }}

            test.use({{ browserName: 'chromium' }});

            test('commodity visibility smoke', async ({{ page }}) => {{
              const snapshotRequests = [];
              const visitedRoutes = [];

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) snapshotRequests.push(url);
              }});

              for (const route of ROUTES) {{
                await page.goto(BASE_URL + route.route, {{ waitUntil: 'networkidle' }});
                await expect(page.getByRole('heading', {{ name: route.headline, exact: true }})).toBeVisible();
                for (const marker of route.markers) {{
                  await expectStableMarker(page, marker);
                }}
                visitedRoutes.push({{
                  route: route.route,
                  headline: route.headline,
                  url: page.url(),
                }});
              }}

              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    requested_surface: 'public',
                    effective_surface: 'public',
                    visited_routes: visitedRoutes,
                    snapshot_requests: snapshotRequests,
                    internal_snapshot_requests: [],
                  }},
                  null,
                  2,
                ),
                'utf8',
              );

              await page.screenshot({{ path: {str(screenshot_path)!r}, fullPage: true }});
            }});
            """
        ).strip()
        + "\n"
    )


def build_graph_home_smoke_spec(
    *,
    base_url: str,
    screenshot_path: Path,
    result_path: Path,
    route_assertions: list[dict[str, Any]],
) -> str:
    routes_json = json.dumps(route_assertions, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTES = {routes_json};
            const BASE_URL = {base_url!r}.replace(/\\/$/, '');
            const DEFAULT_ROUTE = '/';
            const FALLBACK_ROUTE = '/unknown-route';

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
            }}

            async function selectGraphStageHeading(page, graphBox, detailHeadingLocator, targetHeading) {{
              const candidateOffsets = [
                [220, 0],
                [68, 209],
                [-178, 129],
                [-178, -129],
                [68, -209],
              ];
              for (const [dx, dy] of candidateOffsets) {{
                await page.mouse.click(graphBox.x + graphBox.width / 2 + dx, graphBox.y + graphBox.height / 2 + dy);
                await page.waitForTimeout(450);
                const nextHeading = String(await detailHeadingLocator.textContent() || '').trim();
                if (nextHeading === targetHeading) return nextHeading;
              }}
              return '';
            }}

            async function waitForRoute(page, expectedPathname, expectedParams = {{}}) {{
              await page.waitForFunction(({{ expectedPathname, expectedParams }}) => {{
                const pathname = window.location.pathname;
                if (pathname !== expectedPathname) return false;
                const params = new URLSearchParams(window.location.search);
                return Object.entries(expectedParams).every(([key, value]) => params.get(key) === value);
              }}, {{ expectedPathname, expectedParams }});
            }}

            test.use({{ browserName: 'chromium' }});

            test('graph home smoke', async ({{ page }}) => {{
              const snapshotRequests = [];
              const internalSnapshotRequests = [];
              const visitedRoutes = [];
              const graphRoute = ROUTES[0];

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) snapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto(BASE_URL + DEFAULT_ROUTE, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => window.location.pathname.includes('/graph-home'));
              await expect(page.getByRole('heading', {{ name: graphRoute.headline, exact: true }})).toBeVisible();
              for (const marker of graphRoute.markers) {{
                await expectStableMarker(page, marker);
              }}

              const nav = page.getByRole('navigation', {{ name: 'primary-domains' }});
              await expect(nav.getByRole('link', {{ name: '图谱主页' }})).toBeVisible();
              const defaultCenter = await page.locator('.summary-chip').filter({{ hasText: '中心：' }}).first().textContent();
              const terminalLink = page.getByRole('link', {{ name: '去操作终端' }});
              const workspaceLink = page.getByRole('link', {{ name: '去研究工作区' }});
              const searchLink = page.getByRole('link', {{ name: '打开全局搜索' }});
              const terminalLinkHref = await terminalLink.getAttribute('href');
              const workspaceLinkHref = await workspaceLink.getAttribute('href');
              const searchLinkHref = await searchLink.getAttribute('href');
              const researchAuditCases = Array.isArray(graphRoute.research_audit_search_cases)
                ? graphRoute.research_audit_search_cases
                : [];
              const graphCanvas = page.locator('canvas').first();
              await graphCanvas.scrollIntoViewIfNeeded();
              const graphBox = await graphCanvas.boundingBox();
              if (!graphBox) throw new Error('graph_canvas_missing');
              const detailHeadingLocator = page.locator('.graph-home-side .panel-card').first().locator('.panel-card-title');
              const centerChipLocator = page.locator('.graph-home-toolbar .summary-chip').filter({{ hasText: '中心：' }}).first();
              const candidateOffsets = [
                [220, 0],
                [68, 209],
                [-178, 129],
                [-178, -129],
                [68, -209],
              ];
              const selectableHeadings = new Set(['市场输入', '研究判断', '交易逻辑', '执行与风控', '复盘反馈']);
              let selectedHeading = '';
              let selectedCenter = '';
              for (const [dx, dy] of candidateOffsets) {{
                await page.mouse.click(graphBox.x + graphBox.width / 2 + dx, graphBox.y + graphBox.height / 2 + dy);
                await page.waitForTimeout(450);
                const nextHeading = String(await detailHeadingLocator.textContent() || '').trim();
                if (selectableHeadings.has(nextHeading)) {{
                  selectedHeading = nextHeading;
                  selectedCenter = String(await centerChipLocator.textContent() || '').trim().replace(/^中心：/, '');
                  break;
                }}
              }}
              const selectionObserved = selectedHeading !== '';
              let recenterHeading = '';
              if (selectionObserved) {{
                await page.getByRole('button', {{ name: '回到交易中枢' }}).click();
                await expect(detailHeadingLocator).toContainText('交易中枢');
                recenterHeading = String(await detailHeadingLocator.textContent() || '').trim();
              }}
              const researchAuditLinkAssertions = [];
              if (researchAuditCases.length) {{
                for (const auditCase of researchAuditCases) {{
                  const auditResultArtifact = String(auditCase.result_artifact || '');
                  const auditSearchLink = page.getByRole('link', {{ name: `检索 / ${{auditCase.query}}` }}).first();
                  const auditArtifactLink = page.getByRole('link', {{ name: `工件 / ${{auditCase.result_artifact}}` }}).first();
                  const auditRawLink = page.getByRole('link', {{ name: `原始层 / ${{auditCase.raw_path}}` }}).first();
                  await expect(auditSearchLink).toBeVisible();
                  await expect(auditArtifactLink).toBeVisible();
                  await expect(auditRawLink).toBeVisible();
                  const auditSearchHref = await auditSearchLink.getAttribute('href');
                  const auditArtifactHref = await auditArtifactLink.getAttribute('href');
                  const auditRawHref = await auditRawLink.getAttribute('href');
                  await auditSearchLink.click();
                  await page.waitForFunction((route) => `${{window.location.pathname}}${{window.location.search}}` === route, auditCase.search_route);
                  await page.goto(BASE_URL + graphRoute.route, {{ waitUntil: 'networkidle' }});
                  await auditArtifactLink.click();
                  await waitForRoute(page, '/workspace/artifacts', {{ artifact: auditResultArtifact }});
                  await page.goto(BASE_URL + graphRoute.route, {{ waitUntil: 'networkidle' }});
                  await auditRawLink.click();
                  await page.waitForFunction((rawPath) => `${{window.location.pathname}}${{window.location.search}}`.includes('/workspace/raw') && `${{window.location.pathname}}${{window.location.search}}`.includes(encodeURIComponent(rawPath)), auditCase.raw_path);
                  await page.goto(BASE_URL + graphRoute.route, {{ waitUntil: 'networkidle' }});
                  researchAuditLinkAssertions.push({{
                    selected_heading: String(defaultCenter || '').trim().replace(/^中心：/, ''),
                    case_id: String(auditCase.case_id || ''),
                    search_link_href: auditSearchHref,
                    artifact_link_href: auditArtifactHref,
                    raw_link_href: auditRawHref,
                  }});
                }}
              }}

              visitedRoutes.push({{
                route: graphRoute.route,
                headline: graphRoute.headline,
                url: page.url(),
              }});

              await terminalLink.click();
              await page.waitForFunction(() => window.location.pathname.includes('/ops/risk'));
              await expectStableMarker(page, '风险观察');
              visitedRoutes.push({{
                route: '/ops/risk',
                headline: 'Observe / Diagnose / Act',
                url: page.url(),
              }});

              await page.goto(BASE_URL + graphRoute.route, {{ waitUntil: 'networkidle' }});
              await workspaceLink.click();
              await page.waitForFunction(() => window.location.pathname.includes('/workspace/artifacts'));
              await expect(page.getByRole('heading', {{ name: '工件目标池', exact: true }})).toBeVisible();
              visitedRoutes.push({{
                route: '/workspace/artifacts',
                headline: '工件目标池',
                url: page.url(),
              }});

              await page.goto(BASE_URL + graphRoute.route, {{ waitUntil: 'networkidle' }});
              await searchLink.click();
              await page.waitForFunction(() => window.location.pathname.includes('/search'));
              await expect(page.getByRole('heading', {{ name: '全局关键词搜索', exact: true }})).toBeVisible();
              visitedRoutes.push({{
                route: '/search',
                headline: '全局关键词搜索',
                url: page.url(),
              }});

              await page.goto(BASE_URL + FALLBACK_ROUTE, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => window.location.pathname.includes('/graph-home'));
              await expect(page.getByRole('heading', {{ name: graphRoute.headline, exact: true }})).toBeVisible();
              const resolvedRoute = await page.evaluate(() => `${{window.location.pathname}}${{window.location.search}}`);

              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    requested_surface: 'public',
                    effective_surface: 'public',
                    visited_routes: visitedRoutes,
                    snapshot_requests: snapshotRequests,
                    internal_snapshot_requests: internalSnapshotRequests,
                    graph_home_assertion: {{
                      default_route: DEFAULT_ROUTE,
                      fallback_route: FALLBACK_ROUTE,
                      resolved_route: resolvedRoute,
                      default_center: String(defaultCenter || '').trim().replace(/^中心：/, ''),
                      terminal_link_href: terminalLinkHref,
                      workspace_link_href: workspaceLinkHref,
                      search_link_href: searchLinkHref,
                      canvas_selection_assertion: {{
                        selection_observed: selectionObserved,
                        selected_heading: selectedHeading,
                        selected_center: selectedCenter,
                        recenter_heading: recenterHeading,
                      }},
                      research_audit_link_assertions: researchAuditLinkAssertions,
                    }},
                  }},
                  null,
                  2,
                ),
                'utf8',
              );

              await page.screenshot({{ path: {str(screenshot_path)!r}, fullPage: true }});
            }});
            """
        ).strip()
        + "\n"
    )


def build_graph_home_narrow_smoke_spec(
    *,
    base_url: str,
    screenshot_path: Path,
    result_path: Path,
    route_assertions: list[dict[str, Any]],
) -> str:
    routes_json = json.dumps(route_assertions, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTES = {routes_json};
            const BASE_URL = {base_url!r}.replace(/\\/$/, '');
            const DEFAULT_ROUTE = '/';

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
            }}

            test.use({{ browserName: 'chromium', viewport: {{ width: 390, height: 844 }} }});

            test('graph home narrow smoke', async ({{ page }}) => {{
              const snapshotRequests = [];
              const internalSnapshotRequests = [];
              const visitedRoutes = [];
              const graphRoute = ROUTES[0];

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) snapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto(BASE_URL + DEFAULT_ROUTE, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => window.location.pathname.includes('/graph-home'));
              await expect(page.getByRole('heading', {{ name: graphRoute.headline, exact: true }})).toBeVisible();
              for (const marker of graphRoute.markers) {{
                await expectStableMarker(page, marker);
              }}

              const shellTier = await page.locator('.app-shell-grid').getAttribute('data-shell-tier');
              const sidebarToggleVisible = await page.getByRole('button', {{ name: /侧边导航/ }}).isVisible().catch(() => false);
              const sidebarUtilityLink = page.getByRole('link', {{ name: '搜索 / Search' }});
              await expect(sidebarUtilityLink).toBeVisible();
              const sidebarUtilityLinkHref = await sidebarUtilityLink.getAttribute('href');
              const resolvedRoute = await page.evaluate(() => window.location.pathname);

              visitedRoutes.push({{
                route: graphRoute.route,
                headline: graphRoute.headline,
                url: page.url(),
              }});

              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    requested_surface: 'public',
                    effective_surface: 'public',
                    visited_routes: visitedRoutes,
                    snapshot_requests: snapshotRequests,
                    internal_snapshot_requests: internalSnapshotRequests,
                    graph_home_assertion: {{
                      default_route: DEFAULT_ROUTE,
                      resolved_route: resolvedRoute,
                      shell_tier: shellTier,
                      sidebar_toggle_visible: sidebarToggleVisible,
                      sidebar_utility_link_href: sidebarUtilityLinkHref,
                    }},
                  }},
                  null,
                  2,
                ),
                'utf8',
              );

              await page.screenshot({{ path: {str(screenshot_path)!r}, fullPage: true }});
            }});
            """
        ).strip()
        + "\n"
    )


def build_graph_home_pipeline_smoke_spec(
    *,
    base_url: str,
    screenshot_path: Path,
    result_path: Path,
    route_assertions: list[dict[str, Any]],
) -> str:
    routes_json = json.dumps(route_assertions, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTES = {routes_json};
            const BASE_URL = {base_url!r}.replace(/\\/$/, '');
            const DEFAULT_ROUTE = '/';
            const BASE_URL = {base_url!r}.replace(/\\/$/, '');
            const STORAGE_KEY = 'graph_home_pipelines_v1';

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
            }}

            test.use({{ browserName: 'chromium' }});

            test('graph home pipeline smoke', async ({{ page }}) => {{
              const snapshotRequests = [];
              const internalSnapshotRequests = [];
              const visitedRoutes = [];
              const graphRoute = ROUTES[0];

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) snapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto(BASE_URL + DEFAULT_ROUTE, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => window.location.pathname.includes('/graph-home'));
              await expect(page.getByRole('heading', {{ name: graphRoute.headline, exact: true }})).toBeVisible();
              for (const marker of graphRoute.markers) {{
                await expectStableMarker(page, marker);
              }}

              const addToPipelineButton = page.getByRole('button', {{ name: '加入自定义管道' }});
              await addToPipelineButton.click();
              await page.waitForTimeout(300);

              const graphCanvas = page.locator('canvas').first();
              await graphCanvas.scrollIntoViewIfNeeded();
              const graphBox = await graphCanvas.boundingBox();
              if (!graphBox) throw new Error('graph_canvas_missing');
              const detailHeadingLocator = page.locator('.graph-home-side .panel-card').first().locator('.panel-card-title');
              const candidateOffsets = [
                [220, 0],
                [68, 209],
                [-178, 129],
                [-178, -129],
                [68, -209],
              ];
              const selectableHeadings = new Set(['市场输入', '研究判断', '交易逻辑', '执行与风控', '复盘反馈']);
              let selectedHeading = '';
              for (const [dx, dy] of candidateOffsets) {{
                await page.mouse.click(graphBox.x + graphBox.width / 2 + dx, graphBox.y + graphBox.height / 2 + dy);
                await page.waitForTimeout(350);
                const nextHeading = String(await detailHeadingLocator.textContent() || '').trim();
                if (selectableHeadings.has(nextHeading)) {{
                  selectedHeading = nextHeading;
                  break;
                }}
              }}
              expect(selectedHeading).not.toBe('');
              await addToPipelineButton.click();
              await page.waitForTimeout(300);

              const pipelineItems = page.locator('.graph-pipeline-item');
              const initialOrder = await page.locator('.graph-pipeline-item > span:first-child').allTextContents();
              expect(initialOrder.length).toBeGreaterThanOrEqual(2);

              await pipelineItems.nth(1).dragTo(pipelineItems.nth(0));
              await page.waitForTimeout(800);
              const reorderedOrder = await page.locator('.graph-pipeline-item > span:first-child').allTextContents();
              expect(reorderedOrder[0]).toBe(selectedHeading);
              expect(reorderedOrder[1]).toBe('交易中枢');

              const storedAfterDrag = await page.evaluate((key) => JSON.parse(window.localStorage.getItem(key) || 'null'), STORAGE_KEY);
              expect(storedAfterDrag?.pipelines?.[0]?.nodeIds || []).toEqual(['pipeline-execution-risk', 'trade-hub']);

              await page.reload({{ waitUntil: 'networkidle' }});
              const persistedOrder = await page.locator('.graph-pipeline-item > span:first-child').allTextContents();
              expect(persistedOrder[0]).toBe(selectedHeading);
              expect(persistedOrder[1]).toBe('交易中枢');
              const resolvedRoute = await page.evaluate(() => window.location.pathname);

              visitedRoutes.push({{
                route: graphRoute.route,
                headline: graphRoute.headline,
                url: page.url(),
              }});

              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    requested_surface: 'public',
                    effective_surface: 'public',
                    visited_routes: visitedRoutes,
                    snapshot_requests: snapshotRequests,
                    internal_snapshot_requests: internalSnapshotRequests,
                    graph_home_assertion: {{
                      default_route: DEFAULT_ROUTE,
                      resolved_route: resolvedRoute,
                      pipeline_persistence_assertion: {{
                        selected_heading: selectedHeading,
                        initial_order: initialOrder,
                        reordered_order: reorderedOrder,
                        persisted_order: persistedOrder,
                      }},
                    }},
                  }},
                  null,
                  2,
                ),
                'utf8',
              );

              await page.screenshot({{ path: {str(screenshot_path)!r}, fullPage: true }});
            }});
            """
        ).strip()
        + "\n"
    )


def run_playwright_smoke(
    *,
    web_root: Path,
    base_url: str,
    timeout_ms: int,
    screenshot_path: Path,
    mode: str = "public_workspace",
    internal_expectations: dict[str, Any] | None = None,
    route_assertions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    web_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fenlie-dashboard-smoke-", dir=web_root) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        spec_path = temp_dir / "workspace-artifacts.smoke.spec.cjs"
        result_path = temp_dir / "workspace-routes-smoke.result.json"
        if mode in {"internal_alignment", "internal_alignment_manual_probe"}:
            expectations = internal_expectations or {}
            spec = build_internal_alignment_smoke_spec(
                base_url=base_url,
                screenshot_path=screenshot_path,
                result_path=result_path,
                projection_summary_headline=str(expectations.get("headline") or "暂无高价值反馈"),
                top_event_headline=str(expectations.get("top_event_headline") or ""),
                top_action=str(expectations.get("top_action") or ""),
            )
        elif mode == "internal_terminal_focus":
            expectations = internal_expectations or {}
            spec = build_internal_terminal_focus_smoke_spec(
                base_url=base_url,
                screenshot_path=screenshot_path,
                result_path=result_path,
                focus_row_id=str(expectations.get("focus_row_id") or "primary"),
                focus_row_label=str(expectations.get("focus_row_label") or focus_slot_label("primary")),
                visible_markers=list(((route_assertions or [{}])[0] or {}).get("markers") or [TERMINAL_SIGNAL_RISK_TITLE, TERMINAL_FOCUS_SLOTS_TITLE]),
            )
        elif mode == "commodity_visibility":
            spec = build_commodity_visibility_smoke_spec(
                base_url=base_url,
                screenshot_path=screenshot_path,
                result_path=result_path,
                route_assertions=route_assertions or [],
            )
        elif mode == "graph_home":
            spec = build_graph_home_smoke_spec(
                base_url=base_url,
                screenshot_path=screenshot_path,
                result_path=result_path,
                route_assertions=route_assertions or GRAPH_HOME_ROUTE_ASSERTIONS,
            )
        elif mode == "graph_home_narrow":
            spec = build_graph_home_narrow_smoke_spec(
                base_url=base_url,
                screenshot_path=screenshot_path,
                result_path=result_path,
                route_assertions=route_assertions or GRAPH_HOME_ROUTE_ASSERTIONS,
            )
        elif mode == "graph_home_pipeline":
            spec = build_graph_home_pipeline_smoke_spec(
                base_url=base_url,
                screenshot_path=screenshot_path,
                result_path=result_path,
                route_assertions=route_assertions or [{
                    "route": "/graph-home",
                    "nav_label": "图谱主页",
                    "headline": "图谱化主页",
                    "markers": ["加入自定义管道", "创建默认管道"],
                }],
            )
        else:
            spec = build_workspace_routes_smoke_spec(
                base_url=base_url,
                screenshot_path=screenshot_path,
                result_path=result_path,
                route_assertions=route_assertions,
            )
        spec_path.write_text(spec, encoding="utf-8")
        cmd = [
            "npx",
            "@playwright/test",
            "test",
            str(spec_path),
            "-c",
            str(temp_dir),
            "--browser=chromium",
            "--reporter=line",
            "--timeout",
            str(timeout_ms),
        ]
        proc = subprocess.run(cmd, cwd=web_root, text=True, capture_output=True, check=False)
        playwright_result = {}
        if result_path.exists():
            playwright_result = json.loads(result_path.read_text(encoding="utf-8"))
        return {
            "name": "playwright_workspace_routes_smoke",
            "cmd": cmd,
            "cwd": str(web_root),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "spec_path": str(spec_path),
            "screenshot_path": str(screenshot_path),
            "result_path": str(result_path),
            "playwright_result": playwright_result,
        }


def build_artifact_payload(
    *,
    workspace: Path,
    report_path: Path,
    screenshot_path: Path,
    build_result: dict[str, Any] | None,
    server_ready_seconds: float,
    smoke_result: dict[str, Any],
    base_url: str,
    mode: str = "public_workspace",
    expected_route_markers: list[dict[str, Any]] | None = None,
    failure_assertion: dict[str, Any] | None = None,
    force_failed: bool = False,
) -> dict[str, Any]:
    ok = (smoke_result.get("returncode") == 0) and not force_failed
    stdout = str(smoke_result.get("stdout") or "")
    stderr = str(smoke_result.get("stderr") or "")
    playwright_result = smoke_result.get("playwright_result") or {}
    visited_routes = list(playwright_result.get("visited_routes") or [])
    snapshot_requests = list(playwright_result.get("snapshot_requests") or [])
    internal_snapshot_requests = list(playwright_result.get("internal_snapshot_requests") or [])
    theme_assertion = playwright_result.get("theme_assertion") or {
        "route": "/workspace/contracts?theme=light",
        "requested_theme": "light",
        "resolved_theme": "",
    }
    if mode == "public_workspace":
        page_section_assertion = {
            "applicable": True,
            **(playwright_result.get("page_section_assertion") or {
                "route": "/workspace/contracts?page_section=contracts-acceptance-subcommands",
                "page_section": "contracts-acceptance-subcommands",
                "active_label": "",
                "accordion_state": "",
            }),
        }
        contracts_source_head_assertion = {
            "applicable": True,
            **(playwright_result.get("contracts_source_head_assertion") or {
                "route": CONTRACTS_SOURCE_HEAD_ASSERTION["route"],
                "page_section": CONTRACTS_SOURCE_HEAD_ASSERTION["page_section"],
                "source_head_id": CONTRACTS_SOURCE_HEAD_ASSERTION["source_head_id"],
                "accordion_state": "",
                "visible_markers": [],
            }),
        }
        contracts_source_gap_assertion = {
            "applicable": True,
            **(playwright_result.get("contracts_source_gap_assertion") or {
                "route": CONTRACTS_SOURCE_GAP_ASSERTION["route"],
                "page_section": CONTRACTS_SOURCE_GAP_ASSERTION["page_section"],
                "visible_markers": [],
            }),
        }
        default_contracts_acceptance_inspector_assertion = {
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
                    "search_link_href": "",
                    "artifact_link_href": "",
                    "raw_link_href": "",
                },
                "graph_home_smoke": {
                    "route": CONTRACTS_ACCEPTANCE_INSPECTOR_CHECK_ROUTE,
                    "page_section": "contracts-check-graph_home_smoke",
                    "search_link_href": "",
                    "artifact_link_href": "",
                    "raw_link_href": "",
                },
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
                    "search_link_href": "",
                    "artifact_link_href": "",
                    "raw_link_href": "",
                },
                "graph_home_smoke": {
                    "route": CONTRACTS_ACCEPTANCE_INSPECTOR_SUBCOMMAND_ROUTE,
                    "page_section": "contracts-subcommand-graph_home_smoke",
                    "check_route": CONTRACTS_ACCEPTANCE_INSPECTOR_CHECK_ROUTE,
                    "search_link_href": "",
                    "artifact_link_href": "",
                    "raw_link_href": "",
                },
            },
        }
        actual_contracts_acceptance_inspector_assertion = playwright_result.get("contracts_acceptance_inspector_assertion") or {}
        contracts_acceptance_inspector_assertion = {
            "applicable": True,
            **actual_contracts_acceptance_inspector_assertion,
            "checks_by_id": {
                **default_contracts_acceptance_inspector_assertion["checks_by_id"],
                **(actual_contracts_acceptance_inspector_assertion.get("checks_by_id") or {}),
            },
            "subcommands_by_id": {
                **default_contracts_acceptance_inspector_assertion["subcommands_by_id"],
                **(actual_contracts_acceptance_inspector_assertion.get("subcommands_by_id") or {}),
            },
        }
        artifacts_filter_assertion = {
            "applicable": True,
            **(playwright_result.get("artifacts_filter_assertion") or {
                "route": "/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
                "group": "research_cross_section",
                "search_scope": "title",
                "search": "orderflow",
                "source_available": False,
                "active_artifact": "",
                "visible_artifacts": [],
            }),
        }
        artifacts_exit_risk_review_assertion = {
            "applicable": True,
            **(playwright_result.get("artifacts_exit_risk_review_assertion") or {
                "route": ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["route"],
                "group": ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["group"],
                "search_scope": ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["search_scope"],
                "search": ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["search"],
                "source_available": False,
                "section_label": ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["section_label"],
                "active_artifact": "",
                "visible_artifacts": [],
                "visible_markers": [],
            }),
        }
        research_audit_search_assertion = {
            "applicable": True,
            **(playwright_result.get("research_audit_search_assertion") or {
                "route": "/search",
                "cases_available": False,
                "cases": [],
            }),
        }
    else:
        page_section_assertion = {
            "applicable": False,
            "route": "",
            "page_section": "",
            "active_label": "",
            "accordion_state": "",
        }
        contracts_source_head_assertion = {
            "applicable": False,
            "route": "",
            "page_section": "",
            "source_head_id": "",
            "accordion_state": "",
            "visible_markers": [],
        }
        contracts_source_gap_assertion = {
            "applicable": False,
            "route": "",
            "page_section": "",
            "visible_markers": [],
        }
        contracts_acceptance_inspector_assertion = {
            "applicable": False,
            "checks_by_id": {},
            "subcommands_by_id": {},
        }
        artifacts_filter_assertion = {
            "applicable": False,
            "route": "",
            "group": "",
            "search_scope": "",
            "search": "",
            "source_available": False,
            "active_artifact": "",
            "visible_artifacts": [],
        }
        artifacts_exit_risk_review_assertion = {
            "applicable": False,
            "route": "",
            "group": "",
            "search_scope": "",
            "search": "",
            "source_available": False,
            "section_label": "",
            "active_artifact": "",
            "visible_artifacts": [],
            "visible_markers": [],
        }
        research_audit_search_assertion = {
            "applicable": False,
            "route": "",
            "cases_available": False,
            "cases": [],
        }
    terminal_drilldown_assertion = playwright_result.get("terminal_drilldown_assertion")
    if not visited_routes:
        visited_routes = [
            {
                "route": str(route["route"]),
                "headline": str(route["headline"]),
            }
            for route in (expected_route_markers or PUBLIC_WORKSPACE_ROUTE_ASSERTIONS)
        ]
    route_markers = expected_route_markers or PUBLIC_WORKSPACE_ROUTE_ASSERTIONS
    if mode == "internal_alignment_manual_probe":
        action_name = "dashboard_internal_alignment_manual_probe_browser_smoke"
    elif mode == "internal_alignment":
        action_name = "dashboard_internal_alignment_browser_smoke"
    elif mode == "internal_terminal_focus":
        action_name = "dashboard_internal_terminal_focus_browser_smoke"
    elif mode == "commodity_visibility":
        action_name = "dashboard_commodity_visibility_browser_smoke"
    elif mode == "graph_home":
        action_name = "dashboard_graph_home_browser_smoke"
    elif mode == "graph_home_narrow":
        action_name = "dashboard_graph_home_narrow_browser_smoke"
    elif mode == "graph_home_pipeline":
        action_name = "dashboard_graph_home_pipeline_browser_smoke"
    else:
        action_name = "dashboard_workspace_routes_browser_smoke"
    expected_focus_panel = "signal-risk" if mode == "internal_terminal_focus" else "lab-review"
    expected_focus_section = "focus-slots" if mode == "internal_terminal_focus" else "research-heads"
    expected_default_artifact = "price_action_breakout_pullback"
    if mode == "public_workspace" and len(route_markers) > 1:
        workspace_start = route_markers[1]
        expected_default_artifact = str(workspace_start.get("expected_default_artifact") or expected_default_artifact)
        expected_focus_panel = str(workspace_start.get("expected_focus_panel") or expected_focus_panel)
        expected_focus_section = str(workspace_start.get("expected_focus_section") or expected_focus_section)
    snapshot_endpoint_observed = (
        "/data/fenlie_dashboard_internal_snapshot.json"
        if str(playwright_result.get("requested_surface") or "public") == "internal"
        else "/data/fenlie_dashboard_snapshot.json"
    )
    payload = {
        "action": action_name,
        "ok": ok,
        "status": "ok" if ok else "failed",
        "change_class": CHANGE_CLASS,
        "generated_at_utc": fmt_utc(now_utc()),
        "workspace": str(workspace),
        "base_url": base_url,
        "server_ready_seconds": server_ready_seconds,
        "build_run": bool(build_result),
        "build_returncode": build_result.get("returncode") if build_result else None,
        "smoke_returncode": smoke_result.get("returncode"),
        "surface_assertion": {
            "requested_surface": str(playwright_result.get("requested_surface") or "public"),
            "effective_surface": str(playwright_result.get("effective_surface") or "public"),
            "snapshot_endpoint_observed": snapshot_endpoint_observed,
        },
        "routes": visited_routes,
        "network_observation": {
            "public_snapshot_fetch_count": len(snapshot_requests),
            "public_snapshot_fetches": snapshot_requests,
            "internal_snapshot_fetch_count": len(internal_snapshot_requests),
            "internal_snapshot_fetches": internal_snapshot_requests,
        },
        "theme_assertion": theme_assertion,
        "page_section_assertion": page_section_assertion,
        "contracts_source_head_assertion": contracts_source_head_assertion,
        "contracts_source_gap_assertion": contracts_source_gap_assertion,
        "contracts_acceptance_inspector_assertion": contracts_acceptance_inspector_assertion,
        "artifacts_filter_assertion": artifacts_filter_assertion,
        "artifacts_exit_risk_review_assertion": artifacts_exit_risk_review_assertion,
        "research_audit_search_assertion": research_audit_search_assertion,
        "terminal_drilldown_assertion": terminal_drilldown_assertion,
        "graph_home_assertion": playwright_result.get("graph_home_assertion"),
        "projection_assertion": playwright_result.get("projection_assertion"),
        "expected_default_artifact": expected_default_artifact,
        "expected_focus_panel": expected_focus_panel,
        "expected_focus_section": expected_focus_section,
        "expected_route_markers": route_markers,
        "screenshot_path": str(screenshot_path),
        "report_path": str(report_path),
        "playwright_stdout_tail": stdout.strip().splitlines()[-20:],
        "playwright_stderr_tail": stderr.strip().splitlines()[-20:],
    }
    if failure_assertion:
        payload["failure_assertion"] = failure_assertion
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real-browser smoke check for Fenlie workspace route refresh semantics.")
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for temporary static server.")
    parser.add_argument("--port", type=int, default=0, help="Bind port for temporary static server. 0 = auto-pick.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0, help="HTTP readiness timeout.")
    parser.add_argument("--skip-build", action="store_true", help="Skip npm run build before smoke.")
    parser.add_argument(
        "--mode",
        choices=["public_workspace", "internal_alignment", "internal_alignment_manual_probe", "internal_terminal_focus", "commodity_visibility", "graph_home", "graph_home_narrow", "graph_home_pipeline"],
        default="public_workspace",
        help="Smoke scope. public_workspace validates the public route matrix; internal_alignment validates the internal alignment page; internal_alignment_manual_probe temporarily seeds a manual feedback row and verifies the same internal page end-to-end; internal_terminal_focus validates the terminal/internal focus-slot drilldown CTA path; commodity_visibility validates overview + terminal/public commodity reasoning visibility; graph_home validates the default landing page redirect and graph-home quick links; graph_home_narrow validates the same landing page contract under a narrow viewport; graph_home_pipeline validates custom-pipeline drag reorder and reload persistence.",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    web_root = system_root / "dashboard" / "web"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = web_root / "dist"
    runtime_now = now_utc()
    timestamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    if args.mode == "internal_alignment":
        report_path = review_dir / f"{timestamp}_dashboard_internal_alignment_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_internal_alignment_browser_smoke.png"
    elif args.mode == "internal_alignment_manual_probe":
        report_path = review_dir / f"{timestamp}_dashboard_internal_alignment_manual_probe_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_internal_alignment_manual_probe_browser_smoke.png"
    elif args.mode == "internal_terminal_focus":
        report_path = review_dir / f"{timestamp}_dashboard_internal_terminal_focus_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_internal_terminal_focus_browser_smoke.png"
    elif args.mode == "commodity_visibility":
        report_path = review_dir / f"{timestamp}_dashboard_commodity_visibility_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_commodity_visibility_browser_smoke.png"
    elif args.mode == "graph_home":
        report_path = review_dir / f"{timestamp}_dashboard_graph_home_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_graph_home_browser_smoke.png"
    elif args.mode == "graph_home_narrow":
        report_path = review_dir / f"{timestamp}_dashboard_graph_home_narrow_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_graph_home_narrow_browser_smoke.png"
    elif args.mode == "graph_home_pipeline":
        report_path = review_dir / f"{timestamp}_dashboard_graph_home_pipeline_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_graph_home_pipeline_browser_smoke.png"
    else:
        report_path = review_dir / f"{timestamp}_dashboard_workspace_routes_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_workspace_routes_browser_smoke.png"

    build_result: dict[str, Any] | None = None
    port = 0
    base_url = ""
    if args.mode in {"internal_alignment", "internal_alignment_manual_probe"}:
        expected_route_markers = [
            {
                "route": "/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": [],
            }
        ]
    elif args.mode == "internal_terminal_focus":
        expected_route_markers = [
            {
                "route": TERMINAL_SIGNAL_RISK_ROUTE,
                "nav_label": TERMINAL_SIGNAL_RISK_TITLE,
                "headline": TERMINAL_SIGNAL_RISK_TITLE,
                "markers": [TERMINAL_SIGNAL_RISK_TITLE, TERMINAL_FOCUS_SLOTS_TITLE],
            }
        ]
    elif args.mode == "commodity_visibility":
        expected_route_markers = [
            {
                "route": COMMODITY_VISIBILITY_OVERVIEW_ROUTE,
                "nav_label": "总览",
                "headline": "总览",
                "markers": [COMMODITY_REASONING_LINE_TITLE],
            },
            {
                "route": COMMODITY_VISIBILITY_TERMINAL_ROUTE,
                "nav_label": "操作终端",
                "headline": "Observe / Diagnose / Act",
                "markers": [COMMODITY_REASONING_LINE_TITLE],
            },
        ]
    elif args.mode == "graph_home":
        expected_route_markers = load_graph_home_route_assertions(dist_dir=dist_dir)
    elif args.mode == "graph_home_narrow":
        expected_route_markers = list(GRAPH_HOME_ROUTE_ASSERTIONS)
    elif args.mode == "graph_home_pipeline":
        expected_route_markers = [{
            "route": "/graph-home",
            "nav_label": "图谱主页",
            "headline": "图谱化主页",
            "markers": ["加入自定义管道", "创建默认管道"],
        }]
    else:
        expected_route_markers = list(PUBLIC_WORKSPACE_ROUTE_ASSERTIONS)
    server_ready_seconds = 0.0
    smoke_result: dict[str, Any] = {
        "returncode": 1,
        "stdout": "",
        "stderr": "",
        "playwright_result": {},
    }

    probe_context: contextlib.AbstractContextManager[Any]
    if args.mode == "internal_alignment_manual_probe":
        probe_context = temporary_manual_probe(
            workspace=workspace,
            system_root=system_root,
            review_dir=review_dir,
            runtime_now=runtime_now,
        )
    else:
        probe_context = contextlib.nullcontext()

    try:
        port = args.port or choose_port(args.host)
        base_url = f"http://{args.host}:{port}/"

        if not args.skip_build:
            build_result = ensure_success(name="dashboard_build", cmd=["npm", "run", "build"], cwd=web_root)

        if not (dist_dir / "index.html").exists():
            raise FileNotFoundError(f"dist_index_missing: {dist_dir / 'index.html'}")

        with probe_context:
            if args.mode in {"internal_alignment", "internal_alignment_manual_probe"}:
                internal_expectations = load_internal_alignment_expectations(dist_dir=dist_dir)
                expected_route_markers = list(internal_expectations["route_assertions"])
            elif args.mode == "internal_terminal_focus":
                internal_expectations = load_internal_terminal_focus_expectations(dist_dir=dist_dir)
                expected_route_markers = list(internal_expectations["route_assertions"])
            elif args.mode == "commodity_visibility":
                internal_expectations = None
                expected_route_markers = load_commodity_visibility_route_assertions(dist_dir=dist_dir)
            elif args.mode == "graph_home":
                internal_expectations = None
                expected_route_markers = load_graph_home_route_assertions(dist_dir=dist_dir)
            elif args.mode == "graph_home_narrow":
                internal_expectations = None
                expected_route_markers = list(GRAPH_HOME_ROUTE_ASSERTIONS)
            elif args.mode == "graph_home_pipeline":
                internal_expectations = None
                expected_route_markers = [{
                    "route": "/graph-home",
                    "nav_label": "图谱主页",
                    "headline": "图谱化主页",
                    "markers": ["加入自定义管道", "创建默认管道"],
                }]
            else:
                internal_expectations = None
                expected_route_markers = load_public_workspace_route_assertions(dist_dir=dist_dir)

            with http_server(dist_dir=dist_dir, host=args.host, port=port):
                server_ready_seconds = wait_http_ready(base_url, timeout_seconds=args.timeout_seconds)
                smoke_result = run_playwright_smoke(
                    web_root=web_root,
                    base_url=base_url,
                    timeout_ms=int(args.timeout_seconds * 1000),
                    screenshot_path=screenshot_path,
                    mode=args.mode,
                    internal_expectations=internal_expectations,
                    route_assertions=expected_route_markers,
                )
    except Exception as exc:  # noqa: BLE001
        failure_assertion: dict[str, Any] = {
            "failure_stage": str(exc).split(":", 1)[0].strip() or exc.__class__.__name__,
            "failure_detail": str(exc),
        }
        if args.mode == "internal_alignment_manual_probe":
            failure_assertion["probe_feedback_id"] = MANUAL_PROBE_FEEDBACK_ID
            failure_assertion["manual_probe_state"] = inspect_manual_probe_state(review_dir=review_dir)
        payload = build_artifact_payload(
            workspace=workspace,
            report_path=report_path,
            screenshot_path=screenshot_path,
            build_result=build_result,
            server_ready_seconds=server_ready_seconds,
            smoke_result=smoke_result,
            base_url=base_url,
            mode=args.mode,
            expected_route_markers=expected_route_markers,
            failure_assertion=failure_assertion,
            force_failed=True,
        )
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    payload = build_artifact_payload(
        workspace=workspace,
        report_path=report_path,
        screenshot_path=screenshot_path,
        build_result=build_result,
        server_ready_seconds=server_ready_seconds,
        smoke_result=smoke_result,
        base_url=base_url,
        mode=args.mode,
        expected_route_markers=expected_route_markers,
    )
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
