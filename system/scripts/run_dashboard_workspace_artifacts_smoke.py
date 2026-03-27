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
        "route": "#/overview",
        "nav_label": "总览",
        "headline": "总览",
        "markers": ["关键摘要", "系统运行", "调度心跳", "研究主线", "退出风控", "下一步去哪"],
    },
    {
        "route": "#/workspace/artifacts",
        "nav_label": "工件池",
        "headline": "工件目标池",
        "markers": ["研究地图"],
    },
    {
        "route": "#/workspace/alignment",
        "nav_label": "对齐页",
        "headline": "方向对齐投射",
        "markers": ["仅内部可见"],
    },
    {
        "route": "#/workspace/backtests",
        "nav_label": "回测池",
        "headline": "回测主池",
        "markers": ["穿透层 1 / 回测池", "穿透层 2 / 近期比较行"],
    },
    {
        "route": "#/workspace/raw",
        "nav_label": "原始层",
        "headline": "原始快照",
        "markers": ["告警定向原始层", "操作面板"],
    },
    {
        "route": "#/workspace/contracts",
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
    "route": "#/workspace/contracts?page_section=contracts-source-head-operator_panel",
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
    "route": "#/workspace/contracts?page_section=contracts-fallback",
    "page_section": "contracts-fallback",
    "visible_markers": [
        "#/workspace/raw",
        "/data/fenlie_dashboard_snapshot.json",
        "/operator_task_visual_panel.html",
    ],
}
ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION = {
    "route": "#/workspace/artifacts?artifact=price_action_exit_risk_break_even_review_conclusion",
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
TERMINAL_SIGNAL_RISK_ROUTE = "#/terminal/internal?panel=signal-risk&section=focus-slots"
TERMINAL_SIGNAL_RISK_TITLE = "信号发生器与风险节流阀"
TERMINAL_FOCUS_SLOTS_TITLE = "穿透层 3 / 焦点槽位"
TERMINAL_FOCUS_SLOT_LABELS = {
    "primary": "主槽位",
    "followup": "跟进槽位",
    "secondary": "次级槽位",
}
COMMODITY_VISIBILITY_OVERVIEW_ROUTE = "#/overview"
COMMODITY_VISIBILITY_TERMINAL_ROUTE = "#/terminal/public"
COMMODITY_REASONING_LINE_TITLE = "国内商品推理线"


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
        route_assertions[1]["orderflow_filter_route"] = "#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow"
        route_assertions[1]["orderflow_filter_visible_artifacts"] = orderflow_visible_artifacts
        route_assertions[1]["orderflow_filter_active_artifact"] = orderflow_visible_artifacts[0] if orderflow_visible_artifacts else ""
        route_assertions[1]["exit_risk_review_route"] = ARTIFACTS_EXIT_RISK_REVIEW_ASSERTION["route"]
        route_assertions[1]["exit_risk_review_visible_artifacts"] = exit_risk_review_visible_artifacts
        route_assertions[1]["exit_risk_review_active_artifact"] = exit_risk_review_visible_artifacts[0] if exit_risk_review_visible_artifacts else ""
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
            "headline": "执行穿透 / 调度与门禁",
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
    cmd = [
        current_python_executable(),
        "-m",
        "http.server",
        str(port),
        "--bind",
        host,
        "--directory",
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

            function escapeRegExp(value) {{
              return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
            }}

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
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
              const artifactsFilterRoute = String(workspaceStartRoute.orderflow_filter_route || '#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow');
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
              let exitRiskReviewSectionLabel = exitRiskReviewSectionHint;
              let exitRiskReviewVisibleArtifacts = [];
              const themeRoute = '#/workspace/contracts?theme=light';
              const pageSectionRoute = '#/workspace/contracts?page_section=contracts-acceptance-subcommands';
              const contractsSourceHeadRoute = {CONTRACTS_SOURCE_HEAD_ASSERTION["route"]!r};
              const contractsSourceHeadMarkers = {json.dumps(CONTRACTS_SOURCE_HEAD_ASSERTION["visible_markers"], ensure_ascii=False, indent=2)};
              const contractsSourceGapRoute = {CONTRACTS_SOURCE_GAP_ASSERTION["route"]!r};
              const contractsSourceGapMarkers = {json.dumps(CONTRACTS_SOURCE_GAP_ASSERTION["visible_markers"], ensure_ascii=False, indent=2)};
              const contractsSourceGapPayloadKey = 'finding_count';

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) snapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto({base_url!r} + overviewRoute.route, {{ waitUntil: 'networkidle' }});
              for (const marker of overviewRoute.markers) {{
                await expectStableMarker(page, marker);
              }}
              visitedRoutes.push({{
                route: overviewRoute.route,
                headline: overviewRoute.headline,
                url: page.url(),
              }});

              await page.goto({base_url!r} + workspaceStartRoute.route, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction((artifactId) => window.location.hash.includes(`artifact=${{artifactId}}`), defaultWorkspaceArtifact);
              await page.waitForFunction((panelId) => window.location.hash.includes(`panel=${{panelId}}`), defaultWorkspacePanel);
              await page.waitForFunction((sectionId) => window.location.hash.includes(`section=${{sectionId}}`), defaultWorkspaceSection);
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
                await page.goto({base_url!r} + artifactsFilterRoute, {{ waitUntil: 'networkidle' }});
                await page.waitForFunction((fragment) => window.location.hash.includes(fragment), artifactsFilterRoute);
                await page.waitForFunction((artifactId) => window.location.hash.includes(`artifact=${{artifactId}}`), orderflowActiveArtifact);
                for (const artifactId of orderflowArtifacts) {{
                  await expectStableMarker(page, artifactId);
                }}
                const activeOrderflowArtifact = page.locator('.artifact-layer-cross .artifact-button.active .value-text').first();
                await expect(activeOrderflowArtifact).toHaveAttribute('title', orderflowActiveArtifact);
              }}

              if (exitRiskReviewArtifacts.length && exitRiskReviewActiveArtifact) {{
                await page.goto({base_url!r} + exitRiskReviewRoute, {{ waitUntil: 'networkidle' }});
                await page.waitForFunction((fragment) => window.location.hash.includes(fragment), exitRiskReviewRoute);
                await page.waitForFunction((artifactId) => window.location.hash.includes(`artifact=${{artifactId}}`), exitRiskReviewActiveArtifact);
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

              for (const route of ROUTES.slice(2)) {{
                await clickContextNav(page, route.nav_label);
                await page.waitForFunction((fragment) => window.location.hash.includes(fragment), route.route);
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

              await page.goto({base_url!r} + themeRoute, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => window.location.hash.includes('theme=light'));
              await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
              await expect(page.getByRole('button', {{ name: '白天主题' }})).toHaveAttribute('aria-pressed', 'true');
              const contractsPageSectionHrefs = await page.evaluate(() => Array.from(document.querySelectorAll('nav[aria-label="page-sections-nav"] .page-section-link')).map((node) => String(node.getAttribute('href') || '')));
              let pageSectionActiveLabel = '';
              let pageSectionAccordionState = '';
              if (contractsPageSectionHrefs.some((href) => href.includes('page_section=contracts-acceptance-subcommands'))) {{
                await page.goto({base_url!r} + pageSectionRoute, {{ waitUntil: 'networkidle' }});
                await page.waitForFunction(() => window.location.hash.includes('page_section=contracts-acceptance-subcommands'));
                const activePageSection = page.locator('nav[aria-label="page-sections-nav"]').first().locator('.page-section-link.active').first();
                await expect(activePageSection).toContainText('子命令证据');
                pageSectionActiveLabel = ((await activePageSection.textContent()) || '').trim();
              }}

              let contractsSourceHeadAccordionState = '';
              let contractsSourceHeadObservedMarkers = [];
              if (contractsPageSectionHrefs.some((href) => href.includes('page_section=contracts-source-head-operator_panel'))) {{
                await page.goto({base_url!r} + contractsSourceHeadRoute, {{ waitUntil: 'networkidle' }});
                await page.waitForFunction(() => window.location.hash.includes('page_section=contracts-source-head-operator_panel'));
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
                await page.goto({base_url!r} + contractsSourceGapRoute, {{ waitUntil: 'networkidle' }});
                await page.waitForFunction(() => window.location.hash.includes('page_section=contracts-fallback'));
                const contractsSourceGapSection = page.locator('[data-workspace-section="contracts-fallback"]').first();
                await expect(contractsSourceGapSection).toBeVisible();
                for (const marker of contractsSourceGapMarkers) {{
                  await expect(contractsSourceGapSection).toContainText(marker);
                }}
                contractsSourceGapObservedMarkers = contractsSourceGapMarkers;
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
                "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
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
        "国内商品推理线",
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
    route = "#/workspace/alignment?view=internal&page_section=alignment-summary"
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

            test.use({{ browserName: 'chromium' }});

            test('internal alignment smoke', async ({{ page }}) => {{
              const publicSnapshotRequests = [];
              const internalSnapshotRequests = [];

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) publicSnapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto({base_url!r} + ROUTE, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => window.location.hash.includes('view=internal'));
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

            test.use({{ browserName: 'chromium' }});

            test('internal terminal focus smoke', async ({{ page }}) => {{
              const publicSnapshotRequests = [];
              const internalSnapshotRequests = [];

              page.on('response', async (response) => {{
                const url = response.url();
                if (url.includes('fenlie_dashboard_snapshot.json')) publicSnapshotRequests.push(url);
                if (url.includes('fenlie_dashboard_internal_snapshot.json')) internalSnapshotRequests.push(url);
              }});

              await page.goto({base_url!r} + ROUTE, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => window.location.hash.includes('panel=signal-risk') && window.location.hash.includes('section=focus-slots'));
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

              await page.waitForFunction((rowFragment) => window.location.hash.includes(rowFragment), EXPECTED_ROW_FRAGMENT);
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

            function escapeRegExp(value) {{
              return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
            }}

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.locator('body')).toContainText(marker);
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
                await page.goto({base_url!r} + route.route, {{ waitUntil: 'networkidle' }});
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
        "route": "#/workspace/contracts?theme=light",
        "requested_theme": "light",
        "resolved_theme": "",
    }
    if mode == "public_workspace":
        page_section_assertion = {
            "applicable": True,
            **(playwright_result.get("page_section_assertion") or {
                "route": "#/workspace/contracts?page_section=contracts-acceptance-subcommands",
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
        artifacts_filter_assertion = {
            "applicable": True,
            **(playwright_result.get("artifacts_filter_assertion") or {
                "route": "#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
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
        "artifacts_filter_assertion": artifacts_filter_assertion,
        "artifacts_exit_risk_review_assertion": artifacts_exit_risk_review_assertion,
        "terminal_drilldown_assertion": terminal_drilldown_assertion,
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
        choices=["public_workspace", "internal_alignment", "internal_alignment_manual_probe", "internal_terminal_focus", "commodity_visibility"],
        default="public_workspace",
        help="Smoke scope. public_workspace validates the public route matrix; internal_alignment validates the internal alignment page; internal_alignment_manual_probe temporarily seeds a manual feedback row and verifies the same internal page end-to-end; internal_terminal_focus validates the terminal/internal focus-slot drilldown CTA path; commodity_visibility validates overview + terminal/public commodity reasoning visibility.",
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
    else:
        report_path = review_dir / f"{timestamp}_dashboard_workspace_routes_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_workspace_routes_browser_smoke.png"

    build_result: dict[str, Any] | None = None
    port = 0
    base_url = ""
    if args.mode in {"internal_alignment", "internal_alignment_manual_probe"}:
        expected_route_markers = [
            {
                "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
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
                "headline": "执行穿透 / 调度与门禁",
                "markers": [COMMODITY_REASONING_LINE_TITLE],
            },
        ]
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
