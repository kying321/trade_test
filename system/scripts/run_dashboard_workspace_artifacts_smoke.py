#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import socket
import subprocess
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
        "markers": ["研究主线摘要", "hold24_zero"],
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
        "markers": ["告警定向原始层", "ETH 15m 主线"],
    },
    {
        "route": "#/workspace/contracts",
        "nav_label": "契约层",
        "headline": "公开入口拓扑",
        "markers": [
            "公开面验收",
            "root overview 截图",
            "pages overview 截图",
            "root contracts 截图",
            "pages contracts 截图",
            "公开快照拉取次数",
            "内部快照拉取次数",
        ],
    },
]
CHANGE_CLASS = "RESEARCH_ONLY"
MANUAL_PROBE_FEEDBACK_ID = "manual_alignment_browser_smoke_probe"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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
                "python3",
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
                    "python3",
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
        "python3",
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


def build_workspace_routes_smoke_spec(*, base_url: str, screenshot_path: Path, result_path: Path) -> str:
    route_matrix_json = json.dumps(PUBLIC_WORKSPACE_ROUTE_ASSERTIONS, ensure_ascii=False, indent=2)
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
              await expect(page.getByText(new RegExp(escapeRegExp(marker), 'i')).first()).toBeVisible();
            }}

            test.use({{ browserName: 'chromium' }});

            test('workspace routes smoke', async ({{ page }}) => {{
              const snapshotRequests = [];
              const internalSnapshotRequests = [];
              const visitedRoutes = [];
              const overviewRoute = ROUTES[0];
              const workspaceStartRoute = ROUTES[1];
              const artifactsFilterRoute = '#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow';
              const orderflowArtifacts = ['intraday_orderflow_blueprint', 'intraday_orderflow_research_gate_blocker'];
              const themeRoute = '#/workspace/contracts?theme=light';
              const pageSectionRoute = '#/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke';

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
              await page.waitForFunction(() => window.location.hash.includes('artifact=price_action_breakout_pullback'));
              await expect(page).toHaveURL(/artifact=price_action_breakout_pullback/);
              await expect(page).toHaveURL(/panel=lab-review/);
              await expect(page).toHaveURL(/section=research-heads/);

              const activeText = await page.locator('.artifact-layer-mainline .artifact-button.active').textContent();
              expect(activeText || '').toMatch(/price_action_breakout_pullback|ETH 15m 主线/);
              await expect(page.getByRole('heading', {{ name: workspaceStartRoute.headline, exact: true }})).toBeVisible();
              for (const marker of workspaceStartRoute.markers) {{
                await expectStableMarker(page, marker);
              }}
              visitedRoutes.push({{
                route: workspaceStartRoute.route,
                headline: workspaceStartRoute.headline,
                url: page.url(),
              }});

              await page.goto({base_url!r} + artifactsFilterRoute, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction((fragment) => window.location.hash.includes(fragment), artifactsFilterRoute);
              await page.waitForFunction(() => window.location.hash.includes('artifact=intraday_orderflow_blueprint'));
              for (const artifactId of orderflowArtifacts) {{
                await expectStableMarker(page, artifactId);
              }}
              const activeOrderflowArtifact = page.locator('.artifact-layer-cross .artifact-button.active').first();
              await expect(activeOrderflowArtifact).toContainText('intraday_orderflow_blueprint');

              for (const route of ROUTES.slice(2)) {{
                await page.locator('nav[aria-label=\"context-nav\"]').getByText(route.nav_label, {{ exact: true }}).click();
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

              await page.goto({base_url!r} + pageSectionRoute, {{ waitUntil: 'networkidle' }});
              await page.waitForFunction(() => window.location.hash.includes('page_section=contracts-subcommand-workspace_routes_smoke'));
              const activePageSection = page.locator('nav[aria-label="page-sections-nav"]').first().locator('.page-section-link.active').first();
              await expect(activePageSection).toContainText('工作区路由子命令');
              const focusedAccordion = page.locator('[data-accordion-id="contracts-subcommand-workspace_routes_smoke"]').first();
              await expect(focusedAccordion).toHaveAttribute('data-state', 'open');

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
                      page_section: 'contracts-subcommand-workspace_routes_smoke',
                      active_label: ((await activePageSection.textContent()) || '').trim(),
                      accordion_state: await focusedAccordion.getAttribute('data-state'),
                    }},
                    artifacts_filter_assertion: {{
                      route: artifactsFilterRoute,
                      group: 'research_cross_section',
                      search_scope: 'title',
                      search: 'orderflow',
                      active_artifact: 'intraday_orderflow_blueprint',
                      visible_artifacts: orderflowArtifacts,
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
    markers = [value for value in [projection_summary_headline, top_event_headline, top_action] if value]
    markers_json = json.dumps(markers, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTE = {route!r};
            const MARKERS = {markers_json};

            function escapeRegExp(value) {{
              return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
            }}

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.getByText(new RegExp(escapeRegExp(marker), 'i')).first()).toBeVisible();
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
              await expect(page.getByRole('navigation', {{ name: 'page-sections-nav' }})).toBeVisible();
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
                    projection_assertion: {{
                      headline: MARKERS[0] || '',
                      top_event_headline: MARKERS[1] || '',
                      top_action: MARKERS[2] || '',
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
        else:
            spec = build_workspace_routes_smoke_spec(
                base_url=base_url,
                screenshot_path=screenshot_path,
                result_path=result_path,
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
    page_section_assertion = playwright_result.get("page_section_assertion") or {
        "route": "#/workspace/contracts?page_section=contracts-subcommand-workspace_routes_smoke",
        "page_section": "contracts-subcommand-workspace_routes_smoke",
        "active_label": "",
        "accordion_state": "",
    }
    artifacts_filter_assertion = playwright_result.get("artifacts_filter_assertion") or {
        "route": "#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
        "group": "research_cross_section",
        "search_scope": "title",
        "search": "orderflow",
        "active_artifact": "",
        "visible_artifacts": [],
    }
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
    else:
        action_name = "dashboard_workspace_routes_browser_smoke"
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
        "artifacts_filter_assertion": artifacts_filter_assertion,
        "projection_assertion": playwright_result.get("projection_assertion"),
        "expected_default_artifact": "price_action_breakout_pullback",
        "expected_focus_panel": "lab-review",
        "expected_focus_section": "research-heads",
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
        choices=["public_workspace", "internal_alignment", "internal_alignment_manual_probe"],
        default="public_workspace",
        help="Smoke scope. public_workspace validates the public route matrix; internal_alignment validates the internal alignment page; internal_alignment_manual_probe temporarily seeds a manual feedback row and verifies the same internal page end-to-end.",
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
    else:
        report_path = review_dir / f"{timestamp}_dashboard_workspace_routes_browser_smoke.json"
        screenshot_path = review_dir / f"{timestamp}_dashboard_workspace_routes_browser_smoke.png"

    build_result: dict[str, Any] | None = None
    port = 0
    base_url = ""
    expected_route_markers = (
        [
            {
                "route": "#/workspace/alignment?view=internal&page_section=alignment-summary",
                "nav_label": "对齐页",
                "headline": "方向对齐投射",
                "markers": [],
            }
        ]
        if args.mode in {"internal_alignment", "internal_alignment_manual_probe"}
        else list(PUBLIC_WORKSPACE_ROUTE_ASSERTIONS)
    )
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
            else:
                internal_expectations = None

            with http_server(dist_dir=dist_dir, host=args.host, port=port):
                server_ready_seconds = wait_http_ready(base_url, timeout_seconds=args.timeout_seconds)
                smoke_result = run_playwright_smoke(
                    web_root=web_root,
                    base_url=base_url,
                    timeout_ms=int(args.timeout_seconds * 1000),
                    screenshot_path=screenshot_path,
                    mode=args.mode,
                    internal_expectations=internal_expectations,
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
