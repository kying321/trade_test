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


CHANGE_CLASS = "RESEARCH_ONLY"
PANEL_ROUTE = "operator_task_visual_panel.html"
PANEL_SECTION_TITLE = "事件危机地缘层"
COMMODITY_SECTION_TITLE = "国内商品推理线"


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def current_python_executable() -> str:
    return sys.executable or "python3"


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


def choose_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


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


def load_operator_panel_expectations(*, dist_dir: Path) -> dict[str, Any]:
    panel_data_path = dist_dir / "operator_task_visual_panel_data.json"
    if not panel_data_path.exists():
        raise FileNotFoundError(f"operator_panel_data_missing: {panel_data_path}")
    payload = json.loads(panel_data_path.read_text(encoding="utf-8"))
    summary = dict(payload.get("summary") or {})
    markers = [
        PANEL_SECTION_TITLE,
        str(summary.get("event_crisis_primary_theater_brief") or ""),
        str(summary.get("event_crisis_dominant_chain_brief") or ""),
        str(summary.get("event_crisis_safety_margin_brief") or ""),
        str(summary.get("event_crisis_hard_boundary_brief") or ""),
        COMMODITY_SECTION_TITLE,
        str(summary.get("commodity_reasoning_primary_scenario_brief") or ""),
        str(summary.get("commodity_reasoning_primary_chain_brief") or ""),
        str(summary.get("commodity_reasoning_range_scope_brief") or ""),
        str(summary.get("commodity_reasoning_boundary_strength_brief") or ""),
        str(summary.get("commodity_reasoning_invalidator_brief") or ""),
    ]
    markers = [marker for marker in markers if marker.strip()]
    return {
        "route": f"/{PANEL_ROUTE}",
        "markers": markers,
        "summary": summary,
        "panel_data_path": str(panel_data_path),
    }


def build_operator_panel_smoke_spec(
    *,
    base_url: str,
    screenshot_path: Path,
    result_path: Path,
    route: str,
    markers: list[str],
) -> str:
    markers_json = json.dumps(markers, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROUTE = {route!r};
            const BASE_URL = {base_url!r};
            const TARGET = `${{BASE_URL.replace(/\\/$/, '')}}${{ROUTE}}`;
            const MARKERS = {markers_json};

            function escapeRegExp(value) {{
              return value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
            }}

            async function expectStableMarker(page, marker) {{
              await page.waitForFunction((text) => document.body.innerText.toLowerCase().includes(text.toLowerCase()), marker);
              await expect(page.getByText(new RegExp(escapeRegExp(marker), 'i')).first()).toBeVisible();
            }}

            test.use({{ browserName: 'chromium' }});

            test('operator panel browser smoke', async ({{ page }}) => {{
              await page.goto(TARGET, {{ waitUntil: 'networkidle' }});
              for (const marker of MARKERS) {{
                await expectStableMarker(page, marker);
              }}

              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    route: ROUTE,
                    final_url: page.url(),
                    visible_markers: MARKERS,
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
    screenshot_path: Path,
    route: str,
    markers: list[str],
    timeout_ms: int,
) -> dict[str, Any]:
    web_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fenlie-operator-panel-smoke-", dir=web_root) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        spec_path = temp_dir / "operator-panel.smoke.spec.cjs"
        result_path = temp_dir / "operator-panel.smoke.result.json"
        spec = build_operator_panel_smoke_spec(
            base_url=base_url,
            screenshot_path=screenshot_path,
            result_path=result_path,
            route=route,
            markers=markers,
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
            "name": "playwright_operator_panel_browser_smoke",
            "cmd": cmd,
            "cwd": str(web_root),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "spec_path": str(spec_path),
            "result_path": str(result_path),
            "screenshot_path": str(screenshot_path),
            "playwright_result": playwright_result,
        }


def build_artifact_payload(
    *,
    workspace: Path,
    report_path: Path,
    screenshot_path: Path,
    base_url: str,
    server_ready_seconds: float,
    build_result: dict[str, Any] | None,
    smoke_result: dict[str, Any],
    route: str,
    markers: list[str],
    failure_assertion: dict[str, Any] | None = None,
    force_failed: bool = False,
) -> dict[str, Any]:
    ok = (smoke_result.get("returncode") == 0) and not force_failed
    stdout = str(smoke_result.get("stdout") or "")
    stderr = str(smoke_result.get("stderr") or "")
    playwright_result = smoke_result.get("playwright_result") or {}
    payload = {
        "action": "operator_panel_browser_smoke",
        "ok": ok,
        "status": "ok" if ok else "failed",
        "change_class": CHANGE_CLASS,
        "generated_at_utc": fmt_utc(now_utc()),
        "workspace": str(workspace),
        "base_url": base_url,
        "route": route,
        "expected_markers": markers,
        "visible_markers": list(playwright_result.get("visible_markers") or []),
        "final_url": str(playwright_result.get("final_url") or ""),
        "server_ready_seconds": server_ready_seconds,
        "build_run": bool(build_result),
        "build_returncode": build_result.get("returncode") if build_result else None,
        "smoke_returncode": smoke_result.get("returncode"),
        "screenshot_path": str(screenshot_path),
        "report_path": str(report_path),
        "playwright_stdout_tail": stdout.strip().splitlines()[-20:],
        "playwright_stderr_tail": stderr.strip().splitlines()[-20:],
    }
    if failure_assertion:
        payload["failure_assertion"] = failure_assertion
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a browser-level smoke for the operator panel geostrategy + commodity reasoning sections.")
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for temporary static server.")
    parser.add_argument("--port", type=int, default=0, help="Bind port for temporary static server. 0 = auto-pick.")
    parser.add_argument("--timeout-seconds", type=float, default=20.0, help="HTTP readiness and browser timeout.")
    parser.add_argument("--skip-build", action="store_true", help="Skip npm run build before smoke.")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    web_root = system_root / "dashboard" / "web"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = web_root / "dist"
    runtime_now = now_utc()
    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    report_path = review_dir / f"{stamp}_operator_panel_browser_smoke.json"
    screenshot_path = review_dir / f"{stamp}_operator_panel_browser_smoke.png"

    build_result: dict[str, Any] | None = None
    port = args.port or choose_port(args.host)
    base_url = f"http://{args.host}:{port}/"
    server_ready_seconds = 0.0
    smoke_result: dict[str, Any] = {"returncode": 1, "stdout": "", "stderr": "", "playwright_result": {}}
    expectations: dict[str, Any] = {"route": f"/{PANEL_ROUTE}", "markers": []}

    try:
        if not args.skip_build:
            build_result = ensure_success(
                name="operator_panel_refresh",
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
        if not (dist_dir / PANEL_ROUTE).exists():
            raise FileNotFoundError(f"operator_panel_html_missing: {dist_dir / PANEL_ROUTE}")
        expectations = load_operator_panel_expectations(dist_dir=dist_dir)
        with http_server(dist_dir=dist_dir, host=args.host, port=port):
            server_ready_seconds = wait_http_ready(base_url, timeout_seconds=args.timeout_seconds)
            smoke_result = run_playwright_smoke(
                web_root=web_root,
                base_url=base_url,
                screenshot_path=screenshot_path,
                route=str(expectations["route"]),
                markers=list(expectations["markers"]),
                timeout_ms=int(args.timeout_seconds * 1000),
            )
    except Exception as exc:  # noqa: BLE001
        payload = build_artifact_payload(
            workspace=workspace,
            report_path=report_path,
            screenshot_path=screenshot_path,
            base_url=base_url,
            server_ready_seconds=server_ready_seconds,
            build_result=build_result,
            smoke_result=smoke_result,
            route=str(expectations.get("route") or f"/{PANEL_ROUTE}"),
            markers=list(expectations.get("markers") or []),
            failure_assertion={
                "failure_stage": str(exc).split(":", 1)[0].strip() or exc.__class__.__name__,
                "failure_detail": str(exc),
            },
            force_failed=True,
        )
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    payload = build_artifact_payload(
        workspace=workspace,
        report_path=report_path,
        screenshot_path=screenshot_path,
        base_url=base_url,
        server_ready_seconds=server_ready_seconds,
        build_result=build_result,
        smoke_result=smoke_result,
        route=str(expectations["route"]),
        markers=list(expectations["markers"]),
    )
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
