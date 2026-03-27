#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import socket
import ssl
import subprocess
import tempfile
import textwrap
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


REQUEST_TIMEOUT_CAP_SECONDS = 5.0
STRICT_REQUEST_ATTEMPTS = 2
CHANGE_CLASS = "RESEARCH_ONLY"
MIN_BROWSER_RENDER_TIMEOUT_MS = 15000
DEFAULT_ROOT_URL = "https://fuuu.fun"
DEFAULT_PAGES_URL = "https://fenlie.fuuu.fun"
DEFAULT_OPENCLAW_URL = "http://43.153.148.242:3001"
DEFAULT_GATEWAY_URL = "http://43.153.148.242:8787"
PUBLIC_OVERVIEW_ROUTE_PATH = "overview"
PUBLIC_CONTRACTS_ROUTE_PATH = "workspace/contracts"
PUBLIC_OVERVIEW_MARKERS = ["研究主线摘要", "国内商品推理线"]
PUBLIC_CONTRACTS_MARKERS = [
    "公开面验收",
    "穿透层 1 / 验收总览",
    "接口目录",
    "源头主线",
    "回退链",
]


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_title = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._parts.append(data)

    @property
    def title(self) -> str:
        return "".join(self._parts).strip()


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_system_root(workspace: Path) -> Path:
    if (workspace / "system").exists():
        return workspace / "system"
    if workspace.name == "system":
        return workspace
    raise FileNotFoundError(f"cannot resolve system root from {workspace}")


def extract_html_title(payload: str) -> str:
    parser = _TitleParser()
    parser.feed(payload)
    return parser.title


def _timeout_seconds(value: float) -> float:
    if value <= 0:
        raise ValueError("timeout_seconds must be > 0")
    return min(float(value), REQUEST_TIMEOUT_CAP_SECONDS)


def _is_tls_error(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, ssl.SSLError) or isinstance(exc, ssl.SSLError)


def _is_transient_transport_error(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", exc)
    return isinstance(
        reason,
        (
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
            ConnectionResetError,
            ConnectionAbortedError,
            TimeoutError,
            socket.timeout,
            ssl.SSLEOFError,
        ),
    )


def _request_url(
    url: str,
    *,
    timeout_seconds: float,
    allow_insecure_tls_fallback: bool = False,
) -> tuple[int, dict[str, str], str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "fenlie-dashboard-topology-smoke/1.0"})
    timeout = _timeout_seconds(timeout_seconds)

    def _open_with_context(context: ssl.SSLContext | None = None):
        handlers: list[Any] = [urllib.request.ProxyHandler({})]
        if context is not None:
            handlers.append(urllib.request.HTTPSHandler(context=context))
        opener = urllib.request.build_opener(*handlers)
        return opener.open(request, timeout=timeout)

    def _read_response(context: ssl.SSLContext | None = None, *, tls_verification: str) -> tuple[int, dict[str, str], str, str]:
        for attempt in range(STRICT_REQUEST_ATTEMPTS):
            try:
                with _open_with_context(context) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    headers = {key.lower(): value for key, value in response.headers.items()}
                    return int(response.status), headers, body, tls_verification
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                headers = {key.lower(): value for key, value in exc.headers.items()}
                return int(exc.code), headers, body, tls_verification
            except urllib.error.URLError as exc:
                if attempt + 1 >= STRICT_REQUEST_ATTEMPTS or not _is_transient_transport_error(exc):
                    raise
            except Exception as exc:
                if attempt + 1 >= STRICT_REQUEST_ATTEMPTS or not _is_transient_transport_error(exc):
                    raise
        raise RuntimeError("request_attempts_exhausted")

    try:
        return _read_response(tls_verification="strict")
    except urllib.error.URLError as exc:
        if allow_insecure_tls_fallback and url.lower().startswith("https://") and _is_tls_error(exc):
            insecure_context = ssl._create_unverified_context()
            return _read_response(insecure_context, tls_verification="insecure_fallback")
        raise


def probe_text_endpoint(
    *,
    name: str,
    url: str,
    timeout_seconds: float,
    allow_insecure_tls_fallback: bool = False,
) -> dict[str, Any]:
    status_code, headers, body, tls_verification = _request_url(
        url,
        timeout_seconds=timeout_seconds,
        allow_insecure_tls_fallback=allow_insecure_tls_fallback,
    )
    return {
        "name": name,
        "url": url,
        "status_code": status_code,
        "headers": headers,
        "body_preview": body[:240],
        "title": extract_html_title(body),
        "ok": True,
        "timeout_seconds": _timeout_seconds(timeout_seconds),
        "tls_verification": tls_verification,
    }


def probe_json_endpoint(
    *,
    name: str,
    url: str,
    timeout_seconds: float,
    allow_insecure_tls_fallback: bool = False,
) -> dict[str, Any]:
    status_code, headers, body, tls_verification = _request_url(
        url,
        timeout_seconds=timeout_seconds,
        allow_insecure_tls_fallback=allow_insecure_tls_fallback,
    )
    return {
        "name": name,
        "url": url,
        "status_code": status_code,
        "headers": headers,
        "payload": json.loads(body),
        "ok": True,
        "timeout_seconds": _timeout_seconds(timeout_seconds),
        "tls_verification": tls_verification,
    }


def build_public_route_browser_spec(
    *,
    root_url: str,
    route_path: str,
    markers: list[str],
    navigation_timeout_ms: int,
    render_timeout_ms: int,
    result_path: Path,
    screenshot_path: Path,
) -> str:
    markers_json = json.dumps(markers, ensure_ascii=False, indent=2)
    return (
        textwrap.dedent(
            f"""
            const {{ test, expect }} = require('@playwright/test');
            const fs = require('node:fs');

            const ROOT_URL = {root_url!r};
            const ROUTE = `${{ROOT_URL.replace(/\\/$/, '')}}/#/{route_path}`;
            const MARKERS = {markers_json};
            const NAVIGATION_TIMEOUT_MS = {navigation_timeout_ms};
            const RENDER_TIMEOUT_MS = {render_timeout_ms};
            const escapeRegExp = (value) => value.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');

            test.use({{ browserName: 'chromium' }});

            test('public route smoke', async ({{ page }}) => {{
              await page.goto(ROUTE, {{ waitUntil: 'domcontentloaded', timeout: NAVIGATION_TIMEOUT_MS }});
              for (const marker of MARKERS) {{
                await page.waitForFunction(
                  (text) => document.body.innerText.toLowerCase().includes(String(text).toLowerCase()),
                  marker,
                  {{ timeout: RENDER_TIMEOUT_MS }},
                );
                await expect(page.getByText(new RegExp(escapeRegExp(marker), 'i')).first()).toBeVisible({{ timeout: RENDER_TIMEOUT_MS }});
              }}
              await page.screenshot({{ path: {str(screenshot_path)!r}, fullPage: true }});
              fs.writeFileSync(
                {str(result_path)!r},
                JSON.stringify(
                  {{
                    route: ROUTE,
                    final_url: page.url(),
                    markers: MARKERS,
                    screenshot_path: {str(screenshot_path)!r},
                  }},
                  null,
                  2,
                ),
                'utf8',
              );
            }});
            """
        ).strip()
        + "\n"
    )


def build_public_overview_browser_spec(
    *,
    root_url: str,
    navigation_timeout_ms: int,
    render_timeout_ms: int,
    result_path: Path,
    screenshot_path: Path,
) -> str:
    return build_public_route_browser_spec(
        root_url=root_url,
        route_path=PUBLIC_OVERVIEW_ROUTE_PATH,
        markers=PUBLIC_OVERVIEW_MARKERS,
        navigation_timeout_ms=navigation_timeout_ms,
        render_timeout_ms=render_timeout_ms,
        result_path=result_path,
        screenshot_path=screenshot_path,
    )


def run_public_route_browser_smoke(
    *,
    name: str,
    root_url: str,
    route_path: str,
    markers: list[str],
    timeout_seconds: float,
    screenshot_path: Path,
) -> dict[str, Any]:
    web_root = Path(__file__).resolve().parents[1] / "dashboard" / "web"
    navigation_timeout_ms = max(int(float(timeout_seconds) * 1000), MIN_BROWSER_RENDER_TIMEOUT_MS)
    render_timeout_ms = max(navigation_timeout_ms, MIN_BROWSER_RENDER_TIMEOUT_MS)
    screenshot_path = screenshot_path.expanduser().resolve()
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fenlie-public-overview-smoke-", dir=web_root) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        spec_path = temp_dir / "public-overview.smoke.spec.cjs"
        result_path = temp_dir / "public-overview.result.json"
        spec_path.write_text(
            build_public_route_browser_spec(
                root_url=root_url,
                route_path=route_path,
                markers=markers,
                navigation_timeout_ms=navigation_timeout_ms,
                render_timeout_ms=render_timeout_ms,
                result_path=result_path,
                screenshot_path=screenshot_path,
            ),
            encoding="utf-8",
        )
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
            str(render_timeout_ms),
        ]
        proc = subprocess.run(cmd, cwd=web_root, text=True, capture_output=True, check=False)
        result_payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        return {
            "name": name,
            "route": str(result_payload.get("route") or f"{root_url.rstrip('/')}/#/{route_path}"),
            "final_url": str(result_payload.get("final_url") or ""),
            "markers": list(result_payload.get("markers") or markers),
            "screenshot_path": str(result_payload.get("screenshot_path") or screenshot_path),
            "ok": proc.returncode == 0 and bool(result_payload),
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "").strip().splitlines()[-10:],
            "stderr_tail": (proc.stderr or "").strip().splitlines()[-10:],
        }


def run_public_overview_browser_smoke(
    *,
    name: str,
    root_url: str,
    timeout_seconds: float,
    screenshot_path: Path,
) -> dict[str, Any]:
    return run_public_route_browser_smoke(
        name=name,
        root_url=root_url,
        route_path=PUBLIC_OVERVIEW_ROUTE_PATH,
        markers=PUBLIC_OVERVIEW_MARKERS,
        timeout_seconds=timeout_seconds,
        screenshot_path=screenshot_path,
    )


def run_topology_smoke(
    *,
    root_url: str,
    pages_url: str,
    openclaw_url: str,
    gateway_url: str,
    timeout_seconds: float,
    artifact_dir: Path | None = None,
    artifact_prefix: str = "dashboard_public_topology",
    allow_insecure_tls_fallback: bool = False,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.expanduser().resolve() if artifact_dir is not None else Path.cwd()
    root_overview_screenshot_path = artifact_dir / f"{artifact_prefix}_root_overview_browser.png"
    pages_overview_screenshot_path = artifact_dir / f"{artifact_prefix}_pages_overview_browser.png"
    root_contracts_screenshot_path = artifact_dir / f"{artifact_prefix}_root_contracts_browser.png"
    pages_contracts_screenshot_path = artifact_dir / f"{artifact_prefix}_pages_contracts_browser.png"
    root_public = probe_text_endpoint(
        name="root_public",
        url=root_url,
        timeout_seconds=timeout_seconds,
        allow_insecure_tls_fallback=allow_insecure_tls_fallback,
    )
    pages_public = probe_text_endpoint(
        name="pages_public",
        url=pages_url,
        timeout_seconds=timeout_seconds,
        allow_insecure_tls_fallback=allow_insecure_tls_fallback,
    )
    openclaw_legacy = probe_text_endpoint(
        name="openclaw_legacy",
        url=openclaw_url,
        timeout_seconds=timeout_seconds,
        allow_insecure_tls_fallback=allow_insecure_tls_fallback,
    )
    gateway = probe_text_endpoint(
        name="gateway",
        url=gateway_url,
        timeout_seconds=timeout_seconds,
        allow_insecure_tls_fallback=allow_insecure_tls_fallback,
    )
    root_snapshot = probe_json_endpoint(
        name="root_snapshot",
        url=f"{root_url.rstrip('/')}/data/fenlie_dashboard_snapshot.json",
        timeout_seconds=timeout_seconds,
        allow_insecure_tls_fallback=allow_insecure_tls_fallback,
    )
    pages_snapshot = probe_json_endpoint(
        name="pages_snapshot",
        url=f"{pages_url.rstrip('/')}/data/fenlie_dashboard_snapshot.json",
        timeout_seconds=timeout_seconds,
        allow_insecure_tls_fallback=allow_insecure_tls_fallback,
    )
    root_overview_browser = run_public_overview_browser_smoke(
        name="root_overview_browser",
        root_url=root_url,
        timeout_seconds=timeout_seconds,
        screenshot_path=root_overview_screenshot_path,
    )
    pages_overview_browser = run_public_overview_browser_smoke(
        name="pages_overview_browser",
        root_url=pages_url,
        timeout_seconds=timeout_seconds,
        screenshot_path=pages_overview_screenshot_path,
    )
    root_contracts_browser = run_public_route_browser_smoke(
        name="root_contracts_browser",
        root_url=root_url,
        route_path=PUBLIC_CONTRACTS_ROUTE_PATH,
        markers=PUBLIC_CONTRACTS_MARKERS,
        timeout_seconds=timeout_seconds,
        screenshot_path=root_contracts_screenshot_path,
    )
    pages_contracts_browser = run_public_route_browser_smoke(
        name="pages_contracts_browser",
        root_url=pages_url,
        route_path=PUBLIC_CONTRACTS_ROUTE_PATH,
        markers=PUBLIC_CONTRACTS_MARKERS,
        timeout_seconds=timeout_seconds,
        screenshot_path=pages_contracts_screenshot_path,
    )

    root_public["header_public_entry"] = root_public.get("headers", {}).get("x-fenlie-public-entry", "")
    root_snapshot["frontend_public"] = (
        (root_snapshot.get("payload") or {}).get("ui_routes") or {}
    ).get("frontend_public")
    pages_snapshot["frontend_public"] = (
        (pages_snapshot.get("payload") or {}).get("ui_routes") or {}
    ).get("frontend_public")
    root_snapshot["payload_summary"] = {"frontend_public": root_snapshot["frontend_public"]}
    pages_snapshot["payload_summary"] = {"frontend_public": pages_snapshot["frontend_public"]}
    root_snapshot.pop("payload", None)
    pages_snapshot.pop("payload", None)

    checks = {
        "root_public": root_public,
        "pages_public": pages_public,
        "root_snapshot": root_snapshot,
        "pages_snapshot": pages_snapshot,
        "openclaw_legacy": openclaw_legacy,
        "gateway": gateway,
        "root_overview_browser": root_overview_browser,
        "pages_overview_browser": pages_overview_browser,
        "root_contracts_browser": root_contracts_browser,
        "pages_contracts_browser": pages_contracts_browser,
    }
    checks["ok"] = all(
        [
            root_public["status_code"] == 200,
            "Fenlie" in root_public["title"],
            root_public["header_public_entry"] == "root-nav-proxy",
            pages_public["status_code"] == 200,
            "Fenlie" in pages_public["title"],
            root_snapshot["status_code"] == 200,
            root_snapshot["frontend_public"] == DEFAULT_ROOT_URL,
            pages_snapshot["status_code"] == 200,
            pages_snapshot["frontend_public"] == DEFAULT_ROOT_URL,
            openclaw_legacy["status_code"] == 200,
            openclaw_legacy["title"] == "OpenClaw Control",
            gateway["status_code"] == 404,
            bool(root_overview_browser.get("ok")),
            bool(pages_overview_browser.get("ok")),
            bool(root_contracts_browser.get("ok")),
            bool(pages_contracts_browser.get("ok")),
        ]
    )
    return checks


def build_report_payload(
    *,
    workspace: Path,
    report_path: Path,
    checks: dict[str, Any],
    root_url: str,
    pages_url: str,
    openclaw_url: str,
    gateway_url: str,
) -> dict[str, Any]:
    return {
        "action": "dashboard_public_topology_smoke",
        "ok": bool(checks.get("ok")),
        "status": "ok" if checks.get("ok") else "failed",
        "change_class": CHANGE_CLASS,
        "generated_at_utc": fmt_utc(now_utc()),
        "workspace": str(workspace),
        "report_path": str(report_path),
        "entrypoints": {
            "root_url": root_url,
            "pages_url": pages_url,
            "openclaw_url": openclaw_url,
            "gateway_url": gateway_url,
        },
        "expectations": {
            "root_title_contains": "Fenlie",
            "root_header_public_entry": "root-nav-proxy",
            "frontend_public": DEFAULT_ROOT_URL,
            "overview_markers": PUBLIC_OVERVIEW_MARKERS,
            "contracts_markers": PUBLIC_CONTRACTS_MARKERS,
            "legacy_openclaw_title": "OpenClaw Control",
            "gateway_status_code": 404,
        },
        "tls_policy": {
            "allow_insecure_tls_fallback": any(
                item.get("tls_verification") == "insecure_fallback"
                for item in checks.values()
                if isinstance(item, dict)
            ),
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check Fenlie public topology and legacy OpenClaw fallback endpoints.")
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--root-url", default=DEFAULT_ROOT_URL, help="Root public Fenlie entrypoint.")
    parser.add_argument("--pages-url", default=DEFAULT_PAGES_URL, help="Cloudflare Pages fallback entrypoint.")
    parser.add_argument("--openclaw-url", default=DEFAULT_OPENCLAW_URL, help="Legacy OpenClaw entrypoint.")
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL, help="Gateway/API entrypoint.")
    parser.add_argument("--timeout-seconds", type=float, default=5.0, help="Single-request timeout, capped at 5 seconds.")
    parser.add_argument("--allow-insecure-tls-fallback", action="store_true", help="Retry HTTPS probes with unverified TLS when local redirected domains fail strict cert validation.")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    report_path = review_dir / f"{timestamp}_dashboard_public_topology_smoke.json"
    checks = run_topology_smoke(
        root_url=args.root_url,
        pages_url=args.pages_url,
        openclaw_url=args.openclaw_url,
        gateway_url=args.gateway_url,
        timeout_seconds=args.timeout_seconds,
        artifact_dir=review_dir,
        artifact_prefix=f"{timestamp}_dashboard_public_topology",
        allow_insecure_tls_fallback=args.allow_insecure_tls_fallback,
    )
    payload = build_report_payload(
        workspace=workspace,
        report_path=report_path,
        checks=checks,
        root_url=args.root_url,
        pages_url=args.pages_url,
        openclaw_url=args.openclaw_url,
        gateway_url=args.gateway_url,
    )
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
