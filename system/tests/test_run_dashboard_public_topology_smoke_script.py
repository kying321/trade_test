from __future__ import annotations

import importlib.util
import http.client
import json
import ssl
import sys
import urllib.error
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_dashboard_public_topology_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dashboard_public_topology_smoke_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_topology_smoke_reports_expected_public_entry(monkeypatch) -> None:
    mod = load_module()
    screenshot_calls: list[dict[str, str]] = []

    html_by_url = {
        "https://fuuu.fun": "<html><head><title>Fenlie 终端控制台</title></head><body></body></html>",
        "https://fenlie.fuuu.fun": "<html><head><title>Fenlie 终端控制台</title></head><body></body></html>",
        "http://43.153.148.242:3001": "<html><head><title>OpenClaw Control</title></head><body></body></html>",
        "http://43.153.148.242:8787": "Not Found",
    }
    json_by_url = {
        "https://fuuu.fun/data/fenlie_dashboard_snapshot.json": {"ui_routes": {"frontend_public": "https://fuuu.fun"}},
        "https://fenlie.fuuu.fun/data/fenlie_dashboard_snapshot.json": {"ui_routes": {"frontend_public": "https://fuuu.fun"}},
    }

    def fake_probe_text(*, name: str, url: str, timeout_seconds: float, allow_insecure_tls_fallback: bool = False):
        return {
            "name": name,
            "url": url,
            "status_code": 200 if "8787" not in url else 404,
            "headers": {"x-fenlie-public-entry": "root-nav-proxy"} if url == "https://fuuu.fun" else {},
            "body_text": html_by_url[url],
            "title": mod.extract_html_title(html_by_url[url]),
            "ok": True,
            "timeout_seconds": timeout_seconds,
            "tls_verification": "strict",
        }

    def fake_probe_json(*, name: str, url: str, timeout_seconds: float, allow_insecure_tls_fallback: bool = False):
        return {
            "name": name,
            "url": url,
            "status_code": 200,
            "payload": json_by_url[url],
            "ok": True,
            "timeout_seconds": timeout_seconds,
            "tls_verification": "strict",
        }

    def fake_browser_smoke(
        *,
        name: str,
        root_url: str,
        route_path: str,
        markers: list[str],
        timeout_seconds: float,
        screenshot_path: Path,
    ):
        screenshot_calls.append(
            {
                "name": name,
                "root_url": root_url,
                "route_path": route_path,
                "markers": json.dumps(markers, ensure_ascii=False),
                "screenshot_path": str(screenshot_path),
            }
        )
        return {
            "name": name,
            "route": f"{root_url}/{route_path}",
            "final_url": f"{root_url}/{route_path}",
            "markers": markers,
            "screenshot_path": str(screenshot_path),
            "ok": True,
            "returncode": 0,
        }

    monkeypatch.setattr(mod, "probe_text_endpoint", fake_probe_text)
    monkeypatch.setattr(mod, "probe_json_endpoint", fake_probe_json)
    monkeypatch.setattr(
        mod,
        "run_public_overview_browser_smoke",
        lambda *, name, root_url, timeout_seconds, screenshot_path: fake_browser_smoke(
            name=name,
            root_url=root_url,
            route_path="ops/overview",
            markers=["研究主线摘要", "国内商品推理线"],
            timeout_seconds=timeout_seconds,
            screenshot_path=screenshot_path,
        ),
    )
    monkeypatch.setattr(mod, "run_public_route_browser_smoke", fake_browser_smoke)

    checks = mod.run_topology_smoke(
        root_url="https://fuuu.fun",
        pages_url="https://fenlie.fuuu.fun",
        openclaw_url="http://43.153.148.242:3001",
        gateway_url="http://43.153.148.242:8787",
        timeout_seconds=3.0,
        artifact_dir=Path("/tmp/fenlie-review"),
        artifact_prefix="20260321T095337Z_dashboard_public_topology",
    )

    assert checks["root_public"]["title"] == "Fenlie 终端控制台"
    assert checks["root_public"]["header_public_entry"] == "root-nav-proxy"
    assert checks["root_snapshot"]["frontend_public"] == "https://fuuu.fun"
    assert checks["pages_snapshot"]["frontend_public"] == "https://fuuu.fun"
    assert checks["openclaw_legacy"]["title"] == "OpenClaw Control"
    assert checks["gateway"]["status_code"] == 404
    assert checks["root_overview_browser"]["name"] == "root_overview_browser"
    assert checks["pages_overview_browser"]["name"] == "pages_overview_browser"
    assert checks["root_contracts_browser"]["name"] == "root_contracts_browser"
    assert checks["pages_contracts_browser"]["name"] == "pages_contracts_browser"
    assert checks["root_overview_browser"]["route"] == "https://fuuu.fun/ops/overview"
    assert checks["pages_overview_browser"]["route"] == "https://fenlie.fuuu.fun/ops/overview"
    assert checks["root_contracts_browser"]["route"] == "https://fuuu.fun/ops/audits"
    assert checks["pages_contracts_browser"]["route"] == "https://fenlie.fuuu.fun/ops/audits"
    assert checks["root_overview_browser"]["markers"] == ["研究主线摘要", "国内商品推理线"]
    assert checks["root_contracts_browser"]["markers"] == [
        "公开入口拓扑",
        "公开面验收",
        "穿透层 1 / 验收总览",
        "穿透层 2 / 子链状态",
    ]
    assert checks["root_overview_browser"]["screenshot_path"].endswith(
        "20260321T095337Z_dashboard_public_topology_root_overview_browser.png"
    )
    assert checks["pages_overview_browser"]["screenshot_path"].endswith(
        "20260321T095337Z_dashboard_public_topology_pages_overview_browser.png"
    )
    assert checks["root_contracts_browser"]["screenshot_path"].endswith(
        "20260321T095337Z_dashboard_public_topology_root_contracts_browser.png"
    )
    assert checks["pages_contracts_browser"]["screenshot_path"].endswith(
        "20260321T095337Z_dashboard_public_topology_pages_contracts_browser.png"
    )
    assert [call["name"] for call in screenshot_calls] == [
        "root_overview_browser",
        "pages_overview_browser",
        "root_contracts_browser",
        "pages_contracts_browser",
    ]
    assert [call["root_url"] for call in screenshot_calls] == [
        "https://fuuu.fun",
        "https://fenlie.fuuu.fun",
        "https://fuuu.fun",
        "https://fenlie.fuuu.fun",
    ]
    assert [call["route_path"] for call in screenshot_calls] == [
        "ops/overview",
        "ops/overview",
        "ops/audits",
        "ops/audits",
    ]
    assert screenshot_calls[0]["screenshot_path"].endswith(
        "fenlie-review/20260321T095337Z_dashboard_public_topology_root_overview_browser.png"
    )
    assert screenshot_calls[1]["screenshot_path"].endswith(
        "fenlie-review/20260321T095337Z_dashboard_public_topology_pages_overview_browser.png"
    )
    assert (
        screenshot_calls[2]["markers"]
        == '["公开入口拓扑", "公开面验收", "穿透层 1 / 验收总览", "穿透层 2 / 子链状态"]'
    )
    assert screenshot_calls[2]["screenshot_path"].endswith(
        "fenlie-review/20260321T095337Z_dashboard_public_topology_root_contracts_browser.png"
    )
    assert screenshot_calls[3]["screenshot_path"].endswith(
        "fenlie-review/20260321T095337Z_dashboard_public_topology_pages_contracts_browser.png"
    )
    assert checks["ok"] is True


def test_build_public_route_browser_spec_persists_screenshot(tmp_path: Path) -> None:
    mod = load_module()

    result_path = tmp_path / "public-overview.result.json"
    screenshot_path = tmp_path / "public-overview.png"
    spec = mod.build_public_route_browser_spec(
        root_url="https://fuuu.fun",
        route_path="workspace/contracts",
        markers=[
            "公开面验收",
            "穿透层 1 / 验收总览",
            "接口目录",
            "源头主线",
            "回退链",
        ],
        navigation_timeout_ms=5000,
        render_timeout_ms=15000,
        result_path=result_path,
        screenshot_path=screenshot_path,
    )

    assert "page.screenshot" in spec
    assert str(screenshot_path) in spec
    assert str(result_path) in spec
    assert "workspace/contracts" in spec
    assert "穿透层 1 / 验收总览" in spec
    assert "接口目录" in spec
    assert "源头主线" in spec
    assert "回退链" in spec
    assert "NAVIGATION_TIMEOUT_MS = 5000" in spec
    assert "RENDER_TIMEOUT_MS = 15000" in spec


def test_run_public_route_browser_smoke_uses_uncapped_navigation_timeout(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    captured: dict[str, str] = {}

    class DummyProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, cwd, text, capture_output, check):  # noqa: ANN001
        spec_path = Path(cmd[3])
        temp_dir = Path(cmd[5])
        captured["spec"] = spec_path.read_text(encoding="utf-8")
        (temp_dir / "public-overview.result.json").write_text(
            json.dumps(
              {
                "route": "https://fuuu.fun/ops/overview",
                "final_url": "https://fuuu.fun/ops/overview",
                "markers": ["研究主线摘要", "国内商品推理线"],
                "screenshot_path": str(tmp_path / "shot.png"),
              },
              ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return DummyProc()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = mod.run_public_route_browser_smoke(
        name="root_overview_browser",
        root_url="https://fuuu.fun",
        route_path="ops/overview",
        markers=["研究主线摘要", "国内商品推理线"],
        timeout_seconds=30.0,
        screenshot_path=tmp_path / "shot.png",
    )

    assert result["ok"] is True
    assert "NAVIGATION_TIMEOUT_MS = 30000" in captured["spec"]


def test_run_public_route_browser_smoke_keeps_a_30s_browser_timeout_floor(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    captured: dict[str, str] = {}

    class DummyProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, cwd, text, capture_output, check):  # noqa: ANN001
        spec_path = Path(cmd[3])
        temp_dir = Path(cmd[5])
        captured["spec"] = spec_path.read_text(encoding="utf-8")
        (temp_dir / "public-overview.result.json").write_text(
            json.dumps(
                {
                    "route": "https://fuuu.fun/ops/overview",
                    "final_url": "https://fuuu.fun/ops/overview",
                    "markers": ["研究主线摘要", "国内商品推理线"],
                    "screenshot_path": str(tmp_path / "shot.png"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return DummyProc()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result = mod.run_public_route_browser_smoke(
        name="root_overview_browser",
        root_url="https://fuuu.fun",
        route_path="ops/overview",
        markers=["研究主线摘要", "国内商品推理线"],
        timeout_seconds=5.0,
        screenshot_path=tmp_path / "shot.png",
    )

    assert result["ok"] is True
    assert "NAVIGATION_TIMEOUT_MS = 30000" in captured["spec"]


def test_probe_text_endpoint_can_retry_with_insecure_tls(monkeypatch) -> None:
    mod = load_module()

    class DummyResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = {"content-type": "text/html"}

        def read(self) -> bytes:
            return b"<html><head><title>Fenlie \xe7\xbb\x88\xe7\xab\xaf\xe6\x8e\xa7\xe5\x88\xb6\xe5\x8f\xb0</title></head><body></body></html>"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls: list[bool] = []

    class DummyOpener:
        def __init__(self, *, insecure: bool) -> None:
            self.insecure = insecure

        def open(self, request, timeout=0):  # noqa: ANN001
            calls.append(self.insecure)
            if not self.insecure:
                raise urllib.error.URLError(ssl.SSLError("certificate verify failed"))
            return DummyResponse()

    monkeypatch.setattr(
        mod.urllib.request,
        "build_opener",
        lambda *args: DummyOpener(insecure=len(args) > 1),
    )

    payload = mod.probe_text_endpoint(
        name="root_public",
        url="https://fuuu.fun",
        timeout_seconds=3.0,
        allow_insecure_tls_fallback=True,
    )

    assert calls == [False, True]
    assert payload["status_code"] == 200
    assert payload["tls_verification"] == "insecure_fallback"
    assert payload["title"] == "Fenlie 终端控制台"


def test_probe_text_endpoint_retries_transient_eof_after_insecure_tls_fallback(monkeypatch) -> None:
    mod = load_module()

    class DummyResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = {"content-type": "text/html"}

        def read(self) -> bytes:
            return b"<html><head><title>Fenlie \xe7\xbb\x88\xe7\xab\xaf\xe6\x8e\xa7\xe5\x88\xb6\xe5\x8f\xb0</title></head><body></body></html>"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls: list[bool] = []

    class DummyOpener:
        def __init__(self, *, insecure: bool) -> None:
            self.insecure = insecure

        def open(self, request, timeout=0):  # noqa: ANN001
            calls.append(self.insecure)
            if not self.insecure:
                raise urllib.error.URLError(ssl.SSLError("certificate verify failed"))
            if len(calls) == 2:
                raise urllib.error.URLError(ssl.SSLEOFError("EOF occurred in violation of protocol"))
            return DummyResponse()

    monkeypatch.setattr(
        mod.urllib.request,
        "build_opener",
        lambda *args: DummyOpener(insecure=len(args) > 1),
    )

    payload = mod.probe_text_endpoint(
        name="pages_public",
        url="https://fenlie.fuuu.fun",
        timeout_seconds=3.0,
        allow_insecure_tls_fallback=True,
    )

    assert calls == [False, True, True]
    assert payload["status_code"] == 200
    assert payload["tls_verification"] == "insecure_fallback"
    assert payload["title"] == "Fenlie 终端控制台"


def test_request_url_retries_transient_remote_disconnect(monkeypatch) -> None:
    mod = load_module()

    class DummyResponse:
        def __init__(self) -> None:
            self.status = 200
            self.headers = {"content-type": "text/html"}

        def read(self) -> bytes:
            return b"<html><head><title>OpenClaw Control</title></head><body></body></html>"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    calls: list[int] = []

    class DummyOpener:
        def open(self, request, timeout=0):  # noqa: ANN001
            calls.append(timeout)
            if len(calls) == 1:
                raise http.client.RemoteDisconnected("Remote end closed connection without response")
            return DummyResponse()

    monkeypatch.setattr(mod.urllib.request, "build_opener", lambda *args: DummyOpener())

    status_code, headers, body, tls_verification = mod._request_url(
        "http://43.153.148.242:3001",
        timeout_seconds=3.0,
    )

    assert len(calls) == 2
    assert status_code == 200
    assert headers["content-type"] == "text/html"
    assert "OpenClaw Control" in body
    assert tls_verification == "strict"


def test_main_writes_review_artifact(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    review_dir.mkdir(parents=True)

    monkeypatch.setattr(mod, "resolve_system_root", lambda value: workspace / "system")
    monkeypatch.setattr(mod, "now_utc", lambda: mod.dt.datetime(2026, 3, 20, 6, 5, tzinfo=mod.dt.timezone.utc))
    monkeypatch.setattr(
        mod,
        "run_topology_smoke",
        lambda **_: {
            "ok": True,
            "root_public": {"title": "Fenlie 终端控制台", "header_public_entry": "root-nav-proxy", "status_code": 200},
            "pages_public": {"title": "Fenlie 终端控制台", "status_code": 200},
            "root_snapshot": {"frontend_public": "https://fuuu.fun", "status_code": 200},
            "pages_snapshot": {"frontend_public": "https://fuuu.fun", "status_code": 200},
            "openclaw_legacy": {"title": "OpenClaw Control", "status_code": 200},
            "gateway": {"status_code": 404},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dashboard_public_topology_smoke.py",
            "--workspace",
            str(workspace),
        ],
    )

    mod.main()

    report_path = review_dir / "20260320T060500Z_dashboard_public_topology_smoke.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["change_class"] == "RESEARCH_ONLY"
    assert payload["checks"]["root_public"]["header_public_entry"] == "root-nav-proxy"
