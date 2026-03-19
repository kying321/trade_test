from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import uuid


RUNTIME_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "pi" / "scripts"


class _DummyResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def _load_runtime_module(script_name: str):
    script_path = RUNTIME_SCRIPTS_ROOT / script_name
    module_name = f"{script_name.replace('.', '_')}_{uuid.uuid4().hex}"
    sys.path.insert(0, str(RUNTIME_SCRIPTS_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)
        sys.path.pop(0)


def test_binance_spot_exec_caps_proxy_bypass_timeout_to_five_seconds(monkeypatch) -> None:
    monkeypatch.setenv("BINANCE_HTTP_TIMEOUT_SECONDS", "15")
    mod = _load_runtime_module("binance_spot_exec.py")
    seen: dict[str, object] = {}

    def _fake_request(method, url, no_proxy_request_func=None, **kwargs):
        seen["method"] = method
        seen["url"] = url
        seen["timeout"] = kwargs.get("timeout")
        seen["no_proxy_request_func"] = no_proxy_request_func
        return _DummyResponse({"serverTime": 1234567890})

    monkeypatch.setattr(mod, "_net_request_with_proxy_bypass", _fake_request)

    mod.get_server_time_ms("https://api.binance.com")

    assert mod.HTTP_TIMEOUT_SECONDS == 5.0
    assert seen["method"] == "GET"
    assert seen["url"] == "https://api.binance.com/api/v3/time"
    assert seen["timeout"] == 5.0
    assert callable(seen["no_proxy_request_func"])


def test_lie_spot_halfhour_core_caps_http_timeout_to_five_seconds(monkeypatch) -> None:
    monkeypatch.setenv("LIE_HTTP_TIMEOUT_SECONDS", "15")
    mod = _load_runtime_module("lie_spot_halfhour_core.py")
    seen: dict[str, object] = {}

    def _fake_request(method, url, no_proxy_request_func=None, **kwargs):
        seen["method"] = method
        seen["url"] = url
        seen["timeout"] = kwargs.get("timeout")
        seen["no_proxy_request_func"] = no_proxy_request_func
        return _DummyResponse({"ok": True})

    monkeypatch.setattr(mod, "_net_request_with_proxy_bypass", _fake_request)

    mod._http("GET", "https://example.com/test", timeout=10)

    assert mod.HTTP_TIMEOUT_SECONDS == 5.0
    assert seen["method"] == "GET"
    assert seen["url"] == "https://example.com/test"
    assert seen["timeout"] == 5.0
    assert callable(seen["no_proxy_request_func"])
