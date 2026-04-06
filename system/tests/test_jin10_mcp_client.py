from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "lie_engine" / "research" / "jin10_mcp_client.py"


def load_module():
    spec = importlib.util.spec_from_file_location("jin10_mcp_client_module", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyResponse(io.BytesIO):
    def __init__(self, payload: dict[str, object], *, headers: dict[str, str] | None = None, status: int = 200):
        super().__init__(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        self.headers = headers or {}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_client_initialize_persists_session_and_uses_it_for_followup_calls(monkeypatch) -> None:
    mod = load_module()
    requests: list[tuple[str, str, dict[str, str], dict[str, object]]] = []

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        payload = json.loads((request.data or b"{}").decode("utf-8"))
        headers = {k: v for k, v in request.header_items()}
        requests.append((request.get_method(), request.full_url, headers, payload))
        if payload["method"] == "initialize":
            return DummyResponse(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "serverInfo": {"name": "jin10", "version": "1.0.0"},
                        "capabilities": {"tools": {}, "resources": {}},
                    },
                },
                headers={"mcp-session-id": "session-123"},
            )
        if payload["method"] == "tools/list":
            return DummyResponse(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"tools": [{"name": "calendar.latest"}]},
                }
            )
        raise AssertionError(payload["method"])

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    client = mod.Jin10McpClient(server_url="https://mcp.jin10.com/mcp", bearer_token="secret-token")
    init = client.initialize()
    tools = client.list_tools()

    assert init["serverInfo"]["name"] == "jin10"
    assert tools == [{"name": "calendar.latest"}]
    assert client.session_id == "session-123"

    first = requests[0]
    assert first[0] == "POST"
    assert first[1] == "https://mcp.jin10.com/mcp"
    assert first[2]["Authorization"] == "Bearer secret-token"
    assert "application/json" in first[2]["Accept"]
    assert "text/event-stream" in first[2]["Accept"]
    assert first[3]["method"] == "initialize"
    assert first[3]["params"]["clientInfo"]["name"] == "fenlie-jin10-sidecar"

    second = requests[1]
    assert second[2]["Mcp-session-id"] == "session-123"
    assert second[3]["method"] == "tools/list"


def test_client_surfaces_http_401_as_auth_failed(monkeypatch) -> None:
    mod = load_module()

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(json.dumps({"error": {"message": "bad token"}}).encode("utf-8")),
        )

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    client = mod.Jin10McpClient(server_url="https://mcp.jin10.com/mcp", bearer_token="secret-token")

    with pytest.raises(mod.Jin10McpError, match="auth_failed"):
        client.initialize()


def test_client_parses_streamable_http_sse_payload(monkeypatch) -> None:
    mod = load_module()

    sse_body = (
        "event: message\n"
        "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"protocolVersion\":\"2025-03-26\",\"serverInfo\":{\"name\":\"jin10\",\"version\":\"1.0.0\"},\"capabilities\":{\"tools\":{},\"resources\":{}}}}\n\n"
    )

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        return DummyResponse({}, headers={"mcp-session-id": "session-123"}) if False else io.BytesIO(sse_body.encode("utf-8"))

    class SseResponse(io.BytesIO):
        def __init__(self, text: str):
            super().__init__(text.encode("utf-8"))
            self.headers = {"mcp-session-id": "session-123"}
            self.status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda request, timeout=0: SseResponse(sse_body))

    client = mod.Jin10McpClient(server_url="https://mcp.jin10.com/mcp", bearer_token="secret-token")
    init = client.initialize()

    assert init["serverInfo"]["name"] == "jin10"
    assert client.session_id == "session-123"
