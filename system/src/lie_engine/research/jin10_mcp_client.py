from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_PROTOCOL_VERSION = "2025-03-26"


@dataclass
class Jin10McpError(RuntimeError):
    code: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.code}:{self.detail}" if self.detail else self.code


class Jin10McpClient:
    def __init__(
        self,
        *,
        server_url: str,
        bearer_token: str,
        timeout_seconds: float = 5.0,
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        client_name: str = "fenlie-jin10-sidecar",
        client_version: str = "1.0.0",
    ) -> None:
        self.server_url = str(server_url).strip()
        self.bearer_token = str(bearer_token).strip()
        self.timeout_seconds = float(timeout_seconds)
        self.protocol_version = str(protocol_version).strip() or DEFAULT_PROTOCOL_VERSION
        self.client_name = str(client_name).strip() or "fenlie-jin10-sidecar"
        self.client_version = str(client_version).strip() or "1.0.0"
        self._next_id = 1
        self.session_id = ""
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.bearer_token}",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _decode_error_payload(self, exc: urllib.error.HTTPError) -> str:
        if exc.fp is None:
            return ""
        try:
            raw = exc.fp.read().decode("utf-8")
        except Exception:
            return ""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return raw.strip()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or "").strip()
            return str(payload.get("message") or "").strip()
        return ""

    def _decode_message(self, raw: str) -> dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            raise Jin10McpError("invalid_json_payload", "empty_response")
        if text.startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise Jin10McpError("invalid_json_payload", text[:200]) from exc
            if not isinstance(payload, dict):
                raise Jin10McpError("invalid_json_payload", "non_mapping_response")
            return payload

        data_chunks: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                data_chunks.append(stripped[len("data:") :].strip())
                joined = "\n".join(data_chunks).strip()
                if not joined:
                    continue
                try:
                    payload = json.loads(joined)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    raise Jin10McpError("invalid_json_payload", "non_mapping_sse_payload")
                return payload
        raise Jin10McpError("invalid_json_payload", text[:200])

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id,
            "method": str(method),
            "params": params or {},
        }
        self._next_id += 1
        request = urllib.request.Request(
            self.server_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
                for key in ("mcp-session-id", "Mcp-Session-Id", "x-mcp-session-id"):
                    value = response.headers.get(key)
                    if value:
                        self.session_id = str(value).strip()
                        break
        except urllib.error.HTTPError as exc:
            detail = self._decode_error_payload(exc) or exc.reason or ""
            if exc.code in (401, 403):
                raise Jin10McpError("auth_failed", str(detail)) from exc
            raise Jin10McpError(f"http_error:{exc.code}", str(detail)) from exc
        except urllib.error.URLError as exc:
            raise Jin10McpError("transport_error", str(getattr(exc, "reason", exc))) from exc

        message = self._decode_message(raw)
        if message.get("error"):
            error = message.get("error")
            if isinstance(error, dict):
                raise Jin10McpError("rpc_error", str(error.get("message") or error))
            raise Jin10McpError("rpc_error", str(error))
        result = message.get("result")
        if not isinstance(result, dict):
            raise Jin10McpError("invalid_result", str(type(result)))
        return result

    def initialize(self) -> dict[str, Any]:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
        )
        self.server_info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else {}
        self.capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), dict) else {}
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._rpc("tools/list")
        tools = result.get("tools")
        return [row for row in tools if isinstance(row, dict)] if isinstance(tools, list) else []

    def list_resources(self) -> list[dict[str, Any]]:
        result = self._rpc("resources/list")
        resources = result.get("resources")
        return [row for row in resources if isinstance(row, dict)] if isinstance(resources, list) else []

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._rpc(
            "tools/call",
            {
                "name": str(name),
                "arguments": arguments or {},
            },
        )

    def read_resource(self, uri: str) -> dict[str, Any]:
        return self._rpc(
            "resources/read",
            {
                "uri": str(uri),
            },
        )
