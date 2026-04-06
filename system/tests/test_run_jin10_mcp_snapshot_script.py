from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_jin10_mcp_snapshot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_jin10_mcp_snapshot_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_snapshot_marks_missing_token_as_blocked_auth(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"

    result = mod.run_snapshot(
        workspace=workspace,
        server_url="https://mcp.jin10.com/mcp",
        bearer_token="",
        token_source="JIN10_MCP_BEARER_TOKEN",
    )

    assert result["ok"] is False
    assert result["status"] == "blocked_auth_missing"
    assert result["change_class"] == "RESEARCH_ONLY"
    assert result["mode"] == "jin10_mcp_snapshot"
    assert result["token_source"] == "JIN10_MCP_BEARER_TOKEN"
    assert Path(result["artifact_json"]).exists()
    assert Path(result["artifact_md"]).exists()
    latest_json = review_dir / "latest_jin10_mcp_snapshot.json"
    latest_md = review_dir / "latest_jin10_mcp_snapshot.md"
    assert latest_json.exists()
    assert latest_md.exists()


def test_run_snapshot_writes_inventory_and_optional_tool_call(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def initialize(self):
            return {
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "jin10", "version": "1.0.0"},
                "capabilities": {"tools": {}, "resources": {}},
            }

        def list_tools(self):
            return [{"name": "calendar.latest", "description": "latest calendar"}]

        def list_resources(self):
            return [{"uri": "jin10://calendar/latest", "name": "calendar"}]

        def call_tool(self, name: str, arguments: dict[str, object] | None = None):
            return {"content": [{"type": "text", "text": f"{name}:{json.dumps(arguments or {}, ensure_ascii=False)}"}]}

    monkeypatch.setattr(mod, "Jin10McpClient", FakeClient)
    monkeypatch.setattr(mod, "now_utc", lambda: mod.dt.datetime(2026, 4, 6, 12, 0, tzinfo=mod.dt.timezone.utc))

    result = mod.run_snapshot(
        workspace=workspace,
        server_url="https://mcp.jin10.com/mcp",
        bearer_token="secret-token",
        token_source="JIN10_MCP_BEARER_TOKEN",
        summary_profile="none",
        tool_name="calendar.latest",
        tool_args={"symbols": ["XAUUSD"]},
    )

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["tools_total"] == 1
    assert result["resources_total"] == 1
    assert result["server_info"] == {"name": "jin10", "version": "1.0.0"}
    assert result["tool_call"]["name"] == "calendar.latest"
    assert result["tool_call"]["arguments"] == {"symbols": ["XAUUSD"]}
    assert "calendar.latest" in result["tool_call"]["result_preview"]
    payload = json.loads(Path(result["artifact_json"]).read_text(encoding="utf-8"))
    assert payload["tool_call"]["name"] == "calendar.latest"
    assert payload["tools"][0]["name"] == "calendar.latest"


def test_run_snapshot_builds_standard_summary_from_calendar_flash_and_quotes(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def initialize(self):
            return {
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "jin10", "version": "1.0.0"},
                "capabilities": {"tools": {}, "resources": {}},
            }

        def list_tools(self):
            return [
                {"name": "list_calendar"},
                {"name": "list_flash"},
                {"name": "get_quote"},
            ]

        def list_resources(self):
            return [{"uri": "quote://codes", "name": "quote_codes"}]

        def call_tool(self, name: str, arguments: dict[str, object] | None = None):
            if name == "list_calendar":
                return {
                    "data": [
                        {"title": "美国非农就业人口", "pub_time": "2026-04-10T12:30:00Z", "star": 5},
                        {"title": "中国CPI年率", "pub_time": "2026-04-09T01:30:00Z", "star": 3},
                    ],
                    "status": 200,
                    "message": "ok",
                }
            if name == "list_flash":
                return {
                    "data": {
                        "items": [
                            {"time": "2026-04-06T12:01:00Z", "content": "金价短线拉升", "url": "https://x/1"},
                            {"time": "2026-04-06T11:59:00Z", "content": "美元指数走弱", "url": "https://x/2"},
                        ],
                        "next_cursor": "",
                        "has_more": False,
                    },
                    "status": 200,
                    "message": "ok",
                }
            if name == "get_quote":
                code = str((arguments or {}).get("code") or "")
                return {
                    "data": {
                        "code": code,
                        "name": "现货黄金" if code == "XAUUSD" else code,
                        "close": "2325.1",
                        "ups_percent": "0.31",
                        "time": "2026-04-06T12:02:00Z",
                    },
                    "status": 200,
                    "message": "ok",
                }
            raise AssertionError(name)

    monkeypatch.setattr(mod, "Jin10McpClient", FakeClient)
    monkeypatch.setattr(mod, "now_utc", lambda: mod.dt.datetime(2026, 4, 6, 12, 5, tzinfo=mod.dt.timezone.utc))

    result = mod.run_snapshot(
        workspace=workspace,
        server_url="https://mcp.jin10.com/mcp",
        bearer_token="secret-token",
        token_source="JIN10_MCP_BEARER_TOKEN",
        summary_profile="standard",
        quote_codes=["XAUUSD"],
    )

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["summary"]["calendar_total"] == 2
    assert result["summary"]["high_importance_count"] == 1
    assert result["summary"]["high_importance_titles"] == ["美国非农就业人口"]
    assert result["summary"]["flash_total"] == 2
    assert result["summary"]["latest_flash_briefs"] == ["金价短线拉升", "美元指数走弱"]
    assert result["summary"]["quote_watch"][0]["code"] == "XAUUSD"
    assert "calendar=2" in result["recommended_brief"]
    assert result["takeaway"] == "美国非农就业人口"
    payload = json.loads(Path(result["artifact_json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["quote_watch"][0]["code"] == "XAUUSD"


def test_run_snapshot_parses_structured_content_tool_results(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def initialize(self):
            return {
                "protocolVersion": "2025-03-26",
                "serverInfo": {"name": "jin10", "version": "1.0.0"},
                "capabilities": {"tools": {}, "resources": {}},
            }

        def list_tools(self):
            return [{"name": "list_calendar"}, {"name": "list_flash"}, {"name": "get_quote"}]

        def list_resources(self):
            return []

        def call_tool(self, name: str, arguments: dict[str, object] | None = None):
            if name == "list_calendar":
                return {
                    "content": [{"type": "text", "text": "{\"data\":[{\"title\":\"美国3月ISM非制造业PMI\",\"pub_time\":\"2026-04-06 22:00\",\"star\":3}],\"status\":200,\"message\":\"ok\"}"}]
                }
            if name == "list_flash":
                return {
                    "structuredContent": {
                        "data": {
                            "items": [
                                {"content": "美元指数走弱", "time": "2026-04-06T11:59:00+08:00", "url": "https://x/1"},
                            ],
                            "next_cursor": "",
                            "has_more": False,
                        },
                        "status": 200,
                        "message": "ok",
                    }
                }
            if name == "get_quote":
                return {
                    "structuredContent": {
                        "data": {
                            "code": "XAUUSD",
                            "name": "现货黄金",
                            "close": "4666.51",
                            "ups_percent": "-0.20",
                            "time": "2026-04-06T20:30:00+08:00",
                        },
                        "status": 200,
                        "message": "",
                    }
                }
            raise AssertionError(name)

    monkeypatch.setattr(mod, "Jin10McpClient", FakeClient)

    result = mod.run_snapshot(
        workspace=workspace,
        server_url="https://mcp.jin10.com/mcp",
        bearer_token="secret-token",
        token_source="JIN10_MCP_BEARER_TOKEN",
        summary_profile="standard",
        quote_codes=["XAUUSD"],
    )

    assert result["summary"]["calendar_total"] == 1
    assert result["summary"]["flash_total"] == 1
    assert result["summary"]["quote_watch"][0]["name"] == "现货黄金"
