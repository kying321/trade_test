#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lie_engine.research.jin10_mcp_client import Jin10McpClient, Jin10McpError


DEFAULT_SERVER_URL = "https://mcp.jin10.com/mcp"
DEFAULT_SUMMARY_PROFILE = "standard"
DEFAULT_QUOTE_CODES = ["XAUUSD", "USOIL"]


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def workspace_to_system_root(workspace: Path) -> Path:
    workspace = workspace.resolve()
    return workspace if workspace.name == "system" else workspace / "system"


def review_dir_for_workspace(workspace: Path) -> Path:
    return workspace_to_system_root(workspace) / "output" / "review"


def safe_preview(value: Any, limit: int = 280) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _list_data(result: Any) -> list[dict[str, Any]]:
    normalized = _normalize_tool_result(result)
    data = normalized.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _dict_data(result: Any) -> dict[str, Any]:
    normalized = _normalize_tool_result(result)
    data = normalized.get("data")
    return data if isinstance(data, dict) else {}


def _normalize_tool_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    if isinstance(result.get("structuredContent"), dict):
        return result.get("structuredContent") or {}
    if any(key in result for key in ("data", "status", "message")):
        return result
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    return {}


def build_standard_summary(client: Jin10McpClient, quote_codes: list[str]) -> tuple[dict[str, Any], str, str, list[str]]:
    errors: list[str] = []
    summary: dict[str, Any] = {
        "calendar_total": 0,
        "high_importance_count": 0,
        "high_importance_titles": [],
        "flash_total": 0,
        "latest_flash_briefs": [],
        "quote_watch": [],
    }

    try:
        calendar_result = client.call_tool("list_calendar", {})
        calendar_rows = _list_data(calendar_result)
        summary["calendar_total"] = len(calendar_rows)
        high_importance = [
            row for row in calendar_rows if int(row.get("star") or 0) >= 4 and str(row.get("title") or "").strip()
        ]
        summary["high_importance_count"] = len(high_importance)
        summary["high_importance_titles"] = [str(row.get("title")).strip() for row in high_importance[:3]]
    except Jin10McpError as exc:
        errors.append(f"list_calendar:{exc}")

    try:
        flash_result = client.call_tool("list_flash", {})
        flash_rows = [row for row in list((_dict_data(flash_result).get("items") or [])) if isinstance(row, dict)]
        summary["flash_total"] = len(flash_rows)
        summary["latest_flash_briefs"] = [str(row.get("content") or "").strip() for row in flash_rows[:3] if str(row.get("content") or "").strip()]
    except Jin10McpError as exc:
        errors.append(f"list_flash:{exc}")

    quote_watch: list[dict[str, Any]] = []
    for code in [str(code).strip() for code in quote_codes if str(code).strip()]:
        try:
            quote_result = client.call_tool("get_quote", {"code": code})
            quote_data = _dict_data(quote_result)
            if quote_data:
                quote_watch.append(
                    {
                        "code": str(quote_data.get("code") or code),
                        "name": str(quote_data.get("name") or code),
                        "close": quote_data.get("close"),
                        "ups_percent": quote_data.get("ups_percent"),
                        "time": quote_data.get("time"),
                    }
                )
        except Jin10McpError as exc:
            errors.append(f"get_quote:{code}:{exc}")
    summary["quote_watch"] = quote_watch

    recommended_brief = (
        f"calendar={summary['calendar_total']} | flash={summary['flash_total']} | quotes={len(summary['quote_watch'])}"
    )
    takeaway = ""
    if summary["high_importance_titles"]:
        takeaway = str(summary["high_importance_titles"][0])
    elif summary["latest_flash_briefs"]:
        takeaway = str(summary["latest_flash_briefs"][0])
    elif summary["quote_watch"]:
        takeaway = str(summary["quote_watch"][0].get("name") or summary["quote_watch"][0].get("code") or "")
    return summary, recommended_brief, takeaway, errors


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Jin10 MCP Snapshot",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- status: `{payload['status']}`",
        f"- ok: `{str(bool(payload['ok'])).lower()}`",
        f"- change_class: `{payload['change_class']}`",
        f"- server_url: `{payload['server_url']}`",
        f"- token_source: `{payload['token_source']}`",
        f"- tools_total: `{payload['tools_total']}`",
        f"- resources_total: `{payload['resources_total']}`",
    ]
    if payload.get("error"):
        lines.append(f"- error: `{payload['error']}`")
    if payload.get("server_info"):
        lines.append(f"- server_info: `{safe_preview(payload['server_info'])}`")
    if payload.get("tool_call"):
        lines.append(f"- tool_call: `{safe_preview(payload['tool_call'])}`")
    return "\n".join(lines) + "\n"


def write_artifacts(review_dir: Path, payload: dict[str, Any], stamp: str) -> tuple[Path, Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path = review_dir / f"{stamp}_jin10_mcp_snapshot.json"
    md_path = review_dir / f"{stamp}_jin10_mcp_snapshot.md"
    latest_json = review_dir / "latest_jin10_mcp_snapshot.json"
    latest_md = review_dir / "latest_jin10_mcp_snapshot.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    return json_path, md_path


def run_snapshot(
    *,
    workspace: Path,
    server_url: str,
    bearer_token: str,
    token_source: str,
    timeout_seconds: float = 5.0,
    summary_profile: str = DEFAULT_SUMMARY_PROFILE,
    quote_codes: list[str] | None = None,
    tool_name: str = "",
    tool_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_at = now_utc()
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    review_dir = review_dir_for_workspace(workspace)

    payload: dict[str, Any] = {
        "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "jin10_mcp_snapshot",
        "change_class": "RESEARCH_ONLY",
        "ok": False,
        "status": "blocked_auth_missing",
        "server_url": str(server_url),
        "token_source": str(token_source),
        "server_info": {},
        "capabilities": {},
        "tools": [],
        "resources": [],
        "tools_total": 0,
        "resources_total": 0,
        "summary": {},
        "summary_errors": [],
        "recommended_brief": "",
        "takeaway": "",
        "tool_call": None,
        "error": "",
    }

    if not str(bearer_token).strip():
        payload["error"] = "missing_bearer_token"
        json_path, md_path = write_artifacts(review_dir, payload, stamp)
        payload["artifact_json"] = str(json_path)
        payload["artifact_md"] = str(md_path)
        return payload

    client = Jin10McpClient(
        server_url=server_url,
        bearer_token=bearer_token,
        timeout_seconds=timeout_seconds,
    )
    try:
        init = client.initialize()
        tools = client.list_tools()
        try:
            resources = client.list_resources()
        except Jin10McpError:
            resources = []

        payload["server_info"] = getattr(client, "server_info", {}) or (
            init.get("serverInfo") if isinstance(init.get("serverInfo"), dict) else {}
        )
        payload["capabilities"] = getattr(client, "capabilities", {}) or (
            init.get("capabilities") if isinstance(init.get("capabilities"), dict) else {}
        )
        payload["tools"] = tools
        payload["resources"] = resources
        payload["tools_total"] = len(tools)
        payload["resources_total"] = len(resources)

        if str(summary_profile).strip().lower() == "standard":
            summary, recommended_brief, takeaway, errors = build_standard_summary(
                client,
                quote_codes or DEFAULT_QUOTE_CODES,
            )
            payload["summary"] = summary
            payload["summary_errors"] = errors
            payload["recommended_brief"] = recommended_brief
            payload["takeaway"] = takeaway

        if str(tool_name).strip():
            tool_result = client.call_tool(str(tool_name).strip(), tool_args or {})
            payload["tool_call"] = {
                "name": str(tool_name).strip(),
                "arguments": tool_args or {},
                "result_preview": safe_preview(tool_result),
            }

        payload["ok"] = True
        payload["status"] = "ok"
    except Jin10McpError as exc:
        payload["error"] = str(exc)
        if exc.code == "auth_failed":
            payload["status"] = "blocked_auth_failed"
        elif exc.code == "transport_error":
            payload["status"] = "blocked_initialize_failed"
        else:
            payload["status"] = "degraded_request_failed"

    json_path, md_path = write_artifacts(review_dir, payload, stamp)
    payload["artifact_json"] = str(json_path)
    payload["artifact_md"] = str(md_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Jin10 MCP research-sidecar snapshot.")
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--token-env", default="JIN10_MCP_BEARER_TOKEN")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--summary-profile", default=DEFAULT_SUMMARY_PROFILE, choices=["none", "standard"])
    parser.add_argument("--quote-codes-json", default="[]")
    parser.add_argument("--tool-name", default="")
    parser.add_argument("--tool-args-json", default="{}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    tool_args = json.loads(str(args.tool_args_json) or "{}")
    quote_codes_raw = json.loads(str(args.quote_codes_json) or "[]")
    bearer_token = __import__("os").environ.get(str(args.token_env), "")
    payload = run_snapshot(
        workspace=workspace,
        server_url=str(args.server_url),
        bearer_token=str(bearer_token),
        token_source=str(args.token_env),
        timeout_seconds=float(args.timeout_seconds),
        summary_profile=str(args.summary_profile),
        quote_codes=quote_codes_raw if isinstance(quote_codes_raw, list) else [],
        tool_name=str(args.tool_name),
        tool_args=tool_args if isinstance(tool_args, dict) else {},
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
