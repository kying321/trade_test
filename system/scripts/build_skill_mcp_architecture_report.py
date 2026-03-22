#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


WORKSPACE = Path("/Users/jokenrobot/Downloads/Folders/fenlie")
CODEX_CONFIG = Path("/Users/jokenrobot/.codex/config.toml")
OUTPUT_DIR = WORKSPACE / "system/output/review"
DOC_PATH = WORKSPACE / "system/docs/FENLIE_SKILL_MCP_ARCHITECTURE.md"
SKILL_PATH = Path("/Users/jokenrobot/.codex/skills/fenlie-skill-mcp-governance/SKILL.md")
POLICY_PATH = WORKSPACE / "system/config/skill_mcp_routing_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned Fenlie skill/MCP architecture report."
    )
    parser.add_argument(
        "--workspace",
        default=str(WORKSPACE),
        help="Fenlie workspace root.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="UTC timestamp in YYYYMMDDTHHMMSSZ. Defaults to current UTC.",
    )
    return parser.parse_args()


def utc_stamp(explicit: str | None) -> str:
    if explicit:
        return explicit
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_configured_mcps(config_text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for raw in config_text.splitlines():
        line = raw.strip()
        if line.startswith("[mcp_servers.") and line.endswith("]"):
            if current:
                rows.append(current)
            name = line[len("[mcp_servers.") : -1]
            current = {"name": name}
            continue
        if current is None or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key in {"url", "command", "args"}:
            current[key] = value
    if current:
        rows.append(current)
    return rows


def build_payload(stamp: str, workspace: Path) -> Dict[str, Any]:
    config_text = read_text(CODEX_CONFIG)
    policy = json.loads(read_text(POLICY_PATH))
    configured_mcps = parse_configured_mcps(config_text)
    skills = sorted(
        str(path)
        for path in Path("/Users/jokenrobot/.codex/skills").glob("*/SKILL.md")
    )
    agent_skills = sorted(
        str(path)
        for path in Path("/Users/jokenrobot/.agents/skills").glob("*/SKILL.md")
    )
    session_status = [
        {
            "name": "coingecko",
            "status": "verified_tools_available",
            "evidence": "Earlier session successfully queried ETH price, trending coins, and 30d ETH OHLC.",
            "allowed_layers": ["research_enrichment"],
        },
        {
            "name": "figma",
            "status": "verified_auth_and_resources_available",
            "evidence": "resources/list, templates/list, and whoami all succeeded in this session.",
            "allowed_layers": ["ui_rendering", "design_to_code"],
        },
        {
            "name": "playwright",
            "status": "configured_not_resource_driven",
            "evidence": "Configured via npx in Codex; previously used for local preview/browser validation.",
            "allowed_layers": ["ui_rendering", "browser_validation"],
        },
        {
            "name": "jshook",
            "status": "verified_environment_and_search_available",
            "evidence": "doctor_environment succeeded and search_tools returned active browser/network diagnostics in this session.",
            "allowed_layers": ["browser_reverse_engineering", "read_only_diagnostics"],
        },
        {
            "name": "notion",
            "status": "blocked_auth_refresh_failed",
            "evidence": "MCP initialize failed with invalid_grant refresh-token error in this session.",
            "allowed_layers": ["knowledge_capture_once_auth_fixed"],
        },
        {
            "name": "linear",
            "status": "blocked_initialize_failed",
            "evidence": "MCP initialize request to https://mcp.linear.app/mcp failed in this session.",
            "allowed_layers": ["planning_once_connectivity_fixed"],
        },
    ]
    green = [row["name"] for row in session_status if row["status"].startswith("verified_")]
    yellow = [
        row["name"]
        for row in session_status
        if row["status"].startswith("configured_")
    ]
    red = [row["name"] for row in session_status if row["status"].startswith("blocked_")]
    for row in session_status:
        row["health_action"] = policy["status_actions"].get(row["status"], "")

    payload: Dict[str, Any] = {
        "report_type": "fenlie_skill_mcp_architecture",
        "generated_at_utc": stamp,
        "workspace": str(workspace),
        "change_class": "DOC_ONLY",
        "scope": "architecture_review",
        "codex_config_path": str(CODEX_CONFIG),
        "routing_policy_path": str(POLICY_PATH),
        "architecture_doc_path": str(DOC_PATH),
        "governance_skill_path": str(SKILL_PATH),
        "configured_mcp_servers": configured_mcps,
        "installed_skills": skills,
        "installed_agent_skills": agent_skills,
        "session_validated_mcp_status": session_status,
        "health_gate_summary": {
            "green": green,
            "yellow": yellow,
            "red": red,
            "highest_risk_unresolved": "blocked knowledge/planning MCP auth/connectivity remains unresolved for notion and linear",
            "next_validation_target": "playwright smoke-check can be added later, but the current highest-risk issue is blocked notion/linear health",
        },
        "layer_model": policy["layer_model"],
        "mode_to_skill_mcp_routing": policy["mode_to_skill_mcp_routing"],
        "hard_rules": policy["hard_rules"],
        "recommended_next_actions": [
            "Keep CoinGecko in the research-only enrichment lane.",
            "Use Playwright/Figma/JSHook only inside ui_rendering or read-only diagnostics lanes.",
            "Do not route Notion/Linear into workflows until auth/connectivity health is restored.",
            "Use the new fenlie-skill-mcp-governance skill as the entry point whenever MCP inventory or routing changes.",
        ],
    }
    return payload


def build_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Fenlie Skill/MCP Architecture Report")
    lines.append("")
    lines.append(f"- generated_at_utc: `{payload['generated_at_utc']}`")
    lines.append(f"- workspace: `{payload['workspace']}`")
    lines.append(f"- change_class: `{payload['change_class']}`")
    lines.append("")
    lines.append("## Health Gate Summary")
    lines.append("")
    health = payload["health_gate_summary"]
    lines.append(f"- green: `{', '.join(health['green'])}`")
    lines.append(f"- yellow: `{', '.join(health['yellow']) if health['yellow'] else 'none'}`")
    lines.append(f"- red: `{', '.join(health['red'])}`")
    lines.append(f"- highest_risk_unresolved: {health['highest_risk_unresolved']}")
    lines.append(f"- next_validation_target: {health['next_validation_target']}")
    lines.append("")
    lines.append("## Configured MCP Servers")
    lines.append("")
    for row in payload["configured_mcp_servers"]:
        lines.append(f"- `{row.get('name')}`")
        if row.get("url"):
            lines.append(f"  - url: `{row['url']}`")
        if row.get("command"):
            lines.append(f"  - command: `{row['command']}`")
        if row.get("args"):
            lines.append(f"  - args: `{row['args']}`")
    lines.append("")
    lines.append("## Session-Validated MCP Status")
    lines.append("")
    for row in payload["session_validated_mcp_status"]:
        lines.append(f"- `{row['name']}` -> `{row['status']}`")
        lines.append(f"  - evidence: {row['evidence']}")
        lines.append(f"  - allowed_layers: `{', '.join(row['allowed_layers'])}`")
        if row.get("health_action"):
            lines.append(f"  - health_action: {row['health_action']}")
    lines.append("")
    lines.append("## Layer Model")
    lines.append("")
    for name, row in payload["layer_model"].items():
        lines.append(f"### `{name}`")
        lines.append(f"- purpose: {row['purpose']}")
        lines.append(f"- mcp_policy: {row['mcp_policy']}")
        lines.append(f"- preferred_skills: `{', '.join(row['preferred_skills'])}`")
        lines.append("")
    lines.append("## Mode Routing")
    lines.append("")
    for mode, row in payload["mode_to_skill_mcp_routing"].items():
        lines.append(f"### `{mode}`")
        lines.append(f"- primary_skills: `{', '.join(row.get('primary_skills', []))}`")
        if row.get("mcp_allow"):
            lines.append(f"- mcp_allow: `{', '.join(row['mcp_allow'])}`")
        if row.get("mcp_require_health_gate"):
            lines.append(
                f"- mcp_require_health_gate: `{', '.join(row['mcp_require_health_gate'])}`"
            )
        if row.get("mcp_forbid"):
            lines.append(f"- mcp_forbid: `{', '.join(row['mcp_forbid'])}`")
        lines.append("")
    lines.append("## Hard Rules")
    lines.append("")
    for rule in payload["hard_rules"]:
        lines.append(f"- {rule}")
    lines.append("")
    lines.append("## Recommended Next Actions")
    lines.append("")
    for action in payload["recommended_next_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    lines.append("## Key Paths")
    lines.append("")
    lines.append(f"- architecture_doc: `{payload['architecture_doc_path']}`")
    lines.append(f"- governance_skill: `{payload['governance_skill_path']}`")
    lines.append(f"- codex_config: `{payload['codex_config_path']}`")
    lines.append(f"- routing_policy: `{payload['routing_policy_path']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp(args.timestamp)
    payload = build_payload(stamp=stamp, workspace=workspace)
    json_path = OUTPUT_DIR / f"{stamp}_skill_mcp_architecture_report.json"
    md_path = OUTPUT_DIR / f"{stamp}_skill_mcp_architecture_report.md"
    latest_json = OUTPUT_DIR / "latest_skill_mcp_architecture_report.json"
    latest_md = OUTPUT_DIR / "latest_skill_mcp_architecture_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    shutil.copyfile(json_path, latest_json)
    shutil.copyfile(md_path, latest_md)
    print(str(json_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
