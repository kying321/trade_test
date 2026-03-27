#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

CHANGE_CLASS = "RESEARCH_ONLY"
ORDERFLOW_ARTIFACTS_FILTER_ASSERTION = {
    "route": "#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow",
    "group": "research_cross_section",
    "search_scope": "title",
    "search": "orderflow",
    "active_artifact": "intraday_orderflow_blueprint",
    "visible_artifacts": [
        "intraday_orderflow_blueprint",
        "intraday_orderflow_research_gate_blocker",
    ],
}


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


def run_json_script(*, name: str, cmd: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    payload: dict[str, Any] | None = None
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
    return {
        "name": name,
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "payload": payload,
    }


def build_check_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result.get("payload") or {})
    if not payload:
        payload = {
            "ok": False,
            "status": "failed",
        }
        if result.get("returncode"):
            payload["failure_reason"] = "child_exit_nonzero"
        else:
            payload["failure_reason"] = "missing_json_payload"
    payload["returncode"] = result["returncode"]
    payload["cmd"] = result["cmd"]
    payload["cwd"] = result["cwd"]
    if result["returncode"] != 0 or result.get("payload") is None:
        if result.get("stdout"):
            payload["stdout"] = result["stdout"]
        if result.get("stderr"):
            payload["stderr"] = result["stderr"]
    return payload


def build_subcommand_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "returncode": result["returncode"],
        "cmd": result["cmd"],
        "cwd": result["cwd"],
        "payload_present": result.get("payload") is not None,
        "stdout_bytes": len(result.get("stdout") or ""),
        "stderr_bytes": len(result.get("stderr") or ""),
    }
    if result["returncode"] != 0 or result.get("payload") is None:
        if result.get("stdout"):
            payload["stdout"] = result["stdout"]
        if result.get("stderr"):
            payload["stderr"] = result["stderr"]
    return payload


def validate_workspace_routes_payload(payload: dict[str, Any]) -> str | None:
    actual = payload.get("artifacts_filter_assertion")
    if not isinstance(actual, dict):
        return "missing_orderflow_artifacts_filter_assertion"

    if str(actual.get("route") or "") != ORDERFLOW_ARTIFACTS_FILTER_ASSERTION["route"]:
        return "invalid_orderflow_artifacts_filter_assertion"
    if str(actual.get("group") or "") != ORDERFLOW_ARTIFACTS_FILTER_ASSERTION["group"]:
        return "invalid_orderflow_artifacts_filter_assertion"
    if str(actual.get("search_scope") or "") != ORDERFLOW_ARTIFACTS_FILTER_ASSERTION["search_scope"]:
        return "invalid_orderflow_artifacts_filter_assertion"
    if str(actual.get("search") or "") != ORDERFLOW_ARTIFACTS_FILTER_ASSERTION["search"]:
        return "invalid_orderflow_artifacts_filter_assertion"

    visible_artifacts = actual.get("visible_artifacts")
    if not isinstance(visible_artifacts, list):
        return "invalid_orderflow_artifacts_filter_assertion"

    source_available = actual.get("source_available")
    if source_available is False:
        if str(actual.get("active_artifact") or "") != "":
            return "invalid_orderflow_artifacts_filter_assertion"
        if visible_artifacts != []:
            return "invalid_orderflow_artifacts_filter_assertion"
        return None

    if str(actual.get("active_artifact") or "") != ORDERFLOW_ARTIFACTS_FILTER_ASSERTION["active_artifact"]:
        return "invalid_orderflow_artifacts_filter_assertion"
    if visible_artifacts != ORDERFLOW_ARTIFACTS_FILTER_ASSERTION["visible_artifacts"]:
        return "invalid_orderflow_artifacts_filter_assertion"
    return None


def run_acceptance(
    *,
    workspace: Path,
    skip_workspace_build: bool,
    workspace_timeout_seconds: float,
    allow_insecure_tls_fallback: bool = True,
) -> dict[str, Any]:
    system_root = resolve_system_root(workspace)
    scripts_root = system_root / "scripts"

    topology_cmd = [
        "python3",
        str(scripts_root / "run_dashboard_public_topology_smoke.py"),
        "--workspace",
        str(workspace),
    ]
    if allow_insecure_tls_fallback:
        topology_cmd.append("--allow-insecure-tls-fallback")
    workspace_cmd = [
        "python3",
        str(scripts_root / "run_dashboard_workspace_artifacts_smoke.py"),
        "--workspace",
        str(workspace),
        "--timeout-seconds",
        str(workspace_timeout_seconds),
    ]
    if skip_workspace_build:
        workspace_cmd.append("--skip-build")

    topology_result = run_json_script(name="topology_smoke", cmd=topology_cmd, cwd=workspace)
    workspace_result = run_json_script(name="workspace_routes_smoke", cmd=workspace_cmd, cwd=workspace)

    topology_payload = build_check_result(topology_result)
    workspace_payload = build_check_result(workspace_result)
    workspace_validation_failure = validate_workspace_routes_payload(workspace_payload)
    if workspace_validation_failure:
        workspace_payload["ok"] = False
        workspace_payload["status"] = "failed"
        workspace_payload["failure_reason"] = workspace_validation_failure
        workspace_payload["expected_orderflow_artifacts_filter_assertion"] = ORDERFLOW_ARTIFACTS_FILTER_ASSERTION

    ok = (
        topology_result["returncode"] == 0
        and workspace_result["returncode"] == 0
        and bool(topology_payload.get("ok"))
        and bool(workspace_payload.get("ok"))
    )

    return {
        "action": "dashboard_public_acceptance",
        "ok": ok,
        "status": "ok" if ok else "failed",
        "change_class": CHANGE_CLASS,
        "generated_at_utc": fmt_utc(now_utc()),
        "workspace": str(workspace),
        "checks": {
            "topology_smoke": topology_payload,
            "workspace_routes_smoke": workspace_payload,
        },
        "subcommands": {
            "topology_smoke": build_subcommand_result(topology_result),
            "workspace_routes_smoke": build_subcommand_result(workspace_result),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dashboard public acceptance pack (topology + workspace routes).")
    parser.add_argument("--workspace", default=".", help="Fenlie workspace root or system directory.")
    parser.add_argument("--skip-workspace-build", action="store_true", help="Pass --skip-build to workspace routes smoke.")
    parser.add_argument("--workspace-timeout-seconds", type=float, default=45.0, help="Timeout for workspace routes smoke.")
    parser.add_argument("--strict-topology-tls", action="store_true", help="Disable insecure TLS fallback for topology smoke and require strict certificate validation.")
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    system_root = resolve_system_root(workspace)
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    payload = run_acceptance(
        workspace=workspace,
        skip_workspace_build=args.skip_workspace_build,
        workspace_timeout_seconds=args.workspace_timeout_seconds,
        allow_insecure_tls_fallback=not args.strict_topology_tls,
    )
    timestamp = now_utc().strftime("%Y%m%dT%H%M%SZ")
    report_path = review_dir / f"{timestamp}_dashboard_public_acceptance.json"
    report_payload = {
        **payload,
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))
    if not report_payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
