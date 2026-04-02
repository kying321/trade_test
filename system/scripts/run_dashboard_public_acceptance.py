#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

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
RESEARCH_AUDIT_SEARCH_ROUTE = "#/search"
GRAPH_HOME_ROUTE = "#/graph-home"
CONTRACTS_ACCEPTANCE_INSPECTOR_CHECK_ROUTE = "#/workspace/contracts?page_section=contracts-check-graph_home_smoke"
CONTRACTS_ACCEPTANCE_INSPECTOR_SUBCOMMAND_ROUTE = "#/workspace/contracts?page_section=contracts-subcommand-graph_home_smoke"


def _expected_contracts_check_route(check_id: str) -> str:
    return f"#/workspace/contracts?page_section=contracts-check-{check_id}"


def _expected_contracts_subcommand_route(check_id: str) -> str:
    return f"#/workspace/contracts?page_section=contracts-subcommand-{check_id}"


def _extract_contracts_inspector_rows(contracts_inspector: dict[str, Any], row_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    checks_by_id = contracts_inspector.get("checks_by_id")
    subcommands_by_id = contracts_inspector.get("subcommands_by_id")
    check_row = (checks_by_id or {}).get(row_id) if isinstance(checks_by_id, dict) else None
    subcommand_row = (subcommands_by_id or {}).get(row_id) if isinstance(subcommands_by_id, dict) else None
    if isinstance(check_row, dict) and isinstance(subcommand_row, dict):
        return check_row, subcommand_row
    # backward compatibility
    return contracts_inspector, contracts_inspector


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


def current_python_executable() -> str:
    return sys.executable or "python3"


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

    research_audit = payload.get("research_audit_search_assertion")
    if not isinstance(research_audit, dict):
        return "missing_research_audit_search_assertion"
    if str(research_audit.get("route") or "") != RESEARCH_AUDIT_SEARCH_ROUTE:
        return "invalid_research_audit_search_assertion"
    if research_audit.get("cases_available") is not True:
        return "invalid_research_audit_search_assertion"
    cases = research_audit.get("cases")
    if not isinstance(cases, list) or not cases:
        return "invalid_research_audit_search_assertion"
    for case in cases:
        if not isinstance(case, dict):
            return "invalid_research_audit_search_assertion"
        query = str(case.get("query") or "")
        search_route = str(case.get("search_route") or "")
        result_artifact = str(case.get("result_artifact") or "")
        workspace_route = str(case.get("workspace_route") or "")
        raw_path = str(case.get("raw_path") or "")
        if not query or not search_route or not result_artifact or not workspace_route or not raw_path:
            return "invalid_research_audit_search_assertion"
        if not search_route.startswith(f"{RESEARCH_AUDIT_SEARCH_ROUTE}?q=") or "&scope=artifact" not in search_route:
            return "invalid_research_audit_search_assertion"
        if not workspace_route.startswith("#/workspace/artifacts?artifact="):
            return "invalid_research_audit_search_assertion"
        if "/" not in raw_path and "\\" not in raw_path:
            return "invalid_research_audit_search_assertion"

    first_case = cases[0]
    expected_search_route = str(first_case.get("search_route") or "")
    expected_workspace_route = str(first_case.get("workspace_route") or "")
    expected_raw_href = _expected_graph_home_raw_href(str(first_case.get("raw_path") or ""))
    contracts_inspector = payload.get("contracts_acceptance_inspector_assertion")
    if not isinstance(contracts_inspector, dict):
        return "missing_contracts_acceptance_inspector_assertion"
    for row_id in ("topology_smoke", "workspace_routes_smoke", "graph_home_smoke"):
        check_row, subcommand_row = _extract_contracts_inspector_rows(contracts_inspector, row_id)
        expected_check_route = _expected_contracts_check_route(row_id)
        expected_subcommand_route = _expected_contracts_subcommand_route(row_id)
        expected_check_page_section = f"contracts-check-{row_id}"
        expected_subcommand_page_section = f"contracts-subcommand-{row_id}"
        if str(check_row.get("route") or contracts_inspector.get("check_route") or "") != expected_check_route:
            return "invalid_contracts_acceptance_inspector_assertion"
        if str(check_row.get("page_section") or contracts_inspector.get("check_page_section") or "") != expected_check_page_section:
            return "invalid_contracts_acceptance_inspector_assertion"
        if str(subcommand_row.get("route") or contracts_inspector.get("subcommand_route") or "") != expected_subcommand_route:
            return "invalid_contracts_acceptance_inspector_assertion"
        if str(subcommand_row.get("page_section") or contracts_inspector.get("subcommand_page_section") or "") != expected_subcommand_page_section:
            return "invalid_contracts_acceptance_inspector_assertion"
        if row_id == "topology_smoke":
            if str(subcommand_row.get("check_route") or contracts_inspector.get("subcommand_check_route") or contracts_inspector.get("check_route") or "") != expected_check_route:
                return "invalid_contracts_acceptance_inspector_assertion"
            continue
        if str(check_row.get("search_link_href") or contracts_inspector.get("check_search_link_href") or "") != expected_search_route:
            return "invalid_contracts_acceptance_inspector_assertion"
        actual_check_artifact = str(check_row.get("artifact_link_href") or contracts_inspector.get("check_artifact_link_href") or "")
        if not _same_artifact_route(actual_check_artifact, expected_workspace_route):
            return "invalid_contracts_acceptance_inspector_assertion"
        if str(check_row.get("raw_link_href") or contracts_inspector.get("check_raw_link_href") or "") != expected_raw_href:
            return "invalid_contracts_acceptance_inspector_assertion"
        if str(subcommand_row.get("search_link_href") or contracts_inspector.get("subcommand_search_link_href") or "") != expected_search_route:
            return "invalid_contracts_acceptance_inspector_assertion"
        actual_subcommand_artifact = str(subcommand_row.get("artifact_link_href") or contracts_inspector.get("subcommand_artifact_link_href") or "")
        if not _same_artifact_route(actual_subcommand_artifact, expected_workspace_route):
            return "invalid_contracts_acceptance_inspector_assertion"
        if str(subcommand_row.get("raw_link_href") or contracts_inspector.get("subcommand_raw_link_href") or "") != expected_raw_href:
            return "invalid_contracts_acceptance_inspector_assertion"
        if str(subcommand_row.get("check_route") or contracts_inspector.get("subcommand_check_route") or contracts_inspector.get("check_route") or "") != expected_check_route:
            return "invalid_contracts_acceptance_inspector_assertion"
    return None


def _expected_graph_home_raw_href(raw_path: str) -> str:
    from urllib.parse import quote
    return f"#/workspace/raw?artifact={quote(raw_path, safe='')}"


def _same_artifact_route(lhs: str, rhs: str) -> bool:
    lhs_url = urlsplit(lhs.removeprefix("#"))
    rhs_url = urlsplit(rhs.removeprefix("#"))
    if lhs_url.path != rhs_url.path:
        return False
    lhs_artifact = (parse_qs(lhs_url.query).get("artifact") or [""])[0]
    rhs_artifact = (parse_qs(rhs_url.query).get("artifact") or [""])[0]
    return lhs_artifact == rhs_artifact


def validate_graph_home_payload(payload: dict[str, Any], expected_cases: list[dict[str, Any]]) -> str | None:
    graph_home_assertion = payload.get("graph_home_assertion")
    if not isinstance(graph_home_assertion, dict):
        return "missing_graph_home_assertion"
    if str(graph_home_assertion.get("resolved_route") or "") != GRAPH_HOME_ROUTE:
        return "invalid_graph_home_assertion"
    actual_rows = graph_home_assertion.get("research_audit_link_assertions")
    if not isinstance(actual_rows, list) or not actual_rows:
        return "missing_graph_home_research_audit_link_assertions"
    if len(actual_rows) != len(expected_cases):
        return "invalid_graph_home_research_audit_link_assertions"
    actual_by_case = {
        str((row or {}).get("case_id") or ""): row
        for row in actual_rows
        if isinstance(row, dict) and str((row or {}).get("case_id") or "")
    }
    for expected in expected_cases:
        case_id = str(expected.get("case_id") or "")
        actual = actual_by_case.get(case_id)
        if not actual:
            return "invalid_graph_home_research_audit_link_assertions"
        if str(actual.get("search_link_href") or "") != str(expected.get("search_route") or ""):
            return "invalid_graph_home_research_audit_link_assertions"
        if not _same_artifact_route(str(actual.get("artifact_link_href") or ""), str(expected.get("workspace_route") or "")):
            return "invalid_graph_home_research_audit_link_assertions"
        if str(actual.get("raw_link_href") or "") != _expected_graph_home_raw_href(str(expected.get("raw_path") or "")):
            return "invalid_graph_home_research_audit_link_assertions"
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
    python = current_python_executable()

    topology_cmd = [
        python,
        str(scripts_root / "run_dashboard_public_topology_smoke.py"),
        "--workspace",
        str(workspace),
    ]
    if allow_insecure_tls_fallback:
        topology_cmd.append("--allow-insecure-tls-fallback")
    workspace_cmd = [
        python,
        str(scripts_root / "run_dashboard_workspace_artifacts_smoke.py"),
        "--workspace",
        str(workspace),
        "--timeout-seconds",
        str(workspace_timeout_seconds),
    ]
    if skip_workspace_build:
        workspace_cmd.append("--skip-build")
    graph_home_cmd = [
        python,
        str(scripts_root / "run_dashboard_workspace_artifacts_smoke.py"),
        "--workspace",
        str(workspace),
        "--timeout-seconds",
        str(workspace_timeout_seconds),
        "--mode",
        "graph_home",
        "--skip-build",
    ]

    topology_result = run_json_script(name="topology_smoke", cmd=topology_cmd, cwd=workspace)
    workspace_result = run_json_script(name="workspace_routes_smoke", cmd=workspace_cmd, cwd=workspace)
    graph_home_result = run_json_script(name="graph_home_smoke", cmd=graph_home_cmd, cwd=workspace)

    topology_payload = build_check_result(topology_result)
    workspace_payload = build_check_result(workspace_result)
    graph_home_payload = build_check_result(graph_home_result)
    workspace_validation_failure = validate_workspace_routes_payload(workspace_payload)
    if workspace_validation_failure:
        workspace_payload["ok"] = False
        workspace_payload["status"] = "failed"
        workspace_payload["failure_reason"] = workspace_validation_failure
        workspace_payload["expected_orderflow_artifacts_filter_assertion"] = ORDERFLOW_ARTIFACTS_FILTER_ASSERTION
    expected_research_cases = []
    research_audit_assertion = workspace_payload.get("research_audit_search_assertion")
    if isinstance(research_audit_assertion, dict):
        raw_cases = research_audit_assertion.get("cases")
        if isinstance(raw_cases, list):
            expected_research_cases = [row for row in raw_cases if isinstance(row, dict)]
    graph_home_validation_failure = validate_graph_home_payload(graph_home_payload, expected_research_cases)
    if graph_home_validation_failure:
        graph_home_payload["ok"] = False
        graph_home_payload["status"] = "failed"
        graph_home_payload["failure_reason"] = graph_home_validation_failure

    ok = (
        topology_result["returncode"] == 0
        and workspace_result["returncode"] == 0
        and graph_home_result["returncode"] == 0
        and bool(topology_payload.get("ok"))
        and bool(workspace_payload.get("ok"))
        and bool(graph_home_payload.get("ok"))
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
            "graph_home_smoke": graph_home_payload,
        },
        "subcommands": {
            "topology_smoke": build_subcommand_result(topology_result),
            "workspace_routes_smoke": build_subcommand_result(workspace_result),
            "graph_home_smoke": build_subcommand_result(graph_home_result),
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
