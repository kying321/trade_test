#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def load_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary_markdown(payload: dict[str, object], audit_json: Path) -> str:
    checks = payload.get("checks", {}) if isinstance(payload.get("checks", {}), dict) else {}
    branch_protection = checks.get("branch_protection_strict", {}) if isinstance(checks.get("branch_protection_strict", {}), dict) else {}
    hotfix_reaper = checks.get("hotfix_reaper_report", {}) if isinstance(checks.get("hotfix_reaper_report", {}), dict) else {}
    guard = checks.get("dashboard_lane_sync_guard", {}) if isinstance(checks.get("dashboard_lane_sync_guard", {}), dict) else {}
    reason_codes = [str(x).strip() for x in list(guard.get("reason_codes") or []) if str(x).strip()]

    lines = [
        "# Governance Health Audit Summary",
        "",
        f"- artifact: `{audit_json}`",
        f"- overall_pass: `{_bool_text(payload.get('overall_pass'))}`",
        f"- branch_protection_strict.pass: `{_bool_text(branch_protection.get('pass'))}`",
        f"- hotfix_reaper_report.pass: `{_bool_text(hotfix_reaper.get('pass'))}`",
        f"- dashboard_lane_sync_guard.pass: `{_bool_text(guard.get('pass'))}`",
        f"- current_owner: `{str(guard.get('current_owner') or '-')}`",
        f"- recommended_action: `{str(guard.get('recommended_action') or '-')}`",
        f"- reason_codes: `{', '.join(reason_codes) if reason_codes else '-'}`",
        f"- summary: {str(guard.get('summary') or '-').strip()}",
    ]
    return "\n".join(lines) + "\n"


def build_warning_line(payload: dict[str, object]) -> str | None:
    checks = payload.get("checks", {}) if isinstance(payload.get("checks", {}), dict) else {}
    guard = checks.get("dashboard_lane_sync_guard", {}) if isinstance(checks.get("dashboard_lane_sync_guard", {}), dict) else {}
    recommended_action = str(guard.get("recommended_action") or "").strip()
    summary = str(guard.get("summary") or "").strip()
    if not recommended_action or recommended_action == "no_sync_record_ownership":
        return None
    suffix = f" {summary}" if summary else ""
    return f"::warning::dashboard_lane_sync_guard recommended_action={recommended_action}{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render governance audit markdown summary.")
    parser.add_argument("--audit-json", required=True, help="Path to *_branch_governance_audit.json")
    parser.add_argument("--emit-github-warning", action="store_true", help="Emit GitHub warning annotation when manual review is recommended.")
    parser.add_argument("--step-summary-out", help="Optional path to append markdown summary to, such as GITHUB_STEP_SUMMARY.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_json = Path(args.audit_json).expanduser().resolve()
    payload = load_payload(audit_json)
    markdown = build_summary_markdown(payload, audit_json=audit_json)

    if args.step_summary_out:
        summary_path = Path(args.step_summary_out).expanduser()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(markdown)
            if not markdown.endswith("\n"):
                handle.write("\n")

    print(markdown, end="")

    if args.emit_github_warning:
        warning_line = build_warning_line(payload)
        if warning_line:
            print(warning_line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
