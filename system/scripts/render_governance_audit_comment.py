#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


COMMENT_MARKER = "<!-- fenlie-governance-audit-advisory -->"


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def load_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_comment_markdown(payload: dict[str, object]) -> str:
    checks = payload.get("checks", {}) if isinstance(payload.get("checks", {}), dict) else {}
    guard = checks.get("dashboard_lane_sync_guard", {}) if isinstance(checks.get("dashboard_lane_sync_guard", {}), dict) else {}
    reason_codes = [str(x).strip() for x in list(guard.get("reason_codes") or []) if str(x).strip()]

    lines = [
        COMMENT_MARKER,
        "## Governance Audit Advisory",
        "",
        f"- overall_pass: `{_bool_text(payload.get('overall_pass'))}`",
        f"- dashboard_lane_sync_guard.pass: `{_bool_text(guard.get('pass'))}`",
        f"- current_owner: `{str(guard.get('current_owner') or '-')}`",
        f"- recommended_action: `{str(guard.get('recommended_action') or '-')}`",
        f"- reason_codes: `{', '.join(reason_codes) if reason_codes else '-'}`",
        f"- summary: {str(guard.get('summary') or '-').strip()}",
        "",
        "> advisory-only: this comment does not change the required check outcome.",
        "",
        "- workflow artifacts include `*_branch_governance_audit.json` and `*_branch_governance_audit.md`",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render governance audit PR comment markdown.")
    parser.add_argument("--audit-json", required=True, help="Path to *_branch_governance_audit.json")
    parser.add_argument("--comment-out", help="Optional path to write comment markdown.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_json = Path(args.audit_json).expanduser().resolve()
    payload = load_payload(audit_json)
    markdown = build_comment_markdown(payload)

    if args.comment_out:
        out_path = Path(args.comment_out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
