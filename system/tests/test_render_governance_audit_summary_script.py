from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "render_governance_audit_summary.py"


def test_render_governance_audit_summary_prints_lane_owner_and_recommended_action(tmp_path: Path) -> None:
    payload = {
        "overall_pass": True,
        "checks": {
            "branch_protection_strict": {"pass": True, "return_code": 0},
            "hotfix_reaper_report": {"pass": True, "return_code": 0},
            "dashboard_lane_sync_guard": {
                "pass": True,
                "blocking": False,
                "current_owner": "lie",
                "recommended_action": "no_sync_record_ownership",
                "reason_codes": ["single_owner_detected", "pi_missing_dashboard_tree", "lie_owns_dashboard_surface"],
                "summary": "Detected a single dashboard owner branch `lie`; do not blanket-sync other primary branches.",
            },
        },
        "advisory": {
            "dashboard_lane_sync_guard_blocking": False,
            "dashboard_lane_sync_guard_included": True,
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--audit-json", str(audit_path)],
        cwd=SYSTEM_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "Governance Health Audit Summary" in proc.stdout
    assert "- overall_pass: `true`" in proc.stdout
    assert "- dashboard_lane_sync_guard.pass: `true`" in proc.stdout
    assert "- current_owner: `lie`" in proc.stdout
    assert "- recommended_action: `no_sync_record_ownership`" in proc.stdout
    assert "- reason_codes: `single_owner_detected, pi_missing_dashboard_tree, lie_owns_dashboard_surface`" in proc.stdout
    assert proc.stderr == ""


def test_render_governance_audit_summary_emits_warning_when_guard_requests_manual_review(tmp_path: Path) -> None:
    payload = {
        "overall_pass": True,
        "checks": {
            "dashboard_lane_sync_guard": {
                "pass": False,
                "blocking": False,
                "current_owner": "",
                "recommended_action": "ownership_missing_manual_review",
                "reason_codes": ["no_owner_detected"],
                "summary": "No primary branch currently carries the dashboard tree; manual review is required.",
            }
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--audit-json", str(audit_path), "--emit-github-warning"],
        cwd=SYSTEM_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "::warning::dashboard_lane_sync_guard recommended_action=ownership_missing_manual_review" in proc.stdout

