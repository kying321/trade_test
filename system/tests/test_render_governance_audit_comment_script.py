from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "render_governance_audit_comment.py"


def test_render_governance_audit_comment_prints_sticky_marker_and_core_fields(tmp_path: Path) -> None:
    payload = {
        "overall_pass": True,
        "checks": {
            "dashboard_lane_sync_guard": {
                "pass": True,
                "blocking": False,
                "current_owner": "lie",
                "recommended_action": "no_sync_record_ownership",
                "reason_codes": ["single_owner_detected", "pi_missing_dashboard_tree", "lie_owns_dashboard_surface"],
                "summary": "Detected a single dashboard owner branch `lie`; do not blanket-sync other primary branches.",
            }
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
    assert "<!-- fenlie-governance-audit-advisory -->" in proc.stdout
    assert "## Governance Audit Advisory" in proc.stdout
    assert "- overall_pass: `true`" in proc.stdout
    assert "- dashboard_lane_sync_guard.pass: `true`" in proc.stdout
    assert "- current_owner: `lie`" in proc.stdout
    assert "- recommended_action: `no_sync_record_ownership`" in proc.stdout
    assert "workflow artifacts include `*_branch_governance_audit.json` and `*_branch_governance_audit.md`" in proc.stdout
    assert proc.stderr == ""


def test_render_governance_audit_comment_can_write_output_file(tmp_path: Path) -> None:
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
    out_path = tmp_path / "comment.md"
    audit_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--audit-json", str(audit_path), "--comment-out", str(out_path)],
        cwd=SYSTEM_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    written = out_path.read_text(encoding="utf-8")
    assert "<!-- fenlie-governance-audit-advisory -->" in written
    assert "ownership_missing_manual_review" in written
