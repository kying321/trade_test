from __future__ import annotations

from pathlib import Path


WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "governance-health-audit.yml"


def test_governance_health_audit_workflow_publishes_dashboard_lane_sync_summary() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "Run non-destructive governance audit" in workflow
    assert "system/scripts/run_governance_audit_advisory.sh" in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "--emit-github-warning" in workflow
    assert "*_branch_governance_audit.json" in workflow
