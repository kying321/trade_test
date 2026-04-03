from __future__ import annotations

from pathlib import Path


BRANCH_POLICY_WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "branch-policy.yml"
HOTFIX_PR_GATE_WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "hotfix-pr-gate.yml"


def test_branch_policy_workflow_runs_non_blocking_governance_audit_summary() -> None:
    workflow = BRANCH_POLICY_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "issues: write" in workflow
    assert "Run governance audit advisory summary" in workflow
    assert "Upload governance audit advisory artifacts" in workflow
    assert "Render governance audit advisory PR comment body" in workflow
    assert "Upsert governance audit advisory PR comment" in workflow
    assert "run_governance_audit_advisory.sh" in workflow
    assert "upsert_governance_audit_pr_comment.py" in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "--emit-github-warning" in workflow
    assert "governance_audit_pr_comment.md" in workflow
    assert "uses: actions/upload-artifact@v4" in workflow
    assert "name: branch-policy-governance-audit" in workflow
    assert "system/output/review/*_branch_governance_audit.json" in workflow
    assert "system/output/review/*_branch_governance_audit.md" in workflow
    assert "if: always()" in workflow


def test_hotfix_pr_gate_workflow_runs_non_blocking_governance_audit_summary() -> None:
    workflow = HOTFIX_PR_GATE_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "issues: write" in workflow
    assert "Run governance audit advisory summary" in workflow
    assert "Upload governance audit advisory artifacts" in workflow
    assert "Render governance audit advisory PR comment body" in workflow
    assert "Upsert governance audit advisory PR comment" in workflow
    assert "run_governance_audit_advisory.sh" in workflow
    assert "upsert_governance_audit_pr_comment.py" in workflow
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "--emit-github-warning" in workflow
    assert "governance_audit_pr_comment.md" in workflow
    assert "uses: actions/upload-artifact@v4" in workflow
    assert "name: hotfix-pr-gate-governance-audit" in workflow
    assert "system/output/review/*_branch_governance_audit.json" in workflow
    assert "system/output/review/*_branch_governance_audit.md" in workflow
    assert "if: always()" in workflow
