from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = SYSTEM_ROOT / "scripts" / "run_governance_audit_advisory.sh"


def test_governance_audit_advisory_runner_is_non_blocking_and_renders_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir(parents=True)

    audit_script = stub_dir / "audit.sh"
    audit_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'output_dir=""',
                'while [[ $# -gt 0 ]]; do',
                '  case "$1" in',
                '    --output-dir)',
                '      shift',
                '      output_dir="${1:?missing output dir}"',
                '      ;;',
                '  esac',
                '  shift || true',
                'done',
                'test -n "${output_dir}"',
                'mkdir -p "${output_dir}"',
                'json_path="${output_dir}/20260403T010101Z_branch_governance_audit.json"',
                'md_path="${output_dir}/20260403T010101Z_branch_governance_audit.md"',
                "cat >\"${json_path}\" <<'EOF'",
                '{"overall_pass": true, "checks": {"dashboard_lane_sync_guard": {"pass": true, "blocking": false, "current_owner": "lie", "recommended_action": "no_sync_record_ownership", "reason_codes": ["single_owner_detected"], "summary": "stub audit payload"}}}',
                "EOF",
                "echo '# stub md' >\"${md_path}\"",
                'echo "${json_path}"',
                'echo "${md_path}"',
                "exit 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    audit_script.chmod(0o755)

    summary_script = stub_dir / "summary.py"
    summary_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import argparse, pathlib",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--audit-json', required=True)",
                "parser.add_argument('--step-summary-out')",
                "parser.add_argument('--emit-github-warning', action='store_true')",
                "args = parser.parse_args()",
                "if args.step_summary_out:",
                "    path = pathlib.Path(args.step_summary_out)",
                "    path.parent.mkdir(parents=True, exist_ok=True)",
                "    path.write_text('summary ok\\n', encoding='utf-8')",
                "print('summary renderer ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_script.chmod(0o755)

    comment_script = stub_dir / "comment.py"
    comment_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import argparse, pathlib",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--audit-json', required=True)",
                "parser.add_argument('--comment-out')",
                "args = parser.parse_args()",
                "path = pathlib.Path(args.comment_out)",
                "path.parent.mkdir(parents=True, exist_ok=True)",
                "path.write_text('<!-- fenlie-governance-audit-advisory -->\\ncomment ok\\n', encoding='utf-8')",
                "print('comment renderer ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    comment_script.chmod(0o755)

    output_rel = f"system/output/review/pytest_governance_advisory_{tmp_path.name}"
    output_dir = repo_root / output_rel
    shutil.rmtree(output_dir, ignore_errors=True)
    step_summary = tmp_path / "step_summary.md"
    comment_out = tmp_path / "comment.md"

    env = os.environ.copy()
    env["FENLIE_BRANCH_GOV_AUDIT_SCRIPT"] = str(audit_script)
    env["FENLIE_GOV_SUMMARY_SCRIPT"] = str(summary_script)
    env["FENLIE_GOV_COMMENT_SCRIPT"] = str(comment_script)

    proc = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
            "--repo",
            "example/repo",
            "--audit-output-dir",
            output_rel,
            "--step-summary-out",
            str(step_summary),
            "--comment-out",
            str(comment_out),
            "--emit-github-warning",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert step_summary.read_text(encoding="utf-8") == "summary ok\n"
    assert "<!-- fenlie-governance-audit-advisory -->" in comment_out.read_text(encoding="utf-8")
    assert "non-blocking audit_rc=2" in proc.stdout
    assert (output_dir / "20260403T010101Z_branch_governance_audit.json").exists()
    assert json.loads((output_dir / "20260403T010101Z_branch_governance_audit.json").read_text(encoding="utf-8"))["checks"][
        "dashboard_lane_sync_guard"
    ]["current_owner"] == "lie"

    shutil.rmtree(output_dir, ignore_errors=True)
