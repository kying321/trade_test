from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_shadow_learning_continuity.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_shadow_learning_continuity_marks_stable_learning_path(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T111005Z_openclaw_orderflow_executor_heartbeat.json",
        {
            "generated_at_utc": "2026-03-15T11:10:05Z",
            "intent_symbol": "SOLUSDT",
        },
    )
    _write_json(
        review_dir / "20260315T111010Z_openclaw_orderflow_executor_state.json",
        {
            "generated_at_utc": "2026-03-15T11:10:10Z",
            "executor_status": "shadow_guarded_executor_ready",
        },
    )
    _write_json(
        review_dir / "20260315T102000Z_remote_execution_journal.json",
        {
            "generated_at_utc": "2026-03-15T10:20:00Z",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry": {"recorded_at_utc": "2026-03-15T10:20:00Z"},
            "intent_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T072500Z_remote_orderflow_quality_report.json",
        {
            "quality_status": "quality_learning_only_shadow_viable",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T114500Z_remote_shadow_clock_evidence.json",
        {
            "shadow_learning_allowed": True,
        },
    )
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_brief": "guarded_canary_promotion_blocked_shadow_learning_allowed:SOLUSDT:block_promotion_continue_shadow_learning:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:55:40Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["continuity_status"] == "shadow_learning_continuity_stable"
    assert payload["continuity_decision"] == "continue_shadow_learning_collect_feedback"
    assert payload["quality_score"] == 49
    assert payload["shadow_learning_score"] == 65
