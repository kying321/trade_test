from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_orderflow_quality_report.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_orderflow_quality_report_marks_learning_only_shadow_route_as_viable(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T103300Z_remote_orderflow_feedback.json",
        {
            "status": "ok",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "queue_age_status": "queue_warming",
            "ticket_artifact_status": "stale_artifact",
            "guardian_blocked_count": 1,
            "no_fill_count": 1,
            "recent_outcomes": ["not_attempted_wait_trade_readiness"],
            "blocker_detail": "guardian blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T103500Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_learning_only",
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "accept_shadow_learning_only",
            "blocker_detail": "policy blocked",
        },
    )
    _write_json(
        review_dir / "20260315T111015Z_remote_execution_ack_state.json",
        {
            "status": "ok",
            "ack_status": "shadow_learning_ack_recorded",
            "ack_brief": "shadow_learning_ack_recorded:SOLUSDT:not_sent_learning_only:no_fill_execution_not_attempted",
            "blocker_detail": "ack blocked",
        },
    )
    _write_json(
        review_dir / "20260315T111020Z_remote_execution_actor_state.json",
        {
            "status": "ok",
            "actor_status": "shadow_actor_learning_only",
            "actor_brief": "shadow_actor_learning_only:SOLUSDT:shadow_learning_no_transport:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T111400Z_remote_execution_transport_sla.json",
        {
            "status": "ok",
            "transport_sla_status": "shadow_transport_sla_learning_only_no_send",
            "transport_sla_brief": "shadow_transport_sla_learning_only_no_send:SOLUSDT:not_armed_learning_only:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T111500Z_remote_execution_actor_canary_gate.json",
        {
            "status": "ok",
            "canary_gate_status": "shadow_canary_gate_blocked",
            "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T102000Z_remote_execution_journal.json",
        {
            "status": "ok",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "blocker_detail": "journal blocked",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:17:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["quality_status"] == "quality_learning_only_shadow_viable"
    assert payload["quality_recommendation"] == "continue_shadow_learning_until_guardian_clear"
    assert payload["shadow_learning_score"] == 65
    assert payload["execution_readiness_score"] == 5
    assert payload["transport_observability_score"] == 100
    assert payload["quality_score"] == 49
    assert payload["blocker_detail"] == "guardian blocked"
    assert payload["next_transition"] == "live_boundary_hold"
