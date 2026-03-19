from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_orderflow_feedback.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_orderflow_feedback_downranks_guardian_blocked_route(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_status": "queued_wait_trade_readiness",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
            "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT",
            "ticket_artifact_status": "stale_artifact",
            "guard_alignment_brief": "risk_guard_candidate_mismatch:BNBUSDT->SOLUSDT",
        },
    )
    journal_log = review_dir / "remote_execution_journal.jsonl"
    journal_log.write_text(
        json.dumps(
            {
                "entry_key": "entry-1",
                "intent_symbol": "SOLUSDT",
                "risk_verdict_brief": "blocked:ticket_missing:no_actionable_ticket,panic_cooldown_active",
                "execution_outcome": "not_attempted_wait_trade_readiness",
                "fill_status": "no_fill_execution_not_attempted",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100010Z_remote_execution_journal.json",
        {
            "generated_at_utc": "2026-03-15T10:20:00Z",
            "journal_status": "intent_logged_guardian_blocked",
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "journal_path": str(journal_log),
            "last_entry_key": "entry-1",
            "last_entry": {
                "recorded_at_utc": "2026-03-15T10:20:00Z",
                "intent_symbol": "SOLUSDT",
                "risk_reason_codes": [
                    "ticket_missing:no_actionable_ticket",
                    "panic_cooldown_active",
                ],
            },
            "execution_outcome": "not_attempted_wait_trade_readiness",
            "fill_status": "no_fill_execution_not_attempted",
            "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT",
            "ticket_artifact_status": "stale_artifact",
            "guard_alignment_brief": "risk_guard_candidate_mismatch:BNBUSDT->SOLUSDT",
            "blocker_detail": "no edge and stale ticket",
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_openclaw_orderflow_executor_state.json",
        {
            "executor_status": "shadow_guarded_executor_ready",
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "heartbeat_status": "shadow_guarded_idle",
            "idempotency_key_brief": "entry-1",
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": [
                        "ticket_missing:no_actionable_ticket",
                        "panic_cooldown_active",
                    ],
                }
            ]
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["feedback_status"] == "downrank_guardian_blocked_route"
    assert payload["queue_age_status"] == "queue_warming"
    assert payload["dominant_guard_reason"] == "ticket_missing:no_actionable_ticket"
    assert payload["feedback_symbol_entry_count"] == 1
    assert payload["routing_impact"] == "downrank_current_route_until_guardian_clear"
    assert payload["feedback_brief"].startswith("downrank_guardian_blocked_route:SOLUSDT")
    assert payload["blocker_detail"] == (
        "stale_artifact:ticket_row_missing:SOLUSDT"
        " | risk_guard_candidate_mismatch:BNBUSDT->SOLUSDT"
        " | ticket_missing:no_actionable_ticket,panic_cooldown_active"
        " | not_attempted_wait_trade_readiness"
        " | no_fill_execution_not_attempted"
    )
