from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_execution_journal.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_execution_journal_appends_guardian_blocked_entry(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
            "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT",
            "ticket_artifact_status": "stale_artifact",
            "guard_alignment_brief": "risk_guard_candidate_mismatch:BNBUSDT->SOLUSDT",
            "blocker_detail": "ticket row missing and guard candidate mismatch",
            "done_when": "ticket is ready and risk guard aligns",
            "ticket_artifact": str(review_dir / "20260315T095959Z_signal_to_order_tickets.json"),
        },
    )
    _write_json(
        review_dir / "20260315T100005Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket", "panic_cooldown_active"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked",
            "ready_check_scope_market": "portfolio_margin_um",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["journal_status"] == "intent_logged_guardian_blocked"
    assert payload["append_status"] == "appended"
    assert payload["entry_count"] == 1
    assert payload["execution_outcome"] == "not_attempted_wait_trade_readiness"
    assert payload["fill_status"] == "no_fill_execution_not_attempted"
    assert payload["risk_verdict_brief"] == "blocked:ticket_missing:no_actionable_ticket,panic_cooldown_active"
    journal_path = Path(str(payload["journal_path"]))
    assert journal_path.exists()
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_build_remote_execution_journal_skips_duplicate_entry_key(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_wait_trade_readiness",
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_ticket_ready",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
            "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT",
            "ticket_artifact_status": "stale_artifact",
            "guard_alignment_brief": "risk_guard_candidate_mismatch:BNBUSDT->SOLUSDT",
            "blocker_detail": "ticket row missing and guard candidate mismatch",
            "done_when": "ticket is ready and risk guard aligns",
            "ticket_artifact": str(review_dir / "20260315T095959Z_signal_to_order_tickets.json"),
        },
    )
    _write_json(
        review_dir / "20260315T100005Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked",
            "ready_check_scope_market": "portfolio_margin_um",
        },
    )
    run_args = [
        "python3",
        str(SCRIPT_PATH),
        "--review-dir",
        str(review_dir),
        "--now",
        "2026-03-15T10:01:00Z",
    ]

    first = subprocess.run(run_args, text=True, capture_output=True, check=False)
    second = subprocess.run(run_args, text=True, capture_output=True, check=False)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    payload = json.loads(second.stdout)
    assert payload["append_status"] == "duplicate_skipped"
    assert payload["entry_count"] == 1
    journal_path = Path(str(payload["journal_path"]))
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 1


def test_build_remote_execution_journal_marks_execution_contract_blocked_outcome(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "status": "ok",
            "queue_status": "queued_execution_contract_blocked",
            "queue_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:portfolio_margin_um",
            "queue_recommendation": "hold_remote_idle_until_execution_contract_promoted",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "portfolio_margin_um",
            "ticket_match_brief": "fresh_artifact:ticket_row_ready:SOLUSDT",
            "ticket_artifact_status": "fresh_artifact",
            "guard_alignment_brief": "risk_guard_not_ticket_blocked",
            "blocker_detail": "execution contract is shadow only",
            "done_when": "promote non-shadow execution contract",
            "ticket_artifact": str(review_dir / "20260315T095959Z_signal_to_order_tickets.json"),
        },
    )
    _write_json(
        review_dir / "20260315T100005Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {
                    "name": "remote_execution_contract",
                    "status": "blocked",
                    "reason_codes": ["shadow_executor_only_mode"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_remote_execution_identity_state.json",
        {
            "status": "ok",
            "identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked",
            "ready_check_scope_market": "portfolio_margin_um",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:01:01Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["execution_outcome"] == "not_attempted_execution_contract_blocked"
    assert payload["intent_status"] == "queued_execution_contract_blocked"
