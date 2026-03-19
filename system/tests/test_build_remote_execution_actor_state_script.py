from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_execution_actor_state.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_execution_actor_state_marks_learning_only_actor_ready(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104000Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T104001Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T104002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_wait_trade_readiness:SOLUSDT:deprioritize_flow:portfolio_margin_um | blocked:risk_guard | not_attempted_wait_trade_readiness",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-1",
        },
    )
    _write_json(
        review_dir / "20260315T104003Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_status": "shadow_policy_learning_only",
            "policy_decision": "accept_shadow_learning_only",
            "route_symbol": "SOLUSDT",
            "route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
            "blocker_detail": "guardian blocked",
        },
    )
    _write_json(
        review_dir / "20260315T104004Z_openclaw_orderflow_executor_state.json",
        {
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_wait_trade_readiness:portfolio_margin_um",
            "executor_status": "shadow_guarded_executor_ready",
            "service_name": "openclaw-orderflow-executor.service",
            "heartbeat_status": "shadow_guarded_idle",
        },
    )
    _write_json(
        review_dir / "20260315T104005Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_learning_ack_recorded:SOLUSDT:not_sent_learning_only:no_fill_execution_not_attempted",
            "ack_status": "shadow_learning_ack_recorded",
            "ack_decision": "record_learning_without_transport",
            "transport_state": "not_sent_learning_only",
            "route_symbol": "SOLUSDT",
            "route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
            "blocker_detail": "guardian blocked",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["actor_status"] == "shadow_actor_learning_only"
    assert payload["transport_phase"] == "shadow_learning_no_transport"
    assert payload["actor_service_name"] == "remote_execution_actor.service"
    assert payload["backing_service_name"] == "openclaw-orderflow-executor.service"
    assert payload["next_transition"] == "remote_execution_actor_guarded_transport"
    assert Path(str(payload["artifact"])).name == "20260315T104010Z_remote_execution_actor_state.json"


def test_build_remote_execution_actor_state_marks_pending_guard_clear_when_ack_waits(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104000Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:portfolio_margin_um:split_scope:blocked"},
    )
    _write_json(
        review_dir / "20260315T104001Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_ticket_ready:SOLUSDT:long_breakout_pullback:portfolio_margin_um",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "long_breakout_pullback",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T104002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_ticket_ready:SOLUSDT:long_breakout_pullback:portfolio_margin_um | ready:policy_clear | not_attempted",
            "journal_status": "intent_logged_ready",
            "last_entry_key": "entry-2",
        },
    )
    _write_json(
        review_dir / "20260315T104003Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_candidate_ready:SOLUSDT:queued_ticket_ready:feedback_cleared",
            "policy_status": "shadow_policy_candidate_ready",
            "policy_decision": "accept_shadow_candidate",
            "route_symbol": "SOLUSDT",
            "route_action": "long_breakout_pullback",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T104004Z_openclaw_orderflow_executor_state.json",
        {
            "executor_brief": "shadow_guarded_executor_ready:SOLUSDT:queued_ticket_ready:portfolio_margin_um",
            "executor_status": "shadow_guarded_executor_ready",
            "service_name": "openclaw-orderflow-executor.service",
            "heartbeat_status": "shadow_guarded_ticket_ready",
        },
    )
    _write_json(
        review_dir / "20260315T104005Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_transport_pending_ack:SOLUSDT:ready_not_sent_guard_clear_pending:no_fill_execution_not_attempted",
            "ack_status": "shadow_transport_pending_ack",
            "ack_decision": "wait_for_transport_ack",
            "transport_state": "ready_not_sent_guard_clear_pending",
            "route_symbol": "SOLUSDT",
            "route_action": "long_breakout_pullback",
            "remote_market": "portfolio_margin_um",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:11Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["actor_status"] == "shadow_actor_waiting_guard_clear"
    assert payload["transport_phase"] == "candidate_pending_guard_clear"
    assert payload["next_transition"] == "remote_execution_actor_guarded_transport"


def test_build_remote_execution_actor_state_surfaces_runtime_boundary_block(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104000Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:spot:spot_lane:promotion_requested"},
    )
    _write_json(
        review_dir / "20260315T104001Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T104002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot | contract:promotion_requested | not_attempted_execution_contract_blocked",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-3",
        },
    )
    _write_json(
        review_dir / "20260315T104003Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_runtime_boundary_blocked:SOLUSDT:queued_execution_contract_blocked:route_quality_ok",
            "policy_status": "shadow_policy_runtime_boundary_blocked",
            "policy_decision": "hold_requested_runtime_promotion",
            "route_symbol": "SOLUSDT",
            "route_action": "seed_ticket",
            "remote_market": "spot",
            "blocker_detail": "requested runtime missing",
        },
    )
    _write_json(
        review_dir / "20260315T104004Z_openclaw_orderflow_executor_state.json",
        {
            "executor_brief": "executor_runtime_boundary_blocked:SOLUSDT:queued_execution_contract_blocked:spot",
            "executor_status": "executor_runtime_boundary_blocked",
            "service_name": "openclaw-orderflow-executor.service",
            "heartbeat_status": "requested_mode_not_implemented",
            "runtime_boundary_status": "requested_runtime_not_implemented",
        },
    )
    _write_json(
        review_dir / "20260315T104005Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_runtime_boundary_ack_recorded:SOLUSDT:not_sent_runtime_boundary_blocked:no_fill_execution_not_attempted",
            "ack_status": "shadow_runtime_boundary_ack_recorded",
            "ack_decision": "record_runtime_boundary_block_without_transport",
            "transport_state": "not_sent_runtime_boundary_blocked",
            "route_symbol": "SOLUSDT",
            "route_action": "seed_ticket",
            "remote_market": "spot",
            "blocker_detail": "requested runtime missing",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:12Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["actor_status"] == "shadow_actor_runtime_boundary_blocked"
    assert payload["transport_phase"] == "runtime_boundary_blocked_no_transport"
    assert payload["executor_runtime_boundary_status"] == "requested_runtime_not_implemented"


def test_build_remote_execution_actor_state_surfaces_guarded_probe_completion(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104000Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:spot:spot_lane:promotion_requested"},
    )
    _write_json(
        review_dir / "20260315T104001Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "seed_ticket",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T104002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_execution_contract_blocked:SOLUSDT:seed_ticket:spot | contract:promotion_requested | probe_completed",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-4",
        },
    )
    _write_json(
        review_dir / "20260315T104003Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_runtime_boundary_blocked:SOLUSDT:queued_execution_contract_blocked:route_quality_ok",
            "policy_status": "shadow_policy_runtime_boundary_blocked",
            "policy_decision": "hold_requested_runtime_promotion",
            "route_symbol": "SOLUSDT",
            "route_action": "seed_ticket",
            "remote_market": "spot",
            "blocker_detail": "requested runtime missing",
        },
    )
    _write_json(
        review_dir / "20260315T104004Z_openclaw_orderflow_executor_state.json",
        {
            "executor_brief": "spot_live_guarded_probe_completed:SOLUSDT:queued_execution_contract_blocked:spot",
            "executor_status": "spot_live_guarded_probe_completed",
            "service_name": "openclaw-orderflow-executor.service",
            "heartbeat_status": "spot_live_guarded_probe_completed",
            "runtime_boundary_status": "requested_runtime_not_implemented",
            "guarded_exec_probe_status": "probe_completed",
            "guarded_exec_probe_artifact": "/tmp/probe.json",
        },
    )
    _write_json(
        review_dir / "20260315T104005Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_guarded_probe_ack_recorded:SOLUSDT:probe_completed_no_live_send:no_fill_execution_not_attempted",
            "ack_status": "shadow_guarded_probe_ack_recorded",
            "ack_decision": "record_guarded_probe_without_live_transport",
            "transport_state": "probe_completed_no_live_send",
            "route_symbol": "SOLUSDT",
            "route_action": "seed_ticket",
            "remote_market": "spot",
            "guarded_exec_probe_status": "probe_completed",
            "guarded_exec_probe_artifact": "/tmp/probe.json",
            "blocker_detail": "requested runtime missing",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:13Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["actor_status"] == "shadow_actor_guarded_probe_completed"
    assert payload["transport_phase"] == "guarded_probe_completed_no_live_send"
    assert payload["guarded_exec_probe_status"] == "probe_completed"
    assert payload["guarded_exec_probe_artifact"] == "/tmp/probe.json"


def test_build_remote_execution_actor_state_surfaces_guarded_probe_candidate(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104000Z_remote_execution_identity_state.json",
        {"identity_brief": "43.153.148.242:spot:spot_lane:probe_only_contract"},
    )
    _write_json(
        review_dir / "20260315T104001Z_remote_intent_queue.json",
        {
            "queue_brief": "queued_ticket_blocked:SOLUSDT:consider_refresh_before_promotion:spot",
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "consider_refresh_before_promotion",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T104002Z_remote_execution_journal.json",
        {
            "journal_brief": "queued_ticket_blocked:SOLUSDT:consider_refresh_before_promotion:spot | blocked:ticket_missing:no_actionable_ticket | no_probe",
            "journal_status": "intent_logged_guardian_blocked",
            "last_entry_key": "entry-probe-candidate-2",
        },
    )
    _write_json(
        review_dir / "20260315T104003Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_ticket_blocked:ticket_not_allowed",
            "policy_status": "shadow_policy_learning_only",
            "policy_decision": "accept_shadow_learning_only",
            "route_symbol": "SOLUSDT",
            "route_action": "consider_refresh_before_promotion",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T104004Z_openclaw_orderflow_executor_state.json",
        {
            "executor_brief": "spot_live_guarded_probe_capable:SOLUSDT:queued_ticket_blocked:spot",
            "executor_status": "spot_live_guarded_probe_capable",
            "service_name": "openclaw-orderflow-executor.service",
            "heartbeat_status": "spot_live_guarded_probe_capable",
            "runtime_boundary_status": "guarded_probe_only_runtime",
            "guarded_exec_probe_status": "ticket_not_allowed",
        },
    )
    _write_json(
        review_dir / "20260315T104005Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_guarded_probe_candidate_ack_recorded:SOLUSDT:probe_candidate_blocked_no_live_send:no_fill_execution_not_attempted",
            "ack_status": "shadow_guarded_probe_candidate_ack_recorded",
            "ack_decision": "record_guarded_probe_candidate_without_transport",
            "transport_state": "probe_candidate_blocked_no_live_send",
            "route_symbol": "SOLUSDT",
            "route_action": "consider_refresh_before_promotion",
            "remote_market": "spot",
            "guarded_exec_probe_status": "ticket_not_allowed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:14Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["actor_status"] == "shadow_actor_guarded_probe_candidate"
    assert payload["transport_phase"] == "guarded_probe_candidate_no_live_send"
    assert payload["guarded_exec_probe_status"] == "ticket_not_allowed"
