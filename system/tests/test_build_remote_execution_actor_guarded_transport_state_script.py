from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_execution_actor_guarded_transport_state.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_execution_actor_guarded_transport_state_marks_learning_only_preview(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104020Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_learning_only:SOLUSDT:shadow_learning_no_transport:portfolio_margin_um",
            "actor_status": "shadow_actor_learning_only",
            "actor_service_name": "remote_execution_actor.service",
            "transport_phase": "shadow_learning_no_transport",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "blocker_detail": "guardian blocked",
        },
    )
    _write_json(
        review_dir / "20260315T104021Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_status": "shadow_policy_learning_only",
            "policy_decision": "accept_shadow_learning_only",
        },
    )
    _write_json(
        review_dir / "20260315T104022Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_learning_ack_recorded:SOLUSDT:not_sent_learning_only:no_fill_execution_not_attempted",
            "ack_status": "shadow_learning_ack_recorded",
            "ack_decision": "record_learning_without_transport",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:25Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["guarded_transport_status"] == "guarded_transport_preview_learning_only"
    assert payload["guarded_transport_decision"] == "keep_transport_disarmed_learning_only"
    assert payload["send_state"] == "not_armed_learning_only"
    assert payload["next_transition"] == "remote_execution_transport_sla"


def test_build_remote_execution_actor_guarded_transport_state_marks_pending_guard_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104020Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_waiting_guard_clear:SOLUSDT:candidate_pending_guard_clear:portfolio_margin_um",
            "actor_status": "shadow_actor_waiting_guard_clear",
            "actor_service_name": "remote_execution_actor.service",
            "transport_phase": "candidate_pending_guard_clear",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T104021Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_candidate_ready:SOLUSDT:queued_ticket_ready:feedback_cleared",
            "policy_status": "shadow_policy_candidate_ready",
            "policy_decision": "accept_shadow_candidate",
        },
    )
    _write_json(
        review_dir / "20260315T104022Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_transport_pending_ack:SOLUSDT:ready_not_sent_guard_clear_pending:no_fill_execution_not_attempted",
            "ack_status": "shadow_transport_pending_ack",
            "ack_decision": "wait_for_transport_ack",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:26Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["guarded_transport_status"] == "guarded_transport_preview_pending_guard_clear"
    assert payload["guarded_transport_decision"] == "wait_for_guardian_clear_before_send"
    assert payload["send_state"] == "armed_shadow_wait_guard_clear"


def test_build_remote_execution_actor_guarded_transport_state_marks_runtime_boundary_block(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104020Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_runtime_boundary_blocked:SOLUSDT:runtime_boundary_blocked_no_transport:spot",
            "actor_status": "shadow_actor_runtime_boundary_blocked",
            "actor_service_name": "remote_execution_actor.service",
            "transport_phase": "runtime_boundary_blocked_no_transport",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
            "blocker_detail": "requested runtime missing",
        },
    )
    _write_json(
        review_dir / "20260315T104021Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_runtime_boundary_blocked:SOLUSDT:queued_execution_contract_blocked:route_quality_ok",
            "policy_status": "shadow_policy_runtime_boundary_blocked",
            "policy_decision": "hold_requested_runtime_promotion",
        },
    )
    _write_json(
        review_dir / "20260315T104022Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_runtime_boundary_ack_recorded:SOLUSDT:not_sent_runtime_boundary_blocked:no_fill_execution_not_attempted",
            "ack_status": "shadow_runtime_boundary_ack_recorded",
            "ack_decision": "record_runtime_boundary_block_without_transport",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:27Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["guarded_transport_status"] == "guarded_transport_preview_runtime_boundary_blocked"
    assert payload["guarded_transport_decision"] == "do_not_arm_transport_runtime_boundary_blocked"
    assert payload["send_state"] == "not_armed_runtime_boundary_blocked"


def test_build_remote_execution_actor_guarded_transport_state_marks_guarded_probe_completed(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104020Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_guarded_probe_completed:SOLUSDT:guarded_probe_completed_no_live_send:spot",
            "actor_status": "shadow_actor_guarded_probe_completed",
            "actor_service_name": "remote_execution_actor.service",
            "transport_phase": "guarded_probe_completed_no_live_send",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
            "guarded_exec_probe_status": "probe_completed",
            "guarded_exec_probe_artifact": "/tmp/probe.json",
        },
    )
    _write_json(
        review_dir / "20260315T104021Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_runtime_boundary_blocked:SOLUSDT:queued_execution_contract_blocked:route_quality_ok",
            "policy_status": "shadow_policy_runtime_boundary_blocked",
            "policy_decision": "hold_requested_runtime_promotion",
        },
    )
    _write_json(
        review_dir / "20260315T104022Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_guarded_probe_ack_recorded:SOLUSDT:probe_completed_no_live_send:no_fill_execution_not_attempted",
            "ack_status": "shadow_guarded_probe_ack_recorded",
            "ack_decision": "record_guarded_probe_without_live_transport",
            "guarded_exec_probe_status": "probe_completed",
            "guarded_exec_probe_artifact": "/tmp/probe.json",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:40:28Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["guarded_transport_status"] == "guarded_transport_preview_probe_completed"
    assert payload["guarded_transport_decision"] == "keep_transport_disarmed_after_guarded_probe"
    assert payload["send_state"] == "probe_completed_no_live_send"
    assert payload["guarded_exec_probe_status"] == "probe_completed"
    assert payload["guarded_exec_probe_artifact"] == "/tmp/probe.json"


def test_build_remote_execution_actor_guarded_transport_state_marks_guarded_probe_candidate(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104020Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_guarded_probe_candidate:SOLUSDT:guarded_probe_candidate_no_live_send:spot",
            "actor_status": "shadow_actor_guarded_probe_candidate",
            "actor_service_name": "remote_execution_actor.service",
            "transport_phase": "guarded_probe_candidate_no_live_send",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
            "guarded_exec_probe_status": "ticket_not_allowed",
        },
    )
    _write_json(
        review_dir / "20260315T104021Z_remote_orderflow_policy_state.json",
        {
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_ticket_blocked:ticket_not_allowed",
            "policy_status": "shadow_policy_learning_only",
            "policy_decision": "accept_shadow_learning_only",
        },
    )
    _write_json(
        review_dir / "20260315T104022Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_guarded_probe_candidate_ack_recorded:SOLUSDT:probe_candidate_blocked_no_live_send:no_fill_execution_not_attempted",
            "ack_status": "shadow_guarded_probe_candidate_ack_recorded",
            "ack_decision": "record_guarded_probe_candidate_without_transport",
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
            "2026-03-15T10:40:29Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["guarded_transport_status"] == "guarded_transport_preview_probe_candidate_blocked"
    assert payload["guarded_transport_decision"] == "keep_transport_disarmed_probe_candidate"
    assert payload["send_state"] == "probe_candidate_blocked_no_live_send"
