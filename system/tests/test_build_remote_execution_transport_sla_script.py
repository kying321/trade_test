from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_execution_transport_sla.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_execution_transport_sla_marks_learning_only_no_send(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104200Z_remote_execution_actor_guarded_transport.json",
        {
            "guarded_transport_brief": "guarded_transport_preview_learning_only:SOLUSDT:not_armed_learning_only:portfolio_margin_um",
            "guarded_transport_status": "guarded_transport_preview_learning_only",
            "guarded_transport_decision": "keep_transport_disarmed_learning_only",
            "send_state": "not_armed_learning_only",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T104201Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_learning_ack_recorded:SOLUSDT:not_sent_learning_only:no_fill_execution_not_attempted",
            "ack_status": "shadow_learning_ack_recorded",
            "ack_decision": "record_learning_without_transport",
        },
    )
    _write_json(
        review_dir / "20260315T104202Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_learning_only:SOLUSDT:shadow_learning_no_transport:portfolio_margin_um",
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
            "2026-03-15T10:42:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["transport_sla_status"] == "shadow_transport_sla_learning_only_no_send"
    assert payload["transport_sla_decision"] == "accumulate_learning_samples_before_guard_clear"
    assert payload["send_sample_count"] == 0
    assert payload["next_transition"] == "remote_execution_actor_canary_gate"


def test_build_remote_execution_transport_sla_marks_pending_guard_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104200Z_remote_execution_actor_guarded_transport.json",
        {
            "guarded_transport_brief": "guarded_transport_preview_pending_guard_clear:SOLUSDT:armed_shadow_wait_guard_clear:portfolio_margin_um",
            "guarded_transport_status": "guarded_transport_preview_pending_guard_clear",
            "guarded_transport_decision": "wait_for_guardian_clear_before_send",
            "send_state": "armed_shadow_wait_guard_clear",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T104201Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_transport_pending_ack:SOLUSDT:ready_not_sent_guard_clear_pending:no_fill_execution_not_attempted",
            "ack_status": "shadow_transport_pending_ack",
            "ack_decision": "wait_for_transport_ack",
        },
    )
    _write_json(
        review_dir / "20260315T104202Z_remote_execution_actor_state.json",
        {"actor_brief": "shadow_actor_waiting_guard_clear:SOLUSDT:candidate_pending_guard_clear:portfolio_margin_um"},
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:42:11Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["transport_sla_status"] == "shadow_transport_sla_pending_guard_clear"
    assert payload["transport_sla_decision"] == "wait_for_first_guarded_send_sample"
    assert payload["ack_sample_count"] == 0


def test_build_remote_execution_transport_sla_marks_runtime_boundary_blocked_no_send(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104200Z_remote_execution_actor_guarded_transport.json",
        {
            "guarded_transport_brief": "guarded_transport_preview_runtime_boundary_blocked:SOLUSDT:not_armed_runtime_boundary_blocked:spot",
            "guarded_transport_status": "guarded_transport_preview_runtime_boundary_blocked",
            "guarded_transport_decision": "do_not_arm_transport_runtime_boundary_blocked",
            "send_state": "not_armed_runtime_boundary_blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T104201Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_runtime_boundary_ack_recorded:SOLUSDT:not_sent_runtime_boundary_blocked:no_fill_execution_not_attempted",
            "ack_status": "shadow_runtime_boundary_ack_recorded",
            "ack_decision": "record_runtime_boundary_block_without_transport",
        },
    )
    _write_json(
        review_dir / "20260315T104202Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_runtime_boundary_blocked:SOLUSDT:runtime_boundary_blocked_no_transport:spot",
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
            "2026-03-15T10:42:12Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["transport_sla_status"] == "shadow_transport_sla_runtime_boundary_blocked_no_send"
    assert payload["transport_sla_decision"] == "implement_runtime_before_transport_sla"
    assert payload["send_sample_count"] == 0


def test_build_remote_execution_transport_sla_records_guarded_probe_sample_without_live_send(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104200Z_remote_execution_actor_guarded_transport.json",
        {
            "guarded_transport_brief": "guarded_transport_preview_probe_completed:SOLUSDT:probe_completed_no_live_send:spot",
            "guarded_transport_status": "guarded_transport_preview_probe_completed",
            "guarded_transport_decision": "keep_transport_disarmed_after_guarded_probe",
            "send_state": "probe_completed_no_live_send",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
            "guarded_exec_probe_status": "probe_completed",
            "guarded_exec_probe_artifact": "/tmp/probe.json",
        },
    )
    _write_json(
        review_dir / "20260315T104201Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_guarded_probe_ack_recorded:SOLUSDT:probe_completed_no_live_send:no_fill_execution_not_attempted",
            "ack_status": "shadow_guarded_probe_ack_recorded",
            "ack_decision": "record_guarded_probe_without_live_transport",
        },
    )
    _write_json(
        review_dir / "20260315T104202Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_guarded_probe_completed:SOLUSDT:guarded_probe_completed_no_live_send:spot",
            "blocker_detail": "requested runtime missing",
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
            "2026-03-15T10:42:13Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["transport_sla_status"] == "shadow_transport_sla_probe_completed_no_send"
    assert payload["transport_sla_decision"] == "record_guarded_probe_sample_without_live_send"
    assert payload["probe_sample_count"] == 1
    assert payload["guarded_exec_probe_status"] == "probe_completed"
    assert payload["guarded_exec_probe_artifact"] == "/tmp/probe.json"


def test_build_remote_execution_transport_sla_records_guarded_probe_candidate_without_live_send(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T104200Z_remote_execution_actor_guarded_transport.json",
        {
            "guarded_transport_brief": "guarded_transport_preview_probe_candidate_blocked:SOLUSDT:probe_candidate_blocked_no_live_send:spot",
            "guarded_transport_status": "guarded_transport_preview_probe_candidate_blocked",
            "guarded_transport_decision": "keep_transport_disarmed_probe_candidate",
            "send_state": "probe_candidate_blocked_no_live_send",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
            "guarded_exec_probe_status": "ticket_not_allowed",
        },
    )
    _write_json(
        review_dir / "20260315T104201Z_remote_execution_ack_state.json",
        {
            "ack_brief": "shadow_guarded_probe_candidate_ack_recorded:SOLUSDT:probe_candidate_blocked_no_live_send:no_fill_execution_not_attempted",
            "ack_status": "shadow_guarded_probe_candidate_ack_recorded",
            "ack_decision": "record_guarded_probe_candidate_without_transport",
        },
    )
    _write_json(
        review_dir / "20260315T104202Z_remote_execution_actor_state.json",
        {
            "actor_brief": "shadow_actor_guarded_probe_candidate:SOLUSDT:guarded_probe_candidate_no_live_send:spot",
            "blocker_detail": "ticket not allowed",
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
            "2026-03-15T10:42:14Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["transport_sla_status"] == "shadow_transport_sla_probe_candidate_no_send"
    assert payload["transport_sla_decision"] == "record_probe_candidate_without_live_send"
    assert payload["probe_sample_count"] == 0
    assert payload["guarded_exec_probe_status"] == "ticket_not_allowed"
