from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_execution_actor_canary_gate.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_execution_actor_canary_gate_blocks_shadow_canary_when_guardian_not_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T111400Z_remote_execution_transport_sla.json",
        {
            "status": "ok",
            "transport_sla_status": "shadow_transport_sla_blocked_no_send",
            "transport_sla_brief": "shadow_transport_sla_blocked_no_send:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um",
            "blocker_detail": "transport blocked",
        },
    )
    _write_json(
        review_dir / "20260315T111200Z_remote_execution_actor_guarded_transport.json",
        {
            "status": "ok",
            "guarded_transport_status": "guarded_transport_preview_blocked",
            "guarded_transport_brief": "guarded_transport_preview_blocked:SOLUSDT:not_armed_policy_blocked:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T111020Z_remote_execution_actor_state.json",
        {
            "status": "ok",
            "actor_status": "shadow_actor_ready_policy_blocked",
            "actor_brief": "shadow_actor_ready_policy_blocked:SOLUSDT:shadow_only_no_transport:portfolio_margin_um",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T111015Z_remote_execution_ack_state.json",
        {
            "status": "ok",
            "ack_status": "shadow_no_send_ack_recorded",
            "ack_brief": "shadow_no_send_ack_recorded:SOLUSDT:not_sent_policy_blocked:no_fill_execution_not_attempted",
            "blocker_detail": "ack blocked",
        },
    )
    _write_json(
        review_dir / "20260315T103500Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_blocked",
            "policy_brief": "shadow_policy_blocked:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "reject_until_guardian_clear",
            "queue_status": "queued_wait_trade_readiness",
            "scope_router_status": "review_candidate_inside_scope_not_trade_ready",
            "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT",
            "risk_reason_codes": ["ticket_missing:no_actionable_ticket"],
            "blocker_detail": "policy blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked"},
                {"name": "risk_guard", "status": "blocked"},
            ]
        },
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_status": "local_operator_head_outside_remote_live_scope",
            "remote_live_takeover_gate_blocker_detail": "current head outside remote live scope",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:16:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["canary_gate_status"] == "shadow_canary_gate_blocked"
    assert payload["canary_gate_decision"] == "deny_canary_until_guardian_clear"
    assert payload["arm_state"] == "not_armed_guardian_blocked"
    assert payload["next_transition"] == "remote_orderflow_quality_report"


def test_build_remote_execution_actor_canary_gate_surfaces_runtime_boundary_before_guardian_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T111400Z_remote_execution_transport_sla.json",
        {
            "status": "ok",
            "transport_sla_status": "shadow_transport_sla_runtime_boundary_blocked_no_send",
            "transport_sla_brief": "shadow_transport_sla_runtime_boundary_blocked_no_send:SOLUSDT:not_armed_runtime_boundary_blocked:spot",
            "blocker_detail": "requested runtime missing",
        },
    )
    _write_json(
        review_dir / "20260315T111200Z_remote_execution_actor_guarded_transport.json",
        {
            "status": "ok",
            "guarded_transport_status": "guarded_transport_preview_runtime_boundary_blocked",
            "guarded_transport_brief": "guarded_transport_preview_runtime_boundary_blocked:SOLUSDT:not_armed_runtime_boundary_blocked:spot",
        },
    )
    _write_json(
        review_dir / "20260315T111020Z_remote_execution_actor_state.json",
        {
            "status": "ok",
            "actor_status": "shadow_actor_runtime_boundary_blocked",
            "actor_brief": "shadow_actor_runtime_boundary_blocked:SOLUSDT:runtime_boundary_blocked_no_transport:spot",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T111015Z_remote_execution_ack_state.json",
        {
            "status": "ok",
            "ack_status": "shadow_runtime_boundary_ack_recorded",
            "ack_brief": "shadow_runtime_boundary_ack_recorded:SOLUSDT:not_sent_runtime_boundary_blocked:no_fill_execution_not_attempted",
            "blocker_detail": "ack blocked",
        },
    )
    _write_json(
        review_dir / "20260315T103500Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_runtime_boundary_blocked",
            "policy_brief": "shadow_policy_runtime_boundary_blocked:SOLUSDT:queued_execution_contract_blocked:route_quality_ok",
            "policy_decision": "hold_requested_runtime_promotion",
            "queue_status": "queued_execution_contract_blocked",
            "scope_router_status": "review_candidate_inside_remote_scope",
            "ticket_match_brief": "fresh_artifact:ticket_row_ready:SOLUSDT",
            "risk_reason_codes": ["ticket_missing:no_actionable_ticket"],
            "blocker_detail": "policy blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked"},
                {"name": "risk_guard", "status": "blocked"},
            ]
        },
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_status": "remote_live_operator_scope_aligned",
            "remote_live_takeover_gate_blocker_detail": "runtime boundary still blocked",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "seed_ticket",
                "priority_score": 58,
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:16:01Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["canary_gate_status"] == "shadow_canary_gate_runtime_boundary_blocked"
    assert payload["canary_gate_decision"] == "deny_canary_until_runtime_boundary_clears"
    assert payload["arm_state"] == "not_armed_runtime_boundary_blocked"


def test_build_remote_execution_actor_canary_gate_surfaces_guarded_probe_before_runtime_boundary(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T111400Z_remote_execution_transport_sla.json",
        {
            "status": "ok",
            "transport_sla_status": "shadow_transport_sla_probe_completed_no_send",
            "transport_sla_brief": "shadow_transport_sla_probe_completed_no_send:SOLUSDT:probe_completed_no_live_send:spot",
            "guarded_exec_probe_status": "probe_completed",
            "guarded_exec_probe_artifact": "/tmp/probe.json",
            "blocker_detail": "requested runtime missing",
        },
    )
    _write_json(
        review_dir / "20260315T111200Z_remote_execution_actor_guarded_transport.json",
        {
            "status": "ok",
            "guarded_transport_status": "guarded_transport_preview_probe_completed",
            "guarded_transport_brief": "guarded_transport_preview_probe_completed:SOLUSDT:probe_completed_no_live_send:spot",
            "guarded_exec_probe_status": "probe_completed",
            "guarded_exec_probe_artifact": "/tmp/probe.json",
        },
    )
    _write_json(
        review_dir / "20260315T111020Z_remote_execution_actor_state.json",
        {
            "status": "ok",
            "actor_status": "shadow_actor_guarded_probe_completed",
            "actor_brief": "shadow_actor_guarded_probe_completed:SOLUSDT:guarded_probe_completed_no_live_send:spot",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T111015Z_remote_execution_ack_state.json",
        {
            "status": "ok",
            "ack_status": "shadow_guarded_probe_ack_recorded",
            "ack_brief": "shadow_guarded_probe_ack_recorded:SOLUSDT:probe_completed_no_live_send:no_fill_execution_not_attempted",
            "blocker_detail": "ack blocked",
        },
    )
    _write_json(
        review_dir / "20260315T103500Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_runtime_boundary_blocked",
            "policy_brief": "shadow_policy_runtime_boundary_blocked:SOLUSDT:queued_execution_contract_blocked:route_quality_ok",
            "policy_decision": "hold_requested_runtime_promotion",
            "queue_status": "queued_execution_contract_blocked",
            "scope_router_status": "review_candidate_inside_remote_scope",
            "ticket_match_brief": "fresh_artifact:ticket_row_ready:SOLUSDT",
            "risk_reason_codes": ["ticket_missing:no_actionable_ticket"],
            "blocker_detail": "policy blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked"},
                {"name": "risk_guard", "status": "blocked"},
            ]
        },
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_status": "remote_live_operator_scope_aligned",
            "remote_live_takeover_gate_blocker_detail": "probe completed without live send",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "seed_ticket",
                "priority_score": 58,
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:16:02Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["canary_gate_status"] == "shadow_canary_gate_probe_completed_no_send"
    assert payload["canary_gate_decision"] == "review_probe_evidence_before_canary_promotion"
    assert payload["arm_state"] == "probe_completed_review_only"
    assert payload["guarded_exec_probe_status"] == "probe_completed"
    assert payload["guarded_exec_probe_artifact"] == "/tmp/probe.json"


def test_build_remote_execution_actor_canary_gate_blocks_probe_candidate_until_ticket_clears(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T111400Z_remote_execution_transport_sla.json",
        {
            "status": "ok",
            "transport_sla_status": "shadow_transport_sla_probe_candidate_no_send",
            "transport_sla_brief": "shadow_transport_sla_probe_candidate_no_send:SOLUSDT:probe_candidate_blocked_no_live_send:spot",
            "guarded_exec_probe_status": "ticket_not_allowed",
            "blocker_detail": "ticket not allowed",
        },
    )
    _write_json(
        review_dir / "20260315T111200Z_remote_execution_actor_guarded_transport.json",
        {
            "status": "ok",
            "guarded_transport_status": "guarded_transport_preview_probe_candidate_blocked",
            "guarded_transport_brief": "guarded_transport_preview_probe_candidate_blocked:SOLUSDT:probe_candidate_blocked_no_live_send:spot",
        },
    )
    _write_json(
        review_dir / "20260315T111020Z_remote_execution_actor_state.json",
        {
            "status": "ok",
            "actor_status": "shadow_actor_guarded_probe_candidate",
            "actor_brief": "shadow_actor_guarded_probe_candidate:SOLUSDT:guarded_probe_candidate_no_live_send:spot",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260315T111015Z_remote_execution_ack_state.json",
        {
            "status": "ok",
            "ack_status": "shadow_guarded_probe_candidate_ack_recorded",
            "ack_brief": "shadow_guarded_probe_candidate_ack_recorded:SOLUSDT:probe_candidate_blocked_no_live_send:no_fill_execution_not_attempted",
            "blocker_detail": "ack blocked",
        },
    )
    _write_json(
        review_dir / "20260315T103500Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_status": "shadow_policy_learning_only",
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_ticket_blocked:ticket_not_allowed",
            "policy_decision": "accept_shadow_learning_only",
            "queue_status": "queued_ticket_blocked",
            "scope_router_status": "review_candidate_inside_remote_scope",
            "ticket_match_brief": "fresh_artifact:ticket_row_blocked:SOLUSDT",
            "risk_reason_codes": ["ticket_missing:no_actionable_ticket"],
            "blocker_detail": "policy blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260314T140104Z_live_gate_blocker_report.json",
        {
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked"},
                {"name": "risk_guard", "status": "blocked"},
            ]
        },
    )
    _write_json(
        review_dir / "20260314T190200Z_cross_market_operator_state.json",
        {
            "remote_live_operator_alignment_status": "remote_live_operator_scope_aligned",
            "remote_live_takeover_gate_blocker_detail": "ticket still blocks guarded probe",
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "consider_refresh_before_promotion",
                "priority_score": 58,
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:16:03Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["canary_gate_status"] == "shadow_canary_gate_probe_candidate_blocked"
    assert payload["canary_gate_decision"] == "deny_canary_until_probe_candidate_clears"
    assert payload["arm_state"] == "not_armed_probe_candidate_blocked"
