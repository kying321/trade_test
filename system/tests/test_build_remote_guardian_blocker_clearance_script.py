from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_guardian_blocker_clearance.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_guardian_blocker_clearance_prioritizes_time_sync_and_ticket_readiness(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T113000Z_live_gate_blocker_report.json",
        {
            "status": "ok",
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked", "reason_codes": ["rollback_hard"]},
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket"],
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260315T113005Z_cross_market_operator_state.json",
        {
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
            "review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation. "
                "| time-sync=threshold_breach:scope=clock_skew_only"
            ),
            "review_head_done_when": "liquidity_sweep + mss + cvd_confirmation clear and time-sync clears",
        },
    )
    _write_json(
        review_dir / "20260315T113010Z_remote_orderflow_feedback.json",
        {
            "status": "ok",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT",
            "ticket_artifact_status": "stale_artifact",
            "dominant_guard_reason": "ticket_missing:no_actionable_ticket",
        },
    )
    _write_json(
        review_dir / "20260315T113015Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "accept_shadow_learning_only",
            "ticket_match_brief": "stale_artifact:ticket_row_missing:SOLUSDT",
            "risk_reason_codes": ["ticket_missing:no_actionable_ticket"],
        },
    )
    _write_json(
        review_dir / "20260315T113020Z_remote_orderflow_quality_report.json",
        {
            "status": "ok",
            "quality_brief": "quality_degraded_guardian_blocked_shadow_only:SOLUSDT:score_35:portfolio_margin_um",
            "quality_score": 35,
            "blocker_detail": "quality blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T113025Z_remote_live_boundary_hold.json",
        {
            "status": "ok",
            "hold_brief": "live_boundary_hold_active:SOLUSDT:guardian_review_blocked:portfolio_margin_um",
            "blocker_detail": "boundary blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T113027Z_remote_shadow_clock_evidence.json",
        {
            "status": "ok",
            "evidence_status": "shadow_clock_evidence_present",
            "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "shadow_learning_allowed": True,
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T113030Z_system_time_sync_repair_verification_report.json",
        {
            "status": "blocked",
            "verification_brief": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked",
            "cleared": False,
        },
    )
    _write_json(
        review_dir / "20260315T113031Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip",
            "environment_classification": "timed_ntp_via_fake_ip",
            "environment_blocker_detail": "timed_source=NTP; ntp_ip=198.18.0.17; delay_ms=118.702; clash_dns_mode=fake-ip; tun_stack=gvisor",
            "environment_remediation_hint": "exclude macOS timed / UDP 123 from Clash TUN fake-ip handling or provide a direct NTP path, then rerun time_sync_probe",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:35:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["clearance_status"] == "guardian_blocker_clearance_blocked"
    assert payload["route_symbol"] == "SOLUSDT"
    assert payload["top_blocker_code"] == "timed_ntp_via_fake_ip_clearance"
    assert payload["top_blocker_title"] == "Repair fake-ip NTP path before any orderflow promotion"
    assert payload["top_blocker_target_artifact"] == "system_time_sync_repair_verification_report"
    assert "manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip" in payload["top_blocker_detail"]
    assert payload["clearance_score"] == 0
    assert payload["blocked_count"] == 5
    assert payload["time_sync_mode"] == "promotion_blocked_shadow_learning_allowed"
    assert payload["remote_shadow_clock_shadow_learning_allowed"] is True
    assert payload["remote_shadow_clock_evidence_status"] == "shadow_clock_evidence_present"
    assert payload["clearance_items"][1]["blocker_code"] == "guardian_ticket_actionability"
    assert payload["clearance_items"][1]["status"] == "blocked"


def test_build_remote_guardian_blocker_clearance_prefers_ticket_actionability_once_time_sync_clears(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T113000Z_live_gate_blocker_report.json",
        {
            "status": "ok",
            "blockers": [
                {"name": "ops_live_gate", "status": "blocked", "reason_codes": ["rollback_hard"]},
                {
                    "name": "risk_guard",
                    "status": "blocked",
                    "reason_codes": ["ticket_missing:no_actionable_ticket"],
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260315T113005Z_cross_market_operator_state.json",
        {
            "review_head": {
                "status": "review",
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "priority_score": 58,
            },
            "review_head_blocker_detail": (
                "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, "
                "fvg_ob_breaker_retest, cvd_confirmation."
            ),
            "review_head_done_when": "liquidity_sweep + mss + cvd_confirmation clear",
        },
    )
    _write_json(
        review_dir / "20260315T113010Z_remote_orderflow_feedback.json",
        {
            "status": "ok",
            "feedback_status": "downrank_guardian_blocked_route",
            "feedback_brief": "downrank_guardian_blocked_route:SOLUSDT:queue_warming:ticket_missing:no_actionable_ticket",
            "ticket_match_brief": "fresh_artifact:ticket_row_missing:SOLUSDT",
            "ticket_artifact_status": "fresh_artifact",
            "dominant_guard_reason": "ticket_missing:no_actionable_ticket",
        },
    )
    _write_json(
        review_dir / "20260315T113015Z_remote_orderflow_policy_state.json",
        {
            "status": "ok",
            "policy_brief": "shadow_policy_learning_only:SOLUSDT:queued_wait_trade_readiness:downrank_guardian_blocked_route",
            "policy_decision": "accept_shadow_learning_only",
            "ticket_match_brief": "fresh_artifact:ticket_row_missing:SOLUSDT",
            "risk_reason_codes": ["ticket_missing:no_actionable_ticket"],
        },
    )
    _write_json(
        review_dir / "20260315T113018Z_remote_ticket_actionability_state.json",
        {
            "ticket_actionability_status": "crypto_shortline_bias_only_not_ticketed",
            "ticket_actionability_brief": "crypto_shortline_bias_only_not_ticketed:SOLUSDT:run_shortline_slice_backtest_and_wait_for_setup_ready:portfolio_margin_um",
            "ticket_actionability_decision": "run_shortline_slice_backtest_and_wait_for_setup_ready",
            "actionable_ready": False,
            "blocker_title": "Resolve crypto ticket actionability before guarded canary review",
            "blocker_target_artifact": "remote_ticket_actionability_state",
            "blocker_detail": "fresh_artifact:ticket_row_missing:SOLUSDT | ticket_surface=commodity_only:XAUUSD,XAGUSD,COPPER | shortline=Bias_Only; missing_gates=liquidity_sweep,mss,fvg_ob_breaker_retest,cvd_confirmation",
            "next_action": "run_shortline_slice_backtest_then_wait_for_setup_ready",
            "done_when": "SOLUSDT reaches Setup_Ready and a fresh crypto ticket row exists for SOLUSDT",
        },
    )
    _write_json(
        review_dir / "20260315T113020Z_remote_orderflow_quality_report.json",
        {
            "status": "ok",
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
            "blocker_detail": "quality remains learning-only",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T113025Z_remote_live_boundary_hold.json",
        {
            "status": "ok",
            "hold_brief": "live_boundary_hold_active:SOLUSDT:guardian_review_blocked:portfolio_margin_um",
            "blocker_detail": "guardian review blocked",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "time_sync_blocked": False,
        },
    )
    _write_json(
        review_dir / "20260315T113027Z_remote_shadow_clock_evidence.json",
        {
            "status": "ok",
            "evidence_status": "shadow_clock_evidence_present",
            "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "shadow_learning_allowed": True,
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T113030Z_system_time_sync_repair_verification_report.json",
        {
            "status": "ok",
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:35:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["top_blocker_code"] == "guardian_ticket_actionability"
    assert payload["top_blocker_title"] == (
        "Resolve crypto ticket actionability before guarded canary review"
    )
    assert payload["top_blocker_target_artifact"] == "remote_ticket_actionability_state"
    assert "fresh_artifact:ticket_row_missing:SOLUSDT" in payload["top_blocker_detail"]
    assert payload["time_sync_mode"] == "time_sync_clear_shadow_learning_allowed"
    assert payload["ticket_actionability_status"] == "crypto_shortline_bias_only_not_ticketed"
    assert "time-sync=" not in payload["blocker_detail"]
    assert "cleared:SOLUSDT:time_sync_ok" not in payload["blocker_detail"]
