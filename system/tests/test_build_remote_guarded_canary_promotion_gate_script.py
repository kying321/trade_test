from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_guarded_canary_promotion_gate.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_guarded_canary_promotion_gate_blocks_promotion_but_keeps_shadow_learning(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T114510Z_remote_execution_actor_canary_gate.json",
        {
            "canary_gate_status": "shadow_canary_gate_blocked",
            "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T114520Z_remote_orderflow_quality_report.json",
        {
            "quality_status": "quality_learning_only_shadow_viable",
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T114530Z_remote_live_boundary_hold.json",
        {
            "hold_brief": "live_boundary_hold_active:SOLUSDT:guardian_review_blocked:portfolio_margin_um",
            "guardian_blocked": True,
            "review_blocked": True,
            "time_sync_blocked": True,
            "time_sync_mode": "promotion_blocked_shadow_learning_allowed",
            "remote_shadow_clock_evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "remote_shadow_clock_evidence_status": "shadow_clock_evidence_present",
            "remote_shadow_clock_shadow_learning_allowed": True,
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T114540Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:5_blocked:portfolio_margin_um",
            "top_blocker_code": "timed_ntp_via_fake_ip_clearance",
            "top_blocker_title": "Repair fake-ip NTP path before any orderflow promotion",
            "top_blocker_target_artifact": "system_time_sync_repair_verification_report",
            "top_blocker_detail": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked | manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip",
            "top_blocker_done_when": "system time sync repair verification clears",
            "time_sync_mode": "promotion_blocked_shadow_learning_allowed",
            "remote_shadow_clock_evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "remote_shadow_clock_evidence_status": "shadow_clock_evidence_present",
            "remote_shadow_clock_shadow_learning_allowed": True,
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T114500Z_remote_shadow_clock_evidence.json",
        {
            "evidence_status": "shadow_clock_evidence_present",
            "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "shadow_learning_allowed": True,
        },
    )
    _write_json(
        review_dir / "20260315T114550Z_system_time_sync_repair_verification_report.json",
        {
            "status": "blocked",
            "verification_brief": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked",
            "cleared": False,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:46:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert (
        payload["promotion_gate_status"]
        == "guarded_canary_promotion_blocked_shadow_learning_allowed"
    )
    assert payload["promotion_gate_decision"] == "block_promotion_continue_shadow_learning"
    assert payload["shadow_learning_decision"] == "continue_shadow_learning_collect_feedback"
    assert payload["promotion_ready"] is False
    assert payload["shadow_learning_allowed"] is True
    assert payload["promotion_blocker_code"] == "timed_ntp_via_fake_ip_clearance"
    assert payload["promotion_blocker_title"] == "Repair fake-ip NTP path before any orderflow promotion"
    assert payload["promotion_blocker_target_artifact"] == "system_time_sync_repair_verification_report"
    assert payload["time_sync_mode"] == "promotion_blocked_shadow_learning_allowed"


def test_build_remote_guarded_canary_promotion_gate_prefers_latest_guardian_by_stamp_not_mtime(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T001450Z_remote_execution_actor_canary_gate.json",
        {
            "canary_gate_status": "shadow_canary_gate_blocked",
            "canary_gate_brief": "shadow_canary_gate_blocked:SOLUSDT:not_armed_guardian_blocked:portfolio_margin_um",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T001455Z_remote_orderflow_quality_report.json",
        {
            "quality_status": "quality_degraded_guardian_blocked_shadow_only",
            "quality_brief": "quality_degraded_guardian_blocked_shadow_only:SOLUSDT:score_44:portfolio_margin_um",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T001500Z_remote_live_boundary_hold.json",
        {
            "hold_brief": "live_boundary_hold_active:SOLUSDT:guardian_review_blocked:portfolio_margin_um",
            "guardian_blocked": True,
            "review_blocked": True,
            "time_sync_blocked": False,
            "time_sync_mode": "time_sync_clear_shadow_learning_allowed",
            "remote_shadow_clock_evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "remote_shadow_clock_evidence_status": "shadow_clock_evidence_present",
            "remote_shadow_clock_shadow_learning_allowed": True,
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    stale_guardian_path = review_dir / "20260316T001200Z_remote_guardian_blocker_clearance.json"
    _write_json(
        stale_guardian_path,
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Refresh ticket surface before guarded canary review",
            "top_blocker_target_artifact": "remote_ticket_actionability_state",
            "top_blocker_detail": "stale_artifact:ticket_row_blocked:SOLUSDT",
            "top_blocker_done_when": "latest signal_to_order_tickets row for SOLUSDT is fresh and actionable",
            "time_sync_mode": "time_sync_clear_shadow_learning_allowed",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    fresh_guardian_path = review_dir / "20260316T001710Z_remote_guardian_blocker_clearance.json"
    _write_json(
        fresh_guardian_path,
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track liquidity sweep before shortline setup promotion",
            "top_blocker_target_artifact": "crypto_shortline_gate_stack_progress",
            "top_blocker_detail": "fresh_artifact:ticket_row_blocked:SOLUSDT | gate_stack_progress=shortline_gate_stack_blocked_at_liquidity_sweep_proxy_price_blocked",
            "top_blocker_done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
            "time_sync_mode": "time_sync_clear_shadow_learning_allowed",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    stale_mtime = fresh_guardian_path.stat().st_mtime + 60
    os.utime(stale_guardian_path, (stale_mtime, stale_mtime))
    _write_json(
        review_dir / "20260316T001505Z_remote_shadow_clock_evidence.json",
        {
            "evidence_status": "shadow_clock_evidence_present",
            "evidence_brief": "shadow_clock_evidence_present:SOLUSDT:journal_last_entry<=heartbeat<=executor:portfolio_margin_um",
            "shadow_learning_allowed": True,
        },
    )
    _write_json(
        review_dir / "20260316T001510Z_system_time_sync_repair_verification_report.json",
        {
            "status": "cleared",
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
            "2026-03-16T00:17:20Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["promotion_blocker_code"] == "guardian_ticket_actionability"
    assert payload["promotion_blocker_title"] == "Track liquidity sweep before shortline setup promotion"
    assert payload["promotion_blocker_target_artifact"] == "crypto_shortline_gate_stack_progress"
    assert "gate_stack_progress=shortline_gate_stack_blocked_at_liquidity_sweep_proxy_price_blocked" in payload[
        "promotion_blocker_detail"
    ]
