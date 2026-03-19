from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_promotion_unblock_readiness.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_promotion_unblock_readiness_marks_local_time_sync_as_primary_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_shadow_learning_allowed",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_shadow_learning_allowed:SOLUSDT:block_promotion_continue_shadow_learning:portfolio_margin_um",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "system_time_sync_repair_verification_report",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:5_blocked:portfolio_margin_um",
            "top_blocker_title": "Repair time sync before any orderflow promotion",
            "top_blocker_target_artifact": "system_time_sync_repair_verification_report",
            "top_blocker_detail": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked",
            "cleared": False,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
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
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "local_time_sync_primary_blocker_shadow_ready"
    assert payload["readiness_decision"] == "repair_local_fake_ip_ntp_path_then_review_guarded_canary"
    assert payload["remote_preconditions_status"] == "shadow_ready_remote_preconditions_viable"
    assert payload["primary_blocker_scope"] == "timed_ntp_via_fake_ip"
    assert payload["primary_local_repair_required"] is True
    assert payload["primary_local_repair_title"] == (
        "Repair local fake-ip NTP path to unlock guarded canary review"
    )
    assert payload["primary_local_repair_target_artifact"] == (
        "system_time_sync_repair_verification_report"
    )
    assert (
        payload["primary_local_repair_plan_brief"]
        == "manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip"
    )
    assert (
        payload["primary_local_repair_environment_classification"]
        == "timed_ntp_via_fake_ip"
    )
    assert payload["primary_local_repair_detail"] == (
        "blocked:SOLUSDT:probe_blocked+environment_blocked+review_head_time_sync_blocked"
        " | repair_plan=manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip"
        " | time_sync_env=timed_ntp_via_fake_ip:timed_source=NTP; ntp_ip=198.18.0.17; delay_ms=118.702; clash_dns_mode=fake-ip; tun_stack=gvisor"
        " | time_sync_fix_hint=exclude macOS timed / UDP 123 from Clash TUN fake-ip handling or provide a direct NTP path, then rerun time_sync_probe"
    )
    assert "repair_plan=manual_time_repair_required:SOLUSDT:timed_ntp_via_fake_ip" in payload[
        "blocker_detail"
    ]
    assert "time_sync_env=timed_ntp_via_fake_ip:timed_source=NTP; ntp_ip=198.18.0.17" in payload[
        "blocker_detail"
    ]
    assert payload["blocker_detail"].count("ntp_ip=") == 1


def test_build_remote_promotion_unblock_readiness_marks_ticket_actionability_as_primary_blocker_once_time_sync_clears(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "remote_ticket_actionability_state",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Generate fresh crypto signal source before guarded canary review",
            "top_blocker_target_artifact": "remote_ticket_actionability_state",
            "top_blocker_next_action": "generate_fresh_crypto_signal_source_before_rebuild_tickets",
            "top_blocker_detail": "fresh_artifact:ticket_row_missing:SOLUSDT | ticket_surface=commodity_only:XAUUSD,XAGUSD,COPPER",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS; os_time_service is not using NTP as the active source",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert payload["readiness_decision"] == "generate_fresh_crypto_signal_source_then_review_guarded_canary"
    assert payload["remote_preconditions_status"] == "shadow_ready_remote_preconditions_viable"
    assert payload["primary_blocker_scope"] == "guardian_ticket_actionability"
    assert payload["primary_local_repair_required"] is False
    assert payload["primary_local_repair_title"] == (
        "Generate fresh crypto signal source before guarded canary review"
    )
    assert payload["primary_local_repair_target_artifact"] == "remote_ticket_actionability_state"
    assert payload["primary_local_repair_detail"] == (
        "fresh_artifact:ticket_row_missing:SOLUSDT | ticket_surface=commodity_only:XAUUSD,XAGUSD,COPPER"
    )
    assert payload["primary_local_repair_plan_brief"] == ""
    assert payload["primary_local_repair_environment_classification"] == ""
    assert payload["primary_local_repair_environment_blocker_detail"] == ""


def test_build_remote_promotion_unblock_readiness_ignores_future_stamped_guardian_clearance(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T051100Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "remote_ticket_actionability_state",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260316T051110Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    current_guardian_path = review_dir / "20260316T051120Z_remote_guardian_blocker_clearance.json"
    _write_json(
        current_guardian_path,
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track LVN-to-HVN/POC rotation before shortline setup promotion",
            "top_blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "top_blocker_next_action": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "top_blocker_detail": "fresh_artifact:ticket_row_blocked:SOLUSDT | profile_location_watch=profile_location_lvn_key_level_ready_rotation_far",
        },
    )
    future_guardian_path = review_dir / "20260316T082500Z_remote_guardian_blocker_clearance.json"
    _write_json(
        future_guardian_path,
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Refresh ticket surface before guarded canary review",
            "top_blocker_target_artifact": "signal_to_order_tickets",
            "top_blocker_next_action": "refresh_signal_to_order_tickets",
            "top_blocker_detail": "stale_artifact:ticket_row_missing:SOLUSDT",
        },
    )
    future_mtime = current_guardian_path.stat().st_mtime + 60
    os.utime(future_guardian_path, (future_mtime, future_mtime))
    _write_json(
        review_dir / "20260316T051130Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260316T051140Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260316T051145Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS; os_time_service is not using NTP as the active source",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T05:12:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_local_repair_title"] == (
        "Track LVN-to-HVN/POC rotation before shortline setup promotion"
    )
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["readiness_decision"] == "monitor_profile_rotation_toward_hvn_poc_then_review_guarded_canary"


def test_build_remote_promotion_unblock_readiness_uses_signal_quality_watch_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "remote_ticket_actionability_state",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Improve shortline signal quality before guarded canary review",
            "top_blocker_target_artifact": "crypto_shortline_signal_quality_watch",
            "top_blocker_next_action": "improve_shortline_signal_quality_then_recheck_execution_gate",
            "top_blocker_detail": "signal_quality_watch=signal_confidence_below_threshold:SOLUSDT:improve_shortline_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS; os_time_service is not using NTP as the active source",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "improve_shortline_signal_quality_then_review_guarded_canary"
    )
    assert payload["primary_blocker_scope"] == "guardian_ticket_actionability"
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_signal_quality_watch"
    assert payload["primary_local_repair_environment_remediation_hint"] == ""
    assert (
        "signal_quality_watch=signal_confidence_below_threshold:SOLUSDT:improve_shortline_signal_quality_then_recheck_execution_gate:portfolio_margin_um"
        in payload["blocker_detail"]
    )
    assert "quality_score=49" in payload["blocker_detail"]
    assert "a fresh actionable ticket row exists" in payload["done_when"]


def test_build_remote_promotion_unblock_readiness_uses_fill_capacity_watch_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T093100Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "remote_ticket_actionability_state",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260316T093110Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T093120Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Repair value-rotation fill capacity before guarded canary review",
            "top_blocker_target_artifact": "crypto_shortline_fill_capacity_watch",
            "top_blocker_next_action": "repair_value_rotation_fill_capacity_then_recheck_execution_gate",
            "top_blocker_detail": "fill_capacity_watch=value_rotation_scalp_fill_capacity_constrained:SOLUSDT:repair_value_rotation_fill_capacity_then_recheck_execution_gate:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T093130Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260316T093140Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260316T093145Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "",
            "environment_classification": "",
            "environment_blocker_detail": "",
            "environment_remediation_hint": "",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:31:50Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "repair_value_rotation_fill_capacity_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_fill_capacity_watch"


def test_build_remote_promotion_unblock_readiness_uses_pattern_router_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T062530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "remote_ticket_actionability_state",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260316T062540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T062520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track value-rotation alignment before shortline scalp promotion",
            "top_blocker_target_artifact": "crypto_shortline_pattern_router",
            "top_blocker_next_action": "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "top_blocker_detail": "pattern_router=value_rotation_scalp_wait_profile_alignment_far:SOLUSDT:monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um:value_rotation_scalp",
        },
    )
    _write_json(
        review_dir / "20260316T062500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260316T062510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260316T062515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS; os_time_service is not using NTP as the active source",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T06:26:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert (
        payload["readiness_decision"]
        == "monitor_value_rotation_toward_hvn_poc_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_pattern_router"


def test_build_remote_promotion_unblock_readiness_uses_cvd_watch_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T093000Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "remote_ticket_actionability_state",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260316T093010Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T093020Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track CVD confirmation after MSS before shortline setup promotion",
            "top_blocker_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "top_blocker_next_action": "wait_for_cvd_confirmation_then_recheck_execution_gate",
            "top_blocker_detail": "cvd_confirmation_watch=cvd_confirmation_waiting_after_mss:SOLUSDT:wait_for_cvd_confirmation_then_recheck_execution_gate:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T093030Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260316T093040Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260316T093045Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:31:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert payload["readiness_decision"] == "wait_for_cvd_confirmation_then_review_guarded_canary"
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_cvd_confirmation_watch"


def test_build_remote_promotion_unblock_readiness_uses_sizing_watch_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "remote_ticket_actionability_state",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Raise effective shortline size before guarded canary review",
            "top_blocker_target_artifact": "crypto_shortline_sizing_watch",
            "top_blocker_next_action": "raise_effective_shortline_size_then_recheck_execution_gate",
            "top_blocker_detail": "sizing_watch=ticket_size_below_min_notional:SOLUSDT:raise_effective_shortline_size_then_recheck_execution_gate:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "",
            "environment_classification": "",
            "environment_blocker_detail": "",
            "environment_remediation_hint": "",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "raise_effective_shortline_size_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_sizing_watch"


def test_build_remote_promotion_unblock_readiness_uses_setup_transition_watch_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_gate_stack_progress",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track liquidity sweep before shortline setup promotion",
            "top_blocker_target_artifact": "crypto_shortline_gate_stack_progress",
            "top_blocker_next_action": "wait_for_liquidity_sweep_then_refresh_gate_stack",
            "top_blocker_detail": "fresh_artifact:ticket_row_blocked:SOLUSDT | proxy_price_reference_only",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS; os_time_service is not using NTP as the active source",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert payload["readiness_decision"] == "wait_for_liquidity_sweep_then_review_guarded_canary"
    assert payload["primary_blocker_scope"] == "guardian_ticket_actionability"
    assert payload["primary_local_repair_required"] is False
    assert payload["primary_local_repair_title"] == (
        "Track liquidity sweep before shortline setup promotion"
    )
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_gate_stack_progress"
    assert payload["primary_local_repair_detail"] == (
        "fresh_artifact:ticket_row_blocked:SOLUSDT | proxy_price_reference_only"
    )


def test_build_remote_promotion_unblock_readiness_uses_liquidity_sweep_watch_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_liquidity_sweep_watch",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track liquidity sweep before shortline setup promotion",
            "top_blocker_target_artifact": "crypto_shortline_liquidity_sweep_watch",
            "top_blocker_next_action": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "top_blocker_detail": "execution_state=Bias_Only | liquidity_sweep_missing=true | price_reference_blocked=true",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert payload["readiness_decision"] == "wait_for_liquidity_sweep_then_review_guarded_canary"
    assert payload["primary_local_repair_required"] is False
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_liquidity_sweep_watch"
    assert payload["primary_local_repair_title"] == "Track liquidity sweep before shortline setup promotion"


def test_build_remote_promotion_unblock_readiness_uses_liquidity_event_trigger_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track new liquidity sweep event before shortline setup promotion",
            "top_blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "top_blocker_next_action": "wait_for_liquidity_sweep_event_then_recheck_execution_gate",
            "top_blocker_detail": "liquidity_event_trigger=no_liquidity_sweep_event_observed:SOLUSDT:wait_for_liquidity_sweep_event_then_recheck_execution_gate:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "wait_for_liquidity_sweep_event_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_required"] is False
    assert payload["primary_local_repair_target_artifact"] == (
        "crypto_shortline_liquidity_event_trigger"
    )
    assert payload["primary_local_repair_title"] == (
        "Track new liquidity sweep event before shortline setup promotion"
    )


def test_build_remote_promotion_unblock_readiness_uses_persistent_liquidity_pressure_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track persistent orderflow pressure for liquidity sweep confirmation",
            "top_blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "top_blocker_next_action": "monitor_persistent_orderflow_pressure_for_liquidity_sweep",
            "top_blocker_detail": "liquidity_event_trigger=liquidity_sweep_pressure_persisting:SOLUSDT:monitor_persistent_orderflow_pressure_for_liquidity_sweep:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "",
            "environment_classification": "",
            "environment_blocker_detail": "",
            "environment_remediation_hint": "",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "monitor_persistent_orderflow_pressure_for_liquidity_sweep_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_required"] is False
    assert payload["primary_local_repair_target_artifact"] == (
        "crypto_shortline_liquidity_event_trigger"
    )
    assert payload["primary_local_repair_title"] == (
        "Track persistent orderflow pressure for liquidity sweep confirmation"
    )


def test_build_remote_promotion_unblock_readiness_uses_price_approach_action_for_liquidity_event_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Wait for price to approach the liquidity sweep band while orderflow pressure persists",
            "top_blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "top_blocker_next_action": "wait_for_price_to_approach_liquidity_sweep_band_then_recheck_execution_gate",
            "top_blocker_detail": "liquidity_event_trigger=liquidity_sweep_pressure_persisting_far_from_trigger:SOLUSDT:wait_for_price_to_approach_liquidity_sweep_band_then_recheck_execution_gate:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "",
            "environment_classification": "",
            "environment_blocker_detail": "",
            "environment_remediation_hint": "",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "wait_for_price_to_approach_liquidity_sweep_band_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_target_artifact"] == (
        "crypto_shortline_liquidity_event_trigger"
    )


def test_build_remote_promotion_unblock_readiness_uses_live_bars_refresh_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_live_bars_snapshot",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Refresh live bars snapshot before shortline event detection",
            "top_blocker_target_artifact": "crypto_shortline_live_bars_snapshot",
            "top_blocker_next_action": "refresh_live_bars_snapshot_then_recheck_execution_gate",
            "top_blocker_detail": "live_bars_snapshot=bars_live_snapshot_stale:SOLUSDT:refresh_live_bars_snapshot_before_shortline_event_detection:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "",
            "environment_classification": "",
            "environment_blocker_detail": "",
            "environment_remediation_hint": "",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "refresh_live_bars_snapshot_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_target_artifact"] == (
        "crypto_shortline_live_bars_snapshot"
    )


def test_build_remote_promotion_unblock_readiness_uses_price_reference_watch_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_price_reference_watch",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Build executable price reference before shortline setup promotion",
            "top_blocker_target_artifact": "crypto_shortline_price_reference_watch",
            "top_blocker_next_action": "build_price_template_then_recheck_execution_gate",
            "top_blocker_detail": "price_reference_kind=shortline_missing_price_template | execution_price_ready=false | missing_level_fields=entry_price,stop_price,target_price",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert payload["readiness_decision"] == "build_price_template_then_review_guarded_canary"
    assert payload["primary_local_repair_required"] is False
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_price_reference_watch"
    assert payload["primary_local_repair_title"] == (
        "Build executable price reference before shortline setup promotion"
    )


def test_build_remote_promotion_unblock_readiness_uses_liquidity_event_refresh_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T115530Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260315T115540Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115520Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Refresh shortline execution gate after new liquidity sweep event",
            "top_blocker_target_artifact": "crypto_shortline_liquidity_event_trigger",
            "top_blocker_next_action": "refresh_shortline_execution_gate_after_liquidity_event",
            "top_blocker_detail": "liquidity_event_trigger=new_liquidity_sweep_event_detected:SOLUSDT:refresh_shortline_execution_gate_after_liquidity_event:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T115500Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260315T115510Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260315T115515Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "",
            "environment_classification": "",
            "environment_blocker_detail": "",
            "environment_remediation_hint": "",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T11:56:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "refresh_shortline_execution_gate_after_liquidity_event_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_required"] is False
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_liquidity_event_trigger"
    assert payload["primary_local_repair_title"] == (
        "Refresh shortline execution gate after new liquidity sweep event"
    )


def test_build_remote_promotion_unblock_readiness_uses_profile_location_watch_action_for_ticket_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T062210Z_remote_guarded_canary_promotion_gate.json",
        {
            "promotion_gate_status": "guarded_canary_promotion_blocked_guardian_review",
            "promotion_gate_brief": "guarded_canary_promotion_blocked_guardian_review:SOLUSDT:clear_guardian_review_blockers_before_promotion:portfolio_margin_um",
            "shadow_learning_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
            "promotion_blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "promotion_blocker_code": "guardian_ticket_actionability",
        },
    )
    _write_json(
        review_dir / "20260316T062211Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T062212Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:2_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": "Track final LVN-to-HVN/POC rotation before shortline setup promotion",
            "top_blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "top_blocker_next_action": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "top_blocker_detail": "profile_location_watch=profile_location_lvn_key_level_ready_rotation_approaching:SOLUSDT:monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T062213Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_learning_only_shadow_viable:SOLUSDT:score_49:portfolio_margin_um",
            "quality_score": 49,
            "shadow_learning_score": 65,
        },
    )
    _write_json(
        review_dir / "20260316T062214Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260316T062215Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "",
            "environment_classification": "",
            "environment_blocker_detail": "",
            "environment_remediation_hint": "",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T06:23:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        payload["readiness_decision"]
        == "monitor_final_profile_rotation_into_hvn_poc_then_review_guarded_canary"
    )
    assert payload["primary_local_repair_required"] is False
    assert payload["primary_local_repair_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["primary_local_repair_title"] == (
        "Track final LVN-to-HVN/POC rotation before shortline setup promotion"
    )
