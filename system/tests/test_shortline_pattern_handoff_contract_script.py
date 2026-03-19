from __future__ import annotations

import json
import subprocess
from pathlib import Path


TICKET_ACTIONABILITY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_ticket_actionability_state.py"
)
READINESS_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_promotion_unblock_readiness.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_shortline_pattern_handoff_contract_promotes_retest_far_into_readiness(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")

    _write_json(
        review_dir / "20260317T010000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260317T010001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_no_edge_bias_only_micro_veto",
            "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT:deprioritize_flow",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260317T010002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "pattern_family_hint": "imbalance_continuation",
                    "pattern_stage_hint": "imbalance_retest",
                    "pattern_hint_brief": (
                        "imbalance_continuation:imbalance_retest:"
                        "fvg_ob_breaker_retest,route_state=watch:deprioritize_flow"
                    ),
                    "effective_missing_gates": [
                        "fvg_ob_breaker_retest",
                        "route_state=watch:deprioritize_flow",
                    ],
                    "missing_gates": [
                        "profile_location=MID",
                        "cvd_key_level_context",
                        "liquidity_sweep",
                        "mss",
                        "fvg_ob_breaker_retest",
                        "cvd_confirmation",
                        "route_state=watch:deprioritize_flow",
                    ],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260317T010003Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "degraded",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )
    _write_json(
        review_dir / "20260317T010004Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-17T01:00:04Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260317T010004Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-17",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-17",
                    "allowed": False,
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "size_below_min_notional",
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260317T010005Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-17:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260317T010006Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-17",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260317T010007Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "imbalance_continuation_wait_retest:SOLUSDT:wait_for_imbalance_retest_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_status": "imbalance_continuation_wait_retest",
            "diagnosis_decision": "wait_for_imbalance_retest_then_recheck_execution_gate",
            "primary_constraint_code": "pattern_router:imbalance_continuation:imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260317T010008Z_crypto_shortline_setup_transition_watch.json",
        {
            "transition_brief": "imbalance_continuation_wait_retest_far:SOLUSDT:monitor_imbalance_retest_band_then_recheck_execution_gate:portfolio_margin_um",
            "transition_status": "imbalance_continuation_wait_retest_far",
            "transition_decision": "monitor_imbalance_retest_band_then_recheck_execution_gate",
            "primary_missing_gate": "fvg_ob_breaker_retest",
        },
    )
    _write_json(
        review_dir / "20260317T010009Z_crypto_shortline_gate_stack_progress.json",
        {
            "gate_stack_brief": "shortline_gate_stack_blocked_at_fvg_ob_breaker_retest:SOLUSDT:wait_for_imbalance_retest_then_recheck_execution_gate:portfolio_margin_um",
            "gate_stack_status": "shortline_gate_stack_blocked_at_fvg_ob_breaker_retest",
            "gate_stack_decision": "wait_for_imbalance_retest_then_recheck_execution_gate",
            "primary_stage": "fvg_ob_breaker_retest",
        },
    )
    _write_json(
        review_dir / "20260317T010010Z_crypto_shortline_pattern_router.json",
        {
            "pattern_brief": "imbalance_continuation_wait_retest_far:SOLUSDT:monitor_imbalance_retest_band_then_recheck_execution_gate:portfolio_margin_um:imbalance_continuation",
            "pattern_status": "imbalance_continuation_wait_retest_far",
            "pattern_decision": "monitor_imbalance_retest_band_then_recheck_execution_gate",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
            "blocker_title": "Track imbalance retest band before shortline continuation promotion",
            "blocker_target_artifact": "crypto_shortline_pattern_router",
            "next_action": "monitor_imbalance_retest_band_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT enters the final imbalance retest band and the continuation can be reassessed",
        },
    )

    ticket_proc = subprocess.run(
        [
            "python3",
            str(TICKET_ACTIONABILITY_SCRIPT),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-17T01:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert ticket_proc.returncode == 0, ticket_proc.stderr
    ticket_payload = json.loads(ticket_proc.stdout)
    assert ticket_payload["ticket_actionability_status"] == "imbalance_continuation_wait_retest_far"
    assert (
        ticket_payload["ticket_actionability_decision"]
        == "monitor_imbalance_retest_band_then_recheck_execution_gate"
    )
    assert ticket_payload["blocker_target_artifact"] == "crypto_shortline_pattern_router"
    assert ticket_payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"

    _write_json(
        review_dir / "20260317T010101Z_remote_guardian_blocker_clearance.json",
        {
            "clearance_brief": "guardian_blocker_clearance_blocked:SOLUSDT:4_blocked:portfolio_margin_um",
            "top_blocker_code": "guardian_ticket_actionability",
            "top_blocker_title": ticket_payload["blocker_title"],
            "top_blocker_target_artifact": ticket_payload["blocker_target_artifact"],
            "top_blocker_next_action": ticket_payload["ticket_actionability_decision"],
            "top_blocker_detail": ticket_payload["blocker_detail"],
        },
    )
    _write_json(
        review_dir / "20260317T010102Z_remote_guarded_canary_promotion_gate.json",
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
        review_dir / "20260317T010103Z_remote_shadow_learning_continuity.json",
        {
            "continuity_status": "shadow_learning_continuity_stable",
            "continuity_brief": "shadow_learning_continuity_stable:SOLUSDT:shadow_feedback_alive:portfolio_margin_um",
            "continuity_decision": "continue_shadow_learning_collect_feedback",
            "route_symbol": "SOLUSDT",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260317T010104Z_remote_orderflow_quality_report.json",
        {
            "quality_brief": "quality_degraded_guardian_blocked_shadow_only:SOLUSDT:score_44:portfolio_margin_um",
            "quality_score": 44,
            "shadow_learning_score": 60,
        },
    )
    _write_json(
        review_dir / "20260317T010105Z_system_time_sync_repair_verification_report.json",
        {
            "verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "cleared": True,
        },
    )
    _write_json(
        review_dir / "20260317T010106Z_system_time_sync_repair_plan.json",
        {
            "plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "environment_classification": "timed_apns_fallback_ntp_reachable",
            "environment_blocker_detail": "timed_source=APNS",
            "environment_remediation_hint": "kick timed if needed",
        },
    )

    readiness_proc = subprocess.run(
        [
            "python3",
            str(READINESS_SCRIPT),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-17T01:01:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert readiness_proc.returncode == 0, readiness_proc.stderr
    readiness_payload = json.loads(readiness_proc.stdout)
    assert readiness_payload["readiness_status"] == "shadow_ready_ticket_actionability_blocked"
    assert (
        readiness_payload["readiness_decision"]
        == "monitor_imbalance_retest_band_then_review_guarded_canary"
    )
    assert readiness_payload["primary_local_repair_target_artifact"] == "crypto_shortline_pattern_router"
