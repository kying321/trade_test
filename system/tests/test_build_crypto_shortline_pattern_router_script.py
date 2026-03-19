from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_pattern_router.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_pattern_router_routes_value_rotation_scalp(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T090000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T090001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T090002Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_far:SOLUSDT:monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_far",
            "watch_decision": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "profile_rotation_alignment_band": "far",
        },
    )
    _write_json(
        review_dir / "20260316T090003Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_brief": "new_liquidity_sweep_event_detected:SOLUSDT:refresh_shortline_execution_gate_after_liquidity_event:portfolio_margin_um",
            "trigger_status": "new_liquidity_sweep_event_detected",
            "trigger_decision": "refresh_shortline_execution_gate_after_liquidity_event",
        },
    )
    _write_json(
        review_dir / "20260316T090004Z_crypto_shortline_signal_quality_watch.json",
        {
            "watch_brief": "signal_quality_confidence_convexity_below_threshold:SOLUSDT:improve_shortline_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "signal_quality_confidence_convexity_below_threshold",
            "watch_decision": "improve_shortline_signal_quality_then_recheck_execution_gate",
        },
    )
    _write_json(
        review_dir / "20260316T090005Z_crypto_shortline_sizing_watch.json",
        {
            "watch_brief": "ticket_size_below_min_notional:SOLUSDT:raise_effective_shortline_size_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "ticket_size_below_min_notional",
            "watch_decision": "raise_effective_shortline_size_then_recheck_execution_gate",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:10:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["pattern_stage"] == "profile_alignment"
    assert payload["pattern_status"] == "value_rotation_scalp_wait_profile_alignment_far"
    assert (
        payload["pattern_decision"]
        == "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_pattern_router"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["pattern_confidence_score"] == 40


def test_build_crypto_shortline_pattern_router_treats_final_band_as_final_alignment(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T090500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T090501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T090502Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_final_band:SOLUSDT:monitor_last_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_final_band",
            "watch_decision": "monitor_last_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "profile_rotation_alignment_band": "final",
        },
    )
    _write_json(
        review_dir / "20260316T090503Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_brief": "new_liquidity_sweep_event_detected:SOLUSDT:refresh_shortline_execution_gate_after_liquidity_event:portfolio_margin_um",
            "trigger_status": "new_liquidity_sweep_event_detected",
            "trigger_decision": "refresh_shortline_execution_gate_after_liquidity_event",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:10:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["pattern_status"] == "value_rotation_scalp_wait_profile_alignment_final"
    assert (
        payload["pattern_decision"]
        == "monitor_final_value_rotation_into_hvn_poc_then_recheck_execution_gate"
    )


def test_build_crypto_shortline_pattern_router_treats_approaching_as_mid_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T090700Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T090701Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T090702Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_approaching:SOLUSDT:monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_approaching",
            "watch_decision": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "profile_rotation_alignment_band": "approaching",
        },
    )
    _write_json(
        review_dir / "20260316T090703Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_brief": "new_liquidity_sweep_event_detected:SOLUSDT:refresh_shortline_execution_gate_after_liquidity_event:portfolio_margin_um",
            "trigger_status": "new_liquidity_sweep_event_detected",
            "trigger_decision": "refresh_shortline_execution_gate_after_liquidity_event",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:10:20Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["pattern_status"] == "value_rotation_scalp_wait_profile_alignment_approaching"
    assert (
        payload["pattern_decision"]
        == "monitor_value_rotation_into_final_band_then_recheck_execution_gate"
    )
    assert payload["profile_rotation_alignment_band"] == "approaching"


def test_build_crypto_shortline_pattern_router_routes_sweep_reversal_wait_mss(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T091000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T091001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T091002Z_crypto_shortline_mss_watch.json",
        {
            "watch_brief": "mss_waiting_after_liquidity_sweep:SOLUSDT:wait_for_mss_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "mss_waiting_after_liquidity_sweep",
            "watch_decision": "wait_for_mss_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_mss_watch",
            "next_action_target_artifact": "crypto_shortline_mss_watch",
        },
    )
    _write_json(
        review_dir / "20260316T091003Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_brief": "liquidity_sweep_event_persisting:SOLUSDT:refresh_shortline_execution_gate_after_liquidity_event:portfolio_margin_um",
            "trigger_status": "liquidity_sweep_event_persisting",
            "trigger_decision": "refresh_shortline_execution_gate_after_liquidity_event",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:10:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["pattern_family"] == "sweep_reversal"
    assert payload["pattern_stage"] == "mss_confirmation"
    assert payload["pattern_status"] == "sweep_reversal_wait_mss"
    assert (
        payload["pattern_decision"]
        == "wait_for_sweep_reversal_mss_then_recheck_execution_gate"
    )
    assert payload["next_action_target_artifact"] == "crypto_shortline_mss_watch"


def test_build_crypto_shortline_pattern_router_routes_value_rotation_wait_cvd_confirmation(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T092000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T092001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T092002Z_crypto_shortline_cvd_confirmation_watch.json",
        {
            "watch_brief": "value_rotation_scalp_wait_cvd_confirmation:SOLUSDT:wait_for_value_rotation_cvd_confirmation_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "value_rotation_scalp_wait_cvd_confirmation",
            "watch_decision": "wait_for_value_rotation_cvd_confirmation_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "next_action_target_artifact": "crypto_shortline_cvd_confirmation_watch",
        },
    )
    _write_json(
        review_dir / "20260316T092003Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_brief": "liquidity_sweep_event_persisting:SOLUSDT:refresh_shortline_execution_gate_after_liquidity_event:portfolio_margin_um",
            "trigger_status": "liquidity_sweep_event_persisting",
            "trigger_decision": "refresh_shortline_execution_gate_after_liquidity_event",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:20:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["pattern_stage"] == "cvd_confirmation"
    assert payload["pattern_status"] == "value_rotation_scalp_wait_cvd_confirmation"
    assert (
        payload["pattern_decision"]
        == "wait_for_value_rotation_cvd_confirmation_then_recheck_execution_gate"
    )
    assert payload["next_action_target_artifact"] == "crypto_shortline_cvd_confirmation_watch"


def test_build_crypto_shortline_pattern_router_scopes_route_artifacts_to_explicit_symbol(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T093500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260316T093501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T093502Z_crypto_shortline_retest_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_brief": "sol_only_retest:SOLUSDT:recheck:spot",
            "watch_status": "sol_only_retest",
            "watch_decision": "sol_only_decision",
            "blocker_target_artifact": "crypto_shortline_retest_watch",
            "next_action_target_artifact": "crypto_shortline_retest_watch",
        },
    )
    _write_json(
        review_dir / "20260316T093503Z_crypto_shortline_setup_transition_watch.json",
        {
            "route_symbol": "ETHUSDT",
            "transition_brief": "shortline_setup_transition_wait_fvg_ob_breaker_retest:ETHUSDT:wait_for_fvg_ob_breaker_retest_then_recheck_execution_gate:spot",
            "transition_status": "shortline_setup_transition_wait_fvg_ob_breaker_retest",
            "transition_decision": "wait_for_fvg_ob_breaker_retest_then_recheck_execution_gate",
            "primary_missing_gate": "fvg_ob_breaker_retest",
            "blocker_target_artifact": "crypto_shortline_setup_transition_watch",
            "next_action_target_artifact": "crypto_shortline_setup_transition_watch",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--symbol",
            "ETHUSDT",
            "--now",
            "2026-03-16T09:35:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["pattern_family"] == "imbalance_continuation"
    assert payload["pattern_stage"] == "imbalance_retest"
    assert payload["pattern_status"] == "imbalance_continuation_wait_retest"
    assert payload["pattern_decision"] == "wait_for_imbalance_retest_then_recheck_execution_gate"
    assert payload["next_action_target_artifact"] == "crypto_shortline_setup_transition_watch"
    assert payload["artifacts"]["crypto_shortline_retest_watch"] == ""
    assert payload["artifacts"]["crypto_shortline_setup_transition_watch"] != ""
    assert "sol_only" not in payload["blocker_detail"]


def test_build_crypto_shortline_pattern_router_routes_value_rotation_wait_retest(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T092500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T092501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T092502Z_crypto_shortline_retest_watch.json",
        {
            "watch_brief": "value_rotation_scalp_wait_retest:SOLUSDT:wait_for_value_rotation_retest_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "value_rotation_scalp_wait_retest",
            "watch_decision": "wait_for_value_rotation_retest_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_retest_watch",
            "next_action_target_artifact": "crypto_shortline_retest_watch",
        },
    )
    _write_json(
        review_dir / "20260316T092503Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_brief": "liquidity_sweep_event_persisting:SOLUSDT:refresh_shortline_execution_gate_after_liquidity_event:portfolio_margin_um",
            "trigger_status": "liquidity_sweep_event_persisting",
            "trigger_decision": "refresh_shortline_execution_gate_after_liquidity_event",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:25:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["pattern_stage"] == "retest_confirmation"
    assert payload["pattern_status"] == "value_rotation_scalp_wait_retest"
    assert (
        payload["pattern_decision"]
        == "wait_for_value_rotation_retest_then_recheck_execution_gate"
    )
    assert payload["next_action_target_artifact"] == "crypto_shortline_retest_watch"


def test_build_crypto_shortline_pattern_router_routes_imbalance_retest_far(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T092600Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T092601Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T092602Z_crypto_shortline_retest_watch.json",
        {
            "watch_brief": "imbalance_continuation_wait_retest_far:SOLUSDT:monitor_imbalance_retest_band_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "imbalance_continuation_wait_retest_far",
            "watch_decision": "monitor_imbalance_retest_band_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_retest_watch",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "profile_alignment_band": "far",
        },
    )
    _write_json(
        review_dir / "20260316T092603Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_brief": "liquidity_sweep_event_persisting:SOLUSDT:refresh_shortline_execution_gate_after_liquidity_event:portfolio_margin_um",
            "trigger_status": "liquidity_sweep_event_persisting",
            "trigger_decision": "refresh_shortline_execution_gate_after_liquidity_event",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:26:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["pattern_family"] == "imbalance_continuation"
    assert payload["pattern_stage"] == "imbalance_retest"
    assert payload["pattern_status"] == "imbalance_continuation_wait_retest_far"
    assert (
        payload["pattern_decision"]
        == "monitor_imbalance_retest_band_then_recheck_execution_gate"
    )
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
