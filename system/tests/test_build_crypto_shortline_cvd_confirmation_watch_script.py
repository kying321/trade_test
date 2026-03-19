from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_cvd_confirmation_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_cvd_confirmation_watch_marks_waiting_after_mss(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T080000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T080001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T080002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["cvd_confirmation", "route_state=watch:deprioritize_flow"],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing cvd_confirmation.",
                    "micro_signals": {
                        "cvd_ready": False,
                        "cvd_long": False,
                        "cvd_short": False,
                        "local_window_ok": True,
                        "cvd_drift_risk": False,
                        "attack_confirmation_ok": True,
                        "context": "continuation",
                        "veto_hint": "",
                        "cvd_locality_status": "local_window_ok",
                        "attack_side": "buyers",
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T080003Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "aligned",
            "queue_stack_brief": "crypto_hot -> crypto_majors",
            "next_focus_batch": "crypto_hot",
            "runtime_queue": [
                {
                    "batch": "crypto_hot",
                    "eligible_symbols": ["SOLUSDT"],
                    "matching_symbols": [
                        {
                            "symbol": "SOLUSDT",
                            "cvd_context_mode": "continuation",
                            "cvd_veto_hint": "",
                            "cvd_locality_status": "local_window_ok",
                            "cvd_attack_side": "buyers",
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T080004Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:SOLUSDT:recheck_signal_quality:portfolio_margin_um",
            "snapshot_decision": "recheck_signal_quality",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:00:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "cvd_confirmation_waiting_after_mss"
    assert payload["watch_decision"] == "wait_for_cvd_confirmation_then_recheck_execution_gate"
    assert payload["mss_missing"] is False
    assert payload["cvd_confirmation_missing"] is True
    assert payload["cvd_missing_codes"] == ["cvd_confirmation"]
    assert payload["cvd_context_mode"] == "continuation"
    assert payload["live_orderflow_snapshot_status"] == "live_orderflow_snapshot_ready"


def test_build_crypto_shortline_cvd_confirmation_watch_marks_mss_as_precondition(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T081000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T081001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T081002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["mss", "cvd_confirmation"],
                    "micro_signals": {
                        "local_window_ok": True,
                        "cvd_drift_risk": False,
                        "attack_confirmation_ok": True,
                    },
                }
            ]
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:10:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "cvd_confirmation_precondition_mss_pending"
    assert payload["watch_decision"] == "wait_for_mss_before_recheck_cvd_confirmation"
    assert payload["mss_missing"] is True
    assert payload["cvd_confirmation_missing"] is True


def test_build_crypto_shortline_cvd_confirmation_watch_marks_value_rotation_profile_precondition(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T082000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T082001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T082002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["mss", "cvd_confirmation"],
                    "micro_signals": {
                        "local_window_ok": True,
                        "cvd_drift_risk": False,
                        "attack_confirmation_ok": True,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T082003Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_status": "profile_location_lvn_key_level_ready_rotation_far",
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_far:SOLUSDT:monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_decision": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:20:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_cvd_precondition_profile_alignment_far"
    assert (
        payload["watch_decision"]
        == "monitor_value_rotation_toward_hvn_poc_then_recheck_cvd_confirmation"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["cvd_stage"] == "profile_alignment_precondition"
    assert payload["profile_watch_status"] == "profile_location_lvn_key_level_ready_rotation_far"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"


def test_build_crypto_shortline_cvd_confirmation_watch_marks_value_rotation_approaching_precondition(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T082500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T082501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T082502Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["mss", "cvd_confirmation"],
                    "micro_signals": {
                        "local_window_ok": True,
                        "cvd_drift_risk": False,
                        "attack_confirmation_ok": True,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T082503Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_status": "profile_location_lvn_key_level_ready_rotation_approaching",
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_approaching:SOLUSDT:monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_decision": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "profile_rotation_target_bin_distance": 2,
            "profile_rotation_target_distance_bps": 219.298246,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:25:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_cvd_precondition_profile_alignment_approaching"
    assert (
        payload["watch_decision"]
        == "monitor_value_rotation_into_final_band_then_recheck_cvd_confirmation"
    )
    assert payload["profile_alignment_band"] == "approaching"


def test_build_crypto_shortline_cvd_confirmation_watch_marks_value_rotation_wait_cvd_confirmation(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T083000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T083001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T083002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["cvd_confirmation"],
                    "micro_signals": {
                        "cvd_ready": False,
                        "cvd_long": False,
                        "cvd_short": False,
                        "local_window_ok": True,
                        "cvd_drift_risk": False,
                        "attack_confirmation_ok": True,
                        "context": "continuation",
                        "veto_hint": "",
                        "cvd_locality_status": "local_window_ok",
                        "attack_side": "buyers",
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T083003Z_crypto_shortline_mss_watch.json",
        {
            "watch_status": "value_rotation_scalp_mss_wait_cvd_confirmation",
            "watch_brief": "value_rotation_scalp_mss_wait_cvd_confirmation:SOLUSDT:wait_for_value_rotation_cvd_confirmation_then_recheck_mss:portfolio_margin_um",
            "watch_decision": "wait_for_value_rotation_cvd_confirmation_then_recheck_mss",
            "pattern_family": "value_rotation_scalp",
            "blocker_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "next_action_target_artifact": "crypto_shortline_cvd_confirmation_watch",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:30:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_wait_cvd_confirmation"
    assert (
        payload["watch_decision"]
        == "wait_for_value_rotation_cvd_confirmation_then_recheck_execution_gate"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["cvd_stage"] == "cvd_confirmation"
    assert payload["mss_watch_pattern_family"] == "value_rotation_scalp"


def test_build_crypto_shortline_cvd_confirmation_watch_defers_to_imbalance_retest_when_pattern_router_says_so(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T084500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T084501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T084502Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["mss", "cvd_confirmation", "fvg_ob_breaker_retest"],
                    "micro_signals": {
                        "local_window_ok": True,
                        "cvd_drift_risk": False,
                        "attack_confirmation_ok": True,
                        "context": "continuation",
                        "veto_hint": "",
                        "cvd_locality_status": "proxy_from_current_snapshot",
                        "attack_side": "buyers",
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T084503Z_crypto_shortline_pattern_router.json",
        {
            "pattern_status": "imbalance_continuation_wait_retest",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260316T084504Z_crypto_shortline_retest_watch.json",
        {
            "watch_status": "imbalance_continuation_wait_retest",
            "watch_brief": "imbalance_continuation_wait_retest:SOLUSDT:wait_for_imbalance_retest_then_recheck_execution_gate:portfolio_margin_um",
            "watch_decision": "wait_for_imbalance_retest_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_retest_watch",
            "next_action_target_artifact": "crypto_shortline_retest_watch",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:45:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "imbalance_continuation_cvd_deferred_until_retest"
    assert payload["watch_decision"] == "wait_for_imbalance_retest_before_recheck_cvd_confirmation"
    assert payload["pattern_family"] == "imbalance_continuation"
    assert payload["cvd_stage"] == "post_retest_cvd"
    assert payload["next_action_target_artifact"] == "crypto_shortline_retest_watch"


def test_build_crypto_shortline_cvd_confirmation_watch_scopes_route_artifacts_to_explicit_symbol(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T085500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260316T085501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T085502Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "promoted",
                    "missing_gates": ["mss", "cvd_confirmation"],
                    "micro_signals": {
                        "local_window_ok": True,
                        "cvd_drift_risk": False,
                        "attack_confirmation_ok": True,
                    },
                    "blocker_detail": "eth_only_blocker",
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T085503Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_status": "imbalance_continuation_wait_retest",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260316T085504Z_crypto_shortline_retest_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_retest",
            "watch_brief": "sol_only_retest:SOLUSDT:recheck:spot",
        },
    )
    _write_json(
        review_dir / "20260316T085505Z_crypto_shortline_profile_location_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_profile",
            "watch_brief": "sol_only_profile:SOLUSDT:recheck:spot",
        },
    )
    _write_json(
        review_dir / "20260316T085506Z_crypto_shortline_mss_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_mss",
            "watch_brief": "sol_only_mss:SOLUSDT:recheck:spot",
            "pattern_family": "value_rotation_scalp",
        },
    )
    _write_json(
        review_dir / "20260316T085507Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "sol_only_orderflow",
            "snapshot_brief": "sol_only_orderflow:SOLUSDT:recheck:spot",
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
            "2026-03-16T08:55:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["watch_status"] == "cvd_confirmation_precondition_mss_pending"
    assert payload["watch_decision"] == "wait_for_mss_before_recheck_cvd_confirmation"
    assert payload["pattern_family"] == "sweep_reversal"
    assert payload["cvd_stage"] == "mss_precondition"
    assert payload["pattern_router_status"] == ""
    assert payload["profile_watch_status"] == ""
    assert payload["mss_watch_status"] == ""
    assert payload["live_orderflow_snapshot_status"] == ""
    assert payload["source_artifacts"]["crypto_shortline_pattern_router"] == ""
    assert payload["source_artifacts"]["crypto_shortline_retest_watch"] == ""
    assert payload["source_artifacts"]["crypto_shortline_profile_location_watch"] == ""
    assert payload["source_artifacts"]["crypto_shortline_mss_watch"] == ""
    assert payload["source_artifacts"]["crypto_shortline_live_orderflow_snapshot"] == ""
    assert "sol_only" not in payload["blocker_detail"]
    assert "eth_only_blocker" in payload["blocker_detail"]
