from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_retest_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_retest_watch_marks_value_rotation_profile_precondition(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T084000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T084001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T084002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["fvg_ob_breaker_retest"],
                    "structure_signals": {"fvg_long": False, "fvg_short": False},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T084003Z_crypto_shortline_profile_location_watch.json",
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
            "2026-03-16T08:40:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_retest_precondition_profile_alignment_far"
    assert (
        payload["watch_decision"]
        == "monitor_value_rotation_toward_hvn_poc_then_recheck_retest"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["retest_stage"] == "profile_alignment_precondition"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"


def test_build_crypto_shortline_retest_watch_marks_value_rotation_approaching_precondition(
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
                    "missing_gates": ["fvg_ob_breaker_retest"],
                    "structure_signals": {"fvg_long": False, "fvg_short": False},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T084503Z_crypto_shortline_profile_location_watch.json",
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
            "2026-03-16T08:45:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_retest_precondition_profile_alignment_approaching"
    assert payload["watch_decision"] == "monitor_value_rotation_into_final_band_then_recheck_retest"
    assert payload["profile_alignment_band"] == "approaching"


def test_build_crypto_shortline_retest_watch_marks_value_rotation_cvd_precondition(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T085000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T085001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T085002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["fvg_ob_breaker_retest"],
                    "structure_signals": {"fvg_long": False, "fvg_short": False},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T085003Z_crypto_shortline_cvd_confirmation_watch.json",
        {
            "watch_status": "value_rotation_scalp_wait_cvd_confirmation",
            "watch_brief": "value_rotation_scalp_wait_cvd_confirmation:SOLUSDT:wait_for_value_rotation_cvd_confirmation_then_recheck_execution_gate:portfolio_margin_um",
            "watch_decision": "wait_for_value_rotation_cvd_confirmation_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "next_action_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "pattern_family": "value_rotation_scalp",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:50:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_retest_precondition_cvd_pending"
    assert (
        payload["watch_decision"]
        == "wait_for_value_rotation_cvd_confirmation_then_recheck_retest"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["retest_stage"] == "cvd_precondition"
    assert payload["next_action_target_artifact"] == "crypto_shortline_cvd_confirmation_watch"


def test_build_crypto_shortline_retest_watch_marks_value_rotation_wait_retest(
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
        review_dir / "20260316T090002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["fvg_ob_breaker_retest"],
                    "structure_signals": {"fvg_long": False, "fvg_short": False},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T090003Z_crypto_shortline_cvd_confirmation_watch.json",
        {
            "watch_status": "cvd_confirmation_cleared",
            "watch_brief": "cvd_confirmation_cleared:SOLUSDT:review_next_shortline_stage:portfolio_margin_um",
            "watch_decision": "review_next_shortline_stage",
            "pattern_family": "value_rotation_scalp",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:00:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_wait_retest"
    assert (
        payload["watch_decision"]
        == "wait_for_value_rotation_retest_then_recheck_execution_gate"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["retest_stage"] == "retest_confirmation"


def test_build_crypto_shortline_retest_watch_quantizes_imbalance_continuation_band(
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
        review_dir / "20260316T090502Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["fvg_ob_breaker_retest"],
                    "structure_signals": {"fvg_long": False, "fvg_short": False},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T090503Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_status": "profile_location_mid_key_level_ready_rotation_far",
            "watch_brief": "profile_location_mid_key_level_ready_rotation_far:SOLUSDT:monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_decision": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "profile_rotation_alignment_band": "far",
            "profile_rotation_target_bin_distance": 5,
            "profile_rotation_target_distance_bps": 584.817936,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:05:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "imbalance_continuation_wait_retest_far"
    assert (
        payload["watch_decision"]
        == "monitor_imbalance_retest_band_then_recheck_execution_gate"
    )
    assert payload["pattern_family"] == "imbalance_continuation"
    assert payload["retest_stage"] == "retest_confirmation"
    assert payload["profile_alignment_band"] == "far"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"


def test_build_crypto_shortline_retest_watch_scopes_route_artifacts_to_explicit_symbol(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T091500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260316T091501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T091502Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "promoted",
                    "missing_gates": ["fvg_ob_breaker_retest"],
                    "structure_signals": {"fvg_long": False, "fvg_short": False},
                    "blocker_detail": "eth_only_blocker",
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T091503Z_crypto_shortline_profile_location_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_profile",
            "watch_brief": "sol_only_profile:SOLUSDT:recheck:spot",
        },
    )
    _write_json(
        review_dir / "20260316T091504Z_crypto_shortline_mss_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_mss",
            "watch_brief": "sol_only_mss:SOLUSDT:recheck:spot",
        },
    )
    _write_json(
        review_dir / "20260316T091505Z_crypto_shortline_cvd_confirmation_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_cvd",
            "watch_brief": "sol_only_cvd:SOLUSDT:recheck:spot",
            "pattern_family": "value_rotation_scalp",
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
            "2026-03-16T09:15:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["watch_status"] == "imbalance_continuation_wait_retest"
    assert payload["watch_decision"] == "wait_for_imbalance_retest_then_recheck_execution_gate"
    assert payload["pattern_family"] == "imbalance_continuation"
    assert payload["retest_stage"] == "retest_confirmation"
    assert payload["profile_watch_status"] == ""
    assert payload["mss_watch_status"] == ""
    assert payload["cvd_confirmation_watch_status"] == ""
    assert payload["source_artifacts"]["crypto_shortline_profile_location_watch"] == ""
    assert payload["source_artifacts"]["crypto_shortline_mss_watch"] == ""
    assert payload["source_artifacts"]["crypto_shortline_cvd_confirmation_watch"] == ""
    assert "sol_only" not in payload["blocker_detail"]
    assert "eth_only_blocker" in payload["blocker_detail"]
