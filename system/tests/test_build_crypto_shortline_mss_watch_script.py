from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_mss_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_mss_watch_marks_waiting_after_liquidity_sweep(
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
                    "missing_gates": ["mss", "fvg_ob_breaker_retest"],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing mss, fvg_ob_breaker_retest.",
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": True,
                        "mss_long": False,
                        "mss_short": False,
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
            "2026-03-16T08:00:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "mss_waiting_after_liquidity_sweep"
    assert payload["watch_decision"] == "wait_for_mss_then_recheck_execution_gate"
    assert payload["active_sweep_side"] == "short"
    assert payload["mss_signal_side"] == ""
    assert payload["mss_long"] is False
    assert payload["mss_short"] is False


def test_build_crypto_shortline_mss_watch_marks_value_rotation_profile_precondition(
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
                    "missing_gates": ["profile_location=LVN", "mss", "cvd_confirmation"],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing profile_location=LVN, mss, cvd_confirmation.",
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": True,
                        "mss_long": False,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T081003Z_crypto_shortline_profile_location_watch.json",
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
            "2026-03-16T08:10:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_mss_precondition_profile_alignment_far"
    assert payload["watch_decision"] == "monitor_value_rotation_toward_hvn_poc_then_recheck_mss"
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["mss_stage"] == "profile_alignment_precondition"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["profile_watch_status"] == "profile_location_lvn_key_level_ready_rotation_far"


def test_build_crypto_shortline_mss_watch_treats_final_band_as_final_alignment(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T081500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T081501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T081502Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["profile_location=LVN", "mss", "cvd_confirmation"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": True,
                        "mss_long": False,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T081503Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_status": "profile_location_lvn_key_level_ready_rotation_final_band",
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_final_band:SOLUSDT:monitor_last_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_decision": "monitor_last_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
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
            "2026-03-16T08:15:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_mss_precondition_profile_alignment_final"
    assert payload["watch_decision"] == "monitor_final_value_rotation_into_hvn_poc_then_recheck_mss"


def test_build_crypto_shortline_mss_watch_marks_approaching_profile_alignment_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T081700Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T081701Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T081702Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["profile_location=LVN", "mss", "cvd_confirmation"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": True,
                        "mss_long": False,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T081703Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_status": "profile_location_lvn_key_level_ready_rotation_approaching",
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_approaching:SOLUSDT:monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_decision": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "profile_rotation_alignment_band": "approaching",
            "profile_rotation_next_milestone": "into_final_band",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:17:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_mss_precondition_profile_alignment_approaching"
    assert payload["watch_decision"] == "monitor_value_rotation_into_final_band_then_recheck_mss"
    assert payload["profile_alignment_band"] == "approaching"
    assert payload["profile_rotation_next_milestone"] == "into_final_band"


def test_build_crypto_shortline_mss_watch_defers_to_imbalance_retest_when_pattern_router_says_so(
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
                    "missing_gates": ["mss", "fvg_ob_breaker_retest"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T084003Z_crypto_shortline_pattern_router.json",
        {
            "pattern_status": "imbalance_continuation_wait_retest",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260316T084004Z_crypto_shortline_retest_watch.json",
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
            "2026-03-16T08:40:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "imbalance_continuation_mss_deferred_until_retest"
    assert payload["watch_decision"] == "wait_for_imbalance_retest_before_recheck_mss"
    assert payload["pattern_family"] == "imbalance_continuation"
    assert payload["mss_stage"] == "post_retest_mss"
    assert payload["next_action_target_artifact"] == "crypto_shortline_retest_watch"


def test_build_crypto_shortline_mss_watch_scopes_route_artifacts_to_explicit_symbol(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T121000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T121001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260318T121002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": True,
                        "mss_long": False,
                        "mss_short": False,
                    },
                },
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "promoted",
                    "missing_gates": ["mss", "fvg_ob_breaker_retest", "cvd_confirmation"],
                    "blocker_detail": "ETHUSDT remains Bias_Only; missing mss, fvg_ob_breaker_retest, cvd_confirmation.",
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                    },
                },
            ]
        },
    )
    _write_json(
        review_dir / "20260318T121003Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_status": "imbalance_continuation_wait_retest",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260318T121004Z_crypto_shortline_retest_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_brief": "sol_only_retest_watch",
            "watch_status": "imbalance_continuation_wait_retest",
            "watch_decision": "wait_for_sol_retest",
        },
    )
    _write_json(
        review_dir / "20260318T121005Z_crypto_shortline_liquidity_sweep_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_brief": "sol_only_liquidity_watch",
            "watch_status": "liquidity_sweep_waiting_proxy_price_blocked",
            "watch_decision": "wait_for_sol_liquidity",
        },
    )
    _write_json(
        review_dir / "20260318T121006Z_crypto_shortline_profile_location_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_far",
            "watch_brief": "sol_only_profile_watch",
            "watch_decision": "watch_sol_profile",
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
            "2026-03-18T12:10:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["watch_status"] == "mss_precondition_liquidity_context_pending"
    assert payload["pattern_family"] == "pattern_watch_only"
    assert payload["mss_stage"] == "liquidity_precondition"
    assert payload["liquidity_context_brief"] == ""
    assert payload["pattern_router_status"] == ""
    assert payload["source_artifacts"]["crypto_shortline_pattern_router"] == ""
    assert payload["source_artifacts"]["crypto_shortline_retest_watch"] == ""
    assert payload["source_artifacts"]["crypto_shortline_profile_location_watch"] == ""
    assert payload["source_artifacts"]["crypto_shortline_liquidity_sweep_watch"] == ""
    assert "sol_only" not in payload["blocker_detail"]
