from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_setup_transition_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_setup_transition_watch_classifies_primary_missing_gate(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": [
                    "liquidity_sweep",
                    "mss",
                    "fvg_ob_breaker_retest",
                    "cvd_confirmation",
                ]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "cvd_confirmation",
                        "route_state=watch:deprioritize_flow",
                    ],
                    "blocker_detail": "SOLUSDT remains Bias_Only.",
                    "done_when": "SOLUSDT completes liquidity_sweep and mss and cvd_confirmation",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_material_change_trigger.json",
        {
            "trigger_brief": "no_material_orderflow_change_since_cross_section_anchor:SOLUSDT:wait_for_material_orderflow_change_before_rerun:deprioritize_flow",
            "trigger_status": "no_material_orderflow_change_since_cross_section_anchor",
            "trigger_decision": "wait_for_material_orderflow_change_before_rerun",
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "ticket_row_reasons": [
                "confidence_below_threshold",
                "proxy_price_reference_only",
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
            "2026-03-15T10:06:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["transition_status"] == (
        "shortline_setup_transition_wait_liquidity_sweep_proxy_price_blocked"
    )
    assert payload["transition_decision"] == (
        "wait_for_liquidity_sweep_then_recheck_execution_gate"
    )
    assert payload["primary_missing_gate"] == "liquidity_sweep"
    assert payload["remaining_trigger_stack"] == [
        "liquidity_sweep",
        "mss",
        "cvd_confirmation",
        "route_state=watch:deprioritize_flow",
    ]
    assert payload["next_action_target_artifact"] == "crypto_shortline_execution_gate"


def test_build_crypto_shortline_setup_transition_watch_drops_proxy_suffix_when_price_reference_ready(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"trigger_stack": ["liquidity_sweep", "mss"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "ticket_row_reasons": ["proxy_price_reference_only"]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_shortline_price_reference_watch.json",
        {
            "watch_status": "price_reference_ready",
            "watch_brief": "price_reference_ready:SOLUSDT:recheck_shortline_execution_gate_after_price_reference_ready:portfolio_margin_um",
            "watch_decision": "recheck_shortline_execution_gate_after_price_reference_ready",
            "price_reference_blocked": False,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-15T10:06:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["transition_status"] == "shortline_setup_transition_wait_liquidity_sweep"
    assert payload["price_reference_blocked"] is False
    assert payload["ticket_row_reasons"] == []


def test_build_crypto_shortline_setup_transition_watch_prefers_cvd_watch_when_gate_matches(
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
        review_dir / "20260316T091010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T091020Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": ["liquidity_sweep", "mss", "cvd_confirmation"]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["cvd_confirmation"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T091030Z_crypto_shortline_cvd_confirmation_watch.json",
        {
            "watch_brief": "cvd_confirmation_waiting_after_mss:SOLUSDT:wait_for_cvd_confirmation_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "cvd_confirmation_waiting_after_mss",
            "watch_decision": "wait_for_cvd_confirmation_then_recheck_execution_gate",
            "blocker_title": "Track CVD confirmation after MSS before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "next_action": "wait_for_cvd_confirmation_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "done_when": "SOLUSDT confirms local-window-valid CVD direction after MSS, then the shortline execution gate refresh confirms the next stage",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:10:40Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_missing_gate"] == "cvd_confirmation"
    assert payload["transition_status"] == "cvd_confirmation_waiting_after_mss"
    assert payload["transition_decision"] == "wait_for_cvd_confirmation_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_cvd_confirmation_watch"


def test_build_crypto_shortline_setup_transition_watch_prefers_retest_watch_when_gate_matches(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T093000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T093010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T093020Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"trigger_stack": ["fvg_ob_breaker_retest"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["fvg_ob_breaker_retest"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T093030Z_crypto_shortline_retest_watch.json",
        {
            "watch_brief": "value_rotation_scalp_wait_retest:SOLUSDT:wait_for_value_rotation_retest_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "value_rotation_scalp_wait_retest",
            "watch_decision": "wait_for_value_rotation_retest_then_recheck_execution_gate",
            "blocker_title": "Track value-rotation retest before shortline scalp promotion",
            "blocker_target_artifact": "crypto_shortline_retest_watch",
            "next_action": "wait_for_value_rotation_retest_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_retest_watch",
            "done_when": "SOLUSDT completes the required FVG/OB/Breaker retest after value rotation confirmation",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:30:40Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_missing_gate"] == "fvg_ob_breaker_retest"
    assert payload["transition_status"] == "value_rotation_scalp_wait_retest"
    assert (
        payload["transition_decision"]
        == "wait_for_value_rotation_retest_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_retest_watch"


def test_build_crypto_shortline_setup_transition_watch_prefers_pattern_aware_mss_watch_when_gate_matches(
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
        review_dir / "20260316T091010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T091020Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"trigger_stack": ["liquidity_sweep", "mss", "cvd_confirmation"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["mss", "cvd_confirmation"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T091025Z_crypto_shortline_mss_watch.json",
        {
            "watch_brief": "value_rotation_scalp_mss_precondition_profile_alignment_far:SOLUSDT:monitor_value_rotation_toward_hvn_poc_then_recheck_mss:portfolio_margin_um",
            "watch_status": "value_rotation_scalp_mss_precondition_profile_alignment_far",
            "watch_decision": "monitor_value_rotation_toward_hvn_poc_then_recheck_mss",
            "blocker_title": "Track value-rotation alignment before MSS confirmation for shortline scalp",
            "blocker_target_artifact": "crypto_shortline_mss_watch",
            "next_action": "monitor_value_rotation_toward_hvn_poc_then_recheck_mss",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT rotates into the target value area so MSS can be reassessed for the value-rotation scalp",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:10:40Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_missing_gate"] == "mss"
    assert payload["transition_status"] == "value_rotation_scalp_mss_precondition_profile_alignment_far"
    assert payload["transition_decision"] == "monitor_value_rotation_toward_hvn_poc_then_recheck_mss"
    assert payload["blocker_target_artifact"] == "crypto_shortline_mss_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["mss_watch_status"] == "value_rotation_scalp_mss_precondition_profile_alignment_far"


def test_build_crypto_shortline_setup_transition_watch_prefers_profile_watch_when_profile_context_still_missing(
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
        review_dir / "20260316T092010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T092020Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"trigger_stack": ["liquidity_sweep", "mss", "fvg_ob_breaker_retest"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": [
                        "fvg_ob_breaker_retest",
                        "profile_location=LVN",
                        "cvd_key_level_context",
                        "mss",
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T092025Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_far:SOLUSDT:monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_far",
            "watch_decision": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "blocker_title": "Track LVN-to-HVN/POC rotation before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT rotates from LVN toward HVN/POC so the shortline execution gate can reassess the next stage",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T09:20:40Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["raw_primary_missing_gate"] == "mss"
    assert payload["primary_missing_gate"] == "profile_location"
    assert payload["transition_status"] == "profile_location_lvn_key_level_ready_rotation_far"
    assert payload["transition_decision"] == "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["profile_watch_status"] == "profile_location_lvn_key_level_ready_rotation_far"


def test_build_crypto_shortline_setup_transition_watch_prefers_pattern_router_retest_override(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T093100Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T093101Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T093102Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": [
                    "4h_profile_location",
                    "liquidity_sweep",
                    "mss",
                    "fvg_ob_breaker_retest",
                ]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "fvg_ob_breaker_retest",
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T093103Z_crypto_shortline_pattern_router.json",
        {
            "pattern_status": "imbalance_continuation_wait_retest",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260316T093104Z_crypto_shortline_retest_watch.json",
        {
            "watch_brief": "imbalance_continuation_wait_retest:SOLUSDT:wait_for_imbalance_retest_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "imbalance_continuation_wait_retest",
            "watch_decision": "wait_for_imbalance_retest_then_recheck_execution_gate",
            "blocker_title": "Track imbalance retest before shortline continuation promotion",
            "blocker_target_artifact": "crypto_shortline_retest_watch",
            "next_action": "wait_for_imbalance_retest_then_recheck_execution_gate",
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
            "2026-03-16T09:31:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["raw_primary_missing_gate"] == "liquidity_sweep"
    assert payload["primary_missing_gate"] == "fvg_ob_breaker_retest"
    assert payload["transition_status"] == "imbalance_continuation_wait_retest"
    assert payload["transition_decision"] == "wait_for_imbalance_retest_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_retest_watch"


def test_build_crypto_shortline_setup_transition_watch_scopes_route_artifacts_to_explicit_symbol(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T094500Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260316T094501Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T094502Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"trigger_stack": ["liquidity_sweep", "mss", "cvd_confirmation"]},
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "promoted",
                    "missing_gates": ["mss", "cvd_confirmation"],
                    "blocker_detail": "eth_only_blocker",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T094503Z_crypto_shortline_material_change_trigger.json",
        {
            "route_symbol": "SOLUSDT",
            "trigger_brief": "sol_only_material_change",
            "trigger_status": "sol_only_material",
            "trigger_decision": "sol_only_action",
        },
    )
    _write_json(
        review_dir / "20260316T094504Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "route_symbol": "SOLUSDT",
            "ticket_row_reasons": ["proxy_price_reference_only"],
        },
    )
    _write_json(
        review_dir / "20260316T094505Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_status": "imbalance_continuation_wait_retest",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260316T094506Z_crypto_shortline_profile_location_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_profile",
            "watch_brief": "sol_only_profile:SOLUSDT:recheck:spot",
        },
    )
    _write_json(
        review_dir / "20260316T094507Z_crypto_shortline_mss_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_mss",
            "watch_brief": "sol_only_mss:SOLUSDT:recheck:spot",
        },
    )
    _write_json(
        review_dir / "20260316T094508Z_crypto_shortline_cvd_confirmation_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_cvd",
            "watch_brief": "sol_only_cvd:SOLUSDT:recheck:spot",
        },
    )
    _write_json(
        review_dir / "20260316T094509Z_crypto_shortline_retest_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_retest",
            "watch_brief": "sol_only_retest:SOLUSDT:recheck:spot",
        },
    )
    _write_json(
        review_dir / "20260316T094510Z_crypto_shortline_price_reference_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "sol_only_price_reference",
            "watch_brief": "sol_only_price_reference:SOLUSDT:recheck:spot",
            "price_reference_blocked": True,
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
            "2026-03-16T09:45:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["primary_missing_gate"] == "mss"
    assert payload["transition_status"] == "shortline_setup_transition_wait_mss"
    assert payload["transition_decision"] == "wait_for_mss_then_recheck_execution_gate"
    assert payload["ticket_row_reasons"] == []
    assert payload["pattern_router_status"] == ""
    assert payload["profile_watch_status"] == ""
    assert payload["mss_watch_status"] == ""
    assert payload["cvd_confirmation_watch_status"] == ""
    assert payload["retest_watch_status"] == ""
    assert payload["price_reference_watch_status"] == ""
    assert payload["artifacts"]["crypto_shortline_material_change_trigger"] == ""
    assert payload["artifacts"]["crypto_shortline_ticket_constraint_diagnosis"] == ""
    assert payload["artifacts"]["crypto_shortline_pattern_router"] == ""
    assert payload["artifacts"]["crypto_shortline_profile_location_watch"] == ""
    assert payload["artifacts"]["crypto_shortline_mss_watch"] == ""
    assert payload["artifacts"]["crypto_shortline_cvd_confirmation_watch"] == ""
    assert payload["artifacts"]["crypto_shortline_retest_watch"] == ""
    assert payload["artifacts"]["crypto_shortline_price_reference_watch"] == ""
    assert "sol_only" not in payload["blocker_detail"]
    assert "eth_only_blocker" in payload["blocker_detail"]
    assert payload["next_action_target_artifact"] == "crypto_shortline_execution_gate"
    assert payload["remaining_trigger_stack"][0] == "mss"


def test_build_crypto_shortline_setup_transition_watch_canonicalizes_alias_trigger_stack(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T123000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T123001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260318T123002Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": [
                    "4h_profile_location",
                    "liquidity_sweep",
                    "1m_5m_mss_or_choch",
                    "15m_cvd_divergence_or_confirmation",
                    "fvg_ob_breaker_retest",
                    "15m_reversal_or_breakout_candle",
                ]
            },
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "promoted",
                    "missing_gates": ["mss", "fvg_ob_breaker_retest", "cvd_confirmation"],
                    "blocker_detail": "eth_alias_stack_blocker",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260318T123003Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "ETHUSDT",
            "pattern_status": "sweep_reversal_wait_mss",
            "pattern_family": "sweep_reversal",
            "pattern_stage": "mss_confirmation",
        },
    )
    _write_json(
        review_dir / "20260318T123004Z_crypto_shortline_mss_watch.json",
        {
            "route_symbol": "ETHUSDT",
            "watch_status": "mss_waiting_after_liquidity_sweep",
            "watch_brief": "mss_waiting_after_liquidity_sweep:ETHUSDT:wait_for_mss_then_recheck_execution_gate:spot",
            "watch_decision": "wait_for_mss_then_recheck_execution_gate",
        },
    )
    _write_json(
        review_dir / "20260318T123005Z_crypto_shortline_retest_watch.json",
        {
            "route_symbol": "ETHUSDT",
            "watch_status": "imbalance_continuation_wait_retest",
            "watch_brief": "imbalance_continuation_wait_retest:ETHUSDT:wait_for_imbalance_retest_then_recheck_execution_gate:spot",
            "watch_decision": "wait_for_imbalance_retest_then_recheck_execution_gate",
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
            "2026-03-18T12:36:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["raw_primary_missing_gate"] == "mss"
    assert payload["primary_missing_gate"] == "mss"
    assert payload["transition_status"] == "mss_waiting_after_liquidity_sweep"
    assert payload["transition_decision"] == "wait_for_mss_then_recheck_execution_gate"
    assert payload["pattern_router_status"] == "sweep_reversal_wait_mss"
    assert payload["mss_watch_status"] == "mss_waiting_after_liquidity_sweep"
    assert payload["retest_watch_status"] == "imbalance_continuation_wait_retest"
    assert payload["remaining_trigger_stack"][:3] == [
        "mss",
        "cvd_confirmation",
        "fvg_ob_breaker_retest",
    ]
