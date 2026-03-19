from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_gate_stack_progress.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_gate_stack_progress_marks_primary_stage_and_pending_stack(
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
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "cvd_confirmation",
                        "route_state=watch:deprioritize_flow",
                    ],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, cvd_confirmation.",
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
    assert payload["gate_stack_status"] == "shortline_gate_stack_blocked_at_liquidity_sweep_proxy_price_blocked"
    assert payload["gate_stack_decision"] == "wait_for_liquidity_sweep_then_refresh_gate_stack"
    assert payload["primary_stage"] == "liquidity_sweep"
    assert payload["remaining_blocked_stages"] == [
        "liquidity_sweep",
        "mss",
        "cvd_confirmation",
        "fvg_ob_breaker_retest",
        "reversal_or_breakout_candle",
        "route_state",
    ]
    rows = payload["gate_stack_rows"]
    assert rows[0]["stage_code"] == "profile_location"
    assert rows[0]["status"] == "cleared"
    assert rows[1]["stage_code"] == "liquidity_sweep"
    assert rows[1]["status"] == "blocking"
    assert rows[2]["stage_code"] == "mss"
    assert rows[2]["status"] == "pending"
    assert rows[3]["stage_code"] == "cvd_confirmation"
    assert rows[3]["status"] == "pending"
    assert payload["next_action_target_artifact"] == "crypto_shortline_execution_gate"


def test_build_crypto_shortline_gate_stack_progress_prefers_liquidity_watch_target_when_primary_stage_matches(
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
            "shortline_policy": {
                "trigger_stack": [
                    "liquidity_sweep",
                    "1m_5m_mss_or_choch",
                    "15m_cvd_divergence_or_confirmation",
                ]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "cvd_confirmation",
                    ],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, cvd_confirmation.",
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
            "ticket_row_reasons": ["proxy_price_reference_only"]
        },
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_shortline_liquidity_sweep_watch.json",
        {
            "watch_brief": "liquidity_sweep_waiting_proxy_price_blocked:SOLUSDT:wait_for_liquidity_sweep_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "liquidity_sweep_waiting_proxy_price_blocked",
            "watch_decision": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "blocker_title": "Track liquidity sweep before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_liquidity_sweep_watch",
            "next_action": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_liquidity_sweep_watch",
            "done_when": "SOLUSDT records a liquidity sweep with executable price reference, then the shortline execution gate refresh confirms the next stage",
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
    assert payload["primary_stage"] == "liquidity_sweep"
    assert payload["gate_stack_decision"] == "wait_for_liquidity_sweep_then_recheck_execution_gate"


def test_build_crypto_shortline_gate_stack_progress_prefers_cvd_watch_target_when_primary_stage_matches(
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
        review_dir / "20260316T090010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T090020Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": [
                    "4h_profile_location",
                    "liquidity_sweep",
                    "1m_5m_mss_or_choch",
                    "15m_cvd_divergence_or_confirmation",
                ]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": ["cvd_confirmation", "route_state=watch:deprioritize_flow"],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing cvd_confirmation.",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T090030Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {"ticket_row_reasons": []},
    )
    _write_json(
        review_dir / "20260316T090040Z_crypto_shortline_cvd_confirmation_watch.json",
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
            "2026-03-16T09:00:50Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_stage"] == "cvd_confirmation"
    assert payload["gate_stack_decision"] == "wait_for_cvd_confirmation_then_recheck_execution_gate"


def test_build_crypto_shortline_gate_stack_progress_prefers_retest_watch_target_when_primary_stage_matches(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T094000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T094010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T094020Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": ["fvg_ob_breaker_retest", "route_state"]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": ["fvg_ob_breaker_retest", "route_state=watch:deprioritize_flow"],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing fvg_ob_breaker_retest.",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T094030Z_crypto_shortline_retest_watch.json",
        {
            "watch_brief": "value_rotation_scalp_wait_retest:SOLUSDT:wait_for_value_rotation_retest_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "value_rotation_scalp_wait_retest",
            "watch_decision": "wait_for_value_rotation_retest_then_recheck_execution_gate",
            "blocker_title": "Track value-rotation retest before shortline scalp promotion",
            "blocker_target_artifact": "crypto_shortline_retest_watch",
            "next_action": "wait_for_value_rotation_retest_then_recheck_execution_gate",
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
            "2026-03-16T09:40:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_stage"] == "fvg_ob_breaker_retest"
    assert (
        payload["gate_stack_decision"]
        == "wait_for_value_rotation_retest_then_recheck_execution_gate"
    )
    assert payload["next_action_target_artifact"] == "crypto_shortline_retest_watch"
    assert payload["blocker_target_artifact"] == "crypto_shortline_retest_watch"
    assert payload["retest_watch_status"] == "value_rotation_scalp_wait_retest"


def test_build_crypto_shortline_gate_stack_progress_prefers_mss_watch_target_when_primary_stage_matches(
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
            "shortline_policy": {
                "trigger_stack": [
                    "4h_profile_location",
                    "liquidity_sweep",
                    "1m_5m_mss_or_choch",
                    "fvg_ob_breaker_retest",
                ]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [
                        "mss",
                        "fvg_ob_breaker_retest",
                    ],
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing mss, fvg_ob_breaker_retest.",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T080003Z_crypto_shortline_mss_watch.json",
        {
            "watch_brief": "mss_waiting_after_liquidity_sweep:SOLUSDT:wait_for_mss_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "mss_waiting_after_liquidity_sweep",
            "watch_decision": "wait_for_mss_then_recheck_execution_gate",
            "blocker_title": "Track market-structure shift after liquidity sweep before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_mss_watch",
            "next_action": "wait_for_mss_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_mss_watch",
            "done_when": "SOLUSDT confirms MSS after the active liquidity sweep and refreshes the execution gate",
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
    assert payload["primary_stage"] == "mss"
    assert payload["gate_stack_decision"] == "wait_for_mss_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_mss_watch"
    assert payload["mss_watch_status"] == "mss_waiting_after_liquidity_sweep"
    assert payload["next_action_target_artifact"] == "crypto_shortline_mss_watch"
    assert payload["liquidity_sweep_watch_status"] == ""


def test_build_crypto_shortline_gate_stack_progress_drops_proxy_suffix_when_price_reference_ready(
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
    assert payload["gate_stack_status"] == "shortline_gate_stack_blocked_at_liquidity_sweep"
    assert payload["price_reference_blocked"] is False
    assert payload["ticket_row_reasons"] == []


def test_build_crypto_shortline_gate_stack_progress_prefers_profile_location_watch_target(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T062000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T062001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T062002Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": [
                    "4h_profile_location",
                    "liquidity_sweep",
                    "1m_5m_mss_or_choch",
                ]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": [
                        "profile_location=LVN",
                        "cvd_key_level_context",
                        "mss",
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T062003Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_approaching:SOLUSDT:monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_approaching",
            "watch_decision": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "blocker_title": "Track final LVN-to-HVN/POC rotation before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT rotates into HVN or POC and clears key-level context",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T06:22:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_stage"] == "profile_location"
    assert (
        payload["gate_stack_decision"]
        == "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert (
        payload["profile_location_watch_status"]
        == "profile_location_lvn_key_level_ready_rotation_approaching"
    )


def test_build_crypto_shortline_gate_stack_progress_ignores_future_stamped_profile_watch(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T052000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T052001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T052002Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": [
                    "4h_profile_location",
                    "liquidity_sweep",
                    "1m_5m_mss_or_choch",
                ]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": ["profile_location=LVN", "cvd_key_level_context", "mss"],
                    "blocker_detail": "current_gate",
                }
            ],
        },
    )
    current_profile_path = review_dir / "20260316T052003Z_crypto_shortline_profile_location_watch.json"
    _write_json(
        current_profile_path,
        {
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_far:SOLUSDT:monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_far",
            "watch_decision": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "blocker_title": "Track LVN-to-HVN/POC rotation before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "current_profile",
        },
    )
    future_profile_path = review_dir / "20260316T075738Z_crypto_shortline_profile_location_watch.json"
    _write_json(
        future_profile_path,
        {
            "watch_brief": "profile_location_mid_blocked:SOLUSDT:wait_for_profile_location_alignment_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_mid_blocked",
            "watch_decision": "wait_for_profile_location_alignment_then_recheck_execution_gate",
            "blocker_title": "Future stale profile blocker",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action": "wait_for_profile_location_alignment_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "future_profile",
        },
    )
    os.utime(future_profile_path, None)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T05:22:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["artifacts"]["crypto_shortline_profile_location_watch"] == str(current_profile_path)
    assert payload["blocker_title"] == "Track LVN-to-HVN/POC rotation before shortline setup promotion"
    assert payload["next_action"] == "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate"
    assert "current_profile" in payload["done_when"]


def test_build_crypto_shortline_gate_stack_progress_prefers_pattern_router_retest_override(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T091200Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T091201Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T091202Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {
                "trigger_stack": [
                    "4h_profile_location",
                    "liquidity_sweep",
                    "1m_5m_mss_or_choch",
                    "15m_cvd_divergence_or_confirmation",
                    "fvg_ob_breaker_retest",
                ]
            },
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": [
                        "profile_location=MID",
                        "cvd_key_level_context",
                        "liquidity_sweep",
                        "mss",
                        "fvg_ob_breaker_retest",
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T091203Z_crypto_shortline_pattern_router.json",
        {
            "pattern_status": "imbalance_continuation_wait_retest",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260316T091204Z_crypto_shortline_retest_watch.json",
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
            "2026-03-16T09:12:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["primary_stage"] == "fvg_ob_breaker_retest"
    assert payload["gate_stack_status"] == "shortline_gate_stack_blocked_at_fvg_ob_breaker_retest"
    assert payload["gate_stack_decision"] == "wait_for_imbalance_retest_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_retest_watch"
    assert payload["remaining_blocked_stages"][0] == "fvg_ob_breaker_retest"
    assert payload["gate_stack_rows"][0]["status"] == "cleared"
    assert payload["gate_stack_rows"][4]["status"] == "blocking"


def test_build_crypto_shortline_gate_stack_progress_scopes_route_artifacts_to_explicit_symbol(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T114000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T114001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260318T114002Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"trigger_stack": ["1m_5m_mss_or_choch", "fvg_ob_breaker_retest"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                },
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deploy_price_state_only",
                    "route_state": "promoted",
                    "missing_gates": ["mss", "fvg_ob_breaker_retest", "cvd_confirmation"],
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260318T114003Z_crypto_shortline_mss_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "mss_waiting_for_confirmation",
            "watch_decision": "wait_for_sol_mss_then_refresh_gate_stack",
            "blocker_title": "SOL only blocker",
            "blocker_target_artifact": "crypto_shortline_mss_watch",
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
            "2026-03-18T11:45:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["primary_stage"] == "mss"
    assert payload["gate_stack_decision"] == "wait_for_mss_then_refresh_gate_stack"
    assert payload["mss_watch_status"] == ""
    assert payload["artifacts"]["crypto_shortline_mss_watch"] == ""
    assert "SOL only blocker" not in payload["blocker_detail"]


def test_build_crypto_shortline_gate_stack_progress_emits_symbol_rows_for_all_gate_symbols(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T114100Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T114101Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260318T114102Z_crypto_shortline_execution_gate.json",
        {
            "shortline_policy": {"trigger_stack": ["1m_5m_mss_or_choch", "fvg_ob_breaker_retest"]},
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                },
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deploy_price_state_only",
                    "route_state": "promoted",
                    "missing_gates": ["mss", "route_state=promoted:deploy_price_state_only"],
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260318T114103Z_crypto_shortline_mss_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "mss_waiting_for_confirmation",
            "watch_decision": "wait_for_sol_mss_then_refresh_gate_stack",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-18T11:46:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "SOLUSDT"
    assert payload["symbol_count"] == 2
    symbol_rows = {row["symbol"]: row for row in payload["symbols"]}
    assert sorted(symbol_rows) == ["ETHUSDT", "SOLUSDT"]
    eth = symbol_rows["ETHUSDT"]
    assert eth["primary_stage"] == "mss"
    assert eth["gate_stack_decision"] == "wait_for_mss_then_refresh_gate_stack"
    assert eth["mss_watch_status"] == ""
    assert eth["artifacts"]["crypto_shortline_mss_watch"] == ""
    assert eth["route_state"] == "promoted"
