from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_remote_ticket_actionability_state.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_remote_ticket_actionability_state_marks_crypto_bias_only_surface_as_not_ticketed(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T101055Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_stale:SOLUSDT:2026-02-05:age_days=38:recent5_review",
            "freshness_status": "route_signal_row_stale",
            "freshness_decision": "refresh_crypto_signal_source_then_rebuild_tickets",
            "refresh_recommended": True,
        },
    )
    _write_json(
        review_dir / "20260315T101056Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [
                {"symbol": "XAUUSD", "allowed": True, "reasons": []},
                {"symbol": "XAGUSD", "allowed": True, "reasons": []},
                {"symbol": "COPPER", "allowed": False, "reasons": ["spread_too_wide"]},
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_no_edge_bias_only_micro_veto",
            "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT:deprioritize_flow",
            "focus_review_blocker_detail": (
                "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, "
                "fvg_ob_breaker_retest, cvd_confirmation."
            ),
        },
    )
    _write_json(
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "fvg_ob_breaker_retest",
                        "cvd_confirmation",
                    ],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
            "queue_stack_brief": "crypto_hot -> crypto_majors",
            "runtime_queue": [
                {
                    "batch": "crypto_hot",
                    "eligible_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
                    "matching_symbols": [
                        {
                            "symbol": "SOLUSDT",
                            "classification": "Bias_Only",
                            "cvd_context_mode": "watch_only",
                            "cvd_veto_hint": "low_sample_or_gap_risk",
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T101100Z_crypto_shortline_backtest_slice.json",
        {
            "slice_brief": "selected_watch_only_orderflow_slice:SOLUSDT:crypto_hot:4h->15m",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:11:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "crypto_shortline_bias_only_not_ticketed"
    assert (
        payload["ticket_actionability_decision"]
        == "run_shortline_slice_backtest_and_wait_for_setup_ready"
    )
    assert payload["actionable_ready"] is False
    assert payload["ticket_surface_coverage_status"] == "commodity_only"
    assert payload["next_action_target_artifact"] == "crypto_shortline_backtest_slice"
    assert payload["backtest_slice_recommended"] is True
    assert payload["blocker_target_artifact"] == "remote_ticket_actionability_state"
    assert "fresh_artifact:ticket_row_missing:SOLUSDT" in payload["blocker_detail"]
    assert "backtest_slice=selected_watch_only_orderflow_slice:SOLUSDT:crypto_hot:4h->15m" in payload[
        "blocker_detail"
    ]


def test_build_remote_ticket_actionability_state_uses_cross_section_backtest_when_available(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T101055Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_stale:SOLUSDT:2026-02-05:age_days=38:recent5_review",
            "freshness_status": "route_signal_row_stale",
            "freshness_decision": "refresh_crypto_signal_source_then_rebuild_tickets",
            "refresh_recommended": True,
        },
    )
    _write_json(
        review_dir / "20260315T101056Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [
                {"symbol": "XAUUSD", "allowed": True, "reasons": []},
                {"symbol": "XAGUSD", "allowed": True, "reasons": []},
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
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
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )
    _write_json(
        review_dir / "20260315T101100Z_crypto_shortline_backtest_slice.json",
        {
            "slice_brief": "selected_watch_only_orderflow_slice:SOLUSDT:crypto_hot:4h->15m",
        },
    )
    _write_json(
        review_dir / "20260315T101110Z_crypto_shortline_cross_section_backtest.json",
        {
            "backtest_brief": "watch_only_cross_section_positive:SOLUSDT:positive:h60m",
            "backtest_status": "watch_only_cross_section_positive",
            "research_decision": "continue_shadow_learning_wait_for_setup_ready",
            "selected_edge_status": "positive",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:11:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "crypto_shortline_backtest_positive_not_ticketed"
    assert payload["ticket_actionability_decision"] == "continue_shadow_learning_wait_for_setup_ready"
    assert payload["next_action_target_artifact"] == "crypto_shortline_execution_gate"
    assert payload["cross_section_backtest_status"] == "watch_only_cross_section_positive"
    assert payload["cross_section_selected_edge_status"] == "positive"
    assert "cross_section_backtest=watch_only_cross_section_positive:SOLUSDT:positive:h60m" in payload[
        "blocker_detail"
    ]


def test_build_remote_ticket_actionability_state_marks_no_edge_as_gate_cold(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T101055Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_stale:SOLUSDT:2026-02-05:age_days=38:recent5_review",
            "freshness_status": "route_signal_row_stale",
            "freshness_decision": "refresh_crypto_signal_source_then_rebuild_tickets",
            "refresh_recommended": True,
        },
    )
    _write_json(
        review_dir / "20260315T101056Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [
                {"symbol": "XAUUSD", "allowed": True, "reasons": []},
                {"symbol": "XAGUSD", "allowed": True, "reasons": []},
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
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
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": [
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
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )
    _write_json(
        review_dir / "20260315T101100Z_crypto_shortline_backtest_slice.json",
        {
            "slice_brief": "selected_watch_only_orderflow_slice:SOLUSDT:crypto_hot:4h->15m",
        },
    )
    _write_json(
        review_dir / "20260315T101110Z_crypto_shortline_cross_section_backtest.json",
        {
            "backtest_brief": "watch_only_cross_section_no_edge:SOLUSDT:no_edge:h60m",
            "backtest_status": "watch_only_cross_section_no_edge",
            "research_decision": "deprioritize_until_orderflow_improves",
            "selected_edge_status": "no_edge",
        },
    )
    _write_json(
        review_dir / "20260315T101120Z_crypto_shortline_material_change_trigger.json",
        {
            "trigger_brief": "no_material_orderflow_change_since_cross_section_anchor:SOLUSDT:wait_for_material_orderflow_change_before_rerun:deprioritize_flow",
            "trigger_status": "no_material_orderflow_change_since_cross_section_anchor",
            "trigger_decision": "wait_for_material_orderflow_change_before_rerun",
            "rerun_recommended": False,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:11:40Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "crypto_shortline_gate_cold_no_edge_not_ticketed"
    assert (
        payload["ticket_actionability_decision"]
        == "wait_for_material_orderflow_change_then_refresh_execution_gate"
    )
    assert payload["next_action_target_artifact"] == "crypto_shortline_material_change_trigger"
    assert payload["cross_section_backtest_status"] == "watch_only_cross_section_no_edge"
    assert payload["material_change_trigger_status"] == "no_material_orderflow_change_since_cross_section_anchor"
    assert payload["material_change_trigger_rerun_recommended"] is False
    assert "shortline=Bias_Only; missing_gates=liquidity_sweep,mss,fvg_ob_breaker_retest,cvd_confirmation,route_state=watch:deprioritize_flow" in payload[
        "blocker_detail"
    ]
    assert "material_change_trigger=no_material_orderflow_change_since_cross_section_anchor:SOLUSDT:wait_for_material_orderflow_change_before_rerun:deprioritize_flow:wait_for_material_orderflow_change_before_rerun" in payload[
        "blocker_detail"
    ]


def test_build_remote_ticket_actionability_state_reruns_when_material_change_is_detected(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T101055Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_stale:SOLUSDT:2026-02-05:age_days=38:recent5_review",
            "freshness_status": "route_signal_row_stale",
            "freshness_decision": "refresh_crypto_signal_source_then_rebuild_tickets",
            "refresh_recommended": True,
        },
    )
    _write_json(
        review_dir / "20260315T101056Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [{"symbol": "XAUUSD", "allowed": True, "reasons": []}],
        },
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
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
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )
    _write_json(
        review_dir / "20260315T101100Z_crypto_shortline_backtest_slice.json",
        {
            "slice_brief": "selected_watch_only_orderflow_slice:SOLUSDT:crypto_hot:4h->15m",
        },
    )
    _write_json(
        review_dir / "20260315T101110Z_crypto_shortline_cross_section_backtest.json",
        {
            "backtest_brief": "watch_only_cross_section_no_edge:SOLUSDT:no_edge:h60m",
            "backtest_status": "watch_only_cross_section_no_edge",
            "research_decision": "deprioritize_until_orderflow_improves",
            "selected_edge_status": "no_edge",
        },
    )
    _write_json(
        review_dir / "20260315T101120Z_crypto_shortline_material_change_trigger.json",
        {
            "trigger_brief": "material_orderflow_change_detected:SOLUSDT:rerun_shortline_execution_gate_and_recheck_ticket_actionability:deprioritize_flow",
            "trigger_status": "material_orderflow_change_detected",
            "trigger_decision": "rerun_shortline_execution_gate_and_recheck_ticket_actionability",
            "rerun_recommended": True,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:11:50Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "crypto_shortline_material_change_pending_not_ticketed"
    assert payload["ticket_actionability_decision"] == "rerun_shortline_execution_gate_and_recheck_ticket_actionability"
    assert payload["next_action_target_artifact"] == "crypto_shortline_execution_gate"
    assert payload["material_change_trigger_rerun_recommended"] is True


def test_build_remote_ticket_actionability_state_prefers_execution_gate_pattern_hint_in_blocker_detail(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T101056Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "tickets": [
                {
                    "symbol": "SOLUSDT",
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
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
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
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": [
                        "profile_location=MID",
                        "cvd_key_level_context",
                        "liquidity_sweep",
                        "mss",
                        "fvg_ob_breaker_retest",
                        "cvd_confirmation",
                        "route_state=watch:deprioritize_flow",
                    ],
                    "effective_missing_gates": [
                        "fvg_ob_breaker_retest",
                        "cvd_confirmation",
                        "route_state=watch:deprioritize_flow",
                    ],
                    "pattern_family_hint": "imbalance_continuation",
                    "pattern_stage_hint": "imbalance_retest",
                    "pattern_hint_brief": (
                        "imbalance_continuation:imbalance_retest:"
                        "fvg_ob_breaker_retest,cvd_confirmation,route_state=watch:deprioritize_flow"
                    ),
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:11:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "shortline=Bias_Only" in payload["blocker_detail"]
    assert (
        "pattern_hint=imbalance_continuation:imbalance_retest:"
        "fvg_ob_breaker_retest,cvd_confirmation,route_state=watch:deprioritize_flow"
    ) in payload["blocker_detail"]
    assert (
        "effective_missing_gates=fvg_ob_breaker_retest,cvd_confirmation,route_state=watch:deprioritize_flow"
    ) in payload["blocker_detail"]
    assert (
        "raw_missing_gates=profile_location=MID,cvd_key_level_context,liquidity_sweep,mss,"
        "fvg_ob_breaker_retest,cvd_confirmation,route_state=watch:deprioritize_flow"
    ) in payload["blocker_detail"]


def test_build_remote_ticket_actionability_state_marks_stale_crypto_ticket_row_as_source_refresh_required(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T101055Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_stale:SOLUSDT:2026-02-05:age_days=38:recent5_review",
            "freshness_status": "route_signal_row_stale",
            "freshness_decision": "refresh_crypto_signal_source_then_rebuild_tickets",
            "refresh_recommended": True,
        },
    )
    _write_json(
        review_dir / "20260315T101055Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "no_newer_crypto_signal_candidate_available:SOLUSDT:2026-03-06:route_signal_date=2026-02-05",
            "readiness_status": "no_newer_crypto_signal_candidate_available",
            "readiness_decision": "generate_fresh_crypto_signal_source_before_rebuild_tickets",
            "refresh_needed": True,
        },
    )
    _write_json(
        review_dir / "20260315T101056Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "signal_source": {
                "kind": "recent5_review",
                "path": "/tmp/20260306_strategy_recent5_signals.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-06",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": [
                        "stale_signal",
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "size_below_min_notional",
                    ],
                },
                {"symbol": "BTCUSDT", "allowed": False, "reasons": ["signal_not_found"]},
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
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
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:11:50Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "ticket_row_blocked_stale_signal"
    assert (
        payload["ticket_actionability_decision"]
        == "generate_fresh_crypto_signal_source_before_rebuild_tickets"
    )
    assert payload["next_action"] == "generate_fresh_crypto_signal_source_before_rebuild_tickets"
    assert payload["next_action_target_artifact"] == "crypto_signal_source_refresh_readiness"
    assert payload["blocker_target_artifact"] == "crypto_signal_source_refresh_readiness"
    assert payload["blocker_title"] == "Generate fresh crypto signal source before guarded canary review"
    assert payload["ticket_signal_source_kind"] == "recent5_review"
    assert payload["ticket_signal_source_artifact_date"] == "2026-03-06"
    assert payload["ticket_signal_source_age_days"] == 9
    assert payload["signal_source_freshness_status"] == "route_signal_row_stale"
    assert payload["signal_source_refresh_readiness_status"] == (
        "no_newer_crypto_signal_candidate_available"
    )
    assert (
        "ticket_signal_source=recent5_review:20260306_strategy_recent5_signals.json:2026-03-06:age_days=9:matched_symbol_rows"
        in payload["blocker_detail"]
    )
    assert (
        "signal_source_freshness=route_signal_row_stale:SOLUSDT:2026-02-05:age_days=38:recent5_review:refresh_crypto_signal_source_then_rebuild_tickets"
        in payload["blocker_detail"]
    )
    assert (
        "signal_source_refresh_readiness=no_newer_crypto_signal_candidate_available:SOLUSDT:2026-03-06:route_signal_date=2026-02-05:generate_fresh_crypto_signal_source_before_rebuild_tickets"
        in payload["blocker_detail"]
    )


def test_build_remote_ticket_actionability_state_prefers_route_symbol_ticket_surface_over_newer_commodity_surface(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T101055Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:30Z",
            "signal_source": {
                "kind": "recent5_review",
                "path": "/tmp/20260306_strategy_recent5_signals.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-06",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": [
                        "stale_signal",
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "size_below_min_notional",
                    ],
                },
                {"symbol": "BTCUSDT", "allowed": False, "reasons": ["signal_not_found"]},
            ],
        },
    )
    _write_json(
        review_dir / "20260315T101056Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:10:40Z",
            "signal_source": {
                "kind": "explicit",
                "path": "/tmp/20260315_commodity_directional_signals.json",
                "selection_reason": "explicit",
                "artifact_date": "2026-03-15",
            },
            "tickets": [
                {"symbol": "XAUUSD", "allowed": True, "reasons": []},
                {"symbol": "XAGUSD", "allowed": False, "reasons": ["spread_too_wide"]},
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
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
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:11:50Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_artifact"].endswith("20260315T101055Z_signal_to_order_tickets.json")
    assert payload["ticket_surface_coverage_status"] == "route_symbol_present"
    assert payload["ticket_actionability_status"] == "ticket_row_blocked_stale_signal"
    assert payload["next_action_target_artifact"] == "crypto_signal_source_freshness"
    assert payload["blocker_title"] == "Refresh crypto signal source before guarded canary review"


def test_build_remote_ticket_actionability_state_uses_ticket_constraint_diagnosis_for_non_stale_blocked_row(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:00:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260315T142000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-15",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "allowed": False,
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "proxy_price_reference_only",
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_no_edge_bias_only_micro_veto",
            "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "missing_gates": ["liquidity_sweep", "mss"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "degraded"},
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-15:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260315T100060Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-15",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260315T100070Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_route_not_setup_ready_proxy_price_blocked:SOLUSDT:wait_for_setup_ready_and_executable_price_reference:portfolio_margin_um",
            "diagnosis_status": "shortline_route_not_setup_ready_proxy_price_blocked",
            "diagnosis_decision": "wait_for_setup_ready_and_executable_price_reference",
            "primary_constraint_code": "route_not_setup_ready",
            "blocker_title": "Wait for setup-ready route and executable price reference before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_ticket_constraint_diagnosis",
            "next_action": "wait_for_setup_ready_and_executable_price_reference",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists",
        },
    )
    _write_json(
        review_dir / "20260315T100080Z_crypto_shortline_setup_transition_watch.json",
        {
            "transition_brief": "shortline_setup_transition_wait_liquidity_sweep_proxy_price_blocked:SOLUSDT:wait_for_liquidity_sweep_then_recheck_execution_gate:portfolio_margin_um",
            "transition_status": "shortline_setup_transition_wait_liquidity_sweep_proxy_price_blocked",
            "transition_decision": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "primary_missing_gate": "liquidity_sweep",
            "blocker_title": "Wait for liquidity sweep before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_setup_transition_watch",
            "next_action": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists",
        },
    )
    _write_json(
        review_dir / "20260315T100070Z_crypto_shortline_gate_stack_progress.json",
        {
            "gate_stack_brief": "shortline_gate_stack_blocked_at_liquidity_sweep_proxy_price_blocked:SOLUSDT:wait_for_liquidity_sweep_then_refresh_gate_stack:portfolio_margin_um",
            "gate_stack_status": "shortline_gate_stack_blocked_at_liquidity_sweep_proxy_price_blocked",
            "gate_stack_decision": "wait_for_liquidity_sweep_then_refresh_gate_stack",
            "primary_stage": "liquidity_sweep",
            "blocker_title": "Track liquidity sweep before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_gate_stack_progress",
            "next_action": "wait_for_liquidity_sweep_then_refresh_gate_stack",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT clears the remaining shortline trigger stack and keeps an executable price reference",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:07:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == (
        "shortline_gate_stack_blocked_at_liquidity_sweep_proxy_price_blocked"
    )
    assert payload["ticket_actionability_decision"] == (
        "wait_for_liquidity_sweep_then_refresh_gate_stack"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_gate_stack_progress"
    assert payload["next_action_target_artifact"] == "crypto_shortline_execution_gate"
    assert payload["ticket_constraint_primary_code"] == "route_not_setup_ready"
    assert payload["setup_transition_watch_primary_missing_gate"] == "liquidity_sweep"
    assert payload["gate_stack_progress_primary_stage"] == "liquidity_sweep"
    assert payload["gate_stack_progress_artifact"].endswith(
        "_crypto_shortline_gate_stack_progress.json"
    )


def test_build_remote_ticket_actionability_state_prefers_liquidity_sweep_watch_for_first_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:00:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260315T142000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-15",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "allowed": False,
                    "reasons": ["confidence_below_threshold", "proxy_price_reference_only"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_no_edge_bias_only_micro_veto",
            "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "degraded"},
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-15:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260315T100060Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-15",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260315T100070Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_route_not_setup_ready_proxy_price_blocked:SOLUSDT:wait_for_setup_ready_and_executable_price_reference:portfolio_margin_um",
            "diagnosis_status": "shortline_route_not_setup_ready_proxy_price_blocked",
            "diagnosis_decision": "wait_for_setup_ready_and_executable_price_reference",
            "primary_constraint_code": "route_not_setup_ready",
            "blocker_title": "Wait for setup-ready route and executable price reference before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_ticket_constraint_diagnosis",
            "next_action": "wait_for_setup_ready_and_executable_price_reference",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists",
        },
    )
    _write_json(
        review_dir / "20260315T100080Z_crypto_shortline_setup_transition_watch.json",
        {
            "transition_brief": "shortline_setup_transition_wait_liquidity_sweep_proxy_price_blocked:SOLUSDT:wait_for_liquidity_sweep_then_recheck_execution_gate:portfolio_margin_um",
            "transition_status": "shortline_setup_transition_wait_liquidity_sweep_proxy_price_blocked",
            "transition_decision": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "primary_missing_gate": "liquidity_sweep",
            "blocker_title": "Wait for liquidity sweep before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_setup_transition_watch",
            "next_action": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT reaches Setup_Ready with executable price reference and a fresh allowed ticket row exists",
        },
    )
    _write_json(
        review_dir / "20260315T100090Z_crypto_shortline_gate_stack_progress.json",
        {
            "gate_stack_brief": "shortline_gate_stack_blocked_at_liquidity_sweep_proxy_price_blocked:SOLUSDT:wait_for_liquidity_sweep_then_recheck_execution_gate:portfolio_margin_um",
            "gate_stack_status": "shortline_gate_stack_blocked_at_liquidity_sweep_proxy_price_blocked",
            "gate_stack_decision": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "primary_stage": "liquidity_sweep",
            "blocker_title": "Track liquidity sweep before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_liquidity_sweep_watch",
            "next_action": "wait_for_liquidity_sweep_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_liquidity_sweep_watch",
            "done_when": "SOLUSDT records a liquidity sweep with executable price reference, then the shortline execution gate refresh confirms the next stage",
        },
    )
    _write_json(
        review_dir / "20260315T100100Z_crypto_shortline_liquidity_sweep_watch.json",
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
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:07:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "liquidity_sweep_waiting_proxy_price_blocked"
    assert payload["ticket_actionability_decision"] == "wait_for_liquidity_sweep_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_liquidity_sweep_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_liquidity_sweep_watch"


def test_build_remote_ticket_actionability_state_prefers_cvd_confirmation_watch_for_cvd_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260316T092000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T092010Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-16T09:20:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260316T092000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-16",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-16",
                    "allowed": False,
                    "reasons": ["cvd_confirmation"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T092020Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T092030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": ["cvd_confirmation"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T092040Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "aligned"},
    )
    _write_json(
        review_dir / "20260316T092050Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-16:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260316T092060Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-16",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260316T092070Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_route_not_setup_ready:SOLUSDT:wait_for_cvd_confirmation_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_status": "shortline_route_not_setup_ready",
            "diagnosis_decision": "wait_for_cvd_confirmation_then_recheck_execution_gate",
            "primary_constraint_code": "route_not_setup_ready",
            "blocker_title": "Track CVD confirmation before guarded canary review",
        },
    )
    _write_json(
        review_dir / "20260316T092080Z_crypto_shortline_setup_transition_watch.json",
        {
            "transition_brief": "cvd_confirmation_waiting_after_mss:SOLUSDT:wait_for_cvd_confirmation_then_recheck_execution_gate:portfolio_margin_um",
            "transition_status": "cvd_confirmation_waiting_after_mss",
            "transition_decision": "wait_for_cvd_confirmation_then_recheck_execution_gate",
            "primary_missing_gate": "cvd_confirmation",
            "blocker_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "next_action_target_artifact": "crypto_shortline_cvd_confirmation_watch",
        },
    )
    _write_json(
        review_dir / "20260316T092090Z_crypto_shortline_gate_stack_progress.json",
        {
            "gate_stack_brief": "shortline_gate_stack_blocked_at_cvd_confirmation:SOLUSDT:wait_for_cvd_confirmation_then_recheck_execution_gate:portfolio_margin_um",
            "gate_stack_status": "shortline_gate_stack_blocked_at_cvd_confirmation",
            "gate_stack_decision": "wait_for_cvd_confirmation_then_recheck_execution_gate",
            "primary_stage": "cvd_confirmation",
            "blocker_title": "Track CVD confirmation before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_cvd_confirmation_watch",
            "next_action": "wait_for_cvd_confirmation_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_cvd_confirmation_watch",
        },
    )
    _write_json(
        review_dir / "20260316T092100Z_crypto_shortline_cvd_confirmation_watch.json",
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
            "--config",
            str(config_path),
            "--now",
            "2026-03-16T09:20:30Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "cvd_confirmation_waiting_after_mss"
    assert payload["ticket_actionability_decision"] == "wait_for_cvd_confirmation_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_cvd_confirmation_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_cvd_confirmation_watch"
    assert payload["cvd_confirmation_watch_status"] == "cvd_confirmation_waiting_after_mss"


def test_build_remote_ticket_actionability_state_prefers_profile_location_watch_for_first_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260316T062000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T062001Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-16T06:20:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260316T062000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-16",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-16",
                    "allowed": False,
                    "reasons": ["confidence_below_threshold"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T062002Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_bias_only",
            "focus_review_brief": "review_bias_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260316T062003Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [
                        "profile_location=LVN",
                        "cvd_key_level_context",
                        "mss",
                    ],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T062004Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "degraded"},
    )
    _write_json(
        review_dir / "20260316T062005Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-16:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260316T062006Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-16",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260316T062007Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_route_not_setup_ready:SOLUSDT:wait_for_setup_ready:portfolio_margin_um",
            "diagnosis_status": "shortline_route_not_setup_ready",
            "diagnosis_decision": "wait_for_setup_ready",
            "primary_constraint_code": "route_not_setup_ready",
        },
    )
    _write_json(
        review_dir / "20260316T062008Z_crypto_shortline_setup_transition_watch.json",
        {
            "transition_brief": "shortline_setup_transition_wait_fvg_ob_breaker_retest:SOLUSDT:wait_for_fvg_ob_breaker_retest_then_recheck_execution_gate:portfolio_margin_um",
            "transition_status": "shortline_setup_transition_wait_fvg_ob_breaker_retest",
            "transition_decision": "wait_for_fvg_ob_breaker_retest_then_recheck_execution_gate",
            "primary_missing_gate": "fvg_ob_breaker_retest",
        },
    )
    _write_json(
        review_dir / "20260316T062009Z_crypto_shortline_gate_stack_progress.json",
        {
            "gate_stack_brief": "shortline_gate_stack_blocked_at_profile_location:SOLUSDT:wait_for_profile_location_alignment_then_refresh_gate_stack:portfolio_margin_um",
            "gate_stack_status": "shortline_gate_stack_blocked_at_profile_location",
            "gate_stack_decision": "wait_for_profile_location_alignment_then_refresh_gate_stack",
            "primary_stage": "profile_location",
            "blocker_title": "Track profile-location alignment before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_gate_stack_progress",
            "next_action": "wait_for_profile_location_alignment_then_refresh_gate_stack",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT clears profile alignment and refreshes gate stack",
        },
    )
    _write_json(
        review_dir / "20260316T062010Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_approaching:SOLUSDT:monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_approaching",
            "watch_decision": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "blocker_title": "Track final LVN-to-HVN/POC rotation before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action": "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT rotates into HVN/POC and clears key-level context",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-16T06:22:11Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert (
        payload["ticket_actionability_status"]
        == "profile_location_lvn_key_level_ready_rotation_approaching"
    )
    assert (
        payload["ticket_actionability_decision"]
        == "monitor_final_profile_rotation_into_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert (
        payload["profile_location_watch_status"]
        == "profile_location_lvn_key_level_ready_rotation_approaching"
    )
    assert payload["profile_location_watch_artifact"].endswith(
        "_crypto_shortline_profile_location_watch.json"
    )


def test_build_remote_ticket_actionability_state_prefers_mss_watch_for_mss_stage(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260316T082000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T082001Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-16T08:20:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260316T082000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-16",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-16",
                    "allowed": False,
                    "reasons": ["confidence_below_threshold"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T082002Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_bias_only",
            "focus_review_brief": "review_bias_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260316T082003Z_crypto_shortline_execution_gate.json",
        {
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
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T082004Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "degraded"},
    )
    _write_json(
        review_dir / "20260316T082005Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-16:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260316T082006Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-16",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260316T082007Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_route_not_setup_ready:SOLUSDT:wait_for_setup_ready:portfolio_margin_um",
            "diagnosis_status": "shortline_route_not_setup_ready",
            "diagnosis_decision": "wait_for_setup_ready",
            "primary_constraint_code": "route_not_setup_ready",
        },
    )
    _write_json(
        review_dir / "20260316T082008Z_crypto_shortline_setup_transition_watch.json",
        {
            "transition_brief": "shortline_setup_transition_wait_mss:SOLUSDT:wait_for_mss_then_recheck_execution_gate:portfolio_margin_um",
            "transition_status": "shortline_setup_transition_wait_mss",
            "transition_decision": "wait_for_mss_then_recheck_execution_gate",
            "primary_missing_gate": "mss",
        },
    )
    _write_json(
        review_dir / "20260316T082009Z_crypto_shortline_gate_stack_progress.json",
        {
            "gate_stack_brief": "shortline_gate_stack_blocked_at_mss:SOLUSDT:wait_for_mss_then_recheck_execution_gate:portfolio_margin_um",
            "gate_stack_status": "shortline_gate_stack_blocked_at_mss",
            "gate_stack_decision": "wait_for_mss_then_recheck_execution_gate",
            "primary_stage": "mss",
            "blocker_title": "Track market-structure shift before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_gate_stack_progress",
            "next_action": "wait_for_mss_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT confirms MSS and refreshes gate stack",
        },
    )
    _write_json(
        review_dir / "20260316T082010Z_crypto_shortline_mss_watch.json",
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
            "--config",
            str(config_path),
            "--now",
            "2026-03-16T08:22:11Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "mss_waiting_after_liquidity_sweep"
    assert payload["ticket_actionability_decision"] == "wait_for_mss_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_mss_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_mss_watch"
    assert payload["mss_watch_status"] == "mss_waiting_after_liquidity_sweep"
    assert payload["mss_watch_artifact"].endswith("_crypto_shortline_mss_watch.json")


def test_build_remote_ticket_actionability_state_ignores_future_stamped_ticket_surface(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260316T051000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    current_ticket_path = review_dir / "20260316T051000Z_signal_to_order_tickets.json"
    _write_json(
        current_ticket_path,
        {
            "generated_at_utc": "2026-03-16T05:10:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260316T051000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-16",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-16",
                    "allowed": False,
                    "reasons": ["confidence_below_threshold"],
                }
            ],
        },
    )
    future_ticket_path = review_dir / "20260316T082500Z_signal_to_order_tickets.json"
    _write_json(
        future_ticket_path,
        {
            "generated_at_utc": "2026-03-16T08:25:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260316T082500Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-16",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-16",
                    "allowed": False,
                    "reasons": ["stale_signal"],
                }
            ],
        },
    )
    future_mtime = current_ticket_path.stat().st_mtime + 60
    os.utime(future_ticket_path, (future_mtime, future_mtime))
    _write_json(
        review_dir / "20260316T051001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_bias_only",
            "focus_review_brief": "review_bias_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260316T051002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [
                        "profile_location=LVN",
                        "cvd_key_level_context",
                        "mss",
                    ],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T051003Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "degraded"},
    )
    _write_json(
        review_dir / "20260316T051004Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-16:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260316T051005Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-16",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260316T051006Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_route_not_setup_ready:SOLUSDT:wait_for_setup_ready:portfolio_margin_um",
            "diagnosis_status": "shortline_route_not_setup_ready",
            "diagnosis_decision": "wait_for_setup_ready",
            "primary_constraint_code": "route_not_setup_ready",
        },
    )
    _write_json(
        review_dir / "20260316T051007Z_crypto_shortline_setup_transition_watch.json",
        {
            "transition_brief": "shortline_setup_transition_wait_fvg_ob_breaker_retest:SOLUSDT:wait_for_fvg_ob_breaker_retest_then_recheck_execution_gate:portfolio_margin_um",
            "transition_status": "shortline_setup_transition_wait_fvg_ob_breaker_retest",
            "transition_decision": "wait_for_fvg_ob_breaker_retest_then_recheck_execution_gate",
            "primary_missing_gate": "fvg_ob_breaker_retest",
        },
    )
    _write_json(
        review_dir / "20260316T051008Z_crypto_shortline_gate_stack_progress.json",
        {
            "gate_stack_brief": "shortline_gate_stack_blocked_at_profile_location:SOLUSDT:wait_for_profile_location_alignment_then_refresh_gate_stack:portfolio_margin_um",
            "gate_stack_status": "shortline_gate_stack_blocked_at_profile_location",
            "gate_stack_decision": "wait_for_profile_location_alignment_then_refresh_gate_stack",
            "primary_stage": "profile_location",
            "blocker_title": "Track profile-location alignment before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_gate_stack_progress",
            "next_action": "wait_for_profile_location_alignment_then_refresh_gate_stack",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT clears profile alignment and refreshes gate stack",
        },
    )
    _write_json(
        review_dir / "20260316T051009Z_crypto_shortline_profile_location_watch.json",
        {
            "watch_brief": "profile_location_lvn_key_level_ready_rotation_far:SOLUSDT:monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "profile_location_lvn_key_level_ready_rotation_far",
            "watch_decision": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "blocker_title": "Track LVN-to-HVN/POC rotation before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_profile_location_watch",
            "next_action": "monitor_profile_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT rotates into HVN/POC and clears key-level context",
        },
    )
    _write_json(
        review_dir / "20260316T051010Z_crypto_shortline_pattern_router.json",
        {
            "pattern_brief": "value_rotation_scalp_wait_profile_alignment_far:SOLUSDT:monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um:value_rotation_scalp",
            "pattern_status": "value_rotation_scalp_wait_profile_alignment_far",
            "pattern_decision": "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "blocker_title": "Track value-rotation alignment before shortline scalp promotion",
            "blocker_target_artifact": "crypto_shortline_pattern_router",
            "next_action": "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT rotates toward HVN/POC and the value-rotation scalp can be reassessed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-16T05:12:11Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_artifact"].endswith("_signal_to_order_tickets.json")
    assert payload["ticket_artifact"] == str(current_ticket_path)
    assert payload["ticket_artifact_status"] == "fresh_artifact"
    assert payload["ticket_match_brief"] == (
        "fresh_artifact:ticket_row_blocked:SOLUSDT | confidence_below_threshold"
    )
    assert payload["ticket_actionability_status"] == "value_rotation_scalp_wait_profile_alignment_far"
    assert (
        payload["ticket_actionability_decision"]
        == "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_pattern_router"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["pattern_router_family"] == "value_rotation_scalp"


def test_build_remote_ticket_actionability_state_prefers_pattern_router_primary_constraint_code(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    current_ticket_path = review_dir / "20260316T091650Z_signal_to_order_tickets.json"
    _write_json(
        review_dir / "20260316T091600Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        current_ticket_path,
        {
            "generated_at_utc": "2026-03-16T09:16:50Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260316T091645Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-16",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-16",
                    "allowed": False,
                    "reasons": [
                        "fvg_ob_breaker_retest",
                        "route_state=watch:deprioritize_flow",
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T091601Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_bias_only",
            "focus_review_brief": "review_bias_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260316T091602Z_crypto_shortline_execution_gate.json",
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
        review_dir / "20260316T091603Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-16:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260316T091603Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )
    _write_json(
        review_dir / "20260316T091604Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-16",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260316T091605Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "imbalance_continuation_wait_retest_far:SOLUSDT:monitor_imbalance_retest_band_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_status": "imbalance_continuation_wait_retest_far",
            "diagnosis_decision": "monitor_imbalance_retest_band_then_recheck_execution_gate",
            "primary_constraint_code": "pattern_router:imbalance_continuation:imbalance_retest",
            "blocker_target_artifact": "crypto_shortline_pattern_router",
        },
    )
    _write_json(
        review_dir / "20260316T091606Z_crypto_shortline_setup_transition_watch.json",
        {
            "transition_brief": "imbalance_continuation_wait_retest_far:SOLUSDT:monitor_imbalance_retest_band_then_recheck_execution_gate:portfolio_margin_um",
            "transition_status": "imbalance_continuation_wait_retest_far",
            "transition_decision": "monitor_imbalance_retest_band_then_recheck_execution_gate",
            "primary_missing_gate": "fvg_ob_breaker_retest",
        },
    )
    _write_json(
        review_dir / "20260316T091607Z_crypto_shortline_gate_stack_progress.json",
        {
            "gate_stack_brief": "shortline_gate_stack_blocked_at_fvg_ob_breaker_retest:SOLUSDT:wait_for_imbalance_retest_then_recheck_execution_gate:portfolio_margin_um",
            "gate_stack_status": "shortline_gate_stack_blocked_at_fvg_ob_breaker_retest",
            "gate_stack_decision": "wait_for_imbalance_retest_then_recheck_execution_gate",
            "primary_stage": "fvg_ob_breaker_retest",
            "blocker_title": "Wait for imbalance retest before shortline continuation promotion",
            "blocker_target_artifact": "crypto_shortline_retest_watch",
            "next_action": "wait_for_imbalance_retest_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_retest_watch",
            "done_when": "SOLUSDT retests the imbalance zone and the continuation can be reassessed",
        },
    )
    _write_json(
        review_dir / "20260316T091608Z_crypto_shortline_pattern_router.json",
        {
            "pattern_brief": "imbalance_continuation_wait_retest_far:SOLUSDT:monitor_imbalance_retest_band_then_recheck_execution_gate:portfolio_margin_um:imbalance_continuation",
            "pattern_status": "imbalance_continuation_wait_retest_far",
            "pattern_decision": "monitor_imbalance_retest_band_then_recheck_execution_gate",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
            "blocker_title": "Track imbalance retest band before continuation promotion",
            "blocker_target_artifact": "crypto_shortline_pattern_router",
            "next_action": "monitor_imbalance_retest_band_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT enters the final imbalance retest band and the continuation can be reassessed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-16T09:16:59Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "imbalance_continuation_wait_retest_far"
    assert (
        payload["ticket_actionability_decision"]
        == "monitor_imbalance_retest_band_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_pattern_router"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["pattern_router_family"] == "imbalance_continuation"


def test_build_remote_ticket_actionability_state_prefers_price_reference_watch_for_proxy_only_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:00:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260315T142000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-15",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "allowed": False,
                    "reasons": ["proxy_price_reference_only"],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_watch_only",
            "focus_review_brief": "review_watch_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT setup waits executable price reference.",
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Setup_Ready",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "watch_only"},
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-15:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260315T100060Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-15",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260315T100070Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_proxy_price_only:SOLUSDT:build_price_template_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_status": "shortline_proxy_price_only",
            "diagnosis_decision": "build_price_template_then_recheck_execution_gate",
            "primary_constraint_code": "proxy_price_reference_only",
            "blocker_title": "Build executable price reference before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_ticket_constraint_diagnosis",
            "next_action": "build_price_template_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT keeps execution_price_ready=true and the route ticket drops proxy_price_reference_only",
        },
    )
    _write_json(
        review_dir / "20260315T100080Z_crypto_shortline_price_reference_watch.json",
        {
            "watch_brief": "price_reference_missing_template_proxy_only:SOLUSDT:build_price_template_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "price_reference_missing_template_proxy_only",
            "watch_decision": "build_price_template_then_recheck_execution_gate",
            "blocker_title": "Build executable price reference before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_price_reference_watch",
            "next_action": "build_price_template_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_price_reference_watch",
            "done_when": "SOLUSDT keeps execution_price_ready=true, entry/stop/target stay non-zero, and the route ticket drops proxy_price_reference_only",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:07:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "price_reference_missing_template_proxy_only"
    assert payload["ticket_actionability_decision"] == "build_price_template_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_price_reference_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_price_reference_watch"
    assert payload["price_reference_watch_status"] == "price_reference_missing_template_proxy_only"


def test_build_remote_ticket_actionability_state_prefers_signal_quality_watch_for_quality_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:00:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260315T142000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-15",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "allowed": False,
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "size_below_min_notional",
                    ],
                    "signal": {
                        "confidence": 14.0,
                        "convexity_ratio": 1.4,
                    },
                    "sizing": {
                        "quote_usdt": 3.2,
                        "min_notional_usdt": 5.0,
                    },
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_watch_only",
            "focus_review_brief": "review_watch_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT setup waits signal quality recovery.",
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Setup_Ready",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "watch_only"},
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-15:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260315T100060Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-15",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260315T100070Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_signal_quality_blocked:SOLUSDT:improve_shortline_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_status": "shortline_signal_quality_blocked",
            "diagnosis_decision": "improve_shortline_signal_quality_then_recheck_execution_gate",
            "primary_constraint_code": "confidence_below_threshold",
            "blocker_title": "Improve shortline signal quality before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_ticket_constraint_diagnosis",
            "next_action": "improve_shortline_signal_quality_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT clears quality blockers on the route ticket",
        },
    )
    _write_json(
        review_dir / "20260315T100080Z_crypto_shortline_signal_quality_watch.json",
        {
            "watch_brief": "signal_quality_confidence_convexity_below_threshold:SOLUSDT:improve_shortline_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "signal_quality_confidence_convexity_below_threshold",
            "watch_decision": "improve_shortline_signal_quality_then_recheck_execution_gate",
            "blocker_title": "Improve shortline signal quality before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_signal_quality_watch",
            "next_action": "improve_shortline_signal_quality_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_signal_quality_watch",
            "done_when": "SOLUSDT clears confidence/convexity blockers on the route ticket",
        },
    )
    _write_json(
        review_dir / "20260315T100081Z_crypto_shortline_sizing_watch.json",
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
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:07:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "signal_quality_confidence_convexity_below_threshold"
    assert (
        payload["ticket_actionability_decision"]
        == "improve_shortline_signal_quality_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_signal_quality_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_signal_quality_watch"
    assert payload["signal_quality_watch_status"] == "signal_quality_confidence_convexity_below_threshold"
    assert payload["sizing_watch_status"] == "ticket_size_below_min_notional"


def test_build_remote_ticket_actionability_state_accepts_pattern_aware_signal_quality_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:00:00Z",
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "allowed": False,
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                    ],
                    "signal": {"confidence": 14.0, "convexity_ratio": 1.4},
                    "sizing": {"quote_usdt": 6.2, "min_notional_usdt": 5.0},
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_watch_only",
            "focus_review_brief": "review_watch_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT value_rotation_scalp waits quality recovery.",
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Setup_Ready",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "watch_only"},
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-15:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260315T100060Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-15",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260315T100065Z_crypto_shortline_pattern_router.json",
        {
            "pattern_brief": "value_rotation_scalp_wait_profile_alignment_far:SOLUSDT:monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um:value_rotation_scalp",
            "pattern_status": "value_rotation_scalp_wait_profile_alignment_far",
            "pattern_decision": "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "blocker_title": "Track value-rotation alignment before shortline scalp promotion",
            "blocker_target_artifact": "crypto_shortline_pattern_router",
        },
    )
    _write_json(
        review_dir / "20260315T100070Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_signal_quality_blocked:SOLUSDT:improve_shortline_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_status": "shortline_signal_quality_blocked",
            "diagnosis_decision": "improve_shortline_signal_quality_then_recheck_execution_gate",
            "primary_constraint_code": "confidence_below_threshold",
            "blocker_title": "Improve shortline signal quality before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_ticket_constraint_diagnosis",
            "next_action": "improve_shortline_signal_quality_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT clears quality blockers on the route ticket",
        },
    )
    _write_json(
        review_dir / "20260315T100080Z_crypto_shortline_signal_quality_watch.json",
        {
            "watch_brief": "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold:SOLUSDT:improve_value_rotation_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold",
            "watch_decision": "improve_value_rotation_signal_quality_then_recheck_execution_gate",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "blocker_title": "Improve value-rotation signal quality before shortline scalp promotion",
            "blocker_target_artifact": "crypto_shortline_signal_quality_watch",
            "next_action": "improve_value_rotation_signal_quality_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_signal_quality_watch",
            "done_when": "SOLUSDT clears confidence/convexity blockers while the value-rotation scalp remains active",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:07:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert (
        payload["ticket_actionability_status"]
        == "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold"
    )
    assert (
        payload["ticket_actionability_decision"]
        == "improve_value_rotation_signal_quality_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_signal_quality_watch"
    assert payload["signal_quality_watch_status"] == (
        "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold"
    )


def test_build_remote_ticket_actionability_state_prefers_sizing_watch_for_size_blocker(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260315T100000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260315T100010Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-15T10:00:00Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260315T142000Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-15",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "allowed": False,
                    "reasons": ["size_below_min_notional"],
                    "sizing": {
                        "quote_usdt": 3.2,
                        "min_notional_usdt": 5.0,
                    },
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100020Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_watch_only",
            "focus_review_brief": "review_watch_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT setup waits ticket sizing recovery.",
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Setup_Ready",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "watch_only"},
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-15:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260315T100060Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-15",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260315T100070Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_ticket_size_blocked:SOLUSDT:raise_effective_shortline_size_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_status": "shortline_ticket_size_blocked",
            "diagnosis_decision": "raise_effective_shortline_size_then_recheck_execution_gate",
            "primary_constraint_code": "size_below_min_notional",
            "blocker_title": "Raise effective shortline size before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_ticket_constraint_diagnosis",
            "next_action": "raise_effective_shortline_size_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_execution_gate",
            "done_when": "SOLUSDT clears size_below_min_notional on the route ticket",
        },
    )
    _write_json(
        review_dir / "20260315T100080Z_crypto_shortline_signal_quality_watch.json",
        {
            "watch_brief": "signal_quality_watch_clear:SOLUSDT:recheck_execution_gate_after_signal_quality_clear:portfolio_margin_um",
            "watch_status": "signal_quality_watch_clear",
            "watch_decision": "recheck_execution_gate_after_signal_quality_clear",
        },
    )
    _write_json(
        review_dir / "20260315T100081Z_crypto_shortline_sizing_watch.json",
        {
            "watch_brief": "ticket_size_below_min_notional:SOLUSDT:raise_effective_shortline_size_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "ticket_size_below_min_notional",
            "watch_decision": "raise_effective_shortline_size_then_recheck_execution_gate",
            "blocker_title": "Raise effective shortline size before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_sizing_watch",
            "next_action": "raise_effective_shortline_size_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_sizing_watch",
            "done_when": "SOLUSDT keeps quote_usdt above min_notional_usdt and the route ticket drops size_below_min_notional",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-15T10:07:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "ticket_size_below_min_notional"
    assert (
        payload["ticket_actionability_decision"]
        == "raise_effective_shortline_size_then_recheck_execution_gate"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_sizing_watch"
    assert payload["next_action_target_artifact"] == "crypto_shortline_sizing_watch"
    assert payload["signal_quality_watch_status"] == "signal_quality_watch_clear"
    assert payload["sizing_watch_status"] == "ticket_size_below_min_notional"


def test_build_remote_ticket_actionability_state_surfaces_execution_quality_watch(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("shortline: {}\n", encoding="utf-8")
    _write_json(
        review_dir / "20260316T102600Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T102601Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-16T10:26:01Z",
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "path": "/tmp/20260316T102601Z_crypto_shortline_signal_source.json",
                "selection_reason": "matched_symbol_rows",
                "artifact_date": "2026-03-16",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": ["size_below_min_notional"],
                    "sizing": {
                        "quote_usdt": 3.2,
                        "min_notional_usdt": 5.0,
                    },
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T102602Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_watch_only",
            "focus_review_brief": "review_watch_only:SOLUSDT",
            "focus_review_blocker_detail": "SOLUSDT setup waits ticket sizing recovery.",
        },
    )
    _write_json(
        review_dir / "20260316T102603Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Setup_Ready",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": [],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T102604Z_crypto_cvd_queue_handoff.json",
        {"queue_status": "derived", "semantic_status": "watch_only"},
    )
    _write_json(
        review_dir / "20260316T102605Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-16:age_days=0:crypto_shortline_signal_source",
            "freshness_status": "route_signal_row_fresh",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_recommended": False,
        },
    )
    _write_json(
        review_dir / "20260316T102606Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-16",
            "readiness_status": "signal_source_refresh_not_required",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
            "refresh_needed": False,
        },
    )
    _write_json(
        review_dir / "20260316T102607Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_ticket_size_blocked:SOLUSDT:raise_effective_shortline_size_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_status": "shortline_ticket_size_blocked",
            "diagnosis_decision": "raise_effective_shortline_size_then_recheck_execution_gate",
            "primary_constraint_code": "size_below_min_notional",
            "blocker_title": "Raise effective shortline size before guarded canary review",
            "blocker_target_artifact": "crypto_shortline_ticket_constraint_diagnosis",
            "next_action": "raise_effective_shortline_size_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_sizing_watch",
            "done_when": "SOLUSDT clears size_below_min_notional on the route ticket",
        },
    )
    _write_json(
        review_dir / "20260316T102608Z_crypto_shortline_signal_quality_watch.json",
        {
            "watch_brief": "signal_quality_watch_clear:SOLUSDT:recheck_execution_gate_after_signal_quality_clear:portfolio_margin_um",
            "watch_status": "signal_quality_watch_clear",
            "watch_decision": "recheck_execution_gate_after_signal_quality_clear",
        },
    )
    _write_json(
        review_dir / "20260316T102609Z_crypto_shortline_sizing_watch.json",
        {
            "watch_brief": "ticket_size_below_min_notional:SOLUSDT:raise_effective_shortline_size_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "ticket_size_below_min_notional",
            "watch_decision": "raise_effective_shortline_size_then_recheck_execution_gate",
            "blocker_title": "Raise effective shortline size before shortline setup promotion",
            "blocker_target_artifact": "crypto_shortline_sizing_watch",
            "next_action": "raise_effective_shortline_size_then_recheck_execution_gate",
            "next_action_target_artifact": "crypto_shortline_sizing_watch",
            "done_when": "SOLUSDT keeps quote_usdt above min_notional_usdt and the route ticket drops size_below_min_notional",
        },
    )
    _write_json(
        review_dir / "20260316T102610Z_crypto_shortline_execution_quality_watch.json",
        {
            "watch_brief": "value_rotation_scalp_execution_quality_degraded:SOLUSDT:repair_value_rotation_execution_quality_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "value_rotation_scalp_execution_quality_degraded",
            "watch_decision": "repair_value_rotation_execution_quality_then_recheck_execution_gate",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
        },
    )
    _write_json(
        review_dir / "20260316T102611Z_crypto_shortline_slippage_snapshot.json",
        {
            "snapshot_brief": "value_rotation_scalp_post_cost_degraded:SOLUSDT:repair_value_rotation_post_cost_then_recheck_execution_gate:portfolio_margin_um",
            "snapshot_status": "value_rotation_scalp_post_cost_degraded",
            "snapshot_decision": "repair_value_rotation_post_cost_then_recheck_execution_gate",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "post_cost_viable": False,
        },
    )
    _write_json(
        review_dir / "20260316T102612Z_crypto_shortline_fill_capacity_watch.json",
        {
            "watch_brief": "value_rotation_scalp_fill_capacity_constrained:SOLUSDT:repair_value_rotation_fill_capacity_then_recheck_execution_gate:portfolio_margin_um",
            "watch_status": "value_rotation_scalp_fill_capacity_constrained",
            "watch_decision": "repair_value_rotation_fill_capacity_then_recheck_execution_gate",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "fill_capacity_viable": False,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-16T10:27:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "ticket_size_below_min_notional"
    assert payload["execution_quality_watch_status"] == (
        "value_rotation_scalp_execution_quality_degraded"
    )
    assert payload["execution_quality_watch_decision"] == (
        "repair_value_rotation_execution_quality_then_recheck_execution_gate"
    )
    assert payload["slippage_snapshot_status"] == "value_rotation_scalp_post_cost_degraded"
    assert payload["slippage_snapshot_decision"] == (
        "repair_value_rotation_post_cost_then_recheck_execution_gate"
    )
    assert payload["fill_capacity_watch_status"] == (
        "value_rotation_scalp_fill_capacity_constrained"
    )
    assert payload["fill_capacity_watch_decision"] == (
        "repair_value_rotation_fill_capacity_then_recheck_execution_gate"
    )
    assert "execution_quality_watch=value_rotation_scalp_execution_quality_degraded:SOLUSDT:repair_value_rotation_execution_quality_then_recheck_execution_gate:portfolio_margin_um:repair_value_rotation_execution_quality_then_recheck_execution_gate" in payload[
        "blocker_detail"
    ]
    assert "slippage_snapshot=value_rotation_scalp_post_cost_degraded:SOLUSDT:repair_value_rotation_post_cost_then_recheck_execution_gate:portfolio_margin_um:repair_value_rotation_post_cost_then_recheck_execution_gate" in payload[
        "blocker_detail"
    ]
    assert "fill_capacity_watch=value_rotation_scalp_fill_capacity_constrained:SOLUSDT:repair_value_rotation_fill_capacity_then_recheck_execution_gate:portfolio_margin_um:repair_value_rotation_fill_capacity_then_recheck_execution_gate" in payload[
        "blocker_detail"
    ]


def test_build_remote_ticket_actionability_state_scopes_route_artifacts_to_explicit_symbol(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
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
            "focus_review_status": "review_bias_only",
            "focus_review_brief": "review_bias_only:SOLUSDT:deprioritize_flow",
            "focus_review_blocker_detail": "SOL only focus context",
        },
    )
    _write_json(
        review_dir / "20260318T114002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deploy_price_state_only",
                    "route_state": "promoted",
                    "missing_gates": ["mss", "fvg_ob_breaker_retest", "cvd_confirmation"],
                },
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                },
            ]
        },
    )
    _write_json(
        review_dir / "20260318T114003Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
            "runtime_queue": [
                {
                    "batch": "crypto_hot",
                    "eligible_symbols": ["ETHUSDT", "SOLUSDT"],
                    "matching_symbols": [
                        {"symbol": "ETHUSDT", "classification": "Bias_Only"},
                        {"symbol": "SOLUSDT", "classification": "Bias_Only"},
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260318T114004Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-18T11:44:00Z",
            "tickets": [
                {
                    "symbol": "ETHUSDT",
                    "allowed": False,
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "route_not_setup_ready",
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260318T114005Z_crypto_shortline_gate_stack_progress.json",
        {
            "route_symbol": "SOLUSDT",
            "gate_stack_brief": "sol_only_gate_progress",
            "gate_stack_status": "shortline_gate_stack_blocked_at_liquidity_sweep",
            "gate_stack_decision": "wait_for_sol_only_gate",
            "primary_stage": "liquidity_sweep",
            "blocker_title": "SOL only blocker",
            "blocker_target_artifact": "crypto_shortline_gate_stack_progress",
        },
    )
    _write_json(
        review_dir / "20260318T114006Z_crypto_shortline_mss_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_brief": "sol_only_mss_watch",
            "watch_status": "mss_waiting_for_confirmation",
            "watch_decision": "wait_for_sol_mss",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
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
    assert payload["ticket_row_reasons"] == [
        "confidence_below_threshold",
        "convexity_below_threshold",
        "route_not_setup_ready",
    ]
    assert payload["gate_stack_progress_artifact"] == ""
    assert payload["mss_watch_artifact"] == ""
    assert payload["gate_stack_progress_brief"] == ""
    assert payload["mss_watch_brief"] == ""
    assert payload["focus_review_brief"] == ""
    assert payload["focus_review_blocker_detail"] == ""
    assert "SOL only blocker" not in payload["blocker_detail"]
    assert "SOL only focus context" not in payload["blocker_detail"]


def test_build_remote_ticket_actionability_state_ignores_missing_ticket_surface_paths_during_sort(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
shortline:
  supported_symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        review_dir / "20260318T114000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "ETHUSDT",
            "preferred_route_action": "deploy_price_state_only",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T114001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "ETHUSDT",
            "next_focus_symbol": "ETHUSDT",
            "next_focus_action": "deploy_price_state_only",
        },
    )
    _write_json(
        review_dir / "20260318T114002Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deploy_price_state_only",
                    "route_state": "promoted",
                    "missing_gates": ["mss"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260318T114003Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "watch_only",
            "next_focus_batch": "crypto_hot",
            "runtime_queue": [
                {
                    "batch": "crypto_hot",
                    "eligible_symbols": ["ETHUSDT"],
                    "matching_symbols": [{"symbol": "ETHUSDT", "classification": "Bias_Only"}],
                }
            ],
        },
    )
    os.symlink(
        review_dir / "missing_target.json",
        review_dir / "20260318T114500Z_signal_to_order_tickets.json",
    )
    _write_json(
        review_dir / "20260318T114400Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-18T11:44:00Z",
            "tickets": [
                {
                    "symbol": "ETHUSDT",
                    "allowed": False,
                    "reasons": ["route_not_setup_ready"],
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--config",
            str(config_path),
            "--now",
            "2026-03-18T11:45:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_artifact"].endswith("20260318T114400Z_signal_to_order_tickets.json")
    assert payload["ticket_row_reasons"] == ["route_not_setup_ready"]
