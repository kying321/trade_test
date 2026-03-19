from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_ticket_constraint_diagnosis.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_ticket_constraint_diagnosis_classifies_route_not_ready(
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
        review_dir / "20260315T100020Z_signal_to_order_tickets.json",
        {
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "artifact_date": "2026-03-15",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "proxy_price_reference_only",
                        "size_below_min_notional",
                    ],
                    "signal": {
                        "confidence": 24.0,
                        "convexity_ratio": 1.4,
                        "execution_price_ready": False,
                    },
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_signal_source_freshness.json",
        {
            "freshness_brief": "route_signal_row_fresh:SOLUSDT:2026-03-15:age_days=0:crypto_shortline_signal_source",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
        },
    )
    _write_json(
        review_dir / "20260315T100050Z_crypto_signal_source_refresh_readiness.json",
        {
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT:2026-03-15",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
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
    assert payload["ticket_actionability_status"] == (
        "shortline_route_not_setup_ready_proxy_price_blocked"
    )
    assert payload["ticket_actionability_decision"] == (
        "wait_for_setup_ready_and_executable_price_reference"
    )
    assert payload["blocker_title"] == (
        "Wait for setup-ready route and executable price reference before guarded canary review"
    )
    assert payload["next_action_target_artifact"] == "crypto_shortline_execution_gate"
    assert payload["primary_constraint_code"] == "route_not_setup_ready"


def test_build_crypto_shortline_ticket_constraint_diagnosis_prefers_route_symbol_surface(
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
        review_dir / "20260315T100020Z_signal_to_order_tickets.json",
        {
            "signal_source": {"kind": "crypto_shortline_signal_source", "artifact_date": "2026-03-15"},
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "reasons": ["proxy_price_reference_only"],
                    "signal": {"execution_price_ready": False},
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_signal_to_order_tickets.json",
        {
            "signal_source": {"kind": "explicit", "artifact_date": "2026-03-15"},
            "tickets": [{"symbol": "XAUUSD", "allowed": True, "reasons": []}],
        },
    )
    _write_json(
        review_dir / "20260315T100040Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "missing_gates": ["liquidity_sweep"],
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
            "2026-03-15T10:06:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["artifacts"]["signal_to_order_tickets"].endswith(
        "20260315T100020Z_signal_to_order_tickets.json"
    )
    assert payload["ticket_row_reasons"] == ["proxy_price_reference_only"]
    assert payload["ticket_signal_source_kind"] == "crypto_shortline_signal_source"


def test_build_crypto_shortline_ticket_constraint_diagnosis_drops_proxy_block_when_price_reference_ready(
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
        review_dir / "20260315T100020Z_signal_to_order_tickets.json",
        {
            "signal_source": {"kind": "crypto_shortline_signal_source", "artifact_date": "2026-03-15"},
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-15",
                    "reasons": ["proxy_price_reference_only", "confidence_below_threshold"],
                    "signal": {"execution_price_ready": True},
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "missing_gates": ["liquidity_sweep"],
                }
            ]
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
    assert payload["ticket_actionability_status"] == "shortline_route_not_setup_ready"
    assert payload["ticket_actionability_decision"] == "wait_for_setup_ready_then_recheck_execution_gate"
    assert payload["price_reference_blocked"] is False
    assert payload["ticket_row_reasons"] == ["confidence_below_threshold"]


def test_build_crypto_shortline_ticket_constraint_diagnosis_prefers_pattern_router_when_present(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T101000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T101010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260316T101020Z_signal_to_order_tickets.json",
        {
            "signal_source": {
                "kind": "crypto_shortline_signal_source",
                "artifact_date": "2026-03-16",
            },
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-16",
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "size_below_min_notional",
                    ],
                    "signal": {
                        "confidence": 24.0,
                        "convexity_ratio": 1.4,
                        "execution_price_ready": True,
                    },
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T101030Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "missing_gates": ["profile_location=LVN", "mss", "cvd_confirmation"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T101040Z_crypto_shortline_pattern_router.json",
        {
            "pattern_brief": "value_rotation_scalp_wait_profile_alignment_far:SOLUSDT:monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate:portfolio_margin_um:value_rotation_scalp",
            "pattern_status": "value_rotation_scalp_wait_profile_alignment_far",
            "pattern_decision": "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "blocker_title": "Track value-rotation alignment before shortline scalp promotion",
            "blocker_target_artifact": "crypto_shortline_pattern_router",
            "next_action_target_artifact": "crypto_shortline_profile_location_watch",
            "done_when": "SOLUSDT rotates from LVN toward HVN/POC, then the shortline execution gate can reassess whether the value-rotation scalp is executable",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T10:10:50Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ticket_actionability_status"] == "value_rotation_scalp_wait_profile_alignment_far"
    assert payload["ticket_actionability_decision"] == "monitor_value_rotation_toward_hvn_poc_then_recheck_execution_gate"
    assert payload["blocker_title"] == "Track value-rotation alignment before shortline scalp promotion"
    assert payload["blocker_target_artifact"] == "crypto_shortline_pattern_router"
    assert payload["next_action_target_artifact"] == "crypto_shortline_profile_location_watch"
    assert payload["primary_constraint_code"] == "pattern_router:value_rotation_scalp:profile_alignment"
    assert payload["pattern_router_family"] == "value_rotation_scalp"


def test_build_crypto_shortline_ticket_constraint_diagnosis_emits_quantified_gaps(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T111100Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "BNBUSDT",
            "preferred_route_action": "watch",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T111110Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "BNBUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch",
            "focus_review_blocker_detail": "BNBUSDT remains Bias_Only.",
        },
    )
    _write_json(
        review_dir / "20260318T111120Z_signal_to_order_tickets.json",
        {
            "signal_source": {"kind": "crypto_shortline_signal_source", "artifact_date": "2026-03-18"},
            "thresholds": {
                "min_confidence": 66.75,
                "min_convexity": 2.5582666849,
                "base_risk_pct": 0.45,
                "min_notional_usdt": 5.0,
            },
            "tickets": [
                {
                    "symbol": "BNBUSDT",
                    "date": "2026-03-18",
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                        "size_below_min_notional",
                    ],
                    "signal": {
                        "confidence": 12.0,
                        "convexity_ratio": 1.577972,
                        "execution_price_ready": True,
                    },
                    "sizing": {
                        "quote_usdt": 2.5938951196,
                        "min_notional_usdt": 5.0,
                        "equity_usdt": 70.0,
                    },
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260318T111130Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "BNBUSDT",
                    "execution_state": "Bias_Only",
                    "missing_gates": [],
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
            "2026-03-18T11:20:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    gaps = payload["constraint_gaps"]
    assert gaps["confidence_gap"] == 54.75
    assert round(gaps["convexity_gap"], 6) == round(2.5582666849 - 1.577972, 6)
    assert round(gaps["quote_gap_usdt"], 6) == round(5.0 - 2.5938951196, 6)
    assert round(gaps["required_equity_usdt_current_signal"], 6) == round(
        70.0 * 5.0 / 2.5938951196, 6
    )
    assert round(gaps["required_base_risk_pct_current_signal"], 6) == round(
        0.45 * 5.0 / 2.5938951196, 6
    )


def test_build_crypto_shortline_ticket_constraint_diagnosis_allows_symbol_override(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T111100Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "watch",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T111110Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "watch",
        },
    )
    _write_json(
        review_dir / "20260318T111120Z_signal_to_order_tickets.json",
        {
            "signal_source": {"kind": "crypto_shortline_signal_source", "artifact_date": "2026-03-18"},
            "tickets": [
                {"symbol": "SOLUSDT", "date": "2026-03-18", "reasons": ["size_below_min_notional"], "signal": {}},
                {"symbol": "BNBUSDT", "date": "2026-03-18", "reasons": ["confidence_below_threshold"], "signal": {}},
            ],
        },
    )
    _write_json(
        review_dir / "20260318T111130Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {"symbol": "SOLUSDT", "execution_state": "Bias_Only", "missing_gates": []},
                {"symbol": "BNBUSDT", "execution_state": "Bias_Only", "missing_gates": []},
            ]
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--symbol",
            "BNBUSDT",
            "--now",
            "2026-03-18T11:20:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "BNBUSDT"
    assert payload["ticket_row_reasons"] == ["confidence_below_threshold"]


def test_build_crypto_shortline_ticket_constraint_diagnosis_ignores_mismatched_route_scoped_artifacts(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T111100Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "watch",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T111110Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "watch",
            "focus_review_blocker_detail": "SOLUSDT route-only blocker detail.",
        },
    )
    _write_json(
        review_dir / "20260318T111120Z_signal_to_order_tickets.json",
        {
            "signal_source": {"kind": "crypto_shortline_signal_source", "artifact_date": "2026-03-18"},
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "date": "2026-03-18",
                    "reasons": ["confidence_below_threshold"],
                    "signal": {"execution_price_ready": True},
                },
                {
                    "symbol": "BNBUSDT",
                    "date": "2026-03-18",
                    "reasons": ["proxy_price_reference_only", "confidence_below_threshold"],
                    "signal": {"execution_price_ready": False},
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260318T111130Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {"symbol": "SOLUSDT", "execution_state": "Bias_Only", "missing_gates": []},
                {"symbol": "BNBUSDT", "execution_state": "Bias_Only", "missing_gates": []},
            ]
        },
    )
    _write_json(
        review_dir / "20260318T111140Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_brief": "imbalance_continuation_wait_retest_far:SOLUSDT:wait_for_imbalance_retest_then_recheck_execution_gate:spot:imbalance_continuation",
            "pattern_status": "imbalance_continuation_wait_retest_far",
            "pattern_decision": "wait_for_imbalance_retest_then_recheck_execution_gate",
            "pattern_family": "imbalance_continuation",
            "pattern_stage": "imbalance_retest",
        },
    )
    _write_json(
        review_dir / "20260318T111150Z_crypto_shortline_price_reference_watch.json",
        {
            "route_symbol": "SOLUSDT",
            "watch_status": "price_reference_ready",
            "watch_brief": "price_reference_ready:SOLUSDT:recheck_shortline_execution_gate_after_price_reference_ready:spot",
            "watch_decision": "recheck_shortline_execution_gate_after_price_reference_ready",
            "price_reference_blocked": False,
        },
    )
    _write_json(
        review_dir / "20260318T111160Z_crypto_signal_source_freshness.json",
        {
            "route_symbol": "SOLUSDT",
            "freshness_brief": "route_signal_row_fresh:SOLUSDT",
            "freshness_decision": "signal_source_fresh_no_refresh_needed",
        },
    )
    _write_json(
        review_dir / "20260318T111170Z_crypto_signal_source_refresh_readiness.json",
        {
            "route_symbol": "SOLUSDT",
            "readiness_brief": "signal_source_refresh_not_required:SOLUSDT",
            "readiness_decision": "signal_source_fresh_no_refresh_needed",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--symbol",
            "BNBUSDT",
            "--now",
            "2026-03-18T11:20:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "BNBUSDT"
    assert payload["pattern_router_family"] == ""
    assert payload["pattern_router_brief"] == ""
    assert payload["price_reference_blocked"] is True
    assert payload["artifacts"]["crypto_shortline_pattern_router"] == ""
    assert payload["artifacts"]["crypto_shortline_price_reference_watch"] == ""
    assert payload["artifacts"]["crypto_signal_source_freshness"] == ""
    assert payload["artifacts"]["crypto_signal_source_refresh_readiness"] == ""
    assert payload["ticket_row_reasons"] == [
        "proxy_price_reference_only",
        "confidence_below_threshold",
    ]
    assert payload["ticket_actionability_status"] == (
        "shortline_route_not_setup_ready_proxy_price_blocked"
    )
    assert "SOLUSDT route-only blocker detail." not in payload["blocker_detail"]


def test_build_crypto_shortline_ticket_constraint_diagnosis_uses_symbol_gate_stack_progress(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T111200Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "watch",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T111210Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "watch",
        },
    )
    _write_json(
        review_dir / "20260318T111220Z_signal_to_order_tickets.json",
        {
            "signal_source": {"kind": "crypto_shortline_signal_source", "artifact_date": "2026-03-18"},
            "tickets": [
                {
                    "symbol": "ETHUSDT",
                    "date": "2026-03-18",
                    "reasons": ["route_not_setup_ready", "confidence_below_threshold"],
                    "signal": {"execution_price_ready": True, "confidence": 40.0, "convexity_ratio": 2.5},
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260318T111230Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "promoted",
                    "missing_gates": ["mss", "route_state=promoted:deploy_price_state_only"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260318T111240Z_crypto_shortline_gate_stack_progress.json",
        {
            "route_symbol": "SOLUSDT",
            "gate_stack_status": "shortline_gate_stack_blocked_at_liquidity_sweep",
            "symbols": [
                {
                    "symbol": "ETHUSDT",
                    "gate_stack_brief": "shortline_gate_stack_blocked_at_mss:ETHUSDT:wait_for_mss_then_refresh_gate_stack:spot",
                    "gate_stack_status": "shortline_gate_stack_blocked_at_mss",
                    "gate_stack_decision": "wait_for_mss_then_refresh_gate_stack",
                    "primary_stage": "mss",
                    "blocker_title": "Track market-structure shift before shortline setup promotion",
                    "blocker_target_artifact": "crypto_shortline_gate_stack_progress",
                    "next_action_target_artifact": "crypto_shortline_mss_watch",
                    "done_when": "ETHUSDT confirms MSS/CHOCH and keeps executable price reference",
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
            "--symbol",
            "ETHUSDT",
            "--now",
            "2026-03-18T11:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["route_symbol"] == "ETHUSDT"
    assert payload["gate_stack_progress_status"] == "shortline_gate_stack_blocked_at_mss"
    assert payload["primary_constraint_code"] == "gate_stack:mss"
    assert payload["ticket_actionability_status"] == "shortline_gate_stack_blocked_at_mss"
    assert payload["next_action_target_artifact"] == "crypto_shortline_mss_watch"
    assert payload["artifacts"]["crypto_shortline_gate_stack_progress"].endswith(
        "_crypto_shortline_gate_stack_progress.json"
    )
    assert "gate_stack_progress=shortline_gate_stack_blocked_at_mss:ETHUSDT" in payload["blocker_detail"]


def test_build_crypto_shortline_ticket_constraint_diagnosis_prefers_latest_matching_gate_stack_progress(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260318T111300Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "watch",
            "remote_market": "spot",
        },
    )
    _write_json(
        review_dir / "20260318T111310Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "watch",
        },
    )
    _write_json(
        review_dir / "20260318T111320Z_signal_to_order_tickets.json",
        {
            "signal_source": {"kind": "crypto_shortline_signal_source", "artifact_date": "2026-03-18"},
            "tickets": [
                {
                    "symbol": "BNBUSDT",
                    "date": "2026-03-18",
                    "reasons": ["route_not_setup_ready", "size_below_min_notional"],
                    "signal": {"execution_price_ready": True, "confidence": 12.0, "convexity_ratio": 1.5},
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260318T111330Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "BNBUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "review",
                    "missing_gates": ["cvd_key_level_context", "mss"],
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260318T111340Z_crypto_shortline_gate_stack_progress.json",
        {
            "route_symbol": "SOLUSDT",
            "symbols": [
                {
                    "symbol": "BNBUSDT",
                    "gate_stack_brief": "shortline_gate_stack_blocked_at_profile_location:BNBUSDT:wait_for_profile_location_alignment_then_refresh_gate_stack:spot",
                    "gate_stack_status": "shortline_gate_stack_blocked_at_profile_location",
                    "gate_stack_decision": "wait_for_profile_location_alignment_then_refresh_gate_stack",
                    "primary_stage": "profile_location",
                    "blocker_title": "Track profile-location alignment before shortline setup promotion",
                    "blocker_target_artifact": "crypto_shortline_gate_stack_progress",
                    "next_action_target_artifact": "crypto_shortline_execution_gate",
                    "done_when": "BNBUSDT regains profile-location alignment",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260318T111350Z_crypto_shortline_gate_stack_progress.json",
        {
            "route_symbol": "ETHUSDT",
            "gate_stack_brief": "shortline_gate_stack_blocked_at_mss:ETHUSDT:wait_for_mss_then_refresh_gate_stack:spot",
            "gate_stack_status": "shortline_gate_stack_blocked_at_mss",
            "gate_stack_decision": "wait_for_mss_then_refresh_gate_stack",
            "primary_stage": "mss",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--symbol",
            "BNBUSDT",
            "--now",
            "2026-03-18T11:31:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["gate_stack_progress_primary_stage"] == "profile_location"
    assert payload["artifacts"]["crypto_shortline_gate_stack_progress"].endswith(
        "20260318T111340Z_crypto_shortline_gate_stack_progress.json"
    )
