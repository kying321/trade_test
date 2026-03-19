from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_material_change_trigger.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_material_change_trigger_waits_when_anchor_has_no_delta(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "degraded",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
            "runtime_queue": [
                {
                    "eligible_symbols": ["SOLUSDT"],
                    "matching_symbols": [
                        {
                            "symbol": "SOLUSDT",
                            "classification": "watch_only",
                            "cvd_context_mode": "continuation",
                            "cvd_veto_hint": "low_sample_or_gap_risk",
                            "cvd_attack_side": "buyers",
                            "active_reasons": ["trust_low", "low_sample_or_gap_risk"],
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "fvg_ob_breaker_retest",
                        "cvd_confirmation",
                    ],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                        "fvg_long": False,
                        "fvg_short": False,
                    },
                    "micro_signals": {
                        "cvd_ready": False,
                        "quality_ok": False,
                        "trust_ok": False,
                        "context": "continuation",
                        "veto_hint": "low_sample_or_gap_risk",
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
                        "attack_confirmation_ok": True,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_no_edge_bias_only_micro_veto",
            "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
        },
    )
    _write_json(
        review_dir / "20260315T105000Z_crypto_shortline_cross_section_backtest.json",
        {
            "selected_symbol": "SOLUSDT",
            "backtest_brief": "watch_only_cross_section_no_edge:SOLUSDT:no_edge:h60m",
            "backtest_status": "watch_only_cross_section_no_edge",
            "research_decision": "deprioritize_until_orderflow_improves",
            "selected_edge_status": "no_edge",
        },
    )
    _write_json(
        review_dir / "20260315T105400Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": [
                        "liquidity_sweep",
                        "mss",
                        "fvg_ob_breaker_retest",
                        "cvd_confirmation",
                    ],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                        "fvg_long": False,
                        "fvg_short": False,
                    },
                    "micro_signals": {
                        "cvd_ready": False,
                        "quality_ok": False,
                        "trust_ok": False,
                        "context": "continuation",
                        "veto_hint": "low_sample_or_gap_risk",
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
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
            "2026-03-15T10:54:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "no_material_orderflow_change_since_cross_section_anchor"
    assert payload["trigger_decision"] == "wait_for_material_orderflow_change_before_rerun"
    assert payload["rerun_recommended"] is False
    assert payload["changed_dimensions"] == []


def test_build_crypto_shortline_material_change_trigger_detects_execution_delta(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260315T100741Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "semantic_status": "degraded",
            "next_focus_batch": "crypto_hot",
            "next_focus_action": "defer_until_micro_recovers",
            "runtime_queue": [
                {
                    "eligible_symbols": ["SOLUSDT"],
                    "matching_symbols": [
                        {
                            "symbol": "SOLUSDT",
                            "classification": "watch_only",
                            "cvd_context_mode": "continuation",
                            "cvd_veto_hint": "low_sample_or_gap_risk",
                            "cvd_attack_side": "buyers",
                            "active_reasons": ["trust_low"],
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260315T100742Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                        "mss_long": False,
                        "mss_short": False,
                        "fvg_long": False,
                        "fvg_short": False,
                    },
                    "micro_signals": {
                        "cvd_ready": False,
                        "quality_ok": False,
                        "trust_ok": False,
                        "context": "continuation",
                        "veto_hint": "low_sample_or_gap_risk",
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
                        "attack_confirmation_ok": True,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T100744Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "focus_review_status": "review_no_edge_bias_only_micro_veto",
            "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
        },
    )
    _write_json(
        review_dir / "20260315T105000Z_crypto_shortline_cross_section_backtest.json",
        {
            "selected_symbol": "SOLUSDT",
            "backtest_brief": "watch_only_cross_section_no_edge:SOLUSDT:no_edge:h60m",
            "backtest_status": "watch_only_cross_section_no_edge",
            "research_decision": "deprioritize_until_orderflow_improves",
            "selected_edge_status": "no_edge",
        },
    )
    _write_json(
        review_dir / "20260315T105400Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Setup_Ready",
                    "route_state": "watch",
                    "missing_gates": [],
                    "structure_signals": {
                        "sweep_long": True,
                        "sweep_short": False,
                        "mss_long": True,
                        "mss_short": False,
                        "fvg_long": True,
                        "fvg_short": False,
                    },
                    "micro_signals": {
                        "cvd_ready": True,
                        "quality_ok": True,
                        "trust_ok": True,
                        "context": "failed_auction",
                        "veto_hint": "",
                        "attack_side": "buyers",
                        "attack_presence": "buyers_attacking",
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
            "2026-03-15T10:54:20Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "material_orderflow_change_detected"
    assert payload["trigger_decision"] == "rerun_shortline_execution_gate_and_recheck_ticket_actionability"
    assert payload["rerun_recommended"] is True
    assert "execution_state" in payload["changed_dimensions"]
    assert "missing_gates" in payload["changed_dimensions"]
