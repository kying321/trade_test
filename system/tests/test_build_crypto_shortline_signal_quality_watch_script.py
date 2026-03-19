from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_signal_quality_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_signal_quality_watch_tracks_quality_and_size_blockers(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T005000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T005005Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T005010Z_signal_to_order_tickets.json",
        {
            "tickets": [
                {
                    "symbol": "SOLUSDT",
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
                        "conviction": 0.098,
                        "quote_usdt": 3.2,
                        "min_notional_usdt": 5.0,
                        "risk_budget_usdt": 0.31,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T005015Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_signal_quality_blocked:SOLUSDT:improve_shortline_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_decision": "improve_shortline_signal_quality_then_recheck_execution_gate",
        },
    )
    _write_json(
        review_dir / "20260316T005016Z_crypto_shortline_sizing_watch.json",
        {
            "watch_status": "ticket_size_below_min_notional",
            "watch_brief": "ticket_size_below_min_notional:SOLUSDT:raise_effective_shortline_size_then_recheck_execution_gate:portfolio_margin_um",
            "watch_decision": "raise_effective_shortline_size_then_recheck_execution_gate",
        },
    )
    _write_json(
        review_dir / "20260316T005018Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_degraded",
            "snapshot_brief": "live_orderflow_snapshot_degraded:SOLUSDT:repair_live_orderflow_snapshot_quality_then_recheck_signal_quality:portfolio_margin_um",
            "snapshot_decision": "repair_live_orderflow_snapshot_quality_then_recheck_signal_quality",
            "micro_quality_ok": False,
            "time_sync_ok": False,
            "queue_imbalance": 0.12,
            "ofi_norm": 0.08,
            "micro_alignment": 0.15,
            "cvd_delta_ratio": 0.11,
            "cvd_context_mode": "continuation",
            "cvd_veto_hint": "low_sample_or_gap_risk",
            "cvd_locality_status": "outside_local_window",
            "cvd_attack_presence": "buyers_attacking",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T00:50:20Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "signal_quality_confidence_convexity_below_threshold"
    assert (
        payload["watch_decision"]
        == "improve_shortline_signal_quality_then_recheck_execution_gate"
    )
    assert payload["confidence_below_threshold"] is True
    assert payload["convexity_below_threshold"] is True
    assert payload["size_below_min_notional"] is True
    assert payload["sizing_watch_status"] == "ticket_size_below_min_notional"
    assert payload["sizing_watch_decision"] == (
        "raise_effective_shortline_size_then_recheck_execution_gate"
    )
    assert payload["quote_usdt"] == 3.2
    assert payload["min_notional_usdt"] == 5.0
    assert payload["live_orderflow_snapshot_status"] == "live_orderflow_snapshot_degraded"
    assert payload["live_orderflow_snapshot_decision"] == (
        "repair_live_orderflow_snapshot_quality_then_recheck_signal_quality"
    )
    assert payload["micro_quality_ok"] is False
    assert payload["time_sync_ok"] is False
    assert payload["queue_imbalance"] == 0.12
    assert payload["artifacts"]["crypto_shortline_live_orderflow_snapshot"] == str(
        review_dir / "20260316T005018Z_crypto_shortline_live_orderflow_snapshot.json"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_signal_quality_watch"


def test_build_crypto_shortline_signal_quality_watch_becomes_pattern_aware_for_value_rotation(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T005000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T005005Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T005009Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "pattern_status": "value_rotation_scalp_wait_profile_alignment_far",
        },
    )
    _write_json(
        review_dir / "20260316T005010Z_signal_to_order_tickets.json",
        {
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": [
                        "confidence_below_threshold",
                        "convexity_below_threshold",
                    ],
                    "signal": {
                        "confidence": 14.0,
                        "convexity_ratio": 1.4,
                    },
                    "sizing": {
                        "conviction": 0.098,
                        "quote_usdt": 6.2,
                        "min_notional_usdt": 5.0,
                        "risk_budget_usdt": 0.31,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T005015Z_crypto_shortline_ticket_constraint_diagnosis.json",
        {
            "diagnosis_brief": "shortline_signal_quality_blocked:SOLUSDT:improve_shortline_signal_quality_then_recheck_execution_gate:portfolio_margin_um",
            "diagnosis_decision": "improve_shortline_signal_quality_then_recheck_execution_gate",
        },
    )
    _write_json(
        review_dir / "20260316T005018Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:SOLUSDT:recheck_signal_quality_from_live_orderflow:portfolio_margin_um",
            "snapshot_decision": "recheck_signal_quality_from_live_orderflow",
            "micro_quality_ok": True,
            "time_sync_ok": True,
            "queue_imbalance": 0.22,
            "ofi_norm": 0.12,
            "micro_alignment": 0.28,
            "cvd_delta_ratio": 0.21,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T00:50:20Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert (
        payload["watch_status"]
        == "value_rotation_scalp_signal_quality_confidence_convexity_below_threshold"
    )
    assert (
        payload["watch_decision"]
        == "improve_value_rotation_signal_quality_then_recheck_execution_gate"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["pattern_stage"] == "profile_alignment"
    assert payload["artifacts"]["crypto_shortline_pattern_router"] == str(
        review_dir / "20260316T005009Z_crypto_shortline_pattern_router.json"
    )
