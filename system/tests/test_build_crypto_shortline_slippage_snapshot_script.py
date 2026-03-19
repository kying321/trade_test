from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_slippage_snapshot.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_slippage_snapshot_marks_value_rotation_degraded(
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
        review_dir / "20260316T084002Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "pattern_status": "value_rotation_scalp_wait_profile_alignment_far",
        },
    )
    _write_json(
        review_dir / "20260316T084003Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "micro_quality_ok": True,
            "trust_ok": False,
            "time_sync_ok": True,
            "queue_imbalance": 0.03,
            "ofi_norm": 0.26,
            "micro_alignment": 0.14,
            "trade_count": 500,
            "evidence_score": 1.0,
            "cvd_veto_hint": "low_sample_or_gap_risk",
        },
    )
    _write_json(
        review_dir / "20260316T084004Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-16T08:40:04Z",
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": ["confidence_below_threshold", "size_below_min_notional"],
                    "levels": {
                        "entry_price": 84.96,
                        "stop_price": 79.464,
                        "target_price": 95.952,
                    },
                    "sizing": {
                        "quote_usdt": 0.06548524035594559,
                    },
                    "execution": {
                        "order_type_hint": "LIMIT",
                        "max_slippage_bps": 6.0,
                    },
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
            "--now",
            "2026-03-16T08:40:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "value_rotation_scalp_post_cost_degraded"
    assert (
        payload["snapshot_decision"]
        == "repair_value_rotation_post_cost_then_recheck_execution_gate"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["pattern_stage"] == "profile_alignment"
    assert payload["post_cost_viable"] is False
    assert payload["trust_ok"] is False
    assert payload["estimated_roundtrip_cost_bps"] > 0.0


def test_build_crypto_shortline_slippage_snapshot_marks_generic_post_cost_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T084100Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "BTCUSDT",
            "preferred_route_action": "watch_priority_until_long_window_confirms",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T084101Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "BTCUSDT",
            "next_focus_symbol": "BTCUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
        },
    )
    _write_json(
        review_dir / "20260316T084102Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "BTCUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "micro_quality_ok": True,
            "trust_ok": True,
            "time_sync_ok": True,
            "queue_imbalance": 0.21,
            "ofi_norm": 0.22,
            "micro_alignment": 0.19,
            "trade_count": 300,
            "evidence_score": 1.0,
            "cvd_veto_hint": "",
        },
    )
    _write_json(
        review_dir / "20260316T084103Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-16T08:41:03Z",
            "tickets": [
                {
                    "symbol": "BTCUSDT",
                    "allowed": True,
                    "reasons": [],
                    "levels": {
                        "entry_price": 70000.0,
                        "stop_price": 69300.0,
                        "target_price": 71400.0,
                    },
                    "sizing": {
                        "quote_usdt": 30.0,
                    },
                    "execution": {
                        "order_type_hint": "LIMIT",
                        "max_slippage_bps": 4.0,
                    },
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
            "--now",
            "2026-03-16T08:41:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "post_cost_viable"
    assert payload["snapshot_decision"] == "recheck_execution_gate_after_post_cost_clear"
    assert payload["post_cost_viable"] is True
    assert payload["estimated_roundtrip_cost_bps"] > 0.0
    assert payload["cost_to_target_ratio"] is not None
