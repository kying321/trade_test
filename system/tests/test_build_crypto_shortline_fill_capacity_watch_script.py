from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_fill_capacity_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_fill_capacity_watch_marks_value_rotation_constrained(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T085300Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T085301Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T085302Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "pattern_status": "value_rotation_scalp_wait_profile_alignment_far",
        },
    )
    _write_json(
        review_dir / "20260316T085303Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-16T08:53:03Z",
            "tickets": [
                {
                    "symbol": "SOLUSDT",
                    "allowed": False,
                    "reasons": ["size_below_min_notional"],
                    "sizing": {"quote_usdt": 0.06548},
                    "execution": {"max_slippage_bps": 6.0},
                    "levels": {
                        "entry_price": 84.96,
                        "stop_price": 79.464,
                        "target_price": 95.952,
                    },
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T085304Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:SOLUSDT:monitor_micro_quality:portfolio_margin_um",
            "snapshot_decision": "monitor_micro_quality",
            "micro_quality_ok": True,
            "trust_ok": False,
            "time_sync_ok": True,
            "trade_count": 500,
            "evidence_score": 1.0,
            "cvd_veto_hint": "low_sample_or_gap_risk",
        },
    )
    _write_json(
        review_dir / "20260316T085305Z_crypto_shortline_slippage_snapshot.json",
        {
            "snapshot_status": "value_rotation_scalp_post_cost_degraded",
            "snapshot_brief": "value_rotation_scalp_post_cost_degraded:SOLUSDT:repair_value_rotation_post_cost_then_recheck_execution_gate:portfolio_margin_um",
            "snapshot_decision": "repair_value_rotation_post_cost_then_recheck_execution_gate",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "estimated_entry_cost_bps": 5.0,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:53:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_fill_capacity_constrained"
    assert (
        payload["watch_decision"]
        == "repair_value_rotation_fill_capacity_then_recheck_execution_gate"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["pattern_stage"] == "profile_alignment"
    assert payload["fill_capacity_viable"] is False
    assert payload["entry_headroom_bps"] == 1.0
    assert payload["trust_ok"] is False


def test_build_crypto_shortline_fill_capacity_watch_marks_generic_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T085400Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "BTCUSDT",
            "preferred_route_action": "watch_priority_until_long_window_confirms",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T085401Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "BTCUSDT",
            "next_focus_symbol": "BTCUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
        },
    )
    _write_json(
        review_dir / "20260316T085402Z_signal_to_order_tickets.json",
        {
            "generated_at_utc": "2026-03-16T08:54:02Z",
            "tickets": [
                {
                    "symbol": "BTCUSDT",
                    "allowed": True,
                    "reasons": [],
                    "sizing": {"quote_usdt": 30.0},
                    "execution": {"max_slippage_bps": 4.0},
                    "levels": {
                        "entry_price": 70000.0,
                        "stop_price": 69300.0,
                        "target_price": 71400.0,
                    },
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260316T085403Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "BTCUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:BTCUSDT:continue_monitoring:portfolio_margin_um",
            "snapshot_decision": "continue_monitoring",
            "micro_quality_ok": True,
            "trust_ok": True,
            "time_sync_ok": True,
            "trade_count": 300,
            "evidence_score": 1.0,
            "cvd_veto_hint": "",
        },
    )
    _write_json(
        review_dir / "20260316T085404Z_crypto_shortline_slippage_snapshot.json",
        {
            "snapshot_status": "post_cost_viable",
            "snapshot_brief": "post_cost_viable:BTCUSDT:recheck_execution_gate_after_post_cost_clear:portfolio_margin_um",
            "snapshot_decision": "recheck_execution_gate_after_post_cost_clear",
            "estimated_entry_cost_bps": 2.0,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:54:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "fill_capacity_watch_clear"
    assert payload["watch_decision"] == "recheck_execution_gate_after_fill_capacity_clear"
    assert payload["fill_capacity_viable"] is True
    assert payload["entry_headroom_bps"] == 2.0
