from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_execution_quality_watch.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_execution_quality_watch_marks_value_rotation_degraded(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T082000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T082001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T082002Z_crypto_shortline_pattern_router.json",
        {
            "route_symbol": "SOLUSDT",
            "pattern_family": "value_rotation_scalp",
            "pattern_stage": "profile_alignment",
            "pattern_status": "value_rotation_scalp_wait_profile_alignment_far",
        },
    )
    _write_json(
        review_dir / "20260316T082003Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:SOLUSDT:use_live_orderflow_snapshot_for_shortline_signal_quality:portfolio_margin_um",
            "snapshot_decision": "use_live_orderflow_snapshot_for_shortline_signal_quality",
            "micro_quality_ok": True,
            "trust_ok": False,
            "time_sync_ok": True,
            "queue_imbalance": 0.03,
            "ofi_norm": 0.26,
            "micro_alignment": 0.14,
            "cvd_delta_ratio": 0.26,
            "trade_count": 500,
            "evidence_score": 1.0,
            "cvd_veto_hint": "low_sample_or_gap_risk",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:20:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "value_rotation_scalp_execution_quality_degraded"
    assert (
        payload["watch_decision"]
        == "repair_value_rotation_execution_quality_then_recheck_execution_gate"
    )
    assert payload["pattern_family"] == "value_rotation_scalp"
    assert payload["pattern_stage"] == "profile_alignment"
    assert payload["trust_ok"] is False
    assert payload["execution_quality_score"] >= 55.0


def test_build_crypto_shortline_execution_quality_watch_marks_generic_clear(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T082000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "BTCUSDT",
            "preferred_route_action": "watch_priority_until_long_window_confirms",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T082001Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "BTCUSDT",
            "next_focus_symbol": "BTCUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
        },
    )
    _write_json(
        review_dir / "20260316T082003Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "BTCUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:BTCUSDT:use_live_orderflow_snapshot_for_shortline_signal_quality:portfolio_margin_um",
            "snapshot_decision": "use_live_orderflow_snapshot_for_shortline_signal_quality",
            "micro_quality_ok": True,
            "trust_ok": True,
            "time_sync_ok": True,
            "queue_imbalance": 0.21,
            "ofi_norm": 0.22,
            "micro_alignment": 0.19,
            "cvd_delta_ratio": 0.24,
            "trade_count": 300,
            "evidence_score": 1.0,
            "cvd_veto_hint": "",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-16T08:20:10Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["watch_status"] == "execution_quality_watch_clear"
    assert (
        payload["watch_decision"]
        == "recheck_execution_gate_after_execution_quality_clear"
    )
    assert payload["execution_quality_score"] >= 55.0
