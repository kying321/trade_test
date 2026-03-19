from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_liquidity_event_trigger.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bars_with_sweep(symbol: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for day in range(1, 9):
        rows.append(
            {
                "ts": f"2026-03-{day:02d}",
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100.0,
                "source": "test",
                "asset_class": "crypto",
            }
        )
    rows.extend(
        [
            {"ts": "2026-03-09", "symbol": symbol, "open": 99.4, "high": 100.8, "low": 95.0, "close": 100.4, "volume": 110.0, "source": "test", "asset_class": "crypto"},
            {"ts": "2026-03-10", "symbol": symbol, "open": 100.5, "high": 102.0, "low": 100.2, "close": 101.8, "volume": 120.0, "source": "test", "asset_class": "crypto"},
            {"ts": "2026-03-11", "symbol": symbol, "open": 102.7, "high": 103.4, "low": 102.5, "close": 103.0, "volume": 200.0, "source": "test", "asset_class": "crypto"},
            {"ts": "2026-03-12", "symbol": symbol, "open": 102.8, "high": 103.6, "low": 102.6, "close": 103.2, "volume": 220.0, "source": "test", "asset_class": "crypto"},
        ]
    )
    return rows


def test_build_crypto_shortline_liquidity_event_trigger_waits_for_new_sweep_when_current_gate_has_none(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                    },
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss.",
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T235959Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
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
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "no_liquidity_sweep_event_observed"
    assert payload["trigger_decision"] == "wait_for_liquidity_sweep_event_then_recheck_execution_gate"
    assert payload["blocker_target_artifact"] == "crypto_shortline_liquidity_event_trigger"
    assert payload["next_action_target_artifact"] == "crypto_shortline_liquidity_event_trigger"
    assert payload["sweep_event_newly_observed"] is False
    assert payload["current_has_sweep"] is False
    assert payload["previous_has_sweep"] is False


def test_build_crypto_shortline_liquidity_event_trigger_detects_new_sweep_against_previous_gate(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["mss"],
                    "structure_signals": {
                        "sweep_long": True,
                        "sweep_short": False,
                    },
                    "blocker_detail": "SOLUSDT cleared sweep but still misses mss.",
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T235959Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
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
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "new_liquidity_sweep_event_detected"
    assert payload["trigger_decision"] == "refresh_shortline_execution_gate_after_liquidity_event"
    assert payload["next_action_target_artifact"] == "crypto_shortline_execution_gate"
    assert payload["sweep_event_newly_observed"] is True
    assert payload["current_has_sweep"] is True
    assert payload["previous_has_sweep"] is False


def test_build_crypto_shortline_liquidity_event_trigger_prefers_live_bars_snapshot_over_stale_gate(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    bars_dir = output_root / "research" / "20260316_000000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    bars_path = bars_dir / "bars_used.csv"
    pd.DataFrame(_bars_with_sweep("SOLUSDT")).to_csv(bars_path, index=False)

    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                    },
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss.",
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T235959Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
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
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "new_liquidity_sweep_event_detected"
    assert payload["trigger_decision"] == "refresh_shortline_execution_gate_after_liquidity_event"
    assert payload["current_signal_source"] == "bars_live_snapshot"
    assert payload["current_signal_source_artifact"].endswith("bars_used.csv")
    assert payload["previous_signal_source"] == "execution_gate_snapshot"
    assert payload["sweep_event_newly_observed"] is True
    assert payload["current_has_sweep"] is True
    assert payload["previous_has_sweep"] is False


def test_build_crypto_shortline_liquidity_event_trigger_prefers_source_owned_live_bars_snapshot(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000015Z_crypto_shortline_live_bars_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "bars_live_snapshot_ready",
            "structure_signals": {
                "sweep_long": True,
                "sweep_short": False,
            },
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                    },
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss.",
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T235959Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
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
            "--output-root",
            str(tmp_path / "output"),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "new_liquidity_sweep_event_detected"
    assert payload["current_signal_source"] == "crypto_shortline_live_bars_snapshot"
    assert payload["current_signal_source_artifact"].endswith(
        "_crypto_shortline_live_bars_snapshot.json"
    )


def test_build_crypto_shortline_liquidity_event_trigger_marks_persistent_pressure_from_ready_orderflow_snapshot(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T235959Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_status": "liquidity_sweep_pending_orderflow_pressure",
            "orderflow_pressure_present": True,
            "orderflow_pressure_side": "long",
            "pressure_persistence_count": 1,
            "current_sweep_long": False,
            "current_sweep_short": False,
        },
    )
    _write_json(
        review_dir / "20260316T000015Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:SOLUSDT:use_live_orderflow_snapshot_for_shortline_signal_quality:portfolio_margin_um",
            "snapshot_decision": "use_live_orderflow_snapshot_for_shortline_signal_quality",
            "micro_quality_ok": True,
            "time_sync_ok": True,
            "queue_imbalance": 0.24,
            "ofi_norm": 0.18,
            "micro_alignment": 0.21,
            "cvd_delta_ratio": 0.11,
            "cvd_attack_presence": "buyers_attacking",
            "cvd_attack_side": "buyers",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "liquidity_sweep_pressure_persisting"
    assert payload["trigger_decision"] == "monitor_persistent_orderflow_pressure_for_liquidity_sweep"
    assert payload["orderflow_pressure_present"] is True
    assert payload["orderflow_pressure_eligible"] is True
    assert payload["pressure_persistence_state"] == "persisting"
    assert payload["pressure_persistence_count"] == 2


def test_build_crypto_shortline_liquidity_event_trigger_marks_pressure_far_from_sweep_band(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000012Z_crypto_shortline_live_bars_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "bars_live_snapshot_ready",
            "structure_signals": {
                "sweep_long": False,
                "sweep_short": False,
            },
            "distance_low_to_sweep_long_bps": 520.0,
            "distance_high_to_sweep_short_bps": 840.0,
            "sweep_long_reference": 77.12,
            "sweep_short_reference": 94.05,
        },
    )
    _write_json(
        review_dir / "20260316T000015Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:SOLUSDT:use_live_orderflow_snapshot_for_shortline_signal_quality:portfolio_margin_um",
            "snapshot_decision": "use_live_orderflow_snapshot_for_shortline_signal_quality",
            "micro_quality_ok": True,
            "time_sync_ok": True,
            "queue_imbalance": 0.24,
            "ofi_norm": 0.18,
            "micro_alignment": 0.21,
            "cvd_delta_ratio": 0.11,
            "cvd_attack_presence": "buyers_attacking",
            "cvd_attack_side": "buyers",
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T235959Z_crypto_shortline_liquidity_event_trigger.json",
        {
            "trigger_status": "liquidity_sweep_pressure_persisting",
            "orderflow_pressure_present": True,
            "orderflow_pressure_side": "long",
            "pressure_persistence_count": 1,
            "current_sweep_long": False,
            "current_sweep_short": False,
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "liquidity_sweep_pressure_persisting_far_from_trigger"
    assert (
        payload["trigger_decision"]
        == "wait_for_price_to_approach_liquidity_sweep_band_then_recheck_execution_gate"
    )
    assert payload["active_sweep_side"] == "long"
    assert payload["active_sweep_distance_bps"] == 520.0
    assert payload["sweep_proximity_state"] == "far"
    assert payload["pressure_persistence_state"] == "persisting"
    assert payload["pressure_persistence_count"] == 2


def test_build_crypto_shortline_liquidity_event_trigger_requires_fresh_live_bars_for_proximity(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000012Z_crypto_shortline_live_bars_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "bars_live_snapshot_stale",
            "snapshot_brief": "bars_live_snapshot_stale:SOLUSDT:refresh_live_bars_snapshot_before_shortline_event_detection:portfolio_margin_um",
            "snapshot_decision": "refresh_live_bars_snapshot_before_shortline_event_detection",
            "blocker_title": "Refresh live bars snapshot before shortline event detection",
            "blocker_target_artifact": "crypto_shortline_live_bars_snapshot",
            "next_action_target_artifact": "crypto_shortline_live_bars_snapshot",
            "done_when": "SOLUSDT latest bar age returns to <= 2 day(s)",
            "latest_bar_fresh": False,
            "latest_bar_age_days": 4,
            "structure_signals": {
                "sweep_long": False,
                "sweep_short": False,
            },
            "distance_low_to_sweep_long_bps": 520.0,
            "distance_high_to_sweep_short_bps": 840.0,
            "sweep_long_reference": 77.12,
            "sweep_short_reference": 94.05,
        },
    )
    _write_json(
        review_dir / "20260316T000015Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_ready",
            "snapshot_brief": "live_orderflow_snapshot_ready:SOLUSDT:use_live_orderflow_snapshot_for_shortline_signal_quality:portfolio_margin_um",
            "snapshot_decision": "use_live_orderflow_snapshot_for_shortline_signal_quality",
            "micro_quality_ok": True,
            "time_sync_ok": True,
            "queue_imbalance": 0.24,
            "ofi_norm": 0.18,
            "micro_alignment": 0.21,
            "cvd_delta_ratio": 0.11,
            "cvd_attack_presence": "buyers_attacking",
            "cvd_attack_side": "buyers",
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
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
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "liquidity_sweep_pressure_bars_snapshot_stale"
    assert payload["trigger_decision"] == "refresh_live_bars_snapshot_then_recheck_execution_gate"
    assert payload["next_action_target_artifact"] == "crypto_shortline_live_bars_snapshot"
    assert payload["live_bars_snapshot_status"] == "bars_live_snapshot_stale"
    assert payload["live_bars_snapshot_fresh"] is False
    assert payload["live_bars_latest_bar_age_days"] == 4


def test_build_crypto_shortline_liquidity_event_trigger_does_not_promote_pressure_when_snapshot_quality_not_ready(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T000015Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_degraded",
            "snapshot_brief": "live_orderflow_snapshot_degraded:SOLUSDT:repair_live_orderflow_snapshot_quality_then_recheck_signal_quality:portfolio_margin_um",
            "snapshot_decision": "repair_live_orderflow_snapshot_quality_then_recheck_signal_quality",
            "micro_quality_ok": False,
            "time_sync_ok": True,
            "queue_imbalance": 0.31,
            "ofi_norm": 0.28,
            "micro_alignment": 0.22,
            "cvd_delta_ratio": 0.17,
            "cvd_attack_presence": "buyers_attacking",
            "cvd_attack_side": "buyers",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["orderflow_pressure_eligible"] is False
    assert payload["orderflow_pressure_present"] is False
    assert payload["trigger_status"] == "no_liquidity_sweep_event_observed"


def test_build_crypto_shortline_liquidity_event_trigger_surfaces_orderflow_pressure_before_sweep(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260316T000000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T000010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T000016Z_crypto_shortline_live_orderflow_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "live_orderflow_snapshot_degraded",
            "snapshot_brief": "live_orderflow_snapshot_degraded:SOLUSDT:repair_live_orderflow_snapshot_quality_then_recheck_signal_quality:portfolio_margin_um",
            "snapshot_decision": "repair_live_orderflow_snapshot_quality_then_recheck_signal_quality",
            "micro_quality_ok": False,
            "time_sync_ok": False,
            "queue_imbalance": 0.03,
            "ofi_norm": 0.22,
            "micro_alignment": 0.14,
            "cvd_delta_ratio": 0.27,
            "cvd_attack_side": "buyers",
            "cvd_attack_presence": "buyers_attacking",
            "cvd_veto_hint": "low_sample_or_gap_risk",
            "cvd_locality_status": "outside_local_window",
        },
    )
    _write_json(
        review_dir / "20260316T000020Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
                    },
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss.",
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T235959Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_state": "watch",
                    "missing_gates": ["liquidity_sweep", "mss"],
                    "structure_signals": {
                        "sweep_long": False,
                        "sweep_short": False,
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
            "--output-root",
            str(tmp_path / "output"),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trigger_status"] == "no_liquidity_sweep_event_observed"
    assert payload["trigger_decision"] == "wait_for_liquidity_sweep_event_then_recheck_execution_gate"
    assert payload["current_signal_source"] == "crypto_shortline_live_orderflow_snapshot"
    assert payload["orderflow_pressure_present"] is False
    assert payload["orderflow_pressure_eligible"] is False
    assert payload["orderflow_pressure_side"] == "long"
    assert payload["orderflow_snapshot_status"] == "live_orderflow_snapshot_degraded"
    assert payload["current_has_sweep"] is False
