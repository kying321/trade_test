from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_signal_source.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_build_crypto_shortline_signal_source_generates_fresh_rows(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    _write_json(
        review_dir / "20260315T112000Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "watch",
                    "route_state": "watch:deprioritize_flow",
                    "missing_gates": ["mss", "cvd_confirmation"],
                    "blocker_detail": "btc-bias-only",
                    "micro_signals": {
                        "attack_side": "buyers",
                        "micro_alignment": 0.42,
                        "evidence_score": 1.4,
                        "trade_count": 620,
                        "context": "continuation",
                        "veto_hint": "low_sample_or_gap_risk",
                    },
                },
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch:deprioritize_flow",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                    "blocker_detail": "sol-bias-only",
                    "micro_signals": {
                        "attack_side": "sellers",
                        "micro_alignment": -0.25,
                        "evidence_score": 1.1,
                        "trade_count": 210,
                        "context": "retest",
                        "veto_hint": "no_trade",
                    },
                },
            ]
        },
    )
    _write_json(
        review_dir / "20260315T112010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T112020Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )
    _write_json(
        review_dir / "20260314T200000Z_strategy_recent5_signals.json",
        {
            "signals": {
                "BTCUSDT": [
                    {
                        "date": "2026-03-14",
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "entry_price": 70000.0,
                        "stop_price": 69700.0,
                        "target_price": 70600.0,
                        "convexity_ratio": 3.4,
                    }
                ]
            }
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
            "--symbols",
            "BTCUSDT,SOLUSDT",
            "--now",
            "2026-03-15T11:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    meta = json.loads(proc.stdout)
    artifact = Path(str(meta["artifact"]))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["source_kind"] == "crypto_shortline_signal_source"
    assert payload["selection_reason"] == "current_shortline_gate_projection"
    assert payload["route_focus_symbol"] == "SOLUSDT"
    assert payload["signals"]["BTCUSDT"][0]["date"] == "2026-03-15"
    assert payload["signals"]["BTCUSDT"][0]["price_reference_kind"] == "shortline_template_proxy"
    assert payload["signals"]["BTCUSDT"][0]["execution_price_ready"] is False
    assert payload["signals"]["BTCUSDT"][0]["execution_state"] == "Bias_Only"
    assert payload["signals"]["BTCUSDT"][0]["route_action"] == "watch"
    assert payload["signals"]["BTCUSDT"][0]["route_state"] == "watch:deprioritize_flow"
    assert payload["signals"]["BTCUSDT"][0]["missing_gates"] == ["mss", "cvd_confirmation"]
    assert payload["signals"]["SOLUSDT"][0]["side"] == "SHORT"
    assert payload["signals"]["SOLUSDT"][0]["price_reference_kind"] == "shortline_missing_price_template"
    assert payload["summary"]["signal_row_count"] == 2


def test_build_crypto_shortline_signal_source_uses_live_price_template_snapshot(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    _write_json(
        review_dir / "20260315T112000Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch:deprioritize_flow",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                    "blocker_detail": "sol-bias-only",
                    "micro_signals": {
                        "attack_side": "buyers",
                        "micro_alignment": 0.31,
                        "evidence_score": 1.6,
                        "trade_count": 260,
                        "context": "retest",
                        "veto_hint": "low_sample_or_gap_risk",
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T112010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T112020Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )
    _write_json(
        review_dir / "20260315T112030Z_crypto_shortline_price_template_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "template_side": "LONG",
            "template_ready": True,
            "entry_price": 132.5,
            "stop_price": 130.9,
            "target_price": 135.7,
            "artifact": str(review_dir / "20260315T112030Z_crypto_shortline_price_template_snapshot.json"),
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
            "--symbols",
            "SOLUSDT",
            "--now",
            "2026-03-15T11:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    meta = json.loads(proc.stdout)
    artifact = Path(str(meta["artifact"]))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    row = payload["signals"]["SOLUSDT"][0]
    assert row["side"] == "LONG"
    assert row["price_reference_kind"] == "shortline_live_bars_template"
    assert row["execution_price_ready"] is True
    assert row["entry_price"] == 132.5
    assert row["stop_price"] == 130.9
    assert row["target_price"] == 135.7
    assert payload["summary"]["live_price_template_count"] == 1


def test_build_crypto_shortline_signal_source_prefers_live_template_over_stale_proxy(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    _write_json(
        review_dir / "20260315T112000Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch:deprioritize_flow",
                    "missing_gates": ["liquidity_sweep", "mss", "cvd_confirmation"],
                    "blocker_detail": "sol-bias-only",
                    "micro_signals": {
                        "attack_side": "buyers",
                        "micro_alignment": 0.31,
                        "evidence_score": 1.6,
                        "trade_count": 260,
                        "context": "retest",
                        "veto_hint": "low_sample_or_gap_risk",
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T112010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260315T112020Z_crypto_cvd_queue_handoff.json",
        {
            "queue_status": "derived",
            "next_focus_action": "defer_until_micro_recovers",
        },
    )
    _write_json(
        review_dir / "20260314T200000Z_strategy_recent5_signals.json",
        {
            "signals": {
                "SOLUSDT": [
                    {
                        "date": "2026-03-14",
                        "symbol": "SOLUSDT",
                        "side": "LONG",
                        "entry_price": 84.96,
                        "stop_price": 79.464,
                        "target_price": 95.952,
                        "convexity_ratio": 2.5,
                    }
                ]
            }
        },
    )
    live_template_path = review_dir / "20260315T112030Z_crypto_shortline_price_template_snapshot.json"
    _write_json(
        live_template_path,
        {
            "route_symbol": "SOLUSDT",
            "template_side": "LONG",
            "template_ready": True,
            "entry_price": 132.5,
            "stop_price": 130.9,
            "target_price": 135.7,
            "artifact": str(live_template_path),
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
            "--symbols",
            "SOLUSDT",
            "--now",
            "2026-03-15T11:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    meta = json.loads(proc.stdout)
    artifact = Path(str(meta["artifact"]))
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    row = payload["signals"]["SOLUSDT"][0]
    assert row["price_reference_kind"] == "shortline_live_bars_template"
    assert row["execution_price_ready"] is True
    assert row["entry_price"] == 132.5
    assert row["stop_price"] == 130.9
    assert row["target_price"] == 135.7
    assert row["price_reference_source"] == str(live_template_path)
    assert "price_template=20260315T112030Z_crypto_shortline_price_template_snapshot.json" in row["notes"]


def test_build_crypto_shortline_signal_source_uses_symbol_side_live_template_snapshot(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    live_template_path = review_dir / "20260315T112030Z_crypto_shortline_price_template_snapshot.json"
    _write_json(
        review_dir / "20260315T112000Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "BNBUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "watch",
                    "route_state": "watch:deprioritize_flow",
                    "micro_signals": {
                        "attack_side": "buyers",
                        "micro_alignment": 0.22,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T112010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(review_dir / "20260315T112020Z_crypto_cvd_queue_handoff.json", {"queue_status": "derived"})
    _write_json(
        live_template_path,
        {
            "route_symbol": "SOLUSDT",
            "template_side": "LONG",
            "template_ready": True,
            "entry_price": 132.5,
            "stop_price": 130.9,
            "target_price": 135.7,
            "templates": {
                "BNBUSDT": {
                    "LONG": {
                        "template_ready": True,
                        "entry_price": 610.5,
                        "stop_price": 600.0,
                        "target_price": 631.5,
                    }
                }
            },
            "artifact": str(live_template_path),
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
            "--symbols",
            "BNBUSDT",
            "--now",
            "2026-03-15T11:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    meta = json.loads(proc.stdout)
    payload = json.loads(Path(str(meta["artifact"])).read_text(encoding="utf-8"))
    row = payload["signals"]["BNBUSDT"][0]
    assert row["side"] == "LONG"
    assert row["price_reference_kind"] == "shortline_live_bars_template"
    assert row["execution_price_ready"] is True
    assert row["entry_price"] == 610.5
    assert row["price_reference_source"] == str(live_template_path)


def test_build_crypto_shortline_signal_source_uses_short_side_live_template_snapshot(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    live_template_path = review_dir / "20260315T112030Z_crypto_shortline_price_template_snapshot.json"
    _write_json(
        review_dir / "20260315T112000Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch:deprioritize_flow",
                    "micro_signals": {
                        "attack_side": "sellers",
                        "micro_alignment": -0.25,
                    },
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260315T112010Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(review_dir / "20260315T112020Z_crypto_cvd_queue_handoff.json", {"queue_status": "derived"})
    _write_json(
        live_template_path,
        {
            "route_symbol": "SOLUSDT",
            "template_side": "LONG",
            "template_ready": True,
            "entry_price": 132.5,
            "stop_price": 130.9,
            "target_price": 135.7,
            "templates": {
                "SOLUSDT": {
                    "SHORT": {
                        "template_ready": True,
                        "entry_price": 131.8,
                        "stop_price": 133.4,
                        "target_price": 128.6,
                    }
                }
            },
            "artifact": str(live_template_path),
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
            "--symbols",
            "SOLUSDT",
            "--now",
            "2026-03-15T11:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    meta = json.loads(proc.stdout)
    payload = json.loads(Path(str(meta["artifact"])).read_text(encoding="utf-8"))
    row = payload["signals"]["SOLUSDT"][0]
    assert row["side"] == "SHORT"
    assert row["price_reference_kind"] == "shortline_live_bars_template"
    assert row["execution_price_ready"] is True
    assert row["entry_price"] == 131.8
    assert row["stop_price"] == 133.4
    assert row["target_price"] == 128.6
