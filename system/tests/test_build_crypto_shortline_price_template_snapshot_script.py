from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_price_template_snapshot.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _bars(symbol: str) -> list[dict[str, object]]:
    return [
        {"ts": "2026-03-10T00:00:00Z", "symbol": symbol, "open": 129, "high": 130, "low": 128.2, "close": 129.8, "volume": 100, "source": "test", "asset_class": "crypto"},
        {"ts": "2026-03-10T00:15:00Z", "symbol": symbol, "open": 129.8, "high": 130.4, "low": 129.1, "close": 130.1, "volume": 102, "source": "test", "asset_class": "crypto"},
        {"ts": "2026-03-10T00:30:00Z", "symbol": symbol, "open": 130.1, "high": 131.1, "low": 129.7, "close": 130.8, "volume": 111, "source": "test", "asset_class": "crypto"},
        {"ts": "2026-03-10T00:45:00Z", "symbol": symbol, "open": 130.8, "high": 131.2, "low": 130.0, "close": 130.5, "volume": 115, "source": "test", "asset_class": "crypto"},
        {"ts": "2026-03-10T01:00:00Z", "symbol": symbol, "open": 130.5, "high": 131.6, "low": 130.1, "close": 131.4, "volume": 120, "source": "test", "asset_class": "crypto"},
        {"ts": "2026-03-10T01:15:00Z", "symbol": symbol, "open": 131.4, "high": 132.0, "low": 130.8, "close": 131.7, "volume": 133, "source": "test", "asset_class": "crypto"},
    ]


def test_build_crypto_shortline_price_template_snapshot_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    bars_dir = output_root / "research" / "20260316_000000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    bars_path = bars_dir / "bars_used.csv"
    pd.DataFrame(_bars("SOLUSDT")).to_csv(bars_path, index=False)

    _write_json(
        review_dir / "20260316T004000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T004005Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T004010Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "route_action": "deprioritize_flow",
                    "route_state": "watch:deprioritize_flow",
                    "micro_signals": {"attack_side": "buyers", "micro_alignment": 0.2},
                }
            ]
        },
    )
    _write_json(
        review_dir / "20260316T004015Z_crypto_shortline_live_bars_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "bars_live_snapshot_ready",
            "bars_artifact": str(bars_path),
            "structure_signals": {"candle_long": True, "candle_short": False},
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
            "2026-03-16T00:50:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["template_status"] == "price_template_snapshot_ready_live_bars"
    assert payload["template_ready"] is True
    assert payload["template_side"] == "LONG"
    assert float(payload["entry_price"]) > 0.0
    assert float(payload["stop_price"]) > 0.0
    assert float(payload["target_price"]) > float(payload["entry_price"])
    assert payload["execution_price_ready"] is True
    assert payload["price_reference_kind"] == "shortline_live_bars_template"


def test_build_crypto_shortline_price_template_snapshot_missing_live_bars(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        review_dir / "20260316T004000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T004005Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T004010Z_crypto_shortline_execution_gate.json",
        {"symbols": [{"symbol": "SOLUSDT", "micro_signals": {"attack_side": "buyers"}}]},
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
            "2026-03-16T00:50:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["template_status"] == "price_template_snapshot_missing_live_bars"
    assert payload["template_ready"] is False
    assert payload["template_decision"] == "collect_live_bars_snapshot_before_building_price_template"


def test_build_crypto_shortline_price_template_snapshot_builds_symbol_side_templates(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    bars_dir = output_root / "research" / "20260316_000000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    bars_path = bars_dir / "bars_used.csv"
    pd.DataFrame(_bars("SOLUSDT") + _bars("BNBUSDT")).to_csv(bars_path, index=False)

    _write_json(
        review_dir / "20260316T004000Z_remote_intent_queue.json",
        {
            "preferred_route_symbol": "SOLUSDT",
            "preferred_route_action": "deprioritize_flow",
            "remote_market": "portfolio_margin_um",
        },
    )
    _write_json(
        review_dir / "20260316T004005Z_crypto_route_operator_brief.json",
        {
            "review_priority_head_symbol": "SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
        },
    )
    _write_json(
        review_dir / "20260316T004010Z_crypto_shortline_execution_gate.json",
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "micro_signals": {"attack_side": "buyers", "micro_alignment": 0.2},
                },
                {
                    "symbol": "BNBUSDT",
                    "micro_signals": {"attack_side": "sellers", "micro_alignment": -0.2},
                },
            ]
        },
    )
    _write_json(
        review_dir / "20260316T004015Z_crypto_shortline_live_bars_snapshot.json",
        {
            "route_symbol": "SOLUSDT",
            "snapshot_status": "bars_live_snapshot_ready",
            "bars_artifact": str(bars_path),
            "materialized_symbols": ["SOLUSDT", "BNBUSDT"],
            "structure_signals": {"candle_long": True, "candle_short": False},
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
            "2026-03-16T00:50:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["template_symbol_count"] == 2
    assert payload["templates"]["SOLUSDT"]["LONG"]["template_ready"] is True
    assert payload["templates"]["SOLUSDT"]["SHORT"]["template_ready"] is True
    assert payload["templates"]["BNBUSDT"]["LONG"]["template_ready"] is True
    assert payload["templates"]["BNBUSDT"]["SHORT"]["template_ready"] is True
