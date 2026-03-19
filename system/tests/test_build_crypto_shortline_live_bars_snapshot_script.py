from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_live_bars_snapshot.py"
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


def test_build_crypto_shortline_live_bars_snapshot_ready(tmp_path: Path) -> None:
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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--disable-network-refresh",
            "--now",
            "2026-03-12T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "bars_live_snapshot_ready"
    assert payload["snapshot_decision"] == "use_live_bars_snapshot_for_shortline_event_detection"
    assert payload["signal_source"] == "crypto_shortline_live_bars_snapshot"
    assert payload["bars_artifact"].endswith("_crypto_shortline_live_bars_snapshot_bars.csv")
    assert Path(payload["bars_artifact"]).exists()
    assert payload["bars_input_kind"] == "local_bars_file"
    assert payload["structure_signals"]["sweep_long"] is True
    assert payload["sweep_long_reference"] is not None
    assert payload["sweep_short_reference"] is not None
    assert payload["distance_low_to_sweep_long_bps"] is not None
    assert payload["distance_high_to_sweep_short_bps"] is not None
    assert payload["blocker_target_artifact"] == "crypto_shortline_live_bars_snapshot"


def test_build_crypto_shortline_live_bars_snapshot_missing_bars(tmp_path: Path) -> None:
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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--disable-network-refresh",
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "bars_live_snapshot_missing"
    assert payload["snapshot_decision"] == (
        "collect_live_bars_snapshot_before_shortline_event_detection"
    )
    assert payload["blocker_target_artifact"] == "crypto_shortline_live_bars_snapshot"


def test_build_crypto_shortline_live_bars_snapshot_marks_stale_latest_bar(
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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--disable-network-refresh",
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "bars_live_snapshot_stale"
    assert (
        payload["snapshot_decision"]
        == "refresh_live_bars_snapshot_before_shortline_event_detection"
    )
    assert payload["latest_bar_age_days"] == 4
    assert payload["latest_bar_fresh"] is False
    assert payload["max_bar_age_days"] == 2


def test_build_crypto_shortline_live_bars_snapshot_uses_live_fixture_when_local_bars_are_stale(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    bars_dir = output_root / "research" / "20260316_000000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    local_bars_path = bars_dir / "bars_used.csv"
    pd.DataFrame(_bars_with_sweep("SOLUSDT")).to_csv(local_bars_path, index=False)

    live_fixture = tmp_path / "live_bars.csv"
    pd.DataFrame(
        [
            {
                "ts": "2026-03-15T22:00:00Z",
                "symbol": "SOLUSDT",
                "open": 84.0,
                "high": 85.0,
                "low": 83.0,
                "close": 84.5,
                "volume": 150.0,
                "source": "fixture",
                "asset_class": "crypto",
            },
            {
                "ts": "2026-03-16T00:00:00Z",
                "symbol": "SOLUSDT",
                "open": 84.6,
                "high": 86.1,
                "low": 84.2,
                "close": 85.8,
                "volume": 180.0,
                "source": "fixture",
                "asset_class": "crypto",
            },
            {
                "ts": "2026-03-16T00:00:00Z",
                "symbol": "BNBUSDT",
                "open": 610.0,
                "high": 613.0,
                "low": 608.0,
                "close": 612.0,
                "volume": 95.0,
                "source": "fixture",
                "asset_class": "crypto",
            },
        ]
    ).to_csv(live_fixture, index=False)

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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--live-bars-file",
            str(live_fixture),
            "--now",
            "2026-03-16T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["snapshot_status"] == "bars_live_snapshot_ready"
    assert payload["bars_input_kind"] == "live_bars_fixture"
    assert payload["bars_refresh_attempted"] is True
    assert payload["bars_refresh_status"] == "live_refresh_used"
    assert payload["latest_bar_age_days"] == 0
    assert payload["latest_bar_fresh"] is True
    assert Path(payload["bars_artifact"]).exists()
    assert payload["materialized_symbol_count"] == 2
    assert payload["materialized_symbols"] == ["BNBUSDT", "SOLUSDT"]


def test_build_crypto_shortline_live_bars_snapshot_materializes_supported_symbols(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    bars_dir = output_root / "research" / "20260316_000000"
    bars_dir.mkdir(parents=True, exist_ok=True)
    bars_path = bars_dir / "bars_used.csv"
    pd.DataFrame(_bars_with_sweep("SOLUSDT") + _bars_with_sweep("BNBUSDT")).to_csv(
        bars_path, index=False
    )

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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--disable-network-refresh",
            "--now",
            "2026-03-12T00:01:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["materialized_symbol_count"] == 2
    assert payload["materialized_symbols"] == ["BNBUSDT", "SOLUSDT"]
    materialized = pd.read_csv(Path(payload["bars_artifact"]))
    assert sorted(materialized["symbol"].astype(str).unique().tolist()) == ["BNBUSDT", "SOLUSDT"]
