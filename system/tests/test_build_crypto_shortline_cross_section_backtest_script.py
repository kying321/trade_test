from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_shortline_cross_section_backtest.py"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fixture_bars() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    specs = {
        "SOLUSDT": {"base": 100.0, "drift": 0.12, "event_boost": 1.6, "follow": 0.55},
        "BTCUSDT": {"base": 200.0, "drift": 0.02, "event_boost": 0.8, "follow": -0.20},
        "ETHUSDT": {"base": 150.0, "drift": 0.01, "event_boost": 0.3, "follow": -0.05},
        "BNBUSDT": {"base": 120.0, "drift": 0.04, "event_boost": 0.5, "follow": 0.05},
    }
    for symbol, spec in specs.items():
        close = float(spec["base"])
        for index in range(96):
            ts = start + timedelta(minutes=15 * index)
            open_price = close
            high = open_price + 0.30
            low = open_price - 0.30
            close = open_price + float(spec["drift"])
            volume = 1000.0 + index * 3.0
            if index in {20, 44, 68}:
                high = open_price + float(spec["event_boost"]) + 0.4
                low = open_price - 0.2
                close = open_price + float(spec["event_boost"])
                volume = 2400.0
            elif index - 1 in {20, 44, 68}:
                close = open_price + float(spec["follow"])
                high = max(high, close + 0.1)
                low = min(low, close - 0.1)
                volume = 1500.0
            rows.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "symbol": symbol,
                    "open": round(open_price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(close, 4),
                    "volume": round(volume, 4),
                }
            )
    return rows


def test_build_crypto_shortline_cross_section_backtest_detects_positive_watch_only_slice(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    bars_fixture = tmp_path / "bars_fixture.json"
    _write_json(bars_fixture, {"bars": _fixture_bars()})
    _write_json(
        review_dir / "20260315T103900Z_crypto_shortline_backtest_slice.json",
        {
            "slice_brief": "selected_watch_only_orderflow_slice:SOLUSDT:crypto_hot:4h->15m",
            "slice_status": "selected_watch_only_orderflow_slice",
            "research_decision": "run_watch_only_orderflow_cross_section_backtest",
            "selected_symbol": "SOLUSDT",
            "slice_universe": ["SOLUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "slice_universe_brief": "SOLUSDT,BTCUSDT,ETHUSDT,BNBUSDT",
            "structure_timeframe": "4h",
            "execution_timeframe": "15m",
            "holding_window_minutes": {"min": 15, "max": 180},
            "blocker_detail": "watch_only until sweep+mss+cvd align",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--bars-fixture",
            str(bars_fixture),
            "--lookback-days",
            "7",
            "--now",
            "2026-03-02T10:40:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["backtest_status"] == "watch_only_cross_section_positive"
    assert payload["research_decision"] == "continue_shadow_learning_wait_for_setup_ready"
    assert payload["selected_symbol"] == "SOLUSDT"
    assert payload["selected_edge_status"] == "positive"
    assert payload["selected_summary"]["event_count"] >= 3
    assert payload["selected_summary"]["best_expectancy_bps"] > 0
    assert payload["selected_rank_in_universe"] == 1
    assert payload["event_sample_count"] >= payload["selected_summary"]["event_count"]
    assert payload["fetch_errors"] == {}
