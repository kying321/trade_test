from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from tests.helpers import make_bars


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backtest_brooks_price_action_all_market.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("brooks_price_action_market_study_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _trend_pullback_frame() -> pd.DataFrame:
    frame = make_bars("BTCUSDT", n=160, trend=0.16, seed=17, asset_class="crypto").reset_index(drop=True)
    anchor = float(frame.loc[154, "close"])
    custom = [
        (anchor * 1.000, anchor * 1.028, anchor * 0.996, anchor * 1.024, 1.6),
        (anchor * 1.022, anchor * 1.026, anchor * 1.000, anchor * 1.006, 1.3),
        (anchor * 1.005, anchor * 1.010, anchor * 0.988, anchor * 0.995, 1.2),
        (anchor * 0.996, anchor * 1.020, anchor * 0.986, anchor * 1.016, 1.9),
        (anchor * 1.017, anchor * 1.026, anchor * 1.010, anchor * 1.022, 1.5),
    ]
    start = len(frame) - len(custom)
    vol_ref = float(frame["volume"].tail(20).mean())
    for offset, (open_px, high_px, low_px, close_px, vol_mult) in enumerate(custom):
        idx = start + offset
        frame.loc[idx, ["open", "high", "low", "close", "volume"]] = [
            open_px,
            high_px,
            low_px,
            close_px,
            vol_ref * vol_mult,
        ]
    return frame


def _failed_breakout_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-01", periods=180)
    base = 100.0
    rows = []
    for i, ts in enumerate(idx):
        px = base + ((i % 6) - 3) * 0.22
        rows.append(
            {
                "ts": ts,
                "symbol": "300750",
                "open": px - 0.05,
                "high": px + 0.35,
                "low": px - 0.35,
                "close": px + 0.05,
                "volume": 8_000_000,
                "source": "test",
                "asset_class": "equity",
            }
        )
    frame = pd.DataFrame(rows)
    signal_idx = len(frame) - 2
    frame.loc[signal_idx, "open"] = 99.35
    frame.loc[signal_idx, "high"] = 100.10
    frame.loc[signal_idx, "low"] = 98.55
    frame.loc[signal_idx, "close"] = 99.95
    frame.loc[signal_idx, "volume"] = 10_500_000
    next_idx = len(frame) - 1
    frame.loc[next_idx, "open"] = 100.00
    frame.loc[next_idx, "high"] = 100.80
    frame.loc[next_idx, "low"] = 99.90
    frame.loc[next_idx, "close"] = 100.60
    frame.loc[next_idx, "volume"] = 10_000_000
    return frame


def _second_entry_frame() -> pd.DataFrame:
    frame = make_bars("ETHUSDT", n=170, trend=0.14, seed=23, asset_class="crypto").reset_index(drop=True)
    anchor = float(frame.loc[162, "close"])
    vol_ref = float(frame["volume"].tail(20).mean())
    custom = [
        (anchor * 1.010, anchor * 1.024, anchor * 1.004, anchor * 1.020, 1.5),
        (anchor * 1.018, anchor * 1.020, anchor * 0.996, anchor * 1.000, 1.2),
        (anchor * 1.001, anchor * 1.017, anchor * 0.994, anchor * 1.014, 1.6),
        (anchor * 1.013, anchor * 1.016, anchor * 0.998, anchor * 1.002, 1.3),
        (anchor * 1.003, anchor * 1.022, anchor * 0.995, anchor * 1.018, 1.8),
        (anchor * 1.019, anchor * 1.028, anchor * 1.012, anchor * 1.025, 1.4),
    ]
    start = len(frame) - len(custom)
    for offset, (open_px, high_px, low_px, close_px, vol_mult) in enumerate(custom):
        idx = start + offset
        frame.loc[idx, ["open", "high", "low", "close", "volume"]] = [
            open_px,
            high_px,
            low_px,
            close_px,
            vol_ref * vol_mult,
        ]
    return frame


def test_generate_symbol_signals_detects_trend_pullback_long() -> None:
    module = _load_module()
    frame = module.add_price_action_features(_trend_pullback_frame())
    signals = module.generate_symbol_signals(frame)
    trend_signals = signals["trend_pullback_continuation"]
    assert trend_signals, "expected at least one trend pullback signal"
    assert any(row["direction"] == "LONG" for row in trend_signals)


def test_generate_symbol_signals_detects_failed_breakout_reversal_long() -> None:
    module = _load_module()
    frame = module.add_price_action_features(_failed_breakout_frame())
    signals = module.generate_symbol_signals(frame)
    reversal_signals = signals["failed_breakout_reversal"]
    assert reversal_signals, "expected a failed breakout reversal signal"
    assert any(row["direction"] == "LONG" for row in reversal_signals)


def test_generate_symbol_signals_detects_second_entry_long() -> None:
    module = _load_module()
    frame = module.add_price_action_features(_second_entry_frame())
    signals = module.generate_symbol_signals(frame)
    second_entry_signals = signals["second_entry_trend_continuation"]
    assert second_entry_signals, "expected a second-entry continuation signal"
    assert any(row["direction"] == "LONG" for row in second_entry_signals)


def test_script_builds_brooks_market_study_from_local_bars(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    research_a = output_root / "research" / "20260313_100000"
    research_b = output_root / "research" / "20260313_100100"
    research_a.mkdir(parents=True, exist_ok=True)
    research_b.mkdir(parents=True, exist_ok=True)

    short_btc = make_bars("BTCUSDT", n=80, trend=0.12, seed=31, asset_class="crypto")
    short_btc.to_csv(research_a / "bars_used.csv", index=False)

    frames = pd.concat(
        [
            _trend_pullback_frame(),
            _failed_breakout_frame(),
            make_bars("ETHUSDT", n=160, trend=-0.08, seed=44, asset_class="crypto"),
        ],
        ignore_index=True,
    )
    frames.to_csv(research_b / "bars_used.csv", index=False)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--output-root",
            str(output_root),
            "--review-dir",
            str(review_dir),
            "--min-bars",
            "120",
            "--now",
            "2026-03-13T10:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["study_status"] == "complete"
    assert payload["coverage_summary"]["included_symbol_count"] >= 3
    assert payload["coverage_summary"]["coverage_by_asset_class"]["crypto"] >= 2
    assert payload["coverage_summary"]["coverage_by_asset_class"]["equity"] >= 1
    assert any(row["strategy_id"] == "trend_pullback_continuation" for row in payload["strategy_modules"])
    assert any(row["strategy_id"] == "second_entry_trend_continuation" for row in payload["strategy_modules"])
    assert payload["adaptive_route_strategy"]["selection_mode"] == "train_first_two_folds_trade_count_split"
    assert "artifact" in payload and Path(str(payload["artifact"])).exists()
    assert "markdown" in payload and Path(str(payload["markdown"])).exists()
    assert "checksum" in payload and Path(str(payload["checksum"])).exists()
