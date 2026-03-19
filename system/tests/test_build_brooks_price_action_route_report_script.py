from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from tests.helpers import make_bars


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_brooks_price_action_route_report.py"
)


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


def test_builds_brooks_route_report_from_study_and_gate(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    research_dir = output_root / "research" / "20260313_120000"
    review_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    bars = pd.concat(
        [
            _trend_pullback_frame(),
            _failed_breakout_frame(),
        ],
        ignore_index=True,
    )
    bars.to_csv(research_dir / "bars_used.csv", index=False)

    study_payload = {
        "action": "backtest_brooks_price_action_all_market",
        "ok": True,
        "status": "ok",
        "as_of": "2026-03-13T04:00:00Z",
        "adaptive_route_strategy": {
            "selected_routes_by_asset_class": {
                "crypto": ["trend_pullback_continuation"],
                "equity": ["failed_breakout_reversal"],
            },
            "selection_rows": [
                {
                    "asset_class": "crypto",
                    "strategy_id": "trend_pullback_continuation",
                    "selection_score": 88.0,
                    "selected": True,
                    "metrics": {
                        "trade_count": 20,
                        "expectancy_r": 0.12,
                        "profit_factor": 1.20,
                        "positive_symbol_ratio": 0.50,
                    },
                },
                {
                    "asset_class": "equity",
                    "strategy_id": "failed_breakout_reversal",
                    "selection_score": 72.0,
                    "selected": True,
                    "metrics": {
                        "trade_count": 16,
                        "expectancy_r": 0.08,
                        "profit_factor": 1.10,
                        "positive_symbol_ratio": 0.50,
                    },
                },
            ],
        },
    }
    (review_dir / "20260313T040000Z_brooks_price_action_market_study.json").write_text(
        json.dumps(study_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    gate_payload = {
        "action": "build_crypto_shortline_execution_gate",
        "ok": True,
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "execution_state": "Bias_Only",
                "blocker_detail": "BTCUSDT remains Bias_Only; missing liquidity_sweep, mss, cvd_confirmation.",
                "done_when": "BTCUSDT completes liquidity_sweep and mss and cvd_confirmation",
            }
        ],
    }
    (review_dir / "20260313T040001Z_crypto_shortline_execution_gate.json").write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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
            "--recent-signal-age-bars",
            "3",
            "--now",
            "2026-03-13T04:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["candidate_count"] >= 2
    assert payload["selected_routes_brief"] == "crypto:trend_pullback_continuation | equity:failed_breakout_reversal"
    candidates = {row["symbol"]: row for row in payload["current_candidates"]}
    assert candidates["BTCUSDT"]["route_bridge_status"] == "blocked_shortline_gate"
    assert candidates["300750"]["route_bridge_status"] == "manual_structure_route"
    assert "artifact" in payload and Path(str(payload["artifact"])).exists()
    assert "markdown" in payload and Path(str(payload["markdown"])).exists()
    assert "checksum" in payload and Path(str(payload["checksum"])).exists()
