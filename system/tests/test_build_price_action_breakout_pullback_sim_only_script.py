from __future__ import annotations

import json
import importlib.util
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_sim_only.py"
)
SPEC = importlib.util.spec_from_file_location("price_action_breakout_pullback_sim_only", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _build_fixture_rows() -> list[dict[str, object]]:
    start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    breakout_bars = {90, 91, 92, 160, 161, 162, 230, 231, 232}
    pullback_bars = {93, 94, 163, 164, 233, 234}
    reclaim_bars = {95, 165, 235}
    specs = {
        "BTCUSDT": {"base": 90000.0, "drift": 0.0002, "vol": 1200.0},
        "ETHUSDT": {"base": 2200.0, "drift": 0.00015, "vol": 1000.0},
        "BNBUSDT": {"base": 620.0, "drift": 0.0001, "vol": 900.0},
        "SOLUSDT": {"base": 140.0, "drift": 0.00035, "vol": 1500.0},
    }
    rows: list[dict[str, object]] = []
    for symbol, spec in specs.items():
        close = float(spec["base"])
        for idx in range(320):
            ts = start + timedelta(minutes=15 * idx)
            open_price = close
            step = float(spec["drift"]) * (1.0 + ((idx % 12) - 6) * 0.05)
            if symbol == "SOLUSDT" and idx in breakout_bars:
                step += 0.0035
            elif symbol == "SOLUSDT" and idx in pullback_bars:
                step -= 0.0021
            elif symbol == "SOLUSDT" and idx in reclaim_bars:
                step += 0.0032
            close = max(1.0, open_price * (1.0 + step))
            high = max(open_price, close) * 1.0018
            low = min(open_price, close) * 0.9985
            volume = float(spec["vol"]) * (1.0 + (idx % 6) * 0.12)
            if symbol == "SOLUSDT" and idx in breakout_bars:
                volume *= 2.6
            elif symbol == "SOLUSDT" and idx in pullback_bars:
                low *= 0.9972
                volume *= 1.7
            elif symbol == "SOLUSDT" and idx in reclaim_bars:
                high *= 1.0018
                volume *= 2.2
            rows.append(
                {
                    "ts": ts.isoformat().replace("+00:00", "Z"),
                    "symbol": symbol,
                    "open": round(open_price, 6),
                    "high": round(high, 6),
                    "low": round(low, 6),
                    "close": round(close, 6),
                    "volume": round(volume, 6),
                    "asset_class": "crypto",
                    "source": "fixture",
                    "interval": "15m",
                }
            )
    return rows


def test_build_price_action_breakout_pullback_sim_only_runs_with_fixture(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = review_dir / "20260319T000000Z_public_intraday_crypto_bars_dataset.csv"
    pd.DataFrame(_build_fixture_rows()).to_csv(dataset_path, index=False)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--dataset-path",
            str(dataset_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260319T110500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload_out = json.loads(proc.stdout)
    json_path = Path(payload_out["json_path"])
    md_path = Path(payload_out["md_path"])
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["action"] == "build_price_action_breakout_pullback_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["symbol_count"] == 4
    assert payload["cadence_minutes"] == 15
    assert payload["focus_symbol"]
    assert payload["selected_params"]
    assert "structure_filter" in payload["selected_params"]
    assert "vpvr_filter" in payload["selected_params"]
    assert "vpvr_max_distance_atr" in payload["selected_params"]
    assert "daily_stop_r" in payload["selected_params"]
    assert payload["validation_status"]
    assert len(payload["symbol_results"]) == 4
    assert len(payload["ranking"]) == 4


def test_build_price_action_breakout_pullback_sim_only_supports_symbol_filter(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = review_dir / "20260319T000000Z_public_intraday_crypto_bars_dataset.csv"
    pd.DataFrame(_build_fixture_rows()).to_csv(dataset_path, index=False)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--dataset-path",
            str(dataset_path),
            "--review-dir",
            str(review_dir),
            "--symbols",
            "SOLUSDT",
            "--stamp",
            "20260319T110700Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload_out = json.loads(proc.stdout)
    payload = json.loads(Path(payload_out["json_path"]).read_text(encoding="utf-8"))
    assert payload["symbols_requested"] == ["SOLUSDT"]
    assert payload["symbols"] == ["SOLUSDT"]
    assert payload["symbol_count"] == 1
    assert payload["focus_symbol"] == "SOLUSDT"


def test_symbol_sort_key_prioritizes_mixed_positive_over_not_promising() -> None:
    rows = [
        {
            "symbol": "BNBUSDT",
            "validation_status": "not_promising",
            "validation_metrics": {"cumulative_return": -0.01, "trade_count": 2, "profit_factor": 0.7},
        },
        {
            "symbol": "BTCUSDT",
            "validation_status": "mixed_positive",
            "validation_metrics": {"cumulative_return": 0.01, "trade_count": 2, "profit_factor": 1.5},
        },
    ]
    ranked = sorted(rows, key=MODULE.symbol_sort_key)
    assert ranked[0]["symbol"] == "BTCUSDT"
