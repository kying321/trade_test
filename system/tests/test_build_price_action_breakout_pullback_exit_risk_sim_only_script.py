from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_sim_only.py"
)


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
            if symbol == "ETHUSDT" and idx in breakout_bars:
                step += 0.0028
            elif symbol == "ETHUSDT" and idx in pullback_bars:
                step -= 0.0019
            elif symbol == "ETHUSDT" and idx in reclaim_bars:
                step += 0.0026
            close = max(1.0, open_price * (1.0 + step))
            high = max(open_price, close) * 1.0018
            low = min(open_price, close) * 0.9985
            volume = float(spec["vol"]) * (1.0 + (idx % 6) * 0.12)
            if symbol == "ETHUSDT" and idx in breakout_bars:
                volume *= 2.4
            elif symbol == "ETHUSDT" and idx in pullback_bars:
                low *= 0.9974
                volume *= 1.6
            elif symbol == "ETHUSDT" and idx in reclaim_bars:
                high *= 1.0018
                volume *= 2.1
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


def test_build_price_action_breakout_pullback_exit_risk_sim_only_runs_with_fixture(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = review_dir / "20260319T000000Z_public_intraday_crypto_bars_dataset.csv"
    pd.DataFrame(_build_fixture_rows()).to_csv(dataset_path, index=False)

    base_artifact = {
        "focus_symbol": "ETHUSDT",
        "selected_params": {
            "breakout_lookback": 40,
            "breakout_memory_bars": 16,
            "pullback_max_atr": 1.2,
            "stop_buffer_atr": 0.1,
            "target_r": 1.5,
            "max_hold_bars": 16,
            "trend_filter": False,
            "structure_filter": "none",
            "vpvr_filter": False,
            "vpvr_max_distance_atr": 1.0,
            "daily_stop_r": 0.0,
        },
        "symbol_results": [
            {
                "symbol": "ETHUSDT",
                "selected_params": {
                    "breakout_lookback": 40,
                    "breakout_memory_bars": 16,
                    "pullback_max_atr": 1.2,
                    "stop_buffer_atr": 0.1,
                    "target_r": 1.5,
                    "max_hold_bars": 16,
                    "trend_filter": False,
                    "structure_filter": "none",
                    "vpvr_filter": False,
                    "vpvr_max_distance_atr": 1.0,
                    "daily_stop_r": 0.0,
                },
            }
        ],
    }
    base_artifact_path = review_dir / "20260319T101700Z_price_action_breakout_pullback_sim_only.json"
    base_artifact_path.write_text(json.dumps(base_artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--dataset-path",
            str(dataset_path),
            "--base-artifact-path",
            str(base_artifact_path),
            "--review-dir",
            str(review_dir),
            "--symbol",
            "ETHUSDT",
            "--stamp",
            "20260319T111500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload_out = json.loads(proc.stdout)
    json_path = Path(payload_out["json_path"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["base_entry_params"]["breakout_lookback"] == 40
    assert "break_even_trigger_r" in payload["selected_exit_params"]
    assert "trailing_stop_atr" in payload["selected_exit_params"]
    assert "cooldown_after_losses" in payload["selected_exit_params"]
    assert "cooldown_bars" in payload["selected_exit_params"]
    assert payload["selection_policy"] == "train_first_validation_tiebreak"
    assert "validation_leader_exit_params" in payload
    assert "validation_leader_metrics" in payload
    assert payload["candidate_count"] > 0
    assert payload["baseline_validation_status"]
    assert payload["selected_validation_status"]
    assert payload["ranking"]
