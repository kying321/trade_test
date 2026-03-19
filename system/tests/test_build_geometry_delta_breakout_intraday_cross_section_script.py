from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_geometry_delta_breakout_intraday_cross_section.py"
)


def _build_fixture_rows() -> list[dict[str, object]]:
    start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    specs = {
        "BTCUSDT": {"base": 90000.0, "drift": 0.0005, "vol": 1000.0},
        "ETHUSDT": {"base": 2200.0, "drift": 0.0004, "vol": 1200.0},
        "BNBUSDT": {"base": 620.0, "drift": 0.0003, "vol": 900.0},
        "SOLUSDT": {"base": 140.0, "drift": 0.0008, "vol": 1500.0},
    }
    rows: list[dict[str, object]] = []
    for symbol, spec in specs.items():
        close = float(spec["base"])
        for idx in range(320):
            ts = start + timedelta(minutes=15 * idx)
            open_price = close
            step = float(spec["drift"]) * (1.0 + ((idx % 16) - 8) * 0.04)
            if symbol == "SOLUSDT" and idx in {120, 121, 122, 190, 191, 192, 250, 251, 252}:
                step += 0.005
            if symbol == "BTCUSDT" and idx in {170, 171, 172}:
                step += 0.002
            close = max(1.0, open_price * (1.0 + step))
            high = max(open_price, close) * 1.0018
            low = min(open_price, close) * 0.9982
            volume = float(spec["vol"]) * (1.0 + (idx % 7) * 0.10)
            if symbol == "SOLUSDT" and idx in {120, 121, 122, 190, 191, 192, 250, 251, 252}:
                volume *= 3.0
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


def test_build_geometry_delta_breakout_intraday_cross_section_runs_with_fixture(tmp_path: Path) -> None:
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
            "20260319T100500Z",
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
    assert payload["action"] == "build_geometry_delta_breakout_intraday_cross_section"
    assert payload["change_class"] == "RESEARCH_ONLY"
    assert payload["symbol_count"] == 4
    assert payload["cadence_minutes"] == 15
    assert payload["selection_scenario_id"] == "moderate_costs"
    assert payload["selected_params"]
    assert "validation_metrics" in payload
    assert "validation_gross_metrics" in payload
    assert len(payload["validation_scenarios"]) >= 2
    assert len(payload["param_results"]) >= 1
