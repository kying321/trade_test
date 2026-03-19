from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_public_quant_strategy_cross_section_research.py"
)


def _build_fixture_rows() -> list[dict[str, object]]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    specs = {
        "BTCUSDT": {"base": 100.0, "drift": 0.004, "vol": 1200.0},
        "ETHUSDT": {"base": 80.0, "drift": 0.003, "vol": 1000.0},
        "SOLUSDT": {"base": 60.0, "drift": 0.005, "vol": 1500.0},
        "BNBUSDT": {"base": 70.0, "drift": 0.002, "vol": 900.0},
    }
    rows: list[dict[str, object]] = []
    for symbol, spec in specs.items():
        close = float(spec["base"])
        for idx in range(72):
            ts = start + timedelta(days=idx)
            open_price = close
            step = float(spec["drift"]) * (1.0 + ((idx % 7) - 3) * 0.08)
            if symbol == "SOLUSDT" and idx in {18, 19, 20, 38, 39, 40, 58, 59, 60}:
                step += 0.018
            if symbol == "BNBUSDT" and idx in {24, 25, 26, 50, 51}:
                step -= 0.010
            close = max(1.0, open_price * (1.0 + step))
            high = max(open_price, close) * 1.012
            low = min(open_price, close) * 0.988
            volume = float(spec["vol"]) * (1.0 + (idx % 5) * 0.10)
            if symbol == "SOLUSDT" and idx in {20, 40, 60}:
                volume *= 3.5
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
                }
            )
    return rows


def test_build_public_quant_strategy_cross_section_research_runs_with_fixture(tmp_path: Path) -> None:
    dataset_path = tmp_path / "fixture_crypto_daily.csv"
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
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
            "20260318T170500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    stdout_payload = json.loads(proc.stdout)
    json_path = Path(stdout_payload["json_path"])
    md_path = Path(stdout_payload["md_path"])
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["action"] == "build_public_quant_strategy_cross_section_research"
    assert payload["symbol_count"] == 4
    assert len(payload["strategy_results"]) == 6
    assert len(payload["ranking"]) == 6
    assert len(payload["source_catalog"]) >= 6
    assert "research_action_ladder" in payload
    assert "continue_research_families" in payload["research_action_ladder"]
    assert str(payload["recommended_family_id"]).strip() != ""
    assert any(row["family_id"] == "geometry_delta_breakout" for row in payload["strategy_results"])
