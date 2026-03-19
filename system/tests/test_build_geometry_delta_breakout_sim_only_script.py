from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_geometry_delta_breakout_sim_only.py"
)


def _build_fixture_rows() -> list[dict[str, object]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    close = 100.0
    for idx in range(180):
        ts = start + timedelta(hours=idx)
        open_price = close
        step = 0.0015 * (1.0 + ((idx % 12) - 6) * 0.03)
        if idx in {36, 37, 38, 90, 91, 92, 140, 141, 142}:
            step += 0.012
        close = max(1.0, open_price * (1.0 + step))
        high = max(open_price, close) * 1.004
        low = min(open_price, close) * 0.996
        volume = 1000.0 * (1.0 + (idx % 6) * 0.12)
        if idx in {36, 37, 38, 90, 91, 92, 140, 141, 142}:
            volume *= 3.2
        rows.append(
            {
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "symbol": "SOLUSDT",
                "open": round(open_price, 6),
                "high": round(high, 6),
                "low": round(low, 6),
                "close": round(close, 6),
                "volume": round(volume, 6),
                "asset_class": "crypto",
                "source": "fixture",
            }
        )
    return rows


def test_build_geometry_delta_breakout_sim_only_runs_with_fixture(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    bars_path = review_dir / "20260318T000000Z_crypto_shortline_live_bars_snapshot_bars.csv"
    pd.DataFrame(_build_fixture_rows()).to_csv(bars_path, index=False)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--bars-path",
            str(bars_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260318T180500Z",
            "--symbol",
            "SOLUSDT",
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
    assert payload["action"] == "build_geometry_delta_breakout_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["symbol"] == "SOLUSDT"
    assert payload["sim_scope"] == "single_symbol_intraday_60m"
    assert payload["cadence_minutes"] == 60
    assert payload["intraday_basis_status"] == "single_symbol_only"
    assert payload["param_grid_size"] > 0
    assert payload["selected_params"]
    assert payload["selection_scenario_id"] == "moderate_costs"
    assert "validation_metrics" in payload
    assert "validation_gross_metrics" in payload
    assert len(payload["validation_scenarios"]) >= 2
    assert len(payload["param_results"]) >= 1
    assert len(payload["source_catalog"]) >= 1
