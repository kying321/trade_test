from __future__ import annotations

import importlib.util
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("fenlie_exit_risk_forward_compare_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_fixture_rows(total_bars: int = 2880) -> list[dict[str, object]]:
    start = datetime(2026, 2, 19, 8, 0, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    close = 2200.0
    for idx in range(total_bars):
        ts = start + timedelta(minutes=15 * idx)
        open_price = close
        cycle = idx % 240
        step = 0.00012 * (1.0 + ((idx % 12) - 6) * 0.03)
        if cycle in {90, 91, 92}:
            step += 0.0028
        elif cycle in {93, 94}:
            step -= 0.0019
        elif cycle == 95:
            step += 0.0025
        close = max(1.0, open_price * (1.0 + step))
        high = max(open_price, close) * 1.0018
        low = min(open_price, close) * 0.9985
        volume = 1000.0 * (1.0 + (idx % 6) * 0.12)
        if cycle in {90, 91, 92}:
            volume *= 2.4
        elif cycle in {93, 94}:
            low *= 0.9974
            volume *= 1.6
        elif cycle == 95:
            high *= 1.0018
            volume *= 2.1
        rows.append(
            {
                "ts": ts.isoformat().replace("+00:00", "Z"),
                "symbol": "ETHUSDT",
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


def _write_base_artifact(path: Path) -> None:
    base_artifact = {
        "focus_symbol": "ETHUSDT",
        "selected_params": {
            "breakout_lookback": 40,
            "breakout_memory_bars": 16,
            "pullback_max_atr": 1.2,
            "stop_buffer_atr": 0.1,
            "target_r": 2.0,
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
                    "target_r": 2.0,
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
    path.write_text(json.dumps(base_artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_challenge_pair(path: Path) -> None:
    payload = {
        "symbol": "ETHUSDT",
        "research_decision": "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos",
        "challenge_pair": {
            "baseline_exit_params": {
                "max_hold_bars": 16,
                "break_even_trigger_r": 0.0,
                "trailing_stop_atr": 1.5,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "challenger_exit_params": {
                "max_hold_bars": 8,
                "break_even_trigger_r": 0.75,
                "trailing_stop_atr": 1.5,
                "cooldown_after_losses": 0,
                "cooldown_bars": 0,
            },
            "primary_axis": "max_hold_bars",
            "baseline_hold_bars": 16,
            "challenger_hold_bars": 8,
            "shared_trailing_stop_atr": 1.5,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_classify_pair_research_decision_supports_mixed_profile() -> None:
    module = _load_module()

    assert module.classify_pair_research_decision(
        aggregate_return_winner="baseline_pair",
        aggregate_objective_winner="challenger_pair",
        slice_majority_return_winner="baseline_pair",
        slice_majority_objective_winner="tie",
    ) == "mixed_forward_oos_pair_agg_ret_baseline_agg_obj_challenger_slice_ret_baseline_slice_obj_tie"


def test_exit_risk_forward_compare_uses_blocker_pair_and_runs(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = review_dir / "20260321T000000Z_public_intraday_crypto_bars_dataset.csv"
    pd.DataFrame(_build_fixture_rows()).to_csv(dataset_path, index=False)
    base_artifact_path = review_dir / "20260321T000500Z_price_action_breakout_pullback_sim_only.json"
    blocker_path = review_dir / "20260323T024500Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    _write_base_artifact(base_artifact_path)
    _write_challenge_pair(blocker_path)

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--dataset-path",
            str(dataset_path),
            "--base-artifact-path",
            str(base_artifact_path),
            "--challenge-pair-path",
            str(blocker_path),
            "--symbol",
            "ETHUSDT",
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T031000Z",
            "--train-days",
            "20",
            "--validation-days",
            "5",
            "--step-days",
            "5",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    json_path = Path(output["json_path"])
    latest_json_path = Path(output["latest_json_path"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest_json_path.read_text(encoding="utf-8"))

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_forward_compare_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["train_days"] == 20
    assert payload["validation_days"] == 5
    assert payload["step_days"] == 5
    assert payload["validation_window_mode"] == "non_overlapping"
    assert payload["slice_count"] == 2
    assert payload["challenge_pair_source_decision"] == "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos"
    assert [row["config_id"] for row in payload["comparison_configs"]] == ["baseline_pair", "challenger_pair"]
    assert payload["comparison_configs"][0]["exit_params"]["max_hold_bars"] == 16
    assert payload["comparison_configs"][1]["exit_params"]["max_hold_bars"] == 8
    assert set(payload["aggregate_validation_metrics_by_config"].keys()) == {"baseline_pair", "challenger_pair"}
    assert latest_payload["research_decision"] == payload["research_decision"]
