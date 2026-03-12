from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_native_lane_stability_report.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_native_lane_stability_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_classify_stability_marks_beta_as_stable_flow_secondary() -> None:
    module = _load_module()
    short_lane = {"top_return": -0.03, "flow_return": 0.01}
    long_lane = {"top_return": -0.01, "flow_return": 0.02}
    result = module.classify_stability(short_lane, long_lane, lane_name="beta")
    assert result["status"] == "stable_flow_secondary"


def test_main_builds_stability_report(tmp_path: Path) -> None:
    module = _load_module()
    data = {
        "majors": {"native_crypto_family": {"ranked_combos": [
            {"combo_id": "rsi_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 1.0, "trade_count": 20},
            {"combo_id": "taker_oi_breakout", "avg_total_return": -0.01, "avg_timely_hit_rate": 0.58, "trade_count": 10},
        ]}},
        "majors_long": {"native_crypto_family": {"ranked_combos": [
            {"combo_id": "rsi_breakout", "avg_total_return": 0.02, "avg_timely_hit_rate": 1.0, "trade_count": 40},
            {"combo_id": "taker_oi_breakout", "avg_total_return": -0.02, "avg_timely_hit_rate": 0.56, "trade_count": 18},
        ]}},
        "beta": {"native_crypto_family": {"ranked_combos": [
            {"combo_id": "rsi_breakout", "avg_total_return": -0.03, "avg_timely_hit_rate": 1.0, "trade_count": 33},
            {"combo_id": "taker_oi_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 0.58, "trade_count": 18},
        ]}},
        "beta_long": {"native_crypto_family": {"ranked_combos": [
            {"combo_id": "rsi_breakout", "avg_total_return": -0.01, "avg_timely_hit_rate": 1.0, "trade_count": 50},
            {"combo_id": "taker_oi_breakout", "avg_total_return": 0.02, "avg_timely_hit_rate": 0.6, "trade_count": 24},
        ]}},
    }
    for key, payload in data.items():
        path = tmp_path / f"20260310T000000Z_{key}_binance_indicator_combo_native_crypto.json"
        path.write_text(json.dumps({"symbol_group": key, **payload}), encoding="utf-8")

    exit_code = module.main(
        [
            "--review-dir",
            str(tmp_path),
            "--short-majors-group",
            "majors",
            "--short-beta-group",
            "beta",
            "--long-majors-group",
            "majors_long",
            "--long-beta-group",
            "beta_long",
        ]
    )
    assert exit_code == 0
    artifacts = sorted(tmp_path.glob("*_binance_indicator_native_lane_stability_report.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text(encoding="utf-8"))
    assert payload["stability"]["majors"]["status"] == "stable_price_state_primary"
    assert payload["stability"]["beta"]["status"] == "stable_flow_secondary"
    assert payload["lanes"]["majors"]["short"]["top_combo_canonical"] == "rsi_breakout"
    assert payload["lanes"]["beta"]["short"]["flow_combo_canonical"] == "taker_oi_breakout"
    assert "split survives the longer window" in payload["overall_takeaway"]
