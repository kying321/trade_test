from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_beta_leg_report.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_beta_leg_report_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _native_payload(symbol_group: str, ranked: list[dict[str, object]], discarded: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "symbol_group": symbol_group,
        "native_crypto_family": {
            "ranked_combos": ranked,
            "discarded_combos": discarded or [],
        },
    }


def test_main_marks_beta_as_basket_only_when_neither_leg_is_stable(tmp_path: Path) -> None:
    module = _load_module()

    artifacts = {
        "sol": _native_payload(
            "sol",
            [
                {"combo_id": "rsi_breakout", "avg_total_return": 0.02, "avg_timely_hit_rate": 1.0, "trade_count": 12},
                {"combo_id": "taker_oi_ad_breakout", "avg_total_return": 0.004, "avg_timely_hit_rate": 0.58, "trade_count": 8},
            ],
        ),
        "sol_long": _native_payload(
            "sol_long",
            [{"combo_id": "ad_breakout", "avg_total_return": 0.006, "avg_timely_hit_rate": 0.91, "trade_count": 16}],
            discarded=[
                {
                    "combo_id": "taker_oi_ad_breakout",
                    "avg_total_return": -0.001,
                    "avg_timely_hit_rate": 0.42,
                    "trade_count": 9,
                    "discard_reason": "laggy",
                }
            ],
        ),
        "bnb": _native_payload(
            "bnb",
            [
                {"combo_id": "rsi_breakout", "avg_total_return": 0.018, "avg_timely_hit_rate": 1.0, "trade_count": 11},
                {"combo_id": "taker_oi_ad_rsi_breakout", "avg_total_return": 0.007, "avg_timely_hit_rate": 0.59, "trade_count": 8},
            ],
        ),
        "bnb_long": _native_payload(
            "bnb_long",
            [{"combo_id": "ad_breakout", "avg_total_return": -0.01, "avg_timely_hit_rate": 0.83, "trade_count": 15}],
            discarded=[
                {
                    "combo_id": "taker_oi_ad_rsi_breakout",
                    "avg_total_return": -0.004,
                    "avg_timely_hit_rate": 0.39,
                    "trade_count": 7,
                    "discard_reason": "laggy",
                }
            ],
        ),
    }
    for stamp, payload in enumerate(artifacts.values(), start=1):
        path = tmp_path / f"20260310T00000{stamp}Z_{payload['symbol_group']}_binance_indicator_combo_native_crypto.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    stability = tmp_path / "20260310T000010Z_binance_indicator_native_lane_stability_report.json"
    stability.write_text(
        json.dumps({"stability": {"beta": {"status": "stable_flow_secondary", "takeaway": "basket beta still holds"}}}),
        encoding="utf-8",
    )

    exit_code = module.main(["--review-dir", str(tmp_path)])
    assert exit_code == 0

    artifacts_out = sorted(tmp_path.glob("*_binance_indicator_native_beta_leg_report.json"))
    assert artifacts_out
    payload = json.loads(artifacts_out[-1].read_text(encoding="utf-8"))
    assert payload["legs"]["sol"]["stability"]["status"] == "short_window_only_flow"
    assert payload["legs"]["bnb"]["stability"]["status"] == "short_window_only_flow"
    assert payload["legs"]["sol"]["short"]["flow_combo_canonical"] == "taker_oi_cvd_breakout"
    assert payload["legs"]["bnb"]["short"]["flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["legs"]["bnb"]["long"]["top_combo_canonical"] == "cvd_breakout"
    assert payload["basket_verdict"] == "basket_only_flow_secondary"
    assert "Do not promote SOL or BNB alone" in payload["overall_takeaway"]


def test_classify_leg_marks_stable_single_leg_when_both_windows_keep_positive_ranked_flow() -> None:
    module = _load_module()
    short_leg = {
        "flow_source": "ranked",
        "flow_return": 0.01,
    }
    long_leg = {
        "flow_source": "ranked",
        "flow_return": 0.004,
    }

    stability = module.classify_leg(short_leg, long_leg)
    assert stability["status"] == "stable_single_leg_flow_secondary"
    assert "across short and extended windows" in stability["takeaway"]


def test_latest_beta_stability_artifact_prefers_latest_timestamp_not_mtime(tmp_path: Path) -> None:
    module = _load_module()
    newer = tmp_path / "20260310T120000Z_binance_indicator_native_lane_stability_report.json"
    older = tmp_path / "20260310T110000Z_binance_indicator_native_lane_stability_report.json"
    newer.write_text(json.dumps({"stability": {"beta": {"status": "fragile_flow_secondary"}}}), encoding="utf-8")
    older.write_text(json.dumps({"stability": {"beta": {"status": "stable_flow_secondary"}}}), encoding="utf-8")
    newer.touch()
    older.touch()
    path, payload = module.latest_beta_stability_artifact(tmp_path)
    assert path == newer
    assert payload["stability"]["beta"]["status"] == "fragile_flow_secondary"


def test_main_marks_beta_as_fragile_leg_only_when_no_leg_is_stable_and_beta_is_fragile(tmp_path: Path) -> None:
    module = _load_module()
    artifacts = {
        "sol": _native_payload(
            "sol",
            [
                {"combo_id": "rsi_breakout", "avg_total_return": 0.02, "avg_timely_hit_rate": 1.0, "trade_count": 12},
                {"combo_id": "taker_oi_ad_breakout", "avg_total_return": 0.004, "avg_timely_hit_rate": 0.58, "trade_count": 8},
            ],
        ),
        "sol_long": _native_payload(
            "sol_long",
            [{"combo_id": "ad_breakout", "avg_total_return": 0.006, "avg_timely_hit_rate": 0.91, "trade_count": 16}],
            discarded=[
                {
                    "combo_id": "taker_oi_ad_breakout",
                    "avg_total_return": -0.001,
                    "avg_timely_hit_rate": 0.42,
                    "trade_count": 9,
                    "discard_reason": "laggy",
                }
            ],
        ),
        "bnb": _native_payload(
            "bnb",
            [
                {"combo_id": "rsi_breakout", "avg_total_return": 0.018, "avg_timely_hit_rate": 1.0, "trade_count": 11},
                {"combo_id": "taker_oi_ad_rsi_breakout", "avg_total_return": 0.007, "avg_timely_hit_rate": 0.59, "trade_count": 8},
            ],
        ),
        "bnb_long": _native_payload(
            "bnb_long",
            [{"combo_id": "ad_breakout", "avg_total_return": -0.01, "avg_timely_hit_rate": 0.83, "trade_count": 15}],
            discarded=[
                {
                    "combo_id": "taker_oi_ad_rsi_breakout",
                    "avg_total_return": -0.004,
                    "avg_timely_hit_rate": 0.39,
                    "trade_count": 7,
                    "discard_reason": "laggy",
                }
            ],
        ),
    }
    for stamp, payload in enumerate(artifacts.values(), start=1):
        path = tmp_path / f"20260310T00000{stamp}Z_{payload['symbol_group']}_binance_indicator_combo_native_crypto.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

    stability = tmp_path / "20260310T000010Z_binance_indicator_native_lane_stability_report.json"
    stability.write_text(
        json.dumps({"stability": {"beta": {"status": "fragile_flow_secondary", "takeaway": "beta still fragile"}}}),
        encoding="utf-8",
    )

    exit_code = module.main(["--review-dir", str(tmp_path)])
    assert exit_code == 0
    artifacts_out = sorted(tmp_path.glob("*_binance_indicator_native_beta_leg_report.json"))
    payload = json.loads(artifacts_out[-1].read_text(encoding="utf-8"))
    assert payload["legs"]["sol"]["short"]["flow_combo_canonical"] == "taker_oi_cvd_breakout"
    assert payload["legs"]["bnb"]["short"]["flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["basket_verdict"] == "fragile_leg_only_flow"
    assert "research-watch mode only" in payload["overall_takeaway"]
