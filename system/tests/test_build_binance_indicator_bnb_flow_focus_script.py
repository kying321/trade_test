from __future__ import annotations

import importlib.util
import datetime as dt
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_bnb_flow_focus.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("bnb_flow_focus_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_derive_focus_payload_blocks_promotion_until_long_window_confirms() -> None:
    module = _load_module()
    payload = module.derive_focus_payload_from_beta_window(
        {
            "legs": {
                "BNBUSDT": {
                    "flow_window_verdict": "degrades_on_long_window",
                    "price_state_window_verdict": "degrades_on_long_window",
                    "action": "watch_priority_until_long_window_confirms",
                    "action_reason": "BNB has the best short-window beta flow, but it degrades when the window is extended.",
                    "short_flow_combo": "taker_oi_ad_rsi_breakout",
                    "short_flow_return": 0.0159,
                    "short_flow_timely_hit_rate": 0.60,
                    "long_flow_combo": "crowding_filtered_breakout",
                    "long_flow_return": 0.0159,
                    "long_flow_timely_hit_rate": 0.18,
                    "short_top_combo": "rsi_breakout",
                    "short_top_return": 0.012,
                    "long_top_combo": "ad_breakout",
                    "long_top_return": -0.034,
                    "flow_return_delta": 0.0,
                    "flow_timely_hit_rate_delta": -0.42,
                }
            }
        }
    )
    assert payload["symbol"] == "BNBUSDT"
    assert payload["promotion_gate"] == "blocked_until_long_window_confirms"
    assert payload["operator_status"] == "watch_priority"
    assert payload["next_retest_action"] == "rerun_bnb_native_long_window"


def test_derive_focus_payload_from_direct_native_preserves_positive_long_window_floor() -> None:
    module = _load_module()
    payload = module.derive_focus_payload_from_direct_native(
        {
            "native_crypto_family": {
                "ranked_combos": [
                    {
                        "combo_id": "taker_oi_ad_rsi_breakout",
                        "avg_total_return": 0.0159,
                        "avg_timely_hit_rate": 0.61,
                        "trade_count": 9,
                    },
                    {
                        "combo_id": "rsi_breakout",
                        "avg_total_return": 0.0054,
                        "avg_timely_hit_rate": 1.0,
                        "trade_count": 18,
                    },
                ],
                "discarded_combos": [],
            }
        },
        {
            "native_crypto_family": {
                "ranked_combos": [
                    {
                        "combo_id": "taker_oi_ad_rsi_breakout",
                        "avg_total_return": 0.00028,
                        "avg_timely_hit_rate": 0.54,
                        "trade_count": 25,
                    },
                    {
                        "combo_id": "ad_breakout",
                        "avg_total_return": -0.0251,
                        "avg_timely_hit_rate": 0.98,
                        "trade_count": 48,
                    },
                ],
                "discarded_combos": [],
            }
        },
        {
            "native_crypto_family": {
                "ranked_combos": [],
                "discarded_combos": [
                    {
                        "combo_id": "taker_oi_ad_rsi_breakout",
                        "avg_total_return": 0.0159,
                        "avg_timely_hit_rate": 0.13,
                        "trade_count": 9,
                    },
                    {
                        "combo_id": "ad_breakout",
                        "avg_total_return": 0.0940,
                        "avg_timely_hit_rate": 0.99,
                        "trade_count": 68,
                    },
                ],
            }
        },
    )
    assert payload["source_mode"] == "direct_bnb_native"
    assert payload["promotion_gate"] == "blocked_until_long_window_confirms"
    assert payload["flow_window_verdict"] == "degrades_on_long_window"
    assert payload["flow_window_floor"] == "positive_but_weaker"
    assert payload["xlong_flow_window_floor"] == "laggy_positive_only"
    assert payload["price_state_window_floor"] == "negative"
    assert payload["short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["long_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["long_top_combo_canonical"] == "cvd_breakout"
    assert payload["comparative_window_takeaway"].startswith("Long-window flow holds up better")
    assert payload["xlong_comparative_window_takeaway"].startswith("Extra-long flow keeps a raw positive return")
    assert payload["next_retest_action"] == "wait_for_more_bnb_native_data"


def test_latest_beta_leg_window_report_ignores_future_stamped_artifact(tmp_path: Path) -> None:
    module = _load_module()
    module.now_utc = lambda: dt.datetime(2026, 3, 10, 15, 58, tzinfo=dt.timezone.utc)
    current = tmp_path / "20260310T154122Z_binance_indicator_native_beta_leg_window_report.json"
    future = tmp_path / "20260310T234122Z_binance_indicator_native_beta_leg_window_report.json"
    current.write_text(json.dumps({"marker": "current"}), encoding="utf-8")
    future.write_text(json.dumps({"marker": "future"}), encoding="utf-8")
    assert module.latest_beta_leg_window_report(tmp_path) == current


def test_script_writes_bnb_focus_artifacts(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    short_source = review_dir / "20260310T173143Z_bnb_short_binance_indicator_combo_native_crypto.json"
    long_source = review_dir / "20260310T173143Z_bnb_long_binance_indicator_combo_native_crypto.json"
    short_source.write_text(
        json.dumps(
            {
                "native_crypto_family": {
                    "ranked_combos": [
                        {
                            "combo_id": "taker_oi_ad_rsi_breakout",
                            "avg_total_return": 0.0159,
                            "avg_timely_hit_rate": 0.60,
                            "trade_count": 9,
                        },
                        {
                            "combo_id": "rsi_breakout",
                            "avg_total_return": 0.012,
                            "avg_timely_hit_rate": 1.0,
                            "trade_count": 18,
                        },
                    ],
                    "discarded_combos": [],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    long_source.write_text(
        json.dumps(
            {
                "native_crypto_family": {
                    "ranked_combos": [
                        {
                            "combo_id": "taker_oi_ad_rsi_breakout",
                            "avg_total_return": 0.00028,
                            "avg_timely_hit_rate": 0.54,
                            "trade_count": 25,
                        },
                        {
                            "combo_id": "ad_breakout",
                            "avg_total_return": -0.034,
                            "avg_timely_hit_rate": 0.98,
                            "trade_count": 48,
                        },
                    ],
                    "discarded_combos": [],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    xlong_source = review_dir / "20260310T174622Z_bnb_xlong_binance_indicator_combo_native_crypto.json"
    xlong_source.write_text(
        json.dumps(
            {
                "native_crypto_family": {
                    "ranked_combos": [],
                    "discarded_combos": [
                        {
                            "combo_id": "taker_oi_ad_rsi_breakout",
                            "avg_total_return": 0.0159,
                            "avg_timely_hit_rate": 0.13,
                            "trade_count": 9,
                        },
                        {
                            "combo_id": "ad_breakout",
                            "avg_total_return": 0.0940,
                            "avg_timely_hit_rate": 0.99,
                            "trade_count": 68,
                        },
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T16:10:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["symbol"] == "BNBUSDT"
    assert payload["promotion_gate"] == "blocked_until_long_window_confirms"
    assert payload["source_mode"] == "direct_bnb_native"
    assert payload["flow_window_floor"] == "positive_but_weaker"
    assert payload["xlong_flow_window_floor"] == "laggy_positive_only"
    assert payload["short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["long_top_combo_canonical"] == "cvd_breakout"
    assert payload["next_retest_action"] == "wait_for_more_bnb_native_data"
    assert payload["source_artifacts"] == [str(short_source), str(long_source), str(xlong_source)]
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()
