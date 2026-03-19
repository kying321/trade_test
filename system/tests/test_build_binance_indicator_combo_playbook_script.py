from __future__ import annotations

import importlib.util
import datetime as dt
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_combo_playbook.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_combo_playbook_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_classify_playbook_splits_adopt_context_and_retest() -> None:
    module = _load_module()
    native_lane = {
        "lanes": {
            "majors": {
                "deployment_status": "price_state_primary_only",
                "stability_status": "stable_price_state_primary",
                "lane_takeaway": "majors stay price-state",
            },
            "beta": {
                "deployment_status": "price_state_plus_flow_secondary_basket_only",
                "stability_status": "stable_flow_secondary",
                "lane_takeaway": "beta gets flow secondary",
                "leg_constraint": "basket_only_flow_secondary",
                "leg_constraint_takeaway": "Do not promote SOL or BNB alone.",
                "beta_leg_routes": {
                    "sol": {"status": "research_watch_short_window_only"},
                    "bnb": {"status": "research_watch_short_window_only"},
                },
                "beta_leg_focus": {
                    "symbol": "BNB",
                    "recommended_use": "watch_short_window_only",
                    "short_flow_return": 0.016,
                },
            },
        },
        "recommended_live_research_split": {
            "majors": "price_state_primary_only",
            "beta": "price_state_plus_flow_secondary_basket_only",
        },
    }
    beta_window = {
        "overall_takeaway": "Treat beta flow as short-window fragile until a leg keeps ranked positive flow on the long window.",
        "next_focus_symbol": "BNBUSDT",
        "next_focus_action": "watch_priority_until_long_window_confirms",
        "next_focus_reason": "BNB has the best short-window beta flow, but it degrades when the window is extended.",
        "legs": {
            "BNBUSDT": {
                "action": "watch_priority_until_long_window_confirms",
                "action_reason": "BNB has the best short-window beta flow, but it degrades when the window is extended.",
                "flow_window_verdict": "degrades_on_long_window",
                "price_state_window_verdict": "degrades_on_long_window",
            },
            "SOLUSDT": {
                "action": "watch_only",
                "action_reason": "SOL shows short-window flow only and loses ranked flow on the long window.",
                "flow_window_verdict": "degrades_on_long_window",
                "price_state_window_verdict": "stable_across_windows",
            },
        },
    }
    payload = module.classify_playbook(
        {
            "crypto_family": {
                "ranked_combos": [
                    {
                        "combo_id": "rsi_breakout",
                        "avg_total_return": 0.03,
                        "avg_profit_factor": 1.25,
                        "avg_timely_hit_rate": 1.0,
                        "trade_count": 23,
                    }
                ],
                "discarded_combos": [
                    {
                        "combo_id": "taker_oi_breakout",
                        "discard_reason": "laggy_support_resistance_timing",
                        "trade_count": 2,
                        "avg_timely_hit_rate": 0.09,
                        "avg_lag_bars": 0.0,
                    }
                ],
            },
            "commodity_family": {
                "ranked_combos": [
                    {
                        "combo_id": "rsi_breakout",
                        "avg_total_return": -0.02,
                        "avg_profit_factor": 0.92,
                        "avg_timely_hit_rate": 0.84,
                        "trade_count": 62,
                    },
                    {
                        "combo_id": "ad_rsi_vol_breakout",
                        "avg_total_return": 0.01,
                        "avg_profit_factor": 1.05,
                        "avg_timely_hit_rate": 0.74,
                        "trade_count": 41,
                    },
                ],
                "discarded_combos": [],
            },
            "crypto_takeaway": "crypto measured",
            "crypto_practitioner_note": "crypto practitioner",
            "commodity_takeaway": "commodity measured",
            "commodity_practitioner_note": "commodity practitioner",
        },
        native_lane,
        beta_window,
    )
    assert payload["adopt_now"][0]["combo_id"] == "rsi_breakout"
    assert payload["adopt_now"][0]["combo_id_canonical"] == "rsi_breakout"
    commodity_rows = [row for row in payload["research_only"] if row["family"] == "commodity"]
    assert commodity_rows[0]["combo_id"] == "ad_rsi_vol_breakout"
    assert commodity_rows[0]["combo_id_canonical"] == "cvd_rsi_vol_breakout"
    assert payload["context_only"][0]["combo_id"] == "taker_oi_breakout"
    assert payload["context_only"][0]["combo_id_canonical"] == "taker_oi_breakout"
    assert payload["retest_native_crypto"][0]["combo_id"] == "taker_oi_breakout"
    assert payload["retest_native_crypto"][0]["combo_id_canonical"] == "taker_oi_breakout"
    assert payload["measured_vs_practitioner"]["crypto"]["measured"] == "crypto measured"
    assert payload["native_crypto_lanes"]["beta"]["deployment_status"] == "price_state_plus_flow_secondary_basket_only"
    assert payload["native_crypto_lanes"]["beta"]["beta_leg_focus"]["symbol"] == "BNB"
    assert payload["native_crypto_lanes"]["beta"]["beta_leg_window_focus"]["symbol"] == "BNBUSDT"
    assert payload["symbol_routes"]["BTCUSDT"]["action"] == "deploy_price_state_only"
    assert payload["symbol_routes"]["BNBUSDT"]["action"] == "watch_priority_until_long_window_confirms"
    assert payload["symbol_routes"]["BNBUSDT"]["flow_combo_canonical"] is None
    assert payload["symbol_routes"]["BNBUSDT"]["reason"] == "BNB has the best short-window beta flow, but it degrades when the window is extended."
    assert payload["symbol_routes"]["SOLUSDT"]["flow_window_verdict"] == "degrades_on_long_window"
    assert payload["beta_leg_window_takeaway"] == beta_window["overall_takeaway"]
    assert "CVD-lite" in payload["overall_takeaway"]


def test_classify_playbook_demotes_negative_crypto_combo_from_adopt_now() -> None:
    module = _load_module()
    payload = module.classify_playbook(
        {
            "crypto_family": {
                "ranked_combos": [
                    {
                        "combo_id": "rsi_breakout",
                        "avg_total_return": -0.09,
                        "avg_profit_factor": 0.53,
                        "avg_timely_hit_rate": 1.0,
                        "trade_count": 21,
                    }
                ],
                "discarded_combos": [],
            },
            "commodity_family": {"ranked_combos": [], "discarded_combos": []},
            "crypto_takeaway": "crypto measured",
            "crypto_practitioner_note": "crypto practitioner",
            "commodity_takeaway": "commodity measured",
            "commodity_practitioner_note": "commodity practitioner",
        }
    )
    assert payload["adopt_now"] == []
    assert payload["research_only"][0]["family"] == "crypto"
    assert payload["research_only"][0]["combo_id"] == "rsi_breakout"
    assert payload["research_only"][0]["promotion_gate_status"] == "blocked"
    assert "non_positive_return" in payload["research_only"][0]["promotion_gate_reason"]
    assert "profit_factor_not_above_one" in payload["research_only"][0]["promotion_gate_reason"]
    assert "No ETF-proxy crypto combo clears the measured promotion gate" in payload["overall_takeaway"]


def test_latest_native_lane_playbook_prefers_latest_timestamp_not_mtime(tmp_path: Path) -> None:
    module = _load_module()
    older = tmp_path / "20260310T122721Z_binance_indicator_native_lane_playbook.json"
    newer = tmp_path / "20260310T151931Z_binance_indicator_native_lane_playbook.json"
    older.write_text(json.dumps({"marker": "older"}), encoding="utf-8")
    newer.write_text(json.dumps({"marker": "newer"}), encoding="utf-8")
    # Simulate mtime skew where older file is touched later.
    older.touch()
    path, payload = module.latest_native_lane_playbook(tmp_path)
    assert path == newer
    assert payload["marker"] == "newer"


def test_latest_artifact_by_suffix_ignores_future_stamped_artifact(tmp_path: Path) -> None:
    module = _load_module()
    module.now_utc = lambda: dt.datetime(2026, 3, 10, 15, 58, tzinfo=dt.timezone.utc)
    current = tmp_path / "20260310T154122Z_binance_indicator_native_beta_leg_window_report.json"
    future = tmp_path / "20260310T234122Z_binance_indicator_native_beta_leg_window_report.json"
    current.write_text(json.dumps({"marker": "current"}), encoding="utf-8")
    future.write_text(json.dumps({"marker": "future"}), encoding="utf-8")
    path = module.latest_artifact_by_suffix(tmp_path, "binance_indicator_native_beta_leg_window_report")
    assert path == current


def test_latest_artifact_by_suffix_respects_reference_now(tmp_path: Path) -> None:
    module = _load_module()
    current = tmp_path / "20260310T154122Z_binance_indicator_native_beta_leg_window_report.json"
    future = tmp_path / "20260310T234122Z_binance_indicator_native_beta_leg_window_report.json"
    current.write_text(json.dumps({"marker": "current"}), encoding="utf-8")
    future.write_text(json.dumps({"marker": "future"}), encoding="utf-8")
    ref_now = module.parse_now("2026-03-10T23:45:00Z")
    path = module.latest_artifact_by_suffix(
        tmp_path,
        "binance_indicator_native_beta_leg_window_report",
        ref_now,
    )
    assert path == future


def test_classify_playbook_uses_fragile_beta_language_when_lane_is_research_hold() -> None:
    module = _load_module()
    payload = module.classify_playbook(
        {
            "crypto_family": {"ranked_combos": [], "discarded_combos": []},
            "commodity_family": {"ranked_combos": [], "discarded_combos": []},
            "crypto_takeaway": "crypto measured",
            "crypto_practitioner_note": "crypto practitioner",
            "commodity_takeaway": "commodity measured",
            "commodity_practitioner_note": "commodity practitioner",
        },
        {
            "lanes": {
                "majors": {
                    "deployment_status": "price_state_primary_only",
                    "stability_status": "stable_price_state_primary",
                    "lane_takeaway": "majors stay price-state",
                },
                "beta": {
                    "deployment_status": "flow_secondary_research_hold",
                    "stability_status": "fragile_flow_secondary",
                    "lane_takeaway": "beta remains fragile",
                    "leg_constraint": "single_leg_supported",
                    "leg_constraint_takeaway": "At least one beta leg supports standalone flow-secondary deployment.",
                },
            },
            "recommended_live_research_split": {
                "majors": "price_state_primary_only",
                "beta": "flow_secondary_research_hold",
            },
        },
    )
    assert payload["native_crypto_lanes"]["beta"]["deployment_status"] == "flow_secondary_research_hold"
    assert "fragile research-hold secondary layer" in payload["overall_takeaway"]


def test_render_markdown_includes_key_sections() -> None:
    module = _load_module()
    md = module.render_markdown(
        {
            "as_of": "2026-03-10T00:00:00+00:00",
            "source_artifact": "/tmp/source.json",
            "adopt_now": [{"family": "crypto", "combo_id": "rsi_breakout", "combo_id_canonical": "rsi_breakout", "return": 0.03, "timely_hit_rate": 1.0, "reason": "best"}],
            "research_only": [],
            "context_only": [],
            "discard_now": [],
            "retest_native_crypto": [],
            "measured_vs_practitioner": {
                "crypto": {"measured": "m1", "practitioner": "p1"},
                "commodity": {"measured": "m2", "practitioner": "p2"},
            },
            "native_crypto_lanes": {
                "majors": {
                    "deployment_status": "price_state_primary_only",
                    "stability_status": "stable_price_state_primary",
                    "lane_takeaway": "majors stay price-state",
                },
                "beta": {
                    "deployment_status": "price_state_plus_flow_secondary_basket_only",
                    "stability_status": "stable_flow_secondary",
                    "lane_takeaway": "beta gets flow secondary",
                    "leg_constraint": "basket_only_flow_secondary",
                    "leg_constraint_takeaway": "Do not promote SOL or BNB alone.",
                    "beta_leg_window_focus": {
                        "symbol": "BNBUSDT",
                        "action": "watch_priority_until_long_window_confirms",
                        "reason": "BNB degrades on the long window.",
                    },
                },
                "recommended_live_research_split": {
                    "majors": "price_state_primary_only",
                    "beta": "price_state_plus_flow_secondary_basket_only",
                },
            },
            "beta_leg_window_takeaway": "Treat beta flow as short-window fragile until a leg keeps ranked positive flow on the long window.",
            "symbol_routes": {
                "BTCUSDT": {
                    "lane": "majors",
                    "deployment": "price_state_primary_only",
                    "action": "deploy_price_state_only",
                    "reason": "majors stable",
                },
                "BNBUSDT": {
                    "lane": "beta",
                    "deployment": "price_state_plus_flow_secondary_basket_only",
                    "action": "watch_short_window_flow_priority",
                    "reason": "bnb short window only",
                    "flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
                    "flow_window_verdict": "degrades_on_long_window",
                    "price_state_window_verdict": "degrades_on_long_window",
                },
            },
            "overall_takeaway": "keep it simple",
        }
    )
    assert "## Adopt Now" in md
    assert "## Measured Vs Practitioner" in md
    assert "## Native Crypto Lanes" in md
    assert "## Symbol Routes" in md
    assert "beta-window-focus" in md
    assert "beta-window-takeaway" in md
    assert "canonical=" in md
    assert "flow_combo_canonical" in md
    assert "keep it simple" in md
