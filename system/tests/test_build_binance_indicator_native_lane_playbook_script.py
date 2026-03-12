from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_native_lane_playbook.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_native_lane_playbook_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_derive_lane_marks_beta_as_flow_secondary() -> None:
    module = _load_module()
    payload = {
        "native_crypto_family": {
            "ranked_combos": [
                {"combo_id": "rsi_breakout", "avg_total_return": -0.03, "avg_timely_hit_rate": 1.0, "trade_count": 33},
                {"combo_id": "taker_oi_ad_rsi_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 0.58, "trade_count": 18},
                {"combo_id": "crowding_filtered_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 0.58, "trade_count": 18},
                {"combo_id": "taker_oi_ad_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 0.58, "trade_count": 18},
                {"combo_id": "taker_oi_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 0.58, "trade_count": 18},
            ]
        }
    }
    status = {"status": "taker_positive_secondary", "taker_best_combo": "taker_oi_ad_rsi_breakout"}
    stability = {"status": "stable_flow_secondary", "takeaway": "beta stable"}
    beta_leg_report = {
        "basket_verdict": "basket_only_flow_secondary",
        "overall_takeaway": "Beta flow survives only at the basket level.",
        "legs": {
            "sol": {
                "short": {"flow_combo": "taker_oi_ad_breakout", "flow_source": "ranked", "flow_return": 0.004},
                "stability": {"status": "short_window_only_flow", "takeaway": "SOL short window only."},
            },
            "bnb": {
                "short": {"flow_combo": "taker_oi_ad_rsi_breakout", "flow_source": "ranked", "flow_return": 0.016},
                "stability": {"status": "short_window_only_flow", "takeaway": "BNB short window only."},
            },
        },
    }
    lane = module.derive_lane("beta", payload, status, stability, beta_leg_report)
    assert lane["deployment_status"] == "price_state_plus_flow_secondary_basket_only"
    assert lane["flow_family"]["family_size"] == 4
    assert lane["flow_family"]["canonical_combo_ids"] == [
        "taker_oi_cvd_rsi_breakout",
        "crowding_filtered_breakout",
        "taker_oi_cvd_breakout",
        "taker_oi_breakout",
    ]
    assert lane["crowding_incremental_value"] == "none_observed"
    assert lane["stability_status"] == "stable_flow_secondary"
    assert lane["leg_constraint"] == "basket_only_flow_secondary"
    assert lane["beta_leg_focus"]["symbol"] == "BNB"
    assert lane["beta_leg_routes"]["bnb"]["status"] == "research_watch_short_window_only"
    assert lane["beta_leg_routes"]["bnb"]["short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"


def test_main_builds_lane_playbook_from_group_report(tmp_path: Path) -> None:
    module = _load_module()
    majors = tmp_path / "majors.json"
    beta = tmp_path / "beta.json"
    group = tmp_path / "20260310T000000Z_binance_indicator_native_group_report.json"
    stability = tmp_path / "20260310T000001Z_binance_indicator_native_lane_stability_report.json"
    beta_leg = tmp_path / "20260310T000002Z_binance_indicator_native_beta_leg_report.json"

    majors.write_text(
        json.dumps(
            {
                "native_crypto_family": {
                    "ranked_combos": [
                        {"combo_id": "rsi_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 1.0, "trade_count": 20},
                        {"combo_id": "taker_oi_ad_breakout", "avg_total_return": -0.01, "avg_timely_hit_rate": 0.57, "trade_count": 10},
                        {"combo_id": "crowding_filtered_breakout", "avg_total_return": -0.01, "avg_timely_hit_rate": 0.57, "trade_count": 10},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    beta.write_text(
        json.dumps(
            {
                "native_crypto_family": {
                    "ranked_combos": [
                        {"combo_id": "rsi_breakout", "avg_total_return": -0.03, "avg_timely_hit_rate": 1.0, "trade_count": 33},
                        {"combo_id": "taker_oi_ad_rsi_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 0.58, "trade_count": 18},
                        {"combo_id": "crowding_filtered_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 0.58, "trade_count": 18},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    group.write_text(
        json.dumps(
            {
                "source_artifacts": {"majors": str(majors), "beta": str(beta)},
                "groups": {
                    "majors": {"status": "taker_ranked_negative", "taker_best_combo": "taker_oi_ad_breakout"},
                    "beta": {"status": "taker_positive_secondary", "taker_best_combo": "taker_oi_ad_rsi_breakout"},
                },
            }
        ),
        encoding="utf-8",
    )
    stability.write_text(
        json.dumps(
            {
                "stability": {
                    "majors": {
                        "status": "stable_price_state_primary",
                        "takeaway": "majors stable",
                    },
                    "beta": {
                        "status": "stable_flow_secondary",
                        "takeaway": "beta stable",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    beta_leg.write_text(
        json.dumps(
            {
                "basket_verdict": "basket_only_flow_secondary",
                "overall_takeaway": "Beta flow survives only at the basket level.",
                "legs": {
                    "sol": {
                        "short": {"flow_combo": "taker_oi_ad_breakout", "flow_source": "ranked", "flow_return": 0.004},
                        "stability": {"status": "short_window_only_flow", "takeaway": "SOL short window only."},
                    },
                    "bnb": {
                        "short": {"flow_combo": "taker_oi_ad_rsi_breakout", "flow_source": "ranked", "flow_return": 0.016},
                        "stability": {"status": "short_window_only_flow", "takeaway": "BNB short window only."},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(["--review-dir", str(tmp_path)])
    assert exit_code == 0
    artifacts = sorted(tmp_path.glob("*_binance_indicator_native_lane_playbook.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text(encoding="utf-8"))
    assert payload["lanes"]["majors"]["deployment_status"] == "price_state_primary_only"
    assert payload["lanes"]["beta"]["deployment_status"] == "price_state_plus_flow_secondary_basket_only"
    assert payload["lanes"]["beta"]["flow_family"]["canonical_combo_ids"] == [
        "taker_oi_cvd_rsi_breakout",
        "crowding_filtered_breakout",
    ]
    assert payload["lanes"]["beta"]["leg_constraint"] == "basket_only_flow_secondary"
    assert payload["lanes"]["beta"]["beta_leg_focus"]["symbol"] == "BNB"
    assert payload["lanes"]["beta"]["beta_leg_routes"]["bnb"]["short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["recommended_live_research_split"]["beta"] == "price_state_plus_flow_secondary_basket_only"
    assert payload["source_artifacts"]["stability_report"] == str(stability)
    assert payload["source_artifacts"]["beta_leg_report"] == str(beta_leg)
    assert payload["source_resolution"]["majors"] == "configured_path"
    assert payload["source_resolution"]["beta"] == "configured_path"


def test_latest_support_reports_prefer_latest_timestamp_not_mtime(tmp_path: Path) -> None:
    module = _load_module()
    older_group = tmp_path / "20260310T110000Z_binance_indicator_native_group_report.json"
    newer_group = tmp_path / "20260310T120000Z_binance_indicator_native_group_report.json"
    older_group.write_text(json.dumps({"marker": "old-group"}), encoding="utf-8")
    newer_group.write_text(json.dumps({"marker": "new-group"}), encoding="utf-8")
    newer_group.touch()
    older_group.touch()

    older_stability = tmp_path / "20260310T110000Z_binance_indicator_native_lane_stability_report.json"
    newer_stability = tmp_path / "20260310T120000Z_binance_indicator_native_lane_stability_report.json"
    older_stability.write_text(json.dumps({"marker": "old-stability"}), encoding="utf-8")
    newer_stability.write_text(json.dumps({"marker": "new-stability"}), encoding="utf-8")
    newer_stability.touch()
    older_stability.touch()

    older_beta_leg = tmp_path / "20260310T110000Z_binance_indicator_native_beta_leg_report.json"
    newer_beta_leg = tmp_path / "20260310T120000Z_binance_indicator_native_beta_leg_report.json"
    older_beta_leg.write_text(json.dumps({"marker": "old-beta-leg"}), encoding="utf-8")
    newer_beta_leg.write_text(json.dumps({"marker": "new-beta-leg"}), encoding="utf-8")
    newer_beta_leg.touch()
    older_beta_leg.touch()

    group_path, group_payload = module.latest_group_report(tmp_path)
    stability_path, stability_payload = module.latest_stability_report(tmp_path)
    beta_leg_path, beta_leg_payload = module.latest_beta_leg_report(tmp_path)

    assert group_path == newer_group
    assert group_payload["marker"] == "new-group"
    assert stability_path == newer_stability
    assert stability_payload["marker"] == "new-stability"
    assert beta_leg_path == newer_beta_leg
    assert beta_leg_payload["marker"] == "new-beta-leg"


def test_derive_lane_keeps_fragile_leg_constraint_visible() -> None:
    module = _load_module()
    payload = {
        "native_crypto_family": {
            "ranked_combos": [
                {"combo_id": "rsi_breakout", "avg_total_return": -0.03, "avg_timely_hit_rate": 1.0, "trade_count": 33},
                {"combo_id": "taker_oi_ad_rsi_breakout", "avg_total_return": 0.01, "avg_timely_hit_rate": 0.58, "trade_count": 18},
            ]
        }
    }
    status = {"status": "taker_positive_secondary", "taker_best_combo": "taker_oi_ad_rsi_breakout"}
    stability = {"status": "fragile_flow_secondary", "takeaway": "beta fragile"}
    beta_leg_report = {
        "basket_verdict": "fragile_leg_only_flow",
        "overall_takeaway": "Keep SOL and BNB in research-watch mode only.",
    }
    lane = module.derive_lane("beta", payload, status, stability, beta_leg_report)
    assert lane["deployment_status"] == "flow_secondary_research_hold"
    assert lane["leg_constraint"] == "fragile_leg_only_flow"
    assert "research-watch mode only" in lane["lane_takeaway"]
