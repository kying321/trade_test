from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_recent_strategy_backtest_comparison_report.py"
)


def test_builds_recent_strategy_backtest_comparison_report(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    (review_dir / "20260313T050000Z_binance_indicator_combo_etf.json").write_text(
        json.dumps(
            {
                "crypto_family": {
                    "ranked_combos": [
                        {
                            "combo_id": "rsi_breakout",
                            "avg_total_return": -0.09,
                            "avg_profit_factor": 0.53,
                            "avg_win_rate": 0.52,
                            "avg_timely_hit_rate": 1.0,
                            "trade_count": 21,
                        }
                    ]
                },
                "commodity_family": {
                    "ranked_combos": [
                        {
                            "combo_id": "rsi_breakout",
                            "avg_total_return": -0.02,
                            "avg_profit_factor": 0.91,
                            "avg_win_rate": 0.45,
                            "avg_timely_hit_rate": 0.98,
                            "trade_count": 80,
                        },
                        {
                            "combo_id": "ad_rsi_vol_breakout",
                            "avg_total_return": 0.01,
                            "avg_profit_factor": 1.03,
                            "avg_win_rate": 0.43,
                            "avg_timely_hit_rate": 0.74,
                            "trade_count": 60,
                        },
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (review_dir / "20260313T050001Z_binance_indicator_combo_playbook.json").write_text(
        json.dumps(
            {
                "adopt_now": [],
                "research_only": [
                    {"family": "crypto", "combo_id": "rsi_breakout"},
                    {"family": "commodity", "combo_id": "ad_rsi_vol_breakout"},
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (review_dir / "20260313T050002Z_custom_binance_indicator_combo_native_crypto.json").write_text(
        json.dumps(
            {
                "native_crypto_family": {
                    "ranked_combos": [
                        {
                            "combo_id": "rsi_breakout",
                            "avg_total_return": -0.0379,
                            "avg_profit_factor": 0.70,
                            "avg_win_rate": 0.29,
                            "avg_timely_hit_rate": 1.0,
                            "trade_count": 61,
                        }
                    ]
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (review_dir / "20260313T050003Z_binance_indicator_native_lane_stability_report.json").write_text(
        json.dumps(
            {
                "lanes": {
                    "majors": {
                        "long": {
                            "top_combo": "ad_breakout",
                            "top_return": 0.0088,
                            "top_timely_hit_rate": 1.0,
                            "top_trade_count": 68,
                        }
                    },
                    "beta": {
                        "short": {
                            "top_combo": "taker_oi_ad_breakout",
                            "top_return": -0.0091,
                            "top_timely_hit_rate": 0.63,
                            "top_trade_count": 19,
                        }
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (review_dir / "20260313T050004Z_binance_indicator_source_control_report.json").write_text(
        json.dumps(
            {
                "control_verdict": "partial_recovery_without_leadership",
                "control_takeaway": "flow improved but still trails price-state",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (review_dir / "20260313T050005Z_brooks_price_action_market_study.json").write_text(
        json.dumps(
            {
                "adaptive_route_strategy": {
                    "metrics": {
                        "trade_count": 770,
                        "win_rate": 0.4558,
                        "expectancy_r": 0.1064,
                        "profit_factor": 1.2066,
                    },
                    "out_of_sample_metrics": {
                        "trade_count": 246,
                        "win_rate": 0.5162,
                        "expectancy_r": 0.1987,
                        "profit_factor": 1.4483,
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (review_dir / "20260313T050006Z_brooks_price_action_execution_plan.json").write_text(
        json.dumps(
            {
                "head_plan_item": {
                    "symbol": "SC2603",
                    "asset_class": "future",
                    "direction": "LONG",
                    "strategy_id": "second_entry_trend_continuation",
                    "entry_price": 478.8,
                    "stop_price": 471.03,
                    "target_price": 495.88,
                    "rr_ratio": 2.2,
                    "plan_status": "manual_structure_review_now",
                    "plan_blocker_detail": "manual only",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-13T06:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    rows = {row["row_id"]: row for row in payload["rows"]}
    assert payload["strongest_recent_backtest"]["row_id"] == "brooks_adaptive_route_oos"
    assert rows["etf_proxy_crypto_top"]["recommendation"] == "research_only"
    assert rows["etf_proxy_commodity_best_measured"]["strategy_id"] == "ad_rsi_vol_breakout"
    assert rows["native_majors_long_best"]["recommendation"] == "positive_edge_research"
    assert rows["brooks_current_execution_head"]["recommendation"] == "manual_structure_review_now"
    assert payload["crypto_source_control_verdict"] == "partial_recovery_without_leadership"
    assert "artifact" in payload and Path(str(payload["artifact"])).exists()
    assert "markdown" in payload and Path(str(payload["markdown"])).exists()
    assert "checksum" in payload and Path(str(payload["checksum"])).exists()
