from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_beta_leg_window_report.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_beta_leg_window_report_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_summarize_leg_marks_long_window_degradation_for_bnb() -> None:
    module = _load_module()
    row = module.summarize_leg(
        "BNBUSDT",
        {
            "short": {
                "flow_combo": "taker_oi_ad_rsi_breakout",
                "flow_source": "ranked",
                "flow_return": 0.016,
                "flow_timely_hit_rate": 0.60,
                "top_combo": "rsi_breakout",
                "top_return": 0.012,
            },
            "long": {
                "flow_combo": "crowding_filtered_breakout",
                "flow_source": "discarded",
                "flow_return": 0.016,
                "flow_timely_hit_rate": 0.18,
                "top_combo": "ad_breakout",
                "top_return": -0.034,
            },
            "stability": {
                "status": "short_window_only_flow",
                "takeaway": "short only",
            },
        },
    )
    assert row["flow_window_verdict"] == "degrades_on_long_window"
    assert row["price_state_window_verdict"] == "degrades_on_long_window"
    assert row["action"] == "watch_priority_until_long_window_confirms"
    assert row["short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert row["long_top_combo_canonical"] == "cvd_breakout"


def test_main_builds_window_report(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "20260310T000000Z_binance_indicator_native_beta_leg_report.json"
    source.write_text(
        json.dumps(
            {
                "basket_verdict": "fragile_leg_only_flow",
                "legs": {
                    "bnb": {
                        "short": {
                            "flow_combo": "taker_oi_ad_rsi_breakout",
                            "flow_source": "ranked",
                            "flow_return": 0.016,
                            "flow_timely_hit_rate": 0.60,
                            "top_combo": "rsi_breakout",
                            "top_return": 0.012,
                        },
                        "long": {
                            "flow_combo": "crowding_filtered_breakout",
                            "flow_source": "discarded",
                            "flow_return": 0.016,
                            "flow_timely_hit_rate": 0.18,
                            "top_combo": "ad_breakout",
                            "top_return": -0.034,
                        },
                    },
                    "sol": {
                        "short": {
                            "flow_combo": "taker_oi_ad_breakout",
                            "flow_source": "ranked",
                            "flow_return": 0.004,
                            "flow_timely_hit_rate": 0.56,
                            "top_combo": "taker_oi_ad_breakout",
                            "top_return": 0.004,
                        },
                        "long": {
                            "flow_combo": "crowding_filtered_breakout",
                            "flow_source": "discarded",
                            "flow_return": 0.004,
                            "flow_timely_hit_rate": 0.18,
                            "top_combo": "ad_rsi_breakout",
                            "top_return": 0.006,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    rc = module.main(["--review-dir", str(tmp_path)])
    assert rc == 0
    artifacts = sorted(tmp_path.glob("*_binance_indicator_native_beta_leg_window_report.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["next_focus_symbol"] == "BNBUSDT"
    assert payload["legs"]["BNBUSDT"]["flow_window_verdict"] == "degrades_on_long_window"
    assert payload["legs"]["BNBUSDT"]["short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["legs"]["BNBUSDT"]["long_top_combo_canonical"] == "cvd_breakout"
    assert payload["legs"]["SOLUSDT"]["short_flow_combo_canonical"] == "taker_oi_cvd_breakout"
    assert payload["legs"]["SOLUSDT"]["long_top_combo_canonical"] == "cvd_rsi_breakout"
