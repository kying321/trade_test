from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_native_group_report.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_native_group_report_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_classify_group_marks_positive_secondary_when_taker_is_ranked_positive() -> None:
    module = _load_module()
    payload = {
        "native_crypto_family": {
            "ranked_combos": [
                {"combo_id": "rsi_breakout", "avg_total_return": 0.04},
                {
                    "combo_id": "taker_oi_ad_rsi_breakout",
                    "avg_total_return": 0.01,
                    "avg_timely_hit_rate": 0.58,
                },
            ]
        }
    }
    result = module.classify_group(payload)
    assert result["status"] == "taker_positive_secondary"
    assert result["taker_best_combo"] == "taker_oi_ad_rsi_breakout"
    assert result["taker_best_combo_canonical"] == "taker_oi_cvd_rsi_breakout"


def test_main_builds_report_from_latest_group_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    majors_path = tmp_path / "20260310T000000Z_binance_indicator_combo_native_crypto.json"
    beta_path = tmp_path / "20260310T000100Z_binance_indicator_combo_native_crypto.json"
    majors_path.write_text(
        json.dumps(
            {
                "symbol_group": "majors",
                "native_crypto_family": {
                    "ranked_combos": [
                        {"combo_id": "rsi_breakout", "avg_total_return": 0.03},
                        {
                            "combo_id": "taker_oi_breakout",
                            "avg_total_return": -0.01,
                            "avg_timely_hit_rate": 0.50,
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    beta_path.write_text(
        json.dumps(
            {
                "symbol_group": "beta",
                "native_crypto_family": {
                    "ranked_combos": [
                        {"combo_id": "rsi_breakout", "avg_total_return": 0.02},
                        {
                            "combo_id": "taker_oi_ad_rsi_breakout",
                            "avg_total_return": 0.01,
                            "avg_timely_hit_rate": 0.62,
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(["--review-dir", str(tmp_path)])
    assert exit_code == 0
    artifacts = sorted(tmp_path.glob("*_binance_indicator_native_group_report.json"))
    assert artifacts
    payload = json.loads(artifacts[-1].read_text(encoding="utf-8"))
    assert payload["groups"]["majors"]["status"] == "taker_ranked_negative"
    assert payload["groups"]["beta"]["status"] == "taker_positive_secondary"
    assert payload["groups"]["beta"]["taker_best_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert "beta-crypto secondary confirmation layer" in payload["overall_takeaway"]


def test_latest_group_artifact_prefers_latest_timestamp_not_mtime(tmp_path: Path) -> None:
    module = _load_module()
    newer = tmp_path / "20260310T120000Z_binance_indicator_combo_native_crypto.json"
    older = tmp_path / "20260310T110000Z_binance_indicator_combo_native_crypto.json"
    newer.write_text(
        json.dumps({"symbol_group": "beta", "native_crypto_family": {"ranked_combos": []}}),
        encoding="utf-8",
    )
    older.write_text(
        json.dumps({"symbol_group": "beta", "native_crypto_family": {"ranked_combos": []}}),
        encoding="utf-8",
    )
    newer.touch()
    older.touch()
    older_path, _ = module.latest_group_artifact(tmp_path, "beta")
    assert older_path == newer
