from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_source_control_report.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("indicator_source_control_report_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_classify_control_result_marks_partial_recovery_without_leadership() -> None:
    module = _load_module()
    payload = module.classify_control_result(
        {
            "crypto_family": {
                "ranked_combos": [{"combo_id": "rsi_breakout"}],
                "discarded_combos": [{"combo_id": "taker_oi_breakout", "avg_timely_hit_rate": 0.09, "trade_count": 2}],
            }
        },
        {
            "native_crypto_family": {
                "ranked_combos": [
                    {"combo_id": "rsi_breakout"},
                    {"combo_id": "taker_oi_breakout", "avg_timely_hit_rate": 0.58, "trade_count": 37},
                ],
                "discarded_combos": [],
            }
        },
    )
    assert payload["control_verdict"] == "partial_recovery_without_leadership"
    assert payload["native_taker_status"]["status"] == "ranked"
    assert payload["native_taker_status"]["best_combo_canonical"] == "taker_oi_breakout"


def test_render_markdown_includes_verdict_and_top_comparison() -> None:
    module = _load_module()
    md = module.render_markdown(
        {
            "as_of": "2026-03-10T00:00:00+00:00",
            "etf_artifact": "/tmp/etf.json",
            "native_artifact": "/tmp/native.json",
            "control_verdict": "partial_recovery_without_leadership",
            "control_takeaway": "partial recovery",
            "etf_top_combo": "rsi_breakout",
            "native_top_combo": "rsi_breakout",
            "etf_taker_status": {"status": "discarded", "best_combo": "taker_oi_breakout", "best_timely_hit_rate": 0.09},
            "native_taker_status": {"status": "ranked", "best_combo": "taker_oi_breakout", "best_timely_hit_rate": 0.58},
        }
    )
    assert "partial_recovery_without_leadership" in md
    assert "ETF top" in md
    assert "canonical=" in md


def test_latest_artifact_prefers_latest_ok_over_newer_partial_failure(tmp_path: Path) -> None:
    module = _load_module()
    older_ok = tmp_path / "20260312T050000Z_binance_indicator_combo_native_crypto.json"
    newer_partial = tmp_path / "20260312T060000Z_binance_indicator_combo_native_crypto.json"
    older_ok.write_text(json.dumps({"status": "ok", "ok": True}), encoding="utf-8")
    newer_partial.write_text(json.dumps({"status": "partial_failure", "ok": False}), encoding="utf-8")
    assert module.latest_artifact(tmp_path, "binance_indicator_combo_native_crypto") == older_ok
