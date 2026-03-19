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
            "latest_crypto_route_refresh_status": "reused_native_inputs",
            "latest_crypto_route_refresh_brief": "reused_native_inputs:skip_native_refresh:2/2",
            "latest_crypto_route_refresh_artifact": "/tmp/refresh.json",
            "latest_crypto_route_refresh_native_mode": "skip_native_refresh",
            "latest_crypto_route_refresh_reused_native_count": 2,
            "latest_crypto_route_refresh_native_step_count": 2,
            "latest_crypto_route_refresh_reuse_gate_brief": "reuse_non_blocking:skip_native_refresh:2/2",
            "latest_crypto_route_refresh_reuse_level": "informational",
        }
    )
    assert "partial_recovery_without_leadership" in md
    assert "ETF top" in md
    assert "canonical=" in md
    assert "Latest Crypto Route Refresh Audit" in md
    assert "reused_native_inputs:skip_native_refresh:2/2" in md
    assert "reuse_non_blocking:skip_native_refresh:2/2" in md


def test_latest_artifact_prefers_latest_ok_over_newer_partial_failure(tmp_path: Path) -> None:
    module = _load_module()
    older_ok = tmp_path / "20260312T050000Z_binance_indicator_combo_native_crypto.json"
    newer_partial = tmp_path / "20260312T060000Z_binance_indicator_combo_native_crypto.json"
    older_ok.write_text(json.dumps({"status": "ok", "ok": True}), encoding="utf-8")
    newer_partial.write_text(json.dumps({"status": "partial_failure", "ok": False}), encoding="utf-8")
    assert module.latest_artifact(tmp_path, "binance_indicator_combo_native_crypto") == older_ok


def test_latest_artifact_respects_reference_now_for_future_stamps(tmp_path: Path) -> None:
    module = _load_module()
    module.now_utc = lambda: module.parse_now("2026-03-12T13:00:00Z")
    current = tmp_path / "20260312T124000Z_binance_indicator_combo_native_crypto.json"
    future = tmp_path / "20260312T142000Z_binance_indicator_combo_native_crypto.json"
    current.write_text(json.dumps({"status": "ok", "ok": True}), encoding="utf-8")
    future.write_text(json.dumps({"status": "ok", "ok": True}), encoding="utf-8")
    assert module.latest_artifact(tmp_path, "binance_indicator_combo_native_crypto") == current
    ref_now = module.parse_now("2026-03-12T14:25:00Z")
    assert module.latest_artifact(tmp_path, "binance_indicator_combo_native_crypto", ref_now) == future


def test_crypto_route_refresh_audit_lane_reports_reused_native_inputs(tmp_path: Path) -> None:
    module = _load_module()
    refresh = tmp_path / "20260312T114514Z_crypto_route_refresh.json"
    refresh.write_text(
        json.dumps(
            {
                "status": "ok",
                "as_of": "2026-03-12T11:45:14+00:00",
                "native_refresh_mode": "skip_native_refresh",
                "steps": [
                    {"name": "native_custom", "status": "reused_previous_artifact"},
                    {"name": "native_majors", "status": "reused_previous_artifact"},
                    {"name": "build_source_control_report", "status": "ok"},
                ],
            }
        ),
        encoding="utf-8",
    )
    audit = module._crypto_route_refresh_audit_lane(tmp_path)
    assert audit["status"] == "reused_native_inputs"
    assert audit["brief"] == "reused_native_inputs:skip_native_refresh:2/2"
    assert audit["artifact"] == str(refresh)
    assert audit["reused_native_count"] == 2
    assert audit["missing_reused_count"] == 0


def test_crypto_route_refresh_reuse_gate_marks_reused_inputs_non_blocking() -> None:
    module = _load_module()
    gate = module._crypto_route_refresh_reuse_gate(
        {
            "status": "reused_native_inputs",
            "brief": "reused_native_inputs:skip_native_refresh:2/2",
            "native_mode": "skip_native_refresh",
            "reused_native_count": 2,
            "native_step_count": 2,
        }
    )
    assert gate["level"] == "informational"
    assert gate["status"] == "reuse_non_blocking"
    assert gate["brief"] == "reuse_non_blocking:skip_native_refresh:2/2"
    assert gate["blocking"] is False
