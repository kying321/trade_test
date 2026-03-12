from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_route_brief.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("crypto_route_brief_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_build_brief_extracts_compact_route_summary() -> None:
    module = _load_module()
    payload = module.build_brief(
        {
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
            "watch_priority_symbols": ["BNBUSDT"],
            "watch_only_symbols": ["SOLUSDT"],
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_short_window_flow_priority",
            "next_focus_reason": "beta flow is the highest-value watch leg",
            "overall_takeaway": "Deploy majors and keep beta in watch mode.",
        },
        {
            "promotion_gate": "blocked_until_long_window_confirms",
            "promotion_gate_reason": "BNB still needs longer-window confirmation before promotion.",
            "short_flow_combo": "taker_oi_ad_rsi_breakout",
            "short_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
            "long_flow_combo": "taker_oi_ad_rsi_breakout",
            "long_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
            "long_top_combo": "ad_breakout",
            "long_top_combo_canonical": "cvd_breakout",
            "flow_window_verdict": "degrades_on_long_window",
            "flow_window_floor": "positive_but_weaker",
            "price_state_window_floor": "negative",
            "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
            "xlong_flow_window_floor": "laggy_positive_only",
            "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
            "brief": "BNB remains the highest-value beta flow watch leg, but long-window confirmation is still missing.",
            "next_retest_action": "rerun_bnb_native_long_window",
            "next_retest_reason": "Confirm whether BNB can keep ranked positive flow after extending the native sample window.",
        },
    )
    assert payload["deploy_count"] == 2
    assert payload["watch_priority_count"] == 1
    assert payload["watch_only_count"] == 1
    assert payload["next_focus_symbol"] == "BNBUSDT"
    assert "focus: BNBUSDT" in payload["brief_text"]
    assert payload["focus_window_gate"] == "blocked_until_long_window_confirms"
    assert payload["focus_short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["focus_long_top_combo_canonical"] == "cvd_breakout"
    assert "focus-short-flow: taker_oi_cvd_rsi_breakout" in payload["brief_text"]
    assert "focus-gate: blocked_until_long_window_confirms" in payload["brief_text"]
    assert payload["focus_window_floor"] == "positive_but_weaker"
    assert payload["price_state_window_floor"] == "negative"
    assert payload["xlong_flow_window_floor"] == "laggy_positive_only"
    assert "focus-window-floor: positive_but_weaker" in payload["brief_text"]
    assert "price-window-floor: negative" in payload["brief_text"]
    assert payload["comparative_window_takeaway"].startswith("Long-window flow holds up better")
    assert payload["xlong_comparative_window_takeaway"].startswith("Extra-long flow keeps a raw positive return")
    assert "xlong-flow-floor: laggy_positive_only" in payload["brief_text"]
    assert payload["next_retest_action"] == "rerun_bnb_native_long_window"
    assert "next-retest: rerun_bnb_native_long_window" in payload["brief_text"]
    assert payload["focus_brief"].startswith("BNB remains the highest-value beta flow watch leg")


def test_latest_symbol_route_handoff_prefers_latest_timestamp_not_mtime(tmp_path: Path) -> None:
    module = _load_module()
    older = tmp_path / "20260310T152556Z_binance_indicator_symbol_route_handoff.json"
    newer = tmp_path / "20260310T153224Z_binance_indicator_symbol_route_handoff.json"
    older.write_text(json.dumps({"marker": "older"}), encoding="utf-8")
    newer.write_text(json.dumps({"marker": "newer"}), encoding="utf-8")
    older.touch()
    assert module.latest_symbol_route_handoff(tmp_path) == newer


def test_script_writes_artifacts(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    source = review_dir / "20260310T153224Z_binance_indicator_symbol_route_handoff.json"
    bnb_focus = review_dir / "20260310T154122Z_binance_indicator_bnb_flow_focus.json"
    source.write_text(
        json.dumps(
            {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
                "watch_priority_symbols": ["BNBUSDT"],
                "watch_only_symbols": ["SOLUSDT"],
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_short_window_flow_priority",
                "next_focus_reason": "bnb first",
                "overall_takeaway": "Deploy majors and keep beta in watch mode.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    bnb_focus.write_text(
        json.dumps(
            {
                "promotion_gate": "blocked_until_long_window_confirms",
                "promotion_gate_reason": "BNB still needs longer-window confirmation before promotion.",
                "short_flow_combo": "taker_oi_ad_rsi_breakout",
                "short_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
                "long_flow_combo": "taker_oi_ad_rsi_breakout",
                "long_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
                "long_top_combo": "ad_breakout",
                "long_top_combo_canonical": "cvd_breakout",
                "flow_window_verdict": "degrades_on_long_window",
                "flow_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
                "brief": "BNB remains the highest-value beta flow watch leg, but long-window confirmation is still missing.",
                "next_retest_action": "rerun_bnb_native_long_window",
                "next_retest_reason": "Confirm whether BNB can keep ranked positive flow after extending the native sample window.",
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
            "2026-03-10T16:00:00+08:00",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["next_focus_symbol"] == "BNBUSDT"
    assert payload["focus_window_gate"] == "blocked_until_long_window_confirms"
    assert payload["focus_short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["focus_long_top_combo_canonical"] == "cvd_breakout"
    assert payload["focus_window_floor"] == "positive_but_weaker"
    assert payload["price_state_window_floor"] == "negative"
    assert payload["xlong_flow_window_floor"] == "laggy_positive_only"
    assert payload["bnb_flow_focus_artifact"] == str(bnb_focus)
    assert payload["next_retest_action"] == "rerun_bnb_native_long_window"
    assert payload["focus_brief"].startswith("BNB remains the highest-value beta flow watch leg")
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()
