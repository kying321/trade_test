from __future__ import annotations

import importlib.util
import datetime as dt
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_crypto_route_operator_brief.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("crypto_route_operator_brief_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_build_operator_brief_compresses_focus_gate() -> None:
    module = _load_module()
    payload = module.build_operator_brief(
        {
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
            "next_focus_reason": "BNB degrades on long window.",
            "focus_window_gate": "blocked_until_long_window_confirms",
            "focus_short_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
            "focus_long_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
            "focus_long_top_combo_canonical": "cvd_breakout",
            "focus_window_verdict": "degrades_on_long_window",
            "focus_window_floor": "positive_but_weaker",
            "price_state_window_floor": "negative",
            "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
            "xlong_flow_window_floor": "laggy_positive_only",
            "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
            "focus_brief": "BNB remains the highest-value beta flow watch leg, but long-window confirmation is still missing.",
            "next_retest_action": "rerun_bnb_native_long_window",
            "next_retest_reason": "Confirm whether BNB can keep ranked positive flow after extending the native sample window.",
            "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
            "watch_priority_symbols": ["BNBUSDT"],
            "watch_only_symbols": ["SOLUSDT"],
        }
    )
    assert payload["next_focus_symbol"] == "BNBUSDT"
    assert payload["focus_window_gate"] == "blocked_until_long_window_confirms"
    assert payload["focus_short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert "focus-short-flow: taker_oi_cvd_rsi_breakout" in payload["operator_lines"]
    assert "focus-window: degrades_on_long_window" in payload["operator_lines"]
    assert "focus-window-floor: positive_but_weaker" in payload["operator_lines"]
    assert "price-state-window-floor: negative" in payload["operator_lines"]
    assert any(line.startswith("focus-window-note: Long-window flow holds up better") for line in payload["operator_lines"])
    assert "xlong-flow-floor: laggy_positive_only" in payload["operator_lines"]
    assert any(line.startswith("xlong-flow-note: Extra-long flow keeps a raw positive return") for line in payload["operator_lines"])
    assert payload["next_retest_action"] == "rerun_bnb_native_long_window"
    assert "next-retest: rerun_bnb_native_long_window" in payload["operator_lines"]
    assert payload["focus_brief"].startswith("BNB remains the highest-value beta flow watch leg")


def test_latest_crypto_route_brief_ignores_future_stamped_artifact(tmp_path: Path) -> None:
    module = _load_module()
    module.now_utc = lambda: dt.datetime(2026, 3, 10, 15, 57, tzinfo=dt.timezone.utc)
    current = tmp_path / "20260310T155400Z_crypto_route_brief.json"
    future = tmp_path / "20260310T234100Z_crypto_route_brief.json"
    current.write_text(json.dumps({"marker": "current"}), encoding="utf-8")
    future.write_text(json.dumps({"marker": "future"}), encoding="utf-8")
    assert module.latest_crypto_route_brief(tmp_path) == current


def test_script_writes_operator_brief_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "20260310T155400Z_crypto_route_brief.json"
    source.write_text(
        json.dumps(
            {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_focus_reason": "BNB degrades on long window.",
                "focus_window_gate": "blocked_until_long_window_confirms",
                "focus_short_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
                "focus_long_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
                "focus_long_top_combo_canonical": "cvd_breakout",
                "focus_window_verdict": "degrades_on_long_window",
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
                "focus_brief": "BNB remains the highest-value beta flow watch leg, but long-window confirmation is still missing.",
                "next_retest_action": "rerun_bnb_native_long_window",
                "next_retest_reason": "Confirm whether BNB can keep ranked positive flow after extending the native sample window.",
                "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
                "watch_priority_symbols": ["BNBUSDT"],
                "watch_only_symbols": ["SOLUSDT"],
            }
        ),
        encoding="utf-8",
    )
    rc = module.main(
        [
            "--review-dir",
            str(tmp_path),
            "--now",
            "2026-03-10T15:56:00+00:00",
        ]
    )
    assert rc == 0
    artifact = tmp_path / "20260310T155600Z_crypto_route_operator_brief.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["focus_window_gate"] == "blocked_until_long_window_confirms"
    assert payload["focus_short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["focus_long_top_combo_canonical"] == "cvd_breakout"
    assert payload["focus_window_floor"] == "positive_but_weaker"
    assert payload["price_state_window_floor"] == "negative"
    assert payload["comparative_window_takeaway"].startswith("Long-window flow holds up better")
    assert payload["xlong_flow_window_floor"] == "laggy_positive_only"
    assert payload["xlong_comparative_window_takeaway"].startswith("Extra-long flow keeps a raw positive return")
    assert payload["next_retest_action"] == "rerun_bnb_native_long_window"
    assert payload["focus_brief"].startswith("BNB remains the highest-value beta flow watch leg")
    assert payload["artifact"] == str(artifact)
