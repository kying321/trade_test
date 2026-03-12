from __future__ import annotations

import importlib.util
import datetime as dt
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_binance_indicator_symbol_route_handoff.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("symbol_route_handoff_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_classify_handoff_builds_deploy_and_watch_buckets() -> None:
    module = _load_module()
    payload = module.classify_handoff(
        {
            "symbol_routes": {
                "BTCUSDT": {
                    "lane": "majors",
                    "deployment": "price_state_primary_only",
                    "action": "deploy_price_state_only",
                    "reason": "majors stable",
                },
                "ETHUSDT": {
                    "lane": "majors",
                    "deployment": "price_state_primary_only",
                    "action": "deploy_price_state_only",
                    "reason": "majors stable",
                },
                "SOLUSDT": {
                    "lane": "beta",
                    "deployment": "flow_secondary_research_hold",
                    "action": "watch_short_window_flow_only",
                    "reason": "sol short window",
                    "flow_combo": "taker_oi_ad_breakout",
                    "flow_return": 0.004,
                },
                "BNBUSDT": {
                    "lane": "beta",
                    "deployment": "flow_secondary_research_hold",
                    "action": "watch_short_window_flow_priority",
                    "reason": "bnb priority",
                    "flow_combo": "taker_oi_ad_rsi_breakout",
                    "flow_return": 0.016,
                },
            }
        }
    )
    assert payload["deploy_now_symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert payload["watch_priority_symbols"] == ["BNBUSDT"]
    assert payload["watch_only_symbols"] == ["SOLUSDT"]
    assert payload["next_focus_symbol"] == "BNBUSDT"
    assert payload["operator_status"] == "deploy-price-state-plus-beta-watch"
    assert payload["route_stack_brief"] == "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT"
    routes = {row["symbol"]: row for row in payload["routes"]}
    assert routes["BNBUSDT"]["flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert routes["SOLUSDT"]["flow_combo_canonical"] == "taker_oi_cvd_breakout"


def test_classify_handoff_prioritizes_review_candidate_before_watch_only() -> None:
    module = _load_module()
    payload = module.classify_handoff(
        {
            "symbol_routes": {
                "BTCUSDT": {
                    "lane": "majors",
                    "deployment": "price_state_primary_only",
                    "action": "deploy_price_state_only",
                    "reason": "majors stable",
                },
                "ETHUSDT": {
                    "lane": "majors",
                    "deployment": "price_state_primary_only",
                    "action": "deploy_price_state_only",
                    "reason": "majors stable",
                },
                "BNBUSDT": {
                    "lane": "beta",
                    "deployment": "flow_secondary_research_hold",
                    "action": "candidate_flow_secondary",
                    "reason": "bnb review candidate",
                    "flow_combo": "taker_oi_ad_rsi_breakout",
                    "flow_return": 0.016,
                },
                "SOLUSDT": {
                    "lane": "beta",
                    "deployment": "flow_secondary_research_hold",
                    "action": "watch_only",
                    "reason": "sol still watch",
                    "flow_combo": "taker_oi_ad_breakout",
                    "flow_return": 0.004,
                },
            }
        }
    )
    assert payload["review_symbols"] == ["BNBUSDT"]
    assert payload["watch_only_symbols"] == ["SOLUSDT"]
    assert payload["next_focus_symbol"] == "BNBUSDT"
    assert payload["next_focus_action"] == "candidate_flow_secondary"
    assert payload["operator_status"] == "deploy-price-state-plus-beta-review"
    assert payload["route_stack_brief"] == "deploy:BTCUSDT,ETHUSDT | review:BNBUSDT | watch:SOLUSDT"
    assert payload["overall_takeaway"].startswith("Deploy BTC/ETH on price-state only. Review BNB first")
    routes = {row["symbol"]: row for row in payload["routes"]}
    assert routes["BNBUSDT"]["status_label"] == "review"
    assert routes["BNBUSDT"]["route_summary"].startswith("Manual review")


def test_classify_handoff_prefers_window_report_reason_for_beta_routes() -> None:
    module = _load_module()
    payload = module.classify_handoff(
        {
            "symbol_routes": {
                "BNBUSDT": {
                    "lane": "beta",
                    "deployment": "flow_secondary_research_hold",
                    "action": "watch_short_window_flow_priority",
                    "reason": "generic beta reason",
                    "flow_combo": "taker_oi_ad_rsi_breakout",
                    "flow_return": 0.016,
                },
                "SOLUSDT": {
                    "lane": "beta",
                    "deployment": "flow_secondary_research_hold",
                    "action": "watch_short_window_flow_only",
                    "reason": "generic beta reason",
                    "flow_combo": "taker_oi_ad_breakout",
                    "flow_return": 0.004,
                },
            }
        },
        {
            "artifact": "/tmp/window-report.json",
            "legs": {
                "BNBUSDT": {
                    "action": "watch_priority_until_long_window_confirms",
                    "action_reason": "BNB short-window flow degrades on the long window.",
                    "short_flow_combo": "taker_oi_ad_rsi_breakout",
                    "short_flow_combo_canonical": "taker_oi_cvd_rsi_breakout",
                    "long_flow_combo": "crowding_filtered_breakout",
                    "long_flow_combo_canonical": "crowding_filtered_breakout",
                    "short_top_combo": "rsi_breakout",
                    "short_top_combo_canonical": "rsi_breakout",
                    "long_top_combo": "ad_breakout",
                    "long_top_combo_canonical": "cvd_breakout",
                    "flow_window_verdict": "degrades_on_long_window",
                    "price_state_window_verdict": "degrades_on_long_window",
                    "flow_window_floor": "positive_but_weaker",
                    "price_state_window_floor": "negative",
                    "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                },
                "SOLUSDT": {
                    "action": "watch_only",
                    "action_reason": "SOL short-window flow degrades on the long window.",
                    "short_flow_combo": "taker_oi_ad_breakout",
                    "short_flow_combo_canonical": "taker_oi_cvd_breakout",
                    "long_flow_combo": "crowding_filtered_breakout",
                    "long_flow_combo_canonical": "crowding_filtered_breakout",
                    "short_top_combo": "taker_oi_ad_breakout",
                    "short_top_combo_canonical": "taker_oi_cvd_breakout",
                    "long_top_combo": "ad_rsi_breakout",
                    "long_top_combo_canonical": "cvd_rsi_breakout",
                    "flow_window_verdict": "degrades_on_long_window",
                    "price_state_window_verdict": "improves_on_long_window",
                },
            },
        },
    )
    routes = {row["symbol"]: row for row in payload["routes"]}
    assert routes["BNBUSDT"]["reason"] == "BNB short-window flow degrades on the long window."
    assert routes["BNBUSDT"]["flow_window_verdict"] == "degrades_on_long_window"
    assert routes["BNBUSDT"]["flow_window_floor"] == "positive_but_weaker"
    assert routes["BNBUSDT"]["price_state_window_floor"] == "negative"
    assert routes["BNBUSDT"]["comparative_window_takeaway"].startswith("Long-window flow holds up better")
    assert routes["BNBUSDT"]["short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert routes["BNBUSDT"]["long_top_combo_canonical"] == "cvd_breakout"
    assert routes["SOLUSDT"]["reason"] == "SOL short-window flow degrades on the long window."
    assert routes["SOLUSDT"]["short_flow_combo_canonical"] == "taker_oi_cvd_breakout"


def test_classify_handoff_injects_bnb_focus_gate() -> None:
    module = _load_module()
    payload = module.classify_handoff(
        {
            "symbol_routes": {
                "BNBUSDT": {
                    "lane": "beta",
                    "deployment": "flow_secondary_research_hold",
                    "action": "watch_priority_until_long_window_confirms",
                    "reason": "generic beta reason",
                    "flow_combo": "taker_oi_ad_rsi_breakout",
                    "flow_return": 0.016,
                },
            }
        },
        None,
        {
            "artifact": "/tmp/bnb-focus.json",
            "promotion_gate": "blocked_until_long_window_confirms",
            "promotion_gate_reason": "BNB has the best short-window beta flow, but it degrades when the window is extended.",
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
        },
    )
    route = payload["routes"][0]
    assert route["promotion_gate"] == "blocked_until_long_window_confirms"
    assert route["focus_window_verdict"] == "degrades_on_long_window"
    assert route["focus_window_floor"] == "positive_but_weaker"
    assert route["price_state_window_floor"] == "negative"
    assert route["focus_short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert route["focus_long_top_combo_canonical"] == "cvd_breakout"
    assert route["comparative_window_takeaway"].startswith("Long-window flow holds up better")
    assert route["xlong_flow_window_floor"] == "laggy_positive_only"
    assert route["xlong_comparative_window_takeaway"].startswith("Extra-long flow keeps a raw positive return")
    assert route["reason"] == "BNB has the best short-window beta flow, but it degrades when the window is extended."
    assert payload["focus_window_gate"] == "blocked_until_long_window_confirms"
    assert payload["focus_short_flow_combo_canonical"] == "taker_oi_cvd_rsi_breakout"
    assert payload["focus_long_top_combo_canonical"] == "cvd_breakout"
    assert payload["focus_window_floor"] == "positive_but_weaker"
    assert payload["price_state_window_floor"] == "negative"
    assert payload["xlong_flow_window_floor"] == "laggy_positive_only"
    assert payload["bnb_flow_focus_artifact"] == "/tmp/bnb-focus.json"


def test_latest_combo_playbook_prefers_latest_timestamp_not_mtime(tmp_path: Path) -> None:
    module = _load_module()
    older = tmp_path / "20260310T152146Z_binance_indicator_combo_playbook.json"
    newer = tmp_path / "20260310T152245Z_binance_indicator_combo_playbook.json"
    older.write_text(json.dumps({"marker": "older"}), encoding="utf-8")
    newer.write_text(json.dumps({"marker": "newer"}), encoding="utf-8")
    older.touch()
    path = module.latest_combo_playbook(tmp_path)
    assert path == newer


def test_latest_combo_playbook_ignores_future_stamped_artifact(tmp_path: Path) -> None:
    module = _load_module()
    module.now_utc = lambda: dt.datetime(2026, 3, 10, 15, 58, tzinfo=dt.timezone.utc)
    current = tmp_path / "20260310T154509Z_binance_indicator_combo_playbook.json"
    future = tmp_path / "20260310T234509Z_binance_indicator_combo_playbook.json"
    current.write_text(json.dumps({"marker": "current"}), encoding="utf-8")
    future.write_text(json.dumps({"marker": "future"}), encoding="utf-8")
    assert module.latest_combo_playbook(tmp_path) == current


def test_latest_bnb_flow_focus_prefers_richer_direct_native_artifact(tmp_path: Path) -> None:
    module = _load_module()
    older_rich = tmp_path / "20260310T094600Z_binance_indicator_bnb_flow_focus.json"
    newer_sparse = tmp_path / "20260310T172000Z_binance_indicator_bnb_flow_focus.json"
    older_rich.write_text(
        json.dumps(
            {
                "source_mode": "direct_bnb_native",
                "flow_window_floor": "positive_but_weaker",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
            }
        ),
        encoding="utf-8",
    )
    newer_sparse.write_text(
        json.dumps(
            {
                "brief": "older sparse read model",
            }
        ),
        encoding="utf-8",
    )
    path, payload = module.latest_bnb_flow_focus(tmp_path)
    assert path == older_rich
    assert payload is not None
    assert payload["source_mode"] == "direct_bnb_native"


def test_classify_route_maps_long_window_watch_actions_to_watch_labels() -> None:
    module = _load_module()
    bnb = module.classify_route(
        "BNBUSDT",
        {
            "lane": "beta",
            "deployment": "flow_secondary_research_hold",
            "action": "watch_priority_until_long_window_confirms",
            "reason": "bnb window reason",
        },
    )
    sol = module.classify_route(
        "SOLUSDT",
        {
            "lane": "beta",
            "deployment": "flow_secondary_research_hold",
            "action": "watch_only",
            "reason": "sol window reason",
        },
    )
    assert bnb["status_label"] == "watch_priority"
    assert sol["status_label"] == "watch_only"
