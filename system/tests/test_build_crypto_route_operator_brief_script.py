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
            "shortline_market_state_brief": "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade",
            "shortline_execution_gate_brief": "4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle",
            "shortline_session_map_brief": "asia_high_low, london_high_low",
            "shortline_no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
            "shortline_cvd_semantic_status": "ok",
            "shortline_cvd_semantic_takeaway": "BNB remains watch-only at the micro layer.",
            "shortline_cvd_queue_handoff_status": "queue-watch-only",
            "shortline_cvd_queue_handoff_takeaway": "Queue remains valid, but latest micro semantics are watch-only.",
            "shortline_cvd_queue_focus_batch": "crypto_beta",
            "shortline_cvd_queue_focus_action": "defer_until_micro_recovers",
            "focus_execution_state": "Bias_Only",
            "focus_execution_blocker_detail": "BNBUSDT remains route-gated via watch_priority_until_long_window_confirms; keep Bias_Only until route quality improves and 4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle completes; no_sweep_no_mss_no_cvd_no_trade.",
            "focus_execution_done_when": "BNBUSDT route improves and reaches Setup_Ready by completing 4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle",
            "focus_execution_micro_classification": "watch_only",
            "focus_execution_micro_context": "failed_auction",
            "focus_execution_micro_trust_tier": "single_exchange_low",
            "focus_execution_micro_veto": "low_sample_or_gap_risk",
            "focus_execution_micro_locality_status": "local_window_ok",
            "focus_execution_micro_drift_risk": "false",
            "focus_execution_micro_attack_side": "buyers",
            "focus_execution_micro_attack_presence": "buyers_attacking",
            "focus_execution_micro_reasons": ["time_sync_risk"],
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
    assert payload["shortline_market_state_brief"] == "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade"
    assert "shortline-market-state: Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade" in payload["operator_lines"]
    assert "focus-execution-state: Bias_Only" in payload["operator_lines"]
    assert any(line.startswith("focus-execution-done-when: BNBUSDT route improves") for line in payload["operator_lines"])
    assert "micro-class: watch_only" in payload["operator_lines"]
    assert "micro-locality: local_window_ok" in payload["operator_lines"]
    assert "micro-drift: false" in payload["operator_lines"]
    assert "micro-attack: buyers:buyers_attacking" in payload["operator_lines"]
    assert "cvd-queue-status: queue-watch-only" in payload["operator_lines"]
    assert payload["crypto_route_head_source_refresh_status"] == "not_active"


def test_build_operator_brief_preserves_focus_review_lane() -> None:
    module = _load_module()
    payload = module.build_operator_brief(
        {
            "operator_status": "deploy-majors-plus-sol-review",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "next_focus_reason": "Flow does not produce a positive ranked edge even in the short sample.",
            "shortline_market_state_brief": "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade",
            "shortline_execution_gate_brief": "4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle",
            "shortline_session_map_brief": "asia_high_low, london_high_low",
            "focus_execution_state": "Bias_Only",
            "focus_execution_blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation.",
            "focus_execution_done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
            "focus_execution_micro_classification": "watch_only",
            "focus_execution_micro_context": "failed_auction",
            "focus_execution_micro_trust_tier": "single_exchange_low",
            "focus_execution_micro_veto": "low_sample_or_gap_risk",
            "focus_execution_micro_locality_status": "local_window_ok",
            "focus_execution_micro_drift_risk": "false",
            "focus_execution_micro_attack_side": "sellers",
            "focus_execution_micro_attack_presence": "sellers_attacking",
            "focus_review_status": "review_no_edge_bias_only_micro_veto",
            "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
            "focus_review_primary_blocker": "no_edge",
            "focus_review_micro_blocker": "low_sample_or_gap_risk",
            "focus_review_blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation. | micro=watch_only:failed_auction:low_sample_or_gap_risk",
            "focus_review_done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
            "focus_review_score_status": "scored",
            "focus_review_edge_score": 5,
            "focus_review_structure_score": 25,
            "focus_review_micro_score": 20,
            "focus_review_composite_score": 17,
            "focus_review_priority_status": "ready",
            "focus_review_priority_score": 17,
            "focus_review_priority_tier": "deprioritized_review",
            "focus_review_priority_brief": "deprioritized_review:17/100",
            "review_priority_queue_status": "ready",
            "review_priority_queue_count": 1,
            "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73",
            "review_priority_head_symbol": "SOLUSDT",
            "review_priority_head_tier": "review_queue_now",
            "review_priority_head_score": 73,
            "review_priority_queue": [
                {
                    "symbol": "SOLUSDT",
                    "route_action": "deprioritize_flow",
                    "priority_tier": "review_queue_now",
                    "rank": 1,
                }
            ],
            "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
        }
    )
    assert payload["focus_review_status"] == "review_no_edge_bias_only_micro_veto"
    assert payload["focus_review_primary_blocker"] == "no_edge"
    assert payload["focus_review_micro_blocker"] == "low_sample_or_gap_risk"
    assert payload["focus_review_edge_score"] == 5
    assert payload["focus_review_composite_score"] == 17
    assert payload["focus_review_priority_status"] == "ready"
    assert payload["focus_review_priority_score"] == 17
    assert payload["focus_review_priority_tier"] == "deprioritized_review"
    assert payload["review_priority_queue_status"] == "ready"
    assert payload["review_priority_queue_count"] == 1
    assert payload["review_priority_head_symbol"] == "SOLUSDT"
    assert payload["review_priority_head_tier"] == "review_queue_now"
    assert payload["review_priority_head_score"] == 73
    assert payload["focus_execution_micro_locality_status"] == "local_window_ok"
    assert payload["focus_execution_micro_drift_risk"] == "false"
    assert payload["focus_execution_micro_attack_side"] == "sellers"
    assert payload["focus_execution_micro_attack_presence"] == "sellers_attacking"
    assert payload["crypto_route_head_source_refresh_status"] == "not_active"
    assert payload["crypto_route_head_source_refresh_brief"] == "not_active:SOLUSDT:-"
    assert payload["crypto_route_head_source_refresh_symbol"] == "SOLUSDT"
    assert payload["crypto_route_head_source_refresh_route_action"] == "deprioritize_flow"
    assert payload["crypto_route_head_downstream_embedding_status"] == "not_assessed"
    assert payload["crypto_route_head_downstream_embedding_brief"] == "not_assessed:SOLUSDT"
    assert "focus-review-status: review_no_edge_bias_only_micro_veto" in payload["operator_lines"]
    assert "focus-review-primary: no_edge" in payload["operator_lines"]
    assert "focus-review-micro: low_sample_or_gap_risk" in payload["operator_lines"]
    assert "focus-review-scores: edge=5 | structure=25 | micro=20 | composite=17" in payload["operator_lines"]
    assert "focus-review-priority: deprioritized_review | score=17" in payload["operator_lines"]
    assert "review-priority-queue: 1:SOLUSDT:review_queue_now:73" in payload["operator_lines"]
    assert "micro-locality: local_window_ok" in payload["operator_lines"]
    assert "micro-drift: false" in payload["operator_lines"]
    assert "micro-attack: sellers:sellers_attacking" in payload["operator_lines"]
    assert "head-source-refresh: not_active:SOLUSDT:-" not in payload["operator_lines"]


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
    hot_research = tmp_path / "20260310T155000Z_hot_universe_research.json"
    route_refresh = tmp_path / "20260310T155300Z_crypto_route_refresh.json"
    hot_research.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    route_refresh.write_text(
        json.dumps(
            {
                "status": "ok",
                "as_of": "2026-03-10T15:53:00Z",
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
                "shortline_market_state_brief": "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade",
                "shortline_execution_gate_brief": "4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle",
                "shortline_session_map_brief": "asia_high_low, london_high_low",
                "shortline_no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
                "shortline_cvd_semantic_status": "ok",
                "shortline_cvd_semantic_takeaway": "BNB remains watch-only at the micro layer.",
                "shortline_cvd_queue_handoff_status": "queue-watch-only",
                "shortline_cvd_queue_handoff_takeaway": "Queue remains valid, but latest micro semantics are watch-only.",
                "shortline_cvd_queue_focus_batch": "crypto_beta",
                "shortline_cvd_queue_focus_action": "defer_until_micro_recovers",
                "focus_execution_state": "Bias_Only",
                "focus_execution_blocker_detail": "BNBUSDT remains route-gated via watch_priority_until_long_window_confirms; keep Bias_Only until route quality improves and 4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle completes; no_sweep_no_mss_no_cvd_no_trade.",
                "focus_execution_done_when": "BNBUSDT route improves and reaches Setup_Ready by completing 4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle",
                "focus_execution_micro_classification": "watch_only",
                "focus_execution_micro_context": "failed_auction",
                "focus_execution_micro_trust_tier": "single_exchange_low",
                "focus_execution_micro_veto": "low_sample_or_gap_risk",
                "focus_execution_micro_reasons": ["time_sync_risk"],
                "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
                "watch_priority_symbols": ["BNBUSDT"],
                "watch_only_symbols": ["SOLUSDT"],
                "review_priority_head_symbol": "BNBUSDT",
                "review_priority_head_tier": "watch_queue_only",
                "review_priority_head_score": 20,
                "review_priority_queue": [
                    {
                        "symbol": "BNBUSDT",
                        "route_action": "watch_priority_until_long_window_confirms",
                        "priority_tier": "watch_queue_only",
                        "rank": 1,
                    }
                ],
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
    assert payload["shortline_market_state_brief"] == "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade"
    assert payload["focus_execution_state"] == "Bias_Only"
    assert payload["focus_execution_done_when"].startswith("BNBUSDT route improves and reaches Setup_Ready")
    assert payload["focus_execution_micro_classification"] == "watch_only"
    assert payload["shortline_cvd_queue_handoff_status"] == "queue-watch-only"
    assert payload["crypto_route_head_source_refresh_status"] == "ready"
    assert payload["crypto_route_head_source_refresh_brief"] == "ready:BNBUSDT:read_current_artifact"
    assert payload["crypto_route_head_source_refresh_symbol"] == "BNBUSDT"
    assert payload["crypto_route_head_source_refresh_action"] == "read_current_artifact"
    assert payload["crypto_route_head_source_refresh_source_kind"] == "crypto_route"
    assert payload["crypto_route_head_source_refresh_source_health"] == "ready"
    assert payload["crypto_route_head_source_refresh_source_artifact"] == str(artifact)
    assert payload["crypto_route_head_source_refresh_priority_tier"] == "watch_queue_only"
    assert payload["crypto_route_head_source_refresh_route_action"] == "watch_priority_until_long_window_confirms"
    assert payload["crypto_route_head_downstream_embedding_status"] == "carry_over_non_blocking"
    assert payload["crypto_route_head_downstream_embedding_brief"] == "carry_over_non_blocking:BNBUSDT"
    assert payload["crypto_route_head_downstream_embedding_artifact"] == str(hot_research)
    assert payload["crypto_route_head_downstream_embedding_as_of"] == "2026-03-10T15:50:00+00:00"
    assert payload["latest_crypto_route_refresh_status"] == "reused_native_inputs"
    assert payload["latest_crypto_route_refresh_brief"] == "reused_native_inputs:skip_native_refresh:2/2"
    assert payload["latest_crypto_route_refresh_artifact"] == str(route_refresh)
    assert payload["latest_crypto_route_refresh_as_of"] == "2026-03-10T15:53:00+00:00"
    assert payload["latest_crypto_route_refresh_native_mode"] == "skip_native_refresh"
    assert payload["latest_crypto_route_refresh_native_step_count"] == 2
    assert payload["latest_crypto_route_refresh_reused_native_count"] == 2
    assert payload["latest_crypto_route_refresh_missing_reused_count"] == 0
    assert payload["latest_crypto_route_refresh_reuse_level"] == "informational"
    assert payload["latest_crypto_route_refresh_reuse_gate_status"] == "reuse_non_blocking"
    assert payload["latest_crypto_route_refresh_reuse_gate_brief"] == "reuse_non_blocking:skip_native_refresh:2/2"
    assert payload["latest_crypto_route_refresh_reuse_gate_blocking"] is False
    assert "head-source-refresh: ready:BNBUSDT:read_current_artifact" in payload["operator_lines"]
    assert "latest-refresh-audit: reused_native_inputs:skip_native_refresh:2/2" in payload["operator_lines"]
    assert "latest-refresh-reuse-gate: reuse_non_blocking:skip_native_refresh:2/2" in payload["operator_lines"]
    assert "downstream-embedding: carry_over_non_blocking:BNBUSDT" in payload["operator_lines"]
    assert payload["artifact"] == str(artifact)
