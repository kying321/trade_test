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
        {
            "default_market_state": "Bias_Only",
            "setup_ready_state": "Setup_Ready",
            "no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
            "trigger_stack": [
                "4h_profile_location",
                "liquidity_sweep",
                "1m_5m_mss_or_choch",
                "15m_cvd_divergence_or_confirmation",
                "fvg_ob_breaker_retest",
                "15m_reversal_or_breakout_candle",
            ],
            "session_liquidity_map": ["asia_high_low", "london_high_low"],
        },
        None,
        {
            "status": "ok",
            "takeaway": "BNB remains watch-only at the micro layer.",
            "symbols": [
                {
                    "symbol": "BNBUSDT",
                    "classification": "watch_only",
                    "cvd_context_mode": "failed_auction",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "active_reasons": ["time_sync_risk"],
                }
            ],
        },
        {
            "operator_status": "queue-watch-only",
            "takeaway": "Queue remains valid, but latest micro semantics are watch-only.",
            "next_focus_batch": "crypto_beta",
            "next_focus_action": "defer_until_micro_recovers",
            "queue_stack_brief": "crypto_beta -> crypto_hot",
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
    assert payload["shortline_market_state_brief"] == "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade"
    assert payload["shortline_execution_gate_brief"].startswith("4h_profile_location -> liquidity_sweep")
    assert payload["focus_execution_state"] == "Bias_Only"
    assert payload["focus_execution_blocker_detail"].startswith("BNBUSDT remains route-gated")
    assert payload["focus_execution_done_when"].startswith("BNBUSDT route improves and reaches Setup_Ready")
    assert payload["focus_execution_micro_classification"] == "watch_only"


def test_build_brief_appends_environment_blocker_when_micro_capture_is_missing() -> None:
    module = _load_module()
    payload = module.build_brief(
        {
            "operator_status": "crypto-watch",
            "route_stack_brief": "review:SOLUSDT",
            "review_symbols": ["SOLUSDT"],
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "next_focus_reason": "flow review",
            "overall_takeaway": "Keep SOL in review.",
        },
        None,
        {
            "default_market_state": "Bias_Only",
            "setup_ready_state": "Setup_Ready",
            "no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
            "trigger_stack": ["4h_profile_location", "liquidity_sweep"],
            "session_liquidity_map": ["asia_high_low"],
        },
        {
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "blocker_detail": "SOLUSDT remains Bias_Only.",
                    "done_when": "wait for micro recovery",
                    "micro_signals": {
                        "veto_hint": "missing_micro_capture",
                        "cvd_locality_status": "missing_micro_capture",
                        "attack_presence": "missing_micro_capture",
                    },
                }
            ]
        },
        {
            "status": "selected_micro_missing",
            "takeaway": "No usable micro-capture snapshot was available because the current public market data path is environment-blocked.",
            "environment_status": "environment_blocked",
            "environment_classification": "proxy_tls_and_http_access_block",
            "environment_blocker_detail": "proxy_env_present; ssl_verify_failed=binance_spot_public:SOLUSDT",
            "symbols": [],
        },
        None,
    )
    assert payload["shortline_cvd_semantic_environment_status"] == "environment_blocked"
    assert payload["shortline_cvd_semantic_environment_classification"] == "proxy_tls_and_http_access_block"
    assert "env=proxy_tls_and_http_access_block" in payload["focus_execution_blocker_detail"]
    assert "proxy_env_present" in payload["focus_execution_blocker_detail"]
    assert payload["focus_execution_micro_veto"] == "missing_micro_capture"
    assert payload["focus_execution_micro_locality_status"] == "missing_micro_capture"
    assert payload["focus_execution_micro_attack_presence"] == "missing_micro_capture"


def test_build_brief_derives_focus_review_lane_for_deprioritize_flow() -> None:
    module = _load_module()
    payload = module.build_brief(
        {
            "operator_status": "deploy-majors-plus-sol-review",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT",
            "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
            "review_symbols": ["SOLUSDT"],
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "next_focus_reason": "Flow does not produce a positive ranked edge even in the short sample.",
            "overall_takeaway": "Deploy majors and keep SOL under flow review.",
        },
        None,
        {
            "default_market_state": "Bias_Only",
            "setup_ready_state": "Setup_Ready",
            "no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
            "trigger_stack": [
                "4h_profile_location",
                "liquidity_sweep",
                "1m_5m_mss_or_choch",
                "15m_cvd_divergence_or_confirmation",
                "fvg_ob_breaker_retest",
                "15m_reversal_or_breakout_candle",
            ],
            "session_liquidity_map": ["asia_high_low", "london_high_low"],
        },
        {
            "status": "ok",
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation.",
                    "done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
                }
            ],
        },
        {
            "status": "ok",
            "takeaway": "SOL remains watch-only at the micro layer.",
            "time_sync_status": "degraded",
            "time_sync_classification": "fake_ip_dns_intercept",
            "time_sync_intercept_scope": "environment_wide",
            "time_sync_blocker_detail": "intercept_scope=environment_wide; fake_ip_sources=time.google.com,time.cloudflare.com",
            "time_sync_remediation_hint": "disable fake-ip DNS interception or bypass proxy DNS for SNTP/NTP hosts; switching source hostnames is unlikely to help until environment routing is fixed, then rerun time_sync_probe",
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "classification": "watch_only",
                    "cvd_context_mode": "failed_auction",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "low_sample_or_gap_risk",
                    "cvd_locality_status": "proxy_from_current_snapshot",
                    "cvd_drift_risk": False,
                    "cvd_attack_side": "sellers",
                    "cvd_attack_presence": "sellers_attacking",
                    "active_reasons": ["time_sync_risk"],
                }
            ],
        },
        None,
    )
    assert payload["focus_review_status"] == "review_no_edge_bias_only_micro_veto"
    assert payload["focus_review_brief"] == "review_no_edge_bias_only_micro_veto:SOLUSDT"
    assert payload["focus_review_primary_blocker"] == "no_edge"
    assert payload["focus_review_micro_blocker"] == "low_sample_or_gap_risk"
    assert payload["focus_review_edge_score"] == 5
    assert payload["focus_review_structure_score"] == 25
    assert payload["focus_review_micro_score"] == 15
    assert payload["focus_review_composite_score"] == 15
    assert payload["focus_review_priority_status"] == "ready"
    assert payload["focus_review_priority_score"] == 15
    assert payload["focus_review_priority_tier"] == "deprioritized_review"
    assert payload["focus_review_priority_brief"] == "deprioritized_review:15/100"
    assert payload["focus_execution_micro_locality_status"] == "proxy_from_current_snapshot"
    assert payload["focus_execution_micro_drift_risk"] == "false"
    assert payload["focus_execution_micro_attack_side"] == "sellers"
    assert payload["focus_execution_micro_attack_presence"] == "sellers_attacking"
    assert payload["shortline_cvd_semantic_time_sync_classification"] == "fake_ip_dns_intercept"
    assert payload["shortline_cvd_semantic_time_sync_intercept_scope"] == "environment_wide"
    assert payload["shortline_cvd_semantic_time_sync_blocker_detail"] == (
        "intercept_scope=environment_wide; fake_ip_sources=time.google.com,time.cloudflare.com"
    )
    assert "time-sync=fake_ip_dns_intercept:intercept_scope=environment_wide; fake_ip_sources=time.google.com,time.cloudflare.com" in payload[
        "focus_execution_blocker_detail"
    ]
    assert payload["review_priority_queue"][0]["blocker_detail"].endswith(
        "time-sync=fake_ip_dns_intercept:intercept_scope=environment_wide; fake_ip_sources=time.google.com,time.cloudflare.com"
    )
    assert payload["review_priority_queue_status"] == "ready"
    assert payload["review_priority_queue_count"] == 1
    assert payload["review_priority_queue_brief"] == "1:SOLUSDT:review_queue_next:58"
    assert payload["review_priority_head_symbol"] == "SOLUSDT"
    assert payload["review_priority_head_tier"] == "review_queue_next"
    assert payload["review_priority_head_score"] == 58
    assert payload["crypto_route_head_source_refresh_status"] == "not_active"
    assert payload["crypto_route_head_source_refresh_brief"] == "not_active:SOLUSDT:-"
    assert payload["crypto_route_head_source_refresh_symbol"] == "SOLUSDT"
    assert payload["crypto_route_head_source_refresh_route_action"] == "deprioritize_flow"
    assert payload["crypto_route_head_downstream_embedding_status"] == "not_assessed"
    assert payload["crypto_route_head_downstream_embedding_brief"] == "not_assessed:SOLUSDT"
    assert payload["focus_review_done_when"].startswith("SOLUSDT completes liquidity_sweep")
    assert "micro-locality: proxy_from_current_snapshot" in payload["brief_text"]
    assert "micro-drift: false" in payload["brief_text"]
    assert "micro-attack: sellers:sellers_attacking" in payload["brief_text"]
    assert "focus-review: review_no_edge_bias_only_micro_veto | no_edge | low_sample_or_gap_risk" in payload["brief_text"]
    assert "focus-review-scores: edge=5 | structure=25 | micro=15 | composite=15" in payload["brief_text"]
    assert "focus-review-priority: deprioritized_review | score=15" in payload["brief_text"]
    assert "review-priority-queue: 1:SOLUSDT:review_queue_next:58" in payload["brief_text"]
    assert "time-sync: fake_ip_dns_intercept:intercept_scope=environment_wide; fake_ip_sources=time.google.com,time.cloudflare.com" in payload[
        "brief_text"
    ]
    assert "head-source-refresh:" not in payload["brief_text"]


def test_build_brief_rewards_local_cvd_attack_when_level_is_recent() -> None:
    module = _load_module()
    payload = module.build_brief(
        {
            "operator_status": "deploy-majors-plus-sol-review",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT",
            "deploy_now_symbols": ["BTCUSDT", "ETHUSDT"],
            "review_symbols": ["SOLUSDT"],
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "next_focus_reason": "Flow still needs review near key level.",
            "overall_takeaway": "Deploy majors and keep SOL under flow review.",
        },
        None,
        {
            "default_market_state": "Bias_Only",
            "setup_ready_state": "Setup_Ready",
            "no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
            "trigger_stack": [
                "4h_profile_location",
                "liquidity_sweep",
                "1m_5m_mss_or_choch",
                "15m_cvd_divergence_or_confirmation",
                "fvg_ob_breaker_retest",
                "15m_reversal_or_breakout_candle",
            ],
            "session_liquidity_map": ["asia_high_low", "london_high_low"],
        },
        {
            "status": "ok",
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "execution_state": "Bias_Only",
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation.",
                    "done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
                }
            ],
        },
        {
            "status": "ok",
            "takeaway": "SOL has local CVD pressure near the level.",
            "symbols": [
                {
                    "symbol": "SOLUSDT",
                    "classification": "watch_only",
                    "cvd_context_mode": "failed_auction",
                    "cvd_trust_tier_hint": "single_exchange_low",
                    "cvd_veto_hint": "",
                    "cvd_locality_status": "local_window_ok",
                    "cvd_drift_risk": False,
                    "cvd_attack_side": "sellers",
                    "cvd_attack_presence": "sellers_attacking",
                    "active_reasons": ["recent_reversal_zone"],
                }
            ],
        },
        None,
    )
    assert payload["focus_review_micro_score"] == 50
    assert payload["focus_review_composite_score"] == 27
    assert payload["review_priority_head_symbol"] == "SOLUSDT"
    assert payload["review_priority_head_tier"] == "review_queue_now"
    assert payload["review_priority_head_score"] == 88


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
    config = tmp_path / "config.yaml"
    source = review_dir / "20260310T153224Z_binance_indicator_symbol_route_handoff.json"
    bnb_focus = review_dir / "20260310T154122Z_binance_indicator_bnb_flow_focus.json"
    semantic = review_dir / "20260310T154200Z_crypto_cvd_semantic_snapshot.json"
    queue_handoff = review_dir / "20260310T154201Z_crypto_cvd_queue_handoff.json"
    hot_research = review_dir / "20260310T075500Z_hot_universe_research.json"
    route_refresh = review_dir / "20260310T075530Z_crypto_route_refresh.json"
    config.write_text(
        "\n".join(
            [
                "shortline:",
                "  default_market_state: Bias_Only",
                "  setup_ready_state: Setup_Ready",
                "  no_trade_rule: no_sweep_no_mss_no_cvd_no_trade",
                "  trigger_stack: [4h_profile_location, liquidity_sweep, 1m_5m_mss_or_choch, 15m_cvd_divergence_or_confirmation, fvg_ob_breaker_retest, 15m_reversal_or_breakout_candle]",
                "  session_liquidity_map: [asia_high_low, london_high_low]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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
    hot_research.write_text(
        json.dumps(
            {
                "status": "ok",
                "as_of": "2026-03-10T07:55:00+00:00",
                "takeaway": "carry-over research snapshot",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    route_refresh.write_text(
        json.dumps(
            {
                "status": "ok",
                "as_of": "2026-03-10T07:55:30Z",
                "native_refresh_mode": "skip_native_refresh",
                "steps": [
                    {"name": "native_custom", "status": "reused_previous_artifact"},
                    {"name": "native_majors", "status": "reused_previous_artifact"},
                    {"name": "build_source_control_report", "status": "ok"},
                ],
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
    semantic.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "ok",
                "takeaway": "BNB remains watch-only at the micro layer.",
                "symbols": [
                    {
                        "symbol": "BNBUSDT",
                        "classification": "watch_only",
                        "cvd_context_mode": "failed_auction",
                        "cvd_trust_tier_hint": "single_exchange_low",
                        "cvd_veto_hint": "low_sample_or_gap_risk",
                        "active_reasons": ["time_sync_risk"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    queue_handoff.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "ok",
                "operator_status": "queue-watch-only",
                "takeaway": "Queue remains valid, but latest micro semantics are watch-only.",
                "next_focus_batch": "crypto_beta",
                "next_focus_action": "defer_until_micro_recovers",
                "queue_stack_brief": "crypto_beta -> crypto_hot",
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
            "--config",
            str(config),
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
    assert payload["shortline_market_state_brief"] == "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade"
    assert payload["focus_execution_state"] == "Bias_Only"
    assert payload["focus_execution_done_when"].startswith("BNBUSDT route improves and reaches Setup_Ready")
    assert payload["shortline_cvd_semantic_artifact"] == str(semantic)
    assert payload["shortline_cvd_queue_handoff_artifact"] == str(queue_handoff)
    assert payload["focus_execution_micro_classification"] == "watch_only"
    assert payload["shortline_cvd_queue_focus_batch"] == "crypto_beta"
    assert payload["crypto_route_head_source_refresh_status"] == "ready"
    assert payload["crypto_route_head_source_refresh_brief"] == "ready:BNBUSDT:read_current_artifact"
    assert payload["crypto_route_head_source_refresh_symbol"] == "BNBUSDT"
    assert payload["crypto_route_head_source_refresh_action"] == "read_current_artifact"
    assert payload["crypto_route_head_source_refresh_source_kind"] == "crypto_route"
    assert payload["crypto_route_head_source_refresh_source_health"] == "ready"
    assert payload["crypto_route_head_source_refresh_source_artifact"] == str(payload["artifact"])
    assert payload["crypto_route_head_source_refresh_priority_tier"] == "review_queue_next"
    assert payload["crypto_route_head_source_refresh_route_action"] == "watch_priority_until_long_window_confirms"
    assert payload["crypto_route_head_downstream_embedding_status"] == "carry_over_non_blocking"
    assert payload["crypto_route_head_downstream_embedding_brief"] == "carry_over_non_blocking:BNBUSDT"
    assert payload["crypto_route_head_downstream_embedding_artifact"] == str(hot_research)
    assert payload["crypto_route_head_downstream_embedding_as_of"] == "2026-03-10T07:55:00+00:00"
    assert payload["latest_crypto_route_refresh_status"] == "reused_native_inputs"
    assert payload["latest_crypto_route_refresh_brief"] == "reused_native_inputs:skip_native_refresh:2/2"
    assert payload["latest_crypto_route_refresh_artifact"] == str(route_refresh)
    assert payload["latest_crypto_route_refresh_as_of"] == "2026-03-10T07:55:30+00:00"
    assert payload["latest_crypto_route_refresh_native_mode"] == "skip_native_refresh"
    assert payload["latest_crypto_route_refresh_native_step_count"] == 2
    assert payload["latest_crypto_route_refresh_reused_native_count"] == 2
    assert payload["latest_crypto_route_refresh_missing_reused_count"] == 0
    assert payload["latest_crypto_route_refresh_reuse_level"] == "informational"
    assert payload["latest_crypto_route_refresh_reuse_gate_status"] == "reuse_non_blocking"
    assert payload["latest_crypto_route_refresh_reuse_gate_brief"] == "reuse_non_blocking:skip_native_refresh:2/2"
    assert payload["latest_crypto_route_refresh_reuse_gate_blocking"] is False
    assert "head-source-refresh: ready:BNBUSDT:read_current_artifact" in payload["brief_text"]
    assert "latest-refresh-audit: reused_native_inputs:skip_native_refresh:2/2" in payload["brief_text"]
    assert "latest-refresh-reuse-gate: reuse_non_blocking:skip_native_refresh:2/2" in payload["brief_text"]
    assert "downstream-embedding: carry_over_non_blocking:BNBUSDT" in payload["brief_text"]
    assert Path(str(payload["artifact"])).exists()
    assert Path(str(payload["markdown"])).exists()
    assert Path(str(payload["checksum"])).exists()
