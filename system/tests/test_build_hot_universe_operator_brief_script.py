from __future__ import annotations

import datetime as dt
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_hot_universe_operator_brief.py")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_module():
    spec = importlib.util.spec_from_file_location("hot_universe_operator_brief_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_crypto_route_head_source_refresh_prefers_route_lane_when_current_artifact_is_readable() -> None:
    module = _load_module()
    payload = module._crypto_route_head_source_refresh_lane(
        row={
            "slot": "secondary",
            "symbol": "SOLUSDT",
            "source_refresh_action": "read_current_artifact",
            "source_kind": "crypto_route",
            "source_status": "ok",
            "source_recency": "fresh",
            "source_health": "ready",
            "source_artifact": "/tmp/fallback.json",
        },
        reference_now=dt.datetime(2026, 3, 12, 10, 30, tzinfo=dt.timezone.utc),
        source_payload={
            "crypto_route_head_source_refresh_status": "ready",
            "crypto_route_head_source_refresh_brief": "ready:SOLUSDT:read_current_artifact",
            "crypto_route_head_source_refresh_symbol": "SOLUSDT",
            "crypto_route_head_source_refresh_action": "read_current_artifact",
            "crypto_route_head_source_refresh_source_kind": "crypto_route",
            "crypto_route_head_source_refresh_source_health": "ready",
            "crypto_route_head_source_refresh_source_artifact": "/tmp/preferred.json",
            "crypto_route_head_source_refresh_blocker_detail": "preferred route lane blocker",
            "crypto_route_head_source_refresh_done_when": "preferred route lane done_when",
        },
    )
    assert payload["status"] == "ready"
    assert payload["brief"] == "ready:SOLUSDT:read_current_artifact"
    assert payload["slot"] == "secondary"
    assert payload["symbol"] == "SOLUSDT"
    assert payload["source_artifact"] == "/tmp/preferred.json"
    assert payload["blocker_detail"] == "preferred route lane blocker"
    assert payload["done_when"] == "preferred route lane done_when"


def test_source_refresh_pipeline_relevance_is_non_blocking_when_crypto_head_is_ready() -> None:
    module = _load_module()
    payload = module._source_refresh_pipeline_relevance(
        crypto_head_source_refresh={
            "status": "ready",
            "symbol": "SOLUSDT",
            "action": "read_current_artifact",
        },
        pending_steps=[{"rank": 3}],
        deferred_steps=[],
    )
    assert payload["status"] == "non_blocking_for_current_crypto_head"
    assert payload["brief"] == "non_blocking_for_current_crypto_head:SOLUSDT:1"
    assert "does not block the current crypto head" in payload["blocker_detail"]


def test_latest_crypto_route_refresh_source_ignores_future_grace_when_reference_now_provided(
    tmp_path: Path,
) -> None:
    module = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    past_path = review_dir / "20260312T121541Z_crypto_route_refresh.json"
    future_path = review_dir / "20260312T122014Z_crypto_route_refresh.json"
    _write_json(past_path, {"status": "ok"})
    _write_json(future_path, {"status": "ok"})

    selected = module.latest_crypto_route_refresh_source(
        review_dir,
        dt.datetime(2026, 3, 12, 12, 16, 18, tzinfo=dt.timezone.utc),
    )

    assert selected == past_path


def test_build_hot_universe_operator_brief_prefers_non_dry_run(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T100000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all", "precious_metals"],
                "focus_with_regime_filter_batches": ["energy_liquids"],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": ["commodities_benchmark"],
                "avoid_batches": ["energy_gas"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_focus_reason": "BNB degrades on long window.",
                "focus_window_gate": "blocked_until_long_window_confirms",
                "focus_window_verdict": "degrades_on_long_window",
                "focus_brief": "BNB still needs long-window confirmation.",
                "next_retest_action": "rerun_bnb_native_long_window",
                "next_retest_reason": "Retest BNB on a longer native sample.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T110000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {"focus_primary_batches": []},
            "crypto_route_brief": {"operator_status": "watch-all"},
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_status"] == "ok"
    assert payload["source_mode"] == "single-hot-universe-source"
    assert payload["focus_primary_batches"] == ["metals_all", "precious_metals"]
    assert payload["research_queue_batches"] == ["crypto_hot"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"
    assert payload["crypto_next_retest_action"] == "rerun_bnb_native_long_window"
    assert "primary: metals_all, precious_metals" in payload["summary_text"]
    assert "research-queue: crypto_hot" in payload["summary_text"]


def test_build_hot_universe_operator_brief_falls_back_to_rich_dry_run(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T100000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"focus_primary_batches": []},
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260310T110000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": ["energy_liquids"],
                "research_queue_batches": ["crypto_majors"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_focus_reason": "BNB degrades on long window.",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_status"] == "dry_run"
    assert payload["source_mode"] == "single-hot-universe-source"
    assert payload["focus_primary_batches"] == ["metals_all"]
    assert payload["research_queue_batches"] == ["crypto_majors"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"


def test_build_hot_universe_operator_brief_surfaces_brooks_route_and_execution_plan(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260313T040000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "research_queue_batches": ["crypto_hot"],
            },
            "crypto_route_brief": {
                "operator_status": "review-flow-and-deploy-price-state",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT | watch:BNBUSDT",
                "next_focus_symbol": "SOLUSDT",
                "next_focus_action": "deprioritize_flow",
                "next_focus_reason": "Flow does not produce a positive ranked edge even in the short sample.",
            },
        },
    )
    _write_json(
        review_dir / "20260313T041500Z_brooks_price_action_route_report.json",
        {
            "status": "ok",
            "as_of": "2026-03-13T04:15:00+00:00",
            "selected_routes_brief": (
                "equity:three_push_climax_reversal | etf:breakout_pullback_resume | future:breakout_pullback_resume/second_entry_trend_continuation"
            ),
            "candidate_count": 1,
            "current_candidates": [
                {
                    "symbol": "SC2603",
                    "strategy_id": "second_entry_trend_continuation",
                    "direction": "LONG",
                    "route_bridge_status": "manual_structure_route",
                    "route_bridge_blocker_detail": "manual structure lane only",
                    "route_selection_score": 81.68716356888629,
                    "signal_score": 80,
                    "signal_age_bars": 2,
                },
                {
                    "symbol": "AU2406",
                    "strategy_id": "breakout_pullback_resume",
                    "direction": "SHORT",
                    "route_bridge_status": "blocked_shortline_gate",
                    "route_bridge_blocker_detail": "lower timeframe confirmation still missing",
                    "route_selection_score": 64.125,
                    "signal_score": 61,
                    "signal_age_bars": 4,
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260313T042000Z_brooks_price_action_execution_plan.json",
        {
            "status": "ok",
            "as_of": "2026-03-13T04:20:00+00:00",
            "actionable_count": 1,
            "blocked_count": 0,
            "head_plan_item": {
                "symbol": "SC2603",
                "strategy_id": "second_entry_trend_continuation",
                "plan_status": "manual_structure_review_now",
                "execution_action": "review_manual_stop_entry",
                "entry_price": 478.8,
                "stop_price": 471.0336,
                "target_price": 495.8861,
                "rr_ratio": 2.2,
                "plan_blocker_detail": (
                    "Structure route is valid, but this asset class has no automated execution bridge in-system."
                ),
            },
            "plan_items": [
                {
                    "symbol": "SC2603",
                    "asset_class": "future",
                    "direction": "LONG",
                    "strategy_id": "second_entry_trend_continuation",
                    "plan_status": "manual_structure_review_now",
                    "execution_action": "review_manual_stop_entry",
                    "route_selection_score": 81.68716356888629,
                    "signal_score": 80,
                    "signal_age_bars": 2,
                    "entry_price": 478.8,
                    "stop_price": 471.0336,
                    "target_price": 495.8861,
                    "rr_ratio": 2.2,
                    "plan_blocker_detail": (
                        "Structure route is valid, but this asset class has no automated execution bridge in-system."
                    ),
                    "plan_done_when": (
                        "manual trader confirms venue, sizing, and lower-timeframe trigger before placing a discretionary order"
                    ),
                },
                {
                    "symbol": "AU2406",
                    "asset_class": "future",
                    "direction": "SHORT",
                    "strategy_id": "breakout_pullback_resume",
                    "plan_status": "blocked_shortline_gate",
                    "execution_action": "wait_for_shortline_setup_ready",
                    "route_selection_score": 64.125,
                    "signal_score": 61,
                    "signal_age_bars": 4,
                    "entry_price": 551.2,
                    "stop_price": 556.9,
                    "target_price": 538.0,
                    "rr_ratio": 2.32,
                    "plan_blocker_detail": "lower timeframe confirmation still missing",
                    "plan_done_when": "shortline gate stack completes",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260313T042100Z_brooks_structure_review_queue.json",
        {
            "status": "ok",
            "as_of": "2026-03-13T04:21:00+00:00",
            "review_status": "ready",
            "review_brief": "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now",
            "queue_status": "ready",
            "queue_count": 2,
            "queue_brief": (
                "1:SC2603:review_queue_now:manual_structure_review_now:96 | "
                "2:AU2406:blocked_queue:blocked_shortline_gate:53"
            ),
            "priority_status": "ready",
            "priority_brief": "ready:SC2603:96:review_queue_now",
            "queue": [
                {
                    "rank": 1,
                    "symbol": "SC2603",
                    "strategy_id": "second_entry_trend_continuation",
                    "direction": "LONG",
                    "tier": "review_queue_now",
                    "plan_status": "manual_structure_review_now",
                    "execution_action": "review_manual_stop_entry",
                    "route_selection_score": 81.68716356888629,
                    "signal_score": 80,
                    "signal_age_bars": 2,
                    "priority_score": 96,
                    "priority_tier": "review_queue_now",
                    "blocker_detail": (
                        "Structure route is valid, but this asset class has no automated execution bridge in-system."
                    ),
                    "done_when": (
                        "manual trader confirms venue, sizing, and lower-timeframe trigger before placing a discretionary order"
                    ),
                },
                {
                    "rank": 2,
                    "symbol": "AU2406",
                    "strategy_id": "breakout_pullback_resume",
                    "direction": "SHORT",
                    "tier": "blocked_queue",
                    "plan_status": "blocked_shortline_gate",
                    "execution_action": "wait_for_shortline_setup_ready",
                    "route_selection_score": 64.125,
                    "signal_score": 61,
                    "signal_age_bars": 4,
                    "priority_score": 53,
                    "priority_tier": "blocked_review",
                    "blocker_detail": "lower timeframe confirmation still missing",
                    "done_when": "shortline gate stack completes",
                },
            ],
            "head": {
                "rank": 1,
                "symbol": "SC2603",
                "strategy_id": "second_entry_trend_continuation",
                "direction": "LONG",
                "tier": "review_queue_now",
                "plan_status": "manual_structure_review_now",
                "execution_action": "review_manual_stop_entry",
                "route_selection_score": 81.68716356888629,
                "signal_score": 80,
                "signal_age_bars": 2,
                "priority_score": 96,
                "priority_tier": "review_queue_now",
                "blocker_detail": (
                    "Structure route is valid, but this asset class has no automated execution bridge in-system."
                ),
                "done_when": (
                    "manual trader confirms venue, sizing, and lower-timeframe trigger before placing a discretionary order"
                ),
            },
            "blocker_detail": (
                "Structure route is valid, but this asset class has no automated execution bridge in-system."
            ),
            "done_when": (
                "manual trader confirms venue, sizing, and lower-timeframe trigger before placing a discretionary order"
            ),
        },
    )
    _write_json(
        review_dir / "20260313T042200Z_brooks_structure_refresh.json",
        {
            "status": "ok",
            "as_of": "2026-03-13T04:22:00+00:00",
            "review_status": "ready",
            "review_brief": "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now",
            "queue_count": 2,
            "head_symbol": "SC2603",
            "head_action": "review_manual_stop_entry",
            "head_priority_score": 96,
        },
    )
    _write_json(
        review_dir / "20260313T042300Z_cross_market_operator_state.json",
        {
            "status": "ok",
            "as_of": "2026-03-13T04:23:00+00:00",
            "operator_backlog_status": "ready",
            "operator_backlog_count": 4,
            "operator_backlog_brief": (
                "1:waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
                "2:waiting:commodity_fill_evidence:XAGUSD:wait_for_paper_execution_fill_evidence:98 | "
                "3:review:brooks_structure:SC2603:review_manual_stop_entry:96 | "
                "4:review:crypto_route:SOLUSDT:deprioritize_flow:73"
            ),
            "operator_backlog_state_brief": "waiting=2 | review=2 | watch=0 | blocked=0",
            "operator_backlog_priority_totals_brief": "waiting=197 | review=169 | watch=0 | blocked=0",
            "operator_state_lane_heads_brief": (
                "waiting:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
                "review:SC2603:review_manual_stop_entry:96 | "
                "watch:-:-:0 | "
                "blocked:-:-:0"
            ),
            "operator_state_lane_priority_order_brief": (
                "waiting@197:2 > review@169:2 > watch@0:0 > blocked@0:0"
            ),
            "remote_live_operator_alignment_status": "local_operator_active_remote_live_blocked",
            "remote_live_operator_alignment_brief": (
                "local_operator_active_remote_live_blocked:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked"
            ),
            "remote_live_operator_alignment_blocker_detail": (
                "Current local operator head is XAUUSD (commodity_execution_close_evidence/"
                "wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote "
                "automated live remains profitability_confirmed_but_auto_live_blocked."
            ),
            "remote_live_operator_alignment_done_when": (
                "XAUUSD local operator head progresses and remote auto-live blockers are cleared"
            ),
            "remote_live_takeover_gate_status": "blocked_by_remote_live_gate",
            "remote_live_takeover_gate_brief": (
                "blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked"
            ),
            "remote_live_takeover_gate_blocker_detail": (
                "Current local operator head is XAUUSD (commodity_execution_close_evidence/"
                "wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote "
                "automated live remains profitability_confirmed_but_auto_live_blocked."
            ),
            "remote_live_takeover_gate_done_when": (
                "XAUUSD local operator head progresses and remote auto-live blockers are cleared"
            ),
            "operator_head_area": "commodity_execution_close_evidence",
            "operator_head_symbol": "XAUUSD",
            "operator_head_action": "wait_for_paper_execution_close_evidence",
            "operator_head_state": "waiting",
            "operator_head_priority_score": 99,
            "operator_head_priority_tier": "waiting_now",
            "operator_waiting_lane_status": "ready",
            "operator_waiting_lane_count": 2,
            "operator_waiting_lane_brief": (
                "1:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
                "2:XAGUSD:wait_for_paper_execution_fill_evidence:98"
            ),
            "operator_waiting_lane_priority_total": 197,
            "operator_waiting_lane_head_symbol": "XAUUSD",
            "operator_waiting_lane_head_action": "wait_for_paper_execution_close_evidence",
            "operator_waiting_lane_head_priority_score": 99,
            "operator_waiting_lane_head_priority_tier": "waiting_now",
            "operator_review_lane_status": "ready",
            "operator_review_lane_count": 2,
            "operator_review_lane_brief": (
                "3:SC2603:review_manual_stop_entry:96 | 4:SOLUSDT:deprioritize_flow:73"
            ),
            "operator_review_lane_priority_total": 169,
            "operator_review_lane_head_symbol": "SC2603",
            "operator_review_lane_head_action": "review_manual_stop_entry",
            "operator_review_lane_head_priority_score": 96,
            "operator_review_lane_head_priority_tier": "review_queue_now",
            "operator_watch_lane_status": "inactive",
            "operator_watch_lane_count": 0,
            "operator_watch_lane_brief": "-",
            "operator_watch_lane_priority_total": 0,
            "operator_watch_lane_head_symbol": "",
            "operator_watch_lane_head_action": "",
            "operator_watch_lane_head_priority_score": 0,
            "operator_watch_lane_head_priority_tier": "",
            "operator_blocked_lane_status": "inactive",
            "operator_blocked_lane_count": 0,
            "operator_blocked_lane_brief": "-",
            "operator_blocked_lane_priority_total": 0,
            "operator_blocked_lane_head_symbol": "",
            "operator_blocked_lane_head_action": "",
            "operator_blocked_lane_head_priority_score": 0,
            "operator_blocked_lane_head_priority_tier": "",
            "operator_head": {
                "rank": 1,
                "area": "commodity_execution_close_evidence",
                "symbol": "XAUUSD",
                "action": "wait_for_paper_execution_close_evidence",
                "state": "waiting",
                "priority_score": 99,
                "priority_tier": "waiting_now",
                "blocker_detail": "paper execution evidence is present, but position is still OPEN",
                "done_when": "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available",
            },
            "review_backlog_status": "ready",
            "review_backlog_count": 2,
            "review_backlog_brief": (
                "1:brooks_structure:SC2603:review_queue_now:96 | 2:crypto_route:SOLUSDT:review_queue_now:73"
            ),
            "review_head_area": "brooks_structure",
            "review_head_symbol": "SC2603",
            "review_head_action": "review_manual_stop_entry",
            "review_head_priority_score": 96,
            "review_head_priority_tier": "review_queue_now",
            "review_head": {
                "rank": 1,
                "area": "brooks_structure",
                "symbol": "SC2603",
                "action": "review_manual_stop_entry",
                "status": "ready",
                "priority_score": 96,
                "priority_tier": "review_queue_now",
                "blocker_detail": "manual-only structure route still requires venue confirmation",
                "done_when": "cross-market review head is executed manually, promoted, or invalidated",
            },
            "review_backlog": [
                {
                    "rank": 1,
                    "area": "brooks_structure",
                    "symbol": "SC2603",
                    "action": "review_manual_stop_entry",
                    "status": "ready",
                    "priority_score": 96,
                    "priority_tier": "review_queue_now",
                    "blocker_detail": "manual-only structure route still requires venue confirmation",
                    "done_when": "cross-market review head is executed manually, promoted, or invalidated",
                },
                {
                    "rank": 2,
                    "area": "crypto_route",
                    "symbol": "SOLUSDT",
                    "action": "deprioritize_flow",
                    "status": "review",
                    "priority_score": 73,
                    "priority_tier": "review_queue_now",
                    "blocker_detail": "SOLUSDT remains Bias_Only",
                    "done_when": "micro gate recovers",
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260313T042310Z_system_time_sync_repair_plan.json",
        {
            "status": "run_now",
            "plan_brief": "manual_time_repair_required:SC2603:timed_apns_fallback",
            "admin_required": True,
            "done_when": "restore direct NTP path and rerun time_sync_probe",
        },
    )
    _write_json(
        review_dir / "20260313T042311Z_system_time_sync_repair_verification_report.json",
        {
            "status": "blocked",
            "verification_brief": "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked",
            "cleared": False,
        },
    )
    _write_json(
        review_dir / "20260313T042312Z_openclaw_orderflow_blueprint.json",
        {
            "status": "ok",
            "generated_at_utc": "2026-03-13T04:23:12Z",
            "current_status": {
                "current_life_stage": "local_repair_gated_continuity_hardened_remote_guardian",
                "target_life_stage": "orderflow_native_execution_organism",
                "remote_intent_queue_brief": "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um",
                "remote_intent_queue_status": "queued_wait_trade_readiness",
                "remote_intent_queue_recommendation": "hold_remote_idle_until_ticket_ready",
                "remote_execution_journal_brief": (
                    "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um"
                    " | blocked:ticket_missing:no_actionable_ticket | not_attempted_wait_trade_readiness"
                ),
                "remote_execution_journal_status": "intent_logged_guardian_blocked",
                "remote_execution_journal_append_status": "appended",
                "remote_orderflow_feedback_brief": (
                    "downrank_guardian_blocked_route:SC2603:queue_aging_high:ticket_missing:no_actionable_ticket"
                ),
                "remote_orderflow_feedback_status": "downrank_guardian_blocked_route",
                "remote_orderflow_feedback_recommendation": "downrank_until_ticket_fresh_and_guardian_clear",
                "remote_orderflow_policy_brief": (
                    "shadow_policy_blocked:SC2603:queued_wait_trade_readiness:downrank_guardian_blocked_route"
                ),
                "remote_orderflow_policy_status": "shadow_policy_blocked",
                "remote_orderflow_policy_decision": "reject_until_guardian_clear",
                "remote_execution_ack_brief": (
                    "shadow_no_send_ack_recorded:SC2603:not_sent_policy_blocked:no_fill_execution_not_attempted"
                ),
                "remote_execution_ack_status": "shadow_no_send_ack_recorded",
                "remote_execution_ack_decision": "record_reject_without_transport",
                "remote_orderflow_quality_shadow_learning_score": 65,
                "remote_orderflow_quality_execution_readiness_score": 0,
                "remote_orderflow_quality_transport_observability_score": 90,
                "remote_guardian_blocker_clearance_brief": (
                    "guardian_blocker_clearance_blocked:SC2603:5_blocked:portfolio_margin_um"
                ),
                "remote_guardian_blocker_clearance_status": "guardian_blocker_clearance_blocked",
                "remote_guardian_blocker_clearance_score": 0,
                "remote_guardian_blocker_clearance_top_blocker_code": "timed_ntp_via_fake_ip_clearance",
                "remote_guardian_blocker_clearance_top_blocker_title": (
                    "Repair fake-ip NTP path before any orderflow promotion"
                ),
                "remote_guardian_blocker_clearance_top_blocker_target_artifact": (
                    "system_time_sync_repair_verification_report"
                ),
                "remote_guardian_blocker_clearance_top_blocker_next_action": (
                    "repair_fake_ip_ntp_path_then_verify"
                ),
                "remote_guarded_canary_promotion_gate_blocker_code": (
                    "timed_ntp_via_fake_ip_clearance"
                ),
                "remote_shadow_learning_continuity_brief": (
                    "shadow_learning_continuity_stable:SC2603:shadow_feedback_alive:portfolio_margin_um"
                ),
                "remote_shadow_learning_continuity_status": "shadow_learning_continuity_stable",
                "remote_shadow_learning_continuity_decision": "continue_shadow_learning_collect_feedback",
                "remote_promotion_unblock_readiness_brief": (
                    "local_time_sync_primary_blocker_shadow_ready:SC2603:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um"
                ),
                "remote_promotion_unblock_readiness_status": (
                    "local_time_sync_primary_blocker_shadow_ready"
                ),
                "remote_promotion_unblock_readiness_decision": (
                    "repair_local_fake_ip_ntp_path_then_review_guarded_canary"
                ),
                "remote_promotion_unblock_primary_blocker_scope": "timed_ntp_via_fake_ip",
                "remote_promotion_unblock_primary_local_repair_title": (
                    "Repair local fake-ip NTP path to unlock guarded canary review"
                ),
                "remote_promotion_unblock_primary_local_repair_target_artifact": (
                    "system_time_sync_repair_verification_report"
                ),
                "remote_promotion_unblock_primary_local_repair_plan_brief": (
                    "manual_time_repair_required:SC2603:timed_ntp_via_fake_ip"
                ),
                "remote_promotion_unblock_primary_local_repair_environment_classification": (
                    "timed_ntp_via_fake_ip"
                ),
                "remote_promotion_unblock_primary_local_repair_environment_blocker_detail": (
                    "timed_source=NTP; ntp_ip=198.18.0.17; delay_ms=118.702; clash_dns_mode=fake-ip; tun_stack=gvisor"
                ),
                "remote_time_sync_mode": "promotion_blocked_shadow_learning_allowed",
            },
            "immediate_backlog": [
                {
                    "priority": 1,
                    "title": "Repair local fake-ip NTP path to unlock guarded canary review",
                    "target_artifact": "system_time_sync_repair_verification_report",
                    "why": (
                        "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
                    ),
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-13T05:00:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_brooks_route_report_selected_routes_brief"].startswith(
        "equity:three_push_climax_reversal"
    )
    assert payload["source_brooks_route_report_head_symbol"] == "SC2603"
    assert payload["source_brooks_route_report_head_strategy_id"] == "second_entry_trend_continuation"
    assert payload["source_brooks_execution_plan_head_symbol"] == "SC2603"
    assert payload["source_brooks_execution_plan_head_plan_status"] == "manual_structure_review_now"
    assert payload["source_brooks_structure_review_queue_artifact"].endswith(
        "_brooks_structure_review_queue.json"
    )
    assert payload["source_brooks_structure_review_queue_status"] == "ok"
    assert payload["source_brooks_structure_review_queue_brief"] == "ready:SC2603:96:review_queue_now"
    assert payload["source_brooks_structure_refresh_artifact"].endswith(
        "_brooks_structure_refresh.json"
    )
    assert payload["source_brooks_structure_refresh_status"] == "ok"
    assert payload["source_brooks_structure_refresh_brief"] == (
        "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now"
    )
    assert payload["source_brooks_structure_refresh_head_symbol"] == "SC2603"
    assert payload["source_brooks_structure_refresh_head_action"] == "review_manual_stop_entry"
    assert payload["source_brooks_structure_refresh_head_priority_score"] == 96
    assert payload["source_cross_market_operator_state_artifact"].endswith(
        "_cross_market_operator_state.json"
    )
    assert payload["source_cross_market_operator_state_status"] == "ok"
    assert payload["source_cross_market_operator_state_operator_backlog_status"] == "ready"
    assert payload["source_cross_market_operator_state_operator_backlog_count"] == 4
    assert payload["source_cross_market_operator_state_operator_backlog_brief"] == (
        "1:waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
        "2:waiting:commodity_fill_evidence:XAGUSD:wait_for_paper_execution_fill_evidence:98 | "
        "3:review:brooks_structure:SC2603:review_manual_stop_entry:96 | "
        "4:review:crypto_route:SOLUSDT:deprioritize_flow:73"
    )
    assert payload["source_cross_market_operator_state_operator_backlog_state_brief"] == (
        "waiting=2 | review=2 | watch=0 | blocked=0"
    )
    assert payload["source_cross_market_operator_state_operator_backlog_priority_totals_brief"] == (
        "waiting=197 | review=169 | watch=0 | blocked=0"
    )
    assert payload["source_cross_market_operator_state_operator_state_lane_heads_brief"] == (
        "waiting:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
        "review:SC2603:review_manual_stop_entry:96 | "
        "watch:-:-:0 | "
        "blocked:-:-:0"
    )
    assert payload["source_cross_market_operator_state_operator_state_lane_priority_order_brief"] == (
        "waiting@197:2 > review@169:2 > watch@0:0 > blocked@0:0"
    )
    assert payload["source_cross_market_operator_state_operator_head_area"] == "commodity_execution_close_evidence"
    assert payload["source_cross_market_operator_state_operator_head_symbol"] == "XAUUSD"
    assert payload["source_cross_market_operator_state_operator_head_action"] == "wait_for_paper_execution_close_evidence"
    assert payload["source_cross_market_operator_state_operator_head_state"] == "waiting"
    assert payload["source_cross_market_operator_state_operator_head_priority_score"] == 99
    assert payload["source_cross_market_operator_state_operator_head_priority_tier"] == "waiting_now"
    assert payload["cross_market_operator_head_status"] == "waiting"
    assert payload["cross_market_operator_head_brief"] == (
        "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99"
    )
    assert payload["cross_market_operator_head_area"] == "commodity_execution_close_evidence"
    assert payload["cross_market_operator_head_symbol"] == "XAUUSD"
    assert payload["cross_market_operator_head_action"] == "wait_for_paper_execution_close_evidence"
    assert payload["cross_market_operator_head_state"] == "waiting"
    assert payload["cross_market_operator_head_priority_score"] == 99
    assert payload["cross_market_operator_head_priority_tier"] == "waiting_now"
    assert payload["cross_market_operator_head_blocker_detail"] == (
        "paper execution evidence is present, but position is still OPEN"
    )
    assert payload["cross_market_operator_head_done_when"] == (
        "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
    )
    assert payload["cross_market_operator_backlog_count"] == 3
    assert payload["cross_market_operator_backlog_brief"] == (
        "2:waiting:commodity_fill_evidence:XAGUSD:wait_for_paper_execution_fill_evidence:98 | "
        "3:review:brooks_structure:SC2603:review_manual_stop_entry:96 | "
        "4:review:crypto_route:SOLUSDT:deprioritize_flow:73"
    )
    assert payload["cross_market_operator_backlog_state_brief"] == "waiting=2 | review=2 | watch=0 | blocked=0"
    assert payload["cross_market_operator_backlog_priority_totals_brief"] == "waiting=197 | review=169 | watch=0 | blocked=0"
    assert payload["cross_market_operator_lane_heads_brief"] == (
        "waiting:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
        "review:SC2603:review_manual_stop_entry:96 | "
        "watch:-:-:0 | "
        "blocked:-:-:0"
    )
    assert payload["cross_market_operator_lane_priority_order_brief"] == (
        "waiting@197:2 > review@169:2 > watch@0:0 > blocked@0:0"
    )
    assert payload["cross_market_operator_waiting_lane_status"] == "ready"
    assert payload["cross_market_operator_waiting_lane_count"] == 2
    assert payload["cross_market_operator_waiting_lane_brief"] == (
        "1:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
        "2:XAGUSD:wait_for_paper_execution_fill_evidence:98"
    )
    assert payload["cross_market_operator_waiting_lane_priority_total"] == 197
    assert payload["cross_market_operator_waiting_lane_head_symbol"] == "XAUUSD"
    assert payload["cross_market_operator_waiting_lane_head_action"] == "wait_for_paper_execution_close_evidence"
    assert payload["cross_market_operator_waiting_lane_head_priority_score"] == 99
    assert payload["cross_market_operator_waiting_lane_head_priority_tier"] == "waiting_now"
    assert payload["cross_market_operator_review_lane_status"] == "ready"
    assert payload["cross_market_operator_review_lane_count"] == 2
    assert payload["cross_market_operator_review_lane_brief"] == (
        "3:SC2603:review_manual_stop_entry:96 | 4:SOLUSDT:deprioritize_flow:73"
    )
    assert payload["cross_market_operator_review_lane_priority_total"] == 169
    assert payload["cross_market_operator_review_lane_head_symbol"] == "SC2603"
    assert payload["cross_market_operator_review_lane_head_action"] == "review_manual_stop_entry"
    assert payload["cross_market_operator_review_lane_head_priority_score"] == 96
    assert payload["cross_market_operator_review_lane_head_priority_tier"] == "review_queue_now"
    assert payload["cross_market_operator_watch_lane_status"] == "inactive"
    assert payload["cross_market_operator_watch_lane_count"] == 0
    assert payload["cross_market_operator_watch_lane_brief"] == "-"
    assert payload["cross_market_operator_watch_lane_priority_total"] == 0
    assert payload["cross_market_operator_blocked_lane_status"] == "inactive"
    assert payload["cross_market_operator_blocked_lane_count"] == 0
    assert payload["cross_market_operator_blocked_lane_brief"] == "-"
    assert payload["cross_market_operator_blocked_lane_priority_total"] == 0
    assert payload["source_cross_market_operator_state_review_backlog_status"] == "ready"
    assert payload["source_cross_market_operator_state_review_backlog_count"] == 2
    assert payload["source_cross_market_operator_state_review_backlog_brief"] == (
        "1:brooks_structure:SC2603:review_queue_now:96 | 2:crypto_route:SOLUSDT:review_queue_now:73"
    )
    assert payload["source_cross_market_operator_state_review_head_area"] == "brooks_structure"
    assert payload["source_cross_market_operator_state_review_head_symbol"] == "SC2603"
    assert payload["source_cross_market_operator_state_review_head_action"] == "review_manual_stop_entry"
    assert payload["source_cross_market_operator_state_review_head_priority_score"] == 96
    assert payload["source_cross_market_operator_state_review_head_priority_tier"] == "review_queue_now"
    assert payload["cross_market_review_head_status"] == "ready"
    assert payload["cross_market_review_head_brief"] == (
        "ready:brooks_structure:SC2603:review_manual_stop_entry:96"
    )
    assert payload["cross_market_review_head_area"] == "brooks_structure"
    assert payload["cross_market_review_head_symbol"] == "SC2603"
    assert payload["cross_market_review_head_action"] == "review_manual_stop_entry"
    assert payload["cross_market_review_head_priority_score"] == 96
    assert payload["cross_market_review_head_priority_tier"] == "review_queue_now"
    assert payload["cross_market_review_head_blocker_detail"] == (
        "manual-only structure route still requires venue confirmation"
    )
    assert payload["cross_market_review_head_done_when"] == (
        "cross-market review head is executed manually, promoted, or invalidated"
    )
    assert payload["source_system_time_sync_repair_plan_artifact"].endswith(
        "_system_time_sync_repair_plan.json"
    )
    assert payload["source_system_time_sync_repair_plan_status"] == "run_now"
    assert payload["source_system_time_sync_repair_plan_brief"] == (
        "manual_time_repair_required:SC2603:timed_apns_fallback"
    )
    assert payload["source_system_time_sync_repair_plan_admin_required"] is True
    assert payload["source_system_time_sync_repair_verification_artifact"].endswith(
        "_system_time_sync_repair_verification_report.json"
    )
    assert payload["source_system_time_sync_repair_verification_status"] == "blocked"
    assert payload["source_system_time_sync_repair_verification_brief"] == (
        "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
    )
    assert payload["source_system_time_sync_repair_verification_cleared"] is False
    assert payload["source_openclaw_orderflow_blueprint_artifact"].endswith(
        "_openclaw_orderflow_blueprint.json"
    )
    assert payload["source_openclaw_orderflow_blueprint_status"] == "ok"
    assert payload["source_openclaw_orderflow_blueprint_brief"] == (
        "local_repair_gated_continuity_hardened_remote_guardian -> orderflow_native_execution_organism | "
        "P1:system_time_sync_repair_verification_report:Repair local fake-ip NTP path to unlock guarded canary review"
    )
    assert payload["source_openclaw_orderflow_blueprint_current_life_stage"] == (
        "local_repair_gated_continuity_hardened_remote_guardian"
    )
    assert (
        payload["source_openclaw_orderflow_blueprint_target_life_stage"]
        == "orderflow_native_execution_organism"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_intent_queue_brief"] == (
        "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_intent_queue_status"] == (
        "queued_wait_trade_readiness"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_intent_queue_recommendation"] == (
        "hold_remote_idle_until_ticket_ready"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_execution_journal_status"] == (
        "intent_logged_guardian_blocked"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_execution_journal_append_status"] == (
        "appended"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_execution_journal_brief"].startswith(
        "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_orderflow_feedback_status"] == (
        "downrank_guardian_blocked_route"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_orderflow_feedback_recommendation"] == (
        "downrank_until_ticket_fresh_and_guardian_clear"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_orderflow_feedback_brief"].startswith(
        "downrank_guardian_blocked_route:SC2603"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_orderflow_policy_status"] == (
        "shadow_policy_blocked"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_orderflow_policy_decision"] == (
        "reject_until_guardian_clear"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_orderflow_policy_brief"].startswith(
        "shadow_policy_blocked:SC2603"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_execution_ack_status"] == (
        "shadow_no_send_ack_recorded"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_execution_ack_decision"] == (
        "record_reject_without_transport"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_execution_ack_brief"].startswith(
        "shadow_no_send_ack_recorded:SC2603"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_orderflow_quality_shadow_learning_score"] == 65
    assert (
        payload["source_openclaw_orderflow_blueprint_remote_orderflow_quality_execution_readiness_score"]
        == 0
    )
    assert (
        payload["source_openclaw_orderflow_blueprint_remote_orderflow_quality_transport_observability_score"]
        == 90
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_brief"].startswith(
        "guardian_blocker_clearance_blocked:SC2603"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_code"] == (
        "timed_ntp_via_fake_ip_clearance"
    )
    assert payload[
        "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_target_artifact"
    ] == "system_time_sync_repair_verification_report"
    assert payload["source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_brief"] == (
        "shadow_learning_continuity_stable:SC2603:shadow_feedback_alive:portfolio_margin_um"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_status"] == (
        "shadow_learning_continuity_stable"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_decision"] == (
        "continue_shadow_learning_collect_feedback"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief"] == (
        "local_time_sync_primary_blocker_shadow_ready:SC2603:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_status"] == (
        "local_time_sync_primary_blocker_shadow_ready"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_decision"] == (
        "repair_local_fake_ip_ntp_path_then_review_guarded_canary"
    )
    assert payload["source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_blocker_scope"] == (
        "timed_ntp_via_fake_ip"
    )
    assert payload[
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_plan_brief"
    ] == "manual_time_repair_required:SC2603:timed_ntp_via_fake_ip"
    assert payload[
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_environment_classification"
    ] == "timed_ntp_via_fake_ip"
    assert payload[
        "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_target_artifact"
    ] == "system_time_sync_repair_verification_report"
    assert payload[
        "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_code"
    ] == "timed_ntp_via_fake_ip_clearance"
    assert payload["source_openclaw_orderflow_blueprint_remote_time_sync_mode"] == (
        "promotion_blocked_shadow_learning_allowed"
    )
    assert payload["source_openclaw_orderflow_blueprint_top_backlog_title"] == (
        "Repair local fake-ip NTP path to unlock guarded canary review"
    )
    assert payload["source_openclaw_orderflow_blueprint_top_backlog_target_artifact"] == (
        "system_time_sync_repair_verification_report"
    )
    assert payload["source_openclaw_orderflow_blueprint_top_backlog_why"] == (
        "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
    )
    assert payload["cross_market_review_backlog_count"] == 1
    assert payload["cross_market_review_backlog_brief"] == "2:crypto_route:SOLUSDT:review_queue_now:73"
    assert payload["brooks_structure_review_status"] == "ready"
    assert payload["brooks_structure_review_queue_status"] == "ready"
    assert payload["brooks_structure_review_queue_count"] == 2
    assert len(payload["brooks_structure_review_queue"]) == 2
    assert payload["brooks_structure_review_priority_status"] == "ready"
    assert payload["brooks_structure_review_priority_brief"].startswith("ready:SC2603:")
    assert payload["brooks_structure_review_head_symbol"] == "SC2603"
    assert payload["brooks_structure_review_head_tier"] == "review_queue_now"
    assert payload["brooks_structure_review_head_action"] == "review_manual_stop_entry"
    assert abs(payload["brooks_structure_review_head_route_selection_score"] - 81.68716356888629) < 1e-9
    assert payload["brooks_structure_review_head_signal_score"] == 80
    assert payload["brooks_structure_review_head_signal_age_bars"] == 2
    assert payload["brooks_structure_review_head_priority_score"] == 96
    assert payload["brooks_structure_review_head_priority_tier"] == "review_queue_now"
    assert payload["brooks_structure_review_queue"][1]["symbol"] == "AU2406"
    assert payload["brooks_structure_review_queue"][1]["tier"] == "blocked_queue"
    assert payload["brooks_structure_review_queue"][1]["priority_score"] == 53
    assert payload["brooks_structure_review_queue"][1]["priority_tier"] == "blocked_review"
    assert payload["brooks_structure_operator_status"] == "ready"
    assert payload["brooks_structure_operator_brief"] == "ready:SC2603:review_manual_stop_entry:96"
    assert payload["brooks_structure_operator_head_symbol"] == "SC2603"
    assert payload["brooks_structure_operator_head_action"] == "review_manual_stop_entry"
    assert payload["brooks_structure_operator_head_priority_score"] == 96
    assert payload["brooks_structure_operator_head_priority_tier"] == "review_queue_now"
    assert payload["brooks_structure_operator_backlog_count"] == 1
    assert payload["brooks_structure_operator_backlog_brief"] == "2:AU2406:blocked_review:53"
    assert "AU2406" in payload["brooks_structure_review_queue_brief"]
    assert "brooks-route:" in payload["summary_text"]
    assert "brooks-exec-plan:" in payload["summary_text"]
    assert "brooks-review-queue:" in payload["summary_text"]
    assert "brooks-review-priority:" in payload["summary_text"]
    assert "brooks-operator-lane:" in payload["summary_text"]
    assert "brooks-refresh:" in payload["summary_text"]
    assert "cross-market-operator:" in payload["summary_text"]
    assert "cross-market-operator-lanes:" in payload["summary_text"]
    assert "cross-market-operator-lane-heads:" in payload["summary_text"]
    assert "cross-market-operator-lane-order:" in payload["summary_text"]
    assert "cross-market-head:" in payload["summary_text"]
    assert "cross-market-review:" in payload["summary_text"]


def test_build_hot_universe_operator_brief_surfaces_remote_live_history_audit(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260313T010000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"focus_primary_batches": ["metals_all"]},
            "crypto_route_brief": {"operator_status": "watch-all"},
        },
    )
    _write_json(
        review_dir / "20260313T012000Z_remote_live_history_audit.json",
        {
            "generated_at_utc": "2026-03-13T01:20:00Z",
            "market": "portfolio_margin_um",
            "status": "ok",
            "ok": True,
            "window_summaries": [
                {
                    "window_hours": 24,
                    "history_window_label": "24h",
                    "quote_available": 99.5,
                    "open_positions": 1,
                    "closed_pnl": 12.5,
                    "trade_count": 3,
                    "risk_guard_status": "blocked",
                    "risk_guard_reasons": ["ticket_missing:no_actionable_ticket"],
                    "blocked_candidate": {"symbol": "BNBUSDT"},
                },
                {
                    "window_hours": 168,
                    "history_window_label": "7d",
                    "closed_pnl": 4.5,
                    "trade_count": 5,
                },
                {
                    "window_hours": 720,
                    "history_window_label": "30d",
                    "closed_pnl": 15.25,
                    "trade_count": 9,
                    "income_pnl_by_symbol": {"BTCUSDT": 8.5, "ETHUSDT": 1.25},
                    "income_pnl_by_day": {"2026-03-12": 2.5, "2026-03-11": -1.0},
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260313T012100Z_remote_live_handoff.json",
        {
            "status": "ok",
            "generated_at": "2026-03-13T01:21:00Z",
            "operator_handoff": {
                "handoff_state": "ops_live_gate_blocked",
                "operator_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
                "ready_check_scope_market": "portfolio_margin_um",
                "ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
                "account_scope_alignment": {
                    "status": "split_scope_spot_vs_portfolio_margin_um",
                    "brief": "split_scope_spot_vs_portfolio_margin_um",
                    "blocking": False,
                    "blocker_detail": "spot ready-check and unified-account history refer to different execution scopes.",
                },
            },
        },
    )
    _write_json(
        review_dir / "20260313T012200Z_live_gate_blocker_report.json",
        {
            "generated_at": "2026-03-13T01:22:00Z",
            "live_decision": {"current_decision": "do_not_start_formal_live"},
            "remote_live_diagnosis": {
                "status": "profitability_confirmed_but_auto_live_blocked",
                "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                "blocker_detail": (
                    "portfolio_margin_um remote history confirms realized profitability "
                    "(30d pnl=15.25000000 across 9 trades), but automated live remains blocked by "
                    "ops_live_gate, risk_guard."
                ),
                "done_when": (
                    "clear ops_live_gate and risk_guard blockers while keeping the intended ready-check "
                    "scope aligned with the profitable execution account"
                ),
            },
            "remote_live_operator_alignment": {
                "status": "local_operator_active_remote_live_blocked",
                "brief": "local_operator_active_remote_live_blocked:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked",
                "head_area": "commodity_execution_close_evidence",
                "head_symbol": "XAUUSD",
                "head_action": "wait_for_paper_execution_close_evidence",
                "head_state": "waiting",
                "head_priority_score": 99,
                "head_priority_tier": "waiting_now",
                "remote_status": "profitability_confirmed_but_auto_live_blocked",
                "blocker_detail": "Current local operator head is XAUUSD (commodity_execution_close_evidence/wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote automated live remains profitability_confirmed_but_auto_live_blocked.",
                "done_when": "XAUUSD local operator head progresses and remote auto-live blockers are cleared",
            },
            "remote_live_takeover_clearing": {
                "status": "clearing_required",
                "brief": "clearing_required:ops_live_gate+risk_guard",
                "blocking_layers": ["ops_live_gate", "risk_guard"],
                "blocker_detail": "ops_live_gate needs rollback_hard, risk_violations, max_drawdown, slot_anomaly; risk_guard needs ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap",
                "done_when": "ops_live_gate becomes clear and risk_guard reasons become empty",
            },
                "remote_live_takeover_repair_queue": {
                    "status": "ready",
                    "brief": "ready:ops_live_gate:rollback_hard:99",
                    "queue_brief": (
                        "1:ops_live_gate:rollback_hard:99 | 2:ops_live_gate:slot_anomaly:98 | "
                        "3:ops_live_gate:backtest_snapshot:97 | 4:ops_live_gate:ops_status_red:96 | "
                        "5:risk_guard:ticket_missing:no_actionable_ticket:89 | "
                        "6:risk_guard:panic_cooldown_active:88 | 7:risk_guard:open_exposure_above_cap:87"
                    ),
                    "count": 7,
                    "head_area": "ops_live_gate",
                    "head_code": "rollback_hard",
                    "head_action": "clear_ops_live_gate_condition",
                    "head_priority_score": 99,
                    "head_priority_tier": "repair_queue_now",
                    "head_command": "cmd-gate",
                    "head_clear_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
                    "done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
                    "items": [
                        {
                            "rank": 1,
                            "area": "ops_live_gate",
                            "code": "rollback_hard",
                            "action": "clear_ops_live_gate_condition",
                            "priority_tier": "repair_queue_now",
                            "priority_score": 99,
                            "clear_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
                            "goal": "clear ops live gate blockers",
                            "actions": ["reconcile remote ops status"],
                            "command": "cmd-gate",
                        },
                        {
                            "rank": 2,
                            "area": "ops_live_gate",
                            "code": "slot_anomaly",
                            "action": "clear_ops_live_gate_condition",
                            "priority_tier": "repair_queue_now",
                            "priority_score": 98,
                            "clear_when": "clear slot anomaly so ops_live_gate can resume",
                            "goal": "clear ops live gate blockers",
                            "actions": ["reconcile remote ops status"],
                            "command": "cmd-gate",
                        },
                        {
                            "rank": 3,
                            "area": "ops_live_gate",
                            "code": "backtest_snapshot",
                            "action": "clear_ops_live_gate_condition",
                            "priority_tier": "repair_queue_now",
                            "priority_score": 97,
                            "clear_when": "refresh backtest snapshot so ops_live_gate can resume",
                            "goal": "clear ops live gate blockers",
                            "actions": ["reconcile remote ops status"],
                            "command": "cmd-gate",
                        },
                        {
                            "rank": 4,
                            "area": "ops_live_gate",
                            "code": "ops_status_red",
                            "action": "clear_ops_live_gate_condition",
                            "priority_tier": "repair_queue_now",
                            "priority_score": 96,
                            "clear_when": "clear ops red status so ops_live_gate can resume",
                            "goal": "clear ops live gate blockers",
                            "actions": ["reconcile remote ops status"],
                            "command": "cmd-gate",
                        },
                        {
                            "rank": 5,
                            "area": "risk_guard",
                            "code": "ticket_missing:no_actionable_ticket",
                            "action": "clear_risk_guard_condition",
                            "priority_tier": "repair_queue_now",
                            "priority_score": 89,
                            "clear_when": "publish at least one actionable ticket",
                            "goal": "clear risk guard blockers",
                            "actions": ["rebuild order tickets"],
                            "command": "cmd-risk",
                        },
                        {
                            "rank": 6,
                            "area": "risk_guard",
                            "code": "panic_cooldown_active",
                            "action": "clear_risk_guard_condition",
                            "priority_tier": "repair_queue_now",
                            "priority_score": 88,
                            "clear_when": "allow panic cooldown to expire cleanly",
                            "goal": "clear risk guard blockers",
                            "actions": ["wait for cooldown expiry"],
                            "command": "cmd-risk",
                        },
                        {
                            "rank": 7,
                            "area": "risk_guard",
                            "code": "open_exposure_above_cap",
                            "action": "clear_risk_guard_condition",
                            "priority_tier": "repair_queue_now",
                            "priority_score": 87,
                            "clear_when": "reduce open exposure below configured cap",
                            "goal": "clear risk guard blockers",
                            "actions": ["reduce exposure"],
                            "command": "cmd-risk",
                        },
                    ],
                },
            "ops_live_gate_clearing": {
                "status": "clearing_required",
                "conditions_brief": "rollback_hard, risk_violations, max_drawdown, slot_anomaly",
            },
            "risk_guard_clearing": {
                "status": "clearing_required",
                "conditions_brief": "ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap",
            },
        },
    )
    _write_json(
        review_dir / "20260313T012250Z_cross_market_operator_state.json",
        {
            "status": "ok",
            "as_of": "2026-03-13T01:22:50Z",
            "operator_blocked_lane_status": "ready",
            "operator_blocked_lane_count": 7,
            "operator_blocked_lane_brief": "1:ROLLBACK_HARD:repair_queue_now:99",
            "operator_blocked_lane_priority_total": 654,
            "operator_blocked_lane_head_symbol": "ROLLBACK_HARD",
            "operator_blocked_lane_head_action": "clear_ops_live_gate_condition",
            "operator_blocked_lane_head_priority_score": 99,
            "operator_blocked_lane_head_priority_tier": "repair_queue_now",
            "remote_live_takeover_gate_status": "blocked_by_remote_live_gate",
            "remote_live_takeover_gate_brief": (
                "blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked"
            ),
            "remote_live_takeover_gate_blocker_detail": (
                "Current local operator head is XAUUSD (commodity_execution_close_evidence/"
                "wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote "
                "automated live remains profitability_confirmed_but_auto_live_blocked."
            ),
            "remote_live_takeover_gate_done_when": (
                "XAUUSD local operator head progresses and remote auto-live blockers are cleared"
            ),
            "remote_live_takeover_clearing_status": "clearing_required",
            "remote_live_takeover_clearing_brief": "clearing_required:ops_live_gate+risk_guard",
            "remote_live_takeover_clearing_blocker_detail": (
                "ops_live_gate needs rollback_hard, risk_violations, max_drawdown, slot_anomaly; "
                "risk_guard needs ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap"
            ),
            "remote_live_takeover_clearing_done_when": (
                "ops_live_gate becomes clear and risk_guard reasons become empty"
            ),
            "remote_live_takeover_clearing_source_freshness_brief": (
                "ops_reconcile=fresh:0.003h | risk_guard=fresh:5.0s"
            ),
            "remote_live_takeover_slot_anomaly_breakdown_status": "slot_anomaly_active_root_cause",
            "remote_live_takeover_slot_anomaly_breakdown_brief": (
                "slot_anomaly_active_root_cause:2026-03-16"
            ),
            "remote_live_takeover_slot_anomaly_breakdown_artifact": (
                "/tmp/20260316T000000Z_slot_anomaly_breakdown.json"
            ),
            "remote_live_takeover_slot_anomaly_breakdown_repair_focus": (
                "优先修复 slot_anomaly 缺陷并重跑 lie ops-report --date 2026-03-16 --window-days 7"
            ),
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-13T01:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_remote_live_history_audit_status"] == "ok"
    assert payload["source_remote_live_history_audit_market"] == "portfolio_margin_um"
    assert payload["source_remote_live_history_audit_window_brief"] == (
        "24h:12.5pnl/3tr/1open | 7d:4.5pnl/5tr/0open | 30d:15.25pnl/9tr/0open"
    )
    assert payload["source_remote_live_history_audit_quote_available"] == 99.5
    assert payload["source_remote_live_history_audit_open_positions"] == 1
    assert payload["source_remote_live_history_audit_risk_guard_status"] == "blocked"
    assert payload["source_remote_live_history_audit_risk_guard_reasons"] == [
        "ticket_missing:no_actionable_ticket"
    ]
    assert payload["source_remote_live_history_audit_blocked_candidate_symbol"] == "BNBUSDT"
    assert payload["source_remote_live_history_audit_30d_symbol_pnl_brief"] == (
        "BTCUSDT:8.5, ETHUSDT:1.25"
    )
    assert payload["source_remote_live_history_audit_30d_day_pnl_brief"] == (
        "2026-03-11:-1.0, 2026-03-12:2.5"
    )
    assert payload["source_remote_live_handoff_status"] == "ok"
    assert payload["source_remote_live_handoff_state"] == "ops_live_gate_blocked"
    assert payload["source_remote_live_handoff_ready_check_scope_brief"] == (
        "portfolio_margin_um:portfolio_margin_um"
    )
    assert payload["source_remote_live_handoff_account_scope_alignment_brief"] == (
        "split_scope_spot_vs_portfolio_margin_um"
    )
    assert payload["source_live_gate_blocker_live_decision"] == "do_not_start_formal_live"
    assert payload["source_live_gate_blocker_remote_live_diagnosis_status"] == (
        "profitability_confirmed_but_auto_live_blocked"
    )
    assert payload["source_live_gate_blocker_remote_live_diagnosis_brief"] == (
        "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
    )
    assert "portfolio_margin_um remote history confirms realized profitability" in (
        payload["source_live_gate_blocker_remote_live_diagnosis_blocker_detail"]
    )
    assert payload["source_live_gate_blocker_remote_live_operator_alignment_status"] == (
        "local_operator_active_remote_live_blocked"
    )
    assert payload["source_live_gate_blocker_remote_live_operator_alignment_brief"] == (
        "local_operator_active_remote_live_blocked:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked"
    )
    assert payload["source_live_gate_blocker_remote_live_takeover_clearing_status"] == (
        "clearing_required"
    )
    assert payload["source_live_gate_blocker_remote_live_takeover_clearing_brief"] == (
        "clearing_required:ops_live_gate+risk_guard"
    )
    assert payload["source_live_gate_blocker_remote_live_takeover_repair_queue_status"] == "ready"
    assert payload["source_live_gate_blocker_remote_live_takeover_repair_queue_brief"] == (
        "ready:ops_live_gate:rollback_hard:99"
    )
    assert payload["remote_live_takeover_repair_queue_status"] == "ready"
    assert payload["remote_live_takeover_repair_queue_brief"] == (
        "ready:ops_live_gate:rollback_hard:99"
    )
    assert payload["remote_live_takeover_repair_queue_queue_brief"].startswith(
        "1:ops_live_gate:rollback_hard:99"
    )
    assert payload["remote_live_takeover_repair_queue_head_code"] == "rollback_hard"
    assert payload["remote_live_takeover_repair_queue_head_command"] == "cmd-gate"
    assert payload["cross_market_operator_repair_head_status"] == "ready"
    assert payload["cross_market_operator_repair_head_brief"] == (
        "ready:ops_live_gate:rollback_hard:99"
    )
    assert payload["cross_market_operator_repair_head_area"] == "ops_live_gate"
    assert payload["cross_market_operator_repair_head_code"] == "rollback_hard"
    assert payload["cross_market_operator_repair_head_action"] == "clear_ops_live_gate_condition"
    assert payload["cross_market_operator_repair_head_priority_score"] == 99
    assert payload["cross_market_operator_repair_head_priority_tier"] == "repair_queue_now"
    assert payload["cross_market_operator_repair_head_command"] == "cmd-gate"
    assert payload["cross_market_operator_repair_backlog_status"] == "ready"
    assert payload["cross_market_operator_repair_backlog_count"] == 7
    assert payload["cross_market_operator_repair_backlog_priority_total"] == 654
    assert payload["operator_repair_queue_brief"] == (
        "1:ops_live_gate:rollback_hard:clear_ops_live_gate_condition"
        " | 2:ops_live_gate:slot_anomaly:clear_ops_live_gate_condition"
        " | 3:ops_live_gate:backtest_snapshot:clear_ops_live_gate_condition"
        " | 4:ops_live_gate:ops_status_red:clear_ops_live_gate_condition | +3"
    )
    assert payload["operator_repair_queue_count"] == 7
    assert payload["operator_repair_checklist_brief"] == (
        "1:repair:ROLLBACK_HARD:clear_ops_live_gate_condition"
        " | 2:repair:SLOT_ANOMALY:clear_ops_live_gate_condition"
        " | 3:repair:BACKTEST_SNAPSHOT:clear_ops_live_gate_condition"
        " | 4:repair:OPS_STATUS_RED:clear_ops_live_gate_condition | +3"
    )
    assert payload["operator_repair_queue"][0]["command"] == "cmd-gate"
    assert payload["operator_repair_queue"][0]["clear_when"] == (
        "clear hard rollback so ops_live_gate can leave rollback_now state"
    )
    assert payload["operator_repair_checklist"][0]["state"] == "repair"
    assert payload["operator_repair_checklist"][0]["done_when"] == (
        "clear hard rollback so ops_live_gate can leave rollback_now state"
    )
    assert "cross-market-repair-head:" in payload["summary_text"]
    assert "ready:ops_live_gate:rollback_hard:99" in payload["summary_text"]
    assert "repair-queue: 1:ops_live_gate:rollback_hard:clear_ops_live_gate_condition" in payload["summary_text"]
    assert "repair-checklist: 1:repair:ROLLBACK_HARD:clear_ops_live_gate_condition" in payload["summary_text"]
    assert payload["source_cross_market_operator_state_remote_live_takeover_gate_status"] == (
        "blocked_by_remote_live_gate"
    )
    assert payload["source_cross_market_operator_state_remote_live_takeover_gate_brief"] == (
        "blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked"
    )
    assert payload["cross_market_remote_live_takeover_gate_status"] == "blocked_by_remote_live_gate"
    assert payload["cross_market_remote_live_takeover_gate_brief"] == (
        "blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked"
    )
    assert payload["cross_market_remote_live_takeover_clearing_status"] == "clearing_required"
    assert payload["cross_market_remote_live_takeover_clearing_brief"] == (
        "clearing_required:ops_live_gate+risk_guard"
    )
    assert payload["cross_market_remote_live_takeover_clearing_source_freshness_brief"] == (
        "ops_reconcile=fresh:0.003h | risk_guard=fresh:5.0s"
    )
    assert payload["cross_market_remote_live_takeover_slot_anomaly_breakdown_brief"] == (
        "slot_anomaly_active_root_cause:2026-03-16"
    )
    assert "remote-live-operator-alignment:" in payload["summary_text"]
    assert "cross-market-remote-live-takeover-gate:" in payload["summary_text"]
    assert "remote-live-clearing:" in payload["summary_text"]
    assert "remote-live-repair-queue:" in payload["summary_text"]
    assert "cross-market-remote-live-clearing:" in payload["summary_text"]
    assert "cross-market-remote-live-clearing-freshness:" in payload["summary_text"]
    assert "cross-market-remote-live-slot-anomaly:" in payload["summary_text"]


def test_build_hot_universe_operator_brief_surfaces_crypto_shortline_gate_fields(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260312T060000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"focus_primary_batches": []},
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-review",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT | watch-priority:BNBUSDT",
                "next_focus_symbol": "SOLUSDT",
                "next_focus_action": "deprioritize_flow",
                "next_focus_reason": "Flow does not produce a positive ranked edge even in the short sample.",
                "shortline_market_state_brief": "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade",
                "shortline_execution_gate_brief": "4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle",
                "shortline_no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
                "shortline_session_map_brief": "asia_high_low, london_high_low, prior_day_high_low, equal_highs_lows",
                "shortline_cvd_semantic_status": "ok",
                "shortline_cvd_semantic_takeaway": "All current CVD-lite observations are downgraded to watch-only; keep them as review filters until micro quality recovers.",
                "shortline_cvd_queue_handoff_status": "queue-watch-only",
                "shortline_cvd_queue_handoff_takeaway": "Queue priorities remain valid, but the latest micro snapshot downgrades all overlapping crypto symbols to watch-only.",
                "shortline_cvd_queue_focus_batch": "crypto_hot",
                "shortline_cvd_queue_focus_action": "defer_until_micro_recovers",
                "shortline_cvd_queue_stack_brief": "crypto_hot -> crypto_majors",
                "focus_execution_state": "Bias_Only",
                "focus_execution_blocker_detail": "SOLUSDT remains route-gated via deprioritize_flow; keep Bias_Only until route quality improves and the shortline trigger stack completes.",
                "focus_execution_done_when": "SOLUSDT route improves and reaches Setup_Ready by completing the shortline trigger stack",
                "focus_execution_micro_classification": "watch_only",
                "focus_execution_micro_context": "failed_auction",
                "focus_execution_micro_trust_tier": "single_exchange_low",
                "focus_execution_micro_veto": "low_sample_or_gap_risk",
                "focus_execution_micro_reasons": ["time_sync_risk", "trust_low", "low_sample_or_gap_risk"],
                "focus_review_status": "review_no_edge_bias_only_micro_veto",
                "focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
                "focus_review_primary_blocker": "no_edge",
                "focus_review_micro_blocker": "low_sample_or_gap_risk",
                "focus_review_blocker_detail": "SOLUSDT remains route-gated via deprioritize_flow; keep Bias_Only until route quality improves and the shortline trigger stack completes. | micro=watch_only:failed_auction:low_sample_or_gap_risk:time_sync_risk,trust_low,low_sample_or_gap_risk",
                "focus_review_done_when": "SOLUSDT route improves and reaches Setup_Ready by completing the shortline trigger stack",
                "focus_review_score_status": "scored",
                "focus_review_edge_score": 5,
                "focus_review_structure_score": 25,
                "focus_review_micro_score": 20,
                "focus_review_composite_score": 17,
                "focus_review_score_brief": "scored:SOLUSDT:edge=5|structure=25|micro=20|composite=17",
                "focus_review_priority_status": "ready",
                "focus_review_priority_score": 17,
                "focus_review_priority_tier": "deprioritized_review",
                "focus_review_priority_brief": "deprioritized_review:17/100",
                "review_priority_queue_status": "ready",
                "review_priority_queue_count": 2,
                "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25",
                "review_priority_head_symbol": "SOLUSDT",
                "review_priority_head_tier": "review_queue_now",
                "review_priority_head_score": 73,
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-12T06:05:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["crypto_route_shortline_market_state_brief"] == "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade"
    assert payload["crypto_route_shortline_execution_gate_brief"].startswith("4h_profile_location -> liquidity_sweep")
    assert payload["crypto_route_shortline_no_trade_rule"] == "no_sweep_no_mss_no_cvd_no_trade"
    assert payload["crypto_route_shortline_cvd_semantic_status"] == "ok"
    assert payload["crypto_route_shortline_cvd_queue_handoff_status"] == "queue-watch-only"
    assert payload["crypto_route_shortline_cvd_queue_focus_batch"] == "crypto_hot"
    assert payload["crypto_route_shortline_cvd_queue_focus_action"] == "defer_until_micro_recovers"
    assert payload["crypto_route_shortline_cvd_queue_stack_brief"] == "crypto_hot -> crypto_majors"
    assert payload["crypto_route_focus_execution_state"] == "Bias_Only"
    assert payload["crypto_route_focus_execution_blocker_detail"].startswith("SOLUSDT remains route-gated via deprioritize_flow")
    assert payload["crypto_route_focus_execution_done_when"].startswith("SOLUSDT route improves and reaches Setup_Ready")
    assert payload["crypto_route_focus_execution_micro_classification"] == "watch_only"
    assert payload["crypto_route_focus_execution_micro_context"] == "failed_auction"
    assert payload["crypto_route_focus_execution_micro_trust_tier"] == "single_exchange_low"
    assert payload["crypto_route_focus_execution_micro_veto"] == "low_sample_or_gap_risk"
    assert payload["crypto_route_focus_execution_micro_reasons"] == [
        "time_sync_risk",
        "trust_low",
        "low_sample_or_gap_risk",
    ]
    assert payload["crypto_route_focus_review_status"] == "review_no_edge_bias_only_micro_veto"
    assert payload["crypto_route_focus_review_brief"] == "review_no_edge_bias_only_micro_veto:SOLUSDT"
    assert payload["crypto_route_focus_review_primary_blocker"] == "no_edge"
    assert payload["crypto_route_focus_review_micro_blocker"] == "low_sample_or_gap_risk"
    assert payload["crypto_route_focus_review_edge_score"] == 5
    assert payload["crypto_route_focus_review_structure_score"] == 25
    assert payload["crypto_route_focus_review_micro_score"] == 20
    assert payload["crypto_route_focus_review_composite_score"] == 17
    assert payload["crypto_route_focus_review_priority_status"] == "ready"
    assert payload["crypto_route_focus_review_priority_score"] == 17
    assert payload["crypto_route_focus_review_priority_tier"] == "deprioritized_review"
    assert payload["crypto_route_focus_review_priority_brief"] == "deprioritized_review:17/100"
    assert payload["crypto_route_review_priority_queue_status"] == "ready"
    assert payload["crypto_route_review_priority_queue_count"] == 2
    assert payload["crypto_route_review_priority_queue_brief"] == "1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25"
    assert payload["crypto_route_review_priority_head_symbol"] == "SOLUSDT"
    assert payload["crypto_route_review_priority_head_tier"] == "review_queue_now"
    assert payload["crypto_route_review_priority_head_score"] == 73
    assert payload["crypto_route_focus_review_done_when"].startswith(
        "SOLUSDT route improves and reaches Setup_Ready"
    )
    assert "crypto-shortline-market-state: Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade" in payload["summary_text"]
    assert "crypto-cvd-queue: queue-watch-only | crypto_hot:defer_until_micro_recovers | crypto_hot -> crypto_majors" in payload["summary_text"]
    assert "crypto-focus-micro: watch_only | failed_auction | single_exchange_low | low_sample_or_gap_risk" in payload["summary_text"]
    assert "crypto-review-lane: review_no_edge_bias_only_micro_veto | no_edge | low_sample_or_gap_risk" in payload["summary_text"]
    assert "crypto-review-scores: edge=5 | structure=25 | micro=20 | composite=17" in payload["summary_text"]
    assert "crypto-review-priority: deprioritized_review | score=17" in payload["summary_text"]
    assert "crypto-review-queue: ready | 1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25 | head=SOLUSDT:review_queue_now:73" in payload["summary_text"]


def test_build_hot_universe_operator_brief_marks_research_queue_plus_crypto_deploy(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_beta"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_status"] == "research-queue-plus-crypto-deploy-watch"
    assert payload["research_queue_batches"] == ["crypto_majors", "crypto_hot"]
    assert "research-queue: crypto_majors, crypto_hot" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prefers_ok_over_richer_partial_failure_for_sources(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "partial_failure",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "research_queue_batches": ["crypto_hot"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_retest_action": "rerun_bnb_native_long_window",
            },
            "crypto_route_operator_brief": {
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"avoid_batches": ["crypto_hot"]},
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_status"] == "ok"
    assert payload["source_artifact"] == str(review_dir / "20260310T121000Z_hot_universe_research.json")
    assert payload["source_action_status"] == "ok"
    assert payload["source_action_artifact"] == str(review_dir / "20260310T121000Z_hot_universe_research.json")
    assert payload["source_crypto_status"] == "ok"
    assert payload["source_crypto_artifact"] == str(review_dir / "20260310T121000Z_hot_universe_research.json")
    assert payload["operator_research_embedding_quality_status"] == "avoid_only"
    assert payload["operator_research_embedding_quality_brief"] == "avoid_only:crypto_hot"
    assert payload["operator_research_embedding_active_batches"] == []
    assert payload["operator_research_embedding_avoid_batches"] == ["crypto_hot"]
    assert payload["operator_research_embedding_zero_trade_deprioritized_batches"] == []
    assert payload["operator_research_embedding_quality_done_when"] == (
        "latest hot_universe_research promotes at least one focus_primary or research_queue batch"
    )
    assert payload["operator_crypto_route_alignment_focus_area"] == "crypto_route"
    assert payload["operator_crypto_route_alignment_focus_slot"] == "next"
    assert payload["operator_crypto_route_alignment_focus_symbol"] == "BNBUSDT"
    assert payload["operator_crypto_route_alignment_focus_action"] == "watch_priority_until_long_window_confirms"
    assert payload["operator_crypto_route_alignment_status"] == "route_ahead_of_embedding"
    assert payload["operator_crypto_route_alignment_brief"] == "route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot"
    assert payload["operator_crypto_route_alignment_recovery_status"] == "recovery_completed_no_edge"
    assert payload["operator_crypto_route_alignment_recovery_brief"] == "recovery_completed_no_edge:crypto_hot"
    assert payload["operator_crypto_route_alignment_recovery_failed_batch_count"] == 0
    assert payload["operator_crypto_route_alignment_recovery_timed_out_batch_count"] == 0
    assert payload["operator_crypto_route_alignment_recovery_zero_trade_batches"] == ["crypto_hot"]
    assert payload["operator_crypto_route_alignment_cooldown_status"] == "cooldown_active_wait_for_new_market_data"
    assert payload["operator_crypto_route_alignment_cooldown_brief"] == (
        "cooldown_active_wait_for_new_market_data:>2026-03-10"
    )


def test_build_hot_universe_operator_brief_prefers_fresher_ok_over_older_richer_ok_sources(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"avoid_batches": ["crypto_hot"]},
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_retest_action": "rerun_bnb_native_long_window",
            },
            "crypto_route_operator_brief": {
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Older richer route artifact.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Older richer xlong artifact.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {"avoid_batches": ["crypto_hot"]},
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-review",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT | watch-priority:BNBUSDT",
                "next_focus_symbol": "SOLUSDT",
                "next_focus_action": "deprioritize_flow",
            },
            "crypto_route_operator_brief": {
                "comparative_window_takeaway": "Newer route artifact should win even if it is slightly less rich.",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    expected_path = str(review_dir / "20260310T121000Z_hot_universe_research.json")
    assert payload["source_artifact"] == expected_path
    assert payload["source_action_artifact"] == expected_path
    assert payload["source_crypto_artifact"] == expected_path
    assert payload["source_status"] == "ok"
    assert payload["source_crypto_status"] == "ok"
    assert payload["operator_crypto_route_alignment_cooldown_last_research_end_date"] == "2026-03-10"
    assert payload["operator_crypto_route_alignment_cooldown_next_eligible_end_date"] == "2026-03-11"
    assert payload["operator_crypto_route_alignment_cooldown_blocker_detail"] == (
        "latest clean crypto recovery already evaluated data through 2026-03-10 and still found no edge; "
        "rerunning before a later end date is unlikely to change the outcome"
    )
    assert payload["operator_crypto_route_alignment_cooldown_done_when"] == (
        "hot_universe_research end date advances beyond 2026-03-10 or crypto_route focus changes"
    )
    assert payload["operator_crypto_route_alignment_recipe_status"] == "deferred_by_cooldown"
    assert payload["operator_crypto_route_alignment_recipe_brief"] == "deferred_by_cooldown:2026-03-11"
    assert payload["operator_crypto_route_alignment_recipe_blocker_detail"] == (
        "latest clean crypto recovery already evaluated data through 2026-03-10 and still found no edge; "
        "rerunning before a later end date is unlikely to change the outcome"
    )
    assert payload["operator_crypto_route_alignment_recipe_done_when"] == (
        "hot_universe_research end date advances beyond 2026-03-10 or crypto_route focus changes"
    )
    assert payload["operator_crypto_route_alignment_recipe_ready_on_date"] == "2026-03-11"
    assert payload["operator_focus_slot_actionability_backlog_brief"] == "primary:SOLUSDT:recovery_completed_no_edge"
    assert payload["operator_focus_slot_actionability_backlog_count"] == 1
    assert payload["operator_focus_slot_actionable_count"] == 2
    assert payload["operator_focus_slot_actionability_gate_brief"] == "actionability_guarded_by_content:2/3"
    assert payload["operator_focus_slot_actionability_gate_status"] == "actionability_guarded_by_content"
    assert payload["operator_focus_slot_actionability_gate_blocker_detail"] == (
        "SOLUSDT primary content state remains blocked "
        "(route_ahead_of_embedding, recovery_completed_no_edge): "
        "deploy-price-state-plus-beta-review"
    )
    assert payload["operator_focus_slot_actionability_gate_done_when"] == (
        "SOLUSDT regains a positive ranked flow edge or leaves review"
    )
    assert payload["operator_focus_slot_readiness_gate_ready_count"] == 1
    assert payload["operator_focus_slot_readiness_gate_brief"] == "readiness_guarded_by_source_freshness:1/3"
    assert payload["operator_focus_slot_readiness_gate_status"] == "readiness_guarded_by_source_freshness"
    assert payload["operator_focus_slot_readiness_gate_blocking_gate"] == "source_freshness"
    assert payload["operator_focus_slot_readiness_gate_blocker_detail"] == (
        "- followup source requires inspect_source_state (-, -, unknown)"
    )
    assert payload["operator_focus_slot_readiness_gate_done_when"] == "operator_focus_slot_refresh_backlog_count reaches 0"
    assert "research-embedding-quality: avoid_only:crypto_hot" in payload["summary_text"]
    assert "focus-slot-actionability-gate: actionability_guarded_by_content:2/3" in payload["summary_text"]
    assert "focus-slot-readiness-gate: readiness_guarded_by_source_freshness:1/3" in payload["summary_text"]
    assert "crypto-route-alignment-cooldown: cooldown_active_wait_for_new_market_data:>2026-03-10" in payload["summary_text"]
    assert "crypto-route-alignment-recovery-recipe: deferred_by_cooldown:2026-03-11" in payload["summary_text"]
    assert "crypto:SOLUSDT:deprioritize_flow" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prefers_dedicated_crypto_route_payload_over_embedded_route(
    tmp_path: Path,
) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260312T054623Z_hot_universe_research.json",
        {
            "status": "ok",
            "end": "2026-03-12",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_hot", "crypto_majors", "crypto_beta"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260312T054632Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-12T05:46:34+00:00",
            "operator_status": "deploy-price-state-plus-beta-review",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT | watch-priority:BNBUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "next_focus_reason": "Flow does not produce a positive ranked edge even in the short sample.",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-12T05:46:37Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_crypto_route_artifact"] == str(
        review_dir / "20260312T054632Z_crypto_route_operator_brief.json"
    )
    assert payload["crypto_route_status"] == "deploy-price-state-plus-beta-review"
    assert payload["crypto_route_stack_brief"] == (
        "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT | watch-priority:BNBUSDT"
    )
    assert payload["crypto_focus_symbol"] == "SOLUSDT"
    assert payload["crypto_focus_action"] == "deprioritize_flow"
    assert payload["next_focus_area"] == "crypto_route"
    assert payload["next_focus_symbol"] == "SOLUSDT"
    assert payload["next_focus_action"] == "deprioritize_flow"
    assert payload["next_focus_source_artifact"] == str(
        review_dir / "20260312T054632Z_crypto_route_operator_brief.json"
    )


def test_build_hot_universe_operator_brief_surfaces_crypto_route_refresh_audit(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260312T054623Z_hot_universe_research.json",
        {
            "status": "ok",
            "end": "2026-03-12",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_hot"],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-review",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT",
                "next_focus_symbol": "SOLUSDT",
                "next_focus_action": "deprioritize_flow",
                "next_focus_reason": "Flow does not produce a positive ranked edge even in the short sample.",
            },
        },
    )
    _write_json(
        review_dir / "20260312T054632Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-12T05:46:34+00:00",
            "operator_status": "deploy-price-state-plus-beta-review",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | review:SOLUSDT",
            "next_focus_symbol": "SOLUSDT",
            "next_focus_action": "deprioritize_flow",
            "next_focus_reason": "Flow does not produce a positive ranked edge even in the short sample.",
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
                    "route_status_label": "review",
                    "execution_state": "Bias_Only",
                    "micro_classification": "watch_only",
                    "micro_veto": "low_sample_or_gap_risk",
                    "priority_score": 73,
                    "priority_tier": "review_queue_now",
                    "reason": "Flow does not produce a positive ranked edge even in the short sample.",
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation.",
                    "done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation",
                    "rank": 1,
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260312T054633Z_crypto_route_refresh.json",
        {
            "status": "ok",
            "as_of": "2026-03-12T05:46:35Z",
            "crypto_route_refresh_reuse_level": "informational",
            "crypto_route_refresh_reuse_gate_status": "reuse_non_blocking",
            "crypto_route_refresh_reuse_gate_brief": "reuse_non_blocking:skip_native_refresh:2/2",
            "crypto_route_refresh_reuse_gate_blocking": False,
            "crypto_route_refresh_reuse_gate_blocker_detail": (
                "all tracked native steps were intentionally reused; current refresh remains safe for downstream consumption."
            ),
            "crypto_route_refresh_reuse_gate_done_when": (
                "run full native refresh only when fresh native recomputation is explicitly required"
            ),
            "native_refresh_mode": "skip_native_refresh",
            "steps": [
                {
                    "name": "native_custom",
                    "status": "reused_previous_artifact",
                    "artifact": str(review_dir / "20260312T054600Z_custom_binance_indicator_combo_native_crypto.json"),
                },
                {
                    "name": "native_majors",
                    "status": "reused_previous_artifact",
                    "artifact": str(review_dir / "20260312T054601Z_majors_binance_indicator_combo_native_crypto.json"),
                },
                {
                    "name": "build_source_control_report",
                    "status": "ok",
                    "artifact": str(review_dir / "20260312T054632Z_binance_indicator_source_control_report.json"),
                },
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-12T05:46:37Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_crypto_route_refresh_artifact"] == str(
        review_dir / "20260312T054633Z_crypto_route_refresh.json"
    )
    assert payload["source_crypto_route_refresh_status"] == "ok"
    assert payload["source_crypto_route_refresh_as_of"] == "2026-03-12T05:46:35Z"
    assert payload["source_crypto_route_refresh_native_mode"] == "skip_native_refresh"
    assert payload["source_crypto_route_refresh_native_step_count"] == 2
    assert payload["source_crypto_route_refresh_reused_native_count"] == 2
    assert payload["source_crypto_route_refresh_missing_reused_count"] == 0
    assert payload["source_crypto_route_refresh_reuse_status"] == "reused_native_inputs"
    assert payload["source_crypto_route_refresh_reuse_brief"] == (
        "reused_native_inputs:skip_native_refresh:2/2"
    )
    assert payload["source_crypto_route_refresh_reuse_level"] == "informational"
    assert payload["source_crypto_route_refresh_reuse_gate_status"] == "reuse_non_blocking"
    assert payload["source_crypto_route_refresh_reuse_gate_brief"] == (
        "reuse_non_blocking:skip_native_refresh:2/2"
    )
    assert payload["source_crypto_route_refresh_reuse_gate_blocking"] is False
    assert payload["source_crypto_route_refresh_reuse_gate_blocker_detail"] == (
        "all tracked native steps were intentionally reused; current refresh remains safe for downstream consumption."
    )
    assert payload["source_crypto_route_refresh_reuse_gate_done_when"] == (
        "run full native refresh only when fresh native recomputation is explicitly required"
    )
    assert payload["summary_text"] and (
        "crypto-route-refresh-reuse: reused_native_inputs:skip_native_refresh:2/2"
        in payload["summary_text"]
    )
    assert payload["summary_text"] and (
        "crypto-route-refresh-reuse-gate: reuse_non_blocking:skip_native_refresh:2/2"
        in payload["summary_text"]
    )


def test_build_hot_universe_operator_brief_emits_crypto_route_alignment_recovery_recipe(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    universe_path = review_dir / "20260310T120000Z_hot_research_universe.json"
    _write_json(
        universe_path,
        {
            "status": "ok",
            "batches": {
                "crypto_hot": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                "crypto_majors": ["BTCUSDT", "ETHUSDT"],
                "crypto_beta": ["BNBUSDT", "SOLUSDT"],
            },
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "ok",
            "start": "2026-03-07",
            "end": "2026-03-10",
            "universe_file": str(universe_path),
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_hot", "crypto_majors", "crypto_beta"],
            },
            "batch_summary": {
                "ranked_batches": [
                    {
                        "batch": "crypto_hot",
                        "status_label": "deprioritize",
                        "research_trades": 0,
                        "accepted_count": 0,
                    },
                    {
                        "batch": "crypto_majors",
                        "status_label": "deprioritize",
                        "research_trades": 0,
                        "accepted_count": 0,
                    },
                    {
                        "batch": "crypto_beta",
                        "status_label": "deprioritize",
                        "research_trades": 0,
                        "accepted_count": 0,
                    },
                ]
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T121500Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:15:00+00:00",
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
            "next_focus_reason": "BNB degrades on long window.",
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending",
            "execution_retro_status": "paper-execution-retro-pending",
            "next_retro_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_retro_execution_symbol": "XAUUSD",
            "retro_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "retro_status": "awaiting_paper_execution_retro",
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["operator_crypto_route_alignment_status"] == "route_ahead_of_embedding"
    assert payload["operator_crypto_route_alignment_brief"] == (
        "route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta"
    )
    assert payload["operator_crypto_route_alignment_recovery_status"] == "recovery_completed_no_edge"
    assert payload["operator_crypto_route_alignment_recovery_brief"] == (
        "recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta"
    )
    assert payload["operator_crypto_route_alignment_recovery_zero_trade_batches"] == [
        "crypto_hot",
        "crypto_majors",
        "crypto_beta",
    ]
    assert payload["operator_crypto_route_alignment_recipe_window_days"] == 21
    assert payload["operator_crypto_route_alignment_recipe_target_batches"] == [
        "crypto_hot",
        "crypto_majors",
        "crypto_beta",
    ]
    assert payload["operator_crypto_route_alignment_recipe_expected_status"] == "ok"
    assert payload["operator_crypto_route_alignment_recipe_script"] == str(
        SCRIPT_PATH.parent / "run_hot_universe_research.py"
    )
    assert "--run-strategy-lab" in payload["operator_crypto_route_alignment_recipe_command_hint"]
    assert "--start 2026-02-18" in payload["operator_crypto_route_alignment_recipe_command_hint"]
    assert f"--universe-file {shlex.quote(str(universe_path))}" in payload["operator_crypto_route_alignment_recipe_command_hint"]
    assert payload["operator_crypto_route_alignment_recipe_followup_script"] == str(
        SCRIPT_PATH.parent / "refresh_commodity_paper_execution_state.py"
    )
    assert (
        payload["operator_crypto_route_alignment_recipe_verify_hint"]
        == "confirm operator_research_embedding_quality_status leaves avoid_only or operator_crypto_route_alignment_status leaves route_ahead_of_embedding"
    )


def test_build_hot_universe_operator_brief_merges_action_and_crypto_sources(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_beta"],
            },
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_majors"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_retest_action": "wait_for_more_bnb_native_data",
            },
            "crypto_route_operator_brief": {
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
            },
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_action_status"] == "ok"
    assert payload["source_crypto_status"] == "dry_run"
    assert payload["source_mode"] == "merged-action-crypto-sources"
    assert payload["research_queue_batches"] == ["crypto_majors", "crypto_hot"]
    assert payload["crypto_focus_symbol"] == "BNBUSDT"
    assert payload["crypto_focus_window_floor"] == "positive_but_weaker"
    assert payload["crypto_xlong_flow_window_floor"] == "laggy_positive_only"
    assert "crypto-xlong-flow-floor: laggy_positive_only" in payload["summary_text"]


def test_build_hot_universe_operator_brief_merges_commodity_lane(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_beta"],
            },
            "crypto_route_brief": {},
        },
    )
    _write_json(
        review_dir / "20260310T121000Z_hot_universe_research.json",
        {
            "status": "dry_run",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": [],
                "shadow_only_batches": [],
                "avoid_batches": ["crypto_majors"],
            },
            "crypto_route_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
            "crypto_route_operator_brief": {
                "focus_window_floor": "positive_but_weaker",
                "price_state_window_floor": "negative",
                "comparative_window_takeaway": "Long-window flow holds up better than long-window price-state, but it is still too weak for promotion.",
                "xlong_flow_window_floor": "laggy_positive_only",
                "xlong_comparative_window_takeaway": "Extra-long flow keeps a raw positive return, but only in discarded laggy form; keep BNB in watch priority.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122000Z_commodity_execution_lane.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "execution_mode": "paper_first",
            "focus_primary_batches": ["metals_all", "precious_metals"],
            "focus_with_regime_filter_batches": ["energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "leader_symbols_primary": ["XAGUSD", "COPPER", "XAUUSD"],
            "leader_symbols_regime_filter": ["BRENTUSD", "WTIUSD"],
            "next_focus_batch": "metals_all",
            "next_focus_symbols": ["XAGUSD", "COPPER", "XAUUSD"],
            "next_stage": "paper_ticket_lane",
            "route_stack_brief": "paper-primary:metals_all,precious_metals | regime-filter:energy_liquids | shadow:commodities_benchmark",
        },
    )
    _write_json(
        review_dir / "20260310T122500Z_commodity_paper_ticket_lane.json",
        {
            "status": "ok",
            "ticket_status": "paper-ready",
            "ticket_stack_brief": "paper-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
            "paper_ready_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "next_ticket_batch": "metals_all",
            "next_ticket_symbols": ["XAGUSD", "COPPER", "XAUUSD"],
            "tickets": [
                {"ticket_id": "commodity-paper:metals_all"},
                {"ticket_id": "commodity-paper:precious_metals"},
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122700Z_commodity_paper_ticket_book.json",
        {
            "status": "ok",
            "ticket_book_status": "paper-ready",
            "ticket_book_stack_brief": "paper-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
            "actionable_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "shadow_batches": ["commodities_benchmark"],
            "next_ticket_id": "commodity-paper-ticket:metals_all:XAUUSD",
            "next_ticket_batch": "metals_all",
            "next_ticket_symbol": "XAUUSD",
            "actionable_ticket_count": 7,
            "tickets": [
                {"ticket_id": "commodity-paper-ticket:metals_all:XAUUSD"},
                {"ticket_id": "commodity-paper-ticket:precious_metals:XAGUSD"},
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122720Z_commodity_paper_execution_preview.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_mode": "paper_only",
            "preview_ready_batches": ["metals_all", "precious_metals", "energy_liquids"],
            "shadow_only_batches": ["commodities_benchmark"],
            "preview_batch_count": 4,
            "next_execution_batch": "metals_all",
            "next_execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "next_execution_regime_gate": "paper_only",
            "next_execution_weight_hint_sum": 2.3,
            "preview_stack_brief": "paper-execution-ready:metals_all,precious_metals,energy_liquids | shadow:commodities_benchmark",
        },
    )
    _write_json(
        review_dir / "20260310T122740Z_commodity_paper_execution_artifact.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "execution_item_count": 3,
            "actionable_execution_item_count": 3,
            "execution_stack_brief": "paper-execution-artifact:metals_all:XAUUSD, XAGUSD, COPPER",
            "execution_items": [
                {"execution_id": "commodity-paper-execution:metals_all:XAUUSD", "symbol": "XAUUSD"},
                {"execution_id": "commodity-paper-execution:metals_all:XAGUSD", "symbol": "XAGUSD"},
                {"execution_id": "commodity-paper-execution:metals_all:COPPER", "symbol": "COPPER"},
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "execution_item_count": 3,
            "actionable_execution_item_count": 3,
            "queue_depth": 3,
            "actionable_queue_depth": 3,
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "source_execution_status": "planned",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_review_status": "paper-execution-review-pending",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_regime_gate": "paper_only",
            "review_item_count": 3,
            "actionable_review_item_count": 3,
            "next_review_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_review_execution_symbol": "XAUUSD",
            "review_stack_brief": "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER",
            "review_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "review_status": "awaiting_paper_execution_review",
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_mode"] == "merged-action-commodity-crypto-sources"
    assert payload["commodity_route_status"] == "paper-first"
    assert payload["commodity_ticket_status"] == "paper-ready"
    assert payload["commodity_ticket_book_status"] == "paper-ready"
    assert payload["commodity_focus_batch"] == "metals_all"
    assert payload["commodity_focus_symbols"] == ["XAGUSD", "COPPER", "XAUUSD"]
    assert payload["commodity_ticket_focus_batch"] == "metals_all"
    assert payload["commodity_ticket_focus_symbols"] == ["XAGUSD", "COPPER", "XAUUSD"]
    assert payload["commodity_next_ticket_id"] == "commodity-paper-ticket:metals_all:XAUUSD"
    assert payload["commodity_next_ticket_symbol"] == "XAUUSD"
    assert payload["commodity_actionable_ticket_count"] == 7
    assert payload["commodity_execution_preview_status"] == "paper-execution-ready"
    assert payload["commodity_next_execution_batch"] == "metals_all"
    assert payload["commodity_next_execution_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["commodity_next_execution_regime_gate"] == "paper_only"
    assert payload["commodity_execution_artifact_status"] == "paper-execution-artifact-ready"
    assert payload["commodity_execution_batch"] == "metals_all"
    assert payload["commodity_execution_symbols"] == ["XAUUSD", "XAGUSD", "COPPER"]
    assert payload["commodity_actionable_execution_item_count"] == 3
    assert payload["commodity_execution_queue_status"] == "paper-execution-queued"
    assert payload["commodity_queue_depth"] == 3
    assert payload["commodity_actionable_queue_depth"] == 3
    assert payload["commodity_next_queue_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_queue_execution_symbol"] == "XAUUSD"
    assert payload["commodity_execution_review_status"] == "paper-execution-review-pending"
    assert payload["commodity_review_item_count"] == 3
    assert payload["commodity_actionable_review_item_count"] == 3
    assert payload["commodity_next_review_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_review_execution_symbol"] == "XAUUSD"
    assert payload["source_commodity_execution_queue_status"] == "ok"
    assert payload["source_commodity_execution_artifact_status"] == "ok"
    assert payload["source_commodity_execution_review_status"] == "ok"
    assert payload["operator_status"] == "commodity-paper-execution-review-pending-plus-crypto-deploy-watch"
    assert "commodity-route: paper-primary:metals_all,precious_metals | regime-filter:energy_liquids | shadow:commodities_benchmark" in payload["summary_text"]
    assert "commodity-ticket-status: paper-ready" in payload["summary_text"]
    assert "commodity-ticket-book-status: paper-ready" in payload["summary_text"]
    assert "commodity-execution-preview-status: paper-execution-ready" in payload["summary_text"]
    assert "commodity-execution-artifact-status: paper-execution-artifact-ready" in payload["summary_text"]
    assert "commodity-execution-review-status: paper-execution-review-pending" in payload["summary_text"]
    assert "commodity-execution-queue-status: paper-execution-queued" in payload["summary_text"]


def test_build_hot_universe_operator_brief_merges_commodity_execution_retro(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": [],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_focus_reason": "BNB degrades on long window.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T120000Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_brief.json"),
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
            "next_focus_reason": "BNB degrades on long window.",
            "review_priority_queue_status": "ready",
            "review_priority_queue_count": 2,
            "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:20",
            "review_priority_head_symbol": "SOLUSDT",
            "review_priority_head_tier": "review_queue_now",
            "review_priority_head_score": 73,
            "review_priority_queue": [
                {
                    "symbol": "SOLUSDT",
                    "route_action": "deprioritize_flow",
                    "route_status_label": "review",
                    "execution_state": "Bias_Only",
                    "micro_classification": "watch_only",
                    "micro_veto": "low_sample_or_gap_risk",
                    "priority_score": 73,
                    "priority_tier": "review_queue_now",
                    "reason": "Flow does not produce a positive ranked edge even in the short sample.",
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade.",
                    "done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
                    "rank": 1,
                },
                {
                    "symbol": "BNBUSDT",
                    "route_action": "watch_only",
                    "route_status_label": "watch_only",
                    "execution_state": "Bias_Only",
                    "micro_classification": "-",
                    "micro_veto": "missing_micro_capture",
                    "priority_score": 20,
                    "priority_tier": "watch_queue_only",
                    "reason": "BNB keeps a raw positive flow return on the extra-long window, but only in discarded laggy form.",
                    "blocker_detail": "BNBUSDT remains Bias_Only; missing profile_location=MID, liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:watch_only; no_sweep_no_mss_no_cvd_no_trade.",
                    "done_when": "BNBUSDT completes profile_location=MID and liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:watch_only",
                    "rank": 2,
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_review_status": "paper-execution-review-pending",
            "execution_retro_status": "paper-execution-retro-pending",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "retro_item_count": 3,
            "actionable_retro_item_count": 3,
            "next_retro_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_retro_execution_symbol": "XAUUSD",
            "retro_stack_brief": "paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER",
            "retro_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "retro_status": "awaiting_paper_execution_retro",
                    "review_status": "awaiting_paper_execution_review",
                }
            ],
        },
    )

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--review-dir", str(review_dir)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_retro_status"] == "paper-execution-retro-pending"
    assert payload["commodity_retro_item_count"] == 3
    assert payload["commodity_actionable_retro_item_count"] == 3
    assert payload["commodity_next_retro_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_retro_execution_symbol"] == "XAUUSD"
    assert payload["source_commodity_execution_retro_status"] == "ok"
    assert payload["operator_status"] == "commodity-paper-execution-retro-pending-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:retro:XAUUSD | crypto:SOLUSDT:deprioritize_flow"
    assert payload["next_focus_area"] == "commodity_execution_retro"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "review_paper_execution_retro"
    assert payload["next_focus_reason"] == "paper_execution_retro_pending"
    assert payload["followup_focus_area"] == "crypto_route"
    assert payload["followup_focus_target"] == "SOLUSDT"
    assert payload["followup_focus_symbol"] == "SOLUSDT"
    assert payload["followup_focus_action"] == "deprioritize_flow"
    assert payload["followup_focus_reason"] == "Flow does not produce a positive ranked edge even in the short sample."
    assert payload["followup_focus_state"] == "review"
    assert payload["followup_focus_blocker_detail"] == (
        "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade."
    )
    assert payload["followup_focus_done_when"] == (
        "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow"
    )
    assert payload["secondary_focus_area"] == "crypto_route"
    assert payload["secondary_focus_target"] == "SOLUSDT"
    assert payload["secondary_focus_symbol"] == "SOLUSDT"
    assert payload["secondary_focus_action"] == "deprioritize_flow"
    assert payload["secondary_focus_reason"] == "Flow does not produce a positive ranked edge even in the short sample."
    assert payload["operator_crypto_route_alignment_focus_area"] == "crypto_route"
    assert payload["operator_crypto_route_alignment_focus_slot"] == "followup"
    assert payload["operator_crypto_route_alignment_focus_symbol"] == "SOLUSDT"
    assert payload["operator_crypto_route_alignment_focus_action"] == "deprioritize_flow"
    assert payload["operator_crypto_route_alignment_status"] == "aligned"
    assert payload["operator_crypto_route_alignment_brief"] == "aligned:SOLUSDT:crypto_majors, crypto_hot"
    assert "secondary-focus-priority: review_queue_now | score=73 | queue_rank=1" in payload["summary_text"]
    assert "crypto-route-alignment: aligned:SOLUSDT:crypto_majors, crypto_hot" in payload["summary_text"]
    assert "commodity-execution-retro-status: paper-execution-retro-pending" in payload["summary_text"]


def test_build_hot_universe_operator_brief_keeps_retro_focus_for_partial_fill_remainder(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_research_universe.json",
        {
            "status": "ok",
            "batches": {
                "crypto_hot": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
                "crypto_majors": ["BTCUSDT", "ETHUSDT"],
                "crypto_beta": ["BNBUSDT", "SOLUSDT"],
                "metals_all": ["XAUUSD", "XAGUSD", "COPPER"],
            },
        },
    )
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
                "next_focus_reason": "BNB degrades on long window.",
            },
        },
    )
    _write_json(
        review_dir / "20260310T120000Z_crypto_route_operator_brief.json",
        {
            "status": "ok",
            "as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_brief.json"),
            "operator_status": "deploy-price-state-plus-beta-watch",
            "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
            "next_focus_symbol": "BNBUSDT",
            "next_focus_action": "watch_priority_until_long_window_confirms",
            "next_focus_reason": "BNB degrades on long window.",
            "review_priority_queue_status": "ready",
            "review_priority_queue_count": 2,
            "review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:20",
            "review_priority_head_symbol": "SOLUSDT",
            "review_priority_head_tier": "review_queue_now",
            "review_priority_head_score": 73,
            "review_priority_queue": [
                {
                    "symbol": "SOLUSDT",
                    "route_action": "deprioritize_flow",
                    "route_status_label": "review",
                    "execution_state": "Bias_Only",
                    "micro_classification": "watch_only",
                    "micro_veto": "low_sample_or_gap_risk",
                    "priority_score": 73,
                    "priority_tier": "review_queue_now",
                    "reason": "Flow does not produce a positive ranked edge even in the short sample.",
                    "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade.",
                    "done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
                    "rank": 1,
                },
                {
                    "symbol": "BNBUSDT",
                    "route_action": "watch_only",
                    "route_status_label": "watch_only",
                    "execution_state": "Bias_Only",
                    "micro_classification": "-",
                    "micro_veto": "missing_micro_capture",
                    "priority_score": 20,
                    "priority_tier": "watch_queue_only",
                    "reason": "BNB keeps a raw positive flow return on the extra-long window, but only in discarded laggy form.",
                    "blocker_detail": "BNBUSDT remains Bias_Only; missing profile_location=MID, liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:watch_only; no_sweep_no_mss_no_cvd_no_trade.",
                    "done_when": "BNBUSDT completes profile_location=MID and liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:watch_only",
                    "rank": 2,
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending-fill-remainder",
            "actionable_review_item_count": 1,
            "review_pending_symbols": ["XAUUSD"],
            "fill_evidence_pending_count": 2,
            "fill_evidence_pending_symbols": ["XAGUSD", "COPPER"],
            "next_review_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_review_execution_symbol": "XAUUSD",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_fill_evidence_execution_symbol": "XAGUSD",
            "review_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "review_status": "awaiting_paper_execution_review",
                    "paper_execution_evidence_present": True,
                    "paper_entry_price": 5198.10009765625,
                    "paper_stop_price": 4847.7998046875,
                    "paper_target_price": 5758.58056640625,
                    "paper_quote_usdt": 0.15896067200583952,
                    "paper_execution_status": "OPEN",
                    "paper_signal_price_reference_source": "yfinance:GC=F",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending-fill-remainder",
            "execution_retro_status": "paper-execution-close-evidence-pending-fill-remainder",
            "actionable_retro_item_count": 0,
            "close_evidence_pending_count": 1,
            "close_evidence_pending_symbols": ["XAUUSD"],
            "fill_evidence_pending_count": 2,
            "fill_evidence_pending_symbols": ["XAGUSD", "COPPER"],
            "next_retro_execution_id": "",
            "next_retro_execution_symbol": "",
            "next_close_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_close_evidence_execution_symbol": "XAUUSD",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_fill_evidence_execution_symbol": "XAGUSD",
            "retro_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "retro_status": "awaiting_paper_execution_close_evidence",
                    "paper_execution_evidence_present": True,
                    "paper_entry_price": 5198.10009765625,
                    "paper_stop_price": 4847.7998046875,
                    "paper_target_price": 5758.58056640625,
                    "paper_quote_usdt": 0.15896067200583952,
                    "paper_execution_status": "OPEN",
                    "paper_signal_price_reference_source": "yfinance:GC=F",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent for XAGUSD and COPPER."],
            "queue_symbols_with_stale_directional_signal_dates": {
                "XAGUSD": "2026-01-26",
                "COPPER": "2026-01-29",
            },
            "queue_symbols_with_stale_directional_signal_age_days": {
                "XAGUSD": 42,
                "COPPER": 39,
            },
            "stale_directional_signal_watch_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                    "symbol": "XAGUSD",
                    "signal_date": "2026-01-26",
                    "signal_age_days": 42,
                },
                {
                    "execution_id": "commodity-paper-execution:metals_all:COPPER",
                    "symbol": "COPPER",
                    "signal_date": "2026-01-29",
                    "signal_age_days": 39,
                },
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "bridge_partially_bridged_stale_remainder",
            "already_present_count": 1,
            "already_bridged_symbols": ["XAUUSD"],
            "signal_stale_count": 2,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_blocked_symbol": "XAGUSD",
        },
    )

    proc = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--review-dir", str(review_dir), "--now", "2026-03-10T12:30:00Z"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_review_status"] == "paper-execution-close-evidence-pending-fill-remainder"
    assert payload["commodity_execution_retro_status"] == "paper-execution-close-evidence-pending-fill-remainder"
    assert payload["operator_status"] == "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:close-evidence:XAUUSD | crypto:SOLUSDT:deprioritize_flow"
    assert payload["next_focus_area"] == "commodity_execution_close_evidence"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "wait_for_paper_execution_close_evidence"
    assert payload["next_focus_reason"] == "paper_execution_close_evidence_pending"
    assert payload["next_focus_state"] == "waiting"
    assert payload["next_focus_blocker_detail"] == (
        "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
    )
    assert payload["next_focus_done_when"] == (
        "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
    )
    assert payload["followup_focus_area"] == "commodity_fill_evidence"
    assert payload["followup_focus_target"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["followup_focus_symbol"] == "XAGUSD"
    assert payload["followup_focus_action"] == "wait_for_paper_execution_fill_evidence"
    assert payload["followup_focus_reason"] == "paper_execution_fill_evidence_pending"
    assert payload["followup_focus_state"] == "waiting"
    assert payload["followup_focus_blocker_detail"] == (
        "paper execution fill evidence not written; stale directional signal 42d since 2026-01-26"
    )
    assert payload["followup_focus_done_when"] == "XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols"
    assert payload["operator_focus_slots_brief"] == (
        "primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence"
        " | followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence"
        " | secondary:review:SOLUSDT:deprioritize_flow"
    )
    assert payload["operator_focus_slot_sources_brief"] == (
        "primary:commodity_execution_retro | followup:commodity_execution_review | secondary:crypto_route"
    )
    assert payload["operator_focus_slot_status_brief"] == (
        "primary:ok@2026-03-10T12:27:55+00:00"
        " | followup:ok@2026-03-10T12:27:51+00:00"
        " | secondary:ok@2026-03-10T12:00:00+00:00"
    )
    assert payload["operator_focus_slot_recency_brief"] == (
        "primary:fresh:2m | followup:fresh:2m | secondary:carry_over:30m"
    )
    assert payload["operator_focus_slot_health_brief"] == (
        "primary:ready:read_current_artifact"
        " | followup:ready:read_current_artifact"
        " | secondary:carry_over_ok:consider_refresh_before_promotion"
    )
    assert payload["operator_focus_slot_refresh_backlog_brief"] == (
        "secondary:SOLUSDT:consider_refresh_before_promotion"
    )
    assert payload["operator_focus_slot_refresh_backlog_count"] == 1
    assert payload["operator_focus_slot_ready_count"] == 2
    assert payload["operator_focus_slot_total_count"] == 3
    assert payload["operator_focus_slot_promotion_gate_brief"] == (
        "promotion_guarded_by_source_freshness:2/3"
    )
    assert payload["operator_focus_slot_promotion_gate_status"] == "promotion_guarded_by_source_freshness"
    assert payload["operator_focus_slot_promotion_gate_blocker_detail"] == (
        "SOLUSDT secondary source requires consider_refresh_before_promotion "
        "(crypto_route, ok, carry_over, age=30m)"
    )
    assert payload["operator_focus_slot_promotion_gate_done_when"] == (
        "operator_focus_slot_refresh_backlog_count reaches 0"
    )
    assert payload["operator_focus_slot_readiness_gate_ready_count"] == 2
    assert payload["operator_focus_slot_readiness_gate_brief"] == (
        "readiness_guarded_by_source_freshness:2/3"
    )
    assert payload["operator_focus_slot_readiness_gate_status"] == "readiness_guarded_by_source_freshness"
    assert payload["operator_focus_slot_readiness_gate_blocking_gate"] == "source_freshness"
    assert payload["operator_focus_slot_readiness_gate_blocker_detail"] == (
        "SOLUSDT secondary source requires consider_refresh_before_promotion "
        "(crypto_route, ok, carry_over, age=30m)"
    )
    assert payload["operator_focus_slot_readiness_gate_done_when"] == (
        "operator_focus_slot_refresh_backlog_count reaches 0"
    )
    assert payload["operator_crypto_route_alignment_status"] == "aligned"
    assert payload["operator_crypto_route_alignment_brief"] == "aligned:SOLUSDT:crypto_hot"
    assert payload["operator_crypto_route_alignment_focus_area"] == "crypto_route"
    assert payload["operator_crypto_route_alignment_focus_slot"] == "secondary"
    assert payload["operator_crypto_route_alignment_focus_symbol"] == "SOLUSDT"
    assert payload["operator_crypto_route_alignment_focus_action"] == "deprioritize_flow"
    assert payload["operator_source_refresh_queue_brief"] == (
        "1:secondary:SOLUSDT:consider_refresh_before_promotion"
    )
    assert payload["operator_source_refresh_queue_count"] == 1
    assert payload["operator_source_refresh_checklist_brief"] == (
        "1:refresh_recommended:SOLUSDT:consider_refresh_before_promotion"
    )
    expected_refresh_recipe_script = str(SCRIPT_PATH.parent / "refresh_crypto_route_state.py")
    expected_refresh_recipe_command = shlex.join(
        [
            "python3",
            expected_refresh_recipe_script,
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(review_dir.parent),
            "--now",
            "2026-03-10T12:30:00Z",
        ]
    )
    expected_refresh_operator_script = str(SCRIPT_PATH.parent / "build_crypto_route_operator_brief.py")
    expected_refresh_operator_command = shlex.join(
        [
            "python3",
            expected_refresh_operator_script,
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ]
    )
    expected_refresh_recipe_artifact_kind = "crypto_route_brief"
    expected_refresh_recipe_artifact_path_hint = str(review_dir / "*_crypto_route_brief.json")
    expected_refresh_operator_artifact_kind = "crypto_route_operator_brief"
    expected_refresh_operator_artifact_path_hint = str(review_dir / "*_crypto_route_operator_brief.json")
    expected_refresh_research_script = str(SCRIPT_PATH.parent / "run_hot_universe_research.py")
    expected_refresh_research_command = shlex.join(
        [
            "python3",
            expected_refresh_research_script,
            "--output-root",
            str(review_dir.parent),
            "--review-dir",
            str(review_dir),
            "--start",
            "2026-03-07",
            "--end",
            "2026-03-10",
            "--now",
            "2026-03-10T12:30:00Z",
            "--universe-file",
            str(review_dir / "20260310T120000Z_hot_research_universe.json"),
            "--batch",
            "crypto_hot",
            "--batch",
            "crypto_majors",
            "--batch",
            "crypto_beta",
        ]
    )
    expected_refresh_followup_script = str(SCRIPT_PATH.parent / "refresh_commodity_paper_execution_state.py")
    expected_refresh_followup_command = shlex.join(
        [
            "python3",
            expected_refresh_followup_script,
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(review_dir.parent),
            "--context-path",
            str(review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"),
        ]
    )
    expected_refresh_research_artifact_kind = "hot_universe_research"
    expected_refresh_research_artifact_path_hint = str(review_dir / "*_hot_universe_research.json")
    expected_refresh_followup_artifact_kind = "commodity_paper_execution_refresh"
    expected_refresh_followup_artifact_path_hint = str(review_dir / "*_commodity_paper_execution_refresh.json")
    expected_refresh_step_checkpoint_brief = (
        "1:missing:crypto_route_brief | 2:carry_over:crypto_route_operator_brief"
        " | 3:carry_over:hot_universe_research | 4:missing:commodity_paper_execution_refresh"
    )
    expected_refresh_pipeline_pending_brief = (
        "1:refresh_crypto_route_brief:missing:crypto_route_brief"
        " | 2:refresh_crypto_route_operator_brief:carry_over:crypto_route_operator_brief"
        " | 3:refresh_hot_universe_research_embedding:carry_over:hot_universe_research"
        " | 4:refresh_commodity_handoff:missing:commodity_paper_execution_refresh"
    )
    assert payload["next_focus_source_kind"] == "commodity_execution_retro"
    assert payload["next_focus_source_artifact"] == str(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json"
    )
    assert payload["next_focus_source_status"] == "ok"
    assert payload["next_focus_source_as_of"] == "2026-03-10T12:27:55+00:00"
    assert payload["next_focus_source_age_minutes"] == 2
    assert payload["next_focus_source_recency"] == "fresh"
    assert payload["next_focus_source_health"] == "ready"
    assert payload["next_focus_source_refresh_action"] == "read_current_artifact"
    assert payload["followup_focus_source_kind"] == "commodity_execution_review"
    assert payload["followup_focus_source_artifact"] == str(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json"
    )
    assert payload["followup_focus_source_status"] == "ok"
    assert payload["followup_focus_source_as_of"] == "2026-03-10T12:27:51+00:00"
    assert payload["followup_focus_source_age_minutes"] == 2
    assert payload["followup_focus_source_recency"] == "fresh"
    assert payload["followup_focus_source_health"] == "ready"
    assert payload["followup_focus_source_refresh_action"] == "read_current_artifact"
    assert payload["secondary_focus_source_kind"] == "crypto_route"
    assert payload["secondary_focus_source_artifact"] == str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json")
    assert payload["secondary_focus_source_status"] == "ok"
    assert payload["secondary_focus_source_as_of"] == "2026-03-10T12:00:00+00:00"
    assert payload["secondary_focus_source_age_minutes"] == 30
    assert payload["secondary_focus_source_recency"] == "carry_over"
    assert payload["secondary_focus_source_health"] == "carry_over_ok"
    assert payload["secondary_focus_source_refresh_action"] == "consider_refresh_before_promotion"
    assert payload["operator_focus_slot_refresh_head_slot"] == "secondary"
    assert payload["operator_focus_slot_refresh_head_symbol"] == "SOLUSDT"
    assert payload["operator_focus_slot_refresh_head_action"] == "consider_refresh_before_promotion"
    assert payload["operator_focus_slot_refresh_head_health"] == "carry_over_ok"
    assert payload["operator_source_refresh_next_slot"] == "secondary"
    assert payload["operator_source_refresh_next_symbol"] == "SOLUSDT"
    assert payload["operator_source_refresh_next_action"] == "consider_refresh_before_promotion"
    assert payload["operator_source_refresh_next_source_kind"] == "crypto_route"
    assert payload["operator_source_refresh_next_source_health"] == "carry_over_ok"
    assert payload["operator_source_refresh_next_source_artifact"] == str(
        review_dir / "20260310T120000Z_crypto_route_operator_brief.json"
    )
    assert payload["operator_source_refresh_next_state"] == "refresh_recommended"
    assert payload["operator_source_refresh_next_blocker_detail"] == (
        "crypto_route artifact is ok and carry_over, age=30m"
    )
    assert payload["operator_source_refresh_next_done_when"] == (
        "SOLUSDT receives a fresh crypto_route artifact before promotion"
    )
    assert payload["operator_source_refresh_next_recipe_script"] == expected_refresh_recipe_script
    assert payload["operator_source_refresh_next_recipe_command_hint"] == expected_refresh_recipe_command
    assert payload["operator_source_refresh_next_recipe_expected_status"] == "ok"
    assert payload["operator_source_refresh_next_recipe_expected_artifact_kind"] == expected_refresh_recipe_artifact_kind
    assert payload["operator_source_refresh_next_recipe_expected_artifact_path_hint"] == (
        expected_refresh_recipe_artifact_path_hint
    )
    assert payload["operator_source_refresh_next_recipe_note"] == (
        "guarded entrypoint refreshes native crypto route sources before the remaining pipeline steps"
    )
    assert payload["operator_source_refresh_next_recipe_followup_script"] == expected_refresh_followup_script
    assert payload["operator_source_refresh_next_recipe_followup_command_hint"] == expected_refresh_followup_command
    assert payload["operator_source_refresh_next_recipe_verify_hint"] == (
        "rerun commodity refresh and confirm SOLUSDT leaves operator_source_refresh_queue"
    )
    assert payload["crypto_route_head_source_refresh_status"] == "refresh_recommended"
    assert payload["crypto_route_head_source_refresh_brief"] == (
        "refresh_recommended:SOLUSDT:consider_refresh_before_promotion"
    )
    assert payload["crypto_route_head_source_refresh_slot"] == "secondary"
    assert payload["crypto_route_head_source_refresh_symbol"] == "SOLUSDT"
    assert payload["crypto_route_head_source_refresh_action"] == "consider_refresh_before_promotion"
    assert payload["crypto_route_head_source_refresh_source_kind"] == "crypto_route"
    assert payload["crypto_route_head_source_refresh_source_health"] == "carry_over_ok"
    assert payload["crypto_route_head_source_refresh_source_artifact"] == str(
        review_dir / "20260310T120000Z_crypto_route_operator_brief.json"
    )
    assert payload["crypto_route_head_source_refresh_blocker_detail"] == (
        "crypto_route artifact is ok and carry_over, age=30m"
    )
    assert payload["crypto_route_head_source_refresh_done_when"] == (
        "SOLUSDT receives a fresh crypto_route artifact before promotion"
    )
    assert payload["crypto_route_head_source_refresh_recipe_script"] == expected_refresh_recipe_script
    assert payload["crypto_route_head_source_refresh_recipe_expected_status"] == "ok"
    assert payload["crypto_route_head_source_refresh_recipe_expected_artifact_kind"] == (
        expected_refresh_recipe_artifact_kind
    )
    assert payload["crypto_route_head_source_refresh_recipe_expected_artifact_path_hint"] == (
        expected_refresh_recipe_artifact_path_hint
    )
    assert payload["crypto_route_head_source_refresh_recipe_followup_script"] == expected_refresh_followup_script
    assert payload["crypto_route_head_source_refresh_recipe_verify_hint"] == (
        "rerun commodity refresh and confirm SOLUSDT leaves operator_source_refresh_queue"
    )
    assert payload["crypto_route_head_source_refresh_recipe_steps_brief"] == (
        "1:refresh_crypto_route_brief | 2:refresh_crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding | 4:refresh_commodity_handoff"
    )
    assert payload["operator_source_refresh_next_recipe_steps_brief"] == (
        "1:refresh_crypto_route_brief | 2:refresh_crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding | 4:refresh_commodity_handoff"
    )
    assert payload["operator_source_refresh_next_recipe_step_checkpoint_brief"] == (
        expected_refresh_step_checkpoint_brief
    )
    assert payload["operator_source_refresh_pipeline_steps_brief"] == (
        "1:refresh_crypto_route_brief | 2:refresh_crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding | 4:refresh_commodity_handoff"
    )
    assert payload["operator_source_refresh_pipeline_step_checkpoint_brief"] == expected_refresh_step_checkpoint_brief
    assert payload["operator_source_refresh_pipeline_pending_brief"] == expected_refresh_pipeline_pending_brief
    assert payload["operator_source_refresh_pipeline_pending_count"] == 4
    assert payload["operator_source_refresh_pipeline_head_rank"] == "1"
    assert payload["operator_source_refresh_pipeline_head_name"] == "refresh_crypto_route_brief"
    assert payload["operator_source_refresh_pipeline_head_checkpoint_state"] == "missing"
    assert payload["operator_source_refresh_pipeline_head_expected_artifact_kind"] == "crypto_route_brief"
    assert payload["operator_source_refresh_pipeline_head_current_artifact"] == "-"
    assert payload["operator_source_refresh_pipeline_relevance_status"] == "blocking_for_current_crypto_head"
    assert payload["operator_source_refresh_pipeline_relevance_brief"] == (
        "blocking_for_current_crypto_head:SOLUSDT:4"
    )
    assert payload["operator_source_refresh_next_recipe_steps"] == [
            {
                "rank": 1,
                "name": "refresh_crypto_route_brief",
                "script": expected_refresh_recipe_script,
                "command_hint": expected_refresh_recipe_command,
                "expected_status": "ok",
                "expected_artifact_kind": expected_refresh_recipe_artifact_kind,
                "expected_artifact_path_hint": expected_refresh_recipe_artifact_path_hint,
                "current_artifact": "",
                "current_status": "",
                "current_as_of": "",
                "current_age_minutes": None,
                "current_recency": "unknown",
                "checkpoint_state": "missing",
            },
            {
                "rank": 2,
                "name": "refresh_crypto_route_operator_brief",
                "script": expected_refresh_operator_script,
                "command_hint": expected_refresh_operator_command,
                "expected_status": "ok",
                "expected_artifact_kind": expected_refresh_operator_artifact_kind,
                "expected_artifact_path_hint": expected_refresh_operator_artifact_path_hint,
                "current_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
                "current_status": "ok",
                "current_as_of": "2026-03-10T12:00:00+00:00",
                "current_age_minutes": 30,
                "current_recency": "carry_over",
                "checkpoint_state": "carry_over",
            },
            {
                "rank": 3,
                "name": "refresh_hot_universe_research_embedding",
                "script": expected_refresh_research_script,
                "command_hint": expected_refresh_research_command,
                "expected_status": "ok",
                "expected_artifact_kind": expected_refresh_research_artifact_kind,
                "expected_artifact_path_hint": expected_refresh_research_artifact_path_hint,
                "current_artifact": str(review_dir / "20260310T120000Z_hot_universe_research.json"),
                "current_status": "ok",
                "current_as_of": "2026-03-10T12:00:00+00:00",
                "current_age_minutes": 30,
                "current_recency": "carry_over",
                "checkpoint_state": "carry_over",
            },
            {
                "rank": 4,
                "name": "refresh_commodity_handoff",
                "script": expected_refresh_followup_script,
                "command_hint": expected_refresh_followup_command,
                "expected_status": "ok",
                "expected_artifact_kind": expected_refresh_followup_artifact_kind,
                "expected_artifact_path_hint": expected_refresh_followup_artifact_path_hint,
                "current_artifact": "",
                "current_status": "",
                "current_as_of": "",
                "current_age_minutes": None,
                "current_recency": "unknown",
                "checkpoint_state": "missing",
            },
        ]
    assert payload["operator_focus_slot_refresh_backlog"] == [
        {
            "slot": "secondary",
            "symbol": "SOLUSDT",
            "action": "consider_refresh_before_promotion",
            "source_kind": "crypto_route",
            "source_status": "ok",
            "source_recency": "carry_over",
            "source_health": "carry_over_ok",
            "source_age_minutes": 30,
            "source_as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
        }
    ]
    assert payload["operator_source_refresh_queue"] == [
        {
            "rank": 1,
            "slot": "secondary",
            "symbol": "SOLUSDT",
            "action": "consider_refresh_before_promotion",
            "source_kind": "crypto_route",
            "source_status": "ok",
            "source_recency": "carry_over",
            "source_health": "carry_over_ok",
            "source_age_minutes": 30,
            "source_as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
        }
    ]
    assert payload["operator_source_refresh_checklist"] == [
        {
            "rank": 1,
            "slot": "secondary",
            "symbol": "SOLUSDT",
            "action": "consider_refresh_before_promotion",
            "source_kind": "crypto_route",
            "source_status": "ok",
            "source_recency": "carry_over",
            "source_health": "carry_over_ok",
            "source_age_minutes": 30,
            "source_as_of": "2026-03-10T12:00:00+00:00",
            "source_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
            "state": "refresh_recommended",
            "blocker_detail": "crypto_route artifact is ok and carry_over, age=30m",
            "done_when": "SOLUSDT receives a fresh crypto_route artifact before promotion",
            "recipe_script": expected_refresh_recipe_script,
            "recipe_command_hint": expected_refresh_recipe_command,
            "recipe_expected_status": "ok",
            "recipe_expected_artifact_kind": expected_refresh_recipe_artifact_kind,
            "recipe_expected_artifact_path_hint": expected_refresh_recipe_artifact_path_hint,
            "recipe_note": "guarded entrypoint refreshes native crypto route sources before the remaining pipeline steps",
            "recipe_followup_script": expected_refresh_followup_script,
            "recipe_followup_command_hint": expected_refresh_followup_command,
            "recipe_verify_hint": "rerun commodity refresh and confirm SOLUSDT leaves operator_source_refresh_queue",
            "recipe_steps_brief": "1:refresh_crypto_route_brief | 2:refresh_crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding | 4:refresh_commodity_handoff",
            "recipe_step_checkpoint_brief": expected_refresh_step_checkpoint_brief,
            "recipe_steps": [
                    {
                        "rank": 1,
                        "name": "refresh_crypto_route_brief",
                        "script": expected_refresh_recipe_script,
                        "command_hint": expected_refresh_recipe_command,
                        "expected_status": "ok",
                        "expected_artifact_kind": expected_refresh_recipe_artifact_kind,
                        "expected_artifact_path_hint": expected_refresh_recipe_artifact_path_hint,
                        "current_artifact": "",
                        "current_status": "",
                        "current_as_of": "",
                        "current_age_minutes": None,
                        "current_recency": "unknown",
                        "checkpoint_state": "missing",
                    },
                    {
                        "rank": 2,
                        "name": "refresh_crypto_route_operator_brief",
                        "script": expected_refresh_operator_script,
                        "command_hint": expected_refresh_operator_command,
                        "expected_status": "ok",
                        "expected_artifact_kind": expected_refresh_operator_artifact_kind,
                        "expected_artifact_path_hint": expected_refresh_operator_artifact_path_hint,
                        "current_artifact": str(review_dir / "20260310T120000Z_crypto_route_operator_brief.json"),
                        "current_status": "ok",
                        "current_as_of": "2026-03-10T12:00:00+00:00",
                        "current_age_minutes": 30,
                        "current_recency": "carry_over",
                        "checkpoint_state": "carry_over",
                    },
                    {
                        "rank": 3,
                        "name": "refresh_hot_universe_research_embedding",
                        "script": expected_refresh_research_script,
                        "command_hint": expected_refresh_research_command,
                        "expected_status": "ok",
                        "expected_artifact_kind": expected_refresh_research_artifact_kind,
                        "expected_artifact_path_hint": expected_refresh_research_artifact_path_hint,
                        "current_artifact": str(review_dir / "20260310T120000Z_hot_universe_research.json"),
                        "current_status": "ok",
                        "current_as_of": "2026-03-10T12:00:00+00:00",
                        "current_age_minutes": 30,
                        "current_recency": "carry_over",
                        "checkpoint_state": "carry_over",
                    },
                    {
                        "rank": 4,
                        "name": "refresh_commodity_handoff",
                        "script": expected_refresh_followup_script,
                        "command_hint": expected_refresh_followup_command,
                        "expected_status": "ok",
                        "expected_artifact_kind": expected_refresh_followup_artifact_kind,
                        "expected_artifact_path_hint": expected_refresh_followup_artifact_path_hint,
                        "current_artifact": "",
                        "current_status": "",
                        "current_as_of": "",
                        "current_age_minutes": None,
                        "current_recency": "unknown",
                        "checkpoint_state": "missing",
                    },
                ],
            }
        ]
    assert [row["slot"] for row in payload["operator_focus_slots"]] == ["primary", "followup", "secondary"]
    assert payload["operator_focus_slots"][0]["area"] == "commodity_execution_close_evidence"
    assert payload["operator_focus_slots"][0]["symbol"] == "XAUUSD"
    assert payload["operator_focus_slots"][0]["action"] == "wait_for_paper_execution_close_evidence"
    assert payload["operator_focus_slots"][0]["source_kind"] == "commodity_execution_retro"
    assert payload["operator_focus_slots"][1]["area"] == "commodity_fill_evidence"
    assert payload["operator_focus_slots"][1]["symbol"] == "XAGUSD"
    assert payload["operator_focus_slots"][1]["action"] == "wait_for_paper_execution_fill_evidence"
    assert payload["operator_focus_slots"][1]["source_kind"] == "commodity_execution_review"
    assert payload["operator_focus_slots"][2]["area"] == "crypto_route"
    assert payload["operator_focus_slots"][2]["symbol"] == "SOLUSDT"
    assert payload["operator_focus_slots"][2]["action"] == "deprioritize_flow"
    assert payload["operator_focus_slots"][2]["priority_tier"] == "review_queue_now"
    assert payload["operator_focus_slots"][2]["priority_score"] == 73
    assert payload["operator_focus_slots"][2]["queue_rank"] == 1
    assert payload["operator_focus_slots"][2]["source_kind"] == "crypto_route"
    assert payload["secondary_focus_area"] == "crypto_route"
    assert payload["secondary_focus_target"] == "SOLUSDT"
    assert payload["secondary_focus_symbol"] == "SOLUSDT"
    assert payload["secondary_focus_action"] == "deprioritize_flow"
    assert payload["secondary_focus_reason"] == "Flow does not produce a positive ranked edge even in the short sample."
    assert payload["secondary_focus_state"] == "review"
    assert payload["secondary_focus_blocker_detail"] == (
        "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade."
    )
    assert payload["secondary_focus_done_when"] == (
        "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow"
    )
    assert payload["secondary_focus_priority_tier"] == "review_queue_now"
    assert payload["secondary_focus_priority_score"] == 73
    assert payload["secondary_focus_queue_rank"] == 1
    assert payload["operator_action_queue_brief"] == (
        "1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence"
        " | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence"
        " | 3:crypto_route:SOLUSDT:deprioritize_flow"
    )
    assert payload["operator_action_checklist_brief"] == (
        "1:waiting:XAUUSD:wait_for_paper_execution_close_evidence"
        " | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence"
        " | 3:review:SOLUSDT:deprioritize_flow"
    )
    assert "secondary-focus-priority: review_queue_now | score=73 | queue_rank=1" in payload["summary_text"]
    assert "crypto-head-source-refresh: refresh_recommended:SOLUSDT:consider_refresh_before_promotion" in payload["summary_text"]
    assert "focus-slot-refresh-backlog: secondary:SOLUSDT:consider_refresh_before_promotion" in payload["summary_text"]
    assert "focus-slot-promotion-gate: promotion_guarded_by_source_freshness:2/3" in payload["summary_text"]
    assert "focus-slot-readiness-gate: readiness_guarded_by_source_freshness:2/3" in payload["summary_text"]
    assert "source-refresh-queue: 1:secondary:SOLUSDT:consider_refresh_before_promotion" in payload["summary_text"]
    assert "source-refresh-checklist: 1:refresh_recommended:SOLUSDT:consider_refresh_before_promotion" in payload["summary_text"]
    assert payload["operator_action_queue"] == [
        {
            "rank": 1,
            "area": "commodity_execution_close_evidence",
            "target": "commodity-paper-execution:metals_all:XAUUSD",
            "symbol": "XAUUSD",
            "action": "wait_for_paper_execution_close_evidence",
            "reason": "paper_execution_close_evidence_pending",
        },
        {
            "rank": 2,
            "area": "commodity_fill_evidence",
            "target": "commodity-paper-execution:metals_all:XAGUSD",
            "symbol": "XAGUSD",
            "action": "wait_for_paper_execution_fill_evidence",
            "reason": "paper_execution_fill_evidence_pending",
        },
        {
            "rank": 3,
            "area": "crypto_route",
            "target": "SOLUSDT",
            "symbol": "SOLUSDT",
            "action": "deprioritize_flow",
            "reason": "Flow does not produce a positive ranked edge even in the short sample.",
        },
    ]
    assert payload["operator_action_checklist"] == [
        {
            "rank": 1,
            "area": "commodity_execution_close_evidence",
            "target": "commodity-paper-execution:metals_all:XAUUSD",
            "symbol": "XAUUSD",
            "action": "wait_for_paper_execution_close_evidence",
            "reason": "paper_execution_close_evidence_pending",
            "state": "waiting",
            "blocker_detail": "paper execution evidence is present, but position is still OPEN; waiting for close evidence",
            "done_when": "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available",
        },
        {
            "rank": 2,
            "area": "commodity_fill_evidence",
            "target": "commodity-paper-execution:metals_all:XAGUSD",
            "symbol": "XAGUSD",
            "action": "wait_for_paper_execution_fill_evidence",
            "reason": "paper_execution_fill_evidence_pending",
            "state": "waiting",
            "blocker_detail": "paper execution fill evidence not written; stale directional signal 42d since 2026-01-26",
            "done_when": "XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols",
        },
        {
            "rank": 3,
            "area": "crypto_route",
            "target": "SOLUSDT",
            "symbol": "SOLUSDT",
            "action": "deprioritize_flow",
            "reason": "Flow does not produce a positive ranked edge even in the short sample.",
            "state": "review",
            "blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade.",
            "done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
        },
    ]
    assert payload["commodity_remainder_focus_area"] == "commodity_fill_evidence"
    assert payload["commodity_remainder_focus_target"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["commodity_remainder_focus_symbol"] == "XAGUSD"
    assert payload["commodity_remainder_focus_action"] == "wait_for_paper_execution_fill_evidence"
    assert payload["commodity_remainder_focus_reason"] == "paper_execution_fill_evidence_pending"
    assert payload["commodity_remainder_focus_signal_date"] == "2026-01-26"
    assert payload["commodity_remainder_focus_signal_age_days"] == 42
    assert payload["commodity_next_fill_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["commodity_next_fill_evidence_execution_symbol"] == "XAGUSD"
    assert payload["commodity_fill_evidence_pending_count"] == 2
    assert payload["commodity_execution_bridge_stale_signal_dates"] == {
        "COPPER": "2026-01-29",
        "XAGUSD": "2026-01-26",
    }
    assert payload["commodity_execution_bridge_stale_signal_age_days"] == {
        "COPPER": 39,
        "XAGUSD": 42,
    }
    assert payload["commodity_stale_signal_watch_brief"] == "XAGUSD:42d@2026-01-26, COPPER:39d@2026-01-29"
    assert payload["commodity_stale_signal_watch_next_execution_id"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["commodity_stale_signal_watch_next_symbol"] == "XAGUSD"
    assert payload["commodity_stale_signal_watch_next_signal_date"] == "2026-01-26"
    assert payload["commodity_stale_signal_watch_next_signal_age_days"] == 42
    assert payload["commodity_focus_evidence_item_source"] == "retro"
    assert payload["commodity_focus_evidence_summary"]["paper_entry_price"] == 5198.10009765625
    assert payload["commodity_focus_evidence_summary"]["paper_stop_price"] == 4847.7998046875
    assert payload["commodity_focus_evidence_summary"]["paper_target_price"] == 5758.58056640625
    assert payload["commodity_focus_evidence_summary"]["paper_signal_price_reference_source"] == "yfinance:GC=F"
    assert payload["commodity_focus_lifecycle_status"] == "open_position_wait_close_evidence"
    assert payload["commodity_focus_lifecycle_brief"] == "open_position_wait_close_evidence:XAUUSD"
    assert payload["commodity_focus_lifecycle_blocker_detail"] == (
        "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
    )
    assert payload["commodity_focus_lifecycle_done_when"] == (
        "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
    )
    assert payload["commodity_execution_close_evidence_status"] == "close_evidence_pending"
    assert payload["commodity_execution_close_evidence_brief"] == "close_evidence_pending:XAUUSD"
    assert payload["commodity_execution_close_evidence_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_execution_close_evidence_symbol"] == "XAUUSD"
    assert payload["commodity_execution_close_evidence_blocker_detail"] == (
        "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
    )
    assert payload["commodity_execution_close_evidence_done_when"] == (
        "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
    )
    assert payload["commodity_review_pending_symbols"] == []
    assert payload["commodity_review_close_evidence_pending_count"] == 1
    assert payload["commodity_review_close_evidence_pending_symbols"] == ["XAUUSD"]
    assert payload["commodity_next_review_close_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_review_close_evidence_execution_symbol"] == "XAUUSD"
    assert payload["commodity_retro_pending_symbols"] == []
    assert payload["commodity_close_evidence_pending_count"] == 1
    assert payload["commodity_close_evidence_pending_symbols"] == ["XAUUSD"]
    assert payload["commodity_next_close_evidence_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_next_close_evidence_execution_symbol"] == "XAUUSD"
    assert payload["commodity_review_fill_evidence_pending_symbols"] == ["XAGUSD", "COPPER"]
    assert payload["commodity_retro_fill_evidence_pending_symbols"] == ["XAGUSD", "COPPER"]
    assert "commodity-execution-review-status: paper-execution-close-evidence-pending-fill-remainder" in payload["summary_text"]
    assert "commodity-execution-retro-status: paper-execution-close-evidence-pending-fill-remainder" in payload["summary_text"]
    assert "commodity-review-pending-symbols: -" in payload["summary_text"]
    assert "commodity-review-close-evidence-pending-count: 1" in payload["summary_text"]
    assert "commodity-review-close-evidence-pending-symbols: XAUUSD" in payload["summary_text"]
    assert "commodity-retro-pending-symbols: -" in payload["summary_text"]
    assert "commodity-close-evidence-pending-symbols: XAUUSD" in payload["summary_text"]
    assert "commodity-fill-evidence-pending-symbols: XAGUSD, COPPER" in payload["summary_text"]
    assert "next-focus-state: waiting" in payload["summary_text"]
    assert (
        "next-focus-blocker: paper execution evidence is present, but position is still OPEN; waiting for close evidence"
        in payload["summary_text"]
    )
    assert (
        "next-focus-done-when: XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
        in payload["summary_text"]
    )
    assert "followup-focus: commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence" in payload["summary_text"]
    assert "followup-focus-state: waiting" in payload["summary_text"]
    assert "followup-focus-blocker: paper execution fill evidence not written; stale directional signal 42d since 2026-01-26" in payload["summary_text"]
    assert "followup-focus-done-when: XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols" in payload["summary_text"]
    assert "secondary-focus: crypto_route:SOLUSDT:deprioritize_flow" in payload["summary_text"]
    assert "secondary-focus-state: review" in payload["summary_text"]
    assert "secondary-focus-blocker: SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade." in payload["summary_text"]
    assert "secondary-focus-done-when: SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow" in payload["summary_text"]
    assert "focus-slots: primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence | followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | secondary:review:SOLUSDT:deprioritize_flow" in payload["summary_text"]
    assert "focus-slot-sources: primary:commodity_execution_retro | followup:commodity_execution_review | secondary:crypto_route" in payload["summary_text"]
    assert "focus-slot-source-status: primary:ok@2026-03-10T12:27:55+00:00 | followup:ok@2026-03-10T12:27:51+00:00 | secondary:ok@2026-03-10T12:00:00+00:00" in payload["summary_text"]
    assert "focus-slot-source-recency: primary:fresh:2m | followup:fresh:2m | secondary:carry_over:30m" in payload["summary_text"]
    assert "focus-slot-source-health: primary:ready:read_current_artifact | followup:ready:read_current_artifact | secondary:carry_over_ok:consider_refresh_before_promotion" in payload["summary_text"]
    assert "action-queue: 1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence | 3:crypto_route:SOLUSDT:deprioritize_flow" in payload["summary_text"]
    assert "action-checklist: 1:waiting:XAUUSD:wait_for_paper_execution_close_evidence | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | 3:review:SOLUSDT:deprioritize_flow" in payload["summary_text"]
    assert "commodity-execution-bridge-stale-signal-dates: COPPER:2026-01-29, XAGUSD:2026-01-26" in payload["summary_text"]
    assert "commodity-execution-bridge-stale-signal-age-days: COPPER:39, XAGUSD:42" in payload["summary_text"]
    assert "commodity-stale-signal-watch: XAGUSD:42d@2026-01-26, COPPER:39d@2026-01-29" in payload["summary_text"]
    assert "commodity-stale-signal-watch-next-id: commodity-paper-execution:metals_all:XAGUSD" in payload["summary_text"]
    assert "commodity-stale-signal-watch-next-symbol: XAGUSD" in payload["summary_text"]
    assert "commodity-stale-signal-watch-next-signal-date: 2026-01-26" in payload["summary_text"]
    assert "commodity-stale-signal-watch-next-signal-age-days: 42" in payload["summary_text"]
    assert "commodity-remainder-focus-signal-date: 2026-01-26" in payload["summary_text"]
    assert "commodity-remainder-focus-signal-age-days: 42" in payload["summary_text"]
    assert "commodity-focus-paper-evidence: source=retro entry=5198.100098 stop=4847.799805 target=5758.580566 quote=0.158961 status=OPEN ref=yfinance:GC=F" in payload["summary_text"]
    assert "commodity-focus-lifecycle: open_position_wait_close_evidence:XAUUSD" in payload["summary_text"]
    assert "commodity-close-evidence: close_evidence_pending:XAUUSD" in payload["summary_text"]
    assert (
        "commodity-remainder-focus: commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence"
        in payload["summary_text"]
    )


def test_build_hot_universe_operator_brief_falls_back_to_queue_when_fill_evidence_missing(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all", "precious_metals"],
                "focus_with_regime_filter_batches": ["energy_liquids"],
                "research_queue_batches": ["crypto_majors", "crypto_hot"],
                "shadow_only_batches": ["commodities_benchmark"],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "execution_item_count": 3,
            "actionable_execution_item_count": 3,
            "queue_depth": 3,
            "actionable_queue_depth": 3,
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
            "queued_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "queue_rank": 1,
                    "execution_status": "queued",
                    "source_execution_status": "planned",
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_review_status": "paper-execution-awaiting-fill-evidence",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_regime_gate": "paper_only",
            "review_item_count": 3,
            "actionable_review_item_count": 0,
            "fill_evidence_pending_count": 3,
            "next_review_execution_id": "",
            "next_review_execution_symbol": "",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_fill_evidence_execution_symbol": "XAUUSD",
            "review_stack_brief": "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER",
            "review_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "review_status": "awaiting_paper_execution_fill",
                    "paper_execution_evidence_present": False,
                }
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "route_status": "paper-first",
            "ticket_book_status": "paper-ready",
            "execution_preview_status": "paper-execution-ready",
            "execution_artifact_status": "paper-execution-artifact-ready",
            "execution_queue_status": "paper-execution-queued",
            "execution_review_status": "paper-execution-awaiting-fill-evidence",
            "execution_retro_status": "paper-execution-awaiting-fill-evidence",
            "execution_mode": "paper_only",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "execution_ticket_ids": [
                "commodity-paper-ticket:metals_all:XAUUSD",
                "commodity-paper-ticket:metals_all:XAGUSD",
                "commodity-paper-ticket:metals_all:COPPER",
            ],
            "execution_regime_gate": "paper_only",
            "execution_weight_hint_sum": 2.3,
            "retro_item_count": 3,
            "actionable_retro_item_count": 0,
            "fill_evidence_pending_count": 3,
            "next_retro_execution_id": "",
            "next_retro_execution_symbol": "",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_fill_evidence_execution_symbol": "XAUUSD",
            "retro_stack_brief": "paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER",
            "retro_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "retro_status": "awaiting_paper_execution_fill",
                    "review_status": "awaiting_paper_execution_fill",
                    "paper_execution_evidence_present": False,
                }
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_review_status"] == "paper-execution-awaiting-fill-evidence"
    assert payload["commodity_execution_retro_status"] == "paper-execution-awaiting-fill-evidence"
    assert payload["commodity_next_review_execution_id"] == ""
    assert payload["commodity_next_retro_execution_id"] == ""
    assert payload["operator_status"] == "commodity-paper-execution-queued-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:queue:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_queue"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "inspect_paper_execution_queue"
    assert payload["next_focus_reason"] == "paper_execution_queued"
    assert "commodity-execution-review-status: paper-execution-awaiting-fill-evidence" in payload["summary_text"]
    assert "commodity-execution-retro-status: paper-execution-awaiting-fill-evidence" in payload["summary_text"]


def test_build_hot_universe_operator_brief_surfaces_commodity_gap_report(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-awaiting-fill-evidence",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_fill_evidence_execution_symbol": "XAUUSD",
            "review_stack_brief": "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "execution_retro_status": "paper-execution-awaiting-fill-evidence",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_fill_evidence_execution_symbol": "XAUUSD",
            "retro_stack_brief": "paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": [
                "queue_symbols_missing_from_core_universe",
                "core_universe_crypto_only",
                "queue_symbols_missing_from_trade_plans",
            ],
            "root_cause_lines": [
                "Queue symbols are absent from config core universe: XAUUSD, XAGUSD, COPPER.",
                "Config core universe remains crypto-only.",
            ],
            "recommended_actions": [
                "Keep commodity queue in research/paper-planning mode until a real paper execution bridge exists."
            ],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_commodity_execution_gap_status"] == "ok"
    assert payload["commodity_execution_gap_status"] == "blocking_gap_active"
    assert payload["commodity_execution_gap_decision"] == "do_not_assume_commodity_paper_execution_active"
    assert payload["commodity_execution_gap_reason_codes"] == [
        "queue_symbols_missing_from_core_universe",
        "core_universe_crypto_only",
        "queue_symbols_missing_from_trade_plans",
    ]
    assert payload["commodity_execution_gap_root_cause_lines"] == [
        "Queue symbols are absent from config core universe: XAUUSD, XAGUSD, COPPER.",
        "Config core universe remains crypto-only.",
    ]
    assert payload["commodity_execution_gap_recommended_actions"] == [
        "Keep commodity queue in research/paper-planning mode until a real paper execution bridge exists."
    ]
    assert payload["commodity_execution_gap_batch"] == ""
    assert payload["commodity_execution_gap_next_execution_id"] == ""
    assert payload["commodity_execution_gap_next_execution_symbol"] == ""
    assert payload["commodity_gap_focus_batch"] == "metals_all"
    assert payload["commodity_gap_focus_execution_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_gap_focus_symbol"] == "XAUUSD"
    assert payload["operator_status"] == "commodity-paper-execution-gap-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:gap:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_gap"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "resolve_commodity_paper_execution_gap"
    assert payload["next_focus_reason"] == "commodity_execution_gap_active"
    assert "commodity-execution-gap-status: blocking_gap_active" in payload["summary_text"]
    assert "commodity-execution-gap-decision: do_not_assume_commodity_paper_execution_active" in payload["summary_text"]
    assert "commodity-gap-root-cause: Queue symbols are absent from config core universe: XAUUSD, XAGUSD, COPPER." in payload["summary_text"]


def test_build_hot_universe_operator_brief_prioritizes_bridge_blocker(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_missing_directional_signal"],
            "root_cause_lines": ["Fresh signal-to-order tickets still return signal_not_found."],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_missing_directional_signal",
            "signal_missing_count": 3,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_blocked_symbol": "XAUUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_commodity_execution_bridge_status"] == "ok"
    assert payload["commodity_execution_bridge_status"] == "blocked_missing_directional_signal"
    assert payload["commodity_execution_bridge_next_blocked_id"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["commodity_execution_bridge_next_blocked_symbol"] == "XAUUSD"
    assert payload["commodity_execution_bridge_signal_missing_count"] == 3
    assert payload["commodity_execution_bridge_signal_stale_count"] == 0
    assert payload["operator_status"] == "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:bridge-blocked:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_bridge"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "restore_commodity_directional_signal"
    assert payload["next_focus_reason"] == "commodity_bridge_blocked_missing_directional_signal"
    assert "commodity-execution-bridge-status: blocked_missing_directional_signal" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prioritizes_stale_bridge_blocker(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent; the latest bridgeable commodity signals are stale."],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_stale_directional_signal",
            "signal_missing_count": 0,
            "signal_stale_count": 3,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_blocked_symbol": "XAUUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_bridge_status"] == "blocked_stale_directional_signal"
    assert payload["commodity_execution_bridge_signal_missing_count"] == 0
    assert payload["commodity_execution_bridge_signal_stale_count"] == 3
    assert payload["operator_status"] == "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:bridge-stale:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_bridge"
    assert payload["next_focus_action"] == "restore_commodity_directional_signal"
    assert payload["next_focus_reason"] == "commodity_bridge_blocked_stale_directional_signal"
    assert "commodity-execution-bridge-status: blocked_stale_directional_signal" in payload["summary_text"]
    assert "commodity-execution-bridge-signal-stale-count: 3" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prioritizes_partial_stale_bridge_blocker(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent for XAGUSD."],
            "queue_symbols_with_stale_directional_signal_dates": {"XAGUSD": "2026-01-26"},
            "queue_symbols_with_stale_directional_signal_age_days": {"XAGUSD": 42},
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "bridge_partially_bridged_stale_remainder",
            "already_present_count": 1,
            "already_bridged_symbols": ["XAUUSD"],
            "signal_missing_count": 0,
            "signal_stale_count": 1,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_blocked_symbol": "XAGUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_bridge_status"] == "bridge_partially_bridged_stale_remainder"
    assert payload["commodity_execution_bridge_already_present_count"] == 1
    assert payload["commodity_execution_bridge_already_bridged_symbols"] == ["XAUUSD"]
    assert payload["operator_status"] == "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:bridge-stale:XAGUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_bridge"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAGUSD"
    assert payload["next_focus_symbol"] == "XAGUSD"
    assert payload["next_focus_action"] == "restore_commodity_directional_signal"
    assert payload["next_focus_reason"] == "commodity_bridge_blocked_stale_directional_signal"
    assert payload["commodity_execution_bridge_stale_signal_dates"] == {"XAGUSD": "2026-01-26"}
    assert payload["commodity_execution_bridge_stale_signal_age_days"] == {"XAGUSD": 42}
    assert payload["commodity_remainder_focus_signal_date"] == "2026-01-26"
    assert payload["commodity_remainder_focus_signal_age_days"] == 42
    assert "commodity-execution-bridge-already-present-count: 1" in payload["summary_text"]
    assert "commodity-execution-bridge-already-bridged-symbols: XAUUSD" in payload["summary_text"]
    assert "commodity-execution-bridge-stale-signal-dates: XAGUSD:2026-01-26" in payload["summary_text"]
    assert "commodity-execution-bridge-stale-signal-age-days: XAGUSD:42" in payload["summary_text"]
    assert "commodity-remainder-focus-signal-date: 2026-01-26" in payload["summary_text"]
    assert "commodity-remainder-focus-signal-age-days: 42" in payload["summary_text"]


def test_build_hot_universe_operator_brief_prioritizes_proxy_price_bridge_blocker(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_proxy_price_reference_only", "queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": [
                "Commodity directional tickets still use proxy-market prices rather than executable instrument prices."
            ],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "blocked_proxy_price_reference_only",
            "signal_missing_count": 0,
            "signal_stale_count": 3,
            "signal_proxy_price_only_count": 3,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_blocked_symbol": "XAUUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_bridge_status"] == "blocked_proxy_price_reference_only"
    assert payload["commodity_execution_bridge_signal_missing_count"] == 0
    assert payload["commodity_execution_bridge_signal_stale_count"] == 3
    assert payload["commodity_execution_bridge_signal_proxy_price_only_count"] == 3
    assert payload["operator_status"] == "commodity-paper-execution-bridge-blocked-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:bridge-proxy:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_bridge"
    assert payload["next_focus_action"] == "normalize_commodity_execution_price_reference"
    assert payload["next_focus_reason"] == "commodity_bridge_blocked_proxy_price_reference_only"
    assert "commodity-execution-bridge-status: blocked_proxy_price_reference_only" in payload["summary_text"]
    assert "commodity-execution-bridge-signal-proxy-price-only-count: 3" in payload["summary_text"]


def test_build_hot_universe_operator_brief_keeps_review_focus_when_bridge_only_partially_blocked(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    _write_json(
        review_dir / "20260310T122750Z_commodity_paper_execution_queue.json",
        {
            "status": "ok",
            "execution_queue_status": "paper-execution-queued",
            "execution_batch": "metals_all",
            "execution_symbols": ["XAUUSD", "XAGUSD", "COPPER"],
            "next_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_execution_symbol": "XAUUSD",
            "queue_stack_brief": "paper-execution-queue:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122751Z_commodity_paper_execution_review.json",
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending",
            "actionable_review_item_count": 1,
            "next_review_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_review_execution_symbol": "XAUUSD",
            "review_stack_brief": "paper-execution-review:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122755Z_commodity_paper_execution_retro.json",
        {
            "status": "ok",
            "execution_retro_status": "paper-execution-awaiting-fill-evidence",
            "actionable_retro_item_count": 0,
            "retro_stack_brief": "paper-execution-retro:metals_all:XAUUSD, XAGUSD, COPPER",
        },
    )
    _write_json(
        review_dir / "20260310T122800Z_commodity_paper_execution_gap_report.json",
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent for XAGUSD and COPPER."],
        },
    )
    _write_json(
        review_dir / "20260310T122810Z_commodity_paper_execution_bridge.json",
        {
            "status": "ok",
            "bridge_status": "bridge_partially_bridged_stale_remainder",
            "already_present_count": 1,
            "already_bridged_symbols": ["XAUUSD"],
            "signal_missing_count": 0,
            "signal_stale_count": 2,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_blocked_symbol": "XAGUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["commodity_execution_bridge_status"] == "bridge_partially_bridged_stale_remainder"
    assert payload["commodity_execution_bridge_already_present_count"] == 1
    assert payload["commodity_execution_bridge_already_bridged_symbols"] == ["XAUUSD"]
    assert payload["commodity_execution_review_status"] == "paper-execution-review-pending"
    assert payload["operator_status"] == "commodity-paper-execution-review-pending-plus-crypto-deploy-watch"
    assert payload["operator_stack_brief"] == "commodity:review:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms"
    assert payload["next_focus_area"] == "commodity_execution_review"
    assert payload["next_focus_target"] == "commodity-paper-execution:metals_all:XAUUSD"
    assert payload["next_focus_symbol"] == "XAUUSD"
    assert payload["next_focus_action"] == "review_paper_execution"
    assert payload["next_focus_reason"] == "paper_execution_review_pending"
    assert "commodity-execution-bridge-status: bridge_partially_bridged_stale_remainder" in payload["summary_text"]


def test_build_hot_universe_operator_brief_uses_explicit_execution_artifacts(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    _write_json(
        review_dir / "20260310T120000Z_hot_universe_research.json",
        {
            "status": "ok",
            "research_action_ladder": {
                "focus_primary_batches": ["metals_all"],
                "focus_with_regime_filter_batches": [],
                "research_queue_batches": ["crypto_hot"],
                "shadow_only_batches": [],
                "avoid_batches": [],
            },
            "crypto_route_operator_brief": {
                "operator_status": "deploy-price-state-plus-beta-watch",
                "route_stack_brief": "deploy:BTCUSDT,ETHUSDT | watch-priority:BNBUSDT | watch:SOLUSDT",
                "next_focus_symbol": "BNBUSDT",
                "next_focus_action": "watch_priority_until_long_window_confirms",
            },
        },
    )
    queue_path = review_dir / "20260311T084601Z_commodity_paper_execution_queue.json"
    review_path = review_dir / "20260311T084602Z_commodity_paper_execution_review.json"
    retro_path = review_dir / "20260311T084603Z_commodity_paper_execution_retro.json"
    gap_path = review_dir / "20260311T084604Z_commodity_paper_execution_gap_report.json"
    bridge_path = review_dir / "20260311T084600Z_commodity_paper_execution_bridge.json"
    _write_json(queue_path, {"status": "ok", "execution_queue_status": "paper-execution-queued", "execution_symbols": ["COPPER"]})
    _write_json(
        review_path,
        {
            "status": "ok",
            "execution_review_status": "paper-execution-review-pending",
            "actionable_review_item_count": 1,
            "next_review_execution_id": "commodity-paper-execution:metals_all:COPPER",
            "next_review_execution_symbol": "COPPER",
        },
    )
    _write_json(
        retro_path,
        {
            "status": "ok",
            "execution_retro_status": "paper-execution-retro-pending",
            "actionable_retro_item_count": 1,
            "next_retro_execution_id": "commodity-paper-execution:metals_all:COPPER",
            "next_retro_execution_symbol": "COPPER",
        },
    )
    _write_json(
        gap_path,
        {
            "status": "ok",
            "gap_status": "blocking_gap_active",
            "current_decision": "do_not_assume_commodity_paper_execution_active",
            "gap_reason_codes": ["queue_symbols_with_stale_directional_signal"],
            "root_cause_lines": ["Fresh directional combo triggers are absent for XAGUSD."],
        },
    )
    _write_json(
        bridge_path,
        {
            "status": "ok",
            "bridge_status": "blocked_stale_directional_signal",
            "signal_stale_count": 1,
            "next_blocked_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_blocked_symbol": "XAGUSD",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--commodity-execution-queue-json",
            str(queue_path),
            "--commodity-execution-review-json",
            str(review_path),
            "--commodity-execution-retro-json",
            str(retro_path),
            "--commodity-execution-gap-json",
            str(gap_path),
            "--commodity-execution-bridge-json",
            str(bridge_path),
            "--now",
            "2026-03-10T12:30:00Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["source_commodity_execution_queue_artifact"] == str(queue_path.resolve())
    assert payload["source_commodity_execution_review_artifact"] == str(review_path.resolve())
    assert payload["source_commodity_execution_retro_artifact"] == str(retro_path.resolve())
    assert payload["source_commodity_execution_gap_artifact"] == str(gap_path.resolve())
    assert payload["source_commodity_execution_bridge_artifact"] == str(bridge_path.resolve())
    assert payload["commodity_next_retro_execution_symbol"] == "COPPER"
