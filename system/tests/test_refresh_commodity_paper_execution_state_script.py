from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/refresh_commodity_paper_execution_state.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("commodity_refresh_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_now_is_monotonic() -> None:
    mod = _load_module()
    base = mod.parse_now("2026-03-11T08:40:00Z")
    assert mod.step_now(base, 0).isoformat() == "2026-03-11T08:40:00+00:00"
    assert mod.step_now(base, 3).isoformat() == "2026-03-11T08:40:03+00:00"
    assert mod.step_now(base, 5) > mod.step_now(base, 4)


def test_render_context_markdown_mentions_apply_and_stale_symbols() -> None:
    mod = _load_module()
    text = mod.render_context_markdown(
        runtime_now=mod.parse_now("2026-03-11T08:40:10Z"),
        brief={
            "artifact": "/tmp/brief.json",
            "operator_status": "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch",
            "operator_stack_brief": "commodity:close-evidence:XAUUSD | crypto:BNBUSDT:watch_priority_until_long_window_confirms",
            "next_focus_area": "commodity_execution_close_evidence",
            "next_focus_target": "commodity-paper-execution:metals_all:XAUUSD",
            "next_focus_action": "wait_for_paper_execution_close_evidence",
            "next_focus_reason": "paper_execution_close_evidence_pending",
            "next_focus_state": "waiting",
            "next_focus_blocker_detail": (
                "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
            ),
            "next_focus_done_when": (
                "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
            ),
            "followup_focus_area": "commodity_fill_evidence",
            "followup_focus_target": "commodity-paper-execution:metals_all:XAGUSD",
            "followup_focus_action": "wait_for_paper_execution_fill_evidence",
            "followup_focus_state": "waiting",
            "followup_focus_blocker_detail": "paper execution fill evidence not written; stale directional signal 42d since 2026-01-26",
            "followup_focus_done_when": "XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols",
            "operator_focus_slots_brief": (
                "primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence"
                " | followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence"
                " | secondary:watch:BNBUSDT:watch_priority_until_long_window_confirms"
            ),
            "operator_focus_slot_sources_brief": (
                "primary:commodity_execution_retro | followup:commodity_execution_review | secondary:crypto_route"
            ),
            "operator_focus_slot_status_brief": (
                "primary:ok@2026-03-11T08:40:08+00:00"
                " | followup:ok@2026-03-11T08:40:07+00:00"
                " | secondary:ok@2026-03-11T08:40:00+00:00"
            ),
            "operator_focus_slot_recency_brief": (
                "primary:fresh:0m | followup:fresh:0m | secondary:fresh:0m"
            ),
            "operator_focus_slot_health_brief": (
                "primary:ready:read_current_artifact"
                " | followup:ready:read_current_artifact"
                " | secondary:ready:read_current_artifact"
            ),
            "operator_focus_slot_refresh_backlog_brief": "-",
            "operator_focus_slot_refresh_backlog_count": 0,
            "operator_focus_slot_refresh_backlog": [],
            "operator_focus_slot_ready_count": 3,
            "operator_focus_slot_total_count": 3,
            "operator_focus_slot_promotion_gate_brief": "promotion_ready:3/3",
            "operator_focus_slot_promotion_gate_status": "promotion_ready",
            "operator_focus_slot_promotion_gate_blocker_detail": "all 3 focus slots have ready source artifacts",
            "operator_focus_slot_promotion_gate_done_when": "all focus slots continue using read_current_artifact sources",
            "operator_focus_slot_actionability_backlog_brief": "secondary:BNBUSDT:recovery_completed_no_edge",
            "operator_focus_slot_actionability_backlog_count": 1,
            "operator_focus_slot_actionability_backlog": [
                {
                    "slot": "secondary",
                    "symbol": "BNBUSDT",
                    "action": "watch_priority_until_long_window_confirms",
                    "state": "watch",
                    "blocker_detail": "long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta",
                    "done_when": "BNBUSDT gains supporting crypto research edge or leaves priority watch",
                    "alignment_status": "route_ahead_of_embedding",
                    "alignment_brief": "route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta",
                    "alignment_recovery_status": "recovery_completed_no_edge",
                    "alignment_recovery_brief": "recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta",
                }
            ],
            "operator_focus_slot_actionable_count": 2,
            "operator_focus_slot_actionability_gate_brief": "actionability_guarded_by_content:2/3",
            "operator_focus_slot_actionability_gate_status": "actionability_guarded_by_content",
            "operator_focus_slot_actionability_gate_blocker_detail": (
                "BNBUSDT secondary content state remains blocked "
                "(route_ahead_of_embedding, recovery_completed_no_edge): "
                "long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta"
            ),
            "operator_focus_slot_actionability_gate_done_when": (
                "BNBUSDT gains supporting crypto research edge or leaves priority watch"
            ),
            "operator_focus_slot_readiness_gate_ready_count": 2,
            "operator_focus_slot_readiness_gate_brief": "readiness_guarded_by_content:2/3",
            "operator_focus_slot_readiness_gate_status": "readiness_guarded_by_content",
            "operator_focus_slot_readiness_gate_blocking_gate": "content_actionability",
            "operator_focus_slot_readiness_gate_blocker_detail": (
                "BNBUSDT secondary content state remains blocked "
                "(route_ahead_of_embedding, recovery_completed_no_edge): "
                "long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta"
            ),
            "operator_focus_slot_readiness_gate_done_when": (
                "BNBUSDT gains supporting crypto research edge or leaves priority watch"
            ),
            "operator_research_embedding_quality_status": "avoid_only",
            "operator_research_embedding_quality_brief": "avoid_only:crypto_hot, crypto_majors, crypto_beta",
            "operator_research_embedding_quality_blocker_detail": (
                "latest hot_universe_research is fresh and ok, but all tracked batches are avoid/inactive: "
                "crypto_hot, crypto_majors, crypto_beta | zero_trades=crypto_hot, crypto_majors, crypto_beta"
            ),
            "operator_research_embedding_quality_done_when": (
                "latest hot_universe_research promotes at least one focus_primary or research_queue batch"
            ),
            "operator_research_embedding_active_batches": [],
            "operator_research_embedding_avoid_batches": ["crypto_hot", "crypto_majors", "crypto_beta"],
            "operator_research_embedding_zero_trade_deprioritized_batches": [
                "crypto_hot",
                "crypto_majors",
                "crypto_beta",
            ],
            "operator_crypto_route_alignment_focus_slot": "secondary",
            "operator_crypto_route_alignment_status": "route_ahead_of_embedding",
            "operator_crypto_route_alignment_brief": (
                "route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta"
            ),
            "operator_crypto_route_alignment_blocker_detail": (
                "dedicated crypto_route still points to BNBUSDT:watch_priority_until_long_window_confirms, "
                "but latest hot_universe_research has no active crypto batches (avoid_only:crypto_hot, crypto_majors, crypto_beta)."
            ),
            "operator_crypto_route_alignment_done_when": (
                "hot_universe_research promotes at least one crypto batch or crypto_route focus degrades to match embedding"
            ),
            "operator_crypto_route_alignment_recovery_status": "recovery_completed_no_edge",
            "operator_crypto_route_alignment_recovery_brief": (
                "recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta"
            ),
            "operator_crypto_route_alignment_recovery_blocker_detail": (
                "latest crypto alignment recovery artifact finished cleanly, but all targeted crypto batches still "
                "show zero research trades and zero accepted strategy_lab candidates: crypto_hot, crypto_majors, crypto_beta"
            ),
            "operator_crypto_route_alignment_recovery_done_when": (
                "hot_universe_research promotes at least one crypto batch or dedicated crypto_route degrades to match the no-edge embedding"
            ),
            "operator_crypto_route_alignment_recovery_failed_batch_count": 0,
            "operator_crypto_route_alignment_recovery_timed_out_batch_count": 0,
            "operator_crypto_route_alignment_recovery_zero_trade_batches": [
                "crypto_hot",
                "crypto_majors",
                "crypto_beta",
            ],
            "operator_crypto_route_alignment_cooldown_status": "cooldown_active_wait_for_new_market_data",
            "operator_crypto_route_alignment_cooldown_brief": (
                "cooldown_active_wait_for_new_market_data:>2026-03-11"
            ),
            "operator_crypto_route_alignment_cooldown_blocker_detail": (
                "latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; "
                "rerunning before a later end date is unlikely to change the outcome"
            ),
            "operator_crypto_route_alignment_cooldown_done_when": (
                "hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes"
            ),
            "operator_crypto_route_alignment_cooldown_last_research_end_date": "2026-03-11",
            "operator_crypto_route_alignment_cooldown_next_eligible_end_date": "2026-03-12",
            "operator_crypto_route_alignment_recipe_status": "deferred_by_cooldown",
            "operator_crypto_route_alignment_recipe_brief": "deferred_by_cooldown:2026-03-12",
            "operator_crypto_route_alignment_recipe_blocker_detail": (
                "latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; "
                "rerunning before a later end date is unlikely to change the outcome"
            ),
            "operator_crypto_route_alignment_recipe_done_when": (
                "hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes"
            ),
            "operator_crypto_route_alignment_recipe_ready_on_date": "2026-03-12",
            "operator_crypto_route_alignment_recipe_script": "/tmp/run_hot_universe_research.py",
            "operator_crypto_route_alignment_recipe_command_hint": (
                "python3 /tmp/run_hot_universe_research.py --output-root /tmp/output --review-dir /tmp/review "
                "--start 2026-02-18 --end 2026-03-10 --now 2026-03-11T08:40:10Z "
                "--hours-budget 0.08 --max-trials-per-mode 5 --review-days 7 --run-strategy-lab "
                "--strategy-lab-candidate-count 6 --batch-timeout-seconds 30 --universe-file /tmp/universe.json "
                "--batch crypto_hot --batch crypto_majors --batch crypto_beta"
            ),
            "operator_crypto_route_alignment_recipe_expected_status": "ok",
            "operator_crypto_route_alignment_recipe_note": (
                "extend crypto embedding window to 21d and enable strategy_lab because current crypto batches are avoid_only with zero trades"
            ),
            "operator_crypto_route_alignment_recipe_followup_script": "/tmp/refresh_commodity_paper_execution_state.py",
            "operator_crypto_route_alignment_recipe_followup_command_hint": (
                "python3 /tmp/refresh_commodity_paper_execution_state.py --review-dir /tmp/review --output-root /tmp/output --context-path /tmp/review/NEXT_WINDOW_CONTEXT_LATEST.md"
            ),
            "operator_crypto_route_alignment_recipe_verify_hint": (
                "confirm operator_research_embedding_quality_status leaves avoid_only or operator_crypto_route_alignment_status leaves route_ahead_of_embedding"
            ),
            "operator_crypto_route_alignment_recipe_window_days": 21,
            "operator_crypto_route_alignment_recipe_target_batches": ["crypto_hot", "crypto_majors", "crypto_beta"],
            "operator_source_refresh_queue_brief": "-",
            "operator_source_refresh_queue_count": 0,
            "operator_source_refresh_queue": [],
            "operator_source_refresh_checklist_brief": "-",
            "operator_source_refresh_checklist": [],
            "operator_source_refresh_pipeline_steps_brief": "-",
            "operator_source_refresh_pipeline_step_checkpoint_brief": (
                "1:carry_over:crypto_route_brief | 2:carry_over:crypto_route_operator_brief | 3:carry_over:hot_universe_research"
            ),
            "operator_source_refresh_pipeline_pending_brief": "-",
            "operator_source_refresh_pipeline_pending_count": 0,
            "operator_source_refresh_pipeline_head_rank": "-",
            "operator_source_refresh_pipeline_head_name": "-",
            "operator_source_refresh_pipeline_head_checkpoint_state": "-",
            "operator_source_refresh_pipeline_head_expected_artifact_kind": "-",
            "operator_source_refresh_pipeline_head_current_artifact": "-",
            "operator_source_refresh_pipeline_deferred_brief": (
                "1:refresh_crypto_route_brief:carry_over:crypto_route_brief | "
                "2:refresh_crypto_route_operator_brief:carry_over:crypto_route_operator_brief | "
                "3:refresh_hot_universe_research_embedding:carry_over:hot_universe_research"
            ),
            "operator_source_refresh_pipeline_deferred_count": 3,
            "operator_source_refresh_pipeline_deferred_status": "deferred_by_cooldown",
            "operator_source_refresh_pipeline_deferred_until": "2026-03-12",
            "operator_source_refresh_pipeline_deferred_reason": (
                "latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; "
                "rerunning before a later end date is unlikely to change the outcome"
            ),
            "operator_source_refresh_pipeline_deferred_head_rank": "1",
            "operator_source_refresh_pipeline_deferred_head_name": "refresh_crypto_route_brief",
            "operator_source_refresh_pipeline_deferred_head_checkpoint_state": "carry_over",
            "operator_source_refresh_pipeline_deferred_head_expected_artifact_kind": "crypto_route_brief",
            "operator_source_refresh_pipeline_deferred_head_current_artifact": "/tmp/review/20260311T084000Z_crypto_route_brief.json",
            "next_focus_source_kind": "commodity_execution_retro",
            "next_focus_source_artifact": "/tmp/retro.json",
            "next_focus_source_status": "ok",
            "next_focus_source_as_of": "2026-03-11T08:40:08+00:00",
            "next_focus_source_age_minutes": 0,
            "next_focus_source_recency": "fresh",
            "next_focus_source_health": "ready",
            "next_focus_source_refresh_action": "read_current_artifact",
            "followup_focus_source_kind": "commodity_execution_review",
            "followup_focus_source_artifact": "/tmp/review.json",
            "followup_focus_source_status": "ok",
            "followup_focus_source_as_of": "2026-03-11T08:40:07+00:00",
            "followup_focus_source_age_minutes": 0,
            "followup_focus_source_recency": "fresh",
            "followup_focus_source_health": "ready",
            "followup_focus_source_refresh_action": "read_current_artifact",
            "secondary_focus_source_kind": "crypto_route",
            "secondary_focus_source_artifact": "/tmp/crypto.json",
            "secondary_focus_source_status": "ok",
            "secondary_focus_source_as_of": "2026-03-11T08:40:00+00:00",
            "secondary_focus_source_age_minutes": 0,
            "secondary_focus_source_recency": "fresh",
            "secondary_focus_source_health": "ready",
            "secondary_focus_source_refresh_action": "read_current_artifact",
            "operator_focus_slot_refresh_head_slot": "-",
            "operator_focus_slot_refresh_head_symbol": "-",
            "operator_focus_slot_refresh_head_action": "-",
            "operator_focus_slot_refresh_head_health": "-",
            "operator_source_refresh_next_slot": "-",
            "operator_source_refresh_next_symbol": "-",
            "operator_source_refresh_next_action": "-",
            "operator_source_refresh_next_source_kind": "-",
            "operator_source_refresh_next_source_health": "-",
            "operator_source_refresh_next_source_artifact": "-",
            "operator_source_refresh_next_state": "-",
            "operator_source_refresh_next_blocker_detail": "-",
            "operator_source_refresh_next_done_when": "-",
            "operator_source_refresh_next_recipe_script": "-",
            "operator_source_refresh_next_recipe_command_hint": "-",
            "operator_source_refresh_next_recipe_expected_status": "-",
            "operator_source_refresh_next_recipe_expected_artifact_kind": "-",
            "operator_source_refresh_next_recipe_expected_artifact_path_hint": "-",
            "operator_source_refresh_next_recipe_note": "-",
            "operator_source_refresh_next_recipe_followup_script": "-",
            "operator_source_refresh_next_recipe_followup_command_hint": "-",
            "operator_source_refresh_next_recipe_verify_hint": "-",
            "operator_source_refresh_next_recipe_steps_brief": "-",
            "operator_source_refresh_next_recipe_step_checkpoint_brief": "-",
            "operator_source_refresh_next_recipe_steps": [],
            "operator_focus_slots": [
                {
                    "slot": "primary",
                    "area": "commodity_execution_close_evidence",
                    "target": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                    "action": "wait_for_paper_execution_close_evidence",
                    "reason": "paper_execution_close_evidence_pending",
                    "state": "waiting",
                    "blocker_detail": "paper execution evidence is present, but position is still OPEN; waiting for close evidence",
                    "done_when": "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available",
                    "source_kind": "commodity_execution_retro",
                    "source_artifact": "/tmp/retro.json",
                    "source_status": "ok",
                    "source_as_of": "2026-03-11T08:40:08+00:00",
                    "source_age_minutes": 0,
                    "source_recency": "fresh",
                    "source_health": "ready",
                    "source_refresh_action": "read_current_artifact",
                },
                {
                    "slot": "followup",
                    "area": "commodity_fill_evidence",
                    "target": "commodity-paper-execution:metals_all:XAGUSD",
                    "symbol": "XAGUSD",
                    "action": "wait_for_paper_execution_fill_evidence",
                    "reason": "paper_execution_fill_evidence_pending",
                    "state": "waiting",
                    "blocker_detail": "paper execution fill evidence not written; stale directional signal 42d since 2026-01-26",
                    "done_when": "XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols",
                    "source_kind": "commodity_execution_review",
                    "source_artifact": "/tmp/review.json",
                    "source_status": "ok",
                    "source_as_of": "2026-03-11T08:40:07+00:00",
                    "source_age_minutes": 0,
                    "source_recency": "fresh",
                    "source_health": "ready",
                    "source_refresh_action": "read_current_artifact",
                },
                {
                    "slot": "secondary",
                    "area": "crypto_route",
                    "target": "BNBUSDT",
                    "symbol": "BNBUSDT",
                    "action": "watch_priority_until_long_window_confirms",
                    "reason": "secondary_focus",
                    "state": "watch",
                    "blocker_detail": "long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta",
                    "done_when": "BNBUSDT gains supporting crypto research edge or leaves priority watch",
                    "source_kind": "crypto_route",
                    "source_artifact": "/tmp/crypto.json",
                    "source_status": "ok",
                    "source_as_of": "2026-03-11T08:40:00+00:00",
                    "source_age_minutes": 0,
                    "source_recency": "fresh",
                    "source_health": "ready",
                    "source_refresh_action": "read_current_artifact",
                },
            ],
            "operator_action_queue_brief": (
                "1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence"
                " | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence"
                " | 3:crypto_route:BNBUSDT:watch_priority_until_long_window_confirms"
            ),
            "operator_action_checklist_brief": (
                "1:waiting:XAUUSD:wait_for_paper_execution_close_evidence"
                " | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence"
                " | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms"
            ),
            "operator_action_checklist": [
                {
                    "rank": 1,
                    "state": "waiting",
                    "symbol": "XAUUSD",
                    "action": "wait_for_paper_execution_close_evidence",
                    "blocker_detail": "paper execution evidence is present, but position is still OPEN; waiting for close evidence",
                    "done_when": "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available",
                },
                {
                    "rank": 2,
                    "state": "waiting",
                    "symbol": "XAGUSD",
                    "action": "wait_for_paper_execution_fill_evidence",
                    "blocker_detail": "paper execution fill evidence not written; stale directional signal 42d since 2026-01-26",
                    "done_when": "XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols",
                },
                {
                    "rank": 3,
                    "state": "watch",
                    "symbol": "BNBUSDT",
                    "action": "watch_priority_until_long_window_confirms",
                    "blocker_detail": "long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta",
                    "done_when": "BNBUSDT gains supporting crypto research edge or leaves priority watch",
                },
            ],
            "commodity_execution_review_status": "paper-execution-close-evidence-pending-fill-remainder",
            "commodity_execution_retro_status": "paper-execution-retro-pending",
            "commodity_execution_bridge_status": "blocked_stale_directional_signal",
            "commodity_execution_gap_status": "blocking_gap_active",
            "commodity_execution_bridge_already_bridged_symbols": ["XAUUSD"],
            "commodity_review_pending_symbols": [],
            "commodity_review_close_evidence_pending_count": 1,
            "commodity_review_close_evidence_pending_symbols": ["XAUUSD"],
            "commodity_next_review_close_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "commodity_next_review_close_evidence_execution_symbol": "XAUUSD",
            "commodity_retro_pending_symbols": ["XAUUSD"],
            "commodity_retro_fill_evidence_pending_symbols": ["XAGUSD", "COPPER"],
            "commodity_remainder_focus_area": "commodity_fill_evidence",
            "commodity_remainder_focus_target": "commodity-paper-execution:metals_all:XAGUSD",
            "commodity_remainder_focus_action": "wait_for_paper_execution_fill_evidence",
            "commodity_remainder_focus_signal_date": "2026-01-26",
            "commodity_remainder_focus_signal_age_days": 42,
            "commodity_execution_bridge_stale_signal_dates": {
                "XAGUSD": "2026-01-26",
                "COPPER": "2026-01-29",
            },
            "commodity_execution_bridge_stale_signal_age_days": {
                "XAGUSD": 42,
                "COPPER": 39,
            },
            "commodity_stale_signal_watch_items": [
                {"symbol": "XAGUSD", "signal_date": "2026-01-26", "signal_age_days": 42},
                {"symbol": "COPPER", "signal_date": "2026-01-29", "signal_age_days": 39},
            ],
            "commodity_stale_signal_watch_brief": "XAGUSD:42d@2026-01-26, COPPER:39d@2026-01-29",
            "commodity_stale_signal_watch_next_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "commodity_stale_signal_watch_next_symbol": "XAGUSD",
            "commodity_stale_signal_watch_next_signal_date": "2026-01-26",
            "commodity_stale_signal_watch_next_signal_age_days": 42,
            "commodity_next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "commodity_next_fill_evidence_execution_symbol": "XAGUSD",
            "commodity_focus_evidence_summary": {
                "paper_entry_price": 5198.10009765625,
                "paper_stop_price": 4847.7998046875,
                "paper_target_price": 5758.58056640625,
                "paper_quote_usdt": 0.15896067200583952,
                "paper_execution_status": "OPEN",
                "paper_signal_price_reference_source": "yfinance:GC=F",
            },
            "commodity_focus_lifecycle_status": "open_position_wait_close_evidence",
            "commodity_focus_lifecycle_brief": "open_position_wait_close_evidence:XAUUSD",
            "commodity_focus_lifecycle_blocker_detail": (
                "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
            ),
            "commodity_focus_lifecycle_done_when": (
                "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
            ),
            "commodity_execution_close_evidence_status": "close_evidence_pending",
            "commodity_execution_close_evidence_brief": "close_evidence_pending:XAUUSD",
            "commodity_execution_close_evidence_target": "commodity-paper-execution:metals_all:XAUUSD",
            "commodity_execution_close_evidence_symbol": "XAUUSD",
            "commodity_execution_close_evidence_blocker_detail": (
                "paper execution evidence is present, but position is still OPEN; waiting for close evidence"
            ),
            "commodity_execution_close_evidence_done_when": (
                "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
            ),
            "secondary_focus_area": "crypto_route",
            "secondary_focus_target": "BNBUSDT",
            "secondary_focus_symbol": "BNBUSDT",
            "secondary_focus_action": "watch_priority_until_long_window_confirms",
            "secondary_focus_reason": "secondary_focus",
            "secondary_focus_state": "watch",
            "secondary_focus_blocker_detail": "long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta",
            "secondary_focus_done_when": "BNBUSDT gains supporting crypto research edge or leaves priority watch",
        },
        review={
            "artifact": "/tmp/review.json",
            "next_review_execution_symbol": "XAUUSD",
            "next_close_evidence_execution_id": "commodity-paper-execution:metals_all:XAUUSD",
            "next_close_evidence_execution_symbol": "XAUUSD",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_fill_evidence_execution_symbol": "XAGUSD",
            "review_pending_symbols": [],
            "close_evidence_pending_symbols": ["XAUUSD"],
            "fill_evidence_pending_symbols": ["XAGUSD", "COPPER"],
        },
        retro={
            "artifact": "/tmp/retro.json",
            "next_retro_execution_symbol": "XAUUSD",
            "next_fill_evidence_execution_id": "commodity-paper-execution:metals_all:XAGUSD",
            "next_fill_evidence_execution_symbol": "XAGUSD",
            "retro_pending_symbols": ["XAUUSD"],
            "fill_evidence_pending_symbols": ["XAGUSD", "COPPER"],
        },
        gap={
            "artifact": "/tmp/gap.json",
            "queue_symbols_with_stale_directional_signal": ["XAGUSD", "COPPER"],
            "queue_symbols_with_stale_directional_signal_dates": {
                "XAGUSD": "2026-01-26",
                "COPPER": "2026-01-29",
            },
            "queue_symbols_with_stale_directional_signal_age_days": {
                "XAGUSD": 42,
                "COPPER": 39,
            },
            "stale_directional_signal_watch_items": [
                {"symbol": "XAGUSD", "signal_date": "2026-01-26", "signal_age_days": 42},
                {"symbol": "COPPER", "signal_date": "2026-01-29", "signal_age_days": 39},
            ],
            "queue_symbols_with_any_evidence": ["XAUUSD"],
            "queue_symbols_without_any_evidence": ["XAGUSD", "COPPER"],
        },
        bridge={
            "artifact": "/tmp/bridge.json",
        },
        bridge_apply={
            "artifact": "/tmp/bridge_apply.json",
            "applied_execution_ids": ["commodity-paper-execution:metals_all:XAUUSD"],
            "bridge_items": [
                {
                    "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                    "symbol": "XAUUSD",
                }
            ],
        },
    )
    assert "Paper execution evidence was written in this refresh cycle for: `XAUUSD`" in text
    assert "Commodity stale-signal watch remains active for: `XAGUSD, COPPER`" in text
    assert "Commodity stale-signal dates are: `COPPER:2026-01-29, XAGUSD:2026-01-26`" in text
    assert "Commodity stale-signal ages are: `COPPER:39, XAGUSD:42`" in text
    assert "Commodity stale-signal watch priority is: `XAGUSD:42d@2026-01-26, COPPER:39d@2026-01-29`" in text
    assert "Commodity stale-signal watch head is: `XAGUSD target=commodity-paper-execution:metals_all:XAGUSD date=2026-01-26 age=42d`" in text
    assert "Follow-up after primary focus is: `commodity_fill_evidence target=commodity-paper-execution:metals_all:XAGUSD action=wait_for_paper_execution_fill_evidence`" in text
    assert (
        "Primary focus gate is: `state=waiting | blocker=paper execution evidence is present, but position is still OPEN; waiting for close evidence | done_when=XAUUSD paper_execution_status leaves OPEN and close evidence becomes available`"
        in text
    )
    assert "Follow-up gate is: `state=waiting | blocker=paper execution fill evidence not written; stale directional signal 42d since 2026-01-26 | done_when=XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols`" in text
    assert "Secondary focus gate is: `state=watch | blocker=long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta | done_when=BNBUSDT gains supporting crypto research edge or leaves priority watch`" in text
    assert "Focus slots are: `primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence | followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | secondary:watch:BNBUSDT:watch_priority_until_long_window_confirms`" in text
    assert "Focus slot sources are: `primary:commodity_execution_retro | followup:commodity_execution_review | secondary:crypto_route`" in text
    assert "Focus slot source status is: `primary:ok@2026-03-11T08:40:08+00:00 | followup:ok@2026-03-11T08:40:07+00:00 | secondary:ok@2026-03-11T08:40:00+00:00`" in text
    assert "Focus slot source recency is: `primary:fresh:0m | followup:fresh:0m | secondary:fresh:0m`" in text
    assert "Focus slot source health is: `primary:ready:read_current_artifact | followup:ready:read_current_artifact | secondary:ready:read_current_artifact`" in text
    assert "Focus slot refresh backlog is: `-`" in text
    assert "Focus slot promotion gate is: `status=promotion_ready | ready=3/3 | blocker=all 3 focus slots have ready source artifacts | done_when=all focus slots continue using read_current_artifact sources`" in text
    assert "Focus slot actionability gate is: `status=actionability_guarded_by_content | actionable=2/3 | blocker=BNBUSDT secondary content state remains blocked (route_ahead_of_embedding, recovery_completed_no_edge): long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta | done_when=BNBUSDT gains supporting crypto research edge or leaves priority watch`" in text
    assert "Focus slot readiness gate is: `status=readiness_guarded_by_content | blocking_gate=content_actionability | ready=2/3 | blocker=BNBUSDT secondary content state remains blocked (route_ahead_of_embedding, recovery_completed_no_edge): long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta | done_when=BNBUSDT gains supporting crypto research edge or leaves priority watch`" in text
    assert "Research embedding quality is: `status=avoid_only | brief=avoid_only:crypto_hot, crypto_majors, crypto_beta" in text
    assert "Crypto route alignment is: `slot=secondary | status=route_ahead_of_embedding | brief=route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta" in text
    assert "Crypto route alignment recovery outcome is: `status=recovery_completed_no_edge | brief=recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta | failed=0 | timed_out=0 | zero_trade_batches=crypto_hot, crypto_majors, crypto_beta" in text
    assert "Crypto route alignment cooldown is: `status=cooldown_active_wait_for_new_market_data | brief=cooldown_active_wait_for_new_market_data:>2026-03-11 | last_end=2026-03-11 | next_eligible=2026-03-12 | blocker=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | done_when=hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "Crypto route alignment recovery recipe gate is: `status=deferred_by_cooldown | brief=deferred_by_cooldown:2026-03-12 | ready_on=2026-03-12 | blocker=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | done_when=hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "Crypto route alignment recovery template is: `script=/tmp/run_hot_universe_research.py | expected_status=ok | window_days=21 | target_batches=crypto_hot, crypto_majors, crypto_beta" in text
    assert "Source refresh queue is: `-`" in text
    assert "Source refresh pipeline pending is: `-`" in text
    assert "Source refresh pipeline deferred is: `status=deferred_by_cooldown | brief=1:refresh_crypto_route_brief:carry_over:crypto_route_brief | 2:refresh_crypto_route_operator_brief:carry_over:crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding:carry_over:hot_universe_research | until=2026-03-12 | reason=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome`" in text
    assert "Source refresh pipeline checkpoint is: `1:carry_over:crypto_route_brief | 2:carry_over:crypto_route_operator_brief | 3:carry_over:hot_universe_research`" in text
    assert "Source refresh pipeline head is: `step=- name=- state=- artifact=- current=-`" in text
    assert "Source refresh checklist is: `-`" in text
    assert "Focus slot refresh head is: `- slot=- action=- health=-`" in text
    assert "Next source refresh task is: `- slot=- action=- kind=- health=-`" in text
    assert "Next source refresh gate is: `state=- | blocker=- | done_when=-`" in text
    assert "Next source refresh recipe is: `script=- | expected_status=- | expected_artifact=-@- | note=-`" in text
    assert "Next source refresh follow-up is: `script=- | verify=-`" in text
    assert "Next source refresh pipeline is: `-`" in text
    assert "Next source refresh checkpoint is: `-`" in text
    assert "Action checklist is: `1:waiting:XAUUSD:wait_for_paper_execution_close_evidence | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms`" in text
    assert "Next commodity fill-evidence remainder is: `XAGUSD`." in text
    assert "Next commodity remainder signal date is: `2026-01-26`." in text
    assert "Next commodity remainder signal age is: `42 days`." in text
    assert "Current commodity paper evidence summary: `entry=5198.100098 stop=4847.799805 target=5758.580566 quote=0.158961 status=OPEN ref=yfinance:GC=F`" in text
    assert (
        "Commodity focus lifecycle is: `status=open_position_wait_close_evidence | brief=open_position_wait_close_evidence:XAUUSD | blocker=paper execution evidence is present, but position is still OPEN; waiting for close evidence | done_when=XAUUSD paper_execution_status leaves OPEN and close evidence becomes available`"
        in text
    )
    assert (
        "Commodity close-evidence lane is: `status=close_evidence_pending | brief=close_evidence_pending:XAUUSD | target=commodity-paper-execution:metals_all:XAUUSD | symbol=XAUUSD | blocker=paper execution evidence is present, but position is still OPEN; waiting for close evidence | done_when=XAUUSD paper_execution_status leaves OPEN and close evidence becomes available`"
        in text
    )
    assert "- `next_focus_state = waiting`" in text
    assert (
        "- `next_focus_blocker_detail = paper execution evidence is present, but position is still OPEN; waiting for close evidence`"
        in text
    )
    assert (
        "- `next_focus_done_when = XAUUSD paper_execution_status leaves OPEN and close evidence becomes available`"
        in text
    )
    assert "- `followup_focus_area = commodity_fill_evidence`" in text
    assert "- `followup_focus_target = commodity-paper-execution:metals_all:XAGUSD`" in text
    assert "- `followup_focus_action = wait_for_paper_execution_fill_evidence`" in text
    assert "- `followup_focus_state = waiting`" in text
    assert "- `followup_focus_blocker_detail = paper execution fill evidence not written; stale directional signal 42d since 2026-01-26`" in text
    assert "- `followup_focus_done_when = XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols`" in text
    assert "- `secondary_focus_area = crypto_route`" in text
    assert "- `secondary_focus_target = BNBUSDT`" in text
    assert "- `secondary_focus_symbol = BNBUSDT`" in text
    assert "- `secondary_focus_action = watch_priority_until_long_window_confirms`" in text
    assert "- `secondary_focus_reason = secondary_focus`" in text
    assert "- `secondary_focus_state = watch`" in text
    assert "- `secondary_focus_blocker_detail = long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- `secondary_focus_done_when = BNBUSDT gains supporting crypto research edge or leaves priority watch`" in text
    assert "- `operator_focus_slot_sources_brief = primary:commodity_execution_retro | followup:commodity_execution_review | secondary:crypto_route`" in text
    assert "- `operator_focus_slot_status_brief = primary:ok@2026-03-11T08:40:08+00:00 | followup:ok@2026-03-11T08:40:07+00:00 | secondary:ok@2026-03-11T08:40:00+00:00`" in text
    assert "- `operator_focus_slot_recency_brief = primary:fresh:0m | followup:fresh:0m | secondary:fresh:0m`" in text
    assert "- `operator_focus_slot_health_brief = primary:ready:read_current_artifact | followup:ready:read_current_artifact | secondary:ready:read_current_artifact`" in text
    assert "- `operator_focus_slot_refresh_backlog_brief = -`" in text
    assert "- `operator_focus_slot_refresh_backlog_count = 0`" in text
    assert "- `operator_focus_slot_refresh_backlog = []`" in text
    assert "- `operator_focus_slot_ready_count = 3`" in text
    assert "- `operator_focus_slot_total_count = 3`" in text
    assert "- `operator_focus_slot_promotion_gate_brief = promotion_ready:3/3`" in text
    assert "- `operator_focus_slot_promotion_gate_status = promotion_ready`" in text
    assert "- `operator_focus_slot_promotion_gate_blocker_detail = all 3 focus slots have ready source artifacts`" in text
    assert "- `operator_focus_slot_promotion_gate_done_when = all focus slots continue using read_current_artifact sources`" in text
    assert "- `operator_focus_slot_actionability_backlog_brief = secondary:BNBUSDT:recovery_completed_no_edge`" in text
    assert "- `operator_focus_slot_actionability_backlog_count = 1`" in text
    assert "- `operator_focus_slot_actionable_count = 2`" in text
    assert "- `operator_focus_slot_actionability_gate_brief = actionability_guarded_by_content:2/3`" in text
    assert "- `operator_focus_slot_actionability_gate_status = actionability_guarded_by_content`" in text
    assert "- `operator_focus_slot_actionability_gate_blocker_detail = BNBUSDT secondary content state remains blocked (route_ahead_of_embedding, recovery_completed_no_edge): long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- `operator_focus_slot_actionability_gate_done_when = BNBUSDT gains supporting crypto research edge or leaves priority watch`" in text
    assert "- `operator_focus_slot_readiness_gate_ready_count = 2`" in text
    assert "- `operator_focus_slot_readiness_gate_brief = readiness_guarded_by_content:2/3`" in text
    assert "- `operator_focus_slot_readiness_gate_status = readiness_guarded_by_content`" in text
    assert "- `operator_focus_slot_readiness_gate_blocking_gate = content_actionability`" in text
    assert "- `operator_focus_slot_readiness_gate_blocker_detail = BNBUSDT secondary content state remains blocked (route_ahead_of_embedding, recovery_completed_no_edge): long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- `operator_focus_slot_readiness_gate_done_when = BNBUSDT gains supporting crypto research edge or leaves priority watch`" in text
    assert "- `operator_research_embedding_quality_status = avoid_only`" in text
    assert "- `operator_research_embedding_quality_brief = avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- `operator_crypto_route_alignment_focus_slot = secondary`" in text
    assert "- `operator_crypto_route_alignment_status = route_ahead_of_embedding`" in text
    assert "- `operator_crypto_route_alignment_brief = route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- `operator_crypto_route_alignment_recovery_status = recovery_completed_no_edge`" in text
    assert "- `operator_crypto_route_alignment_recovery_brief = recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- `operator_crypto_route_alignment_recovery_zero_trade_batches = crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- `operator_crypto_route_alignment_cooldown_status = cooldown_active_wait_for_new_market_data`" in text
    assert "- `operator_crypto_route_alignment_cooldown_brief = cooldown_active_wait_for_new_market_data:>2026-03-11`" in text
    assert "- `operator_crypto_route_alignment_cooldown_blocker_detail = latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome`" in text
    assert "- `operator_crypto_route_alignment_cooldown_done_when = hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "- `operator_crypto_route_alignment_cooldown_last_research_end_date = 2026-03-11`" in text
    assert "- `operator_crypto_route_alignment_cooldown_next_eligible_end_date = 2026-03-12`" in text
    assert "- `operator_crypto_route_alignment_recipe_status = deferred_by_cooldown`" in text
    assert "- `operator_crypto_route_alignment_recipe_brief = deferred_by_cooldown:2026-03-12`" in text
    assert "- `operator_crypto_route_alignment_recipe_blocker_detail = latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome`" in text
    assert "- `operator_crypto_route_alignment_recipe_done_when = hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "- `operator_crypto_route_alignment_recipe_ready_on_date = 2026-03-12`" in text
    assert "- `operator_crypto_route_alignment_recipe_script = /tmp/run_hot_universe_research.py`" in text
    assert "- `operator_crypto_route_alignment_recipe_expected_status = ok`" in text
    assert "- `operator_crypto_route_alignment_recipe_window_days = 21`" in text
    assert "- `operator_crypto_route_alignment_recipe_target_batches = crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- `operator_source_refresh_queue_brief = -`" in text
    assert "- `operator_source_refresh_queue_count = 0`" in text
    assert "- `operator_source_refresh_queue = []`" in text
    assert "- `operator_source_refresh_checklist_brief = -`" in text
    assert "- `operator_source_refresh_checklist = []`" in text
    assert "- `operator_source_refresh_pipeline_steps_brief = -`" in text
    assert "- `operator_source_refresh_pipeline_step_checkpoint_brief = 1:carry_over:crypto_route_brief | 2:carry_over:crypto_route_operator_brief | 3:carry_over:hot_universe_research`" in text
    assert "- `operator_source_refresh_pipeline_pending_brief = -`" in text
    assert "- `operator_source_refresh_pipeline_pending_count = 0`" in text
    assert "- `operator_source_refresh_pipeline_head_rank = -`" in text
    assert "- `operator_source_refresh_pipeline_head_name = -`" in text
    assert "- `operator_source_refresh_pipeline_head_checkpoint_state = -`" in text
    assert "- `operator_source_refresh_pipeline_head_expected_artifact_kind = -`" in text
    assert "- `operator_source_refresh_pipeline_head_current_artifact = -`" in text
    assert "- `operator_source_refresh_pipeline_deferred_brief = 1:refresh_crypto_route_brief:carry_over:crypto_route_brief | 2:refresh_crypto_route_operator_brief:carry_over:crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding:carry_over:hot_universe_research`" in text
    assert "- `operator_source_refresh_pipeline_deferred_count = 3`" in text
    assert "- `operator_source_refresh_pipeline_deferred_status = deferred_by_cooldown`" in text
    assert "- `operator_source_refresh_pipeline_deferred_until = 2026-03-12`" in text
    assert "- `operator_source_refresh_pipeline_deferred_reason = latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome`" in text
    assert "- `operator_source_refresh_pipeline_deferred_head_rank = 1`" in text
    assert "- `operator_source_refresh_pipeline_deferred_head_name = refresh_crypto_route_brief`" in text
    assert "- `operator_source_refresh_pipeline_deferred_head_checkpoint_state = carry_over`" in text
    assert "- `operator_source_refresh_pipeline_deferred_head_expected_artifact_kind = crypto_route_brief`" in text
    assert "- `operator_source_refresh_pipeline_deferred_head_current_artifact = /tmp/review/20260311T084000Z_crypto_route_brief.json`" in text
    assert "- `next_focus_source_kind = commodity_execution_retro`" in text
    assert "- `next_focus_source_artifact = /tmp/retro.json`" in text
    assert "- `next_focus_source_status = ok`" in text
    assert "- `next_focus_source_as_of = 2026-03-11T08:40:08+00:00`" in text
    assert "- `next_focus_source_age_minutes = 0`" in text
    assert "- `next_focus_source_recency = fresh`" in text
    assert "- `next_focus_source_health = ready`" in text
    assert "- `next_focus_source_refresh_action = read_current_artifact`" in text
    assert "- `followup_focus_source_kind = commodity_execution_review`" in text
    assert "- `followup_focus_source_artifact = /tmp/review.json`" in text
    assert "- `followup_focus_source_status = ok`" in text
    assert "- `followup_focus_source_as_of = 2026-03-11T08:40:07+00:00`" in text
    assert "- `followup_focus_source_age_minutes = 0`" in text
    assert "- `followup_focus_source_recency = fresh`" in text
    assert "- `followup_focus_source_health = ready`" in text
    assert "- `followup_focus_source_refresh_action = read_current_artifact`" in text
    assert "- `secondary_focus_source_kind = crypto_route`" in text
    assert "- `secondary_focus_source_artifact = /tmp/crypto.json`" in text
    assert "- `secondary_focus_source_status = ok`" in text
    assert "- `secondary_focus_source_as_of = 2026-03-11T08:40:00+00:00`" in text
    assert "- `secondary_focus_source_age_minutes = 0`" in text
    assert "- `secondary_focus_source_recency = fresh`" in text
    assert "- `secondary_focus_source_health = ready`" in text
    assert "- `secondary_focus_source_refresh_action = read_current_artifact`" in text
    assert "- `operator_focus_slot_refresh_head_slot = -`" in text
    assert "- `operator_focus_slot_refresh_head_symbol = -`" in text
    assert "- `operator_focus_slot_refresh_head_action = -`" in text
    assert "- `operator_focus_slot_refresh_head_health = -`" in text
    assert "- `operator_source_refresh_next_slot = -`" in text
    assert "- `operator_source_refresh_next_symbol = -`" in text
    assert "- `operator_source_refresh_next_action = -`" in text
    assert "- `operator_source_refresh_next_source_kind = -`" in text
    assert "- `operator_source_refresh_next_source_health = -`" in text
    assert "- `operator_source_refresh_next_source_artifact = -`" in text
    assert "- `operator_source_refresh_next_state = -`" in text
    assert "- `operator_source_refresh_next_blocker_detail = -`" in text
    assert "- `operator_source_refresh_next_done_when = -`" in text
    assert "- `operator_source_refresh_next_recipe_script = -`" in text
    assert "- `operator_source_refresh_next_recipe_command_hint = -`" in text
    assert "- `operator_source_refresh_next_recipe_expected_status = -`" in text
    assert "- `operator_source_refresh_next_recipe_expected_artifact_kind = -`" in text
    assert "- `operator_source_refresh_next_recipe_expected_artifact_path_hint = -`" in text
    assert "- `operator_source_refresh_next_recipe_note = -`" in text
    assert "- `operator_source_refresh_next_recipe_followup_script = -`" in text
    assert "- `operator_source_refresh_next_recipe_followup_command_hint = -`" in text
    assert "- `operator_source_refresh_next_recipe_verify_hint = -`" in text
    assert "- `operator_source_refresh_next_recipe_steps_brief = -`" in text
    assert "- `operator_source_refresh_next_recipe_step_checkpoint_brief = -`" in text
    assert "- `operator_source_refresh_next_recipe_steps = []`" in text
    assert "- `operator_focus_slots_brief = primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence | followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | secondary:watch:BNBUSDT:watch_priority_until_long_window_confirms`" in text
    assert "- `operator_action_queue_brief = 1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence | 3:crypto_route:BNBUSDT:watch_priority_until_long_window_confirms`" in text
    assert "- `operator_action_checklist_brief = 1:waiting:XAUUSD:wait_for_paper_execution_close_evidence | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms`" in text
    assert "- `commodity_remainder_focus_area = commodity_fill_evidence`" in text
    assert "- `commodity_remainder_focus_signal_date = 2026-01-26`" in text
    assert "- `commodity_remainder_focus_signal_age_days = 42`" in text
    assert "- `commodity_execution_bridge_stale_signal_dates = {\"COPPER\": \"2026-01-29\", \"XAGUSD\": \"2026-01-26\"}`" in text
    assert "- `commodity_execution_bridge_stale_signal_age_days = {\"COPPER\": 39, \"XAGUSD\": 42}`" in text
    assert "- `commodity_stale_signal_watch_brief = XAGUSD:42d@2026-01-26, COPPER:39d@2026-01-29`" in text
    assert "- `commodity_stale_signal_watch_next_execution_id = commodity-paper-execution:metals_all:XAGUSD`" in text
    assert "- `commodity_stale_signal_watch_next_symbol = XAGUSD`" in text
    assert "- `commodity_stale_signal_watch_next_signal_date = 2026-01-26`" in text
    assert "- `commodity_stale_signal_watch_next_signal_age_days = 42`" in text
    assert "- `commodity_focus_evidence_summary = {\"paper_entry_price\": 5198.10009765625, \"paper_execution_status\": \"OPEN\", \"paper_quote_usdt\": 0.15896067200583952, \"paper_signal_price_reference_source\": \"yfinance:GC=F\", \"paper_stop_price\": 4847.7998046875, \"paper_target_price\": 5758.58056640625}`" in text
    assert "- `commodity_focus_lifecycle_status = open_position_wait_close_evidence`" in text
    assert "- `commodity_focus_lifecycle_brief = open_position_wait_close_evidence:XAUUSD`" in text
    assert (
        "- `commodity_focus_lifecycle_blocker_detail = paper execution evidence is present, but position is still OPEN; waiting for close evidence`"
        in text
    )
    assert (
        "- `commodity_focus_lifecycle_done_when = XAUUSD paper_execution_status leaves OPEN and close evidence becomes available`"
        in text
    )
    assert "- `commodity_execution_close_evidence_status = close_evidence_pending`" in text
    assert "- `commodity_execution_close_evidence_brief = close_evidence_pending:XAUUSD`" in text
    assert "- `commodity_execution_close_evidence_target = commodity-paper-execution:metals_all:XAUUSD`" in text
    assert "- `commodity_execution_close_evidence_symbol = XAUUSD`" in text
    assert (
        "- `commodity_execution_close_evidence_blocker_detail = paper execution evidence is present, but position is still OPEN; waiting for close evidence`"
        in text
    )
    assert (
        "- `commodity_execution_close_evidence_done_when = XAUUSD paper_execution_status leaves OPEN and close evidence becomes available`"
        in text
    )
    assert "- `commodity_review_close_evidence_pending_symbols = XAUUSD`" in text
    assert "- `commodity_next_review_close_evidence_execution_symbol = XAUUSD`" in text
    assert "- `commodity_next_review_close_evidence_execution_id = commodity-paper-execution:metals_all:XAUUSD`" in text
    assert "- review pending symbols: `-`" in text
    assert "- review close-evidence next symbol: `XAUUSD`" in text
    assert "- review close-evidence next target: `commodity-paper-execution:metals_all:XAUUSD`" in text
    assert "- review close-evidence pending symbols: `XAUUSD`" in text
    assert "- retro pending symbols: `XAUUSD`" in text
    assert "- fill evidence next symbol: `XAGUSD`" in text
    assert "- fill evidence pending symbols: `XAGUSD, COPPER`" in text
    assert "- already bridged symbols: `XAUUSD`" in text
    assert "- stale directional signal dates: `COPPER:2026-01-29, XAGUSD:2026-01-26`" in text
    assert "- stale directional signal ages: `COPPER:39, XAGUSD:42`" in text
    assert "- stale directional watch priority: `XAGUSD:42d@2026-01-26, COPPER:39d@2026-01-29`" in text
    assert "- stale directional watch head: `XAGUSD | commodity-paper-execution:metals_all:XAGUSD | 2026-01-26 | 42d`" in text
    assert "- commodity close-evidence lane: `close_evidence_pending:XAUUSD`" in text
    assert "- action queue: `1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence | 3:crypto_route:BNBUSDT:watch_priority_until_long_window_confirms`" in text
    assert "- action checklist: `1:waiting:XAUUSD:wait_for_paper_execution_close_evidence | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms`" in text
    assert "- focus slot refresh backlog: `-`" in text
    assert "- focus slot promotion gate: `promotion_ready:3/3`" in text
    assert "- focus slot actionability gate: `actionability_guarded_by_content:2/3`" in text
    assert "- focus slot readiness gate: `readiness_guarded_by_content:2/3`" in text
    assert "- research embedding quality: `avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- crypto route alignment slot: `secondary`" in text
    assert "- crypto route alignment: `route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- crypto route alignment recovery outcome: `recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- crypto route alignment cooldown: `cooldown_active_wait_for_new_market_data:>2026-03-11`" in text
    assert "- crypto route alignment recovery recipe gate: `deferred_by_cooldown:2026-03-12`" in text
    assert "- crypto route alignment recovery: `crypto_hot, crypto_majors, crypto_beta@21d`" in text
    assert "- source refresh queue: `-`" in text
    assert "- source refresh checklist: `-`" in text
    assert "- source refresh pipeline: `-`" in text
    assert "- source refresh pipeline deferred: `1:refresh_crypto_route_brief:carry_over:crypto_route_brief | 2:refresh_crypto_route_operator_brief:carry_over:crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding:carry_over:hot_universe_research`" in text
    assert "## Focus Slot Artifacts" in text
    assert "- primary source: `commodity_execution_retro | ok | fresh | ready | read_current_artifact | 0m | 2026-03-11T08:40:08+00:00 | /tmp/retro.json`" in text
    assert "- followup source: `commodity_execution_review | ok | fresh | ready | read_current_artifact | 0m | 2026-03-11T08:40:07+00:00 | /tmp/review.json`" in text
    assert "- secondary source: `crypto_route | ok | fresh | ready | read_current_artifact | 0m | 2026-03-11T08:40:00+00:00 | /tmp/crypto.json`" in text
    assert "## Focus Slot Refresh Backlog" in text
    assert "- No focus slot source refresh backlog remains." in text
    assert "## Research Embedding Quality" in text
    assert "- status=`avoid_only` brief=`avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "## Crypto Route Alignment" in text
    assert "- slot=`secondary` status=`route_ahead_of_embedding` brief=`route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- recovery_outcome=`recovery_completed_no_edge | recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta | failed=0 | timed_out=0 | zero_trade_batches=crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- cooldown=`cooldown_active_wait_for_new_market_data | cooldown_active_wait_for_new_market_data:>2026-03-11 | last_end=2026-03-11 | next_eligible=2026-03-12 | blocker=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | done_when=hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "- recovery_recipe_gate=`deferred_by_cooldown | deferred_by_cooldown:2026-03-12 | ready_on=2026-03-12 | blocker=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | done_when=hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "- recovery=`crypto_hot, crypto_majors, crypto_beta@21d`" in text
    assert "- recovery_script=`/tmp/run_hot_universe_research.py`" in text
    assert "## Source Refresh Queue" in text
    assert "- No source refresh queue remains." in text
    assert "## Source Refresh Pipeline" in text
    assert "- No source refresh pipeline pending remains." in text
    assert "- deferred=`1:refresh_crypto_route_brief:carry_over:crypto_route_brief | 2:refresh_crypto_route_operator_brief:carry_over:crypto_route_operator_brief | 3:refresh_hot_universe_research_embedding:carry_over:hot_universe_research | status=deferred_by_cooldown | until=2026-03-12 | reason=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | checkpoint=1:carry_over:crypto_route_brief | 2:carry_over:crypto_route_operator_brief | 3:carry_over:hot_universe_research`" in text
    assert "- deferred_head=`step 1 | refresh_crypto_route_brief | carry_over | crypto_route_brief | /tmp/review/20260311T084000Z_crypto_route_brief.json`" in text
    assert "## Source Refresh Checklist" in text
    assert "- No source refresh checklist remains." in text
    assert "## Action Checklist" in text
    assert (
        "- 1. `waiting` `XAUUSD` `wait_for_paper_execution_close_evidence` blocker=`paper execution evidence is present, but position is still OPEN; waiting for close evidence` done_when=`XAUUSD paper_execution_status leaves OPEN and close evidence becomes available`"
        in text
    )
    assert "- 2. `waiting` `XAGUSD` `wait_for_paper_execution_fill_evidence` blocker=`paper execution fill evidence not written; stale directional signal 42d since 2026-01-26` done_when=`XAGUSD gains paper evidence and leaves fill_evidence_pending_symbols`" in text
    assert "- 3. `watch` `BNBUSDT` `watch_priority_until_long_window_confirms` blocker=`long-window confirmation still missing; clean crypto recovery still shows no edge in crypto_hot, crypto_majors, crypto_beta` done_when=`BNBUSDT gains supporting crypto research edge or leaves priority watch`" in text
    assert "wait_for_paper_execution_close_evidence" in text
    assert "/tmp/brief.json" in text


def test_derive_runtime_now_advances_past_latest_artifact(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260311T084604Z_hot_universe_operator_brief.json").write_text("{}", encoding="utf-8")
    mod.now_utc = lambda: mod.parse_now("2026-03-11T08:46:00Z")
    derived = mod.derive_runtime_now(review_dir, "")
    assert derived.isoformat() == "2026-03-11T08:46:05+00:00"
