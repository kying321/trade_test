from __future__ import annotations

import importlib.util
import json
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


def test_write_hot_brief_snapshot_persists_refresh_owned_copy(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    source_path = review_dir / "20260316T090525Z_hot_universe_operator_brief.json"
    source_text = json.dumps(
        {"status": "ok", "artifact": str(source_path), "operator_status": "ok"},
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    source_path.write_text(source_text, encoding="utf-8")

    snapshot_path = mod.write_hot_brief_snapshot(
        review_dir,
        stamp="20260316T090525Z",
        brief_payload={"artifact": str(source_path), "operator_status": "ok"},
    )

    assert snapshot_path.name == "20260316T090525Z_commodity_paper_execution_refresh_hot_brief_snapshot.json"
    assert snapshot_path.read_text(encoding="utf-8") == source_text


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
            "operator_crypto_route_alignment_focus_area": "crypto_route",
            "operator_crypto_route_alignment_focus_slot": "secondary",
            "operator_crypto_route_alignment_focus_symbol": "BNBUSDT",
            "operator_crypto_route_alignment_focus_action": "watch_priority_until_long_window_confirms",
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
            "operator_source_refresh_pipeline_relevance_status": "non_blocking_for_current_crypto_head",
            "operator_source_refresh_pipeline_relevance_brief": "non_blocking_for_current_crypto_head:BNBUSDT:3",
            "operator_source_refresh_pipeline_relevance_blocker_detail": (
                "BNBUSDT current source lane is already ready via read_current_artifact; remaining source refresh pipeline work is broader carry-over and does not block the current crypto head."
            ),
            "operator_source_refresh_pipeline_relevance_done_when": (
                "run the remaining source refresh pipeline only when broader refresh freshness is required or the crypto review queue head changes"
            ),
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
            "crypto_route_head_source_refresh_status": "ready",
            "crypto_route_head_source_refresh_brief": "ready:BNBUSDT:read_current_artifact",
            "crypto_route_head_source_refresh_slot": "secondary",
            "crypto_route_head_source_refresh_symbol": "BNBUSDT",
            "crypto_route_head_source_refresh_action": "read_current_artifact",
            "crypto_route_head_source_refresh_source_kind": "crypto_route",
            "crypto_route_head_source_refresh_source_health": "ready",
            "crypto_route_head_source_refresh_source_artifact": "/tmp/crypto.json",
            "crypto_route_head_source_refresh_blocker_detail": (
                "BNBUSDT currently uses a readable crypto_route artifact (ok, fresh)"
            ),
            "crypto_route_head_source_refresh_done_when": (
                "keep BNBUSDT on the current crypto_route artifact while it remains usable"
            ),
            "crypto_route_head_source_refresh_recipe_script": "",
            "crypto_route_head_source_refresh_recipe_command_hint": "",
            "crypto_route_head_source_refresh_recipe_expected_status": "",
            "crypto_route_head_source_refresh_recipe_expected_artifact_kind": "",
            "crypto_route_head_source_refresh_recipe_expected_artifact_path_hint": "",
            "crypto_route_head_source_refresh_recipe_note": "",
            "crypto_route_head_source_refresh_recipe_followup_script": "",
            "crypto_route_head_source_refresh_recipe_followup_command_hint": "",
            "crypto_route_head_source_refresh_recipe_verify_hint": "",
            "crypto_route_head_source_refresh_recipe_steps_brief": "",
            "crypto_route_head_source_refresh_recipe_step_checkpoint_brief": "",
            "crypto_route_head_source_refresh_recipe_steps": [],
            "source_crypto_route_refresh_artifact": "/tmp/crypto-route-refresh.json",
            "source_crypto_route_refresh_status": "ok",
            "source_crypto_route_refresh_as_of": "2026-03-11T08:40:09+00:00",
            "source_crypto_route_refresh_native_mode": "skip_native_refresh",
            "source_crypto_route_refresh_native_step_count": 9,
            "source_crypto_route_refresh_reused_native_count": 9,
            "source_crypto_route_refresh_missing_reused_count": 0,
            "source_crypto_route_refresh_reuse_status": "reused_native_inputs",
            "source_crypto_route_refresh_reuse_brief": "reused_native_inputs:skip_native_refresh:9/9",
            "source_crypto_route_refresh_reuse_note": (
                "crypto_route_refresh is currently reusing 9/9 native inputs via skip_native_refresh"
            ),
            "source_crypto_route_refresh_reuse_done_when": (
                "run full native refresh only when fresh native recomputation is required"
            ),
            "source_crypto_route_refresh_reuse_level": "informational",
            "source_crypto_route_refresh_reuse_gate_status": "reuse_non_blocking",
            "source_crypto_route_refresh_reuse_gate_brief": "reuse_non_blocking:skip_native_refresh:9/9",
            "source_crypto_route_refresh_reuse_gate_blocking": False,
            "source_crypto_route_refresh_reuse_gate_blocker_detail": (
                "all tracked native steps were intentionally reused; current refresh remains safe for downstream consumption."
            ),
            "source_crypto_route_refresh_reuse_gate_done_when": (
                "run full native refresh only when fresh native recomputation is explicitly required"
            ),
            "source_remote_live_history_audit_artifact": "/tmp/remote_live_history_audit.json",
            "source_remote_live_history_audit_status": "ok",
            "source_remote_live_history_audit_as_of": "2026-03-13T01:20:00Z",
            "source_remote_live_history_audit_market": "portfolio_margin_um",
            "source_remote_live_history_audit_window_brief": (
                "24h:12.5pnl/3tr/1open | 7d:4.5pnl/5tr/0open | 30d:15.25pnl/9tr/0open"
            ),
            "source_remote_live_history_audit_quote_available": 99.5,
            "source_remote_live_history_audit_open_positions": 1,
            "source_remote_live_history_audit_risk_guard_status": "blocked",
            "source_remote_live_history_audit_risk_guard_reasons": [
                "ticket_missing:no_actionable_ticket",
                "panic_cooldown_active",
            ],
            "source_remote_live_history_audit_blocked_candidate_symbol": "BNBUSDT",
            "source_remote_live_history_audit_30d_symbol_pnl_brief": "BTCUSDT:8.5, ETHUSDT:1.25",
            "source_remote_live_history_audit_30d_day_pnl_brief": "2026-03-11:-1.0, 2026-03-12:2.5",
            "source_remote_live_handoff_artifact": "/tmp/remote-live-handoff.json",
            "source_remote_live_handoff_status": "ok",
            "source_remote_live_handoff_as_of": "2026-03-13T01:21:00Z",
            "source_remote_live_handoff_state": "ops_live_gate_blocked",
            "source_remote_live_handoff_status_triplet": "runtime-ok / gate-blocked / risk-guard-blocked",
            "source_remote_live_handoff_ready_check_scope_market": "portfolio_margin_um",
            "source_remote_live_handoff_ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
            "source_remote_live_handoff_account_scope_alignment_status": "split_scope_spot_vs_portfolio_margin_um",
            "source_remote_live_handoff_account_scope_alignment_brief": "split_scope_spot_vs_portfolio_margin_um",
            "source_remote_live_handoff_account_scope_alignment_blocking": False,
            "source_remote_live_handoff_account_scope_alignment_blocker_detail": "spot ready-check and unified-account history refer to different execution scopes.",
            "source_live_gate_blocker_artifact": "/tmp/live_gate_blocker_report.json",
            "source_live_gate_blocker_as_of": "2026-03-13T01:22:00Z",
            "source_live_gate_blocker_live_decision": "do_not_start_formal_live",
            "source_live_gate_blocker_remote_live_diagnosis_status": "profitability_confirmed_but_auto_live_blocked",
            "source_live_gate_blocker_remote_live_diagnosis_brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
            "source_live_gate_blocker_remote_live_diagnosis_blocker_detail": "portfolio_margin_um remote history confirms realized profitability (30d pnl=15.25000000 across 9 trades), but automated live remains blocked by ops_live_gate, risk_guard.",
            "source_live_gate_blocker_remote_live_diagnosis_done_when": "clear ops_live_gate and risk_guard blockers while keeping the intended ready-check scope aligned with the profitable execution account",
            "source_live_gate_blocker_remote_live_operator_alignment_status": "local_operator_active_remote_live_blocked",
            "source_live_gate_blocker_remote_live_operator_alignment_brief": "local_operator_active_remote_live_blocked:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked",
            "source_live_gate_blocker_remote_live_operator_alignment_blocker_detail": "Current local operator head is XAUUSD (commodity_execution_close_evidence/wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote automated live remains profitability_confirmed_but_auto_live_blocked.",
            "source_live_gate_blocker_remote_live_operator_alignment_done_when": "XAUUSD local operator head progresses and remote auto-live blockers are cleared",
            "cross_market_remote_live_takeover_gate_status": "blocked_by_remote_live_gate",
            "cross_market_remote_live_takeover_gate_brief": "blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked",
            "cross_market_remote_live_takeover_gate_blocker_detail": "Current local operator head is XAUUSD (commodity_execution_close_evidence/wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote automated live remains profitability_confirmed_but_auto_live_blocked.",
            "cross_market_remote_live_takeover_gate_done_when": "XAUUSD local operator head progresses and remote auto-live blockers are cleared",
            "cross_market_remote_live_takeover_clearing_status": "clearing_required",
            "cross_market_remote_live_takeover_clearing_brief": "clearing_required:ops_live_gate+risk_guard",
            "cross_market_remote_live_takeover_clearing_blocker_detail": "ops_live_gate needs rollback_hard, risk_violations, max_drawdown, slot_anomaly; risk_guard needs ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap",
            "cross_market_remote_live_takeover_clearing_done_when": "ops_live_gate becomes clear and risk_guard reasons become empty",
            "remote_live_takeover_repair_queue_status": "ready",
            "remote_live_takeover_repair_queue_brief": "ready:ops_live_gate:rollback_hard:99",
            "remote_live_takeover_repair_queue_queue_brief": "1:ops_live_gate:rollback_hard:99 | 2:ops_live_gate:slot_anomaly:98 | 3:ops_live_gate:backtest_snapshot:97 | 4:ops_live_gate:ops_status_red:96 | 5:risk_guard:ticket_missing:no_actionable_ticket:89 | 6:risk_guard:panic_cooldown_active:88 | 7:risk_guard:open_exposure_above_cap:87",
            "remote_live_takeover_repair_queue_count": 7,
            "remote_live_takeover_repair_queue_head_area": "ops_live_gate",
            "remote_live_takeover_repair_queue_head_code": "rollback_hard",
            "remote_live_takeover_repair_queue_head_action": "clear_ops_live_gate_condition",
            "remote_live_takeover_repair_queue_head_priority_score": 99,
            "remote_live_takeover_repair_queue_head_priority_tier": "repair_queue_now",
            "remote_live_takeover_repair_queue_head_command": "cmd-gate",
            "remote_live_takeover_repair_queue_head_clear_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
            "remote_live_takeover_repair_queue_done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
            "cross_market_operator_repair_head_status": "ready",
            "cross_market_operator_repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
            "cross_market_operator_repair_head_area": "ops_live_gate",
            "cross_market_operator_repair_head_code": "rollback_hard",
            "cross_market_operator_repair_head_action": "clear_ops_live_gate_condition",
            "cross_market_operator_repair_head_priority_score": 99,
            "cross_market_operator_repair_head_priority_tier": "repair_queue_now",
            "cross_market_operator_repair_head_command": "cmd-gate",
            "cross_market_operator_repair_head_clear_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
            "cross_market_operator_repair_head_done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
            "cross_market_operator_repair_backlog_status": "ready",
            "cross_market_operator_repair_backlog_brief": "1:ops_live_gate:rollback_hard:99 | 2:ops_live_gate:slot_anomaly:98 | 3:ops_live_gate:backtest_snapshot:97 | 4:ops_live_gate:ops_status_red:96 | 5:risk_guard:ticket_missing:no_actionable_ticket:89 | 6:risk_guard:panic_cooldown_active:88 | 7:risk_guard:open_exposure_above_cap:87",
            "cross_market_operator_repair_backlog_count": 7,
            "cross_market_operator_repair_backlog_priority_total": 654,
            "cross_market_operator_repair_backlog_done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
            "source_cross_market_operator_state_remote_live_takeover_gate_status": "blocked_by_remote_live_gate",
            "source_cross_market_operator_state_remote_live_takeover_gate_brief": "blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked",
            "source_cross_market_operator_state_remote_live_takeover_gate_blocker_detail": "Current local operator head is XAUUSD (commodity_execution_close_evidence/wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote automated live remains profitability_confirmed_but_auto_live_blocked.",
            "source_cross_market_operator_state_remote_live_takeover_gate_done_when": "XAUUSD local operator head progresses and remote auto-live blockers are cleared",
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
                " | 4:ops_live_gate:rollback_hard:clear_ops_live_gate_condition"
            ),
            "operator_action_checklist_brief": (
                "1:waiting:XAUUSD:wait_for_paper_execution_close_evidence"
                " | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence"
                " | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms"
                " | 4:repair:ROLLBACK_HARD:clear_ops_live_gate_condition"
            ),
            "operator_repair_queue_brief": (
                "1:ops_live_gate:rollback_hard:clear_ops_live_gate_condition"
                " | 2:ops_live_gate:slot_anomaly:clear_ops_live_gate_condition"
                " | 3:ops_live_gate:backtest_snapshot:clear_ops_live_gate_condition"
                " | 4:ops_live_gate:ops_status_red:clear_ops_live_gate_condition | +3"
            ),
            "operator_repair_queue_count": 7,
            "operator_repair_queue": [
                {
                    "rank": 1,
                    "area": "ops_live_gate",
                    "target": "rollback_hard",
                    "symbol": "ROLLBACK_HARD",
                    "action": "clear_ops_live_gate_condition",
                    "reason": "clear hard rollback so ops_live_gate can leave rollback_now state",
                    "priority_score": 99,
                    "priority_tier": "repair_queue_now",
                    "command": "cmd-gate",
                    "clear_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
                },
                {
                    "rank": 2,
                    "area": "ops_live_gate",
                    "target": "slot_anomaly",
                    "symbol": "SLOT_ANOMALY",
                    "action": "clear_ops_live_gate_condition",
                    "reason": "clear slot anomaly so ops_live_gate can resume",
                    "priority_score": 98,
                    "priority_tier": "repair_queue_now",
                    "command": "cmd-gate",
                    "clear_when": "clear slot anomaly so ops_live_gate can resume",
                },
                {
                    "rank": 3,
                    "area": "ops_live_gate",
                    "target": "backtest_snapshot",
                    "symbol": "BACKTEST_SNAPSHOT",
                    "action": "clear_ops_live_gate_condition",
                    "reason": "refresh backtest snapshot so ops_live_gate can resume",
                    "priority_score": 97,
                    "priority_tier": "repair_queue_now",
                    "command": "cmd-gate",
                    "clear_when": "refresh backtest snapshot so ops_live_gate can resume",
                },
                {
                    "rank": 4,
                    "area": "ops_live_gate",
                    "target": "ops_status_red",
                    "symbol": "OPS_STATUS_RED",
                    "action": "clear_ops_live_gate_condition",
                    "reason": "clear ops red status so ops_live_gate can resume",
                    "priority_score": 96,
                    "priority_tier": "repair_queue_now",
                    "command": "cmd-gate",
                    "clear_when": "clear ops red status so ops_live_gate can resume",
                },
                {
                    "rank": 5,
                    "area": "risk_guard",
                    "target": "ticket_missing:no_actionable_ticket",
                    "symbol": "TICKET_MISSING:NO_ACTIONABLE_TICKET",
                    "action": "clear_risk_guard_condition",
                    "reason": "publish at least one actionable ticket",
                    "priority_score": 89,
                    "priority_tier": "repair_queue_now",
                    "command": "cmd-risk",
                    "clear_when": "publish at least one actionable ticket",
                },
                {
                    "rank": 6,
                    "area": "risk_guard",
                    "target": "panic_cooldown_active",
                    "symbol": "PANIC_COOLDOWN_ACTIVE",
                    "action": "clear_risk_guard_condition",
                    "reason": "allow panic cooldown to expire cleanly",
                    "priority_score": 88,
                    "priority_tier": "repair_queue_now",
                    "command": "cmd-risk",
                    "clear_when": "allow panic cooldown to expire cleanly",
                },
                {
                    "rank": 7,
                    "area": "risk_guard",
                    "target": "open_exposure_above_cap",
                    "symbol": "OPEN_EXPOSURE_ABOVE_CAP",
                    "action": "clear_risk_guard_condition",
                    "reason": "reduce open exposure below configured cap",
                    "priority_score": 87,
                    "priority_tier": "repair_queue_now",
                    "command": "cmd-risk",
                    "clear_when": "reduce open exposure below configured cap",
                },
            ],
            "operator_repair_checklist_brief": (
                "1:repair:ROLLBACK_HARD:clear_ops_live_gate_condition"
                " | 2:repair:SLOT_ANOMALY:clear_ops_live_gate_condition"
                " | 3:repair:BACKTEST_SNAPSHOT:clear_ops_live_gate_condition"
                " | 4:repair:OPS_STATUS_RED:clear_ops_live_gate_condition | +3"
            ),
            "operator_repair_checklist": [
                {
                    "rank": 1,
                    "state": "repair",
                    "symbol": "ROLLBACK_HARD",
                    "action": "clear_ops_live_gate_condition",
                    "blocker_detail": "clear hard rollback so ops_live_gate can leave rollback_now state",
                    "done_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
                }
            ],
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
                {
                    "rank": 4,
                    "state": "repair",
                    "symbol": "ROLLBACK_HARD",
                    "action": "clear_ops_live_gate_condition",
                    "blocker_detail": "Current local operator head is XAUUSD (commodity_execution_close_evidence/wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote automated live remains profitability_confirmed_but_auto_live_blocked.",
                    "done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
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
            "secondary_focus_priority_tier": "watch_queue_only",
            "secondary_focus_priority_score": 20,
            "secondary_focus_queue_rank": 2,
            "crypto_route_shortline_market_state_brief": "Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade",
            "crypto_route_shortline_execution_gate_brief": "4h_profile_location -> liquidity_sweep -> 1m_5m_mss_or_choch -> 15m_cvd_divergence_or_confirmation -> fvg_ob_breaker_retest -> 15m_reversal_or_breakout_candle",
            "crypto_route_shortline_no_trade_rule": "no_sweep_no_mss_no_cvd_no_trade",
            "crypto_route_shortline_session_map_brief": "asia_high_low, london_high_low, prior_day_high_low, equal_highs_lows",
            "crypto_route_shortline_cvd_semantic_status": "ok",
            "crypto_route_shortline_cvd_semantic_takeaway": "All current CVD-lite observations are downgraded to watch-only; keep them as review filters until micro quality recovers.",
            "crypto_route_shortline_cvd_queue_handoff_status": "queue-watch-only",
            "crypto_route_shortline_cvd_queue_handoff_takeaway": "Queue priorities remain valid, but the latest micro snapshot downgrades all overlapping crypto symbols to watch-only.",
            "crypto_route_shortline_cvd_queue_focus_batch": "crypto_hot",
            "crypto_route_shortline_cvd_queue_focus_action": "defer_until_micro_recovers",
            "crypto_route_shortline_cvd_queue_stack_brief": "crypto_hot -> crypto_majors",
            "crypto_route_focus_execution_state": "Bias_Only",
            "crypto_route_focus_execution_blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade.",
            "crypto_route_focus_execution_done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
            "crypto_route_focus_execution_micro_classification": "watch_only",
            "crypto_route_focus_execution_micro_context": "failed_auction",
            "crypto_route_focus_execution_micro_trust_tier": "single_exchange_low",
            "crypto_route_focus_execution_micro_veto": "low_sample_or_gap_risk",
            "crypto_route_focus_execution_micro_reasons": ["time_sync_risk", "trust_low", "low_sample_or_gap_risk"],
            "crypto_route_focus_review_status": "review_no_edge_bias_only_micro_veto",
            "crypto_route_focus_review_brief": "review_no_edge_bias_only_micro_veto:SOLUSDT",
            "crypto_route_focus_review_primary_blocker": "no_edge",
            "crypto_route_focus_review_micro_blocker": "low_sample_or_gap_risk",
            "crypto_route_focus_review_blocker_detail": "SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade.",
            "crypto_route_focus_review_done_when": "SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow",
            "crypto_route_focus_review_score_status": "scored",
            "crypto_route_focus_review_edge_score": 5,
            "crypto_route_focus_review_structure_score": 25,
            "crypto_route_focus_review_micro_score": 20,
            "crypto_route_focus_review_composite_score": 17,
            "crypto_route_focus_review_score_brief": "scored:SOLUSDT:edge=5|structure=25|micro=20|composite=17",
            "crypto_route_focus_review_priority_status": "ready",
            "crypto_route_focus_review_priority_score": 17,
            "crypto_route_focus_review_priority_tier": "deprioritized_review",
            "crypto_route_focus_review_priority_brief": "deprioritized_review:17/100",
            "crypto_route_review_priority_queue_status": "ready",
            "crypto_route_review_priority_queue_count": 2,
            "crypto_route_review_priority_queue_brief": "1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25",
            "crypto_route_review_priority_head_symbol": "SOLUSDT",
            "crypto_route_review_priority_head_tier": "review_queue_now",
            "crypto_route_review_priority_head_score": 73,
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
    assert "Secondary focus priority is: `tier=watch_queue_only | score=20 | queue_rank=2`" in text
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
    assert "Crypto route alignment is: `area=crypto_route | slot=secondary | symbol=BNBUSDT | action=watch_priority_until_long_window_confirms | status=route_ahead_of_embedding | brief=route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta" in text
    assert "Crypto route alignment recovery outcome is: `status=recovery_completed_no_edge | brief=recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta | failed=0 | timed_out=0 | zero_trade_batches=crypto_hot, crypto_majors, crypto_beta" in text
    assert "Crypto route alignment cooldown is: `status=cooldown_active_wait_for_new_market_data | brief=cooldown_active_wait_for_new_market_data:>2026-03-11 | last_end=2026-03-11 | next_eligible=2026-03-12 | blocker=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | done_when=hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "Crypto route alignment recovery recipe gate is: `status=deferred_by_cooldown | brief=deferred_by_cooldown:2026-03-12 | ready_on=2026-03-12 | blocker=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | done_when=hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "Crypto route alignment recovery template is: `script=/tmp/run_hot_universe_research.py | expected_status=ok | window_days=21 | target_batches=crypto_hot, crypto_majors, crypto_beta" in text
    assert "Crypto shortline gate is: `market=Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade | focus_state=Bias_Only | micro=watch_only:failed_auction:low_sample_or_gap_risk | cvd_queue=queue-watch-only:crypto_hot:defer_until_micro_recovers`" in text
    assert "Crypto review priority is: `tier=deprioritized_review | score=17 | brief=deprioritized_review:17/100`" in text
    assert "Crypto review queue is: `status=ready | count=2 | brief=1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25 | head=SOLUSDT:review_queue_now:73`" in text
    assert "Source refresh queue is: `-`" in text
    assert "Crypto route head source refresh is: `status=ready | brief=ready:BNBUSDT:read_current_artifact | slot=secondary | symbol=BNBUSDT | action=read_current_artifact | kind=crypto_route | health=ready`" in text
    assert "Crypto route head source refresh gate is: `blocker=BNBUSDT currently uses a readable crypto_route artifact (ok, fresh) | done_when=keep BNBUSDT on the current crypto_route artifact while it remains usable`" in text
    assert "Source refresh pipeline pending is: `-`" in text
    assert "Source refresh pipeline relevance is: `status=non_blocking_for_current_crypto_head | brief=non_blocking_for_current_crypto_head:BNBUSDT:3 | blocker=BNBUSDT current source lane is already ready via read_current_artifact; remaining source refresh pipeline work is broader carry-over and does not block the current crypto head. | done_when=run the remaining source refresh pipeline only when broader refresh freshness is required or the crypto review queue head changes`" in text
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
    assert "Action checklist is: `1:waiting:XAUUSD:wait_for_paper_execution_close_evidence | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms | 4:repair:ROLLBACK_HARD:clear_ops_live_gate_condition`" in text
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
    assert "- `secondary_focus_priority_tier = watch_queue_only`" in text
    assert "- `secondary_focus_priority_score = 20`" in text
    assert "- `secondary_focus_queue_rank = 2`" in text
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
    assert "- `operator_crypto_route_alignment_focus_area = crypto_route`" in text
    assert "- `operator_crypto_route_alignment_focus_slot = secondary`" in text
    assert "- `operator_crypto_route_alignment_focus_symbol = BNBUSDT`" in text
    assert "- `operator_crypto_route_alignment_focus_action = watch_priority_until_long_window_confirms`" in text
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
    assert "- `operator_source_refresh_pipeline_relevance_status = non_blocking_for_current_crypto_head`" in text
    assert "- `operator_source_refresh_pipeline_relevance_brief = non_blocking_for_current_crypto_head:BNBUSDT:3`" in text
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
    assert "- `crypto_route_head_source_refresh_status = ready`" in text
    assert "- `crypto_route_head_source_refresh_brief = ready:BNBUSDT:read_current_artifact`" in text
    assert "- `crypto_route_head_source_refresh_slot = secondary`" in text
    assert "- `crypto_route_head_source_refresh_symbol = BNBUSDT`" in text
    assert "- `crypto_route_head_source_refresh_action = read_current_artifact`" in text
    assert "- `crypto_route_head_source_refresh_source_kind = crypto_route`" in text
    assert "- `crypto_route_head_source_refresh_source_health = ready`" in text
    assert "- `crypto_route_head_source_refresh_source_artifact = /tmp/crypto.json`" in text
    assert "- `crypto_route_head_source_refresh_blocker_detail = BNBUSDT currently uses a readable crypto_route artifact (ok, fresh)`" in text
    assert "- `crypto_route_head_source_refresh_done_when = keep BNBUSDT on the current crypto_route artifact while it remains usable`" in text
    assert "- `source_crypto_route_refresh_artifact = /tmp/crypto-route-refresh.json`" in text
    assert "- `source_crypto_route_refresh_status = ok`" in text
    assert "- `source_crypto_route_refresh_as_of = 2026-03-11T08:40:09+00:00`" in text
    assert "- `source_crypto_route_refresh_native_mode = skip_native_refresh`" in text
    assert "- `source_crypto_route_refresh_native_step_count = 9`" in text
    assert "- `source_crypto_route_refresh_reused_native_count = 9`" in text
    assert "- `source_crypto_route_refresh_missing_reused_count = 0`" in text
    assert "- `source_crypto_route_refresh_reuse_status = reused_native_inputs`" in text
    assert "- `source_crypto_route_refresh_reuse_brief = reused_native_inputs:skip_native_refresh:9/9`" in text
    assert "- `source_crypto_route_refresh_reuse_note = crypto_route_refresh is currently reusing 9/9 native inputs via skip_native_refresh`" in text
    assert "- `source_crypto_route_refresh_reuse_done_when = run full native refresh only when fresh native recomputation is required`" in text
    assert "- `source_crypto_route_refresh_reuse_level = informational`" in text
    assert "- `source_crypto_route_refresh_reuse_gate_status = reuse_non_blocking`" in text
    assert "- `source_crypto_route_refresh_reuse_gate_brief = reuse_non_blocking:skip_native_refresh:9/9`" in text
    assert "- `source_crypto_route_refresh_reuse_gate_blocking = false`" in text
    assert "- `source_crypto_route_refresh_reuse_gate_blocker_detail = all tracked native steps were intentionally reused; current refresh remains safe for downstream consumption.`" in text
    assert "- `source_crypto_route_refresh_reuse_gate_done_when = run full native refresh only when fresh native recomputation is explicitly required`" in text
    assert "- `source_remote_live_handoff_artifact = /tmp/remote-live-handoff.json`" in text
    assert "- `source_remote_live_handoff_status = ok`" in text
    assert "- `source_remote_live_handoff_ready_check_scope_brief = portfolio_margin_um:portfolio_margin_um`" in text
    assert "- `source_remote_live_handoff_account_scope_alignment_brief = split_scope_spot_vs_portfolio_margin_um`" in text
    assert "- `source_live_gate_blocker_remote_live_diagnosis_brief = profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard`" in text
    assert "## Remote Live History Audit" in text
    assert "- status=`ok` as_of=`2026-03-13T01:20:00Z` market=`portfolio_margin_um` path=`/tmp/remote_live_history_audit.json`" in text
    assert "- windows=`24h:12.5pnl/3tr/1open | 7d:4.5pnl/5tr/0open | 30d:15.25pnl/9tr/0open`" in text
    assert "- risk_guard=`blocked | ticket_missing:no_actionable_ticket, panic_cooldown_active`" in text
    assert "- pnl_30d_by_symbol=`BTCUSDT:8.5, ETHUSDT:1.25`" in text
    assert "## Remote Live Diagnosis" in text
    assert "- status=`profitability_confirmed_but_auto_live_blocked` brief=`profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard` path=`/tmp/live_gate_blocker_report.json`" in text
    assert "- live_decision=`do_not_start_formal_live` as_of=`2026-03-13T01:22:00Z`" in text
    assert "## Remote Live Operator Alignment" in text
    assert "- status=`local_operator_active_remote_live_blocked` brief=`local_operator_active_remote_live_blocked:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked`" in text
    assert "- blocker=`Current local operator head is XAUUSD (commodity_execution_close_evidence/wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote automated live remains profitability_confirmed_but_auto_live_blocked.`" in text
    assert "## Cross-Market Remote Live Takeover Gate" in text
    assert "- status=`blocked_by_remote_live_gate` brief=`blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked`" in text
    assert "- blocker=`Current local operator head is XAUUSD (commodity_execution_close_evidence/wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote automated live remains profitability_confirmed_but_auto_live_blocked.`" in text
    assert "## Cross-Market Remote Live Takeover Clearing" in text
    assert "- status=`clearing_required` brief=`clearing_required:ops_live_gate+risk_guard`" in text
    assert "- blocker=`ops_live_gate needs rollback_hard, risk_violations, max_drawdown, slot_anomaly; risk_guard needs ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap`" in text
    assert "## Remote Live Takeover Repair Queue" in text
    assert "- status=`ready` brief=`ready:ops_live_gate:rollback_hard:99` count=`7`" in text
    assert "- head=`ops_live_gate | rollback_hard | clear_ops_live_gate_condition | priority=99/repair_queue_now`" in text
    assert "- head_command=`cmd-gate`" in text
    assert "## Operator Repair Queue" in text
    assert "- queue=`7 | 1:ops_live_gate:rollback_hard:clear_ops_live_gate_condition | 2:ops_live_gate:slot_anomaly:clear_ops_live_gate_condition | 3:ops_live_gate:backtest_snapshot:clear_ops_live_gate_condition | 4:ops_live_gate:ops_status_red:clear_ops_live_gate_condition | +3`" in text
    assert "- checklist=`1:repair:ROLLBACK_HARD:clear_ops_live_gate_condition | 2:repair:SLOT_ANOMALY:clear_ops_live_gate_condition | 3:repair:BACKTEST_SNAPSHOT:clear_ops_live_gate_condition | 4:repair:OPS_STATUS_RED:clear_ops_live_gate_condition | +3`" in text
    assert "## Cross-Market Operator Repair Head Lane" in text
    assert "- status=`ready` brief=`ready:ops_live_gate:rollback_hard:99`" in text
    assert "- head=`ops_live_gate | ROLLBACK_HARD | clear_ops_live_gate_condition | priority_score=99 | priority_tier=repair_queue_now`" in text
    assert "- `remote_live_takeover_repair_queue_queue_brief = 1:ops_live_gate:rollback_hard:99 | 2:ops_live_gate:slot_anomaly:98 | 3:ops_live_gate:backtest_snapshot:97 | 4:ops_live_gate:ops_status_red:96 | 5:risk_guard:ticket_missing:no_actionable_ticket:89 | 6:risk_guard:panic_cooldown_active:88 | 7:risk_guard:open_exposure_above_cap:87`" in text
    assert "- `cross_market_operator_repair_head_status = ready`" in text
    assert "- `cross_market_operator_repair_head_brief = ready:ops_live_gate:rollback_hard:99`" in text
    assert "- `cross_market_operator_repair_backlog_status = ready`" in text
    assert "- `cross_market_operator_repair_backlog_count = 7`" in text
    assert "- `cross_market_operator_repair_backlog_priority_total = 654`" in text
    assert "- `cross_market_remote_live_takeover_gate_status = blocked_by_remote_live_gate`" in text
    assert "- `cross_market_remote_live_takeover_gate_brief = blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked`" in text
    assert "- `cross_market_remote_live_takeover_clearing_status = clearing_required`" in text
    assert "- `cross_market_remote_live_takeover_clearing_brief = clearing_required:ops_live_gate+risk_guard`" in text
    assert "- `remote_live_takeover_repair_queue_status = ready`" in text
    assert "- `remote_live_takeover_repair_queue_brief = ready:ops_live_gate:rollback_hard:99`" in text
    assert "- `operator_repair_queue_brief = 1:ops_live_gate:rollback_hard:clear_ops_live_gate_condition | 2:ops_live_gate:slot_anomaly:clear_ops_live_gate_condition | 3:ops_live_gate:backtest_snapshot:clear_ops_live_gate_condition | 4:ops_live_gate:ops_status_red:clear_ops_live_gate_condition | +3`" in text
    assert "- `operator_repair_queue_count = 7`" in text
    assert "- `operator_repair_checklist_brief = 1:repair:ROLLBACK_HARD:clear_ops_live_gate_condition | 2:repair:SLOT_ANOMALY:clear_ops_live_gate_condition | 3:repair:BACKTEST_SNAPSHOT:clear_ops_live_gate_condition | 4:repair:OPS_STATUS_RED:clear_ops_live_gate_condition | +3`" in text
    assert "## Remote Live Account Scope" in text
    assert "- ready_scope=`portfolio_margin_um:portfolio_margin_um` market=`portfolio_margin_um`" in text
    assert "- alignment=`split_scope_spot_vs_portfolio_margin_um | blocking=False`" in text
    assert "- `operator_focus_slots_brief = primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence | followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | secondary:watch:BNBUSDT:watch_priority_until_long_window_confirms`" in text
    assert "- `operator_action_queue_brief = 1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence | 3:crypto_route:BNBUSDT:watch_priority_until_long_window_confirms | 4:ops_live_gate:rollback_hard:clear_ops_live_gate_condition`" in text
    assert "- `operator_action_checklist_brief = 1:waiting:XAUUSD:wait_for_paper_execution_close_evidence | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms | 4:repair:ROLLBACK_HARD:clear_ops_live_gate_condition`" in text
    assert "- Remote repair checklist is: `1:repair:ROLLBACK_HARD:clear_ops_live_gate_condition | 2:repair:SLOT_ANOMALY:clear_ops_live_gate_condition | 3:repair:BACKTEST_SNAPSHOT:clear_ops_live_gate_condition | 4:repair:OPS_STATUS_RED:clear_ops_live_gate_condition | +3`" in text
    assert "- `crypto_route_focus_review_priority_status = ready`" in text
    assert "- `crypto_route_focus_review_priority_score = 17`" in text
    assert "- `crypto_route_focus_review_priority_tier = deprioritized_review`" in text
    assert "- `crypto_route_focus_review_priority_brief = deprioritized_review:17/100`" in text
    assert "- `crypto_route_review_priority_queue_status = ready`" in text
    assert "- `crypto_route_review_priority_queue_count = 2`" in text
    assert "- `crypto_route_review_priority_queue_brief = 1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25`" in text
    assert "- `crypto_route_review_priority_head_symbol = SOLUSDT`" in text
    assert "- `crypto_route_review_priority_head_tier = review_queue_now`" in text
    assert "- `crypto_route_review_priority_head_score = 73`" in text
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
    assert "- action queue: `1:commodity_execution_close_evidence:commodity-paper-execution:metals_all:XAUUSD:wait_for_paper_execution_close_evidence | 2:commodity_fill_evidence:commodity-paper-execution:metals_all:XAGUSD:wait_for_paper_execution_fill_evidence | 3:crypto_route:BNBUSDT:watch_priority_until_long_window_confirms | 4:ops_live_gate:rollback_hard:clear_ops_live_gate_condition`" in text
    assert "- action checklist: `1:waiting:XAUUSD:wait_for_paper_execution_close_evidence | 2:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | 3:watch:BNBUSDT:watch_priority_until_long_window_confirms | 4:repair:ROLLBACK_HARD:clear_ops_live_gate_condition`" in text
    assert "- focus slot refresh backlog: `-`" in text
    assert "- focus slot promotion gate: `promotion_ready:3/3`" in text
    assert "- focus slot actionability gate: `actionability_guarded_by_content:2/3`" in text
    assert "- focus slot readiness gate: `readiness_guarded_by_content:2/3`" in text
    assert "- research embedding quality: `avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- crypto route alignment area: `crypto_route`" in text
    assert "- crypto route alignment slot: `secondary`" in text
    assert "- crypto route alignment symbol: `BNBUSDT`" in text
    assert "- crypto route alignment action: `watch_priority_until_long_window_confirms`" in text
    assert "- crypto route alignment: `route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- crypto route alignment recovery outcome: `recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- crypto route alignment cooldown: `cooldown_active_wait_for_new_market_data:>2026-03-11`" in text
    assert "- crypto route alignment recovery recipe gate: `deferred_by_cooldown:2026-03-12`" in text
    assert "- crypto route alignment recovery: `crypto_hot, crypto_majors, crypto_beta@21d`" in text
    assert "- crypto shortline market state: `Bias_Only->Setup_Ready | no_trade=no_sweep_no_mss_no_cvd_no_trade`" in text
    assert "- crypto shortline cvd queue: `queue-watch-only | crypto_hot | defer_until_micro_recovers | crypto_hot -> crypto_majors`" in text
    assert "- crypto shortline micro gate: `watch_only | failed_auction | single_exchange_low | low_sample_or_gap_risk | time_sync_risk, trust_low, low_sample_or_gap_risk`" in text
    assert "- crypto review lane: `review_no_edge_bias_only_micro_veto | no_edge | low_sample_or_gap_risk | SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow`" in text
    assert "- crypto review scores: `edge=5 | structure=25 | micro=20 | composite=17`" in text
    assert "- crypto review priority: `tier=deprioritized_review | score=17 | brief=deprioritized_review:17/100`" in text
    assert "- crypto review queue: `status=ready | count=2 | brief=1:SOLUSDT:review_queue_now:73 | 2:BNBUSDT:watch_queue_only:25 | head=SOLUSDT:review_queue_now:73`" in text
    assert "- crypto head source refresh: `ready:BNBUSDT:read_current_artifact`" in text
    assert "- crypto route refresh audit: `reused_native_inputs:skip_native_refresh:9/9 | mode=skip_native_refresh | reused=9/9 | path=/tmp/crypto-route-refresh.json`" in text
    assert "- crypto route refresh reuse gate: `reuse_non_blocking:skip_native_refresh:9/9 | level=informational | blocking=false | path=/tmp/crypto-route-refresh.json`" in text
    assert "## Crypto Shortline Gate" in text
    assert "- cvd semantic: `ok | All current CVD-lite observations are downgraded to watch-only; keep them as review filters until micro quality recovers.`" in text
    assert "- focus execution gate: `Bias_Only | SOLUSDT remains Bias_Only; missing liquidity_sweep, mss, fvg_ob_breaker_retest, cvd_confirmation, route_state=watch:deprioritize_flow; no_sweep_no_mss_no_cvd_no_trade. | SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow`" in text
    assert "- review lane: `review_no_edge_bias_only_micro_veto | no_edge | low_sample_or_gap_risk | SOLUSDT completes liquidity_sweep and mss and fvg_ob_breaker_retest and cvd_confirmation and route_state=watch:deprioritize_flow`" in text
    assert "- review scores: `edge=5 | structure=25 | micro=20 | composite=17`" in text
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
    assert "- area=`crypto_route` slot=`secondary` symbol=`BNBUSDT` action=`watch_priority_until_long_window_confirms` status=`route_ahead_of_embedding` brief=`route_ahead_of_embedding:BNBUSDT:avoid_only:crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- recovery_outcome=`recovery_completed_no_edge | recovery_completed_no_edge:crypto_hot, crypto_majors, crypto_beta | failed=0 | timed_out=0 | zero_trade_batches=crypto_hot, crypto_majors, crypto_beta`" in text
    assert "- cooldown=`cooldown_active_wait_for_new_market_data | cooldown_active_wait_for_new_market_data:>2026-03-11 | last_end=2026-03-11 | next_eligible=2026-03-12 | blocker=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | done_when=hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "- recovery_recipe_gate=`deferred_by_cooldown | deferred_by_cooldown:2026-03-12 | ready_on=2026-03-12 | blocker=latest clean crypto recovery already evaluated data through 2026-03-11 and still found no edge; rerunning before a later end date is unlikely to change the outcome | done_when=hot_universe_research end date advances beyond 2026-03-11 or crypto_route focus changes`" in text
    assert "- recovery=`crypto_hot, crypto_majors, crypto_beta@21d`" in text
    assert "- recovery_script=`/tmp/run_hot_universe_research.py`" in text
    assert "## Crypto Route Head Source Refresh" in text
    assert "- status=`ready` brief=`ready:BNBUSDT:read_current_artifact` slot=`secondary` symbol=`BNBUSDT` action=`read_current_artifact`" in text
    assert "- source=`crypto_route | ready | /tmp/crypto.json`" in text
    assert "- blocker=`BNBUSDT currently uses a readable crypto_route artifact (ok, fresh)`" in text
    assert "- done_when=`keep BNBUSDT on the current crypto_route artifact while it remains usable`" in text
    assert "- recipe=`- | - | -@-`" in text
    assert "## Crypto Route Refresh Audit" in text
    assert "- status=`ok` as_of=`2026-03-11T08:40:09+00:00` path=`/tmp/crypto-route-refresh.json`" in text
    assert "- native_mode=`skip_native_refresh` reuse=`reused_native_inputs | reused_native_inputs:skip_native_refresh:9/9`" in text
    assert "- counts=`reused=9 | missing=0 | native_steps=9`" in text
    assert "- note=`crypto_route_refresh is currently reusing 9/9 native inputs via skip_native_refresh`" in text
    assert "- done_when=`run full native refresh only when fresh native recomputation is required`" in text
    assert "- reuse_gate=`reuse_non_blocking | reuse_non_blocking:skip_native_refresh:9/9 | level=informational | blocking=false`" in text
    assert "- reuse_gate_blocker=`all tracked native steps were intentionally reused; current refresh remains safe for downstream consumption.`" in text
    assert "- reuse_gate_done_when=`run full native refresh only when fresh native recomputation is explicitly required`" in text
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


def test_render_context_markdown_surfaces_brooks_structure_route() -> None:
    mod = _load_module()
    text = mod.render_context_markdown(
        runtime_now=mod.parse_now("2026-03-13T04:20:00Z"),
        brief={
            "artifact": "/tmp/brief.json",
            "operator_status": "commodity-paper-execution-close-evidence-pending-plus-crypto-review",
            "operator_stack_brief": "commodity:close-evidence:XAUUSD | crypto:SOLUSDT:deprioritize_flow",
            "source_brooks_route_report_artifact": "/tmp/brooks-route.json",
            "source_brooks_route_report_status": "ok",
            "source_brooks_route_report_as_of": "2026-03-13T04:15:00+00:00",
            "source_brooks_route_report_selected_routes_brief": (
                "equity:three_push_climax_reversal | etf:breakout_pullback_resume | future:breakout_pullback_resume/second_entry_trend_continuation"
            ),
            "source_brooks_route_report_candidate_count": 1,
            "source_brooks_route_report_head_symbol": "SC2603",
            "source_brooks_route_report_head_strategy_id": "second_entry_trend_continuation",
            "source_brooks_route_report_head_direction": "LONG",
            "source_brooks_route_report_head_bridge_status": "manual_structure_route",
            "source_brooks_route_report_head_blocker_detail": "manual structure lane only",
            "source_brooks_execution_plan_artifact": "/tmp/brooks-plan.json",
            "source_brooks_execution_plan_status": "ok",
            "source_brooks_execution_plan_as_of": "2026-03-13T04:20:00+00:00",
            "source_brooks_execution_plan_actionable_count": 1,
            "source_brooks_execution_plan_blocked_count": 0,
            "source_brooks_execution_plan_head_symbol": "SC2603",
            "source_brooks_execution_plan_head_strategy_id": "second_entry_trend_continuation",
            "source_brooks_execution_plan_head_plan_status": "manual_structure_review_now",
            "source_brooks_execution_plan_head_execution_action": "review_manual_stop_entry",
            "source_brooks_execution_plan_head_entry_price": 478.8,
            "source_brooks_execution_plan_head_stop_price": 471.0336,
            "source_brooks_execution_plan_head_target_price": 495.8861,
            "source_brooks_execution_plan_head_rr_ratio": 2.2,
            "source_brooks_execution_plan_head_blocker_detail": (
                "Structure route is valid, but this asset class has no automated execution bridge in-system."
            ),
            "source_brooks_structure_review_queue_artifact": (
                "/tmp/review/20260313T042100Z_brooks_structure_review_queue.json"
            ),
            "source_brooks_structure_review_queue_status": "ok",
            "source_brooks_structure_review_queue_as_of": "2026-03-13T04:21:00+00:00",
            "source_brooks_structure_review_queue_brief": "ready:SC2603:96:review_queue_now",
            "source_brooks_structure_refresh_artifact": (
                "/tmp/review/20260313T042200Z_brooks_structure_refresh.json"
            ),
            "source_brooks_structure_refresh_status": "ok",
            "source_brooks_structure_refresh_as_of": "2026-03-13T04:22:00+00:00",
            "source_brooks_structure_refresh_brief": (
                "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now"
            ),
            "source_brooks_structure_refresh_queue_count": 2,
            "source_brooks_structure_refresh_head_symbol": "SC2603",
            "source_brooks_structure_refresh_head_action": "review_manual_stop_entry",
            "source_brooks_structure_refresh_head_priority_score": 96,
            "source_cross_market_operator_state_artifact": (
                "/tmp/review/20260313T042300Z_cross_market_operator_state.json"
            ),
            "source_cross_market_operator_state_status": "ok",
            "source_cross_market_operator_state_as_of": "2026-03-13T04:23:00+00:00",
            "source_cross_market_operator_state_operator_backlog_status": "ready",
            "source_cross_market_operator_state_operator_backlog_count": 4,
            "source_cross_market_operator_state_operator_backlog_brief": (
                "1:waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
                "2:waiting:commodity_fill_evidence:XAGUSD:wait_for_paper_execution_fill_evidence:98 | "
                "3:review:brooks_structure:SC2603:review_manual_stop_entry:96 | "
                "4:review:crypto_route:SOLUSDT:deprioritize_flow:73"
            ),
            "source_cross_market_operator_state_operator_backlog_state_brief": "waiting=2 | review=2 | watch=0 | blocked=0",
            "source_cross_market_operator_state_operator_backlog_priority_totals_brief": "waiting=197 | review=169 | watch=0 | blocked=0",
            "source_cross_market_operator_state_remote_live_takeover_gate_status": "blocked_by_remote_live_gate",
            "source_cross_market_operator_state_remote_live_takeover_gate_brief": (
                "blocked_by_remote_live_gate:commodity_execution_close_evidence:XAUUSD:profitability_confirmed_but_auto_live_blocked"
            ),
            "source_cross_market_operator_state_remote_live_takeover_gate_blocker_detail": (
                "Current local operator head is XAUUSD (commodity_execution_close_evidence/"
                "wait_for_paper_execution_close_evidence, state=waiting, priority=99), while remote "
                "automated live remains profitability_confirmed_but_auto_live_blocked."
            ),
            "source_cross_market_operator_state_remote_live_takeover_gate_done_when": (
                "XAUUSD local operator head progresses and remote auto-live blockers are cleared"
            ),
            "source_cross_market_operator_state_operator_head_area": "commodity_execution_close_evidence",
            "source_cross_market_operator_state_operator_head_symbol": "XAUUSD",
            "source_cross_market_operator_state_operator_head_action": "wait_for_paper_execution_close_evidence",
            "source_cross_market_operator_state_operator_head_state": "waiting",
            "source_cross_market_operator_state_operator_head_priority_score": 99,
            "source_cross_market_operator_state_operator_head_priority_tier": "waiting_now",
            "source_cross_market_operator_state_review_backlog_status": "ready",
            "source_cross_market_operator_state_review_backlog_count": 2,
            "source_cross_market_operator_state_review_backlog_brief": (
                "1:brooks_structure:SC2603:review_queue_now:96 | 2:crypto_route:SOLUSDT:review_queue_now:73"
            ),
            "source_cross_market_operator_state_review_head_area": "brooks_structure",
            "source_cross_market_operator_state_review_head_symbol": "SC2603",
            "source_cross_market_operator_state_review_head_action": "review_manual_stop_entry",
            "source_cross_market_operator_state_review_head_priority_score": 96,
            "source_cross_market_operator_state_review_head_priority_tier": "review_queue_now",
            "cross_market_review_head_status": "ready",
            "cross_market_review_head_brief": (
                "ready:brooks_structure:SC2603:review_manual_stop_entry:96"
            ),
            "cross_market_review_head_area": "brooks_structure",
            "cross_market_review_head_symbol": "SC2603",
            "cross_market_review_head_action": "review_manual_stop_entry",
            "cross_market_review_head_priority_score": 96,
            "cross_market_review_head_priority_tier": "review_queue_now",
            "cross_market_review_head_blocker_detail": (
                "manual-only structure route still requires venue confirmation"
            ),
            "cross_market_review_head_done_when": (
                "cross-market review head is executed manually, promoted, or invalidated"
            ),
            "source_system_time_sync_repair_plan_artifact": "/tmp/review/20260313T042310Z_system_time_sync_repair_plan.json",
            "source_system_time_sync_repair_plan_status": "run_now",
            "source_system_time_sync_repair_plan_brief": "manual_time_repair_required:SC2603:timed_apns_fallback",
            "source_system_time_sync_repair_plan_done_when": "restore direct NTP path and rerun time_sync_probe",
            "source_system_time_sync_repair_plan_admin_required": True,
            "source_system_time_sync_repair_verification_artifact": "/tmp/review/20260313T042311Z_system_time_sync_repair_verification_report.json",
            "source_system_time_sync_repair_verification_status": "blocked",
            "source_system_time_sync_repair_verification_brief": "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked",
            "source_system_time_sync_repair_verification_cleared": False,
            "source_openclaw_orderflow_blueprint_artifact": "/tmp/review/20260313T042312Z_openclaw_orderflow_blueprint.json",
            "source_openclaw_orderflow_blueprint_status": "ok",
            "source_openclaw_orderflow_blueprint_brief": (
                "guarded_remote_guardian -> orderflow_native_execution_organism | "
                "P1:remote_execution_identity_state:Split remote identity and scope state"
            ),
            "source_openclaw_orderflow_blueprint_current_life_stage": "guarded_remote_guardian",
            "source_openclaw_orderflow_blueprint_target_life_stage": "orderflow_native_execution_organism",
            "source_openclaw_orderflow_blueprint_remote_intent_queue_brief": "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um",
            "source_openclaw_orderflow_blueprint_remote_intent_queue_recommendation": "hold_remote_idle_until_ticket_ready",
            "source_openclaw_orderflow_blueprint_remote_execution_journal_brief": (
                "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um"
                " | blocked:ticket_missing:no_actionable_ticket | not_attempted_wait_trade_readiness"
            ),
            "source_openclaw_orderflow_blueprint_remote_orderflow_feedback_brief": (
                "downrank_guardian_blocked_route:SC2603:queue_aging_high:ticket_missing:no_actionable_ticket"
            ),
            "source_openclaw_orderflow_blueprint_remote_orderflow_policy_brief": (
                "shadow_policy_blocked:SC2603:queued_wait_trade_readiness:downrank_guardian_blocked_route"
            ),
            "source_openclaw_orderflow_blueprint_remote_execution_ack_brief": (
                "shadow_no_send_ack_recorded:SC2603:not_sent_policy_blocked:no_fill_execution_not_attempted"
            ),
            "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_brief": (
                "shadow_learning_continuity_stable:SC2603:shadow_feedback_alive:portfolio_margin_um"
            ),
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_code": (
                "timed_ntp_via_fake_ip_clearance"
            ),
            "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_code": (
                "timed_ntp_via_fake_ip_clearance"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief": (
                "local_time_sync_primary_blocker_shadow_ready:SC2603:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um"
            ),
            "source_openclaw_orderflow_blueprint_top_backlog_title": (
                "Repair local time sync to unlock guarded canary review"
            ),
            "source_openclaw_orderflow_blueprint_top_backlog_target_artifact": (
                "system_time_sync_repair_verification_report"
            ),
            "source_openclaw_orderflow_blueprint_top_backlog_why": (
                "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
            ),
            "cross_market_review_backlog_count": 1,
            "cross_market_review_backlog_brief": "2:crypto_route:SOLUSDT:review_queue_now:73",
            "cross_market_operator_head_status": "waiting",
            "cross_market_operator_head_brief": (
                "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99"
            ),
            "cross_market_operator_head_area": "commodity_execution_close_evidence",
            "cross_market_operator_head_symbol": "XAUUSD",
            "cross_market_operator_head_action": "wait_for_paper_execution_close_evidence",
            "cross_market_operator_head_state": "waiting",
            "cross_market_operator_head_priority_score": 99,
            "cross_market_operator_head_priority_tier": "waiting_now",
            "cross_market_operator_head_blocker_detail": (
                "paper execution evidence is present, but position is still OPEN"
            ),
            "cross_market_operator_head_done_when": (
                "XAUUSD paper_execution_status leaves OPEN and close evidence becomes available"
            ),
            "cross_market_operator_backlog_count": 3,
            "cross_market_operator_backlog_brief": (
                "2:waiting:commodity_fill_evidence:XAGUSD:wait_for_paper_execution_fill_evidence:98 | "
                "3:review:brooks_structure:SC2603:review_manual_stop_entry:96 | "
                "4:review:crypto_route:SOLUSDT:deprioritize_flow:73"
            ),
            "cross_market_operator_backlog_state_brief": "waiting=2 | review=2 | watch=0 | blocked=0",
            "cross_market_operator_backlog_priority_totals_brief": "waiting=197 | review=169 | watch=0 | blocked=0",
            "cross_market_operator_lane_heads_brief": (
                "waiting:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
                "review:SC2603:review_manual_stop_entry:96 | "
                "watch:-:-:0 | "
                "blocked:-:-:0"
            ),
            "cross_market_operator_lane_priority_order_brief": (
                "waiting@197:2 > review@169:2 > watch@0:0 > blocked@0:0"
            ),
            "cross_market_operator_waiting_lane_status": "ready",
            "cross_market_operator_waiting_lane_count": 2,
            "cross_market_operator_waiting_lane_brief": (
                "1:XAUUSD:wait_for_paper_execution_close_evidence:99 | 2:XAGUSD:wait_for_paper_execution_fill_evidence:98"
            ),
            "cross_market_operator_waiting_lane_priority_total": 197,
            "cross_market_operator_waiting_lane_head_symbol": "XAUUSD",
            "cross_market_operator_waiting_lane_head_action": "wait_for_paper_execution_close_evidence",
            "cross_market_operator_waiting_lane_head_priority_score": 99,
            "cross_market_operator_waiting_lane_head_priority_tier": "waiting_now",
            "cross_market_operator_review_lane_status": "ready",
            "cross_market_operator_review_lane_count": 2,
            "cross_market_operator_review_lane_brief": (
                "3:SC2603:review_manual_stop_entry:96 | 4:SOLUSDT:deprioritize_flow:73"
            ),
            "cross_market_operator_review_lane_priority_total": 169,
            "cross_market_operator_review_lane_head_symbol": "SC2603",
            "cross_market_operator_review_lane_head_action": "review_manual_stop_entry",
            "cross_market_operator_review_lane_head_priority_score": 96,
            "cross_market_operator_review_lane_head_priority_tier": "review_queue_now",
            "cross_market_operator_watch_lane_status": "inactive",
            "cross_market_operator_watch_lane_count": 0,
            "cross_market_operator_watch_lane_brief": "-",
            "cross_market_operator_watch_lane_priority_total": 0,
            "cross_market_operator_watch_lane_head_symbol": "",
            "cross_market_operator_watch_lane_head_action": "",
            "cross_market_operator_watch_lane_head_priority_score": 0,
            "cross_market_operator_watch_lane_head_priority_tier": "",
            "cross_market_operator_blocked_lane_status": "inactive",
            "cross_market_operator_blocked_lane_count": 0,
            "cross_market_operator_blocked_lane_brief": "-",
            "cross_market_operator_blocked_lane_priority_total": 0,
            "cross_market_operator_blocked_lane_head_symbol": "",
            "cross_market_operator_blocked_lane_head_action": "",
            "cross_market_operator_blocked_lane_head_priority_score": 0,
            "cross_market_operator_blocked_lane_head_priority_tier": "",
            "brooks_structure_review_status": "ready",
            "brooks_structure_review_brief": (
                "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now"
            ),
            "brooks_structure_review_queue_status": "ready",
            "brooks_structure_review_queue_count": 2,
            "brooks_structure_review_queue_brief": (
                "1:SC2603:review_queue_now:manual_structure_review_now | 2:AU2406:blocked_queue:blocked_shortline_gate"
            ),
            "brooks_structure_review_priority_status": "ready",
            "brooks_structure_review_priority_brief": "ready:SC2603:96:review_queue_now",
            "brooks_structure_review_queue": [
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
            "brooks_structure_review_head_rank": 1,
            "brooks_structure_review_head_symbol": "SC2603",
            "brooks_structure_review_head_strategy_id": "second_entry_trend_continuation",
            "brooks_structure_review_head_direction": "LONG",
            "brooks_structure_review_head_tier": "review_queue_now",
            "brooks_structure_review_head_plan_status": "manual_structure_review_now",
            "brooks_structure_review_head_action": "review_manual_stop_entry",
            "brooks_structure_review_head_route_selection_score": 81.68716356888629,
            "brooks_structure_review_head_signal_score": 80,
            "brooks_structure_review_head_signal_age_bars": 2,
            "brooks_structure_review_head_priority_score": 96,
            "brooks_structure_review_head_priority_tier": "review_queue_now",
            "brooks_structure_review_head_blocker_detail": (
                "Structure route is valid, but this asset class has no automated execution bridge in-system."
            ),
            "brooks_structure_review_head_done_when": (
                "Brooks structure head is either executed manually, promoted into an automated bridge, or invalidated."
            ),
            "brooks_structure_review_blocker_detail": (
                "Structure route is valid, but this asset class has no automated execution bridge in-system."
            ),
            "brooks_structure_review_done_when": (
                "Brooks structure head is either executed manually, promoted into an automated bridge, or invalidated."
            ),
            "brooks_structure_operator_status": "ready",
            "brooks_structure_operator_brief": "ready:SC2603:review_manual_stop_entry:96",
            "brooks_structure_operator_head_symbol": "SC2603",
            "brooks_structure_operator_head_strategy_id": "second_entry_trend_continuation",
            "brooks_structure_operator_head_direction": "LONG",
            "brooks_structure_operator_head_action": "review_manual_stop_entry",
            "brooks_structure_operator_head_plan_status": "manual_structure_review_now",
            "brooks_structure_operator_head_priority_score": 96,
            "brooks_structure_operator_head_priority_tier": "review_queue_now",
            "brooks_structure_operator_backlog_count": 1,
            "brooks_structure_operator_backlog_brief": "2:AU2406:blocked_review:53",
            "brooks_structure_operator_blocker_detail": (
                "Structure route is valid, but this asset class has no automated execution bridge in-system."
            ),
            "brooks_structure_operator_done_when": (
                "manual trader confirms venue, sizing, and lower-timeframe trigger before placing a discretionary order"
            ),
        },
        review={},
        retro={},
        gap={},
        bridge={},
        bridge_apply=None,
    )

    assert "## Brooks Structure Route" in text
    assert "## Brooks Structure Review Queue" in text
    assert "SC2603" in text
    assert "AU2406" in text
    assert "manual_structure_review_now" in text
    assert "entry=478.8" in text
    assert "review_queue_now" in text
    assert "route_score=`81.687164`" in text
    assert "priority_score=`96`" in text
    assert "priority_tier=`blocked_review`" in text
    assert "queue_source=`ok | 2026-03-13T04:21:00+00:00 | /tmp/review/20260313T042100Z_brooks_structure_review_queue.json`" in text
    assert "refresh_source=`ok | 2026-03-13T04:22:00+00:00 | /tmp/review/20260313T042200Z_brooks_structure_refresh.json`" in text
    assert "refresh=`ready:SC2603:second_entry_trend_continuation:manual_structure_review_now | queue=2 | head=SC2603:review_manual_stop_entry:96`" in text
    assert "## Brooks Structure Operator Lane" in text
    assert "backlog=`1 | 2:AU2406:blocked_review:53`" in text
    assert "## Cross-Market Operator Head Lane" in text
    assert "status=`waiting` brief=`waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99`" in text
    assert "head=`commodity_execution_close_evidence | XAUUSD | wait_for_paper_execution_close_evidence | state=waiting | priority_score=99 | priority_tier=waiting_now`" in text
    assert "backlog=`3 | 2:waiting:commodity_fill_evidence:XAGUSD:wait_for_paper_execution_fill_evidence:98 | 3:review:brooks_structure:SC2603:review_manual_stop_entry:96 | 4:review:crypto_route:SOLUSDT:deprioritize_flow:73`" in text
    assert "## Cross-Market Operator Backlog" in text
    assert "backlog=`ready | count=4 | states=waiting=2 | review=2 | watch=0 | blocked=0 | totals=waiting=197 | review=169 | watch=0 | blocked=0 | heads=waiting:XAUUSD:wait_for_paper_execution_close_evidence:99 | review:SC2603:review_manual_stop_entry:96 | watch:-:-:0 | blocked:-:-:0 | order=waiting@197:2 > review@169:2 > watch@0:0 > blocked@0:0 | 1:waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99 | 2:waiting:commodity_fill_evidence:XAGUSD:wait_for_paper_execution_fill_evidence:98 | 3:review:brooks_structure:SC2603:review_manual_stop_entry:96 | 4:review:crypto_route:SOLUSDT:deprioritize_flow:73`" in text
    assert "head=`commodity_execution_close_evidence | XAUUSD | wait_for_paper_execution_close_evidence | state=waiting | priority_score=99 | priority_tier=waiting_now`" in text
    assert "## Cross-Market Operator State Lanes" in text
    assert "waiting=`ready | count=2 | total=197 | head=XAUUSD:wait_for_paper_execution_close_evidence:99:waiting_now | 1:XAUUSD:wait_for_paper_execution_close_evidence:99 | 2:XAGUSD:wait_for_paper_execution_fill_evidence:98`" in text
    assert "review=`ready | count=2 | total=169 | head=SC2603:review_manual_stop_entry:96:review_queue_now | 3:SC2603:review_manual_stop_entry:96 | 4:SOLUSDT:deprioritize_flow:73`" in text
    assert "watch=`inactive | count=0 | total=0 | head=-:-:0:- | -`" in text
    assert "blocked=`inactive | count=0 | total=0 | head=-:-:0:- | -`" in text
    assert "## Cross-Market Review Head Lane" in text
    assert "status=`ready` brief=`ready:brooks_structure:SC2603:review_manual_stop_entry:96`" in text
    assert "head=`brooks_structure | SC2603 | review_manual_stop_entry | priority_score=96 | priority_tier=review_queue_now`" in text
    assert "backlog=`1 | 2:crypto_route:SOLUSDT:review_queue_now:73`" in text
    assert "blocker=`manual-only structure route still requires venue confirmation`" in text
    assert "## Cross-Market Review Backlog" in text
    assert "source=`ok | 2026-03-13T04:23:00+00:00 | /tmp/review/20260313T042300Z_cross_market_operator_state.json`" in text
    assert "backlog=`ready | count=2 | 1:brooks_structure:SC2603:review_queue_now:96 | 2:crypto_route:SOLUSDT:review_queue_now:73`" in text
    assert "head=`brooks_structure | SC2603 | review_manual_stop_entry | priority_score=96 | priority_tier=review_queue_now`" in text
    assert "## System Time Sync Repair Plan" in text
    assert "status=`run_now` brief=`manual_time_repair_required:SC2603:timed_apns_fallback`" in text
    assert "artifact=`/tmp/review/20260313T042310Z_system_time_sync_repair_plan.json`" in text
    assert "## System Time Sync Repair Verification" in text
    assert "status=`blocked` brief=`blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked`" in text
    assert "admin_required=`True`" in text
    assert "## OpenClaw Orderflow Blueprint" in text
    assert "life_stage=`guarded_remote_guardian -> orderflow_native_execution_organism`" in text
    assert "intent_queue=`queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um`" in text
    assert "intent_recommendation=`hold_remote_idle_until_ticket_ready`" in text
    assert "execution_journal=`queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um | blocked:ticket_missing:no_actionable_ticket | not_attempted_wait_trade_readiness`" in text
    assert "orderflow_feedback=`downrank_guardian_blocked_route:SC2603:queue_aging_high:ticket_missing:no_actionable_ticket`" in text
    assert "orderflow_policy=`shadow_policy_blocked:SC2603:queued_wait_trade_readiness:downrank_guardian_blocked_route`" in text
    assert "shadow_learning_continuity=`shadow_learning_continuity_stable:SC2603:shadow_feedback_alive:portfolio_margin_um`" in text
    assert "promotion_unblock_readiness=`local_time_sync_primary_blocker_shadow_ready:SC2603:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um`" in text
    assert "backlog_top=`system_time_sync_repair_verification_report:Repair local time sync to unlock guarded canary review:blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked`" in text


def test_derive_runtime_now_advances_past_latest_artifact(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260311T084604Z_hot_universe_operator_brief.json").write_text("{}", encoding="utf-8")
    (review_dir / "20260311T084607Z_cross_market_operator_state.json").write_text("{}", encoding="utf-8")
    mod.now_utc = lambda: mod.parse_now("2026-03-11T08:46:00Z")
    derived = mod.derive_runtime_now(review_dir, "")
    assert derived.isoformat() == "2026-03-11T08:46:08+00:00"
