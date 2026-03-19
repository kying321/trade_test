from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_operator_task_visual_panel.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("operator_task_visual_panel_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_main_builds_visual_panel_and_dashboard_outputs(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    dashboard_dist = tmp_path / "dashboard-dist"
    review_dir.mkdir()
    dashboard_dist.mkdir()

    cross_market = review_dir / "20260314T001000Z_cross_market_operator_state.json"
    brooks_refresh = review_dir / "20260314T000900Z_brooks_structure_refresh.json"
    brooks_queue = review_dir / "20260314T000901Z_brooks_structure_review_queue.json"
    brooks_route = review_dir / "20260314T000902Z_brooks_price_action_route_report.json"
    brooks_plan = review_dir / "20260314T000903Z_brooks_price_action_execution_plan.json"
    crypto_refresh = review_dir / "20260314T000904Z_crypto_route_refresh.json"
    live_gate = review_dir / "20260314T000905Z_live_gate_blocker_report.json"
    remote_live_handoff = review_dir / "20260314T000906Z_remote_live_handoff.json"
    remote_live_history = review_dir / "20260314T000907Z_remote_live_history_audit.json"
    time_sync_repair_plan = review_dir / "20260314T000908Z_system_time_sync_repair_plan.json"
    time_sync_repair_verification = review_dir / "20260314T000909Z_system_time_sync_repair_verification_report.json"
    openclaw_blueprint = review_dir / "20260314T000910Z_openclaw_orderflow_blueprint.json"
    hot_brief = review_dir / "20260314T001100Z_hot_universe_operator_brief.json"

    write_json(
        cross_market,
        {
            "status": "ok",
            "as_of": "2026-03-14T00:10:00Z",
            "operator_backlog_state_counts": {"waiting": 2, "review": 3, "watch": 0, "blocked": 0, "repair": 7},
            "operator_backlog_priority_totals": {"waiting": 197, "review": 199, "watch": 0, "blocked": 0, "repair": 654},
            "operator_state_lane_priority_order_brief": "repair@654:7 > review@199:3 > waiting@197:2 > watch@0:0 > blocked@0:0",
            "operator_waiting_lane_brief": "1:XAUUSD:wait_for_paper_execution_close_evidence:99 | 2:XAGUSD:wait_for_paper_execution_fill_evidence:98",
            "operator_review_lane_brief": "3:SC2603:review_manual_stop_entry:96 | 4:SOLUSDT:deprioritize_flow:58",
            "operator_repair_lane_brief": "6:ROLLBACK_HARD:clear_ops_live_gate_condition:99 | 7:SLOT_ANOMALY:clear_ops_live_gate_condition:98",
            "operator_head_lane": {
                "status": "waiting",
                "brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
                "head": {
                    "symbol": "XAUUSD",
                    "action": "wait_for_paper_execution_close_evidence",
                    "blocker_detail": "waiting for close evidence",
                    "done_when": "close evidence appears",
                },
            },
            "review_head_lane": {
                "status": "ready",
                "brief": "ready:brooks_structure:SC2603:review_manual_stop_entry:96",
                "head": {
                    "symbol": "SC2603",
                    "action": "review_manual_stop_entry",
                    "reason": "second_entry_trend_continuation",
                    "blocker_detail": "manual bridge missing",
                    "done_when": "manual trader confirms trigger",
                },
            },
            "operator_repair_head_lane": {
                "status": "ready",
                "brief": "ready:ops_live_gate:rollback_hard:99",
                "head": {
                    "symbol": "ROLLBACK_HARD",
                    "action": "clear_ops_live_gate_condition",
                    "clear_when": "clear rollback",
                    "reason": "clear hard rollback",
                },
            },
            "operator_focus_slots": [
                {
                    "slot": "primary",
                    "symbol": "XAUUSD",
                    "action": "wait_for_paper_execution_close_evidence",
                    "reason": "paper_execution_close_evidence_pending",
                    "state": "waiting",
                    "priority_score": 99,
                    "priority_tier": "waiting_now",
                    "source_artifact": "/tmp/xau.json",
                    "done_when": "close evidence appears",
                },
                {
                    "slot": "followup",
                    "symbol": "XAGUSD",
                    "action": "wait_for_paper_execution_fill_evidence",
                    "reason": "paper_execution_fill_evidence_pending",
                    "state": "waiting",
                    "priority_score": 98,
                    "priority_tier": "waiting_now",
                    "source_artifact": "/tmp/xag.json",
                    "done_when": "fill evidence appears",
                },
                {
                    "slot": "secondary",
                    "symbol": "SC2603",
                    "action": "review_manual_stop_entry",
                    "reason": "second_entry_trend_continuation",
                    "state": "review",
                    "priority_score": 96,
                    "priority_tier": "review_queue_now",
                    "source_artifact": str(brooks_queue),
                    "done_when": "manual trader confirms trigger",
                },
            ],
            "operator_action_queue": [
                {"rank": 1, "symbol": "XAUUSD", "action": "wait_for_paper_execution_close_evidence", "reason": "paper_execution_close_evidence_pending", "priority_score": 99, "priority_tier": "waiting_now"},
                {"rank": 2, "symbol": "XAGUSD", "action": "wait_for_paper_execution_fill_evidence", "reason": "paper_execution_fill_evidence_pending", "priority_score": 98, "priority_tier": "waiting_now"},
                {"rank": 3, "symbol": "SC2603", "action": "review_manual_stop_entry", "reason": "second_entry_trend_continuation", "priority_score": 96, "priority_tier": "review_queue_now"},
                {"rank": 4, "symbol": "ROLLBACK_HARD", "action": "clear_ops_live_gate_condition", "reason": "clear hard rollback", "priority_score": 99, "priority_tier": "repair_queue_now"},
            ],
            "operator_action_checklist": [
                {"rank": 1, "state": "waiting", "symbol": "XAUUSD", "action": "wait_for_paper_execution_close_evidence", "reason": "paper_execution_close_evidence_pending", "priority_score": 99, "done_when": "close evidence appears"},
                {"rank": 2, "state": "waiting", "symbol": "XAGUSD", "action": "wait_for_paper_execution_fill_evidence", "reason": "paper_execution_fill_evidence_pending", "priority_score": 98, "done_when": "fill evidence appears"},
                {"rank": 3, "state": "review", "symbol": "SC2603", "action": "review_manual_stop_entry", "reason": "second_entry_trend_continuation", "priority_score": 96, "done_when": "manual trigger"},
                {"rank": 4, "state": "repair", "symbol": "ROLLBACK_HARD", "action": "clear_ops_live_gate_condition", "reason": "clear hard rollback", "priority_score": 99, "done_when": "clear rollback"},
            ],
            "operator_repair_queue": [
                {"rank": 1, "area": "ops_live_gate", "symbol": "ROLLBACK_HARD", "action": "clear_ops_live_gate_condition", "priority_score": 99, "command": "live-ops-reconcile-status", "clear_when": "clear rollback"},
                {"rank": 2, "area": "ops_live_gate", "symbol": "SLOT_ANOMALY", "action": "clear_ops_live_gate_condition", "priority_score": 98, "command": "live-ops-reconcile-status", "clear_when": "clear anomaly"},
            ],
            "review_backlog": [
                {"rank": 1, "area": "brooks_structure", "symbol": "SC2603", "action": "review_manual_stop_entry", "priority_score": 96, "priority_tier": "review_queue_now", "reason": "second_entry_trend_continuation", "blocker_detail": "manual bridge missing", "done_when": "manual trader confirms trigger"},
                {"rank": 2, "area": "crypto_route", "symbol": "SOLUSDT", "action": "deprioritize_flow", "priority_score": 58, "priority_tier": "review_queue_next", "reason": "no edge", "blocker_detail": "bias only", "done_when": "micro confirms"},
            ],
            "operator_backlog": [
                {"rank": 1, "area": "commodity_execution_close_evidence", "symbol": "XAUUSD", "action": "wait_for_paper_execution_close_evidence", "state": "waiting", "priority_score": 99},
                {"rank": 2, "area": "commodity_fill_evidence", "symbol": "XAGUSD", "action": "wait_for_paper_execution_fill_evidence", "state": "waiting", "priority_score": 98},
            ],
        },
    )
    for path, payload in [
        (brooks_refresh, {"status": "ok", "as_of": "2026-03-14T00:09:00Z"}),
        (brooks_queue, {"status": "ok", "as_of": "2026-03-14T00:09:01Z"}),
        (brooks_route, {"status": "ok", "as_of": "2026-03-14T00:09:02Z"}),
        (brooks_plan, {"status": "ok", "as_of": "2026-03-14T00:09:03Z"}),
        (crypto_refresh, {"status": "ok", "as_of": "2026-03-14T00:09:04Z"}),
        (live_gate, {"status": "ok", "as_of": "2026-03-14T00:09:05Z"}),
        (remote_live_handoff, {"status": "ok", "as_of": "2026-03-14T00:09:06Z"}),
        (remote_live_history, {"status": "ok", "generated_at_utc": "2026-03-14T00:09:07Z"}),
        (
            time_sync_repair_plan,
            {
                "status": "run_now",
                "generated_at_utc": "2026-03-14T00:09:08Z",
                "plan_brief": "manual_time_repair_required:SC2603:timed_apns_fallback",
                "admin_required": True,
                "done_when": "fix time sync and rerun crypto route refresh",
            },
        ),
        (
            time_sync_repair_verification,
            {
                "status": "blocked",
                "generated_at_utc": "2026-03-14T00:09:09Z",
                "verification_brief": "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked",
                "cleared": False,
            },
        ),
        (
            openclaw_blueprint,
            {
                "status": "ok",
                "generated_at_utc": "2026-03-14T00:09:10Z",
                "current_status": {
                    "current_life_stage": "local_repair_gated_continuity_hardened_remote_guardian",
                    "target_life_stage": "orderflow_native_execution_organism",
                    "remote_intent_queue_brief": "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um",
                    "remote_intent_queue_recommendation": "hold_remote_idle_until_ticket_ready",
                },
                "immediate_backlog": [
                    {
                        "priority": 1,
                        "title": "Repair local time sync to unlock guarded canary review",
                        "target_artifact": "system_time_sync_repair_verification_report",
                    }
                ],
                "control_chain": [
                    {
                        "stage": "research",
                        "label": "研究",
                        "mission": "验证 family 级边际",
                        "source_status": "research_ready",
                        "source_brief": "cross_section_ready:SC2603",
                        "source_decision": "keep_research_fresh",
                        "source_artifact_key": "crypto_shortline_cross_section_backtest",
                        "source_artifact": "/tmp/cross_section.json",
                        "next_target_artifact": "crypto_shortline_cross_section_backtest",
                        "blocking_reason": "-",
                        "interface_fields": [
                            "crypto_shortline_cross_section_backtest_status",
                            "crypto_shortline_cross_section_backtest_decision",
                        ],
                        "optimization_brief": "持续刷新研究层的 OOS 和成本后期望。",
                    },
                    {
                        "stage": "signal",
                        "label": "信号",
                        "mission": "把研究下沉成可执行窗口",
                        "source_status": "ticket_actionability_blocked",
                        "source_brief": "shadow_ready_ticket_actionability_blocked:SC2603",
                        "source_decision": "repair_local_fake_ip_ntp_path_then_review_guarded_canary",
                        "source_artifact_key": "remote_ticket_actionability_state",
                        "source_artifact": "/tmp/ticket_actionability.json",
                        "next_target_artifact": "system_time_sync_repair_verification_report",
                        "blocking_reason": "timed_ntp_via_fake_ip",
                        "interface_fields": [
                            "remote_ticket_actionability_status",
                            "remote_ticket_actionability_decision",
                        ],
                        "optimization_brief": "持续保持 signal handoff 和 guardian 语义一致。",
                    },
                ],
                "continuous_optimization_backlog": [
                    {
                        "priority": 85,
                        "stage": "execution",
                        "label": "执行",
                        "title": "持续优化执行传导链",
                        "why": "持续优化 executor、policy、canary gate 和 promotion gate 的传导链。",
                        "target_artifact": "remote_guarded_canary_promotion_gate",
                        "source_status": "shadow_policy_blocked",
                        "source_artifact": "/tmp/promotion_gate.json",
                        "interface_fields": [
                            "remote_guarded_canary_promotion_gate_status",
                            "remote_execution_actor_canary_gate_status",
                        ],
                        "change_class": "LIVE_GUARD_ONLY",
                    }
                ],
            },
        ),
    ]:
        write_json(path, payload)

    write_json(
        hot_brief,
        {
            "status": "ok",
            "as_of": "2026-03-14T00:11:00Z",
            "operator_action_queue_brief": "1:XAUUSD | 2:XAGUSD | 3:SC2603 | 4:ROLLBACK_HARD",
            "cross_market_operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
            "cross_market_review_head_brief": "ready:brooks_structure:SC2603:review_manual_stop_entry:96",
            "cross_market_operator_repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
            "cross_market_remote_live_takeover_gate_brief": "blocked_by_remote_live_gate:XAUUSD",
            "cross_market_remote_live_takeover_clearing_brief": "clearing_required:ops_live_gate+risk_guard",
            "cross_market_operator_backlog_state_brief": "waiting=2 | review=3 | watch=0 | blocked=0 | repair=7",
            "cross_market_operator_lane_priority_order_brief": "repair@654:7 > review@199:3 > waiting@197:2 > watch@0:0 > blocked@0:0",
            "source_crypto_route_refresh_reuse_gate_brief": "reuse_non_blocking:skip_native_refresh:9/9",
            "source_live_gate_blocker_remote_live_diagnosis_brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
            "source_remote_live_history_audit_window_brief": "24h:14.82pnl/20tr/1open | 7d:0.59pnl/30tr/1open | 30d:18.79pnl/38tr/1open",
            "source_brooks_structure_refresh_brief": "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now",
            "source_brooks_route_report_selected_routes_brief": "equity:three_push_climax_reversal | future:second_entry_trend_continuation",
            "source_cross_market_operator_state_artifact": str(cross_market),
            "source_cross_market_operator_state_status": "ok",
            "source_cross_market_operator_state_as_of": "2026-03-14T00:10:00Z",
            "source_cross_market_operator_state_snapshot_brief": "ok | 2026-03-14T00:10:00Z | operator=ready",
            "source_crypto_route_refresh_artifact": str(crypto_refresh),
            "source_crypto_route_refresh_status": "ok",
            "source_crypto_route_refresh_as_of": "2026-03-14T00:09:04Z",
            "source_live_gate_blocker_artifact": str(live_gate),
            "source_remote_live_handoff_artifact": str(remote_live_handoff),
            "source_remote_live_handoff_ready_check_scope_brief": "portfolio_margin_um:portfolio_margin_um",
            "source_remote_live_history_audit_artifact": str(remote_live_history),
            "source_brooks_structure_refresh_artifact": str(brooks_refresh),
            "source_brooks_structure_review_queue_artifact": str(brooks_queue),
            "source_brooks_structure_review_queue_brief": "ready:SC2603:96:review_queue_now",
            "source_brooks_route_report_artifact": str(brooks_route),
            "source_brooks_execution_plan_artifact": str(brooks_plan),
            "source_system_time_sync_repair_plan_artifact": str(time_sync_repair_plan),
            "source_system_time_sync_repair_plan_status": "run_now",
            "source_system_time_sync_repair_plan_brief": "manual_time_repair_required:SC2603:timed_apns_fallback",
            "source_system_time_sync_repair_plan_admin_required": True,
            "source_system_time_sync_repair_verification_artifact": str(time_sync_repair_verification),
            "source_system_time_sync_repair_verification_status": "blocked",
            "source_system_time_sync_repair_verification_brief": "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked",
            "source_system_time_sync_repair_verification_cleared": False,
            "source_openclaw_orderflow_blueprint_artifact": str(openclaw_blueprint),
            "source_openclaw_orderflow_blueprint_status": "ok",
            "source_openclaw_orderflow_blueprint_brief": (
                "local_repair_gated_continuity_hardened_remote_guardian -> orderflow_native_execution_organism | "
                "P1:system_time_sync_repair_verification_report:Repair local time sync to unlock guarded canary review"
            ),
            "source_openclaw_orderflow_blueprint_current_life_stage": (
                "local_repair_gated_continuity_hardened_remote_guardian"
            ),
            "source_openclaw_orderflow_blueprint_target_life_stage": "orderflow_native_execution_organism",
            "source_openclaw_orderflow_blueprint_remote_intent_queue_brief": "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um",
            "source_openclaw_orderflow_blueprint_remote_intent_queue_status": "queued_wait_trade_readiness",
            "source_openclaw_orderflow_blueprint_remote_intent_queue_recommendation": "hold_remote_idle_until_ticket_ready",
            "source_openclaw_orderflow_blueprint_remote_execution_journal_brief": (
                "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um"
                " | blocked:ticket_missing:no_actionable_ticket | not_attempted_wait_trade_readiness"
            ),
            "source_openclaw_orderflow_blueprint_remote_execution_journal_status": "intent_logged_guardian_blocked",
            "source_openclaw_orderflow_blueprint_remote_execution_journal_append_status": "appended",
            "source_openclaw_orderflow_blueprint_remote_orderflow_feedback_brief": (
                "downrank_guardian_blocked_route:SC2603:queue_aging_high:ticket_missing:no_actionable_ticket"
            ),
            "source_openclaw_orderflow_blueprint_remote_orderflow_feedback_status": (
                "downrank_guardian_blocked_route"
            ),
            "source_openclaw_orderflow_blueprint_remote_orderflow_feedback_recommendation": (
                "downrank_until_ticket_fresh_and_guardian_clear"
            ),
            "source_openclaw_orderflow_blueprint_remote_orderflow_policy_brief": (
                "shadow_policy_blocked:SC2603:queued_wait_trade_readiness:downrank_guardian_blocked_route"
            ),
            "source_openclaw_orderflow_blueprint_remote_orderflow_policy_status": (
                "shadow_policy_blocked"
            ),
            "source_openclaw_orderflow_blueprint_remote_orderflow_policy_decision": (
                "reject_until_guardian_clear"
            ),
            "source_openclaw_orderflow_blueprint_remote_execution_ack_brief": (
                "shadow_no_send_ack_recorded:SC2603:not_sent_policy_blocked:no_fill_execution_not_attempted"
            ),
            "source_openclaw_orderflow_blueprint_remote_execution_ack_status": (
                "shadow_no_send_ack_recorded"
            ),
            "source_openclaw_orderflow_blueprint_remote_execution_ack_decision": (
                "record_reject_without_transport"
            ),
            "source_openclaw_orderflow_blueprint_remote_orderflow_quality_shadow_learning_score": 65,
            "source_openclaw_orderflow_blueprint_remote_orderflow_quality_execution_readiness_score": 0,
            "source_openclaw_orderflow_blueprint_remote_orderflow_quality_transport_observability_score": 90,
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_brief": (
                "guardian_blocker_clearance_blocked:SC2603:5_blocked:portfolio_margin_um"
            ),
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_status": (
                "guardian_blocker_clearance_blocked"
            ),
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_score": 0,
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_code": (
                "timed_ntp_via_fake_ip_clearance"
            ),
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_title": (
                "Repair fake-ip NTP path before any orderflow promotion"
            ),
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_target_artifact": (
                "system_time_sync_repair_verification_report"
            ),
            "source_openclaw_orderflow_blueprint_remote_guardian_blocker_clearance_top_blocker_next_action": (
                "run_manual_time_repair_then_verify"
            ),
            "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_brief": (
                "shadow_learning_continuity_stable:SC2603:shadow_feedback_alive:portfolio_margin_um"
            ),
            "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_status": (
                "shadow_learning_continuity_stable"
            ),
            "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_decision": (
                "continue_shadow_learning_collect_feedback"
            ),
            "source_openclaw_orderflow_blueprint_remote_shadow_learning_continuity_blocker_detail": "",
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief": (
                "local_time_sync_primary_blocker_shadow_ready:SC2603:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_status": (
                "local_time_sync_primary_blocker_shadow_ready"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_decision": (
                "repair_local_fake_ip_ntp_path_then_review_guarded_canary"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_blocker_scope": (
                "timed_ntp_via_fake_ip"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_title": (
                "Repair local fake-ip NTP path to unlock guarded canary review"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_target_artifact": (
                "system_time_sync_repair_verification_report"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_plan_brief": (
                "manual_time_repair_required:SC2603:timed_ntp_via_fake_ip"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_environment_classification": (
                "timed_ntp_via_fake_ip"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_environment_blocker_detail": (
                "timed_source=NTP; ntp_ip=198.18.0.17; delay_ms=118.702; clash_dns_mode=fake-ip; tun_stack=gvisor"
            ),
            "source_openclaw_orderflow_blueprint_remote_guarded_canary_promotion_gate_blocker_code": (
                "timed_ntp_via_fake_ip_clearance"
            ),
            "source_openclaw_orderflow_blueprint_top_backlog_title": "Repair local fake-ip NTP path to unlock guarded canary review",
            "source_openclaw_orderflow_blueprint_top_backlog_target_artifact": "system_time_sync_repair_verification_report",
            "source_openclaw_orderflow_blueprint_top_backlog_why": (
                "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
            ),
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT_PATH),
            "--review-dir",
            str(review_dir),
            "--dashboard-dist",
            str(dashboard_dist),
            "--now",
            "2026-03-14T00:12:00Z",
        ],
    )
    assert mod.main() == 0

    review_json = review_dir / "20260314T001200Z_operator_task_visual_panel.json"
    review_html = review_dir / "20260314T001200Z_operator_task_visual_panel.html"
    dashboard_html = dashboard_dist / "operator_task_visual_panel.html"
    dashboard_json = dashboard_dist / "operator_task_visual_panel_data.json"

    assert review_json.exists()
    assert review_html.exists()
    assert dashboard_html.exists()
    assert dashboard_json.exists()

    payload = json.loads(review_json.read_text(encoding="utf-8"))
    assert payload["summary"]["operator_head"]["symbol"] == "XAUUSD"
    assert payload["summary"]["review_head"]["symbol"] == "SC2603"
    assert payload["summary"]["repair_head"]["symbol"] == "ROLLBACK_HARD"
    assert payload["summary"]["priority_repair_plan_brief"] == "manual_time_repair_required:SC2603:timed_apns_fallback"
    assert payload["summary"]["priority_repair_verification_brief"] == (
        "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
    )
    assert payload["summary"]["openclaw_current_life_stage"] == (
        "local_repair_gated_continuity_hardened_remote_guardian"
    )
    assert payload["summary"]["openclaw_target_life_stage"] == "orderflow_native_execution_organism"
    assert (
        payload["summary"]["openclaw_intent_queue_brief"]
        == "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um"
    )
    assert payload["summary"]["openclaw_intent_queue_recommendation"] == "hold_remote_idle_until_ticket_ready"
    assert (
        payload["summary"]["openclaw_execution_ack_brief"]
        == "shadow_no_send_ack_recorded:SC2603:not_sent_policy_blocked:no_fill_execution_not_attempted"
    )
    assert payload["summary"]["openclaw_execution_ack_status"] == "shadow_no_send_ack_recorded"
    assert payload["summary"]["openclaw_execution_ack_decision"] == "record_reject_without_transport"
    assert payload["summary"]["openclaw_execution_journal_status"] == "intent_logged_guardian_blocked"
    assert payload["summary"]["openclaw_execution_journal_append_status"] == "appended"
    assert payload["summary"]["openclaw_execution_journal_brief"].startswith(
        "queued_wait_trade_readiness:SC2603:review_manual_stop_entry:portfolio_margin_um"
    )
    assert payload["summary"]["openclaw_orderflow_feedback_status"] == "downrank_guardian_blocked_route"
    assert payload["summary"]["openclaw_orderflow_feedback_recommendation"] == (
        "downrank_until_ticket_fresh_and_guardian_clear"
    )
    assert payload["summary"]["openclaw_orderflow_feedback_brief"].startswith(
        "downrank_guardian_blocked_route:SC2603"
    )
    assert payload["summary"]["openclaw_orderflow_policy_status"] == "shadow_policy_blocked"
    assert payload["summary"]["openclaw_orderflow_policy_decision"] == "reject_until_guardian_clear"
    assert payload["summary"]["openclaw_orderflow_policy_brief"].startswith(
        "shadow_policy_blocked:SC2603"
    )
    assert payload["summary"]["openclaw_quality_shadow_learning_score"] == 65
    assert payload["summary"]["openclaw_shadow_learning_continuity_brief"] == (
        "shadow_learning_continuity_stable:SC2603:shadow_feedback_alive:portfolio_margin_um"
    )
    assert payload["summary"]["openclaw_shadow_learning_continuity_status"] == (
        "shadow_learning_continuity_stable"
    )
    assert payload["summary"]["openclaw_shadow_learning_continuity_decision"] == (
        "continue_shadow_learning_collect_feedback"
    )
    assert payload["summary"]["openclaw_promotion_unblock_readiness_brief"] == (
        "local_time_sync_primary_blocker_shadow_ready:SC2603:repair_local_time_sync_then_review_guarded_canary:portfolio_margin_um"
    )
    assert payload["summary"]["openclaw_promotion_unblock_readiness_status"] == (
        "local_time_sync_primary_blocker_shadow_ready"
    )
    assert payload["summary"]["openclaw_promotion_unblock_readiness_decision"] == (
        "repair_local_fake_ip_ntp_path_then_review_guarded_canary"
    )
    assert payload["summary"]["openclaw_promotion_unblock_primary_blocker_scope"] == (
        "timed_ntp_via_fake_ip"
    )
    assert payload["summary"]["openclaw_promotion_unblock_primary_local_repair_plan_brief"] == (
        "manual_time_repair_required:SC2603:timed_ntp_via_fake_ip"
    )
    assert payload["summary"]["openclaw_promotion_unblock_primary_local_repair_environment_classification"] == (
        "timed_ntp_via_fake_ip"
    )
    assert payload["summary"]["openclaw_promotion_unblock_primary_local_repair_target_artifact"] == (
        "system_time_sync_repair_verification_report"
    )
    assert payload["summary"]["openclaw_quality_execution_readiness_score"] == 0
    assert payload["summary"]["openclaw_quality_transport_observability_score"] == 90
    assert payload["summary"]["openclaw_guardian_clearance_brief"].startswith(
        "guardian_blocker_clearance_blocked:SC2603"
    )
    assert payload["summary"]["openclaw_promotion_gate_blocker_code"] == (
        "timed_ntp_via_fake_ip_clearance"
    )
    assert payload["summary"]["openclaw_guardian_clearance_top_blocker_target_artifact"] == (
        "system_time_sync_repair_verification_report"
    )
    assert payload["summary"]["openclaw_top_backlog_target_artifact"] == (
        "system_time_sync_repair_verification_report"
    )
    assert payload["summary"]["openclaw_top_backlog_why"] == (
        "blocked:SC2603:probe_blocked+environment_blocked+review_head_time_sync_blocked"
    )
    assert payload["priority_repair_plan"]["admin_required"] is True
    assert payload["priority_repair_verification"]["status"] == "blocked"
    assert payload["openclaw_orderflow_blueprint"]["status"] == "ok"
    assert payload["control_chain"][0]["stage"] == "research"
    assert payload["control_chain"][1]["stage"] == "signal"
    assert payload["continuous_optimization_backlog"][0]["stage"] == "execution"
    assert payload["lane_cards"][4]["state"] == "repair"
    html_text = review_html.read_text(encoding="utf-8")
    assert "Task Visual Management Panel" in html_text
    assert "OpenClaw Evolution" in html_text
    assert "Transmission Control Chain" in html_text
    assert "Continuous Optimization Backlog" in html_text
    assert "Chain Transmission" in html_text
    assert "Priority Repair Plan" in html_text
    assert "Time Sync Repair Verification" in html_text
    assert "研究 | research" in html_text
    assert "持续优化执行传导链" in html_text
    assert "SC2603" in html_text
    assert "ROLLBACK_HARD" in html_text


def test_build_panel_payload_prioritizes_promotion_unblock_when_time_sync_is_already_clear(
    tmp_path: Path,
) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    dashboard_dist = tmp_path / "dashboard-dist"
    review_dir.mkdir()
    dashboard_dist.mkdir()

    cross_market = review_dir / "20260315T110210Z_cross_market_operator_state.json"
    hot_brief = review_dir / "20260315T110314Z_hot_universe_operator_brief.json"

    write_json(
        cross_market,
        {
            "status": "ok",
            "operator_backlog_state_counts": {"waiting": 1, "review": 1, "watch": 0, "blocked": 0, "repair": 1},
            "operator_backlog_priority_totals": {"waiting": 99, "review": 58, "watch": 0, "blocked": 0, "repair": 99},
            "operator_waiting_lane_brief": "1:XAUUSD:wait_for_paper_execution_close_evidence:99",
            "operator_review_lane_brief": "2:SOLUSDT:deprioritize_flow:58",
            "operator_watch_lane_brief": "-",
            "operator_blocked_lane_brief": "-",
            "operator_repair_lane_brief": "3:ROLLBACK_HARD:clear_ops_live_gate_condition:99",
            "operator_head_lane": {
                "head": {
                    "symbol": "XAUUSD",
                    "action": "wait_for_paper_execution_close_evidence",
                    "blocker_detail": "waiting close evidence",
                    "done_when": "close evidence appears",
                }
            },
            "review_head_lane": {
                "head": {
                    "symbol": "SOLUSDT",
                    "action": "deprioritize_flow",
                    "blocker_detail": "bias only",
                    "done_when": "setup ready",
                }
            },
            "operator_repair_head_lane": {
                "head": {
                    "symbol": "ROLLBACK_HARD",
                    "action": "clear_ops_live_gate_condition",
                    "reason": "clear rollback",
                    "clear_when": "rollback clears",
                }
            },
            "operator_focus_slots": [],
            "operator_action_queue": [],
            "operator_action_checklist": [],
            "operator_repair_queue": [],
            "review_backlog": [],
            "operator_backlog": [],
        },
    )
    write_json(
        hot_brief,
        {
            "status": "ok",
            "cross_market_operator_head_brief": "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99",
            "cross_market_review_head_brief": "review:crypto_route:SOLUSDT:deprioritize_flow:58",
            "cross_market_operator_repair_head_brief": "ready:ops_live_gate:rollback_hard:99",
            "cross_market_remote_live_takeover_gate_brief": "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um",
            "cross_market_remote_live_takeover_clearing_brief": "clearing_required:ops_live_gate+risk_guard",
            "cross_market_operator_backlog_state_brief": "waiting=1 | review=1 | watch=0 | blocked=0 | repair=1",
            "cross_market_operator_lane_priority_order_brief": "repair@99:1 > waiting@99:1 > review@58:1 > watch@0:0 > blocked@0:0",
            "source_cross_market_operator_state_artifact": str(cross_market),
            "source_system_time_sync_repair_plan_brief": "manual_time_repair_required:SOLUSDT:timed_apns_fallback",
            "source_system_time_sync_repair_plan_artifact": str(
                review_dir / "20260315T101040Z_system_time_sync_repair_plan.json"
            ),
            "source_system_time_sync_repair_plan_admin_required": True,
            "source_system_time_sync_repair_verification_brief": "cleared:SOLUSDT:time_sync_ok",
            "source_system_time_sync_repair_verification_artifact": str(
                review_dir / "20260315T101030Z_system_time_sync_repair_verification_report.json"
            ),
            "source_system_time_sync_repair_verification_cleared": True,
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_brief": (
                "shadow_ready_ticket_actionability_blocked:SOLUSDT:resolve_ticket_actionability_then_review_guarded_canary:portfolio_margin_um"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_readiness_status": (
                "shadow_ready_ticket_actionability_blocked"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_blocker_scope": (
                "guardian_ticket_actionability"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_required": False,
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_title": (
                "Resolve crypto ticket actionability before guarded canary review"
            ),
            "source_openclaw_orderflow_blueprint_remote_promotion_unblock_primary_local_repair_target_artifact": (
                "remote_ticket_actionability_state"
            ),
        },
    )

    payload = mod.build_panel_payload(
        review_dir=review_dir,
        dashboard_dist=dashboard_dist,
        reference_now=dt.datetime(2026, 3, 15, 11, 3, 20, tzinfo=dt.timezone.utc),
    )

    assert payload["summary"]["priority_repair_plan_brief"] == (
        "shadow_ready_ticket_actionability_blocked:SOLUSDT:resolve_ticket_actionability_then_review_guarded_canary:portfolio_margin_um"
    )
    assert payload["summary"]["priority_repair_plan_artifact"] == "remote_ticket_actionability_state"
    assert payload["summary"]["priority_repair_plan_admin_required"] is False
    assert payload["priority_repair_plan"]["status"] == "shadow_ready_ticket_actionability_blocked"
