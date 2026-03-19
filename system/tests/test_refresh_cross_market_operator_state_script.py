from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/refresh_cross_market_operator_state.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("cross_market_operator_state_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_now_is_monotonic() -> None:
    mod = _load_module()
    base = mod.parse_now("2026-03-13T13:00:00Z")
    assert mod.step_now(base, 0).isoformat() == "2026-03-13T13:00:00+00:00"
    assert mod.step_now(base, 1).isoformat() == "2026-03-13T13:00:01+00:00"
    assert mod.step_now(base, 1) > mod.step_now(base, 0)


def test_resolve_hot_brief_source_falls_back_when_explicit_brief_was_pruned(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True)
    fallback_path = review_dir / "20260313T125900Z_hot_universe_operator_brief.json"
    fallback_path.write_text(
        json.dumps({"status": "ok", "artifact": str(fallback_path)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    resolved = mod.resolve_hot_brief_source(
        review_dir=review_dir,
        commodity_refresh={"brief_artifact": str(review_dir / "20260313T124800Z_hot_universe_operator_brief.json")},
        reference_now=mod.parse_now("2026-03-13T13:00:00Z"),
    )

    assert resolved == fallback_path


def test_remote_live_takeover_clearing_preserves_source_freshness_brief() -> None:
    mod = _load_module()
    row = mod._derive_remote_live_takeover_clearing(
        {
            "remote_live_takeover_clearing": {
                "status": "clearing_required",
                "brief": "clearing_required:ops_live_gate+risk_guard",
                "blocker_detail": "ops and risk guard both block live takeover",
                "done_when": "clear blockers",
                "ops_live_gate_brief": "rollback_hard",
                "risk_guard_brief": "ticket_missing:no_actionable_ticket",
            },
            "source_freshness": {
                "brief": "ops_reconcile=fresh:0.003h | risk_guard=fresh:5.0s",
            },
        }
    )

    assert row["status"] == "clearing_required"
    assert row["source_freshness_brief"] == "ops_reconcile=fresh:0.003h | risk_guard=fresh:5.0s"


def test_remote_live_takeover_slot_anomaly_breakdown_is_preserved() -> None:
    mod = _load_module()
    row = mod._derive_remote_live_takeover_slot_anomaly_breakdown(
        {
            "slot_anomaly_breakdown": {
                "status": "slot_anomaly_active_root_cause",
                "brief": "slot_anomaly_active_root_cause:2026-03-16",
                "artifact": "/tmp/slot_anomaly_breakdown.json",
                "repair_focus": "优先修复 slot_anomaly 缺陷并重跑 lie ops-report --date 2026-03-16 --window-days 7",
            }
        }
    )

    assert row["status"] == "slot_anomaly_active_root_cause"
    assert row["brief"] == "slot_anomaly_active_root_cause:2026-03-16"
    assert row["artifact"] == "/tmp/slot_anomaly_breakdown.json"


def test_build_review_backlog_dedupes_matching_brooks_and_secondary_focus() -> None:
    mod = _load_module()
    backlog = mod._build_review_backlog(
        {
            "brooks_structure_operator_status": "ready",
            "brooks_structure_operator_head_symbol": "SC2603",
            "brooks_structure_operator_head_action": "review_manual_stop_entry",
            "brooks_structure_operator_head_priority_score": 96,
            "brooks_structure_operator_head_priority_tier": "review_queue_now",
            "brooks_structure_operator_blocker_detail": "manual only",
            "brooks_structure_operator_done_when": "manual trader confirms venue and sizing",
            "secondary_focus_area": "brooks_structure",
            "secondary_focus_symbol": "SC2603",
            "secondary_focus_action": "review_manual_stop_entry",
            "secondary_focus_state": "review",
            "secondary_focus_priority_score": 96,
            "secondary_focus_priority_tier": "review_queue_now",
            "secondary_focus_blocker_detail": "manual only",
            "secondary_focus_done_when": "manual trader confirms venue and sizing",
        }
    )
    assert len(backlog) == 1
    assert backlog[0]["area"] == "brooks_structure"
    assert backlog[0]["symbol"] == "SC2603"
    assert backlog[0]["action"] == "review_manual_stop_entry"


def test_build_review_backlog_from_rows_keeps_all_review_rows() -> None:
    mod = _load_module()
    backlog = mod._build_review_backlog_from_rows(
        [
            {
                "area": "brooks_structure",
                "symbol": "SC2603",
                "action": "review_manual_stop_entry",
                "state": "review",
                "priority_score": 96,
                "priority_tier": "review_queue_now",
            },
            {
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "state": "review",
                "priority_score": 73,
                "priority_tier": "review_queue_now",
            },
            {
                "area": "crypto_route",
                "symbol": "BNBUSDT",
                "action": "watch_priority_until_long_window_confirms",
                "state": "review",
                "priority_score": 45,
                "priority_tier": "review_queue_next",
            },
        ]
    )
    assert [row["symbol"] for row in backlog] == ["SC2603", "SOLUSDT", "BNBUSDT"]
    assert [row["rank"] for row in backlog] == [1, 2, 3]


def test_build_review_backlog_from_rows_demotes_stale_brooks_below_fresh_crypto() -> None:
    mod = _load_module()
    backlog = mod._build_review_backlog_from_rows(
        [
            {
                "area": "brooks_structure",
                "symbol": "SC2603",
                "action": "review_manual_stop_entry",
                "state": "review",
                "priority_score": 96,
                "priority_tier": "review_queue_now",
                "source_refresh_action": "consider_refresh_before_promotion",
                "promotion_ready": False,
            },
            {
                "area": "crypto_route",
                "symbol": "SOLUSDT",
                "action": "deprioritize_flow",
                "state": "review",
                "priority_score": 58,
                "priority_tier": "review_queue_next",
            },
        ]
    )
    assert [row["symbol"] for row in backlog] == ["SOLUSDT", "SC2603"]
    assert backlog[0]["promotion_ready"] is True
    assert backlog[1]["promotion_ready"] is False
    assert backlog[1]["action"] == "consider_refresh_before_promotion"
    assert backlog[1]["review_action"] == "review_manual_stop_entry"


def test_build_review_head_lane_marks_non_promotable_head_as_refresh_required() -> None:
    mod = _load_module()
    backlog = mod._build_review_backlog_from_rows(
        [
            {
                "area": "brooks_structure",
                "symbol": "SC2603",
                "action": "review_manual_stop_entry",
                "state": "review",
                "priority_score": 96,
                "priority_tier": "review_queue_now",
                "reason": "second_entry_trend_continuation",
                "blocker_detail": "manual only | source freshness guard",
                "done_when": "SC2603 receives a fresh brooks artifact before promotion",
                "source_refresh_action": "consider_refresh_before_promotion",
                "promotion_ready": False,
            }
        ]
    )
    lane = mod._build_review_head_lane(backlog, "ready")
    assert lane["status"] == "refresh_required"
    assert lane["brief"] == (
        "refresh_required:brooks_structure:SC2603:consider_refresh_before_promotion:96"
    )
    assert lane["head"]["action"] == "consider_refresh_before_promotion"
    assert lane["head"]["review_action"] == "review_manual_stop_entry"


def test_build_operator_backlog_uses_refresh_action_for_non_promotable_review_rows() -> None:
    mod = _load_module()
    backlog = mod._build_operator_backlog(
        commodity_waiting_rows=[],
        brooks_review_rows=[
            {
                "area": "brooks_structure",
                "symbol": "SC2603",
                "action": "review_manual_stop_entry",
                "state": "review",
                "priority_score": 96,
                "priority_tier": "review_queue_now",
                "reason": "second_entry_trend_continuation",
                "blocker_detail": "manual only | source freshness guard",
                "done_when": "SC2603 receives a fresh brooks artifact before promotion",
                "source_refresh_action": "consider_refresh_before_promotion",
                "promotion_ready": False,
            }
        ],
        crypto_review_rows=[],
        remote_live_takeover_repair_queue=None,
    )
    assert len(backlog) == 1
    assert backlog[0]["action"] == "consider_refresh_before_promotion"
    assert backlog[0]["review_action"] == "review_manual_stop_entry"


def test_build_operator_state_lanes_marks_stale_review_only_lane_as_refresh_required() -> None:
    mod = _load_module()
    lanes = mod._build_operator_state_lanes(
        [
            {
                "area": "brooks_structure",
                "symbol": "SC2603",
                "action": "consider_refresh_before_promotion",
                "review_action": "review_manual_stop_entry",
                "state": "review",
                "priority_score": 96,
                "priority_tier": "review_queue_now",
                "source_refresh_action": "consider_refresh_before_promotion",
                "promotion_ready": False,
            }
        ]
    )
    assert lanes["review"]["status"] == "refresh_required"
    assert lanes["review"]["head_action"] == "consider_refresh_before_promotion"


def test_operator_focus_slots_preserve_source_metadata_from_review_head() -> None:
    mod = _load_module()
    review_head = {
        "area": "crypto_route",
        "target": "SOLUSDT",
        "symbol": "SOLUSDT",
        "action": "inspect_source_state",
        "review_action": "deprioritize_flow",
        "state": "review",
        "priority_score": 58,
        "priority_tier": "review_queue_next",
        "reason": "Flow does not produce a positive ranked edge even in the short sample.",
        "blocker_detail": "source freshness guard: crypto_route_refresh artifact is partial_failure and fresh",
        "done_when": "SOLUSDT receives a fresh crypto_route_refresh artifact before promotion",
        "source_kind": "crypto_route_refresh",
        "source_artifact": "/tmp/20260314T135023Z_crypto_route_refresh.json",
        "source_status": "partial_failure",
        "source_as_of": "2026-03-14T13:50:23Z",
        "source_age_minutes": 0,
        "source_recency": "fresh",
        "source_health": "status_partial_failure",
        "source_refresh_action": "inspect_source_state",
    }
    action_queue, action_checklist, _, _ = mod._build_operator_action_lanes(
        operator_state_lanes={},
        review_head=review_head,
        operator_repair_queue=[],
    )
    slots, _ = mod._build_operator_focus_slots(action_checklist)

    assert len(action_queue) == 1
    assert action_queue[0]["source_kind"] == "crypto_route_refresh"
    assert action_queue[0]["source_status"] == "partial_failure"
    assert len(slots) == 1
    assert slots[0]["slot"] == "primary"
    assert slots[0]["source_kind"] == "crypto_route_refresh"
    assert slots[0]["source_artifact"] == "/tmp/20260314T135023Z_crypto_route_refresh.json"
    assert slots[0]["source_status"] == "partial_failure"
    assert slots[0]["source_health"] == "status_partial_failure"
    assert slots[0]["source_refresh_action"] == "inspect_source_state"


def test_build_brooks_review_rows_marks_carry_over_as_not_promotion_ready() -> None:
    mod = _load_module()
    rows = mod._build_brooks_review_rows(
        reference_now=mod.parse_now("2026-03-14T08:00:00Z"),
        refresh_payload={
            "status": "ok",
            "artifact": "/tmp/20260313T124753Z_brooks_structure_refresh.json",
        },
        review_queue_payload={
            "status": "ok",
            "artifact": "/tmp/20260313T124756Z_brooks_structure_review_queue.json",
            "queue": [
                {
                    "symbol": "SC2603",
                    "execution_action": "review_manual_stop_entry",
                    "priority_score": 96,
                    "priority_tier": "review_queue_now",
                    "plan_status": "manual_structure_review_now",
                    "strategy_id": "second_entry_trend_continuation",
                    "blocker_detail": "manual only",
                    "done_when": "manual trader confirms venue and sizing",
                }
            ],
        },
    )
    assert len(rows) == 1
    assert rows[0]["source_recency"] == "carry_over"
    assert rows[0]["source_health"] == "carry_over_ok"
    assert rows[0]["source_refresh_action"] == "consider_refresh_before_promotion"
    assert rows[0]["promotion_ready"] is False
    assert "source freshness guard" in rows[0]["blocker_detail"]


def test_build_crypto_review_rows_marks_stale_source_as_not_promotion_ready() -> None:
    mod = _load_module()
    rows = mod._build_crypto_review_rows(
        reference_now=mod.parse_now("2026-03-14T08:00:00Z"),
        refresh_payload={
            "status": "ok",
            "artifact": "/tmp/20260313T055019Z_crypto_route_refresh.json",
            "review_priority_queue": [
                {
                    "symbol": "SOLUSDT",
                    "route_action": "deprioritize_flow",
                    "route_status_label": "review",
                    "priority_score": 58,
                    "priority_tier": "review_queue_next",
                    "reason": "no edge",
                    "blocker_detail": "no_edge",
                    "done_when": "micro gate recovers",
                }
            ],
        },
    )
    assert len(rows) == 1
    assert rows[0]["source_recency"] == "stale"
    assert rows[0]["source_health"] == "refresh_required"
    assert rows[0]["source_refresh_action"] == "refresh_source_before_use"
    assert rows[0]["promotion_ready"] is False
    assert "source freshness guard" in rows[0]["blocker_detail"]


def test_build_crypto_review_rows_marks_dependency_carry_over_as_refresh_required() -> None:
    mod = _load_module()
    rows = mod._build_crypto_review_rows(
        reference_now=mod.parse_now("2026-03-14T10:45:00Z"),
        refresh_payload={
            "status": "ok",
            "artifact": "/tmp/20260314T104241Z_crypto_route_refresh.json",
            "steps": [
                {
                    "name": "build_crypto_cvd_semantic_snapshot",
                    "status": "micro_capture_missing",
                },
                {
                    "name": "build_crypto_cvd_queue_handoff",
                    "status": "semantic_snapshot_invalid",
                },
                {
                    "name": "build_crypto_shortline_execution_gate",
                    "status": "carry_over_previous_artifact",
                },
            ],
            "review_priority_queue": [
                {
                    "symbol": "SOLUSDT",
                    "route_action": "deprioritize_flow",
                    "route_status_label": "review",
                    "priority_score": 55,
                    "priority_tier": "review_queue_next",
                    "reason": "no edge",
                    "blocker_detail": "bias_only",
                    "done_when": "micro gate recovers",
                }
            ],
        },
    )
    assert len(rows) == 1
    assert rows[0]["source_health"] == "refresh_required"
    assert rows[0]["source_refresh_action"] == "refresh_source_before_use"
    assert rows[0]["promotion_ready"] is False
    assert "dependency_health=refresh_required" in rows[0]["blocker_detail"]


def test_main_builds_cross_market_review_backlog_from_hot_brief(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    context_path = review_dir / "NEXT_WINDOW_CONTEXT_LATEST.md"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    live_gate_blocker_path = review_dir / "20260313T125900Z_live_gate_blocker_report.json"
    live_gate_blocker_path.write_text(
        json.dumps(
            {
                "remote_live_diagnosis": {
                    "status": "profitability_confirmed_but_auto_live_blocked",
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                    "market": "portfolio_margin_um",
                    "blocker_detail": (
                        "portfolio_margin_um remote history confirms realized profitability, "
                        "but automated live remains blocked by ops_live_gate, risk_guard."
                    ),
                    "done_when": "ops_live_gate becomes clear and risk_guard reasons become empty",
                },
                "remote_live_takeover_clearing": {
                    "status": "clearing_required",
                    "brief": "clearing_required:ops_live_gate+risk_guard",
                    "blocker_detail": (
                        "ops_live_gate needs rollback_hard, risk_violations, max_drawdown, slot_anomaly; "
                        "risk_guard needs ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap"
                    ),
                    "done_when": "ops_live_gate becomes clear and risk_guard reasons become empty",
                    "ops_live_gate_brief": "rollback_hard, risk_violations, max_drawdown, slot_anomaly",
                    "risk_guard_brief": "ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap",
                },
                "remote_live_takeover_repair_queue": {
                    "status": "ready",
                    "brief": "ready:ops_live_gate:rollback_hard:99",
                    "count": 7,
                    "head_area": "ops_live_gate",
                    "head_code": "rollback_hard",
                    "head_action": "clear_ops_live_gate_condition",
                    "head_priority_score": 99,
                    "head_priority_tier": "repair_queue_now",
                    "head_command": "cmd-gate",
                    "head_clear_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
                    "queue_brief": (
                        "1:ops_live_gate:rollback_hard:99 | 2:ops_live_gate:slot_anomaly:98 | "
                        "3:ops_live_gate:backtest_snapshot:97 | 4:ops_live_gate:ops_status_red:96 | "
                        "5:risk_guard:ticket_missing:no_actionable_ticket:89 | 6:risk_guard:panic_cooldown_active:88 | "
                        "7:risk_guard:open_exposure_above_cap:87"
                    ),
                    "items": [
                        {
                            "rank": 1,
                            "area": "ops_live_gate",
                            "code": "rollback_hard",
                            "action": "clear_ops_live_gate_condition",
                            "priority_score": 99,
                            "priority_tier": "repair_queue_now",
                            "command": "cmd-gate",
                            "clear_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
                            "goal": "Clear hard rollback state before any live capital increase.",
                        },
                        {
                            "rank": 2,
                            "area": "ops_live_gate",
                            "code": "slot_anomaly",
                            "action": "clear_ops_live_gate_condition",
                            "priority_score": 98,
                            "priority_tier": "repair_queue_now",
                            "command": "cmd-gate",
                            "clear_when": "reconcile slot state until slot anomaly checks pass",
                        },
                        {
                            "rank": 3,
                            "area": "ops_live_gate",
                            "code": "backtest_snapshot",
                            "action": "clear_ops_live_gate_condition",
                            "priority_score": 97,
                            "priority_tier": "repair_queue_now",
                            "command": "cmd-gate",
                            "clear_when": "refresh the backtest snapshot until snapshot health returns to green",
                        },
                        {
                            "rank": 4,
                            "area": "ops_live_gate",
                            "code": "ops_status_red",
                            "action": "clear_ops_live_gate_condition",
                            "priority_score": 96,
                            "priority_tier": "repair_queue_now",
                            "command": "cmd-gate",
                            "clear_when": "restore ops status from red to a healthy non-red state",
                        },
                        {
                            "rank": 5,
                            "area": "risk_guard",
                            "code": "ticket_missing:no_actionable_ticket",
                            "action": "clear_risk_guard_condition",
                            "priority_score": 89,
                            "priority_tier": "repair_queue_now",
                            "command": "cmd-risk",
                            "clear_when": "generate at least one fresh actionable ticket that survives confidence and size filters",
                        },
                        {
                            "rank": 6,
                            "area": "risk_guard",
                            "code": "panic_cooldown_active",
                            "action": "clear_risk_guard_condition",
                            "priority_score": 88,
                            "priority_tier": "repair_queue_now",
                            "command": "cmd-risk",
                            "clear_when": "wait for or explicitly clear the active panic cooldown before automated routing resumes",
                        },
                        {
                            "rank": 7,
                            "area": "risk_guard",
                            "code": "open_exposure_above_cap",
                            "action": "clear_risk_guard_condition",
                            "priority_score": 87,
                            "priority_tier": "repair_queue_now",
                            "command": "cmd-risk",
                            "clear_when": "reduce or close exposure until open exposure is back under the configured cap",
                        },
                    ],
                    "done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    commodity_review_path = review_dir / "20260313T125901Z_commodity_paper_execution_review.json"
    commodity_review_path.write_text(
        json.dumps(
            {
                "status": "paper-execution-close-evidence-pending-fill-remainder",
                "next_close_evidence_execution_symbol": "XAUUSD",
                "next_fill_evidence_execution_symbol": "XAGUSD",
                "review_items": [
                    {
                        "queue_rank": 1,
                        "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                        "symbol": "XAUUSD",
                        "paper_open_date": "2026-03-11",
                    },
                    {
                        "queue_rank": 2,
                        "execution_id": "commodity-paper-execution:metals_all:XAGUSD",
                        "symbol": "XAGUSD",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    commodity_gap_path = review_dir / "20260313T125902Z_commodity_paper_execution_gap_report.json"
    commodity_gap_path.write_text(
        json.dumps(
            {
                "status": "blocking_gap_active",
                "queue_symbols_with_stale_directional_signal_dates": {"XAGUSD": "2026-03-10"},
                "queue_symbols_with_stale_directional_signal_age_days": {"XAGUSD": 2},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    crypto_route_refresh_path = review_dir / "20260313T125903Z_crypto_route_refresh.json"
    crypto_route_refresh_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "review_priority_queue": [
                    {
                        "rank": 1,
                        "symbol": "SOLUSDT",
                        "route_action": "deprioritize_flow",
                        "route_status_label": "review",
                        "priority_score": 73,
                        "priority_tier": "review_queue_now",
                        "reason": "no edge",
                        "blocker_detail": "no_edge",
                        "done_when": "micro gate recovers",
                    },
                    {
                        "rank": 2,
                        "symbol": "BNBUSDT",
                        "route_action": "watch_priority_until_long_window_confirms",
                        "route_status_label": "watch_priority",
                        "priority_score": 45,
                        "priority_tier": "review_queue_next",
                        "reason": "laggy edge",
                        "blocker_detail": "long window pending",
                        "done_when": "long window confirms",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    brooks_review_queue_path = review_dir / "20260313T125904Z_brooks_structure_review_queue.json"
    brooks_review_queue_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "queue": [
                    {
                        "rank": 1,
                        "symbol": "SC2603",
                        "execution_action": "review_manual_stop_entry",
                        "priority_score": 96,
                        "priority_tier": "review_queue_now",
                        "blocker_detail": "manual only",
                        "done_when": "manual trader confirms venue and sizing",
                        "plan_status": "manual_structure_review_now",
                        "strategy_id": "second_entry_trend_continuation",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    hot_brief_path = review_dir / "hot_universe_operator_brief.json"
    hot_brief_path.write_text(
        json.dumps(
            {
                "operator_status": "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch",
                "operator_stack_brief": "commodity:close-evidence:XAUUSD | crypto:SOLUSDT:deprioritize_flow",
                "source_live_gate_blocker_artifact": str(live_gate_blocker_path),
                "source_live_gate_blocker_remote_live_diagnosis_status": "profitability_confirmed_but_auto_live_blocked",
                "source_live_gate_blocker_remote_live_diagnosis_brief": (
                    "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                ),
                "source_live_gate_blocker_remote_live_takeover_clearing_status": "clearing_required",
                "source_live_gate_blocker_remote_live_takeover_clearing_brief": (
                    "clearing_required:ops_live_gate+risk_guard"
                ),
                "source_live_gate_blocker_remote_live_takeover_clearing_blocker_detail": (
                    "ops_live_gate needs rollback_hard, risk_violations, max_drawdown, slot_anomaly; "
                    "risk_guard needs ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap"
                ),
                "source_live_gate_blocker_remote_live_takeover_clearing_done_when": (
                    "ops_live_gate becomes clear and risk_guard reasons become empty"
                ),
                "source_live_gate_blocker_remote_live_takeover_repair_queue_status": "ready",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_brief": "ready:ops_live_gate:rollback_hard:99",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_count": 7,
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_area": "ops_live_gate",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_code": "rollback_hard",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_action": "clear_ops_live_gate_condition",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_priority_score": 99,
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_priority_tier": "repair_queue_now",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_command": "cmd-gate",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_clear_when": "clear hard rollback so ops_live_gate can leave rollback_now state",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
                "source_live_gate_blocker_ops_live_gate_clearing_brief": (
                    "rollback_hard, risk_violations, max_drawdown, slot_anomaly"
                ),
                "source_live_gate_blocker_risk_guard_clearing_brief": (
                    "ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap"
                ),
                "brooks_structure_operator_status": "ready",
                "brooks_structure_operator_head_symbol": "SC2603",
                "brooks_structure_operator_head_action": "review_manual_stop_entry",
                "brooks_structure_operator_head_priority_score": 96,
                "brooks_structure_operator_head_priority_tier": "review_queue_now",
                "brooks_structure_operator_blocker_detail": "manual only",
                "brooks_structure_operator_done_when": "manual trader confirms venue and sizing",
                "secondary_focus_area": "crypto_route",
                "secondary_focus_symbol": "SOLUSDT",
                "secondary_focus_action": "deprioritize_flow",
                "secondary_focus_state": "review",
                "secondary_focus_priority_score": 73,
                "secondary_focus_priority_tier": "review_queue_now",
                "secondary_focus_blocker_detail": "no_edge",
                "secondary_focus_done_when": "micro gate recovers",
                "operator_action_checklist": [
                    {
                        "rank": 1,
                        "area": "commodity_execution_close_evidence",
                        "target": "commodity-paper-execution:metals_all:XAUUSD",
                        "symbol": "XAUUSD",
                        "action": "wait_for_paper_execution_close_evidence",
                        "reason": "paper_execution_close_evidence_pending",
                        "state": "waiting",
                        "blocker_detail": "waiting close evidence",
                        "done_when": "close evidence lands",
                    },
                    {
                        "rank": 2,
                        "area": "commodity_fill_evidence",
                        "target": "commodity-paper-execution:metals_all:XAGUSD",
                        "symbol": "XAGUSD",
                        "action": "wait_for_paper_execution_fill_evidence",
                        "reason": "paper_execution_fill_evidence_pending",
                        "state": "waiting",
                        "blocker_detail": "waiting fill evidence",
                        "done_when": "fill evidence lands",
                    },
                    {
                        "rank": 3,
                        "area": "crypto_route",
                        "target": "SOLUSDT",
                        "symbol": "SOLUSDT",
                        "action": "deprioritize_flow",
                        "reason": "no edge",
                        "state": "review",
                        "blocker_detail": "no_edge",
                        "done_when": "micro gate recovers",
                    },
                    {
                        "rank": 4,
                        "area": "ops_live_gate",
                        "target": "rollback_hard",
                        "symbol": "ROLLBACK_HARD",
                        "action": "clear_ops_live_gate_condition",
                        "reason": "clear hard rollback so ops_live_gate can leave rollback_now state",
                        "state": "repair",
                        "blocker_detail": "Current local operator head is XAUUSD, while remote automated live remains profitability_confirmed_but_auto_live_blocked.",
                        "done_when": "work through the queued remote live clearing conditions in rank order until the queue empties",
                    },
                ],
                "crypto_route_review_priority_queue": [
                    {
                        "rank": 1,
                        "symbol": "SOLUSDT",
                        "route_action": "deprioritize_flow",
                        "route_status_label": "review",
                        "priority_score": 73,
                        "priority_tier": "review_queue_now",
                        "reason": "no edge",
                        "blocker_detail": "no_edge",
                        "done_when": "micro gate recovers",
                    },
                    {
                        "rank": 2,
                        "symbol": "BNBUSDT",
                        "route_action": "watch_priority_until_long_window_confirms",
                        "route_status_label": "watch_priority",
                        "priority_score": 45,
                        "priority_tier": "review_queue_next",
                        "reason": "laggy edge",
                        "blocker_detail": "long window pending",
                        "done_when": "long window confirms",
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, list[str]]] = []
    refreshed_hot_brief_path = review_dir / "20260313T130006Z_hot_universe_operator_brief.json"

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append((step_name, list(cmd)))
        if "--now" in cmd:
            as_of = cmd[cmd.index("--now") + 1]
        else:
            as_of = "2026-03-13T13:00:00Z"
        if step_name == "refresh_brooks_structure_state":
            return {
                "ok": True,
                "status": "ok",
                "as_of": as_of,
                "artifact": str(review_dir / "brooks_structure_refresh.json"),
                "review_queue_artifact": str(brooks_review_queue_path),
            }
        if step_name == "refresh_commodity_paper_execution_state":
            return {
                "ok": True,
                "status": "ok",
                "as_of": as_of,
                "artifact": str(review_dir / "commodity_paper_execution_refresh.json"),
                "brief_artifact": str(hot_brief_path),
                "review_artifact": str(commodity_review_path),
            }
        if step_name == "refresh_hot_universe_operator_brief":
            refreshed_hot_brief_path.write_text(
                json.dumps({"status": "ok", "artifact": str(refreshed_hot_brief_path)}, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            return {
                "ok": True,
                "status": "ok",
                "as_of": as_of,
                "artifact": str(refreshed_hot_brief_path),
            }
        raise AssertionError(step_name)

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--context-path",
            str(context_path),
            "--now",
            "2026-03-13T13:00:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["downstream_refresh_mode"] == "full_refresh"
    assert payload["downstream_reused_count"] == 0
    assert payload["downstream_refresh_reuse_brief"] == "fresh_downstream_refresh:full_refresh:0/2"
    assert payload["hot_brief_artifact"] == str(refreshed_hot_brief_path)
    assert payload["hot_brief_refresh_status"] == "ok"
    assert payload["hot_brief_refresh_artifact"] == str(refreshed_hot_brief_path)
    assert payload["operator_backlog_status"] == "ready"
    assert payload["operator_backlog_count"] == 12
    assert payload["operator_head_area"] == "commodity_execution_close_evidence"
    assert payload["operator_head_symbol"] == "XAUUSD"
    assert payload["operator_head_action"] == "wait_for_paper_execution_close_evidence"
    assert payload["operator_head_state"] == "waiting"
    assert payload["operator_head_priority_score"] == 99
    assert payload["operator_head_priority_tier"] == "waiting_now"
    assert payload["operator_backlog_state_counts"] == {
        "waiting": 2,
        "review": 3,
        "watch": 0,
        "blocked": 0,
        "repair": 7,
    }
    assert payload["operator_backlog_state_brief"] == "waiting=2 | review=3 | watch=0 | blocked=0 | repair=7"
    assert payload["operator_backlog_priority_totals"] == {
        "waiting": 197,
        "review": 214,
        "watch": 0,
        "blocked": 0,
        "repair": 654,
    }
    assert payload["operator_backlog_priority_totals_brief"] == (
        "waiting=197 | review=214 | watch=0 | blocked=0 | repair=654"
    )
    assert payload["operator_state_lane_heads_brief"] == (
        "waiting:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
        "review:SOLUSDT:deprioritize_flow:73 | "
        "watch:-:-:0 | "
        "blocked:-:-:0 | "
        "repair:ROLLBACK_HARD:clear_ops_live_gate_condition:99"
    )
    assert payload["operator_state_lane_priority_order_brief"] == (
        "repair@654:7 > review@214:3 > waiting@197:2 > watch@0:0 > blocked@0:0"
    )
    assert payload["remote_live_operator_alignment_status"] == "local_operator_head_outside_remote_live_scope"
    assert payload["remote_live_operator_alignment_brief"] == (
        "local_operator_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um"
    )
    assert payload["remote_live_takeover_gate_status"] == "current_head_outside_remote_live_scope"
    assert payload["remote_live_takeover_gate_brief"] == (
        "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um"
    )
    assert "outside remote-live executable scope" in payload["remote_live_operator_alignment_blocker_detail"]
    assert "moves into remote-live executable scope" in payload["remote_live_operator_alignment_done_when"]
    assert payload["remote_live_takeover_clearing_status"] == "clearing_required"
    assert payload["remote_live_takeover_clearing_brief"] == "clearing_required:ops_live_gate+risk_guard"
    assert payload["remote_live_takeover_clearing_ops_live_gate_brief"] == (
        "rollback_hard, risk_violations, max_drawdown, slot_anomaly"
    )
    assert payload["remote_live_takeover_clearing_risk_guard_brief"] == (
        "ticket_missing:no_actionable_ticket, panic_cooldown_active, open_exposure_above_cap"
    )
    assert payload["remote_live_takeover_repair_queue_status"] == "ready"
    assert payload["remote_live_takeover_repair_queue_brief"] == "ready:ops_live_gate:rollback_hard:99"
    assert payload["remote_live_takeover_repair_queue_count"] == 7
    assert payload["remote_live_takeover_repair_queue_head_code"] == "rollback_hard"
    assert payload["remote_live_takeover_repair_queue_head_command"] == "cmd-gate"
    assert payload["operator_repair_queue_brief"] == (
        "1:ops_live_gate:rollback_hard:clear_ops_live_gate_condition | "
        "2:ops_live_gate:slot_anomaly:clear_ops_live_gate_condition | "
        "3:ops_live_gate:backtest_snapshot:clear_ops_live_gate_condition | "
        "4:ops_live_gate:ops_status_red:clear_ops_live_gate_condition | +3"
    )
    assert payload["operator_repair_queue_count"] == 7
    assert payload["operator_repair_checklist_brief"] == (
        "1:repair:ROLLBACK_HARD:clear_ops_live_gate_condition | "
        "2:repair:SLOT_ANOMALY:clear_ops_live_gate_condition | "
        "3:repair:BACKTEST_SNAPSHOT:clear_ops_live_gate_condition | "
        "4:repair:OPS_STATUS_RED:clear_ops_live_gate_condition | +3"
    )
    assert payload["operator_repair_queue"][0]["command"] == "cmd-gate"
    assert payload["operator_repair_checklist"][0]["state"] == "repair"
    assert "1:waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99" in payload["operator_backlog_brief"]
    assert "4:review:crypto_route:BNBUSDT:watch_priority_until_long_window_confirms:45" in payload["operator_backlog_brief"]
    assert "5:review:brooks_structure:SC2603:inspect_source_state:96" in payload["operator_backlog_brief"]
    assert "6:repair:ops_live_gate:ROLLBACK_HARD:clear_ops_live_gate_condition:99" in payload["operator_backlog_brief"]
    assert "10:repair:risk_guard:TICKET_MISSING_NO_ACTIONABLE_TICKET:clear_risk_guard_condition:89" in payload["operator_backlog_brief"]
    assert payload["operator_waiting_lane_status"] == "ready"
    assert payload["operator_waiting_lane_count"] == 2
    assert payload["operator_waiting_lane_brief"] == (
        "1:XAUUSD:wait_for_paper_execution_close_evidence:99 | "
        "2:XAGUSD:wait_for_paper_execution_fill_evidence:98"
    )
    assert payload["operator_waiting_lane_priority_total"] == 197
    assert payload["operator_waiting_lane_head_symbol"] == "XAUUSD"
    assert payload["operator_waiting_lane_head_action"] == "wait_for_paper_execution_close_evidence"
    assert payload["operator_waiting_lane_head_priority_score"] == 99
    assert payload["operator_waiting_lane_head_priority_tier"] == "waiting_now"
    assert payload["operator_review_lane_status"] == "ready"
    assert payload["operator_review_lane_count"] == 3
    assert payload["operator_review_lane_brief"] == (
        "3:SOLUSDT:deprioritize_flow:73 | 4:BNBUSDT:watch_priority_until_long_window_confirms:45 | 5:SC2603:inspect_source_state:96"
    )
    assert payload["operator_review_lane_priority_total"] == 214
    assert payload["operator_review_lane_head_symbol"] == "SOLUSDT"
    assert payload["operator_review_lane_head_action"] == "deprioritize_flow"
    assert payload["operator_review_lane_head_priority_score"] == 73
    assert payload["operator_review_lane_head_priority_tier"] == "review_queue_now"
    assert payload["operator_action_queue"][2]["reason"] == "no edge"
    assert payload["operator_action_checklist"][2]["reason"] == "no edge"
    assert payload["operator_action_checklist"][3]["state"] == "repair"
    assert payload["operator_focus_slots_brief"] == (
        "primary:waiting:XAUUSD:wait_for_paper_execution_close_evidence | "
        "followup:waiting:XAGUSD:wait_for_paper_execution_fill_evidence | "
        "secondary:review:SOLUSDT:deprioritize_flow"
    )
    assert payload["operator_focus_slots"][2]["slot"] == "secondary"
    assert payload["operator_focus_slots"][2]["reason"] == "no edge"
    assert payload["operator_focus_slots"][2]["priority_score"] == 73
    assert payload["operator_focus_slots"][2]["priority_tier"] == "review_queue_now"
    assert payload["operator_focus_slots"][2]["queue_rank"] == 3
    assert payload["operator_watch_lane_status"] == "inactive"
    assert payload["operator_watch_lane_count"] == 0
    assert payload["operator_watch_lane_brief"] == "-"
    assert payload["operator_watch_lane_priority_total"] == 0
    assert payload["operator_blocked_lane_status"] == "inactive"
    assert payload["operator_blocked_lane_count"] == 0
    assert payload["operator_blocked_lane_priority_total"] == 0
    assert payload["operator_blocked_lane_head_symbol"] == ""
    assert payload["operator_blocked_lane_head_action"] == ""
    assert payload["operator_blocked_lane_head_priority_score"] == 0
    assert payload["operator_blocked_lane_head_priority_tier"] == ""
    assert payload["operator_blocked_lane_brief"] == "-"
    assert payload["operator_repair_lane_status"] == "ready"
    assert payload["operator_repair_lane_count"] == 7
    assert payload["operator_repair_lane_priority_total"] == 654
    assert payload["operator_repair_lane_head_symbol"] == "ROLLBACK_HARD"
    assert payload["operator_repair_lane_head_action"] == "clear_ops_live_gate_condition"
    assert payload["operator_repair_lane_head_priority_score"] == 99
    assert payload["operator_repair_lane_head_priority_tier"] == "repair_queue_now"
    assert payload["operator_repair_head_lane_status"] == "ready"
    assert payload["operator_repair_head_lane_brief"] == "ready:ops_live_gate:rollback_hard:99"
    assert payload["operator_repair_head_lane_command"] == "cmd-gate"
    assert payload["operator_repair_head_lane_clear_when"] == (
        "clear hard rollback so ops_live_gate can leave rollback_now state"
    )
    assert payload["operator_repair_lane_brief"].startswith(
        "6:ROLLBACK_HARD:clear_ops_live_gate_condition:99 | 7:SLOT_ANOMALY:clear_ops_live_gate_condition:98"
    )
    assert payload["operator_head_lane_status"] == "waiting"
    assert payload["operator_head_lane_brief"] == (
        "waiting:commodity_execution_close_evidence:XAUUSD:wait_for_paper_execution_close_evidence:99"
    )
    assert payload["review_backlog_status"] == "ready"
    assert payload["review_backlog_count"] == 3
    assert payload["review_head_area"] == "crypto_route"
    assert payload["review_head_symbol"] == "SOLUSDT"
    assert payload["review_head_action"] == "deprioritize_flow"
    assert payload["review_head_priority_score"] == 73
    assert payload["review_head_priority_tier"] == "review_queue_now"
    assert payload["review_head_blocker_detail"] == "no_edge"
    assert payload["review_head_done_when"] == "micro gate recovers"
    assert payload["review_head_lane_status"] == "review"
    assert payload["review_head_lane_brief"] == "review:crypto_route:SOLUSDT:deprioritize_flow:73"
    assert payload["review_backlog"][1]["area"] == "crypto_route"
    assert payload["review_backlog"][1]["symbol"] == "BNBUSDT"
    assert payload["review_backlog"][1]["priority_score"] == 45
    assert payload["review_backlog"][2]["symbol"] == "SC2603"
    assert payload["review_backlog"][2]["priority_score"] == 96
    assert payload["review_backlog"][2]["promotion_ready"] is False
    assert payload["review_backlog"][2]["source_refresh_action"] == "inspect_source_state"
    assert payload["review_backlog_brief"] == (
        "1:crypto_route:SOLUSDT:review_queue_now:73 | "
        "2:crypto_route:BNBUSDT:review_queue_next:45 | "
        "3:brooks_structure:SC2603:review_queue_now:96"
    )
    assert Path(payload["artifact"]).exists()
    assert Path(payload["markdown"]).exists()
    assert Path(payload["checksum"]).exists()
    assert [row["name"] for row in payload["steps"]] == [
        "refresh_brooks_structure_state",
        "refresh_commodity_paper_execution_state",
        "refresh_hot_universe_operator_brief",
    ]
    assert payload["steps"][0]["status"] == "ok"
    assert payload["steps"][1]["status"] == "ok"
    assert payload["steps"][2]["status"] == "ok"
    assert payload["steps"][2]["artifact"] == str(refreshed_hot_brief_path)
    assert calls[0][0] == "refresh_brooks_structure_state"
    assert calls[0][1][1].endswith("refresh_brooks_structure_state.py")
    assert calls[1][0] == "refresh_commodity_paper_execution_state"
    assert calls[1][1][1].endswith("refresh_commodity_paper_execution_state.py")
    assert calls[2][0] == "refresh_hot_universe_operator_brief"
    assert calls[2][1][1].endswith("build_hot_universe_operator_brief.py")

    markdown = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "## Operator Head" in markdown
    assert "## Remote Live Operator Alignment" in markdown
    assert "## Remote Live Takeover Gate" in markdown
    assert "## Remote Live Takeover Clearing" in markdown
    assert "## Remote Live Takeover Repair Queue" in markdown
    assert "## Operator Repair Queue" in markdown
    assert "## Operator Focus Slots" in markdown
    assert "current_head_outside_remote_live_scope:commodity_execution_close_evidence:XAUUSD:portfolio_margin_um" in markdown
    assert "clearing_required:ops_live_gate+risk_guard" in markdown
    assert "ready:ops_live_gate:rollback_hard:99" in markdown
    assert "1:ops_live_gate:rollback_hard:clear_ops_live_gate_condition" in markdown
    assert "1:repair:ROLLBACK_HARD:clear_ops_live_gate_condition" in markdown
    assert "## Operator Backlog" in markdown
    assert "## Operator State Lanes" in markdown
    assert "waiting=197 | review=214 | watch=0 | blocked=0 | repair=654" in markdown
    assert "waiting:XAUUSD:wait_for_paper_execution_close_evidence:99 | review:SOLUSDT:deprioritize_flow:73 | watch:-:-:0 | blocked:-:-:0 | repair:ROLLBACK_HARD:clear_ops_live_gate_condition:99" in markdown
    assert "XAUUSD" in markdown
    assert "BNBUSDT" in markdown
    assert "ROLLBACK_HARD" in markdown
    assert "## Review Backlog" in markdown
    assert "SC2603" in markdown
    assert "SOLUSDT" in markdown
    assert "BNBUSDT" in markdown
    assert "## Review Head" in markdown
    assert "## Downstream Refresh Audit" in markdown
    assert "- mode: `full_refresh`" in markdown
    assert "- reuse: `fresh_downstream_refresh:full_refresh:0/2`" in markdown


def test_main_can_skip_downstream_refresh_and_reuse_latest_refresh_artifacts(
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    live_gate_blocker_path = review_dir / "20260313T124758Z_live_gate_blocker_report.json"
    live_gate_blocker_path.write_text(
        json.dumps(
            {
                "remote_live_diagnosis": {
                    "status": "profitability_confirmed_but_auto_live_blocked",
                    "brief": "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard",
                },
                "remote_live_takeover_clearing": {
                    "status": "clearing_required",
                    "brief": "clearing_required:ops_live_gate+risk_guard",
                    "blocker_detail": "ops+risk",
                    "done_when": "clear ops and risk",
                    "ops_live_gate_brief": "rollback_hard",
                    "risk_guard_brief": "ticket_missing:no_actionable_ticket",
                },
                "remote_live_takeover_repair_queue": {
                    "status": "ready",
                    "brief": "ready:ops_live_gate:rollback_hard:99",
                    "count": 1,
                    "head_area": "ops_live_gate",
                    "head_code": "rollback_hard",
                    "head_action": "clear_ops_live_gate_condition",
                    "head_priority_score": 99,
                    "head_priority_tier": "repair_queue_now",
                    "head_command": "cmd-gate",
                    "head_clear_when": "clear hard rollback",
                    "queue_brief": "1:ops_live_gate:rollback_hard:99",
                    "items": [
                        {
                            "rank": 1,
                            "area": "ops_live_gate",
                            "code": "rollback_hard",
                            "action": "clear_ops_live_gate_condition",
                            "priority_score": 99,
                            "priority_tier": "repair_queue_now",
                            "command": "cmd-gate",
                            "clear_when": "clear hard rollback",
                        }
                    ],
                    "done_when": "repair queue clears",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    commodity_review_path = review_dir / "20260313T124759Z_commodity_paper_execution_review.json"
    commodity_review_path.write_text(
        json.dumps(
            {
                "status": "paper-execution-close-evidence-pending-fill-remainder",
                "next_close_evidence_execution_symbol": "XAUUSD",
                "review_items": [
                    {
                        "queue_rank": 1,
                        "execution_id": "commodity-paper-execution:metals_all:XAUUSD",
                        "symbol": "XAUUSD",
                        "paper_open_date": "2026-03-11",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    commodity_gap_path = review_dir / "20260313T124800Z_commodity_paper_execution_gap_report.json"
    commodity_gap_path.write_text(
        json.dumps({"status": "ok"}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    crypto_route_refresh_path = review_dir / "20260313T124801Z_crypto_route_refresh.json"
    crypto_route_refresh_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "review_priority_queue": [
                    {
                        "rank": 1,
                        "symbol": "SOLUSDT",
                        "route_action": "deprioritize_flow",
                        "route_status_label": "review",
                        "priority_score": 73,
                        "priority_tier": "review_queue_now",
                        "reason": "no edge",
                        "blocker_detail": "no_edge",
                        "done_when": "micro gate recovers",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    hot_brief_path = review_dir / "20260313T124800Z_hot_universe_operator_brief.json"
    hot_brief_path.write_text(
        json.dumps(
            {
                "operator_status": "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch",
                "operator_stack_brief": "commodity:close-evidence:XAUUSD | crypto:SOLUSDT:deprioritize_flow",
                "source_live_gate_blocker_artifact": str(live_gate_blocker_path),
                "source_live_gate_blocker_remote_live_diagnosis_status": "profitability_confirmed_but_auto_live_blocked",
                "source_live_gate_blocker_remote_live_diagnosis_brief": (
                    "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                ),
                "source_live_gate_blocker_remote_live_takeover_clearing_status": "clearing_required",
                "source_live_gate_blocker_remote_live_takeover_clearing_brief": (
                    "clearing_required:ops_live_gate+risk_guard"
                ),
                "source_live_gate_blocker_remote_live_takeover_clearing_blocker_detail": "ops+risk",
                "source_live_gate_blocker_remote_live_takeover_clearing_done_when": "clear ops and risk",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_status": "ready",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_brief": "ready:ops_live_gate:rollback_hard:99",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_count": 1,
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_area": "ops_live_gate",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_code": "rollback_hard",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_action": "clear_ops_live_gate_condition",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_priority_score": 99,
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_priority_tier": "repair_queue_now",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_command": "cmd-gate",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_clear_when": "clear hard rollback",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_done_when": "repair queue clears",
                "brooks_structure_operator_status": "inactive",
                "secondary_focus_area": "crypto_route",
                "secondary_focus_symbol": "SOLUSDT",
                "secondary_focus_action": "deprioritize_flow",
                "secondary_focus_state": "review",
                "secondary_focus_priority_score": 73,
                "secondary_focus_priority_tier": "review_queue_now",
                "secondary_focus_blocker_detail": "no_edge",
                "secondary_focus_done_when": "micro gate recovers",
                "operator_action_checklist": [
                    {
                        "rank": 1,
                        "area": "commodity_execution_close_evidence",
                        "target": "commodity-paper-execution:metals_all:XAUUSD",
                        "symbol": "XAUUSD",
                        "action": "wait_for_paper_execution_close_evidence",
                        "state": "waiting",
                        "blocker_detail": "waiting close evidence",
                        "done_when": "close evidence lands",
                    },
                    {
                        "rank": 2,
                        "area": "crypto_route",
                        "target": "SOLUSDT",
                        "symbol": "SOLUSDT",
                        "action": "deprioritize_flow",
                        "state": "review",
                        "blocker_detail": "no_edge",
                        "done_when": "micro gate recovers",
                    },
                ],
                "crypto_route_review_priority_queue": [
                    {
                        "rank": 1,
                        "symbol": "SOLUSDT",
                        "route_action": "deprioritize_flow",
                        "route_status_label": "review",
                        "priority_score": 73,
                        "priority_tier": "review_queue_now",
                        "reason": "no edge",
                        "blocker_detail": "no_edge",
                        "done_when": "micro gate recovers",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    newer_hot_brief_path = review_dir / "20260313T125900Z_hot_universe_operator_brief.json"
    newer_hot_brief_path.write_text(
        json.dumps(
            {
                "operator_status": "commodity-paper-execution-close-evidence-pending-plus-crypto-deploy-watch",
                "operator_stack_brief": "commodity:close-evidence:XAUUSD | crypto:SOLUSDT:deprioritize_flow",
                "source_live_gate_blocker_artifact": str(live_gate_blocker_path),
                "source_live_gate_blocker_remote_live_diagnosis_status": "profitability_confirmed_but_auto_live_blocked",
                "source_live_gate_blocker_remote_live_diagnosis_brief": (
                    "profitability_confirmed_but_auto_live_blocked:portfolio_margin_um:ops_live_gate+risk_guard"
                ),
                "source_live_gate_blocker_remote_live_takeover_clearing_status": "clearing_required",
                "source_live_gate_blocker_remote_live_takeover_clearing_brief": (
                    "clearing_required:ops_live_gate+risk_guard"
                ),
                "source_live_gate_blocker_remote_live_takeover_clearing_blocker_detail": "ops+risk",
                "source_live_gate_blocker_remote_live_takeover_clearing_done_when": "clear ops and risk",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_status": "ready",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_brief": "ready:ops_live_gate:rollback_hard:99",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_count": 1,
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_area": "ops_live_gate",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_code": "rollback_hard",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_action": "clear_ops_live_gate_condition",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_priority_score": 99,
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_priority_tier": "repair_queue_now",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_command": "cmd-gate",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_head_clear_when": "clear hard rollback",
                "source_live_gate_blocker_remote_live_takeover_repair_queue_done_when": "repair queue clears",
                "brooks_structure_operator_status": "inactive",
                "secondary_focus_area": "crypto_route",
                "secondary_focus_symbol": "SOLUSDT",
                "secondary_focus_action": "deprioritize_flow",
                "secondary_focus_state": "review",
                "secondary_focus_priority_score": 58,
                "secondary_focus_priority_tier": "review_queue_next",
                "secondary_focus_blocker_detail": "proxy locality only",
                "secondary_focus_done_when": "micro gate recovers",
                "operator_action_checklist": [
                    {
                        "rank": 1,
                        "area": "commodity_execution_close_evidence",
                        "target": "commodity-paper-execution:metals_all:XAUUSD",
                        "symbol": "XAUUSD",
                        "action": "wait_for_paper_execution_close_evidence",
                        "state": "waiting",
                        "blocker_detail": "waiting close evidence",
                        "done_when": "close evidence lands",
                    },
                    {
                        "rank": 2,
                        "area": "crypto_route",
                        "target": "SOLUSDT",
                        "symbol": "SOLUSDT",
                        "action": "deprioritize_flow",
                        "state": "review",
                        "blocker_detail": "proxy locality only",
                        "done_when": "micro gate recovers",
                    },
                ],
                "crypto_route_review_priority_queue": [
                    {
                        "rank": 1,
                        "symbol": "SOLUSDT",
                        "route_action": "deprioritize_flow",
                        "route_status_label": "review",
                        "priority_score": 58,
                        "priority_tier": "review_queue_next",
                        "reason": "no edge",
                        "blocker_detail": "proxy locality only",
                        "done_when": "micro gate recovers",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    brooks_refresh_path = review_dir / "20260313T124800Z_brooks_structure_refresh.json"
    brooks_refresh_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "as_of": "2026-03-13T12:48:00Z",
                "artifact": str(brooks_refresh_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    commodity_refresh_path = review_dir / "20260313T124801Z_commodity_paper_execution_refresh.json"
    commodity_refresh_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "as_of": "2026-03-13T12:48:01Z",
                "artifact": str(commodity_refresh_path),
                "brief_artifact": str(hot_brief_path),
                "review_artifact": str(commodity_review_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-13T13:00:00Z",
            "--skip-downstream-refresh",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["downstream_refresh_mode"] == "skip_downstream_refresh"
    assert payload["downstream_reused_count"] == 2
    assert payload["downstream_refresh_reuse_brief"] == "reused_downstream_inputs:skip_downstream_refresh:2/2"
    assert payload["brooks_refresh_artifact"] == str(brooks_refresh_path)
    assert payload["commodity_refresh_artifact"] == str(commodity_refresh_path)
    assert payload["hot_brief_artifact"] == str(hot_brief_path)
    assert payload["steps"][0]["status"] == "reused_previous_artifact"
    assert payload["steps"][1]["status"] == "reused_previous_artifact"
    assert payload["steps"][0]["artifact"] == str(brooks_refresh_path)
    assert payload["steps"][1]["artifact"] == str(commodity_refresh_path)
    assert payload["operator_head_symbol"] == "XAUUSD"
    assert payload["operator_review_lane_head_symbol"] == "SOLUSDT"
    assert payload["operator_review_lane_head_priority_score"] == 73
    markdown = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "- downstream_refresh_mode: `skip_downstream_refresh`" in markdown
    assert "## Downstream Refresh Audit" in markdown
    assert "- reuse: `reused_downstream_inputs:skip_downstream_refresh:2/2`" in markdown
