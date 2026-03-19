from __future__ import annotations

from pathlib import Path
from typing import Any


def _compact_snapshot_parts(*parts: str) -> str:
    return " | ".join([part for part in (str(x).strip() for x in parts) if part and part != "-"])


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def build_cross_market_source_snapshot_chunk(
    source_payload: dict[str, Any] | None,
) -> dict[str, str]:
    payload = dict(source_payload or {})
    operator_snapshot_brief = _compact_snapshot_parts(
        str(payload.get("operator_backlog_status") or ""),
        "count=" + str(int(payload.get("operator_backlog_count") or 0)),
        "head="
        + str(payload.get("operator_head_symbol") or "-")
        + ":"
        + str(payload.get("operator_head_action") or "-"),
        "states=" + str(payload.get("operator_backlog_state_brief") or "-"),
        "order=" + str(payload.get("operator_state_lane_priority_order_brief") or "-"),
    )
    review_snapshot_brief = _compact_snapshot_parts(
        str(payload.get("review_backlog_status") or ""),
        "count=" + str(int(payload.get("review_backlog_count") or 0)),
        "head="
        + str(payload.get("review_head_symbol") or "-")
        + ":"
        + str(payload.get("review_head_action") or "-"),
        "priority="
        + str(payload.get("review_head_priority_score") or "-")
        + ":"
        + str(payload.get("review_head_priority_tier") or "-"),
        str(payload.get("review_backlog_brief") or ""),
    )
    remote_live_snapshot_brief = _compact_snapshot_parts(
        str(payload.get("remote_live_operator_alignment_brief") or ""),
        "gate=" + str(payload.get("remote_live_takeover_gate_brief") or "-"),
        "clearing=" + str(payload.get("remote_live_takeover_clearing_brief") or "-"),
    )
    snapshot_brief = _compact_snapshot_parts(
        str(payload.get("status") or ""),
        str(payload.get("as_of") or ""),
        ("operator=" + operator_snapshot_brief) if operator_snapshot_brief else "",
        ("review=" + review_snapshot_brief) if review_snapshot_brief else "",
        ("remote=" + remote_live_snapshot_brief) if remote_live_snapshot_brief else "",
    )
    return {
        "source_cross_market_operator_state_operator_snapshot_brief": operator_snapshot_brief,
        "source_cross_market_operator_state_review_snapshot_brief": review_snapshot_brief,
        "source_cross_market_operator_state_remote_live_snapshot_brief": remote_live_snapshot_brief,
        "source_cross_market_operator_state_snapshot_brief": snapshot_brief,
    }


def build_crypto_route_source_chunk(
    *,
    crypto_route_source_path: Path | None,
    crypto_route_source_payload: dict[str, Any] | None,
    crypto_route_refresh_source_path: Path | None,
    crypto_route_refresh_source_payload: dict[str, Any] | None,
    crypto_route_refresh_reuse_audit: dict[str, Any] | None,
    crypto_route_refresh_reuse_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    route_payload = dict(crypto_route_source_payload or {})
    reuse_audit = dict(crypto_route_refresh_reuse_audit or {})
    reuse_gate = dict(crypto_route_refresh_reuse_gate or {})
    refresh_payload = dict(crypto_route_refresh_source_payload or {})
    return {
        "source_crypto_route_artifact": str(crypto_route_source_path) if crypto_route_source_path else "",
        "source_crypto_route_status": str(route_payload.get("status") or ""),
        "source_crypto_route_refresh_artifact": str(crypto_route_refresh_source_path)
        if crypto_route_refresh_source_path
        else "",
        "source_crypto_route_refresh_status": str(refresh_payload.get("status") or ""),
        "source_crypto_route_refresh_as_of": str(refresh_payload.get("as_of") or ""),
        "source_crypto_route_refresh_native_mode": str(reuse_audit.get("native_refresh_mode") or ""),
        "source_crypto_route_refresh_native_step_count": reuse_audit.get("native_step_count"),
        "source_crypto_route_refresh_reused_native_count": reuse_audit.get("reused_native_count"),
        "source_crypto_route_refresh_missing_reused_count": reuse_audit.get("missing_reused_count"),
        "source_crypto_route_refresh_reuse_status": str(reuse_audit.get("reuse_status") or ""),
        "source_crypto_route_refresh_reuse_brief": str(reuse_audit.get("reuse_brief") or ""),
        "source_crypto_route_refresh_reuse_note": str(reuse_audit.get("reuse_note") or ""),
        "source_crypto_route_refresh_reuse_done_when": str(reuse_audit.get("done_when") or ""),
        "source_crypto_route_refresh_reuse_level": str(reuse_gate.get("level") or ""),
        "source_crypto_route_refresh_reuse_gate_status": str(reuse_gate.get("status") or ""),
        "source_crypto_route_refresh_reuse_gate_brief": str(reuse_gate.get("brief") or ""),
        "source_crypto_route_refresh_reuse_gate_blocking": reuse_gate.get("blocking"),
        "source_crypto_route_refresh_reuse_gate_blocker_detail": str(
            reuse_gate.get("blocker_detail") or ""
        ),
        "source_crypto_route_refresh_reuse_gate_done_when": str(
            reuse_gate.get("done_when") or ""
        ),
    }


def build_remote_live_source_chunk(
    *,
    remote_live_history_audit_source_path: Path | None,
    remote_live_history_audit_source_payload: dict[str, Any] | None,
    remote_live_history_window_brief: str,
    remote_live_history_snapshot: dict[str, Any] | None,
    remote_live_history_window_map: dict[int, dict[str, Any]] | None,
    remote_live_history_longest: dict[str, Any] | None,
    remote_live_handoff_source_path: Path | None,
    remote_live_handoff_source_payload: dict[str, Any] | None,
    remote_live_handoff_operator_payload: dict[str, Any] | None,
    live_gate_blocker_source_path: Path | None,
    live_gate_blocker_source_payload: dict[str, Any] | None,
    mapping_text_fn: Any,
) -> dict[str, Any]:
    history_payload = dict(remote_live_history_audit_source_payload or {})
    history_snapshot = dict(remote_live_history_snapshot or {})
    history_window_map = dict(remote_live_history_window_map or {})
    history_longest_payload = dict(remote_live_history_longest or {})
    handoff_payload = dict(remote_live_handoff_source_payload or {})
    handoff_operator = dict(remote_live_handoff_operator_payload or {})
    live_gate_payload = dict(live_gate_blocker_source_payload or {})
    account_scope_alignment = _as_dict(handoff_operator.get("account_scope_alignment"))
    live_decision = _as_dict(live_gate_payload.get("live_decision"))
    remote_live_diagnosis = _as_dict(live_gate_payload.get("remote_live_diagnosis"))
    remote_live_operator_alignment = _as_dict(live_gate_payload.get("remote_live_operator_alignment"))
    remote_live_takeover_clearing = _as_dict(live_gate_payload.get("remote_live_takeover_clearing"))
    remote_live_takeover_repair_queue = _as_dict(live_gate_payload.get("remote_live_takeover_repair_queue"))
    ops_live_gate_clearing = _as_dict(live_gate_payload.get("ops_live_gate_clearing"))
    risk_guard_clearing = _as_dict(live_gate_payload.get("risk_guard_clearing"))
    blocked_candidate = _as_dict(history_snapshot.get("blocked_candidate"))
    return {
        "source_remote_live_history_audit_artifact": str(remote_live_history_audit_source_path)
        if remote_live_history_audit_source_path
        else "",
        "source_remote_live_history_audit_status": str(history_payload.get("status") or ""),
        "source_remote_live_history_audit_as_of": str(history_payload.get("generated_at_utc") or ""),
        "source_remote_live_history_audit_market": str(history_payload.get("market") or ""),
        "source_remote_live_history_audit_window_brief": remote_live_history_window_brief,
        "source_remote_live_history_audit_quote_available": history_snapshot.get("quote_available"),
        "source_remote_live_history_audit_open_positions": history_snapshot.get("open_positions"),
        "source_remote_live_history_audit_risk_guard_status": str(
            history_snapshot.get("risk_guard_status") or ""
        ),
        "source_remote_live_history_audit_risk_guard_reasons": list(
            history_snapshot.get("risk_guard_reasons") or []
        ),
        "source_remote_live_history_audit_blocked_candidate_symbol": str(
            blocked_candidate.get("symbol") or ""
        ),
        "source_remote_live_history_audit_24h_closed_pnl": (
            history_window_map.get(24, {}) or {}
        ).get("closed_pnl"),
        "source_remote_live_history_audit_24h_trade_count": (
            history_window_map.get(24, {}) or {}
        ).get("trade_count"),
        "source_remote_live_history_audit_7d_closed_pnl": (
            history_window_map.get(168, {}) or {}
        ).get("closed_pnl"),
        "source_remote_live_history_audit_7d_trade_count": (
            history_window_map.get(168, {}) or {}
        ).get("trade_count"),
        "source_remote_live_history_audit_30d_closed_pnl": (
            history_window_map.get(720, {}) or {}
        ).get("closed_pnl"),
        "source_remote_live_history_audit_30d_trade_count": (
            history_window_map.get(720, {}) or {}
        ).get("trade_count"),
        "source_remote_live_history_audit_30d_symbol_pnl_brief": mapping_text_fn(
            dict(history_longest_payload.get("income_pnl_by_symbol") or {}),
            limit=8,
        ),
        "source_remote_live_history_audit_30d_day_pnl_brief": mapping_text_fn(
            dict(history_longest_payload.get("income_pnl_by_day") or {}),
            limit=8,
        ),
        "source_remote_live_handoff_artifact": str(remote_live_handoff_source_path)
        if remote_live_handoff_source_path
        else "",
        "source_remote_live_handoff_status": str(handoff_payload.get("status") or ""),
        "source_remote_live_handoff_as_of": str(handoff_payload.get("generated_at") or ""),
        "source_remote_live_handoff_state": str(handoff_operator.get("handoff_state") or ""),
        "source_remote_live_handoff_status_triplet": str(
            handoff_operator.get("operator_status_triplet") or ""
        ),
        "source_remote_live_handoff_ready_check_scope_market": str(
            handoff_operator.get("ready_check_scope_market") or ""
        ),
        "source_remote_live_handoff_ready_check_scope_brief": str(
            handoff_operator.get("ready_check_scope_brief") or ""
        ),
        "source_remote_live_handoff_account_scope_alignment_status": str(
            account_scope_alignment.get("status") or ""
        ),
        "source_remote_live_handoff_account_scope_alignment_brief": str(
            account_scope_alignment.get("brief") or ""
        ),
        "source_remote_live_handoff_account_scope_alignment_blocking": account_scope_alignment.get(
            "blocking"
        ),
        "source_remote_live_handoff_account_scope_alignment_blocker_detail": str(
            account_scope_alignment.get("blocker_detail") or ""
        ),
        "source_live_gate_blocker_artifact": str(live_gate_blocker_source_path)
        if live_gate_blocker_source_path
        else "",
        "source_live_gate_blocker_as_of": str(live_gate_payload.get("generated_at") or ""),
        "source_live_gate_blocker_live_decision": str(live_decision.get("current_decision") or ""),
        "source_live_gate_blocker_remote_live_diagnosis_status": str(
            remote_live_diagnosis.get("status") or ""
        ),
        "source_live_gate_blocker_remote_live_diagnosis_brief": str(
            remote_live_diagnosis.get("brief") or ""
        ),
        "source_live_gate_blocker_remote_live_diagnosis_blocker_detail": str(
            remote_live_diagnosis.get("blocker_detail") or ""
        ),
        "source_live_gate_blocker_remote_live_diagnosis_done_when": str(
            remote_live_diagnosis.get("done_when") or ""
        ),
        "source_live_gate_blocker_remote_live_operator_alignment_status": str(
            remote_live_operator_alignment.get("status") or ""
        ),
        "source_live_gate_blocker_remote_live_operator_alignment_brief": str(
            remote_live_operator_alignment.get("brief") or ""
        ),
        "source_live_gate_blocker_remote_live_operator_alignment_blocker_detail": str(
            remote_live_operator_alignment.get("blocker_detail") or ""
        ),
        "source_live_gate_blocker_remote_live_operator_alignment_done_when": str(
            remote_live_operator_alignment.get("done_when") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_clearing_status": str(
            remote_live_takeover_clearing.get("status") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_clearing_brief": str(
            remote_live_takeover_clearing.get("brief") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_clearing_blocker_detail": str(
            remote_live_takeover_clearing.get("blocker_detail") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_clearing_done_when": str(
            remote_live_takeover_clearing.get("done_when") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_status": str(
            remote_live_takeover_repair_queue.get("status") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_brief": str(
            remote_live_takeover_repair_queue.get("brief") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_queue_brief": str(
            remote_live_takeover_repair_queue.get("queue_brief") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_count": int(
            remote_live_takeover_repair_queue.get("count") or 0
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_head_area": str(
            remote_live_takeover_repair_queue.get("head_area") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_head_code": str(
            remote_live_takeover_repair_queue.get("head_code") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_head_action": str(
            remote_live_takeover_repair_queue.get("head_action") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_head_priority_score": int(
            remote_live_takeover_repair_queue.get("head_priority_score") or 0
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_head_priority_tier": str(
            remote_live_takeover_repair_queue.get("head_priority_tier") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_head_command": str(
            remote_live_takeover_repair_queue.get("head_command") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_head_clear_when": str(
            remote_live_takeover_repair_queue.get("head_clear_when") or ""
        ),
        "source_live_gate_blocker_remote_live_takeover_repair_queue_done_when": str(
            remote_live_takeover_repair_queue.get("done_when") or ""
        ),
        "source_live_gate_blocker_ops_live_gate_clearing_brief": str(
            ops_live_gate_clearing.get("conditions_brief") or ""
        ),
        "source_live_gate_blocker_risk_guard_clearing_brief": str(
            risk_guard_clearing.get("conditions_brief") or ""
        ),
    }


def build_brooks_source_chunk(
    *,
    brooks_route_report_source_path: Path | None,
    brooks_route_report_source_payload: dict[str, Any] | None,
    brooks_route_head: dict[str, Any] | None,
    brooks_execution_plan_source_path: Path | None,
    brooks_execution_plan_source_payload: dict[str, Any] | None,
    brooks_execution_head: dict[str, Any] | None,
    brooks_structure_review_queue_source_path: Path | None,
    brooks_structure_review_queue_source_payload: dict[str, Any] | None,
    brooks_structure_refresh_source_path: Path | None,
    brooks_structure_refresh_source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    route_payload = dict(brooks_route_report_source_payload or {})
    route_head = dict(brooks_route_head or {})
    execution_payload = dict(brooks_execution_plan_source_payload or {})
    execution_head = dict(brooks_execution_head or {})
    review_queue_payload = dict(brooks_structure_review_queue_source_payload or {})
    refresh_payload = dict(brooks_structure_refresh_source_payload or {})
    return {
        "source_brooks_route_report_artifact": str(brooks_route_report_source_path)
        if brooks_route_report_source_path
        else "",
        "source_brooks_route_report_status": str(route_payload.get("status") or ""),
        "source_brooks_route_report_as_of": str(route_payload.get("as_of") or ""),
        "source_brooks_route_report_selected_routes_brief": str(
            route_payload.get("selected_routes_brief") or ""
        ),
        "source_brooks_route_report_candidate_count": int(route_payload.get("candidate_count") or 0),
        "source_brooks_route_report_head_symbol": str(route_head.get("symbol") or ""),
        "source_brooks_route_report_head_strategy_id": str(route_head.get("strategy_id") or ""),
        "source_brooks_route_report_head_direction": str(route_head.get("direction") or ""),
        "source_brooks_route_report_head_bridge_status": str(
            route_head.get("route_bridge_status") or ""
        ),
        "source_brooks_route_report_head_blocker_detail": str(
            route_head.get("route_bridge_blocker_detail") or ""
        ),
        "source_brooks_execution_plan_artifact": str(brooks_execution_plan_source_path)
        if brooks_execution_plan_source_path
        else "",
        "source_brooks_execution_plan_status": str(execution_payload.get("status") or ""),
        "source_brooks_execution_plan_as_of": str(execution_payload.get("as_of") or ""),
        "source_brooks_execution_plan_actionable_count": int(
            execution_payload.get("actionable_count") or 0
        ),
        "source_brooks_execution_plan_blocked_count": int(
            execution_payload.get("blocked_count") or 0
        ),
        "source_brooks_execution_plan_head_symbol": str(execution_head.get("symbol") or ""),
        "source_brooks_execution_plan_head_strategy_id": str(
            execution_head.get("strategy_id") or ""
        ),
        "source_brooks_execution_plan_head_plan_status": str(
            execution_head.get("plan_status") or ""
        ),
        "source_brooks_execution_plan_head_execution_action": str(
            execution_head.get("execution_action") or ""
        ),
        "source_brooks_execution_plan_head_entry_price": execution_head.get("entry_price"),
        "source_brooks_execution_plan_head_stop_price": execution_head.get("stop_price"),
        "source_brooks_execution_plan_head_target_price": execution_head.get("target_price"),
        "source_brooks_execution_plan_head_rr_ratio": execution_head.get("rr_ratio"),
        "source_brooks_execution_plan_head_blocker_detail": str(
            execution_head.get("plan_blocker_detail") or ""
        ),
        "source_brooks_structure_review_queue_artifact": str(
            brooks_structure_review_queue_source_path
        )
        if brooks_structure_review_queue_source_path
        else "",
        "source_brooks_structure_review_queue_status": str(review_queue_payload.get("status") or ""),
        "source_brooks_structure_review_queue_as_of": str(review_queue_payload.get("as_of") or ""),
        "source_brooks_structure_review_queue_brief": str(
            review_queue_payload.get("priority_brief")
            or review_queue_payload.get("review_brief")
            or ""
        ),
        "source_brooks_structure_refresh_artifact": str(brooks_structure_refresh_source_path)
        if brooks_structure_refresh_source_path
        else "",
        "source_brooks_structure_refresh_status": str(refresh_payload.get("status") or ""),
        "source_brooks_structure_refresh_as_of": str(refresh_payload.get("as_of") or ""),
        "source_brooks_structure_refresh_brief": str(
            refresh_payload.get("review_brief") or refresh_payload.get("priority_brief") or ""
        ),
        "source_brooks_structure_refresh_queue_count": int(refresh_payload.get("queue_count") or 0),
        "source_brooks_structure_refresh_head_symbol": str(refresh_payload.get("head_symbol") or ""),
        "source_brooks_structure_refresh_head_action": str(refresh_payload.get("head_action") or ""),
        "source_brooks_structure_refresh_head_priority_score": refresh_payload.get(
            "head_priority_score"
        ),
    }


def build_cross_market_runtime_chunk(
    *,
    cross_market_operator_state_source_payload: dict[str, Any] | None,
    cross_market_operator_head_lane: dict[str, Any] | None,
    cross_market_operator_repair_head_lane: dict[str, Any] | None,
    cross_market_review_head_lane: dict[str, Any] | None,
    live_gate_blocker_source_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    source_payload = dict(cross_market_operator_state_source_payload or {})
    operator_head_lane = dict(cross_market_operator_head_lane or {})
    operator_head = _as_dict(operator_head_lane.get("head"))
    repair_head_lane = dict(cross_market_operator_repair_head_lane or {})
    repair_head = _as_dict(repair_head_lane.get("head"))
    review_head_lane = dict(cross_market_review_head_lane or {})
    review_head = _as_dict(review_head_lane.get("head"))
    live_gate_payload = dict(live_gate_blocker_source_payload or {})
    repair_queue = _as_dict(live_gate_payload.get("remote_live_takeover_repair_queue"))
    return {
        "cross_market_operator_head_status": str(operator_head_lane.get("status") or ""),
        "cross_market_operator_head_brief": str(operator_head_lane.get("brief") or ""),
        "cross_market_operator_head_area": str(operator_head.get("area") or ""),
        "cross_market_operator_head_symbol": str(operator_head.get("symbol") or ""),
        "cross_market_operator_head_action": str(operator_head.get("action") or ""),
        "cross_market_operator_head_state": str(
            operator_head.get("state") or operator_head_lane.get("state") or ""
        ),
        "cross_market_operator_head_priority_score": int(operator_head.get("priority_score") or 0),
        "cross_market_operator_head_priority_tier": str(operator_head.get("priority_tier") or ""),
        "cross_market_operator_head_blocker_detail": str(operator_head_lane.get("blocker_detail") or ""),
        "cross_market_operator_head_done_when": str(operator_head_lane.get("done_when") or ""),
        "cross_market_remote_live_takeover_gate_status": str(
            source_payload.get("remote_live_takeover_gate_status") or ""
        ),
        "cross_market_remote_live_takeover_gate_brief": str(
            source_payload.get("remote_live_takeover_gate_brief") or ""
        ),
        "cross_market_remote_live_takeover_gate_blocker_detail": str(
            source_payload.get("remote_live_takeover_gate_blocker_detail") or ""
        ),
        "cross_market_remote_live_takeover_gate_done_when": str(
            source_payload.get("remote_live_takeover_gate_done_when") or ""
        ),
        "cross_market_remote_live_takeover_clearing_status": str(
            source_payload.get("remote_live_takeover_clearing_status") or ""
        ),
        "cross_market_remote_live_takeover_clearing_brief": str(
            source_payload.get("remote_live_takeover_clearing_brief") or ""
        ),
        "cross_market_remote_live_takeover_clearing_blocker_detail": str(
            source_payload.get("remote_live_takeover_clearing_blocker_detail") or ""
        ),
        "cross_market_remote_live_takeover_clearing_done_when": str(
            source_payload.get("remote_live_takeover_clearing_done_when") or ""
        ),
        "cross_market_remote_live_takeover_clearing_source_freshness_brief": str(
            source_payload.get("remote_live_takeover_clearing_source_freshness_brief") or ""
        ),
        "cross_market_remote_live_takeover_slot_anomaly_breakdown_status": str(
            source_payload.get("remote_live_takeover_slot_anomaly_breakdown_status") or ""
        ),
        "cross_market_remote_live_takeover_slot_anomaly_breakdown_brief": str(
            source_payload.get("remote_live_takeover_slot_anomaly_breakdown_brief") or ""
        ),
        "cross_market_remote_live_takeover_slot_anomaly_breakdown_artifact": str(
            source_payload.get("remote_live_takeover_slot_anomaly_breakdown_artifact") or ""
        ),
        "cross_market_remote_live_takeover_slot_anomaly_breakdown_repair_focus": str(
            source_payload.get("remote_live_takeover_slot_anomaly_breakdown_repair_focus") or ""
        ),
        "remote_live_takeover_repair_queue_status": str(repair_queue.get("status") or ""),
        "remote_live_takeover_repair_queue_brief": str(repair_queue.get("brief") or ""),
        "remote_live_takeover_repair_queue_queue_brief": str(repair_queue.get("queue_brief") or ""),
        "remote_live_takeover_repair_queue_count": int(repair_queue.get("count") or 0),
        "remote_live_takeover_repair_queue_head_area": str(repair_queue.get("head_area") or ""),
        "remote_live_takeover_repair_queue_head_code": str(repair_queue.get("head_code") or ""),
        "remote_live_takeover_repair_queue_head_action": str(repair_queue.get("head_action") or ""),
        "remote_live_takeover_repair_queue_head_priority_score": int(
            repair_queue.get("head_priority_score") or 0
        ),
        "remote_live_takeover_repair_queue_head_priority_tier": str(
            repair_queue.get("head_priority_tier") or ""
        ),
        "remote_live_takeover_repair_queue_head_command": str(repair_queue.get("head_command") or ""),
        "remote_live_takeover_repair_queue_head_clear_when": str(
            repair_queue.get("head_clear_when") or ""
        ),
        "remote_live_takeover_repair_queue_done_when": str(repair_queue.get("done_when") or ""),
        "source_cross_market_operator_state_review_head_lane_status": str(
            source_payload.get("review_head_lane_status") or ""
        ),
        "source_cross_market_operator_state_review_head_lane_brief": str(
            source_payload.get("review_head_lane_brief") or ""
        ),
        "source_cross_market_operator_state_operator_head_lane_status": str(
            source_payload.get("operator_head_lane_status") or ""
        ),
        "source_cross_market_operator_state_operator_head_lane_brief": str(
            source_payload.get("operator_head_lane_brief") or ""
        ),
        "source_cross_market_operator_state_operator_repair_head_lane_status": str(
            source_payload.get("operator_repair_head_lane_status") or ""
        ),
        "source_cross_market_operator_state_operator_repair_head_lane_brief": str(
            source_payload.get("operator_repair_head_lane_brief") or ""
        ),
        "cross_market_operator_repair_head_status": str(repair_head_lane.get("status") or ""),
        "cross_market_operator_repair_head_brief": str(repair_head_lane.get("brief") or ""),
        "cross_market_operator_repair_head_area": str(repair_head.get("area") or ""),
        "cross_market_operator_repair_head_code": str(
            repair_head.get("target") or repair_head.get("symbol") or ""
        ),
        "cross_market_operator_repair_head_action": str(repair_head.get("action") or ""),
        "cross_market_operator_repair_head_priority_score": int(repair_head.get("priority_score") or 0),
        "cross_market_operator_repair_head_priority_tier": str(
            repair_head.get("priority_tier") or ""
        ),
        "cross_market_operator_repair_head_command": str(repair_head_lane.get("command") or ""),
        "cross_market_operator_repair_head_clear_when": str(
            repair_head_lane.get("clear_when") or ""
        ),
        "cross_market_operator_repair_head_done_when": str(repair_head_lane.get("done_when") or ""),
        "cross_market_operator_repair_backlog_status": str(repair_head_lane.get("status") or ""),
        "cross_market_operator_repair_backlog_brief": str(
            repair_head_lane.get("backlog_brief") or ""
        ),
        "cross_market_operator_repair_backlog_count": int(repair_head_lane.get("backlog_count") or 0),
        "cross_market_operator_repair_backlog_priority_total": int(
            repair_head_lane.get("priority_total")
            or source_payload.get("operator_repair_lane_priority_total")
            or 0
        ),
        "cross_market_operator_repair_backlog_done_when": str(
            repair_head_lane.get("done_when") or ""
        ),
        "source_cross_market_operator_state_operator_repair_queue_brief": str(
            source_payload.get("operator_repair_queue_brief") or ""
        ),
        "source_cross_market_operator_state_operator_repair_queue_count": int(
            source_payload.get("operator_repair_queue_count") or 0
        ),
        "source_cross_market_operator_state_operator_repair_checklist_brief": str(
            source_payload.get("operator_repair_checklist_brief") or ""
        ),
        "cross_market_operator_backlog_count": int(operator_head_lane.get("backlog_count") or 0),
        "cross_market_operator_backlog_brief": str(operator_head_lane.get("backlog_brief") or ""),
        "cross_market_operator_backlog_state_brief": str(
            source_payload.get("operator_backlog_state_brief") or ""
        ),
        "cross_market_operator_backlog_priority_totals_brief": str(
            source_payload.get("operator_backlog_priority_totals_brief") or ""
        ),
        "cross_market_operator_lane_heads_brief": str(
            source_payload.get("operator_state_lane_heads_brief") or ""
        ),
        "cross_market_operator_lane_priority_order_brief": str(
            source_payload.get("operator_state_lane_priority_order_brief") or ""
        ),
        "cross_market_operator_waiting_lane_status": str(
            source_payload.get("operator_waiting_lane_status") or ""
        ),
        "cross_market_operator_waiting_lane_count": int(
            source_payload.get("operator_waiting_lane_count") or 0
        ),
        "cross_market_operator_waiting_lane_brief": str(
            source_payload.get("operator_waiting_lane_brief") or ""
        ),
        "cross_market_operator_waiting_lane_priority_total": int(
            source_payload.get("operator_waiting_lane_priority_total") or 0
        ),
        "cross_market_operator_waiting_lane_head_symbol": str(
            source_payload.get("operator_waiting_lane_head_symbol") or ""
        ),
        "cross_market_operator_waiting_lane_head_action": str(
            source_payload.get("operator_waiting_lane_head_action") or ""
        ),
        "cross_market_operator_waiting_lane_head_priority_score": int(
            source_payload.get("operator_waiting_lane_head_priority_score") or 0
        ),
        "cross_market_operator_waiting_lane_head_priority_tier": str(
            source_payload.get("operator_waiting_lane_head_priority_tier") or ""
        ),
        "cross_market_operator_review_lane_status": str(
            source_payload.get("operator_review_lane_status") or ""
        ),
        "cross_market_operator_review_lane_count": int(
            source_payload.get("operator_review_lane_count") or 0
        ),
        "cross_market_operator_review_lane_brief": str(
            source_payload.get("operator_review_lane_brief") or ""
        ),
        "cross_market_operator_review_lane_priority_total": int(
            source_payload.get("operator_review_lane_priority_total") or 0
        ),
        "cross_market_operator_review_lane_head_symbol": str(
            source_payload.get("operator_review_lane_head_symbol") or ""
        ),
        "cross_market_operator_review_lane_head_action": str(
            source_payload.get("operator_review_lane_head_action") or ""
        ),
        "cross_market_operator_review_lane_head_priority_score": int(
            source_payload.get("operator_review_lane_head_priority_score") or 0
        ),
        "cross_market_operator_review_lane_head_priority_tier": str(
            source_payload.get("operator_review_lane_head_priority_tier") or ""
        ),
        "cross_market_operator_watch_lane_status": str(
            source_payload.get("operator_watch_lane_status") or ""
        ),
        "cross_market_operator_watch_lane_count": int(
            source_payload.get("operator_watch_lane_count") or 0
        ),
        "cross_market_operator_watch_lane_brief": str(
            source_payload.get("operator_watch_lane_brief") or ""
        ),
        "cross_market_operator_watch_lane_priority_total": int(
            source_payload.get("operator_watch_lane_priority_total") or 0
        ),
        "cross_market_operator_watch_lane_head_symbol": str(
            source_payload.get("operator_watch_lane_head_symbol") or ""
        ),
        "cross_market_operator_watch_lane_head_action": str(
            source_payload.get("operator_watch_lane_head_action") or ""
        ),
        "cross_market_operator_watch_lane_head_priority_score": int(
            source_payload.get("operator_watch_lane_head_priority_score") or 0
        ),
        "cross_market_operator_watch_lane_head_priority_tier": str(
            source_payload.get("operator_watch_lane_head_priority_tier") or ""
        ),
        "cross_market_operator_blocked_lane_status": str(
            source_payload.get("operator_blocked_lane_status") or ""
        ),
        "cross_market_operator_blocked_lane_count": int(
            source_payload.get("operator_blocked_lane_count") or 0
        ),
        "cross_market_operator_blocked_lane_brief": str(
            source_payload.get("operator_blocked_lane_brief") or ""
        ),
        "cross_market_operator_blocked_lane_priority_total": int(
            source_payload.get("operator_blocked_lane_priority_total") or 0
        ),
        "cross_market_operator_blocked_lane_head_symbol": str(
            source_payload.get("operator_blocked_lane_head_symbol") or ""
        ),
        "cross_market_operator_blocked_lane_head_action": str(
            source_payload.get("operator_blocked_lane_head_action") or ""
        ),
        "cross_market_operator_blocked_lane_head_priority_score": int(
            source_payload.get("operator_blocked_lane_head_priority_score") or 0
        ),
        "cross_market_operator_blocked_lane_head_priority_tier": str(
            source_payload.get("operator_blocked_lane_head_priority_tier") or ""
        ),
        "cross_market_operator_repair_lane_status": str(
            source_payload.get("operator_repair_lane_status") or ""
        ),
        "cross_market_operator_repair_lane_count": int(
            source_payload.get("operator_repair_lane_count") or 0
        ),
        "cross_market_operator_repair_lane_brief": str(
            source_payload.get("operator_repair_lane_brief") or ""
        ),
        "cross_market_operator_repair_lane_priority_total": int(
            source_payload.get("operator_repair_lane_priority_total") or 0
        ),
        "cross_market_operator_repair_lane_head_symbol": str(
            source_payload.get("operator_repair_lane_head_symbol") or ""
        ),
        "cross_market_operator_repair_lane_head_action": str(
            source_payload.get("operator_repair_lane_head_action") or ""
        ),
        "cross_market_operator_repair_lane_head_priority_score": int(
            source_payload.get("operator_repair_lane_head_priority_score") or 0
        ),
        "cross_market_operator_repair_lane_head_priority_tier": str(
            source_payload.get("operator_repair_lane_head_priority_tier") or ""
        ),
        "cross_market_review_head_status": str(review_head_lane.get("status") or ""),
        "cross_market_review_head_brief": str(review_head_lane.get("brief") or ""),
        "cross_market_review_head_area": str(review_head.get("area") or ""),
        "cross_market_review_head_symbol": str(review_head.get("symbol") or ""),
        "cross_market_review_head_action": str(review_head.get("action") or ""),
        "cross_market_review_head_priority_score": int(review_head.get("priority_score") or 0),
        "cross_market_review_head_priority_tier": str(review_head.get("priority_tier") or ""),
        "cross_market_review_head_blocker_detail": str(review_head_lane.get("blocker_detail") or ""),
        "cross_market_review_head_done_when": str(review_head_lane.get("done_when") or ""),
        "cross_market_review_backlog_count": int(review_head_lane.get("backlog_count") or 0),
        "cross_market_review_backlog_brief": str(review_head_lane.get("backlog_brief") or ""),
    }


def build_brooks_runtime_chunk(
    *,
    brooks_structure_review_lane: dict[str, Any] | None,
    brooks_structure_operator_lane: dict[str, Any] | None,
) -> dict[str, Any]:
    review_lane = dict(brooks_structure_review_lane or {})
    review_head = _as_dict(review_lane.get("head"))
    operator_lane = dict(brooks_structure_operator_lane or {})
    operator_head = _as_dict(operator_lane.get("head"))
    return {
        "brooks_structure_review_status": str(review_lane.get("status") or ""),
        "brooks_structure_review_brief": str(review_lane.get("brief") or ""),
        "brooks_structure_review_queue_status": str(review_lane.get("queue_status") or ""),
        "brooks_structure_review_queue_count": int(review_lane.get("queue_count") or 0),
        "brooks_structure_review_queue": [
            dict(row)
            for row in list(review_lane.get("queue") or [])
            if isinstance(row, dict)
        ],
        "brooks_structure_review_queue_brief": str(review_lane.get("queue_brief") or ""),
        "brooks_structure_review_priority_status": str(review_lane.get("priority_status") or ""),
        "brooks_structure_review_priority_brief": str(review_lane.get("priority_brief") or ""),
        "brooks_structure_review_head_rank": int(review_head.get("rank") or 0),
        "brooks_structure_review_head_symbol": str(review_head.get("symbol") or ""),
        "brooks_structure_review_head_strategy_id": str(review_head.get("strategy_id") or ""),
        "brooks_structure_review_head_direction": str(review_head.get("direction") or ""),
        "brooks_structure_review_head_tier": str(review_head.get("tier") or ""),
        "brooks_structure_review_head_plan_status": str(review_head.get("plan_status") or ""),
        "brooks_structure_review_head_action": str(review_head.get("execution_action") or ""),
        "brooks_structure_review_head_route_selection_score": review_head.get("route_selection_score"),
        "brooks_structure_review_head_signal_score": int(review_head.get("signal_score") or 0),
        "brooks_structure_review_head_signal_age_bars": int(review_head.get("signal_age_bars") or 0),
        "brooks_structure_review_head_priority_score": int(review_head.get("priority_score") or 0),
        "brooks_structure_review_head_priority_tier": str(review_head.get("priority_tier") or ""),
        "brooks_structure_review_head_blocker_detail": str(review_head.get("blocker_detail") or ""),
        "brooks_structure_review_head_done_when": str(review_head.get("done_when") or ""),
        "brooks_structure_review_blocker_detail": str(review_lane.get("blocker_detail") or ""),
        "brooks_structure_review_done_when": str(review_lane.get("done_when") or ""),
        "brooks_structure_operator_status": str(operator_lane.get("status") or ""),
        "brooks_structure_operator_brief": str(operator_lane.get("brief") or ""),
        "brooks_structure_operator_head_symbol": str(operator_head.get("symbol") or ""),
        "brooks_structure_operator_head_strategy_id": str(operator_head.get("strategy_id") or ""),
        "brooks_structure_operator_head_direction": str(operator_head.get("direction") or ""),
        "brooks_structure_operator_head_action": str(operator_head.get("execution_action") or ""),
        "brooks_structure_operator_head_plan_status": str(operator_head.get("plan_status") or ""),
        "brooks_structure_operator_head_priority_score": int(operator_head.get("priority_score") or 0),
        "brooks_structure_operator_head_priority_tier": str(operator_head.get("priority_tier") or ""),
        "brooks_structure_operator_backlog_count": int(operator_lane.get("backlog_count") or 0),
        "brooks_structure_operator_backlog_brief": str(operator_lane.get("backlog_brief") or ""),
        "brooks_structure_operator_blocker_detail": str(operator_lane.get("blocker_detail") or ""),
        "brooks_structure_operator_done_when": str(operator_lane.get("done_when") or ""),
    }


def build_research_embedding_quality_chunk(
    research_embedding_quality: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = dict(research_embedding_quality or {})
    return {
        "operator_research_embedding_quality_status": str(payload.get("status") or ""),
        "operator_research_embedding_quality_brief": str(payload.get("brief") or ""),
        "operator_research_embedding_quality_blocker_detail": str(
            payload.get("blocker_detail") or ""
        ),
        "operator_research_embedding_quality_done_when": str(payload.get("done_when") or ""),
        "operator_research_embedding_active_batches": list(payload.get("active_batches") or []),
        "operator_research_embedding_avoid_batches": list(payload.get("avoid_batches") or []),
        "operator_research_embedding_zero_trade_deprioritized_batches": list(
            payload.get("zero_trade_deprioritized_batches") or []
        ),
    }
