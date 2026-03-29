from __future__ import annotations

from typing import Any


def _compact_snapshot_parts(*parts: str) -> str:
    return " | ".join([part for part in (str(x).strip() for x in parts) if part and part != "-"])


def cross_market_source_debug_lines(payload: dict[str, Any]) -> list[str]:
    return [
        f"- source_cross_market_operator_state_artifact: `{payload.get('source_cross_market_operator_state_artifact') or ''}`",
        f"- source_cross_market_operator_state_status: `{payload.get('source_cross_market_operator_state_status') or ''}`",
        f"- source_cross_market_operator_state_as_of: `{payload.get('source_cross_market_operator_state_as_of') or ''}`",
        f"- source_cross_market_operator_state_snapshot_brief: `{payload.get('source_cross_market_operator_state_snapshot_brief') or ''}`",
        f"- source_cross_market_operator_state_operator_snapshot_brief: `{payload.get('source_cross_market_operator_state_operator_snapshot_brief') or ''}`",
        f"- source_cross_market_operator_state_review_snapshot_brief: `{payload.get('source_cross_market_operator_state_review_snapshot_brief') or ''}`",
        f"- source_cross_market_operator_state_remote_live_snapshot_brief: `{payload.get('source_cross_market_operator_state_remote_live_snapshot_brief') or ''}`",
        f"- source_cross_market_operator_state_operator_head_lane_brief: `{payload.get('source_cross_market_operator_state_operator_head_lane_brief') or ''}`",
        f"- source_cross_market_operator_state_review_head_lane_brief: `{payload.get('source_cross_market_operator_state_review_head_lane_brief') or ''}`",
        f"- source_cross_market_operator_state_operator_repair_head_lane_brief: `{payload.get('source_cross_market_operator_state_operator_repair_head_lane_brief') or ''}`",
        f"- source_cross_market_operator_state_operator_repair_queue_brief: `{payload.get('source_cross_market_operator_state_operator_repair_queue_brief') or ''}`",
        f"- source_cross_market_operator_state_operator_repair_checklist_brief: `{payload.get('source_cross_market_operator_state_operator_repair_checklist_brief') or ''}`",
    ]


def cross_market_runtime_debug_lines(payload: dict[str, Any]) -> list[str]:
    operator_snapshot = _compact_snapshot_parts(
        str(payload.get("cross_market_operator_head_status") or ""),
        str(payload.get("cross_market_operator_head_brief") or ""),
        "backlog="
        + str(payload.get("cross_market_operator_backlog_count") or 0)
        + ":"
        + str(payload.get("cross_market_operator_backlog_state_brief") or "-"),
        "order=" + str(payload.get("cross_market_operator_lane_priority_order_brief") or "-"),
    )
    remote_live_snapshot = _compact_snapshot_parts(
        str(payload.get("cross_market_remote_live_takeover_gate_brief") or ""),
        "clearing=" + str(payload.get("cross_market_remote_live_takeover_clearing_brief") or "-"),
    )
    repair_snapshot = _compact_snapshot_parts(
        str(payload.get("cross_market_operator_repair_head_brief") or ""),
        "backlog="
        + str(payload.get("cross_market_operator_repair_backlog_count") or 0)
        + ":"
        + str(payload.get("cross_market_operator_repair_backlog_priority_total") or 0),
        str(payload.get("cross_market_operator_repair_backlog_brief") or ""),
    )
    lane_snapshot = _compact_snapshot_parts(
        "waiting=" + str(payload.get("cross_market_operator_waiting_lane_brief") or "-"),
        "review=" + str(payload.get("cross_market_operator_review_lane_brief") or "-"),
        "watch=" + str(payload.get("cross_market_operator_watch_lane_brief") or "-"),
        "blocked=" + str(payload.get("cross_market_operator_blocked_lane_brief") or "-"),
        "repair=" + str(payload.get("cross_market_operator_repair_lane_brief") or "-"),
    )
    review_snapshot = _compact_snapshot_parts(
        str(payload.get("cross_market_review_head_brief") or ""),
        "backlog="
        + str(payload.get("cross_market_review_backlog_count") or 0)
        + ":"
        + str(payload.get("cross_market_review_backlog_brief") or "-"),
    )
    repair_queue_snapshot = _compact_snapshot_parts(
        str(payload.get("remote_live_takeover_repair_queue_status") or ""),
        "count=" + str(payload.get("remote_live_takeover_repair_queue_count") or 0),
        "head="
        + str(payload.get("remote_live_takeover_repair_queue_head_code") or "-")
        + ":"
        + str(payload.get("remote_live_takeover_repair_queue_head_action") or "-"),
    )
    return [
        f"- cross_market_operator_runtime_snapshot: `{operator_snapshot}`",
        f"- cross_market_remote_live_takeover_snapshot: `{remote_live_snapshot}`",
        f"- cross_market_repair_queue_snapshot: `{repair_queue_snapshot}`",
        f"- cross_market_repair_runtime_snapshot: `{repair_snapshot}`",
        f"- cross_market_lane_snapshot: `{lane_snapshot}`",
        f"- cross_market_review_runtime_snapshot: `{review_snapshot}`",
    ]


def remote_live_source_debug_lines(payload: dict[str, Any]) -> list[str]:
    history_snapshot = _compact_snapshot_parts(
        str(payload.get("source_remote_live_history_audit_status") or ""),
        str(payload.get("source_remote_live_history_audit_market") or ""),
        str(payload.get("source_remote_live_history_audit_window_brief") or ""),
        "quote=" + str(payload.get("source_remote_live_history_audit_quote_available") or "-"),
        "open_positions=" + str(payload.get("source_remote_live_history_audit_open_positions") or 0),
        "risk_guard=" + str(payload.get("source_remote_live_history_audit_risk_guard_status") or "-"),
        "blocked_candidate=" + str(payload.get("source_remote_live_history_audit_blocked_candidate_symbol") or "-"),
    )
    handoff_snapshot = _compact_snapshot_parts(
        str(payload.get("source_remote_live_handoff_status") or ""),
        str(payload.get("source_remote_live_handoff_state") or ""),
        "scope=" + str(payload.get("source_remote_live_handoff_ready_check_scope_brief") or "-"),
        "alignment=" + str(payload.get("source_remote_live_handoff_account_scope_alignment_brief") or "-"),
    )
    blocker_snapshot = _compact_snapshot_parts(
        str(payload.get("source_live_gate_blocker_live_decision") or ""),
        str(payload.get("source_live_gate_blocker_remote_live_diagnosis_brief") or ""),
        "alignment=" + str(payload.get("source_live_gate_blocker_remote_live_operator_alignment_brief") or "-"),
        "clearing=" + str(payload.get("source_live_gate_blocker_remote_live_takeover_clearing_brief") or "-"),
    )
    repair_snapshot = _compact_snapshot_parts(
        str(payload.get("source_live_gate_blocker_remote_live_takeover_repair_queue_status") or ""),
        "count=" + str(payload.get("source_live_gate_blocker_remote_live_takeover_repair_queue_count") or 0),
        "head="
        + str(payload.get("source_live_gate_blocker_remote_live_takeover_repair_queue_head_code") or "-")
        + ":"
        + str(payload.get("source_live_gate_blocker_remote_live_takeover_repair_queue_head_action") or "-"),
        str(payload.get("source_live_gate_blocker_remote_live_takeover_repair_queue_brief") or ""),
    )
    return [
        f"- source_remote_live_history_audit_artifact: `{payload.get('source_remote_live_history_audit_artifact') or ''}`",
        f"- source_remote_live_history_audit_as_of: `{payload.get('source_remote_live_history_audit_as_of') or ''}`",
        f"- source_remote_live_history_audit_snapshot: `{history_snapshot}`",
        f"- source_remote_live_handoff_artifact: `{payload.get('source_remote_live_handoff_artifact') or ''}`",
        f"- source_remote_live_handoff_as_of: `{payload.get('source_remote_live_handoff_as_of') or ''}`",
        f"- source_remote_live_handoff_snapshot: `{handoff_snapshot}`",
        f"- source_live_gate_blocker_artifact: `{payload.get('source_live_gate_blocker_artifact') or ''}`",
        f"- source_live_gate_blocker_as_of: `{payload.get('source_live_gate_blocker_as_of') or ''}`",
        f"- source_live_gate_blocker_snapshot: `{blocker_snapshot}`",
        f"- source_live_gate_blocker_repair_snapshot: `{repair_snapshot}`",
    ]


def brooks_source_debug_lines(payload: dict[str, Any]) -> list[str]:
    route_snapshot = _compact_snapshot_parts(
        str(payload.get("source_brooks_route_report_status") or ""),
        str(payload.get("source_brooks_route_report_selected_routes_brief") or ""),
        "head="
        + str(payload.get("source_brooks_route_report_head_symbol") or "-")
        + ":"
        + str(payload.get("source_brooks_route_report_head_strategy_id") or "-")
        + ":"
        + str(payload.get("source_brooks_route_report_head_direction") or "-"),
    )
    execution_snapshot = _compact_snapshot_parts(
        str(payload.get("source_brooks_execution_plan_status") or ""),
        "actionable=" + str(payload.get("source_brooks_execution_plan_actionable_count") or 0),
        "blocked=" + str(payload.get("source_brooks_execution_plan_blocked_count") or 0),
        "head="
        + str(payload.get("source_brooks_execution_plan_head_symbol") or "-")
        + ":"
        + str(payload.get("source_brooks_execution_plan_head_plan_status") or "-")
        + ":"
        + str(payload.get("source_brooks_execution_plan_head_execution_action") or "-"),
    )
    queue_snapshot = _compact_snapshot_parts(
        str(payload.get("source_brooks_structure_review_queue_status") or ""),
        str(payload.get("source_brooks_structure_review_queue_brief") or ""),
    )
    refresh_snapshot = _compact_snapshot_parts(
        str(payload.get("source_brooks_structure_refresh_status") or ""),
        str(payload.get("source_brooks_structure_refresh_brief") or ""),
        "head="
        + str(payload.get("source_brooks_structure_refresh_head_symbol") or "-")
        + ":"
        + str(payload.get("source_brooks_structure_refresh_head_action") or "-")
        + ":"
        + str(payload.get("source_brooks_structure_refresh_head_priority_score") or "-"),
    )
    return [
        f"- source_brooks_route_report_artifact: `{payload.get('source_brooks_route_report_artifact') or ''}`",
        f"- source_brooks_route_report_as_of: `{payload.get('source_brooks_route_report_as_of') or ''}`",
        f"- source_brooks_route_report_snapshot: `{route_snapshot}`",
        f"- source_brooks_execution_plan_artifact: `{payload.get('source_brooks_execution_plan_artifact') or ''}`",
        f"- source_brooks_execution_plan_as_of: `{payload.get('source_brooks_execution_plan_as_of') or ''}`",
        f"- source_brooks_execution_plan_snapshot: `{execution_snapshot}`",
        f"- source_brooks_structure_review_queue_artifact: `{payload.get('source_brooks_structure_review_queue_artifact') or ''}`",
        f"- source_brooks_structure_review_queue_as_of: `{payload.get('source_brooks_structure_review_queue_as_of') or ''}`",
        f"- source_brooks_structure_review_queue_snapshot: `{queue_snapshot}`",
        f"- source_brooks_structure_refresh_artifact: `{payload.get('source_brooks_structure_refresh_artifact') or ''}`",
        f"- source_brooks_structure_refresh_as_of: `{payload.get('source_brooks_structure_refresh_as_of') or ''}`",
        f"- source_brooks_structure_refresh_snapshot: `{refresh_snapshot}`",
    ]


def crypto_route_source_debug_lines(payload: dict[str, Any]) -> list[str]:
    route_snapshot = _compact_snapshot_parts(
        str(payload.get("source_crypto_route_status") or ""),
        str(payload.get("source_crypto_route_artifact") or ""),
    )
    refresh_snapshot = _compact_snapshot_parts(
        str(payload.get("source_crypto_route_refresh_status") or ""),
        str(payload.get("source_crypto_route_refresh_as_of") or ""),
        "mode=" + str(payload.get("source_crypto_route_refresh_native_mode") or "-"),
        "reuse=" + str(payload.get("source_crypto_route_refresh_reuse_brief") or "-"),
        "gate=" + str(payload.get("source_crypto_route_refresh_reuse_gate_brief") or "-"),
    )
    refresh_counts = _compact_snapshot_parts(
        "native_steps=" + str(payload.get("source_crypto_route_refresh_native_step_count") or 0),
        "reused=" + str(payload.get("source_crypto_route_refresh_reused_native_count") or 0),
        "missing=" + str(payload.get("source_crypto_route_refresh_missing_reused_count") or 0),
        "level=" + str(payload.get("source_crypto_route_refresh_reuse_level") or "-"),
    )
    return [
        f"- source_crypto_route_snapshot: `{route_snapshot}`",
        f"- source_crypto_route_refresh_artifact: `{payload.get('source_crypto_route_refresh_artifact') or ''}`",
        f"- source_crypto_route_refresh_snapshot: `{refresh_snapshot}`",
        f"- source_crypto_route_refresh_counts: `{refresh_counts}`",
        f"- source_crypto_route_refresh_done_when: `{payload.get('source_crypto_route_refresh_reuse_done_when') or ''}`",
        f"- source_crypto_route_refresh_gate_blocker_detail: `{payload.get('source_crypto_route_refresh_reuse_gate_blocker_detail') or ''}`",
    ]
