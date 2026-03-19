#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
TS_RE = re.compile(r"(?P<ts>\d{8}T\d{6}Z)")
FUTURE_STAMP_GRACE_MINUTES = 5


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_now(raw: str) -> dt.datetime | None:
    text_value = text(raw)
    if not text_value:
        return None
    for candidate in (text_value, text_value.replace("Z", "+00:00")):
        try:
            parsed = dt.datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_path_timestamp(path: Path) -> dt.datetime | None:
    match = TS_RE.search(path.name)
    if not match:
        return None
    try:
        return dt.datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def find_latest(
    review_dir: Path,
    pattern: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    files = list(review_dir.glob(pattern))
    if not files:
        return None
    effective_now = reference_now or now_utc()
    return max(
        files,
        key=lambda path: (
            0
            if (
                (stamp_dt := parse_path_timestamp(path)) is not None
                and stamp_dt > effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
            )
            else 1,
            fmt_utc(parse_path_timestamp(path)) or "",
            path.stat().st_mtime,
            path.name,
        ),
    )


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_openclaw_orderflow_blueprint.json",
        "*_openclaw_orderflow_blueprint.md",
        "*_openclaw_orderflow_blueprint_checksum.json",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing_candidates: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing_candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing_candidates, key=lambda item: item[0], reverse=True):
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def unwrap_operator_handoff(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("operator_handoff")
    return nested if isinstance(nested, dict) else payload


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def text(value: Any) -> str:
    return str(value or "").strip()


def dedupe_text(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def review_head_brief(cross_market: dict[str, Any]) -> str:
    head = as_dict(cross_market.get("review_head"))
    if head:
        status = text(head.get("status")) or "review"
        area = text(head.get("area"))
        symbol = text(head.get("symbol"))
        action = text(head.get("action"))
        score = text(head.get("priority_score"))
        return ":".join([status, area, symbol, action, score]).strip(":")
    return text(cross_market.get("review_head_brief"))


def queue_brief(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            symbol = text(row.get("symbol")) or text(row.get("target"))
            action = text(row.get("action"))
            if symbol and action:
                parts.append(f"{symbol}:{action}")
            elif symbol:
                parts.append(symbol)
        return " -> ".join(parts)
    return text(value)


def current_life_stage(
    handoff: dict[str, Any],
    live_gate: dict[str, Any],
    cross_market: dict[str, Any],
    *,
    identity_state: dict[str, Any] | None = None,
    scope_router_state: dict[str, Any] | None = None,
    intent_queue_state: dict[str, Any] | None = None,
    execution_journal_state: dict[str, Any] | None = None,
    executor_state: dict[str, Any] | None = None,
    feedback_state: dict[str, Any] | None = None,
    policy_state: dict[str, Any] | None = None,
    ack_state: dict[str, Any] | None = None,
    actor_state: dict[str, Any] | None = None,
    guarded_transport_state: dict[str, Any] | None = None,
    transport_sla_state: dict[str, Any] | None = None,
    canary_gate_state: dict[str, Any] | None = None,
    quality_report_state: dict[str, Any] | None = None,
    live_boundary_hold_state: dict[str, Any] | None = None,
    promotion_gate_state: dict[str, Any] | None = None,
    shadow_learning_continuity_state: dict[str, Any] | None = None,
    promotion_unblock_readiness_state: dict[str, Any] | None = None,
) -> str:
    alignment = text(cross_market.get("remote_live_operator_alignment_brief"))
    diagnosis = text(as_dict(handoff.get("remote_live_diagnosis")).get("brief"))
    blockers = as_list(live_gate.get("blockers"))
    blocker_names = {text(row.get("name")) for row in blockers if isinstance(row, dict)}
    readiness_state = as_dict(promotion_unblock_readiness_state)
    if text(readiness_state.get("readiness_brief")):
        if text(readiness_state.get("readiness_status")) == "shadow_ready_ticket_actionability_blocked":
            return "ticket_actionability_gated_remote_guardian"
        if text(readiness_state.get("readiness_status")).startswith(
            "local_time_sync_primary_blocker"
        ):
            return "local_repair_gated_continuity_hardened_remote_guardian"
        return "promotion_unblock_diagnosed_remote_guardian"
    if text(as_dict(shadow_learning_continuity_state).get("continuity_brief")):
        return "continuity_hardened_promotion_gated_remote_guardian"
    if text(as_dict(promotion_gate_state).get("promotion_gate_brief")):
        return "promotion_gated_remote_guardian"
    if text(as_dict(live_boundary_hold_state).get("hold_brief")):
        return "boundary_hardened_remote_guardian"
    if text(as_dict(quality_report_state).get("quality_brief")):
        return "quality_scored_remote_guardian"
    if text(as_dict(canary_gate_state).get("canary_gate_brief")):
        return "canary_gated_remote_guardian"
    if text(as_dict(transport_sla_state).get("transport_sla_brief")):
        return "transport_sla_shadow_remote_guardian"
    if text(as_dict(guarded_transport_state).get("guarded_transport_brief")):
        return "guarded_transport_preview_remote_guardian"
    if text(as_dict(actor_state).get("actor_brief")):
        return "actor_scaffolded_remote_guardian"
    if text(as_dict(ack_state).get("ack_brief")):
        return "ack_traced_remote_guardian"
    if text(as_dict(policy_state).get("policy_brief")):
        return "policy_scoped_remote_guardian"
    if text(as_dict(feedback_state).get("feedback_brief")):
        return "feedback_learning_remote_guardian"
    if text(as_dict(executor_state).get("executor_brief")):
        return "executor_scaffolded_remote_guardian"
    if text(as_dict(execution_journal_state).get("journal_brief")):
        return "journaled_remote_guardian"
    if text(as_dict(intent_queue_state).get("queue_brief")):
        return "intent_queued_remote_guardian"
    if text(as_dict(identity_state).get("identity_brief")) and text(
        as_dict(scope_router_state).get("scope_router_brief")
    ):
        return "identity_scoped_remote_guardian"
    if text(as_dict(identity_state).get("identity_brief")):
        return "identity_mapped_remote_guardian"
    if "ops_live_gate" in blocker_names or "risk_guard" in blocker_names:
        return "guarded_remote_guardian"
    if "outside_remote_live_scope" in alignment:
        return "scope_split_guardian"
    if "auto_live_blocked" in diagnosis:
        return "profitable_but_guarded_remote_actor"
    return "remote_actor_unknown"


def build_current_status(
    *,
    handoff_path: Path,
    handoff: dict[str, Any],
    live_gate_path: Path,
    live_gate: dict[str, Any],
    cross_market_path: Path,
    cross_market: dict[str, Any],
    hot_brief_path: Path,
    hot_brief: dict[str, Any],
    history_path: Path | None,
    history: dict[str, Any],
    identity_state_path: Path | None,
    identity_state: dict[str, Any],
    scope_router_state_path: Path | None,
    scope_router_state: dict[str, Any],
    intent_queue_state_path: Path | None,
    intent_queue_state: dict[str, Any],
    execution_journal_state_path: Path | None,
    execution_journal_state: dict[str, Any],
    executor_state_path: Path | None,
    executor_state: dict[str, Any],
    feedback_state_path: Path | None,
    feedback_state: dict[str, Any],
    policy_state_path: Path | None,
    policy_state: dict[str, Any],
    ack_state_path: Path | None,
    ack_state: dict[str, Any],
    actor_state_path: Path | None,
    actor_state: dict[str, Any],
    guarded_transport_state_path: Path | None,
    guarded_transport_state: dict[str, Any],
    transport_sla_state_path: Path | None,
    transport_sla_state: dict[str, Any],
    canary_gate_state_path: Path | None,
    canary_gate_state: dict[str, Any],
    quality_report_state_path: Path | None,
    quality_report_state: dict[str, Any],
    live_boundary_hold_state_path: Path | None,
    live_boundary_hold_state: dict[str, Any],
    promotion_gate_state_path: Path | None,
    promotion_gate_state: dict[str, Any],
    shadow_learning_continuity_state_path: Path | None,
    shadow_learning_continuity_state: dict[str, Any],
    promotion_unblock_readiness_state_path: Path | None,
    promotion_unblock_readiness_state: dict[str, Any],
    ticket_actionability_state_path: Path | None,
    ticket_actionability_state: dict[str, Any],
    signal_source_refresh_readiness_path: Path | None,
    signal_source_refresh_readiness: dict[str, Any],
    signal_source_freshness_path: Path | None,
    signal_source_freshness: dict[str, Any],
    material_change_trigger_path: Path | None,
    material_change_trigger: dict[str, Any],
    shortline_backtest_slice_path: Path | None,
    shortline_backtest_slice: dict[str, Any],
    shortline_cross_section_backtest_path: Path | None,
    shortline_cross_section_backtest: dict[str, Any],
    guardian_blocker_clearance_state_path: Path | None,
    guardian_blocker_clearance_state: dict[str, Any],
    shadow_clock_state_path: Path | None,
    shadow_clock_state: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    remote_diag = as_dict(handoff.get("remote_live_diagnosis"))
    live_decision = as_dict(live_gate.get("live_decision"))
    blockers = [
        {
            "name": text(row.get("name")),
            "status": text(row.get("status")),
            "reason_codes": dedupe_text([text(x) for x in as_list(row.get("reason_codes"))]),
        }
        for row in as_list(live_gate.get("blockers"))
        if isinstance(row, dict)
    ]
    return {
        "current_life_stage": current_life_stage(
            handoff,
            live_gate,
            cross_market,
            identity_state=identity_state,
            scope_router_state=scope_router_state,
            intent_queue_state=intent_queue_state,
            execution_journal_state=execution_journal_state,
            executor_state=executor_state,
            feedback_state=feedback_state,
            policy_state=policy_state,
            ack_state=ack_state,
            actor_state=actor_state,
            guarded_transport_state=guarded_transport_state,
            transport_sla_state=transport_sla_state,
            canary_gate_state=canary_gate_state,
            quality_report_state=quality_report_state,
            live_boundary_hold_state=live_boundary_hold_state,
            promotion_gate_state=promotion_gate_state,
            shadow_learning_continuity_state=shadow_learning_continuity_state,
            promotion_unblock_readiness_state=promotion_unblock_readiness_state,
        ),
        "target_life_stage": "orderflow_native_execution_organism",
        "remote_host": text(handoff.get("remote_host")),
        "remote_user": text(handoff.get("remote_user")),
        "remote_project_dir": text(handoff.get("remote_project_dir")),
        "handoff_state": text(handoff.get("handoff_state")),
        "handoff_age_hours": round(
            max(0.0, (reference_now - (parse_path_timestamp(handoff_path) or reference_now)).total_seconds() / 3600.0),
            3,
        ),
        "operator_status_quad": text(handoff.get("operator_status_quad")),
        "focus_stack_brief": text(handoff.get("focus_stack_brief")),
        "next_focus_reason": text(handoff.get("next_focus_reason")),
        "secondary_focus_reason": text(handoff.get("secondary_focus_reason")),
        "ready_check_scope_brief": text(handoff.get("ready_check_scope_brief")),
        "account_scope_alignment_brief": text(as_dict(handoff.get("account_scope_alignment")).get("brief"))
        or text(as_dict(history.get("account_scope_alignment")).get("brief")),
        "remote_live_diagnosis_brief": text(remote_diag.get("brief")),
        "remote_live_diagnosis_blocker_detail": text(remote_diag.get("blocker_detail")),
        "remote_profitability_window": text(remote_diag.get("profitability_window")),
        "remote_profitability_pnl": remote_diag.get("profitability_pnl"),
        "remote_profitability_trade_count": remote_diag.get("profitability_trade_count"),
        "live_decision_summary": text(live_decision.get("summary")),
        "live_decision": text(live_decision.get("current_decision")),
        "live_gate_operator_triplet": text(live_gate.get("operator_status_triplet")),
        "blockers": blockers,
        "cross_market_operator_alignment_brief": text(cross_market.get("remote_live_operator_alignment_brief")),
        "cross_market_takeover_gate_brief": text(cross_market.get("remote_live_takeover_gate_brief")),
        "cross_market_review_head_brief": review_head_brief(cross_market),
        "hot_operator_action_queue": queue_brief(hot_brief.get("operator_action_queue")),
        "history_window_brief": text(as_dict(handoff.get("remote_live_history")).get("window_brief"))
        or text(history.get("window_brief")),
        "remote_execution_identity_brief": text(identity_state.get("identity_brief")),
        "remote_scope_router_brief": text(scope_router_state.get("scope_router_brief")),
        "remote_scope_router_status": text(scope_router_state.get("scope_router_status")),
        "remote_intent_queue_brief": text(intent_queue_state.get("queue_brief")),
        "remote_intent_queue_status": text(intent_queue_state.get("queue_status")),
        "remote_intent_queue_recommendation": text(intent_queue_state.get("queue_recommendation")),
        "remote_execution_journal_brief": text(execution_journal_state.get("journal_brief")),
        "remote_execution_journal_status": text(execution_journal_state.get("journal_status")),
        "remote_execution_journal_append_status": text(execution_journal_state.get("append_status")),
        "openclaw_orderflow_executor_brief": text(executor_state.get("executor_brief")),
        "openclaw_orderflow_executor_status": text(executor_state.get("executor_status")),
        "openclaw_orderflow_executor_service_name": text(executor_state.get("service_name")),
        "remote_orderflow_feedback_brief": text(feedback_state.get("feedback_brief")),
        "remote_orderflow_feedback_status": text(feedback_state.get("feedback_status")),
        "remote_orderflow_feedback_recommendation": text(
            feedback_state.get("feedback_recommendation")
        ),
        "remote_orderflow_policy_brief": text(policy_state.get("policy_brief")),
        "remote_orderflow_policy_status": text(policy_state.get("policy_status")),
        "remote_orderflow_policy_decision": text(policy_state.get("policy_decision")),
        "remote_execution_ack_brief": text(ack_state.get("ack_brief")),
        "remote_execution_ack_status": text(ack_state.get("ack_status")),
        "remote_execution_ack_decision": text(ack_state.get("ack_decision")),
        "remote_execution_actor_brief": text(actor_state.get("actor_brief")),
        "remote_execution_actor_status": text(actor_state.get("actor_status")),
        "remote_execution_actor_service_name": text(actor_state.get("actor_service_name")),
        "remote_execution_actor_transport_phase": text(actor_state.get("transport_phase")),
        "remote_execution_guarded_transport_brief": text(
            guarded_transport_state.get("guarded_transport_brief")
        ),
        "remote_execution_guarded_transport_status": text(
            guarded_transport_state.get("guarded_transport_status")
        ),
        "remote_execution_guarded_transport_decision": text(
            guarded_transport_state.get("guarded_transport_decision")
        ),
        "remote_execution_transport_sla_brief": text(transport_sla_state.get("transport_sla_brief")),
        "remote_execution_transport_sla_status": text(transport_sla_state.get("transport_sla_status")),
        "remote_execution_transport_sla_decision": text(
            transport_sla_state.get("transport_sla_decision")
        ),
        "remote_execution_actor_canary_gate_brief": text(canary_gate_state.get("canary_gate_brief")),
        "remote_execution_actor_canary_gate_status": text(
            canary_gate_state.get("canary_gate_status")
        ),
        "remote_execution_actor_canary_gate_decision": text(
            canary_gate_state.get("canary_gate_decision")
        ),
        "remote_orderflow_quality_report_brief": text(quality_report_state.get("quality_brief")),
        "remote_orderflow_quality_report_status": text(quality_report_state.get("quality_status")),
        "remote_orderflow_quality_report_recommendation": text(
            quality_report_state.get("quality_recommendation")
        ),
        "remote_orderflow_quality_report_score": quality_report_state.get("quality_score"),
        "remote_orderflow_quality_shadow_learning_score": quality_report_state.get(
            "shadow_learning_score"
        ),
        "remote_orderflow_quality_execution_readiness_score": quality_report_state.get(
            "execution_readiness_score"
        ),
        "remote_orderflow_quality_transport_observability_score": quality_report_state.get(
            "transport_observability_score"
        ),
        "remote_live_boundary_hold_brief": text(live_boundary_hold_state.get("hold_brief")),
        "remote_live_boundary_hold_status": text(live_boundary_hold_state.get("hold_status")),
        "remote_live_boundary_hold_decision": text(live_boundary_hold_state.get("hold_decision")),
        "remote_live_boundary_hold_next_transition": text(
            live_boundary_hold_state.get("next_transition")
        ),
        "remote_live_boundary_hold_guardian_blocked": bool(
            live_boundary_hold_state.get("guardian_blocked", False)
        ),
        "remote_live_boundary_hold_review_blocked": bool(
            live_boundary_hold_state.get("review_blocked", False)
        ),
        "remote_live_boundary_hold_time_sync_blocked": bool(
            live_boundary_hold_state.get("time_sync_blocked", False)
        ),
        "remote_guarded_canary_promotion_gate_brief": text(
            promotion_gate_state.get("promotion_gate_brief")
        ),
        "remote_guarded_canary_promotion_gate_status": text(
            promotion_gate_state.get("promotion_gate_status")
        ),
        "remote_guarded_canary_promotion_gate_decision": text(
            promotion_gate_state.get("promotion_gate_decision")
        ),
        "remote_guarded_canary_promotion_gate_shadow_learning_decision": text(
            promotion_gate_state.get("shadow_learning_decision")
        ),
        "remote_guarded_canary_promotion_gate_promotion_ready": bool(
            promotion_gate_state.get("promotion_ready", False)
        ),
        "remote_guarded_canary_promotion_gate_blocker_code": text(
            promotion_gate_state.get("promotion_blocker_code")
        ),
        "remote_guarded_canary_promotion_gate_blocker_title": text(
            promotion_gate_state.get("promotion_blocker_title")
        ),
        "remote_guarded_canary_promotion_gate_blocker_target_artifact": text(
            promotion_gate_state.get("promotion_blocker_target_artifact")
        ),
        "remote_guarded_canary_promotion_gate_blocker_detail": text(
            promotion_gate_state.get("promotion_blocker_detail")
        ),
        "remote_shadow_learning_continuity_brief": text(
            shadow_learning_continuity_state.get("continuity_brief")
        ),
        "remote_shadow_learning_continuity_status": text(
            shadow_learning_continuity_state.get("continuity_status")
        ),
        "remote_shadow_learning_continuity_decision": text(
            shadow_learning_continuity_state.get("continuity_decision")
        ),
        "remote_shadow_learning_continuity_blocker_detail": text(
            shadow_learning_continuity_state.get("blocker_detail")
        ),
        "remote_promotion_unblock_readiness_brief": text(
            promotion_unblock_readiness_state.get("readiness_brief")
        ),
        "remote_promotion_unblock_readiness_status": text(
            promotion_unblock_readiness_state.get("readiness_status")
        ),
        "remote_promotion_unblock_readiness_decision": text(
            promotion_unblock_readiness_state.get("readiness_decision")
        ),
        "remote_promotion_unblock_preconditions_status": text(
            promotion_unblock_readiness_state.get("remote_preconditions_status")
        ),
        "remote_promotion_unblock_primary_blocker_scope": text(
            promotion_unblock_readiness_state.get("primary_blocker_scope")
        ),
        "remote_promotion_unblock_primary_local_repair_required": bool(
            promotion_unblock_readiness_state.get("primary_local_repair_required", False)
        ),
        "remote_promotion_unblock_primary_local_repair_title": text(
            promotion_unblock_readiness_state.get("primary_local_repair_title")
        ),
        "remote_promotion_unblock_primary_local_repair_target_artifact": text(
            promotion_unblock_readiness_state.get("primary_local_repair_target_artifact")
        ),
        "remote_promotion_unblock_primary_local_repair_detail": text(
            promotion_unblock_readiness_state.get("primary_local_repair_detail")
        ),
        "remote_promotion_unblock_primary_local_repair_plan_brief": text(
            promotion_unblock_readiness_state.get("primary_local_repair_plan_brief")
        ),
        "remote_promotion_unblock_primary_local_repair_environment_classification": text(
            promotion_unblock_readiness_state.get(
                "primary_local_repair_environment_classification"
            )
        ),
        "remote_promotion_unblock_primary_local_repair_environment_blocker_detail": text(
            promotion_unblock_readiness_state.get(
                "primary_local_repair_environment_blocker_detail"
            )
        ),
        "remote_promotion_unblock_primary_local_repair_environment_remediation_hint": text(
            promotion_unblock_readiness_state.get(
                "primary_local_repair_environment_remediation_hint"
            )
        ),
        "remote_ticket_actionability_brief": text(
            ticket_actionability_state.get("ticket_actionability_brief")
        ),
        "remote_ticket_actionability_status": text(
            ticket_actionability_state.get("ticket_actionability_status")
        ),
        "remote_ticket_actionability_decision": text(
            ticket_actionability_state.get("ticket_actionability_decision")
        ),
        "remote_ticket_actionability_next_action": text(
            ticket_actionability_state.get("next_action")
        ),
        "remote_ticket_actionability_next_action_target_artifact": text(
            ticket_actionability_state.get("next_action_target_artifact")
        ),
        "crypto_signal_source_refresh_readiness_brief": text(
            signal_source_refresh_readiness.get("readiness_brief")
        ),
        "crypto_signal_source_refresh_readiness_status": text(
            signal_source_refresh_readiness.get("readiness_status")
        ),
        "crypto_signal_source_refresh_readiness_decision": text(
            signal_source_refresh_readiness.get("readiness_decision")
        ),
        "crypto_signal_source_refresh_needed": bool(
            signal_source_refresh_readiness.get("refresh_needed", False)
        ),
        "crypto_signal_source_freshness_brief": text(
            signal_source_freshness.get("freshness_brief")
        ),
        "crypto_signal_source_freshness_status": text(
            signal_source_freshness.get("freshness_status")
        ),
        "crypto_signal_source_freshness_decision": text(
            signal_source_freshness.get("freshness_decision")
        ),
        "crypto_signal_source_freshness_refresh_recommended": bool(
            signal_source_freshness.get("refresh_recommended", False)
        ),
        "crypto_shortline_material_change_trigger_brief": text(
            material_change_trigger.get("trigger_brief")
        ),
        "crypto_shortline_material_change_trigger_status": text(
            material_change_trigger.get("trigger_status")
        ),
        "crypto_shortline_material_change_trigger_decision": text(
            material_change_trigger.get("trigger_decision")
        ),
        "crypto_shortline_material_change_trigger_rerun_recommended": bool(
            material_change_trigger.get("rerun_recommended", False)
        ),
        "crypto_shortline_backtest_slice_brief": text(
            shortline_backtest_slice.get("slice_brief")
        ),
        "crypto_shortline_backtest_slice_status": text(
            shortline_backtest_slice.get("slice_status")
        ),
        "crypto_shortline_backtest_slice_decision": text(
            shortline_backtest_slice.get("research_decision")
        ),
        "crypto_shortline_backtest_slice_selected_symbol": text(
            shortline_backtest_slice.get("selected_symbol")
        ),
        "crypto_shortline_backtest_slice_universe_brief": text(
            shortline_backtest_slice.get("slice_universe_brief")
        ),
        "crypto_shortline_cross_section_backtest_brief": text(
            shortline_cross_section_backtest.get("backtest_brief")
        ),
        "crypto_shortline_cross_section_backtest_status": text(
            shortline_cross_section_backtest.get("backtest_status")
        ),
        "crypto_shortline_cross_section_backtest_decision": text(
            shortline_cross_section_backtest.get("research_decision")
        ),
        "crypto_shortline_cross_section_backtest_selected_edge_status": text(
            shortline_cross_section_backtest.get("selected_edge_status")
        ),
        "remote_time_sync_mode": text(live_boundary_hold_state.get("time_sync_mode"))
        or text(promotion_gate_state.get("time_sync_mode"))
        or text(guardian_blocker_clearance_state.get("time_sync_mode")),
        "remote_shadow_clock_evidence_brief": text(shadow_clock_state.get("evidence_brief"))
        or text(live_boundary_hold_state.get("remote_shadow_clock_evidence_brief"))
        or text(guardian_blocker_clearance_state.get("remote_shadow_clock_evidence_brief")),
        "remote_shadow_clock_evidence_status": text(shadow_clock_state.get("evidence_status"))
        or text(live_boundary_hold_state.get("remote_shadow_clock_evidence_status"))
        or text(guardian_blocker_clearance_state.get("remote_shadow_clock_evidence_status")),
        "remote_shadow_clock_shadow_learning_allowed": bool(
            shadow_clock_state.get("shadow_learning_allowed", False)
        )
        or bool(live_boundary_hold_state.get("remote_shadow_clock_shadow_learning_allowed", False))
        or bool(
            guardian_blocker_clearance_state.get(
                "remote_shadow_clock_shadow_learning_allowed", False
            )
        ),
        "remote_guardian_blocker_clearance_brief": text(
            guardian_blocker_clearance_state.get("clearance_brief")
        ),
        "remote_guardian_blocker_clearance_status": text(
            guardian_blocker_clearance_state.get("clearance_status")
        ),
        "remote_guardian_blocker_clearance_score": guardian_blocker_clearance_state.get(
            "clearance_score"
        ),
        "remote_guardian_blocker_clearance_top_blocker_code": text(
            guardian_blocker_clearance_state.get("top_blocker_code")
        ),
        "remote_guardian_blocker_clearance_top_blocker_title": text(
            guardian_blocker_clearance_state.get("top_blocker_title")
        ),
        "remote_guardian_blocker_clearance_top_blocker_target_artifact": text(
            guardian_blocker_clearance_state.get("top_blocker_target_artifact")
        ),
        "remote_guardian_blocker_clearance_top_blocker_next_action": text(
            guardian_blocker_clearance_state.get("top_blocker_next_action")
        ),
        "remote_guardian_blocker_clearance_top_blocker_done_when": text(
            guardian_blocker_clearance_state.get("top_blocker_done_when")
        ),
        "remote_guardian_blocker_clearance_top_blocker_detail": text(
            guardian_blocker_clearance_state.get("top_blocker_detail")
        ),
        "artifacts": {
            "remote_live_handoff": str(handoff_path),
            "live_gate_blocker": str(live_gate_path),
            "cross_market_operator_state": str(cross_market_path),
            "hot_universe_operator_brief": str(hot_brief_path),
            "remote_live_history_audit": str(history_path) if history_path else "",
            "remote_execution_identity_state": str(identity_state_path) if identity_state_path else "",
            "remote_scope_router_state": str(scope_router_state_path) if scope_router_state_path else "",
            "remote_intent_queue": str(intent_queue_state_path) if intent_queue_state_path else "",
            "remote_execution_journal": str(execution_journal_state_path) if execution_journal_state_path else "",
            "openclaw_orderflow_executor_state": str(executor_state_path) if executor_state_path else "",
            "remote_orderflow_feedback": str(feedback_state_path) if feedback_state_path else "",
            "remote_orderflow_policy_state": str(policy_state_path) if policy_state_path else "",
            "remote_execution_ack_state": str(ack_state_path) if ack_state_path else "",
            "remote_execution_actor_state": str(actor_state_path) if actor_state_path else "",
            "remote_execution_actor_guarded_transport": str(guarded_transport_state_path)
            if guarded_transport_state_path
            else "",
            "remote_execution_transport_sla": str(transport_sla_state_path)
            if transport_sla_state_path
            else "",
            "remote_execution_actor_canary_gate": str(canary_gate_state_path)
            if canary_gate_state_path
            else "",
            "remote_orderflow_quality_report": str(quality_report_state_path)
            if quality_report_state_path
            else "",
            "remote_live_boundary_hold": str(live_boundary_hold_state_path)
            if live_boundary_hold_state_path
            else "",
            "remote_guarded_canary_promotion_gate": str(promotion_gate_state_path)
            if promotion_gate_state_path
            else "",
            "remote_shadow_learning_continuity": str(shadow_learning_continuity_state_path)
            if shadow_learning_continuity_state_path
            else "",
            "remote_promotion_unblock_readiness": str(promotion_unblock_readiness_state_path)
            if promotion_unblock_readiness_state_path
            else "",
            "remote_ticket_actionability_state": str(ticket_actionability_state_path)
            if ticket_actionability_state_path
            else "",
            "crypto_signal_source_refresh_readiness": str(signal_source_refresh_readiness_path)
            if signal_source_refresh_readiness_path
            else "",
            "crypto_signal_source_freshness": str(signal_source_freshness_path)
            if signal_source_freshness_path
            else "",
            "crypto_shortline_material_change_trigger": str(material_change_trigger_path)
            if material_change_trigger_path
            else "",
            "crypto_shortline_backtest_slice": str(shortline_backtest_slice_path)
            if shortline_backtest_slice_path
            else "",
            "crypto_shortline_cross_section_backtest": str(shortline_cross_section_backtest_path)
            if shortline_cross_section_backtest_path
            else "",
            "remote_shadow_clock_evidence": str(shadow_clock_state_path)
            if shadow_clock_state_path
            else "",
            "remote_guardian_blocker_clearance": str(guardian_blocker_clearance_state_path)
            if guardian_blocker_clearance_state_path
            else "",
        },
    }


def build_digital_layers(current: dict[str, Any]) -> list[dict[str, Any]]:
    perception_organs = [
        "remote_live_handoff",
        "live_gate_blocker_report",
        "remote_live_history_audit",
        "cross_market_operator_state",
    ]
    perception_missing = [
        "remote_execution_identity_state",
        "remote_scope_router_state",
        "remote_orderflow_frame",
    ]
    if text(current.get("remote_execution_identity_brief")):
        perception_organs.append("remote_execution_identity_state")
        perception_missing = [item for item in perception_missing if item != "remote_execution_identity_state"]
    if text(current.get("remote_scope_router_brief")):
        perception_organs.append("remote_scope_router_state")
        perception_missing = [item for item in perception_missing if item != "remote_scope_router_state"]
    layers = [
        {
            "layer": "perception",
            "mission": "Turn venue/account/ops telemetry into a single orderflow picture.",
            "current_organs": perception_organs,
            "missing_organs": perception_missing,
            "why_now": (
                "OpenClaw currently knows profitability and gates, but still carries split-scope signals "
                f"({current.get('account_scope_alignment_brief') or 'unknown_scope'})."
            ),
        },
        {
            "layer": "memory",
            "mission": "Keep an append-only truth of intents, executions, cooldowns, and exposure lineage.",
            "current_organs": [
                "latest_remote_live_history_audit",
                "panic_close_all marker",
                "live_risk_guard artifacts",
            ],
            "missing_organs": [
                "remote_intent_queue",
                "remote_execution_journal",
                "remote_position_truth",
                "remote_orderflow_feedback",
                "remote_orderflow_quality_report",
            ],
            "why_now": "Current remote life remembers outcomes, but not a first-class intent->fill->feedback chain.",
        },
        {
            "layer": "cortex",
            "mission": "Convert review heads and tickets into executable, scope-correct intents.",
            "current_organs": [
                "binance_live_takeover.py",
                "live_risk_guard.py",
                "signal_to_order_tickets",
            ],
            "missing_organs": [
                "openclaw_intent_router.py",
                "openclaw_scope_router.py",
                "openclaw_orderflow_policy.py",
            ],
            "why_now": (
                "The current cloud brain is good at saying 'blocked', but not yet organized as "
                "intent selection + scope routing + execution policy."
            ),
        },
        {
            "layer": "actuation",
            "mission": "Execute probes, canaries, and future ticketed orders with explicit idempotency and guardrails.",
            "current_organs": [
                "openclaw_cloud_bridge.sh live-takeover-*",
                "risk daemon systemd unit",
            ],
            "missing_organs": [
                "openclaw_orderflow_executor.py",
                "remote_execution_actor.service",
                "remote_execution_ack_state",
                "remote_execution_actor_canary_gate",
                "remote_live_boundary_hold",
                "remote_guarded_canary_promotion_gate",
                "remote_shadow_learning_continuity",
            ],
            "why_now": (
                "Bridge and executor concerns are still coupled inside one large shell bridge; "
                "future orderflow needs a dedicated actor that can swim while the guardian keeps veto power."
            ),
        },
    ]
    memory_layer = layers[1]
    if text(current.get("remote_intent_queue_brief")):
        current_organs = list(memory_layer.get("current_organs") or [])
        missing_organs = list(memory_layer.get("missing_organs") or [])
        if "remote_intent_queue" not in current_organs:
            current_organs.append("remote_intent_queue")
        memory_layer["current_organs"] = current_organs
        memory_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_intent_queue"
        ]
    if text(current.get("remote_execution_journal_brief")):
        current_organs = list(memory_layer.get("current_organs") or [])
        missing_organs = list(memory_layer.get("missing_organs") or [])
        if "remote_execution_journal" not in current_organs:
            current_organs.append("remote_execution_journal")
        memory_layer["current_organs"] = current_organs
        memory_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_execution_journal"
        ]
    if text(current.get("remote_orderflow_feedback_brief")):
        current_organs = list(memory_layer.get("current_organs") or [])
        missing_organs = list(memory_layer.get("missing_organs") or [])
        if "remote_orderflow_feedback" not in current_organs:
            current_organs.append("remote_orderflow_feedback")
        memory_layer["current_organs"] = current_organs
        memory_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_orderflow_feedback"
        ]
    actuation_layer = layers[3]
    cortex_layer = layers[2]
    if text(current.get("openclaw_orderflow_executor_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "openclaw_orderflow_executor.service" not in current_organs:
            current_organs.append("openclaw_orderflow_executor.service")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "openclaw_orderflow_executor.py"
        ]
    if text(current.get("remote_orderflow_policy_brief")):
        current_organs = list(cortex_layer.get("current_organs") or [])
        missing_organs = list(cortex_layer.get("missing_organs") or [])
        if "openclaw_orderflow_policy.py" not in current_organs:
            current_organs.append("openclaw_orderflow_policy.py")
        cortex_layer["current_organs"] = current_organs
        cortex_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "openclaw_orderflow_policy.py"
        ]
    if text(current.get("remote_execution_ack_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_execution_ack_state" not in current_organs:
            current_organs.append("remote_execution_ack_state")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_execution_ack_state"
        ]
    if text(current.get("remote_execution_actor_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_execution_actor.service" not in current_organs:
            current_organs.append("remote_execution_actor.service")
        missing_organs = [
            item for item in missing_organs if text(item) != "remote_execution_actor.service"
        ]
        for next_org in ("remote_execution_actor_guarded_transport", "remote_execution_transport_sla"):
            if next_org not in missing_organs:
                missing_organs.append(next_org)
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = missing_organs
    if text(current.get("remote_execution_guarded_transport_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_execution_actor_guarded_transport" not in current_organs:
            current_organs.append("remote_execution_actor_guarded_transport")
        missing_organs = [
            item for item in missing_organs if text(item) != "remote_execution_actor_guarded_transport"
        ]
        if "remote_execution_transport_sla" not in missing_organs:
            missing_organs.append("remote_execution_transport_sla")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = missing_organs
    if text(current.get("remote_execution_transport_sla_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_execution_transport_sla" not in current_organs:
            current_organs.append("remote_execution_transport_sla")
        missing_organs = [item for item in missing_organs if text(item) != "remote_execution_transport_sla"]
        if "remote_execution_actor_canary_gate" not in missing_organs:
            missing_organs.append("remote_execution_actor_canary_gate")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = missing_organs
    if text(current.get("remote_execution_actor_canary_gate_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_execution_actor_canary_gate" not in current_organs:
            current_organs.append("remote_execution_actor_canary_gate")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_execution_actor_canary_gate"
        ]
    if text(current.get("remote_orderflow_quality_report_brief")):
        current_organs = list(memory_layer.get("current_organs") or [])
        missing_organs = list(memory_layer.get("missing_organs") or [])
        if "remote_orderflow_quality_report" not in current_organs:
            current_organs.append("remote_orderflow_quality_report")
        memory_layer["current_organs"] = current_organs
        memory_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_orderflow_quality_report"
        ]
    if text(current.get("remote_live_boundary_hold_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_live_boundary_hold" not in current_organs:
            current_organs.append("remote_live_boundary_hold")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_live_boundary_hold"
        ]
    if text(current.get("remote_guarded_canary_promotion_gate_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_guarded_canary_promotion_gate" not in current_organs:
            current_organs.append("remote_guarded_canary_promotion_gate")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_guarded_canary_promotion_gate"
        ]
    if text(current.get("remote_shadow_learning_continuity_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_shadow_learning_continuity" not in current_organs:
            current_organs.append("remote_shadow_learning_continuity")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_shadow_learning_continuity"
        ]
    if text(current.get("remote_promotion_unblock_readiness_brief")):
        current_organs = list(actuation_layer.get("current_organs") or [])
        missing_organs = list(actuation_layer.get("missing_organs") or [])
        if "remote_promotion_unblock_readiness" not in current_organs:
            current_organs.append("remote_promotion_unblock_readiness")
        actuation_layer["current_organs"] = current_organs
        actuation_layer["missing_organs"] = [
            item for item in missing_organs if text(item) != "remote_promotion_unblock_readiness"
        ]
    return layers


def build_control_chain(current: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = as_dict(current.get("artifacts"))

    def pick_stage(
        *,
        stage: str,
        label: str,
        mission: str,
        interface_fields: list[str],
        optimization_brief: str,
        candidates: list[dict[str, str]],
    ) -> dict[str, Any]:
        for candidate in candidates:
            status = text(current.get(candidate.get("status_key") or ""))
            brief = text(current.get(candidate.get("brief_key") or ""))
            decision = text(current.get(candidate.get("decision_key") or ""))
            if not (status or brief or decision):
                continue
            artifact_key = text(candidate.get("artifact_key"))
            target_artifact = text(current.get(candidate.get("target_key") or "")) or artifact_key
            blocking_reason = (
                text(current.get(candidate.get("blocking_key") or ""))
                or decision
                or brief
            )
            return {
                "stage": stage,
                "label": label,
                "mission": mission,
                "source_status": status or "ok",
                "source_brief": brief or "-",
                "source_decision": decision or "-",
                "source_artifact_key": artifact_key,
                "source_artifact": text(artifacts.get(artifact_key)) if artifact_key else "",
                "next_target_artifact": target_artifact or "-",
                "blocking_reason": blocking_reason or "-",
                "interface_fields": interface_fields,
                "optimization_brief": optimization_brief,
            }
        fallback = candidates[0] if candidates else {}
        artifact_key = text(fallback.get("artifact_key"))
        return {
            "stage": stage,
            "label": label,
            "mission": mission,
            "source_status": "not_materialized",
            "source_brief": "-",
            "source_decision": "materialize_stage_source",
            "source_artifact_key": artifact_key,
            "source_artifact": text(artifacts.get(artifact_key)) if artifact_key else "",
            "next_target_artifact": artifact_key or "-",
            "blocking_reason": "source_not_materialized",
            "interface_fields": interface_fields,
            "optimization_brief": optimization_brief,
        }

    return [
        pick_stage(
            stage="research",
            label="研究",
            mission="验证 pattern family 的 OOS、横截面和成本后期望，决定什么值得继续推进。",
            interface_fields=[
                "crypto_shortline_cross_section_backtest_status",
                "crypto_shortline_cross_section_backtest_decision",
                "crypto_shortline_backtest_slice_status",
                "crypto_shortline_backtest_slice_decision",
            ],
            optimization_brief="持续刷新 family 级 cross-section、slice 和成本后期望，让 review head 的边际判断可审计。",
            candidates=[
                {
                    "status_key": "crypto_shortline_cross_section_backtest_status",
                    "brief_key": "crypto_shortline_cross_section_backtest_brief",
                    "decision_key": "crypto_shortline_cross_section_backtest_decision",
                    "artifact_key": "crypto_shortline_cross_section_backtest",
                },
                {
                    "status_key": "crypto_shortline_backtest_slice_status",
                    "brief_key": "crypto_shortline_backtest_slice_brief",
                    "decision_key": "crypto_shortline_backtest_slice_decision",
                    "artifact_key": "crypto_shortline_backtest_slice",
                },
            ],
        ),
        pick_stage(
            stage="signal",
            label="信号",
            mission="把研究结论下沉成当前时刻的可执行窗口，明确当前卡在哪一层 gate。",
            interface_fields=[
                "remote_ticket_actionability_status",
                "remote_ticket_actionability_decision",
                "crypto_signal_source_refresh_readiness_status",
                "crypto_signal_source_freshness_status",
            ],
            optimization_brief="持续把 signal source、ticket actionability 和 pattern family handoff 保持同主语，避免 generic backflow。",
            candidates=[
                {
                    "status_key": "remote_ticket_actionability_status",
                    "brief_key": "remote_ticket_actionability_brief",
                    "decision_key": "remote_ticket_actionability_decision",
                    "artifact_key": "remote_ticket_actionability_state",
                    "target_key": "remote_ticket_actionability_next_action_target_artifact",
                },
                {
                    "status_key": "crypto_signal_source_refresh_readiness_status",
                    "brief_key": "crypto_signal_source_refresh_readiness_brief",
                    "decision_key": "crypto_signal_source_refresh_readiness_decision",
                    "artifact_key": "crypto_signal_source_refresh_readiness",
                },
                {
                    "status_key": "crypto_signal_source_freshness_status",
                    "brief_key": "crypto_signal_source_freshness_brief",
                    "decision_key": "crypto_signal_source_freshness_decision",
                    "artifact_key": "crypto_signal_source_freshness",
                },
            ],
        ),
        pick_stage(
            stage="risk",
            label="风控",
            mission="在执行前给出 veto、边界保持和 promotion clearance，决定现在是否允许继续推进。",
            interface_fields=[
                "remote_guardian_blocker_clearance_status",
                "remote_guardian_blocker_clearance_top_blocker_code",
                "remote_live_boundary_hold_status",
                "remote_promotion_unblock_readiness_status",
            ],
            optimization_brief="持续把 guardian clearance、live boundary hold 和 unblock readiness 收敛成同一套 veto 语义，避免风险判断分叉。",
            candidates=[
                {
                    "status_key": "remote_guardian_blocker_clearance_status",
                    "brief_key": "remote_guardian_blocker_clearance_brief",
                    "decision_key": "remote_guardian_blocker_clearance_top_blocker_next_action",
                    "artifact_key": "remote_guardian_blocker_clearance",
                    "target_key": "remote_guardian_blocker_clearance_top_blocker_target_artifact",
                    "blocking_key": "remote_guardian_blocker_clearance_top_blocker_detail",
                },
                {
                    "status_key": "remote_live_boundary_hold_status",
                    "brief_key": "remote_live_boundary_hold_brief",
                    "decision_key": "remote_live_boundary_hold_decision",
                    "artifact_key": "remote_live_boundary_hold",
                    "target_key": "remote_guarded_canary_promotion_gate_blocker_target_artifact",
                },
                {
                    "status_key": "remote_promotion_unblock_readiness_status",
                    "brief_key": "remote_promotion_unblock_readiness_brief",
                    "decision_key": "remote_promotion_unblock_readiness_decision",
                    "artifact_key": "remote_promotion_unblock_readiness",
                    "target_key": "remote_promotion_unblock_primary_local_repair_target_artifact",
                    "blocking_key": "remote_promotion_unblock_primary_local_repair_detail",
                },
            ],
        ),
        pick_stage(
            stage="execution",
            label="执行",
            mission="把 guardian 同意的 intent 送进 actor/canary/promotion path，并保持 transport policy 可解释。",
            interface_fields=[
                "remote_guarded_canary_promotion_gate_status",
                "remote_execution_actor_canary_gate_status",
                "remote_orderflow_policy_status",
                "openclaw_orderflow_executor_status",
            ],
            optimization_brief="持续优化 executor、policy、canary gate 和 promotion gate 的传导链，避免 transport 与 guardian 耦合。",
            candidates=[
                {
                    "status_key": "remote_guarded_canary_promotion_gate_status",
                    "brief_key": "remote_guarded_canary_promotion_gate_brief",
                    "decision_key": "remote_guarded_canary_promotion_gate_decision",
                    "artifact_key": "remote_guarded_canary_promotion_gate",
                    "target_key": "remote_guarded_canary_promotion_gate_blocker_target_artifact",
                    "blocking_key": "remote_guarded_canary_promotion_gate_blocker_detail",
                },
                {
                    "status_key": "remote_execution_actor_canary_gate_status",
                    "brief_key": "remote_execution_actor_canary_gate_brief",
                    "decision_key": "remote_execution_actor_canary_gate_decision",
                    "artifact_key": "remote_execution_actor_canary_gate",
                },
                {
                    "status_key": "remote_orderflow_policy_status",
                    "brief_key": "remote_orderflow_policy_brief",
                    "decision_key": "remote_orderflow_policy_decision",
                    "artifact_key": "remote_orderflow_policy_state",
                },
                {
                    "status_key": "openclaw_orderflow_executor_status",
                    "brief_key": "openclaw_orderflow_executor_brief",
                    "artifact_key": "openclaw_orderflow_executor_state",
                },
            ],
        ),
        pick_stage(
            stage="reconcile",
            label="对账",
            mission="把 send/ack/no-fill/journal/transport SLA 统一成可追踪的执行真相，避免 source-of-truth 歧义。",
            interface_fields=[
                "remote_execution_ack_status",
                "remote_execution_ack_decision",
                "remote_execution_journal_status",
                "remote_execution_transport_sla_status",
            ],
            optimization_brief="持续强化 ack、journal 和 transport SLA 的闭环，让 execution truth 能独立支撑 halt/reconcile 决策。",
            candidates=[
                {
                    "status_key": "remote_execution_ack_status",
                    "brief_key": "remote_execution_ack_brief",
                    "decision_key": "remote_execution_ack_decision",
                    "artifact_key": "remote_execution_ack_state",
                },
                {
                    "status_key": "remote_execution_journal_status",
                    "brief_key": "remote_execution_journal_brief",
                    "decision_key": "remote_execution_journal_append_status",
                    "artifact_key": "remote_execution_journal",
                },
                {
                    "status_key": "remote_execution_transport_sla_status",
                    "brief_key": "remote_execution_transport_sla_brief",
                    "decision_key": "remote_execution_transport_sla_decision",
                    "artifact_key": "remote_execution_transport_sla",
                },
            ],
        ),
        pick_stage(
            stage="post_trade",
            label="复盘",
            mission="把 fill/no-fill/queue aging 和 shadow learning 反馈回研究与路由，决定哪些 lane 应该降权或提升。",
            interface_fields=[
                "remote_orderflow_feedback_status",
                "remote_orderflow_quality_report_status",
                "remote_shadow_learning_continuity_status",
                "remote_orderflow_quality_report_score",
            ],
            optimization_brief="持续把 feedback、quality report 和 shadow learning continuity 回灌到 research/routing，形成真正的学习闭环。",
            candidates=[
                {
                    "status_key": "remote_orderflow_feedback_status",
                    "brief_key": "remote_orderflow_feedback_brief",
                    "decision_key": "remote_orderflow_feedback_recommendation",
                    "artifact_key": "remote_orderflow_feedback",
                },
                {
                    "status_key": "remote_orderflow_quality_report_status",
                    "brief_key": "remote_orderflow_quality_report_brief",
                    "decision_key": "remote_orderflow_quality_report_recommendation",
                    "artifact_key": "remote_orderflow_quality_report",
                },
                {
                    "status_key": "remote_shadow_learning_continuity_status",
                    "brief_key": "remote_shadow_learning_continuity_brief",
                    "decision_key": "remote_shadow_learning_continuity_decision",
                    "artifact_key": "remote_shadow_learning_continuity",
                    "blocking_key": "remote_shadow_learning_continuity_blocker_detail",
                },
            ],
        ),
    ]


def build_continuous_optimization_backlog(
    current: dict[str, Any],
    control_chain: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    priorities = {
        "research": 60,
        "signal": 70,
        "risk": 80,
        "execution": 85,
        "reconcile": 75,
        "post_trade": 65,
    }
    change_classes = {
        "research": "RESEARCH_ONLY",
        "signal": "RESEARCH_ONLY",
        "risk": "LIVE_GUARD_ONLY",
        "execution": "LIVE_GUARD_ONLY",
        "reconcile": "LIVE_GUARD_ONLY",
        "post_trade": "RESEARCH_ONLY",
    }
    rows: list[dict[str, Any]] = []
    for stage in control_chain:
        if not isinstance(stage, dict):
            continue
        stage_key = text(stage.get("stage"))
        label = text(stage.get("label")) or stage_key or "unknown"
        target_artifact = text(stage.get("next_target_artifact")) or text(stage.get("source_artifact_key"))
        rows.append(
            {
                "priority": priorities.get(stage_key, 50),
                "stage": stage_key,
                "label": label,
                "title": f"持续优化{label}传导链",
                "why": text(stage.get("optimization_brief")) or text(stage.get("blocking_reason")),
                "target_artifact": target_artifact or "-",
                "source_status": text(stage.get("source_status")) or "-",
                "source_artifact": text(stage.get("source_artifact")) or "",
                "interface_fields": as_list(stage.get("interface_fields")),
                "change_class": change_classes.get(stage_key, "RESEARCH_ONLY"),
            }
        )
    rows.sort(key=lambda item: (-int(item.get("priority") or 0), text(item.get("stage"))))
    return rows


def build_refactor_phases(current: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "phase": 1,
            "name": "identity_and_scope",
            "goal": "Split cloud identity, account scope, and remote profitability into one source-of-truth state.",
            "deliverables": [
                "build_remote_execution_identity_state.py",
                "build_remote_scope_router_state.py",
                "remote_execution_identity_state.json",
            ],
            "code_anchors": [
                "scripts/build_remote_live_handoff.py",
                "scripts/build_live_gate_blocker_report.py",
                "scripts/binance_live_takeover.py",
            ],
            "done_when": (
                "OpenClaw can explain venue/account scope without relying on mixed handoff text, and scope mismatches "
                "become first-class artifact fields."
            ),
        },
        {
            "phase": 2,
            "name": "intent_and_memory",
            "goal": "Create an append-only remote intent queue and execution journal before broader automation.",
            "deliverables": [
                "build_remote_intent_queue.py",
                "build_remote_execution_journal.py",
                "remote_intent_queue.json",
                "remote_execution_journal.jsonl",
            ],
            "code_anchors": [
                "scripts/live_risk_guard.py",
                "scripts/build_order_ticket.py",
                "scripts/openclaw_cloud_bridge.sh",
            ],
            "done_when": "Every remote action can be traced from review head -> intent -> risk verdict -> fill/no-fill.",
        },
        {
            "phase": 3,
            "name": "guardian_executor_split",
            "goal": "Keep the guardian as veto-only and move execution into a dedicated orderflow actor.",
            "deliverables": [
                "scripts/openclaw_orderflow_executor.py",
                "scripts/render_openclaw_orderflow_executor_unit.py",
                "scripts/build_remote_execution_ack_state.py",
                "openclaw-orderflow-executor.service",
                "remote_execution_ack_state.json",
            ],
            "code_anchors": [
                "scripts/openclaw_cloud_bridge.sh",
                "scripts/render_live_risk_daemon_systemd_unit.py",
            ],
            "done_when": "Risk daemon remains independent while execution actor owns only transport/execution state.",
        },
        {
            "phase": 4,
            "name": "orderflow_feedback_learning",
            "goal": "Feed fills, slippage, queue aging, and canary outcomes back into selection and throttling.",
            "deliverables": [
                "build_remote_orderflow_feedback.py",
                "remote_orderflow_feedback.json",
                "remote_orderflow_quality_report.json",
            ],
            "code_anchors": [
                "scripts/binance_live_takeover.py",
                "scripts/live_risk_guard.py",
                "scripts/build_live_gate_blocker_report.py",
            ],
            "done_when": "OpenClaw can down-rank stale or low-quality execution lanes using remote fill evidence.",
        },
        {
            "phase": 5,
            "name": "guarded_actor_transport",
            "goal": "Promote the shadow actor into guarded transport and then measure send/ack/fill latency as a first-class SLA.",
            "deliverables": [
                "build_remote_execution_actor_state.py",
                "remote_execution_actor_state.json",
                "remote_execution_actor_guarded_transport",
                "remote_execution_transport_sla",
            ],
            "code_anchors": [
                "scripts/openclaw_orderflow_executor.py",
                "scripts/build_remote_execution_ack_state.py",
                "scripts/openclaw_cloud_bridge.sh",
            ],
            "done_when": "Guardian-approved intents can enter a dedicated actor transport path with explicit transport ack/fill timing and no hidden bridge coupling.",
        },
        {
            "phase": 6,
            "name": "live_boundary_governance",
            "goal": "Make the shadow/live boundary explicit so OpenClaw can explain why transport stays shadow-only and what must clear before any guarded canary review.",
            "deliverables": [
                "build_remote_live_boundary_hold.py",
                "remote_live_boundary_hold.json",
                "remote_execution_actor_canary_gate",
                "remote_orderflow_quality_report",
            ],
            "code_anchors": [
                "scripts/build_remote_execution_actor_canary_gate.py",
                "scripts/build_remote_orderflow_quality_report.py",
                "scripts/build_live_gate_blocker_report.py",
                "scripts/refresh_cross_market_operator_state.py",
            ],
            "done_when": "The final shadow hold is first-class, and the remaining backlog is guardian/review clearance rather than missing architecture.",
        },
    ]


def build_immediate_backlog(current: dict[str, Any]) -> list[dict[str, Any]]:
    scope_alignment = text(current.get("account_scope_alignment_brief"))
    later_phase_materialized = any(
        text(
            current.get(key)
        )
        for key in (
            "remote_guardian_blocker_clearance_brief",
            "remote_live_boundary_hold_brief",
            "remote_guarded_canary_promotion_gate_brief",
            "remote_shadow_learning_continuity_brief",
            "remote_promotion_unblock_readiness_brief",
            "remote_ticket_actionability_brief",
            "crypto_shortline_backtest_slice_brief",
            "crypto_shortline_cross_section_backtest_brief",
        )
    )
    if not text(current.get("remote_execution_identity_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Split remote identity and scope state",
                "why": f"Current scope signal is still mixed: {scope_alignment or 'unknown'}.",
                "target_artifact": "remote_execution_identity_state",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Create remote scope router state",
                "why": "Current cross-market head and remote executable scope are still reconciled indirectly inside handoff text.",
                "target_artifact": "remote_scope_router_state",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Create remote intent queue",
                "why": "Risk guard is blocking stale/non-actionable tickets instead of consuming a first-class remote intent bus.",
                "target_artifact": "remote_intent_queue",
                "change_class": "RESEARCH_ONLY",
            },
            {
                "priority": 4,
                "title": "Split guardian and executor services",
                "why": "Current shell bridge is a transport monolith; future orderflow needs a dedicated execution actor.",
                "target_artifact": "openclaw_orderflow_executor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_scope_router_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Create remote scope router state",
                "why": "Current cross-market head and remote executable scope are still reconciled indirectly inside handoff text.",
                "target_artifact": "remote_scope_router_state",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Create remote intent queue",
                "why": "Risk guard is blocking stale/non-actionable tickets instead of consuming a first-class remote intent bus.",
                "target_artifact": "remote_intent_queue",
                "change_class": "RESEARCH_ONLY",
            },
            {
                "priority": 3,
                "title": "Split guardian and executor services",
                "why": "Current shell bridge is a transport monolith; future orderflow needs a dedicated execution actor.",
                "target_artifact": "openclaw_orderflow_executor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 4,
                "title": "Append execution journal for canary/probe lineage",
                "why": "Remote history is profitable, but lineage from review head to fill remains too implicit.",
                "target_artifact": "remote_execution_journal",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_intent_queue_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Create remote intent queue",
                "why": "Risk guard is blocking stale/non-actionable tickets instead of consuming a first-class remote intent bus.",
                "target_artifact": "remote_intent_queue",
                "change_class": "RESEARCH_ONLY",
            },
            {
                "priority": 2,
                "title": "Append execution journal for canary/probe lineage",
                "why": "Remote history is profitable, but lineage from review head to fill remains too implicit.",
                "target_artifact": "remote_execution_journal",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Split guardian and executor services",
                "why": "Current shell bridge is a transport monolith; future orderflow needs a dedicated execution actor.",
                "target_artifact": "openclaw_orderflow_executor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 4,
                "title": "Feed fill quality back into routing",
                "why": "Once the intent bus exists, the next durable edge comes from intent->fill feedback rather than more handoff text.",
                "target_artifact": "remote_orderflow_feedback",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_execution_journal_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Append execution journal for canary/probe lineage",
                "why": "Remote history is profitable, but lineage from queued intent to fill/no-fill remains too implicit.",
                "target_artifact": "remote_execution_journal",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Split guardian and executor services",
                "why": "Current shell bridge is a transport monolith; future orderflow needs a dedicated execution actor.",
                "target_artifact": "openclaw_orderflow_executor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Feed fill quality back into routing",
                "why": "Once intents are queued, the next durable edge comes from intent->fill feedback rather than more handoff text.",
                "target_artifact": "remote_orderflow_feedback",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 4,
                "title": "Split guardian from scope-aware router policy",
                "why": "Once intents are queued, routing and policy logic should stop living implicitly inside the bridge.",
                "target_artifact": "openclaw_orderflow_policy.py",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("openclaw_orderflow_executor_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Split guardian and executor services",
                "why": "Intent and journal lineage exist, but execution still lacks a dedicated actor service boundary.",
                "target_artifact": "openclaw_orderflow_executor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Feed fill quality back into routing",
                "why": "Once executor scaffolding exists, the next durable edge comes from fill/no-fill feedback.",
                "target_artifact": "remote_orderflow_feedback",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Split guardian from scope-aware router policy",
                "why": "Executor scaffolding still needs a clean policy boundary for intent acceptance.",
                "target_artifact": "openclaw_orderflow_policy.py",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 4,
                "title": "Teach executor to write ack state",
                "why": "A dedicated actor still needs a first-class ack/fill state artifact before canary transport work.",
                "target_artifact": "remote_execution_ack_state",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_orderflow_feedback_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Feed fill quality back into routing",
                "why": "With executor scaffolding in place, the next durable edge comes from intent->fill feedback rather than more bridge text.",
                "target_artifact": "remote_orderflow_feedback",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Split guardian from scope-aware router policy",
                "why": "Executor scaffolding exists, but routing and policy logic should still stop living implicitly inside the bridge.",
                "target_artifact": "openclaw_orderflow_policy.py",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Teach executor to write ack state",
                "why": "A dedicated actor still needs explicit ack/fill state before transport canary work becomes auditable.",
                "target_artifact": "remote_execution_ack_state",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 4,
                "title": "Canary the dedicated executor transport",
                "why": "After executor scaffolding and ack state exist, the next gap is a guarded canary transport path.",
                "target_artifact": "remote_execution_actor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_orderflow_policy_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Split guardian from scope-aware router policy",
                "why": "Executor scaffolding and first feedback now exist, so routing and policy logic should stop living implicitly inside the bridge.",
                "target_artifact": "openclaw_orderflow_policy.py",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Teach executor to write ack state",
                "why": "After feedback is first-class, the next missing execution truth is explicit ack/fill state.",
                "target_artifact": "remote_execution_ack_state",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Canary the dedicated executor transport",
                "why": "Once feedback and ack state exist, the executor can graduate from shadow heartbeat toward guarded transport.",
                "target_artifact": "remote_execution_actor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 4,
                "title": "Materialize remote orderflow quality report",
                "why": "Once feedback exists, the next refinement is a durable quality report that aggregates fills, no-fills, and queue aging over time.",
                "target_artifact": "remote_orderflow_quality_report",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_execution_ack_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Teach executor to write ack state",
                "why": "Policy now separates guardian rejection from route selection, so the next missing execution truth is explicit ack/fill state.",
                "target_artifact": "remote_execution_ack_state",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Canary the dedicated executor transport",
                "why": "Once policy and future ack state exist, the executor can graduate from shadow heartbeat toward guarded transport.",
                "target_artifact": "remote_execution_actor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Materialize remote orderflow quality report",
                "why": "With feedback and policy in place, the next refinement is a durable quality report that aggregates fills, no-fills, and queue aging over time.",
                "target_artifact": "remote_orderflow_quality_report",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 4,
                "title": "Canary guardian-approved orderflow actor",
                "why": "After policy and ack state are source-owned, the remaining gap is a guarded transport actor that honors those decisions.",
                "target_artifact": "remote_execution_actor_guarded_transport",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_execution_actor_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Materialize dedicated remote execution actor state",
                "why": "Executor scaffolding and ack truth now exist, but the actor boundary is still implicit instead of source-owned.",
                "target_artifact": "remote_execution_actor.service",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Canary guardian-approved orderflow actor",
                "why": "After the actor boundary becomes first-class, the next gap is a guarded transport path that still honors policy veto.",
                "target_artifact": "remote_execution_actor_guarded_transport",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Graduate shadow ack into transport SLA",
                "why": "Once the actor boundary is source-owned, transport latency and duplicate handling need explicit SLOs.",
                "target_artifact": "remote_execution_transport_sla",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 4,
                "title": "Materialize remote orderflow quality report",
                "why": "Feedback, policy, ack, and actor state should converge into a durable quality report rather than stay as isolated artifacts.",
                "target_artifact": "remote_orderflow_quality_report",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_execution_guarded_transport_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Canary guardian-approved orderflow actor",
                "why": "Actor state now exists, so the next gap is a guarded transport preview that proves policy veto stays attached to the send boundary.",
                "target_artifact": "remote_execution_actor_guarded_transport",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Graduate shadow ack into transport SLA",
                "why": "After the guarded transport boundary is source-owned, transport timing and duplicate handling need explicit SLOs.",
                "target_artifact": "remote_execution_transport_sla",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Materialize remote orderflow quality report",
                "why": "Feedback, policy, ack, and actor state should converge into a durable quality report rather than stay as isolated artifacts.",
                "target_artifact": "remote_orderflow_quality_report",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_execution_transport_sla_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Graduate shadow ack into transport SLA",
                "why": "Guarded transport preview now exists, so the next learning gap is explicit send/ack/fill timing and duplicate handling before any canary gate is armed.",
                "target_artifact": "remote_execution_transport_sla",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Materialize remote orderflow quality report",
                "why": "Feedback, policy, ack, actor state, and guarded transport preview should converge into a durable quality report.",
                "target_artifact": "remote_orderflow_quality_report",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Prepare guarded actor canary gate",
                "why": "Once SLA semantics are explicit, the remaining safe step is a source-owned gate that proves when a canary would be allowed.",
                "target_artifact": "remote_execution_actor_canary_gate",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_execution_actor_canary_gate_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Prepare guarded actor canary gate",
                "why": "With transport SLA semantics source-owned, the final safe shadow step is an explicit gate that proves when a canary would be allowed.",
                "target_artifact": "remote_execution_actor_canary_gate",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Materialize remote orderflow quality report",
                "why": "With feedback, policy, ack, actor state, guarded transport preview, and SLA semantics in place, the next refinement is a durable quality report that aggregates fills, no-fills, and queue aging over time.",
                "target_artifact": "remote_orderflow_quality_report",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 3,
                "title": "Keep live transport shadow-only until canary gate is explicit",
                "why": "The remaining unsolved problem is no longer architecture; it is the live boundary between shadow artifacts and a real guarded canary.",
                "target_artifact": "live_boundary_hold",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_orderflow_quality_report_brief")) and not later_phase_materialized:
        return [
            {
                "priority": 1,
                "title": "Materialize remote orderflow quality report",
                "why": "With canary gate semantics source-owned, the next refinement is a durable quality report that aggregates fills, no-fills, queue aging, and canary readiness over time.",
                "target_artifact": "remote_orderflow_quality_report",
                "change_class": "LIVE_GUARD_ONLY",
            },
            {
                "priority": 2,
                "title": "Keep live transport shadow-only until canary gate clears",
                "why": "The remaining unsolved problem is no longer architecture; it is the live boundary between shadow artifacts and a real guarded canary.",
                "target_artifact": "live_boundary_hold",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_live_boundary_hold_brief")):
        return [
            {
                "priority": 1,
                "title": "Materialize remote live boundary hold",
                "why": "Canary and quality semantics now exist, so the final safe source artifact is an explicit shadow/live boundary hold that explains why transport stays shadow-only.",
                "target_artifact": "live_boundary_hold",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_guarded_canary_promotion_gate_brief")):
        return [
            {
                "priority": 1,
                "title": "Materialize guarded canary promotion gate",
                "why": "Shadow clock evidence, live boundary hold, and guardian clearance already exist, so promotion should be governed by one first-class gate instead of being inferred across multiple artifacts.",
                "target_artifact": "remote_guarded_canary_promotion_gate",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_shadow_learning_continuity_brief")):
        return [
            {
                "priority": 1,
                "title": "Materialize remote shadow learning continuity",
                "why": "Once promotion is blocked explicitly, the remaining survival question is whether shadow learning is still healthy enough to keep collecting useful feedback.",
                "target_artifact": "remote_shadow_learning_continuity",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]
    if not text(current.get("remote_promotion_unblock_readiness_brief")):
        return [
            {
                "priority": 1,
                "title": "Materialize remote promotion unblock readiness",
                "why": "Shadow learning continuity is already stable; the next missing source-of-truth is whether promotion is only blocked by local time-sync repair or by mixed blockers.",
                "target_artifact": "remote_promotion_unblock_readiness",
                "change_class": "LIVE_GUARD_ONLY",
            },
        ]

    backlog: list[dict[str, Any]] = []
    if text(current.get("remote_promotion_unblock_readiness_status")) in {
        "local_time_sync_primary_blocker_shadow_ready",
        "shadow_ready_ticket_actionability_blocked",
    }:
        backlog_why = text(current.get("remote_promotion_unblock_primary_local_repair_detail"))
        if (
            not backlog_why
            and text(current.get("remote_promotion_unblock_readiness_status"))
            == "local_time_sync_primary_blocker_shadow_ready"
        ):
            backlog_why_parts = [
                text(current.get("remote_promotion_unblock_primary_local_repair_plan_brief")),
            ]
            environment_classification = text(
                current.get("remote_promotion_unblock_primary_local_repair_environment_classification")
            )
            environment_detail = text(
                current.get("remote_promotion_unblock_primary_local_repair_environment_blocker_detail")
            )
            if environment_classification and environment_detail:
                backlog_why_parts.append(
                    f"env={environment_classification}:{environment_detail}"
                )
            remediation_hint = text(
                current.get(
                    "remote_promotion_unblock_primary_local_repair_environment_remediation_hint"
                )
            )
            if remediation_hint:
                backlog_why_parts.append(f"fix_hint={remediation_hint}")
            backlog_why = " | ".join(part for part in backlog_why_parts if text(part))
        backlog.append(
            {
                "priority": 1,
                "title": text(current.get("remote_promotion_unblock_primary_local_repair_title"))
                or (
                    "Repair local time sync to unlock guarded canary review"
                    if text(current.get("remote_promotion_unblock_readiness_status"))
                    == "local_time_sync_primary_blocker_shadow_ready"
                    else "Resolve ticket actionability before guarded canary review"
                ),
                "why": backlog_why or text(current.get("remote_promotion_unblock_readiness_brief")),
                "target_artifact": text(
                    current.get("remote_promotion_unblock_primary_local_repair_target_artifact")
                )
                or (
                    "system_time_sync_repair_verification_report"
                    if text(current.get("remote_promotion_unblock_readiness_status"))
                    == "local_time_sync_primary_blocker_shadow_ready"
                    else "remote_ticket_actionability_state"
                ),
                "change_class": "LIVE_GUARD_ONLY",
            }
        )
        return backlog
    if text(current.get("remote_shadow_learning_continuity_status")) and text(
        current.get("remote_shadow_learning_continuity_status")
    ) != "shadow_learning_continuity_stable":
        backlog.append(
            {
                "priority": 1,
                "title": "Repair shadow learning continuity while promotion is blocked",
                "why": text(current.get("remote_shadow_learning_continuity_blocker_detail"))
                or text(current.get("remote_shadow_learning_continuity_brief")),
                "target_artifact": "remote_shadow_learning_continuity",
                "change_class": "LIVE_GUARD_ONLY",
            }
        )
        return backlog
    if text(current.get("remote_guarded_canary_promotion_gate_brief")):
        backlog.append(
            {
                "priority": 1,
                "title": text(current.get("remote_guarded_canary_promotion_gate_blocker_title"))
                or "Clear guarded canary promotion blockers",
                "why": text(current.get("remote_guarded_canary_promotion_gate_blocker_detail"))
                or text(current.get("remote_guarded_canary_promotion_gate_brief")),
                "target_artifact": text(
                    current.get("remote_guarded_canary_promotion_gate_blocker_target_artifact")
                )
                or "remote_guarded_canary_promotion_gate",
                "change_class": "LIVE_GUARD_ONLY",
            }
        )
        return backlog
    if text(current.get("remote_guardian_blocker_clearance_brief")):
        backlog.append(
            {
                "priority": 1,
                "title": text(current.get("remote_guardian_blocker_clearance_top_blocker_title"))
                or "Clear guardian blockers before any guarded canary review",
                "why": text(current.get("remote_guardian_blocker_clearance_top_blocker_detail"))
                or text(current.get("remote_live_boundary_hold_brief"))
                or "Guardian blockers still prevent the shadow/live boundary from advancing.",
                "target_artifact": text(
                    current.get("remote_guardian_blocker_clearance_top_blocker_target_artifact")
                )
                or "guardian_blocker_clearance",
                "change_class": "LIVE_GUARD_ONLY",
            }
        )
    elif bool(current.get("remote_live_boundary_hold_guardian_blocked")):
        backlog.append(
            {
                "priority": 1,
                "title": "Clear guardian blockers before any guarded canary review",
                "why": text(current.get("remote_live_boundary_hold_brief"))
                or "Guardian blockers still prevent the shadow/live boundary from advancing.",
                "target_artifact": "guardian_blocker_clearance",
                "change_class": "LIVE_GUARD_ONLY",
            }
        )
    if bool(current.get("remote_live_boundary_hold_review_blocked")) or bool(
        current.get("remote_live_boundary_hold_time_sync_blocked")
    ):
        backlog.append(
            {
                "priority": 2,
                "title": "Clear review-head blockers before any guarded canary review",
                "why": text(current.get("cross_market_review_head_brief"))
                or "The current review head is still bias-only or time-sync blocked.",
                "target_artifact": "review_head_clearance",
                "change_class": "RESEARCH_ONLY",
            }
        )
    if not backlog:
        backlog.append(
            {
                "priority": 1,
                "title": "Review guarded canary boundary",
                "why": "The shadow/live boundary is now explicit; the next step is a human review before any guarded canary is considered.",
                "target_artifact": text(current.get("remote_live_boundary_hold_next_transition"))
                or "guarded_canary_review",
                "change_class": "LIVE_GUARD_ONLY",
            }
        )
    return backlog


def build_code_anchors() -> list[dict[str, str]]:
    return [
        {
            "role": "cloud bridge monolith",
            "path": str(SYSTEM_ROOT / "scripts" / "openclaw_cloud_bridge.sh"),
            "why": "Current transport, ready-check, risk-daemon orchestration, and canary actions live here.",
        },
        {
            "role": "venue adapter and canary logic",
            "path": str(SYSTEM_ROOT / "scripts" / "binance_live_takeover.py"),
            "why": "Owns token bucket, timeout, mutex, account/trade telemetry, and canary execution logic.",
        },
        {
            "role": "independent guardian",
            "path": str(SYSTEM_ROOT / "scripts" / "live_risk_guard.py"),
            "why": "Owns ticket freshness, panic cooldown, exposure, and independent live veto logic.",
        },
        {
            "role": "guardian systemd renderer",
            "path": str(SYSTEM_ROOT / "scripts" / "render_live_risk_daemon_systemd_unit.py"),
            "why": "Shows current remote service model: one daemonized guardian, no dedicated orderflow executor yet.",
        },
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    current = as_dict(payload.get("current_status"))
    phases = as_list(payload.get("refactor_phases"))
    backlog = as_list(payload.get("immediate_backlog"))
    layers = as_list(payload.get("digital_layers"))
    control_chain = as_list(payload.get("control_chain"))
    continuous_optimization_backlog = as_list(payload.get("continuous_optimization_backlog"))
    lines = [
        "# OpenClaw Orderflow Blueprint",
        "",
        "## Current status",
        f"- current life stage: `{text(current.get('current_life_stage'))}`",
        f"- target life stage: `{text(current.get('target_life_stage'))}`",
        f"- remote host: `{text(current.get('remote_host'))}`",
        f"- remote project: `{text(current.get('remote_project_dir'))}`",
        f"- handoff state: `{text(current.get('handoff_state'))}`",
        f"- status: `{text(current.get('operator_status_quad'))}`",
        f"- focus stack: `{text(current.get('focus_stack_brief'))}`",
        f"- remote live diagnosis: `{text(current.get('remote_live_diagnosis_brief'))}`",
        f"- live decision: `{text(current.get('live_decision'))}`",
        f"- review head: `{text(current.get('cross_market_review_head_brief'))}`",
        f"- remote alignment: `{text(current.get('cross_market_operator_alignment_brief'))}`",
        f"- identity state: `{text(current.get('remote_execution_identity_brief')) or '-'}`",
        f"- scope router: `{text(current.get('remote_scope_router_brief')) or '-'}`",
        f"- intent queue: `{text(current.get('remote_intent_queue_brief')) or '-'}`",
        f"- execution journal: `{text(current.get('remote_execution_journal_brief')) or '-'}`",
        f"- executor scaffold: `{text(current.get('openclaw_orderflow_executor_brief')) or '-'}`",
        f"- execution actor: `{text(current.get('remote_execution_actor_brief')) or '-'}`",
        f"- orderflow feedback: `{text(current.get('remote_orderflow_feedback_brief')) or '-'}`",
        f"- orderflow policy: `{text(current.get('remote_orderflow_policy_brief')) or '-'}`",
        f"- execution ack: `{text(current.get('remote_execution_ack_brief')) or '-'}`",
        f"- guarded transport: `{text(current.get('remote_execution_guarded_transport_brief')) or '-'}`",
        f"- transport SLA: `{text(current.get('remote_execution_transport_sla_brief')) or '-'}`",
        f"- canary gate: `{text(current.get('remote_execution_actor_canary_gate_brief')) or '-'}`",
        f"- quality report: `{text(current.get('remote_orderflow_quality_report_brief')) or '-'}`",
        f"- guardian clearance: `{text(current.get('remote_guardian_blocker_clearance_brief')) or '-'}`",
        f"- live boundary hold: `{text(current.get('remote_live_boundary_hold_brief')) or '-'}`",
        f"- promotion gate: `{text(current.get('remote_guarded_canary_promotion_gate_brief')) or '-'}`",
        f"- shadow learning continuity: `{text(current.get('remote_shadow_learning_continuity_brief')) or '-'}`",
        f"- promotion unblock readiness: `{text(current.get('remote_promotion_unblock_readiness_brief')) or '-'}`",
        f"- remote ticket actionability: `{text(current.get('remote_ticket_actionability_brief')) or '-'}`",
        f"- crypto signal source refresh readiness: `{text(current.get('crypto_signal_source_refresh_readiness_brief')) or '-'}`",
        f"- crypto signal source freshness: `{text(current.get('crypto_signal_source_freshness_brief')) or '-'}`",
        f"- crypto shortline backtest slice: `{text(current.get('crypto_shortline_backtest_slice_brief')) or '-'}`",
        f"- crypto shortline cross-section backtest: `{text(current.get('crypto_shortline_cross_section_backtest_brief')) or '-'}`",
        f"- remote time sync mode: `{text(current.get('remote_time_sync_mode')) or '-'}`",
        f"- shadow clock evidence: `{text(current.get('remote_shadow_clock_evidence_brief')) or '-'}`",
        "",
        "## Why OpenClaw is not orderflow-native yet",
    ]
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        lines.append(f"- `{text(layer.get('layer'))}`: {text(layer.get('why_now'))}")
    lines.extend(["", "## Refactor phases"])
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        lines.append(f"- `phase {phase.get('phase')}` `{text(phase.get('name'))}`: {text(phase.get('goal'))}")
        deliverables = ", ".join([f"`{text(x)}`" for x in as_list(phase.get("deliverables")) if text(x)])
        if deliverables:
            lines.append(f"  deliverables: {deliverables}")
        done_when = text(phase.get("done_when"))
        if done_when:
            lines.append(f"  done_when: {done_when}")
    lines.extend(["", "## Immediate backlog"])
    for row in backlog:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `P{row.get('priority')}` `{text(row.get('title'))}` -> `{text(row.get('target_artifact'))}`: {text(row.get('why'))}"
        )
    lines.extend(["", "## Transmission control chain"])
    for row in control_chain:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `{text(row.get('stage'))}` `{text(row.get('label'))}`: "
            f"`{text(row.get('source_status'))}` -> `{text(row.get('next_target_artifact'))}` | "
            f"{text(row.get('source_brief'))}"
        )
        lines.append(f"  mission: {text(row.get('mission'))}")
        lines.append(f"  decision: {text(row.get('source_decision'))}")
        interface_fields = ", ".join([f"`{text(item)}`" for item in as_list(row.get("interface_fields")) if text(item)])
        if interface_fields:
            lines.append(f"  interface_fields: {interface_fields}")
    lines.extend(["", "## Continuous optimization backlog"])
    for row in continuous_optimization_backlog:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- `P{row.get('priority')}` `{text(row.get('title'))}` -> `{text(row.get('target_artifact'))}`: {text(row.get('why'))}"
        )
    return "\n".join(lines) + "\n"


def build_payload(
    *,
    handoff_path: Path,
    handoff_payload: dict[str, Any],
    live_gate_path: Path,
    live_gate_payload: dict[str, Any],
    cross_market_path: Path,
    cross_market_payload: dict[str, Any],
    hot_brief_path: Path,
    hot_brief_payload: dict[str, Any],
    history_path: Path | None,
    history_payload: dict[str, Any],
    identity_state_path: Path | None,
    identity_state_payload: dict[str, Any],
    scope_router_state_path: Path | None,
    scope_router_state_payload: dict[str, Any],
    intent_queue_state_path: Path | None,
    intent_queue_state_payload: dict[str, Any],
    execution_journal_state_path: Path | None,
    execution_journal_state_payload: dict[str, Any],
    executor_state_path: Path | None,
    executor_state_payload: dict[str, Any],
    feedback_state_path: Path | None,
    feedback_state_payload: dict[str, Any],
    policy_state_path: Path | None,
    policy_state_payload: dict[str, Any],
    ack_state_path: Path | None,
    ack_state_payload: dict[str, Any],
    actor_state_path: Path | None,
    actor_state_payload: dict[str, Any],
    guarded_transport_state_path: Path | None,
    guarded_transport_state_payload: dict[str, Any],
    transport_sla_state_path: Path | None,
    transport_sla_state_payload: dict[str, Any],
    canary_gate_state_path: Path | None,
    canary_gate_state_payload: dict[str, Any],
    quality_report_state_path: Path | None,
    quality_report_state_payload: dict[str, Any],
    live_boundary_hold_state_path: Path | None,
    live_boundary_hold_state_payload: dict[str, Any],
    promotion_gate_state_path: Path | None,
    promotion_gate_state_payload: dict[str, Any],
    shadow_learning_continuity_state_path: Path | None,
    shadow_learning_continuity_state_payload: dict[str, Any],
    promotion_unblock_readiness_state_path: Path | None,
    promotion_unblock_readiness_state_payload: dict[str, Any],
    ticket_actionability_state_path: Path | None,
    ticket_actionability_state_payload: dict[str, Any],
    signal_source_refresh_readiness_path: Path | None,
    signal_source_refresh_readiness_payload: dict[str, Any],
    signal_source_freshness_path: Path | None,
    signal_source_freshness_payload: dict[str, Any],
    material_change_trigger_path: Path | None,
    material_change_trigger_payload: dict[str, Any],
    shortline_backtest_slice_path: Path | None,
    shortline_backtest_slice_payload: dict[str, Any],
    shortline_cross_section_backtest_path: Path | None,
    shortline_cross_section_backtest_payload: dict[str, Any],
    guardian_blocker_clearance_state_path: Path | None,
    guardian_blocker_clearance_state_payload: dict[str, Any],
    shadow_clock_state_path: Path | None,
    shadow_clock_state_payload: dict[str, Any],
    reference_now: dt.datetime,
) -> dict[str, Any]:
    current = build_current_status(
        handoff_path=handoff_path,
        handoff=handoff_payload,
        live_gate_path=live_gate_path,
        live_gate=live_gate_payload,
        cross_market_path=cross_market_path,
        cross_market=cross_market_payload,
        hot_brief_path=hot_brief_path,
        hot_brief=hot_brief_payload,
        history_path=history_path,
        history=history_payload,
        identity_state_path=identity_state_path,
        identity_state=identity_state_payload,
        scope_router_state_path=scope_router_state_path,
        scope_router_state=scope_router_state_payload,
        intent_queue_state_path=intent_queue_state_path,
        intent_queue_state=intent_queue_state_payload,
        execution_journal_state_path=execution_journal_state_path,
        execution_journal_state=execution_journal_state_payload,
        executor_state_path=executor_state_path,
        executor_state=executor_state_payload,
        feedback_state_path=feedback_state_path,
        feedback_state=feedback_state_payload,
        policy_state_path=policy_state_path,
        policy_state=policy_state_payload,
        ack_state_path=ack_state_path,
        ack_state=ack_state_payload,
        actor_state_path=actor_state_path,
        actor_state=actor_state_payload,
        guarded_transport_state_path=guarded_transport_state_path,
        guarded_transport_state=guarded_transport_state_payload,
        transport_sla_state_path=transport_sla_state_path,
        transport_sla_state=transport_sla_state_payload,
        canary_gate_state_path=canary_gate_state_path,
        canary_gate_state=canary_gate_state_payload,
        quality_report_state_path=quality_report_state_path,
        quality_report_state=quality_report_state_payload,
        live_boundary_hold_state_path=live_boundary_hold_state_path,
        live_boundary_hold_state=live_boundary_hold_state_payload,
        promotion_gate_state_path=promotion_gate_state_path,
        promotion_gate_state=promotion_gate_state_payload,
        shadow_learning_continuity_state_path=shadow_learning_continuity_state_path,
        shadow_learning_continuity_state=shadow_learning_continuity_state_payload,
        promotion_unblock_readiness_state_path=promotion_unblock_readiness_state_path,
        promotion_unblock_readiness_state=promotion_unblock_readiness_state_payload,
        ticket_actionability_state_path=ticket_actionability_state_path,
        ticket_actionability_state=ticket_actionability_state_payload,
        signal_source_refresh_readiness_path=signal_source_refresh_readiness_path,
        signal_source_refresh_readiness=signal_source_refresh_readiness_payload,
        signal_source_freshness_path=signal_source_freshness_path,
        signal_source_freshness=signal_source_freshness_payload,
        material_change_trigger_path=material_change_trigger_path,
        material_change_trigger=material_change_trigger_payload,
        shortline_backtest_slice_path=shortline_backtest_slice_path,
        shortline_backtest_slice=shortline_backtest_slice_payload,
        shortline_cross_section_backtest_path=shortline_cross_section_backtest_path,
        shortline_cross_section_backtest=shortline_cross_section_backtest_payload,
        guardian_blocker_clearance_state_path=guardian_blocker_clearance_state_path,
        guardian_blocker_clearance_state=guardian_blocker_clearance_state_payload,
        shadow_clock_state_path=shadow_clock_state_path,
        shadow_clock_state=shadow_clock_state_payload,
        reference_now=reference_now,
    )
    control_chain = build_control_chain(current)
    return {
        "action": "build_openclaw_orderflow_blueprint",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "current_status": current,
        "digital_layers": build_digital_layers(current),
        "control_chain": control_chain,
        "refactor_phases": build_refactor_phases(current),
        "immediate_backlog": build_immediate_backlog(current),
        "continuous_optimization_backlog": build_continuous_optimization_backlog(current, control_chain),
        "code_anchors": build_code_anchors(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenClaw orderflow blueprint from current remote-live artifacts.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now) or now_utc()
    handoff_path = find_latest(review_dir, "*_remote_live_handoff.json", reference_now)
    live_gate_path = find_latest(review_dir, "*_live_gate_blocker_report.json", reference_now)
    cross_market_path = find_latest(review_dir, "*_cross_market_operator_state.json", reference_now)
    hot_brief_path = find_latest(review_dir, "*_hot_universe_operator_brief.json", reference_now)
    history_latest = review_dir / "latest_remote_live_history_audit.json"
    history_path = (
        history_latest
        if history_latest.exists()
        else find_latest(review_dir, "*_remote_live_history_audit.json", reference_now)
    )
    identity_state_path = find_latest(review_dir, "*_remote_execution_identity_state.json", reference_now)
    scope_router_state_path = find_latest(review_dir, "*_remote_scope_router_state.json", reference_now)
    intent_queue_state_path = find_latest(review_dir, "*_remote_intent_queue.json", reference_now)
    execution_journal_state_path = find_latest(
        review_dir, "*_remote_execution_journal.json", reference_now
    )
    executor_state_path = find_latest(review_dir, "*_openclaw_orderflow_executor_state.json", reference_now)
    feedback_state_path = find_latest(review_dir, "*_remote_orderflow_feedback.json", reference_now)
    policy_state_path = find_latest(review_dir, "*_remote_orderflow_policy_state.json", reference_now)
    ack_state_path = find_latest(review_dir, "*_remote_execution_ack_state.json", reference_now)
    actor_state_path = find_latest(review_dir, "*_remote_execution_actor_state.json", reference_now)
    guarded_transport_state_path = find_latest(
        review_dir, "*_remote_execution_actor_guarded_transport.json"
    )
    transport_sla_state_path = find_latest(
        review_dir, "*_remote_execution_transport_sla.json", reference_now
    )
    canary_gate_state_path = find_latest(
        review_dir, "*_remote_execution_actor_canary_gate.json", reference_now
    )
    quality_report_state_path = find_latest(
        review_dir, "*_remote_orderflow_quality_report.json", reference_now
    )
    live_boundary_hold_state_path = find_latest(
        review_dir, "*_remote_live_boundary_hold.json", reference_now
    )
    promotion_gate_state_path = find_latest(
        review_dir, "*_remote_guarded_canary_promotion_gate.json", reference_now
    )
    shadow_learning_continuity_state_path = find_latest(
        review_dir, "*_remote_shadow_learning_continuity.json", reference_now
    )
    promotion_unblock_readiness_state_path = find_latest(
        review_dir, "*_remote_promotion_unblock_readiness.json", reference_now
    )
    ticket_actionability_state_path = find_latest(
        review_dir, "*_remote_ticket_actionability_state.json", reference_now
    )
    signal_source_refresh_readiness_path = find_latest(
        review_dir, "*_crypto_signal_source_refresh_readiness.json", reference_now
    )
    signal_source_freshness_path = find_latest(
        review_dir, "*_crypto_signal_source_freshness.json", reference_now
    )
    material_change_trigger_path = find_latest(
        review_dir, "*_crypto_shortline_material_change_trigger.json", reference_now
    )
    shortline_backtest_slice_path = find_latest(
        review_dir, "*_crypto_shortline_backtest_slice.json", reference_now
    )
    shortline_cross_section_backtest_path = find_latest(
        review_dir, "*_crypto_shortline_cross_section_backtest.json", reference_now
    )
    shadow_clock_state_path = find_latest(
        review_dir, "*_remote_shadow_clock_evidence.json", reference_now
    )
    guardian_blocker_clearance_state_path = find_latest(
        review_dir, "*_remote_guardian_blocker_clearance.json", reference_now
    )

    missing = [
        name
        for name, path in (
            ("remote_live_handoff", handoff_path),
            ("live_gate_blocker_report", live_gate_path),
            ("cross_market_operator_state", cross_market_path),
            ("hot_universe_operator_brief", hot_brief_path),
        )
        if path is None
    ]
    if missing:
        raise SystemExit(f"missing_required_artifacts:{','.join(missing)}")

    handoff_payload = unwrap_operator_handoff(load_json_mapping(handoff_path))
    live_gate_payload = load_json_mapping(live_gate_path)
    cross_market_payload = load_json_mapping(cross_market_path)
    hot_brief_payload = load_json_mapping(hot_brief_path)
    history_payload = load_json_mapping(history_path) if history_path is not None and history_path.exists() else {}

    payload = build_payload(
        handoff_path=handoff_path,
        handoff_payload=handoff_payload,
        live_gate_path=live_gate_path,
        live_gate_payload=live_gate_payload,
        cross_market_path=cross_market_path,
        cross_market_payload=cross_market_payload,
        hot_brief_path=hot_brief_path,
        hot_brief_payload=hot_brief_payload,
        history_path=history_path,
        history_payload=history_payload,
        identity_state_path=identity_state_path,
        identity_state_payload=load_json_mapping(identity_state_path)
        if identity_state_path is not None and identity_state_path.exists()
        else {},
        scope_router_state_path=scope_router_state_path,
        scope_router_state_payload=load_json_mapping(scope_router_state_path)
        if scope_router_state_path is not None and scope_router_state_path.exists()
        else {},
        intent_queue_state_path=intent_queue_state_path,
        intent_queue_state_payload=load_json_mapping(intent_queue_state_path)
        if intent_queue_state_path is not None and intent_queue_state_path.exists()
        else {},
        execution_journal_state_path=execution_journal_state_path,
        execution_journal_state_payload=load_json_mapping(execution_journal_state_path)
        if execution_journal_state_path is not None and execution_journal_state_path.exists()
        else {},
        executor_state_path=executor_state_path,
        executor_state_payload=load_json_mapping(executor_state_path)
        if executor_state_path is not None and executor_state_path.exists()
        else {},
        feedback_state_path=feedback_state_path,
        feedback_state_payload=load_json_mapping(feedback_state_path)
        if feedback_state_path is not None and feedback_state_path.exists()
        else {},
        policy_state_path=policy_state_path,
        policy_state_payload=load_json_mapping(policy_state_path)
        if policy_state_path is not None and policy_state_path.exists()
        else {},
        ack_state_path=ack_state_path,
        ack_state_payload=load_json_mapping(ack_state_path)
        if ack_state_path is not None and ack_state_path.exists()
        else {},
        actor_state_path=actor_state_path,
        actor_state_payload=load_json_mapping(actor_state_path)
        if actor_state_path is not None and actor_state_path.exists()
        else {},
        guarded_transport_state_path=guarded_transport_state_path,
        guarded_transport_state_payload=load_json_mapping(guarded_transport_state_path)
        if guarded_transport_state_path is not None and guarded_transport_state_path.exists()
        else {},
        transport_sla_state_path=transport_sla_state_path,
        transport_sla_state_payload=load_json_mapping(transport_sla_state_path)
        if transport_sla_state_path is not None and transport_sla_state_path.exists()
        else {},
        canary_gate_state_path=canary_gate_state_path,
        canary_gate_state_payload=load_json_mapping(canary_gate_state_path)
        if canary_gate_state_path is not None and canary_gate_state_path.exists()
        else {},
        quality_report_state_path=quality_report_state_path,
        quality_report_state_payload=load_json_mapping(quality_report_state_path)
        if quality_report_state_path is not None and quality_report_state_path.exists()
        else {},
        live_boundary_hold_state_path=live_boundary_hold_state_path,
        live_boundary_hold_state_payload=load_json_mapping(live_boundary_hold_state_path)
        if live_boundary_hold_state_path is not None and live_boundary_hold_state_path.exists()
        else {},
        promotion_gate_state_path=promotion_gate_state_path,
        promotion_gate_state_payload=load_json_mapping(promotion_gate_state_path)
        if promotion_gate_state_path is not None and promotion_gate_state_path.exists()
        else {},
        shadow_learning_continuity_state_path=shadow_learning_continuity_state_path,
        shadow_learning_continuity_state_payload=load_json_mapping(shadow_learning_continuity_state_path)
        if shadow_learning_continuity_state_path is not None
        and shadow_learning_continuity_state_path.exists()
        else {},
        promotion_unblock_readiness_state_path=promotion_unblock_readiness_state_path,
        promotion_unblock_readiness_state_payload=load_json_mapping(promotion_unblock_readiness_state_path)
        if promotion_unblock_readiness_state_path is not None
        and promotion_unblock_readiness_state_path.exists()
        else {},
        ticket_actionability_state_path=ticket_actionability_state_path,
        ticket_actionability_state_payload=load_json_mapping(ticket_actionability_state_path)
        if ticket_actionability_state_path is not None
        and ticket_actionability_state_path.exists()
        else {},
        signal_source_refresh_readiness_path=signal_source_refresh_readiness_path,
        signal_source_refresh_readiness_payload=load_json_mapping(signal_source_refresh_readiness_path)
        if signal_source_refresh_readiness_path is not None
        and signal_source_refresh_readiness_path.exists()
        else {},
        signal_source_freshness_path=signal_source_freshness_path,
        signal_source_freshness_payload=load_json_mapping(signal_source_freshness_path)
        if signal_source_freshness_path is not None
        and signal_source_freshness_path.exists()
        else {},
        material_change_trigger_path=material_change_trigger_path,
        material_change_trigger_payload=load_json_mapping(material_change_trigger_path)
        if material_change_trigger_path is not None
        and material_change_trigger_path.exists()
        else {},
        shortline_backtest_slice_path=shortline_backtest_slice_path,
        shortline_backtest_slice_payload=load_json_mapping(shortline_backtest_slice_path)
        if shortline_backtest_slice_path is not None
        and shortline_backtest_slice_path.exists()
        else {},
        shortline_cross_section_backtest_path=shortline_cross_section_backtest_path,
        shortline_cross_section_backtest_payload=load_json_mapping(shortline_cross_section_backtest_path)
        if shortline_cross_section_backtest_path is not None
        and shortline_cross_section_backtest_path.exists()
        else {},
        guardian_blocker_clearance_state_path=guardian_blocker_clearance_state_path,
        guardian_blocker_clearance_state_payload=load_json_mapping(guardian_blocker_clearance_state_path)
        if guardian_blocker_clearance_state_path is not None
        and guardian_blocker_clearance_state_path.exists()
        else {},
        shadow_clock_state_path=shadow_clock_state_path,
        shadow_clock_state_payload=load_json_mapping(shadow_clock_state_path)
        if shadow_clock_state_path is not None and shadow_clock_state_path.exists()
        else {},
        reference_now=reference_now,
    )

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_openclaw_orderflow_blueprint.json"
    markdown = review_dir / f"{stamp}_openclaw_orderflow_blueprint.md"
    checksum = review_dir / f"{stamp}_openclaw_orderflow_blueprint_checksum.json"

    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
