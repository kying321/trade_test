#!/usr/bin/env python3
"""Shared notify trend analytics for pi_cycle events."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_json_line(raw: str) -> Optional[Dict[str, Any]]:
    line = str(raw or "").strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _parse_ts_utc(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _as_non_negative(value: Any, default: float) -> float:
    parsed = _as_float(value)
    if parsed is None:
        return float(default)
    return max(0.0, float(parsed))


def _status_weight_score(status: str) -> float:
    if status == "critical":
        return 2.0
    if status == "degraded":
        return 1.0
    return 0.0


def _component_status(
    *,
    count: int,
    total: int,
    rate: Optional[float],
    min_total: int,
    min_count: int,
    degraded_rate: float,
    critical_rate: float,
) -> Dict[str, str]:
    if total < min_total:
        return {"status": "unknown", "reason": "insufficient_samples"}
    if rate is None:
        return {"status": "unknown", "reason": "rate_missing"}
    if count < min_count:
        return {"status": "ok", "reason": "below_min_count"}
    if float(rate) >= float(critical_rate):
        return {"status": "critical", "reason": "rate_critical"}
    if float(rate) >= float(degraded_rate):
        return {"status": "degraded", "reason": "rate_degraded"}
    return {"status": "ok", "reason": "within_threshold"}


def compute_notify_trend(
    path: Path,
    *,
    lines: Optional[List[str]] = None,
    now_utc: Optional[dt.datetime] = None,
    include_latest_event: bool = False,
) -> Dict[str, Any]:
    window_hours = max(0.5, float(os.getenv("PI_GATE_NOTIFY_TREND_HOURS", "24")))
    max_lines = max(500, int(float(os.getenv("PI_GATE_NOTIFY_TREND_MAX_LINES", "5000"))))
    min_attempts = max(1, int(float(os.getenv("PI_GATE_NOTIFY_TREND_MIN_ATTEMPTS", "3"))))
    fail_min_count = max(1, int(float(os.getenv("PI_GATE_NOTIFY_TREND_FAIL_MIN_COUNT", "1"))))
    fail_rate_degraded = float(os.getenv("PI_GATE_NOTIFY_TREND_FAIL_RATE_DEGRADED", "0.40"))
    fail_rate_critical = float(os.getenv("PI_GATE_NOTIFY_TREND_FAIL_RATE_CRITICAL", "0.80"))
    suppression_min_emit_events = max(
        1,
        int(float(os.getenv("PI_GATE_NOTIFY_TREND_SUPPRESSION_MIN_EMIT_EVENTS", "6"))),
    )
    suppression_min_count = max(
        1,
        int(float(os.getenv("PI_GATE_NOTIFY_TREND_SUPPRESSION_MIN_COUNT", "2"))),
    )
    suppression_rate_degraded = float(
        os.getenv("PI_GATE_NOTIFY_TREND_SUPPRESSION_RATE_DEGRADED", "0.40")
    )
    suppression_rate_critical = float(
        os.getenv("PI_GATE_NOTIFY_TREND_SUPPRESSION_RATE_CRITICAL", "0.70")
    )
    min_interval_min_emit_events = max(
        1,
        int(float(os.getenv("PI_GATE_NOTIFY_TREND_MIN_INTERVAL_MIN_EMIT_EVENTS", "6"))),
    )
    min_interval_min_count = max(
        1,
        int(float(os.getenv("PI_GATE_NOTIFY_TREND_MIN_INTERVAL_MIN_COUNT", "2"))),
    )
    min_interval_rate_degraded = float(
        os.getenv("PI_GATE_NOTIFY_TREND_MIN_INTERVAL_RATE_DEGRADED", "0.50")
    )
    min_interval_rate_critical = float(
        os.getenv("PI_GATE_NOTIFY_TREND_MIN_INTERVAL_RATE_CRITICAL", "0.80")
    )
    component_fail_weight = _as_non_negative(os.getenv("PI_GATE_NOTIFY_FAIL_WEIGHT", "0.60"), 0.60)
    component_suppression_weight = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_SUPPRESSION_WEIGHT", "0.25"),
        0.25,
    )
    component_min_interval_weight = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_MIN_INTERVAL_WEIGHT", "0.15"),
        0.15,
    )
    component_patch_apply_weight = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_PATCH_APPLY_WEIGHT", "0.0"),
        0.0,
    )
    component_score_degraded = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_COMPONENT_SCORE_DEGRADED", "0.75"),
        0.75,
    )
    component_score_critical = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_COMPONENT_SCORE_PROTECTIVE", "1.25"),
        1.25,
    )
    if component_score_critical < component_score_degraded:
        component_score_critical = component_score_degraded
    shadow_min_mode_events = max(
        1,
        int(float(os.getenv("PI_GATE_ALERT_COMPONENT_SCORE_SHADOW_MIN_EVENTS", "24"))),
    )
    shadow_max_would_upgrade_rate = _as_non_negative(
        os.getenv("PI_GATE_ALERT_COMPONENT_SCORE_SHADOW_MAX_WOULD_UPGRADE_RATE", "0.30"),
        0.30,
    )
    shadow_max_critical_target_share = _as_non_negative(
        os.getenv("PI_GATE_ALERT_COMPONENT_SCORE_SHADOW_MAX_CRITICAL_SHARE", "0.20"),
        0.20,
    )
    patch_action_min_events = max(
        1,
        int(float(os.getenv("PI_GATE_NOTIFY_TREND_PATCH_ACTION_MIN_EVENTS", "6"))),
    )
    patch_escalated_rate_degraded = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_TREND_PATCH_ESCALATED_RATE_DEGRADED", "0.40"),
        0.40,
    )
    patch_escalated_rate_critical = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_TREND_PATCH_ESCALATED_RATE_CRITICAL", "0.70"),
        0.70,
    )
    if patch_escalated_rate_critical < patch_escalated_rate_degraded:
        patch_escalated_rate_critical = patch_escalated_rate_degraded
    patch_next_action_min_share = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_TREND_PATCH_NEXT_ACTION_MIN_SHARE", "0.34"),
        0.34,
    )
    patch_next_action_min_share = min(1.0, patch_next_action_min_share)
    patch_next_action_min_count = max(
        1,
        int(float(os.getenv("PI_GATE_NOTIFY_TREND_PATCH_NEXT_ACTION_MIN_COUNT", "2"))),
    )
    patch_apply_batch_id = (
        str(os.getenv("PI_GATE_NOTIFY_TREND_PATCH_APPLY_BATCH_ID", "batch_0_proxy_isolation")).strip()
        or "batch_0_proxy_isolation"
    )
    patch_apply_min_events = max(
        1,
        int(float(os.getenv("PI_GATE_NOTIFY_TREND_PATCH_APPLY_MIN_EVENTS", "2"))),
    )
    patch_apply_problem_rate_degraded = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_TREND_PATCH_APPLY_RATE_DEGRADED", "0.34"),
        0.34,
    )
    patch_apply_problem_rate_critical = _as_non_negative(
        os.getenv("PI_GATE_NOTIFY_TREND_PATCH_APPLY_RATE_CRITICAL", "0.67"),
        0.67,
    )
    if patch_apply_problem_rate_critical < patch_apply_problem_rate_degraded:
        patch_apply_problem_rate_critical = patch_apply_problem_rate_degraded
    shadow_max_would_upgrade_rate = min(1.0, shadow_max_would_upgrade_rate)
    shadow_max_critical_target_share = min(1.0, shadow_max_critical_target_share)
    now = now_utc or dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=window_hours)

    out: Dict[str, Any] = {
        "status": "unknown",
        "reason": "",
        "window_hours": window_hours,
        "path": str(path),
        "events": 0,
        "emit_events": 0,
        "notify_records": 0,
        "notify_enabled_events": 0,
        "attempted": 0,
        "sent": 0,
        "failed": 0,
        "fail_rate": None,
        "success_rate": None,
        "min_attempts_required": min_attempts,
        "thresholds": {
            "fail": {
                "min_attempts": min_attempts,
                "min_count": fail_min_count,
                "rate_degraded": fail_rate_degraded,
                "rate_critical": fail_rate_critical,
            },
            "suppression": {
                "min_emit_events": suppression_min_emit_events,
                "min_count": suppression_min_count,
                "rate_degraded": suppression_rate_degraded,
                "rate_critical": suppression_rate_critical,
            },
            "min_interval": {
                "min_emit_events": min_interval_min_emit_events,
                "min_count": min_interval_min_count,
                "rate_degraded": min_interval_rate_degraded,
                "rate_critical": min_interval_rate_critical,
            },
            "component_score": {
                "degraded": component_score_degraded,
                "critical": component_score_critical,
            },
            "component_score_shadow_rollout": {
                "min_mode_events": shadow_min_mode_events,
                "max_would_upgrade_rate": shadow_max_would_upgrade_rate,
                "max_critical_target_share": shadow_max_critical_target_share,
            },
            "patch_action": {
                "min_events": patch_action_min_events,
                "escalated_rate_degraded": patch_escalated_rate_degraded,
                "escalated_rate_critical": patch_escalated_rate_critical,
                "next_action_min_share": patch_next_action_min_share,
                "next_action_min_count": patch_next_action_min_count,
            },
            "patch_apply": {
                "batch_id": patch_apply_batch_id,
                "min_events": patch_apply_min_events,
                "problem_rate_degraded": patch_apply_problem_rate_degraded,
                "problem_rate_critical": patch_apply_problem_rate_critical,
            },
        },
        "component_weights": {
            "fail": component_fail_weight,
            "suppression": component_suppression_weight,
            "min_interval": component_min_interval_weight,
            "patch_apply": component_patch_apply_weight,
        },
        "fail_status": "unknown",
        "fail_reason": "insufficient_samples",
        "suppression_status": "unknown",
        "suppression_reason": "insufficient_samples",
        "min_interval_status": "unknown",
        "min_interval_reason": "insufficient_samples",
        "component_score": None,
        "component_known": False,
        "component_score_status": "unknown",
        "component_score_reason": "insufficient_samples",
        "status_reason": "insufficient_samples",
        "suppressed": 0,
        "suppressed_rate": None,
        "suppression_reasons_top": [],
        "min_interval_escalated": 0,
        "min_interval_escalation_rate": None,
        "min_interval_sources_top": [],
        "component_score_shadow_mode_events": 0,
        "component_score_shadow_would_upgrade": 0,
        "component_score_shadow_would_upgrade_rate": None,
        "component_score_shadow_critical_target_count": 0,
        "component_score_shadow_critical_target_share": None,
        "component_score_shadow_target_status_top": [],
        "component_score_shadow_recommend_enforce": None,
        "component_score_shadow_recommend_reason": "insufficient_shadow_mode_events",
        "patch_action_events": 0,
        "patch_action_escalated": 0,
        "patch_action_escalated_rate": None,
        "patch_action_pending": 0,
        "patch_action_pending_rate": None,
        "patch_action_status": "unknown",
        "patch_action_status_reason": "insufficient_patch_action_events",
        "patch_action_levels_top": [],
        "patch_action_reasons_top": [],
        "patch_next_actions_top": [],
        "patch_action_hints_top": [],
        "patch_dominant_next_action": None,
        "patch_dominant_next_action_count": 0,
        "patch_dominant_next_action_share": None,
        "patch_dominant_action_reason": None,
        "patch_dominant_action_reason_count": 0,
        "patch_dominant_action_reason_share": None,
        "patch_recommended_next_action": None,
        "patch_recommended_reason": "insufficient_patch_action_events",
        "patch_recommended_confidence": None,
        "patch_apply_events": 0,
        "patch_apply_problem_events": 0,
        "patch_apply_problem_rate": None,
        "patch_apply_critical_problem_events": 0,
        "patch_apply_critical_problem_rate": None,
        "patch_apply_status": "unknown",
        "patch_apply_status_reason": "insufficient_patch_apply_events",
        "patch_apply_statuses_top": [],
        "patch_apply_reasons_top": [],
        "patch_apply_modes_top": [],
        "patch_apply_target_batch_id": patch_apply_batch_id,
        "patch_apply_dominant_reason": None,
        "patch_apply_dominant_reason_count": 0,
        "patch_apply_dominant_reason_share": None,
        "failure_reasons_top": [],
        "skip_reasons_top": [],
    }

    local_lines: List[str]
    if lines is None:
        if not path.exists():
            out["reason"] = "pi_cycle_log_missing"
            return out
        try:
            local_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            out["status"] = "degraded"
            out["reason"] = "pi_cycle_log_unreadable"
            return out
    else:
        local_lines = list(lines)

    latest_event: Optional[Dict[str, Any]] = None
    fail_reason_counts: Dict[str, int] = {}
    skip_reason_counts: Dict[str, int] = {}
    suppression_reason_counts: Dict[str, int] = {}
    min_interval_source_counts: Dict[str, int] = {}
    shadow_target_status_counts: Dict[str, int] = {}
    patch_level_counts: Dict[str, int] = {}
    patch_reason_counts: Dict[str, int] = {}
    patch_next_action_counts: Dict[str, int] = {}
    patch_hint_counts: Dict[str, int] = {}
    patch_apply_status_counts: Dict[str, int] = {}
    patch_apply_reason_counts: Dict[str, int] = {}
    patch_apply_mode_counts: Dict[str, int] = {}

    for raw in reversed(local_lines[-max_lines:]):
        obj = _parse_json_line(raw)
        if not obj or str(obj.get("domain") or "") != "pi_cycle":
            continue
        if latest_event is None:
            latest_event = obj

        ts = _parse_ts_utc(obj.get("ts"))
        if ts is None or ts < since:
            continue

        out["events"] = int(out["events"]) + 1
        emit = obj.get("gate_rollout_alert_emit") if isinstance(obj.get("gate_rollout_alert_emit"), dict) else {}
        notify = emit.get("notify") if isinstance(emit.get("notify"), dict) else {}
        if emit:
            out["emit_events"] = int(out["emit_events"]) + 1
            suppressed = bool(emit.get("suppressed"))
            emit_reason = str(emit.get("reason") or "").strip()
            suppression_reason = str(emit.get("suppression_reason") or "").strip()
            if (not suppressed) and emit_reason.startswith("suppressed_notify_trend_"):
                suppressed = True
                if not suppression_reason:
                    suppression_reason = emit_reason.replace("suppressed_", "", 1)
            if suppressed:
                out["suppressed"] = int(out["suppressed"]) + 1
                key = suppression_reason or "unknown"
                suppression_reason_counts[key] = suppression_reason_counts.get(key, 0) + 1

            min_interval_source = str(emit.get("min_interval_source") or "").strip()
            min_interval_base = emit.get("min_interval_sec_base")
            min_interval_effective = emit.get("min_interval_sec_effective")
            escalated = min_interval_source in {"notify_trend_degraded", "notify_trend_critical"}
            if (not escalated) and (min_interval_base is not None) and (min_interval_effective is not None):
                try:
                    escalated = float(min_interval_effective) > float(min_interval_base)
                except Exception:
                    escalated = False
            if escalated:
                out["min_interval_escalated"] = int(out["min_interval_escalated"]) + 1
                key = min_interval_source or "numeric_only"
                min_interval_source_counts[key] = min_interval_source_counts.get(key, 0) + 1

            shadow_mode = str(emit.get("gate_notify_trend_component_score_mode_effective") or "").strip().lower()
            has_shadow_observation = "gate_notify_trend_component_score_shadow_would_upgrade" in emit
            if shadow_mode == "shadow" and has_shadow_observation:
                out["component_score_shadow_mode_events"] = int(out["component_score_shadow_mode_events"]) + 1
                if bool(emit.get("gate_notify_trend_component_score_shadow_would_upgrade")):
                    out["component_score_shadow_would_upgrade"] = int(out["component_score_shadow_would_upgrade"]) + 1
                    target_status = str(
                        emit.get("gate_notify_trend_component_score_shadow_target_status") or "unknown"
                    ).strip().lower()
                    if target_status:
                        shadow_target_status_counts[target_status] = (
                            shadow_target_status_counts.get(target_status, 0) + 1
                        )

            patch_level_raw = str(emit.get("patch_draft_action_level") or "").strip().lower()
            patch_reason_raw = str(emit.get("patch_draft_action_reason") or "").strip()
            patch_next_action_raw = str(emit.get("patch_draft_next_action") or "").strip()
            patch_pending_raw = emit.get("patch_draft_pending")
            patch_hints_raw = emit.get("patch_draft_action_hints")
            has_patch_signal = bool(
                patch_level_raw
                or patch_reason_raw
                or patch_next_action_raw
                or (patch_pending_raw is not None)
                or isinstance(patch_hints_raw, list)
            )
            if has_patch_signal:
                out["patch_action_events"] = int(out["patch_action_events"]) + 1
                patch_level = (
                    patch_level_raw
                    if patch_level_raw in {"observe", "degrade", "shadow_lock"}
                    else "unknown"
                )
                patch_level_counts[patch_level] = patch_level_counts.get(patch_level, 0) + 1
                if patch_level in {"degrade", "shadow_lock"}:
                    out["patch_action_escalated"] = int(out["patch_action_escalated"]) + 1
                if bool(patch_pending_raw):
                    out["patch_action_pending"] = int(out["patch_action_pending"]) + 1
                patch_reason = patch_reason_raw or "unknown"
                patch_reason_counts[patch_reason] = patch_reason_counts.get(patch_reason, 0) + 1
                patch_next_action = patch_next_action_raw or "none"
                patch_next_action_counts[patch_next_action] = (
                    patch_next_action_counts.get(patch_next_action, 0) + 1
                )
                if isinstance(patch_hints_raw, list):
                    for hint in patch_hints_raw[:5]:
                        hint_text = str(hint or "").strip()
                        if not hint_text:
                            continue
                        patch_hint_counts[hint_text] = patch_hint_counts.get(hint_text, 0) + 1

        patch_apply_batch_raw = str(obj.get("cron_policy_apply_batch_batch_id") or "").strip()
        patch_apply_status_raw = str(obj.get("cron_policy_apply_batch_status") or "").strip().lower()
        patch_apply_mode_raw = str(obj.get("cron_policy_apply_batch_mode") or "").strip().lower()
        patch_apply_present_raw = obj.get("cron_policy_apply_batch_present")
        has_patch_apply_signal = bool(
            patch_apply_batch_raw
            or patch_apply_status_raw
            or patch_apply_mode_raw
            or (patch_apply_present_raw is not None)
        )
        if has_patch_apply_signal and patch_apply_batch_raw == patch_apply_batch_id:
            out["patch_apply_events"] = int(out["patch_apply_events"]) + 1
            patch_apply_status = (
                patch_apply_status_raw
                if patch_apply_status_raw in {"ok", "degraded", "critical", "noop", "dry_run", "unknown"}
                else "unknown"
            )
            patch_apply_status_counts[patch_apply_status] = patch_apply_status_counts.get(patch_apply_status, 0) + 1
            patch_apply_mode = patch_apply_mode_raw or "unknown"
            patch_apply_mode_counts[patch_apply_mode] = patch_apply_mode_counts.get(patch_apply_mode, 0) + 1
            selected_actions = max(0, int(obj.get("cron_policy_apply_batch_selected_actions") or 0))
            changed_jobs = max(0, int(obj.get("cron_policy_apply_batch_changed_jobs") or 0))
            failed_jobs = max(0, int(obj.get("cron_policy_apply_batch_failed_jobs") or 0))
            blocked_jobs = max(0, int(obj.get("cron_policy_apply_batch_blocked_jobs") or 0))
            operations_changed = max(0, int(obj.get("cron_policy_apply_batch_operations_changed") or 0))

            is_critical_problem = failed_jobs > 0 or blocked_jobs > 0 or patch_apply_status == "critical"
            is_problem = is_critical_problem
            reason_key = "patch_apply_ok"
            if is_critical_problem:
                reason_key = "patch_apply_failed_or_blocked"
            elif patch_apply_status in {"degraded", "unknown"}:
                is_problem = True
                reason_key = f"patch_apply_status_{patch_apply_status}"
            elif (
                patch_apply_mode == "apply"
                and selected_actions > 0
                and changed_jobs <= 0
                and operations_changed <= 0
            ):
                is_problem = True
                reason_key = "patch_apply_no_effect_apply"

            if is_problem:
                out["patch_apply_problem_events"] = int(out["patch_apply_problem_events"]) + 1
                patch_apply_reason_counts[reason_key] = patch_apply_reason_counts.get(reason_key, 0) + 1
                if is_critical_problem:
                    out["patch_apply_critical_problem_events"] = int(
                        out["patch_apply_critical_problem_events"]
                    ) + 1

        if not notify:
            continue

        out["notify_records"] = int(out["notify_records"]) + 1
        if bool(notify.get("enabled")):
            out["notify_enabled_events"] = int(out["notify_enabled_events"]) + 1
        reason = str(notify.get("reason") or "").strip()
        sent = bool(notify.get("sent"))
        attempted = bool(notify.get("attempt_count")) or bool(notify.get("attempts"))
        if not attempted and reason in {"sent", "remote_error", "http_error"}:
            attempted = True
        if not attempted and reason.startswith("send_failed:"):
            attempted = True

        if attempted:
            out["attempted"] = int(out["attempted"]) + 1
            if sent:
                out["sent"] = int(out["sent"]) + 1
            else:
                out["failed"] = int(out["failed"]) + 1
                key = reason or "unknown"
                fail_reason_counts[key] = fail_reason_counts.get(key, 0) + 1
        else:
            key = reason or "unknown"
            skip_reason_counts[key] = skip_reason_counts.get(key, 0) + 1

    attempted = int(out["attempted"])
    sent = int(out["sent"])
    failed = int(out["failed"])
    emit_events = int(out["emit_events"])
    suppressed = int(out["suppressed"])
    min_interval_escalated = int(out["min_interval_escalated"])
    shadow_mode_events = int(out["component_score_shadow_mode_events"])
    shadow_would_upgrade = int(out["component_score_shadow_would_upgrade"])
    patch_action_events = int(out["patch_action_events"])
    patch_action_escalated = int(out["patch_action_escalated"])
    patch_action_pending = int(out["patch_action_pending"])
    patch_apply_events = int(out["patch_apply_events"])
    patch_apply_problem_events = int(out["patch_apply_problem_events"])
    patch_apply_critical_problem_events = int(out["patch_apply_critical_problem_events"])
    if attempted > 0:
        out["success_rate"] = round(float(sent) / float(attempted), 4)
        out["fail_rate"] = round(float(failed) / float(attempted), 4)
    if emit_events > 0:
        out["suppressed_rate"] = round(float(suppressed) / float(emit_events), 4)
        out["min_interval_escalation_rate"] = round(float(min_interval_escalated) / float(emit_events), 4)
    if shadow_mode_events > 0:
        out["component_score_shadow_would_upgrade_rate"] = round(
            float(shadow_would_upgrade) / float(shadow_mode_events),
            4,
        )
        shadow_critical_target_count = int(shadow_target_status_counts.get("critical", 0))
        out["component_score_shadow_critical_target_count"] = shadow_critical_target_count
        out["component_score_shadow_critical_target_share"] = round(
            float(shadow_critical_target_count) / float(shadow_mode_events),
            4,
        )
    if patch_action_events > 0:
        out["patch_action_escalated_rate"] = round(
            float(patch_action_escalated) / float(patch_action_events),
            4,
        )
        out["patch_action_pending_rate"] = round(
            float(patch_action_pending) / float(patch_action_events),
            4,
        )
    if patch_apply_events > 0:
        out["patch_apply_problem_rate"] = round(
            float(patch_apply_problem_events) / float(patch_apply_events),
            4,
        )
        out["patch_apply_critical_problem_rate"] = round(
            float(patch_apply_critical_problem_events) / float(patch_apply_events),
            4,
        )

    out["failure_reasons_top"] = [
        {"reason": k, "count": v}
        for k, v in sorted(fail_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    out["skip_reasons_top"] = [
        {"reason": k, "count": v}
        for k, v in sorted(skip_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    out["suppression_reasons_top"] = [
        {"reason": k, "count": v}
        for k, v in sorted(suppression_reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    out["min_interval_sources_top"] = [
        {"source": k, "count": v}
        for k, v in sorted(min_interval_source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    out["component_score_shadow_target_status_top"] = [
        {"status": k, "count": v}
        for k, v in sorted(shadow_target_status_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    out["patch_action_levels_top"] = [
        {"level": k, "count": v}
        for k, v in sorted(patch_level_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]
    out["patch_action_reasons_top"] = [
        {"reason": k, "count": v}
        for k, v in sorted(patch_reason_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]
    out["patch_next_actions_top"] = [
        {"action": k, "count": v}
        for k, v in sorted(patch_next_action_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]
    out["patch_action_hints_top"] = [
        {"hint": k, "count": v}
        for k, v in sorted(patch_hint_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]
    out["patch_apply_statuses_top"] = [
        {"status": k, "count": v}
        for k, v in sorted(patch_apply_status_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]
    out["patch_apply_reasons_top"] = [
        {"reason": k, "count": v}
        for k, v in sorted(patch_apply_reason_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]
    out["patch_apply_modes_top"] = [
        {"mode": k, "count": v}
        for k, v in sorted(patch_apply_mode_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
    ]
    top_next_actions = [x for x in out["patch_next_actions_top"] if isinstance(x, dict)]
    if top_next_actions:
        dominant_next_action = str(top_next_actions[0].get("action") or "")
        dominant_next_action_count = int(top_next_actions[0].get("count") or 0)
        out["patch_dominant_next_action"] = dominant_next_action or None
        out["patch_dominant_next_action_count"] = dominant_next_action_count
        if patch_action_events > 0:
            out["patch_dominant_next_action_share"] = round(
                float(dominant_next_action_count) / float(patch_action_events),
                4,
            )
    top_reasons = [x for x in out["patch_action_reasons_top"] if isinstance(x, dict)]
    if top_reasons:
        dominant_reason = str(top_reasons[0].get("reason") or "")
        dominant_reason_count = int(top_reasons[0].get("count") or 0)
        out["patch_dominant_action_reason"] = dominant_reason or None
        out["patch_dominant_action_reason_count"] = dominant_reason_count
        if patch_action_events > 0:
            out["patch_dominant_action_reason_share"] = round(
                float(dominant_reason_count) / float(patch_action_events),
                4,
            )
    patch_component = _component_status(
        count=patch_action_escalated,
        total=patch_action_events,
        rate=_as_float(out.get("patch_action_escalated_rate")),
        min_total=patch_action_min_events,
        min_count=1,
        degraded_rate=patch_escalated_rate_degraded,
        critical_rate=patch_escalated_rate_critical,
    )
    out["patch_action_status"] = patch_component.get("status")
    out["patch_action_status_reason"] = patch_component.get("reason")
    if patch_action_events < patch_action_min_events:
        out["patch_recommended_next_action"] = None
        out["patch_recommended_reason"] = "insufficient_patch_action_events"
        out["patch_recommended_confidence"] = None
    else:
        dominant_next_action = str(out.get("patch_dominant_next_action") or "").strip()
        dominant_next_action_count = int(out.get("patch_dominant_next_action_count") or 0)
        dominant_next_action_share = _as_float(out.get("patch_dominant_next_action_share"))
        if (
            dominant_next_action
            and dominant_next_action != "none"
            and (
                dominant_next_action_count >= patch_next_action_min_count
                or (
                    dominant_next_action_share is not None
                    and float(dominant_next_action_share) >= float(patch_next_action_min_share)
                )
            )
        ):
            out["patch_recommended_next_action"] = dominant_next_action
            out["patch_recommended_reason"] = "dominant_patch_next_action"
            out["patch_recommended_confidence"] = dominant_next_action_share
        else:
            dominant_reason = str(out.get("patch_dominant_action_reason") or "").strip()
            fallback_map = {
                "patch_status_critical": "inspect_patch_status_critical",
                "patch_failed_jobs": "resolve_patch_failures",
                "patch_pending_degraded": "run_guarded_patch_rollout",
                "patch_pending": "continue_patch_rollout",
                "patch_draft_missing": "wait_for_patch_draft",
            }
            fallback_action = fallback_map.get(dominant_reason)
            if fallback_action:
                out["patch_recommended_next_action"] = fallback_action
                out["patch_recommended_reason"] = "dominant_patch_action_reason"
                out["patch_recommended_confidence"] = _as_float(out.get("patch_dominant_action_reason_share"))
            else:
                out["patch_recommended_next_action"] = "continue_observation"
                out["patch_recommended_reason"] = "no_dominant_patch_next_action"
                out["patch_recommended_confidence"] = None
    patch_apply_component = _component_status(
        count=patch_apply_problem_events,
        total=patch_apply_events,
        rate=_as_float(out.get("patch_apply_problem_rate")),
        min_total=patch_apply_min_events,
        min_count=1,
        degraded_rate=patch_apply_problem_rate_degraded,
        critical_rate=patch_apply_problem_rate_critical,
    )
    patch_apply_status = str(patch_apply_component.get("status") or "unknown")
    patch_apply_reason = str(patch_apply_component.get("reason") or "unknown")
    if (
        patch_apply_events >= patch_apply_min_events
        and patch_apply_critical_problem_events > 0
        and patch_apply_status in {"ok", "degraded"}
    ):
        patch_apply_status = "critical"
        patch_apply_reason = "critical_apply_failure_seen"
    patch_apply_component["status"] = patch_apply_status
    patch_apply_component["reason"] = patch_apply_reason
    out["patch_apply_status"] = patch_apply_status
    out["patch_apply_status_reason"] = patch_apply_reason
    top_apply_reasons = [x for x in out["patch_apply_reasons_top"] if isinstance(x, dict)]
    if top_apply_reasons:
        dominant_apply_reason = str(top_apply_reasons[0].get("reason") or "")
        dominant_apply_reason_count = int(top_apply_reasons[0].get("count") or 0)
        out["patch_apply_dominant_reason"] = dominant_apply_reason or None
        out["patch_apply_dominant_reason_count"] = dominant_apply_reason_count
        if patch_apply_events > 0:
            out["patch_apply_dominant_reason_share"] = round(
                float(dominant_apply_reason_count) / float(patch_apply_events),
                4,
            )
    if shadow_mode_events < shadow_min_mode_events:
        out["component_score_shadow_recommend_enforce"] = None
        out["component_score_shadow_recommend_reason"] = "insufficient_shadow_mode_events"
    else:
        shadow_rate = _as_float(out.get("component_score_shadow_would_upgrade_rate"))
        critical_share = _as_float(out.get("component_score_shadow_critical_target_share"))
        if shadow_rate is None:
            out["component_score_shadow_recommend_enforce"] = None
            out["component_score_shadow_recommend_reason"] = "shadow_rate_missing"
        elif float(shadow_rate) > float(shadow_max_would_upgrade_rate):
            out["component_score_shadow_recommend_enforce"] = False
            out["component_score_shadow_recommend_reason"] = "shadow_would_upgrade_rate_high"
        elif critical_share is not None and float(critical_share) > float(shadow_max_critical_target_share):
            out["component_score_shadow_recommend_enforce"] = False
            out["component_score_shadow_recommend_reason"] = "shadow_critical_share_high"
        else:
            out["component_score_shadow_recommend_enforce"] = True
            out["component_score_shadow_recommend_reason"] = "shadow_stable_for_enforce"

    if int(out["events"]) == 0:
        out["reason"] = "no_events_in_window"
        if include_latest_event:
            out["latest_event"] = latest_event
        return out

    fail_component = _component_status(
        count=failed,
        total=attempted,
        rate=_as_float(out.get("fail_rate")),
        min_total=min_attempts,
        min_count=fail_min_count,
        degraded_rate=fail_rate_degraded,
        critical_rate=fail_rate_critical,
    )
    suppression_component = _component_status(
        count=suppressed,
        total=emit_events,
        rate=_as_float(out.get("suppressed_rate")),
        min_total=suppression_min_emit_events,
        min_count=suppression_min_count,
        degraded_rate=suppression_rate_degraded,
        critical_rate=suppression_rate_critical,
    )
    min_interval_component = _component_status(
        count=min_interval_escalated,
        total=emit_events,
        rate=_as_float(out.get("min_interval_escalation_rate")),
        min_total=min_interval_min_emit_events,
        min_count=min_interval_min_count,
        degraded_rate=min_interval_rate_degraded,
        critical_rate=min_interval_rate_critical,
    )
    out["fail_status"] = fail_component.get("status")
    out["fail_reason"] = fail_component.get("reason")
    out["suppression_status"] = suppression_component.get("status")
    out["suppression_reason"] = suppression_component.get("reason")
    out["min_interval_status"] = min_interval_component.get("status")
    out["min_interval_reason"] = min_interval_component.get("reason")

    total_weight = (
        component_fail_weight
        + component_suppression_weight
        + component_min_interval_weight
        + component_patch_apply_weight
    )
    if total_weight <= 0:
        component_fail_weight = 0.60
        component_suppression_weight = 0.25
        component_min_interval_weight = 0.15
        component_patch_apply_weight = 0.10
        total_weight = 1.0
        out["component_weights"] = {
            "fail": component_fail_weight,
            "suppression": component_suppression_weight,
            "min_interval": component_min_interval_weight,
            "patch_apply": component_patch_apply_weight,
        }
    component_known = any(
        str(comp.get("status") or "") in {"ok", "degraded", "critical"}
        for comp in [fail_component, suppression_component, min_interval_component, patch_apply_component]
    )
    out["component_known"] = component_known
    if component_known:
        component_score = round(
            (
                component_fail_weight * _status_weight_score(str(fail_component.get("status") or "unknown"))
                + component_suppression_weight
                * _status_weight_score(str(suppression_component.get("status") or "unknown"))
                + component_min_interval_weight
                * _status_weight_score(str(min_interval_component.get("status") or "unknown"))
                + component_patch_apply_weight
                * _status_weight_score(str(patch_apply_component.get("status") or "unknown"))
            )
            / total_weight,
            4,
        )
        out["component_score"] = component_score
        if component_score >= component_score_critical:
            out["component_score_status"] = "critical"
            out["component_score_reason"] = "score_critical"
        elif component_score >= component_score_degraded:
            out["component_score_status"] = "degraded"
            out["component_score_reason"] = "score_degraded"
        else:
            out["component_score_status"] = "ok"
            out["component_score_reason"] = "score_within_threshold"
    else:
        out["component_score"] = None
        out["component_score_status"] = "unknown"
        out["component_score_reason"] = "insufficient_samples"

    rank = {"unknown": 0, "ok": 1, "degraded": 2, "critical": 3}
    final_status = str(fail_component.get("status") or "unknown")
    final_reason = str(fail_component.get("reason") or "insufficient_samples")
    for comp in [suppression_component, min_interval_component, patch_apply_component]:
        cand_status = str(comp.get("status") or "unknown")
        if rank.get(cand_status, 0) > rank.get(final_status, 0):
            final_status = cand_status
            final_reason = str(comp.get("reason") or "composite_override")
    out["status"] = final_status
    out["status_reason"] = final_reason
    out["reason"] = final_reason
    if include_latest_event:
        out["latest_event"] = latest_event
    return out
