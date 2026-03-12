#!/usr/bin/env python3
"""Generate a non-destructive cron jobs patch draft from cron_policy remediation.

This script reads:
1) cron health report JSON (self_checks.cron_policy.remediation.operations)
2) current jobs.json

And writes:
- patch draft JSON (machine-readable change plan)
- patched jobs candidate JSON (for manual review/apply)
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lie_root_resolver import resolve_lie_system_root

SYSTEM_ROOT = resolve_lie_system_root()
DEFAULT_REPORT = SYSTEM_ROOT / "output" / "review" / f"{dt.date.today().isoformat()}_pi_cron_health.json"
DEFAULT_JOBS = Path(os.path.expanduser("~/.openclaw/cron/jobs.json"))
DEFAULT_DRAFT_OUT = SYSTEM_ROOT / "output" / "review" / f"{dt.date.today().isoformat()}_cron_policy_patch_draft.json"
DEFAULT_PATCHED_JOBS_OUT = (
    SYSTEM_ROOT / "output" / "review" / f"{dt.date.today().isoformat()}_cron_jobs_patch_candidate.json"
)
DEFAULT_CORE_NAMES = {
    "lie-spot-halfhour-core",
    "pairs-scan-2h",
    "pairs-scan-hourly",
    "hourly-market-report",
    "evomap-sync",
    "unstructured-sentinel",
    "web-access-selfcheck-6h",
}


def _read_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"json_not_object:{path}")
    return obj


def _set_path(obj: Dict[str, Any], path: str, value: Any) -> Tuple[Any, bool]:
    parts = [p for p in str(path or "").strip().split(".") if p]
    if not parts:
        raise ValueError("empty_path")
    cur: Dict[str, Any] = obj
    for seg in parts[:-1]:
        nxt = cur.get(seg)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[seg] = nxt
        cur = nxt
    leaf = parts[-1]
    before = cur.get(leaf)
    changed = before != value
    if changed:
        cur[leaf] = value
    return before, changed


def _find_job_index(jobs: List[Dict[str, Any]], job_id: str, name: str) -> Tuple[Optional[int], str]:
    if job_id:
        for i, job in enumerate(jobs):
            if str(job.get("id") or "") == job_id:
                return i, "id"
    if name:
        by_name = [i for i, job in enumerate(jobs) if str(job.get("name") or "") == name]
        if len(by_name) == 1:
            return by_name[0], "name"
        if len(by_name) > 1:
            return None, "name_ambiguous"
    return None, "not_found"


def _resolve_core_names() -> set[str]:
    raw = str(os.getenv("CRON_POLICY_CORE_NAMES", "")).strip()
    if not raw:
        return set(DEFAULT_CORE_NAMES)
    out = {x.strip() for x in raw.split(",") if x.strip()}
    return out if out else set(DEFAULT_CORE_NAMES)


def _is_high_freq_schedule(schedule: Dict[str, Any]) -> bool:
    if not isinstance(schedule, dict):
        return False
    kind = str(schedule.get("kind") or "")
    if kind == "cron":
        expr = str(schedule.get("expr") or "").strip()
        parts = expr.split()
        if len(parts) == 5:
            minute, hour = parts[0], parts[1]
            minute_high = ("*/" in minute) or ("," in minute) or ("-" in minute)
            hour_high = hour == "*" or ("*/" in hour) or ("," in hour) or ("-" in hour)
            return minute_high or hour_high
    if kind == "every":
        try:
            every_ms = int(schedule.get("everyMs") or 0)
        except Exception:
            every_ms = 0
        if 0 < every_ms <= 3600 * 1000:
            return True
    return False


def _frequency_tier(schedule: Dict[str, Any]) -> str:
    if not isinstance(schedule, dict):
        return "unknown"
    kind = str(schedule.get("kind") or "")
    if _is_high_freq_schedule(schedule):
        return "high"
    if kind in {"cron", "every"}:
        return "medium"
    return "low"


def _risk_tier(job: Dict[str, Any], core_names: set[str], operation_count: int) -> str:
    name = str(job.get("name") or "")
    enabled = bool(job.get("enabled", True))
    schedule = job.get("schedule") if isinstance(job.get("schedule"), dict) else {}
    freq = _frequency_tier(schedule)
    if name in core_names:
        return "critical"
    if enabled and freq == "high":
        return "high"
    if enabled and operation_count > 1:
        return "high"
    if enabled:
        return "medium"
    return "low"


def _rollout_batch_for(risk_tier: str, is_core_job: bool) -> str:
    if is_core_job or risk_tier == "critical":
        return "batch_3_core_guarded"
    if risk_tier == "high":
        return "batch_2_non_core_high"
    return "batch_1_non_core_canary"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    return max(int(minimum), _as_int(os.getenv(name, str(default)), default))


def _env_float(name: str, default: float, lower: float = 0.0, upper: float = 1.0) -> float:
    value = _as_float(os.getenv(name, str(default)), default)
    return max(float(lower), min(float(upper), float(value)))


def _build_proxy_isolation_plan(
    *,
    report_obj: Dict[str, Any],
    jobs_copy_list: List[Dict[str, Any]],
    core_names: set[str],
) -> Dict[str, Any]:
    enabled = _env_flag("CRON_PROXY_ISOLATION_ENABLED", default=True)
    min_runs = _env_int("CRON_PROXY_ISOLATION_MIN_RUNS", 3, minimum=1)
    degraded_rate = _env_float("CRON_PROXY_ISOLATION_DEGRADED_FAIL_RATE", 0.50, lower=0.0, upper=1.0)
    critical_rate = _env_float(
        "CRON_PROXY_ISOLATION_CRITICAL_FAIL_RATE",
        0.80,
        lower=degraded_rate,
        upper=1.0,
    )
    degraded_consecutive = _env_int("CRON_PROXY_ISOLATION_DEGRADED_CONSECUTIVE", 2, minimum=1)
    critical_consecutive = _env_int(
        "CRON_PROXY_ISOLATION_CRITICAL_CONSECUTIVE",
        4,
        minimum=degraded_consecutive,
    )
    every_ms_floor = _env_int("CRON_PROXY_ISOLATION_EVERY_MS_FLOOR", 10_800_000, minimum=60_000)
    only_non_core = _env_flag("CRON_PROXY_ISOLATION_ONLY_NON_CORE", default=True)
    cron_pause_on_degraded = _env_flag("CRON_PROXY_ISOLATION_CRON_PAUSE_ON_DEGRADED", default=False)

    checks = report_obj.get("self_checks") if isinstance(report_obj.get("self_checks"), dict) else {}
    trend = checks.get("proxy_error_trend") if isinstance(checks.get("proxy_error_trend"), dict) else {}
    trend_jobs = trend.get("jobs") if isinstance(trend.get("jobs"), list) else []

    recs: List[Dict[str, Any]] = []
    if enabled:
        for item in trend_jobs:
            if not isinstance(item, dict):
                continue
            job_id = str(item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            sample_runs = _as_int(item.get("sample_runs"), 0)
            proxy_rate = _as_float(item.get("proxy_error_rate"), 0.0)
            consecutive = _as_int(item.get("latest_consecutive_proxy_errors"), 0)
            if sample_runs < min_runs:
                continue

            severity = ""
            if consecutive >= critical_consecutive or proxy_rate >= critical_rate:
                severity = "critical"
            elif consecutive >= degraded_consecutive or proxy_rate >= degraded_rate:
                severity = "degraded"
            if not severity:
                continue

            idx, source = _find_job_index(jobs_copy_list, job_id=job_id, name=name)
            if idx is None:
                recs.append(
                    {
                        "id": job_id or None,
                        "name": name or None,
                        "severity": severity,
                        "match_source": source,
                        "sample_runs": sample_runs,
                        "proxy_error_rate": proxy_rate,
                        "latest_consecutive_proxy_errors": consecutive,
                        "core_job": bool(name in core_names),
                        "recommended_isolation": False,
                        "actionable": False,
                        "reason": "job_not_found",
                        "action_reason": "proxy_isolation_job_not_found",
                        "operations": [],
                    }
                )
                continue

            job = jobs_copy_list[idx]
            schedule = job.get("schedule") if isinstance(job.get("schedule"), dict) else {}
            schedule_kind = str(schedule.get("kind") or "")
            is_core_job = bool(str(job.get("name") or "") in core_names)
            enabled_now = bool(job.get("enabled", True))
            action_reason = "proxy_isolation_observe"
            reason = "no_isolation_needed"
            recommended = False
            ops: List[Dict[str, Any]] = []
            reasons: List[str] = []
            hints: List[str] = []

            if only_non_core and is_core_job:
                reason = "core_job_excluded"
                action_reason = "proxy_isolation_core_excluded"
                hints.append("proxy isolation: core job excluded; keep manual review path.")
            elif not enabled_now:
                reason = "already_disabled"
                action_reason = "proxy_isolation_already_disabled"
                hints.append("proxy isolation: job already disabled; monitor for recovery before re-enable.")
            elif severity == "critical":
                recommended = True
                reason = "proxy_isolation_pause_critical"
                action_reason = "proxy_isolation_pause_critical"
                reasons.append("proxy_isolation_pause_critical")
                ops.append({"op": "set", "path": "enabled", "value": False})
                hints.append("proxy isolation: critical trend, propose temporary disable for non-core job.")
            elif schedule_kind == "every":
                current_every = _as_int(schedule.get("everyMs"), 0)
                target_every = max(current_every, every_ms_floor)
                if target_every > current_every:
                    recommended = True
                    reason = "proxy_isolation_slowdown_every"
                    action_reason = "proxy_isolation_slowdown_every"
                    reasons.append("proxy_isolation_slowdown_every")
                    ops.append({"op": "set", "path": "schedule.everyMs", "value": int(target_every)})
                    hints.append(
                        f"proxy isolation: throttle everyMs from {current_every} to {target_every} for non-core job."
                    )
                else:
                    reason = "already_throttled"
                    action_reason = "proxy_isolation_already_throttled"
            elif schedule_kind == "cron" and cron_pause_on_degraded:
                recommended = True
                reason = "proxy_isolation_pause_degraded"
                action_reason = "proxy_isolation_pause_degraded"
                reasons.append("proxy_isolation_pause_degraded")
                ops.append({"op": "set", "path": "enabled", "value": False})
                hints.append("proxy isolation: degraded cron job pause is enabled by policy.")
            else:
                reason = "manual_cron_throttle_required"
                action_reason = "proxy_isolation_manual_throttle_required"
                hints.append("proxy isolation: cron schedule needs manual throttle policy review.")

            key = f"{job_id or ''}:{name or str(job.get('name') or '')}"
            plan_id = hashlib.sha1(f"iso:{key}".encode("utf-8", errors="ignore")).hexdigest()[:12]
            recs.append(
                {
                    "plan_id": plan_id,
                    "id": job_id or None,
                    "name": name or (str(job.get("name") or "") or None),
                    "severity": severity,
                    "match_source": source,
                    "sample_runs": sample_runs,
                    "proxy_error_rate": round(proxy_rate, 4),
                    "latest_consecutive_proxy_errors": consecutive,
                    "proxy_error_runs": _as_int(item.get("proxy_error_runs"), 0),
                    "core_job": is_core_job,
                    "enabled": enabled_now,
                    "schedule_kind": schedule_kind or None,
                    "reason": reason,
                    "action_reason": action_reason,
                    "recommended_isolation": recommended,
                    "actionable": len(ops) > 0,
                    "reasons": reasons,
                    "operations": ops,
                    "hints": hints[:3],
                }
            )

    if recs:
        def _rec_sort_key(item: Dict[str, Any]) -> Tuple[int, int, float, int, str]:
            actionable_rank = 0 if bool(item.get("actionable")) else 1
            severity = str(item.get("severity") or "").strip().lower()
            severity_rank = 0 if severity == "critical" else (1 if severity == "degraded" else 2)
            consecutive = _as_int(item.get("latest_consecutive_proxy_errors"), 0)
            proxy_rate = _as_float(item.get("proxy_error_rate"), 0.0)
            proxy_runs = _as_int(item.get("proxy_error_runs"), 0)
            name = str(item.get("name") or "")
            return (
                actionable_rank,
                severity_rank,
                -consecutive,
                -proxy_rate,
                -proxy_runs,
                name,
            )

        recs.sort(key=_rec_sort_key)

    actionable = [x for x in recs if isinstance(x, dict) and bool(x.get("actionable"))]
    critical_candidates = [
        x for x in recs if isinstance(x, dict) and str(x.get("severity") or "") == "critical"
    ]
    degraded_candidates = [
        x for x in recs if isinstance(x, dict) and str(x.get("severity") or "") == "degraded"
    ]
    manual_only = [x for x in recs if isinstance(x, dict) and not bool(x.get("actionable"))]

    return {
        "enabled": enabled,
        "summary": {
            "candidates_total": len(recs),
            "critical_candidates": len(critical_candidates),
            "degraded_candidates": len(degraded_candidates),
            "actionable_candidates": len(actionable),
            "manual_only_candidates": len(manual_only),
        },
        "thresholds": {
            "min_runs": min_runs,
            "degraded_fail_rate": degraded_rate,
            "critical_fail_rate": critical_rate,
            "degraded_consecutive": degraded_consecutive,
            "critical_consecutive": critical_consecutive,
            "every_ms_floor": every_ms_floor,
            "only_non_core": only_non_core,
            "cron_pause_on_degraded": cron_pause_on_degraded,
        },
        "recommendations": recs[:50],
    }


def _build_rollout_plan(actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_labels = {
        "batch_0_proxy_isolation": "Proxy isolation first (non-core containment)",
        "batch_1_non_core_canary": "Canary non-core (low/medium risk first)",
        "batch_2_non_core_high": "Non-core high risk",
        "batch_3_core_guarded": "Core jobs guarded rollout",
    }
    has_proxy_isolation_batch = any(
        isinstance(x, dict)
        and str(x.get("rollout_batch") or "").strip() == "batch_0_proxy_isolation"
        and int(x.get("operation_count") or 0) > 0
        for x in actions
    )
    ordered = ["batch_1_non_core_canary", "batch_2_non_core_high", "batch_3_core_guarded"]
    if has_proxy_isolation_batch:
        ordered = ["batch_0_proxy_isolation"] + ordered
    by_batch: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ordered}
    for action in actions:
        b = str(action.get("rollout_batch") or "")
        if b not in by_batch:
            continue
        if int(action.get("operation_count") or 0) <= 0:
            continue
        by_batch[b].append(action)

    batches: List[Dict[str, Any]] = []
    apply_order: List[str] = []
    for key in ordered:
        group = by_batch[key]
        if key == "batch_0_proxy_isolation":
            def _batch0_sort_key(item: Dict[str, Any]) -> Tuple[int, int, float, int, int, str]:
                severity = str(item.get("isolation_severity") or "").strip().lower()
                severity_rank = 0 if severity == "critical" else (1 if severity == "degraded" else 2)
                consecutive = _as_int(item.get("isolation_latest_consecutive_proxy_errors"), 0)
                proxy_rate = _as_float(item.get("isolation_proxy_error_rate"), 0.0)
                proxy_runs = _as_int(item.get("isolation_proxy_error_runs"), 0)
                changed_rank = 0 if bool(item.get("changed")) else 1
                return (
                    severity_rank,
                    -consecutive,
                    -proxy_rate,
                    -proxy_runs,
                    changed_rank,
                    str(item.get("name") or ""),
                )

            group.sort(key=_batch0_sort_key)
        else:
            group.sort(
                key=lambda x: (
                    0 if bool(x.get("changed")) else 1,
                    int(x.get("operation_count") or 0),
                    str(x.get("name") or ""),
                )
            )
        for action in group:
            pid = str(action.get("plan_id") or "")
            if pid:
                apply_order.append(pid)
        batches.append(
            {
                "id": key,
                "label": batch_labels[key],
                "count": len(group),
                "plan_ids": [str(x.get("plan_id") or "") for x in group if str(x.get("plan_id") or "")],
                "ops": sum(int(x.get("operation_count") or 0) for x in group),
            }
        )

    return {
        "strategy": (
            "proxy_isolation_then_canary_core_guarded"
            if has_proxy_isolation_batch
            else "canary_then_core_guarded"
        ),
        "batches": batches,
        "apply_order": apply_order,
        "batch_count": len(batches),
        "non_empty_batches": sum(1 for x in batches if int(x.get("count") or 0) > 0),
    }


def build_patch_draft(
    report_obj: Dict[str, Any],
    jobs_obj: Dict[str, Any],
    report_path: Path,
    jobs_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checks = report_obj.get("self_checks") if isinstance(report_obj.get("self_checks"), dict) else {}
    policy = checks.get("cron_policy") if isinstance(checks.get("cron_policy"), dict) else {}
    remediation = policy.get("remediation") if isinstance(policy.get("remediation"), list) else []
    jobs = jobs_obj.get("jobs") if isinstance(jobs_obj.get("jobs"), list) else []
    jobs_copy = copy.deepcopy(jobs_obj)
    jobs_copy_list = jobs_copy.get("jobs") if isinstance(jobs_copy.get("jobs"), list) else []
    core_names = _resolve_core_names()
    isolation_plan = _build_proxy_isolation_plan(
        report_obj=report_obj,
        jobs_copy_list=jobs_copy_list,
        core_names=core_names,
    )

    actions: List[Dict[str, Any]] = []
    action_index: Dict[str, Dict[str, Any]] = {}

    def _job_key(job_id: str, name: str) -> str:
        return f"{job_id or ''}:{name or ''}"

    def _ensure_action(
        *,
        job_id: str,
        name: str,
        match_source: str,
        plan_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = _job_key(job_id, name)
        action = action_index.get(key)
        if action is None:
            action = {
                "plan_id": plan_id or None,
                "id": job_id or None,
                "name": name or None,
                "match_source": match_source,
                "reasons": [],
                "operation_count": 0,
                "operation_applied_count": 0,
                "changed": False,
                "applied": False,
                "errors": [],
                "changes": [],
                "isolation_recommendation": False,
                "isolation_severity": None,
                "isolation_hints": [],
            }
            actions.append(action)
            action_index[key] = action
        else:
            if not action.get("plan_id") and plan_id:
                action["plan_id"] = plan_id
            if not action.get("match_source"):
                action["match_source"] = match_source
        return action

    def _append_reason(action: Dict[str, Any], reason: str) -> None:
        text = str(reason or "").strip()
        if not text:
            return
        reasons = action.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
            action["reasons"] = reasons
        if text not in reasons:
            reasons.append(text)

    def _append_error(action: Dict[str, Any], err: str) -> None:
        text = str(err or "").strip()
        if not text:
            return
        errors = action.get("errors")
        if not isinstance(errors, list):
            errors = []
            action["errors"] = errors
        if text not in errors:
            errors.append(text)

    def _apply_operations(action: Dict[str, Any], job: Dict[str, Any], operations: List[Dict[str, Any]]) -> None:
        for op in operations:
            if not isinstance(op, dict):
                _append_error(action, "invalid_operation")
                continue
            action["operation_count"] = int(action.get("operation_count") or 0) + 1
            if str(op.get("op") or "") != "set":
                _append_error(action, "unsupported_op")
                continue
            path = str(op.get("path") or "")
            if not path:
                _append_error(action, "missing_path")
                continue
            value = op.get("value")
            try:
                before, changed = _set_path(job, path=path, value=value)
            except Exception as e:
                _append_error(action, f"set_failed:{type(e).__name__}")
                continue
            changes = action.get("changes")
            if not isinstance(changes, list):
                changes = []
                action["changes"] = changes
            changes.append(
                {
                    "op": "set",
                    "path": path,
                    "before": before,
                    "after": value,
                    "changed": bool(changed),
                }
            )
            action["operation_applied_count"] = int(action.get("operation_applied_count") or 0) + 1
            if changed:
                action["changed"] = True

    for item in remediation:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("id") or "")
        name = str(item.get("name") or "")
        plan_id = str(item.get("plan_id") or "")
        reasons = item.get("reasons") if isinstance(item.get("reasons"), list) else []
        operations = item.get("operations") if isinstance(item.get("operations"), list) else []
        idx, source = _find_job_index(jobs_copy_list, job_id=job_id, name=name)
        action = _ensure_action(job_id=job_id, name=name, match_source=source, plan_id=plan_id or None)
        for reason in reasons:
            _append_reason(action, str(reason))
        if idx is None:
            _append_error(action, "job_not_found" if source != "name_ambiguous" else "job_name_ambiguous")
            continue

        job = jobs_copy_list[idx]
        _apply_operations(action, job, operations)

    isolation_recs = (
        isolation_plan.get("recommendations") if isinstance(isolation_plan.get("recommendations"), list) else []
    )
    for rec in isolation_recs:
        if not isinstance(rec, dict):
            continue
        operations = rec.get("operations") if isinstance(rec.get("operations"), list) else []
        if not operations:
            continue
        job_id = str(rec.get("id") or "")
        name = str(rec.get("name") or "")
        plan_id = str(rec.get("plan_id") or "")
        idx, source = _find_job_index(jobs_copy_list, job_id=job_id, name=name)
        action = _ensure_action(job_id=job_id, name=name, match_source=source, plan_id=plan_id or None)
        action["isolation_recommendation"] = True
        action["isolation_severity"] = str(rec.get("severity") or "") or action.get("isolation_severity")
        action["isolation_action_reason"] = str(rec.get("action_reason") or "") or action.get(
            "isolation_action_reason"
        )
        action["isolation_proxy_error_rate"] = _as_float(
            rec.get("proxy_error_rate"),
            _as_float(action.get("isolation_proxy_error_rate"), 0.0),
        )
        action["isolation_latest_consecutive_proxy_errors"] = _as_int(
            rec.get("latest_consecutive_proxy_errors"),
            _as_int(action.get("isolation_latest_consecutive_proxy_errors"), 0),
        )
        action["isolation_proxy_error_runs"] = _as_int(
            rec.get("proxy_error_runs"),
            _as_int(action.get("isolation_proxy_error_runs"), 0),
        )
        hints = action.get("isolation_hints")
        if not isinstance(hints, list):
            hints = []
            action["isolation_hints"] = hints
        for hint in rec.get("hints") if isinstance(rec.get("hints"), list) else []:
            text = str(hint or "").strip()
            if text and text not in hints:
                hints.append(text)
        for reason in rec.get("reasons") if isinstance(rec.get("reasons"), list) else []:
            _append_reason(action, str(reason))
        _append_reason(action, str(rec.get("reason") or "proxy_isolation"))

        if idx is None:
            _append_error(action, "job_not_found" if source != "name_ambiguous" else "job_name_ambiguous")
            continue
        job = jobs_copy_list[idx]
        _apply_operations(action, job, operations)

    for action in actions:
        job_id = str(action.get("id") or "")
        name = str(action.get("name") or "")
        idx, _ = _find_job_index(jobs_copy_list, job_id=job_id, name=name)
        if idx is not None:
            job = jobs_copy_list[idx]
            schedule = job.get("schedule") if isinstance(job.get("schedule"), dict) else {}
            action["enabled"] = bool(job.get("enabled", True))
            action["core_job"] = bool(str(job.get("name") or "") in core_names)
            action["frequency_tier"] = _frequency_tier(schedule)
            action["schedule_kind"] = str(schedule.get("kind") or "") if isinstance(schedule, dict) else None
            action["risk_tier"] = _risk_tier(
                job,
                core_names=core_names,
                operation_count=int(action.get("operation_count") or 0),
            )
            rollout_batch_risk = _rollout_batch_for(
                risk_tier=str(action.get("risk_tier") or "medium"),
                is_core_job=bool(action.get("core_job")),
            )
            isolation_priority = (
                bool(action.get("isolation_recommendation"))
                and int(action.get("operation_count") or 0) > 0
                and not bool(action.get("core_job"))
            )
            action["rollout_batch_risk"] = rollout_batch_risk
            action["rollout_batch_reason"] = "proxy_isolation_priority" if isolation_priority else "risk_tier"
            action["rollout_batch"] = (
                "batch_0_proxy_isolation" if isolation_priority else rollout_batch_risk
            )
        reasons = action.get("reasons")
        if isinstance(reasons, list):
            dedup = sorted({str(x).strip() for x in reasons if str(x).strip()})
            action["reasons"] = dedup
        isolation_hints = action.get("isolation_hints")
        if isinstance(isolation_hints, list):
            dedup_hints = sorted({str(h).strip() for h in isolation_hints if str(h).strip()})
            action["isolation_hints"] = dedup_hints
        action["applied"] = int(action.get("operation_applied_count") or 0) > 0 and len(
            action.get("errors") if isinstance(action.get("errors"), list) else []
        ) == 0

    requested = len(actions)
    actionable = sum(1 for x in actions if int(x.get("operation_count") or 0) > 0)
    applied = sum(1 for x in actions if bool(x.get("applied")))
    failed = sum(1 for x in actions if bool(x.get("errors")))
    changed_jobs = sum(1 for x in actions if bool(x.get("changed")))
    operation_total = sum(int(x.get("operation_count") or 0) for x in actions)
    operation_applied_total = sum(int(x.get("operation_applied_count") or 0) for x in actions)

    status = "noop"
    if actionable > 0:
        status = "ok" if failed == 0 else "degraded"
    rollout_plan = _build_rollout_plan(actions)
    isolation_summary = (
        isolation_plan.get("summary") if isinstance(isolation_plan.get("summary"), dict) else {}
    )
    isolation_actionable = int(isolation_summary.get("actionable_candidates") or 0)
    isolation_plan_ids: List[str] = []
    if isinstance(isolation_plan.get("recommendations"), list):
        for rec in isolation_plan.get("recommendations") or []:
            if not isinstance(rec, dict):
                continue
            if not bool(rec.get("actionable")):
                continue
            pid = str(rec.get("plan_id") or "").strip()
            if pid:
                isolation_plan_ids.append(pid)
    isolation_rollout = {
        "strategy": "proxy_isolation_then_canary_core_guarded",
        "enabled": bool(isolation_plan.get("enabled", False)),
        "batch_id": "batch_0_proxy_isolation",
        "next_batch": "batch_0_proxy_isolation" if isolation_actionable > 0 else None,
        "pending": bool(isolation_actionable > 0),
        "actionable_candidates": isolation_actionable,
        "critical_candidates": int(isolation_summary.get("critical_candidates") or 0),
        "manual_only_candidates": int(isolation_summary.get("manual_only_candidates") or 0),
        "plan_ids": isolation_plan_ids[:50],
    }

    draft = {
        "envelope_version": "1.0",
        "domain": "cron_policy_patch_draft",
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "report_path": str(report_path),
        "jobs_path": str(jobs_path),
        "summary": {
            "requested_jobs": requested,
            "actionable_jobs": actionable,
            "applied_jobs": applied,
            "failed_jobs": failed,
            "changed_jobs": changed_jobs,
            "operation_total": operation_total,
            "operation_applied_total": operation_applied_total,
            "core_names_count": len(core_names),
        },
        "isolation_plan": isolation_plan,
        "isolation_rollout": isolation_rollout,
        "actions": actions,
        "rollout_plan": rollout_plan,
        "apply_guide": [
            "Review patch draft actions and changed paths.",
            "Diff patched jobs candidate against current jobs.json.",
            "Apply in small batches and validate via openclaw cron list --json.",
            "For proxy isolation recommendations, prioritize non-core isolation/de-throttle actions before core rollout.",
        ],
    }
    return draft, jobs_copy


def run_patch_draft(
    report_path: Path,
    jobs_path: Path,
    output_path: Path,
    patched_jobs_output: Optional[Path] = None,
) -> Dict[str, Any]:
    report_obj = _read_json(report_path)
    jobs_obj = _read_json(jobs_path)
    draft, patched_jobs = build_patch_draft(
        report_obj=report_obj,
        jobs_obj=jobs_obj,
        report_path=report_path,
        jobs_path=jobs_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if patched_jobs_output is not None:
        patched_jobs_output.parent.mkdir(parents=True, exist_ok=True)
        patched_jobs_output.write_text(
            json.dumps(patched_jobs, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        draft["patched_jobs_output"] = str(patched_jobs_output)
    draft["output_path"] = str(output_path)
    return draft


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-path", default=str(DEFAULT_REPORT))
    ap.add_argument("--jobs-path", default=str(DEFAULT_JOBS))
    ap.add_argument("--output", default=str(DEFAULT_DRAFT_OUT))
    ap.add_argument("--patched-jobs-output", default=str(DEFAULT_PATCHED_JOBS_OUT))
    ap.add_argument("--no-patched-jobs-output", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    report_path = Path(args.report_path).expanduser().resolve()
    jobs_path = Path(args.jobs_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    patched_jobs_output = None if bool(args.no_patched_jobs_output) else Path(args.patched_jobs_output).expanduser().resolve()

    out = run_patch_draft(
        report_path=report_path,
        jobs_path=jobs_path,
        output_path=output_path,
        patched_jobs_output=patched_jobs_output,
    )
    print(json.dumps(out, ensure_ascii=False))
    if bool(args.strict) and str(out.get("status") or "") == "degraded":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
