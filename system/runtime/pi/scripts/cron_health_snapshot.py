#!/usr/bin/env python3
"""Local cron health snapshot (no gateway calls).

Reads ~/.openclaw/cron/jobs.json and writes a compact JSON report to system output.
This is intentionally offline and deterministic for low-friction daily diagnostics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from gate_notify_trend import compute_notify_trend
from lie_root_resolver import resolve_lie_system_root
from memory_fallback_search import search_memory
from signal_registry import probe_signal_registry


SYSTEM_ROOT = resolve_lie_system_root()
DEFAULT_JOBS = Path(os.path.expanduser("~/.openclaw/cron/jobs.json"))
DEFAULT_RUNS_DIR = Path(os.path.expanduser("~/.openclaw/cron/runs"))
DEFAULT_OPENCLAW_CONFIG = Path(os.path.expanduser("~/.openclaw/openclaw.json"))
DEFAULT_PI_CYCLE_LOG = SYSTEM_ROOT / "output" / "logs" / "pi_cycle_events.jsonl"
CORE_JOB_NAMES = {
    "lie-spot-halfhour-core",
    "pairs-scan-2h",
    "pairs-scan-hourly",
    "hourly-market-report",
    "evomap-sync",
    "unstructured-sentinel",
    "web-access-selfcheck-6h",
}


def _is_proxy_error_text(err: str) -> bool:
    e = (err or "").lower()
    if not e:
        return False
    proxy_tokens = [
        "proxy error",
        "proxy_error",
        "http 500",
        "500 internal",
        "bad gateway",
        "gateway timeout",
        "upstream connect error",
        "reverse proxy",
    ]
    return any(token in e for token in proxy_tokens)


def _classify_error(err: str) -> str:
    e = (err or "").lower()
    if not e:
        return "none"
    if _is_proxy_error_text(e):
        return "proxy_5xx"
    if "missing authentication header" in e or "no api key" in e or "auth" in e:
        return "auth"
    if "timeout" in e or "timed out" in e:
        return "timeout"
    if "ssl" in e or "connection" in e or "network" in e:
        return "network"
    if "unsupported channel" in e or "delivery" in e:
        return "delivery"
    return "other"


def _parse_json_payload(raw: str) -> Optional[Any]:
    lines = (raw or "").splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("[WRAPPER]"):
            continue
        if not (s.startswith("{") or s.startswith("[")):
            continue
        candidate = "\n".join(lines[i:]).strip()
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _run_json_cmd(cmd: List[str], timeout_sec: int = 8) -> Dict[str, Any]:
    started = time.time()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
    except Exception as e:
        return {
            "ok": False,
            "reason": f"exec_error:{type(e).__name__}",
            "duration_sec": round(time.time() - started, 3),
            "payload": None,
        }

    payload = _parse_json_payload((p.stdout or "").strip())
    if p.returncode != 0:
        return {
            "ok": False,
            "reason": f"rc_{p.returncode}",
            "duration_sec": round(time.time() - started, 3),
            "payload": payload,
            "stderr": (p.stderr or "")[-300:],
        }
    if payload is None:
        return {
            "ok": False,
            "reason": "no_json_payload",
            "duration_sec": round(time.time() - started, 3),
            "payload": None,
        }
    return {
        "ok": True,
        "reason": "ok",
        "duration_sec": round(time.time() - started, 3),
        "payload": payload,
    }


def _parse_ts_utc(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _normalize_action(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "null", "na", "n/a", "unknown"}:
        return ""
    return text


def _normalize_ops_priority(value: Any) -> str:
    level = str(value or "").strip().lower()
    if level in {"p0", "p1", "p2", "p3"}:
        return level
    return ""


def _derive_ops_next_action(event: Dict[str, Any]) -> Dict[str, Any]:
    priority = _normalize_ops_priority(event.get("ops_next_action_priority"))
    action = _normalize_action(event.get("ops_next_action"))
    reason = str(event.get("ops_next_action_reason") or "").strip()
    secondary = _normalize_action(event.get("ops_next_action_secondary"))
    context = event.get("ops_next_action_context") if isinstance(event.get("ops_next_action_context"), dict) else {}

    if priority and reason:
        return {
            "priority": priority,
            "action": action or "none",
            "reason": reason,
            "secondary": secondary or None,
            "context": context,
            "source": "event",
        }

    core_status = str(event.get("core_proxy_bypass_status") or "").strip().lower()
    core_execution_status = str(event.get("core_execution_status") or "").strip().lower()
    core_execution_reason = str(event.get("core_execution_reason") or "").strip().lower()
    core_execution_decision = str(event.get("core_execution_decision") or "").strip().lower()
    core_execution_order_http = _as_int(event.get("core_execution_order_http"), 0)
    core_execution_order_error = str(event.get("core_execution_order_error") or "").strip()
    core_execution_order_simulated = bool(event.get("core_execution_order_simulated"))
    proxy_trend_status = str(event.get("guard_mode_proxy_error_trend_status") or "").strip().lower()
    patch_apply_status = str(event.get("guard_mode_patch_apply_status") or "").strip().lower()
    guard_effective = str(event.get("guard_mode_effective") or "").strip().lower()
    guard_recommended = str(event.get("guard_mode_recommended") or "").strip().lower()
    recommend_mismatch = bool(guard_effective and guard_recommended and guard_effective != guard_recommended)
    patch_next = _normalize_action(event.get("guard_mode_patch_draft_next_action"))
    patch_next_secondary = _normalize_action(event.get("guard_mode_patch_draft_next_action_secondary"))
    isolation_batch = (
        str(event.get("guard_mode_patch_apply_target_batch") or "").strip()
        or str(event.get("guard_mode_patch_draft_rollout_next_batch") or "").strip()
    )
    bypass_failed = _as_int(event.get("core_proxy_bypass_bypass_failed"), 0)
    no_proxy_retry_exhausted = _as_int(event.get("core_proxy_bypass_no_proxy_retry_exhausted"), 0)
    hint_exception = _as_int(event.get("core_proxy_bypass_hint_exception"), 0)
    hint_response = _as_int(event.get("core_proxy_bypass_hint_response"), 0)
    patch_failed_jobs = _as_int(event.get("guard_mode_patch_apply_failed_jobs"), 0)
    retry_exhausted_degraded = max(
        1, _as_int(os.getenv("PI_OPS_NEXT_ACTION_PROXY_RETRY_EXHAUSTED_DEGRADED", "1"), 1)
    )
    retry_exhausted_critical = max(
        1, _as_int(os.getenv("PI_OPS_NEXT_ACTION_PROXY_RETRY_EXHAUSTED_CRITICAL", "2"), 2)
    )

    next_action = "none"
    next_reason = "stable"
    next_priority = "p3"
    next_secondary = patch_next_secondary or None

    if core_status == "critical" or proxy_trend_status == "critical":
        next_action = f"run_batch:{isolation_batch}" if isolation_batch else "stabilize_proxy_path_and_isolate_jobs"
        next_reason = "core_proxy_bypass_or_proxy_error_critical"
        next_priority = "p0"
        if patch_next:
            next_secondary = patch_next
    elif core_status == "degraded":
        if (
            no_proxy_retry_exhausted >= retry_exhausted_critical
            and proxy_trend_status in {"degraded", "critical"}
        ):
            next_action = (
                f"run_batch:{isolation_batch}" if isolation_batch else "stabilize_proxy_path_and_isolate_jobs"
            )
            next_reason = "core_proxy_retry_exhausted_with_proxy_trend"
            next_priority = "p0"
            if patch_next:
                next_secondary = patch_next
        elif no_proxy_retry_exhausted >= retry_exhausted_degraded:
            next_action = "stabilize_proxy_path_and_raise_timeout_floor"
            next_reason = "core_proxy_retry_exhausted"
            next_priority = "p1"
            if patch_next:
                next_secondary = patch_next
        elif bypass_failed > 0 or hint_exception > 0:
            next_action = "stabilize_proxy_path"
            next_reason = "core_proxy_bypass_degraded_with_failures"
            next_priority = "p1"
            if patch_next:
                next_secondary = patch_next
        else:
            next_action = patch_next or "observe_core_proxy_bypass"
            next_reason = "core_proxy_bypass_degraded"
            next_priority = "p2"
    elif core_execution_status == "critical":
        next_action = "stabilize_execution_path_and_isolate_order_router"
        next_reason = "core_execution_critical"
        next_priority = "p0"
        if patch_next:
            next_secondary = patch_next
    elif core_execution_status == "degraded":
        if core_execution_reason in {
            "exec_error",
            "order_result_error",
            "order_http_non_2xx",
            "missing_order_result",
            "missing_order_http",
        }:
            next_action = "stabilize_execution_path"
            next_reason = "core_execution_degraded_with_errors"
            next_priority = "p1"
            if patch_next:
                next_secondary = patch_next
        else:
            next_action = patch_next or "observe_core_execution"
            next_reason = "core_execution_degraded"
            next_priority = "p2"
    elif patch_apply_status == "critical" or patch_failed_jobs > 0:
        next_action = "inspect_patch_status_critical"
        next_reason = "patch_apply_failure_detected"
        next_priority = "p1"
    elif recommend_mismatch and guard_recommended:
        next_action = f"align_guard_mode:{guard_recommended}"
        next_reason = "guard_mode_recommend_mismatch"
        next_priority = "p2"
    elif patch_next:
        next_action = patch_next
        next_reason = "patch_rollout_pending"
        next_priority = "p2"

    if next_secondary == next_action:
        next_secondary = None

    return {
        "priority": next_priority,
        "action": next_action,
        "reason": next_reason,
        "secondary": next_secondary,
        "context": {
            "core_status": core_status,
            "core_execution_status": core_execution_status,
            "core_execution_reason": core_execution_reason,
            "core_execution_decision": core_execution_decision,
            "core_execution_order_http": core_execution_order_http,
            "core_execution_order_error": core_execution_order_error,
            "core_execution_order_simulated": core_execution_order_simulated,
            "proxy_trend_status": proxy_trend_status,
            "patch_apply_status": patch_apply_status,
            "guard_mode_effective": guard_effective,
            "guard_mode_recommended": guard_recommended,
            "recommend_mismatch": recommend_mismatch,
            "bypass_failed": bypass_failed,
            "no_proxy_retry_exhausted": no_proxy_retry_exhausted,
            "retry_exhausted_degraded_threshold": retry_exhausted_degraded,
            "retry_exhausted_critical_threshold": retry_exhausted_critical,
            "hint_exception": hint_exception,
            "hint_response": hint_response,
            "patch_failed_jobs": patch_failed_jobs,
        },
        "source": "derived",
    }


def _as_ts_utc_from_ms_or_iso(value: Any) -> Optional[dt.datetime]:
    if isinstance(value, (int, float)):
        try:
            iv = int(value)
        except Exception:
            iv = 0
        if iv <= 0:
            return None
        # Cron runs typically use unix epoch in milliseconds.
        if iv > 10_000_000_000:
            iv = int(iv / 1000)
        try:
            return dt.datetime.fromtimestamp(iv, tz=dt.timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        return _parse_ts_utc(value)
    return None


def _probe_component_mode_drift(
    *,
    path: Path,
    window_hours: float,
    max_lines: int,
) -> Dict[str, Any]:
    min_events = max(1, int(float(os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_MIN_EVENTS", "3"))))
    min_consecutive = max(
        1,
        int(float(os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_MIN_CONSECUTIVE", "2"))),
    )
    rate_degraded = max(0.0, min(1.0, float(os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_RATE_DEGRADED", "0.50"))))
    rate_critical = max(
        rate_degraded,
        min(1.0, float(os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_RATE_CRITICAL", "0.80"))),
    )
    out: Dict[str, Any] = {
        "status": "unknown",
        "reason": "insufficient_mode_events",
        "window_hours": float(window_hours),
        "mode_events": 0,
        "drift_events": 0,
        "drift_rate": None,
        "latest_consecutive_drift": 0,
        "latest_drift": None,
        "top_drift_pairs": [],
        "thresholds": {
            "min_events": min_events,
            "min_consecutive": min_consecutive,
            "rate_degraded": rate_degraded,
            "rate_critical": rate_critical,
        },
    }
    if not path.exists():
        out["reason"] = "pi_cycle_log_missing"
        return out

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        out["status"] = "degraded"
        out["reason"] = "pi_cycle_log_unreadable"
        return out

    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=max(0.1, float(window_hours)))
    allowed_modes = {"off", "shadow", "enforce"}
    drift_pair_counts: Dict[str, int] = {}
    consecutive_open = True

    for raw in reversed(lines[-max(1, int(max_lines)) :]):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or str(obj.get("domain") or "") != "pi_cycle":
            continue
        ts = _parse_ts_utc(obj.get("ts"))
        if ts is None or ts < since:
            continue
        emit = obj.get("gate_rollout_alert_emit") if isinstance(obj.get("gate_rollout_alert_emit"), dict) else {}
        configured = str(emit.get("gate_notify_trend_component_score_mode_configured") or "").strip().lower()
        effective = str(emit.get("gate_notify_trend_component_score_mode_effective") or "").strip().lower()
        if configured not in allowed_modes or effective not in allowed_modes:
            continue
        out["mode_events"] = int(out["mode_events"]) + 1
        drift = configured != effective
        if out["latest_drift"] is None:
            out["latest_drift"] = bool(drift)
        if drift:
            out["drift_events"] = int(out["drift_events"]) + 1
            pair = f"{configured}->{effective}"
            drift_pair_counts[pair] = drift_pair_counts.get(pair, 0) + 1
            if consecutive_open:
                out["latest_consecutive_drift"] = int(out["latest_consecutive_drift"]) + 1
        else:
            consecutive_open = False

    mode_events = int(out["mode_events"])
    drift_events = int(out["drift_events"])
    if mode_events > 0:
        out["drift_rate"] = round(float(drift_events) / float(mode_events), 4)
    out["top_drift_pairs"] = [
        {"pair": k, "count": v}
        for k, v in sorted(drift_pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]

    if mode_events < min_events:
        out["status"] = "unknown"
        out["reason"] = "insufficient_mode_events"
        return out
    if drift_events <= 0:
        out["status"] = "ok"
        out["reason"] = "no_mode_drift"
        return out

    drift_rate = float(out.get("drift_rate") or 0.0)
    consecutive = int(out.get("latest_consecutive_drift") or 0)
    if drift_rate >= rate_critical:
        out["status"] = "critical"
        out["reason"] = "mode_drift_rate_high"
    elif consecutive >= min_consecutive:
        out["status"] = "degraded"
        out["reason"] = "mode_drift_persistent"
    elif drift_rate >= rate_degraded:
        out["status"] = "degraded"
        out["reason"] = "mode_drift_rate_elevated"
    else:
        out["status"] = "ok"
        out["reason"] = "mode_drift_within_tolerance"
    return out


def _probe_rollout_rollback_trend(
    *,
    path: Path,
    window_hours: float,
    max_lines: int,
) -> Dict[str, Any]:
    min_events = max(1, int(float(os.getenv("PI_GATE_ROLLBACK_TREND_MIN_EVENTS", "3"))))
    rate_degraded = max(0.0, min(1.0, float(os.getenv("PI_GATE_ROLLBACK_TREND_RATE_DEGRADED", "0.20"))))
    rate_critical = max(
        rate_degraded,
        min(1.0, float(os.getenv("PI_GATE_ROLLBACK_TREND_RATE_CRITICAL", "0.50"))),
    )
    drift_count_critical = max(
        1,
        int(float(os.getenv("PI_GATE_ROLLBACK_COMPONENT_MODE_DRIFT_COUNT_CRITICAL", "2"))),
    )
    out: Dict[str, Any] = {
        "status": "unknown",
        "reason": "insufficient_events",
        "window_hours": float(window_hours),
        "pi_cycle_events": 0,
        "rollback_records": 0,
        "enabled_count": 0,
        "applied_count": 0,
        "trigger_count": 0,
        "trigger_rate": None,
        "trigger_counts": {
            "breach": 0,
            "component_mode_drift": 0,
            "breach_and_component_mode_drift": 0,
        },
        "trigger_shares": {
            "breach": None,
            "component_mode_drift": None,
            "breach_and_component_mode_drift": None,
        },
        "dominant_trigger": None,
        "dominant_trigger_count": 0,
        "dominant_trigger_share": None,
        "action_level": "observe",
        "action_reason": "insufficient_events",
        "action_recommendations": [],
        "latest_trigger": None,
        "latest_reason": None,
        "latest_applied": None,
        "latest_ts": None,
        "top_triggers": [],
        "action_hints": [],
        "thresholds": {
            "min_events": min_events,
            "rate_degraded": rate_degraded,
            "rate_critical": rate_critical,
            "component_mode_drift_count_critical": drift_count_critical,
        },
    }
    if not path.exists():
        out["reason"] = "pi_cycle_log_missing"
        out["action_level"] = "observe"
        out["action_reason"] = "pi_cycle_log_missing"
        return out

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        out["status"] = "degraded"
        out["reason"] = "pi_cycle_log_unreadable"
        out["action_level"] = "degrade"
        out["action_reason"] = "pi_cycle_log_unreadable"
        out["action_recommendations"] = ["health: fix log pipeline before trusting rollback trend metrics."]
        return out

    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=max(0.1, float(window_hours)))
    trigger_counter: Dict[str, int] = {}
    known_triggers = {"breach", "component_mode_drift", "breach_and_component_mode_drift"}

    for raw in reversed(lines[-max(1, int(max_lines)) :]):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or str(obj.get("domain") or "") != "pi_cycle":
            continue
        ts = _parse_ts_utc(obj.get("ts"))
        if ts is None or ts < since:
            continue
        out["pi_cycle_events"] = int(out["pi_cycle_events"]) + 1
        rollback = obj.get("gate_rollout_rollback") if isinstance(obj.get("gate_rollout_rollback"), dict) else {}
        if not rollback:
            continue
        out["rollback_records"] = int(out["rollback_records"]) + 1
        if bool(rollback.get("enabled")):
            out["enabled_count"] = int(out["enabled_count"]) + 1
        applied = bool(rollback.get("applied"))
        if applied:
            out["applied_count"] = int(out["applied_count"]) + 1
        trigger = str(rollback.get("trigger") or "").strip().lower()
        if trigger in known_triggers:
            out["trigger_count"] = int(out["trigger_count"]) + 1
            trigger_counter[trigger] = trigger_counter.get(trigger, 0) + 1
            out["trigger_counts"][trigger] = int((out["trigger_counts"] or {}).get(trigger) or 0) + 1
        if out["latest_ts"] is None:
            out["latest_ts"] = obj.get("ts")
            out["latest_trigger"] = trigger or None
            out["latest_reason"] = rollback.get("reason")
            out["latest_applied"] = applied

    events = int(out["pi_cycle_events"])
    trigger_count = int(out["trigger_count"])
    if events > 0:
        out["trigger_rate"] = round(float(trigger_count) / float(events), 4)
    out["top_triggers"] = [
        {"trigger": k, "count": v}
        for k, v in sorted(trigger_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    if trigger_count > 0:
        trigger_counts = out.get("trigger_counts") if isinstance(out.get("trigger_counts"), dict) else {}
        trigger_shares = out.get("trigger_shares") if isinstance(out.get("trigger_shares"), dict) else {}
        priorities = {
            "component_mode_drift": 0,
            "breach_and_component_mode_drift": 1,
            "breach": 2,
        }
        dominant: Optional[str] = None
        dominant_count = 0
        for key in ["breach", "component_mode_drift", "breach_and_component_mode_drift"]:
            cnt = int(trigger_counts.get(key) or 0)
            trigger_shares[key] = round(float(cnt) / float(trigger_count), 4)
            if cnt <= 0:
                continue
            if dominant is None:
                dominant = key
                dominant_count = cnt
                continue
            current_priority = priorities.get(str(dominant), 9)
            candidate_priority = priorities.get(key, 9)
            if cnt > dominant_count or (cnt == dominant_count and candidate_priority < current_priority):
                dominant = key
                dominant_count = cnt
        out["dominant_trigger"] = dominant
        out["dominant_trigger_count"] = int(dominant_count)
        if dominant and dominant_count > 0:
            out["dominant_trigger_share"] = round(float(dominant_count) / float(trigger_count), 4)

    if events < min_events:
        out["status"] = "unknown"
        out["reason"] = "insufficient_events"
        out["action_level"] = "observe"
        out["action_reason"] = "insufficient_events"
        return out
    if trigger_count <= 0:
        out["status"] = "ok"
        out["reason"] = "no_rollback_triggered"
        out["action_level"] = "observe"
        out["action_reason"] = "no_rollback_triggered"
        return out

    trigger_rate = float(out.get("trigger_rate") or 0.0)
    drift_trigger_count = int(((out.get("trigger_counts") or {}).get("component_mode_drift") or 0))
    if trigger_rate >= rate_critical:
        out["status"] = "critical"
        out["reason"] = "rollback_trigger_rate_high"
    elif drift_trigger_count >= drift_count_critical:
        out["status"] = "critical"
        out["reason"] = "component_mode_drift_rollback_repeated"
    elif trigger_rate >= rate_degraded:
        out["status"] = "degraded"
        out["reason"] = "rollback_trigger_rate_elevated"
    else:
        out["status"] = "degraded"
        out["reason"] = "rollback_triggered"
    status_now = str(out.get("status") or "unknown")
    if status_now == "critical":
        out["action_level"] = "shadow_lock"
        out["action_reason"] = "rollback_trend_critical"
    elif status_now == "degraded":
        out["action_level"] = "degrade"
        out["action_reason"] = "rollback_trend_degraded"
    else:
        out["action_level"] = "observe"
        out["action_reason"] = "rollback_trend_noncritical"

    recs: List[str] = []
    if out["action_level"] == "shadow_lock":
        recs.append("enforce control: set rollout override_mode=shadow and freeze promote path for this window.")
    elif out["action_level"] == "degrade":
        recs.append("degrade guard: keep shadow preference and tighten pre-enforce review.")
    else:
        recs.append("observe: keep monitoring rollback trend without policy change.")

    dominant = str(out.get("dominant_trigger") or "")
    if dominant == "component_mode_drift":
        recs.append("drift focus: inspect configured/effective mode divergence and auto-switch counters.")
    elif dominant == "breach":
        recs.append("breach focus: inspect would_block/cooldown/recover thresholds and data-quality gates.")
    elif dominant == "breach_and_component_mode_drift":
        recs.append("compound focus: require full guard replay before next enforce candidate.")
    out["action_recommendations"] = recs

    hints: List[str] = []
    trigger_counts = out.get("trigger_counts") if isinstance(out.get("trigger_counts"), dict) else {}
    if int(trigger_counts.get("component_mode_drift") or 0) > 0:
        hints.append("drift: audit component-score configured/effective mode drift before next enforce window.")
    if int(trigger_counts.get("breach") or 0) > 0:
        hints.append("breach: inspect would_block/cooldown/recover thresholds and upstream data quality gates.")
    if int(trigger_counts.get("breach_and_component_mode_drift") or 0) > 0:
        hints.append("compound: keep rollout shadowed and require full guard replay prior to promotion.")
    if str(out.get("status") or "") == "critical":
        hints.append("critical: force shadow override and require two stable windows before manual unlock.")
    out["action_hints"] = hints
    return out


def _probe_rollout_proxy_control_trend(
    *,
    path: Path,
    window_hours: float,
    max_lines: int,
) -> Dict[str, Any]:
    min_events = max(1, int(float(os.getenv("PI_GATE_PROXY_CONTROL_TREND_MIN_EVENTS", "3"))))
    rate_degraded = max(
        0.0,
        min(1.0, float(os.getenv("PI_GATE_PROXY_CONTROL_TREND_RATE_DEGRADED", "0.20"))),
    )
    rate_critical = max(
        rate_degraded,
        min(1.0, float(os.getenv("PI_GATE_PROXY_CONTROL_TREND_RATE_CRITICAL", "0.50"))),
    )
    critical_trigger_count = max(
        1,
        int(float(os.getenv("PI_GATE_PROXY_CONTROL_CRITICAL_TRIGGER_COUNT", "1"))),
    )
    out: Dict[str, Any] = {
        "status": "unknown",
        "reason": "insufficient_events",
        "window_hours": float(window_hours),
        "pi_cycle_events": 0,
        "proxy_control_records": 0,
        "enabled_count": 0,
        "applied_count": 0,
        "trigger_count": 0,
        "trigger_rate": None,
        "trigger_counts": {
            "proxy_isolation_critical_candidates": 0,
            "proxy_isolation_actionable_degraded": 0,
            "proxy_isolation_actionable_pending": 0,
        },
        "trigger_shares": {
            "proxy_isolation_critical_candidates": None,
            "proxy_isolation_actionable_degraded": None,
            "proxy_isolation_actionable_pending": None,
        },
        "dominant_trigger": None,
        "dominant_trigger_count": 0,
        "dominant_trigger_share": None,
        "action_level": "observe",
        "action_reason": "insufficient_events",
        "action_recommendations": [],
        "latest_trigger": None,
        "latest_reason": None,
        "latest_applied": None,
        "latest_ts": None,
        "top_triggers": [],
        "action_hints": [],
        "thresholds": {
            "min_events": min_events,
            "rate_degraded": rate_degraded,
            "rate_critical": rate_critical,
            "critical_trigger_count": critical_trigger_count,
        },
    }
    if not path.exists():
        out["reason"] = "pi_cycle_log_missing"
        out["action_level"] = "observe"
        out["action_reason"] = "pi_cycle_log_missing"
        return out

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        out["status"] = "degraded"
        out["reason"] = "pi_cycle_log_unreadable"
        out["action_level"] = "degrade"
        out["action_reason"] = "pi_cycle_log_unreadable"
        out["action_recommendations"] = ["health: fix log pipeline before trusting proxy-control trend metrics."]
        return out

    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=max(0.1, float(window_hours)))
    trigger_counter: Dict[str, int] = {}
    known_triggers = {
        "proxy_isolation_critical_candidates",
        "proxy_isolation_actionable_degraded",
        "proxy_isolation_actionable_pending",
    }

    for raw in reversed(lines[-max(1, int(max_lines)) :]):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or str(obj.get("domain") or "") != "pi_cycle":
            continue
        ts = _parse_ts_utc(obj.get("ts"))
        if ts is None or ts < since:
            continue
        out["pi_cycle_events"] = int(out["pi_cycle_events"]) + 1
        proxy_ctrl = (
            obj.get("gate_rollout_proxy_control")
            if isinstance(obj.get("gate_rollout_proxy_control"), dict)
            else {}
        )
        if not proxy_ctrl:
            continue
        out["proxy_control_records"] = int(out["proxy_control_records"]) + 1
        if bool(proxy_ctrl.get("enabled")):
            out["enabled_count"] = int(out["enabled_count"]) + 1
        applied = bool(proxy_ctrl.get("applied"))
        if applied:
            out["applied_count"] = int(out["applied_count"]) + 1
        trigger = str(proxy_ctrl.get("trigger") or "").strip().lower()
        if trigger in known_triggers:
            out["trigger_count"] = int(out["trigger_count"]) + 1
            trigger_counter[trigger] = trigger_counter.get(trigger, 0) + 1
            out["trigger_counts"][trigger] = int((out["trigger_counts"] or {}).get(trigger) or 0) + 1
        if out["latest_ts"] is None:
            out["latest_ts"] = obj.get("ts")
            out["latest_trigger"] = trigger or None
            out["latest_reason"] = proxy_ctrl.get("reason")
            out["latest_applied"] = applied

    events = int(out["pi_cycle_events"])
    trigger_count = int(out["trigger_count"])
    if events > 0:
        out["trigger_rate"] = round(float(trigger_count) / float(events), 4)
    out["top_triggers"] = [
        {"trigger": k, "count": v}
        for k, v in sorted(trigger_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
    if trigger_count > 0:
        trigger_counts = out.get("trigger_counts") if isinstance(out.get("trigger_counts"), dict) else {}
        trigger_shares = out.get("trigger_shares") if isinstance(out.get("trigger_shares"), dict) else {}
        priorities = {
            "proxy_isolation_critical_candidates": 0,
            "proxy_isolation_actionable_degraded": 1,
            "proxy_isolation_actionable_pending": 2,
        }
        dominant: Optional[str] = None
        dominant_count = 0
        for key in [
            "proxy_isolation_critical_candidates",
            "proxy_isolation_actionable_degraded",
            "proxy_isolation_actionable_pending",
        ]:
            cnt = int(trigger_counts.get(key) or 0)
            trigger_shares[key] = round(float(cnt) / float(trigger_count), 4)
            if cnt <= 0:
                continue
            if dominant is None:
                dominant = key
                dominant_count = cnt
                continue
            current_priority = priorities.get(str(dominant), 9)
            candidate_priority = priorities.get(key, 9)
            if cnt > dominant_count or (cnt == dominant_count and candidate_priority < current_priority):
                dominant = key
                dominant_count = cnt
        out["dominant_trigger"] = dominant
        out["dominant_trigger_count"] = int(dominant_count)
        if dominant and dominant_count > 0:
            out["dominant_trigger_share"] = round(float(dominant_count) / float(trigger_count), 4)

    if events < min_events:
        out["status"] = "unknown"
        out["reason"] = "insufficient_events"
        out["action_level"] = "observe"
        out["action_reason"] = "insufficient_events"
        return out
    if trigger_count <= 0:
        out["status"] = "ok"
        out["reason"] = "no_proxy_control_triggered"
        out["action_level"] = "observe"
        out["action_reason"] = "no_proxy_control_triggered"
        return out

    trigger_rate = float(out.get("trigger_rate") or 0.0)
    critical_count = int(((out.get("trigger_counts") or {}).get("proxy_isolation_critical_candidates") or 0))
    if trigger_rate >= rate_critical:
        out["status"] = "critical"
        out["reason"] = "proxy_control_trigger_rate_high"
    elif critical_count >= critical_trigger_count:
        out["status"] = "critical"
        out["reason"] = "proxy_control_critical_trigger_repeated"
    elif trigger_rate >= rate_degraded:
        out["status"] = "degraded"
        out["reason"] = "proxy_control_trigger_rate_elevated"
    else:
        out["status"] = "degraded"
        out["reason"] = "proxy_control_triggered"

    if str(out.get("status") or "") == "critical":
        out["action_level"] = "shadow_lock"
        out["action_reason"] = "proxy_control_trend_critical"
    else:
        out["action_level"] = "degrade"
        out["action_reason"] = "proxy_control_trend_degraded"

    recs: List[str] = []
    if out["action_level"] == "shadow_lock":
        recs.append("proxy-control: keep rollout in shadow and execute isolation batch before promote path.")
    else:
        recs.append("proxy-control: degrade guard and prioritize proxy isolation batch execution.")
    if str(out.get("dominant_trigger") or "") == "proxy_isolation_critical_candidates":
        recs.append("proxy-control: critical isolation candidates observed; require operator review for upstream path.")
    recs.append("proxy-control: verify proxy timeout floor and stagger patches are converging.")
    out["action_recommendations"] = recs

    hints: List[str] = []
    trigger_counts = out.get("trigger_counts") if isinstance(out.get("trigger_counts"), dict) else {}
    if int(trigger_counts.get("proxy_isolation_critical_candidates") or 0) > 0:
        hints.append("proxy: critical isolation candidates present; run batch_0_proxy_isolation first.")
    if int(trigger_counts.get("proxy_isolation_actionable_degraded") or 0) > 0:
        hints.append("proxy: actionable degraded candidates persisting; keep full guard until trend cools.")
    if int(trigger_counts.get("proxy_isolation_actionable_pending") or 0) > 0:
        hints.append("proxy: pending isolation signals detected; avoid promote-to-enforce this window.")
    if str(out.get("status") or "") == "critical":
        hints.append("critical: lock rollout in shadow for this window and replay core checks before unlock.")
    out["action_hints"] = hints
    return out


def _extract_agent_from_session_key(session_key: str) -> Optional[str]:
    sk = str(session_key or "").strip()
    m = re.match(r"^agent:([^:]+):", sk)
    if not m:
        return None
    agent = str(m.group(1)).strip()
    return agent if agent else None


def _load_last_finished_run(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    for raw in reversed(lines[-200:]):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if str(obj.get("action") or "").lower() != "finished":
            continue
        return obj
    return None


def _load_known_agent_ids(config_path: Optional[Path]) -> List[str]:
    if not config_path or not config_path.exists():
        return []
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    out: List[str] = []
    for item in (cfg.get("agents", {}).get("list") or []):
        aid = str((item or {}).get("id") or "").strip()
        if aid and aid not in out:
            out.append(aid)
    return out


def _build_lane_consistency(
    enabled_jobs: List[Dict[str, Any]],
    runs_dir: Optional[Path],
) -> Dict[str, Any]:
    if not runs_dir or not runs_dir.exists():
        return {
            "status": "unknown",
            "checked_jobs": len(enabled_jobs),
            "jobs_with_runs": 0,
            "mismatch_count": 0,
            "mismatches": [],
            "reason": "runs_dir_missing",
        }

    mismatches: List[Dict[str, Any]] = []
    jobs_with_runs = 0

    for job in enabled_jobs:
        job_id = str(job.get("id") or "").strip()
        job_agent = str(job.get("agentId") or "").strip()
        if not job_id:
            continue

        run_file = runs_dir / f"{job_id}.jsonl"
        last_run = _load_last_finished_run(run_file)
        if not last_run:
            continue
        jobs_with_runs += 1

        runtime_agent = _extract_agent_from_session_key(str(last_run.get("sessionKey") or ""))
        if not runtime_agent or not job_agent:
            continue
        if runtime_agent != job_agent:
            mismatches.append(
                {
                    "id": job_id,
                    "name": job.get("name"),
                    "configured_agent": job_agent,
                    "runtime_agent": runtime_agent,
                    "session_key": str(last_run.get("sessionKey") or ""),
                    "run_at_ms": last_run.get("runAtMs"),
                }
            )

    return {
        "status": "ok" if not mismatches else "degraded",
        "checked_jobs": len(enabled_jobs),
        "jobs_with_runs": jobs_with_runs,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def _build_agent_registry(jobs: List[Dict[str, Any]], known_agents: Optional[List[str]]) -> Dict[str, Any]:
    known = list(known_agents or [])
    if not known:
        return {
            "status": "unknown",
            "known_agents": [],
            "invalid_enabled_count": 0,
            "invalid_disabled_count": 0,
            "invalid_jobs": [],
            "reason": "agent_registry_missing",
        }

    invalid: List[Dict[str, Any]] = []
    for job in jobs:
        aid = str(job.get("agentId") or "").strip()
        if not aid:
            continue
        if aid in known:
            continue
        invalid.append(
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "agent_id": aid,
                "enabled": bool(job.get("enabled", True)),
            }
        )

    invalid_enabled_count = sum(1 for x in invalid if bool(x.get("enabled")))
    invalid_disabled_count = sum(1 for x in invalid if not bool(x.get("enabled")))
    return {
        "status": "ok" if invalid_enabled_count == 0 else "degraded",
        "known_agents": known,
        "invalid_enabled_count": invalid_enabled_count,
        "invalid_disabled_count": invalid_disabled_count,
        "invalid_jobs": invalid[:20],
    }


def _build_cron_backlog(enabled_jobs: List[Dict[str, Any]], now_ms: Optional[int] = None) -> Dict[str, Any]:
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    grace_sec = int(os.getenv("CRON_BACKLOG_GRACE_SEC", "300"))
    overdue: List[Dict[str, Any]] = []

    for job in enabled_jobs:
        st = job.get("state") or {}
        next_run = st.get("nextRunAtMs")
        if not isinstance(next_run, int):
            continue
        delta_sec = int((now - next_run) / 1000)
        if delta_sec <= grace_sec:
            continue
        overdue.append(
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "agentId": job.get("agentId"),
                "overdue_sec": delta_sec,
                "next_run_at_ms": next_run,
            }
        )

    overdue.sort(key=lambda x: int(x.get("overdue_sec") or 0), reverse=True)
    max_overdue = int(overdue[0]["overdue_sec"]) if overdue else 0
    count = len(overdue)
    status = "ok"
    if count > 0:
        status = "degraded"
    if count >= 3 or max_overdue >= 3600:
        status = "critical"

    return {
        "status": status,
        "grace_sec": grace_sec,
        "overdue_count": count,
        "max_overdue_sec": max_overdue,
        "jobs": overdue[:20],
    }


def _build_web_tool_availability(
    jobs: List[Dict[str, Any]],
    config_obj: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    tools = (config_obj or {}).get("tools", {}) if isinstance(config_obj, dict) else {}
    web = tools.get("web", {}) if isinstance(tools, dict) else {}
    search = web.get("search", {}) if isinstance(web, dict) else {}
    api_key = str(search.get("apiKey") or "").strip()
    configured = bool(api_key)

    probe_job = None
    for j in jobs:
        if str(j.get("name") or "") == "web-access-selfcheck-6h":
            probe_job = j
            break

    if probe_job is None:
        return {
            "status": "unknown",
            "configured": configured,
            "probe_job_found": False,
            "reason": "probe_job_missing",
        }

    st = probe_job.get("state") or {}
    last_status = str(st.get("lastStatus") or "").lower()
    consecutive_errors = int(st.get("consecutiveErrors") or 0)
    status = "ok" if configured and last_status == "ok" else "degraded"
    if not configured:
        status = "critical"

    return {
        "status": status,
        "configured": configured,
        "probe_job_found": True,
        "probe_job_id": probe_job.get("id"),
        "probe_last_status": last_status,
        "probe_consecutive_errors": consecutive_errors,
        "probe_last_run_at_ms": st.get("lastRunAtMs"),
        "probe_next_run_at_ms": st.get("nextRunAtMs"),
    }


def _build_cron_policy_audit(
    enabled_jobs: List[Dict[str, Any]],
    proxy_error_recent_job_ids: Optional[set[str]] = None,
) -> Dict[str, Any]:
    missing_timeout: List[Dict[str, Any]] = []
    low_timeout: List[Dict[str, Any]] = []
    proxy_timeout_floor: List[Dict[str, Any]] = []
    proxy_timeout_floor_recent: List[Dict[str, Any]] = []
    proxy_error_jobs: List[Dict[str, Any]] = []
    no_stagger: List[Dict[str, Any]] = []
    remediation_map: Dict[str, Dict[str, Any]] = {}
    recent_job_ids = {str(x) for x in (proxy_error_recent_job_ids or set()) if str(x).strip()}

    min_timeout = int(os.getenv("CRON_MIN_TIMEOUT_SEC", "120"))
    proxy_timeout_floor_sec = int(
        float(
            os.getenv(
                "CRON_PROXY_ERROR_TIMEOUT_FLOOR_SEC",
                str(max(min_timeout, 300)),
            )
        )
    )
    proxy_timeout_floor_sec = max(min_timeout, proxy_timeout_floor_sec)
    require_stagger = os.getenv("CRON_REQUIRE_STAGGER", "1").strip().lower() in {"1", "true", "yes", "on"}
    stagger_max_ms = int(float(os.getenv("CRON_RECOMMEND_STAGGER_MAX_MS", "900000")))
    stagger_step_ms = int(float(os.getenv("CRON_RECOMMEND_STAGGER_STEP_MS", "15000")))
    stagger_max_ms = max(60_000, stagger_max_ms)
    stagger_step_ms = max(1000, min(stagger_step_ms, stagger_max_ms))

    exempt = {
        x.strip()
        for x in os.getenv("CRON_STAGGER_EXEMPT_NAMES", "lie-spot-halfhour-core").split(",")
        if x.strip()
    }

    def _recommend_stagger_ms(job_id: str, name: str) -> int:
        key = f"{job_id}:{name}".encode("utf-8", errors="ignore")
        digest = hashlib.sha1(key).hexdigest()
        seed = int(digest[:8], 16)
        bucket_count = max(1, stagger_max_ms // stagger_step_ms)
        idx = (seed % bucket_count) + 1
        return int(min(stagger_max_ms, idx * stagger_step_ms))

    def _action_for(job_id: str, name: str) -> Dict[str, Any]:
        key = f"{job_id}:{name}"
        action = remediation_map.get(key)
        if action is None:
            action = {
                "id": job_id or None,
                "name": name or None,
                "reasons": [],
            }
            remediation_map[key] = action
        return action

    def _action_add_reason(action: Dict[str, Any], reason: str) -> None:
        reasons = action.get("reasons")
        if not isinstance(reasons, list):
            reasons = []
            action["reasons"] = reasons
        if reason not in reasons:
            reasons.append(reason)

    def _is_high_freq_cron(expr: str) -> bool:
        parts = str(expr or "").strip().split()
        if len(parts) != 5:
            return False
        minute, hour = parts[0], parts[1]
        minute_high = ("*/" in minute) or ("," in minute) or ("-" in minute)
        hour_high = hour == "*" or ("*/" in hour) or ("," in hour) or ("-" in hour)
        return minute_high or hour_high

    for job in enabled_jobs:
        payload = job.get("payload") or {}
        schedule = job.get("schedule") or {}
        state = job.get("state") if isinstance(job.get("state"), dict) else {}
        name = str(job.get("name") or "")
        jid = str(job.get("id") or "")
        timeout = payload.get("timeoutSeconds") if isinstance(payload, dict) else None
        last_status = str(state.get("lastStatus") or "").strip().lower()
        last_error = str(state.get("lastError") or "")
        proxy_error_active = last_status == "error" and _is_proxy_error_text(last_error)
        proxy_error_recent = bool(jid and jid in recent_job_ids)

        if timeout is None:
            missing_timeout.append({"id": jid, "name": name})
            action = _action_for(jid, name)
            _action_add_reason(action, "missing_timeout")
            action["set_timeout_seconds"] = max(int(min_timeout), int(action.get("set_timeout_seconds") or 0))
        else:
            try:
                t = int(timeout)
            except Exception:
                t = 0
            if t < min_timeout:
                low_timeout.append({"id": jid, "name": name, "timeoutSeconds": t})
                action = _action_for(jid, name)
                _action_add_reason(action, "low_timeout")
                action["set_timeout_seconds"] = max(int(min_timeout), int(action.get("set_timeout_seconds") or 0))
            if (proxy_error_active or proxy_error_recent) and t < proxy_timeout_floor_sec:
                source = "active_state" if proxy_error_active else "recent_trend"
                proxy_timeout_floor.append(
                    {
                        "id": jid,
                        "name": name,
                        "timeoutSeconds": t,
                        "recommendedTimeoutSeconds": proxy_timeout_floor_sec,
                        "source": source,
                    }
                )
                if source == "recent_trend":
                    proxy_timeout_floor_recent.append(
                        {
                            "id": jid,
                            "name": name,
                            "timeoutSeconds": t,
                            "recommendedTimeoutSeconds": proxy_timeout_floor_sec,
                        }
                    )
                action = _action_for(jid, name)
                _action_add_reason(
                    action,
                    "proxy_timeout_floor" if proxy_error_active else "proxy_timeout_floor_recent",
                )
                action["set_timeout_seconds"] = max(
                    int(proxy_timeout_floor_sec),
                    int(action.get("set_timeout_seconds") or 0),
                )

        if proxy_error_active:
            proxy_error_jobs.append(
                {
                    "id": jid,
                    "name": name,
                    "lastError": last_error[:220],
                    "consecutiveErrors": int(state.get("consecutiveErrors") or 0),
                    "source": "active_state",
                }
            )
        elif proxy_error_recent:
            proxy_error_jobs.append(
                {
                    "id": jid,
                    "name": name,
                    "lastError": "",
                    "consecutiveErrors": int(state.get("consecutiveErrors") or 0),
                    "source": "recent_trend",
                }
            )

        if (
            require_stagger
            and isinstance(schedule, dict)
            and str(schedule.get("kind") or "") == "cron"
            and name not in exempt
            and _is_high_freq_cron(str(schedule.get("expr") or ""))
        ):
            # Jobs on sharp boundaries without stagger are more prone to burst collisions.
            if schedule.get("staggerMs") is None:
                no_stagger.append({"id": jid, "name": name})
                action = _action_for(jid, name)
                _action_add_reason(action, "missing_stagger")
                action["set_stagger_ms"] = int(
                    action.get("set_stagger_ms") or _recommend_stagger_ms(job_id=jid, name=name)
                )

    status = "ok"
    if missing_timeout or low_timeout or proxy_timeout_floor or no_stagger:
        status = "degraded"

    remediation: List[Dict[str, Any]] = []
    for action in remediation_map.values():
        reasons = action.get("reasons")
        if isinstance(reasons, list):
            action["reasons"] = sorted([str(x) for x in reasons if str(x).strip()])
        operations: List[Dict[str, Any]] = []
        timeout_sec = action.get("set_timeout_seconds")
        if timeout_sec is not None:
            operations.append(
                {
                    "op": "set",
                    "path": "payload.timeoutSeconds",
                    "value": int(timeout_sec),
                }
            )
        stagger_ms = action.get("set_stagger_ms")
        if stagger_ms is not None:
            operations.append(
                {
                    "op": "set",
                    "path": "schedule.staggerMs",
                    "value": int(stagger_ms),
                }
            )
        action["operations"] = operations
        action["operation_count"] = len(operations)
        key_src = f"{action.get('id') or ''}:{action.get('name') or ''}"
        action["plan_id"] = hashlib.sha1(key_src.encode("utf-8", errors="ignore")).hexdigest()[:12]
        remediation.append(action)
    remediation.sort(
        key=lambda x: (
            -len(x.get("reasons") or []),
            str(x.get("name") or ""),
        )
    )
    remediation_operation_count = sum(int(x.get("operation_count") or 0) for x in remediation)
    remediation_ready_count = sum(1 for x in remediation if int(x.get("operation_count") or 0) > 0)

    issue_types: List[str] = []
    if missing_timeout:
        issue_types.append("missing_timeout")
    if low_timeout:
        issue_types.append("low_timeout")
    if proxy_timeout_floor:
        issue_types.append("proxy_timeout_floor")
    if proxy_timeout_floor_recent:
        issue_types.append("proxy_timeout_floor_recent")
    if no_stagger:
        issue_types.append("missing_stagger")

    return {
        "status": status,
        "status_reason": "ok" if not issue_types else ",".join(issue_types),
        "min_timeout_sec": min_timeout,
        "proxy_timeout_floor_sec": proxy_timeout_floor_sec,
        "require_stagger": require_stagger,
        "stagger_exempt_names": sorted(exempt),
        "recommend_stagger_max_ms": stagger_max_ms,
        "recommend_stagger_step_ms": stagger_step_ms,
        "missing_timeout_count": len(missing_timeout),
        "low_timeout_count": len(low_timeout),
        "proxy_timeout_floor_count": len(proxy_timeout_floor),
        "proxy_timeout_floor_recent_count": len(proxy_timeout_floor_recent),
        "proxy_error_job_count": len(proxy_error_jobs),
        "proxy_error_recent_job_count": len(recent_job_ids),
        "no_stagger_count": len(no_stagger),
        "missing_timeout": missing_timeout[:20],
        "low_timeout": low_timeout[:20],
        "proxy_timeout_floor": proxy_timeout_floor[:20],
        "proxy_timeout_floor_recent": proxy_timeout_floor_recent[:20],
        "proxy_error_jobs": proxy_error_jobs[:20],
        "no_stagger": no_stagger[:20],
        "remediation_count": len(remediation),
        "remediation_ready_count": remediation_ready_count,
        "remediation_operation_count": remediation_operation_count,
        "remediation_mode": "job_patch_operations",
        "remediation_apply_hints": [
            "Generate patch from remediation.operations grouped by job id.",
            "Apply to ~/.openclaw/cron/jobs.json and validate with openclaw cron list --json.",
            "Rollout with stagger first; avoid changing all high-frequency jobs at once.",
            "For recurring proxy_5xx jobs, raise timeout floor and keep retries/fallback enabled in pi_cycle_orchestrator.",
        ],
        "remediation": remediation[:50],
    }


def _build_proxy_error_audit(enabled_jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    degraded_job_count = max(1, int(float(os.getenv("CRON_PROXY_ERROR_DEGRADED_JOB_COUNT", "1"))))
    critical_job_count = max(
        degraded_job_count,
        int(float(os.getenv("CRON_PROXY_ERROR_CRITICAL_JOB_COUNT", "2"))),
    )
    degraded_consecutive = max(1, int(float(os.getenv("CRON_PROXY_ERROR_DEGRADED_CONSECUTIVE", "2"))))
    critical_consecutive = max(
        degraded_consecutive,
        int(float(os.getenv("CRON_PROXY_ERROR_CRITICAL_CONSECUTIVE", "4"))),
    )

    proxy_jobs: List[Dict[str, Any]] = []
    core_proxy_jobs = 0
    max_consecutive = 0
    for job in enabled_jobs:
        state = job.get("state") if isinstance(job.get("state"), dict) else {}
        last_status = str(state.get("lastStatus") or "").strip().lower()
        last_error = str(state.get("lastError") or "")
        if last_status != "error" or not _is_proxy_error_text(last_error):
            continue
        consecutive = max(0, int(state.get("consecutiveErrors") or 0))
        max_consecutive = max(max_consecutive, consecutive)
        name = str(job.get("name") or "")
        if name in CORE_JOB_NAMES:
            core_proxy_jobs += 1
        proxy_jobs.append(
            {
                "id": job.get("id"),
                "name": name or None,
                "agentId": job.get("agentId"),
                "consecutiveErrors": consecutive,
                "lastError": last_error[:220],
            }
        )

    proxy_jobs.sort(key=lambda x: int(x.get("consecutiveErrors") or 0), reverse=True)
    proxy_count = len(proxy_jobs)

    status = "ok"
    reason = "no_proxy_errors"
    action_level = "observe"
    action_reason = "no_proxy_errors"
    next_action = "none"
    action_hints: List[str] = []
    if proxy_count > 0:
        status = "degraded"
        reason = "proxy_error_detected"
        action_level = "degrade"
        action_reason = "proxy_error_detected"
        next_action = "apply_proxy_timeout_patch"
        action_hints.append("proxy: inspect upstream path and recent scheduler/network changes.")
        action_hints.append("proxy: raise timeout floor for recurring proxy_5xx jobs, then stagger rollout.")
        action_hints.append("proxy: keep pi_cycle critical-step retries enabled to absorb transient 500 spikes.")
        if proxy_count >= critical_job_count or max_consecutive >= critical_consecutive:
            status = "critical"
            reason = "proxy_error_persistent"
            action_level = "shadow_lock"
            action_reason = "proxy_error_persistent"
            next_action = "stabilize_proxy_path"
        elif proxy_count >= degraded_job_count or max_consecutive >= degraded_consecutive:
            status = "degraded"
            reason = "proxy_error_elevated"
            action_level = "degrade"
            action_reason = "proxy_error_elevated"
            next_action = "apply_proxy_timeout_patch"

    return {
        "status": status,
        "reason": reason,
        "proxy_error_jobs_count": proxy_count,
        "proxy_error_core_jobs_count": core_proxy_jobs,
        "max_consecutive_errors": max_consecutive,
        "thresholds": {
            "degraded_job_count": degraded_job_count,
            "critical_job_count": critical_job_count,
            "degraded_consecutive": degraded_consecutive,
            "critical_consecutive": critical_consecutive,
        },
        "action_level": action_level,
        "action_reason": action_reason,
        "next_action": next_action,
        "action_hints": action_hints[:5],
        "jobs": proxy_jobs[:20],
    }


def _collect_proxy_retry_exhausted_from_pi_cycle(
    *,
    path: Path,
    since: dt.datetime,
    max_lines: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pi_cycle_events": 0,
        "retry_exhausted_events": 0,
        "retry_exhausted_total": 0,
        "retry_exhausted_max": 0,
        "retry_exhausted_rate": None,
        "retry_rows": [],
    }
    if not path.exists():
        return out
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out
    for raw in reversed(lines[-max_lines:]):
        line = str(raw or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if str(obj.get("domain") or "").strip() != "pi_cycle":
            continue
        ts = _parse_ts_utc(obj.get("ts"))
        if ts is not None and ts < since:
            break
        out["pi_cycle_events"] = int(out.get("pi_cycle_events") or 0) + 1
        exhausted = max(0, _as_int(obj.get("core_proxy_bypass_no_proxy_retry_exhausted"), 0))
        if exhausted <= 0:
            continue
        out["retry_exhausted_events"] = int(out.get("retry_exhausted_events") or 0) + 1
        out["retry_exhausted_total"] = int(out.get("retry_exhausted_total") or 0) + exhausted
        out["retry_exhausted_max"] = max(int(out.get("retry_exhausted_max") or 0), exhausted)
        rows = out.get("retry_rows") if isinstance(out.get("retry_rows"), list) else []
        if len(rows) < 20:
            rows.append(
                {
                    "ts": obj.get("ts"),
                    "retry_exhausted": exhausted,
                    "core_proxy_status": obj.get("core_proxy_bypass_status"),
                    "core_proxy_reason": obj.get("core_proxy_bypass_reason"),
                    "ops_next_action_priority": obj.get("ops_next_action_priority"),
                    "ops_next_action_reason": obj.get("ops_next_action_reason"),
                }
            )
            out["retry_rows"] = rows
    total_events = int(out.get("pi_cycle_events") or 0)
    retry_events = int(out.get("retry_exhausted_events") or 0)
    if total_events > 0:
        out["retry_exhausted_rate"] = round(float(retry_events) / float(total_events), 4)
    return out


def _build_proxy_error_trend_audit(
    enabled_jobs: List[Dict[str, Any]],
    runs_dir: Optional[Path],
) -> Dict[str, Any]:
    window_hours = max(1.0, float(os.getenv("CRON_PROXY_TREND_WINDOW_HOURS", "24")))
    max_lines_per_job = max(20, int(float(os.getenv("CRON_PROXY_TREND_MAX_LINES_PER_JOB", "200"))))
    min_runs_per_job = max(1, int(float(os.getenv("CRON_PROXY_TREND_MIN_RUNS_PER_JOB", "3"))))
    degraded_rate = max(0.0, min(1.0, float(os.getenv("CRON_PROXY_TREND_DEGRADED_FAIL_RATE", "0.20"))))
    critical_rate = max(degraded_rate, min(1.0, float(os.getenv("CRON_PROXY_TREND_CRITICAL_FAIL_RATE", "0.45"))))
    degraded_job_count = max(1, int(float(os.getenv("CRON_PROXY_TREND_DEGRADED_JOB_COUNT", "1"))))
    critical_job_count = max(
        degraded_job_count,
        int(float(os.getenv("CRON_PROXY_TREND_CRITICAL_JOB_COUNT", "2"))),
    )
    degraded_consecutive = max(1, int(float(os.getenv("CRON_PROXY_TREND_DEGRADED_CONSECUTIVE", "2"))))
    critical_consecutive = max(
        degraded_consecutive,
        int(float(os.getenv("CRON_PROXY_TREND_CRITICAL_CONSECUTIVE", "4"))),
    )
    retry_degraded_events = max(
        1, int(float(os.getenv("CRON_PROXY_TREND_RETRY_EXHAUSTED_DEGRADED_EVENTS", "1")))
    )
    retry_critical_events = max(
        retry_degraded_events,
        int(float(os.getenv("CRON_PROXY_TREND_RETRY_EXHAUSTED_CRITICAL_EVENTS", "3"))),
    )
    retry_degraded_rate = max(
        0.0,
        min(1.0, float(os.getenv("CRON_PROXY_TREND_RETRY_EXHAUSTED_DEGRADED_RATE", "0.10"))),
    )
    retry_critical_rate = max(
        retry_degraded_rate,
        min(1.0, float(os.getenv("CRON_PROXY_TREND_RETRY_EXHAUSTED_CRITICAL_RATE", "0.30"))),
    )
    retry_degraded_max = max(
        1, int(float(os.getenv("CRON_PROXY_TREND_RETRY_EXHAUSTED_DEGRADED_MAX", "1")))
    )
    retry_critical_max = max(
        retry_degraded_max,
        int(float(os.getenv("CRON_PROXY_TREND_RETRY_EXHAUSTED_CRITICAL_MAX", "2"))),
    )
    pi_cycle_max_lines = max(
        200, int(float(os.getenv("CRON_PROXY_TREND_PI_CYCLE_MAX_LINES", "4000")))
    )

    out: Dict[str, Any] = {
        "status": "unknown",
        "reason": "runs_dir_missing",
        "window_hours": float(window_hours),
        "sample_jobs": 0,
        "jobs_with_recent_runs": 0,
        "jobs_with_proxy_errors": 0,
        "core_jobs_with_proxy_errors": 0,
        "finished_runs_total": 0,
        "proxy_error_runs_total": 0,
        "proxy_error_rate": None,
        "max_consecutive_proxy_errors": 0,
        "pi_cycle_path": str(Path(os.getenv("PI_CYCLE_LOG_JSONL", str(DEFAULT_PI_CYCLE_LOG))).expanduser()),
        "pi_cycle_events_total": 0,
        "retry_exhausted_events": 0,
        "retry_exhausted_total": 0,
        "retry_exhausted_max": 0,
        "retry_exhausted_rate": None,
        "retry_exhausted_rows": [],
        "action_level": "observe",
        "action_reason": "runs_dir_missing",
        "next_action": "none",
        "action_hints": [],
        "recent_proxy_job_ids": [],
        "thresholds": {
            "min_runs_per_job": min_runs_per_job,
            "degraded_fail_rate": degraded_rate,
            "critical_fail_rate": critical_rate,
            "degraded_job_count": degraded_job_count,
            "critical_job_count": critical_job_count,
            "degraded_consecutive": degraded_consecutive,
            "critical_consecutive": critical_consecutive,
            "retry_exhausted_degraded_events": retry_degraded_events,
            "retry_exhausted_critical_events": retry_critical_events,
            "retry_exhausted_degraded_rate": retry_degraded_rate,
            "retry_exhausted_critical_rate": retry_critical_rate,
            "retry_exhausted_degraded_max": retry_degraded_max,
            "retry_exhausted_critical_max": retry_critical_max,
            "pi_cycle_max_lines": pi_cycle_max_lines,
        },
        "jobs": [],
    }
    if not runs_dir or not runs_dir.exists():
        runs_dir = None

    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=float(window_hours))
    rows: List[Dict[str, Any]] = []
    recent_proxy_job_ids: List[str] = []
    core_jobs_with_proxy_errors = 0
    max_consecutive_proxy_errors = 0
    finished_runs_total = 0
    proxy_error_runs_total = 0
    pi_cycle_path = Path(str(out.get("pi_cycle_path") or "")).expanduser()
    retry_signal = _collect_proxy_retry_exhausted_from_pi_cycle(
        path=pi_cycle_path,
        since=since,
        max_lines=pi_cycle_max_lines,
    )

    for job in enabled_jobs:
        if not isinstance(job, dict):
            continue
        jid = str(job.get("id") or "").strip()
        name = str(job.get("name") or "").strip()
        if not jid:
            continue
        out["sample_jobs"] = int(out.get("sample_jobs") or 0) + 1
        if runs_dir is None:
            continue
        run_file = runs_dir / f"{jid}.jsonl"
        if not run_file.exists():
            continue
        try:
            lines = run_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        total_runs = 0
        proxy_runs = 0
        latest_consecutive_proxy = 0
        consecutive_open = True
        latest_proxy_ts: Optional[str] = None
        latest_proxy_error = ""

        for raw in reversed(lines[-max_lines_per_job:]):
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict) or str(event.get("action") or "").lower() != "finished":
                continue
            run_ts = _as_ts_utc_from_ms_or_iso(event.get("runAtMs")) or _as_ts_utc_from_ms_or_iso(event.get("ts"))
            if run_ts is not None and run_ts < since:
                break

            total_runs += 1
            status = str(event.get("status") or "").strip().lower()
            err = str(event.get("error") or "")
            is_proxy = status == "error" and _is_proxy_error_text(err)
            if is_proxy:
                proxy_runs += 1
                if latest_proxy_ts is None and run_ts is not None:
                    latest_proxy_ts = run_ts.isoformat()
                if latest_proxy_error == "":
                    latest_proxy_error = err[:220]
                if consecutive_open:
                    latest_consecutive_proxy += 1
            else:
                consecutive_open = False

        if total_runs <= 0:
            continue
        out["jobs_with_recent_runs"] = int(out.get("jobs_with_recent_runs") or 0) + 1
        finished_runs_total += total_runs
        proxy_error_runs_total += proxy_runs

        if proxy_runs <= 0:
            continue
        if total_runs >= min_runs_per_job:
            recent_proxy_job_ids.append(jid)
        max_consecutive_proxy_errors = max(max_consecutive_proxy_errors, latest_consecutive_proxy)
        if name in CORE_JOB_NAMES:
            core_jobs_with_proxy_errors += 1
        rows.append(
            {
                "id": jid,
                "name": name or None,
                "agentId": job.get("agentId"),
                "sample_runs": int(total_runs),
                "proxy_error_runs": int(proxy_runs),
                "proxy_error_rate": round(float(proxy_runs) / float(total_runs), 4),
                "latest_consecutive_proxy_errors": int(latest_consecutive_proxy),
                "latest_proxy_ts": latest_proxy_ts,
                "latest_proxy_error": latest_proxy_error,
                "core": bool(name in CORE_JOB_NAMES),
            }
        )

    rows.sort(
        key=lambda x: (
            float(x.get("proxy_error_rate") or 0.0),
            int(x.get("latest_consecutive_proxy_errors") or 0),
            int(x.get("proxy_error_runs") or 0),
        ),
        reverse=True,
    )

    proxy_job_count = len(rows)
    proxy_error_rate = None
    if finished_runs_total > 0:
        proxy_error_rate = round(float(proxy_error_runs_total) / float(finished_runs_total), 4)

    out["jobs_with_proxy_errors"] = int(proxy_job_count)
    out["core_jobs_with_proxy_errors"] = int(core_jobs_with_proxy_errors)
    out["finished_runs_total"] = int(finished_runs_total)
    out["proxy_error_runs_total"] = int(proxy_error_runs_total)
    out["proxy_error_rate"] = proxy_error_rate
    out["max_consecutive_proxy_errors"] = int(max_consecutive_proxy_errors)
    out["recent_proxy_job_ids"] = recent_proxy_job_ids[:50]
    out["jobs"] = rows[:20]
    out["pi_cycle_events_total"] = int(retry_signal.get("pi_cycle_events") or 0)
    out["retry_exhausted_events"] = int(retry_signal.get("retry_exhausted_events") or 0)
    out["retry_exhausted_total"] = int(retry_signal.get("retry_exhausted_total") or 0)
    out["retry_exhausted_max"] = int(retry_signal.get("retry_exhausted_max") or 0)
    out["retry_exhausted_rate"] = retry_signal.get("retry_exhausted_rate")
    out["retry_exhausted_rows"] = (
        retry_signal.get("retry_rows")[:20]
        if isinstance(retry_signal.get("retry_rows"), list)
        else []
    )

    retry_events = int(out.get("retry_exhausted_events") or 0)
    retry_total = int(out.get("retry_exhausted_total") or 0)
    retry_max = int(out.get("retry_exhausted_max") or 0)
    retry_rate = float(out.get("retry_exhausted_rate") or 0.0)
    retry_rate_degraded_ready = int(out.get("pi_cycle_events_total") or 0) >= max(2, retry_degraded_events)
    retry_rate_critical_ready = int(out.get("pi_cycle_events_total") or 0) >= max(3, retry_critical_events)
    retry_degraded = (
        retry_events >= retry_degraded_events
        or retry_max >= retry_degraded_max
        or (retry_rate_degraded_ready and retry_rate >= retry_degraded_rate)
    )
    retry_critical = (
        retry_events >= retry_critical_events
        or retry_max >= retry_critical_max
        or (retry_rate_critical_ready and retry_rate >= retry_critical_rate)
    )

    if finished_runs_total <= 0 and retry_events <= 0:
        out["status"] = "unknown"
        out["reason"] = "no_recent_runs"
        out["action_level"] = "observe"
        out["action_reason"] = "no_recent_runs"
        return out
    if proxy_job_count <= 0 and retry_events <= 0:
        out["status"] = "ok"
        out["reason"] = "no_proxy_errors_recent"
        out["action_level"] = "observe"
        out["action_reason"] = "no_proxy_errors_recent"
        return out

    out["status"] = "degraded"
    out["reason"] = "proxy_error_recent_detected"
    out["action_level"] = "degrade"
    out["action_reason"] = "proxy_error_recent_detected"
    out["next_action"] = "apply_proxy_timeout_patch_draft"
    out["action_hints"] = [
        "proxy trend: apply timeout floor patch for repeated proxy_5xx jobs.",
        "proxy trend: keep stagger for high-frequency cron to reduce burst collisions.",
    ]
    if proxy_job_count <= 0 and retry_events > 0:
        out["reason"] = "proxy_retry_exhausted_recent_detected"
        out["action_reason"] = "proxy_retry_exhausted_recent_detected"
        out["next_action"] = "stabilize_proxy_path_and_raise_timeout_floor"
        out["action_hints"] = [
            "proxy trend: no_proxy retry exhaustion detected in pi_cycle path.",
            "proxy trend: raise timeout floor and verify proxy/gateway upstream health.",
        ]

    fail_rate = float(proxy_error_rate or 0.0)
    proxy_critical = (
        proxy_job_count >= critical_job_count
        or max_consecutive_proxy_errors >= critical_consecutive
        or fail_rate >= critical_rate
    )
    proxy_degraded = (
        proxy_job_count >= degraded_job_count
        or max_consecutive_proxy_errors >= degraded_consecutive
        or fail_rate >= degraded_rate
    )

    if proxy_critical or retry_critical:
        out["status"] = "critical"
        if proxy_critical and retry_critical:
            out["reason"] = "proxy_error_and_retry_exhausted_persistent"
            out["action_reason"] = "proxy_error_and_retry_exhausted_persistent"
        elif proxy_critical:
            out["reason"] = "proxy_error_recent_persistent"
            out["action_reason"] = "proxy_error_recent_persistent"
        else:
            out["reason"] = "proxy_retry_exhausted_persistent"
            out["action_reason"] = "proxy_retry_exhausted_persistent"
        out["action_level"] = "shadow_lock"
        out["next_action"] = "stabilize_proxy_path_and_isolate_jobs"
        out["action_hints"] = [
            "proxy trend critical: isolate recurring proxy_5xx jobs and validate gateway upstream path.",
            "proxy trend critical: force conservative guard mode until proxy_fail_rate returns below degraded threshold.",
            f"proxy retry critical: exhausted_events={retry_events}, exhausted_total={retry_total}, exhausted_max={retry_max}.",
        ]
    elif proxy_degraded or retry_degraded:
        out["status"] = "degraded"
        if proxy_degraded and retry_degraded:
            out["reason"] = "proxy_error_and_retry_exhausted_elevated"
            out["action_reason"] = "proxy_error_and_retry_exhausted_elevated"
        elif proxy_degraded:
            out["reason"] = "proxy_error_recent_elevated"
            out["action_reason"] = "proxy_error_recent_elevated"
        else:
            out["reason"] = "proxy_retry_exhausted_elevated"
            out["action_reason"] = "proxy_retry_exhausted_elevated"
            out["next_action"] = "stabilize_proxy_path_and_raise_timeout_floor"
        hints = out.get("action_hints") if isinstance(out.get("action_hints"), list) else []
        hints.append(
            f"proxy retry elevated: exhausted_events={retry_events}, exhausted_total={retry_total}, exhausted_max={retry_max}."
        )
        out["action_hints"] = hints[:6]
    return out


def _probe_gate_rollout() -> Dict[str, Any]:
    path = Path(os.getenv("PI_CYCLE_LOG_JSONL", str(DEFAULT_PI_CYCLE_LOG))).expanduser()
    trend = compute_notify_trend(path, include_latest_event=True)
    latest_obj = trend.get("latest_event") if isinstance(trend.get("latest_event"), dict) else None
    trend_status_raw = str(trend.get("status") or "").strip().lower()
    trend_reason = str(trend.get("reason") or "").strip()
    try:
        drift_window_hours = float(
            os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_WINDOW_HOURS", str(trend.get("window_hours") or "24"))
        )
    except Exception:
        drift_window_hours = 24.0
    drift_window_hours = max(0.1, float(drift_window_hours))
    drift_max_lines = max(
        200,
        int(float(os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_MAX_LINES", "5000"))),
    )
    rollback_max_lines = max(
        200,
        int(float(os.getenv("PI_GATE_ROLLBACK_TREND_MAX_LINES", str(drift_max_lines)))),
    )
    proxy_control_max_lines = max(
        200,
        int(float(os.getenv("PI_GATE_PROXY_CONTROL_TREND_MAX_LINES", str(rollback_max_lines)))),
    )
    drift_signal = _probe_component_mode_drift(
        path=path,
        window_hours=drift_window_hours,
        max_lines=drift_max_lines,
    )
    rollback_signal = _probe_rollout_rollback_trend(
        path=path,
        window_hours=drift_window_hours,
        max_lines=rollback_max_lines,
    )
    proxy_control_signal = _probe_rollout_proxy_control_trend(
        path=path,
        window_hours=drift_window_hours,
        max_lines=proxy_control_max_lines,
    )

    if trend_reason in {"pi_cycle_log_missing", "pi_cycle_log_unreadable"} and not latest_obj:
        return {
            "status": "degraded" if trend_reason == "pi_cycle_log_unreadable" else "unknown",
            "source": "pi_cycle_event",
            "reason": trend_reason,
            "path": str(path),
        }

    if not latest_obj:
        return {
            "status": "unknown",
            "source": "pi_cycle_event",
            "reason": "no_pi_cycle_event",
            "path": str(path),
        }

    obj = latest_obj
    gate = obj.get("gate_rollout") if isinstance(obj.get("gate_rollout"), dict) else {}
    alert = obj.get("gate_rollout_alert") if isinstance(obj.get("gate_rollout_alert"), dict) else {}
    emit = obj.get("gate_rollout_alert_emit") if isinstance(obj.get("gate_rollout_alert_emit"), dict) else {}
    notify = emit.get("notify") if isinstance(emit.get("notify"), dict) else {}
    rollback = obj.get("gate_rollout_rollback") if isinstance(obj.get("gate_rollout_rollback"), dict) else {}
    proxy_control = (
        obj.get("gate_rollout_proxy_control")
        if isinstance(obj.get("gate_rollout_proxy_control"), dict)
        else {}
    )
    digital_life_control = (
        obj.get("gate_rollout_digital_life_control")
        if isinstance(obj.get("gate_rollout_digital_life_control"), dict)
        else {}
    )
    steps = obj.get("steps") if isinstance(obj.get("steps"), list) else []
    retry_attempts_total = 0
    retried_steps_count = 0
    optional_fallback_count = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        attempts = _as_int(step.get("attempt_count"), 1)
        retry_attempts_total += max(1, attempts)
        if attempts > 1:
            retried_steps_count += 1
        if bool(step.get("fallback_applied")):
            optional_fallback_count += 1
    if not gate:
        return {
            "status": "unknown",
            "source": "pi_cycle_event",
            "reason": "gate_rollout_missing",
            "path": str(path),
            "pi_cycle_ts": obj.get("ts"),
        }

    gate_status = str(gate.get("status") or "unknown")
    if gate_status not in {"ok", "degraded", "critical", "unknown"}:
        gate_status = "unknown"
    alert_status = str(alert.get("status") or "unknown")
    if alert_status not in {"ok", "degraded", "critical", "unknown"}:
        alert_status = "unknown"

    status = gate_status
    rank = {"unknown": 0, "ok": 1, "degraded": 2, "critical": 3}
    if rank.get(alert_status, 0) > rank.get(status, 0):
        status = alert_status
    notify_trend_status = trend_status_raw if trend_status_raw in {"ok", "degraded", "critical", "unknown"} else "unknown"
    if rank.get(notify_trend_status, 0) > rank.get(status, 0):
        status = notify_trend_status
    drift_status = str(drift_signal.get("status") or "unknown").strip().lower()
    if drift_status in {"ok", "degraded", "critical", "unknown"} and rank.get(drift_status, 0) > rank.get(status, 0):
        status = drift_status
    rollback_status = str(rollback_signal.get("status") or "unknown").strip().lower()
    if rollback_status in {"ok", "degraded", "critical", "unknown"} and rank.get(rollback_status, 0) > rank.get(status, 0):
        status = rollback_status
    proxy_control_status = str(proxy_control_signal.get("status") or "unknown").strip().lower()
    if (
        proxy_control_status in {"ok", "degraded", "critical", "unknown"}
        and rank.get(proxy_control_status, 0) > rank.get(status, 0)
    ):
        status = proxy_control_status
    proxy_error_status_guard = str(obj.get("guard_mode_proxy_error_status") or "unknown").strip().lower()
    if (
        proxy_error_status_guard in {"ok", "degraded", "critical", "unknown"}
        and rank.get(proxy_error_status_guard, 0) > rank.get(status, 0)
    ):
        status = proxy_error_status_guard
    proxy_error_trend_status_guard = str(
        obj.get("guard_mode_proxy_error_trend_status") or "unknown"
    ).strip().lower()
    if (
        proxy_error_trend_status_guard in {"ok", "degraded", "critical", "unknown"}
        and rank.get(proxy_error_trend_status_guard, 0) > rank.get(status, 0)
    ):
        status = proxy_error_trend_status_guard
    core_proxy_bypass_status = str(obj.get("core_proxy_bypass_status") or "unknown").strip().lower()
    if (
        core_proxy_bypass_status in {"ok", "degraded", "critical", "unknown"}
        and rank.get(core_proxy_bypass_status, 0) > rank.get(status, 0)
    ):
        status = core_proxy_bypass_status
    core_execution_status = str(obj.get("core_execution_status") or "unknown").strip().lower()
    if (
        core_execution_status in {"ok", "degraded", "critical", "unknown"}
        and rank.get(core_execution_status, 0) > rank.get(status, 0)
    ):
        status = core_execution_status
    ops_action = _derive_ops_next_action(obj)
    ops_priority = str(ops_action.get("priority") or "").strip().lower()
    if ops_priority == "p0" and rank.get("critical", 0) > rank.get(status, 0):
        status = "critical"
    elif ops_priority in {"p1", "p2"} and rank.get("degraded", 0) > rank.get(status, 0):
        status = "degraded"
    if bool(proxy_control.get("applied")) and rank.get("degraded", 0) > rank.get(status, 0):
        status = "degraded"
    if bool(digital_life_control.get("applied")) and rank.get("degraded", 0) > rank.get(status, 0):
        status = "degraded"

    return {
        "status": status,
        "source": "pi_cycle_event",
        "path": str(path),
        "pi_cycle_ts": obj.get("ts"),
        "guard_mode_requested": obj.get("guard_mode_requested"),
        "guard_mode_effective": obj.get("guard_mode_effective"),
        "guard_mode_base_effective": obj.get("guard_mode_base_effective"),
        "guard_mode_recommended": obj.get("guard_mode_recommended"),
        "guard_mode_recommend_reason": obj.get("guard_mode_recommend_reason"),
        "guard_mode_auto_override_applied": obj.get("guard_mode_auto_override_applied"),
        "guard_mode_use_rollback_action": obj.get("guard_mode_use_rollback_action"),
        "guard_mode_apply_rollback_action": obj.get("guard_mode_apply_rollback_action"),
        "guard_mode_use_patch_draft_action": obj.get("guard_mode_use_patch_draft_action"),
        "guard_mode_apply_patch_draft_action": obj.get("guard_mode_apply_patch_draft_action"),
        "guard_mode_use_patch_apply_action": obj.get("guard_mode_use_patch_apply_action"),
        "guard_mode_apply_patch_apply_action": obj.get("guard_mode_apply_patch_apply_action"),
        "guard_mode_use_proxy_control_action": obj.get("guard_mode_use_proxy_control_action"),
        "guard_mode_apply_proxy_control_action": obj.get("guard_mode_apply_proxy_control_action"),
        "guard_mode_proxy_control_trigger_threshold": obj.get(
            "guard_mode_proxy_control_trigger_threshold"
        ),
        "guard_mode_patch_pending_threshold": obj.get("guard_mode_patch_pending_threshold"),
        "guard_mode_patch_apply_target_batch": obj.get("guard_mode_patch_apply_target_batch"),
        "guard_mode_rollback_status": obj.get("guard_mode_rollback_status"),
        "guard_mode_rollback_reason": obj.get("guard_mode_rollback_reason"),
        "guard_mode_rollback_action_level": obj.get("guard_mode_rollback_action_level"),
        "guard_mode_rollback_action_reason": obj.get("guard_mode_rollback_action_reason"),
        "guard_mode_rollback_source_path": obj.get("guard_mode_rollback_source_path"),
        "guard_mode_rollback_report_ts": obj.get("guard_mode_rollback_report_ts"),
        "guard_mode_proxy_control_trend_status": obj.get("guard_mode_proxy_control_trend_status"),
        "guard_mode_proxy_control_trend_reason": obj.get("guard_mode_proxy_control_trend_reason"),
        "guard_mode_proxy_control_trend_action_level": obj.get(
            "guard_mode_proxy_control_trend_action_level"
        ),
        "guard_mode_proxy_control_trend_action_reason": obj.get(
            "guard_mode_proxy_control_trend_action_reason"
        ),
        "guard_mode_proxy_control_trend_trigger_count": obj.get(
            "guard_mode_proxy_control_trend_trigger_count"
        ),
        "guard_mode_proxy_control_trend_trigger_rate": obj.get(
            "guard_mode_proxy_control_trend_trigger_rate"
        ),
        "guard_mode_proxy_control_trend_dominant_trigger": obj.get(
            "guard_mode_proxy_control_trend_dominant_trigger"
        ),
        "guard_mode_proxy_control_trend_critical_trigger_count": obj.get(
            "guard_mode_proxy_control_trend_critical_trigger_count"
        ),
        "guard_mode_proxy_error_status": obj.get("guard_mode_proxy_error_status"),
        "guard_mode_proxy_error_reason": obj.get("guard_mode_proxy_error_reason"),
        "guard_mode_proxy_error_action_level": obj.get("guard_mode_proxy_error_action_level"),
        "guard_mode_proxy_error_action_reason": obj.get("guard_mode_proxy_error_action_reason"),
        "guard_mode_proxy_error_jobs_count": obj.get("guard_mode_proxy_error_jobs_count"),
        "guard_mode_proxy_error_core_jobs_count": obj.get("guard_mode_proxy_error_core_jobs_count"),
        "guard_mode_proxy_error_max_consecutive_errors": obj.get(
            "guard_mode_proxy_error_max_consecutive_errors"
        ),
        "guard_mode_proxy_error_trend_status": obj.get("guard_mode_proxy_error_trend_status"),
        "guard_mode_proxy_error_trend_reason": obj.get("guard_mode_proxy_error_trend_reason"),
        "guard_mode_proxy_error_trend_action_level": obj.get(
            "guard_mode_proxy_error_trend_action_level"
        ),
        "guard_mode_proxy_error_trend_action_reason": obj.get(
            "guard_mode_proxy_error_trend_action_reason"
        ),
        "guard_mode_proxy_error_trend_jobs_with_errors": obj.get(
            "guard_mode_proxy_error_trend_jobs_with_errors"
        ),
        "guard_mode_proxy_error_trend_core_jobs_with_errors": obj.get(
            "guard_mode_proxy_error_trend_core_jobs_with_errors"
        ),
        "guard_mode_proxy_error_trend_max_consecutive_errors": obj.get(
            "guard_mode_proxy_error_trend_max_consecutive_errors"
        ),
        "guard_mode_proxy_error_trend_rate": obj.get("guard_mode_proxy_error_trend_rate"),
        "guard_mode_patch_draft_status": obj.get("guard_mode_patch_draft_status"),
        "guard_mode_patch_draft_reason": obj.get("guard_mode_patch_draft_reason"),
        "guard_mode_patch_draft_actionable_jobs": obj.get("guard_mode_patch_draft_actionable_jobs"),
        "guard_mode_patch_draft_changed_jobs": obj.get("guard_mode_patch_draft_changed_jobs"),
        "guard_mode_patch_draft_failed_jobs": obj.get("guard_mode_patch_draft_failed_jobs"),
        "guard_mode_patch_draft_rollout_strategy": obj.get("guard_mode_patch_draft_rollout_strategy"),
        "guard_mode_patch_draft_rollout_non_empty_batches": obj.get(
            "guard_mode_patch_draft_rollout_non_empty_batches"
        ),
        "guard_mode_patch_draft_rollout_apply_order_count": obj.get(
            "guard_mode_patch_draft_rollout_apply_order_count"
        ),
        "guard_mode_patch_draft_rollout_next_batch": obj.get(
            "guard_mode_patch_draft_rollout_next_batch"
        ),
        "guard_mode_patch_draft_pending": obj.get("guard_mode_patch_draft_pending"),
        "guard_mode_patch_draft_action_level": obj.get("guard_mode_patch_draft_action_level"),
        "guard_mode_patch_draft_action_reason": obj.get("guard_mode_patch_draft_action_reason"),
        "guard_mode_patch_draft_action_hints": obj.get("guard_mode_patch_draft_action_hints"),
        "guard_mode_patch_draft_next_action": obj.get("guard_mode_patch_draft_next_action"),
        "guard_mode_patch_draft_next_action_secondary": obj.get(
            "guard_mode_patch_draft_next_action_secondary"
        ),
        "guard_mode_patch_apply_present": obj.get("guard_mode_patch_apply_present"),
        "guard_mode_patch_apply_status": obj.get("guard_mode_patch_apply_status"),
        "guard_mode_patch_apply_reason": obj.get("guard_mode_patch_apply_reason"),
        "guard_mode_patch_apply_batch_id": obj.get("guard_mode_patch_apply_batch_id"),
        "guard_mode_patch_apply_mode": obj.get("guard_mode_patch_apply_mode"),
        "guard_mode_patch_apply_applied_to_source": obj.get("guard_mode_patch_apply_applied_to_source"),
        "guard_mode_patch_apply_apply_skipped_reason": obj.get(
            "guard_mode_patch_apply_apply_skipped_reason"
        ),
        "guard_mode_patch_apply_selected_actions": obj.get("guard_mode_patch_apply_selected_actions"),
        "guard_mode_patch_apply_changed_jobs": obj.get("guard_mode_patch_apply_changed_jobs"),
        "guard_mode_patch_apply_failed_jobs": obj.get("guard_mode_patch_apply_failed_jobs"),
        "guard_mode_patch_apply_blocked_jobs": obj.get("guard_mode_patch_apply_blocked_jobs"),
        "guard_mode_patch_apply_operations_total": obj.get("guard_mode_patch_apply_operations_total"),
        "guard_mode_patch_apply_operations_applied": obj.get("guard_mode_patch_apply_operations_applied"),
        "guard_mode_patch_apply_operations_changed": obj.get("guard_mode_patch_apply_operations_changed"),
        "guard_mode_patch_apply_trend_status": obj.get("guard_mode_patch_apply_trend_status"),
        "guard_mode_patch_apply_trend_reason": obj.get("guard_mode_patch_apply_trend_reason"),
        "guard_mode_patch_apply_trend_events": obj.get("guard_mode_patch_apply_trend_events"),
        "guard_mode_patch_apply_trend_problem_events": obj.get(
            "guard_mode_patch_apply_trend_problem_events"
        ),
        "guard_mode_patch_apply_trend_problem_rate": obj.get("guard_mode_patch_apply_trend_problem_rate"),
        "guard_mode_patch_apply_trend_critical_problem_events": obj.get(
            "guard_mode_patch_apply_trend_critical_problem_events"
        ),
        "guard_mode_patch_apply_trend_critical_problem_rate": obj.get(
            "guard_mode_patch_apply_trend_critical_problem_rate"
        ),
        "guard_mode_patch_apply_trend_target_batch_id": obj.get(
            "guard_mode_patch_apply_trend_target_batch_id"
        ),
        "guard_mode_patch_draft_isolation_enabled": obj.get("guard_mode_patch_draft_isolation_enabled"),
        "guard_mode_patch_draft_isolation_candidates_total": obj.get(
            "guard_mode_patch_draft_isolation_candidates_total"
        ),
        "guard_mode_patch_draft_isolation_actionable_candidates": obj.get(
            "guard_mode_patch_draft_isolation_actionable_candidates"
        ),
        "guard_mode_patch_draft_isolation_critical_candidates": obj.get(
            "guard_mode_patch_draft_isolation_critical_candidates"
        ),
        "guard_mode_patch_draft_isolation_manual_only_candidates": obj.get(
            "guard_mode_patch_draft_isolation_manual_only_candidates"
        ),
        "guard_mode_patch_draft_isolation_rollout_strategy": obj.get(
            "guard_mode_patch_draft_isolation_rollout_strategy"
        ),
        "guard_mode_patch_draft_isolation_rollout_next_batch": obj.get(
            "guard_mode_patch_draft_isolation_rollout_next_batch"
        ),
        "guard_mode_patch_draft_isolation_rollout_pending": obj.get(
            "guard_mode_patch_draft_isolation_rollout_pending"
        ),
        "cron_policy_apply_batch_present": obj.get("cron_policy_apply_batch_present"),
        "cron_policy_apply_batch_status": obj.get("cron_policy_apply_batch_status"),
        "cron_policy_apply_batch_reason": obj.get("cron_policy_apply_batch_reason"),
        "cron_policy_apply_batch_batch_id": obj.get("cron_policy_apply_batch_batch_id"),
        "cron_policy_apply_batch_mode": obj.get("cron_policy_apply_batch_mode"),
        "cron_policy_apply_batch_applied_to_source": obj.get("cron_policy_apply_batch_applied_to_source"),
        "cron_policy_apply_batch_apply_skipped_reason": obj.get("cron_policy_apply_batch_apply_skipped_reason"),
        "cron_policy_apply_batch_selected_actions": obj.get("cron_policy_apply_batch_selected_actions"),
        "cron_policy_apply_batch_changed_jobs": obj.get("cron_policy_apply_batch_changed_jobs"),
        "cron_policy_apply_batch_failed_jobs": obj.get("cron_policy_apply_batch_failed_jobs"),
        "cron_policy_apply_batch_blocked_jobs": obj.get("cron_policy_apply_batch_blocked_jobs"),
        "cron_policy_apply_batch_operations_total": obj.get("cron_policy_apply_batch_operations_total"),
        "cron_policy_apply_batch_operations_applied": obj.get("cron_policy_apply_batch_operations_applied"),
        "cron_policy_apply_batch_operations_changed": obj.get("cron_policy_apply_batch_operations_changed"),
        "cron_policy_apply_batch_output_path": obj.get("cron_policy_apply_batch_output_path"),
        "cron_policy_apply_batch_candidate_jobs_output": obj.get(
            "cron_policy_apply_batch_candidate_jobs_output"
        ),
        "core_proxy_bypass_present": obj.get("core_proxy_bypass_present"),
        "core_proxy_bypass_status": obj.get("core_proxy_bypass_status"),
        "core_proxy_bypass_reason": obj.get("core_proxy_bypass_reason"),
        "core_proxy_bypass_source": obj.get("core_proxy_bypass_source"),
        "core_proxy_bypass_returncode": obj.get("core_proxy_bypass_returncode"),
        "core_proxy_bypass_duration_sec": obj.get("core_proxy_bypass_duration_sec"),
        "core_proxy_bypass_requests_total": obj.get("core_proxy_bypass_requests_total"),
        "core_proxy_bypass_bypass_attempted": obj.get("core_proxy_bypass_bypass_attempted"),
        "core_proxy_bypass_bypass_success": obj.get("core_proxy_bypass_bypass_success"),
        "core_proxy_bypass_bypass_failed": obj.get("core_proxy_bypass_bypass_failed"),
        "core_proxy_bypass_no_proxy_retries": obj.get("core_proxy_bypass_no_proxy_retries"),
        "core_proxy_bypass_no_proxy_retry_exhausted": obj.get("core_proxy_bypass_no_proxy_retry_exhausted"),
        "core_proxy_bypass_hint_response": obj.get("core_proxy_bypass_hint_response"),
        "core_proxy_bypass_hint_exception": obj.get("core_proxy_bypass_hint_exception"),
        "core_proxy_bypass_reason_response_hint": obj.get("core_proxy_bypass_reason_response_hint"),
        "core_proxy_bypass_reason_exception_hint": obj.get("core_proxy_bypass_reason_exception_hint"),
        "core_proxy_bypass_last_reason": obj.get("core_proxy_bypass_last_reason"),
        "core_proxy_bypass_local_requests_total": obj.get("core_proxy_bypass_local_requests_total"),
        "core_proxy_bypass_local_bypass_attempted": obj.get("core_proxy_bypass_local_bypass_attempted"),
        "core_proxy_bypass_local_bypass_success": obj.get("core_proxy_bypass_local_bypass_success"),
        "core_proxy_bypass_local_bypass_failed": obj.get("core_proxy_bypass_local_bypass_failed"),
        "core_proxy_bypass_local_no_proxy_retries": obj.get("core_proxy_bypass_local_no_proxy_retries"),
        "core_proxy_bypass_local_no_proxy_retry_exhausted": obj.get(
            "core_proxy_bypass_local_no_proxy_retry_exhausted"
        ),
        "core_proxy_bypass_exec_requests_total": obj.get("core_proxy_bypass_exec_requests_total"),
        "core_proxy_bypass_exec_bypass_attempted": obj.get("core_proxy_bypass_exec_bypass_attempted"),
        "core_proxy_bypass_exec_bypass_success": obj.get("core_proxy_bypass_exec_bypass_success"),
        "core_proxy_bypass_exec_bypass_failed": obj.get("core_proxy_bypass_exec_bypass_failed"),
        "core_proxy_bypass_exec_no_proxy_retries": obj.get("core_proxy_bypass_exec_no_proxy_retries"),
        "core_proxy_bypass_exec_no_proxy_retry_exhausted": obj.get(
            "core_proxy_bypass_exec_no_proxy_retry_exhausted"
        ),
        "core_execution_present": obj.get("core_execution_present"),
        "core_execution_status": obj.get("core_execution_status"),
        "core_execution_reason": obj.get("core_execution_reason"),
        "core_execution_source": obj.get("core_execution_source"),
        "core_execution_returncode": obj.get("core_execution_returncode"),
        "core_execution_duration_sec": obj.get("core_execution_duration_sec"),
        "core_execution_action": obj.get("core_execution_action"),
        "core_execution_decision": obj.get("core_execution_decision"),
        "core_execution_executor_attempted": obj.get("core_execution_executor_attempted"),
        "core_execution_executor_probe_requested": obj.get("core_execution_executor_probe_requested"),
        "core_execution_executor_probe_effective": obj.get("core_execution_executor_probe_effective"),
        "core_execution_executor_force_probe_on_guardrail": obj.get(
            "core_execution_executor_force_probe_on_guardrail"
        ),
        "core_execution_executor_live_requested": obj.get("core_execution_executor_live_requested"),
        "core_execution_executor_live_effective": obj.get("core_execution_executor_live_effective"),
        "core_execution_executor_cmd": obj.get("core_execution_executor_cmd"),
        "core_execution_order_http": obj.get("core_execution_order_http"),
        "core_execution_order_endpoint": obj.get("core_execution_order_endpoint"),
        "core_execution_order_decision": obj.get("core_execution_order_decision"),
        "core_execution_order_mode": obj.get("core_execution_order_mode"),
        "core_execution_order_reason": obj.get("core_execution_order_reason"),
        "core_execution_order_error": obj.get("core_execution_order_error"),
        "core_execution_order_cap_violation": obj.get("core_execution_order_cap_violation"),
        "core_execution_order_simulated": obj.get("core_execution_order_simulated"),
        "core_execution_guardrail_hit": obj.get("core_execution_guardrail_hit"),
        "core_execution_guardrail_reasons": obj.get("core_execution_guardrail_reasons"),
        "core_execution_paper_fill_gate_mode": obj.get("core_execution_paper_fill_gate_mode"),
        "core_execution_paper_fill_gate_policy": obj.get("core_execution_paper_fill_gate_policy"),
        "core_execution_paper_fill_gate_reason": obj.get("core_execution_paper_fill_gate_reason"),
        "core_execution_paper_fill_gate_cap_violation": obj.get(
            "core_execution_paper_fill_gate_cap_violation"
        ),
        "core_execution_paper_execution_attempted": obj.get(
            "core_execution_paper_execution_attempted"
        ),
        "core_execution_paper_execution_applied": obj.get(
            "core_execution_paper_execution_applied"
        ),
        "core_execution_paper_execution_route": obj.get(
            "core_execution_paper_execution_route"
        ),
        "core_execution_paper_execution_fill_px": obj.get(
            "core_execution_paper_execution_fill_px"
        ),
        "core_execution_paper_execution_signed_slippage_bps": obj.get(
            "core_execution_paper_execution_signed_slippage_bps"
        ),
        "core_execution_paper_execution_fee_rate": obj.get(
            "core_execution_paper_execution_fee_rate"
        ),
        "core_execution_paper_execution_fee_usdt": obj.get(
            "core_execution_paper_execution_fee_usdt"
        ),
        "core_execution_paper_execution_ledger_written": obj.get(
            "core_execution_paper_execution_ledger_written"
        ),
        "ops_next_action": ops_action.get("action"),
        "ops_next_action_reason": ops_action.get("reason"),
        "ops_next_action_priority": ops_action.get("priority"),
        "ops_next_action_secondary": ops_action.get("secondary"),
        "ops_next_action_context": ops_action.get("context"),
        "ops_next_action_source": ops_action.get("source"),
        "fail_steps": obj.get("fail_steps"),
        "fail_steps_critical": obj.get("fail_steps_critical"),
        "fail_steps_optional": obj.get("fail_steps_optional"),
        "fallback_steps_optional": obj.get("fallback_steps_optional"),
        "step_resilience": obj.get("step_resilience"),
        "step_retry_attempts_total": retry_attempts_total,
        "step_retried_count": retried_steps_count,
        "step_optional_fallback_count": optional_fallback_count,
        "rollout_mode_configured": gate.get("rollout_mode_configured"),
        "rollout_mode_effective": gate.get("rollout_mode_effective"),
        "window_hours": gate.get("window_hours"),
        "total_events": gate.get("total_events"),
        "act_events": gate.get("act_events"),
        "would_block_rate": gate.get("would_block_rate"),
        "cooldown_hit_rate": gate.get("cooldown_hit_rate"),
        "recover_confirm_fail_rate": gate.get("recover_confirm_fail_rate"),
        "last_event_ts": gate.get("last_event_ts"),
        "gate_status": gate_status,
        "proxy_control_enabled": proxy_control.get("enabled"),
        "proxy_control_applied": proxy_control.get("applied"),
        "proxy_control_trigger": proxy_control.get("trigger"),
        "proxy_control_reason": proxy_control.get("reason"),
        "proxy_control_expires_ts": proxy_control.get("expires_ts"),
        "proxy_control_control_path": proxy_control.get("control_path"),
        "proxy_control_patch_status": proxy_control.get("patch_status"),
        "proxy_control_patch_pending": proxy_control.get("patch_pending"),
        "proxy_control_isolation_pending": proxy_control.get("isolation_pending"),
        "proxy_control_isolation_actionable_candidates": proxy_control.get(
            "isolation_actionable_candidates"
        ),
        "proxy_control_isolation_critical_candidates": proxy_control.get(
            "isolation_critical_candidates"
        ),
        "proxy_control_isolation_next_batch": proxy_control.get("isolation_next_batch"),
        "digital_life_control_enabled": digital_life_control.get("enabled"),
        "digital_life_control_applied": digital_life_control.get("applied"),
        "digital_life_control_trigger": digital_life_control.get("trigger"),
        "digital_life_control_reason": digital_life_control.get("reason"),
        "digital_life_control_expires_ts": digital_life_control.get("expires_ts"),
        "digital_life_control_control_path": digital_life_control.get("control_path"),
        "alert_status": alert_status,
        "alert_reason": alert.get("reason"),
        "alert_breach_count": len(alert.get("breaches") or []) if isinstance(alert, dict) else 0,
        "alert_emitted": bool(emit.get("emitted")) if isinstance(emit, dict) else False,
        "alert_emit_reason": emit.get("reason") if isinstance(emit, dict) else None,
        "alert_markdown_path": emit.get("md_path") if isinstance(emit, dict) else None,
        "alert_notify_sent": bool(notify.get("sent")) if isinstance(notify, dict) else False,
        "alert_notify_reason": notify.get("reason") if isinstance(notify, dict) else None,
        "alert_guard_mode_recommended": emit.get("guard_mode_recommended") if isinstance(emit, dict) else None,
        "alert_guard_mode_recommend_reason": (
            emit.get("guard_mode_recommend_reason") if isinstance(emit, dict) else None
        ),
        "alert_guard_mode_reason_winner": emit.get("guard_mode_reason_winner") if isinstance(emit, dict) else None,
        "alert_guard_mode_priority_order": (
            emit.get("guard_mode_priority_order") if isinstance(emit, dict) else None
        ),
        "alert_guard_mode_source_recommendations": (
            emit.get("guard_mode_source_recommendations") if isinstance(emit, dict) else None
        ),
        "alert_notify_trend_status": emit.get("gate_notify_trend_status") if isinstance(emit, dict) else None,
        "alert_notify_trend_status_base": emit.get("gate_notify_trend_status_base") if isinstance(emit, dict) else None,
        "alert_notify_trend_status_source": emit.get("gate_notify_trend_status_source") if isinstance(emit, dict) else None,
        "alert_notify_trend_component_score_mode_configured": (
            emit.get("gate_notify_trend_component_score_mode_configured") if isinstance(emit, dict) else None
        ),
        "alert_notify_trend_component_score_mode_effective": (
            emit.get("gate_notify_trend_component_score_mode_effective") if isinstance(emit, dict) else None
        ),
        "alert_notify_trend_component_score_mode_source": (
            emit.get("gate_notify_trend_component_score_mode_source") if isinstance(emit, dict) else None
        ),
        "alert_notify_trend_component_score_auto_switch_enabled": (
            emit.get("gate_notify_trend_component_score_auto_switch_enabled") if isinstance(emit, dict) else None
        ),
        "alert_notify_trend_component_score_auto_recommend_enforce": (
            emit.get("gate_notify_trend_component_score_auto_recommend_enforce")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_auto_recommend_reason": (
            emit.get("gate_notify_trend_component_score_auto_recommend_reason")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_auto_stable_windows": (
            emit.get("gate_notify_trend_component_score_auto_stable_windows")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_auto_unstable_windows": (
            emit.get("gate_notify_trend_component_score_auto_unstable_windows")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_auto_promote_windows": (
            emit.get("gate_notify_trend_component_score_auto_promote_windows")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_auto_demote_windows": (
            emit.get("gate_notify_trend_component_score_auto_demote_windows")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_auto_state_path": (
            emit.get("gate_notify_trend_component_score_auto_state_path")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_auto_transition": (
            emit.get("gate_notify_trend_component_score_auto_transition")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_auto_state_io_degraded": (
            emit.get("gate_notify_trend_component_score_auto_state_io_degraded")
            if isinstance(emit, dict)
            else None
        ),
        "alert_notify_trend_component_score_shadow_would_upgrade": (
            emit.get("gate_notify_trend_component_score_shadow_would_upgrade") if isinstance(emit, dict) else None
        ),
        "alert_notify_trend_component_score_shadow_target_status": (
            emit.get("gate_notify_trend_component_score_shadow_target_status") if isinstance(emit, dict) else None
        ),
        "notify_trend_window_hours": trend.get("window_hours"),
        "notify_trend_events": trend.get("events"),
        "notify_trend_emit_events": trend.get("emit_events"),
        "notify_trend_notify_records": trend.get("notify_records"),
        "notify_trend_notify_enabled_events": trend.get("notify_enabled_events"),
        "notify_trend_attempted": trend.get("attempted"),
        "notify_trend_sent": trend.get("sent"),
        "notify_trend_failed": trend.get("failed"),
        "notify_trend_fail_rate": trend.get("fail_rate"),
        "notify_trend_success_rate": trend.get("success_rate"),
        "notify_trend_fail_status": trend.get("fail_status"),
        "notify_trend_fail_reason": trend.get("fail_reason"),
        "notify_trend_status_reason": trend.get("status_reason"),
        "notify_trend_reason": trend_reason,
        "notify_trend_suppressed": trend.get("suppressed"),
        "notify_trend_suppressed_rate": trend.get("suppressed_rate"),
        "notify_trend_suppression_status": trend.get("suppression_status"),
        "notify_trend_suppression_reason": trend.get("suppression_reason"),
        "notify_trend_suppression_reasons_top": trend.get("suppression_reasons_top"),
        "notify_trend_min_interval_escalated": trend.get("min_interval_escalated"),
        "notify_trend_min_interval_escalation_rate": trend.get("min_interval_escalation_rate"),
        "notify_trend_min_interval_status": trend.get("min_interval_status"),
        "notify_trend_min_interval_reason": trend.get("min_interval_reason"),
        "notify_trend_min_interval_sources_top": trend.get("min_interval_sources_top"),
        "notify_trend_patch_action_events": trend.get("patch_action_events"),
        "notify_trend_patch_action_escalated": trend.get("patch_action_escalated"),
        "notify_trend_patch_action_escalated_rate": trend.get("patch_action_escalated_rate"),
        "notify_trend_patch_action_pending": trend.get("patch_action_pending"),
        "notify_trend_patch_action_pending_rate": trend.get("patch_action_pending_rate"),
        "notify_trend_patch_action_status": trend.get("patch_action_status"),
        "notify_trend_patch_action_status_reason": trend.get("patch_action_status_reason"),
        "notify_trend_patch_action_levels_top": trend.get("patch_action_levels_top"),
        "notify_trend_patch_action_reasons_top": trend.get("patch_action_reasons_top"),
        "notify_trend_patch_next_actions_top": trend.get("patch_next_actions_top"),
        "notify_trend_patch_action_hints_top": trend.get("patch_action_hints_top"),
        "notify_trend_patch_dominant_next_action": trend.get("patch_dominant_next_action"),
        "notify_trend_patch_dominant_next_action_count": trend.get(
            "patch_dominant_next_action_count"
        ),
        "notify_trend_patch_dominant_next_action_share": trend.get(
            "patch_dominant_next_action_share"
        ),
        "notify_trend_patch_dominant_action_reason": trend.get("patch_dominant_action_reason"),
        "notify_trend_patch_dominant_action_reason_count": trend.get(
            "patch_dominant_action_reason_count"
        ),
        "notify_trend_patch_dominant_action_reason_share": trend.get(
            "patch_dominant_action_reason_share"
        ),
        "notify_trend_patch_recommended_next_action": trend.get("patch_recommended_next_action"),
        "notify_trend_patch_recommended_reason": trend.get("patch_recommended_reason"),
        "notify_trend_patch_recommended_confidence": trend.get("patch_recommended_confidence"),
        "notify_trend_patch_apply_events": trend.get("patch_apply_events"),
        "notify_trend_patch_apply_problem_events": trend.get("patch_apply_problem_events"),
        "notify_trend_patch_apply_problem_rate": trend.get("patch_apply_problem_rate"),
        "notify_trend_patch_apply_critical_problem_events": trend.get(
            "patch_apply_critical_problem_events"
        ),
        "notify_trend_patch_apply_critical_problem_rate": trend.get(
            "patch_apply_critical_problem_rate"
        ),
        "notify_trend_patch_apply_status": trend.get("patch_apply_status"),
        "notify_trend_patch_apply_status_reason": trend.get("patch_apply_status_reason"),
        "notify_trend_patch_apply_statuses_top": trend.get("patch_apply_statuses_top"),
        "notify_trend_patch_apply_reasons_top": trend.get("patch_apply_reasons_top"),
        "notify_trend_patch_apply_modes_top": trend.get("patch_apply_modes_top"),
        "notify_trend_patch_apply_target_batch_id": trend.get("patch_apply_target_batch_id"),
        "notify_trend_patch_apply_dominant_reason": trend.get("patch_apply_dominant_reason"),
        "notify_trend_patch_apply_dominant_reason_count": trend.get(
            "patch_apply_dominant_reason_count"
        ),
        "notify_trend_patch_apply_dominant_reason_share": trend.get(
            "patch_apply_dominant_reason_share"
        ),
        "notify_trend_patch_apply_thresholds": (
            (trend.get("thresholds") or {}).get("patch_apply")
            if isinstance(trend.get("thresholds"), dict)
            else None
        ),
        "notify_trend_patch_thresholds": (
            (trend.get("thresholds") or {}).get("patch_action")
            if isinstance(trend.get("thresholds"), dict)
            else None
        ),
        "notify_trend_component_known": trend.get("component_known"),
        "notify_trend_component_score": trend.get("component_score"),
        "notify_trend_component_score_status": trend.get("component_score_status"),
        "notify_trend_component_score_reason": trend.get("component_score_reason"),
        "notify_trend_component_weights": trend.get("component_weights"),
        "notify_trend_component_thresholds": (
            (trend.get("thresholds") or {}).get("component_score")
            if isinstance(trend.get("thresholds"), dict)
            else None
        ),
        "notify_trend_component_score_shadow_mode_events": trend.get("component_score_shadow_mode_events"),
        "notify_trend_component_score_shadow_would_upgrade": trend.get("component_score_shadow_would_upgrade"),
        "notify_trend_component_score_shadow_would_upgrade_rate": trend.get(
            "component_score_shadow_would_upgrade_rate"
        ),
        "notify_trend_component_score_shadow_critical_target_count": trend.get(
            "component_score_shadow_critical_target_count"
        ),
        "notify_trend_component_score_shadow_critical_target_share": trend.get(
            "component_score_shadow_critical_target_share"
        ),
        "notify_trend_component_score_shadow_target_status_top": trend.get(
            "component_score_shadow_target_status_top"
        ),
        "notify_trend_component_score_shadow_recommend_enforce": trend.get(
            "component_score_shadow_recommend_enforce"
        ),
        "notify_trend_component_score_shadow_recommend_reason": trend.get(
            "component_score_shadow_recommend_reason"
        ),
        "notify_trend_component_score_shadow_rollout_thresholds": (
            (trend.get("thresholds") or {}).get("component_score_shadow_rollout")
            if isinstance(trend.get("thresholds"), dict)
            else None
        ),
        "notify_trend_component_score_mode_drift_status": drift_signal.get("status"),
        "notify_trend_component_score_mode_drift_reason": drift_signal.get("reason"),
        "notify_trend_component_score_mode_drift_window_hours": drift_signal.get("window_hours"),
        "notify_trend_component_score_mode_drift_mode_events": drift_signal.get("mode_events"),
        "notify_trend_component_score_mode_drift_events": drift_signal.get("drift_events"),
        "notify_trend_component_score_mode_drift_rate": drift_signal.get("drift_rate"),
        "notify_trend_component_score_mode_drift_latest_consecutive": drift_signal.get(
            "latest_consecutive_drift"
        ),
        "notify_trend_component_score_mode_drift_latest_drift": drift_signal.get("latest_drift"),
        "notify_trend_component_score_mode_drift_top_pairs": drift_signal.get("top_drift_pairs"),
        "notify_trend_component_score_mode_drift_thresholds": drift_signal.get("thresholds"),
        "rollback_enabled": rollback.get("enabled") if isinstance(rollback, dict) else None,
        "rollback_applied": rollback.get("applied") if isinstance(rollback, dict) else None,
        "rollback_trigger": rollback.get("trigger") if isinstance(rollback, dict) else None,
        "rollback_reason": rollback.get("reason") if isinstance(rollback, dict) else None,
        "rollback_expires_ts": rollback.get("expires_ts") if isinstance(rollback, dict) else None,
        "rollback_control_path": rollback.get("control_path") if isinstance(rollback, dict) else None,
        "rollback_component_mode_drift_level": (
            rollback.get("component_mode_drift_level") if isinstance(rollback, dict) else None
        ),
        "rollback_component_mode_drift_min_events": (
            rollback.get("component_mode_drift_min_events") if isinstance(rollback, dict) else None
        ),
        "rollback_component_mode_drift_min_consecutive": (
            rollback.get("component_mode_drift_min_consecutive") if isinstance(rollback, dict) else None
        ),
        "rollback_trend_status": rollback_signal.get("status"),
        "rollback_trend_reason": rollback_signal.get("reason"),
        "rollback_trend_window_hours": rollback_signal.get("window_hours"),
        "rollback_trend_pi_cycle_events": rollback_signal.get("pi_cycle_events"),
        "rollback_trend_records": rollback_signal.get("rollback_records"),
        "rollback_trend_enabled_count": rollback_signal.get("enabled_count"),
        "rollback_trend_applied_count": rollback_signal.get("applied_count"),
        "rollback_trend_trigger_count": rollback_signal.get("trigger_count"),
        "rollback_trend_trigger_rate": rollback_signal.get("trigger_rate"),
        "rollback_trend_trigger_counts": rollback_signal.get("trigger_counts"),
        "rollback_trend_trigger_shares": rollback_signal.get("trigger_shares"),
        "rollback_trend_dominant_trigger": rollback_signal.get("dominant_trigger"),
        "rollback_trend_dominant_trigger_count": rollback_signal.get("dominant_trigger_count"),
        "rollback_trend_dominant_trigger_share": rollback_signal.get("dominant_trigger_share"),
        "rollback_trend_action_level": rollback_signal.get("action_level"),
        "rollback_trend_action_reason": rollback_signal.get("action_reason"),
        "rollback_trend_action_recommendations": rollback_signal.get("action_recommendations"),
        "rollback_trend_top_triggers": rollback_signal.get("top_triggers"),
        "rollback_trend_action_hints": rollback_signal.get("action_hints"),
        "rollback_trend_latest_trigger": rollback_signal.get("latest_trigger"),
        "rollback_trend_latest_reason": rollback_signal.get("latest_reason"),
        "rollback_trend_latest_applied": rollback_signal.get("latest_applied"),
        "rollback_trend_latest_ts": rollback_signal.get("latest_ts"),
        "rollback_trend_thresholds": rollback_signal.get("thresholds"),
        "proxy_control_trend_status": proxy_control_signal.get("status"),
        "proxy_control_trend_reason": proxy_control_signal.get("reason"),
        "proxy_control_trend_window_hours": proxy_control_signal.get("window_hours"),
        "proxy_control_trend_pi_cycle_events": proxy_control_signal.get("pi_cycle_events"),
        "proxy_control_trend_records": proxy_control_signal.get("proxy_control_records"),
        "proxy_control_trend_enabled_count": proxy_control_signal.get("enabled_count"),
        "proxy_control_trend_applied_count": proxy_control_signal.get("applied_count"),
        "proxy_control_trend_trigger_count": proxy_control_signal.get("trigger_count"),
        "proxy_control_trend_trigger_rate": proxy_control_signal.get("trigger_rate"),
        "proxy_control_trend_trigger_counts": proxy_control_signal.get("trigger_counts"),
        "proxy_control_trend_trigger_shares": proxy_control_signal.get("trigger_shares"),
        "proxy_control_trend_dominant_trigger": proxy_control_signal.get("dominant_trigger"),
        "proxy_control_trend_dominant_trigger_count": proxy_control_signal.get("dominant_trigger_count"),
        "proxy_control_trend_dominant_trigger_share": proxy_control_signal.get("dominant_trigger_share"),
        "proxy_control_trend_action_level": proxy_control_signal.get("action_level"),
        "proxy_control_trend_action_reason": proxy_control_signal.get("action_reason"),
        "proxy_control_trend_action_recommendations": proxy_control_signal.get("action_recommendations"),
        "proxy_control_trend_top_triggers": proxy_control_signal.get("top_triggers"),
        "proxy_control_trend_action_hints": proxy_control_signal.get("action_hints"),
        "proxy_control_trend_latest_trigger": proxy_control_signal.get("latest_trigger"),
        "proxy_control_trend_latest_reason": proxy_control_signal.get("latest_reason"),
        "proxy_control_trend_latest_applied": proxy_control_signal.get("latest_applied"),
        "proxy_control_trend_latest_ts": proxy_control_signal.get("latest_ts"),
        "proxy_control_trend_thresholds": proxy_control_signal.get("thresholds"),
        "notify_trend_status": notify_trend_status,
        "notify_trend_thresholds": trend.get("thresholds"),
        "notify_trend_failure_reasons_top": trend.get("failure_reasons_top"),
        "notify_trend_skip_reasons_top": trend.get("skip_reasons_top"),
    }


def _probe_openclaw_version() -> Dict[str, Any]:
    probe = _run_json_cmd(["openclaw", "--version"], timeout_sec=5)
    if probe.get("ok"):
        payload = probe.get("payload")
        return {
            "status": "ok",
            "version": payload,
            "duration_sec": probe.get("duration_sec"),
        }
    # --version is plain text, not JSON; fallback to plain execution parse.
    started = time.time()
    try:
        p = subprocess.run(["openclaw", "--version"], capture_output=True, text=True, timeout=5)
        if p.returncode == 0:
            text = (p.stdout or "").splitlines()
            version = text[-1].strip() if text else ""
            if version:
                return {
                    "status": "ok",
                    "version": version,
                    "duration_sec": round(time.time() - started, 3),
                }
    except Exception:
        pass
    return {"status": "degraded", "version": None}


def _probe_memory_status() -> Dict[str, Any]:
    probe = _run_json_cmd(["openclaw", "memory", "status", "--json"], timeout_sec=15)
    if not probe.get("ok"):
        return {
            "status": "degraded",
            "ok": False,
            "reason": probe.get("reason"),
            "duration_sec": probe.get("duration_sec"),
        }

    payload = probe.get("payload")
    if not isinstance(payload, list):
        return {
            "status": "degraded",
            "ok": False,
            "reason": "unexpected_payload",
            "duration_sec": probe.get("duration_sec"),
        }

    records = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        aid = str(entry.get("agentId") or "")
        st = entry.get("status") or {}
        scan = entry.get("scan") or {}
        files = int((st.get("files") or 0)) if isinstance(st, dict) else 0
        fts_ok = bool((st.get("fts") or {}).get("available")) if isinstance(st, dict) else False
        vector_ok = bool((st.get("vector") or {}).get("available")) if isinstance(st, dict) else False
        issues = len((scan.get("issues") or [])) if isinstance(scan, dict) else 0
        records.append(
            {
                "agentId": aid,
                "files": files,
                "fts_available": fts_ok,
                "vector_available": vector_ok,
                "scan_issues": issues,
            }
        )

    unhealthy = [r for r in records if (not r.get("fts_available")) or int(r.get("scan_issues") or 0) > 0]
    status = "ok" if not unhealthy else "degraded"
    return {
        "status": status,
        "ok": True,
        "duration_sec": probe.get("duration_sec"),
        "agents": records,
        "unhealthy_agents": unhealthy,
    }


def _probe_memory_fallback_local() -> Dict[str, Any]:
    started = time.time()
    root = Path(
        os.getenv("TRADER_WORKSPACE_ROOT", str(Path(__file__).resolve().parents[1]))
    )
    query = os.getenv("MEMORY_FALLBACK_QUERY", "Long-term memory")
    limit = int(os.getenv("MEMORY_FALLBACK_LIMIT", "5"))
    try:
        out = search_memory(query=query, root=root, limit=max(1, limit))
        hits = int(out.get("hit_count") or 0)
        return {
            "status": "ok" if hits > 0 else "degraded",
            "ok": hits > 0,
            "query": query,
            "root": str(root),
            "backend": out.get("backend"),
            "hit_count": hits,
            "duration_sec": round(time.time() - started, 3),
        }
    except Exception as e:
        return {
            "status": "degraded",
            "ok": False,
            "query": query,
            "root": str(root),
            "reason": f"fallback_error:{type(e).__name__}",
            "duration_sec": round(time.time() - started, 3),
        }


def _probe_memory_retrieval() -> Dict[str, Any]:
    primary = _probe_memory_status()
    if str(primary.get("status") or "") == "ok":
        primary["mode"] = "openclaw_memory_status"
        return primary

    fallback = _probe_memory_fallback_local()
    if str(fallback.get("status") or "") == "ok":
        return {
            "status": "ok",
            "ok": True,
            "mode": "fallback_local_search",
            "primary": {
                "status": primary.get("status"),
                "reason": primary.get("reason"),
                "duration_sec": primary.get("duration_sec"),
            },
            "fallback": fallback,
        }

    return {
        "status": "degraded",
        "ok": False,
        "mode": "fallback_failed",
        "primary": primary,
        "fallback": fallback,
    }


def _probe_cron_scheduler() -> Dict[str, Any]:
    probe = _run_json_cmd(["openclaw", "cron", "status", "--json"], timeout_sec=8)
    if not probe.get("ok"):
        return {
            "status": "degraded",
            "ok": False,
            "reason": probe.get("reason"),
            "duration_sec": probe.get("duration_sec"),
        }
    payload = probe.get("payload")
    if not isinstance(payload, dict):
        return {
            "status": "degraded",
            "ok": False,
            "reason": "unexpected_payload",
            "duration_sec": probe.get("duration_sec"),
        }
    enabled = bool(payload.get("enabled", False))
    return {
        "status": "ok" if enabled else "degraded",
        "ok": True,
        "duration_sec": probe.get("duration_sec"),
        "enabled": enabled,
        "jobs": int(payload.get("jobs") or 0),
        "next_wake_at_ms": payload.get("nextWakeAtMs"),
    }


def _default_cron_policy_patch_draft_path() -> Path:
    d = dt.date.today().isoformat()
    return SYSTEM_ROOT / "output" / "review" / f"{d}_cron_policy_patch_draft.json"


def _probe_cron_policy_patch_draft(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path
    if p is None:
        env = str(os.getenv("PI_CRON_POLICY_PATCH_DRAFT_PATH", "")).strip()
        p = Path(env).expanduser().resolve() if env else _default_cron_policy_patch_draft_path().resolve()
    if not p.exists():
        return {
            "status": "unknown",
            "reason": "patch_draft_missing",
            "path": str(p),
            "exists": False,
            "draft_status_raw": None,
            "actionable_jobs": 0,
            "changed_jobs": 0,
            "failed_jobs": 0,
            "operation_total": 0,
            "operation_applied_total": 0,
            "rollout_strategy": None,
            "rollout_batch_count": 0,
            "rollout_non_empty_batches": 0,
            "rollout_apply_order_count": 0,
            "rollout_batches_top": [],
            "rollout_next_batch": None,
            "pending": False,
            "action_level": "observe",
            "action_reason": "patch_draft_missing",
            "action_hints": [],
            "next_action": "wait_for_patch_draft",
        }
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status": "degraded",
            "reason": f"parse_error:{type(e).__name__}",
            "path": str(p),
            "exists": True,
            "draft_status_raw": None,
            "actionable_jobs": 0,
            "changed_jobs": 0,
            "failed_jobs": 0,
            "operation_total": 0,
            "operation_applied_total": 0,
            "rollout_strategy": None,
            "rollout_batch_count": 0,
            "rollout_non_empty_batches": 0,
            "rollout_apply_order_count": 0,
            "rollout_batches_top": [],
            "rollout_next_batch": None,
            "pending": False,
            "action_level": "degrade",
            "action_reason": "patch_draft_unreadable",
            "action_hints": ["patch: draft file unreadable; regenerate patch draft before rollout."],
            "next_action": "regenerate_patch_draft",
        }
    if not isinstance(obj, dict):
        return {
            "status": "degraded",
            "reason": "invalid_payload",
            "path": str(p),
            "exists": True,
            "draft_status_raw": None,
            "actionable_jobs": 0,
            "changed_jobs": 0,
            "failed_jobs": 0,
            "operation_total": 0,
            "operation_applied_total": 0,
            "rollout_strategy": None,
            "rollout_batch_count": 0,
            "rollout_non_empty_batches": 0,
            "rollout_apply_order_count": 0,
            "rollout_batches_top": [],
            "rollout_next_batch": None,
            "pending": False,
            "action_level": "degrade",
            "action_reason": "patch_draft_invalid_payload",
            "action_hints": ["patch: invalid draft payload; rebuild draft from cron_policy remediation."],
            "next_action": "rebuild_patch_draft",
        }

    summary = obj.get("summary") if isinstance(obj.get("summary"), dict) else {}
    rollout = obj.get("rollout_plan") if isinstance(obj.get("rollout_plan"), dict) else {}
    isolation = obj.get("isolation_plan") if isinstance(obj.get("isolation_plan"), dict) else {}
    isolation_rollout = (
        obj.get("isolation_rollout") if isinstance(obj.get("isolation_rollout"), dict) else {}
    )
    isolation_summary = (
        isolation.get("summary") if isinstance(isolation.get("summary"), dict) else {}
    )
    isolation_recs = (
        isolation.get("recommendations") if isinstance(isolation.get("recommendations"), list) else []
    )
    batches = rollout.get("batches") if isinstance(rollout.get("batches"), list) else []
    apply_order = rollout.get("apply_order") if isinstance(rollout.get("apply_order"), list) else []

    draft_status_raw = str(obj.get("status") or "").strip().lower()
    if draft_status_raw in {"ok", "noop"}:
        status = "ok"
    elif draft_status_raw == "degraded":
        status = "degraded"
    elif draft_status_raw == "critical":
        status = "critical"
    else:
        status = "unknown"

    batches_top: List[Dict[str, Any]] = []
    for item in batches[:5]:
        if not isinstance(item, dict):
            continue
        batches_top.append(
            {
                "id": item.get("id"),
                "count": int(item.get("count") or 0),
                "ops": int(item.get("ops") or 0),
            }
        )
    rollout_next_batch = None
    for item in batches_top:
        if not isinstance(item, dict):
            continue
        if int(item.get("count") or 0) > 0 and int(item.get("ops") or 0) > 0:
            rollout_next_batch = str(item.get("id") or "") or None
            break

    actionable_jobs = int(summary.get("actionable_jobs") or 0)
    changed_jobs = int(summary.get("changed_jobs") or 0)
    failed_jobs = int(summary.get("failed_jobs") or 0)
    isolation_candidates_total = int(isolation_summary.get("candidates_total") or 0)
    isolation_actionable_candidates = int(isolation_summary.get("actionable_candidates") or 0)
    isolation_critical_candidates = int(isolation_summary.get("critical_candidates") or 0)
    isolation_manual_only_candidates = int(isolation_summary.get("manual_only_candidates") or 0)
    isolation_enabled = bool(isolation.get("enabled", False))
    isolation_rollout_strategy = (
        str(isolation_rollout.get("strategy") or "").strip() or "proxy_isolation_then_canary_core_guarded"
    )
    isolation_rollout_next_batch = str(isolation_rollout.get("next_batch") or "").strip() or None
    isolation_rollout_pending = bool(isolation_rollout.get("pending"))
    isolation_candidates_top: List[Dict[str, Any]] = []
    for rec in isolation_recs[:5]:
        if not isinstance(rec, dict):
            continue
        isolation_candidates_top.append(
            {
                "id": rec.get("id"),
                "name": rec.get("name"),
                "severity": rec.get("severity"),
                "reason": rec.get("reason"),
                "actionable": bool(rec.get("actionable")),
            }
        )
    rollout_non_empty_batches = int(rollout.get("non_empty_batches") or 0)
    pending = bool(actionable_jobs > 0 or rollout_non_empty_batches > 0 or isolation_rollout_pending)
    action_level = "observe"
    action_reason = "no_pending_changes"
    action_hints: List[str] = []
    next_action = "none"
    next_action_secondary = None
    if status == "critical":
        action_level = "degrade"
        action_reason = "patch_status_critical"
        action_hints.append("patch: freeze rollout and inspect critical draft status.")
        next_action = "inspect_patch_status_critical"
    elif isolation_critical_candidates > 0 and isolation_actionable_candidates > 0:
        action_level = "shadow_lock"
        action_reason = "proxy_isolation_critical_candidates"
        action_hints.append(
            f"proxy isolation: critical non-core candidates={isolation_critical_candidates}; apply isolation ops before rollout."
        )
        next_action = "run_proxy_isolation_batch"
    elif isolation_actionable_candidates > 0:
        action_level = "degrade"
        action_reason = "proxy_isolation_candidates_pending"
        action_hints.append(
            f"proxy isolation: actionable candidates={isolation_actionable_candidates}; apply isolation/de-throttle first."
        )
        next_action = "run_proxy_isolation_batch"
    elif failed_jobs > 0:
        action_level = "degrade"
        action_reason = "patch_failed_jobs"
        action_hints.append(f"patch: resolve failed jobs first (failed_jobs={failed_jobs}).")
        next_action = "resolve_patch_failures"
    elif pending and status == "degraded":
        action_level = "degrade"
        action_reason = "patch_pending_degraded"
        action_hints.append("patch: run guarded rollout; keep core batch gated until canary passes.")
        next_action = "run_guarded_patch_rollout"
    elif pending:
        action_level = "observe"
        action_reason = "patch_pending"
        action_hints.append("patch: continue pending rollout in canary->high->core order.")
        next_action = "continue_patch_rollout"

    isolation_batch = isolation_rollout_next_batch
    if not isolation_batch and isolation_actionable_candidates > 0:
        isolation_batch = "batch_0_proxy_isolation"
    if isolation_batch and next_action == "run_proxy_isolation_batch":
        next_action = f"run_batch:{isolation_batch}"
        action_hints.append(f"proxy isolation: execute `{isolation_batch}` before rollout batches.")

    if isolation_enabled and isolation_manual_only_candidates > 0:
        action_hints.append(
            f"proxy isolation: manual-only candidates={isolation_manual_only_candidates}; keep operator review in loop."
        )
    if rollout_next_batch:
        if isolation_actionable_candidates > 0:
            next_action_secondary = f"run_batch:{rollout_next_batch}"
            action_hints.append(
                f"patch: after isolation batch, continue with `{rollout_next_batch}`."
            )
        else:
            action_hints.append(f"patch: next batch `{rollout_next_batch}` should run first.")
            next_action = f"run_batch:{rollout_next_batch}"
    rollout_strategy = rollout.get("strategy")
    if rollout_strategy:
        action_hints.append(f"patch: follow rollout strategy `{rollout_strategy}`.")

    def _append_batch_apply_hints(action_text: Optional[str], *, suffix: str = "") -> None:
        raw = str(action_text or "").strip()
        if not raw.startswith("run_batch:"):
            return
        batch = raw.split(":", 1)[1].strip()
        if not batch:
            return
        suffix_text = f" {suffix.strip()}" if str(suffix or "").strip() else ""
        action_hints.append(
            f"patch:{suffix_text} dry-run `python3 scripts/cron_policy_apply_batch.py --batch-id {batch}`."
        )
        action_hints.append(
            f"patch:{suffix_text} apply `python3 scripts/cron_policy_apply_batch.py --batch-id {batch} --apply` after review."
        )

    _append_batch_apply_hints(next_action, suffix="(primary)")
    _append_batch_apply_hints(next_action_secondary, suffix="(secondary)")

    return {
        "status": status,
        "reason": "ok" if status != "unknown" else "unknown_status",
        "path": str(p),
        "exists": True,
        "ts": obj.get("ts"),
        "draft_status_raw": draft_status_raw or None,
        "actionable_jobs": actionable_jobs,
        "changed_jobs": changed_jobs,
        "failed_jobs": failed_jobs,
        "isolation_enabled": isolation_enabled,
        "isolation_candidates_total": isolation_candidates_total,
        "isolation_actionable_candidates": isolation_actionable_candidates,
        "isolation_critical_candidates": isolation_critical_candidates,
        "isolation_manual_only_candidates": isolation_manual_only_candidates,
        "isolation_candidates_top": isolation_candidates_top,
        "isolation_rollout_strategy": isolation_rollout_strategy,
        "isolation_rollout_next_batch": isolation_batch,
        "isolation_rollout_pending": isolation_rollout_pending,
        "isolation_thresholds": (
            isolation.get("thresholds") if isinstance(isolation.get("thresholds"), dict) else {}
        ),
        "operation_total": int(summary.get("operation_total") or 0),
        "operation_applied_total": int(summary.get("operation_applied_total") or 0),
        "rollout_strategy": rollout_strategy,
        "rollout_batch_count": int(rollout.get("batch_count") or 0),
        "rollout_non_empty_batches": rollout_non_empty_batches,
        "rollout_apply_order_count": len(apply_order),
        "rollout_batches_top": batches_top,
        "rollout_next_batch": rollout_next_batch,
        "pending": pending,
        "action_level": action_level,
        "action_reason": action_reason,
        "action_hints": action_hints[:8],
        "next_action": next_action,
        "next_action_secondary": next_action_secondary,
    }


def _build_self_checks(
    enabled_jobs: List[Dict[str, Any]],
    all_jobs: List[Dict[str, Any]],
    config_obj: Optional[Dict[str, Any]],
    runs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    version = _probe_openclaw_version()
    scheduler = _probe_cron_scheduler()
    backlog = _build_cron_backlog(enabled_jobs=enabled_jobs)
    proxy_error_trend = _build_proxy_error_trend_audit(enabled_jobs=enabled_jobs, runs_dir=runs_dir)
    proxy_recent_job_ids = {
        str(x)
        for x in (proxy_error_trend.get("recent_proxy_job_ids") or [])
        if str(x).strip()
    }
    policy = _build_cron_policy_audit(
        enabled_jobs=enabled_jobs,
        proxy_error_recent_job_ids=proxy_recent_job_ids,
    )
    proxy_error = _build_proxy_error_audit(enabled_jobs=enabled_jobs)
    signal_registry = probe_signal_registry()
    tools = _build_web_tool_availability(jobs=all_jobs, config_obj=config_obj)
    memory = _probe_memory_retrieval()
    gate_rollout = _probe_gate_rollout()
    patch_draft = _probe_cron_policy_patch_draft()
    status = "ok"
    if any(
        str(x.get("status") or "") in {"degraded", "critical"}
        for x in [
            version,
            scheduler,
            backlog,
            policy,
            proxy_error,
            proxy_error_trend,
            signal_registry,
            tools,
            memory,
            gate_rollout,
            patch_draft,
        ]
    ):
        status = "degraded"
    if any(
        str(x.get("status") or "") == "critical"
        for x in [
            version,
            scheduler,
            backlog,
            policy,
            proxy_error,
            proxy_error_trend,
            signal_registry,
            tools,
            memory,
            gate_rollout,
            patch_draft,
        ]
    ):
        status = "critical"
    return {
        "status": status,
        "openclaw_version": version,
        "cron_scheduler": scheduler,
        "cron_backlog": backlog,
        "cron_policy": policy,
        "proxy_error": proxy_error,
        "proxy_error_trend": proxy_error_trend,
        "signal_registry": signal_registry,
        "tool_availability": {"web_search_web_fetch": tools},
        "memory_retrieval": memory,
        "gate_rollout": gate_rollout,
        "cron_policy_patch_draft": patch_draft,
    }


def _build_report(
    jobs_obj: Dict[str, Any],
    runs_dir: Optional[Path] = None,
    known_agents: Optional[List[str]] = None,
    config_obj: Optional[Dict[str, Any]] = None,
    self_checks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    jobs = jobs_obj.get("jobs") or []

    enabled = [j for j in jobs if bool(j.get("enabled", True))]
    disabled = [j for j in jobs if not bool(j.get("enabled", True))]

    enabled_ok = 0
    enabled_error = 0
    by_error_class: Dict[str, int] = {}
    worst: List[Dict[str, Any]] = []
    core_enabled = [j for j in enabled if str(j.get("name") or "") in CORE_JOB_NAMES]
    scope_jobs = core_enabled if core_enabled else enabled

    for j in enabled:
        st = j.get("state") or {}
        last_status = str(st.get("lastStatus") or "").lower()
        if last_status == "ok":
            enabled_ok += 1
        elif last_status == "error":
            enabled_error += 1
        err = str(st.get("lastError") or "")
        cls = _classify_error(err)
        by_error_class[cls] = by_error_class.get(cls, 0) + 1

        ce = int(st.get("consecutiveErrors") or 0)
        if ce >= 2 or last_status == "error":
            worst.append(
                {
                    "id": j.get("id"),
                    "name": j.get("name"),
                    "agentId": j.get("agentId"),
                    "lastStatus": last_status or None,
                    "consecutiveErrors": ce,
                    "errorClass": cls,
                    "lastError": err[:220] if err else "",
                }
            )

    worst.sort(
        key=lambda x: (int(x.get("consecutiveErrors") or 0), 1 if x.get("lastStatus") == "error" else 0),
        reverse=True,
    )

    # status_all: full fleet; status: scope-based (core if available)
    enabled_error_all = 0
    for j in enabled:
        st = j.get("state") or {}
        if str(st.get("lastStatus") or "").lower() == "error":
            enabled_error_all += 1

    scope_enabled_ok = 0
    scope_enabled_error = 0
    for j in scope_jobs:
        st = j.get("state") or {}
        if str(st.get("lastStatus") or "").lower() == "ok":
            scope_enabled_ok += 1
        elif str(st.get("lastStatus") or "").lower() == "error":
            scope_enabled_error += 1

    health_status_all = "ok"
    if enabled_error_all > 0:
        health_status_all = "degraded"
    if enabled_error_all >= 4:
        health_status_all = "critical"

    health_status_scope = "ok"
    if scope_enabled_error > 0:
        health_status_scope = "degraded"
    if scope_enabled_error >= 2:
        health_status_scope = "critical"

    lane_consistency = _build_lane_consistency(enabled_jobs=enabled, runs_dir=runs_dir)
    agent_registry = _build_agent_registry(jobs=jobs, known_agents=known_agents)
    checks = self_checks or _build_self_checks(
        enabled_jobs=enabled,
        all_jobs=jobs,
        config_obj=config_obj,
        runs_dir=runs_dir,
    )
    status_self_checks = str(checks.get("status") or "unknown").strip().lower()
    if status_self_checks not in {"ok", "degraded", "critical", "unknown"}:
        status_self_checks = "unknown"
    rank = {"unknown": 0, "ok": 1, "degraded": 2, "critical": 3}
    status_guarded = health_status_scope
    if rank.get(status_self_checks, 0) > rank.get(status_guarded, 0):
        status_guarded = status_self_checks

    return {
        "envelope_version": "1.0",
        "domain": "cron_health",
        "ts": now,
        "status": health_status_scope,
        "status_all": health_status_all,
        "status_self_checks": status_self_checks,
        "status_guarded": status_guarded,
        "scope": "core" if core_enabled else "all",
        "summary": {
            "total_jobs": len(jobs),
            "enabled_jobs": len(enabled),
            "disabled_jobs": len(disabled),
            "enabled_ok": enabled_ok,
            "enabled_error": enabled_error,
            "error_class_count": by_error_class,
        },
        "summary_scope": {
            "scope_jobs": len(scope_jobs),
            "scope_enabled_ok": scope_enabled_ok,
            "scope_enabled_error": scope_enabled_error,
        },
        "self_checks": checks,
        "agent_registry": agent_registry,
        "lane_consistency": lane_consistency,
        "worst_jobs": worst[:10],
    }


def _default_output_path() -> Path:
    d = dt.date.today().isoformat()
    return SYSTEM_ROOT / "output" / "review" / f"{d}_pi_cron_health.json"


def _default_markdown_path() -> Path:
    d = dt.date.today().isoformat()
    return SYSTEM_ROOT / "output" / "review" / f"{d}_pi_cron_health.md"


def _build_markdown(report: Dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    scope = report.get("summary_scope") if isinstance(report.get("summary_scope"), dict) else {}
    checks = report.get("self_checks") if isinstance(report.get("self_checks"), dict) else {}
    cron_policy = checks.get("cron_policy") if isinstance(checks.get("cron_policy"), dict) else {}
    proxy_error = checks.get("proxy_error") if isinstance(checks.get("proxy_error"), dict) else {}
    proxy_error_trend = checks.get("proxy_error_trend") if isinstance(checks.get("proxy_error_trend"), dict) else {}
    cron_backlog = checks.get("cron_backlog") if isinstance(checks.get("cron_backlog"), dict) else {}
    memory = checks.get("memory_retrieval") if isinstance(checks.get("memory_retrieval"), dict) else {}
    scheduler = checks.get("cron_scheduler") if isinstance(checks.get("cron_scheduler"), dict) else {}
    signal_registry = checks.get("signal_registry") if isinstance(checks.get("signal_registry"), dict) else {}
    patch_draft = checks.get("cron_policy_patch_draft") if isinstance(checks.get("cron_policy_patch_draft"), dict) else {}
    gate = checks.get("gate_rollout") if isinstance(checks.get("gate_rollout"), dict) else {}

    lines: List[str] = []
    lines.append(f"# Pi Cron Health ({report.get('ts')})")
    lines.append("")
    lines.append(f"- status(core_scope): `{report.get('status')}`")
    lines.append(f"- status_all: `{report.get('status_all')}`")
    lines.append(f"- status_self_checks: `{report.get('status_self_checks')}`")
    lines.append(f"- status_guarded: `{report.get('status_guarded')}`")
    lines.append(f"- scope: `{report.get('scope')}`")
    lines.append("")
    lines.append("## Fleet Summary")
    lines.append("")
    lines.append(
        f"- jobs(total/enabled/disabled): `{summary.get('total_jobs')}` / `{summary.get('enabled_jobs')}` / `{summary.get('disabled_jobs')}`"
    )
    lines.append(f"- enabled(ok/error): `{summary.get('enabled_ok')}` / `{summary.get('enabled_error')}`")
    lines.append(
        f"- scope(ok/error): `{scope.get('scope_enabled_ok')}` / `{scope.get('scope_enabled_error')}`"
    )
    lines.append("")
    lines.append("## Self Checks")
    lines.append("")
    if checks:
        signal_invalid = signal_registry.get("invalid_count")
        if signal_invalid is None:
            errs = signal_registry.get("errors") if isinstance(signal_registry.get("errors"), list) else []
            signal_invalid = len(errs)
        signal_missing = signal_registry.get("missing_count")
        if signal_missing is None:
            signal_missing = 0
        lines.append(
            f"- scheduler(status/enabled/jobs): `{scheduler.get('status')}` / "
            f"`{scheduler.get('enabled')}` / `{scheduler.get('jobs')}`"
        )
        lines.append(
            f"- backlog(status/overdue/max_sec): `{cron_backlog.get('status')}` / "
            f"`{cron_backlog.get('overdue_count')}` / `{cron_backlog.get('max_overdue_sec')}`"
        )
        lines.append(
            f"- memory(status/mode): `{memory.get('status')}` / `{memory.get('mode')}`"
        )
        lines.append(
            f"- signal_registry(status/invalid/missing): `{signal_registry.get('status')}` / "
            f"`{signal_invalid}` / `{signal_missing}`"
        )
        lines.append(
            f"- cron_policy(status/reason): `{cron_policy.get('status')}` / `{cron_policy.get('status_reason')}`"
        )
        lines.append(
            f"- cron_policy_counts(missing_timeout/low_timeout/no_stagger): "
            f"`{cron_policy.get('missing_timeout_count')}` / "
            f"`{cron_policy.get('low_timeout_count')}` / "
            f"`{cron_policy.get('no_stagger_count')}`"
        )
        lines.append(
            f"- cron_policy_proxy(proxy_jobs/recent_jobs/timeout_floor/recent_timeout_floor): "
            f"`{cron_policy.get('proxy_error_job_count')}` / "
            f"`{cron_policy.get('proxy_error_recent_job_count')}` / "
            f"`{cron_policy.get('proxy_timeout_floor_count')}` / "
            f"`{cron_policy.get('proxy_timeout_floor_recent_count')}`"
        )
        lines.append(
            f"- cron_policy_recommend(timeout_min/stagger_step/stagger_max/remediation): "
            f"`{cron_policy.get('min_timeout_sec')}` / "
            f"`{cron_policy.get('recommend_stagger_step_ms')}` / "
            f"`{cron_policy.get('recommend_stagger_max_ms')}` / "
            f"`{cron_policy.get('remediation_count')}`"
        )
        lines.append(
            f"- cron_policy_remediation(ready/ops/mode): "
            f"`{cron_policy.get('remediation_ready_count')}` / "
            f"`{cron_policy.get('remediation_operation_count')}` / "
            f"`{cron_policy.get('remediation_mode')}`"
        )
        lines.append(
            f"- proxy_error(status/reason/jobs/core/max_consecutive): "
            f"`{proxy_error.get('status')}` / "
            f"`{proxy_error.get('reason')}` / "
            f"`{proxy_error.get('proxy_error_jobs_count')}` / "
            f"`{proxy_error.get('proxy_error_core_jobs_count')}` / "
            f"`{proxy_error.get('max_consecutive_errors')}`"
        )
        lines.append(
            f"- proxy_error_action(level/reason/next): "
            f"`{proxy_error.get('action_level')}` / "
            f"`{proxy_error.get('action_reason')}` / "
            f"`{proxy_error.get('next_action')}`"
        )
        proxy_thresholds = proxy_error.get("thresholds") if isinstance(proxy_error.get("thresholds"), dict) else {}
        if proxy_thresholds:
            lines.append(
                f"- proxy_error_thresholds(job_degraded/job_critical/consecutive_degraded/consecutive_critical): "
                f"`{proxy_thresholds.get('degraded_job_count')}` / "
                f"`{proxy_thresholds.get('critical_job_count')}` / "
                f"`{proxy_thresholds.get('degraded_consecutive')}` / "
                f"`{proxy_thresholds.get('critical_consecutive')}`"
            )
        proxy_hints = proxy_error.get("action_hints") if isinstance(proxy_error.get("action_hints"), list) else []
        if proxy_hints:
            lines.append(f"- proxy_error_hints: `{' | '.join([str(x) for x in proxy_hints[:4]])}`")
        proxy_jobs = proxy_error.get("jobs") if isinstance(proxy_error.get("jobs"), list) else []
        if proxy_jobs:
            rendered_proxy_jobs = []
            for item in proxy_jobs[:5]:
                if not isinstance(item, dict):
                    continue
                rendered_proxy_jobs.append(
                    f"{item.get('name')}({item.get('consecutiveErrors')}):{str(item.get('lastError') or '')[:60]}"
                )
            if rendered_proxy_jobs:
                lines.append(f"- proxy_error_jobs_top: `{' | '.join(rendered_proxy_jobs)}`")
        lines.append(
            f"- proxy_error_trend(status/reason/jobs/core/max_consecutive/rate): "
            f"`{proxy_error_trend.get('status')}` / "
            f"`{proxy_error_trend.get('reason')}` / "
            f"`{proxy_error_trend.get('jobs_with_proxy_errors')}` / "
            f"`{proxy_error_trend.get('core_jobs_with_proxy_errors')}` / "
            f"`{proxy_error_trend.get('max_consecutive_proxy_errors')}` / "
            f"`{proxy_error_trend.get('proxy_error_rate')}`"
        )
        lines.append(
            f"- proxy_error_trend_retry(pi_cycle_events/retry_events/retry_total/retry_max/retry_rate): "
            f"`{proxy_error_trend.get('pi_cycle_events_total')}` / "
            f"`{proxy_error_trend.get('retry_exhausted_events')}` / "
            f"`{proxy_error_trend.get('retry_exhausted_total')}` / "
            f"`{proxy_error_trend.get('retry_exhausted_max')}` / "
            f"`{proxy_error_trend.get('retry_exhausted_rate')}`"
        )
        lines.append(
            f"- proxy_error_trend_action(level/reason/next): "
            f"`{proxy_error_trend.get('action_level')}` / "
            f"`{proxy_error_trend.get('action_reason')}` / "
            f"`{proxy_error_trend.get('next_action')}`"
        )
        proxy_trend_thresholds = (
            proxy_error_trend.get("thresholds")
            if isinstance(proxy_error_trend.get("thresholds"), dict)
            else {}
        )
        if proxy_trend_thresholds:
            lines.append(
                f"- proxy_error_trend_thresholds(min_runs/rate_degraded/rate_critical/jobs_degraded/jobs_critical): "
                f"`{proxy_trend_thresholds.get('min_runs_per_job')}` / "
                f"`{proxy_trend_thresholds.get('degraded_fail_rate')}` / "
                f"`{proxy_trend_thresholds.get('critical_fail_rate')}` / "
                f"`{proxy_trend_thresholds.get('degraded_job_count')}` / "
                f"`{proxy_trend_thresholds.get('critical_job_count')}`"
            )
        proxy_trend_hints = (
            proxy_error_trend.get("action_hints")
            if isinstance(proxy_error_trend.get("action_hints"), list)
            else []
        )
        if proxy_trend_hints:
            lines.append(f"- proxy_error_trend_hints: `{' | '.join([str(x) for x in proxy_trend_hints[:4]])}`")
        proxy_trend_jobs = proxy_error_trend.get("jobs") if isinstance(proxy_error_trend.get("jobs"), list) else []
        if proxy_trend_jobs:
            rendered_proxy_trend_jobs = []
            for item in proxy_trend_jobs[:5]:
                if not isinstance(item, dict):
                    continue
                rendered_proxy_trend_jobs.append(
                    f"{item.get('name')}({item.get('proxy_error_runs')}/{item.get('sample_runs')},"
                    f"c={item.get('latest_consecutive_proxy_errors')})"
                )
            if rendered_proxy_trend_jobs:
                lines.append(f"- proxy_error_trend_jobs_top: `{' | '.join(rendered_proxy_trend_jobs)}`")
        lines.append(
            f"- cron_policy_patch_draft(status/reason/actionable/changed/failed): "
            f"`{patch_draft.get('status')}` / "
            f"`{patch_draft.get('reason')}` / "
            f"`{patch_draft.get('actionable_jobs')}` / "
            f"`{patch_draft.get('changed_jobs')}` / "
            f"`{patch_draft.get('failed_jobs')}`"
        )
        lines.append(
            f"- cron_policy_rollout(strategy/batches/non_empty/apply_order): "
            f"`{patch_draft.get('rollout_strategy')}` / "
            f"`{patch_draft.get('rollout_batch_count')}` / "
            f"`{patch_draft.get('rollout_non_empty_batches')}` / "
            f"`{patch_draft.get('rollout_apply_order_count')}`"
        )
        lines.append(
            f"- cron_policy_patch_action(level/reason/next/pending): "
            f"`{patch_draft.get('action_level')}` / "
            f"`{patch_draft.get('action_reason')}` / "
            f"`{patch_draft.get('next_action')}` / "
            f"`{patch_draft.get('pending')}`"
        )
        lines.append(
            f"- cron_policy_patch_action_sequence(primary/secondary): "
            f"`{patch_draft.get('next_action')}` / "
            f"`{patch_draft.get('next_action_secondary')}`"
        )
        lines.append(
            f"- cron_policy_patch_isolation(enabled/candidates/actionable/critical/manual): "
            f"`{patch_draft.get('isolation_enabled')}` / "
            f"`{patch_draft.get('isolation_candidates_total')}` / "
            f"`{patch_draft.get('isolation_actionable_candidates')}` / "
            f"`{patch_draft.get('isolation_critical_candidates')}` / "
            f"`{patch_draft.get('isolation_manual_only_candidates')}`"
        )
        lines.append(
            f"- cron_policy_patch_isolation_rollout(strategy/next/pending): "
            f"`{patch_draft.get('isolation_rollout_strategy')}` / "
            f"`{patch_draft.get('isolation_rollout_next_batch')}` / "
            f"`{patch_draft.get('isolation_rollout_pending')}`"
        )
        rollout_batches_top = patch_draft.get("rollout_batches_top") if isinstance(patch_draft.get("rollout_batches_top"), list) else []
        if rollout_batches_top:
            rendered_batches = ", ".join(
                [
                    f"{str(x.get('id'))}:{int(x.get('count') or 0)}/{int(x.get('ops') or 0)}"
                    for x in rollout_batches_top
                    if isinstance(x, dict)
                ]
            )
            if rendered_batches:
                lines.append(f"- cron_policy_rollout_batches_top: `{rendered_batches}`")
        patch_hints = patch_draft.get("action_hints") if isinstance(patch_draft.get("action_hints"), list) else []
        if patch_hints:
            rendered_hints = " | ".join([str(x) for x in patch_hints[:5]])
            lines.append(f"- cron_policy_patch_hints: `{rendered_hints}`")
        isolation_top = (
            patch_draft.get("isolation_candidates_top")
            if isinstance(patch_draft.get("isolation_candidates_top"), list)
            else []
        )
        if isolation_top:
            rendered_isolation = ", ".join(
                [
                    f"{str(x.get('name') or x.get('id'))}:{x.get('severity')}/{x.get('reason')}/{x.get('actionable')}"
                    for x in isolation_top[:5]
                    if isinstance(x, dict)
                ]
            )
            if rendered_isolation:
                lines.append(f"- cron_policy_patch_isolation_top: `{rendered_isolation}`")
        remediation = cron_policy.get("remediation") if isinstance(cron_policy.get("remediation"), list) else []
        if remediation:
            top = []
            for item in remediation[:5]:
                if not isinstance(item, dict):
                    continue
                op_count = int(item.get("operation_count") or 0)
                top.append(
                    f"{item.get('name')}[{','.join([str(x) for x in (item.get('reasons') or [])])}]"
                    f" timeout={item.get('set_timeout_seconds')} stagger={item.get('set_stagger_ms')}"
                    f" ops={op_count}"
                )
            if top:
                lines.append(f"- cron_policy_remediation_top: `{' | '.join(top)}`")
    else:
        lines.append("- unavailable")
    lines.append("")
    lines.append("## Gate Rollout")
    lines.append("")
    if not gate:
        lines.append("- unavailable")
    else:
        lines.append(f"- status: `{gate.get('status')}`")
        lines.append(f"- mode_effective: `{gate.get('rollout_mode_effective')}`")
        lines.append(f"- mode_configured: `{gate.get('rollout_mode_configured')}`")
        lines.append(f"- pi_cycle_ts: `{gate.get('pi_cycle_ts')}`")
        lines.append(
            f"- guard_mode(current/base/recommended/reason/override): "
            f"`{gate.get('guard_mode_effective')}` / "
            f"`{gate.get('guard_mode_base_effective')}` / "
            f"`{gate.get('guard_mode_recommended')}` / "
            f"`{gate.get('guard_mode_recommend_reason')}` / "
            f"`{gate.get('guard_mode_auto_override_applied')}`"
        )
        lines.append(
            f"- alert_guard_mode(winner/recommended/reason): "
            f"`{gate.get('alert_guard_mode_reason_winner')}` / "
            f"`{gate.get('alert_guard_mode_recommended')}` / "
            f"`{gate.get('alert_guard_mode_recommend_reason')}`"
        )
        lines.append(
            f"- alert_guard_mode_priority_order: "
            f"`{' > '.join(gate.get('alert_guard_mode_priority_order') or [])}`"
        )
        lines.append(
            f"- guard_mode_inputs(requested/use/apply): "
            f"`{gate.get('guard_mode_requested')}` / "
            f"`{gate.get('guard_mode_use_rollback_action')}` / "
            f"`{gate.get('guard_mode_apply_rollback_action')}`"
        )
        lines.append(
            f"- guard_mode_patch_policy(use/apply/pending_threshold): "
            f"`{gate.get('guard_mode_use_patch_draft_action')}` / "
            f"`{gate.get('guard_mode_apply_patch_draft_action')}` / "
            f"`{gate.get('guard_mode_patch_pending_threshold')}`"
        )
        lines.append(
            f"- guard_mode_proxy_policy(use/apply/threshold): "
            f"`{gate.get('guard_mode_use_proxy_control_action')}` / "
            f"`{gate.get('guard_mode_apply_proxy_control_action')}` / "
            f"`{gate.get('guard_mode_proxy_control_trigger_threshold')}`"
        )
        lines.append(
            f"- guard_mode_rollback_signal(status/action/reason): "
            f"`{gate.get('guard_mode_rollback_status')}` / "
            f"`{gate.get('guard_mode_rollback_action_level')}` / "
            f"`{gate.get('guard_mode_rollback_action_reason')}`"
        )
        lines.append(
            f"- guard_mode_proxy_signal(status/action/reason/triggers): "
            f"`{gate.get('guard_mode_proxy_control_trend_status')}` / "
            f"`{gate.get('guard_mode_proxy_control_trend_action_level')}` / "
            f"`{gate.get('guard_mode_proxy_control_trend_action_reason')}` / "
            f"`{gate.get('guard_mode_proxy_control_trend_trigger_count')}`"
        )
        lines.append(
            f"- guard_mode_proxy_error(status/trend_status/jobs/trend_jobs/rate): "
            f"`{gate.get('guard_mode_proxy_error_status')}` / "
            f"`{gate.get('guard_mode_proxy_error_trend_status')}` / "
            f"`{gate.get('guard_mode_proxy_error_jobs_count')}` / "
            f"`{gate.get('guard_mode_proxy_error_trend_jobs_with_errors')}` / "
            f"`{gate.get('guard_mode_proxy_error_trend_rate')}`"
        )
        lines.append(
            f"- guard_mode_patch_draft(status/reason/actionable/changed/failed/pending): "
            f"`{gate.get('guard_mode_patch_draft_status')}` / "
            f"`{gate.get('guard_mode_patch_draft_reason')}` / "
            f"`{gate.get('guard_mode_patch_draft_actionable_jobs')}` / "
            f"`{gate.get('guard_mode_patch_draft_changed_jobs')}` / "
            f"`{gate.get('guard_mode_patch_draft_failed_jobs')}` / "
            f"`{gate.get('guard_mode_patch_draft_pending')}`"
        )
        lines.append(
            f"- guard_mode_patch_rollout(strategy/non_empty/apply_order/next_batch): "
            f"`{gate.get('guard_mode_patch_draft_rollout_strategy')}` / "
            f"`{gate.get('guard_mode_patch_draft_rollout_non_empty_batches')}` / "
            f"`{gate.get('guard_mode_patch_draft_rollout_apply_order_count')}` / "
            f"`{gate.get('guard_mode_patch_draft_rollout_next_batch')}`"
        )
        lines.append(
            f"- guard_mode_patch_action(level/reason/next): "
            f"`{gate.get('guard_mode_patch_draft_action_level')}` / "
            f"`{gate.get('guard_mode_patch_draft_action_reason')}` / "
            f"`{gate.get('guard_mode_patch_draft_next_action')}`"
        )
        lines.append(
            f"- guard_mode_patch_action_sequence(primary/secondary): "
            f"`{gate.get('guard_mode_patch_draft_next_action')}` / "
            f"`{gate.get('guard_mode_patch_draft_next_action_secondary')}`"
        )
        lines.append(
            f"- guard_mode_patch_isolation(enabled/candidates/actionable/critical/manual): "
            f"`{gate.get('guard_mode_patch_draft_isolation_enabled')}` / "
            f"`{gate.get('guard_mode_patch_draft_isolation_candidates_total')}` / "
            f"`{gate.get('guard_mode_patch_draft_isolation_actionable_candidates')}` / "
            f"`{gate.get('guard_mode_patch_draft_isolation_critical_candidates')}` / "
            f"`{gate.get('guard_mode_patch_draft_isolation_manual_only_candidates')}`"
        )
        lines.append(
            f"- guard_mode_patch_isolation_rollout(strategy/next/pending): "
            f"`{gate.get('guard_mode_patch_draft_isolation_rollout_strategy')}` / "
            f"`{gate.get('guard_mode_patch_draft_isolation_rollout_next_batch')}` / "
            f"`{gate.get('guard_mode_patch_draft_isolation_rollout_pending')}`"
        )
        lines.append(
            f"- cron_policy_apply_batch(status/reason/present/batch/mode): "
            f"`{gate.get('cron_policy_apply_batch_status')}` / "
            f"`{gate.get('cron_policy_apply_batch_reason')}` / "
            f"`{gate.get('cron_policy_apply_batch_present')}` / "
            f"`{gate.get('cron_policy_apply_batch_batch_id')}` / "
            f"`{gate.get('cron_policy_apply_batch_mode')}`"
        )
        lines.append(
            f"- cron_policy_apply_batch(summary selected/changed/failed/blocked/ops_changed): "
            f"`{gate.get('cron_policy_apply_batch_selected_actions')}` / "
            f"`{gate.get('cron_policy_apply_batch_changed_jobs')}` / "
            f"`{gate.get('cron_policy_apply_batch_failed_jobs')}` / "
            f"`{gate.get('cron_policy_apply_batch_blocked_jobs')}` / "
            f"`{gate.get('cron_policy_apply_batch_operations_changed')}`"
        )
        lines.append(
            f"- cron_policy_apply_batch(apply/applied/skipped): "
            f"`{gate.get('cron_policy_apply_batch_applied_to_source')}` / "
            f"`{gate.get('cron_policy_apply_batch_apply_skipped_reason')}` / "
            f"`{gate.get('cron_policy_apply_batch_output_path')}`"
        )
        lines.append(
            f"- proxy_control(enabled/applied/trigger/reason): "
            f"`{gate.get('proxy_control_enabled')}` / "
            f"`{gate.get('proxy_control_applied')}` / "
            f"`{gate.get('proxy_control_trigger')}` / "
            f"`{gate.get('proxy_control_reason')}`"
        )
        lines.append(
            f"- proxy_control(isolation actionable/critical/next): "
            f"`{gate.get('proxy_control_isolation_actionable_candidates')}` / "
            f"`{gate.get('proxy_control_isolation_critical_candidates')}` / "
            f"`{gate.get('proxy_control_isolation_next_batch')}`"
        )
        lines.append(
            f"- digital_life_control(enabled/applied/trigger/reason): "
            f"`{gate.get('digital_life_control_enabled')}` / "
            f"`{gate.get('digital_life_control_applied')}` / "
            f"`{gate.get('digital_life_control_trigger')}` / "
            f"`{gate.get('digital_life_control_reason')}`"
        )
        patch_guard_hints = (
            gate.get("guard_mode_patch_draft_action_hints")
            if isinstance(gate.get("guard_mode_patch_draft_action_hints"), list)
            else []
        )
        if patch_guard_hints:
            rendered_patch_guard_hints = " | ".join([str(x) for x in patch_guard_hints[:5]])
            lines.append(f"- guard_mode_patch_hints: `{rendered_patch_guard_hints}`")
        lines.append(
            f"- step_resilience(retry_attempts/retried_steps/optional_fallbacks): "
            f"`{gate.get('step_retry_attempts_total')}` / "
            f"`{gate.get('step_retried_count')}` / "
            f"`{gate.get('step_optional_fallback_count')}`"
        )
        lines.append(
            f"- core_proxy_bypass(status/reason/source/last_reason): "
            f"`{gate.get('core_proxy_bypass_status')}` / "
            f"`{gate.get('core_proxy_bypass_reason')}` / "
            f"`{gate.get('core_proxy_bypass_source')}` / "
            f"`{gate.get('core_proxy_bypass_last_reason')}`"
        )
        lines.append(
            f"- core_proxy_bypass_total(requests/attempted/success/failed): "
            f"`{gate.get('core_proxy_bypass_requests_total')}` / "
            f"`{gate.get('core_proxy_bypass_bypass_attempted')}` / "
            f"`{gate.get('core_proxy_bypass_bypass_success')}` / "
            f"`{gate.get('core_proxy_bypass_bypass_failed')}`"
        )
        lines.append(
            f"- core_proxy_bypass_retry(total_retries/retry_exhausted): "
            f"`{gate.get('core_proxy_bypass_no_proxy_retries')}` / "
            f"`{gate.get('core_proxy_bypass_no_proxy_retry_exhausted')}`"
        )
        lines.append(
            f"- core_proxy_bypass_hints(response/exception/response_hint/exception_hint): "
            f"`{gate.get('core_proxy_bypass_hint_response')}` / "
            f"`{gate.get('core_proxy_bypass_hint_exception')}` / "
            f"`{gate.get('core_proxy_bypass_reason_response_hint')}` / "
            f"`{gate.get('core_proxy_bypass_reason_exception_hint')}`"
        )
        lines.append(
            f"- core_proxy_bypass_split(local_req/local_attempt/local_success/local_failed | "
            f"exec_req/exec_attempt/exec_success/exec_failed): "
            f"`{gate.get('core_proxy_bypass_local_requests_total')}` / "
            f"`{gate.get('core_proxy_bypass_local_bypass_attempted')}` / "
            f"`{gate.get('core_proxy_bypass_local_bypass_success')}` / "
            f"`{gate.get('core_proxy_bypass_local_bypass_failed')}` | "
            f"`{gate.get('core_proxy_bypass_exec_requests_total')}` / "
            f"`{gate.get('core_proxy_bypass_exec_bypass_attempted')}` / "
            f"`{gate.get('core_proxy_bypass_exec_bypass_success')}` / "
            f"`{gate.get('core_proxy_bypass_exec_bypass_failed')}`"
        )
        lines.append(
            f"- core_execution(status/reason/source/decision): "
            f"`{gate.get('core_execution_status')}` / "
            f"`{gate.get('core_execution_reason')}` / "
            f"`{gate.get('core_execution_source')}` / "
            f"`{gate.get('core_execution_decision')}`"
        )
        lines.append(
            f"- core_execution_order(http/endpoint/mode/simulated/error): "
            f"`{gate.get('core_execution_order_http')}` / "
            f"`{gate.get('core_execution_order_endpoint')}` / "
            f"`{gate.get('core_execution_order_mode')}` / "
            f"`{gate.get('core_execution_order_simulated')}` / "
            f"`{gate.get('core_execution_order_error')}`"
        )
        lines.append(
            f"- core_execution_executor(attempted/probe/live/guardrail): "
            f"`{gate.get('core_execution_executor_attempted')}` / "
            f"`{gate.get('core_execution_executor_probe_effective')}` / "
            f"`{gate.get('core_execution_executor_live_effective')}` / "
            f"`{gate.get('core_execution_guardrail_hit')}`"
        )
        lines.append(
            f"- core_execution_paper(attempted/applied/route/fill_px/slippage_bps/fee_usdt/ledger): "
            f"`{gate.get('core_execution_paper_execution_attempted')}` / "
            f"`{gate.get('core_execution_paper_execution_applied')}` / "
            f"`{gate.get('core_execution_paper_execution_route')}` / "
            f"`{gate.get('core_execution_paper_execution_fill_px')}` / "
            f"`{gate.get('core_execution_paper_execution_signed_slippage_bps')}` / "
            f"`{gate.get('core_execution_paper_execution_fee_usdt')}` / "
            f"`{gate.get('core_execution_paper_execution_ledger_written')}`"
        )
        lines.append(
            f"- ops_next_action(priority/action/reason/secondary/source): "
            f"`{gate.get('ops_next_action_priority')}` / "
            f"`{gate.get('ops_next_action')}` / "
            f"`{gate.get('ops_next_action_reason')}` / "
            f"`{gate.get('ops_next_action_secondary')}` / "
            f"`{gate.get('ops_next_action_source')}`"
        )
        lines.append(
            f"- step_failures(critical/optional): "
            f"`{len(gate.get('fail_steps_critical') or [])}` / "
            f"`{len(gate.get('fail_steps_optional') or [])}`"
        )
        fail_steps_critical = gate.get("fail_steps_critical") if isinstance(gate.get("fail_steps_critical"), list) else []
        if fail_steps_critical:
            lines.append(f"- step_fail_critical_list: `{', '.join([str(x) for x in fail_steps_critical[:6]])}`")
        fail_steps_optional = gate.get("fail_steps_optional") if isinstance(gate.get("fail_steps_optional"), list) else []
        if fail_steps_optional:
            lines.append(f"- step_fail_optional_list: `{', '.join([str(x) for x in fail_steps_optional[:6]])}`")
        lines.append(f"- rates(would_block/cooldown/recover_fail): `{gate.get('would_block_rate')}` / `{gate.get('cooldown_hit_rate')}` / `{gate.get('recover_confirm_fail_rate')}`")
        lines.append(f"- events(act/total): `{gate.get('act_events')}` / `{gate.get('total_events')}`")
        lines.append(
            f"- alert(status/emitted/reason): `{gate.get('alert_status')}` / "
            f"`{gate.get('alert_emitted')}` / `{gate.get('alert_emit_reason')}`"
        )
        lines.append(
            f"- notify(sent/reason): `{gate.get('alert_notify_sent')}` / "
            f"`{gate.get('alert_notify_reason')}`"
        )
        lines.append(
            f"- notify_gate_status(base/effective/source): "
            f"`{gate.get('alert_notify_trend_status_base')}` / "
            f"`{gate.get('alert_notify_trend_status')}` / "
            f"`{gate.get('alert_notify_trend_status_source')}`"
        )
        lines.append(
            f"- notify_component_mode(shadow_would_upgrade/target): "
            f"`{gate.get('alert_notify_trend_component_score_mode_effective')}` / "
            f"`{gate.get('alert_notify_trend_component_score_shadow_would_upgrade')}` / "
            f"`{gate.get('alert_notify_trend_component_score_shadow_target_status')}`"
        )
        lines.append(
            f"- notify_component_mode(configured/effective/source): "
            f"`{gate.get('alert_notify_trend_component_score_mode_configured')}` / "
            f"`{gate.get('alert_notify_trend_component_score_mode_effective')}` / "
            f"`{gate.get('alert_notify_trend_component_score_mode_source')}`"
        )
        lines.append(
            f"- notify_component_auto(enabled/recommend/reason): "
            f"`{gate.get('alert_notify_trend_component_score_auto_switch_enabled')}` / "
            f"`{gate.get('alert_notify_trend_component_score_auto_recommend_enforce')}` / "
            f"`{gate.get('alert_notify_trend_component_score_auto_recommend_reason')}`"
        )
        lines.append(
            f"- notify_component_auto_windows(stable/unstable/promote/demote): "
            f"`{gate.get('alert_notify_trend_component_score_auto_stable_windows')}` / "
            f"`{gate.get('alert_notify_trend_component_score_auto_unstable_windows')}` / "
            f"`{gate.get('alert_notify_trend_component_score_auto_promote_windows')}` / "
            f"`{gate.get('alert_notify_trend_component_score_auto_demote_windows')}`"
        )
        lines.append(
            f"- notify_component_auto_transition/path: "
            f"`{gate.get('alert_notify_trend_component_score_auto_transition')}` / "
            f"`{gate.get('alert_notify_trend_component_score_auto_state_path')}`"
        )
        lines.append(
            f"- notify_component_mode_drift_24h(mode/drift/rate/consecutive/status): "
            f"`{gate.get('notify_trend_component_score_mode_drift_mode_events')}` / "
            f"`{gate.get('notify_trend_component_score_mode_drift_events')}` / "
            f"`{gate.get('notify_trend_component_score_mode_drift_rate')}` / "
            f"`{gate.get('notify_trend_component_score_mode_drift_latest_consecutive')}` / "
            f"`{gate.get('notify_trend_component_score_mode_drift_status')}`"
        )
        lines.append(
            f"- notify_component_mode_drift_reason: "
            f"`{gate.get('notify_trend_component_score_mode_drift_reason')}`"
        )
        lines.append(
            f"- rollback_latest(trigger/applied/reason/expires): "
            f"`{gate.get('rollback_trigger')}` / "
            f"`{gate.get('rollback_applied')}` / "
            f"`{gate.get('rollback_reason')}` / "
            f"`{gate.get('rollback_expires_ts')}`"
        )
        lines.append(
            f"- rollback_24h(status/events/records/triggered/rate): "
            f"`{gate.get('rollback_trend_status')}` / "
            f"`{gate.get('rollback_trend_pi_cycle_events')}` / "
            f"`{gate.get('rollback_trend_records')}` / "
            f"`{gate.get('rollback_trend_trigger_count')}` / "
            f"`{gate.get('rollback_trend_trigger_rate')}`"
        )
        rollback_trigger_counts = (
            gate.get("rollback_trend_trigger_counts")
            if isinstance(gate.get("rollback_trend_trigger_counts"), dict)
            else {}
        )
        lines.append(
            f"- rollback_24h_triggers(breach/drift/both): "
            f"`{rollback_trigger_counts.get('breach')}` / "
            f"`{rollback_trigger_counts.get('component_mode_drift')}` / "
            f"`{rollback_trigger_counts.get('breach_and_component_mode_drift')}`"
        )
        lines.append(
            f"- rollback_24h_dominant(trigger/count/share): "
            f"`{gate.get('rollback_trend_dominant_trigger')}` / "
            f"`{gate.get('rollback_trend_dominant_trigger_count')}` / "
            f"`{gate.get('rollback_trend_dominant_trigger_share')}`"
        )
        lines.append(
            f"- rollback_24h_action(level/reason): "
            f"`{gate.get('rollback_trend_action_level')}` / "
            f"`{gate.get('rollback_trend_action_reason')}`"
        )
        lines.append(
            f"- proxy_control_24h(status/events/records/triggered/rate): "
            f"`{gate.get('proxy_control_trend_status')}` / "
            f"`{gate.get('proxy_control_trend_pi_cycle_events')}` / "
            f"`{gate.get('proxy_control_trend_records')}` / "
            f"`{gate.get('proxy_control_trend_trigger_count')}` / "
            f"`{gate.get('proxy_control_trend_trigger_rate')}`"
        )
        proxy_triggers = (
            gate.get("proxy_control_trend_trigger_counts")
            if isinstance(gate.get("proxy_control_trend_trigger_counts"), dict)
            else {}
        )
        lines.append(
            f"- proxy_control_24h_triggers(critical/degraded/pending): "
            f"`{proxy_triggers.get('proxy_isolation_critical_candidates')}` / "
            f"`{proxy_triggers.get('proxy_isolation_actionable_degraded')}` / "
            f"`{proxy_triggers.get('proxy_isolation_actionable_pending')}`"
        )
        lines.append(
            f"- proxy_control_24h_dominant(trigger/count/share): "
            f"`{gate.get('proxy_control_trend_dominant_trigger')}` / "
            f"`{gate.get('proxy_control_trend_dominant_trigger_count')}` / "
            f"`{gate.get('proxy_control_trend_dominant_trigger_share')}`"
        )
        lines.append(
            f"- proxy_control_24h_action(level/reason): "
            f"`{gate.get('proxy_control_trend_action_level')}` / "
            f"`{gate.get('proxy_control_trend_action_reason')}`"
        )
        lines.append(
            f"- notify_component_shadow_24h(mode_events/would_upgrade/rate): "
            f"`{gate.get('notify_trend_component_score_shadow_mode_events')}` / "
            f"`{gate.get('notify_trend_component_score_shadow_would_upgrade')}` / "
            f"`{gate.get('notify_trend_component_score_shadow_would_upgrade_rate')}`"
        )
        lines.append(
            f"- notify_component_shadow_critical(count/share): "
            f"`{gate.get('notify_trend_component_score_shadow_critical_target_count')}` / "
            f"`{gate.get('notify_trend_component_score_shadow_critical_target_share')}`"
        )
        lines.append(
            f"- notify_component_shadow_recommend(enforce/reason): "
            f"`{gate.get('notify_trend_component_score_shadow_recommend_enforce')}` / "
            f"`{gate.get('notify_trend_component_score_shadow_recommend_reason')}`"
        )
        lines.append(
            f"- notify_trend_24h(attempted/sent/failed/success_rate): "
            f"`{gate.get('notify_trend_attempted')}` / `{gate.get('notify_trend_sent')}` / "
            f"`{gate.get('notify_trend_failed')}` / `{gate.get('notify_trend_success_rate')}`"
        )
        lines.append(
            f"- notify_trend_status/fail_rate: `{gate.get('notify_trend_status')}` / "
            f"`{gate.get('notify_trend_fail_rate')}`"
        )
        lines.append(
            f"- notify_components(fail/suppression/min_interval): "
            f"`{gate.get('notify_trend_fail_status')}` / "
            f"`{gate.get('notify_trend_suppression_status')}` / "
            f"`{gate.get('notify_trend_min_interval_status')}`"
        )
        lines.append(
            f"- notify_component_score(status/score): "
            f"`{gate.get('notify_trend_component_score_status')}` / "
            f"`{gate.get('notify_trend_component_score')}`"
        )
        lines.append(
            f"- notify_status_reason: `{gate.get('notify_trend_status_reason')}`"
        )
        lines.append(
            f"- notify_suppression_24h(count/rate): "
            f"`{gate.get('notify_trend_suppressed')}` / `{gate.get('notify_trend_suppressed_rate')}`"
        )
        lines.append(
            f"- min_interval_escalation_24h(count/rate): "
            f"`{gate.get('notify_trend_min_interval_escalated')}` / "
            f"`{gate.get('notify_trend_min_interval_escalation_rate')}`"
        )
        lines.append(
            f"- patch_action_24h(events/escalated/escalated_rate/pending/pending_rate): "
            f"`{gate.get('notify_trend_patch_action_events')}` / "
            f"`{gate.get('notify_trend_patch_action_escalated')}` / "
            f"`{gate.get('notify_trend_patch_action_escalated_rate')}` / "
            f"`{gate.get('notify_trend_patch_action_pending')}` / "
            f"`{gate.get('notify_trend_patch_action_pending_rate')}`"
        )
        lines.append(
            f"- patch_action_status/reason: "
            f"`{gate.get('notify_trend_patch_action_status')}` / "
            f"`{gate.get('notify_trend_patch_action_status_reason')}`"
        )
        lines.append(
            f"- patch_next_action_24h(dominant/count/share): "
            f"`{gate.get('notify_trend_patch_dominant_next_action')}` / "
            f"`{gate.get('notify_trend_patch_dominant_next_action_count')}` / "
            f"`{gate.get('notify_trend_patch_dominant_next_action_share')}`"
        )
        lines.append(
            f"- patch_action_reason_24h(dominant/count/share): "
            f"`{gate.get('notify_trend_patch_dominant_action_reason')}` / "
            f"`{gate.get('notify_trend_patch_dominant_action_reason_count')}` / "
            f"`{gate.get('notify_trend_patch_dominant_action_reason_share')}`"
        )
        lines.append(
            f"- patch_action_recommend(next/reason/confidence): "
            f"`{gate.get('notify_trend_patch_recommended_next_action')}` / "
            f"`{gate.get('notify_trend_patch_recommended_reason')}` / "
            f"`{gate.get('notify_trend_patch_recommended_confidence')}`"
        )
        lines.append(
            f"- patch_apply_24h(events/problem/problem_rate/critical_rate): "
            f"`{gate.get('notify_trend_patch_apply_events')}` / "
            f"`{gate.get('notify_trend_patch_apply_problem_events')}` / "
            f"`{gate.get('notify_trend_patch_apply_problem_rate')}` / "
            f"`{gate.get('notify_trend_patch_apply_critical_problem_rate')}`"
        )
        lines.append(
            f"- patch_apply_status/reason/target_batch: "
            f"`{gate.get('notify_trend_patch_apply_status')}` / "
            f"`{gate.get('notify_trend_patch_apply_status_reason')}` / "
            f"`{gate.get('notify_trend_patch_apply_target_batch_id')}`"
        )
        lines.append(
            f"- patch_apply_dominant_reason(count/share): "
            f"`{gate.get('notify_trend_patch_apply_dominant_reason')}` / "
            f"`{gate.get('notify_trend_patch_apply_dominant_reason_count')}` / "
            f"`{gate.get('notify_trend_patch_apply_dominant_reason_share')}`"
        )
        patch_apply_thresholds = gate.get("notify_trend_patch_apply_thresholds")
        if not isinstance(patch_apply_thresholds, dict):
            patch_apply_thresholds = {}
        lines.append(
            f"- patch_apply_thresholds(min_events/degraded/critical): "
            f"`{patch_apply_thresholds.get('min_events')}` / "
            f"`{patch_apply_thresholds.get('degraded_problem_rate')}` / "
            f"`{patch_apply_thresholds.get('critical_problem_rate')}`"
        )
        fail_top = gate.get("notify_trend_failure_reasons_top") if isinstance(gate.get("notify_trend_failure_reasons_top"), list) else []
        if fail_top:
            rendered = ", ".join(
                [f"{str(x.get('reason'))}:{int(x.get('count') or 0)}" for x in fail_top if isinstance(x, dict)]
            )
            lines.append(f"- notify_fail_top: `{rendered}`")
        suppression_top = gate.get("notify_trend_suppression_reasons_top") if isinstance(gate.get("notify_trend_suppression_reasons_top"), list) else []
        if suppression_top:
            rendered = ", ".join(
                [f"{str(x.get('reason'))}:{int(x.get('count') or 0)}" for x in suppression_top if isinstance(x, dict)]
            )
            lines.append(f"- notify_suppression_top: `{rendered}`")
        min_interval_top = gate.get("notify_trend_min_interval_sources_top") if isinstance(gate.get("notify_trend_min_interval_sources_top"), list) else []
        if min_interval_top:
            rendered = ", ".join(
                [f"{str(x.get('source'))}:{int(x.get('count') or 0)}" for x in min_interval_top if isinstance(x, dict)]
            )
            lines.append(f"- min_interval_source_top: `{rendered}`")
        patch_level_top = gate.get("notify_trend_patch_action_levels_top") if isinstance(gate.get("notify_trend_patch_action_levels_top"), list) else []
        if patch_level_top:
            rendered = ", ".join(
                [f"{str(x.get('level'))}:{int(x.get('count') or 0)}" for x in patch_level_top if isinstance(x, dict)]
            )
            lines.append(f"- patch_action_level_top: `{rendered}`")
        patch_reason_top = gate.get("notify_trend_patch_action_reasons_top") if isinstance(gate.get("notify_trend_patch_action_reasons_top"), list) else []
        if patch_reason_top:
            rendered = ", ".join(
                [f"{str(x.get('reason'))}:{int(x.get('count') or 0)}" for x in patch_reason_top if isinstance(x, dict)]
            )
            lines.append(f"- patch_action_reason_top: `{rendered}`")
        patch_next_top = gate.get("notify_trend_patch_next_actions_top") if isinstance(gate.get("notify_trend_patch_next_actions_top"), list) else []
        if patch_next_top:
            rendered = ", ".join(
                [f"{str(x.get('action'))}:{int(x.get('count') or 0)}" for x in patch_next_top if isinstance(x, dict)]
            )
            lines.append(f"- patch_next_action_top: `{rendered}`")
        patch_hint_top = gate.get("notify_trend_patch_action_hints_top") if isinstance(gate.get("notify_trend_patch_action_hints_top"), list) else []
        if patch_hint_top:
            rendered = " | ".join(
                [f"{str(x.get('hint'))}:{int(x.get('count') or 0)}" for x in patch_hint_top if isinstance(x, dict)]
            )
            lines.append(f"- patch_action_hint_top: {rendered}")
        patch_apply_status_top = gate.get("notify_trend_patch_apply_statuses_top") if isinstance(gate.get("notify_trend_patch_apply_statuses_top"), list) else []
        if patch_apply_status_top:
            rendered = ", ".join(
                [f"{str(x.get('status'))}:{int(x.get('count') or 0)}" for x in patch_apply_status_top if isinstance(x, dict)]
            )
            lines.append(f"- patch_apply_status_top: `{rendered}`")
        patch_apply_reason_top = gate.get("notify_trend_patch_apply_reasons_top") if isinstance(gate.get("notify_trend_patch_apply_reasons_top"), list) else []
        if patch_apply_reason_top:
            rendered = ", ".join(
                [f"{str(x.get('reason'))}:{int(x.get('count') or 0)}" for x in patch_apply_reason_top if isinstance(x, dict)]
            )
            lines.append(f"- patch_apply_reason_top: `{rendered}`")
        patch_apply_mode_top = gate.get("notify_trend_patch_apply_modes_top") if isinstance(gate.get("notify_trend_patch_apply_modes_top"), list) else []
        if patch_apply_mode_top:
            rendered = ", ".join(
                [f"{str(x.get('mode'))}:{int(x.get('count') or 0)}" for x in patch_apply_mode_top if isinstance(x, dict)]
            )
            lines.append(f"- patch_apply_mode_top: `{rendered}`")
        shadow_target_top = gate.get("notify_trend_component_score_shadow_target_status_top") if isinstance(gate.get("notify_trend_component_score_shadow_target_status_top"), list) else []
        if shadow_target_top:
            rendered = ", ".join(
                [f"{str(x.get('status'))}:{int(x.get('count') or 0)}" for x in shadow_target_top if isinstance(x, dict)]
            )
            lines.append(f"- notify_component_shadow_target_top: `{rendered}`")
        drift_top = gate.get("notify_trend_component_score_mode_drift_top_pairs") if isinstance(gate.get("notify_trend_component_score_mode_drift_top_pairs"), list) else []
        if drift_top:
            rendered = ", ".join(
                [f"{str(x.get('pair'))}:{int(x.get('count') or 0)}" for x in drift_top if isinstance(x, dict)]
            )
            lines.append(f"- notify_component_mode_drift_top_pairs: `{rendered}`")
        rollback_top = gate.get("rollback_trend_top_triggers") if isinstance(gate.get("rollback_trend_top_triggers"), list) else []
        if rollback_top:
            rendered = ", ".join(
                [f"{str(x.get('trigger'))}:{int(x.get('count') or 0)}" for x in rollback_top if isinstance(x, dict)]
            )
            lines.append(f"- rollback_24h_top_triggers: `{rendered}`")
        rollback_hints = gate.get("rollback_trend_action_hints") if isinstance(gate.get("rollback_trend_action_hints"), list) else []
        if rollback_hints:
            rendered = " | ".join([str(x) for x in rollback_hints if str(x).strip()])
            lines.append(f"- rollback_24h_action_hints: `{rendered}`")
        rollback_recs = gate.get("rollback_trend_action_recommendations") if isinstance(gate.get("rollback_trend_action_recommendations"), list) else []
        if rollback_recs:
            rendered = " | ".join([str(x) for x in rollback_recs if str(x).strip()])
            lines.append(f"- rollback_24h_action_recommendations: `{rendered}`")
        proxy_top = gate.get("proxy_control_trend_top_triggers") if isinstance(gate.get("proxy_control_trend_top_triggers"), list) else []
        if proxy_top:
            rendered = ", ".join(
                [f"{str(x.get('trigger'))}:{int(x.get('count') or 0)}" for x in proxy_top if isinstance(x, dict)]
            )
            lines.append(f"- proxy_control_24h_top_triggers: `{rendered}`")
        proxy_hints = gate.get("proxy_control_trend_action_hints") if isinstance(gate.get("proxy_control_trend_action_hints"), list) else []
        if proxy_hints:
            rendered = " | ".join([str(x) for x in proxy_hints if str(x).strip()])
            lines.append(f"- proxy_control_24h_action_hints: `{rendered}`")
        proxy_recs = gate.get("proxy_control_trend_action_recommendations") if isinstance(gate.get("proxy_control_trend_action_recommendations"), list) else []
        if proxy_recs:
            rendered = " | ".join([str(x) for x in proxy_recs if str(x).strip()])
            lines.append(f"- proxy_control_24h_action_recommendations: `{rendered}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-path", default=str(DEFAULT_JOBS))
    ap.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    ap.add_argument("--openclaw-config", default=str(DEFAULT_OPENCLAW_CONFIG))
    ap.add_argument("--output", default=str(_default_output_path()))
    ap.add_argument("--markdown-output", default=str(_default_markdown_path()))
    args = ap.parse_args()

    jobs_path = Path(args.jobs_path).expanduser().resolve()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    openclaw_config = Path(args.openclaw_config).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    md_path = Path(args.markdown_output).expanduser().resolve()

    jobs_obj = json.loads(jobs_path.read_text(encoding="utf-8"))
    config_obj = {}
    if openclaw_config.exists():
        config_obj = json.loads(openclaw_config.read_text(encoding="utf-8"))
    known_agents = _load_known_agent_ids(openclaw_config)
    report = _build_report(
        jobs_obj,
        runs_dir=runs_dir,
        known_agents=known_agents,
        config_obj=config_obj,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_build_markdown(report) + "\n", encoding="utf-8")
    report["markdown_output"] = str(md_path)
    report["json_output"] = str(out_path)

    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
