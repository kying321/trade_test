#!/usr/bin/env python3
"""Single-entry orchestrator for Pi half-hour cycle.

Pipeline (default):
1) lie_spot_halfhour_core.py
2) lie_spine_watchdog.py --once
3) neuro_guard_cycle.py --mode fast|full

Designed for cron/automation use:
- lock-protected (prevents duplicate overlapping runs)
- structured JSON output
- event appended to STATE.md and system logs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from gate_notify_trend import compute_notify_trend
from lie_root_resolver import resolve_lie_system_root


WORKSPACE = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = resolve_lie_system_root()
STATE_MD = Path(
    os.getenv("OPENCLAW_STATE_FILE", str(WORKSPACE / "STATE.md"))
)
LOG_JSONL = Path(
    os.getenv("PI_CYCLE_LOG_JSONL", str(SYSTEM_ROOT / "output" / "logs" / "pi_cycle_events.jsonl"))
)
LOCK_PATH = WORKSPACE / ".pi" / "pi_cycle.lock"
if os.getenv("PI_CYCLE_LOCK_PATH"):
    LOCK_PATH = Path(str(os.getenv("PI_CYCLE_LOCK_PATH")))
DEFAULT_GATE_LOG = SYSTEM_ROOT / "output" / "logs" / "cortex_gate_events.jsonl"
DEFAULT_ROLLOUT_STATE = SYSTEM_ROOT / "output" / "logs" / "cortex_gate_rollout_state.json"
DEFAULT_ROLLOUT_CONTROL = SYSTEM_ROOT / "output" / "logs" / "cortex_gate_rollout_control.json"
DEFAULT_GATE_ALERT_STATE = SYSTEM_ROOT / "output" / "logs" / "cortex_gate_alert_state.json"
DEFAULT_GATE_ALERT_HISTORY = SYSTEM_ROOT / "output" / "logs" / "cortex_gate_alert_events.jsonl"
DEFAULT_GATE_ALERT_NOTIFY_STATE = SYSTEM_ROOT / "output" / "logs" / "cortex_gate_alert_notify_state.json"
DEFAULT_GATE_ALERT_NOTIFY_HISTORY = SYSTEM_ROOT / "output" / "logs" / "cortex_gate_alert_notify_events.jsonl"
DEFAULT_GATE_ALERT_COMPONENT_SCORE_STATE = (
    SYSTEM_ROOT / "output" / "logs" / "cortex_gate_alert_component_score_state.json"
)
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def append_state_event(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write("\nPI_CYCLE_EVENT=" + json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def parse_json_tail(text: str) -> Optional[Dict[str, Any]]:
    for raw in reversed((text or "").splitlines()):
        raw = raw.strip()
        if not raw or not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def parse_ts(value: Any) -> Optional[dt.datetime]:
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


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def load_json_file(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(float(os.getenv(name, str(default))))
    except Exception:
        value = int(default)
    return max(int(minimum), value)


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


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


def resolve_component_score_mode(
    *,
    notify_trend: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    configured_mode = str(os.getenv("PI_GATE_ALERT_COMPONENT_SCORE_MODE", "")).strip().lower()
    source = "env"
    if configured_mode not in {"off", "shadow", "enforce"}:
        configured_mode = (
            "enforce" if env_flag("PI_GATE_ALERT_USE_COMPONENT_SCORE_STATUS", default=False) else "off"
        )
        source = "legacy_env"

    out: Dict[str, Any] = {
        "configured_mode": configured_mode,
        "effective_mode": configured_mode,
        "mode_source": source,
        "auto_switch_enabled": False,
        "auto_recommend_enforce": None,
        "auto_recommend_reason": "",
        "auto_state_path": None,
        "auto_promote_windows": None,
        "auto_demote_windows": None,
        "auto_stable_windows": 0,
        "auto_unstable_windows": 0,
        "auto_transition": None,
        "auto_state_io_degraded": False,
    }
    if configured_mode != "shadow":
        out["auto_recommend_reason"] = "configured_not_shadow"
        return out

    auto_enabled = env_flag("PI_GATE_ALERT_COMPONENT_SCORE_AUTO_SWITCH_ENABLED", default=False)
    out["auto_switch_enabled"] = auto_enabled
    recommendation = notify_trend.get("component_score_shadow_recommend_enforce")
    if recommendation is True:
        recommend_enforce: Optional[bool] = True
    elif recommendation is False:
        recommend_enforce = False
    else:
        recommend_enforce = None
    recommend_reason = str(notify_trend.get("component_score_shadow_recommend_reason") or "").strip()
    out["auto_recommend_enforce"] = recommend_enforce
    out["auto_recommend_reason"] = recommend_reason or "recommendation_missing"
    if not auto_enabled:
        return out

    promote_windows = _int_env(
        "PI_GATE_ALERT_COMPONENT_SCORE_AUTO_PROMOTE_WINDOWS",
        3,
        minimum=1,
    )
    demote_windows = _int_env(
        "PI_GATE_ALERT_COMPONENT_SCORE_AUTO_DEMOTE_WINDOWS",
        2,
        minimum=1,
    )
    state_path = Path(
        os.getenv(
            "PI_GATE_ALERT_COMPONENT_SCORE_ROLLOUT_STATE_PATH",
            str(DEFAULT_GATE_ALERT_COMPONENT_SCORE_STATE),
        )
    )
    out["auto_state_path"] = str(state_path)
    out["auto_promote_windows"] = promote_windows
    out["auto_demote_windows"] = demote_windows

    state = load_json_file(state_path) if state_path.exists() else {}
    previous_mode = str(state.get("effective_mode") or "shadow").strip().lower()
    if previous_mode not in {"shadow", "enforce"}:
        previous_mode = "shadow"
    stable_windows = max(0, _int_value(state.get("stable_recommend_windows"), 0))
    unstable_windows = max(0, _int_value(state.get("unstable_recommend_windows"), 0))

    if recommend_enforce is True:
        stable_windows += 1
        unstable_windows = 0
    elif recommend_enforce is False:
        unstable_windows += 1
        stable_windows = 0
    else:
        stable_windows = 0
        unstable_windows = 0

    effective_mode = previous_mode
    transition: Optional[str] = None
    if previous_mode == "shadow" and recommend_enforce is True and stable_windows >= promote_windows:
        effective_mode = "enforce"
        transition = "shadow->enforce"
        stable_windows = 0
        unstable_windows = 0
    elif previous_mode == "enforce" and recommend_enforce is False and unstable_windows >= demote_windows:
        effective_mode = "shadow"
        transition = "enforce->shadow"
        stable_windows = 0
        unstable_windows = 0

    now = dt.datetime.now(dt.timezone.utc)
    state_payload: Dict[str, Any] = {
        "last_updated_ts": now.isoformat(),
        "configured_mode": configured_mode,
        "effective_mode": effective_mode,
        "stable_recommend_windows": stable_windows,
        "unstable_recommend_windows": unstable_windows,
        "promote_windows": promote_windows,
        "demote_windows": demote_windows,
        "last_recommend_enforce": recommend_enforce,
        "last_recommend_reason": recommend_reason or "recommendation_missing",
    }
    if transition:
        state_payload["last_transition"] = transition
        state_payload["last_transition_ts"] = now.isoformat()
    elif isinstance(state.get("last_transition"), str) and state.get("last_transition"):
        state_payload["last_transition"] = state.get("last_transition")
        state_payload["last_transition_ts"] = state.get("last_transition_ts")

    if not dry_run:
        write_ok = write_json_atomic(state_path, state_payload)
        if not write_ok:
            out["auto_state_io_degraded"] = True
    else:
        out["auto_state_io_degraded"] = state_path.exists() and not state

    out["effective_mode"] = effective_mode
    out["mode_source"] = "auto_state"
    out["auto_transition"] = transition
    out["auto_stable_windows"] = stable_windows
    out["auto_unstable_windows"] = unstable_windows
    return out


def resolve_component_mode_drift_signal(
    *,
    log_path: Path,
    current_configured_mode: Optional[str],
    current_effective_mode: Optional[str],
) -> Dict[str, Any]:
    try:
        window_hours = float(os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_WINDOW_HOURS", "24"))
    except Exception:
        window_hours = 24.0
    window_hours = max(0.1, float(window_hours))
    max_lines = _int_env("PI_GATE_COMPONENT_MODE_DRIFT_MAX_LINES", 5000, minimum=200)
    min_events = _int_env("PI_GATE_COMPONENT_MODE_DRIFT_MIN_EVENTS", 3, minimum=1)
    min_consecutive = _int_env("PI_GATE_COMPONENT_MODE_DRIFT_MIN_CONSECUTIVE", 2, minimum=1)
    rate_degraded = max(
        0.0,
        min(1.0, _as_float(os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_RATE_DEGRADED", "0.50")) or 0.50),
    )
    rate_critical = max(
        rate_degraded,
        min(1.0, _as_float(os.getenv("PI_GATE_COMPONENT_MODE_DRIFT_RATE_CRITICAL", "0.80")) or 0.80),
    )
    out: Dict[str, Any] = {
        "status": "unknown",
        "reason": "insufficient_mode_events",
        "window_hours": window_hours,
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

    allowed_modes = {"off", "shadow", "enforce"}
    drift_pair_counts: Dict[str, int] = {}
    consecutive_open = True

    def consume(configured: str, effective: str) -> None:
        nonlocal consecutive_open
        if configured not in allowed_modes or effective not in allowed_modes:
            return
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

    current_configured = str(current_configured_mode or "").strip().lower()
    current_effective = str(current_effective_mode or "").strip().lower()
    if current_configured and current_effective:
        consume(current_configured, current_effective)

    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            lines = []
    else:
        lines = []

    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
    for raw in reversed(lines[-max(1, max_lines) :]):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or str(obj.get("domain") or "") != "pi_cycle":
            continue
        ts = parse_ts(obj.get("ts"))
        if ts is None or ts < since:
            continue
        emit = obj.get("gate_rollout_alert_emit") if isinstance(obj.get("gate_rollout_alert_emit"), dict) else {}
        configured = str(emit.get("gate_notify_trend_component_score_mode_configured") or "").strip().lower()
        effective = str(emit.get("gate_notify_trend_component_score_mode_effective") or "").strip().lower()
        if not configured or not effective:
            continue
        consume(configured, effective)

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


def compute_gate_rollout_snapshot() -> Dict[str, Any]:
    gate_log_path = Path(os.getenv("CORTEX_GATE_LOG", str(DEFAULT_GATE_LOG)))
    rollout_state_path = Path(os.getenv("CORTEX_GATE_ROLLOUT_STATE_PATH", str(DEFAULT_ROLLOUT_STATE)))
    max_lines = int(float(os.getenv("PI_CYCLE_GATE_LOG_MAX_LINES", "5000")))
    window_hours = float(os.getenv("PI_CYCLE_GATE_METRIC_HOURS", "24"))
    window_hours = max(0.1, window_hours)
    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=window_hours)

    rollout_state = load_json_file(rollout_state_path) if rollout_state_path.exists() else {}
    configured_mode = str(rollout_state.get("rollout_mode_configured") or "")
    effective_mode = str(rollout_state.get("rollout_mode_effective") or "")

    summary: Dict[str, Any] = {
        "status": "unknown",
        "window_hours": window_hours,
        "gate_log_path": str(gate_log_path),
        "rollout_state_path": str(rollout_state_path),
        "rollout_mode_configured": configured_mode or None,
        "rollout_mode_effective": effective_mode or None,
        "rollout_started_ts": rollout_state.get("rollout_started_ts"),
        "total_events": 0,
        "act_events": 0,
        "would_block_count": 0,
        "cooldown_hits": 0,
        "recover_confirm_fail_count": 0,
        "would_block_rate": 0.0,
        "cooldown_hit_rate": 0.0,
        "recover_confirm_fail_rate": 0.0,
        "last_event_ts": None,
        "parse_error_count": 0,
    }

    if not gate_log_path.exists():
        summary["status"] = "unknown"
        return summary

    try:
        lines = gate_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        summary["status"] = "degraded"
        return summary

    events: List[Dict[str, Any]] = []
    for raw in reversed(lines[-max(1, max_lines) :]):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            summary["parse_error_count"] += 1
            continue
        if not isinstance(obj, dict):
            continue
        ts = parse_ts(obj.get("ts"))
        if ts is None:
            continue
        if ts < since:
            continue
        events.append(obj)
        if summary.get("rollout_mode_effective") is None and obj.get("rollout_mode_effective"):
            summary["rollout_mode_effective"] = obj.get("rollout_mode_effective")
        if summary.get("last_event_ts") is None:
            summary["last_event_ts"] = ts.isoformat()

    total_events = len(events)
    act_events = 0
    would_block_count = 0
    cooldown_hits = 0
    recover_confirm_fail_count = 0

    for ev in events:
        if str(ev.get("mode") or "") == "ACT":
            act_events += 1
        if bool(ev.get("would_block")):
            would_block_count += 1
        if bool(ev.get("cooldown_active")):
            cooldown_hits += 1

        passed = ev.get("recover_confirm_passed")
        ext = ev.get("gate_extensions") if isinstance(ev.get("gate_extensions"), dict) else {}
        not_applicable = bool(ext.get("recover_confirm_not_applicable"))
        if (passed is False) and (not not_applicable):
            recover_confirm_fail_count += 1

    denominator = act_events if act_events > 0 else total_events
    if denominator > 0:
        would_block_rate = round(would_block_count / denominator, 4)
        cooldown_hit_rate = round(cooldown_hits / denominator, 4)
        recover_confirm_fail_rate = round(recover_confirm_fail_count / denominator, 4)
    else:
        would_block_rate = 0.0
        cooldown_hit_rate = 0.0
        recover_confirm_fail_rate = 0.0

    summary.update(
        {
            "total_events": total_events,
            "act_events": act_events,
            "would_block_count": would_block_count,
            "cooldown_hits": cooldown_hits,
            "recover_confirm_fail_count": recover_confirm_fail_count,
            "would_block_rate": would_block_rate,
            "cooldown_hit_rate": cooldown_hit_rate,
            "recover_confirm_fail_rate": recover_confirm_fail_rate,
            "status": "ok" if total_events > 0 else "unknown",
        }
    )

    return summary


def compute_gate_notify_trend() -> Dict[str, Any]:
    path = Path(os.getenv("PI_CYCLE_LOG_JSONL", str(LOG_JSONL)))
    return compute_notify_trend(path)


def evaluate_gate_rollout_alert(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    min_events = int(float(os.getenv("PI_GATE_ALERT_MIN_EVENTS", "30")))
    wb_max = float(os.getenv("PI_GATE_WOULD_BLOCK_RATE_MAX", "0.12"))
    cd_max = float(os.getenv("PI_GATE_COOLDOWN_HIT_RATE_MAX", "0.20"))
    rc_max = float(os.getenv("PI_GATE_RECOVER_FAIL_RATE_MAX", "0.03"))

    effective = str(snapshot.get("rollout_mode_effective") or "").strip().lower()
    status = str(snapshot.get("status") or "").strip().lower()
    act_events = int(snapshot.get("act_events") or 0)
    total_events = int(snapshot.get("total_events") or 0)
    denominator = act_events if act_events > 0 else total_events

    alert: Dict[str, Any] = {
        "status": "unknown",
        "reason": "",
        "rollout_mode_effective": effective or None,
        "denominator_events": denominator,
        "min_events_required": min_events,
        "thresholds": {
            "would_block_rate_max": wb_max,
            "cooldown_hit_rate_max": cd_max,
            "recover_confirm_fail_rate_max": rc_max,
        },
        "breaches": [],
    }

    if status != "ok":
        alert["reason"] = "gate_rollout_unavailable"
        return alert
    if effective != "enforce":
        alert["reason"] = "rollout_not_enforce"
        return alert
    if denominator < max(1, min_events):
        alert["reason"] = "insufficient_events"
        return alert

    checks = [
        ("would_block_rate", float(snapshot.get("would_block_rate") or 0.0), wb_max),
        ("cooldown_hit_rate", float(snapshot.get("cooldown_hit_rate") or 0.0), cd_max),
        ("recover_confirm_fail_rate", float(snapshot.get("recover_confirm_fail_rate") or 0.0), rc_max),
    ]
    breaches = []
    for metric, value, threshold in checks:
        if value > threshold:
            breaches.append({"metric": metric, "value": round(value, 6), "threshold": float(threshold)})

    alert["breaches"] = breaches
    if not breaches:
        alert["status"] = "ok"
        alert["reason"] = "within_thresholds"
        return alert

    critical = len(breaches) >= 2 or any(
        b["metric"] == "recover_confirm_fail_rate" and float(b["value"]) >= 2.0 * float(b["threshold"])
        for b in breaches
    )
    alert["status"] = "critical" if critical else "degraded"
    alert["reason"] = "threshold_breached"
    return alert


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> bool:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(str(tmp), str(path))
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def write_text_atomic(path: Path, text: str) -> bool:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(path))
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def append_jsonl_with_dirs(path: Path, payload: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def _extract_patch_draft_action(pi_cycle_event: Dict[str, Any]) -> Dict[str, Any]:
    hints_raw = pi_cycle_event.get("guard_mode_patch_draft_action_hints")
    hints: List[str] = []
    if isinstance(hints_raw, list):
        for item in hints_raw[:5]:
            text = str(item or "").strip()
            if text:
                hints.append(text)
    status = str(pi_cycle_event.get("guard_mode_patch_draft_status") or "unknown").strip().lower()
    if status not in {"ok", "degraded", "critical", "unknown"}:
        status = "unknown"
    action_level = str(pi_cycle_event.get("guard_mode_patch_draft_action_level") or "observe").strip().lower()
    if action_level not in {"observe", "degrade", "shadow_lock"}:
        action_level = "observe"
    return {
        "status": status,
        "reason": str(pi_cycle_event.get("guard_mode_patch_draft_reason") or "unknown"),
        "pending": bool(pi_cycle_event.get("guard_mode_patch_draft_pending")),
        "action_level": action_level,
        "action_reason": str(pi_cycle_event.get("guard_mode_patch_draft_action_reason") or "unknown"),
        "next_action": str(pi_cycle_event.get("guard_mode_patch_draft_next_action") or "none"),
        "failed_jobs": max(0, int(pi_cycle_event.get("guard_mode_patch_draft_failed_jobs") or 0)),
        "actionable_jobs": max(0, int(pi_cycle_event.get("guard_mode_patch_draft_actionable_jobs") or 0)),
        "rollout_next_batch": (
            str(pi_cycle_event.get("guard_mode_patch_draft_rollout_next_batch") or "").strip() or None
        ),
        "rollout_strategy": (
            str(pi_cycle_event.get("guard_mode_patch_draft_rollout_strategy") or "").strip() or None
        ),
        "isolation_enabled": bool(pi_cycle_event.get("guard_mode_patch_draft_isolation_enabled", False)),
        "isolation_candidates_total": max(
            0, int(pi_cycle_event.get("guard_mode_patch_draft_isolation_candidates_total") or 0)
        ),
        "isolation_actionable_candidates": max(
            0, int(pi_cycle_event.get("guard_mode_patch_draft_isolation_actionable_candidates") or 0)
        ),
        "isolation_critical_candidates": max(
            0, int(pi_cycle_event.get("guard_mode_patch_draft_isolation_critical_candidates") or 0)
        ),
        "isolation_manual_only_candidates": max(
            0, int(pi_cycle_event.get("guard_mode_patch_draft_isolation_manual_only_candidates") or 0)
        ),
        "isolation_rollout_strategy": (
            str(pi_cycle_event.get("guard_mode_patch_draft_isolation_rollout_strategy") or "").strip() or None
        ),
        "isolation_rollout_next_batch": (
            str(pi_cycle_event.get("guard_mode_patch_draft_isolation_rollout_next_batch") or "").strip() or None
        ),
        "isolation_rollout_pending": bool(pi_cycle_event.get("guard_mode_patch_draft_isolation_rollout_pending")),
        "next_action_secondary": (
            str(pi_cycle_event.get("guard_mode_patch_draft_next_action_secondary") or "").strip() or None
        ),
        "hints": hints,
    }


def _extract_cron_policy_apply_batch_summary(pi_cycle_event: Dict[str, Any]) -> Dict[str, Any]:
    status = str(pi_cycle_event.get("cron_policy_apply_batch_status") or "unknown").strip().lower()
    if status not in {"ok", "degraded", "noop", "dry_run", "unknown"}:
        status = "unknown"
    return {
        "present": bool(pi_cycle_event.get("cron_policy_apply_batch_present", False)),
        "status": status,
        "reason": str(pi_cycle_event.get("cron_policy_apply_batch_reason") or "unknown"),
        "batch_id": str(pi_cycle_event.get("cron_policy_apply_batch_batch_id") or "").strip() or None,
        "mode": str(pi_cycle_event.get("cron_policy_apply_batch_mode") or "").strip() or None,
        "applied_to_source": bool(pi_cycle_event.get("cron_policy_apply_batch_applied_to_source", False)),
        "apply_skipped_reason": (
            str(pi_cycle_event.get("cron_policy_apply_batch_apply_skipped_reason") or "").strip() or None
        ),
        "selected_actions": max(0, int(pi_cycle_event.get("cron_policy_apply_batch_selected_actions") or 0)),
        "changed_jobs": max(0, int(pi_cycle_event.get("cron_policy_apply_batch_changed_jobs") or 0)),
        "failed_jobs": max(0, int(pi_cycle_event.get("cron_policy_apply_batch_failed_jobs") or 0)),
        "blocked_jobs": max(0, int(pi_cycle_event.get("cron_policy_apply_batch_blocked_jobs") or 0)),
        "operations_total": max(0, int(pi_cycle_event.get("cron_policy_apply_batch_operations_total") or 0)),
        "operations_applied": max(0, int(pi_cycle_event.get("cron_policy_apply_batch_operations_applied") or 0)),
        "operations_changed": max(0, int(pi_cycle_event.get("cron_policy_apply_batch_operations_changed") or 0)),
        "output_path": str(pi_cycle_event.get("cron_policy_apply_batch_output_path") or "").strip() or None,
        "candidate_jobs_output": (
            str(pi_cycle_event.get("cron_policy_apply_batch_candidate_jobs_output") or "").strip() or None
        ),
    }


def _extract_notify_trend_patch_action(notify_trend: Dict[str, Any]) -> Dict[str, Any]:
    source = notify_trend if isinstance(notify_trend, dict) else {}
    thresholds_raw = source.get("thresholds")
    thresholds = thresholds_raw if isinstance(thresholds_raw, dict) else {}
    patch_thresholds_raw = thresholds.get("patch_action")
    patch_thresholds = patch_thresholds_raw if isinstance(patch_thresholds_raw, dict) else {}

    status = str(source.get("patch_action_status") or "unknown").strip().lower()
    if status not in {"ok", "degraded", "critical", "unknown"}:
        status = "unknown"

    levels_top = source.get("patch_action_levels_top")
    reasons_top = source.get("patch_action_reasons_top")
    next_actions_top = source.get("patch_next_actions_top")
    hints_top = source.get("patch_action_hints_top")

    return {
        "status": status,
        "status_reason": str(source.get("patch_action_status_reason") or "unknown").strip() or "unknown",
        "events": max(0, _int_value(source.get("patch_action_events"), 0)),
        "escalated": max(0, _int_value(source.get("patch_action_escalated"), 0)),
        "escalated_rate": _as_float(source.get("patch_action_escalated_rate")),
        "pending": max(0, _int_value(source.get("patch_action_pending"), 0)),
        "pending_rate": _as_float(source.get("patch_action_pending_rate")),
        "dominant_next_action": str(source.get("patch_dominant_next_action") or "").strip() or None,
        "dominant_next_action_count": max(0, _int_value(source.get("patch_dominant_next_action_count"), 0)),
        "dominant_next_action_share": _as_float(source.get("patch_dominant_next_action_share")),
        "dominant_action_reason": str(source.get("patch_dominant_action_reason") or "").strip() or None,
        "dominant_action_reason_count": max(0, _int_value(source.get("patch_dominant_action_reason_count"), 0)),
        "dominant_action_reason_share": _as_float(source.get("patch_dominant_action_reason_share")),
        "recommended_next_action": str(source.get("patch_recommended_next_action") or "").strip() or None,
        "recommended_reason": str(source.get("patch_recommended_reason") or "").strip() or None,
        "recommended_confidence": _as_float(source.get("patch_recommended_confidence")),
        "levels_top": levels_top if isinstance(levels_top, list) else [],
        "reasons_top": reasons_top if isinstance(reasons_top, list) else [],
        "next_actions_top": next_actions_top if isinstance(next_actions_top, list) else [],
        "hints_top": hints_top if isinstance(hints_top, list) else [],
        "thresholds": patch_thresholds,
    }


def _extract_notify_trend_patch_apply(notify_trend: Dict[str, Any]) -> Dict[str, Any]:
    source = notify_trend if isinstance(notify_trend, dict) else {}
    thresholds_raw = source.get("thresholds")
    thresholds = thresholds_raw if isinstance(thresholds_raw, dict) else {}
    patch_apply_thresholds_raw = thresholds.get("patch_apply")
    patch_apply_thresholds = (
        patch_apply_thresholds_raw if isinstance(patch_apply_thresholds_raw, dict) else {}
    )

    status = str(source.get("patch_apply_status") or "unknown").strip().lower()
    if status not in {"ok", "degraded", "critical", "unknown"}:
        status = "unknown"

    statuses_top = source.get("patch_apply_statuses_top")
    reasons_top = source.get("patch_apply_reasons_top")
    modes_top = source.get("patch_apply_modes_top")

    return {
        "status": status,
        "status_reason": str(source.get("patch_apply_status_reason") or "unknown").strip() or "unknown",
        "events": max(0, _int_value(source.get("patch_apply_events"), 0)),
        "problem_events": max(0, _int_value(source.get("patch_apply_problem_events"), 0)),
        "problem_rate": _as_float(source.get("patch_apply_problem_rate")),
        "critical_problem_events": max(0, _int_value(source.get("patch_apply_critical_problem_events"), 0)),
        "critical_problem_rate": _as_float(source.get("patch_apply_critical_problem_rate")),
        "target_batch_id": str(source.get("patch_apply_target_batch_id") or "").strip() or None,
        "dominant_reason": str(source.get("patch_apply_dominant_reason") or "").strip() or None,
        "dominant_reason_count": max(0, _int_value(source.get("patch_apply_dominant_reason_count"), 0)),
        "dominant_reason_share": _as_float(source.get("patch_apply_dominant_reason_share")),
        "statuses_top": statuses_top if isinstance(statuses_top, list) else [],
        "reasons_top": reasons_top if isinstance(reasons_top, list) else [],
        "modes_top": modes_top if isinstance(modes_top, list) else [],
        "thresholds": patch_apply_thresholds,
    }


def _guard_reason_source(recommend_reason: Any) -> str:
    reason = str(recommend_reason or "").strip().lower()
    if reason.startswith("rollback_"):
        return "rollback"
    if reason.startswith("proxy_control_"):
        return "proxy_control"
    if reason.startswith("patch_apply_"):
        return "patch_apply"
    if reason.startswith("patch_"):
        return "patch_draft"
    if reason.startswith("digital_life_"):
        return "digital_life"
    if reason == "auto_schedule":
        return "schedule"
    if reason == "manual_guard_mode":
        return "manual"
    return "unknown"


def _extract_guard_mode_trace(pi_cycle_event: Dict[str, Any]) -> Dict[str, Any]:
    requested = str(pi_cycle_event.get("guard_mode_requested") or "unknown").strip().lower()
    base_effective = str(pi_cycle_event.get("guard_mode_base_effective") or "unknown").strip().lower()
    effective = str(pi_cycle_event.get("guard_mode_effective") or "unknown").strip().lower()
    recommended = str(pi_cycle_event.get("guard_mode_recommended") or "unknown").strip().lower()
    recommend_reason = str(pi_cycle_event.get("guard_mode_recommend_reason") or "").strip()

    rollback_status = str(pi_cycle_event.get("guard_mode_rollback_status") or "unknown").strip().lower()
    rollback_action = str(pi_cycle_event.get("guard_mode_rollback_action_level") or "observe").strip().lower()
    rollback_recommend = rollback_action in {"degrade", "shadow_lock"} or rollback_status == "critical"

    proxy_status = str(pi_cycle_event.get("guard_mode_proxy_control_trend_status") or "unknown").strip().lower()
    proxy_action = str(pi_cycle_event.get("guard_mode_proxy_control_trend_action_level") or "observe").strip().lower()
    proxy_trigger_count = max(0, _int_value(pi_cycle_event.get("guard_mode_proxy_control_trend_trigger_count"), 0))
    proxy_critical_count = max(
        0, _int_value(pi_cycle_event.get("guard_mode_proxy_control_trend_critical_trigger_count"), 0)
    )
    proxy_threshold = max(1, _int_value(pi_cycle_event.get("guard_mode_proxy_control_trigger_threshold"), 1))
    proxy_error_status = str(pi_cycle_event.get("guard_mode_proxy_error_status") or "unknown").strip().lower()
    proxy_error_trend_status = str(
        pi_cycle_event.get("guard_mode_proxy_error_trend_status") or "unknown"
    ).strip().lower()
    proxy_error_jobs_count = max(0, _int_value(pi_cycle_event.get("guard_mode_proxy_error_jobs_count"), 0))
    proxy_error_core_jobs_count = max(
        0, _int_value(pi_cycle_event.get("guard_mode_proxy_error_core_jobs_count"), 0)
    )
    proxy_error_trend_jobs_with_errors = max(
        0, _int_value(pi_cycle_event.get("guard_mode_proxy_error_trend_jobs_with_errors"), 0)
    )
    proxy_recommend = (
        proxy_action == "shadow_lock"
        or proxy_status == "critical"
        or (proxy_action == "degrade" and proxy_trigger_count >= proxy_threshold)
        or proxy_critical_count > 0
        or proxy_error_status == "critical"
        or proxy_error_trend_status == "critical"
        or (proxy_error_trend_status == "degraded" and proxy_error_trend_jobs_with_errors > 0)
    )

    patch_status = str(pi_cycle_event.get("guard_mode_patch_draft_status") or "unknown").strip().lower()
    patch_pending = bool(pi_cycle_event.get("guard_mode_patch_draft_pending"))
    patch_failed_jobs = max(0, _int_value(pi_cycle_event.get("guard_mode_patch_draft_failed_jobs"), 0))
    patch_actionable_jobs = max(0, _int_value(pi_cycle_event.get("guard_mode_patch_draft_actionable_jobs"), 0))
    patch_isolation_actionable = max(
        0, _int_value(pi_cycle_event.get("guard_mode_patch_draft_isolation_actionable_candidates"), 0)
    )
    patch_isolation_critical = max(
        0, _int_value(pi_cycle_event.get("guard_mode_patch_draft_isolation_critical_candidates"), 0)
    )
    patch_pending_threshold = max(1, _int_value(pi_cycle_event.get("guard_mode_patch_pending_threshold"), 2))
    patch_recommend = (
        patch_status == "critical"
        or patch_isolation_critical > 0
        or (patch_pending and patch_isolation_actionable > 0)
        or patch_failed_jobs > 0
        or (patch_pending and patch_status == "degraded")
        or (patch_pending and patch_status == "ok" and patch_actionable_jobs >= patch_pending_threshold)
    )
    patch_apply_status = str(pi_cycle_event.get("guard_mode_patch_apply_status") or "unknown").strip().lower()
    patch_apply_batch_id = str(pi_cycle_event.get("guard_mode_patch_apply_batch_id") or "").strip()
    patch_apply_target_batch = str(pi_cycle_event.get("guard_mode_patch_apply_target_batch") or "").strip()
    patch_apply_mode = str(pi_cycle_event.get("guard_mode_patch_apply_mode") or "").strip().lower()
    patch_apply_failed_jobs = max(0, _int_value(pi_cycle_event.get("guard_mode_patch_apply_failed_jobs"), 0))
    patch_apply_blocked_jobs = max(0, _int_value(pi_cycle_event.get("guard_mode_patch_apply_blocked_jobs"), 0))
    patch_apply_selected_actions = max(
        0, _int_value(pi_cycle_event.get("guard_mode_patch_apply_selected_actions"), 0)
    )
    patch_apply_changed_jobs = max(0, _int_value(pi_cycle_event.get("guard_mode_patch_apply_changed_jobs"), 0))
    patch_apply_operations_changed = max(
        0, _int_value(pi_cycle_event.get("guard_mode_patch_apply_operations_changed"), 0)
    )
    patch_apply_trend_status = str(
        pi_cycle_event.get("guard_mode_patch_apply_trend_status") or "unknown"
    ).strip().lower()
    patch_apply_recommend = (
        patch_apply_batch_id == patch_apply_target_batch
        and (
            patch_apply_status in {"critical", "degraded"}
            or patch_apply_trend_status in {"critical", "degraded"}
            or patch_apply_failed_jobs > 0
            or patch_apply_blocked_jobs > 0
            or (
                patch_apply_mode == "apply"
                and patch_apply_selected_actions > 0
                and patch_apply_changed_jobs <= 0
                and patch_apply_operations_changed <= 0
            )
        )
    )

    digital_status = str(pi_cycle_event.get("guard_mode_digital_life_status") or "unknown").strip().lower()
    digital_lifecycle = str(pi_cycle_event.get("guard_mode_digital_life_lifecycle_mode") or "").strip().upper()
    digital_recommend = digital_status == "critical" or (
        digital_status in {"ok", "degraded"} and digital_lifecycle in {"SURVIVE", "STABILIZE"}
    )

    priority_order = ["rollback", "proxy_control", "patch_draft", "patch_apply", "digital_life"]
    winner_source = _guard_reason_source(recommend_reason)
    return {
        "requested": requested,
        "base_effective": base_effective,
        "effective": effective,
        "recommended": recommended,
        "recommend_reason": recommend_reason,
        "winner_source": winner_source,
        "auto_override_applied": bool(pi_cycle_event.get("guard_mode_auto_override_applied")),
        "priority_order": priority_order,
        "source_recommendations": {
            "rollback": rollback_recommend,
            "proxy_control": proxy_recommend,
            "patch_draft": patch_recommend,
            "patch_apply": patch_apply_recommend,
            "digital_life": digital_recommend,
        },
        "signals": {
            "rollback": {
                "enabled": bool(pi_cycle_event.get("guard_mode_use_rollback_action")),
                "apply_enabled": bool(pi_cycle_event.get("guard_mode_apply_rollback_action")),
                "status": rollback_status,
                "action_level": rollback_action,
                "reason": str(pi_cycle_event.get("guard_mode_rollback_reason") or "unknown"),
            },
            "proxy_control": {
                "enabled": bool(pi_cycle_event.get("guard_mode_use_proxy_control_action")),
                "apply_enabled": bool(pi_cycle_event.get("guard_mode_apply_proxy_control_action")),
                "status": proxy_status,
                "action_level": proxy_action,
                "reason": str(pi_cycle_event.get("guard_mode_proxy_control_trend_reason") or "unknown"),
                "trigger_count": proxy_trigger_count,
                "critical_trigger_count": proxy_critical_count,
                "trigger_threshold": proxy_threshold,
                "proxy_error_status": proxy_error_status,
                "proxy_error_trend_status": proxy_error_trend_status,
                "proxy_error_jobs_count": proxy_error_jobs_count,
                "proxy_error_core_jobs_count": proxy_error_core_jobs_count,
                "proxy_error_trend_jobs_with_errors": proxy_error_trend_jobs_with_errors,
            },
            "patch_draft": {
                "enabled": bool(pi_cycle_event.get("guard_mode_use_patch_draft_action")),
                "apply_enabled": bool(pi_cycle_event.get("guard_mode_apply_patch_draft_action")),
                "status": patch_status,
                "pending": patch_pending,
                "failed_jobs": patch_failed_jobs,
                "actionable_jobs": patch_actionable_jobs,
                "isolation_actionable_candidates": patch_isolation_actionable,
                "isolation_critical_candidates": patch_isolation_critical,
                "pending_threshold": patch_pending_threshold,
                "reason": str(pi_cycle_event.get("guard_mode_patch_draft_reason") or "unknown"),
            },
            "patch_apply": {
                "enabled": bool(pi_cycle_event.get("guard_mode_use_patch_apply_action")),
                "apply_enabled": bool(pi_cycle_event.get("guard_mode_apply_patch_apply_action")),
                "target_batch": patch_apply_target_batch or None,
                "batch_id": patch_apply_batch_id or None,
                "status": patch_apply_status,
                "trend_status": patch_apply_trend_status,
                "mode": patch_apply_mode or None,
                "selected_actions": patch_apply_selected_actions,
                "changed_jobs": patch_apply_changed_jobs,
                "failed_jobs": patch_apply_failed_jobs,
                "blocked_jobs": patch_apply_blocked_jobs,
                "operations_changed": patch_apply_operations_changed,
                "reason": str(pi_cycle_event.get("guard_mode_patch_apply_reason") or "unknown"),
                "trend_reason": str(pi_cycle_event.get("guard_mode_patch_apply_trend_reason") or "unknown"),
            },
            "digital_life": {
                "enabled": bool(pi_cycle_event.get("guard_mode_use_digital_life_action")),
                "apply_enabled": bool(pi_cycle_event.get("guard_mode_apply_digital_life_action")),
                "status": digital_status,
                "lifecycle_mode": digital_lifecycle or None,
                "reason": str(pi_cycle_event.get("guard_mode_digital_life_reason") or "unknown"),
            },
        },
    }


def build_gate_alert_markdown(payload: Dict[str, Any]) -> str:
    alert = payload.get("gate_rollout_alert") if isinstance(payload.get("gate_rollout_alert"), dict) else {}
    gate = payload.get("gate_rollout") if isinstance(payload.get("gate_rollout"), dict) else {}
    guard_trace = payload.get("guard_mode_trace") if isinstance(payload.get("guard_mode_trace"), dict) else {}
    patch_action = (
        payload.get("patch_draft_action") if isinstance(payload.get("patch_draft_action"), dict) else {}
    )
    apply_batch = (
        payload.get("cron_policy_apply_batch")
        if isinstance(payload.get("cron_policy_apply_batch"), dict)
        else {}
    )
    notify_patch_action = (
        payload.get("notify_trend_patch_action")
        if isinstance(payload.get("notify_trend_patch_action"), dict)
        else {}
    )
    notify_patch_apply = (
        payload.get("notify_trend_patch_apply")
        if isinstance(payload.get("notify_trend_patch_apply"), dict)
        else {}
    )
    lines = [
        f"# Gate Rollout Alert ({payload.get('ts')})",
        "",
        f"- status: `{payload.get('status')}`",
        f"- reason: `{payload.get('reason')}`",
        f"- level_required: `{payload.get('level_required')}`",
        f"- pi_cycle_event_ts: `{payload.get('pi_cycle_event_ts')}`",
        "",
        "## Metrics",
        "",
        f"- rollout_mode_effective: `{gate.get('rollout_mode_effective')}`",
        f"- would_block_rate: `{gate.get('would_block_rate')}`",
        f"- cooldown_hit_rate: `{gate.get('cooldown_hit_rate')}`",
        f"- recover_confirm_fail_rate: `{gate.get('recover_confirm_fail_rate')}`",
        f"- events(act/total): `{gate.get('act_events')}` / `{gate.get('total_events')}`",
        "",
        "## Guard Mode Reason Chain",
        "",
        f"- requested/base/effective: "
        f"`{guard_trace.get('requested')}` / "
        f"`{guard_trace.get('base_effective')}` / "
        f"`{guard_trace.get('effective')}`",
        f"- recommended/reason/winner: "
        f"`{guard_trace.get('recommended')}` / "
        f"`{guard_trace.get('recommend_reason')}` / "
        f"`{guard_trace.get('winner_source')}`",
        f"- auto_override_applied: `{guard_trace.get('auto_override_applied')}`",
        f"- priority_order: `{' > '.join(guard_trace.get('priority_order') or [])}`",
        "",
        "## Patch Draft Action",
        "",
        f"- patch_status: `{patch_action.get('status')}`",
        f"- patch_reason: `{patch_action.get('reason')}`",
        f"- pending: `{patch_action.get('pending')}`",
        f"- action_level: `{patch_action.get('action_level')}`",
        f"- action_reason: `{patch_action.get('action_reason')}`",
        f"- next_action: `{patch_action.get('next_action')}`",
        f"- next_action_secondary: `{patch_action.get('next_action_secondary')}`",
        f"- failed_jobs: `{patch_action.get('failed_jobs')}`",
        f"- actionable_jobs: `{patch_action.get('actionable_jobs')}`",
        f"- rollout_next_batch: `{patch_action.get('rollout_next_batch')}`",
        f"- rollout_strategy: `{patch_action.get('rollout_strategy')}`",
        f"- isolation(candidates/actionable/critical/manual): "
        f"`{patch_action.get('isolation_candidates_total')}` / "
        f"`{patch_action.get('isolation_actionable_candidates')}` / "
        f"`{patch_action.get('isolation_critical_candidates')}` / "
        f"`{patch_action.get('isolation_manual_only_candidates')}`",
        f"- isolation_rollout(next/strategy/pending): "
        f"`{patch_action.get('isolation_rollout_next_batch')}` / "
        f"`{patch_action.get('isolation_rollout_strategy')}` / "
        f"`{patch_action.get('isolation_rollout_pending')}`",
        "",
        "### Patch Hints",
        "",
    ]
    patch_hints = patch_action.get("hints") if isinstance(patch_action.get("hints"), list) else []
    if not patch_hints:
        lines.append("- none")
    else:
        for hint in patch_hints:
            lines.append(f"- {hint}")
    lines.append("")
    source_recs = (
        guard_trace.get("source_recommendations")
        if isinstance(guard_trace.get("source_recommendations"), dict)
        else {}
    )
    lines.append(
        f"- source_recommendations(rollback/proxy/patch/patch_apply/digital): "
        f"`{source_recs.get('rollback')}` / "
        f"`{source_recs.get('proxy_control')}` / "
        f"`{source_recs.get('patch_draft')}` / "
        f"`{source_recs.get('patch_apply')}` / "
        f"`{source_recs.get('digital_life')}`"
    )
    signals = guard_trace.get("signals") if isinstance(guard_trace.get("signals"), dict) else {}
    rollback_sig = signals.get("rollback") if isinstance(signals.get("rollback"), dict) else {}
    proxy_sig = signals.get("proxy_control") if isinstance(signals.get("proxy_control"), dict) else {}
    patch_sig = signals.get("patch_draft") if isinstance(signals.get("patch_draft"), dict) else {}
    patch_apply_sig = signals.get("patch_apply") if isinstance(signals.get("patch_apply"), dict) else {}
    digital_sig = signals.get("digital_life") if isinstance(signals.get("digital_life"), dict) else {}
    lines.append(
        f"- signal_rollback(enabled/status/action): "
        f"`{rollback_sig.get('enabled')}` / "
        f"`{rollback_sig.get('status')}` / "
        f"`{rollback_sig.get('action_level')}`"
    )
    lines.append(
        f"- signal_proxy(enabled/status/action/triggers): "
        f"`{proxy_sig.get('enabled')}` / "
        f"`{proxy_sig.get('status')}` / "
        f"`{proxy_sig.get('action_level')}` / "
        f"`{proxy_sig.get('trigger_count')}`"
    )
    lines.append(
        f"- signal_patch(enabled/status/pending/failed): "
        f"`{patch_sig.get('enabled')}` / "
        f"`{patch_sig.get('status')}` / "
        f"`{patch_sig.get('pending')}` / "
        f"`{patch_sig.get('failed_jobs')}`"
    )
    lines.append(
        f"- signal_patch_apply(enabled/status/trend/batch/failed/blocked): "
        f"`{patch_apply_sig.get('enabled')}` / "
        f"`{patch_apply_sig.get('status')}` / "
        f"`{patch_apply_sig.get('trend_status')}` / "
        f"`{patch_apply_sig.get('batch_id')}` / "
        f"`{patch_apply_sig.get('failed_jobs')}` / "
        f"`{patch_apply_sig.get('blocked_jobs')}`"
    )
    lines.append(
        f"- signal_digital(enabled/status/lifecycle): "
        f"`{digital_sig.get('enabled')}` / "
        f"`{digital_sig.get('status')}` / "
        f"`{digital_sig.get('lifecycle_mode')}`"
    )
    lines.append("")
    lines.append("## Notify Trend Patch (24h)")
    lines.append("")
    lines.append(
        f"- patch_action(status/reason/events/escalated_rate/pending_rate): "
        f"`{notify_patch_action.get('status')}` / "
        f"`{notify_patch_action.get('status_reason')}` / "
        f"`{notify_patch_action.get('events')}` / "
        f"`{notify_patch_action.get('escalated_rate')}` / "
        f"`{notify_patch_action.get('pending_rate')}`"
    )
    lines.append(
        f"- patch_next_action(dominant/recommended/reason/confidence): "
        f"`{notify_patch_action.get('dominant_next_action')}` / "
        f"`{notify_patch_action.get('recommended_next_action')}` / "
        f"`{notify_patch_action.get('recommended_reason')}` / "
        f"`{notify_patch_action.get('recommended_confidence')}`"
    )
    lines.append(
        f"- patch_action_reason(dominant/count/share): "
        f"`{notify_patch_action.get('dominant_action_reason')}` / "
        f"`{notify_patch_action.get('dominant_action_reason_count')}` / "
        f"`{notify_patch_action.get('dominant_action_reason_share')}`"
    )
    lines.append(
        f"- patch_apply(status/reason/events/problem_rate/critical_rate): "
        f"`{notify_patch_apply.get('status')}` / "
        f"`{notify_patch_apply.get('status_reason')}` / "
        f"`{notify_patch_apply.get('events')}` / "
        f"`{notify_patch_apply.get('problem_rate')}` / "
        f"`{notify_patch_apply.get('critical_problem_rate')}`"
    )
    lines.append(
        f"- patch_apply_dominant(reason/count/share/target_batch): "
        f"`{notify_patch_apply.get('dominant_reason')}` / "
        f"`{notify_patch_apply.get('dominant_reason_count')}` / "
        f"`{notify_patch_apply.get('dominant_reason_share')}` / "
        f"`{notify_patch_apply.get('target_batch_id')}`"
    )
    patch_thresholds = (
        notify_patch_action.get("thresholds")
        if isinstance(notify_patch_action.get("thresholds"), dict)
        else {}
    )
    if patch_thresholds:
        lines.append(
            f"- patch_thresholds: "
            f"`min_events={patch_thresholds.get('min_events')}` / "
            f"`rate_degraded={patch_thresholds.get('rate_degraded')}` / "
            f"`rate_critical={patch_thresholds.get('rate_critical')}`"
        )
    levels_top = notify_patch_action.get("levels_top") if isinstance(notify_patch_action.get("levels_top"), list) else []
    if levels_top:
        rendered = ", ".join(
            f"{item.get('level')}:{item.get('count')}" for item in levels_top[:3] if isinstance(item, dict)
        )
        if rendered:
            lines.append(f"- patch_level_top: `{rendered}`")
    reasons_top = notify_patch_action.get("reasons_top") if isinstance(notify_patch_action.get("reasons_top"), list) else []
    if reasons_top:
        rendered = ", ".join(
            f"{item.get('reason')}:{item.get('count')}" for item in reasons_top[:3] if isinstance(item, dict)
        )
        if rendered:
            lines.append(f"- patch_reason_top: `{rendered}`")
    next_top = notify_patch_action.get("next_actions_top") if isinstance(notify_patch_action.get("next_actions_top"), list) else []
    if next_top:
        rendered = ", ".join(
            f"{item.get('next_action')}:{item.get('count')}" for item in next_top[:3] if isinstance(item, dict)
        )
        if rendered:
            lines.append(f"- patch_next_action_top: `{rendered}`")
    hint_top = notify_patch_action.get("hints_top") if isinstance(notify_patch_action.get("hints_top"), list) else []
    if hint_top:
        rendered = " | ".join(
            f"{item.get('hint')}({item.get('count')})" for item in hint_top[:3] if isinstance(item, dict)
        )
        if rendered:
            lines.append(f"- patch_hint_top: {rendered}")
    lines.append("")
    lines.append("## Apply Batch")
    lines.append("")
    lines.append(
        f"- apply_batch(status/reason/present/batch/mode): "
        f"`{apply_batch.get('status')}` / "
        f"`{apply_batch.get('reason')}` / "
        f"`{apply_batch.get('present')}` / "
        f"`{apply_batch.get('batch_id')}` / "
        f"`{apply_batch.get('mode')}`"
    )
    lines.append(
        f"- apply_batch(summary selected/changed/failed/blocked/ops_changed): "
        f"`{apply_batch.get('selected_actions')}` / "
        f"`{apply_batch.get('changed_jobs')}` / "
        f"`{apply_batch.get('failed_jobs')}` / "
        f"`{apply_batch.get('blocked_jobs')}` / "
        f"`{apply_batch.get('operations_changed')}`"
    )
    lines.append(
        f"- apply_batch(write/applied/skipped): "
        f"`{apply_batch.get('applied_to_source')}` / "
        f"`{apply_batch.get('apply_skipped_reason')}` / "
        f"`{apply_batch.get('output_path')}`"
    )
    lines.append("")
    lines.append("## Breaches")
    lines.append("")
    breaches = alert.get("breaches") if isinstance(alert.get("breaches"), list) else []
    if not breaches:
        lines.append("- none")
    else:
        for b in breaches:
            if isinstance(b, dict):
                lines.append(
                    f"- `{b.get('metric')}`: value=`{b.get('value')}` threshold=`{b.get('threshold')}`"
                )
    lines.append("")
    return "\n".join(lines)


def build_gate_alert_text(payload: Dict[str, Any]) -> str:
    alert = payload.get("gate_rollout_alert") if isinstance(payload.get("gate_rollout_alert"), dict) else {}
    gate = payload.get("gate_rollout") if isinstance(payload.get("gate_rollout"), dict) else {}
    guard_trace = payload.get("guard_mode_trace") if isinstance(payload.get("guard_mode_trace"), dict) else {}
    patch_action = (
        payload.get("patch_draft_action") if isinstance(payload.get("patch_draft_action"), dict) else {}
    )
    apply_batch = (
        payload.get("cron_policy_apply_batch")
        if isinstance(payload.get("cron_policy_apply_batch"), dict)
        else {}
    )
    notify_patch_action = (
        payload.get("notify_trend_patch_action")
        if isinstance(payload.get("notify_trend_patch_action"), dict)
        else {}
    )
    notify_patch_apply = (
        payload.get("notify_trend_patch_apply")
        if isinstance(payload.get("notify_trend_patch_apply"), dict)
        else {}
    )
    breaches = alert.get("breaches") if isinstance(alert.get("breaches"), list) else []
    breach_text = "none"
    if breaches:
        parts: List[str] = []
        for b in breaches[:5]:
            if not isinstance(b, dict):
                continue
            parts.append(
                f"{b.get('metric')}={b.get('value')}>{b.get('threshold')}"
            )
        breach_text = "; ".join(parts) if parts else "none"

    lines = [
        "Pi Gate Rollout Alert",
        f"status={payload.get('status')} reason={payload.get('reason')} level={payload.get('level_required')}",
        f"rollout_mode={gate.get('rollout_mode_effective')} act_events={gate.get('act_events')} total_events={gate.get('total_events')}",
        f"would_block_rate={gate.get('would_block_rate')} cooldown_hit_rate={gate.get('cooldown_hit_rate')} recover_confirm_fail_rate={gate.get('recover_confirm_fail_rate')}",
        f"guard_mode=requested:{guard_trace.get('requested')} base:{guard_trace.get('base_effective')} effective:{guard_trace.get('effective')} recommended:{guard_trace.get('recommended')} reason:{guard_trace.get('recommend_reason')} winner:{guard_trace.get('winner_source')}",
        f"patch_action={patch_action.get('action_level')}:{patch_action.get('action_reason')} next={patch_action.get('next_action')} next2={patch_action.get('next_action_secondary')} status={patch_action.get('status')} pending={patch_action.get('pending')} failed_jobs={patch_action.get('failed_jobs')}",
        f"patch_isolation=actionable:{patch_action.get('isolation_actionable_candidates')} critical:{patch_action.get('isolation_critical_candidates')} next={patch_action.get('isolation_rollout_next_batch')}",
        f"apply_batch=status:{apply_batch.get('status')} reason:{apply_batch.get('reason')} batch:{apply_batch.get('batch_id')} mode:{apply_batch.get('mode')} selected:{apply_batch.get('selected_actions')} changed:{apply_batch.get('changed_jobs')} failed:{apply_batch.get('failed_jobs')}",
        f"notify_patch_recommend={notify_patch_action.get('recommended_next_action')} reason={notify_patch_action.get('recommended_reason')} confidence={notify_patch_action.get('recommended_confidence')} status={notify_patch_action.get('status')}",
        f"notify_patch_apply=status:{notify_patch_apply.get('status')} reason:{notify_patch_apply.get('status_reason')} events:{notify_patch_apply.get('events')} problem_rate:{notify_patch_apply.get('problem_rate')} critical_rate:{notify_patch_apply.get('critical_problem_rate')} dominant:{notify_patch_apply.get('dominant_reason')}",
        f"breaches={breach_text}",
        f"ts={payload.get('ts')}",
    ]
    return "\n".join(lines)


def _post_json(url: str, payload: Dict[str, Any], timeout_sec: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
        status = int(getattr(resp, "status", 0) or resp.getcode() or 0)
        body = resp.read().decode("utf-8", errors="ignore")
    try:
        parsed = json.loads(body) if body else {}
    except Exception:
        parsed = {"raw": body[:500]}
    return {"http_status": status, "body": parsed}


def maybe_notify_gate_rollout_alert(payload: Dict[str, Any]) -> Dict[str, Any]:
    enabled = env_flag("PI_GATE_ALERT_NOTIFY_ENABLED", default=False)
    channel = str(os.getenv("PI_GATE_ALERT_NOTIFY_CHANNEL", "feishu")).strip().lower()
    notify_level = str(os.getenv("PI_GATE_ALERT_NOTIFY_LEVEL", "critical")).strip().lower()
    if notify_level not in {"critical", "degraded"}:
        notify_level = "critical"
    alert_status = str(payload.get("status") or "").strip().lower()
    out: Dict[str, Any] = {
        "enabled": enabled,
        "channel": channel,
        "level_required": notify_level,
        "alert_status": alert_status or None,
        "sent": False,
        "reason": "",
    }
    if not enabled:
        out["reason"] = "disabled"
        return out
    if channel != "feishu":
        out["reason"] = "unsupported_channel"
        return out
    if alert_status not in {"critical", "degraded"}:
        out["reason"] = "unknown_alert_status"
        return out
    if (notify_level == "critical" and alert_status != "critical") or (
        notify_level == "degraded" and alert_status not in {"critical", "degraded"}
    ):
        out["reason"] = "below_level"
        return out

    webhook = str(os.getenv("PI_GATE_ALERT_FEISHU_WEBHOOK", "")).strip()
    if not webhook:
        out["reason"] = "no_webhook"
        return out

    min_interval_sec = int(float(os.getenv("PI_GATE_ALERT_NOTIFY_MIN_INTERVAL_SEC", "1200")))
    state_path = Path(
        os.getenv("PI_GATE_ALERT_NOTIFY_STATE_PATH", str(DEFAULT_GATE_ALERT_NOTIFY_STATE))
    )
    history_path = Path(
        os.getenv("PI_GATE_ALERT_NOTIFY_HISTORY_JSONL", str(DEFAULT_GATE_ALERT_NOTIFY_HISTORY))
    )
    state = load_json_file(state_path) if state_path.exists() else {}
    now = dt.datetime.now(dt.timezone.utc)
    last_ts = parse_ts(state.get("last_notify_ts"))
    if last_ts is not None:
        elapsed = (now - last_ts).total_seconds()
        if elapsed < max(0, min_interval_sec):
            out["reason"] = "cooldown"
            out["cooldown_remaining_sec"] = int(max(0, min_interval_sec - elapsed))
            out["state_path"] = str(state_path)
            return out

    timeout_sec = int(float(os.getenv("PI_GATE_ALERT_NOTIFY_TIMEOUT_SEC", "8")))
    max_retries = int(float(os.getenv("PI_GATE_ALERT_NOTIFY_MAX_RETRIES", "0")))
    retry_backoff_sec = float(os.getenv("PI_GATE_ALERT_NOTIFY_RETRY_BACKOFF_SEC", "0.8"))
    max_retries = max(0, min(max_retries, 3))
    message = {
        "msg_type": "text",
        "content": {"text": build_gate_alert_text(payload)},
    }
    attempts: List[Dict[str, Any]] = []
    last_reason = "send_failed"
    for i in range(0, max_retries + 1):
        try:
            posted = _post_json(webhook, message, timeout_sec=timeout_sec)
            http_status = int(posted.get("http_status") or 0)
            body = posted.get("body")
            remote_code: Optional[int] = None
            if isinstance(body, dict) and body.get("code") is not None:
                try:
                    remote_code = int(body.get("code"))
                except Exception:
                    remote_code = None
            remote_ok = (remote_code == 0) if remote_code is not None else True
            ok = (200 <= http_status < 300) and remote_ok
            attempts.append(
                {
                    "attempt": i + 1,
                    "http_status": http_status,
                    "response_code": remote_code,
                    "ok": bool(ok),
                }
            )
            if ok:
                out.update(
                    {
                        "sent": True,
                        "reason": "sent",
                        "http_status": http_status,
                        "response_code": remote_code,
                        "attempt_count": i + 1,
                        "state_path": str(state_path),
                    }
                )
                write_json_atomic(
                    state_path,
                    {
                        "last_notify_ts": now.isoformat(),
                        "last_notify_status": alert_status,
                        "last_notify_reason": payload.get("reason"),
                    },
                )
                append_jsonl_with_dirs(
                    history_path,
                    {
                        "envelope_version": "1.0",
                        "domain": "gate_rollout_alert_notify",
                        "ts": now.isoformat(),
                        "status": "sent",
                        "reason": payload.get("reason"),
                        "alert_status": alert_status,
                        "level_required": notify_level,
                        "attempts": attempts,
                    },
                )
                return out
            last_reason = "remote_error"
            if isinstance(body, dict):
                out["response_msg"] = str(body.get("msg") or "")[:200]
        except urllib.error.HTTPError as e:
            attempts.append({"attempt": i + 1, "http_status": int(getattr(e, "code", 0) or 0), "ok": False})
            last_reason = "http_error"
        except Exception as e:
            attempts.append({"attempt": i + 1, "error": type(e).__name__, "ok": False})
            last_reason = f"send_failed:{type(e).__name__}"

        if i < max_retries:
            time.sleep(max(0.05, retry_backoff_sec) * (2**i))

    out.update(
        {
            "sent": False,
            "reason": last_reason,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "state_path": str(state_path),
        }
    )
    append_jsonl_with_dirs(
        history_path,
        {
            "envelope_version": "1.0",
            "domain": "gate_rollout_alert_notify",
            "ts": now.isoformat(),
            "status": "failed",
            "reason": last_reason,
            "alert_status": alert_status,
            "level_required": notify_level,
            "attempts": attempts,
        },
    )
    return out


def maybe_emit_gate_rollout_alert(
    *,
    pi_cycle_event: Dict[str, Any],
    gate_rollout: Dict[str, Any],
    gate_rollout_alert: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    enabled = env_flag("PI_GATE_ALERT_EMIT", default=True)
    level_required = str(os.getenv("PI_GATE_ALERT_LEVEL", "critical")).strip().lower()
    if level_required not in {"critical", "degraded"}:
        level_required = "critical"

    out: Dict[str, Any] = {
        "enabled": enabled,
        "emitted": False,
        "reason": "",
        "level_required": level_required,
        "dry_run": bool(dry_run),
        "suppressed": False,
        "suppression_reason": "",
        "gate_notify_trend_status": None,
        "notify": {
            "enabled": env_flag("PI_GATE_ALERT_NOTIFY_ENABLED", default=False),
            "channel": str(os.getenv("PI_GATE_ALERT_NOTIFY_CHANNEL", "feishu")).strip().lower(),
            "sent": False,
            "reason": "not_attempted",
        },
    }
    patch_draft_action = _extract_patch_draft_action(pi_cycle_event)
    apply_batch_summary = _extract_cron_policy_apply_batch_summary(pi_cycle_event)
    guard_mode_trace = _extract_guard_mode_trace(pi_cycle_event)
    out["patch_draft_action_level"] = patch_draft_action.get("action_level")
    out["patch_draft_action_reason"] = patch_draft_action.get("action_reason")
    out["patch_draft_next_action"] = patch_draft_action.get("next_action")
    out["patch_draft_action_hints"] = patch_draft_action.get("hints")
    out["patch_draft_status"] = patch_draft_action.get("status")
    out["patch_draft_pending"] = patch_draft_action.get("pending")
    out["patch_draft_failed_jobs"] = patch_draft_action.get("failed_jobs")
    out["patch_draft_actionable_jobs"] = patch_draft_action.get("actionable_jobs")
    out["patch_draft_rollout_next_batch"] = patch_draft_action.get("rollout_next_batch")
    out["patch_draft_rollout_strategy"] = patch_draft_action.get("rollout_strategy")
    out["cron_policy_apply_batch_present"] = apply_batch_summary.get("present")
    out["cron_policy_apply_batch_status"] = apply_batch_summary.get("status")
    out["cron_policy_apply_batch_reason"] = apply_batch_summary.get("reason")
    out["cron_policy_apply_batch_batch_id"] = apply_batch_summary.get("batch_id")
    out["cron_policy_apply_batch_mode"] = apply_batch_summary.get("mode")
    out["cron_policy_apply_batch_applied_to_source"] = apply_batch_summary.get("applied_to_source")
    out["cron_policy_apply_batch_apply_skipped_reason"] = apply_batch_summary.get("apply_skipped_reason")
    out["cron_policy_apply_batch_selected_actions"] = apply_batch_summary.get("selected_actions")
    out["cron_policy_apply_batch_changed_jobs"] = apply_batch_summary.get("changed_jobs")
    out["cron_policy_apply_batch_failed_jobs"] = apply_batch_summary.get("failed_jobs")
    out["cron_policy_apply_batch_blocked_jobs"] = apply_batch_summary.get("blocked_jobs")
    out["cron_policy_apply_batch_operations_total"] = apply_batch_summary.get("operations_total")
    out["cron_policy_apply_batch_operations_applied"] = apply_batch_summary.get("operations_applied")
    out["cron_policy_apply_batch_operations_changed"] = apply_batch_summary.get("operations_changed")
    out["cron_policy_apply_batch_output_path"] = apply_batch_summary.get("output_path")
    out["cron_policy_apply_batch_candidate_jobs_output"] = apply_batch_summary.get(
        "candidate_jobs_output"
    )
    out["guard_mode_recommended"] = guard_mode_trace.get("recommended")
    out["guard_mode_recommend_reason"] = guard_mode_trace.get("recommend_reason")
    out["guard_mode_reason_winner"] = guard_mode_trace.get("winner_source")
    out["guard_mode_priority_order"] = guard_mode_trace.get("priority_order")
    out["guard_mode_source_recommendations"] = guard_mode_trace.get("source_recommendations")
    if not enabled:
        out["reason"] = "disabled"
        return out

    alert_status = str(gate_rollout_alert.get("status") or "").strip().lower()
    eligible = (alert_status == "critical") or (level_required == "degraded" and alert_status in {"critical", "degraded"})
    if not eligible:
        out["reason"] = "below_level"
        out["alert_status"] = alert_status or None
        return out

    notify_trend = (
        pi_cycle_event.get("gate_notify_trend") if isinstance(pi_cycle_event.get("gate_notify_trend"), dict) else {}
    )
    notify_trend_patch_action = _extract_notify_trend_patch_action(notify_trend)
    notify_trend_patch_apply = _extract_notify_trend_patch_apply(notify_trend)
    notify_trend_status = str(notify_trend.get("status") or "").strip().lower()
    notify_trend_status_reason = str(notify_trend.get("status_reason") or "").strip()
    notify_trend_component_score = notify_trend.get("component_score")
    notify_trend_component_score_status = str(
        notify_trend.get("component_score_status") or ""
    ).strip().lower()
    trend_components = {
        "fail": str(notify_trend.get("fail_status") or "").strip().lower(),
        "suppression": str(notify_trend.get("suppression_status") or "").strip().lower(),
        "min_interval": str(notify_trend.get("min_interval_status") or "").strip().lower(),
        "patch_apply": str(notify_trend_patch_apply.get("status") or "").strip().lower(),
    }
    rank = {"unknown": 0, "ok": 1, "degraded": 2, "critical": 3}
    trigger_component = "aggregate"
    trigger_level = "unknown"
    for name in ["fail", "suppression", "min_interval", "patch_apply"]:
        st = trend_components.get(name, "")
        if rank.get(st, 0) > rank.get(trigger_level, 0):
            trigger_level = st
            trigger_component = name
    if notify_trend_status in {"critical", "degraded"} and rank.get(trigger_level, 0) < rank.get(
        notify_trend_status,
        0,
    ):
        trigger_component = "aggregate"
        trigger_level = notify_trend_status
    component_mode = resolve_component_score_mode(notify_trend=notify_trend, dry_run=bool(dry_run))
    component_score_mode = str(component_mode.get("effective_mode") or "off").strip().lower()
    effective_notify_trend_status = notify_trend_status
    effective_notify_trend_source = "base"
    component_score_would_upgrade = (
        notify_trend_component_score_status in {"ok", "degraded", "critical"}
        and rank.get(notify_trend_component_score_status, 0) > rank.get(effective_notify_trend_status, 0)
    )
    if component_score_mode == "enforce" and component_score_would_upgrade:
        effective_notify_trend_status = notify_trend_component_score_status
        effective_notify_trend_source = "component_score"
        trigger_component = "component_score"
        trigger_level = notify_trend_component_score_status
    if effective_notify_trend_status in {"ok", "degraded", "critical"}:
        out["gate_notify_trend_status"] = effective_notify_trend_status
    if notify_trend_status in {"ok", "degraded", "critical"}:
        out["gate_notify_trend_status_base"] = notify_trend_status
    out["gate_notify_trend_component_score_mode_configured"] = component_mode.get("configured_mode")
    out["gate_notify_trend_component_score_mode_effective"] = component_score_mode
    out["gate_notify_trend_component_score_mode_source"] = component_mode.get("mode_source")
    out["gate_notify_trend_component_score_auto_switch_enabled"] = bool(
        component_mode.get("auto_switch_enabled")
    )
    out["gate_notify_trend_component_score_auto_recommend_enforce"] = component_mode.get(
        "auto_recommend_enforce"
    )
    out["gate_notify_trend_component_score_auto_recommend_reason"] = component_mode.get(
        "auto_recommend_reason"
    )
    out["gate_notify_trend_component_score_auto_stable_windows"] = component_mode.get(
        "auto_stable_windows"
    )
    out["gate_notify_trend_component_score_auto_unstable_windows"] = component_mode.get(
        "auto_unstable_windows"
    )
    out["gate_notify_trend_component_score_auto_promote_windows"] = component_mode.get(
        "auto_promote_windows"
    )
    out["gate_notify_trend_component_score_auto_demote_windows"] = component_mode.get(
        "auto_demote_windows"
    )
    out["gate_notify_trend_component_score_auto_state_path"] = component_mode.get("auto_state_path")
    if component_mode.get("auto_transition"):
        out["gate_notify_trend_component_score_auto_transition"] = component_mode.get("auto_transition")
    if bool(component_mode.get("auto_state_io_degraded")):
        out["gate_notify_trend_component_score_auto_state_io_degraded"] = True
    component_mode_drift_signal = resolve_component_mode_drift_signal(
        log_path=Path(os.getenv("PI_CYCLE_LOG_JSONL", str(LOG_JSONL))),
        current_configured_mode=str(component_mode.get("configured_mode") or ""),
        current_effective_mode=component_score_mode,
    )
    component_mode_drift_status = str(component_mode_drift_signal.get("status") or "unknown").strip().lower()
    if component_mode_drift_status not in {"ok", "degraded", "critical", "unknown"}:
        component_mode_drift_status = "unknown"
    out["gate_notify_trend_component_score_mode_drift_status"] = component_mode_drift_status
    out["gate_notify_trend_component_score_mode_drift_reason"] = component_mode_drift_signal.get("reason")
    out["gate_notify_trend_component_score_mode_drift_window_hours"] = component_mode_drift_signal.get(
        "window_hours"
    )
    out["gate_notify_trend_component_score_mode_drift_mode_events"] = component_mode_drift_signal.get(
        "mode_events"
    )
    out["gate_notify_trend_component_score_mode_drift_events"] = component_mode_drift_signal.get(
        "drift_events"
    )
    out["gate_notify_trend_component_score_mode_drift_rate"] = component_mode_drift_signal.get(
        "drift_rate"
    )
    out["gate_notify_trend_component_score_mode_drift_latest_consecutive"] = component_mode_drift_signal.get(
        "latest_consecutive_drift"
    )
    out["gate_notify_trend_component_score_mode_drift_latest_drift"] = component_mode_drift_signal.get(
        "latest_drift"
    )
    out["gate_notify_trend_component_score_mode_drift_top_pairs"] = component_mode_drift_signal.get(
        "top_drift_pairs"
    )
    out["gate_notify_trend_component_score_mode_drift_thresholds"] = component_mode_drift_signal.get(
        "thresholds"
    )
    out["gate_notify_trend_status_source"] = effective_notify_trend_source
    out["gate_notify_trend_use_component_score_status"] = component_score_mode == "enforce"
    out["gate_notify_trend_component_score_shadow_would_upgrade"] = bool(component_score_would_upgrade)
    if component_score_would_upgrade:
        out["gate_notify_trend_component_score_shadow_target_status"] = notify_trend_component_score_status
    if notify_trend_component_score_status in {"ok", "degraded", "critical"}:
        out["gate_notify_trend_component_score_status"] = notify_trend_component_score_status
    if notify_trend_component_score is not None:
        out["gate_notify_trend_component_score"] = notify_trend_component_score
    if notify_trend_status_reason:
        out["gate_notify_trend_status_reason"] = notify_trend_status_reason
    out["gate_notify_trend_components"] = trend_components
    out["gate_notify_trend_trigger_component"] = trigger_component
    out["gate_notify_trend_trigger_level"] = trigger_level
    out["gate_notify_trend_patch_action_status"] = notify_trend_patch_action.get("status")
    out["gate_notify_trend_patch_action_status_reason"] = notify_trend_patch_action.get("status_reason")
    out["gate_notify_trend_patch_action_events"] = notify_trend_patch_action.get("events")
    out["gate_notify_trend_patch_action_escalated"] = notify_trend_patch_action.get("escalated")
    out["gate_notify_trend_patch_action_escalated_rate"] = notify_trend_patch_action.get("escalated_rate")
    out["gate_notify_trend_patch_action_pending"] = notify_trend_patch_action.get("pending")
    out["gate_notify_trend_patch_action_pending_rate"] = notify_trend_patch_action.get("pending_rate")
    out["gate_notify_trend_patch_dominant_next_action"] = notify_trend_patch_action.get("dominant_next_action")
    out["gate_notify_trend_patch_dominant_next_action_count"] = notify_trend_patch_action.get(
        "dominant_next_action_count"
    )
    out["gate_notify_trend_patch_dominant_next_action_share"] = notify_trend_patch_action.get(
        "dominant_next_action_share"
    )
    out["gate_notify_trend_patch_dominant_action_reason"] = notify_trend_patch_action.get(
        "dominant_action_reason"
    )
    out["gate_notify_trend_patch_dominant_action_reason_count"] = notify_trend_patch_action.get(
        "dominant_action_reason_count"
    )
    out["gate_notify_trend_patch_dominant_action_reason_share"] = notify_trend_patch_action.get(
        "dominant_action_reason_share"
    )
    out["gate_notify_trend_patch_recommended_next_action"] = notify_trend_patch_action.get(
        "recommended_next_action"
    )
    out["gate_notify_trend_patch_recommended_reason"] = notify_trend_patch_action.get("recommended_reason")
    out["gate_notify_trend_patch_recommended_confidence"] = notify_trend_patch_action.get(
        "recommended_confidence"
    )
    out["gate_notify_trend_patch_action_levels_top"] = notify_trend_patch_action.get("levels_top")
    out["gate_notify_trend_patch_action_reasons_top"] = notify_trend_patch_action.get("reasons_top")
    out["gate_notify_trend_patch_next_actions_top"] = notify_trend_patch_action.get("next_actions_top")
    out["gate_notify_trend_patch_action_hints_top"] = notify_trend_patch_action.get("hints_top")
    out["gate_notify_trend_patch_thresholds"] = notify_trend_patch_action.get("thresholds")
    out["gate_notify_trend_patch_apply_status"] = notify_trend_patch_apply.get("status")
    out["gate_notify_trend_patch_apply_status_reason"] = notify_trend_patch_apply.get("status_reason")
    out["gate_notify_trend_patch_apply_events"] = notify_trend_patch_apply.get("events")
    out["gate_notify_trend_patch_apply_problem_events"] = notify_trend_patch_apply.get("problem_events")
    out["gate_notify_trend_patch_apply_problem_rate"] = notify_trend_patch_apply.get("problem_rate")
    out["gate_notify_trend_patch_apply_critical_problem_events"] = notify_trend_patch_apply.get(
        "critical_problem_events"
    )
    out["gate_notify_trend_patch_apply_critical_problem_rate"] = notify_trend_patch_apply.get(
        "critical_problem_rate"
    )
    out["gate_notify_trend_patch_apply_target_batch_id"] = notify_trend_patch_apply.get("target_batch_id")
    out["gate_notify_trend_patch_apply_dominant_reason"] = notify_trend_patch_apply.get("dominant_reason")
    out["gate_notify_trend_patch_apply_dominant_reason_count"] = notify_trend_patch_apply.get(
        "dominant_reason_count"
    )
    out["gate_notify_trend_patch_apply_dominant_reason_share"] = notify_trend_patch_apply.get(
        "dominant_reason_share"
    )
    out["gate_notify_trend_patch_apply_statuses_top"] = notify_trend_patch_apply.get("statuses_top")
    out["gate_notify_trend_patch_apply_reasons_top"] = notify_trend_patch_apply.get("reasons_top")
    out["gate_notify_trend_patch_apply_modes_top"] = notify_trend_patch_apply.get("modes_top")
    out["gate_notify_trend_patch_apply_thresholds"] = notify_trend_patch_apply.get("thresholds")
    suppress_on_critical = env_flag("PI_GATE_ALERT_SUPPRESS_ON_NOTIFY_TREND_CRITICAL", default=True)
    suppress_on_degraded = env_flag("PI_GATE_ALERT_SUPPRESS_ON_NOTIFY_TREND_DEGRADED", default=False)
    suppress_on_mode_drift_critical = env_flag(
        "PI_GATE_ALERT_SUPPRESS_ON_COMPONENT_MODE_DRIFT_CRITICAL",
        default=False,
    )
    out["suppress_on_notify_trend_critical"] = suppress_on_critical
    out["suppress_on_notify_trend_degraded"] = suppress_on_degraded
    out["suppress_on_component_mode_drift_critical"] = suppress_on_mode_drift_critical

    if effective_notify_trend_status == "critical" and suppress_on_critical:
        out["suppressed"] = True
        out["suppression_reason"] = "notify_trend_critical"
        out["reason"] = "suppressed_notify_trend_critical"
        out["alert_status"] = alert_status
        out["notify"]["reason"] = "suppressed"
        return out
    if effective_notify_trend_status == "degraded" and suppress_on_degraded:
        out["suppressed"] = True
        out["suppression_reason"] = "notify_trend_degraded"
        out["reason"] = "suppressed_notify_trend_degraded"
        out["alert_status"] = alert_status
        out["notify"]["reason"] = "suppressed"
        return out
    if component_mode_drift_status == "critical" and suppress_on_mode_drift_critical:
        out["suppressed"] = True
        out["suppression_reason"] = "component_mode_drift_critical"
        out["reason"] = "suppressed_component_mode_drift_critical"
        out["alert_status"] = alert_status
        out["notify"]["reason"] = "suppressed"
        return out

    if dry_run:
        out["reason"] = "dry_run"
        out["alert_status"] = alert_status
        out["notify"]["reason"] = "dry_run"
        return out

    now = dt.datetime.now(dt.timezone.utc)
    min_interval_sec_base = int(float(os.getenv("PI_GATE_ALERT_MIN_INTERVAL_SEC", "1800")))
    min_interval_sec_degraded = int(
        float(os.getenv("PI_GATE_ALERT_MIN_INTERVAL_SEC_ON_NOTIFY_TREND_DEGRADED", str(min_interval_sec_base)))
    )
    min_interval_sec_critical = int(
        float(
            os.getenv(
                "PI_GATE_ALERT_MIN_INTERVAL_SEC_ON_NOTIFY_TREND_CRITICAL",
                str(max(min_interval_sec_base, min_interval_sec_degraded, 3600)),
            )
        )
    )
    min_interval_sec = min_interval_sec_base
    min_interval_source = "base"
    avoid_feedback_escalation = env_flag("PI_GATE_ALERT_AVOID_FEEDBACK_ESCALATION", default=True)
    out["avoid_feedback_escalation"] = avoid_feedback_escalation
    if effective_notify_trend_status == "degraded":
        if avoid_feedback_escalation and trigger_component in {"suppression", "min_interval"}:
            min_interval_sec = min_interval_sec_base
            min_interval_source = "feedback_guard_degraded"
        else:
            min_interval_sec = max(min_interval_sec_base, min_interval_sec_degraded)
            min_interval_source = "notify_trend_degraded"
    elif effective_notify_trend_status == "critical":
        if avoid_feedback_escalation and trigger_component in {"suppression", "min_interval"}:
            min_interval_sec = max(min_interval_sec_base, min_interval_sec_degraded)
            min_interval_source = "feedback_guard_critical"
        else:
            min_interval_sec = max(min_interval_sec_base, min_interval_sec_critical, min_interval_sec_degraded)
            min_interval_source = "notify_trend_critical"
    drift_throttle_enabled = env_flag("PI_GATE_ALERT_ENABLE_COMPONENT_MODE_DRIFT_THROTTLE", default=True)
    out["gate_notify_trend_component_score_mode_drift_throttle_enabled"] = drift_throttle_enabled
    out["gate_notify_trend_component_score_mode_drift_throttle_applied"] = False
    if drift_throttle_enabled and component_mode_drift_status in {"degraded", "critical"}:
        min_interval_sec_drift_degraded = int(
            float(
                os.getenv(
                    "PI_GATE_ALERT_MIN_INTERVAL_SEC_ON_COMPONENT_MODE_DRIFT_DEGRADED",
                    str(max(min_interval_sec_base, min_interval_sec_degraded)),
                )
            )
        )
        min_interval_sec_drift_critical = int(
            float(
                os.getenv(
                    "PI_GATE_ALERT_MIN_INTERVAL_SEC_ON_COMPONENT_MODE_DRIFT_CRITICAL",
                    str(
                        max(
                            min_interval_sec_base,
                            min_interval_sec_degraded,
                            min_interval_sec_critical,
                        )
                    ),
                )
            )
        )
        min_interval_sec_drift_degraded = max(0, min_interval_sec_drift_degraded)
        min_interval_sec_drift_critical = max(
            min_interval_sec_drift_degraded,
            min_interval_sec_drift_critical,
        )
        drift_target = (
            min_interval_sec_drift_critical
            if component_mode_drift_status == "critical"
            else min_interval_sec_drift_degraded
        )
        out["gate_notify_trend_component_score_mode_drift_min_interval_target"] = drift_target
        if drift_target > min_interval_sec:
            min_interval_sec = drift_target
            min_interval_source = f"component_mode_drift_{component_mode_drift_status}"
            out["gate_notify_trend_component_score_mode_drift_throttle_applied"] = True

    state_path = Path(os.getenv("PI_GATE_ALERT_STATE_PATH", str(DEFAULT_GATE_ALERT_STATE)))
    history_path = Path(os.getenv("PI_GATE_ALERT_HISTORY_JSONL", str(DEFAULT_GATE_ALERT_HISTORY)))
    today = now.date().isoformat()
    json_path = Path(
        os.getenv(
            "PI_GATE_ALERT_JSON_PATH",
            str(SYSTEM_ROOT / "output" / "review" / f"{today}_gate_rollout_alert.json"),
        )
    )
    md_path = Path(
        os.getenv(
            "PI_GATE_ALERT_MD_PATH",
            str(SYSTEM_ROOT / "output" / "review" / f"{today}_gate_rollout_alert.md"),
        )
    )

    state = load_json_file(state_path) if state_path.exists() else {}
    last_ts = parse_ts(state.get("last_alert_ts"))
    if last_ts is not None:
        elapsed = (now - last_ts).total_seconds()
        if elapsed < max(0, min_interval_sec):
            out["reason"] = "cooldown"
            out["cooldown_remaining_sec"] = int(max(0, min_interval_sec - elapsed))
            out["state_path"] = str(state_path)
            out["notify"]["reason"] = "emit_cooldown"
            out["min_interval_sec_base"] = min_interval_sec_base
            out["min_interval_sec_effective"] = min_interval_sec
            out["min_interval_source"] = min_interval_source
            return out

    payload = {
        "envelope_version": "1.0",
        "domain": "gate_rollout_alert",
        "ts": now.isoformat(),
        "status": alert_status,
        "reason": gate_rollout_alert.get("reason"),
        "level_required": level_required,
        "pi_cycle_event_ts": pi_cycle_event.get("ts"),
        "gate_rollout": gate_rollout,
        "gate_rollout_alert": gate_rollout_alert,
        "guard_mode_trace": guard_mode_trace,
        "gate_notify_trend_status": effective_notify_trend_status,
        "gate_notify_trend_status_base": notify_trend_status,
        "gate_notify_trend_status_reason": notify_trend_status_reason or None,
        "notify_trend_patch_action": notify_trend_patch_action,
        "notify_trend_patch_apply": notify_trend_patch_apply,
        "patch_draft_action": patch_draft_action,
        "cron_policy_apply_batch": apply_batch_summary,
    }

    ok_json = write_json_atomic(json_path, payload)
    ok_md = write_text_atomic(md_path, build_gate_alert_markdown(payload) + "\n")
    ok_hist = append_jsonl_with_dirs(history_path, payload)
    ok_state = write_json_atomic(
        state_path,
        {
            "last_alert_ts": now.isoformat(),
            "last_alert_status": alert_status,
            "last_alert_reason": gate_rollout_alert.get("reason"),
            "last_json_path": str(json_path),
            "last_md_path": str(md_path),
            "last_history_path": str(history_path),
        },
    )
    all_ok = bool(ok_json and ok_md and ok_hist and ok_state)

    out.update(
        {
            "emitted": all_ok,
            "reason": "emitted" if all_ok else "emit_partial_failure",
            "alert_status": alert_status,
            "state_path": str(state_path),
            "json_path": str(json_path),
            "md_path": str(md_path),
            "history_path": str(history_path),
            "min_interval_sec": min_interval_sec,
            "min_interval_sec_base": min_interval_sec_base,
            "min_interval_sec_effective": min_interval_sec,
            "min_interval_source": min_interval_source,
        }
    )
    if all_ok:
        out["notify"] = maybe_notify_gate_rollout_alert(payload)
    else:
        out["notify"]["reason"] = "emit_partial_failure"
    return out


def maybe_apply_rollout_rollback(
    alert: Dict[str, Any],
    snapshot: Dict[str, Any],
    component_mode_drift: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    enabled_breach = env_flag("PI_GATE_AUTO_ROLLBACK_ON_BREACH", default=False)
    enabled_component_mode_drift = env_flag("PI_GATE_AUTO_ROLLBACK_ON_COMPONENT_MODE_DRIFT", default=False)
    drift_level = str(os.getenv("PI_GATE_AUTO_ROLLBACK_ON_COMPONENT_MODE_DRIFT_LEVEL", "critical")).strip().lower()
    if drift_level not in {"degraded", "critical"}:
        drift_level = "critical"
    drift_min_consecutive = _int_env("PI_GATE_AUTO_ROLLBACK_ON_COMPONENT_MODE_DRIFT_MIN_CONSECUTIVE", 2, minimum=1)
    drift_min_events = _int_env("PI_GATE_AUTO_ROLLBACK_ON_COMPONENT_MODE_DRIFT_MIN_EVENTS", 3, minimum=1)

    out: Dict[str, Any] = {
        "enabled": bool(enabled_breach or enabled_component_mode_drift),
        "enabled_breach": enabled_breach,
        "enabled_component_mode_drift": enabled_component_mode_drift,
        "component_mode_drift_level": drift_level,
        "component_mode_drift_min_consecutive": drift_min_consecutive,
        "component_mode_drift_min_events": drift_min_events,
        "applied": False,
        "reason": "",
        "control_path": str(Path(os.getenv("CORTEX_GATE_ROLLOUT_CONTROL_PATH", str(DEFAULT_ROLLOUT_CONTROL)))),
    }
    if not out["enabled"]:
        out["reason"] = "disabled"
        return out

    breach_triggered = enabled_breach and str(alert.get("status") or "") in {"degraded", "critical"}
    drift_signal = component_mode_drift if isinstance(component_mode_drift, dict) else {}
    drift_status = str(drift_signal.get("status") or "").strip().lower()
    drift_events = _int_value(drift_signal.get("mode_events"), 0)
    drift_consecutive = _int_value(drift_signal.get("latest_consecutive_drift"), 0)
    drift_level_ok = (
        drift_status == "critical" if drift_level == "critical" else drift_status in {"degraded", "critical"}
    )
    drift_triggered = (
        enabled_component_mode_drift
        and drift_level_ok
        and drift_events >= drift_min_events
        and drift_consecutive >= drift_min_consecutive
    )

    trigger = "none"
    if breach_triggered and drift_triggered:
        trigger = "breach_and_component_mode_drift"
    elif breach_triggered:
        trigger = "breach"
    elif drift_triggered:
        trigger = "component_mode_drift"
    out["trigger"] = trigger

    if trigger == "none":
        out["reason"] = "no_breach_or_component_mode_drift"
        return out

    ttl_hours = float(os.getenv("PI_GATE_AUTO_ROLLBACK_TTL_HOURS", "24"))
    now = dt.datetime.now(dt.timezone.utc)
    expires = now + dt.timedelta(hours=max(0.5, ttl_hours))
    control_path = Path(out["control_path"])
    reason = "auto_rollback_gate_rollout_breach"
    if trigger == "component_mode_drift":
        reason = "auto_rollback_component_mode_drift"
    elif trigger == "breach_and_component_mode_drift":
        reason = "auto_rollback_gate_rollout_breach_and_component_mode_drift"
    control_payload = {
        "override_mode": "shadow",
        "source": "pi_cycle_orchestrator",
        "reason": reason,
        "trigger": trigger,
        "created_ts": now.isoformat(),
        "expires_ts": expires.isoformat(),
        "alert": alert,
        "snapshot": {
            "rollout_mode_effective": snapshot.get("rollout_mode_effective"),
            "total_events": snapshot.get("total_events"),
            "act_events": snapshot.get("act_events"),
            "would_block_rate": snapshot.get("would_block_rate"),
            "cooldown_hit_rate": snapshot.get("cooldown_hit_rate"),
            "recover_confirm_fail_rate": snapshot.get("recover_confirm_fail_rate"),
            "last_event_ts": snapshot.get("last_event_ts"),
        },
        "component_mode_drift": {
            "status": drift_status or None,
            "reason": drift_signal.get("reason"),
            "mode_events": drift_events,
            "drift_events": _int_value(drift_signal.get("drift_events"), 0),
            "drift_rate": drift_signal.get("drift_rate"),
            "latest_consecutive_drift": drift_consecutive,
            "latest_drift": drift_signal.get("latest_drift"),
            "top_drift_pairs": drift_signal.get("top_drift_pairs") if drift_signal else [],
            "window_hours": drift_signal.get("window_hours"),
        },
    }
    ok = write_json_atomic(control_path, control_payload)
    out["applied"] = bool(ok)
    out["reason"] = "applied" if ok else "write_failed"
    out["override_mode"] = "shadow"
    out["expires_ts"] = expires.isoformat()
    return out


def maybe_apply_proxy_rollout_control(
    *,
    guard_mode: Dict[str, Any],
    rollback_applied: bool,
) -> Dict[str, Any]:
    enabled = env_flag("PI_PROXY_ROLLOUT_CONTROL_ENABLED", default=False)
    min_actionable = _int_env("PI_PROXY_ROLLOUT_CONTROL_MIN_ACTIONABLE", 1, minimum=1)
    ttl_hours = max(0.5, _as_float(os.getenv("PI_PROXY_ROLLOUT_CONTROL_TTL_HOURS")) or 6.0)
    out: Dict[str, Any] = {
        "enabled": enabled,
        "applied": False,
        "reason": "",
        "trigger": "none",
        "override_mode": "shadow",
        "control_path": str(Path(os.getenv("CORTEX_GATE_ROLLOUT_CONTROL_PATH", str(DEFAULT_ROLLOUT_CONTROL)))),
        "min_actionable": min_actionable,
        "ttl_hours": ttl_hours,
    }
    if not enabled:
        out["reason"] = "disabled"
        return out
    if rollback_applied:
        out["reason"] = "rollback_already_applied"
        return out

    patch_status = str(guard_mode.get("patch_draft_status") or "unknown").strip().lower()
    patch_pending = bool(guard_mode.get("patch_draft_pending"))
    iso_pending = bool(guard_mode.get("patch_draft_isolation_rollout_pending"))
    iso_actionable = _int_value(guard_mode.get("patch_draft_isolation_actionable_candidates"), 0)
    iso_critical = _int_value(guard_mode.get("patch_draft_isolation_critical_candidates"), 0)
    iso_next_batch = (
        str(guard_mode.get("patch_draft_isolation_rollout_next_batch") or "").strip() or None
    )

    trigger = "none"
    reason = "no_trigger"
    if iso_critical >= min_actionable:
        trigger = "proxy_isolation_critical_candidates"
        reason = "auto_proxy_isolation_critical_candidates"
    elif iso_actionable >= min_actionable and patch_status in {"degraded", "critical"}:
        trigger = "proxy_isolation_actionable_degraded"
        reason = "auto_proxy_isolation_actionable_degraded"
    elif iso_actionable >= min_actionable and (iso_pending or patch_pending):
        trigger = "proxy_isolation_actionable_pending"
        reason = "auto_proxy_isolation_actionable_pending"

    out["trigger"] = trigger
    if trigger == "none":
        out["reason"] = reason
        out["patch_status"] = patch_status
        out["patch_pending"] = patch_pending
        out["isolation_pending"] = iso_pending
        out["isolation_actionable_candidates"] = iso_actionable
        out["isolation_critical_candidates"] = iso_critical
        out["isolation_next_batch"] = iso_next_batch
        return out

    now = dt.datetime.now(dt.timezone.utc)
    expires = now + dt.timedelta(hours=ttl_hours)
    control_path = Path(out["control_path"])
    control_payload = {
        "override_mode": "shadow",
        "source": "pi_cycle_orchestrator",
        "reason": reason,
        "trigger": trigger,
        "created_ts": now.isoformat(),
        "expires_ts": expires.isoformat(),
        "proxy_isolation": {
            "patch_status": patch_status,
            "patch_pending": patch_pending,
            "isolation_pending": iso_pending,
            "isolation_actionable_candidates": iso_actionable,
            "isolation_critical_candidates": iso_critical,
            "isolation_next_batch": iso_next_batch,
            "patch_next_action": guard_mode.get("patch_draft_next_action"),
            "patch_next_action_secondary": guard_mode.get("patch_draft_next_action_secondary"),
            "patch_action_reason": guard_mode.get("patch_draft_action_reason"),
            "patch_action_level": guard_mode.get("patch_draft_action_level"),
            "patch_source_path": guard_mode.get("rollback_source_path"),
            "patch_report_ts": guard_mode.get("rollback_report_ts"),
        },
    }
    ok = write_json_atomic(control_path, control_payload)
    out["applied"] = bool(ok)
    out["reason"] = "applied" if ok else "write_failed"
    out["expires_ts"] = expires.isoformat()
    out["patch_status"] = patch_status
    out["patch_pending"] = patch_pending
    out["isolation_pending"] = iso_pending
    out["isolation_actionable_candidates"] = iso_actionable
    out["isolation_critical_candidates"] = iso_critical
    out["isolation_next_batch"] = iso_next_batch
    return out


def maybe_apply_digital_life_rollout_control(
    *,
    guard_mode: Dict[str, Any],
    digital_life_summary: Dict[str, Any],
    rollback_applied: bool,
    proxy_applied: bool = False,
) -> Dict[str, Any]:
    enabled = env_flag("PI_DIGITAL_LIFE_ROLLOUT_CONTROL_ENABLED", default=False)
    out: Dict[str, Any] = {
        "enabled": enabled,
        "applied": False,
        "reason": "",
        "trigger": "none",
        "override_mode": "shadow",
        "control_path": str(Path(os.getenv("CORTEX_GATE_ROLLOUT_CONTROL_PATH", str(DEFAULT_ROLLOUT_CONTROL)))),
        "ttl_hours": float(_as_float(os.getenv("PI_DIGITAL_LIFE_ROLLOUT_CONTROL_TTL_HOURS")) or 6.0),
    }
    if not enabled:
        out["reason"] = "disabled"
        return out
    if rollback_applied:
        out["reason"] = "rollback_already_applied"
        return out
    if proxy_applied:
        out["reason"] = "proxy_control_already_applied"
        return out

    lifecycle = str(
        digital_life_summary.get("lifecycle_mode")
        or guard_mode.get("digital_life_lifecycle_mode")
        or ""
    ).strip().upper()
    status = str(
        digital_life_summary.get("status")
        or guard_mode.get("digital_life_status")
        or "unknown"
    ).strip().lower()
    viability = _as_float(
        digital_life_summary.get("viability_score")
        if digital_life_summary.get("viability_score") is not None
        else guard_mode.get("digital_life_viability_score")
    )
    stress = _as_float(
        digital_life_summary.get("stress_score")
        if digital_life_summary.get("stress_score") is not None
        else guard_mode.get("digital_life_stress_score")
    )
    next_action = (
        str(digital_life_summary.get("next_action") or guard_mode.get("digital_life_next_action") or "").strip()
        or None
    )

    trigger = "none"
    reason = "no_trigger"
    if status == "critical":
        trigger = "digital_life_status_critical"
        reason = "auto_digital_life_status_critical"
    elif lifecycle == "SURVIVE":
        trigger = "digital_life_survive"
        reason = "auto_digital_life_survive"
    elif lifecycle == "STABILIZE":
        trigger = "digital_life_stabilize"
        reason = "auto_digital_life_stabilize"
    out["trigger"] = trigger
    if trigger == "none":
        out["reason"] = reason
        out["lifecycle_mode"] = lifecycle or None
        out["digital_life_status"] = status
        out["next_action"] = next_action
        out["viability_score"] = viability
        out["stress_score"] = stress
        return out

    ttl_hours = max(0.5, float(out["ttl_hours"]))
    now = dt.datetime.now(dt.timezone.utc)
    expires = now + dt.timedelta(hours=ttl_hours)
    control_path = Path(out["control_path"])
    control_payload = {
        "override_mode": "shadow",
        "source": "pi_cycle_orchestrator",
        "reason": reason,
        "trigger": trigger,
        "created_ts": now.isoformat(),
        "expires_ts": expires.isoformat(),
        "digital_life": {
            "status": status,
            "lifecycle_mode": lifecycle or None,
            "next_action": next_action,
            "viability_score": viability,
            "stress_score": stress,
            "summary_status": digital_life_summary.get("status"),
            "summary_reason": digital_life_summary.get("reason"),
        },
    }
    ok = write_json_atomic(control_path, control_payload)
    out["applied"] = bool(ok)
    out["reason"] = "applied" if ok else "write_failed"
    out["expires_ts"] = expires.isoformat()
    out["lifecycle_mode"] = lifecycle or None
    out["digital_life_status"] = status
    out["next_action"] = next_action
    out["viability_score"] = viability
    out["stress_score"] = stress
    return out


def run_step(name: str, cmd: List[str], timeout: int) -> Dict[str, Any]:
    started = time.time()
    p = subprocess.run(
        cmd,
        cwd=str(WORKSPACE),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = p.stdout or ""
    err = p.stderr or ""
    return {
        "name": name,
        "cmd": cmd,
        "returncode": int(p.returncode),
        "duration_sec": round(time.time() - started, 3),
        "json_tail": parse_json_tail(out),
        "stdout_tail": out[-2000:],
        "stderr_tail": err[-1500:],
    }


def _extract_batch_id_from_cmd(cmd: Any) -> Optional[str]:
    if not isinstance(cmd, list):
        return None
    try:
        idx = cmd.index("--batch-id")
    except ValueError:
        return None
    if idx + 1 >= len(cmd):
        return None
    batch = str(cmd[idx + 1] or "").strip()
    return batch or None


def _extract_cron_policy_apply_batch_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "status": "unknown",
        "reason": "step_missing",
        "batch_id": None,
        "mode": None,
        "applied_to_source": False,
        "apply_skipped_reason": None,
        "selected_actions": 0,
        "changed_jobs": 0,
        "failed_jobs": 0,
        "blocked_jobs": 0,
        "operations_total": 0,
        "operations_applied": 0,
        "operations_changed": 0,
        "output_path": None,
        "candidate_jobs_output": None,
        "returncode": None,
        "duration_sec": None,
    }
    if not isinstance(steps, list):
        return out
    step = next(
        (x for x in steps if isinstance(x, dict) and str(x.get("name") or "") == "cron_policy_apply_batch"),
        None,
    )
    if not isinstance(step, dict):
        return out

    out["present"] = True
    out["returncode"] = step.get("returncode")
    out["duration_sec"] = step.get("duration_sec")
    out["batch_id"] = _extract_batch_id_from_cmd(step.get("cmd"))
    rc = _int_value(step.get("returncode"), 1)
    tail = step.get("json_tail") if isinstance(step.get("json_tail"), dict) else {}
    if tail and str(tail.get("domain") or "") == "cron_policy_batch_apply":
        summary = tail.get("summary") if isinstance(tail.get("summary"), dict) else {}
        out.update(
            {
                "status": str(tail.get("status") or "unknown"),
                "reason": "ok",
                "batch_id": str(tail.get("batch_id") or out.get("batch_id") or "").strip() or out.get("batch_id"),
                "mode": tail.get("mode"),
                "applied_to_source": bool(tail.get("applied_to_source", False)),
                "apply_skipped_reason": tail.get("apply_skipped_reason"),
                "selected_actions": max(0, int(summary.get("selected_actions") or 0)),
                "changed_jobs": max(0, int(summary.get("changed_jobs") or 0)),
                "failed_jobs": max(0, int(summary.get("failed_jobs") or 0)),
                "blocked_jobs": max(0, int(summary.get("blocked_jobs") or 0)),
                "operations_total": max(0, int(summary.get("operations_total") or 0)),
                "operations_applied": max(0, int(summary.get("operations_applied") or 0)),
                "operations_changed": max(0, int(summary.get("operations_changed") or 0)),
                "output_path": tail.get("output_path"),
                "candidate_jobs_output": tail.get("candidate_jobs_output"),
            }
        )
        return out

    if bool(tail.get("dry_run")):
        out["status"] = "dry_run"
        out["reason"] = "orchestrator_dry_run"
        out["mode"] = "dry_run"
        return out

    if rc != 0:
        out["status"] = "degraded"
        out["reason"] = "step_failed"
    else:
        out["status"] = "unknown"
        out["reason"] = "missing_batch_payload"
    return out


def _extract_digital_life_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "status": "unknown",
        "reason": "step_missing",
        "lifecycle_mode": None,
        "next_action": None,
        "viability_score": None,
        "stress_score": None,
        "state_path": None,
        "events_path": None,
        "returncode": None,
        "duration_sec": None,
    }
    if not isinstance(steps, list):
        return out
    step = next(
        (x for x in steps if isinstance(x, dict) and str(x.get("name") or "") == "digital_life_core"),
        None,
    )
    if not isinstance(step, dict):
        return out

    out["present"] = True
    out["returncode"] = step.get("returncode")
    out["duration_sec"] = step.get("duration_sec")
    rc = _int_value(step.get("returncode"), 1)
    tail = step.get("json_tail") if isinstance(step.get("json_tail"), dict) else {}

    if tail and str(tail.get("domain") or "") == "digital_life_core":
        persisted = tail.get("persisted") if isinstance(tail.get("persisted"), dict) else {}
        out.update(
            {
                "status": str(tail.get("status") or "unknown"),
                "reason": "ok",
                "lifecycle_mode": tail.get("lifecycle_mode"),
                "next_action": tail.get("next_action"),
                "viability_score": tail.get("viability_score"),
                "stress_score": tail.get("stress_score"),
                "state_path": persisted.get("state_path"),
                "events_path": persisted.get("events_path"),
            }
        )
        return out

    if bool(tail.get("dry_run")):
        out["status"] = "dry_run"
        out["reason"] = "orchestrator_dry_run"
        return out

    if rc != 0:
        out["status"] = "degraded"
        out["reason"] = "step_failed"
    else:
        out["status"] = "unknown"
        out["reason"] = "missing_payload"
    return out


def _extract_core_proxy_bypass_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "status": "unknown",
        "reason": "step_missing",
        "source": None,
        "returncode": None,
        "duration_sec": None,
        "requests_total": 0,
        "bypass_attempted": 0,
        "bypass_success": 0,
        "bypass_failed": 0,
        "no_proxy_retries": 0,
        "no_proxy_retry_exhausted": 0,
        "hint_response": 0,
        "hint_exception": 0,
        "reason_response_hint": 0,
        "reason_exception_hint": 0,
        "last_reason": None,
        "local_requests_total": 0,
        "local_bypass_attempted": 0,
        "local_bypass_success": 0,
        "local_bypass_failed": 0,
        "local_no_proxy_retries": 0,
        "local_no_proxy_retry_exhausted": 0,
        "exec_requests_total": 0,
        "exec_bypass_attempted": 0,
        "exec_bypass_success": 0,
        "exec_bypass_failed": 0,
        "exec_no_proxy_retries": 0,
        "exec_no_proxy_retry_exhausted": 0,
    }
    if not isinstance(steps, list):
        return out
    step = next(
        (x for x in steps if isinstance(x, dict) and str(x.get("name") or "") == "core"),
        None,
    )
    if not isinstance(step, dict):
        return out

    out["present"] = True
    out["returncode"] = step.get("returncode")
    out["duration_sec"] = step.get("duration_sec")
    rc = _int_value(step.get("returncode"), 1)
    tail = step.get("json_tail") if isinstance(step.get("json_tail"), dict) else {}

    if tail and str(tail.get("domain") or "") == "lie_spot_core":
        local = tail.get("net_proxy_bypass") if isinstance(tail.get("net_proxy_bypass"), dict) else {}
        exec_stats = (
            tail.get("exec_net_proxy_bypass")
            if isinstance(tail.get("exec_net_proxy_bypass"), dict)
            else {}
        )
        local_reason_counts = (
            local.get("reason_counts") if isinstance(local.get("reason_counts"), dict) else {}
        )
        exec_reason_counts = (
            exec_stats.get("reason_counts") if isinstance(exec_stats.get("reason_counts"), dict) else {}
        )
        local_requests_total = max(0, _int_value(local.get("requests_total"), 0))
        local_bypass_attempted = max(0, _int_value(local.get("bypass_attempted"), 0))
        local_bypass_success = max(0, _int_value(local.get("bypass_success"), 0))
        local_bypass_failed = max(0, _int_value(local.get("bypass_failed"), 0))
        local_no_proxy_retries = max(0, _int_value(local.get("no_proxy_retries"), 0))
        local_no_proxy_retry_exhausted = max(0, _int_value(local.get("no_proxy_retry_exhausted"), 0))
        local_hint_response = max(0, _int_value(local.get("hint_response"), 0))
        local_hint_exception = max(0, _int_value(local.get("hint_exception"), 0))
        local_reason_response_hint = max(
            0, _int_value(local_reason_counts.get("response_hint"), 0)
        )
        local_reason_exception_hint = max(
            0, _int_value(local_reason_counts.get("exception_hint"), 0)
        )

        exec_requests_total = max(0, _int_value(exec_stats.get("requests_total"), 0))
        exec_bypass_attempted = max(0, _int_value(exec_stats.get("bypass_attempted"), 0))
        exec_bypass_success = max(0, _int_value(exec_stats.get("bypass_success"), 0))
        exec_bypass_failed = max(0, _int_value(exec_stats.get("bypass_failed"), 0))
        exec_no_proxy_retries = max(0, _int_value(exec_stats.get("no_proxy_retries"), 0))
        exec_no_proxy_retry_exhausted = max(
            0, _int_value(exec_stats.get("no_proxy_retry_exhausted"), 0)
        )
        exec_hint_response = max(0, _int_value(exec_stats.get("hint_response"), 0))
        exec_hint_exception = max(0, _int_value(exec_stats.get("hint_exception"), 0))
        exec_reason_response_hint = max(
            0, _int_value(exec_reason_counts.get("response_hint"), 0)
        )
        exec_reason_exception_hint = max(
            0, _int_value(exec_reason_counts.get("exception_hint"), 0)
        )

        requests_total = local_requests_total + exec_requests_total
        bypass_attempted = local_bypass_attempted + exec_bypass_attempted
        bypass_success = local_bypass_success + exec_bypass_success
        bypass_failed = local_bypass_failed + exec_bypass_failed
        no_proxy_retries = local_no_proxy_retries + exec_no_proxy_retries
        no_proxy_retry_exhausted = (
            local_no_proxy_retry_exhausted + exec_no_proxy_retry_exhausted
        )
        hint_response = local_hint_response + exec_hint_response
        hint_exception = local_hint_exception + exec_hint_exception
        reason_response_hint = local_reason_response_hint + exec_reason_response_hint
        reason_exception_hint = local_reason_exception_hint + exec_reason_exception_hint

        last_reason = str(local.get("last_reason") or "").strip() or str(
            exec_stats.get("last_reason") or ""
        ).strip() or None
        out.update(
            {
                "source": "core_json_tail",
                "requests_total": requests_total,
                "bypass_attempted": bypass_attempted,
                "bypass_success": bypass_success,
                "bypass_failed": bypass_failed,
                "no_proxy_retries": no_proxy_retries,
                "no_proxy_retry_exhausted": no_proxy_retry_exhausted,
                "hint_response": hint_response,
                "hint_exception": hint_exception,
                "reason_response_hint": reason_response_hint,
                "reason_exception_hint": reason_exception_hint,
                "last_reason": last_reason,
                "local_requests_total": local_requests_total,
                "local_bypass_attempted": local_bypass_attempted,
                "local_bypass_success": local_bypass_success,
                "local_bypass_failed": local_bypass_failed,
                "local_no_proxy_retries": local_no_proxy_retries,
                "local_no_proxy_retry_exhausted": local_no_proxy_retry_exhausted,
                "exec_requests_total": exec_requests_total,
                "exec_bypass_attempted": exec_bypass_attempted,
                "exec_bypass_success": exec_bypass_success,
                "exec_bypass_failed": exec_bypass_failed,
                "exec_no_proxy_retries": exec_no_proxy_retries,
                "exec_no_proxy_retry_exhausted": exec_no_proxy_retry_exhausted,
            }
        )
        if bypass_attempted <= 0:
            out["status"] = "ok"
            out["reason"] = "no_bypass_needed"
        elif no_proxy_retry_exhausted > 0:
            out["status"] = "degraded"
            out["reason"] = "no_proxy_retry_exhausted"
        elif bypass_failed > 0:
            out["status"] = "degraded"
            out["reason"] = "bypass_failed"
        elif bypass_success <= 0:
            out["status"] = "degraded"
            out["reason"] = "bypass_attempt_without_success"
        else:
            out["status"] = "ok"
            out["reason"] = "bypass_success"
        return out

    if bool(tail.get("dry_run")):
        out["status"] = "dry_run"
        out["reason"] = "orchestrator_dry_run"
        return out

    if rc != 0:
        out["status"] = "degraded"
        out["reason"] = "step_failed"
    else:
        out["status"] = "unknown"
        out["reason"] = "missing_payload"
    return out


def _extract_core_execution_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "status": "unknown",
        "reason": "step_missing",
        "source": None,
        "returncode": None,
        "duration_sec": None,
        "action": None,
        "decision": None,
        "executor_attempted": None,
        "executor_probe_requested": None,
        "executor_probe_effective": None,
        "executor_force_probe_on_guardrail": None,
        "executor_live_requested": None,
        "executor_live_effective": None,
        "executor_cmd": None,
        "order_http": None,
        "order_endpoint": None,
        "order_decision": None,
        "order_mode": None,
        "order_reason": None,
        "order_error": None,
        "order_cap_violation": None,
        "order_simulated": None,
        "guardrail_hit": None,
        "guardrail_reasons": [],
        "paper_fill_gate_mode": None,
        "paper_fill_gate_policy": None,
        "paper_fill_gate_reason": None,
        "paper_fill_gate_cap_violation": None,
        "paper_execution_attempted": None,
        "paper_execution_applied": None,
        "paper_execution_route": None,
        "paper_execution_fill_px": None,
        "paper_execution_signed_slippage_bps": None,
        "paper_execution_fee_rate": None,
        "paper_execution_fee_usdt": None,
        "paper_execution_ledger_written": None,
    }
    if not isinstance(steps, list):
        return out
    step = next(
        (x for x in steps if isinstance(x, dict) and str(x.get("name") or "") == "core"),
        None,
    )
    if not isinstance(step, dict):
        return out

    out["present"] = True
    out["returncode"] = step.get("returncode")
    out["duration_sec"] = step.get("duration_sec")
    rc = _int_value(step.get("returncode"), 1)
    tail = step.get("json_tail") if isinstance(step.get("json_tail"), dict) else {}

    if tail and str(tail.get("domain") or "") == "lie_spot_core":
        out["source"] = "core_json_tail"
        action = str(tail.get("action") or "").strip()
        decision = str(tail.get("decision") or "").strip().lower()
        decision = decision or None
        out["action"] = action or None
        out["decision"] = decision

        executor = tail.get("executor") if isinstance(tail.get("executor"), dict) else {}
        order_result = tail.get("order_result") if isinstance(tail.get("order_result"), dict) else {}
        guardrails = tail.get("guardrails") if isinstance(tail.get("guardrails"), dict) else {}
        paper_fill_gate = (
            tail.get("paper_fill_gate")
            if isinstance(tail.get("paper_fill_gate"), dict)
            else {}
        )
        paper_execution = (
            tail.get("paper_execution")
            if isinstance(tail.get("paper_execution"), dict)
            else {}
        )

        if executor:
            out["executor_attempted"] = (
                bool(executor.get("attempted"))
                if executor.get("attempted") is not None
                else None
            )
            out["executor_probe_requested"] = (
                bool(executor.get("probe_requested"))
                if executor.get("probe_requested") is not None
                else None
            )
            out["executor_probe_effective"] = (
                bool(executor.get("probe_effective"))
                if executor.get("probe_effective") is not None
                else None
            )
            out["executor_force_probe_on_guardrail"] = (
                bool(executor.get("force_probe_on_guardrail"))
                if executor.get("force_probe_on_guardrail") is not None
                else None
            )
            out["executor_live_requested"] = (
                bool(executor.get("live_requested"))
                if executor.get("live_requested") is not None
                else None
            )
            out["executor_live_effective"] = (
                bool(executor.get("live_effective"))
                if executor.get("live_effective") is not None
                else None
            )
            cmd = executor.get("cmd")
            out["executor_cmd"] = cmd if isinstance(cmd, list) else None

        if order_result:
            order_http_raw = order_result.get("http")
            if order_http_raw is not None:
                out["order_http"] = _int_value(order_http_raw, 0)
            out["order_endpoint"] = str(order_result.get("endpoint") or "").strip() or None
            order_decision = str(order_result.get("decision") or "").strip().lower()
            out["order_decision"] = order_decision or None
            out["order_mode"] = str(order_result.get("mode") or "").strip() or None
            out["order_reason"] = str(order_result.get("reason") or "").strip() or None
            out["order_error"] = str(order_result.get("error") or "").strip() or None
            out["order_cap_violation"] = order_result.get("cap_violation")
            endpoint = str(out.get("order_endpoint") or "").lower()
            out["order_simulated"] = bool(
                order_decision == "simulate"
                or str(out.get("order_mode") or "").lower() == "paper_probe_fallback"
                or endpoint.startswith("simulated/")
            )
        else:
            out["order_simulated"] = False

        if guardrails:
            out["guardrail_hit"] = (
                bool(guardrails.get("hit"))
                if guardrails.get("hit") is not None
                else None
            )
            reasons_raw = guardrails.get("reasons")
            if isinstance(reasons_raw, list):
                out["guardrail_reasons"] = [
                    str(x or "").strip() for x in reasons_raw[:20] if str(x or "").strip()
                ]

        if paper_fill_gate:
            out["paper_fill_gate_mode"] = str(paper_fill_gate.get("mode") or "").strip() or None
            out["paper_fill_gate_policy"] = str(paper_fill_gate.get("policy") or "").strip() or None
            out["paper_fill_gate_reason"] = str(paper_fill_gate.get("reason") or "").strip() or None
            out["paper_fill_gate_cap_violation"] = paper_fill_gate.get("cap_violation")

        if paper_execution:
            out["paper_execution_attempted"] = (
                bool(paper_execution.get("attempted"))
                if paper_execution.get("attempted") is not None
                else None
            )
            out["paper_execution_applied"] = (
                bool(paper_execution.get("applied"))
                if paper_execution.get("applied") is not None
                else None
            )
            out["paper_execution_route"] = (
                str(paper_execution.get("route") or "").strip() or None
            )
            out["paper_execution_fill_px"] = paper_execution.get("fill_px")
            out["paper_execution_signed_slippage_bps"] = paper_execution.get(
                "signed_slippage_bps"
            )
            out["paper_execution_fee_rate"] = paper_execution.get("fee_rate")
            out["paper_execution_fee_usdt"] = paper_execution.get("fee_usdt")
            out["paper_execution_ledger_written"] = (
                bool(paper_execution.get("ledger_written"))
                if paper_execution.get("ledger_written") is not None
                else None
            )

        attempted = bool(out.get("executor_attempted"))
        order_http = out.get("order_http")
        order_decision = str(out.get("order_decision") or "").strip().lower() or None
        order_mode = str(out.get("order_mode") or "").strip().lower()
        order_error = str(out.get("order_error") or "").strip()
        order_simulated = bool(out.get("order_simulated"))

        if decision == "error" or order_decision == "error":
            out["status"] = "degraded"
            out["reason"] = "exec_error"
        elif order_error and not (decision == "simulate" or order_simulated):
            out["status"] = "degraded"
            out["reason"] = "order_result_error"
        elif attempted and decision in {"order", "simulate"} and not order_result:
            out["status"] = "degraded"
            out["reason"] = "missing_order_result"
        elif attempted and decision in {"order", "simulate"} and order_http is None:
            out["status"] = "degraded"
            out["reason"] = "missing_order_http"
        elif attempted and decision in {"order", "simulate"} and not (200 <= int(order_http) < 300):
            out["status"] = "degraded"
            out["reason"] = "order_http_non_2xx"
        elif decision == "simulate" or order_simulated:
            out["status"] = "ok"
            out["reason"] = (
                "simulate_local_paper_router"
                if order_mode == "paper_local_router"
                else "simulate_probe_fallback"
            )
        elif decision == "order":
            out["status"] = "ok"
            out["reason"] = "order_executed"
        elif decision == "no-trade":
            out["status"] = "ok"
            if attempted:
                out["reason"] = "no_trade_after_execution_check"
            elif out.get("guardrail_hit") is True:
                out["reason"] = "no_trade_guardrail"
            else:
                out["reason"] = "no_trade"
        elif decision:
            out["status"] = "ok"
            out["reason"] = "core_decision_ok"
        else:
            out["status"] = "unknown"
            out["reason"] = "missing_decision"
        return out

    if bool(tail.get("dry_run")):
        out["status"] = "dry_run"
        out["reason"] = "orchestrator_dry_run"
        return out

    if rc != 0:
        out["status"] = "degraded"
        out["reason"] = "step_failed"
    else:
        out["status"] = "unknown"
        out["reason"] = "missing_payload"
    return out


def _extract_core_paper_mode_readiness_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "status": "unknown",
        "reason": "step_missing",
        "source": None,
        "returncode": None,
        "duration_sec": None,
        "enforce": None,
        "fail_closed": None,
        "gate_blocked": False,
        "gate_reason": None,
        "readiness_status": None,
        "ready_for_paper_mode": None,
        "report_path": None,
        "report_ts": None,
        "report_age_hours": None,
        "report_stale": None,
        "coverage": None,
        "missing_buckets": None,
        "largest_missing_block_hours": None,
        "fail_reasons": [],
        "refresh_attempted": None,
        "refresh_ok": None,
        "refresh_reason": None,
        "refresh_trigger": None,
        "refresh_returncode": None,
        "refresh_duration_sec": None,
        "refresh_throttled": None,
        "refresh_throttle_remaining_sec": None,
        "refresh_lock_path": None,
        "refresh_lock_acquired": None,
        "refresh_state_path": None,
        "refresh_last_attempt_ts": None,
        "refresh_last_success_ts": None,
        "refresh_lock_contention_attempted": None,
        "refresh_lock_contention_ok": None,
        "refresh_lock_contention_reason": None,
        "refresh_lock_contention_returncode": None,
        "refresh_lock_contention_duration_sec": None,
        "refresh_lock_contention_json_path": None,
        "refresh_lock_contention_md_path": None,
        "refresh_lock_contention_csv_path": None,
        "refresh_lock_contention_source_coverage_csv": None,
        "refresh_lock_contention_gate_level": None,
        "refresh_lock_contention_gate_fail_triggered": None,
        "refresh_lock_contention_gate_source_bucket_view": None,
        "refresh_lock_contention_gate_source_bucket_requested": None,
        "refresh_lock_contention_gate_source_bucket_effective": None,
        "refresh_lock_contention_gate_sample_guard_level": None,
        "refresh_lock_contention_gate_sample_guard_fail_triggered": None,
        "refresh_lock_contention_gate_sample_guard_fail_on_level": None,
        "lock_contention_enforce": None,
        "lock_contention_fail_closed": None,
        "lock_contention_report_path": None,
        "lock_contention_report_present": None,
        "lock_contention_overall_level": None,
        "lock_contention_source_bucket_view": None,
        "lock_contention_source_bucket_requested": None,
        "lock_contention_source_bucket_effective": None,
        "lock_contention_fail_triggered": None,
        "lock_contention_generated_at": None,
        "lock_contention_report_age_hours": None,
        "lock_contention_report_stale": None,
        "lock_contention_max_age_hours": None,
        "lock_contention_max_allowed_level": None,
        "lock_contention_required_source_bucket": None,
        "lock_contention_require_bucket_match": None,
        "lock_contention_require_sample_guard": None,
        "lock_contention_sample_guard_level": None,
        "lock_contention_sample_guard_fail_triggered": None,
        "lock_contention_sample_guard_fail_on_level": None,
        "lock_contention_blocked": None,
        "lock_contention_reason": None,
        "core_reason": None,
        "pulse_lock_path": None,
        "pulse_lock_acquired": None,
        "pulse_lock_reason": None,
    }
    if not isinstance(steps, list):
        return out
    step = next(
        (x for x in steps if isinstance(x, dict) and str(x.get("name") or "") == "core"),
        None,
    )
    if not isinstance(step, dict):
        return out

    out["present"] = True
    out["returncode"] = step.get("returncode")
    out["duration_sec"] = step.get("duration_sec")
    rc = _int_value(step.get("returncode"), 1)
    tail = step.get("json_tail") if isinstance(step.get("json_tail"), dict) else {}

    if tail and str(tail.get("domain") or "") == "lie_spot_core":
        out["source"] = "core_json_tail"
        core_reason = str(tail.get("reason") or "").strip().lower() or None
        pulse_lock = tail.get("pulse_lock") if isinstance(tail.get("pulse_lock"), dict) else {}
        readiness = (
            tail.get("paper_mode_readiness")
            if isinstance(tail.get("paper_mode_readiness"), dict)
            else {}
        )
        if not readiness:
            out["status"] = "unknown"
            out["reason"] = "missing_readiness_payload"
            return out

        readiness_status = str(readiness.get("status") or "unknown").strip().lower()
        if readiness_status not in {"ready", "blocked", "unknown"}:
            readiness_status = "unknown"
        gate_blocked = bool(readiness.get("gate_blocked"))
        enforce = readiness.get("enforce")
        fail_closed = readiness.get("fail_closed")
        fail_reasons_raw = readiness.get("fail_reasons")
        fail_reasons: List[str] = []
        if isinstance(fail_reasons_raw, list):
            for item in fail_reasons_raw[:20]:
                text = str(item or "").strip()
                if text:
                    fail_reasons.append(text)
        refresh_lock_contention = (
            readiness.get("refresh_lock_contention")
            if isinstance(readiness.get("refresh_lock_contention"), dict)
            else {}
        )
        lock_contention_gate = (
            readiness.get("lock_contention_gate")
            if isinstance(readiness.get("lock_contention_gate"), dict)
            else {}
        )

        out.update(
            {
                "enforce": bool(enforce) if enforce is not None else None,
                "fail_closed": bool(fail_closed) if fail_closed is not None else None,
                "gate_blocked": gate_blocked,
                "gate_reason": str(readiness.get("gate_reason") or "").strip() or None,
                "readiness_status": readiness_status,
                "ready_for_paper_mode": readiness.get("ready_for_paper_mode"),
                "report_path": str(readiness.get("report_path") or "").strip() or None,
                "report_ts": str(readiness.get("report_ts") or "").strip() or None,
                "report_age_hours": _as_float(readiness.get("report_age_hours")),
                "report_stale": (
                    bool(readiness.get("report_stale"))
                    if readiness.get("report_stale") is not None
                    else None
                ),
                "coverage": _as_float(readiness.get("coverage")),
                "missing_buckets": _int_value(readiness.get("missing_buckets"), 0)
                if readiness.get("missing_buckets") is not None
                else None,
                "largest_missing_block_hours": _as_float(readiness.get("largest_missing_block_hours")),
                "fail_reasons": fail_reasons,
                "refresh_attempted": (
                    bool(readiness.get("refresh_attempted"))
                    if readiness.get("refresh_attempted") is not None
                    else None
                ),
                "refresh_ok": (
                    bool(readiness.get("refresh_ok"))
                    if readiness.get("refresh_ok") is not None
                    else None
                ),
                "refresh_reason": str(readiness.get("refresh_reason") or "").strip() or None,
                "refresh_trigger": str(readiness.get("refresh_trigger") or "").strip() or None,
                "refresh_returncode": (
                    _int_value(readiness.get("refresh_returncode"), 0)
                    if readiness.get("refresh_returncode") is not None
                    else None
                ),
                "refresh_duration_sec": _as_float(readiness.get("refresh_duration_sec")),
                "refresh_throttled": (
                    bool(readiness.get("refresh_throttled"))
                    if readiness.get("refresh_throttled") is not None
                    else None
                ),
                "refresh_throttle_remaining_sec": (
                    _int_value(readiness.get("refresh_throttle_remaining_sec"), 0)
                    if readiness.get("refresh_throttle_remaining_sec") is not None
                    else None
                ),
                "refresh_lock_path": (
                    str(readiness.get("refresh_lock_path") or "").strip() or None
                ),
                "refresh_lock_acquired": (
                    bool(readiness.get("refresh_lock_acquired"))
                    if readiness.get("refresh_lock_acquired") is not None
                    else None
                ),
                "refresh_state_path": (
                    str(readiness.get("refresh_state_path") or "").strip() or None
                ),
                "refresh_last_attempt_ts": (
                    str(readiness.get("refresh_last_attempt_ts") or "").strip() or None
                ),
                "refresh_last_success_ts": (
                    str(readiness.get("refresh_last_success_ts") or "").strip() or None
                ),
                "refresh_lock_contention_attempted": (
                    bool(refresh_lock_contention.get("attempted"))
                    if refresh_lock_contention.get("attempted") is not None
                    else None
                ),
                "refresh_lock_contention_ok": (
                    bool(refresh_lock_contention.get("ok"))
                    if refresh_lock_contention.get("ok") is not None
                    else None
                ),
                "refresh_lock_contention_reason": (
                    str(refresh_lock_contention.get("reason") or "").strip() or None
                ),
                "refresh_lock_contention_returncode": (
                    _int_value(refresh_lock_contention.get("returncode"), 0)
                    if refresh_lock_contention.get("returncode") is not None
                    else None
                ),
                "refresh_lock_contention_duration_sec": _as_float(
                    refresh_lock_contention.get("duration_sec")
                ),
                "refresh_lock_contention_json_path": (
                    str(refresh_lock_contention.get("json_path") or "").strip() or None
                ),
                "refresh_lock_contention_md_path": (
                    str(refresh_lock_contention.get("md_path") or "").strip() or None
                ),
                "refresh_lock_contention_csv_path": (
                    str(refresh_lock_contention.get("csv_path") or "").strip() or None
                ),
                "refresh_lock_contention_source_coverage_csv": (
                    str(refresh_lock_contention.get("source_coverage_csv") or "").strip() or None
                ),
                "refresh_lock_contention_gate_level": (
                    str(refresh_lock_contention.get("gate_level") or "").strip() or None
                ),
                "refresh_lock_contention_gate_fail_triggered": (
                    bool(refresh_lock_contention.get("gate_fail_triggered"))
                    if refresh_lock_contention.get("gate_fail_triggered") is not None
                    else None
                ),
                "refresh_lock_contention_gate_source_bucket_view": (
                    str(refresh_lock_contention.get("gate_source_bucket_view") or "").strip() or None
                ),
                "refresh_lock_contention_gate_source_bucket_requested": (
                    str(refresh_lock_contention.get("gate_source_bucket_requested") or "").strip()
                    or None
                ),
                "refresh_lock_contention_gate_source_bucket_effective": (
                    str(refresh_lock_contention.get("gate_source_bucket_effective") or "").strip()
                    or None
                ),
                "refresh_lock_contention_gate_sample_guard_level": (
                    str(refresh_lock_contention.get("gate_sample_guard_level") or "").strip() or None
                ),
                "refresh_lock_contention_gate_sample_guard_fail_triggered": (
                    bool(refresh_lock_contention.get("gate_sample_guard_fail_triggered"))
                    if refresh_lock_contention.get("gate_sample_guard_fail_triggered") is not None
                    else None
                ),
                "refresh_lock_contention_gate_sample_guard_fail_on_level": (
                    str(refresh_lock_contention.get("gate_sample_guard_fail_on_level") or "").strip()
                    or None
                ),
                "lock_contention_enforce": (
                    bool(lock_contention_gate.get("enforce"))
                    if lock_contention_gate.get("enforce") is not None
                    else None
                ),
                "lock_contention_fail_closed": (
                    bool(lock_contention_gate.get("fail_closed"))
                    if lock_contention_gate.get("fail_closed") is not None
                    else None
                ),
                "lock_contention_report_path": (
                    str(lock_contention_gate.get("report_path") or "").strip() or None
                ),
                "lock_contention_report_present": (
                    bool(lock_contention_gate.get("report_present"))
                    if lock_contention_gate.get("report_present") is not None
                    else None
                ),
                "lock_contention_overall_level": (
                    str(lock_contention_gate.get("overall_level") or "").strip() or None
                ),
                "lock_contention_source_bucket_view": (
                    str(lock_contention_gate.get("source_bucket_view") or "").strip() or None
                ),
                "lock_contention_source_bucket_requested": (
                    str(lock_contention_gate.get("source_bucket_requested") or "").strip() or None
                ),
                "lock_contention_source_bucket_effective": (
                    str(lock_contention_gate.get("source_bucket_effective") or "").strip() or None
                ),
                "lock_contention_fail_triggered": (
                    bool(lock_contention_gate.get("fail_triggered"))
                    if lock_contention_gate.get("fail_triggered") is not None
                    else None
                ),
                "lock_contention_generated_at": (
                    str(lock_contention_gate.get("generated_at") or "").strip() or None
                ),
                "lock_contention_report_age_hours": _as_float(lock_contention_gate.get("report_age_hours")),
                "lock_contention_report_stale": (
                    bool(lock_contention_gate.get("report_stale"))
                    if lock_contention_gate.get("report_stale") is not None
                    else None
                ),
                "lock_contention_max_age_hours": _as_float(lock_contention_gate.get("max_age_hours")),
                "lock_contention_max_allowed_level": (
                    str(lock_contention_gate.get("max_allowed_level") or "").strip() or None
                ),
                "lock_contention_required_source_bucket": (
                    str(lock_contention_gate.get("required_source_bucket") or "").strip() or None
                ),
                "lock_contention_require_bucket_match": (
                    bool(lock_contention_gate.get("require_bucket_match"))
                    if lock_contention_gate.get("require_bucket_match") is not None
                    else None
                ),
                "lock_contention_require_sample_guard": (
                    bool(lock_contention_gate.get("require_sample_guard"))
                    if lock_contention_gate.get("require_sample_guard") is not None
                    else None
                ),
                "lock_contention_sample_guard_level": (
                    str(lock_contention_gate.get("sample_guard_level") or "").strip() or None
                ),
                "lock_contention_sample_guard_fail_triggered": (
                    bool(lock_contention_gate.get("sample_guard_fail_triggered"))
                    if lock_contention_gate.get("sample_guard_fail_triggered") is not None
                    else None
                ),
                "lock_contention_sample_guard_fail_on_level": (
                    str(lock_contention_gate.get("sample_guard_fail_on_level") or "").strip()
                    or None
                ),
                "lock_contention_blocked": (
                    bool(lock_contention_gate.get("blocked"))
                    if lock_contention_gate.get("blocked") is not None
                    else None
                ),
                "lock_contention_reason": (
                    str(lock_contention_gate.get("reason") or "").strip() or None
                ),
                "core_reason": core_reason,
                "pulse_lock_path": str(pulse_lock.get("path") or "").strip() or None,
                "pulse_lock_acquired": (
                    bool(pulse_lock.get("acquired"))
                    if pulse_lock.get("acquired") is not None
                    else None
                ),
                "pulse_lock_reason": str(pulse_lock.get("reason") or "").strip() or None,
            }
        )

        if core_reason == "halfhour_pulse_lock_busy" or out.get("pulse_lock_acquired") is False:
            out["status"] = "degraded"
            out["reason"] = "pulse_lock_busy"
        elif gate_blocked and out.get("refresh_reason") == "refresh_lock_busy":
            out["status"] = "degraded"
            out["reason"] = "paper_mode_refresh_lock_busy"
        elif gate_blocked and out.get("refresh_reason") == "refresh_throttled":
            out["status"] = "degraded"
            out["reason"] = "paper_mode_refresh_throttled"
        elif gate_blocked:
            out["status"] = "degraded"
            out["reason"] = "paper_mode_gate_blocked"
        elif readiness_status == "ready" and bool(readiness.get("ready_for_paper_mode")):
            out["status"] = "ok"
            out["reason"] = "paper_mode_ready"
        elif (readiness_status == "unknown") and bool(enforce) and bool(fail_closed):
            out["status"] = "degraded"
            out["reason"] = "paper_mode_unknown_fail_closed"
        else:
            out["status"] = "ok"
            out["reason"] = "paper_mode_monitor_only"
        return out

    if bool(tail.get("dry_run")):
        out["status"] = "dry_run"
        out["reason"] = "orchestrator_dry_run"
        return out

    if rc != 0:
        out["status"] = "degraded"
        out["reason"] = "step_failed"
    else:
        out["status"] = "unknown"
        out["reason"] = "missing_payload"
    return out


def _extract_core_paper_artifacts_sync_summary(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "status": "unknown",
        "reason": "step_missing",
        "source": None,
        "returncode": None,
        "duration_sec": None,
        "attempted": None,
        "ok": None,
        "sync_reason": None,
        "event_source": None,
        "as_of": None,
        "paper_positions_path": None,
        "broker_snapshot_path": None,
        "paper_positions_written": None,
        "broker_snapshot_written": None,
        "position_rows": None,
        "lock_path": None,
        "lock_acquired": None,
        "lock_wait_sec": None,
        "lock_timeout_sec": None,
        "lock_retry_sec": None,
        "lock_reason": None,
        "stale_guard_blocked": None,
        "existing_as_of": None,
        "target_as_of": None,
    }
    if not isinstance(steps, list):
        return out
    step = next(
        (x for x in steps if isinstance(x, dict) and str(x.get("name") or "") == "core"),
        None,
    )
    if not isinstance(step, dict):
        return out

    out["present"] = True
    out["returncode"] = step.get("returncode")
    out["duration_sec"] = step.get("duration_sec")
    rc = _int_value(step.get("returncode"), 1)
    tail = step.get("json_tail") if isinstance(step.get("json_tail"), dict) else {}

    if tail and str(tail.get("domain") or "") == "lie_spot_core":
        out["source"] = "core_json_tail"
        sync = (
            tail.get("paper_artifacts_sync")
            if isinstance(tail.get("paper_artifacts_sync"), dict)
            else {}
        )
        if not sync:
            out["status"] = "unknown"
            out["reason"] = "missing_paper_artifacts_sync"
            return out

        sync_reason = str(sync.get("reason") or "").strip() or None
        event_source = (
            str(sync.get("event_source") or "").strip()
            or str(tail.get("event_source") or "").strip()
            or None
        )
        out.update(
            {
                "attempted": (
                    bool(sync.get("attempted"))
                    if sync.get("attempted") is not None
                    else None
                ),
                "ok": bool(sync.get("ok")) if sync.get("ok") is not None else None,
                "sync_reason": sync_reason,
                "event_source": event_source,
                "as_of": str(sync.get("as_of") or "").strip() or None,
                "paper_positions_path": str(sync.get("paper_positions_path") or "").strip() or None,
                "broker_snapshot_path": str(sync.get("broker_snapshot_path") or "").strip() or None,
                "paper_positions_written": (
                    bool(sync.get("paper_positions_written"))
                    if sync.get("paper_positions_written") is not None
                    else None
                ),
                "broker_snapshot_written": (
                    bool(sync.get("broker_snapshot_written"))
                    if sync.get("broker_snapshot_written") is not None
                    else None
                ),
                "position_rows": (
                    _int_value(sync.get("position_rows"), 0)
                    if sync.get("position_rows") is not None
                    else None
                ),
                "lock_path": str(sync.get("lock_path") or "").strip() or None,
                "lock_acquired": (
                    bool(sync.get("lock_acquired"))
                    if sync.get("lock_acquired") is not None
                    else None
                ),
                "lock_wait_sec": _as_float(sync.get("lock_wait_sec")),
                "lock_timeout_sec": _as_float(sync.get("lock_timeout_sec")),
                "lock_retry_sec": _as_float(sync.get("lock_retry_sec")),
                "lock_reason": str(sync.get("lock_reason") or "").strip() or None,
                "stale_guard_blocked": (
                    bool(sync.get("stale_guard_blocked"))
                    if sync.get("stale_guard_blocked") is not None
                    else None
                ),
                "existing_as_of": str(sync.get("existing_as_of") or "").strip() or None,
                "target_as_of": str(sync.get("target_as_of") or "").strip() or None,
            }
        )

        if out.get("ok") is True:
            out["status"] = "ok"
            out["reason"] = "paper_artifacts_synced"
        elif sync_reason == "skipped_due_to_pulse_lock_busy":
            out["status"] = "degraded"
            out["reason"] = "pulse_lock_busy"
        elif out.get("stale_guard_blocked") is True or sync_reason == "stale_write_guard_blocked":
            out["status"] = "degraded"
            out["reason"] = "paper_artifacts_stale_guard_blocked"
        elif (
            sync_reason in {"artifacts_lock_busy", "artifacts_lock_unavailable", "lock_unavailable", "lock_timeout"}
            or str(out.get("lock_reason") or "").strip() in {"lock_timeout", "lock_unavailable", "lock_busy"}
            or out.get("lock_acquired") is False
        ):
            out["status"] = "degraded"
            out["reason"] = "paper_artifacts_lock_busy"
        elif out.get("attempted") is False:
            out["status"] = "degraded"
            out["reason"] = "paper_artifacts_not_attempted"
        else:
            out["status"] = "degraded"
            out["reason"] = "paper_artifacts_sync_failed"
        return out

    if bool(tail.get("dry_run")):
        out["status"] = "dry_run"
        out["reason"] = "orchestrator_dry_run"
        return out

    if rc != 0:
        out["status"] = "degraded"
        out["reason"] = "step_failed"
    else:
        out["status"] = "unknown"
        out["reason"] = "missing_payload"
    return out


def _derive_ops_next_action_fields(event: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return {
            "ops_next_action": "none",
            "ops_next_action_reason": "invalid_event",
            "ops_next_action_priority": "p3",
            "ops_next_action_secondary": None,
            "ops_next_action_context": {},
            "ops_next_action_source": "derived",
        }

    existing_priority = _normalize_ops_priority(event.get("ops_next_action_priority"))
    existing_reason = str(event.get("ops_next_action_reason") or "").strip()
    if existing_priority and existing_reason:
        return {
            "ops_next_action": _normalize_action(event.get("ops_next_action")) or "none",
            "ops_next_action_reason": existing_reason,
            "ops_next_action_priority": existing_priority,
            "ops_next_action_secondary": _normalize_action(event.get("ops_next_action_secondary")) or None,
            "ops_next_action_context": (
                event.get("ops_next_action_context")
                if isinstance(event.get("ops_next_action_context"), dict)
                else {}
            ),
            "ops_next_action_source": "event",
        }

    core_status = str(event.get("core_proxy_bypass_status") or "").strip().lower()
    core_execution_status = str(event.get("core_execution_status") or "").strip().lower()
    core_execution_reason = str(event.get("core_execution_reason") or "").strip().lower()
    core_execution_decision = str(event.get("core_execution_decision") or "").strip().lower()
    core_execution_order_http = _int_value(event.get("core_execution_order_http"), 0)
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
    bypass_failed = max(0, _int_value(event.get("core_proxy_bypass_bypass_failed"), 0))
    no_proxy_retry_exhausted = max(
        0, _int_value(event.get("core_proxy_bypass_no_proxy_retry_exhausted"), 0)
    )
    hint_exception = max(0, _int_value(event.get("core_proxy_bypass_hint_exception"), 0))
    hint_response = max(0, _int_value(event.get("core_proxy_bypass_hint_response"), 0))
    patch_failed_jobs = max(0, _int_value(event.get("guard_mode_patch_apply_failed_jobs"), 0))
    retry_exhausted_degraded = _int_env(
        "PI_OPS_NEXT_ACTION_PROXY_RETRY_EXHAUSTED_DEGRADED", 1, minimum=1
    )
    retry_exhausted_critical = _int_env(
        "PI_OPS_NEXT_ACTION_PROXY_RETRY_EXHAUSTED_CRITICAL", 2, minimum=1
    )

    next_action = "none"
    reason = "stable"
    priority = "p3"
    secondary = patch_next_secondary or None

    if core_status == "critical" or proxy_trend_status == "critical":
        next_action = f"run_batch:{isolation_batch}" if isolation_batch else "stabilize_proxy_path_and_isolate_jobs"
        reason = "core_proxy_bypass_or_proxy_error_critical"
        priority = "p0"
        if patch_next:
            secondary = patch_next
    elif core_status == "degraded":
        if (
            no_proxy_retry_exhausted >= retry_exhausted_critical
            and proxy_trend_status in {"degraded", "critical"}
        ):
            next_action = (
                f"run_batch:{isolation_batch}" if isolation_batch else "stabilize_proxy_path_and_isolate_jobs"
            )
            reason = "core_proxy_retry_exhausted_with_proxy_trend"
            priority = "p0"
            if patch_next:
                secondary = patch_next
        elif no_proxy_retry_exhausted >= retry_exhausted_degraded:
            next_action = "stabilize_proxy_path_and_raise_timeout_floor"
            reason = "core_proxy_retry_exhausted"
            priority = "p1"
            if patch_next:
                secondary = patch_next
        elif bypass_failed > 0 or hint_exception > 0:
            next_action = "stabilize_proxy_path"
            reason = "core_proxy_bypass_degraded_with_failures"
            priority = "p1"
            if patch_next:
                secondary = patch_next
        else:
            next_action = patch_next or "observe_core_proxy_bypass"
            reason = "core_proxy_bypass_degraded"
            priority = "p2"
    elif core_execution_status == "critical":
        next_action = "stabilize_execution_path_and_isolate_order_router"
        reason = "core_execution_critical"
        priority = "p0"
        if patch_next:
            secondary = patch_next
    elif core_execution_status == "degraded":
        if core_execution_reason in {
            "exec_error",
            "order_result_error",
            "order_http_non_2xx",
            "missing_order_result",
            "missing_order_http",
        }:
            next_action = "stabilize_execution_path"
            reason = "core_execution_degraded_with_errors"
            priority = "p1"
            if patch_next:
                secondary = patch_next
        else:
            next_action = patch_next or "observe_core_execution"
            reason = "core_execution_degraded"
            priority = "p2"
    elif patch_apply_status == "critical" or patch_failed_jobs > 0:
        next_action = "inspect_patch_status_critical"
        reason = "patch_apply_failure_detected"
        priority = "p1"
    elif recommend_mismatch and guard_recommended:
        next_action = f"align_guard_mode:{guard_recommended}"
        reason = "guard_mode_recommend_mismatch"
        priority = "p2"
    elif patch_next:
        next_action = patch_next
        reason = "patch_rollout_pending"
        priority = "p2"

    if secondary == next_action:
        secondary = None

    return {
        "ops_next_action": next_action,
        "ops_next_action_reason": reason,
        "ops_next_action_priority": priority,
        "ops_next_action_secondary": secondary,
        "ops_next_action_context": {
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
        "ops_next_action_source": "derived",
    }


def _parse_csv_set(raw: str, default: List[str]) -> set[str]:
    txt = str(raw or "").strip()
    if not txt:
        return {str(x).strip() for x in default if str(x).strip()}
    return {part.strip() for part in txt.split(",") if part.strip()}


def acquire_lock(path: Path, stale_sec: int) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.time()

    if path.exists():
        try:
            age = ts - path.stat().st_mtime
        except Exception:
            age = 0
        if age > float(stale_sec):
            try:
                path.unlink()
            except Exception:
                pass

    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {"ok": False, "reason": "lock_busy", "path": str(path)}
    except Exception as e:
        return {"ok": False, "reason": f"lock_error:{type(e).__name__}", "path": str(path)}

    payload = {"pid": os.getpid(), "ts": now_iso()}
    try:
        os.write(fd, (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    return {"ok": True, "reason": "lock_acquired", "path": str(path)}


def release_lock(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def resolve_guard_mode(requested: str) -> Dict[str, Optional[str]]:
    def _latest_cron_gate_rollout_signal() -> Dict[str, Any]:
        explicit = str(os.getenv("PI_CYCLE_CRON_HEALTH_PATH", "")).strip()
        source_path: Optional[Path] = None
        if explicit:
            source_path = Path(explicit)
        else:
            try:
                candidates = sorted(
                    DEFAULT_REVIEW_DIR.glob("*_pi_cron_health.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except Exception:
                candidates = []
            if candidates:
                source_path = candidates[0]

        out: Dict[str, Any] = {
            "status": "unknown",
            "reason": "missing",
            "action_level": "observe",
            "action_reason": "missing",
            "source_path": str(source_path) if source_path else None,
            "report_ts": None,
            "patch_status": "unknown",
            "patch_reason": "missing",
            "patch_actionable_jobs": 0,
            "patch_changed_jobs": 0,
            "patch_failed_jobs": 0,
            "patch_rollout_strategy": None,
            "patch_rollout_non_empty_batches": 0,
            "patch_rollout_apply_order_count": 0,
            "patch_rollout_next_batch": None,
            "patch_isolation_enabled": False,
            "patch_isolation_candidates_total": 0,
            "patch_isolation_actionable_candidates": 0,
            "patch_isolation_critical_candidates": 0,
            "patch_isolation_manual_only_candidates": 0,
            "patch_isolation_rollout_strategy": None,
            "patch_isolation_rollout_next_batch": None,
            "patch_isolation_rollout_pending": False,
            "patch_pending": False,
            "patch_action_level": "observe",
            "patch_action_reason": "missing",
            "patch_action_hints": [],
            "patch_next_action": "wait_for_patch_draft",
            "patch_next_action_secondary": None,
            "patch_apply_present": False,
            "patch_apply_status": "unknown",
            "patch_apply_reason": "missing",
            "patch_apply_batch_id": None,
            "patch_apply_mode": None,
            "patch_apply_applied_to_source": False,
            "patch_apply_apply_skipped_reason": None,
            "patch_apply_selected_actions": 0,
            "patch_apply_changed_jobs": 0,
            "patch_apply_failed_jobs": 0,
            "patch_apply_blocked_jobs": 0,
            "patch_apply_operations_total": 0,
            "patch_apply_operations_applied": 0,
            "patch_apply_operations_changed": 0,
            "patch_apply_trend_status": "unknown",
            "patch_apply_trend_reason": "missing",
            "patch_apply_trend_events": 0,
            "patch_apply_trend_problem_events": 0,
            "patch_apply_trend_problem_rate": None,
            "patch_apply_trend_critical_problem_events": 0,
            "patch_apply_trend_critical_problem_rate": None,
            "patch_apply_trend_target_batch_id": None,
            "proxy_control_trend_status": "unknown",
            "proxy_control_trend_reason": "missing",
            "proxy_control_trend_action_level": "observe",
            "proxy_control_trend_action_reason": "missing",
            "proxy_control_trend_trigger_count": 0,
            "proxy_control_trend_trigger_rate": None,
            "proxy_control_trend_dominant_trigger": None,
            "proxy_control_trend_critical_trigger_count": 0,
            "proxy_error_status": "unknown",
            "proxy_error_reason": "missing",
            "proxy_error_action_level": "observe",
            "proxy_error_action_reason": "missing",
            "proxy_error_jobs_count": 0,
            "proxy_error_core_jobs_count": 0,
            "proxy_error_max_consecutive_errors": 0,
            "proxy_error_trend_status": "unknown",
            "proxy_error_trend_reason": "missing",
            "proxy_error_trend_action_level": "observe",
            "proxy_error_trend_action_reason": "missing",
            "proxy_error_trend_jobs_with_errors": 0,
            "proxy_error_trend_core_jobs_with_errors": 0,
            "proxy_error_trend_max_consecutive_errors": 0,
            "proxy_error_trend_rate": None,
        }
        if source_path is None or not source_path.exists():
            return out
        try:
            obj = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception:
            out["status"] = "degraded"
            out["reason"] = "unreadable"
            out["action_level"] = "degrade"
            out["action_reason"] = "unreadable"
            return out
        if not isinstance(obj, dict):
            out["status"] = "degraded"
            out["reason"] = "invalid_payload"
            out["action_level"] = "degrade"
            out["action_reason"] = "invalid_payload"
            return out
        checks = obj.get("self_checks") if isinstance(obj.get("self_checks"), dict) else {}
        gate = checks.get("gate_rollout") if isinstance(checks.get("gate_rollout"), dict) else {}
        patch = (
            checks.get("cron_policy_patch_draft")
            if isinstance(checks.get("cron_policy_patch_draft"), dict)
            else {}
        )
        proxy_error = checks.get("proxy_error") if isinstance(checks.get("proxy_error"), dict) else {}
        proxy_error_trend = (
            checks.get("proxy_error_trend")
            if isinstance(checks.get("proxy_error_trend"), dict)
            else {}
        )
        status = str(gate.get("rollback_trend_status") or gate.get("status") or "unknown").strip().lower()
        if status not in {"ok", "degraded", "critical", "unknown"}:
            status = "unknown"
        action_level = str(gate.get("rollback_trend_action_level") or "observe").strip().lower()
        if action_level not in {"observe", "degrade", "shadow_lock"}:
            action_level = "observe"
        proxy_control_trend_status = str(gate.get("proxy_control_trend_status") or "unknown").strip().lower()
        if proxy_control_trend_status not in {"ok", "degraded", "critical", "unknown"}:
            proxy_control_trend_status = "unknown"
        proxy_control_trend_action_level = str(gate.get("proxy_control_trend_action_level") or "observe").strip().lower()
        if proxy_control_trend_action_level not in {"observe", "degrade", "shadow_lock"}:
            proxy_control_trend_action_level = "observe"
        proxy_control_trend_reason = str(gate.get("proxy_control_trend_reason") or "unknown")
        proxy_control_trend_action_reason = str(gate.get("proxy_control_trend_action_reason") or "unknown")
        proxy_control_trend_dominant_trigger = (
            str(gate.get("proxy_control_trend_dominant_trigger") or "").strip() or None
        )
        proxy_control_trigger_counts = (
            gate.get("proxy_control_trend_trigger_counts")
            if isinstance(gate.get("proxy_control_trend_trigger_counts"), dict)
            else {}
        )
        proxy_control_trend_trigger_count = max(0, int(gate.get("proxy_control_trend_trigger_count") or 0))
        proxy_control_trend_critical_trigger_count = max(
            0,
            int(proxy_control_trigger_counts.get("proxy_isolation_critical_candidates") or 0),
        )
        proxy_error_status = str(proxy_error.get("status") or "unknown").strip().lower()
        if proxy_error_status not in {"ok", "degraded", "critical", "unknown"}:
            proxy_error_status = "unknown"
        proxy_error_action_level = str(proxy_error.get("action_level") or "observe").strip().lower()
        if proxy_error_action_level not in {"observe", "degrade", "shadow_lock"}:
            proxy_error_action_level = "observe"
        proxy_error_jobs_count = max(0, int(proxy_error.get("proxy_error_jobs_count") or 0))
        proxy_error_core_jobs_count = max(0, int(proxy_error.get("proxy_error_core_jobs_count") or 0))
        proxy_error_max_consecutive_errors = max(0, int(proxy_error.get("max_consecutive_errors") or 0))

        proxy_error_trend_status = str(proxy_error_trend.get("status") or "unknown").strip().lower()
        if proxy_error_trend_status not in {"ok", "degraded", "critical", "unknown"}:
            proxy_error_trend_status = "unknown"
        proxy_error_trend_action_level = str(proxy_error_trend.get("action_level") or "observe").strip().lower()
        if proxy_error_trend_action_level not in {"observe", "degrade", "shadow_lock"}:
            proxy_error_trend_action_level = "observe"
        proxy_error_trend_jobs_with_errors = max(
            0, int(proxy_error_trend.get("jobs_with_proxy_errors") or 0)
        )
        proxy_error_trend_core_jobs_with_errors = max(
            0, int(proxy_error_trend.get("core_jobs_with_proxy_errors") or 0)
        )
        proxy_error_trend_max_consecutive_errors = max(
            0, int(proxy_error_trend.get("max_consecutive_proxy_errors") or 0)
        )
        patch_apply_status = str(gate.get("cron_policy_apply_batch_status") or "unknown").strip().lower()
        if patch_apply_status not in {"ok", "degraded", "critical", "dry_run", "noop", "unknown"}:
            patch_apply_status = "unknown"
        patch_apply_batch_id = str(gate.get("cron_policy_apply_batch_batch_id") or "").strip() or None
        patch_apply_mode = str(gate.get("cron_policy_apply_batch_mode") or "").strip() or None
        patch_apply_reason = str(gate.get("cron_policy_apply_batch_reason") or "unknown")
        patch_apply_present = bool(gate.get("cron_policy_apply_batch_present", False))
        patch_apply_applied_to_source = bool(gate.get("cron_policy_apply_batch_applied_to_source", False))
        patch_apply_apply_skipped_reason = (
            str(gate.get("cron_policy_apply_batch_apply_skipped_reason") or "").strip() or None
        )
        patch_apply_selected_actions = max(0, int(gate.get("cron_policy_apply_batch_selected_actions") or 0))
        patch_apply_changed_jobs = max(0, int(gate.get("cron_policy_apply_batch_changed_jobs") or 0))
        patch_apply_failed_jobs = max(0, int(gate.get("cron_policy_apply_batch_failed_jobs") or 0))
        patch_apply_blocked_jobs = max(0, int(gate.get("cron_policy_apply_batch_blocked_jobs") or 0))
        patch_apply_operations_total = max(0, int(gate.get("cron_policy_apply_batch_operations_total") or 0))
        patch_apply_operations_applied = max(
            0, int(gate.get("cron_policy_apply_batch_operations_applied") or 0)
        )
        patch_apply_operations_changed = max(
            0, int(gate.get("cron_policy_apply_batch_operations_changed") or 0)
        )
        patch_apply_trend_status = str(gate.get("notify_trend_patch_apply_status") or "unknown").strip().lower()
        if patch_apply_trend_status not in {"ok", "degraded", "critical", "unknown"}:
            patch_apply_trend_status = "unknown"
        patch_apply_trend_reason = str(gate.get("notify_trend_patch_apply_status_reason") or "unknown")
        patch_apply_trend_events = max(0, int(gate.get("notify_trend_patch_apply_events") or 0))
        patch_apply_trend_problem_events = max(
            0, int(gate.get("notify_trend_patch_apply_problem_events") or 0)
        )
        patch_apply_trend_critical_problem_events = max(
            0, int(gate.get("notify_trend_patch_apply_critical_problem_events") or 0)
        )
        patch_apply_trend_target_batch_id = (
            str(gate.get("notify_trend_patch_apply_target_batch_id") or "").strip() or None
        )

        # External proxy instability is a direct actuator risk; fold it into proxy-control guard pressure.
        proxy_pressure_level = 0
        proxy_pressure_reason = ""
        if proxy_error_trend_status == "critical":
            proxy_pressure_level = 2
            proxy_pressure_reason = "proxy_error_trend_critical"
        elif proxy_error_trend_status == "degraded":
            proxy_pressure_level = 1
            proxy_pressure_reason = "proxy_error_trend_degraded"
        elif proxy_error_status == "critical":
            proxy_pressure_level = 2
            proxy_pressure_reason = "proxy_error_immediate_critical"
        elif proxy_error_status == "degraded":
            proxy_pressure_level = 1
            proxy_pressure_reason = "proxy_error_immediate_degraded"

        if proxy_pressure_level >= 2:
            proxy_control_trend_status = "critical"
            proxy_control_trend_action_level = "shadow_lock"
            proxy_control_trend_reason = proxy_pressure_reason
            proxy_control_trend_action_reason = proxy_pressure_reason
            if proxy_control_trend_dominant_trigger is None:
                proxy_control_trend_dominant_trigger = (
                    "proxy_error_trend" if proxy_error_trend_status in {"degraded", "critical"} else "proxy_error"
                )
            if proxy_control_trend_critical_trigger_count <= 0:
                proxy_control_trend_critical_trigger_count = 1
            if proxy_control_trend_trigger_count <= 0:
                proxy_control_trend_trigger_count = max(
                    1,
                    proxy_error_trend_jobs_with_errors,
                    proxy_error_jobs_count,
                )
        elif proxy_pressure_level == 1 and proxy_control_trend_status in {"ok", "unknown"}:
            proxy_control_trend_status = "degraded"
            if proxy_control_trend_action_level != "shadow_lock":
                proxy_control_trend_action_level = "degrade"
            proxy_control_trend_reason = proxy_pressure_reason
            proxy_control_trend_action_reason = proxy_pressure_reason
            if proxy_control_trend_dominant_trigger is None:
                proxy_control_trend_dominant_trigger = (
                    "proxy_error_trend" if proxy_error_trend_status in {"degraded", "critical"} else "proxy_error"
                )
            if proxy_control_trend_trigger_count <= 0:
                proxy_control_trend_trigger_count = max(
                    1,
                    proxy_error_trend_jobs_with_errors,
                    proxy_error_jobs_count,
                )
        patch_status = str(patch.get("status") or "unknown").strip().lower()
        if patch_status not in {"ok", "degraded", "critical", "unknown"}:
            patch_status = "unknown"
        patch_actionable_jobs = int(patch.get("actionable_jobs") or 0)
        patch_changed_jobs = int(patch.get("changed_jobs") or 0)
        patch_failed_jobs = int(patch.get("failed_jobs") or 0)
        patch_rollout_strategy = patch.get("rollout_strategy")
        patch_rollout_non_empty_batches = int(patch.get("rollout_non_empty_batches") or 0)
        patch_rollout_apply_order_count = int(patch.get("rollout_apply_order_count") or 0)
        patch_isolation_enabled = bool(patch.get("isolation_enabled", False))
        patch_isolation_candidates_total = int(patch.get("isolation_candidates_total") or 0)
        patch_isolation_actionable_candidates = int(patch.get("isolation_actionable_candidates") or 0)
        patch_isolation_critical_candidates = int(patch.get("isolation_critical_candidates") or 0)
        patch_isolation_manual_only_candidates = int(patch.get("isolation_manual_only_candidates") or 0)
        patch_isolation_rollout_strategy = (
            str(patch.get("isolation_rollout_strategy") or "").strip() or None
        )
        patch_isolation_rollout_next_batch = (
            str(patch.get("isolation_rollout_next_batch") or "").strip() or None
        )
        patch_isolation_rollout_pending = bool(patch.get("isolation_rollout_pending", False))
        patch_next_batch = None
        rollout_batches_top = patch.get("rollout_batches_top") if isinstance(patch.get("rollout_batches_top"), list) else []
        for item in rollout_batches_top:
            if not isinstance(item, dict):
                continue
            try:
                cnt = int(item.get("count") or 0)
                ops = int(item.get("ops") or 0)
            except Exception:
                cnt = 0
                ops = 0
            if cnt > 0 and ops > 0:
                patch_next_batch = str(item.get("id") or "") or None
                break
        patch_pending = bool(
            patch_actionable_jobs > 0 or patch_rollout_non_empty_batches > 0 or patch_isolation_rollout_pending
        )
        patch_action_level = "observe"
        patch_action_reason = "no_pending_changes"
        patch_action_hints: List[str] = []
        patch_next_action = "none"
        patch_next_action_secondary = None
        if patch_status == "critical":
            patch_action_level = "degrade"
            patch_action_reason = "patch_status_critical"
            patch_action_hints.append("patch: freeze rollout and inspect draft status before applying any batch.")
            patch_next_action = "inspect_patch_status_critical"
        elif patch_isolation_critical_candidates > 0 and patch_isolation_actionable_candidates > 0:
            patch_action_level = "shadow_lock"
            patch_action_reason = "proxy_isolation_critical_candidates"
            patch_action_hints.append(
                f"proxy isolation: critical non-core candidates={patch_isolation_critical_candidates}; apply isolation before rollout."
            )
            patch_next_action = "run_proxy_isolation_batch"
        elif patch_isolation_actionable_candidates > 0:
            patch_action_level = "degrade"
            patch_action_reason = "proxy_isolation_candidates_pending"
            patch_action_hints.append(
                f"proxy isolation: actionable candidates={patch_isolation_actionable_candidates}; execute isolation batch first."
            )
            patch_next_action = "run_proxy_isolation_batch"
        elif patch_failed_jobs > 0:
            patch_action_level = "degrade"
            patch_action_reason = "patch_failed_jobs"
            patch_action_hints.append(
                f"patch: resolve failed jobs first (failed_jobs={patch_failed_jobs}) before applying next batch."
            )
            patch_next_action = "resolve_patch_failures"
        elif patch_pending and patch_status == "degraded":
            patch_action_level = "degrade"
            patch_action_reason = "patch_pending_degraded"
            patch_action_hints.append("patch: run guarded rollout; keep core batch blocked until canary succeeds.")
            patch_next_action = "run_guarded_patch_rollout"
        elif patch_pending:
            patch_action_level = "observe"
            patch_action_reason = "patch_pending"
            patch_action_hints.append("patch: pending rollout exists; continue canary->high->core order.")
            patch_next_action = "continue_patch_rollout"
        isolation_batch = patch_isolation_rollout_next_batch
        if not isolation_batch and patch_isolation_actionable_candidates > 0:
            isolation_batch = "batch_0_proxy_isolation"
        if isolation_batch and patch_next_action == "run_proxy_isolation_batch":
            patch_next_action = f"run_batch:{isolation_batch}"
            patch_action_hints.append(f"proxy isolation: execute `{isolation_batch}` first.")
        if patch_next_batch:
            if patch_isolation_actionable_candidates > 0:
                patch_next_action_secondary = f"run_batch:{patch_next_batch}"
                patch_action_hints.append(
                    f"patch: after isolation batch, continue with `{patch_next_batch}`."
                )
            else:
                patch_action_hints.append(f"patch: next batch `{patch_next_batch}` should run first.")
                patch_next_action = f"run_batch:{patch_next_batch}"
        if patch_isolation_enabled and patch_isolation_manual_only_candidates > 0:
            patch_action_hints.append(
                f"proxy isolation: manual-only candidates={patch_isolation_manual_only_candidates}; keep operator review loop."
            )
        if patch_rollout_strategy:
            patch_action_hints.append(f"patch: follow rollout strategy `{patch_rollout_strategy}`.")
        if patch_isolation_rollout_strategy:
            patch_action_hints.append(f"proxy isolation: follow `{patch_isolation_rollout_strategy}`.")
        out.update(
            {
                "status": status,
                "reason": str(gate.get("rollback_trend_reason") or gate.get("reason") or "unknown"),
                "action_level": action_level,
                "action_reason": str(gate.get("rollback_trend_action_reason") or "unknown"),
                "report_ts": obj.get("ts"),
                "patch_status": patch_status,
                "patch_reason": str(patch.get("reason") or "unknown"),
                "patch_actionable_jobs": patch_actionable_jobs,
                "patch_changed_jobs": patch_changed_jobs,
                "patch_failed_jobs": patch_failed_jobs,
                "patch_rollout_strategy": patch_rollout_strategy,
                "patch_rollout_non_empty_batches": patch_rollout_non_empty_batches,
                "patch_rollout_apply_order_count": patch_rollout_apply_order_count,
                "patch_rollout_next_batch": patch_next_batch,
                "patch_isolation_enabled": patch_isolation_enabled,
                "patch_isolation_candidates_total": patch_isolation_candidates_total,
                "patch_isolation_actionable_candidates": patch_isolation_actionable_candidates,
                "patch_isolation_critical_candidates": patch_isolation_critical_candidates,
                "patch_isolation_manual_only_candidates": patch_isolation_manual_only_candidates,
                "patch_isolation_rollout_strategy": patch_isolation_rollout_strategy,
                "patch_isolation_rollout_next_batch": isolation_batch,
                "patch_isolation_rollout_pending": patch_isolation_rollout_pending,
                "patch_pending": patch_pending,
                "patch_action_level": patch_action_level,
                "patch_action_reason": patch_action_reason,
                "patch_action_hints": patch_action_hints[:5],
                "patch_next_action": patch_next_action,
                "patch_next_action_secondary": patch_next_action_secondary,
                "patch_apply_present": patch_apply_present,
                "patch_apply_status": patch_apply_status,
                "patch_apply_reason": patch_apply_reason,
                "patch_apply_batch_id": patch_apply_batch_id,
                "patch_apply_mode": patch_apply_mode,
                "patch_apply_applied_to_source": patch_apply_applied_to_source,
                "patch_apply_apply_skipped_reason": patch_apply_apply_skipped_reason,
                "patch_apply_selected_actions": patch_apply_selected_actions,
                "patch_apply_changed_jobs": patch_apply_changed_jobs,
                "patch_apply_failed_jobs": patch_apply_failed_jobs,
                "patch_apply_blocked_jobs": patch_apply_blocked_jobs,
                "patch_apply_operations_total": patch_apply_operations_total,
                "patch_apply_operations_applied": patch_apply_operations_applied,
                "patch_apply_operations_changed": patch_apply_operations_changed,
                "patch_apply_trend_status": patch_apply_trend_status,
                "patch_apply_trend_reason": patch_apply_trend_reason,
                "patch_apply_trend_events": patch_apply_trend_events,
                "patch_apply_trend_problem_events": patch_apply_trend_problem_events,
                "patch_apply_trend_problem_rate": _as_float(gate.get("notify_trend_patch_apply_problem_rate")),
                "patch_apply_trend_critical_problem_events": patch_apply_trend_critical_problem_events,
                "patch_apply_trend_critical_problem_rate": _as_float(
                    gate.get("notify_trend_patch_apply_critical_problem_rate")
                ),
                "patch_apply_trend_target_batch_id": patch_apply_trend_target_batch_id,
                "proxy_control_trend_status": proxy_control_trend_status,
                "proxy_control_trend_reason": proxy_control_trend_reason,
                "proxy_control_trend_action_level": proxy_control_trend_action_level,
                "proxy_control_trend_action_reason": proxy_control_trend_action_reason,
                "proxy_control_trend_trigger_count": proxy_control_trend_trigger_count,
                "proxy_control_trend_trigger_rate": _as_float(gate.get("proxy_control_trend_trigger_rate")),
                "proxy_control_trend_dominant_trigger": proxy_control_trend_dominant_trigger,
                "proxy_control_trend_critical_trigger_count": proxy_control_trend_critical_trigger_count,
                "proxy_error_status": proxy_error_status,
                "proxy_error_reason": str(proxy_error.get("reason") or "unknown"),
                "proxy_error_action_level": proxy_error_action_level,
                "proxy_error_action_reason": str(proxy_error.get("action_reason") or "unknown"),
                "proxy_error_jobs_count": proxy_error_jobs_count,
                "proxy_error_core_jobs_count": proxy_error_core_jobs_count,
                "proxy_error_max_consecutive_errors": proxy_error_max_consecutive_errors,
                "proxy_error_trend_status": proxy_error_trend_status,
                "proxy_error_trend_reason": str(proxy_error_trend.get("reason") or "unknown"),
                "proxy_error_trend_action_level": proxy_error_trend_action_level,
                "proxy_error_trend_action_reason": str(
                    proxy_error_trend.get("action_reason") or "unknown"
                ),
                "proxy_error_trend_jobs_with_errors": proxy_error_trend_jobs_with_errors,
                "proxy_error_trend_core_jobs_with_errors": proxy_error_trend_core_jobs_with_errors,
                "proxy_error_trend_max_consecutive_errors": proxy_error_trend_max_consecutive_errors,
                "proxy_error_trend_rate": _as_float(proxy_error_trend.get("proxy_error_rate")),
            }
        )
        return out

    def _latest_digital_life_signal() -> Dict[str, Any]:
        explicit = str(os.getenv("PI_CYCLE_DIGITAL_LIFE_STATE_PATH", "")).strip()
        source_path = (
            Path(explicit)
            if explicit
            else Path(
                os.getenv(
                    "PI_DIGITAL_LIFE_STATE_PATH",
                    str(SYSTEM_ROOT / "output" / "logs" / "digital_life_state.json"),
                )
            )
        )
        out: Dict[str, Any] = {
            "status": "unknown",
            "reason": "missing",
            "lifecycle_mode": None,
            "next_action": None,
            "viability_score": None,
            "stress_score": None,
            "source_path": str(source_path),
            "report_ts": None,
        }
        if not source_path.exists():
            return out
        try:
            obj = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception:
            out["status"] = "degraded"
            out["reason"] = "unreadable"
            return out
        if not isinstance(obj, dict):
            out["status"] = "degraded"
            out["reason"] = "invalid_payload"
            return out
        status = str(obj.get("status") or "unknown").strip().lower()
        if status not in {"ok", "degraded", "critical", "unknown"}:
            status = "unknown"
        lifecycle_mode_raw = str(obj.get("lifecycle_mode") or "").strip().upper()
        lifecycle_mode = lifecycle_mode_raw if lifecycle_mode_raw in {"SURVIVE", "STABILIZE", "ADAPT", "EXPLORE"} else None
        out.update(
            {
                "status": status,
                "reason": str(obj.get("reason") or ("ok" if status in {"ok", "degraded"} else status)),
                "lifecycle_mode": lifecycle_mode,
                "next_action": str(obj.get("next_action") or "").strip() or None,
                "viability_score": _as_float(obj.get("viability_score")),
                "stress_score": _as_float(obj.get("stress_score")),
                "report_ts": obj.get("ts"),
            }
        )
        return out

    digital_signal = _latest_digital_life_signal()

    if requested != "auto":
        signal = _latest_cron_gate_rollout_signal()
        return {
            "requested": requested,
            "effective": requested,
            "base_effective": requested,
            "recommended": requested,
            "recommend_reason": "manual_guard_mode",
            "auto_full_hhmm": None,
            "auto_override_applied": False,
            "use_rollback_action": False,
            "apply_rollback_action": False,
            "use_patch_draft_action": False,
            "apply_patch_draft_action": False,
            "use_patch_apply_action": False,
            "apply_patch_apply_action": False,
            "use_proxy_control_action": False,
            "apply_proxy_control_action": False,
            "use_digital_life_action": False,
            "apply_digital_life_action": False,
            "patch_pending_threshold": None,
            "proxy_control_trigger_threshold": None,
            "patch_apply_target_batch": None,
            "rollback_status": signal.get("status"),
            "rollback_reason": signal.get("reason"),
            "rollback_action_level": signal.get("action_level"),
            "rollback_action_reason": signal.get("action_reason"),
            "rollback_source_path": signal.get("source_path"),
            "rollback_report_ts": signal.get("report_ts"),
            "proxy_control_trend_status": signal.get("proxy_control_trend_status"),
            "proxy_control_trend_reason": signal.get("proxy_control_trend_reason"),
            "proxy_control_trend_action_level": signal.get("proxy_control_trend_action_level"),
            "proxy_control_trend_action_reason": signal.get("proxy_control_trend_action_reason"),
            "proxy_control_trend_trigger_count": signal.get("proxy_control_trend_trigger_count"),
            "proxy_control_trend_trigger_rate": signal.get("proxy_control_trend_trigger_rate"),
            "proxy_control_trend_dominant_trigger": signal.get("proxy_control_trend_dominant_trigger"),
            "proxy_control_trend_critical_trigger_count": signal.get(
                "proxy_control_trend_critical_trigger_count"
            ),
            "proxy_error_status": signal.get("proxy_error_status"),
            "proxy_error_reason": signal.get("proxy_error_reason"),
            "proxy_error_action_level": signal.get("proxy_error_action_level"),
            "proxy_error_action_reason": signal.get("proxy_error_action_reason"),
            "proxy_error_jobs_count": signal.get("proxy_error_jobs_count"),
            "proxy_error_core_jobs_count": signal.get("proxy_error_core_jobs_count"),
            "proxy_error_max_consecutive_errors": signal.get("proxy_error_max_consecutive_errors"),
            "proxy_error_trend_status": signal.get("proxy_error_trend_status"),
            "proxy_error_trend_reason": signal.get("proxy_error_trend_reason"),
            "proxy_error_trend_action_level": signal.get("proxy_error_trend_action_level"),
            "proxy_error_trend_action_reason": signal.get("proxy_error_trend_action_reason"),
            "proxy_error_trend_jobs_with_errors": signal.get("proxy_error_trend_jobs_with_errors"),
            "proxy_error_trend_core_jobs_with_errors": signal.get(
                "proxy_error_trend_core_jobs_with_errors"
            ),
            "proxy_error_trend_max_consecutive_errors": signal.get(
                "proxy_error_trend_max_consecutive_errors"
            ),
            "proxy_error_trend_rate": signal.get("proxy_error_trend_rate"),
            "patch_draft_status": signal.get("patch_status"),
            "patch_draft_reason": signal.get("patch_reason"),
            "patch_draft_actionable_jobs": signal.get("patch_actionable_jobs"),
            "patch_draft_changed_jobs": signal.get("patch_changed_jobs"),
            "patch_draft_failed_jobs": signal.get("patch_failed_jobs"),
            "patch_draft_rollout_strategy": signal.get("patch_rollout_strategy"),
            "patch_draft_rollout_non_empty_batches": signal.get("patch_rollout_non_empty_batches"),
            "patch_draft_rollout_apply_order_count": signal.get("patch_rollout_apply_order_count"),
            "patch_draft_rollout_next_batch": signal.get("patch_rollout_next_batch"),
            "patch_draft_isolation_enabled": signal.get("patch_isolation_enabled"),
            "patch_draft_isolation_candidates_total": signal.get("patch_isolation_candidates_total"),
            "patch_draft_isolation_actionable_candidates": signal.get("patch_isolation_actionable_candidates"),
            "patch_draft_isolation_critical_candidates": signal.get("patch_isolation_critical_candidates"),
            "patch_draft_isolation_manual_only_candidates": signal.get("patch_isolation_manual_only_candidates"),
            "patch_draft_isolation_rollout_strategy": signal.get("patch_isolation_rollout_strategy"),
            "patch_draft_isolation_rollout_next_batch": signal.get("patch_isolation_rollout_next_batch"),
            "patch_draft_isolation_rollout_pending": signal.get("patch_isolation_rollout_pending"),
            "patch_draft_pending": signal.get("patch_pending"),
            "patch_draft_action_level": signal.get("patch_action_level"),
            "patch_draft_action_reason": signal.get("patch_action_reason"),
            "patch_draft_action_hints": signal.get("patch_action_hints"),
            "patch_draft_next_action": signal.get("patch_next_action"),
            "patch_draft_next_action_secondary": signal.get("patch_next_action_secondary"),
            "patch_apply_present": signal.get("patch_apply_present"),
            "patch_apply_status": signal.get("patch_apply_status"),
            "patch_apply_reason": signal.get("patch_apply_reason"),
            "patch_apply_batch_id": signal.get("patch_apply_batch_id"),
            "patch_apply_mode": signal.get("patch_apply_mode"),
            "patch_apply_applied_to_source": signal.get("patch_apply_applied_to_source"),
            "patch_apply_apply_skipped_reason": signal.get("patch_apply_apply_skipped_reason"),
            "patch_apply_selected_actions": signal.get("patch_apply_selected_actions"),
            "patch_apply_changed_jobs": signal.get("patch_apply_changed_jobs"),
            "patch_apply_failed_jobs": signal.get("patch_apply_failed_jobs"),
            "patch_apply_blocked_jobs": signal.get("patch_apply_blocked_jobs"),
            "patch_apply_operations_total": signal.get("patch_apply_operations_total"),
            "patch_apply_operations_applied": signal.get("patch_apply_operations_applied"),
            "patch_apply_operations_changed": signal.get("patch_apply_operations_changed"),
            "patch_apply_trend_status": signal.get("patch_apply_trend_status"),
            "patch_apply_trend_reason": signal.get("patch_apply_trend_reason"),
            "patch_apply_trend_events": signal.get("patch_apply_trend_events"),
            "patch_apply_trend_problem_events": signal.get("patch_apply_trend_problem_events"),
            "patch_apply_trend_problem_rate": signal.get("patch_apply_trend_problem_rate"),
            "patch_apply_trend_critical_problem_events": signal.get(
                "patch_apply_trend_critical_problem_events"
            ),
            "patch_apply_trend_critical_problem_rate": signal.get(
                "patch_apply_trend_critical_problem_rate"
            ),
            "patch_apply_trend_target_batch_id": signal.get("patch_apply_trend_target_batch_id"),
            "digital_life_status": digital_signal.get("status"),
            "digital_life_reason": digital_signal.get("reason"),
            "digital_life_lifecycle_mode": digital_signal.get("lifecycle_mode"),
            "digital_life_next_action": digital_signal.get("next_action"),
            "digital_life_viability_score": digital_signal.get("viability_score"),
            "digital_life_stress_score": digital_signal.get("stress_score"),
            "digital_life_source_path": digital_signal.get("source_path"),
            "digital_life_report_ts": digital_signal.get("report_ts"),
        }

    now_local = dt.datetime.now().astimezone()
    full_hhmm = os.getenv("PI_CYCLE_FULL_HHMM", "20:30").strip()
    base_effective = "full" if now_local.strftime("%H:%M") == full_hhmm else "fast"
    effective = base_effective
    recommended = base_effective
    recommend_reason = "auto_schedule"
    auto_override_applied = False
    use_rollback_action = env_flag("PI_CYCLE_AUTO_GUARD_USE_ROLLBACK_ACTION", default=True)
    apply_rollback_action = env_flag("PI_CYCLE_AUTO_GUARD_APPLY_ROLLBACK_ACTION", default=False)
    use_patch_draft_action = env_flag("PI_CYCLE_AUTO_GUARD_USE_PATCH_DRAFT_ACTION", default=True)
    apply_patch_draft_action = env_flag("PI_CYCLE_AUTO_GUARD_APPLY_PATCH_DRAFT_ACTION", default=False)
    use_patch_apply_action = env_flag("PI_CYCLE_AUTO_GUARD_USE_PATCH_APPLY_ACTION", default=True)
    apply_patch_apply_action = env_flag("PI_CYCLE_AUTO_GUARD_APPLY_PATCH_APPLY_ACTION", default=False)
    use_proxy_control_action = env_flag("PI_CYCLE_AUTO_GUARD_USE_PROXY_CONTROL_ACTION", default=True)
    apply_proxy_control_action = env_flag("PI_CYCLE_AUTO_GUARD_APPLY_PROXY_CONTROL_ACTION", default=False)
    use_digital_life_action = env_flag("PI_CYCLE_AUTO_GUARD_USE_DIGITAL_LIFE_ACTION", default=True)
    apply_digital_life_action = env_flag("PI_CYCLE_AUTO_GUARD_APPLY_DIGITAL_LIFE_ACTION", default=False)
    patch_pending_threshold = _int_env("PI_CYCLE_AUTO_GUARD_PATCH_PENDING_THRESHOLD", 2, minimum=1)
    proxy_control_trigger_threshold = _int_env("PI_CYCLE_AUTO_GUARD_PROXY_CONTROL_TRIGGER_THRESHOLD", 1, minimum=1)
    patch_apply_target_batch = (
        str(os.getenv("PI_CYCLE_AUTO_GUARD_PATCH_APPLY_TARGET_BATCH", "batch_0_proxy_isolation")).strip()
        or "batch_0_proxy_isolation"
    )
    signal = _latest_cron_gate_rollout_signal()

    if use_rollback_action:
        action_level = str(signal.get("action_level") or "observe")
        if action_level in {"degrade", "shadow_lock"}:
            recommended = "full"
            recommend_reason = f"rollback_action_{action_level}"
        elif str(signal.get("status") or "") == "critical":
            recommended = "full"
            recommend_reason = "rollback_status_critical"
        if apply_rollback_action and recommended == "full" and base_effective != "full":
            effective = "full"
            auto_override_applied = True

    if use_proxy_control_action and recommended != "full":
        proxy_status = str(signal.get("proxy_control_trend_status") or "unknown")
        proxy_action_level = str(signal.get("proxy_control_trend_action_level") or "observe")
        proxy_trigger_count = max(0, int(signal.get("proxy_control_trend_trigger_count") or 0))
        proxy_critical_trigger_count = max(0, int(signal.get("proxy_control_trend_critical_trigger_count") or 0))
        if proxy_action_level == "shadow_lock":
            recommended = "full"
            recommend_reason = "proxy_control_action_shadow_lock"
        elif proxy_status == "critical":
            recommended = "full"
            recommend_reason = "proxy_control_status_critical"
        elif proxy_action_level == "degrade" and proxy_trigger_count >= proxy_control_trigger_threshold:
            recommended = "full"
            recommend_reason = "proxy_control_action_degrade"
        elif proxy_critical_trigger_count > 0:
            recommended = "full"
            recommend_reason = "proxy_control_critical_trigger"

    if (
        apply_proxy_control_action
        and recommended == "full"
        and str(recommend_reason).startswith("proxy_control_")
        and base_effective != "full"
    ):
        effective = "full"
        auto_override_applied = True

    if use_patch_draft_action and recommended != "full":
        patch_status = str(signal.get("patch_status") or "unknown")
        patch_failed_jobs = max(0, int(signal.get("patch_failed_jobs") or 0))
        patch_actionable_jobs = max(0, int(signal.get("patch_actionable_jobs") or 0))
        patch_isolation_actionable = max(0, int(signal.get("patch_isolation_actionable_candidates") or 0))
        patch_isolation_critical = max(0, int(signal.get("patch_isolation_critical_candidates") or 0))
        patch_pending = bool(signal.get("patch_pending"))
        if patch_status == "critical":
            recommended = "full"
            recommend_reason = "patch_status_critical"
        elif patch_isolation_critical > 0:
            recommended = "full"
            recommend_reason = "patch_proxy_isolation_critical"
        elif patch_pending and patch_isolation_actionable > 0:
            recommended = "full"
            recommend_reason = "patch_proxy_isolation_pending"
        elif patch_failed_jobs > 0:
            recommended = "full"
            recommend_reason = "patch_failed_jobs"
        elif patch_pending and patch_status == "degraded":
            recommended = "full"
            recommend_reason = "patch_pending_degraded"
        elif patch_pending and patch_status == "ok" and patch_actionable_jobs >= patch_pending_threshold:
            recommended = "full"
            recommend_reason = "patch_pending_threshold"

    if (
        apply_patch_draft_action
        and recommended == "full"
        and str(recommend_reason).startswith("patch_")
        and base_effective != "full"
    ):
        effective = "full"
        auto_override_applied = True

    if use_patch_apply_action and recommended != "full":
        patch_apply_batch_id = str(signal.get("patch_apply_batch_id") or "")
        patch_apply_status = str(signal.get("patch_apply_status") or "unknown").strip().lower()
        patch_apply_mode = str(signal.get("patch_apply_mode") or "").strip().lower()
        patch_apply_failed_jobs = max(0, int(signal.get("patch_apply_failed_jobs") or 0))
        patch_apply_blocked_jobs = max(0, int(signal.get("patch_apply_blocked_jobs") or 0))
        patch_apply_selected_actions = max(0, int(signal.get("patch_apply_selected_actions") or 0))
        patch_apply_changed_jobs = max(0, int(signal.get("patch_apply_changed_jobs") or 0))
        patch_apply_operations_changed = max(0, int(signal.get("patch_apply_operations_changed") or 0))
        patch_apply_trend_status = str(signal.get("patch_apply_trend_status") or "unknown").strip().lower()
        patch_apply_trend_target_batch = str(signal.get("patch_apply_trend_target_batch_id") or "").strip()
        is_target_batch = (
            patch_apply_batch_id == patch_apply_target_batch
            or patch_apply_trend_target_batch == patch_apply_target_batch
        )
        if is_target_batch:
            if patch_apply_trend_status == "critical":
                recommended = "full"
                recommend_reason = "patch_apply_trend_critical"
            elif patch_apply_failed_jobs > 0 or patch_apply_blocked_jobs > 0:
                recommended = "full"
                recommend_reason = "patch_apply_batch_failed_or_blocked"
            elif patch_apply_status in {"critical", "degraded"}:
                recommended = "full"
                recommend_reason = f"patch_apply_status_{patch_apply_status}"
            elif (
                patch_apply_mode == "apply"
                and patch_apply_selected_actions > 0
                and patch_apply_changed_jobs <= 0
                and patch_apply_operations_changed <= 0
            ):
                recommended = "full"
                recommend_reason = "patch_apply_no_effect_apply"
            elif patch_apply_trend_status == "degraded":
                recommended = "full"
                recommend_reason = "patch_apply_trend_degraded"

    if (
        apply_patch_apply_action
        and recommended == "full"
        and str(recommend_reason).startswith("patch_apply_")
        and base_effective != "full"
    ):
        effective = "full"
        auto_override_applied = True

    if use_digital_life_action:
        digital_status = str(digital_signal.get("status") or "unknown")
        lifecycle_mode = str(digital_signal.get("lifecycle_mode") or "")
        digital_recommended = False
        digital_recommend_reason = ""
        if digital_status == "critical":
            digital_recommended = True
            digital_recommend_reason = "digital_life_status_critical"
        elif digital_status in {"ok", "degraded"} and lifecycle_mode in {"SURVIVE", "STABILIZE"}:
            digital_recommended = True
            digital_recommend_reason = f"digital_life_{lifecycle_mode.lower()}"

        # Keep higher-priority reasons (rollback/proxy/patch) stable in the audit trail.
        if digital_recommended and recommended != "full":
            recommended = "full"
            recommend_reason = digital_recommend_reason
        if apply_digital_life_action and digital_recommended and base_effective != "full":
            effective = "full"
            auto_override_applied = True

    return {
        "requested": requested,
        "effective": effective,
        "base_effective": base_effective,
        "recommended": recommended,
        "recommend_reason": recommend_reason,
        "auto_full_hhmm": full_hhmm,
        "auto_override_applied": auto_override_applied,
        "use_rollback_action": use_rollback_action,
        "apply_rollback_action": apply_rollback_action,
        "use_patch_draft_action": use_patch_draft_action,
        "apply_patch_draft_action": apply_patch_draft_action,
        "use_patch_apply_action": use_patch_apply_action,
        "apply_patch_apply_action": apply_patch_apply_action,
        "use_proxy_control_action": use_proxy_control_action,
        "apply_proxy_control_action": apply_proxy_control_action,
        "use_digital_life_action": use_digital_life_action,
        "apply_digital_life_action": apply_digital_life_action,
        "patch_pending_threshold": patch_pending_threshold,
        "proxy_control_trigger_threshold": proxy_control_trigger_threshold,
        "patch_apply_target_batch": patch_apply_target_batch,
        "rollback_status": signal.get("status"),
        "rollback_reason": signal.get("reason"),
        "rollback_action_level": signal.get("action_level"),
        "rollback_action_reason": signal.get("action_reason"),
        "rollback_source_path": signal.get("source_path"),
        "rollback_report_ts": signal.get("report_ts"),
        "proxy_control_trend_status": signal.get("proxy_control_trend_status"),
        "proxy_control_trend_reason": signal.get("proxy_control_trend_reason"),
        "proxy_control_trend_action_level": signal.get("proxy_control_trend_action_level"),
        "proxy_control_trend_action_reason": signal.get("proxy_control_trend_action_reason"),
        "proxy_control_trend_trigger_count": signal.get("proxy_control_trend_trigger_count"),
        "proxy_control_trend_trigger_rate": signal.get("proxy_control_trend_trigger_rate"),
        "proxy_control_trend_dominant_trigger": signal.get("proxy_control_trend_dominant_trigger"),
        "proxy_control_trend_critical_trigger_count": signal.get("proxy_control_trend_critical_trigger_count"),
        "proxy_error_status": signal.get("proxy_error_status"),
        "proxy_error_reason": signal.get("proxy_error_reason"),
        "proxy_error_action_level": signal.get("proxy_error_action_level"),
        "proxy_error_action_reason": signal.get("proxy_error_action_reason"),
        "proxy_error_jobs_count": signal.get("proxy_error_jobs_count"),
        "proxy_error_core_jobs_count": signal.get("proxy_error_core_jobs_count"),
        "proxy_error_max_consecutive_errors": signal.get("proxy_error_max_consecutive_errors"),
        "proxy_error_trend_status": signal.get("proxy_error_trend_status"),
        "proxy_error_trend_reason": signal.get("proxy_error_trend_reason"),
        "proxy_error_trend_action_level": signal.get("proxy_error_trend_action_level"),
        "proxy_error_trend_action_reason": signal.get("proxy_error_trend_action_reason"),
        "proxy_error_trend_jobs_with_errors": signal.get("proxy_error_trend_jobs_with_errors"),
        "proxy_error_trend_core_jobs_with_errors": signal.get("proxy_error_trend_core_jobs_with_errors"),
        "proxy_error_trend_max_consecutive_errors": signal.get(
            "proxy_error_trend_max_consecutive_errors"
        ),
        "proxy_error_trend_rate": signal.get("proxy_error_trend_rate"),
        "patch_draft_status": signal.get("patch_status"),
        "patch_draft_reason": signal.get("patch_reason"),
        "patch_draft_actionable_jobs": signal.get("patch_actionable_jobs"),
        "patch_draft_changed_jobs": signal.get("patch_changed_jobs"),
        "patch_draft_failed_jobs": signal.get("patch_failed_jobs"),
        "patch_draft_rollout_strategy": signal.get("patch_rollout_strategy"),
        "patch_draft_rollout_non_empty_batches": signal.get("patch_rollout_non_empty_batches"),
        "patch_draft_rollout_apply_order_count": signal.get("patch_rollout_apply_order_count"),
        "patch_draft_rollout_next_batch": signal.get("patch_rollout_next_batch"),
        "patch_draft_isolation_enabled": signal.get("patch_isolation_enabled"),
        "patch_draft_isolation_candidates_total": signal.get("patch_isolation_candidates_total"),
        "patch_draft_isolation_actionable_candidates": signal.get("patch_isolation_actionable_candidates"),
        "patch_draft_isolation_critical_candidates": signal.get("patch_isolation_critical_candidates"),
        "patch_draft_isolation_manual_only_candidates": signal.get("patch_isolation_manual_only_candidates"),
        "patch_draft_isolation_rollout_strategy": signal.get("patch_isolation_rollout_strategy"),
        "patch_draft_isolation_rollout_next_batch": signal.get("patch_isolation_rollout_next_batch"),
        "patch_draft_isolation_rollout_pending": signal.get("patch_isolation_rollout_pending"),
        "patch_draft_pending": signal.get("patch_pending"),
        "patch_draft_action_level": signal.get("patch_action_level"),
        "patch_draft_action_reason": signal.get("patch_action_reason"),
        "patch_draft_action_hints": signal.get("patch_action_hints"),
        "patch_draft_next_action": signal.get("patch_next_action"),
        "patch_draft_next_action_secondary": signal.get("patch_next_action_secondary"),
        "patch_apply_present": signal.get("patch_apply_present"),
        "patch_apply_status": signal.get("patch_apply_status"),
        "patch_apply_reason": signal.get("patch_apply_reason"),
        "patch_apply_batch_id": signal.get("patch_apply_batch_id"),
        "patch_apply_mode": signal.get("patch_apply_mode"),
        "patch_apply_applied_to_source": signal.get("patch_apply_applied_to_source"),
        "patch_apply_apply_skipped_reason": signal.get("patch_apply_apply_skipped_reason"),
        "patch_apply_selected_actions": signal.get("patch_apply_selected_actions"),
        "patch_apply_changed_jobs": signal.get("patch_apply_changed_jobs"),
        "patch_apply_failed_jobs": signal.get("patch_apply_failed_jobs"),
        "patch_apply_blocked_jobs": signal.get("patch_apply_blocked_jobs"),
        "patch_apply_operations_total": signal.get("patch_apply_operations_total"),
        "patch_apply_operations_applied": signal.get("patch_apply_operations_applied"),
        "patch_apply_operations_changed": signal.get("patch_apply_operations_changed"),
        "patch_apply_trend_status": signal.get("patch_apply_trend_status"),
        "patch_apply_trend_reason": signal.get("patch_apply_trend_reason"),
        "patch_apply_trend_events": signal.get("patch_apply_trend_events"),
        "patch_apply_trend_problem_events": signal.get("patch_apply_trend_problem_events"),
        "patch_apply_trend_problem_rate": signal.get("patch_apply_trend_problem_rate"),
        "patch_apply_trend_critical_problem_events": signal.get(
            "patch_apply_trend_critical_problem_events"
        ),
        "patch_apply_trend_critical_problem_rate": signal.get(
            "patch_apply_trend_critical_problem_rate"
        ),
        "patch_apply_trend_target_batch_id": signal.get("patch_apply_trend_target_batch_id"),
        "digital_life_status": digital_signal.get("status"),
        "digital_life_reason": digital_signal.get("reason"),
        "digital_life_lifecycle_mode": digital_signal.get("lifecycle_mode"),
        "digital_life_next_action": digital_signal.get("next_action"),
        "digital_life_viability_score": digital_signal.get("viability_score"),
        "digital_life_stress_score": digital_signal.get("stress_score"),
        "digital_life_source_path": digital_signal.get("source_path"),
        "digital_life_report_ts": digital_signal.get("report_ts"),
    }


def _batch_id_from_patch_action(action: Any) -> Optional[str]:
    raw = str(action or "").strip()
    if not raw.startswith("run_batch:"):
        return None
    batch_id = raw.split(":", 1)[1].strip()
    return batch_id or None


def _resolve_patch_apply_batch_id(guard_mode: Optional[Dict[str, Any]]) -> str:
    override = str(os.getenv("PI_CYCLE_CRON_POLICY_APPLY_BATCH_ID", "")).strip()
    if override:
        direct = _batch_id_from_patch_action(override)
        return direct or override
    gm = guard_mode if isinstance(guard_mode, dict) else {}
    primary = _batch_id_from_patch_action(gm.get("patch_draft_next_action"))
    if primary:
        return primary
    secondary = _batch_id_from_patch_action(gm.get("patch_draft_next_action_secondary"))
    if secondary:
        return secondary
    return "auto"


def build_plan(
    args: argparse.Namespace,
    effective_guard_mode: str,
    guard_mode: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    cron_health_planned = False
    run_gateway_guard = os.getenv("PI_CYCLE_RUN_GATEWAY_GUARD", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if run_gateway_guard and not bool(args.skip_gateway_guard):
        plan.append(
            {
                "name": "gateway_guard",
                "cmd": ["python3", "scripts/gateway_singleton_guard.py"],
                "timeout": int(args.gateway_guard_timeout_sec),
            }
        )
    if not args.skip_core:
        plan.append(
            {
                "name": "core",
                "cmd": ["python3", "scripts/lie_spot_halfhour_core.py"],
                "timeout": int(args.core_timeout_sec),
            }
        )
    if not args.skip_watchdog:
        plan.append(
            {
                "name": "watchdog",
                "cmd": ["python3", "scripts/lie_spine_watchdog.py", "--once"],
                "timeout": int(args.watchdog_timeout_sec),
            }
        )
    if effective_guard_mode != "off":
        plan.append(
            {
                "name": "neuro_guard",
                "cmd": ["python3", "scripts/neuro_guard_cycle.py", "--mode", effective_guard_mode],
                "timeout": int(args.guard_timeout_sec),
            }
        )
    run_hip = os.getenv("PI_CYCLE_RUN_HIPPOCAMPUS_FULL", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        effective_guard_mode == "full"
        and run_hip
        and not bool(args.skip_hippocampus)
    ):
        plan.append(
            {
                "name": "hippocampus",
                "cmd": ["python3", "scripts/hippocampus.py"],
                "timeout": int(args.hippocampus_timeout_sec),
            }
        )
    run_digital_life = os.getenv("PI_CYCLE_RUN_DIGITAL_LIFE_CORE_FULL", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        effective_guard_mode == "full"
        and run_digital_life
        and not bool(args.skip_digital_life)
    ):
        plan.append(
            {
                "name": "digital_life_core",
                "cmd": ["python3", "scripts/digital_life_core.py", "--window-hours", "24"],
                "timeout": int(args.digital_life_timeout_sec),
            }
        )
    run_cron_health = os.getenv("PI_CYCLE_RUN_CRON_HEALTH_FULL", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        effective_guard_mode == "full"
        and run_cron_health
        and not bool(args.skip_cron_health)
    ):
        plan.append(
            {
                "name": "cron_health",
                "cmd": ["python3", "scripts/cron_health_snapshot.py"],
                "timeout": int(args.cron_health_timeout_sec),
            }
        )
        cron_health_planned = True
    run_cron_policy_patch_draft = os.getenv(
        "PI_CYCLE_RUN_CRON_POLICY_PATCH_DRAFT_FULL",
        "1",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        effective_guard_mode == "full"
        and run_cron_policy_patch_draft
        and cron_health_planned
        and not bool(args.skip_cron_policy_draft)
    ):
        plan.append(
            {
                "name": "cron_policy_patch_draft",
                "cmd": ["python3", "scripts/cron_policy_patch_draft.py"],
                "timeout": int(args.cron_policy_patch_draft_timeout_sec),
            }
        )
    run_cron_policy_apply_batch = os.getenv(
        "PI_CYCLE_RUN_CRON_POLICY_APPLY_BATCH_FULL",
        "0",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        effective_guard_mode == "full"
        and run_cron_policy_apply_batch
        and cron_health_planned
        and not bool(args.skip_cron_policy_apply)
    ):
        apply_batch_id = _resolve_patch_apply_batch_id(guard_mode)
        cmd = [
            "python3",
            "scripts/cron_policy_apply_batch.py",
            "--batch-id",
            str(apply_batch_id or "auto"),
        ]
        if env_flag("PI_CYCLE_CRON_POLICY_APPLY_BATCH_WRITE", default=False):
            cmd.append("--apply")
        if env_flag("PI_CYCLE_CRON_POLICY_APPLY_ALLOW_CORE_BATCH", default=False):
            cmd.append("--allow-core-batch")
        if env_flag("PI_CYCLE_CRON_POLICY_APPLY_ALLOW_DEGRADED", default=False):
            cmd.append("--allow-degraded-apply")
        max_actions = _int_env("PI_CYCLE_CRON_POLICY_APPLY_MAX_ACTIONS", 0, minimum=0)
        if max_actions > 0:
            cmd.extend(["--max-actions", str(max_actions)])
        plan.append(
            {
                "name": "cron_policy_apply_batch",
                "cmd": cmd,
                "timeout": int(args.cron_policy_apply_batch_timeout_sec),
            }
        )
    run_envelope_lint = os.getenv("PI_CYCLE_RUN_ENVELOPE_LINT_FULL", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if (
        effective_guard_mode == "full"
        and run_envelope_lint
        and not bool(args.skip_envelope_lint)
    ):
        envelope_cmd = ["python3", "scripts/envelope_lint.py", "--window-hours", "24"]
        hourly_v2_strict_since = str(
            os.getenv("PI_ENVELOPE_HOURLY_V2_STRICT_SINCE_UTC", "")
        ).strip()
        if hourly_v2_strict_since:
            envelope_cmd.extend(
                ["--hourly-v2-strict-since-utc", hourly_v2_strict_since]
            )
        plan.append(
            {
                "name": "envelope_lint",
                "cmd": envelope_cmd,
                "timeout": int(args.envelope_lint_timeout_sec),
            }
        )
    return plan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard-mode", choices=["fast", "full", "off", "auto"], default="fast")
    ap.add_argument("--skip-core", action="store_true")
    ap.add_argument("--skip-watchdog", action="store_true")
    ap.add_argument("--skip-gateway-guard", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--lock-stale-sec", type=int, default=3600)
    ap.add_argument("--gateway-guard-timeout-sec", type=int, default=30)
    ap.add_argument("--core-timeout-sec", type=int, default=300)
    ap.add_argument("--watchdog-timeout-sec", type=int, default=90)
    ap.add_argument("--guard-timeout-sec", type=int, default=240)
    ap.add_argument("--hippocampus-timeout-sec", type=int, default=180)
    ap.add_argument("--digital-life-timeout-sec", type=int, default=60)
    ap.add_argument("--cron-health-timeout-sec", type=int, default=120)
    ap.add_argument("--cron-policy-patch-draft-timeout-sec", type=int, default=60)
    ap.add_argument("--cron-policy-apply-batch-timeout-sec", type=int, default=90)
    ap.add_argument("--envelope-lint-timeout-sec", type=int, default=120)
    ap.add_argument("--skip-hippocampus", action="store_true")
    ap.add_argument("--skip-digital-life", action="store_true")
    ap.add_argument("--skip-cron-health", action="store_true")
    ap.add_argument("--skip-cron-policy-draft", action="store_true")
    ap.add_argument("--skip-cron-policy-apply", action="store_true")
    ap.add_argument("--skip-envelope-lint", action="store_true")
    args = ap.parse_args()

    started = time.time()
    guard_mode = resolve_guard_mode(args.guard_mode)
    effective_guard_mode = str(guard_mode.get("effective") or "fast")
    apply_batch_summary = _extract_cron_policy_apply_batch_summary([])
    lock = acquire_lock(LOCK_PATH, stale_sec=int(args.lock_stale_sec))
    if not lock.get("ok"):
        core_proxy_bypass_summary = _extract_core_proxy_bypass_summary([])
        core_execution_summary = _extract_core_execution_summary([])
        core_paper_mode_summary = _extract_core_paper_mode_readiness_summary([])
        core_paper_artifacts_summary = _extract_core_paper_artifacts_sync_summary([])
        event = {
            "envelope_version": "1.0",
            "domain": "pi_cycle",
            "ts": now_iso(),
            "status": "skipped",
            "reason": lock.get("reason"),
            "lock": lock,
            "guard_mode_requested": guard_mode.get("requested"),
            "guard_mode_effective": guard_mode.get("effective"),
            "guard_mode_base_effective": guard_mode.get("base_effective"),
            "guard_mode_recommended": guard_mode.get("recommended"),
            "guard_mode_recommend_reason": guard_mode.get("recommend_reason"),
            "auto_full_hhmm": guard_mode.get("auto_full_hhmm"),
            "guard_mode_auto_override_applied": guard_mode.get("auto_override_applied"),
            "guard_mode_use_rollback_action": guard_mode.get("use_rollback_action"),
            "guard_mode_apply_rollback_action": guard_mode.get("apply_rollback_action"),
            "guard_mode_use_patch_draft_action": guard_mode.get("use_patch_draft_action"),
            "guard_mode_apply_patch_draft_action": guard_mode.get("apply_patch_draft_action"),
            "guard_mode_use_patch_apply_action": guard_mode.get("use_patch_apply_action"),
            "guard_mode_apply_patch_apply_action": guard_mode.get("apply_patch_apply_action"),
            "guard_mode_use_proxy_control_action": guard_mode.get("use_proxy_control_action"),
            "guard_mode_apply_proxy_control_action": guard_mode.get("apply_proxy_control_action"),
            "guard_mode_use_digital_life_action": guard_mode.get("use_digital_life_action"),
            "guard_mode_apply_digital_life_action": guard_mode.get("apply_digital_life_action"),
            "guard_mode_patch_pending_threshold": guard_mode.get("patch_pending_threshold"),
            "guard_mode_proxy_control_trigger_threshold": guard_mode.get("proxy_control_trigger_threshold"),
            "guard_mode_patch_apply_target_batch": guard_mode.get("patch_apply_target_batch"),
            "guard_mode_rollback_status": guard_mode.get("rollback_status"),
            "guard_mode_rollback_reason": guard_mode.get("rollback_reason"),
            "guard_mode_rollback_action_level": guard_mode.get("rollback_action_level"),
            "guard_mode_rollback_action_reason": guard_mode.get("rollback_action_reason"),
            "guard_mode_rollback_source_path": guard_mode.get("rollback_source_path"),
            "guard_mode_rollback_report_ts": guard_mode.get("rollback_report_ts"),
            "guard_mode_proxy_control_trend_status": guard_mode.get("proxy_control_trend_status"),
            "guard_mode_proxy_control_trend_reason": guard_mode.get("proxy_control_trend_reason"),
            "guard_mode_proxy_control_trend_action_level": guard_mode.get("proxy_control_trend_action_level"),
            "guard_mode_proxy_control_trend_action_reason": guard_mode.get("proxy_control_trend_action_reason"),
            "guard_mode_proxy_control_trend_trigger_count": guard_mode.get(
                "proxy_control_trend_trigger_count"
            ),
            "guard_mode_proxy_control_trend_trigger_rate": guard_mode.get("proxy_control_trend_trigger_rate"),
            "guard_mode_proxy_control_trend_dominant_trigger": guard_mode.get(
                "proxy_control_trend_dominant_trigger"
            ),
            "guard_mode_proxy_control_trend_critical_trigger_count": guard_mode.get(
                "proxy_control_trend_critical_trigger_count"
            ),
            "guard_mode_proxy_error_status": guard_mode.get("proxy_error_status"),
            "guard_mode_proxy_error_reason": guard_mode.get("proxy_error_reason"),
            "guard_mode_proxy_error_action_level": guard_mode.get("proxy_error_action_level"),
            "guard_mode_proxy_error_action_reason": guard_mode.get("proxy_error_action_reason"),
            "guard_mode_proxy_error_jobs_count": guard_mode.get("proxy_error_jobs_count"),
            "guard_mode_proxy_error_core_jobs_count": guard_mode.get("proxy_error_core_jobs_count"),
            "guard_mode_proxy_error_max_consecutive_errors": guard_mode.get(
                "proxy_error_max_consecutive_errors"
            ),
            "guard_mode_proxy_error_trend_status": guard_mode.get("proxy_error_trend_status"),
            "guard_mode_proxy_error_trend_reason": guard_mode.get("proxy_error_trend_reason"),
            "guard_mode_proxy_error_trend_action_level": guard_mode.get(
                "proxy_error_trend_action_level"
            ),
            "guard_mode_proxy_error_trend_action_reason": guard_mode.get(
                "proxy_error_trend_action_reason"
            ),
            "guard_mode_proxy_error_trend_jobs_with_errors": guard_mode.get(
                "proxy_error_trend_jobs_with_errors"
            ),
            "guard_mode_proxy_error_trend_core_jobs_with_errors": guard_mode.get(
                "proxy_error_trend_core_jobs_with_errors"
            ),
            "guard_mode_proxy_error_trend_max_consecutive_errors": guard_mode.get(
                "proxy_error_trend_max_consecutive_errors"
            ),
            "guard_mode_proxy_error_trend_rate": guard_mode.get("proxy_error_trend_rate"),
            "guard_mode_patch_draft_status": guard_mode.get("patch_draft_status"),
            "guard_mode_patch_draft_reason": guard_mode.get("patch_draft_reason"),
            "guard_mode_patch_draft_actionable_jobs": guard_mode.get("patch_draft_actionable_jobs"),
            "guard_mode_patch_draft_changed_jobs": guard_mode.get("patch_draft_changed_jobs"),
            "guard_mode_patch_draft_failed_jobs": guard_mode.get("patch_draft_failed_jobs"),
            "guard_mode_patch_draft_rollout_strategy": guard_mode.get("patch_draft_rollout_strategy"),
            "guard_mode_patch_draft_rollout_non_empty_batches": guard_mode.get(
                "patch_draft_rollout_non_empty_batches"
            ),
            "guard_mode_patch_draft_rollout_apply_order_count": guard_mode.get(
                "patch_draft_rollout_apply_order_count"
            ),
            "guard_mode_patch_draft_rollout_next_batch": guard_mode.get("patch_draft_rollout_next_batch"),
            "guard_mode_patch_draft_pending": guard_mode.get("patch_draft_pending"),
            "guard_mode_patch_draft_action_level": guard_mode.get("patch_draft_action_level"),
            "guard_mode_patch_draft_action_reason": guard_mode.get("patch_draft_action_reason"),
            "guard_mode_patch_draft_action_hints": guard_mode.get("patch_draft_action_hints"),
            "guard_mode_patch_draft_next_action": guard_mode.get("patch_draft_next_action"),
            "guard_mode_patch_draft_next_action_secondary": guard_mode.get("patch_draft_next_action_secondary"),
            "guard_mode_patch_apply_present": guard_mode.get("patch_apply_present"),
            "guard_mode_patch_apply_status": guard_mode.get("patch_apply_status"),
            "guard_mode_patch_apply_reason": guard_mode.get("patch_apply_reason"),
            "guard_mode_patch_apply_batch_id": guard_mode.get("patch_apply_batch_id"),
            "guard_mode_patch_apply_mode": guard_mode.get("patch_apply_mode"),
            "guard_mode_patch_apply_applied_to_source": guard_mode.get("patch_apply_applied_to_source"),
            "guard_mode_patch_apply_apply_skipped_reason": guard_mode.get("patch_apply_apply_skipped_reason"),
            "guard_mode_patch_apply_selected_actions": guard_mode.get("patch_apply_selected_actions"),
            "guard_mode_patch_apply_changed_jobs": guard_mode.get("patch_apply_changed_jobs"),
            "guard_mode_patch_apply_failed_jobs": guard_mode.get("patch_apply_failed_jobs"),
            "guard_mode_patch_apply_blocked_jobs": guard_mode.get("patch_apply_blocked_jobs"),
            "guard_mode_patch_apply_operations_total": guard_mode.get("patch_apply_operations_total"),
            "guard_mode_patch_apply_operations_applied": guard_mode.get("patch_apply_operations_applied"),
            "guard_mode_patch_apply_operations_changed": guard_mode.get("patch_apply_operations_changed"),
            "guard_mode_patch_apply_trend_status": guard_mode.get("patch_apply_trend_status"),
            "guard_mode_patch_apply_trend_reason": guard_mode.get("patch_apply_trend_reason"),
            "guard_mode_patch_apply_trend_events": guard_mode.get("patch_apply_trend_events"),
            "guard_mode_patch_apply_trend_problem_events": guard_mode.get(
                "patch_apply_trend_problem_events"
            ),
            "guard_mode_patch_apply_trend_problem_rate": guard_mode.get("patch_apply_trend_problem_rate"),
            "guard_mode_patch_apply_trend_critical_problem_events": guard_mode.get(
                "patch_apply_trend_critical_problem_events"
            ),
            "guard_mode_patch_apply_trend_critical_problem_rate": guard_mode.get(
                "patch_apply_trend_critical_problem_rate"
            ),
            "guard_mode_patch_apply_trend_target_batch_id": guard_mode.get("patch_apply_trend_target_batch_id"),
            "guard_mode_digital_life_status": guard_mode.get("digital_life_status"),
            "guard_mode_digital_life_reason": guard_mode.get("digital_life_reason"),
            "guard_mode_digital_life_lifecycle_mode": guard_mode.get("digital_life_lifecycle_mode"),
            "guard_mode_digital_life_next_action": guard_mode.get("digital_life_next_action"),
            "guard_mode_digital_life_viability_score": guard_mode.get("digital_life_viability_score"),
            "guard_mode_digital_life_stress_score": guard_mode.get("digital_life_stress_score"),
            "guard_mode_digital_life_source_path": guard_mode.get("digital_life_source_path"),
            "guard_mode_digital_life_report_ts": guard_mode.get("digital_life_report_ts"),
            "guard_mode_patch_draft_isolation_enabled": guard_mode.get("patch_draft_isolation_enabled"),
            "guard_mode_patch_draft_isolation_candidates_total": guard_mode.get(
                "patch_draft_isolation_candidates_total"
            ),
            "guard_mode_patch_draft_isolation_actionable_candidates": guard_mode.get(
                "patch_draft_isolation_actionable_candidates"
            ),
            "guard_mode_patch_draft_isolation_critical_candidates": guard_mode.get(
                "patch_draft_isolation_critical_candidates"
            ),
            "guard_mode_patch_draft_isolation_manual_only_candidates": guard_mode.get(
                "patch_draft_isolation_manual_only_candidates"
            ),
            "guard_mode_patch_draft_isolation_rollout_strategy": guard_mode.get(
                "patch_draft_isolation_rollout_strategy"
            ),
            "guard_mode_patch_draft_isolation_rollout_next_batch": guard_mode.get(
                "patch_draft_isolation_rollout_next_batch"
            ),
            "guard_mode_patch_draft_isolation_rollout_pending": guard_mode.get(
                "patch_draft_isolation_rollout_pending"
            ),
            "skip_core": bool(args.skip_core),
            "skip_watchdog": bool(args.skip_watchdog),
            "skip_gateway_guard": bool(args.skip_gateway_guard),
            "skip_hippocampus": bool(args.skip_hippocampus),
            "skip_digital_life": bool(args.skip_digital_life),
            "skip_cron_health": bool(args.skip_cron_health),
            "skip_cron_policy_draft": bool(args.skip_cron_policy_draft),
            "skip_cron_policy_apply": bool(args.skip_cron_policy_apply),
            "skip_envelope_lint": bool(args.skip_envelope_lint),
            "digital_life_present": False,
            "digital_life_status": "unknown",
            "digital_life_reason": "step_missing",
            "digital_life_lifecycle_mode": None,
            "digital_life_next_action": None,
            "digital_life_viability_score": None,
            "digital_life_stress_score": None,
            "digital_life_state_path": None,
            "digital_life_events_path": None,
            "cron_policy_apply_batch_present": apply_batch_summary.get("present"),
            "cron_policy_apply_batch_status": apply_batch_summary.get("status"),
            "cron_policy_apply_batch_reason": apply_batch_summary.get("reason"),
            "cron_policy_apply_batch_batch_id": apply_batch_summary.get("batch_id"),
            "cron_policy_apply_batch_mode": apply_batch_summary.get("mode"),
            "cron_policy_apply_batch_applied_to_source": apply_batch_summary.get("applied_to_source"),
            "cron_policy_apply_batch_apply_skipped_reason": apply_batch_summary.get("apply_skipped_reason"),
            "cron_policy_apply_batch_selected_actions": apply_batch_summary.get("selected_actions"),
            "cron_policy_apply_batch_changed_jobs": apply_batch_summary.get("changed_jobs"),
            "cron_policy_apply_batch_failed_jobs": apply_batch_summary.get("failed_jobs"),
            "cron_policy_apply_batch_blocked_jobs": apply_batch_summary.get("blocked_jobs"),
            "cron_policy_apply_batch_operations_total": apply_batch_summary.get("operations_total"),
            "cron_policy_apply_batch_operations_applied": apply_batch_summary.get("operations_applied"),
            "cron_policy_apply_batch_operations_changed": apply_batch_summary.get("operations_changed"),
            "cron_policy_apply_batch_output_path": apply_batch_summary.get("output_path"),
            "cron_policy_apply_batch_candidate_jobs_output": apply_batch_summary.get(
                "candidate_jobs_output"
            ),
            "core_proxy_bypass_present": core_proxy_bypass_summary.get("present"),
            "core_proxy_bypass_status": core_proxy_bypass_summary.get("status"),
            "core_proxy_bypass_reason": core_proxy_bypass_summary.get("reason"),
            "core_proxy_bypass_source": core_proxy_bypass_summary.get("source"),
            "core_proxy_bypass_returncode": core_proxy_bypass_summary.get("returncode"),
            "core_proxy_bypass_duration_sec": core_proxy_bypass_summary.get("duration_sec"),
            "core_proxy_bypass_requests_total": core_proxy_bypass_summary.get("requests_total"),
            "core_proxy_bypass_bypass_attempted": core_proxy_bypass_summary.get("bypass_attempted"),
            "core_proxy_bypass_bypass_success": core_proxy_bypass_summary.get("bypass_success"),
            "core_proxy_bypass_bypass_failed": core_proxy_bypass_summary.get("bypass_failed"),
            "core_proxy_bypass_no_proxy_retries": core_proxy_bypass_summary.get("no_proxy_retries"),
            "core_proxy_bypass_no_proxy_retry_exhausted": core_proxy_bypass_summary.get(
                "no_proxy_retry_exhausted"
            ),
            "core_proxy_bypass_hint_response": core_proxy_bypass_summary.get("hint_response"),
            "core_proxy_bypass_hint_exception": core_proxy_bypass_summary.get("hint_exception"),
            "core_proxy_bypass_reason_response_hint": core_proxy_bypass_summary.get(
                "reason_response_hint"
            ),
            "core_proxy_bypass_reason_exception_hint": core_proxy_bypass_summary.get(
                "reason_exception_hint"
            ),
            "core_proxy_bypass_last_reason": core_proxy_bypass_summary.get("last_reason"),
            "core_proxy_bypass_local_requests_total": core_proxy_bypass_summary.get(
                "local_requests_total"
            ),
            "core_proxy_bypass_local_bypass_attempted": core_proxy_bypass_summary.get(
                "local_bypass_attempted"
            ),
            "core_proxy_bypass_local_bypass_success": core_proxy_bypass_summary.get(
                "local_bypass_success"
            ),
            "core_proxy_bypass_local_bypass_failed": core_proxy_bypass_summary.get(
                "local_bypass_failed"
            ),
            "core_proxy_bypass_local_no_proxy_retries": core_proxy_bypass_summary.get(
                "local_no_proxy_retries"
            ),
            "core_proxy_bypass_local_no_proxy_retry_exhausted": core_proxy_bypass_summary.get(
                "local_no_proxy_retry_exhausted"
            ),
            "core_proxy_bypass_exec_requests_total": core_proxy_bypass_summary.get("exec_requests_total"),
            "core_proxy_bypass_exec_bypass_attempted": core_proxy_bypass_summary.get(
                "exec_bypass_attempted"
            ),
            "core_proxy_bypass_exec_bypass_success": core_proxy_bypass_summary.get(
                "exec_bypass_success"
            ),
            "core_proxy_bypass_exec_bypass_failed": core_proxy_bypass_summary.get("exec_bypass_failed"),
            "core_proxy_bypass_exec_no_proxy_retries": core_proxy_bypass_summary.get(
                "exec_no_proxy_retries"
            ),
            "core_proxy_bypass_exec_no_proxy_retry_exhausted": core_proxy_bypass_summary.get(
                "exec_no_proxy_retry_exhausted"
            ),
            "core_execution_present": core_execution_summary.get("present"),
            "core_execution_status": core_execution_summary.get("status"),
            "core_execution_reason": core_execution_summary.get("reason"),
            "core_execution_source": core_execution_summary.get("source"),
            "core_execution_returncode": core_execution_summary.get("returncode"),
            "core_execution_duration_sec": core_execution_summary.get("duration_sec"),
            "core_execution_action": core_execution_summary.get("action"),
            "core_execution_decision": core_execution_summary.get("decision"),
            "core_execution_executor_attempted": core_execution_summary.get("executor_attempted"),
            "core_execution_executor_probe_requested": core_execution_summary.get(
                "executor_probe_requested"
            ),
            "core_execution_executor_probe_effective": core_execution_summary.get(
                "executor_probe_effective"
            ),
            "core_execution_executor_force_probe_on_guardrail": core_execution_summary.get(
                "executor_force_probe_on_guardrail"
            ),
            "core_execution_executor_live_requested": core_execution_summary.get(
                "executor_live_requested"
            ),
            "core_execution_executor_live_effective": core_execution_summary.get(
                "executor_live_effective"
            ),
            "core_execution_executor_cmd": core_execution_summary.get("executor_cmd"),
            "core_execution_order_http": core_execution_summary.get("order_http"),
            "core_execution_order_endpoint": core_execution_summary.get("order_endpoint"),
            "core_execution_order_decision": core_execution_summary.get("order_decision"),
            "core_execution_order_mode": core_execution_summary.get("order_mode"),
            "core_execution_order_reason": core_execution_summary.get("order_reason"),
            "core_execution_order_error": core_execution_summary.get("order_error"),
            "core_execution_order_cap_violation": core_execution_summary.get("order_cap_violation"),
            "core_execution_order_simulated": core_execution_summary.get("order_simulated"),
            "core_execution_guardrail_hit": core_execution_summary.get("guardrail_hit"),
            "core_execution_guardrail_reasons": core_execution_summary.get("guardrail_reasons"),
            "core_execution_paper_fill_gate_mode": core_execution_summary.get(
                "paper_fill_gate_mode"
            ),
            "core_execution_paper_fill_gate_policy": core_execution_summary.get(
                "paper_fill_gate_policy"
            ),
            "core_execution_paper_fill_gate_reason": core_execution_summary.get(
                "paper_fill_gate_reason"
            ),
            "core_execution_paper_fill_gate_cap_violation": core_execution_summary.get(
                "paper_fill_gate_cap_violation"
            ),
            "core_execution_paper_execution_attempted": core_execution_summary.get(
                "paper_execution_attempted"
            ),
            "core_execution_paper_execution_applied": core_execution_summary.get(
                "paper_execution_applied"
            ),
            "core_execution_paper_execution_route": core_execution_summary.get(
                "paper_execution_route"
            ),
            "core_execution_paper_execution_fill_px": core_execution_summary.get(
                "paper_execution_fill_px"
            ),
            "core_execution_paper_execution_signed_slippage_bps": core_execution_summary.get(
                "paper_execution_signed_slippage_bps"
            ),
            "core_execution_paper_execution_fee_rate": core_execution_summary.get(
                "paper_execution_fee_rate"
            ),
            "core_execution_paper_execution_fee_usdt": core_execution_summary.get(
                "paper_execution_fee_usdt"
            ),
            "core_execution_paper_execution_ledger_written": core_execution_summary.get(
                "paper_execution_ledger_written"
            ),
            "core_paper_mode_readiness_present": core_paper_mode_summary.get("present"),
            "core_paper_mode_readiness_status": core_paper_mode_summary.get("status"),
            "core_paper_mode_readiness_reason": core_paper_mode_summary.get("reason"),
            "core_paper_mode_readiness_source": core_paper_mode_summary.get("source"),
            "core_paper_mode_readiness_returncode": core_paper_mode_summary.get("returncode"),
            "core_paper_mode_readiness_duration_sec": core_paper_mode_summary.get("duration_sec"),
            "core_paper_mode_readiness_enforce": core_paper_mode_summary.get("enforce"),
            "core_paper_mode_readiness_fail_closed": core_paper_mode_summary.get("fail_closed"),
            "core_paper_mode_gate_blocked": core_paper_mode_summary.get("gate_blocked"),
            "core_paper_mode_gate_reason": core_paper_mode_summary.get("gate_reason"),
            "core_paper_mode_readiness_report_status": core_paper_mode_summary.get("readiness_status"),
            "core_paper_mode_ready_for_paper": core_paper_mode_summary.get("ready_for_paper_mode"),
            "core_paper_mode_readiness_report_path": core_paper_mode_summary.get("report_path"),
            "core_paper_mode_readiness_report_ts": core_paper_mode_summary.get("report_ts"),
            "core_paper_mode_readiness_report_age_hours": core_paper_mode_summary.get("report_age_hours"),
            "core_paper_mode_readiness_report_stale": core_paper_mode_summary.get("report_stale"),
            "core_paper_mode_readiness_coverage": core_paper_mode_summary.get("coverage"),
            "core_paper_mode_readiness_missing_buckets": core_paper_mode_summary.get("missing_buckets"),
            "core_paper_mode_readiness_largest_missing_block_hours": core_paper_mode_summary.get(
                "largest_missing_block_hours"
            ),
            "core_paper_mode_readiness_fail_reasons": core_paper_mode_summary.get("fail_reasons"),
            "core_paper_mode_readiness_refresh_attempted": core_paper_mode_summary.get("refresh_attempted"),
            "core_paper_mode_readiness_refresh_ok": core_paper_mode_summary.get("refresh_ok"),
            "core_paper_mode_readiness_refresh_reason": core_paper_mode_summary.get("refresh_reason"),
            "core_paper_mode_readiness_refresh_trigger": core_paper_mode_summary.get("refresh_trigger"),
            "core_paper_mode_readiness_refresh_returncode": core_paper_mode_summary.get("refresh_returncode"),
            "core_paper_mode_readiness_refresh_duration_sec": core_paper_mode_summary.get("refresh_duration_sec"),
            "core_paper_mode_readiness_refresh_throttled": core_paper_mode_summary.get("refresh_throttled"),
            "core_paper_mode_readiness_refresh_throttle_remaining_sec": core_paper_mode_summary.get(
                "refresh_throttle_remaining_sec"
            ),
            "core_paper_mode_readiness_refresh_lock_path": core_paper_mode_summary.get("refresh_lock_path"),
            "core_paper_mode_readiness_refresh_lock_acquired": core_paper_mode_summary.get(
                "refresh_lock_acquired"
            ),
            "core_paper_mode_readiness_refresh_state_path": core_paper_mode_summary.get("refresh_state_path"),
            "core_paper_mode_readiness_refresh_last_attempt_ts": core_paper_mode_summary.get(
                "refresh_last_attempt_ts"
            ),
            "core_paper_mode_readiness_refresh_last_success_ts": core_paper_mode_summary.get(
                "refresh_last_success_ts"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_attempted": core_paper_mode_summary.get(
                "refresh_lock_contention_attempted"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_ok": core_paper_mode_summary.get(
                "refresh_lock_contention_ok"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_reason": core_paper_mode_summary.get(
                "refresh_lock_contention_reason"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_returncode": core_paper_mode_summary.get(
                "refresh_lock_contention_returncode"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_duration_sec": core_paper_mode_summary.get(
                "refresh_lock_contention_duration_sec"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_json_path": core_paper_mode_summary.get(
                "refresh_lock_contention_json_path"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_md_path": core_paper_mode_summary.get(
                "refresh_lock_contention_md_path"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_csv_path": core_paper_mode_summary.get(
                "refresh_lock_contention_csv_path"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_source_coverage_csv": core_paper_mode_summary.get(
                "refresh_lock_contention_source_coverage_csv"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_gate_level": core_paper_mode_summary.get(
                "refresh_lock_contention_gate_level"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_gate_fail_triggered": core_paper_mode_summary.get(
                "refresh_lock_contention_gate_fail_triggered"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_gate_source_bucket_view": core_paper_mode_summary.get(
                "refresh_lock_contention_gate_source_bucket_view"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_gate_source_bucket_requested": core_paper_mode_summary.get(
                "refresh_lock_contention_gate_source_bucket_requested"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_gate_source_bucket_effective": core_paper_mode_summary.get(
                "refresh_lock_contention_gate_source_bucket_effective"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_gate_sample_guard_level": core_paper_mode_summary.get(
                "refresh_lock_contention_gate_sample_guard_level"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_gate_sample_guard_fail_triggered": core_paper_mode_summary.get(
                "refresh_lock_contention_gate_sample_guard_fail_triggered"
            ),
            "core_paper_mode_readiness_refresh_lock_contention_gate_sample_guard_fail_on_level": core_paper_mode_summary.get(
                "refresh_lock_contention_gate_sample_guard_fail_on_level"
            ),
            "core_paper_mode_lock_contention_enforce": core_paper_mode_summary.get("lock_contention_enforce"),
            "core_paper_mode_lock_contention_fail_closed": core_paper_mode_summary.get(
                "lock_contention_fail_closed"
            ),
            "core_paper_mode_lock_contention_report_path": core_paper_mode_summary.get(
                "lock_contention_report_path"
            ),
            "core_paper_mode_lock_contention_report_present": core_paper_mode_summary.get(
                "lock_contention_report_present"
            ),
            "core_paper_mode_lock_contention_overall_level": core_paper_mode_summary.get(
                "lock_contention_overall_level"
            ),
            "core_paper_mode_lock_contention_source_bucket_view": core_paper_mode_summary.get(
                "lock_contention_source_bucket_view"
            ),
            "core_paper_mode_lock_contention_source_bucket_requested": core_paper_mode_summary.get(
                "lock_contention_source_bucket_requested"
            ),
            "core_paper_mode_lock_contention_source_bucket_effective": core_paper_mode_summary.get(
                "lock_contention_source_bucket_effective"
            ),
            "core_paper_mode_lock_contention_fail_triggered": core_paper_mode_summary.get(
                "lock_contention_fail_triggered"
            ),
            "core_paper_mode_lock_contention_generated_at": core_paper_mode_summary.get(
                "lock_contention_generated_at"
            ),
            "core_paper_mode_lock_contention_report_age_hours": core_paper_mode_summary.get(
                "lock_contention_report_age_hours"
            ),
            "core_paper_mode_lock_contention_report_stale": core_paper_mode_summary.get(
                "lock_contention_report_stale"
            ),
            "core_paper_mode_lock_contention_max_age_hours": core_paper_mode_summary.get(
                "lock_contention_max_age_hours"
            ),
            "core_paper_mode_lock_contention_max_allowed_level": core_paper_mode_summary.get(
                "lock_contention_max_allowed_level"
            ),
            "core_paper_mode_lock_contention_required_source_bucket": core_paper_mode_summary.get(
                "lock_contention_required_source_bucket"
            ),
            "core_paper_mode_lock_contention_require_bucket_match": core_paper_mode_summary.get(
                "lock_contention_require_bucket_match"
            ),
            "core_paper_mode_lock_contention_require_sample_guard": core_paper_mode_summary.get(
                "lock_contention_require_sample_guard"
            ),
            "core_paper_mode_lock_contention_sample_guard_level": core_paper_mode_summary.get(
                "lock_contention_sample_guard_level"
            ),
            "core_paper_mode_lock_contention_sample_guard_fail_triggered": core_paper_mode_summary.get(
                "lock_contention_sample_guard_fail_triggered"
            ),
            "core_paper_mode_lock_contention_sample_guard_fail_on_level": core_paper_mode_summary.get(
                "lock_contention_sample_guard_fail_on_level"
            ),
            "core_paper_mode_lock_contention_blocked": core_paper_mode_summary.get("lock_contention_blocked"),
            "core_paper_mode_lock_contention_reason": core_paper_mode_summary.get("lock_contention_reason"),
            "core_paper_mode_core_reason": core_paper_mode_summary.get("core_reason"),
            "core_paper_mode_pulse_lock_path": core_paper_mode_summary.get("pulse_lock_path"),
            "core_paper_mode_pulse_lock_acquired": core_paper_mode_summary.get("pulse_lock_acquired"),
            "core_paper_mode_pulse_lock_reason": core_paper_mode_summary.get("pulse_lock_reason"),
            "core_paper_artifacts_sync_present": core_paper_artifacts_summary.get("present"),
            "core_paper_artifacts_sync_status": core_paper_artifacts_summary.get("status"),
            "core_paper_artifacts_sync_reason": core_paper_artifacts_summary.get("reason"),
            "core_paper_artifacts_sync_source": core_paper_artifacts_summary.get("source"),
            "core_paper_artifacts_sync_returncode": core_paper_artifacts_summary.get("returncode"),
            "core_paper_artifacts_sync_duration_sec": core_paper_artifacts_summary.get("duration_sec"),
            "core_paper_artifacts_sync_attempted": core_paper_artifacts_summary.get("attempted"),
            "core_paper_artifacts_sync_ok": core_paper_artifacts_summary.get("ok"),
            "core_paper_artifacts_sync_sync_reason": core_paper_artifacts_summary.get("sync_reason"),
            "core_paper_artifacts_sync_event_source": core_paper_artifacts_summary.get("event_source"),
            "core_paper_artifacts_sync_as_of": core_paper_artifacts_summary.get("as_of"),
            "core_paper_artifacts_sync_paper_positions_path": core_paper_artifacts_summary.get(
                "paper_positions_path"
            ),
            "core_paper_artifacts_sync_broker_snapshot_path": core_paper_artifacts_summary.get(
                "broker_snapshot_path"
            ),
            "core_paper_artifacts_sync_paper_positions_written": core_paper_artifacts_summary.get(
                "paper_positions_written"
            ),
            "core_paper_artifacts_sync_broker_snapshot_written": core_paper_artifacts_summary.get(
                "broker_snapshot_written"
            ),
            "core_paper_artifacts_sync_position_rows": core_paper_artifacts_summary.get("position_rows"),
            "core_paper_artifacts_sync_lock_path": core_paper_artifacts_summary.get("lock_path"),
            "core_paper_artifacts_sync_lock_acquired": core_paper_artifacts_summary.get("lock_acquired"),
            "core_paper_artifacts_sync_lock_wait_sec": core_paper_artifacts_summary.get("lock_wait_sec"),
            "core_paper_artifacts_sync_lock_timeout_sec": core_paper_artifacts_summary.get("lock_timeout_sec"),
            "core_paper_artifacts_sync_lock_retry_sec": core_paper_artifacts_summary.get("lock_retry_sec"),
            "core_paper_artifacts_sync_lock_reason": core_paper_artifacts_summary.get("lock_reason"),
            "core_paper_artifacts_sync_stale_guard_blocked": core_paper_artifacts_summary.get(
                "stale_guard_blocked"
            ),
            "core_paper_artifacts_sync_existing_as_of": core_paper_artifacts_summary.get("existing_as_of"),
            "core_paper_artifacts_sync_target_as_of": core_paper_artifacts_summary.get("target_as_of"),
        }
        event.update(_derive_ops_next_action_fields(event))
        append_jsonl(LOG_JSONL, event)
        append_state_event(STATE_MD, event)
        print(json.dumps(event, ensure_ascii=False))
        return 0

    plan = build_plan(args, effective_guard_mode=effective_guard_mode, guard_mode=guard_mode)
    steps: List[Dict[str, Any]] = []
    status = "ok"
    fail_steps: List[str] = []
    fail_steps_critical: List[str] = []
    fail_steps_optional: List[str] = []
    optional_step_names = _parse_csv_set(
        os.getenv(
            "PI_CYCLE_OPTIONAL_STEPS",
            "hippocampus,digital_life_core,cron_health,cron_policy_patch_draft,cron_policy_apply_batch,envelope_lint",
        ),
        default=[
            "hippocampus",
            "digital_life_core",
            "cron_health",
            "cron_policy_patch_draft",
            "cron_policy_apply_batch",
            "envelope_lint",
        ],
    )
    critical_step_max_retries = _int_env("PI_CYCLE_CRITICAL_STEP_MAX_RETRIES", 1, minimum=0)
    optional_step_max_retries = _int_env("PI_CYCLE_OPTIONAL_STEP_MAX_RETRIES", 0, minimum=0)
    retry_backoff_sec = max(
        0.0,
        _as_float(os.getenv("PI_CYCLE_STEP_RETRY_BACKOFF_SEC", "1.5")) or 1.5,
    )

    try:
        for step in plan:
            step_name = str(step.get("name") or "")
            is_optional_step = step_name in optional_step_names
            max_retries = optional_step_max_retries if is_optional_step else critical_step_max_retries
            if args.dry_run:
                result = {
                    "name": step_name,
                    "cmd": step["cmd"],
                    "returncode": 0,
                    "duration_sec": 0.0,
                    "json_tail": {"dry_run": True},
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "attempt_count": 1,
                    "max_retries": max_retries,
                    "retried": False,
                    "attempts": [{"attempt": 1, "returncode": 0, "duration_sec": 0.0}],
                    "optional_step": is_optional_step,
                }
            else:
                attempts: List[Dict[str, Any]] = []
                result: Dict[str, Any] = {}
                for attempt_idx in range(0, max_retries + 1):
                    try:
                        result = run_step(step_name, step["cmd"], timeout=int(step["timeout"]))
                    except subprocess.TimeoutExpired as e:
                        result = {
                            "name": step_name,
                            "cmd": step["cmd"],
                            "returncode": 124,
                            "duration_sec": float(step["timeout"]),
                            "json_tail": None,
                            "stdout_tail": (e.stdout or "")[-2000:],
                            "stderr_tail": f"timeout:{step['timeout']}s",
                        }
                    except Exception as e:
                        result = {
                            "name": step_name,
                            "cmd": step["cmd"],
                            "returncode": 1,
                            "duration_sec": 0.0,
                            "json_tail": None,
                            "stdout_tail": "",
                            "stderr_tail": f"runner_error:{type(e).__name__}:{str(e)[:300]}",
                        }
                    attempts.append(
                        {
                            "attempt": attempt_idx + 1,
                            "returncode": int(result.get("returncode", 1)),
                            "duration_sec": result.get("duration_sec"),
                            "stderr_tail": str(result.get("stderr_tail") or "")[-300:],
                        }
                    )
                    if int(result.get("returncode", 1)) == 0:
                        break
                    if attempt_idx < max_retries and retry_backoff_sec > 0:
                        time.sleep(retry_backoff_sec * float(attempt_idx + 1))

                result["attempt_count"] = len(attempts)
                result["max_retries"] = max_retries
                result["retried"] = len(attempts) > 1
                result["attempts"] = attempts
                result["optional_step"] = is_optional_step

            steps.append(result)
            if int(result.get("returncode", 1)) != 0:
                fail_steps.append(step_name)
                if is_optional_step:
                    fail_steps_optional.append(step_name)
                    result["fallback_applied"] = True
                    result["fallback_reason"] = "optional_step_failed_continue"
                else:
                    fail_steps_critical.append(step_name)
                    result["fallback_applied"] = False
                    result["fallback_reason"] = "critical_step_failed"
    finally:
        release_lock(LOCK_PATH)

    if fail_steps_critical:
        status = "failed"
    elif fail_steps_optional and status == "ok":
        status = "degraded"
    apply_batch_summary = _extract_cron_policy_apply_batch_summary(steps)
    digital_life_summary = _extract_digital_life_summary(steps)
    core_proxy_bypass_summary = _extract_core_proxy_bypass_summary(steps)
    core_execution_summary = _extract_core_execution_summary(steps)
    core_paper_mode_summary = _extract_core_paper_mode_readiness_summary(steps)
    core_paper_artifacts_summary = _extract_core_paper_artifacts_sync_summary(steps)

    gate_rollout = compute_gate_rollout_snapshot()
    gate_rollout_alert = evaluate_gate_rollout_alert(gate_rollout)
    gate_notify_trend = compute_gate_notify_trend()
    gate_component_mode_drift = resolve_component_mode_drift_signal(
        log_path=Path(os.getenv("PI_CYCLE_LOG_JSONL", str(LOG_JSONL))),
        current_configured_mode=str(gate_rollout.get("rollout_mode_configured") or ""),
        current_effective_mode=str(gate_rollout.get("rollout_mode_effective") or ""),
    )
    gate_rollout_rollback = maybe_apply_rollout_rollback(
        gate_rollout_alert,
        gate_rollout,
        component_mode_drift=gate_component_mode_drift,
    )
    gate_rollout_proxy_control = maybe_apply_proxy_rollout_control(
        guard_mode=guard_mode if isinstance(guard_mode, dict) else {},
        rollback_applied=bool(gate_rollout_rollback.get("applied")),
    )
    gate_rollout_digital_life_control = maybe_apply_digital_life_rollout_control(
        guard_mode=guard_mode if isinstance(guard_mode, dict) else {},
        digital_life_summary=digital_life_summary if isinstance(digital_life_summary, dict) else {},
        rollback_applied=bool(gate_rollout_rollback.get("applied")),
        proxy_applied=bool(gate_rollout_proxy_control.get("applied")),
    )

    if status == "ok" and str(gate_rollout_alert.get("status") or "") in {"degraded", "critical"}:
        status = "degraded"
    if status == "ok" and str(gate_notify_trend.get("status") or "") in {"degraded", "critical"}:
        status = "degraded"
    if status == "ok" and bool(gate_rollout_rollback.get("applied")):
        status = "degraded"
    if status == "ok" and bool(gate_rollout_proxy_control.get("applied")):
        status = "degraded"
    if status == "ok" and bool(gate_rollout_digital_life_control.get("applied")):
        status = "degraded"
    if status == "ok" and str(core_proxy_bypass_summary.get("status") or "") in {"degraded", "critical"}:
        status = "degraded"
    if status == "ok" and str(core_execution_summary.get("status") or "") in {"degraded", "critical"}:
        status = "degraded"
    if status == "ok" and str(core_paper_mode_summary.get("status") or "") in {"degraded", "critical"}:
        status = "degraded"
    if status == "ok" and str(core_paper_artifacts_summary.get("status") or "") in {"degraded", "critical"}:
        status = "degraded"

    event = {
        "envelope_version": "1.0",
        "domain": "pi_cycle",
        "ts": now_iso(),
        "status": status,
        "duration_sec": round(time.time() - started, 3),
        "guard_mode_requested": guard_mode.get("requested"),
        "guard_mode_effective": guard_mode.get("effective"),
        "guard_mode_base_effective": guard_mode.get("base_effective"),
        "guard_mode_recommended": guard_mode.get("recommended"),
        "guard_mode_recommend_reason": guard_mode.get("recommend_reason"),
        "auto_full_hhmm": guard_mode.get("auto_full_hhmm"),
        "guard_mode_auto_override_applied": guard_mode.get("auto_override_applied"),
            "guard_mode_use_rollback_action": guard_mode.get("use_rollback_action"),
            "guard_mode_apply_rollback_action": guard_mode.get("apply_rollback_action"),
            "guard_mode_use_patch_draft_action": guard_mode.get("use_patch_draft_action"),
            "guard_mode_apply_patch_draft_action": guard_mode.get("apply_patch_draft_action"),
            "guard_mode_use_patch_apply_action": guard_mode.get("use_patch_apply_action"),
            "guard_mode_apply_patch_apply_action": guard_mode.get("apply_patch_apply_action"),
            "guard_mode_use_proxy_control_action": guard_mode.get("use_proxy_control_action"),
            "guard_mode_apply_proxy_control_action": guard_mode.get("apply_proxy_control_action"),
            "guard_mode_use_digital_life_action": guard_mode.get("use_digital_life_action"),
            "guard_mode_apply_digital_life_action": guard_mode.get("apply_digital_life_action"),
            "guard_mode_patch_pending_threshold": guard_mode.get("patch_pending_threshold"),
            "guard_mode_proxy_control_trigger_threshold": guard_mode.get("proxy_control_trigger_threshold"),
            "guard_mode_patch_apply_target_batch": guard_mode.get("patch_apply_target_batch"),
        "guard_mode_rollback_status": guard_mode.get("rollback_status"),
        "guard_mode_rollback_reason": guard_mode.get("rollback_reason"),
        "guard_mode_rollback_action_level": guard_mode.get("rollback_action_level"),
        "guard_mode_rollback_action_reason": guard_mode.get("rollback_action_reason"),
        "guard_mode_rollback_source_path": guard_mode.get("rollback_source_path"),
        "guard_mode_rollback_report_ts": guard_mode.get("rollback_report_ts"),
        "guard_mode_proxy_control_trend_status": guard_mode.get("proxy_control_trend_status"),
        "guard_mode_proxy_control_trend_reason": guard_mode.get("proxy_control_trend_reason"),
        "guard_mode_proxy_control_trend_action_level": guard_mode.get("proxy_control_trend_action_level"),
        "guard_mode_proxy_control_trend_action_reason": guard_mode.get("proxy_control_trend_action_reason"),
        "guard_mode_proxy_control_trend_trigger_count": guard_mode.get("proxy_control_trend_trigger_count"),
        "guard_mode_proxy_control_trend_trigger_rate": guard_mode.get("proxy_control_trend_trigger_rate"),
        "guard_mode_proxy_control_trend_dominant_trigger": guard_mode.get(
            "proxy_control_trend_dominant_trigger"
        ),
        "guard_mode_proxy_control_trend_critical_trigger_count": guard_mode.get(
            "proxy_control_trend_critical_trigger_count"
        ),
        "guard_mode_proxy_error_status": guard_mode.get("proxy_error_status"),
        "guard_mode_proxy_error_reason": guard_mode.get("proxy_error_reason"),
        "guard_mode_proxy_error_action_level": guard_mode.get("proxy_error_action_level"),
        "guard_mode_proxy_error_action_reason": guard_mode.get("proxy_error_action_reason"),
        "guard_mode_proxy_error_jobs_count": guard_mode.get("proxy_error_jobs_count"),
        "guard_mode_proxy_error_core_jobs_count": guard_mode.get("proxy_error_core_jobs_count"),
        "guard_mode_proxy_error_max_consecutive_errors": guard_mode.get(
            "proxy_error_max_consecutive_errors"
        ),
        "guard_mode_proxy_error_trend_status": guard_mode.get("proxy_error_trend_status"),
        "guard_mode_proxy_error_trend_reason": guard_mode.get("proxy_error_trend_reason"),
        "guard_mode_proxy_error_trend_action_level": guard_mode.get("proxy_error_trend_action_level"),
        "guard_mode_proxy_error_trend_action_reason": guard_mode.get("proxy_error_trend_action_reason"),
        "guard_mode_proxy_error_trend_jobs_with_errors": guard_mode.get(
            "proxy_error_trend_jobs_with_errors"
        ),
        "guard_mode_proxy_error_trend_core_jobs_with_errors": guard_mode.get(
            "proxy_error_trend_core_jobs_with_errors"
        ),
        "guard_mode_proxy_error_trend_max_consecutive_errors": guard_mode.get(
            "proxy_error_trend_max_consecutive_errors"
        ),
        "guard_mode_proxy_error_trend_rate": guard_mode.get("proxy_error_trend_rate"),
        "guard_mode_patch_draft_status": guard_mode.get("patch_draft_status"),
        "guard_mode_patch_draft_reason": guard_mode.get("patch_draft_reason"),
        "guard_mode_patch_draft_actionable_jobs": guard_mode.get("patch_draft_actionable_jobs"),
        "guard_mode_patch_draft_changed_jobs": guard_mode.get("patch_draft_changed_jobs"),
        "guard_mode_patch_draft_failed_jobs": guard_mode.get("patch_draft_failed_jobs"),
        "guard_mode_patch_draft_rollout_strategy": guard_mode.get("patch_draft_rollout_strategy"),
        "guard_mode_patch_draft_rollout_non_empty_batches": guard_mode.get(
            "patch_draft_rollout_non_empty_batches"
        ),
        "guard_mode_patch_draft_rollout_apply_order_count": guard_mode.get(
            "patch_draft_rollout_apply_order_count"
        ),
        "guard_mode_patch_draft_rollout_next_batch": guard_mode.get("patch_draft_rollout_next_batch"),
        "guard_mode_patch_draft_pending": guard_mode.get("patch_draft_pending"),
        "guard_mode_patch_draft_action_level": guard_mode.get("patch_draft_action_level"),
        "guard_mode_patch_draft_action_reason": guard_mode.get("patch_draft_action_reason"),
            "guard_mode_patch_draft_action_hints": guard_mode.get("patch_draft_action_hints"),
            "guard_mode_patch_draft_next_action": guard_mode.get("patch_draft_next_action"),
            "guard_mode_patch_draft_next_action_secondary": guard_mode.get("patch_draft_next_action_secondary"),
            "guard_mode_patch_apply_present": guard_mode.get("patch_apply_present"),
            "guard_mode_patch_apply_status": guard_mode.get("patch_apply_status"),
            "guard_mode_patch_apply_reason": guard_mode.get("patch_apply_reason"),
            "guard_mode_patch_apply_batch_id": guard_mode.get("patch_apply_batch_id"),
            "guard_mode_patch_apply_mode": guard_mode.get("patch_apply_mode"),
            "guard_mode_patch_apply_applied_to_source": guard_mode.get("patch_apply_applied_to_source"),
            "guard_mode_patch_apply_apply_skipped_reason": guard_mode.get("patch_apply_apply_skipped_reason"),
            "guard_mode_patch_apply_selected_actions": guard_mode.get("patch_apply_selected_actions"),
            "guard_mode_patch_apply_changed_jobs": guard_mode.get("patch_apply_changed_jobs"),
            "guard_mode_patch_apply_failed_jobs": guard_mode.get("patch_apply_failed_jobs"),
            "guard_mode_patch_apply_blocked_jobs": guard_mode.get("patch_apply_blocked_jobs"),
            "guard_mode_patch_apply_operations_total": guard_mode.get("patch_apply_operations_total"),
            "guard_mode_patch_apply_operations_applied": guard_mode.get("patch_apply_operations_applied"),
            "guard_mode_patch_apply_operations_changed": guard_mode.get("patch_apply_operations_changed"),
            "guard_mode_patch_apply_trend_status": guard_mode.get("patch_apply_trend_status"),
            "guard_mode_patch_apply_trend_reason": guard_mode.get("patch_apply_trend_reason"),
            "guard_mode_patch_apply_trend_events": guard_mode.get("patch_apply_trend_events"),
            "guard_mode_patch_apply_trend_problem_events": guard_mode.get(
                "patch_apply_trend_problem_events"
            ),
            "guard_mode_patch_apply_trend_problem_rate": guard_mode.get("patch_apply_trend_problem_rate"),
            "guard_mode_patch_apply_trend_critical_problem_events": guard_mode.get(
                "patch_apply_trend_critical_problem_events"
            ),
            "guard_mode_patch_apply_trend_critical_problem_rate": guard_mode.get(
                "patch_apply_trend_critical_problem_rate"
            ),
            "guard_mode_patch_apply_trend_target_batch_id": guard_mode.get("patch_apply_trend_target_batch_id"),
            "guard_mode_digital_life_status": guard_mode.get("digital_life_status"),
            "guard_mode_digital_life_reason": guard_mode.get("digital_life_reason"),
            "guard_mode_digital_life_lifecycle_mode": guard_mode.get("digital_life_lifecycle_mode"),
        "guard_mode_digital_life_next_action": guard_mode.get("digital_life_next_action"),
        "guard_mode_digital_life_viability_score": guard_mode.get("digital_life_viability_score"),
        "guard_mode_digital_life_stress_score": guard_mode.get("digital_life_stress_score"),
        "guard_mode_digital_life_source_path": guard_mode.get("digital_life_source_path"),
        "guard_mode_digital_life_report_ts": guard_mode.get("digital_life_report_ts"),
        "guard_mode_patch_draft_isolation_enabled": guard_mode.get("patch_draft_isolation_enabled"),
        "guard_mode_patch_draft_isolation_candidates_total": guard_mode.get(
            "patch_draft_isolation_candidates_total"
        ),
        "guard_mode_patch_draft_isolation_actionable_candidates": guard_mode.get(
            "patch_draft_isolation_actionable_candidates"
        ),
        "guard_mode_patch_draft_isolation_critical_candidates": guard_mode.get(
            "patch_draft_isolation_critical_candidates"
        ),
        "guard_mode_patch_draft_isolation_manual_only_candidates": guard_mode.get(
            "patch_draft_isolation_manual_only_candidates"
        ),
        "guard_mode_patch_draft_isolation_rollout_strategy": guard_mode.get(
            "patch_draft_isolation_rollout_strategy"
        ),
        "guard_mode_patch_draft_isolation_rollout_next_batch": guard_mode.get(
            "patch_draft_isolation_rollout_next_batch"
        ),
        "guard_mode_patch_draft_isolation_rollout_pending": guard_mode.get(
            "patch_draft_isolation_rollout_pending"
        ),
        "skip_core": bool(args.skip_core),
        "skip_watchdog": bool(args.skip_watchdog),
        "skip_gateway_guard": bool(args.skip_gateway_guard),
        "skip_hippocampus": bool(args.skip_hippocampus),
        "skip_digital_life": bool(args.skip_digital_life),
        "skip_cron_health": bool(args.skip_cron_health),
        "skip_cron_policy_draft": bool(args.skip_cron_policy_draft),
        "skip_cron_policy_apply": bool(args.skip_cron_policy_apply),
        "skip_envelope_lint": bool(args.skip_envelope_lint),
        "digital_life_present": digital_life_summary.get("present"),
        "digital_life_status": digital_life_summary.get("status"),
        "digital_life_reason": digital_life_summary.get("reason"),
        "digital_life_lifecycle_mode": digital_life_summary.get("lifecycle_mode"),
        "digital_life_next_action": digital_life_summary.get("next_action"),
        "digital_life_viability_score": digital_life_summary.get("viability_score"),
        "digital_life_stress_score": digital_life_summary.get("stress_score"),
        "digital_life_state_path": digital_life_summary.get("state_path"),
        "digital_life_events_path": digital_life_summary.get("events_path"),
        "digital_life_returncode": digital_life_summary.get("returncode"),
        "digital_life_duration_sec": digital_life_summary.get("duration_sec"),
        "cron_policy_apply_batch_present": apply_batch_summary.get("present"),
        "cron_policy_apply_batch_status": apply_batch_summary.get("status"),
        "cron_policy_apply_batch_reason": apply_batch_summary.get("reason"),
        "cron_policy_apply_batch_batch_id": apply_batch_summary.get("batch_id"),
        "cron_policy_apply_batch_mode": apply_batch_summary.get("mode"),
        "cron_policy_apply_batch_applied_to_source": apply_batch_summary.get("applied_to_source"),
        "cron_policy_apply_batch_apply_skipped_reason": apply_batch_summary.get("apply_skipped_reason"),
        "cron_policy_apply_batch_selected_actions": apply_batch_summary.get("selected_actions"),
        "cron_policy_apply_batch_changed_jobs": apply_batch_summary.get("changed_jobs"),
        "cron_policy_apply_batch_failed_jobs": apply_batch_summary.get("failed_jobs"),
        "cron_policy_apply_batch_blocked_jobs": apply_batch_summary.get("blocked_jobs"),
        "cron_policy_apply_batch_operations_total": apply_batch_summary.get("operations_total"),
        "cron_policy_apply_batch_operations_applied": apply_batch_summary.get("operations_applied"),
        "cron_policy_apply_batch_operations_changed": apply_batch_summary.get("operations_changed"),
        "cron_policy_apply_batch_output_path": apply_batch_summary.get("output_path"),
        "cron_policy_apply_batch_candidate_jobs_output": apply_batch_summary.get(
            "candidate_jobs_output"
        ),
        "core_proxy_bypass_present": core_proxy_bypass_summary.get("present"),
        "core_proxy_bypass_status": core_proxy_bypass_summary.get("status"),
        "core_proxy_bypass_reason": core_proxy_bypass_summary.get("reason"),
        "core_proxy_bypass_source": core_proxy_bypass_summary.get("source"),
        "core_proxy_bypass_returncode": core_proxy_bypass_summary.get("returncode"),
        "core_proxy_bypass_duration_sec": core_proxy_bypass_summary.get("duration_sec"),
        "core_proxy_bypass_requests_total": core_proxy_bypass_summary.get("requests_total"),
        "core_proxy_bypass_bypass_attempted": core_proxy_bypass_summary.get("bypass_attempted"),
        "core_proxy_bypass_bypass_success": core_proxy_bypass_summary.get("bypass_success"),
        "core_proxy_bypass_bypass_failed": core_proxy_bypass_summary.get("bypass_failed"),
        "core_proxy_bypass_no_proxy_retries": core_proxy_bypass_summary.get("no_proxy_retries"),
        "core_proxy_bypass_no_proxy_retry_exhausted": core_proxy_bypass_summary.get(
            "no_proxy_retry_exhausted"
        ),
        "core_proxy_bypass_hint_response": core_proxy_bypass_summary.get("hint_response"),
        "core_proxy_bypass_hint_exception": core_proxy_bypass_summary.get("hint_exception"),
        "core_proxy_bypass_reason_response_hint": core_proxy_bypass_summary.get(
            "reason_response_hint"
        ),
        "core_proxy_bypass_reason_exception_hint": core_proxy_bypass_summary.get(
            "reason_exception_hint"
        ),
        "core_proxy_bypass_last_reason": core_proxy_bypass_summary.get("last_reason"),
        "core_proxy_bypass_local_requests_total": core_proxy_bypass_summary.get(
            "local_requests_total"
        ),
        "core_proxy_bypass_local_bypass_attempted": core_proxy_bypass_summary.get(
            "local_bypass_attempted"
        ),
        "core_proxy_bypass_local_bypass_success": core_proxy_bypass_summary.get(
            "local_bypass_success"
        ),
        "core_proxy_bypass_local_bypass_failed": core_proxy_bypass_summary.get(
            "local_bypass_failed"
        ),
        "core_proxy_bypass_local_no_proxy_retries": core_proxy_bypass_summary.get(
            "local_no_proxy_retries"
        ),
        "core_proxy_bypass_local_no_proxy_retry_exhausted": core_proxy_bypass_summary.get(
            "local_no_proxy_retry_exhausted"
        ),
        "core_proxy_bypass_exec_requests_total": core_proxy_bypass_summary.get("exec_requests_total"),
        "core_proxy_bypass_exec_bypass_attempted": core_proxy_bypass_summary.get(
            "exec_bypass_attempted"
        ),
        "core_proxy_bypass_exec_bypass_success": core_proxy_bypass_summary.get(
            "exec_bypass_success"
        ),
        "core_proxy_bypass_exec_bypass_failed": core_proxy_bypass_summary.get("exec_bypass_failed"),
        "core_proxy_bypass_exec_no_proxy_retries": core_proxy_bypass_summary.get(
            "exec_no_proxy_retries"
        ),
        "core_proxy_bypass_exec_no_proxy_retry_exhausted": core_proxy_bypass_summary.get(
            "exec_no_proxy_retry_exhausted"
        ),
        "core_execution_present": core_execution_summary.get("present"),
        "core_execution_status": core_execution_summary.get("status"),
        "core_execution_reason": core_execution_summary.get("reason"),
        "core_execution_source": core_execution_summary.get("source"),
        "core_execution_returncode": core_execution_summary.get("returncode"),
        "core_execution_duration_sec": core_execution_summary.get("duration_sec"),
        "core_execution_action": core_execution_summary.get("action"),
        "core_execution_decision": core_execution_summary.get("decision"),
        "core_execution_executor_attempted": core_execution_summary.get("executor_attempted"),
        "core_execution_executor_probe_requested": core_execution_summary.get(
            "executor_probe_requested"
        ),
        "core_execution_executor_probe_effective": core_execution_summary.get(
            "executor_probe_effective"
        ),
        "core_execution_executor_force_probe_on_guardrail": core_execution_summary.get(
            "executor_force_probe_on_guardrail"
        ),
        "core_execution_executor_live_requested": core_execution_summary.get(
            "executor_live_requested"
        ),
        "core_execution_executor_live_effective": core_execution_summary.get(
            "executor_live_effective"
        ),
        "core_execution_executor_cmd": core_execution_summary.get("executor_cmd"),
        "core_execution_order_http": core_execution_summary.get("order_http"),
        "core_execution_order_endpoint": core_execution_summary.get("order_endpoint"),
        "core_execution_order_decision": core_execution_summary.get("order_decision"),
        "core_execution_order_mode": core_execution_summary.get("order_mode"),
        "core_execution_order_reason": core_execution_summary.get("order_reason"),
        "core_execution_order_error": core_execution_summary.get("order_error"),
        "core_execution_order_cap_violation": core_execution_summary.get("order_cap_violation"),
        "core_execution_order_simulated": core_execution_summary.get("order_simulated"),
        "core_execution_guardrail_hit": core_execution_summary.get("guardrail_hit"),
        "core_execution_guardrail_reasons": core_execution_summary.get("guardrail_reasons"),
        "core_execution_paper_fill_gate_mode": core_execution_summary.get(
            "paper_fill_gate_mode"
        ),
        "core_execution_paper_fill_gate_policy": core_execution_summary.get(
            "paper_fill_gate_policy"
        ),
        "core_execution_paper_fill_gate_reason": core_execution_summary.get(
            "paper_fill_gate_reason"
        ),
        "core_execution_paper_fill_gate_cap_violation": core_execution_summary.get(
            "paper_fill_gate_cap_violation"
        ),
        "core_execution_paper_execution_attempted": core_execution_summary.get(
            "paper_execution_attempted"
        ),
        "core_execution_paper_execution_applied": core_execution_summary.get(
            "paper_execution_applied"
        ),
        "core_execution_paper_execution_route": core_execution_summary.get(
            "paper_execution_route"
        ),
        "core_execution_paper_execution_fill_px": core_execution_summary.get(
            "paper_execution_fill_px"
        ),
        "core_execution_paper_execution_signed_slippage_bps": core_execution_summary.get(
            "paper_execution_signed_slippage_bps"
        ),
        "core_execution_paper_execution_fee_rate": core_execution_summary.get(
            "paper_execution_fee_rate"
        ),
        "core_execution_paper_execution_fee_usdt": core_execution_summary.get(
            "paper_execution_fee_usdt"
        ),
        "core_execution_paper_execution_ledger_written": core_execution_summary.get(
            "paper_execution_ledger_written"
        ),
        "core_paper_mode_readiness_present": core_paper_mode_summary.get("present"),
        "core_paper_mode_readiness_status": core_paper_mode_summary.get("status"),
        "core_paper_mode_readiness_reason": core_paper_mode_summary.get("reason"),
        "core_paper_mode_readiness_source": core_paper_mode_summary.get("source"),
        "core_paper_mode_readiness_returncode": core_paper_mode_summary.get("returncode"),
        "core_paper_mode_readiness_duration_sec": core_paper_mode_summary.get("duration_sec"),
        "core_paper_mode_readiness_enforce": core_paper_mode_summary.get("enforce"),
        "core_paper_mode_readiness_fail_closed": core_paper_mode_summary.get("fail_closed"),
        "core_paper_mode_gate_blocked": core_paper_mode_summary.get("gate_blocked"),
        "core_paper_mode_gate_reason": core_paper_mode_summary.get("gate_reason"),
        "core_paper_mode_readiness_report_status": core_paper_mode_summary.get("readiness_status"),
        "core_paper_mode_ready_for_paper": core_paper_mode_summary.get("ready_for_paper_mode"),
        "core_paper_mode_readiness_report_path": core_paper_mode_summary.get("report_path"),
        "core_paper_mode_readiness_report_ts": core_paper_mode_summary.get("report_ts"),
        "core_paper_mode_readiness_report_age_hours": core_paper_mode_summary.get("report_age_hours"),
        "core_paper_mode_readiness_report_stale": core_paper_mode_summary.get("report_stale"),
        "core_paper_mode_readiness_coverage": core_paper_mode_summary.get("coverage"),
        "core_paper_mode_readiness_missing_buckets": core_paper_mode_summary.get("missing_buckets"),
        "core_paper_mode_readiness_largest_missing_block_hours": core_paper_mode_summary.get(
            "largest_missing_block_hours"
        ),
        "core_paper_mode_readiness_fail_reasons": core_paper_mode_summary.get("fail_reasons"),
        "core_paper_mode_readiness_refresh_attempted": core_paper_mode_summary.get("refresh_attempted"),
        "core_paper_mode_readiness_refresh_ok": core_paper_mode_summary.get("refresh_ok"),
        "core_paper_mode_readiness_refresh_reason": core_paper_mode_summary.get("refresh_reason"),
        "core_paper_mode_readiness_refresh_trigger": core_paper_mode_summary.get("refresh_trigger"),
        "core_paper_mode_readiness_refresh_returncode": core_paper_mode_summary.get("refresh_returncode"),
        "core_paper_mode_readiness_refresh_duration_sec": core_paper_mode_summary.get("refresh_duration_sec"),
        "core_paper_mode_readiness_refresh_throttled": core_paper_mode_summary.get("refresh_throttled"),
        "core_paper_mode_readiness_refresh_throttle_remaining_sec": core_paper_mode_summary.get(
            "refresh_throttle_remaining_sec"
        ),
        "core_paper_mode_readiness_refresh_lock_path": core_paper_mode_summary.get("refresh_lock_path"),
        "core_paper_mode_readiness_refresh_lock_acquired": core_paper_mode_summary.get(
            "refresh_lock_acquired"
        ),
        "core_paper_mode_readiness_refresh_state_path": core_paper_mode_summary.get("refresh_state_path"),
        "core_paper_mode_readiness_refresh_last_attempt_ts": core_paper_mode_summary.get(
            "refresh_last_attempt_ts"
        ),
        "core_paper_mode_readiness_refresh_last_success_ts": core_paper_mode_summary.get(
            "refresh_last_success_ts"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_attempted": core_paper_mode_summary.get(
            "refresh_lock_contention_attempted"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_ok": core_paper_mode_summary.get(
            "refresh_lock_contention_ok"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_reason": core_paper_mode_summary.get(
            "refresh_lock_contention_reason"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_returncode": core_paper_mode_summary.get(
            "refresh_lock_contention_returncode"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_duration_sec": core_paper_mode_summary.get(
            "refresh_lock_contention_duration_sec"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_json_path": core_paper_mode_summary.get(
            "refresh_lock_contention_json_path"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_md_path": core_paper_mode_summary.get(
            "refresh_lock_contention_md_path"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_csv_path": core_paper_mode_summary.get(
            "refresh_lock_contention_csv_path"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_source_coverage_csv": core_paper_mode_summary.get(
            "refresh_lock_contention_source_coverage_csv"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_gate_level": core_paper_mode_summary.get(
            "refresh_lock_contention_gate_level"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_gate_fail_triggered": core_paper_mode_summary.get(
            "refresh_lock_contention_gate_fail_triggered"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_gate_source_bucket_view": core_paper_mode_summary.get(
            "refresh_lock_contention_gate_source_bucket_view"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_gate_source_bucket_requested": core_paper_mode_summary.get(
            "refresh_lock_contention_gate_source_bucket_requested"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_gate_source_bucket_effective": core_paper_mode_summary.get(
            "refresh_lock_contention_gate_source_bucket_effective"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_gate_sample_guard_level": core_paper_mode_summary.get(
            "refresh_lock_contention_gate_sample_guard_level"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_gate_sample_guard_fail_triggered": core_paper_mode_summary.get(
            "refresh_lock_contention_gate_sample_guard_fail_triggered"
        ),
        "core_paper_mode_readiness_refresh_lock_contention_gate_sample_guard_fail_on_level": core_paper_mode_summary.get(
            "refresh_lock_contention_gate_sample_guard_fail_on_level"
        ),
        "core_paper_mode_lock_contention_enforce": core_paper_mode_summary.get("lock_contention_enforce"),
        "core_paper_mode_lock_contention_fail_closed": core_paper_mode_summary.get(
            "lock_contention_fail_closed"
        ),
        "core_paper_mode_lock_contention_report_path": core_paper_mode_summary.get(
            "lock_contention_report_path"
        ),
        "core_paper_mode_lock_contention_report_present": core_paper_mode_summary.get(
            "lock_contention_report_present"
        ),
        "core_paper_mode_lock_contention_overall_level": core_paper_mode_summary.get(
            "lock_contention_overall_level"
        ),
        "core_paper_mode_lock_contention_source_bucket_view": core_paper_mode_summary.get(
            "lock_contention_source_bucket_view"
        ),
        "core_paper_mode_lock_contention_source_bucket_requested": core_paper_mode_summary.get(
            "lock_contention_source_bucket_requested"
        ),
        "core_paper_mode_lock_contention_source_bucket_effective": core_paper_mode_summary.get(
            "lock_contention_source_bucket_effective"
        ),
        "core_paper_mode_lock_contention_fail_triggered": core_paper_mode_summary.get(
            "lock_contention_fail_triggered"
        ),
        "core_paper_mode_lock_contention_generated_at": core_paper_mode_summary.get(
            "lock_contention_generated_at"
        ),
        "core_paper_mode_lock_contention_report_age_hours": core_paper_mode_summary.get(
            "lock_contention_report_age_hours"
        ),
        "core_paper_mode_lock_contention_report_stale": core_paper_mode_summary.get(
            "lock_contention_report_stale"
        ),
        "core_paper_mode_lock_contention_max_age_hours": core_paper_mode_summary.get(
            "lock_contention_max_age_hours"
        ),
        "core_paper_mode_lock_contention_max_allowed_level": core_paper_mode_summary.get(
            "lock_contention_max_allowed_level"
        ),
        "core_paper_mode_lock_contention_required_source_bucket": core_paper_mode_summary.get(
            "lock_contention_required_source_bucket"
        ),
        "core_paper_mode_lock_contention_require_bucket_match": core_paper_mode_summary.get(
            "lock_contention_require_bucket_match"
        ),
        "core_paper_mode_lock_contention_require_sample_guard": core_paper_mode_summary.get(
            "lock_contention_require_sample_guard"
        ),
        "core_paper_mode_lock_contention_sample_guard_level": core_paper_mode_summary.get(
            "lock_contention_sample_guard_level"
        ),
        "core_paper_mode_lock_contention_sample_guard_fail_triggered": core_paper_mode_summary.get(
            "lock_contention_sample_guard_fail_triggered"
        ),
        "core_paper_mode_lock_contention_sample_guard_fail_on_level": core_paper_mode_summary.get(
            "lock_contention_sample_guard_fail_on_level"
        ),
        "core_paper_mode_lock_contention_blocked": core_paper_mode_summary.get("lock_contention_blocked"),
        "core_paper_mode_lock_contention_reason": core_paper_mode_summary.get("lock_contention_reason"),
        "core_paper_mode_core_reason": core_paper_mode_summary.get("core_reason"),
        "core_paper_mode_pulse_lock_path": core_paper_mode_summary.get("pulse_lock_path"),
        "core_paper_mode_pulse_lock_acquired": core_paper_mode_summary.get("pulse_lock_acquired"),
        "core_paper_mode_pulse_lock_reason": core_paper_mode_summary.get("pulse_lock_reason"),
        "core_paper_artifacts_sync_present": core_paper_artifacts_summary.get("present"),
        "core_paper_artifacts_sync_status": core_paper_artifacts_summary.get("status"),
        "core_paper_artifacts_sync_reason": core_paper_artifacts_summary.get("reason"),
        "core_paper_artifacts_sync_source": core_paper_artifacts_summary.get("source"),
        "core_paper_artifacts_sync_returncode": core_paper_artifacts_summary.get("returncode"),
        "core_paper_artifacts_sync_duration_sec": core_paper_artifacts_summary.get("duration_sec"),
        "core_paper_artifacts_sync_attempted": core_paper_artifacts_summary.get("attempted"),
        "core_paper_artifacts_sync_ok": core_paper_artifacts_summary.get("ok"),
        "core_paper_artifacts_sync_sync_reason": core_paper_artifacts_summary.get("sync_reason"),
        "core_paper_artifacts_sync_event_source": core_paper_artifacts_summary.get("event_source"),
        "core_paper_artifacts_sync_as_of": core_paper_artifacts_summary.get("as_of"),
        "core_paper_artifacts_sync_paper_positions_path": core_paper_artifacts_summary.get(
            "paper_positions_path"
        ),
        "core_paper_artifacts_sync_broker_snapshot_path": core_paper_artifacts_summary.get(
            "broker_snapshot_path"
        ),
        "core_paper_artifacts_sync_paper_positions_written": core_paper_artifacts_summary.get(
            "paper_positions_written"
        ),
        "core_paper_artifacts_sync_broker_snapshot_written": core_paper_artifacts_summary.get(
            "broker_snapshot_written"
        ),
        "core_paper_artifacts_sync_position_rows": core_paper_artifacts_summary.get("position_rows"),
        "core_paper_artifacts_sync_lock_path": core_paper_artifacts_summary.get("lock_path"),
        "core_paper_artifacts_sync_lock_acquired": core_paper_artifacts_summary.get("lock_acquired"),
        "core_paper_artifacts_sync_lock_wait_sec": core_paper_artifacts_summary.get("lock_wait_sec"),
        "core_paper_artifacts_sync_lock_timeout_sec": core_paper_artifacts_summary.get("lock_timeout_sec"),
        "core_paper_artifacts_sync_lock_retry_sec": core_paper_artifacts_summary.get("lock_retry_sec"),
        "core_paper_artifacts_sync_lock_reason": core_paper_artifacts_summary.get("lock_reason"),
        "core_paper_artifacts_sync_stale_guard_blocked": core_paper_artifacts_summary.get(
            "stale_guard_blocked"
        ),
        "core_paper_artifacts_sync_existing_as_of": core_paper_artifacts_summary.get("existing_as_of"),
        "core_paper_artifacts_sync_target_as_of": core_paper_artifacts_summary.get("target_as_of"),
        "dry_run": bool(args.dry_run),
        "lock": lock,
        "plan": [p["name"] for p in plan],
        "fail_steps": fail_steps,
        "fail_steps_critical": fail_steps_critical,
        "fail_steps_optional": fail_steps_optional,
        "fallback_steps_optional": fail_steps_optional,
        "step_resilience": {
            "optional_steps": sorted(optional_step_names),
            "critical_step_max_retries": critical_step_max_retries,
            "optional_step_max_retries": optional_step_max_retries,
            "retry_backoff_sec": retry_backoff_sec,
        },
        "steps": steps,
        "gate_rollout": gate_rollout,
        "gate_rollout_alert": gate_rollout_alert,
        "gate_notify_trend": gate_notify_trend,
        "gate_component_mode_drift": gate_component_mode_drift,
        "gate_rollout_rollback": gate_rollout_rollback,
        "gate_rollout_proxy_control": gate_rollout_proxy_control,
        "gate_rollout_digital_life_control": gate_rollout_digital_life_control,
    }
    event.update(_derive_ops_next_action_fields(event))
    event["gate_rollout_alert_emit"] = maybe_emit_gate_rollout_alert(
        pi_cycle_event=event,
        gate_rollout=gate_rollout,
        gate_rollout_alert=gate_rollout_alert,
        dry_run=bool(args.dry_run),
    )
    append_jsonl(LOG_JSONL, event)
    append_state_event(STATE_MD, event)
    print(json.dumps(event, ensure_ascii=False))
    return 0 if status in {"ok", "degraded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
