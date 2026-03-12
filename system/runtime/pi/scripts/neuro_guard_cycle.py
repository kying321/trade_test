#!/usr/bin/env python3
"""Neuro guard validation cycle for OpenClaw Pi.

Usage:
  python3 scripts/neuro_guard_cycle.py --mode fast
  python3 scripts/neuro_guard_cycle.py --mode full
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

from lie_root_resolver import resolve_lie_system_root

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = resolve_lie_system_root()
REPORT_DIR = SYSTEM_ROOT / "output" / "review"
PI_CYCLE_LOG = Path(os.getenv("PI_CYCLE_LOG_JSONL", str(SYSTEM_ROOT / "output" / "logs" / "pi_cycle_events.jsonl")))


def _compact_tail(text: str, max_chars: int = 1200, max_lines: int = 40) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    lines = raw.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    compact = "\n".join(lines)
    if len(compact) > max_chars:
        compact = compact[-max_chars:]
    return compact


def run_cmd(cmd: List[str], timeout: int = 180) -> Dict[str, Any]:
    started = time.time()
    p = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    dur = round(time.time() - started, 3)
    stdout_tail = _compact_tail(p.stdout or "")
    stderr_tail = _compact_tail(p.stderr or "")
    # unittest成功时通常只需要stderr摘要，stdout里的测试打印会污染报告。
    if int(p.returncode) == 0 and any(str(part) == "unittest" for part in cmd):
        stdout_tail = ""
    return {
        "cmd": cmd,
        "returncode": int(p.returncode),
        "duration_sec": dur,
        "stdout": stdout_tail,
        "stderr": stderr_tail,
    }


def build_steps(mode: str) -> List[Dict[str, Any]]:
    compile_cmd = [
        "python3",
        "-m",
        "py_compile",
        "scripts/cortex_evaluator.py",
        "scripts/cortex_gate.py",
        "scripts/binance_spot_exec.py",
        "scripts/lie_spot_halfhour_core.py",
        "scripts/hippocampus.py",
        "scripts/lie_spine_watchdog.py",
        "scripts/pi_cycle_orchestrator.py",
        "scripts/cron_health_snapshot.py",
        "scripts/hourly_market_snapshot.py",
        "scripts/envelope_lint.py",
        "scripts/gate_notify_trend.py",
        "scripts/cron_policy_patch_draft.py",
        "scripts/cron_policy_apply_batch.py",
        "scripts/test_chaos.py",
        "scripts/chaos_report.py",
        "scripts/reset_paper_state.py",
        "scripts/neuro_guard_cycle.py",
        "tests/test_neuro_loop.py",
        "tests/test_cron_health_snapshot.py",
        "tests/test_hourly_market_snapshot.py",
        "tests/test_neuro_guard_cycle.py",
        "tests/test_cron_policy_apply_batch.py",
    ]

    if mode == "fast":
        test_cmd = [
            "python3",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_neuro_loop.py",
            "-k",
            "Cortex",
            "-k",
            "SpineWatchdog",
            "-v",
        ]
        fast_rollout_cmds = [
            {
                "name": "rollout_reporting_cron",
                "cmd": ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_cron_health_snapshot.py"],
                "timeout": 120,
            },
            {
                "name": "rollout_reporting_hourly",
                "cmd": ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_hourly_market_snapshot.py"],
                "timeout": 120,
            },
            {
                "name": "rollout_reporting_guard_cycle",
                "cmd": ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_neuro_guard_cycle.py"],
                "timeout": 120,
            },
            {
                "name": "rollout_reporting_patch_draft",
                "cmd": ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_cron_policy_patch_draft.py"],
                "timeout": 120,
            },
            {
                "name": "rollout_reporting_apply_batch",
                "cmd": ["python3", "-m", "unittest", "discover", "-s", "tests", "-p", "test_cron_policy_apply_batch.py"],
                "timeout": 120,
            },
        ]
    else:
        test_cmd = ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"]
        fast_rollout_cmds = []

    steps = [
        {"name": "py_compile", "cmd": compile_cmd, "timeout": 120},
        {"name": "unit_tests", "cmd": test_cmd, "timeout": 240},
    ]
    steps.extend(fast_rollout_cmds)

    if mode == "full":
        steps.append(
            {
                "name": "chaos_tests",
                "cmd": ["python3", "scripts/test_chaos.py"],
                "timeout": 120,
            }
        )
        steps.append(
            {
                "name": "core_smoke",
                "cmd": ["python3", "scripts/lie_spot_halfhour_core.py"],
                "timeout": 180,
            }
        )

    return steps


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _normalize_action(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "null", "na", "n/a", "unknown"}:
        return ""
    return text


def _derive_gate_ops_action(gate: Dict[str, Any]) -> Dict[str, Any]:
    core_status = str(gate.get("core_proxy_bypass_status") or "").strip().lower()
    core_execution_status = str(gate.get("core_execution_status") or "").strip().lower()
    core_execution_reason = str(gate.get("core_execution_reason") or "").strip().lower()
    core_execution_decision = str(gate.get("core_execution_decision") or "").strip().lower()
    core_execution_order_http = _int_value(gate.get("core_execution_order_http"), 0)
    core_execution_order_error = str(gate.get("core_execution_order_error") or "").strip()
    core_execution_order_simulated = bool(gate.get("core_execution_order_simulated"))
    proxy_trend_status = str(gate.get("guard_mode_proxy_error_trend_status") or "").strip().lower()
    patch_apply_status = str(gate.get("guard_mode_patch_apply_status") or "").strip().lower()
    guard_effective = str(gate.get("guard_mode_effective") or "").strip().lower()
    guard_recommended = str(gate.get("guard_mode_recommended") or "").strip().lower()
    recommend_mismatch = bool(gate.get("guard_mode_recommend_mismatch"))
    patch_next = _normalize_action(gate.get("guard_mode_patch_draft_next_action"))
    patch_next_secondary = _normalize_action(gate.get("guard_mode_patch_draft_next_action_secondary"))
    isolation_batch = (
        str(gate.get("guard_mode_patch_apply_target_batch") or "").strip()
        or str(gate.get("guard_mode_patch_draft_rollout_next_batch") or "").strip()
    )

    bypass_failed = _int_value(gate.get("core_proxy_bypass_bypass_failed"), 0)
    no_proxy_retry_exhausted = _int_value(gate.get("core_proxy_bypass_no_proxy_retry_exhausted"), 0)
    hint_exception = _int_value(gate.get("core_proxy_bypass_hint_exception"), 0)
    hint_response = _int_value(gate.get("core_proxy_bypass_hint_response"), 0)
    patch_failed_jobs = _int_value(gate.get("guard_mode_patch_apply_failed_jobs"), 0)
    retry_exhausted_degraded = max(
        1,
        _int_value(os.getenv("PI_OPS_NEXT_ACTION_PROXY_RETRY_EXHAUSTED_DEGRADED"), 1),
    )
    retry_exhausted_critical = max(
        1,
        _int_value(os.getenv("PI_OPS_NEXT_ACTION_PROXY_RETRY_EXHAUSTED_CRITICAL"), 2),
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

    if next_action == "none" and patch_next:
        next_action = patch_next
        reason = "patch_rollout_pending"
        priority = "p2"

    if secondary == next_action:
        secondary = None

    return {
        "next_action": next_action,
        "reason": reason,
        "priority": priority,
        "secondary": secondary,
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
    }


def latest_pi_cycle_gate_rollout(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "unknown", "reason": "pi_cycle_log_missing", "path": str(path)}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {"status": "degraded", "reason": "pi_cycle_log_unreadable", "path": str(path)}

    for raw in reversed(lines[-500:]):
        line = str(raw or "").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if str(obj.get("domain") or "") != "pi_cycle":
            continue
        guard_mode = {
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
            "guard_mode_proxy_error_core_jobs_count": obj.get(
                "guard_mode_proxy_error_core_jobs_count"
            ),
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
            "guard_mode_patch_draft_rollout_next_batch": obj.get("guard_mode_patch_draft_rollout_next_batch"),
            "guard_mode_patch_draft_pending": obj.get("guard_mode_patch_draft_pending"),
            "guard_mode_patch_draft_action_level": obj.get("guard_mode_patch_draft_action_level"),
            "guard_mode_patch_draft_action_reason": obj.get("guard_mode_patch_draft_action_reason"),
            "guard_mode_patch_draft_action_hints": obj.get("guard_mode_patch_draft_action_hints"),
            "guard_mode_patch_draft_next_action": obj.get("guard_mode_patch_draft_next_action"),
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
        }
        effective = str(guard_mode.get("guard_mode_effective") or "").strip().lower()
        recommended = str(guard_mode.get("guard_mode_recommended") or "").strip().lower()
        guard_mode["guard_mode_recommend_mismatch"] = bool(effective and recommended and effective != recommended)
        gate = obj.get("gate_rollout") if isinstance(obj.get("gate_rollout"), dict) else {}
        alert = obj.get("gate_rollout_alert") if isinstance(obj.get("gate_rollout_alert"), dict) else {}
        emit = obj.get("gate_rollout_alert_emit") if isinstance(obj.get("gate_rollout_alert_emit"), dict) else {}
        rollback = obj.get("gate_rollout_rollback") if isinstance(obj.get("gate_rollout_rollback"), dict) else {}
        if gate:
            out = dict(gate)
            out.update(guard_mode)
            out["pi_cycle_ts"] = obj.get("ts")
            out["alert"] = alert
            out["emit"] = emit
            out["rollback"] = rollback
            out["notify_trend_patch_action_status"] = obj.get("gate_notify_trend_patch_action_status")
            out["notify_trend_patch_action_status_reason"] = obj.get(
                "gate_notify_trend_patch_action_status_reason"
            )
            out["notify_trend_patch_action_events"] = obj.get("gate_notify_trend_patch_action_events")
            out["notify_trend_patch_action_escalated_rate"] = obj.get(
                "gate_notify_trend_patch_action_escalated_rate"
            )
            out["notify_trend_patch_action_pending_rate"] = obj.get(
                "gate_notify_trend_patch_action_pending_rate"
            )
            out["notify_trend_patch_dominant_next_action"] = obj.get(
                "gate_notify_trend_patch_dominant_next_action"
            )
            out["notify_trend_patch_dominant_next_action_count"] = obj.get(
                "gate_notify_trend_patch_dominant_next_action_count"
            )
            out["notify_trend_patch_dominant_next_action_share"] = obj.get(
                "gate_notify_trend_patch_dominant_next_action_share"
            )
            out["notify_trend_patch_dominant_action_reason"] = obj.get(
                "gate_notify_trend_patch_dominant_action_reason"
            )
            out["notify_trend_patch_dominant_action_reason_count"] = obj.get(
                "gate_notify_trend_patch_dominant_action_reason_count"
            )
            out["notify_trend_patch_dominant_action_reason_share"] = obj.get(
                "gate_notify_trend_patch_dominant_action_reason_share"
            )
            out["notify_trend_patch_recommended_next_action"] = obj.get(
                "gate_notify_trend_patch_recommended_next_action"
            )
            out["notify_trend_patch_recommended_reason"] = obj.get(
                "gate_notify_trend_patch_recommended_reason"
            )
            out["notify_trend_patch_recommended_confidence"] = obj.get(
                "gate_notify_trend_patch_recommended_confidence"
            )
            out["notify_trend_patch_action_levels_top"] = obj.get("gate_notify_trend_patch_action_levels_top")
            out["notify_trend_patch_action_reasons_top"] = obj.get("gate_notify_trend_patch_action_reasons_top")
            out["notify_trend_patch_next_actions_top"] = obj.get("gate_notify_trend_patch_next_actions_top")
            out["notify_trend_patch_action_hints_top"] = obj.get("gate_notify_trend_patch_action_hints_top")
            out["notify_trend_patch_thresholds"] = obj.get("gate_notify_trend_patch_thresholds")
            out["notify_trend_patch_apply_status"] = obj.get("gate_notify_trend_patch_apply_status")
            out["notify_trend_patch_apply_status_reason"] = obj.get(
                "gate_notify_trend_patch_apply_status_reason"
            )
            out["notify_trend_patch_apply_events"] = obj.get("gate_notify_trend_patch_apply_events")
            out["notify_trend_patch_apply_problem_events"] = obj.get(
                "gate_notify_trend_patch_apply_problem_events"
            )
            out["notify_trend_patch_apply_problem_rate"] = obj.get(
                "gate_notify_trend_patch_apply_problem_rate"
            )
            out["notify_trend_patch_apply_critical_problem_events"] = obj.get(
                "gate_notify_trend_patch_apply_critical_problem_events"
            )
            out["notify_trend_patch_apply_critical_problem_rate"] = obj.get(
                "gate_notify_trend_patch_apply_critical_problem_rate"
            )
            out["notify_trend_patch_apply_target_batch_id"] = obj.get(
                "gate_notify_trend_patch_apply_target_batch_id"
            )
            out["notify_trend_patch_apply_dominant_reason"] = obj.get(
                "gate_notify_trend_patch_apply_dominant_reason"
            )
            out["notify_trend_patch_apply_dominant_reason_count"] = obj.get(
                "gate_notify_trend_patch_apply_dominant_reason_count"
            )
            out["notify_trend_patch_apply_dominant_reason_share"] = obj.get(
                "gate_notify_trend_patch_apply_dominant_reason_share"
            )
            out["notify_trend_patch_apply_statuses_top"] = obj.get(
                "gate_notify_trend_patch_apply_statuses_top"
            )
            out["notify_trend_patch_apply_reasons_top"] = obj.get(
                "gate_notify_trend_patch_apply_reasons_top"
            )
            out["notify_trend_patch_apply_modes_top"] = obj.get("gate_notify_trend_patch_apply_modes_top")
            out["notify_trend_patch_apply_thresholds"] = obj.get("gate_notify_trend_patch_apply_thresholds")
            out["alert_emit_patch_action_level"] = emit.get("patch_draft_action_level")
            out["alert_emit_patch_action_reason"] = emit.get("patch_draft_action_reason")
            out["alert_emit_patch_next_action"] = emit.get("patch_draft_next_action")
            out["alert_emit_patch_action_hints"] = emit.get("patch_draft_action_hints")
            out["alert_emit_patch_status"] = emit.get("patch_draft_status")
            out["alert_emit_patch_pending"] = emit.get("patch_draft_pending")
            out["alert_emit_patch_failed_jobs"] = emit.get("patch_draft_failed_jobs")
            out["alert_emit_patch_actionable_jobs"] = emit.get("patch_draft_actionable_jobs")
            out["alert_emit_patch_apply_status"] = emit.get("patch_apply_status")
            out["alert_emit_patch_apply_reason"] = emit.get("patch_apply_reason")
            out["alert_emit_patch_apply_batch_id"] = emit.get("patch_apply_batch_id")
            out["alert_emit_patch_apply_mode"] = emit.get("patch_apply_mode")
            out["alert_emit_patch_apply_selected_actions"] = emit.get("patch_apply_selected_actions")
            out["alert_emit_patch_apply_changed_jobs"] = emit.get("patch_apply_changed_jobs")
            out["alert_emit_patch_apply_failed_jobs"] = emit.get("patch_apply_failed_jobs")
            out["alert_emit_patch_apply_blocked_jobs"] = emit.get("patch_apply_blocked_jobs")
            out["alert_emit_patch_apply_operations_changed"] = emit.get("patch_apply_operations_changed")
            out["alert_emit_patch_apply_trend_status"] = emit.get("patch_apply_trend_status")
            out["alert_emit_patch_apply_trend_reason"] = emit.get("patch_apply_trend_reason")
            out["alert_emit_patch_apply_trend_problem_rate"] = emit.get("patch_apply_trend_problem_rate")
            out["alert_emit_patch_apply_trend_critical_problem_rate"] = emit.get(
                "patch_apply_trend_critical_problem_rate"
            )
            out["alert_guard_mode_recommended"] = emit.get("guard_mode_recommended")
            out["alert_guard_mode_recommend_reason"] = emit.get("guard_mode_recommend_reason")
            out["alert_guard_mode_reason_winner"] = emit.get("guard_mode_reason_winner")
            out["alert_guard_mode_priority_order"] = emit.get("guard_mode_priority_order")
            out["alert_guard_mode_source_recommendations"] = emit.get("guard_mode_source_recommendations")
            ops_action = _derive_gate_ops_action(out)
            out["ops_next_action"] = ops_action.get("next_action")
            out["ops_next_action_reason"] = ops_action.get("reason")
            out["ops_next_action_priority"] = ops_action.get("priority")
            out["ops_next_action_secondary"] = ops_action.get("secondary")
            out["ops_next_action_context"] = ops_action.get("context")
            return out
        out = {
            "status": "unknown",
            "reason": "gate_rollout_missing",
            "path": str(path),
            "pi_cycle_ts": obj.get("ts"),
        }
        out.update(guard_mode)
        return out

    return {"status": "unknown", "reason": "no_pi_cycle_event", "path": str(path)}


def build_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Pi Neuro Guard Report ({summary['ts']})")
    lines.append("")
    lines.append(f"- Mode: `{summary['mode']}`")
    lines.append(f"- Status: `{summary['status']}`")
    lines.append(f"- Duration: `{summary['duration_sec']:.3f}s`")
    lines.append("")
    lines.append("## Steps")
    lines.append("")

    for step in summary.get("steps", []):
        ok = step.get("returncode") == 0
        lines.append(
            f"- `{step.get('name')}`: {'PASS' if ok else 'FAIL'} "
            f"(rc={step.get('returncode')}, {step.get('duration_sec')}s)"
        )
        if step.get("returncode") != 0:
            err = (step.get("stderr") or step.get("stdout") or "").strip()
            if err:
                lines.append(f"  - error: `{err[-300:]}`")

    gate = summary.get("gate_rollout") if isinstance(summary.get("gate_rollout"), dict) else {}
    lines.append("")
    lines.append("## Gate Rollout")
    lines.append("")
    if not gate:
        lines.append("- unavailable")
    else:
        lines.append(f"- status: `{gate.get('status')}`")
        lines.append(f"- mode_effective: `{gate.get('rollout_mode_effective')}`")
        lines.append(f"- mode_configured: `{gate.get('rollout_mode_configured')}`")
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
            f"- guard_mode_patch_inputs(use/apply/threshold): "
            f"`{gate.get('guard_mode_use_patch_draft_action')}` / "
            f"`{gate.get('guard_mode_apply_patch_draft_action')}` / "
            f"`{gate.get('guard_mode_patch_pending_threshold')}`"
        )
        lines.append(
            f"- guard_mode_patch_apply_inputs(use/apply/target): "
            f"`{gate.get('guard_mode_use_patch_apply_action')}` / "
            f"`{gate.get('guard_mode_apply_patch_apply_action')}` / "
            f"`{gate.get('guard_mode_patch_apply_target_batch')}`"
        )
        lines.append(
            f"- guard_mode_proxy_inputs(use/apply/threshold): "
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
            f"- ops_next_action(priority/action/reason/secondary): "
            f"`{gate.get('ops_next_action_priority')}` / "
            f"`{gate.get('ops_next_action')}` / "
            f"`{gate.get('ops_next_action_reason')}` / "
            f"`{gate.get('ops_next_action_secondary')}`"
        )
        lines.append(
            f"- guard_mode_patch_signal(status/action/reason/next/pending): "
            f"`{gate.get('guard_mode_patch_draft_status')}` / "
            f"`{gate.get('guard_mode_patch_draft_action_level')}` / "
            f"`{gate.get('guard_mode_patch_draft_action_reason')}` / "
            f"`{gate.get('guard_mode_patch_draft_next_action')}` / "
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
            f"- guard_mode_patch_apply_signal(status/trend/reason/batch/mode): "
            f"`{gate.get('guard_mode_patch_apply_status')}` / "
            f"`{gate.get('guard_mode_patch_apply_trend_status')}` / "
            f"`{gate.get('guard_mode_patch_apply_reason')}` / "
            f"`{gate.get('guard_mode_patch_apply_batch_id')}` / "
            f"`{gate.get('guard_mode_patch_apply_mode')}`"
        )
        lines.append(
            f"- guard_mode_patch_apply_ops(selected/changed/failed/blocked/ops_changed): "
            f"`{gate.get('guard_mode_patch_apply_selected_actions')}` / "
            f"`{gate.get('guard_mode_patch_apply_changed_jobs')}` / "
            f"`{gate.get('guard_mode_patch_apply_failed_jobs')}` / "
            f"`{gate.get('guard_mode_patch_apply_blocked_jobs')}` / "
            f"`{gate.get('guard_mode_patch_apply_operations_changed')}`"
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
        patch_hints = (
            gate.get("guard_mode_patch_draft_action_hints")
            if isinstance(gate.get("guard_mode_patch_draft_action_hints"), list)
            else []
        )
        if patch_hints:
            rendered_patch_hints = " | ".join(str(x) for x in patch_hints[:5] if str(x).strip())
            if rendered_patch_hints:
                lines.append(f"- guard_mode_patch_hints: {rendered_patch_hints}")
        lines.append(
            f"- guard_mode_recommend_mismatch: `{gate.get('guard_mode_recommend_mismatch')}`"
        )
        lines.append(
            f"- notify_trend_patch_action(status/reason/events/escalated_rate/pending_rate): "
            f"`{gate.get('notify_trend_patch_action_status')}` / "
            f"`{gate.get('notify_trend_patch_action_status_reason')}` / "
            f"`{gate.get('notify_trend_patch_action_events')}` / "
            f"`{gate.get('notify_trend_patch_action_escalated_rate')}` / "
            f"`{gate.get('notify_trend_patch_action_pending_rate')}`"
        )
        lines.append(
            f"- notify_trend_patch_next_action(dominant/recommended/reason/confidence): "
            f"`{gate.get('notify_trend_patch_dominant_next_action')}` / "
            f"`{gate.get('notify_trend_patch_recommended_next_action')}` / "
            f"`{gate.get('notify_trend_patch_recommended_reason')}` / "
            f"`{gate.get('notify_trend_patch_recommended_confidence')}`"
        )
        lines.append(
            f"- notify_trend_patch_action_reason(dominant/count/share): "
            f"`{gate.get('notify_trend_patch_dominant_action_reason')}` / "
            f"`{gate.get('notify_trend_patch_dominant_action_reason_count')}` / "
            f"`{gate.get('notify_trend_patch_dominant_action_reason_share')}`"
        )
        lines.append(
            f"- notify_trend_patch_apply(status/reason/events/problem_rate/critical_rate): "
            f"`{gate.get('notify_trend_patch_apply_status')}` / "
            f"`{gate.get('notify_trend_patch_apply_status_reason')}` / "
            f"`{gate.get('notify_trend_patch_apply_events')}` / "
            f"`{gate.get('notify_trend_patch_apply_problem_rate')}` / "
            f"`{gate.get('notify_trend_patch_apply_critical_problem_rate')}`"
        )
        lines.append(
            f"- notify_trend_patch_apply_dominant(reason/count/share/target_batch): "
            f"`{gate.get('notify_trend_patch_apply_dominant_reason')}` / "
            f"`{gate.get('notify_trend_patch_apply_dominant_reason_count')}` / "
            f"`{gate.get('notify_trend_patch_apply_dominant_reason_share')}` / "
            f"`{gate.get('notify_trend_patch_apply_target_batch_id')}`"
        )
        patch_levels_top = (
            gate.get("notify_trend_patch_action_levels_top")
            if isinstance(gate.get("notify_trend_patch_action_levels_top"), list)
            else []
        )
        if patch_levels_top:
            rendered_patch_levels = ", ".join(
                f"{item.get('level')}:{item.get('count')}"
                for item in patch_levels_top[:3]
                if isinstance(item, dict)
            )
            if rendered_patch_levels:
                lines.append(f"- notify_trend_patch_level_top: `{rendered_patch_levels}`")
        patch_reasons_top = (
            gate.get("notify_trend_patch_action_reasons_top")
            if isinstance(gate.get("notify_trend_patch_action_reasons_top"), list)
            else []
        )
        if patch_reasons_top:
            rendered_patch_reasons = ", ".join(
                f"{item.get('reason')}:{item.get('count')}"
                for item in patch_reasons_top[:3]
                if isinstance(item, dict)
            )
            if rendered_patch_reasons:
                lines.append(f"- notify_trend_patch_reason_top: `{rendered_patch_reasons}`")
        patch_next_top = (
            gate.get("notify_trend_patch_next_actions_top")
            if isinstance(gate.get("notify_trend_patch_next_actions_top"), list)
            else []
        )
        if patch_next_top:
            rendered_patch_next = ", ".join(
                f"{item.get('next_action')}:{item.get('count')}"
                for item in patch_next_top[:3]
                if isinstance(item, dict)
            )
            if rendered_patch_next:
                lines.append(f"- notify_trend_patch_next_action_top: `{rendered_patch_next}`")
        patch_hint_top = (
            gate.get("notify_trend_patch_action_hints_top")
            if isinstance(gate.get("notify_trend_patch_action_hints_top"), list)
            else []
        )
        if patch_hint_top:
            rendered_patch_hints = " | ".join(
                f"{item.get('hint')}({item.get('count')})"
                for item in patch_hint_top[:3]
                if isinstance(item, dict)
            )
            if rendered_patch_hints:
                lines.append(f"- notify_trend_patch_hint_top: {rendered_patch_hints}")
        patch_thresholds = (
            gate.get("notify_trend_patch_thresholds")
            if isinstance(gate.get("notify_trend_patch_thresholds"), dict)
            else {}
        )
        if patch_thresholds:
            lines.append(
                f"- notify_trend_patch_thresholds(min_events/degraded/critical): "
                f"`{patch_thresholds.get('min_events')}` / "
                f"`{patch_thresholds.get('rate_degraded')}` / "
                f"`{patch_thresholds.get('rate_critical')}`"
            )
        patch_apply_thresholds = (
            gate.get("notify_trend_patch_apply_thresholds")
            if isinstance(gate.get("notify_trend_patch_apply_thresholds"), dict)
            else {}
        )
        if patch_apply_thresholds:
            lines.append(
                f"- notify_trend_patch_apply_thresholds(min_events/degraded/critical): "
                f"`{patch_apply_thresholds.get('min_events')}` / "
                f"`{patch_apply_thresholds.get('problem_rate_degraded')}` / "
                f"`{patch_apply_thresholds.get('problem_rate_critical')}`"
            )
        patch_apply_statuses_top = (
            gate.get("notify_trend_patch_apply_statuses_top")
            if isinstance(gate.get("notify_trend_patch_apply_statuses_top"), list)
            else []
        )
        if patch_apply_statuses_top:
            rendered_patch_apply_statuses = ", ".join(
                f"{item.get('status')}:{item.get('count')}"
                for item in patch_apply_statuses_top[:3]
                if isinstance(item, dict)
            )
            if rendered_patch_apply_statuses:
                lines.append(f"- notify_trend_patch_apply_status_top: `{rendered_patch_apply_statuses}`")
        patch_apply_reasons_top = (
            gate.get("notify_trend_patch_apply_reasons_top")
            if isinstance(gate.get("notify_trend_patch_apply_reasons_top"), list)
            else []
        )
        if patch_apply_reasons_top:
            rendered_patch_apply_reasons = ", ".join(
                f"{item.get('reason')}:{item.get('count')}"
                for item in patch_apply_reasons_top[:3]
                if isinstance(item, dict)
            )
            if rendered_patch_apply_reasons:
                lines.append(f"- notify_trend_patch_apply_reason_top: `{rendered_patch_apply_reasons}`")
        patch_apply_modes_top = (
            gate.get("notify_trend_patch_apply_modes_top")
            if isinstance(gate.get("notify_trend_patch_apply_modes_top"), list)
            else []
        )
        if patch_apply_modes_top:
            rendered_patch_apply_modes = ", ".join(
                f"{item.get('mode')}:{item.get('count')}"
                for item in patch_apply_modes_top[:3]
                if isinstance(item, dict)
            )
            if rendered_patch_apply_modes:
                lines.append(f"- notify_trend_patch_apply_mode_top: `{rendered_patch_apply_modes}`")
        lines.append(f"- events(act/total): `{gate.get('act_events')}` / `{gate.get('total_events')}`")
        lines.append(f"- would_block_rate: `{gate.get('would_block_rate')}`")
        lines.append(f"- cooldown_hit_rate: `{gate.get('cooldown_hit_rate')}`")
        lines.append(f"- recover_confirm_fail_rate: `{gate.get('recover_confirm_fail_rate')}`")
        alert = gate.get("alert") if isinstance(gate.get("alert"), dict) else {}
        if alert:
            lines.append(
                f"- alert: `{alert.get('status')}` "
                f"(reason=`{alert.get('reason')}`, breaches={len(alert.get('breaches') or [])})"
            )
        emit = gate.get("emit") if isinstance(gate.get("emit"), dict) else {}
        notify = emit.get("notify") if isinstance(emit.get("notify"), dict) else {}
        if emit:
            lines.append(
                f"- alert_emit: emitted=`{emit.get('emitted')}` "
                f"(reason=`{emit.get('reason')}`, notify_sent=`{notify.get('sent')}`, "
                f"notify_reason=`{notify.get('reason')}`)"
            )
            lines.append(
                f"- alert_emit_patch_action(level/reason/next/status/pending): "
                f"`{gate.get('alert_emit_patch_action_level')}` / "
                f"`{gate.get('alert_emit_patch_action_reason')}` / "
                f"`{gate.get('alert_emit_patch_next_action')}` / "
                f"`{gate.get('alert_emit_patch_status')}` / "
                f"`{gate.get('alert_emit_patch_pending')}`"
            )
            lines.append(
                f"- alert_emit_patch_apply(status/trend/batch/mode/failed): "
                f"`{gate.get('alert_emit_patch_apply_status')}` / "
                f"`{gate.get('alert_emit_patch_apply_trend_status')}` / "
                f"`{gate.get('alert_emit_patch_apply_batch_id')}` / "
                f"`{gate.get('alert_emit_patch_apply_mode')}` / "
                f"`{gate.get('alert_emit_patch_apply_failed_jobs')}`"
            )
            emit_patch_hints = (
                gate.get("alert_emit_patch_action_hints")
                if isinstance(gate.get("alert_emit_patch_action_hints"), list)
                else []
            )
            if emit_patch_hints:
                rendered_emit_patch_hints = " | ".join(
                    str(x) for x in emit_patch_hints[:5] if str(x).strip()
                )
                if rendered_emit_patch_hints:
                    lines.append(f"- alert_emit_patch_hints: {rendered_emit_patch_hints}")
        rollback = gate.get("rollback") if isinstance(gate.get("rollback"), dict) else {}
        if rollback:
            lines.append(
                f"- rollback: applied=`{rollback.get('applied')}` "
                f"(reason=`{rollback.get('reason')}`, path=`{rollback.get('control_path')}`)"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--report-path", default="", help="optional markdown report path")
    args = ap.parse_args()

    started = time.time()
    ts = dt.datetime.now().isoformat()
    steps: List[Dict[str, Any]] = []
    status = "ok"

    for step in build_steps(args.mode):
        result = run_cmd(step["cmd"], timeout=int(step.get("timeout", 180)))
        result["name"] = step["name"]
        steps.append(result)
        if result["returncode"] != 0:
            status = "failed"
            break

    summary = {
        "ts": ts,
        "mode": args.mode,
        "status": status,
        "duration_sec": round(time.time() - started, 3),
        "steps": steps,
        "gate_rollout": latest_pi_cycle_gate_rollout(PI_CYCLE_LOG),
    }

    if args.report_path:
        report_path = Path(args.report_path)
    else:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORT_DIR / f"{dt.date.today().isoformat()}_pi_neuro_guard_{args.mode}.md"
    report_path.write_text(build_markdown(summary) + "\n", encoding="utf-8")
    summary["report_path"] = str(report_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
