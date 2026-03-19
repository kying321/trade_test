#!/usr/bin/env python3
"""Envelope contract lint for Pi runtime events."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lie_root_resolver import resolve_lie_system_root

SYSTEM_ROOT = resolve_lie_system_root()
PI_WORKSPACE = Path(__file__).resolve().parents[1]
STATE_MD = Path(os.getenv("OPENCLAW_STATE_FILE", str(PI_WORKSPACE / "STATE.md")))
ROLLOUT_STATE_PATH = Path(
    os.getenv(
        "CORTEX_GATE_ROLLOUT_STATE_PATH",
        str(SYSTEM_ROOT / "output" / "logs" / "cortex_gate_rollout_state.json"),
    )
)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_ts(ts: Any) -> Optional[dt.datetime]:
    if not isinstance(ts, str) or not ts:
        return None
    value = ts
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def lint_jsonl(
    path: Path,
    expected_domain: str,
    required_fields: List[str],
    since: dt.datetime,
    strict_since: dt.datetime,
    legacy_mode: str,
    extra_required_fields: Optional[List[str]] = None,
    extra_required_since: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    extra_required_fields = list(extra_required_fields or [])
    out: Dict[str, Any] = {
        "path": str(path),
        "expected_domain": expected_domain,
        "required_fields": required_fields,
        "extra_required_fields": extra_required_fields,
        "extra_required_since_utc": extra_required_since.isoformat() if extra_required_since else None,
        "entries_total": 0,
        "entries_in_window": 0,
        "invalid_json": 0,
        "domain_mismatch": 0,
        "missing_fields": 0,
        "latest_ts": None,
        "missing_field_examples": [],
        "strict_window_entries": 0,
        "legacy_window_entries": 0,
        "strict_fail_count": 0,
        "legacy_excluded_count": 0,
    }
    if not path.exists():
        out["missing_file"] = True
        return out

    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        out["read_error"] = True
        return out

    out["entries_total"] = len(lines)
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            out["invalid_json"] += 1
            continue
        if not isinstance(obj, dict):
            out["invalid_json"] += 1
            continue

        ts = parse_ts(obj.get("ts"))
        if ts is None or ts < since:
            continue

        out["entries_in_window"] += 1
        if out["latest_ts"] is None or ts.isoformat() > str(out["latest_ts"]):
            out["latest_ts"] = ts.isoformat()

        is_legacy = ts < strict_since
        if is_legacy and legacy_mode == "ignore_for_strict":
            out["legacy_window_entries"] += 1
            fail_legacy = False
            if obj.get("domain") != expected_domain:
                fail_legacy = True
            missing = [k for k in required_fields if k not in obj]
            if missing:
                fail_legacy = True
            if fail_legacy:
                out["legacy_excluded_count"] += 1
            continue

        out["strict_window_entries"] += 1
        failed = False
        if obj.get("domain") != expected_domain:
            out["domain_mismatch"] += 1
            failed = True

        missing = [k for k in required_fields if k not in obj]
        if extra_required_fields and extra_required_since is not None and ts >= extra_required_since:
            missing.extend([k for k in extra_required_fields if k not in obj])
        if missing:
            out["missing_fields"] += 1
            failed = True
            if len(out["missing_field_examples"]) < 3:
                out["missing_field_examples"].append({"ts": obj.get("ts"), "missing": missing})
        if failed:
            out["strict_fail_count"] += 1
    return out


def lint_state_snapshot(
    since: dt.datetime,
    strict_since: dt.datetime,
    legacy_mode: str,
    extra_required_fields: Optional[List[str]] = None,
    extra_required_since: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    required = [
        "envelope_version",
        "domain",
        "ts",
        "source",
        "cycle_last_ts",
        "cycle_last_status",
        "last_mode",
        "paper_equity",
        "paper_drawdown",
        "risk_state",
        "staleness_sec",
    ]
    extra_required_fields = list(extra_required_fields or [])
    out: Dict[str, Any] = {
        "path": str(STATE_MD),
        "prefix": "HOURLY_SNAPSHOT_EVENT=",
        "required_fields": required,
        "extra_required_fields": extra_required_fields,
        "extra_required_since_utc": extra_required_since.isoformat() if extra_required_since else None,
        "entries_in_window": 0,
        "invalid_json": 0,
        "missing_fields": 0,
        "domain_mismatch": 0,
        "latest_ts": None,
        "missing_field_examples": [],
        "strict_window_entries": 0,
        "legacy_window_entries": 0,
        "strict_fail_count": 0,
        "legacy_excluded_count": 0,
    }
    if not STATE_MD.exists():
        out["missing_file"] = True
        return out

    try:
        with STATE_MD.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        out["read_error"] = True
        return out

    pref = "HOURLY_SNAPSHOT_EVENT="
    for raw in lines:
        if not raw.startswith(pref):
            continue
        payload = raw[len(pref) :].strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            out["invalid_json"] += 1
            continue
        ts = parse_ts(obj.get("ts"))
        if ts is None or ts < since:
            continue
        out["entries_in_window"] += 1
        if out["latest_ts"] is None or ts.isoformat() > str(out["latest_ts"]):
            out["latest_ts"] = ts.isoformat()
        is_legacy = ts < strict_since
        if is_legacy and legacy_mode == "ignore_for_strict":
            out["legacy_window_entries"] += 1
            fail_legacy = False
            if obj.get("domain") != "hourly_snapshot":
                fail_legacy = True
            missing = [k for k in required if k not in obj]
            if missing:
                fail_legacy = True
            if fail_legacy:
                out["legacy_excluded_count"] += 1
            continue
        out["strict_window_entries"] += 1
        failed = False
        if obj.get("domain") != "hourly_snapshot":
            out["domain_mismatch"] += 1
            failed = True
        missing = [k for k in required if k not in obj]
        if extra_required_fields and extra_required_since is not None and ts >= extra_required_since:
            missing.extend([k for k in extra_required_fields if k not in obj])
        if missing:
            out["missing_fields"] += 1
            failed = True
            if len(out["missing_field_examples"]) < 3:
                out["missing_field_examples"].append({"ts": obj.get("ts"), "missing": missing})
        if failed:
            out["strict_fail_count"] += 1
    return out


def aggregate_status(checks: List[Tuple[str, Dict[str, Any]]]) -> str:
    has_critical = False
    has_warn = False
    for _, c in checks:
        if c.get("missing_file") or c.get("read_error"):
            has_critical = True
        strict_fail_count = int(c.get("strict_fail_count", int(c.get("domain_mismatch", 0)) + int(c.get("missing_fields", 0))))
        if c.get("invalid_json", 0) > 0 or strict_fail_count > 0:
            has_warn = True
    if has_critical:
        return "error"
    if has_warn:
        return "degraded"
    return "ok"


def resolve_strict_since(since: dt.datetime, arg_value: str) -> Tuple[dt.datetime, str]:
    if arg_value:
        parsed = parse_ts(arg_value)
        if parsed is not None:
            return parsed, "arg"

    if ROLLOUT_STATE_PATH.exists():
        try:
            obj = json.loads(ROLLOUT_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                parsed = parse_ts(obj.get("rollout_started_ts"))
                if parsed is not None:
                    return parsed, "rollout_state"
        except Exception:
            pass
    return since, "window_start"


def main() -> int:
    ap = argparse.ArgumentParser(description="Lint Pi event envelopes in recent window.")
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--strict-since-utc", type=str, default="")
    ap.add_argument("--hourly-v2-strict-since-utc", type=str, default="")
    ap.add_argument("--legacy-mode", choices=["include", "ignore_for_strict"], default="ignore_for_strict")
    args = ap.parse_args()

    since = now_utc() - dt.timedelta(hours=max(0.1, float(args.window_hours)))
    strict_since, strict_since_source = resolve_strict_since(since=since, arg_value=str(args.strict_since_utc or "").strip())
    legacy_mode = str(args.legacy_mode)
    hourly_v2_strict_raw = str(args.hourly_v2_strict_since_utc or "").strip() or str(
        os.getenv("PI_ENVELOPE_HOURLY_V2_STRICT_SINCE_UTC", "")
    ).strip()
    hourly_v2_strict_since = parse_ts(hourly_v2_strict_raw)
    hourly_v2_extra_fields = [
        "gate_notify_trend_component_known",
        "gate_notify_trend_component_score",
        "gate_notify_trend_component_score_status",
        "gate_notify_trend_component_weights",
        "gate_notify_trend_component_thresholds",
    ]
    checks: List[Tuple[str, Dict[str, Any]]] = []
    checks.append(
        (
            "pi_cycle",
            lint_jsonl(
                SYSTEM_ROOT / "output" / "logs" / "pi_cycle_events.jsonl",
                expected_domain="pi_cycle",
                required_fields=["envelope_version", "domain", "ts", "status", "duration_sec", "plan", "steps"],
                since=since,
                strict_since=strict_since,
                legacy_mode=legacy_mode,
            ),
        )
    )
    checks.append(
        (
            "cortex_gate",
            lint_jsonl(
                SYSTEM_ROOT / "output" / "logs" / "cortex_gate_events.jsonl",
                expected_domain="cortex_gate",
                required_fields=[
                    "envelope_version",
                    "domain",
                    "ts",
                    "function",
                    "mode",
                    "policy",
                    "outcome",
                    "action_class",
                    "reason",
                ],
                since=since,
                strict_since=strict_since,
                legacy_mode=legacy_mode,
            ),
        )
    )
    checks.append(
        (
            "spine_watchdog",
            lint_jsonl(
                SYSTEM_ROOT / "output" / "logs" / "spine_watchdog_events.jsonl",
                expected_domain="spine_watchdog",
                required_fields=["envelope_version", "domain", "ts", "triggered", "reasons", "mode"],
                since=since,
                strict_since=strict_since,
                legacy_mode=legacy_mode,
            ),
        )
    )
    checks.append(
        (
            "hourly_snapshot_state",
            lint_state_snapshot(
                since=since,
                strict_since=strict_since,
                legacy_mode=legacy_mode,
                extra_required_fields=hourly_v2_extra_fields,
                extra_required_since=hourly_v2_strict_since,
            ),
        )
    )
    checks.append(
        (
            "hourly_snapshot_jsonl",
            lint_jsonl(
                SYSTEM_ROOT / "output" / "logs" / "hourly_snapshot_events.jsonl",
                expected_domain="hourly_snapshot",
                required_fields=[
                    "envelope_version",
                    "domain",
                    "ts",
                    "source",
                    "cycle_last_ts",
                    "cycle_last_status",
                    "last_mode",
                    "paper_equity",
                    "paper_drawdown",
                    "risk_state",
                    "staleness_sec",
                ],
                since=since,
                strict_since=strict_since,
                legacy_mode=legacy_mode,
                extra_required_fields=hourly_v2_extra_fields,
                extra_required_since=hourly_v2_strict_since,
            ),
        )
    )

    strict_window_entries = sum(int((check or {}).get("strict_window_entries", 0)) for _, check in checks)
    legacy_window_entries = sum(int((check or {}).get("legacy_window_entries", 0)) for _, check in checks)
    strict_fail_count = sum(int((check or {}).get("strict_fail_count", 0)) for _, check in checks)
    legacy_excluded_count = sum(int((check or {}).get("legacy_excluded_count", 0)) for _, check in checks)

    payload = {
        "envelope_version": "1.0",
        "domain": "envelope_lint",
        "ts": now_utc().isoformat(),
        "window_hours": float(args.window_hours),
        "since_utc": since.isoformat(),
        "strict_since_utc": strict_since.isoformat(),
        "strict_since_source": strict_since_source,
        "hourly_v2_strict_since_utc": hourly_v2_strict_since.isoformat() if hourly_v2_strict_since else None,
        "legacy_mode": legacy_mode,
        "strict_window_entries": strict_window_entries,
        "legacy_window_entries": legacy_window_entries,
        "strict_fail_count": strict_fail_count,
        "legacy_excluded_count": legacy_excluded_count,
        "status": aggregate_status(checks),
        "checks": {name: check for name, check in checks},
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
