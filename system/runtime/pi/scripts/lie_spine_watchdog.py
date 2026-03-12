#!/usr/bin/env python3
"""LiE spinal reflex watchdog.

Autonomic layer that can trigger hard circuit-breaker actions without waiting for
LLM reasoning. It merges two sensors:
1) Cortex mode (IRPOTA state machine)
2) Exchange latency pulse (ICMP ping)

When danger is detected it physically disables trading configs and emits a
structured event envelope.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cortex_evaluator import StateVectorCortex
from lie_root_resolver import resolve_lie_system_root

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional runtime dep
    yaml = None


CRITICAL_MODES = {"SURVIVAL", "STABILIZE", "HIBERNATE", "ROLLBACK"}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _runtime_paths() -> Dict[str, Path]:
    lie_root = resolve_lie_system_root()
    workspace = Path(
        os.getenv(
            "OPENCLAW_STATE_WORKSPACE",
            str(Path(__file__).resolve().parents[1]),
        )
    )
    return {
        "lie_root": lie_root,
        "workspace": workspace,
        "state_md": workspace / "STATE.md",
        "log_jsonl": lie_root / "output" / "logs" / "spine_watchdog_events.jsonl",
        "watchdog_state": lie_root / "output" / "logs" / "spine_watchdog_state.json",
        "params_config": lie_root / "config" / "params_live.yaml",
        "params_artifacts": lie_root / "output" / "artifacts" / "params_live.yaml",
    }


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _append_state_line(state_md: Path, payload: Dict[str, Any]) -> None:
    try:
        state_md.parent.mkdir(parents=True, exist_ok=True)
        with state_md.open("a", encoding="utf-8") as f:
            f.write("\nSPINE_REFLEX_EVENT=" + json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_watchdog_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"latency_violation_streak": 0, "last_trigger_ts": None}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"latency_violation_streak": 0, "last_trigger_ts": None}


def _save_watchdog_state(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _parse_ping_avg_ms(text: str) -> Optional[float]:
    # macOS: round-trip min/avg/max/stddev = 15.059/15.311/15.621/0.236 ms
    # Linux: rtt min/avg/max/mdev = 12.680/13.261/13.861/0.478 ms
    m = re.search(r"(?:round-trip|rtt)\s+min/avg/max/(?:stddev|mdev)\s*=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)/", text)
    if m:
        try:
            return float(m.group(2))
        except Exception:
            return None
    return None


def ping_latency_ms(host: str, timeout_ms: int) -> Optional[float]:
    if os.getenv("SPINE_PING_ENABLED", "1") == "0":
        return None

    timeout_arg = str(max(1000, int(timeout_ms)))
    cmd = ["ping", "-c", "2", "-W", timeout_arg, host]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception:
        return float("inf")

    out = (p.stdout or "") + "\n" + (p.stderr or "")
    avg = _parse_ping_avg_ms(out)
    if avg is not None:
        return avg
    if p.returncode != 0:
        return float("inf")
    return None


def _upsert_yaml_text(text: str, key: str, value: str) -> str:
    pat = re.compile(rf"(?m)^\s*{re.escape(key)}\s*:\s*.*$")
    line = f"{key}: {value}"
    if pat.search(text):
        return pat.sub(line, text)
    if not text.endswith("\n"):
        text += "\n"
    return text + line + "\n"


def _disable_params_file(path: Path, reason: str, ts: str) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "changed": False, "enabled_before": None}

    enabled_before: Optional[bool] = None
    changed = False

    if yaml is not None:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            data = raw if isinstance(raw, dict) else {}
            enabled_before = bool(data.get("enabled", True))
            if enabled_before:
                changed = True
            data["enabled"] = False
            data["reflex_lock"] = "ACTIVE"
            data["reflex_reason"] = reason
            data["reflex_ts"] = ts
            path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        except Exception as e:
            return {
                "path": str(path),
                "exists": True,
                "changed": False,
                "enabled_before": enabled_before,
                "error": str(e)[:180],
            }
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"(?m)^\s*enabled\s*:\s*(\S+)\s*$", text)
        if m:
            enabled_before = m.group(1).strip().lower() in {"1", "true", "yes", "on"}
        else:
            enabled_before = None

        if enabled_before is not False:
            changed = True
        text = _upsert_yaml_text(text, "enabled", "false")
        text = _upsert_yaml_text(text, "reflex_lock", "ACTIVE")
        text = _upsert_yaml_text(text, "reflex_reason", json.dumps(reason, ensure_ascii=False))
        text = _upsert_yaml_text(text, "reflex_ts", json.dumps(ts, ensure_ascii=False))
        path.write_text(text, encoding="utf-8")

    return {
        "path": str(path),
        "exists": True,
        "changed": changed,
        "enabled_before": enabled_before,
    }


def _decide_reasons(
    *,
    mode: str,
    latency_ms: Optional[float],
    latency_threshold_ms: float,
    latency_streak: int,
    latency_streak_required: int,
) -> List[str]:
    reasons: List[str] = []
    if mode in CRITICAL_MODES:
        reasons.append(f"critical_mode:{mode}")

    if latency_ms is not None:
        if latency_ms == float("inf"):
            if latency_streak >= latency_streak_required:
                reasons.append("latency_unreachable")
        elif latency_ms >= latency_threshold_ms and latency_streak >= latency_streak_required:
            reasons.append(
                f"latency_high:{latency_ms:.2f}ms>=th:{latency_threshold_ms:.2f}ms(streak={latency_streak})"
            )

    return reasons


def run_once() -> Dict[str, Any]:
    paths = _runtime_paths()
    ts = _now_iso()

    ping_host = os.getenv("SPINE_PING_TARGET", "api.binance.com")
    latency_threshold_ms = float(os.getenv("SPINE_CRIT_LATENCY_MS", "2000"))
    latency_streak_required = int(os.getenv("SPINE_LATENCY_STREAK_REQUIRED", "2"))

    mode = "STABILIZE"
    debug: Dict[str, Any] = {"reason": "cortex_unavailable"}
    try:
        mode, debug = StateVectorCortex(lie_root=paths["lie_root"]).eval_irpota()
    except Exception as e:
        debug = {"reason": "cortex_eval_error", "error": str(e)[:200]}

    latency_ms = ping_latency_ms(ping_host, int(latency_threshold_ms))

    wd_state = _load_watchdog_state(paths["watchdog_state"])
    streak = int(wd_state.get("latency_violation_streak") or 0)
    if latency_ms is not None and (
        latency_ms == float("inf") or latency_ms >= latency_threshold_ms
    ):
        streak += 1
    else:
        streak = 0

    reasons = _decide_reasons(
        mode=mode,
        latency_ms=latency_ms,
        latency_threshold_ms=latency_threshold_ms,
        latency_streak=streak,
        latency_streak_required=latency_streak_required,
    )
    triggered = len(reasons) > 0

    updates: List[Dict[str, Any]] = []
    if triggered:
        reason_text = ";".join(reasons)
        updates.append(_disable_params_file(paths["params_config"], reason_text, ts))
        updates.append(_disable_params_file(paths["params_artifacts"], reason_text, ts))
        wd_state["last_trigger_ts"] = ts

    wd_state["latency_violation_streak"] = streak
    _save_watchdog_state(paths["watchdog_state"], wd_state)

    event = {
        "envelope_version": "1.0",
        "domain": "spine_watchdog",
        "ts": ts,
        "triggered": triggered,
        "reasons": reasons,
        "mode": mode,
        "mode_trigger": debug.get("trigger") or debug.get("reason"),
        "latency_ms": latency_ms,
        "latency_threshold_ms": latency_threshold_ms,
        "latency_streak": streak,
        "latency_streak_required": latency_streak_required,
        "ping_host": ping_host,
        "updates": updates,
        "state": {
            "watchdog_state_path": str(paths["watchdog_state"]),
            "state_md": str(paths["state_md"]),
            "log_jsonl": str(paths["log_jsonl"]),
        },
    }

    _append_jsonl(paths["log_jsonl"], event)
    _append_state_line(paths["state_md"], event)
    return event


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one check and exit")
    ap.add_argument("--interval-sec", type=int, default=10, help="loop interval in seconds")
    ap.add_argument("--max-loops", type=int, default=0, help="max loop count (0 means unlimited)")
    args = ap.parse_args()

    loops = 0
    while True:
        event = run_once()
        print(json.dumps(event, ensure_ascii=False))

        loops += 1
        if args.once:
            return 0
        if args.max_loops > 0 and loops >= args.max_loops:
            return 0
        time.sleep(max(1, int(args.interval_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
