#!/usr/bin/env python3
"""Digital-life core snapshot for Pi.

Single-cycle objective:
- Read recent gate/runtime signals
- Estimate life-state (survive/adapt/explore)
- Persist append-only event + latest state snapshot
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from lie_root_resolver import resolve_lie_system_root

SYSTEM_ROOT = resolve_lie_system_root()
GATE_LOG_PATH = Path(
    os.getenv("CORTEX_GATE_LOG", str(SYSTEM_ROOT / "output" / "logs" / "cortex_gate_events.jsonl"))
)
DOOMSDAY_PATH = Path(
    os.getenv("CORTEX_DOOMSDAY_VECTOR_PATH", str(SYSTEM_ROOT / "output" / "logs" / "cortex_doomsday_vector.json"))
)
PROBE_BUDGET_PATH = Path(
    os.getenv("CORTEX_PROBE_BUDGET_STATE_PATH", str(SYSTEM_ROOT / "output" / "state" / "cortex_probe_budget.json"))
)
STATE_PATH = Path(
    os.getenv("PI_DIGITAL_LIFE_STATE_PATH", str(SYSTEM_ROOT / "output" / "logs" / "digital_life_state.json"))
)
EVENTS_PATH = Path(
    os.getenv("PI_DIGITAL_LIFE_EVENTS_PATH", str(SYSTEM_ROOT / "output" / "logs" / "digital_life_events.jsonl"))
)


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _parse_ts(value: Any) -> Optional[dt.datetime]:
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def _write_json(path: Path, payload: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return True
    except Exception:
        return False


def _load_recent_gate_events(path: Path, *, window_hours: float, max_lines: int) -> List[Dict[str, Any]]:
    if (not path.exists()) or window_hours <= 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []

    now = _now_utc()
    cutoff = now - dt.timedelta(hours=float(window_hours))
    out: List[Dict[str, Any]] = []
    for raw in lines[-max(50, int(max_lines)) :]:
        raw = raw.strip()
        if not raw or not raw.startswith("{"):
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        ts = _parse_ts(obj.get("ts"))
        if ts is None or ts < cutoff:
            continue
        out.append(obj)
    return out


def _extract_state(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    for obj in reversed(events):
        state = obj.get("state")
        if isinstance(state, dict):
            return state
        debug = obj.get("debug")
        if isinstance(debug, dict):
            state2 = debug.get("state")
            if isinstance(state2, dict):
                return state2
    return {}


def _gate_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(events)
    if total <= 0:
        return {
            "events": 0,
            "would_block_rate": None,
            "cooldown_hit_rate": None,
            "recover_confirm_fail_rate": None,
            "reflex_ratio": None,
            "last_mode": None,
            "last_certificate_level": None,
        }
    would_block = 0
    cooldown_hits = 0
    recover_fail = 0
    reflex = 0
    for obj in events:
        if bool(obj.get("would_block", False)):
            would_block += 1
        if bool(obj.get("cooldown_active", False)):
            cooldown_hits += 1
        if obj.get("recover_confirm_passed") is False:
            recover_fail += 1
        if str(obj.get("actuator_bus") or "").strip().lower() == "reflex":
            reflex += 1
    latest = events[-1]
    return {
        "events": total,
        "would_block_rate": round(float(would_block / total), 6),
        "cooldown_hit_rate": round(float(cooldown_hits / total), 6),
        "recover_confirm_fail_rate": round(float(recover_fail / total), 6),
        "reflex_ratio": round(float(reflex / total), 6),
        "last_mode": latest.get("mode"),
        "last_certificate_level": latest.get("certificate_level"),
    }


def _life_score(state: Dict[str, Any], gate: Dict[str, Any]) -> Dict[str, Any]:
    i_val = _clamp(_safe_float(state.get("I"), 0.0), 0.0, 1.0)
    r_val = _clamp(_safe_float(state.get("R"), 0.0), 0.0, 1.0)
    p_val = _clamp(_safe_float(state.get("P"), 0.0), 0.0, 1.0)
    o_val = _clamp(_safe_float(state.get("O"), 0.0), 0.0, 1.0)
    t_val = max(0, _safe_int(state.get("T"), 3))
    a_val = bool(state.get("A", False))

    core = (i_val + r_val + p_val + o_val) / 4.0
    trust_penalty = _clamp(1.0 - 0.15 * float(max(0, t_val - 1)), 0.4, 1.0)
    align_penalty = 1.0 if a_val else 0.0
    viability = _clamp(core * trust_penalty * align_penalty, 0.0, 1.0)

    block_rate = _safe_float(gate.get("would_block_rate"), 0.0)
    cooldown_rate = _safe_float(gate.get("cooldown_hit_rate"), 0.0)
    recover_fail_rate = _safe_float(gate.get("recover_confirm_fail_rate"), 0.0)
    stress = _clamp(0.5 * block_rate + 0.3 * cooldown_rate + 0.2 * recover_fail_rate, 0.0, 1.0)

    if (not a_val) or viability < 0.35:
        lifecycle = "SURVIVE"
        next_action = "shrink_optionality_and_lock_reflex"
    elif stress > 0.45:
        lifecycle = "STABILIZE"
        next_action = "reduce_action_frequency_and_raise_guard"
    elif viability >= 0.75 and stress <= 0.20:
        lifecycle = "EXPLORE"
        next_action = "enable_small_cmu_experiments"
    else:
        lifecycle = "ADAPT"
        next_action = "maintain_barbell_and_update_local_rules"

    return {
        "viability_score": round(float(viability), 6),
        "stress_score": round(float(stress), 6),
        "lifecycle_mode": lifecycle,
        "next_action": next_action,
        "components": {
            "I": i_val,
            "R": r_val,
            "P": p_val,
            "O": o_val,
            "T": t_val,
            "A": a_val,
            "core": round(core, 6),
            "trust_penalty": round(trust_penalty, 6),
        },
    }


def run_cycle(*, window_hours: float, max_lines: int) -> Dict[str, Any]:
    events = _load_recent_gate_events(GATE_LOG_PATH, window_hours=window_hours, max_lines=max_lines)
    gate = _gate_metrics(events)
    state = _extract_state(events)
    life = _life_score(state, gate)
    doomsday = _load_json(DOOMSDAY_PATH)
    probe_budget = _load_json(PROBE_BUDGET_PATH)

    payload = {
        "envelope_version": "1.0",
        "domain": "digital_life_core",
        "ts": _now_iso(),
        "window_hours": float(window_hours),
        "gate_log_path": str(GATE_LOG_PATH),
        "gate": gate,
        "state": {
            "I": state.get("I"),
            "R": state.get("R"),
            "P": state.get("P"),
            "O": state.get("O"),
            "T": state.get("T"),
            "A": state.get("A"),
        },
        "doomsday_vector": doomsday,
        "probe_budget": {
            "date": probe_budget.get("date"),
            "probe_notional_used": probe_budget.get("probe_notional_used"),
            "probe_projected_loss_used": probe_budget.get("probe_projected_loss_used"),
            "halted": probe_budget.get("halted"),
            "state_path": str(PROBE_BUDGET_PATH),
        },
    }
    payload.update(life)

    persisted_state_ok = _write_json(STATE_PATH, payload)
    persisted_event_ok = _append_jsonl(EVENTS_PATH, payload)
    payload["persisted"] = {
        "state_path": str(STATE_PATH),
        "events_path": str(EVENTS_PATH),
        "state_ok": bool(persisted_state_ok),
        "event_ok": bool(persisted_event_ok),
    }
    payload["status"] = (
        "ok"
        if bool(persisted_state_ok and persisted_event_ok)
        else ("degraded" if (persisted_state_ok or persisted_event_ok) else "error")
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Build one digital-life control snapshot")
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--max-lines", type=int, default=5000)
    args = ap.parse_args()

    payload = run_cycle(window_hours=max(0.1, float(args.window_hours)), max_lines=max(200, int(args.max_lines)))
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if str(payload.get("status")) in {"ok", "degraded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
