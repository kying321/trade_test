#!/usr/bin/env python3
"""Cortex action-gating middleware.

Implements Appendix C reversibility ladder:
- Reject hard actions in SURVIVAL/HIBERNATE/ROLLBACK/STABILIZE.
- Downgrade to SIMULATE or MICRO-PROBE when mode is OBSERVE/LEARN.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from cortex_control_kernel import (
    AppendOnlyAuditLedger,
    DoomsdayVectorStore,
    ProbeBudgetGuard,
    evaluate_doomsday_guard,
    issue_certificate,
    probe_hard_invariants,
    resolve_actuator_bus,
)
from cortex_evaluator import Mode, StateVectorCortex
from lie_root_resolver import resolve_lie_system_root


@dataclass
class GateResult:
    mode: str
    policy: str
    action_class: str
    reason: str
    debug: Dict[str, Any]


class SpinalReflexReject(RuntimeError):
    def __init__(self, gate: GateResult):
        super().__init__(f"spinal_reflex_reject mode={gate.mode} policy={gate.policy} reason={gate.reason}")
        self.gate = gate


_CORTEX_SINGLETON: Optional[StateVectorCortex] = None
_AUDIT_LEDGER_SINGLETON: Optional[AppendOnlyAuditLedger] = None
_AUDIT_LEDGER_PATH_CACHE: Optional[str] = None
_GATE_STATE_FALLBACK: Dict[str, Any] = {}
_ROLLOUT_STATE_FALLBACK: Dict[str, Any] = {}
_STATE_FALLBACK_SCOPE: Dict[str, Optional[str]] = {"gate": None, "rollout": None}

_DEFAULT_SYSTEM_ROOT = resolve_lie_system_root()
_DEFAULT_GATE_LOG_PATH = str(_DEFAULT_SYSTEM_ROOT / "output" / "logs" / "cortex_gate_events.jsonl")
_DEFAULT_GATE_STATE_PATH = str(_DEFAULT_SYSTEM_ROOT / "output" / "logs" / "cortex_gate_state.json")
_DEFAULT_ROLLOUT_STATE_PATH = str(_DEFAULT_SYSTEM_ROOT / "output" / "logs" / "cortex_gate_rollout_state.json")
_DEFAULT_ROLLOUT_CONTROL_PATH = str(_DEFAULT_SYSTEM_ROOT / "output" / "logs" / "cortex_gate_rollout_control.json")


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _now_iso() -> str:
    return _now_utc().isoformat()


def _parse_iso_ts(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    raw = value
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _audit_path() -> Path:
    return Path(os.getenv("CORTEX_GATE_LOG", _DEFAULT_GATE_LOG_PATH))


def _audit_ring_path() -> Path:
    return Path(os.getenv("CORTEX_GATE_RING_LOG", str(_audit_path().with_name("cortex_gate_events.ring.jsonl"))))


def _gate_state_path() -> Path:
    return Path(os.getenv("CORTEX_GATE_STATE_PATH", _DEFAULT_GATE_STATE_PATH))


def _rollout_state_path() -> Path:
    return Path(os.getenv("CORTEX_GATE_ROLLOUT_STATE_PATH", _DEFAULT_ROLLOUT_STATE_PATH))


def _rollout_control_path() -> Path:
    return Path(os.getenv("CORTEX_GATE_ROLLOUT_CONTROL_PATH", _DEFAULT_ROLLOUT_CONTROL_PATH))


def _get_cortex() -> StateVectorCortex:
    global _CORTEX_SINGLETON
    if _CORTEX_SINGLETON is None:
        _CORTEX_SINGLETON = StateVectorCortex()
    return _CORTEX_SINGLETON


def _get_audit_ledger() -> AppendOnlyAuditLedger:
    global _AUDIT_LEDGER_SINGLETON, _AUDIT_LEDGER_PATH_CACHE
    path = _audit_path()
    ring = _audit_ring_path()
    cache_key = f"{path}|{ring}"
    if (_AUDIT_LEDGER_SINGLETON is None) or (_AUDIT_LEDGER_PATH_CACHE != cache_key):
        _AUDIT_LEDGER_SINGLETON = AppendOnlyAuditLedger(path=path, ring_path=ring, ring_limit=512)
        _AUDIT_LEDGER_PATH_CACHE = cache_key
    return _AUDIT_LEDGER_SINGLETON


def _append_gate_audit(entry: Dict[str, Any]) -> None:
    try:
        _get_audit_ledger().append(entry)
    except Exception:
        pass


def _probe_notional(
    cap_probe: Optional[Callable[[Sequence[Any], Dict[str, Any]], Dict[str, float]]],
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> float:
    if cap_probe is None:
        return 0.0
    try:
        measured = cap_probe(args, kwargs)
    except Exception:
        return 0.0
    if not isinstance(measured, dict):
        return 0.0
    if "notional" in measured:
        try:
            return abs(float(measured.get("notional")))
        except Exception:
            return 0.0
    if "qty" in measured and "px" in measured:
        try:
            return abs(float(measured.get("qty"))) * abs(float(measured.get("px")))
        except Exception:
            return 0.0
    return 0.0


def _build_envelope(fn_name: str, gate: GateResult, outcome: str) -> Dict[str, Any]:
    state = (gate.debug or {}).get("state", {}) if isinstance(gate.debug, dict) else {}
    cycle_id = (gate.debug or {}).get("cycle") if isinstance(gate.debug, dict) else None
    ext = (gate.debug or {}).get("gate_extensions", {}) if isinstance(gate.debug, dict) else {}
    if not isinstance(ext, dict):
        ext = {}
    compact_state = {
        "I": state.get("I"),
        "R": state.get("R"),
        "P": state.get("P"),
        "O": state.get("O"),
        "T": state.get("T"),
        "A": state.get("A"),
    }
    return {
        "envelope_version": "1.0",
        "domain": "cortex_gate",
        "ts": _now_iso(),
        "cycle_id": cycle_id,
        "function": fn_name,
        "mode": gate.mode,
        "policy": gate.policy,
        "outcome": outcome,
        "action_class": gate.action_class,
        "reason": gate.reason,
        "rollout_mode_effective": ext.get("rollout_mode_effective"),
        "cooldown_active": bool(ext.get("cooldown_active", False)),
        "recover_confirm_passed": ext.get("recover_confirm_passed"),
        "recover_confirm_hits": ext.get("recover_confirm_hits"),
        "would_block": bool(ext.get("would_block", False)),
        "actuator_bus": ext.get("actuator_bus"),
        "certificate_level": ((ext.get("certificate") or {}) if isinstance(ext.get("certificate"), dict) else {}).get("level"),
        "reflex_reduce_only": bool(ext.get("reflex_reduce_only", False)),
        "gate_extensions": ext,
        "state": compact_state,
        "debug": gate.debug,
    }


def _default_policy(mode: str, on_observe: str, on_learn: str, on_stabilize: str) -> str:
    if mode == Mode.ACT.value:
        return "allow"
    if mode == Mode.LEARN.value:
        return on_learn
    if mode == Mode.OBSERVE.value:
        return on_observe
    if mode == Mode.STABILIZE.value:
        return on_stabilize
    return "reject"


def _default_simulate_payload(
    fn_name: str,
    mode: str,
    action_class: str,
    reason: str,
    debug: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "decision": "simulate",
        "mode": mode,
        "action_class": action_class,
        "function": fn_name,
        "reason": reason,
        "cortex": debug,
    }


def _resolve_cap_limits(
    cap_limits: Optional[Union[Dict[str, float], Callable[..., Dict[str, float]]]],
    gate: Optional[GateResult] = None,
) -> Dict[str, float]:
    if cap_limits is None:
        return {}
    if callable(cap_limits):
        try:
            out = cap_limits(gate)
        except TypeError:
            try:
                out = cap_limits()
            except Exception:
                return {}
        except Exception:
            return {}
        return out if isinstance(out, dict) else {}
    return dict(cap_limits)


def _evaluate_caps(
    cap_probe: Optional[Callable[[Sequence[Any], Dict[str, Any]], Dict[str, float]]],
    cap_limits: Optional[Union[Dict[str, float], Callable[..., Dict[str, float]]]],
    gate: Optional[GateResult],
    args: Sequence[Any],
    kwargs: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if cap_probe is None:
        return None

    limits = _resolve_cap_limits(cap_limits, gate=gate)
    if not limits:
        return None

    try:
        measured = cap_probe(args, kwargs)
    except Exception as e:
        return {"metric": "cap_probe_error", "value": None, "limit": None, "error": str(e)[:200]}
    if not isinstance(measured, dict):
        return {"metric": "cap_probe_invalid", "value": None, "limit": None, "error": "probe_not_dict"}

    for metric, limit in limits.items():
        if metric not in measured:
            continue
        try:
            value = float(measured.get(metric))
            lim = float(limit)
        except Exception:
            continue
        if abs(value) > lim:
            return {
                "metric": metric,
                "value": value,
                "limit": lim,
                "measured": measured,
            }
    return None


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> bool:
    tmp_path: Optional[Path] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(raw)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
        return True
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def _load_state(path: Path, fallback: Dict[str, Any], *, scope: str) -> Tuple[Dict[str, Any], bool]:
    current = str(path)
    if _STATE_FALLBACK_SCOPE.get(scope) != current:
        fallback.clear()
        _STATE_FALLBACK_SCOPE[scope] = current
    if not path.exists():
        return dict(fallback), False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            fallback.clear()
            fallback.update(payload)
            return payload, False
        return dict(fallback), True
    except Exception:
        return dict(fallback), True


def _store_state(path: Path, payload: Dict[str, Any], fallback: Dict[str, Any]) -> bool:
    fallback.clear()
    fallback.update(payload)
    return _atomic_write_json(path, payload)


def _critical_modes() -> set[str]:
    return {Mode.SURVIVAL.value, Mode.HIBERNATE.value, Mode.ROLLBACK.value, Mode.STABILIZE.value}


def _rollout_mode_effective() -> Tuple[str, Dict[str, Any]]:
    configured_env = os.getenv("CORTEX_GATE_ROLLOUT_MODE", "shadow_then_enforce").strip().lower()
    if configured_env not in {"enforce", "shadow", "shadow_then_enforce"}:
        configured_env = "shadow_then_enforce"
    shadow_secs = int(float(os.getenv("CORTEX_GATE_SHADOW_SECS", "86400")))
    now = _now_utc()
    control_path = _rollout_control_path()
    configured = configured_env
    source = "env"
    control_reason = None
    control_expires = None
    if control_path.exists():
        try:
            control_payload = json.loads(control_path.read_text(encoding="utf-8"))
            if isinstance(control_payload, dict):
                override = str(control_payload.get("override_mode") or "").strip().lower()
                expires = _parse_iso_ts(control_payload.get("expires_ts"))
                if override in {"enforce", "shadow", "shadow_then_enforce"}:
                    if expires is None or now < expires:
                        configured = override
                        source = "control_file"
                        control_reason = control_payload.get("reason")
                        control_expires = expires.isoformat() if expires else None
        except Exception:
            pass

    path = _rollout_state_path()
    state, io_degraded = _load_state(path, _ROLLOUT_STATE_FALLBACK, scope="rollout")
    started = _parse_iso_ts(state.get("rollout_started_ts"))
    if started is None:
        started = now
    effective = configured
    if configured == "shadow_then_enforce":
        elapsed = (now - started).total_seconds()
        effective = "shadow" if elapsed < max(1, shadow_secs) else "enforce"
    prev_effective = str(state.get("rollout_mode_effective") or "")
    transition = None
    if prev_effective and prev_effective != effective:
        transition = f"{prev_effective}->{effective}"
    new_state = {
        "rollout_started_ts": started.isoformat(),
        "rollout_mode_configured": configured,
        "rollout_mode_source": source,
        "rollout_mode_effective": effective,
        "shadow_secs": shadow_secs,
        "last_updated_ts": now.isoformat(),
    }
    write_ok = _store_state(path, new_state, _ROLLOUT_STATE_FALLBACK)
    ext = {
        "rollout_mode_configured": configured,
        "rollout_mode_source": source,
        "rollout_mode_effective": effective,
        "shadow_secs": shadow_secs,
        "rollout_started_ts": started.isoformat(),
        "rollout_transition": transition,
        "rollout_state_path": str(path),
        "rollout_control_path": str(control_path),
        "rollout_control_reason": control_reason,
        "rollout_control_expires_ts": control_expires,
    }
    if io_degraded or (not write_ok):
        ext["state_io_degraded"] = True
    return effective, ext


def _update_gate_mode_state(mode: str) -> Dict[str, Any]:
    now = _now_utc()
    path = _gate_state_path()
    state, io_degraded = _load_state(path, _GATE_STATE_FALLBACK, scope="gate")
    prev_mode = str(state.get("last_mode") or "")
    critical_modes = _critical_modes()
    if mode in critical_modes:
        state["last_critical_ts"] = now.isoformat()
    elif prev_mode in critical_modes:
        state["last_noncritical_ts"] = now.isoformat()
    state["last_mode"] = mode
    state["last_updated_ts"] = now.isoformat()
    write_ok = _store_state(path, state, _GATE_STATE_FALLBACK)
    out = dict(state)
    out["gate_state_path"] = str(path)
    if io_degraded or (not write_ok):
        out["state_io_degraded"] = True
    return out


def _cooldown_active(gate_state: Dict[str, Any], cooldown_sec: int, mode: str) -> Tuple[bool, Optional[float]]:
    if mode != Mode.ACT.value or cooldown_sec <= 0:
        return False, None
    exit_ts = _parse_iso_ts(gate_state.get("last_noncritical_ts"))
    crit_ts = _parse_iso_ts(gate_state.get("last_critical_ts"))
    if exit_ts is None:
        return False, None
    if crit_ts is not None and exit_ts < crit_ts:
        return False, None
    elapsed = (_now_utc() - exit_ts).total_seconds()
    return elapsed < float(cooldown_sec), elapsed


def _resolve_threshold(cortex: StateVectorCortex, metric: str, source: str) -> Optional[float]:
    key = f"{metric}_{source}".replace("-", "_")
    try:
        value = getattr(cortex, key)
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _recover_confirm(
    mode: str,
    debug: Dict[str, Any],
    metrics: Sequence[str],
    required: int,
    threshold_source: str,
) -> Dict[str, Any]:
    result = {
        "recover_confirm_passed": True,
        "recover_confirm_hits": 0,
        "recover_confirm_required": int(max(1, required)),
        "recover_confirm_not_applicable": False,
        "recover_confirm_detail": [],
    }
    if mode != Mode.ACT.value:
        return result
    state = debug.get("state")
    if not isinstance(state, dict):
        result["recover_confirm_not_applicable"] = True
        return result

    cortex = _get_cortex()
    hits = 0
    detail = []
    for m in metrics:
        metric = str(m).strip().upper()
        if metric not in {"R", "P", "O", "I", "T", "A"}:
            continue
        value_raw = state.get(metric)
        try:
            value = float(value_raw)
        except Exception:
            detail.append({"metric": metric, "value": value_raw, "threshold": None, "pass": False})
            continue
        threshold = _resolve_threshold(cortex, metric, threshold_source)
        if threshold is None:
            detail.append({"metric": metric, "value": value, "threshold": None, "pass": False})
            continue
        ok = value >= threshold
        if ok:
            hits += 1
        detail.append({"metric": metric, "value": value, "threshold": threshold, "pass": ok})
    req = int(max(1, required))
    result["recover_confirm_hits"] = hits
    result["recover_confirm_required"] = req
    result["recover_confirm_detail"] = detail
    result["recover_confirm_passed"] = hits >= req
    return result


def cortex_gated(
    *,
    action_class: str = "NORMAL_ACT",
    on_observe: str = "micro_probe",
    on_learn: str = "simulate",
    on_stabilize: str = "reject",
    mutate_micro_probe: Optional[
        Callable[[Sequence[Any], Dict[str, Any], GateResult], Tuple[Tuple[Any, ...], Dict[str, Any]]]
    ] = None,
    simulate_result: Optional[Callable[[str, str, Dict[str, Any]], Any]] = None,
    cap_probe: Optional[Callable[[Sequence[Any], Dict[str, Any]], Dict[str, float]]] = None,
    cap_limits: Optional[Union[Dict[str, float], Callable[..., Dict[str, float]]]] = None,
    on_cap_violation: str = "reject",
    reflex_reduce_only_probe: Optional[Callable[[Sequence[Any], Dict[str, Any]], bool]] = None,
    cooldown_sec: int = 180,
    recover_confirm_required: int = 2,
    recover_confirm_metrics: Tuple[str, ...] = ("R", "P", "O"),
    recover_confirm_threshold_source: str = "warning",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for hard action gating.

    policy strings:
    - allow
    - simulate
    - micro_probe
    - reject
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                mode, debug = _get_cortex().eval_irpota()
            except Exception as e:
                mode = Mode.STABILIZE.value
                debug = {"reason": "cortex_eval_error", "error": str(e)[:200]}

            policy = _default_policy(mode, on_observe=on_observe, on_learn=on_learn, on_stabilize=on_stabilize)
            reduce_only_candidate = False
            if reflex_reduce_only_probe is not None:
                try:
                    reduce_only_candidate = bool(reflex_reduce_only_probe(args, kwargs))
                except Exception:
                    reduce_only_candidate = False
            if mode in _critical_modes() and reduce_only_candidate:
                policy = "allow"
                if isinstance(debug, dict):
                    debug = dict(debug)
                    debug.setdefault("trigger", "reflex_reduce_only_allowlist")
                    debug.setdefault("reason", "reflex_reduce_only_allowlist")

            rollout_effective, rollout_ext = _rollout_mode_effective()
            gate_state = _update_gate_mode_state(mode)
            cooldown_active, cooldown_elapsed = _cooldown_active(gate_state, cooldown_sec=int(cooldown_sec), mode=mode)
            recover_ext = _recover_confirm(
                mode=mode,
                debug=debug if isinstance(debug, dict) else {},
                metrics=recover_confirm_metrics,
                required=int(max(1, recover_confirm_required)),
                threshold_source=str(recover_confirm_threshold_source or "warning"),
            )
            recover_ok = bool(recover_ext.get("recover_confirm_passed", True))
            soft_block_reason = None
            if cooldown_active:
                soft_block_reason = "cooldown_guard"
            elif not recover_ok and mode == Mode.ACT.value:
                soft_block_reason = "recover_confirm"

            gate_extensions: Dict[str, Any] = {}
            gate_extensions.update(rollout_ext)
            gate_extensions.update(
                {
                    "cooldown_sec": int(cooldown_sec),
                    "cooldown_active": bool(cooldown_active),
                    "cooldown_elapsed_sec": cooldown_elapsed,
                    "recover_confirm_metrics": list(recover_confirm_metrics),
                    "recover_confirm_threshold_source": str(recover_confirm_threshold_source or "warning"),
                    "gate_state_path": gate_state.get("gate_state_path"),
                    "last_critical_ts": gate_state.get("last_critical_ts"),
                    "last_noncritical_ts": gate_state.get("last_noncritical_ts"),
                    "last_mode": gate_state.get("last_mode"),
                }
            )
            gate_extensions.update(recover_ext)
            if gate_state.get("state_io_degraded"):
                gate_extensions["state_io_degraded"] = True
            if soft_block_reason:
                gate_extensions["soft_block_reason"] = soft_block_reason

            if isinstance(debug, dict):
                debug = dict(debug)
            else:
                debug = {"reason": "non_dict_debug"}
            state_for_probe = debug.get("state", {}) if isinstance(debug.get("state", {}), dict) else {}
            hard_probe = probe_hard_invariants(state_for_probe)
            gate_extensions["hard_invariant_probe"] = hard_probe
            if (
                str(os.getenv("CORTEX_HARD_PROBE_ENFORCE", "0")).strip().lower() in {"1", "true", "yes", "on"}
                and hard_probe.get("status") == "fail"
                and soft_block_reason is None
            ):
                soft_block_reason = "hard_invariant_probe"
                gate_extensions["soft_block_reason"] = soft_block_reason

            if soft_block_reason:
                fallback_policy = on_observe if on_observe in {"simulate", "micro_probe", "reject"} else "reject"
                if rollout_effective == "enforce":
                    policy = fallback_policy
                    debug["trigger"] = soft_block_reason
                    debug["reason"] = "gate_extension_blocked"
                else:
                    gate_extensions["would_block"] = True

            actuator_bus, reflex_reduce_only, bus_reason = resolve_actuator_bus(
                mode=mode,
                policy=policy,
                reduce_only_candidate=reduce_only_candidate,
            )
            certificate = issue_certificate(
                mode=mode,
                policy=policy,
                action_class=action_class,
                actuator_bus=actuator_bus,
                reduce_only=reflex_reduce_only,
                reason=str(debug.get("trigger") or debug.get("reason") or "policy"),
                metadata={
                    "rollout_mode_effective": rollout_effective,
                    "critical_mode": mode in _critical_modes(),
                    "reduce_only_candidate": reduce_only_candidate,
                    "bus_reason": bus_reason,
                },
            )
            gate_extensions["actuator_bus"] = actuator_bus
            gate_extensions["actuator_bus_reason"] = bus_reason
            gate_extensions["reflex_reduce_only"] = bool(reflex_reduce_only)
            gate_extensions["certificate"] = certificate.as_dict()
            debug["gate_extensions"] = gate_extensions

            gate = GateResult(
                mode=mode,
                policy=policy,
                action_class=action_class,
                reason=str(debug.get("trigger") or debug.get("reason") or "policy"),
                debug=debug,
            )

            if gate.policy == "allow":
                cap_violation = _evaluate_caps(cap_probe, cap_limits, gate, args, kwargs)
                if cap_violation is not None:
                    d = dict(gate.debug or {})
                    d["cap_violation"] = cap_violation
                    g = GateResult(
                        mode=gate.mode,
                        policy=gate.policy,
                        action_class=gate.action_class,
                        reason=f"cap_violation:{cap_violation.get('metric')}",
                        debug=d,
                    )
                    cap_shadow = rollout_effective == "shadow" and mode == Mode.ACT.value
                    if on_cap_violation == "simulate" or cap_shadow:
                        if cap_shadow:
                            d["gate_extensions"] = dict(d.get("gate_extensions") or {})
                            d["gate_extensions"]["would_block"] = True
                            d["gate_extensions"]["soft_block_reason"] = "cap_violation_shadow"
                        _append_gate_audit(_build_envelope(fn.__name__, g, outcome="cap_simulated"))
                        if simulate_result is not None:
                            return simulate_result(g.mode, g.reason, g.debug)
                        return _default_simulate_payload(fn.__name__, g.mode, g.action_class, g.reason, g.debug)
                    _append_gate_audit(_build_envelope(fn.__name__, g, outcome="cap_rejected"))
                    raise SpinalReflexReject(g)

                notional = _probe_notional(cap_probe, args, kwargs)
                vector, vector_meta = DoomsdayVectorStore().load()
                doom_guard = evaluate_doomsday_guard(
                    notional=notional,
                    vector=vector,
                    certificate=certificate,
                )
                gate.debug.setdefault("gate_extensions", {})
                gate.debug["gate_extensions"]["doomsday_guard"] = doom_guard
                gate.debug["gate_extensions"]["doomsday_vector_meta"] = vector_meta
                if doom_guard.get("hard_block"):
                    d = dict(gate.debug or {})
                    d["doomsday_guard"] = doom_guard
                    g = GateResult(
                        mode=gate.mode,
                        policy=gate.policy,
                        action_class=gate.action_class,
                        reason="doomsday_guard",
                        debug=d,
                    )
                    doom_shadow = rollout_effective == "shadow" and mode == Mode.ACT.value
                    if doom_shadow:
                        d["gate_extensions"] = dict(d.get("gate_extensions") or {})
                        d["gate_extensions"]["would_block"] = True
                        d["gate_extensions"]["soft_block_reason"] = "doomsday_guard_shadow"
                        _append_gate_audit(_build_envelope(fn.__name__, g, outcome="doomsday_shadow"))
                        if simulate_result is not None:
                            return simulate_result(g.mode, g.reason, g.debug)
                        return _default_simulate_payload(fn.__name__, g.mode, g.action_class, g.reason, g.debug)
                    _append_gate_audit(_build_envelope(fn.__name__, g, outcome="doomsday_rejected"))
                    raise SpinalReflexReject(g)

                _append_gate_audit(_build_envelope(fn.__name__, gate, outcome="executed"))
                return fn(*args, **kwargs)

            if gate.policy == "simulate":
                _append_gate_audit(_build_envelope(fn.__name__, gate, outcome="simulated"))
                if simulate_result is not None:
                    return simulate_result(gate.mode, gate.reason, gate.debug)
                return _default_simulate_payload(fn.__name__, gate.mode, gate.action_class, gate.reason, gate.debug)

            if gate.policy == "micro_probe":
                new_args = tuple(args)
                new_kwargs = dict(kwargs)
                if mutate_micro_probe is not None:
                    new_args, new_kwargs = mutate_micro_probe(args, kwargs, gate)
                cap_violation = _evaluate_caps(cap_probe, cap_limits, gate, new_args, new_kwargs)
                if cap_violation is not None:
                    d = dict(gate.debug or {})
                    d["cap_violation"] = cap_violation
                    g = GateResult(
                        mode=gate.mode,
                        policy=gate.policy,
                        action_class=gate.action_class,
                        reason=f"cap_violation:{cap_violation.get('metric')}",
                        debug=d,
                    )
                    cap_shadow = rollout_effective == "shadow" and mode == Mode.ACT.value
                    if on_cap_violation == "simulate" or cap_shadow:
                        if cap_shadow:
                            d["gate_extensions"] = dict(d.get("gate_extensions") or {})
                            d["gate_extensions"]["would_block"] = True
                            d["gate_extensions"]["soft_block_reason"] = "cap_violation_shadow"
                        _append_gate_audit(_build_envelope(fn.__name__, g, outcome="cap_simulated"))
                        if simulate_result is not None:
                            return simulate_result(g.mode, g.reason, g.debug)
                        return _default_simulate_payload(fn.__name__, g.mode, g.action_class, g.reason, g.debug)
                    _append_gate_audit(_build_envelope(fn.__name__, g, outcome="cap_rejected"))
                    raise SpinalReflexReject(g)

                probe_notional = _probe_notional(cap_probe, new_args, new_kwargs)
                vector, vector_meta = DoomsdayVectorStore().load()
                doom_guard = evaluate_doomsday_guard(
                    notional=probe_notional,
                    vector=vector,
                    certificate=certificate,
                )
                gate.debug.setdefault("gate_extensions", {})
                gate.debug["gate_extensions"]["doomsday_guard"] = doom_guard
                gate.debug["gate_extensions"]["doomsday_vector_meta"] = vector_meta
                budget = ProbeBudgetGuard().inspect(
                    notional=probe_notional,
                    projected_loss=float(doom_guard.get("projected_loss", 0.0)),
                    certificate=certificate,
                )
                gate.debug["gate_extensions"]["probe_budget"] = budget
                if budget.get("would_block") and bool(budget.get("enforce")):
                    d = dict(gate.debug or {})
                    g = GateResult(
                        mode=gate.mode,
                        policy=gate.policy,
                        action_class=gate.action_class,
                        reason=str(budget.get("reason") or "probe_budget_blocked"),
                        debug=d,
                    )
                    _append_gate_audit(_build_envelope(fn.__name__, g, outcome="probe_budget_rejected"))
                    raise SpinalReflexReject(g)

                _append_gate_audit(_build_envelope(fn.__name__, gate, outcome="micro_probe"))
                return fn(*new_args, **new_kwargs)

            _append_gate_audit(_build_envelope(fn.__name__, gate, outcome="rejected"))
            raise SpinalReflexReject(gate)

        return wrapper

    return deco
