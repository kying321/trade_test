#!/usr/bin/env python3
"""Signal registry loader/validator for Pi.

Provides a single contract source for:
- signal metadata validity
- cortex threshold overrides
- runtime limit defaults
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_REGISTRY_PATH = ROOT / "config" / "signal_registry.json"

REQUIRED_SIGNAL_FIELDS = [
    "signal_id",
    "domain",
    "kind",
    "units",
    "value_type",
    "sample_period",
    "trust_tier",
    "cost_class",
    "maps_to",
    "validator",
    "failure_action",
    "audit_tags",
]

ALLOWED_DOMAIN = {"intero", "extero"}
ALLOWED_KIND = {"homeostatic", "predictive", "event"}
ALLOWED_TRUST = {"T0", "T1", "T2", "T3"}
ALLOWED_COST = {"low", "med", "high"}
ALLOWED_VALUE_TYPE = {"float", "int", "bool", "enum", "json"}
ALLOWED_FAILURE_ACTION = {"NO-OP", "OBSERVE", "STABILIZE", "HIBERNATE", "ROLLBACK", "SURVIVAL"}
ALLOWED_MAPS_TO = {"I", "R", "P", "O", "T", "A"}


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def resolve_signal_registry_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()

    env = str(os.getenv("SIGNAL_REGISTRY_PATH", "")).strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_SIGNAL_REGISTRY_PATH


def load_signal_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    p = resolve_signal_registry_path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def validate_signal_registry(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "signal_count": 0,
            "valid_signal_count": 0,
            "errors": ["payload_not_dict"],
        }

    signals = payload.get("signals")
    if not isinstance(signals, list):
        return {
            "ok": False,
            "signal_count": 0,
            "valid_signal_count": 0,
            "errors": ["signals_not_list"],
        }

    valid_signal_count = 0
    for i, sig in enumerate(signals):
        if not isinstance(sig, dict):
            errors.append(f"signals[{i}]_not_dict")
            continue

        miss = [k for k in REQUIRED_SIGNAL_FIELDS if k not in sig]
        if miss:
            errors.append(f"signals[{i}]_missing:{','.join(miss)}")
            continue

        bad: List[str] = []
        if str(sig.get("domain")) not in ALLOWED_DOMAIN:
            bad.append("domain")
        if str(sig.get("kind")) not in ALLOWED_KIND:
            bad.append("kind")
        if str(sig.get("value_type")) not in ALLOWED_VALUE_TYPE:
            bad.append("value_type")
        if str(sig.get("trust_tier")) not in ALLOWED_TRUST:
            bad.append("trust_tier")
        if str(sig.get("cost_class")) not in ALLOWED_COST:
            bad.append("cost_class")
        if str(sig.get("failure_action")) not in ALLOWED_FAILURE_ACTION:
            bad.append("failure_action")

        maps_to = sig.get("maps_to")
        if not isinstance(maps_to, list) or not maps_to:
            bad.append("maps_to")
        else:
            for x in maps_to:
                if str(x) not in ALLOWED_MAPS_TO:
                    bad.append("maps_to")
                    break

        if bad:
            errors.append(f"signals[{i}]_invalid:{','.join(sorted(set(bad)))}")
            continue

        valid_signal_count += 1

    return {
        "ok": len(errors) == 0,
        "signal_count": len(signals),
        "valid_signal_count": valid_signal_count,
        "errors": errors,
    }


def extract_cortex_threshold_overrides(payload: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    thresholds = payload.get("thresholds") if isinstance(payload, dict) else None
    if not isinstance(thresholds, dict):
        return out

    def put_if_num(key: str, val: Any) -> None:
        f = _safe_float(val)
        if f is None:
            return
        out[key] = f

    for k in ["I", "R", "P", "O"]:
        blk = thresholds.get(k)
        if not isinstance(blk, dict):
            continue
        put_if_num(f"{k}_critical", blk.get("critical"))
        put_if_num(f"{k}_warning", blk.get("warning"))

    pblk = thresholds.get("P")
    if isinstance(pblk, dict):
        put_if_num("P_shock_threshold", pblk.get("shock_threshold"))

    return out


def extract_runtime_limits(payload: Dict[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    blk = payload.get("runtime_limits") if isinstance(payload, dict) else None
    if not isinstance(blk, dict):
        return out

    tq = _safe_float(blk.get("token_queue_hard_limit"))
    if tq is not None and tq > 0:
        out["token_queue_hard_limit"] = int(tq)

    budget = _safe_float(blk.get("context_token_budget"))
    if budget is not None and budget > 0:
        out["context_token_budget"] = int(budget)

    return out


def probe_signal_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    p = resolve_signal_registry_path(path)
    if not p.exists():
        return {
            "status": "degraded",
            "path": str(p),
            "exists": False,
            "reason": "registry_missing",
            "signal_count": 0,
            "valid_signal_count": 0,
            "errors": ["registry_missing"],
        }

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status": "degraded",
            "path": str(p),
            "exists": True,
            "reason": f"parse_error:{type(e).__name__}",
            "signal_count": 0,
            "valid_signal_count": 0,
            "errors": ["registry_parse_error"],
        }

    validation = validate_signal_registry(payload if isinstance(payload, dict) else {})
    status = "ok" if bool(validation.get("ok")) else "degraded"
    return {
        "status": status,
        "path": str(p),
        "exists": True,
        "version": payload.get("version") if isinstance(payload, dict) else None,
        "signal_count": int(validation.get("signal_count") or 0),
        "valid_signal_count": int(validation.get("valid_signal_count") or 0),
        "errors": list(validation.get("errors") or [])[:20],
    }


__all__ = [
    "resolve_signal_registry_path",
    "load_signal_registry",
    "validate_signal_registry",
    "extract_cortex_threshold_overrides",
    "extract_runtime_limits",
    "probe_signal_registry",
]
