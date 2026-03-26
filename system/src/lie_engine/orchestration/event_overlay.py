from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json


FAIL_CLOSED_PAYLOAD: dict[str, Any] = {
    "risk_multiplier_override": 0.0,
    "canary_freeze": True,
    "override_reason_codes": ["event_overlay_fail_closed"],
}


def _clamp_risk_multiplier(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_canary_freeze(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError("invalid canary_freeze value")


def _is_payload_expired(payload: dict[str, Any]) -> bool:
    now = datetime.now(timezone.utc)
    for key in ("valid_until_utc", "valid_until"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            ts = datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return True
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < now:
            return True
    return False


def load_event_live_guard_overlay(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return dict(FAIL_CLOSED_PAYLOAD)
    try:
        raw = target.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return dict(FAIL_CLOSED_PAYLOAD)
    if not isinstance(payload, dict):
        return dict(FAIL_CLOSED_PAYLOAD)
    if _is_payload_expired(payload):
        return dict(FAIL_CLOSED_PAYLOAD)
    risk_override = payload.get("risk_multiplier_override")
    if isinstance(risk_override, bool):
        return dict(FAIL_CLOSED_PAYLOAD)
    if risk_override is None:
        risk_override = 1.0
    else:
        try:
            risk_override = float(risk_override)
        except (TypeError, ValueError):
            return dict(FAIL_CLOSED_PAYLOAD)
    try:
        canary_freeze = _parse_canary_freeze(payload.get("canary_freeze"))
    except ValueError:
        return dict(FAIL_CLOSED_PAYLOAD)
    raw_reasons = payload.get("override_reason_codes", [])
    reasons = [entry for entry in raw_reasons if isinstance(entry, str)] if isinstance(raw_reasons, list) else []
    result: dict[str, Any] = {
        "risk_multiplier_override": _clamp_risk_multiplier(risk_override),
        "canary_freeze": canary_freeze,
        "override_reason_codes": reasons,
    }
    return result
