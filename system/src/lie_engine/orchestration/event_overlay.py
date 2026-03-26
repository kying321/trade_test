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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _is_payload_expired(payload: dict[str, Any]) -> bool:
    for key in ("valid_until_utc", "valid_until"):
        valid_until = payload.get(key)
        if not valid_until:
            continue
        try:
            ts = datetime.fromisoformat(str(valid_until))
        except (ValueError, TypeError):
            return True
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return ts < now
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
    if risk_override is None:
        risk_override = 1.0
    else:
        try:
            risk_override = float(risk_override)
        except (TypeError, ValueError):
            return dict(FAIL_CLOSED_PAYLOAD)
    result: dict[str, Any] = {
        "risk_multiplier_override": _clamp_risk_multiplier(risk_override),
        "canary_freeze": _parse_bool(payload.get("canary_freeze")),
        "override_reason_codes": list(payload.get("override_reason_codes", []))
        if isinstance(payload.get("override_reason_codes"), list)
        else [],
    }
    return result
