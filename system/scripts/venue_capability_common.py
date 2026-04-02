from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
VENUE_STATUS_ALLOWED = {"dry_only", "live_blocked", "live_ready", "unknown"}
CAPABILITY_STATUS_ALLOWED = {"ready", "blocked", "unknown"}
REQUIRED_CAPABILITY_FIELDS = (
    "spot_signed_read_status",
    "spot_signed_trade_status",
    "futures_signed_read_status",
    "futures_signed_trade_status",
)
REQUIRED_EXPECTED_STATUS_ALLOWED = {"ready", "blocked"}


def default_capability_artifact() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "venues": {}}


def _parse_utc(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def _normalize_now(now_utc: str | datetime | None) -> datetime:
    if isinstance(now_utc, str):
        parsed = _parse_utc(now_utc)
        if parsed is None:
            raise ValueError("invalid now_utc")
        return parsed
    elif isinstance(now_utc, datetime) and now_utc.tzinfo is not None:
        return now_utc.astimezone(timezone.utc)
    elif now_utc is not None:
        raise ValueError("invalid now_utc")
    return datetime.now(timezone.utc)


def read_venue_capability_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("venue capability artifact must be object")
    return payload


def _blocked(reason: str, venue: str, blockers: list[str] | None = None) -> dict[str, Any]:
    return {
        "live_route_status": "live_blocked",
        "live_route_reason": str(reason),
        "venue": str(venue),
        "venue_blockers": list(blockers or []),
    }


def _validate_root_schema(payload: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "venue_capability_incomplete"
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        return None, "venue_capability_incomplete"
    venues = payload.get("venues")
    if not isinstance(venues, dict):
        return None, "venue_capability_incomplete"
    return payload, None


def _validate_venue_payload(payload: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(payload, dict):
        return None, "venue_capability_incomplete"
    checked_at = payload.get("checked_at_utc")
    status = str(payload.get("status", "")).strip()
    blockers = payload.get("blockers", [])
    if checked_at is None or status not in VENUE_STATUS_ALLOWED or not isinstance(blockers, list):
        return None, "venue_capability_incomplete"
    for field in REQUIRED_CAPABILITY_FIELDS:
        value = str(payload.get(field, "")).strip()
        if value not in CAPABILITY_STATUS_ALLOWED:
            return None, "venue_capability_incomplete"
    return payload, None


def evaluate_live_route_for_requirements(
    path: str | Path,
    venue: str,
    required_statuses: Mapping[str, str],
    now_utc: str | datetime | None = None,
    max_age_seconds: int = 900,
) -> dict[str, Any]:
    artifact_path = Path(path)
    venue_name = str(venue).strip()
    if not artifact_path.exists():
        return _blocked("venue_capability_missing", venue_name)

    try:
        raw_payload = read_venue_capability_artifact(artifact_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return _blocked("venue_capability_incomplete", venue_name)

    payload, root_error = _validate_root_schema(raw_payload)
    if root_error is not None or payload is None:
        return _blocked(root_error or "venue_capability_incomplete", venue_name)

    venues = payload["venues"]
    venue_payload, venue_error = _validate_venue_payload(venues.get(venue_name))
    if venue_error is not None or venue_payload is None:
        return _blocked(venue_error or "venue_capability_incomplete", venue_name)

    try:
        now_value = _normalize_now(now_utc)
    except ValueError:
        return _blocked("venue_capability_incomplete", venue_name)
    checked_at = _parse_utc(venue_payload.get("checked_at_utc"))
    if checked_at is None:
        return _blocked("venue_capability_stale", venue_name, blockers=list(venue_payload.get("blockers", [])))
    if checked_at > now_value:
        return _blocked("venue_capability_stale", venue_name, blockers=list(venue_payload.get("blockers", [])))
    if (now_value - checked_at).total_seconds() > int(max_age_seconds):
        return _blocked("venue_capability_stale", venue_name, blockers=list(venue_payload.get("blockers", [])))

    venue_status = str(venue_payload["status"]).strip()
    blockers = list(venue_payload.get("blockers", []))
    if venue_status == "unknown":
        return _blocked("venue_capability_unknown", venue_name, blockers=blockers)
    if venue_status != "live_ready":
        return _blocked("venue_capability", venue_name, blockers=blockers)

    for key, expected in required_statuses.items():
        expected_value = str(expected).strip()
        if expected_value not in REQUIRED_EXPECTED_STATUS_ALLOWED:
            return _blocked("venue_capability_incomplete", venue_name, blockers=blockers)
        current_value = str(venue_payload.get(str(key), "")).strip()
        if current_value not in CAPABILITY_STATUS_ALLOWED:
            return _blocked("venue_capability_incomplete", venue_name, blockers=blockers)
        if current_value == "unknown":
            return _blocked("venue_capability_unknown", venue_name, blockers=blockers)
        if current_value != expected_value:
            return _blocked("venue_capability", venue_name, blockers=blockers)

    return {
        "live_route_status": "live_ready",
        "live_route_reason": None,
        "venue": venue_name,
        "venue_blockers": blockers,
    }
