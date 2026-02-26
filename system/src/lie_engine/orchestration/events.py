from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_TRACEPARENT_RE = re.compile(r"^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return "{}"


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _normalize_hex_trace_id(raw: str | None) -> str:
    text = str(raw or "").strip().lower()
    if _HEX32_RE.match(text):
        return text
    if not text:
        return ""
    return _sha256_hex(text)[:32]


def _normalize_hex_span_id(raw: str | None) -> str:
    text = str(raw or "").strip().lower()
    if _HEX16_RE.match(text):
        return text
    if not text:
        return ""
    return _sha256_hex(text)[:16]


def normalize_traceparent(raw: str | None) -> str:
    text = str(raw or "").strip().lower()
    if _TRACEPARENT_RE.match(text):
        return text
    return ""


def trace_id_from_traceparent(raw: str | None) -> str:
    tp = normalize_traceparent(raw)
    if not tp:
        return ""
    parts = tp.split("-")
    if len(parts) != 4:
        return ""
    return str(parts[1])


def build_traceparent(
    *,
    trace_id: str,
    span_id: str | None = None,
    sampled: bool = True,
    version: str = "00",
) -> str:
    trace_hex = _normalize_hex_trace_id(trace_id)
    if not trace_hex:
        trace_hex = _sha256_hex(uuid4().hex)[:32]
    span_hex = _normalize_hex_span_id(span_id)
    if not span_hex:
        span_hex = _sha256_hex(uuid4().hex)[:16]
    flags = "01" if sampled else "00"
    ver = str(version or "00").strip().lower()
    if not re.match(r"^[0-9a-f]{2}$", ver):
        ver = "00"
    return f"{ver}-{trace_hex}-{span_hex}-{flags}"


def derive_trace_id(*, source: str, as_of: date | str | None = None, seed: str | None = None) -> str:
    day = as_of.isoformat() if isinstance(as_of, date) else str(as_of or "")
    nonce = str(seed or uuid4().hex)
    return _sha256_hex(f"{source}|{day}|{nonce}")[:32]


@dataclass(slots=True)
class EventEnvelope:
    event_id: str
    trace_id: str
    traceparent: str
    event_ts: str
    as_of: str
    source: str
    event_type: str
    payload_hash: str
    parent_event_id: str = ""
    schema_version: str = "v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "traceparent": self.traceparent,
            "parent_event_id": self.parent_event_id,
            "event_ts": self.event_ts,
            "as_of": self.as_of,
            "source": self.source,
            "event_type": self.event_type,
            "payload_hash": self.payload_hash,
        }


def build_event_envelope(
    *,
    source: str,
    event_type: str,
    payload: dict[str, Any] | None,
    as_of: date | str | None = None,
    trace_id: str | None = None,
    traceparent: str | None = None,
    parent_event_id: str | None = None,
    event_ts: str | None = None,
) -> EventEnvelope:
    ts = str(event_ts or _utc_now_iso())
    as_of_txt = as_of.isoformat() if isinstance(as_of, date) else str(as_of or "")
    canonical = _canonical_json(payload or {})
    payload_hash = _sha256_hex(canonical)
    provided_traceparent = normalize_traceparent(traceparent)
    trace_from_parent = trace_id_from_traceparent(provided_traceparent)
    resolved_trace_id = str(trace_id or trace_from_parent or derive_trace_id(source=source, as_of=as_of_txt))
    trace_id_for_parent = _normalize_hex_trace_id(resolved_trace_id)
    if not trace_id_for_parent:
        trace_id_for_parent = _sha256_hex(f"{source}|{event_type}|{ts}")[:32]
    resolved_traceparent = provided_traceparent or build_traceparent(
        trace_id=trace_id_for_parent,
        span_id=str(parent_event_id or "") or _sha256_hex(f"{source}|{event_type}|{ts}")[:16],
        sampled=True,
    )
    event_id = _sha256_hex(
        f"{resolved_trace_id}|{source}|{event_type}|{ts}|{payload_hash}|{str(parent_event_id or '')}"
    )[:32]
    return EventEnvelope(
        event_id=event_id,
        trace_id=resolved_trace_id,
        traceparent=resolved_traceparent,
        parent_event_id=str(parent_event_id or ""),
        event_ts=ts,
        as_of=as_of_txt,
        source=str(source),
        event_type=str(event_type),
        payload_hash=payload_hash,
    )


def append_event_envelope(
    *,
    output_dir: Path,
    envelope: EventEnvelope,
    payload: dict[str, Any] | None = None,
) -> Path:
    as_of_day = str(envelope.as_of or "")[:10]
    if len(as_of_day) != 10:
        as_of_day = str(envelope.event_ts or "")[:10]
    if len(as_of_day) != 10:
        as_of_day = datetime.now(timezone.utc).date().isoformat()
    path = output_dir / "logs" / "event_stream" / f"{as_of_day}_events.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "envelope": envelope.to_dict(),
        "payload": payload or {},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path
