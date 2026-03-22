#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any


RAW_FIELD_DENYLIST = {
    "raw_transcript",
    "message_text",
    "messages",
    "raw_messages",
    "transcript",
}
DEFAULT_ALIGNMENT_BASE = 50.0
DEFAULT_EXECUTION_BASE = 50.0


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return utc_now()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    system_root = Path(__file__).resolve().parents[1]
    review_default = system_root / "output" / "review"
    parser = argparse.ArgumentParser(description="Build internal conversation feedback projection artifact.")
    parser.add_argument("--review-dir", type=Path, default=review_default)
    parser.add_argument("--manual-events-path", type=Path)
    parser.add_argument("--auto-publish-path", type=Path)
    parser.add_argument("--now", help="Explicit UTC timestamp for deterministic tests.")
    return parser.parse_args()


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): sanitize_value(child)
            for key, child in value.items()
            if str(key) not in RAW_FIELD_DENYLIST
        }
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def parse_created_at(value: Any, fallback: dt.datetime) -> dt.datetime:
    text = safe_text(value)
    if not text:
        return fallback
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def normalize_anchors(value: Any) -> list[dict[str, str]]:
    rows = value if isinstance(value, list) else [value]
    anchors: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        anchor = {
            "route": safe_text(row.get("route")),
            "artifact": safe_text(row.get("artifact")),
            "component": safe_text(row.get("component")),
        }
        if not any(anchor.values()):
            continue
        key = (anchor["route"], anchor["artifact"], anchor["component"])
        if key in seen:
            continue
        seen.add(key)
        anchors.append(anchor)
    return anchors


def normalize_event(row: dict[str, Any], *, fallback_source: str, fallback_now: dt.datetime) -> dict[str, Any]:
    sanitized = sanitize_value(row)
    created_at = parse_created_at(sanitized.get("created_at_utc"), fallback_now)
    return {
        "feedback_id": safe_text(sanitized.get("feedback_id")) or f"{fallback_source}_{created_at.timestamp():.0f}",
        "created_at_utc": fmt_utc(created_at),
        "source": safe_text(sanitized.get("source")) or fallback_source,
        "domain": safe_text(sanitized.get("domain")) or "global",
        "headline": safe_text(sanitized.get("headline")) or "未命名反馈",
        "summary": safe_text(sanitized.get("summary")) or "未提供摘要",
        "recommended_action": safe_text(sanitized.get("recommended_action")) or "保持观察",
        "alignment_delta": to_float(sanitized.get("alignment_delta")),
        "blocker_delta": to_float(sanitized.get("blocker_delta")),
        "execution_delta": to_float(sanitized.get("execution_delta")),
        "readability_delta": to_float(sanitized.get("readability_delta")),
        "impact_score": clamp(to_float(sanitized.get("impact_score"), 0.0)),
        "confidence": clamp(to_float(sanitized.get("confidence"), 0.0), 0.0, 1.0),
        "status": safe_text(sanitized.get("status")) or "active",
        "anchors": normalize_anchors(sanitized.get("anchors")),
    }


def merge_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {"manual": 2, "auto_session": 1}
    merged: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["created_at_utc"]):
        current = merged.get(row["feedback_id"])
        if current is None:
            merged[row["feedback_id"]] = row
            continue
        current_priority = priority.get(safe_text(current.get("source")), 0)
        next_priority = priority.get(safe_text(row.get("source")), 0)
        if next_priority > current_priority:
            merged[row["feedback_id"]] = row
            continue
        if next_priority == current_priority and row["created_at_utc"] >= safe_text(current.get("created_at_utc")):
            merged[row["feedback_id"]] = row
    return sorted(merged.values(), key=lambda item: item["created_at_utc"])


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    alignment = DEFAULT_ALIGNMENT_BASE + sum(to_float(row.get("alignment_delta")) for row in rows)
    blocker = sum(to_float(row.get("blocker_delta")) for row in rows)
    execution = DEFAULT_EXECUTION_BASE + sum(to_float(row.get("execution_delta")) for row in rows)
    readability = sum(to_float(row.get("readability_delta")) for row in rows)
    alignment_score = round(clamp(alignment), 2)
    blocker_pressure = round(clamp(blocker), 2)
    execution_clarity = round(clamp(execution), 2)
    readability_pressure = round(clamp(readability), 2)
    if alignment_score >= 70 and blocker_pressure <= 35 and readability_pressure <= 35:
        drift_state = "aligned"
    elif alignment_score < 45 or blocker_pressure >= 60 or readability_pressure >= 60:
        drift_state = "drifting"
    else:
        drift_state = "watch"
    return {
        "alignment_score": alignment_score,
        "blocker_pressure": blocker_pressure,
        "execution_clarity": execution_clarity,
        "readability_pressure": readability_pressure,
        "drift_state": drift_state,
    }


def build_actions(active_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dedup: dict[str, dict[str, Any]] = {}
    for row in sorted(active_events, key=lambda item: (-to_float(item.get("impact_score")), item["created_at_utc"]), reverse=False):
        action = safe_text(row.get("recommended_action"))
        if not action:
            continue
        current = dedup.get(action)
        if current and to_float(current.get("impact_score")) >= to_float(row.get("impact_score")):
            continue
        dedup[action] = {
            "feedback_id": row["feedback_id"],
            "recommended_action": action,
            "impact_score": round(clamp(to_float(row.get("impact_score"))), 2),
            "anchors": row.get("anchors") or [],
        }
    return sorted(dedup.values(), key=lambda item: (-to_float(item.get("impact_score")), safe_text(item.get("recommended_action"))))[:5]


def build_trends(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trends: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []
    for row in rows:
        running.append(row)
        metrics = aggregate_metrics(running)
        trends.append(
            {
                "feedback_id": row["feedback_id"],
                "created_at_utc": row["created_at_utc"],
                **metrics,
            }
        )
    return trends[-20:]


def build_projection(
    *,
    review_dir: Path,
    runtime_now: dt.datetime,
    manual_path: Path | None = None,
    auto_path: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    manual_path = (manual_path or (review_dir / "conversation_feedback_events_internal.jsonl")).expanduser().resolve()
    auto_path = (auto_path or (review_dir / "latest_conversation_feedback_autopublish_internal.json")).expanduser().resolve()

    raw_events: list[dict[str, Any]] = []
    raw_events.extend(
        normalize_event(row, fallback_source="manual", fallback_now=runtime_now)
        for row in read_jsonl(manual_path)
    )
    if auto_path.exists():
        auto_payload = read_json(auto_path)
        auto_rows = auto_payload if isinstance(auto_payload, list) else [auto_payload]
        raw_events.extend(
            normalize_event(row, fallback_source="auto_session", fallback_now=runtime_now)
            for row in auto_rows
            if isinstance(row, dict)
        )

    merged_events = merge_events(raw_events)
    active_events = [row for row in merged_events if safe_text(row.get("status")) == "active"]
    active_events = sorted(active_events, key=lambda item: (-to_float(item.get("impact_score")), item["created_at_utc"]))[:20]
    trend_rows = build_trends(merged_events)
    summary_metrics = aggregate_metrics(active_events)
    headline_row = active_events[0] if active_events else (merged_events[-1] if merged_events else None)
    anchors = normalize_anchors([anchor for row in active_events for anchor in row.get("anchors") or []])

    payload = {
        "action": "build_conversation_feedback_projection_internal",
        "ok": True,
        "status": "ok" if active_events else "empty",
        "change_class": "RESEARCH_ONLY",
        "visibility": "internal_only",
        "generated_at_utc": fmt_utc(runtime_now),
        "summary": {
            **summary_metrics,
            "headline": safe_text(headline_row.get("headline") if headline_row else "暂无高价值反馈"),
        },
        "events": active_events,
        "actions": build_actions(active_events),
        "trends": trend_rows,
        "anchors": anchors,
    }

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_conversation_feedback_projection_internal.json"
    latest_path = review_dir / "latest_conversation_feedback_projection_internal.json"
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(artifact_path, latest_path)
    return payload, artifact_path, latest_path


def main() -> int:
    args = parse_args()
    review_dir = args.review_dir.expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = parse_now(args.now)
    payload, artifact_path, latest_path = build_projection(
        review_dir=review_dir,
        runtime_now=runtime_now,
        manual_path=args.manual_events_path,
        auto_path=args.auto_publish_path,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": payload["status"],
                "artifact": str(artifact_path),
                "latest_artifact": str(latest_path),
                "events_count": len(payload["events"]),
                "actions_count": len(payload["actions"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
