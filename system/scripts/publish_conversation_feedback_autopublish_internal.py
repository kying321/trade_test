#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

RAW_FIELD_DENYLIST = {
    'raw_transcript',
    'message_text',
    'messages',
    'raw_messages',
    'transcript',
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or '').strip()
    if not text:
        return utc_now()
    parsed = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace('+00:00', 'Z')


def safe_text(value: Any) -> str:
    return str(value or '').strip()


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


def normalize_anchors(value: Any) -> list[dict[str, str]]:
    rows = value if isinstance(value, list) else [value]
    anchors: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        anchor = {
            'route': safe_text(row.get('route')),
            'artifact': safe_text(row.get('artifact')),
            'component': safe_text(row.get('component')),
        }
        if any(anchor.values()):
            anchors.append(anchor)
    return anchors


def normalize_event(row: dict[str, Any], *, now_text: str) -> dict[str, Any]:
    sanitized = sanitize_value(row)
    feedback_id = safe_text(sanitized.get('feedback_id')) or f'auto_session_{now_text}'
    event = {
        'feedback_id': feedback_id,
        'created_at_utc': safe_text(sanitized.get('created_at_utc')) or now_text,
        'source': safe_text(sanitized.get('source')) or 'auto_session',
        'domain': safe_text(sanitized.get('domain')) or 'global',
        'headline': safe_text(sanitized.get('headline')) or '未命名反馈',
        'summary': safe_text(sanitized.get('summary')) or '未提供摘要',
        'recommended_action': safe_text(sanitized.get('recommended_action')) or '保持观察',
        'status': safe_text(sanitized.get('status')) or 'active',
        'anchors': normalize_anchors(sanitized.get('anchors')),
    }
    for key in [
        'alignment_delta',
        'blocker_delta',
        'execution_delta',
        'readability_delta',
        'impact_score',
        'confidence',
    ]:
        if key in sanitized:
            event[key] = sanitized[key]
    return event


def publish_autopublish_payload(*, review_dir: Path, payload: Any, now_text: str | None = None) -> dict[str, Any]:
    review_dir.mkdir(parents=True, exist_ok=True)
    effective_now = fmt_utc(parse_now(now_text))
    rows = payload if isinstance(payload, list) else [payload]
    normalized = [normalize_event(row, now_text=effective_now) for row in rows if isinstance(row, dict)]
    if not normalized:
        raise ValueError('no_valid_feedback_rows')

    output_payload: Any = normalized if isinstance(payload, list) else normalized[0]
    latest_path = review_dir / 'latest_conversation_feedback_autopublish_internal.json'
    latest_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {
        'ok': True,
        'status': 'ok',
        'action': 'publish_conversation_feedback_autopublish_internal',
        'path': str(latest_path),
        'event_count': len(normalized),
        'feedback_ids': [row['feedback_id'] for row in normalized],
    }


def parse_args() -> argparse.Namespace:
    system_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Publish sanitized internal conversation feedback autopublish payload.')
    parser.add_argument('--review-dir', type=Path, default=system_root / 'output' / 'review')
    parser.add_argument('--payload-json', help='Inline JSON payload. Accepts object or array.')
    parser.add_argument('--payload-path', type=Path, help='Path to JSON payload file. Accepts object or array.')
    parser.add_argument('--now', help='Explicit UTC timestamp for deterministic writes.')
    return parser.parse_args()


def load_payload(args: argparse.Namespace) -> Any:
    if args.payload_json:
        return json.loads(args.payload_json)
    if args.payload_path:
        return json.loads(args.payload_path.read_text(encoding='utf-8'))
    raw = sys.stdin.read().strip()
    if not raw:
        raise SystemExit('missing_payload_json')
    return json.loads(raw)


def main() -> None:
    args = parse_args()
    payload = load_payload(args)
    result = publish_autopublish_payload(
        review_dir=args.review_dir,
        payload=payload,
        now_text=args.now,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
