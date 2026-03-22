from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/publish_conversation_feedback_manual_internal.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "publish_conversation_feedback_manual_internal_script",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def test_publish_feedback_manual_appends_sanitized_jsonl_rows(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    payload = {
        "feedback_id": "manual_alignment_fix",
        "domain": "research",
        "headline": "人工结构化反馈：补齐 manual 轨",
        "summary": "manual 轨需要标准写入口，避免直接手改 jsonl。",
        "recommended_action": "提供 publish-manual 与 publish-manual-refresh。",
        "alignment_delta": 9,
        "blocker_delta": 3,
        "execution_delta": 7,
        "readability_delta": 1,
        "impact_score": 88,
        "confidence": 0.84,
        "raw_transcript": "must not persist",
        "messages": ["remove me"],
        "anchors": [
            {
                "route": "/workspace/alignment?view=internal",
                "artifact": "latest_conversation_feedback_projection_internal",
                "component": "AlignmentWorkspace",
            }
        ],
    }

    result = mod.publish_manual_payload(
        review_dir=review_dir,
        payload=payload,
        now_text="2026-03-22T06:40:00Z",
    )

    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    assert manual_path.exists()
    written = _read_jsonl(manual_path)
    assert len(written) == 1
    assert written[0]["feedback_id"] == "manual_alignment_fix"
    assert written[0]["source"] == "manual"
    assert written[0]["status"] == "active"
    assert written[0]["created_at_utc"] == "2026-03-22T06:40:00Z"
    assert "raw_transcript" not in written[0]
    assert "messages" not in written[0]
    assert result["path"] == str(manual_path)
    assert result["event_count"] == 1
    assert result["feedback_ids"] == ["manual_alignment_fix"]


def test_publish_feedback_manual_accepts_list_and_appends_without_overwriting(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    first_result = mod.publish_manual_payload(
        review_dir=review_dir,
        payload={
            "feedback_id": "first_manual",
            "headline": "first headline",
            "summary": "first summary",
            "recommended_action": "first action",
        },
        now_text="2026-03-22T06:41:00Z",
    )
    assert first_result["event_count"] == 1

    result = mod.publish_manual_payload(
        review_dir=review_dir,
        payload=[
            {
                "feedback_id": "second_manual",
                "headline": "second headline",
                "summary": "second summary",
                "recommended_action": "second action",
            },
            {
                "feedback_id": "third_manual",
                "headline": "third headline",
                "summary": "third summary",
                "recommended_action": "third action",
                "message_text": "remove me",
            },
        ],
        now_text="2026-03-22T06:42:00Z",
    )

    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    written = _read_jsonl(manual_path)
    assert [row["feedback_id"] for row in written] == ["first_manual", "second_manual", "third_manual"]
    assert written[0]["source"] == "manual"
    assert written[1]["created_at_utc"] == "2026-03-22T06:42:00Z"
    assert "message_text" not in written[2]
    assert result["event_count"] == 2
    assert result["feedback_ids"] == ["second_manual", "third_manual"]
