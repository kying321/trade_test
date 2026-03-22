from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/publish_conversation_feedback_autopublish_internal.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "publish_conversation_feedback_autopublish_internal_script",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_publish_feedback_autopublish_sanitizes_and_writes_latest(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    payload = {
        "feedback_id": "alignment_projection_rollout",
        "domain": "research",
        "headline": "将高价值对话反馈投射到内部对齐页",
        "summary": "先验证 internal 对齐页，再收紧 fallback。",
        "recommended_action": "刷新 internal snapshot 并检查 alignment 页面。",
        "alignment_delta": 12,
        "blocker_delta": 4,
        "execution_delta": 8,
        "readability_delta": 3,
        "impact_score": 94,
        "confidence": 0.91,
        "raw_transcript": "this must never persist",
        "message_text": "remove me",
        "anchors": [
            {
                "route": "/workspace/alignment?view=internal",
                "artifact": "latest_conversation_feedback_projection_internal",
                "component": "AlignmentWorkspace",
            }
        ],
    }

    result = mod.publish_autopublish_payload(
        review_dir=review_dir,
        payload=payload,
        now_text="2026-03-22T04:00:00Z",
    )

    latest_path = review_dir / "latest_conversation_feedback_autopublish_internal.json"
    assert latest_path.exists()
    written = json.loads(latest_path.read_text(encoding="utf-8"))
    assert written["feedback_id"] == "alignment_projection_rollout"
    assert written["source"] == "auto_session"
    assert written["status"] == "active"
    assert written["created_at_utc"] == "2026-03-22T04:00:00Z"
    assert "raw_transcript" not in written
    assert "message_text" not in written
    assert result["path"] == str(latest_path)
    assert result["event_count"] == 1
    assert result["feedback_ids"] == ["alignment_projection_rollout"]


def test_publish_feedback_autopublish_accepts_list_payload(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    payload = [
        {
            "feedback_id": "first",
            "headline": "first headline",
            "summary": "first summary",
            "recommended_action": "first action",
        },
        {
            "feedback_id": "second",
            "headline": "second headline",
            "summary": "second summary",
            "recommended_action": "second action",
            "messages": ["remove me"],
        },
    ]

    result = mod.publish_autopublish_payload(
        review_dir=review_dir,
        payload=payload,
        now_text="2026-03-22T04:01:00Z",
    )

    latest_path = review_dir / "latest_conversation_feedback_autopublish_internal.json"
    written = json.loads(latest_path.read_text(encoding="utf-8"))
    assert isinstance(written, list)
    assert [row["feedback_id"] for row in written] == ["first", "second"]
    assert written[0]["created_at_utc"] == "2026-03-22T04:01:00Z"
    assert written[1]["source"] == "auto_session"
    assert "messages" not in written[1]
    assert result["event_count"] == 2
    assert result["feedback_ids"] == ["first", "second"]
