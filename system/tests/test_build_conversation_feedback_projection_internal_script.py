from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_conversation_feedback_projection_internal.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_conversation_feedback_projection_internal_script",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_projection_internal_merges_manual_and_auto_sources(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    manual_path = review_dir / "conversation_feedback_events_internal.jsonl"
    auto_path = review_dir / "latest_conversation_feedback_autopublish_internal.json"

    _write_jsonl(
        manual_path,
        [
            {
                "feedback_id": "ui_density_split",
                "created_at_utc": "2026-03-21T12:00:00Z",
                "source": "manual",
                "domain": "research",
                "headline": "拆分研究工作区左栏四控件",
                "summary": "当前左栏同质化过高，需要拆成研究地图 / 状态雷达 / 检索定位 / 当前焦点。",
                "recommended_action": "拆分左栏结构并降低同质化",
                "alignment_delta": 14,
                "blocker_delta": 8,
                "execution_delta": 10,
                "readability_delta": 12,
                "impact_score": 92,
                "confidence": 0.91,
                "status": "active",
                "anchors": {
                    "route": "/workspace/artifacts",
                    "artifact": "price_action_breakout_pullback",
                    "component": "WorkspacePanels",
                },
                "raw_transcript": "这段原始对话文本不应出现在 projection 内",
            },
            {
                "feedback_id": "theme_black_tile_fixed",
                "created_at_utc": "2026-03-21T12:20:00Z",
                "source": "manual",
                "domain": "overview",
                "headline": "亮色主题黑框修正已完成",
                "summary": "日间模式的硬编码深色 tile 已被修正。",
                "recommended_action": "保留亮色 token 收口",
                "alignment_delta": 6,
                "blocker_delta": -10,
                "execution_delta": 8,
                "readability_delta": -14,
                "impact_score": 66,
                "confidence": 0.88,
                "status": "resolved",
                "anchors": {
                    "route": "/overview",
                    "component": "GlobalTopbar",
                },
                "message_text": "resolved event should not expose raw text",
            },
        ],
    )
    _write_json(
        auto_path,
        {
            "feedback_id": "ui_density_split",
            "created_at_utc": "2026-03-21T11:55:00Z",
            "source": "auto_session",
            "domain": "research",
            "headline": "自动版本 headline",
            "summary": "自动版本 summary，应该被 manual 覆盖。",
            "recommended_action": "自动版本 action",
            "alignment_delta": 5,
            "blocker_delta": 5,
            "execution_delta": 5,
            "readability_delta": 5,
            "impact_score": 40,
            "confidence": 0.55,
            "status": "active",
            "anchors": {
                "route": "/workspace/artifacts",
                "component": "WorkspacePanels",
            },
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_conversation_feedback_projection_internal.py",
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-21T12:30:00Z",
        ],
    )

    mod.main()
    result = json.loads(capsys.readouterr().out)

    latest_artifact = Path(result["latest_artifact"])
    timestamped_artifact = Path(result["artifact"])
    projection = json.loads(latest_artifact.read_text(encoding="utf-8"))

    assert timestamped_artifact.exists()
    assert latest_artifact.exists()
    assert projection["change_class"] == "RESEARCH_ONLY"
    assert projection["visibility"] == "internal_only"
    assert projection["summary"]["headline"] == "拆分研究工作区左栏四控件"
    assert projection["summary"]["drift_state"] in {"aligned", "watch", "drifting"}
    assert [row["feedback_id"] for row in projection["events"]] == ["ui_density_split"]
    assert projection["events"][0]["source"] == "manual"
    assert projection["events"][0]["summary"].startswith("当前左栏同质化过高")
    assert "raw_transcript" not in projection["events"][0]
    assert "message_text" not in projection["events"][0]
    assert projection["actions"][0]["recommended_action"] == "拆分左栏结构并降低同质化"
    assert projection["anchors"][0]["route"] == "/workspace/artifacts"
    assert {point["feedback_id"] for point in projection["trends"]} == {
        "ui_density_split",
        "theme_black_tile_fixed",
    }


def test_build_projection_internal_honors_explicit_input_paths(tmp_path: Path, monkeypatch, capsys) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    custom_manual_path = tmp_path / "custom" / "manual.jsonl"
    custom_auto_path = tmp_path / "custom" / "auto.json"

    _write_jsonl(
        custom_manual_path,
        [
            {
                "feedback_id": "manual_override_only",
                "created_at_utc": "2026-03-21T13:00:00Z",
                "source": "manual",
                "domain": "research",
                "headline": "manual override headline",
                "summary": "manual override summary",
                "recommended_action": "manual override action",
                "alignment_delta": 5,
                "blocker_delta": 1,
                "execution_delta": 4,
                "readability_delta": 1,
                "impact_score": 77,
                "confidence": 0.81,
                "status": "active",
            }
        ],
    )
    _write_json(
        custom_auto_path,
        {
            "feedback_id": "auto_override_only",
            "created_at_utc": "2026-03-21T13:05:00Z",
            "source": "auto_session",
            "domain": "overview",
            "headline": "auto override headline",
            "summary": "auto override summary",
            "recommended_action": "auto override action",
            "alignment_delta": 4,
            "blocker_delta": 0,
            "execution_delta": 3,
            "readability_delta": 0,
            "impact_score": 65,
            "confidence": 0.74,
            "status": "active",
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_conversation_feedback_projection_internal.py",
            "--review-dir",
            str(review_dir),
            "--manual-events-path",
            str(custom_manual_path),
            "--auto-publish-path",
            str(custom_auto_path),
            "--now",
            "2026-03-21T13:30:00Z",
        ],
    )

    mod.main()
    result = json.loads(capsys.readouterr().out)
    projection = json.loads(Path(result["latest_artifact"]).read_text(encoding="utf-8"))

    assert {row["feedback_id"] for row in projection["events"]} == {
        "manual_override_only",
        "auto_override_only",
    }
    assert projection["summary"]["headline"] == "manual override headline"
