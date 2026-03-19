from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_adaptive_task_mode_report.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("adaptive_task_mode_report_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_modes_prefers_ui_rendering_for_dashboard_task() -> None:
    mod = load_module()
    selected = mod.select_modes(
        task_summary="完善前端窗口，增加任务可视化管理面板，展示链路传导过程和影响范围",
        changed_paths=[
            "system/dashboard/web/dist/index.html",
            "system/scripts/build_hot_universe_operator_brief.py",
        ],
    )
    assert selected["primary_mode"] == "ui_rendering"
    assert selected["secondary_modes"]


def test_select_modes_prefers_architecture_review_for_drift_audit() -> None:
    mod = load_module()
    selected = mod.select_modes(
        task_summary="重新审查项目全景架构，以防逻辑偏移和 source ownership 漂移",
        changed_paths=[
            "system/scripts/refresh_cross_market_operator_state.py",
            "system/scripts/build_hot_universe_operator_brief.py",
        ],
    )
    assert selected["primary_mode"] == "architecture_review"


def test_select_modes_prefers_live_guard_for_remote_ready_check() -> None:
    mod = load_module()
    selected = mod.select_modes(
        task_summary="统一账户 ready-check 只读诊断，确认 remote live blockers",
        changed_paths=[
            "system/scripts/openclaw_cloud_bridge.sh",
            "system/scripts/binance_live_takeover.py",
        ],
    )
    assert selected["primary_mode"] == "live_guard_diagnostics"


def test_main_writes_artifact_and_markdown(tmp_path: Path, monkeypatch) -> None:
    mod = load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT_PATH),
            "--task-summary",
            "Add a cross-market review queue source field and one consumer",
            "--changed-path",
            "system/scripts/refresh_cross_market_operator_state.py",
            "--changed-path",
            "system/scripts/build_hot_universe_operator_brief.py",
            "--review-dir",
            str(review_dir),
            "--now",
            "2026-03-14T02:00:00Z",
        ],
    )
    assert mod.main() == 0
    artifact = review_dir / "20260314T020000Z_adaptive_task_mode_report.json"
    markdown = review_dir / "20260314T020000Z_adaptive_task_mode_report.md"
    assert artifact.exists()
    assert markdown.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["primary_mode"] == "source_first_implementation"
    assert payload["recommended_change_class"] == "RESEARCH_ONLY"
    assert payload["failure_severity_matrix"]
    assert payload["source_of_truth_registry"]
    assert payload["validation_report_requirements"]
    assert payload["rollout_ladder"]
    text = markdown.read_text(encoding="utf-8")
    assert "# Adaptive Task Mode Report" in text
    assert "## Stop Rules" in text
    assert "## Failure Severity Matrix" in text
    assert "## Source-of-Truth Registry" in text
