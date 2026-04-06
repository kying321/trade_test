from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_cpa_guarded_action.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_cpa_guarded_action_script", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_action_dry_run_writes_receipt(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "latest_cpa_control_plane_snapshot.json").write_text(
        json.dumps(
            {
                "guarded_actions": [
                    {
                        "id": "acceptance_replay_success20",
                        "label": "验收回放 / 历史成功20",
                        "risk_class": "LIVE_GUARD_ONLY",
                        "command": "cd /tmp/mactools && python3 check_five_account_acceptance.py --csv data/registered_success_active20.csv",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    mod.now_utc = lambda: mod.dt.datetime(2026, 4, 7, 1, 30, tzinfo=mod.dt.timezone.utc)
    result = mod.run_action(
        workspace=workspace,
        action_id="acceptance_replay_success20",
        execute=False,
    )

    assert result["ok"] is True
    assert result["mode"] == "cpa_guarded_action"
    assert result["status"] == "dry_run"
    assert result["action_id"] == "acceptance_replay_success20"
    assert result["command"].startswith("cd /tmp/mactools")
    assert Path(result["artifact_json"]).exists()
    latest = review_dir / "latest_cpa_guarded_action_receipt_acceptance_replay_success20.json"
    assert latest.exists()


def test_run_action_execute_records_subprocess_result(monkeypatch, tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "latest_cpa_control_plane_snapshot.json").write_text(
        json.dumps(
            {
                "guarded_actions": [
                    {
                        "id": "retry_candidate_pipeline",
                        "label": "重试 retry_candidate 队列",
                        "risk_class": "LIVE_GUARD_ONLY",
                        "command": "cd /tmp/mactools && python3 run_retry_candidate_pipeline.py --max-attempts 4 --pretty",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class DummyProc:
        returncode = 0
        stdout = '{"accepted": true}\n'
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: DummyProc())
    mod.now_utc = lambda: mod.dt.datetime(2026, 4, 7, 1, 31, tzinfo=mod.dt.timezone.utc)

    result = mod.run_action(workspace=workspace, action_id="retry_candidate_pipeline", execute=True)

    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert result["stdout_preview"] == '{"accepted": true}'
