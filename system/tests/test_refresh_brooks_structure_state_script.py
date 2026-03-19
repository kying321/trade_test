from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/refresh_brooks_structure_state.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("brooks_structure_refresh_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_now_is_monotonic() -> None:
    mod = _load_module()
    base = mod.parse_now("2026-03-13T12:40:00Z")
    assert mod.step_now(base, 0).isoformat() == "2026-03-13T12:40:00+00:00"
    assert mod.step_now(base, 3).isoformat() == "2026-03-13T12:40:03+00:00"
    assert mod.step_now(base, 3) > mod.step_now(base, 2)


def test_main_runs_brooks_refresh_chain_and_writes_refresh_artifact(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    calls: list[tuple[str, list[str]]] = []

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append((step_name, list(cmd)))
        if "--now" in cmd:
            as_of = cmd[cmd.index("--now") + 1]
        else:
            as_of = "2026-03-13T12:40:00Z"
        artifact_name = {
            "build_brooks_price_action_market_study": "brooks_price_action_market_study.json",
            "build_brooks_price_action_route_report": "brooks_price_action_route_report.json",
            "build_brooks_price_action_execution_plan": "brooks_price_action_execution_plan.json",
            "build_brooks_structure_review_queue": "brooks_structure_review_queue.json",
        }[step_name]
        payload = {
            "ok": True,
            "status": "ok",
            "as_of": as_of,
            "artifact": str(review_dir / artifact_name),
        }
        if step_name == "build_brooks_structure_review_queue":
            payload.update(
                {
                    "review_status": "ready",
                    "review_brief": "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now",
                    "queue_status": "ready",
                    "queue_count": 1,
                    "priority_status": "ready",
                    "priority_brief": "ready:SC2603:96:review_queue_now",
                    "head": {
                        "symbol": "SC2603",
                        "strategy_id": "second_entry_trend_continuation",
                        "direction": "LONG",
                        "tier": "review_queue_now",
                        "plan_status": "manual_structure_review_now",
                        "execution_action": "review_manual_stop_entry",
                        "route_selection_score": 81.68,
                        "signal_score": 80,
                        "signal_age_bars": 2,
                        "priority_score": 96,
                        "priority_tier": "review_queue_now",
                        "blocker_detail": "manual only",
                        "done_when": "manual trader confirms venue and sizing",
                    },
                }
            )
        return payload

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-13T12:40:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["study_artifact"] == str(review_dir / "brooks_price_action_market_study.json")
    assert payload["route_report_artifact"] == str(review_dir / "brooks_price_action_route_report.json")
    assert payload["execution_plan_artifact"] == str(review_dir / "brooks_price_action_execution_plan.json")
    assert payload["review_queue_artifact"] == str(review_dir / "brooks_structure_review_queue.json")
    assert payload["review_status"] == "ready"
    assert payload["review_brief"] == "ready:SC2603:second_entry_trend_continuation:manual_structure_review_now"
    assert payload["queue_status"] == "ready"
    assert payload["queue_count"] == 1
    assert payload["priority_status"] == "ready"
    assert payload["priority_brief"] == "ready:SC2603:96:review_queue_now"
    assert payload["head_symbol"] == "SC2603"
    assert payload["head_action"] == "review_manual_stop_entry"
    assert payload["head_priority_score"] == 96
    assert Path(payload["artifact"]).exists()
    assert Path(payload["markdown"]).exists()
    assert Path(payload["checksum"]).exists()

    step_names = [row["name"] for row in payload["steps"]]
    assert step_names == [
        "build_brooks_price_action_market_study",
        "build_brooks_price_action_route_report",
        "build_brooks_price_action_execution_plan",
        "build_brooks_structure_review_queue",
    ]
    study_name, study_cmd = calls[0]
    assert study_name == "build_brooks_price_action_market_study"
    assert study_cmd[1].endswith("backtest_brooks_price_action_all_market.py")
    assert "--output-root" in study_cmd and str(output_root) in study_cmd
    route_name, route_cmd = calls[1]
    assert route_name == "build_brooks_price_action_route_report"
    assert route_cmd[1].endswith("build_brooks_price_action_route_report.py")
    plan_name, plan_cmd = calls[2]
    assert plan_name == "build_brooks_price_action_execution_plan"
    assert plan_cmd[1].endswith("build_brooks_price_action_execution_plan.py")
    queue_name, queue_cmd = calls[3]
    assert queue_name == "build_brooks_structure_review_queue"
    assert queue_cmd[1].endswith("build_brooks_structure_review_queue.py")

    markdown = Path(payload["markdown"]).read_text(encoding="utf-8")
    assert "## Artifacts" in markdown
    assert "## Steps" in markdown
    assert "## Head" in markdown
    assert "review_manual_stop_entry" in markdown


def test_main_reuses_previous_study_artifact_when_local_frames_missing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    output_root = tmp_path / "output"
    review_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    reused_study_path = review_dir / "20260313T124753Z_brooks_price_action_market_study.json"
    reused_study_path.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "ok",
                "as_of": "2026-03-13T12:47:53Z",
                "artifact": str(reused_study_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        if step_name == "build_brooks_price_action_market_study":
            raise RuntimeError("build_brooks_price_action_market_study_failed: no_local_market_frames")
        as_of = cmd[cmd.index("--now") + 1] if "--now" in cmd else "2026-03-13T12:40:00Z"
        artifact_name = {
            "build_brooks_price_action_route_report": "brooks_price_action_route_report.json",
            "build_brooks_price_action_execution_plan": "brooks_price_action_execution_plan.json",
            "build_brooks_structure_review_queue": "brooks_structure_review_queue.json",
        }[step_name]
        payload = {
            "ok": True,
            "status": "ok",
            "as_of": as_of,
            "artifact": str(review_dir / artifact_name),
        }
        if step_name == "build_brooks_structure_review_queue":
            payload.update(
                {
                    "review_status": "inactive",
                    "review_brief": "inactive:-",
                    "queue_status": "inactive",
                    "queue_count": 0,
                    "priority_status": "inactive",
                    "priority_brief": "inactive:-",
                    "head": {},
                }
            )
        return payload

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-14T09:26:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["study_artifact"] == str(reused_study_path)
    assert payload["steps"][0]["name"] == "build_brooks_price_action_market_study"
    assert payload["steps"][0]["status"] == "carry_over_previous_artifact"
    assert payload["steps"][0]["artifact"] == str(reused_study_path)
