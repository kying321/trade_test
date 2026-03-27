from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refresh_brooks_structure_state.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("brooks_structure_refresh_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
        as_of = cmd[cmd.index("--now") + 1] if "--now" in cmd else "2026-03-13T12:40:00Z"
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
                        "execution_action": "review_manual_stop_entry",
                        "priority_score": 96,
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
    assert payload["head_symbol"] == "SC2603"
    assert payload["head_action"] == "review_manual_stop_entry"
    assert payload["head_priority_score"] == 96
    assert Path(payload["artifact"]).exists()
    assert Path(payload["markdown"]).exists()
    assert Path(payload["checksum"]).exists()
    assert [row["name"] for row in payload["steps"]] == [
        "build_brooks_price_action_market_study",
        "build_brooks_price_action_route_report",
        "build_brooks_price_action_execution_plan",
        "build_brooks_structure_review_queue",
    ]
    assert calls[0][0] == "build_brooks_price_action_market_study"
