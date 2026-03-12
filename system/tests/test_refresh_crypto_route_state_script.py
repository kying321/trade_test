from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/refresh_crypto_route_state.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("crypto_route_refresh_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_step_now_is_monotonic() -> None:
    mod = _load_module()
    base = mod.parse_now("2026-03-12T05:40:00Z")
    assert mod.step_now(base, 0).isoformat() == "2026-03-12T05:40:00+00:00"
    assert mod.step_now(base, 4).isoformat() == "2026-03-12T05:40:04+00:00"
    assert mod.step_now(base, 5) > mod.step_now(base, 4)


def test_main_runs_guarded_refresh_chain_and_writes_refresh_artifact(
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
    scripted_wall_clock = {
        "build_source_control_report": "2026-03-12T05:45:00Z",
        "build_native_group_report": "2026-03-12T05:45:01Z",
        "build_native_lane_stability_report": "2026-03-12T05:45:02Z",
        "build_native_beta_leg_report": "2026-03-12T05:45:03Z",
        "build_native_lane_playbook": "2026-03-12T05:45:04Z",
        "build_native_beta_leg_window_report": "2026-03-12T05:45:05Z",
        "build_combo_playbook": "2026-03-12T05:45:06Z",
        "build_symbol_route_handoff": "2026-03-12T05:45:07Z",
    }

    def fake_run_json_step(*, step_name: str, cmd: list[str]) -> dict:
        calls.append((step_name, list(cmd)))
        artifact_name = f"{step_name}.json"
        if step_name == "build_crypto_route_brief":
            artifact_name = "crypto_route_brief.json"
        if "--now" in cmd:
            as_of = cmd[cmd.index("--now") + 1]
        else:
            as_of = scripted_wall_clock[step_name]
        return {
            "ok": True,
            "status": "ok",
            "as_of": as_of,
            "artifact": str(review_dir / artifact_name),
        }

    monkeypatch.setattr(mod, "run_json_step", fake_run_json_step)

    rc = mod.main(
        [
            "--review-dir",
            str(review_dir),
            "--output-root",
            str(output_root),
            "--now",
            "2026-03-12T05:40:00Z",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["crypto_route_brief_artifact"] == str(review_dir / "crypto_route_brief.json")
    assert payload["source_control_artifact"] == str(review_dir / "build_source_control_report.json")
    assert payload["route_handoff_artifact"] == str(review_dir / "build_symbol_route_handoff.json")
    assert payload["artifact"].endswith("_crypto_route_refresh.json")
    assert Path(payload["artifact"]).exists()

    step_names = [row["name"] for row in payload["steps"]]
    assert step_names[:3] == ["native_custom", "native_majors", "native_beta"]
    assert step_names[-3:] == ["build_combo_playbook", "build_symbol_route_handoff", "build_crypto_route_brief"]

    first_name, first_cmd = calls[0]
    assert first_name == "native_custom"
    assert first_cmd[1].endswith("run_backtest_binance_indicator_combo_native_crypto_guarded.py")
    assert "--symbol-group" in first_cmd and "custom" in first_cmd

    last_name, last_cmd = calls[-1]
    assert last_name == "build_crypto_route_brief"
    assert last_cmd[1].endswith("build_crypto_route_brief.py")
    assert "--now" in last_cmd and "2026-03-12T05:45:09Z" in last_cmd
