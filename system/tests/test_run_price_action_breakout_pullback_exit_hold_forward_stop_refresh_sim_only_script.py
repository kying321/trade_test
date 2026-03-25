from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_price_action_breakout_pullback_exit_hold_forward_stop_refresh_sim_only.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_exit_hold_forward_stop_refresh_script",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_main_selects_latest_hold_upstream_and_runs_stop_builder(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    forward_capacity_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json"
    overlap_sidecar_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json"
    handoff_path = review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
    window_consensus_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.json"

    write_json(forward_capacity_path, {"research_decision": "non_overlapping_forward_capacity_supports_40d_watch_45d_plus_insufficient"})
    write_json(overlap_sidecar_path, {"research_decision": "overlapping_long_window_split_keep_non_overlapping_baseline_watch_hold8"})
    write_json(handoff_path, {"research_decision": "use_hold_selection_gate_as_canonical_head"})
    write_json(window_consensus_path, {"research_decision": "hold16_consensus_bias_keep_baseline_watch_long_window_regime_split"})

    seen_cmds: dict[str, list[str]] = {}

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_cmds[name] = list(cmd)
        return {
            "json_path": str(review_dir / "20260323T225000Z_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"),
            "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"),
            "research_decision": "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split",
        }

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_hold_forward_stop_refresh_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-23T22:50:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["mode"] == "exit_hold_forward_stop_refresh_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["stamp"] == "20260323T225000Z"
    assert payload["forward_capacity_path"] == str(forward_capacity_path)
    assert payload["overlap_sidecar_path"] == str(overlap_sidecar_path)
    assert payload["handoff_path"] == str(handoff_path)
    assert payload["window_consensus_path"] == str(window_consensus_path)
    assert payload["research_decision"] == (
        "stop_capacity_limited_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split"
    )
    assert seen_cmds["build_exit_hold_forward_stop"] == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.py"),
        "--forward-capacity-path",
        str(forward_capacity_path),
        "--overlap-sidecar-path",
        str(overlap_sidecar_path),
        "--handoff-path",
        str(handoff_path),
        "--window-consensus-path",
        str(window_consensus_path),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T225000Z",
    ]


def test_main_fails_when_latest_hold_selection_handoff_is_missing(tmp_path: Path) -> None:
    mod = load_module()
    review_dir = tmp_path / "workspace" / "system" / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json",
        {"research_decision": "non_overlapping_forward_capacity_supports_40d_watch_45d_plus_insufficient"},
    )
    write_json(
        review_dir / "latest_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json",
        {"research_decision": "overlapping_long_window_split_keep_non_overlapping_baseline_watch_hold8"},
    )

    with pytest.raises(FileNotFoundError, match="missing_hold_selection_handoff_latest"):
        mod.require_path(
            review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
            "missing_hold_selection_handoff_latest",
        )
