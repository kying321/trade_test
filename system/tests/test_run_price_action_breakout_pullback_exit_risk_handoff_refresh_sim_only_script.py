from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(
    "/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_price_action_breakout_pullback_exit_risk_handoff_refresh_sim_only.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_exit_risk_handoff_refresh_script",
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


def test_main_selects_latest_exit_risk_and_runs_handoff(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    review_dir = system_root / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    older_exit_risk = review_dir / "20260321T130100Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    latest_exit_risk = review_dir / "20260323T201500Z_price_action_breakout_pullback_exit_risk_sim_only.json"
    blocker = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    consensus = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    sidecar = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    tail_capacity = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_tail_capacity_sim_only.json"

    write_json(older_exit_risk, {"generated_at_utc": "2026-03-21T13:01:00Z"})
    write_json(latest_exit_risk, {"generated_at_utc": "2026-03-23T20:15:00Z"})
    write_json(blocker, {"research_decision": "exit_risk_forward_blocker_cleared_promote_challenger_pair"})
    write_json(consensus, {"research_decision": "challenger_pair_promotable_across_current_forward_oos"})
    write_json(sidecar, {"research_decision": "break_even_sidecar_no_observed_delta_keep_anchor"})
    write_json(tail_capacity, {"research_decision": "exit_risk_forward_tail_capacity_supports_40d_watch_45d_plus_requires_longer_dataset"})

    seen_cmds: dict[str, list[str]] = {}

    def fake_run_json(*, name: str, cmd: list[str]) -> dict:
        seen_cmds[name] = list(cmd)
        return {
            "json_path": str(review_dir / "20260323T212000Z_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"),
            "latest_json_path": str(review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"),
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
        }

    monkeypatch.setattr(mod, "run_json", fake_run_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_price_action_breakout_pullback_exit_risk_handoff_refresh_sim_only.py",
            "--workspace",
            str(workspace),
            "--now",
            "2026-03-23T21:20:00Z",
        ],
    )

    mod.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["mode"] == "exit_risk_handoff_refresh_sim_only"
    assert payload["stamp"] == "20260323T212000Z"
    assert payload["exit_risk_path"] == str(latest_exit_risk)
    assert payload["forward_blocker_path"] == str(blocker)
    assert payload["forward_consensus_path"] == str(consensus)
    assert payload["break_even_sidecar_path"] == str(sidecar)
    assert payload["tail_capacity_path"] == str(tail_capacity)
    assert payload["research_decision"] == "use_exit_risk_handoff_as_canonical_anchor"
    assert seen_cmds["build_exit_risk_handoff"] == [
        sys.executable,
        str(system_root / "scripts" / "build_price_action_breakout_pullback_exit_risk_handoff_sim_only.py"),
        "--exit-risk-path",
        str(latest_exit_risk),
        "--forward-blocker-path",
        str(blocker),
        "--forward-consensus-path",
        str(consensus),
        "--break-even-sidecar-path",
        str(sidecar),
        "--tail-capacity-path",
        str(tail_capacity),
        "--review-dir",
        str(review_dir),
        "--stamp",
        "20260323T212000Z",
    ]


def test_main_fails_when_no_exit_risk_artifact_exists(tmp_path: Path) -> None:
    mod = load_module()
    workspace = tmp_path / "workspace"
    review_dir = workspace / "system" / "output" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="no_exit_risk_sim_only_artifact_found"):
        mod.latest_exit_risk_artifact(review_dir)
