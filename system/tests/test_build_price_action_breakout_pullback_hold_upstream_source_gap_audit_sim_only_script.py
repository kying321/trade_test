from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_builder_detects_missing_builder_sources_for_existing_latest_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    scripts_dir = system_root / "scripts"
    review_dir = system_root / "output" / "review"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    # present scripts
    for name in [
        "build_price_action_breakout_pullback_hold_frontier_report_sim_only.py",
        "build_price_action_breakout_pullback_hold_router_hypothesis_sim_only.py",
        "build_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.py",
        "build_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.py",
        "build_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.py",
        "build_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.py",
        "build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py",
        "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.py",
    ]:
        (scripts_dir / name).write_text("# stub\n", encoding="utf-8")

    # latest artifacts exist for all, but two matching builder sources are missing
    for name in [
        "latest_price_action_breakout_pullback_hold_frontier_report_sim_only.json",
        "latest_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json",
        "latest_price_action_breakout_pullback_hold_router_hypothesis_sim_only.json",
        "latest_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json",
        "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json",
        "latest_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.json",
        "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json",
        "latest_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json",
        "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json",
        "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json",
    ]:
        _write_json(review_dir / name, {"artifact": name})

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T232000Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    json_path = Path(output["json_path"])
    latest_json_path = Path(output["latest_json_path"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    latest_payload = json.loads(latest_json_path.read_text(encoding="utf-8"))

    assert payload["action"] == "build_price_action_breakout_pullback_hold_upstream_source_gap_audit_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "hold_upstream_source_gap_detected_missing_builder_sources"
    assert payload["finding_count"] == 2
    assert payload["missing_builder_labels"] == [
        "hold_family_transfer",
        "hold_frontier_cost_sensitivity",
    ]
    assert payload["findings"] == [
        {
            "label": "hold_family_transfer",
            "latest_artifact": str(review_dir / "latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json"),
            "builder_script": str(scripts_dir / "build_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.py"),
            "issue": "latest_artifact_exists_but_builder_script_missing",
        },
        {
            "label": "hold_frontier_cost_sensitivity",
            "latest_artifact": str(review_dir / "latest_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json"),
            "builder_script": str(scripts_dir / "build_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.py"),
            "issue": "latest_artifact_exists_but_builder_script_missing",
        },
    ]
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_marks_chain_consistent_when_no_source_gap_exists(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    system_root = workspace / "system"
    scripts_dir = system_root / "scripts"
    review_dir = system_root / "output" / "review"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("latest_price_action_breakout_pullback_hold_frontier_report_sim_only.json", "build_price_action_breakout_pullback_hold_frontier_report_sim_only.py"),
        ("latest_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json", "build_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.py"),
        ("latest_price_action_breakout_pullback_hold_router_hypothesis_sim_only.json", "build_price_action_breakout_pullback_hold_router_hypothesis_sim_only.py"),
        ("latest_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json", "build_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.py"),
        ("latest_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.json", "build_price_action_breakout_pullback_hold_family_transfer_challenge_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.json", "build_price_action_breakout_pullback_exit_hold_window_consensus_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json", "build_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.json", "build_price_action_breakout_pullback_exit_hold_overlap_sidecar_sim_only.py"),
        ("latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json", "build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py"),
        ("latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json", "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.py"),
    ]

    for latest_name, script_name in pairs:
        _write_json(review_dir / latest_name, {"artifact": latest_name})
        (scripts_dir / script_name).write_text("# stub\n", encoding="utf-8")

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--workspace",
            str(workspace),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260323T232500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "hold_upstream_builder_sources_present_for_current_latest_artifacts"
    assert payload["finding_count"] == 0
    assert payload["missing_builder_labels"] == []
