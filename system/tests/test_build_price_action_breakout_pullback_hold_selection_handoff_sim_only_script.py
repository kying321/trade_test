from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_handoff(tmp_path: Path, *, override_required: bool) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path, Path]:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    gate_blocker_path = review_dir / "20260321T082000Z_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.json"
    frontier_report_path = review_dir / "20260321T080200Z_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json"
    family_transfer_path = review_dir / "20260321T081100Z_price_action_breakout_pullback_hold_family_transfer_sim_only.json"
    router_transfer_path = review_dir / "20260321T081200Z_price_action_breakout_pullback_hold_router_transfer_sim_only.json"

    _write_json(
        gate_blocker_path,
        {
            "research_decision": "block_hold_candidate_promotion_keep_hold16_anchor_reopen_hold12_watch_demote_hold24_and_router",
            "gate_state": {
                "source_head_override_required": (
                    "yes_transfer_evidence_overrides_old_frontier_head"
                    if override_required
                    else "no_frontier_still_canonical"
                )
            },
            "active_baseline": "hold16_zero",
            "local_candidate": "hold8_zero",
            "transfer_watch": ["hold12_zero"],
            "return_candidate": ["hold24_zero"],
            "return_watch": [],
            "demoted_candidate": ["hold24_zero", "pullback_depth_atr_router"],
            "blocked_now": ["promote_hold8_as_new_baseline"],
            "allowed_now": ["keep_hold16_as_current_baseline_anchor"],
            "release_conditions": ["future tail 扩展后重做 non-overlap forward challenge"],
        },
    )
    _write_json(
        frontier_report_path,
        {
            "research_decision": "mixed_forward_profile_hold8_aggregate_hold16_consistency",
        },
    )
    _write_json(
        family_transfer_path,
        {
            "research_decision": "hold12_revived_in_transfer_watch_only",
        },
    )
    _write_json(
        router_transfer_path,
        {
            "research_decision": "router_transfer_failed_keep_pure_hold_selection",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--gate-blocker-path",
            str(gate_blocker_path),
            "--frontier-report-path",
            str(frontier_report_path),
            "--family-transfer-path",
            str(family_transfer_path),
            "--router-transfer-path",
            str(router_transfer_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260321T083000Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc, review_dir, gate_blocker_path, frontier_report_path, family_transfer_path


def test_handoff_uses_gate_override_as_canonical_head(tmp_path: Path) -> None:
    proc, review_dir, gate_blocker_path, frontier_report_path, _ = _run_handoff(
        tmp_path,
        override_required=True,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    json_path = Path(output["json_path"])
    md_path = Path(output["md_path"])
    latest_json_path = Path(output["latest_json_path"])

    assert json_path.exists()
    assert md_path.exists()
    assert latest_json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["action"] == "build_price_action_breakout_pullback_hold_selection_handoff_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "use_hold_selection_gate_as_canonical_head"
    assert payload["source_head_status"] == "gate_override_active"
    assert payload["canonical_source_head"] == str(gate_blocker_path)
    assert payload["superseded_head"] == str(frontier_report_path)
    assert payload["active_baseline"] == "hold16_zero"
    assert payload["local_candidate"] == "hold8_zero"
    assert payload["transfer_watch"] == ["hold12_zero"]
    assert payload["return_candidate"] == ["hold24_zero"]
    assert payload["return_watch"] == []
    assert payload["demoted_candidate"] == ["hold24_zero", "pullback_depth_atr_router"]
    assert payload["source_evidence"] == {
        "gate_research_decision": "block_hold_candidate_promotion_keep_hold16_anchor_reopen_hold12_watch_demote_hold24_and_router",
        "frontier_research_decision": "mixed_forward_profile_hold8_aggregate_hold16_consistency",
        "family_transfer_research_decision": "hold12_revived_in_transfer_watch_only",
        "router_transfer_research_decision": "router_transfer_failed_keep_pure_hold_selection",
    }
    assert json.loads(latest_json_path.read_text(encoding="utf-8"))["research_decision"] == (
        "use_hold_selection_gate_as_canonical_head"
    )
    assert "gate_override_active" in md_path.read_text(encoding="utf-8")


def test_handoff_marks_inconclusive_when_override_not_required(tmp_path: Path) -> None:
    proc, review_dir, gate_blocker_path, frontier_report_path, _ = _run_handoff(
        tmp_path,
        override_required=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    json_path = Path(output["json_path"])
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["research_decision"] == "hold_selection_handoff_inconclusive"
    assert payload["source_head_status"] == "inconclusive"
    assert payload["canonical_source_head"] == str(gate_blocker_path)
    assert payload["superseded_head"] == str(frontier_report_path)
    assert json.loads(latest_json_path.read_text(encoding="utf-8"))["source_head_status"] == "inconclusive"
