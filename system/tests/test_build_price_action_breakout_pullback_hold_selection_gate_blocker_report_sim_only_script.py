from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_gate_blocker(
    tmp_path: Path,
    *,
    frontier_report_payload: dict,
    router_hypothesis_decision: str,
    router_transfer_decision: str,
    family_transfer_decision: str,
) -> dict:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    frontier_report_path = review_dir / "20260321T080200Z_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json"
    frontier_cost_path = review_dir / "20260321T080600Z_price_action_breakout_pullback_exit_hold_cost_guard_sim_only.json"
    router_hypothesis_path = review_dir / "20260321T081000Z_price_action_breakout_pullback_hold_router_hypothesis_sim_only.json"
    router_transfer_path = review_dir / "20260321T081200Z_price_action_breakout_pullback_hold_router_transfer_sim_only.json"
    family_transfer_path = review_dir / "20260321T081100Z_price_action_breakout_pullback_hold_family_transfer_sim_only.json"

    _write_json(
        frontier_report_path,
        frontier_report_payload,
    )
    _write_json(
        frontier_cost_path,
        {"research_decision": "hold16_cost_profile_still_anchor"},
    )
    _write_json(
        router_hypothesis_path,
        {"research_decision": router_hypothesis_decision},
    )
    _write_json(
        router_transfer_path,
        {"research_decision": router_transfer_decision},
    )
    _write_json(
        family_transfer_path,
        {"research_decision": family_transfer_decision},
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--frontier-report-path",
            str(frontier_report_path),
            "--frontier-cost-path",
            str(frontier_cost_path),
            "--router-hypothesis-path",
            str(router_hypothesis_path),
            "--router-transfer-path",
            str(router_transfer_path),
            "--family-transfer-path",
            str(family_transfer_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260321T083500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
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
    latest_payload = json.loads(latest_json_path.read_text(encoding="utf-8"))
    return {
        "payload": payload,
        "markdown": md_path.read_text(encoding="utf-8"),
        "latest_payload": latest_payload,
    }


def test_gate_blocker_report_uses_source_owned_roles_for_return_candidate_and_router_state(tmp_path: Path) -> None:
    result = _run_gate_blocker(
        tmp_path,
        frontier_report_payload={
            "research_decision": "hold16_baseline_reinforced_candidates_mixed",
            "frontier_rows": [
                {"config_id": "hold16_zero", "role": "baseline_anchor"},
                {"config_id": "hold8_zero", "role": "objective_watch_candidate"},
                {"config_id": "hold24_zero", "role": "return_leader_candidate"},
                {"config_id": "hold12_zero", "role": "transfer_watch_candidate"},
            ],
        },
        router_hypothesis_decision="pullback_depth_router_hypothesis_emerges_but_same_sample_only",
        router_transfer_decision="frozen_router_positive_on_historical_transfer_but_future_tail_insufficient",
        family_transfer_decision="hold_family_transfer_consistent_or_inconclusive",
    )
    payload = result["payload"]

    assert payload["action"] == "build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == (
        "block_hold_candidate_promotion_keep_hold16_anchor_reopen_hold12_watch_demote_hold24_and_router"
    )
    assert payload["gate_state"]["source_head_override_required"] == (
        "yes_transfer_evidence_overrides_old_frontier_head"
    )
    assert payload["active_baseline"] == "hold16_zero"
    assert payload["local_candidate"] == "hold8_zero"
    assert payload["transfer_watch"] == ["hold12_zero"]
    assert payload["return_candidate"] == ["hold24_zero"]
    assert payload["return_watch"] == []
    assert payload["demoted_candidate"] == ["hold24_zero", "pullback_depth_atr_router"]
    assert payload["gate_state"]["hold24_promotion"] == "blocked_return_candidate_until_longer_forward_oos"
    assert payload["gate_state"]["hold12_global_drop"] == "blocked_transfer_watch_only"
    assert payload["gate_state"]["dynamic_router_promotion"] == (
        "blocked_future_tail_insufficient_after_positive_historical_transfer"
    )
    assert payload["source_evidence"] == {
        "frontier_report_research_decision": "hold16_baseline_reinforced_candidates_mixed",
        "frontier_cost_research_decision": "hold16_cost_profile_still_anchor",
        "router_hypothesis_research_decision": "pullback_depth_router_hypothesis_emerges_but_same_sample_only",
        "router_transfer_research_decision": "frozen_router_positive_on_historical_transfer_but_future_tail_insufficient",
        "family_transfer_research_decision": "hold_family_transfer_consistent_or_inconclusive",
    }
    assert "return_candidate" in result["markdown"]
    assert result["latest_payload"]["allowed_now"] == [
        "keep_hold16_as_current_baseline_anchor",
        "keep_hold8_as_local_window_candidate_only",
        "treat_hold12_as_transfer_watch_only",
        "treat_hold24_as_demoted_until_new_forward_evidence",
        "collect_longer_forward_tail_and_re-run_transfer_or_forward_challenge",
    ]


def test_gate_blocker_report_distinguishes_frontier_absence_from_transfer_demotion(tmp_path: Path) -> None:
    result = _run_gate_blocker(
        tmp_path,
        frontier_report_payload={
            "research_decision": "hold16_baseline_reinforced_transfer_watch_only",
            "frontier_rows": [
                {"config_id": "hold16_zero", "role": "baseline_anchor"},
                {"config_id": "hold8_zero", "role": "objective_watch_candidate"},
                {"config_id": "hold12_zero", "role": "transfer_watch_candidate"},
            ],
        },
        router_hypothesis_decision="pullback_depth_router_hypothesis_emerges_but_same_sample_only",
        router_transfer_decision="frozen_router_transfer_does_not_beat_hold8_future_tail_insufficient",
        family_transfer_decision="hold_family_transfer_consistent_or_inconclusive",
    )
    payload = result["payload"]

    assert payload["return_candidate"] == []
    assert payload["return_watch"] == []
    assert payload["gate_state"]["hold24_promotion"] == "blocked_frontier_no_active_return_candidate"
    assert payload["gate_state"]["dynamic_router_promotion"] == "blocked_transfer_does_not_beat_hold8"
