from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_builder_completes_primary_anchor_review_and_keeps_baseline_anchor(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_conclusion_path = review_dir / "review_conclusion.json"
    review_packet_path = review_dir / "review_packet.json"
    handoff_path = review_dir / "handoff.json"
    forward_consensus_path = review_dir / "forward_consensus.json"

    _write_json(
        review_conclusion_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_conclusion_ready_keep_baseline_anchor_review_only",
            "review_decision": "keep_baseline_anchor_review_break_even_candidate_next",
            "arbitration_state": "review_only",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold16_trail0_be075",
            "next_research_priority": "review_break_even_candidate_against_primary_forward_anchor",
        },
    )
    _write_json(
        review_packet_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_packet_ready_for_primary_anchor_review",
            "packet_state": "ready",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold16_trail0_be075",
            "evidence_window_count": 6,
        },
    )
    _write_json(
        handoff_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
            "source_head_status": "baseline_anchor_active",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
            "allowed_now": ["keep_baseline_pair_as_current_exit_risk_anchor"],
            "blocked_now": ["promote_challenger_pair_as_new_exit_risk_anchor"],
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-conclusion-path",
            str(review_conclusion_path),
            "--review-packet-path",
            str(review_packet_path),
            "--handoff-path",
            str(handoff_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T184500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))
    latest_payload = json.loads(Path(output["latest_json_path"]).read_text(encoding="utf-8"))

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "break_even_primary_anchor_review_complete_keep_baseline_anchor"
    assert payload["review_state"] == "completed"
    assert payload["review_outcome"] == "baseline_anchor_retained_candidate_remains_review_only"
    assert payload["primary_anchor"] == "hold16_trail0_no_be"
    assert payload["review_candidate"] == "hold16_trail0_be075"
    assert payload["allowed_now"] == [
        "keep_baseline_anchor_as_current_exit_risk_source_head",
        "keep_break_even_candidate_as_review_only_sidecar",
        "wait_fresh_primary_forward_anchor_evidence_before_break_even_candidate_reopen",
    ]
    assert payload["blocked_now"] == [
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
    ]
    assert payload["next_research_priority"] == "wait_fresh_primary_forward_anchor_evidence_before_break_even_candidate_reopen"
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_keeps_watch_only_when_review_conclusion_is_not_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_conclusion_path = review_dir / "review_conclusion.json"
    review_packet_path = review_dir / "review_packet.json"
    handoff_path = review_dir / "handoff.json"
    forward_consensus_path = review_dir / "forward_consensus.json"

    _write_json(
        review_conclusion_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_conclusion_watch_only_keep_baseline_anchor",
            "review_decision": "keep_baseline_anchor_watch_break_even_candidate_only",
            "arbitration_state": "watch_only",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold16_trail0_be075",
            "next_research_priority": "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )
    _write_json(
        review_packet_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_packet_not_ready_keep_watch_only",
            "packet_state": "watch_only",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold16_trail0_be075",
            "evidence_window_count": 2,
        },
    )
    _write_json(
        handoff_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
            "source_head_status": "baseline_anchor_active",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-conclusion-path",
            str(review_conclusion_path),
            "--review-packet-path",
            str(review_packet_path),
            "--handoff-path",
            str(handoff_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T184600Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_primary_anchor_review_watch_only_keep_baseline_anchor"
    assert payload["review_state"] == "watch_only"
    assert payload["review_outcome"] == "baseline_anchor_retained_candidate_stays_watch_only"
    assert payload["allowed_now"] == [
        "keep_baseline_anchor_as_current_exit_risk_source_head",
        "keep_break_even_candidate_as_review_only_sidecar",
    ]
    assert payload["blocked_now"] == [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
    ]
    assert payload["next_research_priority"] == "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_blocks_when_primary_forward_anchor_is_not_confirmed(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_conclusion_path = review_dir / "review_conclusion.json"
    review_packet_path = review_dir / "review_packet.json"
    handoff_path = review_dir / "handoff.json"
    forward_consensus_path = review_dir / "forward_consensus.json"

    _write_json(
        review_conclusion_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_conclusion_ready_keep_baseline_anchor_review_only",
            "review_decision": "keep_baseline_anchor_review_break_even_candidate_next",
            "arbitration_state": "review_only",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold16_trail0_be075",
            "next_research_priority": "review_break_even_candidate_against_primary_forward_anchor",
        },
    )
    _write_json(
        review_packet_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_packet_ready_for_primary_anchor_review",
            "packet_state": "ready",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold16_trail0_be075",
            "evidence_window_count": 6,
        },
    )
    _write_json(
        handoff_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
            "source_head_status": "baseline_anchor_active",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "challenger_pair_promotable_across_current_forward_oos",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-conclusion-path",
            str(review_conclusion_path),
            "--review-packet-path",
            str(review_packet_path),
            "--handoff-path",
            str(handoff_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T184700Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_primary_anchor_review_blocked_primary_forward_anchor_not_confirmed"
    assert payload["review_state"] == "blocked"
    assert payload["review_outcome"] == "primary_forward_anchor_confirmation_required_before_break_even_review"
    assert payload["allowed_now"] == [
        "keep_baseline_anchor_as_current_exit_risk_source_head",
        "refresh_primary_forward_anchor_confirmation_before_break_even_review",
    ]
    assert payload["blocked_now"] == [
        "complete_break_even_candidate_review_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
    ]
    assert payload["next_research_priority"] == "refresh_primary_forward_anchor_confirmation_before_break_even_review"



def test_builder_blocks_primary_anchor_review_with_explicit_upstream_hold_selection_conflict(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_conclusion_path = review_dir / "review_conclusion.json"
    review_packet_path = review_dir / "review_packet.json"
    handoff_path = review_dir / "handoff.json"
    forward_consensus_path = review_dir / "forward_consensus.json"

    _write_json(
        review_conclusion_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_conclusion_blocked_by_upstream_hold_selection_conflict",
            "review_decision": "resolve_upstream_hold_selection_conflict_before_break_even_review_conclusion",
            "arbitration_state": "blocked",
            "primary_anchor": "hold24_trail0_no_be",
            "review_candidate": "hold24_trail0_be075",
            "next_research_priority": "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review",
        },
    )
    _write_json(
        review_packet_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_packet_blocked_by_upstream_hold_selection_conflict",
            "packet_state": "blocked",
            "primary_anchor": "hold24_trail0_no_be",
            "review_candidate": "hold24_trail0_be075",
            "evidence_window_count": 1,
            "next_research_priority": "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review",
        },
    )
    _write_json(
        handoff_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict",
            "source_head_status": "upstream_hold_selection_conflict",
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )
    _write_json(
        forward_consensus_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "baseline_pair_keeps_anchor_across_current_forward_oos",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-conclusion-path",
            str(review_conclusion_path),
            "--review-packet-path",
            str(review_packet_path),
            "--handoff-path",
            str(handoff_path),
            "--forward-consensus-path",
            str(forward_consensus_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T184800Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_primary_anchor_review_blocked_by_upstream_hold_selection_conflict"
    assert payload["review_state"] == "blocked"
    assert payload["review_outcome"] == "resolve_upstream_hold_selection_conflict_before_primary_anchor_review"
    assert payload["allowed_now"] == [
        "resolve_upstream_hold_selection_vs_exit_risk_anchor_conflict",
    ]
    assert payload["blocked_now"] == [
        "complete_break_even_candidate_review_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_while_upstream_hold_selection_conflict_is_active",
    ]
    assert payload["next_research_priority"] == "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review"
