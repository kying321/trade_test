from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_builder_keeps_baseline_anchor_and_marks_candidate_review_only_when_packet_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_packet_path = review_dir / "review_packet.json"
    guarded_review_path = review_dir / "guarded_review.json"
    handoff_path = review_dir / "handoff.json"

    _write_json(
        review_packet_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_packet_ready_for_primary_anchor_review",
            "packet_state": "ready",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold16_trail0_be075",
            "next_research_priority": "review_break_even_candidate_against_primary_forward_anchor",
        },
    )
    _write_json(
        guarded_review_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_guarded_review_ready_keep_baseline_anchor",
            "review_state": "ready",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-packet-path",
            str(review_packet_path),
            "--guarded-review-path",
            str(guarded_review_path),
            "--handoff-path",
            str(handoff_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T181500Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))
    latest_payload = json.loads(Path(output["latest_json_path"]).read_text(encoding="utf-8"))

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
    assert payload["review_decision"] == "keep_baseline_anchor_review_break_even_candidate_next"
    assert payload["arbitration_state"] == "review_only"
    assert payload["primary_anchor"] == "hold16_trail0_no_be"
    assert payload["review_candidate"] == "hold16_trail0_be075"
    assert payload["allowed_now"] == [
        "keep_baseline_anchor_as_current_exit_risk_source_head",
        "review_break_even_candidate_against_primary_forward_anchor",
        "keep_break_even_candidate_review_only_until_fresh_primary_forward_anchor_evidence_clears_promotion",
    ]
    assert payload["blocked_now"] == [
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
    ]
    assert payload["next_research_priority"] == "review_break_even_candidate_against_primary_forward_anchor"
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_keeps_watch_only_when_review_packet_is_not_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_packet_path = review_dir / "review_packet.json"
    guarded_review_path = review_dir / "guarded_review.json"
    handoff_path = review_dir / "handoff.json"

    _write_json(
        review_packet_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_packet_not_ready_keep_watch_only",
            "packet_state": "watch_only",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold16_trail0_be075",
            "next_research_priority": "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )
    _write_json(
        guarded_review_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_guarded_review_not_ready_keep_watch_only",
            "review_state": "watch_only",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-packet-path",
            str(review_packet_path),
            "--guarded-review-path",
            str(guarded_review_path),
            "--handoff-path",
            str(handoff_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T181600Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_review_conclusion_watch_only_keep_baseline_anchor"
    assert payload["review_decision"] == "keep_baseline_anchor_watch_break_even_candidate_only"
    assert payload["arbitration_state"] == "watch_only"
    assert payload["allowed_now"] == [
        "keep_baseline_anchor_as_current_exit_risk_source_head",
        "keep_break_even_candidate_as_review_only_sidecar",
    ]
    assert payload["blocked_now"] == [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
    ]
    assert payload["next_research_priority"] == "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_blocks_when_review_packet_and_handoff_are_misaligned(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_packet_path = review_dir / "review_packet.json"
    guarded_review_path = review_dir / "guarded_review.json"
    handoff_path = review_dir / "handoff.json"

    _write_json(
        review_packet_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_packet_ready_for_primary_anchor_review",
            "packet_state": "ready",
            "primary_anchor": "hold16_trail0_no_be",
            "review_candidate": "hold24_trail0_be075",
            "next_research_priority": "review_break_even_candidate_against_primary_forward_anchor",
        },
    )
    _write_json(
        guarded_review_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_guarded_review_ready_keep_baseline_anchor",
            "review_state": "ready",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-packet-path",
            str(review_packet_path),
            "--guarded-review-path",
            str(guarded_review_path),
            "--handoff-path",
            str(handoff_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T181700Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_review_conclusion_blocked_canonical_alignment_required"
    assert payload["review_decision"] == "repair_review_packet_or_canonical_alignment_before_conclusion"
    assert payload["arbitration_state"] == "blocked"
    assert payload["allowed_now"] == [
        "repair_review_packet_or_canonical_handoff_alignment_before_review_conclusion",
    ]
    assert payload["blocked_now"] == [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_review_conclusion",
    ]
    assert payload["next_research_priority"] == "repair_break_even_review_packet_and_canonical_handoff_alignment"



def test_builder_blocks_review_conclusion_with_explicit_upstream_hold_selection_conflict(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    review_packet_path = review_dir / "review_packet.json"
    guarded_review_path = review_dir / "guarded_review.json"
    handoff_path = review_dir / "handoff.json"

    _write_json(
        review_packet_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_review_packet_blocked_by_upstream_hold_selection_conflict",
            "packet_state": "blocked",
            "primary_anchor": "hold24_trail0_no_be",
            "review_candidate": "hold24_trail0_be075",
            "next_research_priority": "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review",
        },
    )
    _write_json(
        guarded_review_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_guarded_review_blocked_by_upstream_hold_selection_conflict",
            "review_state": "blocked",
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
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

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--review-packet-path",
            str(review_packet_path),
            "--guarded-review-path",
            str(guarded_review_path),
            "--handoff-path",
            str(handoff_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T181800Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_review_conclusion_blocked_by_upstream_hold_selection_conflict"
    assert payload["review_decision"] == "resolve_upstream_hold_selection_conflict_before_break_even_review_conclusion"
    assert payload["arbitration_state"] == "blocked"
    assert payload["allowed_now"] == [
        "resolve_upstream_hold_selection_vs_exit_risk_anchor_conflict",
    ]
    assert payload["blocked_now"] == [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_while_upstream_hold_selection_conflict_is_active",
    ]
    assert payload["next_research_priority"] == "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review"
