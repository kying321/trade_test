from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sidecar_windows() -> list[dict]:
    return [
        {
            "path": "/tmp/compare_30.json",
            "train_days": 30,
            "validation_days": 10,
            "step_days": 5,
            "validation_window_mode": "overlapping",
            "slice_count": 7,
            "winner_by_aggregate_return": "anchor_with_be",
            "winner_by_aggregate_objective": "anchor_with_be",
            "winner_by_slice_majority_return": "anchor_with_be",
            "winner_by_slice_majority_objective": "anchor_with_be",
        },
        {
            "path": "/tmp/compare_40.json",
            "train_days": 40,
            "validation_days": 10,
            "step_days": 5,
            "validation_window_mode": "overlapping",
            "slice_count": 9,
            "winner_by_aggregate_return": "anchor_with_be",
            "winner_by_aggregate_objective": "anchor_with_be",
            "winner_by_slice_majority_return": "anchor_with_be",
            "winner_by_slice_majority_objective": "anchor_with_be",
        },
    ]


def test_builder_marks_review_packet_ready_when_guarded_review_is_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    guarded_review_path = review_dir / "guarded_review.json"
    handoff_path = review_dir / "handoff.json"
    sidecar_path = review_dir / "sidecar.json"

    _write_json(
        guarded_review_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_guarded_review_ready_keep_baseline_anchor",
            "review_state": "ready",
            "review_scope": "same_hold_same_trailing_break_even_delta_only",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "next_research_priority": "run_break_even_candidate_guarded_review_against_primary_forward_anchor",
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
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "evidence_scope": "overlapping_forward_compare_windows_only",
            "source_artifacts": ["/tmp/compare_30.json", "/tmp/compare_40.json"],
            "windows": _sidecar_windows(),
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--guarded-review-path",
            str(guarded_review_path),
            "--handoff-path",
            str(handoff_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T174000Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))
    latest_payload = json.loads(Path(output["latest_json_path"]).read_text(encoding="utf-8"))

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "break_even_review_packet_ready_for_primary_anchor_review"
    assert payload["packet_state"] == "ready"
    assert payload["primary_anchor"] == "hold16_trail0_no_be"
    assert payload["review_candidate"] == "hold16_trail0_be075"
    assert payload["review_scope"] == "same_hold_same_trailing_break_even_delta_only"
    assert payload["evidence_scope"] == "overlapping_forward_compare_windows_only"
    assert payload["evidence_window_count"] == 2
    assert payload["allowed_now"] == [
        "review_break_even_candidate_against_primary_forward_anchor",
        "keep_break_even_candidate_review_only_until_packet_arbitration",
    ]
    assert payload["blocked_now"] == [
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_review_packet_arbitration",
    ]
    assert payload["next_research_priority"] == "review_break_even_candidate_against_primary_forward_anchor"
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_keeps_review_packet_watch_only_when_guarded_review_not_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    guarded_review_path = review_dir / "guarded_review.json"
    handoff_path = review_dir / "handoff.json"
    sidecar_path = review_dir / "sidecar.json"

    _write_json(
        guarded_review_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_guarded_review_not_ready_keep_watch_only",
            "review_state": "watch_only",
            "review_scope": "same_hold_same_trailing_break_even_delta_only",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "next_research_priority": "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
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
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "evidence_scope": "non_overlapping_forward_compare_windows_only",
            "source_artifacts": ["/tmp/compare_30.json"],
            "windows": _sidecar_windows()[:1],
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--guarded-review-path",
            str(guarded_review_path),
            "--handoff-path",
            str(handoff_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T174100Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_review_packet_not_ready_keep_watch_only"
    assert payload["packet_state"] == "watch_only"
    assert payload["allowed_now"] == [
        "keep_break_even_candidate_as_watch_sidecar_only",
        "wait_for_stronger_guarded_review_gate",
    ]
    assert payload["blocked_now"] == [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_review_packet_arbitration",
    ]
    assert payload["next_research_priority"] == "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_blocks_review_packet_when_guarded_review_is_blocked(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    guarded_review_path = review_dir / "guarded_review.json"
    handoff_path = review_dir / "handoff.json"
    sidecar_path = review_dir / "sidecar.json"

    _write_json(
        guarded_review_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_guarded_review_blocked_canonical_alignment_required",
            "review_state": "blocked",
            "review_scope": "same_hold_same_trailing_break_even_delta_only",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
            "next_research_priority": "repair_break_even_sidecar_and_canonical_handoff_alignment",
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
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "evidence_scope": "overlapping_forward_compare_windows_only",
            "source_artifacts": ["/tmp/compare_30.json"],
            "windows": _sidecar_windows()[:1],
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--guarded-review-path",
            str(guarded_review_path),
            "--handoff-path",
            str(handoff_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T174200Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["packet_state"] == "blocked"
    assert payload["allowed_now"] == ["repair_guarded_review_inputs_before_review_packet"]
    assert payload["blocked_now"] == [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_canonical_alignment",
    ]
    assert payload["next_research_priority"] == "repair_break_even_sidecar_and_canonical_handoff_alignment"


def test_builder_blocks_review_packet_with_explicit_upstream_hold_selection_conflict_reason(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    guarded_review_path = review_dir / "guarded_review.json"
    handoff_path = review_dir / "handoff.json"
    sidecar_path = review_dir / "sidecar.json"

    _write_json(
        guarded_review_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_guarded_review_blocked_by_upstream_hold_selection_conflict",
            "review_state": "blocked",
            "review_scope": "same_hold_same_trailing_break_even_delta_only",
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
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "evidence_scope": "overlapping_forward_compare_windows_only",
            "source_artifacts": ["/tmp/compare_30.json"],
            "windows": _sidecar_windows()[:1],
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--guarded-review-path",
            str(guarded_review_path),
            "--handoff-path",
            str(handoff_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T174300Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))
    assert payload["research_decision"] == "break_even_review_packet_blocked_by_upstream_hold_selection_conflict"
    assert payload["packet_state"] == "blocked"
    assert payload["allowed_now"] == [
        "resolve_upstream_hold_selection_vs_exit_risk_anchor_conflict",
    ]
    assert payload["blocked_now"] == [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_while_upstream_hold_selection_conflict_is_active",
    ]
    assert payload["next_research_priority"] == "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review"

