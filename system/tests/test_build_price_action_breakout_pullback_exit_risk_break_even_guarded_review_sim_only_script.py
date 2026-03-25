from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_builder_marks_guarded_review_ready_when_baseline_anchor_and_sidecar_align(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    handoff_path = review_dir / "20260324T171518Z_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"
    blocker_path = review_dir / "20260324T171517Z_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    sidecar_path = review_dir / "20260324T171515Z_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"

    _write_json(
        handoff_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
            "source_head_status": "baseline_anchor_active",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )
    _write_json(
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold12_vs_hold16",
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
            "challenge_pair": {
                "baseline_hold_bars": 16,
                "challenger_hold_bars": 12,
                "shared_trailing_stop_atr": 0.0,
            },
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "guarded_review_ready",
            "promotion_review_ready": True,
            "evidence_scope": "overlapping_forward_compare_windows_only",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "next_research_priority": "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--handoff-path",
            str(handoff_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T172000Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    output = json.loads(proc.stdout)
    payload = json.loads(Path(output["json_path"]).read_text(encoding="utf-8"))
    latest_payload = json.loads(Path(output["latest_json_path"]).read_text(encoding="utf-8"))

    assert payload["action"] == "build_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only"
    assert payload["change_class"] == "SIM_ONLY"
    assert payload["research_decision"] == "break_even_guarded_review_ready_keep_baseline_anchor"
    assert payload["review_state"] == "ready"
    assert payload["active_baseline"] == "hold16_trail0_no_be"
    assert payload["watch_candidate"] == "hold16_trail0_be075"
    assert payload["blocked_now"] == [
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_guarded_review",
    ]
    assert payload["allowed_now"] == [
        "keep_baseline_anchor_as_current_exit_risk_source_head",
        "run_break_even_candidate_guarded_review_packet",
        "treat_break_even_candidate_as_review_only_until_arbitrated",
    ]
    assert payload["next_research_priority"] == "run_break_even_candidate_guarded_review_against_primary_forward_anchor"
    assert payload["review_scope"] == "same_hold_same_trailing_break_even_delta_only"
    assert latest_payload["research_decision"] == payload["research_decision"]


def test_builder_keeps_watch_only_when_sidecar_not_yet_guarded_review_ready(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    handoff_path = review_dir / "handoff.json"
    blocker_path = review_dir / "blocker.json"
    sidecar_path = review_dir / "sidecar.json"

    _write_json(
        handoff_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "use_exit_risk_handoff_as_canonical_anchor",
            "source_head_status": "baseline_anchor_active",
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "next_research_priority": "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )
    _write_json(
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold12_vs_hold16",
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "watch_only",
            "promotion_review_ready": False,
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold16_trail0_be075",
            "next_research_priority": "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--handoff-path",
            str(handoff_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T172100Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_guarded_review_not_ready_keep_watch_only"
    assert payload["review_state"] == "watch_only"
    assert payload["allowed_now"] == [
        "keep_baseline_anchor_as_current_exit_risk_source_head",
        "keep_break_even_candidate_as_watch_sidecar_only",
    ]
    assert payload["blocked_now"] == [
        "run_break_even_candidate_guarded_review_packet",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_guarded_review",
    ]
    assert payload["next_research_priority"] == "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def test_builder_blocks_when_sidecar_and_handoff_candidate_alignment_breaks(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    handoff_path = review_dir / "handoff.json"
    blocker_path = review_dir / "blocker.json"
    sidecar_path = review_dir / "sidecar.json"

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
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold12_vs_hold16",
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "guarded_review_ready",
            "promotion_review_ready": True,
            "active_baseline": "hold16_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--handoff-path",
            str(handoff_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T172200Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_guarded_review_blocked_canonical_alignment_required"
    assert payload["review_state"] == "blocked"
    assert payload["allowed_now"] == [
        "rebuild_break_even_sidecar_or_handoff_until_candidate_alignment_is_restored",
    ]
    assert payload["blocked_now"] == [
        "run_break_even_candidate_guarded_review_packet",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_canonical_alignment",
    ]
    assert payload["next_research_priority"] == "repair_break_even_sidecar_and_canonical_handoff_alignment"


def test_builder_blocks_when_handoff_is_already_blocked_by_upstream_hold_selection_conflict(tmp_path: Path) -> None:
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    handoff_path = review_dir / "handoff.json"
    blocker_path = review_dir / "blocker.json"
    sidecar_path = review_dir / "sidecar.json"

    _write_json(
        handoff_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict",
            "source_head_status": "upstream_hold_selection_conflict",
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
            "next_research_priority": "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review",
        },
    )
    _write_json(
        blocker_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "block_exit_risk_promotion_keep_baseline_anchor_pair_hold16_vs_hold24",
        },
    )
    _write_json(
        sidecar_path,
        {
            "symbol": "ETHUSDT",
            "research_decision": "break_even_sidecar_positive_watch_only",
            "confidence_tier": "guarded_review_ready",
            "promotion_review_ready": True,
            "active_baseline": "hold24_trail0_no_be",
            "watch_candidate": "hold24_trail0_be075",
        },
    )

    proc = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--handoff-path",
            str(handoff_path),
            "--forward-blocker-path",
            str(blocker_path),
            "--break-even-sidecar-path",
            str(sidecar_path),
            "--review-dir",
            str(review_dir),
            "--stamp",
            "20260324T173000Z",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(Path(json.loads(proc.stdout)["json_path"]).read_text(encoding="utf-8"))

    assert payload["research_decision"] == "break_even_guarded_review_blocked_by_upstream_hold_selection_conflict"
    assert payload["review_state"] == "blocked"
    assert payload["allowed_now"] == [
        "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review",
    ]
    assert payload["blocked_now"] == [
        "run_break_even_candidate_guarded_review_packet",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_while_upstream_hold_selection_conflict_is_active",
    ]
    assert payload["next_research_priority"] == "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review"
