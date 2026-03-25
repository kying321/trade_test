#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY primary-anchor review artifact for the break-even candidate."
    )
    parser.add_argument("--review-conclusion-path", required=True)
    parser.add_argument("--review-packet-path", required=True)
    parser.add_argument("--handoff-path", required=True)
    parser.add_argument("--forward-consensus-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def aligned(value: str, *candidates: Any) -> bool:
    if not text(value):
        return False
    seen = [text(candidate) for candidate in candidates if text(candidate)]
    return bool(seen) and all(candidate == value for candidate in seen)


def pick_next_priority(*values: Any, fallback: str) -> str:
    for value in values:
        normalized = text(value)
        if normalized:
            return normalized
    return fallback


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Risk Break Even Primary Anchor Review SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- review_state: `{text(payload.get('review_state'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- review_outcome: `{text(payload.get('review_outcome'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Canonical Pair",
        "",
        f"- primary_anchor: `{text(payload.get('primary_anchor'))}`",
        f"- review_candidate: `{text(payload.get('review_candidate'))}`",
        f"- evidence_window_count: `{int(payload.get('evidence_window_count') or 0)}`",
        "",
        "## Allowed Now",
        "",
    ]
    for row in payload.get("allowed_now", []):
        lines.append(f"- `{text(row)}`")
    lines.extend(["", "## Blocked Now", ""])
    for row in payload.get("blocked_now", []):
        lines.append(f"- `{text(row)}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- `{text(payload.get('research_note'))}`",
            f"- `{text(payload.get('limitation_note'))}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    review_conclusion_path = Path(args.review_conclusion_path).expanduser().resolve()
    review_packet_path = Path(args.review_packet_path).expanduser().resolve()
    handoff_path = Path(args.handoff_path).expanduser().resolve()
    forward_consensus_path = Path(args.forward_consensus_path).expanduser().resolve()

    review_conclusion = load_json_mapping(review_conclusion_path)
    review_packet = load_json_mapping(review_packet_path)
    handoff = load_json_mapping(handoff_path)
    forward_consensus = load_json_mapping(forward_consensus_path)

    symbol = (
        text(review_conclusion.get("symbol"))
        or text(review_packet.get("symbol"))
        or text(handoff.get("symbol"))
        or text(forward_consensus.get("symbol"))
        or "ETHUSDT"
    )

    conclusion_primary_anchor = text(review_conclusion.get("primary_anchor"))
    packet_primary_anchor = text(review_packet.get("primary_anchor"))
    handoff_primary_anchor = text(handoff.get("active_baseline"))
    primary_anchor = conclusion_primary_anchor or packet_primary_anchor or handoff_primary_anchor

    conclusion_candidate = text(review_conclusion.get("review_candidate"))
    packet_candidate = text(review_packet.get("review_candidate"))
    handoff_candidate = text(handoff.get("watch_candidate"))
    review_candidate = conclusion_candidate or packet_candidate or handoff_candidate

    conclusion_decision = text(review_conclusion.get("research_decision"))
    conclusion_state = text(review_conclusion.get("arbitration_state"))
    packet_decision = text(review_packet.get("research_decision"))
    packet_state = text(review_packet.get("packet_state"))
    handoff_decision = text(handoff.get("research_decision"))
    handoff_status = text(handoff.get("source_head_status"))
    forward_consensus_decision = text(forward_consensus.get("research_decision"))

    canonical_handoff_ok = (
        handoff_decision == "use_exit_risk_handoff_as_canonical_anchor"
        and handoff_status == "baseline_anchor_active"
    )
    primary_forward_anchor_confirmed = (
        forward_consensus_decision == "baseline_pair_keeps_anchor_across_current_forward_oos"
    )
    baseline_alignment_ok = aligned(primary_anchor, conclusion_primary_anchor, packet_primary_anchor, handoff_primary_anchor)
    candidate_alignment_ok = aligned(review_candidate, conclusion_candidate, packet_candidate, handoff_candidate)

    review_state = "blocked"
    research_decision = "break_even_primary_anchor_review_blocked_by_conclusion_or_primary_forward_anchor"
    review_outcome = "repair_primary_anchor_review_inputs_before_completion"
    allowed_now = [
        "keep_baseline_anchor_as_current_exit_risk_source_head",
        "repair_primary_anchor_review_inputs_before_break_even_review_completion",
    ]
    blocked_now = [
        "complete_break_even_candidate_review_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
    ]
    next_research_priority = pick_next_priority(
        review_conclusion.get("next_research_priority"),
        review_packet.get("next_research_priority"),
        handoff.get("next_research_priority"),
        forward_consensus.get("next_research_priority"),
        fallback="repair_primary_anchor_review_inputs_before_break_even_review_completion",
    )

    if (
        conclusion_decision == "break_even_review_conclusion_blocked_by_upstream_hold_selection_conflict"
        or packet_decision == "break_even_review_packet_blocked_by_upstream_hold_selection_conflict"
        or handoff_decision == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
        or handoff_status == "upstream_hold_selection_conflict"
    ):
        research_decision = "break_even_primary_anchor_review_blocked_by_upstream_hold_selection_conflict"
        review_outcome = "resolve_upstream_hold_selection_conflict_before_primary_anchor_review"
        allowed_now = [
            "resolve_upstream_hold_selection_vs_exit_risk_anchor_conflict",
        ]
        blocked_now = [
            "complete_break_even_candidate_review_against_primary_forward_anchor",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_while_upstream_hold_selection_conflict_is_active",
        ]
        next_research_priority = pick_next_priority(
            review_conclusion.get("next_research_priority"),
            review_packet.get("next_research_priority"),
            fallback="resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review",
        )
    elif canonical_handoff_ok and not (baseline_alignment_ok and candidate_alignment_ok):
        research_decision = "break_even_primary_anchor_review_blocked_canonical_alignment_required"
        review_outcome = "repair_primary_anchor_review_alignment_before_completion"
        allowed_now = [
            "keep_baseline_anchor_as_current_exit_risk_source_head",
            "repair_primary_anchor_review_alignment_before_break_even_review_completion",
        ]
        next_research_priority = "repair_primary_anchor_review_alignment_before_break_even_review_completion"
    elif canonical_handoff_ok and baseline_alignment_ok and candidate_alignment_ok and not primary_forward_anchor_confirmed:
        research_decision = "break_even_primary_anchor_review_blocked_primary_forward_anchor_not_confirmed"
        review_outcome = "primary_forward_anchor_confirmation_required_before_break_even_review"
        allowed_now = [
            "keep_baseline_anchor_as_current_exit_risk_source_head",
            "refresh_primary_forward_anchor_confirmation_before_break_even_review",
        ]
        next_research_priority = "refresh_primary_forward_anchor_confirmation_before_break_even_review"
    elif (
        canonical_handoff_ok
        and primary_forward_anchor_confirmed
        and baseline_alignment_ok
        and candidate_alignment_ok
        and conclusion_decision == "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
        and conclusion_state == "review_only"
        and packet_decision == "break_even_review_packet_ready_for_primary_anchor_review"
        and packet_state == "ready"
    ):
        review_state = "completed"
        research_decision = "break_even_primary_anchor_review_complete_keep_baseline_anchor"
        review_outcome = "baseline_anchor_retained_candidate_remains_review_only"
        allowed_now = [
            "keep_baseline_anchor_as_current_exit_risk_source_head",
            "keep_break_even_candidate_as_review_only_sidecar",
            "wait_fresh_primary_forward_anchor_evidence_before_break_even_candidate_reopen",
        ]
        blocked_now = [
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
        ]
        next_research_priority = "wait_fresh_primary_forward_anchor_evidence_before_break_even_candidate_reopen"
    elif (
        canonical_handoff_ok
        and primary_forward_anchor_confirmed
        and baseline_alignment_ok
        and candidate_alignment_ok
        and conclusion_state == "watch_only"
        and packet_state in {"watch_only", "ready"}
    ):
        review_state = "watch_only"
        research_decision = "break_even_primary_anchor_review_watch_only_keep_baseline_anchor"
        review_outcome = "baseline_anchor_retained_candidate_stays_watch_only"
        allowed_now = [
            "keep_baseline_anchor_as_current_exit_risk_source_head",
            "keep_break_even_candidate_as_review_only_sidecar",
        ]
        blocked_now = [
            "review_break_even_candidate_against_primary_forward_anchor",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
        ]
        next_research_priority = pick_next_priority(
            review_conclusion.get("next_research_priority"),
            review_packet.get("next_research_priority"),
            fallback="watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        )

    evidence_window_count = int(review_packet.get("evidence_window_count") or 0)
    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "review_conclusion_path": str(review_conclusion_path),
        "review_packet_path": str(review_packet_path),
        "handoff_path": str(handoff_path),
        "forward_consensus_path": str(forward_consensus_path),
        "primary_anchor": primary_anchor,
        "review_candidate": review_candidate,
        "review_state": review_state,
        "review_outcome": review_outcome,
        "evidence_window_count": evidence_window_count,
        "source_evidence": {
            "review_conclusion_research_decision": conclusion_decision,
            "review_conclusion_state": conclusion_state,
            "review_packet_research_decision": packet_decision,
            "review_packet_state": packet_state,
            "handoff_research_decision": handoff_decision,
            "handoff_source_head_status": handoff_status,
            "forward_consensus_research_decision": forward_consensus_decision,
            "baseline_alignment_ok": baseline_alignment_ok,
            "candidate_alignment_ok": candidate_alignment_ok,
            "primary_forward_anchor_confirmed": primary_forward_anchor_confirmed,
        },
        "allowed_now": allowed_now,
        "blocked_now": blocked_now,
        "next_research_priority": next_research_priority,
        "research_decision": research_decision,
        "consumer_rule": (
            "任何 consumer 若要展示 break-even 候选已完成的 primary-anchor review 状态，"
            "必须先读取 `latest_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.json`，"
            "不得直接从 review_conclusion / handoff / forward_consensus 文本侧自行拼装。"
        ),
        "recommended_brief": (
            f"{symbol}:exit_risk_break_even_primary_anchor_review:"
            f"baseline={primary_anchor or 'unknown'},"
            f"candidate={review_candidate or 'unknown'},"
            f"state={review_state},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 primary-anchor review 工件用于正式落盘“已按当前 primary forward anchor 完成评审”这一层，"
            "把 baseline retained / candidate review-only / waiting fresh evidence 的状态固定成单一 source-owned 结果。"
        ),
        "limitation_note": (
            "它不重跑 compare，不替代 canonical handoff 或 forward consensus；"
            "若 review_conclusion / review_packet / handoff / forward_consensus 任一刷新，必须同步重建。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_primary_anchor_review_sim_only.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "latest_json_path": str(latest_json_path),
                "latest_md_path": str(latest_md_path),
                "research_decision": research_decision,
                "review_state": review_state,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
