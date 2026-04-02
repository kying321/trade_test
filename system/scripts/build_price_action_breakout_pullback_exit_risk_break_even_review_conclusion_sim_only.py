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
        description="Build a SIM_ONLY break-even review conclusion from the review packet, guarded review, and canonical handoff."
    )
    parser.add_argument("--review-packet-path", required=True)
    parser.add_argument("--guarded-review-path", required=True)
    parser.add_argument("--handoff-path", required=True)
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
        "# Price Action Breakout Pullback Exit Risk Break Even Review Conclusion SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- arbitration_state: `{text(payload.get('arbitration_state'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- review_decision: `{text(payload.get('review_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Canonical Pair",
        "",
        f"- primary_anchor: `{text(payload.get('primary_anchor'))}`",
        f"- review_candidate: `{text(payload.get('review_candidate'))}`",
        f"- arbitration_scope: `{text(payload.get('arbitration_scope'))}`",
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

    review_packet_path = Path(args.review_packet_path).expanduser().resolve()
    guarded_review_path = Path(args.guarded_review_path).expanduser().resolve()
    handoff_path = Path(args.handoff_path).expanduser().resolve()

    review_packet = load_json_mapping(review_packet_path)
    guarded_review = load_json_mapping(guarded_review_path)
    handoff = load_json_mapping(handoff_path)

    symbol = (
        text(review_packet.get("symbol"))
        or text(guarded_review.get("symbol"))
        or text(handoff.get("symbol"))
        or "ETHUSDT"
    )

    packet_primary_anchor = text(review_packet.get("primary_anchor"))
    guarded_primary_anchor = text(guarded_review.get("active_baseline"))
    handoff_primary_anchor = text(handoff.get("active_baseline"))
    primary_anchor = packet_primary_anchor or guarded_primary_anchor or handoff_primary_anchor

    packet_review_candidate = text(review_packet.get("review_candidate"))
    guarded_review_candidate = text(guarded_review.get("watch_candidate"))
    handoff_review_candidate = text(handoff.get("watch_candidate"))
    review_candidate = packet_review_candidate or guarded_review_candidate or handoff_review_candidate

    packet_state = text(review_packet.get("packet_state"))
    packet_decision = text(review_packet.get("research_decision"))
    guarded_state = text(guarded_review.get("review_state"))
    guarded_decision = text(guarded_review.get("research_decision"))
    handoff_decision = text(handoff.get("research_decision"))
    handoff_source_head_status = text(handoff.get("source_head_status"))

    baseline_alignment_ok = aligned(primary_anchor, packet_primary_anchor, guarded_primary_anchor, handoff_primary_anchor)
    candidate_alignment_ok = aligned(
        review_candidate,
        packet_review_candidate,
        guarded_review_candidate,
        handoff_review_candidate,
    )
    canonical_handoff_ok = (
        handoff_decision == "use_exit_risk_handoff_as_canonical_anchor"
        and handoff_source_head_status == "baseline_anchor_active"
    )

    arbitration_state = "blocked"
    research_decision = "break_even_review_conclusion_blocked_by_review_packet_or_handoff"
    review_decision = "repair_review_packet_or_canonical_alignment_before_conclusion"
    allowed_now = [
        "repair_review_packet_or_canonical_handoff_alignment_before_review_conclusion",
    ]
    blocked_now = [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_review_conclusion",
    ]
    next_research_priority = pick_next_priority(
        review_packet.get("next_research_priority"),
        guarded_review.get("next_research_priority"),
        handoff.get("next_research_priority"),
        fallback="repair_break_even_review_packet_or_canonical_handoff_before_review_conclusion",
    )

    if (
        packet_decision == "break_even_review_packet_blocked_by_upstream_hold_selection_conflict"
        or guarded_decision == "break_even_guarded_review_blocked_by_upstream_hold_selection_conflict"
        or handoff_decision == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
        or handoff_source_head_status == "upstream_hold_selection_conflict"
    ):
        research_decision = "break_even_review_conclusion_blocked_by_upstream_hold_selection_conflict"
        review_decision = "resolve_upstream_hold_selection_conflict_before_break_even_review_conclusion"
        allowed_now = [
            "resolve_upstream_hold_selection_vs_exit_risk_anchor_conflict",
        ]
        blocked_now = [
            "review_break_even_candidate_against_primary_forward_anchor",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_while_upstream_hold_selection_conflict_is_active",
        ]
        next_research_priority = pick_next_priority(
            review_packet.get("next_research_priority"),
            guarded_review.get("next_research_priority"),
            fallback="resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review",
        )
    elif canonical_handoff_ok and not (baseline_alignment_ok and candidate_alignment_ok):
        research_decision = "break_even_review_conclusion_blocked_canonical_alignment_required"
        review_decision = "repair_review_packet_or_canonical_alignment_before_conclusion"
        next_research_priority = "repair_break_even_review_packet_and_canonical_handoff_alignment"
    elif (
        canonical_handoff_ok
        and baseline_alignment_ok
        and candidate_alignment_ok
        and packet_state == "ready"
        and packet_decision == "break_even_review_packet_ready_for_primary_anchor_review"
        and guarded_state == "ready"
        and guarded_decision == "break_even_guarded_review_ready_keep_baseline_anchor"
    ):
        arbitration_state = "review_only"
        research_decision = "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
        review_decision = "keep_baseline_anchor_review_break_even_candidate_next"
        allowed_now = [
            "keep_baseline_anchor_as_current_exit_risk_source_head",
            "review_break_even_candidate_against_primary_forward_anchor",
            "keep_break_even_candidate_review_only_until_fresh_primary_forward_anchor_evidence_clears_promotion",
        ]
        blocked_now = [
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
        ]
        next_research_priority = "review_break_even_candidate_against_primary_forward_anchor"
    elif (
        canonical_handoff_ok
        and baseline_alignment_ok
        and candidate_alignment_ok
        and packet_state == "watch_only"
        and guarded_state in {"watch_only", "ready"}
    ):
        arbitration_state = "watch_only"
        research_decision = "break_even_review_conclusion_watch_only_keep_baseline_anchor"
        review_decision = "keep_baseline_anchor_watch_break_even_candidate_only"
        allowed_now = [
            "keep_baseline_anchor_as_current_exit_risk_source_head",
            "keep_break_even_candidate_as_review_only_sidecar",
        ]
        blocked_now = [
            "review_break_even_candidate_against_primary_forward_anchor",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_fresh_primary_forward_anchor_evidence",
        ]
        next_research_priority = pick_next_priority(
            review_packet.get("next_research_priority"),
            guarded_review.get("next_research_priority"),
            handoff.get("next_research_priority"),
            fallback="watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        )

    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "review_packet_path": str(review_packet_path),
        "guarded_review_path": str(guarded_review_path),
        "handoff_path": str(handoff_path),
        "primary_anchor": primary_anchor,
        "review_candidate": review_candidate,
        "arbitration_scope": "primary_forward_anchor_break_even_delta_only",
        "arbitration_state": arbitration_state,
        "research_decision": research_decision,
        "review_decision": review_decision,
        "source_evidence": {
            "review_packet_research_decision": packet_decision,
            "review_packet_state": packet_state,
            "guarded_review_research_decision": guarded_decision,
            "guarded_review_state": guarded_state,
            "handoff_research_decision": handoff_decision,
            "handoff_source_head_status": handoff_source_head_status,
            "baseline_alignment_ok": baseline_alignment_ok,
            "candidate_alignment_ok": candidate_alignment_ok,
        },
        "allowed_now": allowed_now,
        "blocked_now": blocked_now,
        "next_research_priority": next_research_priority,
        "consumer_rule": (
            "任何需要引用 break-even 候选评审结论的 consumer，必须先读取 "
            "`latest_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.json`，"
            "不得直接从 review packet / guarded review 自行拼装 allowed_now、blocked_now 或下一研究优先级。"
        ),
        "recommended_brief": (
            f"{symbol}:exit_risk_break_even_review_conclusion:"
            f"baseline={primary_anchor or 'unknown'},"
            f"candidate={review_candidate or 'unknown'},"
            f"state={arbitration_state},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 review conclusion 只负责把 review packet 进一步落成 source-owned 仲裁结论层，"
            "明确当前允许动作、阻止动作与下一研究优先级，仍默认保持 baseline anchor。"
        ),
        "limitation_note": (
            "它不重跑 backtest，不替代 primary forward OOS，也不直接提升 break-even candidate 为 canonical anchor；"
            "若 handoff / guarded review / review packet 任一刷新，review conclusion 必须同步重建。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_conclusion_sim_only.md"
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
                "arbitration_state": arbitration_state,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
