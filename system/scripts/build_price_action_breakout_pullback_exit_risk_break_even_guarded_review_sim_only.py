#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
ANCHOR_SLUG_RE = re.compile(r"^hold(?P<hold>\d+)_trail(?P<trail>[0-9]+)_(?P<be>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY guarded-review packet gate for the ETH exit/risk break-even sidecar."
    )
    parser.add_argument("--handoff-path", required=True)
    parser.add_argument("--forward-blocker-path", required=True)
    parser.add_argument("--break-even-sidecar-path", required=True)
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


def parse_anchor_slug(value: str) -> dict[str, str]:
    match = ANCHOR_SLUG_RE.match(text(value))
    if not match:
        return {}
    return match.groupdict()


def is_same_hold_and_trailing_break_even_delta(active_baseline: str, watch_candidate: str) -> bool:
    baseline_parts = parse_anchor_slug(active_baseline)
    candidate_parts = parse_anchor_slug(watch_candidate)
    if not baseline_parts or not candidate_parts:
        return False
    return (
        baseline_parts["hold"] == candidate_parts["hold"]
        and baseline_parts["trail"] == candidate_parts["trail"]
        and baseline_parts["be"] == "no_be"
        and candidate_parts["be"].startswith("be")
        and candidate_parts["be"] != baseline_parts["be"]
    )


def pick_next_priority(*values: Any, fallback: str) -> str:
    for value in values:
        normalized = text(value)
        if normalized:
            return normalized
    return fallback


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Risk Break Even Guarded Review SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- review_state: `{text(payload.get('review_state'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Canonical Pair",
        "",
        f"- active_baseline: `{text(payload.get('active_baseline'))}`",
        f"- watch_candidate: `{text(payload.get('watch_candidate'))}`",
        f"- review_scope: `{text(payload.get('review_scope'))}`",
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

    handoff_path = Path(args.handoff_path).expanduser().resolve()
    forward_blocker_path = Path(args.forward_blocker_path).expanduser().resolve()
    break_even_sidecar_path = Path(args.break_even_sidecar_path).expanduser().resolve()

    handoff = load_json_mapping(handoff_path)
    forward_blocker = load_json_mapping(forward_blocker_path)
    break_even_sidecar = load_json_mapping(break_even_sidecar_path)

    symbol = (
        text(handoff.get("symbol"))
        or text(forward_blocker.get("symbol"))
        or text(break_even_sidecar.get("symbol"))
        or "ETHUSDT"
    )
    handoff_decision = text(handoff.get("research_decision"))
    handoff_head_status = text(handoff.get("source_head_status"))
    blocker_decision = text(forward_blocker.get("research_decision"))
    sidecar_decision = text(break_even_sidecar.get("research_decision"))
    sidecar_confidence_tier = text(break_even_sidecar.get("confidence_tier"))
    sidecar_promotion_review_ready = bool(break_even_sidecar.get("promotion_review_ready"))
    active_baseline = text(handoff.get("active_baseline")) or text(break_even_sidecar.get("active_baseline"))
    watch_candidate = text(handoff.get("watch_candidate")) or text(break_even_sidecar.get("watch_candidate"))

    baseline_alignment_ok = active_baseline and active_baseline == text(break_even_sidecar.get("active_baseline"))
    candidate_alignment_ok = watch_candidate and watch_candidate == text(break_even_sidecar.get("watch_candidate"))
    delta_scope_ok = is_same_hold_and_trailing_break_even_delta(active_baseline, watch_candidate)
    canonical_alignment_ok = baseline_alignment_ok and candidate_alignment_ok and delta_scope_ok

    review_state = "blocked"
    research_decision = "break_even_guarded_review_prerequisites_missing_keep_baseline_anchor"
    allowed_now = [
        "rebuild_canonical_exit_risk_handoff_and_break_even_sidecar_before_review_packet",
    ]
    blocked_now = [
        "run_break_even_candidate_guarded_review_packet",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_canonical_review_gate",
    ]
    next_research_priority = pick_next_priority(
        handoff.get("next_research_priority"),
        forward_blocker.get("next_research_priority"),
        break_even_sidecar.get("next_research_priority"),
        fallback="repair_break_even_review_prerequisites_before_guarded_review",
    )

    prerequisites_ok = (
        handoff_decision == "use_exit_risk_handoff_as_canonical_anchor"
        and handoff_head_status == "baseline_anchor_active"
        and blocker_decision.startswith("block_exit_risk_promotion_keep_baseline_anchor_pair_")
        and sidecar_decision == "break_even_sidecar_positive_watch_only"
    )

    if (
        handoff_decision == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
        or handoff_head_status == "upstream_hold_selection_conflict"
    ):
        research_decision = "break_even_guarded_review_blocked_by_upstream_hold_selection_conflict"
        review_state = "blocked"
        allowed_now = [
            "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review",
        ]
        blocked_now = [
            "run_break_even_candidate_guarded_review_packet",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_while_upstream_hold_selection_conflict_is_active",
        ]
        next_research_priority = "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review"
    elif prerequisites_ok and not canonical_alignment_ok:
        research_decision = "break_even_guarded_review_blocked_canonical_alignment_required"
        review_state = "blocked"
        allowed_now = [
            "rebuild_break_even_sidecar_or_handoff_until_candidate_alignment_is_restored",
        ]
        blocked_now = [
            "run_break_even_candidate_guarded_review_packet",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_canonical_alignment",
        ]
        next_research_priority = "repair_break_even_sidecar_and_canonical_handoff_alignment"
    elif prerequisites_ok and canonical_alignment_ok and sidecar_confidence_tier == "guarded_review_ready" and sidecar_promotion_review_ready:
        research_decision = "break_even_guarded_review_ready_keep_baseline_anchor"
        review_state = "ready"
        allowed_now = [
            "keep_baseline_anchor_as_current_exit_risk_source_head",
            "run_break_even_candidate_guarded_review_packet",
            "treat_break_even_candidate_as_review_only_until_arbitrated",
        ]
        blocked_now = [
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_guarded_review",
        ]
        next_research_priority = "run_break_even_candidate_guarded_review_against_primary_forward_anchor"
    elif prerequisites_ok and canonical_alignment_ok:
        research_decision = "break_even_guarded_review_not_ready_keep_watch_only"
        review_state = "watch_only"
        allowed_now = [
            "keep_baseline_anchor_as_current_exit_risk_source_head",
            "keep_break_even_candidate_as_watch_sidecar_only",
        ]
        blocked_now = [
            "run_break_even_candidate_guarded_review_packet",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_guarded_review",
        ]
        next_research_priority = pick_next_priority(
            handoff.get("next_research_priority"),
            break_even_sidecar.get("next_research_priority"),
            forward_blocker.get("next_research_priority"),
            fallback="watch_break_even_candidate_keep_baseline_anchor_defer_more_tail",
        )

    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "handoff_path": str(handoff_path),
        "forward_blocker_path": str(forward_blocker_path),
        "break_even_sidecar_path": str(break_even_sidecar_path),
        "review_scope": "same_hold_same_trailing_break_even_delta_only",
        "source_evidence": {
            "handoff_research_decision": handoff_decision,
            "handoff_source_head_status": handoff_head_status,
            "forward_blocker_research_decision": blocker_decision,
            "break_even_sidecar_research_decision": sidecar_decision,
            "break_even_sidecar_confidence_tier": sidecar_confidence_tier,
            "break_even_sidecar_promotion_review_ready": sidecar_promotion_review_ready,
            "baseline_alignment_ok": baseline_alignment_ok,
            "candidate_alignment_ok": candidate_alignment_ok,
            "break_even_delta_scope_ok": delta_scope_ok,
        },
        "active_baseline": active_baseline,
        "watch_candidate": watch_candidate,
        "review_state": review_state,
        "research_decision": research_decision,
        "allowed_now": allowed_now,
        "blocked_now": blocked_now,
        "next_research_priority": next_research_priority,
        "consumer_rule": (
            "任何 break-even candidate 进入评审包前，必须先读取 "
            "`latest_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.json`；"
            "不得仅凭 sidecar positive 或 brief 文本直接推动 anchor 变更。"
        ),
        "recommended_brief": (
            f"{symbol}:exit_risk_break_even_guarded_review:"
            f"baseline={active_baseline or 'unknown'},"
            f"candidate={watch_candidate or 'unknown'},"
            f"state={review_state},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 guarded-review 工件把 break-even sidecar 的正向证据进一步收口为 source-owned review gate，"
            "只表达“是否进入评审包”，不直接晋级 canonical anchor。"
        ),
        "limitation_note": (
            "它不替代 primary forward OOS 与 canonical handoff；"
            "若 handoff/blocker/sidecar 任一刷新，guarded review 也必须同步重建。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_guarded_review_sim_only.md"
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
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
