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
HOLD_BARS_RE = re.compile(r"^hold(?P<hold>\d+)(?:_|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY canonical handoff for ETH exit/risk after forward OOS promotion."
    )
    parser.add_argument("--exit-risk-path", required=True)
    parser.add_argument("--forward-blocker-path", required=True)
    parser.add_argument("--forward-consensus-path", required=True)
    parser.add_argument("--break-even-sidecar-path", required=True)
    parser.add_argument("--tail-capacity-path", required=True)
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


def unique_extend(values: list[str], additions: list[str]) -> list[str]:
    seen = set(values)
    for item in additions:
        if item and item not in seen:
            values.append(item)
            seen.add(item)
    return values


def normalize_exit_params(payload: Any) -> dict[str, Any]:
    data = dict(payload or {})
    return {
        "max_hold_bars": int(data.get("max_hold_bars") or 0),
        "break_even_trigger_r": float(data.get("break_even_trigger_r") or 0.0),
        "trailing_stop_atr": float(data.get("trailing_stop_atr") or 0.0),
        "cooldown_after_losses": int(data.get("cooldown_after_losses") or 0),
        "cooldown_bars": int(data.get("cooldown_bars") or 0),
    }


def scaled_decimal(value: float, *, scale: int, width: int = 0) -> str:
    scaled = int(round(float(value or 0.0) * scale))
    if scaled == 0:
        return "0"
    return f"{scaled:0{width}d}" if width else str(scaled)


def format_exit_risk_anchor_slug(params: dict[str, Any]) -> str:
    max_hold_bars = int(params.get("max_hold_bars") or 0)
    trailing = scaled_decimal(float(params.get("trailing_stop_atr") or 0.0), scale=10)
    break_even = float(params.get("break_even_trigger_r") or 0.0)
    break_even_slug = "no_be" if break_even <= 0 else f"be{scaled_decimal(break_even, scale=100, width=3)}"
    return f"hold{max_hold_bars}_trail{trailing}_{break_even_slug}"


def extract_hold_bars(value: Any) -> int:
    match = HOLD_BARS_RE.match(text(value))
    if not match:
        return 0
    return int(match.group("hold"))


def build_transfer_watch(allowed_now: list[str]) -> list[str]:
    watch: list[str] = []
    if "treat_55d_plus_tie_windows_as_watch_only" in allowed_now:
        watch.append("55d_plus_tie_windows")
    return watch


def display_superseded_anchor(value: Any) -> str:
    return text(value) or "none"


def derive_baseline_follow_up_priority(
    *,
    break_even_sidecar_decision: str,
    break_even_sidecar_confidence_tier: str = "",
    tail_capacity_decision: str,
    fallback: str,
) -> str:
    tail_capacity_watch_compatible = {
        "exit_risk_forward_tail_capacity_limited_watch_only",
        "exit_risk_forward_tail_capacity_supports_current_grid_no_tail_extension_required",
    }
    if (
        text(break_even_sidecar_decision) == "break_even_sidecar_positive_watch_only"
        and text(tail_capacity_decision) in tail_capacity_watch_compatible
    ):
        if text(break_even_sidecar_confidence_tier) == "guarded_review_ready":
            return "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail"
        return "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"
    return text(fallback) or "forward_oos_follow_up_pending"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Risk Handoff SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Canonical Anchor",
        "",
        f"- source_head_status: `{text(payload.get('source_head_status'))}`",
        f"- active_baseline: `{text(payload.get('active_baseline'))}`",
        f"- active_baseline_hold_bars: `{int(payload.get('active_baseline_hold_bars') or 0)}`",
        f"- watch_candidate: `{text(payload.get('watch_candidate')) or 'none'}`",
        f"- superseded_anchor: `{display_superseded_anchor(payload.get('superseded_anchor'))}`",
        f"- transfer_watch: `{json.dumps(payload.get('transfer_watch') or [], ensure_ascii=False)}`",
        f"- hold_selection_active_baseline: `{text(payload.get('hold_selection_active_baseline')) or 'none'}`",
        f"- hold_selection_active_hold_bars: `{int(payload.get('hold_selection_active_hold_bars') or 0)}`",
        f"- upstream_hold_alignment_state: `{text(payload.get('upstream_hold_alignment_state'))}`",
        "",
        "## Forward Windows",
        "",
        f"- baseline_windows: `{text(payload.get('baseline_windows'))}`",
        f"- tie_windows: `{text(payload.get('tie_windows'))}`",
        f"- challenger_windows: `{text(payload.get('challenger_windows'))}`",
        "",
        "## Consumer Rule",
        "",
        f"- `{text(payload.get('consumer_rule'))}`",
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


def derive_anchor_state(source_head_status: str) -> str:
    normalized = text(source_head_status)
    if normalized == "challenger_anchor_active":
        return "challenger_promoted"
    if normalized == "baseline_anchor_active":
        return "baseline_retained"
    if normalized == "upstream_hold_selection_conflict":
        return "blocked_by_upstream_hold_selection"
    return "pending"


def aligned_review_lane_ready(
    *,
    source_head_status: str,
    hold_selection_active_baseline: str,
    aligned_review_lane: dict[str, Any],
) -> bool:
    if text(source_head_status) != "upstream_hold_selection_conflict":
        return False
    if (
        text(aligned_review_lane.get("research_decision"))
        != "hold_selection_aligned_break_even_review_lane_ready_but_canonical_handoff_conflict_remains"
    ):
        return False
    if (
        text(aligned_review_lane.get("review_conclusion_research_decision"))
        != "break_even_review_conclusion_ready_keep_baseline_anchor_review_only"
    ):
        return False
    if text(aligned_review_lane.get("review_conclusion_arbitration_state")) != "review_only":
        return False
    if (
        text(aligned_review_lane.get("primary_anchor_review_research_decision"))
        != "break_even_primary_anchor_review_complete_keep_baseline_anchor"
    ):
        return False
    if text(aligned_review_lane.get("hold_selection_active_baseline")) != text(hold_selection_active_baseline):
        return False
    return bool(text(aligned_review_lane.get("active_baseline")))


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    exit_risk_path = Path(args.exit_risk_path).expanduser().resolve()
    forward_blocker_path = Path(args.forward_blocker_path).expanduser().resolve()
    forward_consensus_path = Path(args.forward_consensus_path).expanduser().resolve()
    break_even_sidecar_path = Path(args.break_even_sidecar_path).expanduser().resolve()
    tail_capacity_path = Path(args.tail_capacity_path).expanduser().resolve()

    exit_risk = load_json_mapping(exit_risk_path)
    forward_blocker = load_json_mapping(forward_blocker_path)
    forward_consensus = load_json_mapping(forward_consensus_path)
    break_even_sidecar = load_json_mapping(break_even_sidecar_path)
    tail_capacity = load_json_mapping(tail_capacity_path)
    hold_selection_handoff_path = review_dir / "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json"
    hold_selection_handoff = (
        load_json_mapping(hold_selection_handoff_path)
        if hold_selection_handoff_path.is_file()
        else {}
    )
    aligned_review_lane_path = (
        review_dir / "latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json"
    )
    aligned_review_lane = (
        load_json_mapping(aligned_review_lane_path)
        if aligned_review_lane_path.is_file()
        else {}
    )

    symbol = (
        text(exit_risk.get("symbol"))
        or text(forward_blocker.get("symbol"))
        or text(forward_consensus.get("symbol"))
        or text(break_even_sidecar.get("symbol"))
        or text(tail_capacity.get("symbol"))
        or "ETHUSDT"
    )

    selected_exit_params = normalize_exit_params(exit_risk.get("selected_exit_params"))
    validation_leader_exit_params = normalize_exit_params(exit_risk.get("validation_leader_exit_params"))
    challenge_pair = dict(forward_blocker.get("challenge_pair") or {})
    blocker_baseline_params = normalize_exit_params(challenge_pair.get("baseline_exit_params") or validation_leader_exit_params)
    blocker_challenger_params = normalize_exit_params(challenge_pair.get("challenger_exit_params") or selected_exit_params)

    active_baseline = text(break_even_sidecar.get("active_baseline")) or format_exit_risk_anchor_slug(blocker_baseline_params)
    watch_candidate = text(break_even_sidecar.get("watch_candidate"))
    superseded_anchor = ""
    transfer_watch: list[str] = []
    blocked_now = list(forward_blocker.get("blocked_now") or [])
    allowed_now = list(forward_blocker.get("allowed_now") or [])
    unique_extend(blocked_now, list(forward_consensus.get("blocked_now") or []))
    unique_extend(allowed_now, list(forward_consensus.get("allowed_now") or []))

    forward_blocker_decision = text(forward_blocker.get("research_decision"))
    forward_consensus_decision = text(forward_consensus.get("research_decision"))
    research_decision = "exit_risk_handoff_inconclusive"
    source_head_status = "inconclusive"
    next_research_priority = (
        text(forward_blocker.get("next_research_priority"))
        or text(forward_consensus.get("next_research_priority"))
        or "exit_risk_anchor_pending_refresh"
    )

    if (
        forward_blocker_decision == "exit_risk_forward_blocker_cleared_promote_challenger_pair"
        and forward_consensus_decision == "challenger_pair_promotable_across_current_forward_oos"
    ):
        research_decision = "use_exit_risk_handoff_as_canonical_anchor"
        source_head_status = "challenger_anchor_active"
        active_baseline = format_exit_risk_anchor_slug(blocker_challenger_params)
        superseded_anchor = text(break_even_sidecar.get("active_baseline")) or format_exit_risk_anchor_slug(blocker_baseline_params)
        transfer_watch = build_transfer_watch(allowed_now)
        blocked_now = []
        next_research_priority = (
            text(forward_consensus.get("next_research_priority"))
            or text(forward_blocker.get("next_research_priority"))
            or "refresh_exit_risk_anchor_after_forward_oos_promotion"
        )
    elif (
        forward_blocker_decision.startswith("block_exit_risk_promotion_keep_baseline_anchor_pair_")
        and forward_consensus_decision == "baseline_pair_keeps_anchor_across_current_forward_oos"
    ):
        research_decision = "use_exit_risk_handoff_as_canonical_anchor"
        source_head_status = "baseline_anchor_active"
        active_baseline = text(break_even_sidecar.get("active_baseline")) or format_exit_risk_anchor_slug(blocker_baseline_params)
        transfer_watch = build_transfer_watch(allowed_now)
        next_research_priority = derive_baseline_follow_up_priority(
            break_even_sidecar_decision=text(break_even_sidecar.get("research_decision")),
            break_even_sidecar_confidence_tier=text(break_even_sidecar.get("confidence_tier")),
            tail_capacity_decision=text(tail_capacity.get("research_decision")),
            fallback=(
            text(forward_consensus.get("next_research_priority"))
            or text(forward_blocker.get("next_research_priority"))
            or "forward_oos_follow_up_pending"
            ),
        )

    active_baseline_hold_bars = extract_hold_bars(active_baseline)
    hold_selection_active_baseline = text(hold_selection_handoff.get("active_baseline"))
    hold_selection_active_hold_bars = extract_hold_bars(hold_selection_active_baseline)
    upstream_hold_alignment_state = "unavailable"
    if active_baseline_hold_bars > 0 and hold_selection_active_hold_bars > 0:
        upstream_hold_alignment_state = (
            "aligned"
            if active_baseline_hold_bars == hold_selection_active_hold_bars
            else "conflict"
        )
    elif active_baseline_hold_bars > 0:
        upstream_hold_alignment_state = "hold_selection_unavailable"

    if (
        research_decision == "use_exit_risk_handoff_as_canonical_anchor"
        and source_head_status in {"baseline_anchor_active", "challenger_anchor_active"}
        and upstream_hold_alignment_state == "conflict"
    ):
        research_decision = "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
        source_head_status = "upstream_hold_selection_conflict"
        unique_extend(
            allowed_now,
            [
                "keep_hold_selection_anchor_as_upstream_mainline_gate",
                "keep_exit_risk_candidate_as_domain_local_review_only",
            ],
        )
        unique_extend(
            blocked_now,
            [
                "promote_exit_risk_anchor_that_conflicts_with_hold_selection_baseline",
            ],
        )
        next_research_priority = "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review"

    upstream_conflict_review_only_state = "unavailable"
    if aligned_review_lane_ready(
        source_head_status=source_head_status,
        hold_selection_active_baseline=hold_selection_active_baseline,
        aligned_review_lane=aligned_review_lane,
    ):
        upstream_conflict_review_only_state = "ready"
        unique_extend(
            allowed_now,
            [
                "consume_hold_selection_aligned_break_even_review_lane_as_review_only_evidence",
            ],
        )

    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.md"
    canonical_source_head = str(latest_json_path)

    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_handoff_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "exit_risk_path": str(exit_risk_path),
        "forward_blocker_path": str(forward_blocker_path),
        "forward_consensus_path": str(forward_consensus_path),
        "break_even_sidecar_path": str(break_even_sidecar_path),
        "tail_capacity_path": str(tail_capacity_path),
        "canonical_source_head": canonical_source_head,
        "superseded_head": str(forward_consensus_path),
        "source_head_status": source_head_status,
        "source_evidence": {
            "exit_risk_research_decision": text(exit_risk.get("research_decision")),
            "forward_blocker_research_decision": forward_blocker_decision,
            "forward_consensus_research_decision": forward_consensus_decision,
            "break_even_sidecar_research_decision": text(break_even_sidecar.get("research_decision")),
            "break_even_sidecar_confidence_tier": text(break_even_sidecar.get("confidence_tier")),
            "tail_capacity_research_decision": text(tail_capacity.get("research_decision")),
            "hold_selection_research_decision": text(hold_selection_handoff.get("research_decision")),
            "upstream_hold_alignment_state": upstream_hold_alignment_state,
        },
        "active_baseline": active_baseline,
        "active_baseline_hold_bars": active_baseline_hold_bars,
        "watch_candidate": watch_candidate,
        "superseded_anchor": superseded_anchor,
        "transfer_watch": transfer_watch,
        "hold_selection_handoff_path": str(hold_selection_handoff_path) if hold_selection_handoff_path.is_file() else "",
        "hold_selection_active_baseline": hold_selection_active_baseline,
        "hold_selection_active_hold_bars": hold_selection_active_hold_bars,
        "upstream_hold_alignment_state": upstream_hold_alignment_state,
        "upstream_conflict_review_only_state": upstream_conflict_review_only_state,
        "aligned_review_lane_path": str(aligned_review_lane_path) if aligned_review_lane_path.is_file() else "",
        "aligned_review_lane_research_decision": text(aligned_review_lane.get("research_decision")),
        "aligned_review_lane_active_baseline": text(aligned_review_lane.get("active_baseline")),
        "aligned_review_lane_preferred_watch_candidate": text(aligned_review_lane.get("preferred_watch_candidate")),
        "aligned_review_lane_review_conclusion_research_decision": text(
            aligned_review_lane.get("review_conclusion_research_decision")
        ),
        "aligned_review_lane_primary_anchor_review_research_decision": text(
            aligned_review_lane.get("primary_anchor_review_research_decision")
        ),
        "baseline_windows": int(forward_consensus.get("baseline_windows") or 0),
        "tie_windows": int(forward_consensus.get("tie_windows") or 0),
        "challenger_windows": int(forward_consensus.get("challenger_windows") or 0),
        "blocked_now": blocked_now,
        "allowed_now": allowed_now,
        "next_research_priority": next_research_priority,
        "consumer_rule": (
            "后续所有 ETH exit/risk brief / review / consumer 必须先读取 "
            "`latest_price_action_breakout_pullback_exit_risk_handoff_sim_only.json`；"
            "不得再手工拼 exit_risk + forward_blocker + forward_consensus + sidecar + tail_capacity。"
            "若 upstream_hold_alignment_state=conflict，则必须先尊重 "
            "`latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json` 的上游 hold 主线，"
            "不得把当前 exit/risk anchor 当成可直接提升的 canonical mainline。"
            "若 upstream_conflict_review_only_state=ready，则还必须读取 "
            "`latest_price_action_breakout_pullback_exit_risk_hold_selection_aligned_break_even_review_lane_sim_only.json` "
            "作为 review-only bridge 证据。"
        ),
        "handoff_state": {
            "anchor_state": derive_anchor_state(source_head_status),
            "watch_scope": "55d_plus_tie_windows" if transfer_watch else "none",
            "tail_capacity_scope": text(tail_capacity.get("research_decision")),
            "upstream_hold_alignment_state": upstream_hold_alignment_state,
            "upstream_conflict_review_only_state": upstream_conflict_review_only_state,
        },
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:exit_risk_handoff:"
            f"anchor={active_baseline},"
            f"superseded={display_superseded_anchor(superseded_anchor)},"
            f"watch={watch_candidate or ','.join(transfer_watch) or 'none'},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 handoff 把 exit/risk 的旧 base artifact、forward blocker、forward consensus、break-even sidecar 与 tail capacity "
            "收口成单一 canonical 读取入口。"
        ),
        "limitation_note": (
            "handoff 不新增 OOS 绩效；若后续刷新更长窗 forward compare 或 tail capacity，"
            "必须重新构建 handoff 才能保持 consumer 不漂移。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_handoff_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_handoff_sim_only.md"
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
