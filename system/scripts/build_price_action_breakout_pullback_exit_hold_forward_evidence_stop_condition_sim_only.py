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
        description="Build a SIM_ONLY stop-condition artifact for ETH hold forward evidence extension."
    )
    parser.add_argument("--forward-capacity-path", required=True)
    parser.add_argument("--overlap-sidecar-path", required=True)
    parser.add_argument("--handoff-path", required=True)
    parser.add_argument("--window-consensus-path", required=True)
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


def first_text(*values: Any) -> str:
    for value in values:
        item = text(value)
        if item:
            return item
    return ""


def stop_boundary_scope_slug(forward_capacity_decision: str) -> str:
    decision = text(forward_capacity_decision)
    if decision == "non_overlapping_forward_capacity_limited_but_usable":
        return "capacity_limited"
    if "45d_plus_insufficient" in decision:
        return "45d_plus"
    return "capacity_unspecified"


def stop_boundary_rule_token(forward_capacity_decision: str) -> str:
    scope_slug = stop_boundary_scope_slug(forward_capacity_decision)
    if scope_slug == "capacity_limited":
        return "use_capacity_limited_non_overlap_forward_compare_as_main_evidence"
    if scope_slug == "45d_plus":
        return "use_45d_plus_non_overlap_forward_compare_as_main_evidence"
    return "use_non_overlap_forward_compare_as_main_evidence"


def stop_boundary_human_label(forward_capacity_decision: str) -> str:
    scope_slug = stop_boundary_scope_slug(forward_capacity_decision)
    if scope_slug == "capacity_limited":
        return "capacity-limited non-overlap"
    if scope_slug == "45d_plus":
        return "45d+ non-overlap"
    return "non-overlap stop boundary"


def is_stop_review_long_window_regime_split_decision(decision: str) -> bool:
    value = text(decision)
    return value.startswith("stop_") and "_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split" in value


def is_stop_keep_overlap_shift_exit_risk_decision(decision: str) -> bool:
    value = text(decision)
    return value.startswith("stop_") and "_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos" in value


def is_stop_shift_exit_risk_decision(decision: str) -> bool:
    value = text(decision)
    return value.startswith("stop_") and "_hold_forward_mainline_shift_exit_risk_oos" in value


def classify_stop_condition_decision(
    *,
    forward_capacity_decision: str,
    overlap_sidecar_decision: str,
    handoff_decision: str,
    window_consensus_decision: str,
) -> str:
    scope_slug = stop_boundary_scope_slug(forward_capacity_decision)
    capacity_requires_stop_boundary = (
        "45d_plus_insufficient" in text(forward_capacity_decision)
        or text(forward_capacity_decision) == "non_overlapping_forward_capacity_limited_but_usable"
    )
    overlap_keeps_non_overlapping_baseline = "keep_non_overlapping_baseline" in text(overlap_sidecar_decision)
    if (
        capacity_requires_stop_boundary
        and overlap_keeps_non_overlapping_baseline
        and text(handoff_decision) == "use_hold_selection_gate_as_canonical_head"
    ):
        if "watch_long_window_regime_split" in text(window_consensus_decision):
            return f"stop_{scope_slug}_hold_forward_mainline_keep_overlap_watch_review_long_window_regime_split"
        return f"stop_{scope_slug}_hold_forward_mainline_keep_overlap_watch_shift_exit_risk_oos"
    if capacity_requires_stop_boundary:
        return f"stop_{scope_slug}_hold_forward_mainline_shift_exit_risk_oos"
    return "hold_forward_evidence_stop_condition_inconclusive"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Hold Forward Evidence Stop Condition SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- next_research_priority: `{text(payload.get('next_research_priority'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Blocked Now",
        "",
    ]
    for row in payload.get("blocked_now", []):
        lines.append(f"- `{text(row)}`")
    lines.extend(["", "## Watch Only", ""])
    for row in payload.get("watch_only", []):
        lines.append(f"- `{text(row)}`")
    lines.extend(
        [
            "",
            "## Consumer Rule",
            "",
            f"- `{text(payload.get('consumer_rule'))}`",
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

    forward_capacity_path = Path(args.forward_capacity_path).expanduser().resolve()
    overlap_sidecar_path = Path(args.overlap_sidecar_path).expanduser().resolve()
    handoff_path = Path(args.handoff_path).expanduser().resolve()
    window_consensus_path = Path(args.window_consensus_path).expanduser().resolve()

    forward_capacity = load_json_mapping(forward_capacity_path)
    overlap_sidecar = load_json_mapping(overlap_sidecar_path)
    handoff = load_json_mapping(handoff_path)
    window_consensus = load_json_mapping(window_consensus_path)

    symbol = first_text(
        forward_capacity.get("symbol"),
        overlap_sidecar.get("symbol"),
        handoff.get("symbol"),
        window_consensus.get("symbol"),
        "ETHUSDT",
    )
    forward_capacity_decision = text(forward_capacity.get("research_decision"))
    overlap_sidecar_decision = text(overlap_sidecar.get("research_decision"))
    handoff_decision = text(handoff.get("research_decision"))
    window_consensus_decision = text(window_consensus.get("research_decision"))
    stop_boundary_scope = stop_boundary_scope_slug(forward_capacity_decision)
    stop_boundary_rule = stop_boundary_rule_token(forward_capacity_decision)
    stop_boundary_label = stop_boundary_human_label(forward_capacity_decision)
    research_decision = classify_stop_condition_decision(
        forward_capacity_decision=forward_capacity_decision,
        overlap_sidecar_decision=overlap_sidecar_decision,
        handoff_decision=handoff_decision,
        window_consensus_decision=window_consensus_decision,
    )

    blocked_now: list[str] = []
    watch_only: list[str] = []
    next_research_priority = "hold_forward_compare_review_pending"
    consumer_rule = (
        "后续 consumer 必须先读 stop-condition，再决定是否继续扩 hold forward compare 主证据。"
    )
    research_note = (
        "这份工件把 hold forward compare 的停止边界独立固化为 source-owned 证据："
        f"{stop_boundary_label} 不再继续当主证据，35d/40d overlap 只保留 watch。"
    )
    if is_stop_review_long_window_regime_split_decision(research_decision):
        blocked_now = [
            stop_boundary_rule,
            "extend_hold_forward_train_window_beyond_40d_without_fresh_non_overlap_capacity",
            "promote_hold8_over_hold16_without_long_window_regime_split_review",
        ]
        watch_only = [
            "treat_35d_40d_overlapping_compare_as_watch_sidecar_only",
            "treat_long_window_regime_split_as_review_gate_only",
        ]
        next_research_priority = "review_long_window_regime_split_before_baseline_promotion"
        consumer_rule = (
            f"停止继续把 {stop_boundary_label} hold forward compare 当作主证据；"
            "35d/40d overlapping compare 只能保留为 watch sidecar；"
            "在任何 hold baseline promotion 之前，必须先完成 long-window regime split review。"
        )
        research_note = (
            "这份工件把 hold forward compare 的停止边界与 window consensus 已确认的 long-window regime split 一并固化："
            f"{stop_boundary_label} 继续 blocked，35d/40d overlap 继续 watch-only，"
            "下一步不是回到泛化 compare pending，而是先复核 long-window regime split。"
        )
    elif is_stop_keep_overlap_shift_exit_risk_decision(research_decision):
        blocked_now = [
            stop_boundary_rule,
            "extend_hold_forward_train_window_beyond_40d_without_fresh_non_overlap_capacity",
        ]
        watch_only = ["treat_35d_40d_overlapping_compare_as_watch_sidecar_only"]
        next_research_priority = "exit_risk_forward_oos"
        consumer_rule = (
            f"停止继续把 {stop_boundary_label} hold forward compare 当作主证据；"
            "35d/40d overlapping compare 只能保留为 watch sidecar；"
            "下一步转向 exit/risk OOS，而不是继续扩 hold train window。"
        )
    elif is_stop_shift_exit_risk_decision(research_decision):
        blocked_now = [stop_boundary_rule]
        next_research_priority = "exit_risk_forward_oos"

    payload = {
        "action": "build_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "forward_capacity_path": str(forward_capacity_path),
        "overlap_sidecar_path": str(overlap_sidecar_path),
        "handoff_path": str(handoff_path),
        "window_consensus_path": str(window_consensus_path),
        "source_evidence": {
            "forward_capacity_research_decision": forward_capacity_decision,
            "overlap_sidecar_research_decision": overlap_sidecar_decision,
            "handoff_research_decision": handoff_decision,
            "window_consensus_research_decision": window_consensus_decision,
        },
        "blocked_now": blocked_now,
        "watch_only": watch_only,
        "next_research_priority": next_research_priority,
        "consumer_rule": consumer_rule,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:exit_hold_forward_stop:"
            f"{stop_boundary_scope}={'blocked' if blocked_now else 'open'},"
            f"overlap_35d_40d={'watch_only' if watch_only else 'unspecified'},"
            f"window_consensus={window_consensus_decision or 'unspecified'},"
            f"next={next_research_priority},"
            f"decision={research_decision}"
        ),
        "research_note": research_note,
        "limitation_note": (
            "它不新增任何新的 OOS 绩效，只是把 capacity / overlap / handoff 已经达成的边界显式化；"
            "若未来数据尾部扩展，仍需 fresh non-overlap 工件重新验证。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_evidence_stop_condition_sim_only.md"
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
