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
        description="Build a SIM_ONLY review packet from the break-even guarded-review gate and sidecar evidence."
    )
    parser.add_argument("--guarded-review-path", required=True)
    parser.add_argument("--handoff-path", required=True)
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


def build_window_summary(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in windows:
        summary.append(
            {
                "path": text(row.get("path")),
                "train_days": int(row.get("train_days") or 0),
                "validation_days": int(row.get("validation_days") or 0),
                "step_days": int(row.get("step_days") or 0),
                "validation_window_mode": text(row.get("validation_window_mode")),
                "slice_count": int(row.get("slice_count") or 0),
                "winner_by_aggregate_return": text(row.get("winner_by_aggregate_return")),
                "winner_by_aggregate_objective": text(row.get("winner_by_aggregate_objective")),
                "winner_by_slice_majority_return": text(row.get("winner_by_slice_majority_return")),
                "winner_by_slice_majority_objective": text(row.get("winner_by_slice_majority_objective")),
            }
        )
    return summary


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Risk Break Even Review Packet SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- packet_state: `{text(payload.get('packet_state'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Review Pair",
        "",
        f"- primary_anchor: `{text(payload.get('primary_anchor'))}`",
        f"- review_candidate: `{text(payload.get('review_candidate'))}`",
        f"- review_scope: `{text(payload.get('review_scope'))}`",
        f"- evidence_scope: `{text(payload.get('evidence_scope'))}`",
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
    lines.extend(["", "## Evidence Windows", ""])
    for row in payload.get("evidence_windows", []):
        lines.append(
            f"- train={int(row.get('train_days') or 0)} val={int(row.get('validation_days') or 0)} "
            f"step={int(row.get('step_days') or 0)} mode={text(row.get('validation_window_mode'))} "
            f"| agg_ret={text(row.get('winner_by_aggregate_return'))} "
            f"agg_obj={text(row.get('winner_by_aggregate_objective'))} "
            f"| slice_ret={text(row.get('winner_by_slice_majority_return'))} "
            f"slice_obj={text(row.get('winner_by_slice_majority_objective'))}"
        )
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

    guarded_review_path = Path(args.guarded_review_path).expanduser().resolve()
    handoff_path = Path(args.handoff_path).expanduser().resolve()
    break_even_sidecar_path = Path(args.break_even_sidecar_path).expanduser().resolve()

    guarded_review = load_json_mapping(guarded_review_path)
    handoff = load_json_mapping(handoff_path)
    break_even_sidecar = load_json_mapping(break_even_sidecar_path)

    symbol = (
        text(guarded_review.get("symbol"))
        or text(handoff.get("symbol"))
        or text(break_even_sidecar.get("symbol"))
        or "ETHUSDT"
    )
    primary_anchor = text(guarded_review.get("active_baseline")) or text(handoff.get("active_baseline"))
    review_candidate = text(guarded_review.get("watch_candidate")) or text(handoff.get("watch_candidate"))
    review_scope = text(guarded_review.get("review_scope")) or "same_hold_same_trailing_break_even_delta_only"
    evidence_scope = text(break_even_sidecar.get("evidence_scope"))
    source_artifacts = list(break_even_sidecar.get("source_artifacts") or [])
    evidence_windows = build_window_summary(list(break_even_sidecar.get("windows") or []))
    packet_state = "blocked"
    research_decision = "break_even_review_packet_blocked_by_guarded_review"
    allowed_now = ["repair_guarded_review_inputs_before_review_packet"]
    blocked_now = [
        "review_break_even_candidate_against_primary_forward_anchor",
        "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_canonical_alignment",
    ]
    next_research_priority = (
        text(guarded_review.get("next_research_priority"))
        or "repair_break_even_sidecar_and_canonical_handoff_alignment"
    )

    guarded_review_state = text(guarded_review.get("review_state"))
    guarded_review_decision = text(guarded_review.get("research_decision"))
    handoff_decision = text(handoff.get("research_decision"))
    handoff_source_head_status = text(handoff.get("source_head_status"))
    if (
        guarded_review_decision == "break_even_guarded_review_blocked_by_upstream_hold_selection_conflict"
        or handoff_decision == "exit_risk_handoff_blocked_upstream_hold_selection_anchor_conflict"
        or handoff_source_head_status == "upstream_hold_selection_conflict"
    ):
        packet_state = "blocked"
        research_decision = "break_even_review_packet_blocked_by_upstream_hold_selection_conflict"
        allowed_now = ["resolve_upstream_hold_selection_vs_exit_risk_anchor_conflict"]
        blocked_now = [
            "review_break_even_candidate_against_primary_forward_anchor",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_while_upstream_hold_selection_conflict_is_active",
        ]
        next_research_priority = (
            text(guarded_review.get("next_research_priority"))
            or "resolve_hold_selection_vs_exit_risk_anchor_conflict_before_break_even_review"
        )
    elif guarded_review_state == "ready":
        packet_state = "ready"
        research_decision = "break_even_review_packet_ready_for_primary_anchor_review"
        allowed_now = [
            "review_break_even_candidate_against_primary_forward_anchor",
            "keep_break_even_candidate_review_only_until_packet_arbitration",
        ]
        blocked_now = [
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_review_packet_arbitration",
        ]
        next_research_priority = "review_break_even_candidate_against_primary_forward_anchor"
    elif guarded_review_state == "watch_only":
        packet_state = "watch_only"
        research_decision = "break_even_review_packet_not_ready_keep_watch_only"
        allowed_now = [
            "keep_break_even_candidate_as_watch_sidecar_only",
            "wait_for_stronger_guarded_review_gate",
        ]
        blocked_now = [
            "review_break_even_candidate_against_primary_forward_anchor",
            "promote_break_even_watch_candidate_as_new_exit_risk_anchor_without_review_packet_arbitration",
        ]
        next_research_priority = (
            text(guarded_review.get("next_research_priority"))
            or "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"
        )

    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "guarded_review_path": str(guarded_review_path),
        "handoff_path": str(handoff_path),
        "break_even_sidecar_path": str(break_even_sidecar_path),
        "source_artifacts": source_artifacts,
        "primary_anchor": primary_anchor,
        "review_candidate": review_candidate,
        "packet_state": packet_state,
        "review_focus": "primary_forward_anchor_break_even_delta_review",
        "review_scope": review_scope,
        "evidence_scope": evidence_scope,
        "evidence_window_count": int(len(evidence_windows)),
        "evidence_windows": evidence_windows,
        "source_evidence": {
            "guarded_review_research_decision": guarded_review_decision,
            "guarded_review_state": guarded_review_state,
            "handoff_research_decision": handoff_decision,
            "handoff_source_head_status": handoff_source_head_status,
            "break_even_sidecar_research_decision": text(break_even_sidecar.get("research_decision")),
        },
        "allowed_now": allowed_now,
        "blocked_now": blocked_now,
        "next_research_priority": next_research_priority,
        "research_decision": research_decision,
        "consumer_rule": (
            "任何评审 brief / dashboard drilldown 若要展示 break-even candidate 评审证据，"
            "必须先读取 `latest_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.json`，"
            "不得直接从 sidecar windows 或 guarded review 文本拼装。"
        ),
        "recommended_brief": (
            f"{symbol}:exit_risk_break_even_review_packet:"
            f"anchor={primary_anchor or 'unknown'},"
            f"candidate={review_candidate or 'unknown'},"
            f"state={packet_state},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 review packet 只负责把 guarded-review gate 与 break-even sidecar 窗口证据打包成单一评审入口，"
            "供后续人工/系统审读，不直接改变 canonical anchor。"
        ),
        "limitation_note": (
            "它不新增 forward OOS 绩效，也不替代 guarded review gate；"
            "若 guarded review、handoff 或 sidecar 任何一项刷新，review packet 必须同步重建。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_review_packet_sim_only.md"
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
