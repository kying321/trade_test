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
        description="Build a SIM_ONLY blocker artifact for the next ETH exit/risk forward OOS step."
    )
    parser.add_argument("--exit-risk-path", required=True)
    parser.add_argument("--hold-forward-stop-path", required=True)
    parser.add_argument("--forward-consensus-path", default="")
    parser.add_argument("--break-even-sidecar-path", default="")
    parser.add_argument("--tail-capacity-path", default="")
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


def normalize_exit_params(payload: Any) -> dict[str, Any]:
    data = dict(payload or {})
    return {
        "max_hold_bars": int(data.get("max_hold_bars") or 0),
        "break_even_trigger_r": float(data.get("break_even_trigger_r") or 0.0),
        "trailing_stop_atr": float(data.get("trailing_stop_atr") or 0.0),
        "cooldown_after_losses": int(data.get("cooldown_after_losses") or 0),
        "cooldown_bars": int(data.get("cooldown_bars") or 0),
    }


def classify_forward_blocker_decision(
    *,
    exit_risk_decision: str,
    hold_forward_stop_decision: str,
    selected_exit_params: dict[str, Any],
    validation_leader_exit_params: dict[str, Any],
) -> str:
    shared_trailing = float(selected_exit_params.get("trailing_stop_atr") or 0.0) == float(
        validation_leader_exit_params.get("trailing_stop_atr") or 0.0
    )
    hold_pair_split = int(selected_exit_params.get("max_hold_bars") or 0) != int(
        validation_leader_exit_params.get("max_hold_bars") or 0
    )
    hold_forward_stop_value = text(hold_forward_stop_decision)
    is_overlap_watch_stop = (
        hold_forward_stop_value.startswith("stop_")
        and "_hold_forward_mainline_keep_overlap_watch_" in hold_forward_stop_value
    )
    is_hold_forward_mainline_stop = (
        hold_forward_stop_value.startswith("stop_")
        and "_hold_forward_mainline" in hold_forward_stop_value
    )
    if (
        is_overlap_watch_stop
        and text(exit_risk_decision)
        in {
            "selected_exit_risk_improves_but_train_first_validation_diverges",
            "validation_leader_improves_train_first_selected_not_promoted",
        }
        and shared_trailing
        and hold_pair_split
    ):
        return "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos"
    if is_hold_forward_mainline_stop:
        return "exit_risk_forward_oos_required_before_promotion"
    return "exit_risk_forward_blocker_inconclusive"


def build_challenge_pair(
    *,
    selected_exit_params: dict[str, Any],
    validation_leader_exit_params: dict[str, Any],
) -> dict[str, Any]:
    baseline = dict(validation_leader_exit_params)
    challenger = dict(selected_exit_params)
    return {
        "baseline_exit_params": baseline,
        "challenger_exit_params": challenger,
        "primary_axis": "max_hold_bars",
        "baseline_hold_bars": int(baseline.get("max_hold_bars") or 0),
        "challenger_hold_bars": int(challenger.get("max_hold_bars") or 0),
        "shared_trailing_stop_atr": float(baseline.get("trailing_stop_atr") or 0.0),
    }


def format_challenge_pair_slug(pair: dict[str, Any]) -> str:
    challenger_hold_bars = int(pair.get("challenger_hold_bars") or 0)
    baseline_hold_bars = int(pair.get("baseline_hold_bars") or 0)
    return f"hold{challenger_hold_bars}_vs_hold{baseline_hold_bars}"


def format_trailing_action_slug(value: float) -> str:
    return f"{float(value or 0.0):.1f}".replace(".", "_")


def format_blocked_pair_research_decision(pair: dict[str, Any]) -> str:
    return f"block_exit_risk_promotion_require_forward_oos_pair_{format_challenge_pair_slug(pair)}"


def format_keep_baseline_pair_research_decision(pair: dict[str, Any]) -> str:
    return f"block_exit_risk_promotion_keep_baseline_anchor_pair_{format_challenge_pair_slug(pair)}"


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
        "# Price Action Breakout Pullback Exit Risk Forward Blocker SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- next_research_priority: `{text(payload.get('next_research_priority'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Challenge Pair",
        "",
    ]
    pair = dict(payload.get("challenge_pair") or {})
    lines.extend(
        [
            f"- baseline_hold_bars: `{pair.get('baseline_hold_bars')}`",
            f"- challenger_hold_bars: `{pair.get('challenger_hold_bars')}`",
            f"- shared_trailing_stop_atr: `{pair.get('shared_trailing_stop_atr')}`",
            "",
            "## Blocked Now",
            "",
        ]
    )
    for row in payload.get("blocked_now", []):
        lines.append(f"- `{text(row)}`")
    lines.extend(["", "## Allowed Now", ""])
    for row in payload.get("allowed_now", []):
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

    exit_risk_path = Path(args.exit_risk_path).expanduser().resolve()
    hold_forward_stop_path = Path(args.hold_forward_stop_path).expanduser().resolve()
    forward_consensus_path = Path(args.forward_consensus_path).expanduser().resolve() if text(args.forward_consensus_path) else None
    break_even_sidecar_path = (
        Path(args.break_even_sidecar_path).expanduser().resolve() if text(args.break_even_sidecar_path) else None
    )
    tail_capacity_path = Path(args.tail_capacity_path).expanduser().resolve() if text(args.tail_capacity_path) else None
    exit_risk = load_json_mapping(exit_risk_path)
    hold_forward_stop = load_json_mapping(hold_forward_stop_path)
    forward_consensus = load_json_mapping(forward_consensus_path) if forward_consensus_path else {}
    break_even_sidecar = load_json_mapping(break_even_sidecar_path) if break_even_sidecar_path else {}
    tail_capacity = load_json_mapping(tail_capacity_path) if tail_capacity_path else {}

    symbol = text(exit_risk.get("symbol")) or text(hold_forward_stop.get("symbol")) or "ETHUSDT"
    exit_risk_decision = text(exit_risk.get("research_decision"))
    hold_forward_stop_decision = text(hold_forward_stop.get("research_decision"))
    forward_consensus_decision = text(forward_consensus.get("research_decision"))
    break_even_sidecar_decision = text(break_even_sidecar.get("research_decision"))
    tail_capacity_decision = text(tail_capacity.get("research_decision"))
    selected_exit_params = normalize_exit_params(exit_risk.get("selected_exit_params"))
    validation_leader_exit_params = normalize_exit_params(exit_risk.get("validation_leader_exit_params"))
    challenge_pair = build_challenge_pair(
        selected_exit_params=selected_exit_params,
        validation_leader_exit_params=validation_leader_exit_params,
    )

    base_research_decision = classify_forward_blocker_decision(
        exit_risk_decision=exit_risk_decision,
        hold_forward_stop_decision=hold_forward_stop_decision,
        selected_exit_params=selected_exit_params,
        validation_leader_exit_params=validation_leader_exit_params,
    )
    research_decision = (
        format_blocked_pair_research_decision(challenge_pair)
        if base_research_decision == "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos"
        else base_research_decision
    )

    blocked_now: list[str] = []
    allowed_now: list[str] = []
    next_research_priority = text(hold_forward_stop.get("next_research_priority")) or "exit_risk_forward_oos"
    consumer_rule = (
        "在 hold forward stop-condition 生效后，不允许直接晋级 exit/risk selected 或 validation leader；"
        "必须先完成 forward OOS 挑战对验证。"
    )

    if forward_consensus_decision == "challenger_pair_promotable_across_current_forward_oos":
        research_decision = "exit_risk_forward_blocker_cleared_promote_challenger_pair"
        allowed_now = list(forward_consensus.get("allowed_now") or ["promote_challenger_pair_as_new_exit_risk_anchor"])
        next_research_priority = (
            text(forward_consensus.get("next_research_priority")) or "refresh_exit_risk_anchor_after_forward_oos_promotion"
        )
        consumer_rule = (
            "forward OOS 已完成且 challenger 已具备晋级条件；"
            "允许刷新 exit/risk anchor，但 55d+ tie 窗口仍只保留为 watch 证据。"
        )
    elif (
        forward_consensus_decision == "baseline_pair_keeps_anchor_across_current_forward_oos"
        and base_research_decision == "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos"
    ):
        research_decision = format_keep_baseline_pair_research_decision(challenge_pair)
        blocked_now = list(forward_consensus.get("blocked_now") or ["promote_challenger_pair_as_new_exit_risk_anchor"])
        allowed_now = list(forward_consensus.get("allowed_now") or ["keep_baseline_pair_as_current_exit_risk_anchor"])
        next_research_priority = derive_baseline_follow_up_priority(
            break_even_sidecar_decision=break_even_sidecar_decision,
            break_even_sidecar_confidence_tier=text(break_even_sidecar.get("confidence_tier")),
            tail_capacity_decision=tail_capacity_decision,
            fallback=text(forward_consensus.get("next_research_priority")) or "forward_oos_follow_up_pending",
        )
        consumer_rule = (
            "当前 challenge pair 的 forward OOS 已完成且 baseline 继续保持锚点；"
            "在 fresh challenger 或新增长尾证据出现前，不得把 challenger 晋级为新的 exit/risk anchor。"
        )
    elif base_research_decision == "block_exit_risk_promotion_require_hold8_vs_hold16_forward_oos":
        pair_slug = format_challenge_pair_slug(challenge_pair)
        trailing_slug = format_trailing_action_slug(float(challenge_pair.get("shared_trailing_stop_atr") or 0.0))
        blocked_now = [
            "promote_selected_exit_risk_config_without_forward_oos",
            "promote_validation_leader_exit_risk_config_without_forward_oos",
        ]
        allowed_now = [
            f"run_exit_risk_forward_oos_{pair_slug}_under_trailing_{trailing_slug}",
            "keep_break_even_delta_as_watch_sidecar_until_forward_oos_resolves",
        ]
        next_research_priority = f"exit_risk_forward_oos_{pair_slug}"

    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbol,
        "family": "price_action_breakout_pullback",
        "exit_risk_path": str(exit_risk_path),
        "hold_forward_stop_path": str(hold_forward_stop_path),
        "forward_consensus_path": str(forward_consensus_path) if forward_consensus_path else "",
        "break_even_sidecar_path": str(break_even_sidecar_path) if break_even_sidecar_path else "",
        "tail_capacity_path": str(tail_capacity_path) if tail_capacity_path else "",
        "source_evidence": {
            "exit_risk_research_decision": exit_risk_decision,
            "hold_forward_stop_research_decision": hold_forward_stop_decision,
            "forward_consensus_research_decision": forward_consensus_decision,
            "break_even_sidecar_research_decision": break_even_sidecar_decision,
            "break_even_sidecar_confidence_tier": text(break_even_sidecar.get("confidence_tier")),
            "tail_capacity_research_decision": tail_capacity_decision,
        },
        "challenge_pair": challenge_pair,
        "blocked_now": blocked_now,
        "allowed_now": allowed_now,
        "next_research_priority": next_research_priority,
        "research_decision": research_decision,
        "consumer_rule": consumer_rule,
        "recommended_brief": (
            f"{symbol}:exit_risk_forward_blocker:"
            f"pair={format_challenge_pair_slug(challenge_pair)},"
            f"trail={float(challenge_pair.get('shared_trailing_stop_atr') or 0.0):.1f},"
            f"next={next_research_priority},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 blocker 负责把 exit/risk forward OOS 前后的门控状态收口成单一工件，"
            "避免 consumer 同时读取旧 blocker 与新 consensus。"
        ),
        "limitation_note": (
            "它不新增 OOS 绩效；若 forward consensus 缺失或未晋级，"
            "仍以 blocker/consensus 原工件为准。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_blocker_sim_only.md"
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
