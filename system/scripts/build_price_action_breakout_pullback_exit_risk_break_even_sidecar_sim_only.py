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
        description="Build a SIM_ONLY sidecar from ETH break-even forward compare windows without changing the current exit/risk anchor."
    )
    parser.add_argument("--compare-path", action="append", required=True)
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


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def evidence_scope(windows: list[dict[str, Any]]) -> str:
    modes = {text(row.get("validation_window_mode")) for row in windows if text(row.get("validation_window_mode"))}
    if modes == {"overlapping"}:
        return "overlapping_forward_compare_windows_only"
    if modes == {"non_overlapping"}:
        return "non_overlapping_forward_compare_windows_only"
    if len(modes) > 1:
        return "mixed_validation_window_modes"
    return "window_mode_unspecified"


def classify_sidecar_decision(windows: list[dict[str, Any]]) -> str:
    anchor_no_be_dual = any(
        text(row.get("winner_by_aggregate_return")) == "anchor_no_be"
        and text(row.get("winner_by_aggregate_objective")) == "anchor_no_be"
        for row in windows
    )
    anchor_with_be_dual = any(
        text(row.get("winner_by_aggregate_return")) == "anchor_with_be"
        and text(row.get("winner_by_aggregate_objective")) == "anchor_with_be"
        for row in windows
    )
    tie_dual = windows and all(
        text(row.get("winner_by_aggregate_return")) == "tie"
        and text(row.get("winner_by_aggregate_objective")) == "tie"
        for row in windows
    )
    if tie_dual:
        return "break_even_sidecar_no_observed_delta_keep_anchor"
    if anchor_no_be_dual and not anchor_with_be_dual:
        return "break_even_sidecar_not_promising_keep_anchor"
    if anchor_with_be_dual and not anchor_no_be_dual:
        return "break_even_sidecar_positive_watch_only"
    if anchor_no_be_dual and anchor_with_be_dual:
        return "break_even_sidecar_mixed_keep_anchor_watch_only"
    return "break_even_sidecar_inconclusive_keep_anchor"


def row_is_anchor_with_be_full_consensus(row: dict[str, Any]) -> bool:
    return (
        text(row.get("winner_by_aggregate_return")) == "anchor_with_be"
        and text(row.get("winner_by_aggregate_objective")) == "anchor_with_be"
        and text(row.get("winner_by_slice_majority_return")) == "anchor_with_be"
        and text(row.get("winner_by_slice_majority_objective")) == "anchor_with_be"
    )


def derive_confidence_tier(
    *,
    research_decision: str,
    scope: str,
    windows: list[dict[str, Any]],
) -> str:
    if text(research_decision) != "break_even_sidecar_positive_watch_only":
        return "not_applicable"
    if (
        text(scope) == "overlapping_forward_compare_windows_only"
        and len(windows) >= 3
        and windows
        and all(row_is_anchor_with_be_full_consensus(row) for row in windows)
    ):
        return "guarded_review_ready"
    return "watch_only"


def derive_next_research_priority(*, research_decision: str, confidence_tier: str) -> str:
    if text(research_decision) != "break_even_sidecar_positive_watch_only":
        return "merge_break_even_sidecar_with_primary_forward_evidence"
    if text(confidence_tier) == "guarded_review_ready":
        return "guarded_review_break_even_candidate_keep_baseline_anchor_defer_more_tail"
    return "watch_break_even_candidate_keep_baseline_anchor_defer_more_tail"


def build_limitation_note() -> str:
    return (
        "这是 sidecar 证据，不替代 canonical exit/risk 共识；"
        "即便后续结果继续偏正向，也只能作为 watch sidecar 与主前推证据合并审读，不能直接晋级 anchor。"
    )


def build_window(path: Path) -> dict[str, Any]:
    payload = load_json_mapping(path)
    summary = dict(payload.get("comparison_summary") or {})
    aggregate_metrics = dict(payload.get("aggregate_validation_metrics_by_config") or {})
    aggregate_objective = dict(payload.get("aggregate_validation_objective_by_config") or {})
    comparison_configs = {
        text(row.get("config_id")): normalize_exit_params(row.get("exit_params"))
        for row in payload.get("comparison_configs") or []
        if text(row.get("config_id"))
    }
    symbol = text(payload.get("symbol"))
    if not symbol:
        raise ValueError(f"missing_symbol:{path}")
    return {
        "path": str(path),
        "symbol": symbol,
        "train_days": int(payload.get("train_days") or 0),
        "validation_days": int(payload.get("validation_days") or 0),
        "step_days": int(payload.get("step_days") or 0),
        "validation_window_mode": text(payload.get("validation_window_mode")),
        "slice_count": int(payload.get("slice_count") or 0),
        "research_decision": text(payload.get("research_decision")),
        "winner_by_aggregate_return": text(summary.get("winner_by_aggregate_return")),
        "winner_by_aggregate_objective": text(summary.get("winner_by_aggregate_objective")),
        "winner_by_slice_majority_return": text(summary.get("winner_by_slice_majority_return")),
        "winner_by_slice_majority_objective": text(summary.get("winner_by_slice_majority_objective")),
        "aggregate_metrics_by_config": {
            key: {
                "cumulative_return": metrics.get("cumulative_return"),
                "max_drawdown": metrics.get("max_drawdown"),
                "trade_count": metrics.get("trade_count"),
                "avg_hold_bars": metrics.get("avg_hold_bars"),
                "objective": aggregate_objective.get(key),
            }
            for key, metrics in aggregate_metrics.items()
        },
        "comparison_configs": comparison_configs,
    }


def derive_anchor_slugs(windows: list[dict[str, Any]]) -> tuple[str, str]:
    for row in windows:
        comparison_configs = dict(row.get("comparison_configs") or {})
        baseline_params = normalize_exit_params(comparison_configs.get("anchor_no_be"))
        challenger_params = normalize_exit_params(comparison_configs.get("anchor_with_be"))
        if baseline_params.get("max_hold_bars") and challenger_params.get("max_hold_bars"):
            return (
                format_exit_risk_anchor_slug(baseline_params),
                format_exit_risk_anchor_slug(challenger_params),
            )
    return ("hold16_trail15_no_be", "hold16_trail15_be075")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Risk Break Even Sidecar SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- confidence_tier: `{text(payload.get('confidence_tier'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        f"- next_research_priority: `{text(payload.get('next_research_priority'))}`",
        f"- evidence_scope: `{text(payload.get('evidence_scope'))}`",
        f"- active_baseline: `{text(payload.get('active_baseline'))}`",
        f"- watch_candidate: `{text(payload.get('watch_candidate'))}`",
        f"- anchor_no_be_windows: `{int(payload.get('anchor_no_be_windows') or 0)}`",
        f"- anchor_with_be_windows: `{int(payload.get('anchor_with_be_windows') or 0)}`",
        f"- tie_windows: `{int(payload.get('tie_windows') or 0)}`",
        "",
        "## Window Summary",
        "",
    ]
    for row in payload.get("windows", []):
        lines.append(
            f"- train={row['train_days']} val={row['validation_days']} step={row['step_days']} "
            f"mode={row['validation_window_mode']} slices={row['slice_count']} | "
            f"agg_ret={row['winner_by_aggregate_return']} agg_obj={row['winner_by_aggregate_objective']} | "
            f"slice_ret={row['winner_by_slice_majority_return']} slice_obj={row['winner_by_slice_majority_objective']}"
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

    compare_paths = [Path(raw).expanduser().resolve() for raw in args.compare_path]
    if not compare_paths:
        raise SystemExit("missing_compare_paths")

    windows = [build_window(path) for path in compare_paths]
    symbols = sorted({text(row.get("symbol")) for row in windows})
    if len(symbols) != 1:
        raise SystemExit("mixed_symbols_not_supported")
    windows = sorted(windows, key=lambda row: (-int(row.get("train_days") or 0), int(row.get("validation_days") or 0), int(row.get("step_days") or 0)))

    anchor_no_be_windows = sum(
        1
        for row in windows
        if text(row.get("winner_by_aggregate_return")) == "anchor_no_be"
        and text(row.get("winner_by_aggregate_objective")) == "anchor_no_be"
    )
    anchor_with_be_windows = sum(
        1
        for row in windows
        if text(row.get("winner_by_aggregate_return")) == "anchor_with_be"
        and text(row.get("winner_by_aggregate_objective")) == "anchor_with_be"
    )
    tie_windows = sum(
        1
        for row in windows
        if text(row.get("winner_by_aggregate_return")) == "tie"
        and text(row.get("winner_by_aggregate_objective")) == "tie"
    )
    scope = evidence_scope(windows)
    research_decision = classify_sidecar_decision(windows)
    confidence_tier = derive_confidence_tier(research_decision=research_decision, scope=scope, windows=windows)
    next_research_priority = derive_next_research_priority(
        research_decision=research_decision,
        confidence_tier=confidence_tier,
    )
    active_baseline, watch_candidate = derive_anchor_slugs(windows)

    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbols[0],
        "family": "price_action_breakout_pullback",
        "source_artifacts": [row["path"] for row in windows],
        "evidence_scope": scope,
        "window_count": int(len(windows)),
        "windows": windows,
        "anchor_no_be_windows": int(anchor_no_be_windows),
        "anchor_with_be_windows": int(anchor_with_be_windows),
        "tie_windows": int(tie_windows),
        "research_decision": research_decision,
        "confidence_tier": confidence_tier,
        "promotion_review_ready": confidence_tier == "guarded_review_ready",
        "next_research_priority": next_research_priority,
        "active_baseline": active_baseline,
        "watch_candidate": watch_candidate,
        "consumer_rule": "do_not_override_exit_risk_anchor_without_fresh_primary_forward_confirmation",
        "recommended_brief": (
            f"{symbols[0]}:exit_risk_break_even_sidecar:"
            f"anchor_no_be={anchor_no_be_windows},"
            f"anchor_with_be={anchor_with_be_windows},"
            f"tie={tie_windows},"
            f"confidence={confidence_tier},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "只在当前 exit/risk anchor 上单独检查 break-even 是否值得进入 watch sidecar；"
            "它不参与新的 anchor 晋级，只补一层 source-owned 研究证据。"
        ),
        "limitation_note": build_limitation_note(),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_break_even_sidecar_sim_only.md"
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
