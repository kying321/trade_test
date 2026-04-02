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
        description="Build a SIM_ONLY consensus artifact for exit/risk forward compare windows."
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


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def build_window_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload.get("comparison_summary") or {})
    return {
        "train_days": int(payload.get("train_days") or 0),
        "validation_days": int(payload.get("validation_days") or 0),
        "step_days": int(payload.get("step_days") or 0),
        "research_decision": text(payload.get("research_decision")),
        "winner_by_aggregate_return": text(summary.get("winner_by_aggregate_return")),
        "winner_by_aggregate_objective": text(summary.get("winner_by_aggregate_objective")),
    }


def classify_forward_consensus(windows: list[dict[str, Any]]) -> str:
    counts = count_window_outcomes(windows)
    baseline_windows = counts["baseline_windows"]
    challenger_windows = counts["challenger_windows"]
    tie_windows = counts["tie_windows"]
    has_30d_baseline = any(int(row.get("train_days") or 0) == 30 for row in baseline_windows)
    has_40d_tie = any(int(row.get("train_days") or 0) == 40 for row in tie_windows)
    if has_30d_baseline and has_40d_tie and not challenger_windows:
        return "baseline_pair_keeps_anchor_challenger_not_promoted_across_30d_40d_forward_oos"
    if baseline_windows and not challenger_windows:
        return "baseline_pair_keeps_anchor_across_current_forward_oos"
    if challenger_windows and not baseline_windows:
        return "challenger_pair_promotable_across_current_forward_oos"
    return "exit_risk_forward_consensus_inconclusive"


def count_window_outcomes(windows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    baseline_windows = [
        row
        for row in windows
        if text(row.get("winner_by_aggregate_return")) == "baseline_pair"
        and text(row.get("winner_by_aggregate_objective")) == "baseline_pair"
    ]
    challenger_windows = [
        row
        for row in windows
        if text(row.get("winner_by_aggregate_return")) == "challenger_pair"
        and text(row.get("winner_by_aggregate_objective")) == "challenger_pair"
    ]
    tie_windows = [
        row
        for row in windows
        if text(row.get("winner_by_aggregate_return")) == "tie"
        and text(row.get("winner_by_aggregate_objective")) == "tie"
    ]
    return {
        "baseline_windows": baseline_windows,
        "challenger_windows": challenger_windows,
        "tie_windows": tie_windows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Risk Forward Consensus SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Window Summary",
        "",
    ]
    for row in payload.get("window_summary", []):
        lines.append(
            f"- train={row['train_days']}d val={row['validation_days']}d step={row['step_days']}d | "
            f"decision={row['research_decision']} | agg_ret={row['winner_by_aggregate_return']} | "
            f"agg_obj={row['winner_by_aggregate_objective']}"
        )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    compare_paths = [Path(raw).expanduser().resolve() for raw in args.compare_path]
    windows = [build_window_summary(load_json_mapping(path)) for path in compare_paths]
    windows = sorted(windows, key=lambda row: int(row.get("train_days") or 0))
    symbols = {
        text(load_json_mapping(path).get("symbol"))
        for path in compare_paths
        if text(load_json_mapping(path).get("symbol"))
    }
    if len(symbols) != 1:
        raise SystemExit("mixed_symbols_not_supported")

    challenge_decisions = {
        text(load_json_mapping(path).get("challenge_pair_source_decision"))
        for path in compare_paths
        if text(load_json_mapping(path).get("challenge_pair_source_decision"))
    }
    research_decision = classify_forward_consensus(windows)
    window_counts = count_window_outcomes(windows)
    baseline_windows = len(window_counts["baseline_windows"])
    tie_windows = len(window_counts["tie_windows"])
    challenger_windows = len(window_counts["challenger_windows"])
    blocked_now: list[str] = []
    allowed_now: list[str] = []
    next_research_priority = "forward_oos_follow_up_pending"
    if research_decision == "baseline_pair_keeps_anchor_challenger_not_promoted_across_30d_40d_forward_oos":
        blocked_now = ["promote_challenger_pair_as_new_exit_risk_anchor"]
        allowed_now = [
            "keep_baseline_pair_as_current_exit_risk_anchor",
            "treat_40d_tie_window_as_watch_only",
        ]
        next_research_priority = "collect_more_tail_or_test_break_even_sidecar_separately"
    elif research_decision == "baseline_pair_keeps_anchor_across_current_forward_oos":
        blocked_now = ["promote_challenger_pair_as_new_exit_risk_anchor"]
        allowed_now = ["keep_baseline_pair_as_current_exit_risk_anchor"]
        next_research_priority = "defer_to_canonical_exit_risk_blocker_or_handoff"
    elif research_decision == "challenger_pair_promotable_across_current_forward_oos":
        has_55d_plus_tie = any(int(row.get("train_days") or 0) >= 55 for row in window_counts["tie_windows"])
        allowed_now = ["promote_challenger_pair_as_new_exit_risk_anchor"]
        if has_55d_plus_tie:
            allowed_now.append("treat_55d_plus_tie_windows_as_watch_only")
        next_research_priority = "refresh_exit_risk_anchor_after_forward_oos_promotion"

    symbol = next(iter(symbols), "")
    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "symbol": symbol,
        "compare_paths": [str(path) for path in compare_paths],
        "challenge_pair_source_decisions": sorted(decision for decision in challenge_decisions if decision),
        "window_summary": windows,
        "blocked_now": blocked_now,
        "allowed_now": allowed_now,
        "next_research_priority": next_research_priority,
        "baseline_windows": baseline_windows,
        "tie_windows": tie_windows,
        "challenger_windows": challenger_windows,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:exit_risk_forward_consensus:"
            f"baseline_windows={baseline_windows},"
            f"tie_windows={tie_windows},"
            f"challenger_windows={challenger_windows},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 consensus 把多个 exit/risk forward compare 窗口收口成单一 canonical 结论，"
            "避免 consumer 手工拼多段 30d+ 工件。"
        ),
        "limitation_note": (
            "它不替代原始 forward compare 明细，也不替代 canonical blocker/handoff；"
            "若后续新增 35d/更多 tail 窗口，必须重新构建 consensus。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_consensus_sim_only.md"
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
