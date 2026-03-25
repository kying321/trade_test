#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
BASE_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_sim_only.py")
COMPARE_SCRIPT_PATH = Path(__file__).resolve().with_name(
    "build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py"
)


BASE_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_base", BASE_SCRIPT_PATH)
BASE_MODULE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE_MODULE)

COMPARE_SPEC = importlib.util.spec_from_file_location("fenlie_exit_hold_forward_compare", COMPARE_SCRIPT_PATH)
COMPARE_MODULE = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
COMPARE_SPEC.loader.exec_module(COMPARE_MODULE)


DEFAULT_CANDIDATE_TRAIN_DAYS = [20, 25, 30, 35, 40, 45, 50, 55, 60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY capacity artifact for ETH hold forward compare train windows."
    )
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--candidate-train-days", default="20,25,30,35,40,45,50,55,60")
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def parse_candidate_train_days(raw: str) -> list[int]:
    result: list[int] = []
    for token in text(raw).split(","):
        item = token.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("candidate_train_days_must_be_positive")
        result.append(value)
    if not result:
        raise ValueError("candidate_train_days_required")
    return sorted(dict.fromkeys(result))


def build_capacity_rows(
    frame,
    *,
    candidate_train_days: list[int],
    validation_days: int,
    step_days: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for train_days in candidate_train_days:
        try:
            slices = COMPARE_MODULE.build_forward_slices(
                frame,
                train_days=int(train_days),
                validation_days=int(validation_days),
                step_days=int(step_days),
            )
            slice_count = int(len(slices))
        except ValueError as exc:
            if text(str(exc)) not in {"dataset_too_small_for_forward_compare", "no_forward_slices_built"}:
                raise
            slice_count = 0
        rows.append({"train_days": int(train_days), "slice_count": int(slice_count)})
    return rows


def summarize_capacity_rows(
    capacity_rows: list[dict[str, Any]],
    *,
    min_credible_slices: int,
    min_robust_slices: int,
) -> dict[str, Any]:
    credible_days = [int(row["train_days"]) for row in capacity_rows if int(row.get("slice_count", 0)) >= int(min_credible_slices)]
    robust_days = [int(row["train_days"]) for row in capacity_rows if int(row.get("slice_count", 0)) >= int(min_robust_slices)]
    max_credible = max(credible_days) if credible_days else 0
    max_robust = max(robust_days) if robust_days else 0
    first_insufficient = next(
        (int(row["train_days"]) for row in capacity_rows if int(row["train_days"]) > max_credible and int(row.get("slice_count", 0)) < int(min_credible_slices)),
        0,
    )
    zero_slice = [int(row["train_days"]) for row in capacity_rows if int(row.get("slice_count", 0)) == 0]
    return {
        "max_credible_train_days": int(max_credible),
        "max_robust_train_days": int(max_robust),
        "first_insufficient_train_days": int(first_insufficient),
        "zero_slice_train_days": zero_slice,
    }


def classify_capacity_decision(summary: dict[str, Any]) -> str:
    max_credible = int(summary.get("max_credible_train_days") or 0)
    first_insufficient = int(summary.get("first_insufficient_train_days") or 0)
    if max_credible >= 40 and first_insufficient >= 45:
        return "non_overlapping_forward_capacity_supports_40d_watch_45d_plus_insufficient"
    if max_credible > 0:
        return "non_overlapping_forward_capacity_limited_but_usable"
    return "non_overlapping_forward_capacity_insufficient_for_compare"


def format_days(days: list[int]) -> str:
    return "/".join(f"{int(day)}d" for day in days if int(day) > 0)


def build_research_note(
    capacity_rows: list[dict[str, Any]],
    *,
    validation_days: int,
    summary: dict[str, Any],
) -> str:
    max_credible = int(summary.get("max_credible_train_days") or 0)
    one_slice_days = [int(row["train_days"]) for row in capacity_rows if int(row.get("slice_count", 0)) == 1]
    zero_slice_days = [int(day) for day in summary.get("zero_slice_train_days") or []]
    one_slice_text = format_days(one_slice_days)
    zero_slice_text = f"{int(zero_slice_days[0])}d+" if zero_slice_days else ""
    return (
        f"当前 non-overlapping {int(validation_days)}d validation 仍可把 train window 扩到 {max_credible}d，但 "
        f"{one_slice_text} 只剩 1 个 slice，{zero_slice_text} 已无可用 slice；"
        f"因此 35d/40d 是本轮仍可读的最长窗口，45d+ 不应继续当成有效 forward compare 主证据。"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Hold Forward Window Capacity SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- validation_days: `{int(payload.get('validation_days') or 0)}`",
        f"- step_days: `{int(payload.get('step_days') or 0)}`",
        f"- validation_window_mode: `{text(payload.get('validation_window_mode'))}`",
        f"- max_credible_train_days: `{int(payload.get('max_credible_train_days') or 0)}`",
        f"- max_robust_train_days: `{int(payload.get('max_robust_train_days') or 0)}`",
        f"- first_insufficient_train_days: `{int(payload.get('first_insufficient_train_days') or 0)}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        "",
        "## Capacity Rows",
        "",
    ]
    for row in payload.get("capacity_rows", []):
        lines.append(f"- train_days=`{int(row['train_days'])}` | slice_count=`{int(row['slice_count'])}`")
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
    stamp_dt = BASE_MODULE.parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser().resolve() if text(args.dataset_path) else BASE_MODULE.select_latest_intraday_dataset(review_dir)
    symbol = text(args.symbol).upper()
    candidate_train_days = parse_candidate_train_days(args.candidate_train_days)

    frame = BASE_MODULE.load_frame(dataset_path)
    frame = frame[frame["symbol"] == symbol].copy().sort_values("ts").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"symbol_not_found_in_dataset:{symbol}")

    capacity_rows = build_capacity_rows(
        frame,
        candidate_train_days=candidate_train_days,
        validation_days=int(args.validation_days),
        step_days=int(args.step_days),
    )
    summary = summarize_capacity_rows(capacity_rows, min_credible_slices=2, min_robust_slices=3)
    research_decision = classify_capacity_decision(summary)
    coverage_start = BASE_MODULE.fmt_utc(frame["ts"].min().to_pydatetime().replace(tzinfo=dt.timezone.utc))
    coverage_end = BASE_MODULE.fmt_utc(frame["ts"].max().to_pydatetime().replace(tzinfo=dt.timezone.utc))
    window_mode = COMPARE_MODULE.validation_window_mode(step_days=int(args.step_days), validation_days=int(args.validation_days))
    payload = {
        "action": "build_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "symbol": symbol,
        "coverage_start_utc": coverage_start,
        "coverage_end_utc": coverage_end,
        "cadence_minutes": COMPARE_MODULE.cadence_minutes(frame),
        "validation_days": int(args.validation_days),
        "step_days": int(args.step_days),
        "validation_window_mode": window_mode,
        "candidate_train_days": candidate_train_days,
        "capacity_rows": capacity_rows,
        **summary,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:exit_hold_forward_capacity:{window_mode}:"
            f"max_credible={int(summary['max_credible_train_days'])}d,"
            f"max_robust={int(summary['max_robust_train_days'])}d,"
            f"first_insufficient={int(summary['first_insufficient_train_days'])}d,"
            f"decision={research_decision}"
        ),
        "research_note": build_research_note(
            capacity_rows,
            validation_days=int(args.validation_days),
            summary=summary,
        ),
        "limitation_note": (
            "这个工件只描述当前数据覆盖下的 forward slice 容量，不评价 hold8/hold16 谁更优；"
            "若数据集扩展，必须重新构建后再决定是否继续更长窗比较。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_forward_window_capacity_sim_only.md"
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
