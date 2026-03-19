#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
COMPARE_SCRIPT_PATH = Path(__file__).resolve().with_name(
    "build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py"
)


COMPARE_SPEC = importlib.util.spec_from_file_location("fenlie_hold_forward_compare", COMPARE_SCRIPT_PATH)
COMPARE_MODULE = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
COMPARE_SPEC.loader.exec_module(COMPARE_MODULE)


DELTA_R_THRESHOLD = 0.15

HOLD_CONFIGS = {
    "hold8_zero": {
        "max_hold_bars": 8,
        "break_even_trigger_r": 0.0,
        "trailing_stop_atr": 0.0,
        "cooldown_after_losses": 0,
        "cooldown_bars": 0,
    },
    "hold16_zero": {
        "max_hold_bars": 16,
        "break_even_trigger_r": 0.0,
        "trailing_stop_atr": 0.0,
        "cooldown_after_losses": 0,
        "cooldown_bars": 0,
    },
    "hold24_zero": {
        "max_hold_bars": 24,
        "break_even_trigger_r": 0.0,
        "trailing_stop_atr": 0.0,
        "cooldown_after_losses": 0,
        "cooldown_bars": 0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY casebook for hold8 vs hold24 divergence on the ETH price-state backbone."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=10)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def classify_case(t8: dict[str, Any] | None, t24: dict[str, Any] | None) -> tuple[str, float | None]:
    if t8 and t24:
        delta_r = float(t24["net_r_multiple"]) - float(t8["net_r_multiple"])
        if delta_r > DELTA_R_THRESHOLD:
            return "hold24_extends_better", delta_r
        if delta_r < -DELTA_R_THRESHOLD:
            return "hold8_exits_better", delta_r
        return "near_tie", delta_r
    if t24 and not t8:
        return "only_hold24", None
    if t8 and not t24:
        return "only_hold8", None
    return "unmatched", None


def trim_trade(trade: dict[str, Any] | None) -> dict[str, Any]:
    if not trade:
        return {}
    return {
        "entry_ts_utc": text(trade.get("entry_ts_utc")),
        "exit_ts_utc": text(trade.get("exit_ts_utc")),
        "exit_reason": text(trade.get("exit_reason")),
        "net_r_multiple": float(trade.get("net_r_multiple", 0.0) or 0.0),
        "net_pnl_pct": float(trade.get("net_pnl_pct", 0.0) or 0.0),
        "bars_held": int(trade.get("bars_held", 0) or 0),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Hold Candidate Divergence Casebook SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Divergence Summary",
        "",
        f"- class_counts: `{json.dumps(payload.get('divergence_summary', {}).get('class_counts') or {}, ensure_ascii=False)}`",
        f"- hold24_better_case_count: `{payload.get('divergence_summary', {}).get('hold24_better_case_count')}`",
        f"- hold8_better_case_count: `{payload.get('divergence_summary', {}).get('hold8_better_case_count')}`",
        f"- near_tie_case_count: `{payload.get('divergence_summary', {}).get('near_tie_case_count')}`",
        "",
        "## Top Cases",
        "",
    ]
    for row in payload.get("top_divergence_cases", []):
        lines.append(
            f"- `{row['slice_id']}` | entry=`{row['entry_ts_utc']}` | class=`{row['class']}` | "
            f"delta_r_24_minus_8=`{float(row.get('delta_r_24_minus_8', 0.0)):.3f}`"
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
    stamp_dt = COMPARE_MODULE.BASE_MODULE.parse_stamp(args.stamp)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    base_artifact_path = Path(args.base_artifact_path).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    symbol = text(args.symbol).upper()

    base_entry_params, base_payload = COMPARE_MODULE.EXIT_MODULE.load_base_entry_params(base_artifact_path, symbol)
    frame = COMPARE_MODULE.BASE_MODULE.add_features(COMPARE_MODULE.BASE_MODULE.load_frame(dataset_path))
    frame = frame[frame["symbol"] == symbol].copy().sort_values("ts").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"symbol_not_found_in_dataset:{symbol}")

    slices = COMPARE_MODULE.build_forward_slices(
        frame,
        train_days=int(args.train_days),
        validation_days=int(args.validation_days),
        step_days=int(args.step_days),
    )

    aggregate_selected_metrics_by_config: dict[str, dict[str, Any]] = {}
    selected_trades_by_config: dict[str, list[dict[str, Any]]] = {config_id: [] for config_id in HOLD_CONFIGS}
    divergence_cases: list[dict[str, Any]] = []

    for slice_info in slices:
        per_config_trades: dict[str, dict[str, Any]] = {}
        for config_id, exit_params in HOLD_CONFIGS.items():
            evaluated = COMPARE_MODULE.evaluate_fixed_exit(
                train_frame=slice_info["train_frame"],
                validation_frame=slice_info["validation_frame"],
                base_entry_params=base_entry_params,
                exit_params=dict(exit_params),
            )
            trades = list(evaluated["validation_selected"]["trades"])
            selected_trades_by_config[config_id].extend(trades)
            per_config_trades[config_id] = {text(trade.get("entry_ts_utc")): trade for trade in trades}

        entry_keys = sorted(set(per_config_trades["hold8_zero"]) | set(per_config_trades["hold24_zero"]))
        for entry_ts in entry_keys:
            hold8_trade = per_config_trades["hold8_zero"].get(entry_ts)
            hold24_trade = per_config_trades["hold24_zero"].get(entry_ts)
            hold16_trade = per_config_trades["hold16_zero"].get(entry_ts)
            case_class, delta_r = classify_case(hold8_trade, hold24_trade)
            divergence_cases.append(
                {
                    "slice_id": text(slice_info["slice_id"]),
                    "entry_ts_utc": entry_ts,
                    "class": case_class,
                    "delta_r_24_minus_8": float(delta_r) if delta_r is not None else None,
                    "hold8_trade": trim_trade(hold8_trade),
                    "hold16_trade": trim_trade(hold16_trade),
                    "hold24_trade": trim_trade(hold24_trade),
                }
            )

    for config_id, trades in selected_trades_by_config.items():
        aggregate_selected_metrics_by_config[config_id] = COMPARE_MODULE.aggregate_trade_metrics(
            trades,
            pnl_field="net_pnl_pct",
            r_field="net_r_multiple",
        )

    class_counts = Counter(text(row.get("class")) for row in divergence_cases)
    top_cases = sorted(
        [row for row in divergence_cases if row.get("delta_r_24_minus_8") is not None],
        key=lambda row: abs(float(row.get("delta_r_24_minus_8") or 0.0)),
        reverse=True,
    )[:8]

    research_decision = "dual_candidate_state_reaffirmed_hold24_edge_is_concentrated_and_hold8_avoids_tail_loss"
    if class_counts["hold24_extends_better"] >= 3 and class_counts["hold8_exits_better"] == 0:
        research_decision = "hold24_candidate_strengthened_on_canonical_grid"
    elif class_counts["hold8_exits_better"] >= 3 and class_counts["hold24_extends_better"] == 0:
        research_decision = "hold8_candidate_strengthened_on_canonical_grid"

    payload = {
        "action": "build_price_action_breakout_pullback_hold_candidate_divergence_casebook_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": COMPARE_MODULE.BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "symbol": symbol,
        "canonical_grid": {
            "train_days": int(args.train_days),
            "validation_days": int(args.validation_days),
            "step_days": int(args.step_days),
            "slice_count": int(len(slices)),
            "delta_r_threshold": float(DELTA_R_THRESHOLD),
        },
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "aggregate_selected_metrics_by_config": aggregate_selected_metrics_by_config,
        "divergence_summary": {
            "class_counts": dict(class_counts),
            "hold24_better_case_count": int(class_counts["hold24_extends_better"]),
            "hold8_better_case_count": int(class_counts["hold8_exits_better"]),
            "near_tie_case_count": int(class_counts["near_tie"]),
            "only_hold24_case_count": int(class_counts["only_hold24"]),
            "only_hold8_case_count": int(class_counts["only_hold8"]),
        },
        "top_divergence_cases": top_cases,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:hold_candidate_divergence:{COMPARE_MODULE.BASE_MODULE.SELECTION_SCENARIO_ID}:"
            f"hold24_better={int(class_counts['hold24_extends_better'])},"
            f"hold8_better={int(class_counts['hold8_exits_better'])},"
            f"near_tie={int(class_counts['near_tie'])},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份 casebook 只解释 hold8 与 hold24 在 canonical forward grid 上为何并存；"
            "不改变 hold16 baseline，也不回到已冻结的 simple riders。"
        ),
        "limitation_note": (
            "它使用 canonical 30/10/10 grid 做逐笔差异解释，属于候选分歧说明，"
            "不是最终 promotion 结论。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_candidate_divergence_casebook_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_candidate_divergence_casebook_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_candidate_divergence_casebook_sim_only.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "latest_json_path": str(latest_json_path),
                "research_decision": research_decision,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
