#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
BASE_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_sim_only.py")
EXIT_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_exit_risk_sim_only.py")
COMPARE_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py")


BASE_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_base", BASE_SCRIPT_PATH)
BASE_MODULE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE_MODULE)

EXIT_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_exit_risk", EXIT_SCRIPT_PATH)
EXIT_MODULE = importlib.util.module_from_spec(EXIT_SPEC)
assert EXIT_SPEC is not None and EXIT_SPEC.loader is not None
EXIT_SPEC.loader.exec_module(EXIT_MODULE)

COMPARE_SPEC = importlib.util.spec_from_file_location("fenlie_hold_forward_compare", COMPARE_SCRIPT_PATH)
COMPARE_MODULE = importlib.util.module_from_spec(COMPARE_SPEC)
assert COMPARE_SPEC is not None and COMPARE_SPEC.loader is not None
COMPARE_SPEC.loader.exec_module(COMPARE_MODULE)


ROBUSTNESS_GRIDS: list[dict[str, Any]] = [
    {"grid_id": "train20_valid10_step10", "train_days": 20, "validation_days": 10, "step_days": 10},
    {"grid_id": "train30_valid10_step10", "train_days": 30, "validation_days": 10, "step_days": 10},
    {"grid_id": "train30_valid5_step5", "train_days": 30, "validation_days": 5, "step_days": 5},
    {"grid_id": "train40_valid10_step10", "train_days": 40, "validation_days": 10, "step_days": 10},
]

CANONICAL_GRID = {"grid_id": "train30_valid10_step10", "train_days": 30, "validation_days": 10, "step_days": 10}
FEATURE_CANDIDATES = ["bars_since_breakout", "pullback_depth_atr", "ema20_vs_ema50_atr"]
DELTA_R_THRESHOLD = 0.15
SELECTION_SCENARIO = next(row for row in BASE_MODULE.EXECUTION_COST_SCENARIOS if row["scenario_id"] == BASE_MODULE.SELECTION_SCENARIO_ID)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY price-state hold-router hypothesis report from hold8 vs hold24 divergence cases."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def load_signal_frame(dataset_path: Path, symbol: str, breakout_lookback: int) -> pd.DataFrame:
    frame = BASE_MODULE.add_features(BASE_MODULE.load_frame(dataset_path))
    frame = frame[frame["symbol"] == symbol].copy().sort_values("ts").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"symbol_not_found_in_dataset:{symbol}")
    work = BASE_MODULE.add_breakout_context(frame.copy(), breakout_lookback)
    atr = work["atr_14"].replace(0.0, pd.NA)
    work["ema20_vs_ema50_atr"] = (work["ema_20"] - work["ema_50"]) / atr
    return work


def trim_trade(trade: dict[str, Any] | None) -> dict[str, Any]:
    if not trade:
        return {}
    return {
        "entry_ts_utc": text(trade.get("entry_ts_utc")),
        "exit_ts_utc": text(trade.get("exit_ts_utc")),
        "exit_reason": text(trade.get("exit_reason")),
        "net_r_multiple": float(trade.get("net_r_multiple", 0.0) or 0.0),
        "bars_held": int(trade.get("bars_held", 0) or 0),
    }


def build_divergence_cases(
    *,
    signal_frame: pd.DataFrame,
    base_entry_params: dict[str, Any],
) -> list[dict[str, Any]]:
    slices = COMPARE_MODULE.build_forward_slices(
        signal_frame,
        train_days=int(CANONICAL_GRID["train_days"]),
        validation_days=int(CANONICAL_GRID["validation_days"]),
        step_days=int(CANONICAL_GRID["step_days"]),
    )
    rows: list[dict[str, Any]] = []
    for slice_info in slices:
        hold8 = COMPARE_MODULE.evaluate_fixed_exit(
            train_frame=slice_info["train_frame"],
            validation_frame=slice_info["validation_frame"],
            base_entry_params=base_entry_params,
            exit_params={"max_hold_bars": 8, "break_even_trigger_r": 0.0, "trailing_stop_atr": 0.0, "cooldown_after_losses": 0, "cooldown_bars": 0},
        )
        hold24 = COMPARE_MODULE.evaluate_fixed_exit(
            train_frame=slice_info["train_frame"],
            validation_frame=slice_info["validation_frame"],
            base_entry_params=base_entry_params,
            exit_params={"max_hold_bars": 24, "break_even_trigger_r": 0.0, "trailing_stop_atr": 0.0, "cooldown_after_losses": 0, "cooldown_bars": 0},
        )
        by_entry_8 = {text(trade.get("entry_ts_utc")): trade for trade in hold8["validation_selected"]["trades"]}
        by_entry_24 = {text(trade.get("entry_ts_utc")): trade for trade in hold24["validation_selected"]["trades"]}
        entry_keys = sorted(set(by_entry_8) | set(by_entry_24))
        for entry_ts in entry_keys:
            t8 = by_entry_8.get(entry_ts)
            t24 = by_entry_24.get(entry_ts)
            if not (t8 and t24):
                continue
            delta_r = float(t24["net_r_multiple"]) - float(t8["net_r_multiple"])
            if delta_r > DELTA_R_THRESHOLD:
                case_class = "hold24_better"
            elif delta_r < -DELTA_R_THRESHOLD:
                case_class = "hold8_better"
            else:
                continue
            signal_row = signal_frame[signal_frame["ts"] == pd.Timestamp(entry_ts).tz_localize(None)].tail(1)
            if signal_row.empty:
                continue
            signal = signal_row.iloc[0]
            rows.append(
                {
                    "slice_id": text(slice_info["slice_id"]),
                    "entry_ts_utc": entry_ts,
                    "class": case_class,
                    "delta_r_24_minus_8": round(delta_r, 6),
                    "bars_since_breakout": float(signal["bars_since_breakout"]) if pd.notna(signal["bars_since_breakout"]) else None,
                    "pullback_depth_atr": float(signal["pullback_depth_atr"]) if pd.notna(signal["pullback_depth_atr"]) else None,
                    "ema20_vs_ema50_atr": float(signal["ema20_vs_ema50_atr"]) if pd.notna(signal["ema20_vs_ema50_atr"]) else None,
                    "hold8_trade": trim_trade(t8),
                    "hold24_trade": trim_trade(t24),
                }
            )
    return rows


def derive_router_hypotheses(divergence_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    for feature in FEATURE_CANDIDATES:
        hold24_values = [float(row[feature]) for row in divergence_cases if row["class"] == "hold24_better" and row.get(feature) is not None]
        hold8_values = [float(row[feature]) for row in divergence_cases if row["class"] == "hold8_better" and row.get(feature) is not None]
        if not hold24_values or not hold8_values:
            continue
        if max(hold24_values) < min(hold8_values):
            threshold = (max(hold24_values) + min(hold8_values)) / 2.0
            hypotheses.append(
                {
                    "router_id": f"{feature}_le_threshold_choose_hold24",
                    "feature": feature,
                    "direction": "le_choose_hold24_else_hold8",
                    "threshold": float(round(threshold, 6)),
                    "derivation": {
                        "hold24_better_values": hold24_values,
                        "hold8_better_values": hold8_values,
                    },
                }
            )
        elif min(hold24_values) > max(hold8_values):
            threshold = (min(hold24_values) + max(hold8_values)) / 2.0
            hypotheses.append(
                {
                    "router_id": f"{feature}_gt_threshold_choose_hold24",
                    "feature": feature,
                    "direction": "gt_choose_hold24_else_hold8",
                    "threshold": float(round(threshold, 6)),
                    "derivation": {
                        "hold24_better_values": hold24_values,
                        "hold8_better_values": hold8_values,
                    },
                }
            )
    return hypotheses


def choose_hold(signal_row: pd.Series, hypothesis: dict[str, Any]) -> int:
    feature = text(hypothesis.get("feature"))
    value = signal_row.get(feature)
    if pd.isna(value):
        return 8
    threshold = float(hypothesis["threshold"])
    direction = text(hypothesis.get("direction"))
    if direction == "le_choose_hold24_else_hold8":
        return 24 if float(value) <= threshold else 8
    if direction == "gt_choose_hold24_else_hold8":
        return 24 if float(value) > threshold else 8
    return 8


def apply_cost_full(result: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    fee_bps = float(scenario["fee_bps_per_side"])
    slip_bps = float(scenario["slippage_bps_per_side"])
    entry_mult = (1.0 + slip_bps / 10000.0) * (1.0 + fee_bps / 10000.0)
    exit_mult = (1.0 - slip_bps / 10000.0) * (1.0 - fee_bps / 10000.0)
    stressed_trades: list[dict[str, Any]] = []
    for trade in result["trades"]:
        entry_fill = float(trade["entry_price"]) * entry_mult
        exit_fill = float(trade["exit_price"]) * exit_mult
        risk_per_unit = float(trade["risk_per_unit"])
        net_pnl_pct = (exit_fill / entry_fill) - 1.0
        net_r_multiple = (exit_fill - entry_fill) / risk_per_unit if risk_per_unit > 0.0 else 0.0
        row = dict(trade)
        row["net_pnl_pct"] = round(float(net_pnl_pct), 6)
        row["net_r_multiple"] = round(float(net_r_multiple), 6)
        stressed_trades.append(row)
    return {
        "trades": stressed_trades,
        "metrics": COMPARE_MODULE.aggregate_trade_metrics(stressed_trades, pnl_field="net_pnl_pct", r_field="net_r_multiple"),
    }


def simulate_dynamic_router(frame: pd.DataFrame, base_entry_params: dict[str, Any], hypothesis: dict[str, Any]) -> dict[str, Any]:
    work = BASE_MODULE.add_breakout_context(frame.reset_index(drop=True).copy(), int(base_entry_params["breakout_lookback"]))
    atr = work["atr_14"].replace(0.0, pd.NA)
    work["ema20_vs_ema50_atr"] = (work["ema_20"] - work["ema_50"]) / atr
    entries = BASE_MODULE.signal_mask(work, base_entry_params)
    stop_buffer_atr = float(base_entry_params["stop_buffer_atr"])
    target_r = float(base_entry_params["target_r"])
    trades: list[dict[str, Any]] = []
    idx = 0
    while idx < len(work) - 1:
        if not bool(entries.iloc[idx]):
            idx += 1
            continue
        signal_row = work.iloc[idx]
        entry_idx = idx + 1
        entry_row = work.iloc[entry_idx]
        swing_low = float(signal_row["swing_low_8_prev"]) if pd.notna(signal_row["swing_low_8_prev"]) else float(signal_row["low"])
        stop_price = min(swing_low, float(signal_row["ema_50"])) - float(signal_row["atr_14"]) * stop_buffer_atr
        entry_price = float(entry_row["open"])
        risk_per_unit = entry_price - stop_price
        if not math.isfinite(risk_per_unit) or risk_per_unit <= 0.0:
            idx += 1
            continue
        max_hold = choose_hold(signal_row, hypothesis)
        target_price = entry_price + risk_per_unit * target_r
        exit_idx = entry_idx
        exit_price = float(entry_row["close"])
        exit_reason = "time_exit"
        bars_held = 0
        last_idx = min(len(work) - 1, entry_idx + max_hold - 1)
        for probe_idx in range(entry_idx, last_idx + 1):
            probe = work.iloc[probe_idx]
            bars_held = probe_idx - entry_idx + 1
            low = float(probe["low"])
            high = float(probe["high"])
            if low <= stop_price and high >= target_price:
                exit_idx = probe_idx
                exit_price = stop_price
                exit_reason = "ambiguous_bar_stop_first"
                break
            if low <= stop_price:
                exit_idx = probe_idx
                exit_price = stop_price
                exit_reason = "stop_hit"
                break
            if high >= target_price:
                exit_idx = probe_idx
                exit_price = target_price
                exit_reason = "target_hit"
                break
            if probe_idx == last_idx:
                exit_idx = probe_idx
                exit_price = float(probe["close"])
                exit_reason = "time_exit" if probe_idx < len(work) - 1 else "end_of_data"
        pnl_pct = (exit_price / entry_price) - 1.0
        r_multiple = (exit_price - entry_price) / risk_per_unit
        trades.append(
            {
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "risk_per_unit": round(risk_per_unit, 6),
                "pnl_pct": round(float(pnl_pct), 6),
                "r_multiple": round(float(r_multiple), 6),
                "bars_held": int(bars_held),
                "exit_reason": exit_reason,
                "chosen_hold": int(max_hold),
            }
        )
        idx = exit_idx + 1
    return {"trades": trades, "metrics": EXIT_MODULE.summarize_trade_metrics(trades)}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Hold Router Hypothesis SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Router Hypotheses",
        "",
    ]
    for row in payload.get("router_hypotheses", []):
        lines.append(
            f"- `{row['router_id']}` | feature=`{row['feature']}` | direction=`{row['direction']}` | threshold=`{row['threshold']}`"
        )
    lines.extend(["", "## Router Results", ""])
    for row in payload.get("router_results", []):
        lines.append(
            f"- `{row['router_id']}` | avg_ret=`{float(row['average_metrics']['cumulative_return']):.2%}` | "
            f"avg_obj=`{float(row['average_objective']):.3f}` | chosen=`{json.dumps(row['chosen_hold_counts_total'], ensure_ascii=False)}`"
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
    stamp_dt = BASE_MODULE.parse_stamp(args.stamp)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    base_artifact_path = Path(args.base_artifact_path).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    symbol = text(args.symbol).upper()

    base_entry_params, base_payload = EXIT_MODULE.load_base_entry_params(base_artifact_path, symbol)
    signal_frame = load_signal_frame(dataset_path, symbol, int(base_entry_params["breakout_lookback"]))
    divergence_cases = build_divergence_cases(signal_frame=signal_frame, base_entry_params=base_entry_params)
    router_hypotheses = derive_router_hypotheses(divergence_cases)

    pure_hold_grid_rows: list[dict[str, Any]] = []
    for grid in ROBUSTNESS_GRIDS:
        slices = COMPARE_MODULE.build_forward_slices(
            signal_frame,
            train_days=int(grid["train_days"]),
            validation_days=int(grid["validation_days"]),
            step_days=int(grid["step_days"]),
        )
        aggregate_trades_by_hold = {"hold8_zero": [], "hold16_zero": [], "hold24_zero": []}
        for slice_info in slices:
            for hold, config_id in ((8, "hold8_zero"), (16, "hold16_zero"), (24, "hold24_zero")):
                evaluated = COMPARE_MODULE.evaluate_fixed_exit(
                    train_frame=slice_info["train_frame"],
                    validation_frame=slice_info["validation_frame"],
                    base_entry_params=base_entry_params,
                    exit_params={"max_hold_bars": hold, "break_even_trigger_r": 0.0, "trailing_stop_atr": 0.0, "cooldown_after_losses": 0, "cooldown_bars": 0},
                )
                aggregate_trades_by_hold[config_id].extend(list(evaluated["validation_selected"]["trades"]))
        pure_hold_grid_rows.append(
            {
                "grid_id": text(grid["grid_id"]),
                "aggregate_metrics_by_hold": {
                    config_id: COMPARE_MODULE.aggregate_trade_metrics(trades, pnl_field="net_pnl_pct", r_field="net_r_multiple")
                    for config_id, trades in aggregate_trades_by_hold.items()
                },
            }
        )

    router_results: list[dict[str, Any]] = []
    for hypothesis in router_hypotheses:
        grid_rows: list[dict[str, Any]] = []
        all_router_trades: list[dict[str, Any]] = []
        chosen_hold_counts_total = Counter()
        better_than_hold8_grids = 0
        better_than_hold24_grids = 0
        for grid in ROBUSTNESS_GRIDS:
            slices = COMPARE_MODULE.build_forward_slices(
                signal_frame,
                train_days=int(grid["train_days"]),
                validation_days=int(grid["validation_days"]),
                step_days=int(grid["step_days"]),
            )
            aggregate_router_trades: list[dict[str, Any]] = []
            for slice_info in slices:
                gross = simulate_dynamic_router(slice_info["validation_frame"], base_entry_params, hypothesis)
                chosen_hold_counts_total.update(int(trade["chosen_hold"]) for trade in gross["trades"])
                stressed = apply_cost_full(gross, SELECTION_SCENARIO)
                aggregate_router_trades.extend(list(stressed["trades"]))
            router_metrics = COMPARE_MODULE.aggregate_trade_metrics(aggregate_router_trades, pnl_field="net_pnl_pct", r_field="net_r_multiple")
            all_router_trades.extend(aggregate_router_trades)
            pure_hold_grid = next(row for row in pure_hold_grid_rows if row["grid_id"] == text(grid["grid_id"]))
            hold8_metrics = dict((pure_hold_grid.get("aggregate_metrics_by_hold") or {}).get("hold8_zero") or {})
            hold24_metrics = dict((pure_hold_grid.get("aggregate_metrics_by_hold") or {}).get("hold24_zero") or {})
            router_objective = float(BASE_MODULE.objective(router_metrics))
            hold8_objective = float(BASE_MODULE.objective(hold8_metrics))
            hold24_objective = float(BASE_MODULE.objective(hold24_metrics))
            if router_objective > hold8_objective:
                better_than_hold8_grids += 1
            if router_objective > hold24_objective:
                better_than_hold24_grids += 1
            grid_rows.append(
                {
                    "grid_id": text(grid["grid_id"]),
                    "router_metrics": router_metrics,
                    "router_objective": router_objective,
                    "hold8_objective": hold8_objective,
                    "hold24_objective": hold24_objective,
                }
            )
        aggregate_metrics = COMPARE_MODULE.aggregate_trade_metrics(all_router_trades, pnl_field="net_pnl_pct", r_field="net_r_multiple")
        router_results.append(
            {
                "router_id": text(hypothesis["router_id"]),
                "feature": text(hypothesis["feature"]),
                "direction": text(hypothesis["direction"]),
                "threshold": float(hypothesis["threshold"]),
                "grid_rows": grid_rows,
                "average_metrics": aggregate_metrics,
                "average_objective": float(BASE_MODULE.objective(aggregate_metrics)),
                "chosen_hold_counts_total": dict(chosen_hold_counts_total),
                "better_than_hold8_grids": int(better_than_hold8_grids),
                "better_than_hold24_grids": int(better_than_hold24_grids),
            }
        )

    best_router = max(router_results, key=lambda row: float(row.get("average_objective", 0.0) or 0.0)) if router_results else {}
    research_decision = "no_clean_price_state_router_hypothesis"
    if best_router and int(best_router.get("better_than_hold8_grids", 0)) == len(ROBUSTNESS_GRIDS) and int(best_router.get("better_than_hold24_grids", 0)) == len(ROBUSTNESS_GRIDS):
        research_decision = "pullback_depth_router_hypothesis_emerges_but_same_sample_only"

    payload = {
        "action": "build_price_action_breakout_pullback_hold_router_hypothesis_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "symbol": symbol,
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "canonical_grid": CANONICAL_GRID,
        "selection_scenario_id": BASE_MODULE.SELECTION_SCENARIO_ID,
        "router_hypotheses": router_hypotheses,
        "divergence_case_count": int(len(divergence_cases)),
        "divergence_cases": divergence_cases,
        "pure_hold_grid_rows": pure_hold_grid_rows,
        "router_results": router_results,
        "best_router": best_router,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:hold_router_hypothesis:{BASE_MODULE.SELECTION_SCENARIO_ID}:"
            f"router_count={len(router_hypotheses)},"
            f"best_router={text(best_router.get('router_id')) or '-'},"
            f"better_than_hold8_grids={int(best_router.get('better_than_hold8_grids', 0) or 0)},"
            f"better_than_hold24_grids={int(best_router.get('better_than_hold24_grids', 0) or 0)},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份工件只验证 price-state-only 的动态 hold router 是否存在简单可解释假设；"
            "若有，也只能记作 same-sample hypothesis，不能直接 promotion。"
        ),
        "limitation_note": (
            "threshold 完全从 canonical divergence sample 推导，且 robustness grids 共享同一底层样本；"
            "因此即使结果很强，也只能算 hypothesis，不可直接替换 baseline。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_router_hypothesis_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_router_hypothesis_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_router_hypothesis_sim_only.json"
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
