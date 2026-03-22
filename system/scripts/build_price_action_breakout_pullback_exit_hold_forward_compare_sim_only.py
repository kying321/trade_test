#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
BASE_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_sim_only.py")
EXIT_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_exit_risk_sim_only.py")


BASE_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_base", BASE_SCRIPT_PATH)
BASE_MODULE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE_MODULE)

EXIT_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_exit_risk", EXIT_SCRIPT_PATH)
EXIT_MODULE = importlib.util.module_from_spec(EXIT_SPEC)
assert EXIT_SPEC is not None and EXIT_SPEC.loader is not None
EXIT_SPEC.loader.exec_module(EXIT_MODULE)


COMPARE_EXIT_CONFIGS: list[dict[str, Any]] = [
    {
        "config_id": "hold_8_zero_risk",
        "label": "hold=8, no_be, no_trail, no_cooldown",
        "exit_params": {
            "max_hold_bars": 8,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
    {
        "config_id": "hold_16_zero_risk",
        "label": "hold=16, no_be, no_trail, no_cooldown",
        "exit_params": {
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY forward-slice comparison artifact for ETH 15m breakout-pullback hold=8 vs hold=16."
    )
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--base-artifact-path", default="")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=10)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def choose_winner(left_name: str, left_value: float, right_name: str, right_value: float, *, eps: float = 1e-12) -> str:
    if abs(float(left_value) - float(right_value)) <= eps:
        return "tie"
    return left_name if float(left_value) > float(right_value) else right_name


def cadence_minutes(frame: pd.DataFrame) -> int:
    ts = pd.Series(sorted(frame["ts"].dropna().unique())).astype("datetime64[ns]")
    diffs = ts.diff().dropna()
    if diffs.empty:
        raise ValueError("dataset_missing_cadence")
    cadence = diffs.mode()
    if cadence.empty:
        raise ValueError("dataset_missing_cadence_mode")
    minutes = int(cadence.iloc[0] / pd.Timedelta(minutes=1))
    if minutes <= 0:
        raise ValueError("dataset_invalid_cadence")
    return minutes


def build_forward_slices(
    frame: pd.DataFrame,
    *,
    train_days: int,
    validation_days: int,
    step_days: int,
) -> list[dict[str, Any]]:
    if train_days <= 0 or validation_days <= 0 or step_days <= 0:
        raise ValueError("train_days_validation_days_step_days_must_be_positive")
    symbol_frame = frame.sort_values("ts").reset_index(drop=True).copy()
    cadence = cadence_minutes(symbol_frame)
    bars_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(minutes=cadence))
    train_bars = int(train_days) * bars_per_day
    validation_bars = int(validation_days) * bars_per_day
    step_bars = int(step_days) * bars_per_day
    unique_ts = pd.Series(sorted(symbol_frame["ts"].dropna().unique())).astype("datetime64[ns]")
    if len(unique_ts) < train_bars + validation_bars:
        raise ValueError("dataset_too_small_for_forward_compare")

    slices: list[dict[str, Any]] = []
    valid_start_idx = train_bars
    slice_index = 1
    while valid_start_idx + validation_bars <= len(unique_ts):
        train_end_ts = pd.Timestamp(unique_ts.iloc[valid_start_idx - 1])
        valid_start_ts = pd.Timestamp(unique_ts.iloc[valid_start_idx])
        valid_end_ts = pd.Timestamp(unique_ts.iloc[valid_start_idx + validation_bars - 1])

        train_frame = symbol_frame[symbol_frame["ts"] <= train_end_ts].copy().reset_index(drop=True)
        valid_frame = symbol_frame[(symbol_frame["ts"] >= valid_start_ts) & (symbol_frame["ts"] <= valid_end_ts)].copy().reset_index(drop=True)
        if train_frame.empty or valid_frame.empty:
            valid_start_idx += step_bars
            continue

        slices.append(
            {
                "slice_id": f"slice_{slice_index}",
                "slice_index": int(slice_index),
                "train_start_utc": BASE_MODULE.fmt_utc(
                    pd.Timestamp(train_frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
                ),
                "train_end_utc": BASE_MODULE.fmt_utc(train_end_ts.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "validation_start_utc": BASE_MODULE.fmt_utc(
                    valid_start_ts.to_pydatetime().replace(tzinfo=dt.timezone.utc)
                ),
                "validation_end_utc": BASE_MODULE.fmt_utc(valid_end_ts.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "train_bars": int(len(train_frame)),
                "validation_bars": int(len(valid_frame)),
                "train_frame": train_frame,
                "validation_frame": valid_frame,
            }
        )
        valid_start_idx += step_bars
        slice_index += 1
    if not slices:
        raise ValueError("no_forward_slices_built")
    return slices


def evaluate_fixed_exit(
    *,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    base_entry_params: dict[str, Any],
    exit_params: dict[str, Any],
) -> dict[str, Any]:
    params = dict(base_entry_params)
    params.update(exit_params)

    train_gross = EXIT_MODULE.simulate_symbol_with_exit_risk(train_frame, params)
    validation_gross = EXIT_MODULE.simulate_symbol_with_exit_risk(validation_frame, params)
    train_selected = apply_cost_scenario_full(
        train_gross,
        next(row for row in BASE_MODULE.EXECUTION_COST_SCENARIOS if row["scenario_id"] == BASE_MODULE.SELECTION_SCENARIO_ID),
    )
    validation_selected = apply_cost_scenario_full(
        validation_gross,
        next(row for row in BASE_MODULE.EXECUTION_COST_SCENARIOS if row["scenario_id"] == BASE_MODULE.SELECTION_SCENARIO_ID),
    )
    return {
        "train_gross": train_gross,
        "validation_gross": validation_gross,
        "train_selected": train_selected,
        "validation_selected": validation_selected,
        "train_objective": float(BASE_MODULE.objective(train_selected["metrics"])),
        "validation_objective": float(BASE_MODULE.objective(validation_selected["metrics"])),
        "validation_status": BASE_MODULE.classify_validation(validation_selected["metrics"]),
    }


def aggregate_trade_metrics(trades: list[dict[str, Any]], *, pnl_field: str, r_field: str) -> dict[str, Any]:
    return BASE_MODULE.summarize_trade_metrics(trades, pnl_field=pnl_field, r_field=r_field)


def apply_cost_scenario_full(base_result: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    fee_bps = float(scenario["fee_bps_per_side"])
    slip_bps = float(scenario["slippage_bps_per_side"])
    entry_mult = (1.0 + slip_bps / 10000.0) * (1.0 + fee_bps / 10000.0)
    exit_mult = (1.0 - slip_bps / 10000.0) * (1.0 - fee_bps / 10000.0)
    stressed_trades: list[dict[str, Any]] = []
    for trade in base_result["trades"]:
        entry_fill = float(trade["entry_price"]) * entry_mult
        exit_fill = float(trade["exit_price"]) * exit_mult
        risk_per_unit = float(trade["risk_per_unit"])
        net_pnl_pct = (exit_fill / entry_fill) - 1.0
        net_r_multiple = (exit_fill - entry_fill) / risk_per_unit if risk_per_unit > 0.0 else 0.0
        row = dict(trade)
        row["scenario_id"] = str(scenario["scenario_id"])
        row["net_entry_fill"] = round(entry_fill, 6)
        row["net_exit_fill"] = round(exit_fill, 6)
        row["net_pnl_pct"] = round(float(net_pnl_pct), 6)
        row["net_r_multiple"] = round(float(net_r_multiple), 6)
        stressed_trades.append(row)
    return {
        "scenario_id": str(scenario["scenario_id"]),
        "label": str(scenario["label"]),
        "fee_bps_per_side": fee_bps,
        "slippage_bps_per_side": slip_bps,
        "metrics": BASE_MODULE.summarize_trade_metrics(stressed_trades, pnl_field="net_pnl_pct", r_field="net_r_multiple"),
        "trade_sample": stressed_trades[:8],
        "trades": stressed_trades,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = dict(payload.get("comparison_summary") or {})
    aggregate = dict(payload.get("aggregate_validation_metrics_by_config") or {})
    lines = [
        "# Price Action Breakout Pullback Exit Hold Forward Compare SIM ONLY",
        "",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- dataset_path: `{text(payload.get('dataset_path'))}`",
        f"- base_artifact_path: `{text(payload.get('base_artifact_path'))}`",
        f"- cadence_minutes: `{payload.get('cadence_minutes')}`",
        f"- train_days: `{payload.get('train_days')}`",
        f"- validation_days: `{payload.get('validation_days')}`",
        f"- step_days: `{payload.get('step_days')}`",
        f"- slice_count: `{payload.get('slice_count')}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        "",
        "## Aggregate Validation",
        "",
    ]
    for config in payload.get("comparison_configs", []):
        config_id = text(config.get("config_id"))
        metrics = dict(aggregate.get(config_id) or {})
        lines.extend(
            [
                f"### {config_id}",
                f"- return: `{float(metrics.get('cumulative_return', 0.0)):.2%}`",
                f"- pf: `{float(metrics.get('profit_factor', 0.0)):.2f}`",
                f"- expectancy_r: `{float(metrics.get('expectancy_r', 0.0)):.3f}`",
                f"- trades: `{int(metrics.get('trade_count', 0) or 0)}`",
                f"- max_drawdown: `{float(metrics.get('max_drawdown', 0.0)):.2%}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Summary",
            "",
            f"- winner_by_aggregate_return: `{text(summary.get('winner_by_aggregate_return'))}`",
            f"- winner_by_aggregate_objective: `{text(summary.get('winner_by_aggregate_objective'))}`",
            f"- winner_by_slice_majority_return: `{text(summary.get('winner_by_slice_majority_return'))}`",
            f"- winner_by_slice_majority_objective: `{text(summary.get('winner_by_slice_majority_objective'))}`",
            f"- slice_return_wins: `{json.dumps(summary.get('slice_return_wins') or {}, ensure_ascii=False)}`",
            f"- slice_objective_wins: `{json.dumps(summary.get('slice_objective_wins') or {}, ensure_ascii=False)}`",
            "",
            "## Forward Slices",
            "",
        ]
    )
    for row in payload.get("forward_slices", []):
        lines.append(
            f"- `{row['slice_id']}` | valid=`{row['validation_start_utc']} -> {row['validation_end_utc']}` | "
            f"return_winner=`{row['winner_by_validation_return']}` | objective_winner=`{row['winner_by_validation_objective']}`"
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
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    stamp_dt = BASE_MODULE.parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser().resolve() if text(args.dataset_path) else BASE_MODULE.select_latest_intraday_dataset(review_dir)
    base_artifact_path = Path(args.base_artifact_path).expanduser().resolve() if text(args.base_artifact_path) else EXIT_MODULE.select_latest_price_action_artifact(review_dir)
    symbol = text(args.symbol).upper()

    base_entry_params, base_payload = EXIT_MODULE.load_base_entry_params(base_artifact_path, symbol)
    frame = BASE_MODULE.add_features(BASE_MODULE.load_frame(dataset_path))
    frame = frame[frame["symbol"] == symbol].copy().sort_values("ts").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"symbol_not_found_in_dataset:{symbol}")

    slices = build_forward_slices(
        frame,
        train_days=int(args.train_days),
        validation_days=int(args.validation_days),
        step_days=int(args.step_days),
    )

    per_config_selected_validation_trades: dict[str, list[dict[str, Any]]] = {
        text(item["config_id"]): [] for item in COMPARE_EXIT_CONFIGS
    }
    per_config_gross_validation_trades: dict[str, list[dict[str, Any]]] = {
        text(item["config_id"]): [] for item in COMPARE_EXIT_CONFIGS
    }
    forward_slices: list[dict[str, Any]] = []
    for slice_info in slices:
        row: dict[str, Any] = {
            "slice_id": text(slice_info["slice_id"]),
            "slice_index": int(slice_info["slice_index"]),
            "train_start_utc": text(slice_info["train_start_utc"]),
            "train_end_utc": text(slice_info["train_end_utc"]),
            "validation_start_utc": text(slice_info["validation_start_utc"]),
            "validation_end_utc": text(slice_info["validation_end_utc"]),
            "train_bars": int(slice_info["train_bars"]),
            "validation_bars": int(slice_info["validation_bars"]),
            "configs": {},
        }
        for config in COMPARE_EXIT_CONFIGS:
            config_id = text(config["config_id"])
            evaluated = evaluate_fixed_exit(
                train_frame=slice_info["train_frame"],
                validation_frame=slice_info["validation_frame"],
                base_entry_params=base_entry_params,
                exit_params=dict(config["exit_params"]),
            )
            per_config_selected_validation_trades[config_id].extend(list(evaluated["validation_selected"]["trades"]))
            per_config_gross_validation_trades[config_id].extend(list(evaluated["validation_gross"]["trades"]))
            row["configs"][config_id] = {
                "label": text(config["label"]),
                "exit_params": dict(config["exit_params"]),
                "train_metrics": dict(evaluated["train_selected"]["metrics"]),
                "validation_metrics": dict(evaluated["validation_selected"]["metrics"]),
                "train_gross_metrics": dict(evaluated["train_gross"]["metrics"]),
                "validation_gross_metrics": dict(evaluated["validation_gross"]["metrics"]),
                "validation_status": text(evaluated["validation_status"]),
                "train_objective": float(evaluated["train_objective"]),
                "validation_objective": float(evaluated["validation_objective"]),
                "validation_trade_sample": list(evaluated["validation_selected"]["trade_sample"][:6]),
            }

        hold8_metrics = dict(row["configs"]["hold_8_zero_risk"]["validation_metrics"])
        hold16_metrics = dict(row["configs"]["hold_16_zero_risk"]["validation_metrics"])
        row["winner_by_validation_return"] = choose_winner(
            "hold_8_zero_risk",
            float(hold8_metrics.get("cumulative_return", 0.0) or 0.0),
            "hold_16_zero_risk",
            float(hold16_metrics.get("cumulative_return", 0.0) or 0.0),
        )
        row["winner_by_validation_objective"] = choose_winner(
            "hold_8_zero_risk",
            float(row["configs"]["hold_8_zero_risk"]["validation_objective"]),
            "hold_16_zero_risk",
            float(row["configs"]["hold_16_zero_risk"]["validation_objective"]),
        )
        forward_slices.append(row)

    aggregate_validation_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_validation_gross_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_validation_objective_by_config: dict[str, float] = {}
    for config in COMPARE_EXIT_CONFIGS:
        config_id = text(config["config_id"])
        aggregate_validation_metrics_by_config[config_id] = aggregate_trade_metrics(
            per_config_selected_validation_trades[config_id],
            pnl_field="net_pnl_pct",
            r_field="net_r_multiple",
        )
        aggregate_validation_gross_metrics_by_config[config_id] = aggregate_trade_metrics(
            per_config_gross_validation_trades[config_id],
            pnl_field="pnl_pct",
            r_field="r_multiple",
        )
        aggregate_validation_objective_by_config[config_id] = float(
            BASE_MODULE.objective(aggregate_validation_metrics_by_config[config_id])
        )

    slice_return_wins = {
        config["config_id"]: int(sum(1 for row in forward_slices if row["winner_by_validation_return"] == config["config_id"]))
        for config in COMPARE_EXIT_CONFIGS
    }
    slice_objective_wins = {
        config["config_id"]: int(sum(1 for row in forward_slices if row["winner_by_validation_objective"] == config["config_id"]))
        for config in COMPARE_EXIT_CONFIGS
    }
    tie_return_count = int(sum(1 for row in forward_slices if row["winner_by_validation_return"] == "tie"))
    tie_objective_count = int(sum(1 for row in forward_slices if row["winner_by_validation_objective"] == "tie"))
    slice_majority_return_winner = choose_winner(
        "hold_8_zero_risk",
        float(slice_return_wins["hold_8_zero_risk"]),
        "hold_16_zero_risk",
        float(slice_return_wins["hold_16_zero_risk"]),
        eps=0.0,
    )
    slice_majority_objective_winner = choose_winner(
        "hold_8_zero_risk",
        float(slice_objective_wins["hold_8_zero_risk"]),
        "hold_16_zero_risk",
        float(slice_objective_wins["hold_16_zero_risk"]),
        eps=0.0,
    )

    aggregate_return_winner = choose_winner(
        "hold_8_zero_risk",
        float(aggregate_validation_metrics_by_config["hold_8_zero_risk"].get("cumulative_return", 0.0) or 0.0),
        "hold_16_zero_risk",
        float(aggregate_validation_metrics_by_config["hold_16_zero_risk"].get("cumulative_return", 0.0) or 0.0),
    )
    aggregate_objective_winner = choose_winner(
        "hold_8_zero_risk",
        float(aggregate_validation_objective_by_config["hold_8_zero_risk"]),
        "hold_16_zero_risk",
        float(aggregate_validation_objective_by_config["hold_16_zero_risk"]),
    )

    research_decision = "mixed_forward_profile_hold8_aggregate_hold16_consistency"
    if (
        aggregate_return_winner == "hold_8_zero_risk"
        and aggregate_objective_winner == "hold_8_zero_risk"
        and slice_majority_return_winner == "hold_8_zero_risk"
        and slice_majority_objective_winner == "hold_8_zero_risk"
    ):
        research_decision = "hold_8_forward_leader_keep_sim_only_candidate"
    elif (
        aggregate_return_winner == "hold_16_zero_risk"
        and aggregate_objective_winner == "hold_16_zero_risk"
        and slice_majority_return_winner == "hold_16_zero_risk"
        and slice_majority_objective_winner == "hold_16_zero_risk"
    ):
        research_decision = "hold_16_forward_leader_keep_baseline"

    coverage_start = pd.Timestamp(frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    coverage_end = pd.Timestamp(frame["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    payload = {
        "action": "build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "symbol": symbol,
        "coverage_start_utc": BASE_MODULE.fmt_utc(coverage_start),
        "coverage_end_utc": BASE_MODULE.fmt_utc(coverage_end),
        "cadence_minutes": cadence_minutes(frame),
        "train_days": int(args.train_days),
        "validation_days": int(args.validation_days),
        "step_days": int(args.step_days),
        "slice_count": int(len(forward_slices)),
        "selection_scenario_id": BASE_MODULE.SELECTION_SCENARIO_ID,
        "source_catalog": list(BASE_MODULE.SOURCE_CATALOG),
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "comparison_configs": COMPARE_EXIT_CONFIGS,
        "forward_slices": forward_slices,
        "aggregate_validation_metrics_by_config": aggregate_validation_metrics_by_config,
        "aggregate_validation_gross_metrics_by_config": aggregate_validation_gross_metrics_by_config,
        "aggregate_validation_objective_by_config": aggregate_validation_objective_by_config,
        "comparison_summary": {
            "winner_by_aggregate_return": aggregate_return_winner,
            "winner_by_aggregate_objective": aggregate_objective_winner,
            "winner_by_slice_majority_return": slice_majority_return_winner,
            "winner_by_slice_majority_objective": slice_majority_objective_winner,
            "slice_return_wins": slice_return_wins,
            "slice_objective_wins": slice_objective_wins,
            "slice_return_ties": int(tie_return_count),
            "slice_objective_ties": int(tie_objective_count),
        },
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:exit_hold_forward_compare:{BASE_MODULE.SELECTION_SCENARIO_ID}:"
            f"agg_return_winner={aggregate_return_winner},"
            f"agg_objective_winner={aggregate_objective_winner},"
            f"slices={len(forward_slices)},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "冻结 base entry 参数，只比较 zero-risk 的 hold=8 与 hold=16；"
            "使用 expanding-train + non-overlap forward validation slices 做 source-owned 研究对比。"
        ),
        "limitation_note": (
            "每个 slice 不重新选参，只验证固定 exit 持有窗；"
            "样本仍来自 public 15m OHLCV，结论仅可用于 SIM_ONLY 研究，不可直接放行 live。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "slice_count": len(forward_slices)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
