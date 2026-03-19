#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
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


ROBUSTNESS_GRIDS: list[dict[str, Any]] = [
    {"grid_id": "train20_valid10_step10", "train_days": 20, "validation_days": 10, "step_days": 10},
    {"grid_id": "train30_valid10_step10", "train_days": 30, "validation_days": 10, "step_days": 10},
    {"grid_id": "train30_valid5_step5", "train_days": 30, "validation_days": 5, "step_days": 5},
    {"grid_id": "train40_valid10_step10", "train_days": 40, "validation_days": 10, "step_days": 10},
]

HOLD_CONFIGS: list[dict[str, Any]] = [
    {
        "config_id": "hold8_zero",
        "label": "hold=8",
        "exit_params": {
            "max_hold_bars": 8,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
    {
        "config_id": "hold12_zero",
        "label": "hold=12",
        "exit_params": {
            "max_hold_bars": 12,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
    {
        "config_id": "hold16_zero",
        "label": "hold=16",
        "exit_params": {
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
    {
        "config_id": "hold24_zero",
        "label": "hold=24",
        "exit_params": {
            "max_hold_bars": 24,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY triage report for the pure hold-family (8/12/16/24) on ETH price-state backbone."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def unique_winner(values: list[tuple[str, float]]) -> str:
    best_value = max(value for _, value in values)
    winners = [name for name, value in values if abs(float(value) - float(best_value)) <= 1e-12]
    return winners[0] if len(winners) == 1 else "tie"


def evaluate_grid(
    *,
    frame: pd.DataFrame,
    base_entry_params: dict[str, Any],
    grid: dict[str, Any],
) -> dict[str, Any]:
    slices = COMPARE_MODULE.build_forward_slices(
        frame,
        train_days=int(grid["train_days"]),
        validation_days=int(grid["validation_days"]),
        step_days=int(grid["step_days"]),
    )
    selected_trades_by_config: dict[str, list[dict[str, Any]]] = {text(row["config_id"]): [] for row in HOLD_CONFIGS}
    slice_rows: list[dict[str, Any]] = []
    slice_unique_return_wins = {text(row["config_id"]): 0 for row in HOLD_CONFIGS}
    slice_unique_objective_wins = {text(row["config_id"]): 0 for row in HOLD_CONFIGS}
    slice_return_ties = 0
    slice_objective_ties = 0

    for slice_info in slices:
        row: dict[str, Any] = {
            "slice_id": text(slice_info["slice_id"]),
            "validation_start_utc": text(slice_info["validation_start_utc"]),
            "validation_end_utc": text(slice_info["validation_end_utc"]),
            "configs": {},
        }
        ret_values: list[tuple[str, float]] = []
        obj_values: list[tuple[str, float]] = []
        for config in HOLD_CONFIGS:
            config_id = text(config["config_id"])
            evaluated = COMPARE_MODULE.evaluate_fixed_exit(
                train_frame=slice_info["train_frame"],
                validation_frame=slice_info["validation_frame"],
                base_entry_params=base_entry_params,
                exit_params=dict(config["exit_params"]),
            )
            selected_trades_by_config[config_id].extend(list(evaluated["validation_selected"]["trades"]))
            metrics = dict(evaluated["validation_selected"]["metrics"])
            row["configs"][config_id] = {
                "validation_metrics": metrics,
                "validation_status": text(evaluated["validation_status"]),
                "validation_objective": float(evaluated["validation_objective"]),
            }
            ret_values.append((config_id, float(metrics.get("cumulative_return", 0.0) or 0.0)))
            obj_values.append((config_id, float(evaluated["validation_objective"])))

        return_winner = unique_winner(ret_values)
        objective_winner = unique_winner(obj_values)
        row["winner_by_validation_return"] = return_winner
        row["winner_by_validation_objective"] = objective_winner
        if return_winner == "tie":
            slice_return_ties += 1
        else:
            slice_unique_return_wins[return_winner] += 1
        if objective_winner == "tie":
            slice_objective_ties += 1
        else:
            slice_unique_objective_wins[objective_winner] += 1
        slice_rows.append(row)

    aggregate_selected_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_objective_by_config: dict[str, float] = {}
    for config in HOLD_CONFIGS:
        config_id = text(config["config_id"])
        aggregate_selected_metrics_by_config[config_id] = COMPARE_MODULE.aggregate_trade_metrics(
            selected_trades_by_config[config_id],
            pnl_field="net_pnl_pct",
            r_field="net_r_multiple",
        )
        aggregate_objective_by_config[config_id] = float(
            COMPARE_MODULE.BASE_MODULE.objective(aggregate_selected_metrics_by_config[config_id])
        )

    aggregate_return_winner = unique_winner(
        [(config_id, float((aggregate_selected_metrics_by_config.get(config_id) or {}).get("cumulative_return", 0.0) or 0.0))
         for config_id in aggregate_selected_metrics_by_config]
    )
    aggregate_objective_winner = unique_winner(
        [(config_id, float(aggregate_objective_by_config.get(config_id, 0.0) or 0.0)) for config_id in aggregate_objective_by_config]
    )

    return {
        "grid_id": text(grid["grid_id"]),
        "train_days": int(grid["train_days"]),
        "validation_days": int(grid["validation_days"]),
        "step_days": int(grid["step_days"]),
        "slice_count": int(len(slice_rows)),
        "slice_rows": slice_rows,
        "aggregate_selected_metrics_by_config": aggregate_selected_metrics_by_config,
        "aggregate_objective_by_config": aggregate_objective_by_config,
        "comparison_summary": {
            "winner_by_aggregate_return": aggregate_return_winner,
            "winner_by_aggregate_objective": aggregate_objective_winner,
            "slice_unique_return_wins": slice_unique_return_wins,
            "slice_unique_objective_wins": slice_unique_objective_wins,
            "slice_return_ties": int(slice_return_ties),
            "slice_objective_ties": int(slice_objective_ties),
        },
    }


def build_overall_summary(grid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_return_scheme_wins = {text(row["config_id"]): 0 for row in HOLD_CONFIGS}
    aggregate_objective_scheme_wins = {text(row["config_id"]): 0 for row in HOLD_CONFIGS}
    unique_slice_return_wins = {text(row["config_id"]): 0 for row in HOLD_CONFIGS}
    unique_slice_objective_wins = {text(row["config_id"]): 0 for row in HOLD_CONFIGS}
    aggregate_return_ties = 0
    aggregate_objective_ties = 0
    total_slice_return_ties = 0
    total_slice_objective_ties = 0

    for row in grid_rows:
        summary = dict(row.get("comparison_summary") or {})
        ret_winner = text(summary.get("winner_by_aggregate_return"))
        obj_winner = text(summary.get("winner_by_aggregate_objective"))
        if ret_winner == "tie":
            aggregate_return_ties += 1
        else:
            aggregate_return_scheme_wins[ret_winner] += 1
        if obj_winner == "tie":
            aggregate_objective_ties += 1
        else:
            aggregate_objective_scheme_wins[obj_winner] += 1
        for key, value in (summary.get("slice_unique_return_wins") or {}).items():
            unique_slice_return_wins[text(key)] += int(value or 0)
        for key, value in (summary.get("slice_unique_objective_wins") or {}).items():
            unique_slice_objective_wins[text(key)] += int(value or 0)
        total_slice_return_ties += int(summary.get("slice_return_ties") or 0)
        total_slice_objective_ties += int(summary.get("slice_objective_ties") or 0)

    return {
        "grid_count": int(len(grid_rows)),
        "aggregate_return_scheme_wins": aggregate_return_scheme_wins,
        "aggregate_objective_scheme_wins": aggregate_objective_scheme_wins,
        "aggregate_return_ties": int(aggregate_return_ties),
        "aggregate_objective_ties": int(aggregate_objective_ties),
        "unique_slice_return_wins_total": unique_slice_return_wins,
        "unique_slice_objective_wins_total": unique_slice_objective_wins,
        "total_slice_return_ties": int(total_slice_return_ties),
        "total_slice_objective_ties": int(total_slice_objective_ties),
    }


def classify_research_decision(summary: dict[str, Any]) -> str:
    agg_ret = summary.get("aggregate_return_scheme_wins") or {}
    agg_obj = summary.get("aggregate_objective_scheme_wins") or {}
    slice_ret = summary.get("unique_slice_return_wins_total") or {}
    slice_obj = summary.get("unique_slice_objective_wins_total") or {}
    if (
        int(agg_ret.get("hold24_zero", 0)) >= 4
        and int(agg_obj.get("hold8_zero", 0)) >= 4
        and int(slice_ret.get("hold12_zero", 0)) == 0
        and int(slice_obj.get("hold12_zero", 0)) == 0
        and int(slice_ret.get("hold16_zero", 0)) >= 3
        and int(slice_obj.get("hold16_zero", 0)) >= 3
    ):
        return "drop_hold12_keep_hold16_baseline_add_hold8_and_hold24_candidates"
    if (
        int(agg_ret.get("hold24_zero", 0)) >= 4
        and int(agg_obj.get("hold8_zero", 0)) >= 4
        and int(slice_ret.get("hold12_zero", 0)) == 0
        and int(slice_obj.get("hold12_zero", 0)) == 0
    ):
        return "drop_hold12_add_hold8_and_hold24_candidates"
    return "hold_family_triage_inconclusive"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = dict(payload.get("overall_summary") or {})
    lines = [
        "# Price Action Breakout Pullback Hold Family Triage SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Overall Summary",
        "",
        f"- aggregate_return_scheme_wins: `{json.dumps(summary.get('aggregate_return_scheme_wins') or {}, ensure_ascii=False)}`",
        f"- aggregate_objective_scheme_wins: `{json.dumps(summary.get('aggregate_objective_scheme_wins') or {}, ensure_ascii=False)}`",
        f"- unique_slice_return_wins_total: `{json.dumps(summary.get('unique_slice_return_wins_total') or {}, ensure_ascii=False)}`",
        f"- unique_slice_objective_wins_total: `{json.dumps(summary.get('unique_slice_objective_wins_total') or {}, ensure_ascii=False)}`",
        f"- total_slice_return_ties: `{summary.get('total_slice_return_ties')}`",
        f"- total_slice_objective_ties: `{summary.get('total_slice_objective_ties')}`",
        "",
        "## Grids",
        "",
    ]
    for row in payload.get("grid_rows", []):
        lines.append(f"### {text(row.get('grid_id'))}")
        comp = dict(row.get("comparison_summary") or {})
        lines.append(f"- winner_by_aggregate_return: `{text(comp.get('winner_by_aggregate_return'))}`")
        lines.append(f"- winner_by_aggregate_objective: `{text(comp.get('winner_by_aggregate_objective'))}`")
        lines.append(f"- slice_unique_return_wins: `{json.dumps(comp.get('slice_unique_return_wins') or {}, ensure_ascii=False)}`")
        lines.append(f"- slice_unique_objective_wins: `{json.dumps(comp.get('slice_unique_objective_wins') or {}, ensure_ascii=False)}`")
        for config in HOLD_CONFIGS:
            config_id = text(config["config_id"])
            metrics = dict((row.get("aggregate_selected_metrics_by_config") or {}).get(config_id) or {})
            lines.append(
                f"- `{config_id}` | ret=`{float(metrics.get('cumulative_return', 0.0) or 0.0):.2%}` | "
                f"pf=`{float(metrics.get('profit_factor', 0.0) or 0.0):.2f}` | "
                f"exp_r=`{float(metrics.get('expectancy_r', 0.0) or 0.0):.3f}` | "
                f"trades=`{int(metrics.get('trade_count', 0) or 0)}` | "
                f"dd=`{float(metrics.get('max_drawdown', 0.0) or 0.0):.2%}`"
            )
        lines.append("")
    lines.extend(
        [
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

    grid_rows = [evaluate_grid(frame=frame, base_entry_params=base_entry_params, grid=grid) for grid in ROBUSTNESS_GRIDS]
    overall_summary = build_overall_summary(grid_rows)
    research_decision = classify_research_decision(overall_summary)

    coverage_start = pd.Timestamp(frame["ts"].min()).to_pydatetime().replace(tzinfo=COMPARE_MODULE.BASE_MODULE.dt.timezone.utc)
    coverage_end = pd.Timestamp(frame["ts"].max()).to_pydatetime().replace(tzinfo=COMPARE_MODULE.BASE_MODULE.dt.timezone.utc)
    payload = {
        "action": "build_price_action_breakout_pullback_hold_family_triage_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": COMPARE_MODULE.BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "symbol": symbol,
        "coverage_start_utc": COMPARE_MODULE.BASE_MODULE.fmt_utc(coverage_start),
        "coverage_end_utc": COMPARE_MODULE.BASE_MODULE.fmt_utc(coverage_end),
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "hold_configs": HOLD_CONFIGS,
        "robustness_grids": ROBUSTNESS_GRIDS,
        "grid_rows": grid_rows,
        "overall_summary": overall_summary,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:hold_family_triage:{COMPARE_MODULE.BASE_MODULE.SELECTION_SCENARIO_ID}:"
            f"hold24_agg_ret_grids={int((overall_summary.get('aggregate_return_scheme_wins') or {}).get('hold24_zero', 0))},"
            f"hold8_agg_obj_grids={int((overall_summary.get('aggregate_objective_scheme_wins') or {}).get('hold8_zero', 0))},"
            f"hold16_unique_slice_wins={int((overall_summary.get('unique_slice_return_wins_total') or {}).get('hold16_zero', 0))},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "在冻结 simple riders 之后，继续清理纯 hold 家族；"
            "本轮只比较 8/12/16/24 四个无 rider 的持有窗。"
        ),
        "limitation_note": (
            "它说明哪些 hold 家族值得保留为候选，并不自动替换现有 baseline；"
            "baseline 仍受旧 selection policy 与 source artifact 约束。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_family_triage_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_family_triage_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_family_triage_sim_only.json"
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
