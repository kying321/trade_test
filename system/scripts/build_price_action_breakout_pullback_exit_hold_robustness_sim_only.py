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
    {
        "grid_id": "train20_valid10_step10",
        "train_days": 20,
        "validation_days": 10,
        "step_days": 10,
        "label": "20d train / 10d valid / 10d step",
    },
    {
        "grid_id": "train30_valid10_step10",
        "train_days": 30,
        "validation_days": 10,
        "step_days": 10,
        "label": "30d train / 10d valid / 10d step",
    },
    {
        "grid_id": "train30_valid5_step5",
        "train_days": 30,
        "validation_days": 5,
        "step_days": 5,
        "label": "30d train / 5d valid / 5d step",
    },
    {
        "grid_id": "train40_valid10_step10",
        "train_days": 40,
        "validation_days": 10,
        "step_days": 10,
        "label": "40d train / 10d valid / 10d step",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY robustness report for ETH 15m breakout-pullback hold=8 vs hold=16 across multiple forward-slice grids."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def aggregate_selected_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    return COMPARE_MODULE.aggregate_trade_metrics(trades, pnl_field="net_pnl_pct", r_field="net_r_multiple")


def aggregate_gross_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    return COMPARE_MODULE.aggregate_trade_metrics(trades, pnl_field="pnl_pct", r_field="r_multiple")


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
    per_config_selected_validation_trades: dict[str, list[dict[str, Any]]] = {
        text(item["config_id"]): [] for item in COMPARE_MODULE.COMPARE_EXIT_CONFIGS
    }
    per_config_gross_validation_trades: dict[str, list[dict[str, Any]]] = {
        text(item["config_id"]): [] for item in COMPARE_MODULE.COMPARE_EXIT_CONFIGS
    }
    slice_rows: list[dict[str, Any]] = []
    for slice_info in slices:
        row: dict[str, Any] = {
            "slice_id": text(slice_info["slice_id"]),
            "validation_start_utc": text(slice_info["validation_start_utc"]),
            "validation_end_utc": text(slice_info["validation_end_utc"]),
            "configs": {},
        }
        for config in COMPARE_MODULE.COMPARE_EXIT_CONFIGS:
            config_id = text(config["config_id"])
            evaluated = COMPARE_MODULE.evaluate_fixed_exit(
                train_frame=slice_info["train_frame"],
                validation_frame=slice_info["validation_frame"],
                base_entry_params=base_entry_params,
                exit_params=dict(config["exit_params"]),
            )
            per_config_selected_validation_trades[config_id].extend(list(evaluated["validation_selected"]["trades"]))
            per_config_gross_validation_trades[config_id].extend(list(evaluated["validation_gross"]["trades"]))
            row["configs"][config_id] = {
                "validation_metrics": dict(evaluated["validation_selected"]["metrics"]),
                "validation_objective": float(evaluated["validation_objective"]),
                "validation_status": text(evaluated["validation_status"]),
            }

        hold8_metrics = dict(row["configs"]["hold_8_zero_risk"]["validation_metrics"])
        hold16_metrics = dict(row["configs"]["hold_16_zero_risk"]["validation_metrics"])
        row["winner_by_validation_return"] = COMPARE_MODULE.choose_winner(
            "hold_8_zero_risk",
            float(hold8_metrics.get("cumulative_return", 0.0) or 0.0),
            "hold_16_zero_risk",
            float(hold16_metrics.get("cumulative_return", 0.0) or 0.0),
        )
        row["winner_by_validation_objective"] = COMPARE_MODULE.choose_winner(
            "hold_8_zero_risk",
            float(row["configs"]["hold_8_zero_risk"]["validation_objective"]),
            "hold_16_zero_risk",
            float(row["configs"]["hold_16_zero_risk"]["validation_objective"]),
        )
        slice_rows.append(row)

    aggregate_validation_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_validation_gross_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_validation_objective_by_config: dict[str, float] = {}
    for config in COMPARE_MODULE.COMPARE_EXIT_CONFIGS:
        config_id = text(config["config_id"])
        aggregate_validation_metrics_by_config[config_id] = aggregate_selected_metrics(
            per_config_selected_validation_trades[config_id]
        )
        aggregate_validation_gross_metrics_by_config[config_id] = aggregate_gross_metrics(
            per_config_gross_validation_trades[config_id]
        )
        aggregate_validation_objective_by_config[config_id] = float(
            COMPARE_MODULE.BASE_MODULE.objective(aggregate_validation_metrics_by_config[config_id])
        )

    slice_return_wins = {
        config["config_id"]: int(sum(1 for row in slice_rows if row["winner_by_validation_return"] == config["config_id"]))
        for config in COMPARE_MODULE.COMPARE_EXIT_CONFIGS
    }
    slice_objective_wins = {
        config["config_id"]: int(sum(1 for row in slice_rows if row["winner_by_validation_objective"] == config["config_id"]))
        for config in COMPARE_MODULE.COMPARE_EXIT_CONFIGS
    }
    aggregate_return_winner = COMPARE_MODULE.choose_winner(
        "hold_8_zero_risk",
        float(aggregate_validation_metrics_by_config["hold_8_zero_risk"].get("cumulative_return", 0.0) or 0.0),
        "hold_16_zero_risk",
        float(aggregate_validation_metrics_by_config["hold_16_zero_risk"].get("cumulative_return", 0.0) or 0.0),
    )
    aggregate_objective_winner = COMPARE_MODULE.choose_winner(
        "hold_8_zero_risk",
        float(aggregate_validation_objective_by_config["hold_8_zero_risk"]),
        "hold_16_zero_risk",
        float(aggregate_validation_objective_by_config["hold_16_zero_risk"]),
    )
    slice_majority_return_winner = COMPARE_MODULE.choose_winner(
        "hold_8_zero_risk",
        float(slice_return_wins["hold_8_zero_risk"]),
        "hold_16_zero_risk",
        float(slice_return_wins["hold_16_zero_risk"]),
        eps=0.0,
    )
    slice_majority_objective_winner = COMPARE_MODULE.choose_winner(
        "hold_8_zero_risk",
        float(slice_objective_wins["hold_8_zero_risk"]),
        "hold_16_zero_risk",
        float(slice_objective_wins["hold_16_zero_risk"]),
        eps=0.0,
    )
    scheme_decision = "mixed_profile"
    if (
        aggregate_return_winner == "hold_8_zero_risk"
        and aggregate_objective_winner == "hold_8_zero_risk"
        and slice_majority_return_winner == "hold_8_zero_risk"
        and slice_majority_objective_winner == "hold_8_zero_risk"
    ):
        scheme_decision = "hold_8_scheme_leader"
    elif (
        aggregate_return_winner == "hold_16_zero_risk"
        and aggregate_objective_winner == "hold_16_zero_risk"
        and slice_majority_return_winner == "hold_16_zero_risk"
        and slice_majority_objective_winner == "hold_16_zero_risk"
    ):
        scheme_decision = "hold_16_scheme_leader"
    return {
        "grid_id": text(grid["grid_id"]),
        "label": text(grid["label"]),
        "train_days": int(grid["train_days"]),
        "validation_days": int(grid["validation_days"]),
        "step_days": int(grid["step_days"]),
        "slice_count": int(len(slice_rows)),
        "slice_rows": slice_rows,
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
        },
        "scheme_decision": scheme_decision,
    }


def build_overall_summary(grid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def count(field: str, winner: str) -> int:
        total = 0
        for row in grid_rows:
            if text((row.get("comparison_summary") or {}).get(field)) == winner:
                total += 1
        return total

    return {
        "scheme_count": int(len(grid_rows)),
        "aggregate_return_scheme_wins": {
            "hold_8_zero_risk": count("winner_by_aggregate_return", "hold_8_zero_risk"),
            "hold_16_zero_risk": count("winner_by_aggregate_return", "hold_16_zero_risk"),
            "tie": count("winner_by_aggregate_return", "tie"),
        },
        "aggregate_objective_scheme_wins": {
            "hold_8_zero_risk": count("winner_by_aggregate_objective", "hold_8_zero_risk"),
            "hold_16_zero_risk": count("winner_by_aggregate_objective", "hold_16_zero_risk"),
            "tie": count("winner_by_aggregate_objective", "tie"),
        },
        "slice_majority_return_scheme_wins": {
            "hold_8_zero_risk": count("winner_by_slice_majority_return", "hold_8_zero_risk"),
            "hold_16_zero_risk": count("winner_by_slice_majority_return", "hold_16_zero_risk"),
            "tie": count("winner_by_slice_majority_return", "tie"),
        },
        "slice_majority_objective_scheme_wins": {
            "hold_8_zero_risk": count("winner_by_slice_majority_objective", "hold_8_zero_risk"),
            "hold_16_zero_risk": count("winner_by_slice_majority_objective", "hold_16_zero_risk"),
            "tie": count("winner_by_slice_majority_objective", "tie"),
        },
        "scheme_decision_counts": {
            "hold_8_scheme_leader": int(sum(1 for row in grid_rows if text(row.get("scheme_decision")) == "hold_8_scheme_leader")),
            "hold_16_scheme_leader": int(sum(1 for row in grid_rows if text(row.get("scheme_decision")) == "hold_16_scheme_leader")),
            "mixed_profile": int(sum(1 for row in grid_rows if text(row.get("scheme_decision")) == "mixed_profile")),
        },
    }


def classify_research_decision(summary: dict[str, Any]) -> str:
    agg8 = int((summary.get("aggregate_return_scheme_wins") or {}).get("hold_8_zero_risk", 0))
    agg_obj8 = int((summary.get("aggregate_objective_scheme_wins") or {}).get("hold_8_zero_risk", 0))
    maj16 = int((summary.get("slice_majority_return_scheme_wins") or {}).get("hold_16_zero_risk", 0))
    maj_obj16 = int((summary.get("slice_majority_objective_scheme_wins") or {}).get("hold_16_zero_risk", 0))
    hold8_scheme = int((summary.get("scheme_decision_counts") or {}).get("hold_8_scheme_leader", 0))
    hold16_scheme = int((summary.get("scheme_decision_counts") or {}).get("hold_16_scheme_leader", 0))
    if agg8 >= 3 and agg_obj8 >= 3 and hold8_scheme >= 2:
        return "hold_8_price_state_only_candidate_strengthened"
    if maj16 >= 3 and maj_obj16 >= 3 and hold16_scheme >= 2:
        return "hold_16_consistency_reinforced_keep_baseline"
    return "mixed_robustness_keep_hold16_baseline_hold8_candidate"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = dict(payload.get("overall_summary") or {})
    lines = [
        "# Price Action Breakout Pullback Exit Hold Robustness SIM ONLY",
        "",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- dataset_path: `{text(payload.get('dataset_path'))}`",
        f"- base_artifact_path: `{text(payload.get('base_artifact_path'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        "",
        "## Overall Summary",
        "",
        f"- aggregate_return_scheme_wins: `{json.dumps(summary.get('aggregate_return_scheme_wins') or {}, ensure_ascii=False)}`",
        f"- aggregate_objective_scheme_wins: `{json.dumps(summary.get('aggregate_objective_scheme_wins') or {}, ensure_ascii=False)}`",
        f"- slice_majority_return_scheme_wins: `{json.dumps(summary.get('slice_majority_return_scheme_wins') or {}, ensure_ascii=False)}`",
        f"- slice_majority_objective_scheme_wins: `{json.dumps(summary.get('slice_majority_objective_scheme_wins') or {}, ensure_ascii=False)}`",
        f"- scheme_decision_counts: `{json.dumps(summary.get('scheme_decision_counts') or {}, ensure_ascii=False)}`",
        "",
        "## Grids",
        "",
    ]
    for row in payload.get("grid_rows", []):
        comp = dict(row.get("comparison_summary") or {})
        lines.extend(
            [
                f"### {text(row.get('grid_id'))}",
                f"- label: `{text(row.get('label'))}`",
                f"- slice_count: `{int(row.get('slice_count') or 0)}`",
                f"- scheme_decision: `{text(row.get('scheme_decision'))}`",
                f"- winner_by_aggregate_return: `{text(comp.get('winner_by_aggregate_return'))}`",
                f"- winner_by_aggregate_objective: `{text(comp.get('winner_by_aggregate_objective'))}`",
                f"- winner_by_slice_majority_return: `{text(comp.get('winner_by_slice_majority_return'))}`",
                f"- winner_by_slice_majority_objective: `{text(comp.get('winner_by_slice_majority_objective'))}`",
                "",
            ]
        )
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
        "action": "build_price_action_breakout_pullback_exit_hold_robustness_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": COMPARE_MODULE.BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "symbol": symbol,
        "coverage_start_utc": COMPARE_MODULE.BASE_MODULE.fmt_utc(coverage_start),
        "coverage_end_utc": COMPARE_MODULE.BASE_MODULE.fmt_utc(coverage_end),
        "cadence_minutes": COMPARE_MODULE.cadence_minutes(frame),
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "comparison_configs": COMPARE_MODULE.COMPARE_EXIT_CONFIGS,
        "robustness_grids": ROBUSTNESS_GRIDS,
        "grid_rows": grid_rows,
        "overall_summary": overall_summary,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:exit_hold_robustness:{COMPARE_MODULE.BASE_MODULE.SELECTION_SCENARIO_ID}:"
            f"agg8={int((overall_summary.get('aggregate_return_scheme_wins') or {}).get('hold_8_zero_risk', 0))},"
            f"maj16={int((overall_summary.get('slice_majority_return_scheme_wins') or {}).get('hold_16_zero_risk', 0))},"
            f"grids={int(overall_summary.get('scheme_count') or 0)},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "在允许范围内继续 price-state-only 主线；"
            "使用同一组固定 base entry，对 hold=8 vs hold=16 做多切片网格稳健性比较。"
        ),
        "limitation_note": (
            "它仍然只覆盖 public 15m OHLCV 和固定 entry；"
            "不包含 orderflow，也不构成 live 放行。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_hold_robustness_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_hold_robustness_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_hold_robustness_sim_only.json"
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
