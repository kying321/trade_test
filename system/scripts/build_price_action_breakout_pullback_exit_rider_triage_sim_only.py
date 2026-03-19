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

TRIAGE_CONFIGS: list[dict[str, Any]] = [
    {
        "config_id": "hold16_zero_baseline",
        "label": "baseline hold=16 no rider",
        "exit_params": {
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
    {
        "config_id": "hold8_zero_candidate",
        "label": "candidate hold=8 no rider",
        "exit_params": {
            "max_hold_bars": 8,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
    {
        "config_id": "hold16_be075",
        "label": "baseline + break-even 0.75R",
        "exit_params": {
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.75,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
    {
        "config_id": "hold16_trail15",
        "label": "baseline + trailing 1.5 ATR",
        "exit_params": {
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 1.5,
            "cooldown_after_losses": 0,
            "cooldown_bars": 0,
        },
    },
    {
        "config_id": "hold16_cd2x16",
        "label": "baseline + cooldown after 2 losses for 16 bars",
        "exit_params": {
            "max_hold_bars": 16,
            "break_even_trigger_r": 0.0,
            "trailing_stop_atr": 0.0,
            "cooldown_after_losses": 2,
            "cooldown_bars": 16,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY triage report for simple exit/risk riders on the ETH price-state-only backbone."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


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
    selected_trades_by_config: dict[str, list[dict[str, Any]]] = {text(row["config_id"]): [] for row in TRIAGE_CONFIGS}
    gross_trades_by_config: dict[str, list[dict[str, Any]]] = {text(row["config_id"]): [] for row in TRIAGE_CONFIGS}
    slice_rows: list[dict[str, Any]] = []
    for slice_info in slices:
        row: dict[str, Any] = {
            "slice_id": text(slice_info["slice_id"]),
            "validation_start_utc": text(slice_info["validation_start_utc"]),
            "validation_end_utc": text(slice_info["validation_end_utc"]),
            "configs": {},
        }
        for config in TRIAGE_CONFIGS:
            config_id = text(config["config_id"])
            evaluated = COMPARE_MODULE.evaluate_fixed_exit(
                train_frame=slice_info["train_frame"],
                validation_frame=slice_info["validation_frame"],
                base_entry_params=base_entry_params,
                exit_params=dict(config["exit_params"]),
            )
            selected_trades_by_config[config_id].extend(list(evaluated["validation_selected"]["trades"]))
            gross_trades_by_config[config_id].extend(list(evaluated["validation_gross"]["trades"]))
            row["configs"][config_id] = {
                "validation_metrics": dict(evaluated["validation_selected"]["metrics"]),
                "validation_gross_metrics": dict(evaluated["validation_gross"]["metrics"]),
                "validation_status": text(evaluated["validation_status"]),
                "validation_objective": float(evaluated["validation_objective"]),
            }
        slice_rows.append(row)

    aggregate_selected_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_gross_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_objective_by_config: dict[str, float] = {}
    for config in TRIAGE_CONFIGS:
        config_id = text(config["config_id"])
        aggregate_selected_metrics_by_config[config_id] = COMPARE_MODULE.aggregate_trade_metrics(
            selected_trades_by_config[config_id],
            pnl_field="net_pnl_pct",
            r_field="net_r_multiple",
        )
        aggregate_gross_metrics_by_config[config_id] = COMPARE_MODULE.aggregate_trade_metrics(
            gross_trades_by_config[config_id],
            pnl_field="pnl_pct",
            r_field="r_multiple",
        )
        aggregate_objective_by_config[config_id] = float(
            COMPARE_MODULE.BASE_MODULE.objective(aggregate_selected_metrics_by_config[config_id])
        )

    return {
        "grid_id": text(grid["grid_id"]),
        "train_days": int(grid["train_days"]),
        "validation_days": int(grid["validation_days"]),
        "step_days": int(grid["step_days"]),
        "slice_count": int(len(slice_rows)),
        "slice_rows": slice_rows,
        "aggregate_selected_metrics_by_config": aggregate_selected_metrics_by_config,
        "aggregate_gross_metrics_by_config": aggregate_gross_metrics_by_config,
        "aggregate_objective_by_config": aggregate_objective_by_config,
    }


def compare_to_baseline(
    *,
    grid_rows: list[dict[str, Any]],
    candidate_id: str,
) -> dict[str, Any]:
    better_return = 0
    better_objective = 0
    equal_return = 0
    equal_objective = 0
    worse_return = 0
    worse_objective = 0
    exact_metric_match = True
    for row in grid_rows:
        baseline_metrics = dict((row.get("aggregate_selected_metrics_by_config") or {}).get("hold16_zero_baseline") or {})
        candidate_metrics = dict((row.get("aggregate_selected_metrics_by_config") or {}).get(candidate_id) or {})
        baseline_ret = float(baseline_metrics.get("cumulative_return", 0.0) or 0.0)
        candidate_ret = float(candidate_metrics.get("cumulative_return", 0.0) or 0.0)
        baseline_obj = float((row.get("aggregate_objective_by_config") or {}).get("hold16_zero_baseline", 0.0) or 0.0)
        candidate_obj = float((row.get("aggregate_objective_by_config") or {}).get(candidate_id, 0.0) or 0.0)
        if abs(candidate_ret - baseline_ret) <= 1e-12:
            equal_return += 1
        elif candidate_ret > baseline_ret:
            better_return += 1
        else:
            worse_return += 1
        if abs(candidate_obj - baseline_obj) <= 1e-12:
            equal_objective += 1
        elif candidate_obj > baseline_obj:
            better_objective += 1
        else:
            worse_objective += 1
        if candidate_metrics != baseline_metrics:
            exact_metric_match = False
    return {
        "candidate_id": candidate_id,
        "better_return_grids": int(better_return),
        "equal_return_grids": int(equal_return),
        "worse_return_grids": int(worse_return),
        "better_objective_grids": int(better_objective),
        "equal_objective_grids": int(equal_objective),
        "worse_objective_grids": int(worse_objective),
        "exact_metric_match_to_baseline_all_grids": bool(exact_metric_match),
    }


def build_triage_summary(grid_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_metrics = dict((grid_rows[0].get("aggregate_selected_metrics_by_config") or {}).get("hold16_zero_baseline") or {})
    return {
        "baseline_reference": baseline_metrics,
        "candidate_vs_baseline": [
            compare_to_baseline(grid_rows=grid_rows, candidate_id="hold8_zero_candidate"),
            compare_to_baseline(grid_rows=grid_rows, candidate_id="hold16_be075"),
            compare_to_baseline(grid_rows=grid_rows, candidate_id="hold16_trail15"),
            compare_to_baseline(grid_rows=grid_rows, candidate_id="hold16_cd2x16"),
        ],
    }


def classify_research_decision(summary: dict[str, Any]) -> str:
    rows = {text(row.get("candidate_id")): row for row in summary.get("candidate_vs_baseline", [])}
    hold8 = rows.get("hold8_zero_candidate", {})
    be = rows.get("hold16_be075", {})
    trail = rows.get("hold16_trail15", {})
    cooldown = rows.get("hold16_cd2x16", {})
    if (
        int(hold8.get("better_return_grids", 0)) >= 4
        and int(hold8.get("better_objective_grids", 0)) >= 4
        and int(be.get("worse_objective_grids", 0)) >= 4
        and int(trail.get("worse_objective_grids", 0)) >= 4
        and bool(cooldown.get("exact_metric_match_to_baseline_all_grids"))
    ):
        return "freeze_simple_riders_keep_hold16_baseline_hold8_candidate"
    return "simple_rider_triage_inconclusive"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Exit Rider Triage SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Candidate vs Baseline",
        "",
    ]
    for row in (payload.get("triage_summary") or {}).get("candidate_vs_baseline", []):
        lines.append(
            f"- `{row['candidate_id']}` | better_ret=`{row['better_return_grids']}` | "
            f"equal_ret=`{row['equal_return_grids']}` | worse_ret=`{row['worse_return_grids']}` | "
            f"better_obj=`{row['better_objective_grids']}` | equal_obj=`{row['equal_objective_grids']}` | "
            f"worse_obj=`{row['worse_objective_grids']}` | "
            f"exact_match=`{row['exact_metric_match_to_baseline_all_grids']}`"
        )
    lines.extend(["", "## Grids", ""])
    for row in payload.get("grid_rows", []):
        lines.append(f"### {text(row.get('grid_id'))}")
        for config in TRIAGE_CONFIGS:
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
    triage_summary = build_triage_summary(grid_rows)
    research_decision = classify_research_decision(triage_summary)

    coverage_start = pd.Timestamp(frame["ts"].min()).to_pydatetime().replace(tzinfo=COMPARE_MODULE.BASE_MODULE.dt.timezone.utc)
    coverage_end = pd.Timestamp(frame["ts"].max()).to_pydatetime().replace(tzinfo=COMPARE_MODULE.BASE_MODULE.dt.timezone.utc)
    payload = {
        "action": "build_price_action_breakout_pullback_exit_rider_triage_sim_only",
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
        "triage_configs": TRIAGE_CONFIGS,
        "robustness_grids": ROBUSTNESS_GRIDS,
        "grid_rows": grid_rows,
        "triage_summary": triage_summary,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:exit_rider_triage:{COMPARE_MODULE.BASE_MODULE.SELECTION_SCENARIO_ID}:"
            f"hold8_better_ret_grids={int(((triage_summary.get('candidate_vs_baseline') or [{}])[0]).get('better_return_grids', 0))},"
            f"be_worse_obj_grids={int(((triage_summary.get('candidate_vs_baseline') or [{}, {} ,{}])[1]).get('worse_objective_grids', 0))},"
            f"trail_worse_obj_grids={int(((triage_summary.get('candidate_vs_baseline') or [{}, {}, {}])[2]).get('worse_objective_grids', 0))},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "在 orderflow blocked 的前提下，只继续 price-state-only exit/risk triage；"
            "本轮只测试最小 rider 家族是否值得继续。"
        ),
        "limitation_note": (
            "它不代表穷举最优 exit grid，只是用 source-owned 方式先冻结最简单但无增益的 rider 家族。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_rider_triage_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_rider_triage_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_rider_triage_sim_only.json"
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
