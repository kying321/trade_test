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

HOLD_CONFIGS: dict[str, dict[str, Any]] = {
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
        description="Build a SIM_ONLY cost-sensitivity report for the ETH hold frontier (hold8/hold16/hold24)."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def latest_artifact(review_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(review_dir.glob(pattern))
    if not candidates:
        return None
    return candidates[-1]


def aggregate_metrics_for_scenario(trades: list[dict[str, Any]], scenario_id: str) -> dict[str, Any]:
    if scenario_id == "gross":
        return COMPARE_MODULE.aggregate_trade_metrics(trades, pnl_field="pnl_pct", r_field="r_multiple")
    return COMPARE_MODULE.aggregate_trade_metrics(trades, pnl_field="net_pnl_pct", r_field="net_r_multiple")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback Hold Frontier Cost Sensitivity SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        f"- source_head_status: `{text(payload.get('source_head_status'))}`",
        f"- canonical_source_head: `{text(payload.get('canonical_source_head'))}`",
        "",
        "## Scenario Summary",
        "",
    ]
    for row in payload.get("scenario_rows", []):
        summary = dict(row.get("summary") or {})
        lines.append(
            f"- `{row['scenario_id']}` | winner_by_aggregate_return=`{json.dumps(summary.get('winner_by_aggregate_return') or {}, ensure_ascii=False)}` | "
            f"winner_by_aggregate_objective=`{json.dumps(summary.get('winner_by_aggregate_objective') or {}, ensure_ascii=False)}`"
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
    handoff_path = latest_artifact(review_dir, "latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json")
    handoff_payload = json.loads(handoff_path.read_text(encoding="utf-8")) if handoff_path else {}

    scenario_rows: list[dict[str, Any]] = []
    scenario_role_consistency = []
    for scenario in COMPARE_MODULE.BASE_MODULE.EXECUTION_COST_SCENARIOS:
        scenario_id = text(scenario.get("scenario_id"))
        grid_rows: list[dict[str, Any]] = []
        winner_by_aggregate_return = {config_id: 0 for config_id in HOLD_CONFIGS}
        winner_by_aggregate_objective = {config_id: 0 for config_id in HOLD_CONFIGS}
        for grid in ROBUSTNESS_GRIDS:
            slices = COMPARE_MODULE.build_forward_slices(
                frame,
                train_days=int(grid["train_days"]),
                validation_days=int(grid["validation_days"]),
                step_days=int(grid["step_days"]),
            )
            aggregate_trades_by_config = {config_id: [] for config_id in HOLD_CONFIGS}
            for slice_info in slices:
                for config_id, exit_params in HOLD_CONFIGS.items():
                    params = dict(base_entry_params)
                    params.update(exit_params)
                    gross = COMPARE_MODULE.EXIT_MODULE.simulate_symbol_with_exit_risk(slice_info["validation_frame"], params)
                    if scenario_id == "gross":
                        trades = list(gross["trades"])
                    else:
                        stressed = COMPARE_MODULE.apply_cost_scenario_full(gross, scenario)
                        trades = list(stressed["trades"])
                    aggregate_trades_by_config[config_id].extend(trades)

            aggregate_metrics_by_config: dict[str, dict[str, Any]] = {}
            aggregate_objective_by_config: dict[str, float] = {}
            for config_id, trades in aggregate_trades_by_config.items():
                metrics = aggregate_metrics_for_scenario(trades, scenario_id)
                aggregate_metrics_by_config[config_id] = metrics
                aggregate_objective_by_config[config_id] = float(COMPARE_MODULE.BASE_MODULE.objective(metrics))

            ret_winner = max(
                HOLD_CONFIGS,
                key=lambda config_id: float((aggregate_metrics_by_config.get(config_id) or {}).get("cumulative_return", 0.0) or 0.0),
            )
            obj_winner = max(
                HOLD_CONFIGS,
                key=lambda config_id: float(aggregate_objective_by_config.get(config_id, 0.0) or 0.0),
            )
            winner_by_aggregate_return[ret_winner] += 1
            winner_by_aggregate_objective[obj_winner] += 1
            grid_rows.append(
                {
                    "grid_id": text(grid["grid_id"]),
                    "train_days": int(grid["train_days"]),
                    "validation_days": int(grid["validation_days"]),
                    "step_days": int(grid["step_days"]),
                    "winner_by_aggregate_return": ret_winner,
                    "winner_by_aggregate_objective": obj_winner,
                    "aggregate_metrics_by_config": aggregate_metrics_by_config,
                    "aggregate_objective_by_config": aggregate_objective_by_config,
                }
            )

        summary = {
            "winner_by_aggregate_return": winner_by_aggregate_return,
            "winner_by_aggregate_objective": winner_by_aggregate_objective,
        }
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "label": text(scenario.get("label")),
                "fee_bps_per_side": float(scenario.get("fee_bps_per_side", 0.0) or 0.0),
                "slippage_bps_per_side": float(scenario.get("slippage_bps_per_side", 0.0) or 0.0),
                "grid_rows": grid_rows,
                "summary": summary,
            }
        )
        scenario_role_consistency.append(
            {
                "scenario_id": scenario_id,
                "hold24_return_wins_all_grids": int(winner_by_aggregate_return.get("hold24_zero", 0)) == len(ROBUSTNESS_GRIDS),
                "hold8_objective_wins_all_grids": int(winner_by_aggregate_objective.get("hold8_zero", 0)) == len(ROBUSTNESS_GRIDS),
            }
        )

    dual_role_stable = all(
        bool(row.get("hold24_return_wins_all_grids")) and bool(row.get("hold8_objective_wins_all_grids"))
        for row in scenario_role_consistency
    )
    research_decision = "frontier_cost_sensitivity_mixed"
    if dual_role_stable:
        research_decision = "frontier_roles_cost_stable_keep_hold16_baseline_hold8_objective_hold24_return"
    source_head_status = "frontier_cost_head_active"
    canonical_source_head = str(review_dir / "latest_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json")
    consumer_rule = "cost_sensitivity 可作为当前 frontier 辅助头读取。"
    if text(handoff_payload.get("research_decision")) == "use_hold_selection_gate_as_canonical_head":
        source_head_status = "superseded_by_hold_selection_handoff"
        canonical_source_head = text(handoff_payload.get("canonical_source_head")) or str(handoff_path)
        consumer_rule = text(handoff_payload.get("consumer_rule")) or consumer_rule

    payload = {
        "action": "build_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": COMPARE_MODULE.BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "symbol": symbol,
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "source_head_status": source_head_status,
        "canonical_source_head": canonical_source_head,
        "consumer_rule": consumer_rule,
        "superseded_by_handoff_path": str(handoff_path) if handoff_path else "",
        "cost_scenarios": list(COMPARE_MODULE.BASE_MODULE.EXECUTION_COST_SCENARIOS),
        "robustness_grids": ROBUSTNESS_GRIDS,
        "scenario_rows": scenario_rows,
        "scenario_role_consistency": scenario_role_consistency,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:hold_frontier_cost_sensitivity:{COMPARE_MODULE.BASE_MODULE.SELECTION_SCENARIO_ID}:"
            f"scenarios={len(scenario_rows)},"
            f"dual_role_stable={str(dual_role_stable).lower()},"
            f"head_status={source_head_status},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份工件验证 frontier 角色是否在 gross / moderate / stress 三种成本下保持稳定；"
            "如果稳定，就说明双候选结构不是成本假象。"
            + (" 但当 hold_selection_handoff 已存在时，这份工件只保留为历史成本证据，不再是 canonical head。"
               if source_head_status == "superseded_by_hold_selection_handoff" else "")
        ),
        "limitation_note": (
            "它仍然只基于 60d ETH 15m public OHLCV；"
            "结论是成本稳健性，不是 baseline promotion 结论。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_frontier_cost_sensitivity_sim_only.json"
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
