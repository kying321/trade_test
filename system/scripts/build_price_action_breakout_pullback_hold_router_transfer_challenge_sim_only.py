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
BASE_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_sim_only.py")
EXIT_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_exit_risk_sim_only.py")
COMPARE_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py")
ROUTER_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_hold_router_hypothesis_sim_only.py")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE_MODULE = load_module("fenlie_price_action_breakout_pullback_base", BASE_SCRIPT_PATH)
EXIT_MODULE = load_module("fenlie_price_action_breakout_pullback_exit_risk", EXIT_SCRIPT_PATH)
COMPARE_MODULE = load_module("fenlie_hold_forward_compare", COMPARE_SCRIPT_PATH)
ROUTER_MODULE = load_module("fenlie_hold_router_hypothesis", ROUTER_SCRIPT_PATH)


HOLD_CONFIGS: list[tuple[str, int]] = [
    ("hold8_zero", 8),
    ("hold16_zero", 16),
    ("hold24_zero", 24),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY frozen-threshold transfer challenge report for the price-state hold router."
    )
    parser.add_argument("--long-dataset-path", required=True)
    parser.add_argument("--derivation-dataset-path", required=True)
    parser.add_argument("--base-artifact-path", required=True)
    parser.add_argument("--router-artifact-path", required=True)
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--stamp", required=True)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def frame_window_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "row_count": 0,
            "coverage_start_utc": "",
            "coverage_end_utc": "",
        }
    return {
        "row_count": int(len(frame)),
        "coverage_start_utc": BASE_MODULE.fmt_utc(pd.Timestamp(frame["ts"].min()).to_pydatetime()),
        "coverage_end_utc": BASE_MODULE.fmt_utc(pd.Timestamp(frame["ts"].max()).to_pydatetime()),
    }


def objective(metrics: dict[str, Any]) -> float:
    return float(BASE_MODULE.objective(metrics))


def summarize_strategies(trades_by_strategy: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, float]]:
    metrics_by_strategy: dict[str, Any] = {}
    objective_by_strategy: dict[str, float] = {}
    for strategy_id, trades in trades_by_strategy.items():
        metrics = COMPARE_MODULE.aggregate_trade_metrics(trades, pnl_field="net_pnl_pct", r_field="net_r_multiple")
        metrics_by_strategy[strategy_id] = metrics
        objective_by_strategy[strategy_id] = objective(metrics)
    return metrics_by_strategy, objective_by_strategy


def winner_by_metric(metrics_by_strategy: dict[str, Any], *, field: str) -> str:
    best_name = ""
    best_value: float | None = None
    for name, metrics in metrics_by_strategy.items():
        value = float(metrics.get(field, 0.0) or 0.0)
        if best_value is None or value > best_value:
            best_name = name
            best_value = value
    return best_name


def winner_by_objective(objective_by_strategy: dict[str, float]) -> str:
    best_name = ""
    best_value: float | None = None
    for name, value in objective_by_strategy.items():
        if best_value is None or float(value) > best_value:
            best_name = name
            best_value = float(value)
    return best_name


def evaluate_window(
    *,
    frame: pd.DataFrame,
    base_entry_params: dict[str, Any],
    frozen_router: dict[str, Any],
) -> dict[str, Any]:
    grid_rows: list[dict[str, Any]] = []
    aggregate_trades_by_strategy = {
        "router": [],
        "hold8_zero": [],
        "hold16_zero": [],
        "hold24_zero": [],
    }
    objective_win_counts = Counter()
    return_win_counts = Counter()
    router_vs_hold8_objective = Counter()
    router_vs_hold8_return = Counter()
    router_chosen_hold_counts = Counter()
    eligible_grid_count = 0

    for grid in ROUTER_MODULE.ROBUSTNESS_GRIDS:
        try:
            slices = COMPARE_MODULE.build_forward_slices(
                frame,
                train_days=int(grid["train_days"]),
                validation_days=int(grid["validation_days"]),
                step_days=int(grid["step_days"]),
            )
        except Exception as exc:  # noqa: BLE001
            grid_rows.append(
                {
                    "grid_id": text(grid["grid_id"]),
                    "status": "blocked",
                    "blocked_reason": text(exc),
                }
            )
            continue

        eligible_grid_count += 1
        grid_trades_by_strategy = {
            "router": [],
            "hold8_zero": [],
            "hold16_zero": [],
            "hold24_zero": [],
        }
        chosen_hold_counts = Counter()
        for slice_info in slices:
            router_gross = ROUTER_MODULE.simulate_dynamic_router(slice_info["validation_frame"], base_entry_params, frozen_router)
            for trade in router_gross["trades"]:
                chosen_hold_counts.update([int(trade["chosen_hold"])])
                router_chosen_hold_counts.update([int(trade["chosen_hold"])])
            router_stressed = ROUTER_MODULE.apply_cost_full(router_gross, ROUTER_MODULE.SELECTION_SCENARIO)
            grid_trades_by_strategy["router"].extend(list(router_stressed["trades"]))

            for strategy_id, hold in HOLD_CONFIGS:
                evaluated = COMPARE_MODULE.evaluate_fixed_exit(
                    train_frame=slice_info["train_frame"],
                    validation_frame=slice_info["validation_frame"],
                    base_entry_params=base_entry_params,
                    exit_params={
                        "max_hold_bars": hold,
                        "break_even_trigger_r": 0.0,
                        "trailing_stop_atr": 0.0,
                        "cooldown_after_losses": 0,
                        "cooldown_bars": 0,
                    },
                )
                grid_trades_by_strategy[strategy_id].extend(list(evaluated["validation_selected"]["trades"]))

        metrics_by_strategy, objective_by_strategy = summarize_strategies(grid_trades_by_strategy)
        objective_winner = winner_by_objective(objective_by_strategy)
        return_winner = winner_by_metric(metrics_by_strategy, field="cumulative_return")
        objective_win_counts.update([objective_winner])
        return_win_counts.update([return_winner])

        router_obj = float(objective_by_strategy["router"])
        hold8_obj = float(objective_by_strategy["hold8_zero"])
        if router_obj > hold8_obj:
            router_vs_hold8_objective_result = "win"
            router_vs_hold8_objective.update(["win"])
        elif router_obj < hold8_obj:
            router_vs_hold8_objective_result = "loss"
            router_vs_hold8_objective.update(["loss"])
        else:
            router_vs_hold8_objective_result = "tie"
            router_vs_hold8_objective.update(["tie"])

        router_ret = float(metrics_by_strategy["router"].get("cumulative_return", 0.0) or 0.0)
        hold8_ret = float(metrics_by_strategy["hold8_zero"].get("cumulative_return", 0.0) or 0.0)
        if router_ret > hold8_ret:
            router_vs_hold8_return_result = "win"
            router_vs_hold8_return.update(["win"])
        elif router_ret < hold8_ret:
            router_vs_hold8_return_result = "loss"
            router_vs_hold8_return.update(["loss"])
        else:
            router_vs_hold8_return_result = "tie"
            router_vs_hold8_return.update(["tie"])

        for strategy_id, trades in grid_trades_by_strategy.items():
            aggregate_trades_by_strategy[strategy_id].extend(list(trades))

        grid_rows.append(
            {
                "grid_id": text(grid["grid_id"]),
                "status": "ok",
                "slice_count": int(len(slices)),
                "router_chosen_hold_counts": dict(chosen_hold_counts),
                "metrics_by_strategy": metrics_by_strategy,
                "objective_by_strategy": objective_by_strategy,
                "winner_by_aggregate_return": return_winner,
                "winner_by_aggregate_objective": objective_winner,
                "router_vs_hold8_return": router_vs_hold8_return_result,
                "router_vs_hold8_objective": router_vs_hold8_objective_result,
            }
        )

    aggregate_metrics_by_strategy, aggregate_objective_by_strategy = summarize_strategies(aggregate_trades_by_strategy)
    return {
        "window_summary": frame_window_summary(frame),
        "eligible_grid_count": int(eligible_grid_count),
        "grid_rows": grid_rows,
        "aggregate_metrics_by_strategy": aggregate_metrics_by_strategy,
        "aggregate_objective_by_strategy": aggregate_objective_by_strategy,
        "aggregate_winner_by_return": winner_by_metric(aggregate_metrics_by_strategy, field="cumulative_return"),
        "aggregate_winner_by_objective": winner_by_objective(aggregate_objective_by_strategy),
        "objective_win_counts": dict(objective_win_counts),
        "return_win_counts": dict(return_win_counts),
        "router_vs_hold8_objective": dict(router_vs_hold8_objective),
        "router_vs_hold8_return": dict(router_vs_hold8_return),
        "router_chosen_hold_counts_total": dict(router_chosen_hold_counts),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    hist = dict(payload.get("historical_transfer_challenge") or {})
    fut = dict(payload.get("future_tail_probe") or {})
    frozen = dict(payload.get("frozen_router") or {})
    lines = [
        "# Price Action Breakout Pullback Hold Router Transfer Challenge SIM ONLY",
        "",
        f"- generated_at_utc: `{text(payload.get('generated_at_utc'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Frozen Router",
        "",
        f"- router_id: `{text(frozen.get('router_id'))}`",
        f"- feature: `{text(frozen.get('feature'))}`",
        f"- direction: `{text(frozen.get('direction'))}`",
        f"- threshold: `{frozen.get('threshold')}`",
        "",
        "## Historical Transfer Challenge",
        "",
        f"- coverage: `{text(hist.get('window_summary', {}).get('coverage_start_utc'))} -> {text(hist.get('window_summary', {}).get('coverage_end_utc'))}`",
        f"- row_count: `{hist.get('window_summary', {}).get('row_count')}`",
        f"- eligible_grid_count: `{hist.get('eligible_grid_count')}`",
        f"- aggregate_winner_by_return: `{text(hist.get('aggregate_winner_by_return'))}`",
        f"- aggregate_winner_by_objective: `{text(hist.get('aggregate_winner_by_objective'))}`",
        f"- router_vs_hold8_return: `{json.dumps(hist.get('router_vs_hold8_return', {}), ensure_ascii=False)}`",
        f"- router_vs_hold8_objective: `{json.dumps(hist.get('router_vs_hold8_objective', {}), ensure_ascii=False)}`",
        "",
        "## Future Tail Probe",
        "",
        f"- coverage: `{text(fut.get('window_summary', {}).get('coverage_start_utc'))} -> {text(fut.get('window_summary', {}).get('coverage_end_utc'))}`",
        f"- row_count: `{fut.get('window_summary', {}).get('row_count')}`",
        f"- eligible_grid_count: `{fut.get('eligible_grid_count')}`",
        "",
        "## Notes",
        "",
        f"- `{text(payload.get('research_note'))}`",
        f"- `{text(payload.get('limitation_note'))}`",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    stamp_dt = BASE_MODULE.parse_stamp(args.stamp)
    long_dataset_path = Path(args.long_dataset_path).expanduser().resolve()
    derivation_dataset_path = Path(args.derivation_dataset_path).expanduser().resolve()
    base_artifact_path = Path(args.base_artifact_path).expanduser().resolve()
    router_artifact_path = Path(args.router_artifact_path).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    symbol = text(args.symbol).upper()

    base_entry_params, base_payload = EXIT_MODULE.load_base_entry_params(base_artifact_path, symbol)
    router_payload = json.loads(router_artifact_path.read_text(encoding="utf-8"))
    frozen_router = dict(router_payload.get("best_router") or {})
    if not frozen_router:
        raise ValueError("router_artifact_missing_best_router")

    long_frame = ROUTER_MODULE.load_signal_frame(long_dataset_path, symbol, int(base_entry_params["breakout_lookback"]))
    derivation_frame = ROUTER_MODULE.load_signal_frame(derivation_dataset_path, symbol, int(base_entry_params["breakout_lookback"]))
    derivation_start = pd.Timestamp(derivation_frame["ts"].min())
    derivation_end = pd.Timestamp(derivation_frame["ts"].max())

    historical_transfer_frame = long_frame[long_frame["ts"] < derivation_start].copy().reset_index(drop=True)
    future_tail_frame = long_frame[long_frame["ts"] > derivation_end].copy().reset_index(drop=True)

    historical_transfer = evaluate_window(
        frame=historical_transfer_frame,
        base_entry_params=base_entry_params,
        frozen_router=frozen_router,
    )
    future_tail_probe = evaluate_window(
        frame=future_tail_frame,
        base_entry_params=base_entry_params,
        frozen_router=frozen_router,
    )

    hist_router_obj = float((historical_transfer.get("aggregate_objective_by_strategy") or {}).get("router", 0.0) or 0.0)
    hist_hold8_obj = float((historical_transfer.get("aggregate_objective_by_strategy") or {}).get("hold8_zero", 0.0) or 0.0)
    future_eligible = int(future_tail_probe.get("eligible_grid_count", 0) or 0)
    research_decision = "router_transfer_mixed_keep_router_unpromoted"
    if int(historical_transfer.get("eligible_grid_count", 0) or 0) <= 0:
        research_decision = "historical_transfer_unavailable_keep_router_unpromoted"
    elif hist_router_obj < hist_hold8_obj:
        research_decision = "frozen_router_transfer_does_not_beat_hold8_future_tail_insufficient" if future_eligible == 0 else "frozen_router_transfer_does_not_beat_hold8"
    elif future_eligible == 0:
        research_decision = "frozen_router_positive_on_historical_transfer_but_future_tail_insufficient"

    payload = {
        "action": "build_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": BASE_MODULE.fmt_utc(stamp_dt),
        "symbol": symbol,
        "long_dataset_path": str(long_dataset_path),
        "derivation_dataset_path": str(derivation_dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "router_artifact_path": str(router_artifact_path),
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "selection_scenario_id": BASE_MODULE.SELECTION_SCENARIO_ID,
        "long_dataset_window": frame_window_summary(long_frame),
        "derivation_window": frame_window_summary(derivation_frame),
        "frozen_router": frozen_router,
        "historical_transfer_challenge": historical_transfer,
        "future_tail_probe": future_tail_probe,
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:hold_router_transfer:{BASE_MODULE.SELECTION_SCENARIO_ID}:"
            f"router={text(frozen_router.get('router_id'))},"
            f"hist_grids={int(historical_transfer.get('eligible_grid_count', 0) or 0)},"
            f"hist_obj_winner={text(historical_transfer.get('aggregate_winner_by_objective'))},"
            f"future_grids={future_eligible},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "这份工件冻结 router hypothesis 的 feature/threshold，不再重学；"
            "先看它在与 derivation window 完全不重叠的 transfer window 上是否还能胜出。"
        ),
        "limitation_note": (
            "当前 long dataset 的 derivation 之后只多出 2026-03-18T18:30:00Z 到 2026-03-19T03:15:00Z 的 36 根 15m bars，"
            "不足以做真正 forward challenge；因此当前结论主要来自 historical transfer，而不是新的未来 OOS。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_hold_router_transfer_challenge_sim_only.json"
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
