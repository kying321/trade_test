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
HOLD_FORWARD_COMPARE_SCRIPT_PATH = Path(__file__).resolve().with_name(
    "build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py"
)


BASE_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_base", BASE_SCRIPT_PATH)
BASE_MODULE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE_MODULE)

EXIT_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_exit_risk", EXIT_SCRIPT_PATH)
EXIT_MODULE = importlib.util.module_from_spec(EXIT_SPEC)
assert EXIT_SPEC is not None and EXIT_SPEC.loader is not None
EXIT_SPEC.loader.exec_module(EXIT_MODULE)

HOLD_FORWARD_COMPARE_SPEC = importlib.util.spec_from_file_location(
    "fenlie_price_action_breakout_pullback_exit_hold_forward_compare",
    HOLD_FORWARD_COMPARE_SCRIPT_PATH,
)
HOLD_FORWARD_COMPARE_MODULE = importlib.util.module_from_spec(HOLD_FORWARD_COMPARE_SPEC)
assert HOLD_FORWARD_COMPARE_SPEC is not None and HOLD_FORWARD_COMPARE_SPEC.loader is not None
HOLD_FORWARD_COMPARE_SPEC.loader.exec_module(HOLD_FORWARD_COMPARE_MODULE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SIM_ONLY forward OOS comparison for the exit/risk blocker challenge pair."
    )
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--base-artifact-path", default="")
    parser.add_argument("--challenge-pair-path", required=True)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--train-days", type=int, default=40)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--step-days", type=int, default=10)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def normalize_exit_params(payload: Any) -> dict[str, Any]:
    data = dict(payload or {})
    return {
        "max_hold_bars": int(data.get("max_hold_bars") or 0),
        "break_even_trigger_r": float(data.get("break_even_trigger_r") or 0.0),
        "trailing_stop_atr": float(data.get("trailing_stop_atr") or 0.0),
        "cooldown_after_losses": int(data.get("cooldown_after_losses") or 0),
        "cooldown_bars": int(data.get("cooldown_bars") or 0),
    }


def decision_token(value: str) -> str:
    normalized = text(value)
    if normalized == "baseline_pair":
        return "baseline"
    if normalized == "challenger_pair":
        return "challenger"
    if normalized == "tie":
        return "tie"
    return "mixed"


def classify_pair_research_decision(
    *,
    aggregate_return_winner: str,
    aggregate_objective_winner: str,
    slice_majority_return_winner: str,
    slice_majority_objective_winner: str,
) -> str:
    if (
        aggregate_return_winner == "challenger_pair"
        and aggregate_objective_winner == "challenger_pair"
        and slice_majority_return_winner == "challenger_pair"
        and slice_majority_objective_winner == "challenger_pair"
    ):
        return "challenger_forward_oos_pair_wins"
    if (
        aggregate_return_winner == "baseline_pair"
        and aggregate_objective_winner == "baseline_pair"
        and slice_majority_return_winner == "baseline_pair"
        and slice_majority_objective_winner == "baseline_pair"
    ):
        return "baseline_forward_oos_pair_keeps_anchor"
    return (
        "mixed_forward_oos_pair"
        f"_agg_ret_{decision_token(aggregate_return_winner)}"
        f"_agg_obj_{decision_token(aggregate_objective_winner)}"
        f"_slice_ret_{decision_token(slice_majority_return_winner)}"
        f"_slice_obj_{decision_token(slice_majority_objective_winner)}"
    )


def load_challenge_pair(path: Path, *, symbol: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json_mapping(path)
    pair = dict(payload.get("challenge_pair") or {})
    baseline = normalize_exit_params(pair.get("baseline_exit_params"))
    challenger = normalize_exit_params(pair.get("challenger_exit_params"))
    if not baseline or not challenger:
        raise ValueError(f"missing_challenge_pair:{path}")
    pair_symbol = text(payload.get("symbol") or symbol).upper()
    if pair_symbol != text(symbol).upper():
        raise ValueError(f"challenge_pair_symbol_mismatch:{pair_symbol}:{text(symbol).upper()}")
    return payload, [
        {
            "config_id": "baseline_pair",
            "label": (
                f"baseline hold={baseline['max_hold_bars']}, "
                f"be={baseline['break_even_trigger_r']}, trail={baseline['trailing_stop_atr']}"
            ),
            "exit_params": baseline,
        },
        {
            "config_id": "challenger_pair",
            "label": (
                f"challenger hold={challenger['max_hold_bars']}, "
                f"be={challenger['break_even_trigger_r']}, trail={challenger['trailing_stop_atr']}"
            ),
            "exit_params": challenger,
        },
    ]


def render_markdown(payload: dict[str, Any]) -> str:
    summary = dict(payload.get("comparison_summary") or {})
    aggregate = dict(payload.get("aggregate_validation_metrics_by_config") or {})
    lines = [
        "# Price Action Breakout Pullback Exit Risk Forward Compare SIM ONLY",
        "",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- challenge_pair_path: `{text(payload.get('challenge_pair_path'))}`",
        f"- challenge_pair_source_decision: `{text(payload.get('challenge_pair_source_decision'))}`",
        f"- train_days: `{payload.get('train_days')}`",
        f"- validation_days: `{payload.get('validation_days')}`",
        f"- step_days: `{payload.get('step_days')}`",
        f"- validation_window_mode: `{text(payload.get('validation_window_mode'))}`",
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
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    stamp_dt = BASE_MODULE.parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser().resolve() if text(args.dataset_path) else BASE_MODULE.select_latest_intraday_dataset(review_dir)
    base_artifact_path = Path(args.base_artifact_path).expanduser().resolve() if text(args.base_artifact_path) else EXIT_MODULE.select_latest_price_action_artifact(review_dir)
    challenge_pair_path = Path(args.challenge_pair_path).expanduser().resolve()
    symbol = text(args.symbol).upper()

    base_entry_params, base_payload = EXIT_MODULE.load_base_entry_params(base_artifact_path, symbol)
    challenge_payload, comparison_configs = load_challenge_pair(challenge_pair_path, symbol=symbol)
    frame = BASE_MODULE.add_features(BASE_MODULE.load_frame(dataset_path))
    frame = frame[frame["symbol"] == symbol].copy().sort_values("ts").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"symbol_not_found_in_dataset:{symbol}")

    slices = HOLD_FORWARD_COMPARE_MODULE.build_forward_slices(
        frame,
        train_days=int(args.train_days),
        validation_days=int(args.validation_days),
        step_days=int(args.step_days),
    )

    per_config_selected_validation_trades: dict[str, list[dict[str, Any]]] = {
        text(item["config_id"]): [] for item in comparison_configs
    }
    per_config_gross_validation_trades: dict[str, list[dict[str, Any]]] = {
        text(item["config_id"]): [] for item in comparison_configs
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
        for config in comparison_configs:
            config_id = text(config["config_id"])
            evaluated = HOLD_FORWARD_COMPARE_MODULE.evaluate_fixed_exit(
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

        baseline_metrics = dict(row["configs"]["baseline_pair"]["validation_metrics"])
        challenger_metrics = dict(row["configs"]["challenger_pair"]["validation_metrics"])
        row["winner_by_validation_return"] = HOLD_FORWARD_COMPARE_MODULE.choose_winner(
            "baseline_pair",
            float(baseline_metrics.get("cumulative_return", 0.0) or 0.0),
            "challenger_pair",
            float(challenger_metrics.get("cumulative_return", 0.0) or 0.0),
        )
        row["winner_by_validation_objective"] = HOLD_FORWARD_COMPARE_MODULE.choose_winner(
            "baseline_pair",
            float(row["configs"]["baseline_pair"]["validation_objective"]),
            "challenger_pair",
            float(row["configs"]["challenger_pair"]["validation_objective"]),
        )
        forward_slices.append(row)

    aggregate_validation_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_validation_gross_metrics_by_config: dict[str, dict[str, Any]] = {}
    aggregate_validation_objective_by_config: dict[str, float] = {}
    for config in comparison_configs:
        config_id = text(config["config_id"])
        aggregate_validation_metrics_by_config[config_id] = HOLD_FORWARD_COMPARE_MODULE.aggregate_trade_metrics(
            per_config_selected_validation_trades[config_id],
            pnl_field="net_pnl_pct",
            r_field="net_r_multiple",
        )
        aggregate_validation_gross_metrics_by_config[config_id] = HOLD_FORWARD_COMPARE_MODULE.aggregate_trade_metrics(
            per_config_gross_validation_trades[config_id],
            pnl_field="pnl_pct",
            r_field="r_multiple",
        )
        aggregate_validation_objective_by_config[config_id] = float(
            BASE_MODULE.objective(aggregate_validation_metrics_by_config[config_id])
        )

    slice_return_wins = {
        config["config_id"]: int(sum(1 for row in forward_slices if row["winner_by_validation_return"] == config["config_id"]))
        for config in comparison_configs
    }
    slice_objective_wins = {
        config["config_id"]: int(sum(1 for row in forward_slices if row["winner_by_validation_objective"] == config["config_id"]))
        for config in comparison_configs
    }
    slice_majority_return_winner = HOLD_FORWARD_COMPARE_MODULE.choose_winner(
        "baseline_pair",
        float(slice_return_wins["baseline_pair"]),
        "challenger_pair",
        float(slice_return_wins["challenger_pair"]),
        eps=0.0,
    )
    slice_majority_objective_winner = HOLD_FORWARD_COMPARE_MODULE.choose_winner(
        "baseline_pair",
        float(slice_objective_wins["baseline_pair"]),
        "challenger_pair",
        float(slice_objective_wins["challenger_pair"]),
        eps=0.0,
    )
    aggregate_return_winner = HOLD_FORWARD_COMPARE_MODULE.choose_winner(
        "baseline_pair",
        float(aggregate_validation_metrics_by_config["baseline_pair"].get("cumulative_return", 0.0) or 0.0),
        "challenger_pair",
        float(aggregate_validation_metrics_by_config["challenger_pair"].get("cumulative_return", 0.0) or 0.0),
    )
    aggregate_objective_winner = HOLD_FORWARD_COMPARE_MODULE.choose_winner(
        "baseline_pair",
        float(aggregate_validation_objective_by_config["baseline_pair"]),
        "challenger_pair",
        float(aggregate_validation_objective_by_config["challenger_pair"]),
    )
    research_decision = classify_pair_research_decision(
        aggregate_return_winner=aggregate_return_winner,
        aggregate_objective_winner=aggregate_objective_winner,
        slice_majority_return_winner=slice_majority_return_winner,
        slice_majority_objective_winner=slice_majority_objective_winner,
    )

    coverage_start = pd.Timestamp(frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    coverage_end = pd.Timestamp(frame["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    window_mode = HOLD_FORWARD_COMPARE_MODULE.validation_window_mode(
        step_days=args.step_days,
        validation_days=args.validation_days,
    )
    pair = dict(challenge_payload.get("challenge_pair") or {})
    baseline_hold = int(pair.get("baseline_hold_bars") or comparison_configs[0]["exit_params"]["max_hold_bars"])
    challenger_hold = int(pair.get("challenger_hold_bars") or comparison_configs[1]["exit_params"]["max_hold_bars"])
    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_forward_compare_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "challenge_pair_path": str(challenge_pair_path),
        "challenge_pair_source_decision": text(challenge_payload.get("research_decision")),
        "symbol": symbol,
        "coverage_start_utc": BASE_MODULE.fmt_utc(coverage_start),
        "coverage_end_utc": BASE_MODULE.fmt_utc(coverage_end),
        "cadence_minutes": HOLD_FORWARD_COMPARE_MODULE.cadence_minutes(frame),
        "train_days": int(args.train_days),
        "validation_days": int(args.validation_days),
        "step_days": int(args.step_days),
        "validation_window_mode": window_mode,
        "slice_count": int(len(forward_slices)),
        "selection_scenario_id": BASE_MODULE.SELECTION_SCENARIO_ID,
        "source_catalog": list(BASE_MODULE.SOURCE_CATALOG),
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "base_entry_params": base_entry_params,
        "comparison_configs": comparison_configs,
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
        },
        "research_decision": research_decision,
        "recommended_brief": (
            f"{symbol}:exit_risk_forward_compare:"
            f"pair=hold{challenger_hold}_vs_hold{baseline_hold},"
            f"slices={len(forward_slices)},"
            f"agg_return_winner={aggregate_return_winner},"
            f"decision={research_decision}"
        ),
        "research_note": (
            "复用 blocker 的 challenge_pair 做固定参数 forward OOS；"
            f"当前只验证 hold{challenger_hold} challenger 与 hold{baseline_hold} baseline，"
            "不在每个 slice 内重新选参。"
        ),
        "limitation_note": (
            "这仍是 public 15m OHLCV 的 SIM_ONLY 近似；"
            "若结果要晋级，后续还要继续做更多窗口/尾部与现实执行约束验证。"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.json"
    latest_md_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_forward_compare_sim_only.md"
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
                "slice_count": len(forward_slices),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
