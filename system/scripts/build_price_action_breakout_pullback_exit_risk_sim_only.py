#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
BASE_SCRIPT_PATH = Path(__file__).resolve().with_name("build_price_action_breakout_pullback_sim_only.py")


BASE_SPEC = importlib.util.spec_from_file_location("fenlie_price_action_breakout_pullback_base", BASE_SCRIPT_PATH)
BASE_MODULE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
BASE_SPEC.loader.exec_module(BASE_MODULE)


EXIT_PARAM_GRID: list[dict[str, Any]] = [
    {
        "max_hold_bars": max_hold_bars,
        "break_even_trigger_r": break_even_trigger_r,
        "trailing_stop_atr": trailing_stop_atr,
        "cooldown_after_losses": cooldown_after_losses,
        "cooldown_bars": cooldown_bars,
    }
    for max_hold_bars, break_even_trigger_r, trailing_stop_atr, cooldown_after_losses, cooldown_bars in itertools.product(
        [8, 12, 16, 24],
        [0.0, 0.75, 1.0],
        [0.0, 1.5, 2.0],
        [0, 2],
        [0, 16],
    )
    if (cooldown_after_losses == 0 and cooldown_bars == 0) or (cooldown_after_losses > 0 and cooldown_bars > 0)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exit/risk-only SIM_ONLY research for the Fenlie price-action breakout-pullback base setup.")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--base-artifact-path", default="")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    return parser.parse_args()


def text(value: Any) -> str:
    return str(value or "").strip()


def select_latest_price_action_artifact(review_dir: Path) -> Path:
    latest_alias = review_dir / "latest_price_action_breakout_pullback_sim_only.json"
    if latest_alias.exists():
        return latest_alias
    candidates = sorted(review_dir.glob("*_price_action_breakout_pullback_sim_only.json"))
    if not candidates:
        raise FileNotFoundError("no_price_action_breakout_pullback_sim_only_artifact_found")
    return candidates[-1]


def load_base_entry_params(artifact_path: Path, symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    symbol_upper = symbol.upper()
    for row in payload.get("symbol_results", []):
        if text(row.get("symbol")).upper() == symbol_upper:
            return dict(row.get("selected_params") or {}), payload
    focus_symbol = text(payload.get("focus_symbol")).upper()
    if focus_symbol == symbol_upper:
        return dict(payload.get("selected_params") or {}), payload
    raise ValueError(f"base_artifact_missing_symbol:{symbol_upper}")


def summarize_trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    return BASE_MODULE.summarize_trade_metrics(trades, pnl_field="pnl_pct", r_field="r_multiple")


def simulate_symbol_with_exit_risk(frame: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    work = BASE_MODULE.add_breakout_context(frame.reset_index(drop=True).copy(), int(params["breakout_lookback"]))
    entries = BASE_MODULE.signal_mask(work, params)
    stop_buffer_atr = float(params["stop_buffer_atr"])
    target_r = float(params["target_r"])
    max_hold_bars = int(params["max_hold_bars"])
    break_even_trigger_r = float(params.get("break_even_trigger_r", 0.0) or 0.0)
    trailing_stop_atr = float(params.get("trailing_stop_atr", 0.0) or 0.0)
    cooldown_after_losses = int(params.get("cooldown_after_losses", 0) or 0)
    cooldown_bars = int(params.get("cooldown_bars", 0) or 0)

    trades: list[dict[str, Any]] = []
    consecutive_losses = 0
    cooldown_until_idx = -1
    skipped_due_to_cooldown = 0
    idx = 0

    while idx < len(work) - 1:
        if idx <= cooldown_until_idx:
            skipped_due_to_cooldown += 1
            idx += 1
            continue
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

        target_price = entry_price + risk_per_unit * target_r
        highest_high = float(entry_row["high"])
        active_stop = float(stop_price)
        exit_idx = entry_idx
        exit_price = float(entry_row["close"])
        exit_reason = "time_exit"
        break_even_armed = False
        trail_armed = False
        bars_held = 0

        last_idx = min(len(work) - 1, entry_idx + max_hold_bars - 1)
        for probe_idx in range(entry_idx, last_idx + 1):
            probe = work.iloc[probe_idx]
            bars_held = probe_idx - entry_idx + 1
            low = float(probe["low"])
            high = float(probe["high"])
            atr = float(probe["atr_14"]) if pd.notna(probe["atr_14"]) else float(signal_row["atr_14"])
            highest_high = max(highest_high, high)

            if break_even_trigger_r > 0.0 and highest_high >= entry_price + risk_per_unit * break_even_trigger_r:
                active_stop = max(active_stop, entry_price)
                break_even_armed = True

            if trailing_stop_atr > 0.0 and math.isfinite(atr) and atr > 0.0:
                trailed = highest_high - atr * trailing_stop_atr
                if trailed > active_stop:
                    active_stop = trailed
                    trail_armed = True

            if low <= active_stop and high >= target_price:
                exit_idx = probe_idx
                exit_price = active_stop
                exit_reason = "ambiguous_bar_stop_first"
                break
            if low <= active_stop:
                exit_idx = probe_idx
                exit_price = active_stop
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
                "signal_ts_utc": BASE_MODULE.fmt_utc(pd.Timestamp(signal_row["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "entry_ts_utc": BASE_MODULE.fmt_utc(pd.Timestamp(entry_row["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "exit_ts_utc": BASE_MODULE.fmt_utc(pd.Timestamp(work.iloc[exit_idx]["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "risk_per_unit": round(risk_per_unit, 6),
                "stop_price": round(stop_price, 6),
                "active_stop_at_exit": round(float(active_stop), 6),
                "target_price": round(target_price, 6),
                "pnl_pct": round(float(pnl_pct), 6),
                "r_multiple": round(float(r_multiple), 6),
                "bars_held": int(bars_held),
                "exit_reason": exit_reason,
                "break_even_armed": bool(break_even_armed),
                "trail_armed": bool(trail_armed),
                "pullback_depth_atr": round(float(signal_row["breakout_retest_depth_atr"]), 6) if pd.notna(signal_row["breakout_retest_depth_atr"]) else None,
            }
        )

        if r_multiple <= 0.0:
            consecutive_losses += 1
            if cooldown_after_losses > 0 and consecutive_losses >= cooldown_after_losses and cooldown_bars > 0:
                cooldown_until_idx = exit_idx + cooldown_bars
        else:
            consecutive_losses = 0
        idx = exit_idx + 1

    metrics = summarize_trade_metrics(trades)
    metrics["break_even_trigger_r"] = break_even_trigger_r
    metrics["trailing_stop_atr"] = trailing_stop_atr
    metrics["cooldown_after_losses"] = cooldown_after_losses
    metrics["cooldown_bars"] = cooldown_bars
    metrics["skipped_bars_due_to_cooldown"] = int(skipped_due_to_cooldown)
    return {"trades": trades, "metrics": metrics}


def build_candidate(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, base_params: dict[str, Any], exit_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(base_params)
    params.update(exit_params)
    train_gross = simulate_symbol_with_exit_risk(train_frame, params)
    valid_gross = simulate_symbol_with_exit_risk(valid_frame, params)
    train_scenarios = [BASE_MODULE.apply_cost_scenario(train_gross, scenario) for scenario in BASE_MODULE.EXECUTION_COST_SCENARIOS]
    valid_scenarios = [BASE_MODULE.apply_cost_scenario(valid_gross, scenario) for scenario in BASE_MODULE.EXECUTION_COST_SCENARIOS]
    train_selected = next(row for row in train_scenarios if row["scenario_id"] == BASE_MODULE.SELECTION_SCENARIO_ID)
    valid_selected = next(row for row in valid_scenarios if row["scenario_id"] == BASE_MODULE.SELECTION_SCENARIO_ID)
    return {
        "exit_params": exit_params,
        "train_metrics": train_selected["metrics"],
        "validation_metrics": valid_selected["metrics"],
        "train_gross_metrics": train_gross["metrics"],
        "validation_gross_metrics": valid_gross["metrics"],
        "train_scenarios": train_scenarios,
        "validation_scenarios": valid_scenarios,
        "train_objective": float(BASE_MODULE.objective(train_selected["metrics"])),
        "validation_objective": float(BASE_MODULE.objective(valid_selected["metrics"])),
        "validation_status": BASE_MODULE.classify_validation(valid_selected["metrics"]),
        "validation_trade_sample": valid_selected["trade_sample"][:8],
    }


def rank_key(row: dict[str, Any]) -> tuple[float, int, float]:
    metrics = dict(row.get("validation_metrics") or {})
    return (
        float(row.get("validation_objective", 0.0) or 0.0),
        int(metrics.get("trade_count", 0) or 0),
        float(metrics.get("profit_factor", 0.0) or 0.0),
    )


def classify_research_decision(
    *,
    selected_improves: bool,
    validation_leader_improves: bool,
    selection_diverged_from_validation_leader: bool,
) -> str:
    if not validation_leader_improves:
        return "no_exit_risk_improvement_keep_baseline"
    if not selected_improves:
        return "validation_leader_improves_train_first_selected_not_promoted"
    if selection_diverged_from_validation_leader:
        return "selected_exit_risk_improves_but_train_first_validation_diverges"
    return "selected_exit_risk_improves_over_baseline"


def render_markdown(payload: dict[str, Any]) -> str:
    baseline = dict(payload.get("baseline_validation_metrics") or {})
    selected = dict(payload.get("selected_validation_metrics") or {})
    validation_leader = dict(payload.get("validation_leader_metrics") or {})
    lines = [
        "# Price Action Breakout Pullback Exit Risk SIM ONLY",
        "",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- dataset_path: `{text(payload.get('dataset_path'))}`",
        f"- base_artifact_path: `{text(payload.get('base_artifact_path'))}`",
        f"- coverage_start_utc: `{text(payload.get('coverage_start_utc'))}`",
        f"- coverage_end_utc: `{text(payload.get('coverage_end_utc'))}`",
        f"- selection_policy: `{text(payload.get('selection_policy'))}`",
        f"- research_decision: `{text(payload.get('research_decision'))}`",
        f"- baseline_status: `{text(payload.get('baseline_validation_status'))}`",
        f"- selected_status: `{text(payload.get('selected_validation_status'))}`",
        f"- validation_leader_status: `{text(payload.get('validation_leader_status'))}`",
        "",
        "## Baseline vs Selected",
        "",
        f"- baseline_return: `{float(baseline.get('cumulative_return', 0.0)):.2%}`",
        f"- selected_return: `{float(selected.get('cumulative_return', 0.0)):.2%}`",
        f"- baseline_pf: `{float(baseline.get('profit_factor', 0.0)):.2f}`",
        f"- selected_pf: `{float(selected.get('profit_factor', 0.0)):.2f}`",
        f"- baseline_expectancy_r: `{float(baseline.get('expectancy_r', 0.0)):.3f}`",
        f"- selected_expectancy_r: `{float(selected.get('expectancy_r', 0.0)):.3f}`",
        "",
        "## Validation Leader",
        "",
        f"- validation_leader_return: `{float(validation_leader.get('cumulative_return', 0.0)):.2%}`",
        f"- validation_leader_pf: `{float(validation_leader.get('profit_factor', 0.0)):.2f}`",
        f"- validation_leader_expectancy_r: `{float(validation_leader.get('expectancy_r', 0.0)):.3f}`",
        f"- validation_leader_trade_count: `{int(validation_leader.get('trade_count', 0) or 0)}`",
        "",
        "## Selected Exit Params",
        "",
    ]
    for key, value in (payload.get("selected_exit_params") or {}).items():
        lines.append(f"- `{key} = {value}`")
    lines.extend(["", "## Validation Leader Exit Params", ""])
    for key, value in (payload.get("validation_leader_exit_params") or {}).items():
        lines.append(f"- `{key} = {value}`")
    lines.extend(["", "## Candidate Ranking", ""])
    for row in payload.get("ranking", [])[:8]:
        lines.append(
            f"- `hold={row['max_hold_bars']}, be={row['break_even_trigger_r']}, trail={row['trailing_stop_atr']}, cooldown={row['cooldown_after_losses']}x{row['cooldown_bars']}` | "
            f"status=`{row['validation_status']}` | return=`{row['validation_cumulative_return']:.2%}` | pf=`{row['validation_profit_factor']:.2f}` | expectancy_r=`{row['validation_expectancy_r']:.3f}` | trades=`{row['validation_trade_count']}`"
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
    base_artifact_path = Path(args.base_artifact_path).expanduser().resolve() if text(args.base_artifact_path) else select_latest_price_action_artifact(review_dir)
    symbol = text(args.symbol).upper()

    base_entry_params, base_payload = load_base_entry_params(base_artifact_path, symbol)
    frame = BASE_MODULE.add_features(BASE_MODULE.load_frame(dataset_path))
    frame = frame[frame["symbol"] == symbol].copy()
    if frame.empty:
        raise ValueError(f"symbol_not_found_in_dataset:{symbol}")
    train_end, valid_end = BASE_MODULE.split_frame(frame, float(args.train_ratio))
    train_frame = frame[frame["ts"] <= train_end].copy().reset_index(drop=True)
    valid_frame = frame[frame["ts"] > train_end].copy().reset_index(drop=True)

    baseline_exit_params = {
        "max_hold_bars": int(base_entry_params["max_hold_bars"]),
        "break_even_trigger_r": 0.0,
        "trailing_stop_atr": 0.0,
        "cooldown_after_losses": 0,
        "cooldown_bars": 0,
    }
    baseline = build_candidate(train_frame, valid_frame, base_entry_params, baseline_exit_params)
    candidates = [build_candidate(train_frame, valid_frame, base_entry_params, exit_params) for exit_params in EXIT_PARAM_GRID]
    selected = max(candidates, key=lambda row: (float(row["train_objective"]), float(row["validation_objective"]), rank_key(row)))
    validation_leader = max(candidates, key=lambda row: (float(row["validation_objective"]), rank_key(row)))
    ranking = sorted(
        [
            {
                "max_hold_bars": int(row["exit_params"]["max_hold_bars"]),
                "break_even_trigger_r": float(row["exit_params"]["break_even_trigger_r"]),
                "trailing_stop_atr": float(row["exit_params"]["trailing_stop_atr"]),
                "cooldown_after_losses": int(row["exit_params"]["cooldown_after_losses"]),
                "cooldown_bars": int(row["exit_params"]["cooldown_bars"]),
                "validation_status": row["validation_status"],
                "validation_cumulative_return": float(row["validation_metrics"]["cumulative_return"]),
                "validation_profit_factor": float(row["validation_metrics"]["profit_factor"]),
                "validation_expectancy_r": float(row["validation_metrics"]["expectancy_r"]),
                "validation_trade_count": int(row["validation_metrics"]["trade_count"]),
                "validation_objective": float(row["validation_objective"]),
            }
            for row in candidates
        ],
        key=lambda row: (
            BASE_MODULE.symbol_sort_key({"validation_status": row["validation_status"], "validation_metrics": {"cumulative_return": row["validation_cumulative_return"], "trade_count": row["validation_trade_count"], "profit_factor": row["validation_profit_factor"]}}),
            -float(row["validation_objective"]),
        ),
    )

    coverage_start = pd.Timestamp(frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    coverage_end = pd.Timestamp(frame["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    baseline_metrics = dict(baseline.get("validation_metrics") or {})
    selected_metrics = dict(selected.get("validation_metrics") or {})
    validation_leader_metrics = dict(validation_leader.get("validation_metrics") or {})
    selected_improves = float(selected_metrics.get("cumulative_return", 0.0) or 0.0) > float(baseline_metrics.get("cumulative_return", 0.0) or 0.0) + 1e-9
    validation_leader_improves = float(validation_leader_metrics.get("cumulative_return", 0.0) or 0.0) > float(baseline_metrics.get("cumulative_return", 0.0) or 0.0) + 1e-9
    selection_diverged_from_validation_leader = dict(selected.get("exit_params") or {}) != dict(validation_leader.get("exit_params") or {})
    research_decision = classify_research_decision(
        selected_improves=bool(selected_improves),
        validation_leader_improves=bool(validation_leader_improves),
        selection_diverged_from_validation_leader=bool(selection_diverged_from_validation_leader),
    )

    payload = {
        "action": "build_price_action_breakout_pullback_exit_risk_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": BASE_MODULE.fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "base_artifact_path": str(base_artifact_path),
        "symbol": symbol,
        "coverage_start_utc": BASE_MODULE.fmt_utc(coverage_start),
        "coverage_end_utc": BASE_MODULE.fmt_utc(coverage_end),
        "train_end_utc": BASE_MODULE.fmt_utc(train_end.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "validation_end_utc": BASE_MODULE.fmt_utc(valid_end.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "source_catalog": list(BASE_MODULE.SOURCE_CATALOG),
        "base_focus_symbol": text(base_payload.get("focus_symbol")),
        "selection_policy": "train_first_validation_tiebreak",
        "research_decision": research_decision,
        "base_entry_params": base_entry_params,
        "baseline_exit_params": baseline_exit_params,
        "baseline_validation_status": text(baseline.get("validation_status")),
        "baseline_validation_metrics": baseline_metrics,
        "baseline_validation_gross_metrics": dict(baseline.get("validation_gross_metrics") or {}),
        "selected_exit_params": dict(selected.get("exit_params") or {}),
        "selected_validation_status": text(selected.get("validation_status")),
        "selected_validation_metrics": selected_metrics,
        "selected_validation_gross_metrics": dict(selected.get("validation_gross_metrics") or {}),
        "selected_validation_trade_sample": list(selected.get("validation_trade_sample") or []),
        "selected_improves_over_baseline": bool(selected_improves),
        "validation_leader_exit_params": dict(validation_leader.get("exit_params") or {}),
        "validation_leader_status": text(validation_leader.get("validation_status")),
        "validation_leader_metrics": validation_leader_metrics,
        "validation_leader_gross_metrics": dict(validation_leader.get("validation_gross_metrics") or {}),
        "validation_leader_trade_sample": list(validation_leader.get("validation_trade_sample") or []),
        "validation_leader_improves_over_baseline": bool(validation_leader_improves),
        "selection_diverged_from_validation_leader": bool(selection_diverged_from_validation_leader),
        "ranking": ranking,
        "candidate_count": len(candidates),
        "selection_scenario_id": BASE_MODULE.SELECTION_SCENARIO_ID,
        "research_note": "这轮冻结 ETH 15m 的 base entry，只研究 break-even、trailing、time-exit 和 consecutive-loss cooldown 的 exit/risk 影响。",
        "limitation_note": "当前仍是 30 天 15m 公共 OHLCV 近似，不是真实逐笔成交；exit/risk 结果只能用于 SIM_ONLY 缩窄，不可直接放行 live。",
        "recommended_brief": (
            f"{symbol}:exit_risk:{BASE_MODULE.SELECTION_SCENARIO_ID}:selected_return={float(selected_metrics.get('cumulative_return', 0.0)):.2%},"
            f"baseline_return={float(baseline_metrics.get('cumulative_return', 0.0)):.2%},"
            f"status={text(selected.get('validation_status'))},"
            f"improves={str(bool(selected_improves)).lower()},"
            f"decision={research_decision}"
        ),
    }

    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_exit_risk_sim_only.md"
    latest_json_path = review_dir / "latest_price_action_breakout_pullback_exit_risk_sim_only.json"
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(encoded, encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json_path.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "latest_json_path": str(latest_json_path),
                "symbol": symbol,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
