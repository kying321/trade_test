#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"


SOURCE_CATALOG: list[dict[str, str]] = [
    {
        "source_id": "tradingview_volumebreakout",
        "source_type": "open_source_community_script",
        "title": "VOLUMEBREAKOUT",
        "url": "https://tr.tradingview.com/scripts/volumebreakout/",
        "summary": "用 K 线几何或低级别 intrabar 近似买卖量，做 breakout confirmation。",
    },
    {
        "source_id": "tradingview_volume_profile_basics",
        "source_type": "platform_support_doc",
        "title": "Volume profile indicators: basic concepts",
        "url": "https://www.tradingview.com/support/solutions/43000502040-volume-profile-indicators-basic-concepts/",
        "summary": "成交量分布适合做背景过滤，但当前 SIM_ONLY 只做 geometry delta breakout，不重建真实 VPVR。",
    },
]


PARAM_GRID: list[dict[str, Any]] = [
    {
        "breakout_lookback": breakout_lookback,
        "delta_z_threshold": delta_z_threshold,
        "stop_atr": stop_atr,
        "target_r": target_r,
        "max_hold_bars": max_hold_bars,
        "trend_filter": trend_filter,
    }
    for breakout_lookback, delta_z_threshold, stop_atr, target_r, max_hold_bars, trend_filter in itertools.product(
        [8, 12, 20],
        [-0.25, 0.0, 0.5],
        [1.0, 1.5],
        [1.2, 1.8, 2.5],
        [4, 8, 12],
        [False, True],
    )
]


EXECUTION_COST_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario_id": "gross",
        "label": "毛收益",
        "fee_bps_per_side": 0.0,
        "slippage_bps_per_side": 0.0,
    },
    {
        "scenario_id": "moderate_costs",
        "label": "中性成本",
        "fee_bps_per_side": 5.0,
        "slippage_bps_per_side": 3.0,
    },
    {
        "scenario_id": "stress_costs",
        "label": "压力成本",
        "fee_bps_per_side": 10.0,
        "slippage_bps_per_side": 5.0,
    },
]
SELECTION_SCENARIO_ID = "moderate_costs"


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | pd.Timestamp | None) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bars_missing_columns:{','.join(sorted(missing))}")
    work = frame.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["symbol"] = work["symbol"].astype(str).str.upper().str.strip()
    if "asset_class" not in work.columns:
        work["asset_class"] = "crypto"
    work["asset_class"] = work["asset_class"].astype(str).str.lower().str.strip()
    work = work.dropna(subset=["ts", "symbol", "open", "high", "low", "close", "volume"]).copy()
    work = work.sort_values(["symbol", "ts"]).drop_duplicates(subset=["symbol", "ts"]).reset_index(drop=True)
    return work


def select_latest_bars_snapshot(review_dir: Path) -> Path:
    candidates = sorted(review_dir.glob("*crypto_shortline_live_bars_snapshot_bars.csv"))
    if not candidates:
        raise FileNotFoundError("no_crypto_shortline_live_bars_snapshot_bars_csv_found")
    return candidates[-1]


def select_symbol_frame(frame: pd.DataFrame, symbol: str, min_bars: int) -> pd.DataFrame:
    work = frame[frame["symbol"] == symbol].copy()
    if work.empty:
        raise ValueError(f"symbol_not_found:{symbol}")
    if len(work) < min_bars:
        raise ValueError(f"symbol_bars_below_min:{symbol}:{len(work)}<{min_bars}")
    return work.sort_values("ts").reset_index(drop=True)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["ret_1h"] = work["close"].pct_change()
    work["ret_fwd_1h"] = work["close"].shift(-1) / work["close"] - 1.0
    work["ema_20"] = work["close"].ewm(span=20, adjust=False).mean()
    work["ema_50"] = work["close"].ewm(span=50, adjust=False).mean()
    prev_close = work["close"].shift(1)
    work["true_range"] = pd.concat(
        [
            (work["high"] - work["low"]).abs(),
            (work["high"] - prev_close).abs(),
            (work["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    work["atr_14"] = work["true_range"].ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    work["volume_mean_20"] = work["volume"].rolling(20, min_periods=10).mean()
    work["volume_std_20"] = work["volume"].rolling(20, min_periods=10).std(ddof=0)
    work["volume_z_20"] = (
        (work["volume"] - work["volume_mean_20"]) / work["volume_std_20"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    work["range"] = (work["high"] - work["low"]).clip(lower=1e-9)
    work["delta_geometry"] = ((2.0 * work["close"] - work["high"] - work["low"]) / work["range"]) * work["volume"]
    work["delta_geom_mean_20"] = work["delta_geometry"].rolling(20, min_periods=10).mean()
    work["delta_geom_std_20"] = work["delta_geometry"].rolling(20, min_periods=10).std(ddof=0)
    work["delta_geom_z_20"] = (
        (work["delta_geometry"] - work["delta_geom_mean_20"]) / work["delta_geom_std_20"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    for lookback in (8, 12, 20):
        work[f"roll_high_{lookback}_prev"] = work["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    return work


def split_frame(frame: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) < 80:
        raise ValueError("intraday_dataset_too_small_for_split")
    split_idx = max(48, min(len(frame) - 24, int(math.floor(len(frame) * train_ratio))))
    train = frame.iloc[:split_idx].copy()
    valid = frame.iloc[split_idx:].copy()
    if train.empty or valid.empty:
        raise ValueError("empty_train_or_valid_intraday_frame")
    return train, valid


def signal_mask(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    breakout_lb = int(params["breakout_lookback"])
    delta_z_threshold = float(params["delta_z_threshold"])
    trend_filter = bool(params["trend_filter"])
    breakout = frame["close"] > frame[f"roll_high_{breakout_lb}_prev"]
    delta_ok = frame["delta_geom_z_20"] >= delta_z_threshold
    volume_ok = frame["volume_z_20"] >= 0.0
    trend_ok = pd.Series(True, index=frame.index)
    if trend_filter:
        trend_ok = (frame["close"] > frame["ema_20"]) & (frame["ema_20"] > frame["ema_50"])
    ready = frame["atr_14"].notna() & frame[f"roll_high_{breakout_lb}_prev"].notna()
    return (breakout & delta_ok & volume_ok & trend_ok & ready).fillna(False)


def summarize_trade_metrics(
    trades: list[dict[str, Any]],
    *,
    pnl_field: str,
    r_field: str,
) -> dict[str, Any]:
    returns = pd.Series([float(row[pnl_field]) for row in trades], dtype=float)
    r_series = pd.Series([float(row[r_field]) for row in trades], dtype=float)
    if returns.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "cumulative_return": 0.0,
            "avg_trade_return": 0.0,
            "avg_r_multiple": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_per_trade": 0.0,
            "avg_hold_bars": 0.0,
        }
    equity_series = (1.0 + returns).cumprod()
    drawdown = (equity_series / equity_series.cummax()) - 1.0
    gross_profit = float(r_series[r_series > 0.0].sum())
    gross_loss = float(-r_series[r_series < 0.0].sum())
    std_ret = float(returns.std(ddof=0))
    sharpe = float((returns.mean() / std_ret) * math.sqrt(len(returns))) if std_ret > 0.0 else 0.0
    return {
        "trade_count": int(len(trades)),
        "win_rate": float((r_series > 0.0).mean()),
        "cumulative_return": float(equity_series.iloc[-1] - 1.0),
        "avg_trade_return": float(returns.mean()),
        "avg_r_multiple": float(r_series.mean()),
        "expectancy_r": float(r_series.mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0),
        "max_drawdown": float(abs(drawdown.min())),
        "sharpe_per_trade": sharpe,
        "avg_hold_bars": float(pd.Series([int(row["bars_held"]) for row in trades], dtype=float).mean()),
    }


def simulate_strategy(frame: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    work = frame.reset_index(drop=True).copy()
    entries = signal_mask(work, params)
    stop_atr = float(params["stop_atr"])
    target_r = float(params["target_r"])
    max_hold_bars = int(params["max_hold_bars"])

    trades: list[dict[str, Any]] = []
    equity = 1.0
    equity_curve: list[dict[str, Any]] = []
    idx = 0
    while idx < len(work) - 1:
        if not bool(entries.iloc[idx]):
            idx += 1
            continue
        signal_row = work.iloc[idx]
        entry_idx = idx + 1
        entry_row = work.iloc[entry_idx]
        risk_per_unit = float(signal_row["atr_14"]) * stop_atr
        if not math.isfinite(risk_per_unit) or risk_per_unit <= 0.0:
            idx += 1
            continue
        entry_price = float(entry_row["open"])
        stop_price = entry_price - risk_per_unit
        target_price = entry_price + target_r * risk_per_unit
        exit_idx = entry_idx
        exit_price = float(entry_row["close"])
        exit_reason = "time_exit"
        bars_held = 0
        last_idx = min(len(work) - 1, entry_idx + max_hold_bars - 1)
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
        equity *= 1.0 + pnl_pct
        trade = {
            "signal_ts_utc": fmt_utc(pd.Timestamp(signal_row["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
            "entry_ts_utc": fmt_utc(pd.Timestamp(entry_row["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
            "exit_ts_utc": fmt_utc(pd.Timestamp(work.iloc[exit_idx]["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
            "entry_price": round(entry_price, 6),
            "exit_price": round(exit_price, 6),
            "risk_per_unit": round(risk_per_unit, 6),
            "stop_price": round(stop_price, 6),
            "target_price": round(target_price, 6),
            "pnl_pct": round(float(pnl_pct), 6),
            "r_multiple": round(float(r_multiple), 6),
            "bars_held": int(bars_held),
            "exit_reason": exit_reason,
            "delta_geom_z_20": round(float(signal_row["delta_geom_z_20"]), 6),
            "volume_z_20": round(float(signal_row["volume_z_20"]), 6),
            "breakout_ref": round(float(signal_row[f"roll_high_{int(params['breakout_lookback'])}_prev"]), 6),
        }
        trades.append(trade)
        equity_curve.append(
            {
                "ts_utc": trade["exit_ts_utc"],
                "equity": round(float(equity), 6),
                "trade_r": round(float(r_multiple), 6),
                "trade_pnl_pct": round(float(pnl_pct), 6),
            }
        )
        idx = exit_idx + 1

    metrics = summarize_trade_metrics(trades, pnl_field="pnl_pct", r_field="r_multiple")
    return {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}


def objective(metrics: dict[str, Any]) -> float:
    trade_count = int(metrics.get("trade_count", 0) or 0)
    if trade_count <= 0:
        return -999.0
    return float(
        float(metrics.get("cumulative_return", 0.0)) * 120.0
        + float(metrics.get("sharpe_per_trade", 0.0)) * 8.0
        + min(float(metrics.get("profit_factor", 0.0)), 5.0) * 4.0
        + float(metrics.get("expectancy_r", 0.0)) * 12.0
        - float(metrics.get("max_drawdown", 0.0)) * 120.0
        + min(trade_count, 12) * 0.4
    )


def classify_validation(metrics: dict[str, Any]) -> str:
    trade_count = int(metrics.get("trade_count", 0) or 0)
    cumulative_return = float(metrics.get("cumulative_return", 0.0) or 0.0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    expectancy_r = float(metrics.get("expectancy_r", 0.0) or 0.0)
    if trade_count <= 0:
        return "inactive_oos"
    if trade_count >= 3 and cumulative_return > 0.02 and profit_factor > 1.15 and max_drawdown < 0.08 and expectancy_r > 0.0:
        return "promising_small_sample"
    if cumulative_return > 0.0 and profit_factor > 1.0:
        return "mixed_positive"
    return "not_promising"


def apply_execution_cost_scenario(base_result: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    fee_bps = float(scenario["fee_bps_per_side"])
    slip_bps = float(scenario["slippage_bps_per_side"])
    entry_mult = (1.0 + slip_bps / 10000.0) * (1.0 + fee_bps / 10000.0)
    exit_mult = (1.0 - slip_bps / 10000.0) * (1.0 - fee_bps / 10000.0)
    stressed_trades: list[dict[str, Any]] = []
    equity = 1.0
    equity_curve: list[dict[str, Any]] = []
    for trade in base_result["trades"]:
        entry_price = float(trade["entry_price"])
        exit_price = float(trade["exit_price"])
        risk_per_unit = float(trade["risk_per_unit"])
        entry_fill = entry_price * entry_mult
        exit_fill = exit_price * exit_mult
        net_pnl_pct = (exit_fill / entry_fill) - 1.0
        net_r_multiple = (exit_fill - entry_fill) / risk_per_unit if risk_per_unit > 0.0 else 0.0
        equity *= 1.0 + net_pnl_pct
        row = dict(trade)
        row["scenario_id"] = str(scenario["scenario_id"])
        row["fee_bps_per_side"] = fee_bps
        row["slippage_bps_per_side"] = slip_bps
        row["net_entry_fill"] = round(entry_fill, 6)
        row["net_exit_fill"] = round(exit_fill, 6)
        row["net_pnl_pct"] = round(float(net_pnl_pct), 6)
        row["net_r_multiple"] = round(float(net_r_multiple), 6)
        stressed_trades.append(row)
        equity_curve.append(
            {
                "ts_utc": row["exit_ts_utc"],
                "scenario_id": str(scenario["scenario_id"]),
                "equity": round(float(equity), 6),
                "trade_r": round(float(net_r_multiple), 6),
                "trade_pnl_pct": round(float(net_pnl_pct), 6),
            }
        )
    metrics = summarize_trade_metrics(stressed_trades, pnl_field="net_pnl_pct", r_field="net_r_multiple")
    return {
        "scenario_id": str(scenario["scenario_id"]),
        "label": str(scenario["label"]),
        "fee_bps_per_side": fee_bps,
        "slippage_bps_per_side": slip_bps,
        "metrics": metrics,
        "trade_sample": stressed_trades[:6],
        "equity_curve": equity_curve[:12],
    }


def param_sort_key(row: dict[str, Any]) -> tuple[float, int, float, int, float]:
    valid = dict(row.get("validation_metrics") or {})
    trade_count = int(valid.get("trade_count", 0) or 0)
    robust_trade_flag = 1 if trade_count >= 2 else 0
    return (
        float(row["train_objective"]),
        robust_trade_flag,
        float(row["validation_objective"]),
        trade_count,
        -float(valid.get("max_drawdown", 0.0) or 0.0),
    )


def build_param_result(train_frame: pd.DataFrame, valid_frame: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    train_gross = simulate_strategy(train_frame, params)
    valid_gross = simulate_strategy(valid_frame, params)
    train_scenarios = [apply_execution_cost_scenario(train_gross, scenario) for scenario in EXECUTION_COST_SCENARIOS]
    valid_scenarios = [apply_execution_cost_scenario(valid_gross, scenario) for scenario in EXECUTION_COST_SCENARIOS]
    train_selected = next(row for row in train_scenarios if row["scenario_id"] == SELECTION_SCENARIO_ID)
    valid_selected = next(row for row in valid_scenarios if row["scenario_id"] == SELECTION_SCENARIO_ID)
    return {
        "params": params,
        "train_metrics": train_selected["metrics"],
        "validation_metrics": valid_selected["metrics"],
        "train_gross_metrics": train_gross["metrics"],
        "validation_gross_metrics": valid_gross["metrics"],
        "train_objective": float(objective(train_selected["metrics"])),
        "validation_objective": float(objective(valid_selected["metrics"])),
        "validation_status": classify_validation(valid_selected["metrics"]),
        "validation_trade_sample": valid_selected["trade_sample"][:6],
        "validation_equity_sample": valid_selected["equity_curve"][:6],
        "train_scenarios": train_scenarios,
        "validation_scenarios": valid_scenarios,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Geometry Delta Breakout SIM ONLY",
        "",
        f"- bars_path: `{text(payload.get('bars_path'))}`",
        f"- symbol: `{text(payload.get('symbol'))}`",
        f"- sim_scope: `{text(payload.get('sim_scope'))}`",
        f"- cadence_hours: `{payload.get('cadence_hours')}`",
        f"- coverage: `{text(payload.get('coverage_start_utc'))} -> {text(payload.get('coverage_end_utc'))}`",
        f"- train_end_utc: `{text(payload.get('train_end_utc'))}`",
        f"- intraday_basis_status: `{text(payload.get('intraday_basis_status'))}`",
        f"- selection_scenario: `{text(payload.get('selection_scenario_id'))}`",
        f"- selected_params: `{json.dumps(payload.get('selected_params', {}), ensure_ascii=False)}`",
        f"- validation_status: `{text(payload.get('validation_status'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Validation Metrics",
        "",
    ]
    valid = dict(payload.get("validation_metrics") or {})
    for key in (
        "trade_count",
        "win_rate",
        "cumulative_return",
        "avg_trade_return",
        "avg_r_multiple",
        "expectancy_r",
        "profit_factor",
        "max_drawdown",
        "sharpe_per_trade",
        "avg_hold_bars",
    ):
        value = valid.get(key)
        if isinstance(value, float):
            if "rate" in key or "return" in key or key == "max_drawdown":
                lines.append(f"- {key}: `{value:.4%}`")
            else:
                lines.append(f"- {key}: `{value:.4f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Gross vs Cost Scenarios", ""])
    gross = dict(payload.get("validation_gross_metrics") or {})
    lines.append(
        f"- gross: `return={float(gross.get('cumulative_return', 0.0)):.4%}, pf={float(gross.get('profit_factor', 0.0)):.2f}, expectancy_r={float(gross.get('expectancy_r', 0.0)):.3f}, trades={int(gross.get('trade_count', 0) or 0)}`"
    )
    for row in payload.get("validation_scenarios", []):
        if text(row.get("scenario_id")) == "gross":
            continue
        metrics = dict(row.get("metrics") or {})
        lines.append(
            f"- {row['scenario_id']}: `fee={float(row['fee_bps_per_side']):.1f}bps/side, slip={float(row['slippage_bps_per_side']):.1f}bps/side, return={float(metrics.get('cumulative_return', 0.0)):.4%}, pf={float(metrics.get('profit_factor', 0.0)):.2f}, expectancy_r={float(metrics.get('expectancy_r', 0.0)):.3f}`"
        )
    lines.extend(["", "## Trade Sample", ""])
    for row in payload.get("validation_trade_sample", []):
        lines.append(
            f"- `{row['entry_ts_utc']}` -> `{row['exit_ts_utc']}` | exit=`{row['exit_reason']}` | net_pnl=`{row['net_pnl_pct']:.4%}` | net_r=`{row['net_r_multiple']:.3f}`"
        )
    lines.extend(["", "## Source Catalog", ""])
    for row in payload.get("source_catalog", []):
        lines.append(f"- `{row['source_id']}` | [{row['title']}]({row['url']}) | {row['summary']}")
    lines.extend(["", "## Notes", ""])
    lines.append(f"- `{text(payload.get('research_note'))}`")
    lines.append(f"- `{text(payload.get('limitation_note'))}`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SIM_ONLY intraday geometry-delta-breakout validation on local shortline bars.")
    parser.add_argument("--bars-path", default="")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--symbol", default="SOLUSDT")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--min-bars", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    bars_path = Path(args.bars_path).expanduser().resolve() if text(args.bars_path) else select_latest_bars_snapshot(review_dir)
    stamp_dt = parse_stamp(args.stamp)

    full_frame = load_frame(bars_path)
    symbol = text(args.symbol).upper()
    symbol_frame = select_symbol_frame(full_frame, symbol, int(args.min_bars))
    featured = add_features(symbol_frame)
    train_frame, valid_frame = split_frame(featured, float(args.train_ratio))

    param_results = [build_param_result(train_frame, valid_frame, params) for params in PARAM_GRID]
    ranked = sorted(param_results, key=param_sort_key, reverse=True)
    selected = ranked[0]
    selected_params = dict(selected["params"])
    final_train_gross = simulate_strategy(train_frame, selected_params)
    final_valid_gross = simulate_strategy(valid_frame, selected_params)
    final_train_scenarios = [apply_execution_cost_scenario(final_train_gross, scenario) for scenario in EXECUTION_COST_SCENARIOS]
    final_valid_scenarios = [apply_execution_cost_scenario(final_valid_gross, scenario) for scenario in EXECUTION_COST_SCENARIOS]
    final_train = next(row for row in final_train_scenarios if row["scenario_id"] == SELECTION_SCENARIO_ID)
    final_valid = next(row for row in final_valid_scenarios if row["scenario_id"] == SELECTION_SCENARIO_ID)
    symbol_ts = pd.to_datetime(symbol_frame["ts"], utc=True)
    symbol_delta = symbol_ts.sort_values().diff().dropna()
    cadence = (
        symbol_delta.mode().iloc[0] / pd.Timedelta(hours=1)
        if not symbol_delta.empty
        else None
    )

    cadence_minutes = int(round(float(cadence) * 60.0)) if cadence is not None else None
    sim_scope = f"single_symbol_intraday_{cadence_minutes}m" if cadence_minutes else "single_symbol_intraday"
    payload = {
        "action": "build_geometry_delta_breakout_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "strategy_family": "geometry_delta_breakout",
        "strategy_label": "几何 Delta 突破确认",
        "category": "orderflow_proxy",
        "sim_scope": sim_scope,
        "bars_path": str(bars_path),
        "symbol": symbol,
        "input_symbol_count": int(full_frame["symbol"].nunique()),
        "input_symbols": sorted(full_frame["symbol"].unique().tolist()),
        "bar_rows": int(len(symbol_frame)),
        "coverage_start_utc": fmt_utc(pd.Timestamp(symbol_frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "coverage_end_utc": fmt_utc(pd.Timestamp(symbol_frame["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "train_end_utc": fmt_utc(pd.Timestamp(train_frame["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "validation_start_utc": fmt_utc(pd.Timestamp(valid_frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "cadence_hours": float(cadence) if cadence is not None else None,
        "cadence_minutes": cadence_minutes,
        "intraday_basis_status": "single_symbol_only" if int(full_frame["symbol"].nunique()) == 1 else "multi_symbol_available",
        "cross_section_readiness": "insufficient_multisymbol_intraday_basis" if int(full_frame["symbol"].nunique()) < 4 else "ready_for_intraday_cross_section",
        "source_catalog": SOURCE_CATALOG,
        "param_grid_size": int(len(PARAM_GRID)),
        "selection_scenario_id": SELECTION_SCENARIO_ID,
        "execution_cost_scenarios": EXECUTION_COST_SCENARIOS,
        "selected_params": selected_params,
        "train_metrics": final_train["metrics"],
        "validation_metrics": final_valid["metrics"],
        "train_gross_metrics": final_train_gross["metrics"],
        "validation_gross_metrics": final_valid_gross["metrics"],
        "validation_scenarios": final_valid_scenarios,
        "validation_status": classify_validation(final_valid["metrics"]),
        "train_objective": float(objective(final_train["metrics"])),
        "validation_objective": float(objective(final_valid["metrics"])),
        "validation_trade_sample": final_valid["trade_sample"][:10],
        "validation_equity_sample": final_valid["equity_curve"][:10],
        "param_results": [
            {
                "params": row["params"],
                "train_objective": round(float(row["train_objective"]), 6),
                "validation_objective": round(float(row["validation_objective"]), 6),
                "validation_status": row["validation_status"],
                "validation_trade_count": int(row["validation_metrics"]["trade_count"]),
                "validation_cumulative_return": round(float(row["validation_metrics"]["cumulative_return"]), 6),
                "validation_profit_factor": round(float(row["validation_metrics"]["profit_factor"]), 6),
                "validation_expectancy_r": round(float(row["validation_metrics"]["expectancy_r"]), 6),
                "validation_max_drawdown": round(float(row["validation_metrics"]["max_drawdown"]), 6),
                "validation_gross_cumulative_return": round(float(row["validation_gross_metrics"]["cumulative_return"]), 6),
            }
            for row in ranked[:12]
        ],
        "recommended_brief": (
            f"{symbol}:{SELECTION_SCENARIO_ID}:status={classify_validation(final_valid['metrics'])},"
            f"valid_return={float(final_valid['metrics']['cumulative_return']):.2%},"
            f"gross_return={float(final_valid_gross['metrics']['cumulative_return']):.2%},"
            f"pf={float(final_valid['metrics']['profit_factor']):.2f},"
            f"expectancy_r={float(final_valid['metrics']['expectancy_r']):.3f},"
            f"mdd={float(final_valid['metrics']['max_drawdown']):.2%},"
            f"trades={int(final_valid['metrics']['trade_count'])}"
        ),
        "research_note": (
            "这轮不是新的截面研究，而是把公开网页里的 geometry-delta-breakout 下沉成 "
            f"{cadence_minutes or '?'}m 单 symbol SIM_ONLY 验证。"
        ),
        "limitation_note": (
            f"当前 bars 基座覆盖 {int(full_frame['symbol'].nunique())} 个 symbol，目标 symbol "
            f"{symbol} 有 {int(len(symbol_frame))} 根 {cadence_minutes or '?'}m 数据；"
            "但仍不含真实逐笔 orderflow，本轮虽然补了 fee/slippage 场景，结果仍只能用于参数收缩和失败归因，不能直接放行 live。"
        ),
        "execution_assumptions": {
            "entry_rule": "signal_bar_close_confirmed_then_next_bar_open_entry",
            "exit_rule": "intrabar stop/target check, ambiguous same-bar hit uses stop-first",
            "positioning": "single_position_long_only",
        },
    }

    json_path = review_dir / f"{args.stamp}_geometry_delta_breakout_sim_only.json"
    md_path = review_dir / f"{args.stamp}_geometry_delta_breakout_sim_only.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "validation_status": payload["validation_status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
