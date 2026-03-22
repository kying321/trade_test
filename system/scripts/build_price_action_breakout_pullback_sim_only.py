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
VPVR_PROXY_LOOKBACK_BARS = 96
VPVR_PROXY_BINS = 24


SOURCE_CATALOG: list[dict[str, str]] = [
    {
        "source_id": "priceaction_breakout_strategies",
        "source_type": "public_price_action_article",
        "title": "Price Action Breakout Strategies",
        "url": "https://priceaction.com/price-action-university/beginners/price-action-breakout-strategies/",
        "summary": "关注关键位突破后的延续，而不是在噪声区间里追单。",
    },
    {
        "source_id": "tradeciety_price_action",
        "source_type": "public_price_action_article",
        "title": "Price Action Hub",
        "url": "https://tradeciety.com/price-action",
        "summary": "pullback continuation 是可重复的趋势跟随 setup，前提是背景和触发分开处理。",
    },
    {
        "source_id": "tradezella_break_retest",
        "source_type": "community_strategy_page",
        "title": "Break & Retest Playbook",
        "url": "https://www.tradezella.com/strategies/break-retest-playbook",
        "summary": "突破后第一次受控回踩是更结构化的入场，而不是在 breakout bar 本身追价。",
    },
    {
        "source_id": "tradingview_volume_profile_basics",
        "source_type": "public_indicator_doc",
        "title": "Volume Profile Indicators: Basic Concepts",
        "url": "https://www.tradingview.com/support/solutions/43000502040-volume-profile-indicators-basic-concepts/",
        "summary": "把高成交密集区当作结构背景，用于过滤假突破和过度延伸。",
    },
]


PARAM_GRID: list[dict[str, Any]] = [
    {
        "breakout_lookback": breakout_lookback,
        "breakout_memory_bars": breakout_memory_bars,
        "pullback_max_atr": pullback_max_atr,
        "stop_buffer_atr": stop_buffer_atr,
        "target_r": target_r,
        "max_hold_bars": max_hold_bars,
        "trend_filter": trend_filter,
        "structure_filter": structure_filter,
        "vpvr_filter": vpvr_filter,
        "vpvr_max_distance_atr": vpvr_max_distance_atr,
        "daily_stop_r": daily_stop_r,
    }
    for breakout_lookback, breakout_memory_bars, pullback_max_atr, stop_buffer_atr, target_r, max_hold_bars, trend_filter, structure_filter, vpvr_filter, vpvr_max_distance_atr, daily_stop_r in itertools.product(
        [20, 40],
        [8, 16],
        [0.8, 1.2, 1.6],
        [0.1, 0.3],
        [1.5, 2.0, 2.5],
        [8, 12, 16],
        [False, True],
        ["none", "reclaim"],
        [False],
        [1.0],
        [0.0, 1.5],
    )
]


EXECUTION_COST_SCENARIOS: list[dict[str, Any]] = [
    {"scenario_id": "gross", "label": "毛收益", "fee_bps_per_side": 0.0, "slippage_bps_per_side": 0.0},
    {"scenario_id": "moderate_costs", "label": "中性成本", "fee_bps_per_side": 5.0, "slippage_bps_per_side": 3.0},
    {"scenario_id": "stress_costs", "label": "压力成本", "fee_bps_per_side": 10.0, "slippage_bps_per_side": 5.0},
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
        raise ValueError(f"dataset_missing_columns:{','.join(sorted(missing))}")
    work = frame.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce").dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["symbol"] = work["symbol"].astype(str).str.upper().str.strip()
    work["asset_class"] = work.get("asset_class", "crypto")
    work["asset_class"] = work["asset_class"].astype(str).str.lower().str.strip()
    work = work.dropna(subset=["ts", "symbol", "open", "high", "low", "close", "volume"]).copy()
    work = work.sort_values(["symbol", "ts"]).drop_duplicates(subset=["symbol", "ts"]).reset_index(drop=True)
    return work


def select_latest_intraday_dataset(review_dir: Path) -> Path:
    candidates = sorted(review_dir.glob("*_public_intraday_crypto_bars_dataset.csv"))
    if not candidates:
        raise FileNotFoundError("no_public_intraday_crypto_bars_dataset_found")
    return candidates[-1]


def compute_vpvr_proxy_poc(
    *,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    lookback_bars: int,
    bins: int,
) -> np.ndarray:
    poc = np.full(len(closes), np.nan, dtype=float)
    for end_idx in range(max(lookback_bars, 2), len(closes)):
        start_idx = end_idx - lookback_bars
        window_high = float(np.nanmax(highs[start_idx:end_idx]))
        window_low = float(np.nanmin(lows[start_idx:end_idx]))
        if not math.isfinite(window_high) or not math.isfinite(window_low) or window_high <= window_low:
            continue
        price_span = window_high - window_low
        window_closes = closes[start_idx:end_idx]
        window_volumes = volumes[start_idx:end_idx]
        scaled = np.floor(((window_closes - window_low) / price_span) * bins).astype(int)
        scaled = np.clip(scaled, 0, bins - 1)
        hist = np.bincount(scaled, weights=window_volumes, minlength=bins)
        if hist.size == 0 or float(hist.max()) <= 0.0:
            continue
        poc_idx = int(hist.argmax())
        bin_edges = np.linspace(window_low, window_high, bins + 1)
        poc[end_idx] = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)
    return poc


def compute_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    work = group.sort_values("ts").reset_index(drop=True).copy()
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
    work["ret_fwd_1bar"] = work["close"].shift(-1) / work["close"] - 1.0
    for lookback in (20, 40):
        work[f"roll_high_{lookback}_prev"] = work["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    work["swing_low_8_prev"] = work["low"].rolling(8, min_periods=4).min().shift(1)
    work["bull_trigger_bar"] = (work["close"] > work["open"]) & (work["close"] > work["high"].shift(1))
    work["high_lookback_16_prev"] = work["high"].rolling(16, min_periods=8).max().shift(1)
    work["pullback_depth_atr"] = (work["high_lookback_16_prev"] - work["close"]) / work["atr_14"].replace(0.0, np.nan)
    work["vpvr_proxy_poc_prev"] = compute_vpvr_proxy_poc(
        highs=work["high"].to_numpy(dtype=float),
        lows=work["low"].to_numpy(dtype=float),
        closes=work["close"].to_numpy(dtype=float),
        volumes=work["volume"].to_numpy(dtype=float),
        lookback_bars=VPVR_PROXY_LOOKBACK_BARS,
        bins=VPVR_PROXY_BINS,
    )
    work["vpvr_proxy_distance_atr"] = (work["close"] - work["vpvr_proxy_poc_prev"]) / work["atr_14"].replace(0.0, np.nan)
    return work


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    groups = [compute_symbol_features(group) for _, group in frame.groupby("symbol", sort=True)]
    return pd.concat(groups, ignore_index=True) if groups else frame.copy()


def split_frame(frame: pd.DataFrame, train_ratio: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.Series(sorted(frame["ts"].dropna().unique())).astype("datetime64[ns]")
    if len(unique_dates) < 160:
        raise ValueError("dataset_too_small_for_price_action_intraday_split")
    split_idx = max(96, min(len(unique_dates) - 48, int(math.floor(len(unique_dates) * train_ratio))))
    return pd.Timestamp(unique_dates.iloc[split_idx - 1]), pd.Timestamp(unique_dates.iloc[-1])


def add_breakout_context(frame: pd.DataFrame, breakout_lookback: int) -> pd.DataFrame:
    work = frame.copy()
    breakout_ref_col = f"roll_high_{breakout_lookback}_prev"
    breakout_flag = (work["close"] > work[breakout_ref_col]) & work[breakout_ref_col].notna()
    breakout_index = pd.Series(np.where(breakout_flag, np.arange(len(work), dtype=float), np.nan), index=work.index)
    breakout_ref = pd.Series(np.where(breakout_flag, work[breakout_ref_col], np.nan), index=work.index, dtype=float)
    work["last_breakout_index"] = breakout_index.ffill()
    work["bars_since_breakout"] = np.arange(len(work), dtype=float) - work["last_breakout_index"]
    work["breakout_ref"] = breakout_ref.ffill()
    atr = work["atr_14"].replace(0.0, np.nan)
    work["breakout_retest_depth_atr"] = (work["breakout_ref"] - work["close"]) / atr
    return work


def signal_mask(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    breakout_lookback = int(params["breakout_lookback"])
    breakout_memory_bars = int(params["breakout_memory_bars"])
    pullback_max_atr = float(params["pullback_max_atr"])
    trend_filter = bool(params["trend_filter"])
    structure_filter = text(params.get("structure_filter")) or "none"
    vpvr_filter = bool(params["vpvr_filter"])
    vpvr_max_distance_atr = float(params["vpvr_max_distance_atr"])
    work = add_breakout_context(frame, breakout_lookback)
    atr = work["atr_14"].replace(0.0, np.nan)
    bars_since_breakout = work["bars_since_breakout"]
    breakout_context = bars_since_breakout.between(1.0, float(breakout_memory_bars), inclusive="both")
    pullback_ok = (
        work["breakout_retest_depth_atr"].between(-0.2, pullback_max_atr, inclusive="both")
        & (work["low"] <= (work["breakout_ref"] + 0.2 * atr))
        & (work["low"] >= (work["breakout_ref"] - pullback_max_atr * atr))
        & (work["close"] >= (work["breakout_ref"] - 0.05 * atr))
    )
    trend_ok = pd.Series(True, index=work.index)
    if trend_filter:
        trend_ok = (work["close"] > work["ema_20"]) & (work["ema_20"] > work["ema_50"])
    structure_ok = pd.Series(True, index=work.index)
    if structure_filter == "reclaim":
        structure_ok = (
            (work["low"].shift(1) <= (work["breakout_ref"].shift(1) + 0.1 * atr.shift(1)))
            & (work["close"] >= work["breakout_ref"])
        )
    vpvr_ok = pd.Series(True, index=work.index)
    if vpvr_filter:
        vpvr_ok = (
            work["vpvr_proxy_poc_prev"].notna()
            & (work["breakout_ref"] >= work["vpvr_proxy_poc_prev"])
            & work["vpvr_proxy_distance_atr"].between(-0.2, vpvr_max_distance_atr, inclusive="both")
        )
    return (
        breakout_context
        & pullback_ok
        & work["bull_trigger_bar"]
        & trend_ok
        & structure_ok
        & vpvr_ok
        & work["atr_14"].notna()
    ).fillna(False)


def summarize_trade_metrics(trades: list[dict[str, Any]], *, pnl_field: str, r_field: str) -> dict[str, Any]:
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
    equity = (1.0 + returns).cumprod()
    drawdown = (equity / equity.cummax()) - 1.0
    gross_profit = float(r_series[r_series > 0.0].sum())
    gross_loss = float(-r_series[r_series < 0.0].sum())
    std_ret = float(returns.std(ddof=0))
    sharpe = float((returns.mean() / std_ret) * math.sqrt(len(returns))) if std_ret > 0.0 else 0.0
    return {
        "trade_count": int(len(trades)),
        "win_rate": float((r_series > 0.0).mean()),
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "avg_trade_return": float(returns.mean()),
        "avg_r_multiple": float(r_series.mean()),
        "expectancy_r": float(r_series.mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0),
        "max_drawdown": float(abs(drawdown.min())),
        "sharpe_per_trade": sharpe,
        "avg_hold_bars": float(pd.Series([int(row["bars_held"]) for row in trades], dtype=float).mean()),
    }


def simulate_symbol(frame: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    work = add_breakout_context(frame.reset_index(drop=True).copy(), int(params["breakout_lookback"]))
    entries = signal_mask(work, params)
    stop_buffer_atr = float(params["stop_buffer_atr"])
    target_r = float(params["target_r"])
    max_hold_bars = int(params["max_hold_bars"])
    daily_stop_r = float(params.get("daily_stop_r", 0.0) or 0.0)

    trades: list[dict[str, Any]] = []
    realized_r_by_day: dict[str, float] = {}
    skipped_due_to_daily_stop = 0
    idx = 0
    while idx < len(work) - 1:
        if not bool(entries.iloc[idx]):
            idx += 1
            continue
        signal_row = work.iloc[idx]
        entry_idx = idx + 1
        entry_row = work.iloc[entry_idx]
        entry_day = pd.Timestamp(entry_row["ts"]).strftime("%Y-%m-%d")
        if daily_stop_r > 0.0 and float(realized_r_by_day.get(entry_day, 0.0)) <= -daily_stop_r:
            skipped_due_to_daily_stop += 1
            idx += 1
            continue
        swing_low = float(signal_row["swing_low_8_prev"]) if pd.notna(signal_row["swing_low_8_prev"]) else float(signal_row["low"])
        stop_price = min(swing_low, float(signal_row["ema_50"])) - float(signal_row["atr_14"]) * stop_buffer_atr
        entry_price = float(entry_row["open"])
        risk_per_unit = entry_price - stop_price
        if not math.isfinite(risk_per_unit) or risk_per_unit <= 0.0:
            idx += 1
            continue
        target_price = entry_price + risk_per_unit * target_r
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
        exit_day = pd.Timestamp(work.iloc[exit_idx]["ts"]).strftime("%Y-%m-%d")
        realized_r_by_day[exit_day] = float(realized_r_by_day.get(exit_day, 0.0)) + float(r_multiple)
        trades.append(
            {
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
                "pullback_depth_atr": round(float(signal_row["breakout_retest_depth_atr"]), 6) if pd.notna(signal_row["breakout_retest_depth_atr"]) else None,
            }
        )
        idx = exit_idx + 1
    metrics = summarize_trade_metrics(trades, pnl_field="pnl_pct", r_field="r_multiple")
    metrics["skipped_signals_due_to_daily_stop"] = int(skipped_due_to_daily_stop)
    metrics["daily_stop_trigger_days"] = int(
        sum(1 for value in realized_r_by_day.values() if daily_stop_r > 0.0 and float(value) <= -daily_stop_r)
    )
    return {"trades": trades, "metrics": metrics}


def apply_cost_scenario(base_result: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
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
    metrics = summarize_trade_metrics(stressed_trades, pnl_field="net_pnl_pct", r_field="net_r_multiple")
    return {
        "scenario_id": str(scenario["scenario_id"]),
        "label": str(scenario["label"]),
        "fee_bps_per_side": fee_bps,
        "slippage_bps_per_side": slip_bps,
        "metrics": metrics,
        "trade_sample": stressed_trades[:8],
    }


def objective(metrics: dict[str, Any]) -> float:
    trade_count = int(metrics.get("trade_count", 0) or 0)
    if trade_count <= 0:
        return -999.0
    return float(
        float(metrics.get("cumulative_return", 0.0)) * 120.0
        + float(metrics.get("sharpe_per_trade", 0.0)) * 8.0
        + min(float(metrics.get("profit_factor", 0.0)), 5.0) * 4.0
        + float(metrics.get("expectancy_r", 0.0)) * 10.0
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
    if trade_count >= 4 and cumulative_return > 0.02 and profit_factor > 1.15 and max_drawdown < 0.08 and expectancy_r > 0.0:
        return "promising_small_sample"
    if cumulative_return > 0.0 and profit_factor > 1.0:
        return "mixed_positive"
    return "not_promising"


def symbol_sort_key(row: dict[str, Any]) -> tuple[int, float, int, float]:
    metrics = dict(row.get("validation_metrics") or {})
    status_rank = {
        "promising_small_sample": 0,
        "mixed_positive": 1,
        "not_promising": 2,
        "inactive_oos": 3,
    }
    return (
        status_rank.get(text(row.get("validation_status")), 9),
        -float(metrics.get("cumulative_return", 0.0) or 0.0),
        -int(metrics.get("trade_count", 0) or 0),
        -float(metrics.get("profit_factor", 0.0) or 0.0),
    )


def build_symbol_result(frame: pd.DataFrame, symbol: str, train_end: pd.Timestamp) -> dict[str, Any]:
    symbol_frame = frame[frame["symbol"] == symbol].copy().sort_values("ts").reset_index(drop=True)
    train_frame = symbol_frame[symbol_frame["ts"] <= train_end].copy()
    valid_frame = symbol_frame[symbol_frame["ts"] > train_end].copy()
    candidates: list[dict[str, Any]] = []
    for params in PARAM_GRID:
        train_gross = simulate_symbol(train_frame, params)
        valid_gross = simulate_symbol(valid_frame, params)
        train_scenarios = [apply_cost_scenario(train_gross, scenario) for scenario in EXECUTION_COST_SCENARIOS]
        valid_scenarios = [apply_cost_scenario(valid_gross, scenario) for scenario in EXECUTION_COST_SCENARIOS]
        train_selected = next(row for row in train_scenarios if row["scenario_id"] == SELECTION_SCENARIO_ID)
        valid_selected = next(row for row in valid_scenarios if row["scenario_id"] == SELECTION_SCENARIO_ID)
        candidates.append(
            {
                "params": params,
                "train_metrics": train_selected["metrics"],
                "validation_metrics": valid_selected["metrics"],
                "train_gross_metrics": train_gross["metrics"],
                "validation_gross_metrics": valid_gross["metrics"],
                "train_scenarios": train_scenarios,
                "validation_scenarios": valid_scenarios,
                "train_objective": float(objective(train_selected["metrics"])),
                "validation_objective": float(objective(valid_selected["metrics"])),
                "validation_status": classify_validation(valid_selected["metrics"]),
            }
        )
    best = max(
        candidates,
        key=lambda row: (
            float(row["train_objective"]),
            1 if int(row["validation_metrics"]["trade_count"]) >= 2 else 0,
            float(row["validation_objective"]),
        ),
    )
    valid_selected = next(row for row in best["validation_scenarios"] if row["scenario_id"] == SELECTION_SCENARIO_ID)
    return {
        "symbol": symbol,
        "selected_params": best["params"],
        "train_metrics": best["train_metrics"],
        "validation_metrics": best["validation_metrics"],
        "train_gross_metrics": best["train_gross_metrics"],
        "validation_gross_metrics": best["validation_gross_metrics"],
        "validation_scenarios": best["validation_scenarios"],
        "validation_status": best["validation_status"],
        "train_objective": best["train_objective"],
        "validation_objective": best["validation_objective"],
        "validation_trade_sample": valid_selected["trade_sample"][:6],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Price Action Breakout Pullback SIM ONLY",
        "",
        f"- dataset_path: `{text(payload.get('dataset_path'))}`",
        f"- cadence_minutes: `{payload.get('cadence_minutes')}`",
        f"- symbols: `{','.join(payload.get('symbols', []))}`",
        f"- focus_symbol: `{text(payload.get('focus_symbol'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Symbol Ranking",
        "",
    ]
    for row in payload.get("ranking", []):
        lines.append(
            f"- `{row['symbol']}` | status=`{row['validation_status']}` | return=`{row['validation_cumulative_return']:.2%}` | pf=`{row['validation_profit_factor']:.2f}` | expectancy_r=`{row['validation_expectancy_r']:.3f}` | trades=`{row['validation_trade_count']}`"
        )
    lines.extend(["", "## Source Catalog", ""])
    for row in payload.get("source_catalog", []):
        lines.append(f"- `{row['source_id']}` | [{row['title']}]({row['url']}) | {row['summary']}")
    lines.extend(["", "## Notes", "", f"- `{text(payload.get('research_note'))}`", f"- `{text(payload.get('limitation_note'))}`"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a price-action breakout-pullback SIM_ONLY study on local multi-symbol intraday bars.")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser().resolve() if text(args.dataset_path) else select_latest_intraday_dataset(review_dir)
    frame = load_frame(dataset_path)
    requested_symbols = [item.strip().upper() for item in text(args.symbols).split(",") if item.strip()]
    if requested_symbols:
        frame = frame[frame["symbol"].isin(requested_symbols)].copy()
        if frame.empty:
            raise ValueError(f"symbols_not_found_in_dataset:{','.join(requested_symbols)}")
    frame = add_features(frame)
    train_end, valid_end = split_frame(frame, float(args.train_ratio))
    symbols = sorted(frame["symbol"].unique().tolist())
    symbol_results = [build_symbol_result(frame, symbol, train_end) for symbol in symbols]
    ranking = sorted(
        [
            {
                "symbol": row["symbol"],
                "validation_status": row["validation_status"],
                "validation_cumulative_return": float(row["validation_metrics"]["cumulative_return"]),
                "validation_profit_factor": float(row["validation_metrics"]["profit_factor"]),
                "validation_expectancy_r": float(row["validation_metrics"]["expectancy_r"]),
                "validation_trade_count": int(row["validation_metrics"]["trade_count"]),
                "validation_objective": float(row["validation_objective"]),
            }
            for row in symbol_results
        ],
        key=symbol_sort_key,
    )
    focus_symbol = ranking[0]["symbol"] if ranking else ""
    focus_result = next((row for row in symbol_results if row["symbol"] == focus_symbol), {})
    focus_validation_metrics = dict(focus_result.get("validation_metrics") or {})
    focus_validation_gross_metrics = dict(focus_result.get("validation_gross_metrics") or {})
    sample_symbol = str(symbols[0]) if symbols else ""
    sample_ts = pd.to_datetime(frame.loc[frame["symbol"] == sample_symbol, "ts"], utc=True).sort_values()
    cadence = sample_ts.diff().dropna().mode()
    cadence_minutes = int(cadence.iloc[0] / pd.Timedelta(minutes=1)) if not cadence.empty else None
    coverage_start = pd.Timestamp(frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    coverage_end = pd.Timestamp(frame["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    coverage_days = max(0.0, (coverage_end - coverage_start).total_seconds() / 86400.0)
    payload = {
        "action": "build_price_action_breakout_pullback_sim_only",
        "ok": True,
        "status": "ok",
        "change_class": "SIM_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "coverage_start_utc": fmt_utc(coverage_start),
        "coverage_end_utc": fmt_utc(coverage_end),
        "train_end_utc": fmt_utc(train_end.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "validation_end_utc": fmt_utc(valid_end.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "cadence_minutes": cadence_minutes,
        "symbols": symbols,
        "symbols_requested": requested_symbols,
        "symbol_count": len(symbols),
        "selection_scenario_id": SELECTION_SCENARIO_ID,
        "execution_cost_scenarios": EXECUTION_COST_SCENARIOS,
        "source_catalog": SOURCE_CATALOG,
        "focus_symbol": focus_symbol,
        "focus_symbol_result": focus_result,
        "selected_params": dict(focus_result.get("selected_params") or {}),
        "validation_status": text(focus_result.get("validation_status")),
        "validation_metrics": focus_validation_metrics,
        "validation_gross_metrics": focus_validation_gross_metrics,
        "symbol_results": symbol_results,
        "ranking": ranking,
        "recommended_brief": (
            f"{focus_symbol}:{SELECTION_SCENARIO_ID}:status={text(focus_result.get('validation_status'))},"
            f"valid_return={float(focus_validation_metrics.get('cumulative_return', 0.0)):.2%},"
            f"pf={float(focus_validation_metrics.get('profit_factor', 0.0)):.2f},"
            f"expectancy_r={float(focus_validation_metrics.get('expectancy_r', 0.0)):.3f},"
            f"trades={int(focus_validation_metrics.get('trade_count', 0) or 0)}"
        ),
        "research_note": "这轮把更贴近盘感的 breakout-pullback 价格行为 setup 下沉成 15m 研究，并优先测试结构过滤（如 reclaim）与单日止损约束。",
        "limitation_note": f"当前样本约 {coverage_days:.1f} 天 15m 数据；结构过滤仍是基于 OHLCV 的近似，不是逐笔 orderflow；结论只能用于缩窄 setup，不应用于直接放行 live。",
    }
    json_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_sim_only.json"
    md_path = review_dir / f"{args.stamp}_price_action_breakout_pullback_sim_only.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "focus_symbol": focus_symbol}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
