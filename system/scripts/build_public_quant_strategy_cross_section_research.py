#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_RESEARCH_DIR = DEFAULT_OUTPUT_ROOT / "research"


SOURCE_CATALOG: list[dict[str, str]] = [
    {
        "source_id": "quantbuffet_weekly_crypto_momentum",
        "source_type": "public_quant_blog",
        "title": "Weekly Momentum-Based Cryptocurrency Long-Only Trading Strategy",
        "url": "https://quantbuffet.com/en/2024/12/30/time-series-momentum-factor-in-cryptocurrencies/",
        "summary": "周频动量、z-score 标准化、等权或波动率权重、long-only。",
    },
    {
        "source_id": "fmz_adaptive_rsi_breakout",
        "source_type": "community_strategy_page",
        "title": "Adaptive Market Regime RSI and Breakout Hybrid Quantitative Trading Strategy",
        "url": "https://www.fmz.com/lang/en/strategy/494062",
        "summary": "ADX 切换趋势/震荡状态，震荡用 RSI 均值回复，趋势用 breakout，EMA 过滤，ATR 风控。",
    },
    {
        "source_id": "fmz_turtle_trend_evolution",
        "source_type": "community_strategy_page",
        "title": "Turtle Trend Evolution",
        "url": "https://www.fmz.com/lang/en/strategy/520022",
        "summary": "唐奇安突破 + ADX 趋势强度 + ATR 止损 + Heikin Ashi 平滑。",
    },
    {
        "source_id": "tradingview_ad_line",
        "source_type": "platform_support_doc",
        "title": "Advance/Decline Line",
        "url": "https://www.tradingview.com/support/solutions/43000589092-advance-decline-line/",
        "summary": "用涨跌家数构造市场广度，验证趋势健康度或识别背离。",
    },
    {
        "source_id": "tradingview_volume_profile_basics",
        "source_type": "platform_support_doc",
        "title": "Volume profile indicators: basic concepts",
        "url": "https://www.tradingview.com/support/solutions/43000502040-volume-profile-indicators-basic-concepts/",
        "summary": "POC/VAH/VAL/HVN/LVN 概念与基于成交量分布的支撑阻力解释。",
    },
    {
        "source_id": "tradingview_vpvr_community",
        "source_type": "open_source_community_script",
        "title": "Volume Profile Visible Range (VPVR) with POC Price",
        "url": "https://www.tradingview.com/script/KkKsTgSf/",
        "summary": "可视区间 VPVR、POC、价值区与高/低成交量节点。",
    },
    {
        "source_id": "tradingview_volumebreakout",
        "source_type": "open_source_community_script",
        "title": "VOLUMEBREAKOUT",
        "url": "https://tr.tradingview.com/scripts/volumebreakout/",
        "summary": "通过 K 线几何或低级别 intrabar 近似买卖量，做 breakout confirmation。",
    },
]


FAMILY_CATALOG: list[dict[str, Any]] = [
    {
        "family_id": "weekly_crypto_momentum",
        "label": "周动量轮动",
        "category": "trend_momentum",
        "positioning": "long_only_cross_section",
        "quantization_note": "按 5/10/15 日收益做横截面排名；参考公开周频动量思路，数据不足时用日线近似周调仓。",
        "source_ids": ["quantbuffet_weekly_crypto_momentum"],
    },
    {
        "family_id": "breadth_filtered_momentum",
        "label": "广度过滤动量",
        "category": "breadth_overlay",
        "positioning": "long_only_cross_section",
        "quantization_note": "把 AD Line/Advance-Decline Ratio 作为 universe regime filter，只在广度健康时放行动量排序。",
        "source_ids": ["quantbuffet_weekly_crypto_momentum", "tradingview_ad_line"],
    },
    {
        "family_id": "adaptive_rsi_breakout",
        "label": "ADX 状态切换混合",
        "category": "regime_switching",
        "positioning": "long_only_cross_section",
        "quantization_note": "高 ADX 用 breakout，低 ADX 用 RSI 均值回复；EMA 过滤用 50 日代替 200 日以适配小样本。",
        "source_ids": ["fmz_adaptive_rsi_breakout"],
    },
    {
        "family_id": "turtle_breakout_adx",
        "label": "海龟趋势改良",
        "category": "trend_breakout",
        "positioning": "long_only_cross_section",
        "quantization_note": "Donchian breakout + ADX + ATR；Heikin Ashi 仅用于信号平滑，不做完整持仓复制。",
        "source_ids": ["fmz_turtle_trend_evolution"],
    },
    {
        "family_id": "vpvr_acceptance_reclaim",
        "label": "VPVR 接受区回收",
        "category": "volume_structure",
        "positioning": "long_only_cross_section",
        "quantization_note": "当前数据只有日线 OHLCV，无法重建真实 VPVR；这里用 rolling volume-weighted close 作为 POC proxy。",
        "source_ids": ["tradingview_volume_profile_basics", "tradingview_vpvr_community"],
    },
    {
        "family_id": "geometry_delta_breakout",
        "label": "几何 Delta 突破确认",
        "category": "orderflow_proxy",
        "positioning": "long_only_cross_section",
        "quantization_note": "按 TradingView VOLUMEBREAKOUT 的几何近似思想，用 K 线实体/位置拆分近似买卖量。",
        "source_ids": ["tradingview_volumebreakout"],
    },
]


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | pd.Timestamp | None) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        dt_value = value.to_pydatetime()
    else:
        dt_value = value
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=dt.timezone.utc)
    return dt_value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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
    out = frame.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce").dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out = out.dropna(subset=["ts", "symbol", "open", "high", "low", "close", "volume"]).copy()
    out = out.sort_values(["symbol", "ts"]).drop_duplicates(subset=["symbol", "ts"]).reset_index(drop=True)
    if "asset_class" not in out.columns:
        out["asset_class"] = "crypto"
    else:
        out["asset_class"] = out["asset_class"].astype(str).str.lower().str.strip()
    return out


def select_dataset(
    *,
    dataset_path: str,
    research_dir: Path,
    min_symbols: int,
    min_days: int,
) -> Path:
    if text(dataset_path):
        return Path(dataset_path).expanduser().resolve()
    candidates = sorted(list(research_dir.glob("*/bars_used.parquet")) + list(research_dir.glob("*/bars_used.csv")))
    scored: list[tuple[int, int, str, Path]] = []
    for path in candidates:
        try:
            frame = load_frame(path)
        except Exception:
            continue
        if frame.empty:
            continue
        if set(frame["asset_class"].unique()) != {"crypto"}:
            continue
        symbol_count = int(frame["symbol"].nunique())
        day_count = int(frame["ts"].nunique())
        if symbol_count < min_symbols or day_count < min_days:
            continue
        scored.append((day_count, symbol_count, str(path.parent.name), path))
    if not scored:
        raise FileNotFoundError("no_crypto_bars_used_dataset_found")
    scored.sort(reverse=True)
    return scored[0][3]


def compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / max(1, window), adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / max(1, window), adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_adx(group: pd.DataFrame, window: int) -> pd.Series:
    work = group.copy()
    high = work["high"]
    low = work["low"]
    close = work["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / max(1, window), adjust=False, min_periods=window).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=work.index).ewm(
        alpha=1.0 / max(1, window), adjust=False, min_periods=window
    ).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=work.index).ewm(
        alpha=1.0 / max(1, window), adjust=False, min_periods=window
    ).mean() / atr.replace(0.0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100.0
    return dx.ewm(alpha=1.0 / max(1, window), adjust=False, min_periods=window).mean()


def compute_heikin_ashi(group: pd.DataFrame) -> pd.DataFrame:
    work = group.copy()
    ha_close = (work["open"] + work["high"] + work["low"] + work["close"]) / 4.0
    ha_open = pd.Series(index=work.index, dtype=float)
    for idx, row_index in enumerate(work.index):
        if idx == 0:
            ha_open.loc[row_index] = float((work.loc[row_index, "open"] + work.loc[row_index, "close"]) / 2.0)
        else:
            prev_index = work.index[idx - 1]
            ha_open.loc[row_index] = float((ha_open.loc[prev_index] + ha_close.loc[prev_index]) / 2.0)
    ha_high = pd.concat([work["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([work["low"], ha_open, ha_close], axis=1).min(axis=1)
    return pd.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        },
        index=work.index,
    )


def rolling_volume_weighted_close(frame: pd.DataFrame, lookback: int) -> pd.Series:
    min_periods = max(5, lookback // 2)
    numerator = (frame["close"] * frame["volume"]).groupby(frame["symbol"]).transform(
        lambda s: s.rolling(lookback, min_periods=min_periods).sum()
    )
    denominator = frame["volume"].groupby(frame["symbol"]).transform(
        lambda s: s.rolling(lookback, min_periods=min_periods).sum()
    )
    return numerator / denominator.replace(0.0, np.nan)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    grouped = work.groupby("symbol", group_keys=False)
    work["ret_1d"] = grouped["close"].pct_change()
    work["ret_fwd_1d"] = grouped["close"].shift(-1) / work["close"] - 1.0
    for lookback in (3, 5, 10, 15, 20, 30, 55):
        work[f"ret_{lookback}d"] = grouped["close"].pct_change(lookback)
    work["ema_20"] = grouped["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    work["ema_50"] = grouped["close"].transform(lambda s: s.ewm(span=50, adjust=False).mean())
    prev_close = grouped["close"].shift(1)
    tr = pd.concat(
        [
            (work["high"] - work["low"]).abs(),
            (work["high"] - prev_close).abs(),
            (work["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    work["true_range"] = tr
    work["atr_14"] = work.groupby("symbol")["true_range"].transform(
        lambda s: s.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    )
    work["adx_14"] = grouped.apply(lambda g: compute_adx(g, 14)).reset_index(level=0, drop=True)
    work["rsi_7"] = grouped["close"].transform(lambda s: compute_rsi(s, 7))
    work["rsi_14"] = grouped["close"].transform(lambda s: compute_rsi(s, 14))
    work["volume_mean_20"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    work["volume_std_20"] = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=10).std(ddof=0))
    work["volume_z_20"] = (
        (work["volume"] - work["volume_mean_20"]) / work["volume_std_20"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    work["range"] = (work["high"] - work["low"]).clip(lower=1e-9)
    work["delta_geometry"] = ((2.0 * work["close"] - work["high"] - work["low"]) / work["range"]) * work["volume"]
    work["delta_geom_mean_20"] = grouped["delta_geometry"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    work["delta_geom_std_20"] = grouped["delta_geometry"].transform(lambda s: s.rolling(20, min_periods=10).std(ddof=0))
    work["delta_geom_z_20"] = (
        (work["delta_geometry"] - work["delta_geom_mean_20"]) / work["delta_geom_std_20"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    for lookback in (5, 10, 20, 55):
        work[f"roll_high_{lookback}_prev"] = grouped["high"].transform(
            lambda s, lb=lookback: s.rolling(lb, min_periods=lb).max().shift(1)
        )
        work[f"roll_low_{lookback}_prev"] = grouped["low"].transform(
            lambda s, lb=lookback: s.rolling(lb, min_periods=lb).min().shift(1)
        )
    work["vol_20"] = grouped["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std(ddof=0))
    work["poc_proxy_20"] = rolling_volume_weighted_close(work, 20)
    ha = grouped.apply(compute_heikin_ashi).reset_index(level=0, drop=True)
    for column in ha.columns:
        work[column] = ha[column]
    work["ha_breakout_20"] = work["ha_close"] > grouped["ha_high"].transform(
        lambda s: s.rolling(20, min_periods=20).max().shift(1)
    )
    return work


def add_breadth(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    daily = (
        work.groupby("ts", as_index=False)
        .agg(
            adv_count=("ret_1d", lambda s: int((s > 0.0).sum())),
            dec_count=("ret_1d", lambda s: int((s < 0.0).sum())),
            flat_count=("ret_1d", lambda s: int((s.fillna(0.0) == 0.0).sum())),
            pct_above_ema20=("close", lambda s: 0.0),
        )
        .sort_values("ts")
        .reset_index(drop=True)
    )
    pct_above = (
        work.assign(above_ema20=(work["close"] > work["ema_20"]).astype(float))
        .groupby("ts")["above_ema20"]
        .mean()
        .reset_index(name="pct_above_ema20")
    )
    daily = daily.drop(columns=["pct_above_ema20"]).merge(pct_above, on="ts", how="left")
    daily["net_adv"] = daily["adv_count"] - daily["dec_count"]
    daily["ad_line"] = daily["net_adv"].cumsum()
    daily["ad_ratio"] = daily["adv_count"] / daily["dec_count"].replace(0, np.nan)
    daily["ad_ratio"] = daily["ad_ratio"].replace([np.inf, -np.inf], np.nan).fillna(float(daily["adv_count"].max() + 1))
    daily["ad_line_slope_3"] = daily["ad_line"] - daily["ad_line"].shift(3)
    daily["ad_line_slope_5"] = daily["ad_line"] - daily["ad_line"].shift(5)
    daily["breadth_health"] = (
        (daily["ad_line_slope_3"] > 0.0).astype(float)
        + (daily["ad_ratio"] > 1.0).astype(float)
        + (daily["pct_above_ema20"] >= 0.5).astype(float)
    )
    return work.merge(daily, on="ts", how="left")


def zscore_by_date(series: pd.Series, dates: pd.Series) -> pd.Series:
    grouped = series.groupby(dates)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def score_momentum(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    lookback = int(params["lookback"])
    raw = frame[f"ret_{lookback}d"]
    score = zscore_by_date(raw, frame["ts"])
    return score.where(score > 0.0, 0.0).fillna(0.0)


def score_breadth_filtered_momentum(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    lookback = int(params["lookback"])
    breadth_lb = int(params["breadth_lb"])
    raw = zscore_by_date(frame[f"ret_{lookback}d"], frame["ts"])
    breadth_ok = (frame[f"ad_line_slope_{breadth_lb}"] > 0.0) & (frame["ad_ratio"] > 1.0)
    return raw.where(breadth_ok & (raw > 0.0), 0.0).fillna(0.0)


def score_adaptive_rsi_breakout(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    adx_threshold = float(params["adx_threshold"])
    breakout_lb = int(params["breakout_lookback"])
    rsi_buy = float(params["rsi_buy"])
    trending = frame["adx_14"] >= adx_threshold
    bias_ok = frame["close"] > frame["ema_50"]
    breakout_strength = (
        (frame["close"] / frame[f"roll_high_{breakout_lb}_prev"] - 1.0)
        / frame["atr_14"].replace(0.0, np.nan).div(frame["close"])
    ).replace([np.inf, -np.inf], np.nan)
    breakout_score = breakout_strength.where(
        trending & bias_ok & (frame["close"] > frame[f"roll_high_{breakout_lb}_prev"]),
        0.0,
    )
    mean_reversion_score = (
        ((rsi_buy - frame["rsi_7"]) / max(1.0, rsi_buy)).clip(lower=0.0)
        + (-frame["ret_3d"]).clip(lower=0.0)
    ).where((~trending) & bias_ok & (frame["rsi_7"] < rsi_buy), 0.0)
    volume_bonus = frame["volume_z_20"].clip(lower=0.0).fillna(0.0) * 0.15
    return (breakout_score.fillna(0.0) + mean_reversion_score.fillna(0.0) + volume_bonus).fillna(0.0)


def score_turtle_breakout_adx(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    breakout_lb = int(params["breakout_lookback"])
    adx_threshold = float(params["adx_threshold"])
    volume_threshold = float(params["volume_z_threshold"])
    breakout = (
        (frame["ha_close"] > frame[f"roll_high_{breakout_lb}_prev"])
        & (frame["adx_14"] >= adx_threshold)
        & (frame["volume_z_20"] >= volume_threshold)
    )
    strength = (
        (frame["ha_close"] / frame[f"roll_high_{breakout_lb}_prev"] - 1.0)
        / frame["atr_14"].replace(0.0, np.nan).div(frame["close"])
    ).replace([np.inf, -np.inf], np.nan)
    return strength.where(breakout, 0.0).fillna(0.0)


def score_vpvr_acceptance_reclaim(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    poc_lb = int(params["poc_lookback"])
    volume_threshold = float(params["volume_z_threshold"])
    breakout_lb = int(params["breakout_lookback"])
    poc_proxy = rolling_volume_weighted_close(frame, poc_lb)
    prev_poc = poc_proxy.groupby(frame["symbol"]).shift(1)
    prev_close = frame.groupby("symbol")["close"].shift(1)
    reclaim = (
        (frame["close"] > poc_proxy)
        & (prev_close <= prev_poc)
        & (frame["volume_z_20"] >= volume_threshold)
        & (frame["close"] > frame[f"roll_high_{breakout_lb}_prev"])
    )
    distance = ((frame["close"] / poc_proxy) - 1.0).replace([np.inf, -np.inf], np.nan)
    return (distance + 0.1 * frame["volume_z_20"].clip(lower=0.0)).where(reclaim, 0.0).fillna(0.0)


def score_geometry_delta_breakout(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    breakout_lb = int(params["breakout_lookback"])
    delta_z_threshold = float(params["delta_z_threshold"])
    signal = (
        (frame["close"] > frame[f"roll_high_{breakout_lb}_prev"])
        & (frame["delta_geom_z_20"] >= delta_z_threshold)
        & (frame["volume_z_20"] >= 0.0)
    )
    strength = (
        (frame["close"] / frame[f"roll_high_{breakout_lb}_prev"] - 1.0).replace([np.inf, -np.inf], np.nan)
        + 0.1 * frame["delta_geom_z_20"].clip(lower=0.0)
    )
    return strength.where(signal, 0.0).fillna(0.0)


FAMILY_SCORERS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], pd.Series]] = {
    "weekly_crypto_momentum": score_momentum,
    "breadth_filtered_momentum": score_breadth_filtered_momentum,
    "adaptive_rsi_breakout": score_adaptive_rsi_breakout,
    "turtle_breakout_adx": score_turtle_breakout_adx,
    "vpvr_acceptance_reclaim": score_vpvr_acceptance_reclaim,
    "geometry_delta_breakout": score_geometry_delta_breakout,
}


PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "weekly_crypto_momentum": [
        {"lookback": lookback, "top_k": top_k, "rebalance_every": 5, "weight_mode": weight_mode}
        for lookback, top_k, weight_mode in itertools.product([5, 10, 15], [2, 3], ["equal", "inv_vol"])
    ],
    "breadth_filtered_momentum": [
        {"lookback": lookback, "breadth_lb": breadth_lb, "top_k": top_k, "rebalance_every": 5, "weight_mode": "equal"}
        for lookback, breadth_lb, top_k in itertools.product([5, 10], [3, 5], [2, 3])
    ],
    "adaptive_rsi_breakout": [
        {"adx_threshold": adx_threshold, "breakout_lookback": breakout_lookback, "rsi_buy": rsi_buy, "top_k": top_k, "rebalance_every": 1, "weight_mode": "equal"}
        for adx_threshold, breakout_lookback, rsi_buy, top_k in itertools.product([18, 20, 22], [10, 20], [35, 40], [1, 2])
    ],
    "turtle_breakout_adx": [
        {"breakout_lookback": breakout_lookback, "adx_threshold": adx_threshold, "volume_z_threshold": volume_z_threshold, "top_k": top_k, "rebalance_every": 1, "weight_mode": "equal"}
        for breakout_lookback, adx_threshold, volume_z_threshold, top_k in itertools.product([20, 55], [18, 20, 25], [0.0, 0.5], [1, 2])
    ],
    "vpvr_acceptance_reclaim": [
        {"poc_lookback": poc_lookback, "volume_z_threshold": volume_z_threshold, "breakout_lookback": breakout_lookback, "top_k": top_k, "rebalance_every": 1, "weight_mode": "equal"}
        for poc_lookback, volume_z_threshold, breakout_lookback, top_k in itertools.product([10, 20], [0.5, 1.0], [5, 10], [1, 2])
    ],
    "geometry_delta_breakout": [
        {"breakout_lookback": breakout_lookback, "delta_z_threshold": delta_z_threshold, "top_k": top_k, "rebalance_every": 1, "weight_mode": "equal"}
        for breakout_lookback, delta_z_threshold, top_k in itertools.product([10, 20], [0.0, 0.5, 1.0], [1, 2])
    ],
}


def choose_weights(selected: pd.DataFrame, weight_mode: str) -> dict[str, float]:
    if selected.empty:
        return {}
    if weight_mode == "inv_vol":
        vol = selected["vol_20"].replace(0.0, np.nan)
        inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        total = float(inv.sum())
        if total > 0.0:
            return {str(symbol): float(weight / total) for symbol, weight in zip(selected["symbol"], inv)}
    equal_weight = 1.0 / float(len(selected))
    return {str(symbol): equal_weight for symbol in selected["symbol"]}


def average_rank_ic(chunk: pd.DataFrame) -> float:
    ranked = chunk[["score", "ret_fwd_1d"]].dropna()
    if len(ranked) < 3:
        return float("nan")
    if ranked["score"].nunique() < 2 or ranked["ret_fwd_1d"].nunique() < 2:
        return float("nan")
    return float(ranked["score"].rank().corr(ranked["ret_fwd_1d"].rank(), method="pearson"))


def backtest_long_only(
    frame: pd.DataFrame,
    *,
    score_col: str,
    top_k: int,
    rebalance_every: int,
    weight_mode: str,
) -> dict[str, Any]:
    work = frame.sort_values(["ts", "symbol"]).copy()
    dates = list(pd.Series(sorted(work["ts"].dropna().unique())).astype("datetime64[ns]"))
    positions: dict[str, float] = {}
    trade_count = 0
    daily_rows: list[dict[str, Any]] = []
    holdings_rows: list[dict[str, Any]] = []
    last_selection: set[str] = set()
    for idx, current_ts in enumerate(dates[:-1]):
        day = work[work["ts"] == current_ts].copy()
        next_ts = dates[idx + 1]
        rebalance = idx % max(1, rebalance_every) == 0
        if rebalance:
            selected = day[day[score_col] > 0.0].sort_values(score_col, ascending=False).head(max(1, int(top_k)))
            weights = choose_weights(selected, weight_mode)
            current_selection = set(weights)
            trade_count += len(current_selection.symmetric_difference(last_selection))
            last_selection = current_selection
            positions = weights
        portfolio_return = 0.0
        if positions:
            day_returns = day.set_index("symbol")["ret_fwd_1d"].to_dict()
            portfolio_return = float(sum(float(day_returns.get(symbol, 0.0) or 0.0) * weight for symbol, weight in positions.items()))
        score_ic = average_rank_ic(day.rename(columns={score_col: "score"}))
        daily_rows.append(
            {
                "ts": current_ts,
                "next_ts": next_ts,
                "portfolio_return": portfolio_return,
                "active_positions": int(len(positions)),
                "selected_symbols": sorted(positions.keys()),
                "rank_ic": score_ic,
            }
        )
        for symbol, weight in positions.items():
            holdings_rows.append({"ts": current_ts, "symbol": symbol, "weight": float(weight)})
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        return {
            "daily": daily_rows,
            "holdings": holdings_rows,
            "metrics": {
                "active_days": 0,
                "trade_count": 0,
                "win_rate": 0.0,
                "cumulative_return": 0.0,
                "annual_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "avg_daily_return": 0.0,
                "avg_cross_section_ic": 0.0,
            },
        }
    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()
    equity = daily["equity"]
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0
    avg_ret = float(daily["portfolio_return"].mean())
    std_ret = float(daily["portfolio_return"].std(ddof=0))
    sharpe = float((avg_ret / std_ret) * math.sqrt(365.0)) if std_ret > 0.0 else 0.0
    cumulative_return = float(equity.iloc[-1] - 1.0)
    periods = int(len(daily))
    annual_return = float(equity.iloc[-1] ** (365.0 / max(1, periods)) - 1.0)
    metrics = {
        "active_days": int((daily["active_positions"] > 0).sum()),
        "trade_count": int(trade_count),
        "win_rate": float((daily["portfolio_return"] > 0.0).mean()),
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": float(abs(drawdown.min())),
        "avg_daily_return": avg_ret,
        "avg_cross_section_ic": float(pd.Series(daily["rank_ic"]).dropna().mean() or 0.0),
    }
    return {
        "daily": [
            {
                "ts_utc": fmt_utc(pd.Timestamp(row["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "next_ts_utc": fmt_utc(pd.Timestamp(row["next_ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "portfolio_return": round(float(row["portfolio_return"]), 6),
                "active_positions": int(row["active_positions"]),
                "selected_symbols": list(row["selected_symbols"]),
                "rank_ic": round(float(row["rank_ic"]), 6) if pd.notna(row["rank_ic"]) else None,
            }
            for row in daily.to_dict("records")
        ],
        "holdings": holdings_rows,
        "metrics": metrics,
    }


def objective(metrics: dict[str, Any]) -> float:
    active_days = int(metrics.get("active_days", 0) or 0)
    trade_count = int(metrics.get("trade_count", 0) or 0)
    avg_ic = float(metrics.get("avg_cross_section_ic", 0.0) or 0.0)
    if not math.isfinite(avg_ic):
        avg_ic = 0.0
    activity_penalty = 0.0
    if active_days <= 0 or trade_count <= 0:
        activity_penalty -= 40.0
    elif active_days < 3 or trade_count < 6:
        activity_penalty -= 18.0
    elif active_days < 8 or trade_count < 12:
        activity_penalty -= 8.0
    activity_bonus = min(active_days, 20) * 0.2 + min(trade_count, 24) * 0.08
    return float(
        (float(metrics.get("cumulative_return", 0.0)) * 100.0)
        + (float(metrics.get("sharpe", 0.0)) * 10.0)
        + (avg_ic * 35.0)
        - (float(metrics.get("max_drawdown", 0.0)) * 80.0)
        + (float(metrics.get("win_rate", 0.0)) * 5.0)
        + activity_bonus
        + activity_penalty
    )


def split_dates(frame: pd.DataFrame, train_ratio: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.Series(sorted(frame["ts"].dropna().unique())).astype("datetime64[ns]")
    if len(unique_dates) < 20:
        raise ValueError("dataset_too_small_for_train_valid_split")
    split_idx = max(10, min(len(unique_dates) - 5, int(math.floor(len(unique_dates) * train_ratio))))
    return pd.Timestamp(unique_dates.iloc[split_idx - 1]), pd.Timestamp(unique_dates.iloc[-1])


def classify_validation(metrics: dict[str, Any]) -> str:
    active_days = int(metrics.get("active_days", 0) or 0)
    trade_count = int(metrics.get("trade_count", 0) or 0)
    cumulative_return = float(metrics.get("cumulative_return", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
    if active_days <= 0 or trade_count <= 0:
        return "inactive_oos"
    if cumulative_return > 0.03 and sharpe > 0.5 and max_drawdown < 0.15:
        return "promising_small_sample"
    if cumulative_return > 0.0 and max_drawdown < 0.20:
        return "mixed_positive"
    return "not_promising"


def build_research_action_ladder(strategy_results: list[dict[str, Any]]) -> dict[str, Any]:
    continue_research: list[str] = []
    data_gap: list[str] = []
    deprioritize: list[str] = []
    for row in strategy_results:
        family_id = text(row.get("family_id"))
        status = text(row.get("validation_status"))
        metrics = dict(row.get("validation_metrics") or {})
        validation_return = float(metrics.get("cumulative_return", 0.0) or 0.0)
        validation_ic = float(metrics.get("avg_cross_section_ic", 0.0) or 0.0)
        if not math.isfinite(validation_ic):
            validation_ic = 0.0
        validation_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
        trade_count = int(metrics.get("trade_count", 0) or 0)
        active_days = int(metrics.get("active_days", 0) or 0)
        if status == "inactive_oos":
            data_gap.append(family_id)
            continue
        if trade_count >= 3 and active_days >= 1 and validation_ic > 0.10 and validation_dd <= 0.10:
            continue_research.append(family_id)
            continue
        if validation_return < 0.0 and (validation_ic <= 0.05 or validation_dd > 0.20):
            deprioritize.append(family_id)
            continue
        continue_research.append(family_id)
    if not continue_research and data_gap:
        continue_research = data_gap[:1]
    next_data_requirements: list[str] = []
    if data_gap:
        next_data_requirements.append("为 inactive_oos 家族补更长验证窗或 1h/15m bars，避免零交易策略占用排序头部。")
    if "geometry_delta_breakout" in continue_research:
        next_data_requirements.append("优先给 geometry_delta_breakout 补 1h/15m bars 与真实成交约束，验证正 IC 能否转成正收益。")
    if any(item in deprioritize for item in ("weekly_crypto_momentum", "adaptive_rsi_breakout")):
        next_data_requirements.append("周动量与 ADX-RSI 混合在当前 crypto 小样本上未见 edge，除非扩样本或换市场，否则先降权。")
    return {
        "continue_research_families": continue_research,
        "data_gap_families": data_gap,
        "deprioritize_families": deprioritize,
        "next_data_requirements": next_data_requirements,
        "focus_family": continue_research[0] if continue_research else "",
    }


def build_family_result(frame: pd.DataFrame, family_id: str, meta: dict[str, Any], train_end: pd.Timestamp) -> dict[str, Any]:
    scorer = FAMILY_SCORERS[family_id]
    train_frame = frame[frame["ts"] <= train_end].copy()
    valid_frame = frame[frame["ts"] > train_end].copy()
    if train_frame.empty or valid_frame.empty:
        raise ValueError("empty_train_or_valid_frame")
    candidates: list[dict[str, Any]] = []
    for params in PARAM_GRIDS[family_id]:
        scored_train = train_frame.copy()
        scored_train["strategy_score"] = scorer(scored_train, params)
        train_bt = backtest_long_only(
            scored_train,
            score_col="strategy_score",
            top_k=int(params["top_k"]),
            rebalance_every=int(params["rebalance_every"]),
            weight_mode=str(params["weight_mode"]),
        )
        train_metrics = dict(train_bt["metrics"])
        train_score = objective(train_metrics)
        candidates.append(
            {
                "params": params,
                "train_metrics": train_metrics,
                "train_score": train_score,
            }
        )
    best = max(candidates, key=lambda row: row["train_score"])
    params = dict(best["params"])
    scored_train = train_frame.copy()
    scored_valid = valid_frame.copy()
    scored_train["strategy_score"] = scorer(scored_train, params)
    scored_valid["strategy_score"] = scorer(scored_valid, params)
    train_bt = backtest_long_only(
        scored_train,
        score_col="strategy_score",
        top_k=int(params["top_k"]),
        rebalance_every=int(params["rebalance_every"]),
        weight_mode=str(params["weight_mode"]),
    )
    valid_bt = backtest_long_only(
        scored_valid,
        score_col="strategy_score",
        top_k=int(params["top_k"]),
        rebalance_every=int(params["rebalance_every"]),
        weight_mode=str(params["weight_mode"]),
    )
    all_scored = frame.copy()
    all_scored["strategy_score"] = scorer(all_scored, params)
    sample_signal_rows = (
        all_scored[all_scored["strategy_score"] > 0.0]
        .sort_values(["ts", "strategy_score"], ascending=[False, False])
        .head(12)
    )
    source_rows = [row for row in SOURCE_CATALOG if row["source_id"] in set(meta["source_ids"])]
    return {
        "family_id": family_id,
        "label": meta["label"],
        "category": meta["category"],
        "positioning": meta["positioning"],
        "quantization_note": meta["quantization_note"],
        "source_rows": source_rows,
        "grid_size": int(len(PARAM_GRIDS[family_id])),
        "selected_params": params,
        "train_metrics": train_bt["metrics"],
        "validation_metrics": valid_bt["metrics"],
        "train_objective": float(objective(train_bt["metrics"])),
        "validation_objective": float(objective(valid_bt["metrics"])),
        "validation_status": classify_validation(valid_bt["metrics"]),
        "sample_signals": [
            {
                "ts_utc": fmt_utc(pd.Timestamp(row["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
                "symbol": text(row["symbol"]).upper(),
                "score": round(float(row["strategy_score"]), 6),
                "ret_fwd_1d": round(float(row["ret_fwd_1d"]), 6) if pd.notna(row["ret_fwd_1d"]) else None,
            }
            for row in sample_signal_rows.to_dict("records")
        ],
        "train_daily_sample": list(train_bt["daily"][:8]),
        "validation_daily_sample": list(valid_bt["daily"][:8]),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Public Quant Strategy Cross Section Research",
        "",
        f"- dataset: `{text(payload.get('dataset_path'))}`",
        f"- universe: `{text(payload.get('universe_brief'))}`",
        f"- sample_window: `{text(payload.get('coverage_start_utc'))} -> {text(payload.get('coverage_end_utc'))}`",
        f"- train_end_utc: `{text(payload.get('train_end_utc'))}`",
        f"- strategy_family_count: `{len(payload.get('strategy_results', []))}`",
        f"- recommended_family: `{text(payload.get('recommended_family_id'))}`",
        f"- recommended_brief: `{text(payload.get('recommended_brief'))}`",
        "",
        "## Strategy Ranking",
        "",
    ]
    for row in payload.get("ranking", []):
        lines.append(
            f"- `{row['family_id']}` | category=`{row['category']}` | status=`{row['validation_status']}` | valid_return=`{row['validation_cumulative_return']:.2%}` | valid_sharpe=`{row['validation_sharpe']:.2f}` | valid_mdd=`{row['validation_max_drawdown']:.2%}` | valid_ic=`{row['validation_avg_cross_section_ic']:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Action Ladder",
            "",
            f"- continue_research: `{','.join(payload.get('research_action_ladder', {}).get('continue_research_families', [])) or '-'}`",
            f"- data_gap: `{','.join(payload.get('research_action_ladder', {}).get('data_gap_families', [])) or '-'}`",
            f"- deprioritize: `{','.join(payload.get('research_action_ladder', {}).get('deprioritize_families', [])) or '-'}`",
        ]
    )
    for row in payload.get("research_action_ladder", {}).get("next_data_requirements", []):
        lines.append(f"- next_data_requirement: `{row}`")
    lines.extend(
        [
            "",
            "## Source Catalog",
            "",
        ]
    )
    for row in payload.get("source_catalog", []):
        lines.append(f"- `{row['source_id']}` | [{row['title']}]({row['url']}) | {row['summary']}")
    lines.extend(
        [
            "",
            "## Research Notes",
            "",
            f"- `{text(payload.get('research_note'))}`",
            f"- `{text(payload.get('limitation_note'))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a public-web-informed quant strategy classification and cross-section backtest report.")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--research-dir", default=str(DEFAULT_RESEARCH_DIR))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--min-symbols", type=int, default=6)
    parser.add_argument("--min-days", type=int, default=90)
    parser.add_argument("--train-ratio", type=float, default=0.67)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    research_dir = Path(args.research_dir).expanduser().resolve()
    dataset_path = select_dataset(
        dataset_path=args.dataset_path,
        research_dir=research_dir,
        min_symbols=int(args.min_symbols),
        min_days=int(args.min_days),
    )
    frame = add_breadth(add_features(load_frame(dataset_path)))
    train_end, dataset_end = split_dates(frame, float(args.train_ratio))
    strategy_results = [
        build_family_result(frame, meta["family_id"], meta, train_end)
        for meta in FAMILY_CATALOG
    ]
    ranking = sorted(
        [
            {
                "family_id": row["family_id"],
                "label": row["label"],
                "category": row["category"],
                "validation_status": row["validation_status"],
                "validation_cumulative_return": float(row["validation_metrics"]["cumulative_return"]),
                "validation_sharpe": float(row["validation_metrics"]["sharpe"]),
                "validation_max_drawdown": float(row["validation_metrics"]["max_drawdown"]),
                "validation_avg_cross_section_ic": float(row["validation_metrics"]["avg_cross_section_ic"] or 0.0)
                if math.isfinite(float(row["validation_metrics"]["avg_cross_section_ic"] or 0.0))
                else 0.0,
                "validation_objective": float(row["validation_objective"]),
            }
            for row in strategy_results
        ],
        key=lambda row: (
            text(row.get("validation_status")) == "inactive_oos",
            -float(row["validation_objective"]),
        ),
        reverse=False,
    )
    recommended_pool = [row for row in ranking if text(row.get("validation_status")) != "inactive_oos"]
    recommended = recommended_pool[0] if recommended_pool else (ranking[0] if ranking else {})
    payload = {
        "action": "build_public_quant_strategy_cross_section_research",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "dataset_path": str(dataset_path),
        "dataset_run_dir": str(dataset_path.parent.name),
        "coverage_start_utc": fmt_utc(pd.Timestamp(frame["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "coverage_end_utc": fmt_utc(pd.Timestamp(frame["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "train_end_utc": fmt_utc(train_end.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "validation_end_utc": fmt_utc(dataset_end.to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        "bar_rows": int(len(frame)),
        "symbol_count": int(frame["symbol"].nunique()),
        "universe_brief": ",".join(sorted(frame["symbol"].unique())),
        "source_catalog": SOURCE_CATALOG,
        "family_catalog": FAMILY_CATALOG,
        "strategy_results": strategy_results,
        "ranking": ranking,
        "research_action_ladder": build_research_action_ladder(strategy_results),
        "recommended_family_id": text(recommended.get("family_id")),
        "recommended_brief": (
            f"{text(recommended.get('family_id'))}:"
            f"status={text(recommended.get('validation_status'))},"
            f"valid_return={float(recommended.get('validation_cumulative_return') or 0.0):.2%},"
            f"valid_sharpe={float(recommended.get('validation_sharpe') or 0.0):.2f},"
            f"valid_mdd={float(recommended.get('validation_max_drawdown') or 0.0):.2%},"
            f"valid_ic={float(recommended.get('validation_avg_cross_section_ic') or 0.0):.3f}"
        ),
        "research_note": "本轮只使用公开网页/社区策略摘要与本地现成日线 OHLCV 小样本，不碰 live execution path。",
        "limitation_note": "VPVR/买卖量差真实重建需要 lower-timeframe volume profile；当前结果仅为 daily proxy 研究，不应用于直接实盘放行。",
        "artifacts": {
            "dataset_path": str(dataset_path),
        },
    }
    json_path = review_dir / f"{args.stamp}_public_quant_strategy_cross_section_research.json"
    md_path = review_dir / f"{args.stamp}_public_quant_strategy_cross_section_research.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "recommended_family_id": payload["recommended_family_id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
