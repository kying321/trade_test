#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")

LONG_ONLY_ASSET_CLASSES = {"equity", "etf"}

BOOK_SOURCES: list[dict[str, Any]] = [
    {
        "topic": "trend_pullback_continuation",
        "book_path": "/Users/jokenrobot/Downloads/Al Brooks 价格行为交易 (趋势篇) 修訂版 (阿尔·布鲁克斯Al Brooks) (z-library.sk, 1lib.sk, z-lib.sk).pdf",
        "pages": [105, 106, 113, 117, 127],
        "summary": "Trend spikes usually evolve into channels and then tests; most countertrend attempts fail, so pullbacks in strong trends deserve stronger priors than reversals.",
    },
    {
        "topic": "breakout_pullback_resume",
        "book_path": "/Users/jokenrobot/Downloads/Al Brooks 价格行为交易 (区间篇) 元宵版 (阿尔·布鲁克斯Al Brooks) (z-library.sk, 1lib.sk, z-lib.sk).pdf",
        "pages": [3, 32, 55, 59],
        "summary": "Most breakouts fail, but the successful ones usually offer a breakout test or pullback that resumes in the original direction.",
    },
    {
        "topic": "failed_breakout_reversal",
        "book_path": "/Users/jokenrobot/Downloads/Al Brooks 价格行为交易 (区间篇) 元宵版 (阿尔·布鲁克斯Al Brooks) (z-library.sk, 1lib.sk, z-lib.sk).pdf",
        "pages": [55, 59],
        "summary": "Range edges and failed breakouts are the most repeatable reversal contexts; avoid the middle of the range.",
    },
    {
        "topic": "three_push_climax_reversal",
        "book_path": "/Users/jokenrobot/Downloads/Al Brooks 价格行为交易 (反转篇) 黨生版 (阿尔·布鲁克斯Al Brooks) (z-library.sk, 1lib.sk, z-lib.sk).pdf",
        "pages": [21, 70, 107, 112],
        "summary": "Three-push or wedge exhaustion with a strong reversal bar is a valid reversal context, but only with extra evidence because most reversals fail.",
    },
    {
        "topic": "second_entry_trend_continuation",
        "book_path": "/Users/jokenrobot/Downloads/Al Brooks 价格行为交易 (反转篇) 黨生版 (阿尔·布鲁克斯Al Brooks) (z-library.sk, 1lib.sk, z-lib.sk).pdf",
        "pages": [21],
        "summary": "Many first countertrend reversal attempts fail. In a strong trend, the second entry after the first failed pullback is often the cleaner continuation trade.",
    },
    {
        "topic": "reversal_bar_quality",
        "book_path": "/Users/jokenrobot/Downloads/日本蜡烛图交易技术分析 详细解读价格行为模式 (（美）阿尔·布鲁克斯Al Brooks) (z-library.sk, 1lib.sk, z-lib.sk).pdf",
        "pages": [117, 127],
        "summary": "Strong reversal bars and strong entry bars matter more than named patterns; weak bars inside tight ranges are usually noise.",
    },
]

STRATEGY_SPECS: dict[str, dict[str, Any]] = {
    "trend_pullback_continuation": {
        "label": "Trend Pullback Continuation",
        "base_score": 70,
        "reward_r": 2.0,
        "max_hold_bars": 12,
        "description": "Trade with trend after a shallow pullback and a strong reversal bar.",
        "topic": "trend_pullback_continuation",
    },
    "breakout_pullback_resume": {
        "label": "Breakout Pullback Resume",
        "base_score": 76,
        "reward_r": 2.0,
        "max_hold_bars": 10,
        "description": "Trade a test of a strong breakout after the breakout survives the first pullback.",
        "topic": "breakout_pullback_resume",
    },
    "failed_breakout_reversal": {
        "label": "Failed Breakout Reversal",
        "base_score": 82,
        "reward_r": 1.5,
        "max_hold_bars": 8,
        "description": "Fade failed breakouts at range extremes with a strong reversal bar.",
        "topic": "failed_breakout_reversal",
    },
    "three_push_climax_reversal": {
        "label": "Three-Push Climax Reversal",
        "base_score": 68,
        "reward_r": 1.8,
        "max_hold_bars": 8,
        "description": "Fade a three-push exhaustion only when a strong reversal bar confirms.",
        "topic": "three_push_climax_reversal",
    },
    "second_entry_trend_continuation": {
        "label": "Second-Entry Trend Continuation",
        "base_score": 80,
        "reward_r": 2.2,
        "max_hold_bars": 12,
        "description": "Trade the second pullback entry in a strong trend after an earlier continuation attempt fails.",
        "topic": "second_entry_trend_continuation",
    },
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bars_file(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    frame = pd.read_csv(path)
    if columns:
        available = [column for column in columns if column in frame.columns]
        if available:
            return frame.loc[:, available]
    return frame


def normalize_frame(frame: pd.DataFrame, *, symbol: str, asset_class: str) -> pd.DataFrame:
    work = frame.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).copy()
    work["symbol"] = str(symbol).strip().upper()
    work["asset_class"] = str(asset_class).strip().lower()
    work = work.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
    return work


def select_best_symbol_sources(
    *,
    output_root: Path,
    min_bars: int,
    asset_classes: set[str] | None,
    symbols: set[str] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = sorted(
        list(output_root.glob("research/*/bars_used.parquet"))
        + list(output_root.glob("research/*/bars_used.csv"))
    )
    best: dict[str, dict[str, Any]] = {}
    scanned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for path in candidates:
        try:
            frame = read_bars_file(path, columns=["symbol", "asset_class", "ts"])
        except Exception:
            continue
        if frame.empty or "symbol" not in frame.columns or "ts" not in frame.columns:
            continue
        work = frame.copy()
        work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
        work["asset_class"] = work.get("asset_class", "").astype(str).str.strip().str.lower()
        work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
        work = work.dropna(subset=["symbol", "ts"])
        for symbol, group in work.groupby("symbol", sort=True):
            asset_class = str(group["asset_class"].iloc[0] or "").strip().lower()
            row_count = int(len(group))
            start_ts = group["ts"].min()
            end_ts = group["ts"].max()
            record = {
                "symbol": symbol,
                "asset_class": asset_class,
                "rows": row_count,
                "start": fmt_utc(start_ts.to_pydatetime() if pd.notna(start_ts) else None),
                "end": fmt_utc(end_ts.to_pydatetime() if pd.notna(end_ts) else None),
                "source_path": str(path),
                "source_run": str(path.parent.name),
            }
            scanned.append(record)
            if asset_classes and asset_class not in asset_classes:
                skipped.append({**record, "skip_reason": "asset_class_filtered"})
                continue
            if symbols and symbol not in symbols:
                skipped.append({**record, "skip_reason": "symbol_filtered"})
                continue
            key = (
                row_count,
                record["end"] or "",
                record["source_run"],
                Path(record["source_path"]).name,
            )
            previous = best.get(symbol)
            previous_key = (
                int(previous["rows"]),
                str(previous["end"] or ""),
                str(previous["source_run"]),
                Path(str(previous["source_path"])).name,
            ) if previous else None
            if previous is None or key > previous_key:
                best[symbol] = record
    filtered_best: dict[str, dict[str, Any]] = {}
    for symbol, meta in best.items():
        if int(meta["rows"]) < int(min_bars):
            skipped.append({**meta, "skip_reason": f"min_bars<{min_bars}"})
            continue
        filtered_best[symbol] = meta
    return filtered_best, scanned, skipped


def load_market_dataset(
    *,
    output_root: Path,
    min_bars: int,
    asset_classes: set[str] | None,
    symbols: set[str] | None,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected, scanned, skipped = select_best_symbol_sources(
        output_root=output_root,
        min_bars=min_bars,
        asset_classes=asset_classes,
        symbols=symbols,
    )
    by_path: dict[str, list[str]] = {}
    for symbol, meta in selected.items():
        by_path.setdefault(str(meta["source_path"]), []).append(symbol)
    frames: dict[str, pd.DataFrame] = {}
    coverage: list[dict[str, Any]] = []
    for raw_path, wanted_symbols in by_path.items():
        path = Path(raw_path)
        try:
            frame = read_bars_file(path)
        except Exception:
            for symbol in wanted_symbols:
                skipped.append({**selected[symbol], "skip_reason": "read_failed"})
            continue
        if frame.empty:
            for symbol in wanted_symbols:
                skipped.append({**selected[symbol], "skip_reason": "empty_source"})
            continue
        frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
        for symbol in wanted_symbols:
            meta = selected[symbol]
            symbol_frame = frame[frame["symbol"] == symbol].copy()
            if symbol_frame.empty:
                skipped.append({**meta, "skip_reason": "symbol_missing_in_source"})
                continue
            normalized = normalize_frame(
                symbol_frame,
                symbol=symbol,
                asset_class=str(meta["asset_class"]),
            )
            if len(normalized) < int(min_bars):
                skipped.append({**meta, "skip_reason": f"normalized_min_bars<{min_bars}"})
                continue
            frames[symbol] = normalized
            coverage.append(
                {
                    **meta,
                    "rows_loaded": int(len(normalized)),
                    "start_loaded": fmt_utc(normalized["ts"].iloc[0].to_pydatetime()),
                    "end_loaded": fmt_utc(normalized["ts"].iloc[-1].to_pydatetime()),
                }
            )
    return frames, coverage, scanned, skipped


def add_price_action_features(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["prev_close"] = work["close"].shift(1)
    true_range = pd.concat(
        [
            work["high"] - work["low"],
            (work["high"] - work["prev_close"]).abs(),
            (work["low"] - work["prev_close"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
    work["atr14"] = true_range.rolling(14, min_periods=14).mean()
    work["ema20"] = work["close"].ewm(span=20, adjust=False).mean()
    work["ema50"] = work["close"].ewm(span=50, adjust=False).mean()
    work["ema20_slope5"] = work["ema20"] - work["ema20"].shift(5)
    work["range"] = (work["high"] - work["low"]).clip(lower=1e-9)
    work["body"] = work["close"] - work["open"]
    work["body_abs"] = work["body"].abs()
    work["upper_wick"] = work["high"] - work[["open", "close"]].max(axis=1)
    work["lower_wick"] = work[["open", "close"]].min(axis=1) - work["low"]
    work["close_pos"] = ((work["close"] - work["low"]) / work["range"]).clip(lower=0.0, upper=1.0)
    work["bull_bar"] = work["close"] > work["open"]
    work["bear_bar"] = work["close"] < work["open"]
    work["rolling_high20_prev"] = work["high"].rolling(20, min_periods=20).max().shift(1)
    work["rolling_low20_prev"] = work["low"].rolling(20, min_periods=20).min().shift(1)
    work["rolling_high10_prev"] = work["high"].rolling(10, min_periods=10).max().shift(1)
    work["rolling_low10_prev"] = work["low"].rolling(10, min_periods=10).min().shift(1)
    work["range20"] = work["rolling_high20_prev"] - work["rolling_low20_prev"]
    work["bull_count5"] = work["bull_bar"].rolling(5, min_periods=5).sum()
    work["bear_count5"] = work["bear_bar"].rolling(5, min_periods=5).sum()
    atr = work["atr14"].replace(0.0, np.nan)
    work["ema_spread_atr"] = (work["ema20"] - work["ema50"]).abs() / atr
    work["ema20_slope_atr"] = work["ema20_slope5"] / atr
    work["range20_atr"] = work["range20"] / atr
    work["bull_reversal_bar"] = (
        work["bull_bar"]
        & (work["close_pos"] >= 0.62)
        & (work["lower_wick"] >= work["body_abs"] * 0.35)
    )
    work["bear_reversal_bar"] = (
        work["bear_bar"]
        & (work["close_pos"] <= 0.38)
        & (work["upper_wick"] >= work["body_abs"] * 0.35)
    )
    work["breakout_up_bar"] = (
        (work["close"] > work["rolling_high20_prev"])
        & (work["body_abs"] / work["range"] >= 0.55)
        & ((work["range"] / atr) >= 1.15)
    )
    work["breakout_down_bar"] = (
        (work["close"] < work["rolling_low20_prev"])
        & (work["body_abs"] / work["range"] >= 0.55)
        & ((work["range"] / atr) >= 1.15)
    )
    work["trend_up_context"] = (
        (work["close"] > work["ema20"])
        & (work["ema20"] > work["ema50"])
        & (work["ema20_slope_atr"] > 0.20)
        & (work["bull_count5"] >= 3)
    )
    work["trend_down_context"] = (
        (work["close"] < work["ema20"])
        & (work["ema20"] < work["ema50"])
        & (work["ema20_slope_atr"] < -0.20)
        & (work["bear_count5"] >= 3)
    )
    work["range_context"] = (
        (work["ema_spread_atr"] <= 0.45)
        & (work["ema20_slope_atr"].abs() <= 0.25)
        & (work["range20_atr"] <= 6.0)
    )
    return work


def allow_short(asset_class: str) -> bool:
    return str(asset_class).strip().lower() not in LONG_ONLY_ASSET_CLASSES


def count_pullback_bars(frame: pd.DataFrame, idx: int, *, direction: str) -> int:
    start = max(1, idx - 3)
    window = frame.iloc[start : idx + 1]
    if direction == "LONG":
        return int((window["close"].diff() < 0).sum())
    return int((window["close"].diff() > 0).sum())


def recent_breakout_index(frame: pd.DataFrame, idx: int, *, direction: str, lookback: int = 4) -> int | None:
    if idx <= 0:
        return None
    start = max(0, idx - lookback)
    column = "breakout_up_bar" if direction == "LONG" else "breakout_down_bar"
    recent = frame.iloc[start:idx]
    hits = recent.index[recent[column].astype(bool)].tolist()
    if not hits:
        return None
    return int(hits[-1])


def has_prior_test(frame: pd.DataFrame, idx: int, *, direction: str, tolerance_atr: float) -> bool:
    start = max(5, idx - 12)
    atr = float(frame.iloc[idx]["atr14"] or 0.0)
    if not math.isfinite(atr) or atr <= 0.0:
        return False
    tolerance = tolerance_atr * atr
    if direction == "LONG":
        level = float(frame.iloc[idx]["rolling_low20_prev"] or np.nan)
        if not math.isfinite(level):
            return False
        lows = frame.iloc[start:idx]["low"].to_numpy(dtype=float)
        return bool(np.any(np.abs(lows - level) <= tolerance))
    level = float(frame.iloc[idx]["rolling_high20_prev"] or np.nan)
    if not math.isfinite(level):
        return False
    highs = frame.iloc[start:idx]["high"].to_numpy(dtype=float)
    return bool(np.any(np.abs(highs - level) <= tolerance))


def has_three_push_exhaustion(frame: pd.DataFrame, idx: int, *, direction: str, lookback: int = 14) -> bool:
    if idx < 4:
        return False
    start = max(2, idx - lookback)
    pivots: list[float] = []
    if direction == "SHORT":
        highs = frame["high"].to_numpy(dtype=float)
        for probe in range(start, idx):
            if probe - 1 < 1:
                continue
            left = highs[probe - 2]
            mid = highs[probe - 1]
            right = highs[probe]
            if math.isfinite(left) and math.isfinite(mid) and math.isfinite(right) and mid > left and mid >= right:
                pivots.append(float(mid))
        if len(pivots) < 3:
            return False
        last = pivots[-3:]
        return bool(last[0] < last[1] < last[2])
    lows = frame["low"].to_numpy(dtype=float)
    for probe in range(start, idx):
        if probe - 1 < 1:
            continue
        left = lows[probe - 2]
        mid = lows[probe - 1]
        right = lows[probe]
        if math.isfinite(left) and math.isfinite(mid) and math.isfinite(right) and mid < left and mid <= right:
            pivots.append(float(mid))
    if len(pivots) < 3:
        return False
    last = pivots[-3:]
    return bool(last[0] > last[1] > last[2])


def has_second_entry_setup(frame: pd.DataFrame, idx: int, *, direction: str, lookback: int = 6) -> bool:
    if idx < 4:
        return False
    row = frame.iloc[idx]
    atr = float(row["atr14"] or 0.0)
    if not math.isfinite(atr) or atr <= 0.0:
        return False
    start = max(3, idx - lookback)
    window = frame.iloc[start:idx].copy()
    if window.empty:
        return False
    if direction == "LONG":
        attempts = window[window["bull_reversal_bar"]]
        if attempts.empty:
            return False
        prior = attempts.iloc[-1]
        return bool(
            (len(attempts) >= 2 or count_pullback_bars(frame, idx, direction="LONG") >= 2)
            and float(row["low"]) >= float(prior["low"]) - (0.25 * atr)
            and float(row["high"]) >= float(prior["high"])
        )
    attempts = window[window["bear_reversal_bar"]]
    if attempts.empty:
        return False
    prior = attempts.iloc[-1]
    return bool(
        (len(attempts) >= 2 or count_pullback_bars(frame, idx, direction="SHORT") >= 2)
        and float(row["high"]) <= float(prior["high"]) + (0.25 * atr)
        and float(row["low"]) <= float(prior["low"])
    )


def split_dataframe_folds(frame: pd.DataFrame, *, parts: int = 3) -> list[pd.DataFrame]:
    if frame.empty or parts <= 1:
        return [frame.copy()]
    length = len(frame)
    base = length // parts
    remainder = length % parts
    start = 0
    folds: list[pd.DataFrame] = []
    for idx in range(parts):
        stop = start + base + (1 if idx < remainder else 0)
        folds.append(frame.iloc[start:stop].copy())
        start = stop
    return folds


def route_selection_score(metrics: dict[str, Any]) -> float:
    trade_count = float(metrics.get("trade_count", 0))
    expectancy = float(metrics.get("expectancy_r", 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0))
    positive_symbol_ratio = float(metrics.get("positive_symbol_ratio", 0.0))
    positive_fold_ratio = float(metrics.get("positive_fold_ratio", 0.0))
    sample_score = min(18.0, trade_count * 0.30)
    return float(
        (expectancy * 120.0)
        + max(0.0, (profit_factor - 1.0) * 25.0)
        + (positive_symbol_ratio * 22.0)
        + (positive_fold_ratio * 12.0)
        + sample_score
    )


def build_signal(
    *,
    frame: pd.DataFrame,
    idx: int,
    strategy_id: str,
    direction: str,
    score: int,
    note: str,
) -> dict[str, Any]:
    row = frame.iloc[idx]
    return {
        "strategy_id": strategy_id,
        "strategy_label": STRATEGY_SPECS[strategy_id]["label"],
        "signal_index": int(idx),
        "signal_ts": fmt_utc(row["ts"].to_pydatetime() if isinstance(row["ts"], pd.Timestamp) else None),
        "symbol": str(row["symbol"]).strip().upper(),
        "asset_class": str(row["asset_class"]).strip().lower(),
        "direction": direction,
        "score": int(score),
        "note": note,
        "signal_high": float(row["high"]),
        "signal_low": float(row["low"]),
        "signal_close": float(row["close"]),
        "atr14": float(row["atr14"]),
        "reward_r": float(STRATEGY_SPECS[strategy_id]["reward_r"]),
        "max_hold_bars": int(STRATEGY_SPECS[strategy_id]["max_hold_bars"]),
    }


def generate_symbol_signals(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    signals: dict[str, list[dict[str, Any]]] = {key: [] for key in STRATEGY_SPECS}
    asset_class = str(frame["asset_class"].iloc[0]).strip().lower()
    short_allowed = allow_short(asset_class)
    for idx in range(55, len(frame) - 1):
        row = frame.iloc[idx]
        atr = float(row["atr14"] or 0.0)
        if not math.isfinite(atr) or atr <= 0.0:
            continue
        if not math.isfinite(float(row["rolling_high20_prev"] or np.nan)):
            continue
        bull_reversal = bool(row["bull_reversal_bar"])
        bear_reversal = bool(row["bear_reversal_bar"])
        trend_up = bool(row["trend_up_context"])
        trend_down = bool(row["trend_down_context"])
        range_context = bool(row["range_context"])

        pullback_long = count_pullback_bars(frame, idx, direction="LONG") >= 2 or float(row["low"]) <= float(row["ema20"]) + 0.15 * atr
        pullback_short = count_pullback_bars(frame, idx, direction="SHORT") >= 2 or float(row["high"]) >= float(row["ema20"]) - 0.15 * atr

        if trend_up and bull_reversal and pullback_long and float(row["low"]) > float(row["rolling_low20_prev"]) - 0.35 * atr:
            score = int(STRATEGY_SPECS["trend_pullback_continuation"]["base_score"] + min(12.0, float(row["body_abs"] / row["range"]) * 10.0))
            signals["trend_pullback_continuation"].append(
                build_signal(
                    frame=frame,
                    idx=idx,
                    strategy_id="trend_pullback_continuation",
                    direction="LONG",
                    score=score,
                    note="strong trend, pullback, bullish reversal bar",
                )
            )
        if short_allowed and trend_down and bear_reversal and pullback_short and float(row["high"]) < float(row["rolling_high20_prev"]) + 0.35 * atr:
            score = int(STRATEGY_SPECS["trend_pullback_continuation"]["base_score"] + min(12.0, float(row["body_abs"] / row["range"]) * 10.0))
            signals["trend_pullback_continuation"].append(
                build_signal(
                    frame=frame,
                    idx=idx,
                    strategy_id="trend_pullback_continuation",
                    direction="SHORT",
                    score=score,
                    note="strong trend, pullback, bearish reversal bar",
                )
            )

        if trend_up and bull_reversal and has_second_entry_setup(frame, idx, direction="LONG"):
            score = int(STRATEGY_SPECS["second_entry_trend_continuation"]["base_score"] + min(10.0, float(row["body_abs"] / row["range"]) * 8.0))
            signals["second_entry_trend_continuation"].append(
                build_signal(
                    frame=frame,
                    idx=idx,
                    strategy_id="second_entry_trend_continuation",
                    direction="LONG",
                    score=score,
                    note="strong trend, first pullback attempt failed, second-entry continuation",
                )
            )
        if short_allowed and trend_down and bear_reversal and has_second_entry_setup(frame, idx, direction="SHORT"):
            score = int(STRATEGY_SPECS["second_entry_trend_continuation"]["base_score"] + min(10.0, float(row["body_abs"] / row["range"]) * 8.0))
            signals["second_entry_trend_continuation"].append(
                build_signal(
                    frame=frame,
                    idx=idx,
                    strategy_id="second_entry_trend_continuation",
                    direction="SHORT",
                    score=score,
                    note="strong trend, first pullback attempt failed, second-entry continuation",
                )
            )

        breakout_up_idx = recent_breakout_index(frame, idx, direction="LONG")
        if breakout_up_idx is not None and bull_reversal:
            breakout_row = frame.iloc[breakout_up_idx]
            breakout_level = float(breakout_row["rolling_high20_prev"])
            shallow = float(row["low"]) >= breakout_level - 0.30 * atr
            if shallow and float(row["close"]) >= breakout_level:
                score = int(STRATEGY_SPECS["breakout_pullback_resume"]["base_score"] + min(10.0, float(breakout_row["range"] / breakout_row["atr14"]) * 4.0))
                signals["breakout_pullback_resume"].append(
                    build_signal(
                        frame=frame,
                        idx=idx,
                        strategy_id="breakout_pullback_resume",
                        direction="LONG",
                        score=score,
                        note="recent bullish breakout survived first pullback",
                    )
                )
        breakout_down_idx = recent_breakout_index(frame, idx, direction="SHORT")
        if short_allowed and breakout_down_idx is not None and bear_reversal:
            breakout_row = frame.iloc[breakout_down_idx]
            breakout_level = float(breakout_row["rolling_low20_prev"])
            shallow = float(row["high"]) <= breakout_level + 0.30 * atr
            if shallow and float(row["close"]) <= breakout_level:
                score = int(STRATEGY_SPECS["breakout_pullback_resume"]["base_score"] + min(10.0, float(breakout_row["range"] / breakout_row["atr14"]) * 4.0))
                signals["breakout_pullback_resume"].append(
                    build_signal(
                        frame=frame,
                        idx=idx,
                        strategy_id="breakout_pullback_resume",
                        direction="SHORT",
                        score=score,
                        note="recent bearish breakout survived first pullback",
                    )
                )

        if range_context and bull_reversal:
            failed_low = float(row["low"]) < float(row["rolling_low20_prev"]) - 0.12 * atr
            close_back_in = float(row["close"]) > float(row["rolling_low20_prev"])
            if failed_low and close_back_in and has_prior_test(frame, idx, direction="LONG", tolerance_atr=0.40):
                score = int(STRATEGY_SPECS["failed_breakout_reversal"]["base_score"] + 8)
                signals["failed_breakout_reversal"].append(
                    build_signal(
                        frame=frame,
                        idx=idx,
                        strategy_id="failed_breakout_reversal",
                        direction="LONG",
                        score=score,
                        note="failed downside breakout at range edge",
                    )
                )
        if short_allowed and range_context and bear_reversal:
            failed_high = float(row["high"]) > float(row["rolling_high20_prev"]) + 0.12 * atr
            close_back_in = float(row["close"]) < float(row["rolling_high20_prev"])
            if failed_high and close_back_in and has_prior_test(frame, idx, direction="SHORT", tolerance_atr=0.40):
                score = int(STRATEGY_SPECS["failed_breakout_reversal"]["base_score"] + 8)
                signals["failed_breakout_reversal"].append(
                    build_signal(
                        frame=frame,
                        idx=idx,
                        strategy_id="failed_breakout_reversal",
                        direction="SHORT",
                        score=score,
                        note="failed upside breakout at range edge",
                    )
                )

        if trend_up and bear_reversal and has_three_push_exhaustion(frame, idx, direction="SHORT"):
            score = int(STRATEGY_SPECS["three_push_climax_reversal"]["base_score"] + 6)
            signals["three_push_climax_reversal"].append(
                build_signal(
                    frame=frame,
                    idx=idx,
                    strategy_id="three_push_climax_reversal",
                    direction="SHORT",
                    score=score,
                    note="three-push bullish exhaustion with bearish reversal bar",
                )
            )
        if short_allowed and trend_down and bull_reversal and has_three_push_exhaustion(frame, idx, direction="LONG"):
            score = int(STRATEGY_SPECS["three_push_climax_reversal"]["base_score"] + 6)
            signals["three_push_climax_reversal"].append(
                build_signal(
                    frame=frame,
                    idx=idx,
                    strategy_id="three_push_climax_reversal",
                    direction="LONG",
                    score=score,
                    note="three-push bearish exhaustion with bullish reversal bar",
                )
            )
    return signals


def simulate_signals(
    frame: pd.DataFrame,
    signals: list[dict[str, Any]],
    *,
    reward_r: float | None,
    max_hold_bars: int | None,
) -> list[dict[str, Any]]:
    if not signals:
        return []
    work = frame.reset_index(drop=True).copy()
    trades: list[dict[str, Any]] = []
    next_eligible_index = 0
    for signal in sorted(signals, key=lambda row: (int(row["signal_index"]), -int(row["score"]))):
        signal_index = int(signal["signal_index"])
        if signal_index < next_eligible_index or signal_index + 1 >= len(work):
            continue
        entry_idx = signal_index + 1
        signal_high = float(signal["signal_high"])
        signal_low = float(signal["signal_low"])
        atr = float(signal["atr14"] or 0.0)
        effective_reward_r = float(signal.get("reward_r") if reward_r is None else reward_r)
        effective_max_hold_bars = int(signal.get("max_hold_bars") if max_hold_bars is None else max_hold_bars)
        if atr <= 0.0 or not math.isfinite(atr):
            continue
        bar = work.iloc[entry_idx]
        direction = str(signal["direction"]).strip().upper()
        if direction == "LONG":
            if float(bar["high"]) <= signal_high:
                continue
            entry_price = max(float(bar["open"]), signal_high)
            stop_price = signal_low - (0.05 * atr)
            risk = entry_price - stop_price
            if risk <= 0.0 or not math.isfinite(risk):
                continue
            target_price = entry_price + (risk * effective_reward_r)
        else:
            if float(bar["low"]) >= signal_low:
                continue
            entry_price = min(float(bar["open"]), signal_low)
            stop_price = signal_high + (0.05 * atr)
            risk = stop_price - entry_price
            if risk <= 0.0 or not math.isfinite(risk):
                continue
            target_price = entry_price - (risk * effective_reward_r)

        exit_price = float(work.iloc[min(len(work) - 1, entry_idx + effective_max_hold_bars - 1)]["close"])
        exit_idx = min(len(work) - 1, entry_idx + effective_max_hold_bars - 1)
        exit_reason = "time_stop"

        for probe in range(entry_idx, min(len(work), entry_idx + effective_max_hold_bars)):
            probe_bar = work.iloc[probe]
            high = float(probe_bar["high"])
            low = float(probe_bar["low"])
            if direction == "LONG":
                hit_stop = low <= stop_price
                hit_target = high >= target_price
                if hit_stop and hit_target:
                    exit_price = stop_price
                    exit_idx = probe
                    exit_reason = "stop_first_same_bar"
                    break
                if hit_stop:
                    exit_price = stop_price
                    exit_idx = probe
                    exit_reason = "stop"
                    break
                if hit_target:
                    exit_price = target_price
                    exit_idx = probe
                    exit_reason = "target"
                    break
            else:
                hit_stop = high >= stop_price
                hit_target = low <= target_price
                if hit_stop and hit_target:
                    exit_price = stop_price
                    exit_idx = probe
                    exit_reason = "stop_first_same_bar"
                    break
                if hit_stop:
                    exit_price = stop_price
                    exit_idx = probe
                    exit_reason = "stop"
                    break
                if hit_target:
                    exit_price = target_price
                    exit_idx = probe
                    exit_reason = "target"
                    break
        next_eligible_index = exit_idx + 1
        if direction == "LONG":
            pct_return = (exit_price / entry_price) - 1.0
            r_multiple = (exit_price - entry_price) / risk
        else:
            pct_return = (entry_price / exit_price) - 1.0
            r_multiple = (entry_price - exit_price) / risk
        exit_ts = work.iloc[exit_idx]["ts"]
        trades.append(
            {
                **signal,
                "entry_index": int(entry_idx),
                "entry_ts": fmt_utc(work.iloc[entry_idx]["ts"].to_pydatetime() if isinstance(work.iloc[entry_idx]["ts"], pd.Timestamp) else None),
                "entry_price": float(entry_price),
                "stop_price": float(stop_price),
                "target_price": float(target_price),
                "exit_index": int(exit_idx),
                "exit_ts": fmt_utc(exit_ts.to_pydatetime() if isinstance(exit_ts, pd.Timestamp) else None),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "bars_held": int(exit_idx - entry_idx + 1),
                "risk_r": float(risk),
                "pct_return": float(pct_return),
                "r_multiple": float(r_multiple),
                "reward_r_used": float(effective_reward_r),
                "max_hold_bars_used": int(effective_max_hold_bars),
            }
        )
    return trades


def metrics_from_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "avg_return_pct": 0.0,
            "median_r": 0.0,
            "profit_factor": 0.0,
            "total_r": 0.0,
            "max_drawdown_r": 0.0,
            "avg_bars_held": 0.0,
            "positive_symbol_ratio": 0.0,
            "positive_fold_ratio": 0.0,
        }
    frame = pd.DataFrame(trades).copy()
    returns = frame["r_multiple"].astype(float)
    wins = returns > 0.0
    gross_profit = float(returns[wins].sum())
    gross_loss = float(-returns[returns < 0.0].sum())
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0.0 else float(gross_profit > 0.0) * 99.0
    equity = returns.cumsum()
    max_dd = float((equity - equity.cummax()).min()) if not equity.empty else 0.0
    by_symbol = frame.groupby("symbol")["r_multiple"].sum()
    positive_symbol_ratio = float((by_symbol > 0.0).mean()) if not by_symbol.empty else 0.0
    frame["entry_ts"] = pd.to_datetime(frame["entry_ts"], utc=True)
    positive_fold_ratio = 0.0
    if len(frame) >= 6 and frame["entry_ts"].notna().any():
        ordered = frame.sort_values("entry_ts").reset_index(drop=True)
        folds = split_dataframe_folds(ordered, parts=3)
        fold_scores = [float(chunk["r_multiple"].sum()) > 0.0 for chunk in folds if not chunk.empty]
        positive_fold_ratio = float(np.mean(fold_scores)) if fold_scores else 0.0
    return {
        "trade_count": int(len(frame)),
        "win_rate": float(wins.mean()),
        "expectancy_r": float(returns.mean()),
        "avg_return_pct": float(frame["pct_return"].mean()),
        "median_r": float(returns.median()),
        "profit_factor": profit_factor,
        "total_r": float(returns.sum()),
        "max_drawdown_r": abs(max_dd),
        "avg_bars_held": float(frame["bars_held"].mean()),
        "positive_symbol_ratio": positive_symbol_ratio,
        "positive_fold_ratio": positive_fold_ratio,
    }


def asset_class_breakdown(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trades:
        return []
    frame = pd.DataFrame(trades).copy()
    rows: list[dict[str, Any]] = []
    for asset_class, chunk in frame.groupby("asset_class", sort=True):
        metrics = metrics_from_trades(chunk.to_dict("records"))
        rows.append({"asset_class": str(asset_class), **metrics})
    rows.sort(key=lambda row: (-float(row["expectancy_r"]), row["asset_class"]))
    return rows


def top_symbol_breakdown(trades: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    if not trades:
        return []
    frame = pd.DataFrame(trades).copy()
    rows: list[dict[str, Any]] = []
    for symbol, chunk in frame.groupby("symbol", sort=True):
        metrics = metrics_from_trades(chunk.to_dict("records"))
        rows.append(
            {
                "symbol": str(symbol),
                "asset_class": str(chunk["asset_class"].iloc[0]),
                **metrics,
            }
        )
    rows.sort(key=lambda row: (-float(row["total_r"]), -float(row["expectancy_r"]), row["symbol"]))
    return rows[:limit]


def effectiveness_status(metrics: dict[str, Any]) -> str:
    if int(metrics.get("trade_count", 0)) < 12:
        return "insufficient_sample"
    expectancy = float(metrics.get("expectancy_r", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    positive_symbols = float(metrics.get("positive_symbol_ratio", 0.0))
    positive_folds = float(metrics.get("positive_fold_ratio", 0.0))
    if expectancy > 0.15 and pf > 1.20 and positive_symbols >= 0.55 and positive_folds >= 0.67:
        return "effective"
    if expectancy > 0.0 and pf > 1.0 and positive_symbols >= 0.45:
        return "mixed_positive"
    return "ineffective"


def build_strategy_payload(
    *,
    strategy_id: str,
    strategy_signals: list[dict[str, Any]],
    strategy_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = metrics_from_trades(strategy_trades)
    status = effectiveness_status(metrics)
    spec = STRATEGY_SPECS[strategy_id]
    return {
        "strategy_id": strategy_id,
        "label": spec["label"],
        "description": spec["description"],
        "book_topic": spec["topic"],
        "reward_r": float(spec["reward_r"]),
        "max_hold_bars": int(spec["max_hold_bars"]),
        "signal_count": int(len(strategy_signals)),
        "effectiveness_status": status,
        "metrics": metrics,
        "asset_class_breakdown": asset_class_breakdown(strategy_trades),
        "top_symbols": top_symbol_breakdown(strategy_trades),
        "sample_trades": strategy_trades[:10],
    }


def build_composite_signals(symbol_signals: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for strategy_rows in symbol_signals.values():
        merged.extend(strategy_rows)
    merged.sort(key=lambda row: (int(row["signal_index"]), -int(row["score"]), row["strategy_id"]))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for row in merged:
        key = (int(row["signal_index"]), str(row["direction"]), str(row["symbol"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def trade_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    frame = pd.DataFrame(trades).copy()
    frame["entry_ts"] = pd.to_datetime(frame["entry_ts"], utc=True, errors="coerce")
    return frame.dropna(subset=["entry_ts"]).copy()


def build_adaptive_route_selection(
    strategy_trade_map: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    combined = trade_frame(
        [
            trade
            for trades in strategy_trade_map.values()
            for trade in trades
        ]
    )
    if combined.empty:
        return {
            "selection_mode": "train_first_two_folds_trade_count_split",
            "selection_cutoff_ts": None,
            "selection_rows": [],
            "selected_routes_by_asset_class": {},
            "selected_route_count": 0,
            "selected_asset_class_count": 0,
        }
    ordered = combined.sort_values("entry_ts").reset_index(drop=True)
    cutoff_index = max(0, int(math.ceil(len(ordered) * (2.0 / 3.0))) - 1)
    selection_cutoff_ts = ordered.iloc[cutoff_index]["entry_ts"]
    rows: list[dict[str, Any]] = []
    for strategy_id, trades in strategy_trade_map.items():
        frame = trade_frame(trades)
        if frame.empty:
            continue
        train = frame[frame["entry_ts"] <= selection_cutoff_ts].copy()
        if train.empty:
            continue
        for asset_class, chunk in train.groupby("asset_class", sort=True):
            metrics = metrics_from_trades(chunk.to_dict("records"))
            selection_score = route_selection_score(metrics)
            selected = bool(
                int(metrics.get("trade_count", 0)) >= 12
                and float(metrics.get("expectancy_r", 0.0)) > 0.0
                and float(metrics.get("profit_factor", 0.0)) > 1.0
                and float(metrics.get("positive_symbol_ratio", 0.0)) >= 0.40
            )
            rows.append(
                {
                    "asset_class": str(asset_class),
                    "strategy_id": strategy_id,
                    "selection_score": float(selection_score),
                    "selected": selected,
                    "metrics": metrics,
                }
            )
    rows.sort(
        key=lambda row: (
            row["asset_class"],
            not bool(row["selected"]),
            -float(row["selection_score"]),
            row["strategy_id"],
        )
    )
    selected_routes_by_asset_class: dict[str, list[str]] = {}
    for asset_class in sorted({str(row["asset_class"]) for row in rows}):
        selected_rows = [
            row for row in rows
            if row["asset_class"] == asset_class and row["selected"]
        ]
        if selected_rows:
            selected_routes_by_asset_class[asset_class] = [
                str(row["strategy_id"]) for row in selected_rows[:2]
            ]
    return {
        "selection_mode": "train_first_two_folds_trade_count_split",
        "selection_cutoff_ts": fmt_utc(selection_cutoff_ts.to_pydatetime()),
        "selection_rows": rows,
        "selected_routes_by_asset_class": selected_routes_by_asset_class,
        "selected_route_count": int(sum(len(rows) for rows in selected_routes_by_asset_class.values())),
        "selected_asset_class_count": int(len(selected_routes_by_asset_class)),
    }


def build_adaptive_route_payload(
    *,
    feature_frames: dict[str, pd.DataFrame],
    symbol_signal_store: dict[str, dict[str, list[dict[str, Any]]]],
    selection_payload: dict[str, Any],
) -> dict[str, Any]:
    selected_routes_by_asset_class = {
        str(asset_class): [str(item) for item in strategy_ids]
        for asset_class, strategy_ids in dict(selection_payload.get("selected_routes_by_asset_class") or {}).items()
    }
    adaptive_trades: list[dict[str, Any]] = []
    for symbol, frame in sorted(feature_frames.items()):
        asset_class = str(frame["asset_class"].iloc[0]).strip().lower()
        selected_ids = selected_routes_by_asset_class.get(asset_class, [])
        if not selected_ids:
            continue
        signal_map = {
            strategy_id: symbol_signal_store.get(symbol, {}).get(strategy_id, [])
            for strategy_id in selected_ids
        }
        route_signals = build_composite_signals(signal_map)
        adaptive_trades.extend(
            simulate_signals(
                frame,
                route_signals,
                reward_r=None,
                max_hold_bars=None,
            )
        )
    all_metrics = metrics_from_trades(adaptive_trades)
    cutoff_ts_raw = selection_payload.get("selection_cutoff_ts")
    cutoff_ts = pd.to_datetime(cutoff_ts_raw, utc=True, errors="coerce") if cutoff_ts_raw else pd.NaT
    adaptive_frame = trade_frame(adaptive_trades)
    if adaptive_frame.empty:
        oos_frame = adaptive_frame.copy()
    elif pd.notna(cutoff_ts):
        oos_frame = adaptive_frame[adaptive_frame["entry_ts"] > cutoff_ts].copy()
    else:
        oos_frame = adaptive_frame.copy()
    oos_trades = oos_frame.to_dict("records") if not oos_frame.empty else []
    oos_metrics = metrics_from_trades(oos_trades)
    return {
        "label": "Brooks Adaptive Route",
        "selection_mode": selection_payload.get("selection_mode"),
        "selection_cutoff_ts": cutoff_ts_raw,
        "selection_rows": list(selection_payload.get("selection_rows") or [])[:40],
        "selected_routes_by_asset_class": selected_routes_by_asset_class,
        "selected_route_count": int(selection_payload.get("selected_route_count", 0)),
        "selected_asset_class_count": int(selection_payload.get("selected_asset_class_count", 0)),
        "effectiveness_status": effectiveness_status(all_metrics),
        "metrics": all_metrics,
        "asset_class_breakdown": asset_class_breakdown(adaptive_trades),
        "top_symbols": top_symbol_breakdown(adaptive_trades),
        "out_of_sample_effectiveness_status": effectiveness_status(oos_metrics),
        "out_of_sample_metrics": oos_metrics,
        "out_of_sample_asset_class_breakdown": asset_class_breakdown(oos_trades),
        "out_of_sample_top_symbols": top_symbol_breakdown(oos_trades),
        "sample_trades": adaptive_trades[:12],
    }


def format_selected_routes_by_asset_class(routes: dict[str, list[str]] | None) -> str:
    if not routes:
        return "-"
    chunks: list[str] = []
    for asset_class, strategy_ids in sorted(routes.items()):
        chunks.append(f"{asset_class}:{'/'.join(str(item) for item in strategy_ids)}")
    return ", ".join(chunks)


def summarize_coverage(
    *,
    coverage: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    by_asset: dict[str, int] = {}
    for row in coverage:
        asset = str(row.get("asset_class") or "unknown")
        by_asset[asset] = int(by_asset.get(asset, 0)) + 1
    skipped_by_reason: dict[str, int] = {}
    for row in skipped:
        reason = str(row.get("skip_reason") or "unknown")
        skipped_by_reason[reason] = int(skipped_by_reason.get(reason, 0)) + 1
    return {
        "included_symbol_count": int(len(coverage)),
        "coverage_by_asset_class": dict(sorted(by_asset.items())),
        "skipped_count": int(len(skipped)),
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "included_symbols": [str(row["symbol"]) for row in sorted(coverage, key=lambda row: row["symbol"])],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    coverage = payload.get("coverage_summary", {})
    lines = [
        "# Brooks Price Action All-Market Study",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- study_status: `{payload.get('study_status') or ''}`",
        f"- data_source_mode: `{payload.get('data_source_mode') or ''}`",
        f"- included_symbols: `{coverage.get('included_symbol_count', 0)}`",
        f"- asset_classes: `{', '.join(f'{k}:{v}' for k, v in coverage.get('coverage_by_asset_class', {}).items()) or '-'}`",
        f"- skipped: `{coverage.get('skipped_count', 0)}`",
        f"- skipped_by_reason: `{', '.join(f'{k}:{v}' for k, v in coverage.get('skipped_by_reason', {}).items()) or '-'}`",
        "",
        "## Book Priors",
    ]
    for row in payload.get("book_foundations", []):
        lines.append(
            f"- topic=`{row.get('topic')}` pages=`{','.join(str(x) for x in row.get('pages', []))}` book=`{Path(str(row.get('book_path') or '')).name}`"
        )
        lines.append(f"  - prior: {row.get('summary')}")
    lines.extend(["", "## Strategy Modules"])
    for row in payload.get("strategy_modules", []):
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics", {})
        lines.append(
            f"- `{row.get('strategy_id')}` status=`{row.get('effectiveness_status')}` trades=`{metrics.get('trade_count')}` expectancy_r=`{float(metrics.get('expectancy_r') or 0.0):.4f}` win=`{float(metrics.get('win_rate') or 0.0):.2%}` pf=`{float(metrics.get('profit_factor') or 0.0):.2f}`"
        )
        top_symbols = row.get("top_symbols", [])
        if top_symbols:
            top_text = ", ".join(
                f"{item.get('symbol')}:{float(item.get('total_r') or 0.0):.2f}R"
                for item in top_symbols[:5]
            )
            lines.append(f"  - top_symbols: `{top_text}`")
    composite = payload.get("composite_strategy", {})
    metrics = composite.get("metrics", {})
    adaptive = payload.get("adaptive_route_strategy", {})
    adaptive_metrics = adaptive.get("metrics", {})
    adaptive_oos = adaptive.get("out_of_sample_metrics", {})
    lines.extend(
        [
            "",
            "## Composite",
            f"- status: `{composite.get('effectiveness_status') or ''}`",
            f"- trades: `{metrics.get('trade_count', 0)}`",
            f"- expectancy_r: `{float(metrics.get('expectancy_r') or 0.0):.4f}`",
            f"- win_rate: `{float(metrics.get('win_rate') or 0.0):.2%}`",
            f"- profit_factor: `{float(metrics.get('profit_factor') or 0.0):.2f}`",
            f"- positive_symbol_ratio: `{float(metrics.get('positive_symbol_ratio') or 0.0):.2%}`",
            f"- positive_fold_ratio: `{float(metrics.get('positive_fold_ratio') or 0.0):.2%}`",
            "",
            "## Adaptive Route",
            f"- status: `{adaptive.get('effectiveness_status') or ''}`",
            f"- selection_cutoff_ts: `{adaptive.get('selection_cutoff_ts') or ''}`",
            f"- selected_routes_by_asset_class: `{format_selected_routes_by_asset_class(adaptive.get('selected_routes_by_asset_class'))}`",
            f"- trades: `{adaptive_metrics.get('trade_count', 0)}`",
            f"- expectancy_r: `{float(adaptive_metrics.get('expectancy_r') or 0.0):.4f}`",
            f"- profit_factor: `{float(adaptive_metrics.get('profit_factor') or 0.0):.2f}`",
            f"- oos_status: `{adaptive.get('out_of_sample_effectiveness_status') or ''}`",
            f"- oos_trades: `{adaptive_oos.get('trade_count', 0)}`",
            f"- oos_expectancy_r: `{float(adaptive_oos.get('expectancy_r') or 0.0):.4f}`",
            f"- oos_profit_factor: `{float(adaptive_oos.get('profit_factor') or 0.0):.2f}`",
            "",
            "## Conclusions",
        ]
    )
    for line in payload.get("conclusions", []):
        lines.append(f"- {line}")
    return "\n".join(lines).strip() + "\n"


def prune_review_artifacts(
    review_dir: Path,
    *,
    stem: str,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
    now_dt: dt.datetime | None = None,
) -> tuple[list[str], list[str]]:
    effective_now = now_dt or now_utc()
    cutoff = effective_now - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob(f"*_{stem}*"), key=lambda path: path.stat().st_mtime, reverse=True)

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Brooks-style price action strategy study across all locally available market bars."
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--min-bars", type=int, default=120)
    parser.add_argument("--asset-classes", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    review_dir = Path(args.review_dir).expanduser().resolve()
    runtime_now = parse_now(args.now)
    asset_classes = {
        str(item).strip().lower()
        for item in str(args.asset_classes or "").split(",")
        if str(item).strip()
    } or None
    symbols = {
        str(item).strip().upper()
        for item in str(args.symbols or "").split(",")
        if str(item).strip()
    } or None

    frames, coverage, scanned, skipped = load_market_dataset(
        output_root=output_root,
        min_bars=max(60, int(args.min_bars)),
        asset_classes=asset_classes,
        symbols=symbols,
    )
    if not frames:
        raise SystemExit("no_local_market_frames")

    strategy_signal_map: dict[str, list[dict[str, Any]]] = {key: [] for key in STRATEGY_SPECS}
    strategy_trade_map: dict[str, list[dict[str, Any]]] = {key: [] for key in STRATEGY_SPECS}
    feature_frames: dict[str, pd.DataFrame] = {}
    symbol_signal_store: dict[str, dict[str, list[dict[str, Any]]]] = {}
    composite_trades: list[dict[str, Any]] = []

    for symbol, raw_frame in sorted(frames.items()):
        frame = add_price_action_features(raw_frame)
        feature_frames[symbol] = frame
        symbol_signals = generate_symbol_signals(frame)
        symbol_signal_store[symbol] = symbol_signals
        for strategy_id, rows in symbol_signals.items():
            strategy_signal_map[strategy_id].extend(rows)
            trades = simulate_signals(
                frame,
                rows,
                reward_r=None,
                max_hold_bars=None,
            )
            strategy_trade_map[strategy_id].extend(trades)
        composite_signal_rows = build_composite_signals(symbol_signals)
        composite_trades.extend(
            simulate_signals(
                frame,
                composite_signal_rows,
                reward_r=None,
                max_hold_bars=None,
            )
        )

    strategy_modules = [
        build_strategy_payload(
            strategy_id=strategy_id,
            strategy_signals=strategy_signal_map[strategy_id],
            strategy_trades=strategy_trade_map[strategy_id],
        )
        for strategy_id in STRATEGY_SPECS
    ]
    composite_metrics = metrics_from_trades(composite_trades)
    composite_payload = {
        "label": "Brooks Composite",
        "effectiveness_status": effectiveness_status(composite_metrics),
        "metrics": composite_metrics,
        "asset_class_breakdown": asset_class_breakdown(composite_trades),
        "top_symbols": top_symbol_breakdown(composite_trades),
        "sample_trades": composite_trades[:12],
    }
    adaptive_selection = build_adaptive_route_selection(strategy_trade_map)
    adaptive_route_payload = build_adaptive_route_payload(
        feature_frames=feature_frames,
        symbol_signal_store=symbol_signal_store,
        selection_payload=adaptive_selection,
    )

    effective_modules = [row["strategy_id"] for row in strategy_modules if row["effectiveness_status"] == "effective"]
    mixed_modules = [row["strategy_id"] for row in strategy_modules if row["effectiveness_status"] == "mixed_positive"]
    adaptive_selected = adaptive_route_payload.get("selected_routes_by_asset_class", {})
    conclusions = [
        "Trend-following and breakout-test modules should carry more weight than pure reversal modules because the books repeatedly emphasize that most reversals fail.",
        (
            "Current data shows at least one price-action module with robust positive expectancy."
            if effective_modules
            else "Current local data does not show a universally robust Brooks-style edge; any positive result should be treated as conditional."
        ),
        f"Effective modules: {', '.join(effective_modules) or '-'}; mixed modules: {', '.join(mixed_modules) or '-'}",
        (
            "Composite remains promising across the included market set."
            if composite_payload["effectiveness_status"] in {"effective", "mixed_positive"}
            else "Composite does not yet justify promotion without more microstructure-aware filtering or market-specific routing."
        ),
        (
            "Adaptive routing found no asset-class-specific routes worth keeping out of sample."
            if int(adaptive_route_payload.get("selected_route_count", 0)) == 0
            else f"Adaptive routing selected: {format_selected_routes_by_asset_class(adaptive_selected)}"
        ),
        (
            "Adaptive route improves out-of-sample quality over the naive composite."
            if float(adaptive_route_payload.get("out_of_sample_metrics", {}).get("expectancy_r", 0.0)) > float(composite_metrics.get("expectancy_r", 0.0))
            else "Adaptive route still needs tighter regime filters; out-of-sample edge is not yet superior enough."
        ),
    ]

    payload: dict[str, Any] = {
        "action": "backtest_brooks_price_action_all_market",
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "study_status": "complete",
        "data_source_mode": "local_research_bars_only",
        "min_bars": max(60, int(args.min_bars)),
        "asset_classes_filter": sorted(asset_classes) if asset_classes else [],
        "symbols_filter": sorted(symbols) if symbols else [],
        "book_foundations": BOOK_SOURCES,
        "coverage_summary": summarize_coverage(coverage=coverage, skipped=skipped),
        "coverage_rows": coverage,
        "scanned_rows": scanned[:400],
        "skipped_rows": skipped[:200],
        "strategy_modules": strategy_modules,
        "composite_strategy": composite_payload,
        "adaptive_route_strategy": adaptive_route_payload,
        "conclusions": conclusions,
    }

    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_brooks_price_action_market_study.json"
    markdown_path = review_dir / f"{stamp}_brooks_price_action_market_study.md"
    checksum_path = review_dir / f"{stamp}_brooks_price_action_market_study_checksum.json"

    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(artifact_path),
        "markdown": str(markdown_path),
        "sha256": sha256_file(artifact_path),
        "generated_at": payload.get("as_of"),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir,
        stem="brooks_price_action_market_study",
        current_paths=[artifact_path, markdown_path, checksum_path],
        keep=max(1, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
        now_dt=runtime_now,
    )
    payload["artifact"] = str(artifact_path)
    payload["markdown"] = str(markdown_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["sha256"] = sha256_file(artifact_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
