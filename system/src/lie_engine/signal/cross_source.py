from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TradeAlignmentResult:
    summary: dict[str, Any]
    windows: pd.DataFrame


def _bounded_float(v: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        out = float(v)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    return float(min(max(out, lo), hi))


def bucket_trade_flow(trades: pd.DataFrame, *, window_ms: int) -> pd.DataFrame:
    cols = [
        "bucket_ts_ms",
        "trade_count",
        "buy_qty",
        "sell_qty",
        "net_qty",
        "notional",
        "median_event_ts_ms",
    ]
    if not isinstance(trades, pd.DataFrame) or trades.empty:
        return pd.DataFrame(columns=cols)

    w = max(1, int(window_ms))
    df = trades.copy()
    required = {"event_ts_ms", "qty", "price", "side"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=cols)

    df["event_ts_ms"] = pd.to_numeric(df["event_ts_ms"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["side"] = df["side"].astype(str).str.upper()
    df = df.dropna(subset=["event_ts_ms", "qty", "price"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["bucket_ts_ms"] = (df["event_ts_ms"].astype("int64") // w) * w
    df["buy_qty"] = np.where(df["side"].isin(["BUY", "LONG"]), df["qty"], 0.0)
    df["sell_qty"] = np.where(df["side"].isin(["SELL", "SHORT"]), df["qty"], 0.0)
    df["signed_qty"] = df["buy_qty"] - df["sell_qty"]
    df["notional"] = (df["qty"] * df["price"]).abs()

    agg = (
        df.groupby("bucket_ts_ms", as_index=False)
        .agg(
            trade_count=("event_ts_ms", "size"),
            buy_qty=("buy_qty", "sum"),
            sell_qty=("sell_qty", "sum"),
            net_qty=("signed_qty", "sum"),
            notional=("notional", "sum"),
            median_event_ts_ms=("event_ts_ms", "median"),
        )
        .sort_values("bucket_ts_ms")
        .reset_index(drop=True)
    )
    agg["median_event_ts_ms"] = agg["median_event_ts_ms"].round().astype("int64")
    return agg


def align_trade_windows(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    window_ms: int,
    tolerance_ms: int,
    continuous_gap_ms: int,
) -> TradeAlignmentResult:
    w = max(1, int(window_ms))
    tol = max(1, int(tolerance_ms))
    gap_limit = max(w, int(continuous_gap_ms))

    left_b = bucket_trade_flow(left, window_ms=w)
    right_b = bucket_trade_flow(right, window_ms=w)
    if left_b.empty and right_b.empty:
        return TradeAlignmentResult(
            summary={
                "window_ms": w,
                "bucket_total": 0,
                "overlap_buckets": 0,
                "missing_left_buckets": 0,
                "missing_right_buckets": 0,
                "missing_left_pct": 0.0,
                "missing_right_pct": 0.0,
                "max_missing_left_ms": 0,
                "max_missing_right_ms": 0,
                "median_bucket_skew_ms": 0.0,
                "sync_ok": False,
                "gap_ok": False,
            },
            windows=pd.DataFrame(
                columns=[
                    "bucket_ts_ms",
                    "left_trade_count",
                    "right_trade_count",
                    "left_notional",
                    "right_notional",
                    "left_missing",
                    "right_missing",
                    "bucket_skew_ms",
                ]
            ),
        )

    min_left = int(left_b["bucket_ts_ms"].min()) if not left_b.empty else None
    max_left = int(left_b["bucket_ts_ms"].max()) if not left_b.empty else None
    min_right = int(right_b["bucket_ts_ms"].min()) if not right_b.empty else None
    max_right = int(right_b["bucket_ts_ms"].max()) if not right_b.empty else None

    overlap_start = max(v for v in [min_left, min_right] if v is not None)
    overlap_end = min(v for v in [max_left, max_right] if v is not None)
    if overlap_end < overlap_start:
        overlap_start = min(v for v in [min_left, min_right] if v is not None)
        overlap_end = max(v for v in [max_left, max_right] if v is not None)

    buckets = np.arange(overlap_start, overlap_end + w, w, dtype="int64")
    frame = pd.DataFrame({"bucket_ts_ms": buckets})
    left_join = left_b.rename(
        columns={
            "trade_count": "left_trade_count",
            "buy_qty": "left_buy_qty",
            "sell_qty": "left_sell_qty",
            "net_qty": "left_net_qty",
            "notional": "left_notional",
            "median_event_ts_ms": "left_median_event_ts_ms",
        }
    )
    right_join = right_b.rename(
        columns={
            "trade_count": "right_trade_count",
            "buy_qty": "right_buy_qty",
            "sell_qty": "right_sell_qty",
            "net_qty": "right_net_qty",
            "notional": "right_notional",
            "median_event_ts_ms": "right_median_event_ts_ms",
        }
    )
    frame = frame.merge(left_join, on="bucket_ts_ms", how="left").merge(right_join, on="bucket_ts_ms", how="left")

    frame["left_missing"] = frame["left_trade_count"].isna()
    frame["right_missing"] = frame["right_trade_count"].isna()
    for c in (
        "left_trade_count",
        "left_buy_qty",
        "left_sell_qty",
        "left_net_qty",
        "left_notional",
        "right_trade_count",
        "right_buy_qty",
        "right_sell_qty",
        "right_net_qty",
        "right_notional",
    ):
        frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0.0)
    frame["left_trade_count"] = frame["left_trade_count"].astype("int64")
    frame["right_trade_count"] = frame["right_trade_count"].astype("int64")

    both = (~frame["left_missing"]) & (~frame["right_missing"])
    frame["bucket_skew_ms"] = 0.0
    frame.loc[both, "bucket_skew_ms"] = (
        frame.loc[both, "left_median_event_ts_ms"] - frame.loc[both, "right_median_event_ts_ms"]
    ).astype(float).abs()

    def _max_missing_ms(mask: pd.Series) -> int:
        cur = 0
        best = 0
        for missing in mask.tolist():
            if bool(missing):
                cur += w
                if cur > best:
                    best = cur
            else:
                cur = 0
        return int(best)

    total = int(len(frame))
    missing_left = int(frame["left_missing"].sum())
    missing_right = int(frame["right_missing"].sum())
    overlap = int(both.sum())
    max_missing_left_ms = _max_missing_ms(frame["left_missing"])
    max_missing_right_ms = _max_missing_ms(frame["right_missing"])
    skew_ms = float(frame.loc[both, "bucket_skew_ms"].median()) if overlap > 0 else float("inf")
    sync_ok = bool(np.isfinite(skew_ms) and skew_ms <= tol)
    gap_ok = bool(max_missing_left_ms <= gap_limit and max_missing_right_ms <= gap_limit)

    summary = {
        "window_ms": int(w),
        "bucket_total": total,
        "overlap_buckets": overlap,
        "missing_left_buckets": missing_left,
        "missing_right_buckets": missing_right,
        "missing_left_pct": _bounded_float(missing_left / max(1, total), 0.0, 1.0, 0.0),
        "missing_right_pct": _bounded_float(missing_right / max(1, total), 0.0, 1.0, 0.0),
        "max_missing_left_ms": int(max_missing_left_ms),
        "max_missing_right_ms": int(max_missing_right_ms),
        "median_bucket_skew_ms": float(0.0 if not np.isfinite(skew_ms) else skew_ms),
        "sync_ok": bool(sync_ok),
        "gap_ok": bool(gap_ok),
    }
    keep_cols = [
        "bucket_ts_ms",
        "left_trade_count",
        "right_trade_count",
        "left_notional",
        "right_notional",
        "left_missing",
        "right_missing",
        "bucket_skew_ms",
    ]
    return TradeAlignmentResult(summary=summary, windows=frame[keep_cols].copy())
