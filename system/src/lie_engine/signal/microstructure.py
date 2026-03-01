from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

L2_REQUIRED_FIELDS = (
    "exchange",
    "symbol",
    "event_ts_ms",
    "recv_ts_ms",
    "bids",
    "asks",
    "source",
)
TRADES_REQUIRED_FIELDS = (
    "exchange",
    "symbol",
    "trade_id",
    "event_ts_ms",
    "recv_ts_ms",
    "price",
    "qty",
    "side",
    "source",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _sum_depth_qty(levels: Any, top_n: int = 3) -> float:
    if not isinstance(levels, list):
        return 0.0
    total = 0.0
    for item in levels[: max(1, int(top_n))]:
        if isinstance(item, dict):
            total += _safe_float(item.get("qty"), 0.0)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            total += _safe_float(item[1], 0.0)
    return float(max(0.0, total))


def _missing_fields(df: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        return list(required)
    cols = set(df.columns)
    return [x for x in required if x not in cols]


def summarize_microstructure_snapshot(
    *,
    l2: pd.DataFrame,
    trades: pd.DataFrame,
    cross_source_tolerance_ms: int = 80,
    continuous_gap_ms: int = 2500,
    min_trade_count: int = 30,
) -> dict[str, Any]:
    l2_df = l2.copy() if isinstance(l2, pd.DataFrame) else pd.DataFrame()
    trades_df = trades.copy() if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    l2_missing_fields = _missing_fields(l2_df, L2_REQUIRED_FIELDS)
    trade_missing_fields = _missing_fields(trades_df, TRADES_REQUIRED_FIELDS)

    queue_imbalance = 0.0
    sync_skew_ms_values: list[float] = []
    has_depth = False
    latest_seq = 0
    if not l2_df.empty and not l2_missing_fields:
        l2_df = l2_df.copy()
        l2_df["recv_ts_ms"] = pd.to_numeric(l2_df["recv_ts_ms"], errors="coerce")
        l2_df["event_ts_ms"] = pd.to_numeric(l2_df["event_ts_ms"], errors="coerce")
        l2_df = l2_df.dropna(subset=["recv_ts_ms", "event_ts_ms"]).sort_values("recv_ts_ms")
        if not l2_df.empty:
            latest = l2_df.iloc[-1]
            bid_qty = _sum_depth_qty(latest.get("bids", []), top_n=3)
            ask_qty = _sum_depth_qty(latest.get("asks", []), top_n=3)
            denom = max(1e-9, bid_qty + ask_qty)
            queue_imbalance = float((bid_qty - ask_qty) / denom)
            sync_skew_ms_values.append(abs(_safe_float(latest.get("recv_ts_ms")) - _safe_float(latest.get("event_ts_ms"))))
            latest_seq = int(_safe_float(latest.get("seq"), 0.0))
            has_depth = True

    trade_count = 0
    ofi_norm = 0.0
    max_trade_gap_ms = 0
    if not trades_df.empty and not trade_missing_fields:
        trades_df = trades_df.copy()
        trades_df["event_ts_ms"] = pd.to_numeric(trades_df["event_ts_ms"], errors="coerce")
        trades_df["recv_ts_ms"] = pd.to_numeric(trades_df["recv_ts_ms"], errors="coerce")
        trades_df["qty"] = pd.to_numeric(trades_df["qty"], errors="coerce")
        trades_df["side"] = trades_df["side"].astype(str).str.upper()
        trades_df = trades_df.dropna(subset=["event_ts_ms", "recv_ts_ms", "qty"])
        if not trades_df.empty:
            trades_df = trades_df.sort_values("event_ts_ms")
            trade_count = int(len(trades_df))
            buy_qty = float(trades_df.loc[trades_df["side"].isin(["BUY", "LONG"]), "qty"].sum())
            sell_qty = float(trades_df.loc[trades_df["side"].isin(["SELL", "SHORT"]), "qty"].sum())
            denom = max(1e-9, buy_qty + sell_qty)
            ofi_norm = float((buy_qty - sell_qty) / denom)
            diffs = trades_df["event_ts_ms"].diff().dropna()
            if not diffs.empty:
                max_trade_gap_ms = int(max(0.0, float(diffs.max())))
            sync_skew_ms_values.append(float((trades_df["recv_ts_ms"] - trades_df["event_ts_ms"]).abs().median()))

    sync_skew_ms = float(max(sync_skew_ms_values)) if sync_skew_ms_values else 0.0
    sync_ok = bool(sync_skew_ms <= float(max(1, cross_source_tolerance_ms)))
    gap_ok = bool(max_trade_gap_ms <= int(max(100, continuous_gap_ms)))

    evidence_score = float(min(1.0, max(0.0, trade_count / max(1, int(min_trade_count)))))
    if has_depth:
        evidence_score = float(min(1.0, evidence_score + 0.25))

    micro_alignment = float(np.clip(0.55 * queue_imbalance + 0.45 * ofi_norm, -1.0, 1.0))
    schema_issues: list[str] = []
    if l2_missing_fields:
        schema_issues.append("l2_missing:" + ",".join(l2_missing_fields))
    if trade_missing_fields:
        schema_issues.append("trades_missing:" + ",".join(trade_missing_fields))
    return {
        "has_data": bool(has_depth or trade_count > 0),
        "schema_ok": bool(not l2_missing_fields and not trade_missing_fields),
        "schema_issues": schema_issues,
        "l2_required_fields": list(L2_REQUIRED_FIELDS),
        "trade_required_fields": list(TRADES_REQUIRED_FIELDS),
        "l2_missing_fields": list(l2_missing_fields),
        "trade_missing_fields": list(trade_missing_fields),
        "queue_imbalance": float(np.clip(queue_imbalance, -1.0, 1.0)),
        "ofi_norm": float(np.clip(ofi_norm, -1.0, 1.0)),
        "micro_alignment": micro_alignment,
        "trade_count": int(trade_count),
        "max_trade_gap_ms": int(max_trade_gap_ms),
        "sync_skew_ms": float(sync_skew_ms),
        "sync_ok": bool(sync_ok),
        "gap_ok": bool(gap_ok),
        "evidence_score": float(np.clip(evidence_score, 0.0, 1.0)),
        "latest_seq": int(latest_seq),
    }
