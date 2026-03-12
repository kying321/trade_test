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


def _sign(value: float, eps: float = 1e-9) -> int:
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


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
    buy_qty = 0.0
    sell_qty = 0.0
    trade_notional = 0.0
    cvd_delta_ratio = 0.0
    cvd_price_move_pct = 0.0
    cvd_range_pct = 0.0
    cvd_effort_result_ratio = 0.0
    cvd_context_mode = "unclear"
    cvd_context_note = "no_trade_data"
    cvd_trust_tier_hint = "unavailable"
    if not trades_df.empty and not trade_missing_fields:
        trades_df = trades_df.copy()
        trades_df["event_ts_ms"] = pd.to_numeric(trades_df["event_ts_ms"], errors="coerce")
        trades_df["recv_ts_ms"] = pd.to_numeric(trades_df["recv_ts_ms"], errors="coerce")
        trades_df["qty"] = pd.to_numeric(trades_df["qty"], errors="coerce")
        trades_df["price"] = pd.to_numeric(trades_df["price"], errors="coerce")
        trades_df["side"] = trades_df["side"].astype(str).str.upper()
        trades_df = trades_df.dropna(subset=["event_ts_ms", "recv_ts_ms", "qty", "price"])
        if not trades_df.empty:
            trades_df = trades_df.sort_values("event_ts_ms")
            trade_count = int(len(trades_df))
            buy_qty = float(trades_df.loc[trades_df["side"].isin(["BUY", "LONG"]), "qty"].sum())
            sell_qty = float(trades_df.loc[trades_df["side"].isin(["SELL", "SHORT"]), "qty"].sum())
            denom = max(1e-9, buy_qty + sell_qty)
            ofi_norm = float((buy_qty - sell_qty) / denom)
            trade_notional = float((trades_df["qty"] * trades_df["price"]).abs().sum())
            diffs = trades_df["event_ts_ms"].diff().dropna()
            if not diffs.empty:
                max_trade_gap_ms = int(max(0.0, float(diffs.max())))
            sync_skew_ms_values.append(float((trades_df["recv_ts_ms"] - trades_df["event_ts_ms"]).abs().median()))

            first_price = _safe_float(trades_df.iloc[0]["price"], 0.0)
            last_price = _safe_float(trades_df.iloc[-1]["price"], first_price)
            high_price = _safe_float(trades_df["price"].max(), last_price)
            low_price = _safe_float(trades_df["price"].min(), last_price)
            base_price = max(1e-9, abs(first_price) if first_price else abs(last_price))
            cvd_delta_ratio = float(np.clip(ofi_norm, -1.0, 1.0))
            cvd_price_move_pct = float((last_price - first_price) / base_price)
            cvd_range_pct = float(max(0.0, high_price - low_price) / base_price)
            price_body_pct = abs(cvd_price_move_pct)
            weak_price_result = bool(
                (cvd_range_pct > 0 and price_body_pct <= 0.35 * cvd_range_pct)
                or price_body_pct <= 0.0005
            )
            cvd_effort_result_ratio = float(
                min(25.0, abs(cvd_delta_ratio) / max(1e-6, price_body_pct))
            )

            delta_sign = _sign(cvd_delta_ratio)
            price_sign = _sign(cvd_price_move_pct)
            if trade_count < int(max(1, min_trade_count)) or max_trade_gap_ms > int(max(100, continuous_gap_ms)):
                cvd_context_mode = "unclear"
                cvd_context_note = "low_sample_or_gap_risk"
                cvd_trust_tier_hint = "single_exchange_low"
            elif delta_sign != 0 and price_sign != 0 and delta_sign != price_sign:
                cvd_context_mode = "reversal"
                cvd_context_note = "delta_and_price_move_disagree"
                cvd_trust_tier_hint = "single_exchange_ok"
            elif abs(cvd_delta_ratio) >= 0.15 and weak_price_result:
                if cvd_range_pct >= max(0.001, price_body_pct * 2.0):
                    cvd_context_mode = "failed_auction"
                    cvd_context_note = "range_expanded_but_close_failed_to_hold"
                else:
                    cvd_context_mode = "absorption"
                    cvd_context_note = "aggressive_flow_without_clean_price_result"
                cvd_trust_tier_hint = "single_exchange_ok"
            elif delta_sign != 0 and price_sign != 0 and delta_sign == price_sign:
                cvd_context_mode = "continuation"
                cvd_context_note = "delta_confirms_directional_price_result"
                cvd_trust_tier_hint = "single_exchange_ok"
            else:
                cvd_context_mode = "unclear"
                cvd_context_note = "mixed_trade_flow"
                cvd_trust_tier_hint = "single_exchange_low"

    sync_skew_ms = float(max(sync_skew_ms_values)) if sync_skew_ms_values else 0.0
    sync_ok = bool(sync_skew_ms <= float(max(1, cross_source_tolerance_ms)))
    gap_ok = bool(max_trade_gap_ms <= int(max(100, continuous_gap_ms)))

    evidence_score = float(min(1.0, max(0.0, trade_count / max(1, int(min_trade_count)))))
    if has_depth:
        evidence_score = float(min(1.0, evidence_score + 0.25))

    if trade_count < int(max(1, min_trade_count)):
        cvd_trust_tier_hint = "single_exchange_low" if trade_count > 0 else "unavailable"
    elif trade_count > 0 and sync_ok and gap_ok:
        cvd_trust_tier_hint = "single_exchange_ok"
    elif trade_count > 0:
        cvd_trust_tier_hint = "single_exchange_low"

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
        "buy_qty": float(max(0.0, buy_qty)),
        "sell_qty": float(max(0.0, sell_qty)),
        "net_qty": float(buy_qty - sell_qty),
        "trade_notional": float(max(0.0, trade_notional)),
        "cvd_delta_ratio": float(np.clip(cvd_delta_ratio, -1.0, 1.0)),
        "cvd_price_move_pct": float(cvd_price_move_pct),
        "cvd_range_pct": float(max(0.0, cvd_range_pct)),
        "cvd_effort_result_ratio": float(max(0.0, cvd_effort_result_ratio)),
        "cvd_context_mode": str(cvd_context_mode),
        "cvd_context_note": str(cvd_context_note),
        "cvd_trust_tier_hint": str(cvd_trust_tier_hint),
        "trade_count": int(trade_count),
        "max_trade_gap_ms": int(max_trade_gap_ms),
        "sync_skew_ms": float(sync_skew_ms),
        "sync_ok": bool(sync_ok),
        "gap_ok": bool(gap_ok),
        "evidence_score": float(np.clip(evidence_score, 0.0, 1.0)),
        "latest_seq": int(latest_seq),
    }
