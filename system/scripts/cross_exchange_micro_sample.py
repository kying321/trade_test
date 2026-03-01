#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lie_engine.data.providers import BinanceSpotPublicProvider, BybitSpotPublicProvider
from lie_engine.signal.cross_source import align_trade_windows
from lie_engine.signal.microstructure import (
    L2_REQUIRED_FIELDS,
    TRADES_REQUIRED_FIELDS,
    summarize_microstructure_snapshot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Binance + Bybit cross-exchange microstructure alignment sample.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Spot symbol, e.g. BTCUSDT.")
    parser.add_argument("--minutes", type=int, default=10, help="Lookback minutes.")
    parser.add_argument("--depth", type=int, default=20, help="Orderbook depth levels.")
    parser.add_argument("--limit", type=int, default=500, help="Trade row cap per source.")
    parser.add_argument("--window-ms", type=int, default=200, help="Trade alignment bucket in milliseconds.")
    parser.add_argument(
        "--align-seconds",
        type=int,
        default=90,
        help="Near-real-time trade alignment lookback seconds (separate from --minutes).",
    )
    parser.add_argument("--tolerance-ms", type=int, default=80, help="Cross-source median skew threshold.")
    parser.add_argument("--continuous-gap-ms", type=int, default=2500, help="Max continuous missing bucket gap.")
    parser.add_argument("--rpm", type=int, default=10, help="Per-source token bucket rpm.")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="Per-source timeout in ms (hard cap 5000).")
    parser.add_argument(
        "--out-dir",
        default="output/artifacts/micro_alignment",
        help="Output directory (relative to system/ unless absolute).",
    )
    return parser.parse_args()


def _top_price(levels: object, *, side: str) -> float:
    if not isinstance(levels, list) or not levels:
        return 0.0
    first = levels[0]
    if isinstance(first, dict):
        try:
            return float(first.get("price", 0.0))
        except Exception:
            return 0.0
    if isinstance(first, (list, tuple)) and len(first) >= 1:
        try:
            return float(first[0])
        except Exception:
            return 0.0
    return 0.0


def _l2_compare(a, b) -> dict[str, object]:
    if len(a) == 0 or len(b) == 0:
        return {
            "available": False,
            "event_skew_ms": None,
            "seq_gap_abs": None,
            "mid_gap_bps": None,
        }
    a0 = a.iloc[0]
    b0 = b.iloc[0]
    a_bid = _top_price(a0.get("bids", []), side="bid")
    a_ask = _top_price(a0.get("asks", []), side="ask")
    b_bid = _top_price(b0.get("bids", []), side="bid")
    b_ask = _top_price(b0.get("asks", []), side="ask")
    a_mid = (a_bid + a_ask) / 2.0 if a_bid > 0 and a_ask > 0 else 0.0
    b_mid = (b_bid + b_ask) / 2.0 if b_bid > 0 and b_ask > 0 else 0.0
    mid_gap_bps = 0.0
    if a_mid > 0 and b_mid > 0:
        mid_gap_bps = abs(a_mid - b_mid) / max(1e-9, (a_mid + b_mid) / 2.0) * 10000.0
    return {
        "available": True,
        "event_skew_ms": int(abs(int(a0.get("event_ts_ms", 0)) - int(b0.get("event_ts_ms", 0)))),
        "seq_gap_abs": int(abs(int(a0.get("seq", 0)) - int(b0.get("seq", 0)))),
        "mid_gap_bps": float(mid_gap_bps),
        "binance_top": {"bid": float(a_bid), "ask": float(a_ask)},
        "bybit_top": {"bid": float(b_bid), "ask": float(b_ask)},
    }


def _source_summary(name: str, l2, trades, *, tolerance_ms: int, gap_ms: int, min_trade_count: int) -> dict[str, object]:
    snap = summarize_microstructure_snapshot(
        l2=l2,
        trades=trades,
        cross_source_tolerance_ms=tolerance_ms,
        continuous_gap_ms=gap_ms,
        min_trade_count=min_trade_count,
    )
    out = {
        "source": name,
        "l2_rows": int(len(l2)),
        "trade_rows": int(len(trades)),
        "trade_event_ts_min": int(trades["event_ts_ms"].min()) if ("event_ts_ms" in trades.columns and len(trades) > 0) else None,
        "trade_event_ts_max": int(trades["event_ts_ms"].max()) if ("event_ts_ms" in trades.columns and len(trades) > 0) else None,
    }
    out.update(snap)
    return out


def _safe_time_sync(provider) -> dict[str, object]:  # type: ignore[no-untyped-def]
    fetch_fn = getattr(provider, "fetch_time_sync_sample", None)
    if not callable(fetch_fn):
        return {"available": False}
    try:
        sample = fetch_fn()
    except Exception:
        return {"available": False}
    if not isinstance(sample, dict):
        return {"available": False}
    return {"available": True} | {k: sample.get(k) for k in ("server_ts_ms", "local_mid_ts_ms", "rtt_ms", "offset_ms", "offset_abs_ms")}


def main() -> int:
    args = _parse_args()
    symbol = str(args.symbol).upper().strip()
    now_utc = datetime.now(tz=timezone.utc)
    start_ts = now_utc - timedelta(minutes=max(1, int(args.minutes)))
    end_ts = now_utc

    rpm = max(1, int(args.rpm))
    timeout_ms = min(5000, max(100, int(args.timeout_ms)))
    depth = max(1, int(args.depth))
    limit = max(1, min(1000, int(args.limit)))

    binance = BinanceSpotPublicProvider(rate_limit_per_minute=rpm, request_timeout_ms=timeout_ms)
    bybit = BybitSpotPublicProvider(rate_limit_per_minute=rpm, request_timeout_ms=timeout_ms)

    b_l2 = binance.fetch_l2(symbol=symbol, start_ts=start_ts, end_ts=end_ts, depth=depth)
    b_trades = binance.fetch_trades(symbol=symbol, start_ts=start_ts, end_ts=end_ts, limit=limit)
    y_l2 = bybit.fetch_l2(symbol=symbol, start_ts=start_ts, end_ts=end_ts, depth=depth)
    y_trades = bybit.fetch_trades(symbol=symbol, start_ts=start_ts, end_ts=end_ts, limit=limit)
    align_start_ts = end_ts - timedelta(seconds=max(30, int(args.align_seconds)))
    b_trades_align = binance.fetch_trades(symbol=symbol, start_ts=align_start_ts, end_ts=end_ts, limit=limit)
    y_trades_align = bybit.fetch_trades(symbol=symbol, start_ts=align_start_ts, end_ts=end_ts, limit=limit)

    align = align_trade_windows(
        left=b_trades_align,
        right=y_trades_align,
        window_ms=max(1, int(args.window_ms)),
        tolerance_ms=max(1, int(args.tolerance_ms)),
        continuous_gap_ms=max(1, int(args.continuous_gap_ms)),
    )
    l2_cmp = _l2_compare(b_l2, y_l2)

    payload = {
        "generated_at_utc": now_utc.isoformat(),
        "symbol": symbol,
        "window": {
            "start_utc": start_ts.isoformat(),
            "end_utc": end_ts.isoformat(),
            "lookback_minutes": int(max(1, int(args.minutes))),
            "window_ms": int(max(1, int(args.window_ms))),
        },
        "truth_baseline": {
            "l2_min_fields": list(L2_REQUIRED_FIELDS),
            "trade_min_fields": list(TRADES_REQUIRED_FIELDS),
            "cross_source_tolerance_ms": int(max(1, int(args.tolerance_ms))),
            "continuous_gap_ms": int(max(1, int(args.continuous_gap_ms))),
        },
        "sources": {
            "binance": _source_summary(
                binance.name,
                b_l2,
                b_trades,
                tolerance_ms=max(1, int(args.tolerance_ms)),
                gap_ms=max(1, int(args.continuous_gap_ms)),
                min_trade_count=max(1, int(args.limit) // 5),
            ),
            "bybit": _source_summary(
                bybit.name,
                y_l2,
                y_trades,
                tolerance_ms=max(1, int(args.tolerance_ms)),
                gap_ms=max(1, int(args.continuous_gap_ms)),
                min_trade_count=max(1, int(args.limit) // 5),
            ),
        },
        "time_sync": {
            "binance": _safe_time_sync(binance),
            "bybit": _safe_time_sync(bybit),
        },
        "l2_compare": l2_cmp,
        "trade_alignment": align.summary,
        "trade_alignment_window": {
            "start_utc": align_start_ts.isoformat(),
            "end_utc": end_ts.isoformat(),
            "binance_rows": int(len(b_trades_align)),
            "bybit_rows": int(len(y_trades_align)),
        },
        "trade_alignment_sample": align.windows.head(25).to_dict(orient="records"),
        "verdict": {
            "sync_ok": bool(align.summary.get("sync_ok", False)),
            "gap_ok": bool(align.summary.get("gap_ok", False)),
            "pass": bool(align.summary.get("sync_ok", False)) and bool(align.summary.get("gap_ok", False)),
        },
    }

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    out_path = out_root / f"{stamp}_{symbol}_binance_bybit_alignment.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"artifact": str(out_path), "summary": payload["trade_alignment"], "verdict": payload["verdict"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
