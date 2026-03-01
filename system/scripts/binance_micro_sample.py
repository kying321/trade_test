#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lie_engine.data.providers import BinanceSpotPublicProvider


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Binance Spot public L2 and trade streams.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g. BTCUSDT.")
    parser.add_argument("--minutes", type=int, default=10, help="Lookback minutes for aggTrades sampling.")
    parser.add_argument("--depth", type=int, default=20, help="Depth limit for orderbook snapshot.")
    parser.add_argument("--limit", type=int, default=500, help="Max aggTrades rows (Binance hard cap 1000).")
    parser.add_argument("--rpm", type=int, default=10, help="Token bucket request limit per minute.")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="HTTP timeout in ms (hard capped at 5000).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    end_ts = datetime.now(tz=timezone.utc)
    start_ts = end_ts - timedelta(minutes=max(1, int(args.minutes)))

    provider = BinanceSpotPublicProvider(
        rate_limit_per_minute=max(1, int(args.rpm)),
        request_timeout_ms=min(5000, max(100, int(args.timeout_ms))),
    )

    l2 = provider.fetch_l2(symbol=args.symbol, start_ts=start_ts, end_ts=end_ts, depth=max(1, int(args.depth)))
    trades = provider.fetch_trades(
        symbol=args.symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=max(1, min(1000, int(args.limit))),
    )
    try:
        time_sync = provider.fetch_time_sync_sample()
    except Exception:
        time_sync = {}

    summary = {
        "provider": provider.name,
        "symbol": str(args.symbol).upper(),
        "window_start_utc": start_ts.isoformat(),
        "window_end_utc": end_ts.isoformat(),
        "depth_rows": int(len(l2)),
        "trade_rows": int(len(trades)),
        "depth_columns": list(l2.columns),
        "trade_columns": list(trades.columns),
        "trade_event_ts_min": int(trades["event_ts_ms"].min()) if ("event_ts_ms" in trades.columns and len(trades) > 0) else None,
        "trade_event_ts_max": int(trades["event_ts_ms"].max()) if ("event_ts_ms" in trades.columns and len(trades) > 0) else None,
        "time_sync": (
            {
                "available": True,
                "server_ts_ms": int(time_sync.get("server_ts_ms", 0)),
                "local_mid_ts_ms": int(time_sync.get("local_mid_ts_ms", 0)),
                "rtt_ms": int(time_sync.get("rtt_ms", 0)),
                "offset_ms": int(time_sync.get("offset_ms", 0)),
                "offset_abs_ms": int(time_sync.get("offset_abs_ms", 0)),
            }
            if isinstance(time_sync, dict) and bool(time_sync)
            else {"available": False}
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
