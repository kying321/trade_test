#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from lie_engine.config import load_settings


ENDPOINTS = ("https://api.binance.com", "https://api.binance.us")
INTERVAL_MS = {"30m": 30 * 60 * 1000}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance 90d 30m execution friction backfill")
    p.add_argument("--config", default=None, help="config.yaml path")
    p.add_argument("--start", default="", help="YYYY-MM-DD (default end-90d)")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--interval", default="30m")
    p.add_argument("--notionals", default="5000,20000,50000")
    p.add_argument("--timeout-sec", default="8")
    p.add_argument("--max-calls-per-minute", default="10")
    p.add_argument("--spread-proxy-coeff", default="0.08", help="spread proxy coeff on (high-low)/close in bps")
    p.add_argument("--impact-linear-bps", default="120.0", help="linear impact coeff in bps x participation")
    return p.parse_args()


def _date(v: str) -> date:
    return date.fromisoformat(str(v).strip())


def _symbols(config_path: str | None) -> list[str]:
    settings = load_settings(config_path)
    out: list[str] = []
    for row in settings.universe.get("core", []):
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol:
            out.append(symbol)
    return list(dict.fromkeys(out))


def _throttle(call_ts: list[float], max_per_minute: int) -> None:
    if max_per_minute <= 0:
        return
    window = 60.0
    now = time.monotonic()
    call_ts[:] = [x for x in call_ts if (now - x) < window]
    if len(call_ts) >= max_per_minute:
        wait = window - (now - call_ts[0]) + 0.01
        time.sleep(max(0.05, wait))
    call_ts.append(time.monotonic())


def _request_json(url: str, params: dict[str, Any], timeout: float, call_ts: list[float], max_per_minute: int) -> tuple[int, Any]:
    _throttle(call_ts, max_per_minute)
    try:
        resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "lie-engine/1.0"})
        code = int(resp.status_code)
        payload: Any = None
        try:
            payload = resp.json()
        except Exception:
            payload = None
        return code, payload
    except Exception:
        return 0, None


def _fetch_symbol_klines(
    *,
    symbol: str,
    start: date,
    end: date,
    interval: str,
    timeout: float,
    call_ts: list[float],
    max_per_minute: int,
) -> tuple[pd.DataFrame, str]:
    if interval not in INTERVAL_MS:
        return pd.DataFrame(), ""
    interval_ms = INTERVAL_MS[interval]
    start_ms = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1) - timedelta(milliseconds=1)
    end_ms = int(end_dt.timestamp() * 1000)

    for base in ENDPOINTS:
        rows: list[list[Any]] = []
        cursor = start_ms
        guard = 0
        while cursor <= end_ms:
            code, payload = _request_json(
                f"{base}/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=timeout,
                call_ts=call_ts,
                max_per_minute=max_per_minute,
            )
            if code != 200 or not isinstance(payload, list) or not payload:
                break
            chunk = [x for x in payload if isinstance(x, list) and len(x) >= 11]
            if not chunk:
                break
            rows.extend(chunk)
            last_open = int(chunk[-1][0])
            nxt = last_open + interval_ms
            if nxt <= cursor:
                break
            cursor = nxt
            guard += 1
            if len(chunk) < 1000 or guard > 40:
                break
        if rows:
            df = pd.DataFrame(
                {
                    "ts": [pd.to_datetime(int(r[0]), unit="ms", utc=True).tz_convert(None) for r in rows],
                    "symbol": str(symbol).upper(),
                    "open": [float(r[1]) for r in rows],
                    "high": [float(r[2]) for r in rows],
                    "low": [float(r[3]) for r in rows],
                    "close": [float(r[4]) for r in rows],
                    "base_volume": [float(r[5]) for r in rows],
                    "quote_volume": [float(r[7]) for r in rows],
                    "num_trades": [int(r[8]) for r in rows],
                    "taker_buy_quote_volume": [float(r[10]) for r in rows],
                    "source": base,
                }
            )
            df = df.drop_duplicates(subset=["ts", "symbol"]).sort_values("ts").reset_index(drop=True)
            return df, base
    return pd.DataFrame(), ""


def _percentiles(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p05": float(np.quantile(s, 0.05)),
        "p50": float(np.quantile(s, 0.50)),
        "p95": float(np.quantile(s, 0.95)),
        "p99": float(np.quantile(s, 0.99)),
        "mean": float(s.mean()),
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Binance Execution Friction Backfill | {payload.get('as_of', '')}")
    lines.append("")
    lines.append(f"- status: `{payload.get('status', '')}`")
    lines.append(f"- window: `{payload.get('start', '')} ~ {payload.get('end', '')}`")
    lines.append(f"- interval: `{payload.get('interval', '')}`")
    lines.append(f"- symbols: `{payload.get('symbols', [])}`")
    lines.append(f"- rows: `{payload.get('rows', 0)}`")
    lines.append("")
    lines.append("## Slippage (bps)")
    slip = payload.get("slippage_bps", {})
    if isinstance(slip, dict):
        for k, v in slip.items():
            if not isinstance(v, dict):
                continue
            lines.append(
                f"- `{k}`: p50={float(v.get('p50', 0.0)):.3f}, p95={float(v.get('p95', 0.0)):.3f}, p99={float(v.get('p99', 0.0)):.3f}"
            )
    lines.append("")
    liq = payload.get("liquidity", {})
    qv = liq.get("quote_volume_usdt", {}) if isinstance(liq, dict) else {}
    lines.append("## Liquidity")
    lines.append(
        f"- quote_volume_usdt: p50={float(qv.get('p50', 0.0)):.2f}, p05={float(qv.get('p05', 0.0)):.2f}, p95={float(qv.get('p95', 0.0)):.2f}"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    end = _date(args.end)
    start = _date(args.start) if str(args.start).strip() else (end - timedelta(days=90))
    interval = str(args.interval).strip()
    timeout = float(args.timeout_sec)
    max_calls_per_minute = int(args.max_calls_per_minute)
    spread_proxy_coeff = max(0.0, float(args.spread_proxy_coeff))
    impact_linear_bps = max(0.0, float(args.impact_linear_bps))
    notionals = [float(x.strip()) for x in str(args.notionals).split(",") if str(x).strip()]
    notionals = [x for x in notionals if x > 0]
    if not notionals:
        notionals = [5000.0, 20000.0, 50000.0]

    root = Path(args.config).expanduser().resolve().parent if args.config else Path(__file__).resolve().parents[1]
    review_dir = root / "output" / "review"
    logs_dir = root / "output" / "logs"
    review_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    symbols = _symbols(args.config)
    call_ts: list[float] = []
    frames: list[pd.DataFrame] = []
    source_by_symbol: dict[str, str] = {}
    rows_by_symbol: dict[str, int] = {}

    for symbol in symbols:
        df, src = _fetch_symbol_klines(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            timeout=timeout,
            call_ts=call_ts,
            max_per_minute=max_calls_per_minute,
        )
        if df.empty:
            rows_by_symbol[symbol] = 0
            continue
        frames.append(df)
        source_by_symbol[symbol] = src
        rows_by_symbol[symbol] = int(len(df))

    probes: list[dict[str, Any]] = []
    for base in ENDPOINTS:
        code, payload = _request_json(
            f"{base}/api/v3/time",
            params={},
            timeout=timeout,
            call_ts=call_ts,
            max_per_minute=max_calls_per_minute,
        )
        probes.append({"base": base, "http": int(code), "ok": bool(code == 200), "payload_type": type(payload).__name__})

    if not frames:
        status = "FAILED"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        payload = {
            "as_of": datetime.now().isoformat(),
            "status": status,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "symbols": symbols,
            "rows_by_symbol": rows_by_symbol,
            "source_by_symbol": source_by_symbol,
            "endpoint_probes": probes,
            "reason": "no_klines_rows",
        }
        json_path = review_dir / f"{date.today().isoformat()}_binance_execution_friction_{stamp}.json"
        log_path = logs_dir / f"tests_binance_execution_friction_{stamp}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log_path.write_text(json.dumps({"status": status, "review": str(json_path), "reason": "no_klines_rows"}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": status, "review": str(json_path), "test_log": str(log_path)}, ensure_ascii=False, indent=2))
        return

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "close", "high", "low", "quote_volume"]).sort_values(["symbol", "ts"]).reset_index(drop=True)
    df["ret_abs"] = df.groupby("symbol")["close"].pct_change().abs().fillna(0.0)
    df["spread_bps"] = ((df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)).fillna(0.0) * 10000.0
    # OHLC high-low is not a tradable spread; downscale to avoid overstating friction.
    df["spread_proxy_bps"] = df["spread_bps"] * float(spread_proxy_coeff)

    slippage_bps: dict[str, dict[str, float]] = {}
    for notional in notionals:
        key = f"notional_{int(round(notional))}"
        participation = (float(notional) / df["quote_volume"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 1.0)
        df[f"slippage_bps_{key}"] = df["spread_proxy_bps"] + float(impact_linear_bps) * participation
        slippage_bps[key] = _percentiles(df[f"slippage_bps_{key}"])

    qv_summary = _percentiles(df["quote_volume"])
    target_key = f"notional_{int(round(notionals[min(len(notionals) - 1, 1)]))}"
    slot_rows: list[dict[str, Any]] = []
    df["slot"] = df["ts"].dt.strftime("%H:%M")
    for slot, g in df.groupby("slot"):
        slot_rows.append(
            {
                "slot": str(slot),
                "rows": int(len(g)),
                "slippage_bps_p50": float(np.quantile(g[f"slippage_bps_{target_key}"], 0.50)),
                "slippage_bps_p95": float(np.quantile(g[f"slippage_bps_{target_key}"], 0.95)),
                "slippage_bps_p99": float(np.quantile(g[f"slippage_bps_{target_key}"], 0.99)),
                "quote_volume_usdt_p50": float(np.quantile(g["quote_volume"], 0.50)),
            }
        )
    slot_rows.sort(key=lambda x: x["slot"])

    status = "PASSED" if all(rows_by_symbol.get(s, 0) > 0 for s in symbols) and any(bool(x.get("ok", False)) for x in probes) else "FAILED"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "as_of": datetime.now().isoformat(),
        "status": status,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval": interval,
        "symbols": symbols,
        "rows": int(len(df)),
        "rows_by_symbol": rows_by_symbol,
        "source_by_symbol": source_by_symbol,
        "endpoint_probes": probes,
        "proxy_model": {
            "spread_proxy_coeff": float(spread_proxy_coeff),
            "impact_linear_bps": float(impact_linear_bps),
        },
        "slippage_bps": slippage_bps,
        "liquidity": {"quote_volume_usdt": qv_summary},
        "slot_summary": slot_rows,
    }
    json_path = review_dir / f"{date.today().isoformat()}_binance_execution_friction_{stamp}.json"
    md_path = review_dir / f"{date.today().isoformat()}_binance_execution_friction_{stamp}.md"
    log_path = logs_dir / f"tests_binance_execution_friction_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    log_path.write_text(
        json.dumps(
            {
                "status": status,
                "review_json": str(json_path),
                "review_md": str(md_path),
                "symbols_covered": all(rows_by_symbol.get(s, 0) > 0 for s in symbols),
                "endpoint_time_ok": any(bool(x.get("ok", False)) for x in probes),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "review_json": str(json_path), "review_md": str(md_path), "test_log": str(log_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
