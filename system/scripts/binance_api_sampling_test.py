#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
from typing import Any

import requests

from lie_engine.config import load_settings


ENDPOINTS = ("https://api.binance.com", "https://api.binance.us")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance API availability and sampling test")
    p.add_argument("--config", default=None, help="config.yaml path")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--samples-per-symbol", default="3")
    p.add_argument("--timeout-sec", default="8")
    return p.parse_args()


def _date(v: str) -> date:
    return date.fromisoformat(str(v).strip())


def _symbol_list(config_path: str | None) -> list[str]:
    settings = load_settings(config_path)
    symbols: list[str] = []
    for row in settings.universe.get("core", []):
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if symbol:
            symbols.append(symbol)
    return list(dict.fromkeys(symbols))


def _choose_sample_dates(start: date, end: date, n: int) -> list[date]:
    n = max(1, int(n))
    if end <= start:
        return [start]
    span = max(1, (end - start).days)
    out: list[date] = []
    for i in range(n):
        d = start + timedelta(days=int(round(i * span / max(1, n - 1))))
        out.append(d)
    return out


def _request(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "lie-engine/1.0"})
        payload: Any = None
        try:
            payload = resp.json()
        except Exception:
            payload = None
        return {
            "http": int(resp.status_code),
            "ok": bool(resp.status_code == 200),
            "payload_type": type(payload).__name__,
            "size": int(len(payload)) if isinstance(payload, (list, dict)) else 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"http": 0, "ok": False, "error": str(exc)}


def main() -> None:
    args = _parse_args()
    start = _date(args.start)
    end = _date(args.end)
    samples = int(args.samples_per_symbol)
    timeout = float(args.timeout_sec)

    root = Path(args.config).expanduser().resolve().parent if args.config else Path(__file__).resolve().parents[1]
    review_dir = root / "output" / "review"
    logs_dir = root / "output" / "logs"
    review_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    symbols = _symbol_list(args.config)
    sample_dates = _choose_sample_dates(start=start, end=end, n=samples)

    rows: list[dict[str, Any]] = []
    time_probe_rows: list[dict[str, Any]] = []
    for base in ENDPOINTS:
        time_probe = _request(f"{base}/api/v3/time", params={}, timeout=timeout)
        time_probe_rows.append({"base": base} | time_probe)
        for symbol in symbols:
            for d in sample_dates:
                start_ms = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
                result = _request(
                    f"{base}/api/v3/klines",
                    params={"symbol": symbol, "interval": "1d", "startTime": start_ms, "limit": 5},
                    timeout=timeout,
                )
                rows.append(
                    {
                        "base": base,
                        "symbol": symbol,
                        "sample_date": d.isoformat(),
                    }
                    | result
                )

    per_symbol_ok: dict[str, bool] = {}
    for symbol in symbols:
        sym_rows = [r for r in rows if str(r.get("symbol")) == symbol]
        per_symbol_ok[symbol] = any(bool(x.get("ok", False)) for x in sym_rows)

    status = "PASSED" if symbols and all(per_symbol_ok.values()) and any(bool(x.get("ok", False)) for x in time_probe_rows) else "FAILED"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "as_of": datetime.now().isoformat(),
        "status": status,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "symbols": symbols,
        "samples_per_symbol": samples,
        "time_probes": time_probe_rows,
        "per_symbol_ok": per_symbol_ok,
        "rows": rows,
    }

    review_path = review_dir / f"{date.today().isoformat()}_binance_api_sampling_test_{stamp}.json"
    log_path = logs_dir / f"tests_binance_api_sampling_{stamp}.json"
    review_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(
        json.dumps(
            {
                "status": status,
                "review_path": str(review_path),
                "time_probe_ok": any(bool(x.get("ok", False)) for x in time_probe_rows),
                "symbols_ok": per_symbol_ok,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "review_path": str(review_path), "test_log": str(log_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
