#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lie_engine.research.real_data import fetch_future_daily


def parse_date(raw: str) -> dt.date:
    return dt.date.fromisoformat(str(raw).strip())


def runtime_stamp(as_of: dt.date) -> str:
    return dt.datetime.combine(as_of, dt.time(0, 0), tzinfo=dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_utc_iso_for_date(as_of: dt.date) -> str:
    return dt.datetime.combine(as_of, dt.time(0, 0), tzinfo=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(raw: Any) -> float:
    try:
        return float(raw)
    except Exception:
        return 0.0


def build_directional_ticket(
    *,
    symbol: str,
    frame: pd.DataFrame,
    as_of: dt.date,
    max_age_days: int,
) -> dict[str, Any]:
    ordered = frame.sort_values("ts").reset_index(drop=True) if not frame.empty else pd.DataFrame()
    ordered = ordered[ordered["ts"].dt.date <= as_of].reset_index(drop=True) if not ordered.empty else ordered
    if ordered.empty:
        return {
            "symbol": symbol,
            "date": as_of.isoformat(),
            "age_days": None,
            "allowed": False,
            "reasons": ["signal_not_found"],
            "signal": {
                "side": "",
                "execution_price_ready": False,
                "price_reference_kind": "",
                "price_reference_source": "",
            },
            "levels": {"entry_price": 0.0, "stop_price": 0.0, "target_price": 0.0},
            "sizing": {"equity_usdt": 100000.0, "quote_usdt": 0.0, "risk_budget_usdt": 0.0, "max_alloc_pct": 0.0},
        }

    last = ordered.iloc[-1]
    last_date = pd.Timestamp(last["ts"]).date()
    age_days = max(0, int((as_of - last_date).days))
    if age_days > int(max_age_days):
        return {
            "symbol": symbol,
            "date": last_date.isoformat(),
            "age_days": age_days,
            "allowed": False,
            "reasons": ["stale_signal"],
            "signal": {
                "side": "",
                "execution_price_ready": False,
                "price_reference_kind": "contract_native_daily",
                "price_reference_source": str(last.get("source") or ""),
            },
            "levels": {"entry_price": 0.0, "stop_price": 0.0, "target_price": 0.0},
            "sizing": {"equity_usdt": 100000.0, "quote_usdt": 0.0, "risk_budget_usdt": 0.0, "max_alloc_pct": 0.0},
        }

    closes = ordered["close"].astype(float)
    highs = ordered["high"].astype(float)
    lows = ordered["low"].astype(float)
    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else last_close
    sma20 = float(closes.tail(min(20, len(closes))).mean())
    pullback_pct = (prev_close - last_close) / max(abs(prev_close), 1e-9) if len(closes) >= 2 else 0.0
    rebound_pct = (last_close - prev_close) / max(abs(prev_close), 1e-9) if len(closes) >= 2 else 0.0
    side = ""
    if last_close > sma20 and (last_close >= prev_close or pullback_pct <= 0.01):
        side = "LONG"
    elif last_close < sma20 and (last_close <= prev_close or rebound_pct <= 0.01):
        side = "SHORT"

    if not side:
        return {
            "symbol": symbol,
            "date": last_date.isoformat(),
            "age_days": age_days,
            "allowed": False,
            "reasons": ["signal_not_found"],
            "signal": {
                "side": "",
                "execution_price_ready": False,
                "price_reference_kind": "contract_native_daily",
                "price_reference_source": str(last.get("source") or ""),
            },
            "levels": {"entry_price": 0.0, "stop_price": 0.0, "target_price": 0.0},
            "sizing": {"equity_usdt": 100000.0, "quote_usdt": 0.0, "risk_budget_usdt": 0.0, "max_alloc_pct": 0.0},
        }

    recent_range = max(1.0, float((highs.tail(min(10, len(highs))) - lows.tail(min(10, len(lows)))).mean()))
    entry_price = last_close
    if side == "LONG":
        stop_price = min(float(lows.tail(min(5, len(lows))).min()), entry_price - recent_range * 0.6)
        if stop_price >= entry_price:
            stop_price = entry_price - max(1.0, recent_range * 0.6)
        target_price = entry_price + max(1.0, (entry_price - stop_price) * 2.0)
    else:
        stop_price = max(float(highs.tail(min(5, len(highs))).max()), entry_price + recent_range * 0.6)
        if stop_price <= entry_price:
            stop_price = entry_price + max(1.0, recent_range * 0.6)
        target_price = entry_price - max(1.0, (stop_price - entry_price) * 2.0)

    per_unit_risk = abs(entry_price - stop_price)
    quote_usdt = max(1000.0, min(20000.0, entry_price * 2.0))
    risk_budget = max(50.0, per_unit_risk * (quote_usdt / max(entry_price, 1e-9)))

    return {
        "symbol": symbol,
        "date": last_date.isoformat(),
        "age_days": age_days,
        "allowed": True,
        "reasons": [],
        "signal": {
            "side": side,
            "execution_price_ready": True,
            "price_reference_kind": "contract_native_daily",
            "price_reference_source": str(last.get("source") or ""),
        },
        "levels": {
            "entry_price": round(entry_price, 6),
            "stop_price": round(stop_price, 6),
            "target_price": round(target_price, 6),
        },
        "sizing": {
            "equity_usdt": 100000.0,
            "quote_usdt": round(quote_usdt, 6),
            "risk_budget_usdt": round(risk_budget, 6),
            "max_alloc_pct": 0.05,
        },
    }


def build_signal_bundle(
    *,
    review_dir: Path,
    output_root: Path,
    as_of: dt.date,
    symbols: list[str],
    max_age_days: int,
    enable_state_carry: bool,
    state_carry_max_age_days: int,
) -> dict[str, Any]:
    del output_root, enable_state_carry, state_carry_max_age_days
    tickets: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    missing_symbols: list[str] = []
    stale_symbols: list[str] = []

    for symbol in symbols:
        try:
            frame = fetch_future_daily(symbol, as_of - dt.timedelta(days=90), as_of)
        except Exception:
            frame = pd.DataFrame()
        ticket = build_directional_ticket(symbol=symbol, frame=frame, as_of=as_of, max_age_days=max_age_days)
        tickets.append(ticket)
        signal_rows.append(
            {
                "symbol": symbol,
                "allowed": bool(ticket.get("allowed")),
                "reasons": list(ticket.get("reasons") or []),
                "signal": dict(ticket.get("signal") or {}),
                "levels": dict(ticket.get("levels") or {}),
                "sizing": dict(ticket.get("sizing") or {}),
            }
        )
        reasons = [str(x).strip() for x in ticket.get("reasons", []) if str(x).strip()]
        if "signal_not_found" in reasons:
            missing_symbols.append(symbol)
        if "stale_signal" in reasons:
            stale_symbols.append(symbol)

    stamp = runtime_stamp(as_of)
    signal_json_path = review_dir / f"{stamp}_commodity_directional_signals.json"
    signal_tickets_path = review_dir / f"{stamp}_signal_to_order_tickets.json"
    signal_payload = {
        "generated_at_utc": now_utc_iso_for_date(as_of),
        "as_of": as_of.isoformat(),
        "symbols": symbols,
        "signals": signal_rows,
        "summary": {
            "signal_count": len([row for row in tickets if bool(row.get("allowed"))]),
            "missing_symbols": missing_symbols,
            "stale_symbols": stale_symbols,
        },
    }
    tickets_payload = {
        "generated_at_utc": now_utc_iso_for_date(as_of),
        "as_of": as_of.isoformat(),
        "symbols": symbols,
        "tickets": tickets,
        "summary": {
            "ticket_count": len(tickets),
            "allowed_count": len([row for row in tickets if bool(row.get("allowed"))]),
            "missing_count": len(missing_symbols),
            "stale_count": len(stale_symbols),
        },
    }
    signal_json_path.parent.mkdir(parents=True, exist_ok=True)
    signal_json_path.write_text(json.dumps(signal_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    signal_tickets_path.write_text(json.dumps(tickets_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "ok" if tickets_payload["summary"]["allowed_count"] > 0 else "ok_with_missing"
    return {
        "ok": True,
        "status": status,
        "date": as_of.isoformat(),
        "symbols": symbols,
        "signal_count": tickets_payload["summary"]["allowed_count"],
        "ticket_count": tickets_payload["summary"]["ticket_count"],
        "missing_symbols": missing_symbols,
        "stale_symbols": stale_symbols,
        "json": str(signal_json_path),
        "signal_tickets_json": str(signal_tickets_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build minimal commodity directional signals and signal-to-order tickets.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--max-age-days", type=int, default=14)
    parser.add_argument("--enable-state-carry", action="store_true")
    parser.add_argument("--state-carry-max-age-days", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(str(args.output_root).strip()).expanduser().resolve() if str(args.output_root).strip() else review_dir.parent
    review_dir.mkdir(parents=True, exist_ok=True)
    symbols = [str(x).strip().upper() for x in str(args.symbols or "").split(",") if str(x).strip()]
    payload = build_signal_bundle(
        review_dir=review_dir,
        output_root=output_root,
        as_of=parse_date(args.date),
        symbols=symbols,
        max_age_days=max(1, int(args.max_age_days)),
        enable_state_carry=bool(args.enable_state_carry),
        state_carry_max_age_days=max(0, int(args.state_carry_max_age_days)),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
