#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
import sys
from typing import Any

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binance_live_common import (
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DEFAULT_TIMEOUT_MS,
    BinanceSpotClient,
    PanicTriggered,
    RunHalfhourMutex,
    ensure_parent,
    load_list_ledger,
    now_utc_iso,
    panic_close_all,
    read_json,
    resolve_binance_credentials,
    save_list_ledger,
    to_float,
    write_json,
)


def current_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return raw if isinstance(raw, dict) else {}


def quantize_floor(value: float, step: float) -> float:
    s = max(1e-12, float(step))
    return math.floor(float(value) / s) * s


def infer_lot_constraints(exchange_info: dict[str, Any], symbol: str) -> dict[str, float]:
    out = {
        "step_size": 0.000001,
        "min_qty": 0.000001,
        "min_notional": 5.0,
    }
    symbols = exchange_info.get("symbols", []) if isinstance(exchange_info.get("symbols", []), list) else []
    for row in symbols:
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol", "")).upper() != symbol.upper():
            continue
        filters = row.get("filters", []) if isinstance(row.get("filters", []), list) else []
        for item in filters:
            if not isinstance(item, dict):
                continue
            ftype = str(item.get("filterType", "")).upper()
            if ftype == "LOT_SIZE":
                out["step_size"] = max(1e-12, to_float(item.get("stepSize", out["step_size"]), out["step_size"]))
                out["min_qty"] = max(1e-12, to_float(item.get("minQty", out["min_qty"]), out["min_qty"]))
            if ftype in {"MIN_NOTIONAL", "NOTIONAL"}:
                raw_key = "notional" if "notional" in item else "minNotional"
                out["min_notional"] = max(0.0, to_float(item.get(raw_key, out["min_notional"]), out["min_notional"]))
        break
    return out


def calc_buy_quantity(*, quote_usdt: float, price: float, step_size: float, min_qty: float, min_notional: float) -> float:
    px = max(1e-12, float(price))
    target_quote = max(float(quote_usdt), float(min_notional))
    qty = quantize_floor(target_quote / px, step_size)
    if qty < min_qty:
        qty = quantize_floor(min_qty, step_size)
    if qty * px < min_notional:
        qty = quantize_floor((min_notional / px) + step_size, step_size)
    return max(0.0, qty)


def build_idempotency_key(*, day_key: str, market: str, symbol: str, quote_usdt: float) -> str:
    seed = f"{day_key}:{market.lower()}:{symbol.upper()}:{float(quote_usdt):.4f}:infra_canary_v1"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:28]


def load_budget_state(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    return payload if isinstance(payload, dict) else {}


def load_day_budget(path: Path, day_key: str) -> tuple[dict[str, Any], float]:
    payload = load_budget_state(path)
    days = payload.get("days", {}) if isinstance(payload.get("days", {}), dict) else {}
    day_payload = days.get(day_key, {}) if isinstance(days.get(day_key, {}), dict) else {}
    spent = to_float(day_payload.get("spent_quote_usdt", 0.0), 0.0)
    return payload, max(0.0, spent)


def record_budget_event(path: Path, *, day_key: str, quote_usdt: float, event: dict[str, Any]) -> dict[str, Any]:
    payload = load_budget_state(path)
    days = payload.get("days", {}) if isinstance(payload.get("days", {}), dict) else {}
    day_payload = days.get(day_key, {}) if isinstance(days.get(day_key, {}), dict) else {}
    events = day_payload.get("events", []) if isinstance(day_payload.get("events", []), list) else []
    new_events = list(events)
    new_events.append(event)
    spent = to_float(day_payload.get("spent_quote_usdt", 0.0), 0.0) + max(0.0, float(quote_usdt))
    days[day_key] = {
        "spent_quote_usdt": float(spent),
        "events": new_events[-100:],
        "updated_at_utc": now_utc_iso(),
    }
    payload["days"] = days
    write_json(path, payload)
    return days[day_key]


def summarize_account_ready(account: dict[str, Any], *, symbol: str, quote_usdt: float) -> dict[str, Any]:
    balances = account.get("balances", []) if isinstance(account.get("balances", []), list) else []
    balance_map: dict[str, float] = {}
    for row in balances:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("asset", "")).strip().upper()
        if not asset:
            continue
        balance_map[asset] = to_float(row.get("free", 0.0), 0.0) + to_float(row.get("locked", 0.0), 0.0)
    quote_available = to_float(balance_map.get("USDT", 0.0), 0.0)
    base_asset = symbol[:-4] if symbol.endswith("USDT") else symbol
    base_available = to_float(balance_map.get(base_asset, 0.0), 0.0)
    can_trade = bool(account.get("canTrade", True))
    ready = bool(can_trade and quote_available + 1e-12 >= max(0.0, float(quote_usdt)))
    return {
        "ready": ready,
        "can_trade": can_trade,
        "quote_asset": "USDT",
        "quote_available": float(quote_available),
        "base_asset": base_asset,
        "base_available": float(base_available),
    }


def write_summary(output_root: Path, summary: dict[str, Any]) -> None:
    write_json(output_root / "review" / "latest_binance_infra_canary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent Binance infra canary actor.")
    parser.add_argument("--market", default="spot")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--quote-usdt", type=float, default=5.0)
    parser.add_argument("--daily-budget-cap-usdt", type=float, default=20.0)
    parser.add_argument("--allow-dust", action="store_true")
    parser.add_argument("--mode", choices=["probe", "run", "autopilot-check"], default="probe")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--allow-daemon-env-fallback", action="store_true")
    parser.add_argument("--skip-mutex", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--rate-limit-per-minute", type=int, default=DEFAULT_RATE_LIMIT_PER_MINUTE)
    args = parser.parse_args()

    cwd = Path.cwd()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = cwd / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = cwd / cfg_path

    day_key = current_utc_date().isoformat()
    mode = str(args.mode)
    symbol = str(args.symbol).strip().upper()
    market = str(args.market).strip().lower()
    quote_usdt = max(0.0, float(args.quote_usdt))
    budget_cap = max(0.0, float(args.daily_budget_cap_usdt))

    summary: dict[str, Any] = {
        "ok": True,
        "mode": mode,
        "market": market,
        "symbol": symbol,
        "quote_usdt": quote_usdt,
        "daily_budget_cap_usdt": budget_cap,
        "started_at_utc": now_utc_iso(),
        "config": str(cfg_path),
        "output_root": str(output_root),
        "steps": {},
        "autopilot_allowed": False,
    }

    idempotency_path = output_root / "state" / "infra_canary_idempotency.json"
    budget_path = output_root / "state" / "infra_canary_budget.json"
    owner = f"binance_infra_canary:{day_key}:{mode}:{symbol}"

    try:
        lock_cm = nullcontext() if bool(args.skip_mutex) else RunHalfhourMutex(output_root=output_root, owner=owner, timeout_seconds=5.0)
        with lock_cm:
            _ = load_config(cfg_path)
            api_key, api_secret, cred_source = resolve_binance_credentials(bool(args.allow_daemon_env_fallback))
            summary["steps"]["credentials"] = {
                "source": cred_source,
                "has_api_key": bool(api_key),
                "has_api_secret": bool(api_secret),
            }

            client = BinanceSpotClient(
                api_key=api_key,
                api_secret=api_secret,
                timeout_ms=min(5000, max(100, int(args.timeout_ms))),
                rate_limit_per_minute=max(1, int(args.rate_limit_per_minute)),
            )

            try:
                summary["steps"]["ping"] = {"ok": True, "payload": client.ping()}
                price = to_float(client.ticker_price(symbol), 0.0)
                exchange_info = client.exchange_info(symbol)
                account = client.account()
            except (ConnectionError, TimeoutError, OSError) as exc:
                panic_close_all(output_root, reason="infra_canary_transport_ambiguity", detail=str(exc))

            lot = infer_lot_constraints(exchange_info, symbol)
            planned_qty = calc_buy_quantity(
                quote_usdt=quote_usdt,
                price=price,
                step_size=float(lot["step_size"]),
                min_qty=float(lot["min_qty"]),
                min_notional=float(lot["min_notional"]),
            )
            summary["steps"]["plan"] = {
                "price": float(price),
                "planned_buy_quantity": float(planned_qty),
                "constraints": lot,
            }

            account_ready = summarize_account_ready(account, symbol=symbol, quote_usdt=max(quote_usdt, float(lot["min_notional"])))
            summary["steps"]["account_ready"] = account_ready

            _, spent_today = load_day_budget(budget_path, day_key)
            within_cap = spent_today + quote_usdt <= budget_cap + 1e-12
            summary["steps"]["budget"] = {
                "day_key": day_key,
                "spent_quote_usdt": float(spent_today),
                "pending_quote_usdt": float(quote_usdt),
                "daily_budget_cap_usdt": float(budget_cap),
                "within_cap": bool(within_cap),
            }

            idem_key = build_idempotency_key(day_key=day_key, market=market, symbol=symbol, quote_usdt=quote_usdt)
            recorded_keys = load_list_ledger(idempotency_path, "keys")
            summary["steps"]["idempotency"] = {
                "key": idem_key,
                "skipped": False,
            }

            summary["autopilot_allowed"] = bool(account_ready["ready"] and within_cap and mode in {"probe", "autopilot-check", "run"})
            if mode == "autopilot-check":
                write_summary(output_root, summary)
                return 0

            if not bool(account_ready["ready"]):
                summary["ok"] = False
                summary["autopilot_allowed"] = False
                write_summary(output_root, summary)
                return 2

            if mode == "probe":
                summary["autopilot_allowed"] = bool(within_cap)
                write_summary(output_root, summary)
                return 0

            if not within_cap:
                summary["ok"] = False
                summary["autopilot_allowed"] = False
                summary["steps"]["round_trip"] = {
                    "executed": False,
                    "reason": "daily_budget_exceeded",
                }
                write_summary(output_root, summary)
                return 2

            if idem_key in set(recorded_keys):
                summary["steps"]["idempotency"] = {
                    "key": idem_key,
                    "skipped": True,
                    "reason": "idempotent_skip",
                }
                summary["steps"]["round_trip"] = {
                    "executed": False,
                    "reason": "idempotent_skip",
                }
                write_summary(output_root, summary)
                return 0

            base_asset = str(account_ready["base_asset"])
            buy_client_order_id = f"infra-buy-{idem_key[:16]}"
            try:
                buy_rsp = client.place_market_order(
                    symbol=symbol,
                    side="BUY",
                    quantity=float(planned_qty),
                    client_order_id=buy_client_order_id,
                    quote_order_qty=float(quote_usdt),
                )
            except RuntimeError as exc:
                summary["ok"] = False
                summary["steps"]["round_trip"] = {
                    "executed": False,
                    "reason": "exchange_reject",
                    "error": str(exc),
                }
                write_summary(output_root, summary)
                return 2
            except (ConnectionError, TimeoutError, OSError) as exc:
                panic_close_all(output_root, reason="infra_canary_order_transport_ambiguity", detail=str(exc))

            buy_qty = max(to_float(buy_rsp.get("executedQty", planned_qty), planned_qty), 0.0)
            sell_qty = quantize_floor(buy_qty, float(lot["step_size"]))
            if sell_qty <= 0.0:
                panic_close_all(output_root, reason="infra_canary_invalid_sell_qty", detail=f"buy_qty={buy_qty}")

            sell_client_order_id = f"infra-sell-{idem_key[:15]}"
            try:
                sell_rsp = client.place_market_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=float(sell_qty),
                    client_order_id=sell_client_order_id,
                )
            except RuntimeError as exc:
                summary["ok"] = False
                summary["steps"]["round_trip"] = {
                    "executed": False,
                    "reason": "exchange_reject",
                    "error": str(exc),
                    "phase": "sell",
                }
                write_summary(output_root, summary)
                return 2
            except (ConnectionError, TimeoutError, OSError) as exc:
                panic_close_all(output_root, reason="infra_canary_order_transport_ambiguity", detail=str(exc))

            dust_qty = max(0.0, buy_qty - sell_qty)
            if dust_qty > 1e-12 and not bool(args.allow_dust):
                summary["ok"] = False
                summary["steps"]["round_trip"] = {
                    "executed": False,
                    "reason": "dust_not_allowed",
                    "dust": {"base_asset": base_asset, "base_asset_qty": float(dust_qty)},
                }
                write_summary(output_root, summary)
                return 2

            recorded_keys.append(idem_key)
            save_list_ledger(idempotency_path, key="keys", values=recorded_keys, max_items=10000)
            day_budget = record_budget_event(
                budget_path,
                day_key=day_key,
                quote_usdt=quote_usdt,
                event={
                    "ts_utc": now_utc_iso(),
                    "mode": mode,
                    "market": market,
                    "symbol": symbol,
                    "quote_usdt": float(quote_usdt),
                    "status": "filled",
                },
            )
            summary["steps"]["budget"] = {
                **summary["steps"]["budget"],
                "spent_quote_usdt": float(day_budget.get("spent_quote_usdt", quote_usdt)),
                "within_cap": bool(to_float(day_budget.get("spent_quote_usdt", quote_usdt), quote_usdt) <= budget_cap + 1e-12),
            }
            summary["steps"]["idempotency"] = {
                "key": idem_key,
                "skipped": False,
                "recorded": True,
            }
            summary["steps"]["round_trip"] = {
                "executed": True,
                "orders_submitted": 2,
                "buy": buy_rsp,
                "sell": sell_rsp,
                "dust": {
                    "base_asset": base_asset,
                    "base_asset_qty": float(dust_qty),
                },
            }
            summary["autopilot_allowed"] = False
            write_summary(output_root, summary)
            return 0
    except PanicTriggered as exc:
        summary["ok"] = False
        summary["failure_class"] = "panic"
        summary["error"] = str(exc)
        summary["autopilot_allowed"] = False
        write_summary(output_root, summary)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
