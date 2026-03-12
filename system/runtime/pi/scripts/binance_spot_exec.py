#!/usr/bin/env python3
"""Binance Spot execution layer (ETH/USDT only).

Goals:
- Deterministic, scriptable, auditable.
- Enforce guardrails:
  - symbol whitelist (default ETHUSDT)
  - max_notional_usdt <= min(5, 0.10 * free_usdt)
  - max_daily_drawdown_usdt = 2 (placeholder: requires realized pnl tracking)
- Pre-check Binance filters (minNotional, stepSize, tickSize).
- Safe by default: DRY RUN unless --live.

Notes:
- This script does NOT print secrets.
- It writes a small JSON event line to stdout for cron capture.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hmac
import hashlib
import json
import math
import os
import time
import urllib.parse
from typing import Any, Dict, Tuple

import requests
from net_resilience import get_proxy_bypass_stats
from cortex_gate import SpinalReflexReject, cortex_gated
from net_resilience import request_no_proxy as _net_request_no_proxy
from net_resilience import reset_proxy_bypass_stats
from net_resilience import request_with_proxy_bypass as _net_request_with_proxy_bypass

PUBLIC_BASES = [
    os.getenv("BINANCE_PUBLIC_BASE_URL", "https://api.binance.com"),
    "https://data-api.binance.vision",  # public mirror (no signed endpoints)
    "https://api1.binance.com",
    "https://api.binance.com",
]

# NOTE: In some regions Binance blocks signed/private endpoints (HTTP 451).
# For DRY-RUN safety we do NOT require private endpoints unless --live or
# --probe-order-test is requested.
PRIVATE_BASES = [
    os.getenv("BINANCE_PRIVATE_BASE_URL", os.getenv("BINANCE_PUBLIC_BASE_URL", "https://api.binance.com")),
    "https://api1.binance.com",
    "https://api.binance.com",
]
SYMBOL_DEFAULT = "ETHUSDT"
MAX_ORDER_NOTIONAL = float(os.getenv("CORTEX_MAX_ORDER_NOTIONAL", "5"))
MAX_ORDER_QTY = float(os.getenv("CORTEX_MAX_ORDER_QTY", "0.02"))


def _env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        # Fallback to manual parsing for cron environments
        env_path = "/Users/jokenrobot/.openclaw/.env"
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            os.environ.setdefault(parts[0], parts[1].strip("'\""))
        v = os.getenv(name)
    if not v:
        raise SystemExit(f"missing env {name}")
    return v


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def should_simulate_probe_on_private_unavailable(*, probe_order_test: bool, live: bool) -> bool:
    if live:
        return False
    if not probe_order_test:
        return False
    # Default enabled to keep paper-probe alive under regional endpoint blocks.
    return _env_flag("BINANCE_PROBE_SIMULATE_ON_PRIVATE_UNAVAILABLE", default=True)


def build_simulated_probe_order_result(*, error: str, base_pub: str) -> Dict[str, Any]:
    return {
        "http": 200,
        "endpoint": "simulated/order/test",
        "decision": "simulate",
        "mode": "paper_probe_fallback",
        "reason": "private_endpoint_unavailable",
        "error": error,
        "base": base_pub,
    }


def choose_base(bases: list[str]) -> str:
    last_err: Exception | None = None
    for b in bases:
        try:
            _ = get_server_time_ms(b)
            return b
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"binance endpoint unavailable: {type(last_err).__name__}: {str(last_err)[:200]}")


def _request_no_proxy(method: str, url: str, **kwargs: Any) -> requests.Response:
    return _net_request_no_proxy(method, url, **kwargs)


def _request_with_proxy_bypass(method: str, url: str, **kwargs: Any) -> requests.Response:
    return _net_request_with_proxy_bypass(
        method,
        url,
        no_proxy_request_func=_request_no_proxy,
        **kwargs,
    )


def _emit_event(event: Dict[str, Any]) -> None:
    payload = dict(event)
    payload["net_proxy_bypass"] = get_proxy_bypass_stats()
    print(json.dumps(payload, ensure_ascii=False))


def get_server_time_ms(base: str) -> int:
    r = _request_with_proxy_bypass("GET", base + "/api/v3/time", timeout=10)
    r.raise_for_status()
    return int(r.json()["serverTime"])


def signed_request(method: str, base: str, path: str, api_key: str, api_secret: str, params: Dict[str, Any]) -> requests.Response:
    qs = urllib.parse.urlencode(params, doseq=True)
    sig = hmac.new(api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{base}{path}?{qs}&signature={sig}"
    headers = {"X-MBX-APIKEY": api_key}
    method_u = method.upper()
    if method_u not in {"GET", "POST", "DELETE"}:
        raise ValueError(method)
    return _request_with_proxy_bypass(method_u, url, headers=headers, timeout=15)


def get_price(base: str, symbol: str) -> float:
    r = _request_with_proxy_bypass(
        "GET",
        base + "/api/v3/ticker/price",
        params={"symbol": symbol},
        timeout=10,
    )
    r.raise_for_status()
    return float(r.json()["price"])


def get_exchange_info(base: str, symbol: str) -> Dict[str, Any]:
    r = _request_with_proxy_bypass(
        "GET",
        base + "/api/v3/exchangeInfo",
        params={"symbol": symbol},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("symbols"):
        raise RuntimeError("exchangeInfo missing symbols")
    return data["symbols"][0]


def parse_filters(sym_info: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for f in sym_info.get("filters", []):
        t = f.get("filterType")
        if t:
            out[t] = f
    return out


def quantize_down(x: float, step: float) -> float:
    return math.floor(x / step) * step


def infer_step_size(filters: Dict[str, Any]) -> Tuple[float, float]:
    lot = filters.get("LOT_SIZE") or {}
    step = float(lot.get("stepSize", "0"))
    min_qty = float(lot.get("minQty", "0"))
    if step <= 0:
        raise RuntimeError("missing LOT_SIZE.stepSize")
    return step, min_qty


def infer_tick_size(filters: Dict[str, Any]) -> float:
    pf = filters.get("PRICE_FILTER") or {}
    tick = float(pf.get("tickSize", "0"))
    if tick <= 0:
        # Some symbols may use other filters; but ETHUSDT should have tick.
        raise RuntimeError("missing PRICE_FILTER.tickSize")
    return tick


def infer_min_notional(filters: Dict[str, Any]) -> float:
    mn = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
    # Binance has evolved filters; try common keys
    for k in ("minNotional", "notional"):
        if k in mn:
            try:
                return float(mn[k])
            except Exception:
                pass
    # fallback conservative
    return 5.0


def fetch_account(base: str, api_key: str, api_secret: str, ts_ms: int) -> Dict[str, Any]:
    r = signed_request(
        "GET",
        base,
        "/api/v3/account",
        api_key,
        api_secret,
        {"timestamp": ts_ms, "recvWindow": 5000},
    )
    r.raise_for_status()
    return r.json()


def free_balance(account: Dict[str, Any], asset: str) -> float:
    for b in account.get("balances", []):
        if b.get("asset") == asset:
            return float(b.get("free", "0"))
    return 0.0


@dataclasses.dataclass
class Plan:
    symbol: str
    side: str  # BUY or SELL
    type: str  # MARKET
    qty: float
    px: float
    notional: float
    reason: str


def build_plan(symbol: str, action: str, px: float, free_usdt: float, free_eth: float, filters: Dict[str, Any]) -> Plan | None:
    """Build a tiny notional plan under strict caps.

    Hard cap (as requested by system policy):
      notional <= min(5 USDT, 10% * free_usdt)

    NOTE: This makes real trades unlikely when free_usdt is small (e.g. ~6 USDT),
    which is intentional for safety.
    """
    # action: LONG_ETH, FLAT_USDT, VOLATILITY_PUMP, LIQUIDITY_TRAP_BUY, LIQUIDITY_TRAP_SELL
    step, min_qty = infer_step_size(filters)
    min_notional = infer_min_notional(filters)

    if action in {"LIQUIDITY_TRAP_BUY", "LIQUIDITY_TRAP_SELL"}:
        side = "BUY" if action == "LIQUIDITY_TRAP_BUY" else "SELL"
        if side == "BUY":
            cap_notional = free_usdt * 0.20
            qty = cap_notional / px
        else:
            qty = free_eth * 0.20
            
        qty = quantize_down(qty, step)
        if qty < max(min_qty, step):
            return None
            
        notional = qty * px
        if notional < min_notional:
            return None
        return Plan(symbol=symbol, side=side, type="MARKET", qty=qty, px=px, notional=notional, reason=f"liquidity_vacuum_devour_{side.lower()}")

    min_notional = infer_min_notional(filters)

    if action == "VOLATILITY_PUMP":
        equity = free_usdt + free_eth * px
        if equity <= 0:
            return None
            
        target_eth_val = equity * 0.50
        current_eth_val = free_eth * px
        
        deviation_threshold = float(os.getenv("LIE_VOLATILITY_PUMP_THRESHOLD", "0.05"))
        drift_pct = abs(current_eth_val - target_eth_val) / equity
        
        if drift_pct < deviation_threshold:
            return None
            
        if current_eth_val < target_eth_val:
            shortfall = target_eth_val - current_eth_val
            cap_notional = min(shortfall, free_usdt * 0.98)
            qty = cap_notional / px
            qty = quantize_down(qty, step)
            if qty < max(min_qty, step):
                return None
            notional = qty * px
            if notional < min_notional:
                return None
            return Plan(symbol=symbol, side="BUY", type="MARKET", qty=qty, px=px, notional=notional, reason=f"vol_pump_buy_{drift_pct*100:.1f}pct")
        else:
            excess = current_eth_val - target_eth_val
            qty = excess / px
            qty = min(qty, free_eth)
            qty = quantize_down(qty, step)
            if qty < max(min_qty, step):
                return None
            notional = qty * px
            if notional < min_notional:
                return None
            return Plan(symbol=symbol, side="SELL", type="MARKET", qty=qty, px=px, notional=notional, reason=f"vol_pump_sell_{drift_pct*100:.1f}pct")


    cap_notional = min(5.0, 0.10 * max(0.0, free_usdt))
    # Leave a tiny buffer so rounding doesn't exceed the cap.
    cap_notional = max(0.0, cap_notional * 0.98)

    if cap_notional < min_notional:
        return None

    if action == "LONG_ETH":
        # BUY market for cap_notional
        qty = cap_notional / px
        qty = quantize_down(qty, step)
        if qty < max(min_qty, step):
            return None
        notional = qty * px
        if notional < min_notional or notional > cap_notional * 1.001:
            return None
        return Plan(symbol=symbol, side="BUY", type="MARKET", qty=qty, px=px, notional=notional, reason="LiE_signal_LONG_ETH")

    if action == "FLAT_USDT":
        # SELL market: flatten, but still obey the same notional cap
        if free_eth <= 0:
            return None
        qty_cap = cap_notional / px
        qty = min(free_eth, qty_cap)
        qty = quantize_down(qty, step)
        if qty < max(min_qty, step):
            return None
        notional = qty * px
        if notional < min_notional or notional > cap_notional * 1.001:
            return None
        return Plan(symbol=symbol, side="SELL", type="MARKET", qty=qty, px=px, notional=notional, reason="LiE_signal_FLAT_USDT")

    raise ValueError(action)


def _simulate_order_result(mode: str, reason: str, debug: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "http": 0,
        "endpoint": "gated/simulate",
        "decision": "simulate",
        "mode": mode,
        "reason": reason,
        "cortex": debug,
    }


def _order_cap_probe(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, float]:
    plan = kwargs.get("plan")
    if plan is None and len(args) >= 5:
        plan = args[4]

    qty = float(getattr(plan, "qty", 0.0) or 0.0) if plan is not None else 0.0
    notional = float(getattr(plan, "notional", 0.0) or 0.0) if plan is not None else 0.0
    if notional <= 0 and plan is not None:
        notional = qty * float(getattr(plan, "px", 0.0) or 0.0)
    return {"qty": abs(qty), "notional": abs(notional)}


def _reduce_only_probe(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    plan = kwargs.get("plan")
    if plan is None and len(args) >= 5:
        plan = args[4]
    if plan is None:
        return False
    side = str(getattr(plan, "side", "") or "").upper()
    return side == "SELL"


def _gate_reject_payload(err: SpinalReflexReject) -> Dict[str, Any]:
    cap = {}
    try:
        cap = dict((err.gate.debug or {}).get("cap_violation") or {})
    except Exception:
        cap = {}
    return {
        "mode": err.gate.mode,
        "policy": err.gate.policy,
        "reason": err.gate.reason,
        "cap_violation": cap or None,
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _dynamic_cap_factor(gate: Any) -> float:
    base_factor = 1.0
    mode = str(getattr(gate, "mode", "") or "")
    debug = getattr(gate, "debug", {}) or {}
    state = debug.get("state", {}) if isinstance(debug, dict) else {}
    try:
        i = float(state.get("I", 1.0))
        r = float(state.get("R", 1.0))
        o = float(state.get("O", 1.0))
    except Exception:
        i, r, o = 1.0, 1.0, 1.0
    core = min(i, r, o)

    if mode == "ACT":
        if core >= 0.85:
            base_factor = 1.0
        elif core >= 0.70:
            base_factor = 0.60
        elif core >= 0.55:
            base_factor = 0.35
        else:
            base_factor = 0.15
    elif mode == "LEARN":
        base_factor = 0.10
    elif mode == "OBSERVE":
        base_factor = 0.05
    else:
        base_factor = 0.02

    mult = float(os.getenv("CORTEX_CAP_MULTIPLIER", "1.0"))
    return _clamp(base_factor * mult, 0.02, 1.0)


def _dynamic_order_cap_limits(gate: Any) -> Dict[str, float]:
    factor = _dynamic_cap_factor(gate)
    return {
        "qty": MAX_ORDER_QTY * factor,
        "notional": MAX_ORDER_NOTIONAL * factor,
    }


@cortex_gated(
    action_class="NORMAL_ACT",
    on_observe="simulate",
    on_learn="simulate",
    on_stabilize="reject",
    simulate_result=_simulate_order_result,
    cap_probe=_order_cap_probe,
    cap_limits=_dynamic_order_cap_limits,
    on_cap_violation="reject",
    reflex_reduce_only_probe=_reduce_only_probe,
)
def place_order(base: str, api_key: str, api_secret: str, ts_ms: int, plan: Plan, live: bool) -> Dict[str, Any]:
    params = {
        "symbol": plan.symbol,
        "side": plan.side,
        "type": plan.type,
        "quantity": f"{plan.qty:.8f}",
        "timestamp": ts_ms,
        "recvWindow": 5000,
    }
    path = "/api/v3/order" if live else "/api/v3/order/test"
    r = signed_request("POST", base, path, api_key, api_secret, params)
    # For test endpoint 200 with {}.
    if r.status_code >= 400:
        return {"http": r.status_code, "body": r.text[:500], "endpoint": path, "base": base}
    try:
        data = r.json() if r.text else {}
    except Exception:
        data = {"raw": r.text[:500]}
    data["http"] = r.status_code
    data["endpoint"] = path
    data["base"] = base
    return data


def main() -> None:
    reset_proxy_bypass_stats()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=SYMBOL_DEFAULT)
    ap.add_argument("--action", choices=["LONG_ETH", "FLAT_USDT", "VOLATILITY_PUMP", "LIQUIDITY_TRAP_BUY", "LIQUIDITY_TRAP_SELL"], required=True)
    ap.add_argument("--live", action="store_true", help="place real order (default: test only)")
    ap.add_argument(
        "--probe-order-test",
        action="store_true",
        help="force a /api/v3/order/test call (may fail filters); for connectivity/auth self-check",
    )
    args = ap.parse_args()

    required_exec_role = str(os.getenv("PI_EXECUTION_ROLE_REQUIRED", "trader")).strip().lower() or "trader"
    exec_role = str(os.getenv("PI_EXECUTION_ROLE", required_exec_role)).strip().lower() or required_exec_role
    if exec_role != required_exec_role:
        event = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": args.symbol,
            "action": args.action,
            "live": bool(args.live),
            "decision": "no-trade",
            "reason": f"execution_role_denied(role={exec_role},required={required_exec_role})",
            "execution_role": exec_role,
            "required_execution_role": required_exec_role,
        }
        _emit_event(event)
        return

    try:
        api_key = _env("BINANCE_API_KEY")
        api_secret = _env("BINANCE_SECRET")
    except SystemExit as e:
        # In dry-run mode, missing API keys should not be treated as a hard error.
        # Degrade to a clean no-trade event so the rest of the pipeline stays healthy.
        if (not args.live) and (not args.probe_order_test):
            event = {
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "symbol": args.symbol,
                "action": args.action,
                "live": bool(args.live),
                "decision": "no-trade",
                "reason": f"{e}",
            }
            _emit_event(event)
            return
        raise

    # Choose working endpoints.
    # - public endpoints can use binance.vision mirror
    # - signed/private endpoints may be blocked (HTTP 451)
    try:
        base_pub = choose_base(PUBLIC_BASES)
    except Exception as e:
        event = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": args.symbol,
            "action": args.action,
            "live": bool(args.live),
            "decision": "error",
            "error": f"binance_public_endpoint_unavailable: {type(e).__name__} {str(e)[:200]}",
        }
        _emit_event(event)
        return

    # For dry-run (default), we don't need private endpoints/account. Degrade cleanly.
    if (not args.live) and (not args.probe_order_test):
        try:
            sym_info = get_exchange_info(base_pub, args.symbol)
            px = get_price(base_pub, args.symbol)
        except Exception as e:
            event = {
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "symbol": args.symbol,
                "action": args.action,
                "live": bool(args.live),
                "decision": "error",
                "error": f"public_api_fetch_failed: {type(e).__name__} {str(e)[:200]}",
            }
            _emit_event(event)
            return

        event: Dict[str, Any] = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": args.symbol,
            "px": px,
            "free_usdt": None,
            "free_eth": None,
            "action": args.action,
            "live": False,
            "base_pub": base_pub,
            "base_priv": None,
            "offset_ms": None,
            "decision": "no-trade",
            "reason": "dry_run_no_private_endpoints",
        }
        _emit_event(event)
        return

    # live/probe mode needs private base
    try:
        base_priv = choose_base(PRIVATE_BASES)
    except Exception as e:
        error_text = f"binance_private_endpoint_unavailable: {type(e).__name__} {str(e)[:200]}"
        if should_simulate_probe_on_private_unavailable(
            probe_order_test=bool(args.probe_order_test),
            live=bool(args.live),
        ):
            px: float | None = None
            try:
                px = get_price(base_pub, args.symbol)
            except Exception:
                px = None
            probe_qty = 0.003
            event: Dict[str, Any] = {
                "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                "symbol": args.symbol,
                "px": px,
                "free_usdt": None,
                "free_eth": None,
                "action": args.action,
                "live": bool(args.live),
                "base_pub": base_pub,
                "base_priv": None,
                "offset_ms": None,
                "decision": "simulate",
                "reason": "probe_private_endpoint_unavailable_simulated",
                "order_plan": {
                    "side": "BUY",
                    "type": "MARKET",
                    "qty": probe_qty,
                    "notional": round(probe_qty * px, 6) if px else None,
                    "reason": "probe_order_test_simulated",
                },
                "order_result": build_simulated_probe_order_result(
                    error=error_text,
                    base_pub=base_pub,
                ),
            }
            _emit_event(event)
            return
        event = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": args.symbol,
            "action": args.action,
            "live": bool(args.live),
            "decision": "error",
            "error": error_text,
        }
        _emit_event(event)
        return

    # time sync (use private base, closer to the auth endpoints)
    server_ms = get_server_time_ms(base_priv)
    local_ms = int(time.time() * 1000)
    offset = server_ms - local_ms
    ts_ms = int(time.time() * 1000) + offset

    try:
        sym_info = get_exchange_info(base_pub, args.symbol)
        filters = parse_filters(sym_info)
        px = get_price(base_pub, args.symbol)
        acct = fetch_account(base_priv, api_key, api_secret, ts_ms)
        free_usdt = free_balance(acct, "USDT")
        
        # [PHASE 8 STAGE 3] Live-Fire Capital Unlocking: Physical 500 USDT ceiling
        live_fire_max = float(os.getenv("LIVE_FIRE_MAX_USDT", "500.0"))
        free_usdt = min(free_usdt, live_fire_max)

        free_eth = free_balance(acct, "ETH")
    except Exception as e:
        event = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "symbol": args.symbol,
            "action": args.action,
            "live": bool(args.live),
            "decision": "error",
            "error": f"api_fetch_failed: {type(e).__name__} {str(e)[:200]}",
        }
        _emit_event(event)
        return

    plan = build_plan(args.symbol, args.action, px, free_usdt, free_eth, filters)

    event: Dict[str, Any] = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "symbol": args.symbol,
        "px": px,
        "free_usdt": round(free_usdt, 6),
        "free_eth": round(free_eth, 6),
        "action": args.action,
        "live": bool(args.live),
        "base_pub": base_pub,
        "base_priv": base_priv,
        "offset_ms": offset,
    }

    # Connectivity/auth probe: always hit /order/test with a tiny quantity.
    if args.probe_order_test:
        # Probe should satisfy minNotional to avoid filter failure.
        # ETHUSDT stepSize is typically 0.001; use >= 0.003 ETH (~>$5) as a safe probe.
        probe_qty = 0.003
        probe_plan = Plan(
            symbol=args.symbol,
            side="BUY",
            type="MARKET",
            qty=probe_qty,
            px=px,
            notional=probe_qty * px,
            reason="probe_order_test",
        )
        event["decision"] = "probe"
        event["order_plan"] = {
            "side": probe_plan.side,
            "type": probe_plan.type,
            "qty": probe_plan.qty,
            "notional": round(probe_plan.notional, 6),
            "reason": probe_plan.reason,
        }
        try:
            event["order_result"] = place_order(base_priv, api_key, api_secret, ts_ms, probe_plan, live=False)
        except SpinalReflexReject as e:
            event["decision"] = "no-trade"
            event["reason"] = f"spinal_reflex_reject:{e.gate.mode.lower()}"
            event["order_result"] = {"http": 0, "endpoint": "gated/reject", **_gate_reject_payload(e)}
        _emit_event(event)
        return

    if not plan:
        event["decision"] = "no-trade"
        event["reason"] = "guardrails_or_filters"
        _emit_event(event)
        return

    event["decision"] = "order"
    event["order_plan"] = {
        "side": plan.side,
        "type": plan.type,
        "qty": float(f"{plan.qty:.8f}"),
        "notional": round(plan.notional, 6),
        "reason": plan.reason,
    }

    try:
        resp = place_order(base_priv, api_key, api_secret, ts_ms, plan, live=args.live)
    except SpinalReflexReject as e:
        event["decision"] = "no-trade"
        event["reason"] = f"spinal_reflex_reject:{e.gate.mode.lower()}"
        event["order_result"] = {"http": 0, "endpoint": "gated/reject", **_gate_reject_payload(e)}
        _emit_event(event)
        return

    if isinstance(resp, dict) and resp.get("decision") == "simulate":
        event["decision"] = "simulate"
    event["order_result"] = resp

    _emit_event(event)


if __name__ == "__main__":
    main()
