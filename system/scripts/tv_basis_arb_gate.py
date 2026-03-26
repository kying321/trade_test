#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binance_live_common import BinanceSpotClient, BinanceUsdMMarketClient, to_float
from tv_basis_arb_common import load_strategy_definition


def _symbol_for_strategy(strategy_id: str) -> str:
    strategy = load_strategy_definition(strategy_id)
    symbol = str(strategy.get("symbol", "")).upper()
    if not symbol:
        raise ValueError(f"missing symbol config:{strategy_id}")
    return symbol


def _strategy_gate_config(strategy_id: str) -> dict[str, float]:
    strategy = load_strategy_definition(strategy_id)
    gate = strategy.get("gate")
    if not isinstance(gate, dict):
        raise ValueError(f"missing gate config:{strategy_id}")
    required = (
        "min_basis_bps",
        "max_mark_index_spread_bps",
        "min_open_interest_usdt",
        "max_notional_usdt",
    )
    missing = [key for key in required if key not in gate]
    if missing:
        raise ValueError(f"missing gate config keys:{strategy_id}:{','.join(missing)}")
    return {key: float(gate[key]) for key in required}


def _basis_bps(spot_price: float, perp_mark_price: float) -> float:
    if spot_price <= 0.0:
        return 0.0
    return ((perp_mark_price - spot_price) / spot_price) * 10_000.0


def _mark_index_spread_bps(perp_mark_price: float, perp_index_price: float) -> float:
    if perp_index_price <= 0.0:
        return 0.0
    return abs((perp_mark_price - perp_index_price) / perp_index_price) * 10_000.0


def _parse_requested_notional(requested_notional_usdt: Any) -> tuple[float | None, bool]:
    if isinstance(requested_notional_usdt, bool):
        return None, True
    try:
        value = float(requested_notional_usdt)
    except Exception:
        return None, True
    if value < 0.0:
        return value, True
    return value, False


def _pick_filter(filters: Any, *names: str) -> dict[str, Any]:
    if not isinstance(filters, list):
        return {}
    want = {str(name).upper() for name in names}
    for row in filters:
        if not isinstance(row, dict):
            continue
        if str(row.get("filterType", "")).upper() in want:
            return row
    return {}


def _spot_exchange_constraints(exchange_info: Any) -> dict[str, float]:
    symbols = exchange_info.get("symbols", []) if isinstance(exchange_info, dict) else []
    row = symbols[0] if symbols else {}
    filters = row.get("filters", []) if isinstance(row, dict) else []
    lot = _pick_filter(filters, "LOT_SIZE")
    notional = _pick_filter(filters, "NOTIONAL", "MIN_NOTIONAL")
    return {
        "min_qty": to_float(lot.get("minQty", 0.0), 0.0),
        "step_size": to_float(lot.get("stepSize", 0.0), 0.0),
        "min_notional": to_float(notional.get("minNotional", notional.get("notional", 0.0)), 0.0),
    }


def _perp_exchange_constraints(exchange_info: Any) -> dict[str, float]:
    symbols = exchange_info.get("symbols", []) if isinstance(exchange_info, dict) else []
    row = symbols[0] if symbols else {}
    filters = row.get("filters", []) if isinstance(row, dict) else []
    market_lot = _pick_filter(filters, "MARKET_LOT_SIZE")
    lot = _pick_filter(filters, "LOT_SIZE")
    chosen = market_lot or lot
    notional = _pick_filter(filters, "MIN_NOTIONAL", "NOTIONAL")
    return {
        "min_qty": to_float(chosen.get("minQty", 0.0), 0.0),
        "step_size": to_float(chosen.get("stepSize", 0.0), 0.0),
        "min_notional": to_float(notional.get("notional", notional.get("minNotional", 0.0)), 0.0),
    }


def build_market_snapshot(
    *,
    symbol: str,
    spot_client: BinanceSpotClient | None = None,
    perp_client: BinanceUsdMMarketClient | None = None,
) -> dict[str, Any]:
    spot = spot_client if spot_client is not None else BinanceSpotClient(api_key="", api_secret="")
    perp = perp_client if perp_client is not None else BinanceUsdMMarketClient()
    spot_row = spot.ticker_snapshot(symbol)
    spot_exchange_info = spot.exchange_info(symbol)
    perp_row = perp.mark_index_funding_snapshot(symbol)
    oi_row = perp.open_interest_snapshot(symbol)
    perp_exchange_info = perp.exchange_info(symbol)

    spot_price = to_float(spot_row.get("price", 0.0), 0.0)
    perp_mark_price = to_float(perp_row.get("mark_price", 0.0), 0.0)
    perp_index_price = to_float(perp_row.get("index_price", 0.0), 0.0)
    open_interest_contracts = to_float(oi_row.get("open_interest_contracts", 0.0), 0.0)
    open_interest_usdt = open_interest_contracts * perp_mark_price
    snapshot_time_ms = max(
        int(to_float(spot_row.get("snapshot_time_ms", 0), 0.0)),
        int(to_float(perp_row.get("snapshot_time_ms", 0), 0.0)),
        int(to_float(oi_row.get("snapshot_time_ms", 0), 0.0)),
    )
    snapshot_ts_utc = (
        perp_row.get("snapshot_ts_utc")
        or oi_row.get("snapshot_ts_utc")
        or spot_row.get("snapshot_ts_utc")
    )

    return {
        "symbol": str(symbol).upper(),
        "spot_price": spot_price,
        "perp_mark_price": perp_mark_price,
        "perp_index_price": perp_index_price,
        "funding_rate_8h": to_float(perp_row.get("funding_rate_8h", 0.0), 0.0),
        "next_funding_time_ms": int(to_float(perp_row.get("next_funding_time_ms", 0), 0.0)),
        "open_interest_contracts": open_interest_contracts,
        "open_interest_usdt": open_interest_usdt,
        "snapshot_time_ms": snapshot_time_ms,
        "snapshot_ts_utc": snapshot_ts_utc,
        "exchange_constraints": {
            "spot": _spot_exchange_constraints(spot_exchange_info),
            "perp": _perp_exchange_constraints(perp_exchange_info),
        },
    }


def evaluate_tv_basis_gate(
    *,
    strategy_id: str,
    requested_notional_usdt: float,
    market_snapshot: dict[str, Any],
) -> dict[str, Any]:
    cfg = _strategy_gate_config(strategy_id)
    symbol = _symbol_for_strategy(strategy_id)
    snapshot_symbol = str(market_snapshot.get("symbol") or symbol).upper()
    if snapshot_symbol != symbol:
        raise ValueError(f"symbol mismatch:{snapshot_symbol}")

    spot_price = to_float(market_snapshot.get("spot_price", 0.0), 0.0)
    perp_mark_price = to_float(market_snapshot.get("perp_mark_price", 0.0), 0.0)
    perp_index_price = to_float(market_snapshot.get("perp_index_price", 0.0), 0.0)
    funding_rate_8h = to_float(market_snapshot.get("funding_rate_8h", 0.0), 0.0)
    open_interest_contracts = to_float(market_snapshot.get("open_interest_contracts", 0.0), 0.0)
    open_interest_usdt = to_float(
        market_snapshot.get("open_interest_usdt", open_interest_contracts * perp_mark_price),
        open_interest_contracts * perp_mark_price,
    )
    requested, requested_invalid = _parse_requested_notional(requested_notional_usdt)
    max_notional_usdt = to_float(cfg.get("max_notional_usdt", 20.0), 20.0)
    exchange_constraints = market_snapshot.get("exchange_constraints", {})
    spot_constraints = exchange_constraints.get("spot", {}) if isinstance(exchange_constraints, dict) else {}
    perp_constraints = exchange_constraints.get("perp", {}) if isinstance(exchange_constraints, dict) else {}

    basis_bps = _basis_bps(spot_price, perp_mark_price)
    mark_index_spread_bps = _mark_index_spread_bps(perp_mark_price, perp_index_price)
    estimated_base_qty = 0.0
    estimated_perp_notional_usdt = 0.0
    if requested is not None and spot_price > 0.0:
        estimated_base_qty = float(requested) / float(spot_price)
        estimated_perp_notional_usdt = float(estimated_base_qty) * float(perp_mark_price)

    reasons: list[str] = []
    if basis_bps < to_float(cfg.get("min_basis_bps", 0.0), 0.0):
        reasons.append("basis_below_threshold")
    if mark_index_spread_bps > to_float(cfg.get("max_mark_index_spread_bps", 0.0), 0.0):
        reasons.append("mark_index_spread_above_threshold")
    if open_interest_usdt < to_float(cfg.get("min_open_interest_usdt", 0.0), 0.0):
        reasons.append("open_interest_below_threshold")
    if requested_invalid:
        reasons.append("requested_notional_invalid")
    if requested is not None and requested > max_notional_usdt:
        reasons.append("requested_notional_above_cap")
    spot_min_notional = to_float(spot_constraints.get("min_notional", 0.0), 0.0)
    perp_min_qty = to_float(perp_constraints.get("min_qty", 0.0), 0.0)
    perp_min_notional = to_float(perp_constraints.get("min_notional", 0.0), 0.0)
    if requested is not None and spot_min_notional > 0.0 and float(requested) < spot_min_notional:
        reasons.append("spot_min_notional_unmet")
    if requested is not None and perp_min_qty > 0.0 and estimated_base_qty < perp_min_qty:
        reasons.append("perp_min_qty_unmet")
    if requested is not None and perp_min_notional > 0.0 and estimated_perp_notional_usdt < perp_min_notional:
        reasons.append("perp_min_notional_unmet")

    return {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "passed": not reasons,
        "reasons": reasons,
        "requested_notional_usdt": requested,
        "max_notional_usdt": max_notional_usdt,
        "estimated_base_qty": estimated_base_qty,
        "estimated_perp_notional_usdt": estimated_perp_notional_usdt,
        "basis_bps": basis_bps,
        "mark_index_spread_bps": mark_index_spread_bps,
        "open_interest_usdt": open_interest_usdt,
        "open_interest_contracts": open_interest_contracts,
        "spot_price": spot_price,
        "perp_mark_price": perp_mark_price,
        "perp_index_price": perp_index_price,
        "funding_rate_8h": funding_rate_8h,
        "exchange_constraints": {
            "spot": {
                "min_qty": to_float(spot_constraints.get("min_qty", 0.0), 0.0),
                "step_size": to_float(spot_constraints.get("step_size", 0.0), 0.0),
                "min_notional": spot_min_notional,
            },
            "perp": {
                "min_qty": perp_min_qty,
                "step_size": to_float(perp_constraints.get("step_size", 0.0), 0.0),
                "min_notional": perp_min_notional,
            },
        },
        "snapshot_ts_utc": market_snapshot.get("snapshot_ts_utc"),
        "snapshot_time_ms": int(to_float(market_snapshot.get("snapshot_time_ms", 0), 0.0)),
        "thresholds": {
            "min_basis_bps": to_float(cfg.get("min_basis_bps", 0.0), 0.0),
            "max_mark_index_spread_bps": to_float(cfg.get("max_mark_index_spread_bps", 0.0), 0.0),
            "min_open_interest_usdt": to_float(cfg.get("min_open_interest_usdt", 0.0), 0.0),
            "max_notional_usdt": max_notional_usdt,
        },
    }
