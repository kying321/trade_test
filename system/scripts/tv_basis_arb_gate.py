#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binance_live_common import BinanceSpotClient, BinanceUsdMMarketClient, to_float
from tv_basis_arb_common import STRATEGY_CONFIG

TV_BASIS_GATE_CONFIG: dict[str, dict[str, float | str]] = {
    "tv_basis_btc_spot_perp_v1": {
        "symbol": "BTCUSDT",
        "min_basis_bps": 8.0,
        "max_volatility_bps": 15.0,
        "min_open_interest_usdt": 10_000_000.0,
        "max_notional_usdt": 20.0,
    }
}


def _strategy_gate_config(strategy_id: str) -> dict[str, float | str]:
    cfg = TV_BASIS_GATE_CONFIG.get(str(strategy_id))
    if cfg is None:
        raise ValueError(f"unknown strategy_id:{strategy_id}")
    return cfg


def _symbol_for_strategy(strategy_id: str) -> str:
    base_cfg = STRATEGY_CONFIG.get(str(strategy_id))
    gate_cfg = _strategy_gate_config(strategy_id)
    symbol = str((base_cfg or {}).get("symbol") or gate_cfg.get("symbol") or "").upper()
    if not symbol:
        raise ValueError(f"missing symbol config:{strategy_id}")
    return symbol


def _basis_bps(spot_price: float, perp_mark_price: float) -> float:
    if spot_price <= 0.0:
        return 0.0
    return ((perp_mark_price - spot_price) / spot_price) * 10_000.0


def _volatility_bps(perp_mark_price: float, perp_index_price: float) -> float:
    if perp_index_price <= 0.0:
        return 0.0
    return abs((perp_mark_price - perp_index_price) / perp_index_price) * 10_000.0


def build_market_snapshot(
    *,
    symbol: str,
    spot_client: BinanceSpotClient | None = None,
    perp_client: BinanceUsdMMarketClient | None = None,
) -> dict[str, Any]:
    spot = spot_client if spot_client is not None else BinanceSpotClient(api_key="", api_secret="")
    perp = perp_client if perp_client is not None else BinanceUsdMMarketClient()
    spot_row = spot.ticker_snapshot(symbol)
    perp_row = perp.mark_index_funding_snapshot(symbol)
    oi_row = perp.open_interest_snapshot(symbol)

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
    requested = max(0.0, to_float(requested_notional_usdt, 0.0))
    max_notional_usdt = to_float(cfg.get("max_notional_usdt", 20.0), 20.0)

    basis_bps = _basis_bps(spot_price, perp_mark_price)
    volatility_bps = _volatility_bps(perp_mark_price, perp_index_price)

    reasons: list[str] = []
    if basis_bps < to_float(cfg.get("min_basis_bps", 0.0), 0.0):
        reasons.append("basis_below_threshold")
    if volatility_bps > to_float(cfg.get("max_volatility_bps", 0.0), 0.0):
        reasons.append("volatility_above_threshold")
    if open_interest_usdt < to_float(cfg.get("min_open_interest_usdt", 0.0), 0.0):
        reasons.append("open_interest_below_threshold")
    if requested > max_notional_usdt:
        reasons.append("requested_notional_above_cap")

    return {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "passed": not reasons,
        "reasons": reasons,
        "requested_notional_usdt": requested,
        "max_notional_usdt": max_notional_usdt,
        "basis_bps": basis_bps,
        "volatility_bps": volatility_bps,
        "open_interest_usdt": open_interest_usdt,
        "open_interest_contracts": open_interest_contracts,
        "spot_price": spot_price,
        "perp_mark_price": perp_mark_price,
        "perp_index_price": perp_index_price,
        "funding_rate_8h": funding_rate_8h,
        "snapshot_ts_utc": market_snapshot.get("snapshot_ts_utc"),
        "snapshot_time_ms": int(to_float(market_snapshot.get("snapshot_time_ms", 0), 0.0)),
        "thresholds": {
            "min_basis_bps": to_float(cfg.get("min_basis_bps", 0.0), 0.0),
            "max_volatility_bps": to_float(cfg.get("max_volatility_bps", 0.0), 0.0),
            "min_open_interest_usdt": to_float(cfg.get("min_open_interest_usdt", 0.0), 0.0),
            "max_notional_usdt": max_notional_usdt,
        },
    }
