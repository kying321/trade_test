#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binance_live_common import RunHalfhourMutex, to_float
from bybit_live_common import resolve_bybit_credentials
from tv_basis_arb_common import load_strategy_definition
from tv_basis_arb_state import TvBasisArbStateLedger, load_recovery_state


class BybitBasisLiveAdapter:
    def __init__(
        self,
        *,
        output_root: Path | str,
        spot_client: Any | None = None,
        perp_client: Any | None = None,
        mutex_timeout_seconds: float = 5.0,
    ) -> None:
        self.output_root = Path(output_root)
        self.ledger = TvBasisArbStateLedger(output_root=self.output_root)
        self.spot_client = spot_client
        self.perp_client = perp_client
        self.mutex_timeout_seconds = min(5.0, max(0.1, float(mutex_timeout_seconds)))

    def _ensure_clients(self) -> tuple[Any, Any]:
        if self.spot_client is None or self.perp_client is None:
            api_key, api_secret, _ = resolve_bybit_credentials(True)
            _ = (api_key, api_secret)
            raise RuntimeError("bybit_live_clients_required")
        return self.spot_client, self.perp_client

    def _run_mutex(self) -> RunHalfhourMutex:
        return RunHalfhourMutex(
            output_root=self.output_root,
            owner="bybit_basis_live_adapter",
            timeout_seconds=self.mutex_timeout_seconds,
        )

    @staticmethod
    def _entry_contract(strategy_id: str) -> tuple[float, float]:
        strategy = load_strategy_definition(strategy_id)
        gate = strategy.get("gate", {})
        if not isinstance(gate, dict):
            raise ValueError(f"missing gate config:{strategy_id}")
        return float(gate.get("target_base_qty", 0.0)), float(gate.get("max_quote_budget_usdt", gate.get("max_notional_usdt", 0.0)))

    @staticmethod
    def _order_id(order: dict[str, Any], fallback: str) -> str:
        for key in ("orderId", "order_id", "orderLinkId", "clientOrderId", "client_order_id"):
            raw = order.get(key)
            if raw not in (None, ""):
                return str(raw)
        return fallback

    @staticmethod
    def _filled_base_qty(order: dict[str, Any]) -> float:
        return to_float(order.get("executedQty", order.get("executed_qty", order.get("cumExecQty", order.get("cum_qty", 0.0)))), 0.0)

    @staticmethod
    def _filled_quote_qty(order: dict[str, Any]) -> float:
        return to_float(order.get("cummulativeQuoteQty", order.get("cumExecValue", order.get("executed_quote_qty", 0.0))), 0.0)

    @staticmethod
    def _avg_price(order: dict[str, Any], *, fallback_quote: float, fallback_base: float) -> float:
        avg = to_float(order.get("avgPrice", order.get("avg_price", 0.0)), 0.0)
        if avg > 0.0:
            return avg
        if fallback_base > 0.0 and fallback_quote > 0.0:
            return fallback_quote / fallback_base
        return 0.0

    def _load_attempt_state(self, idempotency_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        attempt = self.ledger._get_attempt(idempotency_key)
        position = self.ledger._get_position(str(attempt["position_key"]))
        return attempt, position

    def _load_recovery(self, position_key: str) -> dict[str, Any]:
        recoveries = load_recovery_state(self.ledger.recovery_path)["recoveries"]
        recovery = recoveries.get(position_key)
        if not isinstance(recovery, dict):
            raise KeyError(f"unknown recovery:{position_key}")
        return dict(recovery)

    @staticmethod
    def _is_transport_ambiguity(exc: Exception) -> bool:
        return isinstance(exc, (ConnectionError, TimeoutError, OSError))

    def execute_entry(
        self,
        *,
        strategy_id: str,
        symbol: str,
        idempotency_key: str,
        requested_notional_usdt: float,
        tv_timestamp: str,
    ) -> dict[str, Any]:
        spot_client, perp_client = self._ensure_clients()
        target_base_qty, max_quote_budget_usdt = self._entry_contract(strategy_id)
        with self._run_mutex():
            attempt = self.ledger.begin_entry(
                strategy_id=strategy_id,
                symbol=symbol,
                idempotency_key=idempotency_key,
                requested_notional_usdt=requested_notional_usdt,
                target_base_qty=target_base_qty,
                max_quote_budget_usdt=max_quote_budget_usdt,
                execution_venue="bybit",
                tv_timestamp=tv_timestamp,
            )
            if str(attempt.get("status", "")) == "open_hedged":
                _, position = self._load_attempt_state(idempotency_key)
                return {"status": "open_hedged", "attempt": attempt, "position": position}
            if str(attempt.get("status", "")) == "needs_recovery":
                _, position = self._load_attempt_state(idempotency_key)
                recovery = self._load_recovery(str(position["position_key"]))
                return {"status": "needs_recovery", "attempt": attempt, "position": position, "recovery": recovery}

            spot_client_order_id = f"{idempotency_key}-spot-buy"
            self.ledger.record_spot_buy_submitting(idempotency_key=idempotency_key, spot_order_id=spot_client_order_id)
            spot_order = spot_client.place_market_order(
                symbol=symbol,
                side="BUY",
                quantity=float(target_base_qty),
                client_order_id=spot_client_order_id,
            )
            spot_order_id = self._order_id(spot_order, spot_client_order_id)
            spot_base_qty = self._filled_base_qty(spot_order)
            spot_quote_qty = self._filled_quote_qty(spot_order)
            position = self.ledger.record_spot_buy_fill(
                idempotency_key=idempotency_key,
                spot_order_id=spot_order_id,
                filled_base_qty=spot_base_qty,
                filled_quote_usdt=spot_quote_qty,
                partial_fill=spot_base_qty <= 0.0 or abs(spot_base_qty - float(target_base_qty)) > 1e-9,
            )

            perp_client_order_id = f"{idempotency_key}-perp-short"
            try:
                perp_order = perp_client.place_market_order(
                    symbol=symbol,
                    side="SELL",
                    quantity=float(target_base_qty),
                    client_order_id=perp_client_order_id,
                )
            except Exception as exc:
                recovery_reason = "perp_short_rejected"
                recovery_action = "flatten_spot_or_complete_hedge"
                if self._is_transport_ambiguity(exc):
                    recovery_reason = "perp_short_transport_ambiguous"
                    recovery_action = "confirm_perp_then_flatten_or_complete_hedge"
                recovery = self.ledger.record_needs_recovery(
                    idempotency_key=idempotency_key,
                    reason=recovery_reason,
                    failure_phase="perp_short_submitting",
                    recovery_action=recovery_action,
                )
                _, recovery_position = self._load_attempt_state(idempotency_key)
                return {
                    "status": "needs_recovery",
                    "attempt": self.ledger._get_attempt(idempotency_key),
                    "position": recovery_position,
                    "recovery": recovery,
                }

            self.ledger.record_perp_short_submitting(
                idempotency_key=idempotency_key,
                perp_order_id=perp_client_order_id,
                target_base_qty=float(target_base_qty),
            )
            perp_order_id = self._order_id(perp_order, perp_client_order_id)
            perp_base_qty = self._filled_base_qty(perp_order)
            spot_avg_price = self._avg_price(spot_order, fallback_quote=spot_quote_qty, fallback_base=spot_base_qty)
            perp_avg_price = self._avg_price(perp_order, fallback_quote=0.0, fallback_base=perp_base_qty)
            basis_bps = 0.0
            if spot_avg_price > 0.0 and perp_avg_price > 0.0:
                basis_bps = ((perp_avg_price - spot_avg_price) / spot_avg_price) * 10_000.0
            open_position = self.ledger.record_open_hedged(
                idempotency_key=idempotency_key,
                perp_order_id=perp_order_id,
                filled_base_qty=perp_base_qty,
                avg_entry_price=perp_avg_price,
                basis_bps=basis_bps,
            )
            return {
                "status": "open_hedged",
                "attempt": self.ledger._get_attempt(idempotency_key),
                "position": open_position,
                "entry_orders": {"spot": spot_order, "perp": perp_order},
                "pre_open_position": position,
            }

    def execute_exit(self, *, idempotency_key: str, close_reason: str = "") -> dict[str, Any]:
        spot_client, perp_client = self._ensure_clients()
        with self._run_mutex():
            attempt, position = self._load_attempt_state(idempotency_key)
            status = str(attempt.get("status", ""))
            if status == "closed":
                return {"status": "closed", "attempt": attempt, "position": position}
            if status == "needs_recovery":
                recovery = self._load_recovery(str(position["position_key"]))
                return {"status": "needs_recovery", "attempt": attempt, "position": position, "recovery": recovery}
            if status != "open_hedged":
                raise RuntimeError(f"position not open_hedged:{status}")

            exit_position = self.ledger.record_exit_pending(idempotency_key=idempotency_key, reason=close_reason)
            perp_qty = to_float(attempt.get("perp_leg", {}).get("filled_base_qty", 0.0), 0.0)
            spot_qty = to_float(attempt.get("spot_leg", {}).get("filled_base_qty", 0.0), 0.0)

            perp_client_order_id = f"{idempotency_key}-perp-close"
            try:
                perp_order = perp_client.place_market_order(
                    symbol=str(attempt.get("symbol", "")),
                    side="BUY",
                    quantity=perp_qty,
                    client_order_id=perp_client_order_id,
                    reduce_only=True,
                )
            except Exception as exc:
                recovery_reason = "perp_close_rejected"
                recovery_action = "retry_perp_close_then_sell_spot"
                if self._is_transport_ambiguity(exc):
                    recovery_reason = "perp_close_transport_ambiguous"
                    recovery_action = "confirm_perp_close_then_sell_spot"
                recovery = self.ledger.record_needs_recovery(
                    idempotency_key=idempotency_key,
                    reason=recovery_reason,
                    failure_phase="exit_pending",
                    recovery_action=recovery_action,
                )
                _, recovery_position = self._load_attempt_state(idempotency_key)
                return {
                    "status": "needs_recovery",
                    "attempt": self.ledger._get_attempt(idempotency_key),
                    "position": recovery_position,
                    "recovery": recovery,
                    "pre_exit_position": exit_position,
                }

            spot_client_order_id = f"{idempotency_key}-spot-sell"
            spot_order = spot_client.place_market_order(
                symbol=str(attempt.get("symbol", "")),
                side="SELL",
                quantity=spot_qty,
                client_order_id=spot_client_order_id,
            )
            closed_position = self.ledger.record_closed(idempotency_key=idempotency_key, close_reason=close_reason)
            return {
                "status": "closed",
                "attempt": self.ledger._get_attempt(idempotency_key),
                "position": closed_position,
                "exit_orders": {"perp": perp_order, "spot": spot_order},
                "pre_exit_position": exit_position,
            }
