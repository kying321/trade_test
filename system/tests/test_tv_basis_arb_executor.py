from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_executor_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "tv_basis_arb_executor.py"
    spec = importlib.util.spec_from_file_location("tv_basis_arb_executor", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class _FakeSpotClient:
    def __init__(self, responses: list[dict[str, str] | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str,
        quote_order_qty: float | None = None,
    ) -> dict[str, str]:
        self.calls.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "client_order_id": client_order_id,
                "quote_order_qty": quote_order_qty,
            }
        )
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakePerpClient:
    def __init__(self, responses: list[dict[str, str] | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> dict[str, str]:
        self.calls.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "client_order_id": client_order_id,
                "reduce_only": reduce_only,
            }
        )
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _open_position(
    *,
    mod,
    tmp_path: Path,
    spot_responses: list[dict[str, str] | Exception],
    perp_responses: list[dict[str, str] | Exception],
):
    spot = _FakeSpotClient(spot_responses)
    perp = _FakePerpClient(perp_responses)
    executor = mod.TvBasisArbExecutor(output_root=tmp_path, spot_client=spot, perp_client=perp)
    executor.execute_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-open",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:32:00Z",
    )
    spot.calls.clear()
    perp.calls.clear()
    return executor, spot, perp


def test_entry_executes_spot_buy_then_perp_short(tmp_path: Path) -> None:
    mod = _load_executor_module()
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-1",
                "executedQty": "0.00010",
                "cummulativeQuoteQty": "10.0",
                "status": "FILLED",
            }
        ]
    )
    perp = _FakePerpClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-1",
                "executedQty": "0.00010",
                "avgPrice": "100120.0",
                "status": "FILLED",
            }
        ]
    )
    executor = mod.TvBasisArbExecutor(output_root=tmp_path, spot_client=spot, perp_client=perp)

    result = executor.execute_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-1",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:30:00Z",
    )

    assert result["status"] == "open_hedged"
    assert [call["side"] for call in spot.calls] == ["BUY"]
    assert [call["side"] for call in perp.calls] == ["SELL"]
    assert spot.calls[0]["quote_order_qty"] == 10.0
    assert perp.calls[0]["quantity"] == pytest.approx(0.00010)
    assert result["position"]["spot_leg"]["status"] == "filled"
    assert result["position"]["perp_leg"]["status"] == "filled"


def test_spot_fill_then_perp_reject_marks_recovery(tmp_path: Path) -> None:
    mod = _load_executor_module()
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-2",
                "executedQty": "0.00008",
                "cummulativeQuoteQty": "8.0",
                "status": "FILLED",
            }
        ]
    )
    perp = _FakePerpClient([RuntimeError("perp_rejected")])
    executor = mod.TvBasisArbExecutor(output_root=tmp_path, spot_client=spot, perp_client=perp)

    result = executor.execute_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-2",
        requested_notional_usdt=8.0,
        tv_timestamp="2026-03-26T12:31:00Z",
    )

    assert result["status"] == "needs_recovery"
    assert result["recovery"]["reason"] == "perp_short_rejected"
    assert result["recovery"]["failure_phase"] == "perp_short_submitting"
    assert result["recovery"]["spot_leg"]["status"] == "filled"
    assert result["recovery"]["perp_leg"]["status"] == "missing"

    recoveries = _read_json(tmp_path / "state" / "tv_basis_arb_recovery.json")
    recovery = next(iter(recoveries["recoveries"].values()))
    assert recovery["reason"] == "perp_short_rejected"
    assert recovery["spot_leg"]["filled_quote_usdt"] == 8.0


def test_exit_executes_perp_close_then_spot_sell(tmp_path: Path) -> None:
    mod = _load_executor_module()
    executor, spot, perp = _open_position(
        mod=mod,
        tmp_path=tmp_path,
        spot_responses=[
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry-3",
                "executedQty": "0.00010",
                "cummulativeQuoteQty": "10.0",
                "status": "FILLED",
            },
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-exit-3",
                "executedQty": "0.00010",
                "cummulativeQuoteQty": "10.01",
                "status": "FILLED",
            },
        ],
        perp_responses=[
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-entry-3",
                "executedQty": "0.00010",
                "avgPrice": "100120.0",
                "status": "FILLED",
            },
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-exit-3",
                "executedQty": "0.00010",
                "avgPrice": "100050.0",
                "status": "FILLED",
            },
        ],
    )
    result = executor.execute_exit(idempotency_key="entry-open", close_reason="basis_compressed")

    assert result["status"] == "closed"
    assert [call["side"] for call in perp.calls] == ["BUY"]
    assert [call["side"] for call in spot.calls] == ["SELL"]
    assert perp.calls[0]["reduce_only"] is True
    assert spot.calls[0]["quantity"] == pytest.approx(0.00010)

    positions = _read_json(tmp_path / "state" / "tv_basis_arb_positions.json")
    position = next(iter(positions["positions"].values()))
    assert position["status"] == "closed"


def test_exit_first_leg_failure_enters_needs_recovery(tmp_path: Path) -> None:
    mod = _load_executor_module()
    executor, spot, perp = _open_position(
        mod=mod,
        tmp_path=tmp_path,
        spot_responses=[
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry-4",
                "executedQty": "0.00010",
                "cummulativeQuoteQty": "10.0",
                "status": "FILLED",
            }
        ],
        perp_responses=[
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-entry-4",
                "executedQty": "0.00010",
                "avgPrice": "100120.0",
                "status": "FILLED",
            },
            RuntimeError("perp_close_rejected"),
        ],
    )

    result = executor.execute_exit(idempotency_key="entry-open", close_reason="basis_compressed")

    assert result["status"] == "needs_recovery"
    assert result["recovery"]["reason"] == "perp_close_rejected"
    assert result["recovery"]["failure_phase"] == "exit_pending"
    assert spot.calls == []
    assert [call["side"] for call in perp.calls] == ["BUY"]

    positions = _read_json(tmp_path / "state" / "tv_basis_arb_positions.json")
    position = next(iter(positions["positions"].values()))
    assert position["status"] == "needs_recovery"


def test_exit_retry_while_recovery_active_returns_existing_recovery_without_new_orders(tmp_path: Path) -> None:
    mod = _load_executor_module()
    executor, spot, perp = _open_position(
        mod=mod,
        tmp_path=tmp_path,
        spot_responses=[
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry-5",
                "executedQty": "0.00010",
                "cummulativeQuoteQty": "10.0",
                "status": "FILLED",
            }
        ],
        perp_responses=[
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-entry-5",
                "executedQty": "0.00010",
                "avgPrice": "100120.0",
                "status": "FILLED",
            },
            RuntimeError("perp_close_rejected"),
        ],
    )

    first = executor.execute_exit(idempotency_key="entry-open", close_reason="basis_compressed")
    spot.calls.clear()
    perp.calls.clear()

    second = executor.execute_exit(idempotency_key="entry-open", close_reason="basis_compressed")

    assert first["status"] == "needs_recovery"
    assert second["status"] == "needs_recovery"
    assert second["recovery"]["reason"] == "perp_close_rejected"
    assert spot.calls == []
    assert perp.calls == []


def test_transport_ambiguity_is_distinct_from_exchange_reject(tmp_path: Path) -> None:
    mod = _load_executor_module()
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-6",
                "executedQty": "0.00008",
                "cummulativeQuoteQty": "8.0",
                "status": "FILLED",
            }
        ]
    )
    perp = _FakePerpClient([ConnectionError("timeout waiting for response")])
    executor = mod.TvBasisArbExecutor(output_root=tmp_path, spot_client=spot, perp_client=perp)

    result = executor.execute_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-6",
        requested_notional_usdt=8.0,
        tv_timestamp="2026-03-26T12:33:00Z",
    )

    assert result["status"] == "needs_recovery"
    assert result["recovery"]["reason"] == "perp_short_transport_ambiguous"
    assert result["recovery"]["perp_leg"]["status"] == "submitting"
    assert result["recovery"]["perp_leg"]["submission_state"] == "transport_ambiguous"
