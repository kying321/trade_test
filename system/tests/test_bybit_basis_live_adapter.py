from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_adapter_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "bybit_basis_live_adapter.py"
    spec = importlib.util.spec_from_file_location("bybit_basis_live_adapter", mod_path)
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


def _open_position(*, mod, tmp_path: Path, spot_responses: list[dict[str, str] | Exception], perp_responses: list[dict[str, str] | Exception]):
    spot = _FakeSpotClient(spot_responses)
    perp = _FakePerpClient(perp_responses)
    adapter = mod.BybitBasisLiveAdapter(output_root=tmp_path, spot_client=spot, perp_client=perp)
    adapter.execute_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-open",
        requested_notional_usdt=160.0,
        tv_timestamp="2026-03-26T12:32:00Z",
    )
    spot.calls.clear()
    perp.calls.clear()
    return adapter, spot, perp


def test_bybit_adapter_executes_spot_buy_then_perp_short(tmp_path: Path) -> None:
    mod = _load_adapter_module()
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-1",
                "executedQty": "0.00200",
                "cummulativeQuoteQty": "141.0",
                "status": "FILLED",
            }
        ]
    )
    perp = _FakePerpClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-1",
                "executedQty": "0.00200",
                "avgPrice": "70600.0",
                "status": "FILLED",
            }
        ]
    )
    adapter = mod.BybitBasisLiveAdapter(output_root=tmp_path, spot_client=spot, perp_client=perp)

    result = adapter.execute_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-1",
        requested_notional_usdt=160.0,
        tv_timestamp="2026-03-26T12:30:00Z",
    )

    assert result["status"] == "open_hedged"
    assert [call["side"] for call in spot.calls] == ["BUY"]
    assert [call["side"] for call in perp.calls] == ["SELL"]
    assert spot.calls[0]["quantity"] == pytest.approx(0.002)
    assert perp.calls[0]["quantity"] == pytest.approx(0.002)
    assert result["position"]["execution_venue"] == "bybit"


def test_bybit_adapter_enters_needs_recovery_when_second_leg_rejects(tmp_path: Path) -> None:
    mod = _load_adapter_module()
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-2",
                "executedQty": "0.00200",
                "cummulativeQuoteQty": "141.0",
                "status": "FILLED",
            }
        ]
    )
    perp = _FakePerpClient([RuntimeError("perp_rejected")])
    adapter = mod.BybitBasisLiveAdapter(output_root=tmp_path, spot_client=spot, perp_client=perp)

    result = adapter.execute_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-2",
        requested_notional_usdt=160.0,
        tv_timestamp="2026-03-26T12:31:00Z",
    )

    assert result["status"] == "needs_recovery"
    assert result["recovery"]["reason"] == "perp_short_rejected"
    assert result["recovery"]["execution_venue"] == "bybit"

    recoveries = _read_json(tmp_path / "state" / "tv_basis_arb_recovery.json")
    recovery = next(iter(recoveries["recoveries"].values()))
    assert recovery["reason"] == "perp_short_rejected"
    assert recovery["execution_venue"] == "bybit"


def test_bybit_adapter_executes_perp_close_then_spot_sell(tmp_path: Path) -> None:
    mod = _load_adapter_module()
    adapter, spot, perp = _open_position(
        mod=mod,
        tmp_path=tmp_path,
        spot_responses=[
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry-3",
                "executedQty": "0.00200",
                "cummulativeQuoteQty": "141.0",
                "status": "FILLED",
            },
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-exit-3",
                "executedQty": "0.00200",
                "cummulativeQuoteQty": "141.10",
                "status": "FILLED",
            },
        ],
        perp_responses=[
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-entry-3",
                "executedQty": "0.00200",
                "avgPrice": "70600.0",
                "status": "FILLED",
            },
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-exit-3",
                "executedQty": "0.00200",
                "avgPrice": "70550.0",
                "status": "FILLED",
            },
        ],
    )
    result = adapter.execute_exit(idempotency_key="entry-open", close_reason="basis_compressed")

    assert result["status"] == "closed"
    assert [call["side"] for call in perp.calls] == ["BUY"]
    assert [call["side"] for call in spot.calls] == ["SELL"]
    assert perp.calls[0]["reduce_only"] is True
    assert spot.calls[0]["quantity"] == pytest.approx(0.002)

    positions = _read_json(tmp_path / "state" / "tv_basis_arb_positions.json")
    position = next(iter(positions["positions"].values()))
    assert position["status"] == "closed"
    assert position["execution_venue"] == "bybit"
