from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_gate_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "tv_basis_arb_gate.py"
    spec = importlib.util.spec_from_file_location("tv_basis_arb_gate", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_snapshot() -> dict[str, float | str]:
    return {
        "symbol": "BTCUSDT",
        "spot_price": 100_000.0,
        "perp_mark_price": 100_120.0,
        "perp_index_price": 100_100.0,
        "open_interest_contracts": 1_200.0,
        "open_interest_usdt": 120_144_000.0,
        "funding_rate_8h": 0.0001,
        "snapshot_ts_utc": "2026-03-26T12:30:00Z",
    }


def _exchange_blocked_snapshot() -> dict[str, float | str | dict[str, dict[str, float]]]:
    return {
        **_base_snapshot(),
        "exchange_constraints": {
            "spot": {
                "min_qty": 0.00001,
                "step_size": 0.00001,
                "min_notional": 5.0,
            },
            "perp": {
                "min_qty": 0.001,
                "step_size": 0.001,
                "min_notional": 100.0,
            },
        },
    }


class _FakeSpotClient:
    def ticker_snapshot(self, symbol: str) -> dict[str, float | str]:
        return {
            "symbol": symbol,
            "price": 100_000.0,
            "snapshot_time_ms": 1_774_534_200_000,
            "snapshot_ts_utc": "2026-03-26T12:30:00Z",
        }

    def exchange_info(self, symbol: str) -> dict[str, object]:
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "LOT_SIZE", "minQty": "0.00001", "stepSize": "0.00001"},
                        {"filterType": "NOTIONAL", "minNotional": "5"},
                    ],
                }
            ]
        }


class _FakePerpClient:
    def mark_index_funding_snapshot(self, symbol: str) -> dict[str, float | str]:
        return {
            "symbol": symbol,
            "mark_price": 100_120.0,
            "index_price": 100_100.0,
            "funding_rate_8h": 0.0001,
            "next_funding_time_ms": 0,
            "snapshot_time_ms": 1_774_534_200_000,
            "snapshot_ts_utc": "2026-03-26T12:30:00Z",
        }

    def open_interest_snapshot(self, symbol: str) -> dict[str, float | str]:
        return {
            "symbol": symbol,
            "open_interest_contracts": 1_200.0,
            "snapshot_time_ms": 1_774_534_200_000,
            "snapshot_ts_utc": "2026-03-26T12:30:00Z",
        }

    def exchange_info(self, symbol: str) -> dict[str, object]:
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "100"},
                    ],
                }
            ]
        }


def test_gate_passes_when_basis_vol_oi_all_green() -> None:
    mod = _load_gate_module()
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=10.0,
        market_snapshot=_base_snapshot(),
    )

    assert bool(result["passed"]) is True
    assert result["reasons"] == []
    assert float(result["requested_notional_usdt"]) == 10.0
    assert float(result["max_notional_usdt"]) == 20.0
    assert float(result["basis_bps"]) > float(result["thresholds"]["min_basis_bps"])
    assert float(result["mark_index_spread_bps"]) < float(result["thresholds"]["max_mark_index_spread_bps"])
    assert "volatility_bps" not in result


def test_gate_blocks_when_basis_below_threshold() -> None:
    mod = _load_gate_module()
    snapshot = _base_snapshot()
    snapshot["perp_mark_price"] = 100_030.0
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=10.0,
        market_snapshot=snapshot,
    )

    assert bool(result["passed"]) is False
    assert "basis_below_threshold" in result["reasons"]


def test_gate_blocks_when_mark_index_spread_above_threshold() -> None:
    mod = _load_gate_module()
    snapshot = _base_snapshot()
    snapshot["perp_index_price"] = 99_900.0
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=10.0,
        market_snapshot=snapshot,
    )

    assert bool(result["passed"]) is False
    assert "mark_index_spread_above_threshold" in result["reasons"]
    assert float(result["mark_index_spread_bps"]) > float(result["thresholds"]["max_mark_index_spread_bps"])


def test_gate_blocks_when_oi_below_threshold() -> None:
    mod = _load_gate_module()
    snapshot = _base_snapshot()
    snapshot["open_interest_usdt"] = 5_000_000.0
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=10.0,
        market_snapshot=snapshot,
    )

    assert bool(result["passed"]) is False
    assert "open_interest_below_threshold" in result["reasons"]


def test_gate_blocks_when_requested_notional_exceeds_cap() -> None:
    mod = _load_gate_module()
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=25.0,
        market_snapshot=_base_snapshot(),
    )

    assert bool(result["passed"]) is False
    assert "requested_notional_above_cap" in result["reasons"]
    assert float(result["requested_notional_usdt"]) == 25.0
    assert float(result["max_notional_usdt"]) == 20.0


def test_gate_blocks_when_requested_notional_is_negative() -> None:
    mod = _load_gate_module()
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=-1.0,
        market_snapshot=_base_snapshot(),
    )

    assert bool(result["passed"]) is False
    assert "requested_notional_invalid" in result["reasons"]
    assert float(result["requested_notional_usdt"]) == -1.0


def test_gate_blocks_when_requested_notional_is_not_numeric() -> None:
    mod = _load_gate_module()
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt="bad-input",
        market_snapshot=_base_snapshot(),
    )

    assert bool(result["passed"]) is False
    assert "requested_notional_invalid" in result["reasons"]
    assert result["requested_notional_usdt"] is None


def test_gate_blocks_when_exchange_constraints_make_20usdt_unexecutable() -> None:
    mod = _load_gate_module()
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=20.0,
        market_snapshot=_exchange_blocked_snapshot(),
    )

    assert bool(result["passed"]) is False
    assert "perp_min_qty_unmet" in result["reasons"]
    assert "perp_min_notional_unmet" in result["reasons"]
    assert float(result["estimated_base_qty"]) == pytest.approx(0.0002)
    assert float(result["estimated_perp_notional_usdt"]) == pytest.approx(20.024)


def test_build_market_snapshot_includes_exchange_constraints() -> None:
    mod = _load_gate_module()
    snapshot = mod.build_market_snapshot(
        symbol="BTCUSDT",
        spot_client=_FakeSpotClient(),
        perp_client=_FakePerpClient(),
    )

    assert snapshot["exchange_constraints"]["spot"]["min_notional"] == 5.0
    assert snapshot["exchange_constraints"]["perp"]["min_qty"] == 0.001
    assert snapshot["exchange_constraints"]["perp"]["min_notional"] == 100.0


def test_strategy_config_is_single_sourced_from_common_contract() -> None:
    mod = _load_gate_module()
    common = mod.load_strategy_definition("tv_basis_btc_spot_perp_v1")

    assert common["symbol"] == "BTCUSDT"
    gate = common["gate"]
    assert gate["max_notional_usdt"] == 20.0
    assert "max_mark_index_spread_bps" in gate
    assert "max_volatility_bps" not in gate
