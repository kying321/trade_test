from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_gate_blocks_when_volatility_above_threshold() -> None:
    mod = _load_gate_module()
    snapshot = _base_snapshot()
    snapshot["perp_index_price"] = 99_900.0
    result = mod.evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=10.0,
        market_snapshot=snapshot,
    )

    assert bool(result["passed"]) is False
    assert "volatility_above_threshold" in result["reasons"]


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
