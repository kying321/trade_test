from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_state_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "tv_basis_arb_state.py"
    spec = importlib.util.spec_from_file_location("tv_basis_arb_state", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_supported_statuses_cover_entry_open_exit_and_recovery_states() -> None:
    mod = _load_state_module()

    assert {
        "entry_pending",
        "spot_buy_submitting",
        "spot_buy_filled_perp_pending",
        "perp_short_submitting",
        "open_hedged",
        "exit_pending",
        "needs_recovery",
        "closed",
    }.issubset(set(mod.SUPPORTED_STATUSES))


def test_entry_flow_creates_open_hedged_position_and_persists_state_files(tmp_path: Path) -> None:
    mod = _load_state_module()
    ledger = mod.TvBasisArbStateLedger(output_root=tmp_path)

    attempt = ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-1",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:30:00Z",
    )
    assert attempt["status"] == "entry_pending"
    ledger.record_spot_buy_submitting(idempotency_key="entry-1", spot_order_id="spot-1")
    ledger.record_spot_buy_fill(
        idempotency_key="entry-1",
        spot_order_id="spot-1",
        filled_base_qty=0.0001,
        filled_quote_usdt=10.0,
    )
    ledger.record_perp_short_submitting(
        idempotency_key="entry-1",
        perp_order_id="perp-1",
        target_base_qty=0.0001,
    )
    position = ledger.record_open_hedged(
        idempotency_key="entry-1",
        perp_order_id="perp-1",
        filled_base_qty=0.0001,
        avg_entry_price=100_100.0,
        basis_bps=12.0,
    )

    assert position["status"] == "open_hedged"
    assert position["strategy_id"] == "tv_basis_btc_spot_perp_v1"
    assert position["symbol"] == "BTCUSDT"
    assert position["spot_leg"]["status"] == "filled"
    assert position["perp_leg"]["status"] == "filled"

    idempotency = _read_json(ledger.idempotency_path)
    positions = _read_json(ledger.positions_path)
    recovery = _read_json(ledger.recovery_path)

    assert ledger.idempotency_path.name == "tv_basis_arb_idempotency.json"
    assert ledger.positions_path.name == "tv_basis_arb_positions.json"
    assert ledger.recovery_path.name == "tv_basis_arb_recovery.json"
    assert idempotency["attempts"]["entry-1"]["status"] == "open_hedged"
    assert positions["positions"][position["position_key"]]["status"] == "open_hedged"
    assert recovery["recoveries"] == {}


def test_begin_entry_persists_target_base_qty_and_budget(tmp_path: Path) -> None:
    mod = _load_state_module()
    ledger = mod.TvBasisArbStateLedger(output_root=tmp_path)

    attempt = ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-baseqty-1",
        requested_notional_usdt=160.0,
        target_base_qty=0.002,
        max_quote_budget_usdt=160.0,
        tv_timestamp="2026-03-26T12:30:00Z",
    )
    positions = _read_json(ledger.positions_path)
    position = next(iter(positions["positions"].values()))

    assert attempt["target_base_qty"] == pytest.approx(0.002)
    assert attempt["max_quote_budget_usdt"] == 160.0
    assert position["target_base_qty"] == pytest.approx(0.002)
    assert position["max_quote_budget_usdt"] == 160.0


def test_partial_fill_then_leg_failure_marks_position_needs_recovery(tmp_path: Path) -> None:
    mod = _load_state_module()
    ledger = mod.TvBasisArbStateLedger(output_root=tmp_path)

    ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-2",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:31:00Z",
    )
    ledger.record_spot_buy_submitting(idempotency_key="entry-2", spot_order_id="spot-2")
    ledger.record_spot_buy_fill(
        idempotency_key="entry-2",
        spot_order_id="spot-2",
        filled_base_qty=0.00008,
        filled_quote_usdt=8.0,
        partial_fill=True,
    )
    recovery = ledger.record_needs_recovery(
        idempotency_key="entry-2",
        reason="perp_short_rejected",
        failure_phase="perp_short_submitting",
        recovery_action="flatten_spot_or_complete_hedge",
    )

    assert recovery["status"] == "needs_recovery"
    assert recovery["reason"] == "perp_short_rejected"
    assert recovery["spot_leg"]["filled_quote_usdt"] == 8.0
    assert recovery["perp_leg"]["status"] == "missing"

    idempotency = _read_json(ledger.idempotency_path)
    positions = _read_json(ledger.positions_path)
    recoveries = _read_json(ledger.recovery_path)
    position_key = recovery["position_key"]

    assert idempotency["attempts"]["entry-2"]["status"] == "needs_recovery"
    assert positions["positions"][position_key]["status"] == "needs_recovery"
    assert recoveries["recoveries"][position_key]["failure_phase"] == "perp_short_submitting"


def test_recovery_state_keeps_target_base_qty_and_budget(tmp_path: Path) -> None:
    mod = _load_state_module()
    ledger = mod.TvBasisArbStateLedger(output_root=tmp_path)

    ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-baseqty-2",
        requested_notional_usdt=160.0,
        target_base_qty=0.002,
        max_quote_budget_usdt=160.0,
        tv_timestamp="2026-03-26T12:31:00Z",
    )
    ledger.record_spot_buy_submitting(idempotency_key="entry-baseqty-2", spot_order_id="spot-baseqty-2")
    ledger.record_spot_buy_fill(
        idempotency_key="entry-baseqty-2",
        spot_order_id="spot-baseqty-2",
        filled_base_qty=0.002,
        filled_quote_usdt=141.0,
        partial_fill=False,
    )
    recovery = ledger.record_needs_recovery(
        idempotency_key="entry-baseqty-2",
        reason="perp_short_rejected",
        failure_phase="perp_short_submitting",
        recovery_action="flatten_spot_or_complete_hedge",
    )

    assert recovery["target_base_qty"] == pytest.approx(0.002)
    assert recovery["max_quote_budget_usdt"] == 160.0


def test_new_entry_is_blocked_while_active_recovery_exists(tmp_path: Path) -> None:
    mod = _load_state_module()
    ledger = mod.TvBasisArbStateLedger(output_root=tmp_path)

    ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-2",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:31:00Z",
    )
    ledger.record_spot_buy_submitting(idempotency_key="entry-2", spot_order_id="spot-2")
    ledger.record_spot_buy_fill(
        idempotency_key="entry-2",
        spot_order_id="spot-2",
        filled_base_qty=0.00008,
        filled_quote_usdt=8.0,
        partial_fill=True,
    )
    ledger.record_needs_recovery(
        idempotency_key="entry-2",
        reason="perp_short_rejected",
        failure_phase="perp_short_submitting",
        recovery_action="flatten_spot_or_complete_hedge",
    )

    allowed, reason = ledger.can_start_entry(strategy_id="tv_basis_btc_spot_perp_v1", symbol="BTCUSDT")
    assert allowed is False
    assert reason == "recovery_required"

    with pytest.raises(mod.StateConflictError, match="recovery"):
        ledger.begin_entry(
            strategy_id="tv_basis_btc_spot_perp_v1",
            symbol="BTCUSDT",
            idempotency_key="entry-3",
            requested_notional_usdt=10.0,
            tv_timestamp="2026-03-26T12:32:00Z",
        )


def test_recovery_is_cleared_when_recovery_position_is_closed(tmp_path: Path) -> None:
    mod = _load_state_module()
    ledger = mod.TvBasisArbStateLedger(output_root=tmp_path)

    ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-4",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:33:00Z",
    )
    ledger.record_spot_buy_submitting(idempotency_key="entry-4", spot_order_id="spot-4")
    ledger.record_spot_buy_fill(
        idempotency_key="entry-4",
        spot_order_id="spot-4",
        filled_base_qty=0.00008,
        filled_quote_usdt=8.0,
        partial_fill=True,
    )
    recovery = ledger.record_needs_recovery(
        idempotency_key="entry-4",
        reason="perp_short_rejected",
        failure_phase="perp_short_submitting",
        recovery_action="flatten_spot_or_complete_hedge",
    )

    closed = ledger.record_closed(idempotency_key="entry-4", close_reason="manual_recovery_flattened")
    allowed, reason = ledger.can_start_entry(strategy_id="tv_basis_btc_spot_perp_v1", symbol="BTCUSDT")
    recovery_state = _read_json(ledger.recovery_path)

    assert closed["status"] == "closed"
    assert allowed is True
    assert reason == ""
    assert recovery_state["recoveries"][recovery["position_key"]]["status"] == "closed"
    assert recovery_state["recoveries"][recovery["position_key"]]["resolved_by"] == "record_closed"


def test_begin_entry_replays_existing_attempt_for_same_idempotency_key(tmp_path: Path) -> None:
    mod = _load_state_module()
    ledger = mod.TvBasisArbStateLedger(output_root=tmp_path)

    first = ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-5",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:34:00Z",
    )
    replayed = ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-5",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:34:00Z",
    )

    assert replayed == first
    assert len(_read_json(ledger.idempotency_path)["attempts"]) == 1


def test_illegal_transition_is_rejected(tmp_path: Path) -> None:
    mod = _load_state_module()
    ledger = mod.TvBasisArbStateLedger(output_root=tmp_path)

    ledger.begin_entry(
        strategy_id="tv_basis_btc_spot_perp_v1",
        symbol="BTCUSDT",
        idempotency_key="entry-6",
        requested_notional_usdt=10.0,
        tv_timestamp="2026-03-26T12:35:00Z",
    )

    with pytest.raises(mod.IllegalTransitionError, match="entry_pending"):
        ledger.record_open_hedged(
            idempotency_key="entry-6",
            perp_order_id="perp-6",
            filled_base_qty=0.0001,
            avg_entry_price=100_100.0,
            basis_bps=12.0,
        )
