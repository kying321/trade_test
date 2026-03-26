from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_webhook_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "tv_basis_arb_webhook.py"
    spec = importlib.util.spec_from_file_location("tv_basis_arb_webhook", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_payload(strategy_id: str = "tv_basis_btc_spot_perp_v1") -> dict[str, str]:
    return {
        "strategy_id": strategy_id,
        "symbol": "BTCUSDT",
        "event_type": "entry_check",
        "tv_timestamp": "2026-03-26T12:30:00Z",
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _state_artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "execution_artifact_path": root / "state" / "tv_basis_arb_idempotency.json",
        "position_artifact_path": root / "state" / "tv_basis_arb_positions.json",
        "closeout_artifact_path": root / "state" / "tv_basis_arb_recovery.json",
    }


def _entry_snapshot() -> dict[str, float | str]:
    return {
        "symbol": "BTCUSDT",
        "spot_price": 70_500.0,
        "perp_mark_price": 70_600.0,
        "perp_index_price": 70_590.0,
        "open_interest_contracts": 1_200.0,
        "open_interest_usdt": 84_720_000.0,
        "funding_rate_8h": 0.0001,
        "snapshot_ts_utc": "2026-03-26T12:30:00Z",
        "snapshot_time_ms": 1_774_534_200_000,
    }


def _blocked_entry_snapshot() -> dict[str, float | str]:
    return {
        **_entry_snapshot(),
        "perp_mark_price": 70_540.0,
        "perp_index_price": 70_530.0,
    }


def _exchange_blocked_entry_snapshot() -> dict[str, float | str | dict[str, dict[str, float]]]:
    return {
        **_entry_snapshot(),
        "exchange_constraints": {
            "spot": {
                "min_qty": 0.00001,
                "step_size": 0.00001,
                "min_notional": 5.0,
            },
            "perp": {
                "min_qty": 0.003,
                "step_size": 0.001,
                "min_notional": 200.0,
            },
        },
    }


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


def test_missing_strategy_id_rejected(tmp_path: Path) -> None:
    webhook = _load_webhook_module()
    payload = _minimal_payload()
    payload.pop("strategy_id")

    with pytest.raises(ValueError, match="strategy_id"):
        webhook.handle_webhook(payload, output_root=tmp_path)


def test_unknown_strategy_id_rejected(tmp_path: Path) -> None:
    webhook = _load_webhook_module()
    payload = _minimal_payload(strategy_id="unknown_strategy")

    with pytest.raises(ValueError, match="strategy_id"):
        webhook.handle_webhook(payload, output_root=tmp_path)


def test_minimal_payload_writes_signal_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    webhook = _load_webhook_module()
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: _blocked_entry_snapshot())
    payload = _minimal_payload()
    result = webhook.handle_webhook(payload, output_root=tmp_path)
    artifact_path = Path(result["signal_artifact_path"])

    assert artifact_path.exists()
    written = _read_json(artifact_path)
    assert written["strategy_id"] == payload["strategy_id"]
    assert written["event_type"] == payload["event_type"]
    assert written["symbol"] == payload["symbol"]
    assert written["tv_timestamp"] == payload["tv_timestamp"]


def test_gate_blocked_artifact_keeps_notional_cap_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webhook = _load_webhook_module()
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: _blocked_entry_snapshot())

    result = webhook.handle_webhook(payload=_minimal_payload(), output_root=tmp_path)
    gate = _read_json(Path(result["gate_artifact_path"]))

    assert result["status"] == "gate_blocked"
    assert gate["action"] == "gate_blocked"
    assert gate["requested_notional_usdt"] == 160.0
    assert gate["max_notional_usdt"] == 160.0
    assert gate["max_quote_budget_usdt"] == 160.0
    assert gate["target_base_qty"] == pytest.approx(0.002)
    assert gate["thresholds"]["max_notional_usdt"] == 160.0


def test_gate_blocked_artifact_keeps_exchange_constraint_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webhook = _load_webhook_module()
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: _exchange_blocked_entry_snapshot())

    result = webhook.handle_webhook(payload=_minimal_payload(), output_root=tmp_path)
    gate = _read_json(Path(result["gate_artifact_path"]))

    assert result["status"] == "gate_blocked"
    assert "perp_min_qty_unmet" in gate["reasons"]
    assert "perp_min_notional_unmet" in gate["reasons"]
    assert gate["target_base_qty"] == pytest.approx(0.002)
    assert gate["estimated_base_qty"] == pytest.approx(0.002)
    assert gate["estimated_perp_notional_usdt"] == pytest.approx(141.2)
    assert gate["exchange_constraints"]["perp"]["min_qty"] == 0.003
    assert gate["exchange_constraints"]["perp"]["min_notional"] == 200.0


def test_symbol_override_mismatch_rejected(tmp_path: Path) -> None:
    webhook = _load_webhook_module()
    payload = _minimal_payload()
    payload["symbol"] = "ETHUSDT"

    with pytest.raises(ValueError, match="symbol mismatch"):
        webhook.handle_webhook(payload, output_root=tmp_path)


def test_signal_written_under_review_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    webhook = _load_webhook_module()
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: _blocked_entry_snapshot())
    payload = _minimal_payload()
    result = webhook.handle_webhook(payload, output_root=tmp_path)
    artifact_path = Path(result["signal_artifact_path"])

    assert (tmp_path / "review") in artifact_path.parents
    assert artifact_path.name.endswith(".json")
    assert "tv_basis_btc_spot_perp_v1" in artifact_path.name


def test_duplicate_alerts_emit_distinct_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    webhook = _load_webhook_module()
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: _blocked_entry_snapshot())
    payload = _minimal_payload()

    first = webhook.handle_webhook(payload, output_root=tmp_path)
    second = webhook.handle_webhook(payload, output_root=tmp_path)

    assert first["signal_artifact_path"] != second["signal_artifact_path"]
    assert Path(first["signal_artifact_path"]).exists()
    assert Path(second["signal_artifact_path"]).exists()


def test_duplicate_alert_id_still_creates_distinct_file_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    webhook = _load_webhook_module()
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: _blocked_entry_snapshot())
    payload = _minimal_payload()
    payload["alert_id"] = "unique-alert"

    first = webhook.handle_webhook(payload, output_root=tmp_path)
    second = webhook.handle_webhook(payload, output_root=tmp_path)

    assert first["signal_artifact_path"] != second["signal_artifact_path"]
    assert Path(first["signal_artifact_path"]).exists()
    assert Path(second["signal_artifact_path"]).exists()


def test_sanitized_alert_id_collisions_remain_distinct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    webhook = _load_webhook_module()
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: _blocked_entry_snapshot())
    payload_a = _minimal_payload()
    payload_b = _minimal_payload()
    payload_a["alert_id"] = "alert/123"
    payload_b["alert_id"] = "alert?123"

    sanitized_a = webhook._sanitize_filename(payload_a["alert_id"])
    sanitized_b = webhook._sanitize_filename(payload_b["alert_id"])
    assert sanitized_a == sanitized_b

    first = webhook.handle_webhook(payload_a, output_root=tmp_path)
    second = webhook.handle_webhook(payload_b, output_root=tmp_path)

    assert first["signal_artifact_path"] != second["signal_artifact_path"]
    assert Path(first["signal_artifact_path"]).exists()
    assert Path(second["signal_artifact_path"]).exists()


def test_entry_check_writes_signal_gate_and_opens_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webhook = _load_webhook_module()
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: _entry_snapshot())
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry",
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
                "orderId": "perp-entry",
                "executedQty": "0.00200",
                "avgPrice": "70600.0",
                "status": "FILLED",
            }
        ]
    )

    result = webhook.handle_webhook(payload=_minimal_payload(), output_root=tmp_path, spot_client=spot, perp_client=perp)

    assert result["status"] == "open_hedged"
    signal_artifact = Path(result["signal_artifact_path"])
    gate_artifact = Path(result["gate_artifact_path"])
    assert signal_artifact.exists()
    assert gate_artifact.exists()

    gate = _read_json(gate_artifact)
    assert gate["event_type"] == "entry_check"
    assert gate["action"] == "execute_entry"
    assert gate["requested_notional_usdt"] == 160.0
    assert gate["target_base_qty"] == pytest.approx(0.002)
    assert gate["max_quote_budget_usdt"] == 160.0
    assert gate["effective_quote_budget_usdt"] == 160.0
    assert gate["holding_time_seconds"] == 0.0
    assert gate["max_holding_seconds"] == 3600.0
    assert gate["runtime_policy"] == {
        "requested_notional_usdt": 160.0,
        "target_base_qty": 0.002,
        "max_quote_budget_usdt": 160.0,
        "exit_basis_bps": 4.0,
        "max_holding_seconds": 3600.0,
    }

    artifact_paths = _state_artifact_paths(tmp_path)
    assert result["execution_artifact_path"] == str(artifact_paths["execution_artifact_path"])
    assert result["position_artifact_path"] == str(artifact_paths["position_artifact_path"])
    assert artifact_paths["execution_artifact_path"].exists()
    assert artifact_paths["position_artifact_path"].exists()
    assert gate["execution_artifact_path"] == str(artifact_paths["execution_artifact_path"])
    assert gate["position_artifact_path"] == str(artifact_paths["position_artifact_path"])

    positions = _read_json(tmp_path / "state" / "tv_basis_arb_positions.json")
    position = next(iter(positions["positions"].values()))
    assert position["status"] == "open_hedged"
    assert position["requested_notional_usdt"] == 160.0
    assert position["target_base_qty"] == pytest.approx(0.002)
    assert position["max_quote_budget_usdt"] == 160.0


def test_exit_check_closes_position_when_basis_reverts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webhook = _load_webhook_module()
    snapshots = iter(
        [
            _entry_snapshot(),
            {
                **_entry_snapshot(),
                "perp_mark_price": 70_520.0,
                "perp_index_price": 70_515.0,
                "snapshot_ts_utc": "2026-03-26T12:45:00Z",
                "snapshot_time_ms": 1_774_535_100_000,
            },
        ]
    )
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: next(snapshots))
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry",
                "executedQty": "0.00200",
                "cummulativeQuoteQty": "141.0",
                "status": "FILLED",
            },
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-exit",
                "executedQty": "0.00200",
                "cummulativeQuoteQty": "141.05",
                "status": "FILLED",
            },
        ]
    )
    perp = _FakePerpClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-entry",
                "executedQty": "0.00200",
                "avgPrice": "70600.0",
                "status": "FILLED",
            },
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-exit",
                "executedQty": "0.00200",
                "avgPrice": "70520.0",
                "status": "FILLED",
            },
        ]
    )

    webhook.handle_webhook(payload=_minimal_payload(), output_root=tmp_path, spot_client=spot, perp_client=perp)
    exit_payload = _minimal_payload()
    exit_payload["event_type"] = "exit_check"
    exit_payload["tv_timestamp"] = "2026-03-26T12:45:00Z"
    result = webhook.handle_webhook(payload=exit_payload, output_root=tmp_path, spot_client=spot, perp_client=perp)

    assert result["status"] == "closed"
    gate = _read_json(Path(result["gate_artifact_path"]))
    assert gate["event_type"] == "exit_check"
    assert gate["action"] == "execute_exit"
    assert gate["close_reason"] == "basis_reverted"
    assert gate["holding_time_seconds"] == 900.0
    assert gate["requested_notional_usdt"] == 160.0
    assert gate["target_base_qty"] == pytest.approx(0.002)

    positions = _read_json(tmp_path / "state" / "tv_basis_arb_positions.json")
    position = next(iter(positions["positions"].values()))
    assert position["status"] == "closed"
    assert position["close_reason"] == "basis_reverted"


def test_exit_check_closes_position_when_max_holding_time_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webhook = _load_webhook_module()
    snapshots = iter(
        [
            _entry_snapshot(),
            {
                **_entry_snapshot(),
                "perp_mark_price": 70_580.0,
                "perp_index_price": 70_575.0,
                "snapshot_ts_utc": "2026-03-26T13:31:00Z",
                "snapshot_time_ms": 1_774_537_860_000,
            },
        ]
    )
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: next(snapshots))
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry",
                "executedQty": "0.00200",
                "cummulativeQuoteQty": "141.0",
                "status": "FILLED",
            },
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-exit",
                "executedQty": "0.00200",
                "cummulativeQuoteQty": "141.08",
                "status": "FILLED",
            },
        ]
    )
    perp = _FakePerpClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-entry",
                "executedQty": "0.00200",
                "avgPrice": "70600.0",
                "status": "FILLED",
            },
            {
                "symbol": "BTCUSDT",
                "orderId": "perp-exit",
                "executedQty": "0.00200",
                "avgPrice": "70580.0",
                "status": "FILLED",
            },
        ]
    )

    webhook.handle_webhook(payload=_minimal_payload(), output_root=tmp_path, spot_client=spot, perp_client=perp)
    exit_payload = _minimal_payload()
    exit_payload["event_type"] = "exit_check"
    exit_payload["tv_timestamp"] = "2026-03-26T13:31:00Z"
    result = webhook.handle_webhook(payload=exit_payload, output_root=tmp_path, spot_client=spot, perp_client=perp)

    assert result["status"] == "closed"
    gate = _read_json(Path(result["gate_artifact_path"]))
    assert gate["close_reason"] == "max_holding_time_exceeded"
    assert gate["action"] == "execute_exit"
    assert gate["holding_time_seconds"] == 3660.0
    assert gate["max_holding_seconds"] == 3600.0

    positions = _read_json(tmp_path / "state" / "tv_basis_arb_positions.json")
    position = next(iter(positions["positions"].values()))
    assert position["status"] == "closed"
    assert position["close_reason"] == "max_holding_time_exceeded"


def test_exit_check_surfaces_pending_recovery_instead_of_no_open_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webhook = _load_webhook_module()
    snapshots = iter(
        [
            _entry_snapshot(),
            {
                **_entry_snapshot(),
                "perp_mark_price": 70_520.0,
                "perp_index_price": 70_515.0,
                "snapshot_ts_utc": "2026-03-26T12:45:00Z",
                "snapshot_time_ms": 1_774_535_100_000,
            },
        ]
    )
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: next(snapshots))
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry",
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
                "orderId": "perp-entry",
                "executedQty": "0.00200",
                "avgPrice": "70600.0",
                "status": "FILLED",
            },
            RuntimeError("perp_close_rejected"),
        ]
    )

    webhook.handle_webhook(payload=_minimal_payload(), output_root=tmp_path, spot_client=spot, perp_client=perp)
    exit_payload = _minimal_payload()
    exit_payload["event_type"] = "exit_check"
    exit_payload["tv_timestamp"] = "2026-03-26T12:45:00Z"
    first = webhook.handle_webhook(payload=exit_payload, output_root=tmp_path, spot_client=spot, perp_client=perp)

    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: pytest.fail("recovery_required should not fetch snapshot"))
    second = webhook.handle_webhook(payload=exit_payload, output_root=tmp_path, spot_client=spot, perp_client=perp)

    assert first["status"] == "needs_recovery"
    assert second["status"] == "needs_recovery"
    assert second["gate"]["action"] == "recovery_required"
    assert second["gate"]["recovery_reason"] == "perp_close_rejected"
    assert second["gate"]["position_key"] == first["execution"]["position"]["position_key"]
    assert second["gate"]["runtime_policy"]["exit_basis_bps"] == 4.0
    gate = _read_json(Path(second["gate_artifact_path"]))
    assert gate["action"] == "recovery_required"
    assert gate["recovery_reason"] == "perp_close_rejected"
    assert gate["closeout_artifact_path"] == str(_state_artifact_paths(tmp_path)["closeout_artifact_path"])


def test_exit_check_recovery_response_emits_review_artifact_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    webhook = _load_webhook_module()
    snapshots = iter(
        [
            _entry_snapshot(),
            {
                **_entry_snapshot(),
                "perp_mark_price": 70_520.0,
                "perp_index_price": 70_515.0,
                "snapshot_ts_utc": "2026-03-26T12:45:00Z",
                "snapshot_time_ms": 1_774_535_100_000,
            },
        ]
    )
    monkeypatch.setattr(webhook, "build_market_snapshot", lambda **_: next(snapshots))
    spot = _FakeSpotClient(
        [
            {
                "symbol": "BTCUSDT",
                "orderId": "spot-entry",
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
                "orderId": "perp-entry",
                "executedQty": "0.00200",
                "avgPrice": "70600.0",
                "status": "FILLED",
            },
            RuntimeError("perp_close_rejected"),
        ]
    )

    webhook.handle_webhook(payload=_minimal_payload(), output_root=tmp_path, spot_client=spot, perp_client=perp)
    exit_payload = _minimal_payload()
    exit_payload["event_type"] = "exit_check"
    exit_payload["tv_timestamp"] = "2026-03-26T12:45:00Z"
    result = webhook.handle_webhook(payload=exit_payload, output_root=tmp_path, spot_client=spot, perp_client=perp)

    artifact_paths = _state_artifact_paths(tmp_path)
    assert result["status"] == "needs_recovery"
    assert result["execution_artifact_path"] == str(artifact_paths["execution_artifact_path"])
    assert result["position_artifact_path"] == str(artifact_paths["position_artifact_path"])
    assert result["closeout_artifact_path"] == str(artifact_paths["closeout_artifact_path"])
    assert artifact_paths["execution_artifact_path"].exists()
    assert artifact_paths["position_artifact_path"].exists()
    assert artifact_paths["closeout_artifact_path"].exists()

    gate = _read_json(Path(result["gate_artifact_path"]))
    assert gate["execution_artifact_path"] == str(artifact_paths["execution_artifact_path"])
    assert gate["position_artifact_path"] == str(artifact_paths["position_artifact_path"])
    assert gate["closeout_artifact_path"] == str(artifact_paths["closeout_artifact_path"])
