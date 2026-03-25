from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "binance_infra_canary.py"
    spec = importlib.util.spec_from_file_location("binance_infra_canary", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_config(path: Path) -> None:
    path.write_text(yaml.safe_dump({"validation": {}}, sort_keys=False), encoding="utf-8")


def _read_summary(output_root: Path) -> dict[str, object]:
    return json.loads((output_root / "review" / "latest_binance_infra_canary.json").read_text(encoding="utf-8"))


class _ReadySpotClient:
    def __init__(self, **kwargs) -> None:
        _ = kwargs
        self.usdt_free = 100.0
        self.btc_free = 0.0
        self.orders: list[dict[str, object]] = []
        self.price = 100_000.0
        self.step_size = 0.000003

    def ping(self) -> dict[str, object]:
        return {}

    def ticker_price(self, symbol: str) -> float:
        _ = symbol
        return self.price

    def exchange_info(self, symbol: str) -> dict[str, object]:
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": str(self.step_size), "minQty": "0.000003"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ],
                }
            ]
        }

    def account(self) -> dict[str, object]:
        return {
            "canTrade": True,
            "balances": [
                {"asset": "USDT", "free": f"{self.usdt_free:.8f}", "locked": "0"},
                {"asset": "BTC", "free": f"{self.btc_free:.8f}", "locked": "0"},
            ],
        }

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str,
        quote_order_qty: float | None = None,
    ) -> dict[str, object]:
        _ = symbol
        self.orders.append(
            {
                "side": side,
                "quantity": quantity,
                "quote_order_qty": quote_order_qty,
                "client_order_id": client_order_id,
            }
        )
        if side == "BUY":
            spent = float(quote_order_qty or 0.0)
            bought_qty = 0.00005
            self.usdt_free -= spent
            self.btc_free += bought_qty
            return {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "status": "FILLED",
                "executedQty": f"{bought_qty:.8f}",
                "cummulativeQuoteQty": f"{spent:.8f}",
                "clientOrderId": client_order_id,
            }
        sell_qty = float(quantity)
        recv = sell_qty * self.price
        self.btc_free -= sell_qty
        self.usdt_free += recv
        return {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "status": "FILLED",
            "executedQty": f"{sell_qty:.8f}",
            "cummulativeQuoteQty": f"{recv:.8f}",
            "clientOrderId": client_order_id,
        }


class _DynamicQuoteClient(_ReadySpotClient):
    def __init__(
        self,
        *,
        price: float = 100_000.0,
        step_size: float = 0.000003,
        min_notional: float = 5.0,
        sell_transport_error: bool = False,
    ) -> None:
        super().__init__()
        self.price = float(price)
        self.step_size = float(step_size)
        self.min_notional = float(min_notional)
        self.sell_transport_error = bool(sell_transport_error)

    def exchange_info(self, symbol: str) -> dict[str, object]:
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": str(self.step_size), "minQty": str(self.step_size)},
                        {"filterType": "MIN_NOTIONAL", "minNotional": str(self.min_notional)},
                    ],
                }
            ]
        }

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str,
        quote_order_qty: float | None = None,
    ) -> dict[str, object]:
        _ = symbol
        self.orders.append(
            {
                "side": side,
                "quantity": quantity,
                "quote_order_qty": quote_order_qty,
                "client_order_id": client_order_id,
            }
        )
        if side == "BUY":
            spent = float(quote_order_qty or 0.0)
            bought_qty = spent / self.price if self.price > 0.0 else 0.0
            self.usdt_free -= spent
            self.btc_free += bought_qty
            return {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "status": "FILLED",
                "executedQty": f"{bought_qty:.8f}",
                "cummulativeQuoteQty": f"{spent:.8f}",
                "clientOrderId": client_order_id,
            }
        if self.sell_transport_error:
            raise ConnectionError("sell socket hang up")
        sell_qty = float(quantity)
        notional = sell_qty * self.price
        if notional + 1e-12 < self.min_notional:
            raise RuntimeError("Filter failure: NOTIONAL")
        self.btc_free -= sell_qty
        self.usdt_free += notional
        return {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "status": "FILLED",
            "executedQty": f"{sell_qty:.8f}",
            "cummulativeQuoteQty": f"{notional:.8f}",
            "clientOrderId": client_order_id,
        }


def test_probe_ignores_strategy_ticket_authority(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    client = _ReadySpotClient()
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"
    ticket_path = out_root / "artifacts" / "signal_to_order_tickets" / "latest.json"
    ticket_path.parent.mkdir(parents=True, exist_ok=True)
    ticket_path.write_text("{not-json", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "probe",
        ],
    )
    rc = mod.main()
    assert rc == 0

    payload = _read_summary(out_root)
    assert bool(payload.get("ok")) is True
    account_ready = payload.get("steps", {}).get("account_ready", {})
    assert isinstance(account_ready, dict)
    assert bool(account_ready.get("ready")) is True


def test_autopilot_check_blocks_when_daily_budget_exceeded(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    client = _ReadySpotClient()
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"
    budget_path = out_root / "state" / "infra_canary_budget.json"
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    today = mod.current_utc_date().isoformat()
    budget_path.write_text(
        json.dumps(
            {
                "days": {
                    today: {
                        "spent_quote_usdt": 20.0,
                        "events": [{"quote_usdt": 20.0, "status": "filled"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "autopilot-check",
            "--daily-budget-cap-usdt",
            "20",
        ],
    )
    rc = mod.main()
    assert rc == 0

    payload = _read_summary(out_root)
    assert bool(payload.get("autopilot_allowed")) is False
    budget = payload.get("steps", {}).get("budget", {})
    assert isinstance(budget, dict)
    assert bool(budget.get("within_cap")) is False


def test_run_round_trip_succeeds_with_dust_allowed(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    client = _ReadySpotClient()
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "run",
            "--allow-dust",
        ],
    )
    rc = mod.main()
    assert rc == 0

    payload = _read_summary(out_root)
    assert bool(payload.get("ok")) is True
    round_trip = payload.get("steps", {}).get("round_trip", {})
    assert isinstance(round_trip, dict)
    assert bool(round_trip.get("executed")) is True
    assert int(round_trip.get("orders_submitted", 0)) == 2
    dust = round_trip.get("dust", {})
    assert isinstance(dust, dict)
    assert float(dust.get("base_asset_qty", 0.0)) > 0.0
    budget_state = json.loads((out_root / "state" / "infra_canary_budget.json").read_text(encoding="utf-8"))
    assert budget_state["days"]


def test_run_raises_effective_quote_to_required_floor_when_requested_quote_is_too_low(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    client = _DynamicQuoteClient(price=100_000.0, step_size=0.000003, min_notional=5.0)
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "run",
            "--quote-usdt",
            "5",
            "--allow-dust",
        ],
    )
    rc = mod.main()
    assert rc == 0

    payload = _read_summary(out_root)
    plan = payload.get("steps", {}).get("plan", {})
    assert isinstance(plan, dict)
    assert float(plan.get("requested_quote_usdt", 0.0)) == 5.0
    assert float(plan.get("required_round_trip_quote_usdt", 0.0)) == 5.1
    assert float(plan.get("effective_quote_usdt", 0.0)) == 5.1
    assert float(plan.get("single_run_cap_usdt", 0.0)) == 12.0
    assert [row["quote_order_qty"] for row in client.orders if str(row["side"]) == "BUY"] == [5.1]

    budget = payload.get("steps", {}).get("budget", {})
    assert isinstance(budget, dict)
    assert float(budget.get("pending_quote_usdt", 0.0)) == 5.1
    assert float(budget.get("spent_quote_usdt", 0.0)) == 5.1


def test_run_skips_gracefully_when_required_quote_exceeds_single_run_cap(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    client = _DynamicQuoteClient(price=100_000.0, step_size=0.000003, min_notional=12.1)
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "run",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert client.orders == []

    payload = _read_summary(out_root)
    round_trip = payload.get("steps", {}).get("round_trip", {})
    assert isinstance(round_trip, dict)
    assert bool(round_trip.get("executed")) is False
    assert str(round_trip.get("reason", "")) == "single_run_cap_exceeded"

    plan = payload.get("steps", {}).get("plan", {})
    assert isinstance(plan, dict)
    assert float(plan.get("requested_quote_usdt", 0.0)) == 10.0
    assert float(plan.get("required_round_trip_quote_usdt", 0.0)) == 12.3
    assert float(plan.get("effective_quote_usdt", 0.0)) == 12.3
    assert str(plan.get("skip_reasons", [""])[0]) == "single_run_cap_exceeded"


def test_sell_ambiguity_recovery_uses_effective_quote_for_budget_and_idempotency(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    client = _DynamicQuoteClient(price=100_000.0, step_size=0.000003, min_notional=11.0, sell_transport_error=True)
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "run",
            "--allow-dust",
        ],
    )
    rc1 = mod.main()
    assert rc1 == 3

    idem_key = mod.build_idempotency_key(
        day_key=mod.current_utc_date().isoformat(),
        market="spot",
        symbol="BTCUSDT",
        quote_usdt=11.1,
    )
    idem_state = json.loads((out_root / "state" / "infra_canary_idempotency.json").read_text(encoding="utf-8"))
    attempt = idem_state["attempts"][idem_key]
    assert str(attempt["status"]) == "needs_recovery"
    assert str(attempt["phase"]) == "sell_transport_ambiguous"
    assert float(attempt["effective_quote_usdt"]) == 11.1
    assert float(attempt["requested_quote_usdt"]) == 10.0
    assert float(attempt["required_round_trip_quote_usdt"]) == 11.1

    budget_state = json.loads((out_root / "state" / "infra_canary_budget.json").read_text(encoding="utf-8"))
    day_state = budget_state["days"][mod.current_utc_date().isoformat()]
    assert float(day_state["spent_quote_usdt"]) == 11.1
    assert any(float(row.get("quote_usdt", 0.0)) == 11.1 for row in day_state["events"])

    first_buy_count = len([row for row in client.orders if str(row["side"]) == "BUY"])
    assert first_buy_count == 1

    rc2 = mod.main()
    assert rc2 == 2
    second_buy_count = len([row for row in client.orders if str(row["side"]) == "BUY"])
    assert second_buy_count == 1

    payload = _read_summary(out_root)
    idem = payload.get("steps", {}).get("idempotency", {})
    assert isinstance(idem, dict)
    assert bool(idem.get("skipped")) is True
    assert str(idem.get("reason", "")) == "recovery_required"


def test_run_skips_when_idempotency_key_already_recorded(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    client = _ReadySpotClient()
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"
    state_path = out_root / "state" / "infra_canary_idempotency.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    idem_key = mod.build_idempotency_key(
        day_key=mod.current_utc_date().isoformat(),
        market="spot",
        symbol="BTCUSDT",
        quote_usdt=10.0,
    )
    state_path.write_text(json.dumps({"keys": [idem_key]}, indent=2), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "run",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert client.orders == []

    payload = _read_summary(out_root)
    idem = payload.get("steps", {}).get("idempotency", {})
    assert isinstance(idem, dict)
    assert bool(idem.get("skipped")) is True
    assert str(idem.get("reason", "")) == "idempotent_skip"


def test_probe_marks_account_ready_false(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _NotReadyClient(_ReadySpotClient):
        def account(self) -> dict[str, object]:
            payload = super().account()
            payload["canTrade"] = False
            return payload

    client = _NotReadyClient()
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "probe",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = _read_summary(out_root)
    account_ready = payload.get("steps", {}).get("account_ready", {})
    assert isinstance(account_ready, dict)
    assert bool(account_ready.get("ready")) is False


def test_run_transport_ambiguity_triggers_panic_class_failure(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _TransportFailClient(_ReadySpotClient):
        def place_market_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: float,
            client_order_id: str,
            quote_order_qty: float | None = None,
        ) -> dict[str, object]:
            _ = (symbol, side, quantity, client_order_id, quote_order_qty)
            raise ConnectionError("socket hang up")

    client = _TransportFailClient()
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "run",
        ],
    )
    rc = mod.main()
    assert rc == 3
    assert (out_root / "state" / "panic_close_all.json").exists()

    payload = _read_summary(out_root)
    assert str(payload.get("failure_class", "")) == "panic"


def test_run_buy_fill_then_sell_ambiguity_records_recovery_state_and_blocks_rerun(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _SellAmbiguousClient(_ReadySpotClient):
        def place_market_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: float,
            client_order_id: str,
            quote_order_qty: float | None = None,
        ) -> dict[str, object]:
            if side == "BUY":
                return super().place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    client_order_id=client_order_id,
                    quote_order_qty=quote_order_qty,
                )
            self.orders.append(
                {
                    "side": side,
                    "quantity": quantity,
                    "quote_order_qty": quote_order_qty,
                    "client_order_id": client_order_id,
                }
            )
            raise ConnectionError("sell socket hang up")

    client = _SellAmbiguousClient()
    monkeypatch.setattr(mod, "BinanceSpotClient", lambda **kwargs: client)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path)
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_infra_canary.py",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--mode",
            "run",
        ],
    )
    rc1 = mod.main()
    assert rc1 == 3

    idem_key = mod.build_idempotency_key(
        day_key=mod.current_utc_date().isoformat(),
        market="spot",
        symbol="BTCUSDT",
        quote_usdt=10.0,
    )
    idem_state = json.loads((out_root / "state" / "infra_canary_idempotency.json").read_text(encoding="utf-8"))
    attempt = idem_state["attempts"][idem_key]
    assert str(attempt["status"]) == "needs_recovery"
    assert str(attempt["phase"]) == "sell_transport_ambiguous"
    assert str(attempt["buy"]["status"]) == "FILLED"

    budget_state = json.loads((out_root / "state" / "infra_canary_budget.json").read_text(encoding="utf-8"))
    day_state = budget_state["days"][mod.current_utc_date().isoformat()]
    assert float(day_state["spent_quote_usdt"]) >= 10.0
    assert any(str(row.get("status")) == "buy_filled_sell_pending" for row in day_state["events"])

    first_buy_count = len([row for row in client.orders if str(row["side"]) == "BUY"])
    assert first_buy_count == 1

    rc2 = mod.main()
    assert rc2 == 2
    second_buy_count = len([row for row in client.orders if str(row["side"]) == "BUY"])
    assert second_buy_count == 1

    payload = _read_summary(out_root)
    idem = payload.get("steps", {}).get("idempotency", {})
    assert isinstance(idem, dict)
    assert bool(idem.get("skipped")) is True
    assert str(idem.get("reason", "")) == "recovery_required"
