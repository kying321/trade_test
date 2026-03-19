from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path
import sys

import yaml


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "binance_live_takeover.py"
    spec = importlib.util.spec_from_file_location("binance_live_takeover", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calc_canary_quantity_respects_min_notional_and_step() -> None:
    mod = _load_module()
    qty = mod.calc_canary_quantity(
        quote_usdt=5.0,
        price=73_000.0,
        step_size=0.001,
        min_qty=0.001,
        min_notional=100.0,
    )
    assert qty >= 0.001
    assert abs((qty / 0.001) - round(qty / 0.001)) < 1e-9
    assert qty * 73_000.0 >= 100.0


def test_activate_config_for_live_switches_expected_fields(tmp_path: Path) -> None:
    mod = _load_module()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {"provider_profile": "opensource_dual"},
                "validation": {
                    "broker_snapshot_source_mode": "paper_engine",
                    "broker_snapshot_live_mapping_profile": "generic",
                    "broker_snapshot_live_fallback_to_paper": True,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    changed = mod.activate_config_for_live(cfg_path)
    assert bool(changed)

    out = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert str(out["data"]["provider_profile"]) == "dual_binance_bybit_public"
    assert str(out["validation"]["broker_snapshot_source_mode"]) == "hybrid_prefer_live"
    assert str(out["validation"]["broker_snapshot_live_mapping_profile"]) == "binance"
    assert bool(out["validation"]["broker_snapshot_live_fallback_to_paper"]) is True


def test_binance_spot_client_buy_order_uses_explicit_quantity_by_default() -> None:
    mod = _load_module()
    client = mod.BinanceSpotClient(api_key="k", api_secret="s")
    captured: dict[str, object] = {}

    def _fake_request(*, method: str, path: str, params: dict[str, object] | None = None, signed: bool = False):
        captured["method"] = method
        captured["path"] = path
        captured["params"] = dict(params or {})
        captured["signed"] = signed
        return {"status": "ok"}

    client._request = _fake_request  # type: ignore[method-assign]
    out = client.place_market_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.001,
        client_order_id="cid-1",
        quote_order_qty=5.0,
    )

    assert out == {"status": "ok"}
    assert str(captured.get("method")) == "POST"
    assert str(captured.get("path")) == "/api/v3/order"
    assert bool(captured.get("signed")) is True
    params = captured.get("params")
    assert isinstance(params, dict)
    assert "quantity" in params
    assert str(params.get("quantity", "")) == "0.00100000"


def test_choose_canary_signal_fails_closed_without_signal_artifact(tmp_path: Path) -> None:
    mod = _load_module()
    selected = mod.choose_canary_signal(output_root=tmp_path, target_date=date(2026, 3, 5), whitelist=["BTCUSDT"])
    assert bool(selected.get("signal_missing", False)) is True
    assert str(selected.get("symbol", "")) == ""
    assert str(selected.get("reason", "")) == "no_latest_signal_artifact"


def test_choose_canary_ticket_prefers_actionable_spot_ticket(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306T000000Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "tickets": [
                    {
                        "symbol": "ETHUSDT",
                        "allowed": False,
                        "reasons": ["size_below_min_notional"],
                        "signal": {"side": "SHORT", "confidence": 42.0, "convexity_ratio": 2.4},
                        "execution": {"mode": "HEDGE_ONLY"},
                    },
                    {
                        "symbol": "BTCUSDT",
                        "date": "2026-03-05",
                        "allowed": False,
                        "reasons": ["size_below_min_notional"],
                        "signal": {"side": "LONG", "confidence": 21.0, "convexity_ratio": 1.9},
                        "execution": {"mode": "SPOT_LONG_OR_SELL"},
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    selected = mod.choose_canary_ticket(output_root=tmp_path, whitelist=["BTCUSDT", "ETHUSDT"])
    assert bool(selected.get("signal_missing", False)) is False
    assert str(selected.get("symbol", "")) == "BTCUSDT"
    assert str(selected.get("side", "")) == "BUY"
    assert str(selected.get("source", "")).endswith("_signal_to_order_tickets.json")


def test_choose_canary_ticket_surfaces_best_blocked_candidate_when_no_actionable_ticket(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306T000000Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "tickets": [
                    {
                        "symbol": "BTCUSDT",
                        "allowed": False,
                        "reasons": ["signal_not_found"],
                        "signal": {"side": "", "confidence": 0.0, "convexity_ratio": 0.0},
                        "execution": {"mode": "NO_SIGNAL"},
                    },
                    {
                        "symbol": "BNBUSDT",
                        "date": "2026-03-03",
                        "age_days": 3,
                        "allowed": False,
                        "reasons": ["confidence_below_threshold", "size_below_min_notional"],
                        "signal": {"side": "LONG", "confidence": 44.1, "convexity_ratio": 3.1},
                        "execution": {"mode": "SPOT_LONG_OR_SELL"},
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    selected = mod.choose_canary_ticket(output_root=tmp_path, whitelist=["BTCUSDT", "BNBUSDT"])
    assert bool(selected.get("signal_missing", False)) is True
    assert str(selected.get("reason", "")) == "no_actionable_ticket"
    blocked = selected.get("blocked_candidate", {})
    assert isinstance(blocked, dict)
    assert str(blocked.get("symbol", "")) == "BNBUSDT"
    assert list(blocked.get("ticket_reasons", [])) == ["confidence_below_threshold", "size_below_min_notional"]
    assert int(blocked.get("age_days", -1)) == 3


def test_choose_canary_ticket_blocks_spot_long_ticket_for_portfolio_margin_market(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306T000000Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "tickets": [
                    {
                        "symbol": "SOLUSDT",
                        "date": "2026-03-06",
                        "age_days": 0,
                        "allowed": True,
                        "reasons": [],
                        "signal": {"side": "LONG", "confidence": 72.0, "convexity_ratio": 3.6},
                        "execution": {"mode": "SPOT_LONG_OR_SELL"},
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    selected = mod.choose_canary_ticket(
        output_root=tmp_path,
        whitelist=["SOLUSDT"],
        market="portfolio_margin_um",
    )
    assert bool(selected.get("signal_missing", False)) is True
    assert str(selected.get("reason", "")) == "target_market_read_only"
    blocked = selected.get("blocked_candidate", {})
    assert isinstance(blocked, dict)
    assert str(blocked.get("symbol", "")) == "SOLUSDT"
    assert list(blocked.get("ticket_reasons", [])) == ["target_market_read_only"]
    assert str(blocked.get("target_market", "")) == "portfolio_margin_um"


def test_build_time_slices_uses_market_specific_windows() -> None:
    mod = _load_module()
    assert mod.trade_query_slice_hours_for_market("spot") == 24
    assert mod.trade_query_slice_hours_for_market("futures_usdm") == 168
    assert mod.trade_query_slice_hours_for_market("portfolio_margin_um") == 168
    slices = mod.build_time_slices(start_ms=0, end_ms=48 * 60 * 60 * 1000, slice_hours=24)
    assert slices == [
        (0, 24 * 60 * 60 * 1000),
        (24 * 60 * 60 * 1000, 48 * 60 * 60 * 1000),
    ]


def test_choose_canary_ticket_surfaces_market_scope_mismatch_reason(tmp_path: Path) -> None:
    mod = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260317T000000Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "tickets": [
                    {
                        "symbol": "SOLUSDT",
                        "date": "2026-03-16",
                        "age_days": 1,
                        "allowed": False,
                        "reasons": ["signal_market_scope_mismatch", "confidence_below_threshold"],
                        "signal": {"side": "LONG", "confidence": 71.5, "convexity_ratio": 3.4},
                        "execution": {"mode": "SPOT_LONG_OR_SELL"},
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    selected = mod.choose_canary_ticket(output_root=tmp_path, whitelist=["SOLUSDT"])
    assert bool(selected.get("signal_missing", False)) is True
    assert str(selected.get("reason", "")) == "signal_market_scope_mismatch"
    blocked = selected.get("blocked_candidate", {})
    assert isinstance(blocked, dict)
    assert str(blocked.get("symbol", "")) == "SOLUSDT"
    assert "signal_market_scope_mismatch" in list(blocked.get("ticket_reasons", []))


def test_fetch_trade_rows_windowed_dedupes_across_slices() -> None:
    mod = _load_module()
    calls: list[tuple[int, int]] = []

    class _Client:
        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000):
            _ = (symbol, limit)
            calls.append((start_ms, end_ms))
            return [
                {"symbol": "BTCUSDT", "id": 1, "time": start_ms + 1, "qty": "0.001"},
                {"symbol": "BTCUSDT", "id": 1, "time": start_ms + 1, "qty": "0.001"},
            ]

    rows, slices = mod.fetch_trade_rows_windowed(
        client=_Client(),
        symbol="BTCUSDT",
        start_ms=0,
        end_ms=48 * 60 * 60 * 1000,
        slice_hours=24,
    )
    assert len(slices) == 2
    assert len(calls) == 2
    assert len(rows) == 2


def test_main_spot_canary_precheck_uses_effective_quote_floor(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            # 5.05 > budget_quote(5.0) but < effective_quote(~5.11), should fail precheck.
            return {"balances": [{"asset": "USDT", "free": "5.05", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

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
            raise AssertionError("precheck should block order submission")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
            "--canary-quote-usdt",
            "5.0",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", True)) is False
    assert str(canary.get("reason", "")) == "precheck_insufficient_quote_balance"
    assert float(canary.get("required", 0.0)) > 5.05


def test_main_blocks_live_order_when_risk_fuse_is_fresh_and_blocked(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "20.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

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
            raise AssertionError("risk fuse should block order submission")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"
    (out_root / "state").mkdir(parents=True, exist_ok=True)
    (out_root / "state" / "live_risk_fuse.json").write_text(
        json.dumps(
            {
                "updated_at_utc": "2026-03-06T00:00:00Z",
                "allowed": False,
                "status": "blocked",
                "reasons": ["open_exposure_above_cap"],
                "artifact": str(out_root / "review" / "20260306T000000Z_live_risk_guard.json"),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
            "--canary-quote-usdt",
            "5.0",
            "--risk-fuse-max-age-seconds",
            "999999999",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", True)) is False
    assert str(canary.get("reason", "")) == "risk_guard_blocked"
    risk_guard = steps.get("risk_guard", {})
    assert isinstance(risk_guard, dict)
    assert bool(risk_guard.get("allowed", True)) is False


def test_main_records_exchange_reject_for_canary_order(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "10.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

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
            raise RuntimeError("http_400:{\"code\":-2010,\"msg\":\"insufficient balance\"}")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    assert bool(payload.get("ok")) is False
    assert str(payload.get("mode", "")) == "live_ready"
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    account_overview = steps.get("account_overview", {})
    assert isinstance(account_overview, dict)
    assert str(account_overview.get("market", "")) == "spot"
    assert float(account_overview.get("quote_available", 0.0)) >= 10.0
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", True)) is False
    assert str(canary.get("reason", "")) == "exchange_reject"


def test_main_409_exchange_conflict_triggers_panic_close_all(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "20.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

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
            raise RuntimeError("http_409:{\"code\":-2011,\"msg\":\"conflict\"}")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    assert "panic_close_all:canary_order_transport_conflict_error" in str(payload.get("panic", ""))
    panic_state = json.loads((out_root / "state" / "panic_close_all.json").read_text(encoding="utf-8"))
    assert str(panic_state.get("reason", "")) == "canary_order_transport_conflict_error"


def test_main_spot_position_price_transport_degrades_without_panic(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs
            self._ticker_calls = 0

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            self._ticker_calls += 1
            if self._ticker_calls >= 2:
                raise ConnectionError("socket closed by peer")
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "BTC", "free": "0.001", "locked": "0.0"}, {"asset": "USDT", "free": "20", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 0

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    assert str(payload.get("panic", "")) == ""
    account_overview = (payload.get("steps", {}) or {}).get("account_overview", {})
    assert isinstance(account_overview, dict)
    warnings = account_overview.get("warnings", [])
    assert isinstance(warnings, list)
    assert warnings
    assert str((warnings[0] or {}).get("reason", "")) == "binance_public_probe_transport_error"
    assert not (out_root / "state" / "panic_close_all.json").exists()


def test_main_public_probe_transport_blocks_without_panic(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            raise ConnectionError("socket closed during public probe")

        def exchange_info(self, symbol: str) -> dict[str, object]:
            _ = symbol
            raise AssertionError("exchange_info should not run after ticker probe failure")

        def account(self) -> dict[str, object]:
            raise AssertionError("account should not run after public probe failure")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    assert str(payload.get("mode", "")) == "live_ready_preflight_blocked"
    canary = (payload.get("steps", {}) or {}).get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", True)) is False
    assert str(canary.get("reason", "")) == "preflight_transport_blocked"
    assert str(canary.get("failed_step", "")) == "canary_plan"
    assert not (out_root / "state" / "panic_close_all.json").exists()


def test_main_account_runtime_error_blocks_without_panic(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            raise RuntimeError("http_451:{\"code\":0,\"msg\":\"restricted\"}")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    assert str(payload.get("mode", "")) == "live_ready_preflight_blocked"
    account_overview = (payload.get("steps", {}) or {}).get("account_overview", {})
    assert isinstance(account_overview, dict)
    assert str(account_overview.get("reason", "")) == "binance_account_runtime_error"
    canary = (payload.get("steps", {}) or {}).get("canary_order", {})
    assert isinstance(canary, dict)
    assert str(canary.get("reason", "")) == "preflight_runtime_blocked"
    assert str(canary.get("failed_step", "")) == "account_overview"
    assert not (out_root / "state" / "panic_close_all.json").exists()


def test_main_trade_and_income_transport_failures_degrade_without_halt(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "20.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            raise ConnectionError("socket closed during user trades")

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            raise TimeoutError("income query timed out")

        def place_market_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: float,
            client_order_id: str,
            quote_order_qty: float | None = None,
        ) -> dict[str, object]:
            _ = (client_order_id, quote_order_qty)
            return {"symbol": symbol, "side": side, "executedQty": f"{quantity:.8f}", "status": "FILLED"}

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 0

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    telemetry = (payload.get("steps", {}) or {}).get("trade_telemetry", {})
    assert isinstance(telemetry, dict)
    assert str(telemetry.get("status", "")) == "degraded"
    warnings = telemetry.get("warnings", [])
    assert isinstance(warnings, list)
    assert len(warnings) == 2
    assert {str((row or {}).get("reason", "")) for row in warnings} == {
        "binance_user_trades_transport_error",
        "binance_income_transport_error",
    }
    feedback = (payload.get("steps", {}) or {}).get("backtest_feedback", {})
    assert isinstance(feedback, dict)
    assert str(feedback.get("status", "")) == "degraded"
    canary = (payload.get("steps", {}) or {}).get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", False)) is True
    assert not (out_root / "state" / "panic_close_all.json").exists()


def test_main_spot_canary_precheck_blocks_insufficient_usdt(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "1.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

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
            raise AssertionError("precheck should block order submission")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    account_overview = steps.get("account_overview", {})
    assert isinstance(account_overview, dict)
    assert str(account_overview.get("market", "")) == "spot"
    assert float(account_overview.get("quote_available", 0.0)) == 1.0
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", True)) is False
    assert str(canary.get("reason", "")) == "precheck_insufficient_quote_balance"


def test_main_signal_missing_blocks_live_order_without_manual_override(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    calls = {"exchange_info": 0, "place_market_order": 0}

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            _ = symbol
            calls["exchange_info"] += 1
            return {"symbols": []}

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "20.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

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
            calls["place_market_order"] += 1
            raise AssertionError("signal_missing should block order submission")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
        ],
    )
    rc = mod.main()
    assert rc == 2
    assert calls["exchange_info"] == 0
    assert calls["place_market_order"] == 0

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    selection = steps.get("signal_selection", {})
    assert isinstance(selection, dict)
    assert bool(selection.get("signal_missing", False)) is True
    canary_plan = steps.get("canary_plan", {})
    assert isinstance(canary_plan, dict)
    assert bool(canary_plan.get("skipped", False)) is True
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert str(canary.get("reason", "")) == "signal_missing"


def test_main_signal_selection_surfaces_blocked_candidate_details(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 600.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            _ = symbol
            return {"symbols": []}

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "10.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

        def place_market_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: float,
            client_order_id: str,
            quote_order_qty: float | None = None,
        ) -> dict[str, object]:
            raise AssertionError(f"unexpected order {symbol} {side} {quantity} {client_order_id} {quote_order_qty}")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BNBUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"
    review_dir = out_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306T000000Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "tickets": [
                    {
                        "symbol": "BNBUSDT",
                        "date": "2026-03-03",
                        "age_days": 3,
                        "allowed": False,
                        "reasons": ["confidence_below_threshold", "size_below_min_notional"],
                        "signal": {"side": "LONG", "confidence": 44.1, "convexity_ratio": 3.1},
                        "execution": {"mode": "SPOT_LONG_OR_SELL"},
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    selection = steps.get("signal_selection", {})
    assert isinstance(selection, dict)
    assert bool(selection.get("signal_missing", False)) is True
    blocked = selection.get("blocked_candidate", {})
    assert isinstance(blocked, dict)
    assert str(blocked.get("symbol", "")) == "BNBUSDT"
    assert list(blocked.get("ticket_reasons", [])) == ["confidence_below_threshold", "size_below_min_notional"]
    assert int(blocked.get("age_days", -1)) == 3


def test_main_consumes_ticket_artifact_without_manual_override(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    captured: dict[str, object] = {}

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "20.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

        def place_market_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: float,
            client_order_id: str,
            quote_order_qty: float | None = None,
        ) -> dict[str, object]:
            captured["symbol"] = symbol
            captured["side"] = side
            captured["quantity"] = quantity
            captured["quote_order_qty"] = quote_order_qty
            return {
                "symbol": symbol,
                "status": "FILLED",
                "executedQty": f"{quantity:.8f}",
                "cummulativeQuoteQty": "5.11",
                "side": side,
            }

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT", "ETHUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"
    review_dir = out_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "20260306T000000Z_signal_to_order_tickets.json").write_text(
        json.dumps(
            {
                "as_of": "2026-03-06",
                "tickets": [
                    {
                        "symbol": "ETHUSDT",
                        "allowed": False,
                        "reasons": ["size_below_min_notional"],
                        "signal": {"side": "SHORT", "confidence": 48.0, "convexity_ratio": 2.5},
                        "execution": {"mode": "HEDGE_ONLY"},
                    },
                    {
                        "symbol": "BTCUSDT",
                        "date": "2026-03-05",
                        "allowed": False,
                        "reasons": ["size_below_min_notional"],
                        "signal": {"side": "LONG", "confidence": 22.0, "convexity_ratio": 1.8},
                        "execution": {"mode": "SPOT_LONG_OR_SELL"},
                    },
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-06",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--canary-quote-usdt",
            "5.0",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert str(captured.get("symbol", "")) == "BTCUSDT"
    assert str(captured.get("side", "")) == "BUY"
    assert captured.get("quote_order_qty") is None

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    selection = steps.get("signal_selection", {})
    assert isinstance(selection, dict)
    assert str(selection.get("selection_kind", "")) == "ticket"
    assert str(selection.get("source", "")).endswith("_signal_to_order_tickets.json")
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", False)) is True


def test_main_spot_canary_buy_submits_explicit_quantity(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    captured: dict[str, object] = {}

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "20.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

        def place_market_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: float,
            client_order_id: str,
            quote_order_qty: float | None = None,
        ) -> dict[str, object]:
            captured["symbol"] = symbol
            captured["side"] = side
            captured["quantity"] = quantity
            captured["client_order_id"] = client_order_id
            captured["quote_order_qty"] = quote_order_qty
            return {
                "symbol": symbol,
                "status": "FILLED",
                "executedQty": f"{quantity:.8f}",
                "cummulativeQuoteQty": "5.11",
                "side": side,
            }

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-05",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
            "--canary-quote-usdt",
            "5.0",
            "--order-quantity",
            "0.00006",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert str(captured.get("symbol", "")) == "BTCUSDT"
    assert str(captured.get("side", "")) == "BUY"
    assert abs(float(captured.get("quantity", 0.0) or 0.0) - 0.00006) < 1e-12
    assert captured.get("quote_order_qty") is None

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", False)) is True


def test_main_portfolio_margin_um_probe_reads_account_and_trade_telemetry(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()

    class _FakePortfolioMarginUmClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_500.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "notional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {
                "accountEquity": "101.5",
                "accountStatus": "NORMAL",
                "assets": [
                    {
                        "asset": "USDT",
                        "availableBalance": "99.0",
                        "marginBalance": "100.2",
                    }
                ],
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "positionAmt": "0.010",
                        "positionSide": "BOTH",
                        "entryPrice": "72000",
                        "markPrice": "73500",
                        "notional": "735.0",
                        "leverage": "5",
                    }
                ],
            }

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return [{"symbol": symbol, "id": 1, "qty": "0.001"}]

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return [{"symbol": "BTCUSDT", "income": "3.25", "time": 1773273600000}]

        def place_market_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: float,
            client_order_id: str,
            quote_order_qty: float | None = None,
        ) -> dict[str, object]:
            raise AssertionError(f"unexpected order {symbol} {side} {quantity} {client_order_id} {quote_order_qty}")

    monkeypatch.setattr(mod, "BinancePortfolioMarginUmClient", _FakePortfolioMarginUmClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-12",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "portfolio_margin_um",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 0

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    account_overview = steps.get("account_overview", {})
    assert isinstance(account_overview, dict)
    assert str(account_overview.get("market", "")) == "portfolio_margin_um"
    assert float(account_overview.get("quote_available", 0.0)) == 99.0
    assert int(account_overview.get("position_count", 0)) == 1
    trade_telemetry = steps.get("trade_telemetry", {})
    assert isinstance(trade_telemetry, dict)
    assert int(trade_telemetry.get("trades", 0)) == 1
    assert int(trade_telemetry.get("income_rows", 0)) == 1
    assert trade_telemetry.get("trade_count_by_symbol", {}) == {"BTCUSDT": 1}
    assert trade_telemetry.get("income_count_by_symbol", {}) == {"BTCUSDT": 1}
    assert trade_telemetry.get("income_pnl_by_symbol", {}) == {"BTCUSDT": 3.25}
    assert trade_telemetry.get("income_pnl_by_day", {}) == {"2026-03-12": 3.25}
    live_snapshot_meta = steps.get("live_snapshot", {})
    assert isinstance(live_snapshot_meta, dict)
    assert str(live_snapshot_meta.get("path", "")).endswith("2026-03-12_portfolio_margin_um.json")
    live_snapshot = json.loads((out_root / "artifacts" / "broker_live_inbox" / "2026-03-12.json").read_text(encoding="utf-8"))
    assert str(live_snapshot.get("source", "")) == "binance_portfolio_margin_um"
    positions = live_snapshot.get("positions", [])
    assert isinstance(positions, list)
    assert float((positions[0] or {}).get("markPrice", 0.0)) == 73500.0
    live_snapshot_market = json.loads(
        (out_root / "artifacts" / "broker_live_inbox" / "2026-03-12_portfolio_margin_um.json").read_text(encoding="utf-8")
    )
    assert str(live_snapshot_market.get("source", "")) == "binance_portfolio_margin_um"
    latest_market = json.loads(
        (out_root / "review" / "latest_binance_live_takeover_portfolio_margin_um.json").read_text(encoding="utf-8")
    )
    assert str(latest_market.get("market", "")) == "portfolio_margin_um"


def test_main_portfolio_margin_um_blocks_live_order_in_read_only_mode(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    calls = {"place_market_order": 0}

    class _FakePortfolioMarginUmClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_500.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [
                            {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "notional": "5"},
                        ],
                    }
                ]
            }

        def account(self) -> dict[str, object]:
            return {
                "accountEquity": "101.5",
                "accountStatus": "NORMAL",
                "assets": [{"asset": "USDT", "availableBalance": "99.0", "marginBalance": "100.2"}],
                "positions": [],
            }

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, start_ms, end_ms, limit)
            return []

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

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
            calls["place_market_order"] += 1
            raise AssertionError("portfolio margin UM should remain read-only")

    monkeypatch.setattr(mod, "BinancePortfolioMarginUmClient", _FakePortfolioMarginUmClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-12",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "portfolio_margin_um",
            "--allow-live-order",
            "--order-symbol",
            "BTCUSDT",
            "--order-side",
            "BUY",
        ],
    )
    rc = mod.main()
    assert rc == 2
    assert calls["place_market_order"] == 0

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    account_overview = steps.get("account_overview", {})
    assert isinstance(account_overview, dict)
    assert str(account_overview.get("market", "")) == "portfolio_margin_um"
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", True)) is False
    assert str(canary.get("reason", "")) == "portfolio_margin_um_read_only_mode"


def test_main_spot_trade_window_over_24h_fetches_trades_in_daily_slices(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    trade_calls: list[tuple[int, int]] = []

    class _FakeSpotClient:
        def __init__(self, **kwargs) -> None:
            _ = kwargs

        def ping(self) -> dict[str, object]:
            return {}

        def ticker_price(self, symbol: str) -> float:
            _ = symbol
            return 73_000.0

        def exchange_info(self, symbol: str) -> dict[str, object]:
            return {"symbols": []}

        def account(self) -> dict[str, object]:
            return {"balances": [{"asset": "USDT", "free": "20.0", "locked": "0.0"}]}

        def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (symbol, limit)
            trade_calls.append((start_ms, end_ms))
            return [{"symbol": "BTCUSDT", "id": len(trade_calls), "time": start_ms + 1, "qty": "0.001"}]

        def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, object]]:
            _ = (start_ms, end_ms, limit)
            return []

        def place_market_order(
            self,
            *,
            symbol: str,
            side: str,
            quantity: float,
            client_order_id: str,
            quote_order_qty: float | None = None,
        ) -> dict[str, object]:
            raise AssertionError(f"unexpected order {symbol} {side} {quantity} {client_order_id} {quote_order_qty}")

    monkeypatch.setattr(mod, "BinanceSpotClient", _FakeSpotClient)
    monkeypatch.setattr(mod, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"validation": {"micro_capture_daemon_symbols": ["BTCUSDT"]}}, sort_keys=False),
        encoding="utf-8",
    )
    out_root = tmp_path / "output"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "binance_live_takeover.py",
            "--date",
            "2026-03-12",
            "--config",
            str(cfg_path),
            "--output-root",
            str(out_root),
            "--market",
            "spot",
            "--trade-window-hours",
            "48",
        ],
    )
    rc = mod.main()
    assert rc == 0
    assert len(trade_calls) == 2
    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    telemetry = (payload.get("steps", {}) or {}).get("trade_telemetry", {})
    assert int(telemetry.get("query_slice_hours", 0)) == 24
    assert int(telemetry.get("trade_slice_count", 0)) == 2
    assert int(telemetry.get("trades", 0)) == 2


def test_detect_lie_daemon_pid_parses_ps_output(monkeypatch) -> None:
    mod = _load_module()

    class _Proc:
        returncode = 0
        stdout = " 1234 /usr/bin/python3 /x/venv/bin/lie run-daemon --poll-seconds 30\n"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: _Proc())
    assert str(mod.detect_lie_daemon_pid()) == "1234"


def test_load_binance_credentials_from_env_file_reads_values(tmp_path: Path) -> None:
    mod = _load_module()
    env_file = tmp_path / "binance.env"
    env_file.write_text(
        "BINANCE_API_KEY='k-123'\nBINANCE_SECRET='s$#-456'\n",
        encoding="utf-8",
    )
    creds = mod.load_binance_credentials_from_env_file(env_file)
    assert str(creds.get("BINANCE_API_KEY", "")) == "k-123"
    assert str(creds.get("BINANCE_SECRET", "")) == "s$#-456"


def test_resolve_binance_credentials_uses_env_file_when_daemon_secret_missing(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    env_file = tmp_path / "binance.env"
    env_file.write_text(
        "BINANCE_API_KEY='k-abc'\nBINANCE_SECRET='s-def'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BINANCE_CREDENTIALS_ENV_FILE", str(env_file))
    monkeypatch.setattr(mod, "load_binance_credentials_from_daemon", lambda: {"BINANCE_API_KEY": "k-abc"})
    monkeypatch.setenv("BINANCE_API_KEY", "")
    monkeypatch.setenv("BINANCE_SECRET", "")

    api_key, api_secret, source = mod.resolve_binance_credentials(True)
    assert api_key == "k-abc"
    assert api_secret == "s-def"
    assert source == "env_file"
