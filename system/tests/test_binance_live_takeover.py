from __future__ import annotations

import importlib.util
import json
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


def test_binance_spot_client_buy_order_uses_quote_order_qty() -> None:
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
    assert "quoteOrderQty" in params
    assert "quantity" not in params


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
            return {"balances": []}

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
        ],
    )
    rc = mod.main()
    assert rc == 2

    payload = json.loads((out_root / "review" / "latest_binance_live_takeover.json").read_text(encoding="utf-8"))
    assert bool(payload.get("ok")) is False
    assert str(payload.get("mode", "")) == "live_ready"
    steps = payload.get("steps", {})
    assert isinstance(steps, dict)
    canary = steps.get("canary_order", {})
    assert isinstance(canary, dict)
    assert bool(canary.get("executed", True)) is False
    assert str(canary.get("reason", "")) == "exchange_reject"


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
