from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_venue_capability_artifact.py"
)
VENUE_COMMON_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "venue_capability_common.py"
)


class _FakeSpotClient:
    def __init__(
        self,
        *,
        enable_futures: bool = False,
        enable_spot_trade: bool = True,
        restrictions_error: Exception | None = None,
        restrictions_payload: dict[str, object] | None = None,
        account_payload: object | None = None,
    ) -> None:
        self._enable_futures = bool(enable_futures)
        self._enable_spot_trade = bool(enable_spot_trade)
        self._restrictions_error = restrictions_error
        self._restrictions_payload = restrictions_payload
        self._account_payload = account_payload

    def account(self) -> object:
        if self._account_payload is not None:
            return self._account_payload
        return {"canTrade": True}

    def _request(self, *, method: str, path: str, params=None, signed: bool = False):
        assert method == "GET"
        assert path == "/sapi/v1/account/apiRestrictions"
        assert signed is True
        if self._restrictions_error is not None:
            raise self._restrictions_error
        if self._restrictions_payload is not None:
            return self._restrictions_payload
        return {
            "enableSpotAndMarginTrading": self._enable_spot_trade,
            "enableFutures": self._enable_futures,
            "ipRestrict": False,
        }


class _FakeFuturesClientReady:
    def _request(self, *, method: str, path: str, params=None, signed: bool = False):
        assert method == "GET"
        assert path == "/fapi/v2/account"
        assert signed is True
        return {"canTrade": True}


class _FakeFuturesClientBlocked:
    def _request(self, *, method: str, path: str, params=None, signed: bool = False):
        raise RuntimeError("futures_probe_denied")


class _FakeFuturesClientInvalid:
    def _request(self, *, method: str, path: str, params=None, signed: bool = False):
        return ["not-a-dict"]


class _FakeBybitSpotClient:
    def __init__(
        self,
        *,
        spot_trade: bool = True,
        futures_trade: bool = True,
        wallet_payload: object | None = None,
    ) -> None:
        self._spot_trade = bool(spot_trade)
        self._futures_trade = bool(futures_trade)
        self._wallet_payload = wallet_payload

    def wallet_balance(self) -> object:
        if self._wallet_payload is not None:
            return self._wallet_payload
        return {"retCode": 0}

    def api_key_info(self) -> dict[str, object]:
        return {
            "retCode": 0,
            "result": {
                "permissions": {
                    "Spot": ["SpotTrade"] if self._spot_trade else [],
                    "ContractTrade": ["Order", "Position"] if self._futures_trade else [],
                },
                "ips": ["*"],
            },
        }


class _FakeBybitFuturesClientReady:
    def futures_account_info(self) -> dict[str, object]:
        return {"retCode": 0}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binance_futures_blocked_when_enable_futures_false() -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    venue_payload = module.build_binance_venue_payload(
        spot_client=_FakeSpotClient(enable_futures=False),
        futures_client=_FakeFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    assert venue_payload["status"] == "live_blocked"
    assert venue_payload["spot_signed_read_status"] == "ready"
    assert venue_payload["spot_signed_trade_status"] == "ready"
    assert venue_payload["futures_signed_read_status"] == "blocked"
    assert venue_payload["futures_signed_trade_status"] == "blocked"
    assert "enableFutures=false" in venue_payload["blockers"]


def test_spot_ready_and_futures_blocked_combination() -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    venue_payload = module.build_binance_venue_payload(
        spot_client=_FakeSpotClient(enable_futures=True),
        futures_client=_FakeFuturesClientBlocked(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    assert venue_payload["status"] == "live_blocked"
    assert venue_payload["spot_signed_read_status"] == "ready"
    assert venue_payload["spot_signed_trade_status"] == "ready"
    assert venue_payload["futures_signed_read_status"] == "blocked"
    assert venue_payload["futures_signed_trade_status"] == "blocked"
    assert any("futures_signed_read_probe_failed" in x for x in venue_payload["blockers"])


def test_live_ready_happy_path() -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    venue_payload = module.build_binance_venue_payload(
        spot_client=_FakeSpotClient(enable_futures=True, enable_spot_trade=True),
        futures_client=_FakeFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
        account_scope="openclaw-system:daemon_env",
    )

    assert venue_payload["status"] == "live_ready"
    assert venue_payload["spot_signed_trade_status"] == "ready"
    assert venue_payload["futures_signed_trade_status"] == "ready"
    assert venue_payload["blockers"] == []


def test_spot_trade_status_does_not_infer_ready_when_spot_trade_permission_disabled() -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    venue_payload = module.build_binance_venue_payload(
        spot_client=_FakeSpotClient(enable_futures=True, enable_spot_trade=False),
        futures_client=_FakeFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
        account_scope="openclaw-system:daemon_env",
    )

    assert venue_payload["status"] == "live_blocked"
    assert venue_payload["spot_signed_read_status"] == "ready"
    assert venue_payload["spot_signed_trade_status"] == "blocked"
    assert "enableSpotAndMarginTrading=false" in venue_payload["blockers"]


def test_api_restrictions_probe_failure_does_not_masquerade_as_enable_futures_false() -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    venue_payload = module.build_binance_venue_payload(
        spot_client=_FakeSpotClient(restrictions_error=RuntimeError("restrictions_unavailable")),
        futures_client=_FakeFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
        account_scope="openclaw-system:daemon_env",
    )

    assert venue_payload["status"] == "unknown"
    assert venue_payload["spot_signed_trade_status"] == "unknown"
    assert venue_payload["futures_signed_read_status"] == "ready"
    assert venue_payload["futures_signed_trade_status"] == "unknown"
    assert any("apiRestrictions_probe_failed" in x for x in venue_payload["blockers"])
    assert "enableFutures=false" not in venue_payload["blockers"]


def test_partial_api_restrictions_payload_maps_trade_statuses_to_unknown() -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    venue_payload = module.build_binance_venue_payload(
        spot_client=_FakeSpotClient(restrictions_payload={"ipRestrict": False}),
        futures_client=_FakeFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
        account_scope="openclaw-system:daemon_env",
    )

    assert venue_payload["status"] == "unknown"
    assert venue_payload["spot_signed_trade_status"] == "unknown"
    assert venue_payload["futures_signed_trade_status"] == "unknown"
    assert any("apiRestrictions_incomplete" in x for x in venue_payload["blockers"])


def test_non_dict_signed_read_payload_adds_probe_blocker() -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    venue_payload = module.build_binance_venue_payload(
        spot_client=_FakeSpotClient(enable_futures=True, enable_spot_trade=True, account_payload=["bad"]),
        futures_client=_FakeFuturesClientInvalid(),
        checked_at_utc="2026-03-26T13:00:00Z",
        account_scope="openclaw-system:daemon_env",
    )

    assert venue_payload["spot_signed_read_status"] == "blocked"
    assert venue_payload["futures_signed_read_status"] == "blocked"
    assert any("spot_signed_read_probe_invalid_payload" in x for x in venue_payload["blockers"])
    assert any("futures_signed_read_probe_invalid_payload" in x for x in venue_payload["blockers"])


def test_writer_schema_is_task1_helper_compatible(tmp_path: Path) -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    common_module = _load_module(VENUE_COMMON_PATH, "venue_capability_common")

    output_root = tmp_path / "output"
    artifact_path = module.build_venue_capability_artifact(
        output_root=output_root,
        spot_client=_FakeSpotClient(enable_futures=False),
        futures_client=_FakeFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    result = common_module.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses={
            "spot_signed_trade_status": "ready",
            "futures_signed_trade_status": "ready",
        },
        now_utc="2026-03-26T13:05:00Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability"
    assert "enableFutures=false" in result["venue_blockers"]
    artifact = common_module.read_venue_capability_artifact(artifact_path)
    venue_payload = artifact["venues"]["binance"]
    assert venue_payload["account_scope"] == "openclaw-system:daemon_env"
    assert venue_payload["ip_restrict"] is False
    assert venue_payload["raw"]["apiRestrictions"]["enableFutures"] is False


def test_writer_unknown_capability_is_preserved_for_task1_helper(tmp_path: Path) -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    common_module = _load_module(VENUE_COMMON_PATH, "venue_capability_common")

    output_root = tmp_path / "output"
    artifact_path = module.build_venue_capability_artifact(
        output_root=output_root,
        spot_client=_FakeSpotClient(restrictions_error=RuntimeError("restrictions_unavailable")),
        futures_client=_FakeFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    result = common_module.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses={
            "spot_signed_trade_status": "ready",
            "futures_signed_trade_status": "ready",
        },
        now_utc="2026-03-26T13:05:00Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_unknown"


def test_build_venue_capability_payload_includes_bybit_when_probed() -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")

    payload = module.build_venue_capability_payload(
        spot_client=_FakeSpotClient(enable_futures=True),
        futures_client=_FakeFuturesClientReady(),
        bybit_spot_client=_FakeBybitSpotClient(spot_trade=True, futures_trade=True),
        bybit_futures_client=_FakeBybitFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    venues = payload["venues"]
    assert "binance" in venues
    assert "bybit" in venues
    assert venues["bybit"]["status"] == "live_ready"


def test_main_preserves_existing_binance_and_bybit_on_single_venue_probe(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")

    output_root = tmp_path / "output"
    artifact_path = output_root / "state" / "venue_capabilities.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "venues": {
                    "binance": {"checked_at_utc": "2026-03-26T10:00:00Z", "status": "live_blocked", "blockers": []},
                    "bybit": {"checked_at_utc": "2026-03-26T10:00:00Z", "status": "live_ready", "blockers": []},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "resolve_binance_credentials", lambda allow_daemon_env_fallback: ("k", "s", "process_env"))
    monkeypatch.setattr(module, "resolve_bybit_credentials", lambda allow_daemon_env_fallback: ("k2", "s2", "process_env"))
    monkeypatch.setattr(
        module,
        "BinanceSpotClient",
        lambda **kwargs: _FakeSpotClient(enable_futures=True, enable_spot_trade=True),
    )
    monkeypatch.setattr(module, "BinanceUsdMMarketClient", lambda **kwargs: _FakeFuturesClientReady())
    monkeypatch.setattr(module, "BybitSignedClient", lambda **kwargs: _FakeBybitSpotClient(spot_trade=True, futures_trade=True))
    monkeypatch.setattr(module, "BybitFuturesClient", lambda **kwargs: _FakeBybitFuturesClientReady())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_venue_capability_artifact.py",
            "--venue",
            "binance",
            "--output-root",
            str(output_root),
        ],
    )

    rc = module.main()
    assert rc == 0
    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(written["venues"].keys()) == {"binance", "bybit"}


def test_main_with_venue_bybit_preserves_existing_binance_entry(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")

    output_root = tmp_path / "output"
    artifact_path = output_root / "state" / "venue_capabilities.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "venues": {
                    "binance": {
                        "checked_at_utc": "2026-03-26T10:00:00Z",
                        "account_scope": "openclaw-system:daemon_env",
                        "status": "live_blocked",
                        "spot_signed_read_status": "ready",
                        "spot_signed_trade_status": "ready",
                        "futures_signed_read_status": "blocked",
                        "futures_signed_trade_status": "blocked",
                        "ip_restrict": False,
                        "blockers": ["enableFutures=false"],
                        "raw": {"apiRestrictions": {"enableFutures": False}},
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "resolve_bybit_credentials", lambda allow_daemon_env_fallback: ("k2", "s2", "process_env"))
    monkeypatch.setattr(module, "BybitSignedClient", lambda **kwargs: _FakeBybitSpotClient(spot_trade=True, futures_trade=True))
    monkeypatch.setattr(module, "BybitFuturesClient", lambda **kwargs: _FakeBybitFuturesClientReady())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_venue_capability_artifact.py",
            "--venue",
            "bybit",
            "--output-root",
            str(output_root),
        ],
    )

    rc = module.main()
    assert rc == 0
    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(written["venues"].keys()) == {"binance", "bybit"}
    assert written["venues"]["binance"]["status"] == "live_blocked"


def test_main_without_existing_artifact_keeps_schema_and_probed_venue_only(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(SCRIPT_PATH, "build_venue_capability_artifact")
    output_root = tmp_path / "output"

    monkeypatch.setattr(module, "resolve_bybit_credentials", lambda allow_daemon_env_fallback: ("k2", "s2", "process_env"))
    monkeypatch.setattr(module, "BybitSignedClient", lambda **kwargs: _FakeBybitSpotClient(spot_trade=True, futures_trade=True))
    monkeypatch.setattr(module, "BybitFuturesClient", lambda **kwargs: _FakeBybitFuturesClientReady())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_venue_capability_artifact.py",
            "--venue",
            "bybit",
            "--output-root",
            str(output_root),
        ],
    )

    rc = module.main()
    assert rc == 0
    written = json.loads((output_root / "state" / "venue_capabilities.json").read_text(encoding="utf-8"))
    assert written["schema_version"] == 1
    assert "bybit" in written["venues"]
    assert "binance" not in written["venues"]
    assert "binance" not in written["venues"]
