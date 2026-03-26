from __future__ import annotations

import importlib.util
import hashlib
import hmac
import json
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "bybit_live_common.py"
)


class _FakeBybitSpotClient:
    def __init__(
        self,
        *,
        spot_trade: bool = True,
        futures_trade: bool = True,
        wallet_payload: object | None = None,
        api_key_info_payload: object | None = None,
        api_key_info_error: Exception | None = None,
    ) -> None:
        self._spot_trade = bool(spot_trade)
        self._futures_trade = bool(futures_trade)
        self._wallet_payload = wallet_payload
        self._api_key_info_payload = api_key_info_payload
        self._api_key_info_error = api_key_info_error

    def wallet_balance(self) -> object:
        if self._wallet_payload is not None:
            return self._wallet_payload
        return {"retCode": 0}

    def api_key_info(self) -> dict[str, object]:
        if self._api_key_info_error is not None:
            raise self._api_key_info_error
        if self._api_key_info_payload is not None:
            return self._api_key_info_payload  # type: ignore[return-value]
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
    def __init__(self, payload: object | None = None) -> None:
        self._payload = payload

    def futures_account_info(self) -> object:
        if self._payload is not None:
            return self._payload
        return {"retCode": 0}


def _load_module(path: Path, name: str):
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bybit_futures_blocked_when_trade_permission_missing() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    payload = module.build_bybit_venue_payload(
        spot_client=_FakeBybitSpotClient(spot_trade=True, futures_trade=False),
        futures_client=_FakeBybitFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    assert payload["status"] == "live_blocked"
    assert payload["spot_signed_trade_status"] == "ready"
    assert payload["futures_signed_read_status"] == "blocked"
    assert payload["futures_signed_trade_status"] == "blocked"
    assert "contractTrade=false" in payload["blockers"]


def test_bybit_live_ready_positive() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    payload = module.build_bybit_venue_payload(
        spot_client=_FakeBybitSpotClient(spot_trade=True, futures_trade=True),
        futures_client=_FakeBybitFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
        account_scope="openclaw-system:process_env",
    )

    assert payload["status"] == "live_ready"
    assert payload["spot_signed_read_status"] == "ready"
    assert payload["spot_signed_trade_status"] == "ready"
    assert payload["futures_signed_trade_status"] == "ready"
    assert payload["blockers"] == []


def test_bybit_client_timeout_is_capped_at_5000ms() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    client = module.BybitSignedClient(api_key="k", api_secret="s", timeout_ms=9000)
    assert int(client.timeout_ms) == 5000


def test_bybit_contract_trade_requires_order_permission() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    spot = _FakeBybitSpotClient(spot_trade=True, futures_trade=True)
    payload = spot.api_key_info()
    payload["result"]["permissions"]["ContractTrade"] = ["Position"]
    spot.api_key_info = lambda: payload  # type: ignore[method-assign]

    venue = module.build_bybit_venue_payload(
        spot_client=spot,
        futures_client=_FakeBybitFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    assert venue["status"] == "live_blocked"
    assert venue["futures_signed_trade_status"] == "blocked"
    assert "contractTrade=false" in venue["blockers"]


def test_bybit_payload_maps_ips_wildcard_to_ip_restrict_false() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    payload = module.build_bybit_venue_payload(
        spot_client=_FakeBybitSpotClient(spot_trade=True, futures_trade=True),
        futures_client=_FakeBybitFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    assert payload["ip_restrict"] is False


def test_bybit_permission_probe_failure_maps_to_unknown() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    payload = module.build_bybit_venue_payload(
        spot_client=_FakeBybitSpotClient(api_key_info_error=RuntimeError("permission_probe_failed")),
        futures_client=_FakeBybitFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    assert payload["status"] == "unknown"
    assert payload["spot_signed_trade_status"] == "unknown"
    assert payload["futures_signed_trade_status"] == "unknown"
    assert any("api_permissions_probe_failed" in item for item in payload["blockers"])


def test_bybit_permission_payload_incomplete_maps_to_unknown() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    payload = module.build_bybit_venue_payload(
        spot_client=_FakeBybitSpotClient(api_key_info_payload={"retCode": 0, "result": {"permissions": {"Spot": ["SpotTrade"]}}}),
        futures_client=_FakeBybitFuturesClientReady(),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    assert payload["status"] == "unknown"
    assert payload["spot_signed_trade_status"] == "unknown"
    assert payload["futures_signed_trade_status"] == "unknown"
    assert "api_permissions_incomplete" in payload["blockers"]


def test_bybit_non_dict_read_payloads_add_probe_blockers() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    payload = module.build_bybit_venue_payload(
        spot_client=_FakeBybitSpotClient(wallet_payload=["bad"]),
        futures_client=_FakeBybitFuturesClientReady(payload=["bad"]),
        checked_at_utc="2026-03-26T13:00:00Z",
    )

    assert payload["status"] == "live_blocked"
    assert payload["spot_signed_read_status"] == "blocked"
    assert payload["futures_signed_read_status"] == "blocked"
    assert any("spot_signed_read_probe_invalid_payload" in item for item in payload["blockers"])
    assert any("futures_signed_read_probe_invalid_payload" in item for item in payload["blockers"])


def test_bybit_signed_client_rejects_missing_retcode() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        return _FakeHTTPResponse({"result": {}})

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        client = module.BybitSignedClient(api_key="test-key", api_secret="test-secret", timeout_ms=5000)
        try:
            client.api_key_info()
        except RuntimeError as exc:
            assert "retCode_None" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for missing retCode")


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_bybit_signed_client_uses_official_v5_auth_headers() -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    captured: dict[str, object] = {}

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _FakeHTTPResponse({"retCode": 0, "result": {}})

    with patch.object(module, "now_epoch_ms", return_value=1700000000123), patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        client = module.BybitSignedClient(api_key="test-key", api_secret="test-secret", timeout_ms=5000)
        client.api_key_info()

    query = ""
    expected_plain = f"1700000000123test-key5000{query}"
    expected_sign = hmac.new(b"test-secret", expected_plain.encode("utf-8"), hashlib.sha256).hexdigest()
    assert str(captured["url"]).endswith("/v5/user/query-api")
    headers = captured["headers"]
    assert headers["x-bapi-api-key"] == "test-key"
    assert headers["x-bapi-timestamp"] == "1700000000123"
    assert headers["x-bapi-recv-window"] == "5000"
    assert headers["x-bapi-sign"] == expected_sign


def test_load_bybit_credentials_from_env_file_reads_values(tmp_path: Path) -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    env_file = tmp_path / "bybit.env"
    env_file.write_text(
        "BYBIT_API_KEY='k-123'\nBYBIT_API_SECRET='s$#-456'\n",
        encoding="utf-8",
    )

    creds = module.load_bybit_credentials_from_env_file(env_file)

    assert str(creds.get("BYBIT_API_KEY", "")) == "k-123"
    assert str(creds.get("BYBIT_SECRET", "")) == "s$#-456"


def test_resolve_bybit_credentials_uses_env_file_when_daemon_secret_missing(monkeypatch, tmp_path: Path) -> None:
    module = _load_module(SCRIPT_PATH, "bybit_live_common")
    env_file = tmp_path / "bybit.env"
    env_file.write_text(
        "BYBIT_API_KEY='k-abc'\nBYBIT_API_SECRET='s-def'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BYBIT_CREDENTIALS_ENV_FILE", str(env_file))
    monkeypatch.setattr(module, "load_bybit_credentials_from_daemon", lambda: {"BYBIT_API_KEY": "k-abc"})
    monkeypatch.setenv("BYBIT_API_KEY", "")
    monkeypatch.setenv("BYBIT_API_SECRET", "")
    monkeypatch.setenv("BYBIT_SECRET", "")

    api_key, api_secret, source = module.resolve_bybit_credentials(True)

    assert api_key == "k-abc"
    assert api_secret == "s-def"
    assert source == "env_file"
