from __future__ import annotations

import importlib.util
from pathlib import Path


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
