from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "venue_capability_common.py"
    spec = importlib.util.spec_from_file_location("venue_capability_common", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_artifact(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ready_binance_payload() -> dict:
    return {
        "schema_version": 1,
        "venues": {
            "binance": {
                "checked_at_utc": "2026-03-26T11:00:00Z",
                "status": "live_ready",
                "spot_signed_read_status": "ready",
                "spot_signed_trade_status": "ready",
                "futures_signed_read_status": "ready",
                "futures_signed_trade_status": "ready",
                "blockers": [],
            }
        },
    }


def _required_trade_capabilities() -> dict[str, str]:
    return {
        "spot_signed_trade_status": "ready",
        "futures_signed_trade_status": "ready",
    }


def test_missing_capability_artifact_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    result = mod.evaluate_live_route_for_requirements(
        path=tmp_path / "missing.json",
        venue="binance",
        required_statuses=_required_trade_capabilities(),
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_missing"
    assert result["venue"] == "binance"
    assert result["venue_blockers"] == []


def test_stale_capability_artifact_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:20:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_stale"


def test_incomplete_schema_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    del payload["venues"]["binance"]["futures_signed_trade_status"]
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_incomplete"


def test_ready_capability_allows_live_ready_candidate(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    _write_artifact(artifact_path, _ready_binance_payload())

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:10:00Z",
    )

    assert result["live_route_status"] == "live_ready"
    assert result["live_route_reason"] is None
    assert result["venue"] == "binance"
    assert result["venue_blockers"] == []


def test_unknown_capability_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    payload["venues"]["binance"]["futures_signed_trade_status"] = "unknown"
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_unknown"


def test_invalid_checked_at_utc_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    payload["venues"]["binance"]["checked_at_utc"] = "invalid-ts"
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_stale"


def test_blocked_capability_maps_to_live_blocked_with_venue_capability_reason(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    payload["venues"]["binance"]["status"] = "live_blocked"
    payload["venues"]["binance"]["futures_signed_trade_status"] = "blocked"
    payload["venues"]["binance"]["blockers"] = ["enableFutures=false"]
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability"
    assert result["venue_blockers"] == ["enableFutures=false"]


def test_default_artifact_matches_contract() -> None:
    mod = _load_module()
    artifact = mod.default_capability_artifact()

    assert artifact == {"schema_version": 1, "venues": {}}
    assert "unknown_stale" not in json.dumps(artifact)


def test_unknown_venue_status_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    payload["venues"]["binance"]["status"] = "unknown"
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_unknown"


def test_dry_only_venue_status_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    payload["venues"]["binance"]["status"] = "dry_only"
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability"


def test_future_checked_at_utc_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    payload["venues"]["binance"]["checked_at_utc"] = "2026-03-26T11:10:00Z"
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_stale"


def test_invalid_now_utc_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    _write_artifact(artifact_path, _ready_binance_payload())

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="not-a-time",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_incomplete"


def test_invalid_schema_version_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    payload["schema_version"] = "oops"
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_incomplete"


def test_missing_schema_version_maps_to_live_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    payload = _ready_binance_payload()
    del payload["schema_version"]
    _write_artifact(artifact_path, payload)

    result = mod.evaluate_live_route_for_requirements(
        path=artifact_path,
        venue="binance",
        required_statuses=_required_trade_capabilities(),
        now_utc="2026-03-26T11:00:01Z",
    )

    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_incomplete"


def test_noncanonical_schema_version_types_are_rejected(tmp_path: Path) -> None:
    mod = _load_module()
    artifact_path = tmp_path / "venue_capability.json"
    for bad_value in ("1", True):
        payload = _ready_binance_payload()
        payload["schema_version"] = bad_value
        _write_artifact(artifact_path, payload)

        result = mod.evaluate_live_route_for_requirements(
            path=artifact_path,
            venue="binance",
            required_statuses=_required_trade_capabilities(),
            now_utc="2026-03-26T11:00:01Z",
        )

        assert result["live_route_status"] == "live_blocked"
        assert result["live_route_reason"] == "venue_capability_incomplete"
