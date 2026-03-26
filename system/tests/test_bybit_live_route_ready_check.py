from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "scripts" / "bybit_live_route_ready_check.py"
    spec = importlib.util.spec_from_file_location("bybit_live_route_ready_check", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_capability(root: Path, *, status: str = "live_ready") -> None:
    path = root / "state" / "venue_capabilities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    checked_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "schema_version": 1,
        "venues": {
            "bybit": {
                "checked_at_utc": checked_at_utc,
                "account_scope": "openclaw-system:process_env",
                "status": status,
                "spot_signed_read_status": "ready",
                "spot_signed_trade_status": "ready",
                "futures_signed_read_status": "ready",
                "futures_signed_trade_status": "ready",
                "ip_restrict": False,
                "blockers": [] if status == "live_ready" else ["blocked"],
                "raw": {},
            }
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _FakeBybitSpotClient:
    def wallet_balance(self) -> dict[str, object]:
        return {"retCode": 0, "result": {}}


class _FakeBybitFuturesClient:
    def futures_account_info(self) -> dict[str, object]:
        return {"retCode": 0, "result": {}}


def test_ready_check_writes_latest_artifact(tmp_path: Path) -> None:
    mod = _load_module()
    _write_capability(tmp_path, status="live_ready")

    payload = mod.run_ready_check(
        output_root=tmp_path,
        spot_client=_FakeBybitSpotClient(),
        futures_client=_FakeBybitFuturesClient(),
    )

    latest = tmp_path / "review" / "latest_bybit_live_route_ready_check.json"
    assert latest.exists()
    assert payload["status"] in {"canary_ready", "blocked"}


def test_ready_check_blocks_when_bybit_capability_not_ready(tmp_path: Path) -> None:
    mod = _load_module()
    _write_capability(tmp_path, status="live_blocked")

    payload = mod.run_ready_check(
        output_root=tmp_path,
        spot_client=_FakeBybitSpotClient(),
        futures_client=_FakeBybitFuturesClient(),
    )

    assert payload["status"] == "blocked"
    assert payload["required_checks"]["venue_capability_ready"] is False


def test_ready_check_can_mark_canary_ready(tmp_path: Path) -> None:
    mod = _load_module()
    _write_capability(tmp_path, status="live_ready")

    payload = mod.run_ready_check(
        output_root=tmp_path,
        spot_client=_FakeBybitSpotClient(),
        futures_client=_FakeBybitFuturesClient(),
    )

    assert payload["ok"] is True
    assert payload["status"] == "canary_ready"


def test_ready_check_marks_all_required_checks_before_canary_ready(tmp_path: Path) -> None:
    mod = _load_module()
    _write_capability(tmp_path, status="live_ready")

    payload = mod.run_ready_check(
        output_root=tmp_path,
        spot_client=_FakeBybitSpotClient(),
        futures_client=_FakeBybitFuturesClient(),
    )

    assert payload["required_checks"]["venue_capability_ready"] is True
    assert payload["required_checks"]["route_selected_bybit"] is True
    assert payload["required_checks"]["baseqty_budget_contract_ready"] is True
    assert payload["required_checks"]["no_active_recovery"] is True
    assert payload["required_checks"]["account_ready"] is True
