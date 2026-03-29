#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binance_live_common import now_utc_iso, write_json
from bybit_live_common import BybitFuturesClient, BybitSignedClient, resolve_bybit_credentials
from tv_basis_arb_common import load_strategy_definition
from tv_basis_arb_state import load_recovery_state
from venue_capability_common import evaluate_live_route_for_requirements


def _latest_ready_check_path(output_root: Path) -> Path:
    return output_root / "review" / "latest_bybit_live_route_ready_check.json"


def _has_active_recovery(output_root: Path, *, strategy_id: str, venue: str) -> bool:
    payload = load_recovery_state(output_root / "state" / "tv_basis_arb_recovery.json")
    for row in payload.get("recoveries", {}).values():
        if not isinstance(row, dict):
            continue
        if str(row.get("strategy_id", "")) != strategy_id:
            continue
        if str(row.get("execution_venue", "")).strip().lower() != venue:
            continue
        if str(row.get("status", "")).strip().lower() == "needs_recovery":
            return True
    return False


def _account_ready(spot_client: Any, futures_client: Any) -> bool:
    return isinstance(spot_client.wallet_balance(), dict) and isinstance(futures_client.futures_account_info(), dict)


def run_ready_check(
    *,
    output_root: Path | str,
    spot_client: Any | None = None,
    futures_client: Any | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    strategy = load_strategy_definition("tv_basis_btc_spot_perp_v1")
    preferred_venue = str(strategy.get("preferred_venue", "")).strip().lower()
    gate = strategy.get("gate", {}) if isinstance(strategy.get("gate", {}), dict) else {}

    if preferred_venue != "bybit":
        payload = {
            "ok": False,
            "status": "blocked",
            "venue": "bybit",
            "strategy_id": "tv_basis_btc_spot_perp_v1",
            "required_checks": {
                "venue_capability_ready": False,
                "route_selected_bybit": False,
                "baseqty_budget_contract_ready": False,
                "no_active_recovery": False,
                "account_ready": False,
            },
            "blockers": ["route_not_selected_bybit"],
        }
        write_json(_latest_ready_check_path(root), payload)
        return payload

    live_route = evaluate_live_route_for_requirements(
        path=root / "state" / "venue_capabilities.json",
        venue="bybit",
        required_statuses={
            "spot_signed_trade_status": "ready",
            "futures_signed_trade_status": "ready",
        },
    )
    venue_capability_ready = str(live_route.get("live_route_status", "")) == "live_ready"
    route_selected_bybit = True
    baseqty_budget_contract_ready = float(gate.get("target_base_qty", 0.0)) > 0.0 and float(gate.get("max_quote_budget_usdt", 0.0)) > 0.0
    no_active_recovery = not _has_active_recovery(root, strategy_id="tv_basis_btc_spot_perp_v1", venue="bybit")

    if spot_client is None or futures_client is None:
        api_key, api_secret, _ = resolve_bybit_credentials(True)
        spot_client = BybitSignedClient(api_key=api_key, api_secret=api_secret)
        futures_client = BybitFuturesClient(api_key=api_key, api_secret=api_secret)
    try:
        account_ready = _account_ready(spot_client, futures_client)
    except Exception:
        account_ready = False

    required_checks = {
        "venue_capability_ready": venue_capability_ready,
        "route_selected_bybit": route_selected_bybit,
        "baseqty_budget_contract_ready": baseqty_budget_contract_ready,
        "no_active_recovery": no_active_recovery,
        "account_ready": account_ready,
    }
    ok = all(required_checks.values())
    payload = {
        "ok": bool(ok),
        "status": "canary_ready" if ok else "blocked",
        "venue": "bybit",
        "strategy_id": "tv_basis_btc_spot_perp_v1",
        "required_checks": required_checks,
        "blockers": [] if ok else list(live_route.get("venue_blockers", [])) + [key for key, value in required_checks.items() if not value],
        "generated_at_utc": now_utc_iso(),
    }
    write_json(_latest_ready_check_path(root), payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bybit live route readiness check.")
    parser.add_argument("--output-root", default="output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_ready_check(output_root=Path(args.output_root))
    return 0 if bool(payload.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
