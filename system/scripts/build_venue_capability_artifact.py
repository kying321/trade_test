#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from binance_live_common import (  # noqa: E402
    BinanceSpotClient,
    BinanceUsdMMarketClient,
    now_utc_iso,
    resolve_binance_credentials,
    write_json,
)
from venue_capability_common import SCHEMA_VERSION  # noqa: E402

SYSTEM_ROOT = SCRIPT_DIR.parents[0]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
ARTIFACT_REL_PATH = Path("state") / "venue_capabilities.json"


def _status_from_probe(ok: bool) -> str:
    return "ready" if ok else "blocked"


def _probe_spot_signed_read(spot_client: Any) -> tuple[bool, str | None]:
    try:
        payload = spot_client.account()
    except Exception as exc:
        return False, f"spot_signed_read_probe_failed:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return False, "spot_signed_read_probe_invalid_payload"
    return True, None


def _probe_spot_api_restrictions(spot_client: Any) -> tuple[dict[str, Any], str | None]:
    try:
        payload = spot_client._request(
            method="GET",
            path="/sapi/v1/account/apiRestrictions",
            params={},
            signed=True,
        )
    except Exception as exc:
        return {}, f"apiRestrictions_probe_failed:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return {}, "apiRestrictions_probe_failed:invalid_payload"
    return payload, None


def _probe_futures_signed_read(futures_client: Any) -> tuple[bool, str | None]:
    try:
        payload = futures_client._request(
            method="GET",
            path="/fapi/v2/account",
            params={},
            signed=True,
        )
    except Exception as exc:
        return False, f"futures_signed_read_probe_failed:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return False, "futures_signed_read_probe_invalid_payload"
    return True, None


def build_binance_venue_payload(
    *,
    spot_client: Any,
    futures_client: Any,
    checked_at_utc: str | None = None,
    account_scope: str = "openclaw-system:daemon_env",
) -> dict[str, Any]:
    blockers: list[str] = []

    spot_read_ok, spot_read_err = _probe_spot_signed_read(spot_client)
    if spot_read_err:
        blockers.append(spot_read_err)
    spot_read_status = _status_from_probe(spot_read_ok)
    restrictions, restrictions_err = _probe_spot_api_restrictions(spot_client)
    if restrictions_err:
        blockers.append(restrictions_err)
    restrictions_available = restrictions_err is None and isinstance(restrictions, dict)
    raw_api_restrictions = restrictions if restrictions_available else None
    ip_restrict = restrictions.get("ipRestrict") if restrictions_available else None
    enable_spot_trade_raw = restrictions.get("enableSpotAndMarginTrading") if restrictions_available else None
    enable_futures_raw = restrictions.get("enableFutures") if restrictions_available else None
    restrictions_complete = restrictions_available and isinstance(enable_spot_trade_raw, bool) and isinstance(enable_futures_raw, bool)
    if restrictions_available and not restrictions_complete:
        blockers.append("apiRestrictions_incomplete")

    if restrictions_complete:
        if bool(enable_spot_trade_raw):
            spot_trade_status = "ready"
        else:
            spot_trade_status = "blocked"
            blockers.append("enableSpotAndMarginTrading=false")
    else:
        spot_trade_status = "unknown"

    futures_read_probe_ok = False
    futures_read_probe_err: str | None = None
    if restrictions_complete and bool(enable_futures_raw):
        futures_read_probe_ok, futures_read_probe_err = _probe_futures_signed_read(futures_client)
        if futures_read_probe_err:
            blockers.append(futures_read_probe_err)
        futures_read_status = _status_from_probe(futures_read_probe_ok)
        futures_trade_status = futures_read_status
    elif restrictions_complete:
        futures_read_status = "blocked"
        futures_trade_status = "blocked"
        blockers.append("enableFutures=false")
    else:
        futures_read_probe_ok, futures_read_probe_err = _probe_futures_signed_read(futures_client)
        if futures_read_probe_err:
            blockers.append(futures_read_probe_err)
        futures_read_status = _status_from_probe(futures_read_probe_ok)
        futures_trade_status = "unknown"

    capability_statuses = (
        spot_read_status,
        spot_trade_status,
        futures_read_status,
        futures_trade_status,
    )
    if any(status == "unknown" for status in capability_statuses):
        status = "unknown"
    elif all(status == "ready" for status in capability_statuses):
        status = "live_ready"
    else:
        status = "live_blocked"

    deduped_blockers = list(dict.fromkeys(str(item).strip() for item in blockers if str(item).strip()))
    return {
        "checked_at_utc": str(checked_at_utc or now_utc_iso()),
        "account_scope": str(account_scope),
        "status": status,
        "spot_signed_read_status": spot_read_status,
        "spot_signed_trade_status": spot_trade_status,
        "futures_signed_read_status": futures_read_status,
        "futures_signed_trade_status": futures_trade_status,
        "ip_restrict": ip_restrict,
        "blockers": deduped_blockers,
        "raw": {"apiRestrictions": raw_api_restrictions},
    }


def build_venue_capability_payload(
    *,
    spot_client: Any,
    futures_client: Any,
    checked_at_utc: str | None = None,
    account_scope: str = "openclaw-system:daemon_env",
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "venues": {
            "binance": build_binance_venue_payload(
                spot_client=spot_client,
                futures_client=futures_client,
                checked_at_utc=checked_at_utc,
                account_scope=account_scope,
            )
        },
    }


def build_venue_capability_artifact(
    *,
    output_root: Path,
    spot_client: Any,
    futures_client: Any,
    checked_at_utc: str | None = None,
    account_scope: str = "openclaw-system:daemon_env",
) -> Path:
    artifact_path = Path(output_root) / ARTIFACT_REL_PATH
    payload = build_venue_capability_payload(
        spot_client=spot_client,
        futures_client=futures_client,
        checked_at_utc=checked_at_utc,
        account_scope=account_scope,
    )
    write_json(artifact_path, payload)
    return artifact_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build venue capability artifact (single-writer).")
    parser.add_argument("--venue", default="binance")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--checked-at-utc", default="")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--allow-daemon-env-fallback", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if str(args.venue).strip().lower() != "binance":
        raise SystemExit("unsupported venue")
    output_root = Path(args.output_root).expanduser().resolve()

    api_key, api_secret, cred_source = resolve_binance_credentials(bool(args.allow_daemon_env_fallback))
    spot_client = BinanceSpotClient(
        api_key=api_key,
        api_secret=api_secret,
        timeout_ms=int(args.timeout_ms),
        rate_limit_per_minute=int(args.rpm),
    )
    futures_client = BinanceUsdMMarketClient(
        api_key=api_key,
        api_secret=api_secret,
        timeout_ms=int(args.timeout_ms),
        rate_limit_per_minute=int(args.rpm),
    )

    build_venue_capability_artifact(
        output_root=output_root,
        spot_client=spot_client,
        futures_client=futures_client,
        checked_at_utc=str(args.checked_at_utc).strip() or None,
        account_scope=f"openclaw-system:{cred_source}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
