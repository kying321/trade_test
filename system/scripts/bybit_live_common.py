#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

from binance_live_common import (  # noqa: F401
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DEFAULT_TIMEOUT_MS,
    TokenBucket,
    detect_lie_daemon_pid,
    load_string_env,
    now_epoch_ms,
    now_utc_iso,
)


def load_bybit_credentials_from_daemon() -> dict[str, str]:
    out: dict[str, str] = {}
    pid = detect_lie_daemon_pid()
    if not pid:
        return out
    env_path = Path("/proc") / pid / "environ"
    if not env_path.exists():
        return out
    try:
        raw = env_path.read_bytes().decode("utf-8", errors="ignore")
    except Exception:
        return out
    for row in raw.split("\x00"):
        if "=" not in row:
            continue
        k, v = row.split("=", 1)
        key = str(k).strip()
        if key in {
            "BYBIT_API_KEY",
            "BYBIT_KEY",
            "BYBIT_API_SECRET",
            "BYBIT_SECRET_KEY",
            "BYBIT_SECRET",
        }:
            out[key] = str(v).strip()
    return out


def load_bybit_credentials_from_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return out
    source_cmd = (
        "set -a; "
        f"source {shlex.quote(str(path))}; "
        "set +a; "
        "printf 'BYBIT_API_KEY=%s\\n' \"${BYBIT_API_KEY:-${BYBIT_KEY:-}}\"; "
        "printf 'BYBIT_SECRET=%s\\n' \"${BYBIT_SECRET_KEY:-${BYBIT_API_SECRET:-${BYBIT_SECRET:-}}}\""
    )
    try:
        proc = subprocess.run(
            ["bash", "-lc", source_cmd],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return out
    if int(proc.returncode) != 0:
        return out
    for raw in str(proc.stdout or "").splitlines():
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        key = str(k).strip()
        if key in {"BYBIT_API_KEY", "BYBIT_SECRET"}:
            out[key] = str(v).strip()
    return out


def resolve_bybit_credentials(allow_daemon_env_fallback: bool) -> tuple[str, str, str]:
    api_key = load_string_env("BYBIT_API_KEY") or load_string_env("BYBIT_KEY")
    api_secret = (
        load_string_env("BYBIT_API_SECRET")
        or load_string_env("BYBIT_SECRET_KEY")
        or load_string_env("BYBIT_SECRET")
    )
    source = "process_env"
    if allow_daemon_env_fallback and (not api_key or not api_secret):
        env_map = load_bybit_credentials_from_daemon()
        if not api_key:
            api_key = str(env_map.get("BYBIT_API_KEY") or env_map.get("BYBIT_KEY") or "").strip()
        if not api_secret:
            api_secret = str(
                env_map.get("BYBIT_API_SECRET")
                or env_map.get("BYBIT_SECRET_KEY")
                or env_map.get("BYBIT_SECRET")
                or ""
            ).strip()
        if env_map:
            source = "daemon_env"
        if not api_key or not api_secret:
            env_file = Path(os.environ.get("BYBIT_CREDENTIALS_ENV_FILE", "~/.openclaw/.env")).expanduser()
            file_map = load_bybit_credentials_from_env_file(env_file)
            if not api_key:
                api_key = str(file_map.get("BYBIT_API_KEY") or "").strip()
            if not api_secret:
                api_secret = str(file_map.get("BYBIT_SECRET") or "").strip()
            if file_map:
                source = "env_file"
    return api_key, api_secret, source


class BybitSignedClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        base_url: str = "https://api.bybit.com",
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.api_secret = str(api_secret or "").strip()
        self.base_url = str(base_url).rstrip("/")
        self.timeout_ms = max(1, min(int(timeout_ms), 5000))
        self.rate_limit_per_minute = max(1, int(rate_limit_per_minute))
        self._bucket = TokenBucket(rate_per_minute=float(self.rate_limit_per_minute), capacity=float(self.rate_limit_per_minute))

    def _request(self, *, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> dict[str, Any]:
        timeout_seconds = max(0.1, min(float(self.timeout_ms) / 1000.0, 5.0))
        if not self._bucket.acquire(timeout_seconds):
            raise TimeoutError("bybit_rate_limit_timeout")
        http_method = str(method).strip().upper()
        endpoint = str(path).strip()
        req_params = {str(k): v for k, v in dict(params or {}).items() if v is not None}

        headers: dict[str, str] = {}
        query_params = dict(req_params)
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("bybit_credentials_missing")
            ts = str(now_epoch_ms())
            recv_window = "5000"
            query_text_for_sign = parse.urlencode(sorted((k, str(v)) for k, v in query_params.items()), doseq=True)
            body_text_for_sign = json.dumps(req_params, separators=(",", ":"), ensure_ascii=False)
            sign_payload = f"{ts}{self.api_key}{recv_window}{query_text_for_sign if http_method == 'GET' else body_text_for_sign}"
            sign = hmac.new(self.api_secret.encode("utf-8"), sign_payload.encode("utf-8"), hashlib.sha256).hexdigest()
            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": ts,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": sign,
            }
        query_text = parse.urlencode(sorted((k, str(v)) for k, v in query_params.items()), doseq=True)
        url = f"{self.base_url}{endpoint}"
        data: bytes | None = None
        if http_method == "GET":
            if query_text:
                url = f"{url}?{query_text}"
        else:
            data = json.dumps(req_params, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url=url, method=http_method, data=data, headers=headers)
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urlerror.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                body = str(exc)
            raise RuntimeError(f"http_{exc.code}:{body[:200]}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"url_error:{exc.reason}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_payload")
        ret_code = payload.get("retCode")
        if ret_code not in (0, "0"):
            raise RuntimeError(f"retCode_{ret_code}:{payload.get('retMsg')}")
        return payload

    def wallet_balance(self) -> dict[str, Any]:
        return self._request(
            method="GET",
            path="/v5/account/wallet-balance",
            params={"accountType": "UNIFIED"},
            signed=True,
        )

    def api_key_info(self) -> dict[str, Any]:
        return self._request(method="GET", path="/v5/user/query-api", params={}, signed=True)


class BybitFuturesClient(BybitSignedClient):
    def futures_account_info(self) -> dict[str, Any]:
        return self._request(method="GET", path="/v5/account/info", params={}, signed=True)


def _status_from_probe(ok: bool) -> str:
    return "ready" if ok else "blocked"


def _probe_spot_signed_read(spot_client: Any) -> tuple[bool, str | None]:
    try:
        payload = spot_client.wallet_balance()
    except Exception as exc:
        return False, f"spot_signed_read_probe_failed:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return False, "spot_signed_read_probe_invalid_payload"
    return True, None


def _probe_permissions(spot_client: Any) -> tuple[dict[str, Any], str | None]:
    try:
        payload = spot_client.api_key_info()
    except Exception as exc:
        return {}, f"api_permissions_probe_failed:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return {}, "api_permissions_probe_failed:invalid_payload"
    return payload, None


def _probe_futures_signed_read(futures_client: Any) -> tuple[bool, str | None]:
    try:
        payload = futures_client.futures_account_info()
    except Exception as exc:
        return False, f"futures_signed_read_probe_failed:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return False, "futures_signed_read_probe_invalid_payload"
    return True, None


def _extract_permissions(payload: dict[str, Any]) -> tuple[bool | None, bool | None]:
    result = payload.get("result")
    permissions = result.get("permissions") if isinstance(result, dict) else None
    if isinstance(permissions, dict):
        spot_rows = permissions.get("Spot")
        futures_rows = permissions.get("ContractTrade")
        if not isinstance(spot_rows, list) or not isinstance(futures_rows, list):
            return None, None
        spot_tokens = {str(row).strip() for row in spot_rows}
        futures_tokens = {str(row).strip() for row in futures_rows}
        spot = "SpotTrade" in spot_tokens
        futures = "Order" in futures_tokens
        return spot, futures
    return None, None


def _extract_ip_restrict(payload: dict[str, Any]) -> bool | None:
    result = payload.get("result")
    ips = result.get("ips") if isinstance(result, dict) else None
    if not isinstance(ips, list):
        return None
    normalized = {str(item).strip() for item in ips if str(item).strip()}
    if not normalized:
        return None
    return "*" not in normalized


def build_bybit_venue_payload(
    *,
    spot_client: Any,
    futures_client: Any,
    checked_at_utc: str | None = None,
    account_scope: str = "openclaw-system:process_env",
) -> dict[str, Any]:
    blockers: list[str] = []

    spot_read_ok, spot_read_err = _probe_spot_signed_read(spot_client)
    if spot_read_err:
        blockers.append(spot_read_err)
    spot_read_status = _status_from_probe(spot_read_ok)

    permission_payload, permission_err = _probe_permissions(spot_client)
    if permission_err:
        blockers.append(permission_err)
    permissions_available = permission_err is None and isinstance(permission_payload, dict)
    raw_permissions = permission_payload if permissions_available else None
    spot_trade_permission, futures_trade_permission = _extract_permissions(permission_payload) if permissions_available else (None, None)
    ip_restrict = _extract_ip_restrict(permission_payload) if permissions_available else None
    permissions_complete = permissions_available and isinstance(spot_trade_permission, bool) and isinstance(futures_trade_permission, bool)
    if permissions_available and not permissions_complete:
        blockers.append("api_permissions_incomplete")

    if permissions_complete:
        if bool(spot_trade_permission):
            spot_trade_status = "ready"
        else:
            spot_trade_status = "blocked"
            blockers.append("spotTrade=false")
    else:
        spot_trade_status = "unknown"

    if permissions_complete and bool(futures_trade_permission):
        futures_read_ok, futures_read_err = _probe_futures_signed_read(futures_client)
        if futures_read_err:
            blockers.append(futures_read_err)
        futures_read_status = _status_from_probe(futures_read_ok)
        futures_trade_status = futures_read_status
    elif permissions_complete:
        futures_read_status = "blocked"
        futures_trade_status = "blocked"
        blockers.append("contractTrade=false")
    else:
        futures_read_ok, futures_read_err = _probe_futures_signed_read(futures_client)
        if futures_read_err:
            blockers.append(futures_read_err)
        futures_read_status = _status_from_probe(futures_read_ok)
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
        "raw": {"apiPermissions": raw_permissions},
    }
