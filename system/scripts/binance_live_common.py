#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

DEFAULT_TIMEOUT_MS = 5000
DEFAULT_RATE_LIMIT_PER_MINUTE = 10


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_epoch_ms() -> int:
    return int(time.time() * 1000)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


class TokenBucket:
    def __init__(self, rate_per_minute: float, capacity: float | None = None) -> None:
        rpm = max(1.0, float(rate_per_minute))
        self.rate_per_second = rpm / 60.0
        self.capacity = float(capacity if capacity is not None else rpm)
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while True:
            sleep_for = 0.0
            with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self.last_refill)
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
                    self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                if self.rate_per_second > 0:
                    sleep_for = (1.0 - self.tokens) / self.rate_per_second
                else:
                    sleep_for = 0.25
            if time.monotonic() + sleep_for > deadline:
                return False
            time.sleep(min(0.25, max(0.01, sleep_for)))


class PanicTriggered(RuntimeError):
    pass


def panic_close_all(output_root: Path, *, reason: str, detail: str = "") -> None:
    marker = {
        "ts_utc": now_utc_iso(),
        "reason": str(reason),
        "detail": str(detail),
        "action": "panic_close_all",
    }
    write_json(output_root / "state" / "panic_close_all.json", marker)
    raise PanicTriggered(f"panic_close_all:{reason}")


class RunHalfhourMutex:
    def __init__(self, *, output_root: Path, owner: str, timeout_seconds: float = 5.0) -> None:
        self.path = output_root / "state" / "run-halfhour-pulse.lock"
        self.owner = owner
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.fd: int | None = None

    def __enter__(self) -> "RunHalfhourMutex":
        ensure_parent(self.path)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError(f"run-halfhour-pulse mutex timeout: {self.timeout_seconds:.1f}s")
                time.sleep(0.05)
        payload = {
            "owner": self.owner,
            "pid": os.getpid(),
            "acquired_at_utc": now_utc_iso(),
        }
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        os.fsync(fd)
        self.fd = fd
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
            self.fd = None
        return False


def load_string_env(name: str) -> str:
    return str(os.environ.get(name, "")).strip()


def detect_lie_daemon_pid() -> str:
    ps_cmd = "ps"
    for candidate in ("/bin/ps", "/usr/bin/ps"):
        if Path(candidate).exists():
            ps_cmd = candidate
            break
    try:
        proc = subprocess.run(
            [ps_cmd, "-eo", "pid=,args="],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return ""
    if int(proc.returncode) != 0:
        return ""
    for raw in str(proc.stdout or "").splitlines():
        row = str(raw).strip()
        if not row:
            continue
        parts = row.split(None, 1)
        if len(parts) != 2:
            continue
        pid_raw, args = parts
        if "lie run-daemon" not in args:
            continue
        if "awk" in args:
            continue
        try:
            pid_num = int(pid_raw)
        except Exception:
            continue
        if pid_num > 1:
            return str(pid_num)
    return ""


def load_binance_credentials_from_daemon() -> dict[str, str]:
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
            "BINANCE_API_KEY",
            "BINANCE_SECRET_KEY",
            "BINANCE_API_SECRET",
            "BINANCE_SECRET",
            "BINANCE_KEY",
        }:
            out[key] = v
    return out


def load_binance_credentials_from_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return out
    source_cmd = (
        "set -a; "
        f"source {shlex.quote(str(path))}; "
        "set +a; "
        "printf 'BINANCE_API_KEY=%s\\n' \"${BINANCE_API_KEY:-${BINANCE_KEY:-}}\"; "
        "printf 'BINANCE_SECRET=%s\\n' \"${BINANCE_SECRET_KEY:-${BINANCE_API_SECRET:-${BINANCE_SECRET:-}}}\""
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
        if key in {"BINANCE_API_KEY", "BINANCE_SECRET"}:
            out[key] = str(v).strip()
    return out


def resolve_binance_credentials(allow_daemon_env_fallback: bool) -> tuple[str, str, str]:
    api_key = (
        load_string_env("BINANCE_API_KEY")
        or load_string_env("BINANCE_KEY")
    )
    api_secret = (
        load_string_env("BINANCE_SECRET_KEY")
        or load_string_env("BINANCE_API_SECRET")
        or load_string_env("BINANCE_SECRET")
    )
    source = "process_env"
    if allow_daemon_env_fallback and (not api_key or not api_secret):
        env_map = load_binance_credentials_from_daemon()
        if not api_key:
            api_key = str(env_map.get("BINANCE_API_KEY") or env_map.get("BINANCE_KEY") or "").strip()
        if not api_secret:
            api_secret = str(
                env_map.get("BINANCE_SECRET_KEY")
                or env_map.get("BINANCE_API_SECRET")
                or env_map.get("BINANCE_SECRET")
                or ""
            ).strip()
        if env_map:
            source = "daemon_env"
        if not api_key or not api_secret:
            env_file = Path(os.environ.get("BINANCE_CREDENTIALS_ENV_FILE", "~/.openclaw/binance.env")).expanduser()
            file_map = load_binance_credentials_from_env_file(env_file)
            if not api_key:
                api_key = str(file_map.get("BINANCE_API_KEY") or "").strip()
            if not api_secret:
                api_secret = str(file_map.get("BINANCE_SECRET") or "").strip()
            if file_map:
                source = "env_file"
    return api_key, api_secret, source


class BinanceSpotClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        base_url: str = "https://api.binance.com",
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.api_secret = str(api_secret or "").strip()
        self.base_url = str(base_url).rstrip("/")
        self.bucket = TokenBucket(rate_per_minute=max(1, int(rate_limit_per_minute)))
        self.timeout_seconds = min(5.0, max(0.1, float(max(100, int(timeout_ms))) / 1000.0))

    def _request(
        self,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        signed: bool = False,
    ) -> Any:
        payload = dict(params or {})
        headers: dict[str, str] = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise RuntimeError("missing_signed_credentials")
            payload["timestamp"] = int(time.time() * 1000)
            payload.setdefault("recvWindow", 5000)

        query = parse.urlencode(payload, doseq=True)
        if signed:
            sig = hmac.new(self.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
            query = f"{query}&signature={sig}"
            headers["X-MBX-APIKEY"] = self.api_key
        elif self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        if not self.bucket.acquire(self.timeout_seconds):
            raise RuntimeError("token_bucket_acquire_timeout")

        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        req = request.Request(url=url, method=method.upper(), headers=headers)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read()
                if not body:
                    return {}
                return json.loads(body.decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = ""
            raise RuntimeError(f"http_{exc.code}:{detail[:240]}") from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise ConnectionError(str(exc)) from exc

    def ping(self) -> dict[str, Any]:
        out = self._request(method="GET", path="/api/v3/ping", params={}, signed=False)
        return out if isinstance(out, dict) else {}

    def ticker_price(self, symbol: str) -> float:
        out = self._request(method="GET", path="/api/v3/ticker/price", params={"symbol": symbol}, signed=False)
        if not isinstance(out, dict):
            return 0.0
        return to_float(out.get("price", 0.0), 0.0)

    def ticker_snapshot(self, symbol: str) -> dict[str, Any]:
        out = self._request(method="GET", path="/api/v3/ticker/price", params={"symbol": symbol}, signed=False)
        if not isinstance(out, dict):
            return {"symbol": str(symbol).upper(), "price": 0.0}
        return {
            "symbol": str(out.get("symbol") or symbol).upper(),
            "price": to_float(out.get("price", 0.0), 0.0),
        }

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        out = self._request(method="GET", path="/api/v3/exchangeInfo", params={"symbol": symbol}, signed=False)
        return out if isinstance(out, dict) else {}

    def account(self) -> dict[str, Any]:
        out = self._request(method="GET", path="/api/v3/account", params={}, signed=True)
        return out if isinstance(out, dict) else {}

    def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
        out = self._request(
            method="GET",
            path="/api/v3/myTrades",
            params={
                "symbol": symbol,
                "startTime": int(start_ms),
                "endTime": int(end_ms),
                "limit": max(1, min(1000, int(limit))),
            },
            signed=True,
        )
        return out if isinstance(out, list) else []

    def realized_pnl_income(self, *, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
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
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "newClientOrderId": client_order_id,
        }
        if side.upper() == "BUY":
            q = float(quote_order_qty if quote_order_qty is not None else 0.0)
            if q > 0.0:
                params["quoteOrderQty"] = f"{q:.8f}"
            else:
                params["quantity"] = f"{quantity:.8f}"
        else:
            params["quantity"] = f"{quantity:.8f}"
        out = self._request(method="POST", path="/api/v3/order", params=params, signed=True)
        return out if isinstance(out, dict) else {}


class BinanceUsdMMarketClient:
    def __init__(
        self,
        *,
        api_key: str = "",
        api_secret: str = "",
        rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        base_url: str = "https://fapi.binance.com",
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.api_secret = str(api_secret or "").strip()
        self.base_url = str(base_url).rstrip("/")
        self.bucket = TokenBucket(rate_per_minute=max(1, int(rate_limit_per_minute)))
        self.timeout_seconds = min(5.0, max(0.1, float(max(100, int(timeout_ms))) / 1000.0))

    def _request(
        self,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        payload = dict(params or {})
        query = parse.urlencode(payload, doseq=True)
        if not self.bucket.acquire(self.timeout_seconds):
            raise RuntimeError("token_bucket_acquire_timeout")
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        req = request.Request(url=url, method=method.upper(), headers={})
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read()
                if not body:
                    return {}
                return json.loads(body.decode("utf-8"))
        except urlerror.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = ""
            raise RuntimeError(f"http_{exc.code}:{detail[:240]}") from exc
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            raise ConnectionError(str(exc)) from exc

    def mark_index_funding_snapshot(self, symbol: str) -> dict[str, Any]:
        out = self._request(method="GET", path="/fapi/v1/premiumIndex", params={"symbol": symbol})
        if not isinstance(out, dict):
            return {
                "symbol": str(symbol).upper(),
                "mark_price": 0.0,
                "index_price": 0.0,
                "funding_rate_8h": 0.0,
                "next_funding_time_ms": 0,
                "snapshot_time_ms": 0,
            }
        return {
            "symbol": str(out.get("symbol") or symbol).upper(),
            "mark_price": to_float(out.get("markPrice", 0.0), 0.0),
            "index_price": to_float(out.get("indexPrice", 0.0), 0.0),
            "funding_rate_8h": to_float(out.get("lastFundingRate", 0.0), 0.0),
            "next_funding_time_ms": to_int(out.get("nextFundingTime", 0), 0),
            "snapshot_time_ms": to_int(out.get("time", 0), 0),
        }

    def open_interest_snapshot(self, symbol: str) -> dict[str, Any]:
        out = self._request(method="GET", path="/fapi/v1/openInterest", params={"symbol": symbol})
        if not isinstance(out, dict):
            return {
                "symbol": str(symbol).upper(),
                "open_interest_contracts": 0.0,
                "snapshot_time_ms": 0,
            }
        return {
            "symbol": str(out.get("symbol") or symbol).upper(),
            "open_interest_contracts": to_float(out.get("openInterest", 0.0), 0.0),
            "snapshot_time_ms": to_int(out.get("time", 0), 0),
        }


def load_list_ledger(path: Path, key: str) -> list[str]:
    payload = read_json(path, {})
    if not isinstance(payload, dict):
        return []
    rows = payload.get(key, [])
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in rows:
        sx = str(x)
        if not sx or sx in seen:
            continue
        seen.add(sx)
        out.append(sx)
    return out


def save_list_ledger(path: Path, *, key: str, values: list[str], max_items: int) -> None:
    trimmed = values[-max(1, int(max_items)) :]
    payload = {
        "updated_at_utc": now_utc_iso(),
        key: trimmed,
    }
    write_json(path, payload)
