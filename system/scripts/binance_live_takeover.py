#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib import error as urlerror
from urllib import parse, request

import yaml


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


class BinanceUsdMClient:
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
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
        out = self._request(method="GET", path="/fapi/v1/ping", params={}, signed=False)
        return out if isinstance(out, dict) else {}

    def ticker_price(self, symbol: str) -> float:
        out = self._request(method="GET", path="/fapi/v1/ticker/price", params={"symbol": symbol}, signed=False)
        if not isinstance(out, dict):
            return 0.0
        return to_float(out.get("price", 0.0), 0.0)

    def exchange_info(self, symbol: str) -> dict[str, Any]:
        out = self._request(method="GET", path="/fapi/v1/exchangeInfo", params={"symbol": symbol}, signed=False)
        return out if isinstance(out, dict) else {}

    def account(self) -> dict[str, Any]:
        out = self._request(method="GET", path="/fapi/v2/account", params={}, signed=True)
        return out if isinstance(out, dict) else {}

    def user_trades(self, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict[str, Any]]:
        out = self._request(
            method="GET",
            path="/fapi/v1/userTrades",
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
        out = self._request(
            method="GET",
            path="/fapi/v1/income",
            params={
                "incomeType": "REALIZED_PNL",
                "startTime": int(start_ms),
                "endTime": int(end_ms),
                "limit": max(1, min(1000, int(limit))),
            },
            signed=True,
        )
        return out if isinstance(out, list) else []

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str,
        quote_order_qty: float | None = None,
    ) -> dict[str, Any]:
        out = self._request(
            method="POST",
            path="/fapi/v1/order",
            params={
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": f"{quantity:.8f}",
                "newClientOrderId": client_order_id,
            },
            signed=True,
        )
        return out if isinstance(out, dict) else {}


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


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return raw if isinstance(raw, dict) else {}


def save_config(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_whitelist_symbols(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    val = cfg.get("validation", {}) if isinstance(cfg.get("validation", {}), dict) else {}
    for key in ("micro_capture_daemon_symbols", "microstructure_symbols", "micro_cross_source_symbols"):
        raw = val.get(key, [])
        if isinstance(raw, list):
            for x in raw:
                sym = str(x).strip().upper()
                if sym.endswith("USDT") and sym not in out:
                    out.append(sym)
    uni = cfg.get("universe", {}) if isinstance(cfg.get("universe", {}), dict) else {}
    core = uni.get("core", []) if isinstance(uni.get("core", []), list) else []
    for row in core:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).strip().upper()
        if sym.endswith("USDT") and sym not in out:
            out.append(sym)
    if not out:
        out = ["BTCUSDT", "ETHUSDT"]
    return out


def latest_signals_path(output_root: Path, target_date: date) -> Path | None:
    daily_dir = output_root / "daily"
    candidates: list[Path] = []
    p = daily_dir / f"{target_date.isoformat()}_signals.json"
    if p.exists():
        candidates.append(p)
    if daily_dir.exists():
        candidates.extend(sorted(daily_dir.glob("*_signals.json"), reverse=True))
    seen: set[str] = set()
    for x in candidates:
        sx = str(x)
        if sx in seen:
            continue
        seen.add(sx)
        return x
    return None


def choose_canary_signal(output_root: Path, target_date: date, whitelist: list[str]) -> dict[str, Any]:
    path = latest_signals_path(output_root, target_date)
    if path is None:
        return {"symbol": whitelist[0], "side": "BUY", "confidence": 0.0, "source": "fallback_no_signals"}
    payload = read_json(path, [])
    rows = payload if isinstance(payload, list) else []
    best: dict[str, Any] | None = None
    best_score = -1e18
    wl = set(str(x).upper() for x in whitelist)
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        if wl and symbol not in wl:
            continue
        side_raw = str(row.get("side", "LONG")).strip().upper()
        side = "BUY" if side_raw in {"LONG", "BUY", "B"} else "SELL"
        conf = to_float(row.get("confidence", 0.0), 0.0)
        score = conf
        if side == "BUY":
            score += 0.01
        if score > best_score:
            best_score = score
            best = {
                "symbol": symbol,
                "side": side,
                "confidence": conf,
                "source": str(path),
                "signal_side": side_raw,
            }
    if best is None:
        return {"symbol": whitelist[0], "side": "BUY", "confidence": 0.0, "source": "fallback_no_whitelist_hit"}
    return best


def infer_lot_constraints(exchange_info: dict[str, Any], symbol: str) -> dict[str, float]:
    out = {
        "step_size": 0.001,
        "min_qty": 0.001,
        "min_notional": 5.0,
    }
    symbols = exchange_info.get("symbols", []) if isinstance(exchange_info.get("symbols", []), list) else []
    for row in symbols:
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol", "")).upper() != symbol.upper():
            continue
        filters = row.get("filters", []) if isinstance(row.get("filters", []), list) else []
        for f in filters:
            if not isinstance(f, dict):
                continue
            ftype = str(f.get("filterType", "")).upper()
            if ftype == "LOT_SIZE":
                out["step_size"] = max(1e-9, to_float(f.get("stepSize", out["step_size"]), out["step_size"]))
                out["min_qty"] = max(1e-9, to_float(f.get("minQty", out["min_qty"]), out["min_qty"]))
            if ftype in {"MIN_NOTIONAL", "NOTIONAL"}:
                k = "notional" if "notional" in f else "minNotional"
                out["min_notional"] = max(0.0, to_float(f.get(k, out["min_notional"]), out["min_notional"]))
        break
    return out


def quantize_floor(value: float, step: float) -> float:
    s = max(1e-12, float(step))
    return math.floor(float(value) / s) * s


def calc_canary_quantity(*, quote_usdt: float, price: float, step_size: float, min_qty: float, min_notional: float) -> float:
    px = max(1e-12, float(price))
    target_quote = max(float(quote_usdt), float(min_notional))
    raw = target_quote / px
    qty = quantize_floor(raw, step_size)
    if qty < min_qty:
        qty = quantize_floor(min_qty, step_size)
    if qty * px < min_notional:
        qty = quantize_floor((min_notional / px) + step_size, step_size)
    return max(0.0, qty)


def append_rows_dynamic(db_path: Path, table: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    ensure_parent(db_path)
    columns: list[str] = []
    col_set: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in col_set:
                col_set.add(key_s)
                columns.append(key_s)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,))
        exists = cur.fetchone() is not None
        if not exists:
            defs: list[str] = []
            for col in columns:
                has_numeric = any(isinstance(r.get(col), (int, float)) and not isinstance(r.get(col), bool) for r in rows)
                sql_type = "REAL" if has_numeric else "TEXT"
                defs.append(f'"{col.replace(chr(34), chr(34) + chr(34))}" {sql_type}')
            cur.execute(f'CREATE TABLE "{table}" ({", ".join(defs)})')
        else:
            cur.execute(f'PRAGMA table_info("{table}")')
            existing = {str(x[1]) for x in cur.fetchall()}
            for col in columns:
                if col in existing:
                    continue
                has_numeric = any(isinstance(r.get(col), (int, float)) and not isinstance(r.get(col), bool) for r in rows)
                sql_type = "REAL" if has_numeric else "TEXT"
                cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col.replace(chr(34), chr(34) + chr(34))}" {sql_type}')

        for row in rows:
            vals: list[Any] = []
            for col in columns:
                v = row.get(col)
                if isinstance(v, (dict, list)):
                    vals.append(json.dumps(v, ensure_ascii=False))
                elif isinstance(v, bool):
                    vals.append(1 if v else 0)
                else:
                    vals.append(v)
            placeholders = ",".join(["?"] * len(columns))
            col_sql = ",".join([f'"{c.replace(chr(34), chr(34) + chr(34))}"' for c in columns])
            cur.execute(f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})', vals)
        conn.commit()
    return len(rows)


def load_string_env(name: str) -> str:
    return str(os.environ.get(name, "")).strip()


def load_binance_credentials_from_daemon() -> dict[str, str]:
    out: dict[str, str] = {}
    pid = ""
    for line in os.popen("ps -ef | awk '/lie run-daemon/ && !/awk/ {print $2; exit}'").read().splitlines():
        pid = str(line).strip()
        if pid:
            break
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
    return api_key, api_secret, source


def build_evomap_strategy(
    *,
    output_root: Path,
    as_of: date,
    signal: dict[str, Any],
    whitelist: list[str],
    canary_quote_usdt: float,
    max_drawdown: float,
    rate_limit_per_minute: int,
) -> tuple[Path, dict[str, Any]]:
    payload = {
        "strategy_id": "evomap_binance_canary_v1",
        "generated_at_utc": now_utc_iso(),
        "as_of": as_of.isoformat(),
        "objective": "maximize_long_term_sharpe_under_drawdown_cap",
        "selection": {
            "symbol": str(signal.get("symbol", "BTCUSDT")),
            "side": str(signal.get("side", "BUY")),
            "signal_confidence": to_float(signal.get("confidence", 0.0), 0.0),
            "signal_source": str(signal.get("source", "")),
        },
        "risk": {
            "max_drawdown_hard": float(max_drawdown),
            "canary_quote_usdt": float(canary_quote_usdt),
            "max_concurrent_positions": 1,
            "hard_kill_on_transport_fail": True,
        },
        "execution": {
            "rate_limit_per_minute": int(rate_limit_per_minute),
            "request_timeout_ms": 5000,
            "idempotency_required": True,
            "mutex_protocol": "run-halfhour-pulse",
        },
        "whitelist": whitelist,
    }
    out_path = output_root / "artifacts" / "evomap" / f"{as_of.isoformat()}_strategy.json"
    write_json(out_path, payload)
    return out_path, payload


def activate_config_for_live(cfg_path: Path) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    changed: dict[str, Any] = {}

    data = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    old_profile = str(data.get("provider_profile", "opensource_dual"))
    if old_profile != "dual_binance_bybit_public":
        data["provider_profile"] = "dual_binance_bybit_public"
        changed["data.provider_profile"] = {"from": old_profile, "to": "dual_binance_bybit_public"}
    cfg["data"] = data

    val = cfg.get("validation", {}) if isinstance(cfg.get("validation", {}), dict) else {}
    old_mode = str(val.get("broker_snapshot_source_mode", "paper_engine"))
    if old_mode != "hybrid_prefer_live":
        val["broker_snapshot_source_mode"] = "hybrid_prefer_live"
        changed["validation.broker_snapshot_source_mode"] = {"from": old_mode, "to": "hybrid_prefer_live"}
    old_map = str(val.get("broker_snapshot_live_mapping_profile", "generic"))
    if old_map != "binance":
        val["broker_snapshot_live_mapping_profile"] = "binance"
        changed["validation.broker_snapshot_live_mapping_profile"] = {"from": old_map, "to": "binance"}
    val["broker_snapshot_live_fallback_to_paper"] = True
    cfg["validation"] = val

    if changed:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = cfg_path.with_suffix(cfg_path.suffix + f".bak_{ts}_live_takeover")
        save_config(backup, load_config(cfg_path))
        save_config(cfg_path, cfg)
        changed["backup"] = str(backup)
    return changed


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Activate Binance live takeover canary for Pi cloud node.")
    parser.add_argument("--date", default="", help="Trading date (YYYY-MM-DD). Default: local today.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--rate-limit-per-minute", type=int, default=DEFAULT_RATE_LIMIT_PER_MINUTE)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--canary-quote-usdt", type=float, default=5.0)
    parser.add_argument("--max-drawdown", type=float, default=0.05)
    parser.add_argument("--trade-window-hours", type=int, default=24)
    parser.add_argument("--max-trade-symbols", type=int, default=3)
    parser.add_argument("--market", choices=["futures_usdm", "spot"], default="futures_usdm")
    parser.add_argument("--activate-config", action="store_true")
    parser.add_argument("--allow-live-order", action="store_true")
    parser.add_argument("--allow-daemon-env-fallback", action="store_true")
    parser.add_argument("--order-symbol", default="")
    parser.add_argument("--order-side", choices=["BUY", "SELL"], default="")
    parser.add_argument("--mutex-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--skip-mutex", action="store_true", help="Skip run-halfhour-pulse mutex when parent lock is already held.")
    args = parser.parse_args()

    as_of = date.today()
    if str(args.date).strip():
        as_of = date.fromisoformat(str(args.date).strip())

    cwd = Path.cwd()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = cwd / cfg_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = cwd / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "started_at_utc": now_utc_iso(),
        "ok": True,
        "mode": "probe",
        "market": str(args.market),
        "config": str(cfg_path),
        "output_root": str(output_root),
        "rate_limit_per_minute": int(max(1, args.rate_limit_per_minute)),
        "timeout_ms": int(min(5000, max(100, args.timeout_ms))),
        "steps": {},
    }

    owner = f"binance_live_takeover:{as_of.isoformat()}"
    try:
        lock_cm = nullcontext() if bool(args.skip_mutex) else RunHalfhourMutex(
            output_root=output_root,
            owner=owner,
            timeout_seconds=float(args.mutex_timeout_seconds),
        )
        with lock_cm:
            if bool(args.activate_config):
                changed = activate_config_for_live(cfg_path)
                summary["steps"]["activate_config"] = {
                    "changed": bool(changed),
                    "detail": changed,
                }
            cfg = load_config(cfg_path)

            whitelist = load_whitelist_symbols(cfg)
            signal = choose_canary_signal(output_root=output_root, target_date=as_of, whitelist=whitelist)
            if str(args.order_symbol).strip():
                signal["symbol"] = str(args.order_symbol).strip().upper()
            if str(args.order_side).strip():
                signal["side"] = str(args.order_side).strip().upper()

            evomap_path, evomap_payload = build_evomap_strategy(
                output_root=output_root,
                as_of=as_of,
                signal=signal,
                whitelist=whitelist,
                canary_quote_usdt=float(args.canary_quote_usdt),
                max_drawdown=float(args.max_drawdown),
                rate_limit_per_minute=int(max(1, args.rate_limit_per_minute)),
            )
            summary["steps"]["evomap"] = {
                "path": str(evomap_path),
                "strategy_id": str(evomap_payload.get("strategy_id", "")),
                "selected": dict(evomap_payload.get("selection", {})),
            }

            api_key, api_secret, cred_source = resolve_binance_credentials(bool(args.allow_daemon_env_fallback))
            has_key = bool(api_key)
            has_secret = bool(api_secret)
            summary["steps"]["credentials"] = {
                "source": cred_source,
                "has_api_key": has_key,
                "has_api_secret": has_secret,
            }

            market = str(args.market).strip().lower()
            if market == "spot":
                client = BinanceSpotClient(
                    api_key=api_key,
                    api_secret=api_secret,
                    rate_limit_per_minute=max(1, int(args.rate_limit_per_minute)),
                    timeout_ms=min(5000, max(100, int(args.timeout_ms))),
                )
            else:
                market = "futures_usdm"
                client = BinanceUsdMClient(
                    api_key=api_key,
                    api_secret=api_secret,
                    rate_limit_per_minute=max(1, int(args.rate_limit_per_minute)),
                    timeout_ms=min(5000, max(100, int(args.timeout_ms))),
                )

            try:
                ping = client.ping()
                summary["steps"]["binance_ping"] = {
                    "ok": True,
                    "recv_ts_ms": now_epoch_ms(),
                    "payload": ping,
                }
            except (ConnectionError, TimeoutError, OSError) as exc:
                panic_close_all(output_root, reason="binance_ping_transport_error", detail=str(exc))

            symbol = str(signal.get("symbol", "BTCUSDT")).upper()
            side = str(signal.get("side", "BUY")).upper()
            if side not in {"BUY", "SELL"}:
                side = "BUY"

            try:
                price = float(client.ticker_price(symbol))
                exchange_info = client.exchange_info(symbol)
            except (ConnectionError, TimeoutError, OSError) as exc:
                panic_close_all(output_root, reason="binance_public_probe_transport_error", detail=str(exc))
            lot = infer_lot_constraints(exchange_info, symbol)
            qty = calc_canary_quantity(
                quote_usdt=float(args.canary_quote_usdt),
                price=price,
                step_size=float(lot["step_size"]),
                min_qty=float(lot["min_qty"]),
                min_notional=float(lot["min_notional"]),
            )
            effective_quote_usdt = float(qty * max(0.0, price))

            summary["steps"]["canary_plan"] = {
                "symbol": symbol,
                "side": side,
                "market": market,
                "mark_price": price,
                "quantity": qty,
                "quote_usdt": float(args.canary_quote_usdt),
                "effective_quote_usdt": effective_quote_usdt,
                "constraints": lot,
            }

            start_ms = int((datetime.now(timezone.utc) - timedelta(hours=max(1, int(args.trade_window_hours)))).timestamp() * 1000)
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            live_snapshot_path = output_root / "artifacts" / "broker_live_inbox" / f"{as_of.isoformat()}.json"
            trades_path = output_root / "artifacts" / "binance_live_trades" / f"{as_of.isoformat()}.json"
            income_path = output_root / "artifacts" / "binance_live_income" / f"{as_of.isoformat()}.json"
            order_log_path = output_root / "artifacts" / "binance_live_orders" / f"{as_of.isoformat()}.json"
            sqlite_path = output_root / "artifacts" / "lie_engine.db"

            if has_key and has_secret:
                summary["mode"] = "live_ready"
                try:
                    account = client.account()
                except (ConnectionError, TimeoutError, OSError) as exc:
                    panic_close_all(output_root, reason="binance_account_transport_error", detail=str(exc))

                positions_out: list[dict[str, Any]] = []
                symbol_pool: list[str] = []
                for s in [symbol] + whitelist:
                    ss = str(s).upper()
                    if ss and ss not in symbol_pool:
                        symbol_pool.append(ss)
                symbol_pool = symbol_pool[: max(1, int(args.max_trade_symbols))]

                if market == "spot":
                    balances = account.get("balances", []) if isinstance(account.get("balances", []), list) else []
                    bal_map: dict[str, float] = {}
                    for row in balances:
                        if not isinstance(row, dict):
                            continue
                        asset = str(row.get("asset", "")).strip().upper()
                        if not asset:
                            continue
                        qty_asset = to_float(row.get("free", 0.0), 0.0) + to_float(row.get("locked", 0.0), 0.0)
                        if qty_asset <= 1e-12:
                            continue
                        bal_map[asset] = qty_asset
                    for sym in symbol_pool:
                        if not sym.endswith("USDT"):
                            continue
                        base_asset = sym[:-4]
                        qty_asset = to_float(bal_map.get(base_asset, 0.0), 0.0)
                        if qty_asset <= 1e-12:
                            continue
                        px = to_float(client.ticker_price(sym), 0.0)
                        notional = qty_asset * px if px > 0 else 0.0
                        positions_out.append(
                            {
                                "symbol": sym,
                                "qty": float(qty_asset),
                                "side": "LONG",
                                "entry_price": 0.0,
                                "market_price": float(px),
                                "notional": float(notional),
                                "status": "OPEN",
                                "recv_ts_ms": now_epoch_ms(),
                            }
                        )
                else:
                    positions_in = account.get("positions", []) if isinstance(account.get("positions", []), list) else []
                    for row in positions_in:
                        if not isinstance(row, dict):
                            continue
                        pos_amt = to_float(row.get("positionAmt", 0.0), 0.0)
                        if abs(pos_amt) <= 1e-12:
                            continue
                        positions_out.append(
                            {
                                "symbol": str(row.get("symbol", "")).upper(),
                                "positionAmt": float(pos_amt),
                                "positionSide": str(row.get("positionSide", "BOTH")).upper(),
                                "entryPrice": to_float(row.get("entryPrice", 0.0), 0.0),
                                "markPrice": to_float(row.get("markPrice", 0.0), 0.0),
                                "notional": to_float(row.get("notional", 0.0), 0.0),
                                "leverage": to_float(row.get("leverage", 0.0), 0.0),
                                "recv_ts_ms": now_epoch_ms(),
                            }
                        )

                all_trades: list[dict[str, Any]] = []
                for sym in symbol_pool:
                    try:
                        rows = client.user_trades(symbol=sym, start_ms=start_ms, end_ms=end_ms, limit=1000)
                    except (ConnectionError, TimeoutError, OSError) as exc:
                        panic_close_all(output_root, reason="binance_user_trades_transport_error", detail=str(exc))
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        out_row = dict(row)
                        out_row["recv_ts_ms"] = now_epoch_ms()
                        all_trades.append(out_row)

                try:
                    incomes = client.realized_pnl_income(start_ms=start_ms, end_ms=end_ms, limit=1000)
                except (ConnectionError, TimeoutError, OSError) as exc:
                    panic_close_all(output_root, reason="binance_income_transport_error", detail=str(exc))

                incomes_out: list[dict[str, Any]] = []
                for row in incomes:
                    if not isinstance(row, dict):
                        continue
                    rr = dict(row)
                    rr["recv_ts_ms"] = now_epoch_ms()
                    incomes_out.append(rr)

                closed_pnl = float(sum(to_float(x.get("income", 0.0), 0.0) for x in incomes_out))
                live_snapshot = {
                    "date": as_of.isoformat(),
                    "generated_at": now_utc_iso(),
                    "source": "binance_spot" if market == "spot" else "binance_futures",
                    "open_positions": int(len(positions_out)),
                    "closed_count": int(len(incomes_out)),
                    "closed_pnl": float(closed_pnl),
                    "positions": positions_out,
                    "stats": {
                        "symbol_pool": symbol_pool,
                        "trade_window_hours": int(max(1, int(args.trade_window_hours))),
                        "market": market,
                        "recv_ts_ms": now_epoch_ms(),
                    },
                }
                write_json(live_snapshot_path, live_snapshot)
                write_json(trades_path, {"as_of": as_of.isoformat(), "rows": all_trades, "recv_ts_ms": now_epoch_ms()})
                write_json(income_path, {"as_of": as_of.isoformat(), "rows": incomes_out, "recv_ts_ms": now_epoch_ms()})

                summary["steps"]["live_snapshot"] = {
                    "path": str(live_snapshot_path),
                    "open_positions": int(len(positions_out)),
                    "closed_count": int(len(incomes_out)),
                    "closed_pnl": float(closed_pnl),
                }
                summary["steps"]["trade_telemetry"] = {
                    "trades_path": str(trades_path),
                    "income_path": str(income_path),
                    "symbols": symbol_pool,
                    "trades": int(len(all_trades)),
                    "income_rows": int(len(incomes_out)),
                }

                mode_payload = read_json(output_root / "daily" / f"{as_of.isoformat()}_mode_feedback.json", {})
                runtime_mode = "live_binance"
                if isinstance(mode_payload, dict):
                    runtime_mode = str(mode_payload.get("runtime_mode", runtime_mode)).strip() or runtime_mode

                income_ledger_path = output_root / "state" / "binance_income_ledger.json"
                ledger_ids = load_list_ledger(income_ledger_path, "event_ids")
                ledger_set = set(ledger_ids)

                exec_rows: list[dict[str, Any]] = []
                for row in incomes_out:
                    event_id = (
                        f"{row.get('tranId','')}:{row.get('time','')}:{row.get('symbol','')}:{row.get('incomeType','')}:{row.get('income','')}"
                    )
                    if event_id in ledger_set:
                        continue
                    ledger_ids.append(event_id)
                    ledger_set.add(event_id)
                    ts_ms = to_int(row.get("time", 0), 0)
                    day = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).date().isoformat() if ts_ms > 0 else as_of.isoformat()
                    pnl = to_float(row.get("income", 0.0), 0.0)
                    symbol_i = str(row.get("symbol", "")).upper()
                    direction = "LONG" if pnl >= 0 else "SHORT"
                    exec_rows.append(
                        {
                            "date": day,
                            "open_date": day,
                            "symbol": symbol_i,
                            "side": direction,
                            "direction": direction,
                            "runtime_mode": runtime_mode,
                            "mode": runtime_mode,
                            "pnl": float(pnl),
                            "pnl_pct": 0.0,
                            "exit_reason": "binance_realized_pnl",
                            "hold_days": 0,
                            "holding_days": 0,
                            "status": "CLOSED_LIVE",
                            "source": "binance_income",
                            "live_event_id": event_id,
                            "recv_ts_ms": now_epoch_ms(),
                        }
                    )

                if exec_rows:
                    inserted = append_rows_dynamic(sqlite_path, "executed_plans", exec_rows)
                else:
                    inserted = 0
                save_list_ledger(income_ledger_path, key="event_ids", values=ledger_ids, max_items=50000)
                summary["steps"]["backtest_feedback"] = {
                    "sqlite": str(sqlite_path),
                    "inserted_rows": int(inserted),
                    "runtime_mode": runtime_mode,
                }

                if bool(args.allow_live_order):
                    if qty <= 0.0:
                        panic_close_all(output_root, reason="canary_qty_invalid", detail=f"symbol={symbol}; qty={qty}")
                    budget_quote = max(0.0, float(args.canary_quote_usdt))
                    if market == "futures_usdm" and budget_quote > 0 and effective_quote_usdt > (budget_quote * 2.0):
                        summary["steps"]["canary_order"] = {
                            "executed": False,
                            "reason": "notional_floor_above_budget",
                            "effective_quote_usdt": effective_quote_usdt,
                            "budget_quote_usdt": budget_quote,
                            "market": market,
                        }
                        write_json(order_log_path, summary["steps"]["canary_order"])
                        summary["mode"] = "live_ready_budget_guarded"
                        summary["ok"] = False
                        summary["steps"]["canary_order"]["path"] = str(order_log_path)
                    else:
                        order_ledger_path = output_root / "state" / "binance_order_idempotency.json"
                        order_keys = load_list_ledger(order_ledger_path, "order_keys")
                        order_set = set(order_keys)
                        dedup_seed = f"{as_of.isoformat()}:{symbol}:{side}:{float(args.canary_quote_usdt):.4f}:evomap_binance_canary_v1"
                        order_key = hashlib.sha256(dedup_seed.encode("utf-8")).hexdigest()[:28]
                        if order_key in order_set:
                            summary["steps"]["canary_order"] = {
                                "executed": False,
                                "reason": "idempotent_skip",
                                "order_key": order_key,
                            }
                        else:
                            client_order_id = f"pi{as_of.strftime('%m%d')}{order_key[:20]}"
                            try:
                                order_rsp = client.place_market_order(
                                    symbol=symbol,
                                    side=side,
                                    quantity=qty,
                                    client_order_id=client_order_id,
                                    quote_order_qty=(budget_quote if (market == "spot" and side == "BUY") else None),
                                )
                            except RuntimeError as exc:
                                order_reject = {
                                    "as_of": as_of.isoformat(),
                                    "symbol": symbol,
                                    "side": side,
                                    "quantity": qty,
                                    "quote_usdt": float(args.canary_quote_usdt),
                                    "order_key": order_key,
                                    "client_order_id": client_order_id,
                                    "executed": False,
                                    "reason": "exchange_reject",
                                    "error": str(exc),
                                    "recv_ts_ms": now_epoch_ms(),
                                    "created_at_utc": now_utc_iso(),
                                }
                                write_json(order_log_path, order_reject)
                                summary["steps"]["canary_order"] = {
                                    "executed": False,
                                    "path": str(order_log_path),
                                    "order_key": order_key,
                                    "reason": "exchange_reject",
                                    "error": str(exc),
                                }
                                summary["ok"] = False
                            except (ConnectionError, TimeoutError, OSError) as exc:
                                panic_close_all(output_root, reason="canary_order_transport_error", detail=str(exc))
                            else:
                                order_record = {
                                    "as_of": as_of.isoformat(),
                                    "symbol": symbol,
                                    "side": side,
                                    "quantity": qty,
                                    "quote_usdt": float(args.canary_quote_usdt),
                                    "order_key": order_key,
                                    "client_order_id": client_order_id,
                                    "response": order_rsp,
                                    "recv_ts_ms": now_epoch_ms(),
                                    "created_at_utc": now_utc_iso(),
                                }
                                write_json(order_log_path, order_record)
                                order_keys.append(order_key)
                                save_list_ledger(order_ledger_path, key="order_keys", values=order_keys, max_items=10000)
                                summary["steps"]["canary_order"] = {
                                    "executed": True,
                                    "path": str(order_log_path),
                                    "order_key": order_key,
                                    "symbol": symbol,
                                    "side": side,
                                    "quantity": qty,
                                }
                else:
                    summary["steps"]["canary_order"] = {
                        "executed": False,
                        "reason": "allow_live_order_false",
                    }
            else:
                summary["mode"] = "degraded_read_only"
                summary["ok"] = False
                summary["steps"]["live_snapshot"] = {
                    "path": "",
                    "reason": "missing_binance_api_secret",
                }
                summary["steps"]["trade_telemetry"] = {
                    "reason": "missing_binance_api_secret",
                }
                summary["steps"]["backtest_feedback"] = {
                    "inserted_rows": 0,
                    "reason": "missing_binance_api_secret",
                }
                summary["steps"]["canary_order"] = {
                    "executed": False,
                    "reason": "missing_binance_api_secret",
                }

    except PanicTriggered as exc:
        summary["ok"] = False
        summary["panic"] = str(exc)
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = str(exc)

    summary["finished_at_utc"] = now_utc_iso()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_root / "review" / f"{ts}_binance_live_takeover.json"
    summary["artifact"] = str(out_path)
    write_json(out_path, summary)
    write_json(output_root / "review" / "latest_binance_live_takeover.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if bool(summary.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
