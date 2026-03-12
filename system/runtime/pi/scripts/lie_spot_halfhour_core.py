#!/usr/bin/env python3
"""Half-hour core: LiE -> Spot (dry-run) driver.

Contract (append to STATE.md):
- One JSON line prefixed with: LIE_DRYRUN_EVENT=
- Then 3 human-readable lines.

This script is designed to be called by OpenClaw cron every 30 minutes.
It is safe-by-default (no real orders; uses Binance /api/v3/order/test).

Guardrails (paper-sim):
- max_daily_drawdown_usdt=2
- max_daily_drawdown_pct=0.05
- consecutive_loss_stop=3

Note: Binance test endpoint does not change real balances, so we maintain a
minimal "paper" portfolio state for drawdown / streak tracking.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time as pytime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests
from cortex_gate import SpinalReflexReject, cortex_gated
from cortex_evaluator import StateVectorCortex
from lie_root_resolver import resolve_lie_system_root
from net_resilience import get_proxy_bypass_stats
from net_resilience import request_no_proxy as _net_request_no_proxy
from net_resilience import reset_proxy_bypass_stats
from net_resilience import request_with_proxy_bypass as _net_request_with_proxy_bypass


LIE_ROOT = resolve_lie_system_root()
TRADER_ROOT = Path(os.getenv("TRADER_WORKSPACE_ROOT", str(Path(__file__).resolve().parents[1])))
OC_ROOT = Path(os.getenv("OPENCLAW_STATE_WORKSPACE", str(TRADER_ROOT)))
STATE_MD = OC_ROOT / "STATE.md"
# Use a global lock to serialize writes to shared paper state across workspaces.
LOCK_PATH = Path(
    os.getenv(
        "LIE_SPOT_PAPER_LOCK_PATH",
        str(LIE_ROOT / "output" / "state" / "spot_paper_state.lock"),
    )
)
PULSE_LOCK_PATH = Path(
    os.getenv(
        "LIE_HALFHOUR_PULSE_LOCK_PATH",
        str(LIE_ROOT / "output" / "state" / "run_halfhour_pulse.lock"),
    )
)
READINESS_REFRESH_LOCK_PATH = Path(
    os.getenv(
        "LIE_PAPER_MODE_READINESS_REFRESH_LOCK_PATH",
        str(LIE_ROOT / "output" / "state" / "paper_mode_readiness_refresh.lock"),
    )
)
READINESS_REFRESH_STATE_PATH = Path(
    os.getenv(
        "LIE_PAPER_MODE_READINESS_REFRESH_STATE_PATH",
        str(LIE_ROOT / "output" / "state" / "paper_mode_readiness_refresh_state.json"),
    )
)
# Link paper state to the actual global instance
PAPER_STATE_PATH = Path(
    os.getenv(
        "LIE_SPOT_PAPER_STATE_PATH",
        str(LIE_ROOT / "output" / "state" / "spot_paper_state.json"),
    )
)
PAPER_CONSECUTIVE_LOSS_ACK_PATH = Path(
    os.getenv(
        "LIE_PAPER_CONSECUTIVE_LOSS_ACK_PATH",
        str(LIE_ROOT / "output" / "state" / "paper_consecutive_loss_ack.json"),
    )
)
PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH = Path(
    os.getenv(
        "LIE_PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH",
        str(LIE_ROOT / "output" / "state" / "paper_consecutive_loss_ack_checksum.json"),
    )
)
PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR = Path(
    os.getenv(
        "LIE_PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR",
        str(LIE_ROOT / "output" / "review" / "paper_consecutive_loss_ack_archive"),
    )
)
PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH = Path(
    os.getenv(
        "LIE_PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH",
        str(PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR / "manifest.jsonl"),
    )
)
PAPER_POSITIONS_OPEN_PATH = Path(
    os.getenv(
        "LIE_PAPER_POSITIONS_OPEN_PATH",
        str(LIE_ROOT / "output" / "artifacts" / "paper_positions_open.json"),
    )
)
BROKER_SNAPSHOT_DIR = Path(
    os.getenv(
        "LIE_PAPER_BROKER_SNAPSHOT_DIR",
        str(LIE_ROOT / "output" / "artifacts" / "broker_snapshot"),
    )
)
PAPER_ARTIFACTS_LOCK_PATH = Path(
    os.getenv(
        "LIE_PAPER_ARTIFACTS_LOCK_PATH",
        str(LIE_ROOT / "output" / "state" / "paper_positions_open.lock"),
    )
)
PAPER_EXECUTION_LEDGER_PATH = Path(
    os.getenv(
        "LIE_PAPER_EXECUTION_LEDGER_PATH",
        str(LIE_ROOT / "output" / "logs" / "paper_execution_ledger.jsonl"),
    )
)
SOURCE_REQUEST_LOG_PATH = Path(
    os.getenv(
        "LIE_SOURCE_REQUEST_LOG_PATH",
        str(LIE_ROOT / "output" / "logs" / "core_source_request_log.jsonl"),
    )
)

BINANCE_SYMBOL = os.getenv("LIE_SPOT_SYMBOL", "ETHUSDT")

MAX_DAILY_DRAWDOWN_USDT = float(os.getenv("LIE_MAX_DAILY_DRAWDOWN_USDT", "2"))
MAX_DAILY_DRAWDOWN_PCT = float(os.getenv("LIE_MAX_DAILY_DRAWDOWN_PCT", "0.05"))
CONSECUTIVE_LOSS_STOP = int(os.getenv("LIE_CONSECUTIVE_LOSS_STOP", "3"))
PAPER_INIT_USDT = float(os.getenv("LIE_PAPER_INIT_USDT", "100"))
CORTEX_MAX_PAPER_NOTIONAL = float(os.getenv("CORTEX_MAX_PAPER_NOTIONAL", "5"))
CORTEX_MAX_PAPER_QTY = float(os.getenv("CORTEX_MAX_PAPER_QTY", "0.02"))

LOCK_GATE_LEVEL_ORDER: Dict[str, int] = {
    "ok": 0,
    "warn": 1,
    "degraded": 2,
    "critical": 3,
}


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


SOURCE_REQUEST_LOG_MAX_BYTES = max(
    0,
    env_int("LIE_SOURCE_REQUEST_LOG_MAX_BYTES", 50 * 1024 * 1024),
)
SOURCE_REQUEST_LOG_KEEP_FILES = max(1, env_int("LIE_SOURCE_REQUEST_LOG_KEEP_FILES", 12))
PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES = max(
    1,
    env_int("LIE_PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES", 12),
)
SOURCE_REQUEST_LOG_ARCHIVE_DIR = Path(
    os.getenv(
        "LIE_SOURCE_REQUEST_LOG_ARCHIVE_DIR",
        str(SOURCE_REQUEST_LOG_PATH.parent / "core_source_request_log_archive"),
    )
)
SOURCE_REQUEST_LOG_MANIFEST_PATH = Path(
    os.getenv(
        "LIE_SOURCE_REQUEST_LOG_MANIFEST_PATH",
        str(SOURCE_REQUEST_LOG_ARCHIVE_DIR / "manifest.jsonl"),
    )
)
SOURCE_REQUEST_LOCK_METRICS_PATH = Path(
    os.getenv(
        "LIE_SOURCE_REQUEST_LOCK_METRICS_PATH",
        str(SOURCE_REQUEST_LOG_PATH.parent / "core_source_request_lock_metrics.jsonl"),
    )
)
SOURCE_REQUEST_LOCK_METRICS_EVERY_N = max(
    1,
    env_int("LIE_SOURCE_REQUEST_LOCK_METRICS_EVERY_N", 1),
)
SOURCE_REQUEST_LOG_FILE_LOCK_PATH = Path(
    os.getenv(
        "LIE_SOURCE_REQUEST_LOG_LOCK_PATH",
        str(SOURCE_REQUEST_LOG_PATH.parent / "core_source_request_log.write.lock"),
    )
)
SOURCE_REQUEST_LOG_ROTATE_PREFIX = "core_source_request_log_"
SOURCE_REQUEST_LOG_LOCK = threading.Lock()
SOURCE_REQUEST_LOCK_METRICS_COUNTER = 0


def event_source_label() -> str:
    raw = str(os.getenv("LIE_EVENT_SOURCE", "prod")).strip().lower()
    return raw or "prod"


def build_exec_command(
    *,
    action: str,
    probe_order_test: bool,
    live_order: bool,
) -> List[str]:
    cmd = [
        "python3",
        str(TRADER_ROOT / "scripts" / "binance_spot_exec.py"),
        "--symbol",
        BINANCE_SYMBOL,
        "--action",
        action,
    ]
    if probe_order_test:
        cmd.append("--probe-order-test")
    if live_order:
        cmd.append("--live")
    return cmd


def should_attempt_probe_on_guardrail(
    *,
    guardrail_hit: bool,
    probe_order_test: bool,
    force_probe_on_guardrail: bool,
) -> bool:
    if not probe_order_test:
        return False
    if not guardrail_hit:
        return False
    return bool(force_probe_on_guardrail)


def now_local_iso() -> str:
    # Asia/Shanghai is +08:00, but do not hardcode tz name; use offset.
    return dt.datetime.now().astimezone().isoformat()


def run_json(cmd: List[str], cwd: Path, env_extra: Optional[Dict[str, str]] = None, timeout_s: int = 240) -> Dict[str, Any]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\\nstdout={out[:800]}\\nstderr={err[:800]}")
    if not out:
        raise RuntimeError(f"empty stdout: {' '.join(cmd)}")
    try:
        return json.loads(out)
    except Exception as e:
        raise RuntimeError(f"stdout not json: {' '.join(cmd)}\\nstdout={out[:800]}\\nstderr={err[:800]}") from e


def _is_invalid_cli_choice(exc: Exception, cmd_name: str) -> bool:
    text = str(exc or "")
    return ("invalid choice" in text) and (f"'{cmd_name}'" in text)


def _slot_arg_from_slot_id(slot_id: str) -> str:
    raw = str(slot_id or "").strip()
    if raw.startswith("intraday:"):
        return raw.split(":", 1)[1]
    return raw


def _http_no_proxy(method: str, url: str, **kwargs: Any) -> requests.Response:
    return _net_request_no_proxy(method, url, **kwargs)


def _infer_http_source(url: str) -> str:
    host = str(urlsplit(str(url or "")).netloc or "").lower()
    path = str(urlsplit(str(url or "")).path or "").lower()
    if "binance" in host:
        return "binance"
    if "coinbase.com" in host:
        return "coinbase"
    if "kraken.com" in host:
        return "kraken"
    if "okx.com" in host:
        return "okx"
    if "coingecko.com" in host:
        return "coingecko"
    if "cryptocompare.com" in host:
        return "cryptocompare"
    if "api.alternative.me" in host:
        return "alternative_fng"
    if "api.blockchain.info" in host:
        if path.startswith("/charts/n-transactions"):
            return "blockchain_charts_n_tx"
        if path.startswith("/stats"):
            return "blockchain_stats_n_tx"
        return "blockchain_info"
    if host in {"127.0.0.1:9999", "localhost:9999"}:
        return "embedding_local"
    return "unknown"


def _extract_symbol_hint(url: str, kwargs: Dict[str, Any]) -> Optional[str]:
    params = kwargs.get("params")
    if isinstance(params, dict):
        for key in ("symbol", "pair", "instId", "fsym", "ids"):
            raw = params.get(key)
            if raw is None:
                continue
            txt = str(raw).strip()
            if txt:
                return txt.upper()
    path = str(urlsplit(str(url or "")).path or "")
    if "/products/" in path:
        try:
            tail = path.split("/products/", 1)[1]
            product = tail.split("/", 1)[0].strip()
            if product:
                return product.upper()
        except Exception:
            return None
    return None


def _source_request_archive_path(ts_utc: dt.datetime, index: int = 0) -> Path:
    stamp = ts_utc.strftime("%Y%m%dT%H%M%S_%fZ")
    suffix = "" if index <= 0 else f"_{index}"
    return SOURCE_REQUEST_LOG_ARCHIVE_DIR / f"{SOURCE_REQUEST_LOG_ROTATE_PREFIX}{stamp}{suffix}.jsonl"


def _consecutive_loss_ack_archive_path(ts_utc: dt.datetime, index: int = 0) -> Path:
    stamp = ts_utc.strftime("%Y%m%dT%H%M%S_%fZ")
    suffix = "" if index <= 0 else f"_{index}"
    return PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR / f"paper_consecutive_loss_ack_{stamp}{suffix}.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _acquire_source_request_file_lock() -> Tuple[Optional[Any], float, str]:
    started = pytime.perf_counter()
    try:
        import fcntl  # type: ignore
    except Exception:
        return None, (pytime.perf_counter() - started) * 1000.0, "no_fcntl"
    lockf = None
    try:
        SOURCE_REQUEST_LOG_FILE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        lockf = SOURCE_REQUEST_LOG_FILE_LOCK_PATH.open("w")
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        return lockf, (pytime.perf_counter() - started) * 1000.0, "fcntl"
    except Exception:
        if lockf is not None:
            try:
                lockf.close()
            except Exception:
                pass
        return None, (pytime.perf_counter() - started) * 1000.0, "lock_error"


def _release_source_request_file_lock(lockf: Optional[Any]) -> None:
    if lockf is None:
        return
    try:
        import fcntl  # type: ignore

        fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lockf.close()
    except Exception:
        pass


def _maybe_append_source_request_lock_metric(metric: Dict[str, Any]) -> None:
    global SOURCE_REQUEST_LOCK_METRICS_COUNTER
    SOURCE_REQUEST_LOCK_METRICS_COUNTER += 1
    if SOURCE_REQUEST_LOCK_METRICS_EVERY_N > 1:
        if (SOURCE_REQUEST_LOCK_METRICS_COUNTER % SOURCE_REQUEST_LOCK_METRICS_EVERY_N) != 0:
            return
    try:
        _append_jsonl(SOURCE_REQUEST_LOCK_METRICS_PATH, metric)
    except Exception:
        return


def _prune_source_request_log_archives() -> None:
    archives = sorted(
        SOURCE_REQUEST_LOG_ARCHIVE_DIR.glob(f"{SOURCE_REQUEST_LOG_ROTATE_PREFIX}*.jsonl"),
        key=lambda p: p.name,
    )
    if len(archives) <= SOURCE_REQUEST_LOG_KEEP_FILES:
        return
    stale = archives[: len(archives) - SOURCE_REQUEST_LOG_KEEP_FILES]
    for p in stale:
        checksum_path = p.with_suffix(p.suffix + ".sha256")
        size_bytes = p.stat().st_size if p.exists() else 0
        if p.exists():
            p.unlink()
        if checksum_path.exists():
            checksum_path.unlink()
        _append_jsonl(
            SOURCE_REQUEST_LOG_MANIFEST_PATH,
            {
                "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": "purged",
                "path": str(p),
                "size_bytes": int(size_bytes),
            },
        )


def _rotate_source_request_log_if_needed() -> None:
    if SOURCE_REQUEST_LOG_MAX_BYTES <= 0:
        return
    if not SOURCE_REQUEST_LOG_PATH.exists():
        return
    size_bytes = SOURCE_REQUEST_LOG_PATH.stat().st_size
    if size_bytes < SOURCE_REQUEST_LOG_MAX_BYTES:
        return

    SOURCE_REQUEST_LOG_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    rotated_at = dt.datetime.now(dt.timezone.utc)
    archive_path = _source_request_archive_path(rotated_at)
    idx = 1
    while archive_path.exists():
        archive_path = _source_request_archive_path(rotated_at, index=idx)
        idx += 1
    SOURCE_REQUEST_LOG_PATH.replace(archive_path)
    checksum = _sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    _append_jsonl(
        SOURCE_REQUEST_LOG_MANIFEST_PATH,
        {
            "ts_utc": rotated_at.isoformat(),
            "event": "rotated",
            "from_path": str(SOURCE_REQUEST_LOG_PATH),
            "to_path": str(archive_path),
            "size_bytes": int(size_bytes),
            "sha256": checksum,
            "checksum_path": str(checksum_path),
        },
    )
    _prune_source_request_log_archives()


def _prune_consecutive_loss_ack_archives() -> None:
    archives = sorted(
        PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR.glob("paper_consecutive_loss_ack_*.json"),
        key=lambda p: p.name,
    )
    if len(archives) <= PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES:
        return
    stale = archives[: len(archives) - PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_KEEP_FILES]
    for p in stale:
        checksum_path = p.with_suffix(p.suffix + ".sha256")
        size_bytes = p.stat().st_size if p.exists() else 0
        if p.exists():
            p.unlink()
        if checksum_path.exists():
            checksum_path.unlink()
        _append_jsonl(
            PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH,
            {
                "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": "purged",
                "path": str(p),
                "size_bytes": int(size_bytes),
            },
        )


def _archive_and_clear_consecutive_loss_ack_payload(
    payload: Dict[str, Any], *, now_dt: dt.datetime, cycle_ts: str
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "archive_ok": False,
        "archive_path": None,
        "archive_checksum_path": None,
        "archive_sha256": None,
        "archive_reason": "not_attempted",
        "live_ack_cleared": False,
        "live_checksum_cleared": False,
    }
    try:
        PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        archive_payload = dict(payload)
        archive_payload["archived_at"] = now_dt.astimezone(dt.timezone.utc).isoformat()
        archive_payload["archive_reason"] = "consumed"
        archive_payload["archive_cycle_ts"] = str(cycle_ts or "").strip() or None
        archive_payload["live_ack_path"] = str(PAPER_CONSECUTIVE_LOSS_ACK_PATH)
        archive_payload["live_checksum_path"] = str(PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH)

        archive_path = _consecutive_loss_ack_archive_path(now_dt)
        idx = 1
        while archive_path.exists():
            archive_path = _consecutive_loss_ack_archive_path(now_dt, index=idx)
            idx += 1
        archive_path.write_text(
            json.dumps(archive_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum = _sha256_file(archive_path)
        checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
        checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
        _append_jsonl(
            PAPER_CONSECUTIVE_LOSS_ACK_ARCHIVE_MANIFEST_PATH,
            {
                "ts_utc": now_dt.astimezone(dt.timezone.utc).isoformat(),
                "event": "consumed_archive",
                "path": str(archive_path),
                "size_bytes": int(archive_path.stat().st_size),
                "sha256": checksum,
                "checksum_path": str(checksum_path),
                "cycle_ts": str(cycle_ts or "").strip() or None,
            },
        )
        out["archive_ok"] = True
        out["archive_path"] = str(archive_path)
        out["archive_checksum_path"] = str(checksum_path)
        out["archive_sha256"] = checksum
        out["archive_reason"] = "archived"

        if PAPER_CONSECUTIVE_LOSS_ACK_PATH.exists():
            PAPER_CONSECUTIVE_LOSS_ACK_PATH.unlink()
            out["live_ack_cleared"] = True
        else:
            out["live_ack_cleared"] = True
        if PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH.exists():
            PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH.unlink()
            out["live_checksum_cleared"] = True
        else:
            out["live_checksum_cleared"] = True

        _prune_consecutive_loss_ack_archives()
        return out
    except Exception as exc:
        out["archive_reason"] = f"archive_failed:{exc.__class__.__name__}"
        return out


def _append_source_request_log(entry: Dict[str, Any]) -> bool:
    try:
        with SOURCE_REQUEST_LOG_LOCK:
            lockf, lock_wait_ms, lock_mode = _acquire_source_request_file_lock()
            metric = {
                "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event_source": str(entry.get("event_source") or event_source_label()),
                "source": str(entry.get("source") or "unknown"),
                "symbol_hint": str(entry.get("symbol_hint") or "UNKNOWN"),
                "lock_mode": lock_mode,
                "lock_acquired": bool(lockf is not None),
                "lock_wait_ms": round(float(lock_wait_ms), 3),
                "pid": int(os.getpid()),
                "thread_id": int(threading.get_ident()),
            }
            try:
                _rotate_source_request_log_if_needed()
                _append_jsonl(SOURCE_REQUEST_LOG_PATH, entry)
                metric["append_ok"] = True
            finally:
                _release_source_request_file_lock(lockf)
            _maybe_append_source_request_lock_metric(metric)
        return True
    except Exception:
        try:
            _maybe_append_source_request_lock_metric(
                {
                    "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "event_source": str(entry.get("event_source") or event_source_label()),
                    "source": str(entry.get("source") or "unknown"),
                    "symbol_hint": str(entry.get("symbol_hint") or "UNKNOWN"),
                    "lock_mode": "exception",
                    "lock_acquired": False,
                    "lock_wait_ms": None,
                    "append_ok": False,
                    "pid": int(os.getpid()),
                    "thread_id": int(threading.get_ident()),
                }
            )
        except Exception:
            pass
        return False


def _http(method: str, url: str, **kwargs: Any) -> requests.Response:
    started = pytime.perf_counter()
    ts_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    endpoint = str(urlsplit(str(url or "")).path or "/")
    source = _infer_http_source(url)
    symbol_hint = _extract_symbol_hint(url, kwargs)
    try:
        resp = _net_request_with_proxy_bypass(
            method,
            url,
            no_proxy_request_func=_http_no_proxy,
            **kwargs,
        )
        latency_ms = (pytime.perf_counter() - started) * 1000.0
        _append_source_request_log(
            {
                "ts_utc": ts_utc,
                "event_source": event_source_label(),
                "source": source,
                "endpoint": endpoint,
                "method": str(method or "").upper(),
                "status": int(resp.status_code),
                "ok": int(resp.status_code) < 400,
                "latency_ms": round(float(latency_ms), 3),
                "symbol_hint": symbol_hint,
            }
        )
        return resp
    except Exception as e:
        latency_ms = (pytime.perf_counter() - started) * 1000.0
        _append_source_request_log(
            {
                "ts_utc": ts_utc,
                "event_source": event_source_label(),
                "source": source,
                "endpoint": endpoint,
                "method": str(method or "").upper(),
                "status": None,
                "ok": False,
                "latency_ms": round(float(latency_ms), 3),
                "symbol_hint": symbol_hint,
                "error_type": type(e).__name__,
                "error": str(e)[:240],
            }
        )
        raise


def coingecko_price(symbol: str) -> float:
    """Fallback spot price via CoinGecko (USD).

    We only need a coarse mark price for paper guardrails and sizing.
    """
    base = symbol.upper()
    for q in ("USDT", "USD"):
        if base.endswith(q):
            base = base[: -len(q)]
            break

    # Minimal mapping (extend as needed)
    id_map = {
        "ETH": "ethereum",
        "BTC": "bitcoin",
        "SOL": "solana",
        "XRP": "ripple",
        "BNB": "binancecoin",
        "DOGE": "dogecoin",
        "ADA": "cardano",
        "TRX": "tron",
        "LTC": "litecoin",
        "DOT": "polkadot",
        "AVAX": "avalanche-2",
        "LINK": "chainlink",
    }
    cg_id = id_map.get(base)
    if not cg_id:
        raise RuntimeError(f"coingecko unsupported symbol: {symbol}")

    r = _http(
        "GET",
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": cg_id, "vs_currencies": "usd"},
        timeout=10,
        headers={"accept": "application/json"},
    )
    r.raise_for_status()
    j = r.json()
    return float(j[cg_id]["usd"])


def _split_symbol(symbol: str) -> Tuple[str, str]:
    s = str(symbol or "").strip().upper()
    for q in ("USDT", "USD"):
        if s.endswith(q):
            base = s[: -len(q)]
            if base:
                return base, q
    raise RuntimeError(f"unsupported quote symbol: {symbol}")


def _coinbase_product_candidates(symbol: str) -> List[str]:
    base, quote = _split_symbol(symbol)
    out: List[str] = []
    if quote == "USDT":
        out.extend([f"{base}-USDT", f"{base}-USD", f"{base}-USDC"])
    else:
        out.extend([f"{base}-USD", f"{base}-USDC"])
    # preserve order while de-duplicating
    seen: set[str] = set()
    uniq: List[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq


def coinbase_price(symbol: str) -> float:
    """Fallback spot price via Coinbase Exchange public ticker."""
    errors: List[str] = []
    for product in _coinbase_product_candidates(symbol):
        try:
            r = _http(
                "GET",
                f"https://api.exchange.coinbase.com/products/{product}/ticker",
                timeout=10,
                headers={"accept": "application/json"},
            )
            r.raise_for_status()
            j = r.json()
            price = (j or {}).get("price") if isinstance(j, dict) else None
            if price is None:
                raise RuntimeError(f"coinbase missing price for {product}")
            return float(price)
        except Exception as e:
            errors.append(f"{product}:{type(e).__name__}:{str(e)[:120]}")
    raise RuntimeError(f"coinbase no ticker for {symbol}: {' | '.join(errors[-4:])}")


def _kraken_pair_candidates(symbol: str) -> List[str]:
    base, quote = _split_symbol(symbol)
    base_aliases = [base]
    if base == "BTC":
        base_aliases.append("XBT")
    quote_aliases = [quote]
    if quote == "USDT":
        quote_aliases.append("USD")

    pairs: List[str] = []
    for b in base_aliases:
        for q in quote_aliases:
            pairs.append(f"{b}{q}")

    seen: set[str] = set()
    uniq: List[str] = []
    for item in pairs:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq


def kraken_price(symbol: str) -> float:
    """Fallback spot price via Kraken public ticker."""
    errors: List[str] = []
    for pair in _kraken_pair_candidates(symbol):
        try:
            r = _http(
                "GET",
                "https://api.kraken.com/0/public/Ticker",
                params={"pair": pair},
                timeout=10,
                headers={"accept": "application/json"},
            )
            r.raise_for_status()
            j = r.json()
            err = (j or {}).get("error") if isinstance(j, dict) else None
            if isinstance(err, list) and err:
                raise RuntimeError(f"kraken_error:{';'.join(str(x) for x in err[:2])}")
            result = (j or {}).get("result") if isinstance(j, dict) else None
            if not isinstance(result, dict) or not result:
                raise RuntimeError(f"kraken empty result for {pair}")
            ticker = next(iter(result.values()))
            if not isinstance(ticker, dict):
                raise RuntimeError(f"kraken invalid ticker for {pair}")
            # c[0] = last trade close
            close = ticker.get("c")
            if isinstance(close, list) and close:
                return float(close[0])
            ask = ticker.get("a")
            if isinstance(ask, list) and ask:
                return float(ask[0])
            bid = ticker.get("b")
            if isinstance(bid, list) and bid:
                return float(bid[0])
            raise RuntimeError(f"kraken missing close for {pair}")
        except Exception as e:
            errors.append(f"{pair}:{type(e).__name__}:{str(e)[:120]}")
    raise RuntimeError(f"kraken no ticker for {symbol}: {' | '.join(errors[-4:])}")


def okx_price(symbol: str) -> float:
    """Fallback spot price via OKX public ticker.

    Endpoint:
    - GET https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT
    """
    s = symbol.upper()
    if s.endswith("USDT"):
        inst = f"{s[:-4]}-USDT"
    elif s.endswith("USD"):
        inst = f"{s[:-3]}-USD"
    else:
        raise RuntimeError(f"okx unsupported quote: {symbol}")

    r = _http(
        "GET",
        "https://www.okx.com/api/v5/market/ticker",
        params={"instId": inst},
        timeout=10,
        headers={"accept": "application/json"},
    )
    r.raise_for_status()
    j = r.json()
    data = j.get("data") if isinstance(j, dict) else None
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"okx empty data for {inst}")
    return float((data[0] or {}).get("last"))


def cryptocompare_price(symbol: str) -> float:
    """Fallback spot price via CryptoCompare public endpoint."""
    s = symbol.upper()
    base = s
    for q in ("USDT", "USD"):
        if base.endswith(q):
            base = base[: -len(q)]
            break
    if not base:
        raise RuntimeError(f"cryptocompare unsupported symbol: {symbol}")

    r = _http(
        "GET",
        "https://min-api.cryptocompare.com/data/price",
        params={"fsym": base, "tsyms": "USD"},
        timeout=10,
        headers={"accept": "application/json"},
    )
    r.raise_for_status()
    j = r.json()
    usd = j.get("USD") if isinstance(j, dict) else None
    if usd is None:
        raise RuntimeError(f"cryptocompare missing USD for {base}")
    return float(usd)


def spot_price(symbol: str) -> Tuple[float, str, Optional[str]]:
    """Get spot price with fallbacks.

    Returns: (price, source, note)
    - source: binance|coinbase|kraken|okx|coingecko|cryptocompare
    - note: diagnostic when fallback is used
    """
    endpoints = [
        "https://data-api.binance.vision",
        "https://api1.binance.com",
        "https://api.binance.com",
    ]
    errs: List[str] = []
    for base in endpoints:
        try:
            r = _http("GET", f"{base}/api/v3/ticker/price", params={"symbol": symbol}, timeout=10)
            r.raise_for_status()
            return float(r.json()["price"]), "binance", None
        except Exception as e:
            errs.append(f"binance:{type(e).__name__}:{str(e)[:120]}")
            continue

    # Fallback 1: Coinbase Exchange
    try:
        px = coinbase_price(symbol)
        note = "; ".join(errs[-4:]) if errs else "binance_failed"
        return px, "coinbase", note
    except Exception as e:
        errs.append(f"coinbase:{type(e).__name__}:{str(e)[:120]}")

    # Fallback 2: Kraken
    try:
        px = kraken_price(symbol)
        note = "; ".join(errs[-4:]) if errs else "binance_failed"
        return px, "kraken", note
    except Exception as e:
        errs.append(f"kraken:{type(e).__name__}:{str(e)[:120]}")

    # Fallback 3: OKX
    try:
        px = okx_price(symbol)
        note = "; ".join(errs[-4:]) if errs else "binance_failed"
        return px, "okx", note
    except Exception as e:
        errs.append(f"okx:{type(e).__name__}:{str(e)[:120]}")

    # Fallback 4: CoinGecko
    try:
        px = coingecko_price(symbol)
        note = "; ".join(errs[-4:]) if errs else "binance_failed"
        return px, "coingecko", note
    except Exception as e:
        errs.append(f"coingecko:{type(e).__name__}:{str(e)[:120]}")

    # Fallback 5: CryptoCompare
    try:
        px = cryptocompare_price(symbol)
        note = "; ".join(errs[-4:]) if errs else "binance_failed"
        return px, "cryptocompare", note
    except Exception as e:
        errs.append(f"cryptocompare:{type(e).__name__}:{str(e)[:120]}")

    raise RuntimeError(f"spot_price_all_sources_failed symbol={symbol} errors={' | '.join(errs[-6:])}")


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _notional_imbalance(bids: List[Any], asks: List[Any]) -> float:
    def _side_total(levels: List[Any]) -> float:
        total = 0.0
        for row in levels:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            px = _to_float(row[0])
            qty = _to_float(row[1])
            if px is None or qty is None or px <= 0 or qty <= 0:
                continue
            total += float(px) * float(qty)
        return total

    bids_notional = _side_total(bids)
    asks_notional = _side_total(asks)
    total = bids_notional + asks_notional
    return (bids_notional - asks_notional) / total if total > 0 else 0.0


def coinbase_book_imbalance(symbol: str) -> float:
    errors: List[str] = []
    for product in _coinbase_product_candidates(symbol):
        try:
            r = _http(
                "GET",
                f"https://api.exchange.coinbase.com/products/{product}/book",
                params={"level": 2},
                timeout=10,
                headers={"accept": "application/json"},
            )
            r.raise_for_status()
            j = r.json()
            bids = (j or {}).get("bids") if isinstance(j, dict) else None
            asks = (j or {}).get("asks") if isinstance(j, dict) else None
            if not isinstance(bids, list) or not isinstance(asks, list):
                raise RuntimeError(f"coinbase malformed book for {product}")
            return _notional_imbalance(bids, asks)
        except Exception as e:
            errors.append(f"{product}:{type(e).__name__}:{str(e)[:120]}")
    raise RuntimeError(f"coinbase no book for {symbol}: {' | '.join(errors[-4:])}")


def kraken_book_imbalance(symbol: str) -> float:
    errors: List[str] = []
    for pair in _kraken_pair_candidates(symbol):
        try:
            r = _http(
                "GET",
                "https://api.kraken.com/0/public/Depth",
                params={"pair": pair, "count": 100},
                timeout=10,
                headers={"accept": "application/json"},
            )
            r.raise_for_status()
            j = r.json()
            err = (j or {}).get("error") if isinstance(j, dict) else None
            if isinstance(err, list) and err:
                raise RuntimeError(f"kraken_error:{';'.join(str(x) for x in err[:2])}")
            result = (j or {}).get("result") if isinstance(j, dict) else None
            if not isinstance(result, dict) or not result:
                raise RuntimeError(f"kraken empty result for {pair}")
            book = next(iter(result.values()))
            if not isinstance(book, dict):
                raise RuntimeError(f"kraken malformed book for {pair}")
            bids = book.get("bids")
            asks = book.get("asks")
            if not isinstance(bids, list) or not isinstance(asks, list):
                raise RuntimeError(f"kraken bids/asks missing for {pair}")
            return _notional_imbalance(bids, asks)
        except Exception as e:
            errors.append(f"{pair}:{type(e).__name__}:{str(e)[:120]}")
    raise RuntimeError(f"kraken no book for {symbol}: {' | '.join(errors[-4:])}")


def get_micro_imbalance(symbol: str) -> float:
    """Fetch orderbook depth to calculate bid/ask volume imbalance.
    Returns value between -1.0 (heavy sell pressure) to +1.0 (heavy buy pressure).
    """
    try:
        r = _http(
            "GET",
            "https://data-api.binance.vision/api/v3/depth",
            params={"symbol": symbol, "limit": 100},
            timeout=5,
        )
        if r.status_code != 200:
            raise RuntimeError(f"binance_depth_http_{int(r.status_code)}")
        data = r.json()
        bids = sum(float(x[0]) * float(x[1]) for x in data.get("bids", []))
        asks = sum(float(x[0]) * float(x[1]) for x in data.get("asks", []))
        total = bids + asks
        return (bids - asks) / total if total > 0 else 0.0
    except Exception:
        pass

    try:
        return coinbase_book_imbalance(symbol)
    except Exception:
        pass

    try:
        return kraken_book_imbalance(symbol)
    except Exception:
        pass

    return 0.0


def _fng_value_to_score(value: Any) -> Optional[float]:
    v = _to_float(value)
    if v is None:
        return None
    v = _clamp(float(v), 0.0, 100.0)
    return _clamp(((v - 50.0) / 50.0) * 3.0, -3.0, 3.0)


def alternative_fng_sentiment() -> float:
    """Public sentiment proxy from Alternative.me Fear & Greed Index (0-100)."""
    r = _http(
        "GET",
        "https://api.alternative.me/fng/",
        params={"limit": 1, "format": "json"},
        timeout=8,
        headers={"accept": "application/json"},
    )
    r.raise_for_status()
    j = r.json()
    data = j.get("data") if isinstance(j, dict) else None
    if not isinstance(data, list) or not data:
        raise RuntimeError("fng missing data")
    row = data[0] if isinstance(data[0], dict) else {}
    score = _fng_value_to_score(row.get("value"))
    if score is None:
        raise RuntimeError("fng missing numeric value")
    return float(score)


def _embedding_sentiment(summary: str) -> float:
    r = _http(
        "POST",
        "http://127.0.0.1:9999/v1/embeddings",
        json={"model": "BAAI/bge-m3", "input": summary},
        timeout=5,
    )
    if int(r.status_code) != 200:
        raise RuntimeError(f"embedding_http_{int(r.status_code)}")
    j_raw = r.json()
    j = j_raw if isinstance(j_raw, dict) else {}
    data = j.get("data") if isinstance(j, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RuntimeError("embedding missing data")
    vec = data[0].get("embedding")
    if not isinstance(vec, list) or not vec:
        raise RuntimeError("embedding empty vector")
    val = sum(float(_to_float(x) or 0.0) for x in vec[:16]) * 10.0
    return float(_clamp(val, -3.0, 3.0))


def get_semantic_sentiment() -> float:
    """Semantic sentiment proxy in [-3, 3].

    Source priority:
    1) Alternative.me Fear & Greed Index (public, no key)
    2) local embedding adapter fallback (127.0.0.1:9999)
    """
    return get_semantic_sentiment_with_source()[0]


def get_semantic_sentiment_with_source() -> Tuple[float, str]:
    """Semantic sentiment proxy plus source label."""
    try:
        return alternative_fng_sentiment(), "alternative_fng"
    except Exception:
        pass

    try:
        summary = "Crypto market regime summary fallback for embedding sentiment."
        return _embedding_sentiment(summary), "embedding_local"
    except Exception:
        return 0.0, "fallback_zero"


def _median(values: List[float]) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if n == 0:
        raise RuntimeError("median empty")
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def blockchain_n_tx_activity_score() -> float:
    """On-chain activity proxy using Blockchain.com confirmed tx/day series."""
    r = _http(
        "GET",
        "https://api.blockchain.info/charts/n-transactions",
        params={"timespan": "7days", "format": "json"},
        timeout=8,
        headers={"accept": "application/json"},
    )
    r.raise_for_status()
    j = r.json()
    rows = j.get("values") if isinstance(j, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("blockchain charts missing values")
    series = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        y = _to_float(row.get("y"))
        if y is None or y <= 0:
            continue
        series.append(float(y))
    if len(series) < 2:
        raise RuntimeError("blockchain charts insufficient series")
    latest = float(series[-1])
    baseline = _median(series[:-1])
    if baseline <= 0:
        raise RuntimeError("blockchain charts invalid baseline")
    ratio = (latest / baseline) - 1.0
    return float(_clamp(ratio * 6.0, -3.0, 3.0))


def blockchain_stats_activity_score() -> float:
    """Fallback on-chain proxy from Blockchain.com stats n_tx snapshot."""
    r = _http(
        "GET",
        "https://api.blockchain.info/stats",
        params={"format": "json"},
        timeout=8,
        headers={"accept": "application/json"},
    )
    r.raise_for_status()
    j = r.json()
    n_tx = _to_float(j.get("n_tx") if isinstance(j, dict) else None)
    if n_tx is None or n_tx <= 0:
        raise RuntimeError("blockchain stats missing n_tx")
    # Coarse normalization around long-run BTC tx/day regime.
    score = ((float(n_tx) - 400000.0) / 200000.0) * 3.0
    return float(_clamp(score, -3.0, 3.0))


def get_onchain_proxy() -> float:
    """On-chain proxy in [-3, 3] via public Blockchain.com activity feeds."""
    return get_onchain_proxy_with_source()[0]


def get_onchain_proxy_with_source() -> Tuple[float, str]:
    """On-chain proxy plus source label."""
    try:
        return blockchain_n_tx_activity_score(), "blockchain_charts_n_tx"
    except Exception:
        pass

    try:
        return blockchain_stats_activity_score(), "blockchain_stats_n_tx"
    except Exception:
        return 0.0, "fallback_zero"


def _load_prev_multimodal(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_hmm_proxy_features(
    micro_imbalance: float,
    sentiment_pca: float,
    onchain_proxy: float,
    prev: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Build lightweight HMM-like predictability proxies.

    This avoids hardcoded predictability fields by turning current multi-modal
    observations and one-step drift into entropy/likelihood/shift estimates.
    """
    prev = prev or {}
    p_micro = float(prev.get("micro_imbalance") or 0.0)
    p_sent = float(prev.get("sentiment_pca") or 0.0)
    p_onch = float(prev.get("onchain_proxy") or 0.0)

    micro_abs = abs(micro_imbalance)
    sent_abs = abs(sentiment_pca)
    onch_abs = abs(onchain_proxy)

    # distribution_shift in [0,1]
    raw_shift = (
        abs(micro_imbalance - p_micro)
        + min(3.0, abs(sentiment_pca - p_sent)) / 3.0
        + min(3.0, abs(onchain_proxy - p_onch)) / 3.0
    ) / 3.0
    distribution_shift = _clamp(raw_shift, 0.0, 1.0)

    # entropy in [0,1.5], higher means regime confusion
    entropy = (
        0.45 * micro_abs
        + 0.35 * min(3.0, sent_abs) / 3.0
        + 0.20 * min(3.0, onch_abs) / 3.0
        + 0.20 * distribution_shift
    )
    entropy = _clamp(entropy, 0.0, 1.5)

    # log-likelihood in [-3.5, -0.3], lower means worse fit
    ll = -0.30 - 2.20 * distribution_shift - 0.90 * (entropy / 1.5)
    log_likelihood = _clamp(ll, -3.5, -0.3)

    return {
        "entropy": float(round(entropy, 6)),
        "log_likelihood": float(round(log_likelihood, 6)),
        "distribution_shift": float(round(distribution_shift, 6)),
    }


def list_daily_files(suffix: str) -> List[Path]:
    d = LIE_ROOT / "output" / "daily"
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.name.endswith(suffix)])


def latest_daily_path(suffix: str, prefer_date: Optional[str] = None) -> Optional[Path]:
    """Pick latest daily artifact path.

    If prefer_date is provided and the exact file exists, return it.
    Otherwise, return the newest file by name sorting (YYYY-MM-DD_...).
    """
    d = LIE_ROOT / "output" / "daily"
    if prefer_date:
        p = d / f"{prefer_date}{suffix}"
        if p.exists():
            return p
    files = list_daily_files(suffix)
    return files[-1] if files else None


def safe_read_json(path: Optional[Path]) -> Any:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_write_json_atomic(path: Path, payload: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
        return True
    except Exception:
        return False


def _append_paper_execution_ledger(entry: Dict[str, Any]) -> bool:
    try:
        max_lines = max(200, int(float(os.getenv("LIE_PAPER_EXECUTION_LEDGER_MAX_LINES", "5000"))))
    except Exception:
        max_lines = 5000
    try:
        max_bytes = max(1_000_000, int(float(os.getenv("LIE_PAPER_EXECUTION_LEDGER_MAX_BYTES", "8000000"))))
    except Exception:
        max_bytes = 8_000_000

    try:
        PAPER_EXECUTION_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(entry)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["checksum"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        with PAPER_EXECUTION_LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

        if PAPER_EXECUTION_LEDGER_PATH.stat().st_size > max_bytes:
            lines = PAPER_EXECUTION_LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            keep = lines[-max_lines:]
            PAPER_EXECUTION_LEDGER_PATH.write_text(
                ("\n".join(keep) + ("\n" if keep else "")),
                encoding="utf-8",
            )
        return True
    except Exception:
        return False


def parse_json_tail(text: str) -> Optional[Dict[str, Any]]:
    for raw in reversed((text or "").splitlines()):
        line = str(raw or "").strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def parse_ts(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        ts = dt.datetime.fromisoformat(raw)
    except Exception:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _lock_level_rank(level: Any) -> int:
    return LOCK_GATE_LEVEL_ORDER.get(str(level or "").strip().lower(), -1)


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def resolve_paper_mode_readiness_path() -> Optional[Path]:
    override = str(os.getenv("LIE_PAPER_MODE_READINESS_PATH", "")).strip()
    if override:
        return Path(override)

    review_dir = Path(
        os.getenv(
            "LIE_PAPER_MODE_READINESS_REVIEW_DIR",
            str(LIE_ROOT / "output" / "review"),
        )
    )
    if not review_dir.exists():
        return None

    utc_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    today_report = review_dir / f"{utc_date}_pi_paper_mode_readiness_7d.json"
    if today_report.exists():
        return today_report

    reports = sorted(review_dir.glob("*_pi_paper_mode_readiness_7d.json"))
    if not reports:
        return None
    return reports[-1]


def resolve_lock_contention_report_path() -> Optional[Path]:
    override = str(os.getenv("LIE_LOCK_CONTENTION_REPORT_PATH", "")).strip()
    if override:
        return Path(override)

    review_dir = Path(
        os.getenv(
            "LIE_LOCK_CONTENTION_REVIEW_DIR",
            os.getenv(
                "LIE_PAPER_MODE_READINESS_REVIEW_DIR",
                str(LIE_ROOT / "output" / "review"),
            ),
        )
    )
    if not review_dir.exists():
        return None

    utc_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
    today_reports = sorted(
        review_dir.glob(f"{utc_date}_pi_paper_artifacts_lock_contention_*.json")
    )
    if today_reports:
        return today_reports[-1]

    reports = sorted(review_dir.glob("*_pi_paper_artifacts_lock_contention_*.json"))
    if not reports:
        return None
    return reports[-1]


def evaluate_lock_contention_gate(
    *,
    payload: Any,
    report_path: Optional[Path],
    enforce: bool,
    fail_closed: bool,
    max_age_hours: float,
    max_allowed_level: str,
    required_source_bucket: str,
    require_bucket_match: bool,
    require_sample_guard: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enforce": enforce,
        "fail_closed": fail_closed,
        "report_path": str(report_path) if report_path else None,
        "report_present": False,
        "overall_level": None,
        "source_bucket_view": None,
        "source_bucket_requested": None,
        "source_bucket_effective": None,
        "fail_triggered": None,
        "generated_at": None,
        "report_age_hours": None,
        "report_stale": None,
        "max_age_hours": max_age_hours,
        "max_allowed_level": max_allowed_level,
        "required_source_bucket": required_source_bucket,
        "require_bucket_match": require_bucket_match,
        "require_sample_guard": require_sample_guard,
        "sample_guard_level": None,
        "sample_guard_fail_triggered": None,
        "sample_guard_fail_on_level": None,
        "blocked": False,
        "reason": "disabled",
    }
    if not enforce:
        return out

    if report_path is None:
        out["reason"] = (
            "lock_contention_report_missing_fail_closed"
            if fail_closed
            else "lock_contention_report_missing_fail_open"
        )
        out["blocked"] = bool(fail_closed)
        return out

    if not isinstance(payload, dict):
        out["reason"] = (
            "lock_contention_report_invalid_fail_closed"
            if fail_closed
            else "lock_contention_report_invalid_fail_open"
        )
        out["blocked"] = bool(fail_closed)
        return out

    out["report_present"] = True
    gate = payload.get("gate")
    gate_dict = gate if isinstance(gate, dict) else {}
    overall_level = str(gate_dict.get("overall_level") or "").strip().lower()
    source_bucket_view = str(gate_dict.get("source_bucket_view") or "").strip().lower()
    source_bucket_requested = str(gate_dict.get("source_bucket_requested") or "").strip().lower()
    source_bucket_effective = str(gate_dict.get("source_bucket_effective") or "").strip().lower()
    fail_triggered = _coerce_bool(gate_dict.get("fail_triggered"))

    out["overall_level"] = overall_level or None
    out["source_bucket_view"] = source_bucket_view or None
    out["source_bucket_requested"] = source_bucket_requested or None
    out["source_bucket_effective"] = source_bucket_effective or None
    out["fail_triggered"] = fail_triggered
    sample_guard = gate_dict.get("sample_guard") if isinstance(gate_dict.get("sample_guard"), dict) else None
    if isinstance(sample_guard, dict):
        out["sample_guard_level"] = str(sample_guard.get("overall_level") or "").strip().lower() or None
        out["sample_guard_fail_triggered"] = _coerce_bool(sample_guard.get("fail_triggered"))
        out["sample_guard_fail_on_level"] = str(sample_guard.get("fail_on_level") or "").strip().lower() or None

    generated_at = parse_ts(payload.get("generated_at") or payload.get("as_of"))
    if generated_at is not None:
        out["generated_at"] = generated_at.isoformat()
        age_hours = max(
            0.0,
            (dt.datetime.now(dt.timezone.utc) - generated_at).total_seconds() / 3600.0,
        )
        out["report_age_hours"] = round(age_hours, 4)
        out["report_stale"] = bool(max_age_hours > 0 and age_hours > max_age_hours)
    else:
        out["report_stale"] = None

    if out["report_stale"] is True:
        out["reason"] = (
            "lock_contention_report_stale_fail_closed"
            if fail_closed
            else "lock_contention_report_stale_fail_open"
        )
        out["blocked"] = bool(fail_closed)
        return out

    if fail_triggered is True:
        out["reason"] = "lock_contention_fail_triggered"
        out["blocked"] = True
        return out

    if require_sample_guard:
        if sample_guard is None:
            out["reason"] = (
                "lock_contention_sample_guard_missing_fail_closed"
                if fail_closed
                else "lock_contention_sample_guard_missing_fail_open"
            )
            out["blocked"] = bool(fail_closed)
            return out
        if out.get("sample_guard_fail_triggered") is True:
            out["reason"] = "lock_contention_sample_guard_fail_triggered"
            out["blocked"] = True
            return out

    normalized_required_bucket = (
        required_source_bucket if required_source_bucket in {"all", "prod", "drill", "unknown"} else "prod"
    )
    if require_bucket_match and normalized_required_bucket != "all":
        if source_bucket_effective != normalized_required_bucket:
            out["reason"] = "lock_contention_source_bucket_mismatch"
            out["blocked"] = True
            return out

    level_rank = _lock_level_rank(overall_level)
    allowed_rank = _lock_level_rank(max_allowed_level)
    if level_rank >= 0 and allowed_rank >= 0 and level_rank >= allowed_rank:
        out["reason"] = "lock_contention_level_exceeded"
        out["blocked"] = True
        return out

    if level_rank < 0:
        out["reason"] = (
            "lock_contention_level_unknown_fail_closed"
            if fail_closed
            else "lock_contention_level_unknown_fail_open"
        )
        out["blocked"] = bool(fail_closed)
        return out

    out["reason"] = "lock_contention_gate_ok"
    return out


def _run_paper_mode_readiness_refresh(reason: str) -> Dict[str, Any]:
    auto_refresh_enabled = env_flag("LIE_PAPER_MODE_READINESS_AUTO_REFRESH", default=True)
    force_refresh = env_flag("LIE_PAPER_MODE_READINESS_FORCE_REFRESH", default=False)
    try:
        min_interval_sec = max(
            0,
            int(float(os.getenv("LIE_PAPER_MODE_READINESS_REFRESH_MIN_INTERVAL_SEC", "900"))),
        )
    except Exception:
        min_interval_sec = 900
    out: Dict[str, Any] = {
        "attempted": False,
        "ok": False,
        "reason": "refresh_disabled",
        "trigger": reason,
        "returncode": None,
        "duration_sec": None,
        "report_path": None,
        "stdout_tail": None,
        "stderr_tail": None,
        "throttled": False,
        "throttle_remaining_sec": None,
        "lock_path": str(READINESS_REFRESH_LOCK_PATH),
        "lock_acquired": None,
        "state_path": str(READINESS_REFRESH_STATE_PATH),
        "last_attempt_ts": None,
        "last_success_ts": None,
        "lock_contention_attempted": False,
        "lock_contention_ok": None,
        "lock_contention_reason": "not_run",
        "lock_contention_returncode": None,
        "lock_contention_duration_sec": None,
        "lock_contention_stdout_tail": None,
        "lock_contention_stderr_tail": None,
        "lock_contention_json_path": None,
        "lock_contention_md_path": None,
        "lock_contention_csv_path": None,
        "lock_contention_source_coverage_csv": None,
        "lock_contention_gate_level": None,
        "lock_contention_gate_fail_triggered": None,
        "lock_contention_gate_source_bucket_view": None,
        "lock_contention_gate_source_bucket_requested": None,
        "lock_contention_gate_source_bucket_effective": None,
        "lock_contention_gate_sample_guard_level": None,
        "lock_contention_gate_sample_guard_fail_triggered": None,
        "lock_contention_gate_sample_guard_fail_on_level": None,
    }
    if (not auto_refresh_enabled) and (not force_refresh):
        return out

    script_path = Path(
        os.getenv(
            "LIE_PAPER_MODE_RECONSTRUCT_SCRIPT",
            str(LIE_ROOT / "scripts" / "reconstruct_pi_halfhour_matrix.py"),
        )
    )
    if not script_path.exists():
        out["reason"] = "reconstruct_script_missing"
        return out

    try:
        timeout_sec = max(
            30,
            int(float(os.getenv("LIE_PAPER_MODE_RECONSTRUCT_TIMEOUT_SEC", "240"))),
        )
    except Exception:
        timeout_sec = 240

    cmd = ["python3", str(script_path)]
    output_dir = str(os.getenv("LIE_PAPER_MODE_RECONSTRUCT_OUTPUT_DIR", "")).strip()
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    warmup_log_path = str(os.getenv("LIE_PAPER_MODE_WARMUP_LOG_PATH", "")).strip()
    if warmup_log_path:
        cmd.extend(["--launchd-log", warmup_log_path])
    for env_name, flag_name in (
        ("LIE_PAPER_MODE_WARMUP_WINDOW_HOURS", "--warmup-window-hours"),
        ("LIE_PAPER_MODE_WARMUP_COVERAGE_MIN", "--warmup-coverage-min"),
        ("LIE_PAPER_MODE_WARMUP_MAX_MISSING_BUCKETS", "--warmup-max-missing-buckets"),
        (
            "LIE_PAPER_MODE_WARMUP_MAX_LARGEST_MISSING_BLOCK_HOURS",
            "--warmup-max-largest-missing-block-hours",
        ),
    ):
        raw_value = str(os.getenv(env_name, "")).strip()
        if raw_value:
            cmd.extend([flag_name, raw_value])

    READINESS_REFRESH_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    READINESS_REFRESH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    lockf = None
    try:
        import fcntl  # type: ignore

        lockf = READINESS_REFRESH_LOCK_PATH.open("w")
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        out["lock_acquired"] = True
    except Exception:
        out["reason"] = "refresh_lock_busy"
        out["lock_acquired"] = False
        return out

    try:
        state_raw = safe_read_json(READINESS_REFRESH_STATE_PATH)
        state = state_raw if isinstance(state_raw, dict) else {}
        prev_attempt_ts = parse_ts(state.get("last_attempt_ts"))
        prev_success_ts = parse_ts(state.get("last_success_ts"))
        if prev_attempt_ts is not None:
            out["last_attempt_ts"] = prev_attempt_ts.isoformat()
        if prev_success_ts is not None:
            out["last_success_ts"] = prev_success_ts.isoformat()

        now_utc = dt.datetime.now(dt.timezone.utc)
        if (not force_refresh) and min_interval_sec > 0 and prev_attempt_ts is not None:
            elapsed_sec = max(0.0, (now_utc - prev_attempt_ts).total_seconds())
            if elapsed_sec < float(min_interval_sec):
                out["reason"] = "refresh_throttled"
                out["throttled"] = True
                out["throttle_remaining_sec"] = int(
                    max(1.0, float(min_interval_sec) - float(elapsed_sec) + 0.999)
                )
                return out

        started = dt.datetime.now(dt.timezone.utc)
        started_iso = started.isoformat()
        out["attempted"] = True
        out["last_attempt_ts"] = started_iso
        safe_write_json_atomic(
            READINESS_REFRESH_STATE_PATH,
            {
                "updated_at": started_iso,
                "last_attempt_ts": started_iso,
                "last_attempt_trigger": reason,
                "last_attempt_result": "running",
                "last_attempt_returncode": None,
                "last_attempt_duration_sec": None,
                "last_success_ts": prev_success_ts.isoformat() if prev_success_ts else None,
            },
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(LIE_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            out["reason"] = "reconstruct_timeout"
            out["duration_sec"] = round(max(0.0, float(elapsed)), 3)
            out["stdout_tail"] = str(exc.stdout or "")[-1200:] or None
            out["stderr_tail"] = str(exc.stderr or "")[-1200:] or None
            safe_write_json_atomic(
                READINESS_REFRESH_STATE_PATH,
                {
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "last_attempt_ts": started_iso,
                    "last_attempt_trigger": reason,
                    "last_attempt_result": out["reason"],
                    "last_attempt_returncode": None,
                    "last_attempt_duration_sec": out["duration_sec"],
                    "last_success_ts": prev_success_ts.isoformat() if prev_success_ts else None,
                },
            )
            return out
        except Exception as exc:
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            out["reason"] = f"reconstruct_runner_error:{type(exc).__name__}"
            out["duration_sec"] = round(max(0.0, float(elapsed)), 3)
            safe_write_json_atomic(
                READINESS_REFRESH_STATE_PATH,
                {
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "last_attempt_ts": started_iso,
                    "last_attempt_trigger": reason,
                    "last_attempt_result": out["reason"],
                    "last_attempt_returncode": None,
                    "last_attempt_duration_sec": out["duration_sec"],
                    "last_success_ts": prev_success_ts.isoformat() if prev_success_ts else None,
                },
            )
            return out

        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        stdout = str(proc.stdout or "")
        stderr = str(proc.stderr or "")
        out["returncode"] = int(proc.returncode)
        out["duration_sec"] = round(max(0.0, float(elapsed)), 3)
        out["stdout_tail"] = stdout[-1200:] or None
        out["stderr_tail"] = stderr[-1200:] or None
        if proc.returncode != 0:
            out["reason"] = "reconstruct_failed"
            safe_write_json_atomic(
                READINESS_REFRESH_STATE_PATH,
                {
                    "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "last_attempt_ts": started_iso,
                    "last_attempt_trigger": reason,
                    "last_attempt_result": out["reason"],
                    "last_attempt_returncode": int(proc.returncode),
                    "last_attempt_duration_sec": out["duration_sec"],
                    "last_success_ts": prev_success_ts.isoformat() if prev_success_ts else None,
                },
            )
            return out

        payload: Optional[Dict[str, Any]] = None
        raw = stdout.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                candidate = json.loads(raw)
                if isinstance(candidate, dict):
                    payload = candidate
            except Exception:
                payload = None
        if payload is None:
            payload = parse_json_tail(stdout)

        report_path_raw = ""
        if isinstance(payload, dict):
            report_path_raw = str(payload.get("paper_readiness_json") or "").strip()
        if report_path_raw:
            out["report_path"] = report_path_raw

        lock_contention_enabled = env_flag("LIE_LOCK_CONTENTION_AUTO_REFRESH", default=True)
        if not lock_contention_enabled:
            out["lock_contention_reason"] = "auto_refresh_disabled"
        else:
            lock_script_path = Path(
                os.getenv(
                    "LIE_LOCK_CONTENTION_REPORT_SCRIPT",
                    str(LIE_ROOT / "scripts" / "report_paper_artifacts_lock_contention.py"),
                )
            )
            if not lock_script_path.exists():
                out["lock_contention_reason"] = "script_missing"
            else:
                try:
                    lock_timeout_sec = max(
                        30,
                        int(float(os.getenv("LIE_LOCK_CONTENTION_REPORT_TIMEOUT_SEC", "240"))),
                    )
                except Exception:
                    lock_timeout_sec = 240

                lock_cmd = ["python3", str(lock_script_path)]
                lock_output_dir = str(os.getenv("LIE_LOCK_CONTENTION_OUTPUT_DIR", output_dir)).strip()
                if lock_output_dir:
                    lock_cmd.extend(["--output-dir", lock_output_dir])

                lock_state_md = str(os.getenv("LIE_LOCK_CONTENTION_STATE_MD", str(STATE_MD))).strip()
                if lock_state_md:
                    lock_cmd.extend(["--state-md", lock_state_md])

                lock_pi_cycle_log = str(os.getenv("LIE_LOCK_CONTENTION_PI_CYCLE_LOG", "")).strip()
                if lock_pi_cycle_log:
                    lock_cmd.extend(["--pi-cycle-log", lock_pi_cycle_log])

                lock_days = str(os.getenv("LIE_LOCK_CONTENTION_DAYS", "")).strip()
                if lock_days:
                    lock_cmd.extend(["--days", lock_days])

                lock_min_pi_cycle = str(
                    os.getenv("LIE_LOCK_CONTENTION_MIN_PI_CYCLE_ATTEMPTED", "")
                ).strip()
                if lock_min_pi_cycle:
                    lock_cmd.extend(["--min-pi-cycle-attempted", lock_min_pi_cycle])

                lock_drill_labels = str(os.getenv("LIE_LOCK_CONTENTION_DRILL_LABELS", "")).strip()
                if lock_drill_labels:
                    lock_cmd.extend(["--drill-labels", lock_drill_labels])

                lock_prod_labels = str(os.getenv("LIE_LOCK_CONTENTION_PROD_LABELS", "")).strip()
                if lock_prod_labels:
                    lock_cmd.extend(["--prod-labels", lock_prod_labels])

                lock_missing_source_bucket = str(
                    os.getenv("LIE_LOCK_CONTENTION_MISSING_SOURCE_BUCKET", "")
                ).strip().lower()
                if lock_missing_source_bucket in {"unknown", "prod", "drill"}:
                    lock_cmd.extend(["--missing-source-bucket", lock_missing_source_bucket])

                lock_gate_source_bucket = str(
                    os.getenv("LIE_LOCK_CONTENTION_GATE_SOURCE_BUCKET", "prod")
                ).strip().lower()
                if lock_gate_source_bucket in {"all", "prod", "drill", "unknown"}:
                    lock_cmd.extend(["--gate-source-bucket", lock_gate_source_bucket])
                lock_gate_source_bucket_view = str(
                    os.getenv("LIE_LOCK_CONTENTION_GATE_SOURCE_BUCKET_VIEW", "")
                ).strip().lower()
                if lock_gate_source_bucket_view in {"effective", "raw"}:
                    lock_cmd.extend(["--gate-source-bucket-view", lock_gate_source_bucket_view])

                lock_missing_source_infer = str(
                    os.getenv("LIE_LOCK_CONTENTION_MISSING_SOURCE_INFER", "")
                ).strip().lower()
                if lock_missing_source_infer in {"off", "default", "nearest"}:
                    lock_cmd.extend(["--missing-source-infer", lock_missing_source_infer])
                lock_missing_source_default_bucket = str(
                    os.getenv("LIE_LOCK_CONTENTION_MISSING_SOURCE_DEFAULT_BUCKET", "")
                ).strip().lower()
                if lock_missing_source_default_bucket in {"prod", "drill", "unknown"}:
                    lock_cmd.extend(
                        ["--missing-source-default-bucket", lock_missing_source_default_bucket]
                    )
                lock_missing_source_nearest_max_minutes = str(
                    os.getenv("LIE_LOCK_CONTENTION_MISSING_SOURCE_NEAREST_MAX_MINUTES", "")
                ).strip()
                if lock_missing_source_nearest_max_minutes:
                    lock_cmd.extend(
                        [
                            "--missing-source-nearest-max-minutes",
                            lock_missing_source_nearest_max_minutes,
                        ]
                    )

                lock_fail_on_level = str(
                    os.getenv("LIE_LOCK_CONTENTION_FAIL_ON_LEVEL", "off")
                ).strip().lower()
                if lock_fail_on_level in {"off", "warn", "degraded", "critical"}:
                    lock_cmd.extend(["--fail-on-level", lock_fail_on_level])

                lock_prod_min_warn = str(
                    os.getenv("LIE_LOCK_CONTENTION_PROD_ATTEMPTED_MIN_WARN", "")
                ).strip()
                if lock_prod_min_warn:
                    lock_cmd.extend(["--prod-attempted-min-warn", lock_prod_min_warn])
                lock_prod_min_degraded = str(
                    os.getenv("LIE_LOCK_CONTENTION_PROD_ATTEMPTED_MIN_DEGRADED", "")
                ).strip()
                if lock_prod_min_degraded:
                    lock_cmd.extend(["--prod-attempted-min-degraded", lock_prod_min_degraded])
                lock_prod_min_critical = str(
                    os.getenv("LIE_LOCK_CONTENTION_PROD_ATTEMPTED_MIN_CRITICAL", "")
                ).strip()
                if lock_prod_min_critical:
                    lock_cmd.extend(["--prod-attempted-min-critical", lock_prod_min_critical])
                lock_prod_min_min_attempted = str(
                    os.getenv("LIE_LOCK_CONTENTION_PROD_ATTEMPTED_MIN_MIN_ATTEMPTED", "")
                ).strip()
                if lock_prod_min_min_attempted:
                    lock_cmd.extend(
                        ["--prod-attempted-min-min-attempted", lock_prod_min_min_attempted]
                    )

                lock_unknown_rate_warn = str(
                    os.getenv("LIE_LOCK_CONTENTION_UNKNOWN_ATTEMPTED_RATE_WARN", "")
                ).strip()
                if lock_unknown_rate_warn:
                    lock_cmd.extend(["--unknown-attempted-rate-warn", lock_unknown_rate_warn])
                lock_unknown_rate_degraded = str(
                    os.getenv("LIE_LOCK_CONTENTION_UNKNOWN_ATTEMPTED_RATE_DEGRADED", "")
                ).strip()
                if lock_unknown_rate_degraded:
                    lock_cmd.extend(
                        ["--unknown-attempted-rate-degraded", lock_unknown_rate_degraded]
                    )
                lock_unknown_rate_critical = str(
                    os.getenv("LIE_LOCK_CONTENTION_UNKNOWN_ATTEMPTED_RATE_CRITICAL", "")
                ).strip()
                if lock_unknown_rate_critical:
                    lock_cmd.extend(
                        ["--unknown-attempted-rate-critical", lock_unknown_rate_critical]
                    )
                lock_unknown_rate_min_attempted = str(
                    os.getenv("LIE_LOCK_CONTENTION_UNKNOWN_ATTEMPTED_RATE_MIN_ATTEMPTED", "")
                ).strip()
                if lock_unknown_rate_min_attempted:
                    lock_cmd.extend(
                        [
                            "--unknown-attempted-rate-min-attempted",
                            lock_unknown_rate_min_attempted,
                        ]
                    )
                lock_sample_guard_fail_on_level = str(
                    os.getenv("LIE_LOCK_CONTENTION_SAMPLE_GUARD_FAIL_ON_LEVEL", "")
                ).strip().lower()
                if lock_sample_guard_fail_on_level in {"off", "warn", "degraded", "critical"}:
                    lock_cmd.extend(
                        ["--sample-guard-fail-on-level", lock_sample_guard_fail_on_level]
                    )

                lock_started = dt.datetime.now(dt.timezone.utc)
                out["lock_contention_attempted"] = True
                try:
                    lock_proc = subprocess.run(
                        lock_cmd,
                        cwd=str(LIE_ROOT),
                        capture_output=True,
                        text=True,
                        timeout=lock_timeout_sec,
                    )
                except subprocess.TimeoutExpired as exc:
                    lock_elapsed = (dt.datetime.now(dt.timezone.utc) - lock_started).total_seconds()
                    out["lock_contention_ok"] = False
                    out["lock_contention_reason"] = "timeout"
                    out["lock_contention_duration_sec"] = round(max(0.0, float(lock_elapsed)), 3)
                    out["lock_contention_stdout_tail"] = str(exc.stdout or "")[-1200:] or None
                    out["lock_contention_stderr_tail"] = str(exc.stderr or "")[-1200:] or None
                except Exception as exc:
                    lock_elapsed = (dt.datetime.now(dt.timezone.utc) - lock_started).total_seconds()
                    out["lock_contention_ok"] = False
                    out["lock_contention_reason"] = f"runner_error:{type(exc).__name__}"
                    out["lock_contention_duration_sec"] = round(max(0.0, float(lock_elapsed)), 3)
                else:
                    lock_elapsed = (dt.datetime.now(dt.timezone.utc) - lock_started).total_seconds()
                    lock_stdout = str(lock_proc.stdout or "")
                    lock_stderr = str(lock_proc.stderr or "")
                    out["lock_contention_returncode"] = int(lock_proc.returncode)
                    out["lock_contention_duration_sec"] = round(max(0.0, float(lock_elapsed)), 3)
                    out["lock_contention_stdout_tail"] = lock_stdout[-1200:] or None
                    out["lock_contention_stderr_tail"] = lock_stderr[-1200:] or None
                    if lock_proc.returncode == 0:
                        out["lock_contention_ok"] = True
                        out["lock_contention_reason"] = "ok"
                    elif lock_proc.returncode == 2:
                        out["lock_contention_ok"] = False
                        out["lock_contention_reason"] = "gate_fail_triggered"
                    else:
                        out["lock_contention_ok"] = False
                        out["lock_contention_reason"] = "report_failed"

                    lock_payload: Optional[Dict[str, Any]] = None
                    lock_raw = lock_stdout.strip()
                    if lock_raw.startswith("{") and lock_raw.endswith("}"):
                        try:
                            parsed_lock = json.loads(lock_raw)
                            if isinstance(parsed_lock, dict):
                                lock_payload = parsed_lock
                        except Exception:
                            lock_payload = None
                    if lock_payload is None:
                        lock_payload = parse_json_tail(lock_stdout)
                    if isinstance(lock_payload, dict):
                        out["lock_contention_json_path"] = (
                            str(lock_payload.get("json_path") or "").strip() or None
                        )
                        out["lock_contention_md_path"] = (
                            str(lock_payload.get("md_path") or "").strip() or None
                        )
                        out["lock_contention_csv_path"] = (
                            str(lock_payload.get("csv_path") or "").strip() or None
                        )
                        out["lock_contention_source_coverage_csv"] = (
                            str(lock_payload.get("source_coverage_csv") or "").strip() or None
                        )
                        gate_payload = lock_payload.get("gate")
                        if isinstance(gate_payload, dict):
                            out["lock_contention_gate_level"] = (
                                str(gate_payload.get("overall_level") or "").strip() or None
                            )
                            out["lock_contention_gate_fail_triggered"] = (
                                bool(gate_payload.get("fail_triggered"))
                                if gate_payload.get("fail_triggered") is not None
                                else None
                            )
                            out["lock_contention_gate_source_bucket_view"] = (
                                str(gate_payload.get("source_bucket_view") or "").strip() or None
                            )
                            out["lock_contention_gate_source_bucket_requested"] = (
                                str(gate_payload.get("source_bucket_requested") or "").strip() or None
                            )
                            out["lock_contention_gate_source_bucket_effective"] = (
                                str(gate_payload.get("source_bucket_effective") or "").strip() or None
                            )
                            sample_guard_payload = (
                                gate_payload.get("sample_guard")
                                if isinstance(gate_payload.get("sample_guard"), dict)
                                else None
                            )
                            if isinstance(sample_guard_payload, dict):
                                out["lock_contention_gate_sample_guard_level"] = (
                                    str(sample_guard_payload.get("overall_level") or "").strip() or None
                                )
                                out["lock_contention_gate_sample_guard_fail_triggered"] = (
                                    bool(sample_guard_payload.get("fail_triggered"))
                                    if sample_guard_payload.get("fail_triggered") is not None
                                    else None
                                )
                                out["lock_contention_gate_sample_guard_fail_on_level"] = (
                                    str(sample_guard_payload.get("fail_on_level") or "").strip() or None
                                )

        completed_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        out["last_success_ts"] = completed_iso
        out["ok"] = True
        out["reason"] = "reconstruct_ok"
        safe_write_json_atomic(
            READINESS_REFRESH_STATE_PATH,
            {
                "updated_at": completed_iso,
                "last_attempt_ts": started_iso,
                "last_attempt_trigger": reason,
                "last_attempt_result": out["reason"],
                "last_attempt_returncode": int(proc.returncode),
                "last_attempt_duration_sec": out["duration_sec"],
                "last_success_ts": completed_iso,
                "last_report_path": out["report_path"],
                "last_lock_contention_attempted": bool(out.get("lock_contention_attempted")),
                "last_lock_contention_ok": out.get("lock_contention_ok"),
                "last_lock_contention_reason": out.get("lock_contention_reason"),
                "last_lock_contention_returncode": out.get("lock_contention_returncode"),
                "last_lock_contention_duration_sec": out.get("lock_contention_duration_sec"),
                "last_lock_contention_json_path": out.get("lock_contention_json_path"),
                "last_lock_contention_source_coverage_csv": out.get("lock_contention_source_coverage_csv"),
                "last_lock_contention_gate_level": out.get("lock_contention_gate_level"),
                "last_lock_contention_gate_fail_triggered": out.get(
                    "lock_contention_gate_fail_triggered"
                ),
                "last_lock_contention_gate_source_bucket_view": out.get(
                    "lock_contention_gate_source_bucket_view"
                ),
                "last_lock_contention_gate_source_bucket_requested": out.get(
                    "lock_contention_gate_source_bucket_requested"
                ),
                "last_lock_contention_gate_source_bucket_effective": out.get(
                    "lock_contention_gate_source_bucket_effective"
                ),
                "last_lock_contention_gate_sample_guard_level": out.get(
                    "lock_contention_gate_sample_guard_level"
                ),
                "last_lock_contention_gate_sample_guard_fail_triggered": out.get(
                    "lock_contention_gate_sample_guard_fail_triggered"
                ),
                "last_lock_contention_gate_sample_guard_fail_on_level": out.get(
                    "lock_contention_gate_sample_guard_fail_on_level"
                ),
            },
        )
        return out
    finally:
        if lockf is not None:
            try:
                lockf.close()
            except Exception:
                pass


def load_paper_mode_readiness_gate() -> Dict[str, Any]:
    enforce = env_flag("LIE_PAPER_MODE_READINESS_ENFORCE", default=True)
    fail_closed = env_flag("LIE_PAPER_MODE_READINESS_FAIL_CLOSED", default=True)
    allow_warmup = env_flag("LIE_PAPER_MODE_READINESS_ALLOW_WARMUP", default=False)
    try:
        max_age_hours = max(
            0.0,
            float(os.getenv("LIE_PAPER_MODE_READINESS_MAX_AGE_HOURS", "36")),
        )
    except Exception:
        max_age_hours = 36.0
    lock_gate_enforce = env_flag("LIE_LOCK_CONTENTION_GATE_ENFORCE", default=True)
    lock_gate_fail_closed = env_flag("LIE_LOCK_CONTENTION_GATE_FAIL_CLOSED", default=False)
    lock_gate_require_bucket_match = env_flag("LIE_LOCK_CONTENTION_GATE_REQUIRE_BUCKET_MATCH", default=False)
    lock_gate_require_sample_guard = env_flag(
        "LIE_LOCK_CONTENTION_GATE_REQUIRE_SAMPLE_GUARD",
        default=False,
    )
    lock_gate_max_allowed_level = str(
        os.getenv("LIE_LOCK_CONTENTION_GATE_MAX_ALLOWED_LEVEL", "degraded")
    ).strip().lower()
    if lock_gate_max_allowed_level not in {"warn", "degraded", "critical"}:
        lock_gate_max_allowed_level = "degraded"
    lock_gate_required_source_bucket = str(
        os.getenv("LIE_LOCK_CONTENTION_GATE_REQUIRED_SOURCE_BUCKET", "prod")
    ).strip().lower()
    if lock_gate_required_source_bucket not in {"all", "prod", "drill", "unknown"}:
        lock_gate_required_source_bucket = "prod"
    try:
        lock_gate_max_age_hours = max(
            0.0,
            float(os.getenv("LIE_LOCK_CONTENTION_GATE_MAX_AGE_HOURS", "36")),
        )
    except Exception:
        lock_gate_max_age_hours = 36.0

    out: Dict[str, Any] = {
        "enabled": True,
        "enforce": enforce,
        "fail_closed": fail_closed,
        "report_path": None,
        "report_present": False,
        "status": "unknown",
        "ready_for_paper_mode": None,
        "coverage": None,
        "missing_buckets": None,
        "largest_missing_block_hours": None,
        "fail_reasons": [],
        "report_ts": None,
        "report_age_hours": None,
        "report_stale": None,
        "max_age_hours": max_age_hours,
        "gate_blocked": False,
        "gate_reason": "enforcement_disabled",
        "warmup_gate": None,
        "warmup_override_enabled": allow_warmup,
        "warmup_override_applied": False,
        "warmup_override_reason": "disabled" if not allow_warmup else "not_evaluated",
        "refresh_attempted": False,
        "refresh_ok": None,
        "refresh_reason": "not_needed",
        "refresh_trigger": None,
        "refresh_returncode": None,
        "refresh_duration_sec": None,
        "refresh_stdout_tail": None,
        "refresh_stderr_tail": None,
        "refresh_throttled": None,
        "refresh_throttle_remaining_sec": None,
        "refresh_lock_path": str(READINESS_REFRESH_LOCK_PATH),
        "refresh_lock_acquired": None,
        "refresh_state_path": str(READINESS_REFRESH_STATE_PATH),
        "refresh_last_attempt_ts": None,
        "refresh_last_success_ts": None,
        "refresh_lock_contention": None,
        "lock_contention_gate": {
            "enforce": lock_gate_enforce,
            "fail_closed": lock_gate_fail_closed,
            "report_path": None,
            "report_present": False,
            "overall_level": None,
            "source_bucket_view": None,
            "source_bucket_requested": None,
            "source_bucket_effective": None,
            "fail_triggered": None,
            "generated_at": None,
            "report_age_hours": None,
            "report_stale": None,
            "max_age_hours": lock_gate_max_age_hours,
            "max_allowed_level": lock_gate_max_allowed_level,
            "required_source_bucket": lock_gate_required_source_bucket,
            "require_bucket_match": lock_gate_require_bucket_match,
            "require_sample_guard": lock_gate_require_sample_guard,
            "sample_guard_level": None,
            "sample_guard_fail_triggered": None,
            "sample_guard_fail_on_level": None,
            "blocked": False,
            "reason": "not_evaluated",
        },
    }
    lock_report_path: Optional[Path] = None
    lock_payload: Any = None

    if not enforce:
        return out

    report_path = resolve_paper_mode_readiness_path()
    payload = safe_read_json(report_path) if report_path else None

    refresh_trigger: Optional[str] = None
    if env_flag("LIE_PAPER_MODE_READINESS_FORCE_REFRESH", default=False):
        refresh_trigger = "forced_refresh"
    elif report_path is None:
        refresh_trigger = "report_missing"
    elif not isinstance(payload, dict):
        refresh_trigger = "report_invalid"
    else:
        pre_report_ts = parse_ts(payload.get("window_end_utc") or payload.get("report_ts"))
        if pre_report_ts is None and fail_closed:
            refresh_trigger = "report_ts_missing"
        elif pre_report_ts is not None and max_age_hours > 0:
            pre_age_hours = max(
                0.0,
                (dt.datetime.now(dt.timezone.utc) - pre_report_ts).total_seconds() / 3600.0,
            )
            if pre_age_hours > max_age_hours:
                refresh_trigger = "report_stale"

    if refresh_trigger:
        refresh = _run_paper_mode_readiness_refresh(refresh_trigger)
        out["refresh_attempted"] = bool(refresh.get("attempted"))
        out["refresh_ok"] = bool(refresh.get("ok")) if refresh.get("attempted") else None
        out["refresh_reason"] = str(refresh.get("reason") or "refresh_unknown")
        out["refresh_trigger"] = str(refresh.get("trigger") or refresh_trigger)
        out["refresh_returncode"] = refresh.get("returncode")
        out["refresh_duration_sec"] = refresh.get("duration_sec")
        out["refresh_stdout_tail"] = refresh.get("stdout_tail")
        out["refresh_stderr_tail"] = refresh.get("stderr_tail")
        out["refresh_throttled"] = (
            bool(refresh.get("throttled"))
            if refresh.get("throttled") is not None
            else None
        )
        out["refresh_throttle_remaining_sec"] = refresh.get("throttle_remaining_sec")
        out["refresh_lock_path"] = str(refresh.get("lock_path") or "").strip() or None
        out["refresh_lock_acquired"] = (
            bool(refresh.get("lock_acquired"))
            if refresh.get("lock_acquired") is not None
            else None
        )
        out["refresh_state_path"] = str(refresh.get("state_path") or "").strip() or None
        out["refresh_last_attempt_ts"] = (
            str(refresh.get("last_attempt_ts") or "").strip() or None
        )
        out["refresh_last_success_ts"] = (
            str(refresh.get("last_success_ts") or "").strip() or None
        )
        out["refresh_lock_contention"] = {
            "attempted": bool(refresh.get("lock_contention_attempted")),
            "ok": (
                bool(refresh.get("lock_contention_ok"))
                if refresh.get("lock_contention_ok") is not None
                else None
            ),
            "reason": str(refresh.get("lock_contention_reason") or "").strip() or None,
            "returncode": refresh.get("lock_contention_returncode"),
            "duration_sec": refresh.get("lock_contention_duration_sec"),
            "stdout_tail": refresh.get("lock_contention_stdout_tail"),
            "stderr_tail": refresh.get("lock_contention_stderr_tail"),
            "json_path": str(refresh.get("lock_contention_json_path") or "").strip() or None,
            "md_path": str(refresh.get("lock_contention_md_path") or "").strip() or None,
            "csv_path": str(refresh.get("lock_contention_csv_path") or "").strip() or None,
            "source_coverage_csv": (
                str(refresh.get("lock_contention_source_coverage_csv") or "").strip() or None
            ),
            "gate_level": str(refresh.get("lock_contention_gate_level") or "").strip() or None,
            "gate_fail_triggered": (
                bool(refresh.get("lock_contention_gate_fail_triggered"))
                if refresh.get("lock_contention_gate_fail_triggered") is not None
                else None
            ),
            "gate_source_bucket_view": (
                str(refresh.get("lock_contention_gate_source_bucket_view") or "").strip() or None
            ),
            "gate_source_bucket_requested": (
                str(refresh.get("lock_contention_gate_source_bucket_requested") or "").strip() or None
            ),
            "gate_source_bucket_effective": (
                str(refresh.get("lock_contention_gate_source_bucket_effective") or "").strip()
                or None
            ),
            "gate_sample_guard_level": (
                str(refresh.get("lock_contention_gate_sample_guard_level") or "").strip() or None
            ),
            "gate_sample_guard_fail_triggered": (
                bool(refresh.get("lock_contention_gate_sample_guard_fail_triggered"))
                if refresh.get("lock_contention_gate_sample_guard_fail_triggered") is not None
                else None
            ),
            "gate_sample_guard_fail_on_level": (
                str(refresh.get("lock_contention_gate_sample_guard_fail_on_level") or "").strip() or None
            ),
        }
        refresh_lock_json_path_raw = str(refresh.get("lock_contention_json_path") or "").strip()
        if refresh_lock_json_path_raw:
            lock_report_path = Path(refresh_lock_json_path_raw)

        refreshed_path_raw = str(refresh.get("report_path") or "").strip()
        if refreshed_path_raw:
            report_path = Path(refreshed_path_raw)
        else:
            report_path = resolve_paper_mode_readiness_path()
        payload = safe_read_json(report_path) if report_path else None

    if lock_report_path is None:
        lock_report_path = resolve_lock_contention_report_path()
    lock_payload = safe_read_json(lock_report_path) if lock_report_path else None
    lock_gate = evaluate_lock_contention_gate(
        payload=lock_payload,
        report_path=lock_report_path,
        enforce=lock_gate_enforce,
        fail_closed=lock_gate_fail_closed,
        max_age_hours=lock_gate_max_age_hours,
        max_allowed_level=lock_gate_max_allowed_level,
        required_source_bucket=lock_gate_required_source_bucket,
        require_bucket_match=lock_gate_require_bucket_match,
        require_sample_guard=lock_gate_require_sample_guard,
    )
    out["lock_contention_gate"] = lock_gate

    out["report_path"] = str(report_path) if report_path else None
    if report_path is None:
        if fail_closed:
            out["gate_blocked"] = True
            out["gate_reason"] = "readiness_report_missing_fail_closed"
        else:
            out["gate_reason"] = "readiness_report_missing_fail_open"
        return out

    if not isinstance(payload, dict):
        if fail_closed:
            out["gate_blocked"] = True
            out["gate_reason"] = "readiness_report_invalid_fail_closed"
        else:
            out["gate_reason"] = "readiness_report_invalid_fail_open"
        return out

    out["report_present"] = True
    status = str(payload.get("status") or "unknown").strip().lower()
    ready = payload.get("ready_for_paper_mode")
    ready_bool: Optional[bool]
    if isinstance(ready, bool):
        ready_bool = ready
    elif ready is None:
        ready_bool = None
    else:
        ready_bool = str(ready).strip().lower() in {"1", "true", "yes", "on"}

    out["status"] = status
    out["ready_for_paper_mode"] = ready_bool
    out["coverage"] = payload.get("coverage")
    out["missing_buckets"] = payload.get("missing_buckets")
    out["largest_missing_block_hours"] = payload.get("largest_missing_block_hours")
    warmup_gate = payload.get("warmup_gate")
    if isinstance(warmup_gate, dict):
        out["warmup_gate"] = warmup_gate
    elif allow_warmup:
        out["warmup_override_reason"] = "warmup_gate_missing"

    fail_reasons: List[str] = []
    raw_fail_reasons = payload.get("fail_reasons")
    if isinstance(raw_fail_reasons, list):
        for item in raw_fail_reasons[:20]:
            text = str(item or "").strip()
            if text:
                fail_reasons.append(text)
    out["fail_reasons"] = fail_reasons

    report_ts = parse_ts(payload.get("window_end_utc") or payload.get("report_ts"))
    if report_ts is not None:
        out["report_ts"] = report_ts.isoformat()
        age_hours = max(
            0.0,
            (dt.datetime.now(dt.timezone.utc) - report_ts).total_seconds() / 3600.0,
        )
        out["report_age_hours"] = round(age_hours, 4)
        out["report_stale"] = bool(max_age_hours > 0 and age_hours > max_age_hours)
    else:
        out["report_stale"] = None

    if bool(lock_gate.get("blocked")):
        out["gate_blocked"] = True
        out["gate_reason"] = f"lock_contention_gate({lock_gate.get('reason')})"
        if allow_warmup:
            out["warmup_override_reason"] = "lock_contention_gate_blocked"
        return out

    if out["report_stale"] is True:
        if fail_closed:
            out["gate_blocked"] = True
            out["gate_reason"] = "paper_mode_readiness_stale_fail_closed"
        else:
            out["gate_reason"] = "paper_mode_readiness_stale_fail_open"
        if allow_warmup:
            out["warmup_override_reason"] = "report_stale"
        return out

    warmup_ok = bool(isinstance(out.get("warmup_gate"), dict) and out["warmup_gate"].get("ok"))
    if status == "blocked" or ready_bool is False:
        if allow_warmup and warmup_ok:
            out["warmup_override_applied"] = True
            out["warmup_override_reason"] = "warmup_gate_ok"
            out["gate_reason"] = "paper_mode_warmup_ready"
            return out
        out["gate_blocked"] = True
        out["gate_reason"] = "paper_mode_readiness_blocked"
        if allow_warmup:
            out["warmup_override_reason"] = "warmup_gate_not_ready"
        return out

    if status == "ready" and ready_bool is True:
        out["gate_reason"] = "paper_mode_readiness_ready"
        if allow_warmup:
            out["warmup_override_reason"] = "not_needed"
        return out

    if fail_closed:
        out["gate_blocked"] = True
        out["gate_reason"] = "paper_mode_readiness_unknown_fail_closed"
    else:
        out["gate_reason"] = "paper_mode_readiness_unknown_fail_open"
    if allow_warmup:
        out["warmup_override_reason"] = "unknown_status"
    return out


@dataclass
class PaperState:
    date: str
    cash_usdt: float
    eth_qty: float
    avg_cost: float  # avg entry cost per ETH
    equity_peak: float
    daily_realized_pnl: float
    consecutive_losses: int
    last_loss_ts: Optional[str] = None


def load_consecutive_loss_ack(*, now_dt: dt.datetime, current_streak: int, stop_threshold: int) -> Dict[str, Any]:
    enabled = env_flag("LIE_PAPER_CONSECUTIVE_LOSS_ACK_ENABLED", default=True)
    out: Dict[str, Any] = {
        "enabled": enabled,
        "ack_path": str(PAPER_CONSECUTIVE_LOSS_ACK_PATH),
        "checksum_path": str(PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH),
        "present": False,
        "checksum_valid": None,
        "guardrail": None,
        "streak_snapshot": None,
        "use_limit": None,
        "uses_remaining": None,
        "active": None,
        "current_streak": int(current_streak),
        "stop_threshold": int(stop_threshold),
        "expires_at": None,
        "expired": None,
        "cooldown_hours_required": None,
        "cooldown_elapsed_hours": None,
        "allow_missing_last_loss_ts": None,
        "applied": False,
        "consume_attempted": False,
        "consume_ok": None,
        "consume_reason": None,
        "archive_attempted": False,
        "archive_ok": None,
        "archive_reason": None,
        "archive_path": None,
        "archive_checksum_path": None,
        "archive_sha256": None,
        "live_ack_cleared": None,
        "live_checksum_cleared": None,
        "reason": "disabled" if not enabled else "not_evaluated",
        "note": None,
    }
    if not enabled:
        return out
    if current_streak < stop_threshold:
        out["reason"] = "below_threshold"
        return out
    payload = safe_read_json(PAPER_CONSECUTIVE_LOSS_ACK_PATH)
    if not isinstance(payload, dict):
        out["reason"] = "ack_missing"
        return out
    out["present"] = True
    checksum_payload = safe_read_json(PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH)
    expected_sha = None
    if isinstance(checksum_payload, dict):
        expected_sha = str(checksum_payload.get("sha256") or "").strip() or None
    if expected_sha:
        try:
            out["checksum_valid"] = bool(_sha256_file(PAPER_CONSECUTIVE_LOSS_ACK_PATH) == expected_sha)
        except Exception:
            out["checksum_valid"] = False
        if out["checksum_valid"] is not True:
            out["reason"] = "checksum_invalid"
            return out
    else:
        out["checksum_valid"] = False
        out["reason"] = "checksum_missing"
        return out

    out["guardrail"] = str(payload.get("guardrail") or "").strip() or None
    out["streak_snapshot"] = int(payload.get("streak_snapshot") or 0)
    out["use_limit"] = int(payload.get("use_limit") or 1)
    out["uses_remaining"] = int(payload.get("uses_remaining") or 0)
    out["active"] = bool(payload.get("active", True))
    out["note"] = str(payload.get("note") or "").strip() or None
    out["allow_missing_last_loss_ts"] = bool(payload.get("allow_missing_last_loss_ts"))
    cooldown_required = max(0.0, float(payload.get("cooldown_hours_required") or 0.0))
    out["cooldown_hours_required"] = round(cooldown_required, 4)
    expires_at = parse_ts(payload.get("expires_at"))
    if expires_at is not None:
        out["expires_at"] = expires_at.isoformat()
        out["expired"] = bool(now_dt > expires_at)
    else:
        out["expired"] = True
    if out["guardrail"] != "consecutive_loss_stop":
        out["reason"] = "guardrail_mismatch"
        return out
    if out["active"] is not True:
        out["reason"] = "ack_inactive"
        return out
    if int(out["uses_remaining"] or 0) <= 0:
        out["reason"] = "ack_exhausted"
        return out
    if out["streak_snapshot"] != int(current_streak):
        out["reason"] = "streak_mismatch"
        return out
    if out["expired"] is not False:
        out["reason"] = "ack_expired"
        return out

    last_loss_ts = parse_ts(payload.get("last_loss_ts"))
    if last_loss_ts is not None:
        elapsed = max(0.0, (now_dt - last_loss_ts).total_seconds() / 3600.0)
        out["cooldown_elapsed_hours"] = round(elapsed, 4)
        if elapsed < cooldown_required:
            out["reason"] = "cooldown_active"
            return out
    elif not out["allow_missing_last_loss_ts"]:
        out["reason"] = "last_loss_ts_missing"
        return out

    out["applied"] = True
    out["reason"] = "ack_active"
    return out


def _write_consecutive_loss_ack_payload(payload: Dict[str, Any], *, now_dt: dt.datetime) -> bool:
    try:
        PAPER_CONSECUTIVE_LOSS_ACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ack_tmp = PAPER_CONSECUTIVE_LOSS_ACK_PATH.with_suffix(PAPER_CONSECUTIVE_LOSS_ACK_PATH.suffix + ".tmp")
        ack_tmp.write_text(raw, encoding="utf-8")
        ack_tmp.replace(PAPER_CONSECUTIVE_LOSS_ACK_PATH)
        checksum_payload = {
            "generated_at": now_dt.astimezone(dt.timezone.utc).isoformat(),
            "artifact": str(PAPER_CONSECUTIVE_LOSS_ACK_PATH),
            "sha256": _sha256_file(PAPER_CONSECUTIVE_LOSS_ACK_PATH),
        }
        checksum_tmp = PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH.with_suffix(
            PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH.suffix + ".tmp"
        )
        checksum_tmp.write_text(
            json.dumps(checksum_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_tmp.replace(PAPER_CONSECUTIVE_LOSS_ACK_CHECKSUM_PATH)
        return True
    except Exception:
        return False


def consume_consecutive_loss_ack(*, ack_state: Dict[str, Any], cycle_ts: str) -> Dict[str, Any]:
    if not bool(ack_state.get("applied")):
        return ack_state
    ack_state["consume_attempted"] = True
    payload = safe_read_json(PAPER_CONSECUTIVE_LOSS_ACK_PATH)
    if not isinstance(payload, dict):
        ack_state["consume_ok"] = False
        ack_state["consume_reason"] = "ack_missing_at_consume"
        return ack_state
    uses_remaining = max(0, int(payload.get("uses_remaining") or 0) - 1)
    payload["uses_remaining"] = uses_remaining
    payload["active"] = bool(uses_remaining > 0)
    payload["consumed_at"] = cycle_ts
    payload["consume_reason"] = "single_use_consumed"
    now_dt = dt.datetime.now(dt.timezone.utc)
    ok = _write_consecutive_loss_ack_payload(payload, now_dt=now_dt)
    ack_state["consume_ok"] = bool(ok)
    ack_state["archive_attempted"] = bool(ok)
    if ok:
        archive_state = _archive_and_clear_consecutive_loss_ack_payload(payload, now_dt=now_dt, cycle_ts=cycle_ts)
        ack_state["archive_ok"] = bool(archive_state.get("archive_ok"))
        ack_state["archive_reason"] = archive_state.get("archive_reason")
        ack_state["archive_path"] = archive_state.get("archive_path")
        ack_state["archive_checksum_path"] = archive_state.get("archive_checksum_path")
        ack_state["archive_sha256"] = archive_state.get("archive_sha256")
        ack_state["live_ack_cleared"] = archive_state.get("live_ack_cleared")
        ack_state["live_checksum_cleared"] = archive_state.get("live_checksum_cleared")
        if bool(archive_state.get("archive_ok")):
            ack_state["consume_reason"] = (
                "single_use_consumed_archived_and_cleared"
                if bool(archive_state.get("live_ack_cleared")) and bool(archive_state.get("live_checksum_cleared"))
                else "single_use_consumed_archived_clear_partial"
            )
        else:
            ack_state["consume_reason"] = "single_use_consumed_inactive_only"
    else:
        ack_state["archive_ok"] = False
        ack_state["archive_reason"] = "inactive_write_failed"
        ack_state["consume_reason"] = "consume_write_failed"
    ack_state["uses_remaining_after_consume"] = uses_remaining
    ack_state["active_after_consume"] = bool(uses_remaining > 0)
    return ack_state


def load_paper_state(date: str, px: float) -> PaperState:
    PAPER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if PAPER_STATE_PATH.exists():
        try:
            raw = json.loads(PAPER_STATE_PATH.read_text(encoding="utf-8"))
            st = PaperState(
                date=str(raw.get("date") or date),
                cash_usdt=float(raw.get("cash_usdt") or 0.0),
                eth_qty=float(raw.get("eth_qty") or 0.0),
                avg_cost=float(raw.get("avg_cost") or 0.0),
                equity_peak=float(raw.get("equity_peak") or 0.0),
                daily_realized_pnl=float(raw.get("daily_realized_pnl") or 0.0),
                consecutive_losses=int(raw.get("consecutive_losses") or 0),
                last_loss_ts=(str(raw.get("last_loss_ts") or "").strip() or None),
            )
        except Exception:
            st = PaperState(date=date, cash_usdt=PAPER_INIT_USDT, eth_qty=0.0, avg_cost=0.0, equity_peak=0.0, daily_realized_pnl=0.0, consecutive_losses=0, last_loss_ts=None)
    else:
        st = PaperState(date=date, cash_usdt=PAPER_INIT_USDT, eth_qty=0.0, avg_cost=0.0, equity_peak=0.0, daily_realized_pnl=0.0, consecutive_losses=0, last_loss_ts=None)

    # reset daily stats on date change, but keep consecutive losses
    if st.date != date:
        st.date = date
        st.daily_realized_pnl = 0.0
        st.equity_peak = 0.0

    equity = st.cash_usdt + st.eth_qty * px
    if st.equity_peak <= 0:
        st.equity_peak = equity
    else:
        st.equity_peak = max(st.equity_peak, equity)

    return st


def save_paper_state(st: PaperState) -> None:
    PAPER_STATE_PATH.write_text(
        json.dumps(
            {
                "date": st.date,
                "cash_usdt": round(st.cash_usdt, 8),
                "eth_qty": round(st.eth_qty, 8),
                "avg_cost": round(st.avg_cost, 8),
                "equity_peak": round(st.equity_peak, 8),
                "daily_realized_pnl": round(st.daily_realized_pnl, 8),
                "consecutive_losses": int(st.consecutive_losses),
                "last_loss_ts": st.last_loss_ts,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _normalize_broker_symbol(raw: Any) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    text = "".join(text.split())
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch for ch in text if ch in allowed)


def _build_open_paper_positions_rows(st: PaperState, px: float, symbol: str) -> List[Dict[str, Any]]:
    sym = _normalize_broker_symbol(symbol)
    qty_signed = float(st.eth_qty)
    qty = abs(qty_signed)
    if (not sym) or qty <= 1e-12:
        return []

    side = "SHORT" if qty_signed < 0 else "LONG"
    mark_px = float(px if px > 0 else (st.avg_cost if st.avg_cost > 0 else 0.0))
    entry_px = float(st.avg_cost if st.avg_cost > 0 else mark_px)
    notional = float(qty * max(0.0, mark_px))
    equity = float(st.cash_usdt + st.eth_qty * max(0.0, mark_px))
    if equity <= 1e-12:
        size_pct = 0.0
    else:
        size_pct = max(0.0, min(100.0, (notional / equity) * 100.0))
    try:
        risk_pct = max(
            0.0,
            float(os.getenv("LIE_PAPER_POSITION_DEFAULT_RISK_PCT", "1.0")),
        )
    except Exception:
        risk_pct = 1.0
    try:
        hold_days = max(1, int(float(os.getenv("LIE_PAPER_POSITION_DEFAULT_HOLD_DAYS", "1"))))
    except Exception:
        hold_days = 1

    if side == "SHORT":
        stop_px = float(entry_px * 1.02) if entry_px > 0 else 0.0
        target_px = float(entry_px * 0.98) if entry_px > 0 else 0.0
    else:
        stop_px = float(entry_px * 0.98) if entry_px > 0 else 0.0
        target_px = float(entry_px * 1.02) if entry_px > 0 else 0.0

    return [
        {
            "open_date": str(st.date),
            "symbol": sym,
            "side": side,
            "size_pct": round(float(size_pct), 8),
            "risk_pct": round(float(risk_pct), 8),
            "entry_price": round(float(entry_px), 8),
            "stop_price": round(float(stop_px), 8),
            "target_price": round(float(target_px), 8),
            "runtime_mode": "paper_sync",
            "hold_days": int(hold_days),
            "status": "OPEN",
            "qty": round(float(qty), 8),
            "notional": round(float(notional), 8),
            "market_price": round(float(mark_px), 8),
        }
    ]


def sync_paper_execution_artifacts(*, as_of: str, st: PaperState, px: float, symbol: str) -> Dict[str, Any]:
    def _parse_iso_day(raw: Any) -> Optional[dt.date]:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return dt.date.fromisoformat(text)
        except Exception:
            return None

    day = str(as_of or st.date or dt.datetime.now(dt.timezone.utc).date().isoformat()).strip()
    if not day:
        day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    target_day = _parse_iso_day(day)
    allow_stale_write = env_flag("LIE_PAPER_POSITIONS_ALLOW_STALE_WRITE", default=False)
    lock_timeout_sec = max(0.0, env_float("LIE_PAPER_ARTIFACTS_LOCK_TIMEOUT_SEC", 2.0))
    lock_retry_sec = max(0.01, env_float("LIE_PAPER_ARTIFACTS_LOCK_RETRY_SEC", 0.05))

    out: Dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "reason": "unknown",
        "as_of": day,
        "paper_positions_path": str(PAPER_POSITIONS_OPEN_PATH),
        "broker_snapshot_path": str(BROKER_SNAPSHOT_DIR / f"{day}.json"),
        "paper_positions_written": False,
        "broker_snapshot_written": False,
        "position_rows": 0,
        "lock_path": str(PAPER_ARTIFACTS_LOCK_PATH),
        "lock_acquired": None,
        "lock_wait_sec": None,
        "lock_timeout_sec": float(lock_timeout_sec),
        "lock_retry_sec": float(lock_retry_sec),
        "lock_reason": None,
        "stale_guard_blocked": False,
        "existing_as_of": None,
        "target_as_of": day,
        "allow_stale_write": bool(allow_stale_write),
    }

    positions = _build_open_paper_positions_rows(st=st, px=float(px), symbol=symbol)
    paper_payload = {
        "as_of": day,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "lie_spot_core",
        "positions": positions,
    }
    broker_payload = {
        "date": day,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "paper_engine",
        "open_positions": int(len(positions)),
        "closed_count": 0,
        "closed_pnl": float(round(st.daily_realized_pnl, 8)),
        "positions": [
            {
                "symbol": str(row.get("symbol") or ""),
                "side": str(row.get("side") or ""),
                "qty": float(row.get("qty") or 0.0),
                "notional": float(row.get("notional") or 0.0),
                "entry_price": float(row.get("entry_price") or 0.0),
                "market_price": float(row.get("market_price") or 0.0),
                "status": str(row.get("status") or "OPEN"),
                "open_date": str(row.get("open_date") or day),
                "size_pct": float(row.get("size_pct") or 0.0),
                "risk_pct": float(row.get("risk_pct") or 0.0),
                "runtime_mode": str(row.get("runtime_mode") or "paper_sync"),
            }
            for row in positions
            if isinstance(row, dict)
        ],
        "stats": {
            "paper_state_path": str(PAPER_STATE_PATH),
            "paper_positions_path": str(PAPER_POSITIONS_OPEN_PATH),
            "position_rows": int(len(positions)),
        },
    }

    try:
        PAPER_POSITIONS_OPEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        BROKER_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        PAPER_ARTIFACTS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        out["reason"] = f"prepare_dirs_failed:{type(exc).__name__}"
        return out

    lockf = None
    try:
        import fcntl  # type: ignore

        lockf = PAPER_ARTIFACTS_LOCK_PATH.open("w")
        start_ts = pytime.monotonic()
        deadline = start_ts + max(0.0, float(lock_timeout_sec))
        while True:
            try:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                out["lock_acquired"] = True
                out["lock_wait_sec"] = round(max(0.0, pytime.monotonic() - start_ts), 6)
                out["lock_reason"] = "lock_acquired"
                break
            except BlockingIOError:
                if pytime.monotonic() >= deadline:
                    out["lock_acquired"] = False
                    out["lock_wait_sec"] = round(max(0.0, pytime.monotonic() - start_ts), 6)
                    out["lock_reason"] = "lock_timeout"
                    out["reason"] = "lock_timeout"
                    return out
                pytime.sleep(lock_retry_sec)
            except Exception:
                out["lock_acquired"] = False
                out["lock_reason"] = "lock_unavailable"
                out["reason"] = "artifacts_lock_unavailable"
                return out
    except Exception:
        out["lock_acquired"] = False
        out["lock_reason"] = "lock_unavailable"
        out["reason"] = "artifacts_lock_unavailable"
        return out

    try:
        existing_payload = safe_read_json(PAPER_POSITIONS_OPEN_PATH)
        existing_as_of_raw = (
            str(existing_payload.get("as_of") or "").strip()
            if isinstance(existing_payload, dict)
            else ""
        )
        if existing_as_of_raw:
            out["existing_as_of"] = existing_as_of_raw
        existing_day = _parse_iso_day(existing_as_of_raw)
        if (
            (not allow_stale_write)
            and (existing_day is not None)
            and (target_day is not None)
            and (existing_day > target_day)
        ):
            out["stale_guard_blocked"] = True
            out["reason"] = "stale_write_guard_blocked"
            return out

        paper_ok = safe_write_json_atomic(PAPER_POSITIONS_OPEN_PATH, paper_payload)
        broker_ok = safe_write_json_atomic(BROKER_SNAPSHOT_DIR / f"{day}.json", broker_payload)
        out["paper_positions_written"] = bool(paper_ok)
        out["broker_snapshot_written"] = bool(broker_ok)
        out["position_rows"] = int(len(positions))

        if paper_ok and broker_ok:
            out["ok"] = True
            out["reason"] = "synced"
        elif (not paper_ok) and (not broker_ok):
            out["reason"] = "paper_and_broker_write_failed"
        elif not paper_ok:
            out["reason"] = "paper_positions_write_failed"
        else:
            out["reason"] = "broker_snapshot_write_failed"
        return out
    finally:
        if lockf is not None:
            try:
                lockf.close()
            except Exception:
                pass


def _micro_probe_paper_fill(
    args: Tuple[Any, ...], kwargs: Dict[str, Any], gate: Any
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    del gate
    ratio = float(os.getenv("CORTEX_MICRO_PROBE_QTY_RATIO", "0.20"))
    min_qty = float(os.getenv("CORTEX_MICRO_PROBE_MIN_QTY", "0.0001"))
    new_args = list(args)
    new_kwargs = dict(kwargs)

    if "qty" in new_kwargs:
        qty = float(new_kwargs.get("qty") or 0.0)
        if qty > 0:
            new_kwargs["qty"] = max(min_qty, qty * ratio)
    elif len(new_args) >= 3:
        qty = float(new_args[2] or 0.0)
        if qty > 0:
            new_args[2] = max(min_qty, qty * ratio)

    return tuple(new_args), new_kwargs


def _simulate_paper_fill(mode: str, reason: str, debug: Dict[str, Any]) -> Tuple[float, float]:
    del mode, reason, debug
    return 0.0, 0.0


def _paper_fill_cap_probe(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, float]:
    qty = float(kwargs.get("qty") or (args[2] if len(args) > 2 else 0.0))
    px = float(kwargs.get("px") or (args[3] if len(args) > 3 else 0.0))
    return {"qty": abs(qty), "notional": abs(qty * px)}


def _paper_fill_reduce_only_probe(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    side = str(kwargs.get("side") or (args[1] if len(args) > 1 else "")).upper()
    return side == "SELL"


def _dynamic_paper_cap_limits(gate: Any) -> Dict[str, float]:
    mode = str(getattr(gate, "mode", "") or "")
    debug = getattr(gate, "debug", {}) or {}
    state = debug.get("state", {}) if isinstance(debug, dict) else {}
    try:
        i = float(state.get("I", 1.0))
        r = float(state.get("R", 1.0))
        o = float(state.get("O", 1.0))
    except Exception:
        i, r, o = 1.0, 1.0, 1.0
    core = min(i, r, o)

    if mode == "ACT":
        if core >= 0.85:
            factor = 1.0
        elif core >= 0.70:
            factor = 0.60
        elif core >= 0.55:
            factor = 0.35
        else:
            factor = 0.15
    elif mode == "LEARN":
        factor = 0.10
    elif mode == "OBSERVE":
        factor = 0.05
    else:
        factor = 0.02

    mult = float(os.getenv("CORTEX_PAPER_CAP_MULTIPLIER", os.getenv("CORTEX_CAP_MULTIPLIER", "1.0")))
    factor = max(0.02, min(1.0, factor * mult))
    return {"qty": CORTEX_MAX_PAPER_QTY * factor, "notional": CORTEX_MAX_PAPER_NOTIONAL * factor}


@cortex_gated(
    action_class="NORMAL_ACT",
    on_observe="micro_probe",
    on_learn="simulate",
    on_stabilize="reject",
    mutate_micro_probe=_micro_probe_paper_fill,
    simulate_result=_simulate_paper_fill,
    cap_probe=_paper_fill_cap_probe,
    cap_limits=_dynamic_paper_cap_limits,
    on_cap_violation="reject",
    reflex_reduce_only_probe=_paper_fill_reduce_only_probe,
)
def apply_paper_fill(st: PaperState, side: str, qty: float, px: float) -> Tuple[float, float]:
    """Apply a paper fill and return (realized_pnl, new_equity)."""
    realized = 0.0
    notional = qty * px

    if side == "BUY":
        if notional > st.cash_usdt:
            # clamp to cash
            qty = st.cash_usdt / px if px > 0 else 0.0
            notional = qty * px
        if qty <= 0:
            return 0.0, st.cash_usdt + st.eth_qty * px

        # update avg cost
        new_qty = st.eth_qty + qty
        if st.eth_qty <= 0:
            st.avg_cost = px
        else:
            st.avg_cost = (st.avg_cost * st.eth_qty + px * qty) / new_qty
        st.eth_qty = new_qty
        st.cash_usdt -= notional

    elif side == "SELL":
        qty = min(qty, st.eth_qty)
        if qty <= 0:
            return 0.0, st.cash_usdt + st.eth_qty * px

        realized = (px - st.avg_cost) * qty
        st.daily_realized_pnl += realized

        st.eth_qty -= qty
        st.cash_usdt += qty * px

        # if fully flat, reset avg cost
        if st.eth_qty <= 1e-12:
            st.eth_qty = 0.0
            st.avg_cost = 0.0

        # update consecutive loss streak only on sells that realize pnl
        if realized < 0:
            st.consecutive_losses += 1
            st.last_loss_ts = dt.datetime.now(dt.timezone.utc).isoformat()
        else:
            st.consecutive_losses = 0
            st.last_loss_ts = None

    else:
        raise ValueError(side)

    equity = st.cash_usdt + st.eth_qty * px
    st.equity_peak = max(st.equity_peak, equity)
    return realized, equity


def _quantize_down(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.floor(float(x) / float(step)) * float(step)


def _build_local_paper_order_plan(
    action: str,
    px: float,
    st: PaperState,
    *,
    mode_hint: Optional[str] = None,
    debug_hint: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if px <= 0:
        return None

    step_qty = max(1e-8, env_float("LIE_LOCAL_PAPER_STEP_QTY", 0.0001))
    min_qty = max(step_qty, env_float("LIE_LOCAL_PAPER_MIN_QTY", step_qty))

    if action == "VOLATILITY_PUMP":
        equity = float(st.cash_usdt) + float(st.eth_qty) * float(px)
        if equity <= 0:
            return None
            
        target_cash = equity * 0.50
        target_eth_val = equity * 0.50
        
        current_eth_val = float(st.eth_qty) * float(px)
        current_cash = float(st.cash_usdt)
        
        # 5% deviation threshold (0.05)
        deviation_threshold = env_float("LIE_VOLATILITY_PUMP_THRESHOLD", 0.05)
        
        drift_pct = abs(current_eth_val - target_eth_val) / equity
        if drift_pct < deviation_threshold:
            return None # Inside the deadband, no trade needed
            
        if current_eth_val < target_eth_val:
            # We need to BUY ETH to restore 50%
            shortfall = target_eth_val - current_eth_val
            cap_notional = min(shortfall, current_cash * 0.98) # max 98% of cash to leave dust
            qty = cap_notional / float(px)
            qty = _quantize_down(qty, step_qty)
            if qty < min_qty:
                return None
            notional = qty * float(px)
            return {
                "side": "BUY",
                "type": "MARKET",
                "qty": float(f"{qty:.8f}"),
                "notional": round(float(notional), 6),
                "reason": f"volatility_pump_buy_drift_{drift_pct*100:.1f}pct",
            }
        else:
            # We need to SELL ETH to restore 50%
            excess = current_eth_val - target_eth_val
            qty = excess / float(px)
            qty = min(qty, float(st.eth_qty))
            qty = _quantize_down(qty, step_qty)
            if qty < min_qty:
                return None
            notional = qty * float(px)
            return {
                "side": "SELL",
                "type": "MARKET",
                "qty": float(f"{qty:.8f}"),
                "notional": round(float(notional), 6),
                "reason": f"volatility_pump_sell_drift_{drift_pct*100:.1f}pct",
            }
            
    if action in {"LIQUIDITY_TRAP_BUY", "LIQUIDITY_TRAP_SELL"}:
        side = "BUY" if action == "LIQUIDITY_TRAP_BUY" else "SELL"
        # Devour 20% chunks of available equity per vacuum trigger
        if side == "BUY":
            cap_notional = float(st.cash_usdt) * 0.20
            qty = cap_notional / float(px)
        else:
            qty = float(st.eth_qty) * 0.20
            
        qty = _quantize_down(qty, step_qty)
        if qty < min_qty:
            return None
            
        notional = qty * float(px)
        return {
            "side": side,
            "type": "MARKET",
            "qty": float(f"{qty:.8f}"),
            "notional": round(float(notional), 6),
            "reason": f"liquidity_vacuum_devour_{side.lower()}",
        }

    # --- Legacy Fallback for direct agent commands ---
    dynamic_limits = {"qty": CORTEX_MAX_PAPER_QTY, "notional": CORTEX_MAX_PAPER_NOTIONAL}
    max_notional = max(
        0.0,
        env_float(
            "LIE_LOCAL_PAPER_MAX_NOTIONAL",
            float(dynamic_limits.get("notional") or CORTEX_MAX_PAPER_NOTIONAL),
        ),
    )
    max_qty = max(
        0.0,
        env_float(
            "LIE_LOCAL_PAPER_MAX_QTY",
            float(dynamic_limits.get("qty") or CORTEX_MAX_PAPER_QTY),
        ),
    )
    buy_cash_ratio = _clamp(env_float("LIE_LOCAL_PAPER_BUY_CASH_RATIO", 0.10), 0.01, 1.0)

    if action == "LONG_ETH":
        cap_notional = min(max_notional, max(0.0, float(st.cash_usdt)) * buy_cash_ratio)
        cap_notional = max(0.0, cap_notional * 0.98)
        qty = min(max_qty, cap_notional / float(px))
        qty = _quantize_down(qty, step_qty)
        if qty < min_qty:
            return None
        notional = qty * float(px)
        if notional <= 0:
            return None
        return {
            "side": "BUY",
            "type": "MARKET",
            "qty": float(f"{qty:.8f}"),
            "notional": round(float(notional), 6),
            "reason": "local_paper_router_long",
        }

    if action == "FLAT_USDT":
        held_qty = max(0.0, float(st.eth_qty))
        if held_qty <= 0:
            return None
        cap_notional = min(max_notional, held_qty * float(px))
        qty = min(max_qty, held_qty, cap_notional / float(px))
        qty = _quantize_down(qty, step_qty)
        if qty < min_qty:
            return None
        notional = qty * float(px)
        if notional <= 0:
            return None
        return {
            "side": "SELL",
            "type": "MARKET",
            "qty": float(f"{qty:.8f}"),
            "notional": round(float(notional), 6),
            "reason": "local_paper_router_flat",
        }

    return None


def _should_route_local_paper_fill(
    *,
    decision: str,
    exec_out: Dict[str, Any],
    guardrail_hit: bool,
    live_order_requested: bool,
) -> bool:
    if not env_flag("LIE_LOCAL_PAPER_ROUTER_ENABLED", default=True):
        return False
    if guardrail_hit:
        return False
    if live_order_requested:
        return False
    if str(decision or "").strip().lower() != "no-trade":
        return False
    reason = str((exec_out or {}).get("reason") or "").strip().lower()
    return reason == "dry_run_no_private_endpoints"


def _effective_paper_fill_price(side: str, mark_px: float) -> Tuple[float, float]:
    base_px = float(mark_px)
    if base_px <= 0:
        return 0.0, 0.0
    slip_bps = env_float("LIE_PAPER_FILL_SLIPPAGE_BPS", 0.0)
    if abs(slip_bps) <= 1e-12:
        return base_px, 0.0
    side_u = str(side or "").upper()
    signed_bps = abs(float(slip_bps)) if side_u == "BUY" else -abs(float(slip_bps))
    fill_px = base_px * (1.0 + signed_bps / 10000.0)
    return float(fill_px), float(signed_bps)


def decide_action(daily_signals: Any, mode_feedback: Any, paper_state: PaperState) -> Tuple[str, Dict[str, Any]]:
    """Return (action, diagnostics)."""
    diag: Dict[str, Any] = {}

    passed = True
    try:
        passed = bool(mode_feedback.get("mode_health", {}).get("passed", True))
    except Exception:
        passed = True

    # --- A.0) Apply Bayesian Pain ---
    # Tighten the gates structurally if we are bleeding in paper state
    bayesian_penalty = 0.0
    if paper_state.consecutive_losses > 0:
        penalty_step = min(3, paper_state.consecutive_losses)
        bayesian_penalty = penalty_step * 4.0

    # Directionless Convexity Mandate: We completely ignore directional signals (LONG/SHORT) 
    # and instead continuously request a "VOLATILITY_PUMP" mechanical rebalance check
    # as long as the Cortex Gate (mode_feedback) states we are healthy enough to ACT.
    
    hmm_probs = mode_feedback.get("regime", {}).get("hmm_probs", {})
    vacuum_trigger = hmm_probs.get("liquidity_vacuum_trigger", 0.0) == 1.0
    vacuum_side = hmm_probs.get("liquidity_vacuum_side", 0.0)
    
    if passed and vacuum_trigger:
        action = "LIQUIDITY_TRAP_BUY" if vacuum_side == 1.0 else "LIQUIDITY_TRAP_SELL"
        convexity_mode = "liquidity_vacuum_devour"
    else:
        action = "VOLATILITY_PUMP" if passed else "FLAT_USDT"
        convexity_mode = "volatility_pumping_50_50"

    diag.update(
        {
            "mode_passed": passed,
            "bayesian_penalty_applied": bayesian_penalty,
            "convexity_mode": convexity_mode,
            "predictive_signals_ignored": True
        }
    )
    return action, diag


def append_state(event: Dict[str, Any], lines: List[str]) -> None:
    STATE_MD.parent.mkdir(parents=True, exist_ok=True)
    with STATE_MD.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("LIE_DRYRUN_EVENT=" + json.dumps(event, ensure_ascii=False) + "\n")
        for ln in lines[:3]:
            f.write(str(ln).rstrip("\n") + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--paper-readiness-path",
        type=str,
        default="",
        help="Override readiness report path (JSON).",
    )
    args = parser.parse_args()
    if args.paper_readiness_path:
        os.environ["LIE_PAPER_MODE_READINESS_PATH"] = str(args.paper_readiness_path).strip()

    reset_proxy_bypass_stats()
    OC_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    PULSE_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # coarse lock (avoid overlapping cron runs)
    try:
        import fcntl  # type: ignore

        lockf = LOCK_PATH.open("w")
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        # If lock cannot be acquired, just exit quietly.
        return 0

    ts = now_local_iso()
    event_source = event_source_label()
    date_str = ts[:10]
    paper_mode_readiness = load_paper_mode_readiness_gate()
    paper_mode_gate_blocked = bool(paper_mode_readiness.get("gate_blocked"))
    paper_mode_gate_reason = str(
        paper_mode_readiness.get("gate_reason") or "paper_mode_readiness_unknown"
    ).strip()
    pulse_lockf = None
    pulse_lock_wait_start = pytime.monotonic()
    pulse_lock_wait_sec: Optional[float] = None
    pulse_lock_hold_sec: Optional[float] = None
    pulse_lock_acquired = False
    pulse_lock_acquired_ts: Optional[float] = None
    pulse_lock_reason = "lock_unknown"
    try:
        import fcntl  # type: ignore

        pulse_lockf = PULSE_LOCK_PATH.open("w")
        fcntl.flock(pulse_lockf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        pulse_lock_wait_sec = round(max(0.0, pytime.monotonic() - pulse_lock_wait_start), 6)
        pulse_lock_acquired = True
        pulse_lock_acquired_ts = pytime.monotonic()
        pulse_lock_reason = "lock_acquired"
    except Exception:
        pulse_lock_wait_sec = round(max(0.0, pytime.monotonic() - pulse_lock_wait_start), 6)
        pulse_lock_reason = "lock_busy_or_unavailable"
        event = {
            "domain": "lie_spot_core",
            "ts": ts,
            "event_source": event_source,
            "bucket": "",
            "symbol": BINANCE_SYMBOL,
            "action": "FLAT_USDT",
            "decision": "no-trade",
            "reason": "halfhour_pulse_lock_busy",
            "order_result": {"http": 0, "endpoint": "skipped_pulse_lock"},
            "lie": {
                "daemon": {
                    "mode": "lock_guard",
                    "command": "run-daemon",
                    "would_run_pulse": False,
                    "pulse_reason": "pulse_lock_busy",
                    "due_slots": [],
                },
                "pulse": {"ran": False, "skipped": True, "reason": "pulse_lock_busy"},
            },
            "guardrails": {
                "max_daily_drawdown_usdt": MAX_DAILY_DRAWDOWN_USDT,
                "consecutive_loss_stop": CONSECUTIVE_LOSS_STOP,
                "hit": True,
                "reasons": ["halfhour_pulse_lock_busy"],
            },
            "paper_mode_readiness": paper_mode_readiness,
            "net_proxy_bypass": get_proxy_bypass_stats(),
            "pulse_lock": {
                "path": str(PULSE_LOCK_PATH),
                "acquired": False,
                "reason": pulse_lock_reason,
                "wait_sec": pulse_lock_wait_sec,
                "hold_sec": None,
            },
            "paper_artifacts_sync": {
                "attempted": False,
                "ok": None,
                "reason": "skipped_due_to_pulse_lock_busy",
                "event_source": event_source,
                "paper_positions_path": str(PAPER_POSITIONS_OPEN_PATH),
                "broker_snapshot_path": str(
                    BROKER_SNAPSHOT_DIR / f"{date_str}.json"
                ),
                "lock_path": str(PAPER_ARTIFACTS_LOCK_PATH),
                "lock_acquired": None,
                "lock_wait_sec": None,
                "lock_timeout_sec": max(0.0, env_float("LIE_PAPER_ARTIFACTS_LOCK_TIMEOUT_SEC", 2.0)),
                "lock_retry_sec": max(0.01, env_float("LIE_PAPER_ARTIFACTS_LOCK_RETRY_SEC", 0.05)),
                "lock_reason": "not_attempted",
                "stale_guard_blocked": False,
                "existing_as_of": None,
                "target_as_of": date_str,
            },
        }
        append_state(
            event,
            [
                "signal=FLAT_USDT bucket=none date=none px_source=none",
                "reason: pulse_lock_busy; skip daemon/pulse to prevent replay thrash",
                "dry-run skipped (pulse lock busy); decision=no-trade",
            ],
        )
        print(json.dumps(event, ensure_ascii=False))
        return 0

    # --- A.0) Inject Bayesian Pain (Anti-Fragile Feedback Loop) ---
    micro_feat = 0.0
    sent_feat = 0.0
    onchain_feat = 0.0
    sent_source = "fallback_zero"
    onchain_source = "fallback_zero"
    hmm_proxy: Dict[str, float] = {
        "entropy": 0.0,
        "log_likelihood": -0.3,
        "distribution_shift": 0.0,
    }
    try:
        px, _, _ = spot_price(BINANCE_SYMBOL)
        st = load_paper_state(date=date_str, px=px)
        
        # --- PHASE 11: Anti-Boiling Frog Ratchet ---
        # Never allow the system to relax its safety constraints. 
        # Mathematically lock the floor so parameters only ever drift upwards (stricter).
        artifacts_dir = LIE_ROOT / "output" / "artifacts"
        params_live_file = artifacts_dir / "params_live.yaml"
        
        existing_conf_min = 60.0
        existing_conv_min = 3.0
        if params_live_file.exists():
            try:
                import yaml
                with open(params_live_file, "r") as f:
                    ex_params = yaml.safe_load(f) or {}
                existing_conf_min = float(ex_params.get("signal_confidence_min", 60.0))
                existing_conv_min = float(ex_params.get("convexity_min", 3.0))
            except Exception:
                pass

        # Bayesian shrink: if we bleed, we tighten the gates
        # default base params
        base_conf_min = 60.0
        base_conv_min = 3.0
        
        if st.consecutive_losses > 0:
            # max penalty capped at 3 losses to avoid locking completely
            penalty = min(3, st.consecutive_losses)
            base_conf_min += penalty * 4.0   
            base_conv_min += penalty * 0.4   
            
        # The Ratchet: The engine may ONLY tighten thresholds, never loosen them.
        conf_min = max(existing_conf_min, base_conf_min)
        conv_min = max(existing_conv_min, base_conv_min)
            
        params_yaml = (
            f"signal_confidence_min: {conf_min:.2f}\n"
            f"convexity_min: {conv_min:.2f}\n"
            "win_rate: 0.45\n"
            "payoff: 2.0\n"
            "hold_days: 5.0\n"
            "max_daily_trades: 2.0\n"
        )
        artifacts_dir = LIE_ROOT / "output" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / "params_live.yaml").write_text(params_yaml, encoding="utf-8")
        
        # --- A.1) Inject Multi-Modal Features for HMM Upgrades ---
        micro_feat = get_micro_imbalance(BINANCE_SYMBOL)
        sent_feat, sent_source = get_semantic_sentiment_with_source()
        onchain_feat, onchain_source = get_onchain_proxy_with_source()
        
        prev_mm = _load_prev_multimodal(artifacts_dir / "live_multimodal.json")
        hmm_proxy = build_hmm_proxy_features(
            micro_imbalance=micro_feat,
            sentiment_pca=sent_feat,
            onchain_proxy=onchain_feat,
            prev=prev_mm,
        )

        multimodal = {
            "micro_imbalance": micro_feat,
            "sentiment_pca": sent_feat,
            "sentiment_source": sent_source,
            "onchain_proxy": onchain_feat,
            "onchain_source": onchain_source,
            "entropy": hmm_proxy["entropy"],
            "log_likelihood": hmm_proxy["log_likelihood"],
            "distribution_shift": hmm_proxy["distribution_shift"],
            "hmm_proxy_version": "v1",
            "ts": ts,
        }
        (artifacts_dir / "live_multimodal.json").write_text(json.dumps(multimodal, ensure_ascii=False), encoding="utf-8")
        
    except Exception:
        # Graceful degradation, let the engine fallback to internal defaults
        pass

    # --- A) LiE core loop ---
    daemon_mode = "legacy_halfhour"
    daemon_cmd = "run-halfhour-daemon"
    try:
        daemon = run_json(
            ["python3", "-m", "lie_engine.cli", "--config", "config.yaml", "run-halfhour-daemon", "--dry-run"],
            cwd=LIE_ROOT,
            env_extra={"PYTHONPATH": "src"},
            timeout_s=240,
        )
    except RuntimeError as exc:
        if not _is_invalid_cli_choice(exc, "run-halfhour-daemon"):
            raise
        daemon_mode = "compat_scheduler"
        daemon_cmd = "run-daemon"
        daemon = run_json(
            [
                "python3",
                "-m",
                "lie_engine.cli",
                "--config",
                "config.yaml",
                "run-daemon",
                "--dry-run",
                "--max-cycles",
                "1",
                "--poll-seconds",
                "1",
            ],
            cwd=LIE_ROOT,
            env_extra={"PYTHONPATH": "src"},
            timeout_s=240,
        )

    date = str(daemon.get("date") or "")
    due_slots = [str(x) for x in (daemon.get("would_execute") or []) if str(x).strip()]
    bucket = str(
        daemon.get("current_bucket")
        or daemon.get("pulse_bucket")
        or (due_slots[0] if due_slots else "")
    )

    pulse_ran = False
    pulse: Optional[Dict[str, Any]] = None
    pulse_reason = ""
    try:
        pulse_reason = str((daemon.get("pulse_preview") or {}).get("reason") or "")
    except Exception:
        pulse_reason = ""

    would_run_pulse = bool(daemon.get("would_run_pulse"))
    if (not would_run_pulse) and due_slots:
        would_run_pulse = True
        if not pulse_reason:
            pulse_reason = "run_slot_due"

    if would_run_pulse and (pulse_reason != "same_bucket"):
        if daemon_mode == "legacy_halfhour":
            try:
                pulse = run_json(
                    ["python3", "-m", "lie_engine.cli", "--config", "config.yaml", "run-halfhour-pulse"],
                    cwd=LIE_ROOT,
                    env_extra={"PYTHONPATH": "src"},
                    timeout_s=240,
                )
                pulse_ran = True
            except RuntimeError as exc:
                if not _is_invalid_cli_choice(exc, "run-halfhour-pulse"):
                    raise
                daemon_mode = "compat_scheduler"

        if (not pulse_ran) and date and due_slots:
            slot_arg = _slot_arg_from_slot_id(due_slots[0])
            pulse = run_json(
                [
                    "python3",
                    "-m",
                    "lie_engine.cli",
                    "--config",
                    "config.yaml",
                    "run-slot",
                    "--date",
                    date,
                    "--slot",
                    slot_arg,
                ],
                cwd=LIE_ROOT,
                env_extra={"PYTHONPATH": "src"},
                timeout_s=240,
            )
            pulse_ran = True
            if not pulse_reason:
                pulse_reason = "run_slot_compat"

    health = run_json(
        ["python3", "-m", "lie_engine.cli", "--config", "config.yaml", "health-check"],
        cwd=LIE_ROOT,
        env_extra={"PYTHONPATH": "src"},
        timeout_s=240,
    )

    # --- B) Spot signal (binary) ---
    daily_signals_path = latest_daily_path("_signals.json", prefer_date=date_str)
    mode_feedback_path = latest_daily_path("_mode_feedback.json", prefer_date=date_str)

    daily_signals = safe_read_json(daily_signals_path) or []
    mode_feedback = safe_read_json(mode_feedback_path) or {}

    # Read paper state here so we can pass it into decide_action
    try:
        px, px_source, px_note = spot_price(BINANCE_SYMBOL)
        paper = load_paper_state(date=date_str or health.get("date") or "", px=px)
    except Exception as e:
        if pulse_lock_acquired and pulse_lock_acquired_ts is not None:
            pulse_lock_hold_sec = round(max(0.0, pytime.monotonic() - pulse_lock_acquired_ts), 6)
        event = {
            "domain": "lie_spot_core",
            "ts": ts,
            "event_source": event_source,
            "decision": "error",
            "error": f"price_fetch_fatal: {type(e).__name__} {str(e)[:200]}",
            "net_proxy_bypass": get_proxy_bypass_stats(),
            "pulse_lock": {
                "path": str(PULSE_LOCK_PATH),
                "acquired": bool(pulse_lock_acquired),
                "reason": pulse_lock_reason,
                "wait_sec": pulse_lock_wait_sec,
                "hold_sec": pulse_lock_hold_sec,
            },
        }
        append_state(event, [f"Critical Price Fetch Error: {e}"])
        print(json.dumps(event, ensure_ascii=False))
        return 1

    action, diag = decide_action(daily_signals, mode_feedback, paper)
    cortex_mode, cortex_debug = StateVectorCortex().eval_irpota()
    if action == "VOLATILITY_PUMP" and cortex_mode in {"SURVIVAL", "STABILIZE", "HIBERNATE", "ROLLBACK", "OBSERVE"}:
        action = "FLAT_USDT"
        diag["cortex_override"] = True
        diag["cortex_override_mode"] = cortex_mode
        diag["cortex_override_reason"] = str(cortex_debug.get("trigger") or cortex_debug.get("reason") or "defensive_mode")
    else:
        diag["cortex_override"] = False
        diag["cortex_override_mode"] = cortex_mode

    # --- C) Price & paper guardrails ---
    paper_equity = paper.cash_usdt + paper.eth_qty * px
    paper.equity_peak = max(paper.equity_peak, paper_equity)
    drawdown = paper.equity_peak - paper_equity
    drawdown_pct = 0.0
    if paper.equity_peak > 1e-9:
        drawdown_pct = max(0.0, drawdown / paper.equity_peak)

    guardrail_hit = False
    guardrail_reasons: List[str] = []
    consecutive_loss_ack = load_consecutive_loss_ack(
        now_dt=dt.datetime.now(dt.timezone.utc),
        current_streak=int(paper.consecutive_losses),
        stop_threshold=int(CONSECUTIVE_LOSS_STOP),
    )
    if MAX_DAILY_DRAWDOWN_USDT > 0 and drawdown >= MAX_DAILY_DRAWDOWN_USDT:
        guardrail_hit = True
        guardrail_reasons.append(f"max_daily_drawdown_usdt_hit(drawdown={drawdown:.4f}>= {MAX_DAILY_DRAWDOWN_USDT})")
    if MAX_DAILY_DRAWDOWN_PCT > 0 and drawdown_pct >= MAX_DAILY_DRAWDOWN_PCT:
        guardrail_hit = True
        guardrail_reasons.append(
            f"max_daily_drawdown_pct_hit(drawdown_pct={drawdown_pct:.4f}>= {MAX_DAILY_DRAWDOWN_PCT:.4f})"
        )
    if paper.consecutive_losses >= CONSECUTIVE_LOSS_STOP and not bool(consecutive_loss_ack.get("applied")):
        guardrail_hit = True
        guardrail_reasons.append(f"consecutive_loss_stop_hit(streak={paper.consecutive_losses}>= {CONSECUTIVE_LOSS_STOP})")
    if paper_mode_gate_blocked:
        guardrail_hit = True
        guardrail_reasons.append(f"paper_mode_readiness_gate({paper_mode_gate_reason})")

    exec_probe_order_test = env_flag("LIE_SPOT_EXEC_PROBE_ORDER_TEST", default=False)
    exec_force_probe_on_guardrail = env_flag(
        "LIE_SPOT_EXEC_FORCE_PROBE_ON_GUARDRAIL",
        default=False,
    )
    exec_live_order_requested = env_flag("LIE_SPOT_EXEC_LIVE_ORDER", default=False)
    exec_probe_on_guardrail = should_attempt_probe_on_guardrail(
        guardrail_hit=guardrail_hit,
        probe_order_test=exec_probe_order_test,
        force_probe_on_guardrail=exec_force_probe_on_guardrail,
    )
    exec_probe_effective = bool(exec_probe_order_test and ((not guardrail_hit) or exec_probe_on_guardrail))
    # Hard guard: never allow live orders while any guardrail is active.
    exec_live_effective = bool(exec_live_order_requested and (not guardrail_hit) and (not exec_probe_effective))
    exec_cmd: Optional[List[str]] = None
    exec_attempted = False
    consecutive_loss_ack = consume_consecutive_loss_ack(
        ack_state=consecutive_loss_ack,
        cycle_ts=ts,
    )

    # --- C) Call execution driver ---
    exec_out: Dict[str, Any] = {}
    realized = 0.0
    decision = "no-trade"
    paper_execution: Dict[str, Any] = {
        "attempted": False,
        "applied": False,
        "route": None,
        "side": None,
        "qty": None,
        "mark_px": None,
        "fill_px": None,
        "signed_slippage_bps": None,
        "fee_rate": None,
        "fee_usdt": None,
        "ledger_path": str(PAPER_EXECUTION_LEDGER_PATH),
        "ledger_written": None,
    }

    if guardrail_hit and (not exec_probe_on_guardrail):
        decision = "no-trade"
        exec_out = {
            "ts": ts,
            "symbol": BINANCE_SYMBOL,
            "px": px,
            "free_usdt": None,
            "free_eth": None,
            "action": action,
            "decision": "no-trade",
            "reason": (
                "paper_mode_readiness_gate"
                if paper_mode_gate_blocked
                else "guardrail_hit"
            ),
            "guardrail_reasons": guardrail_reasons[:8],
            "order_result": {"http": 0, "endpoint": "skipped_guardrail"},
        }
    else:
        exec_cmd = build_exec_command(
            action=action,
            probe_order_test=exec_probe_effective,
            live_order=exec_live_effective,
        )
        exec_attempted = True
        try:
            exec_out = run_json(
                exec_cmd,
                cwd=TRADER_ROOT,
                timeout_s=120,
            )
        except Exception as exc:
            exec_out = {
                "ts": ts,
                "symbol": BINANCE_SYMBOL,
                "px": px,
                "action": action,
                "decision": "error",
                "reason": f"exec_runner_error:{type(exc).__name__}",
                "error": str(exc)[:800],
                "order_result": {"http": 0, "endpoint": "exec_runner_error"},
            }
        decision = str(exec_out.get("decision") or "no-trade")

        if _should_route_local_paper_fill(
            decision=decision,
            exec_out=exec_out,
            guardrail_hit=guardrail_hit,
            live_order_requested=exec_live_order_requested,
        ):
            local_plan = _build_local_paper_order_plan(
                action=action,
                px=float(exec_out.get("px") or px),
                st=paper,
                mode_hint=str(cortex_mode),
                debug_hint={"state": (cortex_debug.get("state") if isinstance(cortex_debug, dict) else {})},
            )
            if isinstance(local_plan, dict):
                decision = "simulate"
                exec_out["decision"] = "simulate"
                exec_out["reason"] = "local_paper_router_simulated"
                exec_out["order_plan"] = local_plan
                exec_out["order_result"] = {
                    "http": 200,
                    "endpoint": "simulated/order/local-paper",
                    "decision": "simulate",
                    "mode": "paper_local_router",
                    "reason": "dry_run_no_private_endpoints",
                    "error": None,
                    "cap_violation": None,
                }

        # paper fill based on executor plan
        plan = exec_out.get("order_plan") if isinstance(exec_out, dict) else None
        order_result = exec_out.get("order_result") if isinstance(exec_out, dict) else None
        order_mode = (
            str((order_result or {}).get("mode") or "").strip().lower()
            if isinstance(order_result, dict)
            else ""
        )
        should_apply_paper_fill = bool(
            isinstance(plan, dict)
            and (
                decision == "order"
                or (decision == "simulate" and order_mode == "paper_local_router")
            )
        )
        if should_apply_paper_fill and isinstance(plan, dict):
            side = str(plan.get("side") or "")
            qty = float(plan.get("qty") or 0.0)
            mark_px = float(exec_out.get("px") or px)
            fill_px, signed_slippage_bps = _effective_paper_fill_price(side, mark_px)
            fee_rate = max(0.0, env_float("LIE_PAPER_FILL_FEE_RATE", 0.0))
            paper_execution["attempted"] = True
            paper_execution["route"] = "paper_local_router" if order_mode == "paper_local_router" else "executor_order"
            paper_execution["side"] = side or None
            paper_execution["qty"] = float(round(qty, 8))
            paper_execution["mark_px"] = float(round(mark_px, 8))
            paper_execution["fill_px"] = float(round(fill_px, 8))
            paper_execution["signed_slippage_bps"] = float(round(signed_slippage_bps, 6))
            paper_execution["fee_rate"] = float(round(fee_rate, 8))
            try:
                cash_before = float(paper.cash_usdt)
                eth_before = float(paper.eth_qty)
                pnl_before = float(paper.daily_realized_pnl)
                realized, paper_equity = apply_paper_fill(
                    paper,
                    side=side,
                    qty=qty,
                    px=fill_px,
                )
                fee_usdt = abs(float(qty) * float(fill_px)) * float(fee_rate)
                if fee_usdt > 0:
                    paper.cash_usdt -= fee_usdt
                    paper.daily_realized_pnl -= fee_usdt
                    realized -= fee_usdt
                paper_execution["fee_usdt"] = float(round(fee_usdt, 8))
                # Simulated fill path returns equity=0.0 sentinel; keep current equity in that case.
                if paper_equity <= 0:
                    paper_equity = paper.cash_usdt + paper.eth_qty * mark_px
                else:
                    paper_equity = paper.cash_usdt + paper.eth_qty * mark_px
                drawdown = paper.equity_peak - paper_equity
                drawdown_pct = 0.0
                if paper.equity_peak > 1e-9:
                    drawdown_pct = max(0.0, drawdown / paper.equity_peak)
                paper_execution["applied"] = True
                paper_execution["ledger_written"] = _append_paper_execution_ledger(
                    {
                        "domain": "paper_execution",
                        "ts": ts,
                        "event_source": event_source,
                        "symbol": BINANCE_SYMBOL,
                        "action": action,
                        "decision": decision,
                        "route": paper_execution.get("route"),
                        "side": side,
                        "qty": float(round(qty, 8)),
                        "mark_px": float(round(mark_px, 8)),
                        "fill_px": float(round(fill_px, 8)),
                        "signed_slippage_bps": float(round(signed_slippage_bps, 6)),
                        "fee_rate": float(round(fee_rate, 8)),
                        "fee_usdt": float(round(fee_usdt, 8)),
                        "notional_usdt": float(round(abs(qty * fill_px), 8)),
                        "realized_pnl_change": float(round(realized, 8)),
                        "paper_daily_realized_pnl_before": float(round(pnl_before, 8)),
                        "paper_daily_realized_pnl_after": float(round(paper.daily_realized_pnl, 8)),
                        "paper_cash_before": float(round(cash_before, 8)),
                        "paper_cash_after": float(round(paper.cash_usdt, 8)),
                        "paper_eth_before": float(round(eth_before, 8)),
                        "paper_eth_after": float(round(paper.eth_qty, 8)),
                        "paper_equity_after": float(round(paper_equity, 8)),
                        "order_mode": order_mode or None,
                    }
                )
            except SpinalReflexReject as e:
                decision = "no-trade"
                exec_out["decision"] = "no-trade"
                exec_out["reason"] = f"paper_fill_gated:{e.gate.mode.lower()}"
                exec_out["paper_fill_gate"] = {
                    "mode": e.gate.mode,
                    "policy": e.gate.policy,
                    "reason": e.gate.reason,
                    "cap_violation": (e.gate.debug or {}).get("cap_violation"),
                }
                paper_execution["applied"] = False
                paper_execution["ledger_written"] = False

    save_paper_state(paper)
    paper_artifacts_sync = sync_paper_execution_artifacts(
        as_of=date_str,
        st=paper,
        px=float(exec_out.get("px") or px),
        symbol=BINANCE_SYMBOL,
    )
    if isinstance(paper_artifacts_sync, dict):
        paper_artifacts_sync["event_source"] = event_source

    # Build event
    order_plan = exec_out.get("order_plan") if isinstance(exec_out, dict) else None
    order_result = exec_out.get("order_result") if isinstance(exec_out, dict) else None
    order_result_payload: Optional[Dict[str, Any]] = None
    if isinstance(order_result, dict):
        order_result_payload = {
            "http": order_result.get("http"),
            "endpoint": order_result.get("endpoint"),
            "decision": order_result.get("decision"),
            "mode": order_result.get("mode"),
            "reason": order_result.get("reason"),
            "error": order_result.get("error"),
            "cap_violation": order_result.get("cap_violation"),
        }
    elif decision == "error":
        order_result_payload = {
            "http": exec_out.get("http") if isinstance(exec_out, dict) else None,
            "endpoint": "exec_error",
            "decision": "error",
            "mode": None,
            "reason": exec_out.get("reason") if isinstance(exec_out, dict) else None,
            "error": exec_out.get("error") if isinstance(exec_out, dict) else None,
            "cap_violation": None,
        }

    if pulse_lock_acquired and pulse_lock_acquired_ts is not None:
        pulse_lock_hold_sec = round(max(0.0, pytime.monotonic() - pulse_lock_acquired_ts), 6)
    event: Dict[str, Any] = {
        "domain": "lie_spot_core",
        "ts": ts,
        "event_source": event_source,
        "bucket": bucket or "",
        "symbol": BINANCE_SYMBOL,
        "action": action,
        "px": float(exec_out.get("px") or px),
        "px_source": px_source,
        "px_fallback_triggered": str(px_source).lower() != "binance",
        "px_note": px_note,
        "free_usdt": exec_out.get("free_usdt"),
        "free_eth": exec_out.get("free_eth"),
        "decision": decision,
        "executor": {
            "attempted": exec_attempted,
            "probe_requested": bool(exec_probe_order_test),
            "probe_effective": bool(exec_probe_effective),
            "force_probe_on_guardrail": bool(exec_force_probe_on_guardrail),
            "live_requested": bool(exec_live_order_requested),
            "live_effective": bool(exec_live_effective),
            "cmd": exec_cmd if exec_cmd else None,
        },
        "order_plan": {
            "notional": (order_plan or {}).get("notional"),
            "qty": (order_plan or {}).get("qty"),
            "side": (order_plan or {}).get("side"),
        }
        if isinstance(order_plan, dict)
        else None,
        "order_result": order_result_payload,
        "paper_fill_gate": exec_out.get("paper_fill_gate") if isinstance(exec_out, dict) else None,
        "multimodal": {
            "micro_imbalance": float(micro_feat),
            "sentiment_pca": float(sent_feat),
            "sentiment_source": sent_source,
            "onchain_proxy": float(onchain_feat),
            "onchain_source": onchain_source,
            "hmm_proxy_version": "v1",
            "entropy": float(hmm_proxy.get("entropy") or 0.0),
            "log_likelihood": float(hmm_proxy.get("log_likelihood") or -0.3),
            "distribution_shift": float(hmm_proxy.get("distribution_shift") or 0.0),
        },
        "lie": {
            "date": date,
            "daemon": {
                "would_run_pulse": would_run_pulse,
                "pulse_reason": pulse_reason,
                "mode": daemon_mode,
                "command": daemon_cmd,
                "due_slots": due_slots[:3],
            },
            "pulse": {
                "ran": pulse_ran,
                "skipped": (pulse or {}).get("skipped") if isinstance(pulse, dict) else None,
                "reason": (pulse or {}).get("reason") if isinstance(pulse, dict) else None,
            },
            "health": {
                "status": health.get("status"),
                "missing": health.get("missing"),
            },
            "cortex": {
                "mode": cortex_mode,
                "reason": cortex_debug.get("reason"),
                "trigger": cortex_debug.get("trigger"),
            },
            "signal_diag": diag,
        },
        "paper": {
            "cash_usdt": round(paper.cash_usdt, 6),
            "eth_qty": round(paper.eth_qty, 6),
            "avg_cost": round(paper.avg_cost, 6),
            "equity": round(paper.cash_usdt + paper.eth_qty * float(exec_out.get("px") or px), 6),
            "equity_peak": round(paper.equity_peak, 6),
            "drawdown": round(drawdown, 6),
            "drawdown_pct": round(drawdown_pct, 6),
            "daily_realized_pnl": round(paper.daily_realized_pnl, 6),
            "consecutive_losses": int(paper.consecutive_losses),
            "last_loss_ts": paper.last_loss_ts,
            "last_realized_pnl": round(realized, 6),
        },
        "guardrails": {
            "max_daily_drawdown_usdt": MAX_DAILY_DRAWDOWN_USDT,
            "max_daily_drawdown_pct": MAX_DAILY_DRAWDOWN_PCT,
            "consecutive_loss_stop": CONSECUTIVE_LOSS_STOP,
            "hit": guardrail_hit,
            "reasons": guardrail_reasons,
        },
        "paper_consecutive_loss_ack": consecutive_loss_ack,
        "paper_mode_readiness": paper_mode_readiness,
        "pulse_lock": {
            "path": str(PULSE_LOCK_PATH),
            "acquired": bool(pulse_lock_acquired),
            "reason": pulse_lock_reason,
            "wait_sec": pulse_lock_wait_sec,
            "hold_sec": pulse_lock_hold_sec,
        },
        "paper_artifacts_sync": paper_artifacts_sync,
        "paper_execution": paper_execution,
        "net_proxy_bypass": get_proxy_bypass_stats(),
        "exec_net_proxy_bypass": (
            exec_out.get("net_proxy_bypass")
            if isinstance(exec_out, dict) and isinstance(exec_out.get("net_proxy_bypass"), dict)
            else None
        ),
    }

    # 3-line summary
    sig_line = f"signal={action} bucket={bucket} date={date} px_source={px_source}{(' note=' + px_note) if px_note else ''}"
    def _fmt2_or_na(v: Any) -> str:
        try:
            return f"{float(v):.2f}"
        except Exception:
            return "na"

    reason_line = (
        f"reason: strong_signal={diag.get('strong_signal')} signals={diag.get('signals_count')} act={action} "
        + f"max_conf={_fmt2_or_na(diag.get('max_confidence'))}>=min{_fmt2_or_na(diag.get('conf_min_effective'))} "
        + f"mode_passed={diag.get('mode_passed')} "
        + f"px={px:.2f} guardrail_hit={guardrail_hit}{(' ' + ';'.join(guardrail_reasons)) if guardrail_reasons else ''}"
    )

    if guardrail_hit and (not exec_attempted):
        sim_line = f"dry-run skipped (guardrail); paper_equity={event['paper']['equity']} dd={event['paper']['drawdown']} streak={event['paper']['consecutive_losses']}"
    else:
        http = (event.get("order_result") or {}).get("http")
        ep = (event.get("order_result") or {}).get("endpoint")
        gate_mode = (event.get("order_result") or {}).get("mode")
        exec_error = (event.get("order_result") or {}).get("error")
        cap_violation = (event.get("order_result") or {}).get("cap_violation")
        notional = ((event.get("order_plan") or {}) if event.get("order_plan") else {}).get("notional")
        qty = ((event.get("order_plan") or {}) if event.get("order_plan") else {}).get("qty")
        sim_line = (
            f"exec decision={decision} probe={event['executor']['probe_effective']} live={event['executor']['live_effective']} "
            f"http={http} endpoint={ep} gate_mode={gate_mode} exec_error={exec_error} cap_violation={cap_violation} "
            f"plan_notional={notional} qty={qty} paper_equity={event['paper']['equity']}"
        )

    append_state(event, [sig_line, reason_line, sim_line])
    print(json.dumps(event, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        # Last-resort: write a minimal error record to STATE.md
        try:
            STATE_MD.parent.mkdir(parents=True, exist_ok=True)
            with STATE_MD.open("a", encoding="utf-8") as f:
                f.write(
                    "\nLIE_DRYRUN_EVENT="
                    + json.dumps(
                        {
                            "ts": now_local_iso(),
                            "event_source": event_source_label(),
                            "decision": "error",
                            "error": str(e)[:800],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.write("signal=UNKNOWN\n")
                f.write("reason=exception\n")
                f.write("result=error_logged\n")
        except Exception:
            pass
        raise
