#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.request


SYSTEM_ROOT = Path(
    str(os.getenv("LIE_SYSTEM_ROOT", "")).strip()
    or str(os.getenv("FENLIE_SYSTEM_ROOT", "")).strip()
    or Path(__file__).resolve().parents[1]
).expanduser().resolve()
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lie_engine.data.storage import write_json, write_markdown


DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_ARTIFACT_TTL_HOURS = 168.0
DEFAULT_KEEP_FILES = 40
DEFAULT_RATE_LIMIT_PER_MINUTE = 30.0
DEFAULT_TIMEOUT_MS = 5000
DEFAULT_CRYPTO_LIMIT = 10
DEFAULT_COMMODITY_SYMBOLS = ["XAUUSD", "XAGUSD", "WTIUSD", "BRENTUSD", "NATGAS", "COPPER"]
DEFAULT_STATIC_CRYPTO = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "BNBUSDT",
    "SUIUSDT",
    "ADAUSDT",
    "LINKUSDT",
    "AVAXUSDT",
]
STABLE_BASES = {
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "DAI",
    "TUSD",
    "PYUSD",
    "USD1",
    "USDE",
    "USDS",
}
TOKENIZED_METAL_BASES = {"XAUT", "PAXG"}
FIAT_BASES = {
    "AED",
    "ARS",
    "AUD",
    "BRL",
    "COP",
    "CZK",
    "EUR",
    "GBP",
    "JPY",
    "MXN",
    "NGN",
    "PLN",
    "RON",
    "RUB",
    "TRY",
    "UAH",
    "ZAR",
}
COINGECKO_ID_TO_SYMBOL = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
    "ripple": "XRPUSDT",
    "dogecoin": "DOGEUSDT",
    "binancecoin": "BNBUSDT",
    "sui": "SUIUSDT",
    "cardano": "ADAUSDT",
    "chainlink": "LINKUSDT",
    "avalanche-2": "AVAXUSDT",
    "tron": "TRXUSDT",
    "toncoin": "TONUSDT",
    "hedera-hashgraph": "HBARUSDT",
    "litecoin": "LTCUSDT",
    "bitcoin-cash": "BCHUSDT",
    "polkadot": "DOTUSDT",
    "aptos": "APTUSDT",
    "near": "NEARUSDT",
    "stellar": "XLMUSDT",
    "pepe": "PEPEUSDT",
    "shiba-inu": "SHIBUSDT",
}
GENERIC_COINGECKO_SYMBOL_ALLOWLIST = {
    "AAVE",
    "ADA",
    "APT",
    "AVAX",
    "BCH",
    "BNB",
    "BTC",
    "CHZ",
    "DOGE",
    "DOT",
    "ENA",
    "ETH",
    "FIL",
    "FLOW",
    "HBAR",
    "LINK",
    "LTC",
    "NEAR",
    "PEPE",
    "SEI",
    "SHIB",
    "SOL",
    "SUI",
    "TAO",
    "TON",
    "TRX",
    "UNI",
    "WIF",
    "WLD",
    "XLM",
    "XMR",
    "XRP",
    "ZRO",
}
PRECIOUS_METAL_SYMBOLS = {"XAUUSD", "XAGUSD"}
ENERGY_SYMBOLS = {"WTIUSD", "BRENTUSD", "NATGAS"}
BASE_METAL_SYMBOLS = {"COPPER"}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_utc_iso() -> str:
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    candidate = text.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(candidate)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def normalize_symbol(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or "").upper())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def evict_old_artifacts(
    *,
    review_dir: Path,
    current_artifact: Path,
    current_checksum: Path,
    current_report: Path,
    now_dt: dt.datetime,
    ttl_hours: float,
    keep_files: int,
) -> int:
    cutoff = now_dt - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {current_artifact.name, current_checksum.name, current_report.name}
    candidates: list[Path] = []
    for pattern in (
        "*_hot_research_universe.json",
        "*_hot_research_universe_checksum.json",
        "*_hot_research_universe.md",
    ):
        candidates.extend(review_dir.glob(pattern))
    deleted = 0
    by_mtime = sorted(
        [path for path in candidates if path.name not in protected],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    keep_names = {path.name for path in by_mtime[: max(0, keep_files - len(protected))]}
    for path in by_mtime:
        stat = path.stat()
        if path.name in keep_names and dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc) >= cutoff:
            continue
        if dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc) >= cutoff and path.name in keep_names:
            continue
        path.unlink(missing_ok=True)
        deleted += 1
    return deleted


class TokenBucket:
    def __init__(self, *, rate_per_minute: float, capacity: float = 3.0) -> None:
        self.rate_per_second = max(1.0, float(rate_per_minute)) / 60.0
        self.capacity = max(1.0, float(capacity))
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def take(self) -> None:
        while True:
            now_mono = time.monotonic()
            elapsed = max(0.0, now_mono - self.last_refill)
            if elapsed > 0.0:
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
                self.last_refill = now_mono
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            wait_seconds = max(0.01, (1.0 - self.tokens) / max(1e-9, self.rate_per_second))
            time.sleep(min(wait_seconds, 0.25))


def request_json(
    *,
    url: str,
    bucket: TokenBucket,
    timeout_ms: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    bucket.take()
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "fenlie-hot-research-universe/1.0",
            "Accept": "application/json",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(request, timeout=max(0.1, timeout_ms / 1000.0)) as response:
        raw = response.read().decode("utf-8")
        payload = json.loads(raw)
        return {
            "ok": True,
            "http_status": int(getattr(response, "status", 200) or 200),
            "payload": payload,
        }


def is_valid_hot_crypto_symbol(symbol: str) -> bool:
    sym = normalize_symbol(symbol)
    if not sym.endswith("USDT") or len(sym) <= 4:
        return False
    base = sym[:-4]
    if base in STABLE_BASES or base in TOKENIZED_METAL_BASES or base in FIAT_BASES:
        return False
    if any(base.endswith(suffix) for suffix in ("UP", "DOWN", "BULL", "BEAR")):
        return False
    return True


def fetch_binance_symbols(bucket: TokenBucket, timeout_ms: int) -> dict[str, Any]:
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = request_json(url=url, bucket=bucket, timeout_ms=timeout_ms)
        payload = response["payload"]
        if not isinstance(payload, list):
            return {"ok": False, "status": "invalid_payload", "http_status": response["http_status"], "symbols": []}
        ranked = sorted(
            [row for row in payload if isinstance(row, dict)],
            key=lambda row: float(row.get("quoteVolume", 0.0) or 0.0),
            reverse=True,
        )
        symbols = [normalize_symbol(str(row.get("symbol", ""))) for row in ranked]
        filtered = [sym for sym in symbols if is_valid_hot_crypto_symbol(sym)]
        return {
            "ok": True,
            "status": "ok",
            "http_status": response["http_status"],
            "symbols": filtered,
            "raw_count": len(payload),
        }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": f"http_{exc.code}", "http_status": int(exc.code), "symbols": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": type(exc).__name__, "symbols": [], "error": str(exc)}


def fetch_bybit_symbols(bucket: TokenBucket, timeout_ms: int) -> dict[str, Any]:
    url = "https://api.bybit.com/v5/market/tickers?category=spot"
    try:
        response = request_json(url=url, bucket=bucket, timeout_ms=timeout_ms)
        payload = response["payload"]
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        rows = result.get("list", []) if isinstance(result, dict) else []
        if not isinstance(rows, list):
            return {"ok": False, "status": "invalid_payload", "http_status": response["http_status"], "symbols": []}
        ranked = sorted(
            [row for row in rows if isinstance(row, dict)],
            key=lambda row: float(row.get("turnover24h", 0.0) or 0.0),
            reverse=True,
        )
        symbols = [normalize_symbol(str(row.get("symbol", ""))) for row in ranked]
        filtered = [sym for sym in symbols if is_valid_hot_crypto_symbol(sym)]
        return {
            "ok": True,
            "status": "ok",
            "http_status": response["http_status"],
            "symbols": filtered,
            "raw_count": len(rows),
        }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": f"http_{exc.code}", "http_status": int(exc.code), "symbols": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": type(exc).__name__, "symbols": [], "error": str(exc)}


def coingecko_to_symbol(row: dict[str, Any]) -> str:
    coin_id = str(row.get("id", "")).strip().lower()
    if coin_id in COINGECKO_ID_TO_SYMBOL:
        return COINGECKO_ID_TO_SYMBOL[coin_id]
    symbol = normalize_symbol(str(row.get("symbol", "")))
    if not symbol or symbol in STABLE_BASES or symbol in TOKENIZED_METAL_BASES:
        return ""
    if len(symbol) > 10 or symbol not in GENERIC_COINGECKO_SYMBOL_ALLOWLIST:
        return ""
    candidate = f"{symbol}USDT"
    return candidate if is_valid_hot_crypto_symbol(candidate) else ""


def fetch_coingecko_symbols(bucket: TokenBucket, timeout_ms: int) -> dict[str, Any]:
    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd&order=volume_desc&per_page=60&page=1&sparkline=false&price_change_percentage=24h"
    )
    try:
        response = request_json(url=url, bucket=bucket, timeout_ms=timeout_ms)
        payload = response["payload"]
        if not isinstance(payload, list):
            return {"ok": False, "status": "invalid_payload", "http_status": response["http_status"], "symbols": []}
        mapped = [coingecko_to_symbol(row) for row in payload if isinstance(row, dict)]
        filtered: list[str] = []
        seen: set[str] = set()
        for symbol in mapped:
            if not symbol or symbol in seen:
                continue
            filtered.append(symbol)
            seen.add(symbol)
        return {
            "ok": True,
            "status": "ok",
            "http_status": response["http_status"],
            "symbols": filtered,
            "raw_count": len(payload),
        }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": f"http_{exc.code}", "http_status": int(exc.code), "symbols": [], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": type(exc).__name__, "symbols": [], "error": str(exc)}


def _unique_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in symbols:
        symbol = normalize_symbol(raw)
        if not symbol or symbol in seen:
            continue
        ordered.append(symbol)
        seen.add(symbol)
    return ordered


def build_batches(crypto: list[str], commodities: list[str]) -> dict[str, list[str]]:
    crypto_ranked = _unique_symbols(crypto)
    commodity_ranked = _unique_symbols(commodities)
    crypto_hot = crypto_ranked[: min(8, len(crypto_ranked))]
    crypto_majors = crypto_ranked[: min(6, len(crypto_ranked))]
    crypto_beta = crypto_ranked[len(crypto_majors) : len(crypto_majors) + min(6, max(0, len(crypto_ranked) - len(crypto_majors)))]
    precious_metals = [symbol for symbol in commodity_ranked if symbol in PRECIOUS_METAL_SYMBOLS]
    energy = [symbol for symbol in commodity_ranked if symbol in ENERGY_SYMBOLS]
    energy_liquids = [symbol for symbol in energy if symbol in {"WTIUSD", "BRENTUSD"}]
    energy_gas = [symbol for symbol in energy if symbol == "NATGAS"]
    base_metals = [symbol for symbol in commodity_ranked if symbol in BASE_METAL_SYMBOLS]
    metals_all = _unique_symbols(precious_metals + base_metals)
    metals_macro = _unique_symbols(precious_metals[:2] + base_metals[:1])
    mixed_macro = _unique_symbols(crypto_majors[:4] + precious_metals + energy[:2])
    mixed_macro_expanded = _unique_symbols(crypto_hot[:6] + commodity_ranked)

    candidates: list[tuple[str, list[str]]] = [
        ("crypto_hot", crypto_hot),
        ("crypto_majors", crypto_majors),
        ("crypto_beta", crypto_beta),
        ("precious_metals", precious_metals),
        ("energy", energy),
        ("energy_liquids", energy_liquids),
        ("energy_gas", energy_gas),
        ("base_metals", base_metals),
        ("metals_all", metals_all),
        ("metals_macro", metals_macro),
        ("commodities_benchmark", commodity_ranked),
        ("mixed_macro", mixed_macro),
        ("mixed_macro_expanded", mixed_macro_expanded),
    ]
    return {name: symbols for name, symbols in candidates if symbols}


def build_universe_payload(
    *,
    review_dir: Path,
    now_dt: dt.datetime,
    crypto_limit: int,
    commodity_symbols: list[str],
    network_mode: str,
    rate_limit_per_minute: float,
    timeout_ms: int,
    artifact_ttl_hours: float,
    keep_files: int,
) -> dict[str, Any]:
    bucket = TokenBucket(rate_per_minute=rate_limit_per_minute)
    statuses: dict[str, Any] = {}
    selected_crypto: list[str] = []
    source_tier = "static_fallback"

    if network_mode == "offline":
        statuses = {
            "binance": {"ok": False, "status": "skipped_offline", "symbols": []},
            "bybit": {"ok": False, "status": "skipped_offline", "symbols": []},
            "coingecko": {"ok": False, "status": "skipped_offline", "symbols": []},
        }
    else:
        statuses["binance"] = fetch_binance_symbols(bucket, timeout_ms)
        statuses["bybit"] = fetch_bybit_symbols(bucket, timeout_ms)
        statuses["coingecko"] = fetch_coingecko_symbols(bucket, timeout_ms)

    for source_name in ("binance", "bybit", "coingecko"):
        source_symbols = list(statuses.get(source_name, {}).get("symbols", []) or [])
        if not source_symbols:
            continue
        selected_crypto = source_symbols[: max(1, crypto_limit)]
        source_tier = source_name
        break

    if not selected_crypto:
        selected_crypto = DEFAULT_STATIC_CRYPTO[: max(1, crypto_limit)]
        source_tier = "static_fallback"

    core_symbols = selected_crypto + commodity_symbols
    batches = build_batches(selected_crypto, commodity_symbols)
    built_at = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    expires_at = (now_dt + dt.timedelta(hours=max(1.0, artifact_ttl_hours))).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict[str, Any] = {
        "action": "build_hot_research_universe",
        "ok": True,
        "status": "ok",
        "built_at_utc": built_at,
        "expires_at_utc": expires_at,
        "network_mode": network_mode,
        "crypto": {
            "selected": selected_crypto,
            "count": len(selected_crypto),
        },
        "commodities": {
            "selected": commodity_symbols,
            "count": len(commodity_symbols),
        },
        "core_symbols": core_symbols,
        "batches": batches,
        "source_tier": source_tier,
        "source_statuses": statuses,
        "artifact_ttl_hours": max(1.0, artifact_ttl_hours),
        "review_dir": str(review_dir),
    }

    stamp = now_dt.strftime("%Y%m%dT%H%M%SZ")
    artifact_path = review_dir / f"{stamp}_hot_research_universe.json"
    checksum_path = review_dir / f"{stamp}_hot_research_universe_checksum.json"
    report_path = review_dir / f"{stamp}_hot_research_universe.md"
    payload["artifact"] = str(artifact_path)
    payload["checksum"] = str(checksum_path)
    payload["report"] = str(report_path)
    payload["artifact_label"] = f"hot-research-universe:{source_tier}"
    payload["artifact_status_label"] = "hot-universe-ok"
    report_lines = [
        "# Hot Research Universe",
        f"- built_at_utc: `{built_at}`",
        f"- source_tier: `{source_tier}`",
        f"- network_mode: `{network_mode}`",
        f"- crypto_selected: `{', '.join(selected_crypto)}`",
        f"- commodity_selected: `{', '.join(commodity_symbols)}`",
        "",
        "## Batches",
    ]
    for batch_name, symbols in batches.items():
        report_lines.append(f"- `{batch_name}`: `{', '.join(symbols)}`")
    write_markdown(report_path, "\n".join(report_lines) + "\n")
    deleted = evict_old_artifacts(
        review_dir=review_dir,
        current_artifact=artifact_path,
        current_checksum=checksum_path,
        current_report=report_path,
        now_dt=now_dt,
        ttl_hours=max(1.0, artifact_ttl_hours),
        keep_files=max(3, keep_files),
    )
    payload["retention"] = {"evicted_files": int(deleted)}
    write_json(artifact_path, payload)
    checksum_payload = {
        "generated_at_utc": built_at,
        "artifact_ttl_hours": max(1.0, artifact_ttl_hours),
        "files": [{"path": str(artifact_path), "sha256": sha256_file(artifact_path)}],
    }
    write_json(checksum_path, checksum_payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a hot crypto + commodity research universe artifact.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--network-mode", choices=["online", "offline"], default="online")
    parser.add_argument("--crypto-limit", type=int, default=DEFAULT_CRYPTO_LIMIT)
    parser.add_argument("--commodity-symbols", nargs="*", default=DEFAULT_COMMODITY_SYMBOLS)
    parser.add_argument("--rate-limit-per-minute", type=float, default=DEFAULT_RATE_LIMIT_PER_MINUTE)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--artifact-ttl-hours", type=float, default=DEFAULT_ARTIFACT_TTL_HOURS)
    parser.add_argument("--keep-files", type=int, default=DEFAULT_KEEP_FILES)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(str(args.output_root)).expanduser().resolve()
    review_dir = Path(str(args.review_dir)).expanduser().resolve()
    if not review_dir.is_absolute():
        review_dir = (output_root / review_dir).resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    now_dt = parse_now(str(args.now))
    payload = build_universe_payload(
        review_dir=review_dir,
        now_dt=now_dt,
        crypto_limit=max(1, int(args.crypto_limit)),
        commodity_symbols=[normalize_symbol(s) for s in args.commodity_symbols if normalize_symbol(s)],
        network_mode=str(args.network_mode).strip().lower(),
        rate_limit_per_minute=max(1.0, float(args.rate_limit_per_minute)),
        timeout_ms=min(5000, max(100, int(args.timeout_ms))),
        artifact_ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
        keep_files=max(3, int(args.keep_files)),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
