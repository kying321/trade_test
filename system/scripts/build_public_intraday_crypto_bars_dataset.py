#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import ssl
import time
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
DEFAULT_INTERVAL = "15m"
DEFAULT_LIMIT = 960
MAX_KLINES_PER_REQUEST = 1500
DEFAULT_TIMEOUT_MS = 5000
DEFAULT_RPM = 12


def parse_stamp(raw: str) -> dt.datetime:
    parsed = dt.datetime.strptime(str(raw).strip(), "%Y%m%dT%H%M%SZ")
    return parsed.replace(tzinfo=dt.timezone.utc)


def fmt_utc(value: dt.datetime | pd.Timestamp | None) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval_to_timedelta(interval: str) -> pd.Timedelta:
    raw = text(interval).lower()
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "3m": pd.Timedelta(minutes=3),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "6h": pd.Timedelta(hours=6),
        "8h": pd.Timedelta(hours=8),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
    }
    if raw not in mapping:
        raise ValueError(f"unsupported_interval:{interval}")
    return mapping[raw]


class TokenBucket:
    def __init__(self, rate_per_minute: int, capacity: int = 2) -> None:
        self.capacity = max(1.0, float(capacity))
        self.tokens = self.capacity
        self.rate_per_second = max(1.0, float(rate_per_minute)) / 60.0
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


def http_get_json(*, url: str, timeout_ms: int, bucket: TokenBucket) -> Any:
    bucket.take()
    req = urllib_request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "fenlie-public-intraday-bars/1.0",
        },
        method="GET",
    )

    def opener(ctx: ssl.SSLContext):
        return urllib_request.build_opener(
            urllib_request.HTTPHandler(),
            urllib_request.HTTPSHandler(context=ctx),
        )

    try:
        with opener(ssl.create_default_context()).open(
            req, timeout=max(0.1, min(5.0, float(timeout_ms) / 1000.0))
        ) as resp:
            payload = resp.read().decode("utf-8")
    except (TimeoutError, urllib_error.HTTPError, urllib_error.URLError) as exc:
        exc_text = str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in exc_text.upper():
            with opener(ssl._create_unverified_context()).open(
                req, timeout=max(0.1, min(5.0, float(timeout_ms) / 1000.0))
            ) as resp:
                payload = resp.read().decode("utf-8")
        else:
            raise RuntimeError(f"public_intraday_http_error:{type(exc).__name__}:{exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("public_intraday_invalid_json") from exc


def fetch_binance_futures_bars(
    *,
    symbol: str,
    interval: str,
    limit: int,
    timeout_ms: int,
    bucket: TokenBucket,
) -> pd.DataFrame:
    interval_td = interval_to_timedelta(interval)
    remaining = max(1, int(limit))
    next_end_time_ms: int | None = None
    chunks: list[pd.DataFrame] = []
    while remaining > 0:
        batch_limit = max(1, min(MAX_KLINES_PER_REQUEST, remaining))
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": batch_limit,
        }
        if next_end_time_ms is not None:
            params["endTime"] = next_end_time_ms
        query = urllib_parse.urlencode(params)
        payload = http_get_json(
            url=f"https://fapi.binance.com/fapi/v1/klines?{query}",
            timeout_ms=timeout_ms,
            bucket=bucket,
        )
        if not isinstance(payload, list) or not payload:
            break
        rows: list[dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, list) or len(row) < 6:
                continue
            rows.append(
                {
                    "ts": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                    "symbol": symbol,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "source": "binance_futures_public",
                    "asset_class": "crypto",
                    "interval": interval,
                }
            )
        if not rows:
            break
        chunk = pd.DataFrame(rows).sort_values("ts").drop_duplicates(subset=["ts", "symbol"]).reset_index(drop=True)
        chunks.append(chunk)
        remaining -= len(chunk)
        if len(chunk) < batch_limit:
            break
        oldest_ts = pd.to_datetime(chunk["ts"].min(), utc=True)
        next_end_time_ms = int((oldest_ts - interval_td).timestamp() * 1000)
        if next_end_time_ms <= 0:
            break
    if not chunks:
        raise RuntimeError(f"public_intraday_empty_klines:{symbol}")
    frame = pd.concat(chunks, ignore_index=True)
    if frame.empty:
        raise RuntimeError(f"public_intraday_empty_rows:{symbol}")
    frame = frame.sort_values("ts").drop_duplicates(subset=["ts", "symbol"]).reset_index(drop=True)
    if len(frame) > limit:
        frame = frame.iloc[-int(limit) :].reset_index(drop=True)
    return frame


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Public Intraday Crypto Bars Dataset",
        "",
        f"- interval: `{text(payload.get('interval'))}`",
        f"- symbol_count: `{payload.get('symbol_count')}`",
        f"- symbols: `{','.join(payload.get('symbols', []))}`",
        f"- coverage: `{text(payload.get('coverage_start_utc'))} -> {text(payload.get('coverage_end_utc'))}`",
        f"- row_count: `{payload.get('row_count')}`",
        f"- cadence_minutes: `{payload.get('cadence_minutes')}`",
        f"- dataset_status: `{text(payload.get('dataset_status'))}`",
        "",
        "## Symbol Coverage",
        "",
    ]
    for row in payload.get("symbol_coverage", []):
        lines.append(
            f"- `{row['symbol']}` | rows=`{row['rows']}` | coverage=`{row['start_utc']} -> {row['end_utc']}`"
        )
    lines.extend(["", "## Notes", "", f"- `{text(payload.get('research_note'))}`", f"- `{text(payload.get('limitation_note'))}`"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only public intraday crypto bars dataset.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp_dt = parse_stamp(args.stamp)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    interval_td = interval_to_timedelta(str(args.interval))
    target_limit = int(args.limit)
    if int(args.days) > 0:
        bars_per_day = max(1, int(pd.Timedelta(days=1) / interval_td))
        target_limit = max(target_limit, int(args.days) * bars_per_day)
    bucket = TokenBucket(int(args.rpm))
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for symbol in symbols:
        try:
            frames.append(
                fetch_binance_futures_bars(
                    symbol=symbol,
                    interval=str(args.interval),
                    limit=target_limit,
                    timeout_ms=int(args.timeout_ms),
                    bucket=bucket,
                )
            )
        except Exception as exc:
            errors.append(f"{symbol}:{type(exc).__name__}:{exc}")
    if not frames:
        raise SystemExit("public_intraday_fetch_failed_for_all_symbols")
    frame = pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"]).reset_index(drop=True)
    dataset_csv = review_dir / f"{args.stamp}_public_intraday_crypto_bars_dataset.csv"
    frame.to_csv(dataset_csv, index=False)
    ts = pd.to_datetime(frame["ts"], utc=True)
    cadence = None
    for _, group in frame.groupby("symbol"):
        delta = pd.to_datetime(group["ts"], utc=True).sort_values().diff().dropna()
        if not delta.empty:
            cadence = int(delta.mode().iloc[0] / pd.Timedelta(minutes=1))
            break
    symbol_coverage = [
        {
            "symbol": symbol,
            "rows": int(len(group)),
            "start_utc": fmt_utc(pd.Timestamp(group["ts"].min()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
            "end_utc": fmt_utc(pd.Timestamp(group["ts"].max()).to_pydatetime().replace(tzinfo=dt.timezone.utc)),
        }
        for symbol, group in frame.groupby("symbol", sort=True)
    ]
    payload = {
        "action": "build_public_intraday_crypto_bars_dataset",
        "ok": True,
        "status": "ok",
        "change_class": "RESEARCH_ONLY",
        "generated_at_utc": fmt_utc(stamp_dt),
        "interval": str(args.interval),
        "days_requested": int(args.days),
        "bars_requested_per_symbol": target_limit,
        "symbols": sorted(frame["symbol"].astype(str).str.upper().unique().tolist()),
        "symbol_count": int(frame["symbol"].nunique()),
        "row_count": int(len(frame)),
        "coverage_start_utc": fmt_utc(pd.Timestamp(ts.min()).to_pydatetime()),
        "coverage_end_utc": fmt_utc(pd.Timestamp(ts.max()).to_pydatetime()),
        "cadence_minutes": cadence,
        "dataset_status": "complete" if len(errors) == 0 else "partial_success",
        "fetch_errors": errors,
        "symbol_coverage": symbol_coverage,
        "dataset_csv": str(dataset_csv),
        "dataset_csv_sha256": sha256_file(dataset_csv),
        "research_note": "这份数据集只服务于 research/SIM_ONLY，不进入 live shortline source 链。",
        "limitation_note": "当前只拉 Binance futures public klines；若后续要做更稳的 intraday 截面，仍要补备用源和成交约束。",
    }
    json_path = review_dir / f"{args.stamp}_public_intraday_crypto_bars_dataset.json"
    md_path = review_dir / f"{args.stamp}_public_intraday_crypto_bars_dataset.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json_path": str(json_path), "md_path": str(md_path), "dataset_csv": str(dataset_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
