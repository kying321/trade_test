#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import ssl
import time
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from pathlib import Path
from typing import Any

import pandas as pd

from build_crypto_shortline_execution_gate import (
    add_common_features,
    find_latest_bars_file,
    read_bars_file,
    structure_signals,
)


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
MAX_SHORTLINE_BAR_AGE_DAYS = 2
DEFAULT_BINANCE_FUTURES_INTERVAL = "1h"
DEFAULT_BINANCE_FUTURES_LIMIT = 240
DEFAULT_BINANCE_TIMEOUT_MS = 5000
DEFAULT_BINANCE_RPM = 12
DEFAULT_SHORTLINE_TEMPLATE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text_value = str(raw or "").strip()
    if not text_value:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    effective = value or now_utc()
    return effective.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def text(value: Any) -> str:
    return str(value or "").strip()


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = list(review_dir.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda item: (artifact_stamp(item), item.stat().st_mtime, item.name))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prune_artifacts(
    review_dir: Path,
    *,
    current_paths: list[Path],
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates: list[Path] = []
    for pattern in (
        "*_crypto_shortline_live_bars_snapshot.json",
        "*_crypto_shortline_live_bars_snapshot.md",
        "*_crypto_shortline_live_bars_snapshot_checksum.json",
        "*_crypto_shortline_live_bars_snapshot_bars.csv",
    ):
        candidates.extend(review_dir.glob(pattern))

    existing: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            existing.append((path.stat().st_mtime, path))
        except OSError:
            continue

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for _, path in sorted(existing, key=lambda item: item[0], reverse=True):
        if path.name in protected:
            survivors.append(path)
            continue
        try:
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            path.unlink(missing_ok=True)
            pruned_age.append(str(path))
        else:
            survivors.append(path)

    pruned_keep: list[str] = []
    for path in survivors[max(1, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def render_markdown(payload: dict[str, Any]) -> str:
    structure = as_dict(payload.get("structure_signals"))
    return "\n".join(
        [
            "# Crypto Shortline Live Bars Snapshot",
            "",
            f"- brief: `{text(payload.get('snapshot_brief'))}`",
            f"- status: `{text(payload.get('snapshot_status'))}`",
            f"- decision: `{text(payload.get('snapshot_decision'))}`",
            f"- route_symbol: `{text(payload.get('route_symbol'))}`",
            f"- latest_bar_ts: `{text(payload.get('latest_bar_ts'))}`",
            f"- bar_count: `{payload.get('bar_count')}`",
            f"- sweep_long: `{structure.get('sweep_long')}`",
            f"- sweep_short: `{structure.get('sweep_short')}`",
            f"- sweep_long_reference: `{payload.get('sweep_long_reference')}`",
            f"- sweep_short_reference: `{payload.get('sweep_short_reference')}`",
            f"- distance_low_to_sweep_long_bps: `{payload.get('distance_low_to_sweep_long_bps')}`",
            f"- distance_high_to_sweep_short_bps: `{payload.get('distance_high_to_sweep_short_bps')}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def _distance_bps(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return (numerator / denominator) * 10000.0


def _parse_bar_ts(raw: Any) -> dt.datetime | None:
    value = text(raw)
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = dt.datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return None
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


class _TokenBucket:
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


def _http_get_json(*, url: str, timeout_ms: int, bucket: _TokenBucket) -> Any:
    bucket.take()
    req = urllib_request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "fenlie-live-bars/1.0",
        },
        method="GET",
    )

    def _opener(ctx: ssl.SSLContext):
        return urllib_request.build_opener(
            urllib_request.HTTPHandler(),
            urllib_request.HTTPSHandler(context=ctx),
        )

    try:
        with _opener(ssl.create_default_context()).open(
            req, timeout=max(0.1, min(5.0, float(timeout_ms) / 1000.0))
        ) as resp:
            payload = resp.read().decode("utf-8")
    except (TimeoutError, urllib_error.HTTPError, urllib_error.URLError) as exc:
        exc_text = str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in exc_text.upper():
            try:
                with _opener(ssl._create_unverified_context()).open(
                    req, timeout=max(0.1, min(5.0, float(timeout_ms) / 1000.0))
                ) as resp:
                    payload = resp.read().decode("utf-8")
            except (TimeoutError, urllib_error.HTTPError, urllib_error.URLError) as fallback_exc:
                raise RuntimeError(
                    f"live_bars_http_error:{type(fallback_exc).__name__}:{fallback_exc}"
                ) from fallback_exc
        else:
            raise RuntimeError(f"live_bars_http_error:{type(exc).__name__}:{exc}") from exc
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError("live_bars_invalid_json") from exc


def _fetch_binance_futures_bars(
    *,
    symbol: str,
    interval: str,
    limit: int,
    timeout_ms: int,
    rpm: int,
) -> pd.DataFrame:
    query = urllib_parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "limit": max(50, min(1500, int(limit))),
        }
    )
    payload = _http_get_json(
        url=f"https://fapi.binance.com/fapi/v1/klines?{query}",
        timeout_ms=timeout_ms,
        bucket=_TokenBucket(rate_per_minute=max(1, int(rpm))),
    )
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"live_bars_empty_klines:{symbol}")
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        open_ts = pd.to_datetime(int(row[0]), unit="ms", utc=True)
        rows.append(
            {
                "ts": open_ts,
                "symbol": symbol,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "source": "binance_futures_live_pull",
                "asset_class": "crypto",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"live_bars_empty_rows:{symbol}")
    return frame.sort_values("ts").drop_duplicates(subset=["ts", "symbol"]).reset_index(drop=True)


def _select_symbol_frame(frame: pd.DataFrame, route_symbol: str) -> pd.DataFrame:
    if "symbol" not in frame.columns:
        return pd.DataFrame()
    out = frame[frame["symbol"].astype(str).str.upper() == route_symbol].copy()
    if out.empty:
        return out
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def _materialize_symbols(route_symbol: str) -> list[str]:
    ordered: list[str] = []
    for raw in (route_symbol, *DEFAULT_SHORTLINE_TEMPLATE_SYMBOLS):
        symbol = text(raw).upper()
        if symbol and symbol not in ordered:
            ordered.append(symbol)
    return ordered


def _select_materialized_frame(frame: pd.DataFrame, route_symbol: str) -> pd.DataFrame:
    if "symbol" not in frame.columns:
        return pd.DataFrame()
    materialize_set = set(_materialize_symbols(route_symbol))
    out = frame[frame["symbol"].astype(str).str.upper().isin(materialize_set)].copy()
    if out.empty:
        return out
    sort_columns = [column for column in ("symbol", "ts") if column in out.columns]
    if sort_columns:
        out = out.sort_values(sort_columns)
    return out.reset_index(drop=True)


def _latest_bar_age_days(symbol_frame: pd.DataFrame, reference_now: dt.datetime) -> tuple[int | None, bool]:
    if symbol_frame.empty:
        return None, False
    latest_bar_dt = _parse_bar_ts(symbol_frame.iloc[-1].get("ts"))
    if latest_bar_dt is None:
        return None, False
    age_days = max(0, (reference_now.date() - latest_bar_dt.date()).days)
    return age_days, age_days <= MAX_SHORTLINE_BAR_AGE_DAYS


def _load_live_refresh_frame(
    *,
    route_symbol: str,
    materialize_symbols: list[str] | None,
    live_bars_file: Path | None,
    interval: str,
    limit: int,
    timeout_ms: int,
    rpm: int,
) -> tuple[pd.DataFrame, str, str]:
    requested_symbols = materialize_symbols or [route_symbol]
    if live_bars_file is not None and live_bars_file.exists():
        live_frame = read_bars_file(live_bars_file)
        materialized = _select_materialized_frame(live_frame, route_symbol)
        return materialized, "live_bars_fixture", str(live_bars_file)
    frames: list[pd.DataFrame] = []
    fetched_symbols: list[str] = []
    for symbol in requested_symbols:
        try:
            frame = _fetch_binance_futures_bars(
                symbol=symbol,
                interval=interval,
                limit=limit,
                timeout_ms=timeout_ms,
                rpm=rpm,
            )
        except Exception:
            continue
        if frame.empty:
            continue
        frames.append(frame)
        fetched_symbols.append(symbol)
    if not frames:
        raise RuntimeError(f"live_bars_empty_klines:{route_symbol}")
    live_frame = pd.concat(frames, ignore_index=True)
    materialized = _select_materialized_frame(live_frame, route_symbol)
    if materialized.empty or _select_symbol_frame(materialized, route_symbol).empty:
        raise RuntimeError(f"live_bars_route_symbol_missing:{route_symbol}")
    return (
        materialized,
        "binance_futures_live_pull",
        f"binance_futures:{','.join(fetched_symbols)}:{interval}:{int(limit)}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-owned live bars snapshot for crypto shortline event detection."
    )
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--bars-file", default="")
    parser.add_argument("--live-bars-file", default="")
    parser.add_argument("--disable-network-refresh", action="store_true")
    parser.add_argument("--binance-interval", default=DEFAULT_BINANCE_FUTURES_INTERVAL)
    parser.add_argument("--binance-limit", type=int, default=DEFAULT_BINANCE_FUTURES_LIMIT)
    parser.add_argument("--binance-timeout-ms", type=int, default=DEFAULT_BINANCE_TIMEOUT_MS)
    parser.add_argument("--binance-rpm", type=int, default=DEFAULT_BINANCE_RPM)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)

    intent_path = find_latest(review_dir, "*_remote_intent_queue.json")
    operator_path = find_latest(review_dir, "*_crypto_route_operator_brief.json")
    if operator_path is None:
        raise SystemExit("missing_required_artifacts:crypto_route_operator_brief")

    intent_payload = (
        load_json_mapping(intent_path) if intent_path is not None and intent_path.exists() else {}
    )
    operator_payload = load_json_mapping(operator_path)

    route_symbol = (
        text(intent_payload.get("preferred_route_symbol"))
        or text(operator_payload.get("review_priority_head_symbol"))
        or text(operator_payload.get("next_focus_symbol"))
    ).upper()
    route_action = text(intent_payload.get("preferred_route_action")) or text(
        operator_payload.get("next_focus_action")
    )
    remote_market = text(intent_payload.get("remote_market")) or "portfolio_margin_um"

    bars_path = (
        Path(args.bars_file).expanduser().resolve()
        if text(args.bars_file)
        else find_latest_bars_file(output_root, [route_symbol] if route_symbol else [])
    )
    live_bars_file = Path(args.live_bars_file).expanduser().resolve() if text(args.live_bars_file) else None
    bars_input_artifact = str(bars_path) if bars_path else ""
    bars_input_kind = "local_bars_file" if bars_path else ""
    bars_refresh_attempted = False
    bars_refresh_status = "not_attempted"
    bars_refresh_error = ""
    materialized_bars_path: Path | None = None
    materialized_symbols: list[str] = []
    materialized_symbol_count = 0

    snapshot_status = "bars_live_snapshot_missing"
    snapshot_decision = "collect_live_bars_snapshot_before_shortline_event_detection"
    blocker_title = "Collect live bars snapshot before shortline shortline event detection"
    blocker_detail = ""
    done_when = "a route symbol and matching bars snapshot are available"
    latest_bar_ts = ""
    latest_bar_open = None
    latest_bar_high = None
    latest_bar_low = None
    latest_bar_close = None
    latest_bar_volume = None
    bar_count = 0
    latest_bar_age_days = None
    latest_bar_fresh = False
    sweep_long_reference = None
    sweep_short_reference = None
    distance_low_to_sweep_long_bps = None
    distance_high_to_sweep_short_bps = None
    distance_close_to_sweep_long_bps = None
    distance_close_to_sweep_short_bps = None
    structure: dict[str, Any] = {
        "sweep_long": False,
        "sweep_short": False,
        "mss_long": False,
        "mss_short": False,
        "fvg_long": False,
        "fvg_short": False,
    }

    if not route_symbol:
        snapshot_status = "route_symbol_missing_for_live_bars_snapshot"
        blocker_title = "Resolve route symbol before collecting live bars snapshot"
        blocker_detail = "route_symbol_missing"
    elif bars_path is None or not bars_path.exists():
        blocker_detail = f"bars_artifact_missing:{route_symbol}"
        if not text(args.bars_file) and not bool(args.disable_network_refresh):
            bars_refresh_attempted = True
            try:
                live_frame, bars_input_kind, bars_input_artifact = _load_live_refresh_frame(
                    route_symbol=route_symbol,
                    materialize_symbols=_materialize_symbols(route_symbol),
                    live_bars_file=live_bars_file,
                    interval=str(args.binance_interval),
                    limit=int(args.binance_limit),
                    timeout_ms=int(args.binance_timeout_ms),
                    rpm=int(args.binance_rpm),
                )
                symbol_frame = _select_symbol_frame(live_frame, route_symbol)
                if not symbol_frame.empty:
                    bars_refresh_status = "live_refresh_used"
                    materialized_frame = live_frame.copy()
                    materialized_bars_path = review_dir / (
                        f"{reference_now.strftime('%Y%m%dT%H%M%SZ')}_crypto_shortline_live_bars_snapshot_bars.csv"
                    )
                    materialized_frame.to_csv(materialized_bars_path, index=False)
                    materialized_symbols = sorted(
                        {
                            text(raw).upper()
                            for raw in materialized_frame.get("symbol", pd.Series(dtype=str)).astype(str).tolist()
                            if text(raw)
                        }
                    )
                    materialized_symbol_count = len(materialized_symbols)
                    bars_path = materialized_bars_path
                else:
                    bars_refresh_status = "live_refresh_symbol_missing"
            except Exception as exc:
                bars_refresh_status = "live_refresh_failed"
                bars_refresh_error = f"{type(exc).__name__}:{exc}"
        else:
            bars_refresh_status = (
                "network_refresh_disabled" if bool(args.disable_network_refresh) else "bars_input_missing"
            )
        if materialized_bars_path is not None and materialized_bars_path.exists():
            frame = read_bars_file(materialized_bars_path)
            symbol_frame = _select_symbol_frame(frame, route_symbol)
            latest = symbol_frame.iloc[-1]
            structure = structure_signals(symbol_frame)
            enriched = add_common_features(symbol_frame)
            enriched_latest = enriched.iloc[-1]
            latest_bar_ts = text(latest.get("ts"))
            latest_bar_open = latest.get("open")
            latest_bar_high = latest.get("high")
            latest_bar_low = latest.get("low")
            latest_bar_close = latest.get("close")
            latest_bar_volume = latest.get("volume")
            bar_count = int(len(symbol_frame))
            latest_bar_dt = _parse_bar_ts(latest_bar_ts)
            if latest_bar_dt is not None:
                latest_bar_age_days = max(0, (reference_now.date() - latest_bar_dt.date()).days)
                latest_bar_fresh = latest_bar_age_days <= MAX_SHORTLINE_BAR_AGE_DAYS
            close_value = float(enriched_latest.get("close") or 0.0)
            high_value = float(enriched_latest.get("high") or 0.0)
            low_value = float(enriched_latest.get("low") or 0.0)
            sweep_long_reference = float(enriched_latest.get("roll_low10_prev") or 0.0)
            sweep_short_reference = float(enriched_latest.get("roll_high10_prev") or 0.0)
            distance_low_to_sweep_long_bps = _distance_bps(
                low_value - sweep_long_reference,
                close_value,
            )
            distance_high_to_sweep_short_bps = _distance_bps(
                sweep_short_reference - high_value,
                close_value,
            )
            distance_close_to_sweep_long_bps = _distance_bps(
                close_value - sweep_long_reference,
                close_value,
            )
            distance_close_to_sweep_short_bps = _distance_bps(
                sweep_short_reference - close_value,
                close_value,
            )
            snapshot_status = "bars_live_snapshot_ready"
            snapshot_decision = "use_live_bars_snapshot_for_shortline_event_detection"
            blocker_title = "Use live bars snapshot for shortline event detection"
            done_when = f"{route_symbol} bars snapshot stays fresh enough for shortline event detection"
            if latest_bar_fresh is False:
                snapshot_status = "bars_live_snapshot_stale"
                snapshot_decision = "refresh_live_bars_snapshot_before_shortline_event_detection"
                blocker_title = "Refresh live bars snapshot before shortline event detection"
                done_when = f"{route_symbol} latest bar age returns to <= {MAX_SHORTLINE_BAR_AGE_DAYS} day(s)"
            blocker_detail = " | ".join(
                [
                    f"bars_artifact={materialized_bars_path.name}",
                    f"bars_input_kind={bars_input_kind}",
                    f"bars_input_artifact={bars_input_artifact or '-'}",
                    f"bars_refresh_status={bars_refresh_status}",
                    (f"bars_refresh_error={bars_refresh_error}" if bars_refresh_error else ""),
                    f"bar_count={bar_count}",
                    f"latest_bar_ts={latest_bar_ts or '-'}",
                    (
                        f"latest_bar_age_days={latest_bar_age_days}"
                        if latest_bar_age_days is not None
                        else ""
                    ),
                    f"latest_bar_fresh={str(bool(latest_bar_fresh)).lower()}",
                    f"sweep_long={str(bool(structure.get('sweep_long', False))).lower()}",
                    f"sweep_short={str(bool(structure.get('sweep_short', False))).lower()}",
                    f"sweep_long_reference={sweep_long_reference:g}",
                    f"sweep_short_reference={sweep_short_reference:g}",
                    (
                        f"distance_low_to_sweep_long_bps={distance_low_to_sweep_long_bps:g}"
                        if distance_low_to_sweep_long_bps is not None
                        else ""
                    ),
                    (
                        f"distance_high_to_sweep_short_bps={distance_high_to_sweep_short_bps:g}"
                        if distance_high_to_sweep_short_bps is not None
                        else ""
                    ),
                ]
            )
    else:
        try:
            frame = read_bars_file(bars_path)
        except Exception as exc:
            snapshot_status = "bars_live_snapshot_unreadable"
            blocker_title = "Repair bars snapshot readability before shortline event detection"
            blocker_detail = f"bars_read_error:{type(exc).__name__}"
        else:
            if "symbol" not in frame.columns:
                snapshot_status = "bars_live_snapshot_symbol_column_missing"
                blocker_title = "Repair bars snapshot schema before shortline event detection"
                blocker_detail = "bars_symbol_column_missing"
            else:
                symbol_frame = frame[frame["symbol"].astype(str).str.upper() == route_symbol].copy()
                if symbol_frame.empty:
                    snapshot_status = "route_symbol_missing_in_live_bars_snapshot"
                    blocker_title = "Collect route symbol bars before shortline event detection"
                    blocker_detail = (
                        f"bars_symbol_missing:{route_symbol}:source={bars_path.name}"
                    )
                else:
                    selected_frame = symbol_frame
                    selected_source_kind = "local_bars_file"
                    selected_source_artifact = str(bars_path)
                    local_age_days, local_is_fresh = _latest_bar_age_days(symbol_frame, reference_now)
                    if (
                        not text(args.bars_file)
                        and not bool(args.disable_network_refresh)
                        and route_symbol
                        and (local_is_fresh is False)
                    ):
                        bars_refresh_attempted = True
                        try:
                            live_frame, selected_source_kind, selected_source_artifact = _load_live_refresh_frame(
                                route_symbol=route_symbol,
                                materialize_symbols=_materialize_symbols(route_symbol),
                                live_bars_file=live_bars_file,
                                interval=str(args.binance_interval),
                                limit=int(args.binance_limit),
                                timeout_ms=int(args.binance_timeout_ms),
                                rpm=int(args.binance_rpm),
                            )
                            live_symbol_frame = _select_symbol_frame(live_frame, route_symbol)
                            live_age_days, live_is_fresh = _latest_bar_age_days(
                                live_symbol_frame, reference_now
                            )
                            if not live_symbol_frame.empty and (
                                live_is_fresh
                                or local_age_days is None
                                or (live_age_days is not None and live_age_days < local_age_days)
                            ):
                                selected_frame = live_symbol_frame
                                bars_refresh_status = "live_refresh_used"
                                bars_input_kind = selected_source_kind
                                bars_input_artifact = selected_source_artifact
                            else:
                                bars_refresh_status = "live_refresh_not_fresher_than_local"
                                bars_input_kind = "local_bars_file"
                                bars_input_artifact = str(bars_path)
                        except Exception as exc:
                            bars_refresh_status = "live_refresh_failed"
                            bars_refresh_error = f"{type(exc).__name__}:{exc}"
                            bars_input_kind = "local_bars_file"
                            bars_input_artifact = str(bars_path)
                    else:
                        bars_input_kind = "explicit_bars_file" if text(args.bars_file) else "local_bars_file"
                        bars_refresh_status = (
                            "network_refresh_disabled"
                            if bool(args.disable_network_refresh)
                            else "local_bars_fresh_enough"
                        )

                    materialized_bars_path = review_dir / (
                        f"{reference_now.strftime('%Y%m%dT%H%M%SZ')}_crypto_shortline_live_bars_snapshot_bars.csv"
                    )
                    materialized_frame = (
                        _select_materialized_frame(frame, route_symbol)
                        if selected_source_kind in {"local_bars_file", "explicit_bars_file"}
                        else (
                            live_frame.copy()
                            if "live_frame" in locals() and isinstance(live_frame, pd.DataFrame)
                            else selected_frame.copy()
                        )
                    )
                    if materialized_frame.empty:
                        materialized_frame = selected_frame.copy()
                    materialized_frame.to_csv(materialized_bars_path, index=False)
                    materialized_symbols = sorted(
                        {
                            text(raw).upper()
                            for raw in materialized_frame.get("symbol", pd.Series(dtype=str)).astype(str).tolist()
                            if text(raw)
                        }
                    )
                    materialized_symbol_count = len(materialized_symbols)
                    symbol_frame = selected_frame
                    symbol_frame = symbol_frame.sort_values("ts").reset_index(drop=True)
                    latest = symbol_frame.iloc[-1]
                    structure = structure_signals(symbol_frame)
                    enriched = add_common_features(symbol_frame)
                    enriched_latest = enriched.iloc[-1]
                    latest_bar_ts = text(latest.get("ts"))
                    latest_bar_open = latest.get("open")
                    latest_bar_high = latest.get("high")
                    latest_bar_low = latest.get("low")
                    latest_bar_close = latest.get("close")
                    latest_bar_volume = latest.get("volume")
                    bar_count = int(len(symbol_frame))
                    latest_bar_dt = _parse_bar_ts(latest_bar_ts)
                    if latest_bar_dt is not None:
                        latest_bar_age_days = max(
                            0,
                            (reference_now.date() - latest_bar_dt.date()).days,
                        )
                        latest_bar_fresh = latest_bar_age_days <= MAX_SHORTLINE_BAR_AGE_DAYS
                    close_value = float(enriched_latest.get("close") or 0.0)
                    high_value = float(enriched_latest.get("high") or 0.0)
                    low_value = float(enriched_latest.get("low") or 0.0)
                    sweep_long_reference = float(
                        enriched_latest.get("roll_low10_prev") or 0.0
                    )
                    sweep_short_reference = float(
                        enriched_latest.get("roll_high10_prev") or 0.0
                    )
                    distance_low_to_sweep_long_bps = _distance_bps(
                        low_value - sweep_long_reference,
                        close_value,
                    )
                    distance_high_to_sweep_short_bps = _distance_bps(
                        sweep_short_reference - high_value,
                        close_value,
                    )
                    distance_close_to_sweep_long_bps = _distance_bps(
                        close_value - sweep_long_reference,
                        close_value,
                    )
                    distance_close_to_sweep_short_bps = _distance_bps(
                        sweep_short_reference - close_value,
                        close_value,
                    )
                    snapshot_status = "bars_live_snapshot_ready"
                    snapshot_decision = "use_live_bars_snapshot_for_shortline_event_detection"
                    blocker_title = "Use live bars snapshot for shortline event detection"
                    done_when = (
                        f"{route_symbol} bars snapshot stays fresh enough for shortline event detection"
                    )
                    if latest_bar_fresh is False:
                        snapshot_status = "bars_live_snapshot_stale"
                        snapshot_decision = (
                            "refresh_live_bars_snapshot_before_shortline_event_detection"
                        )
                        blocker_title = (
                            "Refresh live bars snapshot before shortline event detection"
                        )
                        done_when = (
                            f"{route_symbol} latest bar age returns to <= {MAX_SHORTLINE_BAR_AGE_DAYS} day(s)"
                        )
                    blocker_detail = " | ".join(
                        [
                            f"bars_artifact={(materialized_bars_path or bars_path).name}",
                            f"bars_input_kind={bars_input_kind}",
                            f"bars_input_artifact={bars_input_artifact or '-'}",
                            f"bars_refresh_status={bars_refresh_status}",
                            (f"bars_refresh_error={bars_refresh_error}" if bars_refresh_error else ""),
                            f"bar_count={bar_count}",
                            f"latest_bar_ts={latest_bar_ts or '-'}",
                            (
                                f"latest_bar_age_days={latest_bar_age_days}"
                                if latest_bar_age_days is not None
                                else ""
                            ),
                            f"latest_bar_fresh={str(bool(latest_bar_fresh)).lower()}",
                            f"sweep_long={str(bool(structure.get('sweep_long', False))).lower()}",
                            f"sweep_short={str(bool(structure.get('sweep_short', False))).lower()}",
                            f"sweep_long_reference={sweep_long_reference:g}",
                            f"sweep_short_reference={sweep_short_reference:g}",
                            (
                                f"distance_low_to_sweep_long_bps={distance_low_to_sweep_long_bps:g}"
                                if distance_low_to_sweep_long_bps is not None
                                else ""
                            ),
                            (
                                f"distance_high_to_sweep_short_bps={distance_high_to_sweep_short_bps:g}"
                                if distance_high_to_sweep_short_bps is not None
                                else ""
                            ),
                            f"mss_long={str(bool(structure.get('mss_long', False))).lower()}",
                            f"mss_short={str(bool(structure.get('mss_short', False))).lower()}",
                        ]
                    )

    snapshot_brief = ":".join(
        [snapshot_status, route_symbol or "-", snapshot_decision, remote_market or "-"]
    )
    payload = {
        "action": "build_crypto_shortline_live_bars_snapshot",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "route_symbol": route_symbol,
        "route_action": route_action,
        "remote_market": remote_market,
        "snapshot_status": snapshot_status,
        "snapshot_brief": snapshot_brief,
        "snapshot_decision": snapshot_decision,
        "signal_source": "crypto_shortline_live_bars_snapshot",
        "bars_artifact": str(materialized_bars_path or bars_path or ""),
        "bars_input_artifact": bars_input_artifact,
        "bars_input_kind": bars_input_kind,
        "bars_refresh_attempted": bars_refresh_attempted,
        "bars_refresh_status": bars_refresh_status,
        "bars_refresh_error": bars_refresh_error,
        "materialized_symbols": materialized_symbols,
        "materialized_symbol_count": materialized_symbol_count,
        "latest_bar_ts": latest_bar_ts,
        "latest_bar_open": latest_bar_open,
        "latest_bar_high": latest_bar_high,
        "latest_bar_low": latest_bar_low,
        "latest_bar_close": latest_bar_close,
        "latest_bar_volume": latest_bar_volume,
        "bar_count": bar_count,
        "latest_bar_age_days": latest_bar_age_days,
        "latest_bar_fresh": latest_bar_fresh,
        "max_bar_age_days": MAX_SHORTLINE_BAR_AGE_DAYS,
        "sweep_long_reference": sweep_long_reference,
        "sweep_short_reference": sweep_short_reference,
        "distance_low_to_sweep_long_bps": distance_low_to_sweep_long_bps,
        "distance_high_to_sweep_short_bps": distance_high_to_sweep_short_bps,
        "distance_close_to_sweep_long_bps": distance_close_to_sweep_long_bps,
        "distance_close_to_sweep_short_bps": distance_close_to_sweep_short_bps,
        "structure_signals": structure,
        "blocker_title": blocker_title,
        "blocker_target_artifact": "crypto_shortline_live_bars_snapshot",
        "blocker_detail": blocker_detail,
        "next_action": snapshot_decision,
        "next_action_target_artifact": "crypto_shortline_live_bars_snapshot",
        "done_when": done_when,
        "artifacts": {
            "remote_intent_queue": str(intent_path) if intent_path else "",
            "crypto_route_operator_brief": str(operator_path),
            "bars_live_snapshot": str(materialized_bars_path or bars_path or ""),
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_live_bars_snapshot.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_live_bars_snapshot.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_live_bars_snapshot_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "markdown": str(markdown),
        "markdown_sha256": sha256_file(markdown),
        "generated_at_utc": fmt_utc(reference_now),
    }
    if materialized_bars_path is not None and materialized_bars_path.exists():
        checksum_payload["bars_artifact"] = str(materialized_bars_path)
        checksum_payload["bars_artifact_sha256"] = sha256_file(materialized_bars_path)
    checksum.write_text(
        json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_paths=[artifact, markdown, checksum],
        keep=args.artifact_keep,
        ttl_hours=args.artifact_ttl_hours,
    )
    payload.update(
        {
            "artifact": str(artifact),
            "markdown": str(markdown),
            "checksum": str(checksum),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
