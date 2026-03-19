#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
from typing import Any

import pandas as pd


def resolve_system_root() -> Path:
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
sys.path.insert(0, str(SYSTEM_ROOT / "src"))

DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
COMBO_CANONICAL_ALIASES = {
    "ad_breakout": "cvd_breakout",
    "ad_rsi_breakout": "cvd_rsi_breakout",
    "ad_rsi_vol_breakout": "cvd_rsi_vol_breakout",
    "ad_rsi_reclaim": "cvd_rsi_reclaim",
    "taker_oi_ad_breakout": "taker_oi_cvd_breakout",
    "taker_oi_ad_rsi_breakout": "taker_oi_cvd_rsi_breakout",
}


def load_combo_module():
    script_path = SYSTEM_ROOT / "scripts" / "backtest_binance_indicator_combo_etf.py"
    spec = importlib.util.spec_from_file_location("indicator_combo_etf_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable_to_load_indicator_combo_module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


combo = load_combo_module()


class DeadlineExceeded(TimeoutError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk_data in iter(lambda: handle.read(65536), b""):
            digest.update(chunk_data)
    return digest.hexdigest()


def error_label(exc: BaseException) -> str:
    return f"{type(exc).__name__}:{exc}"


@contextmanager
def deadline_guard(timeout_seconds: float, reason: str):
    seconds = max(0.0, float(timeout_seconds or 0.0))
    if seconds <= 0.0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.getitimer(signal.ITIMER_REAL)

    def _handle_timeout(signum, frame):  # noqa: ANN001, ARG001
        raise DeadlineExceeded(f"{reason}:timeout_seconds_exceeded:{seconds:g}")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0.0 or previous_interval > 0.0:
            signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)


def canonical_combo_id(combo_id: Any) -> str:
    combo_id_text = str(combo_id or "")
    return COMBO_CANONICAL_ALIASES.get(combo_id_text, combo_id_text)


def with_canonical_combo_ids(family_payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(family_payload)
    for key in ("ranked_combos", "discarded_combos"):
        rows = []
        for row in list(enriched.get(key, []) or []):
            if not isinstance(row, dict):
                rows.append(row)
                continue
            updated = dict(row)
            updated["combo_id_canonical"] = canonical_combo_id(updated.get("combo_id"))
            rows.append(updated)
        enriched[key] = rows
    return enriched


def safe_group_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "custom"


def artifact_stem_for_group(group_slug: str) -> str:
    return f"{group_slug}_binance_indicator_combo_native_crypto"


def fetch_futures_klines(
    symbol: str,
    *,
    interval: str,
    limit: int,
    timeout_ms: int,
    bucket: Any,
) -> pd.DataFrame:
    query = {
        "symbol": symbol,
        "interval": interval,
        "limit": max(50, min(1500, int(limit))),
    }
    url = "https://fapi.binance.com/fapi/v1/klines?" + combo.urllib.parse.urlencode(query)
    payload = combo.request_json(url=url, bucket=bucket, timeout_ms=min(5000, max(100, int(timeout_ms))))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"empty_klines:{symbol}")

    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        open_ts = pd.to_datetime(int(row[0]), unit="ms", utc=True)
        rows.append(
            {
                "ts": open_ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"empty_klines_rows:{symbol}")
    frame = frame.sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True)
    return frame


def build_symbol_frame(
    symbol: str,
    *,
    interval: str,
    lookback_bars: int,
    binance_period: str,
    binance_limit: int,
    timeout_ms: int,
    bucket: Any,
    symbol_timeout_seconds: float,
) -> pd.DataFrame:
    with deadline_guard(symbol_timeout_seconds, f"symbol_timeout:{symbol}"):
        klines = fetch_futures_klines(
            symbol,
            interval=interval,
            limit=lookback_bars,
            timeout_ms=timeout_ms,
            bucket=bucket,
        )
        indicator_series = combo.fetch_binance_indicator_series(
            symbol,
            period=binance_period,
            limit=binance_limit,
            timeout_ms=timeout_ms,
            bucket=bucket,
        )
        return combo.build_market_frame(symbol, klines, f"BINANCE_FUTURES:{symbol}", indicator_series)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Binance Indicator Combo Native Crypto Backtest",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- status: `{payload.get('status') or ''}`",
        f"- symbol group: `{payload.get('symbol_group')}`",
        f"- interval: `{payload.get('interval')}`",
        f"- lookback bars: `{payload.get('lookback_bars')}`",
        f"- sample windows: `{payload.get('sample_windows')}`",
        f"- hold bars: `{payload.get('hold_bars')}`",
        "",
        "## Coverage",
    ]
    for row in payload.get("coverage", []):
        lines.append(f"- `{row.get('symbol')}` bars=`{row.get('rows')}`")

    error_items = list(payload.get("error_items") or [])
    if error_items:
        lines.extend(["", "## Failures"])
        for row in error_items:
            lines.append(
                f"- stage=`{row.get('stage') or ''}` symbol=`{row.get('symbol') or '-'}` status=`{row.get('status') or ''}` error=`{row.get('error') or ''}`"
            )

    lines.extend(["", "## Ranked Native Crypto Combos"])
    for row in payload.get("native_crypto_family", {}).get("ranked_combos", [])[:10]:
        lines.append(
            f"- `{row.get('combo_id')}` canonical=`{row.get('combo_id_canonical')}` return=`{float(row.get('avg_total_return') or 0.0):.4f}` "
            f"win=`{float(row.get('avg_win_rate') or 0.0):.2%}` lag=`{row.get('avg_lag_bars')}` "
            f"timely=`{float(row.get('avg_timely_hit_rate') or 0.0):.2%}` trades=`{int(row.get('trade_count') or 0)}`"
        )

    lines.extend(["", "## Discarded"])
    for row in payload.get("native_crypto_family", {}).get("discarded_combos", []):
        lines.append(
            f"- `{row.get('combo_id')}` canonical=`{row.get('combo_id_canonical')}` reason=`{row.get('discard_reason')}` "
            f"lag=`{row.get('avg_lag_bars')}` timely=`{float(row.get('avg_timely_hit_rate') or 0.0):.2%}` trades=`{int(row.get('trade_count') or 0)}`"
        )

    lines.extend(
        [
            "",
            "## Conclusions",
            f"- measured takeaway: {payload.get('native_crypto_takeaway')}",
            f"- practitioner note: {payload.get('native_crypto_practitioner_note')}",
            f"- control note: {payload.get('control_note')}",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def build_native_takeaway(family_payload: dict[str, Any]) -> tuple[str, str]:
    ranked = list(family_payload.get("ranked_combos", []))
    discarded = list(family_payload.get("discarded_combos", []))
    if not ranked:
        return (
            "No native crypto combo survived the lag filter.",
            "Do not promote Binance flow metrics until native 24/7 bars produce at least one timing-qualified setup.",
        )

    top = ranked[0]
    combo_id = str(top.get("combo_id") or "")
    combo_id_canonical = canonical_combo_id(combo_id)
    avg_total_return = float(top.get("avg_total_return") or 0.0)
    timely = float(top.get("avg_timely_hit_rate") or 0.0)
    trades = int(top.get("trade_count") or 0)
    ranked_ids = [str(row.get("combo_id") or "") for row in ranked]
    taker_ranked = [combo_id for combo_id in ranked_ids if combo_id.startswith("taker_oi") or combo_id.startswith("crowding_filtered")]
    taker_discarded = [
        str(row.get("combo_id") or "")
        for row in discarded
        if str(row.get("combo_id") or "").startswith("taker_oi") or str(row.get("combo_id") or "").startswith("crowding_filtered")
    ]

    if combo_id.startswith("taker_oi") or combo_id.startswith("crowding_filtered"):
        measured = (
            f"On native 24/7 Binance futures bars, `{combo_id_canonical}` ranked first with return={avg_total_return:.4f}, timely-hit={timely:.2%}, trades={trades}. "
            "That means venue-native flow metrics can lead once the ETF session mismatch is removed."
        )
    elif taker_ranked:
        measured = (
            f"On native 24/7 Binance futures bars, `{combo_id_canonical}` still ranked first with return={avg_total_return:.4f}, timely-hit={timely:.2%}, trades={trades}. "
            "But taker/OI stacks now survive the lag filter and enter the ranked set, which means source alignment improved them even though they still trail price-state breakout timing."
        )
    elif taker_discarded:
        measured = (
            f"Switching to native 24/7 Binance futures bars did not rescue taker/OI timing. `{combo_id_canonical}` still led with return={avg_total_return:.4f}, timely-hit={timely:.2%}, trades={trades}, "
            "while flow-heavy stacks remained discarded."
        )
    else:
        measured = (
            f"`{combo_id_canonical}` led the native sample with return={avg_total_return:.4f}, timely-hit={timely:.2%}, trades={trades}."
        )

    practitioner = (
        "Practitioner prior still supports using taker imbalance plus open interest as force confirmation, with CVD-lite as the canonical price-state confirmation layer. This sample still says the best role is secondary confirmation unless a native flow stack actually outranks the CVD-lite price-state breakout."
    )
    return measured, practitioner


def build_partial_takeaway(
    *,
    completed_symbols: list[str],
    error_items: list[dict[str, Any]],
) -> tuple[str, str]:
    timed_out_symbols = [str(row.get("symbol") or "") for row in error_items if str(row.get("status") or "") == "timed_out"]
    failed_symbols = [str(row.get("symbol") or "") for row in error_items if str(row.get("status") or "") == "failed"]
    if completed_symbols:
        measured = (
            f"Native crypto backtest completed only {len(completed_symbols)} symbol(s) before entering partial failure. "
            f"Completed={', '.join(completed_symbols)}; timed_out={', '.join(x for x in timed_out_symbols if x) or '-'}; failed={', '.join(x for x in failed_symbols if x) or '-'}."
        )
    else:
        measured = (
            f"Native crypto backtest ended in partial failure before any full symbol frame completed. "
            f"timed_out={', '.join(x for x in timed_out_symbols if x) or '-'}; failed={', '.join(x for x in failed_symbols if x) or '-'}."
        )
    practitioner = (
        "Treat this artifact as diagnostic only. Retry after clearing the timeout/failure path; do not use a partial native source artifact to promote flow-secondary routes."
    )
    return measured, practitioner


def write_artifacts(
    *,
    review_dir: Path,
    runtime_now: dt.datetime,
    group_stem: str,
    artifact_keep: int,
    artifact_ttl_hours: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_{group_stem}.json"
    md_path = review_dir / f"{stamp}_{group_stem}.md"
    checksum_path = review_dir / f"{stamp}_{group_stem}_checksum.json"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    checksum_payload = {
        "artifact": str(json_path),
        "artifact_sha256": sha256_file(json_path),
        "markdown": str(md_path),
        "markdown_sha256": sha256_file(md_path),
        "generated_at": fmt_utc(runtime_now),
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pruned_keep, pruned_age = combo.prune_review_artifacts(
        review_dir,
        stem=group_stem,
        current_paths=[json_path, md_path, checksum_path],
        keep=max(3, int(artifact_keep)),
        ttl_hours=max(24.0, float(artifact_ttl_hours)),
    )
    payload["artifact"] = str(json_path)
    payload["markdown"] = str(md_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["artifact_sha256"] = sha256_file(json_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-test Binance indicator combinations on native 24/7 futures bars.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT")
    parser.add_argument("--symbol-group", default="custom")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback-bars", type=int, default=300)
    parser.add_argument("--sample-windows", type=int, default=3)
    parser.add_argument("--window-bars", type=int, default=40)
    parser.add_argument("--hold-bars", type=int, default=4)
    parser.add_argument("--binance-limit", type=int, default=300)
    parser.add_argument("--binance-period", default="1h")
    parser.add_argument("--rpm", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--per-symbol-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--summarize-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--now", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    runtime_now = dt.datetime.fromisoformat(args.now) if str(args.now).strip() else now_utc()

    bucket = combo.TokenBucket(rate_per_minute=max(1, int(args.rpm)), capacity=5)
    symbols = [item.strip().upper() for item in str(args.symbols).split(",") if item.strip()]
    group_slug = safe_group_slug(str(args.symbol_group))
    group_stem = artifact_stem_for_group(group_slug)

    coverage: list[dict[str, Any]] = []
    asset_frames: dict[str, pd.DataFrame] = {}
    error_items: list[dict[str, Any]] = []
    completed_symbols: list[str] = []
    timed_out_symbols: list[str] = []
    failed_symbols: list[str] = []
    for symbol in symbols:
        try:
            frame = build_symbol_frame(
                symbol,
                interval=str(args.interval),
                lookback_bars=max(100, int(args.lookback_bars)),
                binance_period=str(args.binance_period),
                binance_limit=max(100, int(args.binance_limit)),
                timeout_ms=min(5000, max(100, int(args.timeout_ms))),
                bucket=bucket,
                symbol_timeout_seconds=max(1.0, float(args.per_symbol_timeout_seconds)),
            )
        except DeadlineExceeded as exc:
            timed_out_symbols.append(symbol)
            error_items.append(
                {
                    "stage": "symbol_frame",
                    "symbol": symbol,
                    "status": "timed_out",
                    "error": str(exc),
                }
            )
            continue
        except Exception as exc:  # noqa: BLE001
            failed_symbols.append(symbol)
            error_items.append(
                {
                    "stage": "symbol_frame",
                    "symbol": symbol,
                    "status": "failed",
                    "error": error_label(exc),
                }
            )
            continue
        asset_frames[symbol] = frame
        completed_symbols.append(symbol)
        coverage.append({"symbol": symbol, "rows": int(len(frame))})

    native_family: dict[str, Any] = {"ranked_combos": [], "discarded_combos": []}
    if asset_frames:
        try:
            with deadline_guard(max(1.0, float(args.summarize_timeout_seconds)), "summarize_timeout"):
                native_family = with_canonical_combo_ids(
                    combo.summarize_family(
                        "crypto",
                        asset_frames,
                        hold_bars=max(2, int(args.hold_bars)),
                        sample_windows=max(1, int(args.sample_windows)),
                        window_bars=max(20, int(args.window_bars)),
                    )
                )
        except DeadlineExceeded as exc:
            error_items.append(
                {
                    "stage": "summarize_family",
                    "symbol": "",
                    "status": "timed_out",
                    "error": str(exc),
                }
            )
        except Exception as exc:  # noqa: BLE001
            error_items.append(
                {
                    "stage": "summarize_family",
                    "symbol": "",
                    "status": "failed",
                    "error": error_label(exc),
                }
            )

    if error_items:
        measured_takeaway, practitioner_note = build_partial_takeaway(
            completed_symbols=completed_symbols,
            error_items=error_items,
        )
        status = "partial_failure"
        ok = False
    else:
        measured_takeaway, practitioner_note = build_native_takeaway(native_family)
        status = "ok"
        ok = True

    payload = {
        "ok": ok,
        "status": status,
        "as_of": fmt_utc(runtime_now),
        "symbol_group": str(args.symbol_group),
        "interval": str(args.interval),
        "lookback_bars": int(args.lookback_bars),
        "sample_windows": int(args.sample_windows),
        "window_bars": int(args.window_bars),
        "hold_bars": int(args.hold_bars),
        "per_symbol_timeout_seconds": max(1.0, float(args.per_symbol_timeout_seconds)),
        "summarize_timeout_seconds": max(1.0, float(args.summarize_timeout_seconds)),
        "coverage": coverage,
        "completed_symbols": completed_symbols,
        "completed_symbol_count": len(completed_symbols),
        "timed_out_symbols": timed_out_symbols,
        "timed_out_symbol_count": len(timed_out_symbols),
        "failed_symbols": failed_symbols,
        "failed_symbol_count": len(failed_symbols),
        "error_items": error_items,
        "error_count": len(error_items),
        "source_notes": combo.SOURCE_NOTES
        + [
            {
                "kind": "official",
                "label": "Binance USDⓈ-M Futures Kline/Candlestick Data",
                "url": "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data",
                "takeaway": "Native futures klines remove ETF session mismatch when evaluating Binance flow metrics.",
            }
        ],
        "native_crypto_family": native_family,
        "native_crypto_takeaway": measured_takeaway,
        "native_crypto_practitioner_note": practitioner_note,
        "control_note": "This rerun keeps combo definitions, hold bars, lag filter, and sample windows fixed; only the price source changes from ETF proxies to native Binance futures bars.",
        "artifact_label": f"binance-indicator-combo-native-crypto:{str(args.symbol_group)}:{status}",
    }
    payload = write_artifacts(
        review_dir=review_dir,
        runtime_now=runtime_now,
        group_stem=group_stem,
        artifact_keep=int(args.artifact_keep),
        artifact_ttl_hours=float(args.artifact_ttl_hours),
        payload=payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
