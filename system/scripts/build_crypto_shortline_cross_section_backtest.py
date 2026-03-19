#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_system_root() -> Path:
    env_root = str(os.getenv("LIE_SYSTEM_ROOT", "")).strip() or str(
        os.getenv("FENLIE_SYSTEM_ROOT", "")
    ).strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


SYSTEM_ROOT = resolve_system_root()
DEFAULT_REVIEW_DIR = SYSTEM_ROOT / "output" / "review"
DEFAULT_LOOKBACK_DAYS = 14
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lie_engine.data.providers import BinanceSpotPublicProvider


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


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dedupe_text(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def join_unique(parts: list[Any], *, sep: str = " | ") -> str:
    return sep.join(dedupe_text(parts))


def load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def find_latest(review_dir: Path, pattern: str) -> Path | None:
    files = sorted(review_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


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
        "*_crypto_shortline_cross_section_backtest.json",
        "*_crypto_shortline_cross_section_backtest.md",
        "*_crypto_shortline_cross_section_backtest_checksum.json",
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


def timeframe_minutes(raw: str) -> int:
    value = text(raw).lower()
    if value.endswith("m"):
        return max(1, int(value[:-1] or "1"))
    if value.endswith("h"):
        return max(60, int(value[:-1] or "1") * 60)
    if value.endswith("d"):
        return max(1440, int(value[:-1] or "1") * 1440)
    raise ValueError(f"unsupported_timeframe:{raw}")


def horizon_minutes_from_policy(payload: dict[str, Any]) -> list[int]:
    holding = as_dict(payload.get("holding_window_minutes"))
    minimum = max(15, int(holding.get("min") or 15))
    maximum = max(minimum, int(holding.get("max") or 180))
    candidates = [minimum, minimum * 2, minimum * 4, maximum]
    bounded = [value for value in candidates if minimum <= value <= maximum]
    if maximum not in bounded:
        bounded.append(maximum)
    return sorted({int(value) for value in bounded})


def structure_window_bars(payload: dict[str, Any], execution_minutes: int) -> int:
    structure_tf = text(payload.get("structure_timeframe")) or "4h"
    try:
        structure_minutes = timeframe_minutes(structure_tf)
    except ValueError:
        structure_minutes = 240
    return max(8, int(math.ceil(structure_minutes / max(1, execution_minutes))))


def load_fixture_bars(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("bars") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("bars_fixture_missing_rows")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    required = {"ts", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bars_fixture_missing_columns:{','.join(sorted(missing))}")
    frame = frame.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce").dt.tz_convert(None)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame = frame.dropna(subset=["ts", "symbol", "open", "high", "low", "close", "volume"])
    return frame.sort_values(["symbol", "ts"]).reset_index(drop=True)


def fetch_universe_bars(
    *,
    universe: list[str],
    start: dt.date,
    end: dt.date,
    execution_timeframe: str,
    cutoff: dt.datetime,
    bars_fixture: Path | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if bars_fixture is not None:
        fixture_frame = load_fixture_bars(bars_fixture)
        filtered = fixture_frame[
            (fixture_frame["symbol"].isin(universe))
            & (fixture_frame["ts"] >= pd.Timestamp(start))
            & (fixture_frame["ts"] <= pd.Timestamp(cutoff.replace(tzinfo=None)))
        ].copy()
        errors = {
            symbol: "fixture_missing_symbol"
            for symbol in universe
            if filtered[filtered["symbol"] == symbol].empty
        }
        return filtered, errors

    provider = BinanceSpotPublicProvider(request_timeout_ms=5000, rate_limit_per_minute=20)
    cutoff_naive = cutoff.astimezone(dt.timezone.utc).replace(tzinfo=None)
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    for symbol in universe:
        try:
            frame = provider.fetch_ohlcv(symbol=symbol, start=start, end=end, freq=execution_timeframe)
        except Exception as exc:  # noqa: BLE001
            errors[symbol] = f"{type(exc).__name__}:{exc}"
            continue
        if frame is None or frame.empty:
            errors[symbol] = "empty_bars"
            continue
        frame = frame.copy()
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        frame = frame[(frame["ts"] >= pd.Timestamp(start)) & (frame["ts"] <= pd.Timestamp(cutoff_naive))]
        if frame.empty:
            errors[symbol] = "empty_after_cutoff_filter"
            continue
        frames.append(frame[["ts", "symbol", "open", "high", "low", "close", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume"]), errors
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"]).reset_index(drop=True), errors


def build_event_frame(
    *,
    bars: pd.DataFrame,
    execution_minutes: int,
    structure_bars: int,
    horizon_minutes: list[int],
) -> pd.DataFrame:
    keep_cols = [
        "ts",
        "symbol",
        "event_side",
        "event_direction",
        "close",
        "bar_range",
        "atr",
        "volume_z",
        "close_location",
        "body_ratio",
    ] + [f"ret_{minutes}m_bps" for minutes in horizon_minutes]
    if bars.empty:
        return pd.DataFrame(columns=keep_cols)
    frame = bars.copy().sort_values(["symbol", "ts"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", group_keys=False)
    prev_close = grouped["close"].shift(1)
    frame["prev_high"] = grouped["high"].transform(lambda s: s.rolling(structure_bars).max().shift(1))
    frame["prev_low"] = grouped["low"].transform(lambda s: s.rolling(structure_bars).min().shift(1))
    tr_components = pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    frame["atr"] = grouped.apply(
        lambda g: tr_components.loc[g.index].max(axis=1).rolling(14).mean()
    ).reset_index(level=0, drop=True)
    frame["bar_range"] = (frame["high"] - frame["low"]).abs()
    denominator = frame["bar_range"].replace(0.0, pd.NA)
    frame["close_location"] = ((frame["close"] - frame["low"]) / denominator).fillna(0.5)
    volume_mean = grouped["volume"].transform(lambda s: s.rolling(max(20, structure_bars)).mean())
    volume_std = grouped["volume"].transform(
        lambda s: s.rolling(max(20, structure_bars)).std(ddof=0).replace(0.0, pd.NA)
    )
    frame["volume_z"] = ((frame["volume"] - volume_mean) / volume_std).fillna(0.0)
    frame["body_ratio"] = ((frame["close"] - frame["open"]).abs() / denominator).fillna(0.0)

    long_mask = (
        frame["prev_high"].notna()
        & frame["atr"].notna()
        & (frame["close"] > frame["prev_high"])
        & (frame["close"] > frame["open"])
        & (frame["close_location"] >= 0.60)
        & (frame["body_ratio"] >= 0.45)
        & (frame["bar_range"] >= frame["atr"] * 0.70)
        & (frame["volume_z"] >= -0.25)
    )
    short_mask = (
        frame["prev_low"].notna()
        & frame["atr"].notna()
        & (frame["close"] < frame["prev_low"])
        & (frame["close"] < frame["open"])
        & (frame["close_location"] <= 0.40)
        & (frame["body_ratio"] >= 0.45)
        & (frame["bar_range"] >= frame["atr"] * 0.70)
        & (frame["volume_z"] >= -0.25)
    )
    frame["event_side"] = ""
    frame.loc[long_mask, "event_side"] = "long"
    frame.loc[short_mask, "event_side"] = "short"
    frame["event_direction"] = 0
    frame.loc[long_mask, "event_direction"] = 1
    frame.loc[short_mask, "event_direction"] = -1

    for minutes in horizon_minutes:
        bars_ahead = max(1, int(round(minutes / max(1, execution_minutes))))
        future_close = grouped["close"].shift(-bars_ahead)
        frame[f"ret_{minutes}m_bps"] = (
            frame["event_direction"] * ((future_close / frame["close"]) - 1.0) * 10_000.0
        )

    events = frame[frame["event_direction"] != 0].copy()
    if events.empty:
        return pd.DataFrame(columns=keep_cols)
    return events[keep_cols].reset_index(drop=True)


def summarize_symbol_events(
    *,
    symbol: str,
    events: pd.DataFrame,
    horizon_minutes: list[int],
) -> dict[str, Any]:
    symbol_events = events[events["symbol"] == symbol].copy()
    event_count = int(len(symbol_events))
    long_count = int((symbol_events["event_side"] == "long").sum())
    short_count = int((symbol_events["event_side"] == "short").sum())
    horizon_rows: list[dict[str, Any]] = []
    best_minutes = 0
    best_expectancy = float("-inf")
    best_win_rate = 0.0
    best_sample = 0
    for minutes in horizon_minutes:
        col = f"ret_{minutes}m_bps"
        if col not in symbol_events.columns:
            continue
        series = pd.to_numeric(symbol_events[col], errors="coerce").dropna()
        if series.empty:
            continue
        expectancy = float(series.mean())
        win_rate = float((series > 0.0).mean())
        sample_count = int(series.shape[0])
        horizon_rows.append(
            {
                "horizon_minutes": int(minutes),
                "sample_count": sample_count,
                "expectancy_bps": round(expectancy, 3),
                "median_bps": round(float(series.median()), 3),
                "win_rate": round(win_rate, 4),
            }
        )
        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_minutes = int(minutes)
            best_win_rate = win_rate
            best_sample = sample_count
    latest_row = as_dict(
        symbol_events.sort_values("ts").tail(1).to_dict("records")[0] if event_count else {}
    )
    if event_count < 4:
        edge_status = "low_sample"
    elif best_expectancy >= 8.0 and best_win_rate >= 0.55:
        edge_status = "positive"
    elif best_expectancy >= 0.0:
        edge_status = "mixed"
    else:
        edge_status = "no_edge"
    return {
        "symbol": symbol,
        "event_count": event_count,
        "long_event_count": long_count,
        "short_event_count": short_count,
        "best_horizon_minutes": best_minutes,
        "best_expectancy_bps": round(best_expectancy if best_expectancy > float("-inf") else 0.0, 3),
        "best_win_rate": round(best_win_rate, 4),
        "best_sample_count": best_sample,
        "edge_status": edge_status,
        "latest_event_ts_utc": fmt_utc(
            pd.Timestamp(latest_row.get("ts")).to_pydatetime().replace(tzinfo=dt.timezone.utc)
            if latest_row.get("ts") is not None
            else None
        ),
        "latest_event_side": text(latest_row.get("event_side")),
        "latest_volume_z": round(float(latest_row.get("volume_z") or 0.0), 3)
        if latest_row
        else 0.0,
        "horizon_metrics": horizon_rows,
    }


def classify_backtest(
    *,
    slice_status: str,
    selected_summary: dict[str, Any],
) -> tuple[str, str]:
    selected_edge = text(selected_summary.get("edge_status"))
    selected_events = int(selected_summary.get("event_count") or 0)
    if selected_events <= 0:
        return "cross_section_data_unavailable", "refresh_market_data_then_rerun_shortline_cross_section_backtest"
    if selected_edge == "positive":
        if slice_status == "selected_setup_ready_orderflow_slice":
            return "setup_ready_cross_section_confirmed", "review_guarded_canary_with_cross_section_confirmation"
        return "watch_only_cross_section_positive", "continue_shadow_learning_wait_for_setup_ready"
    if selected_edge == "mixed":
        return "watch_only_cross_section_mixed", "collect_more_orderflow_and_wait_for_setup_ready"
    if selected_edge == "low_sample":
        return "watch_only_cross_section_low_sample", "collect_more_events_then_rerun_shortline_cross_section_backtest"
    return "watch_only_cross_section_no_edge", "deprioritize_until_orderflow_improves"


def build_payload(
    *,
    shortline_slice_path: Path,
    shortline_slice_payload: dict[str, Any],
    bars_fixture: Path | None,
    lookback_days: int,
    reference_now: dt.datetime,
) -> dict[str, Any]:
    slice_universe = dedupe_text(as_list(shortline_slice_payload.get("slice_universe")))
    selected_symbol = text(shortline_slice_payload.get("selected_symbol")).upper()
    execution_timeframe = text(shortline_slice_payload.get("execution_timeframe")) or "15m"
    execution_minutes = timeframe_minutes(execution_timeframe)
    horizon_minutes = horizon_minutes_from_policy(shortline_slice_payload)
    structure_bars = structure_window_bars(shortline_slice_payload, execution_minutes)
    end_date = reference_now.astimezone(dt.timezone.utc).date()
    start_date = end_date - dt.timedelta(days=max(2, int(lookback_days)))
    bars, fetch_errors = fetch_universe_bars(
        universe=slice_universe,
        start=start_date,
        end=end_date,
        execution_timeframe=execution_timeframe,
        cutoff=reference_now,
        bars_fixture=bars_fixture,
    )
    events = build_event_frame(
        bars=bars,
        execution_minutes=execution_minutes,
        structure_bars=structure_bars,
        horizon_minutes=horizon_minutes,
    )
    universe_metrics = [
        summarize_symbol_events(symbol=symbol, events=events, horizon_minutes=horizon_minutes)
        for symbol in slice_universe
    ]
    selected_summary = next(
        (row for row in universe_metrics if text(row.get("symbol")).upper() == selected_symbol.upper()),
        {
            "symbol": selected_symbol,
            "event_count": 0,
            "best_horizon_minutes": 0,
            "best_expectancy_bps": 0.0,
            "best_win_rate": 0.0,
            "edge_status": "low_sample",
            "horizon_metrics": [],
        },
    )
    backtest_status, research_decision = classify_backtest(
        slice_status=text(shortline_slice_payload.get("slice_status")),
        selected_summary=selected_summary,
    )
    ranked = sorted(
        universe_metrics,
        key=lambda item: (float(item.get("best_expectancy_bps") or 0.0), float(item.get("best_win_rate") or 0.0)),
        reverse=True,
    )
    selected_rank = next(
        (
            index + 1
            for index, row in enumerate(ranked)
            if text(row.get("symbol")).upper() == selected_symbol.upper()
        ),
        0,
    )
    latest_events = (
        events.sort_values("ts", ascending=False).head(8).copy()
        if not events.empty
        else pd.DataFrame(columns=["ts", "symbol", "event_side"])
    )
    recent_event_rows = [
        {
            "ts_utc": fmt_utc(
                pd.Timestamp(row["ts"]).to_pydatetime().replace(tzinfo=dt.timezone.utc)
            ),
            "symbol": text(row["symbol"]).upper(),
            "event_side": text(row["event_side"]),
        }
        for row in latest_events.to_dict("records")
    ]
    selected_edge_status = text(selected_summary.get("edge_status")) or "low_sample"
    backtest_brief = ":".join(
        [
            backtest_status,
            selected_symbol or "-",
            selected_edge_status or "-",
            f"h{int(selected_summary.get('best_horizon_minutes') or 0)}m",
        ]
    )
    blocker_detail = join_unique(
        [
            text(shortline_slice_payload.get("blocker_detail")),
            (
                f"backtest_window={start_date.isoformat()}->{end_date.isoformat()}"
                if selected_symbol
                else ""
            ),
            (
                f"selected_events={int(selected_summary.get('event_count') or 0)}"
                f"; selected_expectancy_bps={float(selected_summary.get('best_expectancy_bps') or 0.0):.3f}"
                f"; selected_win_rate={float(selected_summary.get('best_win_rate') or 0.0):.4f}"
            ),
            (
                f"fetch_errors={','.join(f'{k}:{v}' for k, v in fetch_errors.items())}"
                if fetch_errors
                else ""
            ),
        ]
    )
    done_when = (
        f"{selected_symbol or 'route symbol'} has repeatable positive expectancy in shortline cross-section replay "
        "and the execution gate graduates from Bias_Only into Setup_Ready with a fresh ticket row"
    )
    return {
        "action": "build_crypto_shortline_cross_section_backtest",
        "ok": True,
        "status": "ok",
        "generated_at_utc": fmt_utc(reference_now),
        "slice_brief": text(shortline_slice_payload.get("slice_brief")),
        "slice_status": text(shortline_slice_payload.get("slice_status")),
        "slice_research_decision": text(shortline_slice_payload.get("research_decision")),
        "selected_symbol": selected_symbol,
        "slice_universe": slice_universe,
        "slice_universe_brief": ",".join(slice_universe),
        "execution_timeframe": execution_timeframe,
        "lookback_days": int(lookback_days),
        "backtest_status": backtest_status,
        "backtest_brief": backtest_brief,
        "research_decision": research_decision,
        "selected_edge_status": selected_edge_status,
        "selected_rank_in_universe": selected_rank,
        "selected_summary": selected_summary,
        "universe_metrics": universe_metrics,
        "event_sample_count": int(len(events)),
        "event_sample_rows": recent_event_rows,
        "bar_rows": int(len(bars)),
        "coverage_start_utc": fmt_utc(dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc)),
        "coverage_end_utc": fmt_utc(reference_now),
        "structure_window_bars": structure_bars,
        "horizon_minutes": horizon_minutes,
        "fetch_errors": fetch_errors,
        "blocker_detail": blocker_detail,
        "done_when": done_when,
        "artifacts": {
            "crypto_shortline_backtest_slice": str(shortline_slice_path),
            "bars_fixture": str(bars_fixture) if bars_fixture else "",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    selected = as_dict(payload.get("selected_summary"))
    return "\n".join(
        [
            "# Crypto Shortline Cross Section Backtest",
            "",
            f"- brief: `{text(payload.get('backtest_brief'))}`",
            f"- slice: `{text(payload.get('slice_brief'))}`",
            f"- research_decision: `{text(payload.get('research_decision'))}`",
            f"- selected_symbol: `{text(payload.get('selected_symbol'))}`",
            f"- selected_edge_status: `{text(payload.get('selected_edge_status'))}`",
            f"- selected_rank_in_universe: `{payload.get('selected_rank_in_universe')}`",
            f"- selected_event_count: `{selected.get('event_count')}`",
            f"- selected_best_horizon_minutes: `{selected.get('best_horizon_minutes')}`",
            f"- selected_best_expectancy_bps: `{selected.get('best_expectancy_bps')}`",
            f"- selected_best_win_rate: `{selected.get('best_win_rate')}`",
            f"- universe: `{text(payload.get('slice_universe_brief'))}`",
            f"- blocker: `{text(payload.get('blocker_detail'))}`",
            f"- done_when: `{text(payload.get('done_when'))}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build crypto shortline cross-section backtest artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--bars-fixture", default="")
    parser.add_argument("--artifact-keep", type=int, default=12)
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--now", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    review_dir.mkdir(parents=True, exist_ok=True)
    reference_now = parse_now(args.now)
    shortline_slice_path = find_latest(review_dir, "*_crypto_shortline_backtest_slice.json")
    if shortline_slice_path is None:
        raise SystemExit("missing_required_artifact:crypto_shortline_backtest_slice")
    bars_fixture = Path(args.bars_fixture).expanduser().resolve() if text(args.bars_fixture) else None
    payload = build_payload(
        shortline_slice_path=shortline_slice_path,
        shortline_slice_payload=load_json_mapping(shortline_slice_path),
        bars_fixture=bars_fixture,
        lookback_days=int(args.lookback_days),
        reference_now=reference_now,
    )
    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    artifact = review_dir / f"{stamp}_crypto_shortline_cross_section_backtest.json"
    markdown = review_dir / f"{stamp}_crypto_shortline_cross_section_backtest.md"
    checksum = review_dir / f"{stamp}_crypto_shortline_cross_section_backtest_checksum.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    checksum.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_sha256": sha256_file(artifact),
                "markdown": str(markdown),
                "markdown_sha256": sha256_file(markdown),
                "generated_at_utc": fmt_utc(reference_now),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
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
