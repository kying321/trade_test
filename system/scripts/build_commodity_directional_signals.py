#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lie_engine.research.real_data import fetch_commodity_proxy_daily  # noqa: E402


DEFAULT_SYMBOLS = ("XAUUSD", "XAGUSD", "COPPER")
DEFAULT_LOOKBACK_DAYS = 150
DEFAULT_MAX_AGE_DAYS = 14
DEFAULT_ARTIFACT_TTL_HOURS = 168.0
DEFAULT_ARTIFACT_KEEP = 12
DEFAULT_TARGET_MULTIPLE = 1.6
DEFAULT_STATE_CARRY_MAX_AGE_DAYS = 5
PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
COMBO_CANONICAL_ALIASES = {
    "ad_breakout": "cvd_breakout",
    "ad_rsi_breakout": "cvd_rsi_breakout",
    "ad_rsi_vol_breakout": "cvd_rsi_vol_breakout",
    "ad_rsi_reclaim": "cvd_rsi_reclaim",
    "taker_oi_ad_breakout": "taker_oi_cvd_breakout",
    "taker_oi_ad_rsi_breakout": "taker_oi_cvd_rsi_breakout",
}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat()


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def normalize_symbols(raw: str) -> list[str]:
    parts = [str(x).strip().upper() for x in str(raw).split(",")]
    return [x for x in parts if x] or list(DEFAULT_SYMBOLS)


def safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except Exception:
        return float(default)
    if pd.isna(value):
        return float(default)
    return float(value)


def canonical_combo_id(combo_id: Any) -> str:
    combo = str(combo_id or "").strip()
    return COMBO_CANONICAL_ALIASES.get(combo, combo)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_review_artifact(review_dir: Path, suffix: str, reference_now: dt.datetime | None = None) -> Path | None:
    ref = reference_now.astimezone(dt.timezone.utc) if reference_now is not None else None
    candidates: list[tuple[str, Path]] = []
    for path in review_dir.glob(f"*_{suffix}.json"):
        stem = path.name[:16]
        if len(stem) != 16 or stem[8] != "T" or not stem.endswith("Z"):
            continue
        if ref is not None:
            try:
                stamp = dt.datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
            except ValueError:
                continue
            if stamp > ref:
                continue
        candidates.append((stem, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_combo_module() -> Any:
    script_path = SYSTEM_ROOT / "scripts" / "backtest_binance_indicator_combo_etf.py"
    spec = importlib.util.spec_from_file_location("commodity_combo_etf_builder", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("combo_etf_module_load_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loopback_proxy_value(raw: str) -> bool:
    value = str(raw or "").strip().lower()
    return bool(value) and ("127.0.0.1" in value or "localhost" in value)


@contextlib.contextmanager
def disable_loopback_proxy_env() -> Iterator[None]:
    saved: dict[str, str] = {}
    removed: list[str] = []
    for key in PROXY_ENV_KEYS:
        value = os.environ.get(key)
        if value is None:
            continue
        if not loopback_proxy_value(value):
            continue
        saved[key] = value
        removed.append(key)
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key in removed:
            os.environ[key] = saved[key]


def load_cache_frame(
    *,
    output_root: Path,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_root = output_root / "artifacts" / "research_cache"
    if not cache_root.exists():
        return pd.DataFrame(), {"bars_source": "missing", "cache_meta_path": "", "cache_bars_path": ""}

    meta_paths = sorted(cache_root.glob("*_meta.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for meta_path in meta_paths:
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        universe = [str(x).strip().upper() for x in meta_payload.get("universe", []) if str(x).strip()]
        if symbol not in universe:
            continue
        bars_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_bars.parquet"))
        if not bars_path.exists():
            continue
        try:
            bars = pd.read_parquet(bars_path)
        except Exception:
            continue
        if "symbol" not in bars.columns or "ts" not in bars.columns:
            continue
        subset = bars[bars["symbol"].astype(str).str.upper() == symbol].copy()
        if subset.empty:
            continue
        subset["ts"] = pd.to_datetime(subset["ts"], utc=True, errors="coerce")
        subset = subset.dropna(subset=["ts"]).reset_index(drop=True)
        subset = subset[(subset["ts"].dt.date >= start_date) & (subset["ts"].dt.date <= end_date)].reset_index(drop=True)
        if subset.empty:
            continue
        return subset, {
            "bars_source": "cache",
            "cache_meta_path": str(meta_path),
            "cache_bars_path": str(bars_path),
            "bars_end_date": str(subset["ts"].iloc[-1].date()),
        }

    return pd.DataFrame(), {"bars_source": "missing", "cache_meta_path": "", "cache_bars_path": ""}


def fetch_symbol_frame(
    *,
    output_root: Path,
    symbol: str,
    start_date: dt.date,
    end_date: dt.date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fresh_error = ""
    with disable_loopback_proxy_env():
        try:
            fresh = fetch_commodity_proxy_daily(symbol, start_date, end_date)
        except Exception as exc:
            fresh = pd.DataFrame()
            fresh_error = f"{type(exc).__name__}:{exc}"
    if not fresh.empty:
        fresh = fresh.copy().reset_index(drop=True)
        fresh["ts"] = pd.to_datetime(fresh["ts"], utc=True, errors="coerce")
        fresh = fresh.dropna(subset=["ts"]).reset_index(drop=True)
        return fresh, {
            "bars_source": "fresh",
            "fresh_error": fresh_error,
            "cache_meta_path": "",
            "cache_bars_path": "",
            "bars_end_date": str(fresh["ts"].iloc[-1].date()),
        }
    if not fresh_error:
        fresh_error = "fresh_download_empty"

    cached, meta = load_cache_frame(output_root=output_root, symbol=symbol, start_date=start_date, end_date=end_date)
    meta["fresh_error"] = fresh_error
    return cached, meta


def choose_symbol_combo(combo_payload: dict[str, Any], symbol: str) -> dict[str, Any]:
    family = dict(combo_payload.get("commodity_family") or {})
    candidates: list[dict[str, Any]] = []
    for section_rank, section_name in enumerate(("ranked_combos", "discarded_combos")):
        for family_rank, combo in enumerate(family.get(section_name, []) or []):
            if not isinstance(combo, dict):
                continue
            for asset in combo.get("per_asset", []) or []:
                if not isinstance(asset, dict):
                    continue
                if str(asset.get("symbol") or "").strip().upper() != symbol:
                    continue
                candidates.append(
                    {
                        "section_name": section_name,
                        "section_rank": section_rank,
                        "family_rank": family_rank,
                        "combo_id": str(combo.get("combo_id") or "").strip(),
                        "combo_id_canonical": canonical_combo_id(combo.get("combo_id")),
                        "combo_confirmation_indicator": str(combo.get("confirmation_indicator") or "").strip(),
                        "combo_mode": str(combo.get("mode") or "").strip(),
                        "combo_discard_reason": str(combo.get("discard_reason") or "").strip(),
                        "per_asset": dict(asset),
                    }
                )
    if not candidates:
        return {
            "section_name": "",
            "section_rank": 9,
            "family_rank": 999,
            "combo_id": "ad_rsi_breakout",
            "combo_id_canonical": "cvd_rsi_breakout",
            "combo_confirmation_indicator": "cvd_lite_proxy_plus_rsi",
            "combo_mode": "breakout",
            "combo_discard_reason": "",
            "per_asset": {},
        }

    candidates.sort(
        key=lambda row: (
            int(row["section_rank"]),
            1 if row["combo_discard_reason"] else 0,
            -safe_float((row.get("per_asset") or {}).get("score"), 0.0),
            int(row["family_rank"]),
            str(row["combo_id"]),
        )
    )
    return candidates[0]


def derive_confidence(per_asset: dict[str, Any]) -> float:
    timely = max(0.0, min(1.0, safe_float((per_asset.get("lag_metrics") or {}).get("timely_hit_rate"), 0.0)))
    win_rate = max(0.0, min(1.0, safe_float(per_asset.get("win_rate"), 0.0)))
    consistency = max(0.0, min(1.0, safe_float(per_asset.get("consistency"), 0.0)))
    profit_factor = max(0.0, min(1.0, safe_float(per_asset.get("profit_factor"), 0.0) / 2.0))
    confidence = 100.0 * (0.35 * timely + 0.35 * win_rate + 0.15 * consistency + 0.15 * profit_factor)
    return float(round(max(5.0, min(95.0, confidence)), 6))


def latest_true_index(series: pd.Series) -> int:
    values = series.fillna(False)
    return int(values[values].index[-1]) if bool(values.any()) else -1


def build_signal_row(
    *,
    market_frame: pd.DataFrame,
    trigger_idx: int,
    side: str,
    as_of: dt.date,
    max_age_days: int,
    selected_combo: dict[str, Any],
    bars_meta: dict[str, Any],
    signal_kind: str = "combo_trigger",
    anchor_trigger_date: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    row = market_frame.iloc[int(trigger_idx)]
    trigger_date = pd.Timestamp(row["ts"]).date()
    age_days = max(0, int((as_of - trigger_date).days))
    price_reference_source = str(row.get("source") or bars_meta.get("price_reference_source") or "").strip()
    entry_price = safe_float(row.get("close"), 0.0)
    low_price = safe_float(row.get("low"), 0.0)
    high_price = safe_float(row.get("high"), 0.0)
    prev_low_20 = safe_float(row.get("prev_low_20"), 0.0)
    prev_high_20 = safe_float(row.get("prev_high_20"), 0.0)
    if side == "LONG":
        stop_price = prev_low_20 if 0.0 < prev_low_20 < entry_price else low_price
        risk = entry_price - stop_price
        target_price = entry_price + risk * DEFAULT_TARGET_MULTIPLE
    else:
        stop_price = prev_high_20 if prev_high_20 > entry_price else high_price
        risk = stop_price - entry_price
        target_price = entry_price - risk * DEFAULT_TARGET_MULTIPLE
    if entry_price <= 0.0 or stop_price <= 0.0 or target_price <= 0.0 or risk <= 0.0:
        return None, {
            "signal_status": "invalid_signal_levels",
            "trigger_date": trigger_date.isoformat(),
            "age_days": age_days,
        }

    per_asset = dict(selected_combo.get("per_asset") or {})
    combo_id = str(selected_combo.get("combo_id") or "").strip()
    combo_id_canonical = str(selected_combo.get("combo_id_canonical") or canonical_combo_id(combo_id)).strip()
    combo_confirmation_indicator = str(selected_combo.get("combo_confirmation_indicator") or "").strip()
    combo_mode = str(selected_combo.get("combo_mode") or "").strip() or "breakout"
    lag_metrics = dict(per_asset.get("lag_metrics") or {})
    notes = (
        f"combo={combo_id or '-'}"
        f"; combo_canonical={combo_id_canonical or '-'}"
        f"; confirmation_indicator={combo_confirmation_indicator or '-'}"
        f"; mode={combo_mode or '-'}"
        f"; signal_kind={signal_kind or '-'}"
        f"; bars_source={bars_meta.get('bars_source') or '-'}"
        f"; price_reference={price_reference_source or '-'}"
        f"; execution_price_ready=false"
        f"; timely={safe_float(lag_metrics.get('timely_hit_rate'), 0.0):.2%}"
        f"; win_rate={safe_float(per_asset.get('win_rate'), 0.0):.2%}"
        f"; profit_factor={safe_float(per_asset.get('profit_factor'), 0.0):.2f}"
    )
    if anchor_trigger_date:
        notes += f"; anchor_trigger_date={anchor_trigger_date}"
    signal_row = {
        "date": trigger_date.isoformat(),
        "symbol": str(row.get("symbol") or "").strip().upper(),
        "side": side,
        "regime": f"commodity_combo_{combo_mode}",
        "confidence": derive_confidence(per_asset),
        "convexity_ratio": float(DEFAULT_TARGET_MULTIPLE),
        "signal_kind": signal_kind,
        "anchor_trigger_date": anchor_trigger_date,
        "combo_id": combo_id,
        "combo_id_canonical": combo_id_canonical,
        "confirmation_indicator": combo_confirmation_indicator,
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "price_reference_kind": "commodity_proxy_market",
        "price_reference_source": price_reference_source,
        "execution_price_ready": False,
        "factor_flags": [
            f"combo:{combo_id or 'unknown'}",
            f"combo_canonical:{combo_id_canonical or 'unknown'}",
            f"confirmation_indicator:{combo_confirmation_indicator or 'unknown'}",
            f"combo_mode:{combo_mode or 'unknown'}",
            f"signal_kind:{signal_kind or 'unknown'}",
            f"bars_source:{bars_meta.get('bars_source') or 'unknown'}",
            f"price_reference_kind:commodity_proxy_market",
        ],
        "notes": notes,
    }
    return signal_row, {
        "signal_status": "recent_combo_trigger" if age_days <= max(1, int(max_age_days)) else "stale_combo_trigger",
        "trigger_date": trigger_date.isoformat(),
        "age_days": age_days,
        "price_reference_kind": "commodity_proxy_market",
        "price_reference_source": price_reference_source,
        "execution_price_ready": False,
        "signal_kind": signal_kind,
        "anchor_trigger_date": anchor_trigger_date,
    }


def build_state_carry_row(
    *,
    market_frame: pd.DataFrame,
    long_state: pd.Series,
    short_state: pd.Series,
    long_event: pd.Series,
    short_event: pd.Series,
    as_of: dt.date,
    state_carry_max_age_days: int,
    selected_combo: dict[str, Any],
    bars_meta: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    latest_long_state_idx = latest_true_index(long_state)
    latest_short_state_idx = latest_true_index(short_state)
    latest_long_event_idx = latest_true_index(long_event)
    latest_short_event_idx = latest_true_index(short_event)

    candidates: list[tuple[int, str, int]] = []
    if latest_long_state_idx >= 0 and latest_long_event_idx >= 0 and latest_long_event_idx <= latest_long_state_idx:
        candidates.append((latest_long_state_idx, "LONG", latest_long_event_idx))
    if latest_short_state_idx >= 0 and latest_short_event_idx >= 0 and latest_short_event_idx <= latest_short_state_idx:
        candidates.append((latest_short_state_idx, "SHORT", latest_short_event_idx))
    if not candidates:
        return None, None

    candidates.sort(key=lambda row: (row[0], 1 if row[1] == "SHORT" else 0), reverse=True)
    state_idx, side, anchor_idx = candidates[0]
    state_date = pd.Timestamp(market_frame.iloc[int(state_idx)]["ts"]).date()
    state_age_days = max(0, int((as_of - state_date).days))
    if state_age_days > max(1, int(state_carry_max_age_days)):
        return None, None

    anchor_date = pd.Timestamp(market_frame.iloc[int(anchor_idx)]["ts"]).date().isoformat()
    signal_row, signal_meta = build_signal_row(
        market_frame=market_frame,
        trigger_idx=state_idx,
        side=side,
        as_of=as_of,
        max_age_days=state_carry_max_age_days,
        selected_combo=selected_combo,
        bars_meta=bars_meta,
        signal_kind="state_carry",
        anchor_trigger_date=anchor_date,
    )
    if signal_row is None:
        return None, None
    return signal_row, {
        **signal_meta,
        "signal_status": "recent_combo_state_carry",
        "trigger_date": state_date.isoformat(),
        "age_days": state_age_days,
        "anchor_trigger_date": anchor_date,
        "state_carry": True,
    }


def build_symbol_snapshot(
    *,
    symbol: str,
    combo_module: Any,
    combo_payload: dict[str, Any],
    output_root: Path,
    as_of: dt.date,
    lookback_days: int,
    max_age_days: int,
    enable_state_carry: bool,
    state_carry_max_age_days: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start_date = as_of - dt.timedelta(days=max(30, int(lookback_days)))
    raw_frame, bars_meta = fetch_symbol_frame(
        output_root=output_root,
        symbol=symbol,
        start_date=start_date,
        end_date=as_of,
    )
    selected_combo = choose_symbol_combo(combo_payload, symbol)
    selected_combo_id = str(selected_combo.get("combo_id") or "").strip()
    selected_combo_id_canonical = str(selected_combo.get("combo_id_canonical") or canonical_combo_id(selected_combo_id)).strip()
    combo_defs = {str(row.get("combo_id") or ""): row for row in getattr(combo_module, "COMBO_DEFS", [])}
    combo_def = combo_defs.get(selected_combo_id) or combo_defs.get("ad_rsi_breakout")
    diagnostics = {
        "symbol": symbol,
        "bars_source": str(bars_meta.get("bars_source") or ""),
        "bars_end_date": str(bars_meta.get("bars_end_date") or ""),
        "fresh_error": str(bars_meta.get("fresh_error") or ""),
        "cache_meta_path": str(bars_meta.get("cache_meta_path") or ""),
        "cache_bars_path": str(bars_meta.get("cache_bars_path") or ""),
        "selected_combo_id": selected_combo_id,
        "selected_combo_id_canonical": selected_combo_id_canonical,
        "selected_combo_confirmation_indicator": str(selected_combo.get("combo_confirmation_indicator") or ""),
        "selected_combo_mode": str(selected_combo.get("combo_mode") or ""),
        "selected_combo_section": str(selected_combo.get("section_name") or ""),
        "selected_combo_family_rank": int(selected_combo.get("family_rank") or 0),
        "selected_combo_score": safe_float((selected_combo.get("per_asset") or {}).get("score"), 0.0),
        "selected_combo_discard_reason": str(selected_combo.get("combo_discard_reason") or ""),
        "signal_status": "",
        "trigger_date": "",
        "age_days": -1,
        "signal_kind": "",
        "anchor_trigger_date": "",
        "price_reference_kind": "",
        "price_reference_source": "",
        "execution_price_ready": False,
    }
    if raw_frame.empty:
        diagnostics["signal_status"] = "market_data_unavailable"
        return [], diagnostics
    if combo_def is None:
        diagnostics["signal_status"] = "combo_definition_missing"
        return [], diagnostics

    market_frame = combo_module.build_market_frame(symbol, raw_frame.reset_index(drop=True), symbol, None)
    long_state, short_state, long_event, short_event = combo_def["state_fn"](market_frame)
    long_signal = long_state.fillna(False) & long_event.fillna(False)
    short_signal = short_state.fillna(False) & short_event.fillna(False)

    latest_long_idx = latest_true_index(long_signal)
    latest_short_idx = latest_true_index(short_signal)
    if latest_long_idx < 0 and latest_short_idx < 0:
        if enable_state_carry:
            carry_row, carry_meta = build_state_carry_row(
                market_frame=market_frame,
                long_state=long_state,
                short_state=short_state,
                long_event=long_event,
                short_event=short_event,
                as_of=as_of,
                state_carry_max_age_days=state_carry_max_age_days,
                selected_combo=selected_combo,
                bars_meta=bars_meta,
            )
            if carry_row is not None and carry_meta is not None:
                diagnostics["signal_status"] = str(carry_meta.get("signal_status") or "")
                diagnostics["trigger_date"] = str(carry_meta.get("trigger_date") or "")
                diagnostics["age_days"] = int(carry_meta.get("age_days", -1) or -1)
                diagnostics["signal_kind"] = str(carry_meta.get("signal_kind") or "state_carry")
                diagnostics["anchor_trigger_date"] = str(carry_meta.get("anchor_trigger_date") or "")
                diagnostics["price_reference_kind"] = str(carry_meta.get("price_reference_kind") or "")
                diagnostics["price_reference_source"] = str(carry_meta.get("price_reference_source") or "")
                diagnostics["execution_price_ready"] = bool(carry_meta.get("execution_price_ready"))
                return [carry_row], diagnostics
        diagnostics["signal_status"] = "no_combo_trigger_found"
        return [], diagnostics

    if latest_long_idx >= latest_short_idx:
        trigger_idx = latest_long_idx
        trigger_side = "LONG"
    else:
        trigger_idx = latest_short_idx
        trigger_side = "SHORT"

    signal_row, signal_meta = build_signal_row(
        market_frame=market_frame,
        trigger_idx=trigger_idx,
        side=trigger_side,
        as_of=as_of,
        max_age_days=max_age_days,
        selected_combo=selected_combo,
        bars_meta=bars_meta,
        signal_kind="combo_trigger",
    )
    if enable_state_carry and signal_meta.get("age_days", -1) > max(1, int(max_age_days)):
        carry_row, carry_meta = build_state_carry_row(
            market_frame=market_frame,
            long_state=long_state,
            short_state=short_state,
            long_event=long_event,
            short_event=short_event,
            as_of=as_of,
            state_carry_max_age_days=state_carry_max_age_days,
            selected_combo=selected_combo,
            bars_meta=bars_meta,
        )
        if carry_row is not None and carry_meta is not None:
            diagnostics["signal_status"] = str(carry_meta.get("signal_status") or "")
            diagnostics["trigger_date"] = str(carry_meta.get("trigger_date") or "")
            diagnostics["age_days"] = int(carry_meta.get("age_days", -1) or -1)
            diagnostics["signal_kind"] = str(carry_meta.get("signal_kind") or "state_carry")
            diagnostics["anchor_trigger_date"] = str(carry_meta.get("anchor_trigger_date") or "")
            diagnostics["price_reference_kind"] = str(carry_meta.get("price_reference_kind") or "")
            diagnostics["price_reference_source"] = str(carry_meta.get("price_reference_source") or "")
            diagnostics["execution_price_ready"] = bool(carry_meta.get("execution_price_ready"))
            return [carry_row], diagnostics
    diagnostics["signal_status"] = str(signal_meta.get("signal_status") or "")
    diagnostics["trigger_date"] = str(signal_meta.get("trigger_date") or "")
    diagnostics["age_days"] = int(signal_meta.get("age_days", -1) or -1)
    diagnostics["signal_kind"] = str(signal_meta.get("signal_kind") or "combo_trigger")
    diagnostics["anchor_trigger_date"] = str(signal_meta.get("anchor_trigger_date") or "")
    diagnostics["price_reference_kind"] = str(signal_meta.get("price_reference_kind") or "")
    diagnostics["price_reference_source"] = str(signal_meta.get("price_reference_source") or "")
    diagnostics["execution_price_ready"] = bool(signal_meta.get("execution_price_ready"))
    return ([signal_row] if signal_row is not None else []), diagnostics


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Commodity Directional Signals",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- source_combo_artifact: `{payload.get('source_combo_artifact') or ''}`",
        f"- lookback_days: `{payload.get('lookback_days') or 0}`",
        f"- max_age_days: `{payload.get('max_age_days') or 0}`",
        "",
        "## Summary",
        f"- signal_row_count: `{payload.get('summary', {}).get('signal_row_count', 0)}`",
        f"- recent_signal_count: `{payload.get('summary', {}).get('recent_signal_count', 0)}`",
        f"- stale_signal_count: `{payload.get('summary', {}).get('stale_signal_count', 0)}`",
        f"- no_trigger_count: `{payload.get('summary', {}).get('no_trigger_count', 0)}`",
        "",
        "## Symbols",
    ]
    diagnostics = dict(payload.get("symbol_diagnostics") or {})
    signals = dict(payload.get("signals") or {})
    for symbol in payload.get("symbols", []):
        diag = dict(diagnostics.get(symbol) or {})
        rows = list(signals.get(symbol) or [])
        lines.append(
            f"- `{symbol}` status=`{diag.get('signal_status') or ''}` bars=`{diag.get('bars_source') or ''}` price_ref=`{diag.get('price_reference_source') or ''}` combo=`{diag.get('selected_combo_id') or ''}` canonical=`{diag.get('selected_combo_id_canonical') or ''}`"
        )
        if rows:
            row = dict(rows[0])
            lines.append(
                "  signal=`{side}` date=`{date}` kind=`{kind}` combo=`{combo}` canonical=`{canonical}` conf=`{conf:.2f}` conv=`{conv:.2f}` entry=`{entry:.4f}` stop=`{stop:.4f}` target=`{target:.4f}` exec_ready=`{ready}`".format(
                    side=str(row.get("side") or ""),
                    date=str(row.get("date") or ""),
                    kind=str(row.get("signal_kind") or "combo_trigger"),
                    combo=str(row.get("combo_id") or ""),
                    canonical=str(row.get("combo_id_canonical") or ""),
                    conf=safe_float(row.get("confidence"), 0.0),
                    conv=safe_float(row.get("convexity_ratio"), 0.0),
                    entry=safe_float(row.get("entry_price"), 0.0),
                    stop=safe_float(row.get("stop_price"), 0.0),
                    target=safe_float(row.get("target_price"), 0.0),
                    ready=str(bool(row.get("execution_price_ready"))).lower(),
                )
            )
        else:
            lines.append(
                f"  trigger_date=`{diag.get('trigger_date') or ''}` age_days=`{diag.get('age_days')}` fresh_error=`{diag.get('fresh_error') or ''}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def prune_review_artifacts(
    *,
    review_dir: Path,
    current_paths: list[Path],
    now_dt: dt.datetime,
    ttl_hours: float,
    keep: int,
) -> tuple[list[str], list[str]]:
    cutoff = now_dt - dt.timedelta(hours=max(1.0, float(ttl_hours)))
    protected = {path.name for path in current_paths}
    candidates = sorted(review_dir.glob("*_commodity_directional_signals*"), key=lambda path: path.stat().st_mtime, reverse=True)
    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in candidates:
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
    for path in survivors[max(0, int(keep)) :]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build commodity directional signals from combo-backed proxy bars.")
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--output-root", default=str(SYSTEM_ROOT / "output"))
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--date", default="")
    parser.add_argument("--combo-artifact", default="")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--enable-state-carry", action="store_true")
    parser.add_argument("--state-carry-max-age-days", type=int, default=DEFAULT_STATE_CARRY_MAX_AGE_DAYS)
    parser.add_argument("--artifact-ttl-hours", type=float, default=DEFAULT_ARTIFACT_TTL_HOURS)
    parser.add_argument("--artifact-keep", type=int, default=DEFAULT_ARTIFACT_KEEP)
    parser.add_argument("--now", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    runtime_now = parse_now(args.now)
    as_of = dt.date.fromisoformat(str(args.date)) if str(args.date).strip() else runtime_now.date()
    combo_path = (
        Path(str(args.combo_artifact)).expanduser().resolve()
        if str(args.combo_artifact).strip()
        else latest_review_artifact(review_dir, "binance_indicator_combo_etf", runtime_now)
    )
    if combo_path is None or not combo_path.exists():
        raise FileNotFoundError("commodity_combo_artifact_missing")
    combo_payload = json.loads(combo_path.read_text(encoding="utf-8"))
    combo_module = load_combo_module()
    symbols = normalize_symbols(args.symbols)

    signals: dict[str, list[dict[str, Any]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    recent_signal_count = 0
    stale_signal_count = 0
    state_carry_count = 0
    no_trigger_count = 0
    unavailable_count = 0

    for symbol in symbols:
        rows, diag = build_symbol_snapshot(
            symbol=symbol,
            combo_module=combo_module,
            combo_payload=combo_payload,
            output_root=output_root,
            as_of=as_of,
            lookback_days=max(30, int(args.lookback_days)),
            max_age_days=max(1, int(args.max_age_days)),
            enable_state_carry=bool(args.enable_state_carry),
            state_carry_max_age_days=max(1, int(args.state_carry_max_age_days)),
        )
        signals[symbol] = rows
        diagnostics[symbol] = diag
        status = str(diag.get("signal_status") or "")
        if status == "recent_combo_trigger":
            recent_signal_count += 1
        elif status == "recent_combo_state_carry":
            recent_signal_count += 1
            state_carry_count += 1
        elif status == "stale_combo_trigger":
            stale_signal_count += 1
        elif status == "market_data_unavailable":
            unavailable_count += 1
        else:
            no_trigger_count += 1

    payload = {
        "ok": True,
        "status": "ok",
        "as_of": f"{as_of.isoformat()}T00:00:00+00:00",
        "source_combo_artifact": str(combo_path),
        "symbols": symbols,
        "lookback_days": int(max(30, int(args.lookback_days))),
        "max_age_days": int(max(1, int(args.max_age_days))),
        "state_carry_enabled": bool(args.enable_state_carry),
        "state_carry_max_age_days": int(max(1, int(args.state_carry_max_age_days))),
        "signals": signals,
        "symbol_diagnostics": diagnostics,
        "summary": {
            "signal_row_count": int(sum(len(rows) for rows in signals.values())),
            "recent_signal_count": int(recent_signal_count),
            "state_carry_count": int(state_carry_count),
            "stale_signal_count": int(stale_signal_count),
            "no_trigger_count": int(no_trigger_count),
            "market_data_unavailable_count": int(unavailable_count),
        },
    }

    stamp = runtime_now.strftime("%Y%m%dT%H%M%SZ")
    json_path = review_dir / f"{stamp}_commodity_directional_signals.json"
    md_path = review_dir / f"{stamp}_commodity_directional_signals.md"
    checksum_path = review_dir / f"{stamp}_commodity_directional_signals_checksum.json"
    review_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "generated_at": fmt_utc(runtime_now),
        "files": [
            {"path": str(json_path), "sha256": sha256_file(json_path)},
            {"path": str(md_path), "sha256": sha256_file(md_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_review_artifacts(
        review_dir=review_dir,
        current_paths=[json_path, md_path, checksum_path],
        now_dt=runtime_now,
        ttl_hours=max(1.0, float(args.artifact_ttl_hours)),
        keep=max(3, int(args.artifact_keep)),
    )
    payload["artifact"] = str(json_path)
    payload["markdown"] = str(md_path)
    payload["checksum"] = str(checksum_path)
    payload["pruned_keep"] = pruned_keep
    payload["pruned_age"] = pruned_age
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "md": str(md_path),
                "checksum": str(checksum_path),
                "summary": payload["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
