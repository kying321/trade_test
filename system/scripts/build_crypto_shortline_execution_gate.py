#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = SYSTEM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lie_engine.signal.features import add_common_features
from lie_engine.signal.theory import _ict_scores


DEFAULT_OUTPUT_ROOT = SYSTEM_ROOT / "output"
DEFAULT_REVIEW_DIR = DEFAULT_OUTPUT_ROOT / "review"
DEFAULT_MICRO_CAPTURE_DIR = DEFAULT_OUTPUT_ROOT / "artifacts" / "micro_capture"
DEFAULT_CONFIG_PATH = SYSTEM_ROOT / "config.yaml"
TIMESTAMP_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_")
FUTURE_STAMP_GRACE_MINUTES = 5
DEFAULT_SHORTLINE_SUPPORTED = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
DEFAULT_LOCATION_PRIORITY = ("HVN", "POC")
DEFAULT_CVD_LOCAL_WINDOW_MINUTES = 15
DEFAULT_CVD_REFERENCE_MAX_AGE_MINUTES = 15
DEFAULT_CVD_ATTACK_ALIGNMENT_MIN_ABS = 0.05
DEFAULT_TRIGGER_STACK = (
    "4h_profile_location",
    "liquidity_sweep",
    "1m_5m_mss_or_choch",
    "15m_cvd_divergence_or_confirmation",
    "fvg_ob_breaker_retest",
    "15m_reversal_or_breakout_candle",
)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_now(raw: str | None) -> dt.datetime:
    text = str(raw or "").strip()
    if not text:
        return now_utc()
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def fmt_utc(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat()


def resolve_path(raw: str, *, anchor: Path) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return anchor / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_stamp(path: Path) -> str:
    match = TIMESTAMP_RE.match(path.name)
    return match.group("stamp") if match else ""


def parsed_artifact_stamp(path: Path) -> dt.datetime | None:
    stamp = artifact_stamp(path)
    if not stamp:
        return None
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def artifact_sort_key(path: Path, reference_now: dt.datetime | None = None) -> tuple[int, str, float, str]:
    stamp_dt = parsed_artifact_stamp(path)
    effective_now = reference_now or now_utc()
    future_cutoff = effective_now + dt.timedelta(minutes=FUTURE_STAMP_GRACE_MINUTES)
    is_future = bool(stamp_dt and stamp_dt > future_cutoff)
    return (0 if is_future else 1, artifact_stamp(path), path.stat().st_mtime, path.name)


def load_json_mapping(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def normalize_string_list(raw: Any, *, default: tuple[str, ...]) -> list[str]:
    if isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw]
    else:
        items = [str(x).strip() for x in str(raw).split(",")]
    out = [item for item in items if item]
    return out if out else list(default)


def load_shortline_policy(config_path: Path) -> dict[str, Any]:
    payload: Any = {}
    source = "builtin_default"
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            source = str(config_path)
        except Exception:
            payload = {}
            source = f"invalid_config:{config_path}"
    shortline = payload.get("shortline", {}) if isinstance(payload, dict) else {}
    return {
        "source": source,
        "status": str(shortline.get("status", "builtin_default")),
        "default_market_state": str(shortline.get("default_market_state", "Bias_Only")),
        "setup_ready_state": str(shortline.get("setup_ready_state", "Setup_Ready")),
        "no_trade_rule": str(shortline.get("no_trade_rule", "no_sweep_no_mss_no_cvd_no_trade")),
        "profile_lookback_bars": int(shortline.get("profile_lookback_bars", 120) or 120),
        "location_priority": normalize_string_list(
            shortline.get("location_priority", DEFAULT_LOCATION_PRIORITY),
            default=DEFAULT_LOCATION_PRIORITY,
        ),
        "supported_symbols": [s.upper() for s in normalize_string_list(
            shortline.get("supported_symbols", DEFAULT_SHORTLINE_SUPPORTED),
            default=DEFAULT_SHORTLINE_SUPPORTED,
        )],
        "cvd_key_level_only": bool(shortline.get("cvd_key_level_only", True)),
        "cvd_local_window_minutes": int(
            shortline.get("cvd_local_window_minutes", DEFAULT_CVD_LOCAL_WINDOW_MINUTES)
            or DEFAULT_CVD_LOCAL_WINDOW_MINUTES
        ),
        "cvd_reference_max_age_minutes": int(
            shortline.get(
                "cvd_reference_max_age_minutes",
                shortline.get("cvd_local_window_minutes", DEFAULT_CVD_REFERENCE_MAX_AGE_MINUTES),
            )
            or DEFAULT_CVD_REFERENCE_MAX_AGE_MINUTES
        ),
        "cvd_drift_guard_enabled": bool(shortline.get("cvd_drift_guard_enabled", True)),
        "cvd_attack_confirmation_required": bool(
            shortline.get("cvd_attack_confirmation_required", True)
        ),
        "cvd_attack_alignment_min_abs": float(
            shortline.get("cvd_attack_alignment_min_abs", DEFAULT_CVD_ATTACK_ALIGNMENT_MIN_ABS)
            or DEFAULT_CVD_ATTACK_ALIGNMENT_MIN_ABS
        ),
        "trigger_stack": normalize_string_list(
            shortline.get("trigger_stack", DEFAULT_TRIGGER_STACK),
            default=DEFAULT_TRIGGER_STACK,
        ),
    }


def latest_artifact_by_suffix(
    review_dir: Path,
    suffix: str,
    reference_now: dt.datetime | None = None,
) -> Path | None:
    candidates = list(review_dir.glob(f"*_{suffix}.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: artifact_sort_key(path, reference_now))


def find_latest_micro_capture(artifact_dir: Path) -> Path | None:
    files = sorted(
        artifact_dir.glob("*_micro_capture.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def read_bars_file(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    frame = pd.read_csv(path)
    if columns:
        available = [column for column in columns if column in frame.columns]
        if available:
            return frame.loc[:, available]
    return frame


def find_latest_bars_file(output_root: Path, symbols: list[str]) -> Path | None:
    want = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    best_path: Path | None = None
    best_key: tuple[int, float, str] | None = None
    candidates = sorted(
        list(output_root.glob("research/*/bars_used.parquet"))
        + list(output_root.glob("research/*/bars_used.csv")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            frame = read_bars_file(path, columns=["symbol"])
        except Exception:
            continue
        if "symbol" not in frame.columns or frame.empty:
            continue
        present = {str(x).strip().upper() for x in frame["symbol"].astype(str).unique()}
        overlap = len(want & present)
        if overlap <= 0:
            continue
        candidate_key = (overlap, path.stat().st_mtime, path.name)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_path = path
    return best_path


def find_latest_live_bars_snapshot(
    review_dir: Path,
    reference_now: dt.datetime | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    snapshot_path = latest_artifact_by_suffix(
        review_dir,
        "crypto_shortline_live_bars_snapshot",
        reference_now,
    )
    payload = load_json_mapping(snapshot_path) if snapshot_path else None
    return snapshot_path, payload


def materialize_gate_bars(
    *,
    review_dir: Path,
    stamp: str,
    frame: pd.DataFrame,
) -> Path:
    materialized_path = review_dir / f"{stamp}_crypto_shortline_execution_gate_bars.csv"
    frame.to_csv(materialized_path, index=False)
    return materialized_path


def live_snapshot_symbols(payload: dict[str, Any] | None, fallback_route_symbol: str) -> list[str]:
    scoped = payload if isinstance(payload, dict) else {}
    symbols = [
        str(item).strip().upper()
        for item in scoped.get("materialized_symbols", [])
        if str(item).strip()
    ]
    if symbols:
        return list(dict.fromkeys(symbols))
    route_symbol = str(scoped.get("route_symbol") or "").strip().upper() or fallback_route_symbol
    return [route_symbol] if route_symbol else []


def volume_profile_location(
    frame: pd.DataFrame,
    *,
    lookback_bars: int,
    bins: int = 12,
) -> dict[str, Any]:
    work = frame.tail(max(6, int(lookback_bars))).copy()
    if work.empty:
        return {
            "location_tag": "MID",
            "current_bin_volume": 0.0,
            "poc_bin_volume": 0.0,
            "bins": 0,
        }

    closes = pd.to_numeric(work["close"], errors="coerce").ffill().bfill()
    volumes = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    lo = float(closes.min())
    hi = float(closes.max())
    if not math.isfinite(lo) or not math.isfinite(hi):
        return {
            "location_tag": "MID",
            "current_bin_volume": 0.0,
            "poc_bin_volume": 0.0,
            "bins": 0,
        }
    if abs(hi - lo) <= 1e-9:
        return {
            "location_tag": "POC",
            "current_bin_volume": float(volumes.sum()),
            "poc_bin_volume": float(volumes.sum()),
            "bins": 1,
        }

    bin_count = max(6, min(24, int(bins)))
    edges = pd.interval_range(start=lo, end=hi, periods=bin_count)
    node_volumes = [0.0 for _ in range(bin_count)]
    current_idx = 0
    for idx, (close_px, vol) in enumerate(zip(closes.tolist(), volumes.tolist(), strict=False)):
        node_idx = min(
            max(int(((float(close_px) - lo) / max(1e-9, hi - lo)) * bin_count), 0),
            bin_count - 1,
        )
        node_volumes[node_idx] += float(max(0.0, vol))
        if idx == len(closes) - 1:
            current_idx = node_idx
    current_volume = float(node_volumes[current_idx])
    poc_volume = float(max(node_volumes) if node_volumes else 0.0)
    positive = sorted(v for v in node_volumes if v > 0.0)
    is_poc = bool(node_volumes) and current_idx == int(node_volumes.index(poc_volume))
    if is_poc:
        tag = "POC"
    elif positive:
        hvn_threshold = positive[max(0, int(0.75 * (len(positive) - 1)))]
        lvn_threshold = positive[max(0, int(0.25 * (len(positive) - 1)))]
        if current_volume >= hvn_threshold:
            tag = "HVN"
        elif current_volume <= lvn_threshold:
            tag = "LVN"
        else:
            tag = "MID"
    else:
        tag = "MID"
    return {
        "location_tag": tag,
        "current_bin_volume": current_volume,
        "poc_bin_volume": poc_volume,
        "bins": bin_count,
    }


def classify_route_state(action: str) -> str:
    action_text = str(action or "").strip()
    if action_text == "deploy_price_state_only":
        return "promoted"
    if action_text in {"candidate_flow_secondary", "watch_priority_until_long_window_confirms", "watch_short_window_flow_priority"}:
        return "review"
    return "watch"


def safe_float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def structure_signals(frame: pd.DataFrame) -> dict[str, Any]:
    enriched = add_common_features(frame)
    ict_long, ict_short = _ict_scores(enriched)
    cur = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) >= 2 else cur
    recent = enriched.tail(min(4, len(enriched))).copy()
    roll_high10_prev = float(cur.get("roll_high10_prev") or 0.0)
    roll_low10_prev = float(cur.get("roll_low10_prev") or 0.0)
    close_px = float(cur.get("close") or 0.0)
    high_px = float(cur.get("high") or 0.0)
    low_px = float(cur.get("low") or 0.0)
    vol = float(cur.get("volume") or 0.0)
    vol_ma20 = float(cur.get("vol_ma20") or 0.0)
    prev_high = float(prev.get("high") or 0.0)
    prev_low = float(prev.get("low") or 0.0)
    sweep_long = bool(
        ((recent["low"] < recent["roll_low10_prev"]) & (recent["close"] > recent["roll_low10_prev"])).any()
    )
    sweep_short = bool(
        ((recent["high"] > recent["roll_high10_prev"]) & (recent["close"] < recent["roll_high10_prev"])).any()
    )
    mss_long = bool((recent["close"] > recent["roll_high10_prev"]).any())
    mss_short = bool((recent["close"] < recent["roll_low10_prev"]).any())
    shifted_high = recent["high"].shift(1)
    shifted_low = recent["low"].shift(1)
    fvg_long = bool(((recent["low"] > shifted_high) & (recent["volume"] >= 1.2 * recent["vol_ma20"])).fillna(False).any())
    fvg_short = bool(((recent["high"] < shifted_low) & (recent["volume"] >= 1.2 * recent["vol_ma20"])).fillna(False).any())
    candle_long = bool(close_px >= float(cur.get("open") or close_px))
    candle_short = bool(close_px <= float(cur.get("open") or close_px))
    return {
        "ict_long": float(ict_long),
        "ict_short": float(ict_short),
        "sweep_long": sweep_long,
        "sweep_short": sweep_short,
        "mss_long": mss_long,
        "mss_short": mss_short,
        "fvg_long": fvg_long,
        "fvg_short": fvg_short,
        "candle_long": candle_long,
        "candle_short": candle_short,
    }


def micro_signals(row: dict[str, Any] | None, *, shortline: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {
            "cvd_ready": False,
            "cvd_long": False,
            "cvd_short": False,
            "quality_ok": False,
            "trust_ok": False,
            "context": "missing",
            "micro_alignment": 0.0,
            "veto_hint": "missing_micro_capture",
            "local_window_ok": False,
            "cvd_drift_risk": False,
            "cvd_locality_status": "missing_micro_capture",
            "cvd_reference_age_minutes": None,
            "key_level_confirmed": False,
            "attack_side": "unknown",
            "attack_presence": "missing_micro_capture",
            "attack_confirmation_ok": False,
        }
    trade_count = int(row.get("trade_count") or 0)
    evidence_score = float(row.get("evidence_score") or 0.0)
    trust = str(row.get("cvd_trust_tier_hint") or "").strip()
    veto = str(row.get("cvd_veto_hint") or "").strip()
    context = str(row.get("cvd_context_mode") or "unclear").strip()
    alignment = float(row.get("micro_alignment") or 0.0)
    note = str(row.get("cvd_context_note") or "").strip().lower()
    local_window_minutes = int(shortline.get("cvd_local_window_minutes") or DEFAULT_CVD_LOCAL_WINDOW_MINUTES)
    max_age_minutes = int(
        shortline.get("cvd_reference_max_age_minutes") or DEFAULT_CVD_REFERENCE_MAX_AGE_MINUTES
    )
    attack_min_abs = float(
        shortline.get("cvd_attack_alignment_min_abs") or DEFAULT_CVD_ATTACK_ALIGNMENT_MIN_ABS
    )
    reference_age_minutes = safe_float(
        row.get("cvd_reference_age_minutes")
        or row.get("cvd_local_reference_age_minutes")
        or row.get("cvd_local_age_minutes")
    )
    locality_status = str(row.get("cvd_locality_status") or "").strip()
    local_window_ok = True if reference_age_minutes is None else reference_age_minutes <= float(local_window_minutes)
    if not locality_status:
        locality_status = (
            "proxy_from_current_snapshot"
            if reference_age_minutes is None
            else ("local_window_ok" if local_window_ok else "outside_local_window")
        )
    drift_risk = bool(row.get("cvd_drift_risk", False))
    if not drift_risk and "drift" in note:
        drift_risk = True
    if (
        bool(shortline.get("cvd_drift_guard_enabled", True))
        and reference_age_minutes is not None
        and reference_age_minutes > float(max_age_minutes)
    ):
        drift_risk = True
    near_key_level_raw = row.get("cvd_near_key_level")
    key_level_confirmed = True if near_key_level_raw in {None, ""} else bool(near_key_level_raw)
    attack_side = str(row.get("cvd_attack_side") or "").strip().lower()
    if attack_side not in {"buyers", "sellers", "balanced"}:
        if alignment >= attack_min_abs:
            attack_side = "buyers"
        elif alignment <= -attack_min_abs:
            attack_side = "sellers"
        else:
            attack_side = "balanced"
    attack_presence = str(row.get("cvd_attack_presence") or "").strip().lower()
    if not attack_presence:
        attack_presence = {
            "buyers": "buyers_attacking",
            "sellers": "sellers_attacking",
            "balanced": "no_clear_attack",
        }.get(attack_side, "no_clear_attack")
    quality_ok = bool(
        row.get("schema_ok", False)
        and row.get("time_sync_ok", False)
        and row.get("gap_ok", False)
        and trade_count >= 20
        and evidence_score >= 0.75
    )
    trust_ok = trust == "single_exchange_ok"
    attack_confirmation_ok = bool(
        (not bool(shortline.get("cvd_attack_confirmation_required", True)))
        or attack_side in {"buyers", "sellers"}
    )
    cvd_ready = bool(
        quality_ok
        and trust_ok
        and not veto
        and context in {"continuation", "reversal", "absorption", "failed_auction"}
        and local_window_ok
        and not drift_risk
        and attack_confirmation_ok
    )
    return {
        "cvd_ready": cvd_ready,
        "cvd_long": bool(cvd_ready and attack_side == "buyers" and alignment >= attack_min_abs),
        "cvd_short": bool(cvd_ready and attack_side == "sellers" and alignment <= -attack_min_abs),
        "quality_ok": quality_ok,
        "trust_ok": trust_ok,
        "context": context,
        "micro_alignment": alignment,
        "veto_hint": veto,
        "trade_count": trade_count,
        "evidence_score": evidence_score,
        "local_window_ok": local_window_ok,
        "cvd_drift_risk": drift_risk,
        "cvd_locality_status": locality_status,
        "cvd_reference_age_minutes": reference_age_minutes,
        "key_level_confirmed": key_level_confirmed,
        "attack_side": attack_side,
        "attack_presence": attack_presence,
        "attack_confirmation_ok": attack_confirmation_ok,
    }


def build_symbol_gate(
    *,
    symbol: str,
    route_row: dict[str, Any],
    bars: pd.DataFrame,
    micro_row: dict[str, Any] | None,
    shortline: dict[str, Any],
) -> dict[str, Any]:
    structure = structure_signals(bars)
    profile = volume_profile_location(
        bars,
        lookback_bars=int(shortline.get("profile_lookback_bars") or 120),
    )
    micro = micro_signals(micro_row, shortline=shortline)
    location_tag = str(profile.get("location_tag") or "MID")
    location_ok = location_tag in {str(x).strip() for x in shortline.get("location_priority", DEFAULT_LOCATION_PRIORITY)}
    key_level_context_ok = bool(
        location_ok and (not bool(shortline.get("cvd_key_level_only", True)) or micro["key_level_confirmed"])
    )
    route_action = str(route_row.get("action") or "").strip()
    route_state = classify_route_state(route_action)

    long_ready = bool(
        key_level_context_ok
        and structure["sweep_long"]
        and structure["mss_long"]
        and structure["fvg_long"]
        and structure["candle_long"]
        and micro["cvd_long"]
        and route_state == "promoted"
    )
    short_ready = bool(
        key_level_context_ok
        and structure["sweep_short"]
        and structure["mss_short"]
        and structure["fvg_short"]
        and structure["candle_short"]
        and micro["cvd_short"]
        and route_state == "promoted"
    )

    state = str(shortline.get("default_market_state") or "Bias_Only")
    setup_ready = str(shortline.get("setup_ready_state") or "Setup_Ready")
    direction = "-"
    missing: list[str] = []
    if long_ready:
        state = setup_ready
        direction = "LONG"
    elif short_ready:
        state = setup_ready
        direction = "SHORT"
    else:
        if not location_ok:
            missing.append(f"profile_location={location_tag}")
        if bool(shortline.get("cvd_key_level_only", True)) and not key_level_context_ok:
            missing.append("cvd_key_level_context")
        if not (structure["sweep_long"] or structure["sweep_short"]):
            missing.append("liquidity_sweep")
        if not (structure["mss_long"] or structure["mss_short"]):
            missing.append("mss")
        if not (structure["fvg_long"] or structure["fvg_short"]):
            missing.append("fvg_ob_breaker_retest")
        if not micro["local_window_ok"]:
            missing.append("cvd_local_window")
        if bool(shortline.get("cvd_drift_guard_enabled", True)) and micro["cvd_drift_risk"]:
            missing.append("cvd_drift_guard")
        if not micro["attack_confirmation_ok"]:
            missing.append("cvd_attack_confirmation")
        if not (micro["cvd_long"] or micro["cvd_short"]):
            missing.append("cvd_confirmation")
        if route_state != "promoted":
            missing.append(f"route_state={route_state}:{route_action or '-'}")
        if not missing:
            missing.append("reversal_or_breakout_candle")
    missing = list(dict.fromkeys(missing))

    pattern_family_hint = ""
    pattern_stage_hint = ""
    pattern_hint_brief = ""
    effective_missing = list(missing)
    if (
        state != setup_ready
        and micro.get("context") == "continuation"
        and bool(micro.get("key_level_confirmed"))
        and bool(micro.get("attack_confirmation_ok"))
    ):
        pattern_family_hint = "imbalance_continuation"
        continuation_missing: list[str] = []
        if not (structure["fvg_long"] or structure["fvg_short"]):
            pattern_stage_hint = "imbalance_retest"
            continuation_missing.append("fvg_ob_breaker_retest")
        elif not (micro["cvd_long"] or micro["cvd_short"]):
            pattern_stage_hint = "cvd_confirmation"
            continuation_missing.append("cvd_confirmation")
        elif route_state != "promoted":
            pattern_stage_hint = "route_promotion"
        else:
            pattern_stage_hint = "continuation_candidate"
        if route_state != "promoted":
            continuation_missing.append(f"route_state={route_state}:{route_action or '-'}")
        effective_missing = list(dict.fromkeys(continuation_missing or effective_missing))
        pattern_hint_brief = ":".join(
            [
                pattern_family_hint,
                pattern_stage_hint or "raw_gate",
                ",".join(effective_missing) or "ready",
            ]
        )

    blocker = (
        f"{symbol} remains {state}; missing "
        + ", ".join(missing)
        + f"; {str(shortline.get('no_trade_rule') or 'no_sweep_no_mss_no_cvd_no_trade')}."
        if state != setup_ready
        else f"{symbol} completed profile/sweep/mss/retest/cvd gate and is {setup_ready}."
    )
    if state != setup_ready and pattern_hint_brief:
        blocker_parts = [
            f"{symbol} remains {state}",
            f"pattern_hint={pattern_family_hint}:{pattern_stage_hint or 'raw_gate'}",
            f"effective_missing={','.join(effective_missing) or '-'}",
            f"raw_missing={','.join(missing) or '-'}",
            str(shortline.get("no_trade_rule") or "no_sweep_no_mss_no_cvd_no_trade"),
        ]
        blocker = "; ".join(part for part in blocker_parts if str(part).strip()) + "."
    done_when = (
        f"{symbol} completes {' and '.join(effective_missing or missing)}"
        if state != setup_ready
        else f"{symbol} loses setup quality or leaves {setup_ready}"
    )
    return {
        "symbol": symbol,
        "route_action": route_action,
        "route_state": route_state,
        "location_tag": location_tag,
        "execution_state": state,
        "setup_direction": direction,
        "blocker_detail": blocker,
        "done_when": done_when,
        "missing_gates": missing,
        "effective_missing_gates": effective_missing,
        "pattern_family_hint": pattern_family_hint,
        "pattern_stage_hint": pattern_stage_hint,
        "pattern_hint_brief": pattern_hint_brief,
        "structure_signals": structure,
        "micro_signals": micro,
        "profile_proxy": profile,
        "key_level_context_ok": key_level_context_ok,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Crypto Shortline Execution Gate",
        "",
        f"- as_of: `{payload.get('as_of') or ''}`",
        f"- route_handoff_artifact: `{payload.get('route_handoff_artifact') or ''}`",
        f"- bars_artifact: `{payload.get('bars_artifact') or ''}`",
        f"- bars_input_artifact: `{payload.get('bars_input_artifact') or ''}`",
        f"- bars_source_kind: `{payload.get('bars_source_kind') or ''}`",
        f"- live_bars_snapshot_artifact: `{payload.get('live_bars_snapshot_artifact') or ''}`",
        f"- micro_capture_artifact: `{payload.get('micro_capture_artifact') or ''}`",
        f"- setup_ready_symbols: `{', '.join(payload.get('setup_ready_symbols', [])) or '-'}`",
        f"- bias_only_symbols: `{', '.join(payload.get('bias_only_symbols', [])) or '-'}`",
        f"- gate_brief: `{payload.get('gate_brief') or '-'}`",
        "",
        "## Symbols",
    ]
    for row in payload.get("symbols", []):
        if not isinstance(row, dict):
            continue
        lines.extend(
            [
                f"- `{row.get('symbol')}` state=`{row.get('execution_state')}` dir=`{row.get('setup_direction')}` route=`{row.get('route_action') or '-'}` location=`{row.get('location_tag') or '-'}`",
                f"  - blocker: `{row.get('blocker_detail') or '-'}`",
                f"  - done_when: `{row.get('done_when') or '-'}`",
                f"  - missing: `{', '.join(row.get('missing_gates', [])) or '-'}`",
                (
                    "  - micro_local: `"
                    + f"{((row.get('micro_signals') or {}).get('cvd_locality_status') if isinstance(row.get('micro_signals'), dict) else '-') or '-'}"
                    + " | drift="
                    + f"{((row.get('micro_signals') or {}).get('cvd_drift_risk') if isinstance(row.get('micro_signals'), dict) else '-')}"
                    + " | attack="
                    + f"{((row.get('micro_signals') or {}).get('attack_side') if isinstance(row.get('micro_signals'), dict) else '-') or '-'}"
                    + "`"
                ),
            ]
        )
    return "\n".join(lines).strip() + "\n"


def prune_artifacts(
    review_dir: Path,
    *,
    current_artifact: Path,
    current_checksum: Path,
    current_markdown: Path,
    current_bars_path: Path,
    keep: int,
    ttl_hours: float,
) -> tuple[list[str], list[str]]:
    cutoff = now_utc() - dt.timedelta(hours=max(1.0, ttl_hours))
    protected = {
        current_artifact.name,
        current_checksum.name,
        current_markdown.name,
        current_bars_path.name,
    }
    candidates: list[Path] = []
    for pattern in (
        "*_crypto_shortline_execution_gate.json",
        "*_crypto_shortline_execution_gate_checksum.json",
        "*_crypto_shortline_execution_gate.md",
        "*_crypto_shortline_execution_gate_bars.csv",
    ):
        candidates.extend(review_dir.glob(pattern))

    pruned_age: list[str] = []
    survivors: list[Path] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
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
    for path in survivors[keep:]:
        if path.name in protected:
            continue
        path.unlink(missing_ok=True)
        pruned_keep.append(str(path))
    return pruned_keep, pruned_age


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic crypto shortline execution gate artifact.")
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_MICRO_CAPTURE_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--route-handoff-file", default="")
    parser.add_argument("--bars-file", default="")
    parser.add_argument("--micro-capture-file", default="")
    parser.add_argument("--live-bars-snapshot-file", default="")
    parser.add_argument("--now", default="")
    parser.add_argument("--artifact-ttl-hours", type=float, default=168.0)
    parser.add_argument("--artifact-keep", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    review_dir = Path(args.review_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    config_path = resolve_path(args.config, anchor=SYSTEM_ROOT)
    runtime_now = parse_now(args.now)
    review_dir.mkdir(parents=True, exist_ok=True)

    shortline = load_shortline_policy(config_path)
    symbols = list(shortline.get("supported_symbols", DEFAULT_SHORTLINE_SUPPORTED))
    route_path = (
        Path(args.route_handoff_file).expanduser().resolve()
        if str(args.route_handoff_file).strip()
        else latest_artifact_by_suffix(review_dir, "binance_indicator_symbol_route_handoff", runtime_now)
    )
    route_payload = load_json_mapping(route_path)
    if route_payload is None:
        raise FileNotFoundError("no_binance_indicator_symbol_route_handoff_artifact")

    stamp = runtime_now.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    explicit_bars_path = (
        Path(args.bars_file).expanduser().resolve()
        if str(args.bars_file).strip()
        else None
    )
    explicit_live_snapshot_path = (
        Path(args.live_bars_snapshot_file).expanduser().resolve()
        if str(args.live_bars_snapshot_file).strip()
        else None
    )
    live_snapshot_path, live_snapshot_payload = (
        (explicit_live_snapshot_path, load_json_mapping(explicit_live_snapshot_path))
        if explicit_live_snapshot_path and explicit_live_snapshot_path.exists()
        else find_latest_live_bars_snapshot(review_dir, runtime_now)
    )

    bars_source_kind = "local_research_bars"
    bars_input_artifact = ""
    live_bars_snapshot_artifact = str(live_snapshot_path) if live_snapshot_path else ""
    base_bars_path = explicit_bars_path or find_latest_bars_file(output_root, symbols)
    base_bars_frame = read_bars_file(base_bars_path) if base_bars_path and base_bars_path.exists() else None
    route_rows = list(route_payload.get("routes") or [])
    route_symbols = [
        str(row.get("symbol") or "").strip().upper()
        for row in route_rows
        if isinstance(row, dict) and str(row.get("symbol") or "").strip()
    ]
    route_symbol = route_symbols[0] if route_symbols else ""
    bars_path: Path | None = explicit_bars_path or base_bars_path
    bars = base_bars_frame.copy() if isinstance(base_bars_frame, pd.DataFrame) else None

    if explicit_bars_path is not None:
        if not explicit_bars_path.exists():
            raise FileNotFoundError("explicit_bars_artifact_missing")
        bars_source_kind = "explicit_bars_file"
        bars_input_artifact = str(explicit_bars_path)
        bars = read_bars_file(explicit_bars_path)
        bars_path = explicit_bars_path
    else:
        live_bars_artifact = str((live_snapshot_payload or {}).get("bars_artifact") or "").strip()
        live_symbols = live_snapshot_symbols(live_snapshot_payload, route_symbol)
        if live_bars_artifact:
            candidate_live_bars_path = Path(live_bars_artifact).expanduser().resolve()
            if candidate_live_bars_path.exists():
                live_frame = read_bars_file(candidate_live_bars_path)
                if base_bars_frame is not None and not base_bars_frame.empty and live_symbols:
                    live_symbol_set = {symbol for symbol in live_symbols if symbol}
                    merged_frame = base_bars_frame[
                        ~base_bars_frame["symbol"].astype(str).str.upper().isin(live_symbol_set)
                    ].copy()
                    live_symbol_frame = live_frame[
                        live_frame["symbol"].astype(str).str.upper().isin(live_symbol_set)
                    ].copy()
                    if not live_symbol_frame.empty:
                        merged_frame = pd.concat([merged_frame, live_symbol_frame], ignore_index=True)
                        bars = merged_frame
                        bars_source_kind = "merged_base_plus_live_snapshot"
                        bars_input_artifact = str(base_bars_path) if base_bars_path else ""
                        bars_path = materialize_gate_bars(review_dir=review_dir, stamp=stamp, frame=bars)
                if bars is None and not live_frame.empty:
                    bars = live_frame
                    bars_source_kind = "live_snapshot_only"
                    bars_input_artifact = str(candidate_live_bars_path)
                    bars_path = materialize_gate_bars(review_dir=review_dir, stamp=stamp, frame=bars)

        if bars is None and base_bars_path and base_bars_path.exists():
            bars = base_bars_frame
            bars_source_kind = "local_research_bars"
            bars_input_artifact = str(base_bars_path)
            bars_path = base_bars_path

    if bars is None or bars_path is None or not bars_path.exists():
        raise FileNotFoundError("no_local_bars_artifact")

    micro_path = (
        Path(args.micro_capture_file).expanduser().resolve()
        if str(args.micro_capture_file).strip()
        else find_latest_micro_capture(artifact_dir)
    )
    micro_payload = load_json_mapping(micro_path)
    selected_micro = list((micro_payload or {}).get("selected_micro") or [])
    micro_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in selected_micro
        if isinstance(row, dict) and str(row.get("symbol") or "").strip()
    }
    routes = list(route_payload.get("routes") or [])
    route_by_symbol = {
        str(row.get("symbol") or "").strip().upper(): dict(row)
        for row in routes
        if isinstance(row, dict) and str(row.get("symbol") or "").strip()
    }

    symbol_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol_frame = bars[bars["symbol"].astype(str).str.upper() == symbol].copy()
        if symbol_frame.empty:
            continue
        route_row = route_by_symbol.get(symbol, {})
        row = build_symbol_gate(
            symbol=symbol,
            route_row=route_row,
            bars=symbol_frame.sort_values("ts").reset_index(drop=True),
            micro_row=micro_by_symbol.get(symbol),
            shortline=shortline,
        )
        symbol_rows.append(row)

    setup_ready_symbols = [str(row["symbol"]) for row in symbol_rows if row.get("execution_state") == shortline.get("setup_ready_state")]
    bias_only_symbols = [str(row["symbol"]) for row in symbol_rows if row.get("execution_state") == shortline.get("default_market_state")]
    gate_brief = (
        f"setup_ready:{','.join(setup_ready_symbols)}"
        if setup_ready_symbols
        else f"bias_only:{','.join(bias_only_symbols) if bias_only_symbols else '-'}"
    )

    artifact_path = review_dir / f"{stamp}_crypto_shortline_execution_gate.json"
    markdown_path = review_dir / f"{stamp}_crypto_shortline_execution_gate.md"
    checksum_path = review_dir / f"{stamp}_crypto_shortline_execution_gate_checksum.json"
    payload = {
        "ok": True,
        "status": "ok",
        "as_of": fmt_utc(runtime_now),
        "config_path": str(config_path),
        "route_handoff_artifact": str(route_path) if route_path else "",
        "bars_artifact": str(bars_path),
        "bars_input_artifact": bars_input_artifact,
        "bars_source_kind": bars_source_kind,
        "live_bars_snapshot_artifact": live_bars_snapshot_artifact,
        "micro_capture_artifact": str(micro_path) if micro_path else "",
        "shortline_policy": shortline,
        "symbols": symbol_rows,
        "setup_ready_symbols": setup_ready_symbols,
        "bias_only_symbols": bias_only_symbols,
        "gate_brief": gate_brief,
    }
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    checksum_payload = {
        "generated_at_utc": payload["as_of"],
        "files": [
            {"path": str(artifact_path), "sha256": sha256_file(artifact_path)},
            {"path": str(markdown_path), "sha256": sha256_file(markdown_path)},
            {"path": str(bars_path), "sha256": sha256_file(bars_path)},
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pruned_keep, pruned_age = prune_artifacts(
        review_dir,
        current_artifact=artifact_path,
        current_checksum=checksum_path,
        current_markdown=markdown_path,
        current_bars_path=bars_path,
        keep=max(3, int(args.artifact_keep)),
        ttl_hours=float(args.artifact_ttl_hours),
    )
    payload.update(
        {
            "artifact": str(artifact_path),
            "markdown": str(markdown_path),
            "checksum": str(checksum_path),
            "pruned_keep": pruned_keep,
            "pruned_age": pruned_age,
        }
    )
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_payload["files"][0]["sha256"] = sha256_file(artifact_path)
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
