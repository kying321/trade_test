#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XAUUSD")
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_MIN_CONFIDENCE = 60.0
DEFAULT_MIN_CONVEXITY = 3.0
DEFAULT_MAX_AGE_DAYS = 14
DEFAULT_BASE_RISK_PCT = 0.45
DEFAULT_MAX_ALLOC_PER_TRADE_PCT = 0.30
DEFAULT_MIN_NOTIONAL_USDT = 5.0
DEFAULT_ARTIFACT_TTL_HOURS = 168
MAX_SIGNAL_SOURCE_CANDIDATES = 32
DEFAULT_TAKEOVER_MARKET = "spot"
SUPPORTED_TAKEOVER_MARKETS = {"spot", "futures_usdm", "portfolio_margin_um"}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_utc_iso() -> str:
    return now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc_compact() -> str:
    return now_utc().strftime("%Y%m%dT%H%M%SZ")


def parse_now(raw: str) -> datetime | None:
    text_value = str(raw).strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_path(raw: str, *, anchor: Path) -> Path:
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return anchor / path


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def normalize_symbols(raw: str) -> tuple[str, ...]:
    parts = [str(x).strip().upper() for x in str(raw).split(",")]
    out = [x for x in parts if x]
    return tuple(out) if out else DEFAULT_SYMBOLS


def parse_takeover_market(raw: Any) -> str:
    market = str(raw or "").strip().lower()
    return market if market in SUPPORTED_TAKEOVER_MARKETS else ""


def should_auto_narrow_symbols(raw: str) -> bool:
    normalized = normalize_symbols(raw)
    return normalized == DEFAULT_SYMBOLS


def load_policy_thresholds(config_path: Path) -> dict[str, Any]:
    payload: Any = {}
    source = "builtin_default"
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            source = str(config_path)
        except Exception:
            payload = {}
            source = f"invalid_config:{config_path}"
    thresholds = payload.get("thresholds", {}) if isinstance(payload, dict) else {}
    return {
        "signal_confidence_min": float(to_float(thresholds.get("signal_confidence_min", DEFAULT_MIN_CONFIDENCE), DEFAULT_MIN_CONFIDENCE)),
        "convexity_min": float(to_float(thresholds.get("convexity_min", DEFAULT_MIN_CONVEXITY), DEFAULT_MIN_CONVEXITY)),
        "source": source,
        "source_kind": "config_thresholds",
        "runtime_mode": "",
    }


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def derive_policy_thresholds(output_root: Path, *, config_path: Path, as_of: date) -> dict[str, Any]:
    config_thresholds = load_policy_thresholds(config_path)
    runtime_params_live_candidates = [
        output_root / "artifacts" / "runtime_params_live.json",
        output_root / "state" / "output" / "artifacts" / "runtime_params_live.json",
    ]
    for path in runtime_params_live_candidates:
        if not path.exists():
            continue
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        has_override = False
        signal_conf = config_thresholds["signal_confidence_min"]
        convexity = config_thresholds["convexity_min"]
        if "signal_confidence_min" in payload:
            signal_conf = float(to_float(payload.get("signal_confidence_min"), signal_conf))
            has_override = True
        if "convexity_min" in payload:
            convexity = float(to_float(payload.get("convexity_min"), convexity))
            has_override = True
        if has_override:
            return {
                "signal_confidence_min": signal_conf,
                "convexity_min": convexity,
                "source": str(path),
                "source_kind": "runtime_params_live",
                "runtime_mode": str(payload.get("runtime_mode", "")).strip(),
            }

    params_live_path = output_root / "artifacts" / "params_live.yaml"
    if params_live_path.exists():
        params_live = _load_yaml_mapping(params_live_path)
        has_override = False
        signal_conf = config_thresholds["signal_confidence_min"]
        convexity = config_thresholds["convexity_min"]
        if "signal_confidence_min" in params_live:
            signal_conf = float(to_float(params_live.get("signal_confidence_min"), signal_conf))
            has_override = True
        if "convexity_min" in params_live:
            convexity = float(to_float(params_live.get("convexity_min"), convexity))
            has_override = True
        if has_override:
            return {
                "signal_confidence_min": signal_conf,
                "convexity_min": convexity,
                "source": str(params_live_path),
                "source_kind": "params_live",
                "runtime_mode": "",
            }

    candidates = [
        output_root / "daily" / f"{as_of.isoformat()}_mode_feedback.json",
        output_root / "state" / "output" / "daily" / f"{as_of.isoformat()}_mode_feedback.json",
    ]
    fallback_dir = output_root / "state" / "output" / "daily"
    if fallback_dir.exists():
        candidates.extend(sorted(fallback_dir.glob("*_mode_feedback.json"), reverse=True))
    for path in candidates:
        if not path.exists():
            continue
        payload = read_json(path, {})
        runtime_params = payload.get("runtime_params", {}) if isinstance(payload, dict) else {}
        if not isinstance(runtime_params, dict):
            continue
        has_override = False
        signal_conf = config_thresholds["signal_confidence_min"]
        convexity = config_thresholds["convexity_min"]
        if "signal_confidence_min" in runtime_params:
            signal_conf = float(to_float(runtime_params.get("signal_confidence_min"), signal_conf))
            has_override = True
        if "convexity_min" in runtime_params:
            convexity = float(to_float(runtime_params.get("convexity_min"), convexity))
            has_override = True
        if has_override:
            return {
                "signal_confidence_min": signal_conf,
                "convexity_min": convexity,
                "source": str(path),
                "source_kind": "mode_feedback_runtime_params",
                "runtime_mode": str(payload.get("runtime_mode", "")).strip(),
            }

    return config_thresholds


def normalize_takeover_market(raw: Any, *, default: str = DEFAULT_TAKEOVER_MARKET) -> str:
    market = str(raw or "").strip().lower()
    if market in SUPPORTED_TAKEOVER_MARKETS:
        return market
    return str(default).strip().lower() or DEFAULT_TAKEOVER_MARKET


def load_takeover_market(config_path: Path) -> str:
    payload: Any = {}
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    validation = payload.get("validation", {}) if isinstance(payload, dict) else {}
    return normalize_takeover_market(validation.get("binance_live_takeover_market", DEFAULT_TAKEOVER_MARKET))


def find_latest_review_artifact(review_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(review_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_signal_remote_market_context(*, review_dir: Path, now_dt: datetime, ttl_hours: float) -> dict[str, Any]:
    cutoff = now_dt.timestamp() - (max(1.0, float(ttl_hours)) * 3600.0)
    candidates = (
        ("*_remote_ticket_actionability_state.json", "remote_market", "remote_ticket_actionability_state"),
        ("*_remote_scope_router_state.json", "remote_market", "remote_scope_router_state"),
        ("*_remote_execution_identity_state.json", "ready_check_scope_market", "remote_execution_identity_state"),
        ("*_remote_execution_identity_state.json", "remote_market", "remote_execution_identity_state"),
    )
    for pattern, field, kind in candidates:
        path = find_latest_review_artifact(review_dir, pattern)
        if path is None or not path.exists():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        payload = read_json(path, {})
        if not isinstance(payload, dict):
            continue
        market = parse_takeover_market(payload.get(field))
        if market:
            return {
                "remote_market": market,
                "artifact": str(path),
                "source_kind": kind,
            }
    return {
        "remote_market": "",
        "artifact": "",
        "source_kind": "missing",
    }


def load_latest_takeover_payload(*, output_root: Path, market: str) -> tuple[dict[str, Any], str]:
    review_dir = output_root / "review"
    market_name = normalize_takeover_market(market)
    candidates = [
        review_dir / f"latest_binance_live_takeover_{market_name}.json",
        review_dir / "latest_binance_live_takeover.json",
    ]
    for path in candidates:
        payload = read_json(path, {})
        if not isinstance(payload, dict) or not payload:
            continue
        steps = payload.get("steps", {}) if isinstance(payload.get("steps", {}), dict) else {}
        acct = steps.get("account_overview", {}) if isinstance(steps.get("account_overview", {}), dict) else {}
        payload_market_raw = payload.get("market") or acct.get("market")
        payload_market = normalize_takeover_market(payload_market_raw) if str(payload_market_raw).strip() else ""
        if path.name == "latest_binance_live_takeover.json" and payload_market and payload_market != market_name:
            continue
        return payload, str(path)
    return {}, ""


def iter_signal_source_candidates(output_root: Path, review_dir: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path in sorted(review_dir.glob("*_crypto_shortline_signal_source.json"), reverse=True):
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        candidates.append((path, "crypto_shortline_signal_source"))
        if len(candidates) >= MAX_SIGNAL_SOURCE_CANDIDATES:
            return candidates
    for path in sorted(review_dir.glob("*_strategy_recent5_signals.json"), reverse=True):
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        candidates.append((path, "recent5_review"))
        if len(candidates) >= MAX_SIGNAL_SOURCE_CANDIDATES:
            return candidates
    for directory in (output_root / "daily", output_root / "state" / "output" / "daily"):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*_signals.json"), reverse=True):
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            candidates.append((path, "daily_signals"))
            if len(candidates) >= MAX_SIGNAL_SOURCE_CANDIDATES:
                return candidates
    return candidates


def _parse_signal_date(raw: str, fallback: date) -> date:
    text = str(raw).strip()
    if not text:
        return fallback
    try:
        return date.fromisoformat(text[:10])
    except Exception:
        return fallback


def _artifact_date_fallback(path: Path | None, fallback: date) -> date:
    if path is None:
        return fallback
    name = path.name
    for token in (name[:10], name[:8]):
        try:
            if len(token) == 10 and token[4] == "-" and token[7] == "-":
                return date.fromisoformat(token)
            if len(token) == 8 and token.isdigit():
                return date.fromisoformat(f"{token[:4]}-{token[4:6]}-{token[6:8]}")
        except Exception:
            continue
    return fallback


def load_signal_rows(path: Path | None, *, as_of: date, symbols: tuple[str, ...]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {"kind": "missing", "path": "", "row_count": 0, "source_row_count": 0, "matched_symbol_count": 0, "artifact_date": ""}
    payload = read_json(path, {})
    symbol_set = set(symbols)
    artifact_date = _artifact_date_fallback(path, as_of)
    rows_by_symbol: dict[str, dict[str, Any]] = {}
    row_count = 0
    if isinstance(payload, dict) and isinstance(payload.get("signals"), dict):
        signals_map = payload.get("signals", {})
        source_row_count = 0
        for sym in symbols:
            rows = signals_map.get(sym, []) if isinstance(signals_map, dict) else []
            if not isinstance(rows, list) or not rows:
                continue
            first = rows[0]
            if isinstance(first, dict):
                row = dict(first)
                row["date"] = _parse_signal_date(str(row.get("date", "")), artifact_date).isoformat()
                rows_by_symbol[sym] = row
                row_count += len(rows)
        for rows in signals_map.values():
            if isinstance(rows, list):
                source_row_count += len(rows)
        return rows_by_symbol, {
            "kind": "recent5_review",
            "path": str(path),
            "row_count": row_count,
            "source_row_count": int(source_row_count),
            "matched_symbol_count": int(len(rows_by_symbol)),
            "artifact_date": artifact_date.isoformat(),
        }
    if isinstance(payload, list):
        selected: dict[str, tuple[date, dict[str, Any]]] = {}
        source_row_count = 0
        for raw in payload:
            if not isinstance(raw, dict):
                continue
            source_row_count += 1
            sym = str(raw.get("symbol", "")).strip().upper()
            if sym not in symbol_set:
                continue
            row_date = _parse_signal_date(str(raw.get("date", "")), artifact_date)
            prev = selected.get(sym)
            if prev is None or row_date > prev[0]:
                row = dict(raw)
                row["date"] = row_date.isoformat()
                selected[sym] = (row_date, row)
            row_count += 1
        for sym, (_, row) in selected.items():
            rows_by_symbol[sym] = row
        return rows_by_symbol, {
            "kind": "daily_signals",
            "path": str(path),
            "row_count": row_count,
            "source_row_count": int(source_row_count),
            "matched_symbol_count": int(len(rows_by_symbol)),
            "artifact_date": artifact_date.isoformat(),
        }
    return {}, {
        "kind": "invalid",
        "path": str(path),
        "row_count": 0,
        "source_row_count": 0,
        "matched_symbol_count": 0,
        "artifact_date": artifact_date.isoformat(),
    }


def resolve_signal_source(
    *,
    output_root: Path,
    review_dir: Path,
    as_of: date,
    symbols: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    candidates = iter_signal_source_candidates(output_root, review_dir)
    if not candidates:
        return {}, {
            "kind": "missing",
            "path": "",
            "row_count": 0,
            "source_row_count": 0,
            "matched_symbol_count": 0,
            "candidate_count": 0,
            "selected_candidate_index": -1,
            "fallback_used": False,
            "selection_reason": "missing",
        }

    first_meta: dict[str, Any] | None = None
    candidate_count = len(candidates)
    for idx, (path, kind) in enumerate(candidates):
        rows_by_symbol, meta = load_signal_rows(path, as_of=as_of, symbols=symbols)
        meta["kind"] = kind
        meta["candidate_count"] = int(candidate_count)
        meta["selected_candidate_index"] = int(idx)
        meta["fallback_used"] = bool(idx > 0)
        if rows_by_symbol:
            meta["selection_reason"] = "matched_symbol_rows"
            return rows_by_symbol, meta
        if first_meta is None:
            first_meta = dict(meta)

    if first_meta is None:
        return {}, {
            "kind": "missing",
            "path": "",
            "row_count": 0,
            "source_row_count": 0,
            "matched_symbol_count": 0,
            "candidate_count": int(candidate_count),
            "selected_candidate_index": -1,
            "fallback_used": False,
            "selection_reason": "missing",
        }
    first_meta["selection_reason"] = "no_matching_symbol_rows"
    return {}, first_meta


def derive_equity_usdt(*, output_root: Path, override_equity_usdt: float | None, market: str) -> tuple[float, str]:
    if override_equity_usdt is not None and override_equity_usdt > 0.0:
        return float(override_equity_usdt), "cli_override"
    payload, latest_takeover_source = load_latest_takeover_payload(output_root=output_root, market=market)
    steps = payload.get("steps", {}) if isinstance(payload, dict) else {}
    acct = steps.get("account_overview", {}) if isinstance(steps, dict) else {}
    quote = to_float(acct.get("quote_available", 0.0), 0.0)
    if quote > 0.0:
        latest_name = Path(latest_takeover_source).name if latest_takeover_source else "latest_binance_live_takeover.json"
        return quote, f"{latest_name}.account_overview.quote_available"
    return 0.0, "default_zero"


def derive_risk_multiplier(output_root: Path, *, as_of: date) -> tuple[float, str]:
    candidates = [
        output_root / "daily" / f"{as_of.isoformat()}_mode_feedback.json",
        output_root / "state" / "output" / "daily" / f"{as_of.isoformat()}_mode_feedback.json",
    ]
    fallback_dir = output_root / "state" / "output" / "daily"
    if fallback_dir.exists():
        candidates.extend(sorted(fallback_dir.glob("*_mode_feedback.json"), reverse=True))
    for path in candidates:
        if not path.exists():
            continue
        payload = read_json(path, {})
        risk_control = payload.get("risk_control", {}) if isinstance(payload, dict) else {}
        risk_multiplier = to_float(risk_control.get("risk_multiplier", 0.0), 0.0)
        if risk_multiplier > 0.0:
            return risk_multiplier, str(path)
    return 1.0, "default"


def calc_risk_per_unit_pct(*, entry_price: float, stop_price: float) -> float:
    entry = max(0.0, float(entry_price))
    stop = max(0.0, float(stop_price))
    if entry <= 0.0 or stop <= 0.0:
        return 0.0
    return float(abs(entry - stop) / entry * 100.0)


def calc_conviction(*, confidence: float, convexity: float) -> float:
    return float(min(1.0, max(0.0, confidence) / 100.0) * min(1.0, max(0.0, convexity) / 2.0))


def build_ticket_row(
    *,
    symbol: str,
    signal_row: dict[str, Any] | None,
    as_of: date,
    min_confidence: float,
    min_convexity: float,
    max_age_days: int,
    equity_usdt: float,
    equity_source: str,
    risk_multiplier: float,
    risk_multiplier_source: str,
    base_risk_pct: float,
    max_alloc_per_trade_pct: float,
    min_notional_usdt: float,
    takeover_market: str,
    signal_remote_market: str,
) -> dict[str, Any]:
    market_scope_mismatch = bool(signal_remote_market and signal_remote_market != takeover_market)
    target_market_read_only = takeover_market == "portfolio_margin_um"
    market_scope = {
        "takeover_market": takeover_market,
        "signal_remote_market": signal_remote_market,
        "mismatch": bool(market_scope_mismatch),
        "read_only": bool(target_market_read_only),
    }
    if signal_row is None:
        return {
            "symbol": symbol,
            "date": "",
            "age_days": -1,
            "allowed": False,
            "reasons": ["signal_not_found"],
            "signal": {
                "side": "",
                "regime": "",
                "confidence": 0.0,
                "convexity_ratio": 0.0,
                "combo_id": "",
                "combo_id_canonical": "",
                "confirmation_indicator": "",
                "factor_flags": [],
                "notes": "",
            },
            "levels": {
                "entry_price": 0.0,
                "stop_price": 0.0,
                "target_price": 0.0,
                "risk_per_unit_pct": 0.0,
            },
            "sizing": {
                "equity_usdt": float(equity_usdt),
                "equity_source": equity_source,
                "risk_multiplier": float(risk_multiplier),
                "risk_multiplier_source": risk_multiplier_source,
                "conviction": 0.0,
                "risk_budget_usdt": 0.0,
                "quote_usdt": 0.0,
                "min_notional_usdt": float(min_notional_usdt),
                "max_alloc_pct": float(max_alloc_per_trade_pct),
            },
            "execution": {
                "mode": "NO_SIGNAL",
                "order_type_hint": "LIMIT",
                "max_slippage_bps": 0.0,
            },
            "market_scope": market_scope,
        }

    signal_date = _parse_signal_date(str(signal_row.get("date", "")), as_of)
    side = str(signal_row.get("side", "")).strip().upper()
    confidence = to_float(signal_row.get("confidence", 0.0), 0.0)
    convexity = to_float(signal_row.get("convexity_ratio", 0.0), 0.0)
    entry_price = to_float(signal_row.get("entry_price", 0.0), 0.0)
    stop_price = to_float(signal_row.get("stop_price", 0.0), 0.0)
    target_price = to_float(signal_row.get("target_price", 0.0), 0.0)
    price_reference_kind = str(signal_row.get("price_reference_kind", "")).strip()
    price_reference_source = str(signal_row.get("price_reference_source", "")).strip()
    execution_price_ready = bool(signal_row.get("execution_price_ready", True))
    execution_state = str(signal_row.get("execution_state", "")).strip()
    route_action = str(signal_row.get("route_action", "")).strip()
    route_state = str(signal_row.get("route_state", "")).strip()
    missing_gates = [
        str(item).strip()
        for item in signal_row.get("missing_gates", [])
        if str(item).strip()
    ] if isinstance(signal_row.get("missing_gates", []), list) else []
    pattern_hint_brief = str(signal_row.get("pattern_hint_brief", "")).strip()
    risk_per_unit_pct = calc_risk_per_unit_pct(entry_price=entry_price, stop_price=stop_price)
    age_days = max(0, int((as_of - signal_date).days))
    reasons: list[str] = []
    if age_days > int(max_age_days):
        reasons.append("stale_signal")
    if side not in {"LONG", "BUY", "B", "SHORT", "SELL", "S"}:
        reasons.append("invalid_side")
    if confidence < float(min_confidence):
        reasons.append("confidence_below_threshold")
    if convexity < float(min_convexity):
        reasons.append("convexity_below_threshold")
    if not execution_price_ready:
        reasons.append("proxy_price_reference_only")
    if execution_state and execution_state != "Setup_Ready":
        reasons.append("route_not_setup_ready")
    if market_scope_mismatch:
        reasons.append("signal_market_scope_mismatch")
    if target_market_read_only:
        reasons.append("target_market_read_only")

    execution_mode = "SPOT_LONG_OR_SELL"
    order_side = "LONG"
    max_slippage_bps = 6.0
    if side in {"SHORT", "SELL", "S"}:
        execution_mode = "HEDGE_ONLY"
        order_side = "SHORT"
        max_slippage_bps = 12.0
    if symbol == "XAUUSD":
        execution_mode = "HEDGE_ONLY"
        max_slippage_bps = 0.0
        if "unsupported_symbol" not in reasons:
            reasons.append("unsupported_symbol")

    conviction = calc_conviction(confidence=confidence, convexity=convexity)
    risk_budget_usdt = float(max(0.0, equity_usdt) * max(0.0, base_risk_pct) / 100.0 * max(0.0, risk_multiplier) * conviction)
    alloc_cap = float(max(0.0, equity_usdt) * max(0.0, max_alloc_per_trade_pct))
    if risk_per_unit_pct > 0.0:
        quote_usdt = min(alloc_cap, risk_budget_usdt / (risk_per_unit_pct / 100.0))
    else:
        quote_usdt = 0.0
    if quote_usdt + 1e-12 < float(min_notional_usdt):
        reasons.append("size_below_min_notional")

    allowed = len(reasons) == 0
    return {
        "symbol": symbol,
        "date": signal_date.isoformat(),
        "age_days": int(age_days),
        "allowed": bool(allowed),
        "reasons": reasons,
        "signal": {
            "side": order_side,
            "regime": str(signal_row.get("regime", "")),
            "confidence": float(confidence),
            "convexity_ratio": float(convexity),
            "combo_id": str(signal_row.get("combo_id", "")),
            "combo_id_canonical": str(signal_row.get("combo_id_canonical", "")),
            "confirmation_indicator": str(signal_row.get("confirmation_indicator", "")),
            "factor_flags": list(signal_row.get("factor_flags", []) if isinstance(signal_row.get("factor_flags", []), list) else []),
            "notes": str(signal_row.get("notes", "")),
            "price_reference_kind": price_reference_kind,
            "price_reference_source": price_reference_source,
            "execution_price_ready": bool(execution_price_ready),
            "execution_state": execution_state,
            "route_action": route_action,
            "route_state": route_state,
            "missing_gates": list(missing_gates),
            "pattern_hint_brief": pattern_hint_brief,
        },
        "levels": {
            "entry_price": float(entry_price),
            "stop_price": float(stop_price),
            "target_price": float(target_price),
            "risk_per_unit_pct": float(risk_per_unit_pct),
        },
        "sizing": {
            "equity_usdt": float(equity_usdt),
            "equity_source": equity_source,
            "risk_multiplier": float(risk_multiplier),
            "risk_multiplier_source": risk_multiplier_source,
            "conviction": float(conviction),
            "risk_budget_usdt": float(risk_budget_usdt),
            "quote_usdt": float(quote_usdt),
            "min_notional_usdt": float(min_notional_usdt),
            "max_alloc_pct": float(max_alloc_per_trade_pct),
        },
        "execution": {
            "mode": execution_mode,
            "order_type_hint": "LIMIT",
            "max_slippage_bps": float(max_slippage_bps),
        },
        "market_scope": market_scope,
    }


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), int(path.stat().st_size)


def evict_old_ticket_artifacts(*, directory: Path, now_dt: datetime, ttl_hours: float, keep_names: set[str]) -> int:
    cutoff = now_dt.timestamp() - (float(ttl_hours) * 3600.0)
    removed = 0
    for pattern in ("*_signal_to_order_tickets.json", "*_signal_to_order_tickets.md", "*_signal_to_order_tickets_checksum.json"):
        for path in directory.glob(pattern):
            if path.name in keep_names:
                continue
            try:
                mtime = path.stat().st_mtime
            except Exception:
                continue
            if mtime >= cutoff:
                continue
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue
    return removed


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Signal To Order Tickets",
        "",
        f"- generated_at_utc: `{payload.get('generated_at_utc', '')}`",
        f"- as_of: `{payload.get('as_of', '')}`",
        f"- signal_source_path: `{payload.get('signal_source', {}).get('path', '')}`",
        f"- signal_source_kind: `{payload.get('signal_source', {}).get('kind', '')}`",
        f"- takeover_market: `{payload.get('signal_source', {}).get('takeover_market', '')}`",
        f"- signal_remote_market: `{payload.get('signal_source', {}).get('remote_market', '')}`",
        "",
        "## Summary",
        f"- ticket_count: `{payload.get('summary', {}).get('ticket_count', 0)}`",
        f"- allowed_count: `{payload.get('summary', {}).get('allowed_count', 0)}`",
        f"- stale_count: `{payload.get('summary', {}).get('stale_count', 0)}`",
        f"- missing_count: `{payload.get('summary', {}).get('missing_count', 0)}`",
        f"- proxy_price_only_count: `{payload.get('summary', {}).get('proxy_price_only_count', 0)}`",
        f"- market_scope_mismatch_count: `{payload.get('summary', {}).get('market_scope_mismatch_count', 0)}`",
        f"- target_market_read_only_count: `{payload.get('summary', {}).get('target_market_read_only_count', 0)}`",
        "",
        "## Tickets",
        "| symbol | date | side | conf | convexity | allowed | quote | reason |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("tickets", []):
        if not isinstance(row, dict):
            continue
        signal = row.get("signal", {}) if isinstance(row.get("signal", {}), dict) else {}
        sizing = row.get("sizing", {}) if isinstance(row.get("sizing", {}), dict) else {}
        reasons = ",".join(str(x) for x in row.get("reasons", []) if str(x))
        lines.append(
            "| {symbol} | {date} | {side} | {conf:.2f} | {conv:.2f} | {allowed} | {quote:.4f} | {reason} |".format(
                symbol=str(row.get("symbol", "")),
                date=str(row.get("date", "")),
                side=str(signal.get("side", "")),
                conf=to_float(signal.get("confidence", 0.0), 0.0),
                conv=to_float(signal.get("convexity_ratio", 0.0), 0.0),
                allowed="yes" if bool(row.get("allowed", False)) else "no",
                quote=to_float(sizing.get("quote_usdt", 0.0), 0.0),
                reason=reasons or "-",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    system_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build signal-to-order ticket artifact from latest signal snapshot.")
    parser.add_argument("--date", default="", help="Optional as-of date YYYY-MM-DD")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--review-dir", default="output/review")
    parser.add_argument("--output-dir", default="output/review")
    parser.add_argument("--signals-json", default="")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--equity-usdt", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-convexity", type=float, default=None)
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    parser.add_argument("--base-risk-pct", type=float, default=DEFAULT_BASE_RISK_PCT)
    parser.add_argument("--max-alloc-per-trade-pct", type=float, default=DEFAULT_MAX_ALLOC_PER_TRADE_PCT)
    parser.add_argument("--min-notional-usdt", type=float, default=DEFAULT_MIN_NOTIONAL_USDT)
    parser.add_argument("--artifact-ttl-hours", type=int, default=DEFAULT_ARTIFACT_TTL_HOURS)
    parser.add_argument("--now", default="", help="Optional UTC timestamp override for artifact generation")
    args = parser.parse_args()

    reference_now = parse_now(args.now) or now_utc()
    as_of = date.fromisoformat(str(args.date)) if str(args.date).strip() else reference_now.date()
    config_path = resolve_path(args.config, anchor=system_root)
    output_root = resolve_path(args.output_root, anchor=system_root)
    review_dir = resolve_path(args.review_dir, anchor=system_root)
    out_dir = resolve_path(args.output_dir, anchor=system_root)
    policy_thresholds = derive_policy_thresholds(output_root, config_path=config_path, as_of=as_of)
    takeover_market = load_takeover_market(config_path)
    min_confidence = float(args.min_confidence) if args.min_confidence is not None else float(policy_thresholds["signal_confidence_min"])
    min_convexity = float(args.min_convexity) if args.min_convexity is not None else float(policy_thresholds["convexity_min"])
    symbols = normalize_symbols(args.symbols)
    auto_narrow_symbols = should_auto_narrow_symbols(args.symbols)
    equity_usdt, equity_source = derive_equity_usdt(
        output_root=output_root,
        override_equity_usdt=(float(args.equity_usdt) if float(args.equity_usdt) > 0.0 else None),
        market=takeover_market,
    )
    risk_multiplier, risk_multiplier_source = derive_risk_multiplier(output_root, as_of=as_of)

    explicit_signal_path = resolve_path(args.signals_json, anchor=system_root) if str(args.signals_json).strip() else None
    if explicit_signal_path is not None:
        signal_rows, signal_source_meta = load_signal_rows(explicit_signal_path, as_of=as_of, symbols=symbols)
        signal_source_meta["kind"] = "explicit"
        signal_source_meta["candidate_count"] = 1
        signal_source_meta["selected_candidate_index"] = 0
        signal_source_meta["fallback_used"] = False
        signal_source_meta["selection_reason"] = "explicit"
        signal_source_meta["symbol_scope_mode"] = "explicit_signals_json"
    else:
        signal_rows, signal_source_meta = resolve_signal_source(
            output_root=output_root,
            review_dir=review_dir,
            as_of=as_of,
            symbols=symbols,
        )
        if (
            auto_narrow_symbols
            and str(signal_source_meta.get("kind", "")) == "crypto_shortline_signal_source"
            and signal_rows
        ):
            narrowed_symbols = tuple(sym for sym in symbols if sym in signal_rows)
            if narrowed_symbols:
                symbols = narrowed_symbols
                signal_source_meta["symbol_scope_mode"] = "source_narrowed_default_symbols"
        elif auto_narrow_symbols:
            signal_source_meta["symbol_scope_mode"] = "default_symbols"
        else:
            signal_source_meta["symbol_scope_mode"] = "explicit_symbols"

    signal_market_context = {
        "remote_market": "",
        "artifact": "",
        "source_kind": "missing",
    }
    if str(signal_source_meta.get("kind", "")) == "crypto_shortline_signal_source":
        signal_market_context = load_signal_remote_market_context(
            review_dir=review_dir,
            now_dt=reference_now,
            ttl_hours=max(1, int(args.artifact_ttl_hours)),
        )
    signal_source_meta["takeover_market"] = takeover_market
    signal_source_meta["remote_market"] = str(signal_market_context.get("remote_market", ""))
    signal_source_meta["remote_market_artifact"] = str(signal_market_context.get("artifact", ""))
    signal_source_meta["remote_market_source_kind"] = str(signal_market_context.get("source_kind", "missing"))
    signal_source_meta["target_market_read_only"] = takeover_market == "portfolio_margin_um"
    signal_source_meta["market_scope_mismatch"] = bool(
        signal_source_meta["remote_market"] and signal_source_meta["remote_market"] != takeover_market
    )

    tickets = [
        build_ticket_row(
            symbol=sym,
            signal_row=signal_rows.get(sym),
            as_of=as_of,
            min_confidence=min_confidence,
            min_convexity=min_convexity,
            max_age_days=max(1, int(args.max_age_days)),
            equity_usdt=equity_usdt,
            equity_source=equity_source,
            risk_multiplier=risk_multiplier,
            risk_multiplier_source=risk_multiplier_source,
            base_risk_pct=float(args.base_risk_pct),
            max_alloc_per_trade_pct=float(args.max_alloc_per_trade_pct),
            min_notional_usdt=float(args.min_notional_usdt),
            takeover_market=takeover_market,
            signal_remote_market=str(signal_market_context.get("remote_market", "")),
        )
        for sym in symbols
    ]

    payload = {
        "generated_at_utc": reference_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of": as_of.isoformat(),
        "workspace": str(system_root.parent),
        "signal_db": str(output_root / "artifacts" / "lie_engine.db"),
        "signal_source": signal_source_meta,
        "symbols": list(symbols),
        "thresholds": {
            "min_confidence": float(min_confidence),
            "min_convexity": float(min_convexity),
            "max_age_days": int(max(1, int(args.max_age_days))),
            "base_risk_pct": float(args.base_risk_pct),
            "max_alloc_per_trade_pct": float(args.max_alloc_per_trade_pct),
            "min_notional_usdt": float(args.min_notional_usdt),
            "threshold_source": str(policy_thresholds.get("source", "builtin_default")),
            "threshold_source_kind": str(policy_thresholds.get("source_kind", "config_thresholds")),
            "runtime_mode": str(policy_thresholds.get("runtime_mode", "")),
        },
        "sizing_context": {
            "equity_usdt": float(equity_usdt),
            "equity_source": equity_source,
            "risk_multiplier": float(risk_multiplier),
            "risk_multiplier_source": risk_multiplier_source,
        },
        "tickets": tickets,
        "summary": {
            "ticket_count": int(len(tickets)),
            "allowed_count": int(sum(1 for row in tickets if bool(row.get("allowed", False)))),
            "stale_count": int(sum(1 for row in tickets if "stale_signal" in list(row.get("reasons", [])))),
            "missing_count": int(sum(1 for row in tickets if "signal_not_found" in list(row.get("reasons", [])))),
            "proxy_price_only_count": int(sum(1 for row in tickets if "proxy_price_reference_only" in list(row.get("reasons", [])))),
            "market_scope_mismatch_count": int(
                sum(1 for row in tickets if "signal_market_scope_mismatch" in list(row.get("reasons", [])))
            ),
            "target_market_read_only_count": int(
                sum(1 for row in tickets if "target_market_read_only" in list(row.get("reasons", [])))
            ),
        },
    }

    stamp = reference_now.strftime("%Y%m%dT%H%M%SZ")
    out_json = out_dir / f"{stamp}_signal_to_order_tickets.json"
    out_md = out_dir / f"{stamp}_signal_to_order_tickets.md"
    out_checksum = out_dir / f"{stamp}_signal_to_order_tickets_checksum.json"
    write_json(out_json, payload)
    write_text(out_md, render_markdown(payload))
    evicted = evict_old_ticket_artifacts(
        directory=out_dir,
        now_dt=reference_now,
        ttl_hours=max(1, int(args.artifact_ttl_hours)),
        keep_names={out_json.name, out_md.name, out_checksum.name},
    )
    payload["governance"] = {
        "artifact_ttl_hours": int(max(1, int(args.artifact_ttl_hours))),
        "evicted_files": int(evicted),
    }
    write_json(out_json, payload)
    json_sha, json_size = sha256_file(out_json)
    md_sha, md_size = sha256_file(out_md)
    write_json(
        out_checksum,
        {
            "generated_at_utc": reference_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": [
                {"path": str(out_json), "sha256": json_sha, "size_bytes": int(json_size)},
                {"path": str(out_md), "sha256": md_sha, "size_bytes": int(md_size)},
            ],
        },
    )
    print(
        json.dumps(
            {
                "json": str(out_json),
                "md": str(out_md),
                "checksum": str(out_checksum),
                "signal_source": signal_source_meta,
                "summary": payload.get("summary", {}),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
