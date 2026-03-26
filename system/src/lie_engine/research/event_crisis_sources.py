from __future__ import annotations

import datetime as dt
import math
from typing import Iterable, List, Mapping, Sequence, Union

DEFAULT_MARKET_INPUTS: Mapping[str, float] = {
    "credit_liquidity_stress_score": 0.45,
    "energy_geopolitical_stress_score": 0.3,
    "cross_asset_deleveraging_score": 0.25,
    "breadth_score": 0.3,
    "contagion_score": 0.3,
    "persistence_score": 0.3,
    "policy_offset_score": 0.2,
    "confidence_score": 0.5,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ensure_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _parse_timestamp(source: str | None, default: dt.datetime) -> dt.datetime:
    if not source:
        return default
    text = source.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    parsed = dt.datetime.fromisoformat(text)
    return _ensure_utc(parsed)


PRIORITY_ASSET_CONFIG: List[Mapping[str, object]] = [
    {
        "asset": "BTC",
        "class": "crypto",
        "shock_direction_bias": "negative",
        "expected_volatility_rank": 9,
        "contagion_sensitivity": 0.85,
        "base_risk": 0.65,
    },
    {
        "asset": "ETH",
        "class": "crypto",
        "shock_direction_bias": "negative",
        "expected_volatility_rank": 8,
        "contagion_sensitivity": 0.8,
        "base_risk": 0.6,
    },
    {
        "asset": "SOL",
        "class": "crypto",
        "shock_direction_bias": "negative",
        "expected_volatility_rank": 9,
        "contagion_sensitivity": 0.9,
        "base_risk": 0.7,
    },
    {
        "asset": "BNB",
        "class": "crypto",
        "shock_direction_bias": "negative",
        "expected_volatility_rank": 7,
        "contagion_sensitivity": 0.75,
        "base_risk": 0.55,
    },
    {
        "asset": "GOLD",
        "class": "safe_haven",
        "shock_direction_bias": "positive",
        "expected_volatility_rank": 5,
        "contagion_sensitivity": 0.4,
        "base_risk": 0.35,
    },
    {
        "asset": "UST_LONG",
        "class": "credit",
        "shock_direction_bias": "positive",
        "expected_volatility_rank": 4,
        "contagion_sensitivity": 0.3,
        "base_risk": 0.4,
    },
    {
        "asset": "OIL",
        "class": "energy",
        "shock_direction_bias": "mixed",
        "expected_volatility_rank": 6,
        "contagion_sensitivity": 0.5,
        "base_risk": 0.5,
    },
    {
        "asset": "BANKS",
        "class": "financial",
        "shock_direction_bias": "negative",
        "expected_volatility_rank": 7,
        "contagion_sensitivity": 0.8,
        "base_risk": 0.6,
    },
    {
        "asset": "HIGH_YIELD",
        "class": "credit",
        "shock_direction_bias": "negative",
        "expected_volatility_rank": 8,
        "contagion_sensitivity": 0.78,
        "base_risk": 0.65,
    },
]

DEFAULT_PRIORITY_ASSETS: List[str] = [cast_mapping["asset"] for cast_mapping in PRIORITY_ASSET_CONFIG]


def normalize_public_event_rows(
    rows: Sequence[Mapping[str, object]], *, default_ts: dt.datetime
) -> List[Mapping[str, object]]:
    normalized: list[Mapping[str, object]] = []
    default_ts = _ensure_utc(default_ts)
    for row in rows:
        event_ts_raw = row.get("event_ts")
        event_ts = _parse_timestamp(str(event_ts_raw) if event_ts_raw is not None else None, default_ts)
        normalized.append(
            {
                "event_id": row.get("event_id") or f"event-{len(normalized)}",
                "event_ts_utc": event_ts.isoformat().replace("+00:00", "Z"),
                "source": row.get("source", "public"),
                "source_type": row.get("source_type", "public"),
                "headline": row.get("headline"),
                "event_classes": _ensure_str_list(row.get("event_classes")),
                "regions": sorted(set(_ensure_str_list(row.get("regions")))),
                "affected_assets": sorted(set(_ensure_str_list(row.get("affected_assets")))),
                "credibility_score": row.get("credibility_score", 0.5),
                "novelty_score": row.get("novelty_score", 0.5),
            }
        )
    return normalized


def normalize_market_inputs(
    inputs: Mapping[str, object] | None = None
) -> Mapping[str, float]:
    inputs = inputs or {}
    normalized: dict[str, float] = {}
    for key, default in DEFAULT_MARKET_INPUTS.items():
        raw_value = inputs.get(key, default)
        normalized[key] = _clamp01(_safe_float(raw_value, default))
    return normalized


def _ensure_str_list(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        trimmed = value.strip()
        return [trimmed] if trimmed else []
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result
    raise TypeError(f"event field expects str or iterable of str, got {type(value).__name__}")


def _safe_float(value: object | None, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default
