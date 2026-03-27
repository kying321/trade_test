from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Mapping

from lie_engine.research.event_crisis_analogies import build_top_analogues


REGIME_STATES = ["watch", "sector_stress", "cross_asset_contagion", "systemic_risk"]

PRIORITY_ASSET_CONFIG = [
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


STATE_OVERLAY_TTL = dt.timedelta(hours=1)

STATE_OVERLAY = {
    "watch": {
        "risk_multiplier_override": 1.0,
        "gate_tightening_state": "normal",
        "canary_freeze": False,
        "review_required": False,
    },
    "sector_stress": {
        "risk_multiplier_override": 0.9,
        "gate_tightening_state": "moderate",
        "canary_freeze": False,
        "review_required": True,
    },
    "cross_asset_contagion": {
        "risk_multiplier_override": 0.75,
        "gate_tightening_state": "tight",
        "canary_freeze": True,
        "review_required": True,
    },
    "systemic_risk": {
        "risk_multiplier_override": 0.6,
        "gate_tightening_state": "tight",
        "canary_freeze": True,
        "review_required": True,
    },
}


def _serialize_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_event_live_guard_overlay(
    *, regime_snapshot: Dict[str, object], generated_at: dt.datetime | None = None
) -> Dict[str, object]:
    regime_state = regime_snapshot.get("regime_state")
    if regime_state not in STATE_OVERLAY:
        regime_state = "watch"
    override = STATE_OVERLAY[regime_state]
    now = generated_at or dt.datetime.now(dt.timezone.utc)
    overlay: Dict[str, object] = {
        "risk_multiplier_override": override["risk_multiplier_override"],
        "gate_tightening_state": override["gate_tightening_state"],
        "canary_freeze": override["canary_freeze"],
        "review_required": override["review_required"],
        "override_reason_codes": [f"event_state:{regime_state}"],
        "valid_until_utc": _serialize_utc(now + STATE_OVERLAY_TTL),
    }
    return overlay


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _determine_regime_state(
    severity: float,
    systemic: float,
    cross_asset: float,
    contagion: float,
    breadth: float,
) -> str:
    if systemic >= 0.65 or severity >= 0.75:
        return "systemic_risk"
    if contagion >= 0.55 or cross_asset >= 0.5:
        return "cross_asset_contagion"
    if severity >= 0.4 or breadth >= 0.35:
        return "sector_stress"
    return "watch"


def _collect_event_axes(event_rows: Iterable[Dict[str, object]]) -> List[str]:
    axes = []
    for row in event_rows:
        axes.extend(axis for axis in row.get("event_classes", []))
    return sorted(set(axes))


def _collect_headlines(event_rows: Iterable[Dict[str, object]]) -> List[str]:
    headlines = []
    for row in event_rows:
        headline = row.get("headline")
        if headline:
            headlines.append(headline)
    return headlines


GAME_STATE_SEVERITY_BIAS = {
    "stable_competition": 0.0,
    "financial_pressure": 0.06,
    "commodity_weaponization": 0.08,
    "bloc_fragmentation": 0.1,
    "systemic_repricing": 0.12,
}

DOMINANT_CHAIN_INFLUENCE = {
    "credit_intermediary_chain": {"systemic": 0.06, "cross_asset": 0.05, "contagion": 0.03, "breadth": 0.02},
    "usd_liquidity_chain": {"systemic": 0.05, "cross_asset": 0.04, "contagion": 0.02, "breadth": 0.01},
    "risk_off_deleveraging_chain": {"systemic": 0.04, "cross_asset": 0.06, "contagion": 0.04, "breadth": 0.0},
    "financial_sanctions_chain": {"systemic": 0.04, "cross_asset": 0.03, "contagion": 0.05, "breadth": 0.01},
    "energy_supply_chain": {"systemic": 0.03, "cross_asset": 0.02, "contagion": 0.02, "breadth": 0.02},
    "shipping_supply_chain": {"systemic": 0.02, "cross_asset": 0.01, "contagion": 0.015, "breadth": 0.015},
}

DOMINANT_CHAIN_ASSET_BIASES = {
    "usd_liquidity_chain": {"credit": 0.03, "financial": 0.025, "crypto": 0.02},
    "financial_sanctions_chain": {"credit": 0.04, "financial": 0.03},
    "energy_supply_chain": {"energy": 0.04, "safe_haven": 0.015, "credit": 0.01},
    "shipping_supply_chain": {"energy": 0.03, "safe_haven": 0.02},
    "credit_intermediary_chain": {"credit": 0.04, "financial": 0.03},
    "risk_off_deleveraging_chain": {"crypto": 0.05, "credit": 0.025},
}


def _game_state_bias(game_state: str | None) -> float:
    if not game_state:
        return 0.0
    return GAME_STATE_SEVERITY_BIAS.get(game_state, 0.0)


def _dominant_chain_influence(chain: str | None) -> Dict[str, float]:
    if not chain:
        return {"systemic": 0.0, "cross_asset": 0.0, "contagion": 0.0, "breadth": 0.0}
    return DOMINANT_CHAIN_INFLUENCE.get(chain, {"systemic": 0.0, "cross_asset": 0.0, "contagion": 0.0, "breadth": 0.0})


def _dominant_chain_asset_bias(chain: str | None, asset_class: str) -> float:
    if not chain:
        return 0.0
    chain_biases = DOMINANT_CHAIN_ASSET_BIASES.get(chain)
    if not chain_biases:
        return 0.0
    return chain_biases.get(asset_class, 0.0)


def _project_risk(base: float, severity: float, contagion: float, horizon: float) -> float:
    return _clamp01(base * severity * (0.7 + contagion * 0.3) * horizon)


def _raw_severity_score(credit: float, energy: float, cross_asset: float) -> float:
    return _clamp01(credit * 0.45 + energy * 0.25 + cross_asset * 0.3)


def _default_contagion_score(credit: float, energy: float, cross_asset: float) -> float:
    return _clamp01(cross_asset * 0.55 + credit * 0.2 + energy * 0.15 + 0.1)


def build_event_regime_snapshot(
    *,
    event_rows: List[Dict[str, object]],
    market_inputs: Dict[str, float],
    game_state_snapshot: Mapping[str, Any] | None = None,
    transmission_chain_map: Mapping[str, Any] | None = None,
) -> Dict[str, object]:
    credit = market_inputs.get("credit_liquidity_stress_score", 0.4)
    energy = market_inputs.get("energy_geopolitical_stress_score", 0.3)
    cross_asset = market_inputs.get("cross_asset_deleveraging_score", 0.25)
    breadth = _clamp01(market_inputs.get("breadth_score", 0.3))
    contagion = _clamp01(market_inputs.get("contagion_score", 0.3))
    persistence = _clamp01(market_inputs.get("persistence_score", 0.3))
    policy_offset = _clamp01(market_inputs.get("policy_offset_score", 0.2))
    confidence = _clamp01(market_inputs.get("confidence_score", 0.5))

    axes = _collect_event_axes(event_rows)
    axis_bonus = 0.1 if axes else 0.0
    raw_severity = _raw_severity_score(credit, energy, cross_asset)
    game_state = (
        game_state_snapshot.get("game_state") if game_state_snapshot else None
    )
    dominant_chain = (
        transmission_chain_map.get("dominant_chain") if transmission_chain_map else None
    )
    game_state_bias = _game_state_bias(game_state)
    chain_influence = _dominant_chain_influence(dominant_chain)
    severity = _clamp01(raw_severity + axis_bonus + game_state_bias * 0.5)
    systemic_base = credit * 0.35 + breadth * 0.2 + contagion * 0.25 + persistence * 0.2
    systemic = _clamp01(systemic_base + game_state_bias * 0.7 + chain_influence["systemic"])
    breadth = _clamp01(breadth + chain_influence["breadth"])
    contagion = _clamp01(contagion + chain_influence["contagion"])
    cross_asset = _clamp01(cross_asset + chain_influence["cross_asset"])
    headlines = _collect_headlines(event_rows)
    regime_state = _determine_regime_state(severity, systemic, cross_asset, contagion, breadth)

    return {
        "severity_score": severity,
        "breadth_score": breadth,
        "contagion_score": contagion,
        "persistence_score": persistence,
        "policy_offset_score": policy_offset,
        "confidence_score": confidence,
        "event_severity_score": severity,
        "systemic_risk_score": systemic,
        "regime_state": regime_state,
        "primary_axes": axes,
        "headline_drivers": headlines,
        "top_risk_assets": ["BANKS", "HIGH_YIELD", "BTC"],
        "game_state": game_state,
        "dominant_chain": dominant_chain,
    }


def build_event_crisis_analogy(
    *, event_rows: List[Dict[str, object]], market_inputs: Dict[str, float]
) -> Dict[str, object]:
    axes = _collect_event_axes(event_rows)
    return {"top_analogues": build_top_analogues(axes)}


def build_event_asset_shock_map(
    *,
    event_rows: List[Dict[str, object]],
    market_inputs: Dict[str, float],
    transmission_chain_map: Mapping[str, Any] | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    credit = market_inputs.get("credit_liquidity_stress_score", 0.4)
    energy = market_inputs.get("energy_geopolitical_stress_score", 0.3)
    cross_asset = market_inputs.get("cross_asset_deleveraging_score", 0.25)
    computed_severity = _raw_severity_score(credit, energy, cross_asset)
    severity = _clamp01(market_inputs.get("event_severity_score", computed_severity))
    contagion_source = market_inputs.get("contagion_score")
    contagion = (
        _clamp01(contagion_source)
        if contagion_source is not None
        else _default_contagion_score(credit, energy, cross_asset)
    )

    dominant_chain = (
        transmission_chain_map.get("dominant_chain") if transmission_chain_map else None
    )
    assets = []
    for config in PRIORITY_ASSET_CONFIG:
        assets.append(
            {
                "asset": config["asset"],
                "class": config["class"],
                "shock_direction_bias": config["shock_direction_bias"],
                "expected_volatility_rank": config["expected_volatility_rank"],
                "contagion_sensitivity": config["contagion_sensitivity"],
                "risk_1d": _clamp01(
                    _project_risk(config["base_risk"], severity, contagion, 1.0)
                    + _dominant_chain_asset_bias(dominant_chain, config["class"])
                ),
                "risk_3d": _clamp01(
                    _project_risk(config["base_risk"], severity, contagion, 1.2)
                    + _dominant_chain_asset_bias(dominant_chain, config["class"])
                ),
                "risk_7d": _clamp01(
                    _project_risk(config["base_risk"], severity, contagion, 1.4)
                    + _dominant_chain_asset_bias(dominant_chain, config["class"])
                ),
            }
        )
    return {
        "assets": assets,
        "source_event_count": len(event_rows),
        "dominant_chain": dominant_chain,
    }
