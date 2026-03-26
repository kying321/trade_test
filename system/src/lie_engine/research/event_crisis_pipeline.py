from __future__ import annotations

from typing import Dict, Iterable, List

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


def _project_risk(base: float, severity: float, contagion: float, horizon: float) -> float:
    return _clamp01(base * severity * (0.7 + contagion * 0.3) * horizon)


def build_event_regime_snapshot(
    *, event_rows: List[Dict[str, object]], market_inputs: Dict[str, float]
) -> Dict[str, object]:
    credit = market_inputs.get("credit_liquidity_stress_score", 0.4)
    energy = market_inputs.get("energy_geopolitical_stress_score", 0.3)
    cross_asset = market_inputs.get("cross_asset_deleveraging_score", 0.25)
    breadth = _clamp01(market_inputs.get("breadth_score", 0.3))
    contagion = _clamp01(market_inputs.get("contagion_score", 0.3))
    persistence = _clamp01(market_inputs.get("persistence_score", 0.3))
    policy_offset = _clamp01(market_inputs.get("policy_offset_score", 0.2))
    confidence = _clamp01(market_inputs.get("confidence_score", 0.5))

    severity = _clamp01(credit * 0.45 + energy * 0.25 + cross_asset * 0.3)
    systemic = _clamp01(
        credit * 0.35 + breadth * 0.2 + contagion * 0.25 + persistence * 0.2
    )
    axes = _collect_event_axes(event_rows)
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
    }


def build_event_crisis_analogy(
    *, event_rows: List[Dict[str, object]], market_inputs: Dict[str, float]
) -> Dict[str, object]:
    axes = _collect_event_axes(event_rows)
    return {"top_analogues": build_top_analogues(axes)}


def build_event_asset_shock_map(
    *, event_rows: List[Dict[str, object]], market_inputs: Dict[str, float]
) -> Dict[str, List[Dict[str, object]]]:
    severity = _clamp01(market_inputs.get("event_severity_score", 0.5))
    contagion = _clamp01(market_inputs.get("contagion_score", 0.3))

    assets = []
    for config in PRIORITY_ASSET_CONFIG:
        assets.append(
            {
                "asset": config["asset"],
                "class": config["class"],
                "shock_direction_bias": config["shock_direction_bias"],
                "expected_volatility_rank": config["expected_volatility_rank"],
                "contagion_sensitivity": config["contagion_sensitivity"],
                "risk_1d": _project_risk(config["base_risk"], severity, contagion, 1.0),
                "risk_3d": _project_risk(config["base_risk"], severity, contagion, 1.2),
                "risk_7d": _project_risk(config["base_risk"], severity, contagion, 1.4),
            }
        )
    return {"assets": assets, "source_event_count": len(event_rows)}
