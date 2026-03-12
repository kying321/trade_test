from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

import numpy as np
import pandas as pd

from lie_engine.models import RegimeLabel, Side, SignalCandidate
from lie_engine.regime import (
    compute_atr_zscore,
    derive_regime_consensus,
    infer_hmm_state,
    latest_multi_scale_hurst,
)
from lie_engine.signal.features import add_common_features
from lie_engine.signal.range_engine import score_range
from lie_engine.signal.theory import TheoryConfluenceResult, compute_theory_confluence
from lie_engine.signal.trend import score_trend


SHORTABLE_ASSET_CLASS = {"future", "option", "hedge", "crypto", "perp", "perpetual"}


@dataclass(slots=True)
class SignalEngineConfig:
    confidence_min: float
    convexity_min: float
    hurst_trend_thr: float = 0.6
    hurst_mean_thr: float = 0.4
    atr_extreme_thr: float = 2.0
    reward_mult_strong: float = 3.2
    reward_mult_weak: float = 2.0
    reward_mult_range: float = 1.5
    reward_mult_floor: float = 1.2
    reward_mult_ceiling: float = 3.6
    factor_filter_enabled: bool = True
    factor_penalty_max: float = 22.0
    factor_drop_threshold: float = 0.92
    microstructure_enabled: bool = True
    micro_confidence_boost_max: float = 8.0
    micro_penalty_max: float = 10.0
    micro_min_trade_count: int = 30
    theory_enabled: bool = False
    theory_ict_weight: float = 1.0
    theory_brooks_weight: float = 1.0
    theory_lie_weight: float = 1.2
    theory_wyckoff_weight: float = 0.0
    theory_vpa_weight: float = 0.0
    theory_confidence_boost_max: float = 5.0
    theory_penalty_max: float = 6.0
    theory_min_confluence: float = 0.38
    theory_conflict_fuse: float = 0.72


def _bounded(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _convexity_ratio(side: Side, entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    if risk <= 1e-9:
        return 0.0
    if side == Side.LONG:
        reward = max(0.0, target - entry)
    elif side == Side.SHORT:
        reward = max(0.0, entry - target)
    else:
        reward = 0.0
    return reward / risk


def _stop_multiplier(regime: RegimeLabel) -> float:
    if regime in {RegimeLabel.STRONG_TREND, RegimeLabel.DOWNTREND}:
        return 1.5
    elif regime == RegimeLabel.WEAK_TREND:
        return 1.2
    return 1.0


def _base_reward_multiplier(cfg: SignalEngineConfig, regime: RegimeLabel) -> float:
    if regime in {RegimeLabel.STRONG_TREND, RegimeLabel.DOWNTREND}:
        return float(cfg.reward_mult_strong)
    elif regime == RegimeLabel.WEAK_TREND:
        return float(cfg.reward_mult_weak)
    return float(cfg.reward_mult_range)


def _adaptive_reward_multiplier(
    cfg: SignalEngineConfig,
    regime: RegimeLabel,
    trend_score_ratio: float,
    atr_pct: float,
    factor_risk_score: float,
) -> float:
    base = _base_reward_multiplier(cfg, regime)
    adj = 0.0
    ts = _bounded(float(trend_score_ratio), 0.0, 1.0)

    if regime == RegimeLabel.WEAK_TREND:
        if ts < 0.55:
            adj -= 0.35
        elif ts > 0.75:
            adj += 0.15
    elif regime == RegimeLabel.RANGE:
        adj -= 0.15

    if atr_pct > 0.05:
        adj -= 0.30
    elif atr_pct < 0.015 and ts > 0.70:
        adj += 0.20

    adj -= min(0.45, max(0.0, float(factor_risk_score)) * 0.40)
    return _bounded(
        base + adj,
        low=float(cfg.reward_mult_floor),
        high=max(float(cfg.reward_mult_floor), float(cfg.reward_mult_ceiling)),
    )


def _build_trade_levels(side: Side, close: float, atr: float, stop_mult: float, reward_mult: float) -> tuple[float, float, float]:
    risk = max(atr * stop_mult, close * 0.01)

    if side == Side.SHORT:
        stop = close + risk
        target = close - reward_mult * risk
    else:
        stop = close - risk
        target = close + reward_mult * risk
    return close, stop, target


def expand_universe(core_symbols: Iterable[str], bars: pd.DataFrame, max_additions: int) -> list[str]:
    core = list(dict.fromkeys(core_symbols))
    candidates = ["510300", "510500", "159915", "600519", "000001", "601318", "CU2603", "AU2604"]
    if bars.empty:
        return core + candidates[:max_additions]

    latest = bars.sort_values("ts").groupby("symbol", as_index=False).tail(1)
    ranked = latest.sort_values("volume", ascending=False)["symbol"].tolist()
    merged = core + [s for s in ranked if s not in core]
    for c in candidates:
        if c not in merged:
            merged.append(c)
    return merged[: len(core) + max_additions]


def detect_symbol_regime(symbol_df: pd.DataFrame, cfg: SignalEngineConfig) -> RegimeLabel:
    if len(symbol_df) < 30:
        return RegimeLabel.UNCERTAIN

    required = {"ts", "open", "high", "low", "close", "volume"}
    if not required.issubset(set(symbol_df.columns)):
        return RegimeLabel.UNCERTAIN

    try:
        frame = symbol_df.sort_values("ts").tail(180)
        hmm_input = frame[["close", "volume"]].iloc[::3].tail(80).copy()
        if len(hmm_input) >= 25:
            hmm_probs = infer_hmm_state(hmm_input)
        else:
            hmm_probs = {"bull": 0.33, "range": 0.34, "bear": 0.33}

        closes = frame["close"].to_numpy()
        hurst = latest_multi_scale_hurst(closes)

        atr_df = frame[["open", "high", "low", "close"]].tail(120).copy()
        atr_z = compute_atr_zscore(atr_df)
        ts = pd.to_datetime(frame["ts"], errors="coerce")
        valid_ts = ts.dropna()
        if valid_ts.empty:
            return RegimeLabel.UNCERTAIN
        last_ts = valid_ts.max().date()

        state = derive_regime_consensus(
            as_of=last_ts,
            hurst=hurst,
            hmm_probs=hmm_probs,
            atr_z=atr_z,
            trend_thr=cfg.hurst_trend_thr,
            mean_thr=cfg.hurst_mean_thr,
            atr_extreme=cfg.atr_extreme_thr,
        )
        return state.consensus
    except Exception:  # noqa: BLE001
        return RegimeLabel.UNCERTAIN


def _default_market_factor_state(regime: RegimeLabel) -> dict[str, float]:
    if regime in {RegimeLabel.STRONG_TREND, RegimeLabel.DOWNTREND}:
        return {
            "valuation_pressure": 0.25,
            "momentum_preference": 0.75,
            "crowding_aversion": 0.30,
            "small_cap_pressure": 0.25,
            "dividend_preference": 0.35,
            "data_quality_pressure": 0.0,
            "flow_quality_pressure": 0.24,
        }
    if regime == RegimeLabel.WEAK_TREND:
        return {
            "valuation_pressure": 0.45,
            "momentum_preference": 0.55,
            "crowding_aversion": 0.45,
            "small_cap_pressure": 0.35,
            "dividend_preference": 0.50,
            "data_quality_pressure": 0.0,
            "flow_quality_pressure": 0.32,
        }
    if regime == RegimeLabel.RANGE:
        return {
            "valuation_pressure": 0.60,
            "momentum_preference": 0.35,
            "crowding_aversion": 0.55,
            "small_cap_pressure": 0.55,
            "dividend_preference": 0.65,
            "data_quality_pressure": 0.0,
            "flow_quality_pressure": 0.42,
        }
    return {
        "valuation_pressure": 0.65,
        "momentum_preference": 0.40,
        "crowding_aversion": 0.60,
        "small_cap_pressure": 0.65,
        "dividend_preference": 0.70,
        "data_quality_pressure": 0.0,
        "flow_quality_pressure": 0.50,
    }


def _merge_market_factor_state(regime: RegimeLabel, market_factor_state: dict[str, float] | None) -> dict[str, float]:
    merged = _default_market_factor_state(regime)
    if not isinstance(market_factor_state, dict):
        return merged

    for key in (
        "valuation_pressure",
        "momentum_preference",
        "crowding_aversion",
        "small_cap_pressure",
        "dividend_preference",
        "flow_quality_pressure",
    ):
        if key in market_factor_state:
            merged[key] = _bounded(_safe_float(market_factor_state.get(key), merged[key]), 0.0, 1.5)
    if "data_quality_pressure" in market_factor_state:
        merged["data_quality_pressure"] = _bounded(_safe_float(market_factor_state.get("data_quality_pressure"), 0.0), 0.0, 1.5)

    quality_score_7d = _bounded(_safe_float(market_factor_state.get("cross_source_quality_score_7d"), 1.0), 0.0, 1.0)
    fail_ratio_7d = _bounded(_safe_float(market_factor_state.get("cross_source_fail_ratio_7d"), 0.0), 0.0, 1.0)
    insuff_ratio_7d = _bounded(_safe_float(market_factor_state.get("cross_source_insufficient_ratio_7d"), 0.0), 0.0, 1.0)
    cross_stress = _bounded(_safe_float(market_factor_state.get("cross_source_stress"), 0.0), 0.0, 1.5)
    crypto_stress = _bounded(_safe_float(market_factor_state.get("crypto_stress"), 0.0), 0.0, 1.5)
    btc_spread_bps = max(0.0, _safe_float(market_factor_state.get("btc_book_spread_bps"), 0.0))
    btc_funding_abs = max(0.0, _safe_float(market_factor_state.get("btc_funding_abs_8h"), 0.0))
    derived_quality_pressure = _bounded(
        (1.0 - quality_score_7d) * 0.55 + fail_ratio_7d * 0.75 + (cross_stress / 1.5) * 0.35,
        0.0,
        1.5,
    )
    merged["data_quality_pressure"] = _bounded(
        max(float(merged.get("data_quality_pressure", 0.0)), derived_quality_pressure),
        0.0,
        1.5,
    )
    derived_flow_pressure = _bounded(
        (cross_stress / 1.5) * 0.30
        + fail_ratio_7d * 0.22
        + insuff_ratio_7d * 0.16
        + (crypto_stress / 1.5) * 0.18
        + _bounded((btc_spread_bps - 2.0) / 20.0, 0.0, 1.0) * 0.08
        + _bounded((btc_funding_abs - 0.0002) * 2000.0, 0.0, 1.0) * 0.06,
        0.0,
        1.5,
    )
    merged["flow_quality_pressure"] = _bounded(
        max(float(merged.get("flow_quality_pressure", 0.0)), derived_flow_pressure),
        0.0,
        1.5,
    )
    return merged


def _estimate_factor_exposure(df: pd.DataFrame) -> dict[str, float]:
    cur = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else cur
    close = max(1e-9, _safe_float(cur.get("close"), 0.0))
    ma60 = max(1e-9, _safe_float(cur.get("ma60"), close))
    atr14 = _safe_float(cur.get("atr14"), 0.0)
    vol_ma20 = max(1.0, _safe_float(cur.get("vol_ma20"), _safe_float(cur.get("volume"), 1.0)))
    volume = _safe_float(cur.get("volume"), vol_ma20)
    close_series = pd.to_numeric(df["close"], errors="coerce").dropna()

    ret60 = 0.0
    if len(close_series) >= 61:
        past = float(close_series.iloc[-61])
        if abs(past) > 1e-9:
            ret60 = close / past - 1.0
    elif len(close_series) >= 21:
        past = float(close_series.iloc[-21])
        if abs(past) > 1e-9:
            ret60 = close / past - 1.0

    open_px = _safe_float(cur.get("open"), close)
    prev_close = _safe_float(prev.get("close"), close)
    high_px = max(open_px, close, _safe_float(cur.get("high"), close))
    low_px = min(open_px, close, _safe_float(cur.get("low"), close))
    bar_range = max(1e-9, high_px - low_px)
    body = abs(close - open_px)
    upper_wick = max(0.0, high_px - max(close, open_px))
    lower_wick = max(0.0, min(close, open_px) - low_px)
    vol_ratio = max(0.0, volume / max(1e-9, vol_ma20))
    effort_strength = _bounded((vol_ratio - 0.85) / 1.80, 0.0, 2.5)
    result_strength = _bounded((body / max(atr14, 1e-9) - 0.15) / 0.85, 0.0, 2.5)
    flow_effort_gap = _bounded(effort_strength - result_strength, 0.0, 2.5)
    flow_climax = _bounded(
        _bounded((vol_ratio - 1.25) / 1.50, 0.0, 2.5)
        * _bounded((bar_range / max(atr14, 1e-9) - 1.0) / 1.8, 0.0, 2.5),
        0.0,
        2.5,
    )
    flow_absorption_skew = _bounded(
        ((upper_wick - lower_wick) / bar_range) * (0.5 + 0.5 * _bounded((vol_ratio - 0.9) / 1.6, 0.0, 1.0)),
        -2.5,
        2.5,
    )
    # Weak directional result with heavy effort often signals poor signal quality.
    ret1 = (close - prev_close) / max(abs(prev_close), 1e-9)
    flow_effort_gap = _bounded(
        max(flow_effort_gap, max(0.0, effort_strength - _bounded(abs(ret1) / max(atr14 / close, 1e-9), 0.0, 2.5))),
        0.0,
        2.5,
    )

    momentum = _bounded(ret60 / 0.18, -2.5, 2.5)
    crowded = _bounded((volume / vol_ma20 - 1.0) / 1.5, -2.0, 2.5)
    volatility = _bounded((atr14 / close - 0.02) / 0.03, -1.5, 2.5)

    valuation = 0.0
    pe_ttm = _safe_float(cur.get("pe_ttm"), np.nan)
    pb = _safe_float(cur.get("pb"), np.nan)
    if np.isfinite(pe_ttm) and pe_ttm > 0:
        valuation = _bounded((pe_ttm - 25.0) / 25.0, -2.0, 3.0)
    elif np.isfinite(pb) and pb > 0:
        valuation = _bounded((pb - 3.0) / 2.0, -2.0, 3.0)
    else:
        valuation = _bounded((close / ma60 - 1.0) / 0.08, -2.0, 3.0)

    size = 0.0
    market_cap = _safe_float(cur.get("market_cap"), 0.0)
    if market_cap > 0:
        size = _bounded((10.8 - np.log10(max(1.0, market_cap))) / 1.2, -1.5, 2.5)
    else:
        turnover = max(1.0, vol_ma20 * close)
        size = _bounded((9.8 - np.log10(turnover)) / 1.4, -1.5, 2.5)

    dividend = 0.0
    dividend_yield = _safe_float(cur.get("dividend_yield"), np.nan)
    if np.isfinite(dividend_yield) and dividend_yield >= 0:
        dividend = _bounded((0.02 - float(dividend_yield)) / 0.02, -2.0, 3.0)
    else:
        dividend = _bounded(volatility - 0.20, -1.5, 2.5)

    return {
        "valuation": float(valuation),
        "momentum": float(momentum),
        "crowded": float(crowded),
        "size": float(size),
        "volatility": float(volatility),
        "dividend": float(dividend),
        "flow_effort_gap": float(flow_effort_gap),
        "flow_climax": float(flow_climax),
        "flow_absorption_skew": float(flow_absorption_skew),
    }


def _factor_risk_score(
    side: Side,
    exposure: dict[str, float],
    market_factor_state: dict[str, float],
) -> tuple[float, list[str]]:
    valuation = max(0.0, exposure.get("valuation", 0.0))
    momentum = exposure.get("momentum", 0.0)
    crowded = max(0.0, exposure.get("crowded", 0.0))
    size = max(0.0, exposure.get("size", 0.0))
    volatility = max(0.0, exposure.get("volatility", 0.0))
    dividend = max(0.0, exposure.get("dividend", 0.0))
    data_quality_pressure = max(0.0, _safe_float(market_factor_state.get("data_quality_pressure"), 0.0))
    flow_quality_pressure = max(0.0, _safe_float(market_factor_state.get("flow_quality_pressure"), 0.0))
    flow_effort_gap = max(0.0, exposure.get("flow_effort_gap", 0.0))
    flow_climax = max(0.0, exposure.get("flow_climax", 0.0))
    flow_absorption_skew = _safe_float(exposure.get("flow_absorption_skew"), 0.0)

    if side == Side.SHORT:
        valuation *= 0.35
        momentum_mismatch = max(0.0, momentum)
        flow_reversal = max(0.0, -flow_absorption_skew)
    else:
        momentum_mismatch = max(0.0, -momentum)
        flow_reversal = max(0.0, flow_absorption_skew)
    dividend_mismatch = dividend * max(0.0, market_factor_state.get("dividend_preference", 0.0))
    flow_risk = 0.55 * flow_effort_gap + 0.45 * flow_climax + 0.50 * flow_reversal

    raw = (
        valuation * market_factor_state["valuation_pressure"]
        + momentum_mismatch * market_factor_state["momentum_preference"] * 0.9
        + crowded * market_factor_state["crowding_aversion"] * 0.7
        + size * market_factor_state["small_cap_pressure"] * 0.6
        + volatility * 0.45
        + dividend_mismatch * 0.55
        + data_quality_pressure * 0.65
        + flow_risk * flow_quality_pressure * 0.55
    )
    score = _bounded(raw / 3.0, 0.0, 1.5)

    flags: list[str] = []
    if valuation * market_factor_state["valuation_pressure"] > 0.45:
        flags.append("valuation_headwind")
    if momentum_mismatch * market_factor_state["momentum_preference"] > 0.35:
        flags.append("momentum_mismatch")
    if crowded * market_factor_state["crowding_aversion"] > 0.30:
        flags.append("crowded_trade")
    if size * market_factor_state["small_cap_pressure"] > 0.25:
        flags.append("small_cap_risk")
    if dividend_mismatch > 0.25:
        flags.append("low_dividend_headwind")
    if volatility > 1.0:
        flags.append("volatility_spike")
    if side == Side.SHORT and exposure.get("valuation", 0.0) > 0.8:
        flags.append("short_valuation_tailwind")
    if side == Side.LONG and exposure.get("valuation", 0.0) < -0.6:
        flags.append("value_tailwind")
    if data_quality_pressure > 0.35:
        flags.append("data_quality_headwind")
    if flow_effort_gap * flow_quality_pressure > 0.20:
        flags.append("flow_effort_result_divergence")
    if flow_climax * flow_quality_pressure > 0.18:
        flags.append("flow_climax_risk")
    if flow_reversal * flow_quality_pressure > 0.16:
        flags.append("flow_absorption_reversal")
    if flow_quality_pressure > 0.45:
        flags.append("flow_quality_headwind")
    return score, flags


def _microstructure_adjustment(
    *,
    side: Side,
    cfg: SignalEngineConfig,
    micro_factor_state: dict[str, float] | None,
    market_factor_state: dict[str, float] | None = None,
) -> tuple[float, float, list[str], float]:
    if not bool(cfg.microstructure_enabled):
        return 0.0, 0.0, [], 0.0
    if not isinstance(micro_factor_state, dict):
        return 0.0, 0.0, [], 0.0
    has_data = bool(micro_factor_state.get("has_data", False))

    micro_alignment = _bounded(_safe_float(micro_factor_state.get("micro_alignment"), 0.0), -1.0, 1.0)
    if side == Side.SHORT:
        micro_alignment = -micro_alignment
    evidence_score = _bounded(_safe_float(micro_factor_state.get("evidence_score"), 0.0), 0.0, 1.0)
    signed_signal = _bounded(micro_alignment * evidence_score, -1.0, 1.0)

    boost = max(0.0, signed_signal) * float(cfg.micro_confidence_boost_max)
    penalty = max(0.0, -signed_signal) * float(cfg.micro_penalty_max)
    flags: list[str] = []

    sync_ok = bool(micro_factor_state.get("sync_ok", True))
    gap_ok = bool(micro_factor_state.get("gap_ok", True))
    schema_ok = bool(micro_factor_state.get("schema_ok", True))
    time_sync_ok = bool(micro_factor_state.get("time_sync_ok", True))
    trade_count = int(max(0, _safe_float(micro_factor_state.get("trade_count"), 0.0)))
    cvd_context_mode = str(micro_factor_state.get("cvd_context_mode") or "unclear").strip().lower()
    cvd_trust_tier = str(micro_factor_state.get("cvd_trust_tier_hint") or "unavailable").strip().lower()
    cvd_veto_reason = ""

    if not schema_ok:
        penalty += 0.50 * float(cfg.micro_penalty_max)
        flags.append("micro_schema_risk")
    if not has_data:
        penalty = min(float(cfg.micro_penalty_max) * 1.5, penalty)
        return 0.0, penalty, flags, 0.0
    if not sync_ok:
        penalty += 0.35 * float(cfg.micro_penalty_max)
        flags.append("micro_sync_risk")
    if not gap_ok:
        penalty += 0.25 * float(cfg.micro_penalty_max)
        flags.append("micro_gap_risk")
    if not time_sync_ok:
        penalty += 0.20 * float(cfg.micro_penalty_max)
        flags.append("micro_time_sync_risk")
    if trade_count < int(max(1, cfg.micro_min_trade_count)):
        penalty += 0.15 * float(cfg.micro_penalty_max)
        flags.append("micro_low_samples")

    if isinstance(market_factor_state, dict):
        cross_quality = _bounded(_safe_float(market_factor_state.get("cross_source_quality_score_7d"), 1.0), 0.0, 1.0)
        cross_fail = _bounded(_safe_float(market_factor_state.get("cross_source_fail_ratio_7d"), 0.0), 0.0, 1.0)
        cross_stress = _bounded(_safe_float(market_factor_state.get("cross_source_stress"), 0.0), 0.0, 1.5)
        if cvd_trust_tier != "unavailable":
            if cross_quality >= 0.75 and cross_fail <= 0.25 and cross_stress <= 0.35:
                cvd_trust_tier = "cross_exchange_confirmed"
            elif cross_fail >= 0.50 or cross_stress >= 0.75:
                cvd_trust_tier = "cross_exchange_conflicted"

    if (
        cvd_context_mode == "unclear"
        and signed_signal >= 0.20
        and schema_ok
        and sync_ok
        and gap_ok
        and time_sync_ok
        and trade_count >= int(max(1, cfg.micro_min_trade_count))
    ):
        cvd_context_mode = "continuation"

    if cvd_trust_tier != "unavailable":
        flags.append(f"cvd_trust_{cvd_trust_tier}")
    if cvd_context_mode and cvd_context_mode != "unclear":
        flags.append(f"cvd_context_{cvd_context_mode}")

    if cvd_trust_tier == "cross_exchange_conflicted":
        penalty += 0.18 * float(cfg.micro_penalty_max)
        flags.append("cvd_cross_exchange_conflict")
        cvd_veto_reason = "cross_exchange_conflict"
    elif cvd_trust_tier == "single_exchange_low":
        penalty += 0.10 * float(cfg.micro_penalty_max)
        flags.append("cvd_low_quality")
        if not cvd_veto_reason:
            cvd_veto_reason = "low_sample_or_gap_risk"

    if cvd_context_mode == "continuation" and signed_signal > 0.0 and cvd_trust_tier in {
        "single_exchange_ok",
        "cross_exchange_confirmed",
    }:
        boost += 0.12 * float(cfg.micro_confidence_boost_max)
        flags.append("cvd_continuation_confirmed")
    elif cvd_context_mode == "continuation" and signed_signal <= 0.0:
        penalty += 0.12 * float(cfg.micro_penalty_max)
        flags.append("cvd_displacement_without_flow")
        cvd_veto_reason = cvd_veto_reason or "displacement_without_flow"

    if cvd_context_mode == "reversal" and signed_signal > 0.0 and cvd_trust_tier in {
        "single_exchange_ok",
        "cross_exchange_confirmed",
    }:
        boost += 0.10 * float(cfg.micro_confidence_boost_max)
        flags.append("cvd_reversal_confirmed")
    elif cvd_context_mode == "reversal" and signed_signal <= 0.0:
        penalty += 0.10 * float(cfg.micro_penalty_max)
        flags.append("cvd_sweep_without_confirmation")
        cvd_veto_reason = cvd_veto_reason or "sweep_without_delta_confirmation"

    if cvd_context_mode == "absorption":
        penalty += 0.18 * float(cfg.micro_penalty_max)
        flags.append("cvd_absorption_risk")
        cvd_veto_reason = cvd_veto_reason or "effort_result_divergence"
    if cvd_context_mode == "failed_auction":
        penalty += 0.22 * float(cfg.micro_penalty_max)
        flags.append("cvd_failed_auction_risk")
        cvd_veto_reason = cvd_veto_reason or "effort_result_divergence"

    if cvd_veto_reason:
        flags.append(f"cvd_veto_{cvd_veto_reason}")

    penalty = min(float(cfg.micro_penalty_max) * 1.5, penalty)
    return boost, penalty, flags, signed_signal


def _theory_adjustment(
    *,
    side: Side,
    regime: RegimeLabel,
    cfg: SignalEngineConfig,
    df: pd.DataFrame,
    score_ratio: float,
) -> tuple[float, float, list[str], TheoryConfluenceResult | None]:
    if not bool(cfg.theory_enabled):
        return 0.0, 0.0, [], None
    if side == Side.FLAT:
        return 0.0, 0.0, ["theory_flat_side"], None

    result = compute_theory_confluence(
        df=df,
        side=side,
        regime=regime,
        lie_score_ratio=_bounded(score_ratio, 0.0, 1.0),
        ict_weight=float(cfg.theory_ict_weight),
        brooks_weight=float(cfg.theory_brooks_weight),
        lie_weight=float(cfg.theory_lie_weight),
        wyckoff_weight=float(cfg.theory_wyckoff_weight),
        vpa_weight=float(cfg.theory_vpa_weight),
    )
    boost = float(result.confluence) * float(cfg.theory_confidence_boost_max)
    penalty = float(result.conflict) * float(cfg.theory_penalty_max)
    flags = list(result.flags)

    if float(result.conflict) >= float(cfg.theory_conflict_fuse):
        penalty += 0.25 * float(cfg.theory_penalty_max)
        flags.append("theory_conflict_fuse")

    if float(result.confluence) < float(cfg.theory_min_confluence):
        shortfall = (float(cfg.theory_min_confluence) - float(result.confluence)) / max(float(cfg.theory_min_confluence), 1e-9)
        penalty += shortfall * 0.35 * float(cfg.theory_penalty_max)
        flags.append("theory_confluence_low")

    if float(result.confluence) > float(result.conflict) + 0.25:
        boost += 0.15 * float(cfg.theory_confidence_boost_max)
        flags.append("theory_resonance")

    penalty = min(float(cfg.theory_penalty_max) * 1.8, penalty)
    return boost, penalty, flags, result


def generate_signal_for_symbol(
    symbol_df: pd.DataFrame,
    regime: RegimeLabel,
    cfg: SignalEngineConfig,
    market_factor_state: dict[str, float] | None = None,
    micro_factor_state: dict[str, float] | None = None,
) -> SignalCandidate | None:
    if symbol_df.empty or len(symbol_df) < 30:
        return None

    df = add_common_features(symbol_df)
    cur = df.iloc[-1]
    asset_class = str(cur.get("asset_class", "equity"))
    can_short = asset_class in SHORTABLE_ASSET_CLASS

    score_ratio = 0.0
    if regime in {RegimeLabel.STRONG_TREND, RegimeLabel.WEAK_TREND, RegimeLabel.DOWNTREND}:
        scores = score_trend(df)
        pos = float(scores["position"])
        struct = float(scores["structure"])
        mom = float(scores["momentum"])
        side = scores["side"] if isinstance(scores["side"], Side) else Side.FLAT
        regime_boost = 5.0 if regime == RegimeLabel.STRONG_TREND else 0.0
        raw_confidence = ((pos + struct + mom) / 17.0 * 100.0) + regime_boost
        score_ratio = _bounded((pos + struct + mom) / 17.0, 0.0, 1.0)
    elif regime == RegimeLabel.RANGE:
        scores = score_range(df)
        pos = float(scores["ls"])
        struct = float(scores["mr"])
        mom = float(scores["vc"])
        side = scores["side"] if isinstance(scores["side"], Side) else Side.FLAT
        raw_confidence = (pos + struct + mom) / 14.0 * 100.0
        score_ratio = _bounded((pos + struct + mom) / 14.0, 0.0, 1.0)
    else:
        return None

    if side == Side.FLAT:
        return None

    exposure = _estimate_factor_exposure(df)
    state = _merge_market_factor_state(regime, market_factor_state)
    factor_risk_score = 0.0
    factor_flags: list[str] = []
    if cfg.factor_filter_enabled:
        factor_risk_score, factor_flags = _factor_risk_score(side=side, exposure=exposure, market_factor_state=state)

    if cfg.factor_filter_enabled and factor_risk_score >= cfg.factor_drop_threshold:
        return None

    factor_penalty = min(float(cfg.factor_penalty_max), factor_risk_score * float(cfg.factor_penalty_max))
    theory_boost, theory_penalty, theory_flags, theory_state = _theory_adjustment(
        side=side,
        regime=regime,
        cfg=cfg,
        df=df,
        score_ratio=score_ratio,
    )
    micro_boost, micro_penalty, micro_flags, micro_signed = _microstructure_adjustment(
        side=side,
        cfg=cfg,
        micro_factor_state=micro_factor_state,
        market_factor_state=state,
    )
    confidence = _bounded(raw_confidence - factor_penalty + theory_boost + micro_boost - theory_penalty - micro_penalty, 0.0, 100.0)
    close = float(cur["close"])
    atr = float(cur["atr14"])
    atr_pct = atr / max(close, 1e-9)
    reward_mult = _adaptive_reward_multiplier(
        cfg=cfg,
        regime=regime,
        trend_score_ratio=score_ratio,
        atr_pct=atr_pct,
        factor_risk_score=factor_risk_score,
    )

    entry, stop, target = _build_trade_levels(
        side=side,
        close=close,
        atr=atr,
        stop_mult=_stop_multiplier(regime),
        reward_mult=reward_mult,
    )
    convexity = _convexity_ratio(side=side, entry=entry, stop=stop, target=target)

    if side == Side.SHORT and not can_short:
        note = "S点触发但标的不支持直接做空，转译为减仓/平仓+指数对冲腿"
    else:
        note = f"Regime: {regime.value}; target_mult={reward_mult:.2f}"
    if factor_flags:
        note += " | factor=" + ",".join(factor_flags)
    note += (
        " | flow="
        + f"pressure={_safe_float(state.get('flow_quality_pressure'), 0.0):.3f},"
        + f"gap={_safe_float(exposure.get('flow_effort_gap'), 0.0):.3f},"
        + f"climax={_safe_float(exposure.get('flow_climax'), 0.0):.3f},"
        + f"absorb={_safe_float(exposure.get('flow_absorption_skew'), 0.0):.3f}"
    )
    if theory_flags:
        note += " | theory_flags=" + ",".join(theory_flags)
    if theory_state is not None:
        note += (
            " | theory="
            + f"conf={float(theory_state.confluence):.3f},"
            + f"conflict={float(theory_state.conflict):.3f},"
            + f"ict={float(theory_state.ict_align):.3f},"
            + f"brooks={float(theory_state.brooks_align):.3f},"
            + f"wyckoff={float(theory_state.wyckoff_align):.3f},"
            + f"vpa={float(theory_state.vpa_align):.3f},"
            + f"lie={float(theory_state.lie_align):.3f}"
        )
    if micro_flags:
        note += " | micro_flags=" + ",".join(micro_flags)
    if isinstance(micro_factor_state, dict) and bool(micro_factor_state.get("has_data", False)):
        note += (
            " | micro="
            + f"align={_safe_float(micro_factor_state.get('micro_alignment'), 0.0):.3f},"
            + f"signed={micro_signed:.3f},"
            + f"obi={_safe_float(micro_factor_state.get('queue_imbalance'), 0.0):.3f},"
            + f"ofi={_safe_float(micro_factor_state.get('ofi_norm'), 0.0):.3f},"
            + f"cvd={_safe_float(micro_factor_state.get('cvd_delta_ratio'), 0.0):.3f},"
            + f"ctx={str(micro_factor_state.get('cvd_context_mode', 'unclear'))},"
            + f"trust={str(micro_factor_state.get('cvd_trust_tier_hint', 'unavailable'))}"
        )
        if str(micro_factor_state.get("cvd_context_note", "")).strip():
            note += " | cvd_note=" + str(micro_factor_state.get("cvd_context_note"))
    if isinstance(micro_factor_state, dict) and not bool(micro_factor_state.get("schema_ok", True)):
        issues = micro_factor_state.get("schema_issues", [])
        if isinstance(issues, list) and issues:
            note += " | micro_schema=" + ",".join(str(x) for x in issues[:2])

    if confidence < cfg.confidence_min:
        return None
    if convexity < cfg.convexity_min:
        return None

    return SignalCandidate(
        symbol=str(cur["symbol"]),
        side=side,
        regime=regime,
        position_score=pos,
        structure_score=struct,
        momentum_score=mom,
        confidence=confidence,
        convexity_ratio=convexity,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        can_short=can_short,
        factor_exposure_score=float(factor_risk_score),
        factor_penalty=float(factor_penalty + theory_penalty + micro_penalty),
        factor_flags=list(dict.fromkeys(factor_flags + theory_flags + micro_flags)),
        notes=note,
    )


def scan_signals(
    bars: pd.DataFrame,
    regime: RegimeLabel,
    cfg: SignalEngineConfig,
    market_factor_state: dict[str, float] | None = None,
    micro_factor_map: dict[str, dict[str, Any]] | None = None,
) -> list[SignalCandidate]:
    if bars.empty:
        return []

    if regime == RegimeLabel.EXTREME_VOL:
        return []

    out: list[SignalCandidate] = []
    for _, symbol_df in bars.groupby("symbol"):
        sorted_df = symbol_df.sort_values("ts")
        local_regime = detect_symbol_regime(sorted_df, cfg)
        if local_regime == RegimeLabel.UNCERTAIN and regime not in {RegimeLabel.UNCERTAIN, RegimeLabel.EXTREME_VOL}:
            local_regime = regime
        candidate = generate_signal_for_symbol(
            sorted_df,
            regime=local_regime,
            cfg=cfg,
            market_factor_state=market_factor_state,
            micro_factor_state=(micro_factor_map or {}).get(str(sorted_df.iloc[-1].get("symbol", ""))),
        )
        if candidate is not None:
            out.append(candidate)

    out.sort(key=lambda x: (-x.confidence, x.factor_penalty))
    return out
