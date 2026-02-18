from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

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
from lie_engine.signal.trend import score_trend


SHORTABLE_ASSET_CLASS = {"future", "option", "hedge"}


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
        last_ts = ts.max().date() if not ts.dropna().empty else date.today()

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
        }
    if regime == RegimeLabel.WEAK_TREND:
        return {
            "valuation_pressure": 0.45,
            "momentum_preference": 0.55,
            "crowding_aversion": 0.45,
            "small_cap_pressure": 0.35,
            "dividend_preference": 0.50,
        }
    if regime == RegimeLabel.RANGE:
        return {
            "valuation_pressure": 0.60,
            "momentum_preference": 0.35,
            "crowding_aversion": 0.55,
            "small_cap_pressure": 0.55,
            "dividend_preference": 0.65,
        }
    return {
        "valuation_pressure": 0.65,
        "momentum_preference": 0.40,
        "crowding_aversion": 0.60,
        "small_cap_pressure": 0.65,
        "dividend_preference": 0.70,
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
    ):
        if key in market_factor_state:
            merged[key] = _bounded(_safe_float(market_factor_state.get(key), merged[key]), 0.0, 1.5)
    return merged


def _estimate_factor_exposure(df: pd.DataFrame) -> dict[str, float]:
    cur = df.iloc[-1]
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

    if side == Side.SHORT:
        valuation *= 0.35
        momentum_mismatch = max(0.0, momentum)
    else:
        momentum_mismatch = max(0.0, -momentum)
    dividend_mismatch = dividend * max(0.0, market_factor_state.get("dividend_preference", 0.0))

    raw = (
        valuation * market_factor_state["valuation_pressure"]
        + momentum_mismatch * market_factor_state["momentum_preference"] * 0.9
        + crowded * market_factor_state["crowding_aversion"] * 0.7
        + size * market_factor_state["small_cap_pressure"] * 0.6
        + volatility * 0.45
        + dividend_mismatch * 0.55
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
    return score, flags


def generate_signal_for_symbol(
    symbol_df: pd.DataFrame,
    regime: RegimeLabel,
    cfg: SignalEngineConfig,
    market_factor_state: dict[str, float] | None = None,
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
    confidence = _bounded(raw_confidence - factor_penalty, 0.0, 100.0)
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
        factor_penalty=float(factor_penalty),
        factor_flags=factor_flags,
        notes=note,
    )


def scan_signals(
    bars: pd.DataFrame,
    regime: RegimeLabel,
    cfg: SignalEngineConfig,
    market_factor_state: dict[str, float] | None = None,
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
        )
        if candidate is not None:
            out.append(candidate)

    out.sort(key=lambda x: (-x.confidence, x.factor_penalty))
    return out
