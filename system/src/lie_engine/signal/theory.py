from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lie_engine.models import RegimeLabel, Side
from lie_engine.signal.features import candle_pattern_score


@dataclass(slots=True)
class TheoryConfluenceResult:
    confluence: float
    conflict: float
    ict_align: float
    ict_oppose: float
    brooks_align: float
    brooks_oppose: float
    lie_align: float
    lie_oppose: float
    flags: list[str]


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


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    num = 0.0
    den = 0.0
    for value, weight in values:
        w = max(0.0, float(weight))
        v = _bounded(float(value), 0.0, 1.0)
        num += v * w
        den += w
    if den <= 1e-9:
        return 0.0
    return float(num / den)


def _ict_scores(df: pd.DataFrame) -> tuple[float, float]:
    if len(df) < 2:
        return 0.0, 0.0

    prev = df.iloc[-2]
    cur = df.iloc[-1]

    span = max(1e-9, _safe_float(cur.get("roll_high20"), 0.0) - _safe_float(cur.get("roll_low20"), 0.0))
    retrace = (_safe_float(cur.get("close"), 0.0) - _safe_float(cur.get("roll_low20"), 0.0)) / span
    pd_long = 1.0 if 0.618 <= retrace <= 0.786 else 0.0
    pd_short = 1.0 if 0.214 <= retrace <= 0.382 else 0.0

    close = _safe_float(cur.get("close"), 0.0)
    roll_high10_prev = _safe_float(cur.get("roll_high10_prev"), 0.0)
    roll_low10_prev = _safe_float(cur.get("roll_low10_prev"), 0.0)
    bos_long = 1.0 if close > roll_high10_prev else 0.0
    bos_short = 1.0 if close < roll_low10_prev else 0.0

    high = _safe_float(cur.get("high"), 0.0)
    low = _safe_float(cur.get("low"), 0.0)
    sweep_short = 1.0 if high > roll_high10_prev and close < roll_high10_prev else 0.0
    sweep_long = 1.0 if low < roll_low10_prev and close > roll_low10_prev else 0.0

    prev_high = _safe_float(prev.get("high"), 0.0)
    prev_low = _safe_float(prev.get("low"), 0.0)
    volume = _safe_float(cur.get("volume"), 0.0)
    vol_ma20 = max(1e-9, _safe_float(cur.get("vol_ma20"), 0.0))
    gap_up = 1.0 if low > prev_high else 0.0
    gap_down = 1.0 if high < prev_low else 0.0
    vol_boost = 1.0 if volume >= 1.2 * vol_ma20 else 0.65
    fvg_long = gap_up * vol_boost
    fvg_short = gap_down * vol_boost

    ict_long = _weighted_mean(
        [
            (pd_long, 0.25),
            (bos_long, 0.30),
            (sweep_long, 0.20),
            (fvg_long, 0.25),
        ]
    )
    ict_short = _weighted_mean(
        [
            (pd_short, 0.25),
            (bos_short, 0.30),
            (sweep_short, 0.20),
            (fvg_short, 0.25),
        ]
    )
    return ict_long, ict_short


def _brooks_scores(df: pd.DataFrame) -> tuple[float, float]:
    if len(df) < 5:
        return 0.0, 0.0

    prev = df.iloc[-2]
    cur = df.iloc[-1]
    prev2 = df.iloc[-3]
    prev3 = df.iloc[-4]

    open_px = _safe_float(cur.get("open"), 0.0)
    close_px = _safe_float(cur.get("close"), 0.0)
    high = _safe_float(cur.get("high"), close_px)
    low = _safe_float(cur.get("low"), close_px)
    volume = _safe_float(cur.get("volume"), 0.0)
    body = abs(close_px - open_px)
    atr14 = max(1e-9, _safe_float(cur.get("atr14"), high - low))
    bar_range = max(1e-9, high - low)
    upper_wick = max(0.0, high - max(close_px, open_px))
    lower_wick = max(0.0, min(close_px, open_px) - low)
    wick_ratio = (upper_wick + lower_wick) / max(body, 1e-9)
    close_location = _bounded((close_px - low) / bar_range, 0.0, 1.0)
    body_share = _bounded(body / bar_range, 0.0, 1.0)
    body_vs_atr = _bounded(body / atr14, 0.0, 1.0)
    vol_ma20 = max(1e-9, _safe_float(cur.get("vol_ma20"), volume if volume > 0.0 else 1.0))
    vol_support = _bounded((volume / vol_ma20 - 0.6) / 1.0, 0.0, 1.0)

    trend_bar_long = 0.0
    trend_bar_short = 0.0
    if close_px >= open_px:
        trend_bar_long = _bounded(close_location * body_share * body_vs_atr * (0.55 + 0.45 * vol_support), 0.0, 1.0)
    if close_px <= open_px:
        trend_bar_short = _bounded((1.0 - close_location) * body_share * body_vs_atr * (0.55 + 0.45 * vol_support), 0.0, 1.0)

    roll_high10_prev = _safe_float(prev.get("roll_high10_prev"), _safe_float(cur.get("roll_high10_prev"), 0.0))
    roll_low10_prev = _safe_float(prev.get("roll_low10_prev"), _safe_float(cur.get("roll_low10_prev"), 0.0))
    broke_high_revert = high > roll_high10_prev and close_px < roll_high10_prev
    broke_low_revert = low < roll_low10_prev and close_px > roll_low10_prev

    failed_breakout_long = 1.0 if broke_low_revert and wick_ratio > 2.0 else 0.0
    failed_breakout_short = 1.0 if broke_high_revert and wick_ratio > 2.0 else 0.0

    roll_high20 = _safe_float(cur.get("roll_high20"), close_px)
    roll_low20 = _safe_float(cur.get("roll_low20"), close_px)
    roll_high20_prev = _safe_float(prev.get("roll_high20"), roll_high20)
    roll_low20_prev = _safe_float(prev.get("roll_low20"), roll_low20)
    ma20 = _safe_float(cur.get("ma20"), close_px)
    ma60 = _safe_float(cur.get("ma60"), close_px)
    ma_gap = abs(close_px - ma20) / atr14
    pullback_tightness = _bounded(1.0 - ma_gap / 1.8, 0.0, 1.0)
    breakout_pullback_long = pullback_tightness if close_px >= roll_high20 * 0.992 and close_px > ma20 and ma20 >= ma60 else 0.0
    breakout_pullback_short = pullback_tightness if close_px <= roll_low20 * 1.008 and close_px < ma20 and ma20 <= ma60 else 0.0

    pullback_seq = df.tail(6).iloc[:-1]
    pb_close = pullback_seq["close"].to_numpy(dtype=float)
    pb_high = pullback_seq["high"].to_numpy(dtype=float)
    pb_low = pullback_seq["low"].to_numpy(dtype=float)
    pb_ret = np.diff(pb_close)
    leg_threshold = 0.12 * atr14
    down_legs = int(np.sum(pb_ret <= -leg_threshold))
    up_legs = int(np.sum(pb_ret >= leg_threshold))
    prev_high = _safe_float(prev.get("high"), high)
    prev_low = _safe_float(prev.get("low"), low)
    pullback_span = (float(np.max(pb_high)) - float(np.min(pb_low))) if len(pb_high) and len(pb_low) else 0.0
    pullback_noise = _bounded(pullback_span / max(2.4 * atr14, 1e-9), 0.0, 1.0)
    reclaim_long = _bounded((close_px - prev_high) / max(0.6 * atr14, 1e-9) + 0.5, 0.0, 1.0)
    reclaim_short = _bounded((prev_low - close_px) / max(0.6 * atr14, 1e-9) + 0.5, 0.0, 1.0)

    two_legged_pullback_long = 0.0
    if ma20 >= ma60 and close_px >= ma20 * 0.995 and down_legs >= 2 and up_legs >= 1 and close_px >= prev_high * 0.999:
        two_legged_pullback_long = _bounded(0.35 + 0.40 * reclaim_long + 0.25 * (1.0 - pullback_noise), 0.0, 1.0)

    two_legged_pullback_short = 0.0
    if ma20 <= ma60 and close_px <= ma20 * 1.005 and up_legs >= 2 and down_legs >= 1 and close_px <= prev_low * 1.001:
        two_legged_pullback_short = _bounded(0.35 + 0.40 * reclaim_short + 0.25 * (1.0 - pullback_noise), 0.0, 1.0)

    cps = candle_pattern_score(df.tail(6))
    attempt2_break_long = max(
        _safe_float(prev.get("high"), high),
        _safe_float(prev2.get("high"), high),
        _safe_float(prev3.get("high"), high),
    )
    attempt2_break_short = min(
        _safe_float(prev.get("low"), low),
        _safe_float(prev2.get("low"), low),
        _safe_float(prev3.get("low"), low),
    )
    h2_long = (
        1.0
        if cps >= 1.0
        and close_px > _safe_float(prev.get("close"), close_px)
        and _safe_float(prev.get("low"), low) <= _safe_float(prev2.get("low"), low)
        and close_px > attempt2_break_long * 0.998
        else 0.0
    )
    h2_short = (
        1.0
        if cps >= 1.0
        and close_px < _safe_float(prev.get("close"), close_px)
        and _safe_float(prev.get("high"), high) >= _safe_float(prev2.get("high"), high)
        and close_px < attempt2_break_short * 1.002
        else 0.0
    )

    near_lower = close_px <= roll_low20 * 1.01
    near_upper = close_px >= roll_high20 * 0.99
    rsi14 = _safe_float(cur.get("rsi14"), 50.0)
    range_reversal_long = 1.0 if near_lower and rsi14 <= 35 and lower_wick > upper_wick * 1.2 and body <= atr14 * 1.4 else 0.0
    range_reversal_short = 1.0 if near_upper and rsi14 >= 65 and upper_wick > lower_wick * 1.2 and body <= atr14 * 1.4 else 0.0

    recent = df.tail(4)
    high_steps = np.diff(recent["high"].to_numpy(dtype=float))
    low_steps = np.diff(recent["low"].to_numpy(dtype=float))
    micro_channel_long = _bounded(
        0.5 * (float(np.mean(high_steps >= -1e-9)) + float(np.mean(low_steps >= -1e-9))),
        0.0,
        1.0,
    )
    micro_channel_short = _bounded(
        0.5 * (float(np.mean(high_steps <= 1e-9)) + float(np.mean(low_steps <= 1e-9))),
        0.0,
        1.0,
    )

    exhaustion_long = 0.0
    if close_px >= roll_high20_prev * 0.998:
        upside_exhaust = _bounded((upper_wick / max(body, 1e-9) - 0.45) / 1.6, 0.0, 1.0)
        impulse = _bounded((body / max(1.2 * atr14, 1e-9)) - 0.4, 0.0, 1.0)
        exhaustion_long = _bounded(0.5 * upside_exhaust + 0.5 * impulse * vol_support, 0.0, 1.0)

    exhaustion_short = 0.0
    if close_px <= roll_low20_prev * 1.002:
        downside_exhaust = _bounded((lower_wick / max(body, 1e-9) - 0.45) / 1.6, 0.0, 1.0)
        impulse = _bounded((body / max(1.2 * atr14, 1e-9)) - 0.4, 0.0, 1.0)
        exhaustion_short = _bounded(0.5 * downside_exhaust + 0.5 * impulse * vol_support, 0.0, 1.0)

    brooks_long = _weighted_mean(
        [
            (failed_breakout_long, 0.21),
            (breakout_pullback_long, 0.20),
            (h2_long, 0.13),
            (range_reversal_long, 0.16),
            (trend_bar_long, 0.12),
            (micro_channel_long, 0.09),
            (two_legged_pullback_long, 0.09),
        ]
    )
    brooks_short = _weighted_mean(
        [
            (failed_breakout_short, 0.21),
            (breakout_pullback_short, 0.20),
            (h2_short, 0.13),
            (range_reversal_short, 0.16),
            (trend_bar_short, 0.12),
            (micro_channel_short, 0.09),
            (two_legged_pullback_short, 0.09),
        ]
    )
    brooks_long = _bounded(brooks_long * (1.0 - 0.55 * exhaustion_long) + 0.28 * exhaustion_short, 0.0, 1.0)
    brooks_short = _bounded(brooks_short * (1.0 - 0.55 * exhaustion_short) + 0.28 * exhaustion_long, 0.0, 1.0)
    return brooks_long, brooks_short


def compute_theory_confluence(
    *,
    df: pd.DataFrame,
    side: Side,
    regime: RegimeLabel,
    lie_score_ratio: float,
    ict_weight: float,
    brooks_weight: float,
    lie_weight: float,
) -> TheoryConfluenceResult:
    if df.empty or len(df) < 30:
        return TheoryConfluenceResult(
            confluence=0.0,
            conflict=0.0,
            ict_align=0.0,
            ict_oppose=0.0,
            brooks_align=0.0,
            brooks_oppose=0.0,
            lie_align=0.0,
            lie_oppose=0.0,
            flags=["theory_insufficient_history"],
        )

    ict_long, ict_short = _ict_scores(df)
    brooks_long, brooks_short = _brooks_scores(df)
    lie_align = _bounded((_safe_float(lie_score_ratio, 0.0) - 0.30) / 0.70, 0.0, 1.0)
    lie_oppose = _bounded((0.55 - _safe_float(lie_score_ratio, 0.0)) / 0.55, 0.0, 1.0)

    if side == Side.LONG:
        ict_align = ict_long
        ict_oppose = ict_short
        brooks_align = brooks_long
        brooks_oppose = brooks_short
    elif side == Side.SHORT:
        ict_align = ict_short
        ict_oppose = ict_long
        brooks_align = brooks_short
        brooks_oppose = brooks_long
    else:
        return TheoryConfluenceResult(
            confluence=0.0,
            conflict=1.0,
            ict_align=0.0,
            ict_oppose=0.0,
            brooks_align=0.0,
            brooks_oppose=0.0,
            lie_align=0.0,
            lie_oppose=1.0,
            flags=["theory_flat_side"],
        )

    w_ict = max(0.0, float(ict_weight))
    w_brooks = max(0.0, float(brooks_weight))
    w_lie = max(0.0, float(lie_weight))
    total = max(1e-9, w_ict + w_brooks + w_lie)

    confluence = _bounded((w_ict * ict_align + w_brooks * brooks_align + w_lie * lie_align) / total, 0.0, 1.0)
    conflict = _bounded((w_ict * ict_oppose + w_brooks * brooks_oppose + w_lie * lie_oppose) / total, 0.0, 1.0)

    flags: list[str] = []
    if confluence >= 0.70:
        flags.append("theory_confluence_high")
    elif confluence <= 0.30:
        flags.append("theory_confluence_weak")

    if conflict >= 0.60:
        flags.append("theory_conflict_high")

    if abs(ict_align - brooks_align) >= 0.55:
        flags.append("theory_family_divergence")

    if regime in {RegimeLabel.STRONG_TREND, RegimeLabel.WEAK_TREND, RegimeLabel.DOWNTREND} and brooks_oppose >= 0.55:
        flags.append("brooks_countertrend_risk")
    if regime == RegimeLabel.RANGE and max(ict_oppose, brooks_oppose) >= 0.60:
        flags.append("range_breakout_risk")

    return TheoryConfluenceResult(
        confluence=confluence,
        conflict=conflict,
        ict_align=ict_align,
        ict_oppose=ict_oppose,
        brooks_align=brooks_align,
        brooks_oppose=brooks_oppose,
        lie_align=lie_align,
        lie_oppose=lie_oppose,
        flags=flags,
    )
