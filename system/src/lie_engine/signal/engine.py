from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from lie_engine.models import RegimeLabel, Side, SignalCandidate
from lie_engine.signal.features import add_common_features
from lie_engine.signal.range_engine import score_range
from lie_engine.signal.trend import score_trend


SHORTABLE_ASSET_CLASS = {"future", "option", "hedge"}


@dataclass(slots=True)
class SignalEngineConfig:
    confidence_min: float
    convexity_min: float


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
    return reward / risk if risk else 0.0


def _build_trade_levels(side: Side, close: float, atr: float, regime: RegimeLabel) -> tuple[float, float, float]:
    if regime in {RegimeLabel.STRONG_TREND, RegimeLabel.WEAK_TREND, RegimeLabel.DOWNTREND}:
        stop_mult = 1.5
    else:
        stop_mult = 1.0

    risk = max(atr * stop_mult, close * 0.01)
    if side == Side.SHORT:
        stop = close + risk
        target = close - 3.2 * risk
    else:
        stop = close - risk
        target = close + 3.2 * risk
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


def generate_signal_for_symbol(symbol_df: pd.DataFrame, regime: RegimeLabel, cfg: SignalEngineConfig) -> SignalCandidate | None:
    if symbol_df.empty or len(symbol_df) < 30:
        return None

    df = add_common_features(symbol_df)
    cur = df.iloc[-1]
    asset_class = str(cur["asset_class"])
    can_short = asset_class in SHORTABLE_ASSET_CLASS

    if regime in {RegimeLabel.STRONG_TREND, RegimeLabel.WEAK_TREND, RegimeLabel.DOWNTREND}:
        scores = score_trend(df)
        pos = float(scores["position"])
        struct = float(scores["structure"])
        mom = float(scores["momentum"])
        side = scores["side"] if isinstance(scores["side"], Side) else Side.FLAT
        confidence = (pos + struct + mom) / 17.0 * 100.0
    elif regime == RegimeLabel.RANGE:
        scores = score_range(df)
        pos = float(scores["ls"])
        struct = float(scores["mr"])
        mom = float(scores["vc"])
        side = scores["side"] if isinstance(scores["side"], Side) else Side.FLAT
        confidence = (pos + struct + mom) / 14.0 * 100.0
    else:
        return None

    entry, stop, target = _build_trade_levels(
        side=side,
        close=float(cur["close"]),
        atr=float(cur["atr14"]),
        regime=regime,
    )
    convexity = _convexity_ratio(side=side, entry=entry, stop=stop, target=target)

    if side == Side.SHORT and not can_short:
        note = "S点触发但标的不支持直接做空，转译为减仓/平仓+指数对冲腿"
    else:
        note = ""

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
        notes=note,
    )


def scan_signals(bars: pd.DataFrame, regime: RegimeLabel, cfg: SignalEngineConfig) -> list[SignalCandidate]:
    if bars.empty or regime in {RegimeLabel.UNCERTAIN, RegimeLabel.EXTREME_VOL}:
        return []

    out: list[SignalCandidate] = []
    for symbol, symbol_df in bars.groupby("symbol"):
        candidate = generate_signal_for_symbol(symbol_df.sort_values("ts"), regime=regime, cfg=cfg)
        if candidate is not None:
            out.append(candidate)

    out.sort(key=lambda x: x.confidence, reverse=True)
    return out
