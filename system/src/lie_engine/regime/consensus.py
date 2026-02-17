from __future__ import annotations

from datetime import date

from lie_engine.models import RegimeLabel, RegimeState


def _vote_hurst(hurst: float, trend_thr: float, mean_thr: float) -> str:
    if hurst > trend_thr:
        return "trend"
    if hurst < mean_thr:
        return "range"
    return "neutral"


def _vote_hmm(hmm_probs: dict[str, float], min_prob: float = 0.6) -> str:
    state = max(hmm_probs, key=hmm_probs.get)
    if hmm_probs[state] < min_prob:
        return "uncertain"
    if state == "bull":
        return "trend"
    if state == "bear":
        return "downtrend"
    return "range"


def _vote_atr(atr_z: float) -> str:
    if atr_z > 1.0:
        return "trend"
    if -1.0 <= atr_z <= 1.0:
        return "range"
    return "neutral"


def derive_regime_consensus(
    as_of: date,
    hurst: float,
    hmm_probs: dict[str, float],
    atr_z: float,
    trend_thr: float,
    mean_thr: float,
    atr_extreme: float,
) -> RegimeState:
    if atr_z > atr_extreme:
        return RegimeState(
            as_of=as_of,
            hurst=hurst,
            hmm_probs=hmm_probs,
            atr_z=atr_z,
            consensus=RegimeLabel.EXTREME_VOL,
            protection_mode=True,
            rationale="ATR_Z 超过极端阈值，强制保护模式",
        )

    h_vote = _vote_hurst(hurst, trend_thr=trend_thr, mean_thr=mean_thr)
    m_vote = _vote_hmm(hmm_probs)
    a_vote = _vote_atr(atr_z)

    votes = [h_vote, m_vote, a_vote]
    trend_votes = votes.count("trend")
    range_votes = votes.count("range")
    down_votes = votes.count("downtrend")

    if down_votes >= 1 and m_vote == "downtrend" and hurst > trend_thr and atr_z > 1.0:
        consensus = RegimeLabel.DOWNTREND
        rationale = f"H={hurst:.2f}, HMM偏熊, ATR_Z={atr_z:.2f}"
    elif trend_votes >= 2:
        consensus = RegimeLabel.STRONG_TREND if atr_z > 0 else RegimeLabel.WEAK_TREND
        rationale = f"2/3 多数决趋势: votes={votes}"
    elif range_votes >= 2:
        consensus = RegimeLabel.RANGE
        rationale = f"2/3 多数决震荡: votes={votes}"
    else:
        consensus = RegimeLabel.UNCERTAIN
        rationale = f"模型矛盾或置信度不足: votes={votes}"

    return RegimeState(
        as_of=as_of,
        hurst=hurst,
        hmm_probs=hmm_probs,
        atr_z=atr_z,
        consensus=consensus,
        protection_mode=consensus in {RegimeLabel.UNCERTAIN, RegimeLabel.EXTREME_VOL},
        rationale=rationale,
    )
