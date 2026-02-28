from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Sequence

import pandas as pd

from lie_engine.models import NewsEvent, RegimeLabel


@dataclass(slots=True)
class GuardAssessment:
    black_swan_score: float
    black_swan_items: list[str]
    black_swan_trigger: bool
    major_event_window: bool
    cooldown_active: bool
    non_trade_reasons: list[str]

    @property
    def trade_blocked(self) -> bool:
        return bool(self.non_trade_reasons)


def major_event_window(as_of: date, news: Sequence[NewsEvent], lookback_hours: int = 24) -> bool:
    if not news:
        return False
    start_ts = datetime.combine(as_of, time(0, 0)) - timedelta(hours=max(1, int(lookback_hours)))
    end_ts = datetime.combine(as_of, time(23, 59))
    keywords = (
        "fomc",
        "议息",
        "政治局",
        "制裁",
        "战争",
        "停火",
        "交割",
        "tariff",
        "sanction",
        "ceasefire",
    )
    for ev in news:
        if not (start_ts <= ev.ts <= end_ts):
            continue
        txt = f"{ev.title} {ev.content}".lower()
        category_hit = ev.category in {"政策", "地缘"}
        keyword_hit = any(k in txt for k in keywords)
        if (category_hit and ev.importance >= 0.9 and ev.confidence >= 0.8) or (keyword_hit and ev.importance >= 0.8):
            return True
    return False


def loss_cooldown_active(recent_trades: pd.DataFrame, cooldown_losses: int = 3) -> bool:
    if recent_trades.empty or "pnl" not in recent_trades.columns:
        return False
    threshold = max(1, int(cooldown_losses))
    pnl = recent_trades["pnl"].astype(float).tolist()
    if not pnl:
        return False
    streak = 0
    for v in pnl:
        if v < 0:
            streak += 1
            if streak >= threshold:
                return True
        else:
            break
    return False


def black_swan_assessment(
    atr_z: float,
    sentiment: dict[str, float],
    news: Sequence[NewsEvent],
    threshold: float = 70.0,
) -> tuple[float, list[str], bool]:
    score = 0.0
    items: list[str] = []

    if atr_z > 2.0:
        score += 40.0
        items.append("ATR_Z>2.0，进入极端波动区间")
    elif atr_z > 1.2:
        score += 20.0
        items.append("ATR_Z高位，波动扩张加速")

    pcr = float(sentiment.get("pcr_50etf", 0.9))
    if pcr > 1.2:
        score += 20.0
        items.append("PCR>1.2，期权恐慌对冲升温")
    elif pcr > 1.0:
        score += 10.0

    iv = float(sentiment.get("iv_50etf", 0.22))
    if iv > 0.35:
        score += 15.0
        items.append("隐含波动率快速上行")
    elif iv > 0.28:
        score += 8.0

    northbound = float(sentiment.get("northbound_netflow", 0.0))
    if northbound < -1e9:
        score += 20.0
        items.append("北向净流出超过10亿，流动性显著收缩")
    elif northbound < -5e8:
        score += 10.0

    margin_chg = float(sentiment.get("margin_balance_chg", 0.0))
    if margin_chg < -0.02:
        score += 10.0
        items.append("融资余额收缩过快，杠杆偏好下降")

    geo_events = sum(
        1
        for ev in news
        if ev.category == "地缘" and float(ev.importance) >= 0.75 and float(ev.confidence) >= 0.70
    )
    if geo_events:
        score += min(20.0, geo_events * 6.0)
        items.append(f"地缘高冲击事件计数={geo_events}")

    score = float(min(100.0, max(0.0, score)))
    trigger = score >= float(threshold)
    return score, items, trigger


def build_guard_assessment(
    *,
    as_of: date,
    regime: RegimeLabel,
    atr_z: float,
    quality_passed: bool,
    sentiment: dict[str, float],
    news: Sequence[NewsEvent],
    recent_trades: pd.DataFrame,
    lookback_hours: int,
    cooldown_losses: int,
    black_swan_threshold: float,
) -> GuardAssessment:
    bs_score, bs_items, bs_trigger = black_swan_assessment(
        atr_z=atr_z,
        sentiment=sentiment,
        news=news,
        threshold=black_swan_threshold,
    )
    event_window = major_event_window(as_of=as_of, news=news, lookback_hours=lookback_hours)
    cooldown = loss_cooldown_active(recent_trades=recent_trades, cooldown_losses=cooldown_losses)

    non_trade_reasons: list[str] = []
    if event_window:
        non_trade_reasons.append("重大事件窗口：暂停新增仓位")
    if not quality_passed:
        non_trade_reasons.append("数据质量未通过门禁：进入保护模式并禁止开新仓")
    if regime in {RegimeLabel.UNCERTAIN, RegimeLabel.EXTREME_VOL}:
        non_trade_reasons.append("体制冲突/极端波动：不做方向强判")
    if cooldown:
        non_trade_reasons.append("连亏冷却期未结束：暂停新增仓位")
    if bs_trigger:
        non_trade_reasons.append("黑天鹅评分超阈值：保护模式生效")

    return GuardAssessment(
        black_swan_score=bs_score,
        black_swan_items=bs_items,
        black_swan_trigger=bs_trigger,
        major_event_window=event_window,
        cooldown_active=cooldown,
        non_trade_reasons=non_trade_reasons,
    )

