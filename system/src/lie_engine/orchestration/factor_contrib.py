from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_FACTOR_CONTRIB = {
    "macro": 0.15,
    "industry": 0.15,
    "news": 0.20,
    "sentiment": 0.15,
    "fundamental": 0.10,
    "technical": 0.25,
}


def estimate_factor_contrib_120d(
    *,
    bars: pd.DataFrame,
    ingest: Any,
    providers: Sequence[Any],
    lookback_days: int = 120,
) -> dict[str, float]:
    if bars.empty:
        return dict(DEFAULT_FACTOR_CONTRIB)

    work = bars.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    work = work.sort_values(["symbol", "ts"])
    work["ret"] = work.groupby("symbol")["close"].pct_change()
    daily_ret = work.groupby(work["ts"].dt.date)["ret"].mean().dropna().tail(max(20, int(lookback_days)))
    if daily_ret.empty:
        return dict(DEFAULT_FACTOR_CONTRIB)

    work["ma20"] = work.groupby("symbol")["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    work["technical_signal"] = (work["close"] / work["ma20"] - 1.0).replace([np.inf, -np.inf], np.nan)
    technical_daily = work.groupby(work["ts"].dt.date)["technical_signal"].mean().reindex(daily_ret.index).fillna(0.0)
    industry_dispersion = work.groupby(work["ts"].dt.date)["ret"].std(ddof=0).reindex(daily_ret.index).fillna(0.0)

    senti_series: dict[date, float] = {}
    for d in daily_ret.index:
        snap: dict[str, float] = {}
        for provider in providers:
            try:
                snap = {**snap, **provider.fetch_sentiment_factors(d)}
            except Exception:
                continue
        pcr = float(snap.get("pcr_50etf", 0.9))
        iv = float(snap.get("iv_50etf", 0.22))
        north = float(snap.get("northbound_netflow", 0.0))
        margin = float(snap.get("margin_balance_chg", 0.0))
        senti_series[d] = (-pcr) + (-2.0 * iv) + (north / 1e9) + (20.0 * margin)
    sentiment_daily = pd.Series(senti_series).reindex(daily_ret.index).fillna(0.0)

    news_daily: dict[date, float] = {}
    for ev in getattr(ingest, "news", []):
        d = ev.ts.date()
        if d not in news_daily:
            news_daily[d] = 0.0
        news_daily[d] += float(ev.importance) * float(ev.confidence)
    news_series = pd.Series(news_daily).reindex(daily_ret.index).fillna(0.0)

    macro_daily = pd.Series(0.0, index=daily_ret.index)
    macro_df = getattr(ingest, "macro", pd.DataFrame())
    if isinstance(macro_df, pd.DataFrame) and not macro_df.empty and "date" in macro_df.columns:
        macro = macro_df.copy()
        macro["date"] = pd.to_datetime(macro["date"]).dt.date
        macro_sig = pd.Series(0.0, index=daily_ret.index)
        for _, row in macro.iterrows():
            d = row["date"]
            if d not in macro_sig.index:
                continue
            cpi = float(row.get("cpi_yoy", 0.0))
            ppi = float(row.get("ppi_yoy", 0.0))
            lpr = float(row.get("lpr_1y", 3.45))
            macro_sig.loc[d] = (-abs(cpi - ppi)) + (3.6 - lpr)
        macro_daily = macro_sig.replace([np.inf, -np.inf], 0.0).ffill().fillna(0.0)

    factors = {
        "macro": macro_daily,
        "industry": industry_dispersion,
        "news": news_series,
        "sentiment": sentiment_daily,
        "technical": technical_daily,
    }
    contrib: dict[str, float] = {}
    y = daily_ret.astype(float)
    for name, series in factors.items():
        x = series.astype(float)
        if x.std(ddof=0) < 1e-9 or y.std(ddof=0) < 1e-9:
            contrib[name] = 0.01
            continue
        corr = float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        contrib[name] = max(0.01, abs(corr))

    rolling_mean = y.rolling(20, min_periods=5).mean().fillna(0.0)
    rolling_std = y.rolling(20, min_periods=5).std(ddof=0).replace(0.0, np.nan).fillna(1e-6)
    stability = (rolling_mean / rolling_std).replace([np.inf, -np.inf], 0.0)
    contrib["fundamental"] = max(0.01, float(abs(stability.mean())))
    return contrib

