from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from lie_engine.models import AssetClass, NewsEvent


def _seed(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)


def _asset_class_from_symbol(symbol: str) -> AssetClass:
    if symbol.startswith(("LC", "SC", "RB")):
        return AssetClass.FUTURE
    if symbol.startswith("5"):
        return AssetClass.ETF
    return AssetClass.EQUITY


@dataclass(slots=True)
class OpenSourcePrimaryProvider:
    name: str = "open_source_primary"

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
        if freq != "1d":
            raise ValueError("Only daily frequency is supported in v1")
        rng = np.random.default_rng(_seed(self.name, symbol, str(start), str(end)))
        idx = pd.bdate_range(start=start, end=end)
        n = len(idx)
        if n == 0:
            return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])

        base = 8.0 if symbol.startswith("5") else 45.0
        trend = np.linspace(0, 0.12 * n, n)
        noise = rng.normal(0, 1.5, n).cumsum()
        close = np.maximum(0.1, base + trend + noise)
        spread = np.maximum(0.05, np.abs(rng.normal(0.5, 0.25, n)))
        high = close + spread
        low = np.maximum(0.01, close - spread)
        open_ = close + rng.normal(0, 0.2, n)
        volume = np.maximum(1e5, rng.normal(6e6, 2e6, n))

        df = pd.DataFrame(
            {
                "ts": idx,
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "source": self.name,
                "asset_class": _asset_class_from_symbol(symbol).value,
            }
        )
        return df

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        idx = pd.date_range(start=start, end=end, freq="MS")
        if len(idx) == 0:
            idx = pd.DatetimeIndex([pd.Timestamp(start)])
        rng = np.random.default_rng(_seed(self.name, "macro", str(start), str(end)))
        df = pd.DataFrame(
            {
                "date": idx,
                "cpi_yoy": rng.normal(0.8, 0.6, len(idx)),
                "ppi_yoy": rng.normal(-1.2, 0.8, len(idx)),
                "lpr_1y": 3.45 + rng.normal(0, 0.02, len(idx)),
                "source": self.name,
            }
        )
        return df

    def fetch_news(self, start_ts: datetime, end_ts: datetime, lang: str) -> list[NewsEvent]:
        buckets = [
            ("政策", "国务院发布专项支持政策，强调稳增长与产业升级"),
            ("宏观", "CPI/PPI公布后市场对通缩与复苏预期再平衡"),
            ("地缘", "海外地缘局势出现阶段性缓和信号"),
            ("产业链", "锂电排产与原材料价格出现分化"),
        ]
        step_hours = 6
        out: list[NewsEvent] = []
        current = start_ts
        i = 0
        while current <= end_ts:
            c, content = buckets[i % len(buckets)]
            title = f"[{lang}] {c} 监测快讯 {current:%Y%m%d-%H%M}"
            eid = hashlib.md5(f"{title}|{self.name}".encode("utf-8"), usedforsecurity=False).hexdigest()
            conf = 0.88 if c in {"政策", "宏观"} else 0.72
            out.append(
                NewsEvent(
                    event_id=eid,
                    ts=current,
                    title=title,
                    content=content,
                    lang=lang,
                    source=self.name,
                    category=c,
                    confidence=conf,
                    entities=["A股", "恒生科技", "碳酸锂"],
                    importance=0.7 + 0.05 * (i % 3),
                )
            )
            current += timedelta(hours=step_hours)
            i += 1
        return out

    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]:
        rng = np.random.default_rng(_seed(self.name, "sentiment", str(as_of)))
        return {
            "pcr_50etf": float(np.clip(rng.normal(0.9, 0.2), 0.2, 2.0)),
            "iv_50etf": float(np.clip(rng.normal(0.22, 0.06), 0.08, 0.8)),
            "northbound_netflow": float(rng.normal(5e8, 3e8)),
            "margin_balance_chg": float(rng.normal(0.003, 0.01)),
        }


@dataclass(slots=True)
class OpenSourceSecondaryProvider(OpenSourcePrimaryProvider):
    name: str = "open_source_secondary"

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
        primary = OpenSourcePrimaryProvider(name="open_source_primary")
        df = primary.fetch_ohlcv(symbol=symbol, start=start, end=end, freq=freq)
        rng = np.random.default_rng(_seed(self.name, symbol, str(start), str(end), "drift"))
        # Tiny deterministic drift used for cross-source validation tests.
        df["close"] = df["close"] * (1.0 + rng.normal(0.0, 0.0008, len(df)))
        df["open"] = df["open"] * (1.0 + rng.normal(0.0, 0.0008, len(df)))
        high_base = np.maximum(df["open"], df["close"])
        low_base = np.minimum(df["open"], df["close"])
        df["high"] = high_base * (1.0 + np.abs(rng.normal(0.0008, 0.0003, len(df))))
        df["low"] = low_base * (1.0 - np.abs(rng.normal(0.0008, 0.0003, len(df))))
        df["volume"] = df["volume"] * (1.0 + rng.normal(0.0, 0.002, len(df)))
        df["source"] = self.name
        return df


@dataclass(slots=True)
class PaidProviderPlaceholder:
    name: str = "paid_provider_placeholder"

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
        raise NotImplementedError("Paid provider is not configured yet")

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError("Paid provider is not configured yet")

    def fetch_news(self, start_ts: datetime, end_ts: datetime, lang: str) -> list[NewsEvent]:
        raise NotImplementedError("Paid provider is not configured yet")

    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]:
        raise NotImplementedError("Paid provider is not configured yet")
