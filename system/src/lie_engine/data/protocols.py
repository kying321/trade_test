from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, TypedDict

import pandas as pd

from lie_engine.models import NewsEvent


class L2Level(TypedDict):
    price: float
    qty: float


class L2Snapshot(TypedDict):
    exchange: str
    symbol: str
    event_ts_ms: int
    recv_ts_ms: int
    seq: int
    prev_seq: int
    bids: list[L2Level]
    asks: list[L2Level]


class TradeTick(TypedDict):
    exchange: str
    symbol: str
    trade_id: str
    event_ts_ms: int
    recv_ts_ms: int
    price: float
    qty: float
    side: str


class DataProviderProtocol(Protocol):
    name: str

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str) -> pd.DataFrame: ...

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame: ...

    def fetch_news(self, start_ts: datetime, end_ts: datetime, lang: str) -> list[NewsEvent]: ...

    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]: ...

    def fetch_l2(self, symbol: str, start_ts: datetime, end_ts: datetime, depth: int = 20) -> pd.DataFrame: ...

    def fetch_trades(self, symbol: str, start_ts: datetime, end_ts: datetime, limit: int = 2000) -> pd.DataFrame: ...
