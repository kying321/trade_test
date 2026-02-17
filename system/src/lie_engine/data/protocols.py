from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

import pandas as pd

from lie_engine.models import NewsEvent


class DataProviderProtocol(Protocol):
    name: str

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str) -> pd.DataFrame: ...

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame: ...

    def fetch_news(self, start_ts: datetime, end_ts: datetime, lang: str) -> list[NewsEvent]: ...

    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]: ...
