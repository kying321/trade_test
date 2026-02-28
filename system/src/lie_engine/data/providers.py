from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
import hashlib
import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from lie_engine.models import AssetClass, NewsEvent

try:
    import requests
except Exception:  # pragma: no cover - optional dependency fallback
    requests = None


EQUITY_SYMBOL_RE = re.compile(r"^\d{6}$")
FUTURE_SYMBOL_RE = re.compile(r"^([A-Z]{1,3})\d{4}$")
BINANCE_SYMBOL_RE = re.compile(r"^[A-Z]{2,20}(USDT|USD|BUSD|FDUSD|USDC)$")

try:
    import akshare as ak
except Exception:  # pragma: no cover - optional dependency fallback
    ak = None


def _seed(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)


def _asset_class_from_symbol(symbol: str) -> AssetClass:
    token = str(symbol or "").strip().upper()
    if BINANCE_SYMBOL_RE.match(token):
        return AssetClass.CASH
    if symbol.startswith(("LC", "SC", "RB")):
        return AssetClass.FUTURE
    if symbol.startswith("5"):
        return AssetClass.ETF
    return AssetClass.EQUITY


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _to_tushare_ts_code(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    if not EQUITY_SYMBOL_RE.match(token):
        return ""
    suffix = "SH" if token.startswith(("5", "6", "9")) else "SZ"
    return f"{token}.{suffix}"


def _normalize_binance_symbol(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    token = token.replace("/", "").replace("-", "").replace("_", "")
    if BINANCE_SYMBOL_RE.match(token):
        return token
    return ""


def _binance_symbol_candidates(symbol: str) -> list[str]:
    token = _normalize_binance_symbol(symbol)
    if not token:
        return []
    out = [token]
    if token.endswith("USDT"):
        out.append(f"{token[:-4]}USD")
    if token.endswith("USDC"):
        out.append(f"{token[:-4]}USD")
    if token.endswith("FDUSD"):
        out.append(f"{token[:-5]}USD")
    deduped: list[str] = []
    for item in out:
        if item not in deduped:
            deduped.append(item)
    return deduped


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
class TushareBinanceHybridProvider:
    name: str = "tushare_binance_hybrid"
    tushare_token: str = ""
    binance_api_key: str = ""
    binance_endpoints: tuple[str, ...] = ("https://api.binance.com", "https://api.binance.us")
    timeout_sec: float = 10.0
    max_calls_per_minute: int = 10
    fallback: OpenSourcePrimaryProvider = field(
        default_factory=lambda: OpenSourcePrimaryProvider(name="open_source_fallback")
    )
    _call_ts: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.tushare_token:
            self.tushare_token = (
                os.environ.get("TUSHARE_TOKEN")
                or os.environ.get("TS_TOKEN")
                or os.environ.get("TSHARE_TOKEN")
                or ""
            ).strip()
        if not self.binance_api_key:
            self.binance_api_key = (os.environ.get("BINANCE_API_KEY") or "").strip()

    def _throttle(self) -> None:
        if self.max_calls_per_minute <= 0:
            return
        window_sec = 60.0
        while True:
            now = time.monotonic()
            self._call_ts = [x for x in self._call_ts if (now - x) < window_sec]
            if len(self._call_ts) < self.max_calls_per_minute:
                self._call_ts.append(now)
                return
            wait_sec = window_sec - (now - self._call_ts[0]) + 0.01
            time.sleep(max(0.05, wait_sec))

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if requests is None:
            return None
        req_headers = {"User-Agent": "lie-engine/1.0"}
        if headers:
            req_headers.update(headers)
        self._throttle()
        try:
            resp = requests.request(
                method=method,
                url=url,
                params=params,
                json=json_payload,
                headers=req_headers,
                timeout=float(self.timeout_sec),
            )
            if int(resp.status_code) != 200:
                return None
            return resp.json()
        except Exception:
            return None

    @staticmethod
    def _tushare_frame(payload: Any) -> pd.DataFrame:
        if not isinstance(payload, dict):
            return pd.DataFrame()
        if int(payload.get("code", -1)) != 0:
            return pd.DataFrame()
        data = payload.get("data", {})
        if not isinstance(data, dict):
            return pd.DataFrame()
        fields = data.get("fields", [])
        items = data.get("items", [])
        if not isinstance(fields, list) or not isinstance(items, list):
            return pd.DataFrame()
        if not fields:
            return pd.DataFrame()
        if not items:
            return pd.DataFrame(columns=[str(c) for c in fields])
        return pd.DataFrame(items, columns=[str(c) for c in fields])

    def _fetch_tushare_frame(self, *, api_name: str, params: dict[str, Any], fields: str) -> pd.DataFrame:
        if not self.tushare_token:
            return pd.DataFrame()
        payload = self._request_json(
            method="POST",
            url="https://api.waditu.com",
            json_payload={
                "api_name": str(api_name),
                "token": str(self.tushare_token),
                "params": params,
                "fields": str(fields),
            },
        )
        return self._tushare_frame(payload)

    def _fetch_tushare_daily(self, *, symbol: str, start: date, end: date) -> pd.DataFrame:
        ts_code = _to_tushare_ts_code(symbol)
        if not ts_code:
            return _empty_bars()
        apis = ["fund_daily", "daily"] if str(symbol).startswith("5") else ["daily"]
        start_s = start.strftime("%Y%m%d")
        end_s = end.strftime("%Y%m%d")
        for api_name in apis:
            raw = self._fetch_tushare_frame(
                api_name=api_name,
                params={"ts_code": ts_code, "start_date": start_s, "end_date": end_s},
                fields="trade_date,open,high,low,close,vol,amount",
            )
            if raw.empty:
                continue
            trade_col = "trade_date" if "trade_date" in raw.columns else ""
            if not trade_col:
                continue
            volume_col = "vol" if "vol" in raw.columns else "amount"
            if volume_col not in raw.columns:
                continue
            out = pd.DataFrame(
                {
                    "ts": pd.to_datetime(raw[trade_col].astype(str), format="%Y%m%d", errors="coerce"),
                    "symbol": str(symbol),
                    "open": pd.to_numeric(raw.get("open"), errors="coerce"),
                    "high": pd.to_numeric(raw.get("high"), errors="coerce"),
                    "low": pd.to_numeric(raw.get("low"), errors="coerce"),
                    "close": pd.to_numeric(raw.get("close"), errors="coerce"),
                    "volume": pd.to_numeric(raw.get(volume_col), errors="coerce"),
                    "source": f"tushare.{api_name}",
                    "asset_class": _asset_class_from_symbol(symbol).value,
                }
            )
            out = out.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
            if out.empty:
                continue
            out = out[(out["ts"].dt.date >= start) & (out["ts"].dt.date <= end)].sort_values("ts")
            return out.reset_index(drop=True)
        return self._fetch_akshare_equity(symbol=symbol, start=start, end=end)

    def _fetch_akshare_equity(self, *, symbol: str, start: date, end: date) -> pd.DataFrame:
        if ak is None:
            return _empty_bars()
        try:
            raw = ak.stock_zh_a_hist(
                symbol=str(symbol),
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            )
        except Exception:
            return _empty_bars()
        if raw is None or raw.empty:
            return _empty_bars()
        out = pd.DataFrame(
            {
                "ts": pd.to_datetime(raw.get("日期"), errors="coerce"),
                "symbol": str(symbol),
                "open": pd.to_numeric(raw.get("开盘"), errors="coerce"),
                "high": pd.to_numeric(raw.get("最高"), errors="coerce"),
                "low": pd.to_numeric(raw.get("最低"), errors="coerce"),
                "close": pd.to_numeric(raw.get("收盘"), errors="coerce"),
                "volume": pd.to_numeric(raw.get("成交量"), errors="coerce"),
                "source": "akshare.stock_zh_a_hist",
                "asset_class": _asset_class_from_symbol(symbol).value,
            }
        )
        out = out.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
        if out.empty:
            return _empty_bars()
        out = out[(out["ts"].dt.date >= start) & (out["ts"].dt.date <= end)].sort_values("ts")
        return out.reset_index(drop=True)

    def _fetch_akshare_future(self, *, symbol: str, start: date, end: date) -> pd.DataFrame:
        if ak is None:
            return _empty_bars()
        m = FUTURE_SYMBOL_RE.match(str(symbol).strip().upper())
        if not m:
            return _empty_bars()
        cont = f"{m.group(1)}0"
        try:
            raw = ak.futures_zh_daily_sina(symbol=cont)
        except Exception:
            return _empty_bars()
        if raw is None or raw.empty:
            return _empty_bars()
        out = pd.DataFrame(
            {
                "ts": pd.to_datetime(raw.get("date"), errors="coerce"),
                "symbol": str(symbol).strip().upper(),
                "open": pd.to_numeric(raw.get("open"), errors="coerce"),
                "high": pd.to_numeric(raw.get("high"), errors="coerce"),
                "low": pd.to_numeric(raw.get("low"), errors="coerce"),
                "close": pd.to_numeric(raw.get("close"), errors="coerce"),
                "volume": pd.to_numeric(raw.get("volume"), errors="coerce"),
                "source": f"akshare.futures_zh_daily_sina:{cont}",
                "asset_class": AssetClass.FUTURE.value,
            }
        )
        out = out.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
        if out.empty:
            return _empty_bars()
        out = out[(out["ts"].dt.date >= start) & (out["ts"].dt.date <= end)].sort_values("ts")
        return out.reset_index(drop=True)

    def _fetch_binance_klines(self, *, symbol: str, start: date, end: date) -> pd.DataFrame:
        candidates = _binance_symbol_candidates(symbol)
        if not candidates:
            return _empty_bars()
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        end_dt = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC) - timedelta(milliseconds=1)
        resolved_symbol = ""
        ms_1d = 24 * 60 * 60 * 1000
        all_rows: list[list[Any]] = []
        for candidate in candidates:
            cursor_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            rows: list[list[Any]] = []
            guard = 0
            while cursor_ms <= end_ms:
                payload = self._fetch_binance_json(
                    path="/api/v3/klines",
                    params={
                        "symbol": candidate,
                        "interval": "1d",
                        "startTime": cursor_ms,
                        "endTime": end_ms,
                        "limit": 1000,
                    },
                )
                if not isinstance(payload, list) or not payload:
                    break
                chunk = [x for x in payload if isinstance(x, list)]
                if not chunk:
                    break
                rows.extend(chunk)
                last_open_ms = int(chunk[-1][0])
                next_cursor = last_open_ms + ms_1d
                if next_cursor <= cursor_ms:
                    break
                cursor_ms = next_cursor
                guard += 1
                if len(chunk) < 1000 or guard > 20:
                    break
            if rows:
                all_rows = rows
                resolved_symbol = candidate
                break
        if not all_rows:
            return _empty_bars()
        if not resolved_symbol:
            resolved_symbol = candidates[0]

        rows: list[dict[str, Any]] = []
        for row in all_rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            rows.append(
                {
                    "ts": pd.to_datetime(int(row[0]), unit="ms", utc=True).tz_convert(None),
                    "symbol": resolved_symbol,
                    "open": _safe_float(row[1]),
                    "high": _safe_float(row[2]),
                    "low": _safe_float(row[3]),
                    "close": _safe_float(row[4]),
                    "volume": _safe_float(row[5]),
                    "source": "binance.klines",
                    "asset_class": _asset_class_from_symbol(resolved_symbol).value,
                }
            )
        if not rows:
            return _empty_bars()
        out = pd.DataFrame(rows)
        out = out.drop_duplicates(subset=["ts", "symbol"], keep="last")
        out = out[(out["ts"].dt.date >= start) & (out["ts"].dt.date <= end)].sort_values("ts")
        return out.reset_index(drop=True)

    def _fetch_binance_json(self, *, path: str, params: dict[str, Any]) -> Any:
        headers = {"X-MBX-APIKEY": self.binance_api_key} if self.binance_api_key else None
        for base in self.binance_endpoints:
            base_url = str(base).strip().rstrip("/")
            if not base_url:
                continue
            payload = self._request_json(
                method="GET",
                url=f"{base_url}{path}",
                params=params,
                headers=headers,
            )
            if payload is None:
                continue
            return payload
        return None

    def _fetch_binance_24h(self, *, symbol: str) -> dict[str, Any]:
        candidates = _binance_symbol_candidates(symbol)
        if not candidates:
            return {}
        for candidate in candidates:
            payload = self._fetch_binance_json(path="/api/v3/ticker/24hr", params={"symbol": candidate})
            if isinstance(payload, dict) and payload:
                return payload
        return {}

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
        if freq != "1d":
            raise ValueError("Only daily frequency is supported in v1")
        token = str(symbol or "").strip().upper()

        if EQUITY_SYMBOL_RE.match(token):
            ts_df = self._fetch_tushare_daily(symbol=token, start=start, end=end)
            if not ts_df.empty:
                return ts_df

        if FUTURE_SYMBOL_RE.match(token):
            fut_df = self._fetch_akshare_future(symbol=token, start=start, end=end)
            if not fut_df.empty:
                return fut_df

        if BINANCE_SYMBOL_RE.match(token):
            bz_df = self._fetch_binance_klines(symbol=token, start=start, end=end)
            if not bz_df.empty:
                return bz_df

        fallback_df = self.fallback.fetch_ohlcv(symbol=token, start=start, end=end, freq=freq)
        if fallback_df.empty:
            return _empty_bars()
        fallback_df = fallback_df.copy()
        fallback_df["source"] = f"{self.name}.fallback"
        return fallback_df

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        bars = self._fetch_binance_klines(symbol="BTCUSDT", start=start, end=end)
        if bars.empty:
            return self.fallback.fetch_macro(start=start, end=end)
        px = bars[["ts", "close"]].copy().dropna()
        if px.empty:
            return self.fallback.fetch_macro(start=start, end=end)
        px = px.sort_values("ts")
        px["ret_1d"] = px["close"].pct_change().fillna(0.0)
        px["month"] = px["ts"].dt.to_period("M").dt.to_timestamp()
        m_close = px.groupby("month", as_index=False)["close"].last().rename(columns={"month": "date"})
        m_ret = m_close["close"].pct_change().fillna(0.0)
        m_vol = px.groupby("month", as_index=False)["ret_1d"].std(ddof=0).rename(columns={"month": "date", "ret_1d": "vol_1d"})
        merged = m_close.merge(m_vol, on="date", how="left")
        merged["vol_1d"] = merged["vol_1d"].fillna(0.0)
        out = pd.DataFrame(
            {
                "date": merged["date"],
                "cpi_yoy": m_ret * 100.0,
                "ppi_yoy": merged["vol_1d"] * 100.0,
                "lpr_1y": 3.45 + np.clip(-m_ret * 0.25, -0.25, 0.25),
                "source": "binance.macro_proxy",
            }
        )
        if out.empty:
            return self.fallback.fetch_macro(start=start, end=end)
        return out

    def fetch_news(self, start_ts: datetime, end_ts: datetime, lang: str) -> list[NewsEvent]:
        events: list[NewsEvent] = []
        for symbol in ("BTCUSDT", "ETHUSDT"):
            snap = self._fetch_binance_24h(symbol=symbol)
            if not snap:
                continue
            pct = _safe_float(snap.get("priceChangePercent", 0.0))
            quote_vol = _safe_float(snap.get("quoteVolume", 0.0))
            title = f"[{lang}] {symbol} 24h 变动 {pct:.2f}%"
            content = f"quote_volume={quote_vol:.2f} USDT"
            eid = hashlib.md5(f"{self.name}|{symbol}|{end_ts.isoformat()}".encode("utf-8"), usedforsecurity=False).hexdigest()
            events.append(
                NewsEvent(
                    event_id=eid,
                    ts=end_ts,
                    title=title,
                    content=content,
                    lang=lang,
                    source="binance.ticker24h",
                    category="宏观",
                    confidence=0.68,
                    entities=[symbol],
                    importance=0.55 + min(0.30, abs(pct) / 30.0),
                )
            )
        if events:
            return events
        return self.fallback.fetch_news(start_ts=start_ts, end_ts=end_ts, lang=lang)

    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]:
        sentiment = dict(self.fallback.fetch_sentiment_factors(as_of=as_of))
        btc = self._fetch_binance_24h(symbol="BTCUSDT")
        eth = self._fetch_binance_24h(symbol="ETHUSDT")
        if btc:
            sentiment["binance_btc_change_pct"] = _safe_float(btc.get("priceChangePercent", 0.0))
            sentiment["binance_btc_quote_volume"] = _safe_float(btc.get("quoteVolume", 0.0))
        if eth:
            sentiment["binance_eth_change_pct"] = _safe_float(eth.get("priceChangePercent", 0.0))
            sentiment["binance_eth_quote_volume"] = _safe_float(eth.get("quoteVolume", 0.0))

        if self.tushare_token:
            raw = self._fetch_tushare_frame(
                api_name="moneyflow_hsgt",
                params={"trade_date": as_of.strftime("%Y%m%d")},
                fields="trade_date,north_money,south_money",
            )
            if raw.empty:
                raw = self._fetch_tushare_frame(
                    api_name="moneyflow_hsgt",
                    params={
                        "start_date": (as_of - timedelta(days=10)).strftime("%Y%m%d"),
                        "end_date": as_of.strftime("%Y%m%d"),
                    },
                    fields="trade_date,north_money,south_money",
                )
            if not raw.empty and "north_money" in raw.columns:
                north = pd.to_numeric(raw["north_money"], errors="coerce").dropna()
                if not north.empty:
                    sentiment["northbound_netflow"] = float(north.iloc[-1])
            if not raw.empty and "south_money" in raw.columns:
                south = pd.to_numeric(raw["south_money"], errors="coerce").dropna()
                if not south.empty:
                    sentiment["southbound_netflow"] = float(south.iloc[-1])
        return sentiment


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
