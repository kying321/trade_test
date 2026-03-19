from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
import hashlib
import json
import re
import ssl
import threading
import time
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError, URLError

try:
    import certifi
except ImportError:  # pragma: no cover - fallback for minimal environments
    certifi = None

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


def _empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])


def _empty_l2_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["exchange", "symbol", "event_ts_ms", "recv_ts_ms", "seq", "prev_seq", "bids", "asks", "source"])


def _empty_trades_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["exchange", "symbol", "trade_id", "event_ts_ms", "recv_ts_ms", "price", "qty", "side", "source"])


def _as_utc_epoch_ms(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return int(ts.timestamp() * 1000)


def _date_start_ms(d: date) -> int:
    return _as_utc_epoch_ms(datetime.combine(d, dt_time(0, 0, 0), tzinfo=timezone.utc))


def _date_end_ms(d: date) -> int:
    return _as_utc_epoch_ms(datetime.combine(d, dt_time(23, 59, 59), tzinfo=timezone.utc))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _looks_like_crypto_pair(symbol: str) -> bool:
    sym = re.sub(r"[^A-Za-z0-9]", "", str(symbol or "").upper())
    if len(sym) < 6:
        return False
    if not re.search(r"[A-Z]", sym):
        return False
    quote_assets = ("USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH", "BNB", "EUR")
    return any(sym.endswith(q) and len(sym) > len(q) + 1 for q in quote_assets)


def _end_ms_with_clock_skew(end_ms: int, *, request_timeout_ms: int) -> int:
    # Exchanges may emit event_ts slightly ahead of local wall clock.
    # Keep a bounded slack so live sampling does not drop valid trades.
    slack_ms = max(2_000, min(180_000, int(max(1, request_timeout_ms)) * 20))
    return int(end_ms + slack_ms)


class _TokenBucket:
    def __init__(self, *, capacity: float, refill_per_second: float) -> None:
        self.capacity = float(max(1.0, capacity))
        self.refill_per_second = float(max(1e-9, refill_per_second))
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, *, cost: float = 1.0, max_wait_seconds: float = 30.0) -> bool:
        need = float(max(0.0, cost))
        deadline = time.monotonic() + float(max(0.0, max_wait_seconds))
        while True:
            now = time.monotonic()
            with self._lock:
                elapsed = max(0.0, now - self.last_refill)
                if elapsed > 0.0:
                    refill = elapsed * self.refill_per_second
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.last_refill = now
                if self.tokens >= need:
                    self.tokens -= need
                    return True
                deficit = max(0.0, need - self.tokens)
                wait_seconds = deficit / self.refill_per_second
            if now >= deadline:
                return False
            time.sleep(min(0.20, max(0.01, wait_seconds)))


def _public_ssl_context() -> ssl.SSLContext:
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def _public_https_opener(*, ctx: ssl.SSLContext, bypass_env_proxy: bool):
    handlers: list[Any] = []
    if bypass_env_proxy:
        handlers.append(request.ProxyHandler({}))
    handlers.extend([request.HTTPHandler(), request.HTTPSHandler(context=ctx)])
    return request.build_opener(*handlers)


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
            return _empty_ohlcv_frame()

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

    def fetch_l2(self, symbol: str, start_ts: datetime, end_ts: datetime, depth: int = 20) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not provide native L2 stream yet: {symbol}")

    def fetch_trades(self, symbol: str, start_ts: datetime, end_ts: datetime, limit: int = 2000) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not provide native trade ticks yet: {symbol}")


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

    def fetch_l2(self, symbol: str, start_ts: datetime, end_ts: datetime, depth: int = 20) -> pd.DataFrame:
        raise NotImplementedError("Paid provider is not configured yet")

    def fetch_trades(self, symbol: str, start_ts: datetime, end_ts: datetime, limit: int = 2000) -> pd.DataFrame:
        raise NotImplementedError("Paid provider is not configured yet")


@dataclass(slots=True)
class BinanceSpotPublicProvider:
    # REST base for Binance spot public market data.
    name: str = "binance_spot_public"
    base_url: str = "https://api.binance.com"
    request_timeout_ms: int = 5000
    rate_limit_per_minute: int = 10
    rate_limit_wait_seconds: float = 30.0
    allow_insecure_ssl_fallback: bool = True
    bypass_env_proxy: bool = True
    user_agent: str = "lie-engine/0.1"
    _bucket: _TokenBucket = field(init=False, repr=False)
    _timeout_seconds: float = field(init=False, repr=False)
    _ssl_context: ssl.SSLContext = field(init=False, repr=False)
    _last_l2_error: str = field(init=False, repr=False, default="")
    _last_trade_error: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        cap = float(max(1, int(self.rate_limit_per_minute)))
        self._bucket = _TokenBucket(capacity=cap, refill_per_second=cap / 60.0)
        self._timeout_seconds = min(5.0, max(0.1, float(self.request_timeout_ms) / 1000.0))
        self._ssl_context = _public_ssl_context()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        txt = re.sub(r"[^A-Za-z0-9]", "", str(symbol or "").upper())
        return txt

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        mapping = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "2h": 7_200_000,
            "4h": 14_400_000,
            "6h": 21_600_000,
            "8h": 28_800_000,
            "12h": 43_200_000,
            "1d": 86_400_000,
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported Binance kline interval: {interval}")
        return mapping[interval]

    @staticmethod
    def _nearest_depth(limit: int) -> int:
        allowed = [5, 10, 20, 50, 100, 500, 1000, 5000]
        lim = int(max(1, limit))
        for d in allowed:
            if lim <= d:
                return d
        return 5000

    def _http_get_json(self, path: str, params: dict[str, Any]) -> Any:
        if not self._bucket.acquire(cost=1.0, max_wait_seconds=float(self.rate_limit_wait_seconds)):
            raise RuntimeError("token_bucket_acquire_timeout")
        query = parse.urlencode(params, doseq=True)
        url = f"{self.base_url.rstrip('/')}{path}"
        if query:
            url = f"{url}?{query}"
        req = request.Request(
            url=url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        def _open_with_ctx(ctx: ssl.SSLContext) -> tuple[int, str]:
            opener = _public_https_opener(ctx=ctx, bypass_env_proxy=bool(self.bypass_env_proxy))
            with opener.open(req, timeout=self._timeout_seconds) as resp:
                status_raw = getattr(resp, "status", None)
                if status_raw is None:
                    status_raw = resp.getcode()
                status = int(status_raw)
                payload = resp.read().decode("utf-8")
            return status, payload

        try:
            status, payload = _open_with_ctx(self._ssl_context)
        except HTTPError as exc:
            raise RuntimeError(f"http_error:{exc.code}") from exc
        except URLError as exc:
            msg = str(exc.reason) if getattr(exc, "reason", None) is not None else str(exc)
            if self.allow_insecure_ssl_fallback and "CERTIFICATE_VERIFY_FAILED" in msg.upper():
                try:
                    status, payload = _open_with_ctx(ssl._create_unverified_context())
                except Exception as fallback_exc:  # noqa: BLE001
                    raise RuntimeError(f"url_error:{msg}") from fallback_exc
            else:
                raise RuntimeError(f"url_error:{msg}") from exc

        if status >= 400:
            raise RuntimeError(f"http_error:{status}")
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid_json_payload") from exc

    def _http_get_json_timed(self, path: str, params: dict[str, Any]) -> tuple[Any, int, int]:
        if not self._bucket.acquire(cost=1.0, max_wait_seconds=float(self.rate_limit_wait_seconds)):
            raise RuntimeError("token_bucket_acquire_timeout")
        query = parse.urlencode(params, doseq=True)
        url = f"{self.base_url.rstrip('/')}{path}"
        if query:
            url = f"{url}?{query}"
        req = request.Request(
            url=url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )

        def _open_with_ctx(ctx: ssl.SSLContext) -> tuple[int, str, int, int]:
            send_ms = int(time.time() * 1000)
            opener = _public_https_opener(ctx=ctx, bypass_env_proxy=bool(self.bypass_env_proxy))
            with opener.open(req, timeout=self._timeout_seconds) as resp:
                status_raw = getattr(resp, "status", None)
                if status_raw is None:
                    status_raw = resp.getcode()
                status = int(status_raw)
                payload = resp.read().decode("utf-8")
            recv_ms = int(time.time() * 1000)
            return status, payload, send_ms, recv_ms

        try:
            status, payload, send_ms, recv_ms = _open_with_ctx(self._ssl_context)
        except HTTPError as exc:
            raise RuntimeError(f"http_error:{exc.code}") from exc
        except URLError as exc:
            msg = str(exc.reason) if getattr(exc, "reason", None) is not None else str(exc)
            if self.allow_insecure_ssl_fallback and "CERTIFICATE_VERIFY_FAILED" in msg.upper():
                try:
                    status, payload, send_ms, recv_ms = _open_with_ctx(ssl._create_unverified_context())
                except Exception as fallback_exc:  # noqa: BLE001
                    raise RuntimeError(f"url_error:{msg}") from fallback_exc
            else:
                raise RuntimeError(f"url_error:{msg}") from exc

        if status >= 400:
            raise RuntimeError(f"http_error:{status}")
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid_json_payload") from exc
        return parsed, int(send_ms), int(recv_ms)

    def fetch_time_sync_sample(self) -> dict[str, Any]:
        payload, local_send_ms, local_recv_ms = self._http_get_json_timed("/api/v3/time", params={})
        if not isinstance(payload, dict):
            raise RuntimeError("invalid_time_sync_payload")
        server_ms = int(payload.get("serverTime", 0))
        if server_ms <= 0:
            raise RuntimeError("invalid_server_time_ms")
        local_mid_ms = int((local_send_ms + local_recv_ms) // 2)
        rtt_ms = int(max(0, local_recv_ms - local_send_ms))
        offset_ms = int(server_ms - local_mid_ms)
        return {
            "source": self.name,
            "server_ts_ms": int(server_ms),
            "local_send_ts_ms": int(local_send_ms),
            "local_recv_ts_ms": int(local_recv_ms),
            "local_mid_ts_ms": int(local_mid_ms),
            "rtt_ms": int(rtt_ms),
            "offset_ms": int(offset_ms),
            "offset_abs_ms": int(abs(offset_ms)),
        }

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
        sym = self._normalize_symbol(symbol)
        if (not sym) or (not _looks_like_crypto_pair(sym)):
            return _empty_ohlcv_frame()
        try:
            interval_ms = self._interval_to_ms(freq)
        except ValueError:
            return _empty_ohlcv_frame()

        start_ms = _date_start_ms(start)
        end_ms = _date_end_ms(end)
        if end_ms < start_ms:
            return _empty_ohlcv_frame()

        cursor = int(start_ms)
        rows: list[dict[str, Any]] = []
        max_pages = 24
        for _ in range(max_pages):
            if cursor > end_ms:
                break
            params = {
                "symbol": sym,
                "interval": freq,
                "startTime": int(cursor),
                "endTime": int(end_ms),
                "limit": 1000,
            }
            try:
                payload = self._http_get_json("/api/v3/klines", params=params)
            except Exception:
                break
            if not isinstance(payload, list) or not payload:
                break

            for item in payload:
                if not isinstance(item, list) or len(item) < 6:
                    continue
                ts_ms = int(item[0])
                rows.append(
                    {
                        "ts": pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(None),
                        "symbol": sym,
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "source": self.name,
                        "asset_class": "crypto",
                    }
                )
            last_open_ms = int(payload[-1][0])
            next_cursor = int(last_open_ms + interval_ms)
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        if not rows:
            return _empty_ohlcv_frame()
        out = pd.DataFrame(rows)
        out = out.drop_duplicates(subset=["ts", "symbol"]).sort_values("ts").reset_index(drop=True)
        return out

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        idx = pd.date_range(start=start, end=end, freq="MS")
        if len(idx) == 0:
            idx = pd.DatetimeIndex([pd.Timestamp(start)])
        return pd.DataFrame(
            {
                "date": idx,
                "cpi_yoy": np.zeros(len(idx)),
                "ppi_yoy": np.zeros(len(idx)),
                "lpr_1y": np.full(len(idx), 3.45),
                "source": self.name,
            }
        )

    def fetch_news(self, start_ts: datetime, end_ts: datetime, lang: str) -> list[NewsEvent]:
        return []

    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]:
        end_ms = _date_end_ms(as_of)
        start_ms = max(0, end_ms - 24 * 3600 * 1000)
        ret_24h = 0.0
        rv_24h = 0.0
        iv_proxy = 0.20
        funding = 0.0
        spread_bps = 0.0

        try:
            k = self._http_get_json(
                "/api/v3/klines",
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1h",
                    "startTime": int(start_ms),
                    "endTime": int(end_ms),
                    "limit": 1000,
                },
            )
            if isinstance(k, list) and len(k) >= 2:
                closes = [_safe_float(row[4], 0.0) for row in k if isinstance(row, list) and len(row) >= 5]
                closes = [x for x in closes if x > 0.0]
                if len(closes) >= 2:
                    ret_24h = float(closes[-1] / max(1e-9, closes[0]) - 1.0)
                    arr = pd.Series(closes, dtype=float)
                    logret = np.log(arr / arr.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
                    if not logret.empty:
                        # 1h log-return realized volatility projected to 24h.
                        rv_24h = float(max(0.0, logret.std(ddof=0) * np.sqrt(24.0)))
                        iv_proxy = float(min(2.0, max(0.08, rv_24h * 1.2)))
        except Exception:
            pass

        try:
            fr = self._http_get_json(
                "/fapi/v1/fundingRate",
                params={
                    "symbol": "BTCUSDT",
                    "startTime": int(start_ms),
                    "endTime": int(end_ms),
                    "limit": 200,
                },
            )
            if isinstance(fr, list) and fr:
                vals = [_safe_float(x.get("fundingRate"), 0.0) for x in fr if isinstance(x, dict)]
                vals = [x for x in vals if np.isfinite(x)]
                if vals:
                    funding = float(vals[-1])
        except Exception:
            pass

        try:
            depth = self._http_get_json("/api/v3/depth", params={"symbol": "BTCUSDT", "limit": 5})
            if isinstance(depth, dict):
                bids = depth.get("bids", []) if isinstance(depth.get("bids", []), list) else []
                asks = depth.get("asks", []) if isinstance(depth.get("asks", []), list) else []
                if bids and asks and isinstance(bids[0], list) and isinstance(asks[0], list):
                    bid = _safe_float(bids[0][0], 0.0)
                    ask = _safe_float(asks[0][0], 0.0)
                    mid = max(1e-9, 0.5 * (bid + ask))
                    spread_bps = float(max(0.0, (ask - bid) / mid * 10000.0))
        except Exception:
            pass

        # Keep legacy keys for compatibility, append crypto-native factors.
        return {
            "pcr_50etf": 1.0 + min(0.8, max(0.0, spread_bps / 40.0) + max(0.0, abs(funding) * 1200.0)),
            "iv_50etf": float(min(1.2, max(0.08, iv_proxy))),
            "northbound_netflow": float(ret_24h * 1e10),
            "margin_balance_chg": float(np.tanh(ret_24h * 6.0) * 0.03),
            "btc_return_24h": float(ret_24h),
            "btc_realized_vol_24h": float(rv_24h),
            "btc_iv_proxy_24h": float(iv_proxy),
            "btc_funding_rate_8h": float(funding),
            "btc_funding_abs_8h": float(abs(funding)),
            "btc_book_spread_bps": float(spread_bps),
        }

    def fetch_l2(self, symbol: str, start_ts: datetime, end_ts: datetime, depth: int = 20) -> pd.DataFrame:
        self._last_l2_error = ""
        sym = self._normalize_symbol(symbol)
        if (not sym) or (not _looks_like_crypto_pair(sym)):
            return _empty_l2_frame()
        limit = self._nearest_depth(depth)
        try:
            payload = self._http_get_json("/api/v3/depth", params={"symbol": sym, "limit": limit})
        except Exception as exc:
            self._last_l2_error = str(exc).strip() or type(exc).__name__
            return _empty_l2_frame()
        if not isinstance(payload, dict):
            self._last_l2_error = "invalid_payload_type"
            return _empty_l2_frame()

        recv_ts_ms = int(time.time() * 1000)
        seq = int(payload.get("lastUpdateId", 0))
        bids_raw = payload.get("bids", []) if isinstance(payload.get("bids", []), list) else []
        asks_raw = payload.get("asks", []) if isinstance(payload.get("asks", []), list) else []
        bids = [{"price": float(x[0]), "qty": float(x[1])} for x in bids_raw if isinstance(x, list) and len(x) >= 2]
        asks = [{"price": float(x[0]), "qty": float(x[1])} for x in asks_raw if isinstance(x, list) and len(x) >= 2]
        return pd.DataFrame(
            [
                {
                    "exchange": "binance_spot",
                    "symbol": sym,
                    "event_ts_ms": recv_ts_ms,
                    "recv_ts_ms": recv_ts_ms,
                    "seq": seq,
                    "prev_seq": max(0, seq - 1),
                    "bids": bids,
                    "asks": asks,
                    "source": self.name,
                }
            ]
        )

    def fetch_trades(self, symbol: str, start_ts: datetime, end_ts: datetime, limit: int = 2000) -> pd.DataFrame:
        self._last_trade_error = ""
        sym = self._normalize_symbol(symbol)
        if (not sym) or (not _looks_like_crypto_pair(sym)):
            return _empty_trades_frame()
        lim = int(max(1, min(1000, limit)))
        start_ms = _as_utc_epoch_ms(start_ts)
        end_ms = _as_utc_epoch_ms(end_ts)
        end_ms_bound = _end_ms_with_clock_skew(end_ms, request_timeout_ms=self.request_timeout_ms)
        now_ms = int(time.time() * 1000)

        def _parse_rows(raw: Any) -> list[dict[str, Any]]:
            if not isinstance(raw, list):
                return []
            parsed: list[dict[str, Any]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                event_ts_ms = int(item.get("T", 0))
                side = "SELL" if bool(item.get("m", False)) else "BUY"
                parsed.append(
                    {
                        "exchange": "binance_spot",
                        "symbol": sym,
                        "trade_id": str(item.get("a", "")),
                        "event_ts_ms": event_ts_ms,
                        "recv_ts_ms": event_ts_ms,
                        "price": float(item.get("p", 0.0)),
                        "qty": float(item.get("q", 0.0)),
                        "side": side,
                        "source": self.name,
                    }
                )
            return parsed

        rows: list[dict[str, Any]] = []
        recent_window_ms = max(1_000, end_ms - start_ms)
        if end_ms >= now_ms - 60_000 and recent_window_ms <= 3_600_000:
            try:
                recent_payload = self._http_get_json("/api/v3/aggTrades", params={"symbol": sym, "limit": lim})
            except Exception as exc:
                self._last_trade_error = str(exc).strip() or type(exc).__name__
                recent_payload = []
            recent_rows = _parse_rows(recent_payload)
            if recent_rows:
                rows = [x for x in recent_rows if start_ms <= int(x.get("event_ts_ms", 0)) <= end_ms_bound]

        if not rows:
            params = {
                "symbol": sym,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": lim,
            }
            try:
                payload = self._http_get_json("/api/v3/aggTrades", params=params)
            except Exception as exc:
                self._last_trade_error = str(exc).strip() or type(exc).__name__
                return _empty_trades_frame()
            rows = _parse_rows(payload)

        if start_ms <= end_ms_bound:
            rows = [x for x in rows if start_ms <= int(x.get("event_ts_ms", 0)) <= end_ms_bound]
        if not rows:
            return _empty_trades_frame()
        return pd.DataFrame(rows).sort_values(["event_ts_ms", "trade_id"]).reset_index(drop=True)


@dataclass(slots=True)
class BybitSpotPublicProvider:
    name: str = "bybit_spot_public"
    base_url: str = "https://api.bybit.com"
    request_timeout_ms: int = 5000
    rate_limit_per_minute: int = 10
    rate_limit_wait_seconds: float = 30.0
    allow_insecure_ssl_fallback: bool = True
    bypass_env_proxy: bool = True
    user_agent: str = "lie-engine/0.1"
    _bucket: _TokenBucket = field(init=False, repr=False)
    _timeout_seconds: float = field(init=False, repr=False)
    _ssl_context: ssl.SSLContext = field(init=False, repr=False)
    _last_l2_error: str = field(init=False, repr=False, default="")
    _last_trade_error: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        cap = float(max(1, int(self.rate_limit_per_minute)))
        self._bucket = _TokenBucket(capacity=cap, refill_per_second=cap / 60.0)
        self._timeout_seconds = min(5.0, max(0.1, float(self.request_timeout_ms) / 1000.0))
        self._ssl_context = _public_ssl_context()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        txt = re.sub(r"[^A-Za-z0-9]", "", str(symbol or "").upper())
        return txt

    def _http_get_json(self, path: str, params: dict[str, Any]) -> Any:
        if not self._bucket.acquire(cost=1.0, max_wait_seconds=float(self.rate_limit_wait_seconds)):
            raise RuntimeError("token_bucket_acquire_timeout")
        query = parse.urlencode(params, doseq=True)
        url = f"{self.base_url.rstrip('/')}{path}"
        if query:
            url = f"{url}?{query}"
        req = request.Request(
            url=url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )

        def _open_with_ctx(ctx: ssl.SSLContext) -> tuple[int, str]:
            opener = _public_https_opener(ctx=ctx, bypass_env_proxy=bool(self.bypass_env_proxy))
            with opener.open(req, timeout=self._timeout_seconds) as resp:
                status_raw = getattr(resp, "status", None)
                if status_raw is None:
                    status_raw = resp.getcode()
                status = int(status_raw)
                payload = resp.read().decode("utf-8")
            return status, payload

        try:
            status, payload = _open_with_ctx(self._ssl_context)
        except HTTPError as exc:
            raise RuntimeError(f"http_error:{exc.code}") from exc
        except URLError as exc:
            msg = str(exc.reason) if getattr(exc, "reason", None) is not None else str(exc)
            if self.allow_insecure_ssl_fallback and "CERTIFICATE_VERIFY_FAILED" in msg.upper():
                try:
                    status, payload = _open_with_ctx(ssl._create_unverified_context())
                except Exception as fallback_exc:  # noqa: BLE001
                    raise RuntimeError(f"url_error:{msg}") from fallback_exc
            else:
                raise RuntimeError(f"url_error:{msg}") from exc

        if status >= 400:
            raise RuntimeError(f"http_error:{status}")
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid_json_payload") from exc

    def _http_get_json_timed(self, path: str, params: dict[str, Any]) -> tuple[Any, int, int]:
        if not self._bucket.acquire(cost=1.0, max_wait_seconds=float(self.rate_limit_wait_seconds)):
            raise RuntimeError("token_bucket_acquire_timeout")
        query = parse.urlencode(params, doseq=True)
        url = f"{self.base_url.rstrip('/')}{path}"
        if query:
            url = f"{url}?{query}"
        req = request.Request(
            url=url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )

        def _open_with_ctx(ctx: ssl.SSLContext) -> tuple[int, str, int, int]:
            send_ms = int(time.time() * 1000)
            opener = _public_https_opener(ctx=ctx, bypass_env_proxy=bool(self.bypass_env_proxy))
            with opener.open(req, timeout=self._timeout_seconds) as resp:
                status_raw = getattr(resp, "status", None)
                if status_raw is None:
                    status_raw = resp.getcode()
                status = int(status_raw)
                payload = resp.read().decode("utf-8")
            recv_ms = int(time.time() * 1000)
            return status, payload, send_ms, recv_ms

        try:
            status, payload, send_ms, recv_ms = _open_with_ctx(self._ssl_context)
        except HTTPError as exc:
            raise RuntimeError(f"http_error:{exc.code}") from exc
        except URLError as exc:
            msg = str(exc.reason) if getattr(exc, "reason", None) is not None else str(exc)
            if self.allow_insecure_ssl_fallback and "CERTIFICATE_VERIFY_FAILED" in msg.upper():
                try:
                    status, payload, send_ms, recv_ms = _open_with_ctx(ssl._create_unverified_context())
                except Exception as fallback_exc:  # noqa: BLE001
                    raise RuntimeError(f"url_error:{msg}") from fallback_exc
            else:
                raise RuntimeError(f"url_error:{msg}") from exc

        if status >= 400:
            raise RuntimeError(f"http_error:{status}")
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid_json_payload") from exc
        return parsed, int(send_ms), int(recv_ms)

    def fetch_time_sync_sample(self) -> dict[str, Any]:
        payload, local_send_ms, local_recv_ms = self._http_get_json_timed("/v5/market/time", params={})
        if not isinstance(payload, dict) or int(payload.get("retCode", 1)) != 0:
            raise RuntimeError("invalid_time_sync_payload")
        result = payload.get("result", {})
        server_ms = 0
        if isinstance(result, dict):
            nano = str(result.get("timeNano", "")).strip()
            sec = str(result.get("timeSecond", "")).strip()
            if nano.isdigit():
                server_ms = int(int(nano) // 1_000_000)
            elif sec.isdigit():
                server_ms = int(sec) * 1000
        if server_ms <= 0:
            top_level_time = str(payload.get("time", "")).strip()
            if top_level_time.isdigit():
                server_ms = int(top_level_time)
        if server_ms <= 0:
            raise RuntimeError("invalid_server_time_ms")
        local_mid_ms = int((local_send_ms + local_recv_ms) // 2)
        rtt_ms = int(max(0, local_recv_ms - local_send_ms))
        offset_ms = int(server_ms - local_mid_ms)
        return {
            "source": self.name,
            "server_ts_ms": int(server_ms),
            "local_send_ts_ms": int(local_send_ms),
            "local_recv_ts_ms": int(local_recv_ms),
            "local_mid_ts_ms": int(local_mid_ms),
            "rtt_ms": int(rtt_ms),
            "offset_ms": int(offset_ms),
            "offset_abs_ms": int(abs(offset_ms)),
        }

    @staticmethod
    def _limit_depth(limit: int) -> int:
        return int(max(1, min(200, int(limit))))

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
        if str(freq).strip() != "1d":
            return _empty_ohlcv_frame()
        sym = self._normalize_symbol(symbol)
        if (not sym) or (not _looks_like_crypto_pair(sym)):
            return _empty_ohlcv_frame()
        params = {
            "category": "spot",
            "symbol": sym,
            "interval": "D",
            "start": _date_start_ms(start),
            "end": _date_end_ms(end),
            "limit": 1000,
        }
        try:
            payload = self._http_get_json("/v5/market/kline", params=params)
        except Exception:
            return _empty_ohlcv_frame()
        if not isinstance(payload, dict) or int(payload.get("retCode", 1)) != 0:
            return _empty_ohlcv_frame()
        result = payload.get("result", {})
        rows = result.get("list", []) if isinstance(result, dict) else []
        if not isinstance(rows, list) or not rows:
            return _empty_ohlcv_frame()

        out_rows: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, list) or len(item) < 6:
                continue
            ts_ms = int(item[0])
            out_rows.append(
                {
                    "ts": pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(None),
                    "symbol": sym,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "source": self.name,
                    "asset_class": "crypto",
                }
            )
        if not out_rows:
            return _empty_ohlcv_frame()
        out = pd.DataFrame(out_rows)
        out = out.sort_values("ts").drop_duplicates(subset=["ts", "symbol"]).reset_index(drop=True)
        return out

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        idx = pd.date_range(start=start, end=end, freq="MS")
        if len(idx) == 0:
            idx = pd.DatetimeIndex([pd.Timestamp(start)])
        return pd.DataFrame(
            {
                "date": idx,
                "cpi_yoy": np.zeros(len(idx)),
                "ppi_yoy": np.zeros(len(idx)),
                "lpr_1y": np.full(len(idx), 3.45),
                "source": self.name,
            }
        )

    def fetch_news(self, start_ts: datetime, end_ts: datetime, lang: str) -> list[NewsEvent]:
        return []

    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]:
        end_ms = _date_end_ms(as_of)
        start_ms = max(0, end_ms - 24 * 3600 * 1000)
        ret_24h = 0.0
        rv_24h = 0.0
        iv_proxy = 0.20
        funding = 0.0
        spread_bps = 0.0

        try:
            payload = self._http_get_json(
                "/v5/market/kline",
                params={
                    "category": "spot",
                    "symbol": "BTCUSDT",
                    "interval": "60",
                    "start": int(start_ms),
                    "end": int(end_ms),
                    "limit": 1000,
                },
            )
            if isinstance(payload, dict) and int(payload.get("retCode", 1)) == 0:
                rows = payload.get("result", {}).get("list", [])
                if isinstance(rows, list) and rows:
                    closes = [_safe_float(r[4], 0.0) for r in rows if isinstance(r, list) and len(r) >= 5]
                    closes = [x for x in closes if x > 0.0]
                    if len(closes) >= 2:
                        closes = list(reversed(closes))
                        ret_24h = float(closes[-1] / max(1e-9, closes[0]) - 1.0)
                        arr = pd.Series(closes, dtype=float)
                        logret = np.log(arr / arr.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
                        if not logret.empty:
                            rv_24h = float(max(0.0, logret.std(ddof=0) * np.sqrt(24.0)))
                            iv_proxy = float(min(2.0, max(0.08, rv_24h * 1.2)))
        except Exception:
            pass

        try:
            payload = self._http_get_json(
                "/v5/market/tickers",
                params={"category": "linear", "symbol": "BTCUSDT"},
            )
            if isinstance(payload, dict) and int(payload.get("retCode", 1)) == 0:
                rows = payload.get("result", {}).get("list", [])
                if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                    funding = _safe_float(rows[0].get("fundingRate"), 0.0)
        except Exception:
            pass

        try:
            payload = self._http_get_json(
                "/v5/market/orderbook",
                params={"category": "spot", "symbol": "BTCUSDT", "limit": 5},
            )
            if isinstance(payload, dict) and int(payload.get("retCode", 1)) == 0:
                result = payload.get("result", {})
                if isinstance(result, dict):
                    bids = result.get("b", []) if isinstance(result.get("b", []), list) else []
                    asks = result.get("a", []) if isinstance(result.get("a", []), list) else []
                    if bids and asks and isinstance(bids[0], list) and isinstance(asks[0], list):
                        bid = _safe_float(bids[0][0], 0.0)
                        ask = _safe_float(asks[0][0], 0.0)
                        mid = max(1e-9, 0.5 * (bid + ask))
                        spread_bps = float(max(0.0, (ask - bid) / mid * 10000.0))
        except Exception:
            pass

        return {
            "pcr_50etf": 1.0 + min(0.8, max(0.0, spread_bps / 40.0) + max(0.0, abs(funding) * 1200.0)),
            "iv_50etf": float(min(1.2, max(0.08, iv_proxy))),
            "northbound_netflow": float(ret_24h * 1e10),
            "margin_balance_chg": float(np.tanh(ret_24h * 6.0) * 0.03),
            "btc_return_24h": float(ret_24h),
            "btc_realized_vol_24h": float(rv_24h),
            "btc_iv_proxy_24h": float(iv_proxy),
            "btc_funding_rate_8h": float(funding),
            "btc_funding_abs_8h": float(abs(funding)),
            "btc_book_spread_bps": float(spread_bps),
        }

    def fetch_l2(self, symbol: str, start_ts: datetime, end_ts: datetime, depth: int = 20) -> pd.DataFrame:
        self._last_l2_error = ""
        sym = self._normalize_symbol(symbol)
        if (not sym) or (not _looks_like_crypto_pair(sym)):
            return _empty_l2_frame()
        params = {
            "category": "spot",
            "symbol": sym,
            "limit": self._limit_depth(depth),
        }
        try:
            payload = self._http_get_json("/v5/market/orderbook", params=params)
        except Exception as exc:
            self._last_l2_error = str(exc).strip() or type(exc).__name__
            return _empty_l2_frame()
        if not isinstance(payload, dict) or int(payload.get("retCode", 1)) != 0:
            self._last_l2_error = "invalid_payload"
            return _empty_l2_frame()
        result = payload.get("result", {})
        if not isinstance(result, dict):
            self._last_l2_error = "invalid_result_payload"
            return _empty_l2_frame()

        recv_ts_ms = int(time.time() * 1000)
        event_ts_ms = int(result.get("ts", recv_ts_ms))
        seq = int(result.get("seq", result.get("u", 0)))
        bids_raw = result.get("b", []) if isinstance(result.get("b", []), list) else []
        asks_raw = result.get("a", []) if isinstance(result.get("a", []), list) else []
        bids = [{"price": float(x[0]), "qty": float(x[1])} for x in bids_raw if isinstance(x, list) and len(x) >= 2]
        asks = [{"price": float(x[0]), "qty": float(x[1])} for x in asks_raw if isinstance(x, list) and len(x) >= 2]
        return pd.DataFrame(
            [
                {
                    "exchange": "bybit_spot",
                    "symbol": sym,
                    "event_ts_ms": event_ts_ms,
                    "recv_ts_ms": recv_ts_ms,
                    "seq": seq,
                    "prev_seq": max(0, seq - 1),
                    "bids": bids,
                    "asks": asks,
                    "source": self.name,
                }
            ]
        )

    def fetch_trades(self, symbol: str, start_ts: datetime, end_ts: datetime, limit: int = 2000) -> pd.DataFrame:
        self._last_trade_error = ""
        sym = self._normalize_symbol(symbol)
        if (not sym) or (not _looks_like_crypto_pair(sym)):
            return _empty_trades_frame()
        start_ms = _as_utc_epoch_ms(start_ts)
        end_ms = _as_utc_epoch_ms(end_ts)
        end_ms_bound = _end_ms_with_clock_skew(end_ms, request_timeout_ms=self.request_timeout_ms)
        params = {
            "category": "spot",
            "symbol": sym,
            "limit": int(max(1, min(1000, limit))),
        }
        try:
            payload = self._http_get_json("/v5/market/recent-trade", params=params)
        except Exception as exc:
            self._last_trade_error = str(exc).strip() or type(exc).__name__
            return _empty_trades_frame()
        if not isinstance(payload, dict) or int(payload.get("retCode", 1)) != 0:
            self._last_trade_error = "invalid_payload"
            return _empty_trades_frame()
        result = payload.get("result", {})
        rows = result.get("list", []) if isinstance(result, dict) else []
        if not isinstance(rows, list):
            self._last_trade_error = "invalid_result_payload"
            return _empty_trades_frame()

        out: list[dict[str, Any]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            event_ts_ms = int(item.get("time", 0))
            if event_ts_ms < start_ms or event_ts_ms > end_ms_bound:
                continue
            side_raw = str(item.get("side", "")).strip().upper()
            side = "BUY" if side_raw == "BUY" else "SELL" if side_raw == "SELL" else ""
            if not side:
                continue
            out.append(
                {
                    "exchange": "bybit_spot",
                    "symbol": sym,
                    "trade_id": str(item.get("execId", "")),
                    "event_ts_ms": event_ts_ms,
                    "recv_ts_ms": event_ts_ms,
                    "price": float(item.get("price", 0.0)),
                    "qty": float(item.get("size", 0.0)),
                    "side": side,
                    "source": self.name,
                }
            )
        if not out:
            return _empty_trades_frame()
        return pd.DataFrame(out).sort_values(["event_ts_ms", "trade_id"]).reset_index(drop=True)
