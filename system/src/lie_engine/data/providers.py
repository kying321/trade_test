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

import numpy as np
import pandas as pd
import requests

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


@dataclass(slots=True)
class PublicInternetResearchProvider:
    name: str = "public_macro_news"
    request_timeout_ms: int = 5000
    rate_limit_per_minute: int = 60
    rate_limit_wait_seconds: float = 30.0
    allow_insecure_ssl_fallback: bool = True
    user_agent: str = "lie-engine/0.1"
    max_news_lookback_days: int = 3
    _bucket: _TokenBucket = field(init=False, repr=False)
    _timeout_seconds: float = field(init=False, repr=False)
    _ssl_context: ssl.SSLContext = field(init=False, repr=False)
    _macro_cache: dict[str, pd.DataFrame] = field(init=False, repr=False, default_factory=dict)
    _macro_cache_lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)
    _shmet_cache: tuple[float, pd.DataFrame] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        cap = float(max(1, int(self.rate_limit_per_minute)))
        self._bucket = _TokenBucket(capacity=cap, refill_per_second=cap / 60.0)
        self._timeout_seconds = min(5.0, max(0.1, float(self.request_timeout_ms) / 1000.0))
        self._ssl_context = ssl.create_default_context()

    def _http_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if not self._bucket.acquire(cost=1.0, max_wait_seconds=float(self.rate_limit_wait_seconds)):
            raise RuntimeError("token_bucket_acquire_timeout")

        req_headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if headers:
            req_headers.update(headers)

        request_url = str(url)
        if params:
            query = parse.urlencode(params, doseq=True)
            if query:
                request_url = f"{request_url}?{query}"

        data_bytes = None
        if payload is not None:
            data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        req = request.Request(request_url, data=data_bytes, method=method.upper(), headers=req_headers)

        def _open_with_ctx(ctx: ssl.SSLContext) -> tuple[int, str]:
            with request.urlopen(req, timeout=self._timeout_seconds, context=ctx) as resp:
                status_raw = getattr(resp, "status", None)
                if status_raw is None:
                    status_raw = resp.getcode()
                status = int(status_raw)
                body = resp.read().decode("utf-8")
            return status, body

        try:
            status, body = _open_with_ctx(self._ssl_context)
        except HTTPError as exc:
            raise RuntimeError(f"http_error:{exc.code}") from exc
        except URLError as exc:
            msg = str(exc.reason) if getattr(exc, "reason", None) is not None else str(exc)
            if self.allow_insecure_ssl_fallback and "CERTIFICATE_VERIFY_FAILED" in msg.upper():
                try:
                    status, body = _open_with_ctx(ssl._create_unverified_context())
                except Exception as fallback_exc:  # noqa: BLE001
                    raise RuntimeError(f"url_error:{msg}") from fallback_exc
            else:
                raise RuntimeError(f"url_error:{msg}") from exc

        if status >= 400:
            raise RuntimeError(f"http_error:{status}")
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid_json_payload") from exc

    def _fetch_jin10_indicator(
        self,
        attr_id: str,
        label: str,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"jin10:{attr_id}:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        params = {
            "max_date": "",
            "category": "ec",
            "attr_id": str(attr_id),
            "_": "0",
        }
        frames: list[pd.DataFrame] = []
        headers = {
            "x-app-id": "rU6QIu7JHe2gOUeR",
            "x-csrf-token": "x-csrf-token",
            "x-version": "1.0.0",
        }
        for _ in range(240):
            data_json = self._http_json(
                "GET",
                "https://datacenter-api.jin10.com/reports/list_v2",
                params=params,
                headers=headers,
            )
            values = data_json.get("data", {}).get("values", []) if isinstance(data_json, dict) else []
            if not isinstance(values, list) or not values:
                break
            temp_df = pd.DataFrame(values, columns=["日期", "今值", "预测值", "前值"])
            frames.append(temp_df)
            if oldest_date is not None:
                min_dt = pd.to_datetime(temp_df["日期"], errors="coerce").min()
                if pd.notna(min_dt) and min_dt.to_pydatetime().date() <= oldest_date:
                    break
            last_date = pd.to_datetime(temp_df.iloc[-1]["日期"], errors="coerce")
            if pd.isna(last_date):
                break
            params["max_date"] = (last_date.to_pydatetime().date() - timedelta(days=1)).isoformat()

        if frames:
            out = pd.concat(frames, ignore_index=True)
        else:
            out = pd.DataFrame(columns=["日期", "今值", "预测值", "前值"])
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["今值"] = pd.to_numeric(out.get("今值"), errors="coerce")
        out["预测值"] = pd.to_numeric(out.get("预测值"), errors="coerce")
        out["前值"] = pd.to_numeric(out.get("前值"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        out["商品"] = str(label)
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_lpr_table(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:lpr:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        base_params = {
            "reportName": "RPTA_WEB_RATE",
            "columns": "ALL",
            "sortColumns": "TRADE_DATE",
            "sortTypes": "-1",
            "token": "894050c76af8597a853f5b408b759f5d",
            "pageNumber": "1",
            "pageSize": "500",
            "p": "1",
            "pageNo": "1",
            "pageNum": "1",
        }
        first = self._http_json(
            "GET",
            "https://datacenter-web.eastmoney.com/api/data/v1/get",
            params=base_params,
        )
        result = first.get("result", {}) if isinstance(first, dict) else {}
        total_pages = int(result.get("pages", 1) or 1)
        rows: list[pd.DataFrame] = []
        for page in range(1, max(1, total_pages) + 1):
            params = dict(base_params)
            page_str = str(page)
            params.update({"pageNumber": page_str, "pageNo": page_str, "pageNum": page_str, "p": page_str})
            payload = first if page == 1 else self._http_json(
                "GET",
                "https://datacenter-web.eastmoney.com/api/data/v1/get",
                params=params,
            )
            data = payload.get("result", {}).get("data", []) if isinstance(payload, dict) else []
            if not isinstance(data, list) or not data:
                continue
            temp_df = pd.DataFrame(data)
            rows.append(temp_df)
            if oldest_date is not None and "TRADE_DATE" in temp_df.columns:
                min_dt = pd.to_datetime(temp_df["TRADE_DATE"], errors="coerce").min()
                if pd.notna(min_dt) and min_dt.to_pydatetime().date() <= oldest_date:
                    break

        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["TRADE_DATE", "LPR1Y"])
        out["TRADE_DATE"] = pd.to_datetime(out.get("TRADE_DATE"), errors="coerce").dt.normalize()
        out["LPR1Y"] = pd.to_numeric(out.get("LPR1Y"), errors="coerce")
        out["LPR5Y"] = pd.to_numeric(out.get("LPR5Y"), errors="coerce")
        out["RATE_1"] = pd.to_numeric(out.get("RATE_1"), errors="coerce")
        out["RATE_2"] = pd.to_numeric(out.get("RATE_2"), errors="coerce")
        out = out.dropna(subset=["TRADE_DATE"]).sort_values("TRADE_DATE").drop_duplicates(subset=["TRADE_DATE"], keep="last").reset_index(drop=True)
        if latest_date is not None:
            out = out[out["TRADE_DATE"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_daily_energy(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"jin10:daily_energy:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_daily_energy()
        except Exception:
            return pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "沿海六大电库存", "日耗", "存煤可用天数"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        for col in ["沿海六大电库存", "日耗", "存煤可用天数"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_commodity_price_index(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:commodity_price_index:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_commodity_price_index()
        except Exception:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["最新值"] = pd.to_numeric(out.get("最新值"), errors="coerce")
        out["涨跌幅"] = pd.to_numeric(out.get("涨跌幅"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_energy_index(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:energy_index:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_energy_index()
        except Exception:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["最新值"] = pd.to_numeric(out.get("最新值"), errors="coerce")
        out["涨跌幅"] = pd.to_numeric(out.get("涨跌幅"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_freight_index(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        def _empty_frame() -> pd.DataFrame:
            return pd.DataFrame(
                columns=[
                    "截止日期",
                    "波罗的海综合运价指数BDI",
                    "油轮运价指数成品油运价指数BCTI",
                    "油轮运价指数原油运价指数BDTI",
                    "波罗的海超级大灵便型船BSI指数",
                ]
            )

        def _normalize_direct_series(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
            if frame is None or frame.empty:
                return pd.DataFrame(columns=["截止日期", value_name])
            out = frame.copy()
            out["截止日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
            out[value_name] = pd.to_numeric(out.get("最新值"), errors="coerce")
            out = (
                out[["截止日期", value_name]]
                .dropna(subset=["截止日期"])
                .sort_values("截止日期")
                .drop_duplicates(subset=["截止日期"], keep="last")
                .reset_index(drop=True)
            )
            return out

        def _direct_fallback() -> pd.DataFrame:
            try:
                import akshare as ak
            except Exception:
                return _empty_frame()
            frames: list[pd.DataFrame] = []
            direct_specs = [
                ("波罗的海综合运价指数BDI", "macro_shipping_bdi"),
                ("油轮运价指数成品油运价指数BCTI", "macro_shipping_bcti"),
                ("油轮运价指数原油运价指数BDTI", "macro_china_bdti_index"),
            ]
            for column_name, method_name in direct_specs:
                method = getattr(ak, method_name, None)
                if not callable(method):
                    continue
                try:
                    normalized = _normalize_direct_series(method(), column_name)
                except Exception:
                    continue
                if not normalized.empty:
                    frames.append(normalized)
            if not frames:
                return _empty_frame()
            out = frames[0]
            for frame in frames[1:]:
                out = out.merge(frame, on="截止日期", how="outer")
            if "波罗的海超级大灵便型船BSI指数" not in out.columns:
                out["波罗的海超级大灵便型船BSI指数"] = np.nan
            return (
                out[
                    [
                        "截止日期",
                        "波罗的海综合运价指数BDI",
                        "油轮运价指数成品油运价指数BCTI",
                        "油轮运价指数原油运价指数BDTI",
                        "波罗的海超级大灵便型船BSI指数",
                    ]
                ]
                .sort_values("截止日期")
                .drop_duplicates(subset=["截止日期"], keep="last")
                .reset_index(drop=True)
            )

        cache_key = f"eastmoney:freight_index:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_freight_index()
        except Exception:
            out = _direct_fallback()
        if out is None or out.empty:
            out = _direct_fallback()
        if out is None or out.empty:
            return _empty_frame()
        out = out.copy()
        date_column = "截止日期" if "截止日期" in out.columns else "日期" if "日期" in out.columns else None
        if date_column is None:
            return _empty_frame()
        out["截止日期"] = pd.to_datetime(out.get(date_column), errors="coerce").dt.normalize()
        for col in [
            "波罗的海综合运价指数BDI",
            "油轮运价指数成品油运价指数BCTI",
            "油轮运价指数原油运价指数BDTI",
            "波罗的海超级大灵便型船BSI指数",
        ]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["截止日期"]).sort_values("截止日期").drop_duplicates(subset=["截止日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["截止日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["截止日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_oil_hist(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:oil_hist:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.energy_oil_hist()
        except Exception:
            return pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["调整日期", "汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"])
        out = out.copy()
        out["调整日期"] = pd.to_datetime(out.get("调整日期"), errors="coerce").dt.normalize()
        for col in ["汽油价格", "柴油价格", "汽油涨跌", "柴油涨跌"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["调整日期"]).sort_values("调整日期").drop_duplicates(subset=["调整日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["调整日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["调整日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_oil_detail(self, date_text: str) -> pd.DataFrame:
        cache_key = f"eastmoney:oil_detail:{str(date_text)}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.energy_oil_detail(date=str(date_text))
        except Exception:
            return pd.DataFrame(columns=["日期", "地区", "V_0", "V_92", "V_95"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "地区", "V_0", "V_92", "V_95"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        for col in ["V_0", "V_92", "V_95"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["日期"]).reset_index(drop=True)
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_construction_index(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:construction_index:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_construction_index()
        except Exception:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["最新值"] = pd.to_numeric(out.get("最新值"), errors="coerce")
        out["涨跌幅"] = pd.to_numeric(out.get("涨跌幅"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_construction_price_index(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:construction_price_index:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_construction_price_index()
        except Exception:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["最新值"] = pd.to_numeric(out.get("最新值"), errors="coerce")
        out["涨跌幅"] = pd.to_numeric(out.get("涨跌幅"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_real_estate_index(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:real_estate_index:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_real_estate()
        except Exception:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["最新值"] = pd.to_numeric(out.get("最新值"), errors="coerce")
        out["涨跌幅"] = pd.to_numeric(out.get("涨跌幅"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_society_electricity(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"nbs:society_electricity:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_society_electricity()
        except Exception:
            out = self._fetch_society_electricity_http_fallback(oldest_date=oldest_date, latest_date=latest_date)
            if out is None or out.empty:
                return pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
            return out
        if out is None or out.empty:
            out = self._fetch_society_electricity_http_fallback(oldest_date=oldest_date, latest_date=latest_date)
            if out is None or out.empty:
                return pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"])
            return out
        out = out.copy()
        out["统计时间"] = out["统计时间"].astype(str)
        out["date"] = pd.to_datetime(out["统计时间"].str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
        for col in ["全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_society_electricity_http_fallback(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        try:
            from akshare.utils import demjson
        except Exception:
            return pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比", "date"])

        url = "https://quotes.sina.cn/mac/api/jsonp_v3.php/SINAREMOTECALLCALLBACK1601557771972/MacPage_Service.get_pagedata"
        base_params = {
            "cate": "industry",
            "event": "6",
            "from": "0",
            "num": "31",
            "condition": "",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": "https://finance.sina.com.cn/",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

        def _decode_rows(text: str) -> tuple[int, list[list[Any]]]:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("invalid_jsonp_payload")
            payload = demjson.decode(text[start : end + 1])
            data = payload.get("data", []) if isinstance(payload, dict) else []
            count = int(payload.get("count", 0)) if isinstance(payload, dict) else 0
            if not isinstance(data, list):
                data = []
            return count, data

        frames: list[pd.DataFrame] = []
        try:
            first = requests.get(
                url,
                params=base_params,
                headers=headers,
                timeout=self._timeout_seconds,
                verify=False,
            )
            if int(getattr(first, "status_code", 0)) >= 400:
                raise RuntimeError(f"http_error:{getattr(first, 'status_code', 0)}")
            total_count, data = _decode_rows(str(first.text))
        except Exception:
            return pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比", "date"])

        if data:
            frames.append(pd.DataFrame(data))
        page_size = int(base_params["num"])
        total_pages = int(max(1, (max(0, total_count) + page_size - 1) // page_size))
        for idx in range(1, total_pages):
            params = dict(base_params)
            params["from"] = str(idx * page_size)
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout_seconds,
                    verify=False,
                )
                if int(getattr(resp, "status_code", 0)) >= 400:
                    break
                _, page_data = _decode_rows(str(resp.text))
            except Exception:
                break
            if page_data:
                frames.append(pd.DataFrame(page_data))
        if not frames:
            return pd.DataFrame(columns=["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比", "date"])
        out = pd.concat(frames, ignore_index=True)
        out.columns = [
            "统计时间",
            "全社会用电量",
            "全社会用电量同比",
            "各行业用电量合计",
            "各行业用电量合计同比",
            "第一产业用电量",
            "第一产业用电量同比",
            "第二产业用电量",
            "第二产业用电量同比",
            "第三产业用电量",
            "第三产业用电量同比",
            "城乡居民生活用电量合计",
            "城乡居民生活用电量合计同比",
            "城镇居民用电量",
            "城镇居民用电量同比",
            "乡村居民用电量",
            "乡村居民用电量同比",
        ]
        out = out[["统计时间", "全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"]].copy()
        out["统计时间"] = out["统计时间"].astype(str)
        out["date"] = pd.to_datetime(out["统计时间"].str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
        for col in ["全社会用电量", "全社会用电量同比", "第二产业用电量", "第二产业用电量同比"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        return out

    def _fetch_society_traffic_volume(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"sina:society_traffic_volume:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_society_traffic_volume()
        except Exception:
            out = self._fetch_society_traffic_volume_http_fallback(oldest_date=oldest_date, latest_date=latest_date)
            if out is None or out.empty:
                return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
            return out
        if out is None or out.empty:
            out = self._fetch_society_traffic_volume_http_fallback(oldest_date=oldest_date, latest_date=latest_date)
            if out is None or out.empty:
                return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
            return out
        out = out.copy()
        out["统计时间"] = out["统计时间"].astype(str)
        out["date"] = pd.to_datetime(out["统计时间"].str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
        for col in [
            "货运量",
            "货运量同比增长",
            "货物周转量",
            "公里货物周转量同比增长",
            "沿海主要港口货物吞吐量",
            "沿海主要港口货物吞吐量同比增长",
            "其中:外贸货物吞吐量",
            "其中:外贸货物吞吐量同比增长",
        ]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_society_traffic_volume_http_fallback(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        try:
            from akshare.utils import demjson
        except Exception:
            return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
        url = "https://quotes.sina.cn/mac/api/jsonp_v3.php/SINAREMOTECALLCALLBACK1601559094538/MacPage_Service.get_pagedata"
        base_params = {"cate": "industry", "event": "10", "from": "0", "num": "31", "condition": ""}
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": "https://finance.sina.com.cn/",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

        def _decode(text: str) -> tuple[int, list[list[Any]], list[str]]:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("invalid_jsonp_payload")
            payload = demjson.decode(text[start : end + 1])
            count = int(payload.get("count", 0)) if isinstance(payload, dict) else 0
            rows = payload.get("data", {}).get("非累计", []) if isinstance(payload, dict) else []
            config_all = payload.get("config", {}).get("all", []) if isinstance(payload, dict) else []
            cols = [item[1] for item in config_all] if isinstance(config_all, list) else []
            return count, rows if isinstance(rows, list) else [], cols

        frames: list[pd.DataFrame] = []
        columns: list[str] = []
        try:
            resp = requests.get(url, params=base_params, headers=headers, timeout=self._timeout_seconds, verify=False)
            if int(getattr(resp, "status_code", 0)) >= 400:
                raise RuntimeError(f"http_error:{getattr(resp, 'status_code', 0)}")
            total_count, rows, columns = _decode(str(resp.text))
        except Exception:
            return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
        if rows:
            frames.append(pd.DataFrame(rows))
        page_size = int(base_params["num"])
        total_pages = int(max(1, (max(0, total_count) + page_size - 1) // page_size))
        for idx in range(1, total_pages):
            params = dict(base_params)
            params["from"] = str(idx * page_size)
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self._timeout_seconds, verify=False)
                if int(getattr(resp, "status_code", 0)) >= 400:
                    break
                _, rows, _ = _decode(str(resp.text))
            except Exception:
                break
            if rows:
                frames.append(pd.DataFrame(rows))
        if not frames or not columns:
            return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
        out = pd.concat(frames, ignore_index=True)
        out.columns = columns
        keep_cols = [
            "统计时间",
            "货运量",
            "货运量同比增长",
            "货物周转量",
            "公里货物周转量同比增长",
            "沿海主要港口货物吞吐量",
            "沿海主要港口货物吞吐量同比增长",
            "其中:外贸货物吞吐量",
            "其中:外贸货物吞吐量同比增长",
        ]
        out = out[[c for c in keep_cols if c in out.columns]].copy()
        out["统计时间"] = out["统计时间"].astype(str)
        out["date"] = pd.to_datetime(out["统计时间"].str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
        for col in keep_cols[1:]:
            if col in out.columns:
                out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        return out

    def _fetch_passenger_load_factor(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"sina:passenger_load_factor:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_passenger_load_factor()
        except Exception:
            return pd.DataFrame(columns=["统计时间", "客座率", "载运率", "date"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["统计时间", "客座率", "载运率", "date"])
        out = out.copy()
        out["统计时间"] = out["统计时间"].astype(str)
        out["date"] = pd.to_datetime(out["统计时间"].str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
        out["客座率"] = pd.to_numeric(out.get("客座率"), errors="coerce")
        out["载运率"] = pd.to_numeric(out.get("载运率"), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_postal_telecom(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"sina:postal_telecom:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_postal_telecommunicational()
        except Exception:
            return pd.DataFrame(columns=["统计时间", "特快专递", "特快专递同比增长", "电信业务总量", "电信业务总量同比增长", "date"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["统计时间", "特快专递", "特快专递同比增长", "电信业务总量", "电信业务总量同比增长", "date"])
        out = out.copy()
        out["统计时间"] = out["统计时间"].astype(str)
        out["date"] = pd.to_datetime(out["统计时间"].str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
        for col in ["特快专递", "特快专递同比增长", "电信业务总量", "电信业务总量同比增长"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_society_traffic_volume(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"sina:society_traffic_volume:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_society_traffic_volume()
        except Exception:
            out = self._fetch_society_traffic_volume_http_fallback(oldest_date=oldest_date, latest_date=latest_date)
            if out is None or out.empty:
                return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
            return out
        if out is None or out.empty:
            out = self._fetch_society_traffic_volume_http_fallback(oldest_date=oldest_date, latest_date=latest_date)
            if out is None or out.empty:
                return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
            return out
        out = out.copy()
        out["统计时间"] = out["统计时间"].astype(str)
        out["date"] = pd.to_datetime(out["统计时间"].str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
        for col in [
            "货运量",
            "货运量同比增长",
            "货物周转量",
            "公里货物周转量同比增长",
            "沿海主要港口货物吞吐量",
            "沿海主要港口货物吞吐量同比增长",
            "其中:外贸货物吞吐量",
            "其中:外贸货物吞吐量同比增长",
        ]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_society_traffic_volume_http_fallback(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        try:
            from akshare.utils import demjson
        except Exception:
            return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
        url = "https://quotes.sina.cn/mac/api/jsonp_v3.php/SINAREMOTECALLCALLBACK1601559094538/MacPage_Service.get_pagedata"
        base_params = {"cate": "industry", "event": "10", "from": "0", "num": "31", "condition": ""}
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": "https://finance.sina.com.cn/",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

        def _decode(text: str) -> tuple[int, list[list[Any]], list[str]]:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("invalid_jsonp_payload")
            payload = demjson.decode(text[start : end + 1])
            count = int(payload.get("count", 0)) if isinstance(payload, dict) else 0
            rows = payload.get("data", {}).get("非累计", []) if isinstance(payload, dict) else []
            config_all = payload.get("config", {}).get("all", []) if isinstance(payload, dict) else []
            cols = [item[1] for item in config_all] if isinstance(config_all, list) else []
            return count, rows if isinstance(rows, list) else [], cols

        frames: list[pd.DataFrame] = []
        columns: list[str] = []
        try:
            resp = requests.get(url, params=base_params, headers=headers, timeout=self._timeout_seconds, verify=False)
            if int(getattr(resp, "status_code", 0)) >= 400:
                raise RuntimeError(f"http_error:{getattr(resp, 'status_code', 0)}")
            total_count, rows, columns = _decode(str(resp.text))
        except Exception:
            return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
        if rows:
            frames.append(pd.DataFrame(rows))
        page_size = int(base_params["num"])
        total_pages = int(max(1, (max(0, total_count) + page_size - 1) // page_size))
        for idx in range(1, total_pages):
            params = dict(base_params)
            params["from"] = str(idx * page_size)
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=self._timeout_seconds, verify=False)
                if int(getattr(resp, "status_code", 0)) >= 400:
                    break
                _, rows, _ = _decode(str(resp.text))
            except Exception:
                break
            if rows:
                frames.append(pd.DataFrame(rows))
        if not frames or not columns:
            return pd.DataFrame(columns=["统计时间", "货运量", "货运量同比增长", "货物周转量", "公里货物周转量同比增长", "沿海主要港口货物吞吐量", "沿海主要港口货物吞吐量同比增长", "其中:外贸货物吞吐量", "其中:外贸货物吞吐量同比增长", "date"])
        out = pd.concat(frames, ignore_index=True)
        out.columns = columns
        keep_cols = [
            "统计时间",
            "货运量",
            "货运量同比增长",
            "货物周转量",
            "公里货物周转量同比增长",
            "沿海主要港口货物吞吐量",
            "沿海主要港口货物吞吐量同比增长",
            "其中:外贸货物吞吐量",
            "其中:外贸货物吞吐量同比增长",
        ]
        out = out[[c for c in keep_cols if c in out.columns]].copy()
        out["统计时间"] = out["统计时间"].astype(str)
        out["date"] = pd.to_datetime(out["统计时间"].str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
        for col in keep_cols[1:]:
            if col in out.columns:
                out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        return out

    def _fetch_new_house_price(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"nbs:new_house_price:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_new_house_price()
        except Exception:
            return pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "城市", "新建商品住宅价格指数-同比", "新建商品住宅价格指数-环比", "二手住宅价格指数-同比", "二手住宅价格指数-环比"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        for col in [
            "新建商品住宅价格指数-同比",
            "新建商品住宅价格指数-环比",
            "二手住宅价格指数-同比",
            "二手住宅价格指数-环比",
        ]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["日期"]).reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_industrial_production_yoy(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"jin10:industrial_production_yoy:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_industrial_production_yoy()
        except Exception:
            return pd.DataFrame(columns=["日期", "今值"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "今值"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["今值"] = pd.to_numeric(out.get("今值"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_exports_yoy(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"jin10:exports_yoy:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_exports_yoy()
        except Exception:
            return pd.DataFrame(columns=["日期", "今值"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "今值"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["今值"] = pd.to_numeric(out.get("今值"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_imports_yoy(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"jin10:imports_yoy:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_imports_yoy()
        except Exception:
            return pd.DataFrame(columns=["日期", "今值"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "今值"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["今值"] = pd.to_numeric(out.get("今值"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_pmi_manufacturing(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"jin10:pmi_manufacturing:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_pmi_yearly()
        except Exception:
            return pd.DataFrame(columns=["日期", "今值"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "今值"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["今值"] = pd.to_numeric(out.get("今值"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_non_man_pmi(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"jin10:non_man_pmi:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_non_man_pmi()
        except Exception:
            return pd.DataFrame(columns=["日期", "今值"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "今值"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["今值"] = pd.to_numeric(out.get("今值"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_asphalt_inventory(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:asphalt_inventory:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.futures_inventory_em(symbol="沥青")
        except Exception:
            return pd.DataFrame(columns=["日期", "库存", "增减"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "库存", "增减"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["库存"] = pd.to_numeric(out.get("库存"), errors="coerce")
        out["增减"] = pd.to_numeric(out.get("增减"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_inventory_em_symbol(
        self,
        symbol_name: str,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:inventory_em:{symbol_name}:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.futures_inventory_em(symbol=symbol_name)
        except Exception:
            return pd.DataFrame(columns=["日期", "库存", "增减"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "库存", "增减"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["库存"] = pd.to_numeric(out.get("库存"), errors="coerce")
        out["增减"] = pd.to_numeric(out.get("增减"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_lfu_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("低硫燃料油", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_fuel_oil_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("燃油", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_rebar_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("螺纹钢", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_hotcoil_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("热卷", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_coking_coal_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("焦煤", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_coke_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("焦炭", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_iron_ore_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("铁矿石", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_glass_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("玻璃", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_soda_ash_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("纯碱", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_pvc_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("PVC", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_pp_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("聚丙烯", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_methanol_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("甲醇", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_eg_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("乙二醇", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_lpg_inventory(self, *, oldest_date: date | None = None, latest_date: date | None = None) -> pd.DataFrame:
        return self._fetch_inventory_em_symbol("液化石油气", oldest_date=oldest_date, latest_date=latest_date)

    def _fetch_money_supply(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"nbs:money_supply:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_money_supply()
        except Exception:
            return pd.DataFrame(columns=["月份", "货币和准货币(M2)-数量(亿元)", "货币和准货币(M2)-同比增长", "货币(M1)-数量(亿元)", "货币(M1)-同比增长", "流通中的现金(M0)-数量(亿元)", "流通中的现金(M0)-同比增长"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["月份", "货币和准货币(M2)-数量(亿元)", "货币和准货币(M2)-同比增长", "货币(M1)-数量(亿元)", "货币(M1)-同比增长", "流通中的现金(M0)-数量(亿元)", "流通中的现金(M0)-同比增长"])
        out = out.copy()
        out["date"] = pd.to_datetime(out["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False), errors="coerce").dt.normalize()
        for col in ["货币和准货币(M2)-数量(亿元)", "货币和准货币(M2)-同比增长", "货币(M1)-数量(亿元)", "货币(M1)-同比增长", "流通中的现金(M0)-数量(亿元)", "流通中的现金(M0)-同比增长"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_new_financial_credit(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"nbs:new_financial_credit:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_new_financial_credit()
        except Exception:
            return pd.DataFrame(columns=["月份", "当月", "当月-同比增长", "累计"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["月份", "当月", "当月-同比增长", "累计"])
        out = out.copy()
        out["date"] = pd.to_datetime(out["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False), errors="coerce").dt.normalize()
        for col in ["当月", "当月-同比增长", "累计"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_fixed_asset_investment(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"nbs:fixed_asset_investment:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_gdzctz()
        except Exception:
            return pd.DataFrame(columns=["月份", "当月", "同比增长", "环比增长", "自年初累计"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["月份", "当月", "同比增长", "环比增长", "自年初累计"])
        out = out.copy()
        out["date"] = pd.to_datetime(out["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False), errors="coerce").dt.normalize()
        for col in ["当月", "同比增长", "环比增长", "自年初累计"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_consumer_goods_retail(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"nbs:consumer_goods_retail:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_consumer_goods_retail()
        except Exception:
            return pd.DataFrame(columns=["月份", "当月", "同比增长", "环比增长", "累计", "累计-同比增长"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["月份", "当月", "同比增长", "环比增长", "累计", "累计-同比增长"])
        out = out.copy()
        out["date"] = pd.to_datetime(out["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False), errors="coerce").dt.normalize()
        for col in ["当月", "同比增长", "环比增长", "累计", "累计-同比增长"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_enterprise_goods_price(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"nbs:enterprise_goods_price:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_qyspjg()
        except Exception:
            return pd.DataFrame(columns=["月份", "总指数-指数值", "总指数-同比增长", "总指数-环比增长", "煤油电-指数值", "煤油电-同比增长", "煤油电-环比增长"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["月份", "总指数-指数值", "总指数-同比增长", "总指数-环比增长", "煤油电-指数值", "煤油电-同比增长", "煤油电-环比增长"])
        out = out.copy()
        out["date"] = pd.to_datetime(out["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False), errors="coerce").dt.normalize()
        for col in ["总指数-指数值", "总指数-同比增长", "总指数-环比增长", "煤油电-指数值", "煤油电-同比增长", "煤油电-环比增长"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_consumer_confidence(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"nbs:consumer_confidence:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_xfzxx()
        except Exception:
            return pd.DataFrame(
                columns=[
                    "月份",
                    "消费者信心指数-指数值",
                    "消费者信心指数-同比增长",
                    "消费者信心指数-环比增长",
                    "消费者满意指数-指数值",
                    "消费者满意指数-同比增长",
                    "消费者满意指数-环比增长",
                    "消费者预期指数-指数值",
                    "消费者预期指数-同比增长",
                    "消费者预期指数-环比增长",
                ]
            )
        if out is None or out.empty:
            return pd.DataFrame(
                columns=[
                    "月份",
                    "消费者信心指数-指数值",
                    "消费者信心指数-同比增长",
                    "消费者信心指数-环比增长",
                    "消费者满意指数-指数值",
                    "消费者满意指数-同比增长",
                    "消费者满意指数-环比增长",
                    "消费者预期指数-指数值",
                    "消费者预期指数-同比增长",
                    "消费者预期指数-环比增长",
                ]
            )
        out = out.copy()
        out["date"] = pd.to_datetime(out["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False), errors="coerce").dt.normalize()
        for col in [
            "消费者信心指数-指数值",
            "消费者信心指数-同比增长",
            "消费者信心指数-环比增长",
            "消费者满意指数-指数值",
            "消费者满意指数-同比增长",
            "消费者满意指数-环比增长",
            "消费者预期指数-指数值",
            "消费者预期指数-同比增长",
            "消费者预期指数-环比增长",
        ]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["date"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["date"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_bank_financing_index(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"eastmoney:bank_financing:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_bank_financing()
        except Exception:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "最新值", "涨跌幅"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["最新值"] = pd.to_numeric(out.get("最新值"), errors="coerce")
        out["涨跌幅"] = pd.to_numeric(out.get("涨跌幅"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    def _fetch_fx_reserves(
        self,
        *,
        oldest_date: date | None = None,
        latest_date: date | None = None,
    ) -> pd.DataFrame:
        cache_key = f"jin10:fx_reserves:{oldest_date.isoformat() if oldest_date else ''}:{latest_date.isoformat() if latest_date else ''}"
        with self._macro_cache_lock:
            cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        try:
            import akshare as ak
            out = ak.macro_china_fx_reserves_yearly()
        except Exception:
            return pd.DataFrame(columns=["日期", "今值"])
        if out is None or out.empty:
            return pd.DataFrame(columns=["日期", "今值"])
        out = out.copy()
        out["日期"] = pd.to_datetime(out.get("日期"), errors="coerce").dt.normalize()
        out["今值"] = pd.to_numeric(out.get("今值"), errors="coerce")
        out = out.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last").reset_index(drop=True)
        if oldest_date is not None:
            out = out[out["日期"].dt.date >= oldest_date].copy()
        if latest_date is not None:
            out = out[out["日期"].dt.date <= latest_date].copy()
        with self._macro_cache_lock:
            self._macro_cache[cache_key] = out.copy()
        return out

    @staticmethod
    def _empty_macro_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "date",
                "cpi_yoy",
                "ppi_yoy",
                "lpr_1y",
                "source",
                "cpi_source",
                "ppi_source",
                "lpr_source",
                "coastal_power_coal_inventory",
                "coastal_power_coal_daily_burn",
                "coastal_power_coal_days",
                "commodity_price_index",
                "commodity_price_index_pct_chg",
                "energy_index",
                "energy_index_pct_chg",
                "bdi_index",
                "bcti_index",
                "bdti_index",
                "bsi_index",
                "gasoline_price",
                "diesel_price",
                "gasoline_price_delta",
                "diesel_price_delta",
                "diesel_price_regional_mean",
                "gasoline_92_price_regional_mean",
                "gasoline_95_price_regional_mean",
                "construction_index",
                "construction_index_pct_chg",
                "construction_price_index",
                "construction_price_index_pct_chg",
                "real_estate_index",
                "real_estate_index_pct_chg",
                "society_electricity_total",
                "society_electricity_yoy",
                "secondary_industry_electricity",
                "secondary_industry_electricity_yoy",
                "new_house_price_yoy",
                "new_house_price_mom",
                "resale_house_price_yoy",
                "resale_house_price_mom",
                "industrial_production_yoy",
                "exports_yoy",
                "imports_yoy",
                "pmi_manufacturing",
                "pmi_non_manufacturing",
                "fixed_asset_investment_monthly",
                "fixed_asset_investment_yoy",
                "fixed_asset_investment_mom",
                "fixed_asset_investment_cum",
                "retail_sales_monthly",
                "retail_sales_yoy",
                "retail_sales_mom",
                "retail_sales_cum",
                "retail_sales_cum_yoy",
                "enterprise_goods_price_index",
                "enterprise_goods_price_yoy",
                "enterprise_goods_price_mom",
                "energy_goods_price_index",
                "energy_goods_price_yoy",
                "energy_goods_price_mom",
                "consumer_confidence_index",
                "consumer_confidence_yoy",
                "consumer_confidence_mom",
                "consumer_satisfaction_index",
                "consumer_satisfaction_yoy",
                "consumer_satisfaction_mom",
                "consumer_expectation_index",
                "consumer_expectation_yoy",
                "consumer_expectation_mom",
                "m2_level",
                "m2_yoy",
                "m1_level",
                "m1_yoy",
                "m0_level",
                "m0_yoy",
                "new_financial_credit_monthly",
                "new_financial_credit_yoy",
                "new_financial_credit_cum",
                "bank_financing_index",
                "bank_financing_index_pct_chg",
                "fx_reserves",
                "asphalt_inventory",
                "asphalt_inventory_delta",
                "lfu_inventory",
                "lfu_inventory_delta",
                "fuel_oil_inventory",
                "fuel_oil_inventory_delta",
                "rebar_inventory",
                "rebar_inventory_delta",
                "hotcoil_inventory",
                "hotcoil_inventory_delta",
                "coking_coal_inventory",
                "coking_coal_inventory_delta",
                "coke_inventory",
                "coke_inventory_delta",
                "iron_ore_inventory",
                "iron_ore_inventory_delta",
                "cargo_volume",
                "cargo_volume_yoy",
                "cargo_turnover",
                "cargo_turnover_yoy",
                "coastal_port_throughput",
                "coastal_port_throughput_yoy",
                "coastal_port_foreign_trade_throughput",
                "coastal_port_foreign_trade_throughput_yoy",
                "passenger_load_factor",
                "cargo_load_factor",
                "express_delivery_volume",
                "express_delivery_yoy",
                "telecom_business_total",
                "telecom_business_yoy",
                "m2_level",
                "m2_yoy",
                "m1_level",
                "m1_yoy",
                "m0_level",
                "m0_yoy",
                "new_financial_credit_monthly",
                "new_financial_credit_yoy",
                "new_financial_credit_cum",
                "bank_financing_index",
                "bank_financing_index_pct_chg",
                "fx_reserves",
                "glass_inventory",
                "glass_inventory_delta",
                "soda_ash_inventory",
                "soda_ash_inventory_delta",
                "pvc_inventory",
                "pvc_inventory_delta",
                "pp_inventory",
                "pp_inventory_delta",
                "methanol_inventory",
                "methanol_inventory_delta",
                "eg_inventory",
                "eg_inventory_delta",
                "lpg_inventory",
                "lpg_inventory_delta",
                "daily_energy_source",
                "commodity_price_index_source",
                "energy_index_source",
                "freight_index_source",
                "oil_hist_source",
                "oil_detail_source",
                "construction_index_source",
                "construction_price_index_source",
                "real_estate_index_source",
                "society_electricity_source",
                "new_house_price_source",
                "industrial_production_source",
                "exports_source",
                "imports_source",
                "pmi_source",
                "non_man_pmi_source",
                "fixed_asset_investment_source",
                "retail_sales_source",
                "enterprise_goods_price_source",
                "consumer_confidence_source",
                "money_supply_source",
                "new_financial_credit_source",
                "bank_financing_source",
                "fx_reserves_source",
                "asphalt_inventory_source",
                "lfu_inventory_source",
                "fuel_oil_inventory_source",
                "rebar_inventory_source",
                "hotcoil_inventory_source",
                "coking_coal_inventory_source",
                "coke_inventory_source",
                "iron_ore_inventory_source",
                "society_traffic_source",
                "money_supply_source",
                "new_financial_credit_source",
                "bank_financing_source",
                "fx_reserves_source",
                "glass_inventory_source",
                "soda_ash_inventory_source",
                "pvc_inventory_source",
                "pp_inventory_source",
                "methanol_inventory_source",
                "eg_inventory_source",
                "lpg_inventory_source",
            ]
        )

    def fetch_macro(self, start: date, end: date) -> pd.DataFrame:
        oldest_date = start - timedelta(days=45)
        try:
            cpi = self._fetch_jin10_indicator("56", "中国CPI年率报告", oldest_date=oldest_date, latest_date=end).rename(columns={"日期": "date", "今值": "cpi_yoy"})
            ppi = self._fetch_jin10_indicator("60", "中国PPI年率报告", oldest_date=oldest_date, latest_date=end).rename(columns={"日期": "date", "今值": "ppi_yoy"})
            lpr = self._fetch_lpr_table(oldest_date=oldest_date, latest_date=end).rename(columns={"TRADE_DATE": "date", "LPR1Y": "lpr_1y"})
            energy = self._fetch_daily_energy(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "日期": "date",
                    "沿海六大电库存": "coastal_power_coal_inventory",
                    "日耗": "coastal_power_coal_daily_burn",
                    "存煤可用天数": "coastal_power_coal_days",
                }
            )
            commodity = self._fetch_commodity_price_index(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "日期": "date",
                    "最新值": "commodity_price_index",
                    "涨跌幅": "commodity_price_index_pct_chg",
                }
            )
            energy_index = self._fetch_energy_index(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "最新值": "energy_index", "涨跌幅": "energy_index_pct_chg"}
            )
            freight = self._fetch_freight_index(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "截止日期": "date",
                    "波罗的海综合运价指数BDI": "bdi_index",
                    "油轮运价指数成品油运价指数BCTI": "bcti_index",
                    "油轮运价指数原油运价指数BDTI": "bdti_index",
                    "波罗的海超级大灵便型船BSI指数": "bsi_index",
                }
            )
            oil_hist = self._fetch_oil_hist(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "调整日期": "date",
                    "汽油价格": "gasoline_price",
                    "柴油价格": "diesel_price",
                    "汽油涨跌": "gasoline_price_delta",
                    "柴油涨跌": "diesel_price_delta",
                }
            )
            oil_detail_frames: list[pd.DataFrame] = []
            if not oil_hist.empty and "date" in oil_hist.columns:
                for oil_dt in pd.to_datetime(oil_hist["date"], errors="coerce").dropna().dt.date.unique().tolist():
                    detail = self._fetch_oil_detail(oil_dt.strftime("%Y%m%d"))
                    if detail is None or detail.empty:
                        continue
                    detail = detail.copy()
                    detail["date"] = pd.to_datetime(detail.get("日期"), errors="coerce").dt.normalize()
                    detail["diesel_price_regional_mean"] = pd.to_numeric(detail.get("V_0"), errors="coerce")
                    detail["gasoline_92_price_regional_mean"] = pd.to_numeric(detail.get("V_92"), errors="coerce")
                    detail["gasoline_95_price_regional_mean"] = pd.to_numeric(detail.get("V_95"), errors="coerce")
                    agg = (
                        detail.groupby("date", as_index=False)[
                            ["diesel_price_regional_mean", "gasoline_92_price_regional_mean", "gasoline_95_price_regional_mean"]
                        ].mean(numeric_only=True)
                    )
                    oil_detail_frames.append(agg)
            oil_detail = (
                pd.concat(oil_detail_frames, ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
                if oil_detail_frames
                else pd.DataFrame(columns=["date", "diesel_price_regional_mean", "gasoline_92_price_regional_mean", "gasoline_95_price_regional_mean"])
            )
            construction_index = self._fetch_construction_index(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "最新值": "construction_index", "涨跌幅": "construction_index_pct_chg"}
            )
            construction_price = self._fetch_construction_price_index(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "最新值": "construction_price_index", "涨跌幅": "construction_price_index_pct_chg"}
            )
            real_estate = self._fetch_real_estate_index(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "最新值": "real_estate_index", "涨跌幅": "real_estate_index_pct_chg"}
            )
            society_electricity = self._fetch_society_electricity(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "全社会用电量": "society_electricity_total",
                    "全社会用电量同比": "society_electricity_yoy",
                    "第二产业用电量": "secondary_industry_electricity",
                    "第二产业用电量同比": "secondary_industry_electricity_yoy",
                }
            )
            new_house_price = self._fetch_new_house_price(oldest_date=oldest_date, latest_date=end)
            industrial = self._fetch_industrial_production_yoy(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "今值": "industrial_production_yoy"}
            )
            exports = self._fetch_exports_yoy(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "今值": "exports_yoy"}
            )
            imports = self._fetch_imports_yoy(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "今值": "imports_yoy"}
            )
            pmi = self._fetch_pmi_manufacturing(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "今值": "pmi_manufacturing"}
            )
            non_man = self._fetch_non_man_pmi(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "今值": "pmi_non_manufacturing"}
            )
            fixed_asset = self._fetch_fixed_asset_investment(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "当月": "fixed_asset_investment_monthly",
                    "同比增长": "fixed_asset_investment_yoy",
                    "环比增长": "fixed_asset_investment_mom",
                    "自年初累计": "fixed_asset_investment_cum",
                }
            )
            retail_sales = self._fetch_consumer_goods_retail(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "当月": "retail_sales_monthly",
                    "同比增长": "retail_sales_yoy",
                    "环比增长": "retail_sales_mom",
                    "累计": "retail_sales_cum",
                    "累计-同比增长": "retail_sales_cum_yoy",
                }
            )
            enterprise_goods_price = self._fetch_enterprise_goods_price(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "总指数-指数值": "enterprise_goods_price_index",
                    "总指数-同比增长": "enterprise_goods_price_yoy",
                    "总指数-环比增长": "enterprise_goods_price_mom",
                    "煤油电-指数值": "energy_goods_price_index",
                    "煤油电-同比增长": "energy_goods_price_yoy",
                    "煤油电-环比增长": "energy_goods_price_mom",
                }
            )
            consumer_confidence = self._fetch_consumer_confidence(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "消费者信心指数-指数值": "consumer_confidence_index",
                    "消费者信心指数-同比增长": "consumer_confidence_yoy",
                    "消费者信心指数-环比增长": "consumer_confidence_mom",
                    "消费者满意指数-指数值": "consumer_satisfaction_index",
                    "消费者满意指数-同比增长": "consumer_satisfaction_yoy",
                    "消费者满意指数-环比增长": "consumer_satisfaction_mom",
                    "消费者预期指数-指数值": "consumer_expectation_index",
                    "消费者预期指数-同比增长": "consumer_expectation_yoy",
                    "消费者预期指数-环比增长": "consumer_expectation_mom",
                }
            )
            money_supply = self._fetch_money_supply(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "货币和准货币(M2)-数量(亿元)": "m2_level",
                    "货币和准货币(M2)-同比增长": "m2_yoy",
                    "货币(M1)-数量(亿元)": "m1_level",
                    "货币(M1)-同比增长": "m1_yoy",
                    "流通中的现金(M0)-数量(亿元)": "m0_level",
                    "流通中的现金(M0)-同比增长": "m0_yoy",
                }
            )
            new_credit = self._fetch_new_financial_credit(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "当月": "new_financial_credit_monthly",
                    "当月-同比增长": "new_financial_credit_yoy",
                    "累计": "new_financial_credit_cum",
                }
            )
            bank_financing = self._fetch_bank_financing_index(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "最新值": "bank_financing_index", "涨跌幅": "bank_financing_index_pct_chg"}
            )
            fx_reserves = self._fetch_fx_reserves(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "今值": "fx_reserves"}
            )
            asphalt_inventory = self._fetch_asphalt_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "asphalt_inventory", "增减": "asphalt_inventory_delta"}
            )
            lfu_inventory = self._fetch_lfu_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "lfu_inventory", "增减": "lfu_inventory_delta"}
            )
            fuel_oil_inventory = self._fetch_fuel_oil_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "fuel_oil_inventory", "增减": "fuel_oil_inventory_delta"}
            )
            rebar_inventory = self._fetch_rebar_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "rebar_inventory", "增减": "rebar_inventory_delta"}
            )
            hotcoil_inventory = self._fetch_hotcoil_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "hotcoil_inventory", "增减": "hotcoil_inventory_delta"}
            )
            coking_coal_inventory = self._fetch_coking_coal_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "coking_coal_inventory", "增减": "coking_coal_inventory_delta"}
            )
            coke_inventory = self._fetch_coke_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "coke_inventory", "增减": "coke_inventory_delta"}
            )
            iron_ore_inventory = self._fetch_iron_ore_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "iron_ore_inventory", "增减": "iron_ore_inventory_delta"}
            )
            society_traffic = self._fetch_society_traffic_volume(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "货运量": "cargo_volume",
                    "货运量同比增长": "cargo_volume_yoy",
                    "货物周转量": "cargo_turnover",
                    "公里货物周转量同比增长": "cargo_turnover_yoy",
                    "沿海主要港口货物吞吐量": "coastal_port_throughput",
                    "沿海主要港口货物吞吐量同比增长": "coastal_port_throughput_yoy",
                    "其中:外贸货物吞吐量": "coastal_port_foreign_trade_throughput",
                    "其中:外贸货物吞吐量同比增长": "coastal_port_foreign_trade_throughput_yoy",
                }
            )
            passenger = self._fetch_passenger_load_factor(oldest_date=oldest_date, latest_date=end).rename(
                columns={"date": "date", "客座率": "passenger_load_factor", "载运率": "cargo_load_factor"}
            )
            postal = self._fetch_postal_telecom(oldest_date=oldest_date, latest_date=end).rename(
                columns={
                    "date": "date",
                    "特快专递": "express_delivery_volume",
                    "特快专递同比增长": "express_delivery_yoy",
                    "电信业务总量": "telecom_business_total",
                    "电信业务总量同比增长": "telecom_business_yoy",
                }
            )
            glass_inventory = self._fetch_glass_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "glass_inventory", "增减": "glass_inventory_delta"}
            )
            soda_ash_inventory = self._fetch_soda_ash_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "soda_ash_inventory", "增减": "soda_ash_inventory_delta"}
            )
            pvc_inventory = self._fetch_pvc_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "pvc_inventory", "增减": "pvc_inventory_delta"}
            )
            pp_inventory = self._fetch_pp_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "pp_inventory", "增减": "pp_inventory_delta"}
            )
            methanol_inventory = self._fetch_methanol_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "methanol_inventory", "增减": "methanol_inventory_delta"}
            )
            eg_inventory = self._fetch_eg_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "eg_inventory", "增减": "eg_inventory_delta"}
            )
            lpg_inventory = self._fetch_lpg_inventory(oldest_date=oldest_date, latest_date=end).rename(
                columns={"日期": "date", "库存": "lpg_inventory", "增减": "lpg_inventory_delta"}
            )
        except Exception:
            return self._empty_macro_frame()

        cpi = cpi[["date", "cpi_yoy"]].copy() if not cpi.empty else pd.DataFrame(columns=["date", "cpi_yoy"])
        ppi = ppi[["date", "ppi_yoy"]].copy() if not ppi.empty else pd.DataFrame(columns=["date", "ppi_yoy"])
        lpr = lpr[["date", "lpr_1y"]].copy() if not lpr.empty else pd.DataFrame(columns=["date", "lpr_1y"])
        energy = (
            energy[["date", "coastal_power_coal_inventory", "coastal_power_coal_daily_burn", "coastal_power_coal_days"]].copy()
            if not energy.empty
            else pd.DataFrame(columns=["date", "coastal_power_coal_inventory", "coastal_power_coal_daily_burn", "coastal_power_coal_days"])
        )
        commodity = (
            commodity[["date", "commodity_price_index", "commodity_price_index_pct_chg"]].copy()
            if not commodity.empty
            else pd.DataFrame(columns=["date", "commodity_price_index", "commodity_price_index_pct_chg"])
        )
        energy_index = (
            energy_index[["date", "energy_index", "energy_index_pct_chg"]].copy()
            if not energy_index.empty
            else pd.DataFrame(columns=["date", "energy_index", "energy_index_pct_chg"])
        )
        freight = (
            freight[["date", "bdi_index", "bcti_index", "bdti_index", "bsi_index"]].copy()
            if not freight.empty
            else pd.DataFrame(columns=["date", "bdi_index", "bcti_index", "bdti_index", "bsi_index"])
        )
        oil_hist = (
            oil_hist[["date", "gasoline_price", "diesel_price", "gasoline_price_delta", "diesel_price_delta"]].copy()
            if not oil_hist.empty
            else pd.DataFrame(columns=["date", "gasoline_price", "diesel_price", "gasoline_price_delta", "diesel_price_delta"])
        )
        oil_detail = (
            oil_detail[["date", "diesel_price_regional_mean", "gasoline_92_price_regional_mean", "gasoline_95_price_regional_mean"]].copy()
            if not oil_detail.empty
            else pd.DataFrame(columns=["date", "diesel_price_regional_mean", "gasoline_92_price_regional_mean", "gasoline_95_price_regional_mean"])
        )
        construction_index = (
            construction_index[["date", "construction_index", "construction_index_pct_chg"]].copy()
            if not construction_index.empty
            else pd.DataFrame(columns=["date", "construction_index", "construction_index_pct_chg"])
        )
        construction_price = (
            construction_price[["date", "construction_price_index", "construction_price_index_pct_chg"]].copy()
            if not construction_price.empty
            else pd.DataFrame(columns=["date", "construction_price_index", "construction_price_index_pct_chg"])
        )
        real_estate = (
            real_estate[["date", "real_estate_index", "real_estate_index_pct_chg"]].copy()
            if not real_estate.empty
            else pd.DataFrame(columns=["date", "real_estate_index", "real_estate_index_pct_chg"])
        )
        society_electricity = (
            (
                society_electricity.assign(
                    date=(
                        pd.to_datetime(
                            society_electricity["统计时间"].astype(str).str.replace(".", "-", regex=False) + "-01",
                            errors="coerce",
                        ).dt.normalize()
                        if "date" not in society_electricity.columns and "统计时间" in society_electricity.columns
                        else society_electricity.get("date")
                    )
                )[["date", "society_electricity_total", "society_electricity_yoy", "secondary_industry_electricity", "secondary_industry_electricity_yoy"]].copy()
            )
            if not society_electricity.empty
            else pd.DataFrame(columns=["date", "society_electricity_total", "society_electricity_yoy", "secondary_industry_electricity", "secondary_industry_electricity_yoy"])
        )
        new_house_price = (
            new_house_price.groupby("日期", as_index=False)[
                [
                    "新建商品住宅价格指数-同比",
                    "新建商品住宅价格指数-环比",
                    "二手住宅价格指数-同比",
                    "二手住宅价格指数-环比",
                ]
            ]
            .mean(numeric_only=True)
            .rename(
                columns={
                    "日期": "date",
                    "新建商品住宅价格指数-同比": "new_house_price_yoy",
                    "新建商品住宅价格指数-环比": "new_house_price_mom",
                    "二手住宅价格指数-同比": "resale_house_price_yoy",
                    "二手住宅价格指数-环比": "resale_house_price_mom",
                }
            )
            if not new_house_price.empty
            else pd.DataFrame(columns=["date", "new_house_price_yoy", "new_house_price_mom", "resale_house_price_yoy", "resale_house_price_mom"])
        )
        industrial = (
            industrial[["date", "industrial_production_yoy"]].copy()
            if not industrial.empty
            else pd.DataFrame(columns=["date", "industrial_production_yoy"])
        )
        exports = (
            exports[["date", "exports_yoy"]].copy()
            if not exports.empty
            else pd.DataFrame(columns=["date", "exports_yoy"])
        )
        imports = (
            imports[["date", "imports_yoy"]].copy()
            if not imports.empty
            else pd.DataFrame(columns=["date", "imports_yoy"])
        )
        pmi = (
            pmi[["date", "pmi_manufacturing"]].copy()
            if not pmi.empty
            else pd.DataFrame(columns=["date", "pmi_manufacturing"])
        )
        non_man = (
            non_man[["date", "pmi_non_manufacturing"]].copy()
            if not non_man.empty
            else pd.DataFrame(columns=["date", "pmi_non_manufacturing"])
        )
        fixed_asset = (
            (
                fixed_asset.assign(
                    date=(
                        pd.to_datetime(
                            fixed_asset["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False),
                            errors="coerce",
                        ).dt.normalize()
                        if "date" not in fixed_asset.columns and "月份" in fixed_asset.columns
                        else fixed_asset.get("date")
                    )
                )[["date", "fixed_asset_investment_monthly", "fixed_asset_investment_yoy", "fixed_asset_investment_mom", "fixed_asset_investment_cum"]].copy()
            )
            if not fixed_asset.empty
            else pd.DataFrame(columns=["date", "fixed_asset_investment_monthly", "fixed_asset_investment_yoy", "fixed_asset_investment_mom", "fixed_asset_investment_cum"])
        )
        retail_sales = (
            (
                retail_sales.assign(
                    date=(
                        pd.to_datetime(
                            retail_sales["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False),
                            errors="coerce",
                        ).dt.normalize()
                        if "date" not in retail_sales.columns and "月份" in retail_sales.columns
                        else retail_sales.get("date")
                    )
                )[["date", "retail_sales_monthly", "retail_sales_yoy", "retail_sales_mom", "retail_sales_cum", "retail_sales_cum_yoy"]].copy()
            )
            if not retail_sales.empty
            else pd.DataFrame(columns=["date", "retail_sales_monthly", "retail_sales_yoy", "retail_sales_mom", "retail_sales_cum", "retail_sales_cum_yoy"])
        )
        enterprise_goods_price = (
            (
                enterprise_goods_price.assign(
                    date=(
                        pd.to_datetime(
                            enterprise_goods_price["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False),
                            errors="coerce",
                        ).dt.normalize()
                        if "date" not in enterprise_goods_price.columns and "月份" in enterprise_goods_price.columns
                        else enterprise_goods_price.get("date")
                    )
                )[["date", "enterprise_goods_price_index", "enterprise_goods_price_yoy", "enterprise_goods_price_mom", "energy_goods_price_index", "energy_goods_price_yoy", "energy_goods_price_mom"]].copy()
            )
            if not enterprise_goods_price.empty
            else pd.DataFrame(columns=["date", "enterprise_goods_price_index", "enterprise_goods_price_yoy", "enterprise_goods_price_mom", "energy_goods_price_index", "energy_goods_price_yoy", "energy_goods_price_mom"])
        )
        consumer_confidence = (
            (
                consumer_confidence.assign(
                    date=(
                        pd.to_datetime(
                            consumer_confidence["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False),
                            errors="coerce",
                        ).dt.normalize()
                        if "date" not in consumer_confidence.columns and "月份" in consumer_confidence.columns
                        else consumer_confidence.get("date")
                    )
                )[["date", "consumer_confidence_index", "consumer_confidence_yoy", "consumer_confidence_mom", "consumer_satisfaction_index", "consumer_satisfaction_yoy", "consumer_satisfaction_mom", "consumer_expectation_index", "consumer_expectation_yoy", "consumer_expectation_mom"]].copy()
            )
            if not consumer_confidence.empty
            else pd.DataFrame(columns=["date", "consumer_confidence_index", "consumer_confidence_yoy", "consumer_confidence_mom", "consumer_satisfaction_index", "consumer_satisfaction_yoy", "consumer_satisfaction_mom", "consumer_expectation_index", "consumer_expectation_yoy", "consumer_expectation_mom"])
        )
        money_supply = (
            (
                money_supply.assign(
                    date=(
                        pd.to_datetime(
                            money_supply["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False),
                            errors="coerce",
                        ).dt.normalize()
                        if "date" not in money_supply.columns and "月份" in money_supply.columns
                        else money_supply.get("date")
                    )
                )[["date", "m2_level", "m2_yoy", "m1_level", "m1_yoy", "m0_level", "m0_yoy"]].copy()
            )
            if not money_supply.empty
            else pd.DataFrame(columns=["date", "m2_level", "m2_yoy", "m1_level", "m1_yoy", "m0_level", "m0_yoy"])
        )
        new_credit = (
            (
                new_credit.assign(
                    date=(
                        pd.to_datetime(
                            new_credit["月份"].astype(str).str.replace("年", "-", regex=False).str.replace("月份", "-01", regex=False),
                            errors="coerce",
                        ).dt.normalize()
                        if "date" not in new_credit.columns and "月份" in new_credit.columns
                        else new_credit.get("date")
                    )
                )[["date", "new_financial_credit_monthly", "new_financial_credit_yoy", "new_financial_credit_cum"]].copy()
            )
            if not new_credit.empty
            else pd.DataFrame(columns=["date", "new_financial_credit_monthly", "new_financial_credit_yoy", "new_financial_credit_cum"])
        )
        bank_financing = (
            (
                bank_financing.assign(
                    date=(
                        pd.to_datetime(bank_financing["日期"], errors="coerce").dt.normalize()
                        if "date" not in bank_financing.columns and "日期" in bank_financing.columns
                        else bank_financing.get("date")
                    )
                )[["date", "bank_financing_index", "bank_financing_index_pct_chg"]].copy()
            )
            if not bank_financing.empty
            else pd.DataFrame(columns=["date", "bank_financing_index", "bank_financing_index_pct_chg"])
        )
        fx_reserves = (
            (
                fx_reserves.assign(
                    date=(
                        pd.to_datetime(fx_reserves["日期"], errors="coerce").dt.normalize()
                        if "date" not in fx_reserves.columns and "日期" in fx_reserves.columns
                        else fx_reserves.get("date")
                    )
                )[["date", "fx_reserves"]].copy()
            )
            if not fx_reserves.empty
            else pd.DataFrame(columns=["date", "fx_reserves"])
        )
        asphalt_inventory = (
            asphalt_inventory[["date", "asphalt_inventory", "asphalt_inventory_delta"]].copy()
            if not asphalt_inventory.empty
            else pd.DataFrame(columns=["date", "asphalt_inventory", "asphalt_inventory_delta"])
        )
        lfu_inventory = (
            lfu_inventory[["date", "lfu_inventory", "lfu_inventory_delta"]].copy()
            if not lfu_inventory.empty
            else pd.DataFrame(columns=["date", "lfu_inventory", "lfu_inventory_delta"])
        )
        fuel_oil_inventory = (
            fuel_oil_inventory[["date", "fuel_oil_inventory", "fuel_oil_inventory_delta"]].copy()
            if not fuel_oil_inventory.empty
            else pd.DataFrame(columns=["date", "fuel_oil_inventory", "fuel_oil_inventory_delta"])
        )
        rebar_inventory = (
            rebar_inventory[["date", "rebar_inventory", "rebar_inventory_delta"]].copy()
            if not rebar_inventory.empty
            else pd.DataFrame(columns=["date", "rebar_inventory", "rebar_inventory_delta"])
        )
        hotcoil_inventory = (
            hotcoil_inventory[["date", "hotcoil_inventory", "hotcoil_inventory_delta"]].copy()
            if not hotcoil_inventory.empty
            else pd.DataFrame(columns=["date", "hotcoil_inventory", "hotcoil_inventory_delta"])
        )
        coking_coal_inventory = (
            coking_coal_inventory[["date", "coking_coal_inventory", "coking_coal_inventory_delta"]].copy()
            if not coking_coal_inventory.empty
            else pd.DataFrame(columns=["date", "coking_coal_inventory", "coking_coal_inventory_delta"])
        )
        coke_inventory = (
            coke_inventory[["date", "coke_inventory", "coke_inventory_delta"]].copy()
            if not coke_inventory.empty
            else pd.DataFrame(columns=["date", "coke_inventory", "coke_inventory_delta"])
        )
        iron_ore_inventory = (
            iron_ore_inventory[["date", "iron_ore_inventory", "iron_ore_inventory_delta"]].copy()
            if not iron_ore_inventory.empty
            else pd.DataFrame(columns=["date", "iron_ore_inventory", "iron_ore_inventory_delta"])
        )
        society_traffic = (
            (
                society_traffic.assign(
                    date=(
                        pd.to_datetime(
                            society_traffic["统计时间"].astype(str).str.replace(".", "-", regex=False) + "-01",
                            errors="coerce",
                        ).dt.normalize()
                        if "date" not in society_traffic.columns and "统计时间" in society_traffic.columns
                        else society_traffic.get("date")
                    )
                )[
                    [
                        "date",
                        "cargo_volume",
                        "cargo_volume_yoy",
                        "cargo_turnover",
                        "cargo_turnover_yoy",
                        "coastal_port_throughput",
                        "coastal_port_throughput_yoy",
                        "coastal_port_foreign_trade_throughput",
                        "coastal_port_foreign_trade_throughput_yoy",
                    ]
                ].copy()
            )
            if not society_traffic.empty
            else pd.DataFrame(
                columns=[
                    "date",
                    "cargo_volume",
                    "cargo_volume_yoy",
                    "cargo_turnover",
                    "cargo_turnover_yoy",
                    "coastal_port_throughput",
                    "coastal_port_throughput_yoy",
                    "coastal_port_foreign_trade_throughput",
                    "coastal_port_foreign_trade_throughput_yoy",
                ]
            )
        )
        passenger = (
            (
                passenger.assign(
                    date=(
                        pd.to_datetime(passenger["统计时间"].astype(str).str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
                        if "date" not in passenger.columns and "统计时间" in passenger.columns
                        else passenger.get("date")
                    )
                )[["date", "passenger_load_factor", "cargo_load_factor"]].copy()
            )
            if not passenger.empty
            else pd.DataFrame(columns=["date", "passenger_load_factor", "cargo_load_factor"])
        )
        postal = (
            (
                postal.assign(
                    date=(
                        pd.to_datetime(postal["统计时间"].astype(str).str.replace(".", "-", regex=False) + "-01", errors="coerce").dt.normalize()
                        if "date" not in postal.columns and "统计时间" in postal.columns
                        else postal.get("date")
                    )
                )[["date", "express_delivery_volume", "express_delivery_yoy", "telecom_business_total", "telecom_business_yoy"]].copy()
            )
            if not postal.empty
            else pd.DataFrame(columns=["date", "express_delivery_volume", "express_delivery_yoy", "telecom_business_total", "telecom_business_yoy"])
        )
        glass_inventory = (
            glass_inventory[["date", "glass_inventory", "glass_inventory_delta"]].copy()
            if not glass_inventory.empty
            else pd.DataFrame(columns=["date", "glass_inventory", "glass_inventory_delta"])
        )
        soda_ash_inventory = (
            soda_ash_inventory[["date", "soda_ash_inventory", "soda_ash_inventory_delta"]].copy()
            if not soda_ash_inventory.empty
            else pd.DataFrame(columns=["date", "soda_ash_inventory", "soda_ash_inventory_delta"])
        )
        pvc_inventory = (
            pvc_inventory[["date", "pvc_inventory", "pvc_inventory_delta"]].copy()
            if not pvc_inventory.empty
            else pd.DataFrame(columns=["date", "pvc_inventory", "pvc_inventory_delta"])
        )
        pp_inventory = (
            pp_inventory[["date", "pp_inventory", "pp_inventory_delta"]].copy()
            if not pp_inventory.empty
            else pd.DataFrame(columns=["date", "pp_inventory", "pp_inventory_delta"])
        )
        methanol_inventory = (
            methanol_inventory[["date", "methanol_inventory", "methanol_inventory_delta"]].copy()
            if not methanol_inventory.empty
            else pd.DataFrame(columns=["date", "methanol_inventory", "methanol_inventory_delta"])
        )
        eg_inventory = (
            eg_inventory[["date", "eg_inventory", "eg_inventory_delta"]].copy()
            if not eg_inventory.empty
            else pd.DataFrame(columns=["date", "eg_inventory", "eg_inventory_delta"])
        )
        lpg_inventory = (
            lpg_inventory[["date", "lpg_inventory", "lpg_inventory_delta"]].copy()
            if not lpg_inventory.empty
            else pd.DataFrame(columns=["date", "lpg_inventory", "lpg_inventory_delta"])
        )

        macro = (
            cpi.merge(ppi, on="date", how="outer")
            .merge(lpr, on="date", how="outer")
            .merge(energy, on="date", how="outer")
            .merge(commodity, on="date", how="outer")
            .merge(energy_index, on="date", how="outer")
            .merge(freight, on="date", how="outer")
            .merge(oil_hist, on="date", how="outer")
            .merge(oil_detail, on="date", how="outer")
            .merge(construction_index, on="date", how="outer")
            .merge(construction_price, on="date", how="outer")
            .merge(real_estate, on="date", how="outer")
            .merge(society_electricity, on="date", how="outer")
            .merge(new_house_price, on="date", how="outer")
            .merge(industrial, on="date", how="outer")
            .merge(exports, on="date", how="outer")
            .merge(imports, on="date", how="outer")
            .merge(pmi, on="date", how="outer")
            .merge(non_man, on="date", how="outer")
            .merge(fixed_asset, on="date", how="outer")
            .merge(retail_sales, on="date", how="outer")
            .merge(enterprise_goods_price, on="date", how="outer")
            .merge(consumer_confidence, on="date", how="outer")
            .merge(money_supply, on="date", how="outer")
            .merge(new_credit, on="date", how="outer")
            .merge(bank_financing, on="date", how="outer")
            .merge(fx_reserves, on="date", how="outer")
            .merge(asphalt_inventory, on="date", how="outer")
            .merge(lfu_inventory, on="date", how="outer")
            .merge(fuel_oil_inventory, on="date", how="outer")
            .merge(rebar_inventory, on="date", how="outer")
            .merge(hotcoil_inventory, on="date", how="outer")
            .merge(coking_coal_inventory, on="date", how="outer")
            .merge(coke_inventory, on="date", how="outer")
            .merge(iron_ore_inventory, on="date", how="outer")
            .merge(society_traffic, on="date", how="outer")
            .merge(passenger, on="date", how="outer")
            .merge(postal, on="date", how="outer")
            .merge(glass_inventory, on="date", how="outer")
            .merge(soda_ash_inventory, on="date", how="outer")
            .merge(pvc_inventory, on="date", how="outer")
            .merge(pp_inventory, on="date", how="outer")
            .merge(methanol_inventory, on="date", how="outer")
            .merge(eg_inventory, on="date", how="outer")
            .merge(lpg_inventory, on="date", how="outer")
        )
        if macro.empty:
            return self._empty_macro_frame()
        macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()
        macro = macro.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        macro["cpi_yoy"] = pd.to_numeric(macro["cpi_yoy"], errors="coerce").ffill()
        macro["ppi_yoy"] = pd.to_numeric(macro["ppi_yoy"], errors="coerce").ffill()
        macro["lpr_1y"] = pd.to_numeric(macro["lpr_1y"], errors="coerce").ffill()
        macro["coastal_power_coal_inventory"] = pd.to_numeric(macro.get("coastal_power_coal_inventory"), errors="coerce")
        macro["coastal_power_coal_daily_burn"] = pd.to_numeric(macro.get("coastal_power_coal_daily_burn"), errors="coerce")
        macro["coastal_power_coal_days"] = pd.to_numeric(macro.get("coastal_power_coal_days"), errors="coerce")
        macro["commodity_price_index"] = pd.to_numeric(macro.get("commodity_price_index"), errors="coerce")
        macro["commodity_price_index_pct_chg"] = pd.to_numeric(macro.get("commodity_price_index_pct_chg"), errors="coerce")
        macro["energy_index"] = pd.to_numeric(macro.get("energy_index"), errors="coerce")
        macro["energy_index_pct_chg"] = pd.to_numeric(macro.get("energy_index_pct_chg"), errors="coerce")
        macro["bdi_index"] = pd.to_numeric(macro.get("bdi_index"), errors="coerce")
        macro["bcti_index"] = pd.to_numeric(macro.get("bcti_index"), errors="coerce")
        macro["bdti_index"] = pd.to_numeric(macro.get("bdti_index"), errors="coerce")
        macro["bsi_index"] = pd.to_numeric(macro.get("bsi_index"), errors="coerce")
        macro["gasoline_price"] = pd.to_numeric(macro.get("gasoline_price"), errors="coerce")
        macro["diesel_price"] = pd.to_numeric(macro.get("diesel_price"), errors="coerce")
        macro["gasoline_price_delta"] = pd.to_numeric(macro.get("gasoline_price_delta"), errors="coerce")
        macro["diesel_price_delta"] = pd.to_numeric(macro.get("diesel_price_delta"), errors="coerce")
        macro["diesel_price_regional_mean"] = pd.to_numeric(macro.get("diesel_price_regional_mean"), errors="coerce")
        macro["gasoline_92_price_regional_mean"] = pd.to_numeric(macro.get("gasoline_92_price_regional_mean"), errors="coerce")
        macro["gasoline_95_price_regional_mean"] = pd.to_numeric(macro.get("gasoline_95_price_regional_mean"), errors="coerce")
        macro["construction_index"] = pd.to_numeric(macro.get("construction_index"), errors="coerce")
        macro["construction_index_pct_chg"] = pd.to_numeric(macro.get("construction_index_pct_chg"), errors="coerce")
        macro["construction_price_index"] = pd.to_numeric(macro.get("construction_price_index"), errors="coerce")
        macro["construction_price_index_pct_chg"] = pd.to_numeric(macro.get("construction_price_index_pct_chg"), errors="coerce")
        macro["real_estate_index"] = pd.to_numeric(macro.get("real_estate_index"), errors="coerce")
        macro["real_estate_index_pct_chg"] = pd.to_numeric(macro.get("real_estate_index_pct_chg"), errors="coerce")
        macro["society_electricity_total"] = pd.to_numeric(macro.get("society_electricity_total"), errors="coerce")
        macro["society_electricity_yoy"] = pd.to_numeric(macro.get("society_electricity_yoy"), errors="coerce")
        macro["secondary_industry_electricity"] = pd.to_numeric(macro.get("secondary_industry_electricity"), errors="coerce")
        macro["secondary_industry_electricity_yoy"] = pd.to_numeric(macro.get("secondary_industry_electricity_yoy"), errors="coerce")
        macro["new_house_price_yoy"] = pd.to_numeric(macro.get("new_house_price_yoy"), errors="coerce")
        macro["new_house_price_mom"] = pd.to_numeric(macro.get("new_house_price_mom"), errors="coerce")
        macro["resale_house_price_yoy"] = pd.to_numeric(macro.get("resale_house_price_yoy"), errors="coerce")
        macro["resale_house_price_mom"] = pd.to_numeric(macro.get("resale_house_price_mom"), errors="coerce")
        macro["industrial_production_yoy"] = pd.to_numeric(macro.get("industrial_production_yoy"), errors="coerce")
        macro["exports_yoy"] = pd.to_numeric(macro.get("exports_yoy"), errors="coerce")
        macro["imports_yoy"] = pd.to_numeric(macro.get("imports_yoy"), errors="coerce")
        macro["pmi_manufacturing"] = pd.to_numeric(macro.get("pmi_manufacturing"), errors="coerce")
        macro["pmi_non_manufacturing"] = pd.to_numeric(macro.get("pmi_non_manufacturing"), errors="coerce")
        macro["fixed_asset_investment_monthly"] = pd.to_numeric(macro.get("fixed_asset_investment_monthly"), errors="coerce")
        macro["fixed_asset_investment_yoy"] = pd.to_numeric(macro.get("fixed_asset_investment_yoy"), errors="coerce")
        macro["fixed_asset_investment_mom"] = pd.to_numeric(macro.get("fixed_asset_investment_mom"), errors="coerce")
        macro["fixed_asset_investment_cum"] = pd.to_numeric(macro.get("fixed_asset_investment_cum"), errors="coerce")
        macro["retail_sales_monthly"] = pd.to_numeric(macro.get("retail_sales_monthly"), errors="coerce")
        macro["retail_sales_yoy"] = pd.to_numeric(macro.get("retail_sales_yoy"), errors="coerce")
        macro["retail_sales_mom"] = pd.to_numeric(macro.get("retail_sales_mom"), errors="coerce")
        macro["retail_sales_cum"] = pd.to_numeric(macro.get("retail_sales_cum"), errors="coerce")
        macro["retail_sales_cum_yoy"] = pd.to_numeric(macro.get("retail_sales_cum_yoy"), errors="coerce")
        macro["enterprise_goods_price_index"] = pd.to_numeric(macro.get("enterprise_goods_price_index"), errors="coerce")
        macro["enterprise_goods_price_yoy"] = pd.to_numeric(macro.get("enterprise_goods_price_yoy"), errors="coerce")
        macro["enterprise_goods_price_mom"] = pd.to_numeric(macro.get("enterprise_goods_price_mom"), errors="coerce")
        macro["energy_goods_price_index"] = pd.to_numeric(macro.get("energy_goods_price_index"), errors="coerce")
        macro["energy_goods_price_yoy"] = pd.to_numeric(macro.get("energy_goods_price_yoy"), errors="coerce")
        macro["energy_goods_price_mom"] = pd.to_numeric(macro.get("energy_goods_price_mom"), errors="coerce")
        macro["consumer_confidence_index"] = pd.to_numeric(macro.get("consumer_confidence_index"), errors="coerce")
        macro["consumer_confidence_yoy"] = pd.to_numeric(macro.get("consumer_confidence_yoy"), errors="coerce")
        macro["consumer_confidence_mom"] = pd.to_numeric(macro.get("consumer_confidence_mom"), errors="coerce")
        macro["consumer_satisfaction_index"] = pd.to_numeric(macro.get("consumer_satisfaction_index"), errors="coerce")
        macro["consumer_satisfaction_yoy"] = pd.to_numeric(macro.get("consumer_satisfaction_yoy"), errors="coerce")
        macro["consumer_satisfaction_mom"] = pd.to_numeric(macro.get("consumer_satisfaction_mom"), errors="coerce")
        macro["consumer_expectation_index"] = pd.to_numeric(macro.get("consumer_expectation_index"), errors="coerce")
        macro["consumer_expectation_yoy"] = pd.to_numeric(macro.get("consumer_expectation_yoy"), errors="coerce")
        macro["consumer_expectation_mom"] = pd.to_numeric(macro.get("consumer_expectation_mom"), errors="coerce")
        macro["m2_level"] = pd.to_numeric(macro.get("m2_level"), errors="coerce")
        macro["m2_yoy"] = pd.to_numeric(macro.get("m2_yoy"), errors="coerce")
        macro["m1_level"] = pd.to_numeric(macro.get("m1_level"), errors="coerce")
        macro["m1_yoy"] = pd.to_numeric(macro.get("m1_yoy"), errors="coerce")
        macro["m0_level"] = pd.to_numeric(macro.get("m0_level"), errors="coerce")
        macro["m0_yoy"] = pd.to_numeric(macro.get("m0_yoy"), errors="coerce")
        macro["new_financial_credit_monthly"] = pd.to_numeric(macro.get("new_financial_credit_monthly"), errors="coerce")
        macro["new_financial_credit_yoy"] = pd.to_numeric(macro.get("new_financial_credit_yoy"), errors="coerce")
        macro["new_financial_credit_cum"] = pd.to_numeric(macro.get("new_financial_credit_cum"), errors="coerce")
        macro["bank_financing_index"] = pd.to_numeric(macro.get("bank_financing_index"), errors="coerce")
        macro["bank_financing_index_pct_chg"] = pd.to_numeric(macro.get("bank_financing_index_pct_chg"), errors="coerce")
        macro["fx_reserves"] = pd.to_numeric(macro.get("fx_reserves"), errors="coerce")
        macro["asphalt_inventory"] = pd.to_numeric(macro.get("asphalt_inventory"), errors="coerce")
        macro["asphalt_inventory_delta"] = pd.to_numeric(macro.get("asphalt_inventory_delta"), errors="coerce")
        macro["lfu_inventory"] = pd.to_numeric(macro.get("lfu_inventory"), errors="coerce")
        macro["lfu_inventory_delta"] = pd.to_numeric(macro.get("lfu_inventory_delta"), errors="coerce")
        macro["fuel_oil_inventory"] = pd.to_numeric(macro.get("fuel_oil_inventory"), errors="coerce")
        macro["fuel_oil_inventory_delta"] = pd.to_numeric(macro.get("fuel_oil_inventory_delta"), errors="coerce")
        macro["rebar_inventory"] = pd.to_numeric(macro.get("rebar_inventory"), errors="coerce")
        macro["rebar_inventory_delta"] = pd.to_numeric(macro.get("rebar_inventory_delta"), errors="coerce")
        macro["hotcoil_inventory"] = pd.to_numeric(macro.get("hotcoil_inventory"), errors="coerce")
        macro["hotcoil_inventory_delta"] = pd.to_numeric(macro.get("hotcoil_inventory_delta"), errors="coerce")
        macro["coking_coal_inventory"] = pd.to_numeric(macro.get("coking_coal_inventory"), errors="coerce")
        macro["coking_coal_inventory_delta"] = pd.to_numeric(macro.get("coking_coal_inventory_delta"), errors="coerce")
        macro["coke_inventory"] = pd.to_numeric(macro.get("coke_inventory"), errors="coerce")
        macro["coke_inventory_delta"] = pd.to_numeric(macro.get("coke_inventory_delta"), errors="coerce")
        macro["iron_ore_inventory"] = pd.to_numeric(macro.get("iron_ore_inventory"), errors="coerce")
        macro["iron_ore_inventory_delta"] = pd.to_numeric(macro.get("iron_ore_inventory_delta"), errors="coerce")
        macro["cargo_volume"] = pd.to_numeric(macro.get("cargo_volume"), errors="coerce")
        macro["cargo_volume_yoy"] = pd.to_numeric(macro.get("cargo_volume_yoy"), errors="coerce")
        macro["cargo_turnover"] = pd.to_numeric(macro.get("cargo_turnover"), errors="coerce")
        macro["cargo_turnover_yoy"] = pd.to_numeric(macro.get("cargo_turnover_yoy"), errors="coerce")
        macro["coastal_port_throughput"] = pd.to_numeric(macro.get("coastal_port_throughput"), errors="coerce")
        macro["coastal_port_throughput_yoy"] = pd.to_numeric(macro.get("coastal_port_throughput_yoy"), errors="coerce")
        macro["coastal_port_foreign_trade_throughput"] = pd.to_numeric(macro.get("coastal_port_foreign_trade_throughput"), errors="coerce")
        macro["coastal_port_foreign_trade_throughput_yoy"] = pd.to_numeric(macro.get("coastal_port_foreign_trade_throughput_yoy"), errors="coerce")
        macro["passenger_load_factor"] = pd.to_numeric(macro.get("passenger_load_factor"), errors="coerce")
        macro["cargo_load_factor"] = pd.to_numeric(macro.get("cargo_load_factor"), errors="coerce")
        macro["express_delivery_volume"] = pd.to_numeric(macro.get("express_delivery_volume"), errors="coerce")
        macro["express_delivery_yoy"] = pd.to_numeric(macro.get("express_delivery_yoy"), errors="coerce")
        macro["telecom_business_total"] = pd.to_numeric(macro.get("telecom_business_total"), errors="coerce")
        macro["telecom_business_yoy"] = pd.to_numeric(macro.get("telecom_business_yoy"), errors="coerce")
        macro["glass_inventory"] = pd.to_numeric(macro.get("glass_inventory"), errors="coerce")
        macro["glass_inventory_delta"] = pd.to_numeric(macro.get("glass_inventory_delta"), errors="coerce")
        macro["soda_ash_inventory"] = pd.to_numeric(macro.get("soda_ash_inventory"), errors="coerce")
        macro["soda_ash_inventory_delta"] = pd.to_numeric(macro.get("soda_ash_inventory_delta"), errors="coerce")
        macro["pvc_inventory"] = pd.to_numeric(macro.get("pvc_inventory"), errors="coerce")
        macro["pvc_inventory_delta"] = pd.to_numeric(macro.get("pvc_inventory_delta"), errors="coerce")
        macro["pp_inventory"] = pd.to_numeric(macro.get("pp_inventory"), errors="coerce")
        macro["pp_inventory_delta"] = pd.to_numeric(macro.get("pp_inventory_delta"), errors="coerce")
        macro["methanol_inventory"] = pd.to_numeric(macro.get("methanol_inventory"), errors="coerce")
        macro["methanol_inventory_delta"] = pd.to_numeric(macro.get("methanol_inventory_delta"), errors="coerce")
        macro["eg_inventory"] = pd.to_numeric(macro.get("eg_inventory"), errors="coerce")
        macro["eg_inventory_delta"] = pd.to_numeric(macro.get("eg_inventory_delta"), errors="coerce")
        macro["lpg_inventory"] = pd.to_numeric(macro.get("lpg_inventory"), errors="coerce")
        macro["lpg_inventory_delta"] = pd.to_numeric(macro.get("lpg_inventory_delta"), errors="coerce")
        for col in [
            "coastal_power_coal_inventory",
            "coastal_power_coal_daily_burn",
            "coastal_power_coal_days",
            "commodity_price_index",
            "commodity_price_index_pct_chg",
            "energy_index",
            "energy_index_pct_chg",
            "bdi_index",
            "bcti_index",
            "bdti_index",
            "bsi_index",
            "gasoline_price",
            "diesel_price",
            "gasoline_price_delta",
            "diesel_price_delta",
            "diesel_price_regional_mean",
            "gasoline_92_price_regional_mean",
            "gasoline_95_price_regional_mean",
            "construction_index",
            "construction_index_pct_chg",
            "construction_price_index",
            "construction_price_index_pct_chg",
            "real_estate_index",
            "real_estate_index_pct_chg",
            "society_electricity_total",
            "society_electricity_yoy",
            "secondary_industry_electricity",
            "secondary_industry_electricity_yoy",
            "new_house_price_yoy",
            "new_house_price_mom",
            "resale_house_price_yoy",
            "resale_house_price_mom",
            "industrial_production_yoy",
            "exports_yoy",
            "imports_yoy",
            "pmi_manufacturing",
            "pmi_non_manufacturing",
            "fixed_asset_investment_monthly",
            "fixed_asset_investment_yoy",
            "fixed_asset_investment_mom",
            "fixed_asset_investment_cum",
            "retail_sales_monthly",
            "retail_sales_yoy",
            "retail_sales_mom",
            "retail_sales_cum",
            "retail_sales_cum_yoy",
            "enterprise_goods_price_index",
            "enterprise_goods_price_yoy",
            "enterprise_goods_price_mom",
            "energy_goods_price_index",
            "energy_goods_price_yoy",
            "energy_goods_price_mom",
            "consumer_confidence_index",
            "consumer_confidence_yoy",
            "consumer_confidence_mom",
            "consumer_satisfaction_index",
            "consumer_satisfaction_yoy",
            "consumer_satisfaction_mom",
            "consumer_expectation_index",
            "consumer_expectation_yoy",
            "consumer_expectation_mom",
            "m2_level",
            "m2_yoy",
            "m1_level",
            "m1_yoy",
            "m0_level",
            "m0_yoy",
            "new_financial_credit_monthly",
            "new_financial_credit_yoy",
            "new_financial_credit_cum",
            "bank_financing_index",
            "bank_financing_index_pct_chg",
            "fx_reserves",
            "asphalt_inventory",
            "asphalt_inventory_delta",
            "lfu_inventory",
            "lfu_inventory_delta",
            "fuel_oil_inventory",
            "fuel_oil_inventory_delta",
            "rebar_inventory",
            "rebar_inventory_delta",
            "hotcoil_inventory",
            "hotcoil_inventory_delta",
            "coking_coal_inventory",
            "coking_coal_inventory_delta",
            "coke_inventory",
            "coke_inventory_delta",
            "iron_ore_inventory",
            "iron_ore_inventory_delta",
            "cargo_volume",
            "cargo_volume_yoy",
            "cargo_turnover",
            "cargo_turnover_yoy",
            "coastal_port_throughput",
            "coastal_port_throughput_yoy",
            "coastal_port_foreign_trade_throughput",
            "coastal_port_foreign_trade_throughput_yoy",
            "passenger_load_factor",
            "cargo_load_factor",
            "express_delivery_volume",
            "express_delivery_yoy",
            "telecom_business_total",
            "telecom_business_yoy",
            "glass_inventory",
            "glass_inventory_delta",
            "soda_ash_inventory",
            "soda_ash_inventory_delta",
            "pvc_inventory",
            "pvc_inventory_delta",
            "pp_inventory",
            "pp_inventory_delta",
            "methanol_inventory",
            "methanol_inventory_delta",
            "eg_inventory",
            "eg_inventory_delta",
            "lpg_inventory",
            "lpg_inventory_delta",
        ]:
            macro[col] = pd.to_numeric(macro.get(col), errors="coerce").ffill()
        macro["source"] = self.name
        macro["cpi_source"] = macro["cpi_yoy"].apply(lambda v: "jin10:cpi_yoy" if np.isfinite(v) else None)
        macro["ppi_source"] = macro["ppi_yoy"].apply(lambda v: "jin10:ppi_yoy" if np.isfinite(v) else None)
        macro["lpr_source"] = macro["lpr_1y"].apply(lambda v: "eastmoney:lpr_1y" if np.isfinite(v) else None)
        macro["daily_energy_source"] = macro["coastal_power_coal_inventory"].apply(lambda v: "jin10:daily_energy" if np.isfinite(v) else None)
        macro["commodity_price_index_source"] = macro["commodity_price_index"].apply(lambda v: "eastmoney:commodity_price_index" if np.isfinite(v) else None)
        macro["energy_index_source"] = macro["energy_index"].apply(lambda v: "eastmoney:energy_index" if np.isfinite(v) else None)
        macro["freight_index_source"] = (
            macro[["bdi_index", "bcti_index", "bdti_index", "bsi_index"]]
            .apply(lambda row: "eastmoney:freight_index" if any(np.isfinite(pd.to_numeric(row, errors="coerce"))) else None, axis=1)
        )
        macro["oil_hist_source"] = macro["gasoline_price"].apply(lambda v: "eastmoney:oil_hist" if np.isfinite(v) else None)
        macro["oil_detail_source"] = macro["diesel_price_regional_mean"].apply(lambda v: "eastmoney:oil_detail" if np.isfinite(v) else None)
        macro["construction_index_source"] = macro["construction_index"].apply(lambda v: "eastmoney:construction_index" if np.isfinite(v) else None)
        macro["construction_price_index_source"] = macro["construction_price_index"].apply(lambda v: "eastmoney:construction_price_index" if np.isfinite(v) else None)
        macro["real_estate_index_source"] = macro["real_estate_index"].apply(lambda v: "eastmoney:real_estate_index" if np.isfinite(v) else None)
        macro["society_electricity_source"] = macro["society_electricity_total"].apply(lambda v: "nbs:society_electricity" if np.isfinite(v) else None)
        macro["new_house_price_source"] = macro["new_house_price_yoy"].apply(lambda v: "nbs:new_house_price" if np.isfinite(v) else None)
        macro["industrial_production_source"] = macro["industrial_production_yoy"].apply(lambda v: "jin10:industrial_production_yoy" if np.isfinite(v) else None)
        macro["exports_source"] = macro["exports_yoy"].apply(lambda v: "jin10:exports_yoy" if np.isfinite(v) else None)
        macro["imports_source"] = macro["imports_yoy"].apply(lambda v: "jin10:imports_yoy" if np.isfinite(v) else None)
        macro["pmi_source"] = macro["pmi_manufacturing"].apply(lambda v: "jin10:pmi_manufacturing" if np.isfinite(v) else None)
        macro["non_man_pmi_source"] = macro["pmi_non_manufacturing"].apply(lambda v: "jin10:non_man_pmi" if np.isfinite(v) else None)
        macro["fixed_asset_investment_source"] = macro["fixed_asset_investment_cum"].apply(lambda v: "nbs:fixed_asset_investment" if np.isfinite(v) else None)
        macro["retail_sales_source"] = macro["retail_sales_cum"].apply(lambda v: "nbs:consumer_goods_retail" if np.isfinite(v) else None)
        macro["enterprise_goods_price_source"] = macro["enterprise_goods_price_index"].apply(lambda v: "nbs:enterprise_goods_price" if np.isfinite(v) else None)
        macro["consumer_confidence_source"] = macro["consumer_confidence_index"].apply(lambda v: "nbs:consumer_confidence" if np.isfinite(v) else None)
        macro["money_supply_source"] = macro["m2_level"].apply(lambda v: "nbs:money_supply" if np.isfinite(v) else None)
        macro["new_financial_credit_source"] = macro["new_financial_credit_monthly"].apply(lambda v: "nbs:new_financial_credit" if np.isfinite(v) else None)
        macro["bank_financing_source"] = macro["bank_financing_index"].apply(lambda v: "eastmoney:bank_financing" if np.isfinite(v) else None)
        macro["fx_reserves_source"] = macro["fx_reserves"].apply(lambda v: "jin10:fx_reserves" if np.isfinite(v) else None)
        macro["asphalt_inventory_source"] = macro["asphalt_inventory"].apply(lambda v: "eastmoney:asphalt_inventory" if np.isfinite(v) else None)
        macro["lfu_inventory_source"] = macro["lfu_inventory"].apply(lambda v: "eastmoney:lfu_inventory" if np.isfinite(v) else None)
        macro["fuel_oil_inventory_source"] = macro["fuel_oil_inventory"].apply(lambda v: "eastmoney:fuel_oil_inventory" if np.isfinite(v) else None)
        macro["rebar_inventory_source"] = macro["rebar_inventory"].apply(lambda v: "eastmoney:rebar_inventory" if np.isfinite(v) else None)
        macro["hotcoil_inventory_source"] = macro["hotcoil_inventory"].apply(lambda v: "eastmoney:hotcoil_inventory" if np.isfinite(v) else None)
        macro["coking_coal_inventory_source"] = macro["coking_coal_inventory"].apply(lambda v: "eastmoney:coking_coal_inventory" if np.isfinite(v) else None)
        macro["coke_inventory_source"] = macro["coke_inventory"].apply(lambda v: "eastmoney:coke_inventory" if np.isfinite(v) else None)
        macro["iron_ore_inventory_source"] = macro["iron_ore_inventory"].apply(lambda v: "eastmoney:iron_ore_inventory" if np.isfinite(v) else None)
        macro["society_traffic_source"] = macro["cargo_volume"].apply(lambda v: "sina:society_traffic_volume" if np.isfinite(v) else None)
        macro["passenger_load_source"] = macro["passenger_load_factor"].apply(lambda v: "sina:passenger_load_factor" if np.isfinite(v) else None)
        macro["postal_telecom_source"] = macro["express_delivery_volume"].apply(lambda v: "sina:postal_telecom" if np.isfinite(v) else None)
        macro["glass_inventory_source"] = macro["glass_inventory"].apply(lambda v: "eastmoney:glass_inventory" if np.isfinite(v) else None)
        macro["soda_ash_inventory_source"] = macro["soda_ash_inventory"].apply(lambda v: "eastmoney:soda_ash_inventory" if np.isfinite(v) else None)
        macro["pvc_inventory_source"] = macro["pvc_inventory"].apply(lambda v: "eastmoney:pvc_inventory" if np.isfinite(v) else None)
        macro["pp_inventory_source"] = macro["pp_inventory"].apply(lambda v: "eastmoney:pp_inventory" if np.isfinite(v) else None)
        macro["methanol_inventory_source"] = macro["methanol_inventory"].apply(lambda v: "eastmoney:methanol_inventory" if np.isfinite(v) else None)
        macro["eg_inventory_source"] = macro["eg_inventory"].apply(lambda v: "eastmoney:eg_inventory" if np.isfinite(v) else None)
        macro["lpg_inventory_source"] = macro["lpg_inventory"].apply(lambda v: "eastmoney:lpg_inventory" if np.isfinite(v) else None)
        mask = (macro["date"].dt.date >= start) & (macro["date"].dt.date <= end)
        macro = macro.loc[
            mask,
            [
                "date",
                "cpi_yoy",
                "ppi_yoy",
                "lpr_1y",
                "source",
                "cpi_source",
                "ppi_source",
                "lpr_source",
                "coastal_power_coal_inventory",
                "coastal_power_coal_daily_burn",
                "coastal_power_coal_days",
                "commodity_price_index",
                "commodity_price_index_pct_chg",
                "energy_index",
                "energy_index_pct_chg",
                "bdi_index",
                "bcti_index",
                "bdti_index",
                "bsi_index",
                "gasoline_price",
                "diesel_price",
                "gasoline_price_delta",
                "diesel_price_delta",
                "diesel_price_regional_mean",
                "gasoline_92_price_regional_mean",
                "gasoline_95_price_regional_mean",
                "construction_index",
                "construction_index_pct_chg",
                "construction_price_index",
                "construction_price_index_pct_chg",
                "real_estate_index",
                "real_estate_index_pct_chg",
                "society_electricity_total",
                "society_electricity_yoy",
                "secondary_industry_electricity",
                "secondary_industry_electricity_yoy",
                "new_house_price_yoy",
                "new_house_price_mom",
                "resale_house_price_yoy",
                "resale_house_price_mom",
                "industrial_production_yoy",
                "exports_yoy",
                "imports_yoy",
                "pmi_manufacturing",
                "pmi_non_manufacturing",
                "fixed_asset_investment_monthly",
                "fixed_asset_investment_yoy",
                "fixed_asset_investment_mom",
                "fixed_asset_investment_cum",
                "retail_sales_monthly",
                "retail_sales_yoy",
                "retail_sales_mom",
                "retail_sales_cum",
                "retail_sales_cum_yoy",
                "enterprise_goods_price_index",
                "enterprise_goods_price_yoy",
                "enterprise_goods_price_mom",
                "energy_goods_price_index",
                "energy_goods_price_yoy",
                "energy_goods_price_mom",
                "consumer_confidence_index",
                "consumer_confidence_yoy",
                "consumer_confidence_mom",
                "consumer_satisfaction_index",
                "consumer_satisfaction_yoy",
                "consumer_satisfaction_mom",
                "consumer_expectation_index",
                "consumer_expectation_yoy",
                "consumer_expectation_mom",
                "m2_level",
                "m2_yoy",
                "m1_level",
                "m1_yoy",
                "m0_level",
                "m0_yoy",
                "new_financial_credit_monthly",
                "new_financial_credit_yoy",
                "new_financial_credit_cum",
                "bank_financing_index",
                "bank_financing_index_pct_chg",
                "fx_reserves",
                "asphalt_inventory",
                "asphalt_inventory_delta",
                "lfu_inventory",
                "lfu_inventory_delta",
                "fuel_oil_inventory",
                "fuel_oil_inventory_delta",
                "rebar_inventory",
                "rebar_inventory_delta",
                "hotcoil_inventory",
                "hotcoil_inventory_delta",
                "coking_coal_inventory",
                "coking_coal_inventory_delta",
                "coke_inventory",
                "coke_inventory_delta",
                "iron_ore_inventory",
                "iron_ore_inventory_delta",
                "cargo_volume",
                "cargo_volume_yoy",
                "cargo_turnover",
                "cargo_turnover_yoy",
                "coastal_port_throughput",
                "coastal_port_throughput_yoy",
                "coastal_port_foreign_trade_throughput",
                "coastal_port_foreign_trade_throughput_yoy",
                "passenger_load_factor",
                "cargo_load_factor",
                "express_delivery_volume",
                "express_delivery_yoy",
                "telecom_business_total",
                "telecom_business_yoy",
                "glass_inventory",
                "glass_inventory_delta",
                "soda_ash_inventory",
                "soda_ash_inventory_delta",
                "pvc_inventory",
                "pvc_inventory_delta",
                "pp_inventory",
                "pp_inventory_delta",
                "methanol_inventory",
                "methanol_inventory_delta",
                "eg_inventory",
                "eg_inventory_delta",
                "lpg_inventory",
                "lpg_inventory_delta",
                "daily_energy_source",
                "commodity_price_index_source",
                "energy_index_source",
                "freight_index_source",
                "oil_hist_source",
                "oil_detail_source",
                "construction_index_source",
                "construction_price_index_source",
                "real_estate_index_source",
                "society_electricity_source",
                "new_house_price_source",
                "industrial_production_source",
                "exports_source",
                "imports_source",
                "pmi_source",
                "non_man_pmi_source",
                "fixed_asset_investment_source",
                "retail_sales_source",
                "enterprise_goods_price_source",
                "consumer_confidence_source",
                "money_supply_source",
                "new_financial_credit_source",
                "bank_financing_source",
                "fx_reserves_source",
                "asphalt_inventory_source",
                "lfu_inventory_source",
                "fuel_oil_inventory_source",
                "rebar_inventory_source",
                "hotcoil_inventory_source",
                "coking_coal_inventory_source",
                "coke_inventory_source",
                "iron_ore_inventory_source",
                "society_traffic_source",
                "passenger_load_source",
                "postal_telecom_source",
                "glass_inventory_source",
                "soda_ash_inventory_source",
                "pvc_inventory_source",
                "pp_inventory_source",
                "methanol_inventory_source",
                "eg_inventory_source",
                "lpg_inventory_source",
            ],
        ]
        return macro.reset_index(drop=True) if not macro.empty else self._empty_macro_frame()

    def _fetch_baidu_calendar(self, current_date: date) -> pd.DataFrame:
        payload = self._http_json(
            "GET",
            "https://finance.pae.baidu.com/sapi/v1/financecalendar",
            params={
                "start_date": current_date.isoformat(),
                "end_date": current_date.isoformat(),
                "pn": "0",
                "rn": "100",
                "cate": "economic_data",
                "finClientType": "pc",
            },
            headers={
                "accept": "application/vnd.finance-web.v1+json",
                "origin": "https://gushitong.baidu.com",
                "referer": "https://gushitong.baidu.com/",
            },
        )
        calendar_info = payload.get("Result", {}).get("calendarInfo", []) if isinstance(payload, dict) else []
        rows: list[dict[str, Any]] = []
        for item in calendar_info:
            if not isinstance(item, dict):
                continue
            if str(item.get("date", "")) != current_date.isoformat():
                continue
            for row in item.get("list", []) if isinstance(item.get("list", []), list) else []:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "日期": pd.to_datetime(row.get("date"), errors="coerce").date(),
                        "时间": str(row.get("time", "") or "").strip(),
                        "地区": str(row.get("region", "") or "").strip(),
                        "事件": str(row.get("title", "") or "").strip(),
                        "公布": row.get("pubVal"),
                        "预期": row.get("indicateVal"),
                        "前值": row.get("formerVal"),
                        "重要性": pd.to_numeric(pd.Series([row.get("star")]), errors="coerce").iloc[0],
                    }
                )
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["日期", "时间", "地区", "事件", "公布", "预期", "前值", "重要性"])

    def _fetch_shmet_flash(self) -> pd.DataFrame:
        now_mono = time.monotonic()
        if self._shmet_cache is not None:
            cached_at, cached_df = self._shmet_cache
            if now_mono - float(cached_at) <= 120.0:
                return cached_df.copy()
        payload = self._http_json(
            "POST",
            "https://www.shmet.com/api/rest/news/queryNewsflashList",
            payload={"currentPage": 1, "pageSize": 100},
        )
        rows = payload.get("data", {}).get("dataList", []) if isinstance(payload, dict) else []
        out_rows: list[dict[str, Any]] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                push_time = pd.to_datetime(row.get("pushTime"), unit="ms", utc=True, errors="coerce")
                if pd.isna(push_time):
                    continue
                push_time = push_time.tz_convert("Asia/Shanghai")
                content = str(row.get("contentText", "") or row.get("content", "") or "").strip()
                if not content:
                    continue
                out_rows.append({"发布时间": push_time, "内容": content, "重要性": pd.to_numeric(pd.Series([row.get("starLevel")]), errors="coerce").iloc[0]})
        out = pd.DataFrame(out_rows) if out_rows else pd.DataFrame(columns=["发布时间", "内容", "重要性"])
        out = out.sort_values("发布时间").reset_index(drop=True)
        self._shmet_cache = (now_mono, out.copy())
        return out

    def _fetch_cctv_transcript(self, current_date: date) -> pd.DataFrame:
        try:
            import akshare as ak
        except Exception:
            return pd.DataFrame(columns=["date", "title", "content"])
        try:
            raw = ak.news_cctv(date=current_date.strftime("%Y%m%d"))
        except Exception:
            return pd.DataFrame(columns=["date", "title", "content"])
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["date", "title", "content"])
        out = raw.copy()
        cols = {str(c).lower(): str(c) for c in out.columns}
        date_col = cols.get("date", "date")
        title_col = cols.get("title", "title")
        content_col = cols.get("content", "content")
        if any(col not in out.columns for col in [date_col, title_col, content_col]):
            return pd.DataFrame(columns=["date", "title", "content"])
        out = out[[date_col, title_col, content_col]].rename(
            columns={date_col: "date", title_col: "title", content_col: "content"}
        )
        out["date"] = out["date"].astype(str)
        out["title"] = out["title"].astype(str)
        out["content"] = out["content"].astype(str)
        return out.dropna(subset=["title", "content"]).reset_index(drop=True)

    @staticmethod
    def _pick_news_title(text: str, prefix: str) -> str:
        clean = str(text or "").strip()
        match = re.search(r"【([^】]+)】", clean)
        headline = match.group(1).strip() if match else clean[:48].strip()
        if not headline:
            headline = prefix
        return f"{prefix} {headline}".strip()

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        src = str(text or "")
        mapping = [
            ("BTCUSDT", ("比特币", "BTC", "BTCUSDT")),
            ("ETHUSDT", ("以太坊", "ETH", "ETHUSDT")),
            ("BU2606", ("沥青", "BU", "BU2606")),
            ("原油", ("原油", "OPEC", "油价")),
            ("黄金", ("黄金",)),
            ("铜", ("铜",)),
        ]
        out: list[str] = []
        upper = src.upper()
        for label, keys in mapping:
            if any((k.upper() in upper) if re.search(r"[A-Za-z]", k) else (k in src) for k in keys):
                out.append(label)
        return out

    def fetch_news(self, start_ts: datetime, end_ts: datetime, lang: str) -> list[NewsEvent]:
        if str(lang).strip().lower() != "zh":
            return []

        effective_start = max(start_ts, end_ts - timedelta(days=max(0, int(self.max_news_lookback_days) - 1)))
        events: list[NewsEvent] = []
        current_date = effective_start.date()
        while current_date <= end_ts.date():
            try:
                baidu_df = self._fetch_baidu_calendar(current_date)
            except Exception:
                baidu_df = pd.DataFrame(columns=["日期", "时间", "地区", "事件", "公布", "预期", "前值", "重要性"])
            if not baidu_df.empty:
                for _, row in baidu_df.iterrows():
                    d = row.get("日期")
                    hhmm = str(row.get("时间", "") or "00:00").strip() or "00:00"
                    ts = pd.to_datetime(f"{d} {hhmm}", errors="coerce")
                    if pd.isna(ts):
                        continue
                    item_ts = ts.to_pydatetime()
                    if item_ts < effective_start or item_ts > end_ts:
                        continue
                    title = self._pick_news_title(str(row.get("事件", "")), "[Baidu]")
                    content = " | ".join(
                        x
                        for x in [
                            str(row.get("地区", "")).strip(),
                            str(row.get("事件", "")).strip(),
                            f"公布={row.get('公布')}" if row.get("公布") not in {None, "", "未公布"} else "",
                            f"预期={row.get('预期')}" if row.get("预期") not in {None, "", "未公布"} else "",
                            f"前值={row.get('前值')}" if row.get("前值") not in {None, "", "未公布"} else "",
                        ]
                        if x
                    )
                    importance_raw = pd.to_numeric(pd.Series([row.get("重要性")]), errors="coerce").iloc[0]
                    importance = 0.45 + 0.15 * float(max(0.0, min(3.0, importance_raw if np.isfinite(importance_raw) else 1.0)))
                    eid = hashlib.md5(f"{item_ts.isoformat()}|{title}|{content}|{self.name}".encode("utf-8"), usedforsecurity=False).hexdigest()
                    events.append(
                        NewsEvent(
                            event_id=eid,
                            ts=item_ts,
                            title=title,
                            content=content,
                            lang="zh",
                            source=self.name,
                            category="宏观",
                            confidence=0.82,
                            entities=self._extract_entities(f"{title} {content}"),
                            importance=float(min(1.0, max(0.0, importance))),
                        )
                    )
            try:
                cctv_df = self._fetch_cctv_transcript(current_date)
            except Exception:
                cctv_df = pd.DataFrame(columns=["date", "title", "content"])
            if not cctv_df.empty:
                for _, row in cctv_df.iterrows():
                    ts = pd.to_datetime(f"{current_date.isoformat()} 19:00", errors="coerce")
                    if pd.isna(ts):
                        continue
                    item_ts = ts.to_pydatetime()
                    if item_ts < effective_start or item_ts > end_ts:
                        continue
                    title = self._pick_news_title(str(row.get("title", "")), "[CCTV]")
                    content = str(row.get("content", "") or "").strip()
                    if not content:
                        continue
                    eid = hashlib.md5(f"{item_ts.isoformat()}|{title}|{content}|{self.name}".encode("utf-8"), usedforsecurity=False).hexdigest()
                    events.append(
                        NewsEvent(
                            event_id=eid,
                            ts=item_ts,
                            title=title,
                            content=content,
                            lang="zh",
                            source=self.name,
                            category="政策",
                            confidence=0.84,
                            entities=self._extract_entities(f"{title} {content}"),
                            importance=0.72,
                        )
                    )
            current_date += timedelta(days=1)

        try:
            shmet_df = self._fetch_shmet_flash()
        except Exception:
            shmet_df = pd.DataFrame(columns=["发布时间", "内容", "重要性"])
        if not shmet_df.empty:
            for _, row in shmet_df.iterrows():
                ts = pd.to_datetime(row.get("发布时间"), errors="coerce")
                if pd.isna(ts):
                    continue
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.tz_convert("Asia/Shanghai").tz_localize(None)
                item_ts = ts.to_pydatetime()
                if item_ts < effective_start or item_ts > end_ts:
                    continue
                content = str(row.get("内容", "") or "").strip()
                if not content:
                    continue
                title = self._pick_news_title(content, "[SHMET]")
                importance_raw = pd.to_numeric(pd.Series([row.get("重要性")]), errors="coerce").iloc[0]
                importance = 0.40 + 0.15 * float(max(0.0, min(3.0, importance_raw if np.isfinite(importance_raw) else 1.0)))
                eid = hashlib.md5(f"{item_ts.isoformat()}|{title}|{content}|{self.name}".encode("utf-8"), usedforsecurity=False).hexdigest()
                events.append(
                    NewsEvent(
                        event_id=eid,
                        ts=item_ts,
                        title=title,
                        content=content,
                        lang="zh",
                        source=self.name,
                        category="产业链",
                        confidence=0.74,
                        entities=self._extract_entities(f"{title} {content}"),
                        importance=float(min(1.0, max(0.0, importance))),
                    )
                )
        return sorted(events, key=lambda item: item.ts)

    def fetch_sentiment_factors(self, as_of: date) -> dict[str, float]:
        return {}

    def fetch_ohlcv(self, symbol: str, start: date, end: date, freq: str = "1d") -> pd.DataFrame:
        if str(freq).strip() != "1d":
            return _empty_ohlcv_frame()
        sym = re.sub(r"[^A-Za-z0-9]", "", str(symbol or "").upper())
        if not sym or _looks_like_crypto_pair(sym):
            return _empty_ohlcv_frame()

        try:
            from lie_engine.research.real_data import fetch_equity_daily, fetch_future_daily
        except Exception:
            return _empty_ohlcv_frame()

        raw = pd.DataFrame()
        try:
            if re.fullmatch(r"\d{6}", sym):
                raw = fetch_equity_daily(sym, start, end)
            elif re.fullmatch(r"[A-Z]{1,3}\d{4}", sym):
                raw = fetch_future_daily(sym, start, end)
            else:
                return _empty_ohlcv_frame()
        except Exception:
            return _empty_ohlcv_frame()

        if raw is None or raw.empty:
            return _empty_ohlcv_frame()

        out = raw.copy()
        out["ts"] = pd.to_datetime(out.get("ts"), errors="coerce")
        out = out.dropna(subset=["ts"]).copy()
        if out.empty:
            return _empty_ohlcv_frame()
        mask = (out["ts"].dt.date >= start) & (out["ts"].dt.date <= end)
        out = out.loc[mask].copy()
        if out.empty:
            return _empty_ohlcv_frame()

        for col in ["open", "high", "low", "close", "volume"]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce")
        out = out.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
        if out.empty:
            return _empty_ohlcv_frame()

        out["symbol"] = sym
        out["source_detail"] = out.get("source", pd.Series([self.name] * len(out), index=out.index)).astype(str)
        out["source"] = self.name
        if "asset_class" not in out.columns:
            out["asset_class"] = "future" if re.fullmatch(r"[A-Z]{1,3}\d{4}", sym) else "equity"
        out = out.sort_values("ts").drop_duplicates(subset=["ts", "symbol"]).reset_index(drop=True)
        return out[["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class", "source_detail"]]

    def fetch_l2(self, symbol: str, start_ts: datetime, end_ts: datetime, depth: int = 20) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not provide native L2 stream yet: {symbol}")

    def fetch_trades(self, symbol: str, start_ts: datetime, end_ts: datetime, limit: int = 2000) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} does not provide native trade ticks yet: {symbol}")


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
    user_agent: str = "lie-engine/0.1"
    _bucket: _TokenBucket = field(init=False, repr=False)
    _timeout_seconds: float = field(init=False, repr=False)
    _ssl_context: ssl.SSLContext = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cap = float(max(1, int(self.rate_limit_per_minute)))
        self._bucket = _TokenBucket(capacity=cap, refill_per_second=cap / 60.0)
        self._timeout_seconds = min(5.0, max(0.1, float(self.request_timeout_ms) / 1000.0))
        self._ssl_context = ssl.create_default_context()

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
            with request.urlopen(req, timeout=self._timeout_seconds, context=ctx) as resp:
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
            with request.urlopen(req, timeout=self._timeout_seconds, context=ctx) as resp:
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
        _ = (start, end)
        return pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"])

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
        sym = self._normalize_symbol(symbol)
        if (not sym) or (not _looks_like_crypto_pair(sym)):
            return _empty_l2_frame()
        limit = self._nearest_depth(depth)
        try:
            payload = self._http_get_json("/api/v3/depth", params={"symbol": sym, "limit": limit})
        except Exception:
            return _empty_l2_frame()
        if not isinstance(payload, dict):
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
            except Exception:
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
            except Exception:
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
    user_agent: str = "lie-engine/0.1"
    _bucket: _TokenBucket = field(init=False, repr=False)
    _timeout_seconds: float = field(init=False, repr=False)
    _ssl_context: ssl.SSLContext = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cap = float(max(1, int(self.rate_limit_per_minute)))
        self._bucket = _TokenBucket(capacity=cap, refill_per_second=cap / 60.0)
        self._timeout_seconds = min(5.0, max(0.1, float(self.request_timeout_ms) / 1000.0))
        self._ssl_context = ssl.create_default_context()

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
            with request.urlopen(req, timeout=self._timeout_seconds, context=ctx) as resp:
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
            with request.urlopen(req, timeout=self._timeout_seconds, context=ctx) as resp:
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
        _ = (start, end)
        return pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"])

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
        except Exception:
            return _empty_l2_frame()
        if not isinstance(payload, dict) or int(payload.get("retCode", 1)) != 0:
            return _empty_l2_frame()
        result = payload.get("result", {})
        if not isinstance(result, dict):
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
        except Exception:
            return _empty_trades_frame()
        if not isinstance(payload, dict) or int(payload.get("retCode", 1)) != 0:
            return _empty_trades_frame()
        result = payload.get("result", {})
        rows = result.get("list", []) if isinstance(result, dict) else []
        if not isinstance(rows, list):
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
