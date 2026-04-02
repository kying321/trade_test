from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
import akshare as ak

from lie_engine.data.providers import BinanceSpotPublicProvider, BybitSpotPublicProvider, PublicInternetResearchProvider
from lie_engine.models import NewsEvent


EQUITY_RE = re.compile(r"^\d{6}$")
FUTURE_RE = re.compile(r"^([A-Z]{1,3})\d{4}$")
CRYPTO_PAIR_RE = re.compile(r"^[A-Z0-9]{6,20}$")
CRYPTO_QUOTES = ("USDT", "USDC", "BUSD", "FDUSD", "BTC", "ETH", "BNB", "EUR")


POSITIVE_KWS = (
    "上调",
    "买入",
    "增持",
    "超预期",
    "增长",
    "回升",
    "突破",
    "利好",
    "缓和",
    "停火",
    "支撑",
    "去化",
    "回流",
    "改善",
    "回暖",
    "修复",
    "improve",
    "beat",
    "upgrade",
)
NEGATIVE_KWS = (
    "下调",
    "卖出",
    "减持",
    "不及预期",
    "下滑",
    "风险",
    "亏损",
    "利空",
    "袭击",
    "爆炸",
    "冲突",
    "制裁",
    "承压",
    "紧张",
    "断供",
    "升级",
    "downgrade",
    "miss",
    "warn",
)


_BINANCE_DAILY_PROVIDER_LOCK = threading.Lock()
_BINANCE_DAILY_PROVIDER: BinanceSpotPublicProvider | None = None
_BYBIT_DAILY_PROVIDER_LOCK = threading.Lock()
_BYBIT_DAILY_PROVIDER: BybitSpotPublicProvider | None = None
_PUBLIC_NEWS_PROVIDER_LOCK = threading.Lock()
_PUBLIC_NEWS_PROVIDER: PublicInternetResearchProvider | None = None
_PUBLIC_NEWS_CACHE_LOCK = threading.Lock()
_PUBLIC_NEWS_CACHE: dict[tuple[str, str, str], list[NewsEvent]] = {}


@dataclass(slots=True)
class RealDataBundle:
    bars: pd.DataFrame
    universe: list[str]
    news_daily: pd.Series
    report_daily: pd.Series
    news_records: int
    report_records: int
    fetch_stats: dict[str, Any]
    review_bars: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    news_daily_by_symbol: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "symbol", "news_score"]))
    report_daily_by_symbol: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "symbol", "report_score"]))
    cutoff_date: date | None = None
    review_days: int = 0
    review_news_daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    review_report_daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    review_news_daily_by_symbol: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "symbol", "news_score"]))
    review_report_daily_by_symbol: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["date", "symbol", "report_score"]))
    review_news_records: int = 0
    review_report_records: int = 0
    cutoff_ts: str = ""
    bar_max_ts: str = ""
    news_max_ts: str = ""
    report_max_ts: str = ""
    review_bar_max_ts: str = ""
    review_news_max_ts: str = ""
    review_report_max_ts: str = ""


def _date_end_iso(d: date | None) -> str:
    if d is None:
        return ""
    return f"{d.isoformat()}T23:59:59"


def _max_bar_ts_iso(bars: pd.DataFrame) -> str:
    if bars.empty or "ts" not in bars.columns:
        return ""
    ts = pd.to_datetime(bars["ts"], errors="coerce")
    ts = ts.dropna()
    if ts.empty:
        return ""
    return ts.max().to_pydatetime().replace(microsecond=0).isoformat()


def _max_daily_series_ts_iso(series: pd.Series) -> str:
    if series.empty:
        return ""
    idx = pd.to_datetime(pd.Index(series.index), errors="coerce")
    if len(idx) == 0:
        return ""
    ts = pd.Timestamp(idx.max())
    if pd.isna(ts):
        return ""
    return ts.to_pydatetime().replace(hour=23, minute=59, second=59, microsecond=0).isoformat()


def _retry_call(fn, retries: int = 3, base_sleep: float = 0.8):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(base_sleep * (i + 1))
    if last_err:
        raise last_err
    raise RuntimeError("unreachable")


def _normalize_symbol(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(raw or "").upper())


def _is_crypto_symbol(symbol: str) -> bool:
    sym = _normalize_symbol(symbol)
    if not CRYPTO_PAIR_RE.match(sym):
        return False
    return any(sym.endswith(q) and len(sym) > len(q) + 1 for q in CRYPTO_QUOTES)


def _get_binance_provider() -> BinanceSpotPublicProvider:
    global _BINANCE_DAILY_PROVIDER
    if _BINANCE_DAILY_PROVIDER is not None:
        return _BINANCE_DAILY_PROVIDER
    with _BINANCE_DAILY_PROVIDER_LOCK:
        if _BINANCE_DAILY_PROVIDER is None:
            _BINANCE_DAILY_PROVIDER = BinanceSpotPublicProvider(
                request_timeout_ms=5000,
                rate_limit_per_minute=10,
            )
    return _BINANCE_DAILY_PROVIDER


def _get_public_news_provider() -> PublicInternetResearchProvider:
    global _PUBLIC_NEWS_PROVIDER
    if _PUBLIC_NEWS_PROVIDER is not None:
        return _PUBLIC_NEWS_PROVIDER
    with _PUBLIC_NEWS_PROVIDER_LOCK:
        if _PUBLIC_NEWS_PROVIDER is None:
            _PUBLIC_NEWS_PROVIDER = PublicInternetResearchProvider(
                request_timeout_ms=5000,
                rate_limit_per_minute=60,
            )
    return _PUBLIC_NEWS_PROVIDER


def _get_bybit_provider() -> BybitSpotPublicProvider:
    global _BYBIT_DAILY_PROVIDER
    if _BYBIT_DAILY_PROVIDER is not None:
        return _BYBIT_DAILY_PROVIDER
    with _BYBIT_DAILY_PROVIDER_LOCK:
        if _BYBIT_DAILY_PROVIDER is None:
            _BYBIT_DAILY_PROVIDER = BybitSpotPublicProvider(
                request_timeout_ms=5000,
                rate_limit_per_minute=10,
            )
    return _BYBIT_DAILY_PROVIDER


def _symbol_to_yf(symbol: str) -> str:
    if symbol.startswith("6") or symbol.startswith("5"):
        return f"{symbol}.SS"
    if symbol.startswith(("0", "3", "1", "2")):
        return f"{symbol}.SZ"
    return symbol


def _normalize_ohlcv(df: pd.DataFrame, symbol: str, asset_class: str, ts_col: str, open_col: str, high_col: str, low_col: str, close_col: str, volume_col: str, source: str) -> pd.DataFrame:
    def _pick(col: str) -> pd.Series:
        v = df[col]
        if isinstance(v, pd.DataFrame):
            return v.iloc[:, 0]
        return v

    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(_pick(ts_col)),
            "symbol": symbol,
            "open": pd.to_numeric(_pick(open_col), errors="coerce"),
            "high": pd.to_numeric(_pick(high_col), errors="coerce"),
            "low": pd.to_numeric(_pick(low_col), errors="coerce"),
            "close": pd.to_numeric(_pick(close_col), errors="coerce"),
            "volume": pd.to_numeric(_pick(volume_col), errors="coerce"),
            "source": source,
            "asset_class": asset_class,
        }
    )
    out = out.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def fetch_equity_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")

    def _ak():
        raw = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_s, end_date=end_s, adjust="qfq")
        return _normalize_ohlcv(
            raw,
            symbol=symbol,
            asset_class="etf" if symbol.startswith("5") else "equity",
            ts_col="日期",
            open_col="开盘",
            high_col="最高",
            low_col="最低",
            close_col="收盘",
            volume_col="成交量",
            source="akshare.stock_zh_a_hist",
        )

    try:
        return _retry_call(_ak, retries=3)
    except Exception:
        ticker = _symbol_to_yf(symbol)
        raw = yf.download(ticker, start=str(start), end=str(end), interval="1d", auto_adjust=False, progress=False, group_by="column")
        if raw.empty:
            return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [str(c[0]) for c in raw.columns]
        raw = raw.reset_index()
        raw.columns = [str(c) for c in raw.columns]
        cols = {str(c).lower(): str(c) for c in raw.columns}
        date_col = cols.get("date", "Date")
        open_col = cols.get("open", "Open")
        high_col = cols.get("high", "High")
        low_col = cols.get("low", "Low")
        close_col = cols.get("close", "Close")
        volume_col = cols.get("volume", "Volume")
        if any(c not in raw.columns for c in [date_col, open_col, high_col, low_col, close_col, volume_col]):
            return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])
        return _normalize_ohlcv(
            raw,
            symbol=symbol,
            asset_class="etf" if symbol.startswith("5") else "equity",
            ts_col=date_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
            source=f"yfinance:{ticker}",
        )


def _future_cont_symbol(symbol: str) -> str:
    m = FUTURE_RE.match(symbol)
    if not m:
        return symbol
    return f"{m.group(1)}0"


def fetch_future_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    cont = _future_cont_symbol(symbol)

    def _ak():
        raw = ak.futures_zh_daily_sina(symbol=cont)
        out = _normalize_ohlcv(
            raw,
            symbol=symbol,
            asset_class="future",
            ts_col="date",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col="volume",
            source=f"akshare.futures_zh_daily_sina:{cont}",
        )
        return out[(out["ts"].dt.date >= start) & (out["ts"].dt.date <= end)].reset_index(drop=True)

    return _retry_call(_ak, retries=3)


def fetch_crypto_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    sym = _normalize_symbol(symbol)
    if not _is_crypto_symbol(sym):
        return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])
    providers = [_get_binance_provider(), _get_bybit_provider()]
    for provider in providers:
        try:
            df = provider.fetch_ohlcv(symbol=sym, start=start, end=end, freq="1d")
        except Exception:
            df = None
        if df is None or df.empty:
            continue
        out = df.copy()
        out["symbol"] = sym
        out["asset_class"] = "crypto"
        return out
    return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])


def _fetch_one_symbol(symbol: str, start: date, end: date) -> tuple[str, pd.DataFrame, str | None]:
    normalized_symbol = _normalize_symbol(symbol)
    try:
        if EQUITY_RE.match(normalized_symbol):
            df = fetch_equity_daily(normalized_symbol, start, end)
            return normalized_symbol, df, None
        if FUTURE_RE.match(normalized_symbol):
            df = fetch_future_daily(normalized_symbol, start, end)
            return normalized_symbol, df, None
        if _is_crypto_symbol(normalized_symbol):
            df = fetch_crypto_daily(normalized_symbol, start, end)
            return normalized_symbol, df, None
        return normalized_symbol, pd.DataFrame(), f"unsupported_symbol:{normalized_symbol}"
    except Exception as exc:  # noqa: BLE001
        return normalized_symbol, pd.DataFrame(), f"{type(exc).__name__}:{exc}"


def _stock_news_score(title: str, content: str) -> float:
    txt = f"{title} {content}".lower()
    pos = sum(1 for k in POSITIVE_KWS if k.lower() in txt)
    neg = sum(1 for k in NEGATIVE_KWS if k.lower() in txt)
    if pos == neg:
        return 0.0
    return float((pos - neg) / max(1, pos + neg))


def _extract_labeled_numeric(text: str, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        pattern = re.compile(rf"{re.escape(label)}\s*[=:：]\s*([-+]?(?:\d+(?:\.\d+)?|nan))", re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            continue
        raw = str(match.group(1)).strip()
        if raw.lower() == "nan":
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return None


def _inventory_event_direction(title: str, content: str) -> float:
    text = f"{title} {content}"
    if not any(k in text for k in ("仓单变动", "仓单", "库存", "去库", "去化", "累库")):
        return 0.0
    if any(k in text for k in ("去库", "去化", "库存下降", "库存减少", "仓单减少", "仓单下降")):
        return 1.0
    if any(k in text for k in ("累库", "库存增加", "库存上升", "仓单增加", "仓单上升")):
        return -1.0
    delta = _extract_labeled_numeric(text, ("公布", "今值", "实际", "前值", "变动"))
    if delta is None or abs(delta) <= 1e-12:
        return 0.0
    return 1.0 if delta < 0 else -1.0


def _symbol_aliases(symbol: str) -> list[str]:
    sym = _normalize_symbol(symbol)
    aliases = [sym]
    if _is_crypto_symbol(sym):
        for quote in CRYPTO_QUOTES:
            if sym.endswith(quote) and len(sym) > len(quote) + 1:
                base = sym[: -len(quote)]
                aliases.append(base)
                crypto_alias_map = {
                    "BTC": ["比特币"],
                    "ETH": ["以太坊"],
                    "BNB": ["币安币"],
                    "SOL": ["SOLANA", "索拉纳"],
                    "XRP": ["瑞波"],
                }
                aliases.extend(crypto_alias_map.get(base, []))
                break
    elif FUTURE_RE.match(sym):
        root = FUTURE_RE.match(sym).group(1)  # type: ignore[union-attr]
        aliases.append(root)
        future_alias_map = {
            "BU": ["沥青"],
            "SC": ["原油"],
            "AU": ["黄金"],
            "CU": ["铜"],
            "RB": ["螺纹"],
        }
        aliases.extend(future_alias_map.get(root, []))
    return list(dict.fromkeys([str(x).strip() for x in aliases if str(x).strip()]))


def _load_public_news_events(start: date, end: date, lang: str = "zh") -> list[NewsEvent]:
    key = (start.isoformat(), end.isoformat(), str(lang).strip().lower())
    with _PUBLIC_NEWS_CACHE_LOCK:
        cached = _PUBLIC_NEWS_CACHE.get(key)
    if cached is not None:
        return list(cached)

    provider = _get_public_news_provider()
    events = provider.fetch_news(
        start_ts=datetime.combine(start, datetime.min.time()),
        end_ts=datetime.combine(end, datetime.max.time()).replace(microsecond=0),
        lang=lang,
    )
    out = list(events or [])
    with _PUBLIC_NEWS_CACHE_LOCK:
        _PUBLIC_NEWS_CACHE[key] = list(out)
    return out


def _load_public_macro_frame(start: date, end: date) -> pd.DataFrame:
    provider = _get_public_news_provider()
    try:
        macro = provider.fetch_macro(start=start, end=end)
    except Exception:
        return pd.DataFrame()
    if not isinstance(macro, pd.DataFrame) or macro.empty:
        return pd.DataFrame()
    return macro.copy()


def _attach_macro_features_to_bars(bars_df: pd.DataFrame, macro_df: pd.DataFrame | None) -> pd.DataFrame:
    if bars_df.empty or macro_df is None or macro_df.empty or "ts" not in bars_df.columns or "date" not in macro_df.columns:
        return bars_df

    macro = macro_df.copy()
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce").dt.normalize()
    macro = macro.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if macro.empty:
        return bars_df

    macro = macro.drop(columns=[c for c in macro.columns if c == "source" or str(c).endswith("_source")], errors="ignore")
    bars = bars_df.copy()
    bars["date"] = pd.to_datetime(bars["ts"], errors="coerce").dt.normalize()
    bars["__orig_order__"] = np.arange(len(bars))
    macro_cols = ["date"] + [c for c in macro.columns if c != "date" and c not in bars.columns]
    if len(macro_cols) == 1:
        return bars_df
    macro = macro[macro_cols].sort_values("date")

    merged = pd.merge_asof(
        bars.sort_values("date"),
        macro,
        on="date",
        direction="backward",
    )
    return merged.sort_values("__orig_order__").drop(columns=["__orig_order__", "date"], errors="ignore").reset_index(drop=True)


def _aggregate_daily_score_frame(df: pd.DataFrame, value_col: str) -> pd.Series:
    if df.empty or "date" not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=float)

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=["date", value_col])
    if frame.empty:
        return pd.Series(dtype=float)

    def _agg(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return 0.0
        nonzero = s[s.abs() > 1e-12]
        target = nonzero if not nonzero.empty else s
        return float(target.mean())

    return frame.groupby("date")[value_col].apply(_agg).sort_index()


def _aggregate_symbol_daily_score_frame(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty or "date" not in df.columns or "symbol" not in df.columns or value_col not in df.columns:
        return pd.DataFrame(columns=["date", "symbol", value_col])

    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame["symbol"] = frame["symbol"].astype(str)
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol", value_col])
    if frame.empty:
        return pd.DataFrame(columns=["date", "symbol", value_col])

    def _agg(s: pd.Series) -> float:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return 0.0
        nonzero = s[s.abs() > 1e-12]
        target = nonzero if not nonzero.empty else s
        return float(target.mean())

    out = frame.groupby(["date", "symbol"])[value_col].apply(_agg).reset_index(name=value_col)
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def resolve_factor_series_for_bars(
    *,
    bars: pd.DataFrame,
    aggregate_daily: pd.Series,
    by_symbol_daily: pd.DataFrame | None,
    value_col: str,
) -> pd.Series:
    if bars is None or bars.empty or "ts" not in bars.columns:
        return aggregate_daily.astype(float).sort_index() if not aggregate_daily.empty else pd.Series(dtype=float)

    bar_keys = bars.copy()
    bar_keys["date"] = pd.to_datetime(bar_keys["ts"], errors="coerce").dt.date
    bar_keys["symbol"] = bar_keys["symbol"].astype(str)
    bar_keys = bar_keys.dropna(subset=["date", "symbol"])[["date", "symbol"]].drop_duplicates()
    if bar_keys.empty:
        return aggregate_daily.astype(float).sort_index() if not aggregate_daily.empty else pd.Series(dtype=float)

    if by_symbol_daily is None or by_symbol_daily.empty or value_col not in by_symbol_daily.columns:
        if aggregate_daily.empty:
            return pd.Series(dtype=float)
        idx = sorted(bar_keys["date"].unique().tolist())
        return aggregate_daily.astype(float).reindex(idx).fillna(0.0)

    factor = by_symbol_daily.copy()
    factor["date"] = pd.to_datetime(factor["date"], errors="coerce").dt.date
    factor["symbol"] = factor["symbol"].astype(str)
    factor[value_col] = pd.to_numeric(factor[value_col], errors="coerce")
    factor = factor.dropna(subset=["date", "symbol", value_col])
    if factor.empty:
        idx = sorted(bar_keys["date"].unique().tolist())
        return aggregate_daily.astype(float).reindex(idx).fillna(0.0) if not aggregate_daily.empty else pd.Series(0.0, index=idx, dtype=float)

    merged = bar_keys.merge(factor[["date", "symbol", value_col]], on=["date", "symbol"], how="left")
    if merged[value_col].notna().any():
        out = merged.groupby("date")[value_col].mean().sort_index()
        return out.fillna(0.0).astype(float)

    idx = sorted(bar_keys["date"].unique().tolist())
    return aggregate_daily.astype(float).reindex(idx).fillna(0.0) if not aggregate_daily.empty else pd.Series(0.0, index=idx, dtype=float)


def resolve_factor_series_for_exposure(
    *,
    aggregate_daily: pd.Series,
    by_symbol_daily: pd.DataFrame | None,
    daily_symbol_exposure: list[dict[str, Any]] | None,
    value_col: str,
) -> pd.Series:
    if not daily_symbol_exposure:
        return aggregate_daily.astype(float).sort_index() if not aggregate_daily.empty else pd.Series(dtype=float)

    exposure = pd.DataFrame(list(daily_symbol_exposure or []))
    if exposure.empty or not {"date", "symbol", "weight"}.issubset(set(exposure.columns)):
        return aggregate_daily.astype(float).sort_index() if not aggregate_daily.empty else pd.Series(dtype=float)
    exposure["date"] = pd.to_datetime(exposure["date"], errors="coerce").dt.date
    exposure["symbol"] = exposure["symbol"].astype(str)
    exposure["weight"] = pd.to_numeric(exposure["weight"], errors="coerce")
    exposure = exposure.dropna(subset=["date", "symbol", "weight"])
    if exposure.empty:
        return aggregate_daily.astype(float).sort_index() if not aggregate_daily.empty else pd.Series(dtype=float)

    if by_symbol_daily is None or by_symbol_daily.empty or value_col not in by_symbol_daily.columns:
        idx = sorted(exposure["date"].unique().tolist())
        return aggregate_daily.astype(float).reindex(idx).fillna(0.0) if not aggregate_daily.empty else pd.Series(0.0, index=idx, dtype=float)

    factor = by_symbol_daily.copy()
    factor["date"] = pd.to_datetime(factor["date"], errors="coerce").dt.date
    factor["symbol"] = factor["symbol"].astype(str)
    factor[value_col] = pd.to_numeric(factor[value_col], errors="coerce")
    factor = factor.dropna(subset=["date", "symbol", value_col])
    if factor.empty:
        idx = sorted(exposure["date"].unique().tolist())
        return aggregate_daily.astype(float).reindex(idx).fillna(0.0) if not aggregate_daily.empty else pd.Series(0.0, index=idx, dtype=float)

    merged = exposure.merge(factor[["date", "symbol", value_col]], on=["date", "symbol"], how="left")
    merged[value_col] = merged[value_col].fillna(0.0)
    out = (
        merged.assign(weighted_value=merged["weight"] * merged[value_col])
        .groupby("date", as_index=True)["weighted_value"]
        .sum()
        .sort_index()
    )
    return out.astype(float)


def _future_chain_keywords(root: str) -> list[str]:
    mapping = {
        "BU": ["沥青", "原油", "油价", "燃料油", "成品油", "OPEC", "伊朗", "中东"],
        "SC": ["原油", "油价", "OPEC", "伊朗", "中东"],
        "AU": ["黄金", "金价", "美债", "美元", "避险"],
        "CU": ["铜", "精铜", "沪铜"],
        "RB": ["螺纹", "钢材", "热卷"],
    }
    return mapping.get(str(root).upper(), [])


def _public_news_match_relevance(event: NewsEvent, symbol: str) -> float:
    aliases = _symbol_aliases(symbol)
    entity_tokens = {_normalize_symbol(x) for x in list(event.entities or []) if _normalize_symbol(x)}
    normalized_aliases = [_normalize_symbol(alias) for alias in aliases if _normalize_symbol(alias)]
    if any(alias in entity_tokens for alias in normalized_aliases):
        return 1.0
    text_upper = f"{event.title} {event.content}".upper()
    text_raw = f"{event.title} {event.content}"
    for alias in aliases:
        if re.search(r"[A-Za-z0-9]", alias):
            if alias.upper() in text_upper:
                return 1.0
        elif alias in text_raw:
            return 1.0

    sym = _normalize_symbol(symbol)
    category = str(event.category or "").strip()
    if FUTURE_RE.match(sym):
        root = FUTURE_RE.match(sym).group(1)  # type: ignore[union-attr]
        keys = _future_chain_keywords(root)
        if any((k.upper() in text_upper) if re.search(r"[A-Za-z]", k) else (k in text_raw) for k in keys):
            return 0.75
        return 0.0

    if _is_crypto_symbol(sym):
        crypto_macro_keys = ["伊朗", "美国", "爆炸", "袭击", "制裁", "停火", "谈判", "风险偏好", "避险", "流动性"]
        if any((k.upper() in text_upper) if re.search(r"[A-Za-z]", k) else (k in text_raw) for k in crypto_macro_keys):
            return 0.45

    if _is_crypto_symbol(sym):
        if category in {"宏观", "地缘", "政策"}:
            return 0.45
    return 0.0


def _public_news_matches_symbol(event: NewsEvent, symbol: str) -> bool:
    return _public_news_match_relevance(event, symbol) > 0.0


def _public_report_match_relevance(event: NewsEvent, symbol: str) -> float:
    relevance = _public_news_match_relevance(event, symbol)
    if relevance > 0.0:
        return relevance
    sym = _normalize_symbol(symbol)
    category = str(event.category or "").strip()
    if (_is_crypto_symbol(sym) or FUTURE_RE.match(sym)) and category in {"宏观", "地缘", "政策"}:
        return 0.45
    return 0.0


def _public_news_score(event: NewsEvent, *, relevance: float = 1.0) -> float:
    raw = _stock_news_score(event.title, event.content)
    if abs(raw) <= 1e-12:
        raw = _inventory_event_direction(event.title, event.content)
    if abs(raw) <= 1e-12:
        return 0.0
    scale = (
        0.60
        + 0.25 * float(max(0.0, min(1.0, event.confidence)))
        + 0.15 * float(max(0.0, min(1.0, event.importance)))
    ) * float(max(0.0, min(1.0, relevance)))
    return float(np.clip(raw * scale, -2.0, 2.0))


def _is_public_report_event(event: NewsEvent) -> bool:
    title = str(event.title or "").strip()
    content = str(event.content or "").strip()
    category = str(event.category or "").strip()
    if title.startswith("[CCTV]"):
        return True
    if category in {"政策", "宏观"} and len(content) >= 120:
        return True
    return False


def _public_report_score(event: NewsEvent, *, relevance: float = 1.0) -> float:
    raw = _stock_news_score(event.title, event.content)
    if abs(raw) <= 1e-12:
        return 0.0
    scale = (
        0.75
        + 0.15 * float(max(0.0, min(1.0, event.confidence)))
        + 0.10 * float(max(0.0, min(1.0, event.importance)))
    ) * float(max(0.0, min(1.0, relevance)))
    return float(np.clip(raw * scale, -2.0, 2.0))


def _report_score(row: dict[str, Any]) -> float:
    rating = str(row.get("东财评级", "") or "")
    base = 0.0
    if "买入" in rating:
        base = 1.0
    elif "增持" in rating:
        base = 0.6
    elif "中性" in rating:
        base = 0.0
    elif "减持" in rating:
        base = -0.6
    elif "卖出" in rating:
        base = -1.0

    y25 = pd.to_numeric(pd.Series([row.get("2025-盈利预测-收益")]), errors="coerce").iloc[0]
    y26 = pd.to_numeric(pd.Series([row.get("2026-盈利预测-收益")]), errors="coerce").iloc[0]
    growth = 0.0
    if np.isfinite(y25) and np.isfinite(y26) and abs(float(y25)) > 1e-9:
        growth = float(np.clip((float(y26) - float(y25)) / abs(float(y25)), -2.0, 2.0))
    return float(np.clip(base + 0.25 * growth, -2.0, 2.0))


def fetch_symbol_news_and_reports(symbol: str, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_symbol = _normalize_symbol(symbol)
    if not EQUITY_RE.match(normalized_symbol):
        if not (FUTURE_RE.match(normalized_symbol) or _is_crypto_symbol(normalized_symbol)):
            return pd.DataFrame(columns=["date", "symbol", "news_score"]), pd.DataFrame(columns=["date", "symbol", "report_score"])
        news_records: list[dict[str, Any]] = []
        report_records: list[dict[str, Any]] = []
        try:
            for event in _load_public_news_events(start=start, end=end, lang="zh"):
                d = event.ts.date()
                if d < start or d > end:
                    continue
                if _is_public_report_event(event):
                    relevance = _public_report_match_relevance(event, normalized_symbol)
                    if relevance <= 0.0:
                        continue
                    score = _public_report_score(event, relevance=relevance)
                    if abs(score) <= 1e-12:
                        continue
                    report_records.append({"date": d, "symbol": normalized_symbol, "report_score": score})
                    continue
                relevance = _public_news_match_relevance(event, normalized_symbol)
                if relevance <= 0.0:
                    continue
                score = _public_news_score(event, relevance=relevance)
                news_records.append({"date": d, "symbol": normalized_symbol, "news_score": score})
        except Exception:
            pass
        news_out = pd.DataFrame(news_records) if news_records else pd.DataFrame(columns=["date", "symbol", "news_score"])
        report_out = pd.DataFrame(report_records) if report_records else pd.DataFrame(columns=["date", "symbol", "report_score"])
        return news_out, report_out

    news_records: list[dict[str, Any]] = []
    report_records: list[dict[str, Any]] = []

    try:
        news_df = _retry_call(lambda: ak.stock_news_em(symbol=symbol), retries=2)
        if not news_df.empty:
            for row in news_df.to_dict("records"):
                ts = pd.to_datetime(row.get("发布时间"), errors="coerce")
                if pd.isna(ts):
                    continue
                d = ts.date()
                if d < start or d > end:
                    continue
                score = _stock_news_score(str(row.get("新闻标题", "")), str(row.get("新闻内容", "")))
                news_records.append({"date": d, "symbol": symbol, "news_score": score})
    except Exception:
        pass

    try:
        report_df = _retry_call(lambda: ak.stock_research_report_em(symbol=symbol), retries=2)
        if not report_df.empty:
            for row in report_df.to_dict("records"):
                d = pd.to_datetime(row.get("日期"), errors="coerce")
                if pd.isna(d):
                    continue
                dd = d.date()
                if dd < start or dd > end:
                    continue
                score = _report_score(row)
                report_records.append({"date": dd, "symbol": symbol, "report_score": score})
    except Exception:
        pass

    news_out = pd.DataFrame(news_records) if news_records else pd.DataFrame(columns=["date", "symbol", "news_score"])
    report_out = pd.DataFrame(report_records) if report_records else pd.DataFrame(columns=["date", "symbol", "report_score"])
    return news_out, report_out


def load_universe(core_symbols: list[str], max_symbols: int = 120) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in core_symbols:
        sym = _normalize_symbol(s)
        if (not sym) or (sym in seen):
            continue
        out.append(sym)
        seen.add(sym)

    # Crypto mode: keep user core symbols only, avoid equity index contamination.
    if out and all(_is_crypto_symbol(s) for s in out):
        return out[:max_symbols]

    candidates: list[str] = []
    for idx in ("000300", "000905"):
        try:
            cons = ak.index_stock_cons_csindex(symbol=idx)
            if not cons.empty:
                candidates.extend(cons["成分券代码"].astype(str).tolist())
        except Exception:
            continue

    for s in candidates:
        if len(out) >= max_symbols:
            break
        if s in seen:
            continue
        if not EQUITY_RE.match(s):
            continue
        out.append(s)
        seen.add(s)
    return out[:max_symbols]


def load_real_data_bundle(
    *,
    core_symbols: list[str],
    start: date,
    end: date,
    max_symbols: int,
    report_symbol_cap: int = 40,
    workers: int = 8,
    cache_dir: Path | None = None,
    cache_ttl_hours: float = 8.0,
    strict_cutoff: date | None = None,
    review_days: int = 5,
    include_post_review: bool = True,
) -> RealDataBundle:
    cutoff = strict_cutoff or end
    cutoff = min(cutoff, end)
    review_days = max(0, int(review_days))
    review_end = cutoff + timedelta(days=review_days) if include_post_review and review_days > 0 else cutoff

    cache_key_src = (
        "v6|"
        f"{start.isoformat()}|{end.isoformat()}|{cutoff.isoformat()}|{review_end.isoformat()}|"
        f"{max_symbols}|{report_symbol_cap}|{include_post_review}|{','.join(sorted(core_symbols))}"
    )
    cache_key = hashlib.sha1(cache_key_src.encode("utf-8")).hexdigest()[:16]
    cache_meta = None
    bars_cache_path = None
    news_cache_path = None
    report_cache_path = None
    news_symbol_cache_path = None
    report_symbol_cache_path = None
    review_news_cache_path = None
    review_report_cache_path = None
    review_news_symbol_cache_path = None
    review_report_symbol_cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_meta = cache_dir / f"{cache_key}_meta.json"
        bars_cache_path = cache_dir / f"{cache_key}_bars.parquet"
        news_cache_path = cache_dir / f"{cache_key}_news_daily_pre.csv"
        report_cache_path = cache_dir / f"{cache_key}_report_daily_pre.csv"
        news_symbol_cache_path = cache_dir / f"{cache_key}_news_daily_by_symbol_pre.csv"
        report_symbol_cache_path = cache_dir / f"{cache_key}_report_daily_by_symbol_pre.csv"
        review_news_cache_path = cache_dir / f"{cache_key}_news_daily_review.csv"
        review_report_cache_path = cache_dir / f"{cache_key}_report_daily_review.csv"
        review_news_symbol_cache_path = cache_dir / f"{cache_key}_news_daily_by_symbol_review.csv"
        review_report_symbol_cache_path = cache_dir / f"{cache_key}_report_daily_by_symbol_review.csv"

    def _read_series(path: Path | None, value_col: str) -> pd.Series:
        if path is None or not path.exists():
            return pd.Series(dtype=float)
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.Series(dtype=float)
        if df.empty or "date" not in df.columns or value_col not in df.columns:
            return pd.Series(dtype=float)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["date", value_col])
        if df.empty:
            return pd.Series(dtype=float)
        return pd.Series(df[value_col].to_numpy(), index=df["date"].tolist()).sort_index()

    def _read_symbol_frame(path: Path | None, value_col: str) -> pd.DataFrame:
        if path is None or not path.exists():
            return pd.DataFrame(columns=["date", "symbol", value_col])
        try:
            df = pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=["date", "symbol", value_col])
        if df.empty or "date" not in df.columns or "symbol" not in df.columns or value_col not in df.columns:
            return pd.DataFrame(columns=["date", "symbol", value_col])
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["symbol"] = df["symbol"].astype(str)
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["date", "symbol", value_col])
        if df.empty:
            return pd.DataFrame(columns=["date", "symbol", value_col])
        return df[["date", "symbol", value_col]].sort_values(["date", "symbol"]).reset_index(drop=True)

    if (
        cache_meta
        and cache_meta.exists()
        and bars_cache_path
        and news_cache_path
        and report_cache_path
        and news_symbol_cache_path
        and report_symbol_cache_path
        and review_news_cache_path
        and review_report_cache_path
        and review_news_symbol_cache_path
        and review_report_symbol_cache_path
        and bars_cache_path.exists()
        and news_cache_path.exists()
        and report_cache_path.exists()
        and news_symbol_cache_path.exists()
        and report_symbol_cache_path.exists()
        and review_news_cache_path.exists()
        and review_report_cache_path.exists()
        and review_news_symbol_cache_path.exists()
        and review_report_symbol_cache_path.exists()
    ):
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(str(meta.get("created_at")))
            age_hours = (datetime.now() - created_at).total_seconds() / 3600.0
            cache_ttl = float(cache_ttl_hours)
            if cache_ttl > 0.0 and age_hours <= cache_ttl:
                bars_all = pd.read_parquet(bars_cache_path)
                bars_all["ts"] = pd.to_datetime(bars_all["ts"], errors="coerce")
                news_daily = _read_series(news_cache_path, "news_score")
                report_daily = _read_series(report_cache_path, "report_score")
                news_daily_by_symbol = _read_symbol_frame(news_symbol_cache_path, "news_score")
                report_daily_by_symbol = _read_symbol_frame(report_symbol_cache_path, "report_score")
                review_news_daily = _read_series(review_news_cache_path, "news_score")
                review_report_daily = _read_series(review_report_cache_path, "report_score")
                review_news_daily_by_symbol = _read_symbol_frame(review_news_symbol_cache_path, "news_score")
                review_report_daily_by_symbol = _read_symbol_frame(review_report_symbol_cache_path, "report_score")
                try:
                    cached_cutoff = datetime.fromisoformat(str(meta.get("cutoff_date", cutoff.isoformat()))).date()
                except Exception:
                    cached_cutoff = cutoff
                try:
                    cached_review_end = datetime.fromisoformat(str(meta.get("review_end_date", review_end.isoformat()))).date()
                except Exception:
                    cached_review_end = review_end

                bars = bars_all[bars_all["ts"].dt.date <= cached_cutoff].copy()
                review_bars = bars_all[
                    (bars_all["ts"].dt.date > cached_cutoff) & (bars_all["ts"].dt.date <= cached_review_end)
                ].copy()
                bars = bars.sort_values(["ts", "symbol"]).reset_index(drop=True)
                review_bars = review_bars.sort_values(["ts", "symbol"]).reset_index(drop=True)
                cutoff_ts = str(meta.get("cutoff_ts", _date_end_iso(cached_cutoff)))
                bar_max_ts = str(meta.get("bar_max_ts", _max_bar_ts_iso(bars)))
                news_max_ts = str(meta.get("news_max_ts", _max_daily_series_ts_iso(news_daily)))
                report_max_ts = str(meta.get("report_max_ts", _max_daily_series_ts_iso(report_daily)))
                review_bar_max_ts = str(meta.get("review_bar_max_ts", _max_bar_ts_iso(review_bars)))
                review_news_max_ts = str(meta.get("review_news_max_ts", _max_daily_series_ts_iso(review_news_daily)))
                review_report_max_ts = str(meta.get("review_report_max_ts", _max_daily_series_ts_iso(review_report_daily)))

                return RealDataBundle(
                    bars=bars,
                    review_bars=review_bars,
                    universe=list(meta.get("universe", [])),
                    news_daily=news_daily,
                    report_daily=report_daily,
                    news_daily_by_symbol=news_daily_by_symbol,
                    report_daily_by_symbol=report_daily_by_symbol,
                    news_records=int(meta.get("news_records", 0)),
                    report_records=int(meta.get("report_records", 0)),
                    cutoff_date=cached_cutoff,
                    review_days=int(meta.get("review_days", review_days)),
                    review_news_daily=review_news_daily,
                    review_report_daily=review_report_daily,
                    review_news_daily_by_symbol=review_news_daily_by_symbol,
                    review_report_daily_by_symbol=review_report_daily_by_symbol,
                    review_news_records=int(meta.get("review_news_records", 0)),
                    review_report_records=int(meta.get("review_report_records", 0)),
                    cutoff_ts=cutoff_ts,
                    bar_max_ts=bar_max_ts,
                    news_max_ts=news_max_ts,
                    report_max_ts=report_max_ts,
                    review_bar_max_ts=review_bar_max_ts,
                    review_news_max_ts=review_news_max_ts,
                    review_report_max_ts=review_report_max_ts,
                    fetch_stats={
                        **(meta.get("fetch_stats", {}) or {}),
                        "cache_hit": True,
                        "cache_age_hours": age_hours,
                        "cutoff_ts": cutoff_ts,
                        "bar_max_ts": bar_max_ts,
                        "news_max_ts": news_max_ts,
                        "report_max_ts": report_max_ts,
                        "review_bar_max_ts": review_bar_max_ts,
                        "review_news_max_ts": review_news_max_ts,
                        "review_report_max_ts": review_report_max_ts,
                        "cache_path": str(cache_meta),
                    },
                )
        except Exception:
            pass

    universe = load_universe(core_symbols=core_symbols, max_symbols=max_symbols)
    universe_is_all_crypto = bool(universe) and all(_is_crypto_symbol(s) for s in universe)

    bars_frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
        futs = [ex.submit(_fetch_one_symbol, s, start, review_end) for s in universe]
        for f in as_completed(futs):
            symbol, df, err = f.result()
            if err:
                errors[symbol] = err
                continue
            if df is not None and not df.empty:
                bars_frames.append(df)
            else:
                errors[symbol] = "empty"

    bars_all = pd.concat(bars_frames, ignore_index=True) if bars_frames else pd.DataFrame(
        columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"]
    )
    bars_all = bars_all.sort_values(["ts", "symbol"]).reset_index(drop=True)
    public_macro = _load_public_macro_frame(start, review_end)
    bars_all = _attach_macro_features_to_bars(bars_all, public_macro)
    ts_date = pd.to_datetime(bars_all["ts"]).dt.date if not bars_all.empty else pd.Series(dtype="object")
    bars = bars_all[ts_date <= cutoff].copy() if not bars_all.empty else bars_all.copy()
    review_bars = (
        bars_all[(ts_date > cutoff) & (ts_date <= review_end)].copy()
        if not bars_all.empty
        else pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])
    )
    bars = bars.sort_values(["ts", "symbol"]).reset_index(drop=True)
    review_bars = review_bars.sort_values(["ts", "symbol"]).reset_index(drop=True)

    eq_symbols = [s for s in universe if EQUITY_RE.match(s)][: max(5, int(report_symbol_cap))]
    alt_symbols = [s for s in universe if FUTURE_RE.match(s) or _is_crypto_symbol(s)]
    news_symbols = list(dict.fromkeys(eq_symbols + alt_symbols))
    news_frames: list[pd.DataFrame] = []
    report_frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(2, workers // 2)) as ex:
        futs2 = [ex.submit(fetch_symbol_news_and_reports, s, start, review_end) for s in news_symbols]
        for f in as_completed(futs2):
            ndf, rdf = f.result()
            if not ndf.empty:
                news_frames.append(ndf)
            if not rdf.empty:
                report_frames.append(rdf)

    news_all = pd.concat(news_frames, ignore_index=True) if news_frames else pd.DataFrame(columns=["date", "symbol", "news_score"])
    report_all = pd.concat(report_frames, ignore_index=True) if report_frames else pd.DataFrame(columns=["date", "symbol", "report_score"])

    news_df = news_all[news_all["date"] <= cutoff].copy() if not news_all.empty else pd.DataFrame(columns=["date", "symbol", "news_score"])
    report_df = report_all[report_all["date"] <= cutoff].copy() if not report_all.empty else pd.DataFrame(columns=["date", "symbol", "report_score"])
    review_news_df = news_all[(news_all["date"] > cutoff) & (news_all["date"] <= review_end)].copy() if not news_all.empty else pd.DataFrame(columns=["date", "symbol", "news_score"])
    review_report_df = report_all[(report_all["date"] > cutoff) & (report_all["date"] <= review_end)].copy() if not report_all.empty else pd.DataFrame(columns=["date", "symbol", "report_score"])

    news_daily_by_symbol = _aggregate_symbol_daily_score_frame(news_df, "news_score")
    report_daily_by_symbol = _aggregate_symbol_daily_score_frame(report_df, "report_score")
    review_news_daily_by_symbol = _aggregate_symbol_daily_score_frame(review_news_df, "news_score")
    review_report_daily_by_symbol = _aggregate_symbol_daily_score_frame(review_report_df, "report_score")
    news_daily = _aggregate_daily_score_frame(news_daily_by_symbol, "news_score")
    report_daily = _aggregate_daily_score_frame(report_daily_by_symbol, "report_score")
    review_news_daily = _aggregate_daily_score_frame(review_news_daily_by_symbol, "news_score")
    review_report_daily = _aggregate_daily_score_frame(review_report_daily_by_symbol, "report_score")
    cutoff_ts = _date_end_iso(cutoff)
    bar_max_ts = _max_bar_ts_iso(bars)
    news_max_ts = _max_daily_series_ts_iso(news_daily)
    report_max_ts = _max_daily_series_ts_iso(report_daily)
    review_bar_max_ts = _max_bar_ts_iso(review_bars)
    review_news_max_ts = _max_daily_series_ts_iso(review_news_daily)
    review_report_max_ts = _max_daily_series_ts_iso(review_report_daily)

    bundle = RealDataBundle(
        bars=bars,
        review_bars=review_bars,
        universe=universe,
        news_daily=news_daily,
        report_daily=report_daily,
        news_daily_by_symbol=news_daily_by_symbol,
        report_daily_by_symbol=report_daily_by_symbol,
        news_records=int(len(news_df)),
        report_records=int(len(report_df)),
        cutoff_date=cutoff,
        review_days=review_days,
        review_news_daily=review_news_daily,
        review_report_daily=review_report_daily,
        review_news_daily_by_symbol=review_news_daily_by_symbol,
        review_report_daily_by_symbol=review_report_daily_by_symbol,
        review_news_records=int(len(review_news_df)),
        review_report_records=int(len(review_report_df)),
        cutoff_ts=cutoff_ts,
        bar_max_ts=bar_max_ts,
        news_max_ts=news_max_ts,
        report_max_ts=report_max_ts,
        review_bar_max_ts=review_bar_max_ts,
        review_news_max_ts=review_news_max_ts,
        review_report_max_ts=review_report_max_ts,
        fetch_stats={
            "universe_count": len(universe),
            "bars_symbols": int(bars["symbol"].nunique()) if not bars.empty else 0,
            "bars_rows": int(len(bars)),
            "review_bars_rows": int(len(review_bars)),
            "errors": errors,
            "public_macro_rows": int(len(public_macro)),
            "public_macro_columns": [
                c for c in list(getattr(public_macro, "columns", []))
                if c not in {"date", "source"} and not str(c).endswith("_source")
            ],
            "news_daily_by_symbol_rows": int(len(news_daily_by_symbol)),
            "report_daily_by_symbol_rows": int(len(report_daily_by_symbol)),
            "review_news_daily_by_symbol_rows": int(len(review_news_daily_by_symbol)),
            "review_report_daily_by_symbol_rows": int(len(review_report_daily_by_symbol)),
            "strict_cutoff_enforced": True,
            "cutoff_date": cutoff.isoformat(),
            "cutoff_ts": cutoff_ts,
            "review_end_date": review_end.isoformat(),
            "review_days": review_days,
            "bar_max_ts": bar_max_ts,
            "news_max_ts": news_max_ts,
            "report_max_ts": report_max_ts,
            "review_bar_max_ts": review_bar_max_ts,
            "review_news_max_ts": review_news_max_ts,
            "review_report_max_ts": review_report_max_ts,
            "universe_source_notice": (
                "core symbols only (crypto mode)"
                if universe_is_all_crypto
                else "index constituents are latest snapshot; survivorship bias may remain"
            ),
        },
    )
    if (
        cache_meta
        and bars_cache_path
        and news_cache_path
        and report_cache_path
        and news_symbol_cache_path
        and report_symbol_cache_path
        and review_news_cache_path
        and review_report_cache_path
        and review_news_symbol_cache_path
        and review_report_symbol_cache_path
    ):
        try:
            bars_all.to_parquet(bars_cache_path, index=False)
            pd.DataFrame({"date": list(news_daily.index), "news_score": news_daily.values}).to_csv(news_cache_path, index=False)
            pd.DataFrame({"date": list(report_daily.index), "report_score": report_daily.values}).to_csv(report_cache_path, index=False)
            news_daily_by_symbol.to_csv(news_symbol_cache_path, index=False)
            report_daily_by_symbol.to_csv(report_symbol_cache_path, index=False)
            pd.DataFrame({"date": list(review_news_daily.index), "news_score": review_news_daily.values}).to_csv(review_news_cache_path, index=False)
            pd.DataFrame({"date": list(review_report_daily.index), "report_score": review_report_daily.values}).to_csv(review_report_cache_path, index=False)
            review_news_daily_by_symbol.to_csv(review_news_symbol_cache_path, index=False)
            review_report_daily_by_symbol.to_csv(review_report_symbol_cache_path, index=False)
            cache_payload = {
                "created_at": datetime.now().isoformat(),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "cutoff_date": cutoff.isoformat(),
                "cutoff_ts": cutoff_ts,
                "review_end_date": review_end.isoformat(),
                "review_days": review_days,
                "max_symbols": int(max_symbols),
                "report_symbol_cap": int(report_symbol_cap),
                "universe": universe,
                "bars_rows": int(len(bars)),
                "review_bars_rows": int(len(review_bars)),
                "bar_max_ts": bar_max_ts,
                "news_records": int(len(news_df)),
                "report_records": int(len(report_df)),
                "review_news_records": int(len(review_news_df)),
                "review_report_records": int(len(review_report_df)),
                "news_max_ts": news_max_ts,
                "report_max_ts": report_max_ts,
                "review_bar_max_ts": review_bar_max_ts,
                "review_news_max_ts": review_news_max_ts,
                "review_report_max_ts": review_report_max_ts,
                "fetch_stats": bundle.fetch_stats,
            }
            cache_meta.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            bundle.fetch_stats["cache_hit"] = False
            bundle.fetch_stats["cache_path"] = str(cache_meta)
        except Exception:
            pass
    return bundle
