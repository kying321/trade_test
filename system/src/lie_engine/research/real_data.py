from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf


EQUITY_RE = re.compile(r"^\d{6}$")
FUTURE_RE = re.compile(r"^([A-Z]{1,3})\d{4}$")


POSITIVE_KWS = (
    "上调",
    "买入",
    "增持",
    "超预期",
    "增长",
    "回升",
    "突破",
    "利好",
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
    "downgrade",
    "miss",
    "warn",
)


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
    cutoff_date: date | None = None
    review_days: int = 0
    review_news_daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    review_report_daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    review_news_records: int = 0
    review_report_records: int = 0


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


def _fetch_one_symbol(symbol: str, start: date, end: date) -> tuple[str, pd.DataFrame, str | None]:
    try:
        if EQUITY_RE.match(symbol):
            df = fetch_equity_daily(symbol, start, end)
            return symbol, df, None
        if FUTURE_RE.match(symbol):
            df = fetch_future_daily(symbol, start, end)
            return symbol, df, None
        return symbol, pd.DataFrame(), f"unsupported_symbol:{symbol}"
    except Exception as exc:  # noqa: BLE001
        return symbol, pd.DataFrame(), f"{type(exc).__name__}:{exc}"


def _stock_news_score(title: str, content: str) -> float:
    txt = f"{title} {content}".lower()
    pos = sum(1 for k in POSITIVE_KWS if k.lower() in txt)
    neg = sum(1 for k in NEGATIVE_KWS if k.lower() in txt)
    if pos == neg:
        return 0.0
    return float((pos - neg) / max(1, pos + neg))


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
    if not EQUITY_RE.match(symbol):
        return pd.DataFrame(columns=["date", "symbol", "news_score"]), pd.DataFrame(columns=["date", "symbol", "report_score"])

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
        if s not in seen:
            out.append(s)
            seen.add(s)

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
        "v4|"
        f"{start.isoformat()}|{end.isoformat()}|{cutoff.isoformat()}|{review_end.isoformat()}|"
        f"{max_symbols}|{report_symbol_cap}|{include_post_review}|{','.join(sorted(core_symbols))}"
    )
    cache_key = hashlib.sha1(cache_key_src.encode("utf-8")).hexdigest()[:16]
    cache_meta = None
    bars_cache_path = None
    news_cache_path = None
    report_cache_path = None
    review_news_cache_path = None
    review_report_cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_meta = cache_dir / f"{cache_key}_meta.json"
        bars_cache_path = cache_dir / f"{cache_key}_bars.parquet"
        news_cache_path = cache_dir / f"{cache_key}_news_daily_pre.csv"
        report_cache_path = cache_dir / f"{cache_key}_report_daily_pre.csv"
        review_news_cache_path = cache_dir / f"{cache_key}_news_daily_review.csv"
        review_report_cache_path = cache_dir / f"{cache_key}_report_daily_review.csv"

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

    if (
        cache_meta
        and cache_meta.exists()
        and bars_cache_path
        and news_cache_path
        and report_cache_path
        and review_news_cache_path
        and review_report_cache_path
        and bars_cache_path.exists()
        and news_cache_path.exists()
        and report_cache_path.exists()
        and review_news_cache_path.exists()
        and review_report_cache_path.exists()
    ):
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(str(meta.get("created_at")))
            age_hours = (datetime.now() - created_at).total_seconds() / 3600.0
            if age_hours <= max(0.1, float(cache_ttl_hours)):
                bars_all = pd.read_parquet(bars_cache_path)
                bars_all["ts"] = pd.to_datetime(bars_all["ts"], errors="coerce")
                news_daily = _read_series(news_cache_path, "news_score")
                report_daily = _read_series(report_cache_path, "report_score")
                review_news_daily = _read_series(review_news_cache_path, "news_score")
                review_report_daily = _read_series(review_report_cache_path, "report_score")
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

                return RealDataBundle(
                    bars=bars,
                    review_bars=review_bars,
                    universe=list(meta.get("universe", [])),
                    news_daily=news_daily,
                    report_daily=report_daily,
                    news_records=int(meta.get("news_records", 0)),
                    report_records=int(meta.get("report_records", 0)),
                    cutoff_date=cached_cutoff,
                    review_days=int(meta.get("review_days", review_days)),
                    review_news_daily=review_news_daily,
                    review_report_daily=review_report_daily,
                    review_news_records=int(meta.get("review_news_records", 0)),
                    review_report_records=int(meta.get("review_report_records", 0)),
                    fetch_stats={
                        **(meta.get("fetch_stats", {}) or {}),
                        "cache_hit": True,
                        "cache_age_hours": age_hours,
                    },
                )
        except Exception:
            pass

    universe = load_universe(core_symbols=core_symbols, max_symbols=max_symbols)

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
    news_frames: list[pd.DataFrame] = []
    report_frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(2, workers // 2)) as ex:
        futs2 = [ex.submit(fetch_symbol_news_and_reports, s, start, review_end) for s in eq_symbols]
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

    news_daily = news_df.groupby("date")["news_score"].mean().sort_index() if not news_df.empty else pd.Series(dtype=float)
    report_daily = report_df.groupby("date")["report_score"].mean().sort_index() if not report_df.empty else pd.Series(dtype=float)
    review_news_daily = review_news_df.groupby("date")["news_score"].mean().sort_index() if not review_news_df.empty else pd.Series(dtype=float)
    review_report_daily = review_report_df.groupby("date")["report_score"].mean().sort_index() if not review_report_df.empty else pd.Series(dtype=float)

    bundle = RealDataBundle(
        bars=bars,
        review_bars=review_bars,
        universe=universe,
        news_daily=news_daily,
        report_daily=report_daily,
        news_records=int(len(news_df)),
        report_records=int(len(report_df)),
        cutoff_date=cutoff,
        review_days=review_days,
        review_news_daily=review_news_daily,
        review_report_daily=review_report_daily,
        review_news_records=int(len(review_news_df)),
        review_report_records=int(len(review_report_df)),
        fetch_stats={
            "universe_count": len(universe),
            "bars_symbols": int(bars["symbol"].nunique()) if not bars.empty else 0,
            "bars_rows": int(len(bars)),
            "review_bars_rows": int(len(review_bars)),
            "errors": errors,
            "strict_cutoff_enforced": True,
            "cutoff_date": cutoff.isoformat(),
            "review_end_date": review_end.isoformat(),
            "review_days": review_days,
            "universe_source_notice": "index constituents are latest snapshot; survivorship bias may remain",
        },
    )
    if cache_meta and bars_cache_path and news_cache_path and report_cache_path and review_news_cache_path and review_report_cache_path:
        try:
            bars_all.to_parquet(bars_cache_path, index=False)
            pd.DataFrame({"date": list(news_daily.index), "news_score": news_daily.values}).to_csv(news_cache_path, index=False)
            pd.DataFrame({"date": list(report_daily.index), "report_score": report_daily.values}).to_csv(report_cache_path, index=False)
            pd.DataFrame({"date": list(review_news_daily.index), "news_score": review_news_daily.values}).to_csv(review_news_cache_path, index=False)
            pd.DataFrame({"date": list(review_report_daily.index), "report_score": review_report_daily.values}).to_csv(review_report_cache_path, index=False)
            cache_payload = {
                "created_at": datetime.now().isoformat(),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "cutoff_date": cutoff.isoformat(),
                "review_end_date": review_end.isoformat(),
                "review_days": review_days,
                "max_symbols": int(max_symbols),
                "report_symbol_cap": int(report_symbol_cap),
                "universe": universe,
                "bars_rows": int(len(bars)),
                "review_bars_rows": int(len(review_bars)),
                "news_records": int(len(news_df)),
                "report_records": int(len(report_df)),
                "review_news_records": int(len(review_news_df)),
                "review_report_records": int(len(review_report_df)),
                "fetch_stats": bundle.fetch_stats,
            }
            cache_meta.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            bundle.fetch_stats["cache_hit"] = False
            bundle.fetch_stats["cache_path"] = str(cache_meta)
        except Exception:
            pass
    return bundle
