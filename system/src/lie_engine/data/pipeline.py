from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from lie_engine.data.protocols import DataProviderProtocol
from lie_engine.data.quality import DataQualityReport, evaluate_quality
from lie_engine.data.storage import append_sqlite, write_csv, write_json, write_parquet_optional
from lie_engine.models import NewsEvent


@dataclass(slots=True)
class IngestionResult:
    raw_bars: pd.DataFrame
    normalized_bars: pd.DataFrame
    conflicts: pd.DataFrame
    macro: pd.DataFrame
    sentiment: dict[str, float]
    news: list[NewsEvent]
    quality: DataQualityReport


class DataBus:
    def __init__(
        self,
        providers: list[DataProviderProtocol],
        output_dir: Path,
        sqlite_path: Path,
        completeness_min: float,
        conflict_max: float,
    ) -> None:
        self.providers = providers
        self.output_dir = output_dir
        self.sqlite_path = sqlite_path
        self.completeness_min = completeness_min
        self.conflict_max = conflict_max

    def _collect_bars(self, symbols: list[str], start: date, end: date, freq: str = "1d") -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for provider in self.providers:
            for symbol in symbols:
                try:
                    frame = provider.fetch_ohlcv(symbol=symbol, start=start, end=end, freq=freq)
                except NotImplementedError:
                    continue
                frames.append(frame)
        if not frames:
            return pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"])
        raw = pd.concat(frames, ignore_index=True)
        raw["ts"] = pd.to_datetime(raw["ts"])
        return raw

    @staticmethod
    def _resolve_prices(raw_bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if raw_bars.empty:
            return raw_bars.copy(), pd.DataFrame(columns=["ts", "symbol", "field", "values", "max_diff_pct"])

        grouped = raw_bars.groupby(["ts", "symbol"], as_index=False)
        normalized_rows: list[dict[str, Any]] = []
        conflict_rows: list[dict[str, Any]] = []

        for (_, _), g in grouped:
            g = g.sort_values("source")
            row = {
                "ts": g["ts"].iloc[0],
                "symbol": g["symbol"].iloc[0],
                "open": float(g["open"].median()),
                "high": float(g["high"].median()),
                "low": float(g["low"].median()),
                "close": float(g["close"].median()),
                "volume": float(g["volume"].median()),
                "asset_class": g["asset_class"].iloc[0],
                "source_count": int(g["source"].nunique()),
                "data_conflict_flag": 0,
            }

            for field in ["close", "volume"]:
                if len(g) > 1:
                    vals = g[field].to_numpy(dtype=float)
                    vmin = vals.min()
                    vmax = vals.max()
                    denom = max(abs(vmin), abs(vmax), 1.0)
                    max_diff_pct = abs(vmax - vmin) / denom
                    if max_diff_pct > 0.05:
                        row["data_conflict_flag"] = 1
                        conflict_rows.append(
                            {
                                "ts": row["ts"],
                                "symbol": row["symbol"],
                                "field": field,
                                "values": ",".join(f"{v:.6f}" for v in vals),
                                "max_diff_pct": max_diff_pct,
                            }
                        )

            normalized_rows.append(row)

        normalized = pd.DataFrame(normalized_rows).sort_values(["symbol", "ts"]).reset_index(drop=True)
        conflicts = pd.DataFrame(conflict_rows)
        return normalized, conflicts

    @staticmethod
    def _dedupe_news(news_events: list[NewsEvent]) -> list[NewsEvent]:
        seen: set[str] = set()
        deduped: list[NewsEvent] = []
        for ev in sorted(news_events, key=lambda x: x.ts):
            key = hashlib.sha1(
                f"{ev.title.strip().lower()}|{ev.ts:%Y-%m-%d %H}|{','.join(sorted(ev.entities))}".encode("utf-8")
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ev)
        return deduped

    @staticmethod
    def _source_reliability(source: str) -> float:
        src = (source or "").lower()
        official_keys = ("gov", "official", "exchange", "stats", "pbc", "ministry")
        mainstream_keys = (
            "reuters",
            "bloomberg",
            "xinhua",
            "cctv",
            "wsj",
            "ft",
            "open_source_primary",
        )
        if any(k in src for k in official_keys):
            return 0.95
        if any(k in src for k in mainstream_keys):
            return 0.85
        return 0.70

    @staticmethod
    def _extract_category(title: str, content: str, fallback: str) -> str:
        text = f"{title} {content}".lower()
        mapping = {
            "政策": ("政策", "国务院", "央行", "lpr", "mlf", "guideline", "regulation"),
            "宏观": ("cpi", "ppi", "inflation", "gdp", "就业", "非农", "fomc", "利率"),
            "地缘": ("地缘", "冲突", "战", "sanction", "tariff", "制裁", "谈判"),
            "产业链": ("排产", "库存", "供给", "需求", "产业链", "shipment", "产能"),
        }
        for category, keys in mapping.items():
            if any(k.lower() in text for k in keys):
                return category
        return fallback if fallback else "其他"

    def _normalize_news(self, news_events: list[NewsEvent]) -> list[NewsEvent]:
        normalized: list[NewsEvent] = []
        for ev in news_events:
            category = self._extract_category(ev.title, ev.content, ev.category)
            reliability = self._source_reliability(ev.source)
            merged_confidence = min(1.0, max(0.0, 0.65 * float(ev.confidence) + 0.35 * reliability))
            merged_importance = min(1.0, max(0.0, 0.7 * float(ev.importance) + 0.3 * merged_confidence))
            normalized.append(
                NewsEvent(
                    event_id=ev.event_id,
                    ts=ev.ts,
                    title=ev.title,
                    content=ev.content,
                    lang=ev.lang,
                    source=ev.source,
                    category=category,
                    confidence=merged_confidence,
                    entities=list(ev.entities),
                    importance=merged_importance,
                )
            )
        return normalized

    def ingest(
        self,
        symbols: list[str],
        start: date,
        end: date,
        start_ts: datetime,
        end_ts: datetime,
        langs: tuple[str, ...] = ("zh", "en"),
    ) -> IngestionResult:
        raw_bars = self._collect_bars(symbols=symbols, start=start, end=end)
        normalized_bars, conflicts = self._resolve_prices(raw_bars)

        macro_frames = []
        sentiment: dict[str, float] = {}
        news_events: list[NewsEvent] = []
        for provider in self.providers:
            try:
                macro_frames.append(provider.fetch_macro(start=start, end=end))
            except NotImplementedError:
                pass
            try:
                sentiment = {**sentiment, **provider.fetch_sentiment_factors(as_of=end)}
            except NotImplementedError:
                pass
            for lang in langs:
                try:
                    news_events.extend(provider.fetch_news(start_ts=start_ts, end_ts=end_ts, lang=lang))
                except NotImplementedError:
                    continue

        macro = pd.concat(macro_frames, ignore_index=True) if macro_frames else pd.DataFrame()
        news = self._dedupe_news(self._normalize_news(news_events))
        quality = evaluate_quality(
            normalized_bars=normalized_bars,
            conflicts=conflicts,
            completeness_min=self.completeness_min,
            conflict_max=self.conflict_max,
        )

        return IngestionResult(
            raw_bars=raw_bars,
            normalized_bars=normalized_bars,
            conflicts=conflicts,
            macro=macro,
            sentiment=sentiment,
            news=news,
            quality=quality,
        )

    def persist(self, as_of: date, result: IngestionResult) -> None:
        dstr = as_of.isoformat()
        raw_path = self.output_dir / "artifacts" / "raw" / f"{dstr}_bars_raw.csv"
        norm_path = self.output_dir / "artifacts" / "normalized" / f"{dstr}_bars_normalized.csv"
        feat_path = self.output_dir / "artifacts" / "feature" / f"{dstr}_bars_feature.parquet"
        conf_path = self.output_dir / "artifacts" / "normalized" / f"{dstr}_conflicts.csv"
        macro_path = self.output_dir / "artifacts" / "raw" / f"{dstr}_macro.csv"
        news_path = self.output_dir / "artifacts" / "raw" / f"{dstr}_news.json"
        quality_path = self.output_dir / "artifacts" / f"{dstr}_quality.json"

        write_csv(raw_path, result.raw_bars)
        write_csv(norm_path, result.normalized_bars)
        write_csv(conf_path, result.conflicts if not result.conflicts.empty else pd.DataFrame(columns=["ts", "symbol", "field", "values", "max_diff_pct"]))
        write_csv(macro_path, result.macro if not result.macro.empty else pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"]))
        write_json(news_path, [n.to_dict() for n in result.news])
        write_json(quality_path, result.quality.to_dict())

        feature_df = result.normalized_bars.copy()
        if not feature_df.empty:
            feature_df["ret_1d"] = feature_df.groupby("symbol")["close"].pct_change().fillna(0.0)
            feature_df["vol_chg"] = feature_df.groupby("symbol")["volume"].pct_change().fillna(0.0)
        write_parquet_optional(feat_path, feature_df)

        append_sqlite(self.sqlite_path, "bars_normalized", result.normalized_bars)
        append_sqlite(self.sqlite_path, "macro", result.macro if not result.macro.empty else pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"]))
        append_sqlite(self.sqlite_path, "quality", pd.DataFrame([result.quality.to_dict() | {"as_of": dstr}]))
