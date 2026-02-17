from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from lie_engine.data.protocols import DataProviderProtocol
from lie_engine.data.quality import (
    DataQualityReport,
    SourceConfidenceItem,
    SourceConfidenceReport,
    evaluate_quality,
)
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
    source_confidence: SourceConfidenceReport
    quality: DataQualityReport


class DataBus:
    def __init__(
        self,
        providers: list[DataProviderProtocol],
        output_dir: Path,
        sqlite_path: Path,
        completeness_min: float,
        conflict_max: float,
        source_confidence_min: float = 0.75,
        low_confidence_source_ratio_max: float = 0.40,
    ) -> None:
        self.providers = providers
        self.output_dir = output_dir
        self.sqlite_path = sqlite_path
        self.completeness_min = completeness_min
        self.conflict_max = conflict_max
        self.source_confidence_min = float(source_confidence_min)
        self.low_confidence_source_ratio_max = float(low_confidence_source_ratio_max)

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

    def _evaluate_source_confidence(
        self,
        *,
        raw_bars: pd.DataFrame,
        macro: pd.DataFrame | None = None,
        news: list[NewsEvent],
        sentiment_factor_count: dict[str, int],
    ) -> SourceConfidenceReport:
        macro_df = macro.copy() if macro is not None else pd.DataFrame()
        observed_sources: set[str] = {str(getattr(p, "name", "")).strip() for p in self.providers if str(getattr(p, "name", "")).strip()}
        if not raw_bars.empty and "source" in raw_bars.columns:
            observed_sources |= {str(s).strip() for s in raw_bars["source"].dropna().astype(str).tolist() if str(s).strip()}
        if not macro_df.empty and "source" in macro_df.columns:
            observed_sources |= {str(s).strip() for s in macro_df["source"].dropna().astype(str).tolist() if str(s).strip()}
        observed_sources |= {str(ev.source).strip() for ev in news if str(ev.source).strip()}

        if not observed_sources:
            return SourceConfidenceReport(overall_score=1.0, by_source={}, low_confidence_sources=[], details=[])

        max_rows = 0
        bar_consistency_map: dict[str, float] = {}
        bar_coverage_map: dict[str, float] = {}
        bar_rows_map: dict[str, int] = {}
        if not raw_bars.empty and {"ts", "symbol", "close", "source"}.issubset(set(raw_bars.columns)):
            bars = raw_bars.copy()
            bars["source"] = bars["source"].astype(str)
            rows_by_source = bars.groupby("source").size().to_dict()
            max_rows = int(max(rows_by_source.values())) if rows_by_source else 0

            baseline = bars.groupby(["ts", "symbol"], as_index=False)["close"].median().rename(columns={"close": "close_median"})
            merged = bars.merge(baseline, on=["ts", "symbol"], how="left")
            denom = merged["close_median"].abs().clip(lower=1.0)
            merged["close_dev_pct"] = (merged["close"].astype(float) - merged["close_median"].astype(float)).abs() / denom
            for src, grp in merged.groupby("source"):
                mean_dev = float(grp["close_dev_pct"].mean()) if not grp.empty else 1.0
                bar_consistency = float(max(0.0, 1.0 - min(1.0, mean_dev / 0.05)))
                rows = int(len(grp))
                coverage = float(rows / max(1, max_rows))
                bar_consistency_map[str(src)] = bar_consistency
                bar_coverage_map[str(src)] = max(0.0, min(1.0, coverage))
                bar_rows_map[str(src)] = rows

        macro_consistency_map: dict[str, float] = {}
        macro_coverage_map: dict[str, float] = {}
        macro_rows_map: dict[str, int] = {}
        if not macro_df.empty and {"date", "source"}.issubset(set(macro_df.columns)):
            macro_df = macro_df.copy()
            macro_df["source"] = macro_df["source"].astype(str)
            macro_df["date"] = pd.to_datetime(macro_df["date"], errors="coerce")
            macro_df = macro_df.dropna(subset=["date"])
            numeric_cols = [c for c in ["cpi_yoy", "ppi_yoy", "lpr_1y"] if c in macro_df.columns]
            for c in numeric_cols:
                macro_df[c] = pd.to_numeric(macro_df[c], errors="coerce")
            if numeric_cols:
                macro_df = macro_df.dropna(subset=numeric_cols, how="all")
            rows_by_source = macro_df.groupby("source").size().to_dict() if not macro_df.empty else {}
            max_rows = int(max(rows_by_source.values())) if rows_by_source else 0
            if not macro_df.empty and numeric_cols:
                base = (
                    macro_df.groupby("date", as_index=False)[numeric_cols]
                    .median()
                    .rename(columns={c: f"{c}_median" for c in numeric_cols})
                )
                merged = macro_df.merge(base, on="date", how="left")
                dev_cols: list[str] = []
                for c in numeric_cols:
                    med = f"{c}_median"
                    dcol = f"{c}_dev"
                    denom = merged[med].abs().clip(lower=1.0)
                    merged[dcol] = (merged[c].astype(float) - merged[med].astype(float)).abs() / denom
                    dev_cols.append(dcol)
                merged["macro_dev"] = merged[dev_cols].mean(axis=1)
                for src, grp in merged.groupby("source"):
                    mean_dev = float(grp["macro_dev"].mean()) if not grp.empty else 1.0
                    macro_consistency = float(max(0.0, 1.0 - min(1.0, mean_dev / 0.25)))
                    rows = int(len(grp))
                    coverage = float(rows / max(1, max_rows))
                    macro_consistency_map[str(src)] = macro_consistency
                    macro_coverage_map[str(src)] = max(0.0, min(1.0, coverage))
                    macro_rows_map[str(src)] = rows
            else:
                for src, rows in rows_by_source.items():
                    macro_consistency_map[str(src)] = 0.50
                    macro_coverage_map[str(src)] = float(rows / max(1, max_rows))
                    macro_rows_map[str(src)] = int(rows)

        news_scores: dict[str, float] = {}
        news_count: dict[str, int] = {}
        for ev in news:
            src = str(ev.source).strip()
            if not src:
                continue
            news_scores[src] = news_scores.get(src, 0.0) + float(ev.confidence)
            news_count[src] = news_count.get(src, 0) + 1

        details: list[SourceConfidenceItem] = []
        by_source: dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0
        for src in sorted(observed_sources):
            base = float(self._source_reliability(src))
            bar_consistency = float(bar_consistency_map.get(src, 0.50))
            bar_coverage = float(bar_coverage_map.get(src, 0.0))
            macro_consistency = float(macro_consistency_map.get(src, 0.50))
            macro_coverage = float(macro_coverage_map.get(src, 0.0))
            n_count = int(news_count.get(src, 0))
            news_conf = float(news_scores.get(src, 0.0) / n_count) if n_count > 0 else 0.50
            sentiment_cov = 1.0 if int(sentiment_factor_count.get(src, 0)) > 0 else 0.0

            score = (
                0.28 * bar_consistency
                + 0.16 * bar_coverage
                + 0.16 * news_conf
                + 0.12 * base
                + 0.05 * sentiment_cov
                + 0.16 * macro_consistency
                + 0.07 * macro_coverage
            )
            score = float(max(0.0, min(1.0, score)))
            by_source[src] = score

            bars_rows = int(bar_rows_map.get(src, 0))
            macro_rows = int(macro_rows_map.get(src, 0))
            factors = int(sentiment_factor_count.get(src, 0))
            evidence_w = float(max(1.0, bars_rows + macro_rows + 2 * n_count + 0.5 * factors))
            weighted_sum += evidence_w * score
            weight_total += evidence_w

            details.append(
                SourceConfidenceItem(
                    source=src,
                    score=score,
                    base_reliability=base,
                    bar_consistency=bar_consistency,
                    bar_coverage=bar_coverage,
                    news_confidence=news_conf,
                    sentiment_coverage=sentiment_cov,
                    bars_rows=bars_rows,
                    macro_rows=macro_rows,
                    news_events=n_count,
                    sentiment_factors=factors,
                    macro_consistency=macro_consistency,
                    macro_coverage=macro_coverage,
                )
            )

        overall = float(weighted_sum / max(1.0, weight_total))
        low_sources = [src for src, v in by_source.items() if float(v) < float(self.source_confidence_min)]
        return SourceConfidenceReport(
            overall_score=overall,
            by_source=by_source,
            low_confidence_sources=sorted(low_sources),
            details=details,
        )

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
        sentiment_factor_count: dict[str, int] = {}
        news_events: list[NewsEvent] = []
        for provider in self.providers:
            try:
                macro_frames.append(provider.fetch_macro(start=start, end=end))
            except NotImplementedError:
                pass
            try:
                snap = provider.fetch_sentiment_factors(as_of=end)
                if not isinstance(snap, dict):
                    snap = {}
                sentiment = {**sentiment, **snap}
                src_name = str(getattr(provider, "name", "")).strip()
                if src_name:
                    sentiment_factor_count[src_name] = int(len(snap))
            except NotImplementedError:
                pass
            for lang in langs:
                try:
                    news_events.extend(provider.fetch_news(start_ts=start_ts, end_ts=end_ts, lang=lang))
                except NotImplementedError:
                    continue

        macro = pd.concat(macro_frames, ignore_index=True) if macro_frames else pd.DataFrame()
        news = self._dedupe_news(self._normalize_news(news_events))
        source_conf = self._evaluate_source_confidence(
            raw_bars=raw_bars,
            macro=macro,
            news=news,
            sentiment_factor_count=sentiment_factor_count,
        )
        quality = evaluate_quality(
            normalized_bars=normalized_bars,
            conflicts=conflicts,
            completeness_min=self.completeness_min,
            conflict_max=self.conflict_max,
            source_confidence=source_conf,
            source_confidence_min=self.source_confidence_min,
            low_confidence_source_ratio_max=self.low_confidence_source_ratio_max,
        )

        return IngestionResult(
            raw_bars=raw_bars,
            normalized_bars=normalized_bars,
            conflicts=conflicts,
            macro=macro,
            sentiment=sentiment,
            news=news,
            source_confidence=source_conf,
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
        source_conf_path = self.output_dir / "artifacts" / "normalized" / f"{dstr}_source_confidence.json"
        quality_path = self.output_dir / "artifacts" / f"{dstr}_quality.json"

        write_csv(raw_path, result.raw_bars)
        write_csv(norm_path, result.normalized_bars)
        write_csv(conf_path, result.conflicts if not result.conflicts.empty else pd.DataFrame(columns=["ts", "symbol", "field", "values", "max_diff_pct"]))
        write_csv(macro_path, result.macro if not result.macro.empty else pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"]))
        write_json(news_path, [n.to_dict() for n in result.news])
        write_json(source_conf_path, result.source_confidence.to_dict())
        write_json(quality_path, result.quality.to_dict())

        feature_df = result.normalized_bars.copy()
        if not feature_df.empty:
            feature_df["ret_1d"] = feature_df.groupby("symbol")["close"].pct_change().fillna(0.0)
            feature_df["vol_chg"] = feature_df.groupby("symbol")["volume"].pct_change().fillna(0.0)
        write_parquet_optional(feat_path, feature_df)

        append_sqlite(self.sqlite_path, "bars_normalized", result.normalized_bars)
        append_sqlite(self.sqlite_path, "macro", result.macro if not result.macro.empty else pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"]))
        source_conf_df = pd.DataFrame([d.to_dict() | {"as_of": dstr} for d in result.source_confidence.details])
        if source_conf_df.empty:
            source_conf_df = pd.DataFrame(columns=["source", "score", "base_reliability", "bar_consistency", "bar_coverage", "macro_consistency", "macro_coverage", "news_confidence", "sentiment_coverage", "bars_rows", "macro_rows", "news_events", "sentiment_factors", "as_of"])
        append_sqlite(self.sqlite_path, "source_confidence", source_conf_df)
        append_sqlite(self.sqlite_path, "quality", pd.DataFrame([result.quality.to_dict() | {"as_of": dstr}]))
