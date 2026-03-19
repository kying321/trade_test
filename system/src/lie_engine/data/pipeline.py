from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np
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

    def _collect_bars(self, symbols: list[str], start: date, end: date, freq: str = "1d") -> tuple[pd.DataFrame, dict[str, Any]]:
        frames: list[pd.DataFrame] = []
        provider_rows: list[dict[str, Any]] = []
        for provider in self.providers:
            provider_started_mono = time.monotonic()
            provider_name = str(getattr(provider, "name", type(provider).__name__)).strip() or type(provider).__name__
            provider_symbol_hits = 0
            provider_row_count = 0
            symbol_rows: list[dict[str, Any]] = []
            for symbol in symbols:
                symbol_started_mono = time.monotonic()
                try:
                    frame = provider.fetch_ohlcv(symbol=symbol, start=start, end=end, freq=freq)
                except NotImplementedError:
                    symbol_rows.append(
                        {
                            "symbol": str(symbol),
                            "rows": 0,
                            "elapsed_sec": round(time.monotonic() - symbol_started_mono, 3),
                            "outcome": "not_implemented",
                        }
                    )
                    continue
                except Exception:
                    symbol_rows.append(
                        {
                            "symbol": str(symbol),
                            "rows": 0,
                            "elapsed_sec": round(time.monotonic() - symbol_started_mono, 3),
                            "outcome": "error",
                        }
                    )
                    raise
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    symbol_rows.append(
                        {
                            "symbol": str(symbol),
                            "rows": 0,
                            "elapsed_sec": round(time.monotonic() - symbol_started_mono, 3),
                            "outcome": "empty",
                        }
                    )
                    continue
                frames.append(frame)
                provider_symbol_hits += 1
                frame_rows = int(len(frame))
                provider_row_count += frame_rows
                symbol_rows.append(
                    {
                        "symbol": str(symbol),
                        "rows": frame_rows,
                        "elapsed_sec": round(time.monotonic() - symbol_started_mono, 3),
                        "outcome": "rows",
                    }
                )
            slowest_symbols = sorted(
                symbol_rows,
                key=lambda row: (float(row.get("elapsed_sec", 0.0)), int(row.get("rows", 0))),
                reverse=True,
            )[:5]
            provider_rows.append(
                {
                    "provider": provider_name,
                    "requested_symbols": int(len(symbols)),
                    "symbols_with_rows": int(provider_symbol_hits),
                    "rows": int(provider_row_count),
                    "elapsed_sec": round(time.monotonic() - provider_started_mono, 3),
                    "symbols": symbol_rows,
                    "slowest_symbols": slowest_symbols,
                }
            )
        if not frames:
            return (
                pd.DataFrame(columns=["ts", "symbol", "open", "high", "low", "close", "volume", "source", "asset_class"]),
                {
                    "provider_count": int(len(self.providers)),
                    "requested_symbol_count": int(len(symbols)),
                    "raw_row_count": 0,
                    "providers": provider_rows,
                },
            )
        raw = pd.concat(frames, ignore_index=True)
        raw["ts"] = pd.to_datetime(raw["ts"])
        return (
            raw,
            {
                "provider_count": int(len(self.providers)),
                "requested_symbol_count": int(len(symbols)),
                "raw_row_count": int(len(raw)),
                "providers": provider_rows,
            },
        )

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
        review_dir = self.output_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        timing_path = review_dir / f"{end.isoformat()}_data_ingest_timing.json"
        timing_started_mono = time.monotonic()
        ingest_timing: dict[str, Any] = {
            "date": end.isoformat(),
            "status": "running",
            "elapsed_sec": 0.0,
            "current_stage": "",
            "last_completed_stage": "",
            "stages": [],
        }

        def _persist_ingest_timing() -> None:
            ingest_timing["elapsed_sec"] = round(time.monotonic() - timing_started_mono, 3)
            write_json(timing_path, ingest_timing)

        def _run_ingest_stage(
            name: str,
            fn: Any,
            *,
            summary_fn: Any = None,
        ) -> Any:
            stage_started_mono = time.monotonic()
            ingest_timing["status"] = "running"
            ingest_timing["current_stage"] = name
            ingest_timing["current_stage_started_elapsed_sec"] = round(stage_started_mono - timing_started_mono, 3)
            _persist_ingest_timing()
            try:
                result = fn()
            except Exception as exc:
                ingest_timing["status"] = "failed"
                ingest_timing["failed_stage"] = name
                ingest_timing["error"] = f"{type(exc).__name__}:{exc}"
                ingest_timing["current_stage_elapsed_sec"] = round(time.monotonic() - stage_started_mono, 3)
                _persist_ingest_timing()
                raise
            stage_elapsed_sec = round(time.monotonic() - stage_started_mono, 3)
            stage_row: dict[str, Any] = {"name": name, "elapsed_sec": stage_elapsed_sec}
            if summary_fn is not None:
                summary = summary_fn(result)
                if summary:
                    stage_row["summary"] = summary
            ingest_timing.setdefault("stages", []).append(stage_row)
            ingest_timing["last_completed_stage"] = name
            ingest_timing["current_stage"] = ""
            ingest_timing.pop("current_stage_started_elapsed_sec", None)
            ingest_timing.pop("current_stage_elapsed_sec", None)
            ingest_timing.pop("failed_stage", None)
            ingest_timing.pop("error", None)
            _persist_ingest_timing()
            return result

        raw_bars, collect_summary = _run_ingest_stage(
            "collect_bars",
            lambda: self._collect_bars(symbols=symbols, start=start, end=end),
            summary_fn=lambda out: out[1] if isinstance(out, tuple) and len(out) > 1 else {},
        )
        normalized_bars, conflicts = _run_ingest_stage(
            "resolve_prices",
            lambda: self._resolve_prices(raw_bars),
            summary_fn=lambda out: {
                "normalized_rows": int(len(out[0])) if isinstance(out, tuple) and len(out) > 0 else 0,
                "conflict_rows": int(len(out[1])) if isinstance(out, tuple) and len(out) > 1 else 0,
            },
        )

        macro_frames = []
        sentiment_snapshots: list[tuple[str, dict[str, float]]] = []
        sentiment_factor_count: dict[str, int] = {}
        news_events: list[NewsEvent] = []
        enrichment_provider_rows: list[dict[str, Any]] = []

        def _collect_enrichments() -> dict[str, Any]:
            for provider in self.providers:
                provider_started_mono = time.monotonic()
                provider_name = str(getattr(provider, "name", type(provider).__name__)).strip() or type(provider).__name__
                macro_rows = 0
                news_count = 0
                sentiment_count = 0
                try:
                    macro_df = provider.fetch_macro(start=start, end=end)
                    if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
                        macro_frames.append(macro_df)
                        macro_rows = int(len(macro_df))
                except NotImplementedError:
                    pass
                try:
                    snap = provider.fetch_sentiment_factors(as_of=end)
                    if not isinstance(snap, dict):
                        snap = {}
                    src_name = str(getattr(provider, "name", "")).strip()
                    if src_name and snap:
                        clean_snap: dict[str, float] = {}
                        for key, value in snap.items():
                            try:
                                val = float(value)
                            except (TypeError, ValueError):
                                continue
                            if not np.isfinite(val):
                                continue
                            clean_snap[str(key)] = float(val)
                        if clean_snap:
                            sentiment_snapshots.append((src_name, clean_snap))
                            sentiment_factor_count[src_name] = int(len(clean_snap))
                            sentiment_count = int(len(clean_snap))
                except NotImplementedError:
                    pass
                for lang in langs:
                    try:
                        provider_news = provider.fetch_news(start_ts=start_ts, end_ts=end_ts, lang=lang)
                    except NotImplementedError:
                        continue
                    if provider_news:
                        news_events.extend(provider_news)
                        news_count += int(len(provider_news))
                enrichment_provider_rows.append(
                    {
                        "provider": provider_name,
                        "macro_rows": int(macro_rows),
                        "news_events": int(news_count),
                        "sentiment_factors": int(sentiment_count),
                        "elapsed_sec": round(time.monotonic() - provider_started_mono, 3),
                    }
                )
            return {
                "provider_count": int(len(self.providers)),
                "macro_frame_count": int(len(macro_frames)),
                "raw_news_events": int(len(news_events)),
                "sentiment_source_count": int(len(sentiment_snapshots)),
                "providers": enrichment_provider_rows,
            }

        _run_ingest_stage(
            "collect_enrichments",
            _collect_enrichments,
            summary_fn=lambda out: out if isinstance(out, dict) else {},
        )

        macro = pd.concat(macro_frames, ignore_index=True) if macro_frames else pd.DataFrame()
        sentiment: dict[str, float] = {}
        def _normalize_enrichments() -> tuple[dict[str, float], list[NewsEvent]]:
            if sentiment_snapshots:
                seen_sources = sorted({src for src, _ in sentiment_snapshots})
                sentiment["sentiment_source_count"] = float(len(seen_sources))
                by_key: dict[str, list[float]] = {}
                for src_name, snap in sentiment_snapshots:
                    for key, val in snap.items():
                        sentiment[f"{src_name}.{key}"] = float(val)
                        by_key.setdefault(str(key), []).append(float(val))
                for key, vals in by_key.items():
                    if not vals:
                        continue
                    arr = np.asarray(vals, dtype=float)
                    sentiment[key] = float(arr.mean())
                    sentiment[f"{key}__median"] = float(np.median(arr))
                    sentiment[f"{key}__std"] = float(arr.std(ddof=0))
            normalized_news = self._dedupe_news(self._normalize_news(news_events))
            return sentiment, normalized_news

        sentiment, news = _run_ingest_stage(
            "normalize_enrichments",
            _normalize_enrichments,
            summary_fn=lambda out: {
                "sentiment_keys": int(len(out[0])) if isinstance(out, tuple) and len(out) > 0 else 0,
                "deduped_news_events": int(len(out[1])) if isinstance(out, tuple) and len(out) > 1 else 0,
                "macro_rows": int(len(macro)),
            },
        )
        source_conf = _run_ingest_stage(
            "evaluate_source_confidence",
            lambda: self._evaluate_source_confidence(
                raw_bars=raw_bars,
                macro=macro,
                news=news,
                sentiment_factor_count=sentiment_factor_count,
            ),
            summary_fn=lambda out: {
                "overall_score": float(out.overall_score),
                "source_count": int(len(out.by_source)),
                "low_confidence_source_count": int(len(out.low_confidence_sources)),
            },
        )
        quality = _run_ingest_stage(
            "evaluate_quality",
            lambda: evaluate_quality(
                normalized_bars=normalized_bars,
                conflicts=conflicts,
                completeness_min=self.completeness_min,
                conflict_max=self.conflict_max,
                source_confidence=source_conf,
                source_confidence_min=self.source_confidence_min,
                low_confidence_source_ratio_max=self.low_confidence_source_ratio_max,
            ),
            summary_fn=lambda out: {
                "quality_passed": bool(out.passed),
                "completeness_ratio": float(out.completeness),
                "unresolved_conflict_ratio": float(out.unresolved_conflict_ratio),
                "source_confidence_score": float(out.source_confidence_score),
            },
        )
        ingest_timing["status"] = "completed"
        ingest_timing["current_stage"] = ""
        ingest_timing.pop("current_stage_started_elapsed_sec", None)
        ingest_timing.pop("current_stage_elapsed_sec", None)
        ingest_timing.pop("failed_stage", None)
        ingest_timing.pop("error", None)
        _persist_ingest_timing()

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
        review_dir = self.output_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        timing_path = review_dir / f"{as_of.isoformat()}_data_persist_timing.json"
        timing_started_mono = time.monotonic()
        persist_timing: dict[str, Any] = {
            "date": as_of.isoformat(),
            "status": "running",
            "elapsed_sec": 0.0,
            "current_stage": "",
            "last_completed_stage": "",
            "stages": [],
        }

        def _persist_timing() -> None:
            persist_timing["elapsed_sec"] = round(time.monotonic() - timing_started_mono, 3)
            write_json(timing_path, persist_timing)

        def _run_persist_stage(
            name: str,
            fn: Any,
            *,
            summary_fn: Any = None,
        ) -> Any:
            stage_started_mono = time.monotonic()
            persist_timing["status"] = "running"
            persist_timing["current_stage"] = name
            persist_timing["current_stage_started_elapsed_sec"] = round(stage_started_mono - timing_started_mono, 3)
            _persist_timing()
            try:
                out = fn()
            except Exception as exc:
                persist_timing["status"] = "failed"
                persist_timing["failed_stage"] = name
                persist_timing["error"] = f"{type(exc).__name__}:{exc}"
                persist_timing["current_stage_elapsed_sec"] = round(time.monotonic() - stage_started_mono, 3)
                _persist_timing()
                raise
            stage_row: dict[str, Any] = {
                "name": name,
                "elapsed_sec": round(time.monotonic() - stage_started_mono, 3),
            }
            if summary_fn is not None:
                summary = summary_fn(out)
                if summary:
                    stage_row["summary"] = summary
            persist_timing.setdefault("stages", []).append(stage_row)
            persist_timing["last_completed_stage"] = name
            persist_timing["current_stage"] = ""
            persist_timing.pop("current_stage_started_elapsed_sec", None)
            persist_timing.pop("current_stage_elapsed_sec", None)
            persist_timing.pop("failed_stage", None)
            persist_timing.pop("error", None)
            _persist_timing()
            return out

        dstr = as_of.isoformat()
        raw_path = self.output_dir / "artifacts" / "raw" / f"{dstr}_bars_raw.csv"
        norm_path = self.output_dir / "artifacts" / "normalized" / f"{dstr}_bars_normalized.csv"
        feat_path = self.output_dir / "artifacts" / "feature" / f"{dstr}_bars_feature.parquet"
        conf_path = self.output_dir / "artifacts" / "normalized" / f"{dstr}_conflicts.csv"
        macro_path = self.output_dir / "artifacts" / "raw" / f"{dstr}_macro.csv"
        news_path = self.output_dir / "artifacts" / "raw" / f"{dstr}_news.json"
        source_conf_path = self.output_dir / "artifacts" / "normalized" / f"{dstr}_source_confidence.json"
        sentiment_path = self.output_dir / "artifacts" / "normalized" / f"{dstr}_sentiment.json"
        quality_path = self.output_dir / "artifacts" / f"{dstr}_quality.json"

        _run_persist_stage(
            "write_artifacts",
            lambda: (
                write_csv(raw_path, result.raw_bars),
                write_csv(norm_path, result.normalized_bars),
                write_csv(conf_path, result.conflicts if not result.conflicts.empty else pd.DataFrame(columns=["ts", "symbol", "field", "values", "max_diff_pct"])),
                write_csv(macro_path, result.macro if not result.macro.empty else pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"])),
                write_json(news_path, [n.to_dict() for n in result.news]),
                write_json(source_conf_path, result.source_confidence.to_dict()),
                write_json(sentiment_path, {k: float(v) for k, v in result.sentiment.items()}),
                write_json(quality_path, result.quality.to_dict()),
            ),
            summary_fn=lambda _out: {
                "raw_rows": int(len(result.raw_bars)),
                "normalized_rows": int(len(result.normalized_bars)),
                "news_events": int(len(result.news)),
            },
        )

        feature_df = result.normalized_bars.copy()
        if not feature_df.empty:
            feature_df["ret_1d"] = feature_df.groupby("symbol")["close"].pct_change().fillna(0.0)
            feature_df["vol_chg"] = feature_df.groupby("symbol")["volume"].pct_change().fillna(0.0)
        _run_persist_stage(
            "write_feature_parquet",
            lambda: write_parquet_optional(feat_path, feature_df),
            summary_fn=lambda _out: {"feature_rows": int(len(feature_df)), "path": str(feat_path)},
        )

        _run_persist_stage(
            "append_sqlite_core",
            lambda: (
                append_sqlite(self.sqlite_path, "bars_normalized", result.normalized_bars),
                append_sqlite(self.sqlite_path, "macro", result.macro if not result.macro.empty else pd.DataFrame(columns=["date", "cpi_yoy", "ppi_yoy", "lpr_1y", "source"])),
            ),
            summary_fn=lambda _out: {
                "bars_normalized_rows": int(len(result.normalized_bars)),
                "macro_rows": int(len(result.macro)),
                "sqlite_path": str(self.sqlite_path),
            },
        )
        news_rows = [n.to_dict() | {"as_of": dstr} for n in result.news]
        _run_persist_stage(
            "append_sqlite_news",
            lambda: append_sqlite(
                self.sqlite_path,
                "news_events",
                (
                    pd.DataFrame(news_rows)
                    if news_rows
                    else pd.DataFrame(
                        columns=[
                            "event_id",
                            "ts",
                            "title",
                            "content",
                            "lang",
                            "source",
                            "category",
                            "confidence",
                            "entities",
                            "importance",
                            "as_of",
                        ]
                    )
                ),
            ),
            summary_fn=lambda _out: {"news_rows": int(len(news_rows))},
        )
        sentiment_rows = []
        for key, value in result.sentiment.items():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            sentiment_rows.append({"as_of": dstr, "key": str(key), "value": float(val)})
        _run_persist_stage(
            "append_sqlite_sentiment",
            lambda: append_sqlite(
                self.sqlite_path,
                "sentiment_snapshot",
                (
                    pd.DataFrame(sentiment_rows)
                    if sentiment_rows
                    else pd.DataFrame(columns=["as_of", "key", "value"])
                ),
            ),
            summary_fn=lambda _out: {"sentiment_rows": int(len(sentiment_rows))},
        )
        source_conf_df = pd.DataFrame([d.to_dict() | {"as_of": dstr} for d in result.source_confidence.details])
        if source_conf_df.empty:
            source_conf_df = pd.DataFrame(columns=["source", "score", "base_reliability", "bar_consistency", "bar_coverage", "macro_consistency", "macro_coverage", "news_confidence", "sentiment_coverage", "bars_rows", "macro_rows", "news_events", "sentiment_factors", "as_of"])
        _run_persist_stage(
            "append_sqlite_source_confidence",
            lambda: append_sqlite(self.sqlite_path, "source_confidence", source_conf_df),
            summary_fn=lambda _out: {"source_confidence_rows": int(len(source_conf_df))},
        )
        _run_persist_stage(
            "append_sqlite_quality",
            lambda: append_sqlite(self.sqlite_path, "quality", pd.DataFrame([result.quality.to_dict() | {"as_of": dstr}])),
            summary_fn=lambda _out: {"quality_rows": 1},
        )
        persist_timing["status"] = "completed"
        persist_timing["current_stage"] = ""
        persist_timing.pop("current_stage_started_elapsed_sec", None)
        persist_timing.pop("current_stage_elapsed_sec", None)
        persist_timing.pop("failed_stage", None)
        persist_timing.pop("error", None)
        _persist_timing()
