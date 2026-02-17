from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class SourceConfidenceItem:
    source: str
    score: float
    base_reliability: float
    bar_consistency: float
    bar_coverage: float
    news_confidence: float
    sentiment_coverage: float
    bars_rows: int
    news_events: int
    sentiment_factors: int
    macro_consistency: float = 0.0
    macro_coverage: float = 0.0
    macro_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "score": float(self.score),
            "base_reliability": float(self.base_reliability),
            "bar_consistency": float(self.bar_consistency),
            "bar_coverage": float(self.bar_coverage),
            "macro_consistency": float(self.macro_consistency),
            "macro_coverage": float(self.macro_coverage),
            "news_confidence": float(self.news_confidence),
            "sentiment_coverage": float(self.sentiment_coverage),
            "bars_rows": int(self.bars_rows),
            "macro_rows": int(self.macro_rows),
            "news_events": int(self.news_events),
            "sentiment_factors": int(self.sentiment_factors),
        }


@dataclass(slots=True)
class SourceConfidenceReport:
    overall_score: float
    by_source: dict[str, float] = field(default_factory=dict)
    low_confidence_sources: list[str] = field(default_factory=list)
    details: list[SourceConfidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": float(self.overall_score),
            "by_source": {k: float(v) for k, v in self.by_source.items()},
            "low_confidence_sources": list(self.low_confidence_sources),
            "details": [d.to_dict() for d in self.details],
        }


@dataclass(slots=True)
class DataQualityReport:
    completeness: float
    unresolved_conflict_ratio: float
    source_confidence_score: float = 1.0
    low_confidence_source_ratio: float = 0.0
    source_confidence: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.flags) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "completeness": self.completeness,
            "unresolved_conflict_ratio": self.unresolved_conflict_ratio,
            "source_confidence_score": self.source_confidence_score,
            "low_confidence_source_ratio": self.low_confidence_source_ratio,
            "source_confidence": {k: float(v) for k, v in self.source_confidence.items()},
            "flags": list(self.flags),
            "passed": self.passed,
        }


def evaluate_quality(
    normalized_bars: pd.DataFrame,
    conflicts: pd.DataFrame,
    completeness_min: float,
    conflict_max: float,
    source_confidence: SourceConfidenceReport | None = None,
    source_confidence_min: float = 0.70,
    low_confidence_source_ratio_max: float = 0.5,
) -> DataQualityReport:
    required_cols = ["ts", "symbol", "open", "high", "low", "close", "volume", "asset_class"]
    total = len(normalized_bars) * len(required_cols)
    missing = int(normalized_bars[required_cols].isna().sum().sum()) if len(normalized_bars) else total
    completeness = 1.0 if total == 0 else max(0.0, 1 - missing / total)

    unresolved_conflict_ratio = 0.0
    if len(normalized_bars):
        if conflicts.empty:
            unresolved_conflict_ratio = 0.0
        else:
            unique_conflicts = conflicts[[c for c in ["ts", "symbol"] if c in conflicts.columns]].drop_duplicates()
            unresolved_conflict_ratio = len(unique_conflicts) / len(normalized_bars)

    flags: list[str] = []
    if completeness < completeness_min:
        flags.append(f"DATA_COMPLETENESS_LOW:{completeness:.4f}")
    if unresolved_conflict_ratio > conflict_max:
        flags.append(f"UNRESOLVED_CONFLICT_HIGH:{unresolved_conflict_ratio:.4f}")

    source_score = 1.0
    low_conf_ratio = 0.0
    source_map: dict[str, float] = {}
    if source_confidence is not None and source_confidence.by_source:
        source_map = {k: float(v) for k, v in source_confidence.by_source.items()}
        source_score = float(source_confidence.overall_score)
        low_count = sum(1 for v in source_map.values() if float(v) < float(source_confidence_min))
        low_conf_ratio = float(low_count / max(1, len(source_map)))
        if source_score < float(source_confidence_min):
            flags.append(f"SOURCE_CONFIDENCE_LOW:{source_score:.4f}")
        if low_conf_ratio > float(low_confidence_source_ratio_max):
            flags.append(f"LOW_CONFIDENCE_SOURCE_RATIO_HIGH:{low_conf_ratio:.4f}")

    if len(normalized_bars):
        ts = pd.to_datetime(normalized_bars["ts"], utc=False)
        ordered = normalized_bars.copy()
        ordered["ts"] = ts
        ordered = ordered.sort_values(["symbol", "ts"])
        drift = ordered.groupby("symbol")["ts"].diff().dropna()
        if not drift.empty and (drift.dt.total_seconds() < 0).any():
            flags.append("TIMESTAMP_DRIFT_NEGATIVE")

        weekend_rows = ts.dt.dayofweek >= 5
        if weekend_rows.any():
            flags.append("TRADING_DAY_MISMATCH_WEEKEND")

        etf_rows = normalized_bars[normalized_bars["asset_class"] == "etf"]
        if not etf_rows.empty and (etf_rows["close"] > 100).mean() > 0.2:
            flags.append("ETF_INDEX_UNIT_CONFUSION")

        futures_rows = normalized_bars[normalized_bars["asset_class"] == "future"]
        if not futures_rows.empty:
            bad_future_symbol = ~futures_rows["symbol"].str.match(r"^[A-Z]{1,3}\d{4}$")
            if bad_future_symbol.any():
                flags.append("CONTRACT_MONTH_MISMATCH")

    return DataQualityReport(
        completeness=completeness,
        unresolved_conflict_ratio=unresolved_conflict_ratio,
        source_confidence_score=source_score,
        low_confidence_source_ratio=low_conf_ratio,
        source_confidence=source_map,
        flags=flags,
    )
