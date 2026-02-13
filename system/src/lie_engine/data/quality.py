from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DataQualityReport:
    completeness: float
    unresolved_conflict_ratio: float
    flags: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.flags) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "completeness": self.completeness,
            "unresolved_conflict_ratio": self.unresolved_conflict_ratio,
            "flags": list(self.flags),
            "passed": self.passed,
        }


def evaluate_quality(
    normalized_bars: pd.DataFrame,
    conflicts: pd.DataFrame,
    completeness_min: float,
    conflict_max: float,
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
        flags=flags,
    )
