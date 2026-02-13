from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.quality import evaluate_quality


class DataQualityTests(unittest.TestCase):
    def test_quality_pass(self) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.bdate_range("2026-01-01", periods=5),
                "symbol": ["300750"] * 5,
                "open": [1, 2, 3, 4, 5],
                "high": [2, 3, 4, 5, 6],
                "low": [0.5, 1.5, 2.5, 3.5, 4.5],
                "close": [1.2, 2.1, 3.0, 3.9, 5.2],
                "volume": [100, 120, 130, 140, 150],
                "asset_class": ["equity"] * 5,
            }
        )
        conflicts = pd.DataFrame(columns=["ts", "symbol", "field", "values", "max_diff_pct"])
        report = evaluate_quality(df, conflicts, 0.99, 0.005)
        self.assertTrue(report.passed)

    def test_quality_flags_weekend(self) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-01-03", "2026-01-04"]),  # weekend
                "symbol": ["513130", "513130"],
                "open": [1, 1],
                "high": [2, 2],
                "low": [0.5, 0.5],
                "close": [130, 130],
                "volume": [100, 100],
                "asset_class": ["etf", "etf"],
            }
        )
        conflicts = pd.DataFrame({"x": [1]})
        report = evaluate_quality(df, conflicts, 0.99, 0.005)
        self.assertFalse(report.passed)
        self.assertTrue(any("TRADING_DAY_MISMATCH_WEEKEND" in f for f in report.flags))


if __name__ == "__main__":
    unittest.main()
