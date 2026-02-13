from __future__ import annotations

import sys
from pathlib import Path
import unittest
from datetime import date

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.backtest.engine import BacktestConfig, _compute_metrics, run_event_backtest
from tests.helpers import make_multi_symbol_bars


class BacktestTests(unittest.TestCase):
    def test_backtest_runs(self) -> None:
        bars = make_multi_symbol_bars()
        result = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=3, signal_confidence_min=30, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        self.assertGreaterEqual(result.trades, 0)
        self.assertGreaterEqual(result.max_drawdown, 0.0)

    def test_backtest_reproducible(self) -> None:
        bars = make_multi_symbol_bars()
        r1 = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=3, signal_confidence_min=30, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        r2 = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=3, signal_confidence_min=30, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        self.assertAlmostEqual(r1.total_return, r2.total_return, places=8)

    def test_positive_window_ratio_uses_neutral_band(self) -> None:
        equity = pd.Series([1.0, 0.9998, 0.9996, 1.0000])
        trades = pd.DataFrame({"pnl": [-0.0004, 0.0001, -0.0002]})
        window_returns = [-0.0007, -0.0003, 0.0002]
        metrics = _compute_metrics(equity, trades, window_returns)
        positive_window_ratio = metrics[-1]
        self.assertAlmostEqual(positive_window_ratio, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
