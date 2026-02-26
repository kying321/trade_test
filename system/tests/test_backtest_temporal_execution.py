from __future__ import annotations

import sys
from pathlib import Path
import unittest
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.backtest.engine import BacktestConfig, run_event_backtest
from tests.helpers import make_multi_symbol_bars


class BacktestTemporalExecutionTests(unittest.TestCase):
    def test_execution_friction_reduces_return(self) -> None:
        bars = make_multi_symbol_bars()
        base = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=3, signal_confidence_min=30, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        stressed = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(
                max_daily_trades=2,
                hold_days=3,
                signal_confidence_min=30,
                convexity_min=1.2,
                execution_friction_multiplier=1.5,
                execution_extra_slippage_bps=15.0,
            ),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        self.assertLessEqual(stressed.total_return, base.total_return + 1e-12)

    def test_latency_does_not_increase_trade_count(self) -> None:
        bars = make_multi_symbol_bars()
        baseline = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=3, signal_confidence_min=30, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        lagged = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(
                max_daily_trades=2,
                hold_days=3,
                signal_confidence_min=30,
                convexity_min=1.2,
                execution_latency_days=2,
            ),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        self.assertLessEqual(int(lagged.trades), int(baseline.trades))

    def test_negative_latency_is_clamped_to_zero(self) -> None:
        bars = make_multi_symbol_bars()
        baseline = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=3, signal_confidence_min=30, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        neg_latency = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(
                max_daily_trades=2,
                hold_days=3,
                signal_confidence_min=30,
                convexity_min=1.2,
                execution_latency_days=-5,
            ),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        self.assertAlmostEqual(float(neg_latency.total_return), float(baseline.total_return), places=8)
        self.assertEqual(int(neg_latency.trades), int(baseline.trades))


if __name__ == "__main__":
    unittest.main()

