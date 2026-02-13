from __future__ import annotations

import sys
from pathlib import Path
import unittest
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.models import BacktestResult
from lie_engine.review import ReviewThresholds, build_review_delta


class ReviewTests(unittest.TestCase):
    @staticmethod
    def _backtest(
        *,
        positive_window_ratio: float,
        expectancy: float = 0.1,
        max_drawdown: float = 0.08,
        violations: int = 0,
    ) -> BacktestResult:
        return BacktestResult(
            start=date(2020, 1, 1),
            end=date(2026, 2, 13),
            total_return=0.22,
            annual_return=0.08,
            max_drawdown=max_drawdown,
            win_rate=0.48,
            profit_factor=1.7,
            expectancy=expectancy,
            trades=240,
            violations=violations,
            positive_window_ratio=positive_window_ratio,
            equity_curve=[],
            by_asset={},
        )

    def test_failed_gate_confidence_recovers_to_band(self) -> None:
        bt = self._backtest(positive_window_ratio=0.66, expectancy=0.12)
        delta = build_review_delta(
            as_of=date(2026, 2, 13),
            backtest=bt,
            current_params={"win_rate": 0.45, "payoff": 2.0, "signal_confidence_min": 90.0},
            factor_weights={"macro": 0.25, "industry": 0.2, "sentiment": 0.15, "fundamental": 0.15, "technical": 0.25},
            factor_contrib={"macro": 0.2, "industry": 0.2, "sentiment": 0.2, "fundamental": 0.2, "technical": 0.2},
            thresholds=ReviewThresholds(positive_window_ratio_min=0.70, max_drawdown_max=0.18),
        )
        self.assertFalse(delta.pass_gate)
        conf = delta.parameter_changes["signal_confidence_min"]
        self.assertGreaterEqual(conf, 55.0)
        self.assertLessEqual(conf, 70.0)

    def test_pass_gate_keeps_bounded_update(self) -> None:
        bt = self._backtest(positive_window_ratio=0.78, expectancy=0.15)
        delta = build_review_delta(
            as_of=date(2026, 2, 13),
            backtest=bt,
            current_params={"win_rate": 0.45, "payoff": 2.0, "signal_confidence_min": 62.0},
            factor_weights={"macro": 0.25, "industry": 0.2, "sentiment": 0.15, "fundamental": 0.15, "technical": 0.25},
            factor_contrib={"macro": 0.2, "industry": 0.2, "sentiment": 0.2, "fundamental": 0.2, "technical": 0.2},
            thresholds=ReviewThresholds(positive_window_ratio_min=0.70, max_drawdown_max=0.18),
        )
        self.assertTrue(delta.pass_gate)
        conf = delta.parameter_changes["signal_confidence_min"]
        self.assertGreaterEqual(conf, 50.0)
        self.assertLessEqual(conf, 90.0)


if __name__ == "__main__":
    unittest.main()
