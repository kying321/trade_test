from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.models import RegimeLabel, Side, SignalCandidate
from lie_engine.risk import RiskManager, compute_kelly_fraction


class RiskTests(unittest.TestCase):
    def test_kelly_formula(self) -> None:
        f = compute_kelly_fraction(win_rate=0.45, payoff=2.5)
        self.assertAlmostEqual(f, 0.23, places=2)

    def test_risk_manager_caps(self) -> None:
        manager = RiskManager(2.0, 50.0, 15.0, 25.0)
        signal = SignalCandidate(
            symbol="300750",
            side=Side.LONG,
            regime=RegimeLabel.STRONG_TREND,
            position_score=4,
            structure_score=5,
            momentum_score=4,
            confidence=80.0,
            convexity_ratio=3.2,
            entry_price=50.0,
            stop_price=48.0,
            target_price=56.0,
            can_short=False,
        )
        budget = manager.build_budget(1_000_000, used_exposure_pct=0.0)
        plan = manager.size_signal(signal, win_rate=0.45, payoff=2.5, budget=budget, symbol_exposure_pct=0.0, theme_exposure_pct=0.0, protection_mode=False)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertLessEqual(plan.size_pct, 15.0)
        self.assertLessEqual(plan.risk_pct, 2.0)

    def test_short_translation_for_non_shortable(self) -> None:
        manager = RiskManager(2.0, 50.0, 15.0, 25.0)
        signal = SignalCandidate(
            symbol="300750",
            side=Side.SHORT,
            regime=RegimeLabel.DOWNTREND,
            position_score=4,
            structure_score=5,
            momentum_score=4,
            confidence=75.0,
            convexity_ratio=3.4,
            entry_price=50.0,
            stop_price=52.0,
            target_price=44.0,
            can_short=False,
            notes="translate",
        )
        budget = manager.build_budget(1_000_000, used_exposure_pct=0.0)
        plan = manager.size_signal(signal, win_rate=0.5, payoff=2.0, budget=budget, symbol_exposure_pct=0.0, theme_exposure_pct=0.0, protection_mode=False)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.status, "TRANSLATED")


if __name__ == "__main__":
    unittest.main()
