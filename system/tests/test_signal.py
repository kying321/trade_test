from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.models import RegimeLabel
from lie_engine.signal import SignalEngineConfig, scan_signals
from tests.helpers import make_bars


class SignalTests(unittest.TestCase):
    def test_trend_signal_generation(self) -> None:
        bars = make_bars("300750", n=260, trend=0.15, seed=1)
        cfg = SignalEngineConfig(confidence_min=10, convexity_min=1.0)
        signals = scan_signals(bars, regime=RegimeLabel.STRONG_TREND, cfg=cfg)
        self.assertGreaterEqual(len(signals), 1)

    def test_range_signal_generation(self) -> None:
        bars = make_bars("513130", n=260, trend=0.0, seed=9, asset_class="etf")
        cfg = SignalEngineConfig(confidence_min=30, convexity_min=1.2)
        signals = scan_signals(bars, regime=RegimeLabel.RANGE, cfg=cfg)
        self.assertIsInstance(signals, list)


if __name__ == "__main__":
    unittest.main()
