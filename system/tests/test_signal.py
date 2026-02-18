from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.models import RegimeLabel, Side, SignalCandidate
from lie_engine.signal import SignalEngineConfig, scan_signals
from lie_engine.signal.engine import generate_signal_for_symbol
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

    def test_scan_uses_symbol_level_regime(self) -> None:
        bars = pd.concat(
            [
                make_bars("300750", n=260, trend=0.15, seed=10),
                make_bars("513130", n=260, trend=0.0, seed=11, asset_class="etf"),
            ],
            ignore_index=True,
        )
        cfg = SignalEngineConfig(confidence_min=1, convexity_min=0.5)

        def _fake_candidate(symbol_df, regime, cfg, market_factor_state=None):  # type: ignore[no-untyped-def]
            cur = symbol_df.iloc[-1]
            return SignalCandidate(
                symbol=str(cur["symbol"]),
                side=Side.LONG,
                regime=regime,
                position_score=1.0,
                structure_score=1.0,
                momentum_score=1.0,
                confidence=80.0,
                convexity_ratio=2.0,
                entry_price=float(cur["close"]),
                stop_price=float(cur["close"]) * 0.98,
                target_price=float(cur["close"]) * 1.04,
                can_short=True,
            )

        with (
            patch(
                "lie_engine.signal.engine.detect_symbol_regime",
                side_effect=[RegimeLabel.STRONG_TREND, RegimeLabel.RANGE],
            ),
            patch(
                "lie_engine.signal.engine.generate_signal_for_symbol",
                side_effect=_fake_candidate,
            ),
        ):
            signals = scan_signals(bars=bars, regime=RegimeLabel.WEAK_TREND, cfg=cfg)

        regimes = {s.regime for s in signals}
        self.assertEqual(regimes, {RegimeLabel.STRONG_TREND, RegimeLabel.RANGE})

    def test_weak_trend_target_tighter_than_strong(self) -> None:
        bars = make_bars("300750", n=280, trend=0.12, seed=21)
        cfg = SignalEngineConfig(confidence_min=5, convexity_min=0.5, factor_filter_enabled=False)

        strong = generate_signal_for_symbol(bars, regime=RegimeLabel.STRONG_TREND, cfg=cfg)
        weak = generate_signal_for_symbol(bars, regime=RegimeLabel.WEAK_TREND, cfg=cfg)
        self.assertIsNotNone(strong)
        self.assertIsNotNone(weak)
        assert strong is not None
        assert weak is not None

        strong_reward = abs(strong.target_price - strong.entry_price)
        weak_reward = abs(weak.target_price - weak.entry_price)
        self.assertLessEqual(weak_reward, strong_reward + 1e-9)

    def test_factor_penalty_applies_under_valuation_pressure(self) -> None:
        bars = make_bars("300750", n=300, trend=0.25, seed=33)
        bars["pe_ttm"] = 120.0
        cfg = SignalEngineConfig(
            confidence_min=0,
            convexity_min=0.0,
            factor_penalty_max=45.0,
            factor_drop_threshold=2.0,
        )

        signal = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            market_factor_state={
                "valuation_pressure": 1.2,
                "momentum_preference": 0.7,
                "crowding_aversion": 0.6,
                "small_cap_pressure": 0.6,
            },
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertGreater(signal.factor_penalty, 0.0)
        self.assertGreater(signal.factor_exposure_score, 0.0)
        self.assertIn("valuation_headwind", signal.factor_flags)

    def test_dividend_style_headwind_flag(self) -> None:
        bars = make_bars("300750", n=260, trend=0.10, seed=44)
        bars["dividend_yield"] = 0.0
        cfg = SignalEngineConfig(
            confidence_min=0,
            convexity_min=0.0,
            factor_penalty_max=40.0,
            factor_drop_threshold=2.0,
        )
        signal = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.WEAK_TREND,
            cfg=cfg,
            market_factor_state={
                "valuation_pressure": 0.5,
                "momentum_preference": 0.6,
                "crowding_aversion": 0.5,
                "small_cap_pressure": 0.4,
                "dividend_preference": 1.2,
            },
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertIn("low_dividend_headwind", signal.factor_flags)


if __name__ == "__main__":
    unittest.main()
