from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.models import RegimeLabel, Side, SignalCandidate
from lie_engine.signal import SignalEngineConfig, scan_signals
from lie_engine.signal.engine import detect_symbol_regime, generate_signal_for_symbol
from lie_engine.signal.features import add_common_features
from lie_engine.signal.theory import compute_theory_confluence
from tests.helpers import make_bars


class SignalTests(unittest.TestCase):
    def test_detect_symbol_regime_invalid_ts_returns_uncertain(self) -> None:
        bars = make_bars("300750", n=260, trend=0.10, seed=99)
        bars["ts"] = "invalid-ts"
        cfg = SignalEngineConfig(confidence_min=10, convexity_min=1.0)
        regime = detect_symbol_regime(bars, cfg)
        self.assertEqual(regime, RegimeLabel.UNCERTAIN)

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

        def _fake_candidate(symbol_df, regime, cfg, market_factor_state=None, micro_factor_state=None):  # type: ignore[no-untyped-def]
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

    def test_cross_source_quality_headwind_penalizes_signal(self) -> None:
        bars = make_bars("300750", n=300, trend=0.18, seed=45)
        cfg = SignalEngineConfig(
            confidence_min=0,
            convexity_min=0.0,
            factor_penalty_max=45.0,
            factor_drop_threshold=2.0,
        )
        baseline = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            market_factor_state={
                "valuation_pressure": 0.8,
                "momentum_preference": 0.6,
                "crowding_aversion": 0.5,
                "small_cap_pressure": 0.5,
                "dividend_preference": 0.5,
            },
        )
        stressed = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            market_factor_state={
                "valuation_pressure": 0.8,
                "momentum_preference": 0.6,
                "crowding_aversion": 0.5,
                "small_cap_pressure": 0.5,
                "dividend_preference": 0.5,
                "cross_source_quality_score_7d": 0.20,
                "cross_source_fail_ratio_7d": 0.80,
                "cross_source_stress": 1.20,
            },
        )
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(stressed)
        assert baseline is not None
        assert stressed is not None
        self.assertGreater(float(stressed.factor_penalty), float(baseline.factor_penalty))
        self.assertLess(float(stressed.confidence), float(baseline.confidence))
        self.assertIn("data_quality_headwind", stressed.factor_flags)

    def test_flow_effort_result_divergence_penalizes_signal(self) -> None:
        baseline_bars = make_bars("BTCUSDT", n=320, trend=0.14, seed=71, asset_class="future").reset_index(drop=True)
        stressed_bars = baseline_bars.copy(deep=True)
        i = len(stressed_bars) - 1
        anchor = float(stressed_bars.loc[i - 1, "close"])
        open_px = anchor * 1.005
        stressed_bars.loc[i, "open"] = open_px
        stressed_bars.loc[i, "close"] = open_px * 1.0002
        stressed_bars.loc[i, "high"] = open_px * 1.018
        stressed_bars.loc[i, "low"] = open_px * 0.995
        stressed_bars.loc[i, "volume"] = float(stressed_bars["volume"].tail(20).mean()) * 3.6

        cfg = SignalEngineConfig(
            confidence_min=0.0,
            convexity_min=0.0,
            factor_penalty_max=45.0,
            factor_drop_threshold=2.0,
            microstructure_enabled=False,
        )
        market_state = {
            "valuation_pressure": 0.5,
            "momentum_preference": 0.6,
            "crowding_aversion": 0.6,
            "small_cap_pressure": 0.4,
            "dividend_preference": 0.4,
            "flow_quality_pressure": 1.25,
        }
        baseline = generate_signal_for_symbol(
            baseline_bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            market_factor_state=market_state,
        )
        stressed = generate_signal_for_symbol(
            stressed_bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            market_factor_state=market_state,
        )
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(stressed)
        assert baseline is not None
        assert stressed is not None
        self.assertLess(float(stressed.confidence), float(baseline.confidence))
        self.assertIn("flow_effort_result_divergence", stressed.factor_flags)
        self.assertIn("flow=", stressed.notes)

    def test_flow_absorption_reversal_penalizes_trend_long(self) -> None:
        bars = make_bars("BTCUSDT", n=320, trend=0.20, seed=72, asset_class="future").reset_index(drop=True)
        i = len(bars) - 1
        anchor = float(bars.loc[i - 1, "close"])
        bars.loc[i, "open"] = anchor * 1.004
        bars.loc[i, "close"] = anchor * 1.007
        bars.loc[i, "high"] = anchor * 1.032
        bars.loc[i, "low"] = anchor * 0.999
        bars.loc[i, "volume"] = float(bars["volume"].tail(20).mean()) * 2.8

        cfg = SignalEngineConfig(
            confidence_min=0.0,
            convexity_min=0.0,
            factor_penalty_max=45.0,
            factor_drop_threshold=2.0,
            microstructure_enabled=False,
        )
        signal = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            market_factor_state={
                "valuation_pressure": 0.5,
                "momentum_preference": 0.6,
                "crowding_aversion": 0.5,
                "small_cap_pressure": 0.4,
                "dividend_preference": 0.4,
                "flow_quality_pressure": 1.2,
            },
        )
        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.side, Side.LONG)
        self.assertIn("flow_absorption_reversal", signal.factor_flags)

    def test_micro_time_sync_risk_penalizes_signal(self) -> None:
        bars = make_bars("BTCUSDT", n=280, trend=0.12, seed=46, asset_class="future")
        cfg = SignalEngineConfig(
            confidence_min=0.0,
            convexity_min=0.0,
            factor_filter_enabled=False,
            microstructure_enabled=True,
            micro_confidence_boost_max=8.0,
            micro_penalty_max=10.0,
            micro_min_trade_count=10,
        )
        aligned = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            micro_factor_state={
                "has_data": True,
                "schema_ok": True,
                "sync_ok": True,
                "gap_ok": True,
                "time_sync_ok": True,
                "micro_alignment": 0.8,
                "evidence_score": 1.0,
                "trade_count": 120,
            },
        )
        drifted = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            micro_factor_state={
                "has_data": True,
                "schema_ok": True,
                "sync_ok": True,
                "gap_ok": True,
                "time_sync_ok": False,
                "micro_alignment": 0.8,
                "evidence_score": 1.0,
                "trade_count": 120,
            },
        )
        self.assertIsNotNone(aligned)
        self.assertIsNotNone(drifted)
        assert aligned is not None
        assert drifted is not None
        self.assertLess(float(drifted.confidence), float(aligned.confidence))
        self.assertIn("micro_time_sync_risk", drifted.factor_flags)

    def test_theory_confluence_adds_confidence_on_aligned_setup(self) -> None:
        bars = make_bars("BTCUSDT", n=320, trend=0.16, seed=52, asset_class="future").reset_index(drop=True)
        i = len(bars) - 1
        prev = len(bars) - 2
        anchor = float(bars.loc[prev, "high"])
        bars.loc[i, "open"] = anchor * 1.01
        bars.loc[i, "low"] = anchor * 1.02
        bars.loc[i, "close"] = anchor * 1.06
        bars.loc[i, "high"] = anchor * 1.07
        bars.loc[i, "volume"] = float(bars["volume"].tail(20).mean()) * 2.4

        base_cfg = SignalEngineConfig(
            confidence_min=0.0,
            convexity_min=0.0,
            factor_filter_enabled=False,
            microstructure_enabled=False,
            theory_enabled=False,
        )
        theory_cfg = SignalEngineConfig(
            confidence_min=0.0,
            convexity_min=0.0,
            factor_filter_enabled=False,
            microstructure_enabled=False,
            theory_enabled=True,
            theory_confidence_boost_max=9.0,
            theory_penalty_max=1.0,
        )
        base = generate_signal_for_symbol(bars, regime=RegimeLabel.STRONG_TREND, cfg=base_cfg)
        enriched = generate_signal_for_symbol(bars, regime=RegimeLabel.STRONG_TREND, cfg=theory_cfg)
        self.assertIsNotNone(base)
        self.assertIsNotNone(enriched)
        assert base is not None
        assert enriched is not None
        self.assertGreater(float(enriched.confidence), float(base.confidence))
        self.assertIn("theory=", enriched.notes)

    def test_theory_brooks_trendbar_microchannel_biases_long(self) -> None:
        bars = make_bars("BTCUSDT", n=320, trend=0.10, seed=67, asset_class="future").reset_index(drop=True)
        start = len(bars) - 4
        anchor = float(bars.loc[start - 1, "close"])
        for step in range(4):
            i = start + step
            open_px = anchor * (1.004 + 0.005 * step)
            close_px = open_px * (1.006 + 0.001 * step)
            low_px = open_px * 0.997
            high_px = close_px * 1.002
            bars.loc[i, "open"] = open_px
            bars.loc[i, "close"] = close_px
            bars.loc[i, "low"] = low_px
            bars.loc[i, "high"] = high_px
        bars.loc[len(bars) - 1, "volume"] = float(bars["volume"].tail(20).mean()) * 2.2

        featured = add_common_features(bars)
        long_result = compute_theory_confluence(
            df=featured,
            side=Side.LONG,
            regime=RegimeLabel.STRONG_TREND,
            lie_score_ratio=0.56,
            ict_weight=0.8,
            brooks_weight=1.8,
            lie_weight=0.8,
        )
        short_result = compute_theory_confluence(
            df=featured,
            side=Side.SHORT,
            regime=RegimeLabel.STRONG_TREND,
            lie_score_ratio=0.56,
            ict_weight=0.8,
            brooks_weight=1.8,
            lie_weight=0.8,
        )
        self.assertGreater(float(long_result.brooks_align), float(long_result.brooks_oppose))
        self.assertGreater(float(long_result.brooks_align), 0.10)
        self.assertGreater(float(long_result.brooks_align), float(short_result.brooks_align))

    def test_theory_brooks_two_legged_pullback_supports_long(self) -> None:
        bars = make_bars("BTCUSDT", n=320, trend=0.14, seed=68, asset_class="future").reset_index(drop=True)
        seq_start = len(bars) - 6
        anchor = float(bars.loc[seq_start - 1, "close"])
        closes = [anchor * 1.010, anchor * 0.998, anchor * 1.004, anchor * 0.994, anchor * 1.001]
        for step, close_px in enumerate(closes):
            i = seq_start + step
            open_px = close_px * (0.998 if step in {1, 3} else 1.001)
            low_px = min(open_px, close_px) * 0.997
            high_px = max(open_px, close_px) * 1.003
            bars.loc[i, "open"] = open_px
            bars.loc[i, "close"] = close_px
            bars.loc[i, "low"] = low_px
            bars.loc[i, "high"] = high_px
        last = len(bars) - 1
        prev_high = float(bars.loc[last - 1, "high"])
        bars.loc[last, "open"] = prev_high * 0.998
        bars.loc[last, "close"] = prev_high * 1.008
        bars.loc[last, "low"] = prev_high * 0.995
        bars.loc[last, "high"] = prev_high * 1.011
        bars.loc[last, "volume"] = float(bars["volume"].tail(20).mean()) * 1.8

        featured = add_common_features(bars)
        long_result = compute_theory_confluence(
            df=featured,
            side=Side.LONG,
            regime=RegimeLabel.STRONG_TREND,
            lie_score_ratio=0.57,
            ict_weight=0.7,
            brooks_weight=2.0,
            lie_weight=0.7,
        )
        short_result = compute_theory_confluence(
            df=featured,
            side=Side.SHORT,
            regime=RegimeLabel.STRONG_TREND,
            lie_score_ratio=0.57,
            ict_weight=0.7,
            brooks_weight=2.0,
            lie_weight=0.7,
        )
        self.assertGreater(float(long_result.brooks_align), 0.12)
        self.assertGreater(float(long_result.brooks_align), float(long_result.brooks_oppose))
        self.assertGreater(float(long_result.brooks_align), float(short_result.brooks_align))

    def test_theory_brooks_exhaustion_damps_long_alignment(self) -> None:
        base = make_bars("BTCUSDT", n=320, trend=0.16, seed=69, asset_class="future").reset_index(drop=True)
        stressed = base.copy(deep=True)
        i = len(base) - 1
        anchor = float(base.loc[i - 1, "high"])

        base.loc[i, "open"] = anchor * 1.002
        base.loc[i, "close"] = anchor * 1.009
        base.loc[i, "low"] = anchor * 0.998
        base.loc[i, "high"] = anchor * 1.011
        base.loc[i, "volume"] = float(base["volume"].tail(20).mean()) * 1.2

        stressed.loc[i, "open"] = anchor * 1.004
        stressed.loc[i, "close"] = anchor * 1.030
        stressed.loc[i, "low"] = anchor * 0.993
        stressed.loc[i, "high"] = anchor * 1.046
        stressed.loc[i, "volume"] = float(stressed["volume"].tail(20).mean()) * 3.2

        base_featured = add_common_features(base)
        stressed_featured = add_common_features(stressed)
        base_result = compute_theory_confluence(
            df=base_featured,
            side=Side.LONG,
            regime=RegimeLabel.STRONG_TREND,
            lie_score_ratio=0.58,
            ict_weight=0.8,
            brooks_weight=1.9,
            lie_weight=0.8,
        )
        stressed_result = compute_theory_confluence(
            df=stressed_featured,
            side=Side.LONG,
            regime=RegimeLabel.STRONG_TREND,
            lie_score_ratio=0.58,
            ict_weight=0.8,
            brooks_weight=1.9,
            lie_weight=0.8,
        )
        self.assertLess(float(stressed_result.brooks_align), float(base_result.brooks_align))
        self.assertGreater(float(stressed_result.brooks_oppose), float(base_result.brooks_oppose))

    def test_theory_wyckoff_vpa_spring_biases_long(self) -> None:
        bars = make_bars("BTCUSDT", n=320, trend=0.08, seed=70, asset_class="future").reset_index(drop=True)
        i = len(bars) - 1
        prev_low20 = float(bars.iloc[-21:-1]["low"].min())
        vol_anchor = float(bars["volume"].tail(20).mean())
        bars.loc[i, "open"] = prev_low20 * 0.998
        bars.loc[i, "low"] = prev_low20 * 0.985
        bars.loc[i, "close"] = prev_low20 * 1.010
        bars.loc[i, "high"] = prev_low20 * 1.014
        bars.loc[i, "volume"] = vol_anchor * 2.6

        featured = add_common_features(bars)
        long_result = compute_theory_confluence(
            df=featured,
            side=Side.LONG,
            regime=RegimeLabel.WEAK_TREND,
            lie_score_ratio=0.52,
            ict_weight=0.0,
            brooks_weight=0.0,
            lie_weight=0.0,
            wyckoff_weight=1.8,
            vpa_weight=1.6,
        )
        short_result = compute_theory_confluence(
            df=featured,
            side=Side.SHORT,
            regime=RegimeLabel.WEAK_TREND,
            lie_score_ratio=0.52,
            ict_weight=0.0,
            brooks_weight=0.0,
            lie_weight=0.0,
            wyckoff_weight=1.8,
            vpa_weight=1.6,
        )
        self.assertGreater(float(long_result.wyckoff_align), float(long_result.wyckoff_oppose))
        self.assertGreater(float(long_result.vpa_align), float(long_result.vpa_oppose))
        self.assertGreater(float(long_result.confluence), float(short_result.confluence))

    def test_theory_conflict_detected_on_sweep_reclaim_failure(self) -> None:
        bars = make_bars("BTCUSDT", n=320, trend=0.12, seed=53, asset_class="future").reset_index(drop=True)
        i = len(bars) - 1
        prev10_high = float(bars.iloc[-11:-1]["high"].max())
        near_trend = float(bars.iloc[-20:]["close"].mean())
        close_px = max(near_trend * 1.002, prev10_high * 0.985)
        bars.loc[i, "open"] = close_px * 1.001
        bars.loc[i, "close"] = close_px
        bars.loc[i, "high"] = prev10_high * 1.018
        bars.loc[i, "low"] = close_px * 0.992
        bars.loc[i, "volume"] = float(bars["volume"].tail(20).mean()) * 0.85

        featured = add_common_features(bars)
        result = compute_theory_confluence(
            df=featured,
            side=Side.LONG,
            regime=RegimeLabel.STRONG_TREND,
            lie_score_ratio=0.52,
            ict_weight=1.0,
            brooks_weight=1.0,
            lie_weight=1.0,
        )
        self.assertGreater(float(result.conflict), 0.15)
        self.assertGreater(float(result.conflict), float(result.confluence))
        self.assertIn("theory_confluence_weak", result.flags)


if __name__ == "__main__":
    unittest.main()
