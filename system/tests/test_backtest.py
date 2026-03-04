from __future__ import annotations

import sys
from pathlib import Path
import unittest
from datetime import date

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.backtest.engine import (
    BacktestConfig,
    _compute_metrics,
    _effective_proxy_min_points,
    _next_anti_martingale_scale,
    _proxy_history_min_points,
    _resolve_confirmation_scale,
    run_event_backtest,
)
from lie_engine.models import RegimeLabel, RegimeState, Side, SignalCandidate
from tests.helpers import make_bars, make_multi_symbol_bars


class BacktestTests(unittest.TestCase):
    def test_proxy_history_min_points_scales_with_lookback(self) -> None:
        self.assertEqual(_proxy_history_min_points(180), 130)
        self.assertEqual(_proxy_history_min_points(90), 65)
        self.assertEqual(_proxy_history_min_points(40), 50)

    def test_effective_proxy_min_points_relaxes_for_crypto(self) -> None:
        crypto_hist = pd.DataFrame({"asset_class": ["crypto", "crypto", "perp"]})
        mixed_hist = pd.DataFrame({"asset_class": ["equity", "crypto"]})
        self.assertEqual(_effective_proxy_min_points(90, mixed_hist), 65)
        self.assertEqual(_effective_proxy_min_points(90, crypto_hist), 45)

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

    def test_confirmation_scale_distinguishes_confirm_vs_invalidate(self) -> None:
        confirmed_path = pd.DataFrame(
            [
                {"high": 102.0, "low": 99.7},
                {"high": 103.1, "low": 100.2},
            ]
        )
        fail_path = pd.DataFrame(
            [
                {"high": 100.3, "low": 98.6},
                {"high": 101.2, "low": 98.9},
            ]
        )
        win_scale, confirmed = _resolve_confirmation_scale(
            side=Side.LONG,
            entry=100.0,
            stop=95.0,
            probe_path=confirmed_path,
            lookahead=2,
            loss_mult=0.6,
            win_mult=1.1,
        )
        loss_scale, failed = _resolve_confirmation_scale(
            side=Side.LONG,
            entry=100.0,
            stop=95.0,
            probe_path=fail_path,
            lookahead=2,
            loss_mult=0.6,
            win_mult=1.1,
        )
        self.assertTrue(confirmed)
        self.assertFalse(failed)
        self.assertAlmostEqual(win_scale, 1.1, places=6)
        self.assertAlmostEqual(loss_scale, 0.6, places=6)

    def test_anti_martingale_scale_updates_deterministically(self) -> None:
        up = _next_anti_martingale_scale(
            previous_scale=1.0,
            pnl=0.01,
            step=0.2,
            floor=0.6,
            ceiling=1.4,
        )
        down = _next_anti_martingale_scale(
            previous_scale=up,
            pnl=-0.02,
            step=0.2,
            floor=0.6,
            ceiling=1.4,
        )
        self.assertAlmostEqual(up, 1.2, places=6)
        self.assertAlmostEqual(down, 1.0, places=6)

    def test_backtest_allows_crypto_short_when_shortable(self) -> None:
        import lie_engine.backtest.engine as bt_mod

        bars = pd.concat(
            [
                make_bars("BTCUSDT", n=260, trend=-0.04, seed=211, asset_class="crypto"),
                make_bars("ETHUSDT", n=260, trend=-0.03, seed=212, asset_class="crypto"),
            ],
            ignore_index=True,
        )

        original_scan = bt_mod.scan_signals
        original_regime = bt_mod.derive_regime_consensus

        def _fake_scan_signals(*, bars: pd.DataFrame, regime, cfg):  # noqa: ANN001
            row = bars.sort_values("ts").iloc[-1]
            px = float(row["close"])
            return [
                SignalCandidate(
                    symbol=str(row["symbol"]),
                    side=Side.SHORT,
                    regime=regime,
                    position_score=1.0,
                    structure_score=1.0,
                    momentum_score=1.0,
                    confidence=95.0,
                    convexity_ratio=2.0,
                    entry_price=px,
                    stop_price=px * 1.01,
                    target_price=px * 0.99,
                    can_short=True,
                )
            ]

        def _fake_regime(*, as_of, hurst, hmm_probs, atr_z, trend_thr, mean_thr, atr_extreme):  # noqa: ANN001
            return RegimeState(
                as_of=as_of,
                hurst=float(hurst),
                hmm_probs={"trend": 1.0},
                atr_z=float(atr_z),
                consensus=RegimeLabel.WEAK_TREND,
                protection_mode=False,
                rationale="test_force_trend",
            )

        bt_mod.scan_signals = _fake_scan_signals  # type: ignore[assignment]
        bt_mod.derive_regime_consensus = _fake_regime  # type: ignore[assignment]
        try:
            result = run_event_backtest(
                bars=bars,
                start=date(2025, 1, 1),
                end=date(2025, 9, 17),
                cfg=BacktestConfig(
                    max_daily_trades=1,
                    hold_days=2,
                    signal_confidence_min=0.0,
                    convexity_min=0.0,
                    signal_eval_interval=1,
                    regime_recalc_interval=1,
                    proxy_lookback=60,
                ),
                trend_thr=0.6,
                mean_thr=0.4,
                atr_extreme=2.0,
            )
            self.assertGreater(int(result.trades), 0)
            self.assertIn("crypto", result.by_asset)
        finally:
            bt_mod.scan_signals = original_scan  # type: ignore[assignment]
            bt_mod.derive_regime_consensus = original_regime  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
