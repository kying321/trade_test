from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from datetime import date
import json

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
from lie_engine.backtest.walk_forward import run_walk_forward_backtest
from lie_engine.models import BacktestResult, RegimeLabel, RegimeState, Side, SignalCandidate
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

    def test_run_event_backtest_emits_timing_callback(self) -> None:
        bars = make_multi_symbol_bars()
        seen: list[dict[str, object]] = []

        result = run_event_backtest(
            bars=bars,
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=3, signal_confidence_min=30, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
            timing_callback=lambda payload: seen.append(dict(payload)),
        )
        self.assertTrue(seen)
        payload = seen[-1]
        self.assertEqual(str(payload.get("status", "")), "completed")
        self.assertIn("history_slice_elapsed_sec", payload)
        self.assertIn("regime_recalc_elapsed_sec", payload)
        self.assertIn("market_proxy_elapsed_sec", payload)
        self.assertIn("hurst_elapsed_sec", payload)
        self.assertIn("hmm_elapsed_sec", payload)
        self.assertIn("atr_elapsed_sec", payload)
        self.assertIn("derive_regime_elapsed_sec", payload)
        self.assertIn("day_universe_elapsed_sec", payload)
        self.assertIn("signal_scan_elapsed_sec", payload)
        self.assertIn("trade_path_elapsed_sec", payload)
        self.assertIn("metrics_aggregate_elapsed_sec", payload)
        self.assertEqual(int(payload.get("trades", 0)), int(result.trades))

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
            self.assertTrue(bool(result.by_symbol))
            stats = next(iter(result.by_symbol.values()))
            self.assertGreaterEqual(int(stats["trade_count"]), 1)
            self.assertTrue(bool(result.by_symbol_regime))
            regime_stats = next(iter(result.by_symbol_regime.values()))
            self.assertTrue(bool(regime_stats))
        finally:
            bt_mod.scan_signals = original_scan  # type: ignore[assignment]
            bt_mod.derive_regime_consensus = original_regime  # type: ignore[assignment]

    def test_walk_forward_backtest_writes_timing(self) -> None:
        import lie_engine.backtest.walk_forward as wf_mod

        timing_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(timing_dir, ignore_errors=True))
        timing_path = timing_dir / "walk_forward_timing.json"
        original = wf_mod.run_event_backtest

        def _fake_run_event_backtest(*, bars, start, end, cfg, trend_thr, mean_thr, atr_extreme, timing_callback=None):  # noqa: ANN001
            if callable(timing_callback):
                timing_callback(
                    {
                        "status": "completed",
                        "elapsed_sec": 1.25,
                        "current_stage": "",
                        "signal_scan_count": 3,
                        "trade_path_elapsed_sec": 0.42,
                    }
                )
            return BacktestResult(
                start=start,
                end=end,
                total_return=0.1,
                annual_return=0.2,
                max_drawdown=0.05,
                win_rate=0.6,
                profit_factor=1.2,
                expectancy=0.01,
                trades=3,
                violations=0,
                positive_window_ratio=0.8,
                equity_curve=[],
                by_asset={},
            )

        wf_mod.run_event_backtest = _fake_run_event_backtest  # type: ignore[assignment]
        try:
            result = run_walk_forward_backtest(
                bars=pd.DataFrame(),
                start=date(2020, 1, 1),
                end=date(2025, 1, 1),
                trend_thr=0.6,
                mean_thr=0.4,
                atr_extreme=2.0,
                cfg_template=BacktestConfig(),
                train_years=3,
                valid_years=1,
                step_months=12,
                timing_path=timing_path,
            )
        finally:
            wf_mod.run_event_backtest = original  # type: ignore[assignment]

        self.assertTrue(timing_path.exists())
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        self.assertEqual(str(timing.get("status", "")), "completed")
        self.assertEqual(str(timing.get("current_stage", "")), "")
        self.assertGreaterEqual(len(timing.get("windows", [])), 2)
        self.assertIn("event_backtest_timing", timing.get("windows", [])[0])
        self.assertEqual(
            int(timing.get("windows", [])[0].get("event_backtest_timing", {}).get("signal_scan_count", 0)),
            3,
        )
        summary = timing.get("summary", {})
        self.assertEqual(int(summary.get("trades", 0)), int(result.trades))
        self.assertEqual(int(summary.get("window_count", 0)), len(timing.get("windows", [])))


if __name__ == "__main__":
    unittest.main()
