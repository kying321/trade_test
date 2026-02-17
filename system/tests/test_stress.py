from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.backtest.engine import BacktestConfig, run_event_backtest
from tests.helpers import make_bars


def _inject_shock(df: pd.DataFrame, anchor: int, shock_pct: float) -> pd.DataFrame:
    out = df.copy().sort_values("ts").reset_index(drop=True)
    for i in range(anchor, min(anchor + 5, len(out))):
        out.loc[i, "close"] = max(0.1, out.loc[i - 1, "close"] * (1.0 + shock_pct))
        out.loc[i, "open"] = out.loc[i, "close"] * (1.0 + np.sign(shock_pct) * 0.02)
        hi = max(out.loc[i, "open"], out.loc[i, "close"]) * 1.01
        lo = min(out.loc[i, "open"], out.loc[i, "close"]) * 0.99
        out.loc[i, "high"] = hi
        out.loc[i, "low"] = lo
        out.loc[i, "volume"] = out.loc[i, "volume"] * 1.8
    return out


class StressScenarioTests(unittest.TestCase):
    def _multi_symbol(self, start: str, n: int = 900) -> pd.DataFrame:
        frames = [
            make_bars("300750", n=n, start=start, trend=0.02, seed=11, asset_class="equity"),
            make_bars("002050", n=n, start=start, trend=0.01, seed=12, asset_class="equity"),
            make_bars("513130", n=n, start=start, trend=0.005, seed=13, asset_class="etf"),
            make_bars("LC2603", n=n, start=start, trend=-0.005, seed=14, asset_class="future"),
        ]
        return pd.concat(frames, ignore_index=True)

    def test_2015_crash_window_robust(self) -> None:
        bars = self._multi_symbol(start="2014-01-01", n=700)
        bars = _inject_shock(bars, anchor=330, shock_pct=-0.10)
        result = run_event_backtest(
            bars=bars,
            start=date(2015, 1, 1),
            end=date(2015, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=5, signal_confidence_min=35, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        self.assertTrue(np.isfinite(result.max_drawdown))
        self.assertGreaterEqual(result.max_drawdown, 0.0)

    def test_2020_pandemic_and_2022_geopolitical_windows_robust(self) -> None:
        bars = self._multi_symbol(start="2018-01-01", n=1400)
        bars = _inject_shock(bars, anchor=520, shock_pct=-0.08)  # 2020 shock
        bars = _inject_shock(bars, anchor=1030, shock_pct=-0.06)  # 2022 shock

        r2020 = run_event_backtest(
            bars=bars,
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=5, signal_confidence_min=35, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        r2022 = run_event_backtest(
            bars=bars,
            start=date(2022, 1, 1),
            end=date(2022, 12, 31),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=5, signal_confidence_min=35, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        self.assertTrue(np.isfinite(r2020.annual_return))
        self.assertTrue(np.isfinite(r2022.annual_return))
        self.assertGreaterEqual(r2020.max_drawdown, 0.0)
        self.assertGreaterEqual(r2022.max_drawdown, 0.0)

    def test_extreme_gap_and_limit_sequence(self) -> None:
        bars = self._multi_symbol(start="2024-01-01", n=420)
        # Create gap-down + synthetic limit move cluster on one symbol.
        sym = bars["symbol"] == "300750"
        idx = bars[sym].sort_values("ts").index.to_list()
        for j in range(120, min(126, len(idx))):
            i = idx[j]
            prev_close = float(bars.loc[idx[j - 1], "close"])
            bars.loc[i, "open"] = prev_close * 0.90
            bars.loc[i, "close"] = prev_close * 0.90
            bars.loc[i, "high"] = prev_close * 0.91
            bars.loc[i, "low"] = prev_close * 0.89
            bars.loc[i, "volume"] = float(bars.loc[i, "volume"]) * 2.0

        result = run_event_backtest(
            bars=bars,
            start=date(2024, 1, 1),
            end=date(2025, 6, 30),
            cfg=BacktestConfig(max_daily_trades=2, hold_days=5, signal_confidence_min=35, convexity_min=1.2),
            trend_thr=0.6,
            mean_thr=0.4,
            atr_extreme=2.0,
        )
        self.assertTrue(np.isfinite(result.total_return))
        self.assertTrue(np.isfinite(result.positive_window_ratio))


if __name__ == "__main__":
    unittest.main()

