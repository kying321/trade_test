from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.signal.cross_source import align_trade_windows, bucket_trade_flow


class CrossSourceAlignmentTests(unittest.TestCase):
    def test_bucket_trade_flow_basic(self) -> None:
        trades = pd.DataFrame(
            [
                {"event_ts_ms": 1000, "qty": 1.0, "price": 100.0, "side": "BUY"},
                {"event_ts_ms": 1099, "qty": 2.0, "price": 101.0, "side": "SELL"},
                {"event_ts_ms": 1200, "qty": 1.5, "price": 102.0, "side": "BUY"},
            ]
        )
        out = bucket_trade_flow(trades, window_ms=100)
        self.assertEqual(int(len(out)), 2)
        self.assertEqual(int(out.iloc[0]["trade_count"]), 2)
        self.assertAlmostEqual(float(out.iloc[0]["net_qty"]), -1.0, places=6)

    def test_align_trade_windows_gap_and_sync(self) -> None:
        left = pd.DataFrame(
            [
                {"event_ts_ms": 1000, "qty": 1.0, "price": 100.0, "side": "BUY"},
                {"event_ts_ms": 1200, "qty": 1.0, "price": 100.0, "side": "BUY"},
                {"event_ts_ms": 1400, "qty": 1.0, "price": 100.0, "side": "BUY"},
                {"event_ts_ms": 1600, "qty": 1.0, "price": 100.0, "side": "BUY"},
            ]
        )
        right = pd.DataFrame(
            [
                {"event_ts_ms": 1010, "qty": 1.0, "price": 100.0, "side": "BUY"},
                {"event_ts_ms": 1610, "qty": 1.0, "price": 100.0, "side": "BUY"},
            ]
        )
        out = align_trade_windows(
            left=left,
            right=right,
            window_ms=200,
            tolerance_ms=20,
            continuous_gap_ms=100,
        )
        self.assertGreater(int(out.summary["bucket_total"]), 0)
        self.assertFalse(bool(out.summary["gap_ok"]))
        self.assertTrue(bool(out.summary["sync_ok"]))
        self.assertGreater(int(out.summary["missing_right_buckets"]), 0)


if __name__ == "__main__":
    unittest.main()
