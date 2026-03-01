from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.models import RegimeLabel, Side
from lie_engine.signal.engine import SignalEngineConfig, generate_signal_for_symbol
from lie_engine.signal.microstructure import summarize_microstructure_snapshot
from tests.helpers import make_bars


class MicrostructureSignalTests(unittest.TestCase):
    def test_summarize_microstructure_snapshot(self) -> None:
        l2 = pd.DataFrame(
            [
                {
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "event_ts_ms": 1000,
                    "recv_ts_ms": 1002,
                    "seq": 10,
                    "prev_seq": 9,
                    "bids": [{"price": 100.0, "qty": 5.0}, {"price": 99.9, "qty": 4.0}, {"price": 99.8, "qty": 3.0}],
                    "asks": [{"price": 100.1, "qty": 2.0}, {"price": 100.2, "qty": 2.0}, {"price": 100.3, "qty": 2.0}],
                    "source": "binance_spot_public",
                }
            ]
        )
        trades = pd.DataFrame(
            [
                {
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "trade_id": 1,
                    "event_ts_ms": 900,
                    "recv_ts_ms": 901,
                    "price": 100.0,
                    "qty": 2.0,
                    "side": "BUY",
                    "source": "binance_spot_public",
                },
                {
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "trade_id": 2,
                    "event_ts_ms": 1200,
                    "recv_ts_ms": 1201,
                    "price": 99.9,
                    "qty": 1.0,
                    "side": "SELL",
                    "source": "binance_spot_public",
                },
                {
                    "exchange": "binance",
                    "symbol": "BTCUSDT",
                    "trade_id": 3,
                    "event_ts_ms": 1400,
                    "recv_ts_ms": 1401,
                    "price": 100.1,
                    "qty": 3.0,
                    "side": "BUY",
                    "source": "binance_spot_public",
                },
            ]
        )
        out = summarize_microstructure_snapshot(
            l2=l2,
            trades=trades,
            cross_source_tolerance_ms=20,
            continuous_gap_ms=1000,
            min_trade_count=2,
        )
        self.assertTrue(bool(out.get("has_data", False)))
        self.assertGreater(float(out.get("queue_imbalance", 0.0)), 0.0)
        self.assertGreater(float(out.get("ofi_norm", 0.0)), 0.0)
        self.assertTrue(bool(out.get("sync_ok", False)))
        self.assertTrue(bool(out.get("gap_ok", False)))

    def test_summarize_microstructure_snapshot_schema_guard(self) -> None:
        l2 = pd.DataFrame([{"event_ts_ms": 1000, "recv_ts_ms": 1001}])
        trades = pd.DataFrame([{"event_ts_ms": 1000, "recv_ts_ms": 1001, "qty": 1.0}])
        out = summarize_microstructure_snapshot(l2=l2, trades=trades)
        self.assertFalse(bool(out.get("schema_ok", True)))
        self.assertFalse(bool(out.get("has_data", True)))
        self.assertTrue(bool(out.get("l2_missing_fields", [])))
        self.assertTrue(bool(out.get("trade_missing_fields", [])))

    def test_microstructure_adjusts_signal_confidence_directionally(self) -> None:
        bars = make_bars("BTCUSDT", n=280, trend=0.12, seed=121)
        cfg = SignalEngineConfig(
            confidence_min=0.0,
            convexity_min=0.0,
            factor_filter_enabled=False,
            microstructure_enabled=True,
            micro_confidence_boost_max=10.0,
            micro_penalty_max=10.0,
            micro_min_trade_count=10,
        )
        base = generate_signal_for_symbol(bars, regime=RegimeLabel.STRONG_TREND, cfg=cfg)
        self.assertIsNotNone(base)
        assert base is not None

        aligned = 1.0 if base.side == Side.LONG else -1.0
        plus = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            micro_factor_state={
                "has_data": True,
                "micro_alignment": aligned,
                "queue_imbalance": aligned,
                "ofi_norm": aligned,
                "evidence_score": 1.0,
                "trade_count": 300,
                "sync_ok": True,
                "gap_ok": True,
            },
        )
        minus = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            micro_factor_state={
                "has_data": True,
                "micro_alignment": -aligned,
                "queue_imbalance": -aligned,
                "ofi_norm": -aligned,
                "evidence_score": 1.0,
                "trade_count": 300,
                "sync_ok": True,
                "gap_ok": True,
            },
        )

        self.assertIsNotNone(plus)
        self.assertIsNotNone(minus)
        assert plus is not None
        assert minus is not None
        self.assertGreaterEqual(float(plus.confidence), float(base.confidence))
        self.assertLessEqual(float(minus.confidence), float(base.confidence))

    def test_microstructure_schema_risk_penalizes_without_trade_data(self) -> None:
        bars = make_bars("BTCUSDT", n=280, trend=0.12, seed=122)
        cfg = SignalEngineConfig(
            confidence_min=0.0,
            convexity_min=0.0,
            factor_filter_enabled=False,
            microstructure_enabled=True,
            micro_confidence_boost_max=10.0,
            micro_penalty_max=10.0,
        )
        base = generate_signal_for_symbol(bars, regime=RegimeLabel.STRONG_TREND, cfg=cfg)
        risky = generate_signal_for_symbol(
            bars,
            regime=RegimeLabel.STRONG_TREND,
            cfg=cfg,
            micro_factor_state={
                "has_data": False,
                "schema_ok": False,
                "schema_issues": ["trades_missing:event_ts_ms,qty"],
            },
        )
        self.assertIsNotNone(base)
        self.assertIsNotNone(risky)
        assert base is not None
        assert risky is not None
        self.assertLessEqual(float(risky.confidence), float(base.confidence))
        self.assertIn("micro_schema_risk", list(risky.factor_flags))


if __name__ == "__main__":
    unittest.main()
