from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.providers import OpenSourcePrimaryProvider, OpenSourceSecondaryProvider, PaidProviderPlaceholder


class DataProviderMicrostructureTests(unittest.TestCase):
    def test_opensource_primary_microstructure_not_implemented(self) -> None:
        provider = OpenSourcePrimaryProvider()
        start_ts = datetime(2026, 2, 20, 10, 0, 0)
        end_ts = datetime(2026, 2, 20, 10, 5, 0)

        with self.assertRaises(NotImplementedError):
            provider.fetch_l2("BTCUSDT", start_ts, end_ts, depth=20)
        with self.assertRaises(NotImplementedError):
            provider.fetch_trades("BTCUSDT", start_ts, end_ts, limit=1000)

    def test_opensource_secondary_inherits_microstructure_behavior(self) -> None:
        provider = OpenSourceSecondaryProvider()
        start_ts = datetime(2026, 2, 20, 10, 0, 0)
        end_ts = datetime(2026, 2, 20, 10, 5, 0)

        with self.assertRaises(NotImplementedError):
            provider.fetch_l2("BTCUSDT", start_ts, end_ts, depth=20)
        with self.assertRaises(NotImplementedError):
            provider.fetch_trades("BTCUSDT", start_ts, end_ts, limit=1000)

    def test_paid_placeholder_microstructure_not_implemented(self) -> None:
        provider = PaidProviderPlaceholder()
        start_ts = datetime(2026, 2, 20, 10, 0, 0)
        end_ts = datetime(2026, 2, 20, 10, 5, 0)

        with self.assertRaises(NotImplementedError):
            provider.fetch_l2("BTCUSDT", start_ts, end_ts, depth=20)
        with self.assertRaises(NotImplementedError):
            provider.fetch_trades("BTCUSDT", start_ts, end_ts, limit=1000)


if __name__ == "__main__":
    unittest.main()
