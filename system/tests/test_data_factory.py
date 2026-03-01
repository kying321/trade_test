from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.data.factory import build_provider_stack


class DataFactoryTests(unittest.TestCase):
    def test_build_opensource_dual(self) -> None:
        providers = build_provider_stack("opensource_dual")
        self.assertEqual(len(providers), 2)
        self.assertEqual(getattr(providers[0], "name", ""), "open_source_primary")
        self.assertEqual(getattr(providers[1], "name", ""), "open_source_secondary")

    def test_build_opensource_primary(self) -> None:
        providers = build_provider_stack("opensource_primary")
        self.assertEqual(len(providers), 1)
        self.assertEqual(getattr(providers[0], "name", ""), "open_source_primary")

    def test_build_binance_spot_public(self) -> None:
        providers = build_provider_stack("binance_spot_public")
        self.assertEqual(len(providers), 1)
        self.assertEqual(getattr(providers[0], "name", ""), "binance_spot_public")

    def test_build_bybit_spot_public(self) -> None:
        providers = build_provider_stack("bybit_spot_public")
        self.assertEqual(len(providers), 1)
        self.assertEqual(getattr(providers[0], "name", ""), "bybit_spot_public")

    def test_build_dual_binance_bybit_public(self) -> None:
        providers = build_provider_stack("dual_binance_bybit_public")
        self.assertEqual(len(providers), 2)
        self.assertEqual(getattr(providers[0], "name", ""), "binance_spot_public")
        self.assertEqual(getattr(providers[1], "name", ""), "bybit_spot_public")

    def test_build_hybrid_opensource_binance(self) -> None:
        providers = build_provider_stack("hybrid_opensource_binance")
        self.assertEqual(len(providers), 2)
        self.assertEqual(getattr(providers[0], "name", ""), "open_source_primary")
        self.assertEqual(getattr(providers[1], "name", ""), "binance_spot_public")

    def test_build_hybrid_opensource_binance_bybit(self) -> None:
        providers = build_provider_stack("hybrid_opensource_binance_bybit")
        self.assertEqual(len(providers), 3)
        self.assertEqual(getattr(providers[0], "name", ""), "open_source_primary")
        self.assertEqual(getattr(providers[1], "name", ""), "binance_spot_public")
        self.assertEqual(getattr(providers[2], "name", ""), "bybit_spot_public")

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_provider_stack("unknown_profile")


if __name__ == "__main__":
    unittest.main()
