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

    def test_unknown_profile_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_provider_stack("unknown_profile")


if __name__ == "__main__":
    unittest.main()
