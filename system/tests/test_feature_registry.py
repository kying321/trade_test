from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from lie_engine.feature import FeatureRegistry, FeatureSpec


class FeatureRegistryTests(unittest.TestCase):
    def _tmp_registry_path(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name) / "feature_registry.yaml"

    def test_registry_roundtrip(self) -> None:
        path = self._tmp_registry_path()
        reg = FeatureRegistry(path)
        reg.upsert(
            FeatureSpec(
                name="alpha_momentum_20d",
                entity_keys=["symbol", "date"],
                source_layer="feature",
                ttl_minutes=1440,
                owner="research",
                description="20d momentum score",
                tags={"domain": "momentum"},
            )
        )
        reg.save()

        loaded = FeatureRegistry(path).load()
        got = loaded.get("alpha_momentum_20d")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.owner, "research")
        self.assertEqual(got.ttl_minutes, 1440)
        self.assertEqual(got.tags.get("domain"), "momentum")

    def test_invalid_name_rejected(self) -> None:
        path = self._tmp_registry_path()
        reg = FeatureRegistry(path)
        with self.assertRaises(ValueError):
            reg.upsert(
                FeatureSpec(
                    name="9bad-name",
                    entity_keys=["symbol"],
                    source_layer="feature",
                    ttl_minutes=60,
                    owner="ops",
                )
            )

    def test_duplicate_in_file_rejected(self) -> None:
        path = self._tmp_registry_path()
        path.write_text(
            (
                "version: v1\n"
                "features:\n"
                "  - name: f_a\n"
                "    entity_keys: [symbol]\n"
                "    source_layer: feature\n"
                "    ttl_minutes: 30\n"
                "    owner: qa\n"
                "  - name: f_a\n"
                "    entity_keys: [symbol]\n"
                "    source_layer: feature\n"
                "    ttl_minutes: 30\n"
                "    owner: qa\n"
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            FeatureRegistry(path).load()


if __name__ == "__main__":
    unittest.main()

