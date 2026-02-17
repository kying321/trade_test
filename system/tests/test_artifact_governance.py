from __future__ import annotations

import json
import sys
import tempfile
from datetime import date
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.orchestration.artifact_governance import apply_dated_artifact_governance
from lie_engine.orchestration.artifact_governance import collect_dated_artifact_pairs


class ArtifactGovernanceTests(unittest.TestCase):
    def test_collect_pairs_and_apply_rotation_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            review_dir = Path(td)
            old_json = review_dir / "2026-02-10_unit.json"
            old_md = review_dir / "2026-02-10_unit.md"
            recent_json = review_dir / "2026-02-16_unit.json"
            future_json = review_dir / "2026-02-19_unit.json"
            future_md = review_dir / "2026-02-19_unit.md"
            invalid_name = review_dir / "latest_unit.json"

            old_json.write_text('{"k": 1}', encoding="utf-8")
            old_md.write_text("old\n", encoding="utf-8")
            recent_json.write_text('{"k": 2}', encoding="utf-8")
            future_json.write_text('{"k": 3}', encoding="utf-8")
            future_md.write_text("future\n", encoding="utf-8")
            invalid_name.write_text('{"ignore": true}', encoding="utf-8")

            pairs = collect_dated_artifact_pairs(
                directory=review_dir,
                json_glob="*_unit.json",
                md_glob="*_unit.md",
            )
            self.assertEqual(set(pairs.keys()), {"2026-02-10", "2026-02-16", "2026-02-19"})

            meta = apply_dated_artifact_governance(
                as_of=date(2026, 2, 17),
                directory=review_dir,
                json_glob="*_unit.json",
                md_glob="*_unit.md",
                retention_days=3,
                checksum_index_enabled=True,
                checksum_index_filename="unit_checksum_index.json",
            )

            self.assertEqual(int(meta.get("retention_days", 0)), 3)
            self.assertEqual(int(meta.get("rotated_out_count", 0)), 1)
            self.assertEqual(meta.get("rotated_out_dates", []), ["2026-02-10"])
            self.assertFalse(bool(meta.get("rotation_failed", True)))
            self.assertTrue(bool(meta.get("checksum_index_enabled", False)))
            self.assertTrue(bool(meta.get("checksum_index_written", False)))
            self.assertFalse(bool(meta.get("checksum_index_failed", True)))

            self.assertFalse(old_json.exists())
            self.assertFalse(old_md.exists())
            self.assertTrue(recent_json.exists())
            self.assertTrue(future_json.exists())
            self.assertTrue(future_md.exists())
            self.assertTrue(invalid_name.exists())

            index_path = Path(str(meta.get("checksum_index_path", "")))
            self.assertTrue(index_path.exists())
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(int(payload.get("retention_days", 0)), 3)
            self.assertEqual(payload.get("rotated_out_dates", []), ["2026-02-10"])
            self.assertEqual(int(payload.get("entry_count", 0)), 2)

            entries = payload.get("entries", [])
            self.assertEqual([row.get("date") for row in entries], ["2026-02-19", "2026-02-16"])
            self.assertTrue(bool(entries[0].get("pair_complete", False)))
            self.assertFalse(bool(entries[1].get("pair_complete", True)))

    def test_apply_governance_supports_disabled_checksum_and_min_retention_floor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            review_dir = Path(td)
            p_json = review_dir / "2026-02-17_disable.json"
            p_md = review_dir / "2026-02-17_disable.md"
            p_json.write_text('{"v": 1}', encoding="utf-8")
            p_md.write_text("ok\n", encoding="utf-8")

            meta = apply_dated_artifact_governance(
                as_of=date(2026, 2, 17),
                directory=review_dir,
                json_glob="*_disable.json",
                md_glob="*_disable.md",
                retention_days=0,
                checksum_index_enabled=False,
                checksum_index_filename="disable_checksum_index.json",
            )

            self.assertEqual(int(meta.get("retention_days", 0)), 1)
            self.assertEqual(int(meta.get("rotated_out_count", 0)), 0)
            self.assertFalse(bool(meta.get("rotation_failed", True)))
            self.assertFalse(bool(meta.get("checksum_index_enabled", True)))
            self.assertFalse(bool(meta.get("checksum_index_written", True)))
            self.assertFalse(bool(meta.get("checksum_index_failed", True)))
            self.assertEqual(str(meta.get("checksum_index_path", "")), "")
            self.assertEqual(int(meta.get("checksum_index_entries", 0)), 0)
            self.assertFalse((review_dir / "disable_checksum_index.json").exists())


if __name__ == "__main__":
    unittest.main()
