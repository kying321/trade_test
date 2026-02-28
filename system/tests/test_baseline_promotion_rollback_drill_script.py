from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class BaselineRollbackDrillScriptTests(unittest.TestCase):
    def _script_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "scripts" / "baseline_promotion_rollback_drill.sh"

    def test_drill_passes_and_restores_active_file(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        active = td / "active.yaml"
        anchor = td / "anchor.yaml"
        promotion = td / "2026-02-27_baseline_promotion.json"
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        active.write_text("profiles:\n  foo: 1\n", encoding="utf-8")
        anchor.write_text("profiles:\n  foo: 0\n", encoding="utf-8")
        promotion.write_text(
            json.dumps(
                {
                    "active_path": str(active),
                    "rollback_anchor": str(anchor),
                    "as_of": "2026-02-27",
                    "round": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        before = active.read_text(encoding="utf-8")

        cmd = [
            "bash",
            str(self._script_path()),
            "--promotion-file",
            str(promotion),
            "--output-dir",
            str(review_dir),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
        self.assertIn("PASSED baseline rollback drill", proc.stdout)
        self.assertEqual(active.read_text(encoding="utf-8"), before)

        artifact_json = review_dir / f"{date.today().isoformat()}_baseline_rollback_drill.json"
        self.assertTrue(artifact_json.exists())
        payload = json.loads(artifact_json.read_text(encoding="utf-8"))
        self.assertEqual(str(payload.get("status", "")), "passed")
        self.assertTrue(bool((payload.get("validation", {}) or {}).get("restore_match_original", False)))

    def test_drill_fails_when_rollback_anchor_missing(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        active = td / "active.yaml"
        promotion = td / "2026-02-27_baseline_promotion.json"
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        active.write_text("profiles:\n  foo: 1\n", encoding="utf-8")
        promotion.write_text(
            json.dumps(
                {
                    "active_path": str(active),
                    "rollback_anchor": str(td / "missing.yaml"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        cmd = [
            "bash",
            str(self._script_path()),
            "--promotion-file",
            str(promotion),
            "--output-dir",
            str(review_dir),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("rollback anchor missing", proc.stderr)


if __name__ == "__main__":
    unittest.main()
