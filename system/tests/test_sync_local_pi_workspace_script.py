from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SyncLocalPiWorkspaceScriptTests(unittest.TestCase):
    def _run(self, source_root: Path, target_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "sync_local_pi_workspace.py"
        cmd = [
            "python3",
            str(script),
            "--source-root",
            str(source_root),
            "--target-root",
            str(target_root),
            *extra,
        ]
        return subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def _make_source_root(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.yaml").write_text("name: fenlie\n", encoding="utf-8")
        (root / "src" / "lie_engine").mkdir(parents=True, exist_ok=True)
        (root / "src" / "lie_engine" / "__init__.py").write_text("", encoding="utf-8")
        (root / "scripts").mkdir(parents=True, exist_ok=True)
        (root / "scripts" / "tool.py").write_text("print('new')\n", encoding="utf-8")
        (root / "scripts" / "fresh.py").write_text("print('fresh')\n", encoding="utf-8")
        (root / "output").mkdir(parents=True, exist_ok=True)
        (root / "output" / "skip.txt").write_text("skip\n", encoding="utf-8")

    def test_sync_updates_files_without_touching_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            source_root = td_path / "source"
            target_root = td_path / "target"
            self._make_source_root(source_root)

            (target_root / "scripts").mkdir(parents=True, exist_ok=True)
            (target_root / "scripts" / "tool.py").write_text("print('old')\n", encoding="utf-8")
            (target_root / "scripts" / "extra.py").write_text("print('extra')\n", encoding="utf-8")
            (target_root / "output").mkdir(parents=True, exist_ok=True)
            (target_root / "output" / "keep.txt").write_text("keep\n", encoding="utf-8")

            proc = self._run(source_root, target_root)

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["changed"])
            self.assertTrue(payload["lock_acquired"])
            self.assertIn("scripts/tool.py", payload["updated_files"])
            self.assertIn("scripts/fresh.py", payload["created_files"])
            self.assertEqual((target_root / "scripts" / "tool.py").read_text(encoding="utf-8"), "print('new')\n")
            self.assertEqual((target_root / "scripts" / "extra.py").read_text(encoding="utf-8"), "print('extra')\n")
            self.assertEqual((target_root / "output" / "keep.txt").read_text(encoding="utf-8"), "keep\n")
            backup_dir = Path(payload["backup_dir"])
            self.assertTrue((backup_dir / "scripts" / "tool.py").exists())
            self.assertFalse((target_root / "output" / "skip.txt").exists())

    def test_sync_is_idempotent_when_target_matches_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            source_root = td_path / "source"
            target_root = td_path / "target"
            self._make_source_root(source_root)
            subprocess.run(
                ["cp", "-R", f"{source_root}/.", str(target_root)],
                check=True,
                capture_output=True,
                text=True,
            )

            proc = self._run(source_root, target_root, "--no-backup")

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["changed"])
            self.assertTrue(payload["lock_acquired"])
            self.assertIsNone(payload["backup_dir"])
            self.assertEqual(payload["stats"]["updated"], 0)
            self.assertEqual(payload["stats"]["created"], 0)

    def test_sync_prunes_old_backups(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            source_root = td_path / "source"
            target_root = td_path / "target"
            self._make_source_root(source_root)
            (target_root / "scripts").mkdir(parents=True, exist_ok=True)
            (target_root / "scripts" / "tool.py").write_text("print('old')\n", encoding="utf-8")

            backups_root = target_root / "output" / "backups"
            old_a = backups_root / "workspace_sync_20000101T000000Z"
            old_b = backups_root / "workspace_sync_20000102T000000Z"
            old_a.mkdir(parents=True, exist_ok=True)
            old_b.mkdir(parents=True, exist_ok=True)
            stale_ts = 946684800
            os.utime(old_a, (stale_ts, stale_ts))
            os.utime(old_b, (stale_ts + 60, stale_ts + 60))

            proc = self._run(
                source_root,
                target_root,
                "--backup-keep",
                "1",
                "--backup-max-age-hours",
                "1",
            )

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["pruned_backups_age"])
            self.assertFalse(old_a.exists())
            self.assertFalse(old_b.exists())


if __name__ == "__main__":
    unittest.main()
