from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class PublishLocalPiRuntimeScriptsScriptTests(unittest.TestCase):
    def _run(
        self,
        source_root: Path,
        manifest_path: Path,
        target_root: Path,
        output_root: Path,
        *extra: str,
    ) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "publish_local_pi_runtime_scripts.py"
        cmd = [
            "python3",
            str(script),
            "--source-root",
            str(source_root),
            "--manifest-path",
            str(manifest_path),
            "--target-root",
            str(target_root),
            "--output-root",
            str(output_root),
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
        (root / "pi_cycle_halfhour_launchd_runner.sh").write_text("#!/usr/bin/env bash\necho runtime\n", encoding="utf-8")
        (root / "lie_root_resolver.py").write_text("print('resolver')\n", encoding="utf-8")
        (root / "extra_unmanaged.py").write_text("print('unmanaged')\n", encoding="utf-8")
        pycache_dir = root / "__pycache__"
        pycache_dir.mkdir(parents=True, exist_ok=True)
        (pycache_dir / "ignored.cpython-313.pyc").write_bytes(b"bytecode")

    def _write_manifest(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "files": [
                        "pi_cycle_halfhour_launchd_runner.sh",
                        "lie_root_resolver.py",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_publish_updates_managed_files_without_touching_unmanaged(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            source_root = td_path / "source"
            manifest_path = td_path / "runtime_manifest.json"
            target_root = td_path / "target"
            output_root = td_path / "output"
            self._make_source_root(source_root)
            self._write_manifest(manifest_path)
            target_root.mkdir(parents=True, exist_ok=True)
            (target_root / "lie_root_resolver.py").write_text("print('old')\n", encoding="utf-8")
            (target_root / "extra_tool.py").write_text("print('extra')\n", encoding="utf-8")

            proc = self._run(source_root, manifest_path, target_root, output_root)

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["changed"])
            self.assertTrue(payload["lock_acquired"])
            self.assertEqual(payload["manifest_file_count"], 2)
            self.assertIn("lie_root_resolver.py", payload["updated_files"])
            self.assertIn("pi_cycle_halfhour_launchd_runner.sh", payload["created_files"])
            self.assertNotIn("__pycache__/ignored.cpython-313.pyc", payload["updated_files"])
            self.assertFalse((target_root / "__pycache__" / "ignored.cpython-313.pyc").exists())
            self.assertFalse((target_root / "extra_unmanaged.py").exists())
            self.assertEqual((target_root / "extra_tool.py").read_text(encoding="utf-8"), "print('extra')\n")
            backup_dir = Path(payload["backup_dir"])
            self.assertTrue((backup_dir / "lie_root_resolver.py").exists())

    def test_publish_is_idempotent_when_runtime_matches_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            source_root = td_path / "source"
            manifest_path = td_path / "runtime_manifest.json"
            target_root = td_path / "target"
            output_root = td_path / "output"
            self._make_source_root(source_root)
            self._write_manifest(manifest_path)
            target_root.mkdir(parents=True, exist_ok=True)
            for src in source_root.iterdir():
                if not src.is_file():
                    continue
                if src.name == "extra_unmanaged.py":
                    continue
                (target_root / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

            proc = self._run(source_root, manifest_path, target_root, output_root, "--no-backup")

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["changed"])
            self.assertTrue(payload["lock_acquired"])
            self.assertEqual(payload["manifest_file_count"], 2)
            self.assertEqual(payload["stats"]["updated"], 0)
            self.assertEqual(payload["stats"]["created"], 0)


if __name__ == "__main__":
    unittest.main()
