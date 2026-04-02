from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


class RefreshCodexUiStateScriptTests(unittest.TestCase):
    def _write_executable(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _seed_codex_home(self, root: Path) -> Path:
        codex_home = root / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        (codex_home / "config.toml").write_text(
            'service_tier = "fast"\n[model_providers.cpa]\nrequires_openai_auth = true\n',
            encoding="utf-8",
        )
        (codex_home / "config1.toml").write_text(
            'service_tier = "fast"\n',
            encoding="utf-8",
        )
        (codex_home / "state_5.sqlite").write_text("placeholder", encoding="utf-8")
        payload = {
            "electron-saved-workspace-roots": ["/tmp/demo"],
            "active-workspace-roots": ["/tmp/demo"],
            "electron-persisted-atom-state": {
                "codexCloudAccess": "disabled",
                "environment": None,
                "agent-mode": "custom",
                "preferred-non-full-access-agent-mode": None,
                "has-dismissed-skills-apps-tooltip": True,
            },
        }
        (codex_home / ".codex-global-state.json").write_text(json.dumps(payload), encoding="utf-8")
        return codex_home

    def _run_script(
        self,
        codex_home: Path,
        *args: str,
        path_prefix: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "refresh_codex_ui_state.sh"
        env = dict(os.environ)
        env["CODEX_HOME"] = str(codex_home)
        env["HOME"] = str(codex_home.parent)
        if path_prefix is not None:
            env["PATH"] = f"{path_prefix}{os.pathsep}{env.get('PATH', '')}"
        return subprocess.run(
            ["bash", str(script), *args],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_status_reports_fast_tier_and_cloud_access(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            codex_home = self._seed_codex_home(Path(td))

            proc = self._run_script(codex_home, "status")

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("CONFIG_SERVICE_TIER=fast", proc.stdout)
            self.assertIn("CONFIG1_SERVICE_TIER=fast", proc.stdout)
            self.assertIn("CODEX_CLOUD_ACCESS=disabled", proc.stdout)
            self.assertIn("READY_TO_REFRESH=1", proc.stdout)

    def test_refresh_backs_up_state_and_clears_only_target_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            codex_home = self._seed_codex_home(Path(td))

            proc = self._run_script(codex_home, "refresh")

            self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("Refresh complete.", proc.stdout)
            backup_line = next(
                line for line in proc.stdout.splitlines() if line.startswith("BACKUP_DIR=")
            )
            backup_dir = Path(backup_line.split("=", 1)[1])
            self.assertTrue((backup_dir / ".codex-global-state.json.before").exists())

            refreshed = json.loads((codex_home / ".codex-global-state.json").read_text(encoding="utf-8"))
            state = refreshed["electron-persisted-atom-state"]
            self.assertNotIn("codexCloudAccess", state)
            self.assertNotIn("environment", state)
            self.assertEqual(state["agent-mode"], "custom")
            self.assertEqual(refreshed["active-workspace-roots"], ["/tmp/demo"])

    def test_refresh_aborts_when_state_sqlite_is_in_use(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            codex_home = self._seed_codex_home(root)
            fakebin = root / "fakebin"
            self._write_executable(
                fakebin / "lsof",
                "#!/usr/bin/env bash\n"
                "if [[ \"$*\" == *\"state_5.sqlite\"* ]]; then\n"
                "  echo 'codex 999 user 10u REG 1,25 0 0 '$1\n"
                "  exit 0\n"
                "fi\n"
                "exit 1\n",
            )

            proc = self._run_script(codex_home, "refresh", path_prefix=fakebin)

            self.assertEqual(proc.returncode, 2, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("state_5.sqlite is currently in use", proc.stderr)
            self.assertIn("close Codex / Codex App / app-server", proc.stderr)

    def test_restore_recovers_backed_up_global_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            codex_home = self._seed_codex_home(Path(td))

            refresh_proc = self._run_script(codex_home, "refresh")
            self.assertEqual(
                refresh_proc.returncode, 0, msg=f"stdout={refresh_proc.stdout}\nstderr={refresh_proc.stderr}"
            )
            backup_line = next(
                line for line in refresh_proc.stdout.splitlines() if line.startswith("BACKUP_DIR=")
            )
            backup_dir = backup_line.split("=", 1)[1]

            restore_proc = self._run_script(codex_home, "restore", backup_dir)

            self.assertEqual(
                restore_proc.returncode, 0, msg=f"stdout={restore_proc.stdout}\nstderr={restore_proc.stderr}"
            )
            restored = json.loads((codex_home / ".codex-global-state.json").read_text(encoding="utf-8"))
            state = restored["electron-persisted-atom-state"]
            self.assertEqual(state["codexCloudAccess"], "disabled")
            self.assertIn("Restored from:", restore_proc.stdout)


if __name__ == "__main__":
    unittest.main()
