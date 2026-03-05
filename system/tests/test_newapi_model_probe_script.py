from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class NewApiModelProbeScriptTests(unittest.TestCase):
    def _run_probe(self, config_path: Path, *, disable_isolation_write: bool) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "newapi_model_probe.sh"
        cmd = [
            "bash",
            str(script),
            "--base-url",
            "http://127.0.0.1:9",
            "--api-key-env",
            "DUMMY_NEWAPI_KEY",
            "--models",
            "gpt-5.3-codex",
            "--required-models",
            "gpt-5.3-codex",
            "--optional-models",
            "",
            "--samples",
            "1",
            "--retry-transient",
            "0",
            "--isolation-config-path",
            str(config_path),
            "--output-dir",
            "system/output/review",
        ]
        if disable_isolation_write:
            cmd.append("--disable-isolation-write")

        env = dict(os.environ)
        env["DUMMY_NEWAPI_KEY"] = "dummy-key-for-script-test"
        return subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_probe_disable_isolation_write_does_not_mutate_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "probe_config.yaml"
            cfg.write_text("validation:\n  binance_live_takeover_enabled: true\n", encoding="utf-8")

            proc = self._run_probe(cfg, disable_isolation_write=True)

            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("Isolation config write disabled", proc.stderr)
            text = cfg.read_text(encoding="utf-8")
            self.assertIn("binance_live_takeover_enabled: true", text)
            self.assertFalse((cfg.parent / f"{cfg.name}.bak").exists())

    def test_probe_default_isolation_write_mutates_target_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Path(td) / "probe_config.yaml"
            cfg.write_text("validation:\n  binance_live_takeover_enabled: true\n", encoding="utf-8")

            proc = self._run_probe(cfg, disable_isolation_write=False)

            self.assertEqual(proc.returncode, 1, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")
            self.assertIn("Live Takeover isolated via", proc.stderr)
            text = cfg.read_text(encoding="utf-8")
            self.assertIn("binance_live_takeover_enabled: false", text)
            self.assertTrue((cfg.parent / f"{cfg.name}.bak").exists())


if __name__ == "__main__":
    unittest.main()
