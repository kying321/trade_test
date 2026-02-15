from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.orchestration.testing import TestingOrchestrator


class TestingOrchestratorTests(unittest.TestCase):
    def test_test_all_writes_log_payload(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        class _Proc:
            returncode = 0
            stdout = "Ran 1 test in 0.001s\n\nOK\n"
            stderr = "test_x ... ok\n"

        with patch("lie_engine.orchestration.testing.subprocess.run", return_value=_Proc()) as mock_run:
            orch = TestingOrchestrator(root=td, output_dir=td)
            out = orch.test_all()

        self.assertEqual(out["returncode"], 0)
        self.assertIn("error=none", out["summary_line"])
        self.assertEqual(out["mode"], "full")
        self.assertIn("stdout_excerpt", out)
        self.assertIn("stderr_excerpt", out)
        self.assertEqual(mock_run.call_count, 1)
        logs = sorted((td / "logs").glob("tests_*.json"))
        self.assertEqual(len(logs), 1)
        payload = json.loads(logs[0].read_text(encoding="utf-8"))
        self.assertIn("stdout", payload)
        self.assertIn("stderr", payload)

    def test_test_all_fast_mode_uses_deterministic_subset(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        class _Proc:
            returncode = 0
            stdout = "Ran 2 tests in 0.002s\n\nOK\n"
            stderr = ""

        ids = [f"tests.test_mod.Case.test_{i}" for i in range(10)]
        with (
            patch("lie_engine.orchestration.testing.subprocess.run", return_value=_Proc()) as mock_run,
            patch.object(TestingOrchestrator, "_discover_test_ids", return_value=ids),
        ):
            orch = TestingOrchestrator(root=td, output_dir=td)
            out = orch.test_all(
                fast=True,
                fast_ratio=0.2,
                fast_shard_index=0,
                fast_shard_total=1,
                fast_seed="seed-x",
            )

        self.assertEqual(out["mode"], "fast")
        self.assertEqual(out["tests_discovered"], 10)
        self.assertEqual(out["tests_selected"], 2)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[:4], [sys.executable, "-m", "unittest", "-v"])
        self.assertEqual(len(cmd) - 4, 2)


if __name__ == "__main__":
    unittest.main()
