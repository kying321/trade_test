from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class ExecWithAuditScriptTests(unittest.TestCase):
    def _script_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "scripts" / "exec_with_audit.py"

    def test_records_success_and_failure_events(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        success_cmd = [
            sys.executable,
            str(self._script_path()),
            "--root",
            str(td),
            "--source",
            "unit",
            "--tag",
            "ok-case",
            "--",
            "TEST_AUDIT_FLAG=1",
            sys.executable,
            "-c",
            "import os; print('ok-' + os.environ.get('TEST_AUDIT_FLAG', '0'))",
        ]
        proc_ok = subprocess.run(success_cmd, text=True, capture_output=True)
        self.assertEqual(proc_ok.returncode, 0, msg=f"stdout={proc_ok.stdout}\nstderr={proc_ok.stderr}")
        self.assertIn("ok-1", proc_ok.stdout)

        fail_cmd = [
            sys.executable,
            str(self._script_path()),
            "--root",
            str(td),
            "--source",
            "unit",
            "--tag",
            "fail-case",
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(3)",
        ]
        proc_fail = subprocess.run(fail_cmd, text=True, capture_output=True)
        self.assertEqual(proc_fail.returncode, 3, msg=f"stdout={proc_fail.stdout}\nstderr={proc_fail.stderr}")

        log_path = td / "output" / "logs" / "command_exec.ndjson"
        self.assertTrue(log_path.exists())
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 2)

        first = rows[0]
        self.assertEqual(str(first.get("source", "")), "unit")
        self.assertEqual(str(first.get("tag", "")), "ok-case")
        self.assertEqual(int(first.get("returncode", 1)), 0)
        self.assertTrue(bool(first.get("ok", False)))
        self.assertEqual(list(first.get("env_overrides", [])), ["TEST_AUDIT_FLAG"])
        self.assertIn("command", first)
        self.assertIn("timestamp", first)
        self.assertIn("duration_ms", first)

        second = rows[1]
        self.assertEqual(str(second.get("tag", "")), "fail-case")
        self.assertEqual(int(second.get("returncode", 0)), 3)
        self.assertFalse(bool(second.get("ok", True)))


if __name__ == "__main__":
    unittest.main()
