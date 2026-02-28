from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class CommandWhitelist24hScriptTests(unittest.TestCase):
    def _script_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "scripts" / "command_whitelist_24h.py"

    def test_whitelist_aggregation_and_samples(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        logs_dir = td / "output" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        inside_1 = (now - timedelta(hours=1)).isoformat()
        inside_2 = (now - timedelta(hours=2)).isoformat()
        inside_3 = (now - timedelta(hours=3)).isoformat()
        outside_1 = (now - timedelta(hours=30)).isoformat()

        command_file = logs_dir / "command_exec.ndjson"
        command_file.write_text(
            "\n".join(
                [
                    json.dumps({"timestamp": inside_1, "command": "lie run-slot --date 2026-02-28 --slot eod", "returncode": 0}),
                    json.dumps({"timestamp": inside_2, "command": "lie run-slot --date 2026-02-28 --slot eod", "returncode": 1}),
                    json.dumps({"timestamp": inside_3, "command": "bash scripts/auto_git_sync.sh --dry-run", "returncode": 0}),
                    json.dumps({"timestamp": outside_1, "command": "lie run-slot --date 2026-02-27 --slot eod", "returncode": 0}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        tests_log = logs_dir / "tests_20260228_000001.json"
        tests_log.write_text(json.dumps({"returncode": 0, "mode": "fast"}, ensure_ascii=False), encoding="utf-8")
        ts = (now - timedelta(hours=4)).timestamp()
        os.utime(tests_log, (ts, ts))

        cmd = [
            "python3",
            str(self._script_path()),
            "--root",
            str(td),
            "--window-hours",
            "24",
            "--max-samples-per-command",
            "3",
            "--include-tests-log",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        self.assertEqual(proc.returncode, 0, msg=f"stdout={proc.stdout}\nstderr={proc.stderr}")

        out = json.loads(proc.stdout)
        self.assertIn("review_json", out)
        payload = json.loads(Path(str(out["review_json"])) .read_text(encoding="utf-8"))
        self.assertEqual(str(payload.get("status", "")), "FAILED")

        totals = payload.get("totals", {})
        self.assertEqual(int(totals.get("events", 0)), 4)
        self.assertEqual(int(totals.get("success_runs", 0)), 3)

        whitelist = payload.get("whitelist", [])
        keys = {str(x.get("command", "")) for x in whitelist if isinstance(x, dict)}
        self.assertIn("lie run-slot", keys)
        self.assertIn("bash auto_git_sync.sh", keys)
        self.assertIn("lie test-all", keys)

        run_slot = next(x for x in whitelist if str(x.get("command", "")) == "lie run-slot")
        self.assertEqual(int(run_slot.get("total_runs", 0)), 2)
        self.assertEqual(int(run_slot.get("success_runs", 0)), 1)
        self.assertEqual(int(run_slot.get("fail_runs", 0)), 1)
        self.assertEqual(len(run_slot.get("samples", [])), 2)


if __name__ == "__main__":
    unittest.main()
