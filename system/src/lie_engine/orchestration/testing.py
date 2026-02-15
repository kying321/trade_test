from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from typing import Any

from lie_engine.data.storage import write_json


@dataclass(slots=True)
class TestingOrchestrator:
    root: Path
    output_dir: Path

    @staticmethod
    def _iter_test_ids(suite: unittest.TestSuite) -> list[str]:
        out: list[str] = []
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                out.extend(TestingOrchestrator._iter_test_ids(item))
            else:
                out.append(str(item.id()))
        return out

    def _discover_test_ids(self) -> list[str]:
        loader = unittest.TestLoader()
        suite = loader.discover(
            start_dir=str(self.root / "tests"),
            pattern="test_*.py",
            top_level_dir=str(self.root),
        )
        ids = self._iter_test_ids(suite)
        return sorted({x for x in ids if x})

    @staticmethod
    def _hash_int(seed: str, value: str) -> int:
        return int(hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()[:16], 16)

    def _select_fast_subset(
        self,
        test_ids: list[str],
        *,
        fast_ratio: float,
        fast_shard_index: int,
        fast_shard_total: int,
        fast_seed: str,
    ) -> list[str]:
        if not test_ids:
            return []
        ratio = max(0.01, min(1.0, float(fast_ratio)))
        shard_total = max(1, int(fast_shard_total))
        shard_index = max(0, min(shard_total - 1, int(fast_shard_index)))
        shard_ids = [tid for tid in test_ids if self._hash_int(fast_seed, tid) % shard_total == shard_index]
        if not shard_ids:
            shard_ids = list(test_ids)
        ranked = sorted(shard_ids, key=lambda tid: self._hash_int(f"{fast_seed}:select", tid))
        n = max(1, int(math.ceil(len(ranked) * ratio)))
        return ranked[:n]

    @staticmethod
    def _extract_failed_tests(stdout: str, stderr: str) -> list[str]:
        text = f"{stderr}\n{stdout}"
        failed: list[str] = []
        for line in text.splitlines():
            txt = line.strip()
            if txt.endswith("... FAIL") or txt.endswith("... ERROR"):
                failed.append(txt.split(" ... ")[0].strip())
            elif txt.startswith("FAIL: ") or txt.startswith("ERROR: "):
                failed.append(txt.split(": ", 1)[1].strip())
        out: list[str] = []
        seen: set[str] = set()
        for item in failed:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @staticmethod
    def _extract_ran_count(stdout: str, stderr: str) -> int | None:
        text = f"{stdout}\n{stderr}"
        m = re.search(r"Ran\s+(\d+)\s+tests?\s+in\s+[0-9.]+s", text)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _excerpt(text: str, max_lines: int = 24) -> str:
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        keep = max(1, max_lines // 2)
        head = lines[:keep]
        tail = lines[-keep:]
        return "\n".join(head + ["... (truncated) ..."] + tail)

    def test_all(
        self,
        *,
        fast: bool = False,
        fast_ratio: float = 0.10,
        fast_shard_index: int = 0,
        fast_shard_total: int = 1,
        fast_seed: str = "lie-fast-v1",
    ) -> dict[str, Any]:
        env = dict(os.environ)
        env["PYTHONWARNINGS"] = "ignore::ResourceWarning"
        tests_discovered = 0
        tests_selected = 0
        cmd: list[str]
        if fast:
            discovered_ids = self._discover_test_ids()
            tests_discovered = len(discovered_ids)
            selected_ids = self._select_fast_subset(
                discovered_ids,
                fast_ratio=fast_ratio,
                fast_shard_index=fast_shard_index,
                fast_shard_total=fast_shard_total,
                fast_seed=fast_seed,
            )
            tests_selected = len(selected_ids)
            cmd = [sys.executable, "-m", "unittest", "-v", *selected_ids]
        else:
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-t", ".", "-v"]

        proc = subprocess.run(
            cmd,
            cwd=self.root,
            text=True,
            capture_output=True,
            env=env,
        )
        failed_tests = self._extract_failed_tests(proc.stdout, proc.stderr)
        ran_count = self._extract_ran_count(proc.stdout, proc.stderr)
        if ran_count is None:
            ran_count = tests_selected if fast else 0
        summary_line = (
            f"error={'none' if proc.returncode == 0 else 'test_failure'};"
            + f" mode={'fast' if fast else 'full'};"
            + f" discovered={tests_discovered if fast else 'N/A'};"
            + f" selected={tests_selected if fast else ran_count};"
            + f" ran={ran_count};"
            + f" failed={len(failed_tests)}"
        )

        log_payload = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "failed_tests": failed_tests,
            "summary_line": summary_line,
            "mode": "fast" if fast else "full",
            "fast": bool(fast),
            "fast_ratio": float(fast_ratio),
            "fast_shard_index": int(fast_shard_index),
            "fast_shard_total": int(fast_shard_total),
            "fast_seed": str(fast_seed),
            "tests_discovered": int(tests_discovered),
            "tests_selected": int(tests_selected if fast else ran_count),
            "tests_ran": int(ran_count),
        }
        log_path = self.output_dir / "logs" / f"tests_{datetime.now():%Y%m%d_%H%M%S}.json"
        write_json(log_path, log_payload)

        payload = {
            "returncode": proc.returncode,
            "mode": "fast" if fast else "full",
            "fast": bool(fast),
            "fast_ratio": float(fast_ratio),
            "fast_shard_index": int(fast_shard_index),
            "fast_shard_total": int(fast_shard_total),
            "fast_seed": str(fast_seed),
            "tests_discovered": int(tests_discovered),
            "tests_selected": int(tests_selected if fast else ran_count),
            "tests_ran": int(ran_count),
            "failed_tests": failed_tests,
            "summary_line": summary_line,
            "stdout_excerpt": self._excerpt(proc.stdout),
            "stderr_excerpt": self._excerpt(proc.stderr),
            "log_path": str(log_path),
        }
        return payload
