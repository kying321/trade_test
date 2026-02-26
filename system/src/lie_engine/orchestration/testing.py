from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from typing import Any

from lie_engine.data.storage import write_json


@dataclass(slots=True)
class TestingOrchestrator:
    root: Path
    output_dir: Path
    timeout_seconds: int = 1800
    standard_mandatory_prefixes: tuple[str, ...] = (
        "tests.test_config_validation.",
        "tests.test_risk.",
        "tests.test_backtest.",
        "tests.test_backtest_temporal_execution.",
        "tests.test_release_orchestrator.",
    )
    fast_tail_priority_keywords: tuple[str, ...] = (
        "stress",
        "drawdown",
        "temporal",
        "reconcile",
        "rollback",
        "ledger",
        "hard_fail",
        "guard_loop",
        "timeout",
        "risk",
    )
    chaos_mandatory_prefixes: tuple[str, ...] = (
        "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_stress_exec_trendline_controlled_apply_ledger_drift_hard_fail",
        "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_temporal_audit_autofix_skips_unsafe_candidate",
        "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_reconcile_drift_detects_broker_contract_mismatch",
        "tests.test_guard_loop_policy.GuardLoopPolicyTests.test_decide_recovery_heavy_on_health_error",
        "tests.test_data_quality.DataQualityTests.test_low_confidence_source_ratio_can_be_hard_fail",
        "tests.test_testing_orchestrator.",
    )
    shard_workspace_ignore_patterns: tuple[str, ...] = (
        ".git",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".pytest_cache",
        ".venv",
        "node_modules",
        "output",
    )

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

    def _is_tail_priority_test_id(self, test_id: str) -> bool:
        tid = str(test_id).strip().lower()
        if not tid:
            return False
        return any(tok in tid for tok in self.fast_tail_priority_keywords)

    def _materialize_shard_workspace(
        self,
        *,
        fast_shard_index: int,
        shard_workspace_root: str | None = None,
    ) -> Path:
        base_dir = Path(shard_workspace_root).expanduser() if shard_workspace_root else Path(tempfile.gettempdir())
        base_dir.mkdir(parents=True, exist_ok=True)
        sandbox = Path(
            tempfile.mkdtemp(
                prefix=f"lie_shard_{max(0, int(fast_shard_index))}_",
                dir=str(base_dir),
            )
        )
        ignore = shutil.ignore_patterns(*self.shard_workspace_ignore_patterns)
        shutil.copytree(self.root, sandbox, dirs_exist_ok=True, ignore=ignore)
        (sandbox / "output").mkdir(parents=True, exist_ok=True)
        return sandbox

    def _select_fast_subset(
        self,
        test_ids: list[str],
        *,
        fast_ratio: float,
        fast_shard_index: int,
        fast_shard_total: int,
        fast_seed: str,
        fast_tail_priority: bool,
        fast_tail_floor: int,
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
        if not bool(fast_tail_priority):
            return ranked[:n]

        floor = max(0, int(fast_tail_floor))
        global_tail = [
            tid
            for tid in sorted(test_ids, key=lambda tid: self._hash_int(f"{fast_seed}:tail", tid))
            if self._is_tail_priority_test_id(tid)
        ]
        forced_target = min(len(global_tail), max(floor, int(math.ceil(n * 0.25))))
        forced_tail = global_tail[:forced_target]

        selected: list[str] = []
        seen: set[str] = set()
        for tid in forced_tail:
            if tid in seen:
                continue
            seen.add(tid)
            selected.append(tid)
        for tid in ranked:
            if tid in seen:
                continue
            seen.add(tid)
            selected.append(tid)
            if len(selected) >= max(n, forced_target):
                break
        return selected

    def _select_standard_subset(
        self,
        test_ids: list[str],
        *,
        standard_ratio: float,
        fast_seed: str,
        fast_shard_index: int,
        fast_shard_total: int,
        fast_tail_priority: bool,
        fast_tail_floor: int,
    ) -> list[str]:
        if not test_ids:
            return []
        mandatory = {
            tid
            for tid in test_ids
            if any(str(tid).startswith(prefix) for prefix in self.standard_mandatory_prefixes)
        }
        sampled = set(
            self._select_fast_subset(
                test_ids,
                fast_ratio=max(0.05, min(1.0, float(standard_ratio))),
                fast_shard_index=fast_shard_index,
                fast_shard_total=fast_shard_total,
                fast_seed=f"{fast_seed}:standard",
                fast_tail_priority=fast_tail_priority,
                fast_tail_floor=fast_tail_floor,
            )
        )
        selected = list(mandatory | sampled)
        selected.sort(key=lambda tid: self._hash_int(f"{fast_seed}:standard:order", tid))
        return selected

    def _select_chaos_subset(
        self,
        test_ids: list[str],
        *,
        max_tests: int,
        fast_shard_index: int,
        fast_shard_total: int,
        seed: str,
    ) -> list[str]:
        if not test_ids:
            return []
        selected = [
            tid
            for tid in test_ids
            if any(str(tid).startswith(prefix) for prefix in self.chaos_mandatory_prefixes)
        ]
        if not selected:
            selected = [tid for tid in test_ids if self._is_tail_priority_test_id(tid)]
        if not selected:
            selected = list(test_ids)
        selected = sorted(set(selected), key=lambda tid: self._hash_int(f"{seed}:chaos", tid))
        shard_total = max(1, int(fast_shard_total))
        shard_index = max(0, min(shard_total - 1, int(fast_shard_index)))
        shard_ids = [tid for tid in selected if self._hash_int(seed, tid) % shard_total == shard_index]
        if not shard_ids:
            shard_ids = list(selected)
        cap = max(1, int(max_tests))
        return shard_ids[:cap]

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

    def _run_probe_command(
        self,
        *,
        probe_id: str,
        cmd: list[str],
        cwd: Path,
        env: dict[str, str],
        timeout_seconds: int,
        pass_when_nonzero: bool = False,
        require_stdout_token: str = "",
    ) -> dict[str, Any]:
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                text=True,
                capture_output=True,
                env=env,
                timeout=max(5, int(timeout_seconds)),
            )
            rc = int(proc.returncode)
            stdout = str(proc.stdout)
            stderr = str(proc.stderr)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            rc = 124
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")

        if pass_when_nonzero:
            passed = bool(rc != 0 and (not timed_out))
        else:
            passed = bool(rc == 0 and (not timed_out))
        if "syntaxerror" in stderr.lower():
            passed = False
        token = str(require_stdout_token).strip()
        if token:
            passed = bool(passed and token in stdout)
        return {
            "probe_id": str(probe_id),
            "cmd": [str(x) for x in cmd],
            "returncode": int(rc),
            "timed_out": bool(timed_out),
            "passed": bool(passed),
            "stdout_excerpt": self._excerpt(stdout, max_lines=8),
            "stderr_excerpt": self._excerpt(stderr, max_lines=8),
        }

    def _run_chaos_probes(
        self,
        *,
        run_root: Path,
        config_path: Path,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        probes: list[dict[str, Any]] = []
        logs_dir = run_root / "output" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        probe_env = dict(os.environ)
        probe_env["PYTHONWARNINGS"] = "ignore::ResourceWarning"
        probe_env["PYTHONPATH"] = "src"

        bad_cfg = logs_dir / "chaos_bad_config.yaml"
        bad_cfg.write_text("validation:\n  mode_drift_min_live_trades: [broken\n", encoding="utf-8")
        probes.append(
            self._run_probe_command(
                probe_id="config_corruption_rejected",
                cmd=[
                    sys.executable,
                    "-m",
                    "lie_engine.cli",
                    "--config",
                    str(bad_cfg),
                    "validate-config",
                ],
                cwd=run_root,
                env=probe_env,
                timeout_seconds=min(30, int(timeout_seconds)),
                pass_when_nonzero=True,
            )
        )

        sqlite_path = logs_dir / "chaos_truncated.db"
        conn = sqlite3.connect(sqlite_path)
        try:
            conn.execute("create table if not exists t(a integer);")
            conn.execute("insert into t(a) values (1);")
            conn.commit()
        finally:
            conn.close()
        raw = sqlite_path.read_bytes()
        if raw:
            sqlite_path.write_bytes(raw[: max(1, len(raw) // 2)])
        sqlite_probe_script = (
            "import sqlite3,sys\n"
            "p=sys.argv[1]\n"
            "try:\n"
            "    conn=sqlite3.connect(p)\n"
            "    rows=conn.execute('PRAGMA integrity_check').fetchall()\n"
            "    conn.close()\n"
            "    ok=bool(rows) and str(rows[0][0]).lower()=='ok'\n"
            "    print('integrity', rows)\n"
            "    raise SystemExit(0 if ok else 2)\n"
            "except Exception as exc:\n"
            "    print('error', type(exc).__name__)\n"
            "    raise SystemExit(1)\n"
        )
        probes.append(
            self._run_probe_command(
                probe_id="sqlite_truncation_detected",
                cmd=[sys.executable, "-c", sqlite_probe_script, str(sqlite_path)],
                cwd=run_root,
                env=probe_env,
                timeout_seconds=min(30, int(timeout_seconds)),
                pass_when_nonzero=True,
            )
        )

        truncated_json = logs_dir / "chaos_truncated.json"
        truncated_json.write_text('{"a": 1', encoding="utf-8")
        json_probe_script = (
            "from pathlib import Path;import sys;"
            "from lie_engine.engine import LieEngine;"
            "p=Path(sys.argv[1]);"
            "out=LieEngine._load_json_safely(p);"
            "print('safe_empty' if isinstance(out,dict) and len(out)==0 else 'unsafe');"
            "raise SystemExit(0 if isinstance(out,dict) and len(out)==0 else 3)"
        )
        probes.append(
            self._run_probe_command(
                probe_id="json_truncation_safe_load",
                cmd=[sys.executable, "-c", json_probe_script, str(truncated_json)],
                cwd=run_root,
                env=probe_env,
                timeout_seconds=min(30, int(timeout_seconds)),
                pass_when_nonzero=False,
                require_stdout_token="safe_empty",
            )
        )

        kill_target = logs_dir / "chaos_kill_target.tmp"
        worker = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys,time;f=open(sys.argv[1],'w');f.write('partial');f.flush();time.sleep(30)",
                str(kill_target),
            ],
            cwd=run_root,
            env=probe_env,
        )
        time.sleep(0.2)
        worker.kill()
        try:
            worker.wait(timeout=5)
        except Exception:
            pass
        probes.append(
            self._run_probe_command(
                probe_id="kill_recovery_validate_config",
                cmd=[
                    sys.executable,
                    "-m",
                    "lie_engine.cli",
                    "--config",
                    str(config_path),
                    "validate-config",
                ],
                cwd=run_root,
                env=probe_env,
                timeout_seconds=min(30, int(timeout_seconds)),
                pass_when_nonzero=False,
            )
        )
        return probes

    def test_all(
        self,
        *,
        fast: bool = False,
        tier: str | None = None,
        standard_ratio: float = 0.35,
        fast_ratio: float = 0.10,
        fast_shard_index: int = 0,
        fast_shard_total: int = 1,
        fast_seed: str = "lie-fast-v1",
        fast_tail_priority: bool = True,
        fast_tail_floor: int = 3,
        isolate_shard_workspace: bool | None = None,
        shard_workspace_root: str | None = None,
    ) -> dict[str, Any]:
        env = dict(os.environ)
        env["PYTHONWARNINGS"] = "ignore::ResourceWarning"
        timeout_seconds = max(30, int(self.timeout_seconds))
        tests_discovered = 0
        tests_selected = 0
        tail_priority_selected = 0
        workspace_isolation_requested = False
        workspace_isolated = False
        workspace_isolation_error = ""
        workspace_root_used = str(self.root)
        run_root = self.root
        workspace_cleanup_path: Path | None = None
        cmd: list[str]
        requested_tier = str(tier or "").strip().lower()
        if requested_tier not in {"", "fast", "standard", "deep", "full"}:
            requested_tier = ""
        resolved_tier = "fast" if fast else (requested_tier or "deep")
        if resolved_tier == "full":
            resolved_tier = "deep"
        if resolved_tier in {"fast", "standard"}:
            discovered_ids = self._discover_test_ids()
            tests_discovered = len(discovered_ids)
            if resolved_tier == "fast":
                selected_ids = self._select_fast_subset(
                    discovered_ids,
                    fast_ratio=fast_ratio,
                    fast_shard_index=fast_shard_index,
                    fast_shard_total=fast_shard_total,
                    fast_seed=fast_seed,
                    fast_tail_priority=fast_tail_priority,
                    fast_tail_floor=fast_tail_floor,
                )
            else:
                selected_ids = self._select_standard_subset(
                    discovered_ids,
                    standard_ratio=standard_ratio,
                    fast_seed=fast_seed,
                    fast_shard_index=fast_shard_index,
                    fast_shard_total=fast_shard_total,
                    fast_tail_priority=fast_tail_priority,
                    fast_tail_floor=fast_tail_floor,
                )
            tests_selected = len(selected_ids)
            tail_priority_selected = sum(1 for tid in selected_ids if self._is_tail_priority_test_id(tid))
            cmd = [sys.executable, "-m", "unittest", "-v", *selected_ids]
        else:
            cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-t", ".", "-v"]

        if isolate_shard_workspace is None:
            workspace_isolation_requested = bool(resolved_tier in {"fast", "standard"} and int(fast_shard_total) > 1)
        else:
            workspace_isolation_requested = bool(isolate_shard_workspace)
        if workspace_isolation_requested:
            try:
                run_root = self._materialize_shard_workspace(
                    fast_shard_index=fast_shard_index,
                    shard_workspace_root=shard_workspace_root,
                )
                workspace_cleanup_path = run_root
                workspace_isolated = True
                workspace_root_used = str(run_root)
            except Exception as exc:
                workspace_isolation_error = f"{type(exc).__name__}:{exc}"
                run_root = self.root
                workspace_root_used = str(self.root)

        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=run_root,
                text=True,
                capture_output=True,
                env=env,
                timeout=timeout_seconds,
            )
            returncode = int(proc.returncode)
            stdout = str(proc.stdout)
            stderr = str(proc.stderr)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            stderr = (stderr + "\n" if stderr else "") + f"error=test_timeout; timeout_seconds={timeout_seconds}"
        finally:
            if workspace_cleanup_path is not None:
                shutil.rmtree(workspace_cleanup_path, ignore_errors=True)
        failed_tests = self._extract_failed_tests(stdout, stderr)
        if timed_out:
            failed_tests.append("__timeout__")
        ran_count = self._extract_ran_count(stdout, stderr)
        if ran_count is None:
            ran_count = tests_selected if resolved_tier in {"fast", "standard"} else 0
        mode = "full" if resolved_tier == "deep" else resolved_tier
        summary_line = (
            f"error={'none' if returncode == 0 else ('test_timeout' if timed_out else 'test_failure')};"
            + f" mode={mode};"
            + f" discovered={tests_discovered if resolved_tier in {'fast', 'standard'} else 'N/A'};"
            + f" selected={tests_selected if resolved_tier in {'fast', 'standard'} else ran_count};"
            + f" ran={ran_count};"
            + f" failed={len(failed_tests)}"
        )

        log_payload = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "failed_tests": failed_tests,
            "summary_line": summary_line,
            "mode": mode,
            "tier": resolved_tier,
            "fast": bool(resolved_tier == "fast"),
            "standard_ratio": float(standard_ratio),
            "fast_ratio": float(fast_ratio),
            "fast_shard_index": int(fast_shard_index),
            "fast_shard_total": int(fast_shard_total),
            "fast_seed": str(fast_seed),
            "fast_tail_priority": bool(fast_tail_priority),
            "fast_tail_floor": int(max(0, int(fast_tail_floor))),
            "tail_priority_selected": int(tail_priority_selected),
            "workspace_isolation_requested": bool(workspace_isolation_requested),
            "workspace_isolated": bool(workspace_isolated),
            "workspace_isolation_error": str(workspace_isolation_error),
            "workspace_root": str(workspace_root_used),
            "tests_discovered": int(tests_discovered),
            "tests_selected": int(tests_selected if resolved_tier in {"fast", "standard"} else ran_count),
            "tests_ran": int(ran_count),
            "timeout_seconds": int(timeout_seconds),
            "timed_out": bool(timed_out),
        }
        log_path = self.output_dir / "logs" / f"tests_{datetime.now():%Y%m%d_%H%M%S}.json"
        write_json(log_path, log_payload)

        payload = {
            "returncode": returncode,
            "mode": mode,
            "tier": resolved_tier,
            "fast": bool(resolved_tier == "fast"),
            "standard_ratio": float(standard_ratio),
            "fast_ratio": float(fast_ratio),
            "fast_shard_index": int(fast_shard_index),
            "fast_shard_total": int(fast_shard_total),
            "fast_seed": str(fast_seed),
            "fast_tail_priority": bool(fast_tail_priority),
            "fast_tail_floor": int(max(0, int(fast_tail_floor))),
            "tail_priority_selected": int(tail_priority_selected),
            "workspace_isolation_requested": bool(workspace_isolation_requested),
            "workspace_isolated": bool(workspace_isolated),
            "workspace_isolation_error": str(workspace_isolation_error),
            "workspace_root": str(workspace_root_used),
            "tests_discovered": int(tests_discovered),
            "tests_selected": int(tests_selected if resolved_tier in {"fast", "standard"} else ran_count),
            "tests_ran": int(ran_count),
            "failed_tests": failed_tests,
            "summary_line": summary_line,
            "stdout_excerpt": self._excerpt(stdout),
            "stderr_excerpt": self._excerpt(stderr),
            "timeout_seconds": int(timeout_seconds),
            "timed_out": bool(timed_out),
            "log_path": str(log_path),
        }
        return payload

    def test_chaos(
        self,
        *,
        max_tests: int = 24,
        fast_shard_index: int = 0,
        fast_shard_total: int = 1,
        seed: str = "lie-chaos-v1",
        isolate_shard_workspace: bool = True,
        shard_workspace_root: str | None = None,
        include_probes: bool = True,
        probe_timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        discovered_ids = self._discover_test_ids()
        selected_ids = self._select_chaos_subset(
            discovered_ids,
            max_tests=max_tests,
            fast_shard_index=fast_shard_index,
            fast_shard_total=fast_shard_total,
            seed=seed,
        )
        if not selected_ids:
            return {
                "returncode": 0,
                "mode": "chaos",
                "tier": "chaos",
                "tests_discovered": int(len(discovered_ids)),
                "tests_selected": 0,
                "tests_ran": 0,
                "failed_tests": [],
                "summary_line": "error=none; mode=chaos; discovered=0; selected=0; ran=0; failed=0",
                "stdout_excerpt": "",
                "stderr_excerpt": "",
                "timeout_seconds": int(max(30, int(self.timeout_seconds))),
                "timed_out": False,
                "workspace_isolation_requested": bool(isolate_shard_workspace),
                "workspace_isolated": False,
                "workspace_isolation_error": "no_chaos_tests_selected",
                "workspace_root": str(self.root),
                "chaos_probe_count": 0,
                "chaos_probe_failed_count": 0,
                "chaos_probes": [],
                "log_path": "",
            }
        env = dict(os.environ)
        env["PYTHONWARNINGS"] = "ignore::ResourceWarning"
        timeout_seconds = max(30, int(self.timeout_seconds))
        workspace_isolation_requested = bool(isolate_shard_workspace)
        workspace_isolated = False
        workspace_isolation_error = ""
        workspace_root_used = str(self.root)
        run_root = self.root
        workspace_cleanup_path: Path | None = None
        if workspace_isolation_requested:
            try:
                run_root = self._materialize_shard_workspace(
                    fast_shard_index=fast_shard_index,
                    shard_workspace_root=shard_workspace_root,
                )
                workspace_cleanup_path = run_root
                workspace_isolated = True
                workspace_root_used = str(run_root)
            except Exception as exc:
                workspace_isolation_error = f"{type(exc).__name__}:{exc}"
                run_root = self.root
                workspace_root_used = str(self.root)

        cmd = [sys.executable, "-m", "unittest", "-v", *selected_ids]
        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                cwd=run_root,
                text=True,
                capture_output=True,
                env=env,
                timeout=timeout_seconds,
            )
            returncode = int(proc.returncode)
            stdout = str(proc.stdout)
            stderr = str(proc.stderr)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            stderr = (stderr + "\n" if stderr else "") + f"error=test_timeout; timeout_seconds={timeout_seconds}"
        finally:
            if workspace_cleanup_path is not None:
                shutil.rmtree(workspace_cleanup_path, ignore_errors=True)

        failed_tests = self._extract_failed_tests(stdout, stderr)
        if timed_out:
            failed_tests.append("__timeout__")
        ran_count = self._extract_ran_count(stdout, stderr)
        if ran_count is None:
            ran_count = len(selected_ids)
        summary_line = (
            f"error={'none' if returncode == 0 else ('test_timeout' if timed_out else 'test_failure')};"
            + " mode=chaos;"
            + f" discovered={len(discovered_ids)};"
            + f" selected={len(selected_ids)};"
            + f" ran={ran_count};"
            + f" failed={len(failed_tests)}"
        )
        chaos_probes: list[dict[str, Any]] = []
        if bool(include_probes):
            cfg_path = run_root / "config.yaml"
            if not cfg_path.exists():
                cfg_path = self.root / "config.yaml"
            chaos_probes = self._run_chaos_probes(
                run_root=run_root,
                config_path=cfg_path,
                timeout_seconds=max(10, int(probe_timeout_seconds)),
            )
            probe_failed = [
                str(item.get("probe_id", "unknown"))
                for item in chaos_probes
                if isinstance(item, dict) and (not bool(item.get("passed", False)))
            ]
            for pid in probe_failed:
                failed_tests.append(f"__chaos_probe__:{pid}")
            if probe_failed and returncode == 0:
                returncode = 1
            summary_line = summary_line + f"; probes={len(chaos_probes)}; probe_failed={len(probe_failed)}"

        log_payload = {
            "returncode": int(returncode),
            "stdout": stdout,
            "stderr": stderr,
            "failed_tests": failed_tests,
            "summary_line": summary_line,
            "mode": "chaos",
            "tier": "chaos",
            "chaos_seed": str(seed),
            "chaos_tests_discovered": int(len(discovered_ids)),
            "chaos_tests_selected": int(len(selected_ids)),
            "chaos_test_ids": list(selected_ids),
            "timeout_seconds": int(timeout_seconds),
            "timed_out": bool(timed_out),
            "workspace_isolation_requested": bool(workspace_isolation_requested),
            "workspace_isolated": bool(workspace_isolated),
            "workspace_isolation_error": str(workspace_isolation_error),
            "workspace_root": str(workspace_root_used),
            "tests_ran": int(ran_count),
            "chaos_probe_count": int(len(chaos_probes)),
            "chaos_probe_failed_count": int(
                sum(1 for item in chaos_probes if isinstance(item, dict) and (not bool(item.get("passed", False))))
            ),
            "chaos_probes": chaos_probes,
        }
        log_path = self.output_dir / "logs" / f"tests_chaos_{datetime.now():%Y%m%d_%H%M%S}.json"
        write_json(log_path, log_payload)

        return {
            "returncode": int(returncode),
            "mode": "chaos",
            "tier": "chaos",
            "chaos_seed": str(seed),
            "chaos_tests_discovered": int(len(discovered_ids)),
            "chaos_tests_selected": int(len(selected_ids)),
            "chaos_test_ids": list(selected_ids),
            "tests_ran": int(ran_count),
            "failed_tests": failed_tests,
            "summary_line": summary_line,
            "stdout_excerpt": self._excerpt(stdout),
            "stderr_excerpt": self._excerpt(stderr),
            "timeout_seconds": int(timeout_seconds),
            "timed_out": bool(timed_out),
            "workspace_isolation_requested": bool(workspace_isolation_requested),
            "workspace_isolated": bool(workspace_isolated),
            "workspace_isolation_error": str(workspace_isolation_error),
            "workspace_root": str(workspace_root_used),
            "chaos_probe_count": int(len(chaos_probes)),
            "chaos_probe_failed_count": int(
                sum(1 for item in chaos_probes if isinstance(item, dict) and (not bool(item.get("passed", False))))
            ),
            "chaos_probes": chaos_probes,
            "log_path": str(log_path),
        }
