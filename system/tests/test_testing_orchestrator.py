from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch
import json
import subprocess

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
        self.assertEqual(int(out.get("tail_priority_selected", 0)), 0)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[:4], [sys.executable, "-m", "unittest", "-v"])
        self.assertEqual(len(cmd) - 4, 2)

    def test_test_all_fast_mode_forces_tail_priority_floor(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        class _Proc:
            returncode = 0
            stdout = "Ran 5 tests in 0.002s\n\nOK\n"
            stderr = ""

        ids = [
            "tests.test_signal.SignalTests.test_scan",
            "tests.test_data_quality.DataQualityTests.test_low_confidence_source_ratio_can_be_hard_fail",
            "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_stress_exec_trendline_controlled_apply_ledger_drift_hard_fail",
            "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_temporal_audit_autofix_skips_unsafe_candidate",
            "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_reconcile_drift_detects_broker_contract_mismatch",
            "tests.test_misc.MiscTests.test_a",
            "tests.test_misc.MiscTests.test_b",
            "tests.test_misc.MiscTests.test_c",
            "tests.test_misc.MiscTests.test_d",
            "tests.test_misc.MiscTests.test_e",
        ]
        with (
            patch("lie_engine.orchestration.testing.subprocess.run", return_value=_Proc()) as mock_run,
            patch.object(TestingOrchestrator, "_discover_test_ids", return_value=ids),
        ):
            orch = TestingOrchestrator(root=td, output_dir=td)
            out = orch.test_all(
                fast=True,
                fast_ratio=0.1,
                fast_seed="tail-seed",
                fast_tail_priority=True,
                fast_tail_floor=3,
            )

        cmd = mock_run.call_args[0][0]
        selected = list(cmd[4:])
        self.assertGreaterEqual(len(selected), 3)
        self.assertGreaterEqual(int(out.get("tail_priority_selected", 0)), 3)
        self.assertTrue(any("hard_fail" in x for x in selected))
        self.assertTrue(any("temporal" in x.lower() for x in selected))

    def test_test_all_parallel_shard_auto_isolates_workspace(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        (td / "tests").mkdir(parents=True, exist_ok=True)
        (td / "src").mkdir(parents=True, exist_ok=True)

        class _Proc:
            returncode = 0
            stdout = "Ran 1 test in 0.001s\n\nOK\n"
            stderr = ""

        ids = [f"tests.test_mod.Case.test_{i}" for i in range(4)]
        with (
            patch("lie_engine.orchestration.testing.subprocess.run", return_value=_Proc()) as mock_run,
            patch.object(TestingOrchestrator, "_discover_test_ids", return_value=ids),
        ):
            orch = TestingOrchestrator(root=td, output_dir=td)
            out = orch.test_all(
                fast=True,
                fast_ratio=0.5,
                fast_shard_index=1,
                fast_shard_total=4,
                fast_seed="iso-seed",
            )

        self.assertTrue(bool(out.get("workspace_isolation_requested", False)))
        self.assertTrue(bool(out.get("workspace_isolated", False)))
        kwargs = mock_run.call_args.kwargs
        self.assertNotEqual(Path(str(kwargs.get("cwd", td))), td)

    def test_test_chaos_selects_curated_subset(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        class _Proc:
            returncode = 0
            stdout = "Ran 2 tests in 0.002s\n\nOK\n"
            stderr = ""

        ids = [
            "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_stress_exec_trendline_controlled_apply_ledger_drift_hard_fail",
            "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_temporal_audit_autofix_skips_unsafe_candidate",
            "tests.test_signal.SignalTests.test_scan",
        ]
        with (
            patch("lie_engine.orchestration.testing.subprocess.run", return_value=_Proc()) as mock_run,
            patch.object(TestingOrchestrator, "_discover_test_ids", return_value=ids),
        ):
            orch = TestingOrchestrator(root=td, output_dir=td)
            out = orch.test_chaos(
                max_tests=2,
                seed="chaos-seed",
                isolate_shard_workspace=False,
                include_probes=False,
            )

        self.assertEqual(str(out.get("mode", "")), "chaos")
        self.assertEqual(int(out.get("chaos_tests_selected", 0)), 2)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[:4], [sys.executable, "-m", "unittest", "-v"])
        self.assertEqual(len(cmd) - 4, 2)

    def test_test_chaos_probe_failure_escalates_returncode(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        class _Proc:
            returncode = 0
            stdout = "Ran 1 test in 0.001s\n\nOK\n"
            stderr = ""

        ids = ["tests.test_testing_orchestrator.TestingOrchestratorTests.test_test_all_writes_log_payload"]
        with (
            patch("lie_engine.orchestration.testing.subprocess.run", return_value=_Proc()),
            patch.object(TestingOrchestrator, "_discover_test_ids", return_value=ids),
            patch.object(
                TestingOrchestrator,
                "_run_chaos_probes",
                return_value=[{"probe_id": "config_corruption_rejected", "passed": False}],
            ),
        ):
            orch = TestingOrchestrator(root=td, output_dir=td)
            out = orch.test_chaos(
                max_tests=1,
                seed="chaos-seed",
                isolate_shard_workspace=False,
                include_probes=True,
            )

        self.assertEqual(int(out.get("returncode", 0)), 1)
        self.assertIn("__chaos_probe__:config_corruption_rejected", set(out.get("failed_tests", [])))
        self.assertEqual(int(out.get("chaos_probe_count", 0)), 1)
        self.assertEqual(int(out.get("chaos_probe_failed_count", 0)), 1)

    def test_probe_nonzero_with_syntaxerror_is_forced_fail(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "SyntaxError: invalid syntax\n"

        with patch("lie_engine.orchestration.testing.subprocess.run", return_value=_Proc()):
            orch = TestingOrchestrator(root=td, output_dir=td)
            out = orch._run_probe_command(
                probe_id="probe_syntax_guard",
                cmd=[sys.executable, "-c", "print(1)"],
                cwd=td,
                env={},
                timeout_seconds=10,
                pass_when_nonzero=True,
            )

        self.assertEqual(int(out.get("returncode", 0)), 1)
        self.assertFalse(bool(out.get("passed", True)))

    def test_test_all_standard_mode_includes_mandatory_suites(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        class _Proc:
            returncode = 0
            stdout = "Ran 6 tests in 0.010s\n\nOK\n"
            stderr = ""

        ids = [
            "tests.test_config_validation.ConfigValidationTests.test_validate_settings_ok",
            "tests.test_risk.RiskTests.test_budget",
            "tests.test_backtest.BacktestTests.test_backtest_runs",
            "tests.test_backtest_temporal_execution.BacktestTemporalExecutionTests.test_execution_friction_reduces_return",
            "tests.test_release_orchestrator.ReleaseOrchestratorTests.test_gate_report_basic_pass",
            "tests.test_signal.SignalTests.test_scan",
            "tests.test_data_pipeline.DataPipelineTests.test_pipeline_ok",
            "tests.test_storage.StorageTests.test_sqlite",
        ]
        with (
            patch("lie_engine.orchestration.testing.subprocess.run", return_value=_Proc()) as mock_run,
            patch.object(TestingOrchestrator, "_discover_test_ids", return_value=ids),
        ):
            orch = TestingOrchestrator(root=td, output_dir=td)
            out = orch.test_all(
                tier="standard",
                standard_ratio=0.2,
                fast_seed="seed-standard",
            )

        self.assertEqual(out["mode"], "standard")
        self.assertEqual(out["tier"], "standard")
        selected = list(mock_run.call_args[0][0][4:])
        self.assertTrue(any(x.startswith("tests.test_config_validation.") for x in selected))
        self.assertTrue(any(x.startswith("tests.test_risk.") for x in selected))
        self.assertTrue(any(x.startswith("tests.test_backtest.") for x in selected))
        self.assertTrue(any(x.startswith("tests.test_backtest_temporal_execution.") for x in selected))
        self.assertTrue(any(x.startswith("tests.test_release_orchestrator.") for x in selected))
        self.assertGreaterEqual(len(selected), 5)

    def test_test_all_timeout_is_guarded(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))

        timeout_exc = subprocess.TimeoutExpired(cmd=["python", "-m", "unittest"], timeout=1)
        timeout_exc.stdout = "partial output"
        timeout_exc.stderr = "partial error"

        with patch("lie_engine.orchestration.testing.subprocess.run", side_effect=timeout_exc):
            orch = TestingOrchestrator(root=td, output_dir=td, timeout_seconds=60)
            out = orch.test_all()

        self.assertEqual(int(out["returncode"]), 124)
        self.assertTrue(bool(out.get("timed_out", False)))
        self.assertIn("__timeout__", list(out.get("failed_tests", [])))
        self.assertIn("error=test_timeout", str(out.get("summary_line", "")))


if __name__ == "__main__":
    unittest.main()
