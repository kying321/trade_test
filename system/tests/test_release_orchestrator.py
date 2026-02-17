from __future__ import annotations

from contextlib import closing
from datetime import date
import json
from pathlib import Path
import sqlite3
import shutil
import tempfile
import sys
import unittest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.config import SystemSettings
from lie_engine.models import ReviewDelta
from lie_engine.orchestration.release import ReleaseOrchestrator


class ReleaseOrchestratorTests(unittest.TestCase):
    def _make_settings(self) -> SystemSettings:
        return SystemSettings(
            raw={
                "timezone": "Asia/Shanghai",
                "validation": {
                    "data_completeness_min": 0.99,
                    "unresolved_conflict_max": 0.005,
                    "positive_window_ratio_min": 0.70,
                    "max_drawdown_max": 0.18,
                    "required_stable_replay_days": 3,
                    "mode_health_min_samples": 1,
                    "review_loop_fast_test_enabled": True,
                    "review_loop_fast_ratio": 0.2,
                    "review_loop_fast_shard_index": 0,
                    "review_loop_fast_shard_total": 1,
                    "review_loop_fast_seed": "seed-r",
                    "review_loop_fast_then_full": True,
                    "mode_drift_window_days": 120,
                    "mode_drift_min_live_trades": 30,
                    "mode_drift_win_rate_max_gap": 0.12,
                    "mode_drift_profit_factor_max_gap": 0.40,
                    "mode_drift_focus_runtime_mode_only": True,
                },
            }
        )

    def test_gate_report_pass(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(out["passed"])
        self.assertIn("mode_health_ok", out["checks"])
        self.assertTrue(out["checks"]["mode_health_ok"])
        self.assertTrue((review_dir / f"{d.isoformat()}_gate_report.json").exists())

    def test_gate_report_fails_on_mode_health(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: false\n",
            encoding="utf-8",
        )

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(out["passed"])
        self.assertFalse(out["checks"]["mode_health_ok"])

    def test_review_until_pass_failure_writes_defect_plan(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 0.8, "unresolved_conflict_ratio": 0.02},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.5, "max_drawdown": 0.3, "violations": 1},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=["x"], pass_gate=False),
            health_check=lambda as_of, require_review: {"status": "degraded", "missing": ["review"]},
            stable_replay_check=lambda as_of, days: {"passed": False, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 1, "stderr": "test_x ... FAIL", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        self.assertIn("defect_plan", out["rounds"][0])
        plan_json = Path(out["rounds"][0]["defect_plan"]["json"])
        plan_md = Path(out["rounds"][0]["defect_plan"]["md"])
        self.assertTrue(plan_json.exists())
        self.assertTrue(plan_md.exists())

    def test_review_until_pass_runs_fast_then_full_on_round_one(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        calls: list[dict[str, object]] = []

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True)

        def _test_all(**kwargs):
            calls.append(dict(kwargs))
            if kwargs.get("fast"):
                return {"returncode": 0, "summary_line": "error=none; mode=fast", "tests_ran": 3, "failed_tests": []}
            return {"returncode": 0, "summary_line": "error=none; mode=full", "tests_ran": 11, "failed_tests": []}

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=_test_all,
            load_json_safely=lambda p: {},
        )

        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertTrue(out["passed"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(bool(calls[0].get("fast", False)))
        self.assertFalse(bool(calls[1].get("fast", False)))
        self.assertEqual(out["rounds"][0]["tests_mode"], "fast+full")
        self.assertEqual(int(out["rounds"][0]["fast_tests"]["tests_ran"]), 3)
        self.assertEqual(int(out["rounds"][0]["full_tests"]["tests_ran"]), 11)

    def test_review_until_pass_timeout_triggers_fast_fallback_and_tags_defect(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        calls: list[dict[str, object]] = []

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "review_loop_timeout_fallback_enabled": True,
                "review_loop_timeout_fallback_ratio": 0.12,
                "review_loop_timeout_fallback_shard_index": 0,
                "review_loop_timeout_fallback_shard_total": 2,
                "review_loop_timeout_fallback_seed": "timeout-seed",
            }
        )

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True)

        def _test_all(**kwargs):
            calls.append(dict(kwargs))
            if kwargs.get("fast"):
                seed = str(kwargs.get("fast_seed", ""))
                if seed == "timeout-seed":
                    return {
                        "returncode": 0,
                        "summary_line": "error=none; mode=fast; timeout_fallback=true",
                        "tests_ran": 5,
                        "failed_tests": [],
                    }
                return {"returncode": 0, "summary_line": "error=none; mode=fast", "tests_ran": 3, "failed_tests": []}
            return {
                "returncode": 124,
                "timed_out": True,
                "summary_line": "error=test_timeout; mode=full",
                "tests_ran": 0,
                "failed_tests": ["__timeout__"],
            }

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=_test_all,
            load_json_safely=lambda p: {},
        )

        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        self.assertEqual(len(calls), 3)
        round0 = out["rounds"][0]
        self.assertTrue(bool(round0.get("tests_timeout", False)))
        self.assertTrue(bool(round0.get("timeout_fallback_used", False)))
        self.assertTrue(str(round0.get("tests_mode", "")).endswith("timeout-fast"))
        timeout_fb = round0.get("timeout_fallback", {})
        self.assertEqual(int(timeout_fb.get("returncode", 1)), 0)
        self.assertEqual(int(timeout_fb.get("fast_shard_total", 0)), 2)
        self.assertEqual(str(timeout_fb.get("fast_seed", "")), "timeout-seed")
        plan_json = Path(str(round0["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("TEST_TIMEOUT", codes)
        next_actions = [str(x) for x in plan.get("next_actions", [])]
        self.assertTrue(next_actions)
        self.assertIn("fast shard", next_actions[0])

    def test_ops_report_flags_state_instability(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        daily_dir = td / "daily"
        review_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        modes = ["swing", "long", "swing", "long", "swing"]
        risks = [0.60, 0.55, 0.30, 0.20, 0.10]
        sources = [0.82, 0.80, 0.74, 0.72, 0.68]
        health = [True, True, False, False, False]
        for i in range(5):
            day = date.fromordinal(d.toordinal() - (4 - i))
            payload = {
                "runtime_mode": modes[i],
                "mode_health": {"passed": health[i]},
                "risk_control": {
                    "risk_multiplier": risks[i],
                    "source_confidence_score": sources[i],
                },
            }
            (daily_dir / f"{day.isoformat()}_mode_feedback.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "mode_switch_window_days": 10,
                "ops_state_min_samples": 5,
                "mode_switch_max_rate": 0.40,
                "ops_risk_multiplier_floor": 0.35,
                "ops_risk_multiplier_drift_max": 0.20,
                "ops_source_confidence_floor": 0.75,
                "ops_mode_health_fail_days_max": 1,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        self.assertEqual(out["status"], "red")
        state = out["state_stability"]
        self.assertTrue(bool(state.get("active", False)))
        checks = state.get("checks", {})
        self.assertFalse(bool(checks.get("switch_rate_ok", True)))
        self.assertFalse(bool(checks.get("risk_multiplier_floor_ok", True)))
        self.assertFalse(bool(checks.get("source_confidence_floor_ok", True)))
        self.assertFalse(bool(checks.get("mode_health_fail_days_ok", True)))
        alerts = set(state.get("alerts", []))
        self.assertIn("mode_switch_rate_high", alerts)
        self.assertIn("risk_multiplier_too_low", alerts)
        self.assertIn("source_confidence_too_low", alerts)
        self.assertIn("mode_health_fail_days_high", alerts)
        self.assertTrue((review_dir / f"{d.isoformat()}_ops_report.json").exists())
        self.assertTrue((review_dir / f"{d.isoformat()}_ops_report.md").exists())

    def test_review_until_pass_defect_plan_includes_state_stability_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        daily_dir = td / "daily"
        review_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        modes = ["swing", "long", "swing", "long", "swing"]
        risks = [0.60, 0.55, 0.30, 0.20, 0.10]
        sources = [0.82, 0.80, 0.74, 0.72, 0.68]
        health = [True, True, False, False, False]
        for i in range(5):
            day = date.fromordinal(d.toordinal() - (4 - i))
            payload = {
                "runtime_mode": modes[i],
                "mode_health": {"passed": health[i]},
                "risk_control": {
                    "risk_multiplier": risks[i],
                    "source_confidence_score": sources[i],
                },
            }
            (daily_dir / f"{day.isoformat()}_mode_feedback.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "mode_switch_window_days": 10,
                "ops_state_min_samples": 5,
                "mode_switch_max_rate": 0.40,
                "ops_risk_multiplier_floor": 0.35,
                "ops_risk_multiplier_drift_max": 0.20,
                "ops_source_confidence_floor": 0.75,
                "ops_mode_health_fail_days_max": 1,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 1, "stderr": "test_x ... FAIL", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        round0 = out["rounds"][0]
        self.assertTrue(bool(round0.get("state_stability_active", False)))
        self.assertFalse(bool(round0.get("state_stability_ok", True)))
        plan_json = Path(str(round0["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("STATE_MODE_SWITCH", codes)
        self.assertIn("STATE_RISK_MULT_FLOOR", codes)
        self.assertIn("STATE_SOURCE_CONFIDENCE", codes)
        self.assertIn("STATE_MODE_HEALTH_DAYS", codes)
        next_actions = [str(x) for x in plan.get("next_actions", [])]
        self.assertTrue(next_actions)
        self.assertIn("state_stability", next_actions[0])

    def test_gate_report_fails_on_slot_anomaly(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            pre = {
                "quality": {"passed": True, "flags": ["timestamp_drift"], "source_confidence_score": 0.85},
                "protection_mode": False,
                "risk_multiplier": 1.0,
            }
            (logs_dir / f"{dstr}_premarket.json").write_text(json.dumps(pre, ensure_ascii=False), encoding="utf-8")
            intra = {
                "quality_flags": [],
                "protection_mode": False,
                "source_confidence_score": 0.86,
                "risk_multiplier": 1.0,
            }
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intra, ensure_ascii=False), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intra, ensure_ascii=False), encoding="utf-8")
            eod = {
                "checks": {"quality_passed": True, "trade_blocked": False},
                "metrics": {"risk_multiplier": 1.0},
            }
            (manifest_dir / f"eod_{dstr}.json").write_text(json.dumps(eod, ensure_ascii=False), encoding="utf-8")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 0.2,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_slot_source_confidence_floor": 0.75,
                "ops_slot_risk_multiplier_floor": 0.20,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"]["slot_anomaly_ok"]))
        slot = out.get("slot_anomaly", {})
        self.assertTrue(bool(slot.get("active", False)))
        self.assertIn("slot_premarket_anomaly_high", set(slot.get("alerts", [])))
        self.assertFalse(out["passed"])

    def test_review_until_pass_defect_plan_includes_slot_anomaly_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            pre = {
                "quality": {"passed": True, "flags": ["timestamp_drift"], "source_confidence_score": 0.85},
                "protection_mode": False,
                "risk_multiplier": 1.0,
            }
            (logs_dir / f"{dstr}_premarket.json").write_text(json.dumps(pre, ensure_ascii=False), encoding="utf-8")
            intra = {
                "quality_flags": [],
                "protection_mode": False,
                "source_confidence_score": 0.86,
                "risk_multiplier": 1.0,
            }
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intra, ensure_ascii=False), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intra, ensure_ascii=False), encoding="utf-8")
            eod = {
                "checks": {"quality_passed": True, "trade_blocked": False},
                "metrics": {"risk_multiplier": 1.0},
            }
            (manifest_dir / f"eod_{dstr}.json").write_text(json.dumps(eod, ensure_ascii=False), encoding="utf-8")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 0.2,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_slot_source_confidence_floor": 0.75,
                "ops_slot_risk_multiplier_floor": 0.20,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 1, "stderr": "test_x ... FAIL", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        round0 = out["rounds"][0]
        self.assertTrue(bool(round0.get("slot_anomaly_active", False)))
        self.assertFalse(bool(round0.get("slot_anomaly_ok", True)))
        plan_json = Path(str(round0["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("SLOT_ANOMALY", codes)
        self.assertIn("SLOT_PREMARKET_ANOMALY", codes)
        next_actions = [str(x) for x in plan.get("next_actions", [])]
        self.assertTrue(next_actions)
        self.assertIn("slot_anomaly", next_actions[0])

    def test_gate_report_fails_on_mode_drift_gap(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        daily_dir = td / "daily"
        review_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        feedback = {
            "runtime_mode": "swing",
            "history": {
                "modes": {
                    "swing": {
                        "samples": 8,
                        "avg_win_rate": 0.65,
                        "avg_profit_factor": 1.9,
                    }
                }
            },
        }
        (daily_dir / f"{d.isoformat()}_mode_feedback.json").write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE executed_plans (date TEXT, runtime_mode TEXT, mode TEXT, pnl REAL, direction TEXT)"
            )
            rows = []
            for i in range(12):
                pnl = 1.0 if i < 2 else -1.2
                rows.append((d.isoformat(), "swing", "swing", pnl, "LONG"))
            conn.executemany(
                "INSERT INTO executed_plans (date, runtime_mode, mode, pnl, direction) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "mode_drift_window_days": 30,
                "mode_drift_min_live_trades": 10,
                "mode_drift_win_rate_max_gap": 0.15,
                "mode_drift_profit_factor_max_gap": 0.50,
                "mode_drift_focus_runtime_mode_only": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"]["mode_drift_ok"]))
        drift = out.get("mode_drift", {})
        self.assertTrue(bool(drift.get("active", False)))
        self.assertFalse(out["passed"])

    def test_gate_report_fails_on_stress_matrix_trend_drop(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_payload(robustness: float, annual: float, drawdown: float, fail_rows: int) -> dict[str, object]:
            matrix = []
            for i in range(4):
                matrix.append({"mode": "swing", "window": f"w{i}", "status": "no_data" if i < fail_rows else "ok"})
            return {
                "best_mode": "swing",
                "mode_summary": [
                    {
                        "mode": "swing",
                        "robustness_score": robustness,
                        "avg_annual_return": annual,
                        "worst_drawdown": drawdown,
                    }
                ],
                "matrix": matrix,
            }

        (review_dir / "2026-02-11_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.65, 0.18, 0.12, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.62, 0.17, 0.11, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.18, 0.04, 0.32, 2), ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": True,
                "ops_stress_matrix_trend_window_runs": 3,
                "ops_stress_matrix_trend_min_runs": 3,
                "ops_stress_matrix_robustness_drop_max": 0.10,
                "ops_stress_matrix_annual_return_drop_max": 0.05,
                "ops_stress_matrix_drawdown_rise_max": 0.05,
                "ops_stress_matrix_fail_ratio_max": 0.20,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"]["stress_matrix_trend_ok"]))
        self.assertFalse(out["passed"])
        stress = out.get("stress_matrix_trend", {})
        self.assertTrue(bool(stress.get("active", False)))
        self.assertIn("stress_matrix_robustness_drop", set(stress.get("alerts", [])))

    def test_review_until_pass_autoruns_stress_matrix_on_stress_breach(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_payload(robustness: float, annual: float, drawdown: float, fail_rows: int) -> dict[str, object]:
            matrix = []
            for i in range(4):
                matrix.append({"mode": "swing", "window": f"w{i}", "status": "no_data" if i < fail_rows else "ok"})
            return {
                "best_mode": "swing",
                "mode_summary": [
                    {
                        "mode": "swing",
                        "robustness_score": robustness,
                        "avg_annual_return": annual,
                        "worst_drawdown": drawdown,
                    }
                ],
                "matrix": matrix,
            }

        (review_dir / "2026-02-11_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.70, 0.20, 0.10, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.66, 0.18, 0.11, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.20, 0.04, 0.30, 2), ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": True,
                "ops_stress_matrix_trend_window_runs": 3,
                "ops_stress_matrix_trend_min_runs": 3,
                "ops_stress_matrix_robustness_drop_max": 0.10,
                "ops_stress_matrix_annual_return_drop_max": 0.05,
                "ops_stress_matrix_drawdown_rise_max": 0.05,
                "ops_stress_matrix_fail_ratio_max": 0.20,
                "review_loop_stress_matrix_autorun_enabled": True,
                "review_loop_stress_matrix_autorun_on_mode_drift": False,
                "review_loop_stress_matrix_autorun_on_stress_breach": True,
                "review_loop_stress_matrix_autorun_max_runs": 1,
                "review_loop_stress_matrix_autorun_modes": ["swing", "long"],
            }
        )

        calls: list[dict[str, object]] = []

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True)

        def _run_stress(as_of: date, modes: list[str] | None) -> dict[str, object]:
            calls.append({"date": as_of.isoformat(), "modes": list(modes or [])})
            return {
                "date": as_of.isoformat(),
                "best_mode": "swing",
                "mode_count": 2,
                "window_count": 4,
                "paths": {
                    "json": str(review_dir / f"{as_of.isoformat()}_mode_stress_matrix.json"),
                    "md": str(review_dir / f"{as_of.isoformat()}_mode_stress_matrix.md"),
                },
            }

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_stress_matrix=_run_stress,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["modes"], ["swing", "long"])
        round0 = out["rounds"][0]
        self.assertTrue(bool(round0.get("stress_matrix_trend_active", False)))
        self.assertFalse(bool(round0.get("stress_matrix_trend_ok", True)))
        auto = round0.get("stress_matrix_autorun", {})
        self.assertTrue(bool(auto.get("enabled", False)))
        self.assertTrue(bool(auto.get("triggered", False)))
        self.assertTrue(bool(auto.get("ran", False)))
        self.assertIn("stress_trend", set(auto.get("reason_codes", [])))
        out_meta = auto.get("output", {}) if isinstance(auto.get("output", {}), dict) else {}
        self.assertEqual(str(out_meta.get("best_mode", "")), "swing")

    def test_review_until_pass_stress_autorun_honors_cooldown_backoff(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_payload(robustness: float, annual: float, drawdown: float, fail_rows: int) -> dict[str, object]:
            matrix = []
            for i in range(4):
                matrix.append({"mode": "swing", "window": f"w{i}", "status": "no_data" if i < fail_rows else "ok"})
            return {
                "best_mode": "swing",
                "mode_summary": [
                    {
                        "mode": "swing",
                        "robustness_score": robustness,
                        "avg_annual_return": annual,
                        "worst_drawdown": drawdown,
                    }
                ],
                "matrix": matrix,
            }

        (review_dir / "2026-02-11_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.70, 0.20, 0.10, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.66, 0.18, 0.11, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.20, 0.04, 0.30, 2), ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": True,
                "ops_stress_matrix_trend_window_runs": 3,
                "ops_stress_matrix_trend_min_runs": 3,
                "ops_stress_matrix_robustness_drop_max": 0.10,
                "ops_stress_matrix_annual_return_drop_max": 0.05,
                "ops_stress_matrix_drawdown_rise_max": 0.05,
                "ops_stress_matrix_fail_ratio_max": 0.20,
                "review_loop_stress_matrix_autorun_enabled": True,
                "review_loop_stress_matrix_autorun_on_mode_drift": False,
                "review_loop_stress_matrix_autorun_on_stress_breach": True,
                "review_loop_stress_matrix_autorun_max_runs": 3,
                "review_loop_stress_matrix_autorun_cooldown_rounds": 1,
                "review_loop_stress_matrix_autorun_backoff_multiplier": 2.0,
                "review_loop_stress_matrix_autorun_backoff_max_rounds": 4,
                "review_loop_stress_matrix_autorun_modes": ["swing"],
            }
        )

        calls: list[dict[str, object]] = []

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=False)

        def _run_stress(as_of: date, modes: list[str] | None) -> dict[str, object]:
            calls.append({"date": as_of.isoformat(), "modes": list(modes or [])})
            return {
                "date": as_of.isoformat(),
                "best_mode": "swing",
                "mode_count": 1,
                "window_count": 4,
                "paths": {
                    "json": str(review_dir / f"{as_of.isoformat()}_mode_stress_matrix.json"),
                    "md": str(review_dir / f"{as_of.isoformat()}_mode_stress_matrix.md"),
                },
            }

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_stress_matrix=_run_stress,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=4)
        self.assertFalse(out["passed"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["modes"], ["swing"])
        self.assertEqual(calls[1]["modes"], ["swing"])

        round1 = out["rounds"][0]["stress_matrix_autorun"]
        round2 = out["rounds"][1]["stress_matrix_autorun"]
        round3 = out["rounds"][2]["stress_matrix_autorun"]
        round4 = out["rounds"][3]["stress_matrix_autorun"]
        self.assertTrue(bool(round1.get("ran", False)))
        self.assertEqual(int(round1.get("runs_used", 0)), 1)
        self.assertEqual(int(round1.get("next_allowed_round", 0)), 3)
        self.assertEqual(int(round1.get("cooldown_rounds_current", 0)), 2)
        self.assertFalse(bool(round2.get("ran", True)))
        self.assertEqual(str(round2.get("skipped_reason", "")), "cooldown_active")
        self.assertEqual(int(round2.get("runs_used", 0)), 1)
        self.assertTrue(bool(round3.get("ran", False)))
        self.assertEqual(int(round3.get("runs_used", 0)), 2)
        self.assertEqual(int(round3.get("next_allowed_round", 0)), 6)
        self.assertEqual(int(round3.get("cooldown_rounds_current", 0)), 4)
        self.assertFalse(bool(round4.get("ran", True)))
        self.assertEqual(str(round4.get("skipped_reason", "")), "cooldown_active")
        self.assertEqual(int(round4.get("runs_used", 0)), 2)

    def test_review_until_pass_stress_autorun_adaptive_throttles_high_density(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_payload(robustness: float, annual: float, drawdown: float, fail_rows: int) -> dict[str, object]:
            matrix = []
            for i in range(4):
                matrix.append({"mode": "swing", "window": f"w{i}", "status": "no_data" if i < fail_rows else "ok"})
            return {
                "best_mode": "swing",
                "mode_summary": [
                    {
                        "mode": "swing",
                        "robustness_score": robustness,
                        "avg_annual_return": annual,
                        "worst_drawdown": drawdown,
                    }
                ],
                "matrix": matrix,
            }

        (review_dir / "2026-02-11_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.70, 0.20, 0.10, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.66, 0.18, 0.11, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.20, 0.04, 0.30, 2), ensure_ascii=False),
            encoding="utf-8",
        )

        history_rounds = []
        for i in range(4):
            history_rounds.append(
                {
                    "round": i + 1,
                    "stress_matrix_autorun": {
                        "enabled": True,
                        "triggered": True,
                        "attempted": True,
                        "ran": True,
                        "reason_codes": ["stress_trend"],
                        "skipped_reason": "",
                        "runs_used": i + 1,
                        "next_allowed_round": i + 2,
                        "cooldown_remaining_rounds": 0,
                    },
                }
            )
        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps({"date": "2026-02-12", "passed": False, "rounds": history_rounds}, ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": True,
                "ops_stress_matrix_trend_window_runs": 3,
                "ops_stress_matrix_trend_min_runs": 3,
                "ops_stress_matrix_robustness_drop_max": 0.10,
                "ops_stress_matrix_annual_return_drop_max": 0.05,
                "ops_stress_matrix_drawdown_rise_max": 0.05,
                "ops_stress_matrix_fail_ratio_max": 0.20,
                "review_loop_stress_matrix_autorun_enabled": True,
                "review_loop_stress_matrix_autorun_on_mode_drift": False,
                "review_loop_stress_matrix_autorun_on_stress_breach": True,
                "review_loop_stress_matrix_autorun_max_runs": 4,
                "review_loop_stress_matrix_autorun_adaptive_enabled": True,
                "review_loop_stress_matrix_autorun_adaptive_window_days": 3,
                "review_loop_stress_matrix_autorun_adaptive_min_rounds": 2,
                "review_loop_stress_matrix_autorun_adaptive_low_density_threshold": 0.20,
                "review_loop_stress_matrix_autorun_adaptive_high_density_threshold": 0.60,
                "review_loop_stress_matrix_autorun_adaptive_low_density_factor": 2.0,
                "review_loop_stress_matrix_autorun_adaptive_high_density_factor": 0.25,
                "review_loop_stress_matrix_autorun_adaptive_min_runs_floor": 1,
                "review_loop_stress_matrix_autorun_adaptive_max_runs_cap": 5,
                "review_loop_stress_matrix_autorun_modes": ["swing"],
            }
        )

        calls: list[dict[str, object]] = []

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=False)

        def _run_stress(as_of: date, modes: list[str] | None) -> dict[str, object]:
            calls.append({"date": as_of.isoformat(), "modes": list(modes or [])})
            return {
                "date": as_of.isoformat(),
                "best_mode": "swing",
                "mode_count": 1,
                "window_count": 4,
                "paths": {
                    "json": str(review_dir / f"{as_of.isoformat()}_mode_stress_matrix.json"),
                    "md": str(review_dir / f"{as_of.isoformat()}_mode_stress_matrix.md"),
                },
            }

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_stress_matrix=_run_stress,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=2)
        self.assertFalse(out["passed"])
        self.assertEqual(len(calls), 1)

        round1 = out["rounds"][0]["stress_matrix_autorun"]
        round2 = out["rounds"][1]["stress_matrix_autorun"]
        self.assertTrue(bool(round1.get("ran", False)))
        self.assertEqual(int(round1.get("max_runs", 0)), 1)
        self.assertEqual(int(round1.get("max_runs_base", 0)), 4)
        adaptive1 = round1.get("adaptive", {})
        self.assertEqual(str(adaptive1.get("reason", "")), "high_density_throttle")
        self.assertAlmostEqual(float(adaptive1.get("factor", 0.0)), 0.25, places=8)
        self.assertGreaterEqual(float(adaptive1.get("trigger_density", 0.0)), 0.60)
        self.assertFalse(bool(round2.get("ran", True)))
        self.assertEqual(str(round2.get("skipped_reason", "")), "max_runs_reached")
        self.assertEqual(int(round2.get("runs_used", 0)), 1)

    def test_review_until_pass_stress_autorun_adaptive_expands_low_density(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_payload(robustness: float, annual: float, drawdown: float, fail_rows: int) -> dict[str, object]:
            matrix = []
            for i in range(4):
                matrix.append({"mode": "swing", "window": f"w{i}", "status": "no_data" if i < fail_rows else "ok"})
            return {
                "best_mode": "swing",
                "mode_summary": [
                    {
                        "mode": "swing",
                        "robustness_score": robustness,
                        "avg_annual_return": annual,
                        "worst_drawdown": drawdown,
                    }
                ],
                "matrix": matrix,
            }

        (review_dir / "2026-02-11_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.70, 0.20, 0.10, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.66, 0.18, 0.11, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.20, 0.04, 0.30, 2), ensure_ascii=False),
            encoding="utf-8",
        )

        history_rounds = []
        for i in range(10):
            history_rounds.append(
                {
                    "round": i + 1,
                    "stress_matrix_autorun": {
                        "enabled": True,
                        "triggered": False,
                        "attempted": False,
                        "ran": False,
                        "reason_codes": [],
                        "skipped_reason": "",
                        "runs_used": 0,
                        "next_allowed_round": 1,
                        "cooldown_remaining_rounds": 0,
                    },
                }
            )
        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps({"date": "2026-02-12", "passed": True, "rounds": history_rounds}, ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": True,
                "ops_stress_matrix_trend_window_runs": 3,
                "ops_stress_matrix_trend_min_runs": 3,
                "ops_stress_matrix_robustness_drop_max": 0.10,
                "ops_stress_matrix_annual_return_drop_max": 0.05,
                "ops_stress_matrix_drawdown_rise_max": 0.05,
                "ops_stress_matrix_fail_ratio_max": 0.20,
                "review_loop_stress_matrix_autorun_enabled": True,
                "review_loop_stress_matrix_autorun_on_mode_drift": False,
                "review_loop_stress_matrix_autorun_on_stress_breach": True,
                "review_loop_stress_matrix_autorun_max_runs": 1,
                "review_loop_stress_matrix_autorun_adaptive_enabled": True,
                "review_loop_stress_matrix_autorun_adaptive_window_days": 3,
                "review_loop_stress_matrix_autorun_adaptive_min_rounds": 2,
                "review_loop_stress_matrix_autorun_adaptive_low_density_threshold": 0.35,
                "review_loop_stress_matrix_autorun_adaptive_high_density_threshold": 0.90,
                "review_loop_stress_matrix_autorun_adaptive_low_density_factor": 3.0,
                "review_loop_stress_matrix_autorun_adaptive_high_density_factor": 0.50,
                "review_loop_stress_matrix_autorun_adaptive_min_runs_floor": 1,
                "review_loop_stress_matrix_autorun_adaptive_max_runs_cap": 4,
                "review_loop_stress_matrix_autorun_modes": ["swing"],
            }
        )

        calls: list[dict[str, object]] = []

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=False)

        def _run_stress(as_of: date, modes: list[str] | None) -> dict[str, object]:
            calls.append({"date": as_of.isoformat(), "modes": list(modes or [])})
            return {
                "date": as_of.isoformat(),
                "best_mode": "swing",
                "mode_count": 1,
                "window_count": 4,
                "paths": {
                    "json": str(review_dir / f"{as_of.isoformat()}_mode_stress_matrix.json"),
                    "md": str(review_dir / f"{as_of.isoformat()}_mode_stress_matrix.md"),
                },
            }

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_stress_matrix=_run_stress,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=4)
        self.assertFalse(out["passed"])
        self.assertEqual(len(calls), 3)

        round1 = out["rounds"][0]["stress_matrix_autorun"]
        round3 = out["rounds"][2]["stress_matrix_autorun"]
        round4 = out["rounds"][3]["stress_matrix_autorun"]
        self.assertTrue(bool(round1.get("ran", False)))
        self.assertEqual(int(round1.get("max_runs", 0)), 3)
        self.assertEqual(int(round1.get("max_runs_base", 0)), 1)
        adaptive1 = round1.get("adaptive", {})
        self.assertEqual(str(adaptive1.get("reason", "")), "low_density_expand")
        self.assertAlmostEqual(float(adaptive1.get("factor", 0.0)), 3.0, places=8)
        self.assertLessEqual(float(adaptive1.get("trigger_density", 1.0)), 0.35)
        self.assertTrue(bool(round3.get("ran", False)))
        self.assertEqual(int(round3.get("runs_used", 0)), 3)
        self.assertFalse(bool(round4.get("ran", True)))
        self.assertEqual(str(round4.get("skipped_reason", "")), "max_runs_reached")
        self.assertEqual(int(round4.get("runs_used", 0)), 3)

    def test_review_until_pass_defect_plan_includes_stress_matrix_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_payload(robustness: float, annual: float, drawdown: float, fail_rows: int) -> dict[str, object]:
            matrix = []
            for i in range(4):
                matrix.append({"mode": "swing", "window": f"w{i}", "status": "no_data" if i < fail_rows else "ok"})
            return {
                "best_mode": "swing",
                "mode_summary": [
                    {
                        "mode": "swing",
                        "robustness_score": robustness,
                        "avg_annual_return": annual,
                        "worst_drawdown": drawdown,
                    }
                ],
                "matrix": matrix,
            }

        (review_dir / "2026-02-11_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.60, 0.16, 0.11, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.58, 0.15, 0.12, 0), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_payload(0.20, 0.03, 0.30, 2), ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": True,
                "ops_stress_matrix_trend_window_runs": 3,
                "ops_stress_matrix_trend_min_runs": 3,
                "ops_stress_matrix_robustness_drop_max": 0.10,
                "ops_stress_matrix_annual_return_drop_max": 0.05,
                "ops_stress_matrix_drawdown_rise_max": 0.05,
                "ops_stress_matrix_fail_ratio_max": 0.20,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        plan_json = Path(str(out["rounds"][0]["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("STRESS_MATRIX_ROBUSTNESS", codes)
        self.assertIn("STRESS_MATRIX_DRAWDOWN", codes)
        self.assertIn("STRESS_MATRIX_FAIL_RATIO", codes)

    def test_ops_report_includes_stress_autorun_history_metrics(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps(
                {
                    "date": "2026-02-12",
                    "passed": True,
                    "rounds": [
                        {
                            "round": 1,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": True,
                                "ran": True,
                                "reason_codes": ["stress_trend"],
                                "skipped_reason": "",
                                "runs_used": 1,
                                "next_allowed_round": 3,
                                "cooldown_remaining_rounds": 2,
                            },
                        },
                        {
                            "round": 2,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": False,
                                "ran": False,
                                "reason_codes": ["stress_trend"],
                                "skipped_reason": "cooldown_active",
                                "runs_used": 1,
                                "next_allowed_round": 3,
                                "cooldown_remaining_rounds": 1,
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "review_loop_alert_2026-02-13.json").write_text(
            json.dumps(
                {
                    "passed": False,
                    "rounds": [
                        {
                            "round": 1,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": False,
                                "ran": False,
                                "reason_codes": ["mode_drift"],
                                "skipped_reason": "max_runs_reached",
                                "runs_used": 1,
                                "next_allowed_round": 6,
                                "cooldown_remaining_rounds": 3,
                            },
                        },
                        {
                            "round": 2,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": False,
                                "attempted": False,
                                "ran": False,
                                "reason_codes": [],
                                "skipped_reason": "",
                                "runs_used": 1,
                                "next_allowed_round": 6,
                                "cooldown_remaining_rounds": 2,
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_autorun_history_enabled": True,
                "ops_stress_autorun_history_window_days": 3,
                "ops_stress_autorun_history_min_rounds": 1,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        hist = out.get("stress_autorun_history", {})
        self.assertTrue(bool(hist.get("active", False)))
        metrics = hist.get("metrics", {}) if isinstance(hist.get("metrics", {}), dict) else {}
        self.assertEqual(int(metrics.get("rounds_total", 0)), 4)
        self.assertEqual(int(metrics.get("triggered_rounds", 0)), 3)
        self.assertEqual(int(metrics.get("ran_rounds", 0)), 1)
        self.assertEqual(int(metrics.get("skipped_rounds", 0)), 2)
        self.assertEqual(int(metrics.get("cooldown_skip_rounds", 0)), 1)
        self.assertEqual(int(metrics.get("max_runs_skip_rounds", 0)), 1)
        self.assertAlmostEqual(float(metrics.get("trigger_density", 0.0)), 0.75, places=8)
        self.assertAlmostEqual(float(metrics.get("run_rate_when_triggered", 0.0)), (1.0 / 3.0), places=8)
        self.assertAlmostEqual(float(metrics.get("cooldown_efficiency", 0.0)), 0.5, places=8)
        artifacts = hist.get("artifacts", {}) if isinstance(hist.get("artifacts", {}), dict) else {}
        history_artifact = (
            artifacts.get("history", {}) if isinstance(artifacts.get("history", {}), dict) else {}
        )
        self.assertTrue(bool(history_artifact.get("written", False)))
        self.assertTrue(Path(str(history_artifact.get("json", ""))).exists())
        self.assertTrue(Path(str(history_artifact.get("md", ""))).exists())

    def test_gate_report_fails_on_stress_autorun_adaptive_saturation(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps(
                {
                    "date": "2026-02-12",
                    "passed": False,
                    "rounds": [
                        {
                            "round": 1,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": True,
                                "ran": True,
                                "reason_codes": ["stress_trend"],
                                "max_runs": 1,
                                "max_runs_base": 4,
                                "adaptive": {
                                    "reason": "high_density_throttle",
                                    "factor": 0.25,
                                    "trigger_density": 0.85,
                                },
                            },
                        },
                        {
                            "round": 2,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": False,
                                "ran": False,
                                "reason_codes": ["stress_trend"],
                                "skipped_reason": "max_runs_reached",
                                "max_runs": 1,
                                "max_runs_base": 4,
                                "adaptive": {
                                    "reason": "high_density_throttle",
                                    "factor": 0.25,
                                    "trigger_density": 0.85,
                                },
                            },
                        },
                        {
                            "round": 3,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": False,
                                "ran": False,
                                "reason_codes": ["stress_trend"],
                                "skipped_reason": "max_runs_reached",
                                "max_runs": 1,
                                "max_runs_base": 4,
                                "adaptive": {
                                    "reason": "high_density_throttle",
                                    "factor": 0.25,
                                    "trigger_density": 0.85,
                                },
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_autorun_adaptive_monitor_enabled": True,
                "ops_stress_autorun_adaptive_monitor_window_days": 3,
                "ops_stress_autorun_adaptive_monitor_min_rounds": 2,
                "ops_stress_autorun_adaptive_effective_base_ratio_floor": 0.80,
                "ops_stress_autorun_adaptive_effective_base_ratio_ceiling": 2.00,
                "ops_stress_autorun_adaptive_throttle_ratio_max": 0.60,
                "ops_stress_autorun_adaptive_expand_ratio_max": 0.90,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(out["passed"])
        self.assertFalse(bool(out["checks"]["stress_autorun_adaptive_ok"]))
        adaptive = out.get("stress_autorun_adaptive", {})
        self.assertTrue(bool(adaptive.get("active", False)))
        checks = adaptive.get("checks", {}) if isinstance(adaptive.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("effective_base_ratio_floor_ok", True)))
        self.assertFalse(bool(checks.get("throttle_ratio_ok", True)))
        alerts = adaptive.get("alerts", [])
        self.assertIn("stress_autorun_adaptive_ratio_low", set(alerts))
        self.assertIn("stress_autorun_adaptive_throttle_ratio_high", set(alerts))

    def test_ops_report_flags_stress_autorun_adaptive_saturation(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps(
                {
                    "date": "2026-02-12",
                    "passed": False,
                    "rounds": [
                        {
                            "round": 1,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": True,
                                "ran": True,
                                "reason_codes": ["stress_trend"],
                                "max_runs": 1,
                                "max_runs_base": 4,
                                "adaptive": {
                                    "reason": "high_density_throttle",
                                    "factor": 0.25,
                                    "trigger_density": 0.80,
                                },
                            },
                        },
                        {
                            "round": 2,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": False,
                                "ran": False,
                                "reason_codes": ["stress_trend"],
                                "skipped_reason": "max_runs_reached",
                                "max_runs": 1,
                                "max_runs_base": 4,
                                "adaptive": {
                                    "reason": "high_density_throttle",
                                    "factor": 0.25,
                                    "trigger_density": 0.80,
                                },
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_autorun_adaptive_monitor_enabled": True,
                "ops_stress_autorun_adaptive_monitor_window_days": 3,
                "ops_stress_autorun_adaptive_monitor_min_rounds": 2,
                "ops_stress_autorun_adaptive_effective_base_ratio_floor": 0.75,
                "ops_stress_autorun_adaptive_effective_base_ratio_ceiling": 2.00,
                "ops_stress_autorun_adaptive_throttle_ratio_max": 0.50,
                "ops_stress_autorun_adaptive_expand_ratio_max": 0.90,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        self.assertEqual(str(out.get("status", "")), "red")
        adaptive = out.get("stress_autorun_adaptive", {})
        self.assertTrue(bool(adaptive.get("active", False)))
        checks = adaptive.get("checks", {}) if isinstance(adaptive.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("effective_base_ratio_floor_ok", True)))
        self.assertFalse(bool(checks.get("throttle_ratio_ok", True)))

    def test_review_until_pass_defect_plan_includes_stress_autorun_adaptive_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps(
                {
                    "date": "2026-02-12",
                    "passed": False,
                    "rounds": [
                        {
                            "round": 1,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": True,
                                "ran": True,
                                "reason_codes": ["stress_trend"],
                                "max_runs": 1,
                                "max_runs_base": 4,
                                "adaptive": {
                                    "reason": "high_density_throttle",
                                    "factor": 0.25,
                                    "trigger_density": 0.85,
                                },
                            },
                        },
                        {
                            "round": 2,
                            "stress_matrix_autorun": {
                                "enabled": True,
                                "triggered": True,
                                "attempted": False,
                                "ran": False,
                                "reason_codes": ["stress_trend"],
                                "skipped_reason": "max_runs_reached",
                                "max_runs": 1,
                                "max_runs_base": 4,
                                "adaptive": {
                                    "reason": "high_density_throttle",
                                    "factor": 0.25,
                                    "trigger_density": 0.85,
                                },
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_autorun_adaptive_monitor_enabled": True,
                "ops_stress_autorun_adaptive_monitor_window_days": 3,
                "ops_stress_autorun_adaptive_monitor_min_rounds": 2,
                "ops_stress_autorun_adaptive_effective_base_ratio_floor": 0.80,
                "ops_stress_autorun_adaptive_effective_base_ratio_ceiling": 2.00,
                "ops_stress_autorun_adaptive_throttle_ratio_max": 0.50,
                "ops_stress_autorun_adaptive_expand_ratio_max": 0.90,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        plan_json = Path(str(out["rounds"][0]["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("STRESS_AUTORUN_ADAPTIVE_RATIO_LOW", codes)
        self.assertIn("STRESS_AUTORUN_ADAPTIVE_THROTTLE", codes)

    def test_gate_report_fails_on_stress_autorun_reason_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        rounds = []
        for i in range(3):
            rounds.append(
                {
                    "round": i + 1,
                    "stress_matrix_autorun": {
                        "enabled": True,
                        "triggered": True,
                        "attempted": True,
                        "ran": True,
                        "reason_codes": ["stress_trend"],
                        "max_runs": 1,
                        "max_runs_base": 2,
                        "adaptive": {"reason": "high_density_throttle", "factor": 0.5, "trigger_density": 0.80},
                    },
                }
            )
        for i in range(3):
            rounds.append(
                {
                    "round": i + 4,
                    "stress_matrix_autorun": {
                        "enabled": True,
                        "triggered": True,
                        "attempted": True,
                        "ran": True,
                        "reason_codes": ["stress_trend"],
                        "max_runs": 2,
                        "max_runs_base": 2,
                        "adaptive": {"reason": "low_density_expand", "factor": 1.0, "trigger_density": 0.20},
                    },
                }
            )
        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps({"date": "2026-02-12", "passed": False, "rounds": rounds}, ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_autorun_adaptive_monitor_enabled": False,
                "ops_stress_autorun_reason_drift_enabled": True,
                "ops_stress_autorun_reason_drift_window_days": 3,
                "ops_stress_autorun_reason_drift_min_rounds": 4,
                "ops_stress_autorun_reason_drift_recent_rounds": 3,
                "ops_stress_autorun_reason_drift_mix_gap_max": 0.20,
                "ops_stress_autorun_reason_drift_change_point_gap_max": 0.30,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(out["passed"])
        self.assertFalse(bool(out["checks"]["stress_autorun_reason_drift_ok"]))
        drift = out.get("stress_autorun_reason_drift", {})
        self.assertTrue(bool(drift.get("active", False)))
        checks = drift.get("checks", {}) if isinstance(drift.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("reason_mix_gap_ok", True)))
        self.assertFalse(bool(checks.get("change_point_gap_ok", True)))
        alerts = drift.get("alerts", [])
        self.assertIn("stress_autorun_reason_mix_drift", set(alerts))
        self.assertIn("stress_autorun_reason_change_point", set(alerts))
        artifacts = drift.get("artifacts", {}) if isinstance(drift.get("artifacts", {}), dict) else {}
        reason_artifact = (
            artifacts.get("reason_drift", {}) if isinstance(artifacts.get("reason_drift", {}), dict) else {}
        )
        self.assertTrue(bool(reason_artifact.get("written", False)))
        self.assertTrue(Path(str(reason_artifact.get("json", ""))).exists())
        self.assertTrue(Path(str(reason_artifact.get("md", ""))).exists())

    def test_ops_report_flags_stress_autorun_reason_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        rounds = []
        for i in range(4):
            rounds.append(
                {
                    "round": i + 1,
                    "stress_matrix_autorun": {
                        "enabled": True,
                        "triggered": True,
                        "attempted": True,
                        "ran": True,
                        "reason_codes": ["stress_trend"],
                        "max_runs": 1,
                        "max_runs_base": 2,
                        "adaptive": {"reason": "high_density_throttle", "factor": 0.5, "trigger_density": 0.75},
                    },
                }
            )
        for i in range(2):
            rounds.append(
                {
                    "round": i + 5,
                    "stress_matrix_autorun": {
                        "enabled": True,
                        "triggered": True,
                        "attempted": True,
                        "ran": True,
                        "reason_codes": ["stress_trend"],
                        "max_runs": 2,
                        "max_runs_base": 2,
                        "adaptive": {"reason": "low_density_expand", "factor": 1.0, "trigger_density": 0.20},
                    },
                }
            )
        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps({"date": "2026-02-12", "passed": False, "rounds": rounds}, ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_autorun_adaptive_monitor_enabled": False,
                "ops_stress_autorun_reason_drift_enabled": True,
                "ops_stress_autorun_reason_drift_window_days": 3,
                "ops_stress_autorun_reason_drift_min_rounds": 4,
                "ops_stress_autorun_reason_drift_recent_rounds": 2,
                "ops_stress_autorun_reason_drift_mix_gap_max": 0.15,
                "ops_stress_autorun_reason_drift_change_point_gap_max": 0.20,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        self.assertEqual(str(out.get("status", "")), "red")
        drift = out.get("stress_autorun_reason_drift", {})
        self.assertTrue(bool(drift.get("active", False)))
        checks = drift.get("checks", {}) if isinstance(drift.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("reason_mix_gap_ok", True)))
        self.assertFalse(bool(checks.get("change_point_gap_ok", True)))
        artifacts = drift.get("artifacts", {}) if isinstance(drift.get("artifacts", {}), dict) else {}
        reason_artifact = (
            artifacts.get("reason_drift", {}) if isinstance(artifacts.get("reason_drift", {}), dict) else {}
        )
        self.assertTrue(bool(reason_artifact.get("written", False)))
        self.assertTrue(Path(str(reason_artifact.get("json", ""))).exists())

    def test_review_until_pass_defect_plan_includes_stress_autorun_reason_drift_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        rounds = []
        for i in range(3):
            rounds.append(
                {
                    "round": i + 1,
                    "stress_matrix_autorun": {
                        "enabled": True,
                        "triggered": True,
                        "attempted": True,
                        "ran": True,
                        "reason_codes": ["stress_trend"],
                        "max_runs": 1,
                        "max_runs_base": 2,
                        "adaptive": {"reason": "high_density_throttle", "factor": 0.5, "trigger_density": 0.80},
                    },
                }
            )
        for i in range(3):
            rounds.append(
                {
                    "round": i + 4,
                    "stress_matrix_autorun": {
                        "enabled": True,
                        "triggered": True,
                        "attempted": True,
                        "ran": True,
                        "reason_codes": ["stress_trend"],
                        "max_runs": 2,
                        "max_runs_base": 2,
                        "adaptive": {"reason": "low_density_expand", "factor": 1.0, "trigger_density": 0.20},
                    },
                }
            )
        (artifacts_dir / "release_ready_2026-02-12.json").write_text(
            json.dumps({"date": "2026-02-12", "passed": False, "rounds": rounds}, ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_autorun_adaptive_monitor_enabled": False,
                "ops_stress_autorun_reason_drift_enabled": True,
                "ops_stress_autorun_reason_drift_window_days": 3,
                "ops_stress_autorun_reason_drift_min_rounds": 4,
                "ops_stress_autorun_reason_drift_recent_rounds": 3,
                "ops_stress_autorun_reason_drift_mix_gap_max": 0.20,
                "ops_stress_autorun_reason_drift_change_point_gap_max": 0.30,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        plan_json = Path(str(out["rounds"][0]["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("STRESS_AUTORUN_REASON_MIX", codes)
        self.assertIn("STRESS_AUTORUN_REASON_CHANGE_POINT", codes)

    def test_gate_report_fails_on_temporal_audit_leak(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _manifest_payload(created_at: str, cutoff_date: str, bar_max: str, leak: bool) -> dict[str, object]:
            leak_bar = "2026-02-14T09:31:00" if leak else bar_max
            return {
                "run_type": "strategy_lab",
                "run_id": created_at.replace("-", "").replace(":", "").replace("T", "_"),
                "created_at": created_at,
                "checks": {"strict_cutoff_enforced": True},
                "metadata": {
                    "cutoff_date": cutoff_date,
                    "cutoff_ts": f"{cutoff_date}T23:59:59",
                    "bar_max_ts": leak_bar,
                    "news_max_ts": f"{cutoff_date}T23:59:59",
                    "report_max_ts": f"{cutoff_date}T23:59:59",
                },
            }

        (manifest_dir / "strategy_lab_20260211_000000.json").write_text(
            json.dumps(_manifest_payload("2026-02-11T10:00:00", "2026-02-11", "2026-02-11T15:00:00", False), ensure_ascii=False),
            encoding="utf-8",
        )
        (manifest_dir / "strategy_lab_20260212_000000.json").write_text(
            json.dumps(_manifest_payload("2026-02-12T10:00:00", "2026-02-12", "2026-02-12T15:00:00", False), ensure_ascii=False),
            encoding="utf-8",
        )
        (manifest_dir / "strategy_lab_20260213_000000.json").write_text(
            json.dumps(_manifest_payload("2026-02-13T10:00:00", "2026-02-13", "2026-02-13T15:00:00", True), ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_temporal_audit_enabled": True,
                "ops_temporal_audit_lookback_days": 10,
                "ops_temporal_audit_min_samples": 3,
                "ops_temporal_audit_missing_ratio_max": 0.50,
                "ops_temporal_audit_leak_ratio_max": 0.00,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"]["temporal_audit_ok"]))
        temporal = out.get("temporal_audit", {})
        self.assertTrue(bool(temporal.get("active", False)))
        self.assertIn("temporal_audit_leak_detected", set(temporal.get("alerts", [])))
        self.assertFalse(out["passed"])

    def test_gate_report_temporal_audit_autofix_from_summary(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        manifest_dir = td / "artifacts" / "manifests"
        research_dir = td / "research" / "strategy_lab_20260213_010101"
        review_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        research_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        summary_path = research_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "cutoff_date": "2026-02-13",
                    "cutoff_ts": "2026-02-13T23:59:59",
                    "bar_max_ts": "2026-02-13T15:00:00",
                    "news_max_ts": "2026-02-13T22:00:00",
                    "report_max_ts": "2026-02-13T21:00:00",
                    "data_fetch_stats": {
                        "strict_cutoff_enforced": True,
                        "cutoff_date": "2026-02-13",
                        "cutoff_ts": "2026-02-13T23:59:59",
                        "bar_max_ts": "2026-02-13T15:00:00",
                        "news_max_ts": "2026-02-13T22:00:00",
                        "report_max_ts": "2026-02-13T21:00:00",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest_path = manifest_dir / "strategy_lab_20260213_000000.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_type": "strategy_lab",
                    "run_id": "20260213_000000",
                    "created_at": "2026-02-13T10:00:00",
                    "artifacts": {"summary": str(summary_path)},
                    "checks": {"strict_cutoff_enforced": False},
                    "metadata": {
                        "cutoff_date": "2026-02-13",
                        "cutoff_ts": "",
                        "bar_max_ts": "",
                        "news_max_ts": "",
                        "report_max_ts": "",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_temporal_audit_enabled": True,
                "ops_temporal_audit_lookback_days": 10,
                "ops_temporal_audit_min_samples": 1,
                "ops_temporal_audit_missing_ratio_max": 0.00,
                "ops_temporal_audit_leak_ratio_max": 0.00,
                "ops_temporal_audit_autofix_enabled": True,
                "ops_temporal_audit_autofix_max_writes": 3,
                "ops_temporal_audit_autofix_fix_strict_cutoff": True,
                "ops_temporal_audit_autofix_require_safe": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(bool(out["checks"]["temporal_audit_ok"]))
        temporal = out.get("temporal_audit", {})
        metrics = temporal.get("metrics", {}) if isinstance(temporal.get("metrics", {}), dict) else {}
        self.assertGreaterEqual(int(metrics.get("autofix_applied_count", 0)), 1)
        self.assertTrue(bool(metrics.get("autofix_artifact_written", False)))
        self.assertEqual(int(metrics.get("missing_count", 0)), 0)
        temporal_artifacts = temporal.get("artifacts", {}) if isinstance(temporal.get("artifacts", {}), dict) else {}
        autofix_artifact = (
            temporal_artifacts.get("autofix_patch", {})
            if isinstance(temporal_artifacts.get("autofix_patch", {}), dict)
            else {}
        )
        self.assertTrue(bool(autofix_artifact.get("written", False)))
        artifact_json = Path(str(autofix_artifact.get("json", "")))
        artifact_md = Path(str(autofix_artifact.get("md", "")))
        self.assertTrue(artifact_json.exists())
        self.assertTrue(artifact_md.exists())
        artifact_payload = json.loads(artifact_json.read_text(encoding="utf-8"))
        self.assertGreaterEqual(int(artifact_payload.get("applied_count", 0)), 1)
        self.assertTrue(list(artifact_payload.get("events", [])))
        first_event = artifact_payload.get("events", [])[0] if artifact_payload.get("events", []) else {}
        self.assertEqual(str(first_event.get("manifest_path", "")), str(manifest_path))
        patch_delta = first_event.get("patch_delta", {}) if isinstance(first_event.get("patch_delta", {}), dict) else {}
        self.assertIn("bar_max_ts", patch_delta)
        self.assertEqual(str((patch_delta.get("bar_max_ts", {}) or {}).get("source", "")), "summary")

        fixed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fixed_meta = fixed_manifest.get("metadata", {}) if isinstance(fixed_manifest.get("metadata", {}), dict) else {}
        fixed_checks = fixed_manifest.get("checks", {}) if isinstance(fixed_manifest.get("checks", {}), dict) else {}
        self.assertEqual(str(fixed_meta.get("cutoff_ts", "")), "2026-02-13T23:59:59")
        self.assertEqual(str(fixed_meta.get("bar_max_ts", "")), "2026-02-13T15:00:00")
        self.assertEqual(str(fixed_meta.get("news_max_ts", "")), "2026-02-13T22:00:00")
        self.assertEqual(str(fixed_meta.get("report_max_ts", "")), "2026-02-13T21:00:00")
        self.assertTrue(bool(fixed_checks.get("strict_cutoff_enforced", False)))

    def test_gate_report_temporal_audit_autofix_skips_unsafe_candidate(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        manifest_dir = td / "artifacts" / "manifests"
        research_dir = td / "research" / "strategy_lab_20260213_010101"
        review_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        research_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        summary_path = research_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "cutoff_date": "2026-02-13",
                    "cutoff_ts": "2026-02-13T23:59:59",
                    "bar_max_ts": "2026-02-14T09:31:00",
                    "news_max_ts": "2026-02-13T22:00:00",
                    "report_max_ts": "2026-02-13T21:00:00",
                    "data_fetch_stats": {
                        "strict_cutoff_enforced": True,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest_path = manifest_dir / "strategy_lab_20260213_000000.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_type": "strategy_lab",
                    "run_id": "20260213_000000",
                    "created_at": "2026-02-13T10:00:00",
                    "artifacts": {"summary": str(summary_path)},
                    "checks": {"strict_cutoff_enforced": False},
                    "metadata": {
                        "cutoff_date": "2026-02-13",
                        "cutoff_ts": "",
                        "bar_max_ts": "",
                        "news_max_ts": "",
                        "report_max_ts": "",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_temporal_audit_enabled": True,
                "ops_temporal_audit_lookback_days": 10,
                "ops_temporal_audit_min_samples": 1,
                "ops_temporal_audit_missing_ratio_max": 0.00,
                "ops_temporal_audit_leak_ratio_max": 0.00,
                "ops_temporal_audit_autofix_enabled": True,
                "ops_temporal_audit_autofix_max_writes": 3,
                "ops_temporal_audit_autofix_fix_strict_cutoff": True,
                "ops_temporal_audit_autofix_require_safe": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"]["temporal_audit_ok"]))
        temporal = out.get("temporal_audit", {})
        metrics = temporal.get("metrics", {}) if isinstance(temporal.get("metrics", {}), dict) else {}
        self.assertEqual(int(metrics.get("autofix_applied_count", 0)), 0)
        self.assertGreaterEqual(int(metrics.get("autofix_skipped_count", 0)), 1)
        self.assertTrue(bool(metrics.get("autofix_artifact_written", False)))
        series = temporal.get("series", [])
        self.assertTrue(series)
        autofix = series[-1].get("autofix", {}) if isinstance(series[-1], dict) else {}
        self.assertEqual(str(autofix.get("reason", "")), "unsafe_temporal_candidate")
        temporal_artifacts = temporal.get("artifacts", {}) if isinstance(temporal.get("artifacts", {}), dict) else {}
        autofix_artifact = (
            temporal_artifacts.get("autofix_patch", {})
            if isinstance(temporal_artifacts.get("autofix_patch", {}), dict)
            else {}
        )
        artifact_json = Path(str(autofix_artifact.get("json", "")))
        self.assertTrue(artifact_json.exists())
        artifact_payload = json.loads(artifact_json.read_text(encoding="utf-8"))
        self.assertGreaterEqual(int(artifact_payload.get("skipped_count", 0)), 1)
        top_reasons = [str(x.get("reason", "")) for x in artifact_payload.get("top_reasons", []) if isinstance(x, dict)]
        self.assertIn("unsafe_temporal_candidate", set(top_reasons))

        latest_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        latest_meta = latest_manifest.get("metadata", {}) if isinstance(latest_manifest.get("metadata", {}), dict) else {}
        self.assertEqual(str(latest_meta.get("bar_max_ts", "")), "")

    def test_gate_report_temporal_audit_autofix_rotation_and_checksum_index(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        manifest_dir = td / "artifacts" / "manifests"
        research_dir = td / "research" / "strategy_lab_20260213_010101"
        review_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        research_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        stale_json = review_dir / "2026-02-10_temporal_autofix_patch.json"
        stale_md = review_dir / "2026-02-10_temporal_autofix_patch.md"
        stale_json.write_text("{}", encoding="utf-8")
        stale_md.write_text("# stale\n", encoding="utf-8")

        summary_path = research_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "cutoff_date": "2026-02-13",
                    "cutoff_ts": "2026-02-13T23:59:59",
                    "bar_max_ts": "2026-02-13T15:00:00",
                    "news_max_ts": "2026-02-13T22:00:00",
                    "report_max_ts": "2026-02-13T21:00:00",
                    "data_fetch_stats": {
                        "strict_cutoff_enforced": True,
                        "cutoff_date": "2026-02-13",
                        "cutoff_ts": "2026-02-13T23:59:59",
                        "bar_max_ts": "2026-02-13T15:00:00",
                        "news_max_ts": "2026-02-13T22:00:00",
                        "report_max_ts": "2026-02-13T21:00:00",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        manifest_path = manifest_dir / "strategy_lab_20260213_000000.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_type": "strategy_lab",
                    "run_id": "20260213_000000",
                    "created_at": "2026-02-13T10:00:00",
                    "artifacts": {"summary": str(summary_path)},
                    "checks": {"strict_cutoff_enforced": False},
                    "metadata": {
                        "cutoff_date": "2026-02-13",
                        "cutoff_ts": "",
                        "bar_max_ts": "",
                        "news_max_ts": "",
                        "report_max_ts": "",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_temporal_audit_enabled": True,
                "ops_temporal_audit_lookback_days": 10,
                "ops_temporal_audit_min_samples": 1,
                "ops_temporal_audit_missing_ratio_max": 0.00,
                "ops_temporal_audit_leak_ratio_max": 0.00,
                "ops_temporal_audit_autofix_enabled": True,
                "ops_temporal_audit_autofix_max_writes": 3,
                "ops_temporal_audit_autofix_fix_strict_cutoff": True,
                "ops_temporal_audit_autofix_require_safe": True,
                "ops_temporal_audit_autofix_patch_retention_days": 2,
                "ops_temporal_audit_autofix_patch_checksum_index_enabled": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        temporal = out.get("temporal_audit", {}) if isinstance(out.get("temporal_audit", {}), dict) else {}
        temporal_artifacts = temporal.get("artifacts", {}) if isinstance(temporal.get("artifacts", {}), dict) else {}
        autofix_artifact = (
            temporal_artifacts.get("autofix_patch", {})
            if isinstance(temporal_artifacts.get("autofix_patch", {}), dict)
            else {}
        )
        self.assertTrue(bool(autofix_artifact.get("written", False)))
        self.assertEqual(int(autofix_artifact.get("retention_days", 0)), 2)
        self.assertGreaterEqual(int(autofix_artifact.get("rotated_out_count", 0)), 1)
        self.assertIn("2026-02-10", set(autofix_artifact.get("rotated_out_dates", [])))
        self.assertFalse(stale_json.exists())
        self.assertFalse(stale_md.exists())

        self.assertTrue(bool(autofix_artifact.get("checksum_index_enabled", False)))
        self.assertTrue(bool(autofix_artifact.get("checksum_index_written", False)))
        index_path = Path(str(autofix_artifact.get("checksum_index_path", "")))
        self.assertTrue(index_path.exists())
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(int(index_payload.get("retention_days", 0)), 2)
        self.assertGreaterEqual(int(index_payload.get("entry_count", 0)), 1)
        entries = index_payload.get("entries", [])
        self.assertTrue(entries)
        first = entries[0] if isinstance(entries[0], dict) else {}
        self.assertEqual(len(str(first.get("json_sha256", ""))), 64)
        self.assertEqual(len(str(first.get("md_sha256", ""))), 64)

    def test_review_until_pass_defect_plan_includes_temporal_audit_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _manifest_payload(created_at: str, cutoff_date: str, strict: bool, leak: bool) -> dict[str, object]:
            return {
                "run_type": "research_backtest",
                "run_id": created_at.replace("-", "").replace(":", "").replace("T", "_"),
                "created_at": created_at,
                "checks": {"strict_cutoff_enforced": strict},
                "metadata": {
                    "cutoff_date": cutoff_date,
                    "cutoff_ts": f"{cutoff_date}T23:59:59",
                    "bar_max_ts": ("2026-02-14T09:31:00" if leak else f"{cutoff_date}T15:00:00"),
                    "news_max_ts": f"{cutoff_date}T23:59:59",
                    "report_max_ts": f"{cutoff_date}T23:59:59",
                },
            }

        (manifest_dir / "research_backtest_20260211_000000.json").write_text(
            json.dumps(_manifest_payload("2026-02-11T10:00:00", "2026-02-11", True, False), ensure_ascii=False),
            encoding="utf-8",
        )
        (manifest_dir / "research_backtest_20260212_000000.json").write_text(
            json.dumps(_manifest_payload("2026-02-12T10:00:00", "2026-02-12", False, False), ensure_ascii=False),
            encoding="utf-8",
        )
        (manifest_dir / "research_backtest_20260213_000000.json").write_text(
            json.dumps(_manifest_payload("2026-02-13T10:00:00", "2026-02-13", True, True), ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_temporal_audit_enabled": True,
                "ops_temporal_audit_lookback_days": 10,
                "ops_temporal_audit_min_samples": 3,
                "ops_temporal_audit_missing_ratio_max": 0.50,
                "ops_temporal_audit_leak_ratio_max": 0.00,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        plan_json = Path(str(out["rounds"][0]["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("TEMPORAL_AUDIT", codes)
        self.assertIn("TEMPORAL_AUDIT_LEAK", codes)
        self.assertIn("TEMPORAL_AUDIT_STRICT", codes)

    def test_review_until_pass_defect_plan_includes_mode_drift_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        daily_dir = td / "daily"
        review_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        feedback = {
            "runtime_mode": "swing",
            "history": {
                "modes": {
                    "swing": {
                        "samples": 8,
                        "avg_win_rate": 0.60,
                        "avg_profit_factor": 1.8,
                    }
                }
            },
        }
        (daily_dir / f"{d.isoformat()}_mode_feedback.json").write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE executed_plans (date TEXT, runtime_mode TEXT, mode TEXT, pnl REAL, direction TEXT)"
            )
            rows = []
            for i in range(10):
                pnl = 0.8 if i < 2 else -1.0
                rows.append((d.isoformat(), "swing", "swing", pnl, "LONG"))
            conn.executemany(
                "INSERT INTO executed_plans (date, runtime_mode, mode, pnl, direction) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "mode_drift_window_days": 30,
                "mode_drift_min_live_trades": 8,
                "mode_drift_win_rate_max_gap": 0.12,
                "mode_drift_profit_factor_max_gap": 0.40,
                "mode_drift_focus_runtime_mode_only": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 1, "stderr": "test_x ... FAIL", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        round0 = out["rounds"][0]
        self.assertTrue(bool(round0.get("mode_drift_active", False)))
        self.assertFalse(bool(round0.get("mode_drift_ok", True)))
        plan_json = Path(str(round0["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("MODE_DRIFT_WIN_RATE", codes)
        self.assertIn("MODE_DRIFT_PROFIT_FACTOR", codes)
        next_actions = [str(x) for x in plan.get("next_actions", [])]
        self.assertTrue(next_actions)
        self.assertIn("mode_drift", next_actions[0])

    def test_gate_report_fails_on_reconcile_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        daily_dir = td / "daily"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps(
                    {
                        "quality": {"passed": True, "flags": [], "source_confidence_score": 0.90},
                        "risk_multiplier": 1.0,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            intraday = {
                "quality_flags": [],
                "source_confidence_score": 0.90,
                "risk_multiplier": 1.0,
            }
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {
                            "plans": 2,
                            "closed_trades": 1,
                            "closed_pnl": 0.05,
                            "open_positions": 2,
                            "risk_multiplier": 1.0,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text(
                "symbol,side,size_pct,status\nA,LONG,10,ACTIVE\nB,LONG,12,ACTIVE\n",
                encoding="utf-8",
            )

        (td / "artifacts" / "paper_positions_open.json").write_text(
            json.dumps({"as_of": d.isoformat(), "positions": [{"symbol": "A"}, {"symbol": "B"}]}, ensure_ascii=False),
            encoding="utf-8",
        )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute("CREATE TABLE executed_plans (date TEXT, symbol TEXT, pnl REAL, status TEXT)")
            for i in range(3):
                day = date.fromordinal(d.toordinal() - i).isoformat()
                conn.execute(
                    "INSERT INTO latest_positions (date, symbol, size_pct, status) VALUES (?, ?, ?, ?)",
                    (day, "A", 10.0, "ACTIVE"),
                )
                conn.execute(
                    "INSERT INTO executed_plans (date, symbol, pnl, status) VALUES (?, ?, ?, ?)",
                    (day, "A", 0.05, "CLOSED"),
                )
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_reconcile_window_days": 3,
                "ops_reconcile_min_samples": 3,
                "ops_reconcile_missing_ratio_max": 1.0,
                "ops_reconcile_plan_gap_ratio_max": 0.2,
                "ops_reconcile_closed_count_gap_ratio_max": 1.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 1.0,
                "ops_reconcile_open_gap_ratio_max": 1.0,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"]["reconcile_drift_ok"]))
        self.assertTrue(bool(out["reconcile_drift"]["active"]))
        self.assertIn("reconcile_plan_count_gap_high", set(out["reconcile_drift"]["alerts"]))

    def test_gate_report_emits_hard_rollback_recommendation(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        anchor = artifacts_dir / "params_live_backup_2026-02-12.yaml"
        anchor.write_text("signal_confidence_min: 60\n", encoding="utf-8")

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.30, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        rollback = out.get("rollback_recommendation", {})
        self.assertEqual(str(rollback.get("level", "")), "hard")
        self.assertTrue(bool(rollback.get("anchor_ready", False)))
        self.assertEqual(str(rollback.get("target_anchor", "")), str(anchor))
        self.assertIn("max_drawdown", set(rollback.get("reason_codes", [])))

    def test_review_until_pass_defect_plan_includes_hard_rollback_actions(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (artifacts_dir / "params_live_backup_2026-02-12.yaml").write_text(
            "signal_confidence_min: 60\n",
            encoding="utf-8",
        )

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.30, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 1, "stderr": "test_x ... FAIL", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        round0 = out["rounds"][0]
        self.assertEqual(str(round0.get("rollback_level", "")), "hard")
        self.assertTrue(bool(round0.get("rollback_anchor_ready", False)))
        plan_json = Path(str(round0["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("ROLLBACK_RECOMMENDED", codes)
        self.assertIn("ROLLBACK_HARD", codes)
        next_actions = [str(x) for x in plan.get("next_actions", [])]
        self.assertTrue(next_actions)
        self.assertIn("回滚", next_actions[0])

    def test_gate_report_slot_eod_quality_risk_split(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps(
                    {"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": False},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 0,
                            "risk_multiplier": 1.0,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_slot_eod_quality_anomaly_ratio_max": 0.2,
                "ops_slot_eod_risk_anomaly_ratio_max": 1.0,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        slot = out.get("slot_anomaly", {})
        checks = slot.get("checks", {})
        self.assertTrue(bool(slot.get("active", False)))
        self.assertFalse(bool(checks.get("eod_quality_anomaly_ok", True)))
        self.assertTrue(bool(checks.get("eod_risk_anomaly_ok", False)))
        self.assertFalse(bool(out["checks"]["slot_anomaly_ok"]))

    def test_gate_report_slot_eod_regime_bucket_threshold(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        regimes = ["极端波动", "强趋势", "震荡"]
        quality_flags = [False, True, True]
        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps(
                    {"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": quality_flags[i]},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 0,
                            "risk_multiplier": 1.0,
                            "regime": regimes[i],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_slot_eod_quality_anomaly_ratio_max": 1.0,
                "ops_slot_eod_risk_anomaly_ratio_max": 1.0,
                "ops_slot_eod_quality_anomaly_ratio_max_by_regime": {
                    "trend": 1.0,
                    "range": 1.0,
                    "extreme_vol": 0.20,
                },
                "ops_slot_eod_risk_anomaly_ratio_max_by_regime": {
                    "trend": 1.0,
                    "range": 1.0,
                    "extreme_vol": 1.0,
                },
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        slot = out.get("slot_anomaly", {})
        checks = slot.get("checks", {})
        self.assertTrue(bool(slot.get("active", False)))
        self.assertTrue(bool(checks.get("eod_quality_anomaly_ok", False)))
        self.assertFalse(bool(checks.get("eod_quality_regime_bucket_ok", True)))
        self.assertFalse(bool(out["checks"]["slot_anomaly_ok"]))

    def test_gate_report_slot_uses_live_regime_threshold_overrides(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(2):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps(
                    {"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": False if i == 0 else True},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 0,
                            "risk_multiplier": 1.0,
                            "regime": "震荡",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        live_threshold_path = td / "artifacts" / "slot_regime_thresholds_live.yaml"
        live_threshold_path.parent.mkdir(parents=True, exist_ok=True)
        live_threshold_path.write_text(
            yaml.safe_dump(
                {
                    "as_of": d.isoformat(),
                    "ops_slot_eod_quality_anomaly_ratio_max_by_regime": {
                        "trend": 1.0,
                        "range": 0.80,
                        "extreme_vol": 1.0,
                    },
                    "ops_slot_eod_risk_anomaly_ratio_max_by_regime": {
                        "trend": 1.0,
                        "range": 1.0,
                        "extreme_vol": 1.0,
                    },
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 2,
                "ops_slot_min_samples": 2,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_slot_eod_quality_anomaly_ratio_max": 1.0,
                "ops_slot_eod_risk_anomaly_ratio_max": 1.0,
                "ops_slot_eod_quality_anomaly_ratio_max_by_regime": {
                    "trend": 1.0,
                    "range": 0.20,
                    "extreme_vol": 1.0,
                },
                "ops_slot_eod_risk_anomaly_ratio_max_by_regime": {
                    "trend": 1.0,
                    "range": 1.0,
                    "extreme_vol": 1.0,
                },
                "ops_slot_use_live_regime_thresholds": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        slot = out.get("slot_anomaly", {})
        checks = slot.get("checks", {})
        thresholds = slot.get("thresholds", {})
        quality_by_regime = thresholds.get("ops_slot_eod_quality_anomaly_ratio_max_by_regime", {})
        self.assertTrue(bool(slot.get("active", False)))
        self.assertTrue(bool(checks.get("eod_quality_regime_bucket_ok", False)))
        self.assertAlmostEqual(float(quality_by_regime.get("range", 0.0)), 0.80, places=6)
        self.assertTrue(bool(thresholds.get("live_regime_thresholds_applied", False)))

    def test_gate_report_reconcile_broker_snapshot_required_missing(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        daily_dir = td / "daily"
        manifest_dir = td / "artifacts" / "manifests"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps({"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0}),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 0,
                            "risk_multiplier": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text("symbol,side,size_pct,status\n", encoding="utf-8")

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute("CREATE TABLE executed_plans (date TEXT, symbol TEXT, pnl REAL, status TEXT)")
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_reconcile_window_days": 3,
                "ops_reconcile_min_samples": 3,
                "ops_reconcile_missing_ratio_max": 1.0,
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 1.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 1.0,
                "ops_reconcile_open_gap_ratio_max": 1.0,
                "ops_reconcile_require_broker_snapshot": True,
                "ops_reconcile_broker_missing_ratio_max": 0.20,
                "ops_reconcile_broker_gap_ratio_max": 1.0,
                "ops_reconcile_broker_pnl_gap_abs_max": 1.0,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        recon = out.get("reconcile_drift", {})
        self.assertTrue(bool(recon.get("active", False)))
        self.assertFalse(bool(recon.get("checks", {}).get("broker_missing_ratio_ok", True)))
        self.assertFalse(bool(out["checks"]["reconcile_drift_ok"]))
        self.assertIn("reconcile_broker_missing_ratio_high", set(recon.get("alerts", [])))

    def test_gate_report_reconcile_broker_contract_schema_invalid(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        daily_dir = td / "daily"
        manifest_dir = td / "artifacts" / "manifests"
        broker_dir = td / "artifacts" / "broker_snapshot"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        broker_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps({"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0}),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 0,
                            "risk_multiplier": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text("symbol,side,size_pct,status\n", encoding="utf-8")
            (broker_dir / f"{dstr}.json").write_text(
                json.dumps(
                    {
                        "source": "",
                        "open_positions": 0,
                        "closed_count": 0,
                        "closed_pnl": 0.0,
                        "positions": {"symbol": "rb2405"},
                    }
                ),
                encoding="utf-8",
            )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute("CREATE TABLE executed_plans (date TEXT, symbol TEXT, pnl REAL, status TEXT)")
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_reconcile_window_days": 3,
                "ops_reconcile_min_samples": 3,
                "ops_reconcile_missing_ratio_max": 1.0,
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 1.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 1.0,
                "ops_reconcile_open_gap_ratio_max": 1.0,
                "ops_reconcile_broker_missing_ratio_max": 1.0,
                "ops_reconcile_broker_gap_ratio_max": 1.0,
                "ops_reconcile_broker_pnl_gap_abs_max": 1.0,
                "ops_reconcile_broker_contract_schema_invalid_ratio_max": 0.20,
                "ops_reconcile_broker_contract_numeric_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": 1.0,
                "ops_reconcile_require_broker_snapshot": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        recon = out.get("reconcile_drift", {})
        self.assertTrue(bool(recon.get("active", False)))
        self.assertFalse(bool(recon.get("checks", {}).get("broker_contract_schema_ok", True)))
        self.assertFalse(bool(out["checks"]["reconcile_drift_ok"]))
        self.assertIn("reconcile_broker_contract_schema_invalid", set(recon.get("alerts", [])))

    def test_gate_report_writes_broker_canonical_view_on_lint_pass(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        daily_dir = td / "daily"
        manifest_dir = td / "artifacts" / "manifests"
        broker_dir = td / "artifacts" / "broker_snapshot"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        broker_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps({"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0}),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 1,
                            "risk_multiplier": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text("symbol,side,size_pct,status\n", encoding="utf-8")
            (broker_dir / f"{dstr}.json").write_text(
                json.dumps(
                    {
                        "source": "mock_live",
                        "open_positions": 1,
                        "closed_count": 0,
                        "closed_pnl": 0.0,
                        "positions": [
                            {
                                "symbol": "rb/2405",
                                "side": "sell",
                                "qty": "-2",
                                "entry_price": "3500",
                                "market_price": "3490",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute("CREATE TABLE executed_plans (date TEXT, symbol TEXT, pnl REAL, status TEXT)")
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_reconcile_window_days": 3,
                "ops_reconcile_min_samples": 3,
                "ops_reconcile_missing_ratio_max": 1.0,
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 1.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 1.0,
                "ops_reconcile_open_gap_ratio_max": 1.0,
                "ops_reconcile_broker_missing_ratio_max": 1.0,
                "ops_reconcile_broker_gap_ratio_max": 1.0,
                "ops_reconcile_broker_pnl_gap_abs_max": 1.0,
                "ops_reconcile_broker_contract_schema_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_numeric_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": 1.0,
                "ops_reconcile_require_broker_snapshot": True,
                "ops_reconcile_broker_contract_emit_canonical_view": True,
                "ops_reconcile_broker_contract_canonical_dir": "artifacts/broker_snapshot_canonical",
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        recon = out.get("reconcile_drift", {})
        self.assertTrue(bool(recon.get("active", False)))
        self.assertTrue(bool(recon.get("checks", {}).get("broker_contract_canonical_view_ok", False)))
        self.assertGreaterEqual(int(recon.get("metrics", {}).get("broker_canonical_written_days", 0)), 1)

        canonical_path = td / "artifacts" / "broker_snapshot_canonical" / f"{d.isoformat()}.json"
        self.assertTrue(canonical_path.exists())
        canonical_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
        positions = canonical_payload.get("positions", [])
        self.assertEqual(len(positions), 1)
        row = positions[0]
        self.assertEqual(str(row.get("symbol", "")), "RB2405")
        self.assertEqual(str(row.get("side", "")), "SHORT")
        self.assertAlmostEqual(float(row.get("qty", 0.0)), 2.0, places=6)
        self.assertEqual(str(row.get("raw_symbol", "")), "rb/2405")
        self.assertEqual(str(row.get("raw_side", "")), "sell")

    def test_gate_report_reconcile_broker_canonical_view_write_failed(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        daily_dir = td / "daily"
        manifest_dir = td / "artifacts" / "manifests"
        broker_dir = td / "artifacts" / "broker_snapshot"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        broker_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps({"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0}),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 1,
                            "risk_multiplier": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text("symbol,side,size_pct,status\n", encoding="utf-8")
            (broker_dir / f"{dstr}.json").write_text(
                json.dumps(
                    {
                        "source": "mock_live",
                        "open_positions": 1,
                        "closed_count": 0,
                        "closed_pnl": 0.0,
                        "positions": [
                            {
                                "symbol": "rb/2405",
                                "side": "sell",
                                "qty": "-2",
                                "entry_price": "3500",
                                "market_price": "3490",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute("CREATE TABLE executed_plans (date TEXT, symbol TEXT, pnl REAL, status TEXT)")
            conn.commit()

        blocker = td / "artifacts" / "canonical_blocker"
        blocker.parent.mkdir(parents=True, exist_ok=True)
        blocker.write_text("block", encoding="utf-8")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_reconcile_window_days": 3,
                "ops_reconcile_min_samples": 3,
                "ops_reconcile_missing_ratio_max": 1.0,
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 1.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 1.0,
                "ops_reconcile_open_gap_ratio_max": 1.0,
                "ops_reconcile_broker_missing_ratio_max": 1.0,
                "ops_reconcile_broker_gap_ratio_max": 1.0,
                "ops_reconcile_broker_pnl_gap_abs_max": 1.0,
                "ops_reconcile_broker_contract_schema_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_numeric_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": 1.0,
                "ops_reconcile_require_broker_snapshot": True,
                "ops_reconcile_broker_contract_emit_canonical_view": True,
                "ops_reconcile_broker_contract_canonical_dir": "artifacts/canonical_blocker",
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        recon = out.get("reconcile_drift", {})
        self.assertTrue(bool(recon.get("active", False)))
        self.assertFalse(bool(recon.get("checks", {}).get("broker_contract_canonical_view_ok", True)))
        self.assertFalse(bool(out["checks"]["reconcile_drift_ok"]))
        self.assertIn("reconcile_broker_contract_canonical_view_failed", set(recon.get("alerts", [])))
        self.assertGreaterEqual(int(recon.get("metrics", {}).get("broker_canonical_write_fail_days", 0)), 1)

    def test_gate_report_reconcile_broker_row_diff_high(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        daily_dir = td / "daily"
        manifest_dir = td / "artifacts" / "manifests"
        broker_dir = td / "artifacts" / "broker_snapshot"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        broker_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps({"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0}),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 1,
                            "risk_multiplier": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text("symbol,side,size_pct,status\n", encoding="utf-8")
            (broker_dir / f"{dstr}.json").write_text(
                json.dumps(
                    {
                        "source": "mock_live",
                        "open_positions": 1,
                        "closed_count": 0,
                        "closed_pnl": 0.0,
                        "positions": [
                            {
                                "symbol": "rb/2405",
                                "side": "sell",
                                "qty": "-2",
                                "entry_price": "3500",
                                "market_price": "3490",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

        (td / "artifacts" / "paper_positions_open.json").write_text(
            json.dumps(
                {
                    "as_of": d.isoformat(),
                    "positions": [
                        {
                            "open_date": d.isoformat(),
                            "symbol": "AU2406",
                            "side": "LONG",
                            "size_pct": 5.0,
                            "status": "OPEN",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute("CREATE TABLE executed_plans (date TEXT, symbol TEXT, pnl REAL, status TEXT)")
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_reconcile_window_days": 3,
                "ops_reconcile_min_samples": 3,
                "ops_reconcile_missing_ratio_max": 1.0,
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 1.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 1.0,
                "ops_reconcile_open_gap_ratio_max": 1.0,
                "ops_reconcile_broker_missing_ratio_max": 1.0,
                "ops_reconcile_broker_gap_ratio_max": 1.0,
                "ops_reconcile_broker_pnl_gap_abs_max": 1.0,
                "ops_reconcile_broker_contract_schema_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_numeric_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": 1.0,
                "ops_reconcile_broker_row_diff_min_samples": 1,
                "ops_reconcile_broker_row_diff_breach_ratio_max": 0.0,
                "ops_reconcile_broker_row_diff_key_mismatch_max": 0.0,
                "ops_reconcile_broker_row_diff_count_gap_max": 0.0,
                "ops_reconcile_broker_row_diff_notional_gap_max": 1.0,
                "ops_reconcile_broker_row_diff_asof_only": True,
                "ops_reconcile_require_broker_snapshot": True,
                "ops_reconcile_broker_contract_emit_canonical_view": True,
                "ops_reconcile_broker_contract_canonical_dir": "artifacts/broker_snapshot_canonical",
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        recon = out.get("reconcile_drift", {})
        self.assertTrue(bool(recon.get("active", False)))
        self.assertFalse(bool(recon.get("checks", {}).get("broker_row_diff_ok", True)))
        self.assertFalse(bool(out["checks"]["reconcile_drift_ok"]))
        self.assertIn("reconcile_broker_row_diff_high", set(recon.get("alerts", [])))
        self.assertGreaterEqual(int(recon.get("metrics", {}).get("broker_row_diff_samples", 0)), 1)
        row_diff_artifact = ((recon.get("artifacts", {}) or {}).get("row_diff", {}) if isinstance(recon, dict) else {})
        self.assertTrue(bool(row_diff_artifact.get("written", False)))
        artifact_json = Path(str(row_diff_artifact.get("json", "")))
        artifact_md = Path(str(row_diff_artifact.get("md", "")))
        self.assertTrue(artifact_json.exists())
        self.assertTrue(artifact_md.exists())
        artifact_payload = json.loads(artifact_json.read_text(encoding="utf-8"))
        self.assertGreaterEqual(int(artifact_payload.get("sample_rows", 0)), 1)
        self.assertGreaterEqual(int(artifact_payload.get("breach_rows", 0)), 1)
        self.assertTrue(list(artifact_payload.get("top_missing_on_broker", [])))
        self.assertTrue(list(artifact_payload.get("top_extra_on_broker", [])))
        series = recon.get("series", [])
        self.assertTrue(series)
        row_diff = ((series[-1].get("broker", {}) or {}).get("row_diff", {}) if isinstance(series[-1], dict) else {})
        self.assertTrue(bool(row_diff.get("active", False)))
        self.assertTrue(bool(row_diff.get("breached", False)))
        self.assertIn(str(row_diff.get("source", "")), {"canonical_file", "canonical_inline", "canonical_fallback"})

    def test_reconcile_row_diff_artifact_skips_when_no_breach(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        artifact = orch._write_reconcile_row_diff_artifact(
            as_of=d,
            series=[
                {
                    "date": d.isoformat(),
                    "broker": {
                        "row_diff": {
                            "active": True,
                            "skipped": False,
                            "breached": False,
                            "missing_on_broker": [],
                            "extra_on_broker": [],
                        }
                    },
                }
            ],
        )
        self.assertFalse(bool(artifact.get("written", True)))
        self.assertEqual(str(artifact.get("reason", "")), "no_row_diff_breach")
        self.assertEqual(int(artifact.get("sample_rows", -1)), 1)
        self.assertEqual(int(artifact.get("breach_rows", -1)), 0)

    def test_gate_report_reconcile_broker_row_diff_alias_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        daily_dir = td / "daily"
        manifest_dir = td / "artifacts" / "manifests"
        broker_dir = td / "artifacts" / "broker_snapshot"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        broker_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps({"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0}),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 1,
                            "risk_multiplier": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text("symbol,side,size_pct,status\n", encoding="utf-8")
            (broker_dir / f"{dstr}.json").write_text(
                json.dumps(
                    {
                        "source": "mock_live",
                        "open_positions": 1,
                        "closed_count": 0,
                        "closed_pnl": 0.0,
                        "positions": [
                            {
                                "symbol": "rb/2405",
                                "side": "sell",
                                "qty": "-2",
                                "entry_price": "3500",
                                "market_price": "3490",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

        (td / "artifacts" / "paper_positions_open.json").write_text(
            json.dumps(
                {
                    "as_of": d.isoformat(),
                    "positions": [
                        {
                            "open_date": d.isoformat(),
                            "symbol": "AU2406",
                            "side": "LONG",
                            "size_pct": 5.0,
                            "status": "OPEN",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute("CREATE TABLE executed_plans (date TEXT, symbol TEXT, pnl REAL, status TEXT)")
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_reconcile_window_days": 3,
                "ops_reconcile_min_samples": 3,
                "ops_reconcile_missing_ratio_max": 1.0,
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 1.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 1.0,
                "ops_reconcile_open_gap_ratio_max": 1.0,
                "ops_reconcile_broker_missing_ratio_max": 1.0,
                "ops_reconcile_broker_gap_ratio_max": 1.0,
                "ops_reconcile_broker_pnl_gap_abs_max": 1.0,
                "ops_reconcile_broker_contract_schema_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_numeric_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": 1.0,
                "ops_reconcile_broker_row_diff_min_samples": 1,
                "ops_reconcile_broker_row_diff_breach_ratio_max": 1.0,
                "ops_reconcile_broker_row_diff_key_mismatch_max": 1.0,
                "ops_reconcile_broker_row_diff_count_gap_max": 1.0,
                "ops_reconcile_broker_row_diff_notional_gap_max": 1.0,
                "ops_reconcile_broker_row_diff_alias_monitor_enabled": True,
                "ops_reconcile_broker_row_diff_alias_hit_rate_min": 0.50,
                "ops_reconcile_broker_row_diff_unresolved_key_ratio_max": 0.0,
                "ops_reconcile_broker_row_diff_asof_only": True,
                "ops_reconcile_require_broker_snapshot": True,
                "ops_reconcile_broker_contract_emit_canonical_view": True,
                "ops_reconcile_broker_contract_canonical_dir": "artifacts/broker_snapshot_canonical",
                "ops_reconcile_broker_row_diff_symbol_alias_map": {"RB2405.SHFE": "RB2405.SHFE"},
                "ops_reconcile_broker_row_diff_side_alias_map": {"SELLSHORT": "SHORT"},
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        recon = out.get("reconcile_drift", {})
        self.assertTrue(bool(recon.get("active", False)))
        checks = recon.get("checks", {}) if isinstance(recon.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("broker_row_diff_alias_drift_ok", True)))
        self.assertFalse(bool(out["checks"]["reconcile_drift_ok"]))
        self.assertIn("reconcile_broker_row_diff_alias_drift", set(recon.get("alerts", [])))
        metrics = recon.get("metrics", {}) if isinstance(recon.get("metrics", {}), dict) else {}
        self.assertGreaterEqual(float(metrics.get("broker_row_diff_unresolved_key_ratio", 0.0)), 1.0)
        self.assertAlmostEqual(float(metrics.get("broker_row_diff_alias_hit_rate", 0.0)), 0.0, places=8)
        series = recon.get("series", [])
        self.assertTrue(series)
        row_diff = ((series[-1].get("broker", {}) or {}).get("row_diff", {}) if isinstance(series[-1], dict) else {})
        self.assertGreaterEqual(int(row_diff.get("unresolved_keys", 0)), 1)
        self.assertAlmostEqual(float(row_diff.get("alias_hit_rate", 0.0)), 0.0, places=8)

    def test_compare_position_rows_uses_row_diff_alias_maps(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_reconcile_broker_row_diff_symbol_alias_map": {
                    "RB2405": "RB2405.SHFE",
                },
                "ops_reconcile_broker_row_diff_side_alias_map": {
                    "SELLSHORT": "SHORT",
                },
            }
        )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )

        out = orch._compare_position_rows(
            broker_rows=[{"symbol": "rb2405", "side": "sellshort", "qty": 1}],
            system_rows=[{"symbol": "RB2405.SHFE", "side": "SHORT", "qty": 1}],
            broker_source="paper_engine",
        )
        self.assertAlmostEqual(float(out.get("key_mismatch_ratio", 1.0)), 0.0, places=8)
        self.assertAlmostEqual(float(out.get("count_gap_ratio", 1.0)), 0.0, places=8)
        self.assertGreaterEqual(int(out.get("alias_hits", 0)), 1)
        self.assertGreaterEqual(int(out.get("symbol_alias_hits", 0)), 1)
        self.assertGreaterEqual(int(out.get("side_alias_hits", 0)), 1)
        self.assertAlmostEqual(float(out.get("alias_hit_rate", 0.0)), 0.5, places=8)
        self.assertAlmostEqual(float(out.get("unresolved_key_ratio", 1.0)), 0.0, places=8)
        self.assertEqual(list(out.get("missing_on_broker", [])), [])
        self.assertEqual(list(out.get("extra_on_broker", [])), [])

    def test_review_until_pass_defect_plan_includes_reconcile_broker_contract_schema(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        daily_dir = td / "daily"
        manifest_dir = td / "artifacts" / "manifests"
        broker_dir = td / "artifacts" / "broker_snapshot"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        daily_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        broker_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps({"quality": {"passed": True, "flags": [], "source_confidence_score": 0.90}, "risk_multiplier": 1.0}),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday), encoding="utf-8")
            (manifest_dir / f"eod_{dstr}.json").write_text(
                json.dumps(
                    {
                        "checks": {"quality_passed": True},
                        "metrics": {
                            "plans": 0,
                            "closed_trades": 0,
                            "closed_pnl": 0.0,
                            "open_positions": 0,
                            "risk_multiplier": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text("symbol,side,size_pct,status\n", encoding="utf-8")
            (broker_dir / f"{dstr}.json").write_text(
                json.dumps(
                    {
                        "source": "",
                        "open_positions": 0,
                        "closed_count": 0,
                        "closed_pnl": 0.0,
                        "positions": {"symbol": "rb2405"},
                    }
                ),
                encoding="utf-8",
            )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute("CREATE TABLE executed_plans (date TEXT, symbol TEXT, pnl REAL, status TEXT)")
            conn.commit()

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 3,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_reconcile_window_days": 3,
                "ops_reconcile_min_samples": 3,
                "ops_reconcile_missing_ratio_max": 1.0,
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 1.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 1.0,
                "ops_reconcile_open_gap_ratio_max": 1.0,
                "ops_reconcile_broker_missing_ratio_max": 1.0,
                "ops_reconcile_broker_gap_ratio_max": 1.0,
                "ops_reconcile_broker_pnl_gap_abs_max": 1.0,
                "ops_reconcile_broker_contract_schema_invalid_ratio_max": 0.20,
                "ops_reconcile_broker_contract_numeric_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_invalid_ratio_max": 1.0,
                "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": 1.0,
                "ops_reconcile_require_broker_snapshot": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 1, "stderr": "test_x ... FAIL", "stdout": ""},
            load_json_safely=_load_json,
            sqlite_path=db_path,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(out["passed"])
        plan_json = Path(str(out["rounds"][0]["defect_plan"]["json"]))
        plan = json.loads(plan_json.read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("RECONCILE_BROKER_CONTRACT_SCHEMA", codes)


if __name__ == "__main__":
    unittest.main()
