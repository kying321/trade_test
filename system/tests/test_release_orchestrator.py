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


if __name__ == "__main__":
    unittest.main()
