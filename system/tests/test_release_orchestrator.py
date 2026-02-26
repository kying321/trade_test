from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch
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
        self.assertIn("artifact_governance_ok", out["checks"])
        self.assertTrue(bool(out["checks"]["artifact_governance_ok"]))
        artifact_governance = out.get("artifact_governance", {})
        self.assertTrue(bool(artifact_governance.get("active", False)))
        self.assertEqual(int((artifact_governance.get("metrics", {}) or {}).get("profiles_total", 0)), 6)
        release_decision = out.get("release_decision", {})
        self.assertTrue(bool(release_decision.get("decision_id", "")))
        snapshot_path = Path(str(release_decision.get("snapshot_path", "")))
        self.assertTrue(snapshot_path.exists())
        self.assertTrue((review_dir / f"{d.isoformat()}_gate_report.json").exists())
        event = out.get("event_envelope", {})
        self.assertTrue(bool(event.get("event_id", "")))
        self.assertEqual(str(event.get("trace_id", "")), str(out.get("trace_id", "")))
        self.assertEqual(str(event.get("traceparent", "")), str(out.get("traceparent", "")))
        event_stream = Path(str(out.get("event_stream_path", "")))
        self.assertTrue(event_stream.exists())

    def test_run_review_cycle_propagates_trace_to_gate(self) -> None:
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
        out = orch.run_review_cycle(as_of=d, max_rounds=0, trace_id="trace_chain_001", parent_event_id="evt_root")

        self.assertEqual(str(out.get("trace_id", "")), "trace_chain_001")
        gate = out.get("gate_report", {})
        self.assertEqual(str(gate.get("trace_id", "")), "trace_chain_001")
        self.assertTrue(bool(str(gate.get("traceparent", ""))))
        self.assertEqual(str(out.get("traceparent", "")), str((out.get("event_envelope", {}) or {}).get("traceparent", "")))
        chain = out.get("event_chain", {})
        self.assertTrue(bool(chain.get("start_event_id", "")))
        self.assertTrue(bool(chain.get("gate_event_id", "")))
        self.assertTrue(bool(chain.get("final_event_id", "")))
        self.assertTrue(bool(chain.get("gate_traceparent", "")))

    def test_gate_report_release_decision_id_is_stable_for_same_inputs(self) -> None:
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
        out1 = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        out2 = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        decision1 = str((out1.get("release_decision", {}) or {}).get("decision_id", ""))
        decision2 = str((out2.get("release_decision", {}) or {}).get("decision_id", ""))
        self.assertTrue(bool(decision1))
        self.assertEqual(decision1, decision2)

    def test_gate_report_release_decision_freshness_hard_fail_blocks_stale_review(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_delta = review_dir / f"{d.isoformat()}_param_delta.yaml"
        review_delta.write_text("pass_gate: true\nmode_health:\n  passed: true\n", encoding="utf-8")
        old_ts = (datetime.now() - timedelta(hours=8)).timestamp()
        os.utime(review_delta, (old_ts, old_ts))

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "release_decision_freshness_enabled": True,
                "release_decision_freshness_hard_fail": True,
                "release_decision_review_max_staleness_hours": 1,
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
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"].get("release_decision_freshness_ok", True)))
        self.assertFalse(bool(out["passed"]))
        freshness = out.get("release_decision_freshness", {})
        self.assertTrue(bool(freshness.get("monitor_failed", False)))
        self.assertIn("release_decision_review_stale", set(freshness.get("alerts", [])))

    def test_gate_report_release_decision_freshness_monitor_mode_does_not_block(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_delta = review_dir / f"{d.isoformat()}_param_delta.yaml"
        review_delta.write_text("pass_gate: true\nmode_health:\n  passed: true\n", encoding="utf-8")
        old_ts = (datetime.now() - timedelta(hours=8)).timestamp()
        os.utime(review_delta, (old_ts, old_ts))

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "release_decision_freshness_enabled": True,
                "release_decision_freshness_hard_fail": False,
                "release_decision_review_max_staleness_hours": 1,
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
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(bool(out["checks"].get("release_decision_freshness_ok", False)))
        self.assertTrue(bool(out["passed"]))
        freshness = out.get("release_decision_freshness", {})
        self.assertTrue(bool(freshness.get("monitor_failed", False)))
        self.assertIn("release_decision_review_stale", set(freshness.get("alerts", [])))

    def test_gate_report_same_day_before_review_window_does_not_require_review_or_backtest(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 20)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        settings = self._make_settings()
        settings.raw.setdefault("schedule", {})
        settings.raw["schedule"].update({"eod": "15:10", "nightly_review": "20:30"})
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "release_decision_freshness_enabled": True,
                "release_decision_freshness_hard_fail": True,
            }
        )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=(
                lambda as_of, require_review: (
                    {"status": "degraded", "missing": ["review_report", "review_delta"]}
                    if require_review
                    else {"status": "healthy", "missing": []}
                )
            ),
            stable_replay_check=lambda as_of, days: {"passed": False, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )

        class _Now0900(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                base = datetime(2026, 2, 20, 9, 0, 0)
                if tz is not None:
                    return base.replace(tzinfo=tz)
                return base

        with patch("lie_engine.orchestration.release.datetime", _Now0900):
            out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)

        checks = out.get("checks", {})
        self.assertTrue(bool(checks.get("review_pass_gate", False)))
        self.assertTrue(bool(checks.get("health_ok", False)))
        self.assertTrue(bool(checks.get("release_decision_freshness_ok", False)))
        self.assertTrue(bool(checks.get("stable_replay_ok", False)))
        self.assertTrue(bool(checks.get("backtest_snapshot_ok", False)))
        self.assertTrue(bool(checks.get("positive_window_ratio_ok", False)))
        self.assertTrue(bool(checks.get("max_drawdown_ok", False)))
        self.assertTrue(bool(checks.get("risk_violations_ok", False)))
        freshness = out.get("release_decision_freshness", {})
        self.assertFalse(bool((freshness.get("metrics", {}) or {}).get("review_required", True)))
        self.assertEqual(list(freshness.get("alerts", [])), [])
        self.assertEqual(str((out.get("stable_replay", {}) or {}).get("reason", "")), "before_review_window")
        self.assertTrue(bool(out.get("passed", False)))

    def test_gate_report_same_day_after_review_window_requires_review(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 20)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        settings = self._make_settings()
        settings.raw.setdefault("schedule", {})
        settings.raw["schedule"].update({"eod": "15:10", "nightly_review": "20:30"})
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "release_decision_freshness_enabled": True,
                "release_decision_freshness_hard_fail": True,
            }
        )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=(
                lambda as_of, require_review: (
                    {"status": "degraded", "missing": ["review_report", "review_delta"]}
                    if require_review
                    else {"status": "healthy", "missing": []}
                )
            ),
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )

        class _Now2100(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                base = datetime(2026, 2, 20, 21, 0, 0)
                if tz is not None:
                    return base.replace(tzinfo=tz)
                return base

        with patch("lie_engine.orchestration.release.datetime", _Now2100):
            out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)

        checks = out.get("checks", {})
        self.assertFalse(bool(checks.get("review_pass_gate", True)))
        self.assertFalse(bool(checks.get("health_ok", True)))
        self.assertFalse(bool(checks.get("release_decision_freshness_ok", True)))
        freshness = out.get("release_decision_freshness", {})
        self.assertTrue(bool((freshness.get("metrics", {}) or {}).get("review_required", False)))
        self.assertIn("release_decision_review_missing", set(freshness.get("alerts", [])))
        self.assertFalse(bool(out.get("passed", True)))

    def test_gate_report_degradation_guardrail_dashboard_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        for i in range(2):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": True,
                        "triggered": False,
                        "applied": False,
                        "reason": "rollback_cooldown_active",
                        "cooldown": {"rollback_active": True},
                        "snapshot_promotion": {"eligible": False, "promoted": False},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": True,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 2,
                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.4,
                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.4,
                "ops_degradation_guardrail_promotion_latency_days_max": 21,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"].get("degradation_guardrail_dashboard_ok", True)))
        self.assertFalse(bool(out["passed"]))
        dashboard = out.get("degradation_guardrail_dashboard", {})
        self.assertTrue(bool(dashboard.get("active", False)))
        self.assertTrue(bool(dashboard.get("monitor_failed", False)))
        alerts = set(dashboard.get("alerts", []))
        self.assertIn("degradation_guardrail_cooldown_hit_rate_high", alerts)
        self.assertIn("degradation_guardrail_suppressed_density_high", alerts)
        rollback = out.get("rollback_recommendation", {})
        self.assertIn("degradation_guardrail", set(rollback.get("reason_codes", [])))

    def test_ops_report_flags_degradation_guardrail_dashboard(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        for i in range(2):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": True,
                        "triggered": False,
                        "applied": False,
                        "reason": "rollback_cooldown_active",
                        "cooldown": {"rollback_active": True},
                        "snapshot_promotion": {"eligible": False, "promoted": False},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 2,
                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.4,
                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.4,
                "ops_degradation_guardrail_promotion_latency_days_max": 21,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        self.assertEqual(str(out.get("status", "")), "red")
        top_alert_details = out.get("alert_details", [])
        self.assertTrue(isinstance(top_alert_details, list))
        self.assertTrue(any(str((x or {}).get("code", "")) == "degradation_guardrail_cooldown_hit_rate_high" for x in top_alert_details if isinstance(x, dict)))
        dashboard = out.get("degradation_guardrail_dashboard", {})
        self.assertTrue(bool(dashboard.get("active", False)))
        checks = dashboard.get("checks", {}) if isinstance(dashboard.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("cooldown_hit_rate_ok", True)))
        self.assertFalse(bool(checks.get("suppressed_trigger_density_ok", True)))
        dashboard_alert_details = dashboard.get("alert_details", [])
        self.assertTrue(isinstance(dashboard_alert_details, list))
        self.assertTrue(any(str((x or {}).get("severity", "")) in {"warning", "critical", "info"} for x in dashboard_alert_details if isinstance(x, dict)))
        report_md = review_dir / f"{d.isoformat()}_ops_report.md"
        self.assertTrue(report_md.exists())
        report_text = report_md.read_text(encoding="utf-8")
        self.assertIn("## 降级护栏仪表板", report_text)
        self.assertIn("- alert_details:", report_text)

    def test_ops_report_surfaces_dependency_audit_dashboard_adapter_summary(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (review_dir / f"{d.isoformat()}_dependency_audit.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "files_checked": 5,
                    "total_files_checked": 7,
                    "violations": ["dashboard/api/main.py: dashboard_api -> lie_engine.engine"],
                    "core_violations": [],
                    "dashboard_adapter": {
                        "enabled": True,
                        "files_checked": 2,
                        "imports": ["lie_engine.engine"],
                        "violations": ["dashboard/api/main.py: dashboard_api -> lie_engine.engine"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        self.assertEqual(str(out.get("status", "")), "red")
        dependency = out.get("dependency_audit", {}) if isinstance(out.get("dependency_audit", {}), dict) else {}
        self.assertTrue(bool(dependency.get("active", False)))
        self.assertFalse(bool(dependency.get("gate_ok", True)))
        self.assertEqual(int(dependency.get("dashboard_adapter_violations", 0)), 1)
        checks = dependency.get("checks", {}) if isinstance(dependency.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("dashboard_adapter_ok", True)))
        alerts = set(str(x).strip().lower() for x in out.get("alerts", []) if str(x).strip())
        self.assertIn("dependency_layer_violation", alerts)
        self.assertIn("dashboard_adapter_dependency_violation", alerts)
        report_md = review_dir / f"{d.isoformat()}_ops_report.md"
        self.assertTrue(report_md.exists())
        report_text = report_md.read_text(encoding="utf-8")
        self.assertIn("## 依赖分层审计", report_text)
        self.assertIn("dependency_audit(active/gate_ok/dashboard_adapter_ok/artifact_trend_ok)", report_text)

    def test_ops_report_degrades_on_corrupted_dependency_audit_artifact(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (review_dir / f"{d.isoformat()}_dependency_audit.json").write_text(
            "{ this is not valid json }",
            encoding="utf-8",
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.ops_report(as_of=d, window_days=3)
        self.assertEqual(str(out.get("status", "")), "red")
        dependency = out.get("dependency_audit", {}) if isinstance(out.get("dependency_audit", {}), dict) else {}
        self.assertTrue(bool(dependency.get("active", False)))
        self.assertFalse(bool(dependency.get("gate_ok", True)))
        self.assertTrue(bool(dependency.get("artifact_corrupt", False)))
        checks = dependency.get("checks", {}) if isinstance(dependency.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("artifact_ok", True)))
        alerts = set(str(x).strip().lower() for x in out.get("alerts", []) if str(x).strip())
        self.assertIn("dependency_audit_artifact_corrupt", alerts)
        report_md = review_dir / f"{d.isoformat()}_ops_report.md"
        self.assertTrue(report_md.exists())
        report_text = report_md.read_text(encoding="utf-8")
        self.assertIn("artifact_ok/corrupt/error", report_text)

    def test_gate_report_dependency_audit_artifact_trend_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        for day in (d, d - timedelta(days=1), d - timedelta(days=2)):
            artifact = review_dir / f"{day.isoformat()}_dependency_audit.json"
            if day == d - timedelta(days=2):
                artifact.write_text(
                    json.dumps(
                        {
                            "ok": True,
                            "dashboard_adapter": {"enabled": True, "violations": []},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            else:
                artifact.write_text("{ invalid json", encoding="utf-8")

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        settings = self._make_settings()
        settings.validation.update(
            {
                "ops_dependency_audit_artifact_trend_enabled": True,
                "ops_dependency_audit_artifact_trend_gate_hard_fail": True,
                "ops_dependency_audit_artifact_trend_require_active": True,
                "ops_dependency_audit_artifact_trend_window_days": 3,
                "ops_dependency_audit_artifact_trend_min_samples": 1,
                "ops_dependency_audit_artifact_trend_max_stale_days": 14,
                "ops_dependency_audit_artifact_trend_max_corrupt_ratio": 0.20,
                "ops_dependency_audit_artifact_trend_max_missing_ratio": 1.00,
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out.get("passed", True)))
        checks = out.get("checks", {}) if isinstance(out.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("dependency_audit_artifact_trend_ok", True)))
        trend = (
            out.get("dependency_audit_artifact_trend", {})
            if isinstance(out.get("dependency_audit_artifact_trend", {}), dict)
            else {}
        )
        self.assertTrue(bool(trend.get("active", False)))
        self.assertEqual(str(trend.get("status", "")), "corrupt_ratio_high")
        trend_alerts = set(str(x).strip() for x in trend.get("alerts", []) if str(x).strip())
        self.assertIn("dependency_audit_artifact_trend_corrupt_ratio_high", trend_alerts)
        rollback = out.get("rollback_recommendation", {}) if isinstance(out.get("rollback_recommendation", {}), dict) else {}
        reason_codes = set(str(x).strip() for x in rollback.get("reason_codes", []) if str(x).strip())
        self.assertIn("dependency_audit_artifact_trend", reason_codes)

    def test_ops_report_surfaces_dependency_audit_artifact_trend_section(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        for day in (d, d - timedelta(days=1), d - timedelta(days=2)):
            artifact = review_dir / f"{day.isoformat()}_dependency_audit.json"
            if day == d - timedelta(days=2):
                artifact.write_text(
                    json.dumps(
                        {
                            "ok": True,
                            "dashboard_adapter": {"enabled": True, "violations": []},
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            else:
                artifact.write_text("{ invalid json", encoding="utf-8")

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        settings = self._make_settings()
        settings.validation.update(
            {
                "ops_dependency_audit_artifact_trend_enabled": True,
                "ops_dependency_audit_artifact_trend_gate_hard_fail": True,
                "ops_dependency_audit_artifact_trend_require_active": True,
                "ops_dependency_audit_artifact_trend_window_days": 3,
                "ops_dependency_audit_artifact_trend_min_samples": 1,
                "ops_dependency_audit_artifact_trend_max_stale_days": 14,
                "ops_dependency_audit_artifact_trend_max_corrupt_ratio": 0.20,
                "ops_dependency_audit_artifact_trend_max_missing_ratio": 1.00,
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.ops_report(as_of=d, window_days=3)
        self.assertEqual(str(out.get("status", "")), "red")
        trend = (
            out.get("dependency_audit_artifact_trend", {})
            if isinstance(out.get("dependency_audit_artifact_trend", {}), dict)
            else {}
        )
        self.assertTrue(bool(trend.get("active", False)))
        self.assertFalse(bool(trend.get("gate_ok", True)))
        trend_checks = trend.get("checks", {}) if isinstance(trend.get("checks", {}), dict) else {}
        self.assertFalse(bool(trend_checks.get("corrupt_ratio_ok", True)))
        alerts = set(str(x).strip().lower() for x in out.get("alerts", []) if str(x).strip())
        self.assertIn("dependency_audit_artifact_trend_corrupt_ratio_high", alerts)
        report_md = review_dir / f"{d.isoformat()}_ops_report.md"
        self.assertTrue(report_md.exists())
        report_text = report_md.read_text(encoding="utf-8")
        self.assertIn("## 依赖审计工件趋势", report_text)
        self.assertIn("corrupt_ratio/missing_ratio/source_age_days", report_text)

    def test_ops_report_includes_guard_loop_cadence_drillbook(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 4,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": True,
                        "reason_codes": [
                            "CADENCE_DUE_NON_APPLY_STREAK",
                            "CADENCE_DUE_NON_APPLY_HEAVY",
                        ],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": True, "heavy_due": True},
                    },
                    "recovery": {
                        "mode": "heavy",
                        "status": "ok",
                        "reason_codes": ["CADENCE_DUE_NON_APPLY_REPLAY_HEAVY"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 4,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [
                        {
                            "ts": "2026-02-13T20:30:00+08:00",
                            "date": "2026-02-13",
                            "cadence_due": True,
                            "cadence_non_apply_streak": 4,
                            "cadence_non_apply_apply_seen": False,
                            "cadence_non_apply_reason_codes": [
                                "CADENCE_DUE_NON_APPLY_STREAK",
                                "CADENCE_DUE_NON_APPLY_HEAVY",
                            ],
                            "recovery_mode": "heavy",
                        }
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
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        guard = (
            out.get("guard_loop_cadence_non_apply", {})
            if isinstance(out.get("guard_loop_cadence_non_apply", {}), dict)
            else {}
        )
        lift_snapshot = (
            out.get("cadence_non_apply_lift_snapshot", {})
            if isinstance(out.get("cadence_non_apply_lift_snapshot", {}), dict)
            else {}
        )
        self.assertTrue(bool(guard.get("active", False)))
        self.assertIn("guard_loop_cadence_non_apply_heavy", set(guard.get("alerts", [])))
        self.assertIn("cooldown_remaining_days", lift_snapshot)
        report_md = review_dir / f"{d.isoformat()}_ops_report.md"
        text = report_md.read_text(encoding="utf-8")
        self.assertIn("## Guard-Loop Cadence Non-Apply", text)
        self.assertIn("## Guard-Loop Cadence Lift Trend", text)
        self.assertIn("cadence_lift_snapshot(applied/blocked/cooldown_remaining):", text)
        self.assertIn("### Drillbook", text)
        self.assertIn("lie run-halfhour-pulse --date 2026-02-13 --force --max-slot-runs 8", text)

    def test_ops_report_flags_compaction_restore_trend(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "weekly_guardrail_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "date": "2026-02-06",
                            "week_tag": "2026-W06",
                            "status": "ok",
                            "maintenance": {
                                "compact": {
                                    "ran": True,
                                    "status": "ok",
                                    "dry_run": False,
                                },
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "ok",
                                    "checks": {"restore_delta_match": True},
                                },
                            },
                        },
                        {
                            "date": "2026-02-13",
                            "week_tag": "2026-W07",
                            "status": "ok",
                            "maintenance": {
                                "compact": {
                                    "ran": True,
                                    "status": "ok",
                                    "dry_run": False,
                                },
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "error",
                                    "checks": {"restore_delta_match": False},
                                },
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_compaction_restore_trend_enabled": True,
                "ops_compaction_restore_trend_window_weeks": 8,
                "ops_compaction_restore_trend_min_samples": 2,
                "ops_compaction_restore_trend_min_restore_pass_ratio": 1.0,
                "ops_compaction_restore_trend_min_restore_delta_match_ratio": 1.0,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        self.assertEqual(str(out.get("status", "")), "red")
        trend = out.get("compaction_restore_trend", {})
        self.assertTrue(bool(trend.get("active", False)))
        checks = trend.get("checks", {}) if isinstance(trend.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("restore_pass_ratio_ok", True)))
        self.assertFalse(bool(checks.get("restore_delta_match_ratio_ok", True)))
        report_md = review_dir / f"{d.isoformat()}_ops_report.md"
        self.assertTrue(report_md.exists())
        self.assertIn("## Compaction/Restore 趋势", report_md.read_text(encoding="utf-8"))

    def test_ops_report_compaction_restore_trend_single_restore_sample_is_neutral(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "weekly_guardrail_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "date": "2026-02-06",
                            "week_tag": "2026-W06",
                            "status": "ok",
                            "maintenance": {
                                "compact": {
                                    "ran": True,
                                    "status": "ok",
                                    "dry_run": False,
                                },
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "skipped",
                                    "checks": {"restore_delta_match": False},
                                },
                            },
                        },
                        {
                            "date": "2026-02-13",
                            "week_tag": "2026-W07",
                            "status": "ok",
                            "maintenance": {
                                "compact": {
                                    "ran": True,
                                    "status": "ok",
                                    "dry_run": True,
                                },
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": False,
                                    "status": "skipped",
                                    "checks": {"restore_delta_match": False},
                                },
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_compaction_restore_trend_enabled": True,
                "ops_compaction_restore_trend_window_weeks": 8,
                "ops_compaction_restore_trend_min_samples": 2,
                "ops_compaction_restore_trend_min_restore_required_runs": 2,
                "ops_compaction_restore_trend_min_restore_pass_ratio": 1.0,
                "ops_compaction_restore_trend_min_restore_delta_match_ratio": 1.0,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        trend = out.get("compaction_restore_trend", {})
        self.assertTrue(bool(trend.get("active", False)))
        self.assertFalse(bool(trend.get("monitor_failed", True)))
        alerts = set(trend.get("alerts", [])) if isinstance(trend, dict) else set()
        self.assertNotIn("compaction_trend_restore_verify_pass_ratio_low", alerts)
        self.assertNotIn("compaction_trend_restore_verify_delta_ratio_low", alerts)
        self.assertNotIn("compaction_trend_latest_restore_delta_mismatch", alerts)
        checks = trend.get("checks", {}) if isinstance(trend.get("checks", {}), dict) else {}
        self.assertTrue(bool(checks.get("restore_pass_ratio_ok", False)))
        self.assertTrue(bool(checks.get("restore_delta_match_ratio_ok", False)))
        self.assertTrue(bool(checks.get("latest_restore_delta_match_ok", False)))
        metrics = trend.get("metrics", {}) if isinstance(trend.get("metrics", {}), dict) else {}
        self.assertEqual(int(metrics.get("restore_required_runs", 0)), 1)
        self.assertFalse(bool(metrics.get("restore_samples_ready", True)))

    def test_gate_report_degradation_guardrail_dashboard_uses_live_overrides(self) -> None:
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
        for i in range(2):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": True,
                        "triggered": False,
                        "applied": False,
                        "reason": "rollback_cooldown_active",
                        "cooldown": {"rollback_active": True},
                        "snapshot_promotion": {"eligible": False, "promoted": False},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        (artifacts_dir / "degradation_guardrail_dashboard_live.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "as_of": d.isoformat(),
                    "params": {
                        "ops_degradation_guardrail_cooldown_hit_rate_max": 1.0,
                        "ops_degradation_guardrail_suppressed_trigger_density_max": 1.0,
                        "ops_degradation_guardrail_promotion_latency_days_max": 99,
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
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 2,
                "ops_degradation_guardrail_dashboard_use_live_overrides": True,
                "ops_degradation_guardrail_dashboard_live_params_path": "artifacts/degradation_guardrail_dashboard_live.yaml",
                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.1,
                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.1,
                "ops_degradation_guardrail_promotion_latency_days_max": 1,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        dashboard = out.get("degradation_guardrail_dashboard", {})
        thresholds = dashboard.get("thresholds", {}) if isinstance(dashboard.get("thresholds", {}), dict) else {}
        checks = dashboard.get("checks", {}) if isinstance(dashboard.get("checks", {}), dict) else {}
        self.assertTrue(bool(thresholds.get("degradation_guardrail_live_overrides_applied", False)))
        self.assertAlmostEqual(
            float(thresholds.get("ops_degradation_guardrail_cooldown_hit_rate_max", 0.0)),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(thresholds.get("ops_degradation_guardrail_suppressed_trigger_density_max", 0.0)),
            1.0,
            places=6,
        )
        self.assertEqual(
            int(thresholds.get("ops_degradation_guardrail_promotion_latency_days_max", 0)),
            99,
        )
        self.assertTrue(bool(checks.get("cooldown_hit_rate_ok", False)))
        self.assertTrue(bool(checks.get("suppressed_trigger_density_ok", False)))

    def test_degradation_guardrail_burnin_writes_live_overrides(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": True,
                        "triggered": False,
                        "applied": False,
                        "reason": "rollback_cooldown_active",
                        "cooldown": {"rollback_active": True},
                        "snapshot_promotion": {"eligible": False, "promoted": False},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 1,
                "ops_degradation_guardrail_dashboard_use_live_overrides": True,
                "ops_degradation_guardrail_dashboard_live_params_path": "artifacts/degradation_guardrail_dashboard_live.yaml",
                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.1,
                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.1,
                "ops_degradation_guardrail_promotion_latency_days_max": 1,
                "ops_degradation_guardrail_false_positive_target_max": 0.0,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {
                "as_of": as_of.isoformat(),
                "replay_days": int(days or 3),
                "replay_executed": bool(run_eod_replay),
                "passed": True,
                "checks": [{"date": (as_of - timedelta(days=i)).isoformat(), "ok": True} for i in range(int(days or 3))],
            },
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.degradation_guardrail_burnin(as_of=d, days=3, run_stable_replay=True, auto_tune=True)
        self.assertIn("paths", out)
        paths = out.get("paths", {}) if isinstance(out.get("paths", {}), dict) else {}
        self.assertTrue(Path(str(paths.get("json", ""))).exists())
        self.assertTrue(Path(str(paths.get("md", ""))).exists())
        live = out.get("live_overrides", {}) if isinstance(out.get("live_overrides", {}), dict) else {}
        self.assertTrue(bool(live.get("applied", False)))
        live_path = Path(str(live.get("path", "")))
        self.assertTrue(live_path.exists())
        payload = yaml.safe_load(live_path.read_text(encoding="utf-8")) or {}
        params = payload.get("params", {}) if isinstance(payload.get("params", {}), dict) else {}
        self.assertGreaterEqual(
            float(params.get("ops_degradation_guardrail_cooldown_hit_rate_max", 0.0)),
            0.1,
        )
        self.assertGreaterEqual(
            float(params.get("ops_degradation_guardrail_suppressed_trigger_density_max", 0.0)),
            0.1,
        )
        self.assertGreaterEqual(
            int(params.get("ops_degradation_guardrail_promotion_latency_days_max", 0)),
            1,
        )

    def test_degradation_guardrail_burnin_autofills_missing_rollbacks_for_coverage(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_calls: list[str] = []

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 3,
                "ops_degradation_guardrail_dashboard_use_live_overrides": False,
                "ops_degradation_guardrail_false_positive_target_max": 0.50,
                "ops_degradation_guardrail_burnin_autofill_review_if_missing": True,
                "ops_degradation_guardrail_burnin_require_min_samples_for_tune": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        def _run_review(as_of: date) -> ReviewDelta:
            day_tag = as_of.isoformat()
            review_calls.append(day_tag)
            (review_dir / f"{day_tag}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            (review_dir / f"{day_tag}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": False,
                        "triggered": False,
                        "applied": False,
                        "reason": "not_triggered",
                        "cooldown": {"rollback_active": False},
                        "snapshot_promotion": {"eligible": False, "promoted": False},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {
                "as_of": as_of.isoformat(),
                "replay_days": int(days or 3),
                "replay_executed": bool(run_eod_replay),
                "passed": True,
                "checks": [],
            },
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.degradation_guardrail_burnin(as_of=d, days=3, run_stable_replay=False, auto_tune=False)
        summary = out.get("summary", {}) if isinstance(out.get("summary", {}), dict) else {}
        coverage = out.get("coverage", {}) if isinstance(out.get("coverage", {}), dict) else {}
        self.assertEqual(int(summary.get("min_samples", 0)), 3)
        self.assertTrue(bool(summary.get("active_days_ok", False)))
        self.assertTrue(bool(summary.get("rollback_artifact_days_ok", False)))
        self.assertTrue(bool(summary.get("coverage_ok", False)))
        self.assertGreaterEqual(int(summary.get("active_days", 0)), 3)
        self.assertGreaterEqual(int(summary.get("rollback_artifact_days", 0)), 3)
        self.assertEqual(int(coverage.get("autofill_attempted_days", 0)), 3)
        self.assertEqual(int(coverage.get("autofill_succeeded_days", 0)), 3)
        self.assertEqual(int(coverage.get("autofill_failed_days", 0)), 0)
        self.assertEqual(set(review_calls), {(d - timedelta(days=i)).isoformat() for i in range(3)})

    def test_degradation_guardrail_burnin_light_backfill_uses_budgeted_calibration(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        calibration_calls: list[str] = []
        review_calls: list[str] = []

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 3,
                "ops_degradation_guardrail_dashboard_use_live_overrides": False,
                "ops_degradation_guardrail_false_positive_target_max": 0.50,
                "ops_degradation_guardrail_burnin_light_backfill_enabled": True,
                "ops_degradation_guardrail_burnin_light_backfill_max_days_per_run": 2,
                "ops_degradation_guardrail_burnin_autofill_review_if_missing": True,
                "ops_degradation_guardrail_burnin_review_autofill_max_days_per_run": 0,
                "ops_degradation_guardrail_burnin_require_min_samples_for_tune": True,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        def _run_review(as_of: date) -> ReviewDelta:
            review_calls.append(as_of.isoformat())
            return ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            )

        def _run_degradation_calibration(as_of: date) -> dict[str, object]:
            day_tag = as_of.isoformat()
            calibration_calls.append(day_tag)
            (review_dir / f"{day_tag}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": False,
                        "triggered": False,
                        "applied": False,
                        "reason": "not_triggered",
                        "cooldown": {"rollback_active": False},
                        "snapshot_promotion": {"eligible": False, "promoted": False},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return {"applied": False}

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {
                "as_of": as_of.isoformat(),
                "replay_days": int(days or 3),
                "replay_executed": bool(run_eod_replay),
                "passed": True,
                "checks": [],
            },
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_degradation_calibration=_run_degradation_calibration,
        )
        out = orch.degradation_guardrail_burnin(as_of=d, days=3, run_stable_replay=False, auto_tune=False)
        summary = out.get("summary", {}) if isinstance(out.get("summary", {}), dict) else {}
        coverage = out.get("coverage", {}) if isinstance(out.get("coverage", {}), dict) else {}
        self.assertEqual(len(calibration_calls), 2)
        self.assertEqual(set(calibration_calls), {(d - timedelta(days=i)).isoformat() for i in range(2)})
        self.assertEqual(len(review_calls), 0)
        self.assertEqual(int(summary.get("rollback_artifact_days", 0)), 2)
        self.assertFalse(bool(summary.get("coverage_ok", True)))
        self.assertEqual(int(coverage.get("light_backfill_attempted_days", 0)), 2)
        self.assertEqual(int(coverage.get("light_backfill_succeeded_days", 0)), 2)
        self.assertEqual(int(coverage.get("light_backfill_failed_days", 0)), 0)
        self.assertEqual(int(coverage.get("light_backfill_budget_skipped_days", 0)), 1)
        self.assertEqual(int(coverage.get("review_autofill_budget_skipped_days", 0)), 1)

    def test_degradation_guardrail_burnin_low_cost_replay_is_budget_bounded(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        replayed_days: set[str] = set()
        replay_calls: list[str] = []

        for i in range(3):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": False,
                        "triggered": False,
                        "applied": False,
                        "reason": "not_triggered",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 3,
                "ops_degradation_guardrail_dashboard_use_live_overrides": False,
                "ops_degradation_guardrail_false_positive_target_max": 0.50,
                "ops_degradation_guardrail_burnin_light_backfill_enabled": False,
                "ops_degradation_guardrail_burnin_autofill_review_if_missing": False,
                "ops_degradation_guardrail_burnin_low_cost_replay_enabled": True,
                "ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run": 2,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        def _stable_replay(as_of: date, days: int | None, run_eod_replay: bool = True) -> dict[str, object]:
            day_tag = as_of.isoformat()
            replay_calls.append(day_tag)
            replayed_days.add(day_tag)
            return {
                "as_of": day_tag,
                "replay_days": int(days or 1),
                "replay_executed": bool(run_eod_replay),
                "passed": True,
                "checks": [{"date": day_tag, "ok": True}],
            }

        def _fake_gate(
            self: ReleaseOrchestrator,
            as_of: date,
            run_tests: bool = False,
            run_review_if_missing: bool = False,
            run_stable_replay: bool = False,
        ) -> dict[str, object]:
            day_tag = as_of.isoformat()
            sample_ready = day_tag in replayed_days
            return {
                "passed": True,
                "checks": {},
                "degradation_guardrail_dashboard": {
                    "active": bool(sample_ready),
                    "samples": 1 if sample_ready else 0,
                    "monitor_failed": False,
                    "metrics": {
                        "cooldown_hit_rate": 0.05,
                        "suppressed_trigger_density": 0.02,
                        "promotion_latency_avg_days": 1.0,
                    },
                    "thresholds": {},
                },
            }

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=_stable_replay,
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_degradation_calibration=None,
        )
        with patch.object(ReleaseOrchestrator, "gate_report", new=_fake_gate):
            out = orch.degradation_guardrail_burnin(
                as_of=d,
                days=3,
                run_stable_replay=False,
                auto_tune=False,
            )
        summary = out.get("summary", {}) if isinstance(out.get("summary", {}), dict) else {}
        coverage = out.get("coverage", {}) if isinstance(out.get("coverage", {}), dict) else {}
        self.assertEqual(int(summary.get("active_days", 0)), 2)
        self.assertFalse(bool(summary.get("active_days_ok", True)))
        self.assertFalse(bool(summary.get("coverage_ok", True)))
        self.assertEqual(int(coverage.get("low_cost_replay_attempted_days", 0)), 2)
        self.assertEqual(int(coverage.get("low_cost_replay_succeeded_days", 0)), 2)
        self.assertEqual(int(coverage.get("low_cost_replay_failed_days", 0)), 0)
        self.assertEqual(int(coverage.get("low_cost_replay_budget_skipped_days", 0)), 1)
        self.assertEqual(set(replay_calls), {(d - timedelta(days=i)).isoformat() for i in range(2)})

    def test_degradation_guardrail_burnin_low_cost_replay_can_reach_min_samples(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        replayed_days: set[str] = set()

        for i in range(3):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": False,
                        "triggered": False,
                        "applied": False,
                        "reason": "not_triggered",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 3,
                "ops_degradation_guardrail_dashboard_use_live_overrides": False,
                "ops_degradation_guardrail_false_positive_target_max": 0.50,
                "ops_degradation_guardrail_burnin_light_backfill_enabled": False,
                "ops_degradation_guardrail_burnin_autofill_review_if_missing": False,
                "ops_degradation_guardrail_burnin_low_cost_replay_enabled": True,
                "ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run": 3,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        def _stable_replay(as_of: date, days: int | None, run_eod_replay: bool = True) -> dict[str, object]:
            day_tag = as_of.isoformat()
            replayed_days.add(day_tag)
            return {
                "as_of": day_tag,
                "replay_days": int(days or 1),
                "replay_executed": bool(run_eod_replay),
                "passed": True,
                "checks": [{"date": day_tag, "ok": True}],
            }

        def _fake_gate(
            self: ReleaseOrchestrator,
            as_of: date,
            run_tests: bool = False,
            run_review_if_missing: bool = False,
            run_stable_replay: bool = False,
        ) -> dict[str, object]:
            day_tag = as_of.isoformat()
            sample_ready = day_tag in replayed_days
            return {
                "passed": True,
                "checks": {},
                "degradation_guardrail_dashboard": {
                    "active": bool(sample_ready),
                    "samples": 1 if sample_ready else 0,
                    "monitor_failed": False,
                    "metrics": {
                        "cooldown_hit_rate": 0.05,
                        "suppressed_trigger_density": 0.02,
                        "promotion_latency_avg_days": 1.0,
                    },
                    "thresholds": {},
                },
            }

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=_stable_replay,
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_degradation_calibration=None,
        )
        with patch.object(ReleaseOrchestrator, "gate_report", new=_fake_gate):
            out = orch.degradation_guardrail_burnin(
                as_of=d,
                days=3,
                run_stable_replay=False,
                auto_tune=False,
            )
        summary = out.get("summary", {}) if isinstance(out.get("summary", {}), dict) else {}
        coverage = out.get("coverage", {}) if isinstance(out.get("coverage", {}), dict) else {}
        self.assertEqual(int(summary.get("active_days", 0)), 3)
        self.assertTrue(bool(summary.get("active_days_ok", False)))
        self.assertTrue(bool(summary.get("rollback_artifact_days_ok", False)))
        self.assertTrue(bool(summary.get("coverage_ok", False)))
        self.assertEqual(int(coverage.get("low_cost_replay_attempted_days", 0)), 3)
        self.assertEqual(int(coverage.get("low_cost_replay_succeeded_days", 0)), 3)
        self.assertEqual(int(coverage.get("low_cost_replay_failed_days", 0)), 0)
        self.assertEqual(int(coverage.get("low_cost_replay_budget_skipped_days", 0)), 0)

    def test_degradation_guardrail_burnin_budget_audit_auto_tunes_low_cost_replay_budget(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        replayed_days: set[str] = set()
        live_path = td / "artifacts" / "degradation_guardrail_burnin_live.yaml"

        for i in range(3):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": False,
                        "triggered": False,
                        "applied": False,
                        "reason": "not_triggered",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 3,
                "ops_degradation_guardrail_dashboard_use_live_overrides": False,
                "ops_degradation_guardrail_false_positive_target_max": 0.50,
                "ops_degradation_guardrail_burnin_light_backfill_enabled": False,
                "ops_degradation_guardrail_burnin_autofill_review_if_missing": False,
                "ops_degradation_guardrail_burnin_low_cost_replay_enabled": True,
                "ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run": 2,
                "ops_degradation_guardrail_burnin_use_live_overrides": True,
                "ops_degradation_guardrail_burnin_live_params_path": "artifacts/degradation_guardrail_burnin_live.yaml",
                "ops_degradation_guardrail_burnin_budget_audit_enabled": True,
                "ops_degradation_guardrail_burnin_budget_audit_auto_tune": True,
                "ops_degradation_guardrail_burnin_budget_audit_expand_recovery_ratio_min": 0.70,
                "ops_degradation_guardrail_burnin_budget_audit_shrink_recovery_ratio_max": 0.40,
                "ops_degradation_guardrail_burnin_budget_audit_step_days": 1,
                "ops_degradation_guardrail_burnin_budget_audit_min_days": 0,
                "ops_degradation_guardrail_burnin_budget_audit_max_days": 4,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        def _stable_replay(as_of: date, days: int | None, run_eod_replay: bool = True) -> dict[str, object]:
            day_tag = as_of.isoformat()
            replayed_days.add(day_tag)
            return {
                "as_of": day_tag,
                "replay_days": int(days or 1),
                "replay_executed": bool(run_eod_replay),
                "passed": True,
                "checks": [{"date": day_tag, "ok": True}],
            }

        def _fake_gate(
            self: ReleaseOrchestrator,
            as_of: date,
            run_tests: bool = False,
            run_review_if_missing: bool = False,
            run_stable_replay: bool = False,
        ) -> dict[str, object]:
            day_tag = as_of.isoformat()
            sample_ready = day_tag in replayed_days
            return {
                "passed": True,
                "checks": {},
                "degradation_guardrail_dashboard": {
                    "active": bool(sample_ready),
                    "samples": 1 if sample_ready else 0,
                    "monitor_failed": False,
                    "metrics": {
                        "cooldown_hit_rate": 0.05,
                        "suppressed_trigger_density": 0.02,
                        "promotion_latency_avg_days": 1.0,
                    },
                    "thresholds": {},
                },
            }

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=_stable_replay,
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_degradation_calibration=None,
        )
        with patch.object(ReleaseOrchestrator, "gate_report", new=_fake_gate):
            out = orch.degradation_guardrail_burnin(
                as_of=d,
                days=3,
                run_stable_replay=False,
                auto_tune=False,
            )

        audit = (
            out.get("low_cost_replay_budget_audit", {})
            if isinstance(out.get("low_cost_replay_budget_audit", {}), dict)
            else {}
        )
        self.assertEqual(int(audit.get("effective_max_days_per_run", 0)), 2)
        self.assertEqual(int(audit.get("recommended_max_days_per_run", 0)), 3)
        self.assertTrue(bool(audit.get("tune_applied", False)))
        self.assertEqual(str(audit.get("reason", "")), "expand_high_recovery_under_pressure")
        self.assertTrue(live_path.exists())
        live_payload = yaml.safe_load(live_path.read_text(encoding="utf-8")) or {}
        params = live_payload.get("params", {}) if isinstance(live_payload.get("params", {}), dict) else {}
        self.assertEqual(
            int(params.get("ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run", -1)),
            3,
        )

    def test_degradation_guardrail_burnin_uses_low_cost_replay_live_override(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        replay_calls: list[str] = []
        live_path = td / "artifacts" / "degradation_guardrail_burnin_live.yaml"
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "params": {
                        "ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run": 1,
                    },
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        for i in range(3):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": False,
                        "triggered": False,
                        "applied": False,
                        "reason": "not_triggered",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": False,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 3,
                "ops_degradation_guardrail_dashboard_use_live_overrides": False,
                "ops_degradation_guardrail_false_positive_target_max": 0.50,
                "ops_degradation_guardrail_burnin_light_backfill_enabled": False,
                "ops_degradation_guardrail_burnin_autofill_review_if_missing": False,
                "ops_degradation_guardrail_burnin_low_cost_replay_enabled": True,
                "ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run": 3,
                "ops_degradation_guardrail_burnin_use_live_overrides": True,
                "ops_degradation_guardrail_burnin_live_params_path": "artifacts/degradation_guardrail_burnin_live.yaml",
                "ops_degradation_guardrail_burnin_budget_audit_enabled": False,
            }
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        def _stable_replay(as_of: date, days: int | None, run_eod_replay: bool = True) -> dict[str, object]:
            day_tag = as_of.isoformat()
            replay_calls.append(day_tag)
            return {
                "as_of": day_tag,
                "replay_days": int(days or 1),
                "replay_executed": bool(run_eod_replay),
                "passed": True,
                "checks": [{"date": day_tag, "ok": True}],
            }

        def _fake_gate(
            self: ReleaseOrchestrator,
            as_of: date,
            run_tests: bool = False,
            run_review_if_missing: bool = False,
            run_stable_replay: bool = False,
        ) -> dict[str, object]:
            return {
                "passed": True,
                "checks": {},
                "degradation_guardrail_dashboard": {
                    "active": False,
                    "samples": 0,
                    "monitor_failed": False,
                    "metrics": {
                        "cooldown_hit_rate": 0.05,
                        "suppressed_trigger_density": 0.02,
                        "promotion_latency_avg_days": 1.0,
                    },
                    "thresholds": {},
                },
            }

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=_stable_replay,
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
            run_degradation_calibration=None,
        )
        with patch.object(ReleaseOrchestrator, "gate_report", new=_fake_gate):
            out = orch.degradation_guardrail_burnin(
                as_of=d,
                days=3,
                run_stable_replay=False,
                auto_tune=False,
            )
        coverage = out.get("coverage", {}) if isinstance(out.get("coverage", {}), dict) else {}
        burnin_live = (
            out.get("burnin_live_overrides", {})
            if isinstance(out.get("burnin_live_overrides", {}), dict)
            else {}
        )
        audit = (
            out.get("low_cost_replay_budget_audit", {})
            if isinstance(out.get("low_cost_replay_budget_audit", {}), dict)
            else {}
        )
        self.assertTrue(bool(burnin_live.get("applied", False)))
        self.assertEqual(int(coverage.get("low_cost_replay_max_days_per_run", 0)), 1)
        self.assertEqual(int(coverage.get("low_cost_replay_attempted_days", 0)), 1)
        self.assertEqual(len(replay_calls), 1)
        self.assertEqual(int(audit.get("configured_max_days_per_run", 0)), 3)
        self.assertEqual(int(audit.get("effective_max_days_per_run", 0)), 1)

    def test_compact_executed_plans_duplicates_dry_run_keeps_rows(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        d1 = date(2026, 2, 11)
        d2 = date(2026, 2, 12)

        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE executed_plans (date TEXT, symbol TEXT, side TEXT, pnl REAL, status TEXT)"
            )
            rows = [
                (d1.isoformat(), "000001", "LONG", 0.01, "CLOSED"),
                (d1.isoformat(), "000001", "LONG", 0.01, "CLOSED"),
                (d1.isoformat(), "000002", "LONG", 0.02, "CLOSED"),
                (d2.isoformat(), "000003", "SHORT", -0.03, "CLOSED"),
                (d2.isoformat(), "000003", "SHORT", -0.03, "CLOSED"),
            ]
            conn.executemany(
                "INSERT INTO executed_plans (date, symbol, side, pnl, status) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
            sqlite_path=db_path,
        )

        out = orch.compact_executed_plans_duplicates(
            start=d1,
            end=d2,
            chunk_days=1,
            dry_run=True,
        )
        metrics = out.get("metrics", {}) if isinstance(out.get("metrics", {}), dict) else {}
        self.assertEqual(str(out.get("status", "")), "ok")
        self.assertEqual(int(metrics.get("duplicate_rows_found", 0)), 2)
        self.assertEqual(int(metrics.get("deleted_rows", 0)), 0)
        self.assertFalse(bool((out.get("rollback", {}) if isinstance(out.get("rollback", {}), dict) else {}).get("available", False)))
        self.assertTrue(Path(str((out.get("paths", {}) if isinstance(out.get("paths", {}), dict) else {}).get("json", ""))).exists())
        with closing(sqlite3.connect(db_path)) as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM executed_plans").fetchone()[0])
        self.assertEqual(count, 5)

    def test_compact_executed_plans_duplicates_apply_with_cap_and_rollback_artifacts(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        d1 = date(2026, 2, 11)
        d2 = date(2026, 2, 12)

        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE executed_plans (date TEXT, symbol TEXT, side TEXT, pnl REAL, status TEXT)"
            )
            rows = [
                (d1.isoformat(), "000001", "LONG", 0.01, "CLOSED"),
                (d1.isoformat(), "000001", "LONG", 0.01, "CLOSED"),
                (d1.isoformat(), "000001", "LONG", 0.01, "CLOSED"),
                (d2.isoformat(), "000003", "SHORT", -0.03, "CLOSED"),
                (d2.isoformat(), "000003", "SHORT", -0.03, "CLOSED"),
                (d2.isoformat(), "000004", "SHORT", -0.01, "CLOSED"),
            ]
            conn.executemany(
                "INSERT INTO executed_plans (date, symbol, side, pnl, status) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
            sqlite_path=db_path,
        )

        out = orch.compact_executed_plans_duplicates(
            start=d1,
            end=d2,
            chunk_days=1,
            dry_run=False,
            max_delete_rows=2,
        )
        metrics = out.get("metrics", {}) if isinstance(out.get("metrics", {}), dict) else {}
        rollback = out.get("rollback", {}) if isinstance(out.get("rollback", {}), dict) else {}
        self.assertEqual(str(out.get("status", "")), "ok")
        self.assertEqual(int(metrics.get("deleted_rows", 0)), 2)
        self.assertTrue(bool(metrics.get("cap_reached", False)))
        self.assertTrue(bool(rollback.get("available", False)))

        backup_db = Path(str(rollback.get("backup_db_path", "")))
        rollback_sql = Path(str(rollback.get("rollback_sql_path", "")))
        self.assertTrue(backup_db.exists())
        self.assertTrue(rollback_sql.exists())
        with closing(sqlite3.connect(backup_db)) as conn:
            count = int(
                conn.execute("SELECT COUNT(*) FROM executed_plans_deleted").fetchone()[0]
            )
        self.assertEqual(count, 2)
        with closing(sqlite3.connect(db_path)) as conn:
            remaining = int(conn.execute("SELECT COUNT(*) FROM executed_plans").fetchone()[0])
        self.assertEqual(remaining, 4)

    def test_verify_executed_plans_compaction_restore_validates_delta(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        d1 = date(2026, 2, 11)

        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE executed_plans (date TEXT, symbol TEXT, side TEXT, pnl REAL, status TEXT)"
            )
            rows = [
                (d1.isoformat(), "000001", "LONG", 0.01, "CLOSED"),
                (d1.isoformat(), "000001", "LONG", 0.01, "CLOSED"),
                (d1.isoformat(), "000001", "LONG", 0.01, "CLOSED"),
                (d1.isoformat(), "000002", "SHORT", -0.02, "CLOSED"),
            ]
            conn.executemany(
                "INSERT INTO executed_plans (date, symbol, side, pnl, status) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
            sqlite_path=db_path,
        )

        compact = orch.compact_executed_plans_duplicates(
            start=d1,
            end=d1,
            chunk_days=1,
            dry_run=False,
        )
        run_id = str(compact.get("run_id", ""))
        self.assertTrue(bool(run_id))

        verify = orch.verify_executed_plans_compaction_restore(run_id=run_id, keep_temp_db=False)
        checks = verify.get("checks", {}) if isinstance(verify.get("checks", {}), dict) else {}
        metrics = verify.get("metrics", {}) if isinstance(verify.get("metrics", {}), dict) else {}
        self.assertEqual(str(verify.get("status", "")), "ok")
        self.assertTrue(bool(checks.get("restore_delta_match", False)))
        self.assertGreaterEqual(int(metrics.get("backup_rows", 0)), 1)
        self.assertEqual(int(metrics.get("restored_rows_delta", 0)), int(metrics.get("backup_rows", 0)))
        self.assertTrue(Path(str((verify.get("paths", {}) if isinstance(verify.get("paths", {}), dict) else {}).get("json", ""))).exists())
        self.assertTrue(Path(str((verify.get("paths", {}) if isinstance(verify.get("paths", {}), dict) else {}).get("md", ""))).exists())

    def test_degradation_guardrail_threshold_drift_audit_outputs_artifact_and_alerts(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        artifacts_dir = td / "artifacts"
        review_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        for i in range(4):
            day = d - timedelta(days=i)
            burnin_path = review_dir / f"{day.isoformat()}_degradation_guardrail_burnin.json"
            burnin_path.write_text(
                json.dumps(
                    {
                        "live_overrides": {
                            "payload": {
                                "params": {
                                    "ops_degradation_guardrail_cooldown_hit_rate_max": 0.55,
                                    "ops_degradation_guardrail_suppressed_trigger_density_max": 0.45,
                                    "ops_degradation_guardrail_promotion_latency_days_max": 16,
                                },
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        (artifacts_dir / "degradation_guardrail_dashboard_live.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "as_of": d.isoformat(),
                    "params": {
                        "ops_degradation_guardrail_cooldown_hit_rate_max": 0.95,
                        "ops_degradation_guardrail_suppressed_trigger_density_max": 0.85,
                        "ops_degradation_guardrail_promotion_latency_days_max": 30,
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
                "ops_degradation_guardrail_dashboard_live_params_path": "artifacts/degradation_guardrail_dashboard_live.yaml",
                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.60,
                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.50,
                "ops_degradation_guardrail_promotion_latency_days_max": 20,
                "ops_degradation_guardrail_threshold_drift_warn_ratio": 0.10,
                "ops_degradation_guardrail_threshold_drift_critical_ratio": 0.20,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": int(days), "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.degradation_guardrail_threshold_drift_audit(as_of=d, window_days=21)
        self.assertEqual(str(out.get("status", "")), "critical")
        self.assertIn("paths", out)
        paths = out.get("paths", {}) if isinstance(out.get("paths", {}), dict) else {}
        self.assertTrue(Path(str(paths.get("json", ""))).exists())
        self.assertTrue(Path(str(paths.get("md", ""))).exists())
        alerts = set(out.get("alerts", [])) if isinstance(out.get("alerts", []), list) else set()
        self.assertIn(
            "degradation_guardrail_threshold_drift_critical:ops_degradation_guardrail_cooldown_hit_rate_max",
            alerts,
        )
        params = out.get("params", {}) if isinstance(out.get("params", {}), dict) else {}
        self.assertEqual(
            str((params.get("ops_degradation_guardrail_promotion_latency_days_max", {}) or {}).get("severity", "")),
            "critical",
        )

    def test_degradation_guardrail_threshold_drift_audit_flags_insufficient_burnin_samples(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        (review_dir / f"{d.isoformat()}_degradation_guardrail_burnin.json").write_text(
            json.dumps(
                {
                    "live_overrides": {
                        "payload": {
                            "params": {
                                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.80,
                                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.60,
                                "ops_degradation_guardrail_promotion_latency_days_max": 21,
                            },
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_threshold_drift_warn_ratio": 0.25,
                "ops_degradation_guardrail_threshold_drift_critical_ratio": 0.40,
                "ops_degradation_guardrail_threshold_drift_min_burnin_samples": 3,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": int(days), "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.degradation_guardrail_threshold_drift_audit(as_of=d, window_days=21)
        self.assertEqual(str(out.get("status", "")), "insufficient_samples")
        summary = out.get("summary", {}) if isinstance(out.get("summary", {}), dict) else {}
        self.assertEqual(int(summary.get("burnin_samples", 0)), 1)
        self.assertEqual(int(summary.get("burnin_min_samples", 0)), 3)
        self.assertFalse(bool(summary.get("burnin_samples_ok", True)))
        alerts = out.get("alerts", []) if isinstance(out.get("alerts", []), list) else []
        self.assertIn("degradation_guardrail_threshold_drift_insufficient_burnin_samples", alerts)

    def test_degradation_guardrail_threshold_drift_audit_uses_daily_rows_as_samples(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(5):
            day = d - timedelta(days=i)
            rows.append(
                {
                    "date": day.isoformat(),
                    "thresholds": {
                        "ops_degradation_guardrail_cooldown_hit_rate_max": 0.80,
                        "ops_degradation_guardrail_suppressed_trigger_density_max": 0.60,
                        "ops_degradation_guardrail_promotion_latency_days_max": 21,
                    },
                }
            )
        (review_dir / f"{d.isoformat()}_degradation_guardrail_burnin.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "live_overrides": {
                        "payload": {
                            "params": {
                                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.80,
                                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.60,
                                "ops_degradation_guardrail_promotion_latency_days_max": 21,
                            }
                        }
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
                "ops_degradation_guardrail_threshold_drift_warn_ratio": 0.25,
                "ops_degradation_guardrail_threshold_drift_critical_ratio": 0.40,
                "ops_degradation_guardrail_threshold_drift_min_burnin_samples": 5,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": int(days), "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.degradation_guardrail_threshold_drift_audit(as_of=d, window_days=21)
        summary = out.get("summary", {}) if isinstance(out.get("summary", {}), dict) else {}
        self.assertEqual(int(summary.get("burnin_samples", 0)), 5)
        self.assertTrue(bool(summary.get("burnin_samples_ok", False)))
        self.assertEqual(str(out.get("status", "")), "ok")

    def test_gate_report_threshold_drift_autofix_runs_when_artifact_missing(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (review_dir / f"{d.isoformat()}_degradation_guardrail_burnin.json").write_text(
            json.dumps(
                {
                    "live_overrides": {
                        "payload": {
                            "params": {
                                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.8,
                                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.6,
                                "ops_degradation_guardrail_promotion_latency_days_max": 21,
                            }
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_threshold_drift_enabled": True,
                "ops_degradation_guardrail_threshold_drift_gate_hard_fail": True,
                "ops_degradation_guardrail_threshold_drift_require_active": True,
                "ops_degradation_guardrail_threshold_drift_max_staleness_days": 14,
                "ops_degradation_guardrail_threshold_drift_min_burnin_samples": 1,
                "ops_degradation_guardrail_threshold_drift_autofix_enabled": True,
                "ops_degradation_guardrail_threshold_drift_autofix_on_missing": True,
                "ops_degradation_guardrail_threshold_drift_autofix_on_stale": True,
                "ops_degradation_guardrail_threshold_drift_autofix_window_days": 21,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False, run_stable_replay=False)
        self.assertTrue(bool(out.get("checks", {}).get("degradation_guardrail_threshold_drift_ok", False)))
        payload = out.get("degradation_guardrail_threshold_drift", {})
        autofix = payload.get("autofix", {}) if isinstance(payload.get("autofix", {}), dict) else {}
        self.assertTrue(bool(autofix.get("attempted", False)))
        self.assertTrue(bool(autofix.get("applied", False)))
        self.assertEqual(str(autofix.get("reason", "")), "missing_artifact")
        self.assertTrue(bool(payload.get("active", False)))
        self.assertEqual(str(payload.get("status", "")), "ok")

    def test_gate_report_snapshot_chain_gate_passes_with_valid_checksum_chain(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_dir = td / "artifacts" / "baselines" / "degradation_calibration" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        active_path = td / "artifacts" / "baselines" / "degradation_calibration" / "active_snapshot.yaml"
        prev_path = history_dir / "2026-02-12_000001.yaml"
        current_path = history_dir / "2026-02-13_000001.yaml"

        def _chain_checksum(params: dict[str, float], anchor: str) -> str:
            payload = {
                "params": params,
                "rollback_anchor": anchor,
            }
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return hashlib.sha1(canonical.encode("utf-8")).hexdigest()

        prev_params = {
            "ops_slot_degradation_soft_multiplier": 1.30,
            "ops_state_degradation_soft_multiplier": 1.25,
        }
        prev_payload = {
            "schema_version": 1,
            "as_of": "2026-02-12",
            "promoted_at": "2026-02-12T20:30:00",
            "params": prev_params,
            "snapshot_path": str(prev_path),
            "history_path": str(prev_path),
            "rollback_anchor": "",
            "params_checksum": hashlib.sha1(
                json.dumps(prev_params, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "chain_checksum": _chain_checksum(prev_params, ""),
        }
        prev_path.write_text(yaml.safe_dump(prev_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

        current_params = {
            "ops_slot_degradation_soft_multiplier": 1.45,
            "ops_state_degradation_soft_multiplier": 1.35,
        }
        current_payload = {
            "schema_version": 1,
            "as_of": "2026-02-13",
            "promoted_at": "2026-02-13T20:30:00",
            "params": current_params,
            "snapshot_path": str(current_path),
            "history_path": str(current_path),
            "rollback_anchor": str(prev_path),
            "params_checksum": hashlib.sha1(
                json.dumps(current_params, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "chain_checksum": _chain_checksum(current_params, str(prev_path)),
        }
        current_path.write_text(yaml.safe_dump(current_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(yaml.safe_dump(current_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_snapshot_chain_gate_enabled": True,
                "ops_snapshot_chain_gate_hard_fail": True,
                "ops_snapshot_chain_gate_require_active": True,
                "ops_snapshot_chain_gate_require_checksum": True,
                "ops_snapshot_chain_gate_require_history_alignment": True,
                "ops_snapshot_chain_gate_max_depth": 6,
                "ops_degradation_calibration_rollback_active_snapshot_path": "artifacts/baselines/degradation_calibration/active_snapshot.yaml",
                "ops_degradation_calibration_rollback_history_dir": "artifacts/baselines/degradation_calibration/history",
            }
        )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(bool(out.get("checks", {}).get("snapshot_chain_ok", False)))
        self.assertTrue(bool(out.get("passed", False)))
        snapshot_chain = out.get("snapshot_chain", {})
        self.assertTrue(bool(snapshot_chain.get("active", False)))
        self.assertFalse(bool(snapshot_chain.get("monitor_failed", True)))
        self.assertEqual(int((snapshot_chain.get("metrics", {}) or {}).get("chain_depth", 0)), 2)

    def test_gate_report_snapshot_chain_gate_blocks_on_checksum_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_dir = td / "artifacts" / "baselines" / "degradation_calibration" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        active_path = td / "artifacts" / "baselines" / "degradation_calibration" / "active_snapshot.yaml"
        prev_path = history_dir / "2026-02-12_000001.yaml"
        current_path = history_dir / "2026-02-13_000001.yaml"

        prev_params = {"ops_slot_degradation_soft_multiplier": 1.30}
        prev_payload = {
            "schema_version": 1,
            "as_of": "2026-02-12",
            "promoted_at": "2026-02-12T20:30:00",
            "params": prev_params,
            "snapshot_path": str(prev_path),
            "history_path": str(prev_path),
            "rollback_anchor": "",
            "params_checksum": hashlib.sha1(
                json.dumps(prev_params, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "chain_checksum": "bad-chain-checksum",
        }
        prev_path.write_text(yaml.safe_dump(prev_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        current_path.write_text(yaml.safe_dump(prev_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(yaml.safe_dump(prev_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_snapshot_chain_gate_enabled": True,
                "ops_snapshot_chain_gate_hard_fail": True,
                "ops_snapshot_chain_gate_require_active": True,
                "ops_snapshot_chain_gate_require_checksum": True,
                "ops_snapshot_chain_gate_max_depth": 6,
                "ops_degradation_calibration_rollback_active_snapshot_path": "artifacts/baselines/degradation_calibration/active_snapshot.yaml",
                "ops_degradation_calibration_rollback_history_dir": "artifacts/baselines/degradation_calibration/history",
            }
        )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out.get("checks", {}).get("snapshot_chain_ok", True)))
        self.assertFalse(bool(out.get("passed", True)))
        snapshot_chain = out.get("snapshot_chain", {})
        self.assertTrue(bool(snapshot_chain.get("monitor_failed", False)))
        self.assertIn("snapshot_chain_checksum_failed", set(snapshot_chain.get("alerts", [])))
        rollback = out.get("rollback_recommendation", {})
        self.assertIn("snapshot_chain", set(rollback.get("reason_codes", [])))

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

    def test_gate_report_style_drift_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            yaml.safe_dump(
                {
                    "pass_gate": True,
                    "mode_health": {"passed": True},
                    "style_diagnostics": {
                        "active": True,
                        "style_drift_score": 0.03,
                        "drift_gap_max": 0.01,
                        "alerts": ["style_drift:momentum"],
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
                "style_drift_gate_enabled": True,
                "style_drift_gate_hard_fail": True,
                "style_drift_gate_require_active": True,
                "style_drift_gate_allow_alerts": False,
                "style_drift_gate_max_ratio": 2.0,
                "style_drift_gate_max_alerts": 0,
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
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"]["style_drift_ok"]))
        self.assertFalse(bool(out["passed"]))
        self.assertIn("style_drift_gate_ratio_high", set(out.get("style_drift", {}).get("alerts", [])))

    def test_gate_report_style_drift_monitor_mode_does_not_block_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            yaml.safe_dump(
                {
                    "pass_gate": True,
                    "mode_health": {"passed": True},
                    "style_diagnostics": {
                        "active": True,
                        "style_drift_score": 0.03,
                        "drift_gap_max": 0.01,
                        "alerts": ["style_drift:momentum"],
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
                "style_drift_gate_enabled": True,
                "style_drift_gate_hard_fail": False,
                "style_drift_gate_require_active": True,
                "style_drift_gate_allow_alerts": False,
                "style_drift_gate_max_ratio": 2.0,
                "style_drift_gate_max_alerts": 0,
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
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(bool(out["checks"]["style_drift_ok"]))
        self.assertTrue(bool(out["passed"]))
        self.assertTrue(bool(out.get("style_drift", {}).get("monitor_failed", False)))

    def test_gate_report_artifact_governance_legacy_drift_alert(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_reconcile_broker_row_diff_artifact_retention_days": 2,
                "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled": True,
                "ops_artifact_governance_profiles": {
                    "reconcile_row_diff": {
                        "json_glob": "*_reconcile_row_diff.json",
                        "md_glob": "*_reconcile_row_diff.md",
                        "checksum_index_filename": "reconcile_row_diff_checksum_index.json",
                        "retention_days": 5,
                        "checksum_index_enabled": False,
                    }
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
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(bool(out["checks"]["artifact_governance_ok"]))
        artifact_governance = out.get("artifact_governance", {})
        self.assertIn("artifact_governance_legacy_policy_drift", set(artifact_governance.get("alerts", [])))
        self.assertGreaterEqual(
            int((artifact_governance.get("metrics", {}) or {}).get("legacy_policy_drift_profiles", 0)),
            1,
        )

    def test_gate_report_artifact_governance_strict_mode_blocks_baseline_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_artifact_governance_strict_mode_enabled": True,
                "ops_reconcile_broker_row_diff_artifact_retention_days": 5,
                "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled": False,
                "ops_artifact_governance_profiles": {
                    "reconcile_row_diff": {
                        "json_glob": "*_reconcile_row_diff.json",
                        "md_glob": "*_reconcile_row_diff.md",
                        "checksum_index_filename": "reconcile_row_diff_checksum_index.json",
                        "retention_days": 5,
                        "checksum_index_enabled": False,
                    }
                },
                "ops_artifact_governance_profile_baseline": {
                    "reconcile_row_diff": {
                        "retention_days": 30,
                    }
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
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["passed"]))
        self.assertFalse(bool(out["checks"]["artifact_governance_ok"]))
        artifact_governance = out.get("artifact_governance", {})
        alerts = set(artifact_governance.get("alerts", []))
        self.assertIn("artifact_governance_baseline_drift", alerts)
        self.assertIn("artifact_governance_strict_mode_blocked", alerts)
        checks = artifact_governance.get("checks", {}) if isinstance(artifact_governance.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("baseline_freeze_ok", True)))
        self.assertFalse(bool(checks.get("strict_mode_ok", True)))
        metrics = (
            artifact_governance.get("metrics", {}) if isinstance(artifact_governance.get("metrics", {}), dict) else {}
        )
        self.assertTrue(bool(metrics.get("strict_mode_enabled", False)))
        self.assertTrue(bool(metrics.get("strict_mode_blocked", False)))
        self.assertGreaterEqual(int(metrics.get("baseline_drift_profiles", 0)), 1)

    def test_gate_report_artifact_governance_uses_snapshot_baseline(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        active_baseline = td / "artifacts" / "baselines" / "artifact_governance" / "active_baseline.yaml"
        active_baseline.parent.mkdir(parents=True, exist_ok=True)
        active_baseline.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "as_of": "2026-02-12",
                    "profiles": {
                        "reconcile_row_diff": {
                            "json_glob": "*_reconcile_row_diff.json",
                            "md_glob": "*_reconcile_row_diff.md",
                            "checksum_index_filename": "reconcile_row_diff_checksum_index.json",
                            "retention_days": 5,
                            "checksum_index_enabled": False,
                        }
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
                "ops_artifact_governance_strict_mode_enabled": True,
                "ops_artifact_governance_baseline_snapshot_enabled": True,
                "ops_artifact_governance_baseline_snapshot_path": "artifacts/baselines/artifact_governance/active_baseline.yaml",
                "ops_reconcile_broker_row_diff_artifact_retention_days": 5,
                "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled": False,
                "ops_artifact_governance_profiles": {
                    "reconcile_row_diff": {
                        "json_glob": "*_reconcile_row_diff.json",
                        "md_glob": "*_reconcile_row_diff.md",
                        "checksum_index_filename": "reconcile_row_diff_checksum_index.json",
                        "retention_days": 5,
                        "checksum_index_enabled": False,
                    }
                },
                "ops_artifact_governance_profile_baseline": {
                    "reconcile_row_diff": {
                        "retention_days": 30,
                    }
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
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(bool(out["passed"]))
        self.assertTrue(bool(out["checks"]["artifact_governance_ok"]))
        baseline = out.get("artifact_governance", {}).get("baseline", {})
        self.assertEqual(str(baseline.get("source", "")), "snapshot")
        snapshot = baseline.get("snapshot", {})
        self.assertTrue(bool(snapshot.get("found", False)))
        self.assertEqual(str(snapshot.get("path", "")), str(active_baseline))

    def test_review_until_pass_promotes_artifact_governance_baseline_snapshot(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_artifact_governance_baseline_snapshot_enabled": True,
                "ops_artifact_governance_baseline_auto_promote_on_review_pass": True,
                "ops_artifact_governance_baseline_snapshot_path": "artifacts/baselines/artifact_governance/active_baseline.yaml",
                "ops_artifact_governance_baseline_history_dir": "artifacts/baselines/artifact_governance/history",
            }
        )

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True)

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertTrue(bool(out["passed"]))
        promotion = out.get("artifact_governance_baseline_promotion", {})
        self.assertTrue(bool(promotion.get("promoted", False)))
        active_path = Path(str(promotion.get("active_path", "")))
        history_path = Path(str(promotion.get("history_path", "")))
        self.assertTrue(active_path.exists())
        self.assertTrue(history_path.exists())
        active_payload = yaml.safe_load(active_path.read_text(encoding="utf-8")) or {}
        self.assertEqual(str(active_payload.get("snapshot_path", "")), str(history_path))
        release_payload = json.loads((td / "artifacts" / f"release_ready_{d.isoformat()}.json").read_text(encoding="utf-8"))
        self.assertIn("artifact_governance_baseline_promotion", release_payload)
        self.assertIn("artifact_governance_baseline_promotion_path", release_payload)

    def test_review_until_pass_baseline_promotion_carries_rollback_anchor(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        history_dir = td / "artifacts" / "baselines" / "artifact_governance" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        previous_history = history_dir / "2026-02-12_r01_000000.yaml"
        previous_history.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "as_of": "2026-02-12",
                    "profiles": {"reconcile_row_diff": {"retention_days": 30}},
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        active_baseline = td / "artifacts" / "baselines" / "artifact_governance" / "active_baseline.yaml"
        active_baseline.parent.mkdir(parents=True, exist_ok=True)
        active_baseline.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "as_of": "2026-02-12",
                    "snapshot_path": str(previous_history),
                    "history_path": str(previous_history),
                    "rollback_anchor": "",
                    "profiles": {"reconcile_row_diff": {"retention_days": 30}},
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
                "ops_artifact_governance_baseline_snapshot_enabled": True,
                "ops_artifact_governance_baseline_auto_promote_on_review_pass": True,
                "ops_artifact_governance_baseline_snapshot_path": "artifacts/baselines/artifact_governance/active_baseline.yaml",
                "ops_artifact_governance_baseline_history_dir": "artifacts/baselines/artifact_governance/history",
            }
        )

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True)

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertTrue(bool(out["passed"]))
        promotion = out.get("artifact_governance_baseline_promotion", {})
        self.assertEqual(str(promotion.get("rollback_anchor", "")), str(previous_history))
        active_payload = yaml.safe_load(active_baseline.read_text(encoding="utf-8")) or {}
        self.assertEqual(str(active_payload.get("rollback_anchor", "")), str(previous_history))

    def test_review_until_pass_defect_plan_marks_artifact_governance_strict_block(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_artifact_governance_strict_mode_enabled": True,
                "ops_reconcile_broker_row_diff_artifact_retention_days": 5,
                "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled": False,
                "ops_artifact_governance_profiles": {
                    "reconcile_row_diff": {
                        "json_glob": "*_reconcile_row_diff.json",
                        "md_glob": "*_reconcile_row_diff.md",
                        "checksum_index_filename": "reconcile_row_diff_checksum_index.json",
                        "retention_days": 5,
                        "checksum_index_enabled": False,
                    }
                },
                "ops_artifact_governance_profile_baseline": {
                    "reconcile_row_diff": {
                        "retention_days": 30,
                    }
                },
            }
        )

        def _run_review(as_of: date) -> ReviewDelta:
            (review_dir / f"{as_of.isoformat()}_param_delta.yaml").write_text(
                "pass_gate: true\nmode_health:\n  passed: true\n",
                encoding="utf-8",
            )
            return ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True)

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=_run_review,
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(bool(out["passed"]))
        defect_plan = out["rounds"][0].get("defect_plan", {})
        plan_json = Path(str(defect_plan.get("json", "")))
        self.assertTrue(plan_json.exists())
        payload = json.loads(plan_json.read_text(encoding="utf-8"))
        defects = payload.get("defects", []) if isinstance(payload.get("defects", []), list) else []
        codes = {str(item.get("code", "")) for item in defects if isinstance(item, dict)}
        self.assertIn("ARTIFACT_GOVERNANCE_BASELINE_DRIFT", codes)
        self.assertIn("ARTIFACT_GOVERNANCE_STRICT_BLOCKED", codes)

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

    def test_ops_report_uses_halfhour_state_and_skips_stable_replay_side_effect(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "halfhour_pulse_state.json").write_text(
            json.dumps(
                {
                    "date": d.isoformat(),
                    "executed_slots": ["premarket", "intraday:10:30"],
                    "history": [{"pulse_slot": "11:00"}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "halfhour_daemon_state.json").write_text(
            json.dumps(
                {
                    "date": d.isoformat(),
                    "history": [{"bucket": "11:00"}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        stable_calls: list[tuple[date, int | None]] = []

        def _stable_replay(as_of: date, days: int | None) -> dict[str, object]:
            stable_calls.append((as_of, days))
            return {"passed": True, "replay_days": 3, "checks": []}

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=_stable_replay,
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=1)
        self.assertEqual(stable_calls, [])
        scheduler = out.get("scheduler", {})
        self.assertEqual(str(scheduler.get("date", "")), d.isoformat())
        self.assertIn("premarket", set(scheduler.get("executed_slots", [])))
        self.assertIn("intraday:10:30", set(scheduler.get("executed_slots", [])))
        sources = scheduler.get("state_sources", {})
        self.assertTrue(bool(sources.get("halfhour_pulse", False)))
        self.assertTrue(bool(sources.get("halfhour_daemon", False)))

    def test_ops_report_state_switch_dwell_filter_suppresses_one_day_flicker(self) -> None:
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

        modes = ["swing", "ultra_short", "swing", "ultra_short", "swing"]
        for i in range(5):
            day = date.fromordinal(d.toordinal() - (4 - i))
            payload = {
                "runtime_mode": modes[i],
                "mode_health": {"passed": True},
                "risk_control": {
                    "risk_multiplier": 1.0,
                    "source_confidence_score": 0.90,
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
                "mode_switch_max_rate": 0.20,
                "ops_state_mode_switch_min_dwell_days": 2,
                "ops_risk_multiplier_floor": 0.20,
                "ops_risk_multiplier_drift_max": 1.0,
                "ops_source_confidence_floor": 0.50,
                "ops_mode_health_fail_days_max": 5,
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
        state = out["state_stability"]
        checks = state.get("checks", {}) if isinstance(state.get("checks", {}), dict) else {}
        metrics = state.get("metrics", {}) if isinstance(state.get("metrics", {}), dict) else {}
        self.assertTrue(bool(state.get("active", False)))
        self.assertTrue(bool(checks.get("switch_rate_ok", False)))
        self.assertEqual(int(metrics.get("switch_count_raw", 0)), 4)
        self.assertEqual(int(metrics.get("switch_count_effective", 0)), 0)
        self.assertGreaterEqual(int(metrics.get("switch_transient_ignored", 0)), 2)

    def test_gate_report_state_degradation_hysteresis_allows_transient_soft_breach(self) -> None:
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

        modes = ["swing", "swing", "swing", "swing", "swing"]
        risks = [0.42, 0.41, 0.34, 0.43, 0.44]  # one soft floor breach only
        sources = [0.80, 0.81, 0.76, 0.79, 0.82]
        health = [True, True, True, True, True]
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
                "ops_risk_multiplier_drift_max": 0.30,
                "ops_source_confidence_floor": 0.75,
                "ops_mode_health_fail_days_max": 1,
                "ops_state_degradation_enabled": True,
                "ops_state_hysteresis_enabled": True,
                "ops_state_degradation_soft_multiplier": 1.10,
                "ops_state_degradation_hard_multiplier": 1.35,
                "ops_state_degradation_floor_soft_ratio": 0.96,
                "ops_state_degradation_floor_hard_ratio": 0.90,
                "ops_state_hysteresis_soft_streak_days": 2,
                "ops_state_hysteresis_hard_streak_days": 3,
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
        self.assertTrue(bool(out["checks"]["state_stability_ok"]))
        state = out.get("state_stability", {})
        self.assertEqual(str((state.get("degradation", {}) or {}).get("overall_tier", "")), "yellow")
        self.assertTrue(bool((state.get("checks", {}) or {}).get("risk_multiplier_floor_ok", False)))

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

    def test_gate_report_slot_degradation_hysteresis_allows_transient_soft_breach(self) -> None:
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
            pre_flags = ["timestamp_drift"] if i == 0 else []
            pre = {
                "quality": {"passed": True, "flags": pre_flags, "source_confidence_score": 0.85},
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
                "ops_slot_premarket_anomaly_ratio_max": 0.30,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_slot_source_confidence_floor": 0.75,
                "ops_slot_risk_multiplier_floor": 0.20,
                "ops_slot_degradation_enabled": True,
                "ops_slot_hysteresis_enabled": True,
                "ops_slot_degradation_soft_multiplier": 1.20,
                "ops_slot_degradation_hard_multiplier": 1.50,
                "ops_slot_hysteresis_soft_streak_days": 2,
                "ops_slot_hysteresis_hard_streak_days": 3,
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
        self.assertTrue(bool(out["checks"]["slot_anomaly_ok"]))
        slot = out.get("slot_anomaly", {})
        self.assertEqual(str((slot.get("degradation", {}) or {}).get("overall_tier", "")), "yellow")
        self.assertTrue(bool((slot.get("checks", {}) or {}).get("premarket_anomaly_ok", False)))

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

    def test_gate_report_fails_on_stress_matrix_execution_friction(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(
                {
                    "best_mode": "swing",
                    "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                    "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                    "execution_friction": {
                        "scorecard": {
                            "active": True,
                            "status": "critical",
                            "gate_ok": False,
                            "checks": {
                                "annual_drop_ok": False,
                                "drawdown_rise_ok": True,
                                "profit_factor_ratio_ok": True,
                                "fail_ratio_ok": True,
                                "positive_window_ratio_drop_ok": True,
                            },
                            "metrics": {
                                "reference_mode": "swing",
                                "annual_drop": 0.22,
                                "drawdown_rise": 0.03,
                                "min_profit_factor_ratio": 0.81,
                                "max_fail_ratio": 0.10,
                            },
                            "alerts": ["stress_matrix_execution_friction_annual_drop"],
                        }
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
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
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
        self.assertFalse(bool(out["checks"]["stress_matrix_execution_friction_ok"]))
        self.assertFalse(out["passed"])
        stress_exec = out.get("stress_matrix_execution_friction", {})
        self.assertTrue(bool(stress_exec.get("active", False)))
        self.assertFalse(bool(stress_exec.get("gate_ok", True)))

    def test_gate_report_fails_on_stress_matrix_execution_friction_trendline(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload(annual_drop: float, drawdown_rise: float, pf_ratio: float) -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": annual_drop,
                            "drawdown_rise": drawdown_rise,
                            "min_profit_factor_ratio": pf_ratio,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        (review_dir / "2026-02-10_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(0.02, 0.03, 0.96), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-11_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(0.03, 0.03, 0.95), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(0.12, 0.11, 0.73), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(0.14, 0.12, 0.70), ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 4,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 0.04,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 0.04,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 0.10,
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
        self.assertFalse(bool(out["checks"]["stress_matrix_execution_friction_ok"]))
        self.assertFalse(out["passed"])
        stress_exec = out.get("stress_matrix_execution_friction", {})
        trendline = stress_exec.get("trendline", {}) if isinstance(stress_exec.get("trendline", {}), dict) else {}
        self.assertTrue(bool(stress_exec.get("active", False)))
        self.assertTrue(bool(trendline.get("active", False)))
        trendline_checks = trendline.get("checks", {}) if isinstance(trendline.get("checks", {}), dict) else {}
        self.assertFalse(bool(trendline_checks.get("trendline_ok", True)))
        self.assertIn(
            "stress_matrix_execution_friction_trendline_annual_drop_rise",
            set(stress_exec.get("alerts", [])),
        )

    def test_gate_report_fails_on_stress_matrix_execution_friction_trendline_staleness(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        payload = {
            "best_mode": "swing",
            "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
            "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
            "execution_friction": {
                "scorecard": {
                    "active": True,
                    "status": "ok",
                    "gate_ok": True,
                    "checks": {
                        "annual_drop_ok": True,
                        "drawdown_rise_ok": True,
                        "profit_factor_ratio_ok": True,
                        "fail_ratio_ok": True,
                        "positive_window_ratio_drop_ok": True,
                    },
                    "metrics": {
                        "reference_mode": "swing",
                        "annual_drop": 0.05,
                        "drawdown_rise": 0.04,
                        "min_profit_factor_ratio": 0.90,
                        "max_fail_ratio": 0.10,
                    },
                    "alerts": [],
                }
            },
        }
        (review_dir / "2026-02-01_mode_stress_matrix.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": False,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_max_staleness_days": 3,
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
        self.assertFalse(bool(out["checks"]["stress_matrix_execution_friction_ok"]))
        stress_exec = out.get("stress_matrix_execution_friction", {})
        trendline = stress_exec.get("trendline", {}) if isinstance(stress_exec.get("trendline", {}), dict) else {}
        self.assertEqual(str(trendline.get("status", "")), "stale")
        trendline_checks = trendline.get("checks", {}) if isinstance(trendline.get("checks", {}), dict) else {}
        self.assertFalse(bool(trendline_checks.get("source_staleness_ok", True)))
        self.assertIn("stress_matrix_execution_friction_trendline_stale", set(stress_exec.get("alerts", [])))

    def test_gate_report_stress_matrix_execution_friction_trendline_autotune_applied(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload(annual_drop: float, drawdown_rise: float, pf_ratio: float) -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": annual_drop,
                            "drawdown_rise": drawdown_rise,
                            "min_profit_factor_ratio": pf_ratio,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        history = [
            ("2026-02-08", 0.04, 0.04, 0.98),
            ("2026-02-09", 0.07, 0.07, 0.95),
            ("2026-02-10", 0.10, 0.10, 0.92),
            ("2026-02-11", 0.13, 0.13, 0.89),
            ("2026-02-12", 0.16, 0.16, 0.86),
            ("2026-02-13", 0.19, 0.19, 0.83),
        ]
        for dtag, annual_drop, drawdown_rise, pf_ratio in history:
            (review_dir / f"{dtag}_mode_stress_matrix.json").write_text(
                json.dumps(_stress_exec_payload(annual_drop, drawdown_rise, pf_ratio), ensure_ascii=False),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 0.01,
                "ops_stress_matrix_execution_friction_trendline_autotune_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs": 6,
                "ops_stress_matrix_execution_friction_trendline_autotune_min_transitions": 3,
                "ops_stress_matrix_execution_friction_trendline_autotune_quantile": 0.80,
                "ops_stress_matrix_execution_friction_trendline_autotune_step_max": 0.10,
                "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_max_staleness_days": 10,
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
        self.assertTrue(bool(out["checks"]["stress_matrix_execution_friction_ok"]))
        stress_exec = out.get("stress_matrix_execution_friction", {})
        trendline = stress_exec.get("trendline", {}) if isinstance(stress_exec.get("trendline", {}), dict) else {}
        trendline_checks = trendline.get("checks", {}) if isinstance(trendline.get("checks", {}), dict) else {}
        self.assertTrue(bool(trendline_checks.get("trendline_ok", False)))
        auto_tune = trendline.get("auto_tune", {}) if isinstance(trendline.get("auto_tune", {}), dict) else {}
        self.assertTrue(bool(auto_tune.get("applied", False)))
        effective = auto_tune.get("effective_thresholds", {}) if isinstance(auto_tune.get("effective_thresholds", {}), dict) else {}
        self.assertGreater(float(effective.get("annual_drop_rise", 0.0)), 0.01)
        self.assertGreater(float(effective.get("drawdown_rise_delta", 0.0)), 0.01)
        self.assertGreater(float(effective.get("profit_factor_ratio_drop", 0.0)), 0.01)
        proposal = auto_tune.get("proposal", {}) if isinstance(auto_tune.get("proposal", {}), dict) else {}
        self.assertTrue(bool(proposal.get("generated", False)))
        self.assertEqual(str(proposal.get("mode", "")), "proposal_only")
        self.assertTrue(bool(str(proposal.get("proposal_id", "")).strip()))
        proposal_artifact = proposal.get("artifact", {}) if isinstance(proposal.get("artifact", {}), dict) else {}
        self.assertTrue(bool(proposal_artifact.get("written", False)))
        proposal_json = Path(str(proposal_artifact.get("json", "")).strip())
        proposal_md = Path(str(proposal_artifact.get("md", "")).strip())
        self.assertTrue(proposal_json.exists())
        self.assertTrue(proposal_md.exists())
        proposal_doc = json.loads(proposal_json.read_text(encoding="utf-8"))
        self.assertEqual(str(proposal_doc.get("mode", "")), "proposal_only")
        self.assertEqual(str(proposal_doc.get("proposal_id", "")), str(proposal.get("proposal_id", "")))
        scorecards = out.get("scorecards", {}) if isinstance(out.get("scorecards", {}), dict) else {}
        stress_card = (
            scorecards.get("stress_matrix", {})
            if isinstance(scorecards.get("stress_matrix", {}), dict)
            else {}
        )
        exec_card = (
            stress_card.get("execution_friction", {})
            if isinstance(stress_card.get("execution_friction", {}), dict)
            else {}
        )
        self.assertTrue(bool(exec_card.get("trendline_auto_tune_proposal_generated", False)))
        self.assertTrue(bool(exec_card.get("trendline_auto_tune_proposal_written", False)))

    def test_gate_report_stress_exec_trendline_controlled_apply_requires_approval(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload(annual_drop: float, drawdown_rise: float, pf_ratio: float) -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": annual_drop,
                            "drawdown_rise": drawdown_rise,
                            "min_profit_factor_ratio": pf_ratio,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        history = [
            ("2026-02-08", 0.04, 0.04, 0.98),
            ("2026-02-09", 0.07, 0.07, 0.95),
            ("2026-02-10", 0.10, 0.10, 0.92),
            ("2026-02-11", 0.13, 0.13, 0.89),
            ("2026-02-12", 0.16, 0.16, 0.86),
            ("2026-02-13", 0.19, 0.19, 0.83),
        ]
        for dtag, annual_drop, drawdown_rise, pf_ratio in history:
            (review_dir / f"{dtag}_mode_stress_matrix.json").write_text(
                json.dumps(_stress_exec_payload(annual_drop, drawdown_rise, pf_ratio), ensure_ascii=False),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 0.01,
                "ops_stress_matrix_execution_friction_trendline_autotune_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs": 6,
                "ops_stress_matrix_execution_friction_trendline_autotune_min_transitions": 3,
                "ops_stress_matrix_execution_friction_trendline_autotune_quantile": 0.80,
                "ops_stress_matrix_execution_friction_trendline_autotune_step_max": 0.10,
                "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_max_staleness_days": 10,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days": 7,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path": "artifacts/stress_matrix_execution_friction_trendline_autotune_approval.json",
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

        out_pending = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        stress_exec_pending = out_pending.get("stress_matrix_execution_friction", {})
        trendline_pending = (
            stress_exec_pending.get("trendline", {})
            if isinstance(stress_exec_pending.get("trendline", {}), dict)
            else {}
        )
        auto_tune_pending = (
            trendline_pending.get("auto_tune", {})
            if isinstance(trendline_pending.get("auto_tune", {}), dict)
            else {}
        )
        controlled_pending = (
            auto_tune_pending.get("controlled_apply", {})
            if isinstance(auto_tune_pending.get("controlled_apply", {}), dict)
            else {}
        )
        self.assertFalse(bool(auto_tune_pending.get("applied", True)))
        self.assertEqual(str(controlled_pending.get("reason", "")), "manual_approval_missing")
        proposal_pending = (
            auto_tune_pending.get("proposal", {})
            if isinstance(auto_tune_pending.get("proposal", {}), dict)
            else {}
        )
        proposal_id = str(proposal_pending.get("proposal_id", "")).strip()
        self.assertTrue(bool(proposal_id))

        approval_path = td / "artifacts" / "stress_matrix_execution_friction_trendline_autotune_approval.json"
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(
            json.dumps(
                {
                    "approved": True,
                    "proposal_id": proposal_id,
                    "approved_at": "2026-02-13T21:00:00+08:00",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out_applied = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        stress_exec_applied = out_applied.get("stress_matrix_execution_friction", {})
        trendline_applied = (
            stress_exec_applied.get("trendline", {})
            if isinstance(stress_exec_applied.get("trendline", {}), dict)
            else {}
        )
        auto_tune_applied = (
            trendline_applied.get("auto_tune", {})
            if isinstance(trendline_applied.get("auto_tune", {}), dict)
            else {}
        )
        controlled_applied = (
            auto_tune_applied.get("controlled_apply", {})
            if isinstance(auto_tune_applied.get("controlled_apply", {}), dict)
            else {}
        )
        self.assertTrue(bool(auto_tune_applied.get("applied", False)))
        self.assertTrue(bool(controlled_applied.get("applied", False)))
        self.assertEqual(str(controlled_applied.get("reason", "")), "apply_gate_open")
        thresholds = (
            trendline_applied.get("thresholds", {})
            if isinstance(trendline_applied.get("thresholds", {}), dict)
            else {}
        )
        self.assertGreater(
            float(thresholds.get("ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise", 0.0)),
            0.01,
        )

    def test_gate_report_stress_exec_trendline_controlled_apply_blocks_stale(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload(annual_drop: float, drawdown_rise: float, pf_ratio: float) -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": annual_drop,
                            "drawdown_rise": drawdown_rise,
                            "min_profit_factor_ratio": pf_ratio,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        history = [
            ("2026-02-05", 0.04, 0.04, 0.98),
            ("2026-02-06", 0.07, 0.07, 0.95),
            ("2026-02-07", 0.10, 0.10, 0.92),
            ("2026-02-08", 0.13, 0.13, 0.89),
            ("2026-02-09", 0.16, 0.16, 0.86),
            ("2026-02-10", 0.19, 0.19, 0.83),
        ]
        for dtag, annual_drop, drawdown_rise, pf_ratio in history:
            (review_dir / f"{dtag}_mode_stress_matrix.json").write_text(
                json.dumps(_stress_exec_payload(annual_drop, drawdown_rise, pf_ratio), ensure_ascii=False),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 0.01,
                "ops_stress_matrix_execution_friction_trendline_autotune_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs": 6,
                "ops_stress_matrix_execution_friction_trendline_autotune_min_transitions": 3,
                "ops_stress_matrix_execution_friction_trendline_autotune_quantile": 0.80,
                "ops_stress_matrix_execution_friction_trendline_autotune_step_max": 0.10,
                "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_max_staleness_days": 10,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required": False,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days": 1,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path": "artifacts/stress_matrix_execution_friction_trendline_autotune_approval.json",
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
        stress_exec = out.get("stress_matrix_execution_friction", {})
        trendline = (
            stress_exec.get("trendline", {})
            if isinstance(stress_exec.get("trendline", {}), dict)
            else {}
        )
        auto_tune = trendline.get("auto_tune", {}) if isinstance(trendline.get("auto_tune", {}), dict) else {}
        controlled = (
            auto_tune.get("controlled_apply", {})
            if isinstance(auto_tune.get("controlled_apply", {}), dict)
            else {}
        )
        self.assertFalse(bool(auto_tune.get("applied", True)))
        self.assertEqual(str(controlled.get("reason", "")), "proposal_outside_apply_window")
        self.assertIn(
            "stress_matrix_execution_friction_trendline_controlled_apply_stale",
            set(stress_exec.get("alerts", [])),
        )
        thresholds = trendline.get("thresholds", {}) if isinstance(trendline.get("thresholds", {}), dict) else {}
        self.assertAlmostEqual(
            float(thresholds.get("ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise", 0.0)),
            0.01,
            places=8,
        )

    def test_gate_report_stress_exec_trendline_controlled_apply_blocks_duplicate_proposal(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload(annual_drop: float, drawdown_rise: float, pf_ratio: float) -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": annual_drop,
                            "drawdown_rise": drawdown_rise,
                            "min_profit_factor_ratio": pf_ratio,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        history = [
            ("2026-02-08", 0.04, 0.04, 0.98),
            ("2026-02-09", 0.07, 0.07, 0.95),
            ("2026-02-10", 0.10, 0.10, 0.92),
            ("2026-02-11", 0.13, 0.13, 0.89),
            ("2026-02-12", 0.16, 0.16, 0.86),
            ("2026-02-13", 0.19, 0.19, 0.83),
        ]
        for dtag, annual_drop, drawdown_rise, pf_ratio in history:
            (review_dir / f"{dtag}_mode_stress_matrix.json").write_text(
                json.dumps(_stress_exec_payload(annual_drop, drawdown_rise, pf_ratio), ensure_ascii=False),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 0.01,
                "ops_stress_matrix_execution_friction_trendline_autotune_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs": 6,
                "ops_stress_matrix_execution_friction_trendline_autotune_min_transitions": 3,
                "ops_stress_matrix_execution_friction_trendline_autotune_quantile": 0.80,
                "ops_stress_matrix_execution_friction_trendline_autotune_step_max": 0.10,
                "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_max_staleness_days": 10,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days": 7,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path": "artifacts/stress_matrix_execution_friction_trendline_autotune_approval.json",
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path": "artifacts/stress_matrix_execution_friction_trendline_controlled_apply_ledger.json",
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

        out_pending = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        trendline_pending = (
            (out_pending.get("stress_matrix_execution_friction", {}) or {}).get("trendline", {})
            if isinstance((out_pending.get("stress_matrix_execution_friction", {}) or {}).get("trendline", {}), dict)
            else {}
        )
        auto_tune_pending = (
            trendline_pending.get("auto_tune", {})
            if isinstance(trendline_pending.get("auto_tune", {}), dict)
            else {}
        )
        proposal_pending = (
            auto_tune_pending.get("proposal", {})
            if isinstance(auto_tune_pending.get("proposal", {}), dict)
            else {}
        )
        proposal_id = str(proposal_pending.get("proposal_id", "")).strip()
        self.assertTrue(bool(proposal_id))

        approval_path = td / "artifacts" / "stress_matrix_execution_friction_trendline_autotune_approval.json"
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(
            json.dumps(
                {
                    "approved": True,
                    "proposal_id": proposal_id,
                    "approved_at": "2026-02-13T21:00:00+08:00",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        out_apply_1 = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        trendline_apply_1 = (
            (out_apply_1.get("stress_matrix_execution_friction", {}) or {}).get("trendline", {})
            if isinstance((out_apply_1.get("stress_matrix_execution_friction", {}) or {}).get("trendline", {}), dict)
            else {}
        )
        auto_tune_apply_1 = (
            trendline_apply_1.get("auto_tune", {})
            if isinstance(trendline_apply_1.get("auto_tune", {}), dict)
            else {}
        )
        controlled_apply_1 = (
            auto_tune_apply_1.get("controlled_apply", {})
            if isinstance(auto_tune_apply_1.get("controlled_apply", {}), dict)
            else {}
        )
        self.assertTrue(bool(controlled_apply_1.get("applied", False)))
        self.assertEqual(str(controlled_apply_1.get("reason", "")), "apply_gate_open")
        ledger_apply_1 = (
            controlled_apply_1.get("ledger", {})
            if isinstance(controlled_apply_1.get("ledger", {}), dict)
            else {}
        )
        self.assertEqual(int(ledger_apply_1.get("entries", 0)), 1)

        out_apply_2 = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        stress_exec_apply_2 = out_apply_2.get("stress_matrix_execution_friction", {})
        trendline_apply_2 = (
            stress_exec_apply_2.get("trendline", {})
            if isinstance(stress_exec_apply_2.get("trendline", {}), dict)
            else {}
        )
        checks_apply_2 = (
            trendline_apply_2.get("checks", {})
            if isinstance(trendline_apply_2.get("checks", {}), dict)
            else {}
        )
        auto_tune_apply_2 = (
            trendline_apply_2.get("auto_tune", {})
            if isinstance(trendline_apply_2.get("auto_tune", {}), dict)
            else {}
        )
        controlled_apply_2 = (
            auto_tune_apply_2.get("controlled_apply", {})
            if isinstance(auto_tune_apply_2.get("controlled_apply", {}), dict)
            else {}
        )
        ledger_apply_2 = (
            controlled_apply_2.get("ledger", {})
            if isinstance(controlled_apply_2.get("ledger", {}), dict)
            else {}
        )
        self.assertFalse(bool(controlled_apply_2.get("applied", True)))
        self.assertEqual(str(controlled_apply_2.get("reason", "")), "proposal_already_applied")
        self.assertTrue(bool(controlled_apply_2.get("duplicate_blocked", False)))
        self.assertFalse(bool(checks_apply_2.get("controlled_apply_duplicate_ok", True)))
        self.assertIn(
            "stress_matrix_execution_friction_trendline_controlled_apply_duplicate",
            set(stress_exec_apply_2.get("alerts", [])),
        )
        self.assertEqual(int(ledger_apply_2.get("entries", 0)), 1)
        self.assertTrue(bool(ledger_apply_2.get("duplicate_blocked", False)))

    def test_gate_report_stress_exec_trendline_controlled_apply_blocks_on_ledger_write_failure(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload(annual_drop: float, drawdown_rise: float, pf_ratio: float) -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": annual_drop,
                            "drawdown_rise": drawdown_rise,
                            "min_profit_factor_ratio": pf_ratio,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        history = [
            ("2026-02-08", 0.04, 0.04, 0.98),
            ("2026-02-09", 0.07, 0.07, 0.95),
            ("2026-02-10", 0.10, 0.10, 0.92),
            ("2026-02-11", 0.13, 0.13, 0.89),
            ("2026-02-12", 0.16, 0.16, 0.86),
            ("2026-02-13", 0.19, 0.19, 0.83),
        ]
        for dtag, annual_drop, drawdown_rise, pf_ratio in history:
            (review_dir / f"{dtag}_mode_stress_matrix.json").write_text(
                json.dumps(_stress_exec_payload(annual_drop, drawdown_rise, pf_ratio), ensure_ascii=False),
                encoding="utf-8",
            )

        ledger_path = td / "artifacts" / "stress_matrix_execution_friction_trendline_controlled_apply_ledger.json"
        ledger_path.mkdir(parents=True, exist_ok=True)

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 0.01,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 0.01,
                "ops_stress_matrix_execution_friction_trendline_autotune_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs": 6,
                "ops_stress_matrix_execution_friction_trendline_autotune_min_transitions": 3,
                "ops_stress_matrix_execution_friction_trendline_autotune_quantile": 0.80,
                "ops_stress_matrix_execution_friction_trendline_autotune_step_max": 0.10,
                "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_max_staleness_days": 10,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required": False,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days": 7,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path": "artifacts/stress_matrix_execution_friction_trendline_controlled_apply_ledger.json",
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
        stress_exec = out.get("stress_matrix_execution_friction", {})
        trendline = (
            stress_exec.get("trendline", {})
            if isinstance(stress_exec.get("trendline", {}), dict)
            else {}
        )
        checks = trendline.get("checks", {}) if isinstance(trendline.get("checks", {}), dict) else {}
        auto_tune = trendline.get("auto_tune", {}) if isinstance(trendline.get("auto_tune", {}), dict) else {}
        controlled = (
            auto_tune.get("controlled_apply", {})
            if isinstance(auto_tune.get("controlled_apply", {}), dict)
            else {}
        )
        ledger = (
            controlled.get("ledger", {})
            if isinstance(controlled.get("ledger", {}), dict)
            else {}
        )
        self.assertFalse(bool(controlled.get("applied", True)))
        self.assertEqual(str(controlled.get("reason", "")), "ledger_write_failed")
        self.assertFalse(bool(checks.get("controlled_apply_ledger_write_ok", True)))
        self.assertIn(
            "stress_matrix_execution_friction_trendline_controlled_apply_ledger_write_failed",
            set(stress_exec.get("alerts", [])),
        )
        self.assertIn("write_failed", str(ledger.get("write_error", "")))

    def test_gate_report_stress_exec_trendline_controlled_apply_ledger_drift_monitor_only(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload(annual_drop: float, drawdown_rise: float, pf_ratio: float) -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [
                    {
                        "mode": "swing",
                        "robustness_score": 0.6,
                        "avg_annual_return": 0.12,
                        "worst_drawdown": 0.10,
                    }
                ],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": annual_drop,
                            "drawdown_rise": drawdown_rise,
                            "min_profit_factor_ratio": pf_ratio,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        history = [
            ("2026-02-12", 0.10, 0.10, 0.95),
            ("2026-02-13", 0.10, 0.10, 0.95),
        ]
        for dtag, annual_drop, drawdown_rise, pf_ratio in history:
            (review_dir / f"{dtag}_mode_stress_matrix.json").write_text(
                json.dumps(_stress_exec_payload(annual_drop, drawdown_rise, pf_ratio), ensure_ascii=False),
                encoding="utf-8",
            )

        (review_dir / "2026-02-12_gate_report.json").write_text(
            json.dumps(
                {
                    "stress_matrix_execution_friction": {
                        "trendline": {
                            "auto_tune": {
                                "controlled_apply": {
                                    "enabled": True,
                                    "duplicate_blocked": True,
                                }
                            }
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        ledger_path = td / "artifacts" / "stress_matrix_execution_friction_trendline_controlled_apply_ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "date": "2026-02-13",
                            "proposal_id": "old_proposal",
                            "applied": True,
                            "proposal_age_days": 3,
                            "applied_at": "2026-02-13T12:00:00+08:00",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 1.0,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 1.0,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 1.0,
                "ops_stress_matrix_execution_friction_trendline_autotune_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs": 6,
                "ops_stress_matrix_execution_friction_trendline_autotune_min_transitions": 1,
                "ops_stress_matrix_execution_friction_trendline_autotune_quantile": 0.80,
                "ops_stress_matrix_execution_friction_trendline_autotune_step_max": 0.10,
                "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_max_staleness_days": 10,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required": False,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days": 1,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path": "artifacts/stress_matrix_execution_friction_trendline_controlled_apply_ledger.json",
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_staleness_window_days": 7,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_gate_hard_fail": False,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_window_stale_ratio_max": 0.0,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_max": 0.0,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        stress_exec = out.get("stress_matrix_execution_friction", {})
        trendline = (
            stress_exec.get("trendline", {})
            if isinstance(stress_exec.get("trendline", {}), dict)
            else {}
        )
        checks = trendline.get("checks", {}) if isinstance(trendline.get("checks", {}), dict) else {}
        auto_tune = trendline.get("auto_tune", {}) if isinstance(trendline.get("auto_tune", {}), dict) else {}
        controlled = (
            auto_tune.get("controlled_apply", {})
            if isinstance(auto_tune.get("controlled_apply", {}), dict)
            else {}
        )
        ledger = (
            ((trendline.get("auto_tune", {}) or {}).get("controlled_apply", {}) or {}).get("ledger", {})
            if isinstance(((trendline.get("auto_tune", {}) or {}).get("controlled_apply", {}) or {}).get("ledger", {}), dict)
            else {}
        )
        proposal = auto_tune.get("proposal", {}) if isinstance(auto_tune.get("proposal", {}), dict) else {}
        apply_gate = proposal.get("apply_gate", {}) if isinstance(proposal.get("apply_gate", {}), dict) else {}
        drift = ledger.get("drift", {}) if isinstance(ledger.get("drift", {}), dict) else {}
        drillbook = ledger.get("drillbook", {}) if isinstance(ledger.get("drillbook", {}), dict) else {}
        workflows = drillbook.get("workflows", {}) if isinstance(drillbook.get("workflows", {}), dict) else {}
        replay = workflows.get("replay", {}) if isinstance(workflows.get("replay", {}), dict) else {}
        rollback = workflows.get("rollback", {}) if isinstance(workflows.get("rollback", {}), dict) else {}
        self.assertTrue(bool(checks.get("controlled_apply_ledger_drift_ok", False)))
        self.assertTrue(bool(checks.get("controlled_apply_ledger_artifact_ok", False)))
        self.assertTrue(bool(stress_exec.get("gate_ok", False)))
        self.assertIn(
            "stress_matrix_execution_friction_trendline_controlled_apply_ledger_stale_ratio_high",
            set(stress_exec.get("alerts", [])),
        )
        self.assertIn(
            "stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_high",
            set(stress_exec.get("alerts", [])),
        )
        self.assertTrue(bool(drift.get("breached", False)))
        self.assertEqual(str(drift.get("status", "")), "warn")
        self.assertTrue(bool(drillbook.get("written", False)))
        self.assertTrue(Path(str(drillbook.get("json", ""))).exists())
        self.assertTrue(Path(str(drillbook.get("md", ""))).exists())
        self.assertTrue(bool(replay.get("recommended", False)))
        self.assertFalse(bool(rollback.get("recommended", True)))
        self.assertEqual(str(controlled.get("reason", "")), "proposal_not_generated")
        self.assertTrue(bool(apply_gate.get("ledger_replay_recommended", False)))
        self.assertFalse(bool(apply_gate.get("ledger_rollback_recommended", True)))

    def test_gate_report_stress_exec_trendline_controlled_apply_ledger_drift_hard_fail(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload() -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": 0.10,
                            "drawdown_rise": 0.10,
                            "min_profit_factor_ratio": 0.95,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_gate_report.json").write_text(
            json.dumps(
                {
                    "stress_matrix_execution_friction": {
                        "trendline": {
                            "auto_tune": {
                                "controlled_apply": {
                                    "enabled": True,
                                    "duplicate_blocked": True,
                                }
                            }
                        }
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        ledger_path = td / "artifacts" / "stress_matrix_execution_friction_trendline_controlled_apply_ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "date": "2026-02-13",
                            "proposal_id": "old_proposal",
                            "applied": True,
                            "proposal_age_days": 3,
                            "applied_at": "2026-02-13T12:00:00+08:00",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 1,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 1.0,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 1.0,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 1.0,
                "ops_stress_matrix_execution_friction_trendline_autotune_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_autotune_lookback_runs": 6,
                "ops_stress_matrix_execution_friction_trendline_autotune_min_transitions": 1,
                "ops_stress_matrix_execution_friction_trendline_autotune_quantile": 0.80,
                "ops_stress_matrix_execution_friction_trendline_autotune_step_max": 0.10,
                "ops_stress_matrix_execution_friction_trendline_staleness_guard_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_max_staleness_days": 10,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required": False,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days": 1,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path": "artifacts/stress_matrix_execution_friction_trendline_controlled_apply_ledger.json",
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_staleness_window_days": 7,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_gate_hard_fail": True,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_window_stale_ratio_max": 0.0,
                "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_max": 0.0,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        stress_exec = out.get("stress_matrix_execution_friction", {})
        trendline = (
            stress_exec.get("trendline", {})
            if isinstance(stress_exec.get("trendline", {}), dict)
            else {}
        )
        checks = trendline.get("checks", {}) if isinstance(trendline.get("checks", {}), dict) else {}
        auto_tune = trendline.get("auto_tune", {}) if isinstance(trendline.get("auto_tune", {}), dict) else {}
        controlled = (
            auto_tune.get("controlled_apply", {})
            if isinstance(auto_tune.get("controlled_apply", {}), dict)
            else {}
        )
        ledger = (
            ((trendline.get("auto_tune", {}) or {}).get("controlled_apply", {}) or {}).get("ledger", {})
            if isinstance(((trendline.get("auto_tune", {}) or {}).get("controlled_apply", {}) or {}).get("ledger", {}), dict)
            else {}
        )
        proposal = auto_tune.get("proposal", {}) if isinstance(auto_tune.get("proposal", {}), dict) else {}
        apply_gate = proposal.get("apply_gate", {}) if isinstance(proposal.get("apply_gate", {}), dict) else {}
        drift = ledger.get("drift", {}) if isinstance(ledger.get("drift", {}), dict) else {}
        drillbook = ledger.get("drillbook", {}) if isinstance(ledger.get("drillbook", {}), dict) else {}
        workflows = drillbook.get("workflows", {}) if isinstance(drillbook.get("workflows", {}), dict) else {}
        replay = workflows.get("replay", {}) if isinstance(workflows.get("replay", {}), dict) else {}
        rollback = workflows.get("rollback", {}) if isinstance(workflows.get("rollback", {}), dict) else {}
        self.assertFalse(bool(checks.get("controlled_apply_ledger_drift_ok", True)))
        self.assertTrue(bool(checks.get("controlled_apply_ledger_artifact_ok", False)))
        self.assertFalse(bool(stress_exec.get("gate_ok", True)))
        self.assertEqual(str(trendline.get("status", "")), "warn")
        self.assertEqual(str(drift.get("status", "")), "critical")
        self.assertTrue(bool(drillbook.get("written", False)))
        self.assertTrue(Path(str(drillbook.get("json", ""))).exists())
        self.assertTrue(Path(str(drillbook.get("md", ""))).exists())
        self.assertTrue(bool(replay.get("recommended", False)))
        self.assertTrue(bool(rollback.get("recommended", False)))
        self.assertEqual(str(controlled.get("reason", "")), "proposal_not_generated")
        self.assertTrue(bool(apply_gate.get("ledger_replay_recommended", False)))
        self.assertTrue(bool(apply_gate.get("ledger_rollback_recommended", False)))
        self.assertFalse(bool(out.get("passed", True)))

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

    def test_review_until_pass_defect_plan_includes_execution_friction_trendline_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        def _stress_exec_payload(annual_drop: float, drawdown_rise: float, pf_ratio: float) -> dict[str, object]:
            return {
                "best_mode": "swing",
                "mode_summary": [{"mode": "swing", "robustness_score": 0.6, "avg_annual_return": 0.12, "worst_drawdown": 0.10}],
                "matrix": [{"mode": "swing", "window": "w0", "status": "ok"}],
                "execution_friction": {
                    "scorecard": {
                        "active": True,
                        "status": "ok",
                        "gate_ok": True,
                        "checks": {
                            "annual_drop_ok": True,
                            "drawdown_rise_ok": True,
                            "profit_factor_ratio_ok": True,
                            "fail_ratio_ok": True,
                            "positive_window_ratio_drop_ok": True,
                        },
                        "metrics": {
                            "reference_mode": "swing",
                            "annual_drop": annual_drop,
                            "drawdown_rise": drawdown_rise,
                            "min_profit_factor_ratio": pf_ratio,
                            "max_fail_ratio": 0.10,
                        },
                        "alerts": [],
                    }
                },
            }

        (review_dir / "2026-02-10_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(0.02, 0.03, 0.96), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-11_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(0.03, 0.03, 0.95), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-12_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(0.12, 0.11, 0.73), ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-13_mode_stress_matrix.json").write_text(
            json.dumps(_stress_exec_payload(0.14, 0.12, 0.70), ensure_ascii=False),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_matrix_trend_enabled": False,
                "ops_stress_matrix_execution_friction_enabled": True,
                "ops_stress_matrix_execution_friction_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_enabled": True,
                "ops_stress_matrix_execution_friction_trendline_require_active": True,
                "ops_stress_matrix_execution_friction_trendline_require_samples": True,
                "ops_stress_matrix_execution_friction_trendline_recent_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_prior_runs": 2,
                "ops_stress_matrix_execution_friction_trendline_min_runs": 4,
                "ops_stress_matrix_execution_friction_trendline_max_annual_drop_rise": 0.04,
                "ops_stress_matrix_execution_friction_trendline_max_drawdown_rise": 0.04,
                "ops_stress_matrix_execution_friction_trendline_max_profit_factor_ratio_drop": 0.10,
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
        self.assertIn("STRESS_MATRIX_EXECUTION_FRICTION", codes)
        self.assertIn("STRESS_MATRIX_EXECUTION_FRICTION_TRENDLINE", codes)
        self.assertIn("STRESS_MATRIX_EXECUTION_FRICTION_TRENDLINE_ANNUAL_DROP", codes)

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

    def test_ops_report_stress_autorun_history_no_trigger_alert_waits_for_payload_coverage(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
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
                "ops_stress_autorun_history_no_trigger_min_payload_days": 3,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.ops_report(as_of=d, window_days=3)
        hist = out.get("stress_autorun_history", {})
        alerts = set(hist.get("alerts", [])) if isinstance(hist, dict) else set()
        self.assertNotIn("stress_autorun_history_no_triggers", alerts)
        metrics = hist.get("metrics", {}) if isinstance(hist.get("metrics", {}), dict) else {}
        self.assertFalse(bool(metrics.get("no_trigger_signal_ready", True)))

    def test_gate_report_stress_autorun_history_rotation_and_checksum_index(self) -> None:
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

        stale_json = review_dir / "2026-02-10_stress_autorun_history.json"
        stale_md = review_dir / "2026-02-10_stress_autorun_history.md"
        stale_json.write_text("{}", encoding="utf-8")
        stale_md.write_text("# stale\n", encoding="utf-8")

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
                                "reason_codes": ["mode_drift"],
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

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_stress_autorun_history_enabled": True,
                "ops_stress_autorun_history_window_days": 3,
                "ops_stress_autorun_history_min_rounds": 1,
                "ops_stress_autorun_history_retention_days": 2,
                "ops_stress_autorun_history_checksum_index_enabled": True,
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
        hist = out.get("stress_autorun_history", {}) if isinstance(out.get("stress_autorun_history", {}), dict) else {}
        artifacts = hist.get("artifacts", {}) if isinstance(hist.get("artifacts", {}), dict) else {}
        history_artifact = artifacts.get("history", {}) if isinstance(artifacts.get("history", {}), dict) else {}

        self.assertTrue(bool(history_artifact.get("written", False)))
        self.assertEqual(int(history_artifact.get("retention_days", 0)), 2)
        self.assertGreaterEqual(int(history_artifact.get("rotated_out_count", 0)), 1)
        self.assertIn("2026-02-10", set(history_artifact.get("rotated_out_dates", [])))
        self.assertFalse(stale_json.exists())
        self.assertFalse(stale_md.exists())

        self.assertTrue(bool(history_artifact.get("checksum_index_enabled", False)))
        self.assertTrue(bool(history_artifact.get("checksum_index_written", False)))
        index_path = Path(str(history_artifact.get("checksum_index_path", "")))
        self.assertTrue(index_path.exists())
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(int(index_payload.get("retention_days", 0)), 2)
        self.assertGreaterEqual(int(index_payload.get("entry_count", 0)), 1)
        entries = index_payload.get("entries", [])
        self.assertTrue(entries)
        first = entries[0] if isinstance(entries[0], dict) else {}
        self.assertEqual(len(str(first.get("json_sha256", ""))), 64)
        self.assertEqual(len(str(first.get("md_sha256", ""))), 64)

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

    def test_gate_report_stress_autorun_reason_drift_rotation_and_checksum_index(self) -> None:
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

        stale_json = review_dir / "2026-02-10_stress_autorun_reason_drift.json"
        stale_md = review_dir / "2026-02-10_stress_autorun_reason_drift.md"
        stale_json.write_text("{}", encoding="utf-8")
        stale_md.write_text("# stale\n", encoding="utf-8")

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
                "ops_stress_autorun_reason_drift_retention_days": 2,
                "ops_stress_autorun_reason_drift_checksum_index_enabled": True,
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
        drift = out.get("stress_autorun_reason_drift", {}) if isinstance(out.get("stress_autorun_reason_drift", {}), dict) else {}
        artifacts = drift.get("artifacts", {}) if isinstance(drift.get("artifacts", {}), dict) else {}
        reason_artifact = artifacts.get("reason_drift", {}) if isinstance(artifacts.get("reason_drift", {}), dict) else {}

        self.assertTrue(bool(reason_artifact.get("written", False)))
        self.assertEqual(int(reason_artifact.get("retention_days", 0)), 2)
        self.assertGreaterEqual(int(reason_artifact.get("rotated_out_count", 0)), 1)
        self.assertIn("2026-02-10", set(reason_artifact.get("rotated_out_dates", [])))
        self.assertFalse(stale_json.exists())
        self.assertFalse(stale_md.exists())

        self.assertTrue(bool(reason_artifact.get("checksum_index_enabled", False)))
        self.assertTrue(bool(reason_artifact.get("checksum_index_written", False)))
        index_path = Path(str(reason_artifact.get("checksum_index_path", "")))
        self.assertTrue(index_path.exists())
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(int(index_payload.get("retention_days", 0)), 2)
        self.assertGreaterEqual(int(index_payload.get("entry_count", 0)), 1)
        entries = index_payload.get("entries", [])
        self.assertTrue(entries)
        first = entries[0] if isinstance(entries[0], dict) else {}
        self.assertEqual(len(str(first.get("json_sha256", ""))), 64)
        self.assertEqual(len(str(first.get("md_sha256", ""))), 64)

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

    def test_gate_report_reconcile_uses_csv_active_rows_for_plan_gap(self) -> None:
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
                (
                    "symbol,side,size_pct,status\n"
                    "A,LONG,10,ACTIVE\n"
                    "B,LONG,12,ACTIVE\n"
                    "B,LONG,0,CLOSED\n"
                ),
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
                    "INSERT INTO latest_positions (date, symbol, size_pct, status) VALUES (?, ?, ?, ?)",
                    (day, "B", 12.0, "ACTIVE"),
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
        self.assertTrue(bool(out["checks"]["reconcile_drift_ok"]))
        recon = out.get("reconcile_drift", {})
        checks = recon.get("checks", {}) if isinstance(recon.get("checks", {}), dict) else {}
        self.assertTrue(bool(checks.get("plan_count_gap_ok", False)))
        self.assertNotIn("reconcile_plan_count_gap_high", set(recon.get("alerts", [])))

    def test_gate_report_reconcile_ignores_future_open_date_rows(self) -> None:
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
            (daily_dir / f"{dstr}_positions.csv").write_text(
                "symbol,side,size_pct,status\n",
                encoding="utf-8",
            )

        (td / "artifacts" / "paper_positions_open.json").write_text(
            json.dumps({"as_of": d.isoformat(), "positions": []}, ensure_ascii=False),
            encoding="utf-8",
        )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute(
                "CREATE TABLE executed_plans (date TEXT, open_date TEXT, symbol TEXT, pnl REAL, status TEXT)"
            )
            conn.execute(
                "INSERT INTO executed_plans (date, open_date, symbol, pnl, status) VALUES (?, ?, ?, ?, ?)",
                (d.isoformat(), (d + timedelta(days=2)).isoformat(), "A", 0.12, "CLOSED"),
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
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 0.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 0.0,
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
        recon = out.get("reconcile_drift", {})
        checks = recon.get("checks", {}) if isinstance(recon.get("checks", {}), dict) else {}
        alerts = set(recon.get("alerts", []))
        self.assertTrue(bool(checks.get("closed_count_gap_ok", False)))
        self.assertTrue(bool(checks.get("closed_pnl_gap_ok", False)))
        self.assertNotIn("reconcile_closed_count_gap_high", alerts)
        self.assertNotIn("reconcile_closed_pnl_gap_high", alerts)

    def test_gate_report_reconcile_dedupes_duplicate_closed_rows(self) -> None:
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
                            "plans": 0,
                            "closed_trades": 1,
                            "closed_pnl": 0.05,
                            "open_positions": 0,
                            "risk_multiplier": 1.0,
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_positions.csv").write_text(
                "symbol,side,size_pct,status\n",
                encoding="utf-8",
            )

        (td / "artifacts" / "paper_positions_open.json").write_text(
            json.dumps({"as_of": d.isoformat(), "positions": []}, ensure_ascii=False),
            encoding="utf-8",
        )

        db_path = td / "artifacts" / "lie_engine.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute("CREATE TABLE latest_positions (date TEXT, symbol TEXT, size_pct REAL, status TEXT)")
            conn.execute(
                "CREATE TABLE executed_plans (date TEXT, open_date TEXT, symbol TEXT, side TEXT, pnl REAL, status TEXT)"
            )
            for i in range(3):
                day = date.fromordinal(d.toordinal() - i).isoformat()
                row = (day, day, "A", "LONG", 0.05, "CLOSED")
                conn.execute(
                    "INSERT INTO executed_plans (date, open_date, symbol, side, pnl, status) VALUES (?, ?, ?, ?, ?, ?)",
                    row,
                )
                conn.execute(
                    "INSERT INTO executed_plans (date, open_date, symbol, side, pnl, status) VALUES (?, ?, ?, ?, ?, ?)",
                    row,
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
                "ops_reconcile_plan_gap_ratio_max": 1.0,
                "ops_reconcile_closed_count_gap_ratio_max": 0.0,
                "ops_reconcile_closed_pnl_gap_abs_max": 0.0,
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
        recon = out.get("reconcile_drift", {})
        checks = recon.get("checks", {}) if isinstance(recon.get("checks", {}), dict) else {}
        metrics = recon.get("metrics", {}) if isinstance(recon.get("metrics", {}), dict) else {}
        alerts = set(recon.get("alerts", []))
        self.assertTrue(bool(checks.get("closed_count_gap_ok", False)))
        self.assertTrue(bool(checks.get("closed_pnl_gap_ok", False)))
        self.assertNotIn("reconcile_closed_count_gap_high", alerts)
        self.assertNotIn("reconcile_closed_pnl_gap_high", alerts)
        self.assertEqual(int(metrics.get("executed_closed_dedup_pruned_total", 0)), 3)
        self.assertEqual(int(metrics.get("executed_closed_dedup_days", 0)), 3)

    def test_gate_report_missing_backtest_snapshot_does_not_emit_hard_rollback(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {},
            run_review=lambda as_of: ReviewDelta(as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        checks = out.get("checks", {})
        self.assertFalse(bool(checks.get("backtest_snapshot_ok", True)))
        rollback = out.get("rollback_recommendation", {})
        self.assertNotEqual(str(rollback.get("level", "")), "hard")
        reasons = set(rollback.get("reason_codes", []))
        self.assertIn("backtest_snapshot_missing", reasons)
        self.assertNotIn("risk_violations", reasons)
        self.assertNotIn("max_drawdown", reasons)

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

    def test_review_until_pass_defect_plan_includes_release_decision_freshness(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_delta = review_dir / f"{d.isoformat()}_param_delta.yaml"
        review_delta.write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        old_ts = (datetime.now() - timedelta(hours=8)).timestamp()
        os.utime(review_delta, (old_ts, old_ts))

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "release_decision_freshness_enabled": True,
                "release_decision_freshness_hard_fail": True,
                "release_decision_review_max_staleness_hours": 1,
            }
        )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(bool(out.get("passed", True)))
        round0 = out["rounds"][0]
        plan = json.loads(Path(str(round0["defect_plan"]["json"])).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("RELEASE_DECISION_FRESHNESS", codes)

    def test_review_until_pass_defect_plan_includes_degradation_guardrail_breaches(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        for i in range(2):
            day = d - timedelta(days=i)
            (review_dir / f"{day.isoformat()}_degradation_calibration_rollback.json").write_text(
                json.dumps(
                    {
                        "triggered_raw": True,
                        "triggered": False,
                        "applied": False,
                        "reason": "rollback_cooldown_active",
                        "cooldown": {"rollback_active": True},
                        "snapshot_promotion": {"eligible": False, "promoted": False},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_dashboard_enabled": True,
                "ops_degradation_guardrail_dashboard_hard_fail": True,
                "ops_degradation_guardrail_dashboard_window_days": 7,
                "ops_degradation_guardrail_dashboard_min_samples": 2,
                "ops_degradation_guardrail_cooldown_hit_rate_max": 0.4,
                "ops_degradation_guardrail_suppressed_trigger_density_max": 0.4,
                "ops_degradation_guardrail_promotion_latency_days_max": 21,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(bool(out.get("passed", True)))
        round0 = out["rounds"][0]
        plan = json.loads(Path(str(round0["defect_plan"]["json"])).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertTrue(any(code.startswith("DEGRADATION_GUARDRAIL_") for code in codes))

    def test_gate_report_degradation_guardrail_threshold_drift_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (review_dir / f"{d.isoformat()}_degradation_guardrail_threshold_drift.json").write_text(
            json.dumps(
                {
                    "date": d.isoformat(),
                    "status": "critical",
                    "summary": {"warn_count": 1, "critical_count": 1, "burnin_samples": 3},
                    "alerts": [
                        "degradation_guardrail_threshold_drift_critical:ops_degradation_guardrail_cooldown_hit_rate_max"
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
                "ops_degradation_guardrail_threshold_drift_enabled": True,
                "ops_degradation_guardrail_threshold_drift_gate_hard_fail": True,
                "ops_degradation_guardrail_threshold_drift_require_active": True,
                "ops_degradation_guardrail_threshold_drift_max_staleness_days": 14,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out.get("passed", True)))
        self.assertFalse(bool(out.get("checks", {}).get("degradation_guardrail_threshold_drift_ok", True)))
        payload = out.get("degradation_guardrail_threshold_drift", {})
        self.assertTrue(bool(payload.get("active", False)))
        self.assertEqual(str(payload.get("status", "")), "critical")

    def test_gate_report_threshold_drift_insufficient_samples_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (review_dir / f"{d.isoformat()}_degradation_guardrail_threshold_drift.json").write_text(
            json.dumps(
                {
                    "date": d.isoformat(),
                    "status": "insufficient_samples",
                    "summary": {
                        "warn_count": 0,
                        "critical_count": 0,
                        "burnin_samples": 1,
                        "burnin_min_samples": 3,
                        "burnin_samples_ok": False,
                    },
                    "alerts": ["degradation_guardrail_threshold_drift_insufficient_burnin_samples"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_degradation_guardrail_threshold_drift_enabled": True,
                "ops_degradation_guardrail_threshold_drift_gate_hard_fail": True,
                "ops_degradation_guardrail_threshold_drift_require_active": True,
                "ops_degradation_guardrail_threshold_drift_max_staleness_days": 14,
                "ops_degradation_guardrail_threshold_drift_min_burnin_samples": 3,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out.get("passed", True)))
        self.assertFalse(bool(out.get("checks", {}).get("degradation_guardrail_threshold_drift_ok", True)))
        payload = out.get("degradation_guardrail_threshold_drift", {})
        checks = payload.get("checks", {}) if isinstance(payload.get("checks", {}), dict) else {}
        self.assertFalse(bool(checks.get("burnin_samples_ok", True)))
        alerts = payload.get("alerts", []) if isinstance(payload.get("alerts", []), list) else []
        self.assertIn("degradation_guardrail_threshold_drift_insufficient_burnin_samples", alerts)
        self.assertEqual(str(payload.get("status", "")), "insufficient_samples")

    def test_review_until_pass_defect_plan_includes_degradation_guardrail_threshold_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (review_dir / f"{d.isoformat()}_degradation_guardrail_threshold_drift.json").write_text(
            json.dumps(
                {
                    "date": d.isoformat(),
                    "status": "critical",
                    "summary": {"warn_count": 2, "critical_count": 1, "burnin_samples": 4},
                    "alerts": [
                        "degradation_guardrail_threshold_drift_critical:ops_degradation_guardrail_promotion_latency_days_max"
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
                "ops_degradation_guardrail_threshold_drift_enabled": True,
                "ops_degradation_guardrail_threshold_drift_gate_hard_fail": True,
                "ops_degradation_guardrail_threshold_drift_require_active": True,
                "ops_degradation_guardrail_threshold_drift_max_staleness_days": 14,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(bool(out.get("passed", True)))
        round0 = out["rounds"][0]
        plan = json.loads(Path(str(round0["defect_plan"]["json"])).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("DEGRADATION_GUARDRAIL_THRESHOLD_DRIFT_CRITICAL", codes)

    def test_gate_report_compaction_restore_trend_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "weekly_guardrail_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "date": "2026-02-06",
                            "week_tag": "2026-W06",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": False},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "ok",
                                    "checks": {"restore_delta_match": True},
                                },
                            },
                        },
                        {
                            "date": "2026-02-13",
                            "week_tag": "2026-W07",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": False},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "error",
                                    "checks": {"restore_delta_match": False},
                                },
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_compaction_restore_trend_enabled": True,
                "ops_compaction_restore_trend_gate_hard_fail": True,
                "ops_compaction_restore_trend_window_weeks": 8,
                "ops_compaction_restore_trend_min_samples": 2,
                "ops_compaction_restore_trend_min_restore_pass_ratio": 1.0,
                "ops_compaction_restore_trend_min_restore_delta_match_ratio": 1.0,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out.get("passed", True)))
        self.assertFalse(bool(out.get("checks", {}).get("compaction_restore_trend_ok", True)))
        trend = out.get("compaction_restore_trend", {})
        self.assertTrue(bool(trend.get("monitor_failed", False)))
        self.assertFalse(bool(trend.get("gate_ok", True)))
        self.assertTrue(bool(trend.get("hard_fail", False)))
        rollback = out.get("rollback_recommendation", {})
        self.assertIn("compaction_restore_trend", set(rollback.get("reason_codes", [])))

    def test_gate_report_compaction_restore_trend_monitor_mode_does_not_block(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "weekly_guardrail_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "date": "2026-02-06",
                            "week_tag": "2026-W06",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": False},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "ok",
                                    "checks": {"restore_delta_match": True},
                                },
                            },
                        },
                        {
                            "date": "2026-02-13",
                            "week_tag": "2026-W07",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": False},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "error",
                                    "checks": {"restore_delta_match": False},
                                },
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_compaction_restore_trend_enabled": True,
                "ops_compaction_restore_trend_gate_hard_fail": False,
                "ops_compaction_restore_trend_window_weeks": 8,
                "ops_compaction_restore_trend_min_samples": 2,
                "ops_compaction_restore_trend_min_restore_pass_ratio": 1.0,
                "ops_compaction_restore_trend_min_restore_delta_match_ratio": 1.0,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(bool(out.get("passed", False)))
        self.assertTrue(bool(out.get("checks", {}).get("compaction_restore_trend_ok", False)))
        trend = out.get("compaction_restore_trend", {})
        self.assertTrue(bool(trend.get("monitor_failed", False)))
        self.assertTrue(bool(trend.get("gate_ok", False)))
        self.assertFalse(bool(trend.get("hard_fail", True)))

    def test_review_until_pass_defect_plan_includes_compaction_restore_trend(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "weekly_guardrail_state.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "date": "2026-02-06",
                            "week_tag": "2026-W06",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": False},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "ok",
                                    "checks": {"restore_delta_match": True},
                                },
                            },
                        },
                        {
                            "date": "2026-02-13",
                            "week_tag": "2026-W07",
                            "status": "ok",
                            "maintenance": {
                                "compact": {"ran": True, "status": "ok", "dry_run": False},
                                "restore_verify": {
                                    "enabled": True,
                                    "ran": True,
                                    "status": "error",
                                    "checks": {"restore_delta_match": False},
                                },
                            },
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_compaction_restore_trend_enabled": True,
                "ops_compaction_restore_trend_gate_hard_fail": True,
                "ops_compaction_restore_trend_window_weeks": 8,
                "ops_compaction_restore_trend_min_samples": 2,
                "ops_compaction_restore_trend_min_restore_pass_ratio": 1.0,
                "ops_compaction_restore_trend_min_restore_delta_match_ratio": 1.0,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(bool(out.get("passed", True)))
        round0 = out["rounds"][0]
        plan = json.loads(Path(str(round0["defect_plan"]["json"])).read_text(encoding="utf-8"))
        defects = plan.get("defects", []) if isinstance(plan.get("defects", []), list) else []
        codes = {str(x.get("code", "")) for x in defects if isinstance(x, dict)}
        self.assertTrue(any(code.startswith("COMPACTION_RESTORE_TREND") for code in codes))
        compaction_defect = next(
            (
                x
                for x in defects
                if isinstance(x, dict) and str(x.get("code", "")).startswith("COMPACTION_RESTORE_TREND")
            ),
            {},
        )
        auto = compaction_defect.get("auto_remediation", {}) if isinstance(compaction_defect, dict) else {}
        self.assertTrue(bool(auto))
        self.assertTrue(str(auto.get("template_id", "")).startswith("compaction_restore_"))
        patch = auto.get("config_patch", {}) if isinstance(auto.get("config_patch", {}), dict) else {}
        self.assertIn("ops_compaction_restore_trend_gate_hard_fail", patch)

    def test_gate_report_guard_loop_cadence_non_apply_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 4,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": True,
                        "reason_codes": [
                            "CADENCE_DUE_NON_APPLY_STREAK",
                            "CADENCE_DUE_NON_APPLY_HEAVY",
                        ],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": False, "heavy_due": False},
                    },
                    "recovery": {"mode": "none", "status": "skipped", "reason_codes": []},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 4,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [
                        {
                            "ts": "2026-02-13T20:30:00+08:00",
                            "date": "2026-02-13",
                            "cadence_due": True,
                            "cadence_non_apply_streak": 4,
                            "cadence_non_apply_apply_seen": False,
                            "cadence_non_apply_reason_codes": [
                                "CADENCE_DUE_NON_APPLY_STREAK",
                                "CADENCE_DUE_NON_APPLY_HEAVY",
                            ],
                            "recovery_mode": "none",
                        }
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
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out.get("checks", {}).get("guard_loop_cadence_non_apply_ok", True)))
        self.assertFalse(bool(out.get("passed", True)))
        cadence = out.get("guard_loop_cadence_non_apply", {})
        self.assertTrue(bool(cadence.get("monitor_failed", False)))
        self.assertIn("guard_loop_cadence_non_apply_heavy", set(cadence.get("alerts", [])))
        rollback = out.get("rollback_recommendation", {})
        self.assertIn("guard_loop_cadence_non_apply", set(rollback.get("reason_codes", [])))

    def test_gate_ops_report_guard_loop_cadence_preset_scorecard(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 2,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": False,
                        "reason_codes": ["CADENCE_DUE_NON_APPLY_STREAK"],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": False, "heavy_due": False},
                        "trend_preset": {
                            "active": True,
                            "suggested_level": "light",
                            "due_light": True,
                            "due_heavy": False,
                            "replay_allowed": True,
                            "sample_ready": True,
                            "reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                            "metrics": {
                                "applied_rate_delta": -0.22,
                                "cooldown_block_rate_delta": 0.08,
                            },
                        },
                        "trend_retro": {
                            "found": True,
                            "path": str(review_dir / "2026-02-12_autorun_retro.json"),
                            "date": "2026-02-12",
                        },
                    },
                    "recovery": {"mode": "none", "status": "skipped", "reason_codes": []},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 2,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [
                        {
                            "ts": "2026-02-13T20:30:00+08:00",
                            "date": "2026-02-13",
                            "cadence_due": True,
                            "cadence_non_apply_streak": 2,
                            "cadence_non_apply_apply_seen": False,
                            "cadence_non_apply_reason_codes": ["CADENCE_DUE_NON_APPLY_STREAK"],
                            "recovery_mode": "none",
                            "cadence_lift_trend_preset_level": "light",
                            "cadence_lift_trend_due_light": True,
                            "cadence_lift_trend_due_heavy": False,
                            "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
                "ops_guard_loop_cadence_non_apply_require_reason_codes": True,
                "ops_guard_loop_cadence_non_apply_require_recovery_escalation": True,
            }
        )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        gate = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        cadence = gate.get("guard_loop_cadence_non_apply", {})
        cadence_alerts = set(cadence.get("alerts", [])) if isinstance(cadence, dict) else set()
        self.assertIn("guard_loop_cadence_trend_preset_recovery_link_missing", cadence_alerts)
        self.assertFalse(bool(gate.get("checks", {}).get("guard_loop_cadence_non_apply_ok", True)))
        gate_scorecards = (
            gate.get("scorecards", {}).get("guard_loop", {})
            if isinstance(gate.get("scorecards", {}), dict)
            else {}
        )
        preset_card = (
            gate_scorecards.get("cadence_lift_preset", {})
            if isinstance(gate_scorecards.get("cadence_lift_preset", {}), dict)
            else {}
        )
        self.assertEqual(str(preset_card.get("status", "")), "red")
        self.assertFalse(bool(preset_card.get("recovery_link_ok", True)))

        ops = orch.ops_report(as_of=d, window_days=3)
        ops_cards = (
            ops.get("scorecards", {}).get("guard_loop", {})
            if isinstance(ops.get("scorecards", {}), dict)
            else {}
        )
        ops_preset_card = (
            ops_cards.get("cadence_lift_preset", {})
            if isinstance(ops_cards.get("cadence_lift_preset", {}), dict)
            else {}
        )
        self.assertEqual(str(ops_preset_card.get("status", "")), "red")
        self.assertFalse(bool(ops_preset_card.get("recovery_link_ok", True)))
        ops_md = (review_dir / f"{d.isoformat()}_ops_report.md").read_text(encoding="utf-8")
        self.assertIn("trend_preset(checks traceable/recovery/retro): `True` / `False` / `True`", ops_md)

    def test_extract_guard_loop_semantic_payload_legacy_risk_color_marks_semantic_success(self) -> None:
        payload = {
            "daemon": {"would_run_pulse": True},
            "pulse": {"ran": True, "status": "ok", "semantic_ok": True, "reason": "executed"},
            "gap_backfill": {
                "due": True,
                "status": "error",
                "results": {
                    "semantic": {"pulse_ok": True, "autorun_retro_ok": False},
                    "autorun_retro": {"ok": True, "payload": {"status": "red"}},
                },
            },
        }

        extracted = ReleaseOrchestrator._extract_guard_loop_semantic_payload(payload)
        self.assertTrue(bool(extracted.get("gap_backfill_executed", False)))
        self.assertTrue(bool(extracted.get("gap_backfill_pulse_ok", False)))
        self.assertTrue(bool(extracted.get("gap_backfill_autorun_retro_ok", False)))
        self.assertTrue(bool(extracted.get("gap_backfill_semantic_ok", False)))
        self.assertTrue(bool(extracted.get("gap_backfill_legacy_semantic_compat_applied", False)))

    def test_extract_guard_loop_semantic_payload_real_autorun_failure_remains_semantic_failure(self) -> None:
        payload = {
            "daemon": {"would_run_pulse": True},
            "pulse": {"ran": True, "status": "ok", "semantic_ok": True, "reason": "executed"},
            "gap_backfill": {
                "due": True,
                "status": "error",
                "results": {
                    "semantic": {"pulse_ok": True, "autorun_retro_ok": False},
                    "autorun_retro": {"ok": False, "payload": {"status": "error"}},
                },
            },
        }

        extracted = ReleaseOrchestrator._extract_guard_loop_semantic_payload(payload)
        self.assertTrue(bool(extracted.get("gap_backfill_executed", False)))
        self.assertTrue(bool(extracted.get("gap_backfill_pulse_ok", False)))
        self.assertFalse(bool(extracted.get("gap_backfill_autorun_retro_ok", True)))
        self.assertFalse(bool(extracted.get("gap_backfill_semantic_ok", True)))
        self.assertFalse(bool(extracted.get("gap_backfill_legacy_semantic_compat_applied", True)))

    def test_gate_ops_report_guard_loop_semantic_scorecard_and_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "daemon": {"date": "2026-02-13", "would_run_pulse": True},
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": False,
                        "apply_seen": True,
                        "non_apply": False,
                        "streak_windows": 0,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": False,
                        "due_heavy": False,
                        "reason_codes": [],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": False, "heavy_due": False},
                    },
                    "pulse": {
                        "ran": True,
                        "status": "error",
                        "reason": "slot_error",
                        "semantic_ok": False,
                    },
                    "gap_backfill": {
                        "due": True,
                        "status": "error",
                        "results": {
                            "semantic": {"pulse_ok": False, "autorun_retro_ok": False}
                        },
                    },
                    "recovery": {"mode": "none", "status": "skipped", "reason_codes": []},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 0,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        history_rows = [
            {
                "ts": "2026-02-12T20:30:00+08:00",
                "daemon": {"date": "2026-02-12", "would_run_pulse": True},
                "pulse": {"ran": True, "status": "error", "semantic_ok": False, "reason": "slot_error"},
                "gap_backfill": {
                    "due": True,
                    "status": "error",
                    "results": {"semantic": {"pulse_ok": False, "autorun_retro_ok": False}},
                },
            },
            {
                "ts": "2026-02-13T10:30:00+08:00",
                "daemon": {"date": "2026-02-13", "would_run_pulse": True},
                "pulse": {"ran": True, "status": "ok", "semantic_ok": True, "reason": "executed"},
                "gap_backfill": {
                    "due": True,
                    "status": "ok",
                    "results": {"semantic": {"pulse_ok": True, "autorun_retro_ok": True}},
                },
            },
            {
                "ts": "2026-02-13T20:30:00+08:00",
                "daemon": {"date": "2026-02-13", "would_run_pulse": True},
                "pulse": {"ran": True, "status": "error", "semantic_ok": False, "reason": "slot_error"},
                "gap_backfill": {
                    "due": True,
                    "status": "error",
                    "results": {"semantic": {"pulse_ok": False, "autorun_retro_ok": False}},
                },
            },
        ]
        with (logs_dir / "guard_loop_history.jsonl").open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_semantic_drift_window_days": 14,
                "ops_guard_loop_cadence_non_apply_semantic_drift_min_samples": 2,
                "ops_guard_loop_cadence_non_apply_semantic_drift_max_pulse_fail_ratio": 0.20,
                "ops_guard_loop_cadence_non_apply_semantic_drift_max_gap_backfill_fail_ratio": 0.20,
            }
        )

        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        gate = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        cadence = gate.get("guard_loop_cadence_non_apply", {})
        alerts = set(cadence.get("alerts", [])) if isinstance(cadence, dict) else set()
        self.assertIn("guard_loop_cadence_pulse_semantic_failed", alerts)
        self.assertIn("guard_loop_cadence_gap_backfill_semantic_failed", alerts)
        self.assertIn("guard_loop_cadence_pulse_semantic_drift", alerts)
        self.assertIn("guard_loop_cadence_gap_backfill_semantic_drift", alerts)
        self.assertFalse(bool(gate.get("checks", {}).get("guard_loop_cadence_non_apply_ok", True)))
        gate_cards = gate.get("scorecards", {}) if isinstance(gate.get("scorecards", {}), dict) else {}
        guard_loop_cards = gate_cards.get("guard_loop", {}) if isinstance(gate_cards.get("guard_loop", {}), dict) else {}
        semantic_card = (
            guard_loop_cards.get("pulse_backfill_semantic", {})
            if isinstance(guard_loop_cards.get("pulse_backfill_semantic", {}), dict)
            else {}
        )
        self.assertEqual(str(semantic_card.get("status", "")), "red")
        self.assertFalse(bool(semantic_card.get("ok", True)))

        ops = orch.ops_report(as_of=d, window_days=3)
        ops_cards = ops.get("scorecards", {}) if isinstance(ops.get("scorecards", {}), dict) else {}
        ops_guard_loop_cards = (
            ops_cards.get("guard_loop", {})
            if isinstance(ops_cards.get("guard_loop", {}), dict)
            else {}
        )
        ops_semantic_card = (
            ops_guard_loop_cards.get("pulse_backfill_semantic", {})
            if isinstance(ops_guard_loop_cards.get("pulse_backfill_semantic", {}), dict)
            else {}
        )
        self.assertEqual(str(ops_semantic_card.get("status", "")), "red")
        ops_md = (review_dir / f"{d.isoformat()}_ops_report.md").read_text(encoding="utf-8")
        self.assertIn("pulse_semantic(status/ran/ok/reason):", ops_md)
        self.assertIn("semantic_drift(pulse samples/failures/fail_ratio/check):", ops_md)

    def test_review_until_pass_defect_plan_includes_guard_loop_pulse_semantic_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "daemon": {"date": "2026-02-13", "would_run_pulse": True},
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": False,
                        "apply_seen": True,
                        "non_apply": False,
                        "streak_windows": 0,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": False,
                        "due_heavy": False,
                        "reason_codes": [],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": False, "heavy_due": False},
                    },
                    "pulse": {"ran": True, "status": "ok", "reason": "executed", "semantic_ok": True},
                    "gap_backfill": {"due": False, "status": "skipped", "results": {}},
                    "recovery": {"mode": "none", "status": "skipped", "reason_codes": []},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 0,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        with (logs_dir / "guard_loop_history.jsonl").open("w", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    {
                        "ts": "2026-02-12T20:30:00+08:00",
                        "daemon": {"date": "2026-02-12", "would_run_pulse": True},
                        "pulse": {
                            "ran": True,
                            "status": "error",
                            "reason": "slot_error",
                            "semantic_ok": False,
                        },
                        "gap_backfill": {"due": False, "status": "skipped", "results": {}},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fp.write(
                json.dumps(
                    {
                        "ts": "2026-02-13T10:30:00+08:00",
                        "daemon": {"date": "2026-02-13", "would_run_pulse": True},
                        "pulse": {
                            "ran": True,
                            "status": "ok",
                            "reason": "executed",
                            "semantic_ok": True,
                        },
                        "gap_backfill": {"due": False, "status": "skipped", "results": {}},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        def _load_json(path: Path) -> dict[str, object]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 999999,
                "ops_guard_loop_cadence_non_apply_semantic_drift_window_days": 14,
                "ops_guard_loop_cadence_non_apply_semantic_drift_min_samples": 2,
                "ops_guard_loop_cadence_non_apply_semantic_drift_max_pulse_fail_ratio": 0.20,
                "ops_guard_loop_cadence_non_apply_semantic_drift_max_gap_backfill_fail_ratio": 0.35,
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 1, "stderr": "test_x ... FAIL", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(bool(out.get("passed", True)))
        round0 = out["rounds"][0]
        plan = json.loads(Path(str(round0["defect_plan"]["json"])).read_text(encoding="utf-8"))
        codes = {
            str(x.get("code", ""))
            for x in plan.get("defects", [])
            if isinstance(x, dict)
        }
        self.assertIn("GUARD_LOOP_CADENCE_PULSE_SEMANTIC_DRIFT", codes)

    def test_gate_report_guard_loop_cadence_preset_drift_status_bad(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 2,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": False,
                        "reason_codes": ["CADENCE_DUE_NON_APPLY_STREAK"],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": True, "heavy_due": False},
                        "trend_preset": {
                            "active": True,
                            "suggested_level": "light",
                            "due_light": True,
                            "due_heavy": False,
                            "replay_allowed": True,
                            "sample_ready": True,
                            "reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                            "metrics": {
                                "applied_rate_delta": -0.20,
                                "cooldown_block_rate_delta": 0.10,
                            },
                        },
                        "trend_retro": {"found": True, "path": str(review_dir / "2026-02-12_autorun_retro.json"), "date": "2026-02-12"},
                        "trend_preset_drift_audit": {
                            "status": "warn",
                            "samples": 8,
                            "min_samples": 6,
                            "checks": {
                                "min_samples_ok": True,
                                "recovery_link_ok": False,
                                "retro_coverage_ok": True,
                            },
                            "alerts": ["guard_loop_preset_drift_recovery_link_low"],
                            "artifact": {"written": True, "json": str(review_dir / "2026-02-13_guard_loop_preset_drift.json"), "md": str(review_dir / "2026-02-13_guard_loop_preset_drift.md")},
                        },
                    },
                    "recovery": {"mode": "light", "status": "ok", "reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"]},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 2,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
                "ops_guard_loop_cadence_non_apply_require_reason_codes": True,
                "ops_guard_loop_cadence_non_apply_require_recovery_escalation": True,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out.get("checks", {}).get("guard_loop_cadence_non_apply_ok", True)))
        cadence = out.get("guard_loop_cadence_non_apply", {})
        alerts = set(cadence.get("alerts", [])) if isinstance(cadence, dict) else set()
        self.assertIn("guard_loop_cadence_trend_preset_drift_status_bad", alerts)
        scorecards = out.get("scorecards", {}) if isinstance(out.get("scorecards", {}), dict) else {}
        guard_loop_card = scorecards.get("guard_loop", {}) if isinstance(scorecards.get("guard_loop", {}), dict) else {}
        preset_card = (
            guard_loop_card.get("cadence_lift_preset", {})
            if isinstance(guard_loop_card.get("cadence_lift_preset", {}), dict)
            else {}
        )
        self.assertFalse(bool(preset_card.get("drift_status_ok", True)))
        self.assertEqual(str(preset_card.get("drift_status", "")), "warn")

    def test_gate_report_guard_loop_cadence_preset_drift_autotune_unbounded(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 2,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": False,
                        "reason_codes": ["CADENCE_DUE_NON_APPLY_STREAK"],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": True, "heavy_due": False},
                        "trend_preset": {
                            "active": True,
                            "suggested_level": "light",
                            "due_light": True,
                            "due_heavy": False,
                            "replay_allowed": True,
                            "sample_ready": True,
                            "reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                            "metrics": {
                                "applied_rate_delta": -0.20,
                                "cooldown_block_rate_delta": 0.10,
                            },
                        },
                        "trend_retro": {
                            "found": True,
                            "path": str(review_dir / "2026-02-12_autorun_retro.json"),
                            "date": "2026-02-12",
                        },
                        "trend_preset_drift_audit": {
                            "status": "ok",
                            "samples": 12,
                            "min_samples": 6,
                            "checks": {
                                "min_samples_ok": True,
                                "recovery_link_ok": True,
                                "retro_coverage_ok": True,
                            },
                            "alerts": [],
                            "auto_tune": {
                                "enabled": True,
                                "ready": True,
                                "apply_recommended": True,
                                "reason": "density_out_of_band",
                                "policy": {
                                    "step_max": 0.03,
                                    "hit_rate_low": 0.20,
                                    "hit_rate_high": 0.60,
                                    "applied_gap_min": 0.05,
                                    "cooldown_gap_min": 0.05,
                                },
                                "bounds": {
                                    "light_applied_delta_max": {"min": -0.18, "max": -0.12},
                                    "heavy_applied_delta_max": {"min": -0.33, "max": -0.27},
                                    "light_cooldown_delta_min": {"min": 0.12, "max": 0.18},
                                    "heavy_cooldown_delta_min": {"min": 0.27, "max": 0.33},
                                },
                                "suggested_thresholds": {
                                    "light_applied_delta_max": -0.40,
                                    "heavy_applied_delta_max": -0.50,
                                    "light_cooldown_delta_min": 0.10,
                                    "heavy_cooldown_delta_min": 0.20,
                                },
                            },
                            "artifact": {
                                "written": True,
                                "json": str(review_dir / "2026-02-13_guard_loop_preset_drift.json"),
                                "md": str(review_dir / "2026-02-13_guard_loop_preset_drift.md"),
                            },
                        },
                    },
                    "recovery": {
                        "mode": "light",
                        "status": "ok",
                        "reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 2,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
                "ops_guard_loop_cadence_non_apply_require_reason_codes": True,
                "ops_guard_loop_cadence_non_apply_require_recovery_escalation": True,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        cadence = out.get("guard_loop_cadence_non_apply", {})
        alerts = set(cadence.get("alerts", [])) if isinstance(cadence, dict) else set()
        self.assertIn("guard_loop_cadence_trend_preset_drift_auto_tune_unbounded", alerts)
        checks = cadence.get("checks", {}) if isinstance(cadence, dict) else {}
        self.assertFalse(bool(checks.get("trend_preset_drift_auto_tune_bounded_ok", True)))
        scorecards = out.get("scorecards", {}) if isinstance(out.get("scorecards", {}), dict) else {}
        guard_loop_card = scorecards.get("guard_loop", {}) if isinstance(scorecards.get("guard_loop", {}), dict) else {}
        preset_card = (
            guard_loop_card.get("cadence_lift_preset", {})
            if isinstance(guard_loop_card.get("cadence_lift_preset", {}), dict)
            else {}
        )
        self.assertFalse(bool(preset_card.get("drift_auto_tune_bounded_ok", True)))

    def test_gate_report_guard_loop_cadence_preset_drift_trendline_bad(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 2,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": False,
                        "reason_codes": ["CADENCE_DUE_NON_APPLY_STREAK"],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": True, "heavy_due": False},
                        "trend_preset": {
                            "active": True,
                            "suggested_level": "light",
                            "due_light": True,
                            "due_heavy": False,
                            "replay_allowed": True,
                            "sample_ready": True,
                            "reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                            "metrics": {
                                "applied_rate_delta": -0.20,
                                "cooldown_block_rate_delta": 0.10,
                            },
                        },
                        "trend_retro": {
                            "found": True,
                            "path": str(review_dir / "2026-02-12_autorun_retro.json"),
                            "date": "2026-02-12",
                        },
                        "trend_preset_drift_audit": {
                            "status": "ok",
                            "samples": 12,
                            "min_samples": 6,
                            "checks": {
                                "min_samples_ok": True,
                                "recovery_link_ok": True,
                                "retro_coverage_ok": True,
                            },
                            "alerts": [],
                            "trendline": {
                                "status": "warn",
                                "checks": {
                                    "samples_ready_ok": True,
                                    "recovery_link_drop_ok": False,
                                    "retro_found_drop_ok": True,
                                },
                                "alerts": ["guard_loop_preset_drift_trendline_recovery_link_drop"],
                                "windows": {
                                    "recent": {"samples": 4},
                                    "prior": {"samples": 4},
                                },
                                "deltas": {
                                    "recovery_link_light_rate": -0.50,
                                    "recovery_link_heavy_rate": 0.00,
                                    "retro_found_rate": -0.25,
                                },
                            },
                            "auto_tune": {
                                "enabled": True,
                                "ready": True,
                                "apply_recommended": False,
                                "reason": "trendline_guardrail_blocked",
                            },
                            "artifact": {
                                "written": True,
                                "json": str(review_dir / "2026-02-13_guard_loop_preset_drift.json"),
                                "md": str(review_dir / "2026-02-13_guard_loop_preset_drift.md"),
                            },
                        },
                    },
                    "recovery": {
                        "mode": "light",
                        "status": "ok",
                        "reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 2,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
                "ops_guard_loop_cadence_non_apply_require_reason_codes": True,
                "ops_guard_loop_cadence_non_apply_require_recovery_escalation": True,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        cadence = out.get("guard_loop_cadence_non_apply", {})
        alerts = set(cadence.get("alerts", [])) if isinstance(cadence, dict) else set()
        self.assertIn("guard_loop_cadence_trend_preset_drift_trendline_bad", alerts)
        checks = cadence.get("checks", {}) if isinstance(cadence, dict) else {}
        self.assertFalse(bool(checks.get("trend_preset_drift_trendline_ok", True)))
        scorecards = out.get("scorecards", {}) if isinstance(out.get("scorecards", {}), dict) else {}
        guard_loop_card = scorecards.get("guard_loop", {}) if isinstance(scorecards.get("guard_loop", {}), dict) else {}
        preset_card = (
            guard_loop_card.get("cadence_lift_preset", {})
            if isinstance(guard_loop_card.get("cadence_lift_preset", {}), dict)
            else {}
        )
        self.assertFalse(bool(preset_card.get("drift_trendline_ok", True)))

    def test_gate_report_guard_loop_cadence_preset_drift_autotune_handoff_invalid(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 2,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": False,
                        "reason_codes": ["CADENCE_DUE_NON_APPLY_STREAK"],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": True, "heavy_due": False},
                        "trend_preset": {
                            "active": True,
                            "suggested_level": "light",
                            "due_light": True,
                            "due_heavy": False,
                            "replay_allowed": True,
                            "sample_ready": True,
                            "reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                            "metrics": {
                                "applied_rate_delta": -0.20,
                                "cooldown_block_rate_delta": 0.10,
                            },
                        },
                        "trend_retro": {
                            "found": True,
                            "path": str(review_dir / "2026-02-12_autorun_retro.json"),
                            "date": "2026-02-12",
                        },
                        "trend_preset_drift_audit": {
                            "status": "ok",
                            "samples": 12,
                            "min_samples": 6,
                            "checks": {
                                "min_samples_ok": True,
                                "recovery_link_ok": True,
                                "retro_coverage_ok": True,
                            },
                            "alerts": [],
                            "auto_tune": {
                                "enabled": True,
                                "ready": True,
                                "apply_recommended": True,
                                "reason": "density_out_of_band",
                                "policy": {
                                    "step_max": 0.03,
                                    "hit_rate_low": 0.20,
                                    "hit_rate_high": 0.60,
                                    "applied_gap_min": 0.05,
                                    "cooldown_gap_min": 0.05,
                                },
                                "bounds": {
                                    "light_applied_delta_max": {"min": -0.18, "max": -0.12},
                                    "heavy_applied_delta_max": {"min": -0.33, "max": -0.27},
                                    "light_cooldown_delta_min": {"min": 0.12, "max": 0.18},
                                    "heavy_cooldown_delta_min": {"min": 0.27, "max": 0.33},
                                },
                                "suggested_thresholds": {
                                    "light_applied_delta_max": -0.12,
                                    "heavy_applied_delta_max": -0.27,
                                    "light_cooldown_delta_min": 0.18,
                                    "heavy_cooldown_delta_min": 0.33,
                                },
                                "handoff": {
                                    "enabled": True,
                                    "mode": "proposal_only",
                                    "proposal_generated": True,
                                    "proposal_id": "",
                                    "apply_gate": {
                                        "allowed": True,
                                        "reason": "",
                                    },
                                },
                            },
                            "artifact": {
                                "written": True,
                                "json": str(review_dir / "2026-02-13_guard_loop_preset_drift.json"),
                                "md": str(review_dir / "2026-02-13_guard_loop_preset_drift.md"),
                            },
                        },
                    },
                    "recovery": {
                        "mode": "light",
                        "status": "ok",
                        "reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 2,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
                "ops_guard_loop_cadence_non_apply_require_reason_codes": True,
                "ops_guard_loop_cadence_non_apply_require_recovery_escalation": True,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )

        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        cadence = out.get("guard_loop_cadence_non_apply", {})
        alerts = set(cadence.get("alerts", [])) if isinstance(cadence, dict) else set()
        self.assertIn("guard_loop_cadence_trend_preset_drift_auto_tune_handoff_invalid", alerts)
        checks = cadence.get("checks", {}) if isinstance(cadence, dict) else {}
        self.assertFalse(bool(checks.get("trend_preset_drift_auto_tune_handoff_ok", True)))
        scorecards = out.get("scorecards", {}) if isinstance(out.get("scorecards", {}), dict) else {}
        guard_loop_card = scorecards.get("guard_loop", {}) if isinstance(scorecards.get("guard_loop", {}), dict) else {}
        preset_card = (
            guard_loop_card.get("cadence_lift_preset", {})
            if isinstance(guard_loop_card.get("cadence_lift_preset", {}), dict)
            else {}
        )
        self.assertFalse(bool(preset_card.get("drift_auto_tune_handoff_ok", True)))

    def test_review_until_pass_defect_plan_includes_guard_loop_cadence_non_apply(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 4,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": True,
                        "reason_codes": [
                            "CADENCE_DUE_NON_APPLY_STREAK",
                            "CADENCE_DUE_NON_APPLY_HEAVY",
                        ],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": False, "heavy_due": False},
                    },
                    "recovery": {"mode": "none", "status": "skipped", "reason_codes": []},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 4,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [
                        {
                            "ts": "2026-02-13T20:30:00+08:00",
                            "date": "2026-02-13",
                            "cadence_due": True,
                            "cadence_non_apply_streak": 4,
                            "cadence_non_apply_apply_seen": False,
                            "cadence_non_apply_reason_codes": [
                                "CADENCE_DUE_NON_APPLY_STREAK",
                                "CADENCE_DUE_NON_APPLY_HEAVY",
                            ],
                            "recovery_mode": "none",
                        }
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
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 1, "stderr": "test_x ... FAIL", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(bool(out.get("passed", True)))
        round0 = out["rounds"][0]
        self.assertTrue(bool(round0.get("guard_loop_cadence_non_apply_active", False)))
        self.assertFalse(bool(round0.get("guard_loop_cadence_non_apply_ok", True)))
        plan = json.loads(Path(str(round0["defect_plan"]["json"])).read_text(encoding="utf-8"))
        defects = plan.get("defects", []) if isinstance(plan.get("defects", []), list) else []
        codes = {str(x.get("code", "")) for x in defects if isinstance(x, dict)}
        self.assertTrue(any(code.startswith("GUARD_LOOP_CADENCE_NON_APPLY") for code in codes))
        lift_snapshot = plan.get("cadence_non_apply_lift", {}) if isinstance(plan.get("cadence_non_apply_lift", {}), dict) else {}
        self.assertIn("applied", lift_snapshot)
        self.assertIn("blocked_by_cooldown", lift_snapshot)
        self.assertIn("cooldown_remaining_days", lift_snapshot)
        cadence_defect = next(
            (
                x
                for x in defects
                if isinstance(x, dict) and str(x.get("code", "")).startswith("GUARD_LOOP_CADENCE_NON_APPLY")
            ),
            {},
        )
        auto = cadence_defect.get("auto_remediation", {}) if isinstance(cadence_defect, dict) else {}
        self.assertEqual(str(auto.get("template_id", "")), "guard_loop_cadence_non_apply_recover")
        next_actions = [str(x) for x in plan.get("next_actions", [])]
        self.assertTrue(next_actions)
        self.assertIn("guard-loop cadence_non_apply", next_actions[0])
        plan_md = Path(str(round0["defect_plan"]["md"])).read_text(encoding="utf-8")
        self.assertIn("Guard Loop Cadence Lift:", plan_md)
        self.assertIn("cooldown_remaining_days=", plan_md)

    def test_gate_report_guard_loop_cadence_non_apply_rollback_lift_escalates_to_hard(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 5,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": True,
                        "reason_codes": [
                            "CADENCE_DUE_NON_APPLY_STREAK",
                            "CADENCE_DUE_NON_APPLY_HEAVY",
                        ],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": True, "heavy_due": True},
                    },
                    "recovery": {
                        "mode": "heavy",
                        "status": "ok",
                        "reason_codes": ["CADENCE_DUE_NON_APPLY_REPLAY_HEAVY"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 5,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [
                        {
                            "ts": "2026-02-13T20:30:00+08:00",
                            "date": "2026-02-13",
                            "cadence_due": True,
                            "cadence_non_apply_streak": 5,
                            "cadence_non_apply_apply_seen": False,
                            "cadence_non_apply_reason_codes": [
                                "CADENCE_DUE_NON_APPLY_STREAK",
                                "CADENCE_DUE_NON_APPLY_HEAVY",
                            ],
                            "recovery_mode": "heavy",
                        }
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
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
                "ops_guard_loop_cadence_non_apply_rollback_lift_enabled": True,
                "ops_guard_loop_cadence_non_apply_rollback_lift_light_streak_min": 2,
                "ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min": 4,
                "ops_guard_loop_cadence_non_apply_rollback_lift_cooldown_days": 2,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        rollback = out.get("rollback_recommendation", {})
        self.assertEqual(str(rollback.get("level", "")), "hard")
        self.assertIn("guard_loop_cadence_non_apply_lift_hard", set(rollback.get("reason_codes", [])))
        lift = rollback.get("cadence_non_apply_lift", {}) if isinstance(rollback.get("cadence_non_apply_lift", {}), dict) else {}
        self.assertTrue(bool(lift.get("applied", False)))
        self.assertFalse(bool(lift.get("blocked_by_cooldown", True)))
        self.assertEqual(str(lift.get("requested_level", "")), "hard")
        self.assertEqual(str(lift.get("applied_level", "")), "hard")

    def test_gate_report_guard_loop_cadence_non_apply_rollback_lift_respects_cooldown(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        prev = d - timedelta(days=1)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (review_dir / f"{prev.isoformat()}_release_decision_snapshot.json").write_text(
            json.dumps(
                {
                    "date": prev.isoformat(),
                    "decision_id": f"{prev.isoformat()}_seed",
                    "rollback_recommendation": {
                        "level": "hard",
                        "reason_codes": ["guard_loop_cadence_non_apply_lift_hard"],
                        "cadence_non_apply_lift": {
                            "applied": True,
                            "requested_level": "hard",
                            "applied_level": "hard",
                            "reason_code": "guard_loop_cadence_non_apply_lift_hard",
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_loop_last.json").write_text(
            json.dumps(
                {
                    "ts": "2026-02-13T20:30:00+08:00",
                    "cadence_non_apply": {
                        "window_key": "2026-02-13@20:30",
                        "cadence_due": True,
                        "apply_seen": False,
                        "non_apply": True,
                        "streak_windows": 5,
                        "light_threshold": 2,
                        "heavy_threshold": 4,
                        "due_light": True,
                        "due_heavy": True,
                        "reason_codes": [
                            "CADENCE_DUE_NON_APPLY_STREAK",
                            "CADENCE_DUE_NON_APPLY_HEAVY",
                        ],
                        "replay_allowed": True,
                        "recovery_escalation": {"light_due": True, "heavy_due": True},
                    },
                    "recovery": {
                        "mode": "heavy",
                        "status": "ok",
                        "reason_codes": ["CADENCE_DUE_NON_APPLY_REPLAY_HEAVY"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (logs_dir / "guard_state.json").write_text(
            json.dumps(
                {
                    "cadence_non_apply_streak": 5,
                    "cadence_non_apply_last_window_key": "2026-02-13@20:30",
                    "history": [
                        {
                            "ts": "2026-02-13T20:30:00+08:00",
                            "date": "2026-02-13",
                            "cadence_due": True,
                            "cadence_non_apply_streak": 5,
                            "cadence_non_apply_apply_seen": False,
                            "cadence_non_apply_reason_codes": [
                                "CADENCE_DUE_NON_APPLY_STREAK",
                                "CADENCE_DUE_NON_APPLY_HEAVY",
                            ],
                            "recovery_mode": "heavy",
                        }
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
                "ops_guard_loop_cadence_non_apply_enabled": True,
                "ops_guard_loop_cadence_non_apply_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_max_staleness_hours": 48,
                "ops_guard_loop_cadence_non_apply_light_streak_threshold": 2,
                "ops_guard_loop_cadence_non_apply_heavy_streak_threshold": 4,
                "ops_guard_loop_cadence_non_apply_rollback_lift_enabled": True,
                "ops_guard_loop_cadence_non_apply_rollback_lift_light_streak_min": 2,
                "ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min": 4,
                "ops_guard_loop_cadence_non_apply_rollback_lift_cooldown_days": 3,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        rollback = out.get("rollback_recommendation", {})
        self.assertEqual(str(rollback.get("level", "")), "none")
        self.assertIn("guard_loop_cadence_non_apply_lift_cooldown", set(rollback.get("reason_codes", [])))
        lift = rollback.get("cadence_non_apply_lift", {}) if isinstance(rollback.get("cadence_non_apply_lift", {}), dict) else {}
        self.assertFalse(bool(lift.get("applied", True)))
        self.assertTrue(bool(lift.get("blocked_by_cooldown", False)))
        self.assertEqual(int(lift.get("cooldown_remaining_days", 0)), 2)
        ops = orch.ops_report(as_of=d, window_days=3)
        ops_lift = (
            ops.get("cadence_non_apply_lift_snapshot", {})
            if isinstance(ops.get("cadence_non_apply_lift_snapshot", {}), dict)
            else {}
        )
        self.assertFalse(bool(ops_lift.get("applied", True)))
        self.assertTrue(bool(ops_lift.get("blocked_by_cooldown", False)))
        self.assertEqual(int(ops_lift.get("cooldown_remaining_days", 0)), 2)
        ops_md = (review_dir / f"{d.isoformat()}_ops_report.md").read_text(encoding="utf-8")
        self.assertIn(
            "cadence_lift_snapshot(applied/blocked/cooldown_remaining): `false/true/2d`",
            ops_md,
        )

    def test_gate_report_guard_loop_cadence_lift_trend_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        for i in range(2):
            day = d - timedelta(days=i + 1)
            (review_dir / f"{day.isoformat()}_release_decision_snapshot.json").write_text(
                json.dumps(
                    {
                        "date": day.isoformat(),
                        "decision_id": f"{day.isoformat()}_seed",
                        "rollback_recommendation": {
                            "level": "none",
                            "reason_codes": ["guard_loop_cadence_non_apply_lift_cooldown"],
                            "cadence_non_apply_lift": {
                                "requested_level": "hard",
                                "applied_level": "none",
                                "applied": False,
                                "blocked_by_cooldown": True,
                                "reason_code": "guard_loop_cadence_non_apply_lift_cooldown",
                            },
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
                "ops_guard_loop_cadence_non_apply_lift_trend_enabled": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_require_active": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_window_days": 7,
                "ops_guard_loop_cadence_non_apply_lift_trend_min_samples": 2,
                "ops_guard_loop_cadence_non_apply_lift_trend_applied_rate_min": 0.50,
                "ops_guard_loop_cadence_non_apply_lift_trend_cooldown_block_rate_max": 0.40,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertFalse(bool(out["checks"].get("guard_loop_cadence_non_apply_lift_trend_ok", True)))
        self.assertFalse(bool(out["passed"]))
        trend = out.get("guard_loop_cadence_non_apply_lift_trend", {})
        self.assertTrue(bool(trend.get("active", False)))
        self.assertTrue(bool(trend.get("monitor_failed", False)))
        alerts = set(trend.get("alerts", []))
        self.assertIn("guard_loop_cadence_lift_trend_applied_rate_low", alerts)
        self.assertIn("guard_loop_cadence_lift_trend_cooldown_block_rate_high", alerts)
        rollback = out.get("rollback_recommendation", {})
        self.assertIn("guard_loop_cadence_non_apply_lift_trend", set(rollback.get("reason_codes", [])))

    def test_gate_report_guard_loop_cadence_lift_trend_no_requests_is_neutral(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        for i in range(2):
            day = d - timedelta(days=i + 1)
            (review_dir / f"{day.isoformat()}_release_decision_snapshot.json").write_text(
                json.dumps(
                    {
                        "date": day.isoformat(),
                        "decision_id": f"{day.isoformat()}_seed",
                        "rollback_recommendation": {
                            "level": "none",
                            "reason_codes": [],
                            "cadence_non_apply_lift": {
                                "requested_level": "none",
                                "applied_level": "none",
                                "applied": False,
                                "blocked_by_cooldown": False,
                                "reason_code": "",
                            },
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
                "ops_guard_loop_cadence_non_apply_lift_trend_enabled": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_require_active": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_window_days": 7,
                "ops_guard_loop_cadence_non_apply_lift_trend_min_samples": 2,
                "ops_guard_loop_cadence_non_apply_lift_trend_applied_rate_min": 0.50,
                "ops_guard_loop_cadence_non_apply_lift_trend_cooldown_block_rate_max": 0.40,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        self.assertTrue(bool(out["checks"].get("guard_loop_cadence_non_apply_lift_trend_ok", False)))
        trend = (
            out.get("guard_loop_cadence_non_apply_lift_trend", {})
            if isinstance(out.get("guard_loop_cadence_non_apply_lift_trend", {}), dict)
            else {}
        )
        self.assertTrue(bool(trend.get("active", False)))
        self.assertFalse(bool(trend.get("monitor_failed", True)))
        alerts = set(trend.get("alerts", []))
        self.assertNotIn("guard_loop_cadence_lift_trend_applied_rate_low", alerts)
        metrics = trend.get("metrics", {}) if isinstance(trend.get("metrics", {}), dict) else {}
        self.assertEqual(int(metrics.get("requested_count", 0)), 0)
        self.assertEqual(float(metrics.get("applied_rate", 0.0)), 1.0)

    def test_gate_report_guard_loop_cadence_lift_trend_artifact_governance(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        (review_dir / "2026-02-10_cadence_lift_trend.json").write_text(
            json.dumps({"legacy": True}, ensure_ascii=False),
            encoding="utf-8",
        )
        (review_dir / "2026-02-10_cadence_lift_trend.md").write_text(
            "# legacy\n",
            encoding="utf-8",
        )
        for i in range(3):
            day = d - timedelta(days=i + 1)
            reason_codes = ["guard_loop_cadence_non_apply_lift_soft"] if i == 0 else []
            requested_level = "soft" if i == 0 else "none"
            (review_dir / f"{day.isoformat()}_release_decision_snapshot.json").write_text(
                json.dumps(
                    {
                        "date": day.isoformat(),
                        "decision_id": f"{day.isoformat()}_seed",
                        "rollback_recommendation": {
                            "level": "none",
                            "reason_codes": reason_codes,
                            "cadence_non_apply_lift": {
                                "requested_level": requested_level,
                                "applied_level": requested_level,
                                "applied": bool(i == 0),
                                "blocked_by_cooldown": False,
                                "reason_code": reason_codes[0] if reason_codes else "",
                            },
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
                "ops_guard_loop_cadence_non_apply_lift_trend_enabled": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_require_active": False,
                "ops_guard_loop_cadence_non_apply_lift_trend_window_days": 7,
                "ops_guard_loop_cadence_non_apply_lift_trend_min_samples": 1,
                "ops_guard_loop_cadence_non_apply_lift_trend_applied_rate_min": 0.0,
                "ops_guard_loop_cadence_non_apply_lift_trend_cooldown_block_rate_max": 1.0,
                "ops_guard_loop_cadence_non_apply_lift_trend_retention_days": 2,
                "ops_guard_loop_cadence_non_apply_lift_trend_checksum_index_enabled": True,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.gate_report(as_of=d, run_tests=False, run_review_if_missing=False)
        trend = (
            out.get("guard_loop_cadence_non_apply_lift_trend", {})
            if isinstance(out.get("guard_loop_cadence_non_apply_lift_trend", {}), dict)
            else {}
        )
        trend_checks = trend.get("checks", {}) if isinstance(trend.get("checks", {}), dict) else {}
        self.assertTrue(bool(trend_checks.get("artifact_rotation_ok", False)))
        self.assertTrue(bool(trend_checks.get("artifact_checksum_index_ok", False)))
        artifacts = trend.get("artifacts", {}) if isinstance(trend.get("artifacts", {}), dict) else {}
        trend_artifact = artifacts.get("trend", {}) if isinstance(artifacts.get("trend", {}), dict) else {}
        self.assertTrue(bool(trend_artifact.get("written", False)))
        self.assertEqual(int(trend_artifact.get("retention_days", 0)), 2)
        self.assertGreaterEqual(int(trend_artifact.get("rotated_out_count", 0)), 1)
        self.assertFalse(bool(trend_artifact.get("rotation_failed", True)))
        self.assertTrue(bool(trend_artifact.get("checksum_index_written", False)))
        index_path = Path(str(trend_artifact.get("checksum_index_path", "")))
        self.assertTrue(index_path.exists())
        governance = out.get("artifact_governance", {}) if isinstance(out.get("artifact_governance", {}), dict) else {}
        profiles = governance.get("profiles", []) if isinstance(governance.get("profiles", []), list) else []
        row = next(
            (x for x in profiles if isinstance(x, dict) and str(x.get("profile", "")) == "guard_loop_cadence_lift_trend"),
            {},
        )
        self.assertTrue(bool(row))
        self.assertTrue(bool(row.get("artifact_present", False)))
        self.assertFalse(bool(row.get("policy_mismatch", True)))

    def test_review_until_pass_defect_plan_includes_guard_loop_cadence_lift_trend(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text(
            "pass_gate: true\nmode_health:\n  passed: true\n",
            encoding="utf-8",
        )
        for i in range(2):
            day = d - timedelta(days=i + 1)
            (review_dir / f"{day.isoformat()}_release_decision_snapshot.json").write_text(
                json.dumps(
                    {
                        "date": day.isoformat(),
                        "decision_id": f"{day.isoformat()}_seed",
                        "rollback_recommendation": {
                            "level": "none",
                            "reason_codes": ["guard_loop_cadence_non_apply_lift_cooldown"],
                            "cadence_non_apply_lift": {
                                "requested_level": "hard",
                                "applied_level": "none",
                                "applied": False,
                                "blocked_by_cooldown": True,
                                "reason_code": "guard_loop_cadence_non_apply_lift_cooldown",
                            },
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
                "ops_guard_loop_cadence_non_apply_lift_trend_enabled": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_gate_hard_fail": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_require_active": True,
                "ops_guard_loop_cadence_non_apply_lift_trend_window_days": 7,
                "ops_guard_loop_cadence_non_apply_lift_trend_min_samples": 2,
                "ops_guard_loop_cadence_non_apply_lift_trend_applied_rate_min": 0.50,
                "ops_guard_loop_cadence_non_apply_lift_trend_cooldown_block_rate_max": 0.40,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of, parameter_changes={}, factor_weights={}, defects=[], pass_gate=True
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days: {"passed": True, "replay_days": 3, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        out = orch.review_until_pass(as_of=d, max_rounds=1)
        self.assertFalse(bool(out.get("passed", True)))
        round0 = out["rounds"][0]
        self.assertTrue(bool(round0.get("guard_loop_cadence_non_apply_lift_trend_active", False)))
        self.assertFalse(bool(round0.get("guard_loop_cadence_non_apply_lift_trend_ok", True)))
        plan = json.loads(Path(str(round0["defect_plan"]["json"])).read_text(encoding="utf-8"))
        defects = plan.get("defects", []) if isinstance(plan.get("defects", []), list) else []
        codes = {str(x.get("code", "")) for x in defects if isinstance(x, dict)}
        self.assertTrue(any(code.startswith("GUARD_LOOP_CADENCE_LIFT_TREND") for code in codes))
        lift_trend = (
            plan.get("guard_loop_cadence_non_apply_lift_trend", {})
            if isinstance(plan.get("guard_loop_cadence_non_apply_lift_trend", {}), dict)
            else {}
        )
        self.assertTrue(bool(lift_trend.get("active", False)))

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

    def test_gate_report_slot_soft_quality_flags_do_not_count_as_anomaly(self) -> None:
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

        soft_flag = "LOW_CONFIDENCE_SOURCE_RATIO_HIGH:0.5000"
        for i in range(3):
            day = date.fromordinal(d.toordinal() - i)
            dstr = day.isoformat()
            (logs_dir / f"{dstr}_premarket.json").write_text(
                json.dumps(
                    {
                        "quality": {"passed": True, "flags": [soft_flag], "source_confidence_score": 0.90},
                        "risk_multiplier": 1.0,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            intraday = {"quality_flags": [soft_flag], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
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
                            "regime": "震荡",
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
                "ops_slot_premarket_anomaly_ratio_max": 0.10,
                "ops_slot_intraday_anomaly_ratio_max": 0.10,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_slot_eod_quality_anomaly_ratio_max": 1.0,
                "ops_slot_eod_risk_anomaly_ratio_max": 1.0,
                "ops_slot_soft_quality_flag_prefixes": ["LOW_CONFIDENCE_SOURCE_RATIO_HIGH"],
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
        metrics = slot.get("metrics", {})
        self.assertTrue(bool(slot.get("active", False)))
        self.assertTrue(bool(checks.get("premarket_anomaly_ok", False)))
        self.assertTrue(bool(checks.get("intraday_anomaly_ok", False)))
        self.assertEqual(int(metrics.get("soft_quality_flag_slots", 0)), 9)
        self.assertEqual(int(metrics.get("hard_quality_flag_slots", 0)), 0)

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

    def test_gate_report_uses_live_degradation_overrides(self) -> None:
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
            intraday = {"quality_flags": [], "source_confidence_score": 0.90, "risk_multiplier": 1.0}
            (logs_dir / f"{dstr}_intraday_1030.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
            (logs_dir / f"{dstr}_intraday_1430.json").write_text(json.dumps(intraday, ensure_ascii=False), encoding="utf-8")
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
                            "regime": "震荡",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (daily_dir / f"{dstr}_mode_feedback.json").write_text(
                json.dumps(
                    {
                        "runtime_mode": "swing",
                        "risk_control": {"risk_multiplier": 1.0, "source_confidence_score": 0.90},
                        "mode_health": {"passed": True},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        live_path = td / "artifacts" / "degradation_params_live.yaml"
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "params": {
                        "ops_slot_degradation_soft_multiplier": 1.30,
                        "ops_slot_degradation_hard_multiplier": 1.65,
                        "ops_slot_hysteresis_soft_streak_days": 4,
                        "ops_slot_hysteresis_hard_streak_days": 5,
                        "ops_state_degradation_soft_multiplier": 1.20,
                        "ops_state_degradation_hard_multiplier": 1.55,
                        "ops_state_degradation_floor_soft_ratio": 0.88,
                        "ops_state_degradation_floor_hard_ratio": 0.82,
                        "ops_state_hysteresis_soft_streak_days": 4,
                        "ops_state_hysteresis_hard_streak_days": 5,
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
                "mode_switch_window_days": 3,
                "ops_state_min_samples": 1,
                "ops_slot_window_days": 3,
                "ops_slot_min_samples": 1,
                "ops_slot_missing_ratio_max": 1.0,
                "ops_slot_premarket_anomaly_ratio_max": 1.0,
                "ops_slot_intraday_anomaly_ratio_max": 1.0,
                "ops_slot_eod_anomaly_ratio_max": 1.0,
                "ops_slot_eod_quality_anomaly_ratio_max": 1.0,
                "ops_slot_eod_risk_anomaly_ratio_max": 1.0,
                "ops_slot_use_live_regime_thresholds": False,
                "ops_slot_degradation_enabled": True,
                "ops_slot_hysteresis_enabled": True,
                "ops_slot_degradation_soft_multiplier": 1.15,
                "ops_slot_degradation_hard_multiplier": 1.40,
                "ops_slot_hysteresis_soft_streak_days": 2,
                "ops_slot_hysteresis_hard_streak_days": 3,
                "ops_state_degradation_enabled": True,
                "ops_state_hysteresis_enabled": True,
                "ops_state_degradation_soft_multiplier": 1.10,
                "ops_state_degradation_hard_multiplier": 1.35,
                "ops_state_degradation_floor_soft_ratio": 0.96,
                "ops_state_degradation_floor_hard_ratio": 0.90,
                "ops_state_hysteresis_soft_streak_days": 2,
                "ops_state_hysteresis_hard_streak_days": 3,
                "ops_degradation_calibration_use_live_overrides": True,
                "ops_degradation_calibration_live_params_path": "artifacts/degradation_params_live.yaml",
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
        state = out.get("state_stability", {})
        slot_thresholds = slot.get("thresholds", {}) if isinstance(slot.get("thresholds", {}), dict) else {}
        state_thresholds = state.get("thresholds", {}) if isinstance(state.get("thresholds", {}), dict) else {}
        self.assertTrue(bool(slot_thresholds.get("degradation_live_overrides_applied", False)))
        self.assertTrue(bool(state_thresholds.get("degradation_live_overrides_applied", False)))
        self.assertAlmostEqual(float(slot_thresholds.get("ops_slot_degradation_soft_multiplier", 0.0)), 1.30, places=6)
        self.assertAlmostEqual(float(slot_thresholds.get("ops_slot_degradation_hard_multiplier", 0.0)), 1.65, places=6)
        self.assertEqual(int(slot_thresholds.get("ops_slot_hysteresis_soft_streak_days", 0)), 4)
        self.assertEqual(int(slot_thresholds.get("ops_slot_hysteresis_hard_streak_days", 0)), 5)
        self.assertAlmostEqual(float(state_thresholds.get("ops_state_degradation_soft_multiplier", 0.0)), 1.20, places=6)
        self.assertAlmostEqual(float(state_thresholds.get("ops_state_degradation_hard_multiplier", 0.0)), 1.55, places=6)
        self.assertAlmostEqual(float(state_thresholds.get("ops_state_degradation_floor_soft_ratio", 0.0)), 0.88, places=6)
        self.assertAlmostEqual(float(state_thresholds.get("ops_state_degradation_floor_hard_ratio", 0.0)), 0.82, places=6)

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

    def test_reconcile_row_diff_artifact_rotation_and_checksum_index(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        stale_json = review_dir / "2026-02-10_reconcile_row_diff.json"
        stale_md = review_dir / "2026-02-10_reconcile_row_diff.md"
        stale_json.write_text(json.dumps({"date": "2026-02-10", "sample_rows": 1}), encoding="utf-8")
        stale_md.write_text("# stale\n", encoding="utf-8")

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
                            "breached": True,
                            "missing_on_broker": ["RB2405.SHFE|LONG"],
                            "extra_on_broker": [],
                        }
                    },
                }
            ],
            retention_days=2,
            checksum_index_enabled=True,
        )

        self.assertTrue(bool(artifact.get("written", False)))
        self.assertEqual(int(artifact.get("sample_rows", 0)), 1)
        self.assertEqual(int(artifact.get("breach_rows", 0)), 1)
        self.assertEqual(int(artifact.get("retention_days", 0)), 2)
        self.assertEqual(int(artifact.get("rotated_out_count", -1)), 1)
        self.assertEqual(artifact.get("rotated_out_dates", []), ["2026-02-10"])
        self.assertFalse(bool(artifact.get("rotation_failed", True)))
        self.assertTrue(bool(artifact.get("checksum_index_enabled", False)))
        self.assertTrue(bool(artifact.get("checksum_index_written", False)))
        self.assertFalse(bool(artifact.get("checksum_index_failed", True)))
        self.assertFalse(stale_json.exists())
        self.assertFalse(stale_md.exists())

        index_path = Path(str(artifact.get("checksum_index_path", "")))
        self.assertTrue(index_path.exists())
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(int(index_payload.get("retention_days", 0)), 2)
        self.assertEqual(index_payload.get("rotated_out_dates", []), ["2026-02-10"])
        self.assertEqual(int(index_payload.get("entry_count", 0)), 1)

    def test_reconcile_row_diff_artifact_profile_override(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        review_dir.mkdir(parents=True, exist_ok=True)

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_reconcile_broker_row_diff_artifact_retention_days": 2,
                "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled": True,
                "ops_artifact_governance_profiles": {
                    "reconcile_row_diff": {
                        "json_glob": "*_reconcile_row_diff.json",
                        "md_glob": "*_reconcile_row_diff.md",
                        "checksum_index_filename": "custom_reconcile_row_diff_index.json",
                        "retention_days": 5,
                        "checksum_index_enabled": False,
                    }
                },
            }
        )

        stale_json = review_dir / "2026-02-06_reconcile_row_diff.json"
        stale_md = review_dir / "2026-02-06_reconcile_row_diff.md"
        stale_json.write_text(json.dumps({"date": "2026-02-06", "sample_rows": 1}), encoding="utf-8")
        stale_md.write_text("# stale\n", encoding="utf-8")

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
        artifact = orch._write_reconcile_row_diff_artifact(
            as_of=d,
            series=[
                {
                    "date": d.isoformat(),
                    "broker": {
                        "row_diff": {
                            "active": True,
                            "skipped": False,
                            "breached": True,
                            "missing_on_broker": ["RB2405.SHFE|LONG"],
                            "extra_on_broker": [],
                        }
                    },
                }
            ],
            retention_days=2,
            checksum_index_enabled=True,
        )

        self.assertTrue(bool(artifact.get("written", False)))
        self.assertEqual(int(artifact.get("retention_days", 0)), 5)
        self.assertFalse(bool(artifact.get("checksum_index_enabled", True)))
        self.assertFalse(bool(artifact.get("checksum_index_written", True)))
        self.assertEqual(str(artifact.get("checksum_index_path", "")), "")
        self.assertEqual(int(artifact.get("checksum_index_entries", 0)), 0)
        self.assertFalse((review_dir / "custom_reconcile_row_diff_index.json").exists())
        self.assertFalse(stale_json.exists())
        self.assertFalse(stale_md.exists())

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

    def test_build_defect_plan_includes_reconcile_executed_restore_verify(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        review = ReviewDelta(
            as_of=d,
            parameter_changes={},
            factor_weights={},
            defects=[],
            pass_gate=True,
        )
        gate = {"checks": {"reconcile_drift_ok": False}, "metrics": {}}
        reconcile_drift = {
            "active": True,
            "alerts": ["reconcile_executed_dedup_restore_unverified"],
            "checks": {"executed_dedup_restore_verify_ok": False},
            "metrics": {},
            "artifacts": {
                "executed_dedup_restore_verify": {
                    "status": "error",
                    "reason": "restore_delta_mismatch",
                    "path": "/tmp/mock_restore_verify.json",
                }
            },
        }

        plan_artifact = orch._build_defect_plan(
            as_of=d,
            round_no=1,
            review=review,
            tests={"returncode": 0, "stderr": "", "stdout": ""},
            gate=gate,
            reconcile_drift=reconcile_drift,
        )
        plan = json.loads(Path(str(plan_artifact.get("json", ""))).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("RECONCILE_EXECUTED_DEDUP_RESTORE_VERIFY", codes)

    def test_gate_report_frontend_snapshot_trend_hard_fail_blocks_release(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows = [
            {
                "ts": "2026-02-12T20:30:00+08:00",
                "daemon": {"date": "2026-02-12"},
                "frontend_snapshot": {
                    "due": True,
                    "status": "error",
                    "reason": "timeout",
                    "result": {"timed_out": True},
                    "governance": {
                        "rotation_failed": False,
                        "checksum_index_failed": False,
                        "checksum_index_enabled": True,
                        "checksum_index_path": "output/logs/frontend_snapshot_checksum_index.json",
                    },
                    "artifact": {
                        "json_path": "output/logs/frontend_snapshot_20260212_203000.json",
                    },
                },
            },
            {
                "ts": "2026-02-13T20:30:00+08:00",
                "daemon": {"date": "2026-02-13"},
                "frontend_snapshot": {
                    "due": True,
                    "status": "error",
                    "reason": "frontend_snapshot_failed",
                    "result": {"timed_out": False},
                    "governance": {
                        "rotation_failed": False,
                        "checksum_index_failed": False,
                        "checksum_index_enabled": True,
                        "checksum_index_path": "output/logs/frontend_snapshot_checksum_index.json",
                    },
                    "artifact": {
                        "json_path": "output/logs/frontend_snapshot_20260213_203000.json",
                    },
                },
            },
        ]
        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": True,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 2,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 0.40,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 0.40,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 0.50,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 1,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 1,
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        self.assertFalse(bool(out["checks"].get("guard_loop_frontend_snapshot_trend_ok", True)))
        self.assertFalse(bool(out.get("passed", True)))
        trend = out.get("guard_loop_frontend_snapshot_trend", {})
        self.assertTrue(bool(trend.get("active", False)))
        self.assertTrue(bool(trend.get("monitor_failed", False)))
        self.assertIn(
            "guard_loop_frontend_snapshot_trend_failure_streak_high",
            set(trend.get("alerts", [])),
        )

    def test_gate_report_frontend_snapshot_trend_monitor_mode_does_not_block(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows = [
            {
                "ts": "2026-02-12T20:30:00+08:00",
                "daemon": {"date": "2026-02-12"},
                "frontend_snapshot": {
                    "due": True,
                    "status": "error",
                    "reason": "timeout",
                    "result": {"timed_out": True},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                },
            },
            {
                "ts": "2026-02-13T20:30:00+08:00",
                "daemon": {"date": "2026-02-13"},
                "frontend_snapshot": {
                    "due": True,
                    "status": "error",
                    "reason": "frontend_snapshot_failed",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                },
            },
        ]
        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": False,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 2,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 0.20,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 0.20,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 0.20,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 1,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 1,
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        out = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        self.assertTrue(bool(out["checks"].get("guard_loop_frontend_snapshot_trend_ok", False)))
        self.assertTrue(bool(out.get("passed", False)))
        trend = out.get("guard_loop_frontend_snapshot_trend", {})
        self.assertTrue(bool(trend.get("monitor_failed", False)))
        self.assertIn("guard_loop_frontend_snapshot_trend_timeout_ratio_high", set(trend.get("alerts", [])))

    def test_gate_ops_report_frontend_snapshot_replay_convergence_scorecard(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows = [
            {
                "ts": "2026-02-12T20:30:00+08:00",
                "daemon": {"date": "2026-02-12"},
                "recovery": {
                    "mode": "light",
                    "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"],
                },
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-13T20:30:00+08:00",
                "daemon": {"date": "2026-02-13"},
                "recovery": {"mode": "none", "reason_codes": []},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
        ]
        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": False,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 2,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 10,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 10,
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        gate = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        gate_cards = gate.get("scorecards", {}) if isinstance(gate.get("scorecards", {}), dict) else {}
        gate_guard_loop_cards = (
            gate_cards.get("guard_loop", {})
            if isinstance(gate_cards.get("guard_loop", {}), dict)
            else {}
        )
        gate_frontend_card = (
            gate_guard_loop_cards.get("frontend_snapshot_replay_convergence", {})
            if isinstance(gate_guard_loop_cards.get("frontend_snapshot_replay_convergence", {}), dict)
            else {}
        )
        self.assertEqual(str(gate_frontend_card.get("status", "")), "red")
        self.assertEqual(int(gate_frontend_card.get("expected_runs", 0)), 2)
        self.assertEqual(int(gate_frontend_card.get("executed_runs", 0)), 1)
        self.assertEqual(int(gate_frontend_card.get("missed_runs", 0)), 1)
        self.assertAlmostEqual(float(gate_frontend_card.get("convergence_rate", 0.0)), 0.5)
        rollback = (
            gate.get("rollback_recommendation", {})
            if isinstance(gate.get("rollback_recommendation", {}), dict)
            else {}
        )
        self.assertIn(
            "guard_loop_frontend_snapshot_replay_convergence",
            set(rollback.get("reason_codes", [])),
        )

        ops = orch.ops_report(as_of=d, window_days=3)
        ops_cards = ops.get("scorecards", {}) if isinstance(ops.get("scorecards", {}), dict) else {}
        ops_guard_loop_cards = (
            ops_cards.get("guard_loop", {})
            if isinstance(ops_cards.get("guard_loop", {}), dict)
            else {}
        )
        ops_frontend_card = (
            ops_guard_loop_cards.get("frontend_snapshot_replay_convergence", {})
            if isinstance(ops_guard_loop_cards.get("frontend_snapshot_replay_convergence", {}), dict)
            else {}
        )
        self.assertEqual(str(ops_frontend_card.get("status", "")), "red")
        self.assertEqual(int(ops_frontend_card.get("missed_runs", 0)), 1)
        ops_md = (review_dir / f"{d.isoformat()}_ops_report.md").read_text(encoding="utf-8")
        self.assertIn(
            "GuardLoop Scorecard(cadence/lift/preset/semantic/frontend_replay):",
            ops_md,
        )
        self.assertIn(
            "replay_convergence(expected/executed/missed/convergence):",
            ops_md,
        )

    def test_gate_ops_report_frontend_snapshot_antiflap_burnin_artifact(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows = [
            {
                "ts": "2026-02-11T20:30:00+08:00",
                "daemon": {"date": "2026-02-11"},
                "recovery": {
                    "mode": "light",
                    "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"],
                },
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-12T20:30:00+08:00",
                "daemon": {"date": "2026-02-12"},
                "recovery": {
                    "mode": "light",
                    "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"],
                },
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-13T20:30:00+08:00",
                "daemon": {"date": "2026-02-13"},
                "recovery": {
                    "mode": "light",
                    "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"],
                },
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
        ]
        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": False,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 2,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 10,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 10,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio": 0.5,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs": 0,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_retention_days": 30,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_checksum_index_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin": True,
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        gate = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        trend = (
            gate.get("guard_loop_frontend_snapshot_trend", {})
            if isinstance(gate.get("guard_loop_frontend_snapshot_trend", {}), dict)
            else {}
        )
        burnin = (
            trend.get("antiflap_burnin", {})
            if isinstance(trend.get("antiflap_burnin", {}), dict)
            else {}
        )
        self.assertEqual(str(burnin.get("status", "")), "green")
        self.assertTrue(bool(burnin.get("ok", False)))
        self.assertEqual(int(burnin.get("samples", 0)), 3)
        artifact = (
            burnin.get("artifacts", {}).get("burnin", {})
            if isinstance(burnin.get("artifacts", {}), dict)
            else {}
        )
        self.assertTrue(bool(artifact.get("written", False)))
        self.assertTrue(Path(str(artifact.get("json", ""))).exists())
        self.assertTrue(Path(str(artifact.get("md", ""))).exists())

        gate_cards = gate.get("scorecards", {}) if isinstance(gate.get("scorecards", {}), dict) else {}
        gate_guard_loop_cards = (
            gate_cards.get("guard_loop", {})
            if isinstance(gate_cards.get("guard_loop", {}), dict)
            else {}
        )
        gate_burnin_card = (
            gate_guard_loop_cards.get("frontend_snapshot_antiflap_burnin", {})
            if isinstance(gate_guard_loop_cards.get("frontend_snapshot_antiflap_burnin", {}), dict)
            else {}
        )
        self.assertEqual(str(gate_burnin_card.get("status", "")), "green")
        self.assertTrue(bool(gate_burnin_card.get("promotion_eligible", False)))
        self.assertEqual(str(gate_burnin_card.get("promotion_recommendation", "")), "enable_hard_fail")

        ops = orch.ops_report(as_of=d, window_days=3)
        ops_cards = ops.get("scorecards", {}) if isinstance(ops.get("scorecards", {}), dict) else {}
        ops_guard_loop_cards = (
            ops_cards.get("guard_loop", {})
            if isinstance(ops_cards.get("guard_loop", {}), dict)
            else {}
        )
        ops_burnin_card = (
            ops_guard_loop_cards.get("frontend_snapshot_antiflap_burnin", {})
            if isinstance(ops_guard_loop_cards.get("frontend_snapshot_antiflap_burnin", {}), dict)
            else {}
        )
        self.assertEqual(str(ops_burnin_card.get("status", "")), "green")
        ops_md = (review_dir / f"{d.isoformat()}_ops_report.md").read_text(encoding="utf-8")
        self.assertIn("GuardLoop Scorecard(frontend_antiflap_burnin):", ops_md)
        self.assertIn("antiflap_burnin(status/samples/window/min):", ops_md)

    def test_gate_report_frontend_hard_fail_controlled_apply_requires_approval_manifest(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows = [
            {
                "ts": "2026-02-11T20:30:00+08:00",
                "daemon": {"date": "2026-02-11"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-12T20:30:00+08:00",
                "daemon": {"date": "2026-02-12"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-13T20:30:00+08:00",
                "daemon": {"date": "2026-02-13"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
        ]
        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": False,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 2,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 10,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 10,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio": 0.5,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs": 0,
                "ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_manual_approval_required": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_dry_run": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_rollback_guard_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path": "output/artifacts/frontend_snapshot_trend_hard_fail_approval.json",
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        gate = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        trend = gate.get("guard_loop_frontend_snapshot_trend", {})
        burnin = trend.get("antiflap_burnin", {}) if isinstance(trend, dict) else {}
        promotion = burnin.get("promotion", {}) if isinstance(burnin, dict) else {}
        controlled_apply = (
            promotion.get("controlled_apply", {})
            if isinstance(promotion.get("controlled_apply", {}), dict)
            else {}
        )
        apply_gate = (
            controlled_apply.get("apply_gate", {})
            if isinstance(controlled_apply.get("apply_gate", {}), dict)
            else {}
        )
        self.assertEqual(str(apply_gate.get("reason", "")), "manual_approval_missing")
        self.assertFalse(bool(apply_gate.get("apply_recommended", True)))
        decision_envelope = (
            controlled_apply.get("decision_envelope", {})
            if isinstance(controlled_apply.get("decision_envelope", {}), dict)
            else {}
        )
        decision_approval = (
            decision_envelope.get("approval_manifest", {})
            if isinstance(decision_envelope.get("approval_manifest", {}), dict)
            else {}
        )
        self.assertEqual(str(decision_envelope.get("decision", "")), "non_apply")
        self.assertEqual(str(decision_envelope.get("route", "")), "manual_approval_missing")
        self.assertFalse(bool(decision_envelope.get("apply_recommended", True)))
        self.assertFalse(bool(decision_approval.get("found", True)))
        self.assertFalse(bool(decision_approval.get("approved", True)))
        self.assertFalse(bool(decision_approval.get("matches_proposal", True)))
        artifacts = (
            controlled_apply.get("artifacts", {})
            if isinstance(controlled_apply.get("artifacts", {}), dict)
            else {}
        )
        controlled_apply_artifact = (
            artifacts.get("controlled_apply", {})
            if isinstance(artifacts.get("controlled_apply", {}), dict)
            else {}
        )
        self.assertTrue(bool(controlled_apply_artifact.get("written", False)))
        self.assertTrue(Path(str(controlled_apply_artifact.get("json", ""))).exists())

        guard_loop_cards = (
            (gate.get("scorecards", {}) or {}).get("guard_loop", {})
            if isinstance((gate.get("scorecards", {}) or {}).get("guard_loop", {}), dict)
            else {}
        )
        apply_card = (
            guard_loop_cards.get("frontend_snapshot_hard_fail_controlled_apply", {})
            if isinstance(guard_loop_cards.get("frontend_snapshot_hard_fail_controlled_apply", {}), dict)
            else {}
        )
        self.assertEqual(str(apply_card.get("status", "")), "yellow")
        self.assertFalse(bool(apply_card.get("approval_manifest_found", True)))
        self.assertFalse(bool(apply_card.get("approval_manifest_approved", True)))
        self.assertFalse(bool(apply_card.get("approval_manifest_matches_proposal", True)))

        ops = orch.ops_report(as_of=d, window_days=3)
        ops_cards = (
            (ops.get("scorecards", {}) or {}).get("guard_loop", {})
            if isinstance((ops.get("scorecards", {}) or {}).get("guard_loop", {}), dict)
            else {}
        )
        ops_apply_card = (
            ops_cards.get("frontend_snapshot_hard_fail_controlled_apply", {})
            if isinstance(ops_cards.get("frontend_snapshot_hard_fail_controlled_apply", {}), dict)
            else {}
        )
        self.assertEqual(str(ops_apply_card.get("route", "")), "manual_approval_missing")
        self.assertFalse(bool(ops_apply_card.get("approval_manifest_found", True)))
        self.assertGreaterEqual(int(ops_apply_card.get("runbook_count", 0)), 1)

    def test_gate_ops_report_frontend_antiflap_dual_window_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 14)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows: list[dict[str, Any]] = []
        for day in range(1, 15):
            date_txt = f"2026-02-{day:02d}"
            suppressed = bool(day >= 12)
            due = not suppressed
            reason_codes = ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"] if due else []
            history_rows.append(
                {
                    "ts": f"{date_txt}T20:30:00+08:00",
                    "daemon": {"date": date_txt},
                    "recovery": {"mode": "light" if due else "none", "reason_codes": reason_codes},
                    "frontend_snapshot": {
                        "due": True,
                        "status": "ok",
                        "reason": "success",
                        "result": {"timed_out": False},
                        "governance": {"rotation_failed": False, "checksum_index_failed": False},
                        "trend": {
                            "due_light": bool(due),
                            "due_heavy": False,
                            "antiflap": {"suppressed": bool(suppressed)},
                        },
                    },
                }
            )

        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": False,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 14,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 4,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 10,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 10,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs": False,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs": 10,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_enabled": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_long_days": 14,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_min_long_samples": 10,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_suppression_ratio_delta": 0.30,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_replay_missed_ratio_delta": 0.90,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_reason_missing_ratio_delta": 0.90,
                "ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_manual_approval_required": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_dry_run": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_rollback_guard_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path": "output/artifacts/frontend_snapshot_trend_hard_fail_approval.json",
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        gate = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        trend = (
            gate.get("guard_loop_frontend_snapshot_trend", {})
            if isinstance(gate.get("guard_loop_frontend_snapshot_trend", {}), dict)
            else {}
        )
        burnin = (
            trend.get("antiflap_burnin", {})
            if isinstance(trend.get("antiflap_burnin", {}), dict)
            else {}
        )
        dual_window = (
            burnin.get("dual_window", {})
            if isinstance(burnin.get("dual_window", {}), dict)
            else {}
        )
        dual_window_checks = (
            dual_window.get("checks", {})
            if isinstance(dual_window.get("checks", {}), dict)
            else {}
        )
        self.assertEqual(str(dual_window.get("status", "")), "red")
        self.assertFalse(bool(dual_window_checks.get("suppression_ratio_delta_ok", True)))
        self.assertIn(
            "guard_loop_frontend_snapshot_antiflap_burnin_dual_window_suppression_delta_high",
            set(burnin.get("alerts", [])),
        )
        promotion = (
            burnin.get("promotion", {})
            if isinstance(burnin.get("promotion", {}), dict)
            else {}
        )
        controlled_apply = (
            promotion.get("controlled_apply", {})
            if isinstance(promotion.get("controlled_apply", {}), dict)
            else {}
        )
        apply_gate = (
            controlled_apply.get("apply_gate", {})
            if isinstance(controlled_apply.get("apply_gate", {}), dict)
            else {}
        )
        decision_envelope = (
            controlled_apply.get("decision_envelope", {})
            if isinstance(controlled_apply.get("decision_envelope", {}), dict)
            else {}
        )
        self.assertEqual(str(apply_gate.get("reason_pre_rollback", "")), "dual_window_drift_red_non_apply")
        self.assertEqual(str(apply_gate.get("reason", "")), "dual_window_drift_red_non_apply")
        self.assertFalse(bool(apply_gate.get("apply_recommended", True)))
        self.assertEqual(str(decision_envelope.get("decision", "")), "non_apply")
        self.assertEqual(str(decision_envelope.get("route", "")), "dual_window_drift_red_non_apply")
        self.assertFalse(bool(decision_envelope.get("apply_recommended", True)))
        self.assertIn(
            "guard_loop_frontend_snapshot_trend_hard_fail_controlled_apply_dual_window_red_non_apply",
            set(controlled_apply.get("alerts", [])),
        )

        guard_loop_cards = (
            (gate.get("scorecards", {}) or {}).get("guard_loop", {})
            if isinstance((gate.get("scorecards", {}) or {}).get("guard_loop", {}), dict)
            else {}
        )
        burnin_card = (
            guard_loop_cards.get("frontend_snapshot_antiflap_burnin", {})
            if isinstance(guard_loop_cards.get("frontend_snapshot_antiflap_burnin", {}), dict)
            else {}
        )
        self.assertEqual(str(burnin_card.get("dual_window_status", "")), "red")
        apply_card = (
            guard_loop_cards.get("frontend_snapshot_hard_fail_controlled_apply", {})
            if isinstance(guard_loop_cards.get("frontend_snapshot_hard_fail_controlled_apply", {}), dict)
            else {}
        )
        self.assertEqual(str(apply_card.get("decision", "")), "non_apply")
        self.assertEqual(str(apply_card.get("route", "")), "dual_window_drift_red_non_apply")
        self.assertGreaterEqual(int(apply_card.get("runbook_count", 0)), 1)

        ops = orch.ops_report(as_of=d, window_days=7)
        ops_cards = (
            (ops.get("scorecards", {}) or {}).get("guard_loop", {})
            if isinstance((ops.get("scorecards", {}) or {}).get("guard_loop", {}), dict)
            else {}
        )
        ops_apply_card = (
            ops_cards.get("frontend_snapshot_hard_fail_controlled_apply", {})
            if isinstance(ops_cards.get("frontend_snapshot_hard_fail_controlled_apply", {}), dict)
            else {}
        )
        self.assertEqual(str(ops_apply_card.get("route", "")), "dual_window_drift_red_non_apply")
        ops_md = (review_dir / f"{d.isoformat()}_ops_report.md").read_text(encoding="utf-8")
        self.assertIn("GuardLoop Scorecard(frontend_antiflap_dual_window):", ops_md)
        self.assertIn("antiflap_burnin_dual_window(delta suppression/replay/traceability):", ops_md)
        self.assertIn("frontend_hard_fail_apply_envelope(runbook):", ops_md)

    def test_gate_report_frontend_hard_fail_apply_route_audit_weekly_trendline(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 21)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows = [
            {
                "ts": "2026-02-18T20:30:00+08:00",
                "daemon": {"date": "2026-02-18"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-19T20:30:00+08:00",
                "daemon": {"date": "2026-02-19"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-20T20:30:00+08:00",
                "daemon": {"date": "2026-02-20"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-21T20:30:00+08:00",
                "daemon": {"date": "2026-02-21"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
        ]
        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        def _gate_payload(*, decision: str, route: str) -> dict[str, object]:
            apply = bool(decision == "apply")
            return {
                "guard_loop_frontend_snapshot_trend": {
                    "antiflap_burnin": {
                        "promotion": {
                            "controlled_apply": {
                                "enabled": True,
                                "decision_envelope": {
                                    "decision": str(decision),
                                    "route": str(route),
                                    "apply_recommended": bool(apply),
                                },
                                "apply_gate": {
                                    "apply_recommended": bool(apply),
                                    "reason": str(route),
                                },
                            }
                        }
                    }
                }
            }

        prior_routes = [
            ("2026-02-08", "apply", "manual_apply_required"),
            ("2026-02-09", "apply", "manual_apply_required"),
            ("2026-02-10", "apply", "manual_apply_required"),
            ("2026-02-11", "apply", "manual_apply_required"),
            ("2026-02-12", "apply", "manual_apply_required"),
            ("2026-02-13", "non_apply", "manual_approval_missing"),
            ("2026-02-14", "non_apply", "manual_approval_missing"),
        ]
        recent_routes = [
            ("2026-02-15", "non_apply", "manual_approval_missing"),
            ("2026-02-16", "non_apply", "manual_approval_missing"),
            ("2026-02-17", "non_apply", "manual_approval_missing"),
            ("2026-02-18", "non_apply", "manual_approval_missing"),
            ("2026-02-19", "non_apply", "manual_approval_missing"),
            ("2026-02-20", "non_apply", "rollback_guard_blocked"),
            ("2026-02-21", "non_apply", "rollback_guard_blocked"),
        ]
        for day_txt, decision, route in prior_routes + recent_routes:
            (review_dir / f"{day_txt}_gate_report.json").write_text(
                json.dumps(_gate_payload(decision=decision, route=route), ensure_ascii=False),
                encoding="utf-8",
            )

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": True,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 2,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 10,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 10,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs": False,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs": 10,
                "ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_manual_approval_required": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_dry_run": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_rollback_guard_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path": "output/artifacts/frontend_snapshot_trend_hard_fail_approval.json",
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_gate_hard_fail": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_window_days": 14,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_samples": 7,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_non_apply_ratio": 0.50,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_rollback_guard_ratio": 0.01,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_apply_ratio": 0.60,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_recent_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_prior_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_min_samples": 3,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_non_apply_ratio_rise": 0.20,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_rollback_guard_ratio_rise": 0.05,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_apply_ratio_drop": 0.30,
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
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=_load_json,
        )
        gate = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        trend = (
            gate.get("guard_loop_frontend_snapshot_trend", {})
            if isinstance(gate.get("guard_loop_frontend_snapshot_trend", {}), dict)
            else {}
        )
        route_audit = (
            trend.get("hard_fail_apply_route_audit", {})
            if isinstance(trend.get("hard_fail_apply_route_audit", {}), dict)
            else {}
        )
        route_checks = (
            route_audit.get("checks", {})
            if isinstance(route_audit.get("checks", {}), dict)
            else {}
        )
        self.assertTrue(bool(route_audit.get("active", False)))
        self.assertEqual(str(route_audit.get("status", "")), "red")
        self.assertFalse(bool(route_audit.get("ok", True)))
        self.assertFalse(bool(route_checks.get("non_apply_ratio_ok", True)))
        self.assertFalse(bool(route_checks.get("rollback_guard_ratio_ok", True)))
        self.assertFalse(bool(route_checks.get("apply_ratio_ok", True)))
        self.assertFalse(bool(route_checks.get("trendline_non_apply_ratio_rise_ok", True)))
        self.assertFalse(bool(route_checks.get("trendline_rollback_guard_blocked_ratio_rise_ok", True)))
        self.assertFalse(bool(route_checks.get("trendline_apply_ratio_drop_ok", True)))
        self.assertFalse(bool((trend.get("checks", {}) or {}).get("controlled_apply_route_audit_ok", True)))
        self.assertFalse(bool(gate.get("checks", {}).get("guard_loop_frontend_snapshot_trend_ok", True)))
        trend_alerts = set(trend.get("alerts", []))
        self.assertIn(
            "guard_loop_frontend_snapshot_trend_hard_fail_controlled_apply_route_non_apply_ratio_high",
            trend_alerts,
        )
        self.assertIn(
            "guard_loop_frontend_snapshot_trend_hard_fail_controlled_apply_route_trendline_non_apply_rise",
            trend_alerts,
        )

        guard_loop_cards = (
            (gate.get("scorecards", {}) or {}).get("guard_loop", {})
            if isinstance((gate.get("scorecards", {}) or {}).get("guard_loop", {}), dict)
            else {}
        )
        apply_card = (
            guard_loop_cards.get("frontend_snapshot_hard_fail_controlled_apply", {})
            if isinstance(guard_loop_cards.get("frontend_snapshot_hard_fail_controlled_apply", {}), dict)
            else {}
        )
        self.assertEqual(str(apply_card.get("route_audit_status", "")), "red")
        self.assertFalse(bool(apply_card.get("route_audit_ok", True)))

        ops = orch.ops_report(as_of=d, window_days=7)
        ops_cards = (
            (ops.get("scorecards", {}) or {}).get("guard_loop", {})
            if isinstance((ops.get("scorecards", {}) or {}).get("guard_loop", {}), dict)
            else {}
        )
        ops_apply_card = (
            ops_cards.get("frontend_snapshot_hard_fail_controlled_apply", {})
            if isinstance(ops_cards.get("frontend_snapshot_hard_fail_controlled_apply", {}), dict)
            else {}
        )
        self.assertEqual(str(ops_apply_card.get("route_audit_status", "")), "red")
        ops_md = (review_dir / f"{d.isoformat()}_ops_report.md").read_text(encoding="utf-8")
        self.assertIn("hard_fail_controlled_apply_route_audit(status/samples/min/window):", ops_md)
        self.assertIn(
            "hard_fail_controlled_apply_route_audit(trendline rise non_apply/rollback_guard, drop apply):",
            ops_md,
        )

    def test_gate_report_frontend_hard_fail_controlled_apply_blocked_by_rollback_guard(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows = [
            {
                "ts": "2026-02-11T20:30:00+08:00",
                "daemon": {"date": "2026-02-11"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-12T20:30:00+08:00",
                "daemon": {"date": "2026-02-12"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
            {
                "ts": "2026-02-13T20:30:00+08:00",
                "daemon": {"date": "2026-02-13"},
                "recovery": {"mode": "light", "reason_codes": ["FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT"]},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {"due_light": True, "due_heavy": False},
                },
            },
        ]
        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": False,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 2,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 10,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 10,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio": 0.5,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs": 0,
                "ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_manual_approval_required": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_dry_run": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_rollback_guard_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path": "output/artifacts/frontend_snapshot_trend_hard_fail_approval.json",
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.30, "violations": 1},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        gate_first = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        trend_first = (
            gate_first.get("guard_loop_frontend_snapshot_trend", {})
            if isinstance(gate_first.get("guard_loop_frontend_snapshot_trend", {}), dict)
            else {}
        )
        burnin_first = (
            trend_first.get("antiflap_burnin", {})
            if isinstance(trend_first.get("antiflap_burnin", {}), dict)
            else {}
        )
        promotion_first = (
            burnin_first.get("promotion", {})
            if isinstance(burnin_first.get("promotion", {}), dict)
            else {}
        )
        controlled_apply_first = (
            promotion_first.get("controlled_apply", {})
            if isinstance(promotion_first.get("controlled_apply", {}), dict)
            else {}
        )
        proposal_first = (
            controlled_apply_first.get("proposal", {})
            if isinstance(controlled_apply_first.get("proposal", {}), dict)
            else {}
        )
        proposal_id = str(proposal_first.get("proposal_id", ""))
        manifest_path = Path(str(controlled_apply_first.get("approval_manifest_path", "")))
        self.assertTrue(bool(proposal_id))
        self.assertTrue(bool(manifest_path))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "type": "frontend_snapshot_trend_hard_fail_approval",
                    "approved": True,
                    "proposal_id": proposal_id,
                    "approved_at": "2026-02-13T21:00:00+08:00",
                    "as_of": d.isoformat(),
                    "timezone": "Asia/Shanghai",
                    "source": "unit_test",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        gate_second = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        trend_second = (
            gate_second.get("guard_loop_frontend_snapshot_trend", {})
            if isinstance(gate_second.get("guard_loop_frontend_snapshot_trend", {}), dict)
            else {}
        )
        burnin_second = (
            trend_second.get("antiflap_burnin", {})
            if isinstance(trend_second.get("antiflap_burnin", {}), dict)
            else {}
        )
        promotion_second = (
            burnin_second.get("promotion", {})
            if isinstance(burnin_second.get("promotion", {}), dict)
            else {}
        )
        controlled_apply_second = (
            promotion_second.get("controlled_apply", {})
            if isinstance(promotion_second.get("controlled_apply", {}), dict)
            else {}
        )
        apply_gate_second = (
            controlled_apply_second.get("apply_gate", {})
            if isinstance(controlled_apply_second.get("apply_gate", {}), dict)
            else {}
        )
        self.assertEqual(str(apply_gate_second.get("reason", "")), "rollback_guard_blocked")
        self.assertFalse(bool(apply_gate_second.get("rollback_guard_ok", True)))

        review = ReviewDelta(
            as_of=d,
            parameter_changes={},
            factor_weights={},
            defects=[],
            pass_gate=True,
        )
        plan_artifact = orch._build_defect_plan(
            as_of=d,
            round_no=1,
            review=review,
            tests={"returncode": 0, "stderr": "", "stdout": ""},
            gate=gate_second,
        )
        plan = json.loads(Path(str(plan_artifact.get("json", ""))).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn(
            "GUARD_LOOP_FRONTEND_SNAPSHOT_TREND_HARD_FAIL_PROMOTION_ROLLBACK_GUARD",
            codes,
        )

    def test_build_defect_plan_includes_frontend_snapshot_timeout_trend(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        review = ReviewDelta(
            as_of=d,
            parameter_changes={},
            factor_weights={},
            defects=[],
            pass_gate=True,
        )
        gate = {
            "checks": {"guard_loop_frontend_snapshot_trend_ok": False},
            "metrics": {},
            "guard_loop_frontend_snapshot_trend": {
                "alerts": ["guard_loop_frontend_snapshot_trend_timeout_ratio_high"],
                "metrics": {
                    "timeout_ratio": 0.5,
                    "timeout_streak_current": 2,
                },
                "thresholds": {
                    "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 0.2,
                },
                "artifacts": {
                    "snapshot": {
                        "json": "output/logs/frontend_snapshot_20260213_203000.json",
                        "checksum_index_path": "output/logs/frontend_snapshot_checksum_index.json",
                    }
                },
                "samples": 4,
                "min_samples": 2,
            },
        }

        plan_artifact = orch._build_defect_plan(
            as_of=d,
            round_no=1,
            review=review,
            tests={"returncode": 0, "stderr": "", "stdout": ""},
            gate=gate,
        )
        plan = json.loads(Path(str(plan_artifact.get("json", ""))).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("GUARD_LOOP_FRONTEND_SNAPSHOT_TREND_TIMEOUT", codes)

    def test_build_defect_plan_includes_frontend_snapshot_replay_convergence(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        review = ReviewDelta(
            as_of=d,
            parameter_changes={},
            factor_weights={},
            defects=[],
            pass_gate=True,
        )
        gate = {
            "checks": {"guard_loop_frontend_snapshot_trend_ok": True},
            "metrics": {},
            "guard_loop_frontend_snapshot_trend": {
                "alerts": [],
                "metrics": {
                    "replay_expected_runs": 3,
                    "replay_executed_runs": 1,
                    "replay_missed_runs": 2,
                    "replay_convergence_rate": 1.0 / 3.0,
                    "replay_due_but_reason_missing_runs": 1,
                },
            },
        }

        plan_artifact = orch._build_defect_plan(
            as_of=d,
            round_no=1,
            review=review,
            tests={"returncode": 0, "stderr": "", "stdout": ""},
            gate=gate,
        )
        plan = json.loads(Path(str(plan_artifact.get("json", ""))).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("GUARD_LOOP_FRONTEND_SNAPSHOT_REPLAY_CONVERGENCE", codes)
        self.assertIn("GUARD_LOOP_FRONTEND_SNAPSHOT_REPLAY_TRACEABILITY", codes)

    def test_gate_report_rollback_includes_frontend_antiflap_burnin_reason(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)
        review_dir = td / "review"
        logs_dir = td / "logs"
        review_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"{d.isoformat()}_param_delta.yaml").write_text("pass_gate: true\n", encoding="utf-8")

        history_rows = [
            {
                "ts": "2026-02-11T20:30:00+08:00",
                "daemon": {"date": "2026-02-11"},
                "recovery": {"mode": "none", "reason_codes": []},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {
                        "due_light": False,
                        "due_heavy": False,
                        "antiflap": {"suppressed": True},
                    },
                },
            },
            {
                "ts": "2026-02-12T20:30:00+08:00",
                "daemon": {"date": "2026-02-12"},
                "recovery": {"mode": "none", "reason_codes": []},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {
                        "due_light": False,
                        "due_heavy": False,
                        "antiflap": {"suppressed": True},
                    },
                },
            },
            {
                "ts": "2026-02-13T20:30:00+08:00",
                "daemon": {"date": "2026-02-13"},
                "recovery": {"mode": "none", "reason_codes": []},
                "frontend_snapshot": {
                    "due": True,
                    "status": "ok",
                    "reason": "success",
                    "result": {"timed_out": False},
                    "governance": {"rotation_failed": False, "checksum_index_failed": False},
                    "trend": {
                        "due_light": False,
                        "due_heavy": False,
                        "antiflap": {"suppressed": True},
                    },
                },
            },
        ]
        history_path = logs_dir / "guard_loop_history.jsonl"
        with history_path.open("w", encoding="utf-8") as fp:
            for row in history_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        settings = self._make_settings()
        settings.raw.setdefault("validation", {})
        settings.raw["validation"].update(
            {
                "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": False,
                "ops_guard_loop_frontend_snapshot_trend_require_active": True,
                "ops_guard_loop_frontend_snapshot_trend_window_days": 7,
                "ops_guard_loop_frontend_snapshot_trend_min_samples": 2,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 1.0,
                "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 10,
                "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 10,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples": 3,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio": 0.2,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs": True,
                "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs": 0,
            }
        )
        orch = ReleaseOrchestrator(
            settings=settings,
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        gate = orch.gate_report(
            as_of=d,
            run_tests=False,
            run_review_if_missing=False,
            run_stable_replay=False,
        )
        rollback = (
            gate.get("rollback_recommendation", {})
            if isinstance(gate.get("rollback_recommendation", {}), dict)
            else {}
        )
        self.assertIn(
            "guard_loop_frontend_snapshot_antiflap_burnin",
            set(rollback.get("reason_codes", [])),
        )

    def test_build_defect_plan_includes_frontend_antiflap_burnin_and_promotion_ready(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 13)

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        review = ReviewDelta(
            as_of=d,
            parameter_changes={},
            factor_weights={},
            defects=[],
            pass_gate=True,
        )
        gate = {
            "checks": {"guard_loop_frontend_snapshot_trend_ok": True},
            "metrics": {},
            "guard_loop_frontend_snapshot_trend": {
                "alerts": [],
                "metrics": {
                    "replay_expected_runs": 1,
                    "replay_executed_runs": 1,
                    "replay_missed_runs": 0,
                    "replay_convergence_rate": 1.0,
                    "replay_due_but_reason_missing_runs": 0,
                },
                "antiflap_burnin": {
                    "active": True,
                    "ok": False,
                    "status": "red",
                    "alerts": ["guard_loop_frontend_snapshot_antiflap_burnin_oversuppressed"],
                    "metrics": {
                        "suppressed_ratio": 0.8,
                        "replay_missed_runs": 0,
                        "reason_missing_runs": 0,
                    },
                    "promotion": {
                        "enabled": True,
                        "current_hard_fail": False,
                        "eligible": False,
                        "recommendation": "keep_monitor",
                    },
                },
            },
        }
        plan_artifact = orch._build_defect_plan(
            as_of=d,
            round_no=1,
            review=review,
            tests={"returncode": 0, "stderr": "", "stdout": ""},
            gate=gate,
        )
        plan = json.loads(Path(str(plan_artifact.get("json", ""))).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn("GUARD_LOOP_FRONTEND_SNAPSHOT_ANTIFLAP_BURNIN_OVERSUPPRESSION", codes)

        gate_ready = {
            "checks": {"guard_loop_frontend_snapshot_trend_ok": True},
            "metrics": {},
            "guard_loop_frontend_snapshot_trend": {
                "alerts": [],
                "metrics": {
                    "replay_expected_runs": 1,
                    "replay_executed_runs": 1,
                    "replay_missed_runs": 0,
                    "replay_convergence_rate": 1.0,
                    "replay_due_but_reason_missing_runs": 0,
                },
                "antiflap_burnin": {
                    "active": True,
                    "ok": True,
                    "status": "green",
                    "alerts": [],
                    "metrics": {
                        "suppressed_ratio": 0.1,
                        "replay_missed_runs": 0,
                        "reason_missing_runs": 0,
                    },
                    "promotion": {
                        "enabled": True,
                        "current_hard_fail": False,
                        "eligible": True,
                        "recommendation": "enable_hard_fail",
                    },
                },
            },
        }
        plan_artifact_ready = orch._build_defect_plan(
            as_of=d,
            round_no=2,
            review=review,
            tests={"returncode": 0, "stderr": "", "stdout": ""},
            gate=gate_ready,
        )
        plan_ready = json.loads(Path(str(plan_artifact_ready.get("json", ""))).read_text(encoding="utf-8"))
        ready_codes = {str(x.get("code", "")) for x in plan_ready.get("defects", [])}
        self.assertIn("GUARD_LOOP_FRONTEND_SNAPSHOT_TREND_HARD_FAIL_PROMOTION_READY", ready_codes)

    def test_build_defect_plan_includes_frontend_antiflap_dual_window_drift(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 14)

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        review = ReviewDelta(
            as_of=d,
            parameter_changes={},
            factor_weights={},
            defects=[],
            pass_gate=True,
        )
        gate = {
            "checks": {"guard_loop_frontend_snapshot_trend_ok": True},
            "metrics": {},
            "guard_loop_frontend_snapshot_trend": {
                "alerts": [],
                "metrics": {
                    "replay_expected_runs": 1,
                    "replay_executed_runs": 1,
                    "replay_missed_runs": 0,
                    "replay_convergence_rate": 1.0,
                    "replay_due_but_reason_missing_runs": 0,
                },
                "antiflap_burnin": {
                    "active": True,
                    "ok": False,
                    "status": "red",
                    "alerts": [
                        "guard_loop_frontend_snapshot_antiflap_burnin_dual_window_suppression_delta_high"
                    ],
                    "metrics": {
                        "suppressed_ratio": 0.9,
                        "replay_missed_runs": 0,
                        "reason_missing_runs": 0,
                        "dual_window_suppressed_ratio_delta": 0.6,
                    },
                    "promotion": {
                        "enabled": True,
                        "current_hard_fail": False,
                        "eligible": False,
                        "recommendation": "keep_monitor",
                    },
                },
            },
        }
        plan_artifact = orch._build_defect_plan(
            as_of=d,
            round_no=1,
            review=review,
            tests={"returncode": 0, "stderr": "", "stdout": ""},
            gate=gate,
        )
        plan = json.loads(Path(str(plan_artifact.get("json", ""))).read_text(encoding="utf-8"))
        codes = {str(x.get("code", "")) for x in plan.get("defects", [])}
        self.assertIn(
            "GUARD_LOOP_FRONTEND_SNAPSHOT_ANTIFLAP_BURNIN_DUAL_WINDOW_SUPPRESSION_DRIFT",
            codes,
        )

    def test_build_defect_plan_includes_frontend_hard_fail_promotion_dual_window_non_apply(self) -> None:
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(td, ignore_errors=True))
        d = date(2026, 2, 14)

        orch = ReleaseOrchestrator(
            settings=self._make_settings(),
            output_dir=td,
            quality_snapshot=lambda as_of: {"completeness": 1.0, "unresolved_conflict_ratio": 0.0},
            backtest_snapshot=lambda as_of: {"positive_window_ratio": 0.8, "max_drawdown": 0.1, "violations": 0},
            run_review=lambda as_of: ReviewDelta(
                as_of=as_of,
                parameter_changes={},
                factor_weights={},
                defects=[],
                pass_gate=True,
            ),
            health_check=lambda as_of, require_review: {"status": "healthy", "missing": []},
            stable_replay_check=lambda as_of, days, run_eod_replay=True: {"passed": True, "checks": []},
            test_all=lambda **kwargs: {"returncode": 0, "stderr": "", "stdout": ""},
            load_json_safely=lambda p: {},
        )
        review = ReviewDelta(
            as_of=d,
            parameter_changes={},
            factor_weights={},
            defects=[],
            pass_gate=True,
        )
        gate = {
            "checks": {"guard_loop_frontend_snapshot_trend_ok": True},
            "metrics": {},
            "guard_loop_frontend_snapshot_trend": {
                "alerts": [
                    "guard_loop_frontend_snapshot_antiflap_burnin_dual_window_suppression_delta_high",
                ],
                "metrics": {
                    "replay_expected_runs": 1,
                    "replay_executed_runs": 1,
                    "replay_missed_runs": 0,
                    "replay_convergence_rate": 1.0,
                    "replay_due_but_reason_missing_runs": 0,
                },
                "antiflap_burnin": {
                    "active": True,
                    "ok": True,
                    "status": "green",
                    "alerts": [
                        "guard_loop_frontend_snapshot_antiflap_burnin_dual_window_suppression_delta_high",
                    ],
                    "metrics": {
                        "suppressed_ratio": 0.1,
                        "replay_missed_runs": 0,
                        "reason_missing_runs": 0,
                    },
                    "dual_window": {
                        "status": "red",
                        "metrics": {
                            "suppressed_ratio_delta": 0.42,
                            "replay_missed_ratio_delta": 0.18,
                            "reason_missing_ratio_delta": 0.11,
                        },
                        "checks": {
                            "samples_ready_ok": True,
                            "suppression_ratio_delta_ok": False,
                            "replay_missed_ratio_delta_ok": False,
                            "traceability_ratio_delta_ok": False,
                        },
                    },
                    "promotion": {
                        "enabled": True,
                        "current_hard_fail": False,
                        "eligible": True,
                        "recommendation": "enable_hard_fail",
                        "controlled_apply": {
                            "enabled": True,
                            "apply_gate": {
                                "reason": "dual_window_drift_red_non_apply",
                                "rollback_guard_ok": True,
                            },
                            "approval": {
                                "found": True,
                                "approved": True,
                                "matches_proposal": True,
                            },
                            "proposal": {
                                "proposal_id": "proposal_dual_window_non_apply",
                            },
                            "rollback_guard": {
                                "rollback_level": "none",
                                "rollback_active": False,
                            },
                        },
                    },
                },
            },
        }
        plan_artifact = orch._build_defect_plan(
            as_of=d,
            round_no=1,
            review=review,
            tests={"returncode": 0, "stderr": "", "stdout": ""},
            gate=gate,
        )
        plan = json.loads(Path(str(plan_artifact.get("json", ""))).read_text(encoding="utf-8"))
        defects = plan.get("defects", []) if isinstance(plan.get("defects", []), list) else []
        codes = {str(x.get("code", "")) for x in defects if isinstance(x, dict)}
        self.assertIn(
            "GUARD_LOOP_FRONTEND_SNAPSHOT_TREND_HARD_FAIL_PROMOTION_DUAL_WINDOW_NON_APPLY",
            codes,
        )
        self.assertNotIn("GUARD_LOOP_FRONTEND_SNAPSHOT_TREND_HARD_FAIL_PROMOTION_READY", codes)
        dual_window_defect = next(
            (
                x
                for x in defects
                if isinstance(x, dict)
                and str(x.get("code", ""))
                == "GUARD_LOOP_FRONTEND_SNAPSHOT_TREND_HARD_FAIL_PROMOTION_DUAL_WINDOW_NON_APPLY"
            ),
            {},
        )
        defect_inputs = (
            dual_window_defect.get("inputs", {})
            if isinstance(dual_window_defect.get("inputs", {}), dict)
            else {}
        )
        self.assertFalse(bool(defect_inputs.get("suppression_delta_ok", True)))
        self.assertFalse(bool(defect_inputs.get("replay_delta_ok", True)))
        self.assertFalse(bool(defect_inputs.get("traceability_delta_ok", True)))
        next_actions = plan.get("next_actions", []) if isinstance(plan.get("next_actions", []), list) else []
        self.assertTrue(len(next_actions) >= 1)
        self.assertIn("frontend_snapshot dual-window drift", str(next_actions[0]))


if __name__ == "__main__":
    unittest.main()
