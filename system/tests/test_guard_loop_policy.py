from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


def _load_guard_loop_module():
    module_path = Path(__file__).resolve().parents[1] / "infra" / "local" / "guard_loop.py"
    spec = importlib.util.spec_from_file_location("guard_loop_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard_loop = _load_guard_loop_module()


class GuardLoopPolicyTests(unittest.TestCase):
    def test_parse_epoch_like_accepts_epoch_and_iso(self) -> None:
        self.assertEqual(int(guard_loop._parse_epoch_like("1700000000")), 1700000000)
        self.assertGreater(int(guard_loop._parse_epoch_like("2026-02-19T10:00:00+08:00")), 0)

    def test_parse_epoch_like_rejects_invalid(self) -> None:
        self.assertEqual(int(guard_loop._parse_epoch_like("")), 0)
        self.assertEqual(int(guard_loop._parse_epoch_like("not-a-date")), 0)

    def test_decide_frontend_snapshot_run_due_after_review_bucket(self) -> None:
        out = guard_loop._decide_frontend_snapshot_run(
            enabled=True,
            daemon_bucket="20:30",
            min_bucket_minutes=20 * 60 + 30,
            daemon_date="2026-02-22",
            now_epoch=1_700_000_000,
            last_run_epoch=1_699_000_000,
            last_run_date="2026-02-21",
            cooldown_seconds=60 * 60,
        )
        self.assertTrue(bool(out.get("due", False)))
        self.assertIn("frontend_snapshot_due", list(out.get("reasons", [])))

    def test_decide_frontend_snapshot_run_skips_when_already_ran_today(self) -> None:
        out = guard_loop._decide_frontend_snapshot_run(
            enabled=True,
            daemon_bucket="21:00",
            min_bucket_minutes=20 * 60 + 30,
            daemon_date="2026-02-22",
            now_epoch=1_700_000_000,
            last_run_epoch=1_699_999_000,
            last_run_date="2026-02-22",
            cooldown_seconds=60,
        )
        self.assertFalse(bool(out.get("due", True)))
        self.assertIn("frontend_snapshot_already_ran_today", list(out.get("reasons", [])))

    def test_run_frontend_snapshot_tests_skips_when_dashboard_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = guard_loop._run_frontend_snapshot_tests(
                root=Path(td),
                logs_dir=Path(td) / "output" / "logs",
                script_name="test",
                timeout_seconds=120,
            )
        self.assertFalse(bool(out.get("ok", True)))
        self.assertEqual(str(out.get("status", "")), "skipped")
        self.assertEqual(str(out.get("reason", "")), "dashboard_web_missing")

    def test_resolve_frontend_snapshot_artifact_policy_from_config(self) -> None:
        out = guard_loop._resolve_frontend_snapshot_artifact_policy(
            config_data={
                "validation": {
                    "ops_guard_loop_frontend_snapshot_retention_days": 45,
                    "ops_guard_loop_frontend_snapshot_checksum_index_enabled": False,
                }
            },
            defaults={
                "retention_days": 30,
                "checksum_index_enabled": True,
                "checksum_index_filename": "frontend_snapshot_checksum_index.json",
            },
        )
        self.assertEqual(int(out.get("retention_days", 0)), 45)
        self.assertFalse(bool(out.get("checksum_index_enabled", True)))
        self.assertEqual(
            str(out.get("checksum_index_filename", "")),
            "frontend_snapshot_checksum_index.json",
        )

    def test_frontend_snapshot_artifact_governance_rotates_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            logs_dir = Path(td) / "output" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            stale_stamp = "20260201_203000"
            fresh_stamp = "20260222_203000"
            for stamp in (stale_stamp, fresh_stamp):
                (logs_dir / f"frontend_snapshot_{stamp}.json").write_text("{}", encoding="utf-8")
                (logs_dir / f"frontend_snapshot_{stamp}.stdout.log").write_text("ok", encoding="utf-8")
                (logs_dir / f"frontend_snapshot_{stamp}.stderr.log").write_text("", encoding="utf-8")
            out = guard_loop._apply_frontend_snapshot_artifact_governance(
                logs_dir=logs_dir,
                as_of=guard_loop.date.fromisoformat("2026-02-22"),
                retention_days=7,
                checksum_index_enabled=True,
                checksum_index_filename="frontend_snapshot_checksum_index.json",
            )
            self.assertFalse(bool(out.get("rotation_failed", True)))
            self.assertFalse(bool(out.get("checksum_index_failed", True)))
            self.assertGreaterEqual(int(out.get("rotated_out_count", 0)), 1)
            self.assertFalse((logs_dir / f"frontend_snapshot_{stale_stamp}.json").exists())
            self.assertTrue((logs_dir / f"frontend_snapshot_{fresh_stamp}.json").exists())
            self.assertTrue((logs_dir / "frontend_snapshot_checksum_index.json").exists())

    def test_build_frontend_snapshot_trend_signal_escalates_heavy_on_timeout_density(self) -> None:
        out = guard_loop._build_frontend_snapshot_trend_recovery_signal(
            daemon_date="2026-02-22",
            state_history=[
                {
                    "date": "2026-02-19",
                    "ts": "2026-02-19T20:30:00+08:00",
                    "frontend_snapshot_due": True,
                    "frontend_snapshot_status": "ok",
                    "frontend_snapshot_reason": "success",
                },
                {
                    "date": "2026-02-20",
                    "ts": "2026-02-20T20:30:00+08:00",
                    "frontend_snapshot_due": True,
                    "frontend_snapshot_status": "error",
                    "frontend_snapshot_reason": "timeout",
                    "frontend_snapshot_timed_out": True,
                },
                {
                    "date": "2026-02-21",
                    "ts": "2026-02-21T20:30:00+08:00",
                    "frontend_snapshot_due": True,
                    "frontend_snapshot_status": "error",
                    "frontend_snapshot_reason": "timeout",
                    "frontend_snapshot_timed_out": True,
                },
                {
                    "date": "2026-02-22",
                    "ts": "2026-02-22T20:30:00+08:00",
                    "frontend_snapshot_due": True,
                    "frontend_snapshot_status": "ok",
                    "frontend_snapshot_reason": "success",
                },
            ],
            policy={
                "enabled": True,
                "gate_hard_fail": False,
                "require_active": False,
                "window_days": 14,
                "min_samples": 4,
                "max_failure_ratio": 0.75,
                "max_timeout_ratio": 0.20,
                "max_governance_failure_ratio": 1.0,
                "max_failure_streak": 3,
                "max_timeout_streak": 3,
            },
            replay_allowed=True,
        )
        self.assertTrue(bool(out.get("active", False)))
        self.assertTrue(bool(out.get("due_heavy", False)))
        self.assertTrue(bool(out.get("due_light", False)))
        self.assertIn("FRONTEND_SNAPSHOT_TREND_TIMEOUT_RATIO", out.get("reason_codes", []))
        self.assertIn("FRONTEND_SNAPSHOT_TREND_REPLAY_HEAVY", out.get("reason_codes", []))
        metrics = out.get("metrics", {})
        self.assertAlmostEqual(float(metrics.get("timeout_ratio", 0.0)), 0.5, places=6)

    def test_build_frontend_snapshot_trend_signal_escalates_heavy_on_failure_streak(self) -> None:
        out = guard_loop._build_frontend_snapshot_trend_recovery_signal(
            daemon_date="2026-02-22",
            state_history=[
                {
                    "date": "2026-02-20",
                    "ts": "2026-02-20T20:30:00+08:00",
                    "frontend_snapshot_due": True,
                    "frontend_snapshot_status": "ok",
                },
                {
                    "date": "2026-02-21",
                    "ts": "2026-02-21T20:30:00+08:00",
                    "frontend_snapshot_due": True,
                    "frontend_snapshot_status": "error",
                    "frontend_snapshot_reason": "frontend_snapshot_failed",
                },
                {
                    "date": "2026-02-22",
                    "ts": "2026-02-22T20:30:00+08:00",
                    "frontend_snapshot_due": True,
                    "frontend_snapshot_status": "error",
                    "frontend_snapshot_reason": "frontend_snapshot_failed",
                },
            ],
            policy={
                "enabled": True,
                "gate_hard_fail": False,
                "require_active": False,
                "window_days": 14,
                "min_samples": 3,
                "max_failure_ratio": 1.0,
                "max_timeout_ratio": 1.0,
                "max_governance_failure_ratio": 1.0,
                "max_failure_streak": 1,
                "max_timeout_streak": 3,
            },
            replay_allowed=True,
        )
        self.assertTrue(bool(out.get("due_heavy", False)))
        self.assertIn("FRONTEND_SNAPSHOT_TREND_FAILURE_STREAK", out.get("reason_codes", []))
        self.assertIn("FRONTEND_SNAPSHOT_TREND_REPLAY_HEAVY", out.get("reason_codes", []))

    def test_apply_frontend_snapshot_recovery_antiflap_cooldown_suppresses_due(self) -> None:
        trend_payload = {
            "due_light": True,
            "due_heavy": True,
            "reason_codes": [
                "FRONTEND_SNAPSHOT_TREND_TIMEOUT_RATIO",
                "FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT",
                "FRONTEND_SNAPSHOT_TREND_REPLAY_HEAVY",
            ],
            "reasons": ["frontend_snapshot_trend_heavy_escalation"],
            "metrics": {"timeout_streak_current": 3},
        }
        out = guard_loop._apply_frontend_snapshot_recovery_antiflap(
            trend_payload=trend_payload,
            state_history=[],
            now_epoch=1_700_000_000,
            last_recovery_epoch=1_699_999_000,
            policy={
                "enabled": True,
                "cooldown_hours": 1.0,
                "repeat_timeout_window_runs": 6,
                "repeat_timeout_max_escalations": 2,
                "repeat_timeout_min_timeout_streak": 2,
            },
        )
        self.assertTrue(bool(out.get("suppressed", False)))
        self.assertFalse(bool(out.get("due_light", True)))
        self.assertFalse(bool(out.get("due_heavy", True)))
        self.assertIn(
            "FRONTEND_SNAPSHOT_TREND_ANTIFLAP_COOLDOWN_SUPPRESS",
            out.get("reason_codes", []),
        )

    def test_apply_frontend_snapshot_recovery_antiflap_repeat_timeout_suppresses_due(self) -> None:
        trend_payload = {
            "due_light": True,
            "due_heavy": True,
            "reason_codes": [
                "FRONTEND_SNAPSHOT_TREND_TIMEOUT_STREAK",
                "FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT",
                "FRONTEND_SNAPSHOT_TREND_REPLAY_HEAVY",
            ],
            "reasons": ["frontend_snapshot_trend_heavy_escalation"],
            "metrics": {"timeout_streak_current": 3},
        }
        history = [
            {
                "recovery_reason_codes": [
                    "FRONTEND_SNAPSHOT_TREND_TIMEOUT_RATIO",
                    "FRONTEND_SNAPSHOT_TREND_REPLAY_LIGHT",
                ]
            },
            {
                "recovery_reason_codes": [
                    "FRONTEND_SNAPSHOT_TREND_TIMEOUT_STREAK",
                    "FRONTEND_SNAPSHOT_TREND_REPLAY_HEAVY",
                ]
            },
        ]
        out = guard_loop._apply_frontend_snapshot_recovery_antiflap(
            trend_payload=trend_payload,
            state_history=history,
            now_epoch=1_700_000_000,
            last_recovery_epoch=0,
            policy={
                "enabled": True,
                "cooldown_hours": 0.0,
                "repeat_timeout_window_runs": 6,
                "repeat_timeout_max_escalations": 2,
                "repeat_timeout_min_timeout_streak": 2,
            },
        )
        self.assertTrue(bool(out.get("suppressed", False)))
        self.assertFalse(bool(out.get("due_light", True)))
        self.assertFalse(bool(out.get("due_heavy", True)))
        self.assertIn(
            "FRONTEND_SNAPSHOT_TREND_ANTIFLAP_REPEAT_TIMEOUT_SUPPRESS",
            out.get("reason_codes", []),
        )

    def test_resolve_cadence_lift_thresholds_from_config(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_preset_thresholds(
            config_data={
                "validation": {
                    "ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max": -0.20,
                    "ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max": -0.35,
                    "ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min": 0.20,
                    "ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min": 0.40,
                }
            },
            defaults={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
        )
        self.assertAlmostEqual(float(out["light_applied_delta_max"]), -0.20, places=6)
        self.assertAlmostEqual(float(out["heavy_applied_delta_max"]), -0.35, places=6)
        self.assertAlmostEqual(float(out["light_cooldown_delta_min"]), 0.20, places=6)
        self.assertAlmostEqual(float(out["heavy_cooldown_delta_min"]), 0.40, places=6)

    def test_resolve_cadence_lift_thresholds_normalizes_relations(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_preset_thresholds(
            config_data={
                "validation": {
                    "ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max": -0.20,
                    "ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max": -0.05,
                    "ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min": 0.30,
                    "ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min": 0.10,
                }
            },
            defaults={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
        )
        self.assertAlmostEqual(float(out["heavy_applied_delta_max"]), -0.20, places=6)
        self.assertAlmostEqual(float(out["heavy_cooldown_delta_min"]), 0.30, places=6)

    def test_resolve_cadence_lift_preset_drift_policy_from_config(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_preset_drift_policy(
            config_data={
                "validation": {
                    "ops_guard_loop_cadence_lift_trend_preset_drift_window_days": 21,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_retention_days": 45,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_samples": 9,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_recovery_link_rate": 0.85,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_retro_found_rate": 0.70,
                }
            },
            defaults={
                "window_days": 14,
                "retention_days": 30,
                "min_samples": 6,
                "min_recovery_link_rate": 0.75,
                "min_retro_found_rate": 0.60,
            },
        )
        self.assertEqual(int(out["window_days"]), 21)
        self.assertEqual(int(out["retention_days"]), 45)
        self.assertEqual(int(out["min_samples"]), 9)
        self.assertAlmostEqual(float(out["min_recovery_link_rate"]), 0.85, places=6)
        self.assertAlmostEqual(float(out["min_retro_found_rate"]), 0.70, places=6)

    def test_resolve_cadence_lift_preset_drift_autotune_policy_from_config(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_preset_drift_autotune_policy(
            config_data={
                "validation": {
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_min_samples": 12,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_step_max": 0.04,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_low": 0.25,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_high": 0.65,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_applied_gap_min": 0.06,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_cooldown_gap_min": 0.07,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_apply_cooldown_days": 4,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_min_delta": 0.02,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_window_days": 9,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_staleness_guard_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_max_staleness_days": 5,
                }
            },
            defaults={
                "enabled": True,
                "min_samples": 10,
                "step_max": 0.03,
                "hit_rate_low": 0.20,
                "hit_rate_high": 0.60,
                "applied_gap_min": 0.05,
                "cooldown_gap_min": 0.05,
                "handoff_enabled": True,
                "handoff_apply_cooldown_days": 3,
                "handoff_anti_flap_enabled": True,
                "handoff_anti_flap_min_delta": 0.01,
                "handoff_anti_flap_window_days": 7,
                "handoff_staleness_guard_enabled": True,
                "handoff_max_staleness_days": 7,
            },
        )
        self.assertTrue(bool(out["enabled"]))
        self.assertEqual(int(out["min_samples"]), 12)
        self.assertAlmostEqual(float(out["step_max"]), 0.04, places=6)
        self.assertAlmostEqual(float(out["hit_rate_low"]), 0.25, places=6)
        self.assertAlmostEqual(float(out["hit_rate_high"]), 0.65, places=6)
        self.assertAlmostEqual(float(out["applied_gap_min"]), 0.06, places=6)
        self.assertAlmostEqual(float(out["cooldown_gap_min"]), 0.07, places=6)
        self.assertTrue(bool(out["handoff_enabled"]))
        self.assertEqual(int(out["handoff_apply_cooldown_days"]), 4)
        self.assertTrue(bool(out["handoff_anti_flap_enabled"]))
        self.assertAlmostEqual(float(out["handoff_anti_flap_min_delta"]), 0.02, places=6)
        self.assertEqual(int(out["handoff_anti_flap_window_days"]), 9)
        self.assertTrue(bool(out["handoff_staleness_guard_enabled"]))
        self.assertEqual(int(out["handoff_max_staleness_days"]), 5)

    def test_resolve_cadence_lift_runtime_thresholds_applies_accepted_handoff(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_runtime_thresholds(
            daemon_date="2026-02-21",
            base_thresholds={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            handoff_state={
                "last_reason": "apply_gate_open",
                "last_proposal_id": "abc123",
                "last_proposal_date": "2026-02-20",
                "last_suggested_thresholds": {
                    "light_applied_delta_max": -0.18,
                    "heavy_applied_delta_max": -0.34,
                    "light_cooldown_delta_min": 0.20,
                    "heavy_cooldown_delta_min": 0.38,
                },
            },
            auto_tune_policy={"enabled": True, "handoff_enabled": True},
        )
        self.assertTrue(bool(out["applied"]))
        self.assertEqual(str(out["source"]), "handoff")
        self.assertEqual(str(out["reason"]), "handoff_suggested_thresholds_applied")
        self.assertEqual(int(out["proposal_age_days"]), 1)
        effective = out.get("effective_thresholds", {})
        self.assertAlmostEqual(float(effective.get("light_applied_delta_max", 0.0)), -0.18, places=6)
        self.assertAlmostEqual(float(effective.get("heavy_applied_delta_max", 0.0)), -0.34, places=6)
        self.assertAlmostEqual(float(effective.get("light_cooldown_delta_min", 0.0)), 0.20, places=6)
        self.assertAlmostEqual(float(effective.get("heavy_cooldown_delta_min", 0.0)), 0.38, places=6)

    def test_resolve_cadence_lift_runtime_thresholds_skips_unaccepted_handoff(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_runtime_thresholds(
            daemon_date="2026-02-21",
            base_thresholds={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            handoff_state={
                "last_reason": "apply_cooldown_active",
                "last_proposal_id": "abc123",
                "last_proposal_date": "2026-02-21",
                "last_suggested_thresholds": {
                    "light_applied_delta_max": -0.18,
                    "heavy_applied_delta_max": -0.34,
                    "light_cooldown_delta_min": 0.20,
                    "heavy_cooldown_delta_min": 0.38,
                },
            },
            auto_tune_policy={"enabled": True, "handoff_enabled": True},
        )
        self.assertFalse(bool(out["applied"]))
        self.assertEqual(str(out["source"]), "config")
        self.assertEqual(str(out["reason"]), "handoff_not_accepted")

    def test_resolve_cadence_lift_runtime_thresholds_normalizes_relation(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_runtime_thresholds(
            daemon_date="2026-02-21",
            base_thresholds={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            handoff_state={
                "last_reason": "apply_gate_open",
                "last_proposal_id": "abc123",
                "last_proposal_date": "2026-02-21",
                "last_suggested_thresholds": {
                    "light_applied_delta_max": -0.20,
                    "heavy_applied_delta_max": -0.05,
                    "light_cooldown_delta_min": 0.30,
                    "heavy_cooldown_delta_min": 0.10,
                },
            },
            auto_tune_policy={"enabled": True, "handoff_enabled": True},
        )
        self.assertTrue(bool(out["applied"]))
        effective = out.get("effective_thresholds", {})
        self.assertAlmostEqual(float(effective.get("heavy_applied_delta_max", 0.0)), -0.20, places=6)
        self.assertAlmostEqual(float(effective.get("heavy_cooldown_delta_min", 0.0)), 0.30, places=6)

    def test_resolve_cadence_lift_runtime_thresholds_fallbacks_when_handoff_stale(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_runtime_thresholds(
            daemon_date="2026-02-21",
            base_thresholds={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            handoff_state={
                "last_reason": "apply_gate_open",
                "last_proposal_id": "abc123",
                "last_proposal_date": "2026-02-10",
                "last_suggested_thresholds": {
                    "light_applied_delta_max": -0.18,
                    "heavy_applied_delta_max": -0.34,
                    "light_cooldown_delta_min": 0.20,
                    "heavy_cooldown_delta_min": 0.38,
                },
            },
            auto_tune_policy={
                "enabled": True,
                "handoff_enabled": True,
                "handoff_staleness_guard_enabled": True,
                "handoff_max_staleness_days": 3,
            },
        )
        self.assertFalse(bool(out["applied"]))
        self.assertEqual(str(out["source"]), "config")
        self.assertEqual(str(out["reason"]), "handoff_stale_fallback")
        self.assertFalse(bool((out.get("checks", {}) or {}).get("handoff_staleness_ok", True)))
        self.assertTrue(bool((out.get("staleness_guard", {}) or {}).get("stale", False)))

    def test_resolve_cadence_lift_preset_drift_trendline_policy_from_config(self) -> None:
        out = guard_loop._resolve_guard_loop_cadence_lift_preset_drift_trendline_policy(
            config_data={
                "validation": {
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_recent_days": 5,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_prior_days": 6,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_min_samples": 2,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_recovery_link_drop": 0.15,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_retro_found_drop": 0.10,
                }
            },
            defaults={
                "enabled": True,
                "recent_days": 7,
                "prior_days": 7,
                "min_samples": 3,
                "max_recovery_link_drop": 0.20,
                "max_retro_found_drop": 0.20,
            },
        )
        self.assertTrue(bool(out["enabled"]))
        self.assertEqual(int(out["recent_days"]), 5)
        self.assertEqual(int(out["prior_days"]), 6)
        self.assertEqual(int(out["min_samples"]), 2)
        self.assertAlmostEqual(float(out["max_recovery_link_drop"]), 0.15, places=6)
        self.assertAlmostEqual(float(out["max_retro_found_drop"]), 0.10, places=6)

    def test_build_guard_loop_preset_drift_audit_detects_recovery_gap(self) -> None:
        out = guard_loop._build_guard_loop_preset_drift_audit(
            daemon_date="2026-02-21",
            generated_ts="2026-02-21T20:30:00+08:00",
            state_history=[
                {
                    "date": "2026-02-20",
                    "ts": "2026-02-20T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "light",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": False,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                    "cadence_lift_trend_retro_found": False,
                    "recovery_mode": "none",
                    "recovery_reason_codes": [],
                    "cadence_lift_trend_applied_rate_delta": -0.25,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.10,
                },
                {
                    "date": "2026-02-21",
                    "ts": "2026-02-21T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": False,
                    "recovery_mode": "light",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    "cadence_lift_trend_applied_rate_delta": -0.35,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.35,
                },
            ],
            threshold_policy={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            drift_policy={
                "window_days": 14,
                "retention_days": 30,
                "min_samples": 2,
                "min_recovery_link_rate": 0.75,
                "min_retro_found_rate": 0.60,
            },
        )
        self.assertEqual(str(out["status"]), "critical")
        checks = out.get("checks", {})
        self.assertFalse(bool(checks.get("recovery_link_ok", True)))
        self.assertFalse(bool(checks.get("retro_coverage_ok", True)))
        self.assertIn("guard_loop_preset_drift_recovery_link_low", out.get("alerts", []))
        self.assertIn("guard_loop_preset_drift_retro_coverage_low", out.get("alerts", []))

    def test_build_guard_loop_preset_drift_audit_autotune_recommendations_are_bounded(self) -> None:
        out = guard_loop._build_guard_loop_preset_drift_audit(
            daemon_date="2026-02-21",
            generated_ts="2026-02-21T20:30:00+08:00",
            state_history=[
                {
                    "date": "2026-02-18",
                    "ts": "2026-02-18T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "light",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": False,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "light",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    "cadence_lift_trend_applied_rate_delta": -0.40,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.50,
                },
                {
                    "date": "2026-02-19",
                    "ts": "2026-02-19T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "light",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": False,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "light",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    "cadence_lift_trend_applied_rate_delta": -0.38,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.48,
                },
                {
                    "date": "2026-02-20",
                    "ts": "2026-02-20T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "heavy",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_HEAVY"],
                    "cadence_lift_trend_applied_rate_delta": -0.42,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.55,
                },
                {
                    "date": "2026-02-21",
                    "ts": "2026-02-21T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "heavy",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_HEAVY"],
                    "cadence_lift_trend_applied_rate_delta": -0.45,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.52,
                },
            ],
            threshold_policy={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            drift_policy={
                "window_days": 14,
                "retention_days": 30,
                "min_samples": 2,
                "min_recovery_link_rate": 0.75,
                "min_retro_found_rate": 0.60,
            },
            auto_tune_policy={
                "enabled": True,
                "min_samples": 2,
                "step_max": 0.03,
                "hit_rate_low": 0.20,
                "hit_rate_high": 0.60,
                "applied_gap_min": 0.05,
                "cooldown_gap_min": 0.05,
            },
        )
        auto_tune = out.get("auto_tune", {})
        self.assertTrue(bool(auto_tune.get("ready", False)))
        self.assertTrue(bool(auto_tune.get("apply_recommended", False)))
        self.assertTrue(bool(auto_tune.get("bounded_ok", False)))
        suggested = auto_tune.get("suggested_thresholds", {})
        self.assertLess(float(suggested.get("light_applied_delta_max", 0.0)), -0.15)
        self.assertLess(float(suggested.get("heavy_applied_delta_max", 0.0)), -0.30)
        self.assertGreater(float(suggested.get("light_cooldown_delta_min", 0.0)), 0.15)
        self.assertGreater(float(suggested.get("heavy_cooldown_delta_min", 0.0)), 0.30)
        self.assertLessEqual(
            float(suggested.get("heavy_applied_delta_max", 0.0)),
            float(suggested.get("light_applied_delta_max", 0.0)) - 0.05 + 1e-9,
        )
        self.assertGreaterEqual(
            float(suggested.get("heavy_cooldown_delta_min", 0.0)),
            float(suggested.get("light_cooldown_delta_min", 0.0)) + 0.05 - 1e-9,
        )

    def test_build_guard_loop_preset_drift_audit_trendline_can_block_autotune(self) -> None:
        out = guard_loop._build_guard_loop_preset_drift_audit(
            daemon_date="2026-02-21",
            generated_ts="2026-02-21T20:30:00+08:00",
            state_history=[
                {
                    "date": "2026-02-18",
                    "ts": "2026-02-18T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "light",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": False,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "light",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    "cadence_lift_trend_applied_rate_delta": -0.45,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.50,
                },
                {
                    "date": "2026-02-19",
                    "ts": "2026-02-19T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "light",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": False,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "light",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    "cadence_lift_trend_applied_rate_delta": -0.44,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.49,
                },
                {
                    "date": "2026-02-20",
                    "ts": "2026-02-20T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "light",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": False,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                    "cadence_lift_trend_retro_found": False,
                    "recovery_mode": "none",
                    "recovery_reason_codes": [],
                    "cadence_lift_trend_applied_rate_delta": -0.43,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.48,
                },
                {
                    "date": "2026-02-21",
                    "ts": "2026-02-21T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": False,
                    "recovery_mode": "none",
                    "recovery_reason_codes": [],
                    "cadence_lift_trend_applied_rate_delta": -0.42,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.47,
                },
            ],
            threshold_policy={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            drift_policy={
                "window_days": 14,
                "retention_days": 30,
                "min_samples": 2,
                "min_recovery_link_rate": 0.75,
                "min_retro_found_rate": 0.60,
            },
            auto_tune_policy={
                "enabled": True,
                "min_samples": 2,
                "step_max": 0.03,
                "hit_rate_low": 0.20,
                "hit_rate_high": 0.60,
                "applied_gap_min": 0.05,
                "cooldown_gap_min": 0.05,
            },
            trendline_policy={
                "enabled": True,
                "recent_days": 2,
                "prior_days": 2,
                "min_samples": 2,
                "max_recovery_link_drop": 0.10,
                "max_retro_found_drop": 0.10,
            },
        )
        trendline = out.get("trendline", {})
        auto_tune = out.get("auto_tune", {})
        self.assertEqual(str(trendline.get("status", "")), "warn")
        self.assertTrue(bool(auto_tune.get("ready", False)))
        self.assertFalse(bool(auto_tune.get("apply_recommended", True)))
        self.assertEqual(str(auto_tune.get("reason", "")), "trendline_guardrail_blocked")
        checks = auto_tune.get("checks", {})
        self.assertFalse(bool(checks.get("trendline_guard_ok", True)))

    def test_build_guard_loop_preset_drift_audit_handoff_gate_respects_cooldown(self) -> None:
        out = guard_loop._build_guard_loop_preset_drift_audit(
            daemon_date="2026-02-21",
            generated_ts="2026-02-21T20:30:00+08:00",
            state_history=[
                {
                    "date": "2026-02-18",
                    "ts": "2026-02-18T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "light",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": False,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "light",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    "cadence_lift_trend_applied_rate_delta": -0.40,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.50,
                },
                {
                    "date": "2026-02-19",
                    "ts": "2026-02-19T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "heavy",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_HEAVY"],
                    "cadence_lift_trend_applied_rate_delta": -0.42,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.55,
                },
                {
                    "date": "2026-02-20",
                    "ts": "2026-02-20T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "heavy",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_HEAVY"],
                    "cadence_lift_trend_applied_rate_delta": -0.45,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.52,
                },
                {
                    "date": "2026-02-21",
                    "ts": "2026-02-21T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "heavy",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_HEAVY"],
                    "cadence_lift_trend_applied_rate_delta": -0.47,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.53,
                },
            ],
            threshold_policy={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            drift_policy={
                "window_days": 14,
                "retention_days": 30,
                "min_samples": 2,
                "min_recovery_link_rate": 0.75,
                "min_retro_found_rate": 0.60,
            },
            auto_tune_policy={
                "enabled": True,
                "min_samples": 2,
                "step_max": 0.03,
                "hit_rate_low": 0.20,
                "hit_rate_high": 0.60,
                "applied_gap_min": 0.05,
                "cooldown_gap_min": 0.05,
                "handoff_enabled": True,
                "handoff_apply_cooldown_days": 3,
                "handoff_anti_flap_enabled": True,
                "handoff_anti_flap_min_delta": 0.01,
                "handoff_anti_flap_window_days": 7,
            },
            handoff_state={
                "last_proposal_id": "not-duplicate",
                "last_proposal_date": "2026-02-20",
                "last_deltas": {
                    "light_applied_delta_max": -0.03,
                    "heavy_applied_delta_max": -0.03,
                    "light_cooldown_delta_min": 0.03,
                    "heavy_cooldown_delta_min": 0.03,
                },
            },
        )
        auto_tune = out.get("auto_tune", {})
        self.assertTrue(bool(auto_tune.get("apply_recommended", False)))
        handoff = auto_tune.get("handoff", {})
        gate = handoff.get("apply_gate", {})
        self.assertTrue(bool(handoff.get("proposal_generated", False)))
        self.assertFalse(bool(gate.get("allowed", True)))
        self.assertEqual(str(gate.get("reason", "")), "apply_cooldown_active")
        self.assertTrue(bool(gate.get("cooldown_active", False)))

    def test_build_guard_loop_preset_drift_audit_handoff_gate_blocks_anti_flap(self) -> None:
        out = guard_loop._build_guard_loop_preset_drift_audit(
            daemon_date="2026-02-21",
            generated_ts="2026-02-21T20:30:00+08:00",
            state_history=[
                {
                    "date": "2026-02-19",
                    "ts": "2026-02-19T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "light",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": False,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_LIGHT"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "light",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_LIGHT"],
                    "cadence_lift_trend_applied_rate_delta": -0.40,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.50,
                },
                {
                    "date": "2026-02-20",
                    "ts": "2026-02-20T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "heavy",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_HEAVY"],
                    "cadence_lift_trend_applied_rate_delta": -0.45,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.55,
                },
                {
                    "date": "2026-02-21",
                    "ts": "2026-02-21T20:30:00+08:00",
                    "daemon_bucket": "20:30",
                    "cadence_due": True,
                    "cadence_non_apply_apply_seen": False,
                    "cadence_lift_trend_preset_level": "heavy",
                    "cadence_lift_trend_due_light": True,
                    "cadence_lift_trend_due_heavy": True,
                    "cadence_lift_trend_reason_codes": ["CADENCE_LIFT_TREND_PRESET_HEAVY"],
                    "cadence_lift_trend_retro_found": True,
                    "recovery_mode": "heavy",
                    "recovery_reason_codes": ["CADENCE_LIFT_TREND_REPLAY_HEAVY"],
                    "cadence_lift_trend_applied_rate_delta": -0.47,
                    "cadence_lift_trend_cooldown_block_rate_delta": 0.56,
                },
            ],
            threshold_policy={
                "light_applied_delta_max": -0.15,
                "heavy_applied_delta_max": -0.30,
                "light_cooldown_delta_min": 0.15,
                "heavy_cooldown_delta_min": 0.30,
            },
            drift_policy={
                "window_days": 14,
                "retention_days": 30,
                "min_samples": 2,
                "min_recovery_link_rate": 0.75,
                "min_retro_found_rate": 0.60,
            },
            auto_tune_policy={
                "enabled": True,
                "min_samples": 2,
                "step_max": 0.03,
                "hit_rate_low": 0.20,
                "hit_rate_high": 0.60,
                "applied_gap_min": 0.05,
                "cooldown_gap_min": 0.05,
                "handoff_enabled": True,
                "handoff_apply_cooldown_days": 0,
                "handoff_anti_flap_enabled": True,
                "handoff_anti_flap_min_delta": 0.01,
                "handoff_anti_flap_window_days": 30,
            },
            handoff_state={
                "last_proposal_id": "not-duplicate",
                "last_proposal_date": "2026-02-20",
                "last_deltas": {
                    "light_applied_delta_max": -0.03,
                    "heavy_applied_delta_max": -0.03,
                    "light_cooldown_delta_min": -0.03,
                    "heavy_cooldown_delta_min": -0.03,
                },
            },
        )
        auto_tune = out.get("auto_tune", {})
        self.assertTrue(bool(auto_tune.get("apply_recommended", False)))
        handoff = auto_tune.get("handoff", {})
        gate = handoff.get("apply_gate", {})
        self.assertFalse(bool(gate.get("allowed", True)))
        self.assertEqual(str(gate.get("reason", "")), "anti_flap_guardrail_blocked")
        self.assertTrue(bool(gate.get("anti_flap_blocked", False)))

    def test_write_guard_loop_preset_drift_artifact_rotates_old_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            review_dir = Path(td)
            (review_dir / "2026-02-01_guard_loop_preset_drift.json").write_text("{}", encoding="utf-8")
            (review_dir / "2026-02-01_guard_loop_preset_drift.md").write_text("# old\n", encoding="utf-8")
            payload = {
                "date": "2026-02-21",
                "generated_ts": "2026-02-21T20:30:00+08:00",
                "status": "ok",
                "window_days": 14,
                "min_samples": 2,
                "samples": 3,
                "checks": {"min_samples_ok": True, "recovery_link_ok": True, "retro_coverage_ok": True},
                "alerts": [],
                "thresholds": {
                    "light_applied_delta_max": -0.15,
                    "heavy_applied_delta_max": -0.30,
                    "light_cooldown_delta_min": 0.15,
                    "heavy_cooldown_delta_min": 0.30,
                    "min_recovery_link_rate": 0.75,
                    "min_retro_found_rate": 0.60,
                },
                "metrics": {
                    "recovery_link_light_rate": 1.0,
                    "recovery_link_heavy_rate": 1.0,
                    "retro_found_rate": 1.0,
                    "applied_light_hits": 1,
                    "applied_heavy_hits": 1,
                    "cooldown_light_hits": 1,
                    "cooldown_heavy_hits": 1,
                },
                "series": [],
            }
            out = guard_loop._write_guard_loop_preset_drift_artifact(
                review_dir=review_dir,
                daemon_date="2026-02-21",
                payload=payload,
                retention_days=7,
            )
            self.assertTrue(bool(out["written"]))
            self.assertTrue(Path(str(out["json"])).exists())
            self.assertTrue(Path(str(out["md"])).exists())
            self.assertFalse((review_dir / "2026-02-01_guard_loop_preset_drift.json").exists())
            self.assertFalse((review_dir / "2026-02-01_guard_loop_preset_drift.md").exists())
            self.assertGreaterEqual(int(out["rotated_out_count"]), 1)

    def test_classify_missing_before_eod_treats_daily_as_expected(self) -> None:
        out = guard_loop._classify_missing(
            missing=["daily_briefing", "daily_signals", "sqlite"],
            now_minutes=10 * 60,
            eod_minutes=15 * 60 + 10,
            review_minutes=20 * 60 + 30,
        )
        self.assertIn("daily_briefing", out["time_expected_missing"])
        self.assertIn("daily_signals", out["time_expected_missing"])
        self.assertIn("sqlite", out["core_missing"])

    def test_classify_missing_after_review_treats_review_as_core(self) -> None:
        out = guard_loop._classify_missing(
            missing=["review_report", "review_delta"],
            now_minutes=21 * 60,
            eod_minutes=15 * 60 + 10,
            review_minutes=20 * 60 + 30,
        )
        self.assertEqual(out["time_expected_missing"], [])
        self.assertIn("review_report", out["core_missing"])
        self.assertIn("review_delta", out["core_missing"])

    def test_decide_recovery_heavy_on_health_error(self) -> None:
        out = guard_loop._decide_recovery_mode(
            actionable_bad=True,
            health_status="error",
            consecutive_bad=1,
            bad_threshold=2,
            heavy_due_interval=False,
            busy_marker_active=False,
        )
        self.assertEqual(out["mode"], "heavy")
        self.assertIn("health_error", out["reasons"])

    def test_decide_recovery_downgrades_heavy_when_busy_marker_active(self) -> None:
        out = guard_loop._decide_recovery_mode(
            actionable_bad=True,
            health_status="degraded",
            consecutive_bad=3,
            bad_threshold=2,
            heavy_due_interval=False,
            busy_marker_active=True,
        )
        self.assertEqual(out["mode"], "light")
        self.assertTrue(bool(out["busy_marker_downgraded_heavy"]))
        self.assertIn("busy_marker_downgraded_heavy", out["reasons"])

    def test_decide_recovery_none_when_not_actionable(self) -> None:
        out = guard_loop._decide_recovery_mode(
            actionable_bad=False,
            health_status="ok",
            consecutive_bad=0,
            bad_threshold=2,
            heavy_due_interval=True,
            busy_marker_active=False,
        )
        self.assertEqual(out["mode"], "none")
        self.assertFalse(bool(out["heavy_due"]))
        self.assertFalse(bool(out["light_due"]))

    def test_decide_recovery_light_on_frontend_snapshot_trend_without_health_degradation(self) -> None:
        out = guard_loop._decide_recovery_mode(
            actionable_bad=False,
            health_status="ok",
            consecutive_bad=0,
            bad_threshold=2,
            heavy_due_interval=False,
            busy_marker_active=False,
            frontend_snapshot_due_light=True,
            frontend_snapshot_due_heavy=False,
        )
        self.assertEqual(out["mode"], "light")
        self.assertFalse(bool(out["heavy_due"]))
        self.assertTrue(bool(out["light_due"]))
        self.assertIn("frontend_snapshot_trend_light", out["reasons"])

    def test_resolve_heavy_test_tier_auto_prefers_fast_intraday(self) -> None:
        out = guard_loop._resolve_heavy_test_tier(
            configured_tier="auto",
            daemon_bucket="10:30",
            health_status="degraded",
            recovery_reasons=["heavy_interval_elapsed"],
            recovery_reason_codes=[],
            eod_minutes=15 * 60 + 10,
            review_minutes=20 * 60 + 30,
        )
        self.assertEqual(str(out["tier"]), "fast")
        self.assertEqual(str(out["reason"]), "heavy_interval_probe_fast")
        self.assertTrue(bool(out["auto"]))

    def test_resolve_heavy_test_tier_auto_prefers_standard_after_eod(self) -> None:
        out = guard_loop._resolve_heavy_test_tier(
            configured_tier="auto",
            daemon_bucket="15:30",
            health_status="degraded",
            recovery_reasons=["actionable_health_degraded"],
            recovery_reason_codes=[],
            eod_minutes=15 * 60 + 10,
            review_minutes=20 * 60 + 30,
        )
        self.assertEqual(str(out["tier"]), "standard")
        self.assertEqual(str(out["reason"]), "post_eod_window_standard")
        self.assertTrue(bool(out["auto"]))

    def test_resolve_heavy_test_tier_auto_forces_standard_on_health_error(self) -> None:
        out = guard_loop._resolve_heavy_test_tier(
            configured_tier="auto",
            daemon_bucket="10:30",
            health_status="error",
            recovery_reasons=["health_error"],
            recovery_reason_codes=[],
            eod_minutes=15 * 60 + 10,
            review_minutes=20 * 60 + 30,
        )
        self.assertEqual(str(out["tier"]), "standard")
        self.assertEqual(str(out["reason"]), "health_error_forces_standard")

    def test_build_test_all_args_standard_includes_ratio_and_seed(self) -> None:
        args = guard_loop._build_test_all_args(
            tier="standard",
            fast_ratio=0.1,
            standard_ratio=0.3,
            fast_shard_index=1,
            fast_shard_total=4,
            fast_seed="seed-x",
        )
        self.assertEqual(args[0:3], ["test-all", "--tier", "standard"])
        self.assertIn("--standard-ratio", args)
        self.assertIn("--fast-seed", args)
        self.assertIn("seed-x", args)

    def test_is_test_all_timeout_and_success(self) -> None:
        timeout_run = {
            "ok": True,
            "payload": {"returncode": 124, "timed_out": True, "failed_tests": ["__timeout__"]},
        }
        ok_run = {"ok": True, "payload": {"returncode": 0, "timed_out": False, "failed_tests": []}}
        self.assertTrue(bool(guard_loop._is_test_all_timed_out(timeout_run)))
        self.assertFalse(bool(guard_loop._is_test_all_success(timeout_run)))
        self.assertTrue(bool(guard_loop._is_test_all_success(ok_run)))

    def test_is_stable_replay_success_respects_passed(self) -> None:
        ok_run = {"ok": True, "payload": {"passed": True}}
        failed_run = {"ok": True, "payload": {"passed": False}}
        self.assertTrue(bool(guard_loop._is_stable_replay_success(ok_run)))
        self.assertFalse(bool(guard_loop._is_stable_replay_success(failed_run)))

    def test_is_halfhour_pulse_success_checks_payload_semantics(self) -> None:
        ok_run = {
            "ok": True,
            "payload": {
                "skipped": False,
                "slot_errors": [],
                "run_results": [{"slot_id": "eod", "status": "ok"}],
                "ops": {"status": "ok"},
                "health": {"status": "healthy"},
                "weekly_guardrail": {"status": "ok"},
            },
        }
        failed_run = {
            "ok": True,
            "payload": {
                "skipped": False,
                "slot_errors": [{"slot_id": "eod", "error": "boom"}],
                "run_results": [{"slot_id": "eod", "status": "error"}],
                "ops": {"status": "error"},
            },
        }
        self.assertTrue(bool(guard_loop._is_halfhour_pulse_success(ok_run)))
        self.assertFalse(bool(guard_loop._is_halfhour_pulse_success(failed_run)))

    def test_is_halfhour_pulse_success_respects_allowed_skip_reasons(self) -> None:
        skipped_locked = {
            "ok": True,
            "payload": {"skipped": True, "reason": "scheduler_locked"},
        }
        self.assertTrue(
            bool(
                guard_loop._is_halfhour_pulse_success(
                    skipped_locked,
                    allow_skipped_reasons={"scheduler_locked"},
                )
            )
        )
        self.assertFalse(
            bool(
                guard_loop._is_halfhour_pulse_success(
                    skipped_locked,
                    allow_skipped_reasons=set(),
                )
            )
        )

    def test_is_autorun_retro_success_respects_status(self) -> None:
        green_run = {"ok": True, "payload": {"status": "green"}}
        yellow_run = {"ok": True, "payload": {"status": "yellow"}}
        red_run = {"ok": True, "payload": {"status": "red"}}
        self.assertTrue(bool(guard_loop._is_autorun_retro_success(green_run)))
        self.assertTrue(bool(guard_loop._is_autorun_retro_success(yellow_run)))
        self.assertFalse(bool(guard_loop._is_autorun_retro_success(red_run)))

    def test_is_autorun_retro_success_supports_execution_only_fail_set(self) -> None:
        red_run = {"ok": True, "payload": {"status": "red"}}
        self.assertTrue(
            bool(
                guard_loop._is_autorun_retro_success(
                    red_run,
                    fail_statuses={"error", "failed", "fail"},
                )
            )
        )

    def test_resolve_pulse_max_slot_runs_scales_to_due_backlog(self) -> None:
        out = guard_loop._resolve_pulse_max_slot_runs(
            pulse_preview={"due_slots": ["premarket", "intraday:10:30", "intraday:14:30", "eod"]},
            base_runs=2,
            hard_cap=8,
        )
        self.assertEqual(int(out), 4)

    def test_resolve_pulse_max_slot_runs_respects_hard_cap(self) -> None:
        out = guard_loop._resolve_pulse_max_slot_runs(
            pulse_preview={"due_slots": [str(i) for i in range(12)]},
            base_runs=2,
            hard_cap=5,
        )
        self.assertEqual(int(out), 5)

    def test_should_execute_pulse_filters_duplicate_bucket_reasons(self) -> None:
        self.assertFalse(
            bool(guard_loop._should_execute_pulse(daemon_would_run=True, daemon_reason="same_bucket"))
        )
        self.assertFalse(
            bool(guard_loop._should_execute_pulse(daemon_would_run=True, daemon_reason="pulse_already_executed"))
        )
        self.assertTrue(bool(guard_loop._should_execute_pulse(daemon_would_run=True, daemon_reason="")))

    def test_normalize_weekly_controlled_apply_reads_readiness_payload(self) -> None:
        out = guard_loop._normalize_weekly_controlled_apply(
            {
                "controlled_apply_readiness": {
                    "enabled": True,
                    "mode": "controlled_apply",
                    "reason": "cadence_due",
                    "stability_weeks": 3,
                    "cadence_due": True,
                    "effective_delete_budget": 25000,
                }
            },
            source="daemon",
        )
        self.assertTrue(bool(out["available"]))
        self.assertTrue(bool(out["enabled"]))
        self.assertEqual(str(out["mode"]), "controlled_apply")
        self.assertEqual(int(out["stability_weeks"]), 3)
        self.assertTrue(bool(out["cadence_due"]))
        self.assertEqual(int(out["effective_delete_budget"]), 25000)

    def test_normalize_weekly_controlled_apply_supports_direct_payload(self) -> None:
        out = guard_loop._normalize_weekly_controlled_apply(
            {
                "enabled": True,
                "mode": "dry_run",
                "reason": "cadence_not_due",
                "stability_weeks": 2,
                "cadence_due": False,
                "effective_delete_budget": 80,
            },
            source="daemon.pulse_preview",
        )
        self.assertTrue(bool(out["available"]))
        self.assertEqual(str(out["mode"]), "dry_run")
        self.assertEqual(str(out["reason"]), "cadence_not_due")
        self.assertEqual(int(out["stability_weeks"]), 2)
        self.assertEqual(int(out["effective_delete_budget"]), 80)

    def test_resolve_weekly_controlled_apply_falls_back_to_pulse_payload(self) -> None:
        out = guard_loop._resolve_weekly_controlled_apply(
            daemon_payload={},
            pulse_preview={},
            pulse_payload={
                "weekly_guardrail": {
                    "controlled_apply_readiness": {
                        "enabled": True,
                        "mode": "dry_run",
                        "reason": "cadence_not_due",
                        "stability_weeks": 2,
                        "cadence_due": False,
                        "effective_delete_budget": 80,
                    }
                }
            },
        )
        self.assertTrue(bool(out["available"]))
        self.assertEqual(str(out["source"]), "pulse.weekly_guardrail")
        self.assertEqual(int(out["stability_weeks"]), 2)
        self.assertFalse(bool(out["cadence_due"]))
        self.assertEqual(int(out["effective_delete_budget"]), 80)

    def test_resolve_weekly_guardrail_payload_prefers_daemon_then_pulse(self) -> None:
        daemon_payload = {"weekly_guardrail": {"status": "ok", "reason": "executed"}}
        out_daemon = guard_loop._resolve_weekly_guardrail_payload(
            daemon_payload=daemon_payload,
            pulse_preview={},
            pulse_payload={},
        )
        self.assertTrue(bool(out_daemon["available"]))
        self.assertEqual(str(out_daemon["source"]), "daemon")

        out_pulse = guard_loop._resolve_weekly_guardrail_payload(
            daemon_payload={},
            pulse_preview={},
            pulse_payload={"weekly_guardrail": {"status": "ok"}},
        )
        self.assertTrue(bool(out_pulse["available"]))
        self.assertEqual(str(out_pulse["source"]), "pulse")

    def test_detect_controlled_apply_execution_marks_apply_seen(self) -> None:
        out = guard_loop._detect_controlled_apply_execution(
            weekly_guardrail={
                "maintenance": {
                    "compact": {
                        "ran": True,
                        "dry_run": False,
                        "status": "ok",
                        "policy": {"mode": "controlled_apply"},
                    }
                }
            },
            cadence_due=True,
            pulse_ran=True,
        )
        self.assertTrue(bool(out["apply_seen"]))
        self.assertEqual(str(out["policy_mode"]), "controlled_apply")
        self.assertIn("compact_apply_executed", out["evidence"])

    def test_detect_controlled_apply_execution_keeps_non_apply_evidence(self) -> None:
        out = guard_loop._detect_controlled_apply_execution(
            weekly_guardrail={
                "maintenance": {
                    "compact": {
                        "ran": True,
                        "dry_run": True,
                        "status": "ok",
                    }
                }
            },
            cadence_due=True,
            pulse_ran=True,
        )
        self.assertFalse(bool(out["apply_seen"]))
        self.assertIn("compact_dry_run", out["evidence"])

    def test_cadence_non_apply_escalation_counts_unique_windows(self) -> None:
        first = guard_loop._cadence_non_apply_escalation(
            previous_streak=0,
            previous_window_key="",
            window_key="2026-02-20@10:30",
            cadence_due=True,
            apply_seen=False,
            no_pulse_exec=False,
            light_threshold=2,
            heavy_threshold=4,
        )
        self.assertEqual(int(first["streak_windows"]), 1)
        self.assertFalse(bool(first["due_light"]))

        duplicate = guard_loop._cadence_non_apply_escalation(
            previous_streak=int(first["streak_windows"]),
            previous_window_key=str(first["last_window_key"]),
            window_key="2026-02-20@10:30",
            cadence_due=True,
            apply_seen=False,
            no_pulse_exec=False,
            light_threshold=2,
            heavy_threshold=4,
        )
        self.assertEqual(int(duplicate["streak_windows"]), 1)
        self.assertFalse(bool(duplicate["counted_window"]))

        second_window = guard_loop._cadence_non_apply_escalation(
            previous_streak=int(duplicate["streak_windows"]),
            previous_window_key=str(duplicate["last_window_key"]),
            window_key="2026-02-20@11:00",
            cadence_due=True,
            apply_seen=False,
            no_pulse_exec=False,
            light_threshold=2,
            heavy_threshold=4,
        )
        self.assertEqual(int(second_window["streak_windows"]), 2)
        self.assertTrue(bool(second_window["due_light"]))
        self.assertIn("CADENCE_DUE_NON_APPLY_STREAK", second_window["reason_codes"])

        reset = guard_loop._cadence_non_apply_escalation(
            previous_streak=int(second_window["streak_windows"]),
            previous_window_key=str(second_window["last_window_key"]),
            window_key="2026-02-20@11:30",
            cadence_due=True,
            apply_seen=True,
            no_pulse_exec=False,
            light_threshold=2,
            heavy_threshold=4,
        )
        self.assertEqual(int(reset["streak_windows"]), 0)
        self.assertFalse(bool(reset["non_apply"]))

    def test_cadence_non_apply_escalation_heavy_includes_reason_codes(self) -> None:
        out = guard_loop._cadence_non_apply_escalation(
            previous_streak=3,
            previous_window_key="2026-02-20@11:00",
            window_key="2026-02-20@11:30",
            cadence_due=True,
            apply_seen=False,
            no_pulse_exec=True,
            light_threshold=2,
            heavy_threshold=4,
        )
        self.assertTrue(bool(out["due_heavy"]))
        self.assertIn("CADENCE_DUE_NON_APPLY_HEAVY", out["reason_codes"])
        self.assertIn("CADENCE_DUE_NON_APPLY_NO_PULSE_EXEC", out["reason_codes"])

    def test_cadence_lift_trend_recovery_preset_heavy_due(self) -> None:
        out = guard_loop._cadence_lift_trend_recovery_preset(
            cadence_non_apply={"non_apply": True},
            retro_payload={
                "metrics": {
                    "cadence_lift_applied_rate_delta": -0.40,
                    "cadence_lift_cooldown_block_rate_delta": 0.35,
                    "cadence_lift_recent_samples": 5,
                    "cadence_lift_prior_samples": 5,
                },
                "cadence_lift_trend": {
                    "active": True,
                    "alerts": ["guard_loop_cadence_lift_trend_applied_rate_low"],
                },
            },
            replay_allowed=True,
            busy_marker_active=False,
            light_applied_delta_max=-0.15,
            heavy_applied_delta_max=-0.30,
            light_cooldown_delta_min=0.15,
            heavy_cooldown_delta_min=0.30,
        )
        self.assertTrue(bool(out["active"]))
        self.assertEqual(str(out["suggested_level"]), "heavy")
        self.assertTrue(bool(out["due_heavy"]))
        self.assertTrue(bool(out["due_light"]))
        self.assertIn("CADENCE_LIFT_TREND_PRESET_HEAVY", out["reason_codes"])

    def test_cadence_lift_trend_recovery_preset_replay_disabled(self) -> None:
        out = guard_loop._cadence_lift_trend_recovery_preset(
            cadence_non_apply={"non_apply": True},
            retro_payload={
                "metrics": {
                    "cadence_lift_applied_rate_delta": -0.25,
                    "cadence_lift_cooldown_block_rate_delta": 0.20,
                    "cadence_lift_recent_samples": 4,
                    "cadence_lift_prior_samples": 4,
                },
                "cadence_lift_trend": {"active": True, "alerts": []},
            },
            replay_allowed=False,
            busy_marker_active=False,
            light_applied_delta_max=-0.15,
            heavy_applied_delta_max=-0.30,
            light_cooldown_delta_min=0.15,
            heavy_cooldown_delta_min=0.30,
        )
        self.assertEqual(str(out["suggested_level"]), "light")
        self.assertFalse(bool(out["due_light"]))
        self.assertFalse(bool(out["due_heavy"]))
        self.assertIn("CADENCE_LIFT_TREND_PRESET_REPLAY_DISABLED", out["reason_codes"])

    def test_compute_gap_metrics_detects_threshold_breach(self) -> None:
        rows = [
            {"ts": "2026-02-20T08:00:00+08:00"},
            {"ts": "2026-02-20T08:30:00+08:00"},
            {"ts": "2026-02-20T10:00:00+08:00"},
        ]
        out = guard_loop._compute_gap_metrics(rows=rows, threshold_minutes=45.0)
        self.assertEqual(int(out["event_count"]), 3)
        self.assertEqual(int(out["gap_events_over_threshold"]), 1)
        self.assertGreater(float(out["max_gap_minutes"]), 45.0)

    def test_compute_gap_metrics_handles_empty_or_singleton(self) -> None:
        out0 = guard_loop._compute_gap_metrics(rows=[], threshold_minutes=45.0)
        out1 = guard_loop._compute_gap_metrics(rows=[{"ts": "2026-02-20T08:00:00+08:00"}], threshold_minutes=45.0)
        self.assertEqual(int(out0["gap_events_over_threshold"]), 0)
        self.assertEqual(int(out1["gap_events_over_threshold"]), 0)
        self.assertEqual(float(out0["max_gap_minutes"]), 0.0)
        self.assertEqual(float(out1["max_gap_minutes"]), 0.0)

    def test_should_trigger_gap_backfill_due(self) -> None:
        out = guard_loop._should_trigger_gap_backfill(
            gap_events_over_threshold=2,
            daemon_would_run=False,
            pulse_ran=False,
            no_pulse_exec=False,
            busy_marker_active=False,
            now_epoch=1_800_000_000,
            last_gap_backfill_epoch=0,
            cooldown_seconds=3600,
        )
        self.assertTrue(bool(out["due"]))
        self.assertIn("gap_backfill_due", out["reasons"])

    def test_should_trigger_gap_backfill_respects_cooldown(self) -> None:
        out = guard_loop._should_trigger_gap_backfill(
            gap_events_over_threshold=1,
            daemon_would_run=False,
            pulse_ran=False,
            no_pulse_exec=False,
            busy_marker_active=False,
            now_epoch=1_800_000_000,
            last_gap_backfill_epoch=1_799_999_500,
            cooldown_seconds=3600,
        )
        self.assertFalse(bool(out["due"]))
        self.assertIn("gap_backfill_cooldown", out["reasons"])


if __name__ == "__main__":
    unittest.main()
