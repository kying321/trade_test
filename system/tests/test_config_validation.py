from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lie_engine.config import SystemSettings, validate_settings


class ConfigValidationTests(unittest.TestCase):
    def test_validate_settings_ok(self) -> None:
        settings = SystemSettings(
            raw={
                "timezone": "Asia/Shanghai",
                "schedule": {
                    "premarket": "08:40",
                    "intraday_slots": ["10:30", "14:30"],
                    "eod": "15:10",
                    "nightly_review": "20:30",
                },
                "thresholds": {
                    "hurst_trend": 0.6,
                    "hurst_mean_revert": 0.4,
                    "atr_extreme": 2.0,
                    "signal_confidence_min": 60.0,
                    "convexity_min": 3.0,
                },
                "risk": {
                    "max_single_risk_pct": 2.0,
                    "max_total_exposure_pct": 50.0,
                    "max_symbol_pct": 15.0,
                    "max_theme_pct": 25.0,
                    "safety_bucket_pct": 85.0,
                    "convexity_bucket_pct": 15.0,
                },
                "validation": {
                    "data_completeness_min": 0.99,
                    "unresolved_conflict_max": 0.005,
                    "source_confidence_min": 0.75,
                    "low_confidence_source_ratio_max": 0.40,
                    "positive_window_ratio_min": 0.70,
                    "max_drawdown_max": 0.18,
                    "required_stable_replay_days": 3,
                    "cooldown_consecutive_losses": 3,
                    "major_event_window_hours": 24,
                    "factor_lookback_days": 120,
                    "mode_stats_lookback_days": 365,
                    "mode_health_min_samples": 1,
                    "mode_health_min_profit_factor": 1.0,
                    "mode_health_min_win_rate": 0.40,
                    "mode_health_max_drawdown_max": 0.18,
                    "mode_health_max_violations": 0,
                    "mode_adaptive_update_min_samples": 3,
                    "mode_adaptive_update_step": 0.08,
                    "mode_adaptive_good_profit_factor": 1.25,
                    "mode_adaptive_bad_profit_factor": 1.00,
                    "mode_adaptive_good_win_rate": 0.50,
                    "mode_adaptive_bad_win_rate": 0.42,
                    "mode_adaptive_good_drawdown_max": 0.12,
                    "mode_adaptive_bad_drawdown_max": 0.18,
                    "mode_switch_window_days": 20,
                    "mode_switch_max_rate": 0.45,
                    "mode_drift_window_days": 120,
                    "mode_drift_min_live_trades": 30,
                    "mode_drift_win_rate_max_gap": 0.12,
                    "mode_drift_profit_factor_max_gap": 0.40,
                    "mode_drift_focus_runtime_mode_only": True,
                    "ops_slot_window_days": 7,
                    "ops_slot_min_samples": 3,
                    "ops_slot_missing_ratio_max": 0.35,
                    "ops_slot_premarket_anomaly_ratio_max": 0.50,
                    "ops_slot_intraday_anomaly_ratio_max": 0.50,
                    "ops_slot_eod_anomaly_ratio_max": 0.50,
                    "ops_slot_source_confidence_floor": 0.75,
                    "ops_slot_risk_multiplier_floor": 0.20,
                    "ops_reconcile_window_days": 7,
                    "ops_reconcile_min_samples": 3,
                    "ops_reconcile_missing_ratio_max": 0.35,
                    "ops_reconcile_plan_gap_ratio_max": 0.10,
                    "ops_reconcile_closed_count_gap_ratio_max": 0.10,
                    "ops_reconcile_closed_pnl_gap_abs_max": 0.001,
                    "ops_reconcile_open_gap_ratio_max": 0.25,
                    "ops_state_min_samples": 5,
                    "ops_risk_multiplier_floor": 0.35,
                    "ops_risk_multiplier_drift_max": 0.30,
                    "ops_source_confidence_floor": 0.75,
                    "ops_mode_health_fail_days_max": 2,
                    "execution_min_risk_multiplier": 0.20,
                    "source_confidence_floor_risk_multiplier": 0.35,
                    "mode_health_risk_multiplier": 0.50,
                    "mode_health_insufficient_sample_risk_multiplier": 0.85,
                    "review_loop_fast_ratio": 0.10,
                    "review_loop_fast_shard_index": 0,
                    "review_loop_fast_shard_total": 4,
                    "review_backtest_lookback_days": 540,
                },
                "universe": {
                    "core": [
                        {"symbol": "300750", "asset_class": "equity"},
                        {"symbol": "SC2603", "asset_class": "future"},
                    ]
                },
                "data": {"provider_profile": "opensource_dual"},
                "paths": {"output": "output", "sqlite": "output/artifacts/lie_engine.db"},
            }
        )
        out = validate_settings(settings)
        self.assertTrue(out["ok"])
        self.assertEqual(out["summary"]["errors"], 0)

    def test_validate_settings_detects_errors(self) -> None:
        settings = SystemSettings(
            raw={
                "timezone": "Invalid/Timezone",
                "schedule": {"premarket": "25:61", "intraday_slots": [], "eod": "aa:bb", "nightly_review": "23:00"},
                "thresholds": {"hurst_trend": 0.3, "hurst_mean_revert": 0.4, "atr_extreme": -1.0, "convexity_min": 0.0},
                "risk": {"max_single_risk_pct": 20, "max_symbol_pct": 10, "max_total_exposure_pct": 5},
                "validation": {
                    "data_completeness_min": 1.1,
                    "required_stable_replay_days": 0,
                    "review_backtest_lookback_days": 10,
                    "review_backtest_start_date": "2026/02/14",
                    "mode_stats_lookback_days": 7,
                    "mode_health_min_samples": 0,
                    "mode_health_min_profit_factor": 0.0,
                    "mode_health_min_win_rate": 1.2,
                    "mode_health_max_drawdown_max": -0.1,
                    "mode_health_max_violations": -1,
                    "mode_adaptive_update_min_samples": 0,
                    "mode_adaptive_update_step": 1.5,
                    "mode_adaptive_good_profit_factor": 0.0,
                    "mode_adaptive_bad_profit_factor": -0.1,
                    "mode_adaptive_good_win_rate": 0.2,
                    "mode_adaptive_bad_win_rate": 0.5,
                    "mode_adaptive_good_drawdown_max": 0.3,
                    "mode_adaptive_bad_drawdown_max": 0.1,
                    "mode_switch_window_days": 1,
                    "mode_switch_max_rate": 1.5,
                    "mode_drift_window_days": 0,
                    "mode_drift_min_live_trades": 0,
                    "mode_drift_win_rate_max_gap": 1.2,
                    "mode_drift_profit_factor_max_gap": 0.0,
                    "mode_drift_focus_runtime_mode_only": "yes",
                    "ops_slot_window_days": 0,
                    "ops_slot_min_samples": 0,
                    "ops_slot_missing_ratio_max": 1.2,
                    "ops_slot_premarket_anomaly_ratio_max": -0.1,
                    "ops_slot_intraday_anomaly_ratio_max": 1.2,
                    "ops_slot_eod_anomaly_ratio_max": 1.2,
                    "ops_slot_source_confidence_floor": 1.2,
                    "ops_slot_risk_multiplier_floor": -0.1,
                    "ops_reconcile_window_days": 0,
                    "ops_reconcile_min_samples": 0,
                    "ops_reconcile_missing_ratio_max": 1.2,
                    "ops_reconcile_plan_gap_ratio_max": -0.1,
                    "ops_reconcile_closed_count_gap_ratio_max": 1.2,
                    "ops_reconcile_closed_pnl_gap_abs_max": -0.1,
                    "ops_reconcile_open_gap_ratio_max": 1.2,
                    "ops_state_min_samples": 0,
                    "ops_risk_multiplier_floor": -0.2,
                    "ops_risk_multiplier_drift_max": 1.5,
                    "ops_source_confidence_floor": 2.0,
                    "ops_mode_health_fail_days_max": -1,
                    "execution_min_risk_multiplier": 0.8,
                    "source_confidence_floor_risk_multiplier": 0.5,
                    "mode_health_risk_multiplier": -0.1,
                    "mode_health_insufficient_sample_risk_multiplier": 1.2,
                    "review_loop_fast_ratio": 1.5,
                    "review_loop_fast_shard_index": 5,
                    "review_loop_fast_shard_total": 4,
                },
                "universe": {"core": [{"symbol": "BAD"}]},
                "data": {"provider_profile": "bad_profile"},
                "paths": {"output": "", "sqlite": ""},
            }
        )
        out = validate_settings(settings)
        self.assertFalse(out["ok"])
        self.assertGreater(out["summary"]["errors"], 0)
        paths = {x["path"] for x in out.get("errors", [])}
        self.assertIn("validation.source_confidence_floor_risk_multiplier", paths)
        self.assertIn("validation.mode_health_risk_multiplier", paths)
        self.assertIn("validation.mode_health_insufficient_sample_risk_multiplier", paths)
        self.assertIn("validation.mode_adaptive_update_min_samples", paths)
        self.assertIn("validation.mode_adaptive_update_step", paths)
        self.assertIn("validation.mode_adaptive_good_profit_factor", paths)
        self.assertIn("validation.mode_adaptive_bad_profit_factor", paths)
        self.assertIn("validation.mode_adaptive_good_win_rate", paths)
        self.assertIn("validation.mode_adaptive_good_drawdown_max", paths)
        self.assertIn("validation.mode_switch_window_days", paths)
        self.assertIn("validation.mode_switch_max_rate", paths)
        self.assertIn("validation.mode_drift_window_days", paths)
        self.assertIn("validation.mode_drift_min_live_trades", paths)
        self.assertIn("validation.mode_drift_win_rate_max_gap", paths)
        self.assertIn("validation.mode_drift_profit_factor_max_gap", paths)
        self.assertIn("validation.mode_drift_focus_runtime_mode_only", paths)
        self.assertIn("validation.ops_slot_window_days", paths)
        self.assertIn("validation.ops_slot_min_samples", paths)
        self.assertIn("validation.ops_slot_missing_ratio_max", paths)
        self.assertIn("validation.ops_slot_premarket_anomaly_ratio_max", paths)
        self.assertIn("validation.ops_slot_intraday_anomaly_ratio_max", paths)
        self.assertIn("validation.ops_slot_eod_anomaly_ratio_max", paths)
        self.assertIn("validation.ops_slot_source_confidence_floor", paths)
        self.assertIn("validation.ops_slot_risk_multiplier_floor", paths)
        self.assertIn("validation.ops_reconcile_window_days", paths)
        self.assertIn("validation.ops_reconcile_min_samples", paths)
        self.assertIn("validation.ops_reconcile_missing_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_plan_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_closed_count_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_closed_pnl_gap_abs_max", paths)
        self.assertIn("validation.ops_reconcile_open_gap_ratio_max", paths)
        self.assertIn("validation.ops_state_min_samples", paths)
        self.assertIn("validation.ops_risk_multiplier_floor", paths)
        self.assertIn("validation.ops_risk_multiplier_drift_max", paths)
        self.assertIn("validation.ops_source_confidence_floor", paths)
        self.assertIn("validation.ops_mode_health_fail_days_max", paths)


if __name__ == "__main__":
    unittest.main()
