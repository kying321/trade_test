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
                    "ops_temporal_audit_enabled": True,
                    "ops_temporal_audit_lookback_days": 45,
                    "ops_temporal_audit_min_samples": 1,
                    "ops_temporal_audit_missing_ratio_max": 0.20,
                    "ops_temporal_audit_leak_ratio_max": 0.00,
                    "ops_temporal_audit_autofix_enabled": True,
                    "ops_temporal_audit_autofix_max_writes": 3,
                    "ops_temporal_audit_autofix_fix_strict_cutoff": True,
                    "ops_temporal_audit_autofix_require_safe": True,
                    "ops_temporal_audit_autofix_patch_retention_days": 30,
                    "ops_temporal_audit_autofix_patch_checksum_index_enabled": True,
                    "ops_stress_matrix_trend_enabled": True,
                    "ops_stress_matrix_trend_window_runs": 8,
                    "ops_stress_matrix_trend_min_runs": 3,
                    "ops_stress_matrix_robustness_drop_max": 0.15,
                    "ops_stress_matrix_annual_return_drop_max": 0.08,
                    "ops_stress_matrix_drawdown_rise_max": 0.08,
                    "ops_stress_matrix_fail_ratio_max": 0.50,
                    "ops_stress_autorun_history_enabled": True,
                    "ops_stress_autorun_history_window_days": 30,
                    "ops_stress_autorun_history_min_rounds": 3,
                    "ops_stress_autorun_adaptive_monitor_enabled": True,
                    "ops_stress_autorun_adaptive_monitor_window_days": 30,
                    "ops_stress_autorun_adaptive_monitor_min_rounds": 3,
                    "ops_stress_autorun_adaptive_effective_base_ratio_floor": 0.50,
                    "ops_stress_autorun_adaptive_effective_base_ratio_ceiling": 2.00,
                    "ops_stress_autorun_adaptive_throttle_ratio_max": 0.85,
                    "ops_stress_autorun_adaptive_expand_ratio_max": 0.85,
                    "ops_stress_autorun_reason_drift_enabled": True,
                    "ops_stress_autorun_reason_drift_window_days": 30,
                    "ops_stress_autorun_reason_drift_min_rounds": 6,
                    "ops_stress_autorun_reason_drift_recent_rounds": 4,
                    "ops_stress_autorun_reason_drift_mix_gap_max": 0.35,
                    "ops_stress_autorun_reason_drift_change_point_gap_max": 0.45,
                    "ops_slot_window_days": 7,
                    "ops_slot_min_samples": 3,
                    "ops_slot_missing_ratio_max": 0.35,
                    "ops_slot_premarket_anomaly_ratio_max": 0.50,
                    "ops_slot_intraday_anomaly_ratio_max": 0.50,
                    "ops_slot_eod_anomaly_ratio_max": 0.50,
                    "ops_slot_eod_quality_anomaly_ratio_max": 0.50,
                    "ops_slot_eod_risk_anomaly_ratio_max": 0.50,
                    "ops_slot_eod_quality_anomaly_ratio_max_by_regime": {
                        "trend": 0.55,
                        "range": 0.50,
                        "extreme_vol": 0.35,
                    },
                    "ops_slot_eod_risk_anomaly_ratio_max_by_regime": {
                        "trend": 0.55,
                        "range": 0.50,
                        "extreme_vol": 0.30,
                    },
                    "ops_slot_use_live_regime_thresholds": True,
                    "ops_slot_regime_tune_enabled": True,
                    "ops_slot_regime_tune_window_days": 180,
                    "ops_slot_regime_tune_min_days": 20,
                    "ops_slot_regime_tune_step": 0.12,
                    "ops_slot_regime_tune_buffer": 0.08,
                    "ops_slot_regime_tune_floor": 0.10,
                    "ops_slot_regime_tune_ceiling": 0.80,
                    "ops_slot_regime_tune_missing_ratio_hard_cap": 0.80,
                    "ops_slot_source_confidence_floor": 0.75,
                    "ops_slot_risk_multiplier_floor": 0.20,
                    "ops_reconcile_window_days": 7,
                    "ops_reconcile_min_samples": 3,
                    "ops_reconcile_missing_ratio_max": 0.35,
                    "ops_reconcile_plan_gap_ratio_max": 0.10,
                    "ops_reconcile_closed_count_gap_ratio_max": 0.10,
                    "ops_reconcile_closed_pnl_gap_abs_max": 0.001,
                    "ops_reconcile_open_gap_ratio_max": 0.25,
                    "ops_reconcile_broker_missing_ratio_max": 0.50,
                    "ops_reconcile_broker_gap_ratio_max": 0.10,
                    "ops_reconcile_broker_pnl_gap_abs_max": 0.001,
                    "ops_reconcile_broker_contract_schema_invalid_ratio_max": 0.10,
                    "ops_reconcile_broker_contract_numeric_invalid_ratio_max": 0.10,
                    "ops_reconcile_broker_contract_symbol_invalid_ratio_max": 0.10,
                    "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": 0.40,
                    "ops_reconcile_broker_closed_pnl_abs_hard_max": 1e9,
                    "ops_reconcile_broker_position_qty_abs_hard_max": 1e9,
                    "ops_reconcile_broker_position_notional_abs_hard_max": 1e10,
                    "ops_reconcile_broker_price_abs_hard_max": 1e8,
                    "ops_reconcile_broker_row_diff_min_samples": 1,
                    "ops_reconcile_broker_row_diff_breach_ratio_max": 0.20,
                    "ops_reconcile_broker_row_diff_key_mismatch_max": 0.25,
                    "ops_reconcile_broker_row_diff_count_gap_max": 0.25,
                    "ops_reconcile_broker_row_diff_notional_gap_max": 0.50,
                    "ops_reconcile_broker_row_diff_alias_monitor_enabled": True,
                    "ops_reconcile_broker_row_diff_alias_hit_rate_min": 0.00,
                    "ops_reconcile_broker_row_diff_unresolved_key_ratio_max": 0.30,
                    "ops_reconcile_broker_row_diff_asof_only": True,
                    "ops_reconcile_broker_row_diff_symbol_alias_map": {"RB2405": "RB2405.SHFE"},
                    "ops_reconcile_broker_row_diff_side_alias_map": {"SELLSHORT": "SHORT"},
                    "ops_reconcile_require_broker_snapshot": False,
                    "ops_reconcile_broker_contract_emit_canonical_view": True,
                    "ops_reconcile_broker_contract_canonical_dir": "artifacts/broker_snapshot_canonical",
                    "ops_state_min_samples": 5,
                    "ops_risk_multiplier_floor": 0.35,
                    "ops_risk_multiplier_drift_max": 0.30,
                    "ops_source_confidence_floor": 0.75,
                    "ops_mode_health_fail_days_max": 2,
                    "broker_snapshot_source_mode": "paper_engine",
                    "broker_snapshot_live_inbox": "output/artifacts/broker_live_inbox",
                    "broker_snapshot_live_fallback_to_paper": True,
                    "execution_min_risk_multiplier": 0.20,
                    "source_confidence_floor_risk_multiplier": 0.35,
                    "mode_health_risk_multiplier": 0.50,
                    "mode_health_insufficient_sample_risk_multiplier": 0.85,
                    "test_all_timeout_seconds": 1800,
                    "review_loop_fast_ratio": 0.10,
                    "review_loop_fast_shard_index": 0,
                    "review_loop_fast_shard_total": 4,
                    "review_loop_timeout_fallback_enabled": True,
                    "review_loop_timeout_fallback_ratio": 0.08,
                    "review_loop_timeout_fallback_shard_index": 0,
                    "review_loop_timeout_fallback_shard_total": 2,
                    "review_loop_timeout_fallback_seed": "lie-timeout-v1",
                    "review_loop_stress_matrix_autorun_enabled": True,
                    "review_loop_stress_matrix_autorun_on_mode_drift": True,
                    "review_loop_stress_matrix_autorun_on_stress_breach": True,
                    "review_loop_stress_matrix_autorun_max_runs": 1,
                    "review_loop_stress_matrix_autorun_cooldown_rounds": 1,
                    "review_loop_stress_matrix_autorun_backoff_multiplier": 2.0,
                    "review_loop_stress_matrix_autorun_backoff_max_rounds": 8,
                    "review_loop_stress_matrix_autorun_adaptive_enabled": True,
                    "review_loop_stress_matrix_autorun_adaptive_window_days": 30,
                    "review_loop_stress_matrix_autorun_adaptive_min_rounds": 6,
                    "review_loop_stress_matrix_autorun_adaptive_low_density_threshold": 0.20,
                    "review_loop_stress_matrix_autorun_adaptive_high_density_threshold": 0.60,
                    "review_loop_stress_matrix_autorun_adaptive_low_density_factor": 1.50,
                    "review_loop_stress_matrix_autorun_adaptive_high_density_factor": 0.50,
                    "review_loop_stress_matrix_autorun_adaptive_min_runs_floor": 0,
                    "review_loop_stress_matrix_autorun_adaptive_max_runs_cap": 3,
                    "review_loop_stress_matrix_autorun_modes": ["swing", "long"],
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
                    "ops_temporal_audit_enabled": "yes",
                    "ops_temporal_audit_lookback_days": 0,
                    "ops_temporal_audit_min_samples": 0,
                    "ops_temporal_audit_missing_ratio_max": 1.2,
                    "ops_temporal_audit_leak_ratio_max": -0.1,
                    "ops_temporal_audit_autofix_enabled": "yes",
                    "ops_temporal_audit_autofix_max_writes": -1,
                    "ops_temporal_audit_autofix_fix_strict_cutoff": "yes",
                    "ops_temporal_audit_autofix_require_safe": "yes",
                    "ops_temporal_audit_autofix_patch_retention_days": 0,
                    "ops_temporal_audit_autofix_patch_checksum_index_enabled": "yes",
                    "ops_stress_matrix_trend_enabled": "yes",
                    "ops_stress_matrix_trend_window_runs": 1,
                    "ops_stress_matrix_trend_min_runs": 3,
                    "ops_stress_matrix_robustness_drop_max": 1.2,
                    "ops_stress_matrix_annual_return_drop_max": -0.1,
                    "ops_stress_matrix_drawdown_rise_max": 1.2,
                    "ops_stress_matrix_fail_ratio_max": -0.1,
                    "ops_stress_autorun_history_enabled": "yes",
                    "ops_stress_autorun_history_window_days": 0,
                    "ops_stress_autorun_history_min_rounds": 0,
                    "ops_stress_autorun_adaptive_monitor_enabled": "yes",
                    "ops_stress_autorun_adaptive_monitor_window_days": 0,
                    "ops_stress_autorun_adaptive_monitor_min_rounds": 0,
                    "ops_stress_autorun_adaptive_effective_base_ratio_floor": -0.1,
                    "ops_stress_autorun_adaptive_effective_base_ratio_ceiling": 0.0,
                    "ops_stress_autorun_adaptive_throttle_ratio_max": 1.2,
                    "ops_stress_autorun_adaptive_expand_ratio_max": -0.1,
                    "ops_stress_autorun_reason_drift_enabled": "yes",
                    "ops_stress_autorun_reason_drift_window_days": 0,
                    "ops_stress_autorun_reason_drift_min_rounds": 1,
                    "ops_stress_autorun_reason_drift_recent_rounds": 0,
                    "ops_stress_autorun_reason_drift_mix_gap_max": 1.2,
                    "ops_stress_autorun_reason_drift_change_point_gap_max": -0.1,
                    "ops_slot_window_days": 0,
                    "ops_slot_min_samples": 0,
                    "ops_slot_missing_ratio_max": 1.2,
                    "ops_slot_premarket_anomaly_ratio_max": -0.1,
                    "ops_slot_intraday_anomaly_ratio_max": 1.2,
                    "ops_slot_eod_anomaly_ratio_max": 1.2,
                    "ops_slot_eod_quality_anomaly_ratio_max": 1.2,
                    "ops_slot_eod_risk_anomaly_ratio_max": -0.1,
                    "ops_slot_eod_quality_anomaly_ratio_max_by_regime": {"trend": 1.2},
                    "ops_slot_eod_risk_anomaly_ratio_max_by_regime": "bad",
                    "ops_slot_use_live_regime_thresholds": "yes",
                    "ops_slot_regime_tune_enabled": "yes",
                    "ops_slot_regime_tune_window_days": 1,
                    "ops_slot_regime_tune_min_days": 0,
                    "ops_slot_regime_tune_step": 1.2,
                    "ops_slot_regime_tune_buffer": -0.1,
                    "ops_slot_regime_tune_floor": 0.9,
                    "ops_slot_regime_tune_ceiling": 0.1,
                    "ops_slot_regime_tune_missing_ratio_hard_cap": 1.2,
                    "ops_slot_source_confidence_floor": 1.2,
                    "ops_slot_risk_multiplier_floor": -0.1,
                    "ops_reconcile_window_days": 0,
                    "ops_reconcile_min_samples": 0,
                    "ops_reconcile_missing_ratio_max": 1.2,
                    "ops_reconcile_plan_gap_ratio_max": -0.1,
                    "ops_reconcile_closed_count_gap_ratio_max": 1.2,
                    "ops_reconcile_closed_pnl_gap_abs_max": -0.1,
                    "ops_reconcile_open_gap_ratio_max": 1.2,
                    "ops_reconcile_broker_missing_ratio_max": 1.2,
                    "ops_reconcile_broker_gap_ratio_max": -0.1,
                    "ops_reconcile_broker_pnl_gap_abs_max": -0.1,
                    "ops_reconcile_broker_contract_schema_invalid_ratio_max": 1.2,
                    "ops_reconcile_broker_contract_numeric_invalid_ratio_max": -0.1,
                    "ops_reconcile_broker_contract_symbol_invalid_ratio_max": 1.2,
                    "ops_reconcile_broker_contract_symbol_noncanonical_ratio_max": -0.1,
                    "ops_reconcile_broker_closed_pnl_abs_hard_max": 0,
                    "ops_reconcile_broker_position_qty_abs_hard_max": -1,
                    "ops_reconcile_broker_position_notional_abs_hard_max": 0,
                    "ops_reconcile_broker_price_abs_hard_max": -1,
                    "ops_reconcile_broker_row_diff_min_samples": 0,
                    "ops_reconcile_broker_row_diff_breach_ratio_max": 1.2,
                    "ops_reconcile_broker_row_diff_key_mismatch_max": -0.1,
                    "ops_reconcile_broker_row_diff_count_gap_max": 1.2,
                    "ops_reconcile_broker_row_diff_notional_gap_max": -0.1,
                    "ops_reconcile_broker_row_diff_alias_monitor_enabled": "yes",
                    "ops_reconcile_broker_row_diff_alias_hit_rate_min": 1.2,
                    "ops_reconcile_broker_row_diff_unresolved_key_ratio_max": -0.1,
                    "ops_reconcile_broker_row_diff_asof_only": "yes",
                    "ops_reconcile_broker_row_diff_symbol_alias_map": {" ": ""},
                    "ops_reconcile_broker_row_diff_side_alias_map": {"": "UNKNOWN"},
                    "ops_reconcile_require_broker_snapshot": "yes",
                    "ops_reconcile_broker_contract_emit_canonical_view": "yes",
                    "ops_reconcile_broker_contract_canonical_dir": "",
                    "ops_state_min_samples": 0,
                    "ops_risk_multiplier_floor": -0.2,
                    "ops_risk_multiplier_drift_max": 1.5,
                    "ops_source_confidence_floor": 2.0,
                    "ops_mode_health_fail_days_max": -1,
                    "broker_snapshot_source_mode": "bad_mode",
                    "broker_snapshot_live_mapping_profile": "bad_profile",
                    "broker_snapshot_live_mapping_fields": {
                        "source": "source",
                        "position_fields": {"symbol": "symbol"},
                    },
                    "broker_snapshot_live_inbox": "",
                    "broker_snapshot_live_fallback_to_paper": "yes",
                    "execution_min_risk_multiplier": 0.8,
                    "source_confidence_floor_risk_multiplier": 0.5,
                    "mode_health_risk_multiplier": -0.1,
                    "mode_health_insufficient_sample_risk_multiplier": 1.2,
                    "test_all_timeout_seconds": 10,
                    "review_loop_fast_ratio": 1.5,
                    "review_loop_fast_shard_index": 5,
                    "review_loop_fast_shard_total": 4,
                    "review_loop_timeout_fallback_enabled": "yes",
                    "review_loop_timeout_fallback_ratio": 0.0,
                    "review_loop_timeout_fallback_shard_index": 5,
                    "review_loop_timeout_fallback_shard_total": 4,
                    "review_loop_stress_matrix_autorun_enabled": "yes",
                    "review_loop_stress_matrix_autorun_on_mode_drift": "yes",
                    "review_loop_stress_matrix_autorun_on_stress_breach": "yes",
                    "review_loop_stress_matrix_autorun_max_runs": -1,
                    "review_loop_stress_matrix_autorun_cooldown_rounds": -1,
                    "review_loop_stress_matrix_autorun_backoff_multiplier": 0.9,
                    "review_loop_stress_matrix_autorun_backoff_max_rounds": -2,
                    "review_loop_stress_matrix_autorun_adaptive_enabled": "yes",
                    "review_loop_stress_matrix_autorun_adaptive_window_days": 0,
                    "review_loop_stress_matrix_autorun_adaptive_min_rounds": 0,
                    "review_loop_stress_matrix_autorun_adaptive_low_density_threshold": 1.2,
                    "review_loop_stress_matrix_autorun_adaptive_high_density_threshold": -0.1,
                    "review_loop_stress_matrix_autorun_adaptive_low_density_factor": 0.0,
                    "review_loop_stress_matrix_autorun_adaptive_high_density_factor": -1.0,
                    "review_loop_stress_matrix_autorun_adaptive_min_runs_floor": -1,
                    "review_loop_stress_matrix_autorun_adaptive_max_runs_cap": -1,
                    "review_loop_stress_matrix_autorun_modes": ["", "swing"],
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
        self.assertIn("validation.ops_temporal_audit_enabled", paths)
        self.assertIn("validation.ops_temporal_audit_lookback_days", paths)
        self.assertIn("validation.ops_temporal_audit_min_samples", paths)
        self.assertIn("validation.ops_temporal_audit_missing_ratio_max", paths)
        self.assertIn("validation.ops_temporal_audit_leak_ratio_max", paths)
        self.assertIn("validation.ops_temporal_audit_autofix_enabled", paths)
        self.assertIn("validation.ops_temporal_audit_autofix_max_writes", paths)
        self.assertIn("validation.ops_temporal_audit_autofix_fix_strict_cutoff", paths)
        self.assertIn("validation.ops_temporal_audit_autofix_require_safe", paths)
        self.assertIn("validation.ops_temporal_audit_autofix_patch_retention_days", paths)
        self.assertIn("validation.ops_temporal_audit_autofix_patch_checksum_index_enabled", paths)
        self.assertIn("validation.ops_stress_matrix_trend_enabled", paths)
        self.assertIn("validation.ops_stress_matrix_trend_window_runs", paths)
        self.assertIn("validation.ops_stress_matrix_trend_min_runs", paths)
        self.assertIn("validation.ops_stress_matrix_robustness_drop_max", paths)
        self.assertIn("validation.ops_stress_matrix_annual_return_drop_max", paths)
        self.assertIn("validation.ops_stress_matrix_drawdown_rise_max", paths)
        self.assertIn("validation.ops_stress_matrix_fail_ratio_max", paths)
        self.assertIn("validation.ops_stress_autorun_history_enabled", paths)
        self.assertIn("validation.ops_stress_autorun_history_window_days", paths)
        self.assertIn("validation.ops_stress_autorun_history_min_rounds", paths)
        self.assertIn("validation.ops_stress_autorun_adaptive_monitor_enabled", paths)
        self.assertIn("validation.ops_stress_autorun_adaptive_monitor_window_days", paths)
        self.assertIn("validation.ops_stress_autorun_adaptive_monitor_min_rounds", paths)
        self.assertIn("validation.ops_stress_autorun_adaptive_effective_base_ratio_floor", paths)
        self.assertIn("validation.ops_stress_autorun_adaptive_effective_base_ratio_ceiling", paths)
        self.assertIn("validation.ops_stress_autorun_adaptive_throttle_ratio_max", paths)
        self.assertIn("validation.ops_stress_autorun_adaptive_expand_ratio_max", paths)
        self.assertIn("validation.ops_stress_autorun_reason_drift_enabled", paths)
        self.assertIn("validation.ops_stress_autorun_reason_drift_window_days", paths)
        self.assertIn("validation.ops_stress_autorun_reason_drift_min_rounds", paths)
        self.assertIn("validation.ops_stress_autorun_reason_drift_recent_rounds", paths)
        self.assertIn("validation.ops_stress_autorun_reason_drift_mix_gap_max", paths)
        self.assertIn("validation.ops_stress_autorun_reason_drift_change_point_gap_max", paths)
        self.assertIn("validation.ops_slot_window_days", paths)
        self.assertIn("validation.ops_slot_min_samples", paths)
        self.assertIn("validation.ops_slot_missing_ratio_max", paths)
        self.assertIn("validation.ops_slot_premarket_anomaly_ratio_max", paths)
        self.assertIn("validation.ops_slot_intraday_anomaly_ratio_max", paths)
        self.assertIn("validation.ops_slot_eod_anomaly_ratio_max", paths)
        self.assertIn("validation.ops_slot_eod_quality_anomaly_ratio_max", paths)
        self.assertIn("validation.ops_slot_eod_risk_anomaly_ratio_max", paths)
        self.assertIn("validation.ops_slot_eod_quality_anomaly_ratio_max_by_regime.trend", paths)
        self.assertIn("validation.ops_slot_eod_risk_anomaly_ratio_max_by_regime", paths)
        self.assertIn("validation.ops_slot_use_live_regime_thresholds", paths)
        self.assertIn("validation.ops_slot_regime_tune_enabled", paths)
        self.assertIn("validation.ops_slot_regime_tune_window_days", paths)
        self.assertIn("validation.ops_slot_regime_tune_min_days", paths)
        self.assertIn("validation.ops_slot_regime_tune_step", paths)
        self.assertIn("validation.ops_slot_regime_tune_buffer", paths)
        self.assertIn("validation.ops_slot_regime_tune_floor", paths)
        self.assertIn("validation.ops_slot_regime_tune_missing_ratio_hard_cap", paths)
        self.assertIn("validation.ops_slot_source_confidence_floor", paths)
        self.assertIn("validation.ops_slot_risk_multiplier_floor", paths)
        self.assertIn("validation.ops_reconcile_window_days", paths)
        self.assertIn("validation.ops_reconcile_min_samples", paths)
        self.assertIn("validation.ops_reconcile_missing_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_plan_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_closed_count_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_closed_pnl_gap_abs_max", paths)
        self.assertIn("validation.ops_reconcile_open_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_missing_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_pnl_gap_abs_max", paths)
        self.assertIn("validation.ops_reconcile_broker_contract_schema_invalid_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_contract_numeric_invalid_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_contract_symbol_invalid_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_contract_symbol_noncanonical_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_closed_pnl_abs_hard_max", paths)
        self.assertIn("validation.ops_reconcile_broker_position_qty_abs_hard_max", paths)
        self.assertIn("validation.ops_reconcile_broker_position_notional_abs_hard_max", paths)
        self.assertIn("validation.ops_reconcile_broker_price_abs_hard_max", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_min_samples", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_breach_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_key_mismatch_max", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_count_gap_max", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_notional_gap_max", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_alias_monitor_enabled", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_alias_hit_rate_min", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_unresolved_key_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_asof_only", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_symbol_alias_map[ ]", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_side_alias_map", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_side_alias_map[]", paths)
        self.assertIn("validation.ops_reconcile_require_broker_snapshot", paths)
        self.assertIn("validation.ops_reconcile_broker_contract_emit_canonical_view", paths)
        self.assertIn("validation.ops_reconcile_broker_contract_canonical_dir", paths)
        self.assertIn("validation.ops_state_min_samples", paths)
        self.assertIn("validation.ops_risk_multiplier_floor", paths)
        self.assertIn("validation.ops_risk_multiplier_drift_max", paths)
        self.assertIn("validation.ops_source_confidence_floor", paths)
        self.assertIn("validation.ops_mode_health_fail_days_max", paths)
        self.assertIn("validation.broker_snapshot_source_mode", paths)
        self.assertIn("validation.broker_snapshot_live_mapping_profile", paths)
        self.assertIn("validation.broker_snapshot_live_mapping_fields.source", paths)
        self.assertIn("validation.broker_snapshot_live_mapping_fields.position_fields.symbol", paths)
        self.assertIn("validation.broker_snapshot_live_inbox", paths)
        self.assertIn("validation.broker_snapshot_live_fallback_to_paper", paths)
        self.assertIn("validation.test_all_timeout_seconds", paths)
        self.assertIn("validation.review_loop_timeout_fallback_enabled", paths)
        self.assertIn("validation.review_loop_timeout_fallback_ratio", paths)
        self.assertIn("validation.review_loop_timeout_fallback_shard_index", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_enabled", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_on_mode_drift", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_on_stress_breach", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_max_runs", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_cooldown_rounds", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_backoff_multiplier", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_backoff_max_rounds", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_enabled", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_window_days", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_min_rounds", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_low_density_threshold", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_high_density_threshold", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_low_density_factor", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_high_density_factor", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_min_runs_floor", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_adaptive_max_runs_cap", paths)
        self.assertIn("validation.review_loop_stress_matrix_autorun_modes[0]", paths)


if __name__ == "__main__":
    unittest.main()
