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
                    "style_attribution_lookback_days": 180,
                    "style_drift_window_days": 20,
                    "style_drift_min_sample_days": 20,
                    "style_drift_gap_max": 0.01,
                    "style_drift_block_on_alert": False,
                    "style_drift_adaptive_enabled": True,
                    "style_drift_adaptive_confidence_step_max": 6.0,
                    "style_drift_adaptive_trade_reduction_max": 2,
                    "style_drift_adaptive_hold_reduction_max": 2,
                    "style_drift_adaptive_trigger_ratio": 1.0,
                    "style_drift_adaptive_ratio_for_max": 2.0,
                    "style_drift_adaptive_block_ratio": 1.8,
                    "style_drift_gate_enabled": True,
                    "style_drift_gate_require_active": False,
                    "style_drift_gate_allow_alerts": True,
                    "style_drift_gate_max_alerts": 0,
                    "style_drift_gate_max_ratio": 2.0,
                    "style_drift_gate_hard_fail": True,
                    "style_drift_gate_lookback_days": 7,
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
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled": True,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required": True,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days": 7,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path": "artifacts/stress_matrix_execution_friction_trendline_autotune_approval.json",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_enabled": True,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_retention_days": 180,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_staleness_window_days": 30,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path": "artifacts/stress_matrix_execution_friction_trendline_controlled_apply_ledger.json",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_enabled": True,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_gate_hard_fail": False,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_window_stale_ratio_max": 0.20,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_max": 0.30,
                    "ops_stress_autorun_history_enabled": True,
                    "ops_stress_autorun_history_window_days": 30,
                    "ops_stress_autorun_history_min_rounds": 3,
                    "ops_stress_autorun_history_no_trigger_min_payload_days": 14,
                    "ops_stress_autorun_history_retention_days": 30,
                    "ops_stress_autorun_history_checksum_index_enabled": True,
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
                    "ops_stress_autorun_reason_drift_retention_days": 30,
                    "ops_stress_autorun_reason_drift_checksum_index_enabled": True,
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
                    "ops_slot_degradation_enabled": True,
                    "ops_slot_hysteresis_enabled": True,
                    "ops_slot_degradation_soft_multiplier": 1.15,
                    "ops_slot_degradation_hard_multiplier": 1.40,
                    "ops_slot_hysteresis_soft_streak_days": 2,
                    "ops_slot_hysteresis_hard_streak_days": 3,
                    "ops_reconcile_window_days": 7,
                    "ops_reconcile_min_samples": 3,
                    "ops_reconcile_missing_ratio_max": 0.35,
                    "ops_reconcile_plan_gap_ratio_max": 0.10,
                    "ops_reconcile_closed_count_gap_ratio_max": 0.10,
                    "ops_reconcile_closed_pnl_gap_abs_max": 0.001,
                    "ops_reconcile_open_gap_ratio_max": 0.25,
                    "ops_reconcile_executed_dedup_monitor_enabled": True,
                    "ops_reconcile_executed_dedup_gate_hard_fail": False,
                    "ops_reconcile_executed_dedup_pruned_ratio_max": 0.35,
                    "ops_reconcile_executed_dedup_days_ratio_max": 0.50,
                    "ops_reconcile_executed_dedup_restore_verify_enabled": True,
                    "ops_reconcile_executed_dedup_restore_verify_hard_fail": False,
                    "ops_reconcile_executed_dedup_restore_verify_max_age_days": 14,
                    "ops_reconcile_executed_dedup_restore_verify_min_backup_rows": 1,
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
                    "ops_reconcile_broker_row_diff_artifact_retention_days": 30,
                    "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled": True,
                    "ops_reconcile_broker_row_diff_symbol_alias_map": {"RB2405": "RB2405.SHFE"},
                    "ops_reconcile_broker_row_diff_side_alias_map": {"SELLSHORT": "SHORT"},
                    "ops_artifact_governance_profiles": {
                        "reconcile_row_diff": {
                            "json_glob": "*_reconcile_row_diff.json",
                            "md_glob": "*_reconcile_row_diff.md",
                            "checksum_index_filename": "reconcile_row_diff_checksum_index.json",
                            "retention_days": 30,
                            "checksum_index_enabled": True,
                        }
                    },
                    "ops_artifact_governance_strict_mode_enabled": False,
                    "ops_artifact_governance_baseline_snapshot_enabled": True,
                    "ops_artifact_governance_baseline_snapshot_path": "artifacts/baselines/artifact_governance/active_baseline.yaml",
                    "ops_artifact_governance_baseline_history_dir": "artifacts/baselines/artifact_governance/history",
                    "ops_artifact_governance_baseline_auto_promote_on_review_pass": True,
                    "ops_artifact_governance_profile_baseline": {
                        "reconcile_row_diff": {
                            "json_glob": "*_reconcile_row_diff.json",
                            "md_glob": "*_reconcile_row_diff.md",
                            "checksum_index_filename": "reconcile_row_diff_checksum_index.json",
                            "retention_days": 30,
                            "checksum_index_enabled": True,
                        }
                    },
                    "ops_reconcile_require_broker_snapshot": False,
                    "ops_reconcile_broker_contract_emit_canonical_view": True,
                    "ops_reconcile_broker_contract_canonical_dir": "artifacts/broker_snapshot_canonical",
                    "ops_state_min_samples": 5,
                    "ops_risk_multiplier_floor": 0.35,
                    "ops_risk_multiplier_drift_max": 0.30,
                    "ops_source_confidence_floor": 0.75,
                    "ops_mode_health_fail_days_max": 2,
                    "ops_state_switch_rate_max_by_mode": {"ultra_short": 1.2, "swing": 1.0, "long": 0.9},
                    "ops_state_degradation_enabled": True,
                    "ops_state_hysteresis_enabled": True,
                    "ops_state_degradation_soft_multiplier": 1.10,
                    "ops_state_degradation_hard_multiplier": 1.35,
                    "ops_state_degradation_floor_soft_ratio": 0.96,
                    "ops_state_degradation_floor_hard_ratio": 0.90,
                    "ops_state_hysteresis_soft_streak_days": 2,
                    "ops_state_hysteresis_hard_streak_days": 3,
                    "ops_degradation_calibration_enabled": True,
                    "ops_degradation_calibration_window_days": 30,
                    "ops_degradation_calibration_min_samples": 20,
                    "ops_degradation_calibration_fp_target": 0.08,
                    "ops_degradation_calibration_fn_target": 0.05,
                    "ops_degradation_calibration_step_multiplier": 0.03,
                    "ops_degradation_calibration_step_floor_ratio": 0.02,
                    "ops_degradation_calibration_step_streak_days": 1,
                    "ops_degradation_calibration_multiplier_min": 1.0,
                    "ops_degradation_calibration_multiplier_max": 2.0,
                    "ops_degradation_calibration_floor_ratio_min": 0.70,
                    "ops_degradation_calibration_floor_ratio_max": 1.00,
                    "ops_degradation_calibration_streak_min": 1,
                    "ops_degradation_calibration_streak_max": 7,
                    "ops_degradation_calibration_use_live_overrides": True,
                    "ops_degradation_calibration_live_params_path": "artifacts/degradation_params_live.yaml",
                    "ops_degradation_calibration_rollback_enabled": True,
                    "ops_degradation_calibration_rollback_window_days": 30,
                    "ops_degradation_calibration_rollback_recent_days": 5,
                    "ops_degradation_calibration_rollback_min_samples": 20,
                    "ops_degradation_calibration_rollback_fn_rise_min": 0.03,
                    "ops_degradation_calibration_rollback_gate_fail_rise_min": 0.10,
                    "ops_degradation_calibration_rollback_stable_min_samples": 20,
                    "ops_degradation_calibration_rollback_stable_fn_rate_max": 0.05,
                    "ops_degradation_calibration_rollback_stable_gate_fail_ratio_max": 0.20,
                    "ops_degradation_calibration_rollback_auto_promote_on_stable": True,
                    "ops_degradation_calibration_rollback_cooldown_days": 3,
                    "ops_degradation_calibration_rollback_promotion_cooldown_days": 7,
                    "ops_degradation_calibration_rollback_hysteresis_window_days": 7,
                    "ops_degradation_calibration_rollback_trigger_hysteresis_buffer": 0.02,
                    "ops_degradation_calibration_rollback_stable_hysteresis_buffer": 0.01,
                    "ops_degradation_calibration_rollback_active_snapshot_path": "artifacts/baselines/degradation_calibration/active_snapshot.yaml",
                    "ops_degradation_calibration_rollback_history_dir": "artifacts/baselines/degradation_calibration/history",
                    "ops_degradation_guardrail_dashboard_enabled": True,
                    "ops_degradation_guardrail_dashboard_hard_fail": False,
                    "ops_degradation_guardrail_dashboard_window_days": 21,
                    "ops_degradation_guardrail_dashboard_min_samples": 5,
                    "ops_degradation_guardrail_dashboard_use_live_overrides": True,
                    "ops_degradation_guardrail_dashboard_live_params_path": "artifacts/degradation_guardrail_dashboard_live.yaml",
                    "ops_degradation_guardrail_cooldown_hit_rate_max": 0.85,
                    "ops_degradation_guardrail_suppressed_trigger_density_max": 0.65,
                    "ops_degradation_guardrail_promotion_latency_days_max": 14,
                    "ops_degradation_guardrail_false_positive_target_max": 0.25,
                    "ops_degradation_guardrail_burnin_autofill_review_if_missing": True,
                    "ops_degradation_guardrail_burnin_require_min_samples_for_tune": True,
                    "ops_degradation_guardrail_burnin_light_backfill_enabled": True,
                    "ops_degradation_guardrail_burnin_light_backfill_max_days_per_run": 2,
                    "ops_degradation_guardrail_burnin_review_autofill_max_days_per_run": 0,
                    "ops_degradation_guardrail_burnin_low_cost_replay_enabled": True,
                    "ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run": 2,
                    "ops_degradation_guardrail_burnin_use_live_overrides": True,
                    "ops_degradation_guardrail_burnin_live_params_path": "artifacts/degradation_guardrail_burnin_live.yaml",
                    "ops_degradation_guardrail_burnin_budget_audit_enabled": True,
                    "ops_degradation_guardrail_burnin_budget_audit_auto_tune": True,
                    "ops_degradation_guardrail_burnin_budget_audit_expand_recovery_ratio_min": 0.75,
                    "ops_degradation_guardrail_burnin_budget_audit_shrink_recovery_ratio_max": 0.40,
                    "ops_degradation_guardrail_burnin_budget_audit_step_days": 1,
                    "ops_degradation_guardrail_burnin_budget_audit_min_days": 0,
                    "ops_degradation_guardrail_burnin_budget_audit_max_days": 7,
                    "ops_degradation_guardrail_threshold_drift_enabled": True,
                    "ops_degradation_guardrail_threshold_drift_gate_hard_fail": False,
                    "ops_degradation_guardrail_threshold_drift_require_active": False,
                    "ops_degradation_guardrail_threshold_drift_max_staleness_days": 14,
                    "ops_degradation_guardrail_threshold_drift_warn_ratio": 0.25,
                    "ops_degradation_guardrail_threshold_drift_critical_ratio": 0.40,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_enabled": True,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_cooldown_days": 2,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_lookback_days": 30,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_allow_upgrade_during_cooldown": True,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_force_heavy_hard": True,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_light_streak_min": 2,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min": 4,
                    "ops_guard_loop_cadence_non_apply_lift_trend_enabled": True,
                    "ops_guard_loop_cadence_non_apply_lift_trend_gate_hard_fail": False,
                    "ops_guard_loop_cadence_non_apply_lift_trend_require_active": False,
                    "ops_guard_loop_cadence_non_apply_lift_trend_window_days": 30,
                    "ops_guard_loop_cadence_non_apply_lift_trend_min_samples": 7,
                    "ops_guard_loop_cadence_non_apply_lift_trend_applied_rate_min": 0.05,
                    "ops_guard_loop_cadence_non_apply_lift_trend_cooldown_block_rate_max": 0.80,
                    "ops_guard_loop_cadence_non_apply_lift_trend_retention_days": 30,
                    "ops_guard_loop_cadence_non_apply_lift_trend_checksum_index_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max": -0.15,
                    "ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max": -0.30,
                    "ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min": 0.15,
                    "ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min": 0.30,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_window_days": 14,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_retention_days": 30,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_samples": 6,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_recovery_link_rate": 0.75,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_retro_found_rate": 0.60,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_min_samples": 10,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_step_max": 0.03,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_low": 0.20,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_high": 0.60,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_applied_gap_min": 0.05,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_cooldown_gap_min": 0.05,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_apply_cooldown_days": 3,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_min_delta": 0.01,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_window_days": 7,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_staleness_guard_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_max_staleness_days": 7,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_enabled": True,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_recent_days": 7,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_prior_days": 7,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_min_samples": 3,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_recovery_link_drop": 0.20,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_retro_found_drop": 0.20,
                    "ops_guard_loop_frontend_snapshot_trend_enabled": True,
                    "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": False,
                    "ops_guard_loop_frontend_snapshot_trend_require_active": False,
                    "ops_guard_loop_frontend_snapshot_trend_window_days": 14,
                    "ops_guard_loop_frontend_snapshot_trend_min_samples": 4,
                    "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": 0.35,
                    "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 0.20,
                    "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 0.20,
                    "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 2,
                    "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 2,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_enabled": True,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_cooldown_hours": 6.0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_window_runs": 8,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_max_escalations": 2,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_min_timeout_streak": 2,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled": True,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days": 3,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples": 3,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio": 0.50,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs": True,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_retention_days": 30,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_checksum_index_enabled": True,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_enabled": True,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_long_days": 14,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_min_long_samples": 3,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_suppression_ratio_delta": 0.35,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_replay_missed_ratio_delta": 0.10,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_reason_missing_ratio_delta": 0.10,
                    "ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin": False,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_enabled": True,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_manual_approval_required": True,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_max_apply_window_days": 7,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path": "output/artifacts/frontend_snapshot_trend_hard_fail_approval.json",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_dry_run": True,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_rollback_guard_enabled": True,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_retention_days": 60,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_checksum_index_enabled": True,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_enabled": True,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_gate_hard_fail": False,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_window_days": 28,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_samples": 7,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_non_apply_ratio": 0.85,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_rollback_guard_ratio": 0.35,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_apply_ratio": 0.05,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_recent_days": 7,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_prior_days": 7,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_min_samples": 3,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_non_apply_ratio_rise": 0.25,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_rollback_guard_ratio_rise": 0.15,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_apply_ratio_drop": 0.20,
                    "ops_compaction_restore_trend_gate_hard_fail": False,
                    "ops_weekly_guardrail_enabled": True,
                    "ops_weekly_guardrail_weekday": 7,
                    "ops_weekly_guardrail_trigger_slot": "20:30",
                    "ops_weekly_guardrail_burnin_days": 7,
                    "ops_weekly_guardrail_drift_window_days": 56,
                    "ops_weekly_guardrail_auto_tune": True,
                    "ops_weekly_guardrail_run_stable_replay": True,
                    "ops_weekly_guardrail_require_ops_window": True,
                    "ops_weekly_guardrail_require_burnin_coverage": True,
                    "ops_weekly_guardrail_compact_enabled": True,
                    "ops_weekly_guardrail_compact_dry_run": True,
                    "ops_weekly_guardrail_compact_window_days": 180,
                    "ops_weekly_guardrail_compact_chunk_days": 30,
                    "ops_weekly_guardrail_compact_max_delete_rows": None,
                    "ops_weekly_guardrail_compact_verify_restore": True,
                    "ops_weekly_guardrail_compact_verify_keep_temp_db": False,
                    "ops_weekly_guardrail_compact_controlled_apply_enabled": True,
                    "ops_weekly_guardrail_compact_controlled_apply_stability_weeks": 2,
                    "ops_weekly_guardrail_compact_controlled_apply_cadence_weeks": 2,
                    "ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows": 25000,
                    "ops_weekly_guardrail_compact_controlled_apply_require_restore_verify_pass": True,
                    "ops_weekly_guardrail_compact_controlled_apply_require_weekly_status_ok": True,
                    "weekly_guardrail_history_limit": 104,
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
                    "style_attribution_lookback_days": 1,
                    "style_drift_window_days": 1,
                    "style_drift_min_sample_days": 2,
                    "style_drift_gap_max": -0.1,
                    "style_drift_block_on_alert": "yes",
                    "style_drift_adaptive_enabled": "yes",
                    "style_drift_adaptive_confidence_step_max": -0.1,
                    "style_drift_adaptive_trade_reduction_max": -1,
                    "style_drift_adaptive_hold_reduction_max": -1,
                    "style_drift_adaptive_trigger_ratio": 0.0,
                    "style_drift_adaptive_ratio_for_max": 0.0,
                    "style_drift_adaptive_block_ratio": 0.8,
                    "style_drift_gate_enabled": "yes",
                    "style_drift_gate_require_active": "yes",
                    "style_drift_gate_allow_alerts": "yes",
                    "style_drift_gate_max_alerts": -1,
                    "style_drift_gate_max_ratio": 0.0,
                    "style_drift_gate_hard_fail": "yes",
                    "style_drift_gate_lookback_days": 0,
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
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled": "yes",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required": "yes",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days": -1,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path": "",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_enabled": "yes",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_retention_days": 0,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_staleness_window_days": 0,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path": "",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_enabled": "yes",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_gate_hard_fail": "yes",
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_window_stale_ratio_max": 1.2,
                    "ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_max": -0.1,
                    "ops_stress_autorun_history_enabled": "yes",
                    "ops_stress_autorun_history_window_days": 0,
                    "ops_stress_autorun_history_min_rounds": 0,
                    "ops_stress_autorun_history_no_trigger_min_payload_days": 0,
                    "ops_stress_autorun_history_retention_days": 0,
                    "ops_stress_autorun_history_checksum_index_enabled": "yes",
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
                    "ops_stress_autorun_reason_drift_retention_days": 0,
                    "ops_stress_autorun_reason_drift_checksum_index_enabled": "yes",
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
                    "ops_slot_degradation_enabled": "yes",
                    "ops_slot_hysteresis_enabled": "yes",
                    "ops_slot_degradation_soft_multiplier": 0.9,
                    "ops_slot_degradation_hard_multiplier": 0.8,
                    "ops_slot_hysteresis_soft_streak_days": 0,
                    "ops_slot_hysteresis_hard_streak_days": 0,
                    "ops_reconcile_window_days": 0,
                    "ops_reconcile_min_samples": 0,
                    "ops_reconcile_missing_ratio_max": 1.2,
                    "ops_reconcile_plan_gap_ratio_max": -0.1,
                    "ops_reconcile_closed_count_gap_ratio_max": 1.2,
                    "ops_reconcile_closed_pnl_gap_abs_max": -0.1,
                    "ops_reconcile_open_gap_ratio_max": 1.2,
                    "ops_reconcile_executed_dedup_monitor_enabled": "yes",
                    "ops_reconcile_executed_dedup_gate_hard_fail": "yes",
                    "ops_reconcile_executed_dedup_pruned_ratio_max": 1.2,
                    "ops_reconcile_executed_dedup_days_ratio_max": -0.1,
                    "ops_reconcile_executed_dedup_restore_verify_enabled": "yes",
                    "ops_reconcile_executed_dedup_restore_verify_hard_fail": "yes",
                    "ops_reconcile_executed_dedup_restore_verify_max_age_days": 0,
                    "ops_reconcile_executed_dedup_restore_verify_min_backup_rows": 0,
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
                    "ops_reconcile_broker_row_diff_artifact_retention_days": 0,
                    "ops_reconcile_broker_row_diff_artifact_checksum_index_enabled": "yes",
                    "ops_reconcile_broker_row_diff_symbol_alias_map": {" ": ""},
                    "ops_reconcile_broker_row_diff_side_alias_map": {"": "UNKNOWN"},
                    "ops_artifact_governance_profiles": {
                        "": {"json_glob": "*_x.json"},
                        "bad_type": "x",
                        "reconcile_row_diff": {
                            "json_glob": "",
                            "md_glob": 1,
                            "checksum_index_filename": "",
                            "retention_days": 0,
                            "checksum_index_enabled": "yes",
                        },
                    },
                    "ops_artifact_governance_strict_mode_enabled": "yes",
                    "ops_artifact_governance_baseline_snapshot_enabled": "yes",
                    "ops_artifact_governance_baseline_snapshot_path": "",
                    "ops_artifact_governance_baseline_history_dir": 123,
                    "ops_artifact_governance_baseline_auto_promote_on_review_pass": "yes",
                    "ops_artifact_governance_profile_baseline": {
                        "": {"json_glob": "*_x.json"},
                        "bad_type": "x",
                        "reconcile_row_diff": {
                            "json_glob": "",
                            "md_glob": 1,
                            "checksum_index_filename": "",
                            "retention_days": 0,
                            "checksum_index_enabled": "yes",
                        },
                    },
                    "ops_reconcile_require_broker_snapshot": "yes",
                    "ops_reconcile_broker_contract_emit_canonical_view": "yes",
                    "ops_reconcile_broker_contract_canonical_dir": "",
                    "ops_state_min_samples": 0,
                    "ops_risk_multiplier_floor": -0.2,
                    "ops_risk_multiplier_drift_max": 1.5,
                    "ops_source_confidence_floor": 2.0,
                    "ops_mode_health_fail_days_max": -1,
                    "ops_state_switch_rate_max_by_mode": {"": -1.0, "swing": 0.0},
                    "ops_state_degradation_enabled": "yes",
                    "ops_state_hysteresis_enabled": "yes",
                    "ops_state_degradation_soft_multiplier": 0.9,
                    "ops_state_degradation_hard_multiplier": 0.8,
                    "ops_state_degradation_floor_soft_ratio": 1.2,
                    "ops_state_degradation_floor_hard_ratio": -0.1,
                    "ops_state_hysteresis_soft_streak_days": 0,
                    "ops_state_hysteresis_hard_streak_days": 0,
                    "ops_degradation_calibration_enabled": "yes",
                    "ops_degradation_calibration_window_days": 1,
                    "ops_degradation_calibration_min_samples": 0,
                    "ops_degradation_calibration_fp_target": 1.2,
                    "ops_degradation_calibration_fn_target": -0.1,
                    "ops_degradation_calibration_step_multiplier": 0.0,
                    "ops_degradation_calibration_step_floor_ratio": 1.5,
                    "ops_degradation_calibration_step_streak_days": 0,
                    "ops_degradation_calibration_multiplier_min": 0.8,
                    "ops_degradation_calibration_multiplier_max": 0.7,
                    "ops_degradation_calibration_floor_ratio_min": 0.9,
                    "ops_degradation_calibration_floor_ratio_max": 0.8,
                    "ops_degradation_calibration_streak_min": 0,
                    "ops_degradation_calibration_streak_max": 0,
                    "ops_degradation_calibration_use_live_overrides": "yes",
                    "ops_degradation_calibration_live_params_path": "",
                    "ops_degradation_calibration_rollback_enabled": "yes",
                    "ops_degradation_calibration_rollback_window_days": 1,
                    "ops_degradation_calibration_rollback_recent_days": 2,
                    "ops_degradation_calibration_rollback_min_samples": 0,
                    "ops_degradation_calibration_rollback_fn_rise_min": 1.5,
                    "ops_degradation_calibration_rollback_gate_fail_rise_min": -0.1,
                    "ops_degradation_calibration_rollback_stable_min_samples": 0,
                    "ops_degradation_calibration_rollback_stable_fn_rate_max": 1.5,
                    "ops_degradation_calibration_rollback_stable_gate_fail_ratio_max": -0.1,
                    "ops_degradation_calibration_rollback_auto_promote_on_stable": "yes",
                    "ops_degradation_calibration_rollback_cooldown_days": -1,
                    "ops_degradation_calibration_rollback_promotion_cooldown_days": -1,
                    "ops_degradation_calibration_rollback_hysteresis_window_days": 0,
                    "ops_degradation_calibration_rollback_trigger_hysteresis_buffer": 1.5,
                    "ops_degradation_calibration_rollback_stable_hysteresis_buffer": -0.1,
                    "ops_degradation_calibration_rollback_active_snapshot_path": "",
                    "ops_degradation_calibration_rollback_history_dir": 123,
                    "release_decision_freshness_enabled": "yes",
                    "release_decision_freshness_hard_fail": "yes",
                    "release_decision_review_max_staleness_hours": 0,
                    "release_decision_gate_max_staleness_hours": 0,
                    "release_decision_eod_max_staleness_hours": 0,
                    "ops_degradation_guardrail_dashboard_enabled": "yes",
                    "ops_degradation_guardrail_dashboard_hard_fail": "yes",
                    "ops_degradation_guardrail_dashboard_window_days": 6,
                    "ops_degradation_guardrail_dashboard_min_samples": 0,
                    "ops_degradation_guardrail_dashboard_use_live_overrides": "yes",
                    "ops_degradation_guardrail_dashboard_live_params_path": "",
                    "ops_degradation_guardrail_cooldown_hit_rate_max": 1.2,
                    "ops_degradation_guardrail_suppressed_trigger_density_max": -0.1,
                    "ops_degradation_guardrail_promotion_latency_days_max": 0,
                    "ops_degradation_guardrail_false_positive_target_max": 1.5,
                    "ops_degradation_guardrail_burnin_autofill_review_if_missing": "yes",
                    "ops_degradation_guardrail_burnin_require_min_samples_for_tune": "yes",
                    "ops_degradation_guardrail_burnin_light_backfill_enabled": "yes",
                    "ops_degradation_guardrail_burnin_light_backfill_max_days_per_run": -1,
                    "ops_degradation_guardrail_burnin_review_autofill_max_days_per_run": -1,
                    "ops_degradation_guardrail_burnin_low_cost_replay_enabled": "yes",
                    "ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run": -1,
                    "ops_degradation_guardrail_burnin_use_live_overrides": "yes",
                    "ops_degradation_guardrail_burnin_live_params_path": "",
                    "ops_degradation_guardrail_burnin_budget_audit_enabled": "yes",
                    "ops_degradation_guardrail_burnin_budget_audit_auto_tune": "yes",
                    "ops_degradation_guardrail_burnin_budget_audit_expand_recovery_ratio_min": 1.2,
                    "ops_degradation_guardrail_burnin_budget_audit_shrink_recovery_ratio_max": -0.1,
                    "ops_degradation_guardrail_burnin_budget_audit_step_days": 0,
                    "ops_degradation_guardrail_burnin_budget_audit_min_days": -1,
                    "ops_degradation_guardrail_burnin_budget_audit_max_days": -1,
                    "ops_degradation_guardrail_threshold_drift_enabled": "yes",
                    "ops_degradation_guardrail_threshold_drift_gate_hard_fail": "yes",
                    "ops_degradation_guardrail_threshold_drift_require_active": "yes",
                    "ops_degradation_guardrail_threshold_drift_max_staleness_days": -1,
                    "ops_degradation_guardrail_threshold_drift_min_burnin_samples": 0,
                    "ops_degradation_guardrail_threshold_drift_autofix_enabled": "yes",
                    "ops_degradation_guardrail_threshold_drift_autofix_on_missing": "yes",
                    "ops_degradation_guardrail_threshold_drift_autofix_on_stale": "yes",
                    "ops_degradation_guardrail_threshold_drift_autofix_window_days": 0,
                    "ops_degradation_guardrail_threshold_drift_warn_ratio": -0.1,
                    "ops_degradation_guardrail_threshold_drift_critical_ratio": -0.2,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_enabled": "yes",
                    "ops_guard_loop_cadence_non_apply_rollback_lift_cooldown_days": -1,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_lookback_days": 0,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_allow_upgrade_during_cooldown": "yes",
                    "ops_guard_loop_cadence_non_apply_rollback_lift_force_heavy_hard": "yes",
                    "ops_guard_loop_cadence_non_apply_rollback_lift_light_streak_min": 3,
                    "ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min": 2,
                    "ops_guard_loop_cadence_non_apply_lift_trend_enabled": "yes",
                    "ops_guard_loop_cadence_non_apply_lift_trend_gate_hard_fail": "yes",
                    "ops_guard_loop_cadence_non_apply_lift_trend_require_active": "yes",
                    "ops_guard_loop_cadence_non_apply_lift_trend_window_days": 0,
                    "ops_guard_loop_cadence_non_apply_lift_trend_min_samples": 0,
                    "ops_guard_loop_cadence_non_apply_lift_trend_applied_rate_min": -0.1,
                    "ops_guard_loop_cadence_non_apply_lift_trend_cooldown_block_rate_max": 1.2,
                    "ops_guard_loop_cadence_non_apply_lift_trend_retention_days": 0,
                    "ops_guard_loop_cadence_non_apply_lift_trend_checksum_index_enabled": "yes",
                    "ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max": -1.2,
                    "ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max": -0.1,
                    "ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min": 1.2,
                    "ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min": 0.2,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_window_days": 0,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_retention_days": 0,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_samples": 0,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_recovery_link_rate": -0.1,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_min_retro_found_rate": 1.1,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_enabled": "yes",
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_min_samples": 0,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_step_max": -0.1,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_low": 0.8,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_high": 0.3,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_applied_gap_min": 1.2,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_cooldown_gap_min": -0.2,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_enabled": "yes",
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_apply_cooldown_days": -1,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_enabled": "yes",
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_min_delta": 1.2,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_window_days": 0,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_staleness_guard_enabled": "yes",
                    "ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_max_staleness_days": -1,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_enabled": "yes",
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_recent_days": 0,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_prior_days": 0,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_min_samples": 0,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_recovery_link_drop": -0.2,
                    "ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_retro_found_drop": 1.2,
                    "ops_guard_loop_frontend_snapshot_trend_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_gate_hard_fail": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_require_active": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_window_days": 0,
                    "ops_guard_loop_frontend_snapshot_trend_min_samples": 0,
                    "ops_guard_loop_frontend_snapshot_trend_max_failure_ratio": -0.1,
                    "ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio": 1.2,
                    "ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio": 1.3,
                    "ops_guard_loop_frontend_snapshot_trend_max_failure_streak": 0,
                    "ops_guard_loop_frontend_snapshot_trend_max_timeout_streak": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_cooldown_hours": -1.0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_window_runs": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_max_escalations": 2,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_min_timeout_streak": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio": 1.2,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs": "yes",
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs": -1,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_retention_days": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_checksum_index_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_long_days": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_min_long_samples": 0,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_suppression_ratio_delta": 1.2,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_replay_missed_ratio_delta": -0.1,
                    "ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_reason_missing_ratio_delta": 1.4,
                    "ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_manual_approval_required": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_max_apply_window_days": -1,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path": "",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_dry_run": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_rollback_guard_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_retention_days": 0,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_checksum_index_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_enabled": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_gate_hard_fail": "yes",
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_window_days": 0,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_samples": 0,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_non_apply_ratio": 1.2,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_rollback_guard_ratio": -0.1,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_apply_ratio": 1.3,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_recent_days": 0,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_prior_days": 0,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_min_samples": 0,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_non_apply_ratio_rise": 1.2,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_rollback_guard_ratio_rise": -0.1,
                    "ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_apply_ratio_drop": 1.2,
                    "ops_compaction_restore_trend_gate_hard_fail": "yes",
                    "ops_compaction_restore_trend_min_restore_required_runs": 0,
                    "ops_weekly_guardrail_enabled": "yes",
                    "ops_weekly_guardrail_weekday": 9,
                    "ops_weekly_guardrail_trigger_slot": "25:90",
                    "ops_weekly_guardrail_burnin_days": 0,
                    "ops_weekly_guardrail_drift_window_days": 6,
                    "ops_weekly_guardrail_auto_tune": "yes",
                    "ops_weekly_guardrail_run_stable_replay": "yes",
                    "ops_weekly_guardrail_require_ops_window": "yes",
                    "ops_weekly_guardrail_require_burnin_coverage": "yes",
                    "ops_weekly_guardrail_compact_enabled": "yes",
                    "ops_weekly_guardrail_compact_dry_run": "yes",
                    "ops_weekly_guardrail_compact_window_days": 6,
                    "ops_weekly_guardrail_compact_chunk_days": 0,
                    "ops_weekly_guardrail_compact_max_delete_rows": 0,
                    "ops_weekly_guardrail_compact_verify_restore": "yes",
                    "ops_weekly_guardrail_compact_verify_keep_temp_db": "yes",
                    "ops_weekly_guardrail_compact_controlled_apply_enabled": "yes",
                    "ops_weekly_guardrail_compact_controlled_apply_stability_weeks": 0,
                    "ops_weekly_guardrail_compact_controlled_apply_cadence_weeks": 0,
                    "ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows": 0,
                    "ops_weekly_guardrail_compact_controlled_apply_require_restore_verify_pass": "yes",
                    "ops_weekly_guardrail_compact_controlled_apply_require_weekly_status_ok": "yes",
                    "weekly_guardrail_history_limit": 0,
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
        self.assertIn("validation.style_attribution_lookback_days", paths)
        self.assertIn("validation.style_drift_window_days", paths)
        self.assertIn("validation.style_drift_min_sample_days", paths)
        self.assertIn("validation.style_drift_gap_max", paths)
        self.assertIn("validation.style_drift_block_on_alert", paths)
        self.assertIn("validation.style_drift_adaptive_enabled", paths)
        self.assertIn("validation.style_drift_adaptive_confidence_step_max", paths)
        self.assertIn("validation.style_drift_adaptive_trade_reduction_max", paths)
        self.assertIn("validation.style_drift_adaptive_hold_reduction_max", paths)
        self.assertIn("validation.style_drift_adaptive_trigger_ratio", paths)
        self.assertIn("validation.style_drift_adaptive_ratio_for_max", paths)
        self.assertIn("validation.style_drift_adaptive_block_ratio", paths)
        self.assertIn("validation.style_drift_gate_enabled", paths)
        self.assertIn("validation.style_drift_gate_require_active", paths)
        self.assertIn("validation.style_drift_gate_allow_alerts", paths)
        self.assertIn("validation.style_drift_gate_max_alerts", paths)
        self.assertIn("validation.style_drift_gate_max_ratio", paths)
        self.assertIn("validation.style_drift_gate_hard_fail", paths)
        self.assertIn("validation.style_drift_gate_lookback_days", paths)
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
        self.assertIn("validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_enabled", paths)
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_manual_approval_required",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_max_apply_window_days",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_approval_manifest_path",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_retention_days",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_staleness_window_days",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_path",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_drift_gate_hard_fail",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_window_stale_ratio_max",
            paths,
        )
        self.assertIn(
            "validation.ops_stress_matrix_execution_friction_trendline_controlled_apply_ledger_duplicate_block_rate_max",
            paths,
        )
        self.assertIn("validation.ops_stress_autorun_history_enabled", paths)
        self.assertIn("validation.ops_stress_autorun_history_window_days", paths)
        self.assertIn("validation.ops_stress_autorun_history_min_rounds", paths)
        self.assertIn("validation.ops_stress_autorun_history_no_trigger_min_payload_days", paths)
        self.assertIn("validation.ops_stress_autorun_history_retention_days", paths)
        self.assertIn("validation.ops_stress_autorun_history_checksum_index_enabled", paths)
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
        self.assertIn("validation.ops_stress_autorun_reason_drift_retention_days", paths)
        self.assertIn("validation.ops_stress_autorun_reason_drift_checksum_index_enabled", paths)
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
        self.assertIn("validation.ops_slot_degradation_enabled", paths)
        self.assertIn("validation.ops_slot_hysteresis_enabled", paths)
        self.assertIn("validation.ops_slot_degradation_soft_multiplier", paths)
        self.assertIn("validation.ops_slot_degradation_hard_multiplier", paths)
        self.assertIn("validation.ops_slot_hysteresis_soft_streak_days", paths)
        self.assertIn("validation.ops_slot_hysteresis_hard_streak_days", paths)
        self.assertIn("validation.ops_reconcile_window_days", paths)
        self.assertIn("validation.ops_reconcile_min_samples", paths)
        self.assertIn("validation.ops_reconcile_missing_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_plan_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_closed_count_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_closed_pnl_gap_abs_max", paths)
        self.assertIn("validation.ops_reconcile_open_gap_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_executed_dedup_monitor_enabled", paths)
        self.assertIn("validation.ops_reconcile_executed_dedup_gate_hard_fail", paths)
        self.assertIn("validation.ops_reconcile_executed_dedup_pruned_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_executed_dedup_days_ratio_max", paths)
        self.assertIn("validation.ops_reconcile_executed_dedup_restore_verify_enabled", paths)
        self.assertIn("validation.ops_reconcile_executed_dedup_restore_verify_hard_fail", paths)
        self.assertIn("validation.ops_reconcile_executed_dedup_restore_verify_max_age_days", paths)
        self.assertIn("validation.ops_reconcile_executed_dedup_restore_verify_min_backup_rows", paths)
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
        self.assertIn("validation.ops_reconcile_broker_row_diff_artifact_retention_days", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_artifact_checksum_index_enabled", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_symbol_alias_map[ ]", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_side_alias_map", paths)
        self.assertIn("validation.ops_reconcile_broker_row_diff_side_alias_map[]", paths)
        self.assertIn("validation.ops_artifact_governance_profiles[]", paths)
        self.assertIn("validation.ops_artifact_governance_profiles[bad_type]", paths)
        self.assertIn("validation.ops_artifact_governance_profiles[reconcile_row_diff].json_glob", paths)
        self.assertIn("validation.ops_artifact_governance_profiles[reconcile_row_diff].md_glob", paths)
        self.assertIn("validation.ops_artifact_governance_profiles[reconcile_row_diff].checksum_index_filename", paths)
        self.assertIn("validation.ops_artifact_governance_profiles[reconcile_row_diff].retention_days", paths)
        self.assertIn("validation.ops_artifact_governance_profiles[reconcile_row_diff].checksum_index_enabled", paths)
        self.assertIn("validation.ops_artifact_governance_strict_mode_enabled", paths)
        self.assertIn("validation.ops_artifact_governance_baseline_snapshot_enabled", paths)
        self.assertIn("validation.ops_artifact_governance_baseline_snapshot_path", paths)
        self.assertIn("validation.ops_artifact_governance_baseline_history_dir", paths)
        self.assertIn("validation.ops_artifact_governance_baseline_auto_promote_on_review_pass", paths)
        self.assertIn("validation.ops_artifact_governance_profile_baseline[]", paths)
        self.assertIn("validation.ops_artifact_governance_profile_baseline[bad_type]", paths)
        self.assertIn("validation.ops_artifact_governance_profile_baseline[reconcile_row_diff].json_glob", paths)
        self.assertIn("validation.ops_artifact_governance_profile_baseline[reconcile_row_diff].md_glob", paths)
        self.assertIn(
            "validation.ops_artifact_governance_profile_baseline[reconcile_row_diff].checksum_index_filename",
            paths,
        )
        self.assertIn(
            "validation.ops_artifact_governance_profile_baseline[reconcile_row_diff].retention_days",
            paths,
        )
        self.assertIn(
            "validation.ops_artifact_governance_profile_baseline[reconcile_row_diff].checksum_index_enabled",
            paths,
        )
        self.assertIn("validation.ops_reconcile_require_broker_snapshot", paths)
        self.assertIn("validation.ops_reconcile_broker_contract_emit_canonical_view", paths)
        self.assertIn("validation.ops_reconcile_broker_contract_canonical_dir", paths)
        self.assertIn("validation.ops_state_min_samples", paths)
        self.assertIn("validation.ops_risk_multiplier_floor", paths)
        self.assertIn("validation.ops_risk_multiplier_drift_max", paths)
        self.assertIn("validation.ops_source_confidence_floor", paths)
        self.assertIn("validation.ops_mode_health_fail_days_max", paths)
        self.assertIn("validation.ops_state_switch_rate_max_by_mode", paths)
        self.assertIn("validation.ops_state_switch_rate_max_by_mode.", paths)
        self.assertIn("validation.ops_state_switch_rate_max_by_mode.swing", paths)
        self.assertIn("validation.ops_state_degradation_enabled", paths)
        self.assertIn("validation.ops_state_hysteresis_enabled", paths)
        self.assertIn("validation.ops_state_degradation_soft_multiplier", paths)
        self.assertIn("validation.ops_state_degradation_hard_multiplier", paths)
        self.assertIn("validation.ops_state_degradation_floor_soft_ratio", paths)
        self.assertIn("validation.ops_state_degradation_floor_hard_ratio", paths)
        self.assertIn("validation.ops_state_hysteresis_soft_streak_days", paths)
        self.assertIn("validation.ops_state_hysteresis_hard_streak_days", paths)
        self.assertIn("validation.ops_degradation_calibration_enabled", paths)
        self.assertIn("validation.ops_degradation_calibration_window_days", paths)
        self.assertIn("validation.ops_degradation_calibration_min_samples", paths)
        self.assertIn("validation.ops_degradation_calibration_fp_target", paths)
        self.assertIn("validation.ops_degradation_calibration_fn_target", paths)
        self.assertIn("validation.ops_degradation_calibration_step_multiplier", paths)
        self.assertIn("validation.ops_degradation_calibration_step_floor_ratio", paths)
        self.assertIn("validation.ops_degradation_calibration_step_streak_days", paths)
        self.assertIn("validation.ops_degradation_calibration_multiplier_min", paths)
        self.assertIn("validation.ops_degradation_calibration_multiplier_max", paths)
        self.assertIn("validation.ops_degradation_calibration_floor_ratio_max", paths)
        self.assertIn("validation.ops_degradation_calibration_streak_min", paths)
        self.assertIn("validation.ops_degradation_calibration_streak_max", paths)
        self.assertIn("validation.ops_degradation_calibration_use_live_overrides", paths)
        self.assertIn("validation.ops_degradation_calibration_live_params_path", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_enabled", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_window_days", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_recent_days", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_min_samples", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_fn_rise_min", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_gate_fail_rise_min", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_stable_min_samples", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_stable_fn_rate_max", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_stable_gate_fail_ratio_max", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_auto_promote_on_stable", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_cooldown_days", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_promotion_cooldown_days", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_hysteresis_window_days", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_trigger_hysteresis_buffer", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_stable_hysteresis_buffer", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_active_snapshot_path", paths)
        self.assertIn("validation.ops_degradation_calibration_rollback_history_dir", paths)
        self.assertIn("validation.release_decision_freshness_enabled", paths)
        self.assertIn("validation.release_decision_freshness_hard_fail", paths)
        self.assertIn("validation.release_decision_review_max_staleness_hours", paths)
        self.assertIn("validation.release_decision_gate_max_staleness_hours", paths)
        self.assertIn("validation.release_decision_eod_max_staleness_hours", paths)
        self.assertIn("validation.ops_degradation_guardrail_dashboard_enabled", paths)
        self.assertIn("validation.ops_degradation_guardrail_dashboard_hard_fail", paths)
        self.assertIn("validation.ops_degradation_guardrail_dashboard_window_days", paths)
        self.assertIn("validation.ops_degradation_guardrail_dashboard_min_samples", paths)
        self.assertIn("validation.ops_degradation_guardrail_dashboard_use_live_overrides", paths)
        self.assertIn("validation.ops_degradation_guardrail_dashboard_live_params_path", paths)
        self.assertIn("validation.ops_degradation_guardrail_cooldown_hit_rate_max", paths)
        self.assertIn("validation.ops_degradation_guardrail_suppressed_trigger_density_max", paths)
        self.assertIn("validation.ops_degradation_guardrail_promotion_latency_days_max", paths)
        self.assertIn("validation.ops_degradation_guardrail_false_positive_target_max", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_autofill_review_if_missing", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_require_min_samples_for_tune", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_light_backfill_enabled", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_light_backfill_max_days_per_run", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_review_autofill_max_days_per_run", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_low_cost_replay_enabled", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_low_cost_replay_max_days_per_run", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_use_live_overrides", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_live_params_path", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_budget_audit_enabled", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_budget_audit_auto_tune", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_budget_audit_expand_recovery_ratio_min", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_budget_audit_shrink_recovery_ratio_max", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_budget_audit_step_days", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_budget_audit_min_days", paths)
        self.assertIn("validation.ops_degradation_guardrail_burnin_budget_audit_max_days", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_enabled", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_gate_hard_fail", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_require_active", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_max_staleness_days", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_min_burnin_samples", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_autofix_enabled", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_autofix_on_missing", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_autofix_on_stale", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_autofix_window_days", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_warn_ratio", paths)
        self.assertIn("validation.ops_degradation_guardrail_threshold_drift_critical_ratio", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_rollback_lift_enabled", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_rollback_lift_cooldown_days", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_rollback_lift_lookback_days", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_rollback_lift_allow_upgrade_during_cooldown", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_rollback_lift_force_heavy_hard", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_rollback_lift_heavy_streak_min", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_enabled", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_gate_hard_fail", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_require_active", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_window_days", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_min_samples", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_applied_rate_min", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_cooldown_block_rate_max", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_retention_days", paths)
        self.assertIn("validation.ops_guard_loop_cadence_non_apply_lift_trend_checksum_index_enabled", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_light_applied_delta_max", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_heavy_applied_delta_max", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_light_cooldown_delta_min", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_heavy_cooldown_delta_min", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_window_days", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_retention_days", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_min_samples", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_min_recovery_link_rate", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_min_retro_found_rate", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_enabled", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_min_samples", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_step_max", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_hit_rate_high", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_applied_gap_min", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_cooldown_gap_min", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_enabled", paths)
        self.assertIn(
            "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_apply_cooldown_days",
            paths,
        )
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_enabled", paths)
        self.assertIn(
            "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_min_delta",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_anti_flap_window_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_staleness_guard_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_cadence_lift_trend_preset_drift_auto_tune_handoff_max_staleness_days",
            paths,
        )
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_trendline_enabled", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_trendline_recent_days", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_trendline_prior_days", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_trendline_min_samples", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_recovery_link_drop", paths)
        self.assertIn("validation.ops_guard_loop_cadence_lift_trend_preset_drift_trendline_max_retro_found_drop", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_enabled", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_gate_hard_fail", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_require_active", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_window_days", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_min_samples", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_max_failure_ratio", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_max_timeout_ratio", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_max_governance_failure_ratio", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_max_failure_streak", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_max_timeout_streak", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_enabled", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_cooldown_hours", paths)
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_window_runs",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_max_escalations",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_repeat_timeout_min_timeout_streak",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_window_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_min_samples",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_suppression_ratio",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_require_zero_missed_runs",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_max_reason_missing_runs",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_retention_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_checksum_index_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_long_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_min_long_samples",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_suppression_ratio_delta",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_replay_missed_ratio_delta",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_recovery_antiflap_burnin_dual_window_max_reason_missing_ratio_delta",
            paths,
        )
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_gate_promote_on_burnin", paths)
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_enabled", paths)
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_manual_approval_required",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_max_apply_window_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_approval_manifest_path",
            paths,
        )
        self.assertIn("validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_dry_run", paths)
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_rollback_guard_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_retention_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_checksum_index_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_enabled",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_gate_hard_fail",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_window_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_samples",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_non_apply_ratio",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_max_rollback_guard_ratio",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_min_apply_ratio",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_recent_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_prior_days",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_min_samples",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_non_apply_ratio_rise",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_rollback_guard_ratio_rise",
            paths,
        )
        self.assertIn(
            "validation.ops_guard_loop_frontend_snapshot_trend_controlled_apply_route_audit_trendline_max_apply_ratio_drop",
            paths,
        )
        self.assertIn("validation.ops_compaction_restore_trend_gate_hard_fail", paths)
        self.assertIn("validation.ops_compaction_restore_trend_min_restore_required_runs", paths)
        self.assertIn("validation.ops_weekly_guardrail_enabled", paths)
        self.assertIn("validation.ops_weekly_guardrail_weekday", paths)
        self.assertIn("validation.ops_weekly_guardrail_trigger_slot", paths)
        self.assertIn("validation.ops_weekly_guardrail_burnin_days", paths)
        self.assertIn("validation.ops_weekly_guardrail_drift_window_days", paths)
        self.assertIn("validation.ops_weekly_guardrail_auto_tune", paths)
        self.assertIn("validation.ops_weekly_guardrail_run_stable_replay", paths)
        self.assertIn("validation.ops_weekly_guardrail_require_ops_window", paths)
        self.assertIn("validation.ops_weekly_guardrail_require_burnin_coverage", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_enabled", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_dry_run", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_window_days", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_chunk_days", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_max_delete_rows", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_verify_restore", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_verify_keep_temp_db", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_controlled_apply_enabled", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_controlled_apply_stability_weeks", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_controlled_apply_cadence_weeks", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_controlled_apply_delete_budget_rows", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_controlled_apply_require_restore_verify_pass", paths)
        self.assertIn("validation.ops_weekly_guardrail_compact_controlled_apply_require_weekly_status_ok", paths)
        self.assertIn("validation.weekly_guardrail_history_limit", paths)
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
