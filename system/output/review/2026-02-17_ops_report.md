# 运维状态报告 | 2026-02-17

- 状态: `red`
- 发布门槛通过: `False`
- 健康天数占比(7日): `100.00%`
- 最近测试: `0`
- 调度已执行槽位: `NONE`

## 状态稳定性
- active: `True`
- samples: `6` / min=`5`
- switch_rate: `60.00%` (count=3)
- risk_multiplier(min/avg/drift): `0.850` / `0.900` / `+0.000`
- source_confidence(min/avg): `75.68%` / `77.03%`
- mode_health_fail_days: `0`
- alerts: `mode_switch_rate_high`
- `switch_rate_ok`: `False`
- `risk_multiplier_floor_ok`: `True`
- `risk_multiplier_drift_ok`: `True`
- `source_confidence_floor_ok`: `True`
- `mode_health_fail_days_ok`: `True`

## 时间审计
- active: `True`
- samples: `3` / min=`1`
- missing/leak/strict_disabled: `100.00%` / `0.00%` / `0.00%`
- autofix(attempted/applied/failed/skipped): `3` / `0` / `0` / `3`
- autofix_artifact(written/events/applied/failed/skipped): `True` / `3` / `0` / `0` / `3`
- autofix_retention(days/rotated/rotation_failed): `30` / `0` / `False`
- autofix_checksum_index(enabled/written/entries/failed): `True` / `True` / `1` / `False`
- autofix_artifact_md: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/2026-02-17_temporal_autofix_patch.md`
- autofix_checksum_index: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/temporal_autofix_patch_checksum_index.json`
- alerts: `temporal_audit_missing_high`
- `missing_ratio_ok`: `False`
- `leak_ratio_ok`: `True`
- `strict_cutoff_ok`: `True`

## 分时异常
- active: `True`
- samples: `6` / min=`3`
- slots(expected/observed/missing): `24` / `24` / `0`
- anomaly_ratio(pre/intra/eod): `66.67%` / `66.67%` / `83.33%`
- eod_anomaly_split(quality/risk): `83.33%` / `0.00%`
- eod_regime_buckets(quality trend/range/extreme): `0.00%` / `83.33%` / `0.00%`
- eod_regime_buckets(risk trend/range/extreme): `0.00%` / `0.00%` / `0.00%`
- missing_ratio: `0.00%`
- alerts: `slot_premarket_anomaly_high, slot_intraday_anomaly_high, slot_eod_quality_anomaly_high, slot_eod_quality_regime_bucket_anomaly_high, slot_eod_anomaly_high`
- `missing_ratio_ok`: `True`
- `premarket_anomaly_ok`: `False`
- `intraday_anomaly_ok`: `False`
- `eod_quality_anomaly_ok`: `False`
- `eod_risk_anomaly_ok`: `True`
- `eod_quality_regime_bucket_ok`: `False`
- `eod_risk_regime_bucket_ok`: `True`
- `eod_anomaly_ok`: `False`

## 模式漂移
- active: `False`
- scope/compared: `1` / `0`
- runtime_mode: `ultra_short` | focus_runtime_only=`True`
- min_live_trades: `30` | window_days=`120`
- alerts: `mode_drift_missing_baseline:ultra_short, mode_drift_inactive`
- `samples_ok`: `False`
- `win_rate_gap_ok`: `True`
- `profit_factor_gap_ok`: `True`

## 风格漂移门禁
- active: `True`
- source_date: `2026-02-17` | hard_fail=`True`
- drift(score/gap/ratio): `0.01404` / `0.01000` / `1.404`
- alerts(count/monitor_failed/gate_ok): `3` / `False` / `True`
- alerts: `NONE`
- `diag_present_ok`: `True`
- `active_ok`: `True`
- `drift_ratio_ok`: `True`
- `alerts_ok`: `True`

## Stress Matrix 趋势
- active: `False`
- runs: `0` / min=`3`
- robustness(current/base/drop): `0.0000` / `0.0000` / `0.0000`
- annual_return(current/base/drop): `0.00%` / `0.00%` / `0.00%`
- worst_drawdown(current/base/rise): `0.00%` / `0.00%` / `0.00%`
- fail_ratio(current/base/rise): `0.00%` / `0.00%` / `0.00%`
- alerts: `stress_matrix_insufficient_samples`
- `robustness_drop_ok`: `True`
- `annual_return_drop_ok`: `True`
- `drawdown_rise_ok`: `True`
- `fail_ratio_ok`: `True`

## Stress Autorun 历史
- active: `False`
- rounds: `1` / min=`3`
- trigger_density/attempt_rate/run_rate: `0.00%` / `0.00%` / `0.00%`
- skips(cooldown/max_runs/runner_unavailable): `0` / `0` / `0`
- cooldown_efficiency: `0.00%`
- history_artifact(written/rounds): `True` / `1`
- history_retention(days/rotated/rotation_failed): `30` / `0` / `False`
- history_checksum_index(enabled/written/entries/failed): `True` / `True` / `1` / `False`
- history_artifact_md: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/2026-02-17_stress_autorun_history.md`
- history_checksum_index: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/stress_autorun_history_checksum_index.json`
- alerts: `stress_autorun_history_insufficient_rounds`

## Stress Autorun 自适应饱和度
- active: `False`
- rounds: `1` / min=`3`
- effective/base ratio(avg/min/max/latest): `1.000` / `1.000` / `1.000` / `1.000`
- throttle/expand ratio: `0.00%` / `0.00%`
- rounds(throttle/expand/neutral): `0` / `0` / `1`
- alerts: `stress_autorun_adaptive_insufficient_rounds`

## Stress Autorun 原因漂移
- active: `False`
- rounds: `1` / min=`6`
- reason_mix_gap/change_point_gap: `0.000` / `0.000`
- baseline_ratio(high/low/other): `0.00%` / `0.00%` / `0.00%`
- recent_ratio(high/low/other): `0.00%` / `0.00%` / `100.00%`
- alerts: `stress_autorun_reason_drift_insufficient_rounds`
- reason_drift_artifact(written/rounds/transitions/windows): `True` / `1` / `0` / `0`
- reason_drift_retention(days/rotated/rotation_failed): `30` / `0` / `False`
- reason_drift_checksum_index(enabled/written/entries/failed): `True` / `True` / `1` / `False`
- reason_drift_artifact_md: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/2026-02-17_stress_autorun_reason_drift.md`
- reason_drift_checksum_index: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/stress_autorun_reason_drift_checksum_index.json`

## Artifact Governance
- active: `True`
- profiles(total/active/override): `4` / `2` / `4`
- policy(required_missing/policy_mismatch/legacy_drift/baseline_drift): `0` / `0` / `0` / `0`
- strict_mode(enabled/blocked): `False` / `False`
- alerts: `NONE`
- `required_profiles_present_ok`: `True`
- `policy_alignment_ok`: `True`
- `legacy_alignment_ok`: `True`
- `baseline_freeze_ok`: `True`
- `strict_mode_ok`: `True`

## 对账漂移
- active: `True`
- samples: `6` / min=`3`
- breach_ratio(plan/closed/open): `0.00%` / `0.00%` / `0.00%`
- broker(missing/breach_count/breach_pnl): `16.67%` / `0.00%` / `0.00%`
- broker_contract(schema/numeric/symbol/noncanonical): `0.00%` / `0.00%` / `0.00%` / `0.00%`
- broker_canonical(eligible/written/fail): `5` / `5` / `0`
- broker_canonical_normalized(symbol/side): `0.00%` / `0.00%`
- broker_row_diff(samples/breach/key/count/notional/canonical_pref): `0` / `0.00%` / `0.00%` / `0.00%` / `0.00%` / `0.00%`
- broker_row_diff_aliases(symbol/side): `0` / `0`
- broker_row_diff_alias_drift(hit/unresolved_key/check): `0.00%` / `0.00%` / `True`
- broker_row_diff_artifact(written/sample/breach): `False` / `0` / `0`
- broker_row_diff_artifact_retention(days/rotated/rotation_failed): `30` / `0` / `False`
- broker_row_diff_artifact_checksum(enabled/written/entries/failed): `True` / `True` / `0` / `False`
- broker_row_diff_artifact_md: `N/A`
- broker_row_diff_artifact_checksum_index: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/reconcile_row_diff_checksum_index.json`
- broker_row_diff_artifact_reason: `no_row_diff_samples`
- missing_ratio: `0.00%`
- alerts: `NONE`
- `missing_ratio_ok`: `True`
- `plan_count_gap_ok`: `True`
- `closed_count_gap_ok`: `True`
- `closed_pnl_gap_ok`: `True`
- `open_count_gap_ok`: `True`
- `broker_missing_ratio_ok`: `True`
- `broker_count_gap_ok`: `True`
- `broker_pnl_gap_ok`: `True`
- `broker_contract_schema_ok`: `True`
- `broker_contract_numeric_ok`: `True`
- `broker_contract_symbol_ok`: `True`
- `broker_contract_canonical_view_ok`: `True`
- `broker_row_diff_ok`: `True`
- `broker_row_diff_alias_drift_ok`: `True`
- `broker_row_diff_artifact_rotation_ok`: `True`
- `broker_row_diff_artifact_checksum_index_ok`: `True`

## 回滚建议
- active: `True`
- level: `soft`
- score: `5`
- action: `rollback_to_last_stable_anchor_after_partial_recheck`
- anchor_ready: `True`
- target_anchor: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/artifacts/params_live_backup_2026-02-14.yaml`
- reason_codes: `state_stability, temporal_audit, slot_anomaly`

## 最近健康历史
- 2026-02-11: OK | missing=[]
- 2026-02-12: OK | missing=[]
- 2026-02-13: OK | missing=[]
- 2026-02-14: OK | missing=[]
- 2026-02-15: OK | missing=[]
- 2026-02-16: OK | missing=[]
- 2026-02-17: OK | missing=[]

## 门槛检查
- `review_pass_gate`: `True`
- `mode_health_ok`: `True`
- `state_stability_ok`: `False`
- `temporal_audit_ok`: `False`
- `slot_anomaly_ok`: `False`
- `mode_drift_ok`: `True`
- `style_drift_ok`: `True`
- `stress_matrix_trend_ok`: `True`
- `stress_autorun_history_ok`: `True`
- `stress_autorun_adaptive_ok`: `True`
- `stress_autorun_reason_drift_ok`: `True`
- `reconcile_drift_ok`: `True`
- `artifact_governance_ok`: `True`
- `tests_ok`: `True`
- `health_ok`: `True`
- `stable_replay_ok`: `True`
- `data_completeness_ok`: `True`
- `unresolved_conflict_ok`: `True`
- `positive_window_ratio_ok`: `True`
- `max_drawdown_ok`: `True`
- `risk_violations_ok`: `True`
- `rollback_anchor_ready`: `True`
