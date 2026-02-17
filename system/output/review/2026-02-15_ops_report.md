# 运维状态报告 | 2026-02-15

- 状态: `red`
- 发布门槛通过: `False`
- 健康天数占比(7日): `57.14%`
- 最近测试: `0`
- 调度已执行槽位: `NONE`

## 状态稳定性
- active: `False`
- samples: `4` / min=`5`
- switch_rate: `33.33%` (count=1)
- risk_multiplier(min/avg/drift): `0.850` / `0.887` / `+0.075`
- source_confidence(min/avg): `75.68%` / `77.00%`
- mode_health_fail_days: `0`
- alerts: `insufficient_mode_feedback_samples`
- `switch_rate_ok`: `True`
- `risk_multiplier_floor_ok`: `True`
- `risk_multiplier_drift_ok`: `True`
- `source_confidence_floor_ok`: `True`
- `mode_health_fail_days_ok`: `True`

## 分时异常
- active: `True`
- samples: `4` / min=`3`
- slots(expected/observed/missing): `16` / `10` / `6`
- anomaly_ratio(pre/intra/eod): `0.00%` / `0.00%` / `75.00%`
- eod_anomaly_split(quality/risk): `75.00%` / `0.00%`
- eod_regime_buckets(quality trend/range/extreme): `0.00%` / `75.00%` / `0.00%`
- eod_regime_buckets(risk trend/range/extreme): `0.00%` / `0.00%` / `0.00%`
- missing_ratio: `37.50%`
- alerts: `slot_missing_ratio_high, slot_eod_quality_anomaly_high, slot_eod_quality_regime_bucket_anomaly_high, slot_eod_anomaly_high`
- `missing_ratio_ok`: `False`
- `premarket_anomaly_ok`: `True`
- `intraday_anomaly_ok`: `True`
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

## 对账漂移
- active: `True`
- samples: `4` / min=`3`
- breach_ratio(plan/closed/open): `0.00%` / `0.00%` / `0.00%`
- broker(missing/breach_count/breach_pnl): `25.00%` / `0.00%` / `0.00%`
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

## 回滚建议
- active: `True`
- level: `hard`
- score: `13`
- action: `rollback_now_and_lock_parameter_updates`
- anchor_ready: `True`
- target_anchor: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/artifacts/params_live_backup_2026-02-14.yaml`
- reason_codes: `risk_violations, max_drawdown, stable_replay, health_degraded, slot_anomaly, review_gate`

## 最近健康历史
- 2026-02-09: DEGRADED | missing=['daily_briefing', 'daily_signals', 'daily_positions']
- 2026-02-10: DEGRADED | missing=['daily_briefing', 'daily_signals', 'daily_positions']
- 2026-02-11: OK | missing=[]
- 2026-02-12: OK | missing=[]
- 2026-02-13: OK | missing=[]
- 2026-02-14: OK | missing=[]
- 2026-02-15: DEGRADED | missing=['review_report', 'review_delta']

## 门槛检查
- `review_pass_gate`: `False`
- `mode_health_ok`: `True`
- `state_stability_ok`: `True`
- `slot_anomaly_ok`: `False`
- `mode_drift_ok`: `True`
- `reconcile_drift_ok`: `True`
- `tests_ok`: `True`
- `health_ok`: `False`
- `stable_replay_ok`: `False`
- `data_completeness_ok`: `False`
- `unresolved_conflict_ok`: `False`
- `positive_window_ratio_ok`: `False`
- `max_drawdown_ok`: `False`
- `risk_violations_ok`: `False`
- `rollback_anchor_ready`: `True`
