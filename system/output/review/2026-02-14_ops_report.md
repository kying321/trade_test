# 运维状态报告 | 2026-02-14

- 状态: `red`
- 发布门槛通过: `False`
- 健康天数占比(7日): `57.14%`
- 最近测试: `0`
- 调度已执行槽位: `NONE`

## 状态稳定性
- active: `False`
- samples: `3` / min=`5`
- switch_rate: `0.00%` (count=0)
- risk_multiplier(min/avg/drift): `0.850` / `0.900` / `+0.000`
- source_confidence(min/avg): `75.68%` / `76.98%`
- mode_health_fail_days: `0`
- alerts: `insufficient_mode_feedback_samples`
- `switch_rate_ok`: `True`
- `risk_multiplier_floor_ok`: `True`
- `risk_multiplier_drift_ok`: `True`
- `source_confidence_floor_ok`: `True`
- `mode_health_fail_days_ok`: `True`

## 分时异常
- active: `True`
- samples: `3` / min=`3`
- slots(expected/observed/missing): `12` / `9` / `3`
- anomaly_ratio(pre/intra/eod): `0.00%` / `0.00%` / `66.67%`
- eod_anomaly_split(quality/risk): `66.67%` / `0.00%`
- missing_ratio: `25.00%`
- alerts: `slot_eod_quality_anomaly_high, slot_eod_anomaly_high`
- `missing_ratio_ok`: `True`
- `premarket_anomaly_ok`: `True`
- `intraday_anomaly_ok`: `True`
- `eod_quality_anomaly_ok`: `False`
- `eod_risk_anomaly_ok`: `True`
- `eod_anomaly_ok`: `False`

## 模式漂移
- active: `False`
- scope/compared: `1` / `0`
- runtime_mode: `swing` | focus_runtime_only=`True`
- min_live_trades: `30` | window_days=`120`
- alerts: `mode_drift_insufficient_live:swing, mode_drift_inactive`
- `samples_ok`: `False`
- `win_rate_gap_ok`: `True`
- `profit_factor_gap_ok`: `True`

## 对账漂移
- active: `True`
- samples: `3` / min=`3`
- breach_ratio(plan/closed/open): `0.00%` / `0.00%` / `0.00%`
- broker(missing/breach_count/breach_pnl): `0.00%` / `0.00%` / `0.00%`
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
- active: `False`
- level: `none`
- score: `1`
- action: `no_rollback`
- anchor_ready: `True`
- target_anchor: `N/A`
- reason_codes: `slot_anomaly`

## 最近健康历史
- 2026-02-08: DEGRADED | missing=['daily_briefing', 'daily_signals', 'daily_positions']
- 2026-02-09: DEGRADED | missing=['daily_briefing', 'daily_signals', 'daily_positions']
- 2026-02-10: DEGRADED | missing=['daily_briefing', 'daily_signals', 'daily_positions']
- 2026-02-11: OK | missing=[]
- 2026-02-12: OK | missing=[]
- 2026-02-13: OK | missing=[]
- 2026-02-14: OK | missing=[]

## 门槛检查
- `review_pass_gate`: `True`
- `mode_health_ok`: `True`
- `state_stability_ok`: `True`
- `slot_anomaly_ok`: `False`
- `mode_drift_ok`: `True`
- `reconcile_drift_ok`: `True`
- `tests_ok`: `True`
- `health_ok`: `True`
- `stable_replay_ok`: `True`
- `data_completeness_ok`: `True`
- `unresolved_conflict_ok`: `True`
- `positive_window_ratio_ok`: `True`
- `max_drawdown_ok`: `True`
- `risk_violations_ok`: `True`
- `rollback_anchor_ready`: `True`
