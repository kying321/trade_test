# 缺陷修正计划 | 2026-02-17 | Round 1

- 缺陷数量: `9`
- 最大回撤: `0.12308687029242216`
- 正收益窗口占比: `0.9696040075686262`
- 状态稳定性: active=True, alerts=['mode_switch_rate_high']
- 时间审计: active=True, alerts=['temporal_audit_missing_high']
- 分时异常: active=True, alerts=['slot_missing_ratio_high', 'slot_eod_quality_anomaly_high', 'slot_eod_quality_regime_bucket_anomaly_high', 'slot_eod_anomaly_high']
- 模式漂移: active=False, alerts=['mode_drift_missing_baseline:ultra_short', 'mode_drift_inactive']
- 风格漂移门禁: active=True, alerts=[], gate_ok=True
- Stress Matrix 趋势: active=False, alerts=['stress_matrix_insufficient_samples']
- 对账漂移: active=True, alerts=[]
- 回滚建议: level=soft, target=/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/artifacts/params_live_backup_2026-02-14.yaml

## 缺陷分类
- [execution] `SLOT_ANOMALY`: 固定时段执行链路出现异常聚集 | action=优先修复缺失/异常槽位数据后再恢复正常评审节奏。
- [data] `TEMPORAL_AUDIT`: 时间审计链路存在缺陷（截止日/时间戳缺失或越界） | action=先修复 manifest 时间审计字段与 strict_cutoff 标记，再恢复参数更新流程。
- [model] `STATE_MODE_SWITCH`: 模式切换率过高：60.00% > 45.00% | action=降低状态切换敏感度（提高确认窗口/滞后阈值），并复跑近窗模式稳定性。
- [data] `TEMPORAL_AUDIT_MISSING`: 时间审计字段缺失率超限：100.00% > 20.00% | action=补齐 cutoff_ts/bar_max_ts/news_max_ts/report_max_ts 后重跑 strategy-lab/research。
- [execution] `SLOT_MISSING_RATIO`: 分时槽位缺失率超限：37.50% > 35.00% | action=补齐缺失槽位日志/manifest，并核对调度器与数据写盘链路。
- [execution] `SLOT_EOD_QUALITY_ANOMALY`: 收盘主计算质量异常率超限：83.33% > 50.00% | action=优先修复 EOD 质量门禁失败根因（数据完整性/冲突仲裁）后再放开门槛。
- [model] `SLOT_EOD_QUALITY_REGIME_BUCKET`: 收盘质量异常在体制分桶中超限：breaches=1 | action=按 trend/range/extreme_vol 分桶复核 EOD 质量阈值，避免单一阈值掩盖体制差异。
- [execution] `SLOT_EOD_ANOMALY`: 收盘主计算异常率超限：83.33% > 50.00% | action=检查 EOD 主流程质量门禁与风险乘子链路，确保收盘工件稳定产出。
- [risk] `ROLLBACK_RECOMMENDED`: 系统建议参数回滚（level=soft） | action=按回滚建议执行参数回退并冻结自适应更新，完成局部回放后再恢复。

## 修正顺序
1. 优先执行回滚建议并重跑 lie gate-report --date 2026-02-17
2. 回滚后先局部回放，再执行 lie test-all 与 review-loop。
3. 优先修复 slot_anomaly 缺陷并重跑 lie ops-report --date 2026-02-17 --window-days 7
4. 补齐缺失槽位后再进行策略回测与参数更新。
5. 优先修复 state_stability 缺陷并重跑 lie ops-report --date 2026-02-17 --window-days 7
6. 状态稳定后再进行参数修正与回测收敛
7. 仅针对缺陷模块执行局部测试/回放
8. 局部通过后执行 lie test-all
9. 门槛通过后重跑 gate-report 与 ops-report
