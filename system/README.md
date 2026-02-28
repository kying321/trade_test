# 离厄反脆弱交易系统

架构复盘文档：

```bash
docs/ARCHITECTURE_REVIEW.md
docs/PROGRESS.md
docs/DEMO_VERSIONING_BASELINE.md
```

运行：

```bash
python3 -m pip install -e .
lie run-eod --date 2026-02-13
lie test-all
# 快速子样本测试（确定性）
lie test-all --fast --fast-ratio 0.10
# 并行分片（多智能体协作时可覆盖全量）
lie test-all --fast --fast-ratio 1.0 --fast-shard-index 0 --fast-shard-total 4
lie validate-config
lie architecture-audit --date 2026-02-13
lie dependency-audit --date 2026-02-13
# 数据源 profile（config.yaml: data.provider_profile）
# 可选: opensource_dual | opensource_primary | hybrid_with_paid_placeholder | paid_placeholder
# 10小时预算的实盘多模式研究回测（新闻+研报因子）
lie research-backtest --start 2015-01-01 --end 2026-02-13 --hours 10 --max-symbols 120 --max-trials-per-mode 500 --review-days 5
# 单模式定向诊断（可选：ultra_short,swing,long）
lie research-backtest --start 2015-01-01 --end 2026-02-13 --hours 2 --modes swing
# 市场+研报学习新策略并做训练/验证检验
lie strategy-lab --start 2015-01-01 --end 2026-02-13 --candidate-count 12 --review-days 5
# 盘后复盘：会自动尝试融合最近一次已通过(accepted=true)的 strategy-lab 候选参数
lie review --date 2026-02-13
# strategy/research manifest 会写入时间审计字段：
# metadata.cutoff_ts / bar_max_ts / news_max_ts / report_max_ts
# review 加载 strategy-lab 候选时会校验上述字段不越过 cutoff（防前视）
# 评审回测窗口可配置：
# validation.review_backtest_start_date: "2015-01-01"（默认）
# validation.review_backtest_lookback_days: 540（可选，优先于 start_date，适合测试/快速迭代）
# 模式反馈窗口：
# validation.mode_stats_lookback_days: 365（按历史 backtest manifest 聚合超短/中短/长线模式统计）
# 模式健康门禁（用于 review/gate）：
# validation.mode_health_min_samples: 1
# validation.mode_health_min_profit_factor: 1.0
# validation.mode_health_min_win_rate: 0.40
# validation.mode_health_max_drawdown_max: 0.18
# validation.mode_health_max_violations: 0
# 模式阈值自适应（review 小步更新）：
# validation.mode_adaptive_update_enabled: true
# validation.mode_adaptive_update_min_samples: 3
# validation.mode_adaptive_update_step: 0.08
# validation.mode_adaptive_good_profit_factor: 1.25
# validation.mode_adaptive_bad_profit_factor: 1.00
# validation.mode_adaptive_good_win_rate: 0.50
# validation.mode_adaptive_bad_win_rate: 0.42
# validation.mode_adaptive_good_drawdown_max: 0.12
# validation.mode_adaptive_bad_drawdown_max: 0.18
# 执行层动态风险节流（source_confidence + mode_health 联合控制）：
# validation.execution_min_risk_multiplier: 0.20
# validation.source_confidence_floor_risk_multiplier: 0.35
# validation.mode_health_risk_multiplier: 0.50
# validation.mode_health_insufficient_sample_risk_multiplier: 0.85
# 状态稳定性与相变预警（ops-report）：
# validation.mode_switch_window_days: 20
# validation.mode_switch_max_rate: 0.45
# validation.ops_state_min_samples: 5
# validation.ops_risk_multiplier_floor: 0.35
# validation.ops_risk_multiplier_drift_max: 0.30
# validation.ops_source_confidence_floor: 0.75
# validation.ops_mode_health_fail_days_max: 2
# 实盘/回测模式漂移门控（gate/ops/defect-plan）：
# validation.mode_drift_window_days: 120
# validation.mode_drift_min_live_trades: 30
# validation.mode_drift_win_rate_max_gap: 0.12
# validation.mode_drift_profit_factor_max_gap: 0.40
# validation.mode_drift_focus_runtime_mode_only: true
# 时间审计门控（strategy-lab/research manifest）：
# validation.ops_temporal_audit_enabled: true
# validation.ops_temporal_audit_lookback_days: 45
# validation.ops_temporal_audit_min_samples: 1
# validation.ops_temporal_audit_missing_ratio_max: 0.20
# validation.ops_temporal_audit_leak_ratio_max: 0.00
# validation.ops_temporal_audit_autofix_enabled: true
# validation.ops_temporal_audit_autofix_max_writes: 3
# validation.ops_temporal_audit_autofix_fix_strict_cutoff: true
# validation.ops_temporal_audit_autofix_require_safe: true
# validation.ops_temporal_audit_autofix_patch_retention_days: 30
# validation.ops_temporal_audit_autofix_patch_checksum_index_enabled: true
# stress-matrix 趋势门控（当前 run 与历史 run 对比）：
# validation.ops_stress_matrix_trend_enabled: true
# validation.ops_stress_matrix_trend_window_runs: 8
# validation.ops_stress_matrix_trend_min_runs: 3
# validation.ops_stress_matrix_robustness_drop_max: 0.15
# validation.ops_stress_matrix_annual_return_drop_max: 0.08
# validation.ops_stress_matrix_drawdown_rise_max: 0.08
# validation.ops_stress_matrix_fail_ratio_max: 0.50
# validation.ops_stress_autorun_history_enabled: true
# validation.ops_stress_autorun_history_window_days: 30
# validation.ops_stress_autorun_history_min_rounds: 3
# validation.ops_stress_autorun_adaptive_monitor_enabled: true
# validation.ops_stress_autorun_adaptive_monitor_window_days: 30
# validation.ops_stress_autorun_adaptive_monitor_min_rounds: 3
# validation.ops_stress_autorun_adaptive_effective_base_ratio_floor: 0.50
# validation.ops_stress_autorun_adaptive_effective_base_ratio_ceiling: 2.00
# validation.ops_stress_autorun_adaptive_throttle_ratio_max: 0.85
# validation.ops_stress_autorun_adaptive_expand_ratio_max: 0.85
# validation.ops_stress_autorun_reason_drift_enabled: true
# validation.ops_stress_autorun_reason_drift_window_days: 30
# validation.ops_stress_autorun_reason_drift_min_rounds: 6
# validation.ops_stress_autorun_reason_drift_recent_rounds: 4
# validation.ops_stress_autorun_reason_drift_mix_gap_max: 0.35
# validation.ops_stress_autorun_reason_drift_change_point_gap_max: 0.45
# 分时槽位异常门控（premarket/intraday/eod）：
# validation.ops_slot_window_days: 7
# validation.ops_slot_min_samples: 3
# validation.ops_slot_missing_ratio_max: 0.35
# validation.ops_slot_premarket_anomaly_ratio_max: 0.50
# validation.ops_slot_intraday_anomaly_ratio_max: 0.50
# validation.ops_slot_eod_anomaly_ratio_max: 0.50
# validation.ops_slot_eod_quality_anomaly_ratio_max: 0.50
# validation.ops_slot_eod_risk_anomaly_ratio_max: 0.50
# validation.ops_slot_eod_quality_anomaly_ratio_max_by_regime:
#   trend: 0.55
#   range: 0.50
#   extreme_vol: 0.35
# validation.ops_slot_eod_risk_anomaly_ratio_max_by_regime:
#   trend: 0.55
#   range: 0.50
#   extreme_vol: 0.30
# validation.ops_slot_use_live_regime_thresholds: true
# validation.ops_slot_regime_tune_enabled: true
# validation.ops_slot_regime_tune_window_days: 180
# validation.ops_slot_regime_tune_min_days: 20
# validation.ops_slot_regime_tune_step: 0.12
# validation.ops_slot_regime_tune_buffer: 0.08
# validation.ops_slot_regime_tune_floor: 0.10
# validation.ops_slot_regime_tune_ceiling: 0.80
# validation.ops_slot_regime_tune_missing_ratio_hard_cap: 0.80
# validation.ops_slot_source_confidence_floor: 0.75
# validation.ops_slot_risk_multiplier_floor: 0.20
# 执行链路对账漂移门控（manifest / daily csv / sqlite / open-state）：
# validation.ops_reconcile_window_days: 7
# validation.ops_reconcile_min_samples: 3
# validation.ops_reconcile_missing_ratio_max: 0.35
# validation.ops_reconcile_plan_gap_ratio_max: 0.10
# validation.ops_reconcile_closed_count_gap_ratio_max: 0.10
# validation.ops_reconcile_closed_pnl_gap_abs_max: 0.001
# validation.ops_reconcile_open_gap_ratio_max: 0.25
# validation.ops_reconcile_broker_missing_ratio_max: 0.50
# validation.ops_reconcile_broker_gap_ratio_max: 0.10
# validation.ops_reconcile_broker_pnl_gap_abs_max: 0.001
# validation.ops_reconcile_broker_contract_schema_invalid_ratio_max: 0.10
# validation.ops_reconcile_broker_contract_numeric_invalid_ratio_max: 0.10
# validation.ops_reconcile_broker_contract_symbol_invalid_ratio_max: 0.10
# validation.ops_reconcile_broker_contract_symbol_noncanonical_ratio_max: 0.40
# validation.ops_reconcile_broker_closed_pnl_abs_hard_max: 1e9
# validation.ops_reconcile_broker_position_qty_abs_hard_max: 1e9
# validation.ops_reconcile_broker_position_notional_abs_hard_max: 1e10
# validation.ops_reconcile_broker_price_abs_hard_max: 1e8
# validation.ops_reconcile_require_broker_snapshot: false
# validation.ops_reconcile_broker_contract_emit_canonical_view: true
# validation.ops_reconcile_broker_contract_canonical_dir: artifacts/broker_snapshot_canonical
# validation.ops_reconcile_broker_row_diff_min_samples: 1
# validation.ops_reconcile_broker_row_diff_breach_ratio_max: 0.20
# validation.ops_reconcile_broker_row_diff_key_mismatch_max: 0.25
# validation.ops_reconcile_broker_row_diff_count_gap_max: 0.25
# validation.ops_reconcile_broker_row_diff_notional_gap_max: 0.50
# validation.ops_reconcile_broker_row_diff_alias_monitor_enabled: true
# validation.ops_reconcile_broker_row_diff_alias_hit_rate_min: 0.00
# validation.ops_reconcile_broker_row_diff_unresolved_key_ratio_max: 0.30
# validation.ops_reconcile_broker_row_diff_asof_only: true
# validation.ops_reconcile_broker_row_diff_symbol_alias_map: {}
# validation.ops_reconcile_broker_row_diff_side_alias_map: {}
# broker snapshot writer source:
# validation.broker_snapshot_source_mode: paper_engine | live_adapter | hybrid_prefer_live
# validation.broker_snapshot_live_mapping_profile: generic | ibkr | binance | ctp
# validation.broker_snapshot_live_inbox: output/artifacts/broker_live_inbox
# validation.broker_snapshot_live_fallback_to_paper: true
# optional custom mapping override:
# validation.broker_snapshot_live_mapping_fields:
#   positions: [positions]
#   position_fields:
#     symbol: [symbol]
#     qty: [positionAmt]
# broker 快照路径约定（EOD 默认写入 paper_engine；live 可覆写）：
# output/artifacts/broker_snapshot/YYYY-MM-DD.json
# 字段建议：open_positions 或 positions[], closed_pnl
# test timeout guard:
# validation.test_all_timeout_seconds: 1800
# review-loop timeout fallback（full 超时后自动降级到 deterministic fast shard）:
# validation.review_loop_timeout_fallback_enabled: true
# validation.review_loop_timeout_fallback_ratio: 0.08
# validation.review_loop_timeout_fallback_shard_index: 0
# validation.review_loop_timeout_fallback_shard_total: 1
# validation.review_loop_timeout_fallback_seed: lie-timeout-v1
# review-loop stress matrix auto-run（当 mode_drift / stress_trend 失稳时触发）:
# validation.review_loop_stress_matrix_autorun_enabled: true
# validation.review_loop_stress_matrix_autorun_on_mode_drift: true
# validation.review_loop_stress_matrix_autorun_on_stress_breach: true
# validation.review_loop_stress_matrix_autorun_max_runs: 1
# validation.review_loop_stress_matrix_autorun_cooldown_rounds: 1
# validation.review_loop_stress_matrix_autorun_backoff_multiplier: 2.0
# validation.review_loop_stress_matrix_autorun_backoff_max_rounds: 8
# validation.review_loop_stress_matrix_autorun_adaptive_enabled: true
# validation.review_loop_stress_matrix_autorun_adaptive_window_days: 30
# validation.review_loop_stress_matrix_autorun_adaptive_min_rounds: 6
# validation.review_loop_stress_matrix_autorun_adaptive_low_density_threshold: 0.20
# validation.review_loop_stress_matrix_autorun_adaptive_high_density_threshold: 0.60
# validation.review_loop_stress_matrix_autorun_adaptive_low_density_factor: 1.50
# validation.review_loop_stress_matrix_autorun_adaptive_high_density_factor: 0.50
# validation.review_loop_stress_matrix_autorun_adaptive_min_runs_floor: 0
# validation.review_loop_stress_matrix_autorun_adaptive_max_runs_cap: 3
# validation.review_loop_stress_matrix_autorun_modes: [ultra_short, swing, long]
# 自动回滚建议（gate/ops/defect-plan 联动）：
# rollback_recommendation.level: none | soft | hard
# rollback_recommendation.target_anchor: params_live_backup_YYYY-MM-DD.yaml
```

调度：

```bash
# 单槽位执行
lie run-slot --date 2026-02-13 --slot 08:40
lie run-slot --date 2026-02-13 --slot 15:10
lie run-slot --date 2026-02-13 --slot ops

# 单日全流程
lie run-session --date 2026-02-13
lie run-review-cycle --date 2026-02-13 --max-rounds 2
# 生产建议 max-rounds>=1；max-rounds=0 仅用于跳过复审循环的快速联调
# 默认 review-loop 第1轮采用 fast 子样本测试，fast 通过后同轮自动补跑 full 验证

# 守护调度（按 config.yaml 固定时段触发）
lie run-daemon --poll-seconds 30
# 仅预演当前时刻会触发哪些槽位（不执行，不写状态）
lie run-daemon --dry-run

# 本地 cron 安装/卸载
./infra/local/install_cron.sh
./infra/local/uninstall_cron.sh

# 健康检查与失败重试
lie health-check --date 2026-02-13
lie stable-replay --date 2026-02-13 --days 3
lie gate-report --date 2026-02-13 --run-tests
lie ops-report --date 2026-02-13 --window-days 7
lie stress-matrix --date 2026-02-13 --modes ultra_short,swing,long
./infra/local/healthcheck.sh 2026-02-13
./infra/local/retry_slot.sh 2026-02-13 15:10

# 通过测试后自动提交并推送到 GitHub（需先配置 origin 远程）
# 干跑（仅预览将提交的文件）
./scripts/auto_git_sync.sh --dry-run
# 正式执行（默认会先跑 validate-config + test-all）
./scripts/auto_git_sync.sh --branch codex/auto-sync --message "chore(system): periodic sync"
# 可选：把 tests 日志也纳入提交
./scripts/auto_git_sync.sh --include-logs

# 审查未通过时会自动生成：
# output/review/YYYY-MM-DD_defect_plan_roundN.json
# output/review/YYYY-MM-DD_defect_plan_roundN.md
# broker 行级对账偏差超限时会生成：
# output/review/YYYY-MM-DD_reconcile_row_diff.json
# output/review/YYYY-MM-DD_reconcile_row_diff.md
# stress autorun 历史触发分析工件：
# output/review/YYYY-MM-DD_stress_autorun_history.json
# output/review/YYYY-MM-DD_stress_autorun_history.md
# temporal autofix 审计补丁会生成：
# output/review/YYYY-MM-DD_temporal_autofix_patch.json
# output/review/YYYY-MM-DD_temporal_autofix_patch.md
# output/review/temporal_autofix_patch_checksum_index.json
# EOD 同时会维护 paper execution 状态：
# output/artifacts/paper_positions_open.json   # 当前未平仓 paper 持仓
# sqlite: executed_plans                        # 触发止损/止盈/时间止损后的已平仓记录
```
