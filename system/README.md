# 离厄反脆弱交易系统

架构复盘文档：

```bash
docs/ARCHITECTURE_REVIEW.md
docs/PROGRESS.md
docs/DEMO_VERSIONING_BASELINE.md
docs/TERM_ATOM_REGISTRY.md
docs/TIME_SYNC_RUNBOOK.md
docs/THEORY_PROCESS.md
```

运行：

```bash
python3 -m pip install -e .
lie run-eod --date 2026-02-13
lie micro-capture --date 2026-02-13 --symbols BTCUSDT,ETHUSDT
lie test-all
# 快速子样本测试（确定性）
lie test-all --fast --fast-ratio 0.10
# 单次运行超时覆盖（避免 full 模式长时间挂起）
lie test-all --timeout-seconds 600
# 并行分片（多智能体协作时可覆盖全量）
lie test-all --fast --fast-ratio 1.0 --fast-shard-index 0 --fast-shard-total 4
lie validate-config
lie architecture-audit --date 2026-02-13
lie dependency-audit --date 2026-02-13
# 数据源 profile（config.yaml: data.provider_profile）
# 可选: opensource_dual | opensource_primary | binance_spot_public | bybit_spot_public | dual_binance_bybit_public | hybrid_opensource_binance | hybrid_opensource_binance_bybit | hybrid_with_paid_placeholder | paid_placeholder
# 微观信号调制（run-eod；L2/逐笔）
# validation.microstructure_signal_enabled: true
# validation.microstructure_lookback_minutes: 10
# validation.microstructure_depth_levels: 20
# validation.microstructure_trade_limit: 500
# validation.micro_schema_hard_fuse_enabled: true
# validation.micro_schema_max_fail_symbols: 0
# validation.micro_cross_source_audit_enabled: true
# validation.micro_cross_source_build_missing_provider: true
# validation.micro_cross_source_primary: binance_spot_public
# validation.micro_cross_source_secondary: bybit_spot_public
# validation.micro_cross_source_symbols: [BTCUSDT, ETHUSDT]
# validation.micro_cross_source_min_rows_per_side: 1
# validation.micro_cross_source_adaptive_gap_enabled: true
# validation.micro_cross_source_gap_freq_multiplier: 2.0
# validation.micro_cross_source_gap_hist_window_days: 30
# validation.micro_cross_source_gap_hist_quantile: 0.90
# validation.micro_cross_source_gap_hist_multiplier: 1.10
# validation.micro_cross_source_gap_limit_cap_ms: 60000
# validation.micro_cross_source_quality_lookback_days: 7
# validation.micro_time_sync_hard_fuse_enabled: false
# validation.micro_time_sync_max_offset_ms: 5
# validation.micro_time_sync_max_rtt_ms: 120
# validation.micro_time_sync_min_samples: 1
# validation.system_time_sync_probe_enabled: false
# validation.system_time_sync_hard_fuse_enabled: false
# validation.system_time_sync_primary_source: time.google.com
# validation.system_time_sync_secondary_source: time.cloudflare.com
# validation.system_time_sync_probe_timeout_seconds: 5.0
# validation.system_time_sync_max_offset_ms: 5
# validation.system_time_sync_max_rtt_ms: 120
# validation.system_time_sync_min_ok_sources: 1
# validation.micro_capture_daemon_enabled: true
# validation.micro_capture_daemon_interval_minutes: 30
# validation.micro_capture_daemon_symbols: [BTCUSDT, ETHUSDT]
# validation.microstructure_symbols: [BTCUSDT, ETHUSDT]
# validation.micro_cross_source_hard_fuse_enabled: false
# validation.micro_cross_source_align_seconds: 120
# validation.micro_cross_source_window_ms: 200
# validation.micro_cross_source_trade_limit: 300
# validation.micro_cross_source_audit_symbol_cap: 3
# validation.micro_cross_source_min_samples: 1
# validation.micro_cross_source_max_fail_ratio: 0.50
# validation.micro_cross_source_tolerance_ms: 80
# validation.micro_continuous_gap_ms: 2500
# validation.micro_min_trade_count: 30
# validation.micro_confidence_boost_max: 8.0
# validation.micro_penalty_max: 10.0
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
# strategy-lab 候选融合门禁（只采纳对风险/收益有增益的候选）：
# validation.strategy_lab_merge_require_accepted: true
# validation.strategy_lab_merge_require_validation_metrics: true
# validation.strategy_lab_merge_min_validation_annual_return: 0.0
# validation.strategy_lab_merge_max_validation_drawdown: 0.18
# validation.strategy_lab_merge_min_validation_trades: 2
# validation.strategy_lab_merge_min_validation_positive_window_ratio: 0.55
# validation.strategy_lab_merge_min_robustness: 0.30
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
# validation.execution_crypto_stress_risk_multiplier: 0.65
# validation.execution_crypto_stress_full_scale: 1.00
# validation.execution_cross_source_stress_risk_multiplier: 0.70
# validation.execution_cross_source_stress_full_scale: 1.00
# validation.execution_micro_capture_risk_enabled: true
# validation.execution_micro_capture_risk_multiplier: 0.75
# validation.execution_micro_capture_insufficient_sample_risk_multiplier: 0.90
# validation.execution_micro_capture_lookback_days: 7
# validation.execution_micro_capture_min_runs: 4
# validation.execution_micro_capture_pass_ratio_min: 0.70
# validation.execution_micro_capture_schema_ok_ratio_min: 0.90
# validation.execution_micro_capture_time_sync_ok_ratio_min: 0.90
# validation.execution_micro_capture_cross_source_fail_ratio_max: 0.35
# 状态稳定性与相变预警（ops-report）：
# validation.mode_switch_window_days: 20
# validation.mode_switch_max_rate: 0.45
# validation.ops_state_min_samples: 5
# validation.ops_risk_multiplier_floor: 0.35
# validation.ops_risk_multiplier_drift_max: 0.30
# validation.ops_source_confidence_floor: 0.75
# validation.ops_mode_health_fail_days_max: 2
# validation.ops_micro_capture_multiplier_floor: 0.75
# validation.ops_micro_capture_degraded_days_max: 3
# validation.ops_micro_capture_insufficient_days_max: 5
# validation.ops_micro_capture_quality_fail_days_max: 3
# validation.ops_micro_capture_pass_ratio_min: 0.70
# validation.ops_micro_capture_schema_ok_ratio_min: 0.90
# validation.ops_micro_capture_time_sync_ok_ratio_min: 0.90
# validation.ops_micro_capture_cross_source_fail_ratio_max: 0.35
# validation.ops_system_time_sync_monitor_enabled: false
# validation.ops_system_time_sync_fail_days_max: 2
# validation.ops_system_time_sync_inactive_days_max: 5
# validation.ops_system_time_sync_min_ok_sources: 1
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
# 可临时覆盖（不改配置）：
# lie test-all --timeout-seconds 120
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
lie run-slot --date 2026-02-13 --slot micro-capture

# 单日全流程
lie run-session --date 2026-02-13
lie run-review-cycle --date 2026-02-13 --max-rounds 2
# run-review-cycle 默认走守护执行路径（每一步有独立超时、日志与返回码）
# 如需旧版 in-process 行为：追加 --legacy
lie run-review-cycle --date 2026-02-13 --max-rounds 2 --legacy
# 显式守护命令（与上面默认行为一致）
lie run-review-cycle-guarded --date 2026-02-13 --max-rounds 2 --review-timeout-seconds 30 --gate-timeout-seconds 70 --ops-timeout-seconds 70 --ops-window-days 14
# run-slot/run-daemon 的 review 槽默认也走 guarded（内部 skip_mutex，避免与调度锁重入）
# 如需回退旧路径：validation.scheduler_review_use_legacy: true
# 生产建议 max-rounds>=1；max-rounds=0 仅用于跳过复审循环的快速联调
# 默认 review-loop 第1轮采用 fast 子样本测试，fast 通过后同轮自动补跑 full 验证

# 守护调度（按 config.yaml 固定时段触发）
lie run-daemon --poll-seconds 30
# 当 validation.micro_capture_daemon_enabled=true 时，daemon 会按 interval_minutes 自动执行 micro-capture。
# 调度关键段使用 output/state/run-halfhour-pulse.lock 文件互斥，防止并发重放。
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
# NewAPI 路由抽样（默认使用替代名 gemini-3.1-pro-preview-bs，兼容映射 gemini-pro-3.1）
# Key 优先级：环境变量 NEWAPI_API_KEY/X666_API_KEY -> ~/.openclaw/.env -> ~/.openclaw/openclaw.json
# 默认门禁：gpt-5.4(required)，gemini-3.1-pro-preview-bs(optional)
# optional 失败会标记 gate.status=degraded 但不阻断（exit 0）；required 缺失/失败会 exit 1
# 默认隔离写入：hard-fail 时会尝试把 validation.binance_live_takeover_enabled 置为 false
# 测试/演练可加 --disable-isolation-write（仅返回失败码，不改配置）
bash system/scripts/newapi_model_probe.sh \
  --samples 2 \
  --models gpt-5.4,gemini-3.1-pro-preview-bs \
  --required-models gpt-5.4 \
  --optional-models gemini-3.1-pro-preview-bs \
  --retry-transient 1 \
  --retry-backoff-ms 300
# 可选：自定义隔离写入目标（默认 system/config.yaml）
bash system/scripts/newapi_model_probe.sh --isolation-config-path /tmp/probe_isolation.yaml --disable-isolation-write
# Binance 公共市场数据抽样（L2 + aggTrades）
python3 system/scripts/binance_micro_sample.py --symbol BTCUSDT --minutes 10 --depth 20 --limit 500
# Binance + Bybit 同窗对齐抽样（跨源 L2 / 逐笔）
python3 system/scripts/cross_exchange_micro_sample.py --symbol BTCUSDT --minutes 10 --align-seconds 120 --window-ms 200 --tolerance-ms 80 --continuous-gap-ms 2500
# 查看 EOD 自动生成的跨源质量7日周报
cat system/output/reports/cross_source/2026-03-02_quality_7d.md
# 术语原子注册校验（ICT / Al Brooks / LiE-PDF -> 可计算信号）
python3 system/scripts/validate_term_atoms.py

# 通过测试后自动提交并推送到 GitHub（需先配置 origin 远程）
# 干跑（仅预览将提交的文件）
./system/scripts/auto_git_sync.sh --dry-run
# 正式执行（默认会先跑 validate-config + test-all）
./system/scripts/auto_git_sync.sh --branch lie --message "chore(system): periodic sync"
# 若是 Pi 侧改动，使用：
./system/scripts/auto_git_sync.sh --branch pi --message "chore(system): periodic sync (pi)"
# 若请求分支被策略拒绝，可显式回退到主分支（例如 lie）：
./system/scripts/auto_git_sync.sh --branch codex/auto-sync --fallback-branch lie --message "chore(system): periodic sync"
# 若未传 --fallback-branch，脚本会自动回退到“当前允许分支（优先）/lie（兜底）”
# 可选：把 tests 日志也纳入提交
./system/scripts/auto_git_sync.sh --include-logs
# 可选：显式纳入 PROGRESS（默认不纳入，避免与后台自动化抢占）
./system/scripts/auto_git_sync.sh --include-progress
# 安装本地 pre-push 分支门禁（允许 main/pi/lie + 受控 hotfix）
./scripts/install_branch_guard_hook.sh
# 紧急分支（最多24小时）：
# hotfix/<main|pi|lie>/<ticket>/<expires_utc_yyyymmddhhmm>
# 例如：
git checkout -b hotfix/lie/INC12345/202603021200
# hotfix 提交信息必须包含 trailer：
# HOTFIX-APPROVER: <approver>
# HOTFIX-REASON: <reason>
# HOTFIX-EXPIRES: 202603021200

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
