# 阶段性报告（架构增强：源置信度 + 策略学习并入复盘）

## 本阶段完成项
- 数据源置信度升级为四维证据：行情一致性/覆盖、宏观一致性/覆盖、新闻可信度、情绪覆盖。
- 策略学习实验室升级为训练/验证/截止日后复盘三段检验。
- 盘后 `review` 主链路已自动融合最近一次 `accepted=true` 的 strategy-lab 候选参数。
- `run_eod` 与 `run_backtest` 均读取 `params_live.yaml`，保证参数更新闭环落地。
- `run_review` 回测起点支持配置化（`review_backtest_start_date/review_backtest_lookback_days`），生产稳态与测试快速反馈可并存。
- 新增模式反馈工件：`output/daily/YYYY-MM-DD_mode_feedback.json`，按模式聚合历史回测表现并写入日报“模式引擎”段。
- 新增模式健康硬门禁：`run_review` 写入 `mode_health` 审计，退化时自动收敛参数并在 `gate_report` 触发 `mode_health_ok=false` 阻断通过。
- 新增测试导航优化：`lie test-all --fast` 支持确定性子样本+分片并行，输出单行 `error=...` 摘要，完整日志落盘。
- `review-loop` 第1轮已切换为 fast+full 串联：先快测筛错，再全量放行。

## 实测结果（2026-02-14）
- `lie review --date 2026-02-14` 成功读取候选 `trend_convex_01`（cutoff=2026-02-11）。
- 参数融合后写入：
  - `signal_confidence_min=59.1754`
  - `convexity_min=2.3927`
  - `hold_days=11`
  - `max_daily_trades=2`
- 审计文件包含来源与候选明细：`strategy_lab_candidate` 字段已写入 `param_delta.yaml`。

## 风控与收益统计（回测区间：2015-01-01 ~ 2026-02-14）
- 年化收益：4.29%
- 最大回撤：17.96%
- 胜率：44.40%
- 盈亏比（Profit Factor）：1.340
- 风控违规：0
- 正收益窗口占比：95.66%

## 门槛结论
- 最大回撤门槛（<=18%）：通过
- 风控违规（=0）：通过
- 模式健康门禁（mode_health_ok）：通过
- 当前闭环状态：可持续迭代（参数已具备自动学习与落地能力）

## 测试回归
- 全量自动测试：`82/82` 通过（`python -m unittest discover -s tests -p 'test_*.py' -t .`）
- 快速测试示例：`lie test-all --fast --fast-ratio 0.10`，本次执行 `8` 条用例，`error=none`。
- `lie review-loop --date 2026-02-14 --max-rounds 1` 实测：`tests_mode=fast+full`，先快测后全量，均通过。

## 本轮增量（执行层风险节流闭环）
- `run_eod` 已使用 `risk_multiplier` 参与仓位计算（`actual_size = 0.5*Kelly*confidence*risk_multiplier`）。
- `run_premarket` / `run_intraday_check` 已统一输出：
  - `runtime_mode`
  - `mode_health`
  - `risk_control`
  - `risk_multiplier`
- 新增槽位级 manifest：
  - `output/artifacts/manifests/premarket_YYYY-MM-DD.json`
  - `output/artifacts/manifests/intraday_check_YYYY-MM-DD_HHMM.json`
- 日报“模式引擎”新增执行节流展示：`risk_mult/source_mult/mode_mult/mode_reason`。
- 配置与校验已补齐：
  - `validation.execution_min_risk_multiplier`
  - `validation.source_confidence_floor_risk_multiplier`
  - `validation.mode_health_risk_multiplier`
  - `validation.mode_health_insufficient_sample_risk_multiplier`

## 本轮增量（状态稳定性与相变预警）
- `ops-report` 新增 `state_stability` 模块，输出：
  - `switch_rate`
  - `risk_multiplier_min/avg/drift`
  - `source_confidence_min/avg`
  - `mode_health_fail_days`
- 新增阈值门控（配置化）：
  - `validation.mode_switch_window_days`
  - `validation.mode_switch_max_rate`
  - `validation.ops_state_min_samples`
  - `validation.ops_risk_multiplier_floor`
  - `validation.ops_risk_multiplier_drift_max`
  - `validation.ops_source_confidence_floor`
  - `validation.ops_mode_health_fail_days_max`
- 实测 `lie ops-report --date 2026-02-14 --window-days 7` 可见：
  - `state_stability.active=false`（样本不足）
  - `alerts=["insufficient_mode_feedback_samples"]`

## 本轮增量（缺陷计划优先级联动）
- `review-loop` 的 defect plan 已接入 `state_stability`：
  - 当 `state_stability.active=true` 且阈值超限时，自动写入 `STATE_*` 缺陷码。
  - 典型缺陷码：`STATE_MODE_SWITCH`、`STATE_RISK_MULT_FLOOR`、`STATE_RISK_MULT_DRIFT`、`STATE_SOURCE_CONFIDENCE`、`STATE_MODE_HEALTH_DAYS`。
- 修正顺序已联动：
  - 若存在 `STATE_*` 缺陷，`next_actions` 首先要求修复状态稳定性并重跑 `ops-report`，随后再做参数修正/全量测试。

## 本轮增量（模式阈值自适应更新）
- `run_review` 新增 `mode_adaptive_update`：
  - 对 `signal_confidence_min / convexity_min / hold_days / max_daily_trades` 执行有界小步更新（bounded step）。
  - 依据 `mode_history` 的绩效分带（good/bad）自动判断 `expand / tighten / neutral`。
  - 与 `mode_health_guard` 串联，退化时仍由健康门禁优先收敛。
- 审计已落盘 `param_delta.yaml.mode_adaptive`，本日样本不足时状态为：
  - `applied=false`
  - `reason=insufficient_samples`
