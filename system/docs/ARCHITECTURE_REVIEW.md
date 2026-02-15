# 架构整理与复盘（2026-02-13）

## 目标
- 保持“反脆弱优先”不变量：先活下来，再追求收益。
- 将编排与策略规则解耦，减少单点复杂度。
- 在不破坏既有 CLI 契约的前提下提升可维护性。

## 本轮发现
- `src/lie_engine/engine.py` 同时承担编排、门禁规则、因子重估、审计写入，职责过载。
- 门禁逻辑（黑天鹅评分/事件窗口/冷却期）与 EOD、盘前、盘中流程耦合，难以独立验证。
- 因子边际贡献计算嵌入引擎内部，后续替换模型成本高。

## 本轮改造
1. 新增编排策略子模块：
   - `src/lie_engine/orchestration/guards.py`
   - `src/lie_engine/orchestration/factor_contrib.py`
   - `src/lie_engine/orchestration/__init__.py`
2. `LieEngine` 保持外部接口不变，但内部改为调用编排子模块：
   - 门禁评估统一走 `guards` 模块（引擎侧保留 `_evaluate_guards` 兼容封装）
   - 120 日因子重估统一走 `estimate_factor_contrib_120d(...)`
3. 保持兼容性：
   - `LieEngine._major_event_window/_loss_cooldown_active/_black_swan_assessment/_estimate_factor_contrib_120d` 仍保留（作为薄封装）
   - 既有测试中的 monkeypatch 不受破坏

## 架构边界（更新后）
- `data/*`：采集、标准化、质量门禁、存储。
- `regime/*`：体制识别（Hurst/HMM/ATR_Z）。
- `signal/*`：信号引擎与 B/S 候选点。
- `risk/*`：仓位、止损、约束执行。
- `backtest/*`：事件驱动回测与 walk-forward。
- `review/*`：参数更新与审计输出。
- `orchestration/*`：跨模块业务规则编排（门禁、因子重估等可演进逻辑）。
- `engine.py`：流程编排器，仅连接模块，不承载重规则。

## 不变量（必须长期保持）
- 冲突数据不进入模型主路径。
- 不确定体制/极端波动/重大事件窗口默认保护模式。
- 参数更新必须可追溯：变更原因、影响窗口、回滚锚点。
- 所有发布判断必须通过 `run-review-cycle` 闭环。

## 后续建议（下一轮）
1. 继续缩减 `engine.py`：将 `gate_report/ops_report/review_loop` 拆至 `orchestration/release.py`。
2. 将默认模拟数据 Provider 与真实公开源 Provider 解耦成独立 profile（dev/prod）。
3. 引入架构回归测试：校验关键模块不出现跨层反向依赖。

## 增补（2026-02-14）
1. 新增配置校验层：
   - `src/lie_engine/config/validation.py`
   - CLI 支持 `lie validate-config`
   - 引擎初始化默认强校验，避免“带错配置运行”
2. 新增运行工件清单（manifest）：
   - `src/lie_engine/reporting/manifests.py`
   - EOD/回测/复盘/研究回测统一写入 `output/artifacts/manifests/*.json`
3. 新增架构审计入口：
   - CLI 支持 `lie architecture-audit --date YYYY-MM-DD`
   - 输出 `output/review/YYYY-MM-DD_architecture_audit.{json,md}`
4. 研究回测的数据时间隔离策略已落地并固化测试：
   - 截止日前数据用于回测
   - 截止日后 `review_days` 数据仅用于复盘
5. 发布编排逻辑已从 `engine.py` 迁移到独立模块：
   - `src/lie_engine/orchestration/release.py`
   - `LieEngine.gate_report/ops_report/review_until_pass/run_review_cycle` 仅做委托
7. 架构审计逻辑已迁移到编排层：
   - `src/lie_engine/orchestration/audit.py`
   - `LieEngine.architecture_audit` 仅做委托
8. 新增分层依赖白名单测试：
   - `tests/test_layer_dependencies.py`
   - 约束 `config/data/regime/signal/risk/backtest/review/reporting/research/orchestration/engine/cli` 的可依赖层级
9. 数据源组装改为 profile 工厂：
   - `src/lie_engine/data/factory.py`
   - `engine` 不再硬编码 provider 列表，改为 `data.provider_profile` 驱动（支持 dev/prod 分离演进）
10. 新增依赖分层审计能力：
    - `src/lie_engine/orchestration/dependency.py`
    - CLI 新增 `lie dependency-audit --date YYYY-MM-DD`
11. 调度编排已迁移到独立模块：
    - `src/lie_engine/orchestration/scheduler.py`
    - `LieEngine.run_slot/run_session/run_daemon` 仅做委托
    - 新增 `tests/test_scheduler_orchestrator.py` 覆盖槽位路由、会话编排与守护状态持久化
12. 可观测性编排已独立：
    - `src/lie_engine/orchestration/observability.py`
    - `LieEngine.health_check/stable_replay_check` 仅做委托
    - 新增 `tests/test_observability_orchestrator.py` 验证健康检查与稳定回放
13. 守护调度去重语义修正：
    - `run_daemon` 从“按触发时刻 HH:MM 去重”改为“按语义槽位ID去重”
    - 解决 EOD 与 REVIEW 同时刻触发时潜在漏执行问题
    - 新增测试 `test_run_daemon_allows_same_trigger_time_for_eod_and_review`
14. 守护调度新增 dry-run：
    - CLI: `lie run-daemon --dry-run`
    - 仅输出当前时刻的槽位触发评估，不执行策略、不写入调度状态
    - 新增测试 `test_run_daemon_dry_run_does_not_execute_or_persist_state`
15. 测试执行编排已独立：
    - `src/lie_engine/orchestration/testing.py`
    - `LieEngine.test_all` 仅做委托
    - 新增 `tests/test_testing_orchestrator.py`
16. 新增架构边界守卫测试：
   - `tests/test_architecture_boundaries.py`
   - 防止 `orchestration` 反向依赖 `engine`
   - 防止发布方法重新回灌为重逻辑
17. 数据源置信度检验已落地：
    - `DataBus` 新增多源置信度评分（行情一致性、覆盖率、新闻可信度、基础可靠性、情绪因子覆盖）
    - `DataQualityReport` 增加 `source_confidence_score`、`low_confidence_source_ratio`、`source_confidence`
    - 低置信触发质量门禁标记：`SOURCE_CONFIDENCE_LOW` / `LOW_CONFIDENCE_SOURCE_RATIO_HIGH`
18. 新策略学习实验室已落地：
    - `src/lie_engine/research/strategy_lab.py`
    - CLI 新增 `lie strategy-lab --start ... --end ...`
    - 从市场结构与新闻/研报因子学习候选策略，并做训练/验证分段检验输出最优策略
19. 数据源置信度扩展为“行情+宏观+新闻+情绪”四维证据：
    - `SourceConfidenceItem` 新增 `macro_consistency/macro_coverage/macro_rows`
    - 评分与证据权重引入宏观一致性和覆盖度，避免仅靠行情/新闻侧判断源质量
20. `strategy-lab` 与 `review` 已打通：
    - `run_review` 自动读取最近已通过 (`accepted=true`) 的策略实验室工件
    - 采用小步收敛将候选参数并入 `params_live.yaml`（`signal_confidence_min/convexity_min/hold_days/max_daily_trades`）
    - `param_delta.yaml` 增加 `strategy_lab_candidate` 审计字段并记录来源 manifest
21. 回测与执行参数链路一致化：
    - `run_eod` 与 `run_backtest` 均读取 `params_live.yaml`
    - `run_event_backtest` 增加 warmup 历史窗口，保障短验证窗也能稳定计算体制
22. 复盘回测窗口支持配置化降载：
    - `run_review` 改为读取 `validation.review_backtest_start_date` / `validation.review_backtest_lookback_days`
    - 生产默认仍锚定 `2015-01-01`，测试与联调可缩短窗口以提高反馈速度
23. 模式级反馈闭环补齐：
    - `run_eod` 输出 `output/daily/YYYY-MM-DD_mode_feedback.json`
    - 聚合最近 `validation.mode_stats_lookback_days` 的 backtest manifest，按 `ultra_short/swing/long` 输出胜率、盈亏比、回撤、违规与样本数
    - 日报新增“模式引擎”段，展示当日运行模式参数与历史模式表现
24. 模式健康门禁接入复盘与发布：
    - `run_review` 基于 `mode_history + runtime_mode` 评估模式健康度，并写入 `param_delta.yaml.mode_health`
    - 模式退化时强制触发保护收敛：提高 `signal_confidence_min`、下调 `max_daily_trades`、缩短 `hold_days`
    - `gate_report` 新增 `mode_health_ok` 检查项，失败时阻断“通过状态”
25. 测试导航与快速反馈机制：
    - `lie test-all` 新增 `--fast` 确定性子样本执行，支持 `--fast-ratio` 与 `--fast-shard-index/--fast-shard-total` 并行分片
    - 返回值默认精简并提供单行 `summary_line`（含 `error=...`），完整日志写入 `output/logs/tests_*.json`
    - 发布层失败提取优先读取 `failed_tests`，避免依赖长文本 stderr 上下文
    - `review-loop` 第1轮默认执行 `fast`，若通过则同轮补跑 `full` 作为最终放行依据
