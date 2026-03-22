# Fenlie Engine / Dashboard Snapshot 边界审计说明

状态：只读结构债说明  
日期：2026-03-21  
change class：DOC_ONLY

---

## 1. 目的

本说明只回答两件事：

1. `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/engine.py` 现在承担了哪些职责，应该怎样拆边界
2. `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py` 现在承担了哪些职责，下一步应该怎样切成更稳定的 source-to-UI 构建层

本轮**不直接重构**，只给出可执行的切分顺序，避免在当前研究/前端收口阶段引入大范围行为漂移。

---

## 2. 当前文件体量与风险

### 2.1 `engine.py`

- 路径：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/engine.py`
- 当前体量：
  - 约 **7335 行**
- 当前问题不是“单个函数太长”，而是**一个类承载了过多纵向职责**：
  - runtime / mutex / time-sync
  - data ingestion
  - mode/runtime params
  - micro capture / cross-source audit
  - paper execution / broker snapshot
  - review / stress / style drift
  - orchestrator / report / daemon entry

这意味着：

- 测试可以跑，但**认知边界已经失真**
- 任意小改动都容易跨越多个 source-of-truth 层
- 未来继续加 live guard / review / strategy lab 时，回归成本会继续升高

### 2.2 `build_dashboard_frontend_snapshot.py`

- 路径：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py`
- 当前体量：
  - 约 **682 行**
- 当前问题不是“太大到不可读”，而是**一个脚本同时扮演了 4 层角色**：
  - config contract loader
  - artifact path selector
  - payload sanitize / public-internal redaction
  - UI read model composer / writer

这意味着：

- 当前脚本还能维护，但已经开始承受“配置层、审计层、展示层”混杂的压力
- 只要再增加 2~3 个 workspace 维度，就会进一步放大 drift

---

## 3. `engine.py` 的推荐切分边界

下面不是立即改文件，而是给出**下一阶段应形成的模块边界**。

### 边界 A：Runtime Platform / Guard Primitives

建议承载内容：

- `run-halfhour-pulse` mutex
- time-sync probe / dns diagnostics / classification
- clamp / parse / date/time helper
- environment / local runtime safety primitive

当前典型段落：

- `_run_halfhour_mutex*`
- `_time_sync_*`
- `_parse_*`
- `_clamp_*`

推荐目标：

- 新模块建议：
  - `lie_engine/runtime/guards.py`
  - `lie_engine/runtime/time_sync.py`
  - `lie_engine/runtime/primitives.py`

原因：

- 这些逻辑是**平台约束**
- 不应继续和 strategy / paper execution / review loop 混在一个类里

### 边界 B：Runtime Mode / Adaptive Params

建议承载内容：

- `_mode_history_stats`
- `_mode_from_regime`
- `_resolve_runtime_params`
- `_resolved_mode_profiles`
- slot regime threshold tuning

推荐目标：

- 新模块建议：
  - `lie_engine/runtime/mode_profiles.py`
  - `lie_engine/runtime/regime_thresholds.py`

原因：

- 这是“运行参数决策层”
- 既不是 ingestion，也不是 execution
- 未来若继续加 adaptive logic，这部分最适合独立做纯函数化收口

### 边界 C：Paper Execution / Broker Snapshot

建议承载内容：

- paper positions load/save
- broker snapshot normalize/write
- settle open paper positions
- exposure snapshot

当前典型段落：

- `_normalize_live_broker_snapshot`
- `_load_open_paper_positions`
- `_save_open_paper_positions`
- `_resolve_and_write_broker_snapshot`
- `_settle_open_paper_positions`
- `_symbol_exposure_snapshot`

推荐目标：

- 新模块建议：
  - `lie_engine/execution/paper_state.py`
  - `lie_engine/execution/broker_snapshot.py`
  - `lie_engine/execution/paper_settlement.py`

原因：

- 这一段最接近执行/对账层
- 应该和 research/review/time-sync 隔离
- 后续任何 live-path 审计都会优先关注这里

### 边界 D：Micro Capture / Cross-Source Audit

建议承载内容：

- micro capture symbol resolution
- provider sampling
- cross-source quality stats
- capture persistence
- time-sync probe reuse in micro capture

推荐目标：

- 新模块建议：
  - `lie_engine/data/micro_capture_runtime.py`
  - `lie_engine/data/cross_source_audit.py`

原因：

- 这部分本质上是 data quality runtime
- 和 paper execution、review loop 不该继续混合

### 边界 E：Review / Research / Stress / Style Diagnostics

建议承载内容：

- strategy candidate merge gate
- temporal audit merge
- style drift adaptive guard
- review style diagnostics
- stress matrix / review backtest / research backtest / strategy lab

推荐目标：

- 新模块建议：
  - `lie_engine/review/review_runtime.py`
  - `lie_engine/review/style_drift_runtime.py`
  - `lie_engine/review/stress_runtime.py`

原因：

- 这些逻辑是 review-facing 编排
- 它们共同依赖 artifacts 与 thresholds，但不应再依赖执行态细节

### 边界 F：Thin Engine Facade

保留在 `engine.py` 的应当只剩：

- `LieEngine.__init__`
- 高层公开入口：
  - `run_eod`
  - `run_premarket`
  - `run_intraday_check`
  - `run_review`
  - `run_session`
  - `run_daemon`
  - `gate_report`
  - `ops_report`
  - `health_check`

目标：

- `engine.py` 变成 facade / coordinator
- 不是继续承载底层细节实现

---

## 4. `build_dashboard_frontend_snapshot.py` 的推荐切分边界

### 边界 1：Config / Contract Loader

建议承载内容：

- `load_config`
- `latest_review_suffix`
- artifact selection contract 读取
- route contract / source head contract 读取

推荐目标：

- `system/scripts/dashboard_snapshot/contracts.py`

### 边界 2：Surface Path / Redaction / Sanitize

建议承载内容：

- `path_is_sensitive`
- `sanitize_value`
- `path_locator`
- `surface_path`
- public/internal path exposure policy

推荐目标：

- `system/scripts/dashboard_snapshot/surface_policy.py`

### 边界 3：Artifact Summary Builders

建议承载内容：

- `summarize_review_payload`
- `payload_entry`
- `infer_category`
- `build_catalog`
- `build_domain_summary`
- `build_source_heads`
- `build_backtests`

推荐目标：

- `system/scripts/dashboard_snapshot/read_model_builders.py`

### 边界 4：Route/UI Read Model Builders

建议承载内容：

- `build_interface_catalog`
- `build_public_topology`
- `build_surface_contracts`
- `build_workspace_default_focus`

推荐目标：

- `system/scripts/dashboard_snapshot/ui_contract_builders.py`

### 边界 5：Writer / Main Entrypoint

建议承载内容：

- `build_surface_snapshot`
- `main`

推荐目标：

- 保留在当前脚本作为 thin CLI entry

目标形态：

- 当前脚本变成 100~180 行左右的装配器
- 其他逻辑移动到按边界命名的 helper 模块

---

## 5. 建议切分顺序

### 第 1 步：先拆 snapshot builder，再碰 engine

原因：

- 风险更低
- 边界更清楚
- 已有测试：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_dashboard_frontend_snapshot_script.py`
- 更容易通过现有前端 smoke 验证

### 第 2 步：engine 只先抽“纯辅助模块”

优先顺序建议：

1. runtime primitives / time-sync helpers
2. mode/runtime params
3. broker snapshot + paper state

原因：

- 这三块对 facade 化帮助最大
- 同时不要求立刻重写 `run_eod/run_review`

### 第 3 步：最后再抽 review/runtime orchestration

原因：

- review/stress/style drift 目前和 source-owned artifacts 绑定最深
- 如果先动这里，最容易破坏已有研究与审计链

---

## 6. 本轮不做的事

以下内容明确**不在本轮执行**：

- 不拆 `engine.py`
- 不拆 snapshot builder
- 不修改 orchestrator runtime 行为
- 不改变 public/internal snapshot schema
- 不碰 live capital / order routing / execution queue / fund scheduling

---

## 7. 下一步建议

如果继续按风险排序推进，推荐顺序：

1. **先补测试缺口**
   - `build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py`
   - `build_price_action_breakout_pullback_hold_selection_handoff_sim_only.py`
   - `build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.py`
   - `run_operator_panel_refresh.py`
2. **再拆 snapshot builder**
3. **最后才做 engine facade 化**

理由：

- 先补测试，能给后续结构债拆分提供最小护栏
- snapshot builder 比 engine 更适合先做模块化
- engine 当前过大，但它也是高风险 source-owned 中枢，不能在缺测试时直接大拆

