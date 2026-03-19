# Prompt / AGENTS v2 结构草案

## 目标

这版 v2 不是为了“加更多规则”，而是为了把现有提示系统从：

- 强安全
- 强执行
- 强持续推进

升级成：

- 强安全
- 强收敛
- 强任务分型
- 强 source ownership

核心问题不是约束不够，而是当前约束存在：

1. 重复
2. 混层
3. 缺少 stop rule
4. 缺少任务模式切换

---

## v2 总原则

### 1. 把“不可破坏约束”和“默认工作习惯”分开

不可破坏约束必须短、硬、稳定。  
默认工作习惯可以演进，但不能和 live safety、deterministic time 混在一层。

### 2. 把“任务模式”从总提示中拆出来

不是所有任务都应该使用同一套思维姿态。

例如：

- 架构审查优先找环路和重复派生
- UI 任务优先复用 source artifact
- 回测任务优先防 future leak
- live 诊断优先账户 scope 和 blocker

### 3. 增加明确的 stop rule

如果没有 stop rule，模型会持续“有事可做”，但不一定持续“做最值钱的事”。

### 4. source-owned first

凡是能在 source artifact 表达的内容，不应该优先在 top brief、markdown、UI 层派生。

### 5. v2 还需要制度层，不只是行为层

v2 可以先解决“提示词太重、太散、太不收敛”的问题，但如果目标是机构级交易治理，
还必须补一层 machine-checkable policy 思维：

- 故障分级
- source-of-truth registry
- 变更分类
- 验证报告 schema
- shadow / replay / canary rollout

否则 v2 仍然更像“优秀操作员人格”，而不是“完整生产控制面”。

---

## v2 分层结构

建议把最终提示系统收成 5 层。

### Layer 0: Core Safety Guardrails

只保留真正不可破坏的硬边界。

范围：

- live capital 安全
- deterministic time
- mutex / idempotency / timeout
- panic close / panic kill
- 非破坏 git 规则

特征：

- 短
- 绝对
- 不随任务变化

不应该放进这一层的内容：

- 输出风格偏好
- UI 设计建议
- 是否多说两句
- helper 拆分习惯

这一层未来还应该映射到：

- CI policy gates
- runtime guard checks
- merge blockers

### Layer 1: Execution Defaults

这一层只定义通用默认工作方式。

范围：

- 优先 `rg`
- 编辑必须 `apply_patch`
- 先读上下文，再改代码
- 改完跑定向验证
- 避免无关扩 scope

特征：

- 稳定
- 通用
- 可适用于绝大多数任务

### Layer 2: Task Mode

这是 v2 的核心。

每个任务只能有一个 primary mode，允许 0-2 个 secondary modes。

建议保留的模式：

1. `architecture_review`
2. `source_first_implementation`
3. `ui_rendering`
4. `research_backtest`
5. `live_guard_diagnostics`
6. `default_implementation`

这一层决定：

- 任务优先级排序
- 什么算成功
- 什么情况下该停
- 什么动作是 forbidden move

这一层还应该输出：

- `recommended_change_class`
- `validation_report_requirements`
- `rollout_ladder`

### Layer 3: Domain Playbooks

领域知识不要继续塞进总提示，应该外部化成 playbook/skill/doc。

例如：

- ICT + CVD 使用边界
- Brooks price action 结构策略
- live takeover 诊断流程
- review / cross-market lane 语义

作用：

- 降低总提示长度
- 降低无关任务上下文税
- 避免所有任务都背一遍交易哲学

这里也适合挂载：

- source-of-truth registry
- failure severity matrix
- live escalation runbook

### Layer 4: Output Contract

只规定：

- 最终回复必须说明 outcome
- 是否跑了验证
- 是否碰 live
- 关键 artifact 在哪里

不要在这一层放太多排版洁癖式规则。

---

## 推荐的 v2 Prompt Stack

### A. System Prompt

保留：

- 安全边界
- 工具权限边界
- 上网/时效性规则

删除或缩短：

- 大段风格约束
- 过细的格式偏好

### B. Developer Prompt

保留：

- 编码工作流默认
- 不破坏 git
- 何时测试
- 何时汇报

新增：

- primary task mode only
- source-owned first
- stop after convergence

### C. Repo-local AGENTS v2

不再做“全量总法典”，而是做：

1. hard constraints
2. adaptive task mode index
3. domain playbook index
4. migration notes

Repo-local AGENTS 不应该继续同时承担：

- 全局安全规则
- UI 审美
- 交易哲学
- 回复风格
- 技术工作流

但 repo-local 规范应该明确承接 5 个制度层对象：

1. source-of-truth registry
2. severity matrix
3. change class taxonomy
4. validation report schema
5. rollout ladder

这会让 AGENTS 重新变成一个混合层。

---

## v2 模式定义

### 1. `architecture_review`

适用：

- 重新审查架构
- source ownership 漂移
- 环路
- lane/count/state 不一致

目标：

- 找出最高风险的结构问题
- 优先修 source-consumer 环路
- 优先修真实 artifact 已经出现的偏移

成功标准：

- 问题 concrete、file-specific
- 如果修复，source ownership 更简单
- residual risk 和 rollback point 仍然明确

停止条件：

- 找到并修掉最高风险问题后停止
- 不继续顺手做低价值格式整理

### 2. `source_first_implementation`

适用：

- 加新字段
- 加新 lane
- 加新 queue
- 加新 handoff

目标：

- 先在 source artifact 表达
- 再传给一个 consumer

成功标准：

- source 有
- 一个 consumer 正确显示
- change class 没有被无意抬到 `LIVE_EXECUTION_PATH`

停止条件：

- 不为同一个状态继续加 mirror field

### 3. `ui_rendering`

适用：

- dashboard
- panel
- visual window
- 进度面板

目标：

- 可视化现有 source
- 展示链路传导和影响范围

成功标准：

- 用户能看见当前 head / queue / blockers / transmission path

停止条件：

- 如果继续做需要 invent backend semantics，就先停

### 4. `research_backtest`

适用：

- 策略有效性
- 盈亏比
- OOS
- 比较报表

目标：

- 防 future leak
- 生成 recent comparison artifact
- 说明哪里不泛化

额外制度要求：

- 不能只停在 headline metric
- promotion 前至少明确 shadow / replay / canary 顺序
- 研究结论不得伪装成 live-ready 结论

成功标准：

- 有 report
- 有 OOS
- 有 caveat

停止条件：

- 不继续堆无关指标

### 5. `live_guard_diagnostics`

适用：

- ready-check
- probe
- remote live blockers
- 账户 scope

目标：

- 只读
- 账户作用域明确
- blocker / clearing conditions 明确

额外制度要求：

- 把 notification / metrics / dashboard 故障与 execution-path 故障分级
- 只有 execution ambiguity 才允许升级到 trading halt / panic close 语义
- 明确 account scope、venue scope、execution scope 不能混用

成功标准：

- 不把盈利误当 readiness
- 不把 scope 混掉

停止条件：

- blocker 和 clearing condition 一旦清楚就停

---

## v2 Stop Rules

这一部分必须显式进入提示，不再默认靠模型“自己懂”。

### 全局 stop rules

1. 如果连续两轮没有改变以下任一项，就停止并汇报：
   - `state`
   - `queue`
   - `gate`
   - `source ownership`
   - `user-facing capability`

2. 用户只说“继续”时，默认选择：
   - 最高风险未解决问题
   - 不是最低成本可继续拆分项

3. 如果 source artifact 已经足够表达当前事实，禁止继续新增 mirror field。

4. 如果任务已达到 mode-specific success criteria，优先停手，不继续“顺便整理”。

5. 如果 change class 将被抬到 `LIVE_EXECUTION_PATH`，而任务本来可以在 `RESEARCH_ONLY / SIM_ONLY / LIVE_GUARD_ONLY` 完成，就停止并重新分类。

6. 如果故障只属于 `P2`（通知、指标、UI），禁止把它升级成 `panic_close_all()` 风格动作。

---

## v2 要删掉或外移的内容

### 建议从总提示移除

- 大段 UI 设计偏好
- 过细的回复格式要求
- 和当前任务无关的交易哲学
- 所有技能说明的详细展开

### 建议外移到 docs / skills / playbooks

- ICT + CVD 规则
- Brooks 策略结构
- live execution 守护流程
- dashboard 设计语言

---

## v2 推荐的 AGENTS 结构

建议仓库根层 `AGENTS.md` 最终收成下面 7 节。

### 1. Purpose

- 本仓库目标
- 默认交付原则

### 2. Hard Constraints

- live safety
- deterministic time
- mutex / idempotency / timeout / panic

### 3. Coding Defaults

- `rg`
- `apply_patch`
- 定向验证
- 非破坏 git

### 4. Adaptive Task Modes

只放：

- mode 名称
- 适用场景
- selector 入口

不要把每个 mode 的完整长说明都塞进 AGENTS 主体。

### 5. Domain Playbook Index

列出：

- ICT/CVD
- Brooks
- cross-market operator
- live takeover

只给索引，不给全文。

### 6. Stop Rules

必须是独立一节。

### 7. Validation and Reporting Contract

最终回复至少包含：

- outcome
- validation
- live touched or not
- key artifacts

---

## v2 推荐的“新模块任务”执行入口

不再直接“接任务就做”，建议先过一个轻量选择器。

建议流程：

1. 读取任务摘要
2. 看改动路径
3. 选 primary mode
4. 生成：
   - hard guardrails
   - execution path
   - stop rules
   - validation plan
   - change class
   - severity lens
   - rollout ladder

这个机制已经可以由仓库内的：

- [ADAPTIVE_TASK_MODES.md](/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/ADAPTIVE_TASK_MODES.md)
- [build_adaptive_task_mode_report.py](/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_adaptive_task_mode_report.py)

来承载。

---

## 建议的 Prompt v2 模板

下面是结构草案，不是逐字固定文本。

### Block 1: Mission

```text
You are a coding agent working on Fenlie.
Optimize for source ownership, convergence, and safe execution.
```

### Block 2: Hard Guardrails

```text
- Never bypass live-capital safety gates.
- No forward-looking logic in historical indicator/backtest work.
- Prefer source-owned state over consumer-side mirrors.
- Stop after convergence; do not continue low-value cleanup by default.
```

### Block 3: Execution Defaults

```text
- Inspect context first.
- Use rg for search.
- Use apply_patch for edits.
- Run targeted validation after changes.
- Do not widen scope unless blocked.
```

### Block 4: Task Mode

```text
Primary mode: source_first_implementation
Secondary mode: ui_rendering

Mode success criteria:
- source artifact owns the new state
- one consumer shows it

Stop when:
- further work only adds mirror fields or summaries
```

### Block 5: Domain Attachment

```text
Use only the domain playbooks relevant to this task.
Do not load unrelated domain doctrine.
```

### Block 6: Output Contract

```text
Final answer must include:
- what changed
- validation status
- whether live paths were touched
- key artifact or file locations
```

建议再加：

```text
- recommended / actual change class
- residual risk
- rollback point
```

---

## 迁移建议

### Phase 1

先不改平台级 system/developer prompt，只做 repo-local v2：

- 新增 adaptive modes
- 新增 stop rules
- 新增 domain index

### Phase 2

压缩现有 AGENTS：

- 保留 hard constraints
- 删掉重复风格规则
- 把 domain 长文迁移到 docs

### Phase 3

把高频任务都接入 mode selector：

- 架构审查
- source-first 扩展
- UI 可视化
- 回测研究
- live 诊断

---

## 预期收益

如果按这个 v2 执行，预期会有 4 个直接收益：

1. 上下文税更低
2. 长链任务的逻辑漂移更少
3. “继续”时更容易选到高价值下一步
4. helper 拆分、mirror field 扩散、低价值整理会显著下降

---

## 机构级增强补丁

下面这些是 v2 最值得直接吸收的增强项。

### 1. Failure Severity Matrix

- `P0`: execution-path failure or source-of-truth ambiguity  
  允许 trading halt / panic 语义
- `P1`: process or instance failure  
  允许 recycle / kill，但不自动等价于 trading panic
- `P2`: notification, metrics, dashboard, non-trading websocket failure  
  只能 degrade + report

### 2. Source-of-Truth Registry

必须明确以下 truth owners：

- orders
- positions
- queue ownership
- gate state
- score state
- research validity / OOS artifacts

原则：

- source 可表达，就不在 consumer 再造语义
- UI 只能消费，不得暗中成为 source

### 3. Change Class Taxonomy

- `DOC_ONLY`
- `RESEARCH_ONLY`
- `SIM_ONLY`
- `LIVE_GUARD_ONLY`
- `LIVE_EXECUTION_PATH`

默认目标：

- 优先停留在最低风险 change class
- `LIVE_EXECUTION_PATH` 必须是显式升级，不是自然滑进去

### 4. Validation Report Schema

每次有意义改动后，输出至少应包含：

- changed files
- whether live path touched
- recommended / actual change class
- validations run
- validation result
- residual risk
- rollback point

### 5. Rollout Ladder

- `shadow`
- `replay`
- `canary`
- `broader rollout`

机构级系统不应只分成“已实现”和“已上线”两态。

---

## 最终判断

`Prompt / AGENTS v2` 的核心不是“更聪明地说话”，而是：

- 更少的重复约束
- 更清晰的层次
- 更强的 stop rule
- 更明确的 mode 切换
- 更硬的 source-owned first

如果这五点能稳定执行，工作效率和思维质量都会比继续堆长 prompt 更明显地提升。
