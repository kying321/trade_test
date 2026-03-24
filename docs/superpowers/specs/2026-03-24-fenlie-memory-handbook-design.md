# Fenlie 三层记忆半运行手册设计

- 日期：2026-03-24
- 模式：architecture_review
- 推荐 change class：DOC_ONLY
- 范围：只设计并整理 Fenlie 的长期记忆/手册结构；不触碰 live capital、order routing、execution queue、fund scheduling、remote execution authority

## 1. 背景

Fenlie 现有跨会话恢复高度依赖：

1. `AGENTS.md`
2. `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`
3. 各类 source artifact / handoff / review 输出

这条链已经能支撑恢复，但仍存在 4 个结构性问题：

1. **单文件膨胀**  
   `FENLIE_CODEX_MEMORY.md` 同时承担“快速恢复”“长期规则”“深层契约”三种职责，导致内容密度越来越高，新会话和子代理都容易被无关上下文污染。

2. **规则与契约混杂**  
   部分内容属于长期宪法（例如 source-owned state、live boundary、stop rule），部分内容属于领域契约（例如 canonical handoff、allowed_now / blocked_now / next_research_priority），但当前仍混在同一个记忆载体中。

3. **解释层与权威层边界不够显式**  
   项目已经形成 `source artifact -> handoff/context -> UI/summary` 的事实分层，但手册没有把这条权威链写成第一等公民，导致后续维护者容易再次在 consumer 侧重推导状态。

4. **缺少适合“人 + 主代理 + 子代理”协同的读取协议**  
   当前 memory 更像一个长文档，不像可按任务最小读取的半运行手册。

Fenlie 下一阶段目标不是只做“记忆压缩”，而是把这套记忆整理成一个**长期可生长的数字生命运行手册骨架**：它既能帮助人和代理恢复上下文，也能约束系统如何解释、裁决、反馈，但不能越权成 source-of-truth。

## 2. 目标

本轮目标是把长期记忆重构成一个**三层半运行手册**：

1. **shortlist（1 屏）**  
   让新会话、子代理、人工恢复在 30 秒内重新获得系统身份、主线、边界和优先级。

2. **durable rules（长期规则）**  
   固化不轻易改变的行为法则：authority order、live boundary、delegation boundary、validation discipline、stop rule 等。

3. **deep contracts（深层契约）**  
   把系统真正的结构智能写成领域合同：source authority、handoff contract、live boundary contract、research contract、feedback loop contract。

### 2.1 非目标

本轮不做：

- 重写或替换现有 source artifact builder
- 让 memory 文档承载实时状态
- 把 dashboard 或 operator brief 变成新的权威层
- 直接修改 live guard / execution path 的行为
- 在一次设计中把所有领域文档都铺满

## 3. 已比较方案

### 方案 A：单文件三段式

在 `FENLIE_CODEX_MEMORY.md` 内继续分：

- shortlist
- durable rules
- deep contracts

**优点**
- 改动最小
- 冷启动入口不变

**缺点**
- 会继续膨胀
- 子代理仍然容易读到过多无关上下文
- 规则/契约/导航仍然难以清晰分离

### 方案 B：总入口 + 三层分文件（推荐）

引入一个 memory tree：

- 总入口
- shortlist
- durable rules
- contracts 子目录

同时保留 `FENLIE_CODEX_MEMORY.md` 作为兼容引导层。

**优点**
- 最适合“人 + 主代理 + 子代理”分层读取
- 最接近 OpenClaw 已形成的 `source -> handoff -> consumer` 架构
- 可以持续扩展而不让单文件失控

**缺点**
- 需要一次明确拆分
- 需要写清楚每个文件的 owner 和更新规则

### 方案 C：按域拆多本手册

例如拆成：

- operator handbook
- research handbook
- live guard handbook
- dashboard handbook

**优点**
- 领域边界最强

**缺点**
- 当前阶段过重
- 冷启动时入口分散
- 可能过早引入新的碎片化

### 结论

选择 **方案 B**。

原因：

1. 它保持一个清晰入口，但允许按任务最小读取。
2. 它能兼容现有 `AGENTS.md -> FENLIE_CODEX_MEMORY.md` 的启动链。
3. 它最利于把 OpenClaw 风格的 control chain / feedback loop 抽象成 durable contract。

## 4. 核心设计原则

### 4.1 权威层级不可反转

系统长期遵循：

`source-of-truth artifacts > compact handoff/context > UI/operator summaries > chat memory`

手册只能：

- 解释系统如何思考
- 描述恢复顺序
- 约束状态与反馈的表达方式

手册不能：

- 取代 source artifact 宣告当前状态
- 让 summary/UI 越权裁决 queue、gate、position、review state

### 4.2 source-owned first

如果某个 gate、anchor、allowed action、blocked action、queue owner、source head 可以在 source artifact 中表达一次，就不应该在 consumer 侧重复推导。

### 4.3 live boundary 只解释，不越权

研究、dashboard、browser intel、review packet、feedback 学习都可以：

- 降低置信度
- 增加 veto / no-trade
- 触发 review

但不得在没有明确 contract 的情况下直接提升 live authority。

### 4.4 子代理默认最小读取

memory tree 要天然支持：

- 主代理读完整导航与相关 contract
- 子代理只读 shortlist + 单个相关 contract

而不是把整份记忆无差别广播给每个子代理。

### 4.5 设计成“半运行手册”

该手册既服务于：

- 人类恢复上下文
- 主代理裁决路径
- 子代理最小上下文加载

又不能被误认为是实时控制面板或实时 source-of-truth。

## 5. 目标目录树

建议在 `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/` 下新增：

```text
memory/
├── MEMORY_INDEX.md
├── SHORTLIST.md
├── DURABLE_RULES.md
└── contracts/
    ├── SOURCE_AUTHORITY.md
    ├── OPERATOR_HANDOFF.md
    ├── LIVE_BOUNDARY.md
    ├── RESEARCH_CONTRACTS.md
    └── FEEDBACK_LEARNING.md
```

同时保留：

`/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`

但其职责收缩为：

- bootstrap shim
- 兼容入口
- 对新目录的导航

## 6. 文件职责定义

### 6.1 `MEMORY_INDEX.md`

用途：总入口 / 文件地图 / 读取协议。

必须包含：

- authority order
- cold-start read order
- task → file mapping
- 哪些内容不应写入 memory
- 哪些文件是规则层、哪些是契约层

不应包含：

- 大段领域契约细节
- 临时 blocker
- 最新 live 状态

### 6.2 `SHORTLIST.md`

用途：1 屏恢复上下文。

必须回答：

- 这个系统是什么
- 当前 retained mainline 是什么
- 哪些边界绝不可越权
- 哪些方向已证伪或暂不值得扩展
- 冷启动顺序是什么
- “继续 / 下一步”在本项目里的含义

不应包含：

- 临时 review 结果
- 时间敏感的 ready-check
- 某次回测的具体窗口细节

### 6.3 `DURABLE_RULES.md`

用途：长期稳定规则。

必须包含：

- source-owned state rule
- live boundary rule
- delegation rule
- validation rule
- stop rule
- change class rule
- timeout / retry / idempotency / monotonic / mutex 等硬约束

### 6.4 `contracts/SOURCE_AUTHORITY.md`

用途：定义谁有权说了算。

必须包含：

- source artifact / handoff / UI summary 的分层
- canonical head / compatible head / superseded row 的定义
- baseline retained 何时仍然是 canonical source head
- consumer 不得重推导什么

### 6.5 `contracts/OPERATOR_HANDOFF.md`

用途：统一 handoff / playbook 合同。

借鉴 OpenClaw 的 `operator_handoff` / `operator_playbook`：

- `state`
- `status_triplet` / `status_quad`
- `focus_stack`
- `recommended_action`
- `next_focus_area`
- `primary_sequence`
- `sections.runtime / gate / risk_guard / canary`

要明确：

- 哪些字段是 source-owned control fields
- 哪些字段只是 human-readable brief

### 6.6 `contracts/LIVE_BOUNDARY.md`

用途：定义 live 相关的职责边界。

借鉴 OpenClaw 的 control chain：

- guardian = veto-only
- executor = transport / execution only
- canary = guarded promotion path
- ack / reconcile = execution truth
- feedback = 降权/复核输入，而非 live 授权入口

要写清：

- shadow-only / non-executable contract / guarded probe / canary / promotion 的边界
- 哪些证据可以阻止执行
- 哪些证据不足以直接允许执行

### 6.7 `contracts/RESEARCH_CONTRACTS.md`

用途：定义 Fenlie 研究主线的 canonical contract。

必须吸纳：

- `challenge_pair` 命名契约
- blocker / handoff / forward_consensus 的职责边界
- `allowed_now` / `blocked_now` / `next_research_priority`
- baseline retained 但 canonical 仍 active 的语义
- break-even sidecar / guarded review / review-only arbitration 的边界

### 6.8 `contracts/FEEDBACK_LEARNING.md`

用途：定义反馈如何回流而不越权。

借鉴 OpenClaw：

- feedback
- quality report
- shadow learning continuity
- promotion unblock readiness

需要明确：

- 什么反馈只能降权
- 什么反馈能进入 review queue
- 什么反馈能更新优先级
- 什么反馈不能直接改写 live authority

## 7. 读取协议

### 7.1 主代理默认读取

1. `AGENTS.md`
2. `FENLIE_CODEX_MEMORY.md`
3. `memory/MEMORY_INDEX.md`
4. `memory/SHORTLIST.md`
5. 按任务读取相关 contract

### 7.2 子代理默认读取

默认仅读：

- `memory/SHORTLIST.md`
- 一个任务直接相关的 contract

只有在任务明确需要长期规则时，才补读 `DURABLE_RULES.md`。

### 7.3 人类默认读取

默认顺序：

1. `SHORTLIST.md`
2. `DURABLE_RULES.md`
3. 遇到具体争议时看对应 contract

## 8. OpenClaw 借鉴点

本设计有意识借鉴 OpenClaw 已验证的 4 个结构特征：

1. **source -> handoff -> consumer** 的权威链  
2. **intent / guard / execution / reconcile / feedback** 的控制链  
3. **allowed / blocked / next action** 的显式控制字段  
4. **反馈回流但不直接越权执行** 的闭环设计  

具体对齐点：

- `operator_handoff` / `operator_playbook` → `OPERATOR_HANDOFF.md`
- `non_executable_contract` / `shadow_only` / `canary gate` → `LIVE_BOUNDARY.md`
- `canonical handoff` / `review-only arbitration` → `RESEARCH_CONTRACTS.md`
- `remote_orderflow_feedback` / `quality_report` / `shadow_learning_continuity` → `FEEDBACK_LEARNING.md`

## 9. 迁移顺序

### Phase 1：兼容引导

保留旧：

- `FENLIE_CODEX_MEMORY.md`

收缩其职责：

- 高价值摘要
- 启动顺序
- 指向新 memory tree

### Phase 2：落地基础层

创建：

- `memory/MEMORY_INDEX.md`
- `memory/SHORTLIST.md`
- `memory/DURABLE_RULES.md`

### Phase 3：落地深层契约

优先创建：

- `contracts/SOURCE_AUTHORITY.md`
- `contracts/LIVE_BOUNDARY.md`
- `contracts/RESEARCH_CONTRACTS.md`

后续补齐：

- `contracts/OPERATOR_HANDOFF.md`
- `contracts/FEEDBACK_LEARNING.md`

### Phase 4：瘦身旧 memory

当新 tree 稳定可用后，把旧 memory 固化成兼容入口，不再持续承载深层契约细节。

## 10. 验证与验收

本设计完成后，至少应满足以下验收条件：

1. 新会话能在 1 分钟内完成冷启动导航
2. 子代理不需要读取整份 memory 也能执行 bounded task
3. source artifact 与 handoff / summary 的权威边界在文档中可直接引用
4. live boundary、research contract、feedback learning 的边界不再散落在多个大文档里
5. `FENLIE_CODEX_MEMORY.md` 仍可兼容现有启动链，不造成冷启动断裂

## 11. 风险与防错

### 风险 1：新文档树再次膨胀成“伪实时状态仓库”

防错：

- `MEMORY_INDEX.md` 和 `DURABLE_RULES.md` 明确禁止写入临时 blocker / 最新 ready-check / 一次性回测结果

### 风险 2：旧 memory 与新 memory tree 双写漂移

防错：

- 旧 memory 只保留引导摘要
- 深层契约只写入 `contracts/`

### 风险 3：子代理仍被过量上下文污染

防错：

- 明确规定子代理默认只读 `SHORTLIST + 单个相关 contract`

### 风险 4：手册被误解为 live authority

防错：

- 每层文档都强调：handbook explains, source artifacts decide

## 12. 结论

Fenlie 下一阶段的长期记忆不应继续停留在“单个长文件”，而应升级为一个**三层半运行手册**：

- `SHORTLIST` 负责快速恢复
- `DURABLE_RULES` 负责稳定约束
- `DEEP_CONTRACTS` 负责系统逻辑与反馈闭环

同时保留 `FENLIE_CODEX_MEMORY.md` 作为兼容入口，以最小风险完成迁移。

这套结构既服务于人，也服务于主代理和子代理，并且与 OpenClaw 已验证的控制链架构保持一致，但不把解释层提升为 source-of-truth。
