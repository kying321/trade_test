# Fenlie Ops / Workspace 双域上下文条拆分设计

- 日期：2026-03-21
- 模式：ui_rendering
- 推荐 change class：RESEARCH_ONLY
- 范围：只重构前端只读信息架构与交互分层；不触碰 live capital、execution queue、order routing、fund scheduling

## 1. 背景

Fenlie 当前前端已经完成：

1. 公开 / 内部 surface 分层
2. 长页 TOC 与 `page_section` 深链
3. 研究工作区左侧四控件拆分
4. 主题三态、长文本折叠、导航紧凑化

但 `OpsContextStrip` 与 `WorkspaceContextStrip` 仍然存在明显“同构残留”：

- 两者都在顶部重复输出“当前页 / 焦点 / 动作”
- 操作终端仍暴露研究型动作（如 `查看工件池`）
- 研究工作区仍沿用过于“运行态化”的摘要结构
- 用户进入不同域后，难以快速判断自己处于“运行调度域”还是“研究验证域”

问题的本质不是文案没翻译，而是**域职责没有切干净**。

## 2. 目标

本轮目标是把顶部上下文层彻底做成**双域纯化结构**：

### 2.1 操作终端

只表达：

- 视图断言
- 运行态
- 门禁 / 告警
- 运行域动作

### 2.2 研究工作区

只表达：

- 研究对象
- 验证窗口
- 证据链
- 研究域动作

### 2.3 非目标

本轮不做：

- 新后端字段
- snapshot schema 变化
- 新增 live 相关控制按钮
- 大规模页面重排
- 重写 `TerminalPanels.tsx` / `WorkspacePanels.tsx` 的主体内容

## 3. 已确认方案：A / 域纯化拆分

### 3.1 为什么选 A

对比备选：

- B（共用骨架 + slot 替换）改动小，但会保留“本质上同一个 strip 换字样”的问题
- C（顶部极简，信息下沉）会把重复从 strip 挪到首卡，不能解决职责边界

所以选 A：

- `OpsContextStrip` 和 `WorkspaceContextStrip` 继续保留两个独立组件
- 但不再共享同一种信息编排范式
- 字段、动作、badge、视觉层次分开建模

## 4. 设计原则

1. **Source-owned first**：不在 UI 顶层重推导 lane / gate / decision / review，只消费 read model。
2. **域纯化**：运行态不混入研究动作；研究态不混入运行门禁。
3. **首屏即分流**：用户进入页面后，1 秒内能判断当前属于哪个域。
4. **动作有域归属**：任何 CTA 都必须回答“这是运行动作还是研究动作”。
5. **保守改造**：优先改 strip / CTA / 首卡摘要，不碰底层 panels 主体。

## 5. 运行域设计（OpsContextStrip）

### 5.1 头部 copy

- kicker：`运行域 / 视图断言`
- title：`公开终端 / 内部终端` 或 `请求视图 / 实际视图`
- subtitle：`只读模式 / redaction / snapshot surface`

### 5.2 三卡结构

保留 3 张卡，但改职责：

1. **视图断言**
   - requested surface
   - effective surface
   - visibility / redaction

2. **运行态**
   - snapshot source
   - freshness time
   - mode / availability

3. **门禁与告警**
   - top alert
   - fallback chain 或 guardian 相关摘要
   - 不再用“穿透焦点”占一整卡

### 5.3 动作区

只允许运行域动作：

- 清除行焦点（若存在）
- 回总览
- 回终端主层
- 查看运维面板

明确移除：

- `查看工件池`
- 任何 `查看原始层 / 回工件池` 型研究动作

### 5.4 可选的穿透焦点

`panel / section / row` 焦点仍保留，但从三大主卡中下沉为：

- 一个小 badge / subline
- 或动作区上方的次级 meta

它不再作为运行域首要信息。

## 6. 研究域设计（WorkspaceContextStrip）

### 6.1 头部 copy

按 `buildWorkspacePageMeta(section)` 继续输出阶段专属文案，但 strip 本身改成研究链路头：

- artifacts：`工件 / 当前焦点`
- backtests：`回测 / 验证窗口`
- contracts：`契约 / 验收记录`
- raw：`原始 / 最终回退`

### 6.2 三卡结构

1. **研究对象**
   - 当前工件标题
   - decision badge
   - canonical / archive / group

2. **验证窗口**
   - 当前阶段 / page section
   - search scope / tone / group 的上下文说明

3. **证据链**
   - payload key / artifact id
   - path
   - handoff or primary focus hint

### 6.3 动作区

只允许研究域动作：

- 查看原始层
- 回工件池
- 查看回测池 / 契约层（按当前阶段决定）
- 返回终端焦点（仅当 source-owned handoff 存在）

不再出现：

- guardian / canary / promotion / fallback 这类运行域动作

## 7. 页面首卡差异化

strip 拆完后，还要同步压低首屏重复：

### 7.1 操作终端首卡

首卡更像“运行摘要”：

- freshness
- active alerts
- gate summary
- control-chain head

### 7.2 研究工作区首卡

首卡更像“研究摘要”：

- 当前主线
- 当前 artifact
- current validation window
- next research action

这样 strip 负责上下文，首卡负责本页主叙事，不再复写同一句话。

## 8. 视觉与交互

结合 stylekit 的建议，本轮不换大风格，只做**Corporate Clean + Terminal Aesthetic** 的域差异：

### 8.1 操作终端

- 冷蓝 / 冷灰 accent
- badge 更强调 `warning / negative`
- CTA 更偏控制台按钮

### 8.2 研究工作区

- 石墨 / 深绿 / amber accent
- 更强调 group / decision / evidence
- CTA 更偏检索与跳转

### 8.3 一致性要求

- 继续使用现有 token，不新增视觉噪声
- 继续支持 light / dark / system
- 长文本继续走 `ClampText`

## 9. 文件边界

本轮优先修改：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.tsx`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/index.css`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx`

不优先修改：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/components/TerminalPanels.tsx`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/components/WorkspacePanels.tsx`

除非为减少首屏重复必须做最小文案收口。

## 10. 测试要求

至少新增/更新以下断言：

1. 操作终端 strip **不再出现** `查看工件池`
2. 研究工作区 strip **继续出现** `焦点工件`，但不出现 ops 门禁词
3. contracts / backtests / raw 的阶段文案仍然正确
4. 现有 `smoke:workspace-routes` 不受影响
5. `tsc` 与 `App.test.tsx` 全绿

## 11. 风险与回滚

### 风险

- 现有测试大量依赖 `textOf('.workspace-context-strip')` 与 `.ops-context-strip`，重排字段后容易连锁失败
- 如果 CTA 改动过大，可能破坏已有 focus deep-link 回退路径

### 回滚策略

若本轮不稳定，直接回滚以下文件：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.tsx`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/index.css`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx`

## 12. 结论

下一步实现应聚焦于：

1. `OpsContextStrip` 去研究化
2. `WorkspaceContextStrip` 去运行化
3. 首卡摘要去重
4. 测试与 smoke 稳态保持

本轮仍属于前端只读架构收口，change class 保持 `RESEARCH_ONLY`。
