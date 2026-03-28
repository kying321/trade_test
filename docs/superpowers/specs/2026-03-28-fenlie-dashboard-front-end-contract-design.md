# Fenlie Dashboard Front-end Contract Redesign

## 1. Goal

为 Fenlie dashboard/web 建立一套**面向专业垂类控制台**的前端设计合同（front-end contract），目标不是简单“变好看”，而是同时满足：

- 专业深度足够高
- 首次学习成本足够低
- 跨页面交互语义稳定
- 后续 lane / section / artifact 扩展成本低
- source-owned / read-only / deep-link 的系统特性在前端层被正确表达

该设计合同的核心任务是：

1. 固定信息架构层级
2. 固定控件语义
3. 固定页面模板
4. 固定响应式退化策略
5. 固定视觉与密度的优先级顺序

本 spec 约束的是**前端结构与交互语言**，不是一次性视觉重画稿。

---

## 2. Background and Diagnosis

当前 Fenlie 已具备较强的专业面板基础：

- 存在稳定壳层：topbar / sidebar / main stage / inspector
- 已具备 source-owned deep-link 模型：`artifact / panel / section / row / anchor`
- 已具备命令式定位能力：全局搜索、section rail、breadcrumb、handoff link
- 已形成总览 / 操作 / 研究三大域

但当前问题不在“有没有功能”，而在于**交互语言尚未完全收敛**。

### 2.1 当前主要问题

1. **控件语义混用**
   - `nav-link` / `button` / `stack-button` / `chip-button` 视觉上过于接近
   - 用户难以一眼区分“跳页 / 筛选 / 动作 / 选择实体 / 视图切换”

2. **层级职责存在重复解释**
   - topbar 与 sidebar 均在承担部分导航和解释任务
   - 页头与卡片标题都在重复解释当前页意义

3. **页面节奏不统一**
   - 总览更像入口页
   - 操作更像监控墙
   - 研究更像工作台
   - 但三者缺少统一的“首屏回答结构”

4. **响应式退化过于粗糙**
   - 多栏布局在中宽屏容易直接塌成单列
   - 丢失对比视图和操作节奏

5. **文本截断策略过重**
   - 大量关键路径文字默认仅截断，不保证就地展开
   - 影响可读性与可学习性

6. **视觉层已有均衡，但信息密度等级仍不清晰**
   - 高密信息区与低密引导区的视觉区分不够明确
   - 容器样式一致性高于语义差异性

### 2.2 参考原则

本 spec 参考了两类产品逻辑：

- [skills.sh](https://skills.sh/) 的 **search-first / low-chrome / task-first**
- [ui-design-bench](https://ui-design-bench.vercel.app/) 的 **stable frame / predictable comparison / low surprise navigation**

Fenlie 不应复制其外观，而应吸收其**认知负担控制方法**。

---

## 3. Non-Goals

本 spec 首版**不覆盖**以下内容：

- 不重写 read-model / source artifact 结构
- 不改动 live execution / risk / routing authority
- 不新增后端搜索服务
- 不做 marketing site 风格首页
- 不以“酷炫视觉”替代结构清晰度
- 不把 dashboard 变成高度动画化产品

---

## 4. Design Principles

### 4.1 Principle A — Source-owned first, UI-derived second

前端只应表达 source-owned 状态，不应重造状态。

页面要做的是：

- 放大当前 source-owned 焦点
- 降低定位成本
- 缩短从摘要到证据的路径
- 明确当前层级，不做 authority 重算

### 4.2 Principle B — One control, one semantic role

每一类控件只能表达一种主语义。

如果用户需要额外理解“这个控件到底是跳转、筛选、还是动作”，说明语义设计失败。

### 4.3 Principle C — Every page answers three questions first

每个主页面首屏必须优先回答：

1. 现在发生了什么
2. 当前应该看哪条线
3. 下一步可以做什么

### 4.4 Principle D — Search and deep-link are primary capabilities

Fenlie 不是“浏览型”面板，而是“定位 + 穿透 + 回退”面板。

因此：

- 搜索不是附属功能
- anchor / artifact / section / row 深链不是调试特性，而是主交互能力

### 4.5 Principle E — Density is a tool, not a style

高密度应该只出现在：

- 列表
- 证据表格
- 对比区
- inspector

低密度应该只出现在：

- 入口区
- 页头
- 解释型摘要

不能所有地方同密度，也不能所有地方同装饰。

---

## 5. Information Architecture Contract

Fenlie dashboard 采用四层信息架构。

### 5.1 L0 — Global Tool Layer

职责：任何页面都不改变的全局能力。

包含：

- 全局搜索
- 环境状态（只读/内部/快照模式等）
- theme / density preference（后续可扩展）
- 全局 freshness / snapshot timestamp

承载位置：`topbar`

### 5.2 L1 — Domain Layer

职责：一级域切换。

固定为：

- 总览
- 操作终端
- 研究工作区

承载位置：`topbar primary nav`

### 5.3 L2 — Task / Lane Layer

职责：在当前域内部切换工作任务。

例如：

- public / internal
- artifacts / alignment / backtests / contracts / raw
- panel / section rail

承载位置：

- sidebar
- page-local subnav
- section rail

### 5.4 L3 — Evidence Layer

职责：把当前焦点下钻到 source heads / checks / rows / raw snapshot。

承载位置：

- main stage drilldown
- inspector rail
- breadcrumb
- handoff links

### 5.5 IA rule

层级必须严格单向清晰：

- L0 不解释具体任务
- L1 不承载 row-level evidence
- L2 不重复全局状态
- L3 不冒充一级导航

---

## 6. Shell Contract

### 6.1 Topbar Contract

Topbar 只保留四类内容：

1. 品牌与环境身份
2. 一级域切换
3. 全局搜索
4. 全局环境状态

Topbar **不应继续承载**：

- 大段页面解释文案
- 当前页的局部说明
- 与 sidebar 重复的导航信息

### 6.2 Sidebar Contract

Sidebar 只负责当前域内部导航，不负责解释整个系统。

Sidebar 固定三块：

1. 当前域标题 / subtitle
2. 域内页面与阶段导航
3. utility / raw entry

Sidebar 不应再承担：

- 全局 summary
- 变更级别主提示
- 与 topbar 重复的一级域上下文

### 6.3 Main Stage Contract

Main Stage 永远只做两件事：

- 当前任务主要内容
- 当前焦点逐层展开

Main Stage 不应承载全局工具性杂项。

### 6.4 Inspector Contract

Inspector 是 Fenlie 的关键专业特征，必须保留。

职责固定为：

- 当前选择对象摘要
- 当前 focus 关键字段
- 快速 handoff / 回退入口
- 原始证据摘要

后续允许：

- pin / collapse / auto-hide
- row compare mode

但不应变成第二个 sidebar。

---

## 7. Control Taxonomy Contract

这是本 spec 最重要的合同之一。

### 7.1 Tabs

用途：一级域切换。

示例：

- 总览
- 操作终端
- 研究工作区

要求：

- 视觉最稳、最少噪音
- 永远表示“切换主上下文”
- 不承担筛选或动作语义

### 7.2 Segmented Controls

用途：同一上下文内的模式切换。

示例：

- public / internal
- all / module / route / artifact
- title / artifact / path

要求：

- 一眼看出是 mutually exclusive 状态
- 切换后当前页面不变，只改变视图模式

### 7.3 Chips

用途：轻量过滤与状态标记。

示例：

- tone filter
- focus tags
- breadcrumb 中的轻量层级标签（如保留）

要求：

- 不作为主跳页按钮
- 不承担 destructive / confirm 动作

### 7.4 Action Buttons

用途：触发动作。

示例：

- 打开搜索
- 查看全部结果
- 进入源头主线
- 打开原始快照

要求：

- 视觉权重最高
- 与导航项明显区分
- 文案要体现动词

### 7.5 Entity Rows / Cards

用途：选择对象或进入详情。

示例：

- artifact row
- search result row
- topology node card
- target pool row

要求：

- 选中态明确
- hover 表示可聚焦
- 与 action button 不混淆

### 7.6 Rule of separation

当前代码中的统一 button 基类需要拆成至少三层：

- navigation primitives
- selection primitives
- action primitives

否则系统越长，学习成本越高。

---

## 8. Page Template Contract

### 8.1 Overview Template

Overview 的职责不是“展示更多”，而是“正确分流”。

固定结构：

1. 当前系统态
2. 当前优先线（top 1-2）
3. 主入口卡片
4. 特殊高优先 lane 摘要

Overview 不应堆过多并列卡片。

它必须是**决策前厅**。

### 8.2 Ops Template

Ops 是**运行态驾驶舱**。

固定结构：

1. 顶部：当前环境 / 当前 focus / 当前回退路径
2. 中部：4 个主 panel
3. 每个 panel 内部：summary -> drilldown -> row
4. 右侧：当前 focus object inspector

Ops 的目标是：

- 快速看状态
- 快速定位 panel
- 快速进入 section
- 快速进入 row

### 8.3 Research Template

Research 是**证据工作台**。

固定结构：

1. 左侧：artifact map / group / filter / focus
2. 中部：当前 artifact summary 与 target pool
3. 下层：handoff / comparison / contracts / raw
4. 右侧或二级区域：当前 artifact 关键字段 / source evidence

Research 的目标是：

- 建立研究地图
- 明确当前对象
- 减少 artifact 漂移
- 保证从摘要到 raw 的路径短而稳

### 8.4 Search Template

Search 是**系统级定位器**。

固定结构：

1. 查询词
2. scope 切换
3. 分类结果
4. 命中理由
5. 深链跳转

Search 的目标不是“展示很多”，而是“尽快把人带到正确位置”。

---

## 9. Page Rhythm Contract

所有主页面首屏都必须符合统一节奏：

### 9.1 第一屏：State

告诉用户：现在发生了什么。

### 9.2 第二屏：Focus

告诉用户：当前最重要的对象/线是什么。

### 9.3 第三屏：Action

告诉用户：下一步能去哪里。

### 9.4 第四屏以后：Evidence

进入 drilldown、raw、subcommand、source-head、comparison。

如果首屏先给大量 evidence，而没有先给 state/focus/action，用户学习成本会陡增。

---

## 10. Search and Navigation Contract

### 10.1 Search-first

搜索应成为系统一级能力。

要求：

- 任意页面都可快速唤起
- 结果必须能深链落到 section 或 artifact
- 命中理由可解释
- scope 语义稳定

### 10.2 Deep-link as contract

以下 query 语义继续保留并制度化：

- `artifact`
- `panel`
- `section`
- `row`
- `anchor`
- `group`
- `view`
- `search`
- `search_scope`

但这些参数必须分清：

- **导航态参数**
- **视图态参数**
- **过滤态参数**

不能再继续模糊堆积。

### 10.3 Breadcrumb usage

Breadcrumb 只在“已经进入钻取层级”时显示。

如果用户仍在页面总览层，不应该强行显示复杂 breadcrumb。

---

## 11. Responsive Strategy Contract

禁止继续使用“多栏直接一刀切单列”的粗放退化方式。

改为四级响应式策略：

### 11.1 XL (`>=1600px`)

- 三栏完整态
- sidebar / main / inspector 全开

### 11.2 L (`1280-1599px`)

- sidebar 保留
- inspector 收窄
- 主区仍允许两列 panel

### 11.3 M (`960-1279px`)

- inspector 改为 overlay 或可展开抽屉
- sidebar 可折叠
- 主区保留最少双列

### 11.4 S (`<960px`)

- 纵向堆叠
- 搜索优先
- inspector 改为页面内折叠模块

目标不是“所有屏幕都长一样”，而是“每个宽度都保留核心任务能力”。

---

## 12. Text and Readability Contract

### 12.1 Clamp policy

截断只适用于：

- 次级说明文案
- 路径
- 较长标题的非关键场景

以下内容不应默认不可展开：

- 当前焦点对象名
- 当前主线名
- 当前 artifact 标题
- breadcrumb 当前项
- 搜索结果标题

### 12.2 Label discipline

标题命名应遵循：

- 域级：中文稳定名词
- section：任务导向短语
- row：对象名
- action：动词开头

避免同层既有“功能名词”又有“解释句子”的混写。

### 12.3 Notes and microcopy

说明文案只放在真正降低认知成本的位置。

优先放：

- 页头
- 空态
- 错误态
- 搜索提示

减少在每张卡顶部重复说明。

---

## 13. Visual System Contract

### 13.1 Visual priority order

视觉优先级顺序应为：

1. 语义层次
2. 可读性
3. 密度控制
4. 状态区分
5. 装饰感

### 13.2 Decoration budget

Fenlie 允许保留：

- 轻度 glass / blur
- 轻度 gradient
- 轻度 tone 区分（overview / ops / research）

但装饰预算应主要集中在：

- shell 层
- hero / page header
- 特殊高优先卡片

高密信息区必须更克制。

### 13.3 Density tiers

建议建立三档密度 token：

- `comfortable`：overview / page header / hero
- `compact`：panel / summary / search result
- `dense`：tables / evidence / inspector / comparison

后续可由用户切换，但首版先在结构层固定。

---

## 14. Extensibility Contract

为了保证未来继续长功能时不失控，前端必须引入模板化扩展规则。

### 14.1 New domain rule

新增一级域必须满足：

- 有独立主任务
- 有独立 page template
- 不只是现有页面的过滤视图

### 14.2 New section rule

新增 section 必须声明：

- 所属页面模板
- 所属层级（summary/focus/action/evidence）
- 对应 deep-link id
- 对应 search metadata

### 14.3 New control rule

新增控件必须声明：

- semantic role
- active state
- hover state
- selected state
- keyboard behavior
- responsive behavior

不能再“先复用一个看起来差不多的按钮”。

---

## 15. Rollout Plan

### Phase 0 — Contract-first cleanup

先不做视觉大修，只做合同化收敛：

1. 拆分按钮/导航/选择控件基类
2. 收缩 topbar 职责
3. 收缩 sidebar 职责
4. 统一页头模板
5. 建立 inspector 行为合同

### Phase 1 — Template normalization

1. 统一 Overview / Ops / Research / Search 页面模板
2. 建立首屏 state/focus/action 结构
3. 调整 section 节奏

### Phase 2 — Responsive and density

1. 重写多栏退化策略
2. 明确 inspector 在中屏行为
3. 建立 density tier token

### Phase 3 — Visual refinement

1. 调整圆角、阴影、透明度、背景层次
2. 清理装饰噪音
3. 提升专业感与长期可用性

### Phase 4 — Contract enforcement

补充：

- 组件测试
- 浏览器 smoke
- keyboard / deep-link / responsive 验收项

---

## 16. Validation Requirements

后续任何实现 PR，至少要验证：

1. 主导航、域内导航、筛选、动作在视觉和语义上可区分
2. 搜索、breadcrumb、handoff、section link 的导航落点一致
3. 中宽屏不丢失核心任务能力
4. 关键对象名不被无解释截断
5. inspector 不与 sidebar 重复承担导航职责
6. 新增模块仍可自然挂接 search / focus / anchor / inspector

---

## 17. Change Class

本 spec 及其后续前端结构重构，建议 change class：

- `RESEARCH_ONLY`

原因：

- 仅涉及 dashboard/web 表达层
- 不改变 live authority
- 不触碰 order routing / risk execution path
- 不改变 source-of-truth ownership

---

## 18. Final Recommendation

Fenlie 前端的下一阶段目标，不是继续“补面板”，而是把现有能力收敛成一套**稳定、可学习、可扩展的交互语法**。

推荐总方向：

- **Bloomberg/Aladdin 式的信息纪律**
- **Command palette 式的定位效率**
- **Research workbench 式的证据穿透**

在这个方向上，Fenlie 可以保持专业深度，同时把新用户学习成本压到最低。
