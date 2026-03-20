# Fenlie Dashboard 信息架构重构设计

- 日期：2026-03-20
- 模式：ui_rendering
- 推荐 change class：RESEARCH_ONLY
- 约束：只重构前端信息架构、交互层级、页面壳层与只读读模型映射；不触碰 live capital、order routing、execution queue、fund scheduling

## 1. 背景与问题

Fenlie 当前前端已具备较完整的只读读模型与多面板内容，但存在以下结构性问题：

1. 单一左侧导航承载终端、研究、契约、原始层等多类入口，造成导航臃肿。
2. 全局状态、域内状态、局部对象状态混在同一连续视觉流中，用户难以快速判断当前层级。
3. `TerminalPanels.tsx` 与 `WorkspacePanels.tsx` 承担了过多页面编排、筛选、详情与 drill-down 逻辑，导致文件职责失衡。
4. 详情主要通过纵向堆叠展开，缺少统一 inspector、明确回退路径与多层级穿透交互。
5. `public/internal` 虽已有表面区分，但仍缺少完整的访问语义和 fallback 解释。

本次设计的目标不是更换皮肤，而是重构信息架构、导航体系、用户交互路径与页面壳层。

## 2. 目标与非目标

### 2.1 目标

1. 将当前单栏堆叠式导航重构为：
   - 顶层全局导航
   - 域内侧栏导航
   - 内容内局部导航
   - 统一右侧 Inspector
2. 把“操作终端”与“研究工作区”从同一认知空间中解耦，建立机构级多域控制台。
3. 提升用户态可读性、可回退性、可追源性，减少长页面堆叠。
4. 保持 source-owned state，不在 UI 顶层重判 lane、gate、decision、surface 或 research 结果。
5. 在不破坏现有公开访问能力的前提下，渐进替换现有壳层与路由。

### 2.2 非目标

1. 不新增 live execution 能力。
2. 不重写后端 snapshot 构建逻辑。
3. 不改变 research / review / guard 的 source-of-truth 所属。
4. 不把本次重构扩展为视觉营销官网或装饰性动效工程。

## 3. 设计原则

1. **Source-owned first**：lane、queue、gate、blocker、review、surface、artifact group、research metrics 只来自 source artifact / read model。
2. **域分离**：总览、操作终端、研究工作区、契约验收、原始快照、系统状态分域管理。
3. **多层导航**：每层导航只解决一层问题，不混用。
4. **Inspector 统一**：对象详情从纵向堆叠 details 转为统一右侧 Inspector。
5. **渐进替换**：先重构壳层和路由，再逐步迁移页面内容，最后移除旧结构。
6. **中文优先**：header / badge / status / warning 在用户态全部中文化；仅 raw/source 层保留原始字段。

## 4. 信息架构

### 4.1 四层导航模型

#### L1 全局导航
回答“我现在处于哪个业务域？”

- 总览
- 操作终端
- 研究工作区
- 契约验收
- 原始快照
- 系统状态

#### L2 域内导航
回答“当前业务域下我在看哪类对象？”

根据一级域动态切换，不共用一条长菜单。

#### L3 内容导航
回答“当前页面内我聚焦哪个 panel / section / row？”

使用 breadcrumb、section tabs、局部 chips、row focus 实现。

#### L4 Inspector 穿透层
回答“这个对象的状态、来源、字段、关联对象、动作是什么？”

统一由右侧 Inspector Rail 承载。

### 4.2 一级域职责

#### 总览
- 系统态势
- 高优先告警
- 主线研究摘要
- 域入口分流

不承载长表格、raw payload 或深度详情。

#### 操作终端
- 调度与门禁
- 数据与体制
- 信号与风险
- 传导链与动作

只保留影响当前操作判断的 review 摘要，不承载完整研究工作台。

#### 研究工作区
- 主线策略
- 工件池
- 回测池
- 比较墙
- 研究验收

#### 契约验收
- public acceptance
- contract coverage
- workspace smoke
- topology / route audit

#### 原始快照
- public/internal snapshot
- raw payload
- schema / field mapping
- source visibility / fallback 追源

#### 系统状态
- 构建状态
- polling 状态
- source freshness 汇总
- 前端 runtime / degradation 事件

## 5. 页面骨架

### 5.1 统一壳层

#### 顶部 `GlobalTopbar`
- 品牌
- 一级导航
- surface 摘要
- refresh 状态
- change class
- 全局搜索

#### 左侧 `ContextSidebar`
- 当前一级域对应的二级导航
- 域摘要
- 域内快捷入口

#### 主区 `MainStage`
- 当前域页面主体
- 列表 / 面板 / 图表 / detail stage

#### 右侧 `InspectorRail`
- 当前对象状态卡
- 来源卡
- 关键字段卡
- 关联对象卡
- 动作卡

### 5.2 总览页

组件：
- `GlobalHealthStrip`
- `PriorityAlertStack`
- `PrimaryFlowCards`
- `ResearchMainlineCard`
- `QuickEntryGrid`

### 5.3 操作终端页

二级页：
- `OpsOrchestrationPage`
- `OpsDataRegimePage`
- `OpsSignalRiskPage`
- `OpsTransmissionPage`

### 5.4 研究工作区页

二级页：
- `ResearchMainlinePage`
- `ResearchArtifactsPage`
- `ResearchBacktestsPage`
- `ResearchComparisonPage`
- `ResearchVerificationPage`

### 5.5 工程控制平面

独立页面：
- `ContractsPage`
- `RawSnapshotPage`
- `SystemStatusPage`

## 6. 交互契约

### 6.1 主交互类型

1. 切域
2. 切上下文
3. 筛选
4. 聚焦
5. 穿透
6. 回退

### 6.2 筛选层级

#### 全局搜索
跨域搜索 artifact、lane、strategy、snapshot path、section。

#### 域内筛选
位于 `ContextHeader`，对当前域生效。

#### 局部筛选
只作用于当前列表/表格/面板。

所有筛选进入 URL query，保证刷新与分享可恢复。

### 6.3 Drill-down 模型

旧模式：纵向展开 details。  
新模式：主区保持列表上下文 + Inspector 显示详情 + 深层对象进入独立 detail stage。

层级固定为：
1. 列表态
2. 聚焦态
3. 对象详情态
4. 追源态

### 6.4 回退模型

- 追源态 → 详情态：返回详情
- 详情态 → 聚焦态：返回列表
- 聚焦态 → 域默认态：取消聚焦
- 域默认态 → 总览：返回总览

浏览器 Back/Forward、页面内返回按钮、breadcrumb 三者行为保持一致。

### 6.5 `public/internal` 语义

切换器只放在域页顶部，不放到每个小组件内。

切换后同步更新：
- Topbar：requested / effective / redaction level
- ContextHeader：当前域 surface 提示与 fallback 提示
- Inspector：字段裁剪与来源 fallback 说明

fallback 必须显式展示原因，不允许暗中替换。

### 6.6 用户态中文化

状态、badge、warning 在用户态统一中文化。  
仅 raw/source 层保留原始英文键和值。

## 7. 数据与组件契约

### 7.1 Source-owned 范围

只读字段包括但不限于：
- lane head
- queue ownership
- gate state
- blocker
- research decision
- review readiness
- requested/effective surface
- fallback chain
- redaction level
- artifact canonical/archive 归属
- research OOS metrics

UI 不得重新裁定这些语义。

### 7.2 UI 可派生范围

允许的 UI 派生仅包括：
- 中文 label / badge tone
- breadcrumb / 当前对象聚焦
- 局部筛选 / 排序
- inspector 展示组装

### 7.3 Store 分层

#### Source slices
- `snapshotMetaSlice`
- `surfaceSlice`
- `orchestrationSlice`
- `dataRegimeSlice`
- `signalRiskSlice`
- `researchSlice`
- `verificationSlice`

#### UI slices
- `navigationUiSlice`
- `filterUiSlice`
- `inspectorUiSlice`
- `layoutUiSlice`

#### Selectors
只做只读组装，不做业务重判。

### 7.4 路由契约

从当前二分路由过渡为：
- `/overview`
- `/ops/:section?`
- `/research/:section?`
- `/contracts/:section?`
- `/raw/:section?`
- `/system/:section?`

配合 query：
- `panel`
- `section`
- `row`
- `artifact`
- `domain`
- `tone`
- `search`
- `group`
- `symbol`
- `view`
- `window`

### 7.5 `App.tsx` 责任收缩

`App.tsx` 只负责：
- 路由装配
- 顶层数据刷新
- surface 范围切换
- 壳层注入
- 错误边界

不再负责页面内业务编排与字段映射。

### 7.6 组件拆分方向

#### 新壳层
- `AppShell.tsx`
- `GlobalTopbar.tsx`
- `ContextSidebar.tsx`
- `ContextHeader.tsx`
- `InspectorRail.tsx`
- `GlobalNoticeLayer.tsx`

#### 新页面
- `OverviewPage.tsx`
- `OpsPage.tsx`
- `ResearchPage.tsx`
- `ContractsPage.tsx`
- `RawSnapshotPage.tsx`
- `SystemStatusPage.tsx`

#### 新特性页
- `src/features/ops/*`
- `src/features/research/*`

#### Adapter
- `overview-adapter.ts`
- `ops-adapter.ts`
- `research-adapter.ts`
- `inspector-adapter.ts`

## 8. 视觉与控件基线

推荐采用组合基线：

1. **Linear 风格**：壳层、边框、标题、工具条
2. **Data Dense 风格**：表格、状态板、筛选条、比较区
3. **Dark Mode 规范**：交互反馈、hover、focus、状态高亮

不采用装饰性强的 glassmorphism / bento / 营销官网范式。

控件以以下组合为主：
- Segmented Tabs
- Filter Chips
- Navigator List
- Inspector Drawer / Side Rail

尽量避免肥大下拉主导交互。

## 9. 实施顺序

### Phase 0 基线冻结
- 固定当前 acceptance / smoke / route 对照基线

### Phase 1 壳层与一级路由
- 新建 `GlobalTopbar` / `ContextSidebar` / `ContextHeader` / `InspectorRail`
- 先保留旧内容挂载入新壳层

### Phase 2 导航模型抽象
- 替换 `NAV_ITEMS`
- 引入 `PRIMARY_NAV` / `OPS_NAV` / `RESEARCH_NAV` / `CONTRACTS_NAV` / `RAW_NAV` / `SYSTEM_NAV`

### Phase 3 操作终端拆页
- 先拆 `TerminalPanels.tsx`

### Phase 4 研究工作区拆页
- 再拆 `WorkspacePanels.tsx`

### Phase 5 统一 Inspector
- lane / artifact / acceptance / snapshot 统一适配

### Phase 6 工程控制平面独立
- 独立 `ContractsPage` / `RawSnapshotPage` / `SystemStatusPage`

### Phase 7 下线旧结构
- 清理旧导航、旧 details、旧大页面壳

## 10. 验证矩阵

每阶段至少执行：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx tsc --noEmit
npx vitest run src/App.test.tsx src/adapters/read-model.test.ts src/components/ui-kit.test.tsx
```

壳层 / 路由阶段：

```bash
npm run verify:public-surface -- --skip-workspace-build
npm run smoke:workspace-routes
```

数据契约阶段：

```bash
pytest -q /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_dashboard_frontend_snapshot_script.py
```

快照刷新：

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

公开面拓扑：

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

## 11. 风险与回滚

### 11.1 主要风险

1. 只换壳不换交互，导致视觉上更清楚但操作逻辑仍混乱。
2. UI adapter 过度派生，误伤 source-owned state。
3. 路由切换时丢失现有 focus query 与分享链接。
4. 先拆研究页而不拆壳层，导致新旧逻辑交织更严重。

### 11.2 回滚原则

每个阶段独立提交并保留回滚点：
- baseline-before-shell-refactor
- shell-only-routing-introduced
- domain-navigation-cutover
- ops-pages-split
- research-pages-split
- inspector-unified
- control-pages-isolated
- legacy-shell-removed

## 12. 推荐的首次实施范围（V1）

建议先做最小可见收益版本：

1. 引入新一级路由骨架
2. 引入 `GlobalTopbar + ContextSidebar + ContextHeader`
3. 把“总览 / 操作终端 / 研究工作区”从旧壳中分离
4. 暂时复用旧 `TerminalPanels` / `WorkspacePanels` 挂到新舞台
5. 先把导航层级、回退路径、surface 语义做对

这样可以最小成本解决：
- 导航臃肿
- 页面层级不清
- public/internal 语义弱
- 回退路径缺失

## 13. 结论

本次 Fenlie 前端重构的核心，不是视觉换肤，而是信息架构重排。  
推荐方案为：

- **壳层采用总览分流 + 三层导航 + 统一 Inspector**
- **内容采用高信息密度机构级控制台**
- **数据保持 source-owned，UI 只做可解释轻派生**
- **采用渐进替换，不一次性爆破重写**

在该设计下，前端会从“单栏堆叠的大页面”演化为“多域分层、路径明确、追源可回退的机构级只读控制台”。
