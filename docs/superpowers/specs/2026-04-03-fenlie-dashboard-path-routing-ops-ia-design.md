# Fenlie Dashboard Path Routing + Ops IA Design

> Date: 2026-04-03  
> Change class: `RESEARCH_ONLY`

## Goal
把当前 Fenlie dashboard 从 `HashRouter + giant terminal/workspace shells` 推进到 **path-based routing + ops-first page architecture**，先完成第一阶段的可审计落地：

1. 移除生产 hash routing 依赖
2. 引入 `/ops/*` 路由族
3. 保留 `/terminal/*`、`/workspace/*` 旧深链兼容
4. 不碰 live execution / live guard / source-of-truth contracts

## Why This Slice First
当前真正的结构性问题不只是 `/overview`，而是：

- `system/dashboard/web/src/App.tsx` 仍由 `HashRouter` 驱动
- `system/dashboard/web/src/components/TerminalPanels.tsx` 持续混放 observe / diagnose / act 语义
- `system/dashboard/web/src/components/WorkspacePanels.tsx` 继续承担过重的 workspace 聚合职责

因此第一刀必须先切 **router 和 page ownership**，不能继续在旧壳上加卡片。

## Chosen Approach
采用 **渐进式 page-shell 改造**：

### 1. Router 先切换为 path-based
- `HashRouter -> BrowserRouter`
- 保持应用内部 link 全部输出 `/path?query`
- 在应用启动时增加 legacy hash redirect，把 `#/overview`、`#/terminal/public` 等旧入口转成 path

### 2. 先引入 `/ops/*`，不一次性拆多应用
第一阶段新增：

- `/ops/overview`
- `/ops/risk`
- `/ops/audits`
- `/ops/runbooks`
- `/ops/workflow`
- `/ops/traces`

但只做 **壳层 + 首轮内容迁移**：
- `/ops/overview` 先复用现有 `OverviewPage`
- `/ops/risk` 先承接现有 terminal 公开面主内容
- `/ops/audits|runbooks|workflow|traces` 先落占位页或最小 page shell，确保 IA 成型

### 3. Legacy terminal/workspace 保持兼容
- `/terminal/public` -> redirect 到 `/ops/risk`
- `/terminal/internal` -> 暂保留原页面，避免一次性破坏 internal 深链
- `/overview` -> redirect 到 `/ops/overview`
- `/workspace/*` 暂保留，后续第二阶段再进入 `/research/*`

### 4. 不推倒已有 shell primitives
保留并复用：
- `AppShell`
- `GlobalTopbar`
- `ContextSidebar`
- `InspectorRail`

本轮只重排 route ownership 和 nav 信息架构，不先做大规模视觉重写。

## IA Intent
本轮先把用户心智从“一个前端里很多长页”推进为“按任务进入不同页面”：

- **Observe** → `/ops/overview`
- **Diagnose** → `/ops/risk` + `/ops/audits`
- **Act** → `/ops/runbooks` + `/ops/workflow`
- **Trace** → `/ops/traces`

研究域和公开 docs 域保留为下一阶段，不在本轮强行展开成多 app / 多子域。

## Routing Rules
1. 生产输出不再出现 `#/...`
2. 后端/快照中仍可能带 `#/...` 的 route payload，前端统一做 normalize
3. 旧公开入口必须可自动跳转到 path route，不能白屏
4. shared focus/query 参数继续复用现有 query contract，不新增 schema

## UX Rules
1. `/ops/overview` 只做 90 秒总览，不接详细表
2. 可执行动作不从 overview 卡片直接下发
3. `/ops/risk` 先承接现有 terminal 的主观察/诊断面
4. `/ops/audits|runbooks|workflow|traces` 即便首轮只放最小内容，也必须进入左侧导航，明确 page ownership

## File Ownership
### Primary
- `system/dashboard/web/src/App.tsx`
- `system/dashboard/web/src/navigation/nav-config.ts`
- `system/dashboard/web/src/navigation/route-contract.ts`
- `system/dashboard/web/src/utils/focus-links.ts`
- `system/dashboard/web/src/pages/OverviewPage.tsx`
- `system/dashboard/web/src/pages/OpsPage.tsx`

### New Pages
- `system/dashboard/web/src/pages/ops/OpsOverviewPage.tsx`
- `system/dashboard/web/src/pages/ops/OpsRiskPage.tsx`
- `system/dashboard/web/src/pages/ops/OpsAuditsPage.tsx`
- `system/dashboard/web/src/pages/ops/OpsRunbooksPage.tsx`
- `system/dashboard/web/src/pages/ops/OpsWorkflowPage.tsx`
- `system/dashboard/web/src/pages/ops/OpsTracesPage.tsx`

### Tests
- `system/dashboard/web/src/App.test.tsx`
- `system/dashboard/web/src/pages/GraphHomePage.test.tsx`
- `system/dashboard/web/src/search/search-ui.test.tsx`
- any focused nav/route helper tests touched by the router change

## Non-Goals
- 不拆 backend 服务
- 不新增 live actions
- 不变更 snapshot schema
- 不做 research surface 改名
- 不做 design token 全量重写

## Risks
1. 现有 smoke / progress / acceptance 文档大量绑定 `#/...`
2. `window.location.hash` 相关测试需要整体迁移
3. 旧 snapshot payload 仍含 hash route 时，需要稳定 normalize 以避免 broken deep links

## Rollback
若第一阶段不稳定，回滚以下文件即可恢复：
- `system/dashboard/web/src/App.tsx`
- `system/dashboard/web/src/navigation/nav-config.ts`
- `system/dashboard/web/src/navigation/route-contract.ts`
- `system/dashboard/web/src/utils/focus-links.ts`
- `system/dashboard/web/src/pages/ops/*`
- 相关测试文件

## Approval Note
用户已明确批准“按推荐与目标持续实施”；因此本设计文档作为已批准方案的仓库内固化，不额外等待新的设计确认。
