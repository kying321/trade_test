# Fenlie Dashboard V1 壳层与导航重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏现有公开访问、surface 语义、focus deep-link 与 source-owned 展示契约的前提下，完成 Fenlie 前端 V1 壳层与导航重构：引入总览页、顶部全局导航、域内侧栏、上下文头部与只读 Inspector 框架，并把现有 `#/terminal/*`、`#/workspace/*` 真实内容平滑挂入新壳层。

**Architecture:** 本计划只覆盖 V1 子项目：壳层、导航、overview 分流页、baseline freeze 与 query 保留兼容层，不直接切到 `/ops/:subpage`、`/research/:subpage` 等新语义 path。现有 `TerminalPanels.tsx` 与 `WorkspacePanels.tsx` 继续作为 source-owned legacy body，由新 `AppShell` 包裹；后续 canonical path cutover、ops/research 细页拆分、Inspector 实体统一、contracts/raw/system 独立将进入下一份计划。

**Tech Stack:** Vite 7、React 19、TypeScript、React Router、Zustand、Vitest、Python smoke/acceptance 脚本

> 2026-03-21 注：本计划里原先占位的 `run_dashboard_shell_navigation_smoke.py` / `test_run_dashboard_shell_navigation_smoke_script.py` 并未落成独立文件；当前已由
> `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_workspace_artifacts_smoke.py`
> 与
> `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_dashboard_workspace_artifacts_smoke_script.py`
> 承担对应公开 workspace 路由/导航烟测职责。后续阅读本计划时，应以这组现行 runner 作为真实入口。

---

## 0. 范围切分与实现约束

本 spec 覆盖多个独立子系统；按 `writing-plans` 规则，本计划只实施第一子项目：

1. V1 壳层与导航重构（本计划）
2. Canonical `/ops` `/research` path cutover（后续计划）
3. Ops 四页拆分（后续计划）
4. Research 五页拆分（后续计划）
5. Inspector 实体统一与旧 details 下线（后续计划）
6. Contracts / Raw / System 三域独立（后续计划）

### 强约束

- 只读前端重构，change class 保持 `RESEARCH_ONLY`
- 不触碰 live capital、execution queue、order routing、fund scheduling
- 不重判 lane/gate/fallback/redaction/research decision
- 保持 `#/terminal/*`、`#/workspace/*` 为 V1 运行时主路由
- 保留并完整透传 `panel/section/row/artifact/domain/tone/search/group/view/symbol/window` query
- Inspector 动作仅允许导航/复制/打开证据

## 1. 目标文件结构

### Create

- `system/dashboard/web/src/app-shell/AppShell.tsx` — 新壳层四区布局（Topbar / Sidebar / Stage / Inspector）
- `system/dashboard/web/src/app-shell/GlobalTopbar.tsx` — 一级导航、surface 状态、全局搜索入口
- `system/dashboard/web/src/app-shell/ContextSidebar.tsx` — 域内导航、快捷入口
- `system/dashboard/web/src/app-shell/ContextHeader.tsx` — breadcrumb、域内筛选壳、surface 域内提示
- `system/dashboard/web/src/app-shell/GlobalNoticeLayer.tsx` — fallback/stale/degrade 统一提示
- `system/dashboard/web/src/app-shell/InspectorRail.tsx` — V1 只读 inspector 骨架（摘要级）
- `system/dashboard/web/src/navigation/nav-config.ts` — 一级/二级导航配置
- `system/dashboard/web/src/navigation/route-contract.ts` — V1 路由意图、query 保留与未来 canonical route 预留规则
- `system/dashboard/web/src/pages/OverviewPage.tsx` — 总览页
- `system/dashboard/web/src/pages/OpsPage.tsx` — 操作终端壳页，先挂载 legacy terminal body
- `system/dashboard/web/src/pages/ResearchPage.tsx` — 研究工作区壳页，先挂载 legacy workspace body
- `system/dashboard/web/src/pages/page-types.ts` — 一级域/子页类型定义
- `system/dashboard/web/src/navigation/route-contract.test.ts` — legacy alias 与 canonical route 测试
- `system/dashboard/web/src/app-shell/AppShell.test.tsx` — 新壳层导航/回退/notice 测试
- `system/scripts/run_dashboard_workspace_artifacts_smoke.py` — 当前公开 workspace 路由/导航浏览器烟测
- `system/tests/test_run_dashboard_workspace_artifacts_smoke_script.py` — 当前 workspace 路由烟测脚本单测

### Modify

- `system/dashboard/web/src/App.tsx` — 从旧 terminal/workspace 二分路由切到新壳层 + 一级域路由 + legacy alias
- `system/dashboard/web/src/index.css` — 新壳层布局、导航、overview、inspector、notice 样式
- `system/dashboard/web/src/utils/focus-links.ts` — 支持 canonical route 生成，保留 shared query
- `system/dashboard/web/src/utils/dictionary.ts` — 新导航、overview、shell、inspector 中文词条
- `system/dashboard/web/src/App.test.tsx` — 更新对新路由、overview、legacy alias、query 保留的断言

### Reuse without deep rewrite in V1

- `system/dashboard/web/src/components/TerminalPanels.tsx`
- `system/dashboard/web/src/components/WorkspacePanels.tsx`
- `system/dashboard/web/src/store/use-terminal-store.ts`
- `system/dashboard/web/src/adapters/read-model.ts`

### Validate

- `system/dashboard/web/src/App.test.tsx`
- `system/dashboard/web/src/navigation/route-contract.test.ts`
- `system/dashboard/web/src/app-shell/AppShell.test.tsx`
- `system/dashboard/web/src/adapters/read-model.test.ts`
- `system/dashboard/web/src/components/ui-kit.test.tsx`

## 2. 实现总顺序

1. 先冻结 baseline，确保回滚锚点存在
2. 再写 query 保留与 legacy route intent 测试，锁定 URL 语义
3. 再写 AppShell 与 overview 的壳层测试
4. 再实现新壳层，但继续使用旧运行时路由挂载真实内容
5. 再补 context sidebar / header / notice / inspector placeholder
6. 最后统一样式、跑新增 shell smoke 与既有 smoke、提交

---

### Task 0: 冻结 baseline 与回滚锚点

**Files:**
- Modify: 无代码修改；记录 review artifact 路径与基线验证结果
- Test: 现有验证矩阵

- [ ] **Step 1: 跑当前 TypeScript 与关键单测，记录输出**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx tsc --noEmit
npx vitest run src/App.test.tsx src/adapters/read-model.test.ts src/components/ui-kit.test.tsx
```

Expected: PASS

- [ ] **Step 2: 跑当前公开面验收与 workspace smoke，记录 review artifact 路径**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run verify:public-surface -- --skip-workspace-build
npm run smoke:workspace-routes
```

Expected: PASS，输出当前 acceptance / smoke artifact 路径

- [ ] **Step 3: 跑当前拓扑与快照脚本，确认 source 基线可复现**

Run:

```bash
pytest -q /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_dashboard_frontend_snapshot_script.py
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

Expected: PASS

- [ ] **Step 4: 为 baseline 打锚点提交与标签**

```bash
git tag baseline-before-shell-refactor
```

- [ ] **Step 5: 若 baseline 失败，则先停在这里，不进入壳层改造**

```md
任何 baseline FAIL 都先修基线，不允许继续推进 V1 壳层。
```

---

### Task 1: 锁定 V1 query 保留与路由意图契约

**Files:**
- Create: `system/dashboard/web/src/navigation/route-contract.ts`
- Create: `system/dashboard/web/src/navigation/route-contract.test.ts`
- Modify: `system/dashboard/web/src/utils/focus-links.ts`
- Test: `system/dashboard/web/src/navigation/route-contract.test.ts`

- [ ] **Step 1: 写 route contract 的失败测试**

```ts
import { describe, expect, it } from 'vitest';
import {
  buildOverviewLink,
  buildOpsLegacyLink,
  buildResearchLegacyLink,
  preserveSharedQuery,
} from './route-contract';

describe('route-contract', () => {
  it('keeps every shared query key when building overview-to-legacy links', () => {
    expect(
      buildResearchLegacyLink('artifacts', {
        artifact: 'price_action_breakout_pullback',
        group: 'research_mainline',
        view: 'public',
        symbol: 'ETHUSDT',
        window: 'forward-16',
      })
    ).toBe('/workspace/artifacts?artifact=price_action_breakout_pullback&group=research_mainline&view=public&symbol=ETHUSDT&window=forward-16');
  });

  it('preserves panel/section/row when navigating between shell surfaces', () => {
    expect(
      buildOpsLegacyLink('public', {
        panel: 'signal-risk',
        section: 'lane-pressure',
        row: 'ETHUSDT',
        view: 'public',
      })
    ).toBe('/terminal/public?panel=signal-risk&section=lane-pressure&row=ETHUSDT&view=public');
  });
});
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/navigation/route-contract.test.ts
```

Expected: FAIL，提示 `route-contract.ts` 不存在或导出缺失。

- [ ] **Step 3: 实现最小 route contract**

```ts
const SHARED_QUERY_KEYS = [
  'artifact',
  'domain',
  'tone',
  'search',
  'panel',
  'section',
  'row',
  'group',
  'view',
  'symbol',
  'window',
];

export function buildOpsLegacyLink(surface: 'public' | 'internal', focus: Record<string, string | undefined> = {}) {
  return buildPreservedLink(`/terminal/${surface}`, focus);
}

export function buildResearchLegacyLink(section: 'artifacts' | 'backtests' | 'contracts' | 'raw', focus: Record<string, string | undefined> = {}) {
  return buildPreservedLink(`/workspace/${section}`, focus);
}

export function buildOverviewLink(focus: Record<string, string | undefined> = {}) {
  return buildPreservedLink('/overview', focus);
}
```

- [ ] **Step 4: 扩展 `focus-links.ts` 统一使用 shared query 保留器**

```ts
export type SharedFocusState = TerminalFocusState & WorkspaceFocusState & {
  group?: string;
  view?: string;
  symbol?: string;
  window?: string;
};

export function buildOverviewFocusLink(focus: SharedFocusState = {}) {
  const qs = buildQuery({ ...focus });
  return `/overview${qs ? `?${qs}` : ''}`;
}
```

- [ ] **Step 5: 运行测试，确认 route contract 通过**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/navigation/route-contract.test.ts
```

Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/navigation/route-contract.ts \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/navigation/route-contract.test.ts \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/utils/focus-links.ts
git commit -m "test(nav): lock v1 legacy route and shared query contract"
```

---

### Task 2: 建立 V1 一级导航配置与类型约束

**Files:**
- Create: `system/dashboard/web/src/pages/page-types.ts`
- Create: `system/dashboard/web/src/navigation/nav-config.ts`
- Modify: `system/dashboard/web/src/utils/dictionary.ts`
- Test: `system/dashboard/web/src/navigation/route-contract.test.ts`

- [ ] **Step 1: 先写导航配置结构的最小类型定义**

```ts
export type PrimaryDomain = 'overview' | 'ops' | 'research';
export type ResearchLegacySection = 'artifacts' | 'backtests' | 'contracts' | 'raw';
export type OpsSurface = 'public' | 'internal';
```

- [ ] **Step 2: 写 `nav-config.ts` 的失败断言**

```ts
expect(PRIMARY_NAV.map((item) => item.id)).toEqual([
  'overview',
  'ops',
  'research',
]);
```

- [ ] **Step 3: 实现导航配置并补足中文 label key**

```ts
export const PRIMARY_NAV = [
  { id: 'overview', labelKey: 'nav_overview', to: '/overview' },
  { id: 'ops', labelKey: 'nav_ops', to: '/terminal/public' },
  { id: 'research', labelKey: 'nav_research', to: '/workspace/artifacts' },
] as const;

export const RESEARCH_LEGACY_NAV = [
  { id: 'artifacts', labelKey: 'nav_workspace_artifacts', to: '/workspace/artifacts' },
  { id: 'backtests', labelKey: 'nav_workspace_backtests', to: '/workspace/backtests' },
  { id: 'contracts', labelKey: 'nav_workspace_contracts', to: '/workspace/contracts' },
  { id: 'raw', labelKey: 'nav_workspace_raw', to: '/workspace/raw' },
] as const;
```

- [ ] **Step 4: 在 `dictionary.ts` 对新 key 加中文词条**

```ts
const EXTRA_LABELS = {
  nav_overview: '总览',
  nav_ops: '操作终端',
  nav_research: '研究工作区',
};
```

- [ ] **Step 5: 运行关联测试**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/navigation/route-contract.test.ts src/utils/dictionary.test.ts
```

Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/pages/page-types.ts \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/navigation/nav-config.ts \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/utils/dictionary.ts
git commit -m "feat(nav): add dashboard domain navigation config"
```

---

### Task 3: 用测试锁定新壳层结构与 overview 分流页

**Files:**
- Create: `system/dashboard/web/src/app-shell/AppShell.test.tsx`
- Create: `system/dashboard/web/src/pages/OverviewPage.tsx`
- Modify: `system/dashboard/web/src/App.test.tsx`
- Test: `system/dashboard/web/src/app-shell/AppShell.test.tsx`

- [ ] **Step 1: 为新壳层写失败测试**

```tsx
it('renders top-level navigation and overview entry cards', async () => {
  window.location.hash = '#/overview';
  render(<App />);
  expect(await screen.findByText('总览')).toBeInTheDocument();
  expect(screen.getByText('操作终端')).toBeInTheDocument();
  expect(screen.getByText('研究工作区')).toBeInTheDocument();
});

it('shows public/internal summary in shell header rather than per-card toggles', async () => {
  window.location.hash = '#/overview';
  render(<App />);
  expect(await screen.findByText(/请求视图/i)).toBeInTheDocument();
  expect(screen.getByText(/实际视图/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: 更新 `App.test.tsx`，添加 legacy alias 回归用例**

```tsx
it('redirects legacy workspace route into canonical research page while preserving artifact query', async () => {
  window.location.hash = '#/workspace/artifacts?artifact=price_action_breakout_pullback';
  render(<App />);
  await screen.findByText('研究工作区');
  expect(window.location.hash).toContain('#/workspace/artifacts');
  expect(window.location.hash).toContain('artifact=price_action_breakout_pullback');
});
```

- [ ] **Step 3: 实现最小 `OverviewPage.tsx`**

```tsx
export function OverviewPage() {
  return (
    <section>
      <h2>总览</h2>
      <div className="overview-entry-grid">
        <article>操作终端</article>
        <article>研究工作区</article>
        <article>契约验收</article>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: 运行壳层与 App 测试，确认先红后绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/app-shell/AppShell.test.tsx src/App.test.tsx
```

Expected: 初次 FAIL；实现后 PASS

- [ ] **Step 5: 提交**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/app-shell/AppShell.test.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/pages/OverviewPage.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx
git commit -m "test(shell): lock overview and legacy redirect behavior"
```

---

### Task 4: 实现 AppShell 与 overview 路由，但继续使用旧运行时路由挂载真实内容

**Files:**
- Create: `system/dashboard/web/src/app-shell/AppShell.tsx`
- Create: `system/dashboard/web/src/pages/OpsPage.tsx`
- Create: `system/dashboard/web/src/pages/ResearchPage.tsx`
- Modify: `system/dashboard/web/src/App.tsx`
- Test: `system/dashboard/web/src/App.test.tsx`

- [ ] **Step 1: 先写 `AppShell.tsx` 的最小骨架**

```tsx
export function AppShell({ topbar, sidebar, header, notice, inspector, children }: AppShellProps) {
  return (
    <div className="app-shell app-shell-v1">
      <header className="global-topbar">{topbar}</header>
      <div className="app-shell-grid">
        <aside className="context-sidebar">{sidebar}</aside>
        <main className="main-stage">
          {header}
          {notice}
          {children}
        </main>
        <aside className="inspector-rail">{inspector}</aside>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 在 `OpsPage.tsx` 中先挂载现有 `TerminalPanels`**

```tsx
export function OpsPage({ model, focus, surface }: OpsPageProps) {
  return (
    <section data-surface={surface}>
      <TerminalPanels model={model} focus={focus} />
    </section>
  );
}
```

- [ ] **Step 3: 在 `ResearchPage.tsx` 中先挂载现有 `WorkspacePanels`**

```tsx
export function ResearchPage({ model, focus, section }: ResearchPageProps) {
  return <WorkspacePanels model={model} section={section} focus={focus} />;
}
```

> `section` 必须显式限定为 `'artifacts' | 'backtests' | 'contracts' | 'raw'`，禁止任何“未知 section 折回 artifacts”的静默回退。

- [ ] **Step 4: 重写 `App.tsx` 路由层**

```tsx
<Routes>
  <Route path="/" element={<Navigate to="/overview" replace />} />
  <Route path="/overview" element={<OverviewRoute />} />
  <Route path="/terminal/:surface" element={<OpsRoute />} />
  <Route path="/workspace/:section" element={<ResearchRoute />} />
</Routes>
```

> 说明：V1 不引入 `/ops/:subpage`、`/research/:subpage` 运行时路由，避免 path 语义与 legacy body 脱钩。canonical path cutover 放入下一份计划。

- [ ] **Step 5: 在 `App.tsx` 统一解析并透传全量 shared query**

```tsx
const focus: SharedFocusState = {
  artifact: searchParams.get('artifact') || undefined,
  domain: searchParams.get('domain') || undefined,
  tone: searchParams.get('tone') || undefined,
  search: searchParams.get('search') || undefined,
  panel: searchParams.get('panel') || undefined,
  section: searchParams.get('section') || undefined,
  row: searchParams.get('row') || undefined,
  group: searchParams.get('group') || undefined,
  view: searchParams.get('view') || undefined,
  symbol: searchParams.get('symbol') || undefined,
  window: searchParams.get('window') || undefined,
};
```

- [ ] **Step 6: 在 `App.test.tsx` 增加 full query round-trip 用例**

```tsx
it('preserves shared query keys across overview, terminal, workspace and browser navigation', async () => {
  window.location.hash = '#/workspace/artifacts?artifact=price_action_breakout_pullback&group=research_mainline&view=public&symbol=ETHUSDT&window=forward-16&panel=lab-review&section=research-heads&row=price_action_breakout_pullback';
  render(<App />);
  await screen.findByText('研究工作区');
  expect(window.location.hash).toContain('group=research_mainline');
  expect(window.location.hash).toContain('window=forward-16');
});
```

- [ ] **Step 7: 保持 store 只读刷新语义不变**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx src/app-shell/AppShell.test.tsx
```

Expected: PASS，且 `loadDashboardSurface` 的 fallback 行为未变

- [ ] **Step 8: 提交**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/app-shell/AppShell.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/pages/OpsPage.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/pages/ResearchPage.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.tsx
git commit -m "feat(shell): add v1 app shell and canonical domain routes"
```

---

### Task 5: 落地 Topbar / Sidebar / ContextHeader / Notice / Inspector 占位实现

**Files:**
- Create: `system/dashboard/web/src/app-shell/GlobalTopbar.tsx`
- Create: `system/dashboard/web/src/app-shell/ContextSidebar.tsx`
- Create: `system/dashboard/web/src/app-shell/ContextHeader.tsx`
- Create: `system/dashboard/web/src/app-shell/GlobalNoticeLayer.tsx`
- Create: `system/dashboard/web/src/app-shell/InspectorRail.tsx`
- Modify: `system/dashboard/web/src/App.tsx`
- Modify: `system/dashboard/web/src/utils/dictionary.ts`
- Test: `system/dashboard/web/src/app-shell/AppShell.test.tsx`

- [ ] **Step 1: 为 Topbar/Sidebar 写最小渲染断言**

```tsx
expect(screen.getByRole('navigation', { name: 'primary-domains' })).toBeInTheDocument();
expect(screen.getByRole('navigation', { name: 'context-nav' })).toBeInTheDocument();
expect(screen.getByText('请求视图')).toBeInTheDocument();
expect(screen.getByText('实际视图')).toBeInTheDocument();
```

- [ ] **Step 2: 实现 `GlobalTopbar.tsx`**

```tsx
export function GlobalTopbar({ primaryNav, surfaceSummary, searchSlot }: GlobalTopbarProps) {
  return (
    <div className="global-topbar-inner">
      <nav aria-label="primary-domains">{/* PRIMARY_NAV */}</nav>
      <div className="surface-summary-strip">{surfaceSummary}</div>
      <div className="global-search-slot">{searchSlot}</div>
    </div>
  );
}
```

- [ ] **Step 3: 实现 `ContextSidebar.tsx` 与 `ContextHeader.tsx`**

```tsx
export function ContextSidebar({ items, title }: ContextSidebarProps) {
  return <nav aria-label="context-nav">{/* current domain items */}</nav>;
}

export function ContextHeader({ title, breadcrumbs, chips }: ContextHeaderProps) {
  return <section className="context-header">{/* title + breadcrumb + filter shell */}</section>;
}
```

> V1 约束：
> - `ops` 域侧栏只显示 surface 切换与 terminal focus rail 概览
> - `research` 域侧栏继续显示 `artifacts/backtests/contracts/raw`
> - `contracts/raw` 不降级为 placeholder，继续复用 `WorkspacePanels` 现有真实内容

- [ ] **Step 4: 实现 `GlobalNoticeLayer.tsx` 与 `InspectorRail.tsx` 的 V1 只读占位**

```tsx
export function GlobalNoticeLayer({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null;
  return <div className="global-notice-layer">{warnings.map((w) => <p key={w}>{w}</p>)}</div>;
}

export function InspectorRail({ title = '对象检查器', summaryRows = [] }: InspectorRailProps) {
  return <aside><h3>{title}</h3>{summaryRows.length ? /* rows */ null : <p>选择对象后显示摘要</p>}</aside>;
}
```

- [ ] **Step 5: 在 `App.tsx` 中把壳层组件串起来，但不引入写操作**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/app-shell/AppShell.test.tsx src/App.test.tsx
```

Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/app-shell/GlobalTopbar.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/app-shell/ContextSidebar.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/app-shell/ContextHeader.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/app-shell/GlobalNoticeLayer.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/app-shell/InspectorRail.tsx \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.tsx
git commit -m "feat(shell): add topbar sidebar header notice and inspector scaffolds"
```

---

### Task 6: 用 CSS 重排壳层与 overview，保留 legacy body 可读性

**Files:**
- Modify: `system/dashboard/web/src/index.css`
- Test: `system/dashboard/web/src/app-shell/AppShell.test.tsx`

- [ ] **Step 1: 先写最小样式目标清单**

```css
.app-shell-grid { display: grid; grid-template-columns: 240px minmax(0, 1fr) 320px; }
.global-topbar-inner { display: grid; grid-template-columns: 1fr auto auto; }
.overview-entry-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); }
.context-sidebar { position: sticky; top: 12px; }
.inspector-rail { border-left: 1px solid var(--line-soft); }
```

- [ ] **Step 2: 分离旧 `.left-rail` / `.header-bar` 与新壳层样式，避免冲突**

```css
.app-shell-v1 .left-rail { all: unset; }
.app-shell-v1 .global-topbar,
.app-shell-v1 .context-sidebar,
.app-shell-v1 .context-header,
.app-shell-v1 .inspector-rail { /* new panel chrome */ }
```

- [ ] **Step 3: 加入 public/internal 域级视觉分层**

```css
.surface-summary-strip .badge-public { /* blue */ }
.surface-summary-strip .badge-internal { /* violet */ }
.global-notice-layer.is-fallback { border-color: rgba(242, 169, 59, 0.35); }
```

- [ ] **Step 4: 在本地跑 build 确认样式不打断现有 body**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run build
```

Expected: PASS，构建产物生成，legacy body 仍可在新壳层中渲染

- [ ] **Step 5: 提交**

```bash
git add /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/index.css
git commit -m "style(shell): introduce v1 shell layout and overview styling"
```

---

### Task 7: 跑完整验证矩阵并补 shell smoke 回归

**Files:**
- Modify: `system/dashboard/web/src/App.test.tsx`（如 smoke 前发现断言缺口）
- Modify: `system/dashboard/web/src/navigation/route-contract.test.ts`（如需补 query 保留测试）
- Create: `system/scripts/run_dashboard_shell_navigation_smoke.py`
- Create: `system/tests/test_run_dashboard_shell_navigation_smoke_script.py`
- Test: `system/dashboard/web/src/App.test.tsx`

- [ ] **Step 1: 跑 TypeScript 与关键单测**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx tsc --noEmit
npx vitest run \
  src/App.test.tsx \
  src/navigation/route-contract.test.ts \
  src/app-shell/AppShell.test.tsx \
  src/adapters/read-model.test.ts \
  src/components/ui-kit.test.tsx
```

Expected: PASS

- [ ] **Step 2: 跑公开面验收与 workspace 路由 smoke**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run verify:public-surface -- --skip-workspace-build
npm run smoke:workspace-routes
```

Expected: PASS；若输出 review artifact，记录路径进入 handoff

- [ ] **Step 3: 新增 shell navigation smoke，覆盖 overview / terminal / workspace 与 Back/Forward**

```python
SHELL_ROUTE_ASSERTIONS = [
  {"route": "#/overview", "headline": "总览"},
  {"route": "#/terminal/public?panel=signal-risk&section=lane-pressure&row=ETHUSDT", "headline": "操作终端"},
  {"route": "#/workspace/artifacts?artifact=price_action_breakout_pullback&group=research_mainline", "headline": "研究工作区"},
]
```

最低断言必须包含：

```python
await page.goto(base_url + "#/overview", wait_until="networkidle")
await expect(page.get_by_role("heading", name="总览", exact=True)).to_be_visible()

await page.get_by_role("link", name="操作终端").click()
await expect(page).to_have_url(re.compile(r"#/terminal/public"))
await expect(page.get_by_text("请求视图")).to_be_visible()

await page.goto(base_url + "#/workspace/artifacts?artifact=price_action_breakout_pullback&group=research_mainline&panel=lab-review&section=research-heads&row=price_action_breakout_pullback")
await expect(page).to_have_url(re.compile(r"group=research_mainline"))
await expect(page).to_have_url(re.compile(r"panel=lab-review"))
await expect(page.get_by_role("heading", name="研究工作区", exact=True)).to_be_visible()

await page.go_back()
await expect(page.get_by_role("heading", name="操作终端", exact=True)).to_be_visible()
await page.go_forward()
await expect(page).to_have_url(re.compile(r"group=research_mainline"))
await expect(page.get_by_text("请求视图")).to_be_visible()
```

Run:

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_shell_navigation_smoke.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
pytest -q /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_dashboard_shell_navigation_smoke_script.py
```

Expected: PASS，覆盖 overview -> ops -> research、shared query 保持、surface notice、Back/Forward

- [ ] **Step 4: 跑 Python 契约与拓扑脚本**

Run:

```bash
pytest -q /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_dashboard_frontend_snapshot_script.py
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

Expected: PASS

- [ ] **Step 5: 若任一验证失败，只修最小回归，不扩大到 V2 页面拆分**

```md
只允许修：
- legacy alias 丢 query
- surface fallback 提示消失
- overview / ops / research 路由空白
- 新壳层 CSS 把旧 body 遮挡
不允许顺手拆 TerminalPanels / WorkspacePanels
```

- [ ] **Step 6: 提交**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_shell_navigation_smoke.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_dashboard_shell_navigation_smoke_script.py
git commit -m "test(shell): verify v1 shell navigation and legacy route preservation"
```

---

### Task 8: 写实施后 handoff 说明并冻结 V1 回滚点

**Files:**
- Modify: `system/docs/FENLIE_CODEX_MEMORY.md`（仅当验证命令或 durable 路径发生变化时）
- Create or update review artifact path notes in commit body / handoff note（如果项目已有既定 handoff 文件则复用；否则只在最终汇报中列出）

- [ ] **Step 1: 记录 V1 完成后的用户可见变化**

```md
- 总览 / 操作终端 / 研究工作区已成为一级域
- 旧 terminal/workspace deep-link 仍可访问
- panel/section/row query 仍可恢复焦点
- public/internal surface 提示在壳层中集中呈现
```

- [ ] **Step 2: 记录明确未做事项**

```md
- 未拆 TerminalPanels 四页
- 未拆 WorkspacePanels 五页
- 未统一 InspectorEntity
- 未独立 Contracts/Raw/System 页面真实内容
```

- [ ] **Step 3: 标记回滚点**

```bash
git tag fenlie-dashboard-v1-shell-nav-cutover
```

- [ ] **Step 4: 提交（如 Step 1/2 产生文件修改）**

```bash
git add /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md
git commit -m "docs(handoff): record v1 shell nav cutover status"
```

---

## 3. 执行注意事项

### 3.1 Source-owned 边界

实现过程中不要把以下逻辑塞进 UI：

- lane/gate/blocker 再裁定
- fallback/redaction 再解释为新的业务状态
- global search 生成新的优先级分数
- Inspector 动作触发执行或状态写入

### 3.2 何时停止

若连续两轮只是在微调样式或文案，而没有改变：
- 路由可达性
- 导航分层
- 回退路径
- public/internal 解释能力

则应停止并汇报，不继续结构性清理。

### 3.3 后续计划边界

本计划完成后，下一份计划应从以下五者中择一推进：

1. 运行时 path 从 legacy `/terminal/*`、`/workspace/*` 切到 canonical `/ops/*`、`/research/*`
2. `OpsPage` 内 legacy body 的四页拆分
3. `ResearchPage` 内 artifacts/backtests/comparison 真正拆页
4. `InspectorEntity` 统一与旧 details 下线
5. `ContractsPage` / `RawSnapshotPage` / `SystemStatusPage` 内容独立

不要在 V1 计划执行中越界完成它们。
