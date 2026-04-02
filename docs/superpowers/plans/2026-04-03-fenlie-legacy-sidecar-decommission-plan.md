# Fenlie Legacy Sidecar Decommission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 下线冻结的 legacy sidecar shell，确保生产入口只保留 `index.html -> src/main.tsx -> src/App.tsx` 一条主链。

**Architecture:** 当前生产入口已经完全切到 `src/main.tsx` 与 `src/App.tsx`；`src/main.legacy.jsx` 与 `src/App.legacy.jsx` 只剩旁路冻结壳。下线策略采用“先加隔离契约，再删入口，再删 legacy hook/lib，再删 legacy component subtree，最后做回归”的顺序，避免把共享字典、样式与当前 BrowserRouter 主链误删。

**Tech Stack:** React 19, Vite 7, Vitest 3, TypeScript 5, 少量 legacy JS/JSX sidecar 文件

---

## Audited Current Boundary

### Active production chain

- `index.html`
- `src/main.tsx`
- `src/App.tsx`

### Frozen legacy sidecar entry

- `src/main.legacy.jsx`
- `src/App.legacy.jsx`

### Legacy-only hook/lib chain

- `src/hooks/useSnapshot.js`
- `src/hooks/useBacktestSelection.js`
- `src/hooks/useExplorerState.js`
- `src/hooks/usePageNavigation.js`
- `src/lib/display-model.js`
- `src/lib/snapshot-normalizers.js`
- `src/lib/navigation.js`
- `src/lib/dashboard-data.js`
- `src/lib/snapshot-selectors.js`
- `src/lib/formatters.js`
- `src/lib/i18n.js`

### Legacy-only component subtree

- `src/components/ErrorBoundary.legacy.jsx`
- `src/components/HeroSidebar.jsx`
- `src/components/OverviewPanel.jsx`
- `src/components/BacktestWorkbench.jsx`
- `src/components/ExplorerPanel.jsx`
- `src/components/InterfacePanel.jsx`
- `src/components/PageHeader.jsx`
- `src/components/ui.jsx`
- `src/components/charts.jsx`
- `src/components/ArtifactDetail.jsx`
- `src/components/FieldExplorer.jsx`
- `src/components/explorer/ArtifactListPane.jsx`
- `src/components/explorer/ArtifactInspectorPane.jsx`
- `src/components/explorer/ExplorerBreadcrumbs.jsx`
- `src/components/explorer/ExplorerHero.jsx`
- `src/components/explorer/BreadcrumbBar.jsx`
- `src/components/explorer/DomainRail.jsx`
- `src/components/explorer/ExplorerDomainPane.jsx`

### Keep in first-pass decommission

这些文件虽然含 legacy 文案键，但仍被当前主链或共享契约使用，**第一轮不要删**：

- `src/utils/dictionary.ts`
- `src/utils/formatters.ts`
- `src/contracts/dashboard-term-contract.json`
- `src/index.css`

---

## File Map

### Files to create

- `src/legacy-isolation.test.js` — 生产主链不得 import legacy sidecar subtree 的隔离契约测试
- `system/docs/LEGACY_SIDECAR_DECOMMISSION.md` — 简短下线记录（删了什么、保留了什么、回滚点）

### Files to modify

- `src/legacy-entry.test.js` — 保留 production entry / legacy sidecar 冻结契约

### Files to delete (phase-gated)

**Phase A: entry pair**
- `src/main.legacy.jsx`
- `src/App.legacy.jsx`

**Phase B: legacy hooks + libs**
- `src/hooks/useSnapshot.js`
- `src/hooks/useBacktestSelection.js`
- `src/hooks/useExplorerState.js`
- `src/hooks/usePageNavigation.js`
- `src/lib/display-model.js`
- `src/lib/snapshot-normalizers.js`
- `src/lib/navigation.js`
- `src/lib/dashboard-data.js`
- `src/lib/snapshot-selectors.js`
- `src/lib/formatters.js`
- `src/lib/i18n.js`

**Phase C: legacy components**
- `src/components/ErrorBoundary.legacy.jsx`
- `src/components/HeroSidebar.jsx`
- `src/components/OverviewPanel.jsx`
- `src/components/BacktestWorkbench.jsx`
- `src/components/ExplorerPanel.jsx`
- `src/components/InterfacePanel.jsx`
- `src/components/PageHeader.jsx`
- `src/components/ui.jsx`
- `src/components/charts.jsx`
- `src/components/ArtifactDetail.jsx`
- `src/components/FieldExplorer.jsx`
- `src/components/explorer/ArtifactListPane.jsx`
- `src/components/explorer/ArtifactInspectorPane.jsx`
- `src/components/explorer/ExplorerBreadcrumbs.jsx`
- `src/components/explorer/ExplorerHero.jsx`
- `src/components/explorer/BreadcrumbBar.jsx`
- `src/components/explorer/DomainRail.jsx`
- `src/components/explorer/ExplorerDomainPane.jsx`

### Files/tests to delete or update after removal

- `src/legacy-copy-contract.test.js`
- `src/lib/display-model.test.js`
- `src/lib/snapshot-normalizers.test.js`
- `src/legacy-entry.test.js`（保留或收缩为 production-entry-only）
- 任何仍显式读取上述 legacy 文件的 copy contract / entry contract test

---

### Task 1: Lock production-vs-legacy import isolation

**Files:**
- Create: `src/legacy-isolation.test.js`
- Modify: `src/legacy-entry.test.js`
- Test: `src/legacy-isolation.test.js`

- [ ] **Step 1: Write the failing isolation test**

要求：
- 扫描 `src/main.tsx`、`src/App.tsx`、以及当前 active TS/TSX 主链文件
- 断言它们不 import `App.legacy.jsx`、`main.legacy.jsx`、legacy hook/lib subtree、legacy component subtree
- 允许测试文件与 `main.legacy.jsx` 自己引用 legacy 文件

- [ ] **Step 2: Run test to verify it fails if isolation is broken**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/legacy-isolation.test.js --reporter=dot
```

Expected:
- 初次若没有实现则失败
- 若实现正确则在后续阶段作为防回归护栏

- [ ] **Step 3: Implement minimal scan logic**

实现要点：
- 用 `readFileSync + rg-like regex` 即可，不引入 AST 解析
- 允许白名单：
  - `src/main.legacy.jsx`
  - `src/App.legacy.jsx`
  - `**/*.test.*`

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/legacy-isolation.test.js src/legacy-entry.test.js --reporter=dot
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/legacy-isolation.test.js src/legacy-entry.test.js
git commit -m "test: lock legacy sidecar isolation"
```

---

### Task 2: Remove frozen legacy entry pair

**Files:**
- Delete: `src/main.legacy.jsx`
- Delete: `src/App.legacy.jsx`
- Modify: `src/legacy-entry.test.js`
- Test: `src/legacy-entry.test.js`

- [ ] **Step 1: Write/update the failing test**

把 `legacy-entry.test.js` 从“legacy entry 仍存在”改成：
- 生产入口必须指向 `src/main.tsx`
- legacy entry pair 不再要求存在

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/legacy-entry.test.js --reporter=dot
```

Expected:
- FAIL，因为文件尚未删除 / 契约尚未更新

- [ ] **Step 3: Delete the legacy entry pair and update test**

- 删除 `src/main.legacy.jsx`
- 删除 `src/App.legacy.jsx`
- 让 `legacy-entry.test.js` 只验证当前生产入口

- [ ] **Step 4: Run tests**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/legacy-entry.test.js src/legacy-isolation.test.js --reporter=dot
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/legacy-entry.test.js src/legacy-isolation.test.js
git rm src/main.legacy.jsx src/App.legacy.jsx
git commit -m "refactor: remove legacy sidecar entry pair"
```

---

### Task 3: Remove legacy hook/lib data chain

**Files:**
- Delete: `src/hooks/useSnapshot.js`
- Delete: `src/hooks/useBacktestSelection.js`
- Delete: `src/hooks/useExplorerState.js`
- Delete: `src/hooks/usePageNavigation.js`
- Delete: `src/lib/display-model.js`
- Delete: `src/lib/snapshot-normalizers.js`
- Delete: `src/lib/navigation.js`
- Delete: `src/lib/dashboard-data.js`
- Delete: `src/lib/snapshot-selectors.js`
- Delete: `src/lib/formatters.js`
- Delete: `src/lib/i18n.js`
- Delete/Modify: `src/lib/display-model.test.js`
- Delete/Modify: `src/lib/snapshot-normalizers.test.js`
- Test: `src/legacy-isolation.test.js`

- [ ] **Step 1: Update tests to fail on stale legacy imports/files**

让相关 test 改成：
- 不再要求 legacy lib 文件存在
- 或直接删除这些 legacy-only test 文件

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/lib/display-model.test.js src/lib/snapshot-normalizers.test.js src/legacy-isolation.test.js --reporter=dot
```

Expected:
- FAIL，直到 legacy lib/test 同步收口

- [ ] **Step 3: Delete the legacy hook/lib chain**

注意：
- 删除前再次确认无 active import
- `src/contracts/dashboard-term-contract.json` 暂不删 key

- [ ] **Step 4: Run focused regression**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/legacy-isolation.test.js src/App.test.tsx src/search/search-ui.test.tsx src/graph/model.test.ts --reporter=dot
npx tsc --noEmit
```

Expected:
- PASS

- [ ] **Step 5: Commit**

```bash
git add src/legacy-isolation.test.js src/legacy-entry.test.js
git rm src/hooks/useSnapshot.js src/hooks/useBacktestSelection.js src/hooks/useExplorerState.js src/hooks/usePageNavigation.js
git rm src/lib/display-model.js src/lib/snapshot-normalizers.js src/lib/navigation.js src/lib/dashboard-data.js src/lib/snapshot-selectors.js src/lib/formatters.js src/lib/i18n.js
git rm src/lib/display-model.test.js src/lib/snapshot-normalizers.test.js
git commit -m "refactor: remove legacy sidecar data chain"
```

---

### Task 4: Remove legacy component subtree

**Files:**
- Delete: `src/components/ErrorBoundary.legacy.jsx`
- Delete: `src/components/HeroSidebar.jsx`
- Delete: `src/components/OverviewPanel.jsx`
- Delete: `src/components/BacktestWorkbench.jsx`
- Delete: `src/components/ExplorerPanel.jsx`
- Delete: `src/components/InterfacePanel.jsx`
- Delete: `src/components/PageHeader.jsx`
- Delete: `src/components/ui.jsx`
- Delete: `src/components/charts.jsx`
- Delete: `src/components/ArtifactDetail.jsx`
- Delete: `src/components/FieldExplorer.jsx`
- Delete: `src/components/explorer/ArtifactListPane.jsx`
- Delete: `src/components/explorer/ArtifactInspectorPane.jsx`
- Delete: `src/components/explorer/ExplorerBreadcrumbs.jsx`
- Delete: `src/components/explorer/ExplorerHero.jsx`
- Delete: `src/components/explorer/BreadcrumbBar.jsx`
- Delete: `src/components/explorer/DomainRail.jsx`
- Delete: `src/components/explorer/ExplorerDomainPane.jsx`
- Delete/Modify: `src/legacy-copy-contract.test.js`
- Test: `src/legacy-isolation.test.js`

- [ ] **Step 1: Update/remove failing legacy copy contract**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/legacy-copy-contract.test.js --reporter=dot
```

Expected:
- FAIL or become obsolete after file deletion

- [ ] **Step 2: Delete legacy component subtree**

原则：
- 不动 TS 主链组件
- 只删 `App.legacy.jsx` 独占依赖

- [ ] **Step 3: Run focused regression**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/legacy-isolation.test.js src/App.test.tsx src/search/search-ui.test.tsx src/graph/model.test.ts src/adapters/read-model.test.ts --reporter=dot
npx tsc --noEmit
```

Expected:
- PASS

- [ ] **Step 4: Commit**

```bash
git add src/legacy-isolation.test.js
git rm src/legacy-copy-contract.test.js
git rm src/components/ErrorBoundary.legacy.jsx src/components/HeroSidebar.jsx src/components/OverviewPanel.jsx src/components/BacktestWorkbench.jsx src/components/ExplorerPanel.jsx src/components/InterfacePanel.jsx src/components/PageHeader.jsx src/components/ui.jsx src/components/charts.jsx src/components/ArtifactDetail.jsx src/components/FieldExplorer.jsx src/components/explorer/ArtifactListPane.jsx src/components/explorer/ArtifactInspectorPane.jsx src/components/explorer/ExplorerBreadcrumbs.jsx src/components/explorer/ExplorerHero.jsx src/components/explorer/BreadcrumbBar.jsx src/components/explorer/DomainRail.jsx src/components/explorer/ExplorerDomainPane.jsx
git commit -m "refactor: remove legacy sidecar component subtree"
```

---

### Task 5: Write decommission record and final verification

**Files:**
- Create: `system/docs/LEGACY_SIDECAR_DECOMMISSION.md`
- Modify: `system/docs/PROGRESS.md`
- Test: full frontend regression

- [ ] **Step 1: Write the decommission record**

记录：
- 删除了哪些文件
- 保留了哪些共享契约
- 为什么 `dashboard-term-contract.json` 暂不清 unused legacy keys
- 回滚点

- [ ] **Step 2: Update progress handoff**

在 `system/docs/PROGRESS.md` 写入：
- legacy sidecar 已下线
- 生产入口唯一主链

- [ ] **Step 3: Run full verification**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm test
npx tsc --noEmit
```

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_run_dashboard_public_topology_smoke_script.py tests/test_run_dashboard_workspace_artifacts_smoke_script.py tests/test_run_dashboard_public_acceptance_script.py
```

Expected:
- 全绿

- [ ] **Step 4: Commit**

```bash
git add /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/LEGACY_SIDECAR_DECOMMISSION.md /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/PROGRESS.md
git commit -m "docs: record legacy sidecar decommission"
```

---

## Notes / Non-Goals

- 第一轮**不要**顺手清理 `dashboard-term-contract.json` 中所有 `legacy_*` 键；这属于低风险后续清理，不应阻塞 legacy sidecar 下线。
- 第一轮**不要**改动 `src/App.tsx` 的 legacy hash bridge；它仍是线上兼容契约。
- 任何一步只要出现 active runtime import legacy file，即先修隔离，再继续删除。

## Recommended execution order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5

## Rollback strategy

- 回滚点 1：保留 `main.legacy.jsx / App.legacy.jsx`，不进入删除阶段
- 回滚点 2：若 Task 3 失败，只回滚 legacy hook/lib 删除，不回滚主链 path-route 改造
- 回滚点 3：若 Task 4 失败，只恢复 legacy component subtree，不恢复已完成的生产入口冻结契约
