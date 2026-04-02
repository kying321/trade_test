# Legacy Sidecar Decommission Record

## Scope

本次下线只针对已经脱离生产主链的 legacy sidecar：

- 旧入口对：
  - `system/dashboard/web/src/main.legacy.jsx`
  - `system/dashboard/web/src/App.legacy.jsx`
- legacy hook/lib data chain
- legacy component subtree

生产入口保持不变：

- `system/dashboard/web/index.html`
- `system/dashboard/web/src/main.tsx`
- `system/dashboard/web/src/App.tsx`

## What Was Removed

### Legacy entry pair

- `system/dashboard/web/src/main.legacy.jsx`
- `system/dashboard/web/src/App.legacy.jsx`

### Legacy hook/lib data chain

- `system/dashboard/web/src/hooks/useSnapshot.js`
- `system/dashboard/web/src/hooks/useBacktestSelection.js`
- `system/dashboard/web/src/hooks/useExplorerState.js`
- `system/dashboard/web/src/hooks/usePageNavigation.js`
- `system/dashboard/web/src/lib/display-model.js`
- `system/dashboard/web/src/lib/snapshot-normalizers.js`
- `system/dashboard/web/src/lib/navigation.js`
- `system/dashboard/web/src/lib/dashboard-data.js`
- `system/dashboard/web/src/lib/snapshot-selectors.js`
- `system/dashboard/web/src/lib/formatters.js`
- `system/dashboard/web/src/lib/i18n.js`

### Legacy component subtree

- `system/dashboard/web/src/components/ErrorBoundary.legacy.jsx`
- `system/dashboard/web/src/components/HeroSidebar.jsx`
- `system/dashboard/web/src/components/OverviewPanel.jsx`
- `system/dashboard/web/src/components/BacktestWorkbench.jsx`
- `system/dashboard/web/src/components/ExplorerPanel.jsx`
- `system/dashboard/web/src/components/InterfacePanel.jsx`
- `system/dashboard/web/src/components/PageHeader.jsx`
- `system/dashboard/web/src/components/ui.jsx`
- `system/dashboard/web/src/components/charts.jsx`
- `system/dashboard/web/src/components/ArtifactDetail.jsx`
- `system/dashboard/web/src/components/FieldExplorer.jsx`
- `system/dashboard/web/src/components/explorer/ArtifactListPane.jsx`
- `system/dashboard/web/src/components/explorer/ArtifactInspectorPane.jsx`
- `system/dashboard/web/src/components/explorer/ExplorerBreadcrumbs.jsx`
- `system/dashboard/web/src/components/explorer/ExplorerHero.jsx`
- `system/dashboard/web/src/components/explorer/BreadcrumbBar.jsx`
- `system/dashboard/web/src/components/explorer/DomainRail.jsx`
- `system/dashboard/web/src/components/explorer/ExplorerDomainPane.jsx`

### Legacy-only tests removed or narrowed

- `system/dashboard/web/src/legacy-copy-contract.test.js`
- `system/dashboard/web/src/lib/display-model.test.js`
- `system/dashboard/web/src/lib/snapshot-normalizers.test.js`

## What Was Kept

这些文件仍属于当前生产主链或共享契约，不在本次删除范围：

- `system/dashboard/web/src/App.tsx`
- `system/dashboard/web/src/main.tsx`
- `system/dashboard/web/src/utils/dictionary.ts`
- `system/dashboard/web/src/utils/formatters.ts`
- `system/dashboard/web/src/contracts/dashboard-term-contract.json`
- `system/dashboard/web/src/index.css`

## Guardrails Added

新增 runtime 隔离护栏：

- `system/dashboard/web/src/legacy-isolation.js`
- `system/dashboard/web/src/legacy-isolation.test.js`

作用：

- 从 `src/main.tsx` 出发递归收集 active runtime import graph
- 防止未来把 legacy sidecar subtree 重新接回生产主链

## Why legacy contract keys were not cleaned now

`dashboard-term-contract.json` 里仍可能保留一部分 `legacy_*` key。

原因：

1. 这些 key 已不再驱动 active runtime
2. 删除 key 属于低风险二次清理，不应阻塞 sidecar 下线
3. 先完成入口/组件/数据链删除，再做 contract key prune，更容易审计回滚

## Verification Snapshot

本轮下线后至少完成以下回归：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/legacy-isolation.test.js src/legacy-entry.test.js src/App.test.tsx src/search/search-ui.test.tsx src/graph/model.test.ts src/adapters/read-model.test.ts --reporter=dot
npx tsc --noEmit
```

以及完整前端与 Python 验收：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm test
npx tsc --noEmit

cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_run_dashboard_public_topology_smoke_script.py tests/test_run_dashboard_workspace_artifacts_smoke_script.py tests/test_run_dashboard_public_acceptance_script.py
```

## Rollback

如果需要回滚 legacy sidecar，下线回滚应按以下顺序恢复：

1. 恢复 `main.legacy.jsx` 与 `App.legacy.jsx`
2. 恢复 legacy hook/lib data chain
3. 恢复 legacy component subtree
4. 恢复 `legacy-copy-contract.test.js`、`display-model.test.js`、`snapshot-normalizers.test.js`
5. 最后根据需要收窄或移除 `legacy-isolation` 护栏

不应回滚：

- BrowserRouter/path-route 主链改造
- `App.tsx` 中现有 legacy hash bridge 兼容逻辑
