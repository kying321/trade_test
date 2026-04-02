# [Fenlie Dashboard Path Routing + Ops IA] Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the Fenlie dashboard from hash routing to path routing, add the first `/ops/*` route family, and preserve legacy deep-link compatibility without touching any live execution path.

**Architecture:** Reuse the existing shell and read model, switch the router to path-based navigation, introduce `/ops/*` page ownership, and keep `/terminal/*` plus `/workspace/*` compatibility through redirects or alias routes. Normalize any legacy `#/...` route payloads on the frontend instead of demanding a snapshot schema change.

**Tech Stack:** React 19, React Router, TypeScript, Vitest, Testing Library, Vite 7.

---

### Task 1: Lock failing routing tests first

**Files:**
- Modify: `system/dashboard/web/src/App.test.tsx`
- Modify: `system/dashboard/web/src/pages/GraphHomePage.test.tsx`
- Modify: `system/dashboard/web/src/search/search-ui.test.tsx`

- [ ] **Step 1: Add failing tests for path-first app routing**
- [ ] **Step 2: Add failing tests for legacy hash redirect**
- [ ] **Step 3: Add failing tests for new `/ops/overview` and `/ops/risk` navigation targets**
- [ ] **Step 4: Run focused tests and confirm they fail for the expected reason**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx src/pages/GraphHomePage.test.tsx src/search/search-ui.test.tsx --reporter=dot
```

Expected: FAIL because the app still depends on `HashRouter` and current tests assert `window.location.hash`.

### Task 2: Introduce path routing and legacy hash bridge

**Files:**
- Modify: `system/dashboard/web/src/App.tsx`
- Modify: `system/dashboard/web/src/navigation/route-contract.ts`
- Modify: `system/dashboard/web/src/utils/focus-links.ts`

- [ ] **Step 1: Replace `HashRouter` with `BrowserRouter`**
- [ ] **Step 2: Add startup normalization for `#/...` legacy entry URLs**
- [ ] **Step 3: Keep shared query/focus preservation behavior unchanged**
- [ ] **Step 4: Run focused tests and confirm router behavior turns green**

### Task 3: Add `/ops/*` route family and keep legacy aliases

**Files:**
- Modify: `system/dashboard/web/src/App.tsx`
- Modify: `system/dashboard/web/src/navigation/nav-config.ts`
- Create: `system/dashboard/web/src/pages/ops/OpsOverviewPage.tsx`
- Create: `system/dashboard/web/src/pages/ops/OpsRiskPage.tsx`
- Create: `system/dashboard/web/src/pages/ops/OpsAuditsPage.tsx`
- Create: `system/dashboard/web/src/pages/ops/OpsRunbooksPage.tsx`
- Create: `system/dashboard/web/src/pages/ops/OpsWorkflowPage.tsx`
- Create: `system/dashboard/web/src/pages/ops/OpsTracesPage.tsx`

- [ ] **Step 1: Create minimal ops page modules**
- [ ] **Step 2: Route `/ops/overview` to the concise overview surface**
- [ ] **Step 3: Route `/ops/risk` to the existing terminal public observation surface**
- [ ] **Step 4: Add placeholder shells for audits/runbooks/workflow/traces**
- [ ] **Step 5: Redirect `/overview` and `/terminal/public` to the new ops routes while preserving query context**

### Task 4: Update navigation and route consumers

**Files:**
- Modify: `system/dashboard/web/src/navigation/nav-config.ts`
- Modify: `system/dashboard/web/src/pages/GraphHomePage.tsx`
- Modify: `system/dashboard/web/src/search/search-ui.test.tsx`
- Modify any route normalizers that still assume hash-only links

- [ ] **Step 1: Point primary navigation at `/ops/overview` and `/ops/risk`**
- [ ] **Step 2: Ensure graph/search/deep-link consumers can consume either legacy `#/...` or modern `/...` routes**
- [ ] **Step 3: Keep legacy workspace routes functioning unchanged in this phase**

### Task 5: Minimal verification and docs sync

**Files:**
- Modify: `system/docs/PROGRESS.md` (only if implementation lands and evidence is fresh)

- [ ] **Step 1: Run the focused vitest suite for touched files**
- [ ] **Step 2: Run `npm test` if the focused suite passes cleanly**
- [ ] **Step 3: Run `npx tsc --noEmit`**
- [ ] **Step 4: Record only fresh, truthful verification evidence in the final report**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx src/pages/GraphHomePage.test.tsx src/search/search-ui.test.tsx --reporter=dot
npm test
npx tsc --noEmit
```

Expected: All touched frontend route tests pass, the app no longer depends on hash routing, and legacy hash entry URLs redirect into path-based routes.

## Notes
- No subagents: current repo instructions only allow delegation when the user explicitly asks for it.
- No live-path changes: this plan is confined to frontend IA/routing behavior.
