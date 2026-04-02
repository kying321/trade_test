# [Fenlie CPA Entry Visibility] Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the CPA manager obvious from the Fenlie shell by adding high-visibility entry points in the topbar, overview landing area, graph-home quick actions, and sidebar utilities.

**Architecture:** Keep routing unchanged at `/cpa`, and increase discoverability entirely through existing shell primitives. Use test-first changes in the small set of shell/page components that already own navigation visibility.

**Tech Stack:** React, TypeScript, React Router, Vitest, Testing Library.

---

### Task 1: Lock failing tests for visible CPA entry points

**Files:**
- Modify: `system/dashboard/web/src/app-shell/GlobalTopbar.test.tsx`
- Modify: `system/dashboard/web/src/app-shell/ContextSidebar.test.tsx`
- Modify: `system/dashboard/web/src/pages/GraphHomePage.test.tsx`
- Modify: `system/dashboard/web/src/App.test.tsx`

- [ ] **Step 1: Write the failing test for topbar CPA action**
- [ ] **Step 2: Run the focused topbar/sidebar tests and confirm failure**
- [ ] **Step 3: Write the failing test for overview CPA card + graph-home CPA quick entry**
- [ ] **Step 4: Run the focused page/app tests and confirm failure**

### Task 2: Implement the CPA visibility upgrades

**Files:**
- Modify: `system/dashboard/web/src/app-shell/GlobalTopbar.tsx`
- Modify: `system/dashboard/web/src/app-shell/ContextSidebar.tsx`
- Modify: `system/dashboard/web/src/pages/OverviewPage.tsx`
- Modify: `system/dashboard/web/src/pages/GraphHomePage.tsx`
- Optional style touch only if needed: `system/dashboard/web/src/index.css`

- [ ] **Step 1: Add topbar CPA quick action with explicit copy**
- [ ] **Step 2: Add CPA fixed utility link to the sidebar**
- [ ] **Step 3: Add CPA emphasized entry card to the overview next-step grid**
- [ ] **Step 4: Add CPA quick entry to graph-home quick actions**
- [ ] **Step 5: Add only the minimum styling needed to make the entry obvious but consistent**

### Task 3: Verify end-to-end shell discoverability

**Files:**
- Test only

- [ ] **Step 1: Run `vitest` for the touched tests**
- [ ] **Step 2: If needed, run one app-level smoke for `/cpa` route rendering**
- [ ] **Step 3: Confirm no regression in existing CPA route accessibility**
