# Fenlie Dashboard Front-end Contract Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Roll out the approved Fenlie dashboard front-end contract so the shell, navigation, controls, page templates, readability, and responsive behavior become semantically stable, lower-friction, and easier to extend without touching live authority paths.

**Architecture:** Implement this as a contract-first UI refactor on top of the existing dashboard/web structure, keeping source-owned data flow intact while normalizing shell responsibilities and control semantics. Do not rewrite read-model or backend contracts; instead, introduce a thin semantic control layer, a consistent page rhythm, and a progressive responsive shell that preserves focus/search/deep-link behavior.

**Tech Stack:** React, TypeScript, React Router, Vitest, Testing Library, existing `src/index.css` style system, Fenlie dashboard read-model and focus-link utilities.

---

## Scope guard

- This plan applies only to `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web`.
- Change class stays `RESEARCH_ONLY` throughout implementation.
- Do not touch live execution, routing, risk, or source-artifact authority.
- Preserve the current search dual-entry capability; refactor it into the new contract rather than discarding it.
- If implementation begins from a dirty worktree, preserve existing dashboard/web work first via a baseline commit or dedicated worktree before changing behavior.

## File structure map

### Existing files to modify

- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/App.tsx`
  - Own top-level shell state, search toggle, route composition, inspector behavior wiring.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/AppShell.tsx`
  - Own shell region layout and responsive inspector/sidebar slots.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/GlobalTopbar.tsx`
  - Reduce topbar to global-tool + domain-tab responsibility.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/ContextSidebar.tsx`
  - Restrict sidebar to domain-local navigation and utilities.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/ContextHeader.tsx`
  - Normalize page-level state/focus/action framing.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/InspectorRail.tsx`
  - Add explicit empty/pinned/collapsed behavior contract.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/TerminalPanels.tsx`
  - Map ops page controls to semantic primitives and consistent drilldown rhythm.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/WorkspacePanels.tsx`
  - Map research page controls, artifact rails, and evidence sections to the new contract.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/ui-kit.tsx`
  - Tighten `ClampText` policy and shared semantic helpers.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/pages/OverviewPage.tsx`
  - Rebuild overview first-screen rhythm around state/focus/action.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/pages/OpsPage.tsx`
  - Keep route shell thin, enforce ops template contract.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/pages/ResearchPage.tsx`
  - Keep route shell thin, enforce research template contract.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/pages/SearchPage.tsx`
  - Keep search as a locator page with clear state/focus behavior.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/search/SearchOverlay.tsx`
  - Keep overlay focused on query/scope/results, not page explanation.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/search/SearchResults.tsx`
  - Introduce stable result-row semantics and explicit match explanation.
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/index.css`
  - Split style behavior by semantic role, density tier, and responsive shell rules.

### New files to create

- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/control-primitives.tsx`
  - Shared semantic control wrappers: domain tabs, segmented controls, filter chips, action buttons, entity rows.
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/control-primitives.test.tsx`
  - Tests that semantic wrappers expose distinct roles, classes, and keyboard expectations.
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/hooks/use-shell-breakpoint.ts`
  - Encapsulate shell breakpoint tiers (`xl/l/m/s`) and medium-screen inspector mode logic.
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/hooks/use-shell-breakpoint.test.tsx`
  - Tests for breakpoint transitions and cleanup.

### Existing tests to update

- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/App.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/AppShell.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/GlobalTopbar.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/ContextSidebar.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/InspectorRail.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/search/search-ui.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/ui-kit.test.tsx`

---

### Task 1: Preflight baseline and lock the refactor boundary

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/docs/superpowers/plans/2026-03-28-fenlie-dashboard-front-end-contract-rollout.md`
- Inspect only: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/App.tsx`
- Inspect only: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/index.css`
- Test: existing dashboard/web test suite entrypoints

- [ ] **Step 1: Capture the current dashboard/web baseline**

Run:

```bash
git -C /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1 status --short
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npx tsc --noEmit
npm test -- --run src/App.test.tsx src/search/search-ui.test.tsx src/app-shell/AppShell.test.tsx src/app-shell/GlobalTopbar.test.tsx src/app-shell/ContextSidebar.test.tsx src/app-shell/InspectorRail.test.tsx
```

Expected:
- `tsc` passes
- targeted tests pass
- current dirty dashboard/web state is visible and intentionally preserved

- [ ] **Step 2: Record scope guard notes in the plan file before implementation begins**

Add/update a short execution note under the plan header:

```md
Implementation must preserve existing global search behavior and avoid any read-model or live-path changes.
```

- [ ] **Step 3: Create a baseline checkpoint commit or dedicated implementation worktree**

Run one of:

```bash
# Option A: if current dashboard/web changes are already approved baseline
git add system/dashboard/web
git commit -m "chore(ui): checkpoint pre-contract dashboard baseline"

# Option B: if baseline should stay isolated
# create a fresh worktree from current branch head before starting rollout
```

Expected:
- implementation starts from a recoverable checkpoint

- [ ] **Step 4: Commit the preflight note change (if any)**

```bash
git add docs/superpowers/plans/2026-03-28-fenlie-dashboard-front-end-contract-rollout.md
git commit -m "docs(ui): note contract rollout scope guard"
```

### Task 2: Introduce semantic control primitives

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/control-primitives.tsx`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/control-primitives.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/index.css`
- Test: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/control-primitives.test.tsx`

- [ ] **Step 1: Write the failing tests for semantic control roles**

```tsx
it('renders domain tabs separately from action buttons and filter chips', () => {
  render(
    <>
      <DomainTab to="/overview" active>总览</DomainTab>
      <SegmentedOption active onClick={() => undefined}>public</SegmentedOption>
      <ActionButton onClick={() => undefined}>打开搜索</ActionButton>
      <FilterChip active onClick={() => undefined}>warning</FilterChip>
      <EntityRowButton title="持有选择主头" subtitle="research_hold_transfer" onClick={() => undefined} />
    </>
  );

  expect(screen.getByRole('link', { name: '总览' }).className).toContain('control-domain-tab');
  expect(screen.getByRole('button', { name: 'public' }).className).toContain('control-segment-option');
  expect(screen.getByRole('button', { name: '打开搜索' }).className).toContain('control-action-button');
  expect(screen.getByRole('button', { name: 'warning' }).className).toContain('control-filter-chip');
  expect(screen.getByRole('button', { name: /持有选择主头/i }).className).toContain('control-entity-row');
});
```

- [ ] **Step 2: Run the new test and verify it fails because the new primitives do not exist yet**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- --run src/components/control-primitives.test.tsx
```

Expected: FAIL with missing module/export errors.

- [ ] **Step 3: Implement the minimal semantic wrapper layer**

```tsx
export function DomainTab(props: NavLinkProps & { active?: boolean }) {
  return <NavLink {...props} className={joinControlClass('control-domain-tab', props.className)} />;
}

export function SegmentedOption(props: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return <button {...props} data-active={props.active ? 'true' : 'false'} className={joinControlClass('control-segment-option', props.className)} />;
}

export function ActionButton(props: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button {...props} className={joinControlClass('control-action-button', props.className)} />;
}

export function FilterChip(props: ButtonHTMLAttributes<HTMLButtonElement> & { active?: boolean }) {
  return <button {...props} data-active={props.active ? 'true' : 'false'} className={joinControlClass('control-filter-chip', props.className)} />;
}

export function EntityRowButton(props: {
  title: string;
  subtitle?: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button type="button" data-active={props.active ? 'true' : 'false'} className="control-entity-row" onClick={props.onClick}>
      <span>{props.title}</span>
      {props.subtitle ? <small>{props.subtitle}</small> : null}
    </button>
  );
}
```

Also add matching CSS groups in `src/index.css`:

```css
.control-domain-tab { /* primary context switch */ }
.control-segment-option { /* mutually exclusive view mode */ }
.control-filter-chip { /* lightweight filtering */ }
.control-action-button { /* primary action */ }
.control-entity-row { /* selectable object row */ }
```

Required mapping in implementation:

- `DomainTab`: 一级域切换
- `SegmentedOption`: public/internal、scope、search_scope 这类互斥视图切换
- `FilterChip`: tone/filter/tag
- `ActionButton` / `ActionLink`: 打开搜索、查看全部结果、进入源头主线
- `EntityRowButton`: search result、artifact row、topology/entity selection

- [ ] **Step 4: Re-run the primitive tests and make them pass**

Run:

```bash
npm test -- --run src/components/control-primitives.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit the semantic primitive foundation**

```bash
git add src/components/control-primitives.tsx src/components/control-primitives.test.tsx src/index.css
git commit -m "feat(ui): add semantic control primitives"
```

### Task 3: Shrink topbar to global responsibilities only

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/GlobalTopbar.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/App.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/GlobalTopbar.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/search/search-ui.test.tsx`

- [ ] **Step 1: Write failing tests for the new topbar contract**

Add tests asserting:

```tsx
expect(screen.queryByText('职责')).toBeNull();
expect(screen.getByRole('navigation', { name: 'primary-domains' })).toBeTruthy();
expect(screen.getByRole('button', { name: '打开全局搜索' })).toBeTruthy();
expect(screen.getByText(/快照|只读|generated/i)).toBeTruthy();
```

- [ ] **Step 2: Run the topbar tests and verify the current implementation still exposes page-local summary chips**

Run:

```bash
npm test -- --run src/app-shell/GlobalTopbar.test.tsx src/search/search-ui.test.tsx
```

Expected: FAIL because topbar still mixes page-level summary and domain description.

- [ ] **Step 3: Refactor `GlobalTopbar` to four fixed blocks**

Implement a structure like:

```tsx
<header>
  <BrandIdentity />
  <PrimaryDomainTabs />
  <GlobalToolStrip>
    <SearchTrigger />
    <ThemeSwitcher />
  </GlobalToolStrip>
  <EnvironmentStrip>
    <SnapshotFreshness />
    <SurfaceMode />
    <ChangeClass />
  </EnvironmentStrip>
</header>
```

Rules:
- move page-local explanation out of topbar
- keep domain tabs stable across routes
- keep `Cmd/Ctrl+K` and button toggle behavior unchanged

- [ ] **Step 4: Update route composition in `App.tsx` to pass only global summary data**

Use inputs like:

```tsx
globalSummary={{
  changeClass: labelFor(String(model?.meta.change_class || 'RESEARCH_ONLY')),
  chips: [
    { label: '模式', value: labelFor(String(model?.experienceContract.mode || 'read_only_snapshot')) },
    { label: '快照', value: formatDateTime(model?.meta.generated_at_utc) },
  ],
}}
```

Do not pass page-local “职责/当前页” chips anymore.

- [ ] **Step 5: Re-run topbar/search tests and verify they pass**

Run:

```bash
npm test -- --run src/app-shell/GlobalTopbar.test.tsx src/search/search-ui.test.tsx src/App.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit the topbar responsibility split**

```bash
git add src/app-shell/GlobalTopbar.tsx src/App.tsx src/app-shell/GlobalTopbar.test.tsx src/search/search-ui.test.tsx
git commit -m "refactor(ui): reduce topbar to global responsibilities"
```

### Task 4: Restrict sidebar to domain-local navigation only

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/ContextSidebar.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/ContextSidebar.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/index.css`

- [ ] **Step 1: Write failing tests that lock the sidebar to domain-local responsibilities**

Add assertions that:

```tsx
expect(within(utilities).queryByText(/变更级别/)).toBeNull();
expect(within(nav).getAllByRole('link').length).toBeGreaterThan(0);
expect(within(utilities).getByRole('link', { name: '搜索 / Search' })).toBeTruthy();
```

- [ ] **Step 2: Run sidebar tests to verify the old sidebar still behaves as a mixed summary/navigation block**

Run:

```bash
npm test -- --run src/app-shell/ContextSidebar.test.tsx
```

Expected: FAIL for at least one contract expectation.

- [ ] **Step 3: Refactor `ContextSidebar` into three explicit regions**

```tsx
<SidebarDomainIdentity />
<SidebarTaskNav />
<SidebarUtilities />
```

Rules:
- keep collapse behavior
- keep section TOC grouped by task stage
- remove non-navigation summary semantics from sidebar copy
- keep utilities lightweight and secondary

- [ ] **Step 4: Update sidebar CSS so domain-local links and section TOC are visually distinct**

Add separate style hooks such as:

```css
.sidebar-task-nav .control-entity-row { ... }
.sidebar-page-toc .control-filter-chip { ... }
.sidebar-utilities a { ... }
```

- [ ] **Step 5: Re-run sidebar tests and make them pass**

Run:

```bash
npm test -- --run src/app-shell/ContextSidebar.test.tsx src/App.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit the sidebar refactor**

```bash
git add src/app-shell/ContextSidebar.tsx src/app-shell/ContextSidebar.test.tsx src/index.css
git commit -m "refactor(ui): restrict sidebar to domain-local navigation"
```

### Task 5: Normalize the page rhythm to state → focus → action → evidence

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/ContextHeader.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/pages/OverviewPage.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/TerminalPanels.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/WorkspacePanels.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/App.test.tsx`

- [ ] **Step 1: Add failing tests for first-screen page rhythm**

Cover:

```tsx
expect(screen.getByText(/当前状态|state/i)).toBeTruthy();
expect(screen.getByText(/当前焦点|focus/i)).toBeTruthy();
expect(screen.getByText(/下一步|action/i)).toBeTruthy();
```

Apply these assertions to overview and at least one ops/research route.

- [ ] **Step 2: Run the page tests and confirm first-screen rhythm is not yet explicit**

Run:

```bash
npm test -- --run src/App.test.tsx
```

Expected: FAIL because the current pages do not expose explicit state/focus/action structure.

- [ ] **Step 3: Refactor `ContextHeader` to expose normalized slots**

Implement a prop contract like:

```tsx
type ContextHeaderProps = {
  stateSection: ...;
  focusSection: ...;
  actionSection: ...;
  evidenceSection?: ...;
}
```

Use these slots consistently across overview/ops/research/search.

- [ ] **Step 4: Rebuild `OverviewPage` first screen around the new contract**

Target order:
- state summary
- current highest-priority line
- entry actions
- evidence cards below the fold

Keep domestic commodity reasoning visible, but place it under the explicit focus band rather than as a peer to every entry card.

- [ ] **Step 5: Update ops and research entry sections to match the same rhythm**

For ops:

```tsx
<TerminalFocusRail />
<TerminalBreadcrumbs />
<TerminalWorkspaceHandoffs />
<TerminalPanelsGrid />
```

For research:
- left rail as task-local navigation/filter
- main stage starts with current artifact focus summary before deeper evidence walls

- [ ] **Step 6: Re-run route tests and verify the new rhythm**

Run:

```bash
npm test -- --run src/App.test.tsx src/app-shell/AppShell.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit the template normalization**

```bash
git add src/app-shell/ContextHeader.tsx src/pages/OverviewPage.tsx src/components/TerminalPanels.tsx src/components/WorkspacePanels.tsx src/App.test.tsx
git commit -m "refactor(ui): normalize dashboard page rhythm"
```

### Task 6: Rebuild search and entity lists on semantic row/action separation

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/search/SearchResults.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/search/SearchOverlay.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/WorkspacePanels.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/search/search-ui.test.tsx`

- [ ] **Step 1: Write failing tests that distinguish result rows from action buttons**

```tsx
expect(screen.getByRole('button', { name: /持有选择主头/i }).className).toContain('control-entity-row');
expect(screen.getByRole('link', { name: /查看全部结果/i }).className).toContain('control-action-link');
expect(screen.getByText(/标题命中|说明命中|路径命中|关键词命中/)).toBeTruthy();
```

- [ ] **Step 2: Run search tests to confirm rows/actions still share too much styling**

Run:

```bash
npm test -- --run src/search/search-ui.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Refactor search result rows and artifact rows to use semantic row primitives**

Examples:

```tsx
<EntityRowButton
  title={result.title}
  subtitle={result.subtitle}
  badge={CATEGORY_LABELS[result.category]}
  active={activeId === result.id}
  onClick={() => onPick(result)}
/>
```

For artifact list rows, keep selected state but visually separate them from action buttons and chips.

- [ ] **Step 4: Add explicit scope/deep-link/match-explanation coverage**

Add or extend tests for:

```tsx
it('keeps scope switching explicit between all/module/route/artifact', async () => {
  render(<SearchPage ... />);
  await user.click(screen.getByRole('button', { name: '工件' }));
  expect(window.location.hash).toContain('scope=artifact');
});

it('renders match explanation text for a result and navigates with deep-link parameters intact', async () => {
  render(<App />);
  await openSearchAndQuery('国内商品推理线');
  expect(screen.getByText(/标题命中|关键词命中/)).toBeTruthy();
  await user.click(screen.getByRole('button', { name: /国内商品推理线/i }));
  expect(window.location.hash).toContain('anchor=');
});
```

Deep-link validation must cover at least:

- route result -> direct route navigation
- module result -> `anchor=` section landing
- artifact result -> `artifact=` plus optional `panel/section/row` preservation

- [ ] **Step 5: Re-run search tests and targeted research list tests**

Run:

```bash
npm test -- --run src/search/search-ui.test.tsx src/App.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit the list semantics update**

```bash
git add src/search/SearchResults.tsx src/search/SearchOverlay.tsx src/components/WorkspacePanels.tsx src/search/search-ui.test.tsx
git commit -m "refactor(ui): separate locator rows from actions"
```

### Task 7: Tighten inspector behavior and medium-screen responsive shell

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/hooks/use-shell-breakpoint.ts`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/hooks/use-shell-breakpoint.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/AppShell.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/InspectorRail.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/app-shell/InspectorRail.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/index.css`

- [ ] **Step 1: Write failing breakpoint and inspector tests**

Examples:

```tsx
it('returns medium tier between 960 and 1279 width', () => {
  mockWindowWidth(1100);
  const { result } = renderHook(() => useShellBreakpoint());
  expect(result.current.tier).toBe('m');
});

it('renders inspector as collapsible drawer on medium screens', () => {
  mockWindowWidth(1100);
  render(<AppShell ... />);
  expect(screen.getByRole('button', { name: /展开对象检查器/i })).toBeTruthy();
});
```

- [ ] **Step 2: Run breakpoint/inspector tests to confirm the behavior is not implemented yet**

Run:

```bash
npm test -- --run src/hooks/use-shell-breakpoint.test.tsx src/app-shell/InspectorRail.test.tsx src/app-shell/AppShell.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Implement `use-shell-breakpoint` and wire `AppShell` to shell tiers**

```ts
export function useShellBreakpoint() {
  // map width -> xl | l | m | s
}
```

In `AppShell.tsx`:

```tsx
const shellTier = useShellBreakpoint();
const inspectorMode = shellTier.tier === 'm' ? 'drawer' : shellTier.tier === 's' ? 'inline' : 'rail';
```

- [ ] **Step 4: Refactor `InspectorRail` to support `rail | drawer | inline` modes**

Rules:
- keep existing inspector content contract
- add explicit empty state
- allow collapse on medium screens
- never duplicate sidebar navigation responsibilities

- [ ] **Step 5: Update CSS for progressive responsive shell instead of one-shot single-column collapse**

Implement breakpoint-specific shell behavior instead of only:

```css
@media (max-width: 1380px) { grid-template-columns: 1fr; }
```

Use tiered behavior so medium widths still keep main task affordances visible.

- [ ] **Step 6: Re-run shell tests and typecheck**

Run:

```bash
npm test -- --run src/hooks/use-shell-breakpoint.test.tsx src/app-shell/AppShell.test.tsx src/app-shell/InspectorRail.test.tsx
npx tsc --noEmit
```

Expected: PASS.

- [ ] **Step 7: Commit the responsive shell contract**

```bash
git add src/hooks/use-shell-breakpoint.ts src/hooks/use-shell-breakpoint.test.tsx src/app-shell/AppShell.tsx src/app-shell/InspectorRail.tsx src/app-shell/InspectorRail.test.tsx src/index.css
git commit -m "feat(ui): add progressive responsive shell behavior"
```

### Task 8: Tighten readability and clamp policy for critical labels

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/ui-kit.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/ui-kit.test.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/search/SearchResults.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/TerminalPanels.tsx`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/components/WorkspacePanels.tsx`

- [ ] **Step 1: Write failing tests for “critical labels must remain readable”**

Examples:

```tsx
it('shows expand affordance when a critical label overflows', () => {
  mockClampOverflow({ overflows: true });
  render(<ClampText raw="超长焦点对象标题" expandable critical>{'超长焦点对象标题'}</ClampText>);
  expect(screen.getByRole('button', { name: '展开完整内容' })).toBeTruthy();
});
```

- [ ] **Step 2: Run the ui-kit tests and verify the current clamp policy is too generic**

Run:

```bash
npm test -- --run src/components/ui-kit.test.tsx
```

Expected: FAIL.

- [ ] **Step 3: Extend `ClampText` with a critical-readability contract**

Implement a small extension such as:

```tsx
type ClampTextProps = {
  critical?: boolean;
}
```

Behavior:
- if `critical`, do not force `expandable={false}` in calling sites
- preserve title tooltip
- keep layout compact unless overflow actually happens

- [ ] **Step 4: Apply critical clamp usage to high-value labels only**

Examples:
- search result titles
- current artifact title
- current focus row label
- breadcrumb current node

Avoid broad churn: only change labels that represent primary orientation or current selection.

- [ ] **Step 5: Re-run readability tests and focused route tests**

Run:

```bash
npm test -- --run src/components/ui-kit.test.tsx src/search/search-ui.test.tsx src/App.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit the readability refinement**

```bash
git add src/components/ui-kit.tsx src/components/ui-kit.test.tsx src/search/SearchResults.tsx src/components/TerminalPanels.tsx src/components/WorkspacePanels.tsx
git commit -m "refactor(ui): improve readability for critical labels"
```

### Task 9: Final verification, dist check, and docs alignment

**Files:**
- Modify only if needed: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/docs/superpowers/specs/2026-03-28-fenlie-dashboard-front-end-contract-design.md`
- Modify only if needed: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/docs/superpowers/plans/2026-03-28-fenlie-dashboard-front-end-contract-rollout.md`

- [ ] **Step 1: Run the full dashboard/web validation pack**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npx tsc --noEmit
npm test
npm run build
```

Expected:
- all tests pass
- production build succeeds

- [ ] **Step 2: Run focused browser acceptance manually**

Check these flows in local preview:

1. topbar only shows global tools + domain tabs + environment strip
2. sidebar only shows domain-local nav + TOC + utilities
3. overview first screen answers state/focus/action
4. ops and research keep search/deep-link/focus behavior intact
5. medium-width viewport keeps core affordances; inspector becomes drawer instead of disappearing
6. critical labels expose expand affordance when truncated

- [ ] **Step 3: If contract drift was discovered, update the plan/spec docs minimally**

Only document true implementation constraints; do not widen scope.

- [ ] **Step 4: Create the final rollout commit**

```bash
git add system/dashboard/web docs/superpowers/specs/2026-03-28-fenlie-dashboard-front-end-contract-design.md docs/superpowers/plans/2026-03-28-fenlie-dashboard-front-end-contract-rollout.md
git commit -m "refactor(ui): roll out dashboard front-end contract"
```

- [ ] **Step 5: Capture rollback point and acceptance note**

Record:
- final commit SHA
- whether any unresolved responsive/readability edge case remains
- confirmation that live path stayed untouched

---

## Suggested execution order

Use this exact order:

1. Preflight
2. Semantic controls
3. Topbar split
4. Sidebar split
5. Page rhythm normalization
6. Search/list semantics
7. Responsive shell + inspector
8. Readability/clamp refinement
9. Final verification

Do not reorder responsive work ahead of shell semantics. Do not widen into read-model or backend cleanup.

---

## Verification summary

Minimum required before completion:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npx tsc --noEmit
npm test
npm run build
```

If any step introduces behavior drift in search/deep-link/focus traversal, stop and fix before proceeding.
