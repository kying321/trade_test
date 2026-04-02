# Fenlie

> A read-first trading control system for research, risk, routing, and operator workflows.

Fenlie is a trading operations and research platform that keeps **source-owned state** at the center of the system.  
It is designed to separate:

- research vs. execution
- operator surfaces vs. source-of-truth artifacts
- public dashboards vs. internal-only evidence

The current repository contains:

- a Python research / orchestration core
- a React dashboard for operator-facing control surfaces
- graph-based workflow navigation
- artifact-driven audit and acceptance paths
- a newly integrated **CPA management page** for authentication file and quota operations

---

## Highlights

- **Graph Home**  
  A graph-first homepage for navigating pipeline stages, routes, source heads, acceptance chains, and custom pipelines.

- **Read-only operator shell**  
  The dashboard is explicitly structured around read-only and source-owned views before any execution-path escalation.

- **Research workspace**  
  Artifact pools, alignment views, contracts, raw snapshots, and backtest surfaces are all addressable via stable routes.

- **CPA manager**  
  Fenlie now includes a dedicated `/cpa` management page for:
  - Codex auth-file discovery
  - quota checks
  - 401 / 402 cleanup
  - structural-error bucket deletion
  - local error-ledger based delete protection

- **Artifact-driven verification**  
  Public acceptance, workspace route smoke checks, graph-home smoke checks, and dashboard read-model behavior all have test coverage.

---

## Repository Layout

```text
fenlie/
├── system/                  # Core engine, scripts, dashboard, tests
│   ├── dashboard/web/       # React operator UI
│   ├── src/lie_engine/      # Research, data, orchestration core
│   ├── scripts/             # Builders, refreshers, acceptance runners
│   └── tests/               # Python verification
├── docs/                    # Investor / architecture / process docs
├── output/                  # Runtime/output material
└── AGENTS.md                # Codex operator contract
```

For deeper system operation notes, see:

- `system/README.md`
- `system/docs/FENLIE_CODEX_MEMORY.md`

---

## Dashboard Surfaces

The Fenlie dashboard currently exposes these main surfaces:

- **Graph Home** — relationship graph, impact-chain focus, legend layers, repair hints
- **Overview** — top-level system summary and routing
- **Ops** — orchestration, signal-risk, lab review
- **Research Workspace** — artifacts, alignment, backtests, contracts, raw
- **CPA Manager** — auth-files, quota checks, safe deletion controls

Key dashboard paths:

- `#/graph-home`
- `#/overview`
- `#/terminal/public`
- `#/workspace/artifacts`
- `#/cpa`

---

## Quick Start

### 1. Dashboard frontend

```bash
cd system/dashboard/web
npm install
npm run dev
```

### 2. Build production dashboard

```bash
cd system/dashboard/web
npm run build
```

### 3. Run focused frontend verification

```bash
cd system/dashboard/web
npm test -- --run \
  src/pages/GraphHomePage.test.tsx \
  src/graph/model.test.ts \
  src/graph/GraphHomeRenderer.test.tsx \
  src/App.test.tsx \
  src/pages/CpaPage.test.tsx \
  src/cpa/model.test.ts
```

### 4. Python-side verification

```bash
cd system
pytest -q
```

---

## CPA Manager Notes

The Fenlie CPA page is intentionally integrated into the existing dashboard shell rather than kept as a standalone HTML utility.

Current behavior:

- fetches auth-files from the existing management endpoint
- filters to Codex-compatible files
- queries quota through the existing `api-call` path
- classifies failures into structural vs. transient buckets
- stores an error ledger in browser localStorage
- only marks a file deletable when the **same structural error** appears **3 consecutive times**
- never marks transient/network failures as delete-eligible

This keeps the tool operationally useful while reducing accidental deletion risk caused by network noise.

---

## Operating Principles

Fenlie is built around a few strong constraints:

- **Source-owned state first**
- **Read-only by default**
- **Execution-path separation**
- **Guarded rollout over optimistic coupling**
- **Verification before completion**

This repository should not be described as a retail trading bot.  
It is better understood as a **trading control and research operations system**.

---

## What This Repo Is / Is Not

### It is

- a research and operator control platform
- an artifact-driven system
- a dashboard-heavy workflow environment
- a place to evolve guarded execution infrastructure

### It is not

- a one-click production auto-trader
- a guarantee of live-execution readiness
- a substitute for venue-specific compliance or broker integration review

---

## Documentation

Useful starting points:

- `system/README.md`
- `docs/investor/README.md`
- `AGENTS.md`

---

## Status

This repo is under active development.  
Current emphasis is on:

- source-owned dashboard correctness
- graph-home operator experience
- CPA management integration
- read-model / acceptance-chain hardening

---

## Disclaimer

Nothing in this repository should be interpreted as investment advice, a promise of returns, or a statement of production trading readiness.
