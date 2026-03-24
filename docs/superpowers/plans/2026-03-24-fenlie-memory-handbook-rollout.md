# Fenlie Memory Handbook Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Fenlie 的长期记忆从单一大文件演进为“总入口 + shortlist + durable rules + deep contracts”的三层半运行手册，并保持现有冷启动链兼容。

**Architecture:** 保留 `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md` 作为 bootstrap shim，同时在 `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/` 下创建分层文档树。新 tree 负责长期导航、规则、契约；旧 memory 收缩为兼容入口与高价值摘要，避免双写漂移和 consumer 越权。

**Tech Stack:** Markdown, git, shell (`sed` / `rg` / `find`)

---

## File Structure

### Create

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md` — memory tree 总入口，定义 authority order、cold-start read order、task→file mapping、update policy。
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/SHORTLIST.md` — 1 屏恢复上下文。
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/DURABLE_RULES.md` — 长期稳定规则。
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/SOURCE_AUTHORITY.md` — source / handoff / UI 权威边界。
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/LIVE_BOUNDARY.md` — guardian / executor / canary / feedback 的 live 边界契约。
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/RESEARCH_CONTRACTS.md` — research canonical head / blocker / review-only 契约。

### Modify

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md` — 收缩为兼容引导层、保留高价值摘要并指向新的 memory tree。

### Optional follow-up (not in this rollout unless strictly needed)

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/OPERATOR_HANDOFF.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/FEEDBACK_LEARNING.md`

本轮先不落这两份，避免把第一版 rollout 做成过重迁移；它们保留为下一轮 contracts 扩展任务。

---

### Task 1: Create memory tree scaffold

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/SHORTLIST.md`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/DURABLE_RULES.md`

- [ ] **Step 1: Inspect current durable memory content to extract only long-lived material**

Run:

```bash
sed -n '1,260p' /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md
```

Expected: 能看到当前 high-value shortlist、durable rules、研究主线、验证命令与架构边界。

- [ ] **Step 2: Create the `memory/` directory and the three top-level markdown files**

Run:

```bash
mkdir -p /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory
touch /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md
touch /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/SHORTLIST.md
touch /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/DURABLE_RULES.md
```

Expected: `find` 能列出三个新文件。

- [ ] **Step 3: Write `MEMORY_INDEX.md` with navigation-only content**

Include exact sections:

```md
# Fenlie Memory Index

## Purpose
## Authority Order
## Cold Start Read Order
## Task-to-Document Map
## What Belongs Here vs Source Artifacts
## Update Rules
## Memory Tree Layout
```

Requirements:
- 明确 `source-of-truth artifacts > compact handoff/context > UI/operator summaries`
- 写明主代理 / 子代理 / 人类的默认读取顺序
- 明确禁止在该文件里写临时 blocker、实时 live 状态、一次性回测结果

- [ ] **Step 4: Write `SHORTLIST.md` as a strict one-screen recovery file**

Include exact sections:

```md
# Fenlie Shortlist

## Identity
## Authority Order
## Current Main Priorities
## Low-Value / Deprioritized Paths
## Cold Start
## “继续 / 下一步” Semantics
## Never Treat As Source-of-Truth
```

Requirements:
- 保留当前主线：`ETHUSDT / 15m / single-symbol / breakout-pullback`
- 强调 exit/risk + hold-forward arbitration 为研究主线
- 明确列出低价值方向：`geometry_delta` 截面、`VPVR proxy`、`reclaim structure filter`、`daily_stop_r`
- 不写任何带时间敏感性的 ready-check / queue / live state

- [ ] **Step 5: Write `DURABLE_RULES.md` with constitutional rules only**

Include exact sections:

```md
# Fenlie Durable Rules

## Operating Style
## Source-Owned State Rule
## Live Boundary Rule
## Delegation Rule
## Validation Rule
## Stop Rule
## Change Class Rule
## Time / Retry / Idempotency / Timeout Constraints
## Memory Update Rule
```

Requirements:
- 迁移现有 `AGENTS.md` / `FENLIE_CODEX_MEMORY.md` 的长期规则
- 不把某个具体 artifact 的瞬时结论复制进来

- [ ] **Step 6: Verify the three files exist and have required headings**

Run:

```bash
for f in \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/SHORTLIST.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/DURABLE_RULES.md; do
  echo "FILE: $f"
  sed -n '1,80p' "$f"
done
```

Expected: 三个文件均存在，且标题/章节完整。

- [ ] **Step 7: Commit the scaffold**

```bash
git -C /Users/jokenrobot/Downloads/Folders/fenlie add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/SHORTLIST.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/DURABLE_RULES.md
git -C /Users/jokenrobot/Downloads/Folders/fenlie commit -m "docs(memory): add memory tree scaffold"
```

---

### Task 2: Add first-wave deep contracts

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/SOURCE_AUTHORITY.md`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/LIVE_BOUNDARY.md`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/RESEARCH_CONTRACTS.md`

- [ ] **Step 1: Create the contracts directory**

Run:

```bash
mkdir -p /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts
```

Expected: contracts 目录存在。

- [ ] **Step 2: Draft `SOURCE_AUTHORITY.md` from current source-owned principles**

Include exact sections:

```md
# Source Authority Contract

## Purpose
## Authority Chain
## Canonical Heads
## Consumer Obligations
## Superseded vs Active Rows
## Anti-Patterns
## Related Artifacts
```

Requirements:
- 明确 source artifact / handoff / UI summary 的分层
- 解释 baseline retained 仍可作为 canonical source head
- 禁止 consumer 侧重复推导 queue/gate/head

- [ ] **Step 3: Draft `LIVE_BOUNDARY.md` from OpenClaw-style control chain**

Include exact sections:

```md
# Live Boundary Contract

## Purpose
## Control Chain
## Guardian vs Executor
## Shadow / Probe / Canary / Promotion
## Reconcile and Execution Truth
## Feedback Limits
## Failure Severity
## Anti-Patterns
```

Requirements:
- 引入 `guardian = veto-only`、`executor = transport/execution only`
- 明确 feedback / review packet / browser intel 不能直接提升 live authority
- 保留 run-halfhour-pulse、monotonic、idempotency、timeout<=5000ms 等 live discipline

- [ ] **Step 4: Draft `RESEARCH_CONTRACTS.md` from current ETH exit/risk chain**

Include exact sections:

```md
# Research Contracts

## Purpose
## Retained Mainline
## Challenge Pair Naming
## Canonical Research Heads
## Forward Consensus vs Blocker vs Handoff
## Review-Only and Sidecar Rules
## allowed_now / blocked_now / next_research_priority
## Anti-Patterns
```

Requirements:
- 明确 `challenge_pair` 是 `challenger_vs_baseline`
- 明确 baseline retained 但 canonical handoff active 的语义
- sidecar positive 只能进 review，不得直接 promotion

- [ ] **Step 5: Verify each contract file is navigation-grade, not a state dump**

Run:

```bash
for f in /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/*.md; do
  echo "FILE: $f"
  rg -n "2026-|latest_|ready-check|queue aging|pnl=" "$f" || true
done
```

Expected: 没有把临时状态或时间敏感内容大面积抄进 contract。

- [ ] **Step 6: Commit first-wave contracts**

```bash
git -C /Users/jokenrobot/Downloads/Folders/fenlie add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/SOURCE_AUTHORITY.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/LIVE_BOUNDARY.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/RESEARCH_CONTRACTS.md
git -C /Users/jokenrobot/Downloads/Folders/fenlie commit -m "docs(memory): add core deep contracts"
```

---

### Task 3: Shrink legacy memory into a bootstrap shim

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`

- [ ] **Step 1: Back up the current memory file before editing**

Run:

```bash
cp /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md \
   /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md.bak.memory-tree-rollout
```

Expected: 备份文件存在。

- [ ] **Step 2: Replace the current monolithic structure with a bootstrap-compatible shim**

Keep these responsibilities only:

```md
## Bootstrap Order
## High-Value Merge Shortlist
## Memory Tree Pointer
## Reusable Validation Commands
## Durable Capability Notes
```

Requirements:
- 保留 `AGENTS.md -> FENLIE_CODEX_MEMORY.md` 启动兼容
- 明确声明深层契约已迁移到 `system/docs/memory/contracts/`
- 只保留一屏级摘要，不再继续累积 deep contracts 细节

- [ ] **Step 3: Add direct links to the new tree**

Must mention exact paths:

```md
/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md
/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/SHORTLIST.md
/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/DURABLE_RULES.md
/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/
```

- [ ] **Step 4: Verify the old memory file no longer contains the deep-contract wall**

Run:

```bash
rg -n "Break-even|Forward-consensus boundary rule|Exit/risk handoff|Tail-capacity source-owned rule" \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md
```

Expected: 无匹配或只有明确指向新 contracts 的短说明。

- [ ] **Step 5: Commit the bootstrap shim**

```bash
git -C /Users/jokenrobot/Downloads/Folders/fenlie add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md
git -C /Users/jokenrobot/Downloads/Folders/fenlie commit -m "docs(memory): slim legacy memory into bootstrap shim"
```

---

### Task 4: Run minimal rollout validation and write handoff

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`
- Create/verify only if needed: no new source artifact required

- [ ] **Step 1: Verify the full memory tree exists**

Run:

```bash
find /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory -maxdepth 3 -type f | sort
```

Expected: 至少能看到 6 个本轮文件：
- `MEMORY_INDEX.md`
- `SHORTLIST.md`
- `DURABLE_RULES.md`
- `contracts/SOURCE_AUTHORITY.md`
- `contracts/LIVE_BOUNDARY.md`
- `contracts/RESEARCH_CONTRACTS.md`

- [ ] **Step 2: Verify the cold-start chain still works**

Run:

```bash
sed -n '1,120p' /Users/jokenrobot/Downloads/Folders/fenlie/AGENTS.md
printf '\n---\n'
sed -n '1,120p' /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md
printf '\n---\n'
sed -n '1,120p' /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md
```

Expected: AGENTS 仍指向旧 memory；旧 memory 能引导到新 memory index；新 index 能继续分流读取。

- [ ] **Step 3: Verify no live path or execution artifacts were touched**

Run:

```bash
git -C /Users/jokenrobot/Downloads/Folders/fenlie diff --name-only HEAD~3..HEAD
```

Expected: 只包含 docs 路径，不包含 execution / live / scripts / source artifact builder 变更。

- [ ] **Step 4: Write a compact rollout summary in the final response**

The final response must state:
- what changed
- validation results
- whether any live path was touched
- recommended/actual change class
- residual risk
- rollback point
- key files/artifacts

- [ ] **Step 5: Optional final commit if validation text or link fixes were needed**

```bash
git -C /Users/jokenrobot/Downloads/Folders/fenlie status --short
```

Expected: 工作区中本次 DOC_ONLY 变更状态清晰、可审计。
