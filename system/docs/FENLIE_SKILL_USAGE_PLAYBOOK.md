# Fenlie Skill Usage Playbook

This playbook defines the default order for the Fenlie skills that now cover the highest-frequency audit and monitoring loops.

Related architecture and MCP routing doc:

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_SKILL_MCP_ARCHITECTURE.md`

Contract note:

- 下文命令默认指向 **已安装 skill 的 runner 入口**（`~/.codex/skills/.../scripts/*.py`）
- repo 内的 `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/*.py` 主要是 source-owned builder / report / smoke 脚本
- 不要把 “skill runner” 和 “repo 内 builder” 混为一谈；若某条链当前只有 skill runner，没有 repo-local 镜像，按 skill runner 作为 canonical 入口执行

## Skill Set

- `fenlie-daily-ops-checklist`
  Path: `/Users/jokenrobot/.codex/skills/fenlie-daily-ops-checklist`
  Purpose: build a source-driven opening checklist that tells you which Fenlie skills should run next.

- `fenlie-cross-market-refresh-audit`
  Path: `/Users/jokenrobot/.codex/skills/fenlie-cross-market-refresh-audit`
  Purpose: rebuild and audit `cross_market -> hot_brief -> context`.

- `fenlie-remote-live-guard-diagnostics`
  Path: `/Users/jokenrobot/.codex/skills/fenlie-remote-live-guard-diagnostics`
  Purpose: rebuild and audit `remote_live_handoff -> live_gate_blocker -> cross_market -> hot_brief`.

- `fenlie-source-ownership-review`
  Path: `/Users/jokenrobot/.codex/skills/fenlie-source-ownership-review`
  Purpose: detect source-consumer drift, stale review rows, and simple cycle regressions.

- `fenlie-operator-panel-refresh`
  Path: `/Users/jokenrobot/.codex/skills/fenlie-operator-panel-refresh`
  Purpose: rebuild the operator task visual panel and verify panel/dist alignment.

- `fenlie-time-sync-repair-verify`
  Path: `/Users/jokenrobot/.codex/skills/fenlie-time-sync-repair-verify`
  Purpose: rerun the full post-repair system time-sync verification chain after manual/admin macOS clock or network-time fixes.

- `fenlie-skill-mcp-governance`
  Path: `/Users/jokenrobot/.codex/skills/fenlie-skill-mcp-governance`
  Purpose: inventory configured MCPs, record in-session health, and keep the Fenlie skill/MCP routing layers aligned.

## Default Order

### 0. Daily Ops Checklist

Run this first when you want a compact opening snapshot before deciding which deeper audits to run.

```bash
python3 /Users/jokenrobot/.codex/skills/fenlie-daily-ops-checklist/scripts/run_daily_ops_skill_checklist.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

Expected focus:
- `operator_head_brief`
- `review_head_brief`
- `repair_head_brief`
- `remote_live_gate_brief`
- `checklist_brief`

Repo-local source builder (仅产出工件，不替代 skill runner 编排语义)：

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_daily_ops_skill_checklist.py
```

### 1. Source Ownership First

Run this when a change touched source artifacts, top briefs, cross-market routing, or review/repair semantics.

```bash
python3 /Users/jokenrobot/.codex/skills/fenlie-source-ownership-review/scripts/run_source_ownership_review.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

Use the result to decide whether the next step should be source-side cleanup before any more consumer/UI edits.

说明：当前 source ownership review 没有 repo-local mirror runner；canonical 入口就是已安装 skill 下的：

- `/Users/jokenrobot/.codex/skills/fenlie-source-ownership-review/scripts/run_source_ownership_review.py`

### 2. Cross-Market Refresh Audit

Run this after source changes to verify the source chain and current head/backlog semantics.

```bash
python3 /Users/jokenrobot/.codex/skills/fenlie-cross-market-refresh-audit/scripts/run_cross_market_refresh_audit.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie \
  --skip-downstream-refresh
```

Expected focus:
- `review_head_status`
- `review_head_action`
- `operator_review_lane_status`
- `operator_action_queue_brief`

### 3. Remote Live Guard Diagnostics

Run this when the question touches remote live readiness, account scope, ops gate, risk guard, or repair queue.

```bash
python3 /Users/jokenrobot/.codex/skills/fenlie-remote-live-guard-diagnostics/scripts/run_remote_live_guard_diagnostics.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie \
  --skip-downstream-refresh
```

Expected focus:
- `ready_check_scope_brief`
- `account_scope_alignment_brief`
- `remote_live_operator_alignment_brief`
- `remote_live_takeover_gate_brief`
- `remote_live_takeover_repair_queue_brief`

### 4. Operator Panel Refresh

Run this after source and top-brief semantics are stable, or whenever the dashboard needs to reflect the latest state.

```bash
python3 /Users/jokenrobot/.codex/skills/fenlie-operator-panel-refresh/scripts/run_operator_panel_refresh.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

Expected focus:
- `operator_head_brief`
- `review_head_brief`
- `repair_head_brief`
- `lane_state_brief`
- `action_queue_brief`

### 5. Time Sync Repair Verify

Run this after manual/admin macOS time repair work. It rebuilds the environment report, repair verification, repair plan, checklist, and top-level handoff in one pass.

```bash
python3 /Users/jokenrobot/.codex/skills/fenlie-time-sync-repair-verify/scripts/run_time_sync_repair_verify.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

Expected focus:
- `environment_classification`
- `verification_status`
- `repair_plan_brief`
- `priority_repair_verification_brief`

### 6. Skill / MCP Governance

Run this after installing new MCPs, when auth/connectivity drifts, or when it is unclear which MCP belongs to which Fenlie lane.

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_skill_mcp_architecture_report.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

Expected focus:
- configured MCP inventory
- in-session MCP health
- layer routing
- blocked-vs-healthy distinction

## Task-to-Skill Routing

### Source or ranking drift

Use:
1. `fenlie-source-ownership-review`
2. `fenlie-cross-market-refresh-audit`

### Remote-live confusion

Use:
1. `fenlie-remote-live-guard-diagnostics`
2. `fenlie-cross-market-refresh-audit`
3. `fenlie-operator-panel-refresh`

### Dashboard/UI does not match latest source state

Use:
1. `fenlie-cross-market-refresh-audit`
2. `fenlie-operator-panel-refresh`
3. `fenlie-source-ownership-review` if mismatch persists

### Pre-handoff health check

Use:
1. `fenlie-daily-ops-checklist`
2. `fenlie-source-ownership-review`
3. `fenlie-cross-market-refresh-audit`
4. `fenlie-remote-live-guard-diagnostics`
5. `fenlie-operator-panel-refresh`
6. `fenlie-skill-mcp-governance` if MCP routing/auth changed

### Opening-of-day state snapshot

Use:
1. `fenlie-source-ownership-review`
2. `fenlie-daily-ops-checklist`
3. `fenlie-cross-market-refresh-audit`
4. `fenlie-operator-panel-refresh`

### New MCP installed or MCP auth drift

Use:
1. `fenlie-skill-mcp-governance`
2. `fenlie-source-ownership-review` if the new MCP could affect review/dashboard lanes
3. `fenlie-operator-panel-refresh` only after routing is stable

### After macOS time repair

Use:
1. `fenlie-time-sync-repair-verify`
2. `fenlie-cross-market-refresh-audit` only if the verification clears and you need a fresh cross-market audit

## Stop Rules

- If source ownership review is clean and the cross-market audit already reflects the expected head/backlog semantics, do not keep restructuring the same consumer layer.
- If remote-live diagnostics show `current_head_outside_remote_live_scope`, treat that as a scope fact, not a reason to keep editing gates.
- If panel refresh already matches the latest hot brief and cross-market source, stop there; do not add new UI state.

## Current Baseline

At the time this playbook was written, the stable expected interpretation is:

- operator head: `XAUUSD -> wait_for_paper_execution_close_evidence`
- review head: `SC2603 -> consider_refresh_before_promotion`
- repair head: `rollback_hard -> clear_ops_live_gate_condition`
- remote live takeover gate: `current_head_outside_remote_live_scope`

Current environment-specific repair interpretation:
- time-sync repair plan: `manual_time_repair_required:SOLUSDT:timed_apns_fallback_ntp_reachable_fake_ip_residual`

This baseline is descriptive, not prescriptive; the skills should be trusted over the text if artifacts move.
