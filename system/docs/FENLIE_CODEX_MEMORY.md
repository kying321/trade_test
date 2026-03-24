# Fenlie Codex Memory (Bootstrap Shim)

Last updated: 2026-03-25
Purpose: keep `AGENTS.md -> FENLIE_CODEX_MEMORY.md` cold-start compatibility with a one-screen durable shim. Deep contracts have moved out of this legacy file.

---

## Bootstrap Order

Audience: **Main-agent** cold start path for `AGENTS.md -> FENLIE_CODEX_MEMORY.md` compatibility.

1. Read `/Users/jokenrobot/Downloads/Folders/fenlie/AGENTS.md`
2. Read this file (`/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`)
3. Read `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md`
4. Read `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/SHORTLIST.md`
5. Read relevant contracts/source artifacts for the active task

Notes:
- Authority reconstruction must come from source artifacts, not chat memory.
- Live/operator/research state (queues, gates, positions, blockers, OOS decisions) must be re-validated from source-of-truth artifacts.
- For subagents: follow the subagent read order defined in `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md`, instead of defaulting to legacy shim first.

## High-Value Merge Shortlist

1. Authority order is strict: `source-of-truth artifacts > compact handoff/context > UI/operator summaries > chat memory`.
2. Live-path separation is mandatory: research/feedback/browser intel can veto/degrade confidence, but cannot auto-promote live execution authority.
3. Prefer source-owned control fields (queue/gate/blocker/owner/anchor); consumers should read, not re-derive.
4. Research mainline stays narrow: `ETHUSDT / 15m / single-symbol / price_action_breakout_pullback`.
5. Do not hardcode research pair labels; `challenge_pair` must derive from source-owned chain, and concrete naming/interpretation should follow research contracts under `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/`.
6. “继续 / 下一步” means fix highest-risk unresolved issue first; avoid cosmetic motion.
7. Prefer single-entry refresh runners over ad hoc partial reruns.
8. Root-cause repair beats wrapper repair.

## Memory Tree Pointer

Legacy `FENLIE_CODEX_MEMORY.md` is now a bootstrap shim only; it must not accumulate deep-contract walls.

Deep contracts are migrated to:
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/`

Memory tree pointers (durable entry set):
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/MEMORY_INDEX.md` — memory navigation + bootstrap map.
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/SHORTLIST.md` — compact high-value merge shortlist.
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/DURABLE_RULES.md` — durable operating rules and long-lived constraints.
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/` — domain deep contracts (source authority, live boundary, research contracts, etc.).

## Reusable Validation Commands

### Validation-only commands

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system && pytest -q tests/test_build_dashboard_frontend_snapshot_script.py
cd /Users/jokenrobot/Downloads/Folders/fenlie/system && ./scripts/lie-local validate-config
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npm run verify:public-surface
```

### Refresh / rebuild runners (not validation-only)

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_price_action_breakout_pullback_exit_risk_research_chain_sim_only.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

## Durable Capability Notes

- Memory tree + contracts under `system/docs/memory/` is the long-term durable entry; this file remains compatibility bootstrap only.
- Dashboard/public/internal surface contracts use dedicated validation paths; avoid encoding transient runtime state in legacy memory.
- Exit/risk, hold-side, and remote live guard diagnostics must stay source-artifact-driven; this shim only provides durable entry pointers.
