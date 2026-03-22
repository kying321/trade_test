# Fenlie Codex Memory

Last updated: 2026-03-20
Purpose: durable Codex memory for stable project rules, persistent priorities, key paths, and reusable commands. This file is for cross-session recovery, not for real-time source-of-truth state.

---

## 1. Bootstrap Order

Every new Codex session for this repo should bootstrap in this order:

1. Read `/Users/jokenrobot/Downloads/Folders/fenlie/AGENTS.md`
2. Read this file
3. Resolve current task mode
4. Read only the minimal relevant skill(s)
5. Validate source artifacts before changing dashboards, briefs, or UI

Do not assume chat history is available. Reconstruct context from disk.

---

## 2. Durable Operating Rules

- Default language: Chinese
- Response style: direct, concise, technical, pragmatic, audit-friendly
- Before substantial work: state `mode + first concrete step` in one sentence
- If the user only says “continue” or “下一步”: prioritize the highest-risk unresolved issue, not cosmetic cleanup
- Prefer source-owned state: lanes, queues, gates, scores, blockers, review state, positions, and research/OOS metrics belong in source artifacts, not UI-only derivations
- Use the narrowest change class:
  - `DOC_ONLY`
  - `RESEARCH_ONLY`
  - `SIM_ONLY`
  - `LIVE_GUARD_ONLY`
  - `LIVE_EXECUTION_PATH`
- Stop rule: if two consecutive rounds do not change state, queue, gate, source ownership, or user-visible capability, stop and report

---

## 3. Live Boundary Rules

Never treat this memory file as authority for live state.

Live-sensitive work still requires direct source validation from artifacts/scripts.

Hard constraints:

- Any code touching live capital, order routing, execution queues, or fund scheduling must use the `run-halfhour-pulse` file mutex
- Backtests, replays, signal math, and historical indicators must not use `datetime.now()` or forward-looking indices
- Live deadlines/retries/backoff should use monotonic clocks
- Retryable state transitions and external operations need idempotency keys
- Outbound network requests must use timeout `<= 5000ms`
- Notification/dashboard/metrics failures degrade only unless they create execution ambiguity

---

## 4. Delegation Memory

Subagents are optional helpers, not a replacement for main-agent judgment.

Use delegation only when the user explicitly allows delegation/subagents/parallel work.

Current durable user preference (added 2026-03-21):

- Subagents are allowed when the task is unlikely to become critical prior context for later steps, or when the task chain is simple/bounded and context reuse can materially improve efficiency
- Good delegate targets:
  - bounded codebase exploration
  - isolated read-only audits
  - fixture/mock generation
  - screenshot/smoke verification
  - simple non-overlapping code changes
- Keep critical-path judgment local:
  - source-of-truth arbitration
  - live-path decisions
  - final architectural decisions
  - final user-facing conclusions
- If the delegated output is likely to become required prior context for the immediate next local step, prefer doing it in the main agent instead of delegating

Delegate only:

- non-critical sidecar research
- isolated read-only audits
- disjoint code exploration
- screenshot/smoke verification
- non-overlapping write scopes

Keep local:

- blocking critical-path work
- live-path judgment
- source-of-truth arbitration
- source-owned gate decisions
- final architectural decisions
- final user-facing conclusions

Avoid context pollution:

- do not open overlapping subagents for the same scope
- reuse an existing subagent when relevant
- require compact return format:
  - findings
  - changed files
  - validation result
  - unresolved risk

---

## 5. Persistent Project Priorities

### Strategy research mainline

Current durable research mainline:

- `ETHUSDT` `15m` single-symbol breakout-pullback base setup is the main retained strategy line
- Entry-filter direction is not the current priority
- Exit/risk is the current priority
- `max_hold_bars=8 vs 16` remains a durable follow-up comparison theme

Already observed as low-value / not currently worth extending:

- VPVR proxy
- reclaim structure filter
- daily_stop_r

### Dashboard / frontend mainline

Current durable frontend priority:

- keep the workspace/artifacts view source-owned
- preserve canonical artifact aliasing
- preserve default focus on `price_action_breakout_pullback`
- preserve multi-layer drilldown separation
- keep public/internal surfaces explicitly separated

---

## 6. Key Paths

### Repo roots

- Workspace root: `/Users/jokenrobot/Downloads/Folders/fenlie`
- System root: `/Users/jokenrobot/Downloads/Folders/fenlie/system`

### Governance / memory

- Operator rules: `/Users/jokenrobot/Downloads/Folders/fenlie/AGENTS.md`
- This memory file: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`

### Skills / toolchains

- Fenlie skills: `/Users/jokenrobot/.codex/skills/`
- Superpowers skills: `/Users/jokenrobot/.agents/skills/superpowers/`
- Codex config / MCPs: `/Users/jokenrobot/.codex/config.toml`

### Dashboard

- Frontend root: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web`
- Public snapshot:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_snapshot.json`
- Internal snapshot:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_internal_snapshot.json`

### Core scripts

- Snapshot builder:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py`
- Operator panel refresh:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py`
- Real-browser smoke:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_workspace_artifacts_smoke.py`

---

## 7. Reusable Validation Commands

### Dashboard source contract

```bash
pytest -q /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_dashboard_frontend_snapshot_script.py
```

### Local CLI wrapper

Use the repo-owned wrapper first when global `lie` is unavailable or may point at the wrong environment:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
./scripts/lie-local validate-config
./scripts/lie-local test-all --fast --fast-ratio 0.10 --timeout-seconds 600
```

### Frontend TS / unit tests

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx tsc --noEmit
npx vitest run src/App.test.tsx src/adapters/read-model.test.ts src/components/ui-kit.test.tsx
```

### Refresh snapshot / panel

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

### Internal feedback publish + refresh

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
echo '{"feedback_id":"alignment_projection_rollout","headline":"将高价值对话反馈投射到内部对齐页","summary":"用单条命令发布 autopublish 并刷新 internal projection。","recommended_action":"打开 /workspace/alignment?view=internal 检查 headline、events、actions。"}' | npm run feedback:publish-refresh -- --now 2026-03-22T04:15:00Z
```

### Internal manual feedback publish + refresh

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
echo '{"feedback_id":"manual_alignment_fix","headline":"人工结构化反馈：补齐 manual 轨","summary":"manual 轨需要标准写入口，避免直接手改 jsonl。","recommended_action":"提供 publish-manual 与 publish-manual-refresh。"}' | npm run feedback:publish-manual-refresh -- --now 2026-03-22T06:41:00Z
```

### Real-browser smoke

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run verify:public-surface -- --skip-workspace-build
npm run smoke:workspace-routes
```

### Internal manual probe browser smoke

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run smoke:alignment-internal-manual-probe
```

### Compatibility alias kept

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run smoke:workspace-artifacts
```

### Public topology smoke

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

### Cloudflare Pages production deploy

On this machine, Wrangler may time out when `HTTP_PROXY` / `HTTPS_PROXY` are inherited. The package scripts now clear those variables internally, and the deploy chain now fails fast before build in non-interactive sessions that do not provide `CLOUDFLARE_API_TOKEN`, so prefer the wrapped npm entrypoints:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run cf:preflight
npm run cf:deploy
```

Fast auth check:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run cf:whoami
```

Non-interactive note:

- if `CLOUDFLARE_API_TOKEN` is missing, `cf:preflight` / `cf:whoami` / `cf:deploy` now stop immediately with an actionable token reminder instead of spending time on `build` and failing at Wrangler auth later
- if a local Wrangler OAuth cache exists but is expired, the preflight now reports that precise cause and still requires `CLOUDFLARE_API_TOKEN` for non-interactive deploys
- local dashboard deploys may provide `CLOUDFLARE_API_TOKEN` either via environment variable or via `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/.env.cloudflare.local`

---

## 8. Memory Update Rules

Update this file only when one of these changes durably:

- operating rules
- delegation rules
- key paths
- stable mainline strategy priorities
- reusable validation commands
- durable architecture boundaries

Do not write ephemeral content here:

- temporary blockers
- transient failing tests
- one-off timestamps as “current truth”
- live queue/head state
- short-lived operator status

Those belong in source artifacts or handoff reports, not in memory.

---

## 9. Current Durable Capability Notes

- CoinGecko MCP is part of the Codex tool configuration and should be checked from tool visibility in each fresh session when needed
- Superpowers is installed locally under:
  - `/Users/jokenrobot/.codex/superpowers`
  - `/Users/jokenrobot/.agents/skills/superpowers`
- Real browser smoke for workspace artifacts is now scriptable and produces review artifacts plus screenshots
- `verify:public-surface` 已可作为单一公开面验收入口；聚合报告会在子命令失败时保留 stdout/stderr 与 payload presence 审计字段
- Cloudflare Pages production deploy for the Fenlie dashboard uses wrapped npm entrypoints that self-clear inherited proxy variables before calling Wrangler, and now fail fast in non-interactive sessions that do not provide `CLOUDFLARE_API_TOKEN`

---

## 10. Default Recovery Pattern for New Sessions

If a new session starts cold:

1. Read `AGENTS.md`
2. Read this memory file
3. Check current tool/MCP visibility if the task depends on MCPs
4. Read only the artifacts directly relevant to the current task
5. Run the smallest validation needed before editing
6. Keep live path untouched unless the task explicitly requires it
