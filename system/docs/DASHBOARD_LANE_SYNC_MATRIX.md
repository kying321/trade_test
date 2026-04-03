# Dashboard Lane Sync Matrix (Pi / LiE)

> Date: 2026-04-03  
> Change class: `DOC_ONLY`

## Purpose

这份文档定义 **当 `pi` 将来真的需要接入 dashboard/public-surface 能力时，应该如何做最小切片同步**。

它的目标不是推动 `lie -> pi` 整支合并，而是把“什么时候不同步、什么时候只同步最小能力块、每块要验什么”写清楚，避免后续把大漂移误读为必须回灌。

## Current Ownership Fact

经 2026-04-03 本地仓库审计：

- `pi` 当前不包含 `system/dashboard/web`
- `lie` 当前承载：
  - `system/dashboard/web`
  - `system/scripts/run_dashboard_*`
  - `system/scripts/serve_spa_fallback.py`
  - `system/tests/test_run_dashboard_*`
  - `system/docs/FENLIE_DASHBOARD_*`
  - `docs/superpowers/specs/2026-04-03-fenlie-dashboard-path-routing-ops-ia-design.md`
  - `docs/superpowers/plans/2026-04-03-fenlie-dashboard-path-routing-ops-ia-implementation-plan.md`

因此当前默认结论是：

1. dashboard/public surface 的 git lane ownership 在 `lie`
2. `pi..lie` 漂移很大，本身**不是**同步理由
3. `pi` 若要接 dashboard，只能做 **capability-sliced sync**

## Non-Sync Signals

以下事实**不能单独**触发 `lie -> pi` 同步：

- 云端服务仍叫 `pi-dashboard-web.service`
- 云端目录仍叫 `openclaw-system/dashboard/web`
- 历史页面或端口仍保留 `OpenClaw` 命名
- root domain / Pages / Worker route 的历史部署遗留
- `git diff pi..lie` 很大

这些只说明历史部署或命名惯性，不等于当前 repo ownership。

## Sync Decision Rule

只有同时满足以下条件，才应该评估某个 dashboard slice 是否进入 `pi`：

1. `pi` 出现明确 consumer / owner / runbook 需求
2. 该需求不能继续由 `lie` lane 承担
3. 可以用**有限文件集合**表达，而不是要求 `lie` 全量追平
4. 能为该切片单独定义验收命令与回滚点

若任一条件不满足，结论就是：**不同步，只记录 ownership。**

## Slice Matrix

| Slice | 何时才同步到 `pi` | 最小文件范围 | 最小验收 | 回滚点 |
| --- | --- | --- | --- | --- |
| Docs / ownership | `pi` 需要阅读 dashboard 契约，但不接前端源码 | `system/docs/BRANCH_POLICY.md` `system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md` `system/docs/DASHBOARD_LANE_SYNC_MATRIX.md` `system/docs/PROGRESS.md` | 文档链接/引用自洽；`git diff --check` | revert docs commit |
| Acceptance tooling | `pi` 需要验证公开路由/拓扑，但仍不接完整前端 | `system/scripts/run_dashboard_public_acceptance.py` `system/scripts/run_dashboard_public_topology_smoke.py` `system/scripts/run_dashboard_workspace_artifacts_smoke.py` `system/scripts/serve_spa_fallback.py` `system/tests/test_run_dashboard_*` | `cd system && pytest -q tests/test_run_dashboard_public_topology_smoke_script.py tests/test_run_dashboard_workspace_artifacts_smoke_script.py tests/test_run_dashboard_public_acceptance_script.py` | revert tooling commit |
| Public deploy packaging | `pi` 需要接 Cloudflare Pages / public dist packaging，但 route shell 仍不迁移 | `system/dashboard/web/public/_headers` `system/dashboard/web/public/_redirects` `system/dashboard/web/wrangler.jsonc` `system/dashboard/web/scripts/cloudflare-*.mjs` `system/dashboard/web/package.json` | `cd system/dashboard/web && npm run cf:preflight`；若凭证齐全，再跑 `npm run verify:public-surface -- --skip-workspace-build` | revert deploy-packaging commit |
| Route shell / IA | `pi` 需要真正承接 path-route shell 与 `/ops/*` IA | `system/dashboard/web/src/App.tsx` `system/dashboard/web/src/main.tsx` `system/dashboard/web/src/navigation/*` `system/dashboard/web/src/pages/ops/*` `system/dashboard/web/src/pages/GraphHomePage.tsx` `system/dashboard/web/src/search/*` 及对应测试 | `cd system/dashboard/web && npx vitest run src/App.test.tsx src/pages/GraphHomePage.test.tsx src/search/search-ui.test.tsx --reporter=dot && npx tsc --noEmit` | revert route-shell commit |
| Full dashboard ownership transfer | 只有在 `pi` 被明确指定为 dashboard owner 时才允许 | 上述全部 + `system/dashboard/web/src/**` 全量 + deploy/runbook docs | `cd system/dashboard/web && npm run verify:release-checklist`；再补公开拓扑/公开验收外部 smoke | 回退到 transfer 前 tag / merge-base |

## Recommended Order

若未来必须跨 lane 带入 dashboard 能力，推荐顺序是：

1. Docs / ownership
2. Acceptance tooling
3. Public deploy packaging
4. Route shell / IA
5. Full dashboard ownership transfer

不要跳过前 1-3 步直接做 4/5，除非已有人明确批准 `pi` 接管 dashboard。

## Hard No

以下动作在没有额外书面批准前都不应该做：

- 直接发起 blanket `lie -> pi` merge
- 仅因为 `pi..lie` 漂移过大就做 catch-up PR
- 只凭云端服务名里有 `pi` 就判定 `pi` 是 dashboard owner
- 在 `pi` 尚未具备 `system/dashboard/web` 工程树时，假设它应该跑完整 frontend release checklist

## Minimal Audit Checklist

每次准备跨 lane 带 dashboard slice 前，至少回答这 6 个问题：

1. 是谁要求 `pi` 消费这块能力？
2. 该能力为什么不能继续留在 `lie`？
3. 最小文件集合是什么？
4. 验收命令是什么？
5. 回滚点是什么？
6. 这是 docs/tooling/deploy/route/full-transfer 中的哪一类？

回答不完整时，不进入同步阶段。

## Evidence Snapshot

本文件基于以下审计事实写成：

- `git merge-base pi lie = ab86d0d4`
- `pi` 自 merge-base 以来：24 commits
- `lie` 自 merge-base 以来：78 commits
- `git ls-tree pi system/dashboard/web` 为空
- `origin/HEAD = origin/main`

这些事实用于解释“当前 lane 是分叉稳定态”，不是用于推动自动同步。
