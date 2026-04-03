# Branch Policy (Pi / LiE)

## Stable lanes
- Default: `main`, `pi`, `lie`
- Override (dual-project mode): set `GOV_PRIMARY_BRANCHES=pi,lie` for local scripts and workflow env.
- CI baseline (current): GitHub workflows use `GOV_PRIMARY_BRANCHES=pi,lie`.

## Lane ownership boundary (2026-04-03 dashboard/public surface audit)
- `pi` 与 `lie` 是稳定 lane，不是默认双向镜像；`git diff` 很大本身**不构成**整支回灌理由。
- 经本地仓库核对：
  - `git ls-tree pi system/dashboard/web` 为空，`pi` 当前**不承载** Fenlie dashboard/public deploy 工程树。
  - `lie` 当前承载：
    - `system/dashboard/web`
    - `system/scripts/run_dashboard_*`
    - `system/scripts/serve_spa_fallback.py`
    - `system/tests/test_run_dashboard_*`
    - `system/docs/FENLIE_DASHBOARD_*`
    - `docs/superpowers/specs/2026-04-03-fenlie-dashboard-path-routing-ops-ia-design.md`
    - `docs/superpowers/plans/2026-04-03-fenlie-dashboard-path-routing-ops-ia-implementation-plan.md`
- 结论：
  1. 当前 dashboard/public surface 能力属于 `lie` lane 范围，不应因为 `pi..lie` 漂移过大就发起 blanket `lie -> pi` merge。
  2. 如需把某项 dashboard 能力引入 `pi`，必须按**能力切片**提出最小同步清单（文件范围、验收项、回滚点），而不是做 lane-wide catch-up。
  3. `systemd`/云端服务命名、历史目录名、旧 OpenClaw 术语都**不能单独**作为 git lane ownership 依据；ownership 以当前 repo 内容和文档契约为准。
- 具体切片规则见：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/DASHBOARD_LANE_SYNC_MATRIX.md`

## Emergency lane (time-boxed)
- Pattern: `hotfix/<base>/<ticket>/<expires_utc_yyyymmddhhmm>` where `<base>` is in `GOV_PRIMARY_BRANCHES`.
- Example: `hotfix/lie/INC12345/202603021200`

## Hard constraints
1. Expiry window: hotfix branch must expire within 24 hours from creation.
2. Approval trailer: latest commit message must include:
   - `HOTFIX-APPROVER: <approver>`
   - `HOTFIX-REASON: <reason>`
   - `HOTFIX-EXPIRES: <yyyymmddhhmm>`
3. Trailer consistency: `HOTFIX-EXPIRES` must equal branch suffix timestamp.
4. Merge-back requirement: hotfix commit must be merged back to target base branch and hotfix branch deleted before expiry.

## Enforcement points
- Local: `.git/hooks/pre-push` -> `system/scripts/branch_policy_guard.sh`
- Remote: GitHub Action `.github/workflows/branch-policy.yml` (job: `enforce-branch-policy`)
- Remote: GitHub Action `.github/workflows/hotfix-pr-gate.yml` (job: `hotfix-pr-gate`)
- Remote (scheduled): `.github/workflows/hotfix-reaper.yml` (every 30 minutes)

## GitHub branch protection (GOV_PRIMARY_BRANCHES)
- Script: `system/scripts/github_branch_protection.sh`
- Check posture:
  - `bash system/scripts/github_branch_protection.sh check --profile minimal`
  - `bash system/scripts/github_branch_protection.sh check --profile strict`
- Apply posture:
  - `bash system/scripts/github_branch_protection.sh apply --profile minimal`
  - `bash system/scripts/github_branch_protection.sh apply --profile strict`

Current baseline (applied on 2026-03-01):
- `required_status_checks.strict = true`
- `required_status_checks.contexts = ["enforce-branch-policy", "hotfix-pr-gate"]`
- `required_linear_history = true`
- `allow_force_pushes = false`
- `allow_deletions = false`

## Hotfix lifecycle automation
- Scanner/reaper script: `system/scripts/hotfix_reaper.sh`
- Report only:
  - `bash system/scripts/hotfix_reaper.sh --mode report`
- Enforce reap:
  - `bash system/scripts/hotfix_reaper.sh --mode reap`

## Governance drill & audit
- Friction drill (destructive, creates/cleans probe PRs and branches):
  - `bash system/scripts/branch_governance_friction.sh --confirm-destructive`
- Non-destructive audit:
  - `bash system/scripts/branch_governance_audit.sh`
  - 当前产物除 branch protection / hotfix reaper 外，还会附带
    `checks.dashboard_lane_sync_guard`
  - 该检查是 **advisory-only / non-blocking**：
    - 用于提示 `dashboard/public surface` 的当前 lane ownership
    - 默认不把 `pi..lie` 大漂移解释成 blanket sync 理由
- GitHub scheduled audit workflow:
  - `.github/workflows/governance-health-audit.yml` (hourly + manual dispatch)
  - 当前 workflow 会把最新 `*_branch_governance_audit.json` 渲染到 `GITHUB_STEP_SUMMARY`
  - 当 `dashboard_lane_sync_guard.recommended_action != no_sync_record_ownership` 时，会额外发出 **non-blocking** GitHub warning annotation
- PR 侧 workflow 也已接入同类 advisory summary：
  - `.github/workflows/branch-policy.yml`
  - `.github/workflows/hotfix-pr-gate.yml`
  - 仅补充审计可见性，不改变原有 required checks 的通过/失败语义
  - 现在还会上传 `*_branch_governance_audit.{json,md}` 作为 PR 侧可下载证据附件
  - 现在还会用 sticky comment 方式 upsert `fenlie-governance-audit-advisory` PR comment，集中暴露 lane ownership / recommended_action
  - 三个 workflow 现统一经由：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_governance_audit_advisory.sh`
    收敛 advisory runner，避免 shell 片段继续分叉
- GitHub manual friction workflow (destructive, explicit confirmation required):
  - `.github/workflows/governance-friction-drill.yml` (`workflow_dispatch`, `confirm_destructive=true`)
