# Branch Policy (Pi / LiE)

## Stable lanes
- `main`
- `pi`
- `lie`

## Emergency lane (time-boxed)
- Pattern: `hotfix/<main|pi|lie>/<ticket>/<expires_utc_yyyymmddhhmm>`
- Example: `hotfix/lie/INC12345/202603021200`

## Hard constraints
1. Expiry window: hotfix branch must expire within 24 hours from creation.
2. Approval trailer: latest commit message must include:
   - `HOTFIX-APPROVER: <approver>`
   - `HOTFIX-REASON: <reason>`
   - `HOTFIX-EXPIRES: <yyyymmddhhmm>`
3. Trailer consistency: `HOTFIX-EXPIRES` must equal branch suffix timestamp.
4. Merge-back requirement: hotfix commit must be merged back to target base (`main`/`pi`/`lie`) and hotfix branch deleted before expiry.

## Enforcement points
- Local: `.git/hooks/pre-push` -> `system/scripts/branch_policy_guard.sh`
- Remote: GitHub Action `.github/workflows/branch-policy.yml` (job: `enforce-branch-policy`)
- Remote: GitHub Action `.github/workflows/hotfix-pr-gate.yml` (job: `hotfix-pr-gate`)
- Remote (scheduled): `.github/workflows/hotfix-reaper.yml` (every 30 minutes)

## GitHub branch protection (main/pi/lie)
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
- GitHub scheduled audit workflow:
  - `.github/workflows/governance-health-audit.yml` (hourly + manual dispatch)
- GitHub manual friction workflow (destructive, explicit confirmation required):
  - `.github/workflows/governance-friction-drill.yml` (`workflow_dispatch`, `confirm_destructive=true`)
