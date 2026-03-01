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
- Remote: GitHub Action `.github/workflows/branch-policy.yml`

## GitHub branch protection (main/pi/lie)
- Script: `system/scripts/github_branch_protection.sh`
- Check posture:
  - `bash system/scripts/github_branch_protection.sh check`
- Apply posture:
  - `bash system/scripts/github_branch_protection.sh apply`

Current baseline (applied on 2026-03-01):
- `required_status_checks.strict = true`
- `required_status_checks.contexts = ["branch-policy"]`
- `required_linear_history = true`
- `allow_force_pushes = false`
- `allow_deletions = false`
