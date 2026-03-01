#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/github_branch_protection.sh check [--repo owner/name]
  scripts/github_branch_protection.sh apply [--repo owner/name]

Behavior:
  - check: print current protection posture for main/pi/lie, exit non-zero if any branch is unprotected.
  - apply: enforce minimal policy on main/pi/lie:
      * required status checks: branch-policy (strict)
      * required linear history: true
      * allow force pushes: false
      * allow deletions: false
USAGE
}

mode="check"
repo=""

if [[ $# -gt 0 ]]; then
  case "$1" in
    check|apply)
      mode="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown mode '$1'" >&2
      usage
      exit 2
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      shift
      repo="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg '$1'" >&2
      usage
      exit 2
      ;;
  esac
  shift || true
done

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI is required." >&2
  exit 2
fi

if [[ -z "$repo" ]]; then
  repo="$(gh repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null || true)"
fi

if [[ -z "$repo" ]]; then
  echo "ERROR: cannot resolve repository. Pass --repo owner/name." >&2
  exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh auth required. Run: gh auth login" >&2
  exit 2
fi

tmp_payload="$(mktemp)"
trap 'rm -f "$tmp_payload"' EXIT

cat >"$tmp_payload" <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["branch-policy"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": false,
  "lock_branch": false,
  "allow_fork_syncing": false
}
JSON

declare -a branches=("main" "pi" "lie")
fail=0

printf '%-8s %-12s %-6s %-6s %-6s %-20s\n' "branch" "protected" "strict" "linear" "force" "contexts"
printf '%-8s %-12s %-6s %-6s %-6s %-20s\n' "------" "---------" "------" "------" "-----" "--------"

for branch in "${branches[@]}"; do
  if [[ "$mode" == "apply" ]]; then
    gh api \
      --method PUT \
      "repos/${repo}/branches/${branch}/protection" \
      -H 'Accept: application/vnd.github+json' \
      --input "$tmp_payload" >/dev/null
  fi

  set +e
  gh api \
    "repos/${repo}/branches/${branch}/protection" \
    -H 'Accept: application/vnd.github+json' >/tmp/.branch_protect_${branch}.json 2>/tmp/.branch_protect_${branch}.err
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    fail=1
    printf '%-8s %-12s %-6s %-6s %-6s %-20s\n' "$branch" "NO" "-" "-" "-" "-"
    continue
  fi

  strict="$(gh api "repos/${repo}/branches/${branch}/protection" --jq '.required_status_checks.strict')"
  linear="$(gh api "repos/${repo}/branches/${branch}/protection" --jq '.required_linear_history.enabled')"
  force="$(gh api "repos/${repo}/branches/${branch}/protection" --jq '.allow_force_pushes.enabled')"
  contexts="$(gh api "repos/${repo}/branches/${branch}/protection" --jq '.required_status_checks.contexts | join(",")')"

  printf '%-8s %-12s %-6s %-6s %-6s %-20s\n' "$branch" "YES" "$strict" "$linear" "$force" "${contexts:-none}"

  if [[ "$strict" != "true" || "$linear" != "true" || "$force" != "false" ]]; then
    fail=1
  fi
  if [[ "$contexts" != *"branch-policy"* ]]; then
    fail=1
  fi
done

if [[ "$mode" == "check" && $fail -ne 0 ]]; then
  echo "ERROR: branch protection posture not compliant." >&2
  exit 2
fi

if [[ "$mode" == "apply" ]]; then
  echo "[branch-protection] applied policy to ${repo} (main/pi/lie)"
else
  echo "[branch-protection] check complete for ${repo}"
fi
