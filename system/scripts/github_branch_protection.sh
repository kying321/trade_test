#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/github_branch_protection.sh check [--repo owner/name] [--profile minimal|strict] [--context NAME]
  scripts/github_branch_protection.sh apply [--repo owner/name] [--profile minimal|strict] [--context NAME]

Behavior:
  - check: print current protection posture for main/pi/lie, exit non-zero if posture is non-compliant.
  - apply: enforce policy on main/pi/lie:
      * required status checks: enforce-branch-policy (strict)
      * required linear history: true
      * allow force pushes: false
      * allow deletions: false

Profiles:
  - minimal: keeps admin bypass enabled.
  - strict:  disables admin bypass and requires conversation resolution.
USAGE
}

mode="check"
repo=""
profile="minimal"
required_context="enforce-branch-policy"

# Simple token bucket for outbound API traffic.
bucket_capacity=5
bucket_refill_per_sec=5
bucket_tokens=$bucket_capacity
bucket_last_refill="$(date +%s)"

bucket_acquire() {
  local now elapsed refill
  now="$(date +%s)"
  elapsed=$(( now - bucket_last_refill ))
  if (( elapsed > 0 )); then
    refill=$(( elapsed * bucket_refill_per_sec ))
    bucket_tokens=$(( bucket_tokens + refill ))
    if (( bucket_tokens > bucket_capacity )); then
      bucket_tokens=$bucket_capacity
    fi
    bucket_last_refill=$now
  fi
  while (( bucket_tokens <= 0 )); do
    sleep 1
    now="$(date +%s)"
    elapsed=$(( now - bucket_last_refill ))
    if (( elapsed > 0 )); then
      refill=$(( elapsed * bucket_refill_per_sec ))
      bucket_tokens=$(( bucket_tokens + refill ))
      if (( bucket_tokens > bucket_capacity )); then
        bucket_tokens=$bucket_capacity
      fi
      bucket_last_refill=$now
    fi
  done
  bucket_tokens=$(( bucket_tokens - 1 ))
}

gh_api() {
  bucket_acquire
  if command -v timeout >/dev/null 2>&1; then
    GH_HTTP_TIMEOUT=5 timeout 5s gh api "$@"
    return
  fi
  if command -v gtimeout >/dev/null 2>&1; then
    GH_HTTP_TIMEOUT=5 gtimeout 5s gh api "$@"
    return
  fi
  GH_HTTP_TIMEOUT=5 gh api "$@"
}

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
    --profile)
      shift
      profile="${1:-}"
      ;;
    --context)
      shift
      required_context="${1:-}"
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
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required." >&2
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

case "$profile" in
  minimal|strict) ;;
  *)
    echo "ERROR: unknown profile '${profile}'. Use minimal|strict." >&2
    exit 2
    ;;
esac

tmp_payload="$(mktemp)"
tmp_resp="$(mktemp -d)"
trap 'rm -f "$tmp_payload"; rm -rf "$tmp_resp"' EXIT

enforce_admins="false"
required_conversation_resolution="false"
if [[ "$profile" == "strict" ]]; then
  enforce_admins="true"
  required_conversation_resolution="true"
fi

cat >"$tmp_payload" <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["__REQUIRED_CONTEXT__"]
  },
  "enforce_admins": __ENFORCE_ADMINS__,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": __REQUIRED_CONVERSATION_RESOLUTION__,
  "lock_branch": false,
  "allow_fork_syncing": false
}
JSON

sed -i.bak \
  -e "s/__REQUIRED_CONTEXT__/${required_context}/g" \
  -e "s/__ENFORCE_ADMINS__/${enforce_admins}/g" \
  -e "s/__REQUIRED_CONVERSATION_RESOLUTION__/${required_conversation_resolution}/g" \
  "$tmp_payload"
rm -f "${tmp_payload}.bak"

declare -a branches=("main" "pi" "lie")
fail=0

printf '%-8s %-10s %-6s %-6s %-6s %-6s %-10s %-24s\n' "branch" "protected" "strict" "linear" "force" "admin" "conv" "contexts"
printf '%-8s %-10s %-6s %-6s %-6s %-6s %-10s %-24s\n' "------" "---------" "------" "------" "-----" "-----" "----" "--------"

for branch in "${branches[@]}"; do
  if [[ "$mode" == "apply" ]]; then
    gh_api \
      --method PUT \
      "repos/${repo}/branches/${branch}/protection" \
      -H 'Accept: application/vnd.github+json' \
      --input "$tmp_payload" >/dev/null
  fi

  set +e
  gh_api \
    "repos/${repo}/branches/${branch}/protection" \
    -H 'Accept: application/vnd.github+json' >"${tmp_resp}/branch_${branch}.json" 2>"${tmp_resp}/branch_${branch}.err"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    fail=1
    printf '%-8s %-10s %-6s %-6s %-6s %-6s %-10s %-24s\n' "$branch" "NO" "-" "-" "-" "-" "-" "-"
    continue
  fi

  strict="$(jq -r '.required_status_checks.strict' <"${tmp_resp}/branch_${branch}.json")"
  linear="$(jq -r '.required_linear_history.enabled' <"${tmp_resp}/branch_${branch}.json")"
  force="$(jq -r '.allow_force_pushes.enabled' <"${tmp_resp}/branch_${branch}.json")"
  admins="$(jq -r '.enforce_admins.enabled' <"${tmp_resp}/branch_${branch}.json")"
  conv="$(jq -r '.required_conversation_resolution.enabled' <"${tmp_resp}/branch_${branch}.json")"
  contexts="$(jq -r '.required_status_checks.contexts | join(",")' <"${tmp_resp}/branch_${branch}.json")"

  printf '%-8s %-10s %-6s %-6s %-6s %-6s %-10s %-24s\n' "$branch" "YES" "$strict" "$linear" "$force" "$admins" "$conv" "${contexts:-none}"

  if [[ "$strict" != "true" || "$linear" != "true" || "$force" != "false" ]]; then
    fail=1
  fi
  if [[ "$contexts" != *"${required_context}"* ]]; then
    fail=1
  fi
  if [[ "$profile" == "strict" && ( "$admins" != "true" || "$conv" != "true" ) ]]; then
    fail=1
  fi
done

if [[ "$mode" == "check" && $fail -ne 0 ]]; then
  echo "ERROR: branch protection posture not compliant (profile=${profile}, context=${required_context})." >&2
  exit 2
fi

if [[ "$mode" == "apply" ]]; then
  echo "[branch-protection] applied policy to ${repo} (main/pi/lie, profile=${profile}, context=${required_context})"
else
  echo "[branch-protection] check complete for ${repo} (profile=${profile}, context=${required_context})"
fi
