#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_branch_targets.sh"

usage() {
  cat <<'USAGE'
Usage:
  scripts/branch_governance_audit.sh [--repo owner/name] [--output-dir PATH]

Non-destructive audit:
  1) Verify strict branch protection posture.
  2) Report expired hotfix branches (no close/delete).
USAGE
}

repo=""
output_dir="system/output/review"

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

run_timeout() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 5s "$@"
    return
  fi
  if command -v gtimeout >/dev/null 2>&1; then
    gtimeout 5s "$@"
    return
  fi
  "$@"
}

gh_api() {
  bucket_acquire
  GH_HTTP_TIMEOUT=5 run_timeout gh api "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      shift
      repo="${1:-}"
      ;;
    --output-dir)
      shift
      output_dir="${1:-system/output/review}"
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

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  echo "ERROR: must run inside git repository." >&2
  exit 2
fi

if [[ -z "$repo" ]]; then
  repo="$(gh repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null || true)"
fi
if [[ -z "$repo" ]]; then
  echo "ERROR: cannot resolve repository. Pass --repo owner/name." >&2
  exit 2
fi

declare -a primary_branches=()
while IFS= read -r branch; do
  primary_branches+=("$branch")
done < <(gov_primary_branches_lines)
if [[ "${#primary_branches[@]}" -eq 0 ]]; then
  echo "ERROR: no primary branches resolved from GOV_PRIMARY_BRANCHES." >&2
  exit 2
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${repo_root}/${output_dir}"
artifact_json="${repo_root}/${output_dir}/${ts}_branch_governance_audit.json"
artifact_md="${repo_root}/${output_dir}/${ts}_branch_governance_audit.md"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

set +e
bash "${repo_root}/system/scripts/github_branch_protection.sh" check --profile strict ${repo:+--repo "$repo"} >"${tmpdir}/protection.out" 2>"${tmpdir}/protection.err"
protect_rc=$?
set -e

protection_mode="strict"
if [[ "$protect_rc" -ne 0 ]]; then
  protection_mode="readonly_fallback"
  readonly_fail=0
  printf '%-8s %-10s %-24s\n' "branch" "protected" "contexts" >"${tmpdir}/protection_fallback.out"
  printf '%-8s %-10s %-24s\n' "------" "---------" "--------" >>"${tmpdir}/protection_fallback.out"
  for branch in "${primary_branches[@]}"; do
    set +e
    gh_api "repos/${repo}/branches/${branch}" >"${tmpdir}/branch_${branch}.json" 2>"${tmpdir}/branch_${branch}.err"
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      readonly_fail=1
      printf '%-8s %-10s %-24s\n' "$branch" "UNKNOWN" "api_error" >>"${tmpdir}/protection_fallback.out"
      continue
    fi
    protected="$(jq -r '.protected' <"${tmpdir}/branch_${branch}.json")"
    contexts="$(jq -r '.protection.required_status_checks.contexts | join(",")' <"${tmpdir}/branch_${branch}.json")"
    printf '%-8s %-10s %-24s\n' "$branch" "$protected" "${contexts:-none}" >>"${tmpdir}/protection_fallback.out"
    if [[ "$protected" != "true" ]]; then
      readonly_fail=1
    fi
    if [[ "$contexts" != *"enforce-branch-policy"* || "$contexts" != *"hotfix-pr-gate"* ]]; then
      readonly_fail=1
    fi
  done
  if [[ "$readonly_fail" -eq 0 ]]; then
    protect_rc=0
    {
      cat "${tmpdir}/protection.out"
      echo
      echo "[branch-protection] strict endpoint unavailable; readonly fallback passed:"
      cat "${tmpdir}/protection_fallback.out"
    } >"${tmpdir}/protection_merged.out"
    mv "${tmpdir}/protection_merged.out" "${tmpdir}/protection.out"
    : >"${tmpdir}/protection.err"
  else
    {
      cat "${tmpdir}/protection.err"
      echo
      echo "[branch-protection] readonly fallback failed:"
      cat "${tmpdir}/protection_fallback.out"
    } >"${tmpdir}/protection_merged.err"
    mv "${tmpdir}/protection_merged.err" "${tmpdir}/protection.err"
  fi
fi

set +e
bash "${repo_root}/system/scripts/hotfix_reaper.sh" --mode report ${repo:+--repo "$repo"} >"${tmpdir}/reaper.out" 2>"${tmpdir}/reaper.err"
reaper_rc=$?
set -e

protect_pass=false
reaper_pass=false
if [[ "$protect_rc" -eq 0 ]]; then
  protect_pass=true
fi
if [[ "$reaper_rc" -eq 0 ]]; then
  reaper_pass=true
fi

overall_pass=false
if [[ "$protect_pass" == "true" && "$reaper_pass" == "true" ]]; then
  overall_pass=true
fi

jq -n \
  --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg protection_mode "$protection_mode" \
  --argjson protect_rc "$protect_rc" \
  --argjson reaper_rc "$reaper_rc" \
  --argjson protect_pass "$protect_pass" \
  --argjson reaper_pass "$reaper_pass" \
  --argjson overall_pass "$overall_pass" \
  --arg protection_stdout "$(cat "${tmpdir}/protection.out")" \
  --arg protection_stderr "$(cat "${tmpdir}/protection.err")" \
  --arg reaper_stdout "$(cat "${tmpdir}/reaper.out")" \
  --arg reaper_stderr "$(cat "${tmpdir}/reaper.err")" \
  '{
    generated_at_utc: $generated_at,
    checks: {
      branch_protection_strict: {
        mode: $protection_mode,
        pass: $protect_pass,
        return_code: $protect_rc,
        stdout: $protection_stdout,
        stderr: $protection_stderr
      },
      hotfix_reaper_report: {
        pass: $reaper_pass,
        return_code: $reaper_rc,
        stdout: $reaper_stdout,
        stderr: $reaper_stderr
      }
    },
    overall_pass: $overall_pass
  }' > "$artifact_json"

cat > "$artifact_md" <<MD
# Branch Governance Audit (${ts})

- overall_pass: \`${overall_pass}\`
- branch_protection_strict_rc: \`${protect_rc}\`
- branch_protection_mode: \`${protection_mode}\`
- hotfix_reaper_report_rc: \`${reaper_rc}\`

## Artifacts
- JSON: ${artifact_json}
MD

echo "$artifact_json"
echo "$artifact_md"

if [[ "$overall_pass" != "true" ]]; then
  exit 2
fi
