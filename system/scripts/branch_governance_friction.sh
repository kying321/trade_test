#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/branch_governance_friction.sh [--repo owner/name] [--output-dir PATH] --confirm-destructive

Destructive tests:
  A) Direct push to protected main must be rejected.
  B) hotfix/main/* PR to lie must fail hotfix-pr-gate.
  C) Expired hotfix branch is injected and must be reaped (PR closed + branch deleted).

Safety:
  - Requires --confirm-destructive.
  - Uses short-lived temporary worktrees.
  - Cleans up probe PRs/branches on exit best-effort.
USAGE
}

repo=""
output_dir="system/output/review"
confirm_destructive="false"

# Simple token bucket for outbound network traffic.
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

gh_run() {
  bucket_acquire
  GH_HTTP_TIMEOUT=5 run_timeout gh "$@"
}

git_net() {
  bucket_acquire
  run_timeout git "$@"
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
    --confirm-destructive)
      confirm_destructive="true"
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

if [[ "$confirm_destructive" != "true" ]]; then
  echo "ERROR: destructive tests require --confirm-destructive." >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh CLI is required." >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required." >&2
  exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh auth required. Run: gh auth login" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  echo "ERROR: must run inside git repository." >&2
  exit 2
fi

if [[ -z "$repo" ]]; then
  repo="$(gh_run repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null || true)"
fi
if [[ -z "$repo" ]]; then
  echo "ERROR: cannot resolve repository. Pass --repo owner/name." >&2
  exit 2
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${repo_root}/${output_dir}"
artifact_json="${repo_root}/${output_dir}/${ts}_branch_governance_friction_evidence.json"
artifact_md="${repo_root}/${output_dir}/${ts}_branch_governance_friction_evidence.md"

tmpdir="$(mktemp -d)"
wt_a=""
wt_b=""
wt_c=""
b_pr_num=""
c_pr_num=""
b_branch=""
c_branch=""

cleanup() {
  set +e
  if [[ -n "$wt_a" && -d "$wt_a" ]]; then
    git -C "$repo_root" worktree remove "$wt_a" --force >/dev/null 2>&1 || true
  fi
  if [[ -n "$wt_b" && -d "$wt_b" ]]; then
    git -C "$repo_root" worktree remove "$wt_b" --force >/dev/null 2>&1 || true
  fi
  if [[ -n "$wt_c" && -d "$wt_c" ]]; then
    git -C "$repo_root" worktree remove "$wt_c" --force >/dev/null 2>&1 || true
  fi
  if [[ -n "$b_pr_num" ]]; then
    gh_run pr close "$b_pr_num" --repo "$repo" --delete-branch >/dev/null 2>&1 || true
  fi
  if [[ -n "$c_pr_num" ]]; then
    gh_run pr close "$c_pr_num" --repo "$repo" --delete-branch >/dev/null 2>&1 || true
  fi
  if [[ -n "$b_branch" ]]; then
    git_net -C "$repo_root" push origin --delete "$b_branch" >/dev/null 2>&1 || true
  fi
  if [[ -n "$c_branch" ]]; then
    git_net -C "$repo_root" push origin --delete "$c_branch" >/dev/null 2>&1 || true
  fi
  rm -rf "$tmpdir"
}
trap cleanup EXIT

write_string_json_file() {
  local in_file="$1"
  local out_file="$2"
  jq -Rs . <"$in_file" >"$out_file"
}

git_net -C "$repo_root" fetch --all --prune >/dev/null

# ---------- Test A ----------
wt_a="/tmp/fenlie_gov_a_${ts}"
git -C "$repo_root" worktree add -d "$wt_a" origin/main >/dev/null 2>&1
git -C "$wt_a" checkout -b "probe/direct_push_${ts}" >/dev/null 2>&1
git -C "$wt_a" commit --allow-empty -m "test(governance): direct push denial probe ${ts}" >/dev/null 2>&1

a_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
git_net -C "$wt_a" push --porcelain origin HEAD:main >"${tmpdir}/a_push.out" 2>"${tmpdir}/a_push.err"
a_rc=$?
set -e
a_end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

write_string_json_file "${tmpdir}/a_push.out" "${tmpdir}/a_push.out.json"
write_string_json_file "${tmpdir}/a_push.err" "${tmpdir}/a_push.err.json"
git -C "$repo_root" worktree remove "$wt_a" --force >/dev/null 2>&1 || true
wt_a=""

# ---------- Test B ----------
wt_b="/tmp/fenlie_gov_b_${ts}"
git -C "$repo_root" worktree add -d "$wt_b" origin/lie >/dev/null 2>&1
exp_b="$(date -u -v+2H +%Y%m%d%H%M)"
b_branch="hotfix/main/GATE_MISMATCH/${exp_b}"

git -C "$wt_b" checkout -b "$b_branch" >/dev/null 2>&1
git -C "$wt_b" commit --allow-empty \
  -m "test(governance): hotfix base mismatch probe ${ts}" \
  -m "HOTFIX-APPROVER: codex" \
  -m "HOTFIX-REASON: verify hotfix-pr-gate blocks mismatched base" \
  -m "HOTFIX-EXPIRES: ${exp_b}" >/dev/null 2>&1

b_push_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
git_net -C "$wt_b" push --porcelain -u origin "$b_branch" >"${tmpdir}/b_push.out" 2>"${tmpdir}/b_push.err"
b_push_rc=$?
set -e
b_push_end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

b_pr_url="$(gh_run pr create --repo "$repo" --base lie --head "$b_branch" --title "test: hotfix-pr-gate mismatch ${ts}" --body "intentional mismatch: hotfix/main/* PR to lie for gate verification")"
b_pr_num="$(gh_run pr view "$b_pr_url" --repo "$repo" --json number --jq '.number')"

sleep 10
set +e
gh_run pr checks "$b_pr_num" --repo "$repo" >"${tmpdir}/b_checks.out" 2>"${tmpdir}/b_checks.err"
b_checks_rc=$?
set -e
gh_run pr view "$b_pr_num" --repo "$repo" --json state,url,headRefName,baseRefName,mergeStateStatus,statusCheckRollup >"${tmpdir}/b_pr_view.json"

set +e
gh_run pr close "$b_pr_num" --repo "$repo" --delete-branch >"${tmpdir}/b_close.out" 2>"${tmpdir}/b_close.err"
b_close_rc=$?
set -e
b_final_view="$(gh_run pr view "$b_pr_num" --repo "$repo" --json state,closedAt,mergedAt,url,headRefName,baseRefName)"
printf '%s\n' "$b_final_view" >"${tmpdir}/b_final_view.json"

write_string_json_file "${tmpdir}/b_push.out" "${tmpdir}/b_push.out.json"
write_string_json_file "${tmpdir}/b_push.err" "${tmpdir}/b_push.err.json"
write_string_json_file "${tmpdir}/b_checks.out" "${tmpdir}/b_checks.out.json"
write_string_json_file "${tmpdir}/b_checks.err" "${tmpdir}/b_checks.err.json"
write_string_json_file "${tmpdir}/b_close.out" "${tmpdir}/b_close.out.json"
write_string_json_file "${tmpdir}/b_close.err" "${tmpdir}/b_close.err.json"

git -C "$repo_root" worktree remove "$wt_b" --force >/dev/null 2>&1 || true
wt_b=""

# ---------- Test C ----------
wt_c="/tmp/fenlie_gov_c_${ts}"
git -C "$repo_root" worktree add -d "$wt_c" origin/lie >/dev/null 2>&1
exp_c="$(date -u -v-1H +%Y%m%d%H%M)"
c_branch="hotfix/lie/REAPER_EXPIRED/${exp_c}"

git -C "$wt_c" checkout -b "$c_branch" >/dev/null 2>&1
git -C "$wt_c" commit --allow-empty \
  -m "test(governance): expired hotfix reaper probe ${ts}" \
  -m "HOTFIX-APPROVER: codex" \
  -m "HOTFIX-REASON: verify reaper closes PR and deletes expired branch" \
  -m "HOTFIX-EXPIRES: ${exp_c}" >/dev/null 2>&1

c_push_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
git_net -C "$wt_c" push --no-verify --porcelain -u origin "$c_branch" >"${tmpdir}/c_push.out" 2>"${tmpdir}/c_push.err"
c_push_rc=$?
set -e
c_push_end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

c_pr_url="$(gh_run pr create --repo "$repo" --base lie --head "$c_branch" --title "test: expired hotfix reaper ${ts}" --body "intentional expired hotfix branch for reaper verification")"
c_pr_num="$(gh_run pr view "$c_pr_url" --repo "$repo" --json number --jq '.number')"

sleep 5
c_reap_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
bash "${repo_root}/system/scripts/hotfix_reaper.sh" --repo "$repo" --mode reap >"${tmpdir}/c_reap.out" 2>"${tmpdir}/c_reap.err"
c_reap_rc=$?
set -e
c_reap_end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

gh_run pr view "$c_pr_num" --repo "$repo" --json state,url,headRefName,baseRefName,mergedAt >"${tmpdir}/c_pr_view.json"
c_final_view="$(gh_run pr view "$c_pr_num" --repo "$repo" --json state,closedAt,mergedAt,url,headRefName,baseRefName)"
printf '%s\n' "$c_final_view" >"${tmpdir}/c_final_view.json"

set +e
gh_run api "repos/${repo}/branches" --paginate --jq '.[].name' | rg -x "$c_branch" >"${tmpdir}/c_branch_exists.out" 2>"${tmpdir}/c_branch_exists.err"
c_branch_exists_rc=$?
set -e

write_string_json_file "${tmpdir}/c_push.out" "${tmpdir}/c_push.out.json"
write_string_json_file "${tmpdir}/c_push.err" "${tmpdir}/c_push.err.json"
write_string_json_file "${tmpdir}/c_reap.out" "${tmpdir}/c_reap.out.json"
write_string_json_file "${tmpdir}/c_reap.err" "${tmpdir}/c_reap.err.json"

git -C "$repo_root" worktree remove "$wt_c" --force >/dev/null 2>&1 || true
wt_c=""

# ---------- Verdict ----------
b_checks_text="$(cat "${tmpdir}/b_checks.out" 2>/dev/null || true)"
if printf '%s' "$b_checks_text" | rg -q 'hotfix-pr-gate[[:space:]]+fail'; then
  b_gate_fail_detected=true
else
  b_gate_fail_detected=false
fi

a_pass=false
b_pass=false
c_pass=false

if [[ "$a_rc" -ne 0 ]]; then
  a_pass=true
fi
if [[ "$b_gate_fail_detected" == "true" && "$b_close_rc" -eq 0 ]]; then
  b_pass=true
fi
c_final_state="$(jq -r '.state' <"${tmpdir}/c_final_view.json")"
if [[ "$c_reap_rc" -eq 0 && "$c_branch_exists_rc" -ne 0 && "$c_final_state" == "CLOSED" ]]; then
  c_pass=true
fi

overall_pass=false
if [[ "$a_pass" == "true" && "$b_pass" == "true" && "$c_pass" == "true" ]]; then
  overall_pass=true
fi

jq -n \
  --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg repo "$repo" \
  --arg a_start "$a_start" \
  --arg a_end "$a_end" \
  --argjson a_rc "$a_rc" \
  --slurpfile a_stdout "${tmpdir}/a_push.out.json" \
  --slurpfile a_stderr "${tmpdir}/a_push.err.json" \
  --arg b_branch "hotfix/main/GATE_MISMATCH/${exp_b}" \
  --arg b_pr_url "$b_pr_url" \
  --argjson b_pr_num "$b_pr_num" \
  --arg b_push_start "$b_push_start" \
  --arg b_push_end "$b_push_end" \
  --argjson b_push_rc "$b_push_rc" \
  --argjson b_checks_rc "$b_checks_rc" \
  --argjson b_close_rc "$b_close_rc" \
  --argjson b_gate_fail_detected "$b_gate_fail_detected" \
  --slurpfile b_push_stdout "${tmpdir}/b_push.out.json" \
  --slurpfile b_push_stderr "${tmpdir}/b_push.err.json" \
  --slurpfile b_checks "${tmpdir}/b_checks.out.json" \
  --slurpfile b_checks_err "${tmpdir}/b_checks.err.json" \
  --argjson b_pr_view "$(cat "${tmpdir}/b_pr_view.json")" \
  --argjson b_final_view "$(cat "${tmpdir}/b_final_view.json")" \
  --slurpfile b_close_stdout "${tmpdir}/b_close.out.json" \
  --slurpfile b_close_stderr "${tmpdir}/b_close.err.json" \
  --arg c_branch "hotfix/lie/REAPER_EXPIRED/${exp_c}" \
  --arg c_pr_url "$c_pr_url" \
  --argjson c_pr_num "$c_pr_num" \
  --arg c_push_start "$c_push_start" \
  --arg c_push_end "$c_push_end" \
  --argjson c_push_rc "$c_push_rc" \
  --arg c_reap_start "$c_reap_start" \
  --arg c_reap_end "$c_reap_end" \
  --argjson c_reap_rc "$c_reap_rc" \
  --argjson c_branch_exists_rc "$c_branch_exists_rc" \
  --slurpfile c_push_stdout "${tmpdir}/c_push.out.json" \
  --slurpfile c_push_stderr "${tmpdir}/c_push.err.json" \
  --slurpfile c_reap_stdout "${tmpdir}/c_reap.out.json" \
  --slurpfile c_reap_stderr "${tmpdir}/c_reap.err.json" \
  --argjson c_pr_view "$(cat "${tmpdir}/c_pr_view.json")" \
  --argjson c_final_view "$(cat "${tmpdir}/c_final_view.json")" \
  --argjson a_pass "$a_pass" \
  --argjson b_pass "$b_pass" \
  --argjson c_pass "$c_pass" \
  --argjson overall_pass "$overall_pass" \
  '{
    generated_at_utc: $generated_at,
    repository: $repo,
    results: {
      test_a_direct_push_deny_main: {
        pass: $a_pass,
        start_utc: $a_start,
        end_utc: $a_end,
        return_code: $a_rc,
        stdout: $a_stdout[0],
        stderr: $a_stderr[0]
      },
      test_b_hotfix_base_mismatch_gate: {
        pass: $b_pass,
        branch: $b_branch,
        pr_number: $b_pr_num,
        pr_url: $b_pr_url,
        push_start_utc: $b_push_start,
        push_end_utc: $b_push_end,
        push_return_code: $b_push_rc,
        push_stdout: $b_push_stdout[0],
        push_stderr: $b_push_stderr[0],
        checks_return_code: $b_checks_rc,
        checks_text: $b_checks[0],
        checks_stderr: $b_checks_err[0],
        gate_fail_detected: $b_gate_fail_detected,
        pr_view_before_close: $b_pr_view,
        close_return_code: $b_close_rc,
        close_stdout: $b_close_stdout[0],
        close_stderr: $b_close_stderr[0],
        pr_view_after_close: $b_final_view
      },
      test_c_expired_hotfix_reaper: {
        pass: $c_pass,
        branch: $c_branch,
        pr_number: $c_pr_num,
        pr_url: $c_pr_url,
        push_start_utc: $c_push_start,
        push_end_utc: $c_push_end,
        push_return_code: $c_push_rc,
        push_stdout: $c_push_stdout[0],
        push_stderr: $c_push_stderr[0],
        reap_start_utc: $c_reap_start,
        reap_end_utc: $c_reap_end,
        reap_return_code: $c_reap_rc,
        reap_stdout: $c_reap_stdout[0],
        reap_stderr: $c_reap_stderr[0],
        post_reap_branch_exists_rc: $c_branch_exists_rc,
        pr_view_before_final: $c_pr_view,
        pr_view_after_final: $c_final_view
      }
    },
    overall_pass: $overall_pass
  }' > "$artifact_json"

cat > "$artifact_md" <<MD
# Branch Governance Friction Evidence (${ts})

- Repository: \`${repo}\`
- Overall pass: \`${overall_pass}\`

## A. Direct Push Deny (main)
- pass: \`${a_pass}\`
- return_code: \`${a_rc}\`
- window: \`${a_start}\` -> \`${a_end}\`

## B. Hotfix Base Mismatch Gate
- pass: \`${b_pass}\`
- branch: \`hotfix/main/GATE_MISMATCH/${exp_b}\`
- PR: #${b_pr_num} (${b_pr_url})
- checks_return_code: \`${b_checks_rc}\`
- gate_fail_detected: \`${b_gate_fail_detected}\`

## C. Expired Hotfix Reaper
- pass: \`${c_pass}\`
- branch: \`hotfix/lie/REAPER_EXPIRED/${exp_c}\`
- PR: #${c_pr_num} (${c_pr_url})
- reap_return_code: \`${c_reap_rc}\`
- post_reap_branch_exists_rc: \`${c_branch_exists_rc}\` (non-zero means deleted)
- final_pr_state: \`${c_final_state}\`

## Artifacts
- JSON: ${artifact_json}
MD

echo "$artifact_json"
echo "$artifact_md"

if [[ "$overall_pass" != "true" ]]; then
  echo "ERROR: branch governance friction drill failed." >&2
  exit 2
fi
