#!/usr/bin/env bash
set -euo pipefail

readonly HOTFIX_PATTERN='^hotfix/(main|pi|lie)/([A-Za-z0-9._-]+)/([0-9]{12})$'

usage() {
  cat <<'USAGE'
Usage:
  scripts/hotfix_reaper.sh [--repo owner/name] [--mode report|reap]

Modes:
  - report: only print expired hotfix branches.
  - reap: close open PRs from expired hotfix branches and delete branch refs.
USAGE
}

repo=""
mode="report"

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

gh_run() {
  bucket_acquire
  if command -v timeout >/dev/null 2>&1; then
    GH_HTTP_TIMEOUT=5 timeout 5s gh "$@"
    return
  fi
  if command -v gtimeout >/dev/null 2>&1; then
    GH_HTTP_TIMEOUT=5 gtimeout 5s gh "$@"
    return
  fi
  GH_HTTP_TIMEOUT=5 gh "$@"
}

gh_api() {
  gh_run api "$@"
}

to_epoch_utc() {
  local yyyymmddhhmm="$1"
  if date -u -j -f "%Y%m%d%H%M" "$yyyymmddhhmm" "+%s" >/dev/null 2>&1; then
    date -u -j -f "%Y%m%d%H%M" "$yyyymmddhhmm" "+%s"
    return 0
  fi
  date -u -d "${yyyymmddhhmm:0:4}-${yyyymmddhhmm:4:2}-${yyyymmddhhmm:6:2} ${yyyymmddhhmm:8:2}:${yyyymmddhhmm:10:2}:00" "+%s"
}

urlencode_ref() {
  local raw="$1"
  printf '%s' "$raw" | sed 's/\//%2F/g'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      shift
      repo="${1:-}"
      ;;
    --mode)
      shift
      mode="${1:-report}"
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

if [[ "$mode" != "report" && "$mode" != "reap" ]]; then
  echo "ERROR: mode must be report|reap." >&2
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

if [[ -z "$repo" ]]; then
  repo="$(gh_run repo view --json nameWithOwner --jq '.nameWithOwner' 2>/dev/null || true)"
fi

if [[ -z "$repo" ]]; then
  echo "ERROR: cannot resolve repository. Pass --repo owner/name." >&2
  exit 2
fi

now_epoch="$(date -u +%s)"
scan_total=0
scan_hotfix=0
expired_total=0
deleted_total=0
closed_pr_total=0
fail_total=0

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

gh_api --paginate "repos/${repo}/branches?per_page=100" --jq '.[].name' > "${tmpdir}/branches.txt"

echo "[hotfix-reaper] repo=${repo} mode=${mode} now_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

while IFS= read -r branch; do
  [[ -z "$branch" ]] && continue
  scan_total=$(( scan_total + 1 ))

  if [[ ! "$branch" =~ $HOTFIX_PATTERN ]]; then
    continue
  fi
  scan_hotfix=$(( scan_hotfix + 1 ))

  base="${BASH_REMATCH[1]}"
  ticket="${BASH_REMATCH[2]}"
  expires="${BASH_REMATCH[3]}"

  if ! exp_epoch="$(to_epoch_utc "$expires" 2>/dev/null)"; then
    echo "[hotfix-reaper] skip malformed branch=${branch} expires=${expires}"
    fail_total=$(( fail_total + 1 ))
    continue
  fi

  if (( exp_epoch > now_epoch )); then
    continue
  fi

  expired_total=$(( expired_total + 1 ))
  echo "[hotfix-reaper] expired branch=${branch} base=${base} ticket=${ticket} expires_utc=${expires}"

  prs_json="${tmpdir}/prs_${expired_total}.json"
  gh_run pr list \
    --repo "$repo" \
    --head "$branch" \
    --state open \
    --json number,url > "$prs_json"

  open_pr_count="$(jq 'length' < "$prs_json")"

  if [[ "$mode" == "report" ]]; then
    continue
  fi

  if (( open_pr_count > 0 )); then
    while IFS=$'\t' read -r pr_num pr_url; do
      [[ -z "${pr_num:-}" ]] && continue
      gh_run pr close "$pr_num" \
        --repo "$repo" \
        --comment "Auto-closed by hotfix reaper: branch \`${branch}\` expired at \`${expires}\` UTC."
      closed_pr_total=$(( closed_pr_total + 1 ))
      echo "[hotfix-reaper] closed_pr=${pr_num} url=${pr_url}"
    done < <(jq -r '.[] | "\(.number)\t\(.url)"' < "$prs_json")
  fi

  encoded_branch="$(urlencode_ref "$branch")"
  if gh_api --method DELETE "repos/${repo}/git/refs/heads/${encoded_branch}" >/dev/null 2>&1; then
    deleted_total=$(( deleted_total + 1 ))
    echo "[hotfix-reaper] deleted branch=${branch}"
  else
    fail_total=$(( fail_total + 1 ))
    echo "[hotfix-reaper] delete_failed branch=${branch}" >&2
  fi
done < "${tmpdir}/branches.txt"

echo "[hotfix-reaper] summary scan_total=${scan_total} hotfix_total=${scan_hotfix} expired_total=${expired_total} closed_pr_total=${closed_pr_total} deleted_total=${deleted_total} fail_total=${fail_total}"

if [[ "$mode" == "report" && "$expired_total" -gt 0 ]]; then
  exit 2
fi

if [[ "$fail_total" -gt 0 ]]; then
  exit 2
fi
