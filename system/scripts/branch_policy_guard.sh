#!/usr/bin/env bash
set -euo pipefail

readonly HOTFIX_MAX_HOURS=24

usage() {
  cat <<'USAGE'
Usage: scripts/branch_policy_guard.sh [--branch NAME] [--source NAME]

Valid branches:
  - main
  - pi
  - lie

Emergency branch (time-boxed):
  - hotfix/<main|pi|lie>/<ticket>/<expires_utc_yyyymmddhhmm>

Hotfix requirements:
  - Expires within 24 hours from now.
  - Latest commit message includes:
      HOTFIX-APPROVER: <approver>
      HOTFIX-REASON: <reason>
      HOTFIX-EXPIRES: <yyyymmddhhmm>
USAGE
}

is_primary_branch() {
  case "$1" in
    main|pi|lie) return 0 ;;
    *) return 1 ;;
  esac
}

parse_hotfix_branch() {
  local b="$1"
  if [[ "$b" =~ ^hotfix/(main|pi|lie)/([A-Za-z0-9._-]+)/([0-9]{12})$ ]]; then
    HOTFIX_BASE="${BASH_REMATCH[1]}"
    HOTFIX_TICKET="${BASH_REMATCH[2]}"
    HOTFIX_EXPIRES="${BASH_REMATCH[3]}"
    return 0
  fi
  return 1
}

to_epoch_utc() {
  local yyyymmddhhmm="$1"

  if date -u -j -f "%Y%m%d%H%M" "$yyyymmddhhmm" "+%s" >/dev/null 2>&1; then
    date -u -j -f "%Y%m%d%H%M" "$yyyymmddhhmm" "+%s"
    return 0
  fi

  date -u -d "${yyyymmddhhmm:0:4}-${yyyymmddhhmm:4:2}-${yyyymmddhhmm:6:2} ${yyyymmddhhmm:8:2}:${yyyymmddhhmm:10:2}:00" "+%s"
}

extract_trailer() {
  local key="$1"
  local body="$2"
  printf '%s\n' "$body" | awk -F': *' -v k="$key" '$1==k {sub(/^ +/, "", $2); print $2; exit}'
}

validate_hotfix_commit_trailers() {
  local body approver reason expires
  body="$(git log -1 --pretty=%B 2>/dev/null || true)"

  approver="$(extract_trailer "HOTFIX-APPROVER" "$body")"
  reason="$(extract_trailer "HOTFIX-REASON" "$body")"
  expires="$(extract_trailer "HOTFIX-EXPIRES" "$body")"

  if [[ -z "$approver" ]]; then
    echo "ERROR: HOTFIX-APPROVER trailer is required for hotfix branch." >&2
    return 2
  fi
  if [[ -z "$reason" ]]; then
    echo "ERROR: HOTFIX-REASON trailer is required for hotfix branch." >&2
    return 2
  fi
  if [[ -z "$expires" ]]; then
    echo "ERROR: HOTFIX-EXPIRES trailer is required for hotfix branch." >&2
    return 2
  fi
  if [[ "$expires" != "$HOTFIX_EXPIRES" ]]; then
    echo "ERROR: HOTFIX-EXPIRES trailer (${expires}) does not match branch suffix (${HOTFIX_EXPIRES})." >&2
    return 2
  fi
}

validate_hotfix_window() {
  local now_epoch exp_epoch diff max_window
  now_epoch="$(date -u +%s)"
  exp_epoch="$(to_epoch_utc "$HOTFIX_EXPIRES")"
  max_window=$(( HOTFIX_MAX_HOURS * 3600 ))

  if (( exp_epoch <= now_epoch )); then
    echo "ERROR: hotfix branch has expired: ${HOTFIX_EXPIRES} UTC." >&2
    return 2
  fi

  diff=$(( exp_epoch - now_epoch ))
  if (( diff > max_window )); then
    echo "ERROR: hotfix branch expiry exceeds ${HOTFIX_MAX_HOURS}h window: ${HOTFIX_EXPIRES} UTC." >&2
    return 2
  fi
}

branch_name=""
source_name="manual"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      shift
      branch_name="${1:-}"
      ;;
    --source)
      shift
      source_name="${1:-manual}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      # Ignore extra args to keep hook integration tolerant.
      ;;
  esac
  shift || true
done

if [[ -z "$branch_name" ]]; then
  branch_name="$(git branch --show-current 2>/dev/null || true)"
fi

if [[ -z "$branch_name" ]]; then
  echo "ERROR: branch policy guard cannot validate detached HEAD (source=${source_name})." >&2
  exit 2
fi

if is_primary_branch "$branch_name"; then
  echo "[branch-policy] PASS source=${source_name} branch=${branch_name} mode=primary"
  exit 0
fi

if parse_hotfix_branch "$branch_name"; then
  validate_hotfix_window
  validate_hotfix_commit_trailers
  echo "[branch-policy] PASS source=${source_name} branch=${branch_name} mode=hotfix base=${HOTFIX_BASE} ticket=${HOTFIX_TICKET} expires_utc=${HOTFIX_EXPIRES}"
  exit 0
fi

echo "ERROR: branch '${branch_name}' is blocked by policy (source=${source_name})." >&2
echo "Allowed branches: main, pi, lie, hotfix/<main|pi|lie>/<ticket>/<yyyymmddhhmm>" >&2
exit 2
