#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/branch_policy_guard.sh [--branch NAME]

Valid branches:
  - main
  - pi
  - lie
USAGE
}

is_allowed_branch() {
  case "$1" in
    main|pi|lie) return 0 ;;
    *) return 1 ;;
  esac
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
      # Ignore extra args to make hook integration tolerant.
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

if ! is_allowed_branch "$branch_name"; then
  echo "ERROR: branch '${branch_name}' is blocked by policy (source=${source_name})." >&2
  echo "Allowed branches: main, pi, lie" >&2
  exit 2
fi

echo "[branch-policy] PASS source=${source_name} branch=${branch_name}"
