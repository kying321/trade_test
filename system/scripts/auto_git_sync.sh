#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_branch_targets.sh"

usage() {
  local primary_branches
  primary_branches="$(gov_primary_branches_join '/')"
  cat <<'EOF' | sed -e "s|__PRIMARY_BRANCHES__|${primary_branches}|g"
Usage: system/scripts/auto_git_sync.sh [options]

Options:
  --dry-run           Show what would be done, do not commit/push.
  --skip-tests        Skip `lie validate-config` + `lie test-all`.
  --include-logs      Include `system/output/logs/tests_*.json` in commit.
  --no-review-files   Do not include `system/output/review/*` files.
  --include-progress  Include `system/docs/PROGRESS.md` (default: excluded).
  --branch NAME       Target git branch (`__PRIMARY_BRANCHES__`).
  --fallback-branch N Fallback branch when --branch is blocked.
  --message TEXT      Commit message.
  -h, --help          Show this help.
EOF
}

dry_run=0
run_tests=1
include_logs=0
include_review_files=1
include_progress_file=0
if [[ "${AUTO_GIT_INCLUDE_PROGRESS:-0}" == "1" ]]; then
  include_progress_file=1
fi
branch_name="${AUTO_GIT_BRANCH:-lie}"
fallback_branch_name="${AUTO_GIT_FALLBACK_BRANCH:-}"
commit_message="${AUTO_GIT_MESSAGE:-chore(system): automated sync $(date '+%Y-%m-%d %H:%M:%S')}"

is_allowed_branch() {
  gov_is_primary_branch "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      ;;
    --skip-tests)
      run_tests=0
      ;;
    --include-logs)
      include_logs=1
      ;;
    --no-review-files)
      include_review_files=0
      ;;
    --include-progress)
      include_progress_file=1
      ;;
    --branch)
      shift
      branch_name="${1:-}"
      ;;
    --fallback-branch)
      shift
      fallback_branch_name="${1:-}"
      ;;
    --message)
      shift
      commit_message="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ -z "$branch_name" ]]; then
  echo "ERROR: branch name is empty." >&2
  exit 2
fi
if [[ -z "$commit_message" ]]; then
  echo "ERROR: commit message is empty." >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  echo "ERROR: not inside a git repository." >&2
  exit 2
fi
cd "$repo_root"

current_branch="$(git branch --show-current 2>/dev/null || true)"
if ! is_allowed_branch "$branch_name"; then
  if [[ -n "$fallback_branch_name" ]]; then
    if is_allowed_branch "$fallback_branch_name"; then
      echo "[auto-git-sync] WARNING: branch '$branch_name' is blocked; fallback to '$fallback_branch_name'." >&2
      branch_name="$fallback_branch_name"
    else
      echo "ERROR: branch '$branch_name' is blocked." >&2
      echo "ERROR: fallback branch '$fallback_branch_name' is also blocked." >&2
      echo "Allowed branches: $(gov_primary_branches_join ', ')" >&2
      exit 2
    fi
  elif [[ -n "$current_branch" ]] && is_allowed_branch "$current_branch"; then
    echo "[auto-git-sync] WARNING: branch '$branch_name' is blocked; auto-fallback to current branch '$current_branch'." >&2
    branch_name="$current_branch"
  elif is_allowed_branch "lie"; then
    echo "[auto-git-sync] WARNING: branch '$branch_name' is blocked; auto-fallback to 'lie'." >&2
    branch_name="lie"
  else
    echo "ERROR: branch '$branch_name' is blocked." >&2
    echo "Allowed branches: $(gov_primary_branches_join ', ')" >&2
    exit 2
  fi
fi

lock_dir="$repo_root/.git/auto_git_sync.lock.d"
if ! mkdir "$lock_dir" 2>/dev/null; then
  if [[ -f "$lock_dir/pid" ]]; then
    lock_pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
    if [[ -n "$lock_pid" ]] && ! kill -0 "$lock_pid" 2>/dev/null; then
      rm -rf "$lock_dir" 2>/dev/null || true
      mkdir "$lock_dir" 2>/dev/null || true
    fi
  fi
fi
if [[ ! -d "$lock_dir" ]]; then
  echo "ERROR: auto_git_sync lock is busy: $lock_dir" >&2
  exit 3
fi
echo "$$" > "$lock_dir/pid"
trap 'rm -rf "$lock_dir" >/dev/null 2>&1 || true' EXIT

if [[ -e "$repo_root/.git/index.lock" ]]; then
  echo "ERROR: .git/index.lock exists; git is busy, aborting auto sync." >&2
  exit 3
fi

if [[ $dry_run -eq 0 ]]; then
  if ! git remote get-url origin >/dev/null 2>&1; then
    echo "ERROR: origin remote is not configured." >&2
    echo "Hint: git remote add origin <your-github-repo-url>" >&2
    exit 2
  fi
fi

if [[ $run_tests -eq 1 ]]; then
  if [[ ! -d "$repo_root/system" ]]; then
    echo "ERROR: expected system/ directory not found." >&2
    exit 2
  fi
  echo "[auto-git-sync] Running validation tests..."
  (
    cd "$repo_root/system"
    if command -v lie >/dev/null 2>&1; then
      PYTHONDONTWRITEBYTECODE=1 lie validate-config
      PYTHONDONTWRITEBYTECODE=1 lie test-all
    else
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m lie_engine.cli validate-config
      PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m lie_engine.cli test-all
    fi
  )
fi

changed_files=()
while IFS= read -r line; do
  if [[ -n "$line" ]]; then
    changed_files+=("$line")
  fi
done < <(
  {
    git diff --name-only
    git diff --cached --name-only
    git ls-files --others --exclude-standard
  } | awk 'NF>0' | sort -u
)

eligible_files=()
for f in "${changed_files[@]}"; do
  case "$f" in
    system/src/lie_engine/*|system/tests/*|system/config.yaml|system/config.daemon.test.yaml|system/README.md|system/scripts/*)
      ;;
    system/docs/PROGRESS.md)
      if [[ $include_progress_file -eq 0 ]]; then
        continue
      fi
      ;;
    system/output/review/*)
      if [[ $include_review_files -eq 0 ]]; then
        continue
      fi
      case "$f" in
        *.md|*.json|*.yaml) ;;
        *) continue ;;
      esac
      ;;
    system/output/logs/tests_*.json)
      if [[ $include_logs -eq 0 ]]; then
        continue
      fi
      ;;
    *)
      continue
      ;;
  esac

  case "$f" in
    *"__pycache__"*|*.pyc|system/src/lie_antifragile_engine.egg-info/*|system/output/artifacts/*|system/output/daily/*|system/output/.DS_Store)
      continue
      ;;
  esac
  eligible_files+=("$f")
done

if [[ ${#eligible_files[@]} -eq 0 ]]; then
  echo "[auto-git-sync] No eligible files to commit."
  exit 0
fi

echo "[auto-git-sync] Eligible files:"
printf '  - %s\n' "${eligible_files[@]}"

if [[ $dry_run -eq 1 ]]; then
  echo "[auto-git-sync] Dry-run mode. Nothing committed or pushed."
  exit 0
fi

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$branch_name" ]]; then
  if git show-ref --verify --quiet "refs/heads/$branch_name"; then
    git checkout "$branch_name"
  else
    git checkout -b "$branch_name"
  fi
fi

git add -- "${eligible_files[@]}"
if git diff --cached --quiet; then
  echo "[auto-git-sync] No staged changes after filtering."
  exit 0
fi

git commit -m "$commit_message"

if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git push
else
  git push -u origin "$branch_name"
fi

echo "[auto-git-sync] Done."
