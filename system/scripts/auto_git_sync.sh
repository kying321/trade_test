#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/auto_git_sync.sh [options]

Options:
  --dry-run           Show what would be done, do not commit/push.
  --skip-tests        Skip `lie validate-config` + `lie test-all`.
  --include-logs      Include `system/output/logs/tests_*.json` in commit.
  --no-review-files   Do not include `system/output/review/*` files.
  --branch NAME       Target git branch (must start with `codex/`).
  --message TEXT      Commit message.
  -h, --help          Show this help.
EOF
}

dry_run=0
run_tests=1
include_logs=0
include_review_files=1
branch_name="${AUTO_GIT_BRANCH:-codex/auto-sync}"
commit_message="${AUTO_GIT_MESSAGE:-chore(system): automated sync $(date '+%Y-%m-%d %H:%M:%S')}"

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
    --branch)
      shift
      branch_name="${1:-}"
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
if [[ "$branch_name" != codex/* ]]; then
  echo "ERROR: branch must start with 'codex/'." >&2
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
      lie validate-config
      lie test-all
    else
      PYTHONPATH=src python3 -m lie_engine.cli validate-config
      PYTHONPATH=src python3 -m lie_engine.cli test-all
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
    system/src/lie_engine/*|system/tests/*|system/config.yaml|system/config.daemon.test.yaml|system/README.md|system/docs/PROGRESS.md|system/scripts/*)
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
  git checkout -B "$branch_name"
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
