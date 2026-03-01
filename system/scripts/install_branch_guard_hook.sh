#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  echo "ERROR: not inside a git repository." >&2
  exit 2
fi

hook_path="${repo_root}/.git/hooks/pre-push"

cat > "$hook_path" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  exit 0
fi

branch_name="$(git branch --show-current 2>/dev/null || true)"
bash "${repo_root}/system/scripts/branch_policy_guard.sh" --source pre-push --branch "${branch_name}"
HOOK

chmod +x "$hook_path"
echo "[branch-policy] installed pre-push hook -> ${hook_path}"
