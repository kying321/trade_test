#!/usr/bin/env bash
set -euo pipefail

PATH="${PATH}:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin"

CODEX_HOME_DIR="${CODEX_HOME:-${HOME}/.codex}"
GLOBAL_STATE_PATH="${CODEX_GLOBAL_STATE_PATH:-${CODEX_HOME_DIR}/.codex-global-state.json}"
STATE_DB_PATH="${CODEX_STATE_DB_PATH:-${CODEX_HOME_DIR}/state_5.sqlite}"
CONFIG_PATH="${CODEX_CONFIG_PATH:-${CODEX_HOME_DIR}/config.toml}"
CONFIG1_PATH="${CODEX_CONFIG1_PATH:-${CODEX_HOME_DIR}/config1.toml}"
BACKUP_ROOT="${CODEX_UI_REFRESH_BACKUP_ROOT:-${CODEX_HOME_DIR}/backups_state/ui-refresh}"

MODE="${1:-help}"
ARG2="${2:-}"

usage() {
  cat <<'EOF'
Usage:
  refresh_codex_ui_state.sh status
  refresh_codex_ui_state.sh refresh
  refresh_codex_ui_state.sh restore [backup_dir]

Environment overrides:
  CODEX_HOME=/path/to/.codex
  CODEX_GLOBAL_STATE_PATH=/path/to/.codex-global-state.json
  CODEX_STATE_DB_PATH=/path/to/state_5.sqlite

Notes:
  - status: print current fast-tier + UI-state summary and whether refresh is currently safe.
  - refresh: backup .codex-global-state.json, remove stale UI access keys, keep user prefs/workspace roots.
  - restore: restore .codex-global-state.json from a previous backup.
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing command: $cmd" >&2
    exit 1
  fi
}

read_service_tier() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    return 0
  fi
  sed -nE 's/^[[:space:]]*service_tier[[:space:]]*=[[:space:]]*"([^"]+)".*$/\1/p' "$path" | head -n 1
}

json_field() {
  local key="$1"
  python3 - "$GLOBAL_STATE_PATH" "$key" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    raise SystemExit(0)

payload = json.loads(path.read_text(encoding="utf-8"))
state = payload.get("electron-persisted-atom-state", {})
value = state.get(key, "")
if value is None:
    value = ""
print(value)
PY
}

count_active_codex_processes() {
  if ! command -v pgrep >/dev/null 2>&1; then
    echo "0"
    return 0
  fi

  python3 - <<'PY'
import subprocess

try:
    proc = subprocess.run(
        ["pgrep", "-af", "Codex.app|/Applications/Codex.app|app-server|(^|/)codex($| )"],
        capture_output=True,
        text=True,
        check=False,
    )
except FileNotFoundError:
    print(0)
    raise SystemExit(0)

lines = []
for raw in proc.stdout.splitlines():
    line = raw.strip()
    if not line:
        continue
    if "pgrep -af" in line:
        continue
    lines.append(line)
print(len(lines))
PY
}

state_db_in_use() {
  if [[ ! -f "$STATE_DB_PATH" ]]; then
    return 1
  fi
  if ! command -v lsof >/dev/null 2>&1; then
    return 1
  fi
  local out=""
  out="$(lsof "$STATE_DB_PATH" 2>/dev/null || true)"
  [[ -n "$out" ]]
}

ensure_refresh_safe() {
  if state_db_in_use; then
    cat >&2 <<EOF
state_5.sqlite is currently in use: $STATE_DB_PATH
Please close Codex / Codex App / app-server and retry.
EOF
    exit 2
  fi
}

latest_backup_dir() {
  if [[ ! -d "$BACKUP_ROOT" ]]; then
    return 1
  fi
  ls -1dt "$BACKUP_ROOT"/backup-* 2>/dev/null | head -n 1
}

write_status() {
  local config_tier config1_tier cloud_access environment agent_mode preferred state_db_busy process_count ready
  config_tier="$(read_service_tier "$CONFIG_PATH")"
  config1_tier="$(read_service_tier "$CONFIG1_PATH")"
  cloud_access="$(json_field "codexCloudAccess")"
  environment="$(json_field "environment")"
  agent_mode="$(json_field "agent-mode")"
  preferred="$(json_field "preferred-non-full-access-agent-mode")"
  process_count="$(count_active_codex_processes)"
  if state_db_in_use; then
    state_db_busy=1
    ready=0
  else
    state_db_busy=0
    ready=1
  fi

  echo "CODEX_HOME=$CODEX_HOME_DIR"
  echo "GLOBAL_STATE_PATH=$GLOBAL_STATE_PATH"
  echo "STATE_DB_PATH=$STATE_DB_PATH"
  echo "CONFIG_SERVICE_TIER=${config_tier}"
  echo "CONFIG1_SERVICE_TIER=${config1_tier}"
  echo "CODEX_CLOUD_ACCESS=${cloud_access}"
  echo "ENVIRONMENT=${environment}"
  echo "AGENT_MODE=${agent_mode}"
  echo "PREFERRED_NON_FULL_ACCESS_AGENT_MODE=${preferred}"
  echo "STATE_DB_IN_USE=${state_db_busy}"
  echo "ACTIVE_CODEX_PROCESS_COUNT=${process_count}"
  echo "READY_TO_REFRESH=${ready}"
}

refresh_global_state() {
  python3 - "$GLOBAL_STATE_PATH" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if path.exists():
    payload = json.loads(path.read_text(encoding="utf-8"))
else:
    payload = {}

state = payload.setdefault("electron-persisted-atom-state", {})
for key in ("codexCloudAccess", "environment"):
    state.pop(key, None)

path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

cmd_status() {
  require_cmd python3
  write_status
}

cmd_refresh() {
  require_cmd python3
  ensure_refresh_safe

  local ts backup_dir process_count
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  backup_dir="${BACKUP_ROOT}/backup-${ts}"
  mkdir -p "$backup_dir"

  if [[ -f "$GLOBAL_STATE_PATH" ]]; then
    cp "$GLOBAL_STATE_PATH" "${backup_dir}/.codex-global-state.json.before"
  else
    printf '{}\n' > "${backup_dir}/.codex-global-state.json.before"
  fi
  write_status > "${backup_dir}/status.before.txt"

  refresh_global_state

  write_status > "${backup_dir}/status.after.txt"
  process_count="$(count_active_codex_processes)"
  echo "Refresh complete."
  echo "BACKUP_DIR=$backup_dir"
  echo "ACTIVE_CODEX_PROCESS_COUNT=${process_count}"
  if [[ "$process_count" != "0" ]]; then
    echo "Note: Codex-related processes still appear active; if UI state does not recover, fully close Codex / Codex App / app-server before retrying." >&2
  fi
}

cmd_restore() {
  require_cmd python3

  local backup_dir="$ARG2"
  if [[ -z "$backup_dir" ]]; then
    backup_dir="$(latest_backup_dir || true)"
  fi
  if [[ -z "$backup_dir" || ! -f "${backup_dir}/.codex-global-state.json.before" ]]; then
    echo "Backup not found. Provide backup dir explicitly." >&2
    exit 1
  fi

  mkdir -p "$(dirname "$GLOBAL_STATE_PATH")"
  cp "${backup_dir}/.codex-global-state.json.before" "$GLOBAL_STATE_PATH"
  echo "Restored from: $backup_dir"
}

case "$MODE" in
  status)
    cmd_status
    ;;
  refresh)
    cmd_refresh
    ;;
  restore)
    cmd_restore
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac
