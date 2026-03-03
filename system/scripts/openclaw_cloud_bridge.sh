#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  system/scripts/openclaw_cloud_bridge.sh <action>

Actions:
  whitelist               Print executable action whitelist.
  sample-whitelist        Run whitelist sampling and render 24h success report.
  sample-whitelist-gate   Run sampling and fail-fast when whitelist gate is red.
  cut-local               Disable local OpenClaw launchd services.
  probe-cloud             Probe cloud host/project availability.
  compare                 Compare local/remote git heads.
  backup-remote           Create remote tgz backup snapshot.
  tunnel-up               Start SSH local forwarding (9999/8000/5173).
  tunnel-probe            Probe local forwarded ports.
  tunnel-down             Close forwarding tunnel.
  sync-dry-run            Rsync preview (delete-aware, no mutation).
  sync-apply              Rsync apply (keeps remote-only files).
  sync-apply-prune        Rsync apply with remote prune (destructive).
  remote-clean-junk       Remove remote __pycache__/*.pyc.
  validate-remote-config  Run remote `lie validate-config`.

Environment:
  CLOUD_HOST              default: 43.153.148.242
  CLOUD_USER              default: ubuntu
  CLOUD_PROJECT_DIR       default: /home/ubuntu/openclaw-system
  CLOUD_PASS              optional password for sshpass
  SAMPLE_ROUNDS           default: 1
  SAMPLE_WINDOW_HOURS     default: 24
  WHITELIST_ENFORCE       default: false (true -> non-pass gate exits 3)
  WHITELIST_MIN_TOTAL_SUCCESS_RATE   default: 0.95
  WHITELIST_MIN_ACTION_SUCCESS_RATE  default: 0.80
  WHITELIST_MIN_SAMPLES_PER_ACTION   default: 1
  WHITELIST_REQUIRE_LAST_RC_ZERO     default: true
  WHITELIST_REQUIRED_ACTIONS         default: probe-cloud,compare,tunnel-up,tunnel-probe,validate-remote-config,tunnel-down,sync-dry-run

Notes:
  - SSH connect timeout is hard-limited to 5s for bridge reliability checks.
  - sample-whitelist appends sample records to output/logs jsonl and renders json/md artifacts.
  - With WHITELIST_ENFORCE=true, gate failure exits with code 3 (fuse).
USAGE
}

now_utc_iso() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

now_utc_compact() {
  date -u +"%Y%m%dT%H%M%SZ"
}

now_epoch_ms() {
  python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
}

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" ]]; then
  echo "ERROR: must run inside repository." >&2
  exit 2
fi

system_root="${repo_root}/system"
if [[ ! -d "${system_root}" ]]; then
  echo "ERROR: missing system directory: ${system_root}" >&2
  exit 2
fi

cloud_host="${CLOUD_HOST:-43.153.148.242}"
cloud_user="${CLOUD_USER:-ubuntu}"
cloud_project_dir="${CLOUD_PROJECT_DIR:-/home/ubuntu/openclaw-system}"
cloud_pass="${CLOUD_PASS:-}"

sample_rounds="${SAMPLE_ROUNDS:-1}"
sample_window_hours="${SAMPLE_WINDOW_HOURS:-24}"
whitelist_enforce="${WHITELIST_ENFORCE:-false}"
whitelist_min_total_success_rate="${WHITELIST_MIN_TOTAL_SUCCESS_RATE:-0.95}"
whitelist_min_action_success_rate="${WHITELIST_MIN_ACTION_SUCCESS_RATE:-0.80}"
whitelist_min_samples_per_action="${WHITELIST_MIN_SAMPLES_PER_ACTION:-1}"
whitelist_require_last_rc_zero="${WHITELIST_REQUIRE_LAST_RC_ZERO:-true}"
whitelist_required_actions="${WHITELIST_REQUIRED_ACTIONS:-probe-cloud,compare,tunnel-up,tunnel-probe,validate-remote-config,tunnel-down,sync-dry-run}"

output_dir="${system_root}/output/review"
log_dir="${system_root}/output/logs"
samples_log="${log_dir}/openclaw_bridge_whitelist_samples.jsonl"
tunnel_socket="${TMPDIR:-/tmp}/openclaw_bridge_${cloud_host//./_}.sock"

mkdir -p "${output_dir}" "${log_dir}"

if ! [[ "${sample_rounds}" =~ ^[0-9]+$ ]] || (( sample_rounds <= 0 )); then
  echo "ERROR: SAMPLE_ROUNDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${sample_window_hours}" =~ ^[0-9]+$ ]] || (( sample_window_hours <= 0 )); then
  echo "ERROR: SAMPLE_WINDOW_HOURS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${whitelist_min_samples_per_action}" =~ ^[0-9]+$ ]] || (( whitelist_min_samples_per_action <= 0 )); then
  echo "ERROR: WHITELIST_MIN_SAMPLES_PER_ACTION must be a positive integer." >&2
  exit 2
fi

validate_rate_0_1() {
  local name="$1"
  local raw="$2"
  python3 - "$name" "$raw" <<'PY'
import sys
name = sys.argv[1]
raw = sys.argv[2]
try:
    val = float(raw)
except Exception:
    print(f"ERROR: {name} must be numeric.", file=sys.stderr)
    raise SystemExit(2)
if not (0.0 <= val <= 1.0):
    print(f"ERROR: {name} must be in [0,1].", file=sys.stderr)
    raise SystemExit(2)
PY
}

validate_rate_0_1 "WHITELIST_MIN_TOTAL_SUCCESS_RATE" "${whitelist_min_total_success_rate}"
validate_rate_0_1 "WHITELIST_MIN_ACTION_SUCCESS_RATE" "${whitelist_min_action_success_rate}"

is_true() {
  local raw
  raw="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

ssh_opts=(
  -o ConnectTimeout=5
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=2
)

ssh_exec() {
  local remote_cmd="$1"
  local -a opts
  opts=("${ssh_opts[@]}")
  if [[ -z "${cloud_pass}" ]]; then
    opts+=(-o BatchMode=yes)
  fi
  if [[ -n "${cloud_pass}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${cloud_pass}" sshpass -e ssh "${opts[@]}" "${cloud_user}@${cloud_host}" "${remote_cmd}"
  else
    ssh "${opts[@]}" "${cloud_user}@${cloud_host}" "${remote_cmd}"
  fi
}

rsync_exec() {
  local ssh_cmd
  ssh_cmd="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=2"
  if [[ -n "${cloud_pass}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${cloud_pass}" rsync --contimeout=5 --timeout=30 -e "sshpass -e ${ssh_cmd}" "$@"
  else
    rsync --contimeout=5 --timeout=30 -e "${ssh_cmd} -o BatchMode=yes" "$@"
  fi
}

remote_workdir_expr='if [ -d "'"${cloud_project_dir}"'/system" ]; then printf "%s" "'"${cloud_project_dir}"'/system"; elif [ -d "'"${cloud_project_dir}"'" ]; then printf "%s" "'"${cloud_project_dir}"'"; else exit 41; fi'

action_whitelist() {
  cat <<'WL'
cut-local
probe-cloud
compare
backup-remote
tunnel-up
tunnel-probe
tunnel-down
sync-dry-run
sync-apply
sync-apply-prune
remote-clean-junk
validate-remote-config
sample-whitelist
sample-whitelist-gate
WL
}

action_cut_local() {
  if ! command -v launchctl >/dev/null 2>&1; then
    echo "launchctl not found; skip local launchd cut."
    return 0
  fi
  local uid svc
  uid="$(id -u)"
  for svc in ai.openclaw.gateway ai.openclaw.adaptor ai.openclaw.pi_cycle_halfhour; do
    launchctl disable "gui/${uid}/${svc}" >/dev/null 2>&1 || true
    launchctl bootout "gui/${uid}/${svc}" >/dev/null 2>&1 || true
  done
  echo "local launchd services disabled/bootout attempted."
}

action_probe_cloud() {
  ssh_exec "set -e; test -d '${cloud_project_dir}'; echo host=\$(hostname); echo user=\$(whoami); echo project_dir='${cloud_project_dir}'; uname -srm"
}

action_compare() {
  local local_head remote_head
  local_head="$(git -C "${repo_root}" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
  remote_head="$(
    ssh_exec "set -e; if [ -d '${cloud_project_dir}/.git' ]; then git -C '${cloud_project_dir}' rev-parse --short HEAD; elif [ -d '${cloud_project_dir}/system/.git' ]; then git -C '${cloud_project_dir}/system' rev-parse --short HEAD; else printf '%s' 'nogit'; fi"
  )"
  cat <<OUT
{
  "local_head": "${local_head}",
  "remote_head": "${remote_head}",
  "same_head": $([[ "${local_head}" == "${remote_head}" ]] && echo "true" || echo "false")
}
OUT
}

action_backup_remote() {
  local ts
  ts="$(now_utc_compact)"
  ssh_exec "set -e; test -d '${cloud_project_dir}'; mkdir -p '${cloud_project_dir}/.backup'; tar -C '${cloud_project_dir}' -czf '${cloud_project_dir}/.backup/openclaw_${ts}.tgz' .; echo backup='${cloud_project_dir}/.backup/openclaw_${ts}.tgz'"
}

action_tunnel_up() {
  local -a opts
  opts=("${ssh_opts[@]}")
  if [[ -z "${cloud_pass}" ]]; then
    opts+=(-o BatchMode=yes)
  fi
  if ssh "${opts[@]}" -S "${tunnel_socket}" -O check "${cloud_user}@${cloud_host}" >/dev/null 2>&1; then
    echo "tunnel already up: ${tunnel_socket}"
    return 0
  fi
  if [[ -n "${cloud_pass}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${cloud_pass}" sshpass -e ssh "${opts[@]}" \
      -fN -M -S "${tunnel_socket}" \
      -L 127.0.0.1:19999:127.0.0.1:9999 \
      -L 127.0.0.1:18000:127.0.0.1:8000 \
      -L 127.0.0.1:15173:127.0.0.1:5173 \
      "${cloud_user}@${cloud_host}"
  else
    ssh "${opts[@]}" \
      -fN -M -S "${tunnel_socket}" \
      -L 127.0.0.1:19999:127.0.0.1:9999 \
      -L 127.0.0.1:18000:127.0.0.1:8000 \
      -L 127.0.0.1:15173:127.0.0.1:5173 \
      "${cloud_user}@${cloud_host}"
  fi
  echo "tunnel up: ${tunnel_socket}"
}

probe_local_port() {
  local label="$1"
  local port="$2"
  python3 - "$label" "$port" <<'PY'
import socket
import sys

label = sys.argv[1]
port = int(sys.argv[2])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(5.0)
try:
    s.connect(("127.0.0.1", port))
except Exception as exc:
    print(f"{label}:down:{exc}")
    raise SystemExit(1)
finally:
    s.close()
print(f"{label}:up")
PY
}

action_tunnel_probe() {
  probe_local_port "adaptor_9999" 19999
  probe_local_port "api_8000" 18000
  probe_local_port "dashboard_5173" 15173
}

action_tunnel_down() {
  local -a opts
  opts=("${ssh_opts[@]}")
  if [[ -z "${cloud_pass}" ]]; then
    opts+=(-o BatchMode=yes)
  fi
  set +e
  ssh "${opts[@]}" -S "${tunnel_socket}" -O exit "${cloud_user}@${cloud_host}" >/dev/null 2>&1
  local rc=$?
  set -e
  if (( rc == 0 )); then
    echo "tunnel down: ${tunnel_socket}"
  else
    echo "no active tunnel to close: ${tunnel_socket}"
  fi
  return 0
}

sync_common() {
  local mode="$1"
  local -a rsync_args
  rsync_args=(
    -az
    --itemize-changes
    --exclude ".git/"
    --exclude "__pycache__/"
    --exclude "*.pyc"
    --exclude "output/"
    --exclude "dashboard/node_modules/"
    --exclude "dashboard/.next/"
    --exclude "dashboard/out/"
  )
  if [[ "${mode}" == "dry-run" ]]; then
    rsync_args+=(--dry-run)
  elif [[ "${mode}" == "apply-prune" ]]; then
    rsync_args+=(--delete)
  fi

  ssh_exec "mkdir -p '${cloud_project_dir}'"
  rsync_exec "${rsync_args[@]}" \
    "${system_root}/src/" "${cloud_user}@${cloud_host}:${cloud_project_dir}/src/"
  rsync_exec "${rsync_args[@]}" \
    "${system_root}/scripts/" "${cloud_user}@${cloud_host}:${cloud_project_dir}/scripts/"
  rsync_exec "${rsync_args[@]}" \
    "${system_root}/docs/" "${cloud_user}@${cloud_host}:${cloud_project_dir}/docs/"
  rsync_exec "${rsync_args[@]}" \
    "${system_root}/tests/" "${cloud_user}@${cloud_host}:${cloud_project_dir}/tests/"
  rsync_exec "${rsync_args[@]}" \
    "${system_root}/config.yaml" "${cloud_user}@${cloud_host}:${cloud_project_dir}/config.yaml"
  rsync_exec "${rsync_args[@]}" \
    "${system_root}/pyproject.toml" "${cloud_user}@${cloud_host}:${cloud_project_dir}/pyproject.toml"
}

action_sync_dry_run() {
  sync_common "dry-run"
}

action_sync_apply() {
  sync_common "apply"
}

action_sync_apply_prune() {
  sync_common "apply-prune"
}

action_remote_clean_junk() {
  ssh_exec "set -e; wd=\$(${remote_workdir_expr}); find \"\$wd\" -type d -name '__pycache__' -prune -exec rm -rf {} +; find \"\$wd\" -type f -name '*.pyc' -delete; echo cleaned=\"\$wd\""
}

action_validate_remote_config() {
  ssh_exec "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; PYTHONPATH=src python3 -m lie_engine.cli --config config.yaml validate-config >/tmp/lie_validate_config.json; cat /tmp/lie_validate_config.json"
}

append_sample_record() {
  local action="$1"
  local rc="$2"
  local started_utc="$3"
  local duration_ms="$4"
  local ok
  if (( rc == 0 )); then
    ok="true"
  else
    ok="false"
  fi
  printf '{"timestamp_utc":"%s","action":"%s","command":"scripts/openclaw_cloud_bridge.sh %s","rc":%d,"ok":%s,"duration_ms":%d}\n' \
    "${started_utc}" "${action}" "${action}" "${rc}" "${ok}" "${duration_ms}" >> "${samples_log}"
}

run_action_by_name() {
  local action="$1"
  case "${action}" in
    whitelist) action_whitelist ;;
    sample-whitelist) action_sample_whitelist ;;
    sample-whitelist-gate) action_sample_whitelist_gate ;;
    cut-local) action_cut_local ;;
    probe-cloud) action_probe_cloud ;;
    compare) action_compare ;;
    backup-remote) action_backup_remote ;;
    tunnel-up) action_tunnel_up ;;
    tunnel-probe) action_tunnel_probe ;;
    tunnel-down) action_tunnel_down ;;
    sync-dry-run) action_sync_dry_run ;;
    sync-apply) action_sync_apply ;;
    sync-apply-prune) action_sync_apply_prune ;;
    remote-clean-junk) action_remote_clean_junk ;;
    validate-remote-config) action_validate_remote_config ;;
    *)
      echo "ERROR: unknown action '${action}'" >&2
      usage
      return 2
      ;;
  esac
}

action_sample_whitelist_gate() {
  local old_enforce
  old_enforce="${whitelist_enforce}"
  whitelist_enforce="true"
  action_sample_whitelist
  whitelist_enforce="${old_enforce}"
}

action_sample_whitelist() {
  local -a actions
  actions=(
    "probe-cloud"
    "compare"
    "tunnel-up"
    "tunnel-probe"
    "validate-remote-config"
    "tunnel-down"
    "sync-dry-run"
  )

  local round action start_ms end_ms started_utc rc duration_ms
  for (( round = 1; round <= sample_rounds; round++ )); do
    for action in "${actions[@]}"; do
      started_utc="$(now_utc_iso)"
      start_ms="$(now_epoch_ms)"
      set +e
      run_action_by_name "${action}" >/dev/null 2>&1
      rc=$?
      set -e
      end_ms="$(now_epoch_ms)"
      duration_ms=$(( end_ms - start_ms ))
      append_sample_record "${action}" "${rc}" "${started_utc}" "${duration_ms}"
    done
  done

  local artifact_ts artifact_json artifact_md
  artifact_ts="$(now_utc_compact)"
  artifact_json="${output_dir}/${artifact_ts}_openclaw_bridge_whitelist_24h.json"
  artifact_md="${output_dir}/${artifact_ts}_openclaw_bridge_whitelist_24h.md"

  python3 - "$samples_log" "$sample_window_hours" "$artifact_json" "$artifact_md" "$artifact_ts" "${whitelist_min_total_success_rate}" "${whitelist_min_action_success_rate}" "${whitelist_min_samples_per_action}" "${whitelist_required_actions}" "${whitelist_require_last_rc_zero}" <<'PY'
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

log_path = Path(sys.argv[1])
window_hours = int(sys.argv[2])
artifact_json = Path(sys.argv[3])
artifact_md = Path(sys.argv[4])
artifact_ts = sys.argv[5]
min_total_success_rate = float(sys.argv[6])
min_action_success_rate = float(sys.argv[7])
min_samples_per_action = int(sys.argv[8])
required_actions = [x.strip() for x in str(sys.argv[9]).split(",") if x.strip()]
require_last_rc_zero = str(sys.argv[10]).strip().lower() in {"1", "true", "yes", "y", "on"}

now = datetime.now(timezone.utc)
cutoff = now - timedelta(hours=window_hours)
records = []
if log_path.exists():
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
            ts = datetime.strptime(str(row.get("timestamp_utc", "")), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts >= cutoff:
            row["_ts"] = ts
            records.append(row)

records.sort(key=lambda x: x["_ts"])
by_action = defaultdict(list)
for row in records:
    by_action[str(row.get("action", "unknown"))].append(row)

rows = []
for action in sorted(by_action.keys()):
    grp = by_action[action]
    samples = len(grp)
    success = sum(1 for r in grp if bool(r.get("ok", False)))
    rate = (success / samples) if samples else 0.0
    tail = grp[-3:]
    rows.append(
        {
            "action": action,
            "command": f"scripts/openclaw_cloud_bridge.sh {action}",
            "samples": samples,
            "success": success,
            "success_rate": round(rate, 4),
            "last_timestamp_utc": tail[-1].get("timestamp_utc") if tail else None,
            "last_rc": int(tail[-1].get("rc", 0)) if tail else None,
            "sample_tail": [
                {
                    "timestamp_utc": str(x.get("timestamp_utc")),
                    "rc": int(x.get("rc", 0)),
                    "ok": bool(x.get("ok", False)),
                    "duration_ms": int(x.get("duration_ms", 0)),
                }
                for x in tail
            ],
        }
    )

row_by_action = {str(r["action"]): r for r in rows}

total_samples = len(records)
total_success = sum(1 for r in records if bool(r.get("ok", False)))
total_rate = (total_success / total_samples) if total_samples else 0.0

gate_reasons: list[str] = []
if total_samples <= 0:
    gate_reasons.append("no_samples_in_window")
if total_rate < min_total_success_rate:
    gate_reasons.append(f"total_success_rate({total_rate:.4f})<min_total({min_total_success_rate:.4f})")
for action in required_actions:
    row = row_by_action.get(action)
    if row is None:
        gate_reasons.append(f"missing_action:{action}")
        continue
    samples = int(row.get("samples", 0))
    rate = float(row.get("success_rate", 0.0))
    last_rc = row.get("last_rc", None)
    if samples < min_samples_per_action:
        gate_reasons.append(f"samples:{action}:{samples}<min({min_samples_per_action})")
    if rate < min_action_success_rate:
        gate_reasons.append(f"success_rate:{action}:{rate:.4f}<min({min_action_success_rate:.4f})")
    if require_last_rc_zero and last_rc not in {0, None}:
        gate_reasons.append(f"last_rc_nonzero:{action}:{last_rc}")

gate_pass = len(gate_reasons) == 0

summary = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "window_hours": window_hours,
    "cutoff_utc": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "total_samples": total_samples,
    "total_success": total_success,
    "total_success_rate": round(total_rate, 4),
    "commands": rows,
    "gate": {
        "pass": bool(gate_pass),
        "required_actions": required_actions,
        "min_total_success_rate": round(min_total_success_rate, 4),
        "min_action_success_rate": round(min_action_success_rate, 4),
        "min_samples_per_action": int(min_samples_per_action),
        "require_last_rc_zero": bool(require_last_rc_zero),
        "reasons": gate_reasons,
    },
}
artifact_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    f"# OpenClaw Bridge Whitelist 24h ({artifact_ts})",
    "",
    f"- window_hours: `{window_hours}`",
    f"- cutoff_utc: `{summary['cutoff_utc']}`",
    f"- total_samples: `{total_samples}`",
    f"- total_success_rate: `{summary['total_success_rate']:.4f}`",
    f"- gate_pass: `{bool(summary['gate']['pass'])}`",
    f"- gate_reasons: `{';'.join(summary['gate']['reasons']) if summary['gate']['reasons'] else 'none'}`",
    "",
    "| command | samples | success_rate | last_timestamp_utc | last_rc |",
    "| --- | ---: | ---: | --- | ---: |",
]
for row in rows:
    lines.append(
        f"| `{row['command']}` | {int(row['samples'])} | {float(row['success_rate']):.4f} | "
        + f"{row['last_timestamp_utc'] or '-'} | {row['last_rc'] if row['last_rc'] is not None else '-'} |"
    )
artifact_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(str(artifact_json))
print(str(artifact_md))
print(f"gate_pass={str(gate_pass).lower()}")
PY

  echo "whitelist sample artifacts:"
  echo "  - ${artifact_json}"
  echo "  - ${artifact_md}"

  local gate_pass
  gate_pass="$(
    python3 - "$artifact_json" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    payload = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    print("false")
    raise SystemExit(0)
print("true" if bool(payload.get("gate", {}).get("pass", False)) else "false")
PY
  )"
  if is_true "${whitelist_enforce}" && [[ "${gate_pass}" != "true" ]]; then
    echo "FUSE: whitelist gate failed (see ${artifact_json})." >&2
    return 3
  fi
}

if (( $# != 1 )); then
  usage
  exit 2
fi

run_action_by_name "$1"
