#!/usr/bin/env bash
set -euo pipefail

# Li E (离厄) one-shot guard loop.
# - Deterministic: absolute paths, no reliance on cwd.
# - Safe: throttles heavy recovery actions.
# - Output: concise summary for cron/Codex.

ROOT="/Users/jokenrobot/Downloads/离厄—技术分析原理/system"
CFG="${ROOT}/config.yaml"
PYMOD="lie_engine.cli"

TODAY="${1:-$(date +%F)}"
DAYS="${2:-3}"

STATE_DIR="${ROOT}/output/logs"
GUARD_STATE_JSON="${STATE_DIR}/guard_state.json"

mkdir -p "${STATE_DIR}"

run_cli() {
  PYTHONPATH=src python3 -m "${PYMOD}" --config "${CFG}" "$@"
}

now_epoch() { python3 - <<'PY'
import time; print(int(time.time()))
PY
}

json_get() {
  # json_get <json_string> <python_expr>
  python3 - "$1" "$2" <<'PY'
import json,sys
s=sys.argv[1]
expr=sys.argv[2]
try:
  j=json.loads(s)
except Exception:
  j={}
print(eval(expr,{'j':j}))
PY
}

load_guard_state() {
  if [[ -f "${GUARD_STATE_JSON}" ]] && [[ -s "${GUARD_STATE_JSON}" ]]; then
    cat "${GUARD_STATE_JSON}" || true
  else
    echo '{"consecutive_bad":0,"last_heavy_ts":0}'
  fi
}

save_guard_state() {
  # save_guard_state <json_string>
  python3 - "${GUARD_STATE_JSON}" "$1" <<'PY'
import json,sys
path=sys.argv[1]
raw=sys.argv[2]
obj=json.loads(raw)
with open(path,'w',encoding='utf-8') as f:
  json.dump(obj,f,ensure_ascii=False,indent=2)
PY
}

echo "[lie-guard] now=$(date -Iseconds) root=${ROOT}"

# 1) daemon dry-run
DAEMON_JSON="$(run_cli run-halfhour-daemon --dry-run 2>/dev/null || true)"
if [[ -z "${DAEMON_JSON}" ]]; then
  echo "[lie-guard] daemon_dry_run: FAILED (no output)"
  CURRENT_BUCKET="?"
  WOULD_RUN_PULSE="false"
  SKIP_REASON="daemon_no_output"
  DAEMON_DATE="${TODAY}"
  OBSERVED_SLOT=""
  PREVIEW_DUE_COUNT=0
else
  CURRENT_BUCKET="$(json_get "${DAEMON_JSON}" "j.get('current_bucket','?')")"
  WOULD_RUN_PULSE="$(json_get "${DAEMON_JSON}" "bool(j.get('would_run_pulse', False))")"
  # pulse_preview.reason is the canonical skip reason
  SKIP_REASON="$(json_get "${DAEMON_JSON}" "(j.get('pulse_preview') or {}).get('reason','')")"
  DAEMON_DATE="$(json_get "${DAEMON_JSON}" "j.get('date','${TODAY}')")"
  OBSERVED_SLOT="$(json_get "${DAEMON_JSON}" "(j.get('pulse_preview') or {}).get('observed_slot','')")"
  PREVIEW_DUE_COUNT="$(json_get "${DAEMON_JSON}" "(lambda x: len(x) if isinstance(x, list) else 0)((j.get('pulse_preview') or {}).get('due_slots', []))")"
  echo "[lie-guard] daemon.bucket=${CURRENT_BUCKET} would_run_pulse=${WOULD_RUN_PULSE} reason=${SKIP_REASON}"
fi

# 2) pulse (only if needed)
PULSE_RAN="false"
PULSE_ERR=""
PULSE_MAX_SLOT_RUNS=2
if [[ "${PREVIEW_DUE_COUNT:-0}" =~ ^[0-9]+$ ]]; then
  if (( PREVIEW_DUE_COUNT > PULSE_MAX_SLOT_RUNS )); then
    PULSE_MAX_SLOT_RUNS=${PREVIEW_DUE_COUNT}
  fi
fi
if (( PULSE_MAX_SLOT_RUNS > 8 )); then
  PULSE_MAX_SLOT_RUNS=8
fi
if [[ "${WOULD_RUN_PULSE}" == "True" || "${WOULD_RUN_PULSE}" == "true" ]]; then
  echo "[lie-guard] pulse: running max_slot_runs=${PULSE_MAX_SLOT_RUNS} due_count=${PREVIEW_DUE_COUNT:-0}"
  PULSE_ARGS=(run-halfhour-pulse --date "${DAEMON_DATE}" --max-slot-runs "${PULSE_MAX_SLOT_RUNS}")
  if [[ -n "${OBSERVED_SLOT}" ]]; then
    PULSE_ARGS+=(--slot "${OBSERVED_SLOT}")
  fi
  if run_cli "${PULSE_ARGS[@]}" >/dev/null 2>&1; then
    PULSE_RAN="true"
  else
    PULSE_ERR="pulse_failed"
  fi
else
  echo "[lie-guard] pulse: skip"
fi

# 3) health-check
HEALTH_JSON="$(run_cli health-check 2>/dev/null || true)"
HEALTH_STATUS="unknown"
MISSING="[]"
if [[ -n "${HEALTH_JSON}" ]]; then
  HEALTH_STATUS="$(json_get "${HEALTH_JSON}" "j.get('status','unknown')")"
  MISSING="$(json_get "${HEALTH_JSON}" "j.get('missing', [])")"
fi

echo "[lie-guard] health.status=${HEALTH_STATUS} missing=${MISSING}"

# 4) throttled recovery
STATE_JSON="$(load_guard_state)"
CONSEC_BAD="$(json_get "${STATE_JSON}" "(lambda v: int(v) if str(v).isdigit() else 0)((j.get('consecutive_bad') or 0))")"
LAST_HEAVY="$(json_get "${STATE_JSON}" "(lambda v: int(v) if str(v).isdigit() else 0)((j.get('last_heavy_ts') or 0))")"
NOW="$(now_epoch)"

RECOVERY_MODE="none"   # none|light|heavy
RECOVERY_ERRS=()

if [[ "${HEALTH_STATUS}" == "ok" ]]; then
  CONSEC_BAD=0
else
  CONSEC_BAD=$((CONSEC_BAD+1))

  # heavy if: error OR 2 consecutive non-ok OR heavy older than 6h
  if [[ "${HEALTH_STATUS}" == "error" ]] || [[ "${CONSEC_BAD}" -ge 2 ]] || (( NOW - LAST_HEAVY > 21600 )); then
    RECOVERY_MODE="heavy"
  else
    RECOVERY_MODE="light"
  fi

  if [[ "${RECOVERY_MODE}" == "light" ]]; then
    echo "[lie-guard] recovery: light (stable-replay days=1)"
    run_cli stable-replay --date "${TODAY}" --days 1 >/dev/null 2>&1 || RECOVERY_ERRS+=("stable-replay")
  else
    echo "[lie-guard] recovery: heavy (stable-replay days=${DAYS} + test-all fast)"
    run_cli stable-replay --date "${TODAY}" --days "${DAYS}" >/dev/null 2>&1 || RECOVERY_ERRS+=("stable-replay")
    run_cli test-all --fast --fast-ratio 0.10 >/dev/null 2>&1 || RECOVERY_ERRS+=("test-all")
    LAST_HEAVY=$NOW
  fi
fi

# Persist guard state
GUARD_OUT_JSON=$(python3 - <<PY
import json
print(json.dumps({"consecutive_bad": ${CONSEC_BAD}, "last_heavy_ts": ${LAST_HEAVY}}, ensure_ascii=False))
PY
)
save_guard_state "$GUARD_OUT_JSON"

# Final concise summary line (for cron)
FAILS=()
[[ -n "${PULSE_ERR}" ]] && FAILS+=("${PULSE_ERR}")
for e in "${RECOVERY_ERRS[@]:-}"; do FAILS+=("${e}"); done
FAILS_JOINED="${FAILS[*]:-}"

NEXT_ACTION=""
if [[ "${HEALTH_STATUS}" != "ok" ]]; then
  NEXT_ACTION="inspect_missing_artifacts_or_data_sources"
else
  NEXT_ACTION="continue"
fi

echo "[lie-guard] summary bucket=${CURRENT_BUCKET} pulse_ran=${PULSE_RAN} health=${HEALTH_STATUS} recovery=${RECOVERY_MODE} fails=${FAILS_JOINED:-none} next=${NEXT_ACTION}"
