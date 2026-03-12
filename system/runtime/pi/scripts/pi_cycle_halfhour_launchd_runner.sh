#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/jokenrobot/.openclaw/workspaces/pi"
LOG="/Users/jokenrobot/.openclaw/logs/pi_cycle_launchd.log"
PY_BIN="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
FENLIE_SYSTEM_ROOT="${FENLIE_SYSTEM_ROOT:-/Users/jokenrobot/.openclaw/workspaces/pi/fenlie-system}"
BRIDGE_SCRIPT_SOURCE="${FENLIE_SYSTEM_ROOT}/scripts/openclaw_cloud_bridge.sh"
BRIDGE_SCRIPT_LOCAL="${ROOT}/scripts/openclaw_cloud_bridge.sh"
BRIDGE_SCRIPT="${BRIDGE_SCRIPT_SOURCE}"
WHITELIST_GATE_ENABLED="${WHITELIST_GATE_ENABLED:-true}"
WHITELIST_GATE_MAX_AGE_MINUTES="${WHITELIST_GATE_MAX_AGE_MINUTES:-90}"
WHITELIST_GATE_WINDOW_HOURS="${WHITELIST_GATE_WINDOW_HOURS:-8}"
WHITELIST_GATE_MIN_TOTAL="${WHITELIST_GATE_MIN_TOTAL:-0.95}"
WHITELIST_GATE_MIN_ACTION="${WHITELIST_GATE_MIN_ACTION:-0.66}"
PI_LAUNCHD_GATE_ONLY="${PI_LAUNCHD_GATE_ONLY:-false}"
CLOUD_PASS_KEYCHAIN_SERVICE="${CLOUD_PASS_KEYCHAIN_SERVICE:-openclaw.pi.cloud_pass}"
PI_LAUNCHD_RETRO_ENABLED="${PI_LAUNCHD_RETRO_ENABLED:-true}"
PI_LAUNCHD_RETRO_WINDOW_HOURS="${PI_LAUNCHD_RETRO_WINDOW_HOURS:-12}"
PI_LAUNCHD_RETRO_PREFIX="${PI_LAUNCHD_RETRO_PREFIX:-pi_launchd_auto_retro}"
PI_LAUNCHD_RETRO_REVIEW_DIR="${PI_LAUNCHD_RETRO_REVIEW_DIR:-${FENLIE_SYSTEM_ROOT}/output/review}"
PI_LAUNCHD_RETRO_SCRIPT="${PI_LAUNCHD_RETRO_SCRIPT:-${FENLIE_SYSTEM_ROOT}/scripts/pi_launchd_night_retro.py}"
PI_LAUNCHD_RETRO_SAMPLE_LOG="${PI_LAUNCHD_RETRO_SAMPLE_LOG:-${FENLIE_SYSTEM_ROOT}/output/logs/openclaw_bridge_whitelist_samples.jsonl}"
export FENLIE_SYSTEM_ROOT
export LIE_SYSTEM_ROOT="${LIE_SYSTEM_ROOT:-$FENLIE_SYSTEM_ROOT}"
export LIE_PAPER_MODE_READINESS_ALLOW_WARMUP="${LIE_PAPER_MODE_READINESS_ALLOW_WARMUP:-true}"
export LIE_PAPER_MODE_WARMUP_LOG_PATH="${LIE_PAPER_MODE_WARMUP_LOG_PATH:-$LOG}"
export LIE_PAPER_MODE_WARMUP_WINDOW_HOURS="${LIE_PAPER_MODE_WARMUP_WINDOW_HOURS:-24}"
export LIE_PAPER_MODE_WARMUP_COVERAGE_MIN="${LIE_PAPER_MODE_WARMUP_COVERAGE_MIN:-0.80}"
export LIE_PAPER_MODE_WARMUP_MAX_MISSING_BUCKETS="${LIE_PAPER_MODE_WARMUP_MAX_MISSING_BUCKETS:-8}"
export LIE_PAPER_MODE_WARMUP_MAX_LARGEST_MISSING_BLOCK_HOURS="${LIE_PAPER_MODE_WARMUP_MAX_LARGEST_MISSING_BLOCK_HOURS:-2.0}"

# Runtime governance: launchd is the single scheduler, trader executes, scout reads.
export PI_PREFERRED_GATEWAY_LABEL="ai.openclaw.gateway"
export PI_EXECUTION_ROLE="trader"
export PI_MARKET_DATA_ROLE="scout"
export LIE_MAX_DAILY_DRAWDOWN_PCT="0.05"
export LIE_MAX_DAILY_DRAWDOWN_USDT="5"

mkdir -p "$(dirname "$LOG")"
start_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "[$start_ts] pi_cycle_launchd start" >>"$LOG"

if [ ! -x "$PY_BIN" ]; then
  PY_BIN="python3"
fi

load_cloud_pass_from_keychain() {
  if [ -n "${CLOUD_PASS:-}" ]; then
    return 0
  fi
  if ! command -v security >/dev/null 2>&1; then
    return 0
  fi
  set +e
  local keychain_pass
  keychain_pass="$(security find-generic-password -a "$USER" -s "$CLOUD_PASS_KEYCHAIN_SERVICE" -w 2>/dev/null)"
  local rc=$?
  set -e
  if [ "$rc" -eq 0 ] && [ -n "$keychain_pass" ]; then
    export CLOUD_PASS="$keychain_pass"
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pi_cycle_launchd info=cloud_pass_loaded_from_keychain service=$CLOUD_PASS_KEYCHAIN_SERVICE" >>"$LOG"
    return 0
  fi
  return 0
}

load_cloud_pass_from_keychain

run_retro_report() {
  case "$(printf '%s' "$PI_LAUNCHD_RETRO_ENABLED" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) ;;
    *) return 0 ;;
  esac
  if [ ! -r "$PI_LAUNCHD_RETRO_SCRIPT" ]; then
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pi_cycle_launchd warn=retro_script_missing path=$PI_LAUNCHD_RETRO_SCRIPT" >>"$LOG"
    return 0
  fi
  mkdir -p "$PI_LAUNCHD_RETRO_REVIEW_DIR"
  set +e
  retro_output="$("$PY_BIN" "$PI_LAUNCHD_RETRO_SCRIPT" \
    --window-hours "$PI_LAUNCHD_RETRO_WINDOW_HOURS" \
    --out-prefix "$PI_LAUNCHD_RETRO_PREFIX" \
    --review-dir "$PI_LAUNCHD_RETRO_REVIEW_DIR" \
    --launchd-log "$LOG" \
    --sample-log "$PI_LAUNCHD_RETRO_SAMPLE_LOG" 2>&1)"
  retro_rc=$?
  set -e
  if [ -n "$retro_output" ]; then
    printf '%s\n' "$retro_output" >>"$LOG"
  fi
  if [ "$retro_rc" -ne 0 ]; then
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pi_cycle_launchd warn=retro_report_failed rc=$retro_rc" >>"$LOG"
    return 0
  fi
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pi_cycle_launchd info=retro_report_ok" >>"$LOG"
  return 0
}

sync_bridge_script_copy() {
  local src="$1"
  local dst="$2"
  local tmp="${dst}.tmp.$$"
  if [ ! -r "$src" ]; then
    return 1
  fi
  if [ ! -f "$dst" ] || ! cmp -s "$src" "$dst"; then
    cp "$src" "$tmp" || return 1
    chmod 755 "$tmp" || true
    mv "$tmp" "$dst" || {
      rm -f "$tmp"
      return 1
    }
  fi
  [ -x "$dst" ] || chmod 755 "$dst" || true
  return 0
}

# Hard pre-gate for launchd chain: fail fast on stale/red whitelist evidence.
case "$(printf '%s' "$WHITELIST_GATE_ENABLED" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    if [ -x "$BRIDGE_SCRIPT_SOURCE" ]; then
      BRIDGE_SCRIPT="$BRIDGE_SCRIPT_SOURCE"
      sync_bridge_script_copy "$BRIDGE_SCRIPT_SOURCE" "$BRIDGE_SCRIPT_LOCAL" || true
    elif sync_bridge_script_copy "$BRIDGE_SCRIPT_SOURCE" "$BRIDGE_SCRIPT_LOCAL"; then
      BRIDGE_SCRIPT="$BRIDGE_SCRIPT_LOCAL"
    elif [ -x "$BRIDGE_SCRIPT_LOCAL" ]; then
      BRIDGE_SCRIPT="$BRIDGE_SCRIPT_LOCAL"
      echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pi_cycle_launchd warn=bridge_source_missing_using_cached_copy src=$BRIDGE_SCRIPT_SOURCE dst=$BRIDGE_SCRIPT_LOCAL" >>"$LOG"
    else
      BRIDGE_SCRIPT="$BRIDGE_SCRIPT_SOURCE"
    fi
    if [ ! -x "$BRIDGE_SCRIPT" ]; then
      echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pi_cycle_launchd fuse rc=3 reason=missing_bridge_script path=$BRIDGE_SCRIPT" >>"$LOG"
      exit 3
    fi
    set +e
    SAMPLE_WINDOW_HOURS="$WHITELIST_GATE_WINDOW_HOURS" \
    WHITELIST_ASSERT_MAX_AGE_MINUTES="$WHITELIST_GATE_MAX_AGE_MINUTES" \
    WHITELIST_MIN_TOTAL_SUCCESS_RATE="$WHITELIST_GATE_MIN_TOTAL" \
    WHITELIST_MIN_ACTION_SUCCESS_RATE="$WHITELIST_GATE_MIN_ACTION" \
    WHITELIST_MIN_SAMPLES_PER_ACTION=1 \
    WHITELIST_REQUIRE_LAST_RC_ZERO=true \
    /bin/bash "$BRIDGE_SCRIPT" ensure-whitelist-gate >>"$LOG" 2>&1
    gate_rc=$?
    set -e
    if [ "$gate_rc" -ne 0 ]; then
      echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pi_cycle_launchd fuse rc=$gate_rc reason=whitelist_gate_failed" >>"$LOG"
      exit "$gate_rc"
    fi
    ;;
esac

case "$(printf '%s' "$PI_LAUNCHD_GATE_ONLY" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y|on)
    run_retro_report
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] pi_cycle_launchd gate-only done rc=0" >>"$LOG"
    exit 0
    ;;
esac

cd "$ROOT"
set +e
"$PY_BIN" scripts/pi_cycle_orchestrator.py --guard-mode auto >>"$LOG" 2>&1
rc=$?
set -e

end_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
run_retro_report
echo "[$end_ts] pi_cycle_launchd done rc=$rc" >>"$LOG"
exit "$rc"
