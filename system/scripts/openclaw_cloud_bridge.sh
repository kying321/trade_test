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
  assert-whitelist-gate   Assert latest whitelist artifact gate without resampling.
  ensure-whitelist-gate   Assert first, then auto-resample+reassert on failure.
  live-takeover-probe     Run remote Binance takeover probe (config+evomap+telemetry, no order).
  live-takeover-canary    Run remote Binance takeover canary (includes minimal live order).
  live-takeover-ready-check  Check whether remote account is ready for canary order (balance + creds).
  live-takeover-autopilot  Ready-check first, place canary only when ready; otherwise skip gracefully.
  live-risk-guard         Run remote independent risk gate (ticket freshness / panic fuse / exposure / drawdown).
  live-risk-daemon-start  Start remote risk daemon (periodic fuse refresh).
  live-risk-daemon-status Show remote risk daemon status.
  live-risk-daemon-stop   Stop remote risk daemon.
  live-risk-daemon-install-service  Install and enable remote systemd service for risk daemon.
  live-risk-daemon-service-status   Show remote systemd service status for risk daemon.
  live-risk-daemon-security-status  Show remote systemd verify/security summary for risk daemon.
  live-risk-daemon-mdwe-probe  Run a remote transient MemoryDenyWriteExecute compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-protecthome-probe  Run a remote transient ProtectHome=read-only compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-procsubset-probe  Run a remote transient ProcSubset=pid compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-privateusers-probe  Run a remote transient PrivateUsers=true compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-privatenetwork-probe  Run a remote transient PrivateNetwork=true compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-ipdeny-probe  Run a remote transient IPAddressDeny=any compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-devicepolicy-probe  Run a remote transient DevicePolicy=closed compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-afunix-probe  Run a remote transient RestrictAddressFamilies=AF_UNIX compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-noaf-probe  Run a remote transient RestrictAddressFamilies=none compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-syscallfilter-probe  Run a remote transient SystemCallFilter=@system-service compatibility probe for the risk daemon without changing the installed service.
  live-risk-daemon-syscallfilter-tight-probe  Run a remote transient SystemCallFilter tightening probe (@system-service with ~@resources and ~@privileged) without changing the installed service.
  live-risk-daemon-journal  Show recent remote risk daemon journal/log lines.
  live-risk-daemon-remove-service   Disable and remove remote systemd service for risk daemon.
  live-ops-reconcile-status Show latest remote ops-report reconcile gate status.
  live-ops-reconcile-refresh Run remote ops-report and refresh reconcile artifact.
  remote-live-handoff      Generate a local read-only handoff artifact summarizing remote ready-check, risk daemon, ops gate, and journal context.
  remote-live-notification-preview  Generate a local read-only notification preview artifact from the latest remote live handoff.
  remote-live-notification-dry-run  Generate a local read-only dry-run notification artifact with validated Telegram/Feishu request bodies.
  remote-live-notification-send  Generate a local notification send artifact; defaults to delivery=none unless explicitly enabled.
  bootstrap-remote-runtime  Install remote Python runtime dependencies for Lie engine.
  sync-local-pi-workspace   Sync repo system/ into the local OpenClaw PI workspace (preserves output/ runtime state).
  publish-local-pi-runtime-scripts  Publish repo-managed PI runtime scripts into the local OpenClaw workspace scripts/ with run-halfhour-pulse mutex protection.
  prepare-local-pi-runtime  Sync local PI workspace, publish local PI runtime scripts, ensure runtime model, then run a gate-only launchd smoke.
  smoke-local-pi-cycle      Prepare local PI runtime, then run one full local PI half-hour cycle smoke.
  run-local-pi-recovery-lab  Copy current local PI paper state into an isolated lab and execute recovery writes there (backfill -> ack -> runtime consume/archive) without touching the real workspace.
  snapshot-local-pi-recovery-state  Create a mutex-protected checkpoint of local PI paper recovery state before any real mutation.
  restore-local-pi-recovery-state   Dry-run or restore local PI paper recovery state from a checkpoint manifest.
  rollback-local-pi-recovery-state  Dry-run or restore local PI paper recovery state from the latest checkpoint (or an explicit checkpoint).
  backfill-local-pi-last-loss-ts  Dry-run or write a local paper-state last_loss_ts backfill from paper_execution_ledger.
  local-pi-consecutive-loss-guardrail-status  Read-only local paper consecutive-loss guardrail status, including strict/fallback last_loss_ts backfill preview.
  local-pi-ack-archive-status  Read-only local paper ack live/archive status, including latest consumed archive and manifest tail.
  local-pi-recovery-handoff    Generate a single read-only handoff artifact summarizing local recovery state, latest checkpoint, archive, and retro.
  apply-local-pi-recovery-step  Execute or preview the recommended local paper recovery step derived from consecutive-loss guardrail status.
  run-local-pi-recovery-flow  Execute a bounded multi-step local paper recovery flow (backfill -> ack -> full-cycle), stopping at safe boundaries.
  ack-local-pi-consecutive-loss-guardrail  Dry-run or write a manual ack for the local paper consecutive-loss guardrail.
  ensure-local-openclaw-runtime-model  Ensure local ~/.openclaw/openclaw.json can resolve openai/gpt-5.4 through the 9999 proxy.
  ensure-remote-openclaw-runtime-model Ensure remote ~/.openclaw/openclaw.json can resolve openai/gpt-5.4 through the 9999 proxy.
  live-fast-skill         Run fast order+risk skill (plan + guarded live order + optional auto-close).
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
  FENLIE_SYSTEM_ROOT       optional absolute path to system root (preferred for launchd).
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
  WHITELIST_ASSERT_MAX_AGE_MINUTES   default: 90 (artifact freshness guard)
  LIVE_TAKEOVER_DATE       optional YYYY-MM-DD
  LIVE_TAKEOVER_CANARY_USDT          default: 5
  LIVE_TAKEOVER_MAX_DRAWDOWN         default: 0.05
  LIVE_TAKEOVER_RATE_LIMIT_PER_MINUTE default: 10
  LIVE_TAKEOVER_TIMEOUT_MS           default: 5000
  LIVE_TAKEOVER_TRADE_WINDOW_HOURS   default: 24
  LIVE_TAKEOVER_MARKET               default: spot (spot|futures_usdm)
  LIVE_TAKEOVER_ALLOW_DAEMON_ENV_FALLBACK default: true
  LIVE_TAKEOVER_FORWARD_LOCAL_CREDS  default: false (forward local BINANCE_API_KEY/BINANCE_SECRET to remote run env)
  LIVE_FAST_SKILL_SYMBOLS            default: BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XAUUSD
  LIVE_FAST_SKILL_MAX_AGE_DAYS       default: 14
  LIVE_FAST_SKILL_MIN_CONFIDENCE     optional override; default inherits config.yaml thresholds.signal_confidence_min
  LIVE_FAST_SKILL_MIN_CONVEXITY      optional override; default inherits config.yaml thresholds.convexity_min
  LIVE_FAST_SKILL_DECISION_TTL_SECONDS default: 300
  LIVE_FAST_SKILL_AUTO_CLOSE         default: true
  LIVE_FAST_SKILL_CLOSE_DELAY_SECONDS default: 0
  LIVE_RISK_GUARD_TICKET_FRESHNESS_SECONDS default: 900
  LIVE_RISK_GUARD_PANIC_COOLDOWN_SECONDS   default: 1800
  LIVE_RISK_GUARD_MAX_DAILY_LOSS_RATIO     default: 0.05
  LIVE_RISK_GUARD_MAX_OPEN_EXPOSURE_RATIO  default: 0.50
  LIVE_RISK_DAEMON_POLL_SECONDS            default: 60
  LIVE_RISK_DAEMON_GUARD_TIMEOUT_SECONDS   default: 45
  LIVE_RISK_DAEMON_HISTORY_LIMIT           default: 12
  LIVE_RISK_DAEMON_UNIT_NAME               default: fenlie-live-risk-daemon.service
  LIVE_RISK_DAEMON_JOURNAL_LINES           default: 80
  LIVE_RISK_DAEMON_SECURITY_ACCEPT_MAX_EXPOSURE default: 4.0
  LIVE_RISK_DAEMON_PROTECTHOME_PROBE_TIMEOUT_SECONDS default: 60
  LIVE_RISK_DAEMON_PROCSUBSET_PROBE_TIMEOUT_SECONDS default: 60
  LIVE_RISK_DAEMON_PRIVATEUSERS_PROBE_TIMEOUT_SECONDS default: 60
  LIVE_RISK_DAEMON_PRIVATENETWORK_PROBE_TIMEOUT_SECONDS default: 60
  LIVE_RISK_DAEMON_DEVICEPOLICY_PROBE_TIMEOUT_SECONDS default: 60
  LIVE_RISK_DAEMON_SYSCALLFILTER_PROBE_TIMEOUT_SECONDS default: 60
  LIVE_RISK_DAEMON_SYSCALLFILTER_TIGHT_PROBE_TIMEOUT_SECONDS default: 60
  LIVE_OPS_RECONCILE_MAX_AGE_HOURS         default: 48
  LIVE_OPS_RECONCILE_WINDOW_DAYS           default: 7
  LOCAL_PI_WORKSPACE_ROOT                  default: $HOME/.openclaw/workspaces/pi
  LOCAL_PI_WORKSPACE_SYSTEM_ROOT           default: $LOCAL_PI_WORKSPACE_ROOT/fenlie-system
  LOCAL_PI_WORKSPACE_BACKUP_KEEP           default: 5
  LOCAL_PI_WORKSPACE_BACKUP_MAX_AGE_HOURS  default: 168
  LOCAL_PI_WORKSPACE_DRY_RUN               default: false
  LOCAL_PI_WORKSPACE_NO_BACKUP             default: false
  LOCAL_PI_RUNTIME_SCRIPTS_SOURCE_ROOT     default: $FENLIE_SYSTEM_ROOT/runtime/pi/scripts
  LOCAL_PI_RUNTIME_SCRIPTS_TARGET_ROOT     default: $LOCAL_PI_WORKSPACE_ROOT/scripts
  LOCAL_PI_RUNTIME_SCRIPTS_BACKUP_KEEP     default: 5
  LOCAL_PI_RUNTIME_SCRIPTS_BACKUP_MAX_AGE_HOURS default: 168
  LOCAL_PI_RUNTIME_SCRIPTS_DRY_RUN         default: false
  LOCAL_PI_RUNTIME_SCRIPTS_NO_BACKUP       default: false
  LOCAL_PI_LAUNCHD_RUNNER_PATH             default: $HOME/.openclaw/workspaces/pi/scripts/pi_cycle_halfhour_launchd_runner.sh
  LOCAL_PI_LAUNCHD_LOG_PATH                default: $HOME/.openclaw/logs/pi_cycle_launchd.log
  LOCAL_PI_GATE_WINDOW_HOURS               default: 8
  LOCAL_PI_PREPARE_BEFORE_FULL_SMOKE       default: true
  LOCAL_PI_RECOVERY_LAB_PARENT_DIR         default: $LOCAL_PI_WORKSPACE_SYSTEM_ROOT/output/review
  LOCAL_PI_RECOVERY_LAB_KEEP               default: 6
  LOCAL_PI_RECOVERY_LAB_TTL_HOURS          default: 72
  LOCAL_PI_RECOVERY_LAB_ALLOW_FALLBACK_WRITE default: true
  LOCAL_PI_RECOVERY_CHECKPOINT_DIR         default: $LOCAL_PI_WORKSPACE_SYSTEM_ROOT/output/review/local_pi_recovery_checkpoints
  LOCAL_PI_RECOVERY_CHECKPOINT_KEEP        default: 12
  LOCAL_PI_RECOVERY_CHECKPOINT_MAX_AGE_HOURS default: 168
  LOCAL_PI_RECOVERY_CHECKPOINT_NOTE        optional operator note for snapshot
  LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT     required for restore-local-pi-recovery-state
  LOCAL_PI_RECOVERY_RESTORE_WRITE          default: false
  LOCAL_PI_RECOVERY_RESTORE_EXPECTED_STATE_FINGERPRINT optional compare-and-set for restore
  LOCAL_PI_LAST_LOSS_TS_BACKFILL_ALLOW_LATEST_LOSS_FALLBACK default: false
  LOCAL_PI_LAST_LOSS_TS_BACKFILL_WRITE     default: false
  LOCAL_PI_CONSECUTIVE_LOSS_STOP_THRESHOLD default: 3
  LOCAL_PI_CONSECUTIVE_LOSS_ACK_TTL_HOURS  default: 24
  LOCAL_PI_CONSECUTIVE_LOSS_ACK_COOLDOWN_HOURS default: 12
  LOCAL_PI_CONSECUTIVE_LOSS_ACK_ALLOW_MISSING_LAST_LOSS_TS default: false
  LOCAL_PI_CONSECUTIVE_LOSS_ACK_WRITE      default: false
  LOCAL_PI_CONSECUTIVE_LOSS_ACK_NOTE       optional operator note for ack artifact
  LOCAL_PI_ACK_ARCHIVE_MANIFEST_TAIL       default: 5
  LOCAL_PI_RECOVERY_HANDOFF_KEEP           default: 12
  LOCAL_PI_RECOVERY_APPLY_WRITE            default: false
  LOCAL_PI_RECOVERY_ALLOW_FALLBACK_WRITE   default: false
  LOCAL_PI_RECOVERY_RUN_FULL_CYCLE         default: false
  LOCAL_PI_RECOVERY_VERIFY_AFTER_STEP      default: true
  LOCAL_PI_RECOVERY_FLOW_MAX_STEPS         default: 3
  LOCAL_PI_RECOVERY_ENFORCE_PROJECTION     default: true
  LOCAL_PI_RECOVERY_AUTO_SNAPSHOT_BEFORE_WRITE default: true
  LOCAL_PI_RECOVERY_AUTO_SNAPSHOT_NOTE     optional operator note for auto snapshot
  LOCAL_PI_RECOVERY_AUTO_ROLLBACK_ON_FAILURE default: true
  LOCAL_PI_RECOVERY_AUTO_ROLLBACK_WRITE    default: false
  OPENCLAW_IDEMPOTENCY_TTL_SECONDS   default: 1800 (dedupe live actions)
  OPENCLAW_IDEMPOTENCY_MAX_ENTRIES   default: 500
  OPENCLAW_ARTIFACT_TTL_HOURS        default: 168 (whitelist artifact eviction)
  OPENCLAW_RSYNC_TIMEOUT_SECONDS     default: 5 (hard-capped at 5)

Notes:
  - SSH connect timeout is hard-limited to 5s for bridge reliability checks.
  - rsync timeout is hard-limited to 5s (OPENCLAW_RSYNC_TIMEOUT_SECONDS > 5 is clamped).
  - live-takeover-* transport fatal (ssh 255 / socket / 409 conflict) triggers panic marker + SIGKILL.
  - live-takeover-canary/autopilot/live-fast-skill are idempotent within OPENCLAW_IDEMPOTENCY_TTL_SECONDS.
  - live-fast-skill defaults to auto-close for canary round-trip cleanup.
  - sample-whitelist appends sample records to output/logs jsonl and renders json/md artifacts.
  - sample-whitelist writes SHA-256 checksum metadata and evicts old artifacts by TTL.
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_system_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
system_root="${FENLIE_SYSTEM_ROOT:-}"

if [[ -n "${system_root}" && ! -d "${system_root}" ]]; then
  echo "ERROR: FENLIE_SYSTEM_ROOT does not exist: ${system_root}" >&2
  exit 2
fi

if [[ -z "${system_root}" ]]; then
  if [[ -n "${repo_root}" && -d "${repo_root}/system" ]]; then
    system_root="${repo_root}/system"
  elif [[ -d "${script_system_root}/src" && -d "${script_system_root}/scripts" ]]; then
    system_root="${script_system_root}"
  else
    echo "ERROR: unable to resolve system root. Set FENLIE_SYSTEM_ROOT explicitly." >&2
    exit 2
  fi
fi

if [[ -z "${repo_root}" ]]; then
  repo_root="$(git -C "${system_root}/.." rev-parse --show-toplevel 2>/dev/null || true)"
fi
if [[ -z "${repo_root}" ]]; then
  repo_root="$(cd "${system_root}/.." && pwd)"
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
whitelist_assert_max_age_minutes="${WHITELIST_ASSERT_MAX_AGE_MINUTES:-90}"
live_takeover_date="${LIVE_TAKEOVER_DATE:-}"
live_takeover_canary_usdt="${LIVE_TAKEOVER_CANARY_USDT:-5}"
live_takeover_max_drawdown="${LIVE_TAKEOVER_MAX_DRAWDOWN:-0.05}"
live_takeover_rate_limit_per_minute="${LIVE_TAKEOVER_RATE_LIMIT_PER_MINUTE:-10}"
live_takeover_timeout_ms="${LIVE_TAKEOVER_TIMEOUT_MS:-5000}"
live_takeover_trade_window_hours="${LIVE_TAKEOVER_TRADE_WINDOW_HOURS:-24}"
live_takeover_market="$(printf '%s' "${LIVE_TAKEOVER_MARKET:-spot}" | tr '[:upper:]' '[:lower:]')"
live_takeover_allow_daemon_env_fallback="${LIVE_TAKEOVER_ALLOW_DAEMON_ENV_FALLBACK:-true}"
live_takeover_forward_local_creds="${LIVE_TAKEOVER_FORWARD_LOCAL_CREDS:-false}"
live_fast_skill_symbols="${LIVE_FAST_SKILL_SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XAUUSD}"
live_fast_skill_max_age_days="${LIVE_FAST_SKILL_MAX_AGE_DAYS:-14}"
live_fast_skill_min_confidence="${LIVE_FAST_SKILL_MIN_CONFIDENCE:-}"
live_fast_skill_min_convexity="${LIVE_FAST_SKILL_MIN_CONVEXITY:-}"
live_fast_skill_decision_ttl_seconds="${LIVE_FAST_SKILL_DECISION_TTL_SECONDS:-300}"
live_fast_skill_auto_close="${LIVE_FAST_SKILL_AUTO_CLOSE:-true}"
live_fast_skill_close_delay_seconds="${LIVE_FAST_SKILL_CLOSE_DELAY_SECONDS:-0}"
live_risk_guard_ticket_freshness_seconds="${LIVE_RISK_GUARD_TICKET_FRESHNESS_SECONDS:-900}"
live_risk_guard_panic_cooldown_seconds="${LIVE_RISK_GUARD_PANIC_COOLDOWN_SECONDS:-1800}"
live_risk_guard_max_daily_loss_ratio="${LIVE_RISK_GUARD_MAX_DAILY_LOSS_RATIO:-0.05}"
live_risk_guard_max_open_exposure_ratio="${LIVE_RISK_GUARD_MAX_OPEN_EXPOSURE_RATIO:-0.50}"
live_risk_daemon_poll_seconds="${LIVE_RISK_DAEMON_POLL_SECONDS:-60}"
live_risk_daemon_guard_timeout_seconds="${LIVE_RISK_DAEMON_GUARD_TIMEOUT_SECONDS:-45}"
live_risk_daemon_history_limit="${LIVE_RISK_DAEMON_HISTORY_LIMIT:-12}"
live_risk_daemon_unit_name="${LIVE_RISK_DAEMON_UNIT_NAME:-fenlie-live-risk-daemon.service}"
live_risk_daemon_journal_lines="${LIVE_RISK_DAEMON_JOURNAL_LINES:-80}"
live_risk_daemon_security_accept_max_exposure="${LIVE_RISK_DAEMON_SECURITY_ACCEPT_MAX_EXPOSURE:-4.0}"
live_risk_daemon_mdwe_probe_timeout_seconds="${LIVE_RISK_DAEMON_MDWE_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_protecthome_probe_timeout_seconds="${LIVE_RISK_DAEMON_PROTECTHOME_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_procsubset_probe_timeout_seconds="${LIVE_RISK_DAEMON_PROCSUBSET_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_privateusers_probe_timeout_seconds="${LIVE_RISK_DAEMON_PRIVATEUSERS_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_privatenetwork_probe_timeout_seconds="${LIVE_RISK_DAEMON_PRIVATENETWORK_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_ipdeny_probe_timeout_seconds="${LIVE_RISK_DAEMON_IPDENY_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_devicepolicy_probe_timeout_seconds="${LIVE_RISK_DAEMON_DEVICEPOLICY_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_afunix_probe_timeout_seconds="${LIVE_RISK_DAEMON_AFUNIX_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_noaf_probe_timeout_seconds="${LIVE_RISK_DAEMON_NOAF_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_syscallfilter_probe_timeout_seconds="${LIVE_RISK_DAEMON_SYSCALLFILTER_PROBE_TIMEOUT_SECONDS:-60}"
live_risk_daemon_syscallfilter_tight_probe_timeout_seconds="${LIVE_RISK_DAEMON_SYSCALLFILTER_TIGHT_PROBE_TIMEOUT_SECONDS:-60}"
live_ops_reconcile_max_age_hours="${LIVE_OPS_RECONCILE_MAX_AGE_HOURS:-48}"
live_ops_reconcile_window_days="${LIVE_OPS_RECONCILE_WINDOW_DAYS:-7}"
local_pi_workspace_root="${LOCAL_PI_WORKSPACE_ROOT:-${HOME}/.openclaw/workspaces/pi}"
local_pi_workspace_system_root="${LOCAL_PI_WORKSPACE_SYSTEM_ROOT:-${local_pi_workspace_root}/fenlie-system}"
local_pi_workspace_backup_keep="${LOCAL_PI_WORKSPACE_BACKUP_KEEP:-5}"
local_pi_workspace_backup_max_age_hours="${LOCAL_PI_WORKSPACE_BACKUP_MAX_AGE_HOURS:-168}"
local_pi_workspace_dry_run="${LOCAL_PI_WORKSPACE_DRY_RUN:-false}"
local_pi_workspace_no_backup="${LOCAL_PI_WORKSPACE_NO_BACKUP:-false}"
local_pi_runtime_scripts_source_root="${LOCAL_PI_RUNTIME_SCRIPTS_SOURCE_ROOT:-${system_root}/runtime/pi/scripts}"
local_pi_runtime_scripts_target_root="${LOCAL_PI_RUNTIME_SCRIPTS_TARGET_ROOT:-${local_pi_workspace_root}/scripts}"
local_pi_runtime_scripts_backup_keep="${LOCAL_PI_RUNTIME_SCRIPTS_BACKUP_KEEP:-5}"
local_pi_runtime_scripts_backup_max_age_hours="${LOCAL_PI_RUNTIME_SCRIPTS_BACKUP_MAX_AGE_HOURS:-168}"
local_pi_runtime_scripts_dry_run="${LOCAL_PI_RUNTIME_SCRIPTS_DRY_RUN:-false}"
local_pi_runtime_scripts_no_backup="${LOCAL_PI_RUNTIME_SCRIPTS_NO_BACKUP:-false}"
local_pi_launchd_runner_path="${LOCAL_PI_LAUNCHD_RUNNER_PATH:-${HOME}/.openclaw/workspaces/pi/scripts/pi_cycle_halfhour_launchd_runner.sh}"
local_pi_launchd_log_path="${LOCAL_PI_LAUNCHD_LOG_PATH:-${HOME}/.openclaw/logs/pi_cycle_launchd.log}"
local_pi_gate_window_hours="${LOCAL_PI_GATE_WINDOW_HOURS:-8}"
local_pi_prepare_before_full_smoke="${LOCAL_PI_PREPARE_BEFORE_FULL_SMOKE:-true}"
local_pi_recovery_lab_parent_dir="${LOCAL_PI_RECOVERY_LAB_PARENT_DIR:-${local_pi_workspace_system_root}/output/review}"
local_pi_recovery_lab_keep="${LOCAL_PI_RECOVERY_LAB_KEEP:-6}"
local_pi_recovery_lab_ttl_hours="${LOCAL_PI_RECOVERY_LAB_TTL_HOURS:-72}"
local_pi_recovery_lab_allow_fallback_write="${LOCAL_PI_RECOVERY_LAB_ALLOW_FALLBACK_WRITE:-true}"
local_pi_recovery_checkpoint_dir="${LOCAL_PI_RECOVERY_CHECKPOINT_DIR:-${local_pi_workspace_system_root}/output/review/local_pi_recovery_checkpoints}"
local_pi_recovery_checkpoint_keep="${LOCAL_PI_RECOVERY_CHECKPOINT_KEEP:-12}"
local_pi_recovery_checkpoint_max_age_hours="${LOCAL_PI_RECOVERY_CHECKPOINT_MAX_AGE_HOURS:-168}"
local_pi_recovery_checkpoint_note="${LOCAL_PI_RECOVERY_CHECKPOINT_NOTE:-}"
local_pi_recovery_restore_checkpoint="${LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT:-}"
local_pi_recovery_restore_write="${LOCAL_PI_RECOVERY_RESTORE_WRITE:-false}"
local_pi_recovery_restore_expected_state_fingerprint="${LOCAL_PI_RECOVERY_RESTORE_EXPECTED_STATE_FINGERPRINT:-}"
local_pi_last_loss_ts_backfill_allow_latest_loss_fallback="${LOCAL_PI_LAST_LOSS_TS_BACKFILL_ALLOW_LATEST_LOSS_FALLBACK:-false}"
local_pi_last_loss_ts_backfill_write="${LOCAL_PI_LAST_LOSS_TS_BACKFILL_WRITE:-false}"
local_pi_consecutive_loss_stop_threshold="${LOCAL_PI_CONSECUTIVE_LOSS_STOP_THRESHOLD:-3}"
local_pi_consecutive_loss_ack_ttl_hours="${LOCAL_PI_CONSECUTIVE_LOSS_ACK_TTL_HOURS:-24}"
local_pi_consecutive_loss_ack_cooldown_hours="${LOCAL_PI_CONSECUTIVE_LOSS_ACK_COOLDOWN_HOURS:-12}"
local_pi_consecutive_loss_ack_allow_missing_last_loss_ts="${LOCAL_PI_CONSECUTIVE_LOSS_ACK_ALLOW_MISSING_LAST_LOSS_TS:-false}"
local_pi_consecutive_loss_ack_write="${LOCAL_PI_CONSECUTIVE_LOSS_ACK_WRITE:-false}"
local_pi_consecutive_loss_ack_note="${LOCAL_PI_CONSECUTIVE_LOSS_ACK_NOTE:-}"
local_pi_ack_archive_manifest_tail="${LOCAL_PI_ACK_ARCHIVE_MANIFEST_TAIL:-5}"
local_pi_recovery_handoff_keep="${LOCAL_PI_RECOVERY_HANDOFF_KEEP:-12}"
remote_live_handoff_keep="${REMOTE_LIVE_HANDOFF_KEEP:-12}"
remote_live_notification_delivery="${REMOTE_LIVE_NOTIFICATION_DELIVERY:-none}"
remote_live_notification_timeout_ms="${REMOTE_LIVE_NOTIFICATION_TIMEOUT_MS:-5000}"
remote_live_notification_rate_limit_per_minute="${REMOTE_LIVE_NOTIFICATION_RATE_LIMIT_PER_MINUTE:-6}"
remote_live_notification_idempotency_ttl_seconds="${REMOTE_LIVE_NOTIFICATION_IDEMPOTENCY_TTL_SECONDS:-1800}"
remote_live_notification_idempotency_max_entries="${REMOTE_LIVE_NOTIFICATION_IDEMPOTENCY_MAX_ENTRIES:-200}"
local_pi_recovery_apply_write="${LOCAL_PI_RECOVERY_APPLY_WRITE:-false}"
local_pi_recovery_allow_fallback_write="${LOCAL_PI_RECOVERY_ALLOW_FALLBACK_WRITE:-false}"
local_pi_recovery_run_full_cycle="${LOCAL_PI_RECOVERY_RUN_FULL_CYCLE:-false}"
local_pi_recovery_verify_after_step="${LOCAL_PI_RECOVERY_VERIFY_AFTER_STEP:-true}"
local_pi_recovery_flow_max_steps="${LOCAL_PI_RECOVERY_FLOW_MAX_STEPS:-3}"
local_pi_recovery_enforce_projection="${LOCAL_PI_RECOVERY_ENFORCE_PROJECTION:-true}"
local_pi_recovery_auto_snapshot_before_write="${LOCAL_PI_RECOVERY_AUTO_SNAPSHOT_BEFORE_WRITE:-true}"
local_pi_recovery_auto_snapshot_note="${LOCAL_PI_RECOVERY_AUTO_SNAPSHOT_NOTE:-}"
local_pi_recovery_auto_rollback_on_failure="${LOCAL_PI_RECOVERY_AUTO_ROLLBACK_ON_FAILURE:-true}"
local_pi_recovery_auto_rollback_write="${LOCAL_PI_RECOVERY_AUTO_ROLLBACK_WRITE:-false}"
local_pi_recovery_artifacts_enabled="${LOCAL_PI_RECOVERY_ARTIFACTS_ENABLED:-true}"
local_pi_recovery_artifact_dir="${LOCAL_PI_RECOVERY_ARTIFACT_DIR:-${local_pi_workspace_system_root}/output/review}"

if [[ "${live_takeover_market}" != "spot" && "${live_takeover_market}" != "futures_usdm" ]]; then
  echo "ERROR: LIVE_TAKEOVER_MARKET must be one of: spot, futures_usdm." >&2
  exit 2
fi

output_dir="${system_root}/output/review"
log_dir="${system_root}/output/logs"
state_dir="${system_root}/output/state"
samples_log="${log_dir}/openclaw_bridge_whitelist_samples.jsonl"
tunnel_socket="${TMPDIR:-/tmp}/openclaw_bridge_${cloud_host//./_}.sock"
idempotency_lock="${state_dir}/run-halfhour-pulse.lock"
idempotency_ledger="${state_dir}/openclaw_bridge_idempotency.json"
idempotency_ttl_seconds="${OPENCLAW_IDEMPOTENCY_TTL_SECONDS:-1800}"
idempotency_max_entries="${OPENCLAW_IDEMPOTENCY_MAX_ENTRIES:-500}"
artifact_ttl_hours="${OPENCLAW_ARTIFACT_TTL_HOURS:-168}"

mkdir -p "${output_dir}" "${log_dir}" "${state_dir}"

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
if ! [[ "${whitelist_assert_max_age_minutes}" =~ ^[0-9]+$ ]] || (( whitelist_assert_max_age_minutes <= 0 )); then
  echo "ERROR: WHITELIST_ASSERT_MAX_AGE_MINUTES must be a positive integer." >&2
  exit 2
fi
if ! [[ "${idempotency_ttl_seconds}" =~ ^[0-9]+$ ]] || (( idempotency_ttl_seconds <= 0 )); then
  echo "ERROR: OPENCLAW_IDEMPOTENCY_TTL_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${idempotency_max_entries}" =~ ^[0-9]+$ ]] || (( idempotency_max_entries < 100 )); then
  echo "ERROR: OPENCLAW_IDEMPOTENCY_MAX_ENTRIES must be an integer >= 100." >&2
  exit 2
fi
if ! [[ "${artifact_ttl_hours}" =~ ^[0-9]+$ ]] || (( artifact_ttl_hours <= 0 )); then
  echo "ERROR: OPENCLAW_ARTIFACT_TTL_HOURS must be a positive integer." >&2
  exit 2
fi
if [[ "${remote_live_notification_delivery}" != "none" && "${remote_live_notification_delivery}" != "telegram" && "${remote_live_notification_delivery}" != "feishu" && "${remote_live_notification_delivery}" != "all" ]]; then
  echo "ERROR: REMOTE_LIVE_NOTIFICATION_DELIVERY must be one of: none, telegram, feishu, all." >&2
  exit 2
fi
if ! [[ "${remote_live_notification_timeout_ms}" =~ ^[0-9]+$ ]] || (( remote_live_notification_timeout_ms <= 0 )); then
  echo "ERROR: REMOTE_LIVE_NOTIFICATION_TIMEOUT_MS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${remote_live_notification_rate_limit_per_minute}" =~ ^[0-9]+$ ]] || (( remote_live_notification_rate_limit_per_minute <= 0 )); then
  echo "ERROR: REMOTE_LIVE_NOTIFICATION_RATE_LIMIT_PER_MINUTE must be a positive integer." >&2
  exit 2
fi
if ! [[ "${remote_live_notification_idempotency_ttl_seconds}" =~ ^[0-9]+$ ]] || (( remote_live_notification_idempotency_ttl_seconds <= 0 )); then
  echo "ERROR: REMOTE_LIVE_NOTIFICATION_IDEMPOTENCY_TTL_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${remote_live_notification_idempotency_max_entries}" =~ ^[0-9]+$ ]] || (( remote_live_notification_idempotency_max_entries <= 0 )); then
  echo "ERROR: REMOTE_LIVE_NOTIFICATION_IDEMPOTENCY_MAX_ENTRIES must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_fast_skill_max_age_days}" =~ ^[0-9]+$ ]] || (( live_fast_skill_max_age_days <= 0 )); then
  echo "ERROR: LIVE_FAST_SKILL_MAX_AGE_DAYS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_fast_skill_decision_ttl_seconds}" =~ ^[0-9]+$ ]] || (( live_fast_skill_decision_ttl_seconds <= 0 )); then
  echo "ERROR: LIVE_FAST_SKILL_DECISION_TTL_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_fast_skill_close_delay_seconds}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: LIVE_FAST_SKILL_CLOSE_DELAY_SECONDS must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "${live_risk_guard_ticket_freshness_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_guard_ticket_freshness_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_GUARD_TICKET_FRESHNESS_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_guard_panic_cooldown_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_guard_panic_cooldown_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_GUARD_PANIC_COOLDOWN_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_poll_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_poll_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_POLL_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_guard_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_guard_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_GUARD_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_history_limit}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_history_limit <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_HISTORY_LIMIT must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_mdwe_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_mdwe_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_MDWE_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_protecthome_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_protecthome_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_PROTECTHOME_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_procsubset_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_procsubset_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_PROCSUBSET_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_privateusers_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_privateusers_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_PRIVATEUSERS_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_privatenetwork_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_privatenetwork_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_PRIVATENETWORK_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_ipdeny_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_ipdeny_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_IPDENY_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_devicepolicy_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_devicepolicy_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_DEVICEPOLICY_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_afunix_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_afunix_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_AFUNIX_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_noaf_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_noaf_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_NOAF_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_syscallfilter_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_syscallfilter_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_SYSCALLFILTER_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_syscallfilter_tight_probe_timeout_seconds}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_syscallfilter_tight_probe_timeout_seconds <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_SYSCALLFILTER_TIGHT_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ -z "${live_risk_daemon_unit_name}" ]]; then
  echo "ERROR: LIVE_RISK_DAEMON_UNIT_NAME must be non-empty." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_journal_lines}" =~ ^[0-9]+$ ]] || (( live_risk_daemon_journal_lines <= 0 )); then
  echo "ERROR: LIVE_RISK_DAEMON_JOURNAL_LINES must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_risk_daemon_security_accept_max_exposure}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: LIVE_RISK_DAEMON_SECURITY_ACCEPT_MAX_EXPOSURE must be a numeric value." >&2
  exit 2
fi
if ! python3 - "${live_risk_daemon_security_accept_max_exposure}" <<'PY' >/dev/null 2>&1
import sys
value = float(sys.argv[1])
raise SystemExit(0 if 0.0 <= value <= 10.0 else 1)
PY
then
  echo "ERROR: LIVE_RISK_DAEMON_SECURITY_ACCEPT_MAX_EXPOSURE must be between 0.0 and 10.0." >&2
  exit 2
fi
if ! [[ "${live_ops_reconcile_max_age_hours}" =~ ^[0-9]+$ ]] || (( live_ops_reconcile_max_age_hours <= 0 )); then
  echo "ERROR: LIVE_OPS_RECONCILE_MAX_AGE_HOURS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${live_ops_reconcile_window_days}" =~ ^[0-9]+$ ]] || (( live_ops_reconcile_window_days <= 0 )); then
  echo "ERROR: LIVE_OPS_RECONCILE_WINDOW_DAYS must be a positive integer." >&2
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
validate_rate_0_1 "LIVE_RISK_GUARD_MAX_DAILY_LOSS_RATIO" "${live_risk_guard_max_daily_loss_ratio}"
validate_rate_0_1 "LIVE_RISK_GUARD_MAX_OPEN_EXPOSURE_RATIO" "${live_risk_guard_max_open_exposure_ratio}"

validate_positive_float() {
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
if val <= 0.0:
    print(f"ERROR: {name} must be > 0.", file=sys.stderr)
    raise SystemExit(2)
PY
}

if [[ -n "${live_fast_skill_min_confidence}" ]]; then
  validate_positive_float "LIVE_FAST_SKILL_MIN_CONFIDENCE" "${live_fast_skill_min_confidence}"
fi
if [[ -n "${live_fast_skill_min_convexity}" ]]; then
  validate_positive_float "LIVE_FAST_SKILL_MIN_CONVEXITY" "${live_fast_skill_min_convexity}"
fi

is_true() {
  local raw
  raw="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

build_live_takeover_idempotency_material() {
  local date_norm
  date_norm="${live_takeover_date:-auto}"
  printf 'host=%s|user=%s|project=%s|date=%s|market=%s|canary_usdt=%s|max_dd=%s|window_h=%s' \
    "${cloud_host}" \
    "${cloud_user}" \
    "${cloud_project_dir}" \
    "${date_norm}" \
    "${live_takeover_market}" \
    "${live_takeover_canary_usdt}" \
    "${live_takeover_max_drawdown}" \
    "${live_takeover_trade_window_hours}"
}

idempotency_guard() {
  local action="$1"
  local material="$2"
  python3 - "${idempotency_ledger}" "${idempotency_lock}" "${action}" "${material}" "${idempotency_ttl_seconds}" "${idempotency_max_entries}" <<'PY'
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

ledger_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
action = str(sys.argv[3]).strip() or "unknown"
material = str(sys.argv[4])
ttl_seconds = max(1.0, float(sys.argv[5]))
max_entries = max(100, int(sys.argv[6]))

now_epoch = time.time()
now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
key_seed = f"{action}|{material}"
idem_key = hashlib.sha256(key_seed.encode("utf-8")).hexdigest()
material_sha = hashlib.sha256(material.encode("utf-8")).hexdigest()

lock_path.parent.mkdir(parents=True, exist_ok=True)
lock_fd = lock_path.open("a+", encoding="utf-8")
try:
    import fcntl

    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
    payload: dict[str, object]
    if ledger_path.exists():
        try:
            loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
            payload = loaded if isinstance(loaded, dict) else {}
        except Exception:
            payload = {}
    else:
        payload = {}
    entries_raw = payload.get("entries", {})
    entries = entries_raw if isinstance(entries_raw, dict) else {}

    stale_keys: list[str] = []
    for key, row in entries.items():
        if not isinstance(row, dict):
            stale_keys.append(str(key))
            continue
        try:
            expires_epoch = float(row.get("expires_epoch", 0.0))
        except Exception:
            expires_epoch = 0.0
        if expires_epoch <= now_epoch:
            stale_keys.append(str(key))
    for key in stale_keys:
        entries.pop(key, None)

    hit = entries.get(idem_key)
    if isinstance(hit, dict):
        out = {
            "allowed": False,
            "reason": "idempotent_hit",
            "action": action,
            "idempotency_key": idem_key,
            "ledger_path": str(ledger_path),
            "hit": hit,
            "generated_at_utc": now_utc,
        }
        payload["entries"] = entries
        payload["updated_at_utc"] = now_utc
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, ledger_path)
        print(json.dumps(out, ensure_ascii=False))
        raise SystemExit(3)

    entries[idem_key] = {
        "action": action,
        "material_sha256": material_sha,
        "created_at_utc": now_utc,
        "created_epoch": now_epoch,
        "expires_epoch": now_epoch + ttl_seconds,
    }
    if len(entries) > max_entries:
        ranked = sorted(
            entries.items(),
            key=lambda kv: float(kv[1].get("expires_epoch", 0.0)) if isinstance(kv[1], dict) else 0.0,
        )
        entries = dict(ranked[-max_entries:])
    payload["entries"] = entries
    payload["updated_at_utc"] = now_utc
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, ledger_path)
    out = {
        "allowed": True,
        "action": action,
        "idempotency_key": idem_key,
        "ledger_path": str(ledger_path),
        "ttl_seconds": ttl_seconds,
        "generated_at_utc": now_utc,
    }
    print(json.dumps(out, ensure_ascii=False))
finally:
    try:
        import fcntl

        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    lock_fd.close()
PY
}

panic_spinal_reflex() {
  local reason="$1"
  local detail="$2"
  local panic_path="${state_dir}/openclaw_bridge_panic.json"
  local compact_detail
  compact_detail="$(printf '%s' "${detail}" | tr '\n\r\t' '   ' | tr -s ' ')"
  if [[ -z "${compact_detail}" ]]; then
    compact_detail="reason=${reason}; detail=none"
  fi
  if (( ${#compact_detail} > 900 )); then
    compact_detail="${compact_detail:0:900}"
  fi
  python3 - "${panic_path}" "${reason}" "${compact_detail}" <<'PY'
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
reason = str(sys.argv[2])
detail = str(sys.argv[3])
payload = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "action": "panic_spinal_reflex",
    "reason": reason,
    "detail": detail,
}
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, path)
print(str(path))
PY
  echo "PANIC: ${reason}; detail=${compact_detail}" >&2
  kill -9 "$$"
}

run_live_takeover_remote() {
  local action="$1"
  local remote_cmd="$2"
  local output rc
  set +e
  output="$(ssh_exec "${remote_cmd}" 2>&1)"
  rc=$?
  set -e
  if (( rc == 0 )); then
    printf '%s' "${output}"
    return 0
  fi

  local lower
  lower="$(printf '%s' "${output}" | tr '[:upper:]' '[:lower:]')"
  if (( rc == 255 )) || grep -Eq "(409 conflict|socket|broken pipe|connection reset|connection refused|timed out|network is unreachable|kex_exchange_identification)" <<<"${lower}"; then
    panic_spinal_reflex \
      "openclaw_live_transport_failure" \
      "action=${action}; rc=${rc}; output=${output}"
  fi
  printf '%s' "${output}" >&2
  return ${rc}
}

build_live_takeover_cred_env_arg() {
  if ! is_true "${live_takeover_forward_local_creds}"; then
    echo ""
    return 0
  fi
  local api_key api_secret
  api_key="${BINANCE_API_KEY:-${BINANCE_KEY:-}}"
  api_secret="${BINANCE_SECRET_KEY:-${BINANCE_API_SECRET:-${BINANCE_SECRET:-}}}"
  if [[ -z "${api_key}" || -z "${api_secret}" ]]; then
    echo "ERROR: LIVE_TAKEOVER_FORWARD_LOCAL_CREDS=true requires local BINANCE_API_KEY and BINANCE_SECRET/BINANCE_API_SECRET." >&2
    return 2
  fi
  printf 'BINANCE_API_KEY=%q BINANCE_SECRET=%q ' "${api_key}" "${api_secret}"
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
  local rc
  opts=("${ssh_opts[@]}")
  if [[ -z "${cloud_pass}" ]]; then
    opts+=(-o BatchMode=yes)
  fi
  if [[ -n "${cloud_pass}" ]] && command -v sshpass >/dev/null 2>&1; then
    set +e
    SSHPASS="${cloud_pass}" sshpass -e ssh "${opts[@]}" "${cloud_user}@${cloud_host}" "${remote_cmd}"
    rc=$?
    set -e
    if (( rc == 255 )); then
      sleep 1
      set +e
      SSHPASS="${cloud_pass}" sshpass -e ssh "${opts[@]}" "${cloud_user}@${cloud_host}" "${remote_cmd}"
      rc=$?
      set -e
      return ${rc}
    fi
    return ${rc}
  else
    set +e
    ssh "${opts[@]}" "${cloud_user}@${cloud_host}" "${remote_cmd}"
    rc=$?
    set -e
    if (( rc == 255 )); then
      sleep 1
      set +e
      ssh "${opts[@]}" "${cloud_user}@${cloud_host}" "${remote_cmd}"
      rc=$?
      set -e
      return ${rc}
    fi
    return ${rc}
  fi
}

rsync_exec() {
  local ssh_cmd
  local rsync_timeout
  rsync_timeout="${OPENCLAW_RSYNC_TIMEOUT_SECONDS:-5}"
  if ! [[ "${rsync_timeout}" =~ ^[0-9]+$ ]] || (( rsync_timeout <= 0 )); then
    rsync_timeout=5
  fi
  if (( rsync_timeout > 5 )); then
    rsync_timeout=5
  fi
  ssh_cmd="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=2"
  if [[ -n "${cloud_pass}" ]] && command -v sshpass >/dev/null 2>&1; then
    SSHPASS="${cloud_pass}" rsync --contimeout=5 --timeout="${rsync_timeout}" -e "sshpass -e ${ssh_cmd}" "$@"
  else
    rsync --contimeout=5 --timeout="${rsync_timeout}" -e "${ssh_cmd} -o BatchMode=yes" "$@"
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
live-takeover-probe
live-takeover-canary
live-takeover-ready-check
live-takeover-autopilot
live-risk-guard
live-risk-daemon-start
live-risk-daemon-status
live-risk-daemon-stop
live-risk-daemon-install-service
live-risk-daemon-service-status
live-risk-daemon-journal
live-risk-daemon-remove-service
live-ops-reconcile-status
live-ops-reconcile-refresh
bootstrap-remote-runtime
ensure-local-openclaw-runtime-model
ensure-remote-openclaw-runtime-model
ack-local-pi-consecutive-loss-guardrail
publish-local-pi-runtime-scripts
backfill-local-pi-last-loss-ts
local-pi-consecutive-loss-guardrail-status
apply-local-pi-recovery-step
run-local-pi-recovery-flow
live-fast-skill
sample-whitelist
sample-whitelist-gate
assert-whitelist-gate
ensure-whitelist-gate
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

run_live_risk_guard_remote() {
  local date_arg min_conf_arg min_conv_arg
  date_arg=""
  min_conf_arg=""
  min_conv_arg=""
  if [[ -n "${live_takeover_date}" ]]; then
    date_arg="--date ${live_takeover_date}"
  fi
  if [[ -n "${live_fast_skill_min_confidence}" ]]; then
    min_conf_arg="--ticket-min-confidence ${live_fast_skill_min_confidence}"
  fi
  if [[ -n "${live_fast_skill_min_convexity}" ]]; then
    min_conv_arg="--ticket-min-convexity ${live_fast_skill_min_convexity}"
  fi
  run_live_takeover_remote \
    "live-risk-guard" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; PYTHONPATH=src python3 scripts/live_risk_guard.py ${date_arg} --config config.yaml --output-root output --review-dir output/review --ticket-freshness-seconds ${live_risk_guard_ticket_freshness_seconds} --panic-cooldown-seconds ${live_risk_guard_panic_cooldown_seconds} --max-daily-loss-ratio ${live_risk_guard_max_daily_loss_ratio} --max-open-exposure-ratio ${live_risk_guard_max_open_exposure_ratio} --refresh-tickets --ticket-symbols ${live_fast_skill_symbols} --ticket-max-age-days ${live_fast_skill_max_age_days} ${min_conf_arg} ${min_conv_arg}"
}

run_guarded_exec_remote() {
  local guard_mode="$1"
  local allow_live_flag="$2"
  local cred_env_arg="$3"
  local daemon_env_arg="$4"
  local date_arg="$5"
  local allow_live_arg min_conf_arg min_conv_arg
  allow_live_arg=""
  min_conf_arg=""
  min_conv_arg=""
  if [[ "${allow_live_flag}" == "true" ]]; then
    allow_live_arg="--allow-live-order"
  fi
  if [[ -n "${live_fast_skill_min_confidence}" ]]; then
    min_conf_arg="--ticket-min-confidence ${live_fast_skill_min_confidence}"
  fi
  if [[ -n "${live_fast_skill_min_convexity}" ]]; then
    min_conv_arg="--ticket-min-convexity ${live_fast_skill_min_convexity}"
  fi
  run_live_takeover_remote \
    "guarded-exec-${guard_mode}" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; ${cred_env_arg}PYTHONPATH=src python3 scripts/guarded_exec.py ${date_arg} --mode ${guard_mode} --market ${live_takeover_market} --canary-quote-usdt ${live_takeover_canary_usdt} --rate-limit-per-minute ${live_takeover_rate_limit_per_minute} --timeout-ms ${live_takeover_timeout_ms} --max-drawdown ${live_takeover_max_drawdown} --trade-window-hours ${live_takeover_trade_window_hours} --risk-fuse-max-age-seconds ${live_risk_guard_ticket_freshness_seconds} --panic-cooldown-seconds ${live_risk_guard_panic_cooldown_seconds} --max-daily-loss-ratio ${live_risk_guard_max_daily_loss_ratio} --max-open-exposure-ratio ${live_risk_guard_max_open_exposure_ratio} --refresh-tickets --ticket-symbols ${live_fast_skill_symbols} --ticket-max-age-days ${live_fast_skill_max_age_days} ${min_conf_arg} ${min_conv_arg} --idempotency-ttl-seconds ${idempotency_ttl_seconds} --idempotency-max-entries ${idempotency_max_entries} ${allow_live_arg} ${daemon_env_arg}"
}

action_live_risk_guard() {
  run_live_risk_guard_remote
}

run_live_risk_daemon_status_remote() {
  run_live_takeover_remote \
    "live-risk-daemon-status" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; python3 - 'output/state/live_risk_daemon.json' <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {}
if path.exists():
    try:
        loaded = json.loads(path.read_text(encoding='utf-8'))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        payload = {'status': 'invalid_state', 'error': str(exc)}
pid = int(payload.get('pid', 0) or 0)
alive = False
if pid > 0:
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
out = {
    'action': 'live-risk-daemon-status',
    'state_path': str(path),
    'exists': bool(path.exists()),
    'running': bool(payload.get('running', False)) and bool(alive),
    'pid': pid,
    'pid_alive': bool(alive),
    'status': str(payload.get('status', 'missing') if path.exists() else 'missing'),
    'payload': payload,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

run_live_risk_daemon_service_status_remote() {
  run_live_takeover_remote \
    "live-risk-daemon-service-status" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${live_risk_daemon_unit_name}' <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

unit_name = str(sys.argv[1]).strip()
state_path = Path('output/state/live_risk_daemon.json')
unit_path = Path('/etc/systemd/system') / unit_name

payload = {}
if state_path.exists():
    try:
        loaded = json.loads(state_path.read_text(encoding='utf-8'))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        payload = {'status': 'invalid_state', 'error': str(exc)}

installed = unit_path.exists()
show_map = {}
if installed:
    proc = subprocess.run(
        ['systemctl', 'show', unit_name, '-p', 'Id', '-p', 'ActiveState', '-p', 'SubState', '-p', 'UnitFileState', '-p', 'MainPID', '-p', 'FragmentPath'],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in (proc.stdout or '').splitlines():
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        show_map[k] = v

systemd_main_pid = int(show_map.get('MainPID', '0') or 0)
systemd_active = str(show_map.get('ActiveState', 'inactive' if installed else 'missing'))
payload_pid = int(payload.get('pid', 0) or 0)
payload_running = bool(payload.get('running', False))
payload_pid_alive = False
if payload_pid > 0:
    try:
        os.kill(payload_pid, 0)
        payload_pid_alive = True
    except OSError:
        payload_pid_alive = False
payload_updated_at = str(payload.get('updated_at_utc') or '')
payload_age_seconds = None
if state_path.exists():
    try:
        payload_age_seconds = max(0.0, time.time() - float(state_path.stat().st_mtime))
    except OSError:
        payload_age_seconds = None
payload_alignment_reasons = []
if not state_path.exists():
    payload_alignment_reasons.append('state_missing')
if installed and systemd_active == 'active' and systemd_main_pid <= 0:
    payload_alignment_reasons.append('systemd_main_pid_missing')
if installed and systemd_active == 'active' and payload_pid != systemd_main_pid:
    payload_alignment_reasons.append(
        f'payload_pid_mismatch(payload={payload_pid},systemd={systemd_main_pid})'
    )
if installed and systemd_active == 'active' and not payload_running:
    payload_alignment_reasons.append('payload_not_running')
if installed and systemd_active == 'active' and not payload_pid_alive:
    payload_alignment_reasons.append('payload_pid_not_alive')

payload_alignment = {
    'aligned': not payload_alignment_reasons,
    'systemd_active_state': systemd_active,
    'systemd_main_pid': systemd_main_pid,
    'payload_pid': payload_pid,
    'payload_running': payload_running,
    'payload_pid_alive': payload_pid_alive,
    'payload_updated_at_utc': payload_updated_at,
    'payload_age_seconds': payload_age_seconds,
    'reasons': payload_alignment_reasons,
}

out = {
    'action': 'live-risk-daemon-service-status',
    'service_installed': bool(installed),
    'unit_name': unit_name,
    'unit_path': str(unit_path),
    'systemd': {
        'id': show_map.get('Id', unit_name),
        'active_state': show_map.get('ActiveState', 'inactive' if installed else 'missing'),
        'sub_state': show_map.get('SubState', ''),
        'unit_file_state': show_map.get('UnitFileState', ''),
        'main_pid': int(show_map.get('MainPID', '0') or 0),
        'fragment_path': show_map.get('FragmentPath', str(unit_path) if installed else ''),
    },
    'payload_alignment': payload_alignment,
    'payload': payload,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

run_live_risk_daemon_security_status_remote() {
  run_live_takeover_remote \
    "live-risk-daemon-security-status" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${live_risk_daemon_unit_name}' <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

unit_name = str(sys.argv[1]).strip()
unit_path = Path('/etc/systemd/system') / unit_name

def tail_lines(text: str, limit: int = 12):
    return [x.rstrip() for x in str(text or '').splitlines() if x.strip()][-limit:]

verify = {'ok': False, 'returncode': 127, 'stderr_tail': [], 'stdout_tail': [], 'target': str(unit_path)}
security = {
    'returncode': 127,
    'overall_exposure': None,
    'overall_rating': None,
    'summary_line': '',
    'findings': [],
    'stdout_tail': [],
    'stderr_tail': [],
}

if unit_path.exists():
    proc_verify = subprocess.run(
        ['systemd-analyze', 'verify', str(unit_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    verify = {
        'ok': proc_verify.returncode == 0,
        'returncode': int(proc_verify.returncode),
        'stderr_tail': tail_lines(proc_verify.stderr),
        'stdout_tail': tail_lines(proc_verify.stdout),
        'target': str(unit_path),
    }

    proc_security = subprocess.run(
        ['systemd-analyze', 'security', unit_name],
        text=True,
        capture_output=True,
        check=False,
    )
    lines = [x.rstrip() for x in (proc_security.stdout or '').splitlines() if x.strip()]
    summary_line = ''
    score = None
    rating = None
    for line in reversed(lines):
        if 'Overall exposure level for ' not in line:
            continue
        summary_line = line.strip()
        match = re.search(r':\\s*([0-9]+(?:\\.[0-9]+)?)\\s+([A-Za-z]+)', summary_line)
        if match:
            try:
                score = float(match.group(1))
            except ValueError:
                score = None
            rating = match.group(2)
        break
    security = {
        'returncode': int(proc_security.returncode),
        'overall_exposure': score,
        'overall_rating': rating,
        'summary_line': summary_line,
        'findings': [line.strip() for line in lines if line.lstrip().startswith('✗')][:10],
        'stdout_tail': tail_lines(proc_security.stdout),
        'stderr_tail': tail_lines(proc_security.stderr),
    }

out = {
    'action': 'live-risk-daemon-security-status',
    'service_installed': bool(unit_path.exists()),
    'unit_name': unit_name,
    'unit_path': str(unit_path),
    'verify': verify,
    'security': security,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

run_live_risk_daemon_journal_remote() {
  run_live_takeover_remote \
    "live-risk-daemon-journal" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${live_risk_daemon_unit_name}' '${live_risk_daemon_journal_lines}' <<'PY'
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

unit_name = str(sys.argv[1]).strip()
line_limit = max(1, int(sys.argv[2]))
unit_path = Path('/etc/systemd/system') / unit_name
state_path = Path('output/state/live_risk_daemon.json')
log_path = Path('output/logs/live_risk_daemon.out.log')

payload = {}
if state_path.exists():
    try:
        loaded = json.loads(state_path.read_text(encoding='utf-8'))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        payload = {'status': 'invalid_state', 'error': str(exc)}

show_map = {}
if unit_path.exists():
    proc = subprocess.run(
        ['systemctl', 'show', unit_name, '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID'],
        text=True,
        capture_output=True,
        check=False,
    )
    for line in (proc.stdout or '').splitlines():
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        show_map[k] = v

source = 'missing'
lines = []
journal_rc = None
if unit_path.exists():
    proc = subprocess.run(
        ['journalctl', '-u', unit_name, '-n', str(line_limit), '--no-pager', '-o', 'short-iso'],
        text=True,
        capture_output=True,
        check=False,
    )
    journal_rc = int(proc.returncode)
    lines = [str(x) for x in (proc.stdout or '').splitlines() if str(x).strip()][-line_limit:]
    if lines:
        source = 'systemd_journal'

if not lines and log_path.exists():
    try:
        lines = list(deque(log_path.read_text(encoding='utf-8', errors='replace').splitlines(), maxlen=line_limit))
        if lines:
            source = 'log_file'
    except Exception:
        lines = []

out = {
    'action': 'live-risk-daemon-journal',
    'source': source,
    'service_installed': bool(unit_path.exists()),
    'unit_name': unit_name,
    'unit_path': str(unit_path),
    'line_limit': int(line_limit),
    'journal_returncode': journal_rc,
    'log_path': str(log_path),
    'systemd': {
        'active_state': show_map.get('ActiveState', 'missing'),
        'sub_state': show_map.get('SubState', ''),
        'main_pid': int(show_map.get('MainPID', '0') or 0),
    },
    'payload': payload,
    'lines': lines,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

run_live_ops_reconcile_status_remote() {
  local date_value
  date_value="${live_takeover_date}"
  run_live_takeover_remote \
    "live-ops-reconcile-status" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; python3 - '${date_value}' '${live_ops_reconcile_max_age_hours}' <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

requested_date = str(sys.argv[1]).strip()
max_age_hours = max(1, int(sys.argv[2]))
review_dir = Path('output/review')

target = review_dir / f'{requested_date}_ops_report.json' if requested_date else None
latest = None
if target is not None and target.exists():
    latest = target
else:
    files = []
    if review_dir.exists():
        files = sorted(
            review_dir.glob('*_ops_report.json'),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        )
    latest = files[-1] if files else None

out = {
    'action': 'live-ops-reconcile-status',
    'ok': False,
    'status': 'missing',
    'reason_code': 'ops_reconcile_artifact_missing',
    'requested_date': requested_date,
    'max_age_hours': int(max_age_hours),
    'artifact_path': '',
    'artifact_date': '',
    'artifact_age_hours': None,
    'artifact_mtime_utc': '',
    'gate_passed': None,
    'ops_status': '',
    'gate_failed_checks': [],
    'live_gate': {},
    'state_stability_live': {},
    'reconcile_active': False,
    'reconcile_ok': False,
    'samples': 0,
    'min_samples': 0,
    'alerts': [],
    'checks': {},
    'metrics': {},
    'artifacts': {},
}

if latest is None:
    print(json.dumps(out, ensure_ascii=False, indent=2))
    raise SystemExit(0)

payload = {}
try:
    loaded = json.loads(latest.read_text(encoding='utf-8'))
    payload = loaded if isinstance(loaded, dict) else {}
except Exception as exc:
    out['status'] = 'invalid'
    out['reason_code'] = f'ops_reconcile_artifact_invalid:{exc}'
    out['artifact_path'] = str(latest)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    raise SystemExit(0)

mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
age_hours = max(0.0, (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0)
reconcile = payload.get('reconcile_drift', {}) if isinstance(payload.get('reconcile_drift', {}), dict) else {}
checks = reconcile.get('checks', {}) if isinstance(reconcile.get('checks', {}), dict) else {}
metrics = reconcile.get('metrics', {}) if isinstance(reconcile.get('metrics', {}), dict) else {}
artifacts = reconcile.get('artifacts', {}) if isinstance(reconcile.get('artifacts', {}), dict) else {}
alerts = [str(x) for x in (reconcile.get('alerts', []) if isinstance(reconcile.get('alerts', []), list) else [])[:8]]
active = bool(reconcile.get('active', False))
reconcile_ok = all(bool(v) for v in checks.values()) if active else True
fresh = bool(age_hours <= float(max_age_hours))

out.update(
    {
        'artifact_path': str(latest),
        'artifact_date': str(payload.get('date', '')),
        'artifact_age_hours': float(age_hours),
        'artifact_mtime_utc': mtime.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'gate_passed': payload.get('gate_passed'),
        'ops_status': str(payload.get('status', '')),
        'gate_failed_checks': [
            str(x)
            for x in (payload.get('gate_failed_checks', []) if isinstance(payload.get('gate_failed_checks', []), list) else [])
            if str(x).strip()
        ],
        'live_gate': payload.get('live_gate', {}) if isinstance(payload.get('live_gate', {}), dict) else {},
        'state_stability_live': (
            payload.get('state_stability_live', {})
            if isinstance(payload.get('state_stability_live', {}), dict)
            else {}
        ),
        'reconcile_active': bool(active),
        'reconcile_ok': bool(reconcile_ok),
        'samples': int(reconcile.get('samples', 0) or 0),
        'min_samples': int(reconcile.get('min_samples', 0) or 0),
        'alerts': alerts,
        'checks': checks,
        'metrics': {
            'missing_ratio': float(metrics.get('missing_ratio', 0.0) or 0.0),
            'plan_gap_breach_ratio': float(metrics.get('plan_gap_breach_ratio', 0.0) or 0.0),
            'closed_count_gap_breach_ratio': float(metrics.get('closed_count_gap_breach_ratio', 0.0) or 0.0),
            'closed_pnl_gap_breach_ratio': float(metrics.get('closed_pnl_gap_breach_ratio', 0.0) or 0.0),
            'open_gap_breach_ratio': float(metrics.get('open_gap_breach_ratio', 0.0) or 0.0),
            'broker_missing_ratio': float(metrics.get('broker_missing_ratio', 0.0) or 0.0),
            'broker_count_breach_ratio': float(metrics.get('broker_count_breach_ratio', 0.0) or 0.0),
            'broker_pnl_breach_ratio': float(metrics.get('broker_pnl_breach_ratio', 0.0) or 0.0),
            'broker_row_diff_breach_ratio': float(metrics.get('broker_row_diff_breach_ratio', 0.0) or 0.0),
            'broker_row_diff_alias_hit_rate': float(metrics.get('broker_row_diff_alias_hit_rate', 0.0) or 0.0),
            'broker_row_diff_unresolved_key_ratio': float(metrics.get('broker_row_diff_unresolved_key_ratio', 0.0) or 0.0),
        },
        'artifacts': artifacts,
    }
)

if not fresh:
    out['status'] = 'stale'
    out['reason_code'] = 'ops_reconcile_artifact_stale'
    out['ok'] = False
elif active and not reconcile_ok:
    out['status'] = 'blocked'
    out['reason_code'] = 'ops_reconcile_drift_blocked'
    out['ok'] = False
elif not active:
    out['status'] = 'inactive'
    out['reason_code'] = ''
    out['ok'] = True
else:
    out['status'] = 'passed'
    out['reason_code'] = ''
    out['ok'] = True

print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

run_live_ops_reconcile_refresh_remote() {
  local date_value start_cmd start_json pid deadline poll_cmd poll_json
  date_value="${live_takeover_date:-$(date -u +%F)}"
  start_cmd="$(cat <<EOF
set -e
wd=\$(${remote_workdir_expr})
cd "\$wd"
python3 - '${date_value}' '${live_ops_reconcile_window_days}' <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

date_value = str(sys.argv[1]).strip()
window_days = str(sys.argv[2]).strip()
wd = Path.cwd()
stdout_log = wd / 'output' / 'logs' / f'live_ops_reconcile_refresh_{date_value}.stdout.log'
stderr_log = wd / 'output' / 'logs' / f'live_ops_reconcile_refresh_{date_value}.stderr.log'
target = wd / 'output' / 'review' / f'{date_value}_ops_report.json'
stdout_log.parent.mkdir(parents=True, exist_ok=True)
target.parent.mkdir(parents=True, exist_ok=True)
preexisting_exists = target.exists()
preexisting_mtime_ns = int(target.stat().st_mtime_ns) if preexisting_exists else 0

pid = 0
ps_cmd = '/bin/ps' if Path('/bin/ps').exists() else '/usr/bin/ps'
proc = subprocess.run([ps_cmd, '-eo', 'pid=,args='], text=True, capture_output=True, check=False)
needle = f'python3 -m lie_engine.cli --config config.yaml ops-report --date {date_value} --window-days {window_days}'
for raw in (proc.stdout or '').splitlines():
    row = str(raw).strip()
    if not row:
        continue
    parts = row.split(None, 1)
    if len(parts) != 2:
        continue
    pid_raw, args = parts
    if needle not in args or 'grep' in args:
        continue
    try:
        pid = int(pid_raw)
    except Exception:
        pid = 0
    if pid > 0:
        break

started = False
if pid <= 0:
    env = dict(os.environ)
    env['PYTHONPATH'] = 'src'
    out_fh = stdout_log.open('w', encoding='utf-8')
    err_fh = stderr_log.open('w', encoding='utf-8')
    try:
        child = subprocess.Popen(
            ['python3', '-m', 'lie_engine.cli', '--config', 'config.yaml', 'ops-report', '--date', date_value, '--window-days', window_days],
            cwd=str(wd),
            stdout=out_fh,
            stderr=err_fh,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        pid = int(child.pid)
        started = True
    finally:
        out_fh.close()
        err_fh.close()

out = {
    'action': 'live-ops-reconcile-refresh',
    'status': 'started' if started else 'running',
    'pid': int(pid),
    'started': bool(started),
    'date': date_value,
    'window_days': int(window_days),
    'artifact_path': str(target),
    'artifact_existed_before': bool(preexisting_exists),
    'artifact_mtime_ns_before': int(preexisting_mtime_ns),
    'stdout_log': str(stdout_log),
    'stderr_log': str(stderr_log),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
EOF
)"
  start_json="$(ssh_exec "${start_cmd}")"
  pid="$(python3 - "${start_json}" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
print(int(data.get('pid', 0) or 0))
PY
)"
  if ! [[ "${pid}" =~ ^[0-9]+$ ]] || (( pid <= 0 )); then
    echo "${start_json}"
    return 2
  fi
  deadline=$(( $(date +%s) + 180 ))
  while (( $(date +%s) <= deadline )); do
    poll_cmd="$(cat <<EOF
set -e
wd=\$(${remote_workdir_expr})
cd "\$wd"
python3 - '${pid}' '${date_value}' '${start_json}' <<'PY'
import json
import os
import sys
from collections import deque
from pathlib import Path

pid = int(sys.argv[1])
date_value = str(sys.argv[2]).strip()
start_payload = json.loads(sys.argv[3]) if len(sys.argv) > 3 and str(sys.argv[3]).strip() else {}
target = Path('output/review') / f'{date_value}_ops_report.json'
stdout_log = Path('output/logs') / f'live_ops_reconcile_refresh_{date_value}.stdout.log'
stderr_log = Path('output/logs') / f'live_ops_reconcile_refresh_{date_value}.stderr.log'

alive = False
if pid > 0:
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False

mtime_ns_before = int(start_payload.get('artifact_mtime_ns_before', 0) or 0)
target_mtime_ns = int(target.stat().st_mtime_ns) if target.exists() else 0
artifact_updated = bool(target.exists() and target_mtime_ns > mtime_ns_before)

def tail_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return list(deque(path.read_text(encoding='utf-8', errors='replace').splitlines(), maxlen=20))
    except Exception:
        return []

out = {
    'action': 'live-ops-reconcile-refresh',
    'pid': int(pid),
    'alive': bool(alive),
    'finished': not bool(alive),
    'artifact_exists': bool(target.exists()),
    'artifact_updated': bool(artifact_updated),
    'artifact_mtime_ns': int(target_mtime_ns),
    'artifact_mtime_ns_before': int(mtime_ns_before),
    'artifact_path': str(target),
    'stdout_log': str(stdout_log),
    'stderr_log': str(stderr_log),
    'stdout_tail': tail_lines(stdout_log),
    'stderr_tail': tail_lines(stderr_log),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
EOF
)"
    poll_json="$(ssh_exec "${poll_cmd}")"
    if python3 - "${poll_json}" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
raise SystemExit(0 if bool(data.get('finished', False)) else 1)
PY
    then
      if python3 - "${poll_json}" <<'PY'
import json
import sys
data = json.loads(sys.argv[1])
raise SystemExit(0 if bool(data.get('artifact_updated', False)) else 1)
PY
      then
        run_live_ops_reconcile_status_remote
        return 0
      fi
      echo "${poll_json}"
      return 2
    fi
    sleep 5
  done
  python3 - "${start_json}" "${poll_json:-{}}" <<'PY'
import json
import sys

start = json.loads(sys.argv[1])
poll = json.loads(sys.argv[2]) if len(sys.argv) > 2 and str(sys.argv[2]).strip() else {}
out = {
    'action': 'live-ops-reconcile-refresh',
    'status': 'running',
    'timeout': True,
    'start': start,
    'poll': poll,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
  return 0
}

action_bootstrap_remote_runtime() {
  local remote_cmd
  remote_cmd="$(cat <<EOF
set -e
wd=\$(${remote_workdir_expr})
cd "\$wd"
export PIP_DISABLE_PIP_VERSION_CHECK=1
if ! python3 -m pip --version >/dev/null 2>&1; then
  python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
mode='user'
upgrade_ok=1
install_ok=1
upgrade_log=\$(mktemp)
install_log=\$(mktemp)
set +e
python3 -m pip install --user --upgrade pip setuptools wheel >"\${upgrade_log}" 2>&1
rc_upgrade=\$?
set -e
if (( rc_upgrade != 0 )) && grep -qi 'externally-managed-environment' "\${upgrade_log}"; then
  mode='break_system_packages'
  set +e
  python3 -m pip install --break-system-packages --user --upgrade pip setuptools wheel >"\${upgrade_log}" 2>&1
  rc_upgrade=\$?
  set -e
fi
if (( rc_upgrade != 0 )); then
  upgrade_ok=0
fi
set +e
if [[ "\${mode}" = 'break_system_packages' ]]; then
  python3 -m pip install --break-system-packages --user -e . >"\${install_log}" 2>&1
else
  python3 -m pip install --user -e . >"\${install_log}" 2>&1
fi
rc_install=\$?
set -e
if (( rc_install != 0 )) && [[ "\${mode}" = 'user' ]] && grep -qi 'externally-managed-environment' "\${install_log}"; then
  mode='break_system_packages'
  set +e
  python3 -m pip install --break-system-packages --user -e . >"\${install_log}" 2>&1
  rc_install=\$?
  set -e
fi
if (( rc_install != 0 )); then
  install_ok=0
fi
python3 - "\${mode}" "\${upgrade_ok}" "\${install_ok}" "\${upgrade_log}" "\${install_log}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

mode = str(sys.argv[1]).strip() or 'user'
upgrade_ok = bool(int(sys.argv[2]))
install_ok = bool(int(sys.argv[3]))
upgrade_log_path = Path(sys.argv[4])
install_log_path = Path(sys.argv[5])
mods = {}
for name in ('numpy', 'pandas', 'yaml', 'akshare', 'yfinance', 'tqdm'):
    try:
        mod = __import__(name)
        mods[name] = getattr(mod, '__version__', 'ok')
    except Exception as exc:
        mods[name] = f'missing:{exc.__class__.__name__}'
out = {
    'action': 'bootstrap-remote-runtime',
    'mode': mode,
    'upgrade_ok': bool(upgrade_ok),
    'install_ok': bool(install_ok),
    'python': sys.executable,
    'python_version': sys.version.split()[0],
    'pip_version': '',
    'modules': mods,
    'upgrade_log_tail': [],
    'install_log_tail': [],
}
try:
    pip_proc = subprocess.run([sys.executable, '-m', 'pip', '--version'], text=True, capture_output=True, check=False)
    out['pip_version'] = str((pip_proc.stdout or '').strip())
except Exception:
    out['pip_version'] = ''
for key, path in (('upgrade_log_tail', upgrade_log_path), ('install_log_tail', install_log_path)):
    try:
        rows = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        rows = []
    out[key] = rows[-20:]
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
rc_final=0
if (( install_ok == 0 )); then
  rc_final=2
elif (( upgrade_ok == 0 )); then
  rc_final=2
fi
rm -f "\${upgrade_log}" "\${install_log}" >/dev/null 2>&1 || true
exit "\${rc_final}"
EOF
)"
  ssh_exec "${remote_cmd}"
}

action_ensure_local_openclaw_runtime_model() {
  python3 "${system_root}/scripts/ensure_openclaw_runtime_model.py" \
    --config "${HOME}/.openclaw/openclaw.json"
}

action_sync_local_pi_workspace() {
  local -a cmd
  cmd=(
    python3
    "${system_root}/scripts/sync_local_pi_workspace.py"
    --source-root "${system_root}"
    --target-root "${local_pi_workspace_system_root}"
    --backup-keep "${local_pi_workspace_backup_keep}"
    --backup-max-age-hours "${local_pi_workspace_backup_max_age_hours}"
    --pulse-lock-path "${local_pi_workspace_system_root}/output/state/run_halfhour_pulse.lock"
  )
  if is_true "${local_pi_workspace_dry_run}"; then
    cmd+=(--dry-run)
  fi
  if is_true "${local_pi_workspace_no_backup}"; then
    cmd+=(--no-backup)
  fi
  "${cmd[@]}"
}

action_publish_local_pi_runtime_scripts() {
  local -a cmd
  cmd=(
    python3
    "${system_root}/scripts/publish_local_pi_runtime_scripts.py"
    --source-root "${local_pi_runtime_scripts_source_root}"
    --target-root "${local_pi_runtime_scripts_target_root}"
    --output-root "${local_pi_workspace_system_root}/output"
    --backup-keep "${local_pi_runtime_scripts_backup_keep}"
    --backup-max-age-hours "${local_pi_runtime_scripts_backup_max_age_hours}"
    --pulse-lock-path "${local_pi_workspace_system_root}/output/state/run_halfhour_pulse.lock"
  )
  if is_true "${local_pi_runtime_scripts_dry_run}"; then
    cmd+=(--dry-run)
  fi
  if is_true "${local_pi_runtime_scripts_no_backup}"; then
    cmd+=(--no-backup)
  fi
  "${cmd[@]}"
}

action_prepare_local_pi_runtime() {
  local tmp_sync
  local tmp_publish
  local tmp_model
  local tmp_gate_log
  tmp_sync="$(mktemp)"
  tmp_publish="$(mktemp)"
  tmp_model="$(mktemp)"
  tmp_gate_log="$(mktemp)"

  action_sync_local_pi_workspace >"${tmp_sync}"
  action_publish_local_pi_runtime_scripts >"${tmp_publish}"
  action_ensure_local_openclaw_runtime_model >"${tmp_model}"

  if [[ ! -x "${local_pi_launchd_runner_path}" ]]; then
    python3 - "${tmp_sync}" "${tmp_publish}" "${tmp_model}" "${local_pi_launchd_runner_path}" <<'PY'
import json
import sys
from pathlib import Path

sync = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
publish = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
model = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
runner = str(sys.argv[4]).strip()
out = {
    'action': 'prepare_local_pi_runtime',
    'ok': False,
    'status': 'runner_missing',
    'sync': sync,
    'runtime_scripts': publish,
    'runtime_model': model,
    'gate_smoke': {
        'ok': False,
        'runner_path': runner,
        'returncode': 3,
        'log_tail': [],
    },
    'error': f'runner_missing:{runner}',
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    rm -f "${tmp_sync}" "${tmp_publish}" "${tmp_model}" "${tmp_gate_log}" >/dev/null 2>&1 || true
    return 3
  fi

  set +e
  PI_LAUNCHD_GATE_ONLY=true \
  WHITELIST_GATE_WINDOW_HOURS="${local_pi_gate_window_hours}" \
  FENLIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  LIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  "${local_pi_launchd_runner_path}" >"${tmp_gate_log}" 2>&1
  local gate_rc=$?
  set -e

  python3 - "${tmp_sync}" "${tmp_publish}" "${tmp_model}" "${tmp_gate_log}" "${local_pi_launchd_runner_path}" "${gate_rc}" <<'PY'
import json
import sys
from pathlib import Path

sync = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
publish = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
model = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
gate_log = Path(sys.argv[4])
runner = str(sys.argv[5]).strip()
gate_rc = int(sys.argv[6])
lines = []
if gate_log.exists():
    try:
        lines = gate_log.read_text(encoding='utf-8', errors='replace').splitlines()[-20:]
    except Exception:
        lines = []
ok = bool(sync.get('ok')) and bool(publish.get('ok')) and bool(model.get('ok')) and gate_rc == 0
status = 'ok' if ok else 'gate_failed'
out = {
    'action': 'prepare_local_pi_runtime',
    'ok': ok,
    'status': status,
    'sync': sync,
    'runtime_scripts': publish,
    'runtime_model': model,
    'gate_smoke': {
        'ok': gate_rc == 0,
        'runner_path': runner,
        'returncode': gate_rc,
        'log_tail': lines,
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
  local rc_out=$?
  rm -f "${tmp_sync}" "${tmp_publish}" "${tmp_model}" "${tmp_gate_log}" >/dev/null 2>&1 || true
  if (( rc_out != 0 )); then
    return "${rc_out}"
  fi
  return "${gate_rc}"
}

action_smoke_local_pi_cycle() {
  local tmp_prepare
  local tmp_full_log
  local before_latest
  local after_latest
  local rc_prepare
  local rc_full

  tmp_prepare="$(mktemp)"
  tmp_full_log="$(mktemp)"
  before_latest="$(ls -1t "${local_pi_workspace_system_root}/output/review"/*_pi_launchd_auto_retro.json 2>/dev/null | head -n 1 || true)"
  rc_prepare=0

  if is_true "${local_pi_prepare_before_full_smoke}"; then
    set +e
    action_prepare_local_pi_runtime >"${tmp_prepare}"
    rc_prepare=$?
    set -e
  else
    printf '%s\n' '{"action":"prepare_local_pi_runtime","ok":true,"status":"skipped","skipped":true}' >"${tmp_prepare}"
  fi

  if (( rc_prepare != 0 )); then
    python3 - "${tmp_prepare}" "${local_pi_launchd_runner_path}" "${rc_prepare}" <<'PY'
import json
import sys
from pathlib import Path

prepare = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
runner = str(sys.argv[2]).strip()
rc_prepare = int(sys.argv[3])
out = {
    "action": "smoke_local_pi_cycle",
    "ok": False,
    "status": "prepare_failed",
    "prepare": prepare,
    "full_cycle": {
        "ok": False,
        "runner_path": runner,
        "returncode": rc_prepare,
        "latest_envelope": "",
        "log_tail": [],
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    rm -f "${tmp_prepare}" "${tmp_full_log}" >/dev/null 2>&1 || true
    return "${rc_prepare}"
  fi

  if [[ ! -x "${local_pi_launchd_runner_path}" ]]; then
    python3 - "${tmp_prepare}" "${local_pi_launchd_runner_path}" <<'PY'
import json
import sys
from pathlib import Path

prepare = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
runner = str(sys.argv[2]).strip()
out = {
    "action": "smoke_local_pi_cycle",
    "ok": False,
    "status": "runner_missing",
    "prepare": prepare,
    "full_cycle": {
        "ok": False,
        "runner_path": runner,
        "returncode": 3,
        "latest_envelope": "",
        "log_tail": [],
    },
    "error": f"runner_missing:{runner}",
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    rm -f "${tmp_prepare}" "${tmp_full_log}" >/dev/null 2>&1 || true
    return 3
  fi

  set +e
  FENLIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  LIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  "${local_pi_launchd_runner_path}" >"${tmp_full_log}" 2>&1
  rc_full=$?
  set -e
  after_latest="$(ls -1t "${local_pi_workspace_system_root}/output/review"/*_pi_launchd_auto_retro.json 2>/dev/null | head -n 1 || true)"

  python3 - "${tmp_prepare}" "${tmp_full_log}" "${local_pi_launchd_log_path}" "${local_pi_launchd_runner_path}" "${before_latest}" "${after_latest}" "${rc_full}" <<'PY'
import json
import sys
from pathlib import Path

prepare = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
tmp_full_log = Path(sys.argv[2])
launchd_log = Path(sys.argv[3])
runner = str(sys.argv[4]).strip()
before_latest = str(sys.argv[5]).strip()
after_latest = str(sys.argv[6]).strip()
rc_full = int(sys.argv[7])

log_lines = []
for path in (launchd_log, tmp_full_log):
    if not path.exists():
        continue
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        lines = []
    if lines:
        log_lines = lines[-40:]
        break

latest_envelope = after_latest or before_latest
out = {
    "action": "smoke_local_pi_cycle",
    "ok": bool(prepare.get("ok")) and rc_full == 0,
    "status": "ok" if bool(prepare.get("ok")) and rc_full == 0 else "full_cycle_failed",
    "prepare": prepare,
    "full_cycle": {
        "ok": rc_full == 0,
        "runner_path": runner,
        "returncode": rc_full,
        "latest_envelope": latest_envelope,
        "log_tail": log_lines,
        "retro_artifact_rotated": bool(after_latest) and after_latest != before_latest,
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
  local rc_out=$?
  rm -f "${tmp_prepare}" "${tmp_full_log}" >/dev/null 2>&1 || true
  if (( rc_out != 0 )); then
    return "${rc_out}"
  fi
  return "${rc_full}"
}

action_run_local_pi_recovery_lab() {
  local -a cmd
  cmd=(
    python3
    "${system_root}/scripts/run_local_pi_recovery_lab.py"
    --workspace-system-root "${local_pi_workspace_system_root}"
    --lab-parent-dir "${local_pi_recovery_lab_parent_dir}"
    --artifact-ttl-hours "${local_pi_recovery_lab_ttl_hours}"
    --keep-labs "${local_pi_recovery_lab_keep}"
  )
  if is_true "${local_pi_recovery_lab_allow_fallback_write}"; then
    cmd+=(--allow-fallback-write)
  fi
  FENLIE_SYSTEM_ROOT="${system_root}" \
  LIE_SYSTEM_ROOT="${system_root}" \
  "${cmd[@]}"
}

action_snapshot_local_pi_recovery_state() {
  local -a cmd
  cmd=(
    python3
    "${system_root}/scripts/snapshot_local_pi_recovery_state.py"
    --workspace-system-root "${local_pi_workspace_system_root}"
    --checkpoint-dir "${local_pi_recovery_checkpoint_dir}"
    --checkpoint-keep "${local_pi_recovery_checkpoint_keep}"
    --checkpoint-max-age-hours "${local_pi_recovery_checkpoint_max_age_hours}"
    --pulse-lock-path "${local_pi_workspace_system_root}/output/state/run_halfhour_pulse.lock"
  )
  if [[ -n "${local_pi_recovery_checkpoint_note}" ]]; then
    cmd+=(--note "${local_pi_recovery_checkpoint_note}")
  fi
  FENLIE_SYSTEM_ROOT="${system_root}" \
  LIE_SYSTEM_ROOT="${system_root}" \
  "${cmd[@]}"
}

action_restore_local_pi_recovery_state() {
  local -a cmd
  if [[ -z "${local_pi_recovery_restore_checkpoint}" ]]; then
    python3 - <<'PY'
import json
print(json.dumps({
    "action": "restore_local_pi_recovery_state",
    "ok": False,
    "status": "checkpoint_missing",
    "reason": "LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT is required",
}, ensure_ascii=False, indent=2))
PY
    return 4
  fi
  cmd=(
    python3
    "${system_root}/scripts/restore_local_pi_recovery_state.py"
    --workspace-system-root "${local_pi_workspace_system_root}"
    --checkpoint "${local_pi_recovery_restore_checkpoint}"
    --pulse-lock-path "${local_pi_workspace_system_root}/output/state/run_halfhour_pulse.lock"
  )
  if [[ -n "${local_pi_recovery_restore_expected_state_fingerprint}" ]]; then
    cmd+=(--expected-current-state-fingerprint "${local_pi_recovery_restore_expected_state_fingerprint}")
  fi
  if is_true "${local_pi_recovery_restore_write}"; then
    cmd+=(--write)
  else
    cmd+=(--dry-run)
  fi
  FENLIE_SYSTEM_ROOT="${system_root}" \
  LIE_SYSTEM_ROOT="${system_root}" \
  "${cmd[@]}"
}

action_rollback_local_pi_recovery_state() {
  local checkpoint_path="${local_pi_recovery_restore_checkpoint}"
  if [[ -z "${checkpoint_path}" ]]; then
    checkpoint_path="$(python3 - "${local_pi_recovery_checkpoint_dir}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
dirs = [p for p in root.glob("*_local_pi_recovery_checkpoint") if p.is_dir()]
dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
latest = dirs[0] / "checkpoint.json" if dirs else None
print("" if latest is None else str(latest))
PY
)"
  fi
  if [[ -z "${checkpoint_path}" ]]; then
    python3 - <<'PY'
import json
print(json.dumps({
    "action": "rollback_local_pi_recovery_state",
    "ok": False,
    "status": "checkpoint_not_found",
    "reason": "no recovery checkpoint available",
}, ensure_ascii=False, indent=2))
PY
    return 4
  fi
  (
    local_pi_recovery_restore_checkpoint="${checkpoint_path}"
    action_restore_local_pi_recovery_state
  )
}

action_ack_local_pi_consecutive_loss_guardrail() {
  local script_path
  local ack_path
  local checksum_path
  local pulse_lock_path
  local expected_state_fingerprint
  local -a cmd

  script_path="${local_pi_workspace_system_root}/scripts/ack_paper_consecutive_loss_guardrail.py"
  ack_path="${local_pi_workspace_system_root}/output/state/paper_consecutive_loss_ack.json"
  checksum_path="${local_pi_workspace_system_root}/output/state/paper_consecutive_loss_ack_checksum.json"
  pulse_lock_path="${local_pi_workspace_system_root}/output/state/run_halfhour_pulse.lock"
  expected_state_fingerprint="${LOCAL_PI_EXPECTED_STATE_FINGERPRINT:-}"

  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "ack_local_pi_consecutive_loss_guardrail",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  cmd=(
    python3 "${script_path}"
    --state-path "${local_pi_workspace_system_root}/output/state/spot_paper_state.json"
    --ack-path "${ack_path}"
    --checksum-path "${checksum_path}"
    --pulse-lock-path "${pulse_lock_path}"
    --ttl-hours "${local_pi_consecutive_loss_ack_ttl_hours}"
    --cooldown-hours "${local_pi_consecutive_loss_ack_cooldown_hours}"
  )
  if [[ -n "${expected_state_fingerprint}" ]]; then
    cmd+=(--expected-state-fingerprint "${expected_state_fingerprint}")
  fi
  if is_true "${local_pi_consecutive_loss_ack_allow_missing_last_loss_ts}"; then
    cmd+=(--allow-missing-last-loss-ts)
  fi
  if [[ -n "${local_pi_consecutive_loss_ack_note}" ]]; then
    cmd+=(--note "${local_pi_consecutive_loss_ack_note}")
  fi
  if is_true "${local_pi_consecutive_loss_ack_write}"; then
    cmd+=(--write)
  fi

  FENLIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  LIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  "${cmd[@]}"
}

action_local_pi_consecutive_loss_guardrail_status() {
  local script_path
  local ack_path
  local checksum_path
  local -a cmd

  script_path="${local_pi_workspace_system_root}/scripts/paper_consecutive_loss_guardrail_status.py"
  ack_path="${local_pi_workspace_system_root}/output/state/paper_consecutive_loss_ack.json"
  checksum_path="${local_pi_workspace_system_root}/output/state/paper_consecutive_loss_ack_checksum.json"

  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "local_pi_consecutive_loss_guardrail_status",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  cmd=(
    python3 "${script_path}"
    --state-path "${local_pi_workspace_system_root}/output/state/spot_paper_state.json"
    --ack-path "${ack_path}"
    --checksum-path "${checksum_path}"
    --cooldown-hours "${local_pi_consecutive_loss_ack_cooldown_hours}"
    --stop-threshold "${local_pi_consecutive_loss_stop_threshold}"
  )
  if is_true "${local_pi_consecutive_loss_ack_allow_missing_last_loss_ts}"; then
    cmd+=(--allow-missing-last-loss-ts)
  fi

  FENLIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  LIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  "${cmd[@]}"
}

action_local_pi_ack_archive_status() {
  local script_path
  local ack_path
  local checksum_path
  local archive_dir
  local manifest_path
  local -a cmd

  script_path="${local_pi_workspace_system_root}/scripts/paper_consecutive_loss_ack_archive_status.py"
  ack_path="${local_pi_workspace_system_root}/output/state/paper_consecutive_loss_ack.json"
  checksum_path="${local_pi_workspace_system_root}/output/state/paper_consecutive_loss_ack_checksum.json"
  archive_dir="${local_pi_workspace_system_root}/output/review/paper_consecutive_loss_ack_archive"
  manifest_path="${archive_dir}/manifest.jsonl"

  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "local_pi_ack_archive_status",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  cmd=(
    python3 "${script_path}"
    --ack-path "${ack_path}"
    --checksum-path "${checksum_path}"
    --archive-dir "${archive_dir}"
    --archive-manifest-path "${manifest_path}"
    --manifest-tail "${local_pi_ack_archive_manifest_tail}"
  )

  FENLIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  LIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  "${cmd[@]}"
}

action_local_pi_recovery_handoff() {
  local script_path
  local review_dir
  local checkpoint_dir
  local -a cmd

  script_path="${local_pi_workspace_system_root}/scripts/build_local_pi_recovery_handoff.py"
  review_dir="${local_pi_workspace_system_root}/output/review"
  checkpoint_dir="${review_dir}/local_pi_recovery_checkpoints"

  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "local_pi_recovery_handoff",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  cmd=(
    python3 "${script_path}"
    --workspace-system-root "${local_pi_workspace_system_root}"
    --review-dir "${review_dir}"
    --checkpoint-dir "${checkpoint_dir}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${local_pi_recovery_handoff_keep}"
    --stop-threshold "${local_pi_consecutive_loss_stop_threshold}"
    --cooldown-hours "${local_pi_consecutive_loss_ack_cooldown_hours}"
  )

  FENLIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  LIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  "${cmd[@]}"
}

action_remote_live_handoff() {
  local script_path
  local tmp_ready tmp_daemon tmp_ops tmp_journal tmp_security
  local rc_ready rc_daemon rc_ops rc_journal rc_security
  local latest_noaf_probe
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_handoff.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "remote_live_handoff",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  tmp_ready="$(mktemp)"
  tmp_daemon="$(mktemp)"
  tmp_ops="$(mktemp)"
  tmp_journal="$(mktemp)"
  tmp_security="$(mktemp)"

  set +e
  action_live_takeover_ready_check >"${tmp_ready}" 2>/dev/null
  rc_ready=$?
  action_live_risk_daemon_status >"${tmp_daemon}" 2>/dev/null
  rc_daemon=$?
  action_live_ops_reconcile_status >"${tmp_ops}" 2>/dev/null
  rc_ops=$?
  action_live_risk_daemon_journal >"${tmp_journal}" 2>/dev/null
  rc_journal=$?
  action_live_risk_daemon_security_status >"${tmp_security}" 2>/dev/null
  rc_security=$?
  set -e
  latest_noaf_probe="$(ls -1t "${output_dir}"/*_remote_live_noaf_probe.json 2>/dev/null | head -n 1 || true)"

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --ready-check-file "${tmp_ready}"
    --ready-check-returncode "${rc_ready}"
    --risk-daemon-status-file "${tmp_daemon}"
    --risk-daemon-status-returncode "${rc_daemon}"
    --ops-status-file "${tmp_ops}"
    --ops-status-returncode "${rc_ops}"
    --journal-file "${tmp_journal}"
    --journal-returncode "${rc_journal}"
    --security-status-file "${tmp_security}"
    --security-status-returncode "${rc_security}"
    --noaf-probe-file "${latest_noaf_probe}"
    --security-accept-max-exposure "${live_risk_daemon_security_accept_max_exposure}"
    --remote-host "${cloud_host}"
    --remote-user "${cloud_user}"
    --remote-project-dir "${cloud_project_dir}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_ready}" "${tmp_daemon}" "${tmp_ops}" "${tmp_journal}" "${tmp_security}" >/dev/null 2>&1 || true
}

action_remote_live_notification_preview() {
  local script_path
  local tmp_handoff
  local rc_handoff
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_notification_preview.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "remote_live_notification_preview",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  tmp_handoff="$(mktemp)"
  set +e
  action_remote_live_handoff >"${tmp_handoff}" 2>/dev/null
  rc_handoff=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --handoff-file "${tmp_handoff}"
    --handoff-returncode "${rc_handoff}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_handoff}" >/dev/null 2>&1 || true
}

action_remote_live_notification_dry_run() {
  local script_path
  local tmp_preview
  local rc_preview
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_notification_dry_run.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "remote_live_notification_dry_run",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  tmp_preview="$(mktemp)"
  set +e
  action_remote_live_notification_preview >"${tmp_preview}" 2>/dev/null
  rc_preview=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --preview-file "${tmp_preview}"
    --preview-returncode "${rc_preview}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_preview}" >/dev/null 2>&1 || true
}

action_remote_live_notification_send() {
  local script_path
  local tmp_dry_run
  local rc_dry_run
  local -a cmd

  script_path="${system_root}/scripts/send_remote_live_notification.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "remote_live_notification_send",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  tmp_dry_run="$(mktemp)"
  set +e
  action_remote_live_notification_dry_run >"${tmp_dry_run}" 2>/dev/null
  rc_dry_run=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --state-dir "${state_dir}"
    --dry-run-file "${tmp_dry_run}"
    --dry-run-returncode "${rc_dry_run}"
    --delivery "${remote_live_notification_delivery}"
    --timeout-ms "${remote_live_notification_timeout_ms}"
    --rate-limit-per-minute "${remote_live_notification_rate_limit_per_minute}"
    --idempotency-ttl-seconds "${remote_live_notification_idempotency_ttl_seconds}"
    --idempotency-max-entries "${remote_live_notification_idempotency_max_entries}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_dry_run}" >/dev/null 2>&1 || true
}

action_apply_local_pi_recovery_step() {
  local tmp_status
  local tmp_step
  local tmp_snapshot
  local tmp_post
  local next_action
  local status_state_fingerprint
  local snapshot_rc=0
  local snapshot_attempted=false
  local snapshot_required=false
  local step_rc=0
  local step_attempted=false
  local execution_mode="preview"
  tmp_status="$(mktemp)"
  tmp_step="$(mktemp)"
  tmp_snapshot="$(mktemp)"
  tmp_post="$(mktemp)"

  snapshot_recovery_step_if_needed() {
    local snapshot_reason="$1"
    local snapshot_note
    snapshot_attempted=true
    snapshot_note="${local_pi_recovery_auto_snapshot_note:-recovery_step:${snapshot_reason}}"
    set +e
    (
      local_pi_recovery_checkpoint_note="${snapshot_note}"
      action_snapshot_local_pi_recovery_state
    ) >"${tmp_snapshot}"
    snapshot_rc=$?
    set -e
    return "${snapshot_rc}"
  }

  action_local_pi_consecutive_loss_guardrail_status >"${tmp_status}"
  next_action="$(python3 - "${tmp_status}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(str(payload.get('recovery_plan', {}).get('next_action') or '').strip())
PY
)"
  status_state_fingerprint="$(python3 - "${tmp_status}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(str(payload.get('state_fingerprint') or '').strip())
PY
)"

  case "${next_action}" in
    write_manual_ack)
      if is_true "${local_pi_recovery_apply_write}"; then
        execution_mode="ack_write"
        snapshot_required=true
        if ! is_true "${local_pi_recovery_auto_snapshot_before_write}" || snapshot_recovery_step_if_needed "write_manual_ack"; then
          step_attempted=true
          set +e
          (
            LOCAL_PI_EXPECTED_STATE_FINGERPRINT="${status_state_fingerprint}"
            local_pi_consecutive_loss_ack_write="true"
            action_ack_local_pi_consecutive_loss_guardrail
          ) >"${tmp_step}"
          step_rc=$?
          set -e
        else
          execution_mode="snapshot_failed"
        fi
      else
        step_attempted=true
        execution_mode="ack_preview"
        (
          LOCAL_PI_EXPECTED_STATE_FINGERPRINT="${status_state_fingerprint}"
          local_pi_consecutive_loss_ack_write="false"
          action_ack_local_pi_consecutive_loss_guardrail
        ) >"${tmp_step}"
      fi
      ;;
    write_strict_last_loss_ts_backfill)
      if is_true "${local_pi_recovery_apply_write}"; then
        execution_mode="strict_backfill_write"
        snapshot_required=true
        if ! is_true "${local_pi_recovery_auto_snapshot_before_write}" || snapshot_recovery_step_if_needed "write_strict_last_loss_ts_backfill"; then
          step_attempted=true
          set +e
          (
            LOCAL_PI_EXPECTED_STATE_FINGERPRINT="${status_state_fingerprint}"
            local_pi_last_loss_ts_backfill_write="true"
            action_backfill_local_pi_last_loss_ts
          ) >"${tmp_step}"
          step_rc=$?
          set -e
        else
          execution_mode="snapshot_failed"
        fi
      else
        step_attempted=true
        execution_mode="strict_backfill_preview"
        (
          LOCAL_PI_EXPECTED_STATE_FINGERPRINT="${status_state_fingerprint}"
          local_pi_last_loss_ts_backfill_write="false"
          action_backfill_local_pi_last_loss_ts
        ) >"${tmp_step}"
      fi
      ;;
    review_fallback_last_loss_ts_backfill)
      if is_true "${local_pi_recovery_apply_write}" && is_true "${local_pi_recovery_allow_fallback_write}"; then
        execution_mode="fallback_backfill_write"
        snapshot_required=true
        if ! is_true "${local_pi_recovery_auto_snapshot_before_write}" || snapshot_recovery_step_if_needed "review_fallback_last_loss_ts_backfill"; then
          step_attempted=true
          set +e
          (
            LOCAL_PI_EXPECTED_STATE_FINGERPRINT="${status_state_fingerprint}"
            local_pi_last_loss_ts_backfill_allow_latest_loss_fallback="true"
            local_pi_last_loss_ts_backfill_write="true"
            action_backfill_local_pi_last_loss_ts
          ) >"${tmp_step}"
          step_rc=$?
          set -e
        else
          execution_mode="snapshot_failed"
        fi
      else
        step_attempted=true
        execution_mode="fallback_backfill_preview"
        (
          LOCAL_PI_EXPECTED_STATE_FINGERPRINT="${status_state_fingerprint}"
          local_pi_last_loss_ts_backfill_allow_latest_loss_fallback="true"
          local_pi_last_loss_ts_backfill_write="false"
          action_backfill_local_pi_last_loss_ts
        ) >"${tmp_step}"
      fi
      ;;
    run_full_cycle_with_existing_ack)
      if is_true "${local_pi_recovery_run_full_cycle}"; then
        step_attempted=true
        execution_mode="full_cycle"
        set +e
        action_smoke_local_pi_cycle >"${tmp_step}"
        step_rc=$?
        set -e
      fi
      ;;
  esac

  if is_true "${local_pi_recovery_verify_after_step}"; then
    action_local_pi_consecutive_loss_guardrail_status >"${tmp_post}"
  fi

  python3 - "${tmp_status}" "${tmp_step}" "${tmp_snapshot}" "${step_attempted}" "${execution_mode}" "${snapshot_required}" \
    "${snapshot_attempted}" "${snapshot_rc}" "${step_rc}" \
    "${local_pi_recovery_apply_write}" "${local_pi_recovery_allow_fallback_write}" "${local_pi_recovery_run_full_cycle}" \
    "${local_pi_recovery_verify_after_step}" "${tmp_post}" "${system_root}" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
step_path = Path(sys.argv[2])
snapshot_path = Path(sys.argv[3])
step_attempted = sys.argv[4].strip().lower() == 'true'
execution_mode = sys.argv[5].strip()
snapshot_required = sys.argv[6].strip().lower() == 'true'
snapshot_attempted = sys.argv[7].strip().lower() == 'true'
snapshot_rc = int(sys.argv[8])
step_rc = int(sys.argv[9])
write_requested = sys.argv[10].strip().lower() in {'1','true','yes','on'}
fallback_write_allowed = sys.argv[11].strip().lower() in {'1','true','yes','on'}
full_cycle_requested = sys.argv[12].strip().lower() in {'1','true','yes','on'}
verify_after_step = sys.argv[13].strip().lower() in {'1','true','yes','on'}
post_path = Path(sys.argv[14])
system_root = Path(sys.argv[15]).expanduser().resolve()
snapshot_result = None
if snapshot_attempted and snapshot_path.exists() and snapshot_path.stat().st_size:
    snapshot_result = json.loads(snapshot_path.read_text(encoding='utf-8'))
step_result = None
if step_attempted and step_path.exists() and step_path.stat().st_size:
    step_result = json.loads(step_path.read_text(encoding='utf-8'))
post_status = None
if verify_after_step and post_path.exists() and post_path.stat().st_size:
    post_status = json.loads(post_path.read_text(encoding='utf-8'))
pre_state_fingerprint = str(status.get('state_fingerprint') or '').strip() or None
post_state_fingerprint = (
    None
    if not isinstance(post_status, dict)
    else (str(post_status.get('state_fingerprint') or '').strip() or None)
)
pre_ack = status.get('ack') if isinstance(status.get('ack'), dict) else {}
post_ack = post_status.get('ack') if isinstance(post_status, dict) and isinstance(post_status.get('ack'), dict) else {}
pre_ack_present = bool(pre_ack.get('present')) if isinstance(pre_ack, dict) else False
post_ack_present = bool(post_ack.get('present')) if isinstance(post_ack, dict) else False
mutation_detected_reasons = []
if isinstance(step_result, dict) and bool(step_result.get('write_performed')):
    mutation_detected_reasons.append('step_result_write_performed')
if execution_mode in {'strict_backfill_write', 'fallback_backfill_write'} and pre_state_fingerprint and post_state_fingerprint and pre_state_fingerprint != post_state_fingerprint:
    mutation_detected_reasons.append('state_fingerprint_changed')
if execution_mode == 'ack_write' and pre_ack_present != post_ack_present:
    mutation_detected_reasons.append('ack_presence_changed')
mutation_detected = bool(mutation_detected_reasons)
rollback_guidance = None
if isinstance(snapshot_result, dict) and bool(snapshot_result.get('ok')):
    checkpoint_manifest = str(snapshot_result.get('checkpoint_manifest') or '').strip()
    if checkpoint_manifest:
        rollback_guidance = {
            'checkpoint_manifest': checkpoint_manifest,
            'dry_run_command': (
                f'cd {system_root} && '
                f'LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT="{checkpoint_manifest}" '
                'scripts/openclaw_cloud_bridge.sh rollback-local-pi-recovery-state'
            ),
            'write_command': (
                f'cd {system_root} && '
                'LOCAL_PI_RECOVERY_RESTORE_WRITE=true '
                f'LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT="{checkpoint_manifest}" '
                'scripts/openclaw_cloud_bridge.sh rollback-local-pi-recovery-state'
            ),
        }
operator_note = None
if rollback_guidance is not None:
    operator_note = {
        'summary': 'Recovery write step has a checkpoint-backed rollback path.',
        'when_to_use': 'Use rollback if post-write verification fails or you want to revert the most recent recovery mutation.',
        'recommended_first_action': rollback_guidance['dry_run_command'],
        'escalation_action': rollback_guidance['write_command'],
        'checkpoint_manifest': rollback_guidance['checkpoint_manifest'],
    }
ok = bool(status.get('ok')) and (not snapshot_attempted or snapshot_rc == 0) and (not step_attempted or step_rc == 0)
status_text = 'ok' if ok else ('snapshot_failed' if snapshot_attempted and snapshot_rc != 0 else 'step_failed')
out = {
    'action': 'apply_local_pi_recovery_step',
    'ok': ok,
    'status': status_text,
    'next_action': status.get('recovery_plan', {}).get('next_action'),
    'action_level': status.get('recovery_plan', {}).get('action_level'),
    'write_projection': status.get('write_projection'),
    'write_requested': write_requested,
    'fallback_write_allowed': fallback_write_allowed,
    'full_cycle_requested': full_cycle_requested,
    'verify_after_step': verify_after_step,
    'snapshot_required': snapshot_required,
    'snapshot_attempted': snapshot_attempted,
    'snapshot_result': snapshot_result,
    'rollback_guidance': rollback_guidance,
    'operator_note': operator_note,
    'step_attempted': step_attempted,
    'execution_mode': execution_mode,
    'mutation_detected': mutation_detected,
    'mutation_detected_reasons': mutation_detected_reasons,
    'status_payload': status,
    'step_result': step_result,
    'post_status_payload': post_status,
    'post_next_action': None if not isinstance(post_status, dict) else post_status.get('recovery_plan', {}).get('next_action'),
    'post_action_level': None if not isinstance(post_status, dict) else post_status.get('recovery_plan', {}).get('action_level'),
    'post_write_projection': None if not isinstance(post_status, dict) else post_status.get('write_projection'),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
  local rc_out=$?
  rm -f "${tmp_status}" "${tmp_step}" "${tmp_snapshot}" "${tmp_post}" >/dev/null 2>&1 || true
  if (( rc_out != 0 )); then
    return "${rc_out}"
  fi
  if is_true "${snapshot_attempted}" && (( snapshot_rc != 0 )); then
    return "${snapshot_rc}"
  fi
  if is_true "${step_attempted}" && (( step_rc != 0 )); then
    return "${step_rc}"
  fi
  return 0
}

action_run_local_pi_recovery_flow() {
  local tmp_dir
  local step_idx=0
  local max_steps
  local last_rc=0
  local stopped_reason="max_steps_reached"
  local rollback_attempted=false
  local rollback_rc=0
  local rollback_checkpoint=""
  local rollback_write_requested=false

  max_steps="${local_pi_recovery_flow_max_steps}"
  if [[ ! "${max_steps}" =~ ^[0-9]+$ ]] || (( max_steps < 1 )); then
    max_steps=3
  fi

  tmp_dir="$(mktemp -d)"
  local rollback_file="${tmp_dir}/rollback.json"

  while (( step_idx < max_steps )); do
    local step_file="${tmp_dir}/step_${step_idx}.json"
    set +e
    action_apply_local_pi_recovery_step >"${step_file}"
    last_rc=$?
    set -e
    if (( last_rc != 0 )); then
      if is_true "${local_pi_recovery_auto_rollback_on_failure}"; then
        rollback_checkpoint="$(python3 - "${step_file}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
execution_mode = str(payload.get('execution_mode') or '').strip()
step_attempted = bool(payload.get('step_attempted'))
mutation_detected = bool(payload.get('mutation_detected'))
snapshot = payload.get('snapshot_result')
if (
    step_attempted
    and execution_mode.endswith('_write')
    and mutation_detected
    and isinstance(snapshot, dict)
    and bool(snapshot.get('ok'))
):
    print(str(snapshot.get('checkpoint_manifest') or '').strip())
PY
)"
        if [[ -n "${rollback_checkpoint}" ]]; then
          rollback_attempted=true
          rollback_write_requested=false
          if is_true "${local_pi_recovery_auto_rollback_write}"; then
            rollback_write_requested=true
          fi
          set +e
          (
            local_pi_recovery_restore_checkpoint="${rollback_checkpoint}"
            local_pi_recovery_restore_write="${rollback_write_requested}"
            action_rollback_local_pi_recovery_state
          ) >"${rollback_file}"
          rollback_rc=$?
          set -e
        fi
      fi
      stopped_reason="step_failed"
      break
    fi
    if python3 - "${step_file}" "${local_pi_recovery_enforce_projection}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
enforce_projection = sys.argv[2].strip().lower() in {'1', 'true', 'yes', 'on'}
step_attempted = bool(payload.get('step_attempted'))
execution_mode = str(payload.get('execution_mode') or '').strip()
next_action = str(payload.get('next_action') or '').strip()
post_next_action = str(payload.get('post_next_action') or '').strip()
write_projection = payload.get('write_projection')

if not step_attempted:
    raise SystemExit(10)
if execution_mode.endswith('_preview') or execution_mode == 'preview':
    raise SystemExit(11)
if enforce_projection:
    if not isinstance(write_projection, dict):
        raise SystemExit(14)
    projected_steps = write_projection.get('projected_steps')
    if not isinstance(projected_steps, list) or not projected_steps:
        raise SystemExit(15)
    first_step = projected_steps[0]
    expected_map = {
        'ack_write': 'ack_write',
        'strict_backfill_write': 'strict_backfill_write',
        'fallback_backfill_write': 'fallback_backfill_write',
        'full_cycle': 'full_cycle_gate_ready',
    }
    expected_simulated_step = expected_map.get(execution_mode, execution_mode)
    actual_simulated_step = str(first_step.get('simulated_step') or '').strip()
    if expected_simulated_step != actual_simulated_step:
        raise SystemExit(16)
    if execution_mode == 'full_cycle':
        is_terminal = bool(first_step.get('terminal')) or str(write_projection.get('terminal_action') or '').strip() == 'full_cycle_gate_ready'
        if not is_terminal:
            raise SystemExit(17)
if not post_next_action:
    raise SystemExit(12)
if enforce_projection:
    projected_next_action = str(first_step.get('projected_next_action') or '').strip()
    if projected_next_action and projected_next_action != post_next_action:
        raise SystemExit(18)
if post_next_action == next_action:
    raise SystemExit(13)
raise SystemExit(0)
PY
    then
      step_idx=$((step_idx + 1))
      continue
    else
      case "$?" in
        10) stopped_reason="no_step_attempted" ;;
        11) stopped_reason="preview_mode" ;;
        12) stopped_reason="no_follow_up_action" ;;
        13) stopped_reason="no_progression" ;;
        14) stopped_reason="projection_missing" ;;
        15) stopped_reason="projection_empty" ;;
        16) stopped_reason="projection_step_mismatch" ;;
        17) stopped_reason="projection_terminal_mismatch" ;;
        18) stopped_reason="projection_next_action_mismatch" ;;
        *) stopped_reason="step_termination_condition" ;;
      esac
      break
    fi
  done

  python3 - "${tmp_dir}" "${max_steps}" "${last_rc}" "${stopped_reason}" "${rollback_file}" \
    "${rollback_attempted}" "${rollback_rc}" "${rollback_checkpoint}" "${rollback_write_requested}" \
    "${local_pi_recovery_enforce_projection}" \
    "${local_pi_recovery_artifacts_enabled}" "${local_pi_recovery_artifact_dir}" "${artifact_ttl_hours}" "${system_root}" <<'PY'
import json
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

tmp_dir = Path(sys.argv[1])
max_steps = int(sys.argv[2])
last_rc = int(sys.argv[3])
stopped_reason = sys.argv[4]
rollback_path = Path(sys.argv[5])
rollback_attempted = sys.argv[6].strip().lower() in {'1','true','yes','on'}
rollback_rc = int(sys.argv[7])
rollback_checkpoint = sys.argv[8].strip() or None
rollback_write_requested = sys.argv[9].strip().lower() in {'1','true','yes','on'}
projection_guard_enforced = sys.argv[10].strip().lower() in {'1','true','yes','on'}
artifacts_enabled = sys.argv[11].strip().lower() in {'1','true','yes','on'}
artifact_dir = Path(sys.argv[12]).expanduser().resolve()
artifact_ttl_hours = max(1.0, float(sys.argv[13]))
system_root = Path(sys.argv[14]).expanduser().resolve()

steps = []
for path in sorted(tmp_dir.glob("step_*.json")):
    if path.stat().st_size <= 0:
        continue
    steps.append(json.loads(path.read_text(encoding='utf-8')))

checkpoints = []
for idx, step in enumerate(steps):
    snapshot_result = step.get('snapshot_result') if isinstance(step, dict) else None
    if not isinstance(snapshot_result, dict) or not bool(snapshot_result.get('ok')):
        continue
    checkpoints.append(
        {
            'step_index': idx,
            'checkpoint_root': snapshot_result.get('checkpoint_root'),
            'checkpoint_manifest': snapshot_result.get('checkpoint_manifest'),
            'checkpoint_checksum': snapshot_result.get('checkpoint_checksum'),
            'state_fingerprint': snapshot_result.get('state_fingerprint'),
        }
    )

last_payload = steps[-1] if steps else {}
final_status = None
if isinstance(last_payload, dict):
    final_status = last_payload.get('post_status_payload') or last_payload.get('status_payload')
rollback_result = None
if rollback_attempted and rollback_path.exists() and rollback_path.stat().st_size > 0:
    rollback_result = json.loads(rollback_path.read_text(encoding='utf-8'))
latest_checkpoint_manifest = None
if checkpoints:
    latest_checkpoint_manifest = str(checkpoints[-1].get('checkpoint_manifest') or '').strip() or None
rollback_guidance = None
guidance_checkpoint = rollback_checkpoint or latest_checkpoint_manifest
if guidance_checkpoint:
    rollback_guidance = {
        'checkpoint_manifest': guidance_checkpoint,
        'dry_run_command': (
            f'cd {system_root} && '
            f'LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT="{guidance_checkpoint}" '
            'scripts/openclaw_cloud_bridge.sh rollback-local-pi-recovery-state'
        ),
        'write_command': (
            f'cd {system_root} && '
            'LOCAL_PI_RECOVERY_RESTORE_WRITE=true '
            f'LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT="{guidance_checkpoint}" '
            'scripts/openclaw_cloud_bridge.sh rollback-local-pi-recovery-state'
        ),
    }
operator_note = None
if rollback_guidance is not None:
    if rollback_attempted and isinstance(rollback_result, dict):
        summary = f"Rollback hook was attempted after a failed write step; latest rollback status is {rollback_result.get('status') or 'unknown'}."
        when_to_use = 'Review rollback_result first. If dry-run looked correct and state still needs reversion, use the write command.'
    else:
        summary = 'Latest recovery checkpoint is available for rollback if a subsequent recovery write needs to be reverted.'
        when_to_use = 'Use the dry-run command first after any real recovery write if verification looks wrong.'
    operator_note = {
        'summary': summary,
        'when_to_use': when_to_use,
        'recommended_first_action': rollback_guidance['dry_run_command'],
        'escalation_action': rollback_guidance['write_command'],
        'checkpoint_manifest': rollback_guidance['checkpoint_manifest'],
    }

def classify_step_status(step: dict) -> str:
    if not isinstance(step, dict):
        return 'unknown'
    execution_mode = str(step.get('execution_mode') or '').strip()
    status = str(step.get('status') or '').strip()
    mutation_detected = bool(step.get('mutation_detected'))
    rollback_guidance_step = step.get('rollback_guidance')
    if execution_mode.endswith('_preview') or execution_mode == 'preview':
        return 'preview'
    if status == 'snapshot_failed':
        return 'snapshot-failed'
    if status == 'step_failed':
        if mutation_detected and isinstance(rollback_guidance_step, dict):
            return 'rollback-ready'
        if mutation_detected:
            return 'write-failed'
        return 'write-failed-no-mutation'
    if execution_mode.endswith('_write'):
        return 'write-ok' if mutation_detected else 'write-noop'
    if execution_mode == 'full_cycle':
        return 'execute'
    return status or execution_mode or 'unknown'

if stopped_reason == 'preview_mode':
    flow_status_label = 'preview'
elif rollback_attempted and isinstance(rollback_result, dict):
    flow_status_label = 'rollback-attempted'
elif stopped_reason == 'step_failed' and rollback_guidance is not None:
    flow_status_label = 'rollback-ready'
elif stopped_reason == 'step_failed':
    flow_status_label = 'write-failed'
elif stopped_reason == 'no_follow_up_action':
    flow_status_label = 'flow-complete'
else:
    flow_status_label = str(stopped_reason or 'unknown')

out = {
    'action': 'run_local_pi_recovery_flow',
    'ok': bool(steps) and last_rc == 0,
    'status': 'ok' if bool(steps) and last_rc == 0 else 'flow_failed',
    'max_steps': max_steps,
    'step_count': len(steps),
    'stopped_reason': stopped_reason,
    'last_returncode': last_rc,
    'steps': steps,
    'checkpoints': checkpoints,
    'checkpoint_count': len(checkpoints),
    'final_status_payload': final_status,
    'final_next_action': None if not isinstance(last_payload, dict) else (last_payload.get('post_next_action') or last_payload.get('next_action')),
    'final_action_level': None if not isinstance(last_payload, dict) else (last_payload.get('post_action_level') or last_payload.get('action_level')),
    'initial_write_projection': None if not steps else steps[0].get('write_projection'),
    'final_write_projection': None if not isinstance(last_payload, dict) else (last_payload.get('post_write_projection') or last_payload.get('write_projection')),
    'projection_guard_enforced': projection_guard_enforced,
    'rollback_attempted': rollback_attempted,
    'rollback_returncode': rollback_rc,
    'rollback_checkpoint_manifest': rollback_checkpoint,
    'rollback_write_requested': rollback_write_requested,
    'rollback_result': rollback_result,
    'rollback_guidance': rollback_guidance,
    'operator_note': operator_note,
    'artifact': None,
    'checksum': None,
    'step_artifacts': [],
    'artifact_ttl_hours': artifact_ttl_hours,
    'artifact_status_label': flow_status_label,
    'artifact_label': f'recovery-flow:{stopped_reason}',
    'artifact_tags': [
        'local-pi',
        'recovery-flow',
        str(stopped_reason or 'unknown'),
    ] + (['rollback-attempted'] if rollback_attempted else []),
}

if artifacts_enabled:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ts = now.strftime('%Y%m%dT%H%M%SZ')
    artifact_path = artifact_dir / f'{ts}_local_pi_recovery_flow.json'
    checksum_path = artifact_dir / f'{ts}_local_pi_recovery_flow_checksum.json'
    out['artifact'] = str(artifact_path)
    out['checksum'] = str(checksum_path)
    step_artifacts = []
    for idx, step in enumerate(steps):
        step_artifact_path = artifact_dir / f'{ts}_local_pi_recovery_flow_step_{idx:02d}.json'
        step_checksum_path = artifact_dir / f'{ts}_local_pi_recovery_flow_step_{idx:02d}_checksum.json'
        step_execution_mode = str(step.get('execution_mode') or 'unknown').strip() or 'unknown'
        step_status_label = classify_step_status(step)
        step_tags = [
            'local-pi',
            'recovery-step',
            step_execution_mode,
            step_status_label,
        ]
        if bool(step.get('mutation_detected')):
            step_tags.append('mutation-detected')
        if bool(step.get('snapshot_attempted')):
            step_tags.append('has-checkpoint')
        step_payload = {
            'action': 'local_pi_recovery_flow_step',
            'generated_at': now.isoformat(),
            'step_index': idx,
            'flow_artifact': str(artifact_path),
            'artifact_status_label': step_status_label,
            'artifact_label': f'recovery-step:{step_execution_mode}',
            'artifact_tags': step_tags,
            'operator_note': step.get('operator_note'),
            'rollback_guidance': step.get('rollback_guidance'),
            'payload': step,
        }
        step_artifact_path.write_text(json.dumps(step_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        step_digest = hashlib.sha256(step_artifact_path.read_bytes()).hexdigest()
        step_checksum_payload = {
            'generated_at': now.isoformat(),
            'artifact_ttl_hours': artifact_ttl_hours,
            'files': [
                {
                    'path': str(step_artifact_path),
                    'sha256': step_digest,
                    'size_bytes': int(step_artifact_path.stat().st_size),
                }
            ],
        }
        step_checksum_path.write_text(json.dumps(step_checksum_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        step_artifacts.append(
            {
                'step_index': idx,
                'artifact': str(step_artifact_path),
                'checksum': str(step_checksum_path),
            }
        )
    out['step_artifacts'] = step_artifacts
    artifact_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    checksum_payload = {
        'generated_at': now.isoformat(),
        'artifact_ttl_hours': artifact_ttl_hours,
        'files': [
            {
                'path': str(artifact_path),
                'sha256': digest,
                'size_bytes': int(artifact_path.stat().st_size),
            }
        ],
    }
    checksum_path.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    cutoff = now - timedelta(hours=artifact_ttl_hours)
    for pattern in (
        '*_local_pi_recovery_flow.json',
        '*_local_pi_recovery_flow_checksum.json',
        '*_local_pi_recovery_flow_step_*.json',
        '*_local_pi_recovery_flow_step_*_checksum.json',
    ):
        for candidate in artifact_dir.glob(pattern):
            if candidate.name in {artifact_path.name, checksum_path.name}:
                continue
            if candidate.name in {Path(item['artifact']).name for item in step_artifacts}:
                continue
            if candidate.name in {Path(item['checksum']).name for item in step_artifacts}:
                continue
            try:
                mtime = datetime.fromtimestamp(candidate.stat().st_mtime, tz=timezone.utc)
            except Exception:
                continue
            if mtime < cutoff:
                try:
                    candidate.unlink()
                except Exception:
                    pass
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
  local rc_out=$?
  rm -rf "${tmp_dir}" >/dev/null 2>&1 || true
  if (( rc_out != 0 )); then
    return "${rc_out}"
  fi
  return "${last_rc}"
}

action_backfill_local_pi_last_loss_ts() {
  local script_path
  local pulse_lock_path
  local expected_state_fingerprint
  local -a cmd

  script_path="${local_pi_workspace_system_root}/scripts/backfill_paper_last_loss_ts.py"
  pulse_lock_path="${local_pi_workspace_system_root}/output/state/run_halfhour_pulse.lock"
  expected_state_fingerprint="${LOCAL_PI_EXPECTED_STATE_FINGERPRINT:-}"

  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "backfill_local_pi_last_loss_ts",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  cmd=(
    python3 "${script_path}"
    --state-path "${local_pi_workspace_system_root}/output/state/spot_paper_state.json"
    --ledger-path "${local_pi_workspace_system_root}/output/logs/paper_execution_ledger.jsonl"
    --pulse-lock-path "${pulse_lock_path}"
  )
  if [[ -n "${expected_state_fingerprint}" ]]; then
    cmd+=(--expected-state-fingerprint "${expected_state_fingerprint}")
  fi
  if is_true "${local_pi_last_loss_ts_backfill_allow_latest_loss_fallback}"; then
    cmd+=(--allow-latest-loss-fallback)
  fi
  if is_true "${local_pi_last_loss_ts_backfill_write}"; then
    cmd+=(--write)
  fi

  FENLIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  LIE_SYSTEM_ROOT="${local_pi_workspace_system_root}" \
  "${cmd[@]}"
}

action_ensure_remote_openclaw_runtime_model() {
  ssh_exec "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; python3 scripts/ensure_openclaw_runtime_model.py --config \"\$HOME/.openclaw/openclaw.json\""
}

live_risk_daemon_service_installed_from_json() {
  local service_json="$1"
  python3 - "${service_json}" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
raise SystemExit(0 if bool(data.get('service_installed', False)) else 1)
PY
}

rewrite_live_risk_daemon_service_action() {
  local service_json="$1"
  local action_name="$2"
  python3 - "${service_json}" "${action_name}" <<'PY'
import json
import sys

data = json.loads(sys.argv[1])
data['action'] = str(sys.argv[2]).strip() or data.get('action', '')
data['delegated'] = 'systemd_service'
payload = data.get('payload', {}) if isinstance(data.get('payload', {}), dict) else {}
systemd = data.get('systemd', {}) if isinstance(data.get('systemd', {}), dict) else {}
main_pid = int(systemd.get('main_pid', 0) or 0)
active_state = str(systemd.get('active_state', ''))
data['exists'] = bool(data.get('service_installed', False))
data['running'] = bool(payload.get('running', False)) or (active_state == 'active' and main_pid > 0)
data['pid'] = int(payload.get('pid', 0) or 0) if int(payload.get('pid', 0) or 0) > 0 else main_pid
data['pid_alive'] = active_state == 'active' and main_pid > 0
data['status'] = str(payload.get('status', '')).strip() or active_state or 'missing'
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
}

run_live_risk_daemon_service_start_remote() {
  run_live_takeover_remote \
    "live-risk-daemon-start" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${live_risk_daemon_unit_name}' <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

unit_name = str(sys.argv[1]).strip()
unit_path = Path('/etc/systemd/system') / unit_name
state_path = Path('output/state/live_risk_daemon.json')

payload = {}
if state_path.exists():
    try:
        loaded = json.loads(state_path.read_text(encoding='utf-8'))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        payload = {'status': 'invalid_state', 'error': str(exc)}

show_before = subprocess.run(
    ['systemctl', 'show', unit_name, '-p', 'ActiveState', '-p', 'SubState', '-p', 'UnitFileState', '-p', 'MainPID', '-p', 'FragmentPath'],
    text=True,
    capture_output=True,
    check=False,
)
before = {}
for line in (show_before.stdout or '').splitlines():
    if '=' not in line:
        continue
    k, v = line.split('=', 1)
    before[k] = v
already_running = before.get('ActiveState', '') == 'active' and int(before.get('MainPID', '0') or 0) > 0

if unit_path.exists():
    subprocess.run(['systemctl', 'start', unit_name], check=True)
    deadline = time.time() + 5.0
    after = {}
    while time.time() < deadline:
        show_after = subprocess.run(
            ['systemctl', 'show', unit_name, '-p', 'ActiveState', '-p', 'SubState', '-p', 'UnitFileState', '-p', 'MainPID', '-p', 'FragmentPath'],
            text=True,
            capture_output=True,
            check=False,
        )
        after = {}
        for line in (show_after.stdout or '').splitlines():
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            after[k] = v
        if after.get('ActiveState', '') == 'active' and int(after.get('MainPID', '0') or 0) > 0:
            break
        time.sleep(0.2)
else:
    after = before

state_deadline = time.time() + 5.0
target_pid = int(after.get('MainPID', '0') or 0)
while time.time() < state_deadline:
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding='utf-8'))
            payload = loaded if isinstance(loaded, dict) else {}
        except Exception as exc:
            payload = {'status': 'invalid_state', 'error': str(exc)}
        if bool(payload.get('running', False)) and (target_pid <= 0 or int(payload.get('pid', 0) or 0) == target_pid):
            break
    time.sleep(0.2)

out = {
    'action': 'live-risk-daemon-start',
    'delegated': 'systemd_service',
    'service_installed': bool(unit_path.exists()),
    'started': after.get('ActiveState', '') == 'active',
    'already_running': bool(already_running),
    'unit_name': unit_name,
    'unit_path': str(unit_path),
    'systemd': {
        'active_state': after.get('ActiveState', 'missing'),
        'sub_state': after.get('SubState', ''),
        'unit_file_state': after.get('UnitFileState', ''),
        'main_pid': int(after.get('MainPID', '0') or 0),
        'fragment_path': after.get('FragmentPath', str(unit_path) if unit_path.exists() else ''),
    },
    'payload': payload,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

run_live_risk_daemon_service_stop_remote() {
  run_live_takeover_remote \
    "live-risk-daemon-stop" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${live_risk_daemon_unit_name}' <<'PY'
import json
import subprocess
import sys
import time
from pathlib import Path

unit_name = str(sys.argv[1]).strip()
unit_path = Path('/etc/systemd/system') / unit_name
state_path = Path('output/state/live_risk_daemon.json')

if unit_path.exists():
    subprocess.run(['systemctl', 'stop', unit_name], check=True)

deadline = time.time() + 5.0
show_map = {}
while time.time() < deadline:
    proc = subprocess.run(
        ['systemctl', 'show', unit_name, '-p', 'ActiveState', '-p', 'SubState', '-p', 'UnitFileState', '-p', 'MainPID', '-p', 'FragmentPath'],
        text=True,
        capture_output=True,
        check=False,
    )
    show_map = {}
    for line in (proc.stdout or '').splitlines():
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        show_map[k] = v
    if show_map.get('ActiveState', 'inactive') != 'active' and int(show_map.get('MainPID', '0') or 0) == 0:
        break
    time.sleep(0.2)

payload = {}
state_deadline = time.time() + 5.0
while time.time() < state_deadline:
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding='utf-8'))
            payload = loaded if isinstance(loaded, dict) else {}
        except Exception as exc:
            payload = {'status': 'invalid_state', 'error': str(exc)}
        if not bool(payload.get('running', False)):
            break
    time.sleep(0.2)

out = {
    'action': 'live-risk-daemon-stop',
    'delegated': 'systemd_service',
    'service_installed': bool(unit_path.exists()),
    'stopped': show_map.get('ActiveState', 'inactive') != 'active',
    'force_killed': False,
    'unit_name': unit_name,
    'unit_path': str(unit_path),
    'systemd': {
        'active_state': show_map.get('ActiveState', 'inactive'),
        'sub_state': show_map.get('SubState', ''),
        'unit_file_state': show_map.get('UnitFileState', ''),
        'main_pid': int(show_map.get('MainPID', '0') or 0),
        'fragment_path': show_map.get('FragmentPath', str(unit_path) if unit_path.exists() else ''),
    },
    'payload': payload,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

action_live_risk_daemon_status() {
  local service_json=""
  service_json="$(run_live_risk_daemon_service_status_remote 2>/dev/null || true)"
  if [[ -n "${service_json}" ]] && live_risk_daemon_service_installed_from_json "${service_json}"; then
    rewrite_live_risk_daemon_service_action "${service_json}" "live-risk-daemon-status"
    return 0
  fi
  run_live_risk_daemon_status_remote
}

action_live_risk_daemon_start() {
  local service_json=""
  service_json="$(run_live_risk_daemon_service_status_remote 2>/dev/null || true)"
  if [[ -n "${service_json}" ]] && live_risk_daemon_service_installed_from_json "${service_json}"; then
    run_live_risk_daemon_service_start_remote
    return 0
  fi
  local date_value min_conf_value min_conv_value
  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  run_live_takeover_remote \
    "live-risk-daemon-start" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; PYTHONPATH=src python3 - '${date_value}' '${live_risk_daemon_poll_seconds}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_daemon_history_limit}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

date_value = str(sys.argv[1]).strip()
poll_seconds = str(sys.argv[2]).strip()
guard_timeout_seconds = str(sys.argv[3]).strip()
history_limit = str(sys.argv[4]).strip()
ticket_freshness_seconds = str(sys.argv[5]).strip()
panic_cooldown_seconds = str(sys.argv[6]).strip()
max_daily_loss_ratio = str(sys.argv[7]).strip()
max_open_exposure_ratio = str(sys.argv[8]).strip()
ticket_symbols = str(sys.argv[9]).strip()
ticket_max_age_days = str(sys.argv[10]).strip()
min_conf_value = str(sys.argv[11]).strip()
min_conv_value = str(sys.argv[12]).strip()

state_path = Path('output/state/live_risk_daemon.json')
log_path = Path('output/logs/live_risk_daemon.out.log')
state_path.parent.mkdir(parents=True, exist_ok=True)
log_path.parent.mkdir(parents=True, exist_ok=True)

payload = {}
if state_path.exists():
    try:
        loaded = json.loads(state_path.read_text(encoding='utf-8'))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        payload = {'status': 'invalid_state', 'error': str(exc)}

pid = int(payload.get('pid', 0) or 0)
alive = False
if pid > 0 and bool(payload.get('running', False)):
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False

if alive:
    out = {
        'action': 'live-risk-daemon-start',
        'started': False,
        'already_running': True,
        'pid': pid,
        'payload': payload,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    raise SystemExit(0)

cmd = [
    'python3',
    'scripts/live_risk_daemon.py',
    '--config', 'config.yaml',
    '--output-root', 'output',
    '--review-dir', 'output/review',
    '--poll-seconds', poll_seconds,
    '--guard-timeout-seconds', guard_timeout_seconds,
    '--history-limit', history_limit,
    '--ticket-freshness-seconds', ticket_freshness_seconds,
    '--panic-cooldown-seconds', panic_cooldown_seconds,
    '--max-daily-loss-ratio', max_daily_loss_ratio,
    '--max-open-exposure-ratio', max_open_exposure_ratio,
    '--ticket-symbols', ticket_symbols,
    '--ticket-max-age-days', ticket_max_age_days,
]
if date_value:
    cmd.extend(['--date', date_value])
if min_conf_value:
    cmd.extend(['--ticket-min-confidence', min_conf_value])
if min_conv_value:
    cmd.extend(['--ticket-min-convexity', min_conv_value])

env = dict(os.environ)
env['PYTHONPATH'] = 'src'
with log_path.open('a', encoding='utf-8') as log_fh:
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
time.sleep(1.0)

alive = False
try:
    os.kill(proc.pid, 0)
    alive = True
except OSError:
    alive = False

payload = {}
if state_path.exists():
    try:
        loaded = json.loads(state_path.read_text(encoding='utf-8'))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        payload = {'status': 'invalid_state', 'error': str(exc)}

out = {
    'action': 'live-risk-daemon-start',
    'started': bool(alive),
    'already_running': False,
    'pid': int(proc.pid),
    'state_exists': bool(state_path.exists()),
    'payload': payload,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

action_live_risk_daemon_stop() {
  local service_json=""
  service_json="$(run_live_risk_daemon_service_status_remote 2>/dev/null || true)"
  if [[ -n "${service_json}" ]] && live_risk_daemon_service_installed_from_json "${service_json}"; then
    run_live_risk_daemon_service_stop_remote
    return 0
  fi
  run_live_takeover_remote \
    "live-risk-daemon-stop" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; state='output/state/live_risk_daemon.json'; python3 - \"\$state\" <<'PY'
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(sys.argv[1])
checksum_path = state_path.with_name(f'{state_path.stem}_checksum.json')
payload = {}
if state_path.exists():
    try:
        loaded = json.loads(state_path.read_text(encoding='utf-8'))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        payload = {'status': 'invalid_state', 'error': str(exc)}
pid = int(payload.get('pid', 0) or 0)
alive = False
if pid > 0:
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
forced = False
if alive:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        alive = False
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            alive = False
            break
        time.sleep(0.1)
    if alive:
        try:
            os.kill(pid, signal.SIGKILL)
            forced = True
        except OSError:
            pass
        time.sleep(0.2)
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False

if state_path.parent.exists():
    payload['status'] = 'stopped'
    payload['running'] = False
    payload['stop_reason'] = 'bridge_stop_force_kill' if forced else 'bridge_stop'
    payload['next_run_at_utc'] = None
    payload['stopped_at_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    payload['updated_at_utc'] = payload['stopped_at_utc']
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    checksum_path.write_text(
        json.dumps(
            {
                'generated_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'files': [{'path': str(state_path), 'sha256': digest, 'size_bytes': int(state_path.stat().st_size)}],
            },
            ensure_ascii=False,
            indent=2,
        ) + '\\n',
        encoding='utf-8',
    )

out = {
    'action': 'live-risk-daemon-stop',
    'pid': pid,
    'stopped': not alive,
    'force_killed': bool(forced),
    'state_path': str(state_path),
    'checksum': str(checksum_path),
    'payload': payload,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

action_live_risk_daemon_service_status() {
  run_live_risk_daemon_service_status_remote
}

action_live_risk_daemon_security_status() {
  run_live_risk_daemon_security_status_remote
}

action_live_risk_daemon_journal() {
  run_live_risk_daemon_journal_remote
}

action_live_ops_reconcile_status() {
  run_live_ops_reconcile_status_remote
}

action_live_ops_reconcile_refresh() {
  run_live_ops_reconcile_refresh_remote
}

action_live_risk_daemon_mdwe_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_mdwe_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_mdwe_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-mdwe-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_mdwe_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'mdwe_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-mdwe-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-mdwe-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
        'RestrictSUIDSGID': True,
        'SystemCallArchitectures': 'native',
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_protecthome_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_protecthome_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_protecthome_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-protecthome-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_protecthome_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'protecthome_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-protecthome-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
 ] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-protecthome-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'ProtectHome': 'read-only',
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_procsubset_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_procsubset_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_procsubset_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-procsubset-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_procsubset_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'procsubset_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-procsubset-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-procsubset-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_privateusers_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_privateusers_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_privateusers_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-privateusers-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_privateusers_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'privateusers_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-privateusers-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'PrivateUsers=true',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-privateusers-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'PrivateUsers': True,
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_privatenetwork_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_privatenetwork_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_privatenetwork_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-privatenetwork-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_privatenetwork_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'privatenetwork_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-privatenetwork-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'PrivateUsers=true',
    '--property', 'PrivateNetwork=true',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-privatenetwork-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'PrivateNetwork': True,
        'PrivateUsers': True,
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_ipdeny_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_ipdeny_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_ipdeny_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-ipdeny-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_ipdeny_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'ipdeny_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-ipdeny-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'PrivateUsers=true',
    '--property', 'PrivateNetwork=true',
    '--property', 'IPAddressDeny=any',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-ipdeny-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'IPAddressDeny': 'any',
        'PrivateNetwork': True,
        'PrivateUsers': True,
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_devicepolicy_probe() {
  local script_path tmp_probe rc
  script_path="${system_root}/scripts/build_remote_live_devicepolicy_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_devicepolicy_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  local date_value min_conf_value min_conv_value
  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-devicepolicy-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_devicepolicy_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'devicepolicy_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-devicepolicy-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'DevicePolicy=closed',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'PrivateUsers=true',
    '--property', 'PrivateNetwork=true',
    '--property', 'IPAddressDeny=any',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=AF_UNIX',
    '--property', 'SystemCallFilter=@system-service',
    '--property', 'SystemCallFilter=~@resources',
    '--property', 'SystemCallFilter=~@privileged',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-devicepolicy-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'DevicePolicy': 'closed',
        'PrivateDevices': True,
        'PrivateNetwork': True,
        'IPAddressDeny': 'any',
        'RestrictAddressFamilies': 'AF_UNIX',
        'PrivateUsers': True,
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'MemoryDenyWriteExecute': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status') or ''),
    'probe_payload_reasons': probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons'), list) else [],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY" \
    >"${tmp_probe}" 2>&1
  rc=$?
  set -e
  python3 "${script_path}" \
    --review-dir "${output_dir}" \
    --probe-file "${tmp_probe}" \
    --probe-returncode "${rc}" \
    --artifact-ttl-hours "${artifact_ttl_hours}" \
    --artifact-keep 12
  rm -f "${tmp_probe}"
}

action_live_risk_daemon_afunix_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_afunix_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_afunix_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-afunix-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_afunix_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'afunix_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-afunix-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'PrivateUsers=true',
    '--property', 'PrivateNetwork=true',
    '--property', 'IPAddressDeny=any',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=AF_UNIX',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-afunix-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'RestrictAddressFamilies': 'AF_UNIX',
        'IPAddressDeny': 'any',
        'PrivateNetwork': True,
        'PrivateUsers': True,
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_noaf_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_noaf_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_noaf_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-noaf-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_noaf_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'noaf_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-noaf-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'PrivateUsers=true',
    '--property', 'PrivateNetwork=true',
    '--property', 'IPAddressDeny=any',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'RestrictAddressFamilies=none',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', 'SystemCallFilter=@system-service',
    '--property', 'SystemCallFilter=~@resources',
    '--property', 'SystemCallFilter=~@privileged',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-noaf-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'RestrictAddressFamilies': 'none',
        'IPAddressDeny': 'any',
        'PrivateNetwork': True,
        'PrivateUsers': True,
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_syscallfilter_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_syscallfilter_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_syscallfilter_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-syscallfilter-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_syscallfilter_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'syscallfilter_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-syscallfilter-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'PrivateUsers=true',
    '--property', 'PrivateNetwork=true',
    '--property', 'IPAddressDeny=any',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'SystemCallFilter=@system-service',
    '--property', 'RestrictAddressFamilies=AF_UNIX',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-syscallfilter-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'SystemCallFilter': '@system-service',
        'RestrictAddressFamilies': 'AF_UNIX',
        'IPAddressDeny': 'any',
        'PrivateNetwork': True,
        'PrivateUsers': True,
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_syscallfilter_tight_probe() {
  local script_path
  local tmp_probe
  local rc_probe
  local date_value min_conf_value min_conv_value
  local -a cmd

  script_path="${system_root}/scripts/build_remote_live_syscallfilter_tight_probe.py"
  if [[ ! -r "${script_path}" ]]; then
    python3 - "${script_path}" <<'PY'
import json
import sys
out = {
    "action": "live_risk_daemon_syscallfilter_tight_probe",
    "ok": False,
    "status": "script_missing",
    "script_path": str(sys.argv[1]).strip(),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 4
  fi

  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  tmp_probe="$(mktemp)"
  set +e
  run_live_takeover_remote \
    "live-risk-daemon-syscallfilter-tight-probe" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${live_risk_daemon_syscallfilter_tight_probe_timeout_seconds}' '${date_value}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' <<'PY'
import json
import os
import pwd
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

service_user = str(sys.argv[1]).strip()
_installed_unit_name = str(sys.argv[2]).strip()
probe_timeout_seconds = max(5, int(sys.argv[3]))
date_value = str(sys.argv[4]).strip()
guard_timeout_seconds = max(5, int(sys.argv[5]))
ticket_freshness_seconds = max(1, int(sys.argv[6]))
panic_cooldown_seconds = max(1, int(sys.argv[7]))
max_daily_loss_ratio = float(sys.argv[8])
max_open_exposure_ratio = float(sys.argv[9])
ticket_symbols = str(sys.argv[10]).strip()
ticket_max_age_days = max(1, int(sys.argv[11]))
min_conf_value = str(sys.argv[12]).strip()
min_conv_value = str(sys.argv[13]).strip()

wd = Path.cwd()
probe_root = wd / 'output' / 'artifacts' / 'syscallfilter_tight_probe' / f'probe_{int(time.time())}'
probe_output_root = probe_root / 'output'
probe_review_dir = probe_output_root / 'review'
probe_state_dir = probe_output_root / 'state'
probe_output_root.mkdir(parents=True, exist_ok=True)
probe_review_dir.mkdir(parents=True, exist_ok=True)
probe_state_dir.mkdir(parents=True, exist_ok=True)
user_entry = pwd.getpwnam(service_user)
for path in (probe_root, probe_output_root, probe_review_dir, probe_state_dir):
    os.chown(path, user_entry.pw_uid, user_entry.pw_gid)
probe_unit = f'fenlie-live-risk-daemon-syscallfilter-tight-probe-{int(time.time())}'

daemon_cmd = [
    '/usr/bin/env',
    'PYTHONPATH=src',
    'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3',
    'scripts/live_risk_daemon.py',
    '--config',
    'config.yaml',
    '--output-root',
    str(probe_output_root),
    '--review-dir',
    str(probe_review_dir),
    '--poll-seconds',
    '1',
    '--guard-timeout-seconds',
    str(guard_timeout_seconds),
    '--history-limit',
    '1',
    '--ticket-freshness-seconds',
    str(ticket_freshness_seconds),
    '--panic-cooldown-seconds',
    str(panic_cooldown_seconds),
    '--max-daily-loss-ratio',
    f'{max_daily_loss_ratio:.6f}',
    '--max-open-exposure-ratio',
    f'{max_open_exposure_ratio:.6f}',
    '--ticket-symbols',
    ticket_symbols,
    '--ticket-max-age-days',
    str(ticket_max_age_days),
    '--max-cycles',
    '1',
]
if date_value:
    daemon_cmd.extend(['--date', date_value])
if min_conf_value:
    daemon_cmd.extend(['--ticket-min-confidence', str(float(min_conf_value))])
if min_conv_value:
    daemon_cmd.extend(['--ticket-min-convexity', str(float(min_conv_value))])

run_cmd = [
    'sudo', '-n',
    'systemd-run',
    '--wait',
    '--collect',
    '--service-type=exec',
    '--property', f'User={service_user}',
    '--property', f'WorkingDirectory={wd}',
    '--property', 'Environment=PYTHONUNBUFFERED=1',
    '--property', 'Environment=PYTHONDONTWRITEBYTECODE=1',
    '--property', 'NoNewPrivileges=true',
    '--property', 'PrivateTmp=true',
    '--property', 'PrivateDevices=true',
    '--property', 'ProtectSystem=strict',
    '--property', 'ProtectHostname=true',
    '--property', 'ProtectControlGroups=true',
    '--property', 'ProtectKernelTunables=true',
    '--property', 'ProtectKernelModules=true',
    '--property', 'ProtectKernelLogs=true',
    '--property', 'ProtectClock=true',
    '--property', 'ProtectProc=invisible',
    '--property', 'ProtectHome=read-only',
    '--property', 'ProcSubset=pid',
    '--property', 'PrivateUsers=true',
    '--property', 'PrivateNetwork=true',
    '--property', 'IPAddressDeny=any',
    '--property', 'RestrictNamespaces=true',
    '--property', 'RestrictSUIDSGID=true',
    '--property', 'RestrictRealtime=true',
    '--property', 'LockPersonality=true',
    '--property', 'SystemCallArchitectures=native',
    '--property', 'SystemCallFilter=@system-service',
    '--property', 'SystemCallFilter=~@resources',
    '--property', 'SystemCallFilter=~@privileged',
    '--property', 'RestrictAddressFamilies=AF_UNIX',
    '--property', 'MemoryDenyWriteExecute=true',
    '--property', f'ReadWritePaths={probe_output_root} {probe_review_dir}',
    '--unit', probe_unit,
    '--',
] + daemon_cmd

started_at = time.time()
proc = subprocess.run(run_cmd, text=True, capture_output=True, check=False, timeout=probe_timeout_seconds)
finished_at = time.time()
journal = subprocess.run(
    ['sudo', '-n', 'journalctl', '-u', probe_unit, '-n', '40', '--no-pager', '-o', 'cat'],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text, limit=20):
    return list(deque([x.rstrip() for x in str(text or '').splitlines() if x.strip()], maxlen=limit))

probe_payload = {}
probe_files = sorted(probe_review_dir.glob('*_live_risk_guard.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if probe_files:
    try:
        loaded = json.loads(probe_files[0].read_text(encoding='utf-8'))
        probe_payload = loaded if isinstance(loaded, dict) else {}
    except Exception:
        probe_payload = {}

out = {
    'action': 'live-risk-daemon-syscallfilter-tight-probe',
    'ok': bool(proc.returncode == 0),
    'status': 'compatible' if proc.returncode == 0 else 'incompatible',
    'unit': probe_unit,
    'timeout_seconds': int(probe_timeout_seconds),
    'returncode': int(proc.returncode),
    'started_at_epoch': float(started_at),
    'finished_at_epoch': float(finished_at),
    'duration_seconds': max(0.0, float(finished_at - started_at)),
    'properties': {
        'SystemCallFilter': ['@system-service', '~@resources', '~@privileged'],
        'RestrictAddressFamilies': 'AF_UNIX',
        'IPAddressDeny': 'any',
        'PrivateNetwork': True,
        'PrivateUsers': True,
        'ProtectProc': 'invisible',
        'ProtectHome': 'read-only',
        'ProcSubset': 'pid',
        'MemoryDenyWriteExecute': True,
        'NoNewPrivileges': True,
        'LockPersonality': True,
    },
    'stdout_tail': tail_lines(proc.stdout),
    'stderr_tail': tail_lines(proc.stderr),
    'journal_tail': tail_lines(journal.stdout),
    'probe_output_root': str(probe_output_root),
    'probe_artifact': str(probe_files[0]) if probe_files else '',
    'probe_payload_status': str(probe_payload.get('status', '')),
    'probe_payload_reasons': [
        str(x) for x in (probe_payload.get('reasons', []) if isinstance(probe_payload.get('reasons', []), list) else [])[:8]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
shutil.rmtree(probe_root, ignore_errors=True)
PY" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e

  cmd=(
    python3 "${script_path}"
    --review-dir "${output_dir}"
    --probe-file "${tmp_probe}"
    --probe-returncode "${rc_probe}"
    --artifact-ttl-hours "${artifact_ttl_hours}"
    --artifact-keep "${remote_live_handoff_keep}"
  )

  "${cmd[@]}"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true
}

action_live_risk_daemon_install_service() {
  local date_value min_conf_value min_conv_value
  date_value="${live_takeover_date}"
  min_conf_value="${live_fast_skill_min_confidence}"
  min_conv_value="${live_fast_skill_min_convexity}"
  run_live_takeover_remote \
    "live-risk-daemon-install-service" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${cloud_user}' '${live_risk_daemon_unit_name}' '${date_value}' '${live_risk_daemon_poll_seconds}' '${live_risk_daemon_guard_timeout_seconds}' '${live_risk_daemon_history_limit}' '${live_risk_guard_ticket_freshness_seconds}' '${live_risk_guard_panic_cooldown_seconds}' '${live_risk_guard_max_daily_loss_ratio}' '${live_risk_guard_max_open_exposure_ratio}' '${live_fast_skill_symbols}' '${live_fast_skill_max_age_days}' '${min_conf_value}' '${min_conv_value}' '${live_risk_daemon_security_accept_max_exposure}' <<'PY'
import argparse
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

service_user = str(sys.argv[1]).strip()
unit_name = str(sys.argv[2]).strip()
date_value = str(sys.argv[3]).strip()
poll_seconds = int(sys.argv[4])
guard_timeout_seconds = int(sys.argv[5])
history_limit = int(sys.argv[6])
ticket_freshness_seconds = int(sys.argv[7])
panic_cooldown_seconds = int(sys.argv[8])
max_daily_loss_ratio = float(sys.argv[9])
max_open_exposure_ratio = float(sys.argv[10])
ticket_symbols = str(sys.argv[11]).strip()
ticket_max_age_days = int(sys.argv[12])
min_conf_value = str(sys.argv[13]).strip()
min_conv_value = str(sys.argv[14]).strip()
max_accepted_exposure = float(sys.argv[15])

wd = Path.cwd()
state_path = wd / 'output' / 'state' / 'live_risk_daemon.json'
unit_path = Path('/etc/systemd/system') / unit_name
install_started_monotonic = time.time()

if state_path.exists():
    try:
        existing = json.loads(state_path.read_text(encoding='utf-8'))
    except Exception:
        existing = {}
    pid = int(existing.get('pid', 0) or 0)
    running = bool(existing.get('running', False))
    if running and pid > 0:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        time.sleep(0.2)

renderer_path = wd / 'scripts' / 'render_live_risk_daemon_systemd_unit.py'
spec = importlib.util.spec_from_file_location('render_live_risk_daemon_systemd_unit', renderer_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f'unable to load renderer: {renderer_path}')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
args = argparse.Namespace(
    project_dir=str(wd),
    user=service_user,
    date=date_value,
    poll_seconds=poll_seconds,
    guard_timeout_seconds=guard_timeout_seconds,
    history_limit=history_limit,
    ticket_freshness_seconds=ticket_freshness_seconds,
    panic_cooldown_seconds=panic_cooldown_seconds,
    max_daily_loss_ratio=max_daily_loss_ratio,
    max_open_exposure_ratio=max_open_exposure_ratio,
    ticket_symbols=ticket_symbols,
    ticket_equity_usdt=0.0,
    ticket_min_confidence=float(min_conf_value) if min_conf_value else None,
    ticket_min_convexity=float(min_conv_value) if min_conv_value else None,
    ticket_max_age_days=ticket_max_age_days,
)
unit_path.write_text(mod.render_unit(args), encoding='utf-8')
subprocess.run(['systemctl', 'daemon-reload'], check=True)
subprocess.run(['systemctl', 'enable', '--now', unit_name], check=True)

show = subprocess.run(
    ['systemctl', 'show', unit_name, '-p', 'ActiveState', '-p', 'SubState', '-p', 'UnitFileState', '-p', 'MainPID', '-p', 'FragmentPath'],
    text=True,
    capture_output=True,
    check=False,
)
show_map = {}
for line in (show.stdout or '').splitlines():
    if '=' not in line:
        continue
    k, v = line.split('=', 1)
    show_map[k] = v

systemd_main_pid = int(show_map.get('MainPID', '0') or 0)
systemd_active_state = str(show_map.get('ActiveState', '') or '')

def load_payload():
    if not state_path.exists():
        return {}
    try:
        loaded = json.loads(state_path.read_text(encoding='utf-8'))
        return loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        return {'status': 'invalid_state', 'error': str(exc)}


def evaluate_payload_alignment(payload):
    payload = payload if isinstance(payload, dict) else {}
    payload_pid = int(payload.get('pid', 0) or 0)
    payload_running = bool(payload.get('running', False))
    payload_pid_alive = False
    if payload_pid > 0:
        try:
            os.kill(payload_pid, 0)
            payload_pid_alive = True
        except OSError:
            payload_pid_alive = False
    payload_age_seconds = None
    if state_path.exists():
        try:
            payload_age_seconds = max(0.0, time.time() - float(state_path.stat().st_mtime))
        except OSError:
            payload_age_seconds = None
    reasons = []
    if not state_path.exists():
        reasons.append('state_missing')
    if systemd_active_state == 'active' and systemd_main_pid <= 0:
        reasons.append('systemd_main_pid_missing')
    if systemd_active_state == 'active' and payload_pid != systemd_main_pid:
        reasons.append(f'payload_pid_mismatch(payload={payload_pid},systemd={systemd_main_pid})')
    if systemd_active_state == 'active' and not payload_running:
        reasons.append('payload_not_running')
    if systemd_active_state == 'active' and not payload_pid_alive:
        reasons.append('payload_pid_not_alive')
    if systemd_active_state == 'active' and state_path.exists():
        try:
            if float(state_path.stat().st_mtime) < install_started_monotonic:
                reasons.append('payload_not_refreshed_after_install')
        except OSError:
            pass
    return {
        'aligned': not reasons,
        'systemd_active_state': systemd_active_state,
        'systemd_main_pid': systemd_main_pid,
        'payload_pid': payload_pid,
        'payload_running': payload_running,
        'payload_pid_alive': payload_pid_alive,
        'payload_updated_at_utc': str(payload.get('updated_at_utc') or ''),
        'payload_age_seconds': payload_age_seconds,
        'reasons': reasons,
    }

payload = load_payload()
payload_alignment = evaluate_payload_alignment(payload)
alignment_wait = {
    'attempted': bool(systemd_active_state == 'active' and systemd_main_pid > 0),
    'deadline_seconds': 12.0,
    'poll_interval_seconds': 0.5,
    'iterations': 0,
}
if alignment_wait['attempted'] and not payload_alignment['aligned']:
    deadline = time.time() + float(alignment_wait['deadline_seconds'])
    while time.time() <= deadline:
        alignment_wait['iterations'] += 1
        time.sleep(float(alignment_wait['poll_interval_seconds']))
        payload = load_payload()
        payload_alignment = evaluate_payload_alignment(payload)
        if payload_alignment['aligned']:
            break

verify_proc = subprocess.run(
    ['systemd-analyze', 'verify', str(unit_path)],
    text=True,
    capture_output=True,
    check=False,
)
security_proc = subprocess.run(
    ['systemd-analyze', 'security', unit_name],
    text=True,
    capture_output=True,
    check=False,
)

def tail_lines(text: str, limit: int = 12):
    return [x.rstrip() for x in str(text or '').splitlines() if x.strip()][-limit:]

security_lines = [x.rstrip() for x in (security_proc.stdout or '').splitlines() if x.strip()]
summary_line = ''
overall_exposure = None
overall_rating = None
for line in reversed(security_lines):
    if 'Overall exposure level for ' not in line:
        continue
    summary_line = line.strip()
    match = re.search(r':\\s*([0-9]+(?:\\.[0-9]+)?)\\s+([A-Za-z]+)', summary_line)
    if match:
        try:
            overall_exposure = float(match.group(1))
        except ValueError:
            overall_exposure = None
        overall_rating = match.group(2)
    break

security_acceptance_reasons = []
if verify_proc.returncode != 0:
    security_acceptance_status = 'failed'
    security_acceptance_reasons.append('systemd_verify_failed')
elif security_proc.returncode != 0:
    security_acceptance_status = 'review'
    security_acceptance_reasons.append('systemd_security_probe_failed')
elif overall_exposure is None:
    security_acceptance_status = 'review'
    security_acceptance_reasons.append('security_exposure_missing')
elif overall_exposure > max_accepted_exposure:
    security_acceptance_status = 'review'
    security_acceptance_reasons.append(
        f'security_exposure_above_threshold({overall_exposure:.1f}>{max_accepted_exposure:.1f})'
    )
else:
    security_acceptance_status = 'accepted'

out = {
    'action': 'live-risk-daemon-install-service',
    'installed': True,
    'unit_name': unit_name,
    'unit_path': str(unit_path),
    'systemd': {
        'active_state': show_map.get('ActiveState', ''),
        'sub_state': show_map.get('SubState', ''),
        'unit_file_state': show_map.get('UnitFileState', ''),
        'main_pid': int(show_map.get('MainPID', '0') or 0),
        'fragment_path': show_map.get('FragmentPath', str(unit_path)),
    },
    'payload_alignment': payload_alignment,
    'payload_alignment_wait': alignment_wait,
    'verify': {
        'ok': verify_proc.returncode == 0,
        'returncode': int(verify_proc.returncode),
        'target': str(unit_path),
        'stdout_tail': tail_lines(verify_proc.stdout),
        'stderr_tail': tail_lines(verify_proc.stderr),
    },
    'security': {
        'returncode': int(security_proc.returncode),
        'overall_exposure': overall_exposure,
        'overall_rating': overall_rating,
        'summary_line': summary_line,
        'findings': [line.strip() for line in security_lines if line.lstrip().startswith('✗')][:10],
        'stdout_tail': tail_lines(security_proc.stdout),
        'stderr_tail': tail_lines(security_proc.stderr),
    },
    'security_acceptance': {
        'status': security_acceptance_status,
        'max_allowed_exposure': max_accepted_exposure,
        'observed_exposure': overall_exposure,
        'observed_rating': overall_rating,
        'reasons': security_acceptance_reasons,
    },
    'payload': payload,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

action_live_risk_daemon_remove_service() {
  run_live_takeover_remote \
    "live-risk-daemon-remove-service" \
    "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; sudo -n /usr/bin/python3 - '${live_risk_daemon_unit_name}' <<'PY'
import json
import subprocess
import sys
from pathlib import Path

unit_name = str(sys.argv[1]).strip()
unit_path = Path('/etc/systemd/system') / unit_name
state_path = Path('output/state/live_risk_daemon.json')
installed = unit_path.exists()
if installed:
    subprocess.run(['systemctl', 'disable', '--now', unit_name], text=True, capture_output=True, check=False)
    try:
        unit_path.unlink()
    except FileNotFoundError:
        pass
    subprocess.run(['systemctl', 'daemon-reload'], check=True)

payload = {}
if state_path.exists():
    try:
        loaded = json.loads(state_path.read_text(encoding='utf-8'))
        payload = loaded if isinstance(loaded, dict) else {}
    except Exception as exc:
        payload = {'status': 'invalid_state', 'error': str(exc)}

out = {
    'action': 'live-risk-daemon-remove-service',
    'removed': bool(installed),
    'unit_name': unit_name,
    'unit_path': str(unit_path),
    'payload': payload,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY"
}

action_live_takeover_probe() {
  local date_arg daemon_env_arg cred_env_arg guarded_stdout
  date_arg=""
  daemon_env_arg=""
  cred_env_arg="$(build_live_takeover_cred_env_arg)"
  if [[ -n "${live_takeover_date}" ]]; then
    date_arg="--date ${live_takeover_date}"
  fi
  if is_true "${live_takeover_allow_daemon_env_fallback}"; then
    daemon_env_arg="--allow-daemon-env-fallback"
  fi
  guarded_stdout="$(run_guarded_exec_remote "probe" "false" "${cred_env_arg}" "${daemon_env_arg}" "${date_arg}")"
  python3 - "${guarded_stdout}" <<'PY'
import json
import sys

def parse_payload(text: str):
    clean = str(text or "").strip()
    if not clean:
        return {}
    try:
        payload = json.loads(clean)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    for line in reversed([x.strip() for x in clean.splitlines() if x.strip()]):
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {"raw": clean}

guarded = parse_payload(str(sys.argv[1] if len(sys.argv) > 1 else ""))
out = {
    "action": "live-takeover-probe",
    "executed": bool(guarded.get("executed", False)),
    "status": str(guarded.get("status", "unknown")),
    "guarded_exec": guarded,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
}

action_live_takeover_canary() {
  local date_arg daemon_env_arg cred_env_arg idem_material idem_json rc_idem guarded_stdout ready_json rc_ready tmp_ready
  date_arg=""
  daemon_env_arg=""
  cred_env_arg="$(build_live_takeover_cred_env_arg)"
  if [[ -n "${live_takeover_date}" ]]; then
    date_arg="--date ${live_takeover_date}"
  fi
  if is_true "${live_takeover_allow_daemon_env_fallback}"; then
    daemon_env_arg="--allow-daemon-env-fallback"
  fi

  tmp_ready="$(mktemp)"
  set +e
  action_live_takeover_ready_check >"${tmp_ready}" 2>/dev/null
  rc_ready=$?
  set -e
  ready_json="$(cat "${tmp_ready}")"
  rm -f "${tmp_ready}" >/dev/null 2>&1 || true

  if (( rc_ready == 3 )); then
    python3 - "${ready_json}" <<'PY'
import json
import sys

text = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
try:
    payload = json.loads(text) if text else {}
except Exception:
    payload = {}
out = {
    "executed": False,
    "ready": bool(payload.get("ready", False)),
    "reason": str(payload.get("reason", "not_ready")),
    "reasons": payload.get("reasons", []),
    "market": str(payload.get("market", "unknown")),
    "quote_available": float(payload.get("quote_available", 0.0) or 0.0),
    "required_quote": float(payload.get("required_quote", 0.0) or 0.0),
    "action": "live-takeover-canary",
    "status": "skipped_not_ready",
    "probe_returncode": int(payload.get("probe_returncode", 3) or 3),
    "ops_reconcile": payload.get("ops_reconcile", {}),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 0
  fi
  if (( rc_ready != 0 )); then
    echo "FUSE: live-takeover-ready-check failed unexpectedly before canary (rc=${rc_ready})." >&2
    if [[ -n "${ready_json}" ]]; then
      echo "${ready_json}" >&2
    fi
    return 2
  fi

  idem_material="$(build_live_takeover_idempotency_material)"
  set +e
  idem_json="$(idempotency_guard "live-takeover-canary" "${idem_material}")"
  rc_idem=$?
  set -e
  if (( rc_idem == 3 )); then
    echo "${idem_json}"
    echo "idempotency gate: skipped duplicate live-takeover-canary within ttl=${idempotency_ttl_seconds}s."
    return 0
  fi
  if (( rc_idem != 0 )); then
    echo "ERROR: idempotency guard failed for live-takeover-canary (rc=${rc_idem})." >&2
    if [[ -n "${idem_json}" ]]; then
      echo "${idem_json}" >&2
    fi
    return "${rc_idem}"
  fi

  guarded_stdout="$(run_guarded_exec_remote "canary" "true" "${cred_env_arg}" "${daemon_env_arg}" "${date_arg}")"
  python3 - "${guarded_stdout}" <<'PY'
import json
import sys

def parse_payload(text: str):
    clean = str(text or "").strip()
    if not clean:
        return {}
    try:
        payload = json.loads(clean)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    for line in reversed([x.strip() for x in clean.splitlines() if x.strip()]):
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {"raw": clean}

guarded = parse_payload(str(sys.argv[1] if len(sys.argv) > 1 else ""))
status = str(guarded.get("status", "unknown"))
out = {
    "action": "live-takeover-canary",
    "executed": bool(guarded.get("executed", False)),
    "status": "skipped_risk_blocked" if status == "downgraded_probe_risk_guard_blocked" else status,
    "guarded_exec": guarded,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
}

action_live_takeover_ready_check() {
  local date_arg daemon_env_arg cred_env_arg probe_json ops_json tmp_json rc_probe rc_ops rc_ready_payload tmp_probe tmp_ops
  date_arg=""
  daemon_env_arg=""
  cred_env_arg="$(build_live_takeover_cred_env_arg)"
  if [[ -n "${live_takeover_date}" ]]; then
    date_arg="--date ${live_takeover_date}"
  fi
  if is_true "${live_takeover_allow_daemon_env_fallback}"; then
    daemon_env_arg="--allow-daemon-env-fallback"
  fi

  tmp_probe="$(mktemp)"
  set +e
  run_guarded_exec_remote "probe" "false" "${cred_env_arg}" "${daemon_env_arg}" "${date_arg}" >"${tmp_probe}" 2>/dev/null
  rc_probe=$?
  set -e
  probe_json="$(cat "${tmp_probe}")"
  rm -f "${tmp_probe}" >/dev/null 2>&1 || true

  tmp_ops="$(mktemp)"
  set +e
  run_live_ops_reconcile_status_remote >"${tmp_ops}" 2>/dev/null
  rc_ops=$?
  set -e
  ops_json="$(cat "${tmp_ops}")"
  rm -f "${tmp_ops}" >/dev/null 2>&1 || true

  tmp_json="$(mktemp)"
  set +e
  python3 - "${rc_probe}" "${rc_ops}" "${probe_json}" "${ops_json}" >"${tmp_json}" <<'PY'
import json
import sys

rc_probe = int(sys.argv[1] if len(sys.argv) > 1 else "0")
rc_ops = int(sys.argv[2] if len(sys.argv) > 2 else "0")
probe_text = str(sys.argv[3] if len(sys.argv) > 3 else "")
ops_text = str(sys.argv[4] if len(sys.argv) > 4 else "")

def parse_payload(text: str):
    clean = str(text or "").strip()
    if not clean:
        return {}
    try:
        payload = json.loads(clean)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    for line in reversed([x.strip() for x in clean.splitlines() if x.strip()]):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}

payload = parse_payload(probe_text)
ops_payload = parse_payload(ops_text)
if not payload:
    print(json.dumps({"ready": False, "reason": "invalid_probe_json", "probe_returncode": rc_probe, "ops_returncode": rc_ops}, ensure_ascii=False))
    raise SystemExit(3)

guarded = payload if isinstance(payload, dict) else {}
risk_block = guarded.get("risk_guard", {}) if isinstance(guarded.get("risk_guard", {}), dict) else {}
risk = risk_block.get("payload", {}) if isinstance(risk_block.get("payload", {}), dict) else {}
backup_intel = risk.get("backup_intel", {}) if isinstance(risk.get("backup_intel", {}), dict) else {}
takeover = guarded.get("takeover", {}) if isinstance(guarded.get("takeover", {}), dict) else {}
takeover_payload = takeover.get("payload", {}) if isinstance(takeover.get("payload", {}), dict) else {}
steps = takeover_payload.get("steps", {}) if isinstance(takeover_payload.get("steps", {}), dict) else {}
creds = steps.get("credentials", {}) if isinstance(steps.get("credentials", {}), dict) else {}
plan = steps.get("canary_plan", {}) if isinstance(steps.get("canary_plan", {}), dict) else {}
acct = steps.get("account_overview", {}) if isinstance(steps.get("account_overview", {}), dict) else {}

def _f(v, d=0.0):
    try:
        return float(v)
    except Exception:
        return float(d)

market = str(takeover_payload.get("market", "")).strip().lower()
has_key = bool(creds.get("has_api_key", False))
has_secret = bool(creds.get("has_api_secret", False))
reason = "ready"
reasons = []
ready = bool(has_key and has_secret)

available = _f(acct.get("quote_available", 0.0), 0.0)
required = _f(plan.get("effective_quote_usdt", plan.get("quote_usdt", 0.0)), 0.0)

if not ready:
    reasons.append("missing_api_credentials")
elif market == "spot" and available + 1e-12 < required:
    reasons.append("insufficient_quote_balance")
elif market == "futures_usdm" and available <= 0.0:
    reasons.append("insufficient_futures_balance")

if int(risk_block.get("returncode", 0)) == 3:
    reasons.append("risk_guard_blocked")
if not isinstance(ops_payload, dict) or not ops_payload:
    reasons.append("ops_reconcile_invalid")
elif not bool(ops_payload.get("ok", False)):
    reasons.append(str(ops_payload.get("reason_code", "ops_reconcile_blocked") or "ops_reconcile_blocked"))
else:
    live_gate = ops_payload.get("live_gate", {}) if isinstance(ops_payload.get("live_gate", {}), dict) else {}
    if live_gate:
        if not bool(live_gate.get("ok", False)):
            reasons.append("ops_live_gate_blocked")
    else:
        if ops_payload.get("gate_passed") is not True:
            reasons.append("ops_gate_failed")
        if str(ops_payload.get("ops_status", "")).strip().lower() == "red":
            reasons.append("ops_status_red")

ready = len(reasons) == 0
reason = reasons[0] if reasons else "ready"
ops_gate = {
    "ok": bool(ops_payload.get("ok", False)) if isinstance(ops_payload, dict) else False,
    "gate_passed": ops_payload.get("gate_passed") if isinstance(ops_payload, dict) else None,
    "ops_status": str(ops_payload.get("ops_status", "")) if isinstance(ops_payload, dict) else "",
    "gate_failed_checks": (
        [str(x) for x in ops_payload.get("gate_failed_checks", [])]
        if isinstance(ops_payload.get("gate_failed_checks", []), list)
        else []
    ) if isinstance(ops_payload, dict) else [],
}
ops_live_gate = ops_payload.get("live_gate", {}) if isinstance(ops_payload.get("live_gate", {}), dict) else {}

out = {
    "ready": bool(ready),
    "reason": str(reason),
    "reasons": reasons,
    "market": market or "unknown",
    "probe_returncode": int(rc_probe),
    "has_api_key": bool(has_key),
    "has_api_secret": bool(has_secret),
    "quote_available": float(available),
    "required_quote": float(required),
    "artifact": str(guarded.get("artifact", "")),
    "takeover_artifact": str(takeover_payload.get("artifact", "")),
    "guarded_exec": guarded,
    "risk_guard": risk,
    "backup_intel": backup_intel,
    "ops_gate": ops_gate,
    "ops_live_gate": ops_live_gate,
    "ops_reconcile": ops_payload,
    "ops_returncode": int(rc_ops),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
raise SystemExit(0 if ready else 3)
PY
  rc_ready_payload=$?
  cat "${tmp_json}"
  rm -f "${tmp_json}" >/dev/null 2>&1 || true
  return "${rc_ready_payload}"
}

action_live_takeover_autopilot() {
  local ready_json rc_ready idem_material idem_json rc_idem tmp_ready

  tmp_ready="$(mktemp)"
  set +e
  action_live_takeover_ready_check >"${tmp_ready}" 2>/dev/null
  rc_ready=$?
  set -e
  ready_json="$(cat "${tmp_ready}")"
  rm -f "${tmp_ready}" >/dev/null 2>&1 || true

  if (( rc_ready == 0 )); then
    idem_material="$(build_live_takeover_idempotency_material)"
    set +e
    idem_json="$(idempotency_guard "live-takeover-autopilot" "${idem_material}")"
    rc_idem=$?
    set -e
    if (( rc_idem == 3 )); then
      echo "${idem_json}"
      echo "idempotency gate: skipped duplicate live-takeover-autopilot within ttl=${idempotency_ttl_seconds}s."
      return 0
    fi
    if (( rc_idem != 0 )); then
      echo "ERROR: idempotency guard failed for live-takeover-autopilot (rc=${rc_idem})." >&2
      if [[ -n "${idem_json}" ]]; then
        echo "${idem_json}" >&2
      fi
      return "${rc_idem}"
    fi
    echo "${ready_json}"
    action_live_takeover_canary
    return $?
  fi
  if (( rc_ready == 3 )); then
    python3 - "${ready_json}" <<'PY'
import json
import sys

text = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
try:
    payload = json.loads(text) if text else {}
except Exception:
    payload = {}
out = {
    "executed": False,
    "ready": bool(payload.get("ready", False)),
    "reason": str(payload.get("reason", "not_ready")),
    "reasons": payload.get("reasons", []),
    "market": str(payload.get("market", "unknown")),
    "quote_available": float(payload.get("quote_available", 0.0) or 0.0),
    "required_quote": float(payload.get("required_quote", 0.0) or 0.0),
    "action": "live-takeover-autopilot",
    "status": "skipped_not_ready",
    "probe_returncode": int(payload.get("probe_returncode", 3) or 3),
    "ops_reconcile": payload.get("ops_reconcile", {}),
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 0
  fi
  echo "FUSE: live-takeover-ready-check failed unexpectedly (rc=${rc_ready})." >&2
  if [[ -n "${ready_json}" ]]; then
    echo "${ready_json}" >&2
  fi
  return 2
}

action_live_fast_skill() {
  local idem_material idem_json rc_idem tmp_ready ready_json rc_ready
  tmp_ready="$(mktemp)"
  set +e
  action_live_takeover_ready_check >"${tmp_ready}" 2>/dev/null
  rc_ready=$?
  set -e
  ready_json="$(cat "${tmp_ready}")"
  rm -f "${tmp_ready}" >/dev/null 2>&1 || true

  if (( rc_ready == 3 )); then
    python3 - "${ready_json}" <<'PY'
import json
import sys

text = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
try:
    ready = json.loads(text) if text else {}
except Exception:
    ready = {"ready": False, "reason": "invalid_ready_payload"}
out = {
    "action": "live-fast-skill",
    "executed": False,
    "status": "skipped_not_ready",
    "ready_check": ready,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
    return 0
  fi
  if (( rc_ready != 0 )); then
    echo "FUSE: live-fast-skill ready-check failed unexpectedly (rc=${rc_ready})." >&2
    if [[ -n "${ready_json}" ]]; then
      echo "${ready_json}" >&2
    fi
    return 2
  fi

  idem_material="$(build_live_takeover_idempotency_material)|fast_skill|symbols=${live_fast_skill_symbols}|max_age=${live_fast_skill_max_age_days}|min_conf=${live_fast_skill_min_confidence:-config}|min_conv=${live_fast_skill_min_convexity:-config}"
  set +e
  idem_json="$(idempotency_guard "live-fast-skill" "${idem_material}")"
  rc_idem=$?
  set -e
  if (( rc_idem == 3 )); then
    echo "${idem_json}"
    echo "idempotency gate: skipped duplicate live-fast-skill within ttl=${idempotency_ttl_seconds}s."
    return 0
  fi
  if (( rc_idem != 0 )); then
    echo "ERROR: idempotency guard failed for live-fast-skill (rc=${rc_idem})." >&2
    if [[ -n "${idem_json}" ]]; then
      echo "${idem_json}" >&2
    fi
    return "${rc_idem}"
  fi

  local quote_available
  quote_available="$(python3 - "${ready_json}" <<'PY'
import json
import sys

text = str(sys.argv[1] if len(sys.argv) > 1 else "").strip()
try:
    payload = json.loads(text) if text else {}
except Exception:
    payload = {}
try:
    print(float(payload.get("quote_available", 0.0) or 0.0))
except Exception:
    print(0.0)
PY
)"

  local date_arg daemon_env_arg cred_env_arg
  date_arg=""
  daemon_env_arg=""
  cred_env_arg="$(build_live_takeover_cred_env_arg)"
  if [[ -n "${live_takeover_date}" ]]; then
    date_arg="--date ${live_takeover_date}"
  fi
  if is_true "${live_takeover_allow_daemon_env_fallback}"; then
    daemon_env_arg="--allow-daemon-env-fallback"
  fi
  local min_conf_arg min_conv_arg auto_unwind_arg pipeline_stdout
  min_conf_arg=""
  min_conv_arg=""
  auto_unwind_arg=""
  if [[ -n "${live_fast_skill_min_confidence}" ]]; then
    min_conf_arg="--live-min-confidence ${live_fast_skill_min_confidence}"
  fi
  if [[ -n "${live_fast_skill_min_convexity}" ]]; then
    min_conv_arg="--live-min-convexity ${live_fast_skill_min_convexity}"
  fi
  if is_true "${live_fast_skill_auto_close}"; then
    auto_unwind_arg="--auto-unwind-live-canary"
  fi
  if (( live_fast_skill_close_delay_seconds > 0 )); then
    echo "WARN: LIVE_FAST_SKILL_CLOSE_DELAY_SECONDS is ignored after wrapper refactor; auto unwind now executes immediately." >&2
  fi

  pipeline_stdout="$(
    run_live_takeover_remote \
      "live-fast-skill" \
      "set -e; wd=\$(${remote_workdir_expr}); cd \"\$wd\"; ${cred_env_arg}PYTHONPATH=src python3 scripts/run_ticketed_canary.py ${date_arg} --symbols ${live_fast_skill_symbols} --market ${live_takeover_market} --equity-usdt ${quote_available} --canary-quote-usdt ${live_takeover_canary_usdt} --use-selected-quote ${min_conf_arg} ${min_conv_arg} --live-max-age-days ${live_fast_skill_max_age_days} --decision-ttl-seconds ${live_fast_skill_decision_ttl_seconds} --run-live-canary ${auto_unwind_arg} --rate-limit-per-minute ${live_takeover_rate_limit_per_minute} --timeout-ms ${live_takeover_timeout_ms} --max-drawdown ${live_takeover_max_drawdown} --trade-window-hours ${live_takeover_trade_window_hours} --risk-fuse-max-age-seconds ${live_risk_guard_ticket_freshness_seconds} --panic-cooldown-seconds ${live_risk_guard_panic_cooldown_seconds} --max-daily-loss-ratio ${live_risk_guard_max_daily_loss_ratio} --max-open-exposure-ratio ${live_risk_guard_max_open_exposure_ratio} ${daemon_env_arg}"
  )"

  python3 - "${ready_json}" "${pipeline_stdout}" "${live_fast_skill_auto_close}" <<'PY'
import json
import sys

def parse_json_lines(text: str):
    clean = str(text or "").strip()
    if not clean:
        return {}
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    for line in reversed([x.strip() for x in clean.splitlines() if x.strip()]):
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {"raw": clean}

ready = parse_json_lines(str(sys.argv[1] if len(sys.argv) > 1 else ""))
pipeline = parse_json_lines(str(sys.argv[2] if len(sys.argv) > 2 else ""))
auto_close_raw = str(sys.argv[3] if len(sys.argv) > 3 else "true").strip().lower()
auto_close_enabled = auto_close_raw in {"1", "true", "yes", "y", "on"}

status = str(pipeline.get("status", "unknown"))
guarded = pipeline.get("guarded_exec", {}) if isinstance(pipeline.get("guarded_exec", {}), dict) else {}
guarded_payload = guarded.get("payload", {}) if isinstance(guarded.get("payload", {}), dict) else {}

out = {
    "action": "live-fast-skill",
    "executed": bool(status not in {"skipped_no_executable_candidate", "error", "panic", ""}),
    "status": status,
    "ready_check": ready,
    "pipeline": pipeline,
    "guarded_exec": guarded_payload,
    "auto_close": {
        "enabled": bool(auto_close_enabled),
        "status": "attempted" if bool((pipeline.get("auto_unwind", {}) if isinstance(pipeline.get("auto_unwind", {}), dict) else {}).get("attempted", False)) else ("disabled" if not auto_close_enabled else "skipped_no_open_fill"),
        "result": (pipeline.get("auto_unwind", {}) if isinstance(pipeline.get("auto_unwind", {}), dict) else {}),
    },
}
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
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
    assert-whitelist-gate) action_assert_whitelist_gate ;;
    ensure-whitelist-gate) action_ensure_whitelist_gate ;;
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
    live-takeover-probe) action_live_takeover_probe ;;
    live-takeover-canary) action_live_takeover_canary ;;
    live-takeover-ready-check) action_live_takeover_ready_check ;;
    live-takeover-autopilot) action_live_takeover_autopilot ;;
    live-risk-guard) action_live_risk_guard ;;
    live-risk-daemon-start) action_live_risk_daemon_start ;;
    live-risk-daemon-status) action_live_risk_daemon_status ;;
    live-risk-daemon-stop) action_live_risk_daemon_stop ;;
  live-risk-daemon-install-service) action_live_risk_daemon_install_service ;;
  live-risk-daemon-service-status) action_live_risk_daemon_service_status ;;
  live-risk-daemon-mdwe-probe) action_live_risk_daemon_mdwe_probe ;;
  live-risk-daemon-protecthome-probe) action_live_risk_daemon_protecthome_probe ;;
  live-risk-daemon-procsubset-probe) action_live_risk_daemon_procsubset_probe ;;
  live-risk-daemon-privateusers-probe) action_live_risk_daemon_privateusers_probe ;;
  live-risk-daemon-privatenetwork-probe) action_live_risk_daemon_privatenetwork_probe ;;
  live-risk-daemon-ipdeny-probe) action_live_risk_daemon_ipdeny_probe ;;
  live-risk-daemon-devicepolicy-probe) action_live_risk_daemon_devicepolicy_probe ;;
  live-risk-daemon-afunix-probe) action_live_risk_daemon_afunix_probe ;;
  live-risk-daemon-noaf-probe) action_live_risk_daemon_noaf_probe ;;
  live-risk-daemon-syscallfilter-probe) action_live_risk_daemon_syscallfilter_probe ;;
  live-risk-daemon-syscallfilter-tight-probe) action_live_risk_daemon_syscallfilter_tight_probe ;;
  live-risk-daemon-security-status) action_live_risk_daemon_security_status ;;
  live-risk-daemon-journal) action_live_risk_daemon_journal ;;
  live-risk-daemon-remove-service) action_live_risk_daemon_remove_service ;;
  live-ops-reconcile-status) action_live_ops_reconcile_status ;;
  live-ops-reconcile-refresh) action_live_ops_reconcile_refresh ;;
  bootstrap-remote-runtime) action_bootstrap_remote_runtime ;;
  sync-local-pi-workspace) action_sync_local_pi_workspace ;;
  publish-local-pi-runtime-scripts) action_publish_local_pi_runtime_scripts ;;
  prepare-local-pi-runtime) action_prepare_local_pi_runtime ;;
  smoke-local-pi-cycle) action_smoke_local_pi_cycle ;;
  run-local-pi-recovery-lab) action_run_local_pi_recovery_lab ;;
  snapshot-local-pi-recovery-state) action_snapshot_local_pi_recovery_state ;;
  restore-local-pi-recovery-state) action_restore_local_pi_recovery_state ;;
  rollback-local-pi-recovery-state) action_rollback_local_pi_recovery_state ;;
  backfill-local-pi-last-loss-ts) action_backfill_local_pi_last_loss_ts ;;
  local-pi-consecutive-loss-guardrail-status) action_local_pi_consecutive_loss_guardrail_status ;;
  local-pi-ack-archive-status) action_local_pi_ack_archive_status ;;
  local-pi-recovery-handoff) action_local_pi_recovery_handoff ;;
  remote-live-handoff) action_remote_live_handoff ;;
  remote-live-notification-preview) action_remote_live_notification_preview ;;
  remote-live-notification-dry-run) action_remote_live_notification_dry_run ;;
  remote-live-notification-send) action_remote_live_notification_send ;;
  apply-local-pi-recovery-step) action_apply_local_pi_recovery_step ;;
  run-local-pi-recovery-flow) action_run_local_pi_recovery_flow ;;
  ack-local-pi-consecutive-loss-guardrail) action_ack_local_pi_consecutive_loss_guardrail ;;
  ensure-local-openclaw-runtime-model) action_ensure_local_openclaw_runtime_model ;;
  ensure-remote-openclaw-runtime-model) action_ensure_remote_openclaw_runtime_model ;;
  live-fast-skill) action_live_fast_skill ;;
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

action_assert_whitelist_gate() {
  local latest_json
  latest_json="$(ls -1t "${output_dir}"/*_openclaw_bridge_whitelist_24h.json 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest_json}" || ! -f "${latest_json}" ]]; then
    echo "FUSE: missing whitelist artifact under ${output_dir}" >&2
    return 3
  fi

  python3 - "$latest_json" "${whitelist_assert_max_age_minutes}" "${whitelist_min_total_success_rate}" "${whitelist_min_action_success_rate}" "${whitelist_min_samples_per_action}" "${whitelist_required_actions}" "${whitelist_require_last_rc_zero}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

artifact = Path(sys.argv[1])
max_age_minutes = int(sys.argv[2])
min_total_success_rate = float(sys.argv[3])
min_action_success_rate = float(sys.argv[4])
min_samples_per_action = int(sys.argv[5])
required_actions = [x.strip() for x in str(sys.argv[6]).split(",") if x.strip()]
require_last_rc_zero = str(sys.argv[7]).strip().lower() in {"1", "true", "yes", "y", "on"}

try:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"FUSE: invalid artifact json: {artifact} ({exc})", file=sys.stderr)
    raise SystemExit(3)

generated = str(payload.get("generated_at_utc", ""))
try:
    generated_dt = datetime.strptime(generated, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
except Exception:
    generated_dt = datetime.fromtimestamp(artifact.stat().st_mtime, tz=timezone.utc)
age_minutes = max(0.0, (datetime.now(timezone.utc) - generated_dt).total_seconds() / 60.0)

rows = payload.get("commands", [])
if not isinstance(rows, list):
    rows = []
row_by_action: dict[str, dict] = {}
for row in rows:
    if isinstance(row, dict):
        row_by_action[str(row.get("action", ""))] = row

total_success_rate = float(payload.get("total_success_rate", 0.0))
reasons: list[str] = []
if age_minutes > float(max_age_minutes):
    reasons.append(f"artifact_stale:{age_minutes:.1f}m>max_age({max_age_minutes}m)")
if total_success_rate < min_total_success_rate:
    reasons.append(f"total_success_rate({total_success_rate:.4f})<min_total({min_total_success_rate:.4f})")

for action in required_actions:
    row = row_by_action.get(action)
    if row is None:
        reasons.append(f"missing_action:{action}")
        continue
    samples = int(row.get("samples", 0))
    rate = float(row.get("success_rate", 0.0))
    last_rc = row.get("last_rc", None)
    if samples < min_samples_per_action:
        reasons.append(f"samples:{action}:{samples}<min({min_samples_per_action})")
    if rate < min_action_success_rate:
        reasons.append(f"success_rate:{action}:{rate:.4f}<min({min_action_success_rate:.4f})")
    if require_last_rc_zero and last_rc not in {0, None}:
        reasons.append(f"last_rc_nonzero:{action}:{last_rc}")

if reasons:
    print(
        "FUSE: whitelist assert failed; "
        + f"artifact={artifact}; generated_at_utc={generated_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}; "
        + f"age_minutes={age_minutes:.1f}; reasons={'|'.join(reasons)}",
        file=sys.stderr,
    )
    raise SystemExit(3)

print(
    "ASSERT PASS: whitelist gate healthy; "
    + f"artifact={artifact}; generated_at_utc={generated_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}; "
    + f"age_minutes={age_minutes:.1f}; total_success_rate={total_success_rate:.4f}"
)
PY
}

action_ensure_whitelist_gate() {
  set +e
  action_assert_whitelist_gate
  local rc_assert=$?
  set -e
  if (( rc_assert == 0 )); then
    echo "ENSURE PASS: whitelist assert passed without resample."
    return 0
  fi

  echo "ensure-whitelist-gate: assert failed (rc=${rc_assert}), running sample-whitelist-gate for recovery..."
  set +e
  action_sample_whitelist_gate
  local rc_sample=$?
  set -e
  if (( rc_sample != 0 )); then
    echo "FUSE: ensure-whitelist-gate failed after resample (sample_rc=${rc_sample})." >&2
    return 3
  fi

  set +e
  action_assert_whitelist_gate
  local rc_assert2=$?
  set -e
  if (( rc_assert2 != 0 )); then
    echo "FUSE: ensure-whitelist-gate assert failed after successful sample (assert_rc=${rc_assert2})." >&2
    return 3
  fi
  echo "ENSURE PASS: whitelist gate recovered via resample."
  return 0
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

  local artifact_ts artifact_json artifact_md artifact_checksum
  artifact_ts="$(now_utc_compact)"
  artifact_json="${output_dir}/${artifact_ts}_openclaw_bridge_whitelist_24h.json"
  artifact_md="${output_dir}/${artifact_ts}_openclaw_bridge_whitelist_24h.md"
  artifact_checksum="${output_dir}/${artifact_ts}_openclaw_bridge_whitelist_checksum.json"

  python3 - "$samples_log" "$sample_window_hours" "$artifact_json" "$artifact_md" "$artifact_ts" "${whitelist_min_total_success_rate}" "${whitelist_min_action_success_rate}" "${whitelist_min_samples_per_action}" "${whitelist_required_actions}" "${whitelist_require_last_rc_zero}" "$artifact_checksum" "${artifact_ttl_hours}" <<'PY'
import hashlib
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
artifact_checksum = Path(sys.argv[11])
artifact_ttl_hours = max(1.0, float(sys.argv[12]))


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest(), int(path.stat().st_size)


def _evict_old_whitelist_artifacts(*, directory: Path, now_utc: datetime, ttl_hours: float, keep_names: set[str]) -> int:
    cutoff = now_utc - timedelta(hours=max(1.0, float(ttl_hours)))
    removed = 0
    patterns = (
        "*_openclaw_bridge_whitelist_24h.json",
        "*_openclaw_bridge_whitelist_24h.md",
        "*_openclaw_bridge_whitelist_checksum.json",
    )
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.name in keep_names:
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            except Exception:
                continue
            if mtime >= cutoff:
                continue
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue
    return removed

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
    "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
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

evicted_files = _evict_old_whitelist_artifacts(
    directory=artifact_json.parent,
    now_utc=now,
    ttl_hours=artifact_ttl_hours,
    keep_names={artifact_json.name, artifact_md.name, artifact_checksum.name},
)
summary["governance"] = {
    "artifact_ttl_hours": float(artifact_ttl_hours),
    "evicted_files": int(evicted_files),
}
artifact_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

json_sha, json_size = _sha256_file(artifact_json)
md_sha, md_size = _sha256_file(artifact_md)
checksum_payload = {
    "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "artifact_ttl_hours": float(artifact_ttl_hours),
    "files": [
        {
            "path": str(artifact_json),
            "sha256": str(json_sha),
            "size_bytes": int(json_size),
        },
        {
            "path": str(artifact_md),
            "sha256": str(md_sha),
            "size_bytes": int(md_size),
        },
    ],
}
artifact_checksum.write_text(json.dumps(checksum_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(str(artifact_json))
print(str(artifact_md))
print(str(artifact_checksum))
print(f"gate_pass={str(gate_pass).lower()}")
PY

  echo "whitelist sample artifacts:"
  echo "  - ${artifact_json}"
  echo "  - ${artifact_md}"
  echo "  - ${artifact_checksum}"

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
