#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_SCRIPT="$ROOT_DIR/infra/local/run_slot.sh"
LOG_DIR="$ROOT_DIR/output/logs"
mkdir -p "$LOG_DIR"

TAG="# lie-antifragile-scheduler"

CURRENT="$(crontab -l 2>/dev/null || true)"
FILTERED="$(printf '%s\n' "$CURRENT" | sed '/lie-antifragile-scheduler/d')"

ADD=$(cat <<CRON
40 8 * * 1-5 TZ=Asia/Shanghai $RUN_SCRIPT \$(date +\%F) 08:40 >> $LOG_DIR/cron_0840.log 2>&1 $TAG
30 10 * * 1-5 TZ=Asia/Shanghai $RUN_SCRIPT \$(date +\%F) 10:30 >> $LOG_DIR/cron_1030.log 2>&1 $TAG
30 14 * * 1-5 TZ=Asia/Shanghai $RUN_SCRIPT \$(date +\%F) 14:30 >> $LOG_DIR/cron_1430.log 2>&1 $TAG
10 15 * * 1-5 TZ=Asia/Shanghai $RUN_SCRIPT \$(date +\%F) 15:10 >> $LOG_DIR/cron_1510.log 2>&1 $TAG
30 20 * * 1-5 TZ=Asia/Shanghai $RUN_SCRIPT \$(date +\%F) 20:30 >> $LOG_DIR/cron_2030.log 2>&1 $TAG
45 20 * * 1-5 TZ=Asia/Shanghai $RUN_SCRIPT \$(date +\%F) ops >> $LOG_DIR/cron_2045_ops.log 2>&1 $TAG
CRON
)

printf '%s\n%s\n' "$FILTERED" "$ADD" | crontab -
echo "Installed cron schedules for Liè Antifragile system."
