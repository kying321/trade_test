#!/usr/bin/env bash
set -euo pipefail

CURRENT="$(crontab -l 2>/dev/null || true)"
printf '%s\n' "$CURRENT" | sed '/lie-antifragile-scheduler/d' | crontab -
echo "Removed Liè Antifragile cron schedules."
