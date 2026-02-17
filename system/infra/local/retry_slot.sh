#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATE_ARG="${1:-$(date +%F)}"
SLOT_ARG="${2:-15:10}"
MAX_RETRY="${MAX_RETRY:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"

attempt=1
while [ "$attempt" -le "$MAX_RETRY" ]; do
  if lie --config "$ROOT_DIR/config.yaml" run-slot --date "$DATE_ARG" --slot "$SLOT_ARG"; then
    echo "slot $SLOT_ARG succeeded on attempt $attempt"
    exit 0
  fi
  echo "slot $SLOT_ARG failed on attempt $attempt; retrying in $WAIT_SECONDS sec"
  sleep "$WAIT_SECONDS"
  attempt=$((attempt+1))
done

echo "slot $SLOT_ARG failed after $MAX_RETRY attempts"
exit 1
