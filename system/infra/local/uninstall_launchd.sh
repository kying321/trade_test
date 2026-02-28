#!/usr/bin/env bash
set -euo pipefail

PLIST_PATH="$HOME/Library/LaunchAgents/com.lie.antifragile.daemon.plist"

if [[ -f "$PLIST_PATH" ]]; then
  launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
  rm -f "$PLIST_PATH"
  echo "Removed launchd agent: com.lie.antifragile.daemon"
else
  echo "launchd agent not found: $PLIST_PATH"
fi
