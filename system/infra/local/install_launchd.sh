#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TEMPLATE="$ROOT_DIR/infra/local/launchd/com.lie.antifragile.daemon.plist.template"
AGENT_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$AGENT_DIR/com.lie.antifragile.daemon.plist"

mkdir -p "$AGENT_DIR" "$ROOT_DIR/output/logs"

python3 - "$TEMPLATE" "$PLIST_PATH" "$ROOT_DIR" <<'PY'
from pathlib import Path
import sys

template_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
root_dir = sys.argv[3]
text = template_path.read_text(encoding="utf-8")
out_path.write_text(text.replace("__ROOT_DIR__", root_dir), encoding="utf-8")
PY

launchctl unload "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl load "$PLIST_PATH"

echo "Installed launchd agent: com.lie.antifragile.daemon"
echo "Plist: $PLIST_PATH"
echo "Logs: $ROOT_DIR/output/logs/launchd_daemon.out.log"
