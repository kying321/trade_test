#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATE_ARG="${1:-$(date +%F)}"
AUDIT_SOURCE="${AUDIT_SOURCE:-local-healthcheck}"
AUDIT_TAG="${AUDIT_TAG:-health-check}"

python3 "$ROOT_DIR/scripts/exec_with_audit.py" \
  --root "$ROOT_DIR" \
  --source "$AUDIT_SOURCE" \
  --tag "$AUDIT_TAG" \
  -- PYTHONPATH="$ROOT_DIR/src" python3 -m lie_engine.cli --config "$ROOT_DIR/config.yaml" health-check --date "$DATE_ARG"
