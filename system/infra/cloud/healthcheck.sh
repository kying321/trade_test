#!/usr/bin/env sh
set -eu

ROOT_DIR="/app"
DATE_ARG="${DATE:-$(date +%F)}"
AUDIT_SOURCE="${AUDIT_SOURCE:-cloud-healthcheck}"
AUDIT_TAG="${AUDIT_TAG:-health-check}"

python3 "$ROOT_DIR/scripts/exec_with_audit.py" \
  --root "$ROOT_DIR" \
  --source "$AUDIT_SOURCE" \
  --tag "$AUDIT_TAG" \
  -- PYTHONPATH="$ROOT_DIR/src" python3 -m lie_engine.cli --config "$ROOT_DIR/config.yaml" health-check --date "$DATE_ARG"
