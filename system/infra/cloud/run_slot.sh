#!/usr/bin/env sh
set -eu

ROOT_DIR="/app"
DATE_ARG="${DATE:-$(date +%F)}"
SLOT_ARG="${SLOT:-08:40}"
MAX_REVIEW_ROUNDS="${MAX_REVIEW_ROUNDS:-2}"
AUDIT_SOURCE="${AUDIT_SOURCE:-cloud-slot}"
AUDIT_TAG="${AUDIT_TAG:-run-slot-${SLOT_ARG}}"

cd "$ROOT_DIR"
python3 "$ROOT_DIR/scripts/exec_with_audit.py" \
  --root "$ROOT_DIR" \
  --source "$AUDIT_SOURCE" \
  --tag "$AUDIT_TAG" \
  -- PYTHONPATH="$ROOT_DIR/src" python3 -m lie_engine.cli --config "$ROOT_DIR/config.yaml" run-slot \
    --date "$DATE_ARG" \
    --slot "$SLOT_ARG" \
    --max-review-rounds "$MAX_REVIEW_ROUNDS"
