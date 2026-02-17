#!/usr/bin/env sh
set -eu

ROOT_DIR="/app"
DATE_ARG="${DATE:-$(date +%F)}"
SLOT_ARG="${SLOT:-08:40}"
MAX_REVIEW_ROUNDS="${MAX_REVIEW_ROUNDS:-2}"

cd "$ROOT_DIR"
python3 -m lie_engine.cli --config "$ROOT_DIR/config.yaml" run-slot \
  --date "$DATE_ARG" \
  --slot "$SLOT_ARG" \
  --max-review-rounds "$MAX_REVIEW_ROUNDS"
