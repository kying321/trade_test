#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATE_ARG="${1:-$(date +%F)}"
SLOT_ARG="${2:-08:40}"
MAX_REVIEW_ROUNDS="${MAX_REVIEW_ROUNDS:-2}"

lie --config "$ROOT_DIR/config.yaml" run-slot \
  --date "$DATE_ARG" \
  --slot "$SLOT_ARG" \
  --max-review-rounds "$MAX_REVIEW_ROUNDS"
