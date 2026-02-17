#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATE_ARG="${1:-$(date +%F)}"
lie --config "$ROOT_DIR/config.yaml" health-check --date "$DATE_ARG"
