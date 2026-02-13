#!/usr/bin/env sh
set -eu

ROOT_DIR="/app"
DATE_ARG="${DATE:-$(date +%F)}"
python3 -m lie_engine.cli --config "$ROOT_DIR/config.yaml" health-check --date "$DATE_ARG"
