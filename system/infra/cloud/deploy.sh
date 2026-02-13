#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="lie-engine:latest"

cd "$(dirname "$0")/../.."
docker build -f infra/cloud/Dockerfile -t "$IMAGE_NAME" .
echo "Built $IMAGE_NAME"
