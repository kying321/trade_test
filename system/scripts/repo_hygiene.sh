#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "[repo-hygiene] root=${ROOT}"

echo "[repo-hygiene] removing local caches..."
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
find . -type f -name ".DS_Store" -delete

echo "[repo-hygiene] done. current git summary:"
git status --short | sed -n '1,120p'

