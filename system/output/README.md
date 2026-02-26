# Output Directory Policy

This directory is runtime-only and should not be committed.

Subdirectories:
- `artifacts/`: generated machine artifacts (cache, manifests, sqlite snapshots).
- `daily/`: day-level generated briefings/signals/positions.
- `logs/`: daemon/test/runtime logs and lock/marker files.
- `review/`: generated review reports and retrospectives.
- `state/`: runtime state checkpoints.

Repository policy:
- Keep directory structure local for operations.
- Do not version generated files under `output/`.
- Move hand-written long-lived documents into `docs/`.

