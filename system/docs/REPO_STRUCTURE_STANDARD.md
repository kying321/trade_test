# Repository Structure Standard

## 1. Scope
This standard defines what must be committed and what must remain runtime-local.

## 2. Commit Zones
- `system/src/`: production source code.
- `system/tests/`: test code.
- `system/docs/`: design docs, runbooks, ADRs, governance rules.
- `system/infra/`: deployment and environment automation scripts.
- `system/scripts/`: operational scripts (idempotent, deterministic).
- `system/config*.yaml`: versioned baseline config.

## 3. Runtime-Only Zones
- `system/output/**`
- `**/__pycache__/`, `*.pyc`, `*.egg-info/`
- `.DS_Store`, lock/pid/flag files
- root `*.pdf` raw research source files

Runtime outputs are valuable for operations but should not be source-of-truth in Git.
Persisted decisions should be summarized into `system/docs/` if long-lived.

## 4. Naming Rules
- Use lowercase snake_case for files/scripts.
- Use ISO dates for generated report names: `YYYY-MM-DD_*`.
- Keep branch naming `codex/*` for automation pushes.

## 5. Hygiene Rules
- No generated binaries or caches in commits.
- No machine-local lock/pid files in commits.
- No mixed concerns in one commit (code + bulk runtime artifacts).

## 6. Recommended Workflow
1. Implement scoped change.
2. Run validation/tests.
3. Sync only commit-zone files.
4. Push branch and open PR.
