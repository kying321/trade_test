# Entrypoint Audit Matrix

## Goal
All operational entrypoints must route through `scripts/exec_with_audit.py` so `output/logs/command_exec.ndjson` is continuously populated.

## Entrypoints

1. Manual shell
- Command:
  - `python scripts/exec_with_audit.py --root "$PWD" --source manual --tag review-cycle -- PYTHONPATH="$PWD/src" python -m lie_engine.cli --config config.yaml run-review-cycle --date YYYY-MM-DD --max-rounds 2`

2. Local cron
- Installer:
  - `./infra/local/install_cron.sh`
- Current status:
  - cron lines export `AUDIT_SOURCE=cron` and `AUDIT_TAG=slot-*`
  - `infra/local/run_slot.sh` and `infra/local/healthcheck.sh` call `exec_with_audit.py`

3. macOS launchd
- Installer:
  - `./infra/local/install_launchd.sh`
- Uninstaller:
  - `./infra/local/uninstall_launchd.sh`
- Agent:
  - `com.lie.antifragile.daemon`
  - wraps `run-daemon --poll-seconds 30` via `exec_with_audit.py`

4. Cloud cron / container
- `infra/cloud/retry_slot.sh` -> `infra/cloud/run_slot.sh` -> `exec_with_audit.py`
- `infra/cloud/healthcheck.sh` -> `exec_with_audit.py`
- `infra/cloud/k8s-cronjobs.yaml` sets `AUDIT_SOURCE=k8s-cron` + slot tags
- `infra/docker-compose.yml` wraps `run-daemon` with `exec_with_audit.py`

5. CI gate
- Workflow:
  - `.github/workflows/pr-gate.yml`
- Runs audited `validate-config`, audited `test-all --fast`, then generates 24h whitelist.

## Verification Checklist
- `tail -n 5 output/logs/command_exec.ndjson`
- `python scripts/command_whitelist_24h.py --include-tests-log`
- Confirm whitelist JSON contains command timestamps and return codes.
