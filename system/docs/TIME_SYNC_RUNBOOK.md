# Time Sync Runbook

## Goal
- Build a verifiable time authority plane for trading runtime.
- Keep `ntp_max_offset_ms <= 5` as the hard target.
- Feed time quality into LiE diagnostics via `time-sync-probe` and `run-eod`.

## Runtime Hooks
- CLI command:
  - `PYTHONPATH=src python -m lie_engine.cli --config config.yaml time-sync-probe --date YYYY-MM-DD`
- EOD integration:
  - `run-eod` now writes `system_time_sync_probe` artifact and sqlite tables:
    - `system_time_sync_daily`
    - `system_time_sync_source_state`

## Config Keys
- `validation.system_time_sync_probe_enabled`
- `validation.system_time_sync_hard_fuse_enabled`
- `validation.system_time_sync_primary_source`
- `validation.system_time_sync_secondary_source`
- `validation.system_time_sync_probe_timeout_seconds` (max 5.0)
- `validation.system_time_sync_max_offset_ms`
- `validation.system_time_sync_max_rtt_ms`
- `validation.system_time_sync_min_ok_sources`

## Linux (Cloud) Baseline
1. Install chrony:
```bash
sudo apt-get update
sudo apt-get install -y chrony
```
2. Configure `/etc/chrony/chrony.conf`:
```conf
pool time.google.com iburst maxsources 2
pool time.cloudflare.com iburst maxsources 2
makestep 0.1 3
rtcsync
maxupdateskew 50.0
driftfile /var/lib/chrony/chrony.drift
```
3. Restart and verify:
```bash
sudo systemctl restart chrony
chronyc tracking
chronyc sources -v
```

## Guard Strategy
- Monitoring mode:
  - `system_time_sync_probe_enabled=true`
  - `system_time_sync_hard_fuse_enabled=false`
- Hard fuse mode:
  - `system_time_sync_hard_fuse_enabled=true`
  - trigger non-trade reason when `ok_sources < min_ok_sources`.

## Practical Note
- If offset is stable around `>1000ms`, this is host clock drift and must be fixed at OS layer (chrony/host time policy) before enabling hard fuse.
