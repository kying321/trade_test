# OpenClaw Cloud Bridge Runbook

## Goal
- Hard-disable local OpenClaw runtime entrypoints (`9999/18790/18793`).
- Keep cloud OpenClaw as the single active execution brain.
- Keep local machine as a control plane for sync, diff, and tunnel probes.

## Script
- `scripts/openclaw_cloud_bridge.sh`

## Required env
- `CLOUD_HOST` (default: `43.153.148.242`)
- `CLOUD_USER` (default: `ubuntu`)
- `CLOUD_PROJECT_DIR` (default: `/home/ubuntu/openclaw-system`)
- `CLOUD_PASS` (optional if key auth is configured)
- `FENLIE_SYSTEM_ROOT` (recommended for launchd: `/Users/jokenrobot/.openclaw/workspaces/pi/fenlie-system`)

## Keychain secret governance (recommended)
- 避免在 LaunchAgent plist 明文注入 `CLOUD_PASS`。
- 先写入 Keychain（service: `openclaw.pi.cloud_pass`）：
```bash
security add-generic-password -U \
  -a "$USER" \
  -s "openclaw.pi.cloud_pass" \
  -w '***'
```
- LaunchAgent 使用：
  - `CLOUD_PASS_KEYCHAIN_SERVICE=openclaw.pi.cloud_pass`
  - 不设置 `CLOUD_PASS` 明文值。
- `pi_cycle_halfhour_launchd_runner.sh` 会在 gate 前自动执行：
  - `security find-generic-password -a "$USER" -s "$CLOUD_PASS_KEYCHAIN_SERVICE" -w`

## Command sequence (recommended)
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
export CLOUD_PASS='***'

scripts/openclaw_cloud_bridge.sh cut-local
scripts/openclaw_cloud_bridge.sh probe-cloud
scripts/openclaw_cloud_bridge.sh compare
scripts/openclaw_cloud_bridge.sh backup-remote
scripts/openclaw_cloud_bridge.sh tunnel-up
scripts/openclaw_cloud_bridge.sh tunnel-probe
scripts/openclaw_cloud_bridge.sh sync-dry-run
scripts/openclaw_cloud_bridge.sh sync-apply
scripts/openclaw_cloud_bridge.sh remote-clean-junk
scripts/openclaw_cloud_bridge.sh validate-remote-config
scripts/openclaw_cloud_bridge.sh tunnel-down
scripts/openclaw_cloud_bridge.sh live-takeover-probe
scripts/openclaw_cloud_bridge.sh live-takeover-canary
scripts/openclaw_cloud_bridge.sh live-takeover-ready-check
scripts/openclaw_cloud_bridge.sh live-takeover-autopilot
scripts/openclaw_cloud_bridge.sh sample-whitelist
scripts/openclaw_cloud_bridge.sh sample-whitelist-gate
scripts/openclaw_cloud_bridge.sh assert-whitelist-gate
scripts/openclaw_cloud_bridge.sh ensure-whitelist-gate
```

## Local forwarded ports
- `127.0.0.1:19999` -> cloud `127.0.0.1:9999` (adaptor)
- `127.0.0.1:18000` -> cloud `127.0.0.1:8000` (api)
- `127.0.0.1:15173` -> cloud `127.0.0.1:5173` (dashboard)

## Live takeover (Binance + evoMap)
- 远端执行（不下单，仅激活配置/策略/成交回流探测）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
LIVE_TAKEOVER_CANARY_USDT=5 \
LIVE_TAKEOVER_MAX_DRAWDOWN=0.05 \
LIVE_TAKEOVER_RATE_LIMIT_PER_MINUTE=10 \
LIVE_TAKEOVER_MARKET=spot \
LIVE_TAKEOVER_FORWARD_LOCAL_CREDS=true \
scripts/openclaw_cloud_bridge.sh live-takeover-probe
```
- 远端执行（最小资金 canary 实盘，受幂等键保护）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
LIVE_TAKEOVER_CANARY_USDT=5 \
LIVE_TAKEOVER_MAX_DRAWDOWN=0.05 \
LIVE_TAKEOVER_RATE_LIMIT_PER_MINUTE=10 \
LIVE_TAKEOVER_MARKET=spot \
LIVE_TAKEOVER_FORWARD_LOCAL_CREDS=true \
scripts/openclaw_cloud_bridge.sh live-takeover-canary
```
- canary 前置就绪检查（余额/凭据）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
LIVE_TAKEOVER_MARKET=spot \
scripts/openclaw_cloud_bridge.sh live-takeover-ready-check
```
- 自动化接管（先检查，满足条件再 canary 下单；不满足则跳过并返回结构化原因）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
LIVE_TAKEOVER_MARKET=spot \
scripts/openclaw_cloud_bridge.sh live-takeover-autopilot
```
- 产物：
  - `output/review/*_binance_live_takeover.json`
  - `output/artifacts/evomap/*_strategy.json`
  - `output/artifacts/broker_live_inbox/YYYY-MM-DD.json`（仅在 Binance signed 凭据完整时生成）
  - `output/artifacts/binance_live_trades/YYYY-MM-DD.json`
  - `output/artifacts/binance_live_income/YYYY-MM-DD.json`

## Sync policy
- Sync scope: `src/`, `scripts/`, `docs/`, `tests/`, `config.yaml`, `pyproject.toml`
- Excludes: `.git`, `__pycache__`, `*.pyc`, `output/`, `dashboard/node_modules`, `dashboard/.next`, `dashboard/out`
- `sync-dry-run`: delete-aware预演（看见将删除什么）
- `sync-apply`: 安全落地（不删除云端 remote-only 文件）
- `sync-apply-prune`: 破坏式收敛（会删除云端 remote-only，只有在白名单裁决后使用）

## 24h 命令白名单采样
- 用途：输出“真实可执行命令 + 时间戳 + 返回码 + 最近24h成功率”证据，避免失效入口继续挂在守护链路。
- 命令：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh sample-whitelist
```
- 熔断命令（失败返回码 `3`）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
WHITELIST_ENFORCE=true \
WHITELIST_MIN_TOTAL_SUCCESS_RATE=0.95 \
WHITELIST_MIN_ACTION_SUCCESS_RATE=0.80 \
WHITELIST_MIN_SAMPLES_PER_ACTION=1 \
scripts/openclaw_cloud_bridge.sh sample-whitelist
```
- 等价快捷命令：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh sample-whitelist-gate
```
- 轻量断言命令（不重采样，适合放在 30m launchd 前置）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
WHITELIST_ASSERT_MAX_AGE_MINUTES=90 \
WHITELIST_MIN_TOTAL_SUCCESS_RATE=0.95 \
WHITELIST_MIN_ACTION_SUCCESS_RATE=0.66 \
scripts/openclaw_cloud_bridge.sh assert-whitelist-gate
```
- 自愈命令（先断言，失败后自动重采样再断言）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
WHITELIST_ASSERT_MAX_AGE_MINUTES=90 \
WHITELIST_MIN_TOTAL_SUCCESS_RATE=0.95 \
WHITELIST_MIN_ACTION_SUCCESS_RATE=0.66 \
scripts/openclaw_cloud_bridge.sh ensure-whitelist-gate
```
- 产物：
  - `output/review/*_openclaw_bridge_whitelist_24h.json`
  - `output/review/*_openclaw_bridge_whitelist_24h.md`
  - `output/logs/openclaw_bridge_whitelist_samples.jsonl`

## Night retro (自动化复盘补全)
- 一键生成指定时间窗复盘报告（JSON + Markdown）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
python3 scripts/pi_launchd_night_retro.py \
  --launchd-log /Users/jokenrobot/.openclaw/logs/pi_cycle_launchd.log \
  --sample-log /Users/jokenrobot/.openclaw/workspaces/pi/fenlie-system/output/logs/openclaw_bridge_whitelist_samples.pre_migration_20260304T030458Z.jsonl \
  --sample-log /Users/jokenrobot/.openclaw/workspaces/pi/fenlie-system/output/logs/openclaw_bridge_whitelist_samples.jsonl \
  --review-dir /Users/jokenrobot/.openclaw/workspaces/pi/fenlie-system/output/review \
  --start-utc 2026-03-03T19:50:00Z \
  --end-utc 2026-03-04T03:10:00Z \
  --out-prefix pi_automation_night_retro
```
- 默认行为（不传 start/end）：回溯最近 `12h`。

## Rollback
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh tunnel-down || true
launchctl enable gui/$(id -u)/ai.openclaw.gateway || true
launchctl enable gui/$(id -u)/ai.openclaw.adaptor || true
launchctl enable gui/$(id -u)/ai.openclaw.pi_cycle_halfhour || true
launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/ai.openclaw.gateway.plist" || true
launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/ai.openclaw.adaptor.plist" || true
launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/ai.openclaw.pi_cycle_halfhour.plist" || true
```
