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
- `INFRA_CANARY_SYMBOL` (default: `BTCUSDT`, 固定 infra canary symbol)
- `INFRA_CANARY_QUOTE_USDT` (default: `10`, 目标 round-trip quote)
- `INFRA_CANARY_SINGLE_RUN_CAP_USDT` (default: `12`, 动态最小可回平金额的单次上限)
- `INFRA_CANARY_DAILY_BUDGET_CAP_USDT` (default: `20`, autopilot/budget guard)
- `INFRA_CANARY_ALLOW_DUST` (default: `true`, 允许 dust residual but must be audited)

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
scripts/openclaw_cloud_bridge.sh infra-canary-probe
scripts/openclaw_cloud_bridge.sh infra-canary-run
scripts/openclaw_cloud_bridge.sh infra-canary-autopilot
scripts/openclaw_cloud_bridge.sh sample-whitelist
scripts/openclaw_cloud_bridge.sh sample-whitelist-gate
scripts/openclaw_cloud_bridge.sh assert-whitelist-gate
scripts/openclaw_cloud_bridge.sh ensure-whitelist-gate
```

## Local forwarded ports
- `127.0.0.1:19999` -> cloud `127.0.0.1:9999` (adaptor)
- `127.0.0.1:18000` -> cloud `127.0.0.1:8000` (api)
- `127.0.0.1:15173` -> cloud `127.0.0.1:5173` (dashboard)

## Cloud 9999 adaptor runtime source-of-truth (2026-03-19)
- 云端运行模型：
  - `~/.openclaw/openclaw.json`
  - `agents.defaults.model.primary = openai/gpt-5.4`
  - `models.providers.openai.baseUrl = http://127.0.0.1:9999/v1`
- 云端 9999 实际监听服务：
  - systemd: `pi-adaptor.service`
  - process: `/usr/bin/node /home/ubuntu/adaptor.js`
- 当前上游 key 路由 source-of-truth：
  - `/home/ubuntu/adaptor.js`
  - `/home/ubuntu/.openclaw/.env`
- 当前 policy：
  - `GET /v1/models` 固定使用 key pool 第 1 把
  - `POST` 请求按 `NEWAPI_API_KEYS` 做 request-level round-robin
  - 若 `NEWAPI_API_KEYS` 缺失，才 fallback 到 `NEWAPI_API_KEY` / `X666_API_KEY` / `OPENAI_API_KEY`
- 当前已验证的 key pool 目标：
  - `sk-MOc...Z7rf`
  - `sk-VOB...AgSX`
- 当前 smoke-check 证据：
  - `journalctl -u pi-adaptor.service`
    - `KeyPool loaded 2 upstream key(s) from env chain`
  - `/tmp/proxy_debug.log`
    - 最近请求应出现 `key_slot=1/2 -> 2/2 -> 1/2 -> 2/2`
- 当前直接回滚：
```bash
cp /home/ubuntu/adaptor.js.bak_20260319T063533Z /home/ubuntu/adaptor.js
cp /home/ubuntu/.openclaw/.env.bak_20260319T063800Z /home/ubuntu/.openclaw/.env
sudo systemctl restart pi-adaptor.service
```
- 说明：
  - `openclaw.json` 中 legacy provider `apiKey` 字段仍可能存在，但不再是 9999 上游轮询的 source-of-truth
  - 9999 上游轮询以 `adaptor.js + ~/.openclaw/.env` 为准

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

## Infra canary (独立执行基础设施探活)
- 所有 infra canary 命令都是对 `scripts/binance_infra_canary.py` 的封装，默认：`symbol=BTCUSDT`、`market=spot`、`requested_quote_usdt=10`、`single_run_cap_usdt=12`、`allow_dust=true`、`daily_budget_cap_usdt=20`。
- `infra-canary-probe` 在云端只校验 credentials、账户余额、budget/idempotency ledger、panic/transport guard，它永远不读取策略 ticket 数据，也不下单。
- `infra-canary-run` 执行一次 round-trip（买入 `BTCUSDT`；目标 quote 默认 `10 USDT`，若交易所约束要求更高则自动抬高到 `required_round_trip_quote_usdt`；若所需金额超过 `12 USDT` 则 graceful skip），并把 `dust`、`budget`、`idempotency`、`steps` 产物写入 `output/state` 和 `output/review` 目录。
- `infra-canary-autopilot` 会先调用 `binance_infra_canary.py --mode autopilot-check`，只有当返回的 JSON 同时满足 `ok=true` 和 `autopilot_allowed=true` 才继续触发 `infra-canary-run`，否则输出 `skipped_not_ready`/`ready_to_run` 等结构化 payload 并优雅退出，不会放行策略路径。
- `autopilot_allowed` 必须是 bool；非 bool 会被视为 gate error，整体 run 直接跳过。即便 `autopilot_allowed=true`，仍然受 daily budget、幂等、mutex、panic guard 限制。
- Infra canary 成功只证明 execution plumbing 没问题，**不代表策略 ready**；操作文档/brief 必须写明 `infra canary success ≠ strategy ready`。
- 历史残留提示：旧版固定 `5 USDT` canary 若已留下 `needs_recovery / sell_exchange_reject`，仍以 `output/state/infra_canary_idempotency.json` 为准，不会因新版动态金额合同自动清除。

## TradingView Basis Arbitrage 路径
- 需要 TradingView 信号触发的基差套利请参考 `system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`，内含 webhook payload、entry/exit gate、状态账本、恢复流程与 smoke 命令。
- 同样强调：**套利成功 ≠ infra canary 成功 ≠ strategy ticket ready**，只有 gate/pass、artifact 不留 `needs_recovery`、state 账本干净之后才算策略真正就绪。

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
