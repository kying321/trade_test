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

## Local PI workspace sync
- 目标：把 repo `system/` 安全同步到本地 OpenClaw runtime workspace `~/.openclaw/workspaces/pi/fenlie-system`，避免 launchd 实际执行代码与仓库主线分叉。
- 行为：
  - 只做增量覆盖，不删除 target 独有文件
  - 永远跳过 `output/`、缓存目录和 `node_modules/`
  - 覆盖前把旧文件备份到 `output/backups/workspace_sync_*`
  - 自动清理过旧/过多的 `workspace_sync_*` 备份
  - 非 dry-run 同步会先持有本地 `run_halfhour_pulse.lock`，避免与 launchd/full-cycle 写路径并发冲突
- 直接执行：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh sync-local-pi-workspace
```
- 发布 repo 托管的本地 PI runtime 热路径脚本：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh publish-local-pi-runtime-scripts
```
- 当前托管的 runtime 热路径由显式 manifest 驱动：
  - [runtime_manifest.json](/Users/jokenrobot/Downloads/Folders/fenlie/system/runtime/pi/runtime_manifest.json)
  - [runtime/pi/scripts](/Users/jokenrobot/Downloads/Folders/fenlie/system/runtime/pi/scripts)
- `publish-local-pi-runtime-scripts` 只会发布 manifest 中列出的文件，不会自动把目录里的临时脚本带进 runtime
- runtime 脚本发布同样持有本地 `run_halfhour_pulse.lock`，并把覆盖前版本备份到 `output/backups/runtime_script_publish_*`
- 一次性准备本地 PI runtime（同步 workspace + 发布 runtime 脚本 + 修 runtime model + gate-only smoke）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh prepare-local-pi-runtime
```
- 一次性准备并跑完整本地 PI full-cycle smoke：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh smoke-local-pi-cycle
```
- 运行隔离恢复演练（复制真实 paper state 到临时 lab，执行 fallback backfill -> manual ack -> runtime consume/archive，不改真实 workspace）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh run-local-pi-recovery-lab
```
- `run-local-pi-recovery-lab` 的报告现在会额外返回：
  - `artifact_status_label` / `artifact_label` / `artifact_tags`
  - `projection_validation`
    - 用来对齐隔离执行链和 `write_projection`，明确哪些步骤已经验证通过，哪些终态仍未在 lab 中真实执行
  - `operator_note`
    - 用于在离线查看 lab 报告时直接给出“进入真实写入前该先看什么”的操作提示
- 创建真实 workspace 的恢复 checkpoint：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh snapshot-local-pi-recovery-state
```
- 使用 checkpoint 做 restore dry-run：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT="/abs/path/to/checkpoint.json" \
scripts/openclaw_cloud_bridge.sh restore-local-pi-recovery-state
```
- 使用最新 checkpoint 做统一 rollback dry-run：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh rollback-local-pi-recovery-state
```
- 生成单文件恢复交接工件（当前 guardrail、最新 checkpoint、最新 archive、最新 retro 汇总）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh local-pi-recovery-handoff
```
- 查看或写入本地 paper 连亏 guardrail ack：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh backfill-local-pi-last-loss-ts
scripts/openclaw_cloud_bridge.sh local-pi-consecutive-loss-guardrail-status
scripts/openclaw_cloud_bridge.sh local-pi-ack-archive-status
scripts/openclaw_cloud_bridge.sh apply-local-pi-recovery-step
scripts/openclaw_cloud_bridge.sh run-local-pi-recovery-flow
scripts/openclaw_cloud_bridge.sh ack-local-pi-consecutive-loss-guardrail
```
- `local-pi-consecutive-loss-guardrail-status` 现在会同时返回：
  - `state_fingerprint`
  - 当前 streak / cooldown / ack 可用性
  - strict `last_loss_ts` 回填候选
  - fallback `last_loss_ts` 回填候选
  - `ack_archive`
    - 最近 live ack 文件状态
    - 最近 consumed archive 摘要
    - archive manifest tail
  - `write_projection`
    - 在不改 state 的前提下，投影“如果开启写入，恢复链会推进到哪一步”
    - 当前可直接看出是否会按 `fallback backfill -> manual ack -> full-cycle gate ready` 推进
- `local-pi-ack-archive-status` 会单独返回：
  - live `paper_consecutive_loss_ack.json` 是否存在、checksum 是否有效
  - 最近一次 consumed archive 的路径、checksum、payload 摘要
  - archive manifest 的 tail 事件
  - 当前 archive 文件数与保留上限提示
- `apply-local-pi-recovery-step` 默认只预演 `recovery_plan` 推荐动作：
  - fallback backfill 只做 preview
  - manual ack 默认不写入
  - 只有显式设置对应环境变量才会执行真实写入或 full-cycle
  - 返回里会透传 `write_projection` 和 `post_write_projection`
  - 实际触发 backfill/ack 时，会自动携带当前 `state_fingerprint`
    - 底层写入脚本按 compare-and-set 校验
    - 如果 `spot_paper_state.json` 在预演和写入之间发生变化，会返回 `state_fingerprint_mismatch`
  - 默认开启 `LOCAL_PI_RECOVERY_AUTO_SNAPSHOT_BEFORE_WRITE=true`
    - 在真实 `fallback backfill` / `strict backfill` / `manual ack` 写入前，会先自动执行一次 `snapshot-local-pi-recovery-state`
    - step 输出会包含 `snapshot_result`
    - step 输出也会包含 `rollback_guidance`
      - 直接给出基于该 checkpoint 的 rollback dry-run / rollback write 命令
    - step 输出还会包含 `operator_note`
      - 直接说明何时该用 rollback，以及优先执行哪条命令
    - 如果 auto snapshot 失败，step 会直接以 `snapshot_failed` 停止，不继续写 state
- `run-local-pi-recovery-flow` 会串联多步恢复，但同样默认停在 preview：
  - 预演模式下通常会在 `preview_mode` 停住
  - 每次运行都会写 `*_local_pi_recovery_flow.json`、对应 checksum，以及逐步的 `*_local_pi_recovery_flow_step_*.json` 到本地 workspace `output/review`
  - 顶层会额外给出 `initial_write_projection` / `final_write_projection`
  - 顶层还会汇总 `checkpoints` / `checkpoint_count`
    - 这些来自 flow 内各写步骤自动创建的恢复 checkpoint
  - 顶层还会返回 `rollback_guidance`
    - 默认指向最近一次可用 recovery checkpoint
  - 顶层还会返回 `operator_note`
    - 用于离线查看 flow artifact 时直接给出 rollback 操作建议
  - flow artifact 与逐步 step artifact 现在都会带：
    - `artifact_label`
    - `artifact_status_label`
    - `artifact_tags`
    - `operator_note`（若存在）
  - 默认开启 `LOCAL_PI_RECOVERY_AUTO_ROLLBACK_ON_FAILURE=true`
    - 仅当真实写步骤失败、该步骤前已经自动创建 checkpoint、且 flow 能确认恢复状态已实际变更时，才会触发 rollback hook
    - 默认只做 rollback dry-run
    - 若显式设置 `LOCAL_PI_RECOVERY_AUTO_ROLLBACK_WRITE=true`，才会执行真实 rollback 写回
    - flow 顶层会返回 `rollback_attempted` / `rollback_result`
  - 默认开启 `LOCAL_PI_RECOVERY_ENFORCE_PROJECTION=true`
    - 只有实际执行步骤和 `write_projection.projected_steps[0]` 一致时，flow 才允许自动推进到下一步
    - 若实际推进和投影不一致，会以 `projection_*` 停止原因直接停住
- `rollback-local-pi-recovery-state` 是 `restore-local-pi-recovery-state` 的便捷入口：
  - 若未显式传 `LOCAL_PI_RECOVERY_RESTORE_CHECKPOINT`，会自动选择 `LOCAL_PI_RECOVERY_CHECKPOINT_DIR` 里最新的 checkpoint
  - 默认仍是 dry-run
  - 只有显式设置 `LOCAL_PI_RECOVERY_RESTORE_WRITE=true` 才会真正写回 state
- 如果历史迁移导致 `last_loss_ts` 缺失，可先做只读回填预演；默认要求 ledger 尾部连续亏损段与当前 `consecutive_losses` 精确匹配：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh backfill-local-pi-last-loss-ts
```
- 仅在你明确接受“用最新亏损成交时间替代精确 streak 对齐”时，才打开 fallback：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
LOCAL_PI_LAST_LOSS_TS_BACKFILL_ALLOW_LATEST_LOSS_FALLBACK=true \
scripts/openclaw_cloud_bridge.sh backfill-local-pi-last-loss-ts
```
- 真正写入 ack 需要显式开启：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
LOCAL_PI_CONSECUTIVE_LOSS_ACK_WRITE=true \
LOCAL_PI_CONSECUTIVE_LOSS_ACK_ALLOW_MISSING_LAST_LOSS_TS=true \
scripts/openclaw_cloud_bridge.sh ack-local-pi-consecutive-loss-guardrail
```
- 本地 paper gate 还支持 warmup override：当 7d 历史覆盖率仍因迁移期缺口不达标，但最近 launchd 半小时调度足够稳定时，只对本地 paper cycle 放宽 `paper_mode_readiness`，不影响任何 live gate。
- 连续亏损 guardrail 现在也支持显式 `manual ack`，但仍默认 fail-closed：
  - 只有 fresh ack artifact、TTL 未过期、streak 精确匹配、且冷却条件满足时，才会暂时放开 `consecutive_loss_stop_hit`。
  - 没有 ack 时，系统不会自动恢复。
  - ack 现在是 single-use：`uses_remaining=1`，runtime 一旦采用，当前周期后立即消费，不能长期悬挂。
- 常用环境变量：
  - `LOCAL_PI_WORKSPACE_SYSTEM_ROOT` 默认 `~/.openclaw/workspaces/pi/fenlie-system`
  - `LOCAL_PI_WORKSPACE_BACKUP_KEEP` 默认 `5`
  - `LOCAL_PI_WORKSPACE_BACKUP_MAX_AGE_HOURS` 默认 `168`
  - `LOCAL_PI_WORKSPACE_DRY_RUN=true` 可先做计划预览
  - `LOCAL_PI_WORKSPACE_NO_BACKUP=true` 可禁用本次覆盖备份
  - `LOCAL_PI_LAUNCHD_RUNNER_PATH` 默认 `~/.openclaw/workspaces/pi/scripts/pi_cycle_halfhour_launchd_runner.sh`
  - `LOCAL_PI_LAUNCHD_LOG_PATH` 默认 `~/.openclaw/logs/pi_cycle_launchd.log`
  - `LOCAL_PI_GATE_WINDOW_HOURS` 默认 `8`
  - `LOCAL_PI_PREPARE_BEFORE_FULL_SMOKE=false` 可跳过 `prepare-local-pi-runtime`，直接跑 full-cycle smoke
  - `LOCAL_PI_LAST_LOSS_TS_BACKFILL_ALLOW_LATEST_LOSS_FALLBACK=true` 可允许回填器用“最近亏损成交时间”替代精确 streak 对齐
  - `LOCAL_PI_LAST_LOSS_TS_BACKFILL_WRITE=true` 才会真正把回填结果写回 `spot_paper_state.json`
  - `LOCAL_PI_CONSECUTIVE_LOSS_STOP_THRESHOLD` 默认 `3`
  - `LOCAL_PI_CONSECUTIVE_LOSS_ACK_TTL_HOURS` 默认 `24`
  - `LOCAL_PI_CONSECUTIVE_LOSS_ACK_COOLDOWN_HOURS` 默认 `12`
  - `LOCAL_PI_CONSECUTIVE_LOSS_ACK_ALLOW_MISSING_LAST_LOSS_TS=true` 仅用于历史迁移场景
  - `LOCAL_PI_CONSECUTIVE_LOSS_ACK_WRITE=true` 才会真正落盘 ack artifact
  - `LOCAL_PI_CONSECUTIVE_LOSS_ACK_NOTE` 可写操作员备注
  - `LIE_PAPER_MODE_READINESS_ALLOW_WARMUP=true` 开启本地 warmup override
  - `LIE_PAPER_MODE_WARMUP_WINDOW_HOURS` 默认 `24`
  - `LIE_PAPER_MODE_WARMUP_COVERAGE_MIN` 默认 `0.80`
  - `LIE_PAPER_MODE_WARMUP_MAX_MISSING_BUCKETS` 默认 `8`
  - `LIE_PAPER_MODE_WARMUP_MAX_LARGEST_MISSING_BLOCK_HOURS` 默认 `2.0`

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

## Live risk daemon (systemd-first)
- 目标：把 `live_risk_fuse.json` 和最新 ticket 刷新从“调用时生成”升级成云端常驻服务，降低 probe/canary 前同步阻塞。
- 安装并启用 `systemd` 服务：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-install-service
```
- 安装输出现在会额外返回 `security_acceptance` 摘要：
  - `status=accepted|review|failed`
  - 默认阈值 `LIVE_RISK_DAEMON_SECURITY_ACCEPT_MAX_EXPOSURE=4.0`
  - 只做安装后验收摘要，不会因为评分偏高而自动卸载或停服务
- 当前 renderer 会默认附带一组保守加固项：
  - `ProtectSystem=strict` + `ReadWritePaths=output/{logs,state,review,artifacts}`
  - `PrivateTmp=true`
  - `PrivateDevices=true`
  - `ProtectControlGroups=true`
  - `ProtectKernelTunables=true`
  - `ProtectKernelModules=true`
  - `ProtectKernelLogs=true`
  - `ProtectClock=true`
  - `ProtectProc=invisible`
  - `ProtectHome=read-only`
  - `ProcSubset=pid`
  - `PrivateUsers=true`
  - `PrivateNetwork=true`
  - `IPAddressDeny=any`
  - `RestrictAddressFamilies=AF_UNIX`
  - `SystemCallFilter=@system-service`
  - `CapabilityBoundingSet=` / `AmbientCapabilities=`
  - `PYTHONDONTWRITEBYTECODE=1`（避免在只读源码树下生成 `__pycache__`）
- 查看服务状态：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-service-status
```
- `live-risk-daemon-service-status` 现在会额外返回 `payload_alignment`，用于判断 `systemd MainPID` 和 `output/state/live_risk_daemon.json` 是否已经对齐。
- 查看服务的 `systemd-analyze verify/security` 摘要：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-security-status
```
- 查看最近 daemon 日志（优先 `journalctl`，没有 service 时回退到 `output/logs/live_risk_daemon.out.log`）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-journal
```
- 停止并移除服务：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-remove-service
```
- 兼容语义：
  - 一旦远端已安装 `fenlie-live-risk-daemon.service`，`live-risk-daemon-start/status/stop` 会自动委托给 `systemd`，不再走手工 `Popen/kill` 路径。
  - `live-takeover-probe`、`live-takeover-canary`、`live-fast-skill` 会优先消费 daemon 维护的 fresh fuse；只有 fuse 过旧或缺失时才同步回退到 `live_risk_guard.py --refresh-tickets`。
  - `live_risk_guard.py` 现在还会读取可选的 `output/state/backup_web_intel.json`。这层仅有 `risk_only` 权限：可以追加 `no-trade` / `bias conflict` / `high-risk flag` 阻断，但不能放宽现有 gate。
  - `live-takeover-ready-check` 现在还会读取最新 `*_ops_report.json` 的 `reconcile_drift` 摘要；如果工件缺失/过旧，或 fresh artifact 显示 `reconcile_drift` 失败，则 `autopilot` / `live-fast-skill` 会直接返回 not ready。
- 可单独查看对账门状态：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh live-ops-reconcile-status
```
- 生成远端 live 交接工件（聚合 ready-check、risk daemon、ops gate、journal）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh remote-live-handoff
```
  - handoff 现在还会聚合 `live-risk-daemon-security-status`，把 `systemd-analyze verify/security` 的结果写进工件，便于量化服务硬化状态。
  - handoff 现在还会给出 `security_acceptance_status`、`security_acceptance_reasons`、`security_acceptance_max_allowed_exposure`，默认沿用 `LIVE_RISK_DAEMON_SECURITY_ACCEPT_MAX_EXPOSURE=4.0` 的 operator 验收阈值。
  - handoff 现在还会给出 `daemon_payload_alignment_ok` 和 `daemon_payload_alignment_reasons`，用于识别 service 已 active 但 state payload 仍滞后的观察窗口。
  - handoff 现在还会给出统一的 `runtime_status_label` 和 `runtime_attention_reasons`，把 daemon 对齐状态和安全验收状态压成单一 operator 视图。
  - handoff 现在还会给出 `gate_status_label` / `gate_attention_reasons` 和 `risk_guard_status_label` / `risk_guard_attention_reasons`，把 `ops_live_gate` 与 `risk_guard` 压成可快速阅读的两段业务状态标签。
  - handoff 现在还会给出 `operator_status_triplet` 和 `operator_status_summary`，把 `runtime / gate / risk_guard` 三段状态进一步压成单行摘要。
  - handoff 现在还会给出 `next_focus_area` 和 `next_focus_reason`，直接提示 operator 先看 `runtime`、`gate`、`risk_guard` 还是已可进入 `canary`。
  - handoff 现在还会给出 `next_focus_command`，直接给出第一条建议执行的 bridge 命令。
  - handoff 现在还会给出 `next_focus_commands`，输出最小排障序列，减少 operator 自己拼第二步、第三步。
  - handoff 现在还会给出结构化 `operator_playbook`，固定输出 `runtime / gate / risk_guard / canary` 四段状态、摘要和命令序列，便于下游直接消费。
  - `operator_playbook.sections.runtime` 现在还会给出 `address_family_floor` 和 `address_family_note`，直接说明当前主机上已经验证过的地址族收紧边界。
  - handoff 现在还会给出 `operator_playbook_md`，把当前焦点、最小命令序列和四段状态压成可直接阅读的 Markdown 摘要。
  - handoff 现在还会给出 `operator_handoff_md`，把总体状态、当前 blocker、安全摘要和 playbook 合并成单页 Markdown 交接文本。
  - handoff 现在还会给出 `operator_handoff_brief`，把状态、当前 focus 和第一条操作命令压成超短摘要，便于通知或快速扫读。
  - handoff 现在还会给出 `address_family_floor`、`address_family_probe_status` 和 `address_family_recommendation`，用于记录最近一次 `RestrictAddressFamilies=none` 只读 probe 的结论，避免重复尝试已证伪的收紧路径。
  - handoff 现在还会给出 `operator_notification`，固定输出 `level / title / body / command / tags`，便于后续接消息推送模板。
  - handoff 现在还会给出 `operator_notification.plain_text` 和 `operator_notification.markdown`，可直接复用到 Telegram/通知消息模板。
  - handoff 现在还会给出 `operator_notification_templates.telegram / feishu / generic`，把消息模板进一步收口成平台级载荷。
  - handoff 还会给出 `security_top_risks` 和 `security_recommendations`，用于快速定位下一步的 service hardening 方向。
- 生成远端 live 通知预览工件（从最新 handoff 提取通知模板，不真实发送消息）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh remote-live-notification-preview
```
  - preview 工件会包含 `notification`、`notification_templates`、checksum 以及 TTL/LRU 清理结果。
- 生成远端 live 通知 dry-run 工件（校验 Telegram/Feishu 请求体，不真实发送）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh remote-live-notification-dry-run
```
  - dry-run 工件会输出 `telegram.request`、`feishu.request` 以及模板校验结果，便于接真实 sender 前做只读验收。
- 生成远端 live 通知 send 工件（默认 `delivery=none`，只有显式设置 `REMOTE_LIVE_NOTIFICATION_DELIVERY=telegram|feishu|all` 才会尝试出网）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh remote-live-notification-send
```
  - sender 会复用最新 dry-run 工件，补充幂等、token bucket、5 秒超时和 checksum/TTL/LRU 治理。
  - Telegram 真实发送需要：
    - `REMOTE_LIVE_NOTIFICATION_TELEGRAM_TOKEN`
    - `REMOTE_LIVE_NOTIFICATION_TELEGRAM_CHAT_ID`
  - Feishu 真实发送需要：
    - `REMOTE_LIVE_NOTIFICATION_FEISHU_HOOK_TOKEN`
    - 或 `REMOTE_LIVE_NOTIFICATION_FEISHU_WEBHOOK_URL`
- 运行远端 `MemoryDenyWriteExecute` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-mdwe-probe
```
  - probe 会用 `systemd-run --wait --collect` 在远端跑一次 `live_risk_daemon.py --max-cycles 1`。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
- 运行远端 `ProtectHome=read-only` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-protecthome-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外启用 `ProtectHome=read-only` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
- 运行远端 `ProcSubset=pid` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-procsubset-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外启用 `ProcSubset=pid` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
- 运行远端 `PrivateUsers=true` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-privateusers-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外启用 `PrivateUsers=true` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
- 运行远端 `PrivateNetwork=true` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-privatenetwork-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外启用 `PrivateNetwork=true` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
  - probe 验收通过后，正式 `live-risk-daemon-install-service` 会把 `PrivateNetwork=true` 写入 renderer 生成的 unit。
- 运行远端 `IPAddressDeny=any` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-ipdeny-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外启用 `IPAddressDeny=any` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
  - probe 验收通过后，正式 `live-risk-daemon-install-service` 会把 `IPAddressDeny=any` 写入 renderer 生成的 unit。
- 运行远端 `DevicePolicy=closed` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-devicepolicy-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外启用 `DevicePolicy=closed` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
  - 只有 probe 明确兼容，才应把 renderer 继续收紧到 `DevicePolicy=closed`。
- 运行远端 `RestrictAddressFamilies=AF_UNIX` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-afunix-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外把地址族收紧为 `AF_UNIX` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
  - probe 验收通过后，正式 `live-risk-daemon-install-service` 会把 renderer 里的 `RestrictAddressFamilies` 收紧为 `AF_UNIX`。
- 运行远端 `RestrictAddressFamilies=none` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-noaf-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外把地址族收紧为 `none` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
  - 只有 probe 明确兼容，才值得考虑把 renderer 从 `AF_UNIX` 继续收紧到 `none`。
- 运行远端 `SystemCallFilter=@system-service` 兼容性 probe（使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-syscallfilter-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外启用 `SystemCallFilter=@system-service` 做一次单周期兼容性验证。
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
  - probe 验收通过后，正式 `live-risk-daemon-install-service` 会把 renderer 里的 `SystemCallFilter` 收紧到 `@system-service`。
- 运行远端 `SystemCallFilter` 收紧 probe（在 `@system-service` 基础上额外拒绝 `@resources` 和 `@privileged`，使用 transient unit，只读评估，不改已安装 service）：
```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
scripts/openclaw_cloud_bridge.sh live-risk-daemon-syscallfilter-tight-probe
```
  - probe 会保留当前 daemon 的硬化配置，并额外启用：
    - `SystemCallFilter=@system-service`
    - `SystemCallFilter=~@resources`
    - `SystemCallFilter=~@privileged`
  - 这一步不会修改当前 `fenlie-live-risk-daemon.service`，只会生成本地 probe artifact 供验收。
  - probe 验收通过后，正式 `live-risk-daemon-install-service` 会把 renderer 里的 `SystemCallFilter` deny-list 一并写入 unit。
  - 可手动刷新最新对账工件（依赖远端 Python runtime 已具备 Lie 依赖）：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh live-ops-reconcile-refresh
```
  - 如果远端 `ops-report` 因依赖缺失无法刷新，可先执行：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh bootstrap-remote-runtime
```
  - 如果本地或远端 OpenClaw 已经切到 `openai/gpt-5.4`，但实际 cron/gateway 仍报 `Unknown model: openai/gpt-5.4`，先补运行时注册：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
scripts/openclaw_cloud_bridge.sh ensure-local-openclaw-runtime-model
scripts/openclaw_cloud_bridge.sh ensure-remote-openclaw-runtime-model
```
  - 关键状态文件：
    - `/home/ubuntu/openclaw-system/output/state/live_risk_daemon.json`
    - `/home/ubuntu/openclaw-system/output/state/live_risk_fuse.json`

## Backup web intel state writer
- 用途：把外部定时任务/网页采集器生成的 JSON 规范化后写入 `output/state/backup_web_intel.json`，供 `live_risk_guard.py` 以 `risk_only` 权限消费。
- 命令：
```bash
cd /Users/jokenrobot/Downloads/fenlie/system
python3 scripts/build_backup_web_intel.py \
  --input-json /path/to/collector_output.json \
  --output-root output \
  --ttl-seconds 7200
```
- 语义：
  - 只接受结构化 JSON 输入，不直接抓网页。
  - 会强制把 `fallback_trade_authority` 收敛为 `risk_only`。
  - 输入无效时不会覆盖已有的 last-good `backup_web_intel.json`。

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
