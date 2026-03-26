# TradingView Basis Arbitrage Runbook

## 目标
- 把 TradingView 的基差/展期信号交给本地 `tv_basis_arb` 链路，产生命令、门控和状态审计。
- 保证信号、门控、策略头寸和恢复状态都写进 `output/review` 与 `output/state`，方便审计与可视化。

## 触发链路
1. 入口脚本：`system/scripts/tv_basis_arb_webhook.py` 负责解析 TradingView 发过来的 JSON、记录 signal/ gate artifacts，然后调用 `TvBasisArbExecutor` 执行 entry/exit。
2. 门控：`system/scripts/tv_basis_arb_gate.py` 提供 `evaluate_tv_basis_gate`，返回 `passed`、`reasons` 和 `thresholds`；`system/scripts/tv_basis_arb_webhook.py` 在 `entry_check`/`exit_check` 中写入 `gate` 记录并在通过时触发执行器。
3. 执行器：`system/scripts/tv_basis_arb_executor.py` 通过 `TvBasisArbStateLedger` 维护幂等、头寸、恢复，同步 spot/perp 下单并写回 artifacts。

## Venue capability boundary
- source-of-truth artifact：`output/state/venue_capabilities.json`
- 当前 same-venue `spot + perp` 路径在进入真实 live execution 前，会先读取 venue capability artifact，而不是默认假设 Binance futures 可用。
- 当前已知 Binance 诊断结论应被表达为：
  - `spot ready`
  - `futures blocked`
  - blocker=`enableFutures=false`

### 状态语义
- `dry_only`
  - 只允许本地测试 / fake executor / replay / dry acceptance
- `live_blocked`
  - 代码路径存在，但 venue/account capability 不满足；本策略当前 Binance futures 路径应视为这一类
- `live_ready`
  - venue/account capability 满足，且策略侧 gate 也允许

## Webhook payload exemplar
```json
{
  "strategy_id": "tv_basis_btc_spot_perp_v1",
  "symbol": "BTCUSDT",
  "event_type": "entry_check",
  "tv_timestamp": "2026-03-26T12:30:00Z",
  "alert_id": "tv-20260326-1230"
}
```
`strategy_id` 只能是 `tv_basis_btc_spot_perp_v1`，`event_type` 上报 `entry_check` 或 `exit_check`，`tv_timestamp` 统一 ISO-8601（UTC），`alert_id` 可选但会记日志。

## 入口/出口门控说明
- 入口门控由 `evaluate_tv_basis_gate` 执行：
  - `basis_bps` 必须 ≥ `min_basis_bps`（目前 8 bps）。
  - `mark_index_spread_bps` ≤ `max_mark_index_spread_bps`（目前 15 bps）。
  - `open_interest_usdt` ≥ `min_open_interest_usdt`（≥ 10,000,000 USDT）。
  - v1 entry 合同是固定 `target_base_qty = 0.002 BTC`，不是旧的 `20 USDT` quote authority；预算上限由 `max_quote_budget_usdt = 160.0` 控制。
  - 若 `output/state/venue_capabilities.json` 显示当前 venue/account 缺少 same-venue `spot + perp` 所需能力，entry route 会在 executor 前直接 fail-close 为 `live_blocked`。
  - gate 会先检查 `exchange_constraints`（spot/perp 最小数量、最小名义）和 `effective_quote_budget_usdt`，只有这些条件都通过才允许进入执行器。
  - `gate` artifact 会写入 `target_base_qty`、`max_quote_budget_usdt`、`effective_quote_budget_usdt`、`estimated_quote_for_target_usdt`、`estimated_perp_notional_usdt`、`snapshot_ts_utc`、`thresholds`、`reasons` 等字段；若被 capability 阻断，还会写 `live_route_status`、`live_route_reason`、`venue`、`venue_blockers`。
- 出口门控每次 `exit_check` 会：
  - 先检查 `state/tv_basis_arb_recovery.json`，只要存在 `needs_recovery` 状态就返回 `recovery_required`，不会再做行情调用。
  - 如果头寸还在 `open_hedged`，根据 `runtime_policy.exit_basis_bps`（4 bps）或 `holding_time_seconds >= max_holding_seconds`（3600s）判断是否 `should_exit`。
  - 出口 artifact 写入 `basis_bps`、`should_exit`、`reasons`、`close_reason` 和最新快照；如果决定退出就调用执行器出场。

## 状态账本（State Ledger）说明
- 类：`system/scripts/tv_basis_arb_state.py` 中的 `TvBasisArbStateLedger`。它维护三张 JSON：
  1. `output/state/tv_basis_arb_idempotency.json`（`attempts` 键）——每次 entry/exit 执行的幂等记录，包含 leg、status、idempotency_key、时间戳。
  2. `output/state/tv_basis_arb_positions.json`（`positions` 键）——活仓/平仓状态和 `status`：`entry_pending`→`open_hedged`→`exit_pending`→`closed`，并持久化 `target_base_qty` / `max_quote_budget_usdt`；`needs_recovery` 会插入特殊状态并阻挡新 entry。
  3. `output/state/tv_basis_arb_recovery.json`（`recoveries` 键）——当 `perp_short`、`exit_close` 拒单或传输模糊时会写 `recovery_reason`、`recovery_action`、`failure_phase`，并保留对应的 `target_base_qty` / `max_quote_budget_usdt`，直到人工/脚本确认 `close_reason` 才改成 `closed`。
- 每次调用会更新 `updated_at_utc`，可以用 `jq`/`cat` 追踪最新 `position_key`。
- Ledger 在 entry 前调用 `_get_attempt_or_none`，避免重复下单；恢复状态会阻止新的 entry，直到 `recovery.status` 变更不是 `needs_recovery`・或人为清理文件。

## 恢复路径说明
- `TvBasisArbExecutor` 在 `execute_entry`/`execute_exit` 里用 `RunHalfhourMutex` 保护 `state/run-halfhour-pulse.lock`，失败入 `needs_recovery` 的场景包括：
  - Spot 买入成功但 perp 卖空被拒（`perp_short_rejected` 或 `transport_ambiguous`）。
  - 退出时 perp close 或 spot sell 被拒，分别产出 `perp_close_*` 或 `spot_sell_*` 的恢复记录。
  - 任何 `needs_recovery` 会写入 `state/tv_basis_arb_recovery.json` 并通过 `exit_check` 以 `recovery_required` 形式曝光；同时 `output/review` 会生成 `*_gate.json`、`*_signal.json`、`closeout_artifact` 路径。
- base-qty 驱动只把“数量/预算不可执行”的风险前移到 gate，并不会消除真实执行风险；任何 partial fill、reject、transport ambiguity 仍然必须进 `needs_recovery`。
- 恢复后续：人工确认该 `position_key` 需要再次开/平时，掀起 `exit_check`（避免重复 entry 直到确定 `recovery.status` 变更）；可直接编辑 `state/tv_basis_arb_recovery.json` 以非 `needs_recovery` 状态后再次 entry。

## 烟雾命令（Smoke commands）
### 本地 webhook 服务器（演示和调试）
```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
PYTHONPATH=./system/scripts python3 - <<'PY'
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from tv_basis_arb_webhook import handle_webhook

OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(exist_ok=True)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        try:
            result = handle_webhook(payload, output_root=OUTPUT_ROOT)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode("utf-8"))

HTTPServer(("127.0.0.1", 8787), Handler).serve_forever()
PY
```
这个服务器会直接写 `output/review/tv_basis_arb`、`output/state/*`，`TvBasisArbExecutor` 默认会尝试从 `BINANCE_API_KEY` / `BINANCE_SECRET_KEY` 读取凭据；测试时可以用 `monkeypatch` 或手动替换 `spot_client`/`perp_client`。

默认 runtime policy 会按本地策略合同使用：

- `target_base_qty = 0.002 BTC`
- `max_quote_budget_usdt = 160.0`
- `requested_notional_usdt = 160.0`（仅作为预算 ceiling 兼容字段）

### curl 示例（entry_check）
```bash
curl -sS http://127.0.0.1:8787 \
  -H 'Content-Type: application/json' \
  -d '{"strategy_id":"tv_basis_btc_spot_perp_v1","symbol":"BTCUSDT","event_type":"entry_check","tv_timestamp":"2026-03-26T12:30:00Z"}'
```
响应中包含 `gate_artifact_path` 与 `execution_artifact_path` 路径，可用 `cat` 检查 `output/state` 中的 `tv_basis_arb_idempotency.json`、`tv_basis_arb_positions.json`、`tv_basis_arb_recovery.json`。

如果 gate 因 exchange constraints 或 budget 被阻断，应看到：

- `status = gate_blocked`
- `execution = null`
- `reasons` 中出现 `quote_budget_exceeded` / `perp_min_qty_unmet` / `perp_min_notional_unmet` 等 fail-close reason

如果 gate 因 venue capability 被阻断，应看到：

- `status = gate_blocked`
- `gate.live_route_status = live_blocked`
- `gate.live_route_reason = venue_capability` / `venue_capability_missing` / `venue_capability_stale` / `venue_capability_incomplete` / `venue_capability_unknown`
- `gate.venue = binance`
- `gate.venue_blockers` 中出现如 `enableFutures=false`

## 运行审计要点
- 浏览 `output/review/tv_basis_arb` 下的 `*_gate.json`/`*_signal.json`，确认每条 signal 记载的 `snapshot_ts_utc` 与 `reasons`。
- `TvBasisArbExecutor` 记录 `entry_orders`/`exit_orders`，需要把 `spot_leg` / `perp_leg` 的 `status` 和 `filled_qty` 与 Binance API 返回值对齐。
- `state/run-halfhour-pulse.lock` 反映最近一次入锁时间，运行失败可通过 `tail -n 20 output/state/run-halfhour-pulse.lock` 查看。

> 注意：套利成功 ≠ infra canary 成功 ≠ strategy ticket ready，infra canary 只是 infra plumbing 检查，策略就绪还要等 gate、状态账本、恢复检查全部干净。

> 额外注意：即使本地/云端 dry acceptance 通过，真实 acceptance 仍受 `output/state/venue_capabilities.json` 裁决约束。对于当前 Binance futures 路径，只要 blocker 仍是 `enableFutures=false`，就应明确视为 `live_blocked`，而不是 real-ready。
