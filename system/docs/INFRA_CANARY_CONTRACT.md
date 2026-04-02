# Infra Canary Contract（基础设施演练合同）

## Authority boundary（权限边界）
- 只允许读取：账户 ready / credentials / panic 状态 / open exposure 摘要 / infra canary 预算和幂等 ledger。
- 禁止触碰任何策略票的 authority：不读 `signal_to_order_tickets`、convexity/confidence、route gate、remote_ticket_actionability、策略 sizing 结果。
- 只验证 execution infrastructure：order flow、mutex、timeout、buffer ledger、dust residual，都由独立 `infra_canary_run` artifact 记录。

## 固定参数与默认值
- symbol：`BTCUSDT`
- market：`spot`
- round_trip：`true`（买入 + 成交 + 卖出）
- requested quote：`10 USDT`
- single_run_cap_usdt：`12`
- effective quote：按交易所 `min_notional`、步长、手续费损耗计算“最小可安全回平金额”，并取 `max(requested_quote_usdt, required_round_trip_quote_usdt)`
- daily_budget_cap_usdt：`20`
- allow_dust：`true`

这些默认参数不随策略票变化，保持可审计、低自由度，防止 infra canary 越权。

若 `required_round_trip_quote_usdt > single_run_cap_usdt`：
- 直接 `graceful skip`
- 不下单
- 记录结构化 skip reason

## Success criteria（成功判定）
1. buy order ack/filled，实际买入数量被记录。
2. sell order ack/filled，最终持仓归零或仅剩 dust。
3. open exposure/transport/ack 状态没有 source ambiguity，order log 和 artifact 完整。
4. 预算、幂等、mutex 和 panic guard 都在 artifact 里核对通过。
5. 输出结构化 artifact 包含 `steps`、`round_trip`、`dust`、`budget`、`idempotency`、`autopilot_allowed` 等字段。

## Failure severity（失败分级）
- **degrade/report only**：余额不足、日预算耗尽、交易对不可下单、exchange reject、dust 残留、autopilot guard 拦下。记录 reason 并退出，但不 panic/halt。
- **panic/halt**：buy 或 sell ack 状态不明、fill 与订单状态不一致、回平后仍有非 dust 仓位、transport/命令执行结果不可裁决时才可调用 panic_close_all 或 panic 级别。

## Budget / dust / idempotency
- `output/state/infra_canary_budget.json` 记录每日 quote 花费、剩余 budget、run 次数。
- `output/state/infra_canary_idempotency.json` 防止重复下单，命中时直接 `probe`/`run` 退出。
- 剩余 dust 是允许的（`allow_dust=true`），但必须写入 `dust` summary，不能视为策略持仓。
- 所有 artifact 和 telemetry 需通过 `infra_canary_run_id` 建立一次性 trace。
- 历史残留说明：若旧版固定 5 USDT canary 已留下 `needs_recovery / sell_exchange_reject` attempt，该状态仍保持 source-of-truth，不会被新版 cap/skip 原因覆盖。

## Infra Canary 与 strategy canary 的关系
- strategy canary（`binance_live_takeover.py` + `signal_to_order_tickets`）证明信号/策略 readiness；infra canary 只证明 execution infrastructure 可用。
- Infra canary 不能提升策略 authority，也不能用来绕过 strategy confidence、convexity、route gate。
- 操作人员必须在 brief/回放中明确：`infra canary 成功 ≠ strategy ready`。

## Autopilot guard
- `scripts/openclaw_cloud_bridge.sh infra-canary-autopilot` 先调用 `scripts/binance_infra_canary.py --mode autopilot-check`。
- 只有当 autopilot check 返回 `ok=true` 且 `autopilot_allowed=true` 时才继续执行 `infra-canary-run`，否则串行输出 `skipped_not_ready`/`ready_to_run` 等结构化 JSON 并优雅退出。
- `autopilot_allowed` 精确定义了 infra canary 自动落地的 authority；字段类型必须为 bool，非 bool 会视为 gate error。
- autopilot check 本身不下单，主要验证 credentials、budget/ledger、panic、open exposure 和 idempotency。
- autopilot run 仍受 daily budget/幂等/mutex 约束，永远不会因 `autopilot_allowed` 变成 true 就绕过防线。

## 问题提示
- infra canary artifact 背后仍需人类审计：即便 `infra-canary-autopilot` 说 `check_ok=true`，也不能借此判定策略 ready；策略需要独立的 `signal_to_order_tickets` 路径。
- dust 量、预算消耗、幂等状态、mutex 锁定周期都需要记录在 `steps` 区块，以便后续 operator 可追溯。
