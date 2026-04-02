# Fenlie TradingView 基差套利 BTC Base-Qty 合同修订设计

- 日期：2026-03-26
- 模式：architecture_review
- 推荐 change class：LIVE_EXECUTION_PATH
- 当前文档阶段：设计修订 / 待实现
- 范围：把 `tv_basis_btc_spot_perp_v1` 从“20 USDT quote 驱动”修订为“固定 `0.002 BTC` base-qty + `160 USDT` 预算上限”的可执行合同；本设计只定义合同、门控、执行与验收边界，不直接修改 live order 代码

## 1. 背景与修订原因

`/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/docs/superpowers/specs/2026-03-26-tradingview-basis-arbitrage-design.md`
已经定义了 TradingView 作为套利编排入口、Fenlie 本地持有 authority 的总体架构，并给出了 v1 约束：

- `strategy_id = tv_basis_btc_spot_perp_v1`
- `symbol = BTCUSDT`
- 同交易所 `spot long + perp short`
- 入口多因子门控、出口 basis 回归 + 最长持仓时间

但后续真实 readiness check 证明，原 v1 的 `max_notional_usdt = 20` 合同在 Binance `BTCUSDT` 永续侧不可执行：

1. futures signed 账户权限当前未通过（`401 / -2015`），这是账户层 blocker；
2. 即便权限修复，`BTCUSDT` 当前 Binance USDM 约束仍要求：
   - `perp min_qty = 0.001`
   - `perp step_size = 0.001`
   - `perp min_notional = 100`
3. 按当时市场价格，`20 USDT` 只能换到约 `0.000282 BTC`，远低于永续侧最小数量与最小名义；
4. 如果继续沿用“spot 先按 quoteOrderQty 买入，再用实际 filled qty 去做 perp short”的路径，系统会在真实下单阶段才触发 reject，并把套利路径打进 `needs_recovery`。

因此，问题已经不是“权限修好就能上线”，而是**原合同本身不可执行**。  
本修订设计的目标是把 `tv_basis_btc_spot_perp_v1` 收敛到一个真实可执行、审计清晰、且能在 gate 阶段 fail-close 的合同。

## 2. 设计目标

本设计只解决当前最高风险主线：

1. 保留 `BTCUSDT` 与 same-venue `spot long + perp short`；
2. 移除不可执行的 `20 USDT quote 驱动` 合同；
3. 引入固定 `base-qty` 合同，确保 spot/perp 两腿数量对齐；
4. 在 **entry gate** 阶段就把交易所最小约束与预算约束纳入 fail-close；
5. 不做多品种 / 多 venue / 动态 sizing 平台化扩展。

## 3. 已比较方案

### 方案 A1：固定 base-qty 合同（推荐）

将 `tv_basis_btc_spot_perp_v1` 定义为：

- `target_base_qty = 0.002 BTC`
- `max_quote_budget_usdt = 160.0`

gate 负责：

- basis / spread / OI
- exchange min constraints
- `0.002 BTC` 预估现货花费是否超预算

executor 负责：

- spot 按 `quantity=0.002` 买入
- perp 按 `quantity=0.002` 卖空

**优点**

- 两腿数量合同最清晰
- 最容易审计
- 最小化“spot fill qty -> perp step mismatch”结构性问题
- 对当前代码改动范围最小

**缺点**

- 单次资金预算抬高到约 `160 USDT`
- 仍需要 futures 权限修复

### 方案 A2：动态可执行 qty

gate 每次按实时价格与交易所约束计算最小可执行 base qty，再把该 qty 传给 executor。

**优点**

- 更通用，未来多品种复用更好

**缺点**

- 当前阶段复杂度过高
- live-path 变量更多
- 不适合作为第一条可执行套利链路的修复方案

### 方案 A3：继续 quote 驱动，perp 做 rounding

保留 spot `quoteOrderQty`，永续腿按 step 向上/向下取整。

**优点**

- 表面改动较小

**缺点**

- 天然制造 residual / hedge mismatch
- 容易把路径带进 `needs_recovery`
- 不符合“双腿一致性优先”的目标

### 结论

选择 **方案 A1：固定 base-qty 合同**。

## 4. 最终合同

### 4.1 策略 ID

保持不变：

- `strategy_id = tv_basis_btc_spot_perp_v1`

### 4.2 支持标的

保持不变：

- `symbol = BTCUSDT`

### 4.3 目标数量与预算

修订为：

- `target_base_qty = 0.002`
- `max_quote_budget_usdt = 160.0`

说明：

1. `0.002 BTC` 是基于 Binance USDM 当前 `min_qty=0.001 + step=0.001 + min_notional=100` 向上对齐后的最小可执行数量；
2. 按审计时点价格，对应现货侧预算下限约 `141.81 USDT`；
3. `160.0 USDT` 作为预算上限，给 spot market buy 预留波动缓冲；
4. 若后续交易所约束变化，`target_base_qty` 可能需要再修订，但 v1 先固定，不做动态 sizing。

### 4.4 兼容字段

为了减少实现面改动，兼容期内可保留：

- `max_notional_usdt`
- `requested_notional_usdt`

但语义需要统一：

- `requested_notional_usdt` 在兼容期内代表 **quote budget ceiling**
- 真实下单 authority 由 `target_base_qty` 驱动，而不是由 `requested_notional_usdt` 驱动

新 artifact / runtime policy 必须显式写出：

- `target_base_qty`
- `max_quote_budget_usdt`

避免 operator 把预算字段误当成最终下单数量 authority。

## 5. Gate 设计

### 5.1 仍保留的门控

entry gate 继续要求：

- `basis_bps >= min_basis_bps`
- `mark_index_spread_bps <= max_mark_index_spread_bps`
- `open_interest_usdt >= min_open_interest_usdt`

### 5.2 新增的执行可行性门控

entry gate 必须新增：

1. **spot 最小约束检查**
   - `target_base_qty >= spot.min_qty`
   - `target_base_qty * spot_price >= spot.min_notional`

2. **perp 最小约束检查**
   - `target_base_qty >= perp.min_qty`
   - `target_base_qty * perp_mark_price >= perp.min_notional`

3. **预算检查**
   - `estimated_quote_for_target_usdt = target_base_qty * spot_price`
   - 若 `estimated_quote_for_target_usdt > max_quote_budget_usdt`，则 fail-close

### 5.3 新增 fail-close reasons

至少新增：

- `spot_min_qty_unmet`
- `spot_min_notional_unmet`
- `perp_min_qty_unmet`
- `perp_min_notional_unmet`
- `quote_budget_exceeded`

命名约束：

- 若当前主线已经存在等价 fail-close reason，则优先复用既有命名，避免只因合同修订引入无必要的 artifact / test churn；
- 本次修订不应为了追求语义完整而重命名已经进入 gate artifact 的 reason key，除非实现上确实无法兼容。

### 5.4 Gate 输出字段

entry gate artifact 必须新增：

- `target_base_qty`
- `max_quote_budget_usdt`
- `estimated_quote_for_target_usdt`
- `estimated_perp_notional_usdt`
- `exchange_constraints`

这样审计时可以直接看到：

- 当前策略打算下多少 BTC
- 当前估算会花多少钱
- 预算是否超标
- 被哪个交易所最小约束拦下

## 6. Executor 设计

### 6.1 Entry

entry executor 改为：

1. **spot**
   - `side=BUY`
   - 使用 `quantity=target_base_qty`
   - 不再用 `quoteOrderQty`

2. **perp**
   - `side=SELL`
   - 使用同一 `target_base_qty`

### 6.2 为什么必须改成 base-qty 驱动

原实现是：

- spot 按 quote 下单
- perp 用 spot 实际 filled qty 去 short

这会导致：

- 一旦 filled qty 落在 perp step 不能接受的区域
- reject 发生在第二腿
- 系统被迫进入 `needs_recovery`

base-qty 驱动能把 authority 前移为：

- 先确定两腿共同数量
- gate 先校验其可执行性
- executor 只按已批准数量下单

### 6.3 Partial fill / transport ambiguity 约束

base-qty 驱动不会消除 live execution 风险，只是把“数量不可执行”的风险前移到 gate。  
entry / exit 仍需保留当前 recovery 语义：

- spot 部分成交但 perp 未成交 -> `needs_recovery`
- perp transport ambiguity -> `needs_recovery`
- exit 任一腿 reject / ambiguity -> `needs_recovery`

也就是说，本次修订改变的是 **下单 authority**，不是取消 recovery state machine。

### 6.4 Exit

exit 仍按当前 open 头寸中的 `filled_base_qty` 对称平仓，不需要本次重构。

## 7. 状态与 Artifact 约束

### 7.1 Position / Attempt Ledger

entry 打开头寸后，state 需要记录：

- `target_base_qty`
- `spot_leg.filled_base_qty`
- `spot_leg.filled_quote_usdt`
- `perp_leg.filled_base_qty`
- `max_quote_budget_usdt`

### 7.2 审计可追溯要求

operator 需要从 artifact 直接看出：

1. 该次策略目标数量是多少；
2. 该目标数量是否满足交易所最小约束；
3. 预算上限是多少；
4. 实际 spot 花费是否仍在预算之内；
5. 若被阻断，是被 basis/spread/OI 拦下，还是被 exchange constraints / budget 拦下。

## 8. 实现边界

### 8.1 本次会改的文件

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_common.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_gate.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_executor.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_gate.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_executor.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py`

### 8.2 本次不会改的内容

- 多 symbol 支持
- 多 venue 支持
- 动态 qty sizing
- 可视化 dashboard 扩展
- TradingView payload 扩展为自带 qty/budget authority

## 9. 验收标准

以下全部满足才算本设计落地完成：

1. `tv_basis_btc_spot_perp_v1` 合同改为 `target_base_qty=0.002 + max_quote_budget_usdt=160`
2. gate 会在 executor 之前 fail-close 不可执行数量/预算
3. executor 按固定 `0.002 BTC` 驱动两腿 entry
4. webhook gate artifact 透传新合同字段
5. 本地 broad pytest 通过
6. 云端 dry acceptance 证明：
   - 当给定不可执行约束时，不再误进 executor
   - 当给定可执行约束时，能形成一致的 base-qty entry plan（仍不在本阶段下真实单）

## 10. 真实 acceptance 前置条件

即使本修订实现完成，**仍然不能立刻做真实 acceptance**。  
至少还需要：

1. 修复远端 futures signed 权限（当前为 `401 / -2015`）
2. 确认 spot/perp 两个账户都允许 `BTCUSDT` 对应腿的真实下单
3. 确认远端资金至少覆盖：
   - `~160 USDT` 现货预算
   - 永续侧保证金与手续费余量
4. recovery ledger 为空
5. 单次真实 acceptance 仍先走 canary 思路，不放宽到连续自动执行

## 11. 风险与回滚

### 11.1 残留风险

- `0.002 BTC` 是基于当前交易所约束与价格缓冲做的 v1 固定值，若交易所约束再变，需要再次修订；
- spot market buy 按 base qty 仍可能出现轻微滑点，但预算 fail-close 已把主要风险前移；
- futures 权限问题未修复前，本设计只能完成 dry 验收，不能进入真实 acceptance。

### 11.2 回滚策略

若实现后发现问题，可以回滚到当前 quote-driven 合同版本，但前提是：

- 保留已完成的 exchange constraint fail-close 逻辑；
- 不允许再回到“20 USDT 明知不可执行却还能误进 executor”的状态。
