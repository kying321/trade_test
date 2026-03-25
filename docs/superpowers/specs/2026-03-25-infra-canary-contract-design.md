# Fenlie Infra Canary Contract 设计

- 日期：2026-03-25
- 模式：architecture_review
- 推荐 change class：LIVE_EXECUTION_PATH
- 当前文档阶段：设计 / 未实现
- 范围：定义一条独立于策略 ticket 的真实云端执行 canary 合同；不在本设计文档中直接修改 live order 代码

## 1. 背景

Fenlie 当前已经具备一条可用的云端现货接管路径：

- 远端账户、凭证、余额、基础配置已验证可用
- `openclaw_cloud_bridge.sh live-takeover-ready-check` 可以稳定返回 `ready=true`
- `binance_live_takeover.py` 已具备：
  - `run-halfhour-pulse` mutex
  - `<=5000ms` timeout
  - idempotency
  - 订单执行与 telemetry 记录

但当前真实 `canary` 仍绑定在策略 ticket 上，导致“链路探活”与“策略可交易”混在同一 authority 路径里。

本轮 fresh 云端证据已经表明：

1. source freshness 问题已修复  
   云端 `crypto_shortline_signal_source`、`signal_to_order_tickets` 已从 2026-03-18 刷到 2026-03-25。

2. 当前阻断已收敛为 fresh ticket constraint  
   最新 blocked candidate（ETHUSDT）已是当天信号，但仍被以下理由阻断：
   - `confidence_below_threshold`
   - `convexity_below_threshold`
   - `route_not_setup_ready`
   - `size_below_min_notional`

3. 微账户与 current sizing contract 存在结构性冲突  
   当前云端现货账户 `equity_usdt ≈ 10`，而：
   - `max_alloc_per_trade_pct = 0.3`
   - `min_notional_usdt = 5.0`

   这意味着单笔策略票理论最大配置约 `3 USDT`，天然小于最小名义金额 `5 USDT`。  
   因此当前 strategy-first contract 下，微账户 canary 几乎不可能出现 actionable ticket。

结论：如果用户要做**真实资金 canary**，就必须在“策略 authority”之外，新增一条**纯执行基础设施 authority** 路径。

## 2. 目标

设计一个独立的 `infra_canary_contract`，用于验证：

1. 云端账户可以真实下单
2. 现货现买现卖的 send / ack / fill / close 路径可用
3. idempotency / mutex / timeout / panic-close 机制可用
4. 真实成交审计、dust 残留、预算消耗都能被 source artifact 记录

### 2.1 非目标

本合同**不**用于：

- 证明策略已 ready
- 替代 `signal_to_order_tickets`
- 绕过策略 confidence / convexity / route gate 并把结果写回策略 authority
- 作为多品种套利执行框架
- 直接升级为正式 live 交易模式

## 3. 已比较方案

### 方案 A：在现有 `binance_live_takeover.py` 中增加 `--infra-canary`

让当前脚本支持一条分支：

- 跳过策略 ticket 选择
- 固定 symbol / fixed quote
- 仍复用现有 execution plumbing

**优点**
- 改动最小
- 复用现有 client / telemetry / order log

**缺点**
- 同一脚本里混入两种 authority
- 容易把 infra canary 与 strategy canary 混淆
- 审计边界不够清晰

### 方案 B：新增独立 `binance_infra_canary.py`（推荐）

新增一条独立入口，只负责基础设施 canary：

- 独立 artifact
- 独立 ledger
- 独立 budget
- 不读取策略票面 authority

**优点**
- authority 最干净
- 最符合 source-of-truth 分层
- 最适合未来扩展成独立 live-infra lane

**缺点**
- 实现文件多于方案 A
- 需要新增 bridge 动作与测试

### 方案 C：伪造固定 canary ticket 走原有链

人为构造一张“固定 BTCUSDT 5 USDT canary ticket”，再走原有 `signal_to_order_tickets -> ready-check -> canary` 逻辑。

**优点**
- 表面复用最多

**缺点**
- 直接污染 source authority
- 把 infra 意图伪装成 strategy ticket
- 最难审计，最容易留下长期歧义

### 结论

选择 **方案 B**。

原因：

1. 需要明确把“执行基础设施验证”与“策略可交易性验证”拆开
2. 需要最清晰的 authority 边界
3. 未来 TradingView / webhook / 套利扩展也更需要一个独立 execution actor，而不是继续污染 strategy canary

## 4. 核心设计原则

### 4.1 Authority 分层不可反转

`infra_canary_contract` 只证明：

- execution actor 可用
- 真实下单与回平链路可用
- 风控/幂等/互斥/超时机制可用

它**不能**：

- 证明某个 symbol 的策略 edge 成立
- 提升 `signal_to_order_tickets` 的 authority
- 让 dashboard / brief / operator consumer 误把 canary 执行成功解读为策略 ready

### 4.2 只允许固定、可审计、低自由度参数

默认固定：

- `market = spot`
- `symbol = BTCUSDT`
- `quote_usdt = 5`
- `mode = round_trip`
- `daily_budget_cap_usdt = 20`

不允许：

- 任意 symbol
- 任意 quote
- 任意自动放量

### 4.3 Dust 允许但必须显式记录

由于 spot 买卖和手续费可能留下残余 `BTC dust`，合同允许：

- canary 成功但残留 dust

前提是：

- dust 数量被完整记录
- 不产生 source ambiguity
- 不让系统把 dust 当成主动持仓

### 4.4 自动触发允许，但必须仍受硬 guard 约束

用户明确允许：

- `ready-check`
- `autopilot`

自动触发 `infra canary`

但自动触发仍必须受以下 guard 限制：

- 日预算硬上限
- idempotency
- mutex
- panic / open exposure ambiguity fuse

### 4.5 失败分级必须保持保守

只有“执行状态不明 / source ambiguity”才允许进入 panic/halt 级别。  
余额不足、预算耗尽、dust 残留、exchange reject 仅能 degrade/report。

## 5. 合同定义

### 5.1 输入 authority

允许读取：

- 账户 ready / credential ready
- 余额 / account overview
- panic state
- open exposure summary
- 当日 canary budget ledger
- 当前 canary idempotency ledger

禁止读取作为准入条件：

- `signal_to_order_tickets`
- `remote_ticket_actionability_state`
- strategy confidence / convexity
- route setup readiness

### 5.2 固定执行参数

默认值：

- `market = spot`
- `symbol = BTCUSDT`
- `quote_usdt = 5`
- `round_trip = true`
- `allow_dust = true`
- `auto_trigger = allowed`
- `daily_budget_cap_usdt = 20`

### 5.3 执行流程

`infra_canary_run`：

1. 进入 `run-halfhour-pulse` mutex
2. 读取并校验：
   - credentials
   - account overview
   - panic state
   - open exposure ambiguity
   - 当日预算
   - idempotency ledger
3. 生成独立 `infra_canary_run_id`
4. 按固定参数下 `BUY BTCUSDT spot 5 USDT`
5. 等待 ack / fill
6. 读取实际买入数量
7. 用实际可卖数量下 `SELL`
8. 等待 ack / fill
9. 读取剩余 dust / fee / fill summary
10. 写独立 artifact / ledger / markdown summary

### 5.4 成功判定

以下同时满足才视为 success：

1. buy ack / fill 成功
2. sell ack / fill 成功
3. 无 source ambiguity
4. 剩余仓位为 0 或仅剩 dust
5. 审计 artifact 落盘完整

### 5.5 失败分级

#### degrade / report-only

- 账户余额不足
- 日预算超限
- 交易对不可下单
- exchange reject
- dust 残留
- autopilot guard 未满足

#### panic / halt-worthy

仅在：

- buy / sell ack 状态不明
- fill 与订单状态不一致
- 回平后持仓状态不一致
- transport 失败导致执行状态不可裁决

### 5.6 Source artifacts

建议新增：

- `output/review/*_infra_canary_run.json`
- `output/review/*_infra_canary_run.md`
- `output/review/*_infra_canary_run_checksum.json`

状态文件：

- `output/state/infra_canary_idempotency.json`
- `output/state/infra_canary_budget.json`

其中 `infra_canary_budget.json` 至少记录：

- date
- total_quote_spent_usdt
- run_count
- remaining_budget_usdt

## 6. 桥接入口设计

建议在 `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/openclaw_cloud_bridge.sh` 中新增：

- `infra-canary-probe`
- `infra-canary-run`
- `infra-canary-autopilot`

行为：

### 6.1 `infra-canary-probe`

只校验：

- 凭证
- 账户
- panic
- 预算
- idempotency readiness

不下单。

### 6.2 `infra-canary-run`

真实执行一次 round-trip。

### 6.3 `infra-canary-autopilot`

先跑 probe；若满足条件，则自动触发一次 run。  
即使用户允许自动触发，也不得绕过预算与 idempotency。

## 7. 建议文件

新增：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_infra_canary.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_binance_infra_canary.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/INFRA_CANARY_CONTRACT.md`

修改：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/openclaw_cloud_bridge.sh`

非目标：

- 本轮不修改 TradingView / webhook / 套利模块

## 8. 与现有系统的关系

### 8.1 与 `binance_live_takeover.py`

现有 `binance_live_takeover.py` 仍保留 strategy-first 语义。  
它继续依赖 `signal_to_order_tickets` 和 `risk_guard`。

### 8.2 与 dashboard / operator brief

consumer 需要能区分：

- `strategy canary blocked`
- `infra canary succeeded`

不能把后者解读成：

- strategy promotion ready
- live strategy gate cleared

### 8.3 与未来 TradingView / 套利扩展

未来如果引入：

- TradingView webhook signal actor
- multi-symbol arbitrage lane

都可以复用 `infra_canary_contract` 的基础 execution actor / order plumbing，但不能反过来污染当前 contract 的 authority 定义。

## 9. 风险与取舍

### 主要收益

- 解决“微账户无法出 actionable strategy ticket，但仍想验证 execution chain”的实际问题
- 保持 strategy-first contract 不被破坏
- 给未来 webhook / 套利 execution actor 留出清晰入口

### 主要风险

- 这是明确的 `LIVE_EXECUTION_PATH` 变更
- 自动触发意味着需要更严格的预算与幂等控制
- 若 artifact 命名、consumer 展示、operator brief 分层不清晰，容易被误读成“策略 ready”

## 10. 推荐实施顺序

1. 先落 spec
2. 再写 implementation plan
3. 再实现最小 `infra-canary-probe/run`
4. 通过测试与 dry audit 后，再决定是否开放 `autopilot`

## 11. 当前设计结论

在当前 fresh 云端证据下，**新增独立 `infra_canary_contract` 是合理的**。  
它不解决策略本身的 blocked ticket，但能把：

- execution infrastructure 验证
和
- strategy ticket readiness

两条 authority path 明确拆开。
