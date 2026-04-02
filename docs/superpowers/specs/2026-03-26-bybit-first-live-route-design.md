# Fenlie Bybit-First 替代 Venue Live 主线设计

- 日期：2026-03-26
- 模式：architecture_review
- 推荐 change class：LIVE_EXECUTION_PATH
- 当前文档阶段：设计 / 待实现
- 范围：在已完成的 venue capability boundary 之上，为 `tv_basis_btc_spot_perp_v1` 设计第一条可真实落地的替代 venue live 主线，默认目标为 Bybit；本设计不实现多 venue 路由器

## 1. 背景

当前 `tv_basis_btc_spot_perp_v1` 已具备：

- `base-qty + budget` 合同
- 本地 broad pytest
- 云端 dry acceptance
- Binance same-venue futures 路径的 source-owned capability boundary

同时，当前 Binance 真实 blocker 已明确：

- `spot` signed 能力可用
- `futures` signed 能力不可用
- `enableFutures=false`
- 该路径已被正确 fail-close 为 `live_blocked`

这意味着继续在 Binance futures 路径上投入 live 验证时间收益很低。  
下一步更合理的方向是：在不推翻当前 `tv_basis_*` 主链的前提下，接入第一条**真正可 live 的替代 venue 主线**。

仓库现状显示：

- 已有 `Bybit public provider`
- 尚无 Bybit live execution actor
- 刚完成的 venue capability boundary 正适合作为替代 venue 的上游 gate

因此，本设计默认选择 **Bybit-first**。

## 2. 目标

新增一条最小可用的 Bybit same-venue live 主线，用于承接当前 Binance futures 被阻断的 `tv_basis_btc_spot_perp_v1`。

目标包括：

1. 保留 `tv_basis_*` 现有策略主链，不重做策略层
2. 新增 Bybit 的 capability / readiness / execution 接口
3. 支持 `BTCUSDT` same-venue `spot long + perp short`
4. 能完成：
   - local tests
   - dry acceptance
   - readiness check
   - 到达 `canary-ready` 完成态

## 3. 非目标

本设计不做：

- 多 venue 自动路由器
- 同时接多个替代 venue
- 多策略统一 execution framework 大重构
- 跨交易所套利
- 多品种一起 live rollout

## 4. 方案比较

### 方案 A：Bybit-first，最小可用 live 主线（推荐）

只为 Bybit 增加：

- capability writer
- signed spot/perp client
- same-venue execution adapter
- `tv_basis` route 切换点

**优点**

- 范围最小
- 最快到达第一条可实盘验证的替代路径
- 最大化复用当前 `tv_basis_*` 和 venue capability boundary

**缺点**

- 先是单 venue 定制，不够抽象

### 方案 B：先做 venue-agnostic connector 抽象，再接 Bybit

**优点**

- 长期结构最好

**缺点**

- 当前范围明显变大
- 会拖慢第一条替代 live 主线落地

### 方案 C：先做半自动 venue 切换

只生成 ticket / venue-ready signal，不直接自动下单。

**优点**

- 风险最低

**缺点**

- 不满足当前“替代 venue live 主线”目标

### 结论

选择 **方案 A：Bybit-first，最小可用 live 主线**。

## 5. 核心设计

### 5.1 继续复用 venue capability boundary

Bybit 不能绕过刚建立的 capability boundary。  
与 Binance 一样，Bybit 也必须继续写入**同一个**：

- `output/state/venue_capabilities.json`

不是新增第二个 writer，而是**扩展现有 single-writer / single-schema 合同**。  
最小新增为：

```json
{
  "schema_version": 1,
  "venues": {
    "bybit": {
      "checked_at_utc": "...",
      "account_scope": "openclaw-system:...",
      "status": "live_ready | live_blocked | unknown",
      "spot_signed_read_status": "ready | blocked | unknown",
      "spot_signed_trade_status": "ready | blocked | unknown",
      "futures_signed_read_status": "ready | blocked | unknown",
      "futures_signed_trade_status": "ready | blocked | unknown",
      "ip_restrict": false,
      "blockers": [],
      "raw": {}
    }
  }
}
```

实现约束：

- 继续复用现有单一 writer：
  - `system/scripts/build_venue_capability_artifact.py`
- 不允许新增并行的 Bybit-only capability writer 文件
- freshness / TTL / schema_version / fail-closed 规则必须与既有 boundary 完全一致

### 5.2 `tv_basis` 不再默认绑定 Binance

当前 `tv_basis_btc_spot_perp_v1` 不应继续把 Binance 视为默认 live route。  
最小修订方向：

- 增加一个明确的 venue 选择字段，例如：
  - `preferred_venue = bybit`

策略层继续只表达：

- symbol
- base qty
- budget
- gate
- executor contract

venue-specific 差异下沉到 adapter。

关键约束：

- 第一阶段采用 **hard-switch 到 Bybit** 的单一语义
- 不做“Binance blocked 时再 fallback Bybit”的隐式多 venue router
- 对 `tv_basis_btc_spot_perp_v1`，live candidate 就是：
  - `preferred_venue = bybit`

### 5.3 新的 Bybit live 组成

最小实现建议拆成三个部件：

1. **Bybit live common**
   - signed REST client
   - spot account / perp account / restrictions / exchange info
   - order placement helper

2. **Bybit capability extension**
   - 扩展现有 `build_venue_capability_artifact.py`
   - 只读探测 Bybit spot/perp signed 能力
   - 写入同一个 `venue_capabilities.json`

3. **Bybit basis execution adapter**
   - 让 `tv_basis` 能在 `preferred_venue=bybit` 时执行：
     - spot buy
     - perp short
     - exit close

### 5.4 与当前 `tv_basis_*` 的关系

不改变：

- `tv_basis_arb_gate.py` 的核心策略条件语义
- `tv_basis` 的主合同（`0.002 BTC + 160 USDT`）

最小改变：

- `tv_basis_arb_webhook.py` 在 live 路径**先决定 venue，再走该 venue 的 market snapshot / gate / executor**
- `tv_basis_arb_state.py` / 相关 artifact 必须持久化实际执行归属，例如：
  - `execution_venue = bybit`
- execution 层使用 Bybit adapter，而不是继续假设 Binance

这样 exit / recovery / audit 才能知道后续应继续使用哪套 client/adapter。

### 5.5 第一阶段只支持一个策略

为避免 scope 膨胀，Bybit-first live 主线第一阶段只支持：

- `tv_basis_btc_spot_perp_v1`
- `BTCUSDT`

不同时扩 ETH/SOL/BNB。

## 6. 数据流

1. TradingView webhook 触发 `tv_basis_btc_spot_perp_v1`
2. 根据策略合同先解析：
   - `preferred_venue = bybit`
3. 读取 `venue_capabilities.json` 中的 `bybit`
4. 若 Bybit capability 不是 `live_ready`，则 route fail-close
5. 若 Bybit capability 就绪，再走 Bybit 版 market snapshot / strategy gate
6. gate 通过后，走 Bybit live adapter
7. 继续写回：
   - review artifacts
   - position / recovery ledger

## 7. 验收顺序

必须按以下顺序：

1. Bybit capability extension 本地单测
2. Bybit signed client / adapter 单测
3. `tv_basis` route 切换单测
4. dry acceptance
5. readiness check

本 spec 的完成态停在：

- `canary-ready`

也就是：

- Bybit capability = `live_ready`
- route 切换完成
- dry acceptance 通过
- readiness check 通过

**单次真实 canary acceptance 不属于本 spec 的完成判定**，它应作为后续 operator 决策与执行阶段。

### 7.1 readiness check 的唯一入口

为避免 planning 时再次出现验收口径漂移，Bybit-first 主线必须定义一个唯一 readiness check 入口。  
最小建议：

- `system/scripts/bybit_live_route_ready_check.py`

职责：

- 只读校验 Bybit live 主线是否达到 `canary-ready`
- 不下单
- 只读取 source-owned artifacts 和只读 account/market 信息

### 7.2 readiness check 输出 artifact

readiness check 必须写一个明确的 source-owned artifact，例如：

- `output/review/latest_bybit_live_route_ready_check.json`

最小字段：

```json
{
  "ok": true,
  "status": "canary_ready | blocked",
  "venue": "bybit",
  "strategy_id": "tv_basis_btc_spot_perp_v1",
  "required_checks": {
    "venue_capability_ready": true,
    "route_selected_bybit": true,
    "baseqty_budget_contract_ready": true,
    "no_active_recovery": true,
    "account_ready": true
  },
  "blockers": []
}
```

### 7.3 `canary-ready` 的机械判定

`canary-ready` 不能由 dashboard / operator brief 再次推导。  
它必须来自明确 artifact。

本设计选择：

- **以 readiness check artifact 作为唯一判定来源**

当且仅当以下条件全部满足时：

- `ok = true`
- `status = canary_ready`
- `required_checks.venue_capability_ready = true`
- `required_checks.route_selected_bybit = true`
- `required_checks.baseqty_budget_contract_ready = true`
- `required_checks.no_active_recovery = true`
- `required_checks.account_ready = true`

才可认为该主线达到 `canary-ready`。

## 8. 风险

### 8.1 主要风险

- Bybit account/region/API 能力未确认
- Bybit spot/perp 合约最小下单约束可能与 Binance 不同
- 若把“替代 venue 接入”和“统一抽象重构”绑在一起，范围会失控

### 8.2 风险控制

- 第一阶段只做 Bybit
- 第一阶段只做 `BTCUSDT`
- 继续复用 capability boundary
- 先 dry，再 readiness，再单次 real acceptance

## 9. 验收标准

以下全部满足才算本设计落地完成：

1. Bybit 能写出 source-owned capability artifact
2. `venue_capabilities.json` 扩展后保留既有 `binance` 条目，并新增 `bybit` 条目；不允许误写成 bybit-only payload
3. `tv_basis_btc_spot_perp_v1` 可在 venue 选择上切到 Bybit
4. Bybit same-venue `spot + perp` 有最小 live adapter
5. local tests / dry acceptance / readiness check 全绿
6. `output/review/latest_bybit_live_route_ready_check.json` 可作为唯一 source-owned 产物，明确判断：`tv_basis_btc_spot_perp_v1` 已到达 `canary-ready`

## 10. 回滚

若替代 venue 接入失败，应能回到：

- Binance 路径继续 `live_blocked`
- `tv_basis` 仍可 dry / replay / fake executor
- 不允许因为替代 venue 失败而破坏已有的 Binance fail-close boundary
