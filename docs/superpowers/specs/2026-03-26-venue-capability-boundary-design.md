# Fenlie Venue Capability Boundary 设计

- 日期：2026-03-26
- 模式：architecture_review
- 推荐 change class：LIVE_GUARD_ONLY
- 当前文档阶段：设计 / 待实现
- 范围：冻结 Binance 合约实盘线，并为后续替代 venue 接入建立统一的 venue capability boundary；本设计不实现新的实盘 connector

## 1. 背景

当前 `tv_basis_btc_spot_perp_v1` 已完成：

- `base-qty + budget` 合同修订
- 本地 broad pytest
- 云端 dry acceptance

但真实 acceptance 被交易所账户能力阻断。远端只读诊断已确认：

- spot signed API 正常
- `apiRestrictions.enableFutures = false`
- futures signed API 返回 `401 / -2015`
- `ipRestrict = false`
- daemon env 与 `~/.openclaw/binance.env` 的 key 指纹一致

所以当前 blocker 已不在代码层，而在 **venue/account capability** 层。

如果策略层继续默认把 “代码可执行” 等同于 “venue 可实盘”，会产生两个问题：

1. operator 会误把 dry-valid 路径当成 live-ready；
2. 后续替代 venue 接入时，策略又会重新绑死在 Binance futures 能力假设上。

本设计的目标是把这种“交易所能力可用性”前移为一个 source-owned boundary。

## 2. 目标

建立一个最小、可审计、source-owned 的 venue capability boundary，用来回答：

1. 某个 venue / account / route 当前是否具备 live 能力；
2. 该能力是否支持：
   - 现货
   - 合约 / perp / futures
   - signed read
   - signed trade
3. 当前策略应该被归类为：
   - `dry_only`
   - `live_blocked`
   - `live_ready`

## 3. 非目标

本设计不做：

- 替代 venue 的实盘 connector
- 多 venue 路由器
- 统一订单抽象层大重构
- 新的实盘验收流程
- 自动修复交易所权限

## 4. 方案比较

### 方案 A：继续把能力判断散落在各脚本里

例如：

- `tv_basis_arb_webhook.py` 自己判断 Binance futures 能不能用
- `binance_live_takeover.py` 自己再维护另一套 ready-check

**优点**

- 表面上改动最少

**缺点**

- 同一类 blocker 会在多个脚本里重复表达
- source-of-truth 不清晰
- 不利于替代 venue 接入

### 方案 B：新增单一 venue capability artifact（推荐）

新增一个 source-owned artifact，由诊断脚本或 guard runner 写入，策略和 runbook 只读它。

**优点**

- authority 清晰
- operator 容易审计
- 后续替代 venue 接入能复用同一边界

**缺点**

- 需要补一个新的 guard artifact 生成步骤

### 方案 C：直接跳到替代 venue 接入，不先做能力边界

**优点**

- 更快进入新 venue 主线

**缺点**

- Binance 这次暴露出的能力边界问题会在新 venue 再出现一遍
- 策略层仍然缺少统一 live capability source

### 结论

选择 **方案 B**。

## 5. 核心设计

### 5.1 新的 source-owned artifact

新增一个统一能力产物，例如：

- `output/state/venue_capabilities.json`

最小结构：

```json
{
  "schema_version": 1,
  "venues": {
    "binance": {
      "checked_at_utc": "2026-03-26T04:40:32Z",
      "account_scope": "openclaw-system:daemon_env",
      "status": "live_blocked",
      "spot_signed_read_status": "ready",
      "spot_signed_trade_status": "ready",
      "futures_signed_read_status": "blocked",
      "futures_signed_trade_status": "blocked",
      "ip_restrict": false,
      "blockers": ["enableFutures=false"],
      "raw": {
        "apiRestrictions": {
          "enableFutures": false
        }
      }
    }
  }
}
```

枚举约束：

- artifact 中统一使用小写状态值：
  - `dry_only`
  - `live_blocked`
  - `live_ready`
  - `unknown`
- 不允许同时混用 `DRY_ONLY` / `LIVE_BLOCKED` 这类大写枚举。

capability 字段类型约束：

- 不再使用布尔型 `spot_signed_trade=true/false`
- 统一使用字符串状态字段：
  - `spot_signed_read_status`
  - `spot_signed_trade_status`
  - `futures_signed_read_status`
  - `futures_signed_trade_status`

这些 capability status 字段只允许：

- `ready`
- `blocked`
- `unknown`

说明：

- `unknown_stale` 只是 consumer 侧 reason / 派生态，**不是** source artifact 的 schema 枚举值；
- source artifact 本身只落 `ready|blocked|unknown`，stale 由 consumer 用 `checked_at_utc + max_age_seconds` 计算。

### 5.1.1 Freshness / TTL 约束

capability artifact 必须带 freshness contract。  
最小实现建议：

- `max_age_seconds = 900`（15 分钟）

consumer 在读取 `checked_at_utc` 后必须执行：

1. 若 artifact 不存在 -> 视为 `unknown`
2. 若 `checked_at_utc` 缺失 / 非法 -> 视为 `unknown`
3. 若当前时间 - `checked_at_utc` > `max_age_seconds` -> 视为 `unknown_stale`

关键约束：

- stale / unknown **不得**被解释成 `live_ready`
- stale / unknown 必须 fail-close 到：
  - `live_blocked`（若当前路径是 live gate）
  - 或 `dry_only`（若当前路径是 operator-only / dry-only lane）

### 5.1.2 单一 writer 约束

最小实现阶段只允许一个 writer：

- `system/scripts/build_venue_capability_artifact.py`

其他 runner / guard / bridge 脚本只能：

- 触发刷新
- 或消费 artifact

但不能各自直接写同一路径，避免出现多 writer 抢写和 schema 漂移。

### 5.2 策略层消费方式

像 `tv_basis_btc_spot_perp_v1` 这类 same-venue `spot + perp` 路径，在进入 live execution 前必须读取该 artifact。

若需要：

- `spot_signed_trade_status = ready`
- `futures_signed_trade_status = ready`

但 artifact 显示对应能力缺失，则策略应直接判定：

- `live_route_status = live_blocked`
- `live_route_reason = venue_capability`

而不是进入 executor 再在交易所报错。

### 5.2.1 缺失 / 部分字段 / unknown 的映射

这是边界层的核心 fail-closed 规则，必须写死：

1. **artifact 缺失**
   - consumer 输出：
     - `live_route_status = live_blocked`
     - `live_route_reason = venue_capability_missing`

2. **artifact 存在但字段缺失**
   - 例如缺：
     - `status`
     - `checked_at_utc`
     - `spot_signed_trade_status`
     - `futures_signed_trade_status`
   - consumer 输出：
     - `live_route_status = live_blocked`
     - `live_route_reason = venue_capability_incomplete`

3. **artifact stale / unknown**
   - consumer 输出：
     - `live_route_status = live_blocked`
     - `live_route_reason = venue_capability_stale`
       或 `venue_capability_unknown`

4. **明确能力缺失**
   - 例如 `futures_signed_trade_status=blocked`
   - consumer 输出：
     - `live_route_status = live_blocked`
     - `live_route_reason = venue_capability`

只有在以下条件全部满足时，才允许给出 `live_ready`：

- artifact 新鲜
- schema 完整
- 所需能力字段完整
- 对应能力显式为 `ready`
- 其他策略 gate 也通过

### 5.3 能力状态分层

建议统一三档：

- `DRY_ONLY`
  - 只允许本地测试 / fake executor / replay / dry acceptance
- `LIVE_BLOCKED`
  - 代码路径具备，但 venue/account capability 不满足
- `LIVE_READY`
  - venue/account capability 满足，且策略侧其他 gate 也允许

实现约束：

- artifact 内部落盘统一用小写：
  - `dry_only`
  - `live_blocked`
  - `live_ready`
  - `unknown`
- UI / runbook 若要展示大写，只能在展示层转换，不能污染 source artifact。

### 5.4 与现有系统的关系

#### 对 `tv_basis_arb_*`

不改变：

- gate
- executor
- state

只是在 live 路径判断前新增：

- `venue capability` blocking layer

#### 对 `binance_live_takeover.py`

现有 ready-check 仍保留，但它是 Binance-specific actor。  
venue capability boundary 负责更上游、更通用的“这个 venue 的这类能力是否存在”。

#### 对 `openclaw_cloud_bridge.sh`

桥接脚本可以继续作为操作入口，但最终状态应来自 source-owned capability artifact，而不是 shell 输出文本。

## 6. 最小落地方式

### 6.1 先做只读诊断写盘

新增一个最小 runner，例如：

- `system/scripts/build_venue_capability_artifact.py`

职责：

- 只读调用当前 venue 的 signed read / restrictions API
- 写出标准化 capability artifact
- 不下单
- 对于没有直接 restrictions API 的 venue：
  - `*_signed_read_status` 可以通过 signed read 探测确认
  - `*_signed_trade_status` 默认不得通过推断直接给 `ready`
  - 若没有显式 permission API，最小实现应写成：
    - `*_signed_trade_status = unknown`
    - 并让 consumer fail-close

### 6.2 策略与 runbook 只读该 artifact

当前阶段不需要把所有 live actor 都重构成统一框架。  
只要保证：

- `tv basis`
- operator/runbook
- 后续替代 venue design

都读同一个 capability source 即可。

## 7. 对当前 Binance 合约线的裁决

本设计落地后，Binance same-venue futures 主线应明确标记为：

- `live_blocked`
- blocker=`enableFutures=false`

且在 operator / review / runbook 中明确写明：

- 当前仅 `dry-valid`
- 不应继续做 Binance futures 实盘 acceptance

## 8. 对替代 venue 的意义

后续如果切到新 venue，这层边界可以复用：

- 同样写 capability artifact
- 同样输出：
  - spot 是否可用
  - perp/futures 是否可用
  - signed trade 是否可用
  - blocker 是什么

这样替代 venue 的 connector 只需要关注“如何下单”，不再背负“如何解释账户权限状态”。

## 9. 验收标准

以下全部满足才算本设计落地完成：

1. 有统一的 source-owned venue capability artifact
2. Binance 当前状态能稳定写成：
   - `spot signed ok`
   - `futures signed blocked`
   - blocker=`enableFutures=false`
3. `tv_basis_btc_spot_perp_v1` 不再把 Binance futures 误判成 live-ready
4. runbook/operator 文案明确区分：
   - `dry_only`
   - `live_blocked`
   - `live_ready`
5. 为后续替代 venue 预留统一能力接口，但不要求本次接入新的 live connector

## 10. 风险与回滚

### 10.1 风险

- 如果 capability artifact 不是 source-owned，而只是 CLI 文本，后续仍会回到多处重复判断；
- 如果过早把它做成“大一统 execution abstraction”，会扩大范围并拖慢主线。

### 10.2 回滚

即使回滚实现，也不应回到“策略默认假设 Binance futures 可 live”的状态。  
最差也应保留显式 blocker 文案与 dry-only 裁决。
