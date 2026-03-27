# Fenlie 事件管道博弈层扩展设计

- 日期：2026-03-27
- 模式：architecture_review
- 推荐 change class：`RESEARCH_ONLY -> LIVE_GUARD_ONLY`
- 当前文档阶段：扩展设计 / 待实现
- 范围：在不破坏当前已闭合的事件量化主线前提下，为 Fenlie 增加“大国博弈层 / 传导链 / safety margin”三层结构解释系统，使系统能更好判断当前事件严重程度、后续金融与经济演化路径，以及何时必须触发硬边界。

## 1. 背景

截至当前主线，Fenlie 已经具备：

- `event_intake`
- `event_regime_snapshot`
- `event_crisis_analogy`
- `event_asset_shock_map`
- `event_live_guard_overlay`
- dashboard / operator panel / refresh summary / `LIVE_GUARD_ONLY` 风控接线

这条主线已经能够：

- 识别事件状态
- 做历史危机类比
- 给出资产 shock map
- 把风险压缩到 `event_live_guard_overlay.json`

但当前系统仍缺少三类更底层的问题回答能力：

1. **谁在博弈，主战场是什么，策略目标是什么**
2. **风险沿哪些链路从地缘/金融/能源传到资产和经济**
3. **系统离危险边界还有多远，哪些 hard boundary 已触发**

用户明确希望：

- 增加大国博弈论和策略集
- 增加多层次架构
- 增加多因子传导链剖析
- 增加 safety margin 与 hard boundary
- 不着急上实盘，更关心严重程度与后续演化可能性

因此，本扩展设计目标不是另起一套系统，而是在当前已闭合事件管道上**补充结构层**。

## 2. 目标

在现有事件管道基础上新增三类解释层 artifacts：

1. `event_game_state_snapshot`
2. `event_transmission_chain_map`
3. `event_safety_margin_snapshot`

并回答：

1. 当前大国博弈主战场是什么；
2. 各主体的目标函数、约束条件、可行动作集是什么；
3. 风险从博弈层如何传导到美元、能源、信用、风险偏好和加密资产；
4. 当前 safety margin 是否在压缩；
5. 哪些 hard boundary 已经触发，未来如何压缩进现有 `event_live_guard_overlay`。

## 3. 非目标

本扩展设计不做：

- 新建平行事件系统；
- 直接修改 executor / order routing / capital 路径；
- 把大国博弈层直接当作交易 alpha；
- 自动恢复风险、自动放行 live；
- 复杂博弈求解器（如 Nash equilibrium / agent-based simulation）；
- 推翻当前 `event_live_guard_overlay` 作为执行层唯一 authority 的结构。

## 4. 与当前主线的兼容方式

当前已闭合事件主线必须保持稳定：

- `event_intake`
- `event_regime_snapshot`
- `event_crisis_analogy`
- `event_asset_shock_map`
- `event_live_guard_overlay`
- dashboard / operator / guard 接线

扩展层的兼容顺序固定为：

`event_intake`
-> `event_game_state_snapshot`
-> `event_transmission_chain_map`
-> `event_regime_snapshot`
-> `event_crisis_analogy`
-> `event_asset_shock_map`
-> `event_safety_margin_snapshot`
-> `event_live_guard_overlay`

其中：

- `event_game_state_snapshot / transmission_chain_map / safety_margin_snapshot` 是**解释层**
- `event_live_guard_overlay` 仍是**执行层唯一输出**

guard / runtime / execution 层继续只允许消费：

- `event_live_guard_overlay.json`

而不直接读取博弈层 artifacts。

## 5. 新增 artifacts 合同

### 5.1 `latest_event_game_state_snapshot.json`

用途：回答当前是谁在博弈、主战场是什么、当前 game state 是什么。

最小字段：

- `generated_at_utc`
- `primary_theater`
- `game_state`
- `confidence_score`
- `dominant_conflict_axes[]`
- `dominant_transmission_axes[]`
- `systemic_escalation_probability`
- `policy_relief_probability`
- `actors[]`

每个 `actors[]` 项最小字段：

- `actor`
- `theater`
- `primary_objectives[]`
- `hard_constraints[]`
- `available_actions[]`
- `observed_actions[]`
- `dominant_strategy`
- `escalation_level`
- `deescalation_probability`
- `market_impact_channels[]`

### 5.2 `latest_event_transmission_chain_map.json`

用途：回答风险沿哪条链传播、当前哪条是主链、传播有多快多强。

最小字段：

- `generated_at_utc`
- `primary_theater`
- `dominant_chain`
- `chains[]`

每个 `chains[]` 项最小字段：

- `chain_id`
- `origin`
- `intermediate_nodes[]`
- `terminal_assets[]`
- `intensity_score`
- `velocity_score`
- `confidence_score`
- `status`

`status` 枚举：

- `watch`
- `active`
- `dominant`

### 5.3 `latest_event_safety_margin_snapshot.json`

用途：回答当前离危险区还有多远、哪些硬边界已经触发。

最小字段：

- `generated_at_utc`
- `liquidity_margin`
- `credit_margin`
- `energy_margin`
- `policy_margin`
- `system_margin_score`
- `hard_boundaries`
- `boundary_reasons[]`

`hard_boundaries` 最小字段：

- `canary_hard_block`
- `new_risk_hard_block`
- `shadow_only_boundary`

### 5.4 字段约束

- 所有 `score / probability / margin` 统一 `0.0 ~ 1.0`
- 所有时间戳统一 `*_utc`，ISO8601 + `Z`
- 列表字段未知时用空列表，不用 `null`
- 状态字段必须冻结枚举，不允许自由字符串漂移

## 6. 大国博弈层主体与状态机

### 6.1 第一阶段主体集合

- `united_states`
- `china`
- `european_union`
- `russia`
- `opec_plus_gulf`
- `regional_conflict_nodes`
  - `iran`
  - `israel`
  - `ukraine`
  - `red_sea_shipping`
  - `hormuz_shipping`

### 6.2 第一阶段主战场

默认第一主战场：

- `usd_liquidity_and_sanctions`

次级战场：

- `energy_supply_and_shipping`
- `technology_export_controls`
- `regional_conflict_escalation`

### 6.3 `game_state` 状态机

固定五档：

- `stable_competition`
- `financial_pressure`
- `commodity_weaponization`
- `bloc_fragmentation`
- `systemic_repricing`

### 6.4 升级逻辑

- `stable_competition -> financial_pressure`
  - 美元抽紧、制裁加码、银行/支付链摩擦、信用利差扩张
- `financial_pressure -> commodity_weaponization`
  - 能源/航运被显著武器化、油价跳升、 chokepoint 风险放大
- `* -> bloc_fragmentation`
  - 结算体系、供应链、金融通道出现显著阵营分裂
- `* -> systemic_repricing`
  - 多条传导链同时 active 且发生跨资产重定价

## 7. 从策略集到传导链

### 7.1 冲突轴（dominant_conflict_axes）

建议冻结：

- `usd_liquidity_pressure`
- `sanctions_financial_fragmentation`
- `energy_supply_pressure`
- `shipping_chokepoint_pressure`
- `technology_supply_chain_pressure`
- `regional_conflict_escalation`

### 7.2 主传导链

建议冻结：

- `usd_liquidity_chain`
- `financial_sanctions_chain`
- `energy_supply_chain`
- `shipping_supply_chain`
- `credit_intermediary_chain`
- `risk_off_deleveraging_chain`

### 7.3 映射原则

- `usd_liquidity_pressure`
  -> `usd_liquidity_chain`, `credit_intermediary_chain`, `risk_off_deleveraging_chain`
- `sanctions_financial_fragmentation`
  -> `financial_sanctions_chain`, `credit_intermediary_chain`
- `energy_supply_pressure`
  -> `energy_supply_chain`, `shipping_supply_chain`
- `shipping_chokepoint_pressure`
  -> `shipping_supply_chain`, `energy_supply_chain`
- `technology_supply_chain_pressure`
  -> 供应链/盈利/资本开支传导
- `regional_conflict_escalation`
  -> 可能同时激活能源、航运、制裁、风险偏好链

## 8. 从传导链到 safety margin / hard boundary

### 8.1 margins

- `liquidity_margin`
  <- `usd_liquidity_chain`, `financial_sanctions_chain`, `risk_off_deleveraging_chain`
- `credit_margin`
  <- `credit_intermediary_chain`, `usd_liquidity_chain`
- `energy_margin`
  <- `energy_supply_chain`, `shipping_supply_chain`
- `policy_margin`
  <- 政策缓冲、库存释放、央行/财政兜底、停火/降级概率

### 8.2 `system_margin_score`

不做简单平均。对：

- `dominant_chain`
- `active` 中最危险的链
- 已触发的 `hard_boundary`

提高权重，避免最危险链被平均掉。

### 8.3 `hard_boundaries`

- `canary_hard_block`
  - `systemic_repricing`
  - `regime_state = systemic_risk`
  - `credit_margin` 跌破硬阈值
  - `risk_off_deleveraging_chain = dominant`
- `new_risk_hard_block`
  - 多链 active 且至少一条 dominant
  - `system_margin_score` 跌破硬阈值
  - 银行/信用/能源/crypto 共振恶化
- `shadow_only_boundary`
  - 尚未全面 hard block，但安全边际明显压缩

## 9. 与现有 overlay 的关系

未来关系固定为：

- `event_game_state_snapshot`
- `event_transmission_chain_map`
- `event_regime_snapshot`
- `event_asset_shock_map`
- `event_safety_margin_snapshot`

共同压缩进：

- `event_live_guard_overlay`

执行层原则仍不变：

1. 解释层不能直接放行 live
2. 执行层只能读 overlay
3. source-of-truth 必须是 artifact，而非 UI/operator summary

## 10. 当前事件的默认结构判断框架

若以当前用户关注的事件作为默认模板，第一阶段结构判断框架为：

### `game_state`

- 倾向 `financial_pressure`

### `dominant_conflict_axes`

- `usd_liquidity_pressure`
- `energy_supply_pressure`
- `sanctions_financial_fragmentation`

### `dominant_chain`

- `credit_intermediary_chain`
- `usd_liquidity_chain`
- `risk_off_deleveraging_chain`

若中东能源与航运进一步升级，再提高：

- `energy_supply_chain`
- `shipping_supply_chain`

### `hard_boundaries`

第一阶段默认更接近：

- `canary_hard_block = true`
- `new_risk_hard_block = conditional`
- `shadow_only_boundary = true`

即：

- 先禁止轻率试错
- 先压风险
- 再看是否真的进入系统性危机

## 11. 下一版 implementation plan 分阶段拆法

### 阶段 A：博弈层最小骨架

- change class：`RESEARCH_ONLY`
- 目标：生出 `game_state / transmission_chain` 两个新解释层 artifacts
- 不接 overlay / dashboard / operator
- 不提前生成 `safety_margin_snapshot`

### 阶段 B：生成 `safety_margin` 并让现有 regime / shock map 吸收博弈层

- change class：`RESEARCH_ONLY`
- 目标：
  - 先生成 `event_safety_margin_snapshot`
  - 再让 `event_regime_snapshot` / `event_asset_shock_map` 读取新 artifacts

### 阶段 C：安全边际压缩进 overlay

- change class：`LIVE_GUARD_ONLY`
- 目标：让 `event_live_guard_overlay` 吸收 `system_margin_score / hard_boundaries / dominant_chain`

### 阶段 D：operator / dashboard 展示增强

- change class：`RESEARCH_ONLY`（或少量 `LIVE_GUARD_ONLY`）
- 目标：增加主战场、主传导链、安全边际的 drilldown 展示

### 推荐顺序

严格按：

- `A -> B -> C -> D`

因为最安全的顺序是：

- 先解释
- 再补 safety margin
- 再压缩
- 再执行
- 最后展示

## 12. 残余风险

- 新增博弈层会显著增加结构解释能力，但也会增加建模复杂度；
- 若直接跳到 overlay/执行层，容易破坏当前已闭合主线；
- 目前最稳妥的做法仍然是：冻结当前主线为基线，再按 `A -> B -> C -> D` 分阶段推进。
