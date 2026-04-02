# Fenlie 黑天鹅事件量化管道设计

- 日期：2026-03-26
- 模式：architecture_review
- 推荐 change class：`RESEARCH_ONLY -> LIVE_GUARD_ONLY`
- 当前文档阶段：设计 / 待实现
- 范围：围绕仍在发展的黑天鹅/危机事件，建立一条可复现、可审计、可持续迭代的公开信息事件量化管道，把事件冲击沉淀为 source-owned artifacts，并下发到 Fenlie 的 research / risk / gate / operator 层；本设计第一阶段不直接改 live executor，不要求新增交易所注册。

## 1. 背景

当前 Fenlie 已具备：

- regime / stress / risk multiplier / operator brief / review artifact 等成熟入口；
- `source-of-truth artifacts > compact handoff/context > UI/operator summaries > chat memory` 的 authority 约束；
- `run-halfhour-pulse`、idempotency、timeout<=5000ms 等 live 路径硬规则；
- 公开市场与加密资产的已有数据源、review 链与 operator 可视化面板。

用户当前提出的新需求不是单一策略修修补补，而是：

1. 把“仍在发展的黑天鹅事件”持续纳入交易系统；
2. 广泛收集公开信息并做事件量化；
3. 与历史能源危机、金融危机做结构类比；
4. 把结果落实到研究、风控、gate、operator，而不是只写一篇临时分析。

截至 2026-03-26，当前事件的初始公共信息特征更接近：

- 私募信贷 / 信用流动性压力抬升；
- 能源 / 地缘供给冲击存在放大尾部风险；
- 风险资产存在跨资产去杠杆与避险切换可能。

这类事件的核心困难不在“新闻多”，而在：

- 事件仍在发展，结论会变；
- headline 容易比价格传播更快、更噪声；
- operator 需要的是可执行的风险结构，而不是一次性的叙事；
- 如果直接把事件塞进交易信号，容易过拟合并污染 live 路径。

因此，本设计选择建立一条 **事件体制侧车（event regime sidecar）**，先做 source-owned 事件量化与风险覆盖层，再逐步接入 live guard。

## 2. 目标

建立一条最小但完整的 Fenlie 事件量化管道，用来回答：

1. 当前事件属于哪类风险冲击；
2. 当前冲击强度、传播范围、持续性和系统性程度如何；
3. 它与哪些历史危机 archetype 相似、又有哪些关键不同；
4. 哪些资产最可能在 1d / 3d / 7d 视角下出现方向性偏置与波动放大；
5. Fenlie 是否应该自动降风险、收紧 gate、冻结 canary、切 shadow / SIM_ONLY；
6. 上述结论如何以 source-owned artifact 的形式沉淀，并被 risk / gate / operator 消费。

第一阶段服务的重点资产/市场：

- Crypto：`BTC / ETH / SOL / BNB`
- 宏观代理：黄金、长债、美元、原油
- 信用/金融代理：银行股、高收益债、波动率指标
- 执行落点：优先服务 `OKX / Binance` 可交易标的，其余作为风险代理输入

## 3. 非目标

本设计第一阶段不做：

- 直接把新闻/文本输入 live executor；
- 自动恢复风险、自动放行 live、自动提高杠杆；
- 仅凭 headline 直接升级到系统性危机；
- 把事件分数直接作为开仓 alpha；
- 要求用户新增注册交易所；
- 新建一套脱离现有 Fenlie artifact / review / operator 体系的并行系统。

## 4. 方案比较

### 方案 A：事件风险覆盖层

只新增一层事件 risk sidecar，输出：

- `event_severity_score`
- `asset_shock_watchlist`
- `risk_multiplier_override`
- `live_gate_override`

**优点**

- 上线最快
- 最安全
- 最适合“自动降级、不自动放行”

**缺点**

- 更像风险护栏，不形成完整历史类比与研究框架
- 对研究和 operator 的解释力不足

### 方案 B：事件体制侧车 + 历史危机类比库（推荐）

核心结构：

1. 事件 intake
2. 事件标准化
3. 历史危机类比库
4. 传播/冲击量化
5. 风控/研究输出

**优点**

- 可解释性强
- 既能服务 research，也能下发到 live guard
- 最适合“事件仍在发展”的场景
- 后续可接入更多资产与 venue，不污染策略内核

**缺点**

- 范围比方案 A 大
- 需要更清晰的 artifact 合同与单入口 runner

### 方案 C：事件直接入策略模型

直接把事件特征并入 signal / sizing / execution 模型。

**优点**

- 理论上对 alpha 最直接

**缺点**

- 过拟合风险最高
- 历史样本不充分
- 在当前事件发展期风险过大
- 不适合作为第一阶段

### 结论

选择 **方案 B**，但分两段落地：

- 第一段：`RESEARCH_ONLY`
- 第二段：`LIVE_GUARD_ONLY`

## 5. 总体架构

事件量化管道作为一条独立 sidecar，位于：

`公开事件/市场代理输入 -> 事件标准化 -> 历史危机类比 -> 传播/冲击量化 -> 风控/研究输出`

### 5.1 输入层

第一阶段只使用公开、可复现、可时间戳化的数据：

- 新闻 / 监管 / 机构公告
- 价格与风险代理：
  - `BTC / ETH / SOL / BNB`
  - 黄金 / 长债 / 美元 / 原油
  - 银行股 / 高收益债 / 波动率
  - 信用压力代理、流动性压力代理、能源冲击代理

### 5.2 中间层

拆成 4 个 source-owned artifact：

1. `event_intake`
2. `event_regime_snapshot`
3. `event_crisis_analogy`
4. `event_asset_shock_map`

### 5.3 输出层

只输出两类 authority：

- **研究输出**
  - 态势摘要
  - 历史类比
  - 观察资产清单
- **风控输出**
  - `risk_multiplier_override`
  - `gate_tightening_state`
  - `canary_freeze_state`

### 5.4 权限边界

严格冻结为：

- **可自动降级**
- **不可自动放行**

可自动：

- 下调 `risk_multiplier`
- 收紧 `gate`
- 禁止新 `canary`
- 将策略切到 `shadow / SIM_ONLY`

不可自动：

- 恢复正常风险
- 扩大仓位
- 放行新的 live execution

### 5.5 与现有 Fenlie 的接线

第一阶段只接到：

- regime / stress / review artifact
- operator brief / dashboard summary
- risk multiplier / live guard gate

不直接接 live executor。

## 6. 事件分类与评分框架

### 6.1 事件分类

每个事件允许多标签，第一阶段固定 7 类：

1. `credit_deterioration`
2. `liquidity_redemption_stress`
3. `bank_intermediary_chain_risk`
4. `energy_supply_shock`
5. `policy_geopolitical_shock`
6. `sovereign_fx_rate_shock`
7. `crypto_market_infrastructure_shock`

对当前 2026-03 事件，默认主标签应优先落到：

- `credit_deterioration`
- `liquidity_redemption_stress`
- `bank_intermediary_chain_risk`

如能源冲击继续扩散，再叠加：

- `energy_supply_shock`
- `policy_geopolitical_shock`

### 6.2 一级风险因子

第一阶段只做 3 个一级因子：

1. `credit_liquidity_stress`
2. `energy_geopolitical_stress`
3. `cross_asset_deleveraging`

### 6.3 核心分数

每个刷新窗口产出以下 `0~1` 分数：

- `severity_score`
- `breadth_score`
- `contagion_score`
- `persistence_score`
- `policy_offset_score`
- `confidence_score`

并合成：

- `event_severity_score`
- `systemic_risk_score`

建议在合成时提高以下维度权重：

- `contagion_score`
- `breadth_score`
- `persistence_score`

避免被单条 headline 误导。

## 7. 历史危机类比库

第一阶段的 crisis archetype library 至少包含：

1. `oil_shock_1973_1979`
2. `ltcm_russia_1998`
3. `gfc_2008`
4. `eurozone_sovereign_2011`
5. `energy_credit_2014_2016`
6. `covid_liquidity_2020`
7. `europe_energy_2022`
8. `regional_banks_2023`
9. `private_credit_redemption_2026`

### 7.1 类比输出格式

`event_crisis_analogy` 不输出一句“最像 2008”，而输出：

- `top_analogues[]`
  - `archetype_id`
  - `similarity_score`
  - `match_axes[]`
  - `mismatch_axes[]`

### 7.2 设计原则

- 找“冲击结构相似”，不找“故事完全一样”
- 类比只给结构参考，不给机械预测

## 8. 当前事件的量化模板

### 8.1 因子 1：`credit_liquidity_stress`

观测指标：

- 限赎 / 融资收紧事件计数（7d / 14d / 30d）
- HY OAS / B / CCC 利差
- 资管 / 银行股相对弱势
- BDC 折价代理
- 高可信来源的同主题连续报道

输出：

- `credit_liquidity_stress_score`

### 8.2 因子 2：`energy_geopolitical_stress`

观测指标：

- WTI / Brent 1d / 3d / 7d 变动
- 天然气与能源波动
- 是否出现 IEA / 政府库存释放
- 中东、航运、出口限制 headline 密度
- 油价与长债 / 美元 / 黄金共振强度

输出：

- `energy_geopolitical_stress_score`

### 8.3 因子 3：`cross_asset_deleveraging`

观测指标：

- `BTC / ETH / SOL / BNB` 联动跌幅
- `VIX` 变化
- 美元走强程度
- 长债是否被买入
- 银行股 / HY / crypto 是否同步弱势

输出：

- `cross_asset_deleveraging_score`

## 9. 组合评分与状态机

### 9.1 建议权重

`event_severity_score`

- `credit_liquidity_stress_score`: `0.45`
- `energy_geopolitical_stress_score`: `0.25`
- `cross_asset_deleveraging_score`: `0.30`

`systemic_risk_score`

- `credit_liquidity_stress_score`: `0.35`
- `breadth_score`: `0.20`
- `contagion_score`: `0.25`
- `persistence_score`: `0.20`

### 9.2 状态机

固定 4 档：

- `watch`
- `sector_stress`
- `cross_asset_contagion`
- `systemic_risk`

### 9.3 升级原则

当以下组合持续多个刷新窗口时升级：

- `credit_liquidity_stress_score` 明显抬升；
- `cross_asset_deleveraging_score` 同步抬升；
- headline、价格、信用代理三者一致确认；
- 外溢开始影响银行 / 资管 / HY / 美元 / 加密。

### 9.4 降级原则

降级必须更慢、更保守：

- 连续多窗口缓和
- 信用利差回落
- 联动解除
- 政策缓冲得到市场确认

建议使用：

- EWMA
- rolling z-score
- change-point / threshold crossing
- hysteresis（升容易、降更难）

## 10. 资产冲击映射

### 10.1 第一阶段服务资产

- Crypto：`BTC / ETH / SOL / BNB`
- 避险：黄金、长债、美元
- 能源：原油
- 信用/金融：银行股、高收益债

### 10.2 方向性模板

#### Crypto

- `BTC`：先偏风险资产卖出，后期才可能转向替代储备叙事
- `ETH`：通常比 `BTC` 更弱
- `SOL`：高 beta，波动最敏感
- `BNB`：对交易所/平台风险情绪更敏感

#### 黄金

- 中期偏避险受益
- 短期在流动性挤兑下也可能被卖，仍定义为高波动

#### 长债

- 在信用衰退主导模板下偏多
- 在能源再通胀模板下方向可能反复

#### 原油

- 是区分“能源危机”与“信用危机”主导权的关键代理
- 若地缘供给冲击主导，偏多且高波动
- 若需求塌缩主导，后期可反转

#### 银行股 / 资管股 / 高收益债

- 对当前事件最敏感
- 应作为系统性风险外溢的核心代理资产

### 10.3 输出字段

`event_asset_shock_map` 中每个资产至少输出：

- `shock_direction_bias`
- `expected_volatility_rank`
- `contagion_sensitivity`
- `risk_1d`
- `risk_3d`
- `risk_7d`

## 11. 数据源分层

### 11.1 A 层：事件/机构信息源

优先来源：

- Reuters（存元数据/摘要，不存长文全文）
- IMF / BIS / ECB / 美联储 / IEA / EIA / SEC / ESMA
- 大型机构公开披露 / 新闻稿
- 监管公告、基金限赎/减值/融资收紧公告

用途：

- `event_intake`

### 11.2 B 层：市场风险代理

第一阶段公开代理：

- `VIXCLS`（FRED）
- 高收益债利差（FRED）
- 单 B / CCC 利差（FRED）
- 银行股 / 资管股价格代理
- 黄金 / 长债 / 美元 / 原油价格代理
- CoinGecko `BTC / ETH / SOL / BNB`

用途：

- `event_regime_snapshot`
- `event_asset_shock_map`

### 11.3 C 层：历史危机知识底座

优先来源：

- IMF GFSR
- BIS Quarterly Review
- IEA 能源危机复盘
- World Bank / OECD / ECB 公开危机复盘

用途：

- `event_crisis_analogy`

## 12. source-owned artifact 合同

### 12.1 `event_intake`

路径建议：

- `system/output/review/<stamp>_event_intake.json`
- `system/output/review/latest_event_intake.json`

最小字段：

- `events[]`
  - `event_id`
  - `event_ts_utc`
  - `source`
  - `source_type`
  - `headline`
  - `event_classes[]`
  - `regions[]`
  - `affected_assets[]`
  - `credibility_score`
  - `novelty_score`

### 12.2 `event_regime_snapshot`

路径建议：

- `system/output/review/<stamp>_event_regime_snapshot.json`
- `system/output/review/latest_event_regime_snapshot.json`

最小字段：

- `severity_score`
- `breadth_score`
- `contagion_score`
- `persistence_score`
- `policy_offset_score`
- `confidence_score`
- `event_severity_score`
- `systemic_risk_score`
- `regime_state`
- `primary_axes`
- `headline_drivers`
- `top_risk_assets`

### 12.3 `event_crisis_analogy`

路径建议：

- `system/output/review/<stamp>_event_crisis_analogy.json`
- `system/output/review/latest_event_crisis_analogy.json`

最小字段：

- `top_analogues[]`
  - `archetype_id`
  - `similarity_score`
  - `match_axes[]`
  - `mismatch_axes[]`

### 12.4 `event_asset_shock_map`

路径建议：

- `system/output/review/<stamp>_event_asset_shock_map.json`
- `system/output/review/latest_event_asset_shock_map.json`

最小字段：

- `assets[]`
  - `asset`
  - `class`
  - `shock_direction_bias`
  - `expected_volatility_rank`
  - `contagion_sensitivity`
  - `risk_1d`
  - `risk_3d`
  - `risk_7d`

### 12.5 `event_live_guard_overlay`

路径建议：

- `system/output/state/event_live_guard_overlay.json`

最小字段：

- `risk_multiplier_override`
- `gate_tightening_state`
- `canary_freeze`
- `shadow_only_assets[]`
- `review_required`
- `override_reason_codes[]`
- `valid_until_utc`

该 artifact 是事件管道唯一允许下游 live guard 消费的执行面输出。

## 13. 单入口 runner 与运行模式

### 13.1 唯一入口

建议实现：

- `system/scripts/run_event_crisis_pipeline.py`

串行产出：

1. `event_intake`
2. `event_regime_snapshot`
3. `event_crisis_analogy`
4. `event_asset_shock_map`
5. `event_live_guard_overlay`
6. `event_crisis_operator_summary`

### 13.2 运行模式

支持 3 种模式：

- `snapshot`
- `hourly`
- `eod_summary`

### 13.3 刷新节奏

按用户确认的第一阶段频率：

- 主刷新：每 1 小时
- 重大异动：手动即时重跑
- 每日收盘后：产出 crisis summary

### 13.4 工件写入顺序

固定顺序：

1. `latest_event_intake.json`
2. `latest_event_regime_snapshot.json`
3. `latest_event_crisis_analogy.json`
4. `latest_event_asset_shock_map.json`
5. `event_live_guard_overlay.json`
6. `latest_event_crisis_operator_summary.json`

## 14. 与 Fenlie 的接线顺序

### 14.1 第一步：只接 review / operator

先接到：

- operator brief
- dashboard terminal / signal-risk / stress summary
- handoff packet

这一步只显示，不自动影响 live。

### 14.2 第二步：接 risk multiplier

将：

- `event_live_guard_overlay.risk_multiplier_override`

叠加到现有风险倍率体系，但只允许向下压，不允许上提。

### 14.3 第三步：接 gate tightening

将：

- `gate_tightening_state`
- `canary_freeze`

接到 live guard / canary guard。

### 14.4 第四步：未来再考虑接研究模型

只有在 sidecar 稳定后，才考虑把事件特征接到研究 / 回测层。

## 15. 风控动作映射

### `watch`

- 只发 operator 警报
- 不动 live

### `sector_stress`

- 下调风险倍率
- 收紧高 beta 标的
- 增加人工 review 权重

### `cross_asset_contagion`

- 冻结新 canary
- 将部分策略切 `shadow / SIM_ONLY`
- 只保留低风险 / 高流动性策略

### `systemic_risk`

- 大幅收紧 `risk_multiplier`
- 禁止事件敏感策略新开仓
- 只保留观察、对冲、清理路径

## 16. 当前事件的第一阶段初始默认判断

截至 2026-03-26，默认初始状态不应直接设为 `systemic_risk`，而应设为：

- `sector_stress`

若后续银行、HY、美元、VIX、crypto 出现持续联动，再升级到：

- `cross_asset_contagion`

这是更保守、可审计、且与当前公开证据更匹配的初始判断。

## 17. 实施边界

### 第一阶段必须做

- 公开事件 intake
- 三因子评分
- 历史危机类比
- 资产冲击地图
- 风险 overlay
- operator summary
- 最小 dashboard / terminal 接线

### 第一阶段明确不做

- 自动恢复风险
- 自动放行 live
- 直接入策略信号
- LLM 自由发挥式结论
- 未带时间戳约束的实时推理缓存

## 18. 推荐实施顺序

### 阶段 1：`RESEARCH_ONLY`

- 产出全部事件 source artifacts + summary
- 不接 live gate

### 阶段 2：`LIVE_GUARD_ONLY`

- 只把 overlay 接到 risk / gate
- 不碰 executor

## 19. 验收标准

以下全部满足，才算第一阶段完成：

1. 存在唯一 runner：`run_event_crisis_pipeline.py`
2. 五类 source-owned artifacts 可稳定生成
3. 每小时模式与手动 snapshot 模式均可运行
4. `event_live_guard_overlay` 只能自动降级，不能自动放行
5. operator / dashboard 只消费 artifact，不自重算 authority
6. 可对当前 2026-03 事件产出：
   - `sector_stress` 初始判断
   - 历史类比 top list
   - `BTC / ETH / SOL / BNB / 黄金 / 长债 / 原油 / 银行股 / 高收益债` 冲击映射

## 20. 公开参考资料

为保证公开、可复现与后续扩展，第一阶段优先依赖以下公共来源：

- [IMF GFSR 2025-04](https://www.imf.org/en/Publications/GFSR/Issues/2025/04/22/global-financial-stability-report-april-2025)
- [IMF：Private Credit Warrants Closer Watch](https://www.imf.org/en/Blogs/Articles/2024/04/08/fast-growing-USD2-trillion-private-credit-market-warrants-closer-watch)
- [BIS Quarterly Review 2025-03](https://www.bis.org/publ/qtrpdf/r_qt2503.htm)
- [IEA：2022-2023 能源危机经验](https://www.iea.org/reports/gas-market-lessons-from-the-2022-2023-energy-crisis)
- [FRED: VIXCLS](https://fred.stlouisfed.org/series/VIXCLS)
- [FRED: ICE BofA US High Yield OAS](https://fred.stlouisfed.org/series/BAMLH0A0HYM2)
- [EIA Open Data](https://www.eia.gov/opendata/v1/)
- [CoinGecko API](https://www.coingecko.com/en/api)
- [Axios 2026-03-25 私募信贷板块综述](https://www.axios.com/2026/03/25/private-credit-apollo-ares)

## 21. 残余风险

- 公开新闻与官方材料的发布时间存在延迟；
- 同一事件在初期可能出现高噪声与来源冲突；
- 历史类比只能提供结构参考，不能机械外推收益；
- 若后续需要把事件特征深入接入 alpha，必须单独做新的设计与回测验证；
- 若未来 venue 侧执行从 Binance/Bybit 转向 OKX，这条事件管道仍应保持 venue-agnostic，只在 `event_live_guard_overlay` 层输出统一风险覆盖，而不绑死某家交易所。
