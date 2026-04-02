# Fenlie 商业计划书（路演版）

> 版本：2026-03-24  
> 面向对象：量化/金融机构投资人  
> 当前定位：**Crypto-first 的机构级反脆弱量化研究、风控编排与执行控制平台**  
> 重要说明：本文严格区分 **已验证 / 验证中 / 规划中**。本文不构成任何收益承诺、代客理财承诺或未审计商业数据披露。

---

## 1. 一句话定位

Fenlie 是一套面向量化团队的 **反脆弱交易操作系统**：它把原本分散的研究、回测、风控、操作监控、执行接管与审计证据，收敛到一条可验证、可追溯、可部署的控制链中。

对投资人而言，Fenlie 不是单一策略，不是普通行情面板，也不是只会下单的执行机器人；它要解决的是 **量化团队从“研究想法”到“安全运行”之间的系统断裂**。

---

## 2. 核心判断

### 2.1 市场现实

今天大量量化团队的能力被拆散在几个割裂层里：

1. **研究层**  
   策略想法、回测脚本、外部数据、参数扫描常常分散在 notebook、临时脚本、聊天记录和人工经验里。

2. **风控层**  
   很多团队把风控理解为“加止损”或“限制仓位”，而不是一套贯穿研究、上线、异常回退、审计的体系。

3. **执行层**  
   下单与接管路径经常是另一个系统，研究证据并不会自然传递到执行前决策与运行监控。

4. **运维与审计层**  
   当策略失效、数据漂移、界面异常、门禁退化、执行路径变化时，很少有团队能说清：  
   **到底哪里变了、谁拥有状态、为什么现在还能继续跑。**

Fenlie 的机会不在于“再发明一个 alpha 因子”，而在于：  
**先把研究—风控—执行—可观测性这条链做成真正可运营的系统。**

### 2.2 我们的判断

未来高质量量化团队的分水岭，不只是“谁找到更多策略”，而是谁能建立：

- 更快的研究到验证闭环；
- 更强的 source-of-truth 纪律；
- 更稳的门禁、回退与执行接管能力；
- 更低的人为断层与组织漂移成本。

Fenlie 要服务的正是这一类团队。

### 2.3 外部市场信号（截至 2026-03-24 的公开来源）

以下不是 Fenlie 自身经营数据，而是外部市场背景，用于解释“为什么现在值得做量化控制面基础设施”。

1. **算法交易软件与平台市场仍在增长**  
   Grand View Research 在其 2025 页面中写到：全球算法交易市场 **2024 年约为 210.6 亿美元**，并预计到 **2030 年达到 429.9 亿美元**，2025–2030 年复合增速约 **12.9%**。  
   这说明，量化交易相关的软件、平台、执行与基础设施预算并不是缩减赛道，而是持续增长赛道。  
   来源：<https://www.grandviewresearch.com/industry-analysis/algorithmic-trading-market-report>

2. **数字资产已进入机构配置讨论，而不再只是边缘实验**  
   PwC 与 AIMA 的 2024 全球加密对冲基金报告显示：其调查样本中，**47% 的传统对冲基金** 已有数字资产敞口；同时，传统对冲基金对数字资产 **衍生品交易的使用率升至 58%**。  
   这意味着机构需求正在从“是否接触”向“如何更专业地接触、交易与管理”迁移。  
   来源：<https://www.pwc.com/gx/en/industries/financial-services/crypto-services/sixth-annual-global-crypto-hedge-fund-report.html>

3. **机构不只想持有 crypto，也在扩展到 stablecoin、tokenization 与更深的工作流**  
   Coinbase 与 EY-Parthenon 于 2025 年发布的机构数字资产调查提到：  
   - 超过 **四分之三** 的受访机构投资者预计会在 2025 年增加数字资产配置；  
   - **59%** 计划将超过 **5% AUM** 配置到数字资产或相关产品；  
   - **84%** 的机构已在使用或有兴趣使用 stablecoin；  
   - **76%** 的机构计划在 2026 年前投资某种形式的 tokenized assets。  
   这说明需求不再局限于“买币”，而正在外溢到更复杂的研究、交易、清算与控制流程。  
   来源：<https://www.coinbase.com/en-br/institutional/research-insights/research/market-intelligence/2025-institutional-investor-survey>

基于上述公开信息，我们的判断是：  
**真正有机会的，不只是单一策略产品，而是能把研究、风控、运营控制和执行接管装入一条可信链的量化基础设施。**

---

## 3. 问题定义：Fenlie 解决什么

Fenlie 主要解决四个系统级问题。

### 3.1 研究链断裂

很多策略研究在“idea -> backtest -> 结论”之间缺乏统一工件链，导致：

- 结果复现困难；
- 参数选择难以审计；
- 新会话/新成员接手成本极高；
- UI 展示内容与真实研究状态脱节。

### 3.2 风控不是控制面

大量系统只在执行时刻做风控，而不是把风控前移为：

- 数据质量门禁；
- source ownership 审查；
- 发布/回退规则；
- 研究与执行之间的晋升标准。

### 3.3 执行接管风险高

研究系统与执行系统分离时，最常见问题不是“没有单子”，而是：

- 上线前没有统一 readiness 证据；
- 执行路径与研究结论没有可追踪映射；
- 异常时不能快速判断是否应降级、阻断还是回退。

### 3.4 团队协同不可审计

当项目持续数周或数月迭代后，团队常常会失去共同参照系：

- 当前主线到底是什么；
- 哪些结论已证伪；
- 哪些模块只是试验性 sidecar；
- 哪些页面展示的是 public，哪些是 internal。

Fenlie 的一个重要价值，是把这些“本该存在但常常缺席”的管理与工程纪律做成系统本身。

---

## 4. 解决方案：Fenlie 平台结构

Fenlie 当前的产品形态可拆成四层。

### 4.1 Research OS（研究操作系统）

作用：

- 管理研究主线；
- 连接数据、回测、参数与 review 工件；
- 强制使用 source-owned artifact 作为研究状态锚点；
- 为后续 dashboard、contracts、handoff、验收链提供唯一事实层。

当前典型主线：

- 单币种：`ETHUSDT`
- 周期：`15m`
- 结构：`breakout-pullback`
- 当前重点：`exit/risk`

当前研究纪律明确区分：

- 已证伪/暂无增益方向；
- watch-only sidecar；
- 当前 canonical anchor；
- 下一步研究优先级。

**状态：已验证**

### 4.2 Operator Console（运营控制台）

作用：

- 统一展示研究、contracts、raw、overview、terminal 等页面；
- 暴露异常、门禁、source head、工件摘要、对齐投射；
- 给操作员提供可穿透、可回溯、可区分 public/internal 的操作界面。

Fenlie 控制台的设计目标不是“漂亮”，而是：

- 高密度；
- 可审计；
- 对异常敏感；
- 能快速定位 source artifact 与当前焦点。

**状态：已验证**

### 4.3 Guarded Execution Bridge（受控执行桥）

作用：

- 连接研究/控制面与云端执行接管链；
- 在不破坏门禁规则的前提下，支持 probe、canary、ready-check、takeover 路线；
- 通过幂等、白名单、桥接脚本、source-owned review 工件降低执行接管风险。

该层当前更准确的说法是：

- **已完成关键运行路径与桥接治理设计**
- **具备验证型接管与审查能力**
- **仍处于从“工程可行”走向“产品化可规模部署”的阶段**

**状态：验证中**

### 4.4 Private Deployment & Enablement（私有化部署与研究赋能服务）

作用：

- 为小型量化团队、crypto-native desk、研究型资管技术组提供：
  - 私有部署；
  - 研究链搭建；
  - dashboard/operator surface 定制；
  - 研究到控制面迁移；
  - 执行桥 pilot 与风控流程梳理。

这部分是 Fenlie 最自然的商业化起点，因为它匹配当前最成熟的能力：  
**研究、控制面、证据链、工程化交付。**

**状态：规划中 / 可 pilot 化**

---

## 5. 典型场景解决方案

本项目不是抽象平台叙事，而是针对明确使用场景。

### 场景 A：Crypto 量化团队的研究链标准化

#### 客户问题

- 研究依赖脚本和人工记忆；
- 参数和回测窗口无法追溯；
- 多轮讨论后不知道当前主线和已证伪项；
- 新成员接手需要重新阅读大量上下文。

#### Fenlie 方案

- 用 source-owned review artifact 固定研究主线；
- 用 SIM_ONLY / RESEARCH_ONLY 纪律隔离研究与 live path；
- 将主线、watch、blocked、canonical anchor、next priority 写入工件链；
- 通过 dashboard 工作区直接暴露研究地图、状态雷达、焦点与 source-head。

#### 直接价值

- 降低研究漂移；
- 降低重复实验；
- 降低上下文切换成本；
- 提高研究结论的组织可继承性。

#### 当前状态

**已验证**。当前 repo 内已有 ETH 15m breakout-pullback 主线、exit/risk canonical handoff 与多轮 review 工件。

---

### 场景 B：交易运营控制台与异常暴露

#### 客户问题

- 量化团队常常没有真正可用的运营控制台；
- 运行界面只能看到结果，不能看到来源与门禁；
- public/internal 信息混用，导致协作时边界模糊；
- UI 与 source-of-truth 漂移后，团队无法快速发现。

#### Fenlie 方案

- 将 overview / workspace / terminal 做成分域控制台；
- 严格区分 public/internal snapshot；
- 用 release-checklist、workspace routes smoke、public acceptance 做发布前硬门槛；
- 让研究工件、contracts、alignment、raw 层形成可穿透结构。

#### 直接价值

- 让控制台从“展示层”升级为“运营与审计层”；
- 降低错误认知引发的错误决策；
- 提高上架、回归与远端拓扑核验效率。

#### 当前状态

**已验证**。当前已完成多轮 Cloudflare Pages 部署、公开拓扑 smoke、workspace routes smoke、internal alignment smoke 与 public acceptance。

---

### 场景 C：研究到执行接管的受控晋升

#### 客户问题

- 很多团队的执行接管是隐性流程；
- 当从研究进入实盘或准实盘时，缺乏统一 readiness 与回退规则；
- 多个系统之间的桥接状态难以审计。

#### Fenlie 方案

- 通过 OpenClaw cloud bridge 统一本地控制面与云端执行面；
- 用 cut-local / probe-cloud / compare / sync / ready-check / canary / autopilot 等脚本化链路固化流程；
- 使用白名单采样、幂等、限流、timeout、takeover readiness 产物管理执行风险。

#### 直接价值

- 把“执行接管”从经验动作变成受控链路；
- 为未来机构 pilot 的执行模块打基础；
- 为策略、运营与执行之间建立统一交付接口。

#### 当前状态

**验证中**。现阶段应被描述为“工程可行性与控制面治理已建立”，而不是“已全面商业化执行平台”。

---

## 6. 产品与服务说明

Fenlie 的商业化不应从“卖策略收益”开始，而应从“卖系统能力”开始。

### 6.1 产品形态

#### 产品 A：Fenlie Research OS

适合对象：

- crypto quant team
- 小型量化研究团队
- 研究驱动型交易团队

交付内容：

- 研究工件链；
- 回测/review 管线；
- source-owned 研究主线管理；
- 研究工作区与 source-head drilldown。

#### 产品 B：Fenlie Operator Console

适合对象：

- 交易运营团队；
- 技术负责人；
- 需要统一 public/internal 控制台的量化组织。

交付内容：

- overview / workspace / terminal；
- alignment internal 层；
- contracts / raw / artifacts 页面；
- 部署与验收 runbook。

#### 产品 C：Fenlie Guarded Execution Bridge

适合对象：

- 有明确执行接管需求的 crypto-native desk；
- 需要云端桥接、ready-check、canary 的团队。

交付内容：

- bridge runbook；
- probe / canary / ready-check / takeover 脚本；
- 白名单与桥接守护；
- 审计与回退路径。

### 6.2 服务形态

Fenlie 的早期服务形态建议包含三种：

1. **私有化部署与环境落地**
2. **研究链/控制台定制与导入**
3. **执行桥 pilot 与风控流程梳理**

这使项目能够在不依赖大规模标准化销售的情况下，先获得：

- 高价值客户反馈；
- 真实场景数据；
- 更强的产品路线判断。

---

## 7. 技术架构与护城河

Fenlie 的护城河不在某个单点功能，而在几个组合式壁垒。

### 7.1 八层架构与边界清晰

当前架构边界已明确覆盖：

- `data`
- `regime`
- `signal`
- `risk`
- `backtest`
- `review`
- `orchestration`
- `engine`

其价值在于：

- 避免 engine 继续吞掉所有职责；
- 让门禁、调度、发布、测试、审计成为可独立演进模块；
- 更适合机构环境下的长期维护与审计。

### 7.2 Source-owned artifact discipline

这是 Fenlie 最关键的系统哲学之一。

核心原则：

- lane / queue / gate / blocker / research state 以 source artifact 为准；
- dashboard 和 top brief 不得随意重导事实；
- 研究、contracts、UI 都从同一工件链读取状态。

这意味着：

- UI 更可信；
- 协作更稳定；
- 审计更直接；
- 新会话/新成员接手成本更低。

### 7.3 发布与验收链

Fenlie 已不是“写完页面就算上线”，而是形成了明确的发布前验证链：

- build
- test
- type-check
- dashboard Python contracts
- workspace routes smoke
- alignment internal smoke
- public acceptance

这类能力对机构客户十分关键，因为它把“产品稳定性”变成了可重复、可检查的流程。

### 7.4 研究与执行边界管理

Fenlie 当前已明确建立：

- `DOC_ONLY`
- `RESEARCH_ONLY`
- `SIM_ONLY`
- `LIVE_GUARD_ONLY`
- `LIVE_EXECUTION_PATH`

不同变更等级的边界纪律，为后续执行桥产品化提供了基础。

### 7.5 可迭代的 sidecar 研究体系

Fenlie 当前已开始把订单流类研究做成 sidecar 路径，例如：

- OI sidecar
- CVD / OI / 成交量三位一体研究方向
- context-only triad posture

这意味着平台能够吸收外部研究增强，但不会让 sidecar 未经验证就污染主执行路径。

---

## 8. 当前进展与已验证事实

本节只写当前项目内可追溯的事实。

### 8.1 控制台与部署链

截至 **2026-03-22**，Fenlie dashboard 已形成：

- Cloudflare Pages 部署能力；
- public topology smoke；
- workspace routes smoke；
- internal alignment smoke；
- public acceptance。

在本地 release-checklist 中，已出现明确结果：

- `24 files / 125 tests passed`
- `workspace-routes => ok = true`
- `alignment-internal => ok = true`
- `verify:public-surface => ok = true`

### 8.2 研究主线

当前 durable research mainline 明确为：

- `ETHUSDT`
- `15m`
- `breakout-pullback`
- 当前重点：`exit/risk`

同时已明确：

- `VPVR proxy` 无增益；
- `reclaim structure filter` 无增益；
- `daily_stop_r` 无增益；
- 当前不优先继续扩 entry filter。

### 8.3 Orderflow research-only 链

截至 **2026-03-23**，Fenlie 已把订单流研究增强继续往前推进，但保持边界清晰：

- OI sidecar 结论：`use_oi_sidecar_as_triad_input`
- triad sidecar 结论：`keep_triad_context_only_until_oi_integration_and_trust_upgrade`

这说明：

- 平台具备吸纳订单流增强证据的能力；
- 但不会把尚未升级信任的 sidecar 强行晋升为主信号。

### 8.4 测试与工程编排

Fenlie 已把 test-all 编排升级为：

- full mode 优先 `pytest`
- 可提取失败样例
- 能纳入更统一的工程验证链

此前项目级回归已记录：

- `tests_selected=447`
- `tests_ran=447`
- `failed=0`

对于机构级产品叙事，这一点很关键：  
Fenlie 不是“概念验证型脚本集合”，而是在往 **可部署、可验证的系统工程** 前进。

---

## 9. 商业模式

Fenlie 早期最合理的商业模式，不应依赖“分成收益”或“销售单一策略信号”，而应依赖平台与服务能力。

### 9.1 第一阶段：高价值交付型收入

主要包括：

1. **私有化部署费**
2. **控制台/研究链落地费**
3. **流程梳理与研究赋能费**
4. **执行桥 pilot 集成费**

适合对象：

- 5–30 人规模的 crypto 量化团队；
- 有交易、有研究、有技术负责人，但缺统一控制面的团队。

### 9.2 第二阶段：平台订阅收入

当通用化程度提高后，可演进为：

- 年度授权；
- 按团队席位/环境授权；
- operator console / execution bridge 模块化收费。

### 9.3 第三阶段：执行与风控模块增购

在执行桥成熟后，可进一步提供：

- readiness / canary / takeover 模块；
- 受控执行接入；
- 风控与审计增强模块。

当前应谨慎表达为：

**中长期收入扩展方向，而非已完全成熟的现时收入主项。**

---

## 10. Go-To-Market：为什么先从 Crypto 量化团队切入

Fenlie 当前最合适的首发楔子不是泛金融大客户，而是 **Crypto-first 的量化团队**。

### 10.1 原因一：当前证据最匹配

当前公开研究主线、订单流 sidecar、执行桥 runbook、operator console 验收链，最直接映射的是真实 crypto 场景。

### 10.2 原因二：痛点更集中

Crypto 量化团队更常见以下问题：

- 研究迭代频繁；
- 执行与风控变化快；
- 团队规模小但技术复杂度高；
- 需要更快部署、更快试错、更强可观测性。

Fenlie 正好匹配这类需求。

### 10.3 原因三：产品学习速度更快

相比一开始就进入传统大机构，crypto 团队更有机会：

- 接受 pilot 交付；
- 快速反馈控制台和研究链问题；
- 帮助我们打磨 execution bridge 的产品边界。

### 10.4 扩张路径

建议路径：

1. **Crypto quant team**
2. **研究型资管技术组**
3. **多资产/跨市场机构控制面**

这样可以保证：

- 证据链与产品成熟度匹配；
- 商业叙事不夸张；
- 平台扩张路径更稳。

### 10.5 可比平台与市场空白

Fenlie 不应被简单归类为某一类现有产品的复制品。

#### 现有常见平台大致分三类

1. **研究/回测引擎型**  
   如 QuantConnect LEAN。其官方文档强调其是 **开源、机构级、多资产** 的算法交易引擎，可用于 research、backtesting 与 live。  
   这类平台擅长“算法引擎”，但不天然等于“组织控制面”。  
   来源：<https://www.quantconnect.com/docs/v2/lean-engine>

2. **策略市场/机器人平台型**  
   如 FMZ。其官方站点强调策略开发、量化学习资源、策略出租/出售、社区，以及 trading terminal / docker / 实盘运行能力。  
   这类平台擅长“策略供给与运行入口”，但对 source-owned research governance 的强调相对较弱。  
   来源：<https://www.fmz.com/lang/en/>

3. **社区脚本与图表生态型**  
   如 TradingView Community Scripts。其官方帮助页明确指出社区脚本包含 indicators、strategies、libraries，其中 strategies 支持回测、commission、margin、trailing stop 等功能。  
   这类平台擅长 idea discovery 和脚本扩散，但不等于机构级研究—控制—执行一体化系统。  
   来源：<https://www.tradingview.com/support/solutions/43000558522-what-are-community-scripts/>

#### Fenlie 的空白带

Fenlie 想占据的不是“又一个脚本库”或“又一个回测引擎”，而是三类产品之间的空白层：

- 把研究工件链做成 source-of-truth；
- 把 dashboard 做成 operator control plane；
- 把 execution bridge 做成带门禁、回退、验收的受控接管路径。

这也是我们选择从 Crypto 量化团队切入、再向更机构化团队扩张的核心逻辑。

---

## 11. 12–18 个月路线图

### 阶段 1：产品收口（0–6 个月）

目标：

- 巩固 Research OS + Operator Console；
- 完成 investor-grade / client-grade 演示面；
- 把研究、contracts、alignment、deployment、runbook 打磨成标准交付。

交付重点：

- 更稳定的研究工件与主线管理；
- 更清晰的 public/internal 控制台；
- 更完整的文档、演示、验收与交付模板。

### 阶段 2：执行桥 pilot（6–12 个月）

目标：

- 把 execution bridge 从工程 runbook 升级为 pilot 产品能力；
- 强化 readiness、canary、回退、审计。

交付重点：

- 客户级部署模板；
- 桥接治理；
- 风险与合规边界增强；
- 试点客户反馈闭环。

### 阶段 3：模块化商业化（12–18 个月）

目标：

- 推进 operator console、research OS、execution bridge 的模块化包装；
- 形成更稳定的授权与服务边界。

交付重点：

- 模块化 SKU；
- 多环境部署与权限边界；
- 更标准化的集成与运维手册。

---

## 12. 融资用途

本版不强行写具体融资金额，而采用“用途 + 里程碑”表达。

### 12.1 产品化与控制台完善

用途：

- 控制台信息架构继续产品化；
- public/internal 视图边界强化；
- 更强的 operator workflow 与演示能力。

### 12.2 研究链与数据增强

用途：

- 继续推进多策略、多数据 sidecar；
- 完善研究工件链、review 编排与 source-owned 规则；
- 提升策略验证与研究迁移效率。

### 12.3 执行桥与风控治理

用途：

- 强化 execution bridge 的产品化抽象；
- 加强 readiness、canary、回退与白名单治理；
- 为 pilot 客户准备更完整的安全与审计路径。

### 12.4 商业化试点

用途：

- 首批 pilot 交付；
- 行业访谈与客户定制沉淀；
- 形成可复用的交付模板与报价结构。

### 12.5 当前推荐融资口径

基于当前内部三年模型与 18 个月 runway 粗算，Fenlie 当前更适合的融资讨论区间为：

- **基准推荐区间：\$2.2M–\$2.8M**

该区间目标不是“高速扩张”，而是完成以下三件事：

1. 把控制台与研究链从强工程系统收口为更稳定的产品形态；
2. 拿下首批可复用的 pilot 客户与交付模板；
3. 推进 execution bridge 的产品化治理、安全边界与试点验证。

同时，我们保留两档辅助口径：

- **Lean：\$1.8M–\$2.0M**，更适合极度克制扩张
- **Accelerate：\$3.5M–\$4.5M**，适合更激进推进团队和 pilot 布局

当前对外建议默认使用 **Base 区间**，不在正式材料中写死估值与条款。

---

## 13. 风险与合规

### 13.1 不把研究结果包装成收益承诺

Fenlie 当前核心卖点不是“稳定赚钱的黑盒策略”，而是：

- 研究链；
- 控制面；
- 审计与执行治理。

### 13.2 不夸大 execution 成熟度

OpenClaw cloud bridge 与 take-over 路线应被准确描述为：

- **验证中的执行接管体系**
- **正在走向产品化的桥接与门禁能力**

而不是“已规模化商用执行网络”。

### 13.3 严格区分 public / internal / live

Fenlie 已形成较清晰的边界纪律：

- public surface
- internal alignment
- research-only
- live-sensitive path

后续商业化必须继续维持这类边界，而不是为演示效果牺牲控制纪律。

---

## 14. 为什么 Fenlie 值得投资

Fenlie 的价值不在于单点“聪明”，而在于三个更难复制的组合：

1. **研究、控制面、执行接管不是分离卖点，而是一条统一链**
2. **平台强调 source-owned、门禁、验收、回退，而不是只强调功能数量**
3. **当前已从个人系统思维演进到可产品化、可部署、可审计的工程结构**

如果说多数量化系统关注的是“如何多找一个信号”，Fenlie 关注的是：  
**如何把信号、风控、执行、运营和组织协作装进同一台可运营的机器里。**

这也是它最具机构价值的地方。

---

## 15. 投资合作预期

本轮更适合寻找认同以下观点的投资人：

- 量化基础设施比单一策略更有长期价值；
- 机构级控制面与执行治理将成为重要市场机会；
- crypto-first 是合理切入口，但终局不止于 crypto；
- 工程纪律与系统可审计性，本身就是产品壁垒。

Fenlie 适合的资金，不应只是买流量或买故事，而应帮助它完成：

- 产品化收口；
- pilot 客户落地；
- 执行桥合规化与模块化；
- 从“强系统”走向“强产品”。

---

## 16. 附录：当前关键证据索引

以下路径用于投资材料内部审计，不建议直接放到对外简版 BP 中。

### 架构与进度

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/PROGRESS.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/ARCHITECTURE_REVIEW.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/THEORY_PROCESS.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_SKILL_MCP_ARCHITECTURE.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md`

### 研究与订单流增强

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_oi_sidecar.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_intent_triad_sidecar.json`

### 控制台与公开验收

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260322T140455Z_dashboard_workspace_routes_browser_smoke.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260322T140326Z_dashboard_internal_alignment_browser_smoke.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260322T140330Z_dashboard_public_topology_smoke.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260322T140501Z_dashboard_public_acceptance.json`

### 部署参考

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_ENGINEERING_RUNBOOK.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md`
