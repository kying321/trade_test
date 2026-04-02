# Fenlie 市场与竞品格局备忘（带来源）

> 更新时间：2026-03-24  
> 用途：融资问答、BP 附录、投资人会前预读  
> 说明：本文只使用公开来源做市场背景与竞品框架，不代表 Fenlie 的经营数据。

---

## 1. 为什么现在做 Fenlie

### 1.1 算法交易基础设施仍是增长赛道

Grand View Research 在其 2025 页面中披露：

- 全球算法交易市场 **2024 年约 210.6 亿美元**
- 预计到 **2030 年约 429.9 亿美元**
- 2025–2030 年 CAGR 约 **12.9%**

来源：<https://www.grandviewresearch.com/industry-analysis/algorithmic-trading-market-report>

### 对 Fenlie 的启发

这说明量化交易相关的软件、平台、执行与基础设施预算并未退潮。  
Fenlie 不一定需要先成为最大的平台，但它所处的“控制面与研究/执行基础设施”赛道本身具备成长空间。

---

## 2. 为什么先从 Crypto-first 切入

### 2.1 数字资产已经进入机构工作流

PwC 与 AIMA 的 2024 Global Crypto Hedge Fund Report 里提到：

- 调查样本中，**47% 的传统对冲基金** 已有数字资产敞口
- 传统对冲基金中，数字资产 **衍生品使用率升至 58%**

来源：<https://www.pwc.com/gx/en/industries/financial-services/crypto-services/sixth-annual-global-crypto-hedge-fund-report.html>

### 2.2 机构正在扩展配置与工作流复杂度

Coinbase 与 EY-Parthenon 2025 Institutional Investor Survey 提到：

- 超过 **四分之三** 的受访机构预计会在 2025 年增加数字资产配置
- **59%** 预计将超过 **5% AUM** 配到数字资产或相关产品
- **84%** 已使用或有兴趣使用 stablecoin
- **76%** 计划在 2026 年前投资某种形式的 tokenized assets

来源：<https://www.coinbase.com/en-br/institutional/research-insights/research/market-intelligence/2025-institutional-investor-survey>

### 对 Fenlie 的启发

这意味着机构需求正从“是否接触 crypto”升级为：

- 如何研究；
- 如何控制风险；
- 如何把交易、风控与运营链路做得更专业；
- 如何建立更可信的执行接管流程。

Fenlie 以 crypto-first 切入，并不是因为它只想做 crypto，而是因为这个场景最能快速暴露问题，也最需要控制面产品。

---

## 3. 竞品格局：现有平台能做什么

Fenlie 的市场定位不应简单理解为“去跟某一个平台正面对打”。  
更准确的说法是：**Fenlie 处在几类现有产品的交界空白地带。**

### 3.1 研究/回测引擎型：QuantConnect LEAN

QuantConnect 官方文档对 LEAN 的描述要点：

- 开源算法交易引擎
- 机构级
- 可用于 research、backtesting、optimization、live trading
- 覆盖多资产类别

来源：<https://www.quantconnect.com/docs/v2/lean-engine>

#### 这类产品的强项

- 算法引擎成熟
- 回测/实盘一体化路径清晰
- 适合作为研究与策略运行底座

#### 这类产品的空白

- 不一定天然解决组织级 source-owned artifact governance
- 不一定天然提供围绕 research handoff / operator alignment / public-internal surface 的控制面

---

### 3.2 策略市场/机器人平台型：FMZ

FMZ 官方站点长期强调：

- 量化交易平台
- 策略开发与社区
- 策略出租/购买
- trading terminal、docker、实盘与多交易所支持

来源：<https://www.fmz.com/lang/en/>

#### 这类产品的强项

- 上手快
- 社区活跃
- 适合策略原型、机器人部署和运行入口

#### 这类产品的空白

- 更偏策略市场与运行平台
- 对“研究工件链、控制面审计、source ownership 纪律”的产品表达相对较弱

---

### 3.3 社区脚本生态型：TradingView Community Scripts

TradingView 官方帮助中心明确：

- 社区脚本分为 indicators、strategies、libraries
- strategies 支持 backtesting、commission、margin、trailing stop 等

来源：<https://www.tradingview.com/support/solutions/43000558522-what-are-community-scripts/>

#### 这类产品的强项

- idea discovery 极强
- 社区反馈快
- 对结构、形态、短线脚本的传播效率高

#### 这类产品的空白

- 更适合发现想法，不等于形成团队级研究资产
- 不提供机构级 operator control plane
- 不提供受控 execution bridge 与验收链

---

## 4. Fenlie 的独特定位

Fenlie 的差异化不在“比 QuantConnect 更像引擎”或“比 FMZ 更多策略”，而在于：

### 4.1 它把研究状态做成 source-of-truth

研究主线、blocked、watch、canonical anchor、next priority 等信息不只留在会议或聊天里，而是尽量沉淀到 artifact 和 review 链。

### 4.2 它把控制台做成运营面

Fenlie 的 dashboard 不是简单展示结果，而是：

- 暴露 source-head；
- 暴露 public / internal 差异；
- 暴露 alignment、contracts、artifacts、raw 等页面层次；
- 绑定 smoke、acceptance、release-checklist。

### 4.3 它把 execution 当作受控晋升，不是直接黑盒自动化

Fenlie 的 execution bridge 不是“自动下单炫技”，而是：

- probe
- ready-check
- canary
- takeover
- rollback
- whitelist

这种路径更适合机构环境的产品化。

---

## 5. 对投资人的正确结论

### 不该怎么理解 Fenlie

- 不是单一高收益策略项目
- 不是只做前端控制台
- 不是立即大规模商用 execution SaaS

### 应该怎么理解 Fenlie

Fenlie 是一个正在形成中的 **量化研究—控制—执行基础设施平台**，其最强价值在于：

1. 把团队研究资产系统化；
2. 把运营控制面工程化；
3. 把执行接管做成带门禁与审计的产品路径。

---

## 6. 融资问答时的推荐表述

### 如果投资人问：你们和 QuantConnect 有什么不同？

推荐答法：

> QuantConnect 更像成熟的算法研究/运行引擎；Fenlie 当前更强调 source-owned research governance、operator control plane 和受控 execution bridge。我们并不把自己定位成单一回测引擎的替代品。

### 如果投资人问：你们和 FMZ/TradingView 有什么不同？

推荐答法：

> FMZ 和 TradingView 更擅长策略原型、社区与脚本扩散；Fenlie 更关注如何把这些研究信号沉淀为可追溯、可审计、可接管的组织系统。

### 如果投资人问：你们为什么先做 crypto？

推荐答法：

> 因为当前最需要这套研究—控制—执行控制面的团队，正集中在 crypto-native 量化场景；产品学习速度更快，也更容易快速验证控制面价值。

---

## 7. 一句话总结

Fenlie 最值得投的，不是它“可能跑出一个策略”，而是它正在把量化团队最容易失控的那条链——  
**研究、风控、控制台、执行接管** ——做成一套真正可运营的系统。

