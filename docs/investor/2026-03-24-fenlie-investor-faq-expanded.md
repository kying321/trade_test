# Fenlie Investor FAQ Expanded

> 版本：2026-03-24  
> 用途：作为 `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-investor-qa-battlecard.md` 的扩展版，用于更长会谈、会后追问、数据室 FAQ 区或创始人自用问答稿。  
> 原则：**回答要稳、边界要清楚、避免收益承诺与成熟度夸大。**

---

## 1. 定位与产品

### Q1：Fenlie 到底是什么？

Fenlie 是一套面向量化团队的控制面基础设施，核心是把研究、风控、运营控制与 execution 接管收敛到同一条可验证、可审计的系统链中。

### Q2：为什么你们不是一个策略项目？

因为策略只是 Fenlie 的一个输入层。Fenlie 更大的价值在于把研究治理、控制面、release-checklist、handoff、execution boundary 做成系统能力。

### Q3：为什么你们也不是普通 dashboard？

因为 dashboard 在 Fenlie 里不是展示壳，而是 operator control plane。它连接 source-owned artifacts、contracts、alignment、验收链和 public/internal 边界。

### Q4：你们和回测平台最大的区别是什么？

回测平台通常擅长策略运行或回测本身；Fenlie 更强调研究治理、控制面与受控 execution bridge 的系统一体化。

### Q5：你们为什么一直强调 source-of-truth？

因为量化团队一旦失去 source-of-truth，就很容易出现“策略在跑，但没人说得清当前真实状态”的问题。Fenlie 的方法是把关键状态归回 source-owned artifacts。

---

## 2. 市场与切入点

### Q6：为什么先从 crypto-first 开始？

因为当前 Fenlie 的证据链、研究主线和 execution 边界最匹配 crypto-native quant 团队。它是 wedge，不是终局边界。

### Q7：你们会不会被限制在 crypto？

不会。当前只是从最匹配场景切入，长期可以向 research-driven digital asset fund tech group 与更广义机构控制面扩展。

### Q8：为什么这个市场现在值得做？

因为算法交易市场仍在增长，机构数字资产参与度持续提升，而研究、控制与 execution 的复杂度也在同步上升，基础设施需求比以前更真实。

### Q9：为什么客户不能继续自己拼脚本？

当然可以，但随着团队扩张，脚本、人工记忆和零散页面很快会失效。Fenlie 解决的是“个人能用”到“团队可持续运营”之间的断层。

---

## 3. 研究与策略

### Q10：你们当前最明确的研究主线是什么？

当前 durable research mainline 是 ETHUSDT 15m breakout-pullback，重点在 exit/risk，而不是继续堆 entry filter。

### Q11：你们为什么不把研究结果讲成收益故事？

因为这会弱化 Fenlie 真正的系统价值，也容易造成错误预期。Fenlie 的价值是研究如何被治理、如何进入控制面与后续接管边界，而不是单次回测看起来多漂亮。

### Q12：OI / triad / orderflow 这些增强项现在是什么位置？

OI sidecar 已进入研究增强链；triad 当前仍是 context-only。这个边界是刻意保留的，用来防止 sidecar 越级变成主执行信号。

### Q13：为什么要保留 blocked、watch-only、canonical 这些概念？

因为这能防止团队长期迭代后重新失去上下文。Fenlie 不是只关心“发现什么有用”，也关心“什么被证伪、什么不能晋升”。

---

## 4. Execution 与边界

### Q14：你们 execution 到底做到什么程度？

更准确的说法是：Fenlie 已建立 execution bridge 的关键工程路径与治理框架，适合 probe、ready-check、canary、takeover、rollback 这类验证型流程，但仍应按验证中产品能力表述。

### Q15：为什么不把 execution 当成最大卖点？

因为 execution 最容易被过度营销。Fenlie 的长期价值在于研究—控制—接管的一体化治理，而不是先把 execution 夸成成熟 SaaS。

### Q16：你们有没有稳定接管 live capital？

当前对外不应这么表述。Fenlie 会坚持把验证中能力、工程路径与成熟商用边界区分开。

### Q17：你们如何理解 execution 风险？

不是简单理解为“加几个风控参数”，而是把 readiness、rollback、acceptance、source-of-truth 和 operator 控制一起考虑。

---

## 5. 商业模式与交付

### Q18：你们怎么赚钱？

早期以高价值交付为主，包括私有部署、Research OS 落地、Operator Console Deployment、workflow enablement 与 execution pilot add-on；中长期再提高模块授权占比。

### Q19：为什么不是一开始就纯 SaaS？

因为 Fenlie 解决的是复杂组织问题。早期更合理的方式是通过高价值交付形成模板，而不是假设客户会低摩擦自助购买。

### Q20：你们当前推荐卖什么包？

优先顺序是：Research OS Foundation -> Operator Console Deployment -> Deployment Enablement -> Guarded Execution Pilot Add-on。

### Q21：execution pilot 为什么不是默认首包？

因为如果研究和控制面不稳，execution 只会放大问题。正确路径是先把主线、控制和流程边界做稳。

### Q22：你们怎么避免 pilot 变成无限定制？

通过 scope freeze、milestone、acceptance 标准和降级/回退条件控制，而不是靠临场解释。

---

## 6. 客户与 go-to-market

### Q23：你们当前最理想的 pilot 客户是谁？

是那些已经有一定研究能力、感受到研究—控制—执行链条断裂、又愿意接受私有部署与 pilot 共建的 crypto-native quant 团队。

### Q24：什么样的客户你们不该优先追？

只想买“自动赚钱机器人”、不关心治理、没有明确 owner、或者一开始就希望 Fenlie 全量替换其全部系统的客户。

### Q25：你们怎么判断一个 pilot 值不值得做？

不只看收入，还看能否验证可复用边界、能否沉淀 acceptance 模板、能否帮助 Fenlie 识别最值钱的控制面能力。

---

## 7. 护城河与竞争

### Q26：别人为什么不能很快复制你们？

单点功能可以被复制，但长期维持研究治理、source-owned 状态管理、operator control plane 与 execution boundary 一致性，很难靠拼凑实现。

### Q27：你们的核心壁垒是 AI 吗？

不是。AI 是加速器，真正的壁垒在于系统纪律、证据链与治理方法。

### Q28：你们与 QuantConnect / FMZ / TradingView 的关系是什么？

它们擅长不同层：回测引擎、策略池、脚本社区。Fenlie 占据的空白带是 research governance + operator control plane + guarded execution bridge。

---

## 8. 融资与财务

### Q29：为什么推荐 $2.2M–$2.8M？

因为这一区间最平衡：足以覆盖 18–24 个月 runway，并支撑产品化、研究增强、execution bridge 治理与首批 pilot 落地。

### Q30：为什么你们的三年模型这么保守？

因为当前更重要的是可信度，而不是用激进曲线换短期注意力。Fenlie 更适合先建立真实交付能力与模板复用。

### Q31：你们现在有收入或客户可以公开吗？

若没有新增可审计事实，就不应夸大。当前应把重点放在产品化、pilot 设计与融资用途逻辑上。

### Q32：投资人为什么现在该投，而不是等你们更成熟？

因为当前 Fenlie 已经有明确产品结构和工程证据，但仍处在“投入可以显著提升可复用性”的阶段。现在的资金最能放大产品化收口效率。

---

## 9. 风险与治理

### Q33：Fenlie 最大风险是什么？

不是单点技术能不能写出来，而是产品化节奏：如何把高密度工程资产转成可复用模板，同时不牺牲治理边界。

### Q34：execution 会带来合规风险吗？

任何涉及 execution 的产品都必须认真看待合规边界，因此 Fenlie 当前坚持把 execution 单独降温表达，不把验证中能力包装成成熟金融服务。

### Q35：为什么你们反复强调 public/internal 分层？

因为量化系统里“谁能看到什么、谁基于什么做判断”本身就是治理问题。public/internal 分层是 Fenlie 控制面的重要组成。

### Q36：如果团队扩大，Fenlie 的优势是什么？

团队越大，口径漂移、handoff 混乱和 source-of-truth 丢失越严重。Fenlie 的价值会随着组织复杂度上升而增加。

---

## 10. 未来路线图

### Q37：未来 12 个月最关键的三件事？

1. 把控制台与研究链继续产品化
2. 拿下首批高匹配 pilot 客户并形成模板
3. 把 execution bridge 从工程路径继续推进为更可复用、可治理的产品模块

### Q38：未来 18 个月最重要的判断标准是什么？

不是“加了多少功能”，而是：

- 是否形成稳定交付模板
- 是否有高质量 pilot 结论
- 是否让 execution 更可解释、更可验收、更可回退

### Q39：Fenlie 长期会成为哪类公司？

理想形态是：一家以量化团队控制面、研究治理和受控接管能力为核心的系统基础设施公司，而不是单点策略软件或单点执行工具公司。

---

## 11. 结尾型问题

### Q40：如果只能用一句话总结 Fenlie，怎么说？

Fenlie 正在构建一套量化团队真正缺失的系统层：把研究、控制、风控边界与 execution 接管装进同一条可验证、可审计、可回退的链里。

### Q41：如果投资人只记住一件事，希望是什么？

希望对方记住：Fenlie 的价值不在某个单点功能，而在于它如何让量化团队更接近“组织级可持续运行”。

---

## 12. 一句话标准

> FAQ 扩展版的任务，不是把每个问题都答得更大，而是让所有延伸追问最终都回到 Fenlie 的系统边界与产品化逻辑上。
