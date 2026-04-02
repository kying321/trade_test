# Fenlie Data Room Index & Request List

> 版本：2026-03-24  
> 用途：给创始人、投资人或顾问一个统一的数据室骨架，明确：现在能放什么、还缺什么、哪些内容不能过度承诺。  
> 原则：**先有结构，再逐步填充。** 没有可审计事实的栏目宁可标记缺失，也不要补虚构内容。

---

## 1. 当前结论

Fenlie 现阶段已适合建立一个**早期融资级数据室**，但不适合伪装成“成熟规模化 SaaS 公司”的重型数据室。

更准确的做法是：

- 先建立一套 **Founding / Seed 阶段可用的数据室骨架**
- 以产品、架构、证据、融资逻辑为主
- 对 traction、收入、合规成熟度保持克制

这意味着 Fenlie 数据室的中心应该是：

1. 项目定位
2. 产品与架构
3. 研究与控制面证据
4. execution maturity boundary
5. 融资用途与里程碑

而不是：

- 大量虚构客户证明
- 过度完整的财务历史
- 不存在的规模化运营报表

---

## 2. 推荐数据室目录结构

建议建立如下目录：

```text
fenlie-data-room/
  01_company_overview/
  02_product_and_architecture/
  03_market_and_gtm/
  04_financial_model_and_fundraise/
  05_evidence_and_validation/
  06_execution_boundary_and_risk_governance/
  07_pilot_program/
  08_legal_and_admin/
  09_appendix/
```

---

## 3. 每个目录建议放什么

### 01_company_overview

**目的：** 先让投资人理解 Fenlie 是什么，不是什么。

建议放入：

1. Executive Summary  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-executive-summary.md`
2. Business Plan  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-business-plan.md`
3. Roadshow Outline  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-roadshow-outline.md`
4. Investor Pack Index  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-investor-pack-index.md`

**当前状态：Ready**

---

### 02_product_and_architecture

**目的：** 证明 Fenlie 是系统，不是零散脚本。

建议放入：

1. Pitch Deck Full Copy  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-pitch-deck-full-copy.md`
2. Pitch Deck Visual Brief  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-pitch-deck-visual-brief.md`
3. Product Architecture Visual Brief  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-product-architecture-visual-brief.md`
4. Architecture Review（内部来源）  
   `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/ARCHITECTURE_REVIEW.md`

**当前状态：Ready / 可精选对外**

说明：Architecture Review 适合作为内部支撑件，不一定直接首轮对外发原文。

---

### 03_market_and_gtm

**目的：** 证明这不是伪需求，并说明为什么从 crypto-first 切入。

建议放入：

1. Market & Competitive Landscape  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-market-and-competitive-landscape.md`
2. Pilot Customer Profile One-Pager  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-pilot-customer-profile-one-pager.md`
3. Deployment & Enablement Package One-Pager  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-deployment-enablement-package-one-pager.md`

**当前状态：Partial -> 本轮补齐骨架后可用**

---

### 04_financial_model_and_fundraise

**目的：** 说明融资金额、runway 与资源投入逻辑。

建议放入：

1. TAM / SAM / SOM and 3Y Financial Model  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-tam-sam-som-and-3y-financial-model.md`
2. Fundraise Scenarios and Runway  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-fundraise-scenarios-and-runway.md`

**当前状态：Ready（但属于规划假设，不是业绩承诺）**

---

### 05_evidence_and_validation

**目的：** 让投资人看到 Fenlie 有真实工程证据，而不只是故事。

建议放入：

1. Investor Evidence Matrix  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-investor-evidence-matrix.md`
2. Investor Q&A Battlecard  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-investor-qa-battlecard.md`
3. Diligence Readiness Checklist  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-diligence-readiness-checklist.md`
4. 公开/内部 acceptance 与架构验证的精选截图或摘要（后续可补）

**当前状态：Ready / Partial**

说明：精选截图与摘要可以后续补，不必第一天就追求完整数据室。

---

### 06_execution_boundary_and_risk_governance

**目的：** 单独管理 execution 的成熟度边界，避免对外口径失真。

建议放入：

1. Execution Maturity Note  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-execution-maturity-note.md`
2. OpenClaw Cloud Bridge（内部支撑）  
   `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md`
3. 风险治理原则页（后续建议补一页对外摘要）

**当前状态：Partial**

说明：execution 的正确做法是单独分区，而不是混进产品宣传页里。

---

### 07_pilot_program

**目的：** 让投资人看到“钱怎么转化为可验证客户路径”。

建议放入：

1. Pilot Customer Profile One-Pager  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-pilot-customer-profile-one-pager.md`
2. Deployment & Enablement Package One-Pager  
   `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-deployment-enablement-package-one-pager.md`
3. 未来可补：Pilot milestone / readiness template / acceptance template

**当前状态：Partial -> 本轮补齐核心一页件后可推进**

---

### 08_legal_and_admin

**目的：** 给后续更深尽调留入口。

建议放入：

- 公司主体信息
- cap table（若已有且适合披露）
- 轮次文件
- 主要协议模板
- 顾问 / 法务状态说明

**当前状态：Not Ready / 视真实情况逐步补充**

说明：如果现在没有可披露内容，宁可留空并标记“upon request”。

---

### 09_appendix

**目的：** 存放不适合放主线但能支撑深问的资料。

建议放入：

- 参考市场来源
- 演示版截图
- 术语表
- 对外版本更新记录

**当前状态：Partial**

---

## 4. 数据室资料 readiness 表

| 目录 | 当前状态 | 适合现在开放吗 | 备注 |
|---|---|---:|---|
| 01_company_overview | Ready | 是 | 可作为首轮共享包 |
| 02_product_and_architecture | Ready/Partial | 是 | 建议精选，不全量抛内部工程文档 |
| 03_market_and_gtm | Partial | 是 | 本轮新增 one-pager 后更完整 |
| 04_financial_model_and_fundraise | Ready | 是 | 明确为规划假设 |
| 05_evidence_and_validation | Ready/Partial | 是 | 是当前可信度核心 |
| 06_execution_boundary_and_risk_governance | Partial | 是，但需控口径 | 不要过度承诺 |
| 07_pilot_program | Partial | 是 | 适合会后补发 |
| 08_legal_and_admin | Not Ready / unknown | 否或按需 | 必须基于真实披露条件 |
| 09_appendix | Partial | 视场景 | 做辅助不做主包 |

---

## 5. 第一版数据室建议开放的最小集合

如果现在就要发一个“简版数据室”，建议只放：

1. Executive Summary
2. Business Plan
3. Market & Competitive Landscape
4. TAM / SAM / SOM & 3Y model
5. Investor Evidence Matrix
6. Diligence Readiness Checklist
7. Execution Maturity Note
8. Product Architecture Visual Brief
9. Pilot Customer Profile One-Pager
10. Deployment & Enablement Package One-Pager

这 10 份已经足够支撑早期认真讨论。

---

## 6. 投资人 request list 模板

若投资人进入下一轮，可用下面清单作为 request tracker：

### A. 必发

- [ ] Executive Summary
- [ ] Business Plan
- [ ] Financial Model / Runway
- [ ] Evidence Matrix
- [ ] Execution Maturity Note

### B. 按需发

- [ ] Product Architecture Visual
- [ ] Pilot Customer Profile
- [ ] Deployment Package One-Pager
- [ ] Q&A Battlecard
- [ ] 市场与竞品说明

### C. 进入深入 diligence 后再考虑

- [ ] 法务主体材料
- [ ] cap table
- [ ] 协议模板
- [ ] 更正式的治理与安全摘要
- [ ] pilot readiness / acceptance 模板

---

## 7. 数据室红线

数据室里当前不要放：

- 虚构客户 logo
- 未审计收入口径
- 含糊的 AUM / 代客资金表述
- 把 execution pilot 写成成熟商用 SaaS
- 任何收益承诺或策略保证

---

## 8. 下一步最值钱的动作

完成本文件后，最值得继续补的是：

1. 一版可发送 PDF 数据室目录封面
2. Pilot milestone / acceptance 模板
3. 一页风险与治理原则摘要
4. 如果有真实进展，再逐步补 legal/admin 资料

---

## 9. 一句话标准

> Fenlie 数据室的目标，不是看起来“像一家更大公司”，而是让投资人看到：这是一套边界清楚、证据真实、适合继续下注产品化收口的系统项目。
