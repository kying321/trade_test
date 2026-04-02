# Fenlie Deployment & Enablement Package One-Pager

> 版本：2026-03-24  
> 用途：定义 Fenlie 当前最合理的早期交付包结构，帮助创始人向投资人或 pilot 客户解释“钱怎么换成什么交付”。  
> 原则：**先卖高价值交付包，不急着假装成低价自助 SaaS。**

---

## 1. 一句话结论

Fenlie 当前最合理的商业化入口，不是一个“开箱即用、零服务”的通用 SaaS 套餐，而是：

> **以私有部署、研究链落地、控制台定制、execution pilot 与 enablement 为核心的高价值交付包。**

这既符合当前产品成熟度，也符合目标客户的真实购买方式。

---

## 2. 为什么当前要以交付包为主

原因很简单：

1. Fenlie 解决的是复杂组织问题，不是单个表单工具问题；
2. 客户真正买的不是页面，而是：研究治理、控制链、接管边界与 operator workflow；
3. execution bridge 仍处验证中，更适合按 pilot / add-on 交付；
4. 早期高价值交付能反向帮助产品标准化。

因此，早期交付包应被视为：

- 现金流入口
- 产品模板沉淀入口
- 客户场景验证入口

---

## 3. 当前建议的 4 类交付包

### Package A｜Research OS Foundation

**适合客户：**

- 已有研究能力，但主线管理混乱的团队

**核心交付：**

- 研究主线梳理
- review artifact 结构落地
- canonical anchor / next priority 规则化
- 回测 / review 输出收口

**客户拿到的价值：**

- 研究不再散落在脚本与会话中
- 新成员接手成本下降
- 后续控制台与 contracts 有清晰上游

**当前适配度：高**

---

### Package B｜Operator Console Deployment

**适合客户：**

- 已有一定研究链，希望建立更真实的运营控制面

**核心交付：**

- overview / workspace / contracts / raw / alignment 的部署与调优
- public / internal 分层梳理
- source-head / gate / artifact 显示结构收口
- 验收链与发布前检查接入

**客户拿到的价值：**

- dashboard 从展示层变成控制层
- source-of-truth 更易被团队识别
- operator 能更快定位异常与当前主线

**当前适配度：高**

---

### Package C｜Deployment Enablement & Workflow Advisory

**适合客户：**

- 系统已有雏形，但团队流程混乱、角色协作断裂的团队

**核心交付：**

- 角色与流程梳理
- release-checklist / acceptance / runbook 设计
- operator 日常工作流与 handoff 规范
- public/internal surface 的使用边界定义

**客户拿到的价值：**

- 系统不是“装上了”，而是真能跑起来
- 降低团队对单点个人记忆的依赖
- 为更深的 execution pilot 打基础

**当前适配度：高**

---

### Package D｜Guarded Execution Pilot Add-on

**适合客户：**

- 已理解 Fenlie 主体价值，且有清晰 execution 场景与 owner 的高成熟度团队

**核心交付：**

- probe / ready-check / canary / rollback 流程设计或接入
- execution bridge pilot 范围界定
- 接管治理、验收与边界说明
- 与上游研究 / 控制面的对接

**客户拿到的价值：**

- execution 不再是黑盒自动化
- 团队能用更受控方式推进接管验证
- 为后续复用 execution 模块沉淀模板

**当前适配度：中（必须严格控口径）**

说明：这不是默认首包，而是 add-on / pilot 包。

---

## 4. 推荐售卖顺序

Fenlie 当前最推荐的售卖顺序：

1. **Research OS Foundation**
2. **Operator Console Deployment**
3. **Deployment Enablement & Workflow Advisory**
4. **Guarded Execution Pilot Add-on**

逻辑是：

- 先把研究和控制面做稳
- 再把流程与 operator 能力做起来
- 最后才推进 execution pilot

这与 Fenlie 当前成熟度完全一致。

---

## 5. 不建议的售卖方式

当前不建议：

- 一上来卖 execution 全包
- 一上来承诺全面替换客户现有系统
- 一上来做低价自助 SaaS 打包页
- 一上来用策略收益做成交主叙事

这些方式都会把 Fenlie 拉向不适合当前阶段的商业形态。

---

## 6. 交付周期建议

### 短周期包（2–6 周）

适合：

- Research OS Foundation
- Workflow Advisory

特点：

- 切问题更快
- 更容易成交
- 更容易形成模板

### 中周期包（4–10 周）

适合：

- Operator Console Deployment
- Console + contracts / alignment 调优

特点：

- 对产品实物感知更强
- 更容易沉淀截图与案例资产

### Pilot 周期包（6–12+ 周）

适合：

- Guarded Execution Pilot Add-on

特点：

- 必须有清晰 owner
- 必须有边界、验收与 rollback 说明
- 必须受控，不可模糊扩大范围

---

## 7. 成交与验收时应强调什么

客户或投资人最需要听到的不是“我们能做很多事”，而是：

- 这次交付的边界是什么
- 完成后客户会得到哪类系统能力
- 哪些是已验证能力，哪些是验证中能力
- 什么叫验收通过
- 什么情况下只能做 pilot，不能做产品承诺

因此每个交付包都应有：

1. 范围定义
2. 所需客户配合
3. 交付物清单
4. 验收标准
5. 风险边界

---

## 8. 与财务模型的关系

当前这些交付包的存在意义，不只是“怎么报价”，更是解释：

- 为什么 Fenlie 早期收入以高价值交付为主
- 为什么三年模型不是从一开始就靠订阅放量
- 为什么 execution pilot 更像高价 add-on，而不是默认主包

因此，交付包结构应与以下文档保持一致：

- `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-tam-sam-som-and-3y-financial-model.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-fundraise-scenarios-and-runway.md`

---

## 9. 推荐对外话术

### 对投资人

> Fenlie 早期不是靠一个通用自助 SaaS 起量，而是通过私有部署、控制台落地、workflow enablement 与 execution pilot 逐步沉淀可复用模板，再提高模块化授权比例。

### 对潜在 pilot 客户

> 我们不会一上来卖给你一个“万能大平台”，而是先从最值钱、最能落地的研究链、控制面和受控接管环节开始，逐步把系统搭稳。

---

## 10. 一句话标准

> Fenlie 当前最好的商业化方式，不是卖一个“看起来完整”的产品壳，而是卖一组能把客户研究、控制与受控接管真正做稳的高价值交付包。
