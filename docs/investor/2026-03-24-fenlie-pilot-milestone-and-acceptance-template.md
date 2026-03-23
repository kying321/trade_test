# Fenlie Pilot Milestone & Acceptance Template

> 版本：2026-03-24  
> 用途：定义 Fenlie pilot 项目的标准推进框架，避免 pilot 变成无限扩 scope 的模糊定制。  
> 原则：**先定义边界与验收，再讨论扩展。**

---

## 1. 模板用途

这不是客户合同，而是 Fenlie 在推进 pilot 前应先内部对齐、再对外确认的一页模板。

它回答 5 个问题：

1. 本次 pilot 做什么
2. 本次 pilot 不做什么
3. 每个阶段的里程碑是什么
4. 什么叫验收通过
5. 哪些风险一旦出现必须降级或回退

---

## 2. Pilot 基础信息模板

- **客户名称：**
- **客户类型：** Crypto-native quant team / research-driven fund tech group / other
- **Pilot 类型：**
  - Research OS Foundation
  - Operator Console Deployment
  - Workflow Enablement
  - Guarded Execution Pilot Add-on
- **客户 owner：**
- **Fenlie owner：**
- **起止时间：**
- **边界说明：**
- **不在范围内的内容：**

---

## 3. 标准里程碑结构

### Milestone 0｜Discovery & Boundary Freeze

**目标：**

- 明确当前系统问题
- 确认本次 pilot 的边界
- 确认客户 owner、Fenlie owner、沟通节奏

**输出物：**

- 当前问题列表
- 范围清单
- 成功标准草案
- 风险边界说明

**通过标准：**

- 双方确认“做什么 / 不做什么”
- 不再用模糊愿景替代 scope

---

### Milestone 1｜Architecture / Workflow Mapping

**目标：**

- 把客户当前研究、控制、接管流程映射出来
- 识别 source-of-truth 缺口、handoff 断点、operator 痛点

**输出物：**

- 当前架构映射
- 问题优先级列表
- 推荐接入路径

**通过标准：**

- 客户确认映射结果基本准确
- Fenlie 确认本次 pilot 不会被拉成全面系统重构

---

### Milestone 2｜Core Delivery

**目标：**

- 完成本次 pilot 最核心的交付内容

**可选内容按包不同替换：**

- Research OS：主线 / review / canonical anchor / next priority
- Console：workspace / contracts / alignment / public/internal
- Enablement：release-checklist / runbook / operator workflow
- Execution pilot：probe / ready-check / canary / rollback 受控路径

**输出物：**

- 已部署或已接入内容
- 核心交付说明
- 使用与限制说明

**通过标准：**

- 客户能够真实使用核心交付
- 输出物与最初 scope 一致
- 仍保留边界，不因“顺手加一点”失控扩 scope

---

### Milestone 3｜Acceptance & Evidence Capture

**目标：**

- 按预先约定的 acceptance 标准做验收
- 形成可复用证据与经验

**输出物：**

- 验收结果
- 问题列表
- 后续建议：close / extend / phase-2

**通过标准：**

- 验收指标逐项有结论
- 结果可归类为：通过 / 部分通过 / 不通过
- 不把“客户感觉还行”当成唯一验收方式

---

## 4. 验收标准模板

### A. 范围一致性

- [ ] 实际交付与 scope 一致
- [ ] 未发生未经确认的大范围扩项
- [ ] 客户知道哪些能力仍不在范围内

### B. 组织可用性

- [ ] 客户 owner 能独立复述新流程/新界面用途
- [ ] 至少 1 名非 owner 角色也能使用关键交付
- [ ] 关键状态不再完全依赖人工口头同步

### C. 系统可解释性

- [ ] 当前主线 / gate / 控制边界更清楚
- [ ] operator 能更快定位当前状态
- [ ] blocked / watch-only / canonical 等边界不再混乱

### D. 风险边界清晰度

- [ ] 客户理解哪些是已验证能力
- [ ] 客户理解哪些只是验证中或 pilot 能力
- [ ] execution 相关能力没有被误解成成熟商用承诺

### E. 后续判断

- [ ] 值得进入下一阶段
- [ ] 需要先修复当前问题
- [ ] 不适合继续扩展

---

## 5. Pilot 失败或降级条件模板

出现以下任一情况，应考虑暂停、降级或回退：

1. 客户 owner 缺失或长期不可用
2. 范围持续漂移且无法冻结
3. 客户将验证中 execution 能力持续误解为成熟承诺
4. source-of-truth 无法确定，导致交付对象本身不稳定
5. 双方对验收标准始终无法达成一致

这类情况不应靠“继续多做一点”来掩盖。

---

## 6. Phase-2 决策模板

当 pilot 接近完成时，只做以下三种决策之一：

### 结果 A｜Close

- pilot 目标已完成
- 当前不进入下一阶段

### 结果 B｜Extend Narrowly

- 目标基本成立
- 仅围绕少量明确问题延伸

### 结果 C｜Phase-2 Upgrade

- 已验证价值明显
- 进入更高一层交付包
- 重新定义 scope、时间、验收

不建议出现“含糊继续合作”这种没有边界的状态。

---

## 7. 一句话标准

> Fenlie 的 pilot 不应证明“我们什么都能做”，而应证明：我们能在明确边界内，把最值钱的研究、控制或受控接管能力落下来，并留下可复用模板。
