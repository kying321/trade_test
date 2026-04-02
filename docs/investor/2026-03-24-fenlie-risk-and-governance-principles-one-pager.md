# Fenlie Risk & Governance Principles One-Pager

> 版本：2026-03-24  
> 用途：用一页纸解释 Fenlie 为什么不是“更激进的自动交易工具”，而是把风险治理嵌入研究、控制与受控接管的系统。  
> 原则：**先讲治理边界，再讲能力边界。**

---

## 1. 一句话结论

Fenlie 的风险治理核心，不是“多加几个风控开关”，而是：

> **把研究、source-of-truth、门禁、operator 控制与 execution 接管统一到一条可审计、可回退、可分层表达的系统链里。**

---

## 2. Fenlie 的治理原则

### 原则 1｜Source-of-Truth 优先于展示层

Fenlie 的设计前提是：

- lane
- queue
- gate
- blocker
- review state
- research mainline

这些关键状态必须属于 source-owned artifacts，而不是只存在于 dashboard 文案或人工记忆里。

**为什么重要：**

没有 source-of-truth，就没有真正的 operator control plane。

---

### 原则 2｜研究、控制与执行边界必须分层

Fenlie 不把研究信号、operator 视图和 execution 接管混成一个黑盒。

正确关系是：

- 研究层：形成可审计结论
- 控制层：暴露当前状态、门禁与异常
- 接管层：在受控前提下推进 execution 验证

**为什么重要：**

这让系统更容易解释“现在允许做什么、不允许做什么”。

---

### 原则 3｜验证中能力不能包装成成熟能力

Fenlie 当前明确区分：

- 已验证
- 验证中
- 规划中

尤其对 execution bridge，必须坚持：

- 可谈 pilot / probe / ready-check / canary / rollback
- 不夸成成熟规模化 execution SaaS

**为什么重要：**

治理不是写在 PR 里，而是写在对外口径与产品边界里。

---

### 原则 4｜失败分级必须清楚

Fenlie 默认不把所有故障都当作“应立刻停机”的同类问题。

治理上要区分：

- execution ambiguity
- source-of-truth ambiguity
- process fault
- notification / dashboard degradation

**为什么重要：**

否则团队会被低优先级故障牵着走，也会让真正高风险问题被淹没。

---

### 原则 5｜回退能力与晋升能力同等重要

Fenlie 不是只关心“如何上线”，同样关心：

- 如何 canary
- 如何 rollback
- 如何保留 blocked / watch-only 结论
- 如何不让 sidecar 越级变成主线

**为什么重要：**

没有回退，就没有真正的受控接管。

---

## 3. 对投资人最重要的治理含义

从投资视角看，Fenlie 的风险治理价值在于：

1. 让产品不是靠单人经验运行
2. 让系统状态更可解释
3. 让 execution 扩张建立在治理基础上
4. 让产品化收口更可信，而不是更激进

这也是为什么 Fenlie 的工程纪律本身就是商业壁垒的一部分。

---

## 4. 当前应如何表述 execution 风险治理

推荐表述：

> Fenlie 当前 execution bridge 的关键价值，不是“更快自动化”，而是通过 probe、ready-check、canary、rollback 等受控流程，把执行风险纳入系统治理边界。

不推荐表述：

- 已经自动化解决 execution 风险
- 已具备成熟大规模生产接管能力
- 已形成标准化低门槛 execution 网络

---

## 5. 当前最稳的治理卖点

如果只允许说 4 个点，建议说：

1. source-owned artifact discipline
2. public / internal 分层
3. release-checklist / acceptance / rollback 纪律
4. execution bridge 的受控验证边界

这 4 个点最能代表 Fenlie 不是“普通量化工具”。

---

## 6. 一句话标准

> Fenlie 的治理价值，不在于把系统包装得更自动，而在于让团队更清楚地知道：当前什么是真的、什么被允许、什么应该先停下来。
