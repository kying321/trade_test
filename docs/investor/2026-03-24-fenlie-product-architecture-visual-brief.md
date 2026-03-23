# Fenlie Product Architecture Visual Brief

> 版本：2026-03-24  
> 用途：产出一张可放进路演 deck、BP、尽调数据室或会后补充邮件中的“Fenlie 产品架构单页图”。  
> 目标：让投资人在 **30 秒内** 明白 Fenlie 不是策略脚本，不是单一 dashboard，也不是脱离治理的 execution 工具，而是一套把研究、控制、受控接管串起来的系统。

---

## 1. 这张图必须回答的 4 个问题

无论设计成 PPT 页、PDF 一页还是 Figma 结构图，都必须在一张图里回答：

1. **Fenlie 的核心模块是什么？**
2. **这些模块之间如何传导？**
3. **source-of-truth 在哪里？**
4. **execution 为什么不能脱离治理单独理解？**

如果一张图不能同时回答这四个问题，就还不够好。

---

## 2. 推荐单页结构

最推荐的画法不是复杂系统图，而是 **四层主结构 + 两条横向治理带**。

### 2.1 四层主结构（自左向右或自上而下均可）

#### Layer A｜Research OS

放在最上游，表达“研究不是散脚本，而是被治理的系统入口”。

建议子标签：

- strategy research
- backtest / review
- canonical anchor
- next priority

#### Layer B｜Source-Owned Artifact & Control Chain

这层是整张图的核心。  
必须让投资人明白：Fenlie 的真正中枢不是 UI，而是 source-owned artifact 与控制链。

建议子标签：

- source-owned artifacts
- review / handoff
- contracts
- gates / blockers

#### Layer C｜Operator Console

表达 Fenlie 的控制台是“控制面”，不是展示层。

建议子标签：

- overview
- workspace
- raw / contracts
- alignment / terminal

#### Layer D｜Guarded Execution Bridge

放在最右侧或最下游，清楚显示它是系统的受控出口，不是独立主角。

建议子标签：

- probe
- ready-check
- canary
- takeover / rollback

### 2.2 两条横向治理带

建议在四层结构之上或之下，再叠两条横向带：

#### 横带 1｜Risk / Governance

用来表达：

- source-of-truth discipline
- release-checklist
- readiness
- rollback
- auditability

#### 横带 2｜Visibility / Deployment Boundary

用来表达：

- public / internal separation
- pilot / validation boundary
- private deployment / enablement

这两条横带很重要，因为它们把 Fenlie 从“功能拼图”提升成“治理系统”。

---

## 3. 最推荐的构图方案

### 方案 A｜左到右传导图（推荐）

```text
Research OS
   ↓
Source-Owned Artifact & Control Chain
   ↓
Operator Console
   ↓
Guarded Execution Bridge
```

在每层旁边用 3–4 个关键词辅助。  
上方悬浮两条 governance 带：

- Risk / Governance
- Visibility / Deployment Boundary

**优点：**

- 投资人最好理解
- 最适合路演 slide
- 容易与“问题 -> 解决方案 -> execution 边界”叙事串联

### 方案 B｜中轴控制面图

中间放 `Source-Owned Artifact & Control Chain`，上下左右分别挂：

- 上：Research OS
- 左：External inputs / strategy ideas
- 右：Operator Console
- 下：Guarded Execution Bridge

**优点：**

- 更能凸显 source-of-truth 中枢地位

**缺点：**

- 对第一次接触的投资人稍微抽象

### 方案 C｜四象限图（不推荐优先）

除非版面很受限，否则不建议用四象限图。原因是它容易把传导关系讲弱，只剩“我们有四块模块”的平铺感。

---

## 4. 推荐图上必须出现的文字

建议主标题：

**Fenlie Architecture: Research -> Control -> Guarded Execution**

建议副标题：

**把研究、source-owned 控制链、运营控制面与受控执行接管收敛到同一条系统路径**

建议每层只保留如下短句：

### Research OS

> 将研究主线、回测、review 与 canonical anchor 收敛为团队资产。

### Source-Owned Artifact & Control Chain

> 用 artifact、contracts、handoff、gate 固定系统当前真实状态。

### Operator Console

> 将 overview、workspace、contracts、alignment、terminal 做成真正的运营控制面。

### Guarded Execution Bridge

> 通过 probe、ready-check、canary、rollback 实现受控接管，而不是黑盒自动化。

---

## 5. 视觉分层建议

### 5.1 色彩

- Research OS：冷蓝
- Source-Owned Artifact / Control Chain：亮蓝白或高对比青蓝（视觉中心）
- Operator Console：中性冷灰蓝
- Guarded Execution Bridge：琥珀偏黄 + `验证中` badge

治理横带：

- Risk / Governance：低饱和冷绿
- Visibility / Deployment Boundary：低饱和灰白

### 5.2 视觉重心

整张图必须把 **Source-Owned Artifact & Control Chain** 做成视觉中心。  
这是 Fenlie 与普通产品最不一样的地方。

其次是：

- Operator Console：让人看到“控制面”
- Execution Bridge：让人看到“受控出口，但不是夸大主角”

### 5.3 maturity 标记

至少两个地方要打 badge：

- Research OS：`已验证`
- Operator Console：`已验证`
- Guarded Execution Bridge：`验证中`

如需标 `Private Deployment / Enablement`，建议放在图外侧，标 `规划中 / pilot friendly`。

---

## 6. 一页图上不要出现什么

不要出现：

- 八层技术明细全部展开
- 太多文件名、脚本名、命令名
- 数据结构细节
- 具体交易所 logo 或交易品种 logo
- 把 execution 画成最亮、最大、最像收入中心的一层

这张图是投资级架构图，不是内部工程图。

---

## 7. 路演中的讲法（60 秒版）

建议口播：

> Fenlie 的核心不是某个策略，而是这条系统链。最上游是 Research OS，把研究和 review 收敛成团队资产；中间是 source-owned artifact 与控制链，决定系统当前真实状态；外显层是 Operator Console，让团队能在 public 与 internal 边界内做运营控制；最下游或最右侧才是 Guarded Execution Bridge，它不是黑盒自动化，而是通过 probe、ready-check、canary 与 rollback 做受控接管。这也是为什么我们说 Fenlie 更像量化团队的控制面基础设施，而不是单一回测器或执行器。

---

## 8. 可直接给设计师的布局稿

### 横版 16:9（路演最推荐）

- 左 20%：Research OS
- 中 30%：Source-Owned Artifact & Control Chain（最大）
- 右 25%：Operator Console
- 最右 15%：Guarded Execution Bridge
- 顶部两条横带：Risk/Governance + Visibility/Deployment Boundary
- 底部：一句总括 + maturity legend

### 竖版 A4 / PDF（尽调附件推荐）

- 顶部：一句定位 + maturity legend
- 中部四层垂直堆叠
- 左右侧边栏：治理横带说明
- 底部：3 条“不要误解成什么”的反向定义

---

## 9. 反向定义（建议放在页脚）

建议在图页底部放一行很小但很关键的反向定义：

- 不只是回测平台
- 不只是 dashboard
- 不只是执行机器人
- 而是量化团队的研究—控制—受控接管系统

这行字对投资人建立正确定位非常有效。

---

## 10. 建议与哪些材料一起用

这张图最适合搭配：

1. `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-pitch-deck-full-copy.md`
2. `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-pitch-deck-visual-brief.md`
3. `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-execution-maturity-note.md`
4. `/Users/jokenrobot/Downloads/Folders/fenlie/docs/investor/2026-03-24-fenlie-business-plan.md`

如果只给投资人看一张架构图，优先给这一张。

---

## 11. 一句话制作标准

> 一张好的 Fenlie 架构图，必须让投资人一眼看见：真正掌控系统的是 source-owned control chain，而 execution 只是这条链上的受控出口。
