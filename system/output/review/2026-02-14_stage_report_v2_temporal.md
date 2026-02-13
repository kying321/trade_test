# 阶段性报告 v2（时间隔离版）

- 生成时间：2026-02-14
- 回测主区间：2015-01-01 ~ 2026-02-13
- 评估目标：现有架构与算法下的风险控制能力、胜率、盈亏比

## 1) 架构状态（当前）

- 已完成主链路：`data -> regime -> signal -> risk -> backtest -> review -> reporting`。
- 研究与执行分离：输出交易建议，不直接实盘下单。
- 数据源：AkShare 主源 + yfinance 回退；新闻与研报接入并按日聚合。
- 关键新增：**严格时间隔离机制**
  - 回测只使用截止日及之前信息（K线/新闻/研报）。
  - 截止日之后 `review_days` 的资料只进入复盘通道，不参与回测评分。
  - 工件拆分为：
    - `news_daily_pre_cutoff.csv`
    - `report_daily_pre_cutoff.csv`
    - `news_daily_post_cutoff_review.csv`
    - `report_daily_post_cutoff_review.csv`

## 2) 风险控制能力（实测结论）

- 已具备的风控结构：
  - Kelly 折扣仓位、总暴露/单品种/主题约束。
  - ATR 极端波动保护、重大事件窗口禁开仓、连亏冷却。
  - 优化目标已加入回撤超限重罚与零交易惩罚。
- 结果结论：
  - 结构完整，但参数层尚未把风险压到发布门槛。
  - 在真实长区间样本下，三模式最大回撤仍显著超过 18%，当前仅适合“研究态”。

## 3) 胜率与盈亏比（payoff）

口径说明：以下为三模式当前代表性最佳参数重放结果（同一区间、同一数据口径）。

- `ultra_short`
  - 胜率（win_rate）：**45.54%**
  - 盈亏比（payoff / profit_factor）：**1.145**
  - 年化收益：1.57%
  - 最大回撤：39.28%
  - 交易数：1322
- `swing`
  - 胜率：**43.95%**
  - 盈亏比：**1.027**
  - 年化收益：-0.09%
  - 最大回撤：54.08%
  - 交易数：926
- `long`
  - 胜率：**33.45%**
  - 盈亏比：**1.211**
  - 年化收益：15.13%
  - 最大回撤：60.64%
  - 交易数：562

## 4) 阶段判断

- 系统已经满足“可持续迭代研究平台”标准。
- 系统尚未满足“发布就绪”标准，核心瓶颈是回撤控制。
- 下一阶段优先级：
  1. 先把三模式回撤压到 `<=18%`（否则不进入发布态）。
  2. 在风控约束稳定后，再优化收益与因子权重。

## 5) 关键工件

- 阶段报告：`/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/2026-02-14_stage_report_v2_temporal.md`
- 三模式结果目录：
  - `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_000819`
  - `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_001211`
  - `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_001554`
- 时间隔离验证跑（样例）：
  - `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_073624`
