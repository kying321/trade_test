# 阶段性报告（反脆弱交易系统）

- 报告时间：2026-02-14
- 回测区间：2015-01-01 ~ 2026-02-13
- 样本范围：40标的（A股/ETF/期货混合），含新闻与研报因子
- 评估口径：当前最新算法与风险惩罚版本（risk-v2）

## 1. 现有架构状态

- 分层架构已落地：`data -> regime -> signal -> risk -> backtest -> review -> reporting`。
- 研究执行分离：实盘建议输出与参数更新闭环已实现，不直接自动下单。
- 数据侧能力：
  - 行情抓取：AkShare 主源 + yfinance 回退。
  - 新闻/研报：`stock_news_em` + `stock_research_report_em`，按日聚合因子。
  - 缓存机制：同参数下命中本地缓存，减少重复抓取耗时。
- 调度与测试：CLI 全链路与 `lie test-all` 可执行，最新全量测试通过。

## 2. 风险控制能力评估（当前版本）

- 已实现的控制机制：
  - 仓位风险框架（Kelly折扣、组合暴露上限、主题/单品种约束）。
  - 体制冲突降级、重大事件窗口禁开仓、连亏冷却。
  - ATR 极端波动保护与门禁降级。
  - 目标函数新增“回撤超限重罚 + 零交易惩罚 + 最低交易量约束”。
- 实测结果结论：
  - 结构上已有风险治理能力，但在真实长区间回测下，回撤仍显著超限。
  - 三模式均触发 `violations=1`（主要因 `max_drawdown > 18%`），说明参数层尚未把风控目标压到可发布水平。

## 3. 胜率与盈亏比（payoff/profit factor）

- Ultra-short：
  - win_rate = 45.54%
  - payoff(profit_factor) = 1.145
  - annual_return = 1.57%
  - max_drawdown = 39.28%
  - trades = 1322
- Swing：
  - win_rate = 43.95%
  - payoff(profit_factor) = 1.027
  - annual_return = -0.09%
  - max_drawdown = 54.08%
  - trades = 926
- Long：
  - win_rate = 33.45%
  - payoff(profit_factor) = 1.211
  - annual_return = 15.13%
  - max_drawdown = 60.64%
  - trades = 562

## 4. 当前判断

- 收益结构具备一定凸性迹象（尤其 long 的 payoff>1 且年化为正），但风险约束明显未达标。
- 阶段结论：
  - 目前系统可用于“研究与参数筛选”，不满足“发布就绪/生产级风控”标准。
  - 下一阶段必须优先压缩回撤，再追求收益优化。

## 5. 关键工件

- 报告：`/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/review/2026-02-14_stage_report.md`
- 三模式回测：
  - `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_000819`
  - `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_001211`
  - `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_001554`
- 全量测试结果：`/tmp/lie_test_all_after_changes.json`
