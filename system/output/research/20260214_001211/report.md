# 实盘多模式回测与优化报告 | 2026-02-14

## 运行概览
- 时间预算: `0.04` 小时
- 实际耗时: `0.06` 小时
- 区间: `2015-01-01 ~ 2026-02-13`
- 覆盖标的数: `40`
- 行情记录数: `92560`
- 新闻记录数: `250`
- 研报记录数: `3905`

## 模式结果
### swing
- trials: `2`
- best_score: `-3.750022`
- best_params: `{'hold_days': 4, 'max_daily_trades': 3, 'signal_confidence_min': 39.24149425590886, 'convexity_min': 2.1272857535042933, 'news_weight': 0.6973680290593639, 'report_weight': 0.09417734788764953}`
- best_metrics: `{'annual_return': -0.0009223303166532304, 'max_drawdown': 0.5407713446578746, 'positive_window_ratio': 0.9175905395417591, 'trades': 926, 'factor_alignment': -0.010143591697139477, 'score': -3.7500222922318267}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_001211/swing/trials.csv`

## 说明
- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。
- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。
- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。
