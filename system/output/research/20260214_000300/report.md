# 实盘多模式回测与优化报告 | 2026-02-14

## 运行概览
- 时间预算: `0.06` 小时
- 实际耗时: `0.07` 小时
- 区间: `2015-01-01 ~ 2026-02-13`
- 覆盖标的数: `40`
- 行情记录数: `92560`
- 新闻记录数: `250`
- 研报记录数: `3905`

## 模式结果
### ultra_short
- trials: `2`
- best_score: `-0.289265`
- best_params: `{'hold_days': 1, 'max_daily_trades': 7, 'signal_confidence_min': 42.183439595852676, 'convexity_min': 2.223597040891943, 'news_weight': 0.5694114399233269, 'report_weight': 0.2260870441170738}`
- best_metrics: `{'annual_return': 0.005520411924236912, 'max_drawdown': 0.38889424717236565, 'positive_window_ratio': 0.9386548410938654, 'trades': 1722, 'factor_alignment': 0.010654602761135485, 'score': -0.2892648968977408}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260214_000300/ultra_short/trials.csv`

## 说明
- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。
- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。
- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。
