# 实盘多模式回测与优化报告 | 2026-02-13

## 运行概览
- 时间预算: `0.12` 小时
- 实际耗时: `0.14` 小时
- 区间: `2015-01-01 ~ 2026-02-13`
- 覆盖标的数: `40`
- 行情记录数: `92560`
- 新闻记录数: `250`
- 研报记录数: `3905`

## 模式结果
### ultra_short
- trials: `2`
- best_score: `-0.268711`
- best_params: `{'hold_days': 1, 'max_daily_trades': 7, 'signal_confidence_min': 49.095775021947496, 'convexity_min': 2.910475376821049, 'news_weight': 0.5694114399233269, 'report_weight': 0.2260870441170738}`
- best_metrics: `{'annual_return': -0.015069638986422329, 'max_drawdown': 0.3730346923731592, 'positive_window_ratio': 0.9541759053954176, 'trades': 1152, 'factor_alignment': 0.024980495177161294, 'score': -0.26871114018591086}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_233402/ultra_short/trials.csv`

### swing
- trials: `1`
- best_score: `-1.200000`
- best_params: `{'hold_days': 5, 'max_daily_trades': 5, 'signal_confidence_min': 66.24754896266192, 'convexity_min': 3.7746278718499924, 'news_weight': 0.44341419882733113, 'report_weight': 0.2272387217847769}`
- best_metrics: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'factor_alignment': 0.0, 'score': -1.2}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_233402/swing/trials.csv`

## 说明
- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。
- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。
- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。
