# 实盘多模式回测与优化报告 | 2026-02-13

## 运行概览
- 时间预算: `0.06` 小时
- 实际耗时: `0.06` 小时
- 区间: `2015-01-01 ~ 2026-02-13`
- 覆盖标的数: `40`
- 行情记录数: `92560`
- 新闻记录数: `250`
- 研报记录数: `3905`

## 模式结果
### long
- trials: `2`
- best_score: `-0.608667`
- best_params: `{'hold_days': 13, 'max_daily_trades': 2, 'signal_confidence_min': 35.49375760280728, 'convexity_min': 1.7145017897238457, 'news_weight': 0.5694114399233269, 'report_weight': 0.2260870441170738}`
- best_metrics: `{'annual_return': 0.15097827804470776, 'max_drawdown': 0.6302238693106693, 'positive_window_ratio': 0.8699186991869918, 'trades': 923, 'factor_alignment': -0.004575161329058393, 'score': -0.6086666201546795}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_235218/long/trials.csv`

## 说明
- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。
- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。
- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。
