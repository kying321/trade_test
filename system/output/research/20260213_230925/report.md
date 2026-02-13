# 实盘多模式回测与优化报告 | 2026-02-13

## 运行概览
- 时间预算: `0.00` 小时
- 实际耗时: `0.03` 小时
- 区间: `2015-01-01 ~ 2026-02-13`
- 覆盖标的数: `10`
- 行情记录数: `16072`
- 新闻记录数: `70`
- 研报记录数: `1204`

## 模式结果
### ultra_short
- trials: `1`
- best_score: `-0.392534`
- best_params: `{'hold_days': 1, 'max_daily_trades': 7, 'signal_confidence_min': 50.36074539132183, 'convexity_min': 2.9171958398227646, 'news_weight': 0.6973680290593639, 'report_weight': 0.09417734788764953}`
- best_metrics: `{'annual_return': -0.04247005440364415, 'max_drawdown': 0.4246801848632986, 'positive_window_ratio': 0.9693160813308688, 'trades': 348, 'factor_alignment': -0.008662076028917809, 'score': -0.39253415578229084}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_230925/ultra_short/trials.csv`

### swing
- trials: `1`
- best_score: `-0.450000`
- best_params: `{'hold_days': 8, 'max_daily_trades': 5, 'signal_confidence_min': 70.11761016568165, 'convexity_min': 3.6865543326646897, 'news_weight': 0.12811363267554587, 'report_weight': 0.45038593789556713}`
- best_metrics: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'factor_alignment': 0.0, 'score': -0.44999999999999996}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_230925/swing/trials.csv`

### long
- trials: `1`
- best_score: `-0.450000`
- best_params: `{'hold_days': 24, 'max_daily_trades': 1, 'signal_confidence_min': 85.58324463200385, 'convexity_min': 4.002822336225861, 'news_weight': 0.82276161327083, 'report_weight': 0.44341419882733113}`
- best_metrics: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'factor_alignment': 0.0, 'score': -0.44999999999999996}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_230925/long/trials.csv`

## 说明
- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。
- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。
- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。
