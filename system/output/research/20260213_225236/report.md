# 实盘多模式回测与优化报告 | 2026-02-13

## 运行概览
- 时间预算: `0.01` 小时
- 实际耗时: `0.04` 小时
- 区间: `2019-01-01 ~ 2026-02-13`
- 覆盖标的数: `20`
- 行情记录数: `33423`
- 新闻记录数: `120`
- 研报记录数: `1550`

## 模式结果
### ultra_short
- trials: `1`
- best_score: `0.629364`
- best_params: `{'hold_days': 3, 'max_daily_trades': 6, 'signal_confidence_min': 66.40248303393514, 'convexity_min': 2.751371380490387, 'news_weight': 0.22520718999059186, 'report_weight': 0.30016628491122543}`
- best_metrics: `{'annual_return': 0.028813270820369663, 'max_drawdown': 0.11062114858755367, 'positive_window_ratio': 0.9930555555555556, 'trades': 99, 'factor_alignment': -0.0017943199157969864, 'score': 0.6293640059965524}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_225236/ultra_short/trials.csv`

### swing
- trials: `1`
- best_score: `0.800000`
- best_params: `{'hold_days': 6, 'max_daily_trades': 5, 'signal_confidence_min': 45.173755050663964, 'convexity_min': 3.7709482041186395, 'news_weight': 0.7970694287520462, 'report_weight': 0.4679349528437208}`
- best_metrics: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'factor_alignment': 0.0, 'score': 0.8}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_225236/swing/trials.csv`

### long
- trials: `1`
- best_score: `-0.209788`
- best_params: `{'hold_days': 31, 'max_daily_trades': 1, 'signal_confidence_min': 64.18804519932552, 'convexity_min': 2.913634845431549, 'news_weight': 0.4450763058826466, 'report_weight': 0.5045482589579533}`
- best_metrics: `{'annual_return': -0.041207995207039994, 'max_drawdown': 0.3314842022727872, 'positive_window_ratio': 0.9837962962962963, 'trades': 35, 'factor_alignment': -0.002345167468880875, 'score': -0.20978767154309208}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_225236/long/trials.csv`

## 说明
- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。
- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。
- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。
