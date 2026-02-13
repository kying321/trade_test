# 实盘多模式回测与优化报告 | 2026-02-13

## 运行概览
- 时间预算: `0.03` 小时
- 实际耗时: `0.05` 小时
- 区间: `2019-01-01 ~ 2026-02-13`
- 覆盖标的数: `20`
- 行情记录数: `4081`
- 新闻记录数: `120`
- 研报记录数: `1550`

## 模式结果
### ultra_short
- trials: `3`
- best_score: `0.106542`
- best_params: `{'hold_days': 3, 'max_daily_trades': 7, 'signal_confidence_min': 53.60473984981963, 'convexity_min': 3.0974791137218673, 'news_weight': 0.44051757049084117, 'report_weight': 0.28139966172944864}`
- best_metrics: `{'annual_return': 0.03890905633085562, 'max_drawdown': 0.21464410052969218, 'positive_window_ratio': 0.9861030689056167, 'trades': 58, 'factor_alignment': 0.024425482846571693, 'score': 0.10654158662204571}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_224905/ultra_short/trials.csv`

### swing
- trials: `3`
- best_score: `0.806648`
- best_params: `{'hold_days': 12, 'max_daily_trades': 2, 'signal_confidence_min': 78.0, 'convexity_min': 3.1534285513391094, 'news_weight': 0.6812513307246433, 'report_weight': 0.2324600588926867}`
- best_metrics: `{'annual_return': 0.03064452720928279, 'max_drawdown': 0.020761053492556858, 'positive_window_ratio': 0.9994209612044007, 'trades': 2, 'factor_alignment': 0.025691035608774622, 'score': 0.8066476160106901}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_224905/swing/trials.csv`

### long
- trials: `3`
- best_score: `0.531242`
- best_params: `{'hold_days': 15, 'max_daily_trades': 2, 'signal_confidence_min': 64.31955411810094, 'convexity_min': 2.569141757252601, 'news_weight': 0.5206659851092754, 'report_weight': 0.287780130225491}`
- best_metrics: `{'annual_return': 0.020695422287392296, 'max_drawdown': 0.16082241317083634, 'positive_window_ratio': 0.9924724956572091, 'trades': 17, 'factor_alignment': 0.006365588175561115, 'score': 0.531241836015801}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_224905/long/trials.csv`

## 说明
- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。
- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。
- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。
