# 实盘多模式回测与优化报告 | 2026-02-13

## 运行概览
- 时间预算: `0.01` 小时
- 实际耗时: `0.02` 小时
- 区间: `2022-01-01 ~ 2026-02-13`
- 覆盖标的数: `10`
- 行情记录数: `9546`
- 新闻记录数: `70`
- 研报记录数: `538`

## 模式结果
### ultra_short
- trials: `1`
- best_score: `0.570808`
- best_params: `{'hold_days': 1, 'max_daily_trades': 3, 'signal_confidence_min': 52.474725185404026, 'convexity_min': 2.402996715246715, 'news_weight': 0.028689008371944547, 'report_weight': 0.14792608457745593}`
- best_metrics: `{'annual_return': -0.0059004724370140815, 'max_drawdown': 0.12155984235197947, 'positive_window_ratio': 0.9839679358717435, 'trades': 65, 'factor_alignment': 0.03174019386148559, 'score': 0.5708081236978605}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_225524/ultra_short/trials.csv`

### swing
- trials: `1`
- best_score: `-0.164617`
- best_params: `{'hold_days': 7, 'max_daily_trades': 5, 'signal_confidence_min': 47.323879013088494, 'convexity_min': 2.1114574785583153, 'news_weight': 0.9483284532917751, 'report_weight': 0.6218835927963828}`
- best_metrics: `{'annual_return': 0.138350340969279, 'max_drawdown': 0.4028894271376794, 'positive_window_ratio': 0.93687374749499, 'trades': 212, 'factor_alignment': -0.016452090837447905, 'score': -0.16461718893993044}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_225524/swing/trials.csv`

### long
- trials: `1`
- best_score: `0.200000`
- best_params: `{'hold_days': 32, 'max_daily_trades': 1, 'signal_confidence_min': 71.87587071950767, 'convexity_min': 4.055960267047038, 'news_weight': 0.2753088157611293, 'report_weight': 0.13796807286695534}`
- best_metrics: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'factor_alignment': 0.0, 'score': 0.20000000000000007}`
- trial_log: `/Users/jokenrobot/Downloads/离厄—技术分析原理/system/output/research/20260213_225524/long/trials.csv`

## 说明
- 目标函数 = 年化收益 + 正窗口占比 + 新闻/研报因子对齐 - 回撤惩罚 - 违规惩罚。
- 超短线/中短线/长线通过 `hold_days`、`max_daily_trades`、置信阈值、凸性阈值的不同约束实现。
- 新闻与研报因子按日期聚合后与策略收益序列做对齐相关，作为优化反馈项。
