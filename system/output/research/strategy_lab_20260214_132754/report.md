# 策略学习实验室报告 | 2026-02-14

## 概览
- 区间: `2026-02-10 ~ 2026-02-14`
- 训练截止: `2026-02-11`
- 验证起点: `2026-02-12`
- 覆盖标的: `1`
- 行情记录: `4`
- 新闻记录: `6`
- 研报记录: `0`

## 市场学习信号
- `trend_strength_z`: `0.0000`
- `volatility_z`: `0.0000`
- `tail_risk_z`: `-1.1935`
- `mean_return`: `0.0006`
- `volatility`: `0.0209`

## 报告学习信号
- `news_bias_z`: `0.0000`
- `report_bias_z`: `0.0000`
- `news_report_agreement`: `0.0000`
- `news_bias`: `0.0000`
- `report_bias`: `0.0000`

## 候选策略评分
### trend_convex_01
- rationale: 趋势驱动+凸性约束
- params: `{'signal_confidence_min': 50.0, 'convexity_min': 2.4386913871128724, 'hold_days': 11, 'max_daily_trades': 2}`
- train: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0}`
- validation: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- align(train/valid): `0.0000/0.0000`
- score: `0.250000`

### report_momentum_02
- rationale: 研报偏向驱动的动量延续
- params: `{'signal_confidence_min': 48.0, 'convexity_min': 2.238691387112872, 'hold_days': 14, 'max_daily_trades': 2}`
- train: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0}`
- validation: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- align(train/valid): `0.0000/0.0000`
- score: `0.250000`

### news_reversion_03
- rationale: 新闻冲击后的均值回归
- params: `{'signal_confidence_min': 45.0, 'convexity_min': 2.0386913871128725, 'hold_days': 7, 'max_daily_trades': 3}`
- train: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0}`
- validation: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- align(train/valid): `0.0000/0.0000`
- score: `0.250000`

## 结论
- 最优策略: `trend_convex_01`
- 最优参数: `{'signal_confidence_min': 50.0, 'convexity_min': 2.4386913871128724, 'hold_days': 11, 'max_daily_trades': 2}`
