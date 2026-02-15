# 策略学习实验室报告 | 2026-02-14

## 概览
- 区间: `2026-01-15 ~ 2026-02-11`
- 训练截止: `2026-01-28`
- 验证起点: `2026-01-29`
- 复盘检验窗口: `2026-02-12 ~ 2026-02-13`
- 覆盖标的: `1`
- 行情记录: `20`
- 复盘行情记录: `2`
- 新闻记录: `6`
- 研报记录: `1`
- 复盘新闻记录: `2`
- 复盘研报记录: `0`

## 市场学习信号
- `trend_strength_z`: `0.0000`
- `volatility_z`: `0.0000`
- `tail_risk_z`: `-0.9637`
- `mean_return`: `0.0022`
- `volatility`: `0.0145`

## 报告学习信号
- `news_bias_z`: `0.0000`
- `report_bias_z`: `0.5000`
- `news_report_agreement`: `0.0000`
- `news_bias`: `0.0000`
- `report_bias`: `0.2153`

## 候选策略评分
### trend_convex_01
- rationale: 趋势驱动+凸性约束
- params: `{'signal_confidence_min': 48.543436937742044, 'convexity_min': 2.392740144520729, 'hold_days': 11, 'max_daily_trades': 2}`
- train: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0}`
- validation: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- review: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- align(train/valid/review): `0.0000/0.0000/0.0000`
- robustness: `0.8225` | accepted: `True`
- score: `0.554500`

### report_momentum_02
- rationale: 研报偏向驱动的动量延续
- params: `{'signal_confidence_min': 46.543436937742044, 'convexity_min': 2.192740144520729, 'hold_days': 14, 'max_daily_trades': 2}`
- train: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0}`
- validation: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- review: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- align(train/valid/review): `0.0000/0.0000/0.0000`
- robustness: `0.8225` | accepted: `True`
- score: `0.554500`

### news_reversion_03
- rationale: 新闻冲击后的均值回归
- params: `{'signal_confidence_min': 43.543436937742044, 'convexity_min': 1.9927401445207291, 'hold_days': 7, 'max_daily_trades': 3}`
- train: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0}`
- validation: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- review: `{'annual_return': 0.0, 'max_drawdown': 0.0, 'positive_window_ratio': 1.0, 'trades': 0, 'violations': 0}`
- align(train/valid/review): `0.0000/0.0000/0.0000`
- robustness: `0.8225` | accepted: `True`
- score: `0.554500`

## 结论
- 最优策略: `trend_convex_01`
- 最优参数: `{'signal_confidence_min': 48.543436937742044, 'convexity_min': 2.392740144520729, 'hold_days': 11, 'max_daily_trades': 2}`
