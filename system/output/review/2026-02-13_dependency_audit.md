# 依赖分层审计 | 2026-02-13

- 结果: `PASS`
- 检查文件数: `47`
- 违规数: `0`

## 层级依赖图
- `backtest` -> `backtest, models, regime, signal`
- `cli` -> `config, engine, models`
- `config` -> `config`
- `data` -> `data, models`
- `engine` -> `backtest, config, data, models, orchestration, regime, reporting, research, review, risk, signal`
- `orchestration` -> `config, data, models, orchestration`
- `regime` -> `models, regime`
- `reporting` -> `data, models, reporting`
- `research` -> `backtest, data, research`
- `review` -> `models, review`
- `risk` -> `models, risk`
- `signal` -> `models, regime, signal`

