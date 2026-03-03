# Theory Practice Log (2026-03-02)

## Scope
- objective: continue deep search and practical integration of ICT / Al Brooks / LiE-PDF into executable signals
- mode: deterministic OHLCV proxies first, then microstructure-native upgrade path

## External Source Snapshot
- Binance Spot websocket stream schema (aggTrade, trade, kline, depth):
  - https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md
- Binance derivatives depth stream semantics (`U/u/pu` continuity):
  - https://binance-docs.github.io/apidocs/delivery_testnet/cn/
- Order-flow imbalance and queue imbalance evidence:
  - https://arxiv.org/abs/2202.04843
  - https://link.springer.com/article/10.1007/s00521-026-11210-w
- Model uncertainty zones / regime sensitivity:
  - https://academic.oup.com/imaman/article/34/2/355/6547686
- Al Brooks primary publication entry:
  - https://www.wiley.com/en-us/Reading+Price+Charts+Bar+by+Bar%3A+The+Technical+Analysis+of+Price+Action+for+the+Serious+Trader-p-9780470443958

## Theory -> Factor Mapping (Implemented)
- new module: `src/lie_engine/signal/theory.py`
- added runtime fields in `SignalEngineConfig`:
  - `theory_enabled`
  - `theory_ict_weight / theory_brooks_weight / theory_lie_weight`
  - `theory_confidence_boost_max / theory_penalty_max`
  - `theory_min_confluence / theory_conflict_fuse`
- computed outputs:
  - `confluence` (aligned multi-theory evidence)
  - `conflict` (opposing multi-theory evidence)
  - theory flags (`theory_conflict_high`, `theory_resonance`, etc.)
- applied in `generate_signal_for_symbol`:
  - confidence = raw - factor_penalty - theory_penalty - micro_penalty + theory_boost + micro_boost
  - signal notes include auditable theory telemetry

## Practical Backtest Wiring
- `BacktestConfig` now carries theory weights and enable switch.
- `run_event_backtest(...)` forwards theory params into signal engine.
- `strategy_lab` candidate generator now searches:
  - risk params: confidence/convexity/hold/trade-count
  - theory params: `theory_ict_weight/theory_brooks_weight/theory_lie_weight`

## Local Theory Corpus Usage
- local revised corpus under `深度修正版/` was used as deterministic mapping reference:
  - `01_ICT原理_修正版.md`
  - `03_价格行为原理_修正版.md`
  - `17_反脆弱交易系统.md`
- implementation keeps those narrative concepts in bar-based proxy form to avoid lookahead leakage.

## Verification Hooks
- unit updates:
  - `tests/test_signal.py`
    - theory confluence can lift confidence under aligned setup
    - theory conflict detection on sweep-reclaim-failure setup
- recommended runtime checks:
  - `PYTHONPATH=system/src python3 -m pytest system/tests/test_signal.py -q`
  - `PYTHONPATH=system/src python3 -m lie_engine.cli strategy-lab --start <YYYY-MM-DD> --end <YYYY-MM-DD> --candidate-count 12`

## 2026-03-04 Ablation Run
- script: `system/scripts/run_theory_ablation.py`
- command:
  - `PYTHONPATH=system/src python3 system/scripts/run_theory_ablation.py --start 2024-01-01 --end 2026-02-13 --max-symbols 8 --report-symbol-cap 5 --workers 2 --signal-confidence-min 5 --convexity-min 0.3 --max-daily-trades 5 --hold-days 5 --proxy-lookback 180`
- run output:
  - summary: `system/output/research/theory_ablation_20260304_013627/summary.json`
  - report: `system/output/research/theory_ablation_20260304_013627/report.md`
- data source:
  - live pull failed in restricted environment, fallback used:
  - `output/artifacts/normalized/2026-02-28_bars_normalized.csv`
- key metrics (theory_on_01 vs baseline_off):
  - annual_return: `0.4053` vs `0.4023` (`+0.0030`)
  - max_drawdown: `0.1193` vs `0.1193` (`+0.0000`)
  - positive_window_ratio: both `0.9100`
  - trades: `288` vs `293` (`-5`)
  - objective: `0.4799` vs `0.4769` (`+0.0030`)
