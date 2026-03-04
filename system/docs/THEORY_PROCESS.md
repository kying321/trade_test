# Theory Integration Process (Selective Adoption)

## 1) Goal
- Only adopt theory pieces that measurably improve risk control or profitability.
- Reject narrative-only ideas that cannot pass backtest and stability gates.

## 2) Selective Rules
- Priority A: reduce drawdown/tail risk while keeping return non-negative.
- Priority B: improve return consistency (positive window ratio / robustness) without increasing drawdown.
- Reject: no out-of-sample lift, weak trade activity, or unstable cross-window behavior.

## 3) Mandatory Steps

### Step A: Theory Ablation
```bash
PYTHONPATH=src python3 scripts/run_theory_ablation.py \
  --start 2024-01-01 --end 2026-03-03 \
  --max-symbols 8 --report-symbol-cap 6 --workers 2 \
  --signal-confidence-min 5 --convexity-min 0.3 \
  --max-daily-trades 5 --hold-days 5 --proxy-lookback 180
```

### Step B: Cross-window Stability
```bash
PYTHONPATH=src python3 scripts/run_strategy_stability.py \
  --end 2026-03-03 \
  --window-days 365,240,180,120 \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT \
  --candidate-count 8 --max-drawdown-target 0.05 \
  --review-max-drawdown-target 0.07 --drawdown-soft-band 0.03
```

### Step C: Strategy Lab Candidate
```bash
PYTHONPATH=src python3 -m lie_engine.cli strategy-lab \
  --start 2025-01-01 --end 2026-03-03 \
  --candidate-count 10 --review-days 3
```

### Step D: Review Merge (Small Step)
```bash
PYTHONPATH=src python3 -m lie_engine.cli review --date 2026-03-03
```

## 4) Merge Gate (Hard)
- `candidate.accepted == true`
- `validation_metrics.annual_return >= 0`
- `validation_metrics.max_drawdown <= validation.max_drawdown_max` (default 0.18)
- `validation_metrics.trades >= 2`
- `validation_metrics.positive_window_ratio >= 0.55`
- `robustness_score >= 0.30` (if provided)

Candidates failing any gate are skipped with explicit note:
- `strategy_lab_candidate_skipped=...`

