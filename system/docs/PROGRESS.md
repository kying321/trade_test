# Progress Handoff

## Current State (2026-02-14)
- Architecture loop is active: data -> regime -> signal -> risk -> backtest -> review -> gate.
- Mode feedback artifact is online: `output/daily/YYYY-MM-DD_mode_feedback.json`.
- Mode health gate is online: `gate_report.checks.mode_health_ok`.
- Execution risk throttle is online:
  - `run_eod` applies `risk_multiplier` into position sizing.
  - `run_premarket` / `run_intraday_check` now output `runtime_mode + mode_health + risk_control + risk_multiplier`.
  - `run_premarket` / `run_intraday_check` now emit manifests for slot-level traceability.
- State stability monitoring is online (`ops_report`):
  - mode switch rate
  - risk-multiplier floor + drift
  - source-confidence floor
  - mode-health fail-days
  - threshold breaches are surfaced in `state_stability.alerts`.
- Review-loop defect planning is now state-aware:
  - `STATE_*` defect codes are emitted when `state_stability` thresholds breach.
  - `next_actions` is reordered to prioritize state stabilization first.
- Mode-level adaptive review update is online:
  - `run_review` now applies bounded `mode_adaptive_update` on
    `signal_confidence_min / convexity_min / hold_days / max_daily_trades`
    based on `mode_history` performance bands.
  - Audit includes `mode_adaptive` payload in `param_delta.yaml`.
- Mode drift monitor is online:
  - `gate_report.checks.mode_drift_ok` compares live performance vs backtest baseline.
  - `ops_report.mode_drift` exposes drift status/alerts (win-rate & profit-factor gaps).
  - review-loop defect plan emits `MODE_DRIFT_*` codes and prioritizes drift fixes.
- Slot anomaly monitor is online:
  - `gate_report.checks.slot_anomaly_ok` evaluates premarket/intraday/eod anomalies.
  - `ops_report.slot_anomaly` exposes missing/anomaly ratios and threshold checks.
  - review-loop defect plan emits `SLOT_*` codes and prioritizes slot-chain repairs.
- Reconciliation drift monitor is online:
  - `gate_report.checks.reconcile_drift_ok` validates eod manifest vs daily csv/sqlite/open-state consistency.
  - `ops_report.reconcile_drift` exposes missing ratio and plan/close/open drift breaches.
  - review-loop defect plan emits `RECONCILE_*` codes and prioritizes reconciliation repair.
- Automatic rollback recommendation is online:
  - `gate_report.rollback_recommendation` outputs `level/score/reason_codes/target_anchor`.
  - `ops_report.rollback_recommendation` is included in status scoring and markdown output.
  - defect plan emits `ROLLBACK_*` codes and puts rollback actions at top priority when needed.
- Paper execution loop is online:
  - `run_eod` now settles previous paper positions using day high/low/close with
    stop-loss / take-profit / time-stop priority.
  - Closed trades are appended to sqlite `executed_plans` (pnl, exit_reason, mode fields).
  - Open paper positions are persisted in `output/artifacts/paper_positions_open.json`.
- Exposure snapshot bug fixed:
  - risk budgeting now reads only latest-date `latest_positions` rows, avoiding
    historical ACTIVE rows cumulative pollution.
- Review audit now records:
  - `runtime_mode`
  - `mode_history`
  - `mode_health`

## Testing Workflow
- Full suite:
  - `lie test-all`
- Fast deterministic subset (for quick agent iteration):
  - `lie test-all --fast --fast-ratio 0.10`
- Review loop default:
  - Round-1 runs fast subset first, then auto-runs full suite if fast passes.
- Parallel shard usage (multi-agent):
  - Agent A: `--fast --fast-ratio 1.0 --fast-shard-index 0 --fast-shard-total 4`
  - Agent B: `--fast --fast-ratio 1.0 --fast-shard-index 1 --fast-shard-total 4`
  - Agent C: `--fast --fast-ratio 1.0 --fast-shard-index 2 --fast-shard-total 4`
  - Agent D: `--fast --fast-ratio 1.0 --fast-shard-index 3 --fast-shard-total 4`

## Machine-Friendly Logging Rules
- `lie test-all` returns compact payload with one-line `summary_line` containing `error=...`.
- Full stdout/stderr are persisted to:
  - `output/logs/tests_YYYYMMDD_HHMMSS.json`
- Downstream gate/review reads `failed_tests` first, then falls back to stderr parsing.

## Next Priorities
1. Add broker snapshot adapter (paper/live) so reconcile monitor can compare against external broker state, not only local artifacts.
2. Trace and eliminate intermittent sqlite `ResourceWarning` during unittest runs.
3. Fix `lie test-all` full-run blocking path in CLI orchestration.
