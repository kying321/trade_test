# Progress Handoff

## Current State (2026-02-15)
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
- Style-drift adaptive guard + release gate linkage is online:
  - `run_review` now applies severity-scaled收敛（`signal_confidence_min / max_daily_trades / hold_days`）并产出 `style_drift_guard` 审计。
  - `gate_report.checks.style_drift_ok` supports monitor-only / hard-fail modes (`style_drift_gate_hard_fail`).
  - `ops_report.style_drift` + defect plan now expose `STYLE_DRIFT_*` codes and gate failures.
- Slot anomaly monitor is online:
  - `gate_report.checks.slot_anomaly_ok` evaluates premarket/intraday/eod anomalies.
  - `ops_report.slot_anomaly` exposes missing/anomaly ratios and threshold checks.
  - review-loop defect plan emits `SLOT_*` codes and prioritizes slot-chain repairs.
- Reconciliation drift monitor is online:
  - `gate_report.checks.reconcile_drift_ok` validates eod manifest vs daily csv/sqlite/open-state consistency.
  - `ops_report.reconcile_drift` exposes missing ratio and plan/close/open drift breaches.
  - review-loop defect plan emits `RECONCILE_*` codes and prioritizes reconciliation repair.
- Broker snapshot contract linter is online in reconciliation:
  - schema lint (`source/open_positions/closed_count/closed_pnl/positions`) + numeric hard range + symbol normalization.
  - new checks: `broker_contract_schema_ok / broker_contract_numeric_ok / broker_contract_symbol_ok`.
  - new defect codes: `RECONCILE_BROKER_CONTRACT_SCHEMA / NUMERIC / SYMBOL`.
- Broker canonical snapshot view is online:
  - when broker contract lint passes, system writes canonical snapshot to
    `output/artifacts/broker_snapshot_canonical/YYYY-MM-DD.json`.
  - canonical view includes normalized `symbol/side/qty/notional` plus `raw_symbol/raw_side`.
  - new check/alert/defect:
    `broker_contract_canonical_view_ok` /
    `reconcile_broker_contract_canonical_view_failed` /
    `RECONCILE_BROKER_CANONICAL_VIEW`.
- Canonical row-level reconciliation is online:
  - broker-vs-system行级比较优先使用 canonical 视图（`canonical_file` / `canonical_inline`），无可用 canonical 时回退 `canonical_fallback`。
  - checks/alerts/defect:
    `broker_row_diff_ok` /
    `reconcile_broker_row_diff_high` /
    `RECONCILE_BROKER_ROW_DIFF`.
  - metrics include:
    `broker_row_diff_*`（samples/breach/key_mismatch/count_gap/notional_gap/canonical_preferred）。
- Reconcile row-diff drill-down artifact is online:
  - when row-diff has breaches, system exports top mismatched keys + source-map hints to
    `output/review/YYYY-MM-DD_reconcile_row_diff.json` and `.md`.
  - gate/ops/review now carry artifact metadata (`written/path/reason/sample_rows/breach_rows`)
    for faster hotfix routing.
- Reconcile row-diff alias dictionary + CI lint is online:
  - configurable alias maps:
    `validation.ops_reconcile_broker_row_diff_symbol_alias_map` /
    `validation.ops_reconcile_broker_row_diff_side_alias_map`.
  - row-level diff now applies alias normalization before key matching, reducing false mismatch due to venue symbol/side enums.
  - config validation now lints alias maps in tests/CI (empty or non-canonical entries rejected).
- Reconcile row-diff alias drift monitor is online:
  - metrics now expose `broker_row_diff_alias_hit_rate` and `broker_row_diff_unresolved_key_ratio`.
  - configurable guardrails:
    `validation.ops_reconcile_broker_row_diff_alias_monitor_enabled` /
    `validation.ops_reconcile_broker_row_diff_alias_hit_rate_min` /
    `validation.ops_reconcile_broker_row_diff_unresolved_key_ratio_max`.
  - new alert/check/defect path:
    `reconcile_broker_row_diff_alias_drift` /
    `broker_row_diff_alias_drift_ok` /
    `RECONCILE_BROKER_ROW_DIFF_ALIAS_DRIFT`.
- Mode-aware stress matrix report is online:
  - CLI: `lie stress-matrix --date YYYY-MM-DD --modes ultra_short,swing,long`.
  - output: `output/review/YYYY-MM-DD_mode_stress_matrix.json` + `.md`.
  - evaluation grid: `2015_crash / 2020_pandemic / 2022_geopolitical / extreme_gap` windows.
- Stress-matrix trend monitor is online in gate/ops/review-loop:
  - compares latest stress-matrix run vs rolling baseline runs.
  - checks: `stress_matrix_trend_ok` with robustness/annual/drawdown/fail-ratio gates.
  - defect codes: `STRESS_MATRIX_ROBUSTNESS / ANNUAL_RETURN / DRAWDOWN / FAIL_RATIO`.
- Review-loop stress-matrix auto-trigger is online:
  - when `mode_drift` or `stress_matrix_trend` breaches, `review_until_pass` can auto-run stress matrix once per cycle (configurable).
  - round audit now includes `stress_matrix_autorun` payload with trigger reason, run status, and output artifact paths.
- Review-loop stress-matrix cooldown/backoff is online:
  - auto-trigger now supports per-round cooldown and exponential backoff,
    preventing repeated stress reruns in noisy multi-round loops.
  - config keys:
    `review_loop_stress_matrix_autorun_cooldown_rounds` /
    `review_loop_stress_matrix_autorun_backoff_multiplier` /
    `review_loop_stress_matrix_autorun_backoff_max_rounds`.
  - round audit now records cooldown state (`next_allowed_round`, `cooldown_remaining_rounds`, `runs_used`).
- Stress-matrix historical trigger analytics is online:
  - gate/ops now expose `stress_autorun_history` over rolling review-loop rounds.
  - metrics include trigger density, skip reason distribution, and cooldown efficiency.
  - compliance artifact:
    `output/review/YYYY-MM-DD_stress_autorun_history.json` + `.md`.
- Stress-matrix auto-trigger adaptive guardrail is online:
  - review loop now computes dynamic `max_runs` from recent trigger density (history + current rounds).
  - high trigger-density windows throttle stress reruns; low-density windows expand bounded rerun budget.
  - round audit includes `max_runs_base / max_runs / adaptive.reason / adaptive.factor / adaptive.trigger_density`.
- Stress-matrix auto-trigger adaptive saturation monitor is online:
  - gate/ops now evaluate rolling `effective_max_runs / base_max_runs` trend and throttle/expand occupancy.
  - checks: `stress_autorun_adaptive_ok` with floor/ceiling + throttle-ratio + expand-ratio gates.
  - review-loop defect plan emits:
    `STRESS_AUTORUN_ADAPTIVE_RATIO_LOW / RATIO_HIGH / THROTTLE / EXPAND`.
- Stress-matrix auto-trigger adaptive reason drift monitor is online:
  - gate/ops now evaluate `high_density_throttle / low_density_expand` reason-mix drift and change-point gap.
  - checks: `stress_autorun_reason_drift_ok` with `reason_mix_gap` + `change_point_gap` gates.
  - review-loop defect plan emits:
    `STRESS_AUTORUN_REASON_MIX / STRESS_AUTORUN_REASON_CHANGE_POINT`.
- Stress-matrix auto-trigger reason-drift artifact export is online:
  - compliance artifact:
    `output/review/YYYY-MM-DD_stress_autorun_reason_drift.json` + `.md`.
  - artifact includes top reason transitions and rolling per-window drift trace (`mix_gap/change_point_gap`).
  - gate/ops payload now carries artifact metadata (`written/path/reason/transition_count/window_trace_points`).
- Temporal audit monitor is online in gate/ops/review-loop:
  - validates recent `strategy_lab` / `research_backtest` manifests for
    `cutoff_date/cutoff_ts/bar_max_ts/news_max_ts/report_max_ts` completeness and leakage.
  - checks: `temporal_audit_ok` with `missing/leak/strict_cutoff` gates.
  - defect codes: `TEMPORAL_AUDIT / TEMPORAL_AUDIT_MISSING / TEMPORAL_AUDIT_LEAK / TEMPORAL_AUDIT_STRICT`.
- Temporal-audit auto-fix assistant is online:
  - for manifests with missing temporal metadata, system can backfill from `artifacts.summary` and `data_fetch_stats`.
  - write-back is guarded by safe temporal validation (`*_max_ts <= cutoff_date`) to avoid front-running leakage.
  - audit payload now includes autofix telemetry (`attempted/applied/failed/skipped`, per-manifest reason).
- Temporal-audit autofix patch artifact is online:
  - each temporal autofix attempt is exported as compliance trace artifact:
    `output/review/YYYY-MM-DD_temporal_autofix_patch.json` + `.md`.
  - artifact includes manifest path, summary source, field-level delta (`before/after/source`) and strict-cutoff patch trace.
  - gate/ops/review payload now includes `temporal_audit.artifacts.autofix_patch`.
- Temporal-audit patch retention/rotation policy is online:
  - `validation.ops_temporal_audit_autofix_patch_retention_days` controls N-day keep window.
  - stale `YYYY-MM-DD_temporal_autofix_patch.{json,md}` artifacts beyond retention are auto-pruned.
  - checksum index is generated at `output/review/temporal_autofix_patch_checksum_index.json`
    with per-file `sha256` + byte-size for compliance traceability.
- Strict cutoff temporal-audit chain is online:
  - `real_data` now emits `cutoff_ts / bar_max_ts / news_max_ts / report_max_ts` (+ review variants).
  - `strategy_lab` / `research_backtest` summaries and manifests now carry these fields.
  - review candidate loader rejects strategy-lab manifests that violate cutoff temporal bounds.
- Slot EOD anomaly is split:
  - quality-path: `eod_quality_anomaly_ok`
  - risk-path: `eod_risk_anomaly_ok`
  - keeps compatibility with aggregate `eod_anomaly_ok`.
- Slot EOD regime-bucket thresholding is online:
  - quality/risk anomaly thresholds now support `trend / range / extreme_vol` bucket maps.
  - new checks: `eod_quality_regime_bucket_ok` / `eod_risk_regime_bucket_ok`.
  - ops report now exposes per-bucket quality/risk anomaly ratios for threshold tuning.
- Automated regime-bucket threshold tuner is online:
  - `run_review` now updates `output/artifacts/slot_regime_thresholds_live.yaml` from rolling EOD manifests.
  - tuning is bounded by `step/buffer/floor/ceiling` and only applies on buckets with enough samples.
  - `gate/ops` can consume live threshold maps when `validation.ops_slot_use_live_regime_thresholds=true`.
  - missing-aware hard guard is online: when slot missing ratio breaches
    `validation.ops_slot_regime_tune_missing_ratio_hard_cap`, tuning is skipped with reason `slot_missing_ratio_high`.
- Broker snapshot adapter is online in reconciliation:
  - optional path: `output/artifacts/broker_snapshot/YYYY-MM-DD.json`
  - checks: `broker_missing_ratio_ok / broker_count_gap_ok / broker_pnl_gap_ok`
  - can be hardened by `validation.ops_reconcile_require_broker_snapshot=true`.
- Broker snapshot producer is online in EOD:
  - `run_eod` now writes `output/artifacts/broker_snapshot/YYYY-MM-DD.json` on every run.
  - payload includes `source=paper_engine`, `open_positions`, `closed_count`, `closed_pnl`, and `positions[]`.
  - eod manifest artifact now includes `broker_snapshot`.
- Live broker snapshot adapter writer is online:
  - source mode switch: `validation.broker_snapshot_source_mode`.
  - `live_adapter` reads `validation.broker_snapshot_live_inbox/YYYY-MM-DD.json` and normalizes to broker snapshot contract.
  - fallback behavior is controlled by `validation.broker_snapshot_live_fallback_to_paper`.
- Live broker snapshot mapping templates are online:
  - mapping profile: `validation.broker_snapshot_live_mapping_profile` (`generic/ibkr/binance/ctp`).
  - optional field override: `validation.broker_snapshot_live_mapping_fields`.
  - position side is canonicalized to `LONG/SHORT/FLAT` (兼容 `BUY/SELL/BOTH/2/3` 枚举).
- Test orchestration timeout guard is online:
  - `lie test-all` now enforces `validation.test_all_timeout_seconds`.
  - timeout returns `returncode=124`, emits `error=test_timeout`, and marks `failed_tests` with `__timeout__`.
- Review-loop timeout fallback is online:
  - `review_until_pass` detects timeout and auto-runs deterministic fast-shard fallback.
  - round payload now includes `tests_timeout` and `timeout_fallback`.
  - defect plan emits `TEST_TIMEOUT` and prioritizes timeout-specific repair order.
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
- Stress reason-drift artifact retention/checksum is online:
  - reason-drift artifact now supports retention rotation:
    `validation.ops_stress_autorun_reason_drift_retention_days`.
  - checksum index now supports audit trace:
    `validation.ops_stress_autorun_reason_drift_checksum_index_enabled`.
  - compliance artifact:
    `output/review/stress_autorun_reason_drift_checksum_index.json`.
  - gate/ops/review defect plan now surfaces
    `STRESS_AUTORUN_REASON_ARTIFACT_ROTATION` /
    `STRESS_AUTORUN_REASON_CHECKSUM_INDEX`.
- Stress history artifact retention/checksum is online:
  - history artifact now supports retention rotation:
    `validation.ops_stress_autorun_history_retention_days`.
  - checksum index now supports audit trace:
    `validation.ops_stress_autorun_history_checksum_index_enabled`.
  - compliance artifact:
    `output/review/stress_autorun_history_checksum_index.json`.
- Shared artifact governance utility is online:
  - common module:
    `src/lie_engine/orchestration/artifact_governance.py`.
  - unified flow (`collect -> rotate -> checksum index`) is now reused by:
    `temporal_autofix_patch` / `stress_autorun_history` / `stress_autorun_reason_drift`.
  - release orchestrator now routes these artifact chains through one governance entry,
    reducing duplicated retention/checksum logic and keeping behavior一致。
- Reconcile row-diff artifact governance is online:
  - row-diff drilldown artifact now supports retention + checksum index:
    `validation.ops_reconcile_broker_row_diff_artifact_retention_days` /
    `validation.ops_reconcile_broker_row_diff_artifact_checksum_index_enabled`.
  - compliance index artifact:
    `output/review/reconcile_row_diff_checksum_index.json`.
  - gate/ops/review defect plan now surfaces:
    `RECONCILE_BROKER_ROW_DIFF_ARTIFACT_ROTATION` /
    `RECONCILE_BROKER_ROW_DIFF_ARTIFACT_CHECKSUM_INDEX`.
- Profile-based artifact governance config is online:
  - validation key:
    `validation.ops_artifact_governance_profiles`.
  - per-profile declarative fields:
    `json_glob / md_glob / checksum_index_filename / retention_days / checksum_index_enabled`.
  - release governance routing now supports profile override while remaining backward-compatible with legacy retention/checksum keys.
  - current onboarded profiles:
    `temporal_autofix_patch / stress_autorun_history / stress_autorun_reason_drift / reconcile_row_diff`.
- Artifact governance compliance snapshot is online:
  - gate/ops now include `artifact_governance` section with profile-level policy snapshot.
  - monitors:
    `required_profiles_present_ok / policy_alignment_ok`.
  - drift alerts:
    `artifact_governance_policy_mismatch` /
    `artifact_governance_legacy_policy_drift`.
  - defect plan emits:
    `ARTIFACT_GOVERNANCE_PROFILE_MISSING` /
    `ARTIFACT_GOVERNANCE_POLICY_MISMATCH` /
    `ARTIFACT_GOVERNANCE_LEGACY_DRIFT`.
- Governance strict mode + baseline freeze is online:
  - new validation keys:
    `validation.ops_artifact_governance_strict_mode_enabled` /
    `validation.ops_artifact_governance_profile_baseline`.
  - gate/ops now surface strict freeze checks:
    `legacy_alignment_ok / baseline_freeze_ok / strict_mode_ok`.
  - strict mode blocks release when profile/policy/legacy/baseline drifts exist:
    `artifact_governance_strict_mode_blocked`.
  - defect plan emits:
    `ARTIFACT_GOVERNANCE_BASELINE_DRIFT` /
    `ARTIFACT_GOVERNANCE_STRICT_BLOCKED`.

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
1. Add baseline snapshot promotion workflow（从“手工 baseline”升级为“review 通过后自动固化 baseline”，并支持回滚锚点）。
