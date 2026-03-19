# Adaptive Task Modes

This playbook exists to reduce prompt drift and context tax for new module work.

## Core idea

Do not run every task with the same mental posture.

Pick one primary execution mode per task, keep hard guardrails stable, and stop
once the mode-specific success criteria are satisfied.

## Hard guardrails

These apply in every mode:

- Keep live-capital work read-only unless gates are explicitly clear.
- For backtests, signal detection, and historical indicator math, do not use
  forward-looking array access or implicit wall-clock time.
- Prefer source-owned state over consumer-side mirror fields.
- If a new field can be expressed in a source artifact, do that before adding
  another top-level derived field in a consumer.
- If two consecutive turns do not change `state`, `queue`, `gate`,
  `source ownership`, or `user-facing capability`, stop and report instead of
  continuing low-value cleanup.

## Institutional overlays

These do not replace hard guardrails. They make the modes more suitable for
production trading governance.

### Failure severity matrix

- `P0`: execution-path failure or source-of-truth ambiguity. This is the only
  class that may justify trading halt or `panic_close_all()`.
- `P1`: process or instance failure. Recycle/kill the instance if needed, but
  do not treat it as a trading panic unless execution truth is ambiguous.
- `P2`: notification, metrics, dashboard, or non-trading websocket failure.
  Degrade and report; do not map these directly to trading panic behavior.

### Source-of-truth registry

- Orders and positions: read venue/account-backed artifacts as truth; do not
  rebuild ownership in consumers.
- Queues, lanes, gates, and scores: express them in source artifacts first,
  then propagate downstream.
- Live blockers and clearing conditions: consume blocker artifacts directly,
  not UI or top-brief reconstructions.
- Research validity and OOS claims: tie them to deterministic research
  artifacts, not narrative summaries.

### Change classes

- `DOC_ONLY`
- `RESEARCH_ONLY`
- `SIM_ONLY`
- `LIVE_GUARD_ONLY`
- `LIVE_EXECUTION_PATH`

Every new module task should identify the narrowest change class that still
fits the work. `LIVE_EXECUTION_PATH` is the exception class, not the default.

### Time governance

- Backtests, replays, and historical indicators must use explicit event
  timestamps only.
- Live deadlines, retries, and backoff should use monotonic clocks.
- Normalize business timestamps to UTC.
- Do not mix wall clock time into strategy, portfolio, simulation, or
  reconciliation logic.

### Rollout ladder

- `shadow`
- `replay`
- `canary`
- `broader_rollout`

If work approaches live behavior, prefer climbing this ladder instead of
jumping directly from code change to broad execution.

## Modes

### `architecture_review`

Use when the task is about drift, ownership, cycles, duplicated state, stale
consumers, or “re-check the full architecture”.

Priorities:

1. Find source/consumer cycles.
2. Find duplicated state derivation.
3. Find lane/count/priority mismatches that already affect real artifacts.
4. Stop after the highest-signal findings are explicit.

Success:

- The top architectural risks are concrete and file-specific.
- If fixes are applied, source ownership becomes simpler, not broader.
- Residual risk and rollback point are still explicit after the review.

### `source_first_implementation`

Use when adding a new queue, lane, field, or handoff behavior.

Priorities:

1. Add behavior to the source artifact first.
2. Then propagate to one consumer.
3. Reuse existing source payload where possible.
4. Avoid adding mirror fields unless required for compatibility.

Success:

- New capability exists in a source artifact.
- One downstream consumer shows it correctly.
- No extra mirror layer was added without need.
- The chosen change class stays below `LIVE_EXECUTION_PATH` unless explicitly required.

### `ui_rendering`

Use when the task is a panel, dashboard, visual window, workflow visibility, or
operator UX improvement.

Priorities:

1. Consume existing source artifacts.
2. Do not invent backend state just for the UI.
3. Make transmission paths and impact scope visible.
4. Prefer stable, maintainable rendering over deep framework surgery.

Success:

- A user can see current head, queue, blockers, and chain propagation.
- UI does not require new backend semantics unless justified.

### `research_backtest`

Use when the task is strategy validity, cross-market testing, win rate,
profitability, OOS effectiveness, or comparison reports.

Priorities:

1. Protect against future leakage.
2. Compare in-sample and out-of-sample explicitly.
3. Prefer reports over raw metric dumps.
4. Record where the strategy does *not* generalize.

Success:

- A recent comparison artifact exists.
- OOS caveats are explicit.
- Claims are tied to real metrics, not interpretation only.
- Any suggestion of promotion still respects the shadow/replay/canary ladder.

### `live_guard_diagnostics`

Use when the task is probe, ready-check, canary, remote handoff, account scope,
or live blocker analysis.

Priorities:

1. Read-only first.
2. Make account scope explicit.
3. Separate profitability confirmation from execution readiness.
4. End with blocker and clearing conditions.

Success:

- Account scope is known.
- `ops_live_gate` and `risk_guard` blockers are explicit.
- No order was placed.
- Failure severity stays classified; non-trading failures are not escalated as trading panic events.

### `default_implementation`

Use when no other mode clearly dominates.

Priorities:

1. Make the smallest useful change.
2. Validate the touched path.
3. Stop after user-facing value appears.

## Stop rules

- Stop if the task result is already source-complete and one consumer shows it.
- Stop if further work is only naming, helper splitting, or mirror-field growth.
- Stop if the next likely step changes no lane, gate, queue, state, or user view.
- Stop and ask only if source ownership or live safety would otherwise become unclear.

## Recommended workflow

1. Run the selector script before starting a new module task.
2. Use the selected mode as the primary execution posture.
3. Treat secondary modes as constraints, not equal priorities.
4. Record the change class, validations run, residual risk, and rollback point.
5. After finishing, check whether the mode-specific success criteria were met.

## Selector

Use:

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_adaptive_task_mode_report.py \
  --task-summary "Add a task visualization panel for operator flow" \
  --changed-path system/dashboard/web/dist/index.html \
  --changed-path system/scripts/build_hot_universe_operator_brief.py
```

This produces a stamped report in `system/output/review/` with:

- selected primary mode
- secondary modes
- hard guardrails
- execution path
- stop rules
- validation plan
