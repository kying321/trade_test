# ICT + CVD Integration For Fenlie

## Executive Summary
- Fenlie can support a useful `ICT + CVD-lite` confirmation layer now.
- Fenlie cannot support strict institutional-grade CVD yet.
- The correct integration point is the existing microstructure adjustment path:
  - collect trade and L2 evidence
  - compute short-horizon delta confirmation factors
  - apply boost, penalty, or veto
- CVD should not become a primary alpha authority in the current stack.

## Repo-Grounded Assessment

### What Fenlie already has
- Provider-level access to:
  - REST order book snapshots via `fetch_l2(...)`
  - recent trade pulls via `fetch_trades(...)`
- Cross-source trade bucketing:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/cross_source.py`
  - `bucket_trade_flow(...)`
- Existing microstructure confirmation layer:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/microstructure.py`
  - `queue_imbalance`
  - `ofi_norm`
  - `micro_alignment`
  - `evidence_score`
- Existing microstructure integration into confidence/risk:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/engine.py`
  - `_microstructure_adjustment(...)`
- Existing cross-source audit and gating:
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/engine.py`
  - `sync_ok`
  - `gap_ok`
  - `time_sync_ok`

### What Fenlie does not have yet
- Persistent websocket diff-depth replay
- Locally reconstructed order book over long sessions
- Replayable event store for strict absorption/failed-auction inference
- Uniform microstructure support across all provider profiles

## Why This Is CVD-Lite, Not Strict CVD

Fenlie currently relies on public exchange REST pulls plus short-horizon alignment logic. That is enough for:
- rolling delta confirmation
- sweep divergence confirmation
- effort vs result penalty
- short-lived no-trade veto

It is not enough for:
- exact session-long cumulative volume delta over fully replayed order flow
- robust queue replenishment analysis
- book-level absorption inference with event continuity guarantees

## Exchange Capability Cross-Check

Primary sources used to verify the current boundary:
- Binance REST market data:
  - [Binance Spot Market Data Endpoints](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
- Binance websocket local book guidance:
  - [Binance Spot WebSocket Streams](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
- Bybit orderbook:
  - [Bybit V5 Orderbook](https://bybit-exchange.github.io/docs/v5/market/orderbook)
- Bybit recent trades:
  - [Bybit V5 Recent Public Trades](https://bybit-exchange.github.io/docs/v5/market/recent-trade)

Important implications from those docs:
- Binance `GET /api/v3/depth` is a snapshot, not a replayable local book.
- Binance strict local book maintenance requires websocket diff-depth plus snapshot synchronization.
- Bybit orderbook endpoint is also snapshot format.
- Bybit recent trade endpoint exposes taker side, which is enough for approximate delta, not full book replay.

## Fenlie-Usable ICT + CVD Factors

Before listing the factors, the most important design upgrade is semantic:
- do not treat CVD as a scalar only
- classify it into:
  - `context`
  - `trust tier`
  - `veto reason`

That is the practical bridge between:
- ICT structure
- microstructure evidence
- operator review artifacts

### Semantic Layers

#### 1. CVD Context Mode
- `continuation`
  - BOS or displacement is being confirmed by directional flow
- `reversal`
  - sweep or reclaim is being confirmed by non-confirming or reversing delta
- `absorption`
  - effort is present but price result is weak
- `failed_auction`
  - extreme probe cannot sustain outside prior range
- `unclear`
  - evidence is too weak or conflicted

#### 2. CVD Trust Tier
- `single_exchange_low`
  - one venue only, thin evidence, penalty-only
- `single_exchange_ok`
  - one venue is usable, limited boost allowed
- `cross_exchange_confirmed`
  - Binance and Bybit align, boost/penalty/veto allowed
- `cross_exchange_conflicted`
  - venues disagree, downgrade or veto
- `unavailable`
  - profile/data quality cannot support CVD semantics

#### 3. CVD Veto Reason
- `sweep_without_delta_confirmation`
- `displacement_without_flow`
- `effort_result_divergence`
- `cross_exchange_conflict`
- `low_sample_or_gap_risk`
- `stale_micro_context`
- `structure_missing`

These identifiers matter because they keep:
- review artifacts deterministic
- handoff payloads compact
- runtime veto logic auditable

### 1. CVD Sweep Divergence
- ICT anchor:
  - `ICT_LIQ_SWEEP_PROXY`
- Intent:
  - confirm failed break/reclaim when price sweeps liquidity but delta does not confirm the new extreme
- Fenlie role:
  - boost reversal setup if sweep + reclaim + divergence align
  - veto breakout continuation if sweep happens but delta keeps extending in breakout direction

### 2. CVD Displacement Confirmation
- ICT anchor:
  - `ICT_BOS_PROXY`
  - `ICT_FVG_PROXY`
- Intent:
  - confirm that displacement or BOS has actual aggressive trade support
- Fenlie role:
  - boost continuation confidence
  - reduce false-breakout entries

### 3. CVD Absorption Reversal
- ICT/Wyckoff bridge:
  - sweep, spring, upthrust, failed breakout
- Intent:
  - detect large effort with weak result near liquidity
- Fenlie role:
  - penalty or veto against breakout chase
  - confirmation for reclaim-based reversal

### 4. CVD Failed Auction Proxy
- ICT anchor:
  - PD array edge
  - failed breakout context
- Intent:
  - treat marginal new highs/lows with delta stall and range re-entry as no-trade or fade context
- Fenlie role:
  - veto only

### 5. CVD Effort-Result Divergence
- Fenlie bridge:
  - `flow_effort_gap`
  - `flow_absorption_skew`
- Intent:
  - penalize setups where delta/notional effort is high but structure progress is weak
- Fenlie role:
  - penalty only

## Risk Rules

### Rule 1: Profile Floor
- Only enable ICT+CVD on public-spot profiles with trade-side and L2 availability.
- Disable for:
  - `opensource_dual`
  - `opensource_primary`
  - placeholder-only profiles

### Rule 2: Micro Evidence Fuse
- Require:
  - `schema_ok`
  - `sync_ok`
  - `gap_ok`
  - `time_sync_ok`
- If any fail:
  - no CVD veto authority
  - degrade to neutral or penalty-only

### Rule 3: Low-Sample Fuse
- Require minimum trade count and evidence score.
- If samples are thin:
  - no divergence claims
  - no sweep veto

### Rule 4: Cross-Source Degrade
- If cross-source audit is unstable:
  - cut boost
  - disable veto
  - surface a `cvd_quality_risk` flag

### Rule 5: Short Half-Life
- ICT+CVD evidence is intraday microstructure evidence.
- It should decay quickly and must not leak into daily regime state.

### Rule 6: Structure First
- CVD cannot create direction.
- It only confirms or blocks structure already detected by ICT or price action proxies.

## Recommended Runtime Output Shape

When ICT+CVD is later wired into the engine, each evaluated factor should emit:
- `factor_id`
- `context_mode`
- `trust_tier`
- `active`
- `boost`
- `penalty`
- `veto`
- `veto_reason`
- `half_life_seconds`
- `source_quality`

That is more useful than only storing:
- `net_qty`
- `trade_count`
- `alignment`

because the operator and review layers need semantics, not just raw flow numbers.

## Current Project Assessment

### Current config
- `data.provider_profile = dual_binance_bybit_public`
- `validation.micro_cross_source_audit_enabled = true`
- `validation.micro_min_trade_count = 20`

### Practical conclusion
- Current deployed config is suitable for `CVD-lite`.
- Current deployed config is not sufficient for strict order-flow CVD.

### Confidence level
- High confidence for:
  - short-horizon trade delta confirmation
  - divergence/confirmation/veto factors
- Low confidence for:
  - long-session cumulative delta
  - book-replenishment absorption
  - auction-quality inference without replayed book state

## Recommended Implementation Path

### Phase 1: Spec and Readiness
- Add factor spec
- Add readiness assessment script
- No runtime behavior change

### Phase 2: Extend Micro Factor State
- Add:
  - rolling delta
  - short-session cumulative delta
  - divergence score
  - effort-result score

### Phase 3: Signal Integration
- Feed ICT+CVD factors into:
  - boost
  - penalty
  - veto
- Keep authority at confirm-and-veto only

### Phase 4: Strict CVD Upgrade
- Introduce websocket diff-depth reconstruction
- Add replayable local book store
- Reclassify absorption and failed-auction factors out of proxy mode

## Implementation Files To Touch Later
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/microstructure.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/cross_source.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/signal/engine.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/engine.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/config/ict_cvd_factor_spec.yaml`

## Bottom Line
- Fenlie can support ICT + CVD as a confirmation layer now.
- Fenlie should not market its current stack as strict CVD.
- The safe path is:
  - structure first
  - CVD-lite confirms
  - cross-source quality gates trust
  - no primary alpha authority until websocket local-book reconstruction exists
