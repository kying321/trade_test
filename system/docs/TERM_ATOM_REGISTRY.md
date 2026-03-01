# Term Atom Registry

## Purpose
This document defines a single contract from narrative theory to executable factors:

- term language: ICT / Al Brooks / LiE-PDF
- computable expression: formula and input fields
- falsification: invalidation conditions
- evaluation: metric windows and directions

Machine-readable source of truth:

- `system/config/term_atoms.yaml`

Validation script:

- `python3 system/scripts/validate_term_atoms.py`

## Runtime Mapping (Current Engine)
Current mappings already implemented in code:

- trend model scoring (17-point denominator): `system/src/lie_engine/signal/trend.py` and `system/src/lie_engine/signal/engine.py`
- range model scoring (14-point denominator): `system/src/lie_engine/signal/range_engine.py` and `system/src/lie_engine/signal/engine.py`
- feature inputs (`ma20/ma60/atr14/rsi14/roll_high/roll_low`): `system/src/lie_engine/signal/features.py`

Interpretation:

- ICT and Al Brooks are represented by deterministic proxies on OHLCV bars.
- LiE-PDF model scoring is explicit in code and is the current production baseline.
- Native microstructure atoms are staged as `pending` until L2 and trade tick streams are connected.

## Truth Baseline
Minimum truth baseline is defined in `term_atoms.yaml`:

- required L2 fields: `exchange,symbol,event_ts_ms,recv_ts_ms,seq,prev_seq,bids,asks`
- required trade fields: `exchange,symbol,trade_id,event_ts_ms,recv_ts_ms,price,qty,side`
- hard time discipline:
  - `ntp_max_offset_ms <= 5`
  - `cross_source_tolerance_ms = 80`
- fuse conditions:
  - continuous gap `>=2500ms`
  - consecutive gaps `>=3`
  - stale stream `>=1500ms`
- retention:
  - L2/trades/news_labels all `>=90 days`

## How To Extend
When adding a new term atom:

1. Add atom entry to `system/config/term_atoms.yaml`.
2. Validate schema and coverage:
   - `python3 system/scripts/validate_term_atoms.py`
3. If formula is executable now, map the atom id into signal code and add tests.
4. If formula requires microstructure, keep it as `pending` until source quality gates pass.

## Non-Goals (Current Stage)
- No claim of native microstructure alpha before cross-source L2 alignment is in production.
- No promotion from `proxy` to `active` without metric evidence.
