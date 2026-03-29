# TradingView Basis Arbitrage BTC Base-Qty Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `tv_basis_btc_spot_perp_v1` 从不可执行的 `20 USDT` quote-driven 合同修订为可执行的 `0.002 BTC` base-qty + `160 USDT` 预算上限合同，并让 gate 在 executor 之前 fail-close 不可执行情形。

**Architecture:** 保留现有 `tv_basis_arb_common -> tv_basis_arb_gate -> tv_basis_arb_webhook -> tv_basis_arb_executor -> tv_basis_arb_state` 主链，不新增新的 execution path。核心修订是把 entry authority 从 `requested_notional_usdt` 切换为 `target_base_qty`，并把预算视为独立的 fail-close 约束而不是最终下单数量 authority。

**Tech Stack:** Python 3、repo-owned scripts、JSON state/review artifacts、Pytest、bash/ssh dry acceptance

---

## 0. 范围与约束

- change class 目标：`LIVE_EXECUTION_PATH`
- 保持：
  - `strategy_id = tv_basis_btc_spot_perp_v1`
  - `symbol = BTCUSDT`
  - same-venue `spot long + perp short`
  - 现有 recovery state machine / mutex / idempotency
- 修订：
  - `target_base_qty = 0.002`
  - `max_quote_budget_usdt = 160.0`
- 不做：
  - 多 symbol / 多 venue
  - 动态 sizing
  - futures 权限修复
  - 真实下单 acceptance

### 0.1 Effective budget 规则（必须写死）

兼容期内，gate 的预算 authority 统一定义为：

```python
effective_quote_budget_usdt = min(
    max_quote_budget_usdt,
    requested_notional_usdt if requested_notional_usdt is a valid positive number else max_quote_budget_usdt,
)
```

含义：

- `target_base_qty` 决定真正想下多少 BTC；
- `effective_quote_budget_usdt` 只决定“当前预算是否允许执行这个 target_base_qty”；
- 对于 `tv_basis_btc_spot_perp_v1` 的默认 webhook runtime policy，`requested_notional_usdt` 应直接等于 `max_quote_budget_usdt`，即 `160.0`；
- 任何小于 `160.0` 的传入预算只能让 gate 更严格，不能放宽预算上限。

## 1. 目标文件结构

### Modify

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_common.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_gate.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/binance_live_common.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_state.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_executor.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_gate.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_state.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_executor.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py`

### Source-owned outputs to preserve

- `output/state/tv_basis_arb_idempotency.json`
- `output/state/tv_basis_arb_positions.json`
- `output/state/tv_basis_arb_recovery.json`
- `output/review/*_tv_basis_arb_signal.json`
- `output/review/*_tv_basis_arb_gate.json`
- `output/review/*_tv_basis_arb_execution.json`
- `output/review/*_tv_basis_arb_position.json`
- `output/review/*_tv_basis_arb_closeout.json`

## 2. Task 1：先把策略合同单独切到 base-qty + budget

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_common.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_gate.py`

- [ ] **Step 1: 在现有 common-contract 断言里先写 failing test**

```python
def test_strategy_config_includes_baseqty_and_budget_contract() -> None:
    common = load_strategy_definition("tv_basis_btc_spot_perp_v1")
    gate = common["gate"]
    assert gate["target_base_qty"] == 0.002
    assert gate["max_quote_budget_usdt"] == 160.0
    assert gate["max_notional_usdt"] == 160.0
```

- [ ] **Step 2: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_gate.py -k baseqty_and_budget_contract
```

Expected: FAIL（合同字段尚未存在）

- [ ] **Step 3: 在 `tv_basis_arb_common.py` 最小修改策略合同**

要求：

- 增加 `target_base_qty = 0.002`
- 增加 `max_quote_budget_usdt = 160.0`
- 将兼容字段 `max_notional_usdt` 同步到 `160.0`
- 不删除旧字段

- [ ] **Step 4: 跑测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_gate.py -k baseqty_and_budget_contract
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_common.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_gate.py
git commit -m "feat(live): add tv basis baseqty strategy contract"
```

## 3. Task 2：让 gate 使用 target_base_qty + effective budget

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_gate.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/binance_live_common.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_gate.py`

- [ ] **Step 1: 写 target_base_qty 输出 failing test**

```python
def test_gate_outputs_target_base_qty_and_effective_budget() -> None:
    result = evaluate_tv_basis_gate(...)
    assert result["target_base_qty"] == 0.002
    assert result["max_quote_budget_usdt"] == 160.0
    assert result["effective_quote_budget_usdt"] == 160.0
```

- [ ] **Step 2: 写预算阻断 failing test**

```python
def test_gate_blocks_when_effective_budget_is_below_estimated_quote() -> None:
    result = evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=140.0,
        market_snapshot=executable_snapshot,
    )
    assert result["passed"] is False
    assert "quote_budget_exceeded" in result["reasons"]
```

- [ ] **Step 3: 写可执行通过 failing test**

```python
def test_gate_passes_when_target_base_qty_and_budget_are_both_executable() -> None:
    result = evaluate_tv_basis_gate(
        strategy_id="tv_basis_btc_spot_perp_v1",
        requested_notional_usdt=160.0,
        market_snapshot=executable_snapshot,
    )
    assert result["passed"] is True
```

- [ ] **Step 4: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_gate.py -k 'effective_budget or target_base_qty_and_budget_are_both_executable'
```

Expected: FAIL

- [ ] **Step 5: 最小补齐 `binance_live_common.py` 所需 helper**

只允许补：

- 现有 `BinanceUsdMMarketClient.exchange_info()` 缺失接口
- snapshot 所需最小字段

禁止扩大为新的交易框架。

- [ ] **Step 6: 在 `tv_basis_arb_gate.py` 实现新合同逻辑**

要求：

- 从策略定义读取 `target_base_qty` / `max_quote_budget_usdt`
- 计算：
  - `effective_quote_budget_usdt`
  - `estimated_quote_for_target_usdt`
  - `estimated_perp_notional_usdt`
- fail-close reason 继续使用现有命名：
  - `spot_min_qty_unmet`
  - `spot_min_notional_unmet`
  - `perp_min_qty_unmet`
  - `perp_min_notional_unmet`
  - `quote_budget_exceeded`
- 保留 basis / spread / OI 判定

- [ ] **Step 7: 跑 gate 全量测试**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_gate.py
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_gate.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/binance_live_common.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_gate.py
git commit -m "fix(live): gate tv basis on baseqty budget contract"
```

## 4. Task 3：把 state ledger 显式升级到新合同字段

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_state.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_state.py`

- [ ] **Step 1: 为 begin_entry persistence 写 failing test**

```python
def test_begin_entry_persists_target_base_qty_and_budget() -> None:
    position = ledger.begin_entry(...)
    assert position["target_base_qty"] == 0.002
    assert position["max_quote_budget_usdt"] == 160.0
```

- [ ] **Step 2: 为 replay / recovery persistence 写 failing test**

```python
def test_recovery_state_keeps_target_base_qty_and_budget() -> None:
    recovery = ledger.record_needs_recovery(...)
    assert recovery["target_base_qty"] == 0.002
    assert recovery["max_quote_budget_usdt"] == 160.0
```

- [ ] **Step 3: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_state.py -k 'target_base_qty or max_quote_budget_usdt'
```

Expected: FAIL

- [ ] **Step 4: 在 `tv_basis_arb_state.py` 最小实现新字段持久化**

要求：

- attempt / position / recovery 都能保留：
  - `target_base_qty`
  - `max_quote_budget_usdt`
- 兼容已有 `requested_notional_usdt`
- 不破坏 replay mismatch / predecessor 规则

- [ ] **Step 5: 跑 state 全量测试**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_state.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_state.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_state.py
git commit -m "fix(live): persist tv basis baseqty contract in state ledger"
```

## 5. Task 4：把 executor entry 改成固定 base-qty 下单

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_executor.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_executor.py`

- [ ] **Step 1: 为 entry 两腿同数量写 failing test**

```python
def test_entry_executes_spot_buy_and_perp_short_with_same_target_base_qty() -> None:
    result = executor.execute_entry(...)
    assert spot.calls[0]["quantity"] == pytest.approx(0.002)
    assert spot.calls[0]["quote_order_qty"] is None
    assert perp.calls[0]["quantity"] == pytest.approx(0.002)
```

- [ ] **Step 2: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_executor.py -k same_target_base_qty
```

Expected: FAIL

- [ ] **Step 3: 在 `tv_basis_arb_executor.py` 最小实现 base-qty entry**

要求：

- entry 读取 `target_base_qty`
- spot `BUY quantity=target_base_qty`
- perp `SELL quantity=target_base_qty`
- 不再让 spot `quoteOrderQty` 成为第一腿 authority
- 继续保留：
  - `RunHalfhourMutex`
  - idempotency
  - partial fill / reject / transport ambiguity -> `needs_recovery`

- [ ] **Step 4: 跑 executor 全量测试**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_executor.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_executor.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_executor.py
git commit -m "fix(live): execute tv basis entry with fixed baseqty"
```

## 6. Task 5：更新 webhook runtime policy 与 artifact 透传

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py`

- [ ] **Step 1: 为 runtime policy 写 failing test**

```python
def test_runtime_policy_defaults_to_baseqty_and_max_budget() -> None:
    policy = current_runtime_policy(...)
    assert policy["target_base_qty"] == 0.002
    assert policy["max_quote_budget_usdt"] == 160.0
    assert policy["requested_notional_usdt"] == 160.0
```

- [ ] **Step 2: 为 gate artifact 透传写 failing test**

```python
def test_entry_gate_artifact_keeps_target_base_qty_and_budget_fields() -> None:
    result = handle_webhook(...)
    gate = read_json(...)
    assert gate["target_base_qty"] == 0.002
    assert gate["max_quote_budget_usdt"] == 160.0
    assert gate["effective_quote_budget_usdt"] == 160.0
```

- [ ] **Step 3: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_webhook.py -k 'target_base_qty or max_quote_budget_usdt or effective_quote_budget_usdt'
```

Expected: FAIL

- [ ] **Step 4: 在 `tv_basis_arb_webhook.py` 最小实现 runtime policy / artifact 对齐**

要求：

- runtime policy 增加：
  - `target_base_qty`
  - `max_quote_budget_usdt`
  - `requested_notional_usdt = max_quote_budget_usdt`（默认）
- gate artifact 透传：
  - `target_base_qty`
  - `max_quote_budget_usdt`
  - `effective_quote_budget_usdt`
  - `estimated_quote_for_target_usdt`

- [ ] **Step 5: 跑 webhook 全量测试**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_webhook.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py
git commit -m "fix(live): surface tv basis baseqty budget policy in webhook artifacts"
```

## 7. Task 6：更新 runbook 与 operator 文案

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`

- [ ] **Step 1: 更新 runbook 合同描述**

必须改成：

- `target_base_qty = 0.002 BTC`
- `max_quote_budget_usdt = 160.0`
- 不再把 `20 USDT` 写成真实下单 authority

- [ ] **Step 2: 更新 gate / smoke / recovery 文案**

要求：

- 强调 gate 先检查 exchange constraints + budget
- 强调 base-qty 驱动不消除 `needs_recovery`
- 强调真实 acceptance 仍受 futures 权限与账户条件限制

- [ ] **Step 3: 文档自检**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
rg -n "20 USDT|quoteOrderQty|target_base_qty|max_quote_budget_usdt" \
  system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md \
  system/scripts/tv_basis_arb_common.py \
  system/scripts/tv_basis_arb_webhook.py
```

Expected:

- runbook 中不再把 `20 USDT` 写成真实 authority
- 新字段已出现

- [ ] **Step 4: Commit**

```bash
git add /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md
git commit -m "docs(live): align tv basis runbook to baseqty contract"
```

## 8. Task 7：全量回归与云端 dry acceptance

**Files:**
- Verify only

- [ ] **Step 1: 跑本地 broad pytest**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q \
  tests/test_tv_basis_arb_gate.py \
  tests/test_tv_basis_arb_state.py \
  tests/test_tv_basis_arb_executor.py \
  tests/test_tv_basis_arb_webhook.py \
  tests/test_binance_infra_canary.py \
  tests/test_binance_live_takeover.py
```

Expected: PASS

- [ ] **Step 2: 定点同步云端脚本**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
rsync -az --itemize-changes -e 'ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
  system/scripts/binance_live_common.py \
  system/scripts/tv_basis_arb_common.py \
  system/scripts/tv_basis_arb_gate.py \
  system/scripts/tv_basis_arb_state.py \
  system/scripts/tv_basis_arb_executor.py \
  system/scripts/tv_basis_arb_webhook.py \
  ubuntu@43.153.148.242:/home/ubuntu/openclaw-system/scripts/
```

Expected: changed files synced, no error

- [ ] **Step 3: 跑云端“阻断型” dry harness（保证不会真实下单）**

Run:

```bash
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=yes ubuntu@43.153.148.242 <<'REMOTE'
set -euo pipefail
cd /home/ubuntu/openclaw-system
PYTHONPATH=scripts python3 - <<'PY'
import json
from pathlib import Path
import tv_basis_arb_webhook

def blocked_snapshot():
    return {
        "symbol": "BTCUSDT",
        "spot_price": 100000.0,
        "perp_mark_price": 100120.0,
        "perp_index_price": 100100.0,
        "open_interest_contracts": 1200.0,
        "open_interest_usdt": 120144000.0,
        "snapshot_ts_utc": "2026-03-26T12:30:00Z",
        "snapshot_time_ms": 1774534200000,
        "exchange_constraints": {
            "spot": {"min_qty": 0.00001, "step_size": 0.00001, "min_notional": 5.0},
            "perp": {"min_qty": 0.001, "step_size": 0.001, "min_notional": 100.0},
        },
    }

def forbid_execute(*args, **kwargs):
    raise AssertionError("executor must not run in blocked dry harness")

tv_basis_arb_webhook.build_market_snapshot = lambda **_: blocked_snapshot()
tv_basis_arb_webhook.TvBasisArbExecutor.execute_entry = forbid_execute
result = tv_basis_arb_webhook.handle_webhook(
    {
        "strategy_id": "tv_basis_btc_spot_perp_v1",
        "symbol": "BTCUSDT",
        "event_type": "entry_check",
        "tv_timestamp": "2026-03-26T12:30:00Z",
    },
    output_root=Path("/tmp/tv_basis_blocked_dry"),
)
print(json.dumps(result, ensure_ascii=False))
PY
REMOTE
```

Expected:

- `status=gate_blocked`
- `execution=null`
- 不触发任何真实下单

- [ ] **Step 4: 跑云端“可执行型” dry harness（fake executor，不真实下单）**

Run:

```bash
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=yes ubuntu@43.153.148.242 <<'REMOTE'
set -euo pipefail
cd /home/ubuntu/openclaw-system
PYTHONPATH=scripts python3 - <<'PY'
import json
from pathlib import Path
import tv_basis_arb_webhook

def executable_snapshot():
    return {
        "symbol": "BTCUSDT",
        "spot_price": 70000.0,
        "perp_mark_price": 70100.0,
        "perp_index_price": 70090.0,
        "open_interest_contracts": 1200.0,
        "open_interest_usdt": 84120000.0,
        "snapshot_ts_utc": "2026-03-26T12:30:00Z",
        "snapshot_time_ms": 1774534200000,
        "exchange_constraints": {
            "spot": {"min_qty": 0.00001, "step_size": 0.00001, "min_notional": 5.0},
            "perp": {"min_qty": 0.001, "step_size": 0.001, "min_notional": 100.0},
        },
    }

def fake_execute(self, *, strategy_id, symbol, idempotency_key, requested_notional_usdt, tv_timestamp):
    return {
        "status": "open_hedged",
        "position": {
            "position_key": "dry-pos-1",
            "status": "open_hedged",
            "strategy_id": strategy_id,
            "symbol": symbol,
            "target_base_qty": 0.002,
            "max_quote_budget_usdt": 160.0,
        },
    }

tv_basis_arb_webhook.build_market_snapshot = lambda **_: executable_snapshot()
tv_basis_arb_webhook.TvBasisArbExecutor.execute_entry = fake_execute
result = tv_basis_arb_webhook.handle_webhook(
    {
        "strategy_id": "tv_basis_btc_spot_perp_v1",
        "symbol": "BTCUSDT",
        "event_type": "entry_check",
        "tv_timestamp": "2026-03-26T12:30:00Z",
    },
    output_root=Path("/tmp/tv_basis_executable_dry"),
)
print(json.dumps(result, ensure_ascii=False))
PY
REMOTE
```

Expected:

- gate artifact 含 `target_base_qty` / `max_quote_budget_usdt`
- fake executor 被调用但不发真实订单
- 不出现意外 `needs_recovery`

- [ ] **Step 5: 记录真实 acceptance blocker**

若 futures 权限仍未修复，必须在验收结论中明确：

- implementation complete
- real acceptance still blocked by remote futures permission / account readiness

- [ ] **Step 6: 结束前检查工作树干净**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
git status --short
```

Expected: no uncommitted changes before handoff

## 9. 交付判定

以下全部满足才算本计划完成：

1. `tv_basis_btc_spot_perp_v1` 合同改为 base-qty + budget
2. gate 在 executor 之前就能阻断不可执行的数量/预算
3. state ledger 明确持久化新合同字段
4. executor entry 按固定 `0.002 BTC` 驱动 spot/perp 两腿
5. webhook / artifact 能审计新合同字段与 effective budget
6. runbook 文案完成同步
7. 本地 broad pytest 通过
8. 云端 dry acceptance 使用安全 harness 通过
9. 若真实 acceptance 仍被账户权限拦住，必须明确报告 blocker，而不是强行推进
