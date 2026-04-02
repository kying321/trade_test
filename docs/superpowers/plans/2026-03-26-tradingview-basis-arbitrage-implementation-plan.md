# TradingView Basis Arbitrage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fenlie 增加一个由 TradingView webhook 触发、自动执行 `BTCUSDT` 同交易所 `现货多 + 永续空` 基差套利的 v1 子系统，同时保持 authority 边界清晰、双腿执行可审计、故障可恢复。

**Architecture:** 采用“TradingView 只发策略 ID / 事件，Fenlie 本地读取固定策略配置并执行”的路径。实现拆成 4 个清晰单元：webhook ingress、basis gate evaluator、hedged executor、position/recovery controller；执行层尽量复用现有 `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py`，但绝不复用 `signal_to_order_tickets` 作为套利 authority。

**Tech Stack:** Python 3、stdlib `http.server`、repo-owned scripts、JSON artifacts / state ledgers、Pytest、bash runbook

---

## 0. 约束

- change class 目标：`LIVE_EXECUTION_PATH`
- TradingView webhook 只能触发 `strategy_id`，不能成为风控 authority
- v1 范围固定：
  - `strategy_id = tv_basis_btc_spot_perp_v1`
  - `symbol = BTCUSDT`
  - `spot long + perp short`
  - 因子门控：`basis + volatility + OI`
  - 单次名义金额上限：`20 USDT`
  - 平仓规则：`basis 回归阈值 + 最长持仓时间`
- 若任一腿成交而另一腿失败，必须进入 `needs_recovery`
- 任何真实执行都必须继续满足：
  - `run-halfhour-pulse` mutex
  - timeout `<=5000ms`
  - idempotency
  - source-owned state ledger
- 本计划不包含：
  - 多品种套利
  - 跨交易所套利
  - TradingView 直接下发名义金额 / leverage / entry/exit 参数

## 1. 目标文件结构

### Create

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_common.py`
  - 公共数据结构、枚举、JSON helper、webhook schema helper

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_gate.py`
  - basis / volatility / OI 门控计算
  - 只做判定，不下单

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_state.py`
  - idempotency / open positions / recovery state ledger
  - entry / exit / recovery 状态机工具

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_executor.py`
  - 现货买入 + 永续做空执行器
  - 平仓与 recovery 动作

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_webhook.py`
  - 接收 TradingView webhook
  - 调用 gate / executor / state
  - 输出 source-owned artifacts

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_gate.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_state.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_executor.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_webhook.py`

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`

### Modify

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py`
  - 如缺 Binance 现货 / 永续 mark price / OI / funding / position helpers，则最小增补

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md`
  - 仅在最后文档阶段补 webhook 服务部署 / smoke 命令（不在前置任务里碰）

### Source-owned state / artifacts（由实现落盘）

- `output/state/tv_basis_arb_idempotency.json`
- `output/state/tv_basis_arb_positions.json`
- `output/state/tv_basis_arb_recovery.json`
- `output/review/*_tv_basis_arb_signal.json`
- `output/review/*_tv_basis_arb_gate.json`
- `output/review/*_tv_basis_arb_execution.json`
- `output/review/*_tv_basis_arb_position.json`
- `output/review/*_tv_basis_arb_closeout.json`

## 2. Task 1：锁定 webhook ingress contract

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_common.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_webhook.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_webhook.py`

- [ ] **Step 1: 写 webhook schema 失败测试**

至少覆盖：

```python
def test_rejects_missing_strategy_id() -> None:
    payload = {"symbol": "BTCUSDT", "event_type": "entry_check"}
    assert validate_webhook_payload(payload).ok is False

def test_rejects_unknown_strategy_id() -> None:
    payload = {"strategy_id": "wrong", "symbol": "BTCUSDT", "event_type": "entry_check"}
    assert validate_webhook_payload(payload).ok is False

def test_accepts_minimal_entry_check_payload() -> None:
    payload = {
        "strategy_id": "tv_basis_btc_spot_perp_v1",
        "symbol": "BTCUSDT",
        "event_type": "entry_check",
        "tv_timestamp": "2026-03-26T00:00:00Z",
    }
    assert validate_webhook_payload(payload).ok is True
```

- [ ] **Step 2: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_webhook.py
```

Expected: FAIL（新模块尚未实现）

- [ ] **Step 3: 实现最小 schema validator 与 webhook skeleton**

`tv_basis_arb_common.py` 负责：
- `strategy_id` 常量
- `event_type` 校验
- JSON helper / schema helper

`tv_basis_arb_webhook.py` 先只实现：
- payload 解析
- schema 验证
- `signal artifact` 落盘

- [ ] **Step 4: 跑测试转绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_webhook.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_common.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_webhook.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_webhook.py
git commit -m "feat(live): add tradingview basis webhook contract"
```

## 3. Task 2：实现 basis / volatility / OI 门控

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_gate.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_gate.py`
- Modify as needed: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py`

- [ ] **Step 1: 先写 gate 失败测试**

至少覆盖：

```python
def test_entry_gate_passes_when_basis_vol_oi_all_green() -> None:
    snapshot = {...}
    gate = evaluate_entry_gate(snapshot, config=cfg)
    assert gate.allowed is True

def test_entry_gate_blocks_when_basis_below_threshold() -> None:
    gate = evaluate_entry_gate(snapshot, config=cfg)
    assert gate.allowed is False
    assert "basis_below_threshold" in gate.reasons

def test_entry_gate_blocks_when_oi_below_threshold() -> None:
    ...

def test_entry_gate_blocks_when_requested_notional_exceeds_20_usdt_cap() -> None:
    gate = evaluate_entry_gate(snapshot, config=cfg, requested_notional_usdt=25.0)
    assert gate.allowed is False
    assert "notional_exceeds_cap" in gate.reasons
```

- [ ] **Step 2: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_gate.py
```

Expected: FAIL

- [ ] **Step 3: 为 gate evaluator 提供最小 market snapshot helper**

优先复用 `binance_live_common.py`，按最小需要扩充：
- 现货 ticker
- 永续 mark/index/funding/open-interest 查询 helper

注意：
- 不把 helper 做成泛化交易框架
- 只满足 BTCUSDT basis gate 需求

- [ ] **Step 4: 实现 `tv_basis_arb_gate.py`**

输出至少包含：

```python
{
  "allowed": False,
  "basis_bps": 42.1,
  "volatility_1h": 0.013,
  "open_interest": 123456.0,
  "reasons": ["basis_below_threshold"],
}
```

并显式包含单次名义金额硬约束，例如：

```python
{
  "allowed": False,
  "requested_notional_usdt": 25.0,
  "max_notional_usdt": 20.0,
  "reasons": ["notional_exceeds_cap"],
}
```

- [ ] **Step 5: 跑测试转绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_gate.py
```

Expected: PASS

- [ ] **Step 6: 回归 `binance_live_common` / live actor**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_binance_infra_canary.py tests/test_binance_live_takeover.py
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_gate.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_gate.py
git commit -m "feat(live): add basis arbitrage gate evaluator"
```

## 4. Task 3：实现 state ledger 与双腿状态机

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_state.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_state.py`

- [ ] **Step 1: 先写状态机失败测试**

至少覆盖：

```python
def test_entry_creates_open_hedged_position() -> None:
    ...

def test_partial_fill_enters_needs_recovery() -> None:
    ...

def test_new_entry_blocked_while_recovery_exists() -> None:
    ...
```

- [ ] **Step 2: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_state.py
```

Expected: FAIL

- [ ] **Step 3: 实现最小 ledger**

需要落盘：
- `tv_basis_arb_idempotency.json`
- `tv_basis_arb_positions.json`
- `tv_basis_arb_recovery.json`

并支持至少这些状态：
- `entry_pending`
- `spot_buy_submitting`
- `spot_buy_filled_perp_pending`
- `perp_short_submitting`
- `open_hedged`
- `exit_pending`
- `needs_recovery`
- `closed`

- [ ] **Step 4: 跑测试转绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_state.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_state.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_state.py
git commit -m "feat(live): add basis arbitrage state ledger"
```

## 5. Task 4：实现双腿执行器

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_executor.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_executor.py`
- Modify as needed: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py`

- [ ] **Step 1: 先写执行器失败测试**

至少覆盖：

```python
def test_entry_executes_spot_buy_then_perp_short() -> None:
    ...

def test_spot_fill_then_perp_reject_marks_recovery() -> None:
    ...

def test_exit_executes_perp_close_then_spot_sell() -> None:
    ...
```

- [ ] **Step 2: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_executor.py
```

Expected: FAIL

- [ ] **Step 3: 实现执行器**

要求：
- 复用 `binance_live_common.py`
- 保持 `run-halfhour-pulse` mutex
- timeout `<=5000ms`
- 每条腿都写 artifact / ledger
- 若一腿成交另一腿失败，必须 `needs_recovery`

- [ ] **Step 4: 跑测试转绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_executor.py
```

Expected: PASS

- [ ] **Step 5: 交叉回归**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_binance_infra_canary.py tests/test_binance_live_takeover.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_executor.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_executor.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py
git commit -m "feat(live): add basis arbitrage executor"
```

## 6. Task 5：把 webhook、gate、state、executor 串起来

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_webhook.py`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_webhook.py`

- [ ] **Step 1: 为 entry_check / exit_check 写失败集成测试**

至少覆盖：

```python
def test_entry_check_writes_signal_and_gate_artifacts() -> None:
    ...

def test_entry_check_opens_position_when_gate_passes() -> None:
    ...

def test_exit_check_closes_position_when_basis_reverts() -> None:
    ...

def test_exit_check_closes_position_when_max_holding_time_exceeded() -> None:
    ...
```

- [ ] **Step 2: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_webhook.py
```

Expected: FAIL

- [ ] **Step 3: 串联执行路径**

`tv_basis_arb_webhook.py` 必须：
- 校验 webhook schema
- 读取固定策略配置
- 调用 gate evaluator
- 调用 executor / state controller
- 在 entry / exit artifact 中写入 notional / holding-time 审计字段
- 产出：
  - `tv_basis_arb_signal`
  - `tv_basis_arb_gate`
  - `tv_basis_arb_execution`
  - `tv_basis_arb_position`
  - `tv_basis_arb_closeout`

- [ ] **Step 4: 跑测试转绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_tv_basis_arb_webhook.py
```

Expected: PASS

- [ ] **Step 5: 全套 Python 回归**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q \
  tests/test_tv_basis_arb_gate.py \
  tests/test_tv_basis_arb_state.py \
  tests/test_tv_basis_arb_executor.py \
  tests/test_tv_basis_arb_webhook.py \
  tests/test_binance_infra_canary.py \
  tests/test_binance_live_takeover.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/tv_basis_arb_webhook.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_tv_basis_arb_webhook.py
git commit -m "feat(live): wire tradingview basis arbitrage flow"
```

## 7. Task 6：运行手册与验收命令

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md`

- [ ] **Step 1: 写 runbook**

必须包含：
- webhook payload 示例
- entry / exit gate 解释
- state ledger 解释
- recovery 解释
- `套利成功 != 基础设施 canary 成功 != strategy ticket ready`

- [ ] **Step 2: 增加最小 smoke 命令**

例如：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
python3 scripts/tv_basis_arb_webhook.py --host 127.0.0.1 --port 8788
```

以及：

```bash
curl -X POST http://127.0.0.1:8788/ \
  -H 'content-type: application/json' \
  -d '{"strategy_id":"tv_basis_btc_spot_perp_v1","symbol":"BTCUSDT","event_type":"entry_check","tv_timestamp":"2026-03-26T00:00:00Z"}'
```

- [ ] **Step 3: 提交**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md
git commit -m "docs(live): add tradingview basis arbitrage runbook"
```

## 8. Task 7：最小真实验收

**Files:**
- Verify only

- [ ] **Step 1: 本地 webhook smoke**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q \
  tests/test_tv_basis_arb_gate.py \
  tests/test_tv_basis_arb_state.py \
  tests/test_tv_basis_arb_executor.py \
  tests/test_tv_basis_arb_webhook.py
```

Expected: PASS

- [ ] **Step 2: 配置校验**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
PYTHONPATH=src python3 -m lie_engine.cli --config config.yaml validate-config
```

Expected: `ok: true`

- [ ] **Step 3: 云端 dry acceptance（不下单）**

要求：
- webhook ingress 可启动
- entry_check 不会直接因 schema 问题崩溃
- gate artifact 落盘，且包含 notional cap 判定字段

- [ ] **Step 4: 单次真实 acceptance（谨慎）**

仅当：
- 两腿账户可用
- 名义金额 <= 20 USDT
- recovery 队列为空
- 最长持仓时间 / exit gate 参数已能在 artifact 中被审计

才允许做一次真实 v1 acceptance。

## 9. 非目标 / 延后项

本计划不包含：
- ETH / SOL 多品种套利
- 跨交易所套利
- TradingView 多策略 ID
- 可视化 dashboard 面板扩展
- Pine Script 本身的社区发布/托管

## 10. 交付判定

以下全部满足才算完成：

1. webhook payload contract 固定且可校验
2. `basis + volatility + OI` gate 可独立测试
3. 双腿 executor 有明确 recovery state
4. `tv_basis_btc_spot_perp_v1` 可自动 entry / exit
5. artifact / state ledger 完整
6. 文档写明 authority 边界与风险
