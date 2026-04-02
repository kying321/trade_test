# Bybit-First Live Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `tv_basis_btc_spot_perp_v1` 增加第一条 Bybit-first 替代 venue live 主线，在不破坏现有 Binance fail-close boundary 的前提下，把主线推进到 `canary-ready`。

**Architecture:** 继续复用现有 `venue_capabilities.json` 的 single-writer / single-schema 合同，并把 `preferred_venue=bybit` 作为第一阶段的 hard-switch。新增 Bybit signed common、Bybit capability extension、Bybit basis live adapter 和 Bybit readiness check；`tv_basis` 只在 live 路径上切 venue，不重做策略 gate 核心语义。

**Tech Stack:** Python 3、repo-owned scripts、JSON state/review artifacts、Pytest、bash/ssh dry validation

---

## 0. 范围与边界

- change class 目标：`LIVE_EXECUTION_PATH`
- 本计划只做：
  - Bybit capability extension
  - Bybit signed spot/perp common
  - Bybit-first route selection
  - Bybit basis live adapter
  - readiness check 到 `canary-ready`
- 本计划不做：
  - 多 venue 自动路由器
  - 多策略
  - 跨交易所套利
  - 单次真实 canary 下单

### 0.1 Hard-switch 语义冻结

本计划保留字段名：

- `preferred_venue = bybit`

但第一阶段必须把它实现成 **hard-switch 字段**，不是“偏好/可 fallback”字段。  
执行与测试都必须冻结以下语义：

- `preferred_venue = bybit` 时，live candidate 只能是 Bybit
- Bybit blocked 时必须 fail-close
- 不允许隐式回退 Binance

### 0.2 Bybit-first execution owner 约束

为防止范围悄悄扩大，本计划明确：

- `tv_basis_arb_webhook.py` 拥有 venue dispatch 决策权
- `webhook -> bybit adapter` 是第一阶段唯一 live 执行路径
- `tv_basis_arb_executor.py` 继续保持 Binance-scoped 旧路径，不作为 Bybit venue dispatch owner
- 若 Bybit adapter 需要复用状态迁移，只能复用 `tv_basis_arb_state.py` / ledger 语义，不能把 Bybit live 分发逻辑塞回 `tv_basis_arb_executor.py`

## 1. 目标文件结构

### Create

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_live_common.py`
  - Bybit signed REST client（spot + perp）
  - 只做本项目所需最小接口

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_basis_live_adapter.py`
  - `tv_basis_btc_spot_perp_v1` 的 Bybit same-venue execution adapter

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_live_route_ready_check.py`
  - 唯一 readiness check 入口
  - 输出 `output/review/latest_bybit_live_route_ready_check.json`

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_live_common.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_basis_live_adapter.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_live_route_ready_check.py`

### Modify

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_venue_capability_artifact.py`
  - 扩展现有 single-writer，新增 `bybit` 条目

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_common.py`
  - 策略合同增加 `preferred_venue = bybit`

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py`
  - live 路径先 hard-switch 到 Bybit
  - 读取 `venues.bybit`

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_state.py`
  - 持久化 `execution_venue`

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_venue_capability_artifact.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_state.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`

### Source-owned artifacts

- `output/state/venue_capabilities.json`
- `output/review/latest_bybit_live_route_ready_check.json`

## 2. Task 1：把 strategy route hard-switch 到 Bybit

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_common.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_gate.py`

- [ ] **Step 1: 写 strategy contract failing test**

```python
def test_strategy_config_sets_preferred_venue_to_bybit() -> None:
    common = load_strategy_definition("tv_basis_btc_spot_perp_v1")
    assert common["preferred_venue"] == "bybit"
```

- [ ] **Step 2: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_gate.py -k preferred_venue
```

Expected: FAIL

- [ ] **Step 3: 最小修改 strategy contract**

要求：

- 为 `tv_basis_btc_spot_perp_v1` 增加：
  - `preferred_venue = bybit`
- 不引入 fallback 逻辑

- [ ] **Step 4: 跑测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_gate.py -k preferred_venue
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_common.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_gate.py
git commit -m "feat(live): set tv basis preferred venue to bybit"
```

## 3. Task 2：扩展 single-writer，把 Bybit 写进 `venue_capabilities.json`

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_venue_capability_artifact.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_live_common.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_venue_capability_artifact.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_live_common.py`

### Task 2 合同前置说明：`--venue` 的允许语义

single-writer 不意味着一定“每次都 probe 全部 venue”，但必须保证输出契约稳定。  
本计划固定：

- 继续使用同一个 writer：
  - `build_venue_capability_artifact.py`
- 若传 `--venue=bybit`，表示：
  - **至少**刷新 `bybit`
  - 对未 probe venue 不能静默删除
- 未 probe venue 的来源固定为：
  - 先读取既有 `output/state/venue_capabilities.json`
  - 保留其他 venue 现有条目
  - 仅更新本次 probe 的 venue
- 若既有 artifact 不存在，则未 probe venue 不强行伪造，但输出 schema 仍必须合法
- freshness / TTL / fail-close 规则仍由同一个 single-schema 合同约束，不能因 `--venue` 分支而出现不同判定口径

- [ ] **Step 1: 写 Bybit capability failing test**

```python
def test_build_venue_capability_artifact_adds_bybit_entry() -> None:
    payload = build_venue_capability_payload(...)
    assert "bybit" in payload["venues"]
```

- [ ] **Step 2: 写 Bybit futures blocked failing test**

```python
def test_bybit_capability_marks_futures_blocked_when_trade_permission_missing() -> None:
    venue = build_bybit_venue_payload(...)
    assert venue["status"] == "live_blocked"
    assert venue["futures_signed_trade_status"] == "blocked"
```

- [ ] **Step 3: 写 single-writer 入口语义 failing test**

```python
def test_single_writer_keeps_binance_and_adds_bybit() -> None:
    payload = build_venue_capability_payload(...)
    assert "binance" in payload["venues"]
    assert "bybit" in payload["venues"]
```

- [ ] **Step 4: 写 CLI / main merge 入口语义 failing test**

```python
def test_main_with_venue_bybit_preserves_existing_binance_entry() -> None:
    payload = run_main(...)
    assert "binance" in payload["venues"]
    assert "bybit" in payload["venues"]
```

- [ ] **Step 5: 写 CLI / main 无既有 artifact 入口语义 failing test**

```python
def test_main_with_venue_bybit_without_existing_artifact_keeps_schema_valid() -> None:
    payload = run_main(...)
    assert "bybit" in payload["venues"]
    assert "binance" not in payload["venues"]
```

- [ ] **Step 6: 写 Bybit live-ready failing test**

```python
def test_bybit_capability_can_mark_live_ready() -> None:
    venue = build_bybit_venue_payload(...)
    assert venue["status"] == "live_ready"
```

- [ ] **Step 7: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_build_venue_capability_artifact.py tests/test_bybit_live_common.py
```

Expected: FAIL

- [ ] **Step 8: 最小实现 `bybit_live_common.py`**

要求：

- Bybit signed spot/perp REST client
- 只实现当前能力/ready-check所需最小接口：
  - spot signed read
  - perp signed read
  - restrictions / account / exchange info 所需接口
- 继续遵守 timeout <= 5000ms / rate-limit

- [ ] **Step 9: 扩展 `build_venue_capability_artifact.py`**

要求：

- 继续使用同一个 single-writer
- 继续保持 `schema_version = 1`
- **若既有 artifact 已存在**，merge 后应同时保留：
  - `binance`
  - `bybit`
- **若既有 artifact 不存在**，只要求输出 schema 合法且至少包含本次 probed venue；不伪造未 probe venue
- CLI/入口语义也必须冻结：
  - 不允许新增第二个 Bybit-only writer
  - 若保留 `--venue` 或等价过滤参数，它只能控制 probe/refresh 范围，不能改变 single-schema 输出契约

- [ ] **Step 10: 跑测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_build_venue_capability_artifact.py tests/test_bybit_live_common.py
```

Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_venue_capability_artifact.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_live_common.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_venue_capability_artifact.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_live_common.py
git commit -m "feat(guard): extend venue capability artifact for bybit"
```

## 4. Task 3：Bybit same-venue live adapter

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_basis_live_adapter.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_basis_live_adapter.py`

- [ ] **Step 1: 写 entry 两腿 failing test**

```python
def test_bybit_adapter_executes_spot_buy_then_perp_short() -> None:
    result = adapter.execute_entry(...)
    assert result["status"] == "open_hedged"
```

- [ ] **Step 2: 写 recovery failing test**

```python
def test_bybit_adapter_enters_needs_recovery_when_second_leg_rejects() -> None:
    result = adapter.execute_entry(...)
    assert result["status"] == "needs_recovery"
```

- [ ] **Step 3: 写 exit failing test**

```python
def test_bybit_adapter_executes_perp_close_then_spot_sell() -> None:
    result = adapter.execute_exit(...)
    assert result["status"] == "closed"
```

- [ ] **Step 4: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_bybit_basis_live_adapter.py
```

Expected: FAIL

- [ ] **Step 5: 最小实现 adapter**

要求：

- 继续遵守：
  - `RunHalfhourMutex`
  - idempotency
  - partial fill / reject / transport ambiguity -> `needs_recovery`
- 不重写 `tv_basis` 状态机
- 只为 Bybit 同 venue `spot + perp` 提供最小执行外壳

- [ ] **Step 6: 跑测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_bybit_basis_live_adapter.py
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_basis_live_adapter.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_basis_live_adapter.py
git commit -m "feat(live): add bybit basis live adapter"
```

## 5. Task 4：让 `tv_basis` 走 Bybit-first live route，并闭合 `execution_venue` 承接链

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_state.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_basis_live_adapter.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_state.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_basis_live_adapter.py`

- [ ] **Step 1: 写 route selection failing test**

```python
def test_entry_check_consumes_bybit_capability_when_preferred_venue_is_bybit() -> None:
    result = handle_webhook(...)
    assert result["gate"]["venue"] == "bybit"
```

- [ ] **Step 2: 写 no-fallback failing test**

```python
def test_live_route_does_not_fallback_to_binance_when_bybit_blocked() -> None:
    result = handle_webhook(...)
    assert result["status"] == "gate_blocked"
    assert result["gate"]["venue"] == "bybit"
```

- [ ] **Step 3: 写 adapter dispatch failing test**

```python
def test_live_ready_bybit_route_dispatches_to_bybit_adapter() -> None:
    result = handle_webhook(...)
    assert result["status"] == "open_hedged"
    assert result["gate"]["venue"] == "bybit"
    assert bybit_adapter_called is True
```

- [ ] **Step 4: 写 execution_venue persistence failing test**

```python
def test_open_position_persists_execution_venue() -> None:
    position = ...
    assert position["execution_venue"] == "bybit"
```

- [ ] **Step 5: 写 exit/recovery dispatch failing test**

```python
def test_exit_and_recovery_follow_execution_venue() -> None:
    result = ...
    assert result["execution_venue"] == "bybit"
    assert bybit_exit_adapter_called is True
    assert route_not_selected_from_preferred_venue is True
```

场景约束：

- 该测试必须显式构造 `preferred_venue` 与 persisted `execution_venue` 可分离的情形
- 并机械证明 exit/recovery dispatch 只读取 source-owned `execution_venue`

- [ ] **Step 6: 写 recovery artifact persistence failing test**

```python
def test_recovery_artifact_persists_execution_venue() -> None:
    recovery = ...
    assert recovery["execution_venue"] == "bybit"
```

- [ ] **Step 7: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_webhook.py tests/test_tv_basis_arb_state.py tests/test_bybit_basis_live_adapter.py -k 'bybit or execution_venue or dispatch'
```

Expected: FAIL

- [ ] **Step 8: 最小实现 route switch**

要求：

- live candidate 只按 `preferred_venue = bybit`
- 不做 Binance -> Bybit 隐式 fallback
- venue dispatch owner 明确为 `tv_basis_arb_webhook.py` 的 live entry route
- `tv_basis_arb_webhook.py` 也负责 exit/recovery 的 venue dispatch resolution
- `execution_venue` 必须写入 state / artifact
- `recovery` artifact 也必须持久化 `execution_venue`
- exit / recovery 只能从 source-owned `execution_venue` resolve 正确 adapter

- [ ] **Step 9: 跑测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_webhook.py tests/test_tv_basis_arb_state.py tests/test_bybit_basis_live_adapter.py -k 'bybit or execution_venue or dispatch'
```

Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_basis_live_adapter.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_state.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_basis_live_adapter.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_state.py
git commit -m "fix(live): route tv basis live execution to bybit"
```

## 6. Task 5：Bybit readiness check 到 `canary-ready`

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_live_route_ready_check.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_live_route_ready_check.py`

- [ ] **Step 1: 写 readiness artifact failing test**

```python
def test_ready_check_writes_latest_artifact() -> None:
    payload = run_ready_check(...)
    assert payload["status"] in {"canary_ready", "blocked"}
```

- [ ] **Step 2: 写 blocked failing test**

```python
def test_ready_check_blocks_when_bybit_capability_not_ready() -> None:
    payload = run_ready_check(...)
    assert payload["status"] == "blocked"
```

- [ ] **Step 3: 写 canary-ready failing test**

```python
def test_ready_check_can_mark_canary_ready() -> None:
    payload = run_ready_check(...)
    assert payload["ok"] is True
    assert payload["status"] == "canary_ready"
```

- [ ] **Step 4: 写完整 required_checks failing test**

```python
def test_ready_check_marks_all_required_checks_before_canary_ready() -> None:
    payload = run_ready_check(...)
    assert payload["required_checks"]["venue_capability_ready"] is True
    assert payload["required_checks"]["route_selected_bybit"] is True
    assert payload["required_checks"]["baseqty_budget_contract_ready"] is True
    assert payload["required_checks"]["no_active_recovery"] is True
    assert payload["required_checks"]["account_ready"] is True
```

- [ ] **Step 5: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_bybit_live_route_ready_check.py
```

Expected: FAIL

- [ ] **Step 6: 最小实现 readiness check**

要求：

- 唯一入口：`system/scripts/bybit_live_route_ready_check.py`
- 唯一 latest artifact：
  - `output/review/latest_bybit_live_route_ready_check.json`
- 通过字段至少包含：
  - `venue_capability_ready`
  - `route_selected_bybit`
  - `baseqty_budget_contract_ready`
  - `no_active_recovery`
  - `account_ready`

- [ ] **Step 7: 跑测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_bybit_live_route_ready_check.py
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/bybit_live_route_ready_check.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_bybit_live_route_ready_check.py
git commit -m "feat(guard): add bybit live route readiness check"
```

## 7. Task 6：文档同步 + 全量验证

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`

- [ ] **Step 1: 更新 runbook**

要求：

- 不再写成 Binance-only 路径
- 增加 `preferred_venue = bybit`
- 增加 Bybit capability / readiness / canary-ready 入口

- [ ] **Step 2: Commit runbook**

```bash
git add /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md
git commit -m "docs(live): align runbook to bybit-first live route"
```

- [ ] **Step 3: 跑本地 broad pytest**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q \
  tests/test_venue_capability_common.py \
  tests/test_build_venue_capability_artifact.py \
  tests/test_bybit_live_common.py \
  tests/test_bybit_basis_live_adapter.py \
  tests/test_bybit_live_route_ready_check.py \
  tests/test_tv_basis_arb_webhook.py \
  tests/test_tv_basis_arb_gate.py \
  tests/test_tv_basis_arb_state.py \
  tests/test_tv_basis_arb_executor.py \
  tests/test_binance_infra_canary.py \
  tests/test_binance_live_takeover.py
```

Expected: PASS

- [ ] **Step 4: 跑 dry acceptance**

要求：

- 不真实下单
- 验证 Bybit-first route 选择、capability、adapter 接线和 readiness artifact

- [ ] **Step 5: 跑真实 readiness check 并断言 latest artifact 为 `canary_ready`**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
python3 scripts/bybit_live_route_ready_check.py --output-root output
cat output/review/latest_bybit_live_route_ready_check.json
```

Expected:

- `ok = true`
- `status = canary_ready`
- `required_checks.venue_capability_ready = true`
- `required_checks.route_selected_bybit = true`
- `required_checks.baseqty_budget_contract_ready = true`
- `required_checks.no_active_recovery = true`
- `required_checks.account_ready = true`

- [ ] **Step 6: 交付前检查工作树干净**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
git status --short
```

Expected: no uncommitted changes

## 8. 交付判定

以下全部满足才算完成：

1. Bybit 被接入到现有 `venue_capabilities.json` single-writer 合同
2. `tv_basis_btc_spot_perp_v1` live route 可 hard-switch 到 Bybit
3. state/recovery 可持久化 `execution_venue = bybit`
4. 有最小 Bybit same-venue live adapter
5. 有唯一 readiness check 和 `latest_bybit_live_route_ready_check.json`
6. local tests / dry acceptance 全绿
7. `output/review/latest_bybit_live_route_ready_check.json` 是唯一 `canary-ready` 判定来源，且其状态为 `canary_ready`
