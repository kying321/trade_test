# Fenlie Infra Canary Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fenlie 增加一条独立于策略 ticket 的真实云端 `infra canary` 执行路径，用于验证 `BTCUSDT spot 5 USDT` 往返回平链路，同时保留 mutex、timeout、idempotency、budget、panic 审计边界。

**Architecture:** 采用独立 `binance_infra_canary.py` actor，而不是把逻辑塞进现有 `binance_live_takeover.py`。为避免复制已有 Binance client / mutex / panic plumbing，先抽出一个最小共享模块，再让 `binance_live_takeover.py` 和 `binance_infra_canary.py` 共用；`openclaw_cloud_bridge.sh` 只新增 `infra-canary-probe / run / autopilot` 三个动作，不改变现有 strategy-first live-takeover 语义。

**Tech Stack:** Python 3、repo-owned scripts、Pytest、bash bridge script、JSON artifacts / state ledgers

---

## 0. 约束

- change class 目标：`LIVE_EXECUTION_PATH`
- 现有 `signal_to_order_tickets`、`remote_ticket_actionability_state`、strategy confidence/convexity/route gate 不得作为 `infra canary` 准入条件
- 必须继续使用 `run-halfhour-pulse` mutex
- 外部请求 timeout 必须 `<=5000ms`
- 自动触发允许，但必须受 `daily_budget_cap_usdt=20`、idempotency、panic / source ambiguity guard 约束
- 成功允许产生 dust，但必须完整记录，不得让 consumer 将其解读为 strategy 持仓
- 当前计划只覆盖 `infra canary`；TradingView / webhook / 套利是后续独立子项目

## 1. 目标文件结构

### Create

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py`
  - 承载共享的 Binance live execution helper
  - 包括：`TokenBucket`、`RunHalfhourMutex`、`panic_close_all`、`write_json`、`read_json`、`load_list_ledger`、`save_list_ledger`、`resolve_binance_credentials`、`BinanceSpotClient`、最小 numeric helpers

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_infra_canary.py`
  - 独立 `infra canary` actor
  - 提供 `probe / run / autopilot-ready-check` 所需 JSON 输出

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_binance_infra_canary.py`
  - 覆盖 budget、idempotency、round-trip、dust、panic/ambiguity 分级

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_openclaw_cloud_bridge_script.py`
  - 覆盖新增 `infra-canary-probe / run / autopilot` shell actions

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/INFRA_CANARY_CONTRACT.md`
  - 运行手册版本的 contract 摘要

### Modify

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_takeover.py`
  - 改为导入共享 helper，不改变现有 strategy-first 语义

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/openclaw_cloud_bridge.sh`
  - 新增 `infra-canary-probe`
  - 新增 `infra-canary-run`
  - 新增 `infra-canary-autopilot`
  - 新增对应 env 默认值与 usage 文案

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_binance_live_takeover.py`
  - 如共享模块抽出导致 import/behavior 改变，则补回归断言

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md`
  - 补 `infra canary` 用法、边界、预算、dust 解释

### State / Artifact Contract（由实现落盘）

- `output/state/infra_canary_idempotency.json`
- `output/state/infra_canary_budget.json`
- `output/review/*_infra_canary_run.json`
- `output/review/*_infra_canary_run.md`
- `output/review/*_infra_canary_run_checksum.json`

## 2. Task 1：抽出共享 live execution primitives

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_takeover.py`
- Test: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_binance_live_takeover.py`

- [ ] **Step 1: 为共享模块抽取写失败回归测试**

在 `test_binance_live_takeover.py` 中新增/补强最小回归断言，确认以下行为在抽取前后不变：

```python
def test_calc_canary_quantity_respects_min_notional_and_step() -> None:
    qty = mod.calc_canary_quantity(
        quote_usdt=5.0,
        price=73_000.0,
        step_size=0.001,
        min_qty=0.001,
        min_notional=100.0,
    )
    assert qty >= 0.001
```

还要覆盖：
- `activate_config_for_live` 仍只改原约定字段
- spot precheck 仍在余额不足时阻断
- `allow_live_order_false` 仍保持现有行为

- [ ] **Step 2: 跑回归测试，确认当前基线为绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_binance_live_takeover.py
```

Expected: PASS（绿基线），便于后续抽取时识别回归。

- [ ] **Step 3: 抽出共享 helper 到 `binance_live_common.py`**

最小抽取范围：
- `TokenBucket`
- `RunHalfhourMutex`
- `panic_close_all`
- `write_json` / `read_json`
- `to_float` / `to_int`
- `load_list_ledger` / `save_list_ledger`
- `resolve_binance_credentials`
- `BinanceSpotClient`

`binance_live_takeover.py` 继续保留：
- strategy signal selection
- current canary ticket path
- takeover-specific summary fields

- [ ] **Step 4: 改写 `binance_live_takeover.py` 导入共享 helper**

要求：
- 现有 CLI 参数不变
- 现有 JSON 结构不变
- 现有 tests 不新增预期差异

- [ ] **Step 5: 重新跑回归测试**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_binance_live_takeover.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_common.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_live_takeover.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_binance_live_takeover.py
git commit -m "refactor(live): extract shared binance execution helpers"
```

## 3. Task 2：实现独立 `binance_infra_canary.py`

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_infra_canary.py`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_binance_infra_canary.py`

- [ ] **Step 1: 先写失败测试，锁定 contract 边界**

新增测试至少覆盖：

```python
def test_probe_ignores_strategy_ticket_authority() -> None:
    payload = run_probe(...)
    assert payload["gate"]["reads_strategy_ticket"] is False

def test_run_blocks_when_daily_budget_exceeded() -> None:
    payload = run_canary(...)
    assert payload["ok"] is False
    assert payload["reason"] == "daily_budget_exceeded"

def test_run_round_trip_succeeds_with_dust_allowed() -> None:
    payload = run_canary(...)
    assert payload["ok"] is True
    assert payload["round_trip"]["buy_executed"] is True
    assert payload["round_trip"]["sell_executed"] is True
    assert payload["dust"]["allowed"] is True
```

还要覆盖：
- idempotency skip
- account-ready false
- transport ambiguity => panic class failure
- no open source ambiguity => success

- [ ] **Step 2: 跑新测试，确认先红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_binance_infra_canary.py
```

Expected: FAIL，提示新脚本或关键函数尚未实现。

- [ ] **Step 3: 写最小实现**

`binance_infra_canary.py` 最少需要实现：

- CLI 参数：
  - `--market`（默认 `spot`）
  - `--symbol`（默认 `BTCUSDT`）
  - `--quote-usdt`（默认 `5`）
  - `--daily-budget-cap-usdt`（默认 `20`）
  - `--allow-dust`
  - `--mode probe|run|autopilot-check`
  - `--output-root`
  - `--config`
  - `--allow-daemon-env-fallback`
  - `--skip-mutex`

- 独立 state：
  - `infra_canary_idempotency.json`
  - `infra_canary_budget.json`

- 核心流程：
  - `probe`: 只检查账户、凭证、panic、budget、ledger
  - `run`: buy -> fill -> sell -> fill -> dust summary
  - `autopilot-check`: 只返回“是否允许自动触发”

- [ ] **Step 4: 成功/失败分级写进 JSON artifact**

最少输出字段：

```python
payload = {
    "ok": True,
    "mode": "round_trip",
    "symbol": "BTCUSDT",
    "market": "spot",
    "quote_usdt": 5.0,
    "budget": {...},
    "idempotency": {...},
    "round_trip": {
        "buy_executed": True,
        "sell_executed": True,
    },
    "dust": {
        "allowed": True,
        "residual_asset": "BTC",
        "residual_qty": 0.000001,
    },
}
```

- [ ] **Step 5: 跑新测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_binance_infra_canary.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/binance_infra_canary.py \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_binance_infra_canary.py
git commit -m "feat(live): add independent infra canary actor"
```

## 4. Task 3：桥接 shell 动作与自动触发入口

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/openclaw_cloud_bridge.sh`
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_openclaw_cloud_bridge_script.py`

- [ ] **Step 1: 为 shell action dispatch 写失败测试**

参考 `test_auto_git_sync_script.py` 的 fake executable 模式，新增 shell 测试：

```python
def test_usage_lists_infra_canary_actions() -> None:
    proc = subprocess.run([...], capture_output=True, text=True)
    assert "infra-canary-probe" in proc.stdout
    assert "infra-canary-run" in proc.stdout
    assert "infra-canary-autopilot" in proc.stdout
```

以及：
- `infra-canary-probe` 生成正确远端命令
- `infra-canary-run` 带 `--mode run`
- `infra-canary-autopilot` 先跑 `--mode autopilot-check`
- 仅当 `autopilot-check.ok == true` 时才允许继续 `--mode run`
- 当 budget 用尽、idempotency 命中、panic 未清或账户未 ready 时，`infra-canary-autopilot` 必须 graceful skip，且不得触发真实 `run`

- [ ] **Step 2: 跑 shell 测试先红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_openclaw_cloud_bridge_script.py
```

Expected: FAIL

- [ ] **Step 3: 修改 `openclaw_cloud_bridge.sh`**

新增：
- usage 文案
- env 默认值：
  - `INFRA_CANARY_SYMBOL=BTCUSDT`
  - `INFRA_CANARY_QUOTE_USDT=5`
  - `INFRA_CANARY_DAILY_BUDGET_CAP_USDT=20`
  - `INFRA_CANARY_ALLOW_DUST=true`
- 三个 action：
  - `infra-canary-probe`
  - `infra-canary-run`
  - `infra-canary-autopilot`

要求：
- 仍走 5 秒 SSH connect timeout
- 不破坏既有 `live-takeover-*` 动作
- `infra-canary-autopilot` 必须消费 `binance_infra_canary.py --mode autopilot-check` 的结构化输出
- `autopilot-check.ok=false` 时只返回 skip / blocked 结果，不得下单

- [ ] **Step 4: 跑 shell 测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_openclaw_cloud_bridge_script.py
```

Expected: PASS

- [ ] **Step 5: 为 autopilot guard 单独补一条 shell 回归**

新增测试目标：

```python
def test_infra_canary_autopilot_skips_run_when_autopilot_check_blocks() -> None:
    proc = subprocess.run([...], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "infra-canary-run" not in invoked_remote_commands
    assert "blocked" in proc.stdout or "skip" in proc.stdout
```

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q tests/test_openclaw_cloud_bridge_script.py -k autopilot
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/openclaw_cloud_bridge.sh \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_openclaw_cloud_bridge_script.py
git commit -m "feat(live): add infra canary bridge actions"
```

## 5. Task 4：运行手册与 operator-facing 文档

**Files:**
- Create: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/INFRA_CANARY_CONTRACT.md`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md`

- [ ] **Step 1: 写 runtime contract 摘要文档**

`INFRA_CANARY_CONTRACT.md` 必须覆盖：
- authority 边界
- success criteria
- failure severity
- budget / dust / idempotency
- 与 strategy canary 的区别

- [ ] **Step 2: 更新 cloud bridge runbook**

在 `OPENCLAW_CLOUD_BRIDGE.md` 增加：
- `infra-canary-probe`
- `infra-canary-run`
- `infra-canary-autopilot`
- 说明这不是策略 ready 信号
- 明确写出：`infra-canary-autopilot` 只有在 `autopilot-check.ok=true` 时才允许继续 `run`

- [ ] **Step 3: 可选补一条 docs smoke**

如有现成 docs lint/grep 校验，可加最小检查；否则至少人工确认文档命令、路径、参数与脚本一致。

- [ ] **Step 4: Commit**

```bash
git add \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/INFRA_CANARY_CONTRACT.md \
  /Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md
git commit -m "docs(live): document infra canary contract and bridge usage"
```

## 6. Task 5：最小验证集

**Files:**
- Verify only

- [ ] **Step 1: 跑 Python 单测集**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
pytest -q \
  tests/test_binance_live_takeover.py \
  tests/test_binance_infra_canary.py \
  tests/test_openclaw_cloud_bridge_script.py
```

Expected: PASS

- [ ] **Step 2: 跑 config 验证**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
PYTHONPATH=src python3 -m lie_engine.cli --config config.yaml validate-config
```

Expected: `ok: true`

- [ ] **Step 3: 本地 dry contract smoke（不下单）**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
python3 scripts/binance_infra_canary.py \
  --mode probe \
  --config config.yaml \
  --output-root output
```

Expected:
- 返回结构化 JSON
- 明确 `reads_strategy_ticket = false` 或等价语义
- 不生成订单

- [ ] **Step 4: bridge dry smoke（不下单）**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
bash scripts/openclaw_cloud_bridge.sh infra-canary-probe
```

Expected:
- SSH / bridge 路径可用
- 返回 probe JSON
- 不触发真实订单

- [ ] **Step 5: autopilot guard smoke（应 graceful skip 或 check-only）**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
bash scripts/openclaw_cloud_bridge.sh infra-canary-autopilot
```

Expected:
- 先消费 `autopilot-check`
- 若 `autopilot-check.ok=false`，则只返回 skip / blocked JSON
- 不触发真实 `run`

- [ ] **Step 6: 代码 review / self-audit**

手工核对：
- `signal_to_order_tickets` 未被读作准入条件
- `run-halfhour-pulse` mutex 仍存在
- timeout 仍 `<=5000ms`
- idempotency / budget ledger 是 source-owned
- autopilot 不会绕过 budget / panic / ambiguity

## 7. Task 6：真实云端验收（单次）

**Files:**
- Verify only

- [ ] **Step 1: 真实云端 probe**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
bash scripts/openclaw_cloud_bridge.sh infra-canary-probe
```

Expected:
- account ready
- budget ready
- idempotency ready
- 不下单

- [ ] **Step 2: 真实云端 run（单次）**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
bash scripts/openclaw_cloud_bridge.sh infra-canary-run
```

Expected:
- buy ack/fill
- sell ack/fill
- 允许 dust
- 生成独立 `infra_canary_run` artifact

- [ ] **Step 3: autopilot smoke**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system
bash scripts/openclaw_cloud_bridge.sh infra-canary-autopilot
```

Expected:
- 若当日预算已用完或 idempotency 已命中，应 graceful skip
- 不得重复打单

- [ ] **Step 4: Commit（若验收阶段需要同步 docs/fixtures/consumer 文案）**

```bash
git status --short
```

Expected: 除非验收暴露必须修复的问题，否则不追加代码改动。

## 8. 非目标 / 延后项

本计划不包含：
- TradingView / webhook 接入
- 多品种套利执行
- dashboard 新 metric / 新面板
- strategy ticket sizing contract 调整
- 将现有 `live-takeover-canary` 重命名或删除

这些应在 `infra canary` 实现稳定后，另起新 spec / 新 plan。

## 9. 交付判定

以下全部满足才算完成：

1. `binance_infra_canary.py` 可独立 `probe/run/autopilot-check`
2. `openclaw_cloud_bridge.sh` 可触发 `infra-canary-probe/run/autopilot`
3. 不读取 strategy ticket 作为准入 authority
4. 有独立 idempotency / budget / artifact
5. 单测通过
6. 单次真实云端 round-trip 可执行，并允许 dust
7. 文档已更新，明确“infra success != strategy ready”
