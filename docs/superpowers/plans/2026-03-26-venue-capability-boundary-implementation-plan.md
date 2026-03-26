# Venue Capability Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fenlie 增加 source-owned 的 venue capability boundary，冻结 Binance 合约实盘线的误放行风险，并让 `tv_basis_btc_spot_perp_v1` 在 venue/account capability 缺失时 fail-close 为 `live_blocked`。

**Architecture:** 采用“单一 writer + 多处只读消费”的最小方案：新增一个统一的 capability schema/helper，再由单独的 `build_venue_capability_artifact.py` 写出 `output/state/venue_capabilities.json`。`tv_basis_arb_webhook.py` 只在真实 live 路径（未注入 fake clients）读取该 artifact 并做前置阻断，不改动现有 gate/executor/state 主链的业务逻辑。

**Tech Stack:** Python 3、repo-owned scripts、JSON state artifacts、Pytest、bash/ssh read-only diagnostics

---

## 0. 范围与边界

- change class 目标：`LIVE_GUARD_ONLY`
- 本计划只做：
  - 统一 capability schema / freshness / fail-closed 规则
  - Binance capability artifact writer
  - `tv_basis_btc_spot_perp_v1` 对 capability artifact 的消费
  - runbook 文案同步
- 本计划不做：
  - 新的 live connector
  - 替代 venue 下单适配
  - 多 venue 路由器
  - 自动修复交易所权限
  - 真实下单 acceptance

### 0.1 status 命名约束（必须写死）

为避免实现者把三套状态混在一起，本计划固定如下：

1. **artifact 的 `<venue>.status`**
   - 只表示 venue route verdict
   - 只允许：
     - `dry_only`
     - `live_blocked`
     - `live_ready`
     - `unknown`

2. **capability 字段**
   - 只表示单个能力状态
   - 例如：
     - `spot_signed_trade_status`
     - `futures_signed_trade_status`
   - 只允许：
     - `ready`
     - `blocked`
     - `unknown`

3. **webhook 顶层 `result["status"]`**
   - 保持现有结果命名空间，不承担 venue route verdict
   - capability 被阻断时，不新增新的顶层状态枚举；统一继续返回：
     - `gate_blocked`
   - 具体 route verdict 通过 gate artifact 字段表达：
     - `live_route_status`
     - `live_route_reason`

## 1. 目标文件结构

### Create

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/venue_capability_common.py`
  - capability schema/defaults
  - freshness / stale 计算
  - consumer-side fail-closed 判定 helper

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_venue_capability_artifact.py`
  - 单一 writer
  - Binance read-only probe
  - 写 `output/state/venue_capabilities.json`

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_venue_capability_common.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_venue_capability_artifact.py`

### Modify

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py`
  - 在真实 live 路径前读取 capability artifact
  - blocked / missing / stale / unknown -> `live_blocked`

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py`
  - 增加 capability artifact 缺失/blocked/stale 的 gate 测试

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`
  - 增加 `dry_only / live_blocked / live_ready` 说明
  - 增加 `venue_capabilities.json` 入口

### Source-owned artifact

- `output/state/venue_capabilities.json`

## 2. Task 1：先落 capability schema/helper

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/venue_capability_common.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_venue_capability_common.py`

- [ ] **Step 1: 写缺失 artifact 的 failing test**

```python
def test_missing_capability_artifact_maps_to_live_blocked() -> None:
    result = load_and_evaluate_capability(...)
    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_missing"
```

- [ ] **Step 2: 写 stale artifact 的 failing test**

```python
def test_stale_capability_artifact_maps_to_live_blocked() -> None:
    result = load_and_evaluate_capability(...)
    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_stale"
```

- [ ] **Step 3: 写 incomplete schema 的 failing test**

```python
def test_incomplete_capability_artifact_maps_to_live_blocked() -> None:
    result = load_and_evaluate_capability(...)
    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_incomplete"
```

- [ ] **Step 4: 写 ready capability 的 failing test**

```python
def test_ready_capability_allows_live_ready_candidate() -> None:
    result = load_and_evaluate_capability(...)
    assert result["live_route_status"] == "live_ready"
```

- [ ] **Step 5: 写 unknown / 非法时间戳的 failing test**

```python
def test_unknown_capability_maps_to_live_blocked() -> None:
    result = load_and_evaluate_capability(...)
    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_unknown"

def test_invalid_checked_at_utc_maps_to_live_blocked() -> None:
    result = load_and_evaluate_capability(...)
    assert result["live_route_status"] == "live_blocked"
    assert result["live_route_reason"] == "venue_capability_stale"
```

- [ ] **Step 6: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_venue_capability_common.py
```

Expected: FAIL

- [ ] **Step 7: 在 `venue_capability_common.py` 最小实现 helper**

要求：

- schema 固定字段：
  - `schema_version`
  - `venues`
  - `<venue>.checked_at_utc`
  - `<venue>.status`
  - `spot_signed_read_status`
  - `spot_signed_trade_status`
  - `futures_signed_read_status`
  - `futures_signed_trade_status`
  - `blockers`
- `<venue>.status` 只允许：
  - `dry_only`
  - `live_blocked`
  - `live_ready`
  - `unknown`
- capability `*_status` 只允许：
  - `ready`
  - `blocked`
  - `unknown`
- freshness contract：`max_age_seconds = 900`
- `unknown_stale` 只能是 consumer-side reason，不写回 source schema

- [ ] **Step 8: 跑测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_venue_capability_common.py
```

Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/venue_capability_common.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_venue_capability_common.py
git commit -m "feat(guard): add venue capability schema helper"
```

## 3. Task 2：新增单一 writer `build_venue_capability_artifact.py`

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_venue_capability_artifact.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_venue_capability_artifact.py`
- Modify if needed: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/binance_live_common.py`

- [ ] **Step 1: 写 Binance futures blocked 的 failing test**

```python
def test_build_binance_capability_artifact_marks_futures_blocked_when_enable_futures_false() -> None:
    payload = run_probe(...)
    venue = payload["venues"]["binance"]
    assert venue["status"] == "live_blocked"
    assert venue["futures_signed_trade_status"] == "blocked"
    assert "enableFutures=false" in venue["blockers"]
```

- [ ] **Step 2: 写 ready spot + blocked futures 的 failing test**

```python
def test_build_binance_capability_artifact_keeps_spot_ready_when_spot_signed_succeeds() -> None:
    payload = run_probe(...)
    venue = payload["venues"]["binance"]
    assert venue["spot_signed_read_status"] == "ready"
    assert venue["spot_signed_trade_status"] == "ready"
```

- [ ] **Step 3: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_build_venue_capability_artifact.py
```

Expected: FAIL

- [ ] **Step 4: 最小实现 writer 脚本**

要求：

- 只读 probe Binance：
  - spot signed read/account
  - `/sapi/v1/account/apiRestrictions`
  - futures signed read/account
- writer 只写：
  - `output/state/venue_capabilities.json`
- 单一 writer：不得让别的脚本直接写同一路径
- 不下单

- [ ] **Step 5: 如果需要，最小补齐 helper/导入**

只允许补：

- 现有 client 组合导入
- `/sapi/v1/account/apiRestrictions` 的最小只读 probe helper
- 少量纯只读 parsing helper

明确禁止：

- 新增新的 Binance client class
- 改 order path
- 改 `binance_live_takeover.py` 的 takeover contract
- 做 connector abstraction

- [ ] **Step 6: 跑测试确认转绿**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_build_venue_capability_artifact.py
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_venue_capability_artifact.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_venue_capability_artifact.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/binance_live_common.py
git commit -m "feat(guard): add venue capability artifact writer"
```

## 4. Task 3：让 `tv_basis_arb_webhook.py` 消费 capability artifact

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py`

- [ ] **Step 1: 写 capability artifact 缺失时的 failing test**

```python
def test_entry_check_returns_live_blocked_when_capability_artifact_missing(tmp_path: Path) -> None:
    result = handle_webhook(..., output_root=tmp_path)
    assert result["status"] == "gate_blocked"
    assert result["gate"]["live_route_status"] == "live_blocked"
    assert result["gate"]["live_route_reason"] == "venue_capability_missing"
```

- [ ] **Step 2: 写 futures blocked 时的 failing test**

```python
def test_entry_check_returns_live_blocked_when_futures_trade_status_blocked(tmp_path: Path) -> None:
    result = handle_webhook(...)
    assert result["status"] == "gate_blocked"
    assert result["gate"]["live_route_status"] == "live_blocked"
    assert result["gate"]["live_route_reason"] == "venue_capability"
```

- [ ] **Step 3: 写 stale artifact 的 failing test**

```python
def test_entry_check_returns_live_blocked_when_capability_artifact_is_stale(tmp_path: Path) -> None:
    result = handle_webhook(...)
    assert result["status"] == "gate_blocked"
    assert result["gate"]["live_route_status"] == "live_blocked"
    assert result["gate"]["live_route_reason"] == "venue_capability_stale"
```

- [ ] **Step 4: 写 fake clients 注入时不消费 capability 的 failing test**

```python
def test_entry_check_with_fake_clients_bypasses_live_capability_block(tmp_path: Path) -> None:
    result = handle_webhook(..., spot_client=fake_spot, perp_client=fake_perp)
    assert result["status"] == "open_hedged"
```

- [ ] **Step 5: 跑测试确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_webhook.py -k 'live_blocked or capability'
```

Expected: FAIL

- [ ] **Step 6: 最小实现 capability gate**

要求：

- 只有在 `spot_client is None and perp_client is None` 时才走 live capability block
- fake clients / dry harness 不受影响
- blocked/missing/stale/incomplete/unknown 都必须 fail-close 到 `live_blocked`
- `result["status"]` 继续保持 `gate_blocked`
- gate artifact 追加：
  - `live_route_status`
  - `live_route_reason`
  - `venue`
  - `venue_blockers`

- [ ] **Step 7: 跑 webhook 全量测试**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_tv_basis_arb_webhook.py
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/tv_basis_arb_webhook.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_tv_basis_arb_webhook.py
git commit -m "fix(guard): block tv basis live route on venue capability"
```

## 5. Task 4：同步 runbook 文案

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md`

- [ ] **Step 1: 补 capability artifact 入口文案**

必须写明：

- `output/state/venue_capabilities.json` 是 source-of-truth
- Binance 当前可被裁决为 `live_blocked`

- [ ] **Step 2: 补状态语义文案**

必须区分：

- `dry_only`
- `live_blocked`
- `live_ready`

- [ ] **Step 3: 文档自检**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
rg -n "venue_capabilities|live_blocked|dry_only|live_ready|enableFutures=false" \
  system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md \
  system/scripts/build_venue_capability_artifact.py \
  system/scripts/tv_basis_arb_webhook.py
```

Expected: 新文案都能检出

- [ ] **Step 4: Commit**

```bash
git add /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/TRADINGVIEW_BASIS_ARB_RUNBOOK.md
git commit -m "docs(guard): document venue capability boundary"
```

## 6. Task 5：全量回归与远端部署/安全验证

**Files:**
- Verify only

- [ ] **Step 1: 跑本地 pytest**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q \
  tests/test_venue_capability_common.py \
  tests/test_build_venue_capability_artifact.py \
  tests/test_tv_basis_arb_webhook.py \
  tests/test_tv_basis_arb_gate.py \
  tests/test_tv_basis_arb_state.py \
  tests/test_tv_basis_arb_executor.py \
  tests/test_binance_infra_canary.py \
  tests/test_binance_live_takeover.py
```

Expected: PASS

- [ ] **Step 2: 定点同步远端脚本（部署步骤，不属于只读验证）**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
rsync -az --itemize-changes -e 'ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
  system/scripts/venue_capability_common.py \
  system/scripts/build_venue_capability_artifact.py \
  system/scripts/tv_basis_arb_webhook.py \
  ubuntu@43.153.148.242:/home/ubuntu/openclaw-system/scripts/
```

Expected: changed files synced, no error

- [ ] **Step 3: 远端生成临时 capability artifact（写 `/tmp`，不碰正式 `output/`）**

Run:

```bash
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=yes ubuntu@43.153.148.242 \
  'cd /home/ubuntu/openclaw-system && PYTHONPATH=scripts python3 scripts/build_venue_capability_artifact.py --venue binance --output-root /tmp/venue_capability_probe'
```

Expected: 写出 `/tmp/venue_capability_probe/state/venue_capabilities.json`

- [ ] **Step 4: 远端验证 `tv_basis` 被 capability fail-close（使用临时目录）**

Run:

```bash
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new -o BatchMode=yes ubuntu@43.153.148.242 <<'REMOTE'
set -euo pipefail
cd /home/ubuntu/openclaw-system
PYTHONPATH=scripts python3 - <<'PY'
import json
from pathlib import Path
import tv_basis_arb_webhook

result = tv_basis_arb_webhook.handle_webhook(
    {
        "strategy_id": "tv_basis_btc_spot_perp_v1",
        "symbol": "BTCUSDT",
        "event_type": "entry_check",
        "tv_timestamp": "2026-03-26T12:30:00Z",
    },
    output_root=Path("/tmp/venue_capability_probe"),
)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
REMOTE
```

Expected:

- `status = gate_blocked`
- `gate.live_route_status = live_blocked`
- `gate.live_route_reason = venue_capability`
- 不进入真实 executor

- [ ] **Step 5: 结束前确认工作树干净**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
git status --short
```

Expected: no uncommitted changes

## 7. 交付判定

以下全部满足才算完成：

1. 有统一 `venue capability` schema/helper
2. 有单一 writer 生成 `output/state/venue_capabilities.json`
3. Binance 当前状态可稳定写成：
   - `spot ready`
   - `futures blocked`
   - blocker=`enableFutures=false`
4. `tv_basis_btc_spot_perp_v1` 在无能力时返回：
   - `result.status = gate_blocked`
   - `gate.live_route_status = live_blocked`
5. fake clients / dry harness 不被 capability block 误伤
6. runbook 明确区分 `dry_only / live_blocked / live_ready`
7. 本地回归通过
8. 远端部署 + 临时目录验证通过
