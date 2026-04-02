# Event Crisis Geostrategy Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前已闭合的事件量化主线之上，为 Fenlie 增加“大国博弈层 / 传导链 / safety margin”三层结构解释系统，并以最小风险顺序将解释层逐步压缩进现有 regime / asset shock map / live guard overlay。

**Architecture:** 保留现有 `event_intake -> event_regime_snapshot -> event_crisis_analogy -> event_asset_shock_map -> event_live_guard_overlay` 主线不变，新增三个解释层 artifacts：`event_game_state_snapshot`、`event_transmission_chain_map`、`event_safety_margin_snapshot`。分阶段按 `A -> B -> C -> D` 推进：先生成解释层，再让现有 regime/shock map 吸收，再将 safety margin 压缩进 overlay，最后做 dashboard/operator 展示增强。

**Tech Stack:** Python 3、repo-owned scripts、JSON review/state artifacts、Pytest、TypeScript dashboard read-model、现有 operator panel / dashboard snapshot / LIVE_GUARD_ONLY 接线

---

## 0. 范围与阶段边界

- 阶段 A：`RESEARCH_ONLY`
- 阶段 B：`RESEARCH_ONLY`
- 阶段 C：`LIVE_GUARD_ONLY`
- 阶段 D：`RESEARCH_ONLY`

### 0.1 当前主线基线（必须保持稳定）

当前基线 artifacts / 接线已经存在，扩展不得破坏：

- `latest_event_intake.json`
- `latest_event_regime_snapshot.json`
- `latest_event_crisis_analogy.json`
- `latest_event_asset_shock_map.json`
- `latest_event_crisis_operator_summary.json`
- `event_live_guard_overlay.json`
- dashboard snapshot / read-model / operator panel / guard 接线

### 0.2 新增 artifacts（冻结）

扩展目标 artifacts：

- `system/output/review/latest_event_game_state_snapshot.json`
- `system/output/review/latest_event_transmission_chain_map.json`
- `system/output/review/latest_event_safety_margin_snapshot.json`

### 0.3 执行 authority 边界

即便扩展后，执行层仍只允许直接读取：

- `system/output/state/event_live_guard_overlay.json`

guard / runtime / execution 不能直接读取：

- `event_game_state_snapshot`
- `event_transmission_chain_map`
- `event_safety_margin_snapshot`

## 1. 目标文件结构

### Create

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_game_state.py`
  - 大国主体、主战场、策略集、`game_state` 状态机
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_transmission.py`
  - 冲突轴到传导链映射、dominant chain / intensity / velocity
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_safety_margin.py`
  - liquidity/credit/energy/policy margin 与 hard boundaries 压缩
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_game_state.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_transmission.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_safety_margin.py`

### Modify

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_pipeline.py`
  - 吸收新解释层 artifacts 并压缩进 regime/shock/overlay
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_event_crisis_pipeline.py`
  - 单入口 runner 扩展为生成三类新 artifacts
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_pipeline.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_event_crisis_pipeline.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py`

## 2. Task A1：`event_game_state_snapshot` 最小骨架（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_game_state.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_game_state.py`

- [ ] **Step 1: 写 actor/state failing test**

```python
def test_build_event_game_state_snapshot_includes_required_actors_and_state() -> None:
    from lie_engine.research.event_game_state import build_event_game_state_snapshot

    payload = build_event_game_state_snapshot(event_rows=[...], market_inputs={...})
    actors = {row["actor"] for row in payload["actors"]}
    assert {"united_states", "china", "european_union", "russia", "opec_plus_gulf"} <= actors
    assert payload["game_state"] in {
        "stable_competition",
        "financial_pressure",
        "commodity_weaponization",
        "bloc_fragmentation",
        "systemic_repricing",
    }
```

- [ ] **Step 2: 写 contract failing test**

```python
def test_event_game_state_snapshot_contract_fields_are_stable() -> None:
    payload = build_event_game_state_snapshot(event_rows=[...], market_inputs={...})
    assert payload["generated_at_utc"].endswith("Z")
    assert 0.0 <= float(payload["confidence_score"]) <= 1.0
    assert 0.0 <= float(payload["systemic_escalation_probability"]) <= 1.0
    assert 0.0 <= float(payload["policy_relief_probability"]) <= 1.0
    assert isinstance(payload["dominant_conflict_axes"], list)
    assert isinstance(payload["dominant_transmission_axes"], list)
    assert payload["game_state"] in {
        "stable_competition",
        "financial_pressure",
        "commodity_weaponization",
        "bloc_fragmentation",
        "systemic_repricing",
    }
```

- [ ] **Step 3: 跑 tests 确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src pytest -q tests/test_event_game_state.py
```

Expected: FAIL

- [ ] **Step 4: 最小实现 game state 纯逻辑**

要求：

- 固定第一阶段主体集合与主战场
- 固定 `game_state` 五档状态机
- 输出字段符合扩展 spec 合同
- 明确锁住：
  - `generated_at_utc` + ISO8601 `Z`
  - 所有 `score/probability` 在 `[0.0, 1.0]`
  - 未知列表字段用空列表，不用 `null`
  - 状态字段使用冻结枚举
- 保持纯逻辑、无网络 I/O

- [ ] **Step 5: 跑 tests 确认转绿**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_game_state.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_game_state.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_game_state.py
git commit -m "feat(research): add event game state snapshot"
```

## 3. Task A2：`event_transmission_chain_map` 最小骨架（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_transmission.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_transmission.py`

- [ ] **Step 1: 写 dominant chain failing test**

```python
def test_build_event_transmission_chain_map_emits_dominant_chain() -> None:
    payload = build_event_transmission_chain_map(game_state_snapshot={...})
    assert payload["dominant_chain"] in {
        "usd_liquidity_chain",
        "financial_sanctions_chain",
        "energy_supply_chain",
        "shipping_supply_chain",
        "credit_intermediary_chain",
        "risk_off_deleveraging_chain",
    }
    assert payload["chains"]
```

- [ ] **Step 2: 写 contract failing test**

```python
def test_event_transmission_chain_map_contract_fields_are_stable() -> None:
    payload = build_event_transmission_chain_map(game_state_snapshot={...})
    assert payload["generated_at_utc"].endswith("Z")
    assert payload["dominant_chain"] in {
        "usd_liquidity_chain",
        "financial_sanctions_chain",
        "energy_supply_chain",
        "shipping_supply_chain",
        "credit_intermediary_chain",
        "risk_off_deleveraging_chain",
    }
    assert isinstance(payload["chains"], list)
    assert all(0.0 <= float(row["intensity_score"]) <= 1.0 for row in payload["chains"])
    assert all(0.0 <= float(row["velocity_score"]) <= 1.0 for row in payload["chains"])
    assert all(0.0 <= float(row["confidence_score"]) <= 1.0 for row in payload["chains"])
    assert all(row["status"] in {"watch", "active", "dominant"} for row in payload["chains"])
```

- [ ] **Step 3: 跑 tests 确认先红**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_transmission.py
```

Expected: FAIL

- [ ] **Step 4: 最小实现传导链映射**

要求：

- 固定 dominant conflict axes
- 固定六条传导链
- 输出 `intensity_score / velocity_score / confidence_score / status`
- 明确锁住：
  - `generated_at_utc` + ISO8601 `Z`
  - 所有 `score` 在 `[0.0, 1.0]`
  - `chains[]` 空时用空列表，不用 `null`
  - `status` 只允许 `watch|active|dominant`
- 保持纯逻辑、无网络 I/O

- [ ] **Step 5: 跑 tests 确认转绿**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_transmission.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_transmission.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_transmission.py
git commit -m "feat(research): add event transmission chain map"
```

## 4. Task B1：让现有 regime / shock map 吸收博弈层（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_pipeline.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_pipeline.py`

- [ ] **Step 1: 写 regime absorption failing test**

```python
def test_regime_snapshot_absorbs_game_state_and_transmission_inputs() -> None:
    payload = build_event_regime_snapshot(
        event_rows=[...],
        market_inputs={...},
        game_state_snapshot={...},
        transmission_chain_map={...},
    )
    assert payload["regime_state"] in {"sector_stress", "cross_asset_contagion", "systemic_risk"}
```

- [ ] **Step 2: 写 asset map absorption failing test**

```python
def test_asset_shock_map_absorbs_dominant_chain_at_artifact_level() -> None:
    payload = build_event_asset_shock_map(
        event_rows=[...],
        market_inputs={...},
        transmission_chain_map={...},
    )
    assert payload["dominant_chain"] in {
        "usd_liquidity_chain",
        "financial_sanctions_chain",
        "energy_supply_chain",
        "shipping_supply_chain",
        "credit_intermediary_chain",
        "risk_off_deleveraging_chain",
    }
```

- [ ] **Step 3: 跑 tests 确认先红**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_crisis_pipeline.py -k absorb
```

Expected: FAIL

- [ ] **Step 4: 最小实现压缩**

要求：

- 让 `event_regime_snapshot` 吸收 `game_state_snapshot / transmission_chain_map`
- 让 `event_asset_shock_map` 吸收 `dominant_chain`
- 保持输出字段不回退
- `dominant_chain` 放在 artifact-level，不放到每个资产 row 里

- [ ] **Step 5: 跑 tests 确认转绿**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_crisis_pipeline.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_pipeline.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_pipeline.py
git commit -m "feat(research): absorb geostrategy signals into event regime"
```

## 5. Task B2：`event_safety_margin_snapshot` 最小骨架（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_safety_margin.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_safety_margin.py`

- [ ] **Step 1: 写 safety margin failing test**

```python
def test_build_event_safety_margin_snapshot_outputs_margins_and_boundaries() -> None:
    payload = build_event_safety_margin_snapshot(
        game_state_snapshot={...},
        transmission_chain_map={...},
        regime_snapshot={...},
    )
    assert 0.0 <= float(payload["system_margin_score"]) <= 1.0
    assert "canary_hard_block" in payload["hard_boundaries"]
    assert "new_risk_hard_block" in payload["hard_boundaries"]
    assert "shadow_only_boundary" in payload["hard_boundaries"]
```

- [ ] **Step 2: 写 contract failing test**

```python
def test_event_safety_margin_snapshot_contract_fields_are_stable() -> None:
    payload = build_event_safety_margin_snapshot(
        game_state_snapshot={...},
        transmission_chain_map={...},
        regime_snapshot={...},
    )
    assert payload["generated_at_utc"].endswith("Z")
    assert 0.0 <= float(payload["liquidity_margin"]) <= 1.0
    assert 0.0 <= float(payload["credit_margin"]) <= 1.0
    assert 0.0 <= float(payload["energy_margin"]) <= 1.0
    assert 0.0 <= float(payload["policy_margin"]) <= 1.0
    assert 0.0 <= float(payload["system_margin_score"]) <= 1.0
    assert isinstance(payload["boundary_reasons"], list)
    assert set(payload["hard_boundaries"].keys()) == {
        "canary_hard_block",
        "new_risk_hard_block",
        "shadow_only_boundary",
    }
```

- [ ] **Step 3: 跑 tests 确认先红**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_safety_margin.py
```

Expected: FAIL

- [ ] **Step 4: 最小实现 safety margin 与 hard boundaries**

要求：

- 生成 `liquidity/credit/energy/policy_margin`
- 生成 `system_margin_score`
- 生成三类 hard boundaries
- `policy_margin` 需要显式吸收 `policy_relief_probability`
- `liquidity_margin` 需要显式吸收：
  - `usd_liquidity_chain`
  - `financial_sanctions_chain`
  - `risk_off_deleveraging_chain`
- `credit_margin` 需要显式吸收：
  - `credit_intermediary_chain`
  - `usd_liquidity_chain`
- `energy_margin` 需要显式吸收：
  - `energy_supply_chain`
  - `shipping_supply_chain`
- `system_margin_score` 不能简单平均；必须对：
  - `dominant_chain`
  - 最危险 `active` chain
  - 已触发 `hard_boundaries`
  提高权重
- `hard_boundaries` 至少显式考虑：
  - `game_state = systemic_repricing`
  - `regime_state = systemic_risk`
  - `risk_off_deleveraging_chain = dominant`
- 不直接接 overlay
- 字段合同必须锁住：
  - 所有 score/margin 在 `[0.0, 1.0]`
  - `generated_at_utc` + ISO8601 `Z`
  - `boundary_reasons` 用列表，不用 `null`
  - 其他未知列表字段也用空列表，不用 `null`
  - `hard_boundaries` 只用冻结字段名

- [ ] **Step 5: 跑 tests 确认转绿**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_safety_margin.py
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_safety_margin.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_safety_margin.py
git commit -m "feat(research): add event safety margin snapshot"
```

## 6. Task B3：runner 扩展为按固定顺序生成三类新 artifacts（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_event_crisis_pipeline.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_event_crisis_pipeline.py`

- [ ] **Step 1: 写 runner artifact failing test**

```python
def test_run_event_crisis_pipeline_writes_geostrategy_artifacts(tmp_path: Path) -> None:
    artifacts = run_pipeline(...)
    assert (tmp_path / "review" / "latest_event_game_state_snapshot.json").exists()
    assert (tmp_path / "review" / "latest_event_transmission_chain_map.json").exists()
    assert (tmp_path / "review" / "latest_event_safety_margin_snapshot.json").exists()
    assert artifacts["game_state"].name == "latest_event_game_state_snapshot.json"
    assert artifacts["transmission"].name == "latest_event_transmission_chain_map.json"
    assert artifacts["safety_margin"].name == "latest_event_safety_margin_snapshot.json"
```

- [ ] **Step 2: 跑 test 确认先红**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_run_event_crisis_pipeline.py -k geostrategy
```

Expected: FAIL

- [ ] **Step 3: 最小扩展 runner**

要求：

- 单入口 runner 不变
- 固定顺序必须落实为：
  - `event_intake`
  - `event_game_state_snapshot`
  - `event_transmission_chain_map`
  - `event_regime_snapshot`
  - `event_crisis_analogy`
  - `event_asset_shock_map`
  - `event_safety_margin_snapshot`
  - `event_live_guard_overlay`
  - `event_crisis_operator_summary`
- 新三层 artifacts 先以解释层形式落地；不在本阶段改变 overlay 压缩逻辑
- 保持现有 `latest_event_*` 不回退

- [ ] **Step 4: 跑 test 确认转绿**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_run_event_crisis_pipeline.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_event_crisis_pipeline.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_event_crisis_pipeline.py
git commit -m "feat(research): expand event crisis runner with geostrategy artifacts"
```

## 7. Task C1：将 safety margin 压缩进 overlay（LIVE_GUARD_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_pipeline.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_pipeline.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_event_crisis_pipeline.py`

- [ ] **Step 1: 写 overlay compression failing test**

```python
def test_live_guard_overlay_uses_margin_and_hard_boundaries() -> None:
    overlay = build_event_live_guard_overlay(
        regime_snapshot={...},
        safety_margin_snapshot={...},
        transmission_chain_map={...},
        generated_at=...,
    )
    assert overlay["canary_freeze"] is True
    assert overlay["risk_multiplier_override"] < 1.0
```

- [ ] **Step 2: 跑 test 确认先红**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_crisis_pipeline.py tests/test_run_event_crisis_pipeline.py -k overlay
```

Expected: FAIL

- [ ] **Step 3: 最小实现 overlay 压缩**

要求：

- 只压缩，不放权
- `hard_boundaries` 可以触发 `canary_freeze`
- `system_margin_score` 只能向下压 `risk_multiplier_override`
- 保持 `valid_until_utc`、`override_reason_codes` 合同
- overlay 压缩的输入顺序必须显式体现：
  - 先有 `game_state_snapshot / transmission_chain_map / regime_snapshot / asset_shock_map / safety_margin_snapshot`
  - 再压缩为 `event_live_guard_overlay`

- [ ] **Step 4: 跑 test 确认转绿**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_event_crisis_pipeline.py tests/test_run_event_crisis_pipeline.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_pipeline.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_pipeline.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_event_crisis_pipeline.py
git commit -m "feat(guard): compress safety margins into event overlay"
```

## 8. Task D1：dashboard / operator 展示增强（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_event_crisis_pipeline.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py`

- [ ] **Step 1: 写 dashboard exposure failing test**

```python
def test_dashboard_snapshot_exposes_geostrategy_artifacts(tmp_path: Path) -> None:
    snapshot = build_public_snapshot(tmp_path)
    assert "event_game_state_snapshot" in snapshot["artifact_payloads"]
    assert "event_transmission_chain_map" in snapshot["artifact_payloads"]
    assert "event_safety_margin_snapshot" in snapshot["artifact_payloads"]
```

- [ ] **Step 2: 写 read-model failing test**

```ts
it('maps geostrategy artifacts into existing terminal views', () => {
  const model = buildTerminalReadModel(snapshotWithGeostrategyArtifacts);
  expect(model.dataRegime.microCapture.some((metric) => metric.id === 'game-state')).toBe(true);
  expect(model.signalRisk.repairPlan.some((metric) => metric.id === 'safety-margin')).toBe(true);
});
```

- [ ] **Step 3: 跑 tests 确认先红**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_build_dashboard_frontend_snapshot_script.py -k geostrategy
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- src/adapters/read-model.test.ts -t geostrategy
```

Expected: FAIL

- [ ] **Step 4: 最小实现展示增强**

要求：

- 只消费新 artifacts
- 仍然不在 UI 层重算 authority
- `run_event_crisis_pipeline.py` 的 operator summary 只透传：
  - 主战场
  - dominant chain
  - safety margin brief
  - hard boundary brief
- `build_operator_task_visual_panel.py` / `run_operator_panel_refresh.py` 继续只消费该 summary，不直接读取解释层 artifacts 做重算

- [ ] **Step 5: 跑 tests 确认转绿**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_build_dashboard_frontend_snapshot_script.py tests/test_build_operator_task_visual_panel_script.py tests/test_run_operator_panel_refresh_script.py
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- src/adapters/read-model.test.ts -t geostrategy
npx tsc --noEmit
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_event_crisis_pipeline.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py
git commit -m "feat(research): surface geostrategy crisis layers"
```

## 9. 端到端回归与交付判定

- [ ] **Step 1: 跑 Python 核心回归**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
mkdir -p latest_logs
PYTHONPATH=src pytest -q \
  tests/test_event_crisis_sources.py \
  tests/test_event_crisis_analogies.py \
  tests/test_event_crisis_pipeline.py \
  tests/test_run_event_crisis_pipeline.py \
  tests/test_event_game_state.py \
  tests/test_event_transmission.py \
  tests/test_event_safety_margin.py \
  tests/test_build_dashboard_frontend_snapshot_script.py \
  tests/test_build_operator_task_visual_panel_script.py \
  tests/test_run_operator_panel_refresh_script.py \
  tests/test_event_overlay.py \
  tests/test_orchestration.py \
  tests/test_engine_integration.py
```

Expected: PASS

- [ ] **Step 2: 跑前端 focused 回归**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- src/adapters/read-model.test.ts -t geostrategy
npx tsc --noEmit
```

Expected: PASS

- [ ] **Step 3: 跑 runner smoke**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src python3 scripts/run_event_crisis_pipeline.py --mode snapshot --output-root output
```

Expected:

- `latest_event_game_state_snapshot.json`
- `latest_event_transmission_chain_map.json`
- `latest_event_safety_margin_snapshot.json`
- `event_live_guard_overlay.json`

- [ ] **Step 4: 工作树干净**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
git status --short
```

Expected: no uncommitted changes

## 10. 交付判定

以下全部满足才算扩展版完成：

1. 三个新增解释层 artifacts 可稳定生成
2. 现有 regime / shock map 能吸收博弈层输入
3. `event_live_guard_overlay` 能吸收 safety margin / hard boundaries，并仍保持只降级
4. dashboard / operator 能展示主战场 / 主传导链 / safety margin / hard boundaries
5. 全部回归通过，且不破坏当前已闭合主线
