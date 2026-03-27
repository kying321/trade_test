# Domestic Commodity Reasoning Line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fenlie 增加一条独立于交易执行的“国内大宗商品逻辑推理分析线”，产出情景树、传导链、边界强度与结论摘要，并在 dashboard/operator 中提供单独入口。

**Architecture:** 复用当前 event crisis geostrategy 链的 artifact-first 模式，但建立一条完全独立的 commodity reasoning 管线：`事件/研究 artifacts -> 情景树 -> 传导链 -> 验证环 -> 边界强度 -> 摘要`。UI 层只消费 reasoning artifacts 与 summary，不重算 authority；operator 只透传 reasoning summary 的 compact fields。

**Tech Stack:** Python 3、repo-owned scripts、JSON review artifacts、Pytest、TypeScript read-model、现有 dashboard snapshot / operator panel refresh 链

---

## 0. 范围与边界

- change class：`RESEARCH_ONLY`
- 首版对象：国内大宗商品通用框架，粒度固定为：
  - `板块 -> 品种 -> 具体合约`
- 首个验证样本：
  - `BU2606`
- 主输入：
  - 事件类 source artifacts
  - 研究类 source artifacts
- 验证环：
  - 截面数据
  - 截面新闻
  - 可得的同花顺期货通/自定义指标派生观测
- 禁止项：
  - 不接 live execution
  - 不产出 order/risk/capital 动作
  - 验证环不能替代 source authority

## 1. 目标文件结构

### Create

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_scenario.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_transmission.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_validation.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_boundary.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_summary.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_domestic_commodity_reasoning.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_scenario.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_transmission.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_validation.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_boundary.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_domestic_commodity_reasoning.py`

### Modify

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py`

## 2. Task 1：情景树 builder（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_scenario.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_scenario.py`

- [ ] **Step 1: 写 failing tests**

```python
def test_build_commodity_reasoning_scenario_tree_emits_primary_and_secondary_scenarios() -> None:
    payload = build_commodity_reasoning_scenario_tree(
        event_artifacts=[...],
        research_artifacts=[...],
        contract_focus="BU2606",
    )
    assert payload["primary_scenario"]
    assert isinstance(payload["secondary_scenarios"], list)
    assert payload["scenario_nodes"]
```

```python
def test_commodity_reasoning_scenario_tree_contract_fields_are_stable() -> None:
    payload = build_commodity_reasoning_scenario_tree(...)
    assert payload["generated_at_utc"].endswith("Z")
    assert all(0.0 <= float(row["confidence_score"]) <= 1.0 for row in payload["scenario_nodes"])
    assert all(isinstance(row["trigger_conditions"], list) for row in payload["scenario_nodes"])
    assert all(isinstance(row["invalidators"], list) for row in payload["scenario_nodes"])
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src pytest -q tests/test_commodity_reasoning_scenario.py
```

Expected: FAIL

- [ ] **Step 3: 最小实现 scenario builder**

要求：

- 只读事件/研究类 artifacts
- 生成主情景 / 次情景 / scenario_nodes
- 输出板块、品种、合约焦点（首个样本允许 `BU2606`）
- 不引入行情因子 authority
- 所有分数字段锁在 `[0,1]`

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q tests/test_commodity_reasoning_scenario.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_scenario.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_scenario.py
git commit -m "feat(research): add commodity reasoning scenario tree"
```

## 3. Task 2：传导链 builder（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_transmission.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_transmission.py`

- [ ] **Step 1: 写 failing tests**

```python
def test_build_commodity_reasoning_transmission_map_outputs_sector_to_contract_chain() -> None:
    payload = build_commodity_reasoning_transmission_map(
        scenario_tree={...},
        contract_focus="BU2606",
    )
    assert payload["primary_chain"]
    assert payload["chains"]
    assert any(row["contract"] == "BU2606" for row in payload["chains"])
```

```python
def test_commodity_reasoning_transmission_map_contract_is_stable() -> None:
    payload = build_commodity_reasoning_transmission_map(...)
    assert payload["generated_at_utc"].endswith("Z")
    assert all(0.0 <= float(row["confidence_score"]) <= 1.0 for row in payload["chains"])
    assert all(isinstance(row["path_nodes"], list) for row in payload["chains"])
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q tests/test_commodity_reasoning_transmission.py
```

- [ ] **Step 3: 最小实现 transmission builder**

要求：

- 从 scenario tree 推导板块/品种/合约三级链
- 输出 `range_scope` 与 `boundary_strength` 初值
- 不在这里做 validation ring 重算

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q tests/test_commodity_reasoning_transmission.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_transmission.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_transmission.py
git commit -m "feat(research): add commodity reasoning transmission map"
```

## 4. Task 3：验证环 helper（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_validation.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_validation.py`

- [ ] **Step 1: 写 failing tests**

```python
def test_build_commodity_reasoning_validation_ring_collects_counter_evidence() -> None:
    payload = build_commodity_reasoning_validation_ring(
        transmission_map={...},
        cross_section_news=[...],
        cross_section_data=[...],
    )
    assert isinstance(payload["counter_evidence"], list)
    assert isinstance(payload["scope_adjustments"], list)
```

```python
def test_validation_ring_never_promotes_authority() -> None:
    payload = build_commodity_reasoning_validation_ring(...)
    assert payload["promotion_allowed"] is False
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q tests/test_commodity_reasoning_validation.py
```

- [ ] **Step 3: 最小实现 validation helper**

要求：

- 允许输入截面数据 / 截面新闻 / 期货通衍生观测
- 只输出：
  - `counter_evidence`
  - `scope_adjustments`
  - `boundary_pressure`
  - `review_required`
- 必须硬锁：
  - `promotion_allowed = false`

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q tests/test_commodity_reasoning_validation.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_validation.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_validation.py
git commit -m "feat(research): add commodity reasoning validation ring"
```

## 5. Task 4：边界强度与 summary（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_boundary.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_summary.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_boundary.py`

- [ ] **Step 1: 写 failing tests**

```python
def test_build_commodity_reasoning_boundary_strength_outputs_range_and_fragility() -> None:
    payload = build_commodity_reasoning_boundary_strength(
        transmission_map={...},
        validation_ring={...},
    )
    assert payload["boundary_rows"]
    assert all(isinstance(row["fragility_flags"], list) for row in payload["boundary_rows"])
```

```python
def test_build_commodity_reasoning_summary_outputs_operator_briefs() -> None:
    payload = build_commodity_reasoning_summary(
        scenario_tree={...},
        transmission_map={...},
        boundary_strength={...},
    )
    assert payload["primary_scenario_brief"]
    assert payload["primary_chain_brief"]
    assert payload["boundary_strength_brief"]
    assert payload["contracts_in_focus"]
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q tests/test_commodity_reasoning_boundary.py
```

- [ ] **Step 3: 最小实现 boundary + summary**

要求：

- boundary artifact 输出：
  - `range_scope`
  - `boundary_strength`
  - `persistence_strength`
  - `fragility_flags`
  - `counter_evidence`
- summary artifact 输出：
  - `headline`
  - `primary_scenario_brief`
  - `primary_chain_brief`
  - `range_scope_brief`
  - `boundary_strength_brief`
  - `invalidator_brief`
  - `contracts_in_focus`

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q tests/test_commodity_reasoning_boundary.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_boundary.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/commodity_reasoning_summary.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_commodity_reasoning_boundary.py
git commit -m "feat(research): add commodity reasoning boundary and summary"
```

## 6. Task 5：runner 与 artifacts 落地（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_domestic_commodity_reasoning.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_domestic_commodity_reasoning.py`

- [ ] **Step 1: 写 failing tests**

```python
def test_run_domestic_commodity_reasoning_writes_all_artifacts(tmp_path: Path) -> None:
    artifacts = run_pipeline(...)
    assert (tmp_path / "review" / "latest_commodity_reasoning_scenario_tree.json").exists()
    assert (tmp_path / "review" / "latest_commodity_reasoning_transmission_map.json").exists()
    assert (tmp_path / "review" / "latest_commodity_reasoning_boundary_strength.json").exists()
    assert (tmp_path / "review" / "latest_commodity_reasoning_summary.json").exists()
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q tests/test_run_domestic_commodity_reasoning.py
```

- [ ] **Step 3: 最小实现 runner**

要求：

- 单入口输出 4 个 artifacts：
  - `latest_commodity_reasoning_scenario_tree.json`
  - `latest_commodity_reasoning_transmission_map.json`
  - `latest_commodity_reasoning_boundary_strength.json`
  - `latest_commodity_reasoning_summary.json`
- 允许显式 `--contract BU2606`
- 支持空 validation ring
- 不接 live guard / execution

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q tests/test_run_domestic_commodity_reasoning.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_domestic_commodity_reasoning.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_domestic_commodity_reasoning.py
git commit -m "feat(research): add domestic commodity reasoning runner"
```

## 7. Task 6：dashboard snapshot / read-model（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py`

- [ ] **Step 1: 写 failing tests**

```python
def test_dashboard_snapshot_exposes_commodity_reasoning_artifacts(tmp_path: Path) -> None:
    snapshot = build_public_snapshot(tmp_path)
    assert "commodity_reasoning_scenario_tree" in snapshot["artifact_payloads"]
    assert "commodity_reasoning_transmission_map" in snapshot["artifact_payloads"]
    assert "commodity_reasoning_boundary_strength" in snapshot["artifact_payloads"]
    assert "commodity_reasoning_summary" in snapshot["artifact_payloads"]
```

```ts
it('maps commodity reasoning artifacts into terminal views', () => {
  const model = buildTerminalReadModel(snapshotWithCommodityReasoning);
  expect(model.dataRegime.microCapture.some((metric) => metric.id === 'commodity-scenario')).toBe(true);
  expect(model.signalRisk.repairPlan.some((metric) => metric.id === 'commodity-boundary')).toBe(true);
});
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src pytest -q tests/test_build_dashboard_frontend_snapshot_script.py -k commodity_reasoning
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- src/adapters/read-model.test.ts -t commodity
```

- [ ] **Step 3: 最小实现 snapshot + read-model**

要求：

- dashboard 只暴露新 artifacts
- read-model 只消费新 artifacts
- 不在 UI 层重算 reasoning authority

- [ ] **Step 4: 跑 tests 转绿**

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src pytest -q tests/test_build_dashboard_frontend_snapshot_script.py
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- src/adapters/read-model.test.ts -t commodity
npx tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py
git commit -m "feat(research): surface commodity reasoning artifacts"
```

## 8. Task 7：operator 入口（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py`

- [ ] **Step 1: 写 failing tests**

```python
def test_operator_panel_summary_exposes_commodity_reasoning_briefs(...) -> None:
    ...
    assert payload["summary"]["commodity_reasoning_primary_scenario_brief"]
    assert payload["summary"]["commodity_reasoning_primary_chain_brief"]
    assert payload["summary"]["commodity_reasoning_boundary_strength_brief"]
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q \
  tests/test_build_operator_task_visual_panel_script.py \
  tests/test_run_operator_panel_refresh_script.py -k commodity_reasoning
```

- [ ] **Step 3: 最小实现 operator compact contract**

要求：

- operator 只透传：
  - `commodity_reasoning_primary_scenario_brief`
  - `commodity_reasoning_primary_chain_brief`
  - `commodity_reasoning_range_scope_brief`
  - `commodity_reasoning_boundary_strength_brief`
  - `commodity_reasoning_invalidator_brief`
- 不能重算 authority

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q \
  tests/test_build_operator_task_visual_panel_script.py \
  tests/test_run_operator_panel_refresh_script.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py
git commit -m "feat(research): add commodity reasoning operator lane"
```

## 9. Task 8：端到端验收（RESEARCH_ONLY）

**Files:**
- No new files required

- [ ] **Step 1: 跑 Python reasoning 回归**

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src pytest -q \
  tests/test_commodity_reasoning_scenario.py \
  tests/test_commodity_reasoning_transmission.py \
  tests/test_commodity_reasoning_validation.py \
  tests/test_commodity_reasoning_boundary.py \
  tests/test_run_domestic_commodity_reasoning.py
```

- [ ] **Step 2: 跑 dashboard/operator 回归**

```bash
PYTHONPATH=src pytest -q \
  tests/test_build_dashboard_frontend_snapshot_script.py \
  tests/test_build_operator_task_visual_panel_script.py \
  tests/test_run_operator_panel_refresh_script.py
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- src/adapters/read-model.test.ts -t commodity
npx tsc --noEmit
```

- [ ] **Step 3: 跑真实 runner 样本**

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src python3 scripts/run_domestic_commodity_reasoning.py \
  --workspace /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1 \
  --contract BU2606
```

Expected:

- 生成 `latest_commodity_reasoning_*.json`
- 可被 dashboard/operator 消费

- [ ] **Step 4: 最终提交**

```bash
git add /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research \
        /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts \
        /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests \
        /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters
git commit -m "feat(research): add domestic commodity reasoning line"
```

## 10. 停止规则

- 若 Task 3 发现验证环输入需要独立 connector / browser 抓取，先停在 helper 设计，不把外部 GUI 工具接入主线
- 若 Task 5 无法在 `BU2606` 样本上落出最小 artifacts，就先停在 Python reasoning 链，不提前做 UI
- 若两轮连续没有改变 source-owned artifact 或 user-visible capability，按 stop rule 停止并汇报
