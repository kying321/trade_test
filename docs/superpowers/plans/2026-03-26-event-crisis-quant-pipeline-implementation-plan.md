# Event Crisis Quant Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Fenlie 增加一条公开信息驱动的黑天鹅事件量化管道，先以 `RESEARCH_ONLY` 落地 source-owned 事件 artifacts 与 dashboard/operator 消费，再以 `LIVE_GUARD_ONLY` 把事件 overlay 安全接入 risk/gate。

**Architecture:** 采用独立 `event regime sidecar`，核心纯逻辑放在 `system/src/lie_engine/research/`，由单入口 `system/scripts/run_event_crisis_pipeline.py` 写入 review/state artifacts。第一阶段只新增 research artifacts 与 UI/operator 消费，不碰 executor；第二阶段再把 `event_live_guard_overlay.json` 叠加到现有 `guards.py` / `engine.py` 的风控链，且只能自动降级，不能自动放行。

**Tech Stack:** Python 3、repo-owned scripts、JSON review/state artifacts、Pytest、TypeScript dashboard read-model、现有 Fenlie operator panel / dashboard snapshot 管线

---

## 0. 范围与分阶段边界

- 第一阶段 change class：`RESEARCH_ONLY`
- 第二阶段 change class：`LIVE_GUARD_ONLY`
- 本计划不实现：
  - 直接改 live executor
  - 事件分数直接入 signal alpha
  - 自动恢复风险 / 自动放行 live
  - 付费新闻抓取 / 非公开数据源

### 0.1 目标 artifacts（冻结）

第一阶段必须生成并维护以下 source-owned artifacts：

- `system/output/review/latest_event_intake.json`
- `system/output/review/latest_event_regime_snapshot.json`
- `system/output/review/latest_event_crisis_analogy.json`
- `system/output/review/latest_event_asset_shock_map.json`
- `system/output/review/latest_event_crisis_operator_summary.json`

第二阶段才允许写：

- `system/output/state/event_live_guard_overlay.json`

### 0.2 单入口 runner 冻结

唯一入口固定为：

- `system/scripts/run_event_crisis_pipeline.py`

支持模式：

- `snapshot`
- `hourly`
- `eod_summary`

不允许再新增第二个并行 runner 去写同一组 `latest_event_*` artifacts。

## 1. 目标文件结构

### Create

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_sources.py`
  - 公开事件/市场代理输入的标准化加载器
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_analogies.py`
  - 历史危机 archetype library 与相似度计算
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_pipeline.py`
  - 纯函数：事件分类、评分、asset shock map、operator summary、overlay 生成
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/orchestration/event_overlay.py`
  - `event_live_guard_overlay.json` 的读取、clamp、fail-closed 处理
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_event_crisis_pipeline.py`
  - 单入口 CLI runner
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_sources.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_analogies.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_pipeline.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_overlay.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_event_crisis_pipeline.py`

### Modify

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py`
  - 将 `latest_event_*` artifacts 暴露进 dashboard snapshot
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts`
  - 为事件体制、类比、shock map、live guard overlay 增加 read-model
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py`
  - 将事件 operator summary 摘要接入 summary/source rows
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py`
  - 面板刷新 summary 新增事件 brief 字段
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/orchestration/guards.py`
  - 第二阶段：将静态 `black_swan_assessment` 与事件 overlay 协调
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/engine.py`
  - 第二阶段：加载 event overlay，并只允许向下压 risk/gate
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/reporting/daily.py`
  - 事件摘要字段对齐 daily briefing
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_orchestration.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_engine_integration.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py`

## 2. Task 1：事件量化核心纯逻辑（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_analogies.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_pipeline.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_analogies.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_pipeline.py`

- [ ] **Step 1: 写 archetype library failing test**

```python
def test_analogy_library_contains_required_archetypes() -> None:
    from lie_engine.research.event_crisis_analogies import build_default_archetypes

    ids = {row["archetype_id"] for row in build_default_archetypes()}
    assert "gfc_2008" in ids
    assert "energy_credit_2014_2016" in ids
    assert "private_credit_redemption_2026" in ids
```

- [ ] **Step 2: 跑 test 确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q tests/test_event_crisis_analogies.py -k archetype
```

Expected: FAIL

- [ ] **Step 3: 写 `event_regime_snapshot` 纯函数 failing test**

```python
def test_build_event_regime_snapshot_outputs_scores_and_regime_state() -> None:
    snapshot = build_event_regime_snapshot(
        event_rows=[...],
        market_inputs={...},
    )
    assert snapshot["regime_state"] in {"watch", "sector_stress", "cross_asset_contagion", "systemic_risk"}
    assert 0.0 <= float(snapshot["event_severity_score"]) <= 1.0
    assert 0.0 <= float(snapshot["systemic_risk_score"]) <= 1.0
```

- [ ] **Step 4: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_event_crisis_pipeline.py -k regime_snapshot
```

Expected: FAIL

- [ ] **Step 5: 写 asset shock map failing test**

```python
def test_build_event_asset_shock_map_covers_priority_assets() -> None:
    payload = build_event_asset_shock_map(...)
    assets = {row["asset"] for row in payload["assets"]}
    assert {"BTC", "ETH", "SOL", "BNB", "GOLD", "UST_LONG", "OIL", "BANKS", "HIGH_YIELD"} <= assets
```

- [ ] **Step 6: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_event_crisis_pipeline.py -k shock_map
```

Expected: FAIL

- [ ] **Step 7: 最小实现 archetype library 与纯函数**

要求：

- `event_crisis_analogies.py` 只放静态 archetype 定义与相似度函数
- `event_crisis_pipeline.py` 只放纯逻辑，不做网络 I/O
- 产出字段必须与 spec 中的 artifact 合同一致
- 初始默认状态对当前 2026-03 事件偏向 `sector_stress`

- [ ] **Step 8: 跑 tests 确认转绿**

Run:

```bash
pytest -q tests/test_event_crisis_analogies.py tests/test_event_crisis_pipeline.py
```

Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_analogies.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_pipeline.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_analogies.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_pipeline.py
git commit -m "feat(research): add event crisis scoring core"
```

## 3. Task 2：公开输入标准化与单入口 runner（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_sources.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_event_crisis_pipeline.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_sources.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_event_crisis_pipeline.py`

- [ ] **Step 1: 写 source normalization failing test**

```python
def test_normalize_public_event_rows_keeps_timestamp_and_classes() -> None:
    rows = normalize_public_event_rows([...])
    assert rows[0]["event_ts_utc"].endswith("Z")
    assert rows[0]["event_classes"]
```

- [ ] **Step 2: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_event_crisis_sources.py -k normalize_public_event_rows
```

Expected: FAIL

- [ ] **Step 3: 写 runner artifact contract failing test**

```python
def test_run_event_crisis_pipeline_writes_latest_artifacts(tmp_path: Path) -> None:
    payload = run_pipeline(output_root=tmp_path, mode="snapshot", event_rows=[...], market_inputs={...})
    assert (tmp_path / "review" / "latest_event_intake.json").exists()
    assert (tmp_path / "review" / "latest_event_regime_snapshot.json").exists()
    assert (tmp_path / "review" / "latest_event_crisis_analogy.json").exists()
    assert (tmp_path / "review" / "latest_event_asset_shock_map.json").exists()
    assert (tmp_path / "review" / "latest_event_crisis_operator_summary.json").exists()
```

- [ ] **Step 4: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_run_event_crisis_pipeline.py -k latest_artifacts
```

Expected: FAIL

- [ ] **Step 5: 最小实现 sources + runner**

要求：

- `event_crisis_sources.py` 负责：
  - 公开事件行标准化
  - 市场代理输入标准化
  - 默认 priority assets 集合
- `run_event_crisis_pipeline.py` 负责：
  - `snapshot | hourly | eod_summary`
  - 单入口串行写 artifact
  - 保持 `latest_*` 路径稳定
- 网络请求必须：
  - `timeout <= 5000ms`
  - 有 rate limit
  - 写入显式时间戳
- 第一版允许先用 fixture / injected rows 走通测试，不要求一次性打通全部真实公开源

- [ ] **Step 6: 跑 tests 确认转绿**

Run:

```bash
pytest -q tests/test_event_crisis_sources.py tests/test_run_event_crisis_pipeline.py
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/research/event_crisis_sources.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_event_crisis_pipeline.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_crisis_sources.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_event_crisis_pipeline.py
git commit -m "feat(research): add event crisis pipeline runner"
```

## 4. Task 3：dashboard snapshot 与 terminal read-model 接线（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py`

- [ ] **Step 1: 写 snapshot 暴露事件 artifacts 的 failing test**

```python
def test_snapshot_includes_event_crisis_artifacts(tmp_path: Path) -> None:
    payload = build_snapshot(...)
    assert "event_regime_snapshot" in payload["artifact_payloads"]
    assert "event_crisis_analogy" in payload["artifact_payloads"]
    assert "event_asset_shock_map" in payload["artifact_payloads"]
```

- [ ] **Step 2: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_build_dashboard_frontend_snapshot_script.py -k event_crisis
```

Expected: FAIL

- [ ] **Step 3: 写 read-model failing test**

```ts
it('maps event crisis artifacts into data-regime and signal-risk views', () => {
  const model = buildReadModel(snapshotFixtureWithEventCrisisArtifacts);
  expect(model.dataRegime.sourceConfidence.length).toBeGreaterThan(0);
  expect(model.signalRisk.repairPlan.some((row) => String(row.value).includes('event'))).toBe(true);
});
```

- [ ] **Step 4: 跑 test 确认先红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- --runInBand src/adapters/read-model.test.ts -t event
```

Expected: FAIL

- [ ] **Step 5: 最小实现 snapshot + read-model**

要求：

- `build_dashboard_frontend_snapshot.py` 将 `latest_event_*` artifacts 加入 `artifact_payloads`
- `read-model.ts` 只消费 artifact，不自己重算 authority
- 第一版只要把事件体制、类比、shock map、overlay 摘要挂到现有 panel/section，不必新建整块复杂 UI

- [ ] **Step 6: 跑 tests 确认转绿**

Run:

```bash
pytest -q tests/test_build_dashboard_frontend_snapshot_script.py -k event_crisis
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- --runInBand src/adapters/read-model.test.ts -t event
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_dashboard_frontend_snapshot.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.ts \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web/src/adapters/read-model.test.ts \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py
git commit -m "feat(research): surface event crisis artifacts in dashboard"
```

## 5. Task 4：operator panel / refresh summary 接线（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py`
- Test: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py`

- [ ] **Step 1: 写 operator summary failing test**

```python
def test_operator_panel_summary_surfaces_event_crisis_briefs(tmp_path: Path) -> None:
    payload = build_operator_task_visual_panel(...)
    assert "event_crisis_regime_brief" in payload["summary"]
    assert "event_crisis_top_analogue_brief" in payload["summary"]
```

- [ ] **Step 2: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_build_dashboard_frontend_snapshot_script.py -k operator_panel_summary_surfaces_event
```

Expected: FAIL

- [ ] **Step 3: 最小实现 operator 接线**

要求：

- `build_operator_task_visual_panel.py` 读取 `latest_event_crisis_operator_summary.json`
- summary 只透传 event brief 字段，不重算事件分数
- `run_operator_panel_refresh.py` 的 summary JSON 也要透出这些字段，便于 handoff

- [ ] **Step 4: 跑 test 确认转绿**

Run:

```bash
pytest -q tests/test_build_dashboard_frontend_snapshot_script.py -k operator_panel_summary_surfaces_event
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_dashboard_frontend_snapshot_script.py
git commit -m "feat(research): expose event crisis summary in operator panel"
```

## 6. Task 5：event overlay 纯读取与 fail-closed contract（LIVE_GUARD_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/orchestration/event_overlay.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_overlay.py`

- [ ] **Step 1: 写 overlay loader failing test**

```python
def test_load_event_live_guard_overlay_clamps_and_fail_closes(tmp_path: Path) -> None:
    path = tmp_path / "event_live_guard_overlay.json"
    path.write_text('{"risk_multiplier_override": 1.2, "canary_freeze": true}', encoding="utf-8")
    payload = load_event_live_guard_overlay(path)
    assert payload["risk_multiplier_override"] == 1.0
    assert payload["canary_freeze"] is True
```

- [ ] **Step 2: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_event_overlay.py
```

Expected: FAIL

- [ ] **Step 3: 最小实现 overlay helper**

要求：

- 缺失文件 / 非法 JSON / 过期字段 -> fail-closed
- `risk_multiplier_override` 只能在 `[0.0, 1.0]`
- overlay 只能表达降级，不表达放行

- [ ] **Step 4: 跑 test 确认转绿**

Run:

```bash
pytest -q tests/test_event_overlay.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/orchestration/event_overlay.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_event_overlay.py
git commit -m "feat(guard): add event live guard overlay loader"
```

## 7. Task 6：把 event overlay 接入 guards / engine（LIVE_GUARD_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/orchestration/guards.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/engine.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/reporting/daily.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_orchestration.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_engine_integration.py`

- [ ] **Step 1: 写 guard integration failing test**

```python
def test_build_guard_assessment_adds_event_overlay_reason_when_canary_frozen() -> None:
    out = build_guard_assessment(..., event_overlay={"canary_freeze": True, "override_reason_codes": ["event_crisis_contagion"]})
    assert out.trade_blocked
    assert any("event" in reason.lower() for reason in out.non_trade_reasons)
```

- [ ] **Step 2: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_orchestration.py -k event_overlay
```

Expected: FAIL

- [ ] **Step 3: 写 engine risk multiplier failing test**

```python
def test_engine_execution_risk_control_respects_event_overlay_floor(self) -> None:
    out = eng._execution_risk_control(...)
    self.assertAlmostEqual(float(out.get("risk_multiplier", 0.0)), 0.5, places=6)
```

- [ ] **Step 4: 跑 test 确认先红**

Run:

```bash
pytest -q tests/test_engine_integration.py -k event_overlay
```

Expected: FAIL

- [ ] **Step 5: 最小实现 guard / engine 接线**

要求：

- `guards.py` 保留现有 `black_swan_assessment`，但允许叠加 event overlay reason
- `engine.py` 在现有 `risk_control` 上叠加 `event_live_guard_overlay`
- 叠加逻辑只能向下压 `risk_multiplier`
- `canary_freeze=true` 时必须进入保护/阻断路径
- `daily.py` 输出事件 regime / analogue 简报字段

- [ ] **Step 6: 跑 tests 确认转绿**

Run:

```bash
pytest -q tests/test_event_overlay.py tests/test_orchestration.py -k event_overlay
pytest -q tests/test_engine_integration.py -k event_overlay
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/orchestration/guards.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/engine.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/src/lie_engine/reporting/daily.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_orchestration.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_engine_integration.py
git commit -m "feat(guard): apply event crisis overlay to risk and gate"
```

## 8. Task 7：端到端回归与文档收口

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/FENLIE_SKILL_USAGE_PLAYBOOK.md`（如需新增 runner 用法）
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/docs/FENLIE_CODEX_MEMORY.md`（仅在需要补 reusable validation commands 时最小更新）

- [ ] **Step 1: 跑 RESEARCH_ONLY 回归**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q \
  tests/test_event_crisis_sources.py \
  tests/test_event_crisis_analogies.py \
  tests/test_event_crisis_pipeline.py \
  tests/test_run_event_crisis_pipeline.py \
  tests/test_build_dashboard_frontend_snapshot_script.py
```

Expected: PASS

- [ ] **Step 2: 跑 dashboard read-model 回归**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/dashboard/web
npm test -- --runInBand src/adapters/read-model.test.ts
```

Expected: PASS

- [ ] **Step 3: 跑 LIVE_GUARD_ONLY 回归**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
pytest -q \
  tests/test_event_overlay.py \
  tests/test_orchestration.py \
  tests/test_engine_integration.py
```

Expected: PASS

- [ ] **Step 4: 运行 snapshot 模式 smoke**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src python3 scripts/run_event_crisis_pipeline.py --mode snapshot --output-root output
```

Expected:

- `system/output/review/latest_event_intake.json`
- `system/output/review/latest_event_regime_snapshot.json`
- `system/output/review/latest_event_crisis_analogy.json`
- `system/output/review/latest_event_asset_shock_map.json`
- `system/output/review/latest_event_crisis_operator_summary.json`
- `system/output/state/event_live_guard_overlay.json`（仅在第二阶段接线完成后）

- [ ] **Step 5: 工作树干净**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
git status --short
```

Expected: no uncommitted changes

## 9. 交付判定

以下全部满足才算完成：

1. 存在唯一 runner：`system/scripts/run_event_crisis_pipeline.py`
2. 五类 `latest_event_*` review artifacts 可稳定生成
3. `event_live_guard_overlay.json` 仅表达降级，不表达放行
4. dashboard / operator 只消费 artifacts，不自重算 authority
5. 当前 2026-03 事件可输出：
   - `sector_stress` 初始判断
   - top crisis analogues
   - `BTC / ETH / SOL / BNB / GOLD / UST_LONG / OIL / BANKS / HIGH_YIELD` 冲击映射
6. 第二阶段接线后，event overlay 能向下压 `risk_multiplier` 并冻结新 canary
