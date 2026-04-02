# Operator Source Chain Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 恢复 `brooks/commodity/hot_brief/cross_market -> operator panel` 的 source-owned 生成链，让 operator panel / refresh / browser smoke 在正常路径下不再依赖 degraded fallback。

**Architecture:** 不直接回滚整套历史控制面，而是按最小闭包分阶段恢复：先恢复 `build_hot_universe_operator_brief` 所需 helper，再恢复 `refresh_brooks_structure_state`、`refresh_commodity_paper_execution_state`、`refresh_cross_market_operator_state`，最后把 `run_operator_panel_refresh.py` 与 browser smoke 切回 full-source 正常路径。恢复期间保留当前 `degraded-but-runnable` fallback，直到 full-source 路径通过回归后再决定是否收紧。

**Tech Stack:** Python 3、repo-owned scripts、JSON review artifacts、Pytest、现有 operator panel / dashboard refresh / browser smoke 脚本

---

## 0. 范围、边界与恢复顺序

- change class：`RESEARCH_ONLY`
- 本计划只恢复 review/public/dist 监控链，不触碰 live execution
- authority 仍遵守：`source-of-truth artifacts > compact handoff/context > UI/operator summaries > chat memory`
- 恢复期间允许 `build_operator_task_visual_panel.py` 保持 `degraded-but-runnable`
- **不要** 一次性回植历史大闭包；严格按以下顺序：
  1. `build_hot_universe_operator_brief.py`
  2. `refresh_brooks_structure_state.py`
  3. `refresh_commodity_paper_execution_state.py`
  4. `refresh_cross_market_operator_state.py`
  5. `run_operator_panel_refresh.py` / `run_operator_panel_browser_smoke.py` 正常链路收口

## 1. 目标文件闭包

### Create / restore

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_hot_universe_operator_brief.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/operator_brief_debug_snapshots.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/operator_brief_source_chunks.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_brooks_structure_state.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_commodity_paper_execution_state.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_cross_market_operator_state.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_hot_universe_operator_brief_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_brooks_structure_state_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_commodity_paper_execution_state_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_cross_market_operator_state_script.py`

### Modify

- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_browser_smoke.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py`
- `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_browser_smoke_script.py`

### Historical reference source

- `git show 22cb60f0:system/scripts/refresh_cross_market_operator_state.py`
- `git show 22cb60f0:system/scripts/refresh_commodity_paper_execution_state.py`
- `git show 22cb60f0:system/scripts/refresh_brooks_structure_state.py`
- `git show 22cb60f0:system/scripts/build_hot_universe_operator_brief.py`
- `git show 22cb60f0:system/scripts/operator_brief_debug_snapshots.py`
- `git show 22cb60f0:system/scripts/operator_brief_source_chunks.py`

## 2. Task 1：恢复 hot brief 最小闭包（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_hot_universe_operator_brief.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/operator_brief_debug_snapshots.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/operator_brief_source_chunks.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_hot_universe_operator_brief_script.py`

- [ ] **Step 1: 从历史测试中提炼最小 failing case**

至少保留以下断言意图：

```python
def test_build_hot_universe_operator_brief_prefers_non_dry_run(tmp_path: Path) -> None:
    ...
    payload = run_script(...)
    assert payload["source_status"] == "ok"
    assert payload["source_mode"] == "single-hot-universe-source"
    assert payload["focus_primary_batches"]
    assert "summary_text" in payload
```

- [ ] **Step 2: 先跑单测确认红**

Run:

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src pytest -q tests/test_build_hot_universe_operator_brief_script.py
```

Expected: FAIL（脚本/依赖文件缺失）

- [ ] **Step 3: 从历史 commit 最小恢复 helper + builder**

要求：

- 先从 `22cb60f0` 恢复 helper/builder 原结构
- 不做大规模改名
- 仅修当前 import/path 兼容问题
- 保持输出 review artifact 风格，不引入 UI 层逻辑

- [ ] **Step 4: 跑 targeted tests 转绿**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_build_hot_universe_operator_brief_script.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_hot_universe_operator_brief.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/operator_brief_debug_snapshots.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/operator_brief_source_chunks.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_hot_universe_operator_brief_script.py
git commit -m "feat(research): restore hot universe operator brief builder"
```

## 3. Task 2：恢复 Brooks 结构 refresh（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_brooks_structure_state.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_brooks_structure_state_script.py`

- [ ] **Step 1: 写最小 failing test**

至少锁住：

```python
def test_step_now_is_monotonic() -> None:
    ...

def test_refresh_brooks_structure_state_writes_review_artifact(tmp_path: Path) -> None:
    ...
    assert report["status"] in {"ok", "partial_failure", "blocked"}
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q tests/test_refresh_brooks_structure_state_script.py
```

- [ ] **Step 3: 从 `22cb60f0` 最小恢复脚本**

要求：

- 保持 review artifact 产物落点
- 不引入新的 source ownership
- 所有时间使用显式 `--now` / artifact stamp 推导

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q tests/test_refresh_brooks_structure_state_script.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_brooks_structure_state.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_brooks_structure_state_script.py
git commit -m "feat(research): restore brooks structure refresh"
```

## 4. Task 3：恢复 commodity paper execution refresh（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_commodity_paper_execution_state.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_commodity_paper_execution_state_script.py`

- [ ] **Step 1: 写最小 failing tests**

至少锁住：

```python
def test_refresh_commodity_paper_execution_state_emits_brief_snapshot(tmp_path: Path) -> None:
    ...
    assert payload["brief_artifact"].endswith("hot_universe_operator_brief.json")
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q tests/test_refresh_commodity_paper_execution_state_script.py
```

- [ ] **Step 3: 从 `22cb60f0` 恢复脚本，保留最小闭包**

要求：

- 复用 `operator_context_sections.py` 历史依赖（如缺失则一并恢复）
- 输出必须能把 `brief_artifact` 指向 `*_hot_universe_operator_brief.json`
- 不要修改 panel/UI 逻辑

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q tests/test_refresh_commodity_paper_execution_state_script.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_commodity_paper_execution_state.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_commodity_paper_execution_state_script.py
git commit -m "feat(research): restore commodity paper execution refresh"
```

## 5. Task 4：恢复 cross-market operator state（RESEARCH_ONLY）

**Files:**
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_cross_market_operator_state.py`
- Create: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_cross_market_operator_state_script.py`

- [ ] **Step 1: 写最小 failing tests**

至少锁住：

```python
def test_resolve_hot_brief_source_falls_back_when_explicit_brief_was_pruned(tmp_path: Path) -> None:
    ...

def test_build_review_head_lane_marks_non_promotable_head_as_refresh_required() -> None:
    ...
```

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q tests/test_refresh_cross_market_operator_state_script.py
```

- [ ] **Step 3: 从 `22cb60f0` 恢复脚本**

要求：

- 仅恢复 source-owned 行为
- 输出 `*_cross_market_operator_state.json`
- 能给 `build_operator_task_visual_panel.py` 提供：
  - `operator_head_lane`
  - `review_head_lane`
  - `operator_repair_head_lane`
  - `operator_focus_slots`
  - `operator_action_queue`
  - `operator_repair_queue`

- [ ] **Step 4: 跑 tests 转绿**

```bash
PYTHONPATH=src pytest -q tests/test_refresh_cross_market_operator_state_script.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/refresh_cross_market_operator_state.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_refresh_cross_market_operator_state_script.py
git commit -m "feat(research): restore cross-market operator state refresh"
```

## 6. Task 5：把 refresh/browser smoke 切回 full-source 正常路径（RESEARCH_ONLY）

**Files:**
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_browser_smoke.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py`
- Modify: `/Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_browser_smoke_script.py`

- [ ] **Step 1: 写 failing test，要求 refresh 真正拉起 full-source chain**

至少锁住：

```python
def test_main_refreshes_panel_and_snapshot_into_public_and_dist(...) -> None:
    ...
    assert seen_cmds["refresh_cross_market_operator_state"] ...
```

或若采用 commodity refresh 间接产物路径，至少断言：

- 不再只依赖 degraded panel
- `hot_brief` / `cross_market` 在 refresh 正常路径下实际生成

- [ ] **Step 2: 跑 tests 确认红**

```bash
PYTHONPATH=src pytest -q \
  tests/test_build_operator_task_visual_panel_script.py \
  tests/test_run_operator_panel_refresh_script.py \
  tests/test_run_operator_panel_browser_smoke_script.py
```

- [ ] **Step 3: 最小修改 refresh 调度**

要求：

- 优先走 full-source chain
- degraded fallback 只保留为最后兜底
- `run_operator_panel_browser_smoke.py` 真实执行时应看到 full-source panel，而不是靠缺失源工件降级

- [ ] **Step 4: 跑 targeted tests 转绿**

```bash
PYTHONPATH=src pytest -q \
  tests/test_build_operator_task_visual_panel_script.py \
  tests/test_run_operator_panel_refresh_script.py \
  tests/test_run_operator_panel_browser_smoke_script.py
```

- [ ] **Step 5: Commit**

```bash
git add \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_refresh.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/build_operator_task_visual_panel.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts/run_operator_panel_browser_smoke.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_build_operator_task_visual_panel_script.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_refresh_script.py \
  /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests/test_run_operator_panel_browser_smoke_script.py
git commit -m "feat(research): restore full-source operator panel refresh chain"
```

## 7. Task 6：端到端验收（RESEARCH_ONLY）

**Files:**
- No new files required; produce runtime artifacts under `system/output/review/`

- [ ] **Step 1: 跑 source-chain focused Python 回归**

```bash
cd /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system
PYTHONPATH=src pytest -q \
  tests/test_build_hot_universe_operator_brief_script.py \
  tests/test_refresh_brooks_structure_state_script.py \
  tests/test_refresh_commodity_paper_execution_state_script.py \
  tests/test_refresh_cross_market_operator_state_script.py \
  tests/test_build_operator_task_visual_panel_script.py \
  tests/test_run_operator_panel_refresh_script.py \
  tests/test_run_operator_panel_browser_smoke_script.py
```

Expected: PASS

- [ ] **Step 2: 跑真实 refresh**

```bash
PYTHONPATH=src python3 scripts/run_operator_panel_refresh.py \
  --workspace /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
```

Expected:

- 生成新的 `*_operator_task_visual_panel.json/html`
- public/dist 同步成功
- summary 不再依赖 degraded 主头

- [ ] **Step 3: 跑真实 browser smoke**

```bash
PYTHONPATH=src python3 scripts/run_operator_panel_browser_smoke.py \
  --workspace /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1
```

Expected:

- `ok = true`
- `visible_markers` 含 geostrategy 4 brief
- 报告与截图落地 `system/output/review/`

- [ ] **Step 4: 人工验收点**

确认：

- operator panel `status` 不再因缺 `hot_brief` / `cross_market` 落入 degraded
- browser smoke 通过时 `expected_markers` 不再只有 section title
- full-source chain 恢复后，degraded fallback 仍保留但不再是正常路径

- [ ] **Step 5: Final Commit**

```bash
git add /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/scripts \
        /Users/jokenrobot/.config/superpowers/worktrees/fenlie/codex-tv-basis-arb-v1/system/tests
git commit -m "feat(research): restore operator source chain"
```

## 8. 风险与停止规则

- 该恢复闭包体量很大；若 Task 1 恢复后发现 helper 与当前 repo 结构严重冲突，先停在 Task 1，总结冲突点，不要继续盲推
- 若 Task 3/4 依赖更多已删除 builder（例如 remote_* / live_gate builder 的额外闭包），应新开子计划，不要在本计划内无限扩张
- 若 full-source 恢复两轮后仍无法改变 source-owned state 或 user-visible capability，按 stop rule 停止并汇报
