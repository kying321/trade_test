# Fenlie Ops / Workspace 双域上下文条拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Fenlie 前端顶部上下文层拆成真正的“运行域”与“研究域”两套信息架构，减少操作终端与研究工作区的首屏重复与跨域动作混放。

**Architecture:** 继续保留 `OpsContextStrip` 与 `WorkspaceContextStrip` 两个独立组件，但重排它们的字段 contract、CTA 分组与视觉语义。实现只触及 `App.tsx`、`index.css`、`App.test.tsx`，尽量不动 `TerminalPanels.tsx` / `WorkspacePanels.tsx` 主体内容，以降低 source-owned 读模型风险。

**Tech Stack:** Vite 7、React 19、TypeScript、React Router、Vitest、Playwright smoke

---

## 0. 约束

- change class 固定为 `RESEARCH_ONLY`
- 不触碰 live capital、execution queue、order routing、fund scheduling
- 不修改 snapshot schema、不新增后端字段
- 不使用 subagent，除非用户明确再次授权

## 1. 目标文件结构

### Modify

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.tsx`
  - 重排 `OpsContextStrip`
  - 重排 `WorkspaceContextStrip`
  - 必要时抽出 strip 内动作构建 helper

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/index.css`
  - 为 ops/workspace 两套 strip 拉开视觉层次
  - 调整卡片与按钮的域级 accent

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx`
  - 新增双域 strip 差异化断言
  - 更新旧断言以适配新结构

### Optional Modify（仅当必须）

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/components/WorkspacePanels.tsx`
  - 仅为减少首屏重复的最小文案收口

## 2. Task 1：用测试先锁定双域职责边界

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx`

- [ ] **Step 1: 为 ops strip 写失败断言**

新增测试目标：

- `#/terminal/public` 下：
  - strip 包含 `运行上下文` 或新的运行域 title
  - strip **不包含** `查看工件池`
  - strip **不包含** `焦点工件`

- [ ] **Step 2: 为 workspace strip 写失败断言**

新增测试目标：

- `#/workspace/contracts` 或 `#/workspace/backtests` 下：
  - strip 包含阶段专属文案
  - strip 包含 `焦点工件`
  - strip **不包含** `查看运维面板`
  - strip **不包含** `视图断言`

- [ ] **Step 3: 跑单测验证它们先红**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx -t "ops|workspace|context strip" --reporter=dot
```

Expected: FAIL，说明当前结构尚未满足双域纯化。

## 3. Task 2：重构 `OpsContextStrip`

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.tsx`

- [ ] **Step 1: 调整 ops 头部 copy**

把当前：

- `运行上下文`
- `公开终端 / 内部终端`
- `只读模式 / redaction`

收口成更明确的运行域 copy，例如：

- kicker：`运行域 / 视图断言`
- title：`公开终端 → 公开终端`
- subtitle：`只读快照 / 公开摘要`

- [ ] **Step 2: 重排 ops 三卡结构**

目标顺序：

1. 视图断言
2. 运行态
3. 门禁与告警

并移除“穿透焦点占一整卡”的当前实现。

- [ ] **Step 3: 下沉 `panel / section / row` 焦点**

把 focus summary 改成次级 meta：

- 可放在动作区上方的小字
- 或放入 badge / subline

但不再作为主卡之一。

- [ ] **Step 4: 清理 ops 动作区**

保留：

- 清除行焦点（条件显示）
- 回总览
- 回终端主层
- 查看运维面板

删除：

- `查看工件池`

- [ ] **Step 5: 跑相关单测**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx -t "renders terminal and workspace routes with the new TS shell|ops" --reporter=dot
```

Expected: PASS

## 4. Task 3：重构 `WorkspaceContextStrip`

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.tsx`

- [ ] **Step 1: 保留阶段专属 head copy**

继续使用：

- `buildWorkspacePageMeta(section).headerBarKicker`
- `headerBarTitle`
- `headerBarSubtitle`

但明确 strip 是研究链路头，不承载运行门禁语义。

- [ ] **Step 2: 重排 workspace 三卡**

目标顺序：

1. 研究对象
2. 验证窗口
3. 证据链

其中：

- 研究对象：title / decision / path
- 验证窗口：section / search scope / current note
- 证据链：payload key / artifact id / group / handoff hint

- [ ] **Step 3: 清理 workspace 动作区**

优先保留：

- 查看原始层
- 回工件池
- 查看回测池或契约层（按当前 section 最小补足）
- source-owned handoff links

明确不加入：

- `查看运维面板`
- guardian / canary / promotion 类动作

- [ ] **Step 4: 跑相关单测**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx -t "renders stage-specific workspace context copy|workspace" --reporter=dot
```

Expected: PASS

## 5. Task 4：拉开视觉层次

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/index.css`

- [ ] **Step 1: 为 ops strip 增强运行域 accent**

建议：

- 更强的 `surface-accent`
- `ops-context-card` 使用更冷的 tile 背景
- CTA 更偏控制台按钮质感

- [ ] **Step 2: 为 workspace strip 增强研究域 accent**

建议：

- `workspace-context-card` 使用更柔和的 panel-soft / graphitic 背景
- CTA 更偏证据/检索跳转风格

- [ ] **Step 3: 保持 light / dark / system 兼容**

不新增硬编码暗底，继续走主题 token。

- [ ] **Step 4: 跑 UI 相关测试**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx src/app-shell/AppShell.test.tsx src/components/ui-kit.test.tsx --reporter=dot
```

Expected: PASS

## 6. Task 5：压低首屏重复

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/components/WorkspacePanels.tsx`（仅当必要）
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx`

- [ ] **Step 1: 检查 backtests / contracts / raw 首卡是否仍复写 strip 信息**

如果首卡标题或 meta 与 strip 语义重复，则做最小调整：

- strip 负责“上下文”
- 首卡负责“本页主叙事”

- [ ] **Step 2: 若需要，最小修改页面首卡文案**

例如：

- ops 首卡偏“运行摘要”
- workspace 首卡偏“研究摘要”

- [ ] **Step 3: 跑重复度相关测试**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx -t "singular|stage-specific|workspace" --reporter=dot
```

Expected: PASS

## 7. Task 6：最小验证集

**Files:**
- Validate only

- [ ] **Step 1: 跑完整前端单测**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx --reporter=dot
```

Expected: PASS

- [ ] **Step 2: 跑 TypeScript**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx tsc --noEmit
```

Expected: PASS

- [ ] **Step 3: 跑 workspace 路由 smoke**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run smoke:workspace-routes
```

Expected: PASS，并生成新的 review artifact。

- [ ] **Step 4: 如用户要求公网同步，再单独部署**

Run:

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
env -u HTTP_PROXY -u HTTPS_PROXY npm run cf:deploy
```

Expected: PASS

## 8. 完成定义

满足以下条件才算完成：

1. ops strip 不再混入 research CTA
2. workspace strip 不再混入 ops 运行门禁语义
3. backtests / contracts / raw 首屏重复文案继续下降
4. `App.test.tsx` 全绿
5. `tsc --noEmit` 通过
6. `smoke:workspace-routes` 通过

## 9. 备注

- 按仓库规则，本计划默认采用 **Inline Execution**，不启用 subagent。
- 若用户后续明确要求并行执行，再拆成 disjoint write scopes。
