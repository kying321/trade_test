# Fenlie Dashboard Engineering Runbook

> 状态：工程使用手册 / 上架前操作说明  
> 适用范围：`/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web`  
> 目标读者：需要本地启动、回归测试、公开验收、Cloudflare 发布、上线回滚的工程操作者  
> change class：`DOC_ONLY`

---

## 1. 文档目的

这份文档不是产品介绍，而是 **工程操作手册**。

它解决四类问题：

1. **怎么在本地把 Fenlie dashboard 跑起来**
2. **怎么确认只读公开面和本地内部面边界没有被破坏**
3. **怎么在发布前完成最小充分验证**
4. **怎么把前端发布到 Cloudflare Pages / 根域代理链，并在异常时快速回滚**

如果你正在做上架前整理，默认按本文档顺序执行，不要凭记忆跳步骤。

---

## 2. 系统边界与原则

### 2.1 当前前端定位

Fenlie dashboard 当前是一个 **只读前端控制台**，主要职责：

- 暴露公开面 / 内部面 / workspace 多页面信息结构
- 消费只读快照与 review 工件
- 进行公开入口验收与 workspace 路由刷新验证

### 2.2 不属于前端的职责

以下内容 **不属于这个前端工程的变更范围**：

- live capital
- order routing
- execution queue
- fund scheduling
- source-of-truth 的 live gate 决策

这意味着：

- 前端只消费只读快照
- 前端 smoke / acceptance 失败，只能说明**展示面、路由、发布或剥离边界有问题**
- 不应把前端验证结果误当成 live path 放行依据

### 2.3 source-of-truth 规则

前端页面里看到的这些概念：

- queue
- gate
- blocker
- review state
- OOS / backtest metrics

都必须来自 snapshot / review artifact，而不是在 UI 顶层二次编造。

---

## 3. 关键目录与文件

## 3.1 工作区根目录

```bash
/Users/jokenrobot/Downloads/Folders/fenlie
```

## 3.2 前端工程目录

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
```

## 3.3 关键只读数据输出

公开快照：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_snapshot.json
```

内部快照：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_internal_snapshot.json
```

## 3.4 关键脚本

前端快照构建：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py
```

运维面板刷新与同步：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py
```

公开入口烟测：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py
```

workspace 多路由 smoke：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_workspace_artifacts_smoke.py
```

聚合公开验收：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_acceptance.py
```

## 3.5 Cloudflare 配置

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/wrangler.jsonc
```

---

## 4. 依赖与环境前提

你至少需要：

- Node.js / npm
- Python 3
- 可运行 Playwright / Chromium 的本地环境
- Wrangler 登录态（用于 Cloudflare 发布）

## 4.1 建议先确认

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
node -v
npm -v
python3 --version
npx wrangler whoami
```

如果 `npx wrangler whoami` 失败，先处理 Cloudflare 登录，不要直接进入发布。

---

## 5. package scripts 总表

当前 `package.json` 中和工程操作最相关的脚本如下：

### 5.1 本地开发

```bash
npm run dev
```

行为：

1. 刷新 operator panel
2. 刷新 dashboard snapshot
3. 启动 Vite 本地开发服务（`127.0.0.1:5173`）

### 5.2 构建

```bash
npm run build
```

行为：

1. 刷新 operator panel
2. 刷新 dashboard snapshot
3. 生成 `dist`

### 5.3 内部快照剥离

```bash
npm run strip:internal
```

用途：

- 发布到 Pages 前，删除 `dist/data/fenlie_dashboard_internal_snapshot.json`
- 防止内部只读快照进入公开分发面

### 5.4 聚合公开验收

```bash
npm run verify:public-surface
```

等价脚本：

```bash
python3 ../../scripts/run_dashboard_public_acceptance.py --workspace ../../..
```

### 5.5 单一 release-checklist 验证

```bash
npm run verify:release-checklist
```

用途：

- 一次性执行 `build -> test -> tsc -> workspace routes smoke -> public acceptance`
- 作为当前 dashboard 上架前最小充分验证的单一入口
- 默认对 smoke / public acceptance 走 `--skip-build`，避免在同一条链里重复 build

### 5.6 workspace 多路由 smoke

```bash
npm run smoke:workspace-routes
```

公开 workspace 路由矩阵烟测，默认验证：

- `#/overview`
- `#/workspace/artifacts`
- `#/workspace/alignment`（公开锁定态）
- `#/workspace/backtests`
- `#/workspace/raw`
- `#/workspace/contracts`

### 5.7 internal alignment 浏览器 smoke

```bash
npm run smoke:alignment-internal
```

用途：

- 真实浏览器验证 `#/workspace/alignment?view=internal`
- 确认只拉取 internal snapshot
- 确认页面能看到当前 projection headline / 顶部 event / 顶部 action

最近会生成如下 review 工件：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_dashboard_internal_alignment_browser_smoke.json
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_dashboard_internal_alignment_browser_smoke.png
```

### 5.7.1 internal alignment manual probe 浏览器 smoke

```bash
npm run smoke:alignment-internal-manual-probe
```

用途：

- 临时注入一条高优先级 manual feedback probe 到 `conversation_feedback_events_internal.jsonl`
- 自动执行 refresh，让 probe 进入 internal projection / internal snapshot
- 用真实浏览器验证 `#/workspace/alignment?view=internal&page_section=alignment-summary`
- 完成后自动恢复原 manual 文件并再次 refresh，避免污染当前主链

最近会生成如下 review 工件：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_dashboard_internal_alignment_manual_probe_browser_smoke.json
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_dashboard_internal_alignment_manual_probe_browser_smoke.png
```

### 5.8 发布 internal autopublish 输入

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
echo '{"feedback_id":"alignment_projection_rollout","headline":"将高价值对话反馈投射到内部对齐页","summary":"先验证 internal 对齐页，再收紧 fallback。","recommended_action":"刷新 internal snapshot 并检查 alignment 页面。"}' | npm run feedback:publish -- --now 2026-03-22T04:05:00Z
```

用途：

- 用标准命令写入 `latest_conversation_feedback_autopublish_internal.json`
- 自动补齐：
  - `source=auto_session`
  - `status=active`
  - `created_at_utc`
- 自动剥离 raw 字段：
  - `raw_transcript`
  - `message_text`
  - `messages`
  - `raw_messages`
  - `transcript`

写入目标：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_conversation_feedback_autopublish_internal.json
```

写入完成后，执行：

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

这样 internal projection 与 internal snapshot 会在下一轮刷新中同步更新。

### 5.9 发布 + refresh 一条链

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
echo '{"feedback_id":"alignment_projection_rollout","headline":"将高价值对话反馈投射到内部对齐页","summary":"用单条命令发布 autopublish 并刷新 internal projection。","recommended_action":"打开 /workspace/alignment?view=internal 检查 headline、events、actions。"}' | npm run feedback:publish-refresh -- --now 2026-03-22T04:15:00Z
```

用途：

- 用单条命令完成**发布 + refresh 一条链**
- 第一段仍走 `publish_conversation_feedback_autopublish_internal.py`
- 第二段自动执行 `run_operator_panel_refresh.py --workspace ../../..`
- 适合快速把当前高价值反馈投射进 internal snapshot，供下一轮轮询或手动刷新读取

执行完成后，以下只读工件会同步更新：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_conversation_feedback_autopublish_internal.json
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_conversation_feedback_projection_internal.json
/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_internal_snapshot.json
```

### 5.10 发布 manual 结构化反馈

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
echo '{"feedback_id":"manual_alignment_fix","headline":"人工结构化反馈：补齐 manual 轨","summary":"manual 轨需要标准写入口，避免直接手改 jsonl。","recommended_action":"提供 publish-manual 与 publish-manual-refresh。"}' | npm run feedback:publish-manual -- --now 2026-03-22T06:40:00Z
```

用途：

- 追加写入人工结构化事件，不覆盖历史 manual 轨
- 目标文件固定为 `conversation_feedback_events_internal.jsonl`
- 自动补齐：
  - `source=manual`
  - `status=active`
  - `created_at_utc`
- 同样会自动剥离 raw 字段，避免把聊天原文写入 internal projection 源

写入目标：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/conversation_feedback_events_internal.jsonl
```

如果希望写入后立刻刷新 internal projection / snapshot，使用：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
echo '{"feedback_id":"manual_alignment_fix","headline":"人工结构化反馈：补齐 manual 轨","summary":"manual 轨需要标准写入口，避免直接手改 jsonl。","recommended_action":"提供 publish-manual 与 publish-manual-refresh。"}' | npm run feedback:publish-manual-refresh -- --now 2026-03-22T06:41:00Z
```

这样会串行执行：

1. `publish_conversation_feedback_manual_internal.py`
2. `run_operator_panel_refresh.py --workspace ../../..`

适合把人工审核结论快速投射到 `#/workspace/alignment?view=internal`。

兼容别名：

```bash
npm run smoke:workspace-artifacts
```

### 5.7 测试

```bash
npm test
```

### 5.8 Cloudflare 发布

```bash
npm run cf:deploy
```

内部实际顺序：

1. `npm run build`
2. `npm run strip:internal`
3. `npx wrangler pages deploy dist --project-name fenlie-dashboard-web --branch main --commit-dirty=true`

---

## 6. 推荐操作流（本地到上架）

如果你要做一次完整的上架前工程整理，按下面顺序：

0. **可直接先跑单一入口：`npm run verify:release-checklist`**
1. **刷新数据与 panel**
2. **本地开发预览**
3. **单元 / TS 回归**
4. **workspace routes smoke**
5. **public topology smoke**
6. **聚合公开验收**
7. **Cloudflare deploy**
8. **根域 / Pages / 旧 OpenClaw / gateway 线上复核**

下面展开写。

---

## 7. 第一步：刷新只读数据与面板

### 7.1 刷新前端快照

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie \
  --public-dir /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public
```

### 7.2 刷新 operator panel 并同步 public

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_operator_panel_refresh.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

用途：

- 把 operator panel 和 dashboard snapshot 刷到 public
- 同步非敏感产物到 dist 链路

### 7.3 日常更简单的入口

通常不需要手敲上面两条，直接：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run refresh:data
npm run refresh:panel
```

---

## 8. 第二步：本地开发预览

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm install
npm run dev
```

启动后默认：

- Host：`127.0.0.1`
- Port：`5173`

本地访问：

```bash
http://127.0.0.1:5173/
```

推荐手动检查页面：

- `#/overview`
- `#/terminal/public`
- `#/terminal/internal`
- `#/workspace/artifacts`
- `#/workspace/backtests`
- `#/workspace/raw`
- `#/workspace/contracts`

---

## 9. 第三步：本地工程回归

### 9.1 TypeScript + 单测

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx vitest run src/App.test.tsx src/adapters/read-model.test.ts src/utils/dictionary.test.ts --reporter=dot
npx tsc --noEmit
```

你也可以直接跑全部：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm test
npx tsc --noEmit
```

### 9.2 推荐判断标准

- `vitest` 全绿
- `tsc --noEmit` 无报错
- 不接受“只是 smoke 通过但 TS 有错”

---

## 10. 第四步：workspace 多路由 smoke

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run smoke:workspace-routes
```

这条命令会：

1. 先执行 `npm run build`
2. 临时拉起静态服务
3. 用真实浏览器依次访问：
   - `#/overview`
   - `#/workspace/artifacts`
   - `#/workspace/backtests`
   - `#/workspace/raw`
   - `#/workspace/contracts`

### 10.1 它重点验证什么

- `#/overview` 是否仍能看到 `研究主线摘要 / hold24_zero`
- 默认是否仍锚定 `price_action_breakout_pullback`
- 路由切换是否触发新的 public snapshot GET
- 是否错误拉取内部快照
- contracts 页 section deep-link 是否还能展开子命令证据
- `#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow` 是否仍能命中：
  - `intraday_orderflow_blueprint`
  - `intraday_orderflow_research_gate_blocker`

### 10.2 常用可选参数

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_workspace_artifacts_smoke.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie \
  --host 127.0.0.1 \
  --port 0 \
  --timeout-seconds 20
```

### 10.3 典型输出工件

输出到：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_dashboard_workspace_routes_browser_smoke.json
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_dashboard_workspace_routes_browser_smoke.png
```

---

## 11. 第五步：公开入口拓扑烟测

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

默认验证目标：

- `https://fuuu.fun`
- `https://fenlie.fuuu.fun`
- `http://43.153.148.242:3001`
- `http://43.153.148.242:8787`

### 11.1 期望

根域：

- 返回 Fenlie 页面标题
- 返回 `x-fenlie-public-entry: root-nav-proxy`

Pages：

- 返回 Fenlie 页面标题

旧 OpenClaw：

- 返回 `OpenClaw Control`

gateway/API：

- `404 Not Found` 属于**当前预期**

### 11.2 严格 TLS 与 fallback

默认脚本支持本地重定向域场景下的 TLS fallback。

如果你要严格验证，不允许不安全 TLS 回退：

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie \
  --timeout-seconds 5
```

如果你明确需要允许不安全 TLS fallback：

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie \
  --allow-insecure-tls-fallback
```

当前 topology smoke 除了 JSON 报告，还会额外落四张公开路由截图，便于审计根域与 Pages 回退入口是否都真实渲染出 `#/overview` 与 `#/workspace/contracts` 的关键 marker：

- `*_dashboard_public_topology_root_overview_browser.png`
- `*_dashboard_public_topology_pages_overview_browser.png`
- `*_dashboard_public_topology_root_contracts_browser.png`
- `*_dashboard_public_topology_pages_contracts_browser.png`

其中 contracts 路由要求真实浏览器同时看到：

- `公开面验收`
- `root overview 截图`
- `pages overview 截图`
- `root contracts 截图`
- `pages contracts 截图`
- `公开快照拉取次数`
- `内部快照拉取次数`

---

## 12. 第六步：聚合公开验收

如果要一次性把“根域/Pages/旧 OpenClaw/gateway + workspace 五页面（含 `#/overview`）”一起验掉：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run verify:public-surface -- --skip-workspace-build
```

如果要强制 workspace routes smoke 在聚合里重新 build：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run verify:public-surface
```

如果要关闭 topology TLS fallback：

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_acceptance.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie \
  --strict-topology-tls
```

### 12.1 产物

聚合验收会生成 review 工件：

```bash
/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*_dashboard_public_acceptance.json
```

建议把它作为**上架前总报告**保存。

### 12.2 当前聚合硬门槛

除了 root/pages/openclaw/gateway 入口拓扑与 workspace 五页面真实浏览器 smoke，当前聚合验收还会硬校验：

- `artifacts_filter_assertion.route`
  - `#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow`
- `artifacts_filter_assertion.active_artifact`
  - `intraday_orderflow_blueprint`
- `artifacts_filter_assertion.visible_artifacts`
  - `intraday_orderflow_blueprint`
  - `intraday_orderflow_research_gate_blocker`

如果这组 orderflow research-only 断言缺失或值漂移，`verify:public-surface` 会直接失败，而不是继续报 `ok=true`。

---

## 13. 第七步：Cloudflare Pages 发布

### 13.1 先检查账号

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run cf:whoami
```

说明：该脚本已内置清理 `HTTP_PROXY` / `HTTPS_PROXY`，避免本机代理变量影响 Wrangler。

### 13.2 正式发布

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run cf:deploy
```

说明：该脚本已内置清理 `HTTP_PROXY` / `HTTPS_PROXY`，并保留 `build -> strip:internal -> wrangler pages deploy` 的固定顺序。

### 13.3 这条命令做了什么

1. build
2. strip internal snapshot
3. 发布到 Cloudflare Pages production

### 13.4 当前对外入口结构

- 主入口：`https://fuuu.fun`
- Pages 回退：`https://fenlie.fuuu.fun`
- 旧 OpenClaw：`http://43.153.148.242:3001`
- gateway/API：`http://43.153.148.242:8787`

注意：

- `fuuu.fun` 当前通过 Cloudflare Worker route 代理到 Fenlie 公开入口
- `fenlie.fuuu.fun` 是 Pages 稳定回退面
- `8787` 不是前端站点

---

## 14. 第八步：发布后复核

发布后不要只看 Pages deploy 成功，要做 **入口层复核**。

### 14.1 手动检查

浏览器打开：

- [https://fuuu.fun](https://fuuu.fun)
- [https://fenlie.fuuu.fun](https://fenlie.fuuu.fun)
- `http://43.153.148.242:3001`

### 14.2 自动复核

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

和：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run verify:public-surface -- --skip-workspace-build
```

### 14.3 发布后重点观察

- 根域标题是否还是 Fenlie
- 根域头部是否仍有 `x-fenlie-public-entry: root-nav-proxy`
- `fenlie_dashboard_internal_snapshot.json` 是否未进入公开发布面
- contracts 页 topology / workspace routes 是否仍可正常打开

---

## 15. 上架前最小检查清单

建议按下面 checklist 勾：

### 15.1 代码与构建

- [ ] `npx vitest run src/App.test.tsx src/adapters/read-model.test.ts src/utils/dictionary.test.ts --reporter=dot`
- [ ] `npx tsc --noEmit`
- [ ] `npm run build`

### 15.2 页面与路由

- [ ] `npm run smoke:workspace-routes`
- [ ] `#/overview` 正常
- [ ] `#/workspace/artifacts` 正常
- [ ] `#/workspace/backtests` 正常
- [ ] `#/workspace/raw` 正常
- [ ] `#/workspace/contracts` 正常

### 15.3 公开入口

- [ ] `run_dashboard_public_topology_smoke.py` 通过
- [ ] `https://fuuu.fun` 正常
- [ ] `https://fenlie.fuuu.fun` 正常
- [ ] `http://43.153.148.242:3001` 正常
- [ ] `http://43.153.148.242:8787` 维持预期 `404`

### 15.4 公开面剥离边界

- [ ] `npm run cf:deploy` 前已执行 `strip:internal`
- [ ] 公开面不暴露 `fenlie_dashboard_internal_snapshot.json`
- [ ] public acceptance 工件已生成

---

## 16. 常见故障与处理

## 16.1 `workspace-context-strip` 或页面元素找不到

常见原因：

- 刚改了路由结构 / query 逻辑
- contracts / workspace 页面 DOM 层级发生变化
- 测试断言仍依赖旧文案

处理：

1. 单跑对应 vitest case
2. 先确认是真回归还是旧断言失效
3. 再看 smoke 是否也失败

## 16.2 `旧 OpenClaw` / `Fenlie 主入口` 多重命中

这通常不是产品错误，而是 **同一文案在 topology 节点与 flow rail 中重复出现**。

测试写法上不要再假设唯一命中，优先使用：

- `getAllByText(...)`
- 容器 scoped query

## 16.3 topology smoke 因 TLS 失败

如果是本地映射/重定向环境：

- 先确认是否需要 `--allow-insecure-tls-fallback`

如果是正式发布环境：

- 优先修复证书/代理链
- 不建议靠 fallback 掩盖正式发布问题

## 16.4 workspace smoke 失败但 build 正常

说明问题更可能在：

- 路由刷新逻辑
- snapshot fetch 次数
- internal/public 边界
- page_section deep-link 展开逻辑

先看生成的：

- `*_dashboard_workspace_routes_browser_smoke.json`
- `*_dashboard_workspace_routes_browser_smoke.png`

## 16.5 Pages 正常但根域异常

这通常说明：

- Pages deploy 没问题
- 问题在 Cloudflare Worker route / root proxy

优先排查：

- `fuuu.fun/*` route
- `fenlie-root-nav-proxy`
- `x-fenlie-public-entry` header

---

## 17. 回滚手册

## 17.1 Cloudflare Pages 回滚

如果需要删除 Pages 项目：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx wrangler pages project delete fenlie-dashboard-web
```

## 17.2 根域代理回滚

移除或回滚以下 Worker / route：

- Worker：`fenlie-root-nav-proxy`
- 路由：`fuuu.fun/*`

## 17.3 云端静态站回滚

已知备份：

```bash
/home/ubuntu/openclaw-system/dashboard/web/dist.bak_20260319T082355Z
```

## 17.4 本地配置回滚

如需移除 Pages 配置，删除：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/wrangler.jsonc`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/_headers`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/_redirects`

---

## 18. 推荐实操序列（可直接照抄）

### 18.0 单命令版（当前推荐）

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm install
npm run verify:release-checklist
```

### 18.1 本地验证版

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm install
npm run build
npx vitest run src/App.test.tsx src/adapters/read-model.test.ts src/utils/dictionary.test.ts --reporter=dot
npx tsc --noEmit
npm run smoke:workspace-routes
python3 ../../scripts/run_dashboard_public_topology_smoke.py --workspace ../../..
npm run verify:public-surface -- --skip-workspace-build
```

### 18.2 正式发布版

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm install
npm run build
npx vitest run src/App.test.tsx src/adapters/read-model.test.ts src/utils/dictionary.test.ts --reporter=dot
npx tsc --noEmit
npm run smoke:workspace-routes
python3 ../../scripts/run_dashboard_public_topology_smoke.py --workspace ../../..
npm run verify:public-surface -- --skip-workspace-build
npm run cf:whoami
npm run cf:deploy
python3 ../../scripts/run_dashboard_public_topology_smoke.py --workspace ../../..
npm run verify:public-surface -- --skip-workspace-build
```

---

## 19. 产物归档建议

做上架前整理时，至少保留这几类 review 工件：

1. `*_dashboard_workspace_routes_browser_smoke.json`
2. `*_dashboard_workspace_routes_browser_smoke.png`
3. `*_dashboard_public_acceptance.json`
4. `*_dashboard_public_topology_smoke.json`（如果单独执行）

建议把“最近一次通过”的这几份路径整理进 release note。

---

## 20. 相关文档

现有发布/入口事实文档：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md`

持久记忆与常用命令：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`

---

## 21. 结论

如果你只想要一句话版本：

> **先 build + test + tsc + workspace smoke，再跑 topology smoke 和 public acceptance，最后再 deploy；deploy 后必须再次复核根域、Pages 回退和 workspace 五页面。**
