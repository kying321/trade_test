# Fenlie Skill / MCP Architecture

## Purpose

这份文档定义 Fenlie 的 **skill 层** 和 **MCP 层** 如何协同，但不越权进入 live source-of-truth。

目标不是“多装几个 MCP 就多走几条路径”，而是：

1. 把 MCP 放进明确分层；
2. 把 skill 变成稳定入口；
3. 把 live path 与 research/browser/knowledge path 严格隔离；
4. 把当前会话里真正可用、真正 blocked 的能力写清楚。

对应 source-owned 报告：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_skill_mcp_architecture_report.json`

对应 source-owned routing policy：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/config/skill_mcp_routing_policy.json`

## Layering

### 1. Source-Owned Runtime Layer

职责：

- execution
- queue ownership
- gate state
- live capital
- fund scheduling
- remote/live risk guard

原则：

- **最终 live execution path 不允许依赖 MCP**
- source artifact / guarded client / runtime script 才是 source-of-truth

典型 skill：

- `trade-live-exec-guard`
- `fenlie-order-risk-exec`
- `fenlie-remote-live-guard-diagnostics`

### 2. Source Review / Audit Layer

职责：

- source ownership
- cross-market refresh
- operator panel refresh
- daily checklist
- top-brief drift detection

原则：

- 优先本地 artifact / script
- MCP 只做补充诊断，不改 source state

典型 skill：

- `fenlie-source-ownership-review`
- `fenlie-cross-market-refresh-audit`
- `fenlie-operator-panel-refresh`
- `fenlie-daily-ops-checklist`

### 3. Research Enrichment Layer

职责：

- SIM_ONLY / RESEARCH_ONLY 数据补充
- 非 source-of-truth 的外部市场上下文
- 策略研究 sidecar

原则：

- 允许外部 MCP，但只能进 **research sidecar**
- 不能把 MCP 数据静默升格为 live 路由输入

当前推荐：

- `coingecko`：research-only enrichment
- `jin10`：research-only macro/calendar/news sidecar

典型 skill：

- `fenlie-intraday-strategy-research`
- `codex-autoresearch`

### 4. UI / Browser Surface Layer

职责：

- 前端页面验证
- 只读浏览器流量与页面行为诊断
- design-to-code

原则：

- 允许 browser/design MCP
- 禁止把浏览器会话当 execution source-of-truth

当前推荐：

- `playwright`
- `jshook`
- `figma`

### 5. Knowledge / Planning Layer

职责：

- 文档沉淀
- PM/issue/spec 同步
- 知识库 capture

原则：

- 必须先过 auth/connectivity health
- blocked 就明确写 blocked，不假装可用

当前推荐：

- `notion`
- `linear`

## Current MCP Topology

本地 Codex 配置路径：

- `/Users/jokenrobot/.codex/config.toml`

当前配置 MCP：

- `notion`
- `jshook`
- `coingecko`
- `playwright`
- `figma`
- `linear`

### Session-Validated Status

#### `coingecko`

- 状态：`verified_tools_available`
- 结论：可用于 research-only 数据补充链

#### `figma`

- 状态：`verified_auth_and_resources_available`
- 结论：可用于 UI / design-to-code / design context

#### `playwright`

- 状态：`configured_not_resource_driven`
- 结论：适合前端验证与浏览器 smoke test

#### `jshook`

- 状态：`verified_environment_and_search_available`
- 结论：适合 browser reverse engineering / read-only diagnostics
- 参考：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/JSHOOK_MCP_FENLIE_USAGE.md`

#### `notion`

- 状态：`blocked_auth_refresh_failed`
- 结论：当前 auth 失败，先不要把它纳入标准工作流

#### `linear`

- 状态：`blocked_initialize_failed`
- 结论：当前连接失败，先不要把它纳入标准工作流

## Health Gate Interpretation

- `green`：已经在当前会话完成健康验证，可进入其批准的非 live lane
- `yellow`：已配置但还缺一条 session smoke-check，不应直接升级成默认入口
- `red`：auth / initialize 失败，必须先修复，再谈 skill 路由

当前最高风险未决项不是 UI 或文档，而是：

- `notion` auth 失败
- `linear` initialize 失败

也就是说，knowledge/planning 层目前仍是 **blocked lane**。

## Skill Routing

### architecture_review

优先 skill：

- `fenlie-source-ownership-review`
- `fenlie-skill-mcp-governance`

MCP 策略：

- 允许做盘点与分层：`coingecko`, `figma`, `playwright`, `jshook`
- `notion` / `linear` 必须先过 health gate

### source_first_implementation

优先 skill：

- `fenlie-cross-market-refresh-audit`
- `fenlie-remote-live-guard-diagnostics`

MCP 策略：

- 可用于只读验证：`playwright`, `figma`
- 禁止把 MCP 放进最终 execution path

### ui_rendering

优先 skill：

- `fenlie-operator-panel-refresh`
- `fenlie-skill-mcp-governance`

MCP 策略：

- 允许：`playwright`, `figma`, `jshook`
- 只做只读验证、页面诊断、设计实现

### research_backtest

优先 skill：

- `fenlie-intraday-strategy-research`
- `codex-autoresearch`

MCP 策略：

- 允许：`coingecko`
- `notion` 仅在 auth 恢复后做知识沉淀
- 禁止把 MCP 数据直接升级成 source-of-truth 回测/live 输入

### live_guard_diagnostics

优先 skill：

- `fenlie-remote-live-guard-diagnostics`
- `fenlie-source-ownership-review`

MCP 策略：

- 允许只读浏览器诊断：`playwright`, `jshook`
- 禁止浏览器侧变更 queue ownership / execution state

## Hard Rules

1. live capital、order routing、execution queue、fund scheduling 的最终路径 **不允许依赖 MCP**
2. browser MCP 只能做 read-only diagnostics / reverse engineering
3. research MCP 只能做 sidecar enrichment，不能静默覆盖 source-owned 指标
4. blocked auth / blocked initialize 必须明确记为 blocked
5. skill 是入口，MCP 是能力补充；不要把 MCP 直接当工作流

## Jin10 MCP Research Sidecar

新增的 Jin10 MCP 集成采用 **research sidecar** 形态，而不是替换现有 source-owned Jin10 HTTP provider。

规则：

- 只允许进入 `research_enrichment / research_backtest`
- 只输出 review/research artifact
- 不允许直接覆盖 `PublicInternetResearchProvider` 产出的 source-owned 宏观字段
- 不允许进入 live routing / capital / queue mutation

Repo 内 runner：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_jin10_mcp_snapshot.py`

Repo 内 client：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/jin10_mcp_client.py`

默认 token 环境变量：

- `JIN10_MCP_BEARER_TOKEN`

默认 server URL：

- `https://mcp.jin10.com/mcp`

典型用法：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie
export JIN10_MCP_BEARER_TOKEN=...
python3 system/scripts/run_jin10_mcp_snapshot.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie \
  --tool-name calendar.latest \
  --tool-args-json '{"symbols":["XAUUSD"]}'
```

产物：

- `system/output/review/latest_jin10_mcp_snapshot.json`
- `system/output/review/latest_jin10_mcp_snapshot.md`

## Axios Public-Site Research Sidecar

Axios 接入采用 **public-site research sidecar**，只读取公开可抓取 sitemap/robots/news feed，不抓首页动态壳层，也不进入 live path。

规则：

- 只允许进入 `research_enrichment / research_backtest`
- 只输出 review artifact
- 只允许读取公开 sitemap / robots / news xml
- 不允许把站点抓取结果上升为 live/source-owned authority

Repo 内 runner：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_axios_site_snapshot.py`

Repo 内 client：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/axios_site_client.py`

产物：

- `system/output/review/latest_axios_site_snapshot.json`
- `system/output/review/latest_axios_site_snapshot.md`

## Unified External Intelligence

Jin10 与 Axios 之上增加统一 read-model：

- 汇总 runner：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_external_intelligence_snapshot.py`
- 单入口 refresh runner：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_external_intelligence_refresh.py`

规则：

- unified snapshot 只能聚合 sidecar，不得反向改写各 source-owned provider
- 只有 `status in {ok, partial}` 且 `ok != false` 的 sidecar 才能计入 active sources
- blocked auth / blocked initialize 必须继续保持 blocked，不得因 latest artifact 存在而被误计为 active
- refresh runner 允许重建 dashboard frontend snapshot，但仍属于 `RESEARCH_ONLY`
- 当被 `run_operator_panel_refresh.py` 作为上游步骤调用时，应使用 `--skip-dashboard-snapshot`，避免重复构建 frontend snapshot
- `run_operator_panel_refresh.py` 对该步骤采用 degrade-only 语义：外部情报 refresh 失败不能阻断 panel/dashboard 主刷新

产物：

- `system/output/review/latest_external_intelligence_snapshot.json`
- `system/output/review/latest_external_intelligence_snapshot.md`
- `system/output/review/latest_external_intelligence_refresh.json`
- `system/output/review/latest_external_intelligence_refresh.md`

## Recommended Operating Pattern

### 当新增 MCP 时

1. 先更新 `/Users/jokenrobot/.codex/config.toml`
2. 用 `fenlie-skill-mcp-governance` 盘点可见性、auth、routing layer
3. 更新本架构文档
4. 生成新的 source-owned architecture report
5. 只有通过 health gate 的 MCP，才进入对应 lane

### 当某个 MCP auth 漂移时

1. 标记 blocked
2. 从默认路由里降级
3. 不要让下游 skill 假定它仍可用

## New Governance Entry Point

新增本地 skill：

- `/Users/jokenrobot/.codex/skills/fenlie-skill-mcp-governance/SKILL.md`

用途：

- 新 MCP 安装后的盘点
- MCP auth / visibility / routing 漂移诊断
- skill/MCP 分层维护
- 输出 source-owned architecture report

## Consumer Rule

后续如果有人问：

- “现在有哪些 MCP 真的能用？”
- “哪个 skill 应该配哪个 MCP？”
- “哪些 MCP 可以进入研究链，哪些绝不能进 live path？”

先读：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_skill_mcp_architecture_report.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_SKILL_MCP_ARCHITECTURE.md`

不要只看 `.codex/config.toml` 就假设能力已经健康可用。
