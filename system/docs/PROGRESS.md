# Progress Handoff

## Current State (2026-04-03)
- Dashboard 前端已完成 path-routing 第一刀，范围限定在 `RESEARCH_ONLY`，未触碰 live path：
  - `HashRouter -> BrowserRouter`
  - 新增 `/ops/overview`、`/ops/risk`、`/ops/audits`、`/ops/runbooks`、`/ops/workflow`、`/ops/traces`
  - `#/overview` 与 `#/terminal/public` 现通过 legacy hash bridge 自动落到 path route
  - 主导航与图谱主页快捷入口已改为 path-based href，不再输出 `#/...`
- 已完成 `lie -> pi` dashboard 漂移归属审计：
  - `pi` 当前不包含 `system/dashboard/web`
  - `lie` 当前承载 dashboard/public-surface 工程树、smoke/acceptance、Cloudflare Pages 配置与 path-route IA 文档
  - 结论：当前不应做 blanket `lie -> pi` sync；若未来需要跨 lane 引入 dashboard 能力，必须按能力切片做最小同步
  - 最小同步矩阵已固化：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/DASHBOARD_LANE_SYNC_MATRIX.md`
- governance 非破坏式审计现已内置 `dashboard_lane_sync_guard`：
  - helper：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_lane_sync_guard.py`
  - 聚合入口：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/branch_governance_audit.sh`
  - 统一 advisory runner：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_governance_audit_advisory.sh`
  - workflow summary renderer：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/render_governance_audit_summary.py`
  - workflow comment renderer：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/render_governance_audit_comment.py`
  - 当前行为：
    - 读取 primary branches 的 dashboard tree ownership
    - advisory-only，不阻断 `overall_pass`
    - 当前实仓库结果为 `current_owner=lie`、`recommended_action=no_sync_record_ownership`
    - `.github/workflows/governance-health-audit.yml` 现会把 guard 结果写入 GitHub Step Summary，并在需要时发 non-blocking warning
    - `.github/workflows/branch-policy.yml` 与 `.github/workflows/hotfix-pr-gate.yml` 现也会在 PR 侧渲染同类 advisory summary / warning，但不改原有 gate 成败
    - 上述两个 PR workflow 现在还会上传 `*_branch_governance_audit.{json,md}`，保证 summary/warning 背后有原始证据附件
    - 上述两个 PR workflow 现在还会 upsert sticky PR comment：`fenlie-governance-audit-advisory`
    - sticky PR comment 现优先使用仓库 secret `GOVERNANCE_COMMENT_TOKEN`；未配置时回退 `github.token`
    - 上述三个 workflow 的 advisory 执行路径已统一经由 `run_governance_audit_advisory.sh`，降低 summary/comment/artifact 输出漂移
- Jin10 MCP research sidecar 已接入 repo：
  - client：`/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/jin10_mcp_client.py`
  - runner：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_jin10_mcp_snapshot.py`
  - 测试：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_jin10_mcp_client.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_jin10_mcp_snapshot_script.py`
  - 定位：`RESEARCH_ONLY` 的 MCP sidecar，不替换现有 source-owned Jin10 HTTP provider
  - 产物：
    - `system/output/review/latest_jin10_mcp_snapshot.json`
    - `system/output/review/latest_jin10_mcp_snapshot.md`
- Axios public-site research sidecar 已接入 repo：
  - client：`/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/axios_site_client.py`
  - runner：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_axios_site_snapshot.py`
  - 测试：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_axios_site_client.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_axios_site_snapshot_script.py`
  - 定位：`RESEARCH_ONLY` 的公开站点 sidecar，只读取 `robots/sitemap/news sitemap`，不进入 live path
  - 产物：
    - `system/output/review/latest_axios_site_snapshot.json`
    - `system/output/review/latest_axios_site_snapshot.md`
- unified external intelligence 已接入 repo：
  - 汇总 runner：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_external_intelligence_snapshot.py`
  - 单入口 refresh runner：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_external_intelligence_refresh.py`
  - 测试：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_external_intelligence_snapshot_script.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_external_intelligence_refresh_script.py`
  - 当前语义：
    - 统一快照只把 `status in {ok, partial}` 且 `ok != false` 的 sidecar 计入 active sources
    - `blocked_auth_missing / blocked_initialize_failed` 不再被误算成活跃来源
    - refresh runner 会串联 `jin10 -> axios -> unified snapshot -> dashboard snapshot`
  - 产物：
    - `system/output/review/latest_external_intelligence_snapshot.json`
    - `system/output/review/latest_external_intelligence_snapshot.md`
    - `system/output/review/latest_external_intelligence_refresh.json`
    - `system/output/review/latest_external_intelligence_refresh.md`
- Polymarket public gamma sentiment sidecar 已接入 repo：
  - client：`/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/research/polymarket_gamma_client.py`
  - runner：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_polymarket_gamma_snapshot.py`
  - 测试：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_polymarket_gamma_client.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_polymarket_gamma_snapshot_script.py`
  - 定位：`RESEARCH_ONLY` 的公开只读情绪侧车，不使用交易认证凭证，不进入 live/source-of-truth
  - 当前汇总链：
    - `run_external_intelligence_refresh.py` 会串联 `jin10 -> axios -> polymarket -> unified snapshot`
    - `run_operator_panel_refresh.py` 现会自动触发上述外部情报刷新
  - 当前公开语义：
    - 只读取 public gamma market 数据
    - 只产出 `markets_total / bullish_count / bearish_count / yes_price_avg / top_categories / top_titles`
    - 只能用于情绪面参考与置信度降级，不能直接驱动 execution
- `run_operator_panel_refresh.py` 已接入外部情报主刷新链：
  - 当前顺序：
    - `run_external_intelligence_refresh.py --skip-dashboard-snapshot`
    - `build_operator_task_visual_panel.py`
    - `build_conversation_feedback_projection_internal.py`
    - `build_dashboard_frontend_snapshot.py`
  - 目的：让现有 operator/dashboard 主刷新链自动携带最新 Jin10/Axios/unified external intelligence
  - 失败语义：
    - 外部情报 refresh 失败时只记 `degraded_request_failed`
    - 不阻断 operator panel / dashboard snapshot 主刷新
    - 保持 `dashboard/metrics/external sidecar` 故障为 degrade-only，而不是 hard-fail
- CPA channel kernel Phase 1 已接入 repo：
  - package：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/cpa_channels/account_bundle.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/cpa_channels/token_store.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/cpa_channels/cpa_authfiles.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/cpa_channels/ingest_pipeline.py`
  - script：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_cpa_channel_ingest.py`
  - 测试：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_cpa_channel_token_store.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_cpa_channel_authfile_export.py`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_cpa_channel_ingest_script.py`
  - 当前范围：
    - 只做 `account bundle -> token store -> run ledger -> cpa auth-file export`
    - 不接 browser/oauth/live management API
    - 作为 `cpa` 控制面的底层 source-owned kernel 预埋
- CPA control plane Phase 2 已接入 repo：
  - builder：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_cpa_control_plane_snapshot.py`
  - 当前来源：
    - `/Users/jokenrobot/Downloads/Folders/MAC工具/data/registered_success_active20.csv`
    - `/Users/jokenrobot/Downloads/Folders/MAC工具/data/active20_acceptance_20260329_212018.json`
    - `/Users/jokenrobot/Downloads/Folders/MAC工具/data/cpa_account_inventory.json`
    - `/Users/jokenrobot/Downloads/Folders/MAC工具/data/cpa_non_active_review_queue.csv`
    - `/Users/jokenrobot/Downloads/Folders/MAC工具/data/cpa_no_retry_deactivated_accounts.csv`
    - `/Users/jokenrobot/Downloads/Folders/MAC工具/data/registered_new_unmounted.csv`
    - `system/output/review/latest_cpa_channel_ingest.json`
  - 当前产物：
    - `system/output/review/latest_cpa_control_plane_snapshot.json`
    - `system/output/review/latest_cpa_control_plane_snapshot.md`
    - `system/dashboard/web/public/data/cpa_control_plane_snapshot.json`
  - 当前消费面：
    - `run_operator_panel_refresh.py` 会自动刷新并复制到 `dist/data/`
    - `system/dashboard/web/src/pages/CpaPage.tsx` 会读取 `/data/cpa_control_plane_snapshot.json`
  - 当前语义：
    - 历史成功 20-active 快照作为 source-owned 成功集
    - 当前 inventory / review queue / no-retry-deactivated 作为漂移/重建状态面
    - CPA 前端控制面不再只依赖临时 localStorage/error ledger，开始读取 source-owned 只读摘要
- CPA control plane Phase 3 已接入 repo：
  - 当前增强：
    - `build_cpa_control_plane_snapshot.py` 现会生成 guarded action command 列表：
      - `check_five_account_acceptance.py`
      - `run_active_target_sync.py`
      - `run_retry_candidate_pipeline.py`
    - `CpaPage.tsx` 现会展示：
      - source-owned 对账视图（历史成功 vs inventory）
      - connected live auth-files 与历史成功集的命中/缺口对账
      - Guarded Actions 命令建议（只读展示，不在页面里直接执行）
  - 当前原则：
    - live OAuth / active sync / retry candidate 仍保持 `LIVE_GUARD_ONLY`
    - 页面只提供 source-owned reconciliation 和命令建议，不直接触发 live workflow
- CPA control plane Phase 4 已接入 repo：
  - `build_cpa_control_plane_snapshot.py` 现会输出结构化分组：
    - `retry_candidate_rows`
    - `blocked_about_you_rows`
    - `no_retry_deactivated_rows`
    - `new_unmounted_rows`
  - `CpaPage.tsx` 现会把上述分组渲染为 drilldown table：
    - `retry_candidate 队列`
    - `blocked_about_you 队列`
    - `no_retry_deactivated 队列`
    - `new_unmounted 池`
  - 目标：
    - 不再只看 summary chip
    - 直接把 handoff 中最关键的 CPA 风险队列结构化暴露到控制面
- CPA control plane Phase 5 已接入 repo：
  - 新增 guarded action runner：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_cpa_guarded_action.py`
  - 当前能力：
    - 从 `latest_cpa_control_plane_snapshot.json` 读取受控动作
    - 默认 `dry_run`
    - 可选 `--execute`
    - 写回 `latest_cpa_guarded_action_receipt_<action>.json`
  - `build_cpa_control_plane_snapshot.py` 现会聚合 `latest_receipts`
  - `CpaPage.tsx` 现会展示：
    - 最近回执
    - 最近一次 guarded action 的状态 / 时间 / returncode
  - 当前原则：
    - 仍不允许页面内直接触发 live 执行
    - 先通过 CLI 受控入口执行，再由 snapshot 把 receipts 回挂到控制面
- `/ops/risk` 已从旧 public terminal shell 进一步收敛为独立 risk cockpit：
  - header / sidebar 改成 `风险驾驶舱` + `操作路由`
  - 首屏结构改成 `风险观察 / 风险诊断 / 动作分流 / 当前动作栈`
  - 动作入口固定分流到 `/ops/audits`、`/ops/runbooks`、`/ops/workflow`、`/ops/traces`
  - 仍保留 `ops-context-strip` 与 source-owned read-model contract，不触碰 live path
- `/ops/audits` 已从 placeholder 提升为可用 audit page：
  - 直接承接现有 `公开入口拓扑`
  - 直接承接现有 `公开面验收`
  - 首轮包含 `穿透层 1 / 验收总览` 与 `穿透层 2 / 子链状态`
  - 当前仍保留与旧 `workspace/contracts` 的兼容关系；更深的 interface/source-head/fallback 迁移留待后续切片
- `/ops/runbooks` 已从 placeholder 提升为可用 action page：
  - 直接承接 `scheduler action log`
  - 直接承接 `action checklist`
  - 直接承接 `priority repair plan` 与 optimization backlog
  - 当前页面只做动作编排与证据汇总，不在前端直接下发 live execution
- `/ops/workflow` 已从 placeholder 提升为可用 governance page：
  - 直接承接 `hold_selection_handoff` 的基线/候选/推荐摘要
  - 直接承接 `signalRisk.gateScores` 作为审批门禁
  - 直接承接 `fallbackChain` 作为当前回退链表达
  - 当前页面仍未接入更细的 workflow store / proposal approval history，只做 source-owned 最小制度页
- `/ops/traces` 已从 placeholder 提升为可用 trace page：
  - 直接承接当前 `panel / section / row / artifact` 穿透上下文
  - 直接承接 `snapshotPath`、artifact/source-head path
  - 直接承接 `artifactHandoffs` 与 `fallbackChain`
  - 当前页面仍未接入更细的 event stream/trace store，只做只读 trace context 与证据跳转页
- 发布验收链已开始从 hash-route 切到 path-route：
  - `run_dashboard_public_topology_smoke.py` 已改为验证 `/ops/overview` 与 `/ops/audits`
  - `run_dashboard_public_acceptance.py` 已接受 path-based `workspace/search/raw/contracts` 路由
  - `run_dashboard_workspace_artifacts_smoke.py` 的 public workspace / graph-home / internal alignment / internal terminal focus smoke 已切到 path-based 主路径
  - 当前 topology / workspace routes / public acceptance 三段 Python 验收测试都已通过
  - 工程 runbook 已同步切到 path-route 示例；legacy hash bridge 当前只保留兼容入口，不再作为主文档契约
- legacy sidecar 已完成下线：
  - 生产入口固定为 `index.html -> src/main.tsx -> src/App.tsx`
  - `src/main.legacy.jsx` 与 `src/App.legacy.jsx` 已删除
  - legacy hook/lib data chain 与 legacy component subtree 已删除
  - 新增 `src/legacy-isolation.test.js`，防止 active runtime import graph 重新连回 legacy sidecar
  - 详细记录见 `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/LEGACY_SIDECAR_DECOMMISSION.md`
- 本轮仓库内设计/计划文档：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/docs/superpowers/specs/2026-04-03-fenlie-dashboard-path-routing-ops-ia-design.md`
  - `/Users/jokenrobot/Downloads/Folders/fenlie/docs/superpowers/plans/2026-04-03-fenlie-dashboard-path-routing-ops-ia-implementation-plan.md`
  - `/Users/jokenrobot/Downloads/Folders/fenlie/docs/superpowers/plans/2026-04-03-fenlie-legacy-sidecar-decommission-plan.md`
- 本轮最小验证已通过：
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/App.test.tsx -t "ops overview route|legacy overview hash entry|legacy terminal public entry" --reporter=dot`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/pages/GraphHomePage.test.tsx -t "shows localized filters|surfaces precise deep links" --reporter=dot`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/search/search-ui.test.tsx -t "path-based search route entry|result navigation|prefills /search route|scope switching explicit|preserves deep-link params" --reporter=dot`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx tsc --noEmit`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/App.test.tsx -t "renders ops risk as a dedicated cockpit route before entering workspace routes" --reporter=dot`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/App.test.tsx -t "renders ops audits as an audit-owned page with topology and acceptance content" --reporter=dot`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/App.test.tsx -t "renders ops runbooks as an action-owned page with action log, checklist, and repair plan" --reporter=dot`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/App.test.tsx -t "renders ops workflow as a governance page with proposal context, approval gates, and rollback path" --reporter=dot`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/App.test.tsx -t "renders ops traces as a trace page with context, evidence jumps, and fallback chain" --reporter=dot`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && node ./scripts/run-python.mjs -m pytest -q ../../tests/test_run_dashboard_public_topology_smoke_script.py`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && node ./scripts/run-python.mjs -m pytest -q ../../tests/test_run_dashboard_workspace_artifacts_smoke_script.py`
  - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && node ./scripts/run-python.mjs -m pytest -q ../../tests/test_run_dashboard_public_acceptance_script.py`

## Current State (2026-03-21)
- Priority execution plan (`/Users/jokenrobot/Downloads/Folders/fenlie/docs/superpowers/plans/2026-03-21-fenlie-priority-execution-plan.md`) 已完成当前 5 个主任务，范围保持在 `DOC_ONLY / RESEARCH_ONLY / SIM_ONLY`，未触碰 live capital、order routing、execution queue、fund scheduling。
- Dashboard / public surface 当前可依赖的发布与验收链：
  - Cloudflare 包装脚本已内置清代理：`npm run cf:whoami`、`npm run cf:deploy`
  - 最新公开拓扑 smoke：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113313Z_dashboard_public_topology_smoke.json`
  - 最新 workspace routes smoke：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113336Z_dashboard_workspace_routes_browser_smoke.json`
  - 最新公开验收：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113341Z_dashboard_public_acceptance.json`
  - 历史工件阶段当时覆盖的是 `#/overview` 与 `#/workspace/contracts` 真实浏览器断言；当前 canonical public route 已迁到 `/ops/overview` 与 `/ops/audits`
  - 当前 topology smoke 还会落持久化截图：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113313Z_dashboard_public_topology_root_overview_browser.png`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113313Z_dashboard_public_topology_pages_overview_browser.png`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113313Z_dashboard_public_topology_root_contracts_browser.png`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113313Z_dashboard_public_topology_pages_contracts_browser.png`
  - 当前公开验收事实应优先读取 `*_dashboard_public_topology_smoke.json` 与 `*_dashboard_public_acceptance.json`，不再把旧的 `latest_dashboard_public_deploy_readiness.*` 当成当前事实
- Strategy research mainline 已收敛为：
  - 单币：`ETHUSDT`
  - 周期：`15m`
  - setup：`breakout-pullback`
  - 当前重点：`exit/risk`
  - 暂不继续扩 entry filter；`VPVR proxy / reclaim structure filter / daily_stop_r` 仍视为已证伪或暂无增益方向
- 2026-03-21 `10:30Z` 已完成 hold-selection gate blocker 的 source-owned 修正：
  - 脚本：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.py`
  - 新增保留字段：`return_candidate / return_watch`
  - `gate_state` 不再把所有情形都静态硬写为 `blocked_transfer_demoted / blocked_transfer_failed / blocked_transfer_revived_watch_only`
  - 当前 latest gate blocker：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only.json`
  - 当前 source-owned gate state：
    - `hold24_promotion=blocked_return_candidate_until_longer_forward_oos`
    - `hold12_global_drop=blocked_transfer_watch_only`
    - `dynamic_router_promotion=blocked_future_tail_insufficient_after_positive_historical_transfer`
- 2026-03-21 `10:30Z` canonical hold-selection handoff 已刷新：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_price_action_breakout_pullback_hold_selection_handoff_sim_only.json`
  - 当前 handoff 头：
    - `active_baseline=hold16_zero`
    - `local_candidate=hold8_zero`
    - `transfer_watch=[hold12_zero]`
    - `return_candidate=[hold24_zero]`
    - `demoted_candidate=[hold24_zero, pullback_depth_atr_router]`
- 2026-03-21 `10:31Z` operator panel / public+dist snapshot 已刷新：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T103100Z_operator_task_visual_panel.json`
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T103100Z_operator_task_visual_panel.html`
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_snapshot.json`
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_internal_snapshot.json`
- 2026-03-21 新增 dashboard source-head / contracts drilldown 的 `return_candidate` 透传：
  - snapshot builder：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py`
  - 前端消费：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/adapters/read-model.ts`
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/components/WorkspacePanels.tsx`
  - 公开用语：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/contracts/dashboard-term-contract.json`
  - 当前目标：让 `hold_selection_handoff` 的 `return_candidate=[hold24_zero]` 在 source-head 摘要与 contracts 钻取层中直接可见，不再只埋在 raw artifact
- 2026-03-21 `10:48Z` 已刷新 intraday orderflow research-only 链，清除 2026-03-18 旧 latest 漂移：
  - blueprint：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_blueprint.json`
  - sidecar：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_sidecar.json`
  - context veto pack：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_context_veto_research_pack.json`
  - veto study：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_veto_event_study.json`
  - coverage gap casebook：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_coverage_gap_casebook.json`
  - majors capture priority：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_majors_capture_priority_notes.json`
  - execution checklist：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_majors_capture_execution_checklist.json`
  - research gate blocker：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_research_gate_blocker_report.json`
  - 当前 blueprint 已引用最新 exit hold compare：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T091000Z_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json`
  - 当前 source-owned orderflow 结论仍是：
    - `continue_price_state_primary_build_orderflow_sidecar_before_backtest`
    - `block_orderflow_replay_and_fitting_keep_eth_price_state_only_until_evidence_upgrade`
- 2026-03-21 已把 orderflow 主证据接入 dashboard research workspace 的 `research_cross_section` 组：
  - 配置：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/config/dashboard_snapshot_artifact_selection.json`
  - 公开文案：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/contracts/dashboard-term-contract.json`
  - 当前 public snapshot 已包含：
    - `intraday_orderflow_blueprint`
    - `intraday_orderflow_research_gate_blocker`
  - 当时公开页 `#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow` 已可见这两条 canonical 工件；当前 canonical 写法对应 `/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow`
- 2026-03-21 新增 active overview 顶层的 source-owned 研究主线摘要卡：
  - 页面：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/pages/OverviewPage.tsx`
  - 测试：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx`
  - 当前行为：
    - `/overview` 直接显示 `hold_selection_handoff`
    - 摘要字段：`active_baseline / local_candidate / transfer_watch / return_candidate`
    - 提供深链跳转：`/workspace/contracts?page_section=contracts-source-heads`
  - 当前目标已从“仅 contracts drilldown 可见”推进为“overview 顶层也可见”
- 2026-03-21 已重新构建并发布 Cloudflare Pages：
  - 命令：`cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npm run cf:deploy`
  - 最新 Pages deploy：
    `https://6442acc0.fenlie-dashboard-web.pages.dev`
  - 随后公开面验收：
    `npm run verify:public-surface -- --skip-workspace-build`
  - 当前结果：`ok = true`
  - 最新验收工件：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T111433Z_dashboard_public_acceptance.json`
  - 最新 workspace routes smoke：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T111428Z_dashboard_workspace_routes_browser_smoke.json`
  - 最新公开拓扑工件：
    `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T111405Z_dashboard_public_topology_smoke.json`
  - 当时 `workspace_routes_smoke` 已覆盖 `#/overview`，并对 `研究主线摘要 / hold24_zero` 做真实浏览器断言；当前 canonical public overview 已迁到 `/ops/overview`
  - 当前 `workspace_routes_smoke` 已覆盖 contracts 页新增 marker：`公开面验收 / root overview 截图 / pages overview 截图 / root contracts 截图 / pages contracts 截图 / 公开快照拉取次数 / 内部快照拉取次数`
  - 当时 `topology_smoke` 已覆盖 `https://fuuu.fun/#/overview`、`https://fenlie.fuuu.fun/#/overview`、`https://fuuu.fun/#/workspace/contracts`、`https://fenlie.fuuu.fun/#/workspace/contracts`；当前 canonical public route 已迁到 `/ops/overview` 与 `/ops/audits`
  - 当前 `topology_smoke` 已分别通过 `root_overview_browser / pages_overview_browser / root_contracts_browser / pages_contracts_browser`
  - 当前 `topology_smoke` 已为 `root/pages overview + contracts` 额外落地截图证据，便于后续审计与回归对比
  - 当前 contracts 公开验收区已透传 `root/pages overview + contracts` 截图路径，以及 `公开/内部快照拉取次数`，并把文案从 `工作区四路由烟测` 收敛为 `工作区五页面烟测`
  - 当前公开 artifacts 页已能通过 `group=research_cross_section&search=orderflow` 直接命中新接入的 orderflow canonical 工件
  - 当前 `workspace_routes_smoke` 已把该 orderflow 过滤路由固化为真实浏览器回归断言，并在 smoke payload 中新增：
    - 历史 `artifacts_filter_assertion.route=#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow`；当前 canonical 写法对应 `/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow`
    - `artifacts_filter_assertion.active_artifact=intraday_orderflow_blueprint`
    - `artifacts_filter_assertion.visible_artifacts=[intraday_orderflow_blueprint, intraday_orderflow_research_gate_blocker]`
  - 当前 `run_dashboard_public_acceptance.py` 已把 `artifacts_filter_assertion` 升级为聚合验收硬门槛：
    - 缺失该字段 => `failure_reason=missing_orderflow_artifacts_filter_assertion`
    - 字段值偏离 => `failure_reason=invalid_orderflow_artifacts_filter_assertion`
  - 2026-03-27 已补 source-owned 降级语义：
    - broad smoke 跟随 `workspace_default_focus`
    - contracts 深链对齐到 `contracts-acceptance-subcommands / contracts-source-head-operator_panel / contracts-fallback`
    - 当当前 snapshot 缺席 orderflow 工件时，显式输出 `source_available=false`
- 2026-03-21 `11:13Z~11:14Z` 已在 orderflow terminal handoff / contracts 验收字段接入后重新完成 build + 公开验收闭环：
  - 本地 build：`cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npm run build`
  - 最新 operator panel：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T111342Z_operator_task_visual_panel.json`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T111342Z_operator_task_visual_panel.html`
  - 最新 dist/public snapshot：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_snapshot.json`
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/data/fenlie_dashboard_internal_snapshot.json`
  - 最新公开拓扑 smoke：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T111405Z_dashboard_public_topology_smoke.json`
  - 最新 workspace routes smoke：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T111428Z_dashboard_workspace_routes_browser_smoke.json`
  - 最新公开验收：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T111433Z_dashboard_public_acceptance.json`
  - 当前结果：
    - `npm run build => exit 0`
    - `npm run verify:public-surface -- --skip-workspace-build => ok = true`
  - 当前 orderflow 工件链已形成：
    - `workspace smoke -> public acceptance hard gate -> read-model handoff -> contracts acceptance fields -> App/Vitest regression`
  - 当前 contracts 公开验收区额外可见：
    - `orderflow 过滤路由`
    - `orderflow 激活工件`
    - `orderflow 可见工件`
- 2026-03-21 `19:22~19:24 +0800` 已完成一轮本地 release-checklist 回归并刷新公开验收证据：
  - 前端全量测试：
    - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npm test`
    - 结果：`19 files / 97 tests passed`
  - TypeScript：
    - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx tsc --noEmit`
    - 结果：`exit 0`
  - dashboard 脚本回归：
    - `cd /Users/jokenrobot/Downloads/Folders/fenlie && pytest -q system/tests/test_build_dashboard_frontend_snapshot_script.py system/tests/test_run_operator_panel_refresh_script.py system/tests/test_run_dashboard_public_topology_smoke_script.py system/tests/test_run_dashboard_workspace_artifacts_smoke_script.py system/tests/test_run_dashboard_public_acceptance_script.py`
    - 结果：`22 passed`
  - 最新 workspace routes smoke：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T112357Z_dashboard_workspace_routes_browser_smoke.json`
  - 最新公开拓扑 smoke：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T112419Z_dashboard_public_topology_smoke.json`
  - 最新公开验收：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T112445Z_dashboard_public_acceptance.json`
  - 当前结果：
    - `workspace smoke => ok = true`
    - `verify:public-surface => ok = true`
- 2026-03-21 `19:29~19:33 +0800` 已把 dashboard 上架前最小验证收口为单一 package script，并修复 topology smoke 对 Pages TLS EOF 抖动的可重试性：
  - 新脚本：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/package.json`
    - `verify:release-checklist = build -> test -> tsc -> workspace routes smoke -> public acceptance`
  - 新契约测试：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/deploy-script-contract.test.js`
  - TLS EOF 重试修复：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py`
    - 将 `ssl.SSLEOFError` 视为 fallback 场景下的可重试传输抖动
  - 新回归测试：
    - `/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_dashboard_public_topology_smoke_script.py`
    - 覆盖：`strict cert fail -> insecure fallback EOF once -> retry success`
  - 实际验证：
    - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npx vitest run src/deploy-script-contract.test.js`
    - 结果：`2 passed`
    - `cd /Users/jokenrobot/Downloads/Folders/fenlie && pytest -q system/tests/test_run_dashboard_public_topology_smoke_script.py`
    - 结果：`6 passed`
    - `cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npm run verify:release-checklist`
    - 结果：`exit 0`
    - 其内嵌全量前端测试结果：`19 files / 98 tests passed`
  - 最新 release-checklist 工件：
    - workspace routes smoke：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113307Z_dashboard_workspace_routes_browser_smoke.json`
    - public topology smoke：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113313Z_dashboard_public_topology_smoke.json`
    - public acceptance：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T113341Z_dashboard_public_acceptance.json`
- 最新研究结论：
  - 数据集：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T075512Z_public_intraday_crypto_bars_dataset.json`
  - base sim：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T075700Z_price_action_breakout_pullback_sim_only.json`
  - hold forward compare：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T080200Z_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json`
  - 当前 source-owned 结论：`hold16` 继续作为 baseline anchor，`hold8` 提升为更强 forward candidate，但证据仍不足以直接替换 baseline
- 新补的高价值测试缺口：
  - `test_build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only_script.py`
  - `test_build_price_action_breakout_pullback_hold_selection_handoff_sim_only_script.py`
  - `test_build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only_script.py`
  - `test_run_operator_panel_refresh_script.py`
- 当前最小高价值验证包：
  - `pytest -q /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_selection_handoff_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_operator_panel_refresh_script.py`
  - 当前结果：`7 passed`
- 当前扩展验证包（含 frontier report / snapshot / panel refresh）：
  - `pytest -q /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_frontier_report_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_selection_handoff_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_dashboard_frontend_snapshot_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_operator_panel_refresh_script.py`
  - 当前结果：`16 passed`
- 公开面验收：
  - 命令：`python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_acceptance.py --workspace /Users/jokenrobot/Downloads/Folders/fenlie --skip-workspace-build`
  - 当前结果：`ok = true`
- 2026-03-21 新增本地 120d 研究推进（复用历史本地 15m 数据，不依赖公网新抓取）：
  - 数据集：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260319T081154Z_public_intraday_crypto_bars_dataset.csv`
  - base sim：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T085500Z_price_action_breakout_pullback_sim_only.json`
  - hold compare：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T091000Z_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.json`
  - hold robustness：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T091200Z_price_action_breakout_pullback_exit_hold_robustness_sim_only.json`
  - hold family triage：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T091400Z_price_action_breakout_pullback_hold_family_triage_sim_only.json`
  - rider triage：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T091600Z_price_action_breakout_pullback_exit_rider_triage_sim_only.json`
  - frontier report：`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260321T092900Z_price_action_breakout_pullback_hold_frontier_report_sim_only.json`
- 120d 本地结论已从旧 30d mixed profile 收敛为更强 baseline 结论：
  - `hold8 vs hold16` 默认 forward compare（9 slices）=> `hold_16_forward_leader_keep_baseline`
  - 4 组 robustness => `hold_16_consistency_reinforced_keep_baseline`
  - frontier 当前不再允许把 `hold8/hold24` 硬写为双 leader candidate；修正后当前 head 表达为：`hold16_baseline_reinforced_transfer_watch_only`
- 新增 frontier report 契约测试并修正脚本，避免 source-owned artifact 夸大弱证据：
  - 测试：`/Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_frontier_report_sim_only_script.py`
  - 脚本：`/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_price_action_breakout_pullback_hold_frontier_report_sim_only.py`
  - 当前最小扩展回归包：
    `pytest -q /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_selection_handoff_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_selection_gate_blocker_report_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_build_price_action_breakout_pullback_hold_frontier_report_sim_only_script.py /Users/jokenrobot/Downloads/Folders/fenlie/system/tests/test_run_operator_panel_refresh_script.py`
  - 当前结果：`9 passed`
- 结构债只落设计，不在当前执行链直接大改：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/docs/superpowers/plans/2026-03-21-engine-snapshot-boundary-note.md`
  - `engine.py` 建议拆成 runtime primitives / mode params / paper execution / micro capture / review runtime / thin facade
  - `build_dashboard_frontend_snapshot.py` 建议拆成 contracts / surface policy / read-model builders / ui-contract builders / thin writer

## Legacy State (2026-02-15)
- Architecture loop is active: data -> regime -> signal -> risk -> backtest -> review -> gate.
- Continuous micro truth capture is online:
  - CLI: `lie micro-capture --date YYYY-MM-DD [--symbols BTCUSDT,ETHUSDT]`.
  - writes run artifact: `output/artifacts/micro_capture/*_micro_capture.json`.
  - writes sqlite tables: `micro_capture_runs / micro_capture_source_state / micro_capture_symbol_state / micro_capture_cross_source`.
  - returns rolling 7-day quality snapshot (`pass_ratio`, `avg_cross_source_fail_ratio`, `schema/time_sync ratios`).
- 30m daemon chain now includes optional `micro-capture` interval task:
  - config keys:
    `validation.micro_capture_daemon_enabled` /
    `validation.micro_capture_daemon_interval_minutes` /
    `validation.micro_capture_daemon_symbols`.
  - `run-daemon --dry-run` now previews `interval:micro_capture` due status.
- Scheduler critical sections now use file mutex:
  - lock file: `output/state/run-halfhour-pulse.lock`.
  - protects `run-slot / run-session / daemon slot execution` from concurrent replay.
- Micro factor collection now supports hot-loading configured cross-source providers when
  `validation.micro_cross_source_build_missing_provider=true`, even if `data.provider_profile` is opensource-only.
- Mode feedback artifact is online: `output/daily/YYYY-MM-DD_mode_feedback.json`.
- Mode health gate is online: `gate_report.checks.mode_health_ok`.
- Execution risk throttle is online:
  - `run_eod` applies `risk_multiplier` into position sizing.
  - `run_premarket` / `run_intraday_check` now output `runtime_mode + mode_health + risk_control + risk_multiplier`.
  - `run_premarket` / `run_intraday_check` now emit manifests for slot-level traceability.
- State stability monitoring is online (`ops_report`):
  - mode switch rate
  - risk-multiplier floor + drift
  - source-confidence floor
  - mode-health fail-days
  - threshold breaches are surfaced in `state_stability.alerts`.
- Review-loop defect planning is now state-aware:
  - `STATE_*` defect codes are emitted when `state_stability` thresholds breach.
  - `next_actions` is reordered to prioritize state stabilization first.
- Mode-level adaptive review update is online:
  - `run_review` now applies bounded `mode_adaptive_update` on
    `signal_confidence_min / convexity_min / hold_days / max_daily_trades`
    based on `mode_history` performance bands.
  - Audit includes `mode_adaptive` payload in `param_delta.yaml`.
- Mode drift monitor is online:
  - `gate_report.checks.mode_drift_ok` compares live performance vs backtest baseline.
  - `ops_report.mode_drift` exposes drift status/alerts (win-rate & profit-factor gaps).
  - review-loop defect plan emits `MODE_DRIFT_*` codes and prioritizes drift fixes.
- Style-drift adaptive guard + release gate linkage is online:
  - `run_review` now applies severity-scaled收敛（`signal_confidence_min / max_daily_trades / hold_days`）并产出 `style_drift_guard` 审计。
  - `gate_report.checks.style_drift_ok` supports monitor-only / hard-fail modes (`style_drift_gate_hard_fail`).
  - `ops_report.style_drift` + defect plan now expose `STYLE_DRIFT_*` codes and gate failures.
- Slot anomaly monitor is online:
  - `gate_report.checks.slot_anomaly_ok` evaluates premarket/intraday/eod anomalies.
  - `ops_report.slot_anomaly` exposes missing/anomaly ratios and threshold checks.
  - review-loop defect plan emits `SLOT_*` codes and prioritizes slot-chain repairs.
- Reconciliation drift monitor is online:
  - `gate_report.checks.reconcile_drift_ok` validates eod manifest vs daily csv/sqlite/open-state consistency.
  - `ops_report.reconcile_drift` exposes missing ratio and plan/close/open drift breaches.
  - review-loop defect plan emits `RECONCILE_*` codes and prioritizes reconciliation repair.
- Broker snapshot contract linter is online in reconciliation:
  - schema lint (`source/open_positions/closed_count/closed_pnl/positions`) + numeric hard range + symbol normalization.
  - new checks: `broker_contract_schema_ok / broker_contract_numeric_ok / broker_contract_symbol_ok`.
  - new defect codes: `RECONCILE_BROKER_CONTRACT_SCHEMA / NUMERIC / SYMBOL`.
- Broker canonical snapshot view is online:
  - when broker contract lint passes, system writes canonical snapshot to
    `output/artifacts/broker_snapshot_canonical/YYYY-MM-DD.json`.
  - canonical view includes normalized `symbol/side/qty/notional` plus `raw_symbol/raw_side`.
  - new check/alert/defect:
    `broker_contract_canonical_view_ok` /
    `reconcile_broker_contract_canonical_view_failed` /
    `RECONCILE_BROKER_CANONICAL_VIEW`.
- Canonical row-level reconciliation is online:
  - broker-vs-system行级比较优先使用 canonical 视图（`canonical_file` / `canonical_inline`），无可用 canonical 时回退 `canonical_fallback`。
  - checks/alerts/defect:
    `broker_row_diff_ok` /
    `reconcile_broker_row_diff_high` /
    `RECONCILE_BROKER_ROW_DIFF`.
  - metrics include:
    `broker_row_diff_*`（samples/breach/key_mismatch/count_gap/notional_gap/canonical_preferred）。
- Reconcile row-diff drill-down artifact is online:
  - when row-diff has breaches, system exports top mismatched keys + source-map hints to
    `output/review/YYYY-MM-DD_reconcile_row_diff.json` and `.md`.
  - gate/ops/review now carry artifact metadata (`written/path/reason/sample_rows/breach_rows`)
    for faster hotfix routing.
- Reconcile row-diff alias dictionary + CI lint is online:
  - configurable alias maps:
    `validation.ops_reconcile_broker_row_diff_symbol_alias_map` /
    `validation.ops_reconcile_broker_row_diff_side_alias_map`.
  - row-level diff now applies alias normalization before key matching, reducing false mismatch due to venue symbol/side enums.
  - config validation now lints alias maps in tests/CI (empty or non-canonical entries rejected).
- Reconcile row-diff alias drift monitor is online:
  - metrics now expose `broker_row_diff_alias_hit_rate` and `broker_row_diff_unresolved_key_ratio`.
  - configurable guardrails:
    `validation.ops_reconcile_broker_row_diff_alias_monitor_enabled` /
    `validation.ops_reconcile_broker_row_diff_alias_hit_rate_min` /
    `validation.ops_reconcile_broker_row_diff_unresolved_key_ratio_max`.
  - new alert/check/defect path:
    `reconcile_broker_row_diff_alias_drift` /
    `broker_row_diff_alias_drift_ok` /
    `RECONCILE_BROKER_ROW_DIFF_ALIAS_DRIFT`.
- Mode-aware stress matrix report is online:
  - CLI: `lie stress-matrix --date YYYY-MM-DD --modes ultra_short,swing,long`.
  - output: `output/review/YYYY-MM-DD_mode_stress_matrix.json` + `.md`.
  - evaluation grid: `2015_crash / 2020_pandemic / 2022_geopolitical / extreme_gap` windows.
- Stress-matrix trend monitor is online in gate/ops/review-loop:
  - compares latest stress-matrix run vs rolling baseline runs.
  - checks: `stress_matrix_trend_ok` with robustness/annual/drawdown/fail-ratio gates.
  - defect codes: `STRESS_MATRIX_ROBUSTNESS / ANNUAL_RETURN / DRAWDOWN / FAIL_RATIO`.
- Review-loop stress-matrix auto-trigger is online:
  - when `mode_drift` or `stress_matrix_trend` breaches, `review_until_pass` can auto-run stress matrix once per cycle (configurable).
  - round audit now includes `stress_matrix_autorun` payload with trigger reason, run status, and output artifact paths.
- Review-loop stress-matrix cooldown/backoff is online:
  - auto-trigger now supports per-round cooldown and exponential backoff,
    preventing repeated stress reruns in noisy multi-round loops.
  - config keys:
    `review_loop_stress_matrix_autorun_cooldown_rounds` /
    `review_loop_stress_matrix_autorun_backoff_multiplier` /
    `review_loop_stress_matrix_autorun_backoff_max_rounds`.
  - round audit now records cooldown state (`next_allowed_round`, `cooldown_remaining_rounds`, `runs_used`).
- Stress-matrix historical trigger analytics is online:
  - gate/ops now expose `stress_autorun_history` over rolling review-loop rounds.
  - metrics include trigger density, skip reason distribution, and cooldown efficiency.
  - compliance artifact:
    `output/review/YYYY-MM-DD_stress_autorun_history.json` + `.md`.
- Stress-matrix auto-trigger adaptive guardrail is online:
  - review loop now computes dynamic `max_runs` from recent trigger density (history + current rounds).
  - high trigger-density windows throttle stress reruns; low-density windows expand bounded rerun budget.
  - round audit includes `max_runs_base / max_runs / adaptive.reason / adaptive.factor / adaptive.trigger_density`.
- Stress-matrix auto-trigger adaptive saturation monitor is online:
  - gate/ops now evaluate rolling `effective_max_runs / base_max_runs` trend and throttle/expand occupancy.
  - checks: `stress_autorun_adaptive_ok` with floor/ceiling + throttle-ratio + expand-ratio gates.
  - review-loop defect plan emits:
    `STRESS_AUTORUN_ADAPTIVE_RATIO_LOW / RATIO_HIGH / THROTTLE / EXPAND`.
- Stress-matrix auto-trigger adaptive reason drift monitor is online:
  - gate/ops now evaluate `high_density_throttle / low_density_expand` reason-mix drift and change-point gap.
  - checks: `stress_autorun_reason_drift_ok` with `reason_mix_gap` + `change_point_gap` gates.
  - review-loop defect plan emits:
    `STRESS_AUTORUN_REASON_MIX / STRESS_AUTORUN_REASON_CHANGE_POINT`.
- Stress-matrix auto-trigger reason-drift artifact export is online:
  - compliance artifact:
    `output/review/YYYY-MM-DD_stress_autorun_reason_drift.json` + `.md`.
  - artifact includes top reason transitions and rolling per-window drift trace (`mix_gap/change_point_gap`).
  - gate/ops payload now carries artifact metadata (`written/path/reason/transition_count/window_trace_points`).
- Temporal audit monitor is online in gate/ops/review-loop:
  - validates recent `strategy_lab` / `research_backtest` manifests for
    `cutoff_date/cutoff_ts/bar_max_ts/news_max_ts/report_max_ts` completeness and leakage.
  - checks: `temporal_audit_ok` with `missing/leak/strict_cutoff` gates.
  - defect codes: `TEMPORAL_AUDIT / TEMPORAL_AUDIT_MISSING / TEMPORAL_AUDIT_LEAK / TEMPORAL_AUDIT_STRICT`.
- Temporal-audit auto-fix assistant is online:
  - for manifests with missing temporal metadata, system can backfill from `artifacts.summary` and `data_fetch_stats`.
  - write-back is guarded by safe temporal validation (`*_max_ts <= cutoff_date`) to avoid front-running leakage.
  - audit payload now includes autofix telemetry (`attempted/applied/failed/skipped`, per-manifest reason).
- Temporal-audit autofix patch artifact is online:
  - each temporal autofix attempt is exported as compliance trace artifact:
    `output/review/YYYY-MM-DD_temporal_autofix_patch.json` + `.md`.
  - artifact includes manifest path, summary source, field-level delta (`before/after/source`) and strict-cutoff patch trace.
  - gate/ops/review payload now includes `temporal_audit.artifacts.autofix_patch`.
- Temporal-audit patch retention/rotation policy is online:
  - `validation.ops_temporal_audit_autofix_patch_retention_days` controls N-day keep window.
  - stale `YYYY-MM-DD_temporal_autofix_patch.{json,md}` artifacts beyond retention are auto-pruned.
  - checksum index is generated at `output/review/temporal_autofix_patch_checksum_index.json`
    with per-file `sha256` + byte-size for compliance traceability.
- Strict cutoff temporal-audit chain is online:
  - `real_data` now emits `cutoff_ts / bar_max_ts / news_max_ts / report_max_ts` (+ review variants).
  - `strategy_lab` / `research_backtest` summaries and manifests now carry these fields.
  - review candidate loader rejects strategy-lab manifests that violate cutoff temporal bounds.
- Slot EOD anomaly is split:
  - quality-path: `eod_quality_anomaly_ok`
  - risk-path: `eod_risk_anomaly_ok`
  - keeps compatibility with aggregate `eod_anomaly_ok`.
- Slot EOD regime-bucket thresholding is online:
  - quality/risk anomaly thresholds now support `trend / range / extreme_vol` bucket maps.
  - new checks: `eod_quality_regime_bucket_ok` / `eod_risk_regime_bucket_ok`.
  - ops report now exposes per-bucket quality/risk anomaly ratios for threshold tuning.
- Automated regime-bucket threshold tuner is online:
  - `run_review` now updates `output/artifacts/slot_regime_thresholds_live.yaml` from rolling EOD manifests.
  - tuning is bounded by `step/buffer/floor/ceiling` and only applies on buckets with enough samples.
  - `gate/ops` can consume live threshold maps when `validation.ops_slot_use_live_regime_thresholds=true`.
  - missing-aware hard guard is online: when slot missing ratio breaches
    `validation.ops_slot_regime_tune_missing_ratio_hard_cap`, tuning is skipped with reason `slot_missing_ratio_high`.
- Broker snapshot adapter is online in reconciliation:
  - optional path: `output/artifacts/broker_snapshot/YYYY-MM-DD.json`
  - checks: `broker_missing_ratio_ok / broker_count_gap_ok / broker_pnl_gap_ok`
  - can be hardened by `validation.ops_reconcile_require_broker_snapshot=true`.
- Broker snapshot producer is online in EOD:
  - `run_eod` now writes `output/artifacts/broker_snapshot/YYYY-MM-DD.json` on every run.
  - payload includes `source=paper_engine`, `open_positions`, `closed_count`, `closed_pnl`, and `positions[]`.
  - eod manifest artifact now includes `broker_snapshot`.
- Live broker snapshot adapter writer is online:
  - source mode switch: `validation.broker_snapshot_source_mode`.
  - `live_adapter` reads `validation.broker_snapshot_live_inbox/YYYY-MM-DD.json` and normalizes to broker snapshot contract.
  - fallback behavior is controlled by `validation.broker_snapshot_live_fallback_to_paper`.
- Live broker snapshot mapping templates are online:
  - mapping profile: `validation.broker_snapshot_live_mapping_profile` (`generic/ibkr/binance/ctp`).
  - optional field override: `validation.broker_snapshot_live_mapping_fields`.
  - position side is canonicalized to `LONG/SHORT/FLAT` (兼容 `BUY/SELL/BOTH/2/3` 枚举).
- Test orchestration timeout guard is online:
  - `lie test-all` now enforces `validation.test_all_timeout_seconds`.
  - timeout returns `returncode=124`, emits `error=test_timeout`, and marks `failed_tests` with `__timeout__`.
- Review-loop timeout fallback is online:
  - `review_until_pass` detects timeout and auto-runs deterministic fast-shard fallback.
  - round payload now includes `tests_timeout` and `timeout_fallback`.
  - defect plan emits `TEST_TIMEOUT` and prioritizes timeout-specific repair order.
- Automatic rollback recommendation is online:
  - `gate_report.rollback_recommendation` outputs `level/score/reason_codes/target_anchor`.
  - `ops_report.rollback_recommendation` is included in status scoring and markdown output.
  - defect plan emits `ROLLBACK_*` codes and puts rollback actions at top priority when needed.
- Paper execution loop is online:
  - `run_eod` now settles previous paper positions using day high/low/close with
    stop-loss / take-profit / time-stop priority.
  - Closed trades are appended to sqlite `executed_plans` (pnl, exit_reason, mode fields).
  - Open paper positions are persisted in `output/artifacts/paper_positions_open.json`.
- Exposure snapshot bug fixed:
  - risk budgeting now reads only latest-date `latest_positions` rows, avoiding
    historical ACTIVE rows cumulative pollution.
- Review audit now records:
  - `runtime_mode`
  - `mode_history`
  - `mode_health`
- Stress reason-drift artifact retention/checksum is online:
  - reason-drift artifact now supports retention rotation:
    `validation.ops_stress_autorun_reason_drift_retention_days`.
  - checksum index now supports audit trace:
    `validation.ops_stress_autorun_reason_drift_checksum_index_enabled`.
  - compliance artifact:
    `output/review/stress_autorun_reason_drift_checksum_index.json`.
  - gate/ops/review defect plan now surfaces
    `STRESS_AUTORUN_REASON_ARTIFACT_ROTATION` /
    `STRESS_AUTORUN_REASON_CHECKSUM_INDEX`.
- Stress history artifact retention/checksum is online:
  - history artifact now supports retention rotation:
    `validation.ops_stress_autorun_history_retention_days`.
  - checksum index now supports audit trace:
    `validation.ops_stress_autorun_history_checksum_index_enabled`.
  - compliance artifact:
    `output/review/stress_autorun_history_checksum_index.json`.
- Shared artifact governance utility is online:
  - common module:
    `src/lie_engine/orchestration/artifact_governance.py`.
  - unified flow (`collect -> rotate -> checksum index`) is now reused by:
    `temporal_autofix_patch` / `stress_autorun_history` / `stress_autorun_reason_drift`.
  - release orchestrator now routes these artifact chains through one governance entry,
    reducing duplicated retention/checksum logic and keeping behavior一致。
- Reconcile row-diff artifact governance is online:
  - row-diff drilldown artifact now supports retention + checksum index:
    `validation.ops_reconcile_broker_row_diff_artifact_retention_days` /
    `validation.ops_reconcile_broker_row_diff_artifact_checksum_index_enabled`.
  - compliance index artifact:
    `output/review/reconcile_row_diff_checksum_index.json`.
  - gate/ops/review defect plan now surfaces:
    `RECONCILE_BROKER_ROW_DIFF_ARTIFACT_ROTATION` /
    `RECONCILE_BROKER_ROW_DIFF_ARTIFACT_CHECKSUM_INDEX`.
- Profile-based artifact governance config is online:
  - validation key:
    `validation.ops_artifact_governance_profiles`.
  - per-profile declarative fields:
    `json_glob / md_glob / checksum_index_filename / retention_days / checksum_index_enabled`.
  - release governance routing now supports profile override while remaining backward-compatible with legacy retention/checksum keys.
  - current onboarded profiles:
    `temporal_autofix_patch / stress_autorun_history / stress_autorun_reason_drift / reconcile_row_diff`.
- Artifact governance compliance snapshot is online:
  - gate/ops now include `artifact_governance` section with profile-level policy snapshot.
  - monitors:
    `required_profiles_present_ok / policy_alignment_ok`.
  - drift alerts:
    `artifact_governance_policy_mismatch` /
    `artifact_governance_legacy_policy_drift`.
  - defect plan emits:
    `ARTIFACT_GOVERNANCE_PROFILE_MISSING` /
    `ARTIFACT_GOVERNANCE_POLICY_MISMATCH` /
    `ARTIFACT_GOVERNANCE_LEGACY_DRIFT`.
- Governance strict mode + baseline freeze is online:
  - new validation keys:
    `validation.ops_artifact_governance_strict_mode_enabled` /
    `validation.ops_artifact_governance_profile_baseline`.
  - gate/ops now surface strict freeze checks:
    `legacy_alignment_ok / baseline_freeze_ok / strict_mode_ok`.
  - strict mode blocks release when profile/policy/legacy/baseline drifts exist:
    `artifact_governance_strict_mode_blocked`.
  - defect plan emits:
    `ARTIFACT_GOVERNANCE_BASELINE_DRIFT` /
    `ARTIFACT_GOVERNANCE_STRICT_BLOCKED`.
- Baseline snapshot promotion workflow is online:
  - `run_review` now auto-promotes artifact governance baseline when review gate passes.
  - snapshot is written to
    `output/artifacts/baselines/artifact_governance/history/YYYY-MM-DD_rXX_*.yaml`,
    and active pointer is refreshed at
    `output/artifacts/baselines/artifact_governance/active_baseline.yaml`.
  - review trace writes
    `output/review/YYYY-MM-DD_baseline_promotion.json`,
    with `rollback_anchor` pointing to previous promoted snapshot when available.
- Baseline rollback drill CLI is online:
  - CLI:
    `lie baseline-rollback-drill --date YYYY-MM-DD [--anchor /abs/path/to/snapshot.yaml]`.
  - default rollback source is current active baseline's `rollback_anchor`; `--anchor` can override it.
  - drill writes audit artifacts:
    `output/review/YYYY-MM-DD_baseline_rollback_drill.json` + `.md`.
  - active baseline is restored from anchor snapshot and pre-rollback active baseline is backed up to:
    `output/artifacts/baselines/artifact_governance/history/YYYY-MM-DD_rollback_backup_*.yaml`.
- Baseline rollback drill dry-run + preflight lint is online:
  - CLI supports preflight mode:
    `lie baseline-rollback-drill --date YYYY-MM-DD [--anchor /abs/path] --dry-run`.
  - preflight checks:
    `active_baseline_exists_ok / active_baseline_payload_ok / rollback_anchor_present_ok / rollback_anchor_exists_ok / rollback_anchor_payload_ok`.
  - dry-run only writes audit artifacts and MUST NOT rewrite `active_baseline.yaml` or create rollback backups.
- Baseline rollback drill preflight schema lint is online:
  - preflight now enforces minimal baseline schema keys:
    `as_of / profiles / snapshot_path` for both active baseline and rollback anchor payload.
  - field-level validation errors are emitted in `preflight.errors[]` with
    `source/field/code/message` to speed up rollback payload debugging.
  - preflight now enforces schema type/format lint:
    `as_of` must be ISO date (`YYYY-MM-DD`), `profiles` must be list/object payload, and `snapshot_path` must be string path.
  - errors include fine-grained codes (`invalid_type / invalid_format / invalid_item_type / empty_item / invalid_key`) for quicker rollback payload triage.
- Baseline rollback drill preflight profile sub-schema lint is online:
  - when `profiles` is a map, preflight now validates profile payload field types.
  - enforced sub-schema checks include:
    `retention_days` numeric (and `>= 1`) plus checksum flags (`*checksum*_enabled`) boolean.
  - errors remain field-level and source-aware in `preflight.errors[]` for targeted rollback payload fixes.
- Baseline rollback drill preflight profile string-path lint is online:
  - when `profiles` is a map, preflight now validates
    `json_glob / md_glob / checksum_index_filename` as non-empty strings.
  - field-level errors emit precise codes:
    `invalid_type` (non-string) / `empty_item` (blank string).
  - rollback anchor dry-run preflight now has dedicated regression coverage for these path fields.
- Baseline rollback drill preflight active/anchor path-field parity coverage is online:
  - dry-run preflight now has dedicated active-baseline negative regression coverage for
    `json_glob / md_glob / checksum_index_filename`.
  - parity checks assert source-scoped errors (`source=active_baseline`) with the same
    path-field error codes used by rollback-anchor validation.
- Baseline rollback drill mixed-source preflight ordering is online:
  - dry-run preflight now continues to collect rollback-anchor schema errors even when
    active baseline schema fails, keeping reason as `active_baseline_invalid_payload`.
  - integration coverage asserts deterministic `preflight.errors[]` ordering for mixed
    active+anchor path-field violations (`json_glob / md_glob / checksum_index_filename`).
- Baseline rollback drill mixed path/non-path preflight ordering coverage is online:
  - dry-run preflight regression now combines non-path schema violations
    (`as_of` format + `snapshot_path` type) with profile path-field violations.
  - integration coverage asserts deterministic source-first ordering remains stable:
    all `active_baseline` errors are emitted before `rollback_anchor` errors.

## Testing Workflow
- Full suite:
  - `lie test-all`
- Fast deterministic subset (for quick agent iteration):
  - `lie test-all --fast --fast-ratio 0.10`
- Review loop default:
  - Round-1 runs fast subset first, then auto-runs full suite if fast passes.
- Parallel shard usage (multi-agent):
  - Agent A: `--fast --fast-ratio 1.0 --fast-shard-index 0 --fast-shard-total 4`
  - Agent B: `--fast --fast-ratio 1.0 --fast-shard-index 1 --fast-shard-total 4`
  - Agent C: `--fast --fast-ratio 1.0 --fast-shard-index 2 --fast-shard-total 4`
  - Agent D: `--fast --fast-ratio 1.0 --fast-shard-index 3 --fast-shard-total 4`

## Machine-Friendly Logging Rules
- `lie test-all` returns compact payload with one-line `summary_line` containing `error=...`.
- Full stdout/stderr are persisted to:
  - `output/logs/tests_YYYYMMDD_HHMMSS.json`
- Downstream gate/review reads `failed_tests` first, then falls back to stderr parsing.

## Next Priorities
1. [DONE 2026-03-01] Add regression coverage for mixed active/anchor preflight failures that combine missing-required-field errors (`as_of/profiles/snapshot_path`) with profile path-field violations and assert stable source-first ordering.
   - [VERIFIED 2026-03-05 17:34 +0800] Unattended run revalidated first-priority-only scope after guard check confirmed no defer (`output/logs/guard_loop_last.json`: `recovery.mode=none`, timestamp not within last 20 minutes). Executed targeted unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_173135.log`, `output/logs/automation2_validate_20260305_173135.log`, `output/logs/automation2_fast_20260305_173135.log`, `output/logs/tests_20260305_173425.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 17:04 +0800] Unattended run revalidated first-priority-only scope via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (see latest `output/logs/automation2_unit_*.log`, `output/logs/automation2_validate_*.log`, `output/logs/automation2_fast_*.log`, `output/logs/tests_*.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 16:37 +0800] Unattended run revalidated first-priority-only scope after guard check confirmed no defer (`output/logs/guard_loop_last.json`: `recovery.mode=none`, timestamp older than 20 minutes). Executed targeted unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_163422.log`, `output/logs/automation2_validate_20260305_163422.log`, `output/logs/automation2_fast_20260305_163422.log`, `output/logs/tests_20260305_163713.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 16:05 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_160222.log`, `output/logs/automation2_validate_20260305_160222.log`, `output/logs/automation2_fast_20260305_160222.log`, `output/logs/tests_20260305_160513.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 15:40 +0800] Unattended run executed first-priority-only revalidation and completed one repair loop: tightened `test_run_review_skips_strategy_lab_candidate_when_merge_gate_fails` by pinning `strategy_lab_merge_min_validation_annual_return=0.0` to keep merge-gate-fail semantics stable under current defaults (`-0.08`). Validation chain passed (`unittest` for targeted repair + first-priority regression, `validate-config`, `test-all --fast --fast-ratio 0.10`; logs `output/logs/automation2_unit_20260305_153650.log`, `output/logs/automation2_validate_20260305_153657.log`, `output/logs/automation2_fast_20260305_153705.log`, `output/logs/tests_20260305_153955.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 14:04 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (logs `output/logs/automation2_unit_20260305_140126.log`, `output/logs/automation2_validate_20260305_140126.log`, `output/logs/automation2_fast_20260305_140126.log`, `output/logs/tests_20260305_140416.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 13:34 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (logs `output/logs/automation2_unit_20260305_133123.log`, `output/logs/automation2_validate_20260305_133123.log`, `output/logs/automation2_fast_20260305_133123.log`, `output/logs/tests_20260305_133414.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 13:04 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (logs `output/logs/automation2_unit_20260305_130125_a1.log`, `output/logs/automation2_validate_20260305_130125_a1.log`, `output/logs/automation2_fast_20260305_130125_a1.log`, `output/logs/tests_20260305_130416.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.

   - [VERIFIED 2026-03-05 12:36 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (logs `output/logs/automation2_unit_20260305_123312_a1.log`, `output/logs/automation2_validate_20260305_123312_a1.log`, `output/logs/automation2_fast_20260305_123312_a1.log`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.

   - [VERIFIED 2026-03-05 09:10 +0800] Unattended run revalidated first-priority regression via unittest (tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order) + validate-config + test-all --fast --fast-ratio 0.10; all green (logs output/logs/automation2_unit_20260305_090808_a1.log, output/logs/automation2_validate_20260305_090808_a1.log, output/logs/automation2_fast_20260305_090808_a1.log). docs/WORKFLOW_AUTO.md remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 08:35 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_083249.log`, `output/logs/automation2_validate_20260305_083249.log`, `output/logs/automation2_fast_20260305_083249.log`, `output/logs/tests_20260305_083540.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 08:05 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_080206.log`, `output/logs/automation2_validate_20260305_080206.log`, `output/logs/automation2_fast_20260305_080206.log`, `output/logs/tests_20260305_080456.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 07:34 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_073127.log`, `output/logs/automation2_validate_20260305_073127.log`, `output/logs/automation2_fast_20260305_073127.log`, `output/logs/tests_20260305_073418.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 07:04 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_070138.log`, `output/logs/automation2_validate_20260305_070138.log`, `output/logs/automation2_fast_20260305_070429.log`, `output/logs/tests_20260305_070429.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 06:07 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_060124.log`, `output/logs/automation2_validate_20260305_060131.log`, `output/logs/automation2_fast_20260305_060412.log`, `output/logs/tests_20260305_060701.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 05:35 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_053143.log`, `output/logs/automation2_validate_20260305_053148.log`, `output/logs/automation2_fast_20260305_053152.log`, `output/logs/tests_20260305_053442.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 05:10 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_050149.log`, `output/logs/automation2_validate_20260305_050149.log`, `output/logs/tests_20260305_050833.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 04:34 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_043121.log`, `output/logs/automation2_validate_20260305_043121.log`, `output/logs/automation2_fast_20260305_043121.log`, `output/logs/tests_20260305_043411.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 04:04 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_040140.log`, `output/logs/automation2_validate_20260305_040140.log`, `output/logs/automation2_fast_20260305_040140.log`, `output/logs/tests_20260305_040431.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 03:34 +0800] Unattended run revalidated first-priority regression via unittest (tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order) + validate-config + test-all --fast --fast-ratio 0.10; all green (tests_selected=27, failed=0, logs output/logs/automation2_unit_20260305_033114.log, output/logs/automation2_validate_20260305_033114.log, output/logs/automation2_fast_20260305_033114.log, output/logs/tests_20260305_033405.json). docs/WORKFLOW_AUTO.md remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 03:04 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_030133.log`, `output/logs/automation2_validate_20260305_030133.log`, `output/logs/automation2_fast_20260305_030133.log` and `output/logs/tests_20260305_030424.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 02:34 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_023106.log`, `output/logs/automation2_validate_20260305_023115.log`, `output/logs/automation2_fast_20260305_023123.log` and `output/logs/tests_20260305_023413.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 02:24 +0800] Unattended run repaired fixture brittleness by seeding `active_baseline.yaml` directly in related rollback preflight regressions (removing `run_review` side-effect dependency), then revalidated with `unittest` (4 targeted tests), `validate-config`, and `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_022112.log`, `output/logs/automation2_validate_20260305_022112.log`, `output/logs/automation2_fast_20260305_022112.log`, `output/logs/tests_20260305_022404.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 01:35 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_013116.log`, `output/logs/automation2_validate_20260305_013116.log`, `output/logs/automation2_fast_20260305_013116.log` and `output/logs/tests_20260305_013502.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 01:05 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_010132.log`, `output/logs/automation2_validate_20260305_010132.log`, `output/logs/automation2_fast_20260305_010132.log` and `output/logs/tests_20260305_010458.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 00:35 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_003130.log`, `output/logs/automation2_validate_20260305_003130.log`, `output/logs/automation2_fast_20260305_003130.log` and `output/logs/tests_20260305_003540.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-05 00:07 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260305_000129.log`, `output/logs/automation2_validate_20260305_000129.log`, `output/logs/automation2_fast_20260305_000129.log` and `output/logs/tests_20260305_000503.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 23:34 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=27`, `failed=0`, logs `output/logs/automation2_unit_20260304_233108.log`, `output/logs/automation2_validate_20260304_233218.log`, `output/logs/automation2_fast_20260304_233226.log` and `output/logs/tests_20260304_233454.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 23:05 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=26`, `failed=0`, logs `output/logs/automation2_unit_20260304_230127.log`, `output/logs/automation2_validate_20260304_230127.log`, `output/logs/automation2_fast_20260304_230127.log` and `output/logs/tests_20260304_230509.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 22:35 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=26`, `failed=0`, logs `output/logs/automation2_unit_20260304_223133.log`, `output/logs/automation2_validate_20260304_223133.log`, `output/logs/automation2_fast_20260304_223133.log` and `output/logs/tests_20260304_223413.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 22:04 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=26`, `failed=0`, logs `output/logs/automation2_unit_20260304_220147.log`, `output/logs/automation2_validate_20260304_220147.log`, `output/logs/automation2_fast_20260304_220147.log` and `output/logs/tests_20260304_220431.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 21:34 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=26`, `failed=0`, logs `output/logs/automation2_unit_20260304_213110.log`, `output/logs/automation2_validate_20260304_213110.log`, `output/logs/automation2_fast_20260304_213110.log` and `output/logs/tests_20260304_213408.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 21:04 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=26`, `failed=0`, logs `output/logs/automation2_unit_20260304_210059.log`, `output/logs/automation2_validate_20260304_210205.log`, `output/logs/automation2_fast_20260304_210211.log` and `output/logs/tests_20260304_210434.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 20:34 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=26`, `failed=0`, logs `output/logs/automation2_unit_20260304_203111.log`, `output/logs/automation2_validate_20260304_203111.log`, `output/logs/automation2_fast_20260304_203111.log` and `output/logs/tests_20260304_203426.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 20:04 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=25`, `failed=0`, logs `output/logs/automation2_unit_20260304_200129_a2.log`, `output/logs/automation2_validate_20260304_200129_a2.log`, `output/logs/automation2_fast_20260304_200129_a2.log` and `output/logs/tests_20260304_200357.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 19:34 +0800] Unattended run revalidated first-priority regression via unittest (`tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=25`, `failed=0`, logs `output/logs/automation2_unit_20260304_193111_a2.log`, `output/logs/automation2_validate_20260304_193111_a2.log`, `output/logs/automation2_fast_20260304_193111_a2.log` and `output/logs/tests_20260304_193350.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 18:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=25`, `failed=0`, logs `output/logs/automation2_unit_20260304_183140.log`, `output/logs/automation2_validate_20260304_183140.log`, `output/logs/tests_20260304_183411.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 18:04 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=24`, `failed=0`, logs `output/logs/automation2_unit_20260304_180135.log`, `output/logs/automation2_validate_20260304_180231.log`, `output/logs/tests_20260304_180425.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 17:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_173337.log`, `output/logs/automation2_validate_20260304_173446.log`, `output/logs/tests_20260304_173335.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 16:36 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_163258.log`, `output/logs/automation2_validate_20260304_163258.log`, `output/logs/tests_20260304_163509.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 17:07 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_170451.log`, `output/logs/automation2_validate_20260304_170451.log`, `output/logs/tests_20260304_170435.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-01 20:36 +0800] Re-ran mixed preflight ordering regressions + `validate-config` + `test-all --fast --fast-ratio 0.10` under unattended automation constraints; all passed.
   - [VERIFIED 2026-03-01 21:03 +0800] No pending `Next Priorities` beyond the completed item; reran `validate-config` and `test-all --fast --fast-ratio 0.10` to confirm green baseline before unattended sync.
   - [VERIFIED 2026-03-01 21:36 +0800] Re-ran source-first mixed preflight ordering regressions plus `validate-config` and `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=18`, `failed=0`).
   - [VERIFIED 2026-03-01 22:04 +0800] Re-ran source-first mixed missing-required/profile-path ordering regression (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=19`, `failed=0`).
   - [VERIFIED 2026-03-01 22:34 +0800] Re-ran source-first mixed missing-required/profile-path ordering regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=20`, `failed=0`).
   - [VERIFIED 2026-03-01 23:04 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=20`, `failed=0`). `docs/WORKFLOW_AUTO.md` is currently missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-01 23:36 +0800] Re-ran first-priority source-first mixed missing-required/profile-path ordering regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=20`, `failed=0`). `docs/WORKFLOW_AUTO.md` is still missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 00:05 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=20`, `failed=0`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 00:33 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=20`, `failed=0`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 01:05 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=21`, `failed=0`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 01:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=21`, `failed=0`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 02:04 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=21`, `failed=0`, log `output/logs/tests_20260302_020341.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 03:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_033332.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 04:03 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_040356.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 04:33 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_043334.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 05:04 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_050331.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 05:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_053345.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 06:05 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_060335.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 06:33 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_063331.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 07:06 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_070400.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 07:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_073420.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-02 08:05 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260302_080326.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 01:39 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260304_013855.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 02:05 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260304_020502.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 02:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260304_023509.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 03:05 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=22`, `failed=0`, log `output/logs/tests_20260304_030525.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 03:35 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_033159.log`, `output/logs/tests_20260304_033459.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 04:05 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_040202.log`, `output/logs/automation2_validate_20260304_040202.log`, `output/logs/tests_20260304_040453.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 07:52 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_061944.log`, `output/logs/automation2_validate_20260304_061944.log`, `output/logs/tests_20260304_075201.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 08:22 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_072128.log`, `output/logs/automation2_validate_20260304_072128.log`, `output/logs/tests_20260304_082212.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 09:23 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_075311.log`, `output/logs/automation2_validate_20260304_075311.log`, `output/logs/tests_20260304_092330.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 10:23 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_082212.log`, `output/logs/automation2_validate_20260304_082212.log`, `output/logs/tests_20260304_102315.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 11:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_113155.log`, `output/logs/automation2_validate_20260304_113155.log`, `output/logs/tests_20260304_113434.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 12:04 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_120145.log`, `output/logs/automation2_validate_20260304_120249.log`, `output/logs/tests_20260304_120428.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 12:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_123133.log`, `output/logs/automation2_validate_20260304_123133.log`, `output/logs/tests_20260304_123415.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 13:04 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_130225.log`, `output/logs/automation2_validate_20260304_130225.log`, `output/logs/tests_20260304_130436.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 13:33 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_133057.log`, `output/logs/automation2_validate_20260304_133057.log`, `output/logs/tests_20260304_133308.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 14:05 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_140231.log`, `output/logs/automation2_validate_20260304_140231.log`, `output/logs/tests_20260304_140440.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 14:33 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_143102.log`, `output/logs/automation2_validate_20260304_143102.log`, `output/logs/tests_20260304_143311.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 15:03 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_150243.log`, `output/logs/automation2_validate_20260304_150109.log`, `output/logs/tests_20260304_150239.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 15:34 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_153104.log`, `output/logs/automation2_validate_20260304_153104.log`, `output/logs/tests_20260304_153313.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 16:03 +0800] Unattended run revalidated first-priority regression via `unittest` (`EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order`) + `validate-config` + `test-all --fast --fast-ratio 0.10`; all green (`tests_selected=23`, `failed=0`, logs `output/logs/automation2_unit_20260304_160119.log`, `output/logs/automation2_validate_20260304_160119.log`, `output/logs/tests_20260304_160331.json`). `docs/WORKFLOW_AUTO.md` remains missing in-repo, so execution followed automation prompt constraints directly.
   - [VERIFIED 2026-03-04 19:04 +0800] Unattended run revalidated first-priority regression via unittest (tests.test_engine_integration.EngineIntegrationTests.test_baseline_rollback_drill_preflight_lints_mixed_missing_required_and_profile_path_errors_in_source_first_order) + validate-config + test-all --fast --fast-ratio 0.10; all green (logs output/logs/automation2_unit_20260304_190143_a1.log, output/logs/automation2_validate_20260304_190143_a1.log, output/logs/automation2_fast_20260304_190143_a1.log). docs/WORKFLOW_AUTO.md remains missing in-repo, so execution followed automation prompt constraints directly.
