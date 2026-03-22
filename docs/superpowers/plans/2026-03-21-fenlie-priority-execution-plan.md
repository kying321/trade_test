# Fenlie Priority Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收敛当前 Fenlie 的最高风险未完成项，按发布阻塞优先顺序逐个执行，不扩大到 live path。

**Architecture:** 先处理会阻塞前端发布与后续重构验证的回归问题，再统一文档/脚本入口契约，最后才回到研究主线与结构债。所有步骤坚持 source-first、只做最小变更、每步都要求独立验证。

**Tech Stack:** Python, React 19, TypeScript, Vite, Vitest, Pytest, Cloudflare Pages

---

### Task 1: 收口前端回归失败（最高优先级）

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/active-copy-contract.test.js`
- Possibly modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/contracts/dashboard-term-contract.json`
- Possibly modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.tsx`
- Test: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/App.test.tsx`
- Test: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/src/active-copy-contract.test.js`

- [x] Step 1: 复现当前失败测试并记录失败原因
- [x] Step 2: 定位是“测试过时”还是“实现越界”
- [x] Step 3: 先改测试或契约，再做最小实现修正
- [x] Step 4: 跑目标测试，确保 2 个失败收敛
- [x] Step 5: 跑 `npm test` 与 `npx tsc --noEmit` 做前端回归

### Task 2: 清理前端发布契约与工作树噪音

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/package.json`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/_headers`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/_redirects`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/root_proxy/src/index.js`
- Review: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md`
- Review: `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_dashboard_public_deploy_readiness.md`

- [x] Step 1: 校对 deploy doc / readiness artifact / root_proxy 当前事实是否一致
- [x] Step 2: 修正明显漂移项，不改变 live/API 路径
- [x] Step 3: 跑 `verify:public-surface` / topology smoke 的最小验证

### Task 3: 统一 skill / memory / script 入口契约

**Files:**
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`
- Modify: `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_SKILL_USAGE_PLAYBOOK.md`
- Review: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_daily_ops_skill_checklist.py`
- Review: `/Users/jokenrobot/.codex/skills/fenlie-source-ownership-review/scripts/run_source_ownership_review.py`

- [x] Step 1: 识别失效入口名
- [x] Step 2: 文档改为指向真实入口
- [x] Step 3: 跑最小 help/调用验证

### Task 4: 继续 ETHUSDT 15m exit/risk 主线

**Files:**
- Review: `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260319T105300Z_price_action_breakout_pullback_exit_risk_sim_only.json`
- Modify/Run: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_price_action_breakout_pullback_exit_hold_forward_compare_sim_only.py`
- Output: `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/*hold_forward_compare*.json`

- [x] Step 1: 校对现有 `max_hold_bars=8 vs 16` 主线证据
- [x] Step 2: 跑更长窗 forward compare
- [x] Step 3: 只输出 SIM_ONLY 结论，不碰 live 链

### Task 5: 结构债收口（低优先级）

**Files:**
- Review: `/Users/jokenrobot/Downloads/Folders/fenlie/system/src/lie_engine/engine.py`
- Review: `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_dashboard_frontend_snapshot.py`

- [x] Step 1: 标记 `engine.py` 和 snapshot builder 的切分边界
- [x] Step 2: 只写设计或切分计划，不在当前执行链直接大重构
