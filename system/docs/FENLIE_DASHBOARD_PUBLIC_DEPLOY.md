# Fenlie Dashboard Public Deploy

> 更完整的工程操作手册见：
>
> `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_ENGINEERING_RUNBOOK.md`

## 当前状态（2026-03-21，根域已切到 Fenlie 导航模板）

- 本地前端工程：
  - `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web`
- 本地 build 已可用：
  - `npm test`
  - `npm run build`
- 云端静态站已部署到：
  - `43.153.148.242:/home/ubuntu/openclaw-system/dashboard/web/dist`
- 云端本地预览服务：
  - `http://127.0.0.1:5173/`
- Cloudflare Pages 项目已创建并已成功部署：
  - 项目：`fenlie-dashboard-web`
  - 最近验证过的部署地址：`https://6442acc0.fenlie-dashboard-web.pages.dev`
- Cloudflare Pages 自定义域已生效：
  - `https://fenlie.fuuu.fun`
- 根域公开入口现已切换为：
  - `https://fuuu.fun`
- 旧 `OpenClaw Control` 已迁移到：
  - `http://43.153.148.242:3001`
- gateway/API 保持：
  - `http://43.153.148.242:8787`

## 已确认的入口事实

### 云端 5173

以下 systemd 服务当前正常提供 Fenlie 静态站：

- 服务：
  - `pi-dashboard-web.service`
- 配置：
  - `/etc/systemd/system/pi-dashboard-web.service`
- WorkingDirectory：
  - `/home/ubuntu/openclaw-system/dashboard/web`
- ExecStart：
  - `/usr/bin/npx serve -s dist -l 5173`

### fuuu.fun / fenlie.fuuu.fun / 旧 OpenClaw

当前 `fuuu.fun` 已通过 Cloudflare Worker route 覆盖到 Fenlie 公开导航模板，公开访问返回标题为：

- `Fenlie 终端控制台`

同时响应头应包含：

- `x-fenlie-public-entry: root-nav-proxy`

当前 `fenlie.fuuu.fun` 仍绑定到 Cloudflare Pages 项目，并从外部视角验证返回：

- `HTTP/2 200`
- 页面标题：`Fenlie 终端控制台`

旧公开页 `OpenClaw Control` 已从根域移走，当前保留在：

- `http://43.153.148.242:3001`

这说明：

1. `fuuu.fun` 当前不是直接用 DNS 指向云端静态站，而是通过 Cloudflare Worker route 代理到 Fenlie Pages 入口
2. `fenlie.fuuu.fun` 仍然是稳定回退入口
3. 旧 `OpenClaw Control` 没有丢失，只是迁移到独立端口 `3001`
4. `8787` 仍是 gateway/API，不是前端页面；返回 `404 Not Found` 属于当前预期

## 当前前端已补齐的 Cloudflare Pages 配置

前端工程现在已经具备最小 Pages 部署配置：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/wrangler.jsonc`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/_headers`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/_redirects`

可用命令：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run cf:preflight
npm run cf:whoami
npm run cf:deploy
npm run verify:public-surface -- --skip-workspace-build
npm run smoke:workspace-routes
```

本地推荐做法（避免每次手动 export）：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
cat > .env.cloudflare.local <<'EOF'
CLOUDFLARE_API_TOKEN=你的_cloudflare_api_token
EOF
```

说明：

- `.env.cloudflare.local` 已被 repo 根 `.gitignore` 的 `.env.*` 规则覆盖，不会进入版本库
- `cf:preflight` / `cf:whoami` / `cf:deploy` 现在都会优先读取该文件里的 `CLOUDFLARE_API_TOKEN`
- 如果环境变量里已经显式提供 `CLOUDFLARE_API_TOKEN`，则环境变量优先

补充：`cf:whoami` / `cf:deploy` 现已内置 `env -u HTTP_PROXY -u HTTPS_PROXY`，避免本机继承代理变量时 Cloudflare API 超时。

补充：`cf:preflight` 现已作为 Cloudflare 发布前置检查；在像 Codex 这类 non-interactive 会话里，如果缺少 `CLOUDFLARE_API_TOKEN`，会直接失败并提示补 token，而不是先跑完整 `build` 再在 Wrangler 鉴权阶段报错。

补充：预检现在还会识别本机 Wrangler OAuth 缓存是否已过期；如果存在类似：

- `/Users/jokenrobot/Library/Preferences/.wrangler/config/default.toml`

里的过期 OAuth，会直接报：

- `Detected expired Wrangler OAuth cache ... Export CLOUDFLARE_API_TOKEN and retry.`

含义是：本机曾登录过 Wrangler，但当前缓存 access token 已过期；在 non-interactive 会话里不要再指望 OAuth 自动刷新，直接提供 `CLOUDFLARE_API_TOKEN` 更稳定。

说明：`cf:deploy` 现在固定使用 `--branch main --commit-dirty=true`，确保自定义域拿到 production deploy，而不是只生成 preview alias。

补充：`cf:deploy` 现已内置 `strip:internal`，会在 Pages 发布前删除：

- `dist/data/fenlie_dashboard_internal_snapshot.json`

目的：保留同一前端的 public/internal 双路由架构，但避免把本地/私有只读内部快照直接暴露到公开 Pages 分发面。

历史说明：`latest_dashboard_public_deploy_readiness.{json,md}` 属于 2026-03-19 的早期阻塞快照，现仅保留作归档；当前公开部署是否健康，以最新：

- `*_dashboard_public_topology_smoke.json`
- `*_dashboard_public_acceptance.json`

作为只读验收事实。

当前最新公开验收工件：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260325T062149Z_dashboard_public_topology_smoke.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260325T062219Z_dashboard_workspace_routes_browser_smoke.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260325T062139Z_dashboard_internal_alignment_browser_smoke.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260325T062143Z_dashboard_internal_terminal_focus_browser_smoke.json`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260325T062226Z_dashboard_public_acceptance.json`


## 当前入口拓扑

- 根域公开入口：
  - `https://fuuu.fun`
  - 责任层：Cloudflare Worker route → Fenlie Pages
- Pages 回退入口：
  - `https://fenlie.fuuu.fun`
- 旧 OpenClaw 前端：
  - `http://43.153.148.242:3001`
- gateway/API：
  - `http://43.153.148.242:8787`

## 推荐上线方式

### 当前推荐

1. 把 `https://fuuu.fun` 视为 Fenlie 主公开入口
2. 把 `https://fenlie.fuuu.fun` 视为 Pages 回退入口
3. 把 `http://43.153.148.242:3001` 视为旧 OpenClaw 临时回退入口
4. 后续如需正式恢复旧控制页，优先给它单独子域，而不是再抢占根域

## 公开入口烟测

新增只读烟测脚本：

```bash
python3 /Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/run_dashboard_public_topology_smoke.py \
  --workspace /Users/jokenrobot/Downloads/Folders/fenlie
```

期望：

- `https://fuuu.fun` 返回 Fenlie 标题并带 `x-fenlie-public-entry: root-nav-proxy`
- `https://fenlie.fuuu.fun` 返回 Fenlie 标题
- 两侧快照 JSON 的 `frontend_public` 都等于 `https://fuuu.fun`
- `http://43.153.148.242:3001` 返回 `OpenClaw Control`
- `http://43.153.148.242:8787` 返回 `404 Not Found`
- `https://fuuu.fun/#/overview` 与 `https://fenlie.fuuu.fun/#/overview` 都能看到 `研究主线摘要 / 国内商品推理线`
- `https://fuuu.fun/#/workspace/contracts` 与 `https://fenlie.fuuu.fun/#/workspace/contracts` 都能看到 `公开面验收 / 穿透层 1 / 验收总览 / 接口目录 / 源头主线 / 回退链`
- `https://fuuu.fun/#/workspace/contracts?page_section=contracts-source-head-operator_panel`
  与
  `https://fenlie.fuuu.fun/#/workspace/contracts?page_section=contracts-source-head-operator_panel`
  都能看到：
  - `状态`
  - `研究结论`
  - `生成时间`
  - `路径`
- 会落 root/pages 双路由截图证据：
  - `*_dashboard_public_topology_root_overview_browser.png`
  - `*_dashboard_public_topology_pages_overview_browser.png`
  - `*_dashboard_public_topology_root_contracts_browser.png`
  - `*_dashboard_public_topology_pages_contracts_browser.png`
  - `*_dashboard_public_topology_root_contracts_source_head_browser.png`
  - `*_dashboard_public_topology_pages_contracts_source_head_browser.png`

当前最新截图证据前缀：

- `20260325T062149Z_dashboard_public_topology_root_overview_browser.png`
- `20260325T062149Z_dashboard_public_topology_pages_overview_browser.png`
- `20260325T062149Z_dashboard_public_topology_root_contracts_browser.png`
- `20260325T062149Z_dashboard_public_topology_pages_contracts_browser.png`

当前最新 source-head 深链截图证据前缀：

- `20260325T062149Z_dashboard_public_topology_root_contracts_source_head_browser.png`
- `20260325T062149Z_dashboard_public_topology_pages_contracts_source_head_browser.png`

本轮额外确认：

- `workspace_routes_smoke` 现已把这条 orderflow 过滤路由纳入真实浏览器回归断言，并在 smoke payload 中固化：
  - `artifacts_filter_assertion.route=#/workspace/artifacts?group=research_cross_section&search_scope=title&search=orderflow`
  - `artifacts_filter_assertion.source_available=true|false`
  - `artifacts_filter_assertion.active_artifact`
  - `artifacts_filter_assertion.visible_artifacts`
- 当当前 source snapshot 缺失 orderflow canonical 工件时，公开验收接受：
  - `source_available=false`
  - `active_artifact=""`
  - `visible_artifacts=[]`

## 工作区多路由烟测

当前推荐把 workspace 的公开面回归检查固定为：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run smoke:workspace-routes
```

此命令会验证同一只读公开会话内以下五个页面：

- `#/overview`
- `#/workspace/artifacts`
- `#/workspace/backtests`
- `#/workspace/raw`
- `#/workspace/contracts`

并断言：

- workspace 默认焦点跟随 source-owned `workspace_default_focus`
- route change 每次都会触发新的 public snapshot GET
- 不会请求 `/data/fenlie_dashboard_internal_snapshot.json`

兼容说明：

- `npm run smoke:workspace-artifacts` 仍保留，但现在只是 `smoke:workspace-routes` 的兼容别名

## 单一公开面验收入口

如果要在一次命令里同时检查：

- 根域 / Pages / 旧 OpenClaw / gateway 的入口拓扑
- workspace 五页面公开面刷新与 internal 剥离边界

使用：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run verify:public-surface -- --skip-workspace-build
```

等价的 npm Python 启动链为：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
node ./scripts/run-python.mjs ../../scripts/run_dashboard_public_acceptance.py --workspace ../../.. --strict-topology-tls --skip-workspace-build
```

解释器约束：

- `verify:public-surface` 不再直接假设裸 `python3`
- dashboard npm 下的 Python 入口统一走 `node ./scripts/run-python.mjs`
- 启动器优先选择 `CONDA_PREFIX/bin/python3`
- 若 `CONDA_PREFIX` 不可用，才 fallback 到 PATH 里的 `python3`
- 如果你要在 `system/dashboard/web` 目录手工复现 npm 子进程行为，优先使用这条 launcher 命令，避免本机 shell 与 npm PATH/证书链不一致

输出会聚合两条只读验收链：

1. `run_dashboard_public_topology_smoke.py`
2. `run_dashboard_workspace_artifacts_smoke.py`

并生成统一 review 工件：

- `*_dashboard_public_acceptance.json`

最新已验证通过的聚合工件：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260325T062226Z_dashboard_public_acceptance.json`

当前本地还新增了单一上架前验证入口：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npm run verify:release-checklist
```

该入口会执行：

- `npm run build`
- `npm test`
- `npx tsc --noEmit`
- `npm run test:dashboard-python-contracts`
- `npm run smoke:workspace-routes -- --skip-build`
- `npm run smoke:alignment-internal -- --skip-build`
- `npm run smoke:terminal-internal-focus -- --skip-build`
- `npm run smoke:graph-home -- --skip-build`
- `npm run smoke:graph-home-narrow -- --skip-build`
- `npm run smoke:graph-home-pipeline -- --skip-build`
- `npm run verify:public-surface -- --skip-workspace-build`

当前本地已验证结果（graph-home 三条 smoke 已纳入 `verify:release-checklist` 候选链并单独通过）：

- `25 files / 148 tests passed`
- `dashboard python contracts => 79 passed`
- `workspace routes smoke => ok = true`
- `internal alignment smoke => ok = true`
- `terminal/internal focus browser smoke => ok = true`
- `graph-home browser smoke => ok = true`
- `graph-home narrow browser smoke => ok = true`
- `graph-home pipeline browser smoke => ok = true`
- `public acceptance => ok = true`

用途：减少上架前人工串命令导致的漏跑/顺序漂移。

当前聚合入口还会在子命令异常时保留：

- 子命令 `returncode`
- 原始 `stdout/stderr`
- `payload_present` 与输出字节数

目的：避免再次出现“单独 smoke 通过，但聚合报告吞掉失败证据链”的冷启动排障盲区。

## 回滚点

### 云端静态站回滚

云端旧版 dist 备份在：

- `/home/ubuntu/openclaw-system/dashboard/web/dist.bak_20260319T082355Z`

### Cloudflare Pages 回滚

删除 Pages 项目即可：

```bash
cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web
npx wrangler pages project delete fenlie-dashboard-web
```

### 本地工程回滚

删除以下文件即可撤销 Cloudflare Pages 配置：

- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/wrangler.jsonc`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/_headers`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/public/_redirects`

### 根域 Worker route 回滚

移除或回滚以下 Worker 即可把根域从 Fenlie 导航模板切走：

- Worker：`fenlie-root-nav-proxy`
- 路由：`fuuu.fun/*`
