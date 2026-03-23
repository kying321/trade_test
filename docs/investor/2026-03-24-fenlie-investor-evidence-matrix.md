# Fenlie 投资人证据矩阵

> 用途：路演前内部校对 / 投资人问答 / 尽调准备  
> 原则：所有对外表述都必须能映射到内部证据；不能映射的内容，不进入正式路演口径。

---

## 使用方式

每一条对外说法都按 4 列管理：

1. **对外说法**
2. **成熟度标签**
3. **内部证据**
4. **推荐讲法 / 禁止讲法**

成熟度统一使用：

- `已验证`
- `验证中`
- `规划中`

---

## A. 公司定位与产品定义

| 对外说法 | 成熟度 | 内部证据 | 推荐讲法 / 禁止讲法 |
|---|---|---|---|
| Fenlie 是反脆弱量化研究、风控编排与执行控制平台 | 已验证 | `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/ARCHITECTURE_REVIEW.md`；`/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/PROGRESS.md` | 推荐讲“平台”“控制链”“研究到执行的一体化”；不要讲成“单个神奇策略产品” |
| Fenlie 不是普通 dashboard，而是运营控制面 | 已验证 | `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_ENGINEERING_RUNBOOK.md`；`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/20260322T140501Z_dashboard_public_acceptance.json` | 推荐讲“可观测、可验收、可回归”；不要讲成“只是前端改版” |
| Fenlie 以 source-owned artifact 为核心纪律 | 已验证 | `/Users/jokenrobot/Downloads/Folders/fenlie/AGENTS.md`；`/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md`；相关 read-model / snapshot 架构说明 | 推荐讲“source-of-truth discipline”；不要讲“页面里自动推理出所有业务状态” |

---

## B. 研究能力

| 对外说法 | 成熟度 | 内部证据 | 推荐讲法 / 禁止讲法 |
|---|---|---|---|
| 当前已有明确的加密超短线研究主线 | 已验证 | `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/PROGRESS.md`；`/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_CODEX_MEMORY.md` | 推荐讲 `ETHUSDT / 15m / breakout-pullback / exit-risk mainline`；不要把所有策略都说成成熟 |
| 已能区分已证伪方向、watch-only sidecar 和 canonical anchor | 已验证 | `latest_price_action_breakout_pullback_*` 系列工件；`PROGRESS.md` | 推荐讲“研究纪律与工件化”；不要讲“AI 自动找到最优策略” |
| 订单流研究增强已进入研究链 | 已验证 | `/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_oi_sidecar.json`；`/Users/jokenrobot/Downloads/Folders/fenlie/system/output/review/latest_intraday_orderflow_intent_triad_sidecar.json` | 推荐讲“研究增强”“sidecar evidence”；不要讲“订单流已成为当前主执行引擎” |
| OI 已接入并作为 triad 输入 | 已验证 | `latest_intraday_orderflow_oi_sidecar.json` 中 `research_decision=use_oi_sidecar_as_triad_input` | 推荐讲“OI 已入链”；不要讲“订单流已 fully trusted” |
| triad 目前仍是 context-only | 已验证 | `latest_intraday_orderflow_intent_triad_sidecar.json` 中 `research_decision=keep_triad_context_only_until_oi_integration_and_trust_upgrade` | 推荐主动降温；禁止讲“triad 已是 execution-grade signal” |

---

## C. 控制台与工程成熟度

| 对外说法 | 成熟度 | 内部证据 | 推荐讲法 / 禁止讲法 |
|---|---|---|---|
| Fenlie 已具备公开部署与验收链 | 已验证 | `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/PROGRESS.md`；`/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/FENLIE_DASHBOARD_PUBLIC_DEPLOY.md` | 推荐讲 Pages 部署、public acceptance、拓扑 smoke；不要讲“已经服务大量公网用户” |
| 发布前存在固定 release-checklist | 已验证 | `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/PROGRESS.md`；`/Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web/package.json` | 推荐讲 build/test/tsc/python contracts/workspace smoke/public acceptance；不要模糊成“我们一般会测一下” |
| 控制台已形成 public/internal 分层 | 已验证 | `fenlie_dashboard_snapshot.json` 与 `fenlie_dashboard_internal_snapshot.json`；internal alignment 相关 smoke 工件 | 推荐讲“可区分公开面与内部面”；不要讲“所有信息都在公开面可见” |
| 前端全量门禁已跑通过 | 已验证 | `PROGRESS.md` 中 `24 files / 125 tests passed`；`53 passed` Python contracts | 推荐讲“有固定门禁链”；不要讲“零问题”或“永不出错” |

---

## D. 执行桥与接管能力

| 对外说法 | 成熟度 | 内部证据 | 推荐讲法 / 禁止讲法 |
|---|---|---|---|
| Fenlie 已建立云端执行桥 runbook | 验证中 | `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/OPENCLAW_CLOUD_BRIDGE.md` | 推荐讲“受控桥接、probe、canary、ready-check”；不要讲“已全面商业化执行网络” |
| 执行桥具备白名单、限流、幂等、超时治理 | 验证中 | `OPENCLAW_CLOUD_BRIDGE.md`；桥接脚本与 review 产物 | 推荐讲“控制能力已存在”；不要讲“所有市场都已大规模稳定运行” |
| 研究到执行接管已有工程路径 | 验证中 | `live-takeover-*` runbook、bridge 脚本、相关工件路径 | 推荐讲“工程可行 + 产品化推进中”；不要讲“已成为主要收入来源” |

---

## E. 商业模式与市场切入

| 对外说法 | 成熟度 | 内部证据 | 推荐讲法 / 禁止讲法 |
|---|---|---|---|
| 先从 Crypto 量化团队切入最合理 | 规划中但有证据支撑 | 当前研究主线、订单流增强、执行桥路径均偏 crypto 场景；相关文档同上 | 推荐讲“与现阶段产品成熟度最匹配”；不要讲“这是唯一市场” |
| 早期商业化以私有部署、研究赋能、控制台落地为主 | 规划中 | 当前最成熟能力集中在研究链、控制台、验收链 | 推荐讲“高价值交付型收入”；不要讲“已经有标准化大规模订阅收入” |
| 后续可扩展到研究型资管技术组与多资产控制面 | 规划中 | 八层架构、控制面、source-owned discipline 的可扩展性 | 推荐讲“扩张路径”；不要讲“已经完成跨市场规模复制” |

---

## F. 明确禁止的外部说法

以下内容在未补充独立证据前，**禁止进入正式路演**：

1. 已有明确营收、ARR、MRR、AUM
2. 已有付费机构客户或长期合同
3. 已形成稳定可规模化 execution SaaS
4. 任意策略存在可承诺收益
5. triad/CVD/OI 已成为主执行信号
6. 已完成合规牌照或机构监管接入

---

## G. 推荐路演口径模板

### 1. 关于产品

推荐：

> “Fenlie 的核心不是单一策略，而是把研究、风控、控制台和执行接管装进同一条可验证链。”  

避免：

> “我们已经有一套成熟的全自动赚钱系统。”

### 2. 关于订单流与策略

推荐：

> “我们已经把 OI 等订单流增强证据接入研究链，但当前仍把 triad 定位为 context-only sidecar，不直接晋升为执行级信号。”  

避免：

> “我们已经找到了稳定有效的订单流必胜策略。”

### 3. 关于 execution

推荐：

> “执行桥的关键治理能力已经建立，当前重点是把它从工程路径继续产品化、pilot 化。”  

避免：

> “我们已经拥有成熟的大规模自动执行网络。”

### 4. 关于商业化

推荐：

> “早期收入更适合来自私有部署、控制台落地、研究赋能与 execution pilot。”  

避免：

> “我们已经是标准化 SaaS 高速增长阶段。”

---

## H. 路演前最后检查清单

- [ ] 每个核心说法都有对应证据路径
- [ ] 每个产品能力都有成熟度标签
- [ ] 没有出现收益承诺
- [ ] 没有把验证中能力讲成已规模化
- [ ] 没有虚构客户、收入、估值
- [ ] Q&A 中执行、合规、商业化三类问题都有降温答案

