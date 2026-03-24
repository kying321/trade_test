# Fenlie Durable Rules

## Operating Style
- 默认语言：中文，语气直接、简洁、技术性强、便于审计。靠 facts + contracts 表述，避免无端推测。
- 任何 substantial work 前必须声明 mode（architecture_review/source_first_implementation/API 触发等）与首步。
- 只选用最窄 change class，并注明 DOC_ONLY/RESEARCH_ONLY/SIM_ONLY/LIVE_GUARD_ONLY/LIVE_EXECUTION_PATH。

## Source-Owned State Rule
- 凡是能在 source artifact（dataset、handoff JSON、control field）中写明的 lane、gate、anchor、queue ownership、review state，不得在 consumer 侧重推。
- control fields（allowed_now、blocked_now、next_research_priority、source_head_status）是权威，其他文档引用前必须回源验证。

## Live Boundary Rule
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/LIVE_BOUNDARY.md` 是 live-boundary 的 canonical contract；本章只保留 cold-start 所需的 durable 摘要，不重复维护执行细节长清单。
- live capital、order routing、execution queue、fund scheduling 相关代码必须使用 `run-halfhour-pulse` mutex。
- live-path request / retry / deadline 语义遵循 canonical contract：使用 monotonic clock、timeout ≤ 5000ms、retryable 外部操作附带 idempotency key，并保持 rate limiting。
- guardian / executor 的 authority split 必须保持：feedback、review、browser intel 只能 veto / degrade / trigger review，不能直接提升 live 权限。

## Delegation Rule
- 仅当用户明确允许 subagent/parallel work 且任务不是 critical path 时才委派。
- 子代理默认只读 `memory/SHORTLIST.md` 与单个 contract，避免 context 泄露。
- 重要决策（source-of-truth arbitration、live-path judgment、最终 UI 结论）必须留在主代理手上。

## Validation Rule
- 任何变更前先验证 source artifacts。
- 推荐命令集包含：`cd /Users/jokenrobot/Downloads/Folders/fenlie/system && pytest -q tests/test_build_dashboard_frontend_snapshot_script.py`、`cd /Users/jokenrobot/Downloads/Folders/fenlie/system && ./scripts/lie-local validate-config`、`cd /Users/jokenrobot/Downloads/Folders/fenlie/system/dashboard/web && npm run verify:public-surface` 等。
- 记录验证结果在最终回复里，确保 no-touch live path。

## Stop Rule
- 继续/下一步只能在高危 unresolved issue（state/queue/gate/source ownership 发生可验证变更）存在时才触发。
- 若连续两轮未改变上述元素，就应停止当前 loop 并报告，避免伪 motion。

## Change Class Rule
- 选择最窄变更类别：DOC_ONLY、RESEARCH_ONLY、SIM_ONLY、LIVE_GUARD_ONLY、LIVE_EXECUTION_PATH。
- 如果任务可能影响面向 execution 的路径，应在票据/手册中明确说明，并根据 change class 选择对应的 review。

## Time
- 研究/backtest 不得使用 `datetime.now()` 或 forward-looking index；只能用固定 event timestamp。
- 所有 live deadline/backoff 都依赖 monotonic clock，具体执行语义以 `memory/contracts/LIVE_BOUNDARY.md` 为准。

## Retry
- live/outbound retry 只保留一条 durable 要求：必须带 backoff，且具体重试语义以 `memory/contracts/LIVE_BOUNDARY.md` 为准。

## Idempotency
- 任何 retryable 外部请求均需携带 idempotency key；更细的执行侧幂等语义仅在 canonical contract 维护。

## Timeout Constraints
- 对外请求 timeout 需保持在 ≤5000ms；rate limit / timeout 失败处理以 `memory/contracts/LIVE_BOUNDARY.md` 为准。

## Memory Update Rule
- 仅当 durable rule、delegation、key path、validation command 等长期结构改变时更新本文件。
- 临时 blocker、ready-check、某次回测结果、live queue/position 状态不得写入本章；这类内容属于 source artifacts 或 handoff brief。
- 旧的大文件（如 `FENLIE_CODEX_MEMORY.md`）现仅保留兼容启动与指向新 tree 的 summary，不再承担 deep contract 内容。
