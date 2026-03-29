# Domestic Futures Execution Bridge Contract

## Purpose
定义国内期货/商品执行桥的 canonical source contract，明确“策略路径有效”和“执行桥可用”是两个独立维度，避免 consumer 侧把 `manual_structure_review_now`、`paper_only`、`review_manual_stop_entry` 等消费层标签误当成 source-of-truth。

## Scope
- 适用于 `asset_class = future` 且目标市场为国内期货/国内商品的信号、queue、ticket、review、paper、canary、execution 链路。
- 本 contract 只裁决桥接能力、允许动作、阻断原因、晋级条件与执行真相来源，不替代策略质量、研究主线或价格结构判断。
- live 资金、下单、路由、fund scheduling 的最终纪律仍以 `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/LIVE_BOUNDARY.md` 为准。

## Canonical Principles
- source-owned artifact 必须显式给出桥接能力字段；consumer 只能读取、翻译、降级，不能自造桥接结论。
- `route valid`、`signal ready`、`brooks structure ready` 不等于 `execution bridge ready`。
- 若结构/策略路径有效但执行桥缺失，系统必须停在 `manual_only` 或 `paper_only`，而不是伪装成可自动执行。
- consumer 层标签（如 `manual_structure_review_now`、`review_manual_stop_entry`、`paper_only`）必须从 source bridge stage 单向派生，不能反向充当 source。

## Canonical Capability Fields
任何国内期货执行桥 source artifact 至少应提供以下字段：

- `symbol`: 合约标识，如 `SC2603`
- `asset_class`: 当前仍使用现有枚举；国内期货保持 `future`
- `venue`: 执行 venue；未确定时必须留空或显式标注未知
- `account_scope`: 账户作用域，如 paper / manual / guarded / live
- `adapter_kind`: 执行适配器类型，如 `none` / `manual` / `paper_engine` / `broker_api`
- `bridge_stage`: canonical 桥接阶段
- `allowed_actions`: 当前允许动作列表
- `blocked_actions`: 当前阻断动作列表
- `blocker_code`: 机器可读阻断码
- `blocker_detail`: 审计可读阻断详情
- `promotion_gates`: 晋级必须满足的 gates
- `demotion_triggers`: 降级触发条件
- `execution_truth_source`: 执行真相来源，如 `manual_confirmation` / `paper_ledger` / `broker_ack_reconcile`
- `owner_artifact`: 当前桥接结论所属 source artifact
- `as_of`: 明确时间戳
- `evidence_refs`: 证据引用列表

若历史 artifact 已存在 `execution_mode`、`route_bridge_status`、`plan_status` 等字段，它们只能作为 consumer-facing compatibility 字段，必须能一对一回指到 `bridge_stage` 与上述 canonical capability fields。

## Bridge Stages

### `research_only`
- 含义：仅允许研究、情景树、传导链、route observation。
- 允许：`research_refresh`、`watch_only`、`source_rebuild`
- 禁止：`paper_apply`、`auto_route`、`auto_send`、`live_promotion`
- 真相来源：无执行真相，仅研究 artifact

### `paper_only`
- 含义：允许进入 paper ticket / paper queue / retro / ledger 验证，但没有 live 权限。
- 允许：`paper_apply`、`paper_review`、`retro_reconcile`
- 禁止：`live_send`、`live_promotion`
- 真相来源：`paper_ledger`、`executed_plans`、paper execution evidence
- 说明：现有 `commodity_paper_execution_*` 家族中的 `execution_mode = paper_only` 应视为该阶段的消费层映射，而不是独立 authority。

### `manual_only`
- 含义：结构/策略路径有效，但系统内没有自动执行桥；只能停留在人工结构复核或人工下单准备。
- 允许：`review_manual_stop_entry`、`queue_review`、`source_refresh`
- 禁止：`auto_route`、`auto_send`、`live_promotion`
- 真相来源：`manual_confirmation`
- 默认阻断码：`no_automated_execution_bridge`

### `guarded_canary`
- 含义：桥接链路已明确，但只能在受限规模、显式 gate、幂等、mutex、panic guard 下做受控探针。
- 允许：`guarded_probe`、`guarded_canary_send`
- 禁止：策略层面的自动大规模 promotion
- 真相来源：`broker_ack_reconcile`
- 说明：guarded canary 证明 execution plumbing，可验证 transport/reconcile，不等于策略 ready。

### `executable`
- 含义：venue/account/adapter/guardian/executor/reconcile 全链路已显式接通，可作为真正 executable source lane。
- 允许：source contract 明确列出的执行动作
- 真相来源：`broker_ack_reconcile`
- 说明：该阶段仍受 live boundary、risk guard、run-halfhour-pulse mutex 约束

## Promotion and Demotion Rules
- `research_only -> paper_only`：
  - 必须有 paper execution owner artifact
  - 必须有 paper execution evidence / ledger truth source
  - 不得依赖 proxy-only price reference 伪装成可执行价格
- `research_only -> manual_only`：
  - 路径/结构判断已有效
  - 但 venue/account/adapter 未形成可自动执行桥
- `manual_only -> paper_only`：
  - 必须把 paper bridge、paper ledger、queue apply 证据补齐
  - 不能仅凭“人工本来也能下单”跳过 paper capability contract
- `paper_only -> guarded_canary`：
  - 必须明确 venue、account_scope、adapter_kind
  - 必须具备 execution truth / reconcile / ack 回流
  - 必须满足 live boundary 的 guardian/executor/mutex/idempotency 约束
- `guarded_canary -> executable`：
  - 只能由 source contract 明确晋级
  - UI、brief、review packet、browser intel 不得直接触发
- 任意阶段降级：
  - 一旦 `execution_truth_source` 缺失、adapter 失效、账户作用域变化、source-of-truth 歧义出现，必须回退到较低阶段并写明 `blocker_code`

## Current Verified Canonical Example

### `SC2603 / brooks_structure`
当前已验证的 source-owned 消费结果应解释为：

- `bridge_stage = manual_only`
- `allowed_actions = [review_manual_stop_entry, queue_review, source_refresh]`
- `blocked_actions = [auto_route, auto_send, live_promotion]`
- `blocker_code = no_automated_execution_bridge`
- `blocker_detail = Structure route is valid, but this asset class has no automated execution bridge in-system.`
- `execution_truth_source = manual_confirmation`

当前 consumer-facing 兼容映射：
- `route_bridge_status = manual_structure_route`
- `plan_status = manual_structure_review_now`
- `execution_action = review_manual_stop_entry`

这些字段说明“结构 route 有效但桥接不可自动化”，**不说明**“已具备自动执行能力”。

### Domestic commodity paper families
现有 `commodity_paper_execution_*` 家族若输出 `execution_mode = paper_only`，其含义应解释为：

- 已进入 `paper_only`
- 仍然不是 `guarded_canary` 或 `executable`
- 不能把 paper evidence 直接翻译为 live promotion authority

## Consumer Obligations
- operator brief、dashboard、workspace route smoke、review queue、panel 都必须读取 source-owned bridge stage，不得拼接新 blocker 语义。
- 若同时出现 `paper_only` 与 `manual_only` 线索，consumer 必须回源到 `owner_artifact` 与 `as_of` 仲裁，不能自行择优。
- 若没有 capability artifact，只能显式显示“bridge capability unknown / unresolved”，不能自动推断为可执行。

## Anti-Patterns
- 把 `manual_structure_review_now` 当成 source-of-truth，而不回源 bridge contract。
- 把“人工可做”误读为“系统已可自动做”。
- 把 `paper_only` 当成 live ready。
- 用 dashboard/UI 文案替代 `blocker_code` / `blocker_detail` 的 canonical source。
- 在没有 venue/account/adapter truth 的情况下宣布 `guarded_canary` 或 `executable`。

## Related Artifacts
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/SOURCE_AUTHORITY.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/docs/memory/contracts/LIVE_BOUNDARY.md`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_domestic_futures_execution_bridge_capability.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_brooks_price_action_execution_plan.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_brooks_structure_review_queue.py`
- `/Users/jokenrobot/Downloads/Folders/fenlie/system/scripts/build_commodity_paper_execution_gap_report.py`
