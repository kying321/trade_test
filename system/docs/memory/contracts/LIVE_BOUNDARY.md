# Live Boundary Contract

## Purpose
定义 live authority 的最小可执行边界，确保研究/反馈/展示层不会越权到执行层，同时保留可审计的升级与回退路径。

## Control Chain
默认控制链：
`intel/research -> ticketization -> risk_guard -> execution_contract -> broker`

硬约束（live discipline）：
- 触及 live capital、order routing、execution queue、fund scheduling 的流程必须持有 `run-halfhour-pulse` 文件互斥。
- live deadline/retry/backoff 使用 monotonic clock，不以 wall-clock 直接判超时。
- retryable 外部操作必须携带 idempotency key。
- outbound request 必须限流并设置 `timeout <= 5000ms`。

## Guardian vs Executor
- `guardian = veto-only`：只负责拒绝、降级、阻断、触发 no-trade/review，不负责授权提升。
- `executor = transport/execution only`：只负责把已授权指令可靠送达并回传执行事实，不负责策略裁决或权限提升。
- guardian 与 executor 都不能绕过 source-owned gate 直接“放行”。

## Shadow / Probe / Canary / Promotion
- Shadow：只读/回放/仿真，不产生 live 执行权。
- Guarded Probe：受限可执行探针，用于验证链路与约束，不代表策略晋升。
- Canary：小范围受控执行，目标是验证执行稳定性与风控一致性。
- Promotion：仅在 source contract 明确满足升级条件时进行，不能由展示层或反馈层直接触发。

## Reconcile and Execution Truth
- execution truth 以 `ack / journal / reconcile / broker fill` 为准。
- history 盈利、UI 指标、brief 叙述都不是 executable truth。
- 若执行回执与展示层冲突，必须以执行侧日志和对账结果裁决。

## Feedback Limits
- feedback、review packet、browser intel、backup intel 只能：
  - 降低置信度
  - 触发 veto/no-trade
  - 触发 review/rebuild
- 这些输入不能直接提升 live authority，不能直接跳过 control chain 进入 promotion。

## Failure Severity
- 仅“执行路径失败”或“source-of-truth 歧义”可触发 trading halt / panic_close_all。
- 进程级故障可触发 recycle/kill，但仍需保持执行事实可追溯。
- 通知、dashboard、metrics、Telegram 409 等非执行故障默认 degrade/report，除非已造成执行歧义。

## Anti-Patterns
- 用 review 结论直接替代 guardian gate。
- 把 browser 观察或 UI 偏好当作 live 放行依据。
- 在 executor 中实现策略裁决或权限判断。
- 未持有 run-halfhour-pulse 就进行 live 资金/下单相关操作。
