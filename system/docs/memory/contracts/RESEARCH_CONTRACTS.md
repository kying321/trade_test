# Research Contracts

## Purpose
定义研究链路中“主线保留、挑战命名、canonical head、阻断与交接”的统一契约，避免 consumer 侧随意解释导致研究结论越权。

## Retained Mainline
- 当前 durable 主线保持：`ETHUSDT / 15m / single-symbol / price_action_breakout_pullback`（简称 breakout-pullback）。
- 主线关注 exit/risk 与 hold-forward arbitration，不把一次性高分结果自动视为晋升依据。
- 主线保留是长期策略约束，不等于冻结挑战实验；挑战必须通过 contract 字段进入。

## Challenge Pair Naming
- `challenge_pair` 命名必须为 `challenger_vs_baseline`。
- 前缀是挑战者，后缀是基线；命名顺序表达角色，不表达优先级高低。
- consumer 不得根据“看起来更优”擅自改写 pair 顺序或重命名。

## Canonical Research Heads
- canonical research head 由 source-owned handoff 的 active 头定义。
- `baseline_retained` 且 canonical handoff active 的语义是：基线继续作为有效裁决头被保留，并可被执行链消费。
- `baseline_retained` 不应被误读为“尚未决策”或“自动等待下一次晋升”。

## Forward Consensus vs Blocker vs Handoff
- `forward_consensus`：窗口一致性层，用于汇总 forward 信号强弱，不是最终 action arbiter。
- blocker：给出当前禁止推进的原因与边界，约束“现在不能做什么”。
- handoff：提供当前可消费的 canonical 控制面，定义执行/研究消费者应如何读取。
- `forward_consensus` 与 handoff/blocker 冲突时，以 handoff/blocker 的显式控制字段优先。
- 若 handoff 与 blocker 对同一 action/head 给出不一致信号，consumer 必须降级为待仲裁/待复核，不得自行并集或自行择优。

## Review-Only and Sidecar Rules
- sidecar positive（包括 break-even、辅助证据、外部情报增强）只能进入 guarded review。
- review-only 结论不能直接 promotion 到 canonical head。
- 任何 promotion 必须回到主链 source artifact，经明确 gate 满足后生效。

## allowed_now / blocked_now / next_research_priority
- `allowed_now`：当前允许推进的最小动作集合，供执行前置与研究调度读取。
- `blocked_now`：当前被阻断动作及原因，必须可回溯到 source 字段。
- `next_research_priority`：下一步研究优先方向，用于排队，不等于自动执行授权。
- 这三项属于外层调度/摘要最小必读控制字段之一组；research consumer 仍需同时读取 `challenge_pair`、active handoff/head 与其它 source-owned 控制字段。
- 外层 summary 不得再造这三项的并行版本，也不得用摘要字段覆盖 source 字段。

## Anti-Patterns
- 把 `forward_consensus` 当作最终执行授权。
- 将 sidecar positive 直接写成 canonical promotion 结论。
- 在 UI/brief 里硬编码当前 challenge_pair 或 baseline 角色。
- 在 handoff 未 active 时，consumer 自行宣布“已切换主线”。
