# Fenlie Shortlist

## Identity
Fenlie 是以 ETHUSDT / 15m / single-symbol breakout-pullback 为主线的策略研究与操作协同平台，长期保持源头 artifact（exit/risk、hold-forward、contract）控制的权威链。

## Authority Order
1. source artifacts（dataset、builder outputs、handoff JSON）
2. compact handoff/context（NEXT_WINDOW_CONTEXT_LATEST、contracts 中的 control fields）
3. UI/operator summaries
4. 本地记忆与 chat 口述说明

## Current Main Priorities
- 保持 `ETHUSDT / 15m / single-symbol / breakout-pullback` 研究主线，强调 price-action breakout-pullback 的 exit/risk 以及 hold-forward arbitration。
- Exit/risk chain 以 challenge_pair 为中心，输出 allowed_now/blocked_now/next_research_priority，不在 consumer 侧硬编码特定 hold_bars 组合。
- Hold-forward arbitration 继续围绕 forward_capacity、overlap_sidecar、hold_selection_handoff、window_consensus 的 source-owned gate，确保 baseline-anchor 与 forward tail 信息一致。

## Low-Value
- geometry_delta 截面推理（跨 symbol / cross-section 的 breakout 变体）
- VPVR proxy 之类的密度推导

## Deprioritized Paths
- reclaim structure filter 的后验节点
- daily_stop_r 作为主要止损路径
这些方向目前不再作为主线研究。

## Cold Start
1. `AGENTS.md`
2. `FENLIE_CODEX_MEMORY.md`
3. `memory/MEMORY_INDEX.md`
4. `memory/SHORTLIST.md`
5. 具体任务相关的 contract 文件

## “继续 / 下一步” Semantics
“继续”只在先决条件（state/queue/gate/source ownership）发生变化后才触发；优先处理能改变任何 source-owned gate、状态、手工 queue 或 user-visible 能力的未解决高危问题。连续两轮都没改变这些内容就需要停止并总结。

## Never Treat As Source-of-Truth
记忆树是解释型手册——它提醒你如何恢复与判断，但永远不能代替 source artifacts、handoff 或 live contract。本文件中提及的优先级与规则必须回到对应 artifact 验证后再采取行动。
